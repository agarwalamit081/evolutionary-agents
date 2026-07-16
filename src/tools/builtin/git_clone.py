"""git-clone builtin + pgvector code index (Phase 5 I2).

Autonomy gap: the agent could read a single file/page (file_reader/web_scraper)
but could not ingest an EXTERNAL repository's source into its semantic memory.
``git_clone`` clones a public git repo into a confined workspace subdir, walks
the code files, chunks each per function/class (lightweight AST chunker), embeds
every chunk via the existing ``EmbeddingGenerator``, and stores the chunks as
``ColdMemory`` episodes (``episode_type="code"``) so a later ``code_search``
recalls them by semantic similarity — the agent's "codebase memory" for repos it
has read. Distinct from ``corpus`` (indexed web PAGES) and ``memory_search``
(tier memory): this indexes source CODE by symbol.

Security (defense-in-depth, mirroring web_scraper/_paths):
  * SSRF — ``assert_public_host`` rejects private/loopback/link-local hosts (the
    same guard web_scraper/http_request use); blocking DNS, so wrapped in
    ``asyncio.to_thread``.
  * Path confinement — the clone lands under ``<workspace_root>/clones/<slug>``
    and the resolved dir is verified to stay inside the workspace; the slug is
    regex-sanitized so a URL can't name a traversal target. Symlinks inside the
    repo are skipped at walk time so a malicious link can't read outside dest.
  * Resource caps — ``max_files``/``max_file_bytes``/``max_total_bytes``/
    ``max_chunks`` bound a maliciously-large or pathologically-structured repo.
    The walk + index stop at the first cap hit.

Default-off (``GIT_CLONE_ENABLED``): when disabled the handler is a no-op that
tells the caller. The index write is non-fatal (CostTracker-resilience pattern):
a DB/embedding failure logs + the status reports a partial index rather than
raising — indexing is an aid, never a run-killer.
"""

from __future__ import annotations

import ast
import asyncio
import os
import re
import shutil
from pathlib import Path
from typing import Any

from loguru import logger

from src.tools.builtin._net_safety import assert_public_host

#: Code-file extensions the v1 AST chunker understands. Python only — the
#: ``_chunk_python`` AST split is language-specific; other languages fall through
#: to a whole-file chunk. Kept narrow on purpose (extend by adding a chunker, not
#: by globbing everything into one embedding).
_CODE_SUFFIXES: frozenset[str] = frozenset({".py"})

#: Directory names never descended into during the walk (VCS, caches, vendored
#: deps, build output). Keeps the index off generated/vendored code that would
#: drown the repo's own symbols.
_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".tox",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "dist",
        "build",
        "site-packages",
    }
)

#: A safe path-component charset for the clone slug. Anything else collapses to
#: ``-`` so a URL tail can never smuggle a separator/``..`` into the dest path.
_REPO_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _git_clone_settings() -> Any:
    """Call-time accessor for the ``GitCloneSettings`` group.

    Lazy import (not a module-top binding) so unit tests can patch
    ``src.config.settings.get_settings`` and have it take effect here — the same
    idiom ``metrics``/``schedule_task`` use.
    """
    # Lazy import so tests can patch src.config.settings.get_settings.
    from src.config.settings import get_settings

    return get_settings().git_clone


def _clone_dest(url: str) -> Path:
    """Confined clone destination: ``<workspace_root>/clones/<slug>``.

    The slug is the URL's last path segment (``.git`` stripped), sanitized to a
    single safe path component. The resolved dir is verified to stay inside the
    workspace ``clones/`` dir so a malformed URL (or a symlink already at that
    path) can't escape the workspace. Raises ``ValueError`` on escape; callers
    translate that to their ``ERROR:`` surface string.
    """
    from src.tools._paths import workspace_root

    base = (workspace_root() / "clones").resolve()
    tail = url.rstrip("/").rsplit("/", 1)[-1] or "repo"
    tail = tail.removesuffix(".git") or "repo"
    slug = _REPO_SLUG_RE.sub("-", tail).strip("-.") or "repo"
    dest = (base / slug).resolve()
    # Defense-in-depth: a pre-existing symlink at base/slug pointing outside, or
    # any resolution trick, leaves dest outside base ⇒ refuse before touching it.
    if not dest.is_relative_to(base):
        raise ValueError(f"clone destination escapes workspace: {dest}")
    return dest


async def _run_clone(
    url: str, dest: Path, ref: str, timeout_s: int
) -> tuple[bool, str]:
    """Clone ``url`` into ``dest`` via a ``git`` subprocess (``--depth 1``).

    Returns ``(ok, message)``; ``message`` is empty on success or a short reason
    on failure (git missing / timeout / non-zero rc). A shallow single-branch
    clone keeps the footprint bounded; ``ref`` is checked out best-effort after
    the clone (a shallow clone may not carry every ref — a failed checkout is
    logged, not fatal, since the clone itself succeeded). Blocking git I/O runs
    under ``asyncio.wait_for`` so a hanging remote can't wedge the node.
    """
    args = ["git", "clone", "--depth", "1", "--single-branch", url, str(dest)]
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return False, "git is not installed in this environment."
    try:
        _out, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        await _terminate(proc)
        return False, f"git clone timed out after {timeout_s}s."
    if proc.returncode != 0:
        msg = (stderr.decode("utf-8", "replace").strip()[:300]) if stderr else "unknown"
        return False, f"git clone failed (rc={proc.returncode}): {msg}"

    if ref:
        await _checkout_ref(dest, ref, timeout_s)
    return True, ""


async def _terminate(proc: asyncio.subprocess.Process) -> None:
    """Best-effort SIGKILL of a runaway subprocess; never raises."""
    try:
        proc.kill()
    except ProcessLookupError:
        pass
    except OSError:  # pragma: no cover — defensive
        pass


async def _checkout_ref(dest: Path, ref: str, timeout_s: int) -> None:
    """Best-effort ``git checkout`` of ``ref`` into the shallow clone at ``dest``.

    A shallow ``--depth 1`` clone may not carry every ref, so a failed or timed-
    out checkout is logged and swallowed — the clone itself already succeeded, so
    indexing proceeds on whatever HEAD landed. Self-contained (its own process
    lifecycle) so a timeout never references an unbound handle.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(dest),
            "checkout",
            ref,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:  # pragma: no cover — git present if the clone ran
        logger.warning(f"git_clone: checkout ref={ref!r} failed (git missing)")
        return
    try:
        _out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        await _terminate(proc)
        logger.warning(f"git_clone: checkout ref={ref!r} timed out")
        return
    if proc.returncode != 0:
        msg = (err.decode("utf-8", "replace").strip()[:200]) if err else ""
        logger.warning(f"git_clone: checkout ref={ref!r} failed: {msg}")


def _chunk_python(source: str, rel_path: str) -> list[dict[str, str]]:
    """Split one Python source into per-symbol chunks.

    Each top-level ``def``/``async def``/``class`` becomes its own chunk; the
    remaining module-level statements (imports, constants, guard blocks) collapse
    into one ``<module>`` chunk. Each chunk's content carries a self-describing
    header (``path :: kind symbol``) so the embedding + the stored content stay
    anchored to their origin. A ``SyntaxError`` degrades to a single whole-file
    chunk — we never lose the file, only the per-symbol granularity.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [
            {
                "path": rel_path,
                "symbol": "<module>",
                "kind": "module",
                "content": f"# {rel_path} :: module (unparsed)\n{source}",
            }
        ]

    chunks: list[dict[str, str]] = []
    module_lines: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            seg = ast.get_source_segment(source, node) or ""
            chunks.append(
                {
                    "path": rel_path,
                    "symbol": node.name,
                    "kind": "function",
                    "content": f"# {rel_path} :: def {node.name}\n{seg}",
                }
            )
        elif isinstance(node, ast.ClassDef):
            seg = ast.get_source_segment(source, node) or ""
            chunks.append(
                {
                    "path": rel_path,
                    "symbol": node.name,
                    "kind": "class",
                    "content": f"# {rel_path} :: class {node.name}\n{seg}",
                }
            )
        else:
            seg = ast.get_source_segment(source, node)
            if seg:
                module_lines.append(seg)

    if module_lines:
        chunks.append(
            {
                "path": rel_path,
                "symbol": "<module>",
                "kind": "module",
                "content": f"# {rel_path} :: module-level\n" + "\n".join(module_lines),
            }
        )
    return chunks


def _walk_code_files(
    dest: Path, max_files: int, max_file_bytes: int, max_total_bytes: int
) -> list[tuple[Path, str]]:
    """Gather code files under ``dest`` honoring count/size/total caps.

    Returns ``[(abs_path, rel_path)]`` capped at ``max_files``. Skips VCS/cache
    dirs (``_SKIP_DIRS``), symlinks (so a repo link can't read outside dest),
    empty files, and files over ``max_file_bytes``. Stops once the cumulative
    size exceeds ``max_total_bytes``. Never raises — a stat/read miss just skips.
    """
    out: list[tuple[Path, str]] = []
    total = 0
    for root, dirs, files in os.walk(dest):
        # Mutate dirs in place so os.walk does not descend into skipped/junk dirs.
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for name in sorted(files):
            if len(out) >= max_files:
                return out
            p = Path(root) / name
            if p.suffix.lower() not in _CODE_SUFFIXES:
                continue
            try:
                # lstat to detect symlinks WITHOUT following them (a repo symlink
                # pointing outside dest must never be indexed).
                if p.is_symlink():
                    continue
                size = p.stat().st_size
            except OSError:
                continue
            if size == 0 or size > max_file_bytes:
                continue
            total += size
            if total > max_total_bytes:
                return out
            out.append((p, str(p.relative_to(dest))))
    return out


async def _index_chunks(chunks: list[dict[str, str]], repo_url: str) -> int:
    """Embed + store chunks as ``ColdMemory`` ``episode_type="code"`` rows.

    Returns the count stored. Non-fatal: a DB/embedding outage logs at DEBUG and
    returns 0 — the index is an aid, never a run-killer (CostTracker-resilience
    posture). One bad chunk can't abort the batch (per-store try/except).
    """
    if not chunks:
        return 0
    try:
        from src.config.settings import get_settings
        from src.db.session import get_session
        from src.memory.cold import ColdMemory
        from src.memory.embeddings import EmbeddingGenerator

        generator = EmbeddingGenerator(get_settings())
        stored = 0
        async with get_session() as session:
            memory = ColdMemory(session, generator=generator)
            for ch in chunks:
                tags = [
                    f"repo:{repo_url}",
                    f"path:{ch['path']}",
                    f"symbol:{ch['symbol']}",
                    f"kind:{ch['kind']}",
                ]
                try:
                    await memory.store(
                        episode_type="code",
                        content=ch["content"],
                        importance=0.6,
                        context_tags=tags,
                    )
                    stored += 1
                except Exception as exc:  # noqa: BLE001 — one bad chunk never aborts
                    logger.debug(
                        f"git_clone store failed {ch['path']}::{ch['symbol']}: {exc}"
                    )
        return stored
    except Exception as exc:  # noqa: BLE001 — non-fatal: indexing is never a run-killer
        logger.debug(f"git_clone index unavailable: {exc}")
        return 0


async def git_clone(url: str, ref: str = "", max_files: int | None = None) -> str:
    """Clone a public git repo and index its code into semantic memory.

    The clone lands in a confined workspace subdir (SSRF-guarded URL, path-
    confined destination, size/count-capped walk). Python files are split per
    function/class, each chunk embedded + stored as a recallable ``code`` episode.
    After indexing, ``code_search`` recalls a symbol by natural-language query.

    Args:
        url: Public ``http(s)`` git URL to clone (private/loopback hosts refused).
        ref: Optional branch/tag/commit to check out after the shallow clone
            (best-effort; a shallow clone may not carry every ref).
        max_files: Optional cap on files walked; can only LOWER the configured
            ``GIT_CLONE_MAX_FILES`` ceiling (a caller can't raise it).

    Returns:
        A one-line status (files/chunks stored) or a short ``ERROR:`` reason
        when the call was rejected (feature disabled, bad URL, SSRF, clone
        failure). Never raises.
    """
    settings = _git_clone_settings()
    if not settings.enabled:
        return (
            "git_clone is disabled (GIT_CLONE_ENABLED=false). Ask the operator "
            "to enable repo indexing before cloning external code."
        )

    url = (url or "").strip()
    ref = (ref or "").strip()
    if not url:
        return "git_clone needs a non-empty `url`."

    # SSRF guard — blocking DNS, so off the event loop. Returns None when safe,
    # or an "ERROR: ..." string the surface echoes verbatim.
    err = await asyncio.to_thread(assert_public_host, url)
    if err:
        return err

    try:
        dest = _clone_dest(url)
    except ValueError as exc:
        return f"ERROR: {exc}"

    # A caller may only TIGHTEN the file cap, never loosen it past the configured
    # ceiling — bounds a runaway walk regardless of the caller's argument.
    caller_files = int(max_files) if max_files else settings.max_files
    cap_files = max(1, min(settings.max_files, caller_files))

    # Idempotent-ish re-clone: clear a prior clone at this dest first. dest is
    # already confinement-verified by _clone_dest, so this rmtree cannot escape.
    try:
        if dest.exists():
            shutil.rmtree(dest)
        # Ensure the confined clones/ parent exists. git creates the leaf dir
        # but its parent-dir behavior is version-ambiguous; mkdir is idempotent
        # and keeps a fresh-clone (no clones/ yet) from failing.
        dest.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning(f"git_clone could not prep clone dest {dest}: {exc}")

    ok, msg = await _run_clone(url, dest, ref, settings.clone_timeout_s)
    if not ok:
        return f"ERROR: {msg}"

    files = _walk_code_files(
        dest, cap_files, settings.max_file_bytes, settings.max_total_bytes
    )
    chunks: list[dict[str, str]] = []
    for abs_path, rel_path in files:
        try:
            source = abs_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.debug(f"git_clone read failed {rel_path}: {exc}")
            continue
        chunks.extend(_chunk_python(source, rel_path))
        if len(chunks) >= settings.max_chunks:
            chunks = chunks[: settings.max_chunks]
            break

    stored = await _index_chunks(chunks, url)

    logger.info(
        f"git_clone: url={url} ref={ref or 'default'} files={len(files)} "
        f"chunks={len(chunks)} stored={stored} dest={dest}"
    )
    return (
        f"Cloned {url} and indexed {stored}/{len(chunks)} code chunk(s) from "
        f"{len(files)} file(s) into semantic memory. Use code_search to recall a "
        f"symbol by natural-language query."
    )


def _tag_value(tags: Any, key: str) -> str:
    """Extract ``key:value`` from a ``context_tags`` list; ``""`` if absent."""
    if not tags:
        return ""
    prefix = f"{key}:"
    for tag in tags:
        if isinstance(tag, str) and tag.startswith(prefix):
            return tag[len(prefix) :]
    return ""


async def code_search(query: str, top_k: int = 5) -> str:
    """Semantic search over indexed repo code (``episode_type="code"``).

    Recalls symbols/chunks previously indexed by ``git_clone``, ranked by
    embedding similarity. Distinct from ``corpus_search`` (indexed web pages) and
    ``memory_search`` (tier memory): this is the code-index recall leg.

    Args:
        query: Natural-language query describing the symbol/behavior to find.
        top_k: Maximum results to return (default 5).

    Returns:
        Formatted ranked results, each showing kind/symbol/path + a snippet, or a
        "0 results" message. Never raises — an unavailable index returns 0.
    """
    query = (query or "").strip()
    if not query:
        return "code_search needs a non-empty `query`."
    limit = max(1, int(top_k))
    try:
        from src.config.settings import get_settings
        from src.db.session import get_session
        from src.memory.cold import ColdMemory
        from src.memory.embeddings import EmbeddingGenerator

        generator = EmbeddingGenerator(get_settings())
        async with get_session() as session:
            memory = ColdMemory(session, generator=generator)
            rows = await memory.search_by_query(
                query, limit=limit, episode_type="code"
            )
    except Exception as exc:  # noqa: BLE001 — non-fatal: search never blocks a run
        logger.debug(f"code_search unavailable: {exc}")
        return f"Code search for '{query[:60]}' returned 0 results."

    if not rows:
        return (
            f"Code search for '{query[:60]}' returned 0 results. "
            "(Index a repo with git_clone first.)"
        )

    lines = [f"Code search for '{query[:60]}' — {len(rows)} result(s):"]
    for i, row in enumerate(rows, 1):
        tags = row.get("context_tags")
        sym = _tag_value(tags, "symbol") or "?"
        path = _tag_value(tags, "path") or "?"
        kind = _tag_value(tags, "kind") or "?"
        sim = row.get("similarity")
        snippet = " ".join(str(row.get("content") or "").split())[:160]
        lines.append(f"{i}. {kind} {sym} @ {path} [sim={sim}] {snippet}")
    return "\n".join(lines)


TOOL_DEFINITION_CLONE = {
    "name": "git_clone",
    "handler": git_clone,
    "description": (
        "Clone a PUBLIC git repository and index its source code into the "
        "agent's semantic code memory (default-off; no-op when disabled). Walks "
        "the Python files under SSRF + path-confinement + size caps, splits each "
        "per function/class, embeds every chunk, and stores them as recallable "
        "code episodes. After indexing, use `code_search` to recall a symbol or "
        "definition by natural-language query instead of re-cloning. Use this "
        "when a goal needs to reason about an external codebase the agent has "
        "not seen before. Never raises."
    ),
    # Each call fetches + mutates the index (disk + DB rows); never serve cached.
    "cacheable": False,
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": (
                    "Public http(s) git URL to clone (e.g. "
                    "'https://github.com/owner/repo.git'). Private/loopback "
                    "hosts are refused."
                ),
            },
            "ref": {
                "type": "string",
                "description": (
                    "Optional branch/tag/commit to check out after the shallow "
                    "clone. Omit for the default branch."
                ),
                "default": "",
            },
            "max_files": {
                "type": "integer",
                "description": (
                    "Optional cap on the number of files walked (can only lower "
                    "the configured ceiling). Omit for the default."
                ),
            },
        },
        "required": ["url"],
    },
}

TOOL_DEFINITION_SEARCH = {
    "name": "code_search",
    "handler": code_search,
    "description": (
        "Semantic search over source code the agent previously indexed with "
        "git_clone (recall by symbol/definition, not keyword). Ranks indexed "
        "code chunks by embedding similarity so a natural-language query returns "
        "the matching function/class. Use this to recall an external repo's "
        "internals without re-cloning. Distinct from corpus_search (indexed web "
        "pages) and memory_search (tier memory). Never raises."
    ),
    "cacheable": True,
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language query describing the symbol/behavior to find.",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum results to return (default 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}

# Registry convenience: expose the primary (indexing) definition.
TOOL_DEFINITION = TOOL_DEFINITION_CLONE
