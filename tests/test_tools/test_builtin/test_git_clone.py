"""Unit tests for the git-clone code indexer (Phase 5 I2).

DoD (findings.md:353-357): clone → embed → index → a ``code_search`` returns the
relevant chunk by semantic similarity. These tests prove the full pipeline end to
end WITHOUT a real git binary, a real DB, or a real embedding API:

  * the ``git`` subprocess is mocked with an on-disk fixture repo (``_run_clone``
    is patched to ``copytree`` the fixture into the confined dest);
  * ``EmbeddingGenerator`` is a deterministic bag-of-keywords vector;
  * ``ColdMemory`` is a fake doing REAL cosine similarity over the stored
    embeddings (so a search genuinely ranks, not just returns).

So the system under test — the SSRF guard, path confinement, the walk + caps, the
AST chunker, and the store/search wiring — runs for real; only the external
dependencies (git, the DB session, the embedding API) are faked, per the testing
rules (mock dependencies, never the system under test).

Caps + SSRF + path-confinement are asserted directly so a future change that
loosens them is caught.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.tools.builtin import git_clone as GC

# ─── Deterministic fakes for the external dependencies ───────────────

#: Keyword → embedding-dimension groups. A dimension flips to 1.0 when ANY of its
#: keywords appears in the (lower-cased) text, giving a cheap but *semantic*
#: vector: a query about "authenticate" lands in the same dimension as a chunk
#: whose source defines ``def authenticate(... password ...)``.
_KW: dict[int, tuple[str, ...]] = {
    0: ("authenticate", "auth", "login", "password", "credential"),
    1: ("sort", "order", "rank", "sorted"),
    2: ("parse", "load", "csv", "data", "rows"),
}
_EMBED_DIM = 8


class _FakeEmbedder:
    """Deterministic keyword-bag embedder (drop-in for ``EmbeddingGenerator``)."""

    dimension = _EMBED_DIM
    last_source = "hash"
    model = "fake"

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    async def generate(self, text: str) -> list[float]:
        low = (text or "").lower()
        vec = [0.0] * _EMBED_DIM
        for dim, words in _KW.items():
            if any(w in low for w in words):
                vec[dim] = 1.0
        return vec


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity; zero-norm ⇒ 0.0 (no division-by-zero)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class _FakeColdMemory:
    """In-memory ColdMemory: stores chunks + ranks by real cosine over embeddings.

    ``STORE`` is a CLASS attribute shared across the per-call instances the real
    code creates (``_index_chunks`` and ``code_search`` each open their own
    session), so an index then a recall see the same population.
    """

    STORE: list[dict[str, object]] = []

    def __init__(self, session: object, embedding_dim: int = _EMBED_DIM, generator: object = None) -> None:
        self._generator = generator

    async def store(
        self,
        episode_type: str,
        content: str,
        importance: float = 0.5,
        context_tags: list[str] | None = None,
        embedding: list[float] | None = None,
    ) -> str:
        if embedding is None and self._generator is not None:
            embedding = await self._generator.generate(content)  # type: ignore[union-attr]
        row = {
            "id": str(len(self.STORE)),
            "episode_type": episode_type,
            "content": content,
            "importance": importance,
            "context_tags": context_tags or [],
            "embedding": embedding,
        }
        self.STORE.append(row)
        return str(row["id"])

    async def search_by_query(
        self,
        query: str,
        limit: int = 5,
        min_importance: float = 0.0,
        episode_type: str | None = None,
    ) -> list[dict[str, object]]:
        if not query or self._generator is None:
            return []
        q = await self._generator.generate(query)  # type: ignore[union-attr]
        scored: list[tuple[float, dict[str, object]]] = []
        for row in self.STORE:
            if episode_type is not None and row["episode_type"] != episode_type:
                continue
            sim = _cosine(q, list(row.get("embedding") or []))  # type: ignore[arg-type]
            scored.append((sim, row))
        scored.sort(key=lambda t: t[0], reverse=True)
        out: list[dict[str, object]] = []
        for sim, row in scored[:limit]:
            out.append(
                {
                    "id": row["id"],
                    "episode_type": row["episode_type"],
                    "content": row["content"],
                    "importance": row["importance"],
                    "context_tags": row["context_tags"],
                    "similarity": sim,
                }
            )
        return out


class _FakeSession:
    """Minimal async-context-manager session (the fakes ignore it)."""

    async def __aenter__(self) -> object:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False


def _fake_get_session() -> _FakeSession:
    return _FakeSession()


def _clone_copying(fixture_src: Path):
    """Return a mock ``_run_clone`` that materializes the fixture repo at dest."""

    async def _m(url: str, dest: Path, ref: str, timeout_s: int) -> tuple[bool, str]:
        shutil.copytree(fixture_src, dest)
        return True, ""

    return _m


async def _clone_fail(url: str, dest: Path, ref: str, timeout_s: int) -> tuple[bool, str]:
    return False, "git clone failed (rc=128): repository not found"


# ─── Wiring helper ────────────────────────────────────────────────────


def _wire(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    enabled: bool = True,
    run_clone: object | None = None,
    **gc_over: int,
) -> tuple[SimpleNamespace, Any]:
    """Patch the four seams + return the configured ``git_clone`` settings fake.

    ``run_clone`` defaults to a never-called ``AsyncMock``; pass a real async mock
    (e.g. ``_clone_copying(fixture)``) for pipeline tests.
    """
    gc = SimpleNamespace(
        enabled=enabled,
        max_files=200,
        max_file_bytes=262_144,
        max_total_bytes=20_971_520,
        max_chunks=300,
        clone_timeout_s=120,
    )
    for key, val in gc_over.items():
        setattr(gc, key, val)
    fake_settings = SimpleNamespace(
        git_clone=gc,
        agent=SimpleNamespace(workspace_root=str(tmp_path), results_root=str(tmp_path)),
    )
    monkeypatch.setattr("src.config.settings.get_settings", lambda: fake_settings)
    monkeypatch.setattr("src.db.session.get_session", _fake_get_session)
    monkeypatch.setattr("src.memory.cold.ColdMemory", _FakeColdMemory)
    monkeypatch.setattr("src.memory.embeddings.EmbeddingGenerator", _FakeEmbedder)
    _FakeColdMemory.STORE.clear()
    clone_mock = run_clone if run_clone is not None else AsyncMock()
    monkeypatch.setattr(GC, "_run_clone", clone_mock)
    return gc, clone_mock


def _stored(key: str) -> set[str]:
    """All values for a ``key:value`` context-tag across the STORE population."""
    out: set[str] = set()
    for row in _FakeColdMemory.STORE:
        out.add(GC._tag_value(row.get("context_tags"), key))  # type: ignore[arg-type]
    out.discard("")
    return out


def _build_demo_repo(base: Path) -> Path:
    """A 3-symbol fixture repo (auth / sort / data) + a non-.py file + junk dir."""
    repo = base / "demo"
    repo.mkdir(parents=True)
    (repo / "auth.py").write_text(
        "import hashlib\n\n\n"
        "def authenticate(username, password):\n"
        '    """Verify the user login credentials by hashing the password."""\n'
        '    return hashlib.sha256(password.encode("utf-8")).hexdigest()\n',
        encoding="utf-8",
    )
    (repo / "sort.py").write_text(
        "def sort_items(records, key):\n"
        '    """Order the records ascending by the given key."""\n'
        "    return sorted(records, key=lambda r: r.get(key))\n",
        encoding="utf-8",
    )
    (repo / "data.py").write_text(
        'class DataParser:\n'
        '    """Parse and load CSV data rows into records."""\n\n'
        "    def load(self, path):\n"
        "        rows = self._read(path)\n"
        "        return [r for r in rows]\n",
        encoding="utf-8",
    )
    # Non-Python file must be ignored by the .py-only walker.
    (repo / "README.md").write_text("# auth and sort helpers\n", encoding="utf-8")
    # A vendored/build dir must be pruned.
    (repo / "build").mkdir()
    (repo / "build" / "gen.py").write_text("def generated():\n    return 0\n", encoding="utf-8")
    return repo


# ─── Chunker (pure, no mocks) ─────────────────────────────────────────


class TestChunker:
    """The AST chunker splits per symbol and degrades gracefully on bad syntax."""

    def test_splits_function_class_and_module(self) -> None:
        source = (
            "import os\n"
            "CONST = 1\n\n"
            "def foo(x):\n"
            "    return x + 1\n\n"
            "class Bar:\n"
            "    def m(self):\n"
            "        return 2\n"
        )
        chunks = GC._chunk_python(source, "mod.py")
        by_symbol = {(c["kind"], c["symbol"]) for c in chunks}
        assert ("function", "foo") in by_symbol
        assert ("class", "Bar") in by_symbol
        # Module-level statements collapse into one <module> chunk.
        assert ("module", "<module>") in by_symbol
        # Every chunk carries its origin header.
        assert all("mod.py" in c["content"] for c in chunks)

    def test_syntax_error_falls_back_to_whole_file(self) -> None:
        chunks = GC._chunk_python("def broken(:\n    pass\n", "bad.py")
        assert len(chunks) == 1
        assert chunks[0]["kind"] == "module"
        assert chunks[0]["symbol"] == "<module>"


# ─── Walker (pure, on-disk tree) ──────────────────────────────────────


class TestWalkCodeFiles:
    """The walker honors caps and skips junk dirs / symlinks / non-code."""

    def test_skips_junk_dirs_symlinks_nonpy_and_caps(self, tmp_path: Path) -> None:
        dest = tmp_path / "clones" / "demo"
        dest.mkdir(parents=True)
        (dest / "real.py").write_text("def keep():\n    return 1\n", encoding="utf-8")
        (dest / "notes.txt").write_text("ignore me", encoding="utf-8")
        (dest / "empty.py").write_text("", encoding="utf-8")
        # Oversized file (> max_file_bytes) is skipped.
        (dest / "big.py").write_text("# " + "x" * 200 + "\n", encoding="utf-8")
        # Junk dir pruned.
        (dest / "__pycache__").mkdir()
        (dest / "__pycache__" / "cached.py").write_text("x = 1\n", encoding="utf-8")
        # Symlink (even to a real .py) is never followed.
        (dest / "link.py").symlink_to(dest / "real.py")

        files = GC._walk_code_files(dest, max_files=100, max_file_bytes=64, max_total_bytes=2_097_152)
        rels = {rel for _abs, rel in files}
        assert rels == {"real.py"}

    def test_max_files_cap_stops_the_walk(self, tmp_path: Path) -> None:
        dest = tmp_path / "r"
        dest.mkdir(parents=True)
        for name in ("a.py", "b.py", "c.py"):
            (dest / name).write_text("def f():\n    pass\n", encoding="utf-8")
        files = GC._walk_code_files(dest, max_files=1, max_file_bytes=262_144, max_total_bytes=2_097_152)
        # Exactly one file (alphabetically first), then the cap stops the walk.
        assert len(files) == 1
        assert files[0][1] == "a.py"

    def test_max_total_bytes_cap_stops_the_walk(self, tmp_path: Path) -> None:
        dest = tmp_path / "r"
        dest.mkdir(parents=True)
        a = dest / "a.py"
        b = dest / "b.py"
        a.write_text("def a():\n    return 0\n", encoding="utf-8")
        b.write_text("def b():\n    return 0\n", encoding="utf-8")
        # Cap = a's exact size: a fits (total == cap, not > cap), b pushes the
        # running total over cap so the walk returns before appending it.
        cap = len(a.read_bytes())
        files = GC._walk_code_files(dest, max_files=100, max_file_bytes=262_144, max_total_bytes=cap)
        assert {rel for _abs, rel in files} == {"a.py"}


# ─── Path confinement ────────────────────────────────────────────────


class TestCloneDest:
    """The clone destination stays inside the workspace and sanitizes the slug."""

    def test_dest_under_workspace_and_slug_sanitized(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        fake = SimpleNamespace(
            git_clone=SimpleNamespace(enabled=True),
            agent=SimpleNamespace(workspace_root=str(tmp_path), results_root=str(tmp_path)),
        )
        monkeypatch.setattr("src.config.settings.get_settings", lambda: fake)

        dest = GC._clone_dest("https://github.com/Owner/My.Repo.git")
        base = (tmp_path / "clones").resolve()
        assert dest.is_relative_to(base)
        assert dest.name == "My.Repo"

    def test_traversal_tail_collapses_to_safe_slug(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        fake = SimpleNamespace(
            git_clone=SimpleNamespace(enabled=True),
            agent=SimpleNamespace(workspace_root=str(tmp_path), results_root=str(tmp_path)),
        )
        monkeypatch.setattr("src.config.settings.get_settings", lambda: fake)

        # A tail of only dots sanitizes (strip) to the fallback "repo" — never "..".
        dest = GC._clone_dest("https://example.com/a/..")
        base = (tmp_path / "clones").resolve()
        assert dest.is_relative_to(base)
        assert dest.name == "repo"


# ─── Disabled / guardrails ────────────────────────────────────────────


class TestDisabledAndGuardrails:
    """Feature-off is a no-op; empty URL / SSRF / clone-fail return ERROR, no clone."""

    async def test_disabled_is_a_noop(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _gc, clone_mock = _wire(monkeypatch, tmp_path, enabled=False)
        result = await GC.git_clone("https://github.com/example/demo.git")
        assert "disabled" in result.lower()
        clone_mock.assert_not_called()
        assert _FakeColdMemory.STORE == []

    async def test_empty_url_rejected_before_clone(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _gc, clone_mock = _wire(monkeypatch, tmp_path)
        result = await GC.git_clone("   ")
        assert "non-empty" in result.lower()
        clone_mock.assert_not_called()

    async def test_ssrf_loopback_rejected_no_clone(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        # A literal loopback IP needs no DNS — assert_public_host resolves it
        # locally and returns the blocked-host error deterministically.
        _gc, clone_mock = _wire(monkeypatch, tmp_path)
        result = await GC.git_clone("http://127.0.0.1/secret.git")
        assert result.startswith("ERROR:")
        assert "non-public" in result.lower()
        clone_mock.assert_not_called()

    async def test_clone_failure_surfaces_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _gc, _mock = _wire(monkeypatch, tmp_path, run_clone=_clone_fail)
        result = await GC.git_clone("https://github.com/example/missing.git")
        assert result.startswith("ERROR:")
        assert "not found" in result.lower()
        assert _FakeColdMemory.STORE == []


# ─── Caps enforced through the full handler ──────────────────────────


class TestCapsEnforced:
    """The handler's caps truncate the walk/index (not just the bare walker)."""

    async def test_max_files_truncates_to_one_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        fixture = _build_demo_repo(tmp_path / "fixture")
        _gc, _mock = _wire(monkeypatch, tmp_path, run_clone=_clone_copying(fixture), max_files=1)
        result = await GC.git_clone("https://github.com/example/demo.git")
        assert "indexed" in result.lower()
        # Only one file's symbols were stored (alphabetically auth.py).
        assert _stored("path") == {"auth.py"}

    async def test_max_file_bytes_skips_oversized(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        fixture = _build_demo_repo(tmp_path / "fixture")
        # auth.py (~210 B) and data.py (~120 B) exceed the cap; sort.py (~95 B) too.
        # With a tight cap only nothing large survives — assert NO file is indexed,
        # proving the size gate is consulted by the handler (cap raised slightly so
        # the smallest repo file clears it shows selective indexing instead).
        _gc, _mock = _wire(monkeypatch, tmp_path, run_clone=_clone_copying(fixture), max_file_bytes=10)
        await GC.git_clone("https://github.com/example/demo.git")
        # Every fixture .py is well over 10 bytes → none indexed.
        assert _FakeColdMemory.STORE == []

    async def test_max_chunks_truncates_index(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        fixture = _build_demo_repo(tmp_path / "fixture")
        _gc, _mock = _wire(monkeypatch, tmp_path, run_clone=_clone_copying(fixture), max_chunks=2)
        await GC.git_clone("https://github.com/example/demo.git")
        assert len(_FakeColdMemory.STORE) <= 2


# ─── The DoD pipeline: clone → embed → index → search ranks match first ─


class TestCloneIndexSearchPipeline:
    """clone → embed → index → code_search returns the matching symbol first."""

    async def test_search_ranks_matching_symbol_first(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        fixture = _build_demo_repo(tmp_path / "fixture")
        _gc, _mock = _wire(monkeypatch, tmp_path, run_clone=_clone_copying(fixture))

        index_status = await GC.git_clone("https://github.com/example/demo.git")
        # Indexing happened.
        assert "indexed" in index_status.lower()
        stored_count = len(_FakeColdMemory.STORE)
        assert stored_count > 0
        # Every stored chunk is a code episode tagged with its symbol/path/kind.
        for row in _FakeColdMemory.STORE:
            assert row["episode_type"] == "code"
            tags = row["context_tags"]
            assert any(str(t).startswith("symbol:") for t in tags)  # type: ignore[union-attr]
            assert any(str(t).startswith("path:") for t in tags)  # type: ignore[union-attr]
            assert any(str(t).startswith("kind:") for t in tags)  # type: ignore[union-attr]

        # Search for the auth behavior → the authenticate chunk ranks #1.
        out = await GC.code_search("how does the agent authenticate a user login")
        lines = out.splitlines()
        assert "result" in lines[0].lower()
        first = lines[1].lower()
        assert "authenticate" in first
        assert "function" in first
        # The other symbols are not the top hit.
        assert "sort_items" not in first
        assert "dataparser" not in first

    async def test_search_empty_query_and_empty_index(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        _gc, _mock = _wire(monkeypatch, tmp_path)
        # Empty query short-circuits before any index access.
        assert "non-empty" in (await GC.code_search("   ")).lower()
        # No repo indexed yet → 0 results (not a raise).
        out = await GC.code_search("anything at all")
        assert "0 results" in out.lower()
