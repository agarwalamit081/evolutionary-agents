"""Evolution→live promotion gate (Phase 8).

Promotes a deployed PROMPT mutation to the live agent *safely*. A mutation that
already passed the evolution pipeline (safety + sandbox + A/B + post-deploy
verify) is only promoted once a golden **canary** eval run with it scores at or
above ``EvalSettings.eval_canary_min_score``. On success a *versioned* artifact
is written under ``evolved_handlers_dir/prompts/`` and a ``current.json`` pointer
manifest is updated; the prompt builder reads that pointer and splices the
promoted suffixes into the target node's system prompt (tagged ``[evolved]``).

Scope: PROMPT mutations only (the dominant deployed type per battery-03).
CODE/TOOL mutations already reach the live agent through the DB tool registry;
CODE-into-core-src stays in the shadow repo (out of scope).

Rollback: the pointer keeps a per-node version history, so a regression restores
the previous version. This reuses the ``git_tracker.rollback`` *pattern* (restore
a known-good prior state) but applied to the evolved-prompts pointer, which lives
outside the shadow repo under ``.turing/evolved``.

The promotion is opt-in (``EVOLUTION_PROMOTE_TO_LIVE``); off by default so nothing
is ever promoted until the operator turns it on.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.graph.enums import MutationType

if TYPE_CHECKING:
    from src.config.settings import Settings

# A canary scores a candidate (node, suffixes) on 0.0–1.0, or returns None when
# it could not produce a signal (no golden suite / eval disabled / inconclusive).
# The gate treats None as "do not promote" (safe default).
Canary = Callable[[str, list[str]], Awaitable[float | None]]

_PROMPTS_SUBDIR = "prompts"
_POINTER_NAME = "current.json"
# The heuristic + LLM PROMPT payload targets the execute node by default (the
# dominant deployed target). Used when a payload omits ``target_node``.
_DEFAULT_TARGET_NODE = "execute"


def _utc_now_iso() -> str:
    """ISO-8601 UTC timestamp for version metadata."""
    return datetime.now(timezone.utc).isoformat()


def _sha8(record: dict[str, Any]) -> str:
    """Stable 8-char content hash for a version record.

    Identity is the promoted *content* (node + suffixes + reason + source) — NOT
    ``sha`` or ``promoted_at`` — so re-promoting identical guidance yields the
    same version (deduped history) while a genuinely different promotion gets a
    new one.
    """
    payload = {
        k: v for k, v in record.items() if k not in {"sha", "promoted_at"}
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:8]


def parse_prompt_payload(
    proposal: dict[str, Any],
) -> tuple[str, list[str]] | None:
    """Extract ``(target_node, suffixes)`` from a PROMPT mutation proposal.

    The PROMPT payload is JSON: ``{"target_node": "execute", "suffixes": [...]}``
    (heuristic + LLM paths both emit this shape). Returns ``None`` for a
    non-PROMPT mutation, an unparseable payload, or one without a non-empty
    ``suffixes`` list — so a malformed/garbage/free-text mutation never promotes.
    ``target_node`` defaults to ``"execute"`` when omitted.

    Args:
        proposal: The evolution mutation proposal (carries ``mutation_type`` +
            ``mutated_content``).

    Returns:
        ``(node, suffixes)`` or ``None`` when the proposal is not promotable.
    """
    if proposal.get("mutation_type") != MutationType.PROMPT:
        return None
    content = proposal.get("mutated_content", "")
    if not isinstance(content, str) or not content.strip():
        return None
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    suffixes = payload.get("suffixes")
    if not isinstance(suffixes, list) or not suffixes:
        return None
    clean = [s for s in suffixes if isinstance(s, str) and s.strip()]
    if not clean:
        return None
    node = payload.get("target_node")
    if not isinstance(node, str) or not node.strip():
        node = _DEFAULT_TARGET_NODE
    return node.strip(), clean


class PromotionGate:
    """Versioned promotion of PROMPT mutations to the live agent, canary-gated.

    On a promotable PROMPT proposal: runs the configured canary; if it scores
    >= ``eval_canary_min_score`` writes a versioned artifact + updates the
    ``current.json`` pointer so the prompt builder loads the suffixes. A failing
    or inconclusive canary leaves the prior pointer untouched. ``rollback``
    restores the previous promoted version for a node.
    """

    def __init__(
        self,
        *,
        handlers_dir: str | Path | None = None,
        canary: Canary | None = None,
        min_score: float | None = None,
        settings: Settings | None = None,
    ) -> None:
        from src.config.settings import get_settings

        self._settings = settings or get_settings()
        evolution = self._settings.evolution
        self._handlers_dir = Path(handlers_dir or evolution.evolved_handlers_dir)
        self._prompts_dir = self._handlers_dir / _PROMPTS_SUBDIR
        self._canary = canary
        eval_cfg = self._settings.eval
        self._min_score = (
            eval_cfg.eval_canary_min_score if min_score is None else min_score
        )

    @property
    def prompts_dir(self) -> Path:
        """Directory holding versioned prompt artifacts + the pointer manifest."""
        return self._prompts_dir

    def _pointer_path(self) -> Path:
        return self._prompts_dir / _POINTER_NAME

    def _read_pointer(self) -> dict[str, Any]:
        path = self._pointer_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug(f"Evolved-prompts pointer unreadable ({exc}); treating as empty")
            return {}

    def _write_pointer(self, data: dict[str, Any]) -> None:
        self._prompts_dir.mkdir(parents=True, exist_ok=True)
        self._pointer_path().write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _read_version(self, name: str) -> dict[str, Any]:
        if not name:
            return {}
        path = self._prompts_dir / name
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def current_suffixes(self, node: str) -> list[str]:
        """Active promoted suffixes for ``node`` (empty when none promoted)."""
        entry = self._read_pointer().get(node)
        if not isinstance(entry, dict):
            return []
        suffixes = entry.get("suffixes")
        if not isinstance(suffixes, list):
            return []
        return [s for s in suffixes if isinstance(s, str)]

    async def promote(self, proposal: dict[str, Any]) -> dict[str, Any]:
        """Canary-gate a PROMPT proposal and promote it on a passing score.

        Args:
            proposal: The deployed mutation proposal.

        Returns:
            Dict with ``promoted`` (bool) and, on success, ``node``/``version``/
            ``sha``/``canary_score``; on rejection, a ``reason``.
        """
        parsed = parse_prompt_payload(proposal)
        if parsed is None:
            return {"promoted": False, "reason": "not a promotable PROMPT mutation"}
        node, suffixes = parsed

        # Canary gate: promotion requires a passing eval canary. No canary → no
        # promotion (safe); the evolve node logs this and may wire one next run.
        if self._canary is None:
            return {"promoted": False, "reason": "no canary wired", "node": node}
        try:
            score = await self._canary(node, suffixes)
        except Exception as exc:  # noqa: BLE001 — canary is pluggable; never abort the cycle
            logger.warning(f"Promotion canary errored: {exc}")
            return {"promoted": False, "reason": f"canary error: {exc}", "node": node}
        if score is None:
            return {"promoted": False, "reason": "canary inconclusive", "node": node}
        if score < self._min_score:
            logger.info(
                f"PROMPT mutation for '{node}' NOT promoted: canary {score:.2f} "
                f"< {self._min_score:.2f} (prior pointer retained)"
            )
            return {
                "promoted": False,
                "reason": "canary below threshold",
                "node": node,
                "canary_score": score,
            }

        return self._install(node, suffixes, score, proposal)

    def _install(
        self,
        node: str,
        suffixes: list[str],
        canary_score: float,
        proposal: dict[str, Any],
    ) -> dict[str, Any]:
        """Write the versioned artifact + update the pointer (passing canary)."""
        promoted_at = _utc_now_iso()
        version_record: dict[str, Any] = {
            "node": node,
            "suffixes": suffixes,
            "canary_score": round(float(canary_score), 4),
            "promoted_at": promoted_at,
            "reason": proposal.get("rationale") or proposal.get("description", ""),
            "source": proposal.get("model_used") or "heuristic",
        }
        sha = _sha8(version_record)
        version_record["sha"] = sha
        version_name = f"{node}.{sha}.json"

        # Immutable versioned artifact (audit trail; never auto-deleted).
        self._prompts_dir.mkdir(parents=True, exist_ok=True)
        (self._prompts_dir / version_name).write_text(
            json.dumps(version_record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Pointer manifest: active entry + per-node history for rollback.
        pointer = self._read_pointer()
        entry = pointer.get(node)
        if not isinstance(entry, dict):
            entry = {}
        raw_history = entry.get("history")
        history: list[dict[str, Any]] = (
            [h for h in raw_history if isinstance(h, dict)]
            if isinstance(raw_history, list)
            else []
        )
        # Dedup: re-promoting identical content doesn't grow the history.
        if not history or history[-1].get("sha") != sha:
            history.append({
                "sha": sha,
                "version": version_name,
                "canary_score": round(float(canary_score), 4),
                "promoted_at": promoted_at,
            })
        entry.update({
            "active": version_name,
            "active_sha": sha,
            "suffixes": suffixes,
            "canary_score": round(float(canary_score), 4),
            "promoted_at": promoted_at,
            "history": history,
        })
        pointer[node] = entry
        self._write_pointer(pointer)

        logger.info(
            f"PROMPT mutation PROMOTED to live for '{node}': {version_name} "
            f"(canary {canary_score:.2f})"
        )
        return {
            "promoted": True,
            "node": node,
            "version": version_name,
            "sha": sha,
            "canary_score": canary_score,
        }

    def rollback(self, node: str) -> dict[str, Any]:
        """Restore the prior promoted version for ``node`` (regression recovery).

        Pops the active entry off the per-node history and reverts the pointer to
        the previous version's suffixes — or removes the node when no prior
        version remains. Versioned files are retained as an audit trail. Mirrors
        ``git_tracker.rollback``'s restore-known-good-state semantics, applied to
        the evolved-prompts pointer (which lives outside the shadow repo).

        Args:
            node: The node whose active promotion should be reverted.

        Returns:
            Dict with ``rolled_back`` (bool), ``restored`` (version or None),
            and ``removed`` (the popped version).
        """
        pointer = self._read_pointer()
        entry = pointer.get(node)
        if not isinstance(entry, dict):
            return {
                "rolled_back": False,
                "reason": "no promoted version for node",
                "node": node,
            }
        raw_history = entry.get("history")
        history: list[dict[str, Any]] = (
            [h for h in raw_history if isinstance(h, dict)]
            if isinstance(raw_history, list)
            else []
        )
        if not history:
            return {"rolled_back": False, "reason": "no history to revert to", "node": node}

        removed = history.pop()
        if history:
            prev = history[-1]
            prev_version = self._read_version(prev.get("version", ""))
            prev_suffixes = prev_version.get("suffixes")
            entry.update({
                "active": prev.get("version"),
                "active_sha": prev.get("sha"),
                "suffixes": prev_suffixes if isinstance(prev_suffixes, list) else [],
                "canary_score": prev.get("canary_score"),
                "promoted_at": prev.get("promoted_at"),
                "history": history,
            })
            pointer[node] = entry
            self._write_pointer(pointer)
            logger.info(
                f"Rolled back '{node}': {removed.get('version')} → {prev.get('version')}"
            )
            return {
                "rolled_back": True,
                "node": node,
                "restored": prev.get("version"),
                "removed": removed.get("version"),
            }

        # No prior version → drop the node from the live pointer entirely.
        pointer.pop(node, None)
        self._write_pointer(pointer)
        logger.info(
            f"Rolled back '{node}': removed last promoted version "
            f"({removed.get('version')})"
        )
        return {
            "rolled_back": True,
            "node": node,
            "restored": None,
            "removed": removed.get("version"),
        }


class GoldenCanary:
    """Eval canary: run a golden GoalSpec subset with candidate suffixes active.

    Applies the candidate suffixes via an in-process builder override (so the
    canary run sees them WITHOUT mutating the on-disk pointer), runs each golden
    goal through ``BenchmarkHarness`` — which runs the verify-node correctness
    checks when ``EVAL_ENABLED`` — and returns the mean ``correctness_score``.
    Returns ``None`` when no goal produced a score (eval disabled / no spec
    reached verify) so the gate treats the canary as inconclusive and does not
    promote. Defaults to the cheapest deterministic goal (``battery04_q01``).
    """

    def __init__(
        self,
        gateway: Any,
        tools: Any,
        sub_agent_registry: Any,
        *,
        goal_ids: list[str] | None = None,
        harness: Any | None = None,
    ) -> None:
        from src.eval.golden import GOLDEN_SPECS

        if harness is not None:
            self._harness = harness
        else:
            from src.eval.harness import BenchmarkHarness

            self._harness = BenchmarkHarness(gateway, tools, sub_agent_registry)
        ids = goal_ids or ["battery04_q01"]
        self._suite = [GOLDEN_SPECS[i] for i in ids if i in GOLDEN_SPECS]

    async def score(self, node: str, suffixes: list[str]) -> float | None:
        """Run the golden suite with ``suffixes`` active for ``node``; mean score."""
        from src.graph.prompts.builder import (
            clear_evolved_candidate,
            set_evolved_candidate,
        )

        if not self._suite:
            return None
        set_evolved_candidate(node, suffixes)
        scores: list[float] = []
        try:
            for spec in self._suite:
                result = await self._harness.run_benchmark(
                    spec.to_benchmark_goal(), spec=spec
                )
                if result.correctness_score is not None:
                    scores.append(float(result.correctness_score))
        finally:
            clear_evolved_candidate(node)
        if not scores:
            return None
        return sum(scores) / len(scores)
