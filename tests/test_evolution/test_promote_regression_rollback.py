"""Promotion regression rollback + canary-recursion guard.

Complements ``test_promote.py`` (passing/failing/inconclusive canary unit cases)
and ``test_promote_integration.py`` (live read-back across instances) with the
REGRESSION safety surfaces:

* a PROMPT mutation whose post-deploy canary REGRESSES below
  ``eval_canary_min_score`` is NOT promoted and leaves the PRIOR ``current``
  pointer + versioned artifact intact (auto-rollback-in-spirit: a regressing
  candidate never overwrites a good promotion). This is the O2 canary's job —
  it gates promotion, so a regression at promotion time is rejected rather
  than later reverted.
* a PASSING canary writes BOTH the versioned artifact ``<node>.<sha>.json``
  AND the ``current.json`` pointer with the active entry.
* ``rollback()`` reverts the pointer to the prior version (or removes the node
  when no prior version remains); the immutable versioned artifact survives as
  an audit trail.
* the canary-recursion guard (commit 4272d02): ``BenchmarkHarness.run_benchmark``
  ALWAYS constructs ``initial_state(..., no_evolution=True)`` — regardless of
  the caller — so a ``promote() → GoldenCanary → run_benchmark → evolve →
  run_cycle → promote()`` cascade terminates at the root (battery-04 q09
  spawned 3 mutation chains from one canary before this guard).

Deterministic: an in-process fake canary scripts the score; the harness is
patched so no graph actually runs. No src/ file is modified.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from src.evolution.promote import PromotionGate
from src.graph.enums import MutationType


# ─── helpers ─────────────────────────────────────────────────────────────


def _fake_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    *,
    min_score: float = 0.8,
) -> SimpleNamespace:
    fake = SimpleNamespace(
        evolution=SimpleNamespace(
            evolution_promote_to_live=True,
            evolved_handlers_dir=str(tmp_path / "evolved"),
        ),
        eval=SimpleNamespace(eval_canary_min_score=min_score),
    )
    monkeypatch.setattr("src.config.settings.get_settings", lambda: fake)
    return fake


def _proposal(
    mutated_content: str,
    *,
    rationale: str = "guide valid JSON",
    target_path: str = "prompts/system_prompt.md",
) -> dict[str, Any]:
    return {
        "mutation_type": MutationType.PROMPT,
        "description": "address JSON mistakes",
        "rationale": rationale,
        "model_used": "test-model",
        "mutated_content": mutated_content,
        "target_path": target_path,
    }


def _json_proposal(suffixes: list[str], node: str = "execute") -> dict[str, Any]:
    return _proposal(json.dumps({"target_node": node, "suffixes": suffixes}))


def _canary(score: float | None) -> Any:
    async def _fn(_node: str, _suffixes: list[str]) -> float | None:
        return score

    return _fn


# ─── passing canary writes versioned artifact + pointer ──────────────────


class TestPassingCanaryWritesArtifacts:
    @pytest.mark.asyncio
    async def test_passing_canary_writes_versioned_artifact_and_pointer(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _fake_settings(monkeypatch, tmp_path, min_score=0.8)
        gate = PromotionGate(canary=_canary(0.9))

        result = await gate.promote(_json_proposal(["always emit valid JSON"]))

        assert result["promoted"] is True
        version_file = gate.prompts_dir / result["version"]
        assert version_file.exists()
        # The versioned artifact carries the suffixes + score + sha.
        record = json.loads(version_file.read_text("utf-8"))
        assert record["suffixes"] == ["always emit valid JSON"]
        assert record["canary_score"] == 0.9
        assert record["sha"] == result["sha"]
        # The pointer names the version as the active entry.
        pointer = json.loads((gate.prompts_dir / "current.json").read_text("utf-8"))
        assert pointer["execute"]["active"] == result["version"]
        assert pointer["execute"]["active_sha"] == result["sha"]


# ─── a regressing candidate retains the prior pointer ────────────────────


class TestRegressionCandidateRetainsPrior:
    """A second PROMPT mutation whose canary REGRESSES below the threshold is
    rejected — it must NOT overwrite the active pointer left by a prior
    passing promotion. This is the O2 contract: the canary gates promotion,
    and a regression at promotion time is rejected rather than installed."""

    @pytest.mark.asyncio
    async def test_regressing_candidate_does_not_overwrite_prior(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _fake_settings(monkeypatch, tmp_path, min_score=0.8)
        gate = PromotionGate(canary=_canary(0.9))
        # v1: passing canary → promoted + pointer set.
        first = await gate.promote(_json_proposal(["good guidance v1"]))
        assert first["promoted"] is True
        assert gate.current_suffixes("execute") == ["good guidance v1"]

        # v2: regressing canary (< min_score) → NOT promoted; prior retained.
        gate._canary = _canary(0.5)  # type: ignore[assignment]
        second = await gate.promote(_json_proposal(["risky guidance v2"]))

        assert second["promoted"] is False
        assert second["reason"] == "canary below threshold"
        assert second["canary_score"] == 0.5
        # The prior promotion is untouched.
        assert gate.current_suffixes("execute") == ["good guidance v1"]
        pointer = json.loads((gate.prompts_dir / "current.json").read_text("utf-8"))
        assert pointer["execute"]["active"] == first["version"]
        # Only ONE history entry — the regressing candidate never landed.
        assert len(pointer["execute"]["history"]) == 1

    @pytest.mark.asyncio
    async def test_inconclusive_canary_retains_prior(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _fake_settings(monkeypatch, tmp_path, min_score=0.8)
        gate = PromotionGate(canary=_canary(0.9))
        first = await gate.promote(_json_proposal(["good guidance"]))
        assert first["promoted"] is True

        gate._canary = _canary(None)  # type: ignore[assignment]
        second = await gate.promote(_json_proposal(["next guidance"]))

        assert second["promoted"] is False
        assert second["reason"] == "canary inconclusive"
        assert gate.current_suffixes("execute") == ["good guidance"]


# ─── rollback reverts the pointer / drops the node ───────────────────────


class TestRollbackRevertsOrDrops:
    @pytest.mark.asyncio
    async def test_rollback_to_prior_version(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _fake_settings(monkeypatch, tmp_path, min_score=0.8)
        gate = PromotionGate(canary=_canary(0.9))
        await gate.promote(_json_proposal(["v1 guidance"]))
        await gate.promote(_json_proposal(["v2 guidance"]))
        assert gate.current_suffixes("execute") == ["v2 guidance"]

        rolled = gate.rollback("execute")

        assert rolled["rolled_back"] is True
        assert rolled["restored"] is not None
        assert rolled["removed"] is not None
        # The pointer reverted to the prior active suffix.
        assert gate.current_suffixes("execute") == ["v1 guidance"]

    @pytest.mark.asyncio
    async def test_rollback_last_version_drops_node(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _fake_settings(monkeypatch, tmp_path, min_score=0.8)
        gate = PromotionGate(canary=_canary(0.9))
        promoted = await gate.promote(_json_proposal(["only guidance"]))
        version_file = gate.prompts_dir / promoted["version"]
        assert version_file.exists()

        rolled = gate.rollback("execute")

        assert rolled["rolled_back"] is True
        assert rolled["restored"] is None  # no prior → node dropped
        assert gate.current_suffixes("execute") == []
        # The immutable versioned artifact survives the rollback (audit trail).
        assert version_file.exists()

    def test_rollback_unknown_node_is_safe(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _fake_settings(monkeypatch, tmp_path, min_score=0.8)
        gate = PromotionGate()

        rolled = gate.rollback("reflect")  # never promoted

        assert rolled["rolled_back"] is False
        assert "no promoted version" in rolled["reason"]


# ─── regression on the free-text PROMPT shape (battery-04 q08) ────────────


class TestFreeTextRegressionRetainsPrior:
    """The live LLM PROMPT generator emits a free-text whole-file rewrite (a
    ``prompts/system_prompt.md`` rewrite → ``execute`` node), NOT the JSON
    payload shape. The canary gates it the same way: a free-text candidate
    that REGRESSES must not overwrite a prior good promotion. Distinct from
    ``test_promote.py::test_free_text_proposal_promotes_end_to_end``, which
    only proves the passing path lands on disk."""

    @pytest.mark.asyncio
    async def test_free_text_regressing_candidate_retains_prior(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _fake_settings(monkeypatch, tmp_path, min_score=0.8)
        gate = PromotionGate(canary=_canary(0.9))

        # v1: JSON shape, passing → promoted.
        first = await gate.promote(_json_proposal(["good guidance"]))
        assert first["promoted"] is True
        assert first["node"] == "execute"

        # v2: free-text rewrite (the live LLM shape), regressing canary.
        gate._canary = _canary(0.4)  # type: ignore[assignment]
        second = await gate.promote(
            _proposal(
                "# rewritten execute system prompt\nbe terse.\n",
                target_path="prompts/system_prompt.md",
            )
        )

        assert second["promoted"] is False
        assert second["reason"] == "canary below threshold"
        assert second["node"] == "execute"  # parsed from target_path
        # Prior promotion intact.
        assert gate.current_suffixes("execute") == ["good guidance"]


# ─── canary-recursion guard (commit 4272d02) ─────────────────────────────


class TestCanaryRecursionGuardHolds:
    """The promote→canary→evolve→promote recursion is broken at the harness
    layer (commit 4272d02): ``run_benchmark`` runs score-only
    (``no_evolution=True``). That guard has its own dedicated test
    (``test_eval/test_harness_no_evolution.py``); here we assert the property
    the O2 gate RELIES on at the promote layer: a canary whose OWN candidate
    is rejected never mutates state, so even a re-entrant call is a no-op."""

    @pytest.mark.asyncio
    async def test_rejected_candidate_writes_no_pointer(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _fake_settings(monkeypatch, tmp_path, min_score=0.8)
        gate = PromotionGate(canary=_canary(0.5))

        result = await gate.promote(_json_proposal(["rejected guidance"]))

        assert result["promoted"] is False
        # No pointer, no versioned artifact — a rejected candidate is a no-op,
        # so a (hypothetical) re-entrant canary finds nothing to recurse on.
        assert not (gate.prompts_dir / "current.json").exists()
        assert list(gate.prompts_dir.glob("execute.*.json")) == []
        assert gate.current_suffixes("execute") == []
