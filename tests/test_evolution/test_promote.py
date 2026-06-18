"""Phase 8 — evolution→live promotion gate (``src.evolution.promote``).

Covers the canary-gated promotion lifecycle for PROMPT mutations:
``parse_prompt_payload`` → ``promote`` (passing/failing/inconclusive/no-canary) →
versioned artifact + ``current.json`` pointer → ``rollback``. The gate is driven
with a fake async canary so the versioning/gating logic is deterministic and
free of a live LLM/DB. ``GoldenCanary`` is tested with an injected fake harness
so its override-application + mean-score wiring is verified without a full graph
run (the full-graph canary runs in Phase 10).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from src.evolution.promote import (
    GoldenCanary,
    PromotionGate,
    parse_prompt_payload,
)
from src.graph.enums import MutationType


# ---------------------------------------------------------------------------
# Settings fakes — PromotionGate reads get_settings() for evolved_handlers_dir +
# eval_canary_min_score. A single monkeypatch covers both the gate and the
# builder path (which lazy-imports the same get_settings).
# ---------------------------------------------------------------------------


def _fake_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    *,
    promote_on: bool = True,
    handlers_dir: Any | None = None,
    min_score: float = 0.8,
) -> SimpleNamespace:
    fake = SimpleNamespace(
        evolution=SimpleNamespace(
            evolution_promote_to_live=promote_on,
            evolved_handlers_dir=str(handlers_dir or (tmp_path / "evolved")),
        ),
        eval=SimpleNamespace(eval_canary_min_score=min_score),
    )
    monkeypatch.setattr("src.config.settings.get_settings", lambda: fake)
    return fake


@pytest.fixture(autouse=True)
def _clear_candidate_override() -> Iterator[None]:
    """Ensure no in-process canary override leaks between tests."""
    from src.graph.prompts.builder import clear_evolved_candidate

    clear_evolved_candidate()
    yield
    clear_evolved_candidate()


def _prompt_proposal(suffixes: list[str], node: str = "execute") -> dict[str, Any]:
    """A PROMPT mutation proposal carrying the canonical JSON payload."""
    return {
        "mutation_type": MutationType.PROMPT,
        "description": "address JSON mistakes",
        "rationale": "guide the model toward valid JSON",
        "model_used": "test-model",
        "mutated_content": json.dumps({"target_node": node, "suffixes": suffixes}),
    }


# ---------------------------------------------------------------------------
# parse_prompt_payload
# ---------------------------------------------------------------------------


class TestParsePromptPayload:
    def test_parses_valid_payload(self) -> None:
        parsed = parse_prompt_payload(_prompt_proposal(["a", "b"], "plan"))
        assert parsed == ("plan", ["a", "b"])

    def test_defaults_target_node_to_execute(self) -> None:
        proposal = {
            "mutation_type": MutationType.PROMPT,
            "mutated_content": json.dumps({"suffixes": ["only"]}),
        }
        assert parse_prompt_payload(proposal) == ("execute", ["only"])

    def test_rejects_non_prompt_mutation(self) -> None:
        proposal = {
            "mutation_type": MutationType.CODE,
            "mutated_content": json.dumps({"target_node": "execute", "suffixes": ["x"]}),
        }
        assert parse_prompt_payload(proposal) is None

    def test_rejects_non_json_content(self) -> None:
        proposal = {"mutation_type": MutationType.PROMPT, "mutated_content": "free text"}
        assert parse_prompt_payload(proposal) is None

    def test_rejects_missing_or_empty_suffixes(self) -> None:
        no_suffixes = {"mutation_type": MutationType.PROMPT, "mutated_content": "{}"}
        empty = {
            "mutation_type": MutationType.PROMPT,
            "mutated_content": json.dumps({"target_node": "execute", "suffixes": []}),
        }
        only_non_str = {
            "mutation_type": MutationType.PROMPT,
            "mutated_content": json.dumps({"suffixes": [1, 2]}),
        }
        assert parse_prompt_payload(no_suffixes) is None
        assert parse_prompt_payload(empty) is None
        assert parse_prompt_payload(only_non_str) is None


# ---------------------------------------------------------------------------
# promote / pointer / versioning
# ---------------------------------------------------------------------------


class TestPromote:
    @pytest.mark.asyncio
    async def test_passing_canary_promotes_and_writes_pointer(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _fake_settings(monkeypatch, tmp_path)
        gate = PromotionGate(canary=_canary(0.9))

        result = await gate.promote(_prompt_proposal(["be strict about JSON"]))

        assert result["promoted"] is True
        assert result["node"] == "execute"
        assert result["canary_score"] == 0.9
        # Versioned artifact + pointer both written.
        version_file = gate.prompts_dir / result["version"]
        assert version_file.exists()
        assert version_file.read_text(encoding="utf-8").count("be strict about JSON") == 1
        assert (gate.prompts_dir / "current.json").exists()
        # current_suffixes reads back the promoted suffixes.
        assert gate.current_suffixes("execute") == ["be strict about JSON"]

    @pytest.mark.asyncio
    async def test_failing_canary_does_not_promote(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _fake_settings(monkeypatch, tmp_path)
        gate = PromotionGate(canary=_canary(0.5))

        result = await gate.promote(_prompt_proposal(["x"]))

        assert result["promoted"] is False
        assert result["reason"] == "canary below threshold"
        assert result["canary_score"] == 0.5
        # No versioned artifact, no pointer.
        assert not (gate.prompts_dir / "current.json").exists()
        assert gate.current_suffixes("execute") == []

    @pytest.mark.asyncio
    async def test_inconclusive_canary_does_not_promote(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _fake_settings(monkeypatch, tmp_path)
        gate = PromotionGate(canary=_canary(None))

        result = await gate.promote(_prompt_proposal(["x"]))

        assert result["promoted"] is False
        assert result["reason"] == "canary inconclusive"
        assert gate.current_suffixes("execute") == []

    @pytest.mark.asyncio
    async def test_no_canary_does_not_promote(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _fake_settings(monkeypatch, tmp_path)
        gate = PromotionGate(canary=None)

        result = await gate.promote(_prompt_proposal(["x"]))

        assert result["promoted"] is False
        assert result["reason"] == "no canary wired"

    @pytest.mark.asyncio
    async def test_canary_error_does_not_promote(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _fake_settings(monkeypatch, tmp_path)

        async def boom(_node: str, _s: list[str]) -> float | None:
            raise RuntimeError("canary blew up")

        gate = PromotionGate(canary=boom)
        result = await gate.promote(_prompt_proposal(["x"]))

        assert result["promoted"] is False
        assert "canary error" in result["reason"]

    @pytest.mark.asyncio
    async def test_non_prompt_proposal_does_not_promote(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _fake_settings(monkeypatch, tmp_path)
        gate = PromotionGate(canary=_canary(0.99))

        result = await gate.promote(
            {"mutation_type": MutationType.CODE, "mutated_content": "print(1)"}
        )

        assert result["promoted"] is False
        assert "not a promotable PROMPT mutation" in result["reason"]

    @pytest.mark.asyncio
    async def test_two_distinct_promotions_append_history(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _fake_settings(monkeypatch, tmp_path)
        gate = PromotionGate(canary=_canary(0.9))

        await gate.promote(_prompt_proposal(["first"]))
        await gate.promote(_prompt_proposal(["second"]))

        pointer = json.loads((gate.prompts_dir / "current.json").read_text("utf-8"))
        assert len(pointer["execute"]["history"]) == 2
        # Active entry is the most recent promotion.
        assert gate.current_suffixes("execute") == ["second"]

    @pytest.mark.asyncio
    async def test_identical_re_promotion_dedups_history(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _fake_settings(monkeypatch, tmp_path)
        gate = PromotionGate(canary=_canary(0.9))

        await gate.promote(_prompt_proposal(["same"]))
        await gate.promote(_prompt_proposal(["same"]))

        pointer = json.loads((gate.prompts_dir / "current.json").read_text("utf-8"))
        assert len(pointer["execute"]["history"]) == 1


# ---------------------------------------------------------------------------
# rollback
# ---------------------------------------------------------------------------


class TestRollback:
    @pytest.mark.asyncio
    async def test_rollback_restores_previous_version(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _fake_settings(monkeypatch, tmp_path)
        gate = PromotionGate(canary=_canary(0.9))
        first = await gate.promote(_prompt_proposal(["first"]))
        await gate.promote(_prompt_proposal(["second"]))
        assert gate.current_suffixes("execute") == ["second"]

        rolled = gate.rollback("execute")

        assert rolled["rolled_back"] is True
        assert rolled["removed"] is not None
        assert rolled["restored"] == first["version"]
        assert gate.current_suffixes("execute") == ["first"]

    @pytest.mark.asyncio
    async def test_rollback_removes_node_when_no_prior_version(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _fake_settings(monkeypatch, tmp_path)
        gate = PromotionGate(canary=_canary(0.9))
        await gate.promote(_prompt_proposal(["only"]))

        rolled = gate.rollback("execute")

        assert rolled["rolled_back"] is True
        assert rolled["restored"] is None
        assert gate.current_suffixes("execute") == []

    def test_rollback_unknown_node_is_noop(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _fake_settings(monkeypatch, tmp_path)
        gate = PromotionGate(canary=_canary(0.9))
        rolled = gate.rollback("execute")
        assert rolled["rolled_back"] is False

    @pytest.mark.asyncio
    async def test_versioned_files_retained_after_rollback(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _fake_settings(monkeypatch, tmp_path)
        gate = PromotionGate(canary=_canary(0.9))
        first = await gate.promote(_prompt_proposal(["first"]))
        second = await gate.promote(_prompt_proposal(["second"]))

        gate.rollback("execute")

        # Both immutable version artifacts survive as an audit trail.
        assert (gate.prompts_dir / first["version"]).exists()
        assert (gate.prompts_dir / second["version"]).exists()


# ---------------------------------------------------------------------------
# GoldenCanary — override application + mean score (fake harness)
# ---------------------------------------------------------------------------


class _FakeHarness:
    """Stand-in BenchmarkHarness: records the active override + returns set scores."""

    def __init__(self, scores: list[float | None]) -> None:
        self._scores = list(scores)
        self.seen_overrides: list[list[str]] = []

    async def run_benchmark(self, goal: Any, spec: Any | None = None) -> Any:
        from src.eval.models import BenchmarkResult
        from src.graph.prompts.builder import evolved_suffixes_for_node

        self.seen_overrides.append(evolved_suffixes_for_node("execute"))
        score = self._scores.pop(0)
        return BenchmarkResult(
            goal_name=goal.name,
            category=goal.category,
            success=True,
            total_latency_ms=1,
            total_tokens=0,
            total_cost_usd=0.0,
            iterations=1,
            correctness_score=score,
        )


class TestGoldenCanary:
    @pytest.mark.asyncio
    async def test_returns_mean_of_scores(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _fake_settings(monkeypatch, tmp_path, promote_on=True)
        harness = _FakeHarness([0.8, 0.6])
        canary = GoldenCanary(None, None, None, harness=harness, goal_ids=["battery04_q01"])

        # battery04_q01 is a single goal → mean of its one score.
        score = await canary.score("execute", ["candidate guidance"])

        assert score == 0.8
        # The candidate override was visible to the harness during the run.
        assert harness.seen_overrides == [["candidate guidance"]]

    @pytest.mark.asyncio
    async def test_returns_none_when_no_score_signal(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _fake_settings(monkeypatch, tmp_path, promote_on=True)
        harness = _FakeHarness([None])
        canary = GoldenCanary(None, None, None, harness=harness, goal_ids=["battery04_q01"])

        score = await canary.score("execute", ["x"])

        assert score is None

    @pytest.mark.asyncio
    async def test_override_cleared_after_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _fake_settings(monkeypatch, tmp_path, promote_on=True)
        canary = GoldenCanary(
            None, None, None, harness=_FakeHarness([0.9]), goal_ids=["battery04_q01"]
        )
        from src.graph.prompts.builder import evolved_suffixes_for_node

        await canary.score("execute", ["temp"])

        # After the canary, no candidate override leaks (promotion is opt-in
        # pointer-based, not override-based).
        assert evolved_suffixes_for_node("execute") == []


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _canary(score: float | None) -> Any:
    """Build a fake async canary returning a fixed score."""

    async def _fn(_node: str, _suffixes: list[str]) -> float | None:
        return score

    return _fn
