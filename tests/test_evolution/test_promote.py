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

    def test_free_text_rewrite_is_accepted_as_suffix(self) -> None:
        """Regression (battery-04 q08): the LLM PROMPT generator emits a whole
        prompt-file rewrite as FREE TEXT (not JSON) — e.g. a rewritten
        ``prompts/system_prompt.md`` (832 chars, confirmed live on mutation
        4c5c11d4). Before the fix this returned ``None``, so ``promote()``
        silently no-op'd and O2 never fired on a real run. The fix treats the
        entire block as ONE promoted suffix for the node implied by
        ``target_path``; the canary then gates whether it actually promotes."""
        proposal = {
            "mutation_type": MutationType.PROMPT,
            "mutated_content": "You are an AI agent. Generate artifacts from scratch.",
        }
        assert parse_prompt_payload(proposal) == (
            "execute",
            ["You are an AI agent. Generate artifacts from scratch."],
        )

    def test_free_text_uses_target_path_to_derive_node(self) -> None:
        """A free-text rewrite of a plan/reflect/verify prompt file maps to that
        node; a global/unknown prompt (system_prompt.md) defaults to execute."""
        plan = {
            "mutation_type": MutationType.PROMPT,
            "target_path": "prompts/plan_prompt.md",
            "mutated_content": "Plan more carefully.",
        }
        verify = {
            "mutation_type": MutationType.PROMPT,
            "target_path": "prompts/verify.md",
            "mutated_content": "Double-check outputs.",
        }
        system = {
            "mutation_type": MutationType.PROMPT,
            "target_path": "prompts/system_prompt.md",
            "mutated_content": "You are an AI agent.",
        }
        assert parse_prompt_payload(plan) == ("plan", ["Plan more carefully."])
        assert parse_prompt_payload(verify) == ("verify", ["Double-check outputs."])
        assert parse_prompt_payload(system) == ("execute", ["You are an AI agent."])

    def test_free_text_whitespace_is_stripped(self) -> None:
        proposal = {
            "mutation_type": MutationType.PROMPT,
            "mutated_content": "  padded guidance  ",
        }
        assert parse_prompt_payload(proposal) == ("execute", ["padded guidance"])

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

    @pytest.mark.asyncio
    async def test_free_text_proposal_promotes_end_to_end(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """End-to-end regression (battery-04 q08, mutation 4c5c11d4): the FREE-TEXT
        PROMPT mutation shape the live LLM generator emits — a whole prompt-file
        rewrite of ``prompts/system_prompt.md`` (832 chars, confirmed deployed in
        the ``mutations`` table) — flows through the *full* ``promote()`` path:
        parse → canary → versioned-artifact + ``current.json`` write → live
        read-back. Before the ``parse_prompt_payload`` fix this shape returned
        ``None`` at parse, so ``promote()`` no-op'd and the live pointer was never
        written (O2 never fired on a real run). This ties the real deployed-
        mutation shape to the write path that ``_prompt_proposal`` (JSON-shaped)
        alone does not cover."""
        _fake_settings(monkeypatch, tmp_path)
        gate = PromotionGate(canary=_canary(0.9))

        # The exact shape of mutation 4c5c11d4: PROMPT + target_path + free text.
        proposal = {
            "mutation_type": MutationType.PROMPT,
            "target_path": "prompts/system_prompt.md",
            "description": "address recurring failure patterns",
            "rationale": "generate stage artifacts from scratch, verify tool names",
            "model_used": "deepseek-v4-flash",
            "mutated_content": (
                "You are an AI agent. Generate stage artifacts from scratch; "
                "verify tool names before use; use code_executor/file_writer "
                "for intermediate outputs."
            ),
        }

        result = await gate.promote(proposal)

        assert result["promoted"] is True
        # system_prompt.md carries no plan/execute/reflect/verify token → execute.
        assert result["node"] == "execute"
        # Versioned artifact + pointer both written from the free-text mutation.
        version_file = gate.prompts_dir / result["version"]
        assert version_file.exists()
        assert "Generate stage artifacts from scratch" in version_file.read_text("utf-8")
        assert (gate.prompts_dir / "current.json").exists()
        # Live read-back (builder's read path) surfaces the promoted suffix verbatim.
        assert gate.current_suffixes("execute") == [
            "You are an AI agent. Generate stage artifacts from scratch; "
            "verify tool names before use; use code_executor/file_writer "
            "for intermediate outputs."
        ]


# ---------------------------------------------------------------------------
# VCS-tracked promotion mirror (Phase 5 G2)
# ---------------------------------------------------------------------------


class TestPromoteVcsTracked:
    """G2 — a passing promotion mirrors its artifact into a VCS-tracked tree and
    (when wired) commits it; a FAILING canary mirrors nothing and commits nothing.
    All behavior is opt-in: an absent ``tracked_prompts_dir`` ⇒ byte-identical
    legacy promotion."""

    @pytest.mark.asyncio
    async def test_tracked_dir_mirrors_artifact_on_promote(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """A passing promotion mirrors the versioned artifact into the tracked dir
        (no auto-commit unless a ``vcs_commit`` is wired)."""
        _fake_settings(monkeypatch, tmp_path)
        tracked = tmp_path / "tracked"
        gate = PromotionGate(canary=_canary(0.9), tracked_prompts_dir=tracked)

        result = await gate.promote(_prompt_proposal(["be strict"]))

        assert result["promoted"] is True
        assert tracked.exists()
        tracked_file = tracked / result["version"]
        assert tracked_file.exists()
        # The mirror matches the immutable versioned artifact byte-for-byte.
        versioned = gate.prompts_dir / result["version"]
        assert tracked_file.read_text("utf-8") == versioned.read_text("utf-8")
        # No vcs_commit wired ⇒ no commit hash in the result.
        assert "vcs_commit" not in result

    @pytest.mark.asyncio
    async def test_vcs_commit_invoked_on_promote(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """An injected ``vcs_commit`` is called with the tracked artifact path and
        its returned hash lands in the result."""
        _fake_settings(monkeypatch, tmp_path)
        tracked = tmp_path / "tracked"
        calls: list[tuple[str, str]] = []

        async def _commit(path: str, message: str) -> str:
            calls.append((path, message))
            return "deadbeef"

        gate = PromotionGate(
            canary=_canary(0.9), tracked_prompts_dir=tracked, vcs_commit=_commit
        )

        result = await gate.promote(_prompt_proposal(["be strict"]))

        assert result["promoted"] is True
        assert result["vcs_commit"] == "deadbeef"
        assert len(calls) == 1
        assert calls[0][0].endswith(result["version"])
        assert "promote execute" in calls[0][1]

    @pytest.mark.asyncio
    async def test_gate_fail_writes_no_mirror_and_never_commits(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """DoD: a FAILING canary flips no pointer, writes no tracked mirror, and
        never invokes the ``vcs_commit`` (a regression stays un-promoted AND
        un-committed)."""
        _fake_settings(monkeypatch, tmp_path)
        tracked = tmp_path / "tracked"
        called: list[tuple[str, str]] = []

        async def _commit(path: str, message: str) -> str:
            called.append((path, message))
            return "should-not-happen"

        gate = PromotionGate(
            canary=_canary(0.5), tracked_prompts_dir=tracked, vcs_commit=_commit
        )

        result = await gate.promote(_prompt_proposal(["x"]))

        assert result["promoted"] is False
        assert result["reason"] == "canary below threshold"
        assert not tracked.exists()
        assert called == []

    @pytest.mark.asyncio
    async def test_default_off_no_tracked_mirror(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """Default-off: the minimal settings fake lacks the G2 knobs, so with no
        explicit params promotion is byte-identical to legacy — no mirror, no
        commit key."""
        _fake_settings(monkeypatch, tmp_path)
        gate = PromotionGate(canary=_canary(0.9))

        result = await gate.promote(_prompt_proposal(["be strict"]))

        assert result["promoted"] is True
        assert gate.tracked_prompts_dir is None
        assert "vcs_commit" not in result

    @pytest.mark.asyncio
    async def test_vcs_commit_failure_does_not_block_promotion(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """A best-effort commit that raises must never unwind a promotion that
        already passed its canary (the live pointer is the source of truth)."""
        _fake_settings(monkeypatch, tmp_path)
        tracked = tmp_path / "tracked"

        async def _boom(_path: str, _message: str) -> str:
            raise RuntimeError("git not available")

        gate = PromotionGate(
            canary=_canary(0.9), tracked_prompts_dir=tracked, vcs_commit=_boom
        )

        result = await gate.promote(_prompt_proposal(["be strict"]))

        assert result["promoted"] is True
        assert gate.current_suffixes("execute") == ["be strict"]
        # Commit failed ⇒ no hash; the tracked FILE was still written before the call.
        assert "vcs_commit" not in result
        assert (tracked / result["version"]).exists()


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
