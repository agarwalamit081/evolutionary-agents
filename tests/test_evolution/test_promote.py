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

import asyncio
import json
import time
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from src.evolution.promote import (
    GoldenCanary,
    PromotionGate,
    classify_payload,
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
    canary_timeout_s: float = 180.0,
    canary_goals: list[str] | None = None,
) -> SimpleNamespace:
    """Minimal settings fake for the promotion gate + canary.

    Patches get_settings on BOTH ``src.config.settings`` (what
    ``PromotionGate`` lazy-imports) AND ``src.config`` (the package re-export
    that ``GoldenCanary.score`` / ``BenchmarkHarness.run_benchmark`` import).
    Without both bindings the canary silently reads the REAL 180s default for
    ``promotion_canary_timeout_s``, so a time-box regression test can't force an
    abandon. With both, the canary's inline budget is genuinely controllable.
    """
    fake = SimpleNamespace(
        evolution=SimpleNamespace(
            evolution_promote_to_live=promote_on,
            evolved_handlers_dir=str(handlers_dir or (tmp_path / "evolved")),
            promotion_canary_timeout_s=canary_timeout_s,
            promotion_canary_goals=(
                list(canary_goals) if canary_goals is not None else ["battery04_q01"]
            ),
        ),
        eval=SimpleNamespace(eval_canary_min_score=min_score),
    )
    monkeypatch.setattr("src.config.settings.get_settings", lambda: fake)
    monkeypatch.setattr("src.config.get_settings", lambda: fake)
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


class _ScriptedHarness:
    """Stand-in BenchmarkHarness whose ``run_benchmark`` runs a scripted list of
    behaviors, one per goal. A behavior is a ``(kind, value)`` tuple:

    * ``("score", float | None)`` — return immediately with that ``correctness_score``.
    * ``("sleep", seconds)`` — ``await asyncio.sleep(seconds)``; under a canary
      budget ``asyncio.timeout`` cancels the call mid-sleep, modeling a
      non-converging goal. If it survives (no budget), it scores ``None``.

    Records the call count so a time-boxed abandonment + a sibling goal's success
    can be asserted together. Does NOT call the real BenchmarkHarness (no
    clean_run_subdir / gateway) so it stays hermetic.
    """

    def __init__(self, behaviors: list[tuple[str, Any]]) -> None:
        self._behaviors = list(behaviors)
        self.call_count = 0

    async def run_benchmark(self, goal: Any, spec: Any | None = None) -> Any:
        from src.eval.models import BenchmarkResult

        self.call_count += 1
        kind, value = self._behaviors.pop(0)
        if kind == "sleep":
            await asyncio.sleep(float(value))  # cancelled by asyncio.timeout on overrun
            value = None  # reached only when the budget is disabled (escaped the cancel)
        return BenchmarkResult(
            goal_name=goal.name,
            category=goal.category,
            success=True,
            total_latency_ms=1,
            total_tokens=0,
            total_cost_usd=0.0,
            iterations=1,
            correctness_score=value,
        )


class _RecordingGateway:
    """Fake gateway recording every ``set_run_id`` so the canary's run_id
    save/set/restore can be asserted. Mirrors the real ``LLMGateway.set_run_id``
    contract (sets ``self._run_id``) without a live client / cost tracker."""

    def __init__(self, initial_run_id: str | None = "api-live-run") -> None:
        self._run_id = initial_run_id
        self.set_calls: list[str | None] = []

    def set_run_id(self, run_id: str | None) -> None:
        self.set_calls.append(run_id)
        self._run_id = run_id


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
# Canary time-box + gateway run_id isolation (regression: adhoc-eval-proof-1)
# ---------------------------------------------------------------------------


class TestCanaryDoesNotBlockOrContaminate:
    """Regression (run ``adhoc-eval-proof-1``, 2026-06-27): with
    ``EVOLUTION_PROMOTE_TO_LIVE`` on, ``run_cycle`` scores a deployed PROMPT
    mutation by running a full golden goal INLINE inside the live run's evolve
    node, ON THE LIVE GATEWAY. Two design defects surfaced:

    1. A non-converging goal (q01 tz-shift) blocked ``run_cycle`` → ``evolve`` →
       the live run until the worker wall-clock (1800s) killed it — a 30-min
       held-hostage run.
    2. The canary's battery-goal LLM calls attributed to the LIVE run_id (shared
       gateway, run_id not scoped) — cost contamination.

    The fix: ``GoldenCanary.score`` time-boxes each per-goal ``run_benchmark``
    (``promotion_canary_timeout_s``) and ``BenchmarkHarness.run_benchmark``
    scopes the gateway run_id to its bench thread_id. These tests pin both.
    """

    @pytest.mark.asyncio
    async def test_time_box_abandons_non_converging_goal_and_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """A 0.05s budget vs a 3s 'goal' (a non-converging run): the canary must
        abandon it within the budget and return inconclusive (``None``) — NOT
        park the caller for the full 3s. The elapsed guard is the real
        regression discriminator: a broken/absent budget would take ~3s."""
        _fake_settings(monkeypatch, tmp_path, canary_timeout_s=0.05)
        harness = _ScriptedHarness([("sleep", 3.0)])
        canary = GoldenCanary(
            None, None, None, harness=harness, goal_ids=["battery04_q01"]
        )

        start = time.monotonic()
        score = await canary.score("execute", ["candidate"])
        elapsed = time.monotonic() - start

        assert score is None  # abandoned → no score → no promotion
        assert harness.call_count == 1  # the goal WAS attempted, then abandoned
        # Abandoned within the budget, NOT parked for the 3s sleep.
        assert elapsed < 1.0
        # The candidate override is cleared even on the abandon path (finally ran).
        from src.graph.prompts.builder import evolved_suffixes_for_node

        assert evolved_suffixes_for_node("execute") == []

    @pytest.mark.asyncio
    async def test_one_goal_timeout_another_scores_returns_mean_of_scored(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """Two goals: q01 sleeps past the budget (abandoned, no score); q02
        completes instantly with 0.8. The suite must NOT abort on q01's abandon —
        it scores q02 and returns the mean of the SCORED goals only."""
        _fake_settings(monkeypatch, tmp_path, canary_timeout_s=0.05)
        harness = _ScriptedHarness([("sleep", 3.0), ("score", 0.8)])
        canary = GoldenCanary(
            None,
            None,
            None,
            harness=harness,
            goal_ids=["battery04_q01", "battery04_q02"],
        )

        score = await canary.score("execute", ["candidate"])

        assert score == 0.8  # mean of the one scored goal
        assert harness.call_count == 2  # both attempted; q01's abandon didn't abort q02

    @pytest.mark.asyncio
    async def test_budget_disabled_runs_unbounded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """``promotion_canary_timeout_s <= 0`` is the offline escape hatch: no
        time-box, the goal runs to completion normally."""
        _fake_settings(monkeypatch, tmp_path, canary_timeout_s=0.0)
        harness = _ScriptedHarness([("score", 0.7)])
        canary = GoldenCanary(
            None, None, None, harness=harness, goal_ids=["battery04_q01"]
        )

        score = await canary.score("execute", ["candidate"])

        assert score == 0.7

    @pytest.mark.asyncio
    async def test_run_benchmark_scopes_gateway_run_id_to_bench_thread(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """The canary reuses the live run's gateway; its battery-goal cost must
        bill to the canary's own ``bench-`` thread_id, NOT the live run that
        parked for evolution. ``run_benchmark`` saves the live run_id, scopes the
        gateway to the bench thread_id for the duration of ``ainvoke``, and
        restores the live run_id in ``finally``."""
        from src.eval.harness import BenchmarkHarness
        from src.eval.models import BenchmarkGoal

        _fake_settings(monkeypatch, tmp_path)

        # The live run parked its gateway at this run_id; the canary must NOT bill to it.
        gateway = _RecordingGateway(initial_run_id="api-adhoc-eval-proof-1")
        observed: dict[str, Any] = {}

        async def fake_ainvoke(_state: dict[str, Any]) -> dict[str, Any]:
            # Captured INSIDE the graph run — the run_id billed to these calls.
            observed["run_id_during_ainvoke"] = gateway._run_id
            return {}  # _extract_result tolerates an empty state

        compiled = SimpleNamespace(ainvoke=fake_ainvoke)

        def fake_compile_task_graph(**_kwargs: Any) -> Any:
            return compiled

        monkeypatch.setattr("src.graph.task_graph.compile_task_graph", fake_compile_task_graph)
        monkeypatch.setattr("src.graph.factory.initial_state", lambda **_kw: {})
        # clean_run_subdir reads agent.results_root via the resolver; neutralize it so
        # the test does not need a full agent settings object.
        monkeypatch.setattr("src.tools._paths.clean_run_subdir", lambda _rid: False)

        # Deliberate fake substitution: gateway is a recording stand-in and the
        # tools/registry are unused (the compiled graph is monkeypatched out).
        harness = BenchmarkHarness(gateway, tools=None, sub_agent_registry=None)  # type: ignore[arg-type]
        goal = BenchmarkGoal(
            name="t-goal", description="unit", goal_text="g", category="complex"
        )

        result = await harness.run_benchmark(goal)

        # The run came back (no exception); the bench thread_id was used mid-run.
        assert result.goal_name == "t-goal"
        assert observed["run_id_during_ainvoke"].startswith("bench-t-goal-")
        assert observed["run_id_during_ainvoke"] != "api-adhoc-eval-proof-1"
        # After the run the prior LIVE run_id was restored (the parked run resumes
        # billing to itself; a cancelled/errored canary never leaks the bench id).
        assert gateway._run_id == "api-adhoc-eval-proof-1"
        # set_run_id sequence: [scope to bench thread_id, restore live run_id].
        first_call = gateway.set_calls[0]
        assert first_call is not None
        assert first_call.startswith("bench-t-goal-")
        assert gateway.set_calls[-1] == "api-adhoc-eval-proof-1"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _canary(score: float | None) -> Any:
    """Build a fake async canary returning a fixed score."""

    async def _fn(_node: str, _suffixes: list[str]) -> float | None:
        return score

    return _fn


# ---------------------------------------------------------------------------
# classify_payload — the JSON-vs-free-text SHAPE heuristic promoted from
# scripts/inspect_mutation.py (now exposed via `main.py --inspect-mutation`).
# Diagnostic companion to parse_prompt_payload: reports whether the gate will
# PARSE a stored mutations row. The free-text label is the regression anchor for
# the stale-framing fix — free-text IS promotable now (whole-file rewrite → one
# promoted suffix), NOT "gate cannot promote".
# ---------------------------------------------------------------------------


class TestClassifyPayload:
    """Shape labels must track the CURRENT parser behavior, not the old script."""

    def test_prompt_json_object_parses(self) -> None:
        proposal = {
            "mutation_type": MutationType.PROMPT,
            "mutated_content": json.dumps(
                {"target_node": "execute", "suffixes": ["Be terse."]}
            ),
        }
        label = classify_payload(proposal)
        assert "json-object" in label
        assert "target_node" in label  # the structured shape the parser pulls

    def test_prompt_json_array_is_unparsable(self) -> None:
        # A JSON array is the ONE PROMPT shape the gate drops (not a {…} dict).
        proposal = {
            "mutation_type": MutationType.PROMPT,
            "mutated_content": json.dumps(["a", "b"]),
        }
        label = classify_payload(proposal)
        assert "json-array" in label
        assert "None" in label  # parse_prompt_payload → None

    def test_prompt_free_text_is_promotable_not_dropped(self) -> None:
        # Regression anchor: free-text whole-file rewrite IS promotable now (the
        # parser treats the whole block as one suffix). The old script's "gate
        # cannot promote" label was stale after the free-text fix.
        proposal = {
            "mutation_type": MutationType.PROMPT,
            "mutated_content": "You are a meticulous execute node. Always emit JSON.",
        }
        label = classify_payload(proposal)
        assert "free-text" in label
        assert "one promoted suffix" in label
        assert "cannot promote" not in label  # the stale framing is gone

    def test_prompt_markup_script_is_free_text_variant(self) -> None:
        proposal = {
            "mutation_type": MutationType.PROMPT,
            "mutated_content": "<system>\nYou are execute.\n</system>",
        }
        label = classify_payload(proposal)
        assert "markup-script" in label
        assert "one promoted suffix" in label  # still a promotable free-text shape

    def test_code_payload_is_module_rewrite(self) -> None:
        proposal = {
            "mutation_type": MutationType.CODE,
            "mutated_content": "def f():\n    return 42\n",
            "target_path": "graph/nodes/execute.py",
        }
        label = classify_payload(proposal)
        assert "code" in label and "module rewrite" in label
        assert "shadow target_path" in label  # parse_code_payload target

    def test_unhandled_type_has_no_parser(self) -> None:
        # TOOL/WORKFLOW/MEMORY/etc. reach live via other paths (DB registry,
        # shadow repo) — the promotion gate has no parser for them.
        proposal = {
            "mutation_type": MutationType.TOOL,
            "mutated_content": "register_tool(name='x', ...)",
        }
        label = classify_payload(proposal)
        assert "no promotion parser" in label
        assert "tool" in label

    def test_empty_content_is_reported_empty_not_free_text(self) -> None:
        # An empty mutation is not promotable regardless of type — classify must
        # not mislabel it "free-text → one promoted suffix".
        for blank in ("", "   ", "\n\t"):
            proposal = {"mutation_type": MutationType.PROMPT, "mutated_content": blank}
            assert "empty" in classify_payload(proposal)

    def test_non_str_content_does_not_crash(self) -> None:
        # Defensive: a malformed row with non-string mutated_content classifies
        # as empty rather than raising (the proposal comes from a DB row).
        proposal = {"mutation_type": MutationType.PROMPT, "mutated_content": None}
        assert "empty" in classify_payload(proposal)

    def test_label_matches_actual_parser_outcome_for_each_shape(self) -> None:
        # Cross-check: the label's promotability verdict agrees with what the real
        # parser returns for the SAME proposal (json-object parses; array does not).
        good = {
            "mutation_type": MutationType.PROMPT,
            "mutated_content": json.dumps({"target_node": "plan", "suffixes": ["x"]}),
        }
        bad = {
            "mutation_type": MutationType.PROMPT,
            "mutated_content": json.dumps(["x"]),
        }
        assert parse_prompt_payload(good) is not None
        assert "None" not in classify_payload(good)
        assert parse_prompt_payload(bad) is None
        assert "None" in classify_payload(bad)


# ---------------------------------------------------------------------------
# promotion_canary_goals — configurable canary benchmark (channel-B unblock)
# ---------------------------------------------------------------------------


class TestPromotionCanaryGoalsConfig:
    """``EvolutionSettings.promotion_canary_goals`` env parsing.

    The canary benchmark was hardcoded to ``battery04_q01``; under a stack where
    q01 non-converges (loops past the inline budget) the canary was always
    inconclusive → channel-B prompt promotion never fired. The field makes the
    benchmark operator-configurable (CSV/JSON/list) so a CONVERGING goal can be
    chosen. Default is unchanged (``battery04_q01``).
    """

    def test_default_is_q01(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.config.settings import EvolutionSettings

        monkeypatch.delenv("PROMOTION_CANARY_GOALS", raising=False)
        assert EvolutionSettings().promotion_canary_goals == ["battery04_q01"]

    def test_csv_env_splits_and_strips(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.config.settings import EvolutionSettings

        monkeypatch.setenv("PROMOTION_CANARY_GOALS", "a, b ,c")
        assert EvolutionSettings().promotion_canary_goals == ["a", "b", "c"]

    def test_single_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.config.settings import EvolutionSettings

        monkeypatch.setenv("PROMOTION_CANARY_GOALS", "probe_analytics_recall")
        assert EvolutionSettings().promotion_canary_goals == ["probe_analytics_recall"]

    def test_json_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.config.settings import EvolutionSettings

        monkeypatch.setenv("PROMOTION_CANARY_GOALS", '["x", "y"]')
        assert EvolutionSettings().promotion_canary_goals == ["x", "y"]

    def test_blank_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.config.settings import EvolutionSettings

        monkeypatch.setenv("PROMOTION_CANARY_GOALS", "")
        assert EvolutionSettings().promotion_canary_goals == ["battery04_q01"]


class TestGoldenCanaryGoalSelection:
    """``GoldenCanary`` resolves its suite from settings (``goal_ids=None``) or
    the explicit override; unresolvable goals are skipped. A dummy harness avoids
    constructing the real ``BenchmarkHarness`` (suite selection is independent of
    the gateway/tools/registry)."""

    def test_default_suite_is_q01(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _fake_settings(monkeypatch, tmp_path)
        gc = GoldenCanary(None, None, None, harness=object())
        assert [s.spec_id for s in gc._suite] == ["battery04_q01"]

    def test_settings_goals_override_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _fake_settings(monkeypatch, tmp_path, canary_goals=["probe_analytics_recall"])
        gc = GoldenCanary(None, None, None, harness=object())
        assert [s.spec_id for s in gc._suite] == ["probe_analytics_recall"]

    def test_explicit_goal_ids_override_settings(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _fake_settings(monkeypatch, tmp_path, canary_goals=["probe_analytics_recall"])
        gc = GoldenCanary(
            None, None, None, goal_ids=["battery04_q05"], harness=object()
        )
        assert [s.spec_id for s in gc._suite] == ["battery04_q05"]

    def test_unresolvable_goal_skipped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _fake_settings(monkeypatch, tmp_path, canary_goals=["does_not_exist"])
        gc = GoldenCanary(None, None, None, harness=object())
        assert gc._suite == []



# ---------------------------------------------------------------------------
# Multi-goal canary gate (Track-1): GoldenCanary.score returns 0.0 when ANY
# scored goal has passed=False — the q04 multi-goal collapse the mean-only
# aggregation averaged away (1.0→0.167 while the single-goal canary promoted 15×).
# ---------------------------------------------------------------------------


class _PassedHarness:
    """Stand-in BenchmarkHarness returning BenchmarkResults with controlled
    ``correctness_score`` AND ``passed`` — so the multi-goal gate (any
    ``passed is False`` → 0.0) can be exercised independently of the score mean."""

    def __init__(self, results: list[tuple[float | None, bool | None]]) -> None:
        self._results = list(results)
        self.call_count = 0

    async def run_benchmark(self, goal: Any, spec: Any | None = None) -> Any:
        from src.eval.models import BenchmarkResult

        self.call_count += 1
        score, passed = self._results.pop(0)
        return BenchmarkResult(
            goal_name=goal.name,
            category=goal.category,
            success=True,
            total_latency_ms=1,
            total_tokens=0,
            total_cost_usd=0.0,
            iterations=1,
            correctness_score=score,
            passed=passed,
        )


class TestGoldenCanaryMultiGoalGate:
    """Track-1: a single goal whose strict ``passed`` is False fails the whole
    canary (returns 0.0) even when the mean ``correctness_score`` is high."""

    @pytest.mark.asyncio
    async def test_one_failed_goal_zeros_score_even_when_mean_is_high(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """The exact q04 shape: goal A perfect (1.0, passed=True); goal B scores
        0.875 (7/8 checks) but passed=False (one sub-goal collapsed). Mean 0.9375
        would clear min_score 0.8 and promote under the OLD mean-only aggregation;
        the multi-goal gate must return 0.0."""
        _fake_settings(monkeypatch, tmp_path, promote_on=True, min_score=0.8)
        harness = _PassedHarness([(1.0, True), (0.875, False)])
        canary = GoldenCanary(
            None,
            None,
            None,
            harness=harness,
            goal_ids=["battery04_q01", "battery04_q02"],
        )
        score = await canary.score("execute", ["candidate"])
        assert score == 0.0
        assert harness.call_count == 2  # both goals ran

    @pytest.mark.asyncio
    async def test_all_passing_returns_mean(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _fake_settings(monkeypatch, tmp_path, promote_on=True, min_score=0.8)
        harness = _PassedHarness([(1.0, True), (0.8, True)])
        canary = GoldenCanary(
            None, None, None, harness=harness,
            goal_ids=["battery04_q01", "battery04_q02"],
        )
        score = await canary.score("execute", ["candidate"])
        assert score == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_none_passed_is_not_treated_as_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """A goal whose checks never ran (passed=None) carries no pass/fail signal
        — it must NOT trip the gate (only ``passed is False`` does). Its score
        still counts toward the mean."""
        _fake_settings(monkeypatch, tmp_path, promote_on=True, min_score=0.8)
        harness = _PassedHarness([(1.0, None), (0.8, True)])
        canary = GoldenCanary(
            None, None, None, harness=harness,
            goal_ids=["battery04_q01", "battery04_q02"],
        )
        score = await canary.score("execute", ["candidate"])
        assert score == pytest.approx(0.9)
