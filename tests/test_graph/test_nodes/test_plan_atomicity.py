"""Plan node per-step atomicity (Feature C) — ``plan_atomicity_enforce`` parity.

``plan_quality`` is ALWAYS computed + attached as advisory telemetry. The opt-in
gate ``plan_atomicity_enforce`` (default off) additionally decomposes a
``too_coarse`` step (>=2 conjunction/clause markers) into atomic sub-steps via a
bounded, deterministic heuristic (zero LLM cost). The split is guarded by
``atomicity_replan_done`` so a reflect→plan loop can't re-split the same step
forever — which also makes it idempotent on a single pass.

These are pure-logic tests of the plan node's atomicity path (the pure helpers
+ the gated ``plan_node`` entry with a heuristic plan and no gateway), so the
LLM is never invoked.
"""

from __future__ import annotations


import pytest

from src.config import get_settings
from src.graph.enums import Phase, Strategy, TaskComplexity
from src.graph.models import Goal, PlanStep
from src.graph.nodes.plan import (
    _COARSE_SPLIT_CAP,
    _split_coarse_step,
    _validate_step_atomicity,
    plan_node,
)


# ─── pure helpers ───────────────────────────────────────────────────────────


def _step(desc: str, expected: str = "a result") -> PlanStep:
    return PlanStep(description=desc, expected_output=expected)


class TestValidateStepAtomicity:
    def test_should_flag_too_coarse_when_two_or_more_conjunctions(self) -> None:
        s = _step("fetch the data and clean it then write to disk")
        q = _validate_step_atomicity([s])
        assert q.per_step[0].flag == "too_coarse"
        assert q.too_coarse_count == 1
        assert q.atomic is False

    def test_should_flag_atomic_when_single_well_scoped_action(self) -> None:
        s = _step("fetch the orders.csv file")
        q = _validate_step_atomicity([s])
        assert q.per_step[0].flag == "atomic"
        assert q.atomic is True
        assert q.too_coarse_count == 0

    def test_should_flag_too_fine_when_underworded_and_no_expected_output(self) -> None:
        # Fewer than 3 words and NO expected_output → under-specified.
        s = PlanStep(description="do it", expected_output="")
        q = _validate_step_atomicity([s])
        assert q.per_step[0].flag == "too_fine"
        assert q.too_fine_count == 1
        assert q.atomic is False

    def test_should_treat_semicolons_as_clause_markers(self) -> None:
        s = _step("parse; validate; persist")
        q = _validate_step_atomicity([s])
        assert q.per_step[0].flag == "too_coarse"

    def test_should_count_flags_across_multiple_steps(self) -> None:
        steps = [
            _step("fetch the data and clean it then store"),  # too_coarse
            _step("go"),  # too_fine (1 word, no expected → but expected set here)
        ]
        # second has expected_output so it's atomic, not too_fine
        q = _validate_step_atomicity(steps)
        assert q.too_coarse_count == 1
        assert q.atomic is False


class TestSplitCoarseStep:
    def test_should_decompose_coarse_step_into_clauses(self) -> None:
        s = _step("fetch the data and clean it then write to disk", expected="csv")
        parts = _split_coarse_step(s)
        assert len(parts) == 3
        assert [p.description for p in parts] == [
            "fetch the data", "clean it", "write to disk"
        ]
        # Each sub-step inherits the parent's tool_name + expected_output context.
        for p in parts:
            assert p.expected_output == "csv"
            assert p.tool_name == s.tool_name

    def test_should_return_unchanged_when_single_clause(self) -> None:
        # An already-atomic step has <2 clauses → returned as a singleton (this
        # is what makes the split idempotent: re-splitting a sub-step no-ops).
        s = _step("fetch the data")
        parts = _split_coarse_step(s)
        assert parts == [s]

    def test_should_cap_clause_count_at_split_cap(self) -> None:
        # Build a step with more clauses than the cap → only the cap survives.
        clauses = " and ".join(f"step{i}" for i in range(_COARSE_SPLIT_CAP + 4))
        s = _step(clauses)
        parts = _split_coarse_step(s)
        assert len(parts) == _COARSE_SPLIT_CAP

    def test_each_substep_gets_fresh_id(self) -> None:
        s = _step("do a and do b")
        parts = _split_coarse_step(s)
        ids = {p.id for p in parts}
        assert len(ids) == len(parts)
        assert s.id not in ids  # new ids, not the parent's


# ─── plan_node gated entry ──────────────────────────────────────────────────


def _state(goal_text: str = "build the report") -> dict[str, object]:
    goal = Goal(text=goal_text, complexity=TaskComplexity.SIMPLE)
    return {
        "current_goal": goal,
        "strategy": Strategy.REACT,
        "iteration_count": 0,
        "max_iterations": 60,
        "messages": [],
    }


class TestPlanNodeAtomicityGate:
    async def test_should_attach_plan_quality_even_when_gate_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # plan_quality is ALWAYS attached as advisory telemetry — gate off only
        # suppresses the split, not the computation.
        monkeypatch.setattr(get_settings().agent, "plan_atomicity_enforce", False)
        # No gateway → heuristic plan; coarse steps pass through unchanged.
        out = await plan_node(_state(), gateway=None)  # type: ignore[arg-type]
        assert "plan_quality" in out
        assert out["phase"] == Phase.RETRIEVE_MEMORY

    async def test_should_not_split_coarse_step_when_gate_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(get_settings().agent, "plan_atomicity_enforce", False)
        # Inject a single coarse heuristic step by overriding _generate_plan.
        coarse = _step("fetch X and clean it then write Y")
        import src.graph.nodes.plan as plan_mod

        monkeypatch.setattr(plan_mod, "_generate_plan", lambda *a, **k: [coarse])
        out = await plan_node(_state(), gateway=None)  # type: ignore[arg-type]
        assert len(out["plan_steps"]) == 1  # unchanged — gate off
        assert out["plan_quality"]["too_coarse_count"] == 1
        assert "atomicity_replan_done" not in out  # not attempted

    async def test_should_split_coarse_step_when_gate_on(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(get_settings().agent, "plan_atomicity_enforce", True)
        coarse = _step("fetch X and clean it then write Y")
        import src.graph.nodes.plan as plan_mod

        monkeypatch.setattr(plan_mod, "_generate_plan", lambda *a, **k: [coarse])
        out = await plan_node(_state(), gateway=None)  # type: ignore[arg-type]
        assert len(out["plan_steps"]) == 3  # decomposed
        assert out["plan_quality"]["too_coarse_count"] == 0
        assert out["atomicity_replan_done"] is True

    async def test_should_be_idempotent_on_rerun(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Re-running plan on an already-split plan (atomicity_replan_done=True)
        # does NOT re-split — the loop guard holds.
        monkeypatch.setattr(get_settings().agent, "plan_atomicity_enforce", True)
        coarse = _step("fetch X and clean it then write Y")
        import src.graph.nodes.plan as plan_mod

        monkeypatch.setattr(plan_mod, "_generate_plan", lambda *a, **k: [coarse])
        state = _state()
        first = await plan_node(state, gateway=None)  # type: ignore[arg-type]
        assert len(first["plan_steps"]) == 3

        # Second pass carries atomicity_replan_done=True from the first result.
        state2 = {**_state(), "atomicity_replan_done": True}
        monkeypatch.setattr(plan_mod, "_generate_plan",
                            lambda *a, **k: first["plan_steps"])
        second = await plan_node(state2, gateway=None)  # type: ignore[arg-type]
        # Same count — no further splitting (idempotent).
        assert len(second["plan_steps"]) == 3

    async def test_should_leave_atomic_plan_unchanged_when_gate_on(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(get_settings().agent, "plan_atomicity_enforce", True)
        atomic = _step("fetch the orders.csv file")
        import src.graph.nodes.plan as plan_mod

        monkeypatch.setattr(plan_mod, "_generate_plan", lambda *a, **k: [atomic])
        out = await plan_node(_state(), gateway=None)  # type: ignore[arg-type]
        assert len(out["plan_steps"]) == 1
        assert out["plan_quality"]["atomic"] is True
