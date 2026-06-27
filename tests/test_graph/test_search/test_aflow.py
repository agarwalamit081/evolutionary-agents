"""AFlow workflow-topology optimizer — deterministic unit suite (Phase 5 G3b).

The optimizer is fully DI (gateway + store + run_fn + settings + curve_verdict +
cost_fn all injected), so every branch is exercised without an LLM, a DB, or a
live agent run. The runtime hook (``aflow_techniques_for``) shares a process-global
in-process candidate dict with the optimizer; an autouse fixture clears it before
and after every test (mirrors ``test_builder_evolved``'s override teardown).
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.graph.enums import TaskComplexity
from src.graph.prompts.builder import (
    aflow_candidate_for,
    aflow_techniques_for,
    clear_aflow_candidate,
    select_techniques_for_node,
    set_aflow_candidate,
)
from src.graph.prompts.technique_selector import TechniqueSelector
from src.graph.search.aflow import (
    AFlowOptimizer,
    AflowPolicyStore,
    resolve_policy,
    technique_names_for_node,
)

# ── Constants ─────────────────────────────────────────────────────────────────
NODE = "execute"
CATEGORY = "code"
_AVAILABLE = technique_names_for_node(NODE)
NAMES_A = _AVAILABLE[:2]  # distinct candidate A (improves under run_fn)
NAMES_B = _AVAILABLE[2:4]  # distinct candidate B (worse under run_fn)


def _seed(spec_id: str = "s1") -> SimpleNamespace:
    """An opaque seed spec — the optimizer only forwards it to run_fn."""
    return SimpleNamespace(spec_id=spec_id, goal_text="write a python sorting fn", max_iterations=5)


def _fake_settings(
    *,
    enabled: bool = True,
    max_candidates: int = 2,
    improvement_margin: float = 0.0,
    max_cost_usd: float = 0.0,
    preflight_curve_clear: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        enabled=enabled,
        max_candidates=max_candidates,
        improvement_margin=improvement_margin,
        max_cost_usd=max_cost_usd,
        preflight_curve_clear=preflight_curve_clear,
    )


def _gateway(content: str) -> MagicMock:
    """A mock gateway whose one ``acompletion`` call returns ``content`` (JSON)."""
    from src.llm.models import LLMResponse

    gateway = MagicMock()
    gateway.acompletion = AsyncMock(
        return_value=LLMResponse(
            content=content,
            model="glm-4.7",
            provider="zai",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            cost_usd=0.0,
        )
    )
    return gateway


def _candidates_json(candidates: list[list[str]]) -> str:
    import json

    return json.dumps({"candidates": candidates})


async def _clear_verdict() -> dict[str, Any]:
    """A non-regressed, conclusive C1 verdict so the search proceeds."""
    return {"regressed": False, "inconclusive": False}


@pytest.fixture(autouse=True)
def _clear_aflow() -> Iterator[None]:
    """The candidate override is process-global — clear it per test."""
    clear_aflow_candidate()
    yield
    clear_aflow_candidate()


@pytest.fixture(autouse=True)
def _aflow_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force ``aflow.enabled=False`` for the builder hook's pointer path.

    Makes the byte-identical / fall-through assertions deterministic regardless of
    the live .env (the hook reads ``get_settings().aflow.enabled`` before the pointer).
    """
    fake_settings = SimpleNamespace(aflow=SimpleNamespace(enabled=False))
    # ``get_settings`` is imported lazily INSIDE ``aflow_techniques_for`` from
    # ``src.config.settings``, so patch it there (it is not a builder attribute).
    monkeypatch.setattr(
        "src.config.settings.get_settings", lambda: fake_settings
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestAFlowOptimizer:
    async def test_applies_accepted_modification_and_persists(self, tmp_path: Any) -> None:
        """A candidate that beats baseline+margin is persisted + resolvable by the hook."""

        async def run_fn(_spec: Any) -> float | None:
            active = aflow_candidate_for(NODE, CATEGORY)
            if active is None:
                return 0.5  # baseline (override OFF)
            return 0.8 if list(active) == NAMES_A else 0.4  # A improves, B worse

        store = AflowPolicyStore(root_dir=str(tmp_path))
        optimizer = AFlowOptimizer(
            _gateway(_candidates_json([NAMES_A, NAMES_B])),
            store,
            run_fn,
            _fake_settings(max_candidates=2, improvement_margin=0.0),
            curve_verdict=_clear_verdict,
        )

        result = await optimizer.optimize(NODE, CATEGORY, seeds=[_seed()])

        # Promoted + persisted.
        assert result.promoted is True
        assert result.names == NAMES_A
        assert result.best_score == pytest.approx(0.8)
        assert store.current_policy(NODE, CATEGORY) == NAMES_A

        # The builder hook resolves exactly those techniques (candidate path).
        set_aflow_candidate(NODE, CATEGORY, NAMES_A)
        techniques = aflow_techniques_for(NODE, CATEGORY)
        assert techniques is not None
        assert [t.name for t in techniques] == NAMES_A

        # A different category falls through to the heuristic (None).
        assert aflow_techniques_for(NODE, "reasoning") is None

    async def test_reject_if_worse(self, tmp_path: Any) -> None:
        """No candidate beats the baseline → nothing persisted, hook unaffected."""

        async def run_fn(_spec: Any) -> float | None:
            return 0.5 if aflow_candidate_for(NODE, CATEGORY) is None else 0.4

        store = AflowPolicyStore(root_dir=str(tmp_path))
        optimizer = AFlowOptimizer(
            _gateway(_candidates_json([NAMES_A, NAMES_B])),
            store,
            run_fn,
            _fake_settings(max_candidates=2),
            curve_verdict=_clear_verdict,
        )

        result = await optimizer.optimize(NODE, CATEGORY, seeds=[_seed()])

        assert result.promoted is False
        assert result.reason == "no improvement"
        assert store.current_policy(NODE, CATEGORY) is None
        # Hook returns nothing for an inactive (no-pointer, off) category.
        assert aflow_techniques_for(NODE, CATEGORY) is None

    async def test_preflight_curve_guard_aborts(self, tmp_path: Any) -> None:
        """A regressed C1 verdict skips before any gateway/run_fn call."""

        async def regressed_verdict() -> dict[str, Any]:
            return {"regressed": True, "inconclusive": False}

        run_calls = 0

        async def run_fn(_spec: Any) -> float | None:
            nonlocal run_calls
            run_calls += 1
            return 0.5

        gateway = _gateway(_candidates_json([NAMES_A]))
        optimizer = AFlowOptimizer(
            gateway,
            AflowPolicyStore(root_dir=str(tmp_path)),
            run_fn,
            _fake_settings(),
            curve_verdict=regressed_verdict,
        )

        result = await optimizer.optimize(NODE, CATEGORY, seeds=[_seed()])

        assert result.skipped is True
        assert result.reason.startswith("curve guard")
        assert gateway.acompletion.await_count == 0
        assert run_calls == 0

    async def test_off_is_byte_identical(self, tmp_path: Any) -> None:
        """enabled=False → optimize() is a no-op and the hook is transparent."""

        async def run_fn(_spec: Any) -> float | None:
            return 0.5

        gateway = _gateway(_candidates_json([NAMES_A]))
        optimizer = AFlowOptimizer(
            gateway,
            AflowPolicyStore(root_dir=str(tmp_path)),
            run_fn,
            _fake_settings(enabled=False),
        )

        result = await optimizer.optimize(NODE, CATEGORY, seeds=[_seed()])

        assert result.skipped is True
        assert result.reason == "off"
        assert gateway.acompletion.await_count == 0
        # No candidate + disabled → hook returns None (heuristic owns selection).
        assert aflow_techniques_for(NODE, CATEGORY) is None

        # And select_techniques_for_node equals the heuristic selector directly.
        goal = "write a python function that sorts a list of numbers"
        via_hook = select_techniques_for_node(TaskComplexity.COMPLEX, NODE, goal_text=goal)
        pattern = TechniqueSelector.infer_goal_pattern(goal)
        via_selector = TechniqueSelector().select(
            complexity=TaskComplexity.COMPLEX, node=NODE, goal_pattern=pattern
        )
        assert [t.name for t in via_hook] == [t.name for t in via_selector]

    async def test_never_raises_on_gateway_error(self, tmp_path: Any) -> None:
        """A gateway error surfaces as a structured result, never an exception."""

        async def run_fn(_spec: Any) -> float | None:
            return 0.5  # baseline must succeed so we reach the proposal call

        gateway = MagicMock()
        gateway.acompletion = AsyncMock(side_effect=RuntimeError("boom"))

        optimizer = AFlowOptimizer(
            gateway,
            AflowPolicyStore(root_dir=str(tmp_path)),
            run_fn,
            _fake_settings(),
            curve_verdict=_clear_verdict,
        )

        result = await optimizer.optimize(NODE, CATEGORY, seeds=[_seed()])

        assert result.promoted is False
        assert result.reason.startswith("error:")

    async def test_budget_cap_honored(self, tmp_path: Any) -> None:
        """max_cost_usd hit mid-search stops before any candidate is evaluated."""

        eval_calls = 0

        async def run_fn(_spec: Any) -> float | None:
            nonlocal eval_calls
            eval_calls += 1
            return 0.5

        async def over_budget() -> float:
            return 1.0  # already over the 0.5 cap → loop body never runs

        optimizer = AFlowOptimizer(
            _gateway(_candidates_json([NAMES_A, NAMES_B])),
            AflowPolicyStore(root_dir=str(tmp_path)),
            run_fn,
            _fake_settings(max_candidates=2, max_cost_usd=0.5),
            curve_verdict=_clear_verdict,
            cost_fn=over_budget,
        )

        result = await optimizer.optimize(NODE, CATEGORY, seeds=[_seed()])

        assert result.promoted is False
        # Only the baseline seeds were scored — no candidate evaluated.
        assert eval_calls == 1


class TestAflowPolicyStoreAndResolve:
    def test_policy_roundtrip_and_unknown_names_skipped(self, tmp_path: Any) -> None:
        """install→current roundtrip; unknown names are dropped, never injected."""
        store = AflowPolicyStore(root_dir=str(tmp_path))

        assert store.current_policy(NODE, CATEGORY) is None
        store.install_policy(NODE, CATEGORY, NAMES_A, score=0.8, baseline=0.5)
        assert store.current_policy(NODE, CATEGORY) == NAMES_A

        # A policy with one unknown + one real name drops the unknown.
        resolved = resolve_policy(["definitely_unknown", NAMES_A[0]], NODE)
        assert [t.name for t in resolved] == [NAMES_A[0]]

        # All-unknown → empty (nothing injected).
        assert resolve_policy(["totally_unknown", "also_fake"], NODE) == []
