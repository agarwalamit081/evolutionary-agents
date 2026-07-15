"""End-to-end wiring test for :class:`PromptOptimizer` (Phase 2 C2).

Distinct from ``test_optimizer_engine.py`` (which overrides ALL four seams and
pins each branch in isolation): this drives the REAL ``optimize()`` with the
seams running their REAL code, faking only the LEAF external dependencies the
real seam code reaches:

  * real ``_curve_verdict``  → the leaf ``CapabilityCurve.detect_regression`` is
    patched (construction is cheap: ``EvalStore`` has no ``__init__`` and
    ``CapabilityCurve`` reads only 3 config floats — no DB until detect runs).
  * real ``_compile``        → ``src.optimizer.engine.dspy`` is replaced with a
    fleshed-out ``_CompileFakeDspy`` (Predict/Example/MIPROv2/compiled.predictors)
    so the REAL extract-the-instruction flow runs and yields a candidate.
  * the two concessions: ``_build_canary`` (the full tools/sub-agents/GoldenCanary
    stack) and ``_resolve_lm`` (deep ``ModelRouter``+settings read) ARE overridden
    — they are heavy/deep-stack seams, not the optimize() decision logic under
    test. Everything else (curve read, compile, baseline/candidate scoring order,
    margin, proposal shape, promote call) runs through the real engine code.

Proves the candidate → score → guard → promote sequence composes end-to-end, and
that the real curve-guard wiring (not an override) skips on a regression.
"""

from __future__ import annotations

from typing import Any

import pytest

# engine.py imports dspy at module top → guard the module on import.
pytest.importorskip("dspy")

from src.optimizer.engine import PromptOptimizer  # noqa: E402 — after importorskip
from src.optimizer.models import OptimizeRequest  # noqa: E402


# ── Leaf fakes (same shape as the engine test's, kept here for independence) ──


class _FakeGateway:
    def __init__(self, *_a: Any, **_kw: Any) -> None:
        self.tracker: Any = None

    def set_run_id(self, _run_id: str) -> None: ...

    def set_cost_tracker(self, tracker: Any) -> None:
        self.tracker = tracker


class _FakeLLMSettings:
    anthropic_api_base: str | None = None
    alibaba_api_base: str | None = None

    def get_provider_key(self, _provider: str) -> str | None:
        return None


class _FakeSettings:
    def __init__(self, opt: Any) -> None:
        self.optimizer = opt
        self.llm = _FakeLLMSettings()


class _FakeTracker:
    def __init__(self, *_a: Any, spend: float = 0.0, **_kw: Any) -> None:
        self._spend = spend

    async def get_run_spend(self, _run_id: str) -> float:
        return self._spend

    async def record_usage(self, *args: Any, **_kw: Any) -> None: ...


class _FakeSession:
    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False


class _FakeCanary:
    def __init__(self, *, baseline: float | None, candidate: float | None) -> None:
        self.baseline = baseline
        self.candidate = candidate
        self.score_calls: list[tuple[str, tuple[str, ...]]] = []

    async def score(self, node: str, suffixes: list[str]) -> float | None:
        self.score_calls.append((node, tuple(suffixes)))
        return self.baseline if not suffixes else self.candidate


class _FakePromotionGate:
    def __init__(self, *_a: Any, result: dict[str, Any], **_kw: Any) -> None:
        self.result = result
        self.proposals: list[dict[str, Any]] = []

    async def promote(self, proposal: dict[str, Any]) -> dict[str, Any]:
        self.proposals.append(proposal)
        return self.result


# ── DSPy fake fleshed out so the REAL _compile() runs ────────────────────────


class _FakeLM:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakePredictor:
    def __init__(self, instructions: str) -> None:
        self.instructions = instructions


class _FakeCompiled:
    """A compiled student whose single predictor carries the optimized instruction."""

    def __init__(self, instruction: str) -> None:
        self._predictors = [_FakePredictor(instruction)]

    def predictors(self) -> list[_FakePredictor]:
        return self._predictors


class _FakeStudent:
    """Accepts the ``setattr(student, "instructions", seed)`` the real _compile does."""


class _FakeExample:
    def __init__(self, **_ex: Any) -> None: ...

    def with_inputs(self, _field: str) -> _FakeExample:
        return self


class _FakeTeleprompter:
    """MIPROv2/COPRO share this shape: compile() -> _FakeCompiled."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def compile(self, _student: Any, trainset: Any = None, **_kw: Any) -> _FakeCompiled:
        return _FakeCompiled("OPTIMIZED INSTRUCTION")


class _CompileFakeDspy:
    """Module-level ``engine.dspy`` replacement with the full _compile surface."""

    LM = _FakeLM
    # ``dspy.Predict(sig)`` → a student (the real signature class is a callable).
    Predict = staticmethod(lambda _sig: _FakeStudent())
    Example = _FakeExample
    GEPA = _FakeTeleprompter
    MIPROv2 = _FakeTeleprompter
    COPRO = _FakeTeleprompter

    @staticmethod
    def configure(**_kw: Any) -> None: ...


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_opt(**overrides: Any) -> Any:
    from src.config.settings import OptimizerSettings

    return OptimizerSettings(_env_file=None, **overrides)


def _wire_base(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the optimize()-local imports (gateway/session/tracker)."""
    monkeypatch.setattr("src.llm.gateway.LLMGateway", _FakeGateway)
    monkeypatch.setattr("src.db.session.get_session", lambda: _FakeSession())
    monkeypatch.setattr("src.llm.cost_tracker.CostTracker", lambda *_a, **_k: _FakeTracker())


def _wire_gate(
    monkeypatch: pytest.MonkeyPatch, *, result: dict[str, Any]
) -> _FakePromotionGate:
    """Patch PromotionGate with a factory returning the pre-built fake gate."""
    gate = _FakePromotionGate(result=result)
    monkeypatch.setattr("src.evolution.promote.PromotionGate", lambda *_a, **_k: gate)
    return gate


class _IntegrationOptimizer(PromptOptimizer):
    """Real optimize(); overrides only the two heavy/deep-stack seams.

    ``_build_canary`` (full tools/sub-agents/GoldenCanary stack) and
    ``_resolve_lm`` (deep ModelRouter+settings read) are NOT the optimize()
    decision logic under test — faking them isolates the wiring under test from
    the heavy runtime stack. ``_curve_verdict`` and ``_compile`` run REAL.
    """

    def __init__(self, settings: Any, canary: _FakeCanary) -> None:
        super().__init__(settings)
        self._fake_canary = canary

    async def _build_canary(self, _gateway: Any, _opt: Any, _node: str) -> Any:
        return self._fake_canary

    def _resolve_lm(self, _node: str) -> tuple[str, dict[str, Any]]:
        return "fake-model", {}

    def _resolve_reflection_model(self, _node: str) -> tuple[str, dict[str, Any]]:
        # Real optimize() now resolves a proposal LM (MIPROv2/COPRO's prompt_model);
        # the integration fake supplies one so the REAL _compile runs without a
        # deep ModelRouter+settings read.
        return "fake-reflection-model", {}


# ── Tests ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_candidate_score_guard_promote_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real optimize(): compile yields a candidate → scores → curve clear → promote."""
    from src.evolution.promote import parse_prompt_payload

    _wire_base(monkeypatch)
    monkeypatch.setattr("src.optimizer.engine.dspy", _CompileFakeDspy)
    gate = _wire_gate(monkeypatch, result={"promoted": True, "canary_score": 0.8})
    # Real _curve_verdict reads detect_regression() — patch the leaf to "clear".
    async def _clear(_self: Any) -> dict[str, Any]:
        return {}

    monkeypatch.setattr("src.eval.curve.CapabilityCurve.detect_regression", _clear)

    canary = _FakeCanary(baseline=0.5, candidate=0.8)
    opt = _IntegrationOptimizer(
        _FakeSettings(_make_opt(require_curve_clear=False)), canary
    )
    resp = await opt.optimize(OptimizeRequest())

    # Baseline is scored BEFORE the candidate (real ordering in optimize()).
    assert canary.score_calls[0] == ("classify", ())
    assert canary.score_calls[1] == ("classify", ("OPTIMIZED INSTRUCTION",))

    assert resp.promoted is True
    assert resp.reason == "promoted"
    assert resp.suffixes == ["OPTIMIZED INSTRUCTION"]  # extracted from the real compile
    assert resp.baseline == 0.5
    assert resp.candidate_score == 0.8

    # The promote proposal round-trips through parse_prompt_payload (real shape).
    assert len(gate.proposals) == 1
    parsed = parse_prompt_payload(gate.proposals[0])
    assert parsed is not None  # a valid PROMPT proposal always parses
    node, suffixes = parsed
    assert node == "classify"
    assert suffixes == ["OPTIMIZED INSTRUCTION"]


@pytest.mark.asyncio
async def test_mipro_receives_prompt_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: MIPROv2 (default) takes ``prompt_model=``; the real _compile()
    must pass the resolved proposal LM through. Captures the MIPROv2 constructor
    kwargs (COPRO shares the ``prompt_model`` param)."""
    captured: dict[str, Any] = {}

    class _Capturing(_FakeTeleprompter):
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)
            super().__init__(**kwargs)

    _wire_base(monkeypatch)
    monkeypatch.setattr("src.optimizer.engine.dspy", _CompileFakeDspy)
    # MIPROv2 is the default backend; swap in the capturing subclass so the real
    # _compile()'s MIPROv2(...) construction is observable.
    monkeypatch.setattr(_CompileFakeDspy, "MIPROv2", _Capturing)
    _wire_gate(monkeypatch, result={"promoted": True, "canary_score": 0.9})

    async def _clear(_self: Any) -> dict[str, Any]:
        return {}

    monkeypatch.setattr("src.eval.curve.CapabilityCurve.detect_regression", _clear)

    canary = _FakeCanary(baseline=0.5, candidate=0.9)
    opt = _IntegrationOptimizer(
        _FakeSettings(_make_opt(require_curve_clear=False)), canary
    )
    # Pin the backend explicitly so the assertion is hermetic to .env drift (the
    # default is also dspy-mipro, but this test targets the MIPROv2 compile path).
    await opt.optimize(OptimizeRequest(backend="dspy-mipro"))

    assert "prompt_model" in captured, f"MIPROv2 missing prompt_model: {captured}"
    assert captured["prompt_model"] is not None


@pytest.mark.asyncio
async def test_real_curve_guard_skips_on_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real _curve_verdict → detect_regression(regressed) → skip (wiring is real)."""
    _wire_base(monkeypatch)
    # No dspy/gate wiring needed: the curve guard returns before compile/promote.
    async def _regressed(_self: Any) -> dict[str, Any]:
        return {"regressed": True, "current": 0.3}

    monkeypatch.setattr("src.eval.curve.CapabilityCurve.detect_regression", _regressed)

    opt = _IntegrationOptimizer(
        _FakeSettings(_make_opt(require_curve_clear=True)),  # the default; explicit
        _FakeCanary(baseline=0.5, candidate=0.8),
    )
    resp = await opt.optimize(OptimizeRequest())

    assert resp.promoted is False
    assert resp.reason == "curve guard: regressed"
