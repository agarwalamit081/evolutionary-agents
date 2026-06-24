"""Decision-logic tests for :class:`PromptOptimizer` (Phase 2 C2).

The engine's ``optimize()`` is a branchy state machine (textgrad → un-shipped
node → C1 curve guard → budget cp1 → no-signal → compile → budget cp2 → margin
→ promote). Each branch is pinned here WITHOUT a real LLM / DB / DSPy compile:

  * ``_TestOptimizer(PromptOptimizer)`` overrides the four async/external seams
    (``_curve_verdict`` / ``_build_canary`` / ``_resolve_lm`` / ``_compile``).
  * The in-function LOCAL imports inside ``optimize()`` (``LLMGateway``,
    ``get_session``, ``CostTracker``, ``PromotionGate``) are patched at their
    SOURCE modules (a local ``from x.y import Z`` re-reads ``x.y.Z`` each call,
    so patching the attribute on ``x.y`` is what the local import sees).
  * ``dspy`` is module-level in ``engine.py`` → ``monkeypatch`` replaces
    ``src.optimizer.engine.dspy`` with ``_FakeDspy`` (``FakeDspy.LM`` records
    kwargs so the ``cache=False`` + ``callbacks`` invariant is asserted).

The two entry guards (textgrad backend / un-shipped node) raise
:class:`ConfigurationError` BEFORE any gateway construction, so they need no
mocks at all. ``CostAccountingCallback`` is exercised directly (its own unit
test) since it is the cost-ledger seam for the otherwise-gateway-bypassing
``dspy.LM`` calls.
"""

from __future__ import annotations

from typing import Any

import pytest

# engine.py imports dspy + dspy.utils.callback at module top → guard the module.
pytest.importorskip("dspy")

from src.graph.enums import MutationType  # noqa: E402 — after importorskip
from src.optimizer.engine import CostAccountingCallback, PromptOptimizer  # noqa: E402
from src.optimizer.models import (  # noqa: E402
    ConfigurationError,
    OptimizeRequest,
)
from src.optimizer.profiles import NodeProfile, get_profile  # noqa: E402


# ── Fakes ────────────────────────────────────────────────────────────────────


class _FakeGateway:
    """Replaces ``LLMGateway``; only set_run_id / set_cost_tracker are touched."""

    def __init__(self, *_a: Any, **_kw: Any) -> None:
        self.run_id: str | None = None
        self.tracker: Any = None

    def set_run_id(self, run_id: str) -> None:
        self.run_id = run_id

    def set_cost_tracker(self, tracker: Any) -> None:
        self.tracker = tracker


class _FakeLLMSettings:
    """The slice of ``settings.llm`` the (overridden) seams would read."""

    anthropic_api_base: str | None = None
    alibaba_api_base: str | None = None

    def get_provider_key(self, _provider: str) -> str | None:
        return None


class _FakeSettings:
    """Stands in for ``Settings`` — only ``.optimizer`` + ``.llm`` are read."""

    def __init__(self, opt: Any) -> None:
        self.optimizer = opt
        self.llm = _FakeLLMSettings()


class _FakeTracker:
    """Replaces ``CostTracker``; spend is configurable per test."""

    def __init__(self, *_a: Any, spend: float = 0.0, **_kw: Any) -> None:
        self._spend = spend
        self.recorded: list[tuple[Any, ...]] = []

    async def get_run_spend(self, _run_id: str) -> float:
        return self._spend

    async def record_usage(self, *args: Any, **_kw: Any) -> None:
        self.recorded.append(args)


class _FakeSession:
    """Async CM for ``async with get_session() as session`` (session unused)."""

    def __init__(self, *_a: Any, **_kw: Any) -> None:
        pass

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *_exc: object) -> bool:
        return False


class _FakeCanary:
    """``GoldenCanary`` stand-in: baseline score (empty suffixes) vs candidate."""

    def __init__(self, *, baseline: float | None, candidate: float | None) -> None:
        self.baseline = baseline
        self.candidate = candidate
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    async def score(self, node: str, suffixes: list[str]) -> float | None:
        self.calls.append((node, tuple(suffixes)))
        return self.baseline if not suffixes else self.candidate


class _FakePromotionGate:
    """Captures the proposal + replays a promote() result."""

    def __init__(self, *_a: Any, result: dict[str, Any], **_kw: Any) -> None:
        self.result = result
        self.proposals: list[dict[str, Any]] = []

    async def promote(self, proposal: dict[str, Any]) -> dict[str, Any]:
        self.proposals.append(proposal)
        return self.result


class _FakeLM:
    """``dspy.LM`` stand-in; records construction kwargs."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeDspy:
    """Module-level ``engine.dspy`` replacement (only ``.LM`` is exercised)."""

    LM = _FakeLM

    @staticmethod
    def configure(**_kw: Any) -> None:
        """No-op: ``_compile`` is overridden so configure() never runs."""


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_opt(**overrides: Any) -> Any:
    from src.config.settings import OptimizerSettings

    return OptimizerSettings(_env_file=None, **overrides)


def _wire_base(monkeypatch: pytest.MonkeyPatch, *, spend: float = 0.0) -> _FakeTracker:
    """Patch the optimize()-local imports that run before any branch returns.

    ``LLMGateway`` / ``get_session`` / ``CostTracker`` are constructed
    unconditionally (gateway + budget machinery) so every optimize() call needs
    them faked. ``CostTracker`` is patched with a FACTORY returning the pre-built
    tracker (the engine calls ``CostTracker(session, settings)`` positionally, so
    the class itself can't carry the per-test ``spend``). Returns the tracker so
    a test can assert on spend/records.
    """
    tracker = _FakeTracker(spend=spend)
    monkeypatch.setattr("src.llm.gateway.LLMGateway", _FakeGateway)
    monkeypatch.setattr("src.db.session.get_session", _FakeSession)
    monkeypatch.setattr("src.llm.cost_tracker.CostTracker", lambda *_a, **_k: tracker)
    return tracker


def _wire_gate(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: dict[str, Any] | None = None,
) -> _FakePromotionGate:
    """Patch PromotionGate with a FACTORY returning the pre-built fake gate.

    The engine constructs the gate at ``PromotionGate(canary=..., settings=...)``
    (line 184) — BEFORE baseline is even computed — so EVERY test that reaches
    ``_build_canary`` also constructs it (not just the promote-path ones). The
    patched value must therefore be CALLABLE (a factory), not the instance.
    """
    gate = _FakePromotionGate(result=result or {"promoted": False, "reason": "not reached"})
    monkeypatch.setattr("src.evolution.promote.PromotionGate", lambda *_a, **_k: gate)
    return gate


class _TestOptimizer(PromptOptimizer):
    """Overrides the async/external seams; fakes injected via attributes."""

    def __init__(
        self,
        settings: Any,
        *,
        verdict: dict[str, Any] | None = None,
        canary: _FakeCanary | None = None,
        candidate: str = "candidate instruction",
        compile_raises: BaseException | None = None,
        model_id: str = "fake-model",
    ) -> None:
        super().__init__(settings)
        self._verdict = verdict if verdict is not None else {}
        self._fake_canary = canary
        self._candidate = candidate
        self._compile_raises = compile_raises
        self._model_id = model_id
        self.compile_calls = 0
        self.lm_seen: Any = None

    async def _curve_verdict(self) -> dict[str, Any]:
        return self._verdict

    async def _build_canary(self, _gateway: Any, _opt: Any) -> Any:
        assert self._fake_canary is not None  # configured by the test
        return self._fake_canary

    def _resolve_lm(self, _node: str) -> tuple[str, dict[str, Any]]:
        return self._model_id, {}

    def _compile(self, _backend: str, _opt: Any, lm: Any, _profile: NodeProfile) -> str:
        self.compile_calls += 1
        self.lm_seen = lm
        if self._compile_raises is not None:
            raise self._compile_raises
        return self._candidate


# ── Entry guards (raise before gateway construction → no mocks) ──────────────


@pytest.mark.asyncio
async def test_textgrad_backend_raises_configuration_error() -> None:
    """textgrad (torch) is deferred — a guarded seam, not a silent stub."""
    opt = _TestOptimizer(_FakeSettings(_make_opt()))
    with pytest.raises(ConfigurationError, match="textgrad"):
        await opt.optimize(OptimizeRequest(node="classify", backend="textgrad"))


@pytest.mark.asyncio
async def test_unshipped_node_raises_configuration_error() -> None:
    """Only 'classify' ships; any other node → ConfigurationError (not a stub)."""
    assert get_profile("reflect") is None  # the signal the engine turns into an error
    opt = _TestOptimizer(_FakeSettings(_make_opt()))
    with pytest.raises(ConfigurationError, match="reflect"):
        await opt.optimize(OptimizeRequest(node="reflect"))


# ── C1 curve guard ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_curve_guard_regressed_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    """A regressed curve → skip (never consolidate during a known regression)."""
    _wire_base(monkeypatch)
    opt = _TestOptimizer(_FakeSettings(_make_opt()), verdict={"regressed": True, "current": 0.3})
    resp = await opt.optimize(OptimizeRequest())
    assert resp.promoted is False
    assert resp.reason == "curve guard: regressed"


@pytest.mark.asyncio
async def test_curve_guard_inconclusive_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    """An inconclusive curve (too few nights) → skip (safe)."""
    _wire_base(monkeypatch)
    opt = _TestOptimizer(
        _FakeSettings(_make_opt()), verdict={"inconclusive": True, "current": 0.4}
    )
    resp = await opt.optimize(OptimizeRequest())
    assert resp.promoted is False
    assert resp.reason == "curve guard: inconclusive"


# ── Budget checkpoint 1 (before any spend) ───────────────────────────────────


@pytest.mark.asyncio
async def test_budget_over_cap_skips_before_compile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Already over cap at entry → skip before compile (no further spend)."""
    _wire_base(monkeypatch, spend=999.0)  # >> default max_cost_usd (0.50)
    opt = _TestOptimizer(
        # require_curve_clear=False so the curve guard isn't the thing that skips.
        _FakeSettings(_make_opt(require_curve_clear=False)),
    )
    resp = await opt.optimize(OptimizeRequest())
    assert resp.promoted is False
    assert resp.reason == "budget"
    assert opt.compile_calls == 0  # never compiled → no spend


# ── No eval signal ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_eval_signal_baseline_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    """A None baseline score → skip (nothing to improve against)."""
    _wire_base(monkeypatch)
    _wire_gate(monkeypatch)  # constructed before baseline is read; promote() never called
    opt = _TestOptimizer(
        _FakeSettings(_make_opt(require_curve_clear=False)),
        canary=_FakeCanary(baseline=None, candidate=0.8),
    )
    resp = await opt.optimize(OptimizeRequest())
    assert resp.promoted is False
    assert resp.reason == "no eval signal"


# ── Compile failure ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compile_failure_is_a_structured_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    """A DSPy compile exception → structured skip (never raised out of optimize)."""
    _wire_base(monkeypatch)
    monkeypatch.setattr("src.optimizer.engine.dspy", _FakeDspy)
    _wire_gate(monkeypatch)  # constructed before compile; promote() never called
    opt = _TestOptimizer(
        _FakeSettings(_make_opt(require_curve_clear=False)),
        canary=_FakeCanary(baseline=0.5, candidate=0.8),
        compile_raises=RuntimeError("GEPA blew up"),
    )
    resp = await opt.optimize(OptimizeRequest())
    assert resp.promoted is False
    assert resp.reason.startswith("compile failed")
    assert "GEPA blew up" in resp.reason


# ── No improvement ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_candidate_not_beating_baseline_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    """Candidate below baseline → skip (a worse prompt is never promoted)."""
    _wire_base(monkeypatch)
    monkeypatch.setattr("src.optimizer.engine.dspy", _FakeDspy)
    _wire_gate(monkeypatch)  # constructed before the margin check; promote() never called
    opt = _TestOptimizer(
        _FakeSettings(_make_opt(require_curve_clear=False)),
        canary=_FakeCanary(baseline=0.8, candidate=0.5),
    )
    resp = await opt.optimize(OptimizeRequest())
    assert resp.promoted is False
    assert resp.reason == "no improvement"
    assert resp.baseline == 0.8
    assert resp.candidate_score == 0.5


# ── Promote (happy path) ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_candidate_beats_baseline_is_promoted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Candidate > baseline + curve/budget clear → PromotionGate.promote fires."""
    _wire_base(monkeypatch)
    monkeypatch.setattr("src.optimizer.engine.dspy", _FakeDspy)
    # promote() is called; its result flows into resp. The gate reference is
    # held only in the parse-payload test below — this test pins the LM invariant.
    _wire_gate(monkeypatch, result={"promoted": True, "canary_score": 0.8})

    opt = _TestOptimizer(
        _FakeSettings(_make_opt(require_curve_clear=False)),
        canary=_FakeCanary(baseline=0.5, candidate=0.8),
    )
    resp = await opt.optimize(OptimizeRequest())

    assert resp.promoted is True
    assert resp.reason == "promoted"
    assert resp.baseline == 0.5
    assert resp.candidate_score == 0.8  # result.canary_score wins
    assert resp.suffixes == ["candidate instruction"]

    # The candidate LM was built with cache disabled (candidates change each
    # trial) and the cost-accounting callback scoped to it.
    assert opt.lm_seen is not None
    assert opt.lm_seen.kwargs["model"] == "litellm/fake-model"
    assert opt.lm_seen.kwargs["cache"] is False
    assert len(opt.lm_seen.kwargs["callbacks"]) == 1


@pytest.mark.asyncio
async def test_promoted_proposal_is_parse_prompt_payload_parseable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The promote proposal round-trips through parse_prompt_payload."""
    from src.evolution.promote import parse_prompt_payload

    _wire_base(monkeypatch)
    monkeypatch.setattr("src.optimizer.engine.dspy", _FakeDspy)
    gate = _wire_gate(monkeypatch, result={"promoted": True, "canary_score": 0.8})

    opt = _TestOptimizer(
        _FakeSettings(_make_opt(require_curve_clear=False)),
        canary=_FakeCanary(baseline=0.5, candidate=0.8),
    )
    await opt.optimize(OptimizeRequest())

    assert len(gate.proposals) == 1
    proposal = gate.proposals[0]
    assert proposal["mutation_type"] == MutationType.PROMPT
    parsed = parse_prompt_payload(proposal)
    assert parsed == ("classify", ["candidate instruction"])


# ── CostAccountingCallback (the gateway-bypass cost seam) ────────────────────


def test_cost_accounting_callback_records_usage() -> None:
    """on_lm_start/on_lm_end capture (model, provider, in, out) per dspy.LM call."""
    cb = CostAccountingCallback(provider_for=lambda _m: "openai")

    class _Instance:
        model = "litellm/gpt-4o-mini-2024-07-18"

    cb.on_lm_start("call-1", _Instance(), inputs={})
    cb.on_lm_end(
        "call-1",
        outputs={"usage": {"prompt_tokens": 12, "completion_tokens": 7}},
    )

    assert cb.records == [("gpt-4o-mini-2024-07-18", "openai", 12, 7)]


def test_cost_accounting_callback_strips_litellm_prefix_and_skips_errors() -> None:
    """The ``litellm/`` prefix is stripped; exceptions/empty outputs record nothing."""
    cb = CostAccountingCallback(provider_for=lambda _m: "deepseek")

    class _Instance:
        model = "litellm/deepseek-v4-flash"

    cb.on_lm_start("ok", _Instance(), inputs={})
    cb.on_lm_end("ok", outputs={"usage": {"input_tokens": 5, "output_tokens": 9}})
    cb.on_lm_end("err", None, exception=RuntimeError("boom"))  # skipped

    assert cb.records == [("deepseek-v4-flash", "deepseek", 5, 9)]
