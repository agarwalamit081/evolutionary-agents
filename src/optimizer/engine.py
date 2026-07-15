"""Metric-driven prompt optimizer engine (Phase 2 C2: DSPy MIPROv2/COPRO).

``PromptOptimizer.optimize`` searches a better prompt for a node and — when a
candidate beats the baseline — promotes it through the EXISTING
:class:`~src.evolution.promote.PromotionGate` (canary-gated, auto-rollback). It
is the missing piece C1 left out: C1 gave the system *measurement* (detect a
regression + roll back); this gives it *improvement* (turn the golden canary
into an objective and search against it).

Architecture (forced by the DSPy teleprompter API — see plan §1.3):
  The teleprompter (MIPROv2/COPRO) optimizes a DSPy *student module's* predictor
  instruction; it never hands candidate instructions back to caller code. The
  real golden canary runs the full agent graph, so it CANNOT be the in-loop
  metric (too expensive, and it does not return a per-example float the
  teleprompter can score against). So:

  1. The teleprompter searches candidate instructions against a CHEAP proxy
     metric over a DSPy student (bounds cost — each metric call is one cheap LLM
     call) and uses that for its in-loop scoring.
  2. The optimized instruction is VALIDATED against the REAL golden canary (full
     agent runs) — baseline (current prompt) vs candidate.
  3. If the candidate beats the baseline by the configured margin, it goes
     through ``PromotionGate`` whose own canary final-gate enforces the absolute
     ``eval_canary_min_score`` floor before the versioned write.

  The eval metric stays the promotion gate -> DoD ("prompts improve against the
  eval metric, automatically") is satisfied.

  GEPA is DEFERRED: it is now the external ``gepa`` package (a different
  functional API — ``gepa.optimize`` + ``GEPAAdapter``), not the ``dspy.GEPA``
  teleprompter this engine originally targeted. The ``dspy-gepa`` backend raises
  a clear ``ConfigurationError`` at the optimize() entry; use ``dspy-mipro``
  (default) or ``dspy-copro``.

Preconditions (default-off, both must hold before any spend):
  * ``OPTIMIZER_ENABLED`` (the scheduler only registers the job when true); and
  * C1's capability curve is clear — :meth:`optimize` refuses when
    ``CapabilityCurve.detect_regression`` reports ``regressed`` or
    ``inconclusive`` so we never consolidate during a known/temporary regression.

Cost bounding: DSPy's ``dspy.LM`` calls litellm DIRECTLY (not through our
:class:`~src.llm.gateway.LLMGateway`), so :class:`CostAccountingCallback` scopes
to that one LM and accumulates usage; the engine flushes it to the shared
``cost_ledger`` under the optimizer ``run_id``. The canary's own agent runs DO
flow through the wired gateway (same shared DB), so all optimizer spend is
attributed to one ``run_id`` and checked against ``OPTIMIZER_MAX_COST_USD`` at
two checkpoints (pre-compile and pre-promote).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import dspy
from dspy.utils.callback import BaseCallback
from loguru import logger

from src.config import get_settings
from src.config.settings import Settings
from src.graph.enums import MutationType, TaskComplexity
from src.optimizer.models import ConfigurationError, OptimizeRequest, OptimizeResponse, UsageReport
from src.optimizer.profiles import NodeProfile, get_profile

# Provider api_base pins for the providers whose default endpoint the gateway
# overrides (must match src/llm/gateway.py's _configure_litellm pins). Others
# fall through to litellm's provider default.
_ANTHROPIC_BASE = "https://api.anthropic.com"
_ALIBABA_BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def _read_usage(usage: Any, *keys: str) -> Any:
    """Read the first present value from a usage record (dict OR object).

    dspy 3.x stores ``history[-1]["usage"]`` as ``dict(response.usage)`` — a
    PLAIN dict (keys ``prompt_tokens``/``completion_tokens``/``total_tokens``)
    — NOT the raw litellm ``Usage`` object. ``getattr`` on that dict silently
    returns the default and under-reports tokens (the bug that flushed 0 tokens
    for real GEPA calls); dict access is correct here. Attribute access is kept
    as a fallback so a future dspy that stops dictifying can't silently zero
    the ledger again.
    """
    if isinstance(usage, dict):
        for key in keys:
            value = usage.get(key)
            if value:
                return value
        return 0
    for key in keys:
        value = getattr(usage, key, None)
        if value:
            return value
    return 0


class CostAccountingCallback(BaseCallback):
    """Capture DSPy ``dspy.LM`` call usage for cost-ledger attribution.

    DSPy's ``dspy.LM`` calls litellm directly, bypassing the gateway wrapper —
    so without this hook those calls are invisible to the cost ledger. The
    callback is scoped to one LM (``dspy.LM(callbacks=[cb])``) and accumulates
    ``(model, provider, input_tokens, output_tokens)`` per call. ``on_lm_end``
    is SYNC (DSPy calls it from the compile loop), so it cannot await the async
    DB write itself; the engine flushes :attr:`records` to
    :meth:`CostTracker.record_usage` on the async loop after ``compile``.
    """

    def __init__(
        self,
        *,
        provider_for: Callable[[str], str],
        litellm_to_key: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self._provider_for = provider_for
        # dspy.LM is given the provider-prefixed litellm id (e.g.
        # 'deepseek/deepseek-v4-flash'); the cost ledger / get_model_spec key on
        # the BARE registry id ('deepseek-v4-flash'), so map each litellm id back
        # to its bare key on call start. (A leading 'litellm/' prefix is NOT used
        # — litellm itself rejects 'litellm/<provider>/<model>' as 'LLM Provider
        # NOT provided'; the engine passes the provider-prefixed id verbatim.)
        self._litellm_to_key: dict[str, str] = litellm_to_key or {}
        self._models: dict[str, str] = {}
        # dspy 3.x: token usage lives on the LM instance's `history`, not in the
        # on_lm_end `outputs` (a bare list of {text, reasoning_content}). Capture
        # the instance at on_lm_start so on_lm_end can read usage back.
        self._instances: dict[str, Any] = {}
        self.records: list[tuple[str, str, int, int]] = []

    def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]) -> None:
        model = str(getattr(instance, "model", "") or "")
        # Resolve the provider-prefixed litellm model id back to the bare
        # registry key the cost ledger / get_model_spec know (e.g.
        # 'deepseek/deepseek-v4-flash' → 'deepseek-v4-flash'). An unknown id
        # (no map entry) passes through unchanged.
        model = self._litellm_to_key.get(model, model)
        self._models[call_id] = model
        self._instances[call_id] = instance

    def on_lm_end(
        self,
        call_id: str,
        outputs: Any,
        exception: Exception | None = None,
    ) -> None:
        # dspy 3.x: `outputs` is a list of {text, reasoning_content} dicts (NO
        # usage). On exception/empty there is nothing to attribute.
        if exception is not None or not outputs:
            return
        model = self._models.get(call_id, "")
        if not model:
            return
        in_t, out_t = self._usage_for(call_id)
        self.records.append((model, self._provider_for(model), in_t, out_t))

    def _usage_for(self, call_id: str) -> tuple[int, int]:
        """Best-effort token usage from the captured LM instance's history.

        dspy 3.x appends each LM call to ``instance.history`` (a list of dicts);
        the latest record carries token usage under ``"usage"``. Sequential GEPA
        student/reflection calls make ``history[-1]`` the just-completed call.
        Observability-only — a missing history/usage records zeros and never
        raises (the cost ledger is best-effort, the CostTracker-resilience
        pattern). See ``_read_usage`` for why the dict-vs-object form matters.
        """
        instance = self._instances.get(call_id)
        history = getattr(instance, "history", None)
        if not isinstance(history, list) or not history:
            return 0, 0
        last = history[-1]
        usage = last.get("usage") if isinstance(last, dict) else None
        if usage is None:
            return 0, 0
        in_t = int(_read_usage(usage, "prompt_tokens", "input_tokens") or 0)
        out_t = int(_read_usage(usage, "completion_tokens", "output_tokens") or 0)
        return in_t, out_t


class PromptOptimizer:
    """Search a better prompt for a node against the golden canary metric."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def optimize(self, req: OptimizeRequest) -> OptimizeResponse:
        """Run one optimization attempt for ``req.node``; return the outcome.

        Never raises for a runtime failure (compile error, canary error, budget)
        — those become a structured ``OptimizeResponse(promoted=False, reason=...)``.
        :class:`ConfigurationError` IS raised for an unsupported backend/node
        (a caller bug, not a runtime condition).
        """
        opt = self._settings.optimizer
        node = (req.node or opt.target_node).strip()
        backend = (req.backend or opt.backend).strip()

        if backend == "textgrad":
            raise ConfigurationError(
                "textgrad backend deferred (torch); use dspy-mipro/copro"
            )
        if backend == "dspy-gepa":
            raise ConfigurationError(
                "dspy-gepa backend deferred (GEPA is now the external `gepa` "
                "package with a different API — gepa.optimize/GEPAAdapter — not "
                "a dspy teleprompter); use dspy-mipro (default) or dspy-copro"
            )
        profile = get_profile(node)
        if profile is None:
            raise ConfigurationError(
                f"no optimizer profile for node '{node}'; only 'classify' ships in v1"
            )

        run_id = f"optimizer-{node}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"

        # Own gateway against the SHARED cost_ledger DB so canary agent-run spend
        # is attributed to run_id (and the budget check / circuit breaker apply).
        from src.llm.gateway import LLMGateway

        gateway = LLMGateway(self._settings)
        gateway.set_run_id(run_id)

        from src.db.session import get_session
        from src.llm.cost_tracker import CostTracker

        usage = UsageReport()
        async with get_session() as cost_session:
            tracker = CostTracker(cost_session, self._settings)
            gateway.set_cost_tracker(tracker)

            # ── C1 guard: do not consolidate during a known/temporary regression.
            # Call detect_regression() directly (NOT CurveRegressionGate.run,
            # which may auto-rollback — an undesired side effect from here).
            if opt.require_curve_clear:
                verdict = await self._curve_verdict()
                if verdict.get("regressed") or verdict.get("inconclusive"):
                    state = "regressed" if verdict.get("regressed") else "inconclusive"
                    logger.info(
                        f"Optimizer skipping '{node}': curve {state} "
                        f"(current={verdict.get('current')})"
                    )
                    return OptimizeResponse(
                        node=node, promoted=False, reason=f"curve guard: {state}", usage=usage
                    )

            # ── Budget checkpoint 1: refuse before any spend if already over cap
            # (guards a re-entrant / misconfigured nightly run).
            if await tracker.get_run_spend(run_id) >= opt.max_cost_usd:
                logger.warning(
                    f"Optimizer over cap before compile: >= ${opt.max_cost_usd:.4f}"
                )
                return OptimizeResponse(node=node, promoted=False, reason="budget", usage=usage)

            # ── Golden canary (full agent runs) + promotion gate (own canary).
            canary = await self._build_canary(gateway, opt, node)
            from src.evolution.promote import PromotionGate

            promotion_gate = PromotionGate(canary=canary.score, settings=self._settings)

            # ── Baseline: the current node prompt's canary score (no candidate).
            baseline = await canary.score(node, [])
            if baseline is None:
                logger.info(f"Optimizer: no eval signal for baseline of '{node}'")
                return OptimizeResponse(
                    node=node, promoted=False, reason="no eval signal", usage=usage
                )

            # ── DSPy student LM (cheap tier) + reflection LM (stronger). Both are
            # built with the provider-prefixed litellm id — dspy.LM passes the
            # ``model`` string straight to litellm, which routes by the leading
            # provider segment, so a bare registry key ('deepseek-v4-flash') makes
            # litellm raise 'LLM Provider NOT provided'; a 'litellm/' prefix is
            # rejected for the same reason. _resolve_credentials keeps returning
            # the bare key (provider/pricing lookups key on it); _litellm_id
            # expands it to 'deepseek/deepseek-v4-flash' for the call.
            model_id, lm_kwargs = self._resolve_lm(node)
            rmodel_id, rlm_kwargs = self._resolve_reflection_model(node)
            student_litellm = self._litellm_id(model_id)
            reflection_litellm = self._litellm_id(rmodel_id)
            cb = CostAccountingCallback(
                provider_for=self._provider_for,
                # Map each litellm id back to its bare registry key so the cost
                # callback's get_model_spec/pricing lookup succeeds.
                litellm_to_key={student_litellm: model_id, reflection_litellm: rmodel_id},
            )
            lm = dspy.LM(
                model=student_litellm,
                cache=False,  # candidate prompts change between trials
                callbacks=[cb],
                max_tokens=opt.max_tokens,
                temperature=opt.temperature,
                **lm_kwargs,
            )

            # ── GEPA REQUIRES a reflection_lm (its probe raises "requires a
            # reflection language model" without it); MIPROv2/COPRO propose
            # instructions with it. Shares the cost callback so student +
            # reflection attribute to one run_id/budget.
            reflection_lm = dspy.LM(
                model=reflection_litellm,
                cache=False,
                callbacks=[cb],
                max_tokens=opt.max_tokens,
                temperature=1.0,  # reflection/instruction-proposal favors higher temp
                **rlm_kwargs,
            )

            # ── GEPA/MIPROv2/COPRO compile against the proxy metric (sync → thread).
            logger.info(
                f"Optimizer compiling '{node}' via {backend} "
                f"(model={model_id}, max_trials={opt.max_trials}, "
                f"max_candidates={opt.max_candidates})"
            )
            compile_exc: Exception | None = None
            try:
                optimized = await asyncio.to_thread(
                    self._compile, backend, opt, lm, reflection_lm, profile
                )
            except Exception as exc:  # noqa: BLE001 — surface as a structured skip
                compile_exc = exc
                optimized = ""
            finally:
                await self._flush_usage(tracker, cb.records, run_id, usage)

            if compile_exc is not None:
                logger.warning(f"Optimizer compile failed for '{node}': {compile_exc}")
                return OptimizeResponse(
                    node=node,
                    promoted=False,
                    reason=f"compile failed: {compile_exc}",
                    baseline=baseline,
                    usage=usage,
                )
            candidate = (optimized or "").strip()
            if not candidate:
                return OptimizeResponse(
                    node=node, promoted=False, reason="empty candidate", baseline=baseline, usage=usage
                )

            # ── Validate the candidate against the REAL golden canary.
            candidate_score = await canary.score(node, [candidate])

            # ── Budget checkpoint 2: refresh spend (baseline + DSPy + candidate)
            # before the promote path re-runs the canary final-gate.
            usage.cost_usd = await tracker.get_run_spend(run_id)
            if usage.cost_usd >= opt.max_cost_usd:
                logger.warning(
                    f"Optimizer over cap before promote: ${usage.cost_usd:.4f} "
                    f">= ${opt.max_cost_usd:.4f}"
                )
                return OptimizeResponse(
                    node=node,
                    promoted=False,
                    reason="budget",
                    baseline=baseline,
                    candidate_score=candidate_score,
                    suffixes=[candidate],
                    usage=usage,
                )

            if candidate_score is None:
                return OptimizeResponse(
                    node=node,
                    promoted=False,
                    reason="no eval signal (candidate)",
                    baseline=baseline,
                    suffixes=[candidate],
                    usage=usage,
                )

            # ── Margin: candidate must not regress the baseline (and clear an
            # optional explicit improvement margin). The absolute floor is
            # enforced separately by PromotionGate (eval_canary_min_score).
            improves = candidate_score >= baseline and (
                opt.canary_min_score is None
                or (candidate_score - baseline) >= opt.canary_min_score
            )
            if not improves:
                logger.info(
                    f"Optimizer: '{node}' candidate {candidate_score:.3f} "
                    f"did not beat baseline {baseline:.3f}"
                )
                return OptimizeResponse(
                    node=node,
                    promoted=False,
                    reason="no improvement",
                    baseline=baseline,
                    candidate_score=candidate_score,
                    suffixes=[candidate],
                    usage=usage,
                )

            # ── Promote: the gate re-runs its own canary final-gate (absolute
            # floor) + writes the versioned artifact + auto-rollback pointer.
            proposal = {
                "mutation_type": MutationType.PROMPT,
                "mutated_content": json.dumps({"target_node": node, "suffixes": [candidate]}),
            }
            result = await promotion_gate.promote(proposal)
            promoted = bool(result.get("promoted"))
            reason = "promoted" if promoted else (result.get("reason") or "promotion rejected")
            logger.info(
                f"Optimizer promote '{node}': {reason} "
                f"(canary={result.get('canary_score', candidate_score)})"
            )
            return OptimizeResponse(
                node=node,
                promoted=promoted,
                reason=reason,
                baseline=baseline,
                candidate_score=result.get("canary_score", candidate_score),
                suffixes=[candidate],
                usage=usage,
            )

    # ── C1 guard ────────────────────────────────────────────────────────────

    async def _curve_verdict(self) -> dict[str, Any]:
        """Capability-curve regression verdict (observability-only read)."""
        from src.eval.curve import CapabilityCurve
        from src.eval.store import EvalStore

        return await CapabilityCurve(EvalStore()).detect_regression()

    # ── Canary stack ────────────────────────────────────────────────────────

    async def _build_canary(self, gateway: Any, opt: Any, node: str) -> Any:
        """Build the GoldenCanary, mirroring the runner/evolve node stack."""
        from src.agents.registry import SubAgentRegistry
        from src.evolution.promote import GoldenCanary
        from src.runner import _create_tool_registry, _load_dynamic_tools, _load_sub_agents

        tools = _create_tool_registry()
        if tools is not None:
            await _load_dynamic_tools(tools, self._settings)
        sub_agent_registry = SubAgentRegistry()
        await _load_sub_agents(sub_agent_registry, self._settings)
        return GoldenCanary(
            gateway, tools, sub_agent_registry, goal_ids=self._pick_goal_ids(opt, node)
        )

    def _pick_goal_ids(self, opt: Any, node: str) -> list[str]:
        """Golden spec ids for the canary — node-aware.

        Prefers specs whose ``target_node`` matches ``node`` (so the canary score
        tracks that node's decision — e.g. the classify-sensitive specs for
        ``target_node=classify``), then fills with universal
        (``target_node is None``) data-correctness specs, truncated to
        ``eval_spec_limit``. Without the node-tagged specs a node-prompt candidate
        can never lift the canary: data-correctness scores are inert to node prose,
        so a promotion is structurally impossible.
        """
        from src.eval.golden import GOLDEN_SPECS

        limit = max(1, int(opt.eval_spec_limit))
        wanted = (node or "").strip().lower()
        tagged = [
            sid for sid, spec in GOLDEN_SPECS.items() if (spec.target_node or "").lower() == wanted
        ]
        universal = [sid for sid, spec in GOLDEN_SPECS.items() if spec.target_node is None]
        ids = (tagged + universal)[:limit]
        return ids or ["battery04_q01"]

    # ── LM resolution ───────────────────────────────────────────────────────

    def _resolve_lm(self, node: str) -> tuple[str, dict[str, Any]]:
        """Resolve the cheap-tier student model + provider credentials for node."""
        from src.llm.model_router import ModelRouter

        model_id = ModelRouter(self._settings).route(TaskComplexity.SIMPLE, node=node)
        return self._resolve_credentials(model_id)

    def _reflection_model_or_none(self) -> str | None:
        """A pinned reflection model id, or None to fall back to COMPLEX-tier routing.

        Guards the pydantic-settings inline-comment leak: ``KEY=   # comment`` parses
        as the comment *text* (pydantic-settings 2.x does not strip inline comments),
        so a blank ``OPTIMIZER_REFLECTION_MODEL`` line that carries a trailing comment
        would otherwise reach ``dspy.LM`` as an invalid model id. A valid id is a single
        non-empty token (no whitespace, no leading ``#``) that resolves to a registered
        model spec; anything else is treated as unset.
        """
        from src.config.model_registry import get_model_spec

        raw = self._settings.optimizer.reflection_model.strip()
        if not raw or raw.startswith("#") or any(ch.isspace() for ch in raw):
            return None
        return raw if get_model_spec(raw) is not None else None

    def _resolve_reflection_model(self, node: str) -> tuple[str, dict[str, Any]]:
        """Resolve the reflection/proposal model (stronger than the student).

        GEPA requires a ``reflection_lm`` and MIPROv2/COPRO a ``prompt_model``;
        all three benefit from a stronger model than the cheap student. An
        explicit ``OptimizerSettings.reflection_model`` pins one; otherwise a
        COMPLEX-tier model is routed (genuinely stronger than the SIMPLE student
        — e.g. glm-4.7 — and not anthropic-blocked).
        """
        from src.llm.model_router import ModelRouter

        model_id = self._reflection_model_or_none() or ModelRouter(self._settings).route(
            TaskComplexity.COMPLEX, node=node
        )
        return self._resolve_credentials(model_id)

    def _resolve_credentials(self, model_id: str) -> tuple[str, dict[str, Any]]:
        """Resolve provider key + api_base for a model id (shared by both LMs)."""
        lm_kwargs: dict[str, Any] = {}
        api_key = self._settings.llm.get_provider_key(self._provider_for_id(model_id))
        if api_key:
            lm_kwargs["api_key"] = api_key
        base = self._api_base_for(model_id)
        if base:
            lm_kwargs["api_base"] = base
        return model_id, lm_kwargs

    def _litellm_id(self, model_id: str) -> str:
        """The provider-prefixed litellm model id for a (bare) registry key.

        dspy.LM hands ``model`` straight to litellm, which routes by the leading
        provider segment — so the bare registry key ('deepseek-v4-flash') MUST be
        expanded to its provider-prefixed litellm id ('deepseek/deepseek-v4-flash');
        a bare key makes litellm raise ``BadRequestError: LLM Provider NOT
        provided. You passed model=deepseek-v4-flash`` (verified against
        litellm/dspy: a leading ``litellm/`` prefix is rejected for the same
        reason). An unknown key (no registered spec) falls through unchanged so
        litellm surfaces the error visibly rather than a silent mis-route.
        """
        from src.config.model_registry import get_model_spec

        spec = get_model_spec(model_id)
        return spec.model_id if spec else model_id

    def _provider_for_id(self, model_id: str) -> str:
        from src.config.model_registry import get_model_spec

        spec = get_model_spec(model_id)
        return spec.provider if spec else "unknown"

    def _provider_for(self, model_id: str) -> str:
        # Bound method passed to CostAccountingCallback.
        return self._provider_for_id(model_id)

    def _api_base_for(self, model_id: str) -> str | None:
        provider = self._provider_for_id(model_id)
        llm = self._settings.llm
        if provider == "anthropic":
            return llm.anthropic_api_base or _ANTHROPIC_BASE
        if provider == "alibaba":
            return llm.alibaba_api_base or _ALIBABA_BASE
        return None

    # ── DSPy compile (sync; runs in a worker thread) ────────────────────────

    def _compile(
        self,
        backend: str,
        opt: Any,
        lm: Any,
        reflection_lm: Any,
        profile: NodeProfile,
    ) -> str:
        """Run the teleprompter and return the optimized instruction string.

        ``lm`` is the cheap student (set as DSPy's global default for the
        predictor); ``reflection_lm`` is the stronger proposal model passed as
        ``prompt_model`` to MIPROv2/COPRO (each benefits from a stronger model
        than the cheap student). ``dspy-gepa`` is guarded out at the entry to
        ``optimize`` (deferred — see module docstring) so it never reaches here.
        """
        dspy.configure(lm=lm)
        student = dspy.Predict(profile.signature_def)
        # Anchor the search at the current node prompt. MIPROv2/COPRO rewrite the
        # predictor's ``instructions`` during search; this seeds the starting
        # point. ``setattr`` (not attribute assignment) because DSPy's type stubs
        # do not declare ``instructions`` on ``Predict`` (it is set lazily).
        setattr(student, "instructions", profile.seed_instruction)
        trainset = [
            dspy.Example(**ex).with_inputs(profile.input_field) for ex in profile.examples
        ]

        if backend == "dspy-mipro":
            tele = dspy.MIPROv2(
                metric=profile.metric,
                prompt_model=reflection_lm,
                num_candidates=opt.max_candidates,
                auto="light",
            )
            compiled = tele.compile(
                student,
                trainset=trainset,
                num_trials=max(1, int(opt.max_trials or 1)),
                requires_permission_to_run=False,  # unattended container: never prompt
            )
        elif backend == "dspy-copro":
            tele = dspy.COPRO(
                prompt_model=reflection_lm, metric=profile.metric, breadth=opt.max_candidates
            )
            compiled = tele.compile(student, trainset=trainset, eval_kwargs={})
        else:  # pragma: no cover — guarded by the entry checks
            raise ConfigurationError(f"unsupported optimizer backend '{backend}'")

        return self._extract_instruction(compiled, profile.seed_instruction)

    def _extract_instruction(self, compiled: Any, fallback: str) -> str:
        """Read the optimized instruction off the compiled student's predictor.

        MIPROv2/COPRO rewrite the predictor's ``.instructions`` during search; a
        single-predictor student exposes it on its predictor. Falls back to the
        seed when no predictor carries a non-empty instruction.
        """
        try:
            predictors = (
                compiled.predictors() if hasattr(compiled, "predictors") else [compiled]
            )
        except Exception:  # noqa: BLE001 — tolerate unusual compiled shapes
            predictors = [compiled]
        for pr in predictors:
            instr = getattr(pr, "instructions", None)
            if isinstance(instr, str) and instr.strip():
                return instr.strip()
        return fallback

    # ── Cost flush ──────────────────────────────────────────────────────────

    async def _flush_usage(
        self,
        tracker: Any,
        records: list[tuple[str, str, int, int]],
        run_id: str,
        usage: UsageReport,
    ) -> None:
        """Flush accumulated DSPy usage to the shared cost ledger under run_id."""
        for model, provider, in_t, out_t in records:
            usage.input_tokens += in_t
            usage.output_tokens += out_t
            usage.calls += 1
            try:
                await tracker.record_usage(
                    model, provider, in_t, out_t, run_id=run_id
                )
            except Exception as exc:  # noqa: BLE001 — observability-only, never abort
                logger.debug(f"Optimizer usage record failed ({model}): {exc}")
