"""AFlow / ADAS workflow-topology optimization, bounded scope (Phase 5 G3b).

A *real search primitive* over the agent's planning-topology policy — distinct
from LATS (G3a, search over execution trajectories), from the mutation engine
(prompts/code/tools), and from C2's ``PromptOptimizer`` (single-node prompt
TEXT). AFlow searches which prompting TECHNIQUES (operators) get wired into each
graph node, in what order, PER TASK CATEGORY — the one rewirable topology surface
(``TechniqueSelector``), since the LangGraph node graph is fixed + run-control-
hardened at compile time and "optimizing the pipeline structure" is neither safe
nor cheap.

Bounded scope (owner-approved): it learns a per-(node, category) override of
``TechniqueSelector``'s selection — an ordered list of technique *names* — NOT
the compiled pipeline structure. The override is spliced at runtime via the
builder's ``aflow_techniques_for`` hook (mirrors the evolved-prompt trio), so a
host run is byte-identical until a policy is installed AND ``AFLOW_ENABLED``.

Offline optimizer (never raises, mirrors ``PromptOptimizer``): for a (node,
category) seed set, measure the baseline (override OFF), propose N candidate
policies via one gateway call, evaluate each via an injected ``run_fn`` (full
agent runs reading ``correctness_score``), keep the best if it clears the
improvement margin, and persist it through an ``AflowPolicyStore`` pointer
(mirrors ``PromotionGate``). A pre-flight ``CapabilityCurve`` regression gate
(C1) refuses to consolidate during a known regression or insufficient curve
data. Cost is bounded by ``max_candidates`` + the seed set; ``max_cost_usd`` is
an advisory mid-search cap wired by the caller (the dominant, hard bounds are
``max_candidates`` and the per-run budget hard-stop inside ``execute_run``).

Default-off (``AFLOW_ENABLED``). When off, the builder hook short-circuits before
any pointer read → selection is byte-identical, and ``optimize()`` is a no-op.

Entry point: ``python main.py --aflow``. A scheduler job is deferred — it would
need full in-scheduler agent wiring (BenchmarkHarness + tool/sub-agent load) that
no existing in-scheduler job (curve-gate/governance) requires; the CLI is the v1
trigger, and a job can mirror ``add_optimizer_job`` once an AFlow runner is wired.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import json_repair
from loguru import logger

from src.graph.enums import TaskComplexity
from src.graph.prompts.technique_selector import (
    TECHNIQUE_REGISTRY,
    Technique,
    TechniqueSelector,
)

if TYPE_CHECKING:
    from src.eval.models import GoalSpec

# A run_fn scores one GoalSpec on 0.0–1.0 (its correctness_score), or None when
# it produced no signal. Injected so the CLI wires a real BenchmarkHarness runner
# and unit tests wire a deterministic fake that branches on the active candidate.
RunFn = Callable[["GoalSpec"], Awaitable[float | None]]
# Pre-flight C1 gate verdict: returns CapabilityCurve.detect_regression()'s dict
# ({regressed, inconclusive, ...}). Injected so tests inject a regressed verdict
# without a DB; the default builds CapabilityCurve lazily (best-effort).
CurveVerdictFn = Callable[[], Awaitable[dict[str, Any]]]
# Advisory cumulative-spend read (USD) for the mid-search cap. The default is a
# no-op (0.0); the CLI wires a real ledger read. Injected for deterministic tests.
CostFn = Callable[[], Awaitable[float]]

# Goal patterns that resolve to None (generic text) bucket under this label.
_CATEGORY_FALLBACK = "general"

# ── Prompts (module-level .format() constants — LATS idiom; no f-strings) ─────

_AFLOW_PROPOSE_SYSTEM = (
    "You are a workflow-topology optimizer for an autonomous agent. The agent wires "
    "named reasoning TECHNIQUES into each graph node's prompt. Given the task category "
    "'{category}' and the node '{node}', propose {n} DISTINCT candidate ORDERED lists of "
    "technique names that would best advance outcomes for this (node, category). Each "
    "list is a priority ordering — techniques earlier in the list lead. Use ONLY names "
    "from the available set below. Output STRICT JSON only: "
    '{{"candidates": [["name", ...], ...]}}. Each candidate must differ in ordering or '
    "membership from the current selection."
)

_AFLOW_PROPOSE_USER = (
    "TASK CATEGORY: {category}\n"
    "NODE: {node}\n"
    "AVAILABLE TECHNIQUES (by name): {available}\n"
    "CURRENT SELECTION: {current}\n\n"
    "Propose {n} distinct ordered candidate technique-lists that would improve the "
    "agent's correctness on '{category}' tasks at the '{node}' node. Return ONLY the "
    "JSON object."
)


# ── Result ─────────────────────────────────────────────────────────────────────


@dataclass
class AflowResult:
    """Outcome of one ``optimize()`` call for a single (node, category).

    Never an exception — a runtime failure (gateway error, parse failure, budget)
    becomes a structured ``reason`` with ``promoted=False`` (mirrors
    ``OptimizeResponse``). ``skipped`` marks a deliberate no-op (off / curve guard
    / no seeds / no signal), distinct from a searched-but-no-improvement result.
    """

    node: str
    category: str
    promoted: bool = False
    skipped: bool = False
    reason: str = ""
    baseline: float | None = None
    best_score: float | None = None
    names: list[str] = field(default_factory=list)


# ── Persistence (mirrors PromotionGate's versioned pointer) ───────────────────


def _sha8(node: str, category: str, names: list[str]) -> str:
    """Stable 8-char content hash for a policy version (dedups identical content)."""
    raw = json.dumps(
        {"node": node, "category": category, "names": names},
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:8]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AflowPolicyStore:
    """Versioned pointer of installed AFlow technique policies (mirrors PromotionGate).

    Keyed by ``"<node>|<category>"``. The runtime builder hook reads
    ``current_policy``; the optimizer writes ``install_policy``. Lives under
    ``.turing/aflow`` (gitignored scratch, like ``.turing/evolved``) unless an
    explicit ``root_dir`` is given (tests). ``current.json`` is the active pointer;
    ``policies/<node>.<category>.<sha>.json`` are immutable versioned artifacts.
    """

    _POINTER_NAME = "current.json"
    _POLICIES_SUBDIR = "policies"

    def __init__(self, root_dir: str | Path | None = None) -> None:
        self._root = Path(root_dir) if root_dir is not None else Path(".turing", "aflow")
        self._policies_dir = self._root / self._POLICIES_SUBDIR

    @staticmethod
    def key(node: str, category: str) -> str:
        """Pointer key for a (node, category) pair."""
        return f"{node}|{category}"

    def _pointer_path(self) -> Path:
        return self._root / self._POINTER_NAME

    def _read_pointer(self) -> dict[str, Any]:
        path = self._pointer_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug(f"AFlow pointer unreadable ({exc}); treating as empty")
            return {}

    def _write_pointer(self, data: dict[str, Any]) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        self._pointer_path().write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def current_policy(self, node: str, category: str) -> list[str] | None:
        """Active technique-name policy for (node, category); None when none installed."""
        entry = self._read_pointer().get(self.key(node, category))
        if not isinstance(entry, dict):
            return None
        names = entry.get("names")
        if not isinstance(names, list):
            return None
        clean = [n for n in names if isinstance(n, str) and n.strip()]
        return clean or None

    def install_policy(
        self, node: str, category: str, names: list[str], score: float, baseline: float
    ) -> dict[str, Any]:
        """Write the versioned policy artifact + flip the pointer to it.

        Returns the install manifest (``version`` + ``sha``). Best-effort on disk
        writes — the live pointer is the source of truth the builder reads.
        """
        sha = _sha8(node, category, names)
        installed_at = _utc_now_iso()
        version_name = f"{node}.{category}.{sha}.json"
        record = {
            "node": node,
            "category": category,
            "names": names,
            "score": round(float(score), 4),
            "baseline": round(float(baseline), 4),
            "sha": sha,
            "installed_at": installed_at,
        }
        self._policies_dir.mkdir(parents=True, exist_ok=True)
        (self._policies_dir / version_name).write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        pointer = self._read_pointer()
        pointer[self.key(node, category)] = {
            "active": version_name,
            "names": names,
            "score": round(float(score), 4),
            "baseline": round(float(baseline), 4),
            "sha": sha,
            "installed_at": installed_at,
        }
        self._write_pointer(pointer)
        logger.info(
            f"AFlow policy installed for {node}/{category}: {version_name} "
            f"(score={score:.3f}, baseline={baseline:.3f})"
        )
        return {"installed": True, "version": version_name, "sha": sha}


# ── Resolution helpers (shared by the builder hook + tests) ───────────────────


def technique_names_for_node(node: str) -> list[str]:
    """Technique names the registry keys on for ``node`` (the valid policy vocab)."""
    return [t.name for t in TECHNIQUE_REGISTRY if node in t.nodes]


def resolve_policy(names: list[str], node: str, budget_tokens: int = 512) -> list[Technique]:
    """Resolve an ordered name-list to Techniques (drop unknown / off-node), budget-capped.

    Names not in the registry or not keyed on ``node`` are silently dropped — a
    policy can never inject a technique the registry doesn't wire for that node.
    The budget cap mirrors ``TechniqueSelector.select``'s logic in the policy's
    OWN order (the order AFlow proposed is the priority), guaranteeing the first
    survivor survives a tight budget. Never raises.
    """
    by_name = {t.name: t for t in TECHNIQUE_REGISTRY if node in t.nodes}
    resolved: list[Technique] = []
    for name in names:
        technique = by_name.get(name)
        if technique is None or technique in resolved:
            continue
        resolved.append(technique)
    if not resolved:
        return []
    capped: list[Technique] = []
    spent = 0
    for technique in resolved:
        if spent + technique.token_cost_estimate <= budget_tokens:
            capped.append(technique)
            spent += technique.token_cost_estimate
    if not capped:  # guarantee the strongest survivor beats a tight budget
        capped = [resolved[0]]
    return capped


def bucket_specs_by_category(specs: list[GoalSpec]) -> dict[str, list[GoalSpec]]:
    """Group GoalSpecs by inferred goal pattern (None → ``general``)."""
    out: dict[str, list[GoalSpec]] = {}
    for spec in specs:
        category = (
            TechniqueSelector.infer_goal_pattern(getattr(spec, "goal_text", None))
            or _CATEGORY_FALLBACK
        )
        out.setdefault(category, []).append(spec)
    return out


# ── Optimizer ──────────────────────────────────────────────────────────────────


class AFlowOptimizer:
    """Per-(node, category) technique-policy search; never raises.

    Args:
        gateway: LLM gateway for the one proposal call per (node, category).
        store: ``AflowPolicyStore`` for persistence (install + pre-existing policy).
        run_fn: fitness — scores one GoalSpec on 0.0–1.0 (a full agent run's
            correctness_score). Injected so this is deterministic-testable and
            decoupled from the CLI/scheduler wiring.
        settings: ``AflowSettings`` (the knobs). ``Any``-typed so a minimal fake
            suffices for tests.
        curve_verdict: optional injected C1 verdict (tests); default builds
            ``CapabilityCurve`` lazily (best-effort; a read error → inconclusive).
        cost_fn: optional injected cumulative-spend read (CLI/tests); default 0.0
            (the advisory cap is inert unless the caller wires a real ledger read).
    """

    def __init__(
        self,
        gateway: Any,
        store: AflowPolicyStore,
        run_fn: RunFn,
        settings: Any,
        *,
        curve_verdict: CurveVerdictFn | None = None,
        cost_fn: CostFn | None = None,
    ) -> None:
        self._gateway = gateway
        self._store = store
        self._run_fn = run_fn
        self._settings = settings
        self._curve_verdict = curve_verdict or self._default_curve_verdict
        self._cost_fn = cost_fn or self._default_cost_fn
        # Cached across categories within one optimizer instance so a CLI loop
        # over (node, category) pairs computes the verdict once.
        self._verdict_cache: dict[str, Any] | None = None

    async def _default_curve_verdict(self) -> dict[str, Any]:
        try:
            from src.eval.curve import CapabilityCurve  # noqa: PLC0415
            from src.eval.store import EvalStore  # noqa: PLC0415

            return await CapabilityCurve(EvalStore()).detect_regression()
        except Exception as exc:  # noqa: BLE001 — gate must never abort; inconclusive is the safe skip
            logger.debug(f"AFlow curve verdict failed ({exc}); treating as inconclusive")
            return {"regressed": False, "inconclusive": True}

    async def _default_cost_fn(self) -> float:
        return 0.0  # advisory cap inert unless the CLI wires a real ledger read

    async def _verdict(self) -> dict[str, Any]:
        if self._verdict_cache is None:
            self._verdict_cache = await self._curve_verdict()
        return self._verdict_cache

    async def optimize(
        self, node: str, category: str, *, seeds: list[GoalSpec]
    ) -> AflowResult:
        """Search a better technique policy for (node, category); persist if it improves.

        Never raises — a runtime failure is a structured ``AflowResult(reason=...)``.
        Returns a ``skipped`` result when off / curve-guarded / seedless / signal-less.
        """
        try:
            return await self._optimize_inner(node, category, seeds)
        except Exception as exc:  # noqa: BLE001 — never raises
            logger.warning(f"AFlow optimize failed for {node}/{category}: {exc}")
            return AflowResult(node=node, category=category, reason=f"error: {exc}")

    async def _optimize_inner(
        self, node: str, category: str, seeds: list[GoalSpec]
    ) -> AflowResult:
        settings = self._settings
        if not getattr(settings, "enabled", False):
            return AflowResult(node=node, category=category, skipped=True, reason="off")

        # Pre-flight C1 gate (cached across categories within one optimizer).
        if getattr(settings, "preflight_curve_clear", True):
            verdict = await self._verdict()
            if verdict.get("regressed") or verdict.get("inconclusive"):
                state = "regressed" if verdict.get("regressed") else "inconclusive"
                logger.info(f"AFlow skipping {node}/{category}: curve {state}")
                return AflowResult(
                    node=node,
                    category=category,
                    skipped=True,
                    reason=f"curve guard: {state}",
                )

        if not seeds:
            return AflowResult(
                node=node, category=category, skipped=True, reason="no seeds"
            )

        from src.graph.prompts.builder import (  # noqa: PLC0415 — lazy; builder↔aflow are mutually lazy
            clear_aflow_candidate,
            set_aflow_candidate,
        )

        # Baseline: override OFF (clear any stale candidate defensively).
        clear_aflow_candidate()
        baseline = await self._mean_score(seeds)
        if baseline is None:
            return AflowResult(
                node=node, category=category, skipped=True, reason="no baseline signal"
            )

        # Propose candidates (one gateway call).
        candidates = await self._propose(node, category, int(getattr(settings, "max_candidates", 3)))
        if not candidates:
            return AflowResult(
                node=node,
                category=category,
                reason="no candidates proposed",
                baseline=baseline,
            )

        # Evaluate each candidate; keep the best that clears the improvement margin.
        margin = float(getattr(settings, "improvement_margin", 0.0))
        max_cost = float(getattr(settings, "max_cost_usd", 0.0))
        best_names: list[str] | None = None
        best_score = baseline
        for names in candidates:
            # Advisory mid-search cost cap (best-effort; inert unless CLI wires it).
            if max_cost > 0 and await self._cost_fn() >= max_cost:
                logger.info(f"AFlow cost cap hit mid-search for {node}/{category}")
                break
            set_aflow_candidate(node, category, names)
            try:
                score = await self._mean_score(seeds)
            finally:
                clear_aflow_candidate(node, category)
            if score is None:
                continue
            if score > best_score:
                best_score = score
                best_names = names

        if best_names is None or best_score < baseline + margin:
            logger.info(
                f"AFlow {node}/{category}: no improvement "
                f"(best={best_score:.3f} vs baseline {baseline:.3f} + margin {margin})"
            )
            return AflowResult(
                node=node,
                category=category,
                reason="no improvement",
                baseline=baseline,
                best_score=best_score if best_names is not None else None,
            )

        self._store.install_policy(node, category, best_names, best_score, baseline)
        return AflowResult(
            node=node,
            category=category,
            promoted=True,
            reason="promoted",
            baseline=baseline,
            best_score=best_score,
            names=list(best_names),
        )

    async def _mean_score(self, seeds: list[GoalSpec]) -> float | None:
        """Mean run_fn score over seeds (None when no seed produced a signal).

        One bad seed never aborts — it's skipped (logged at DEBUG). The override
        is whatever the caller set around the call (baseline = none, candidate =
        the policy under trial).
        """
        scores: list[float] = []
        for spec in seeds:
            try:
                score = await self._run_fn(spec)
            except Exception as exc:  # noqa: BLE001 — one bad seed never aborts the search
                logger.debug(f"AFlow run_fn failed for a seed: {exc}")
                continue
            if score is not None:
                scores.append(float(score))
        if not scores:
            return None
        return sum(scores) / len(scores)

    async def _propose(
        self, node: str, category: str, max_candidates: int
    ) -> list[list[str]]:
        """Ask the gateway for up to ``max_candidates`` distinct candidate policies."""
        available = technique_names_for_node(node)
        if not available:
            return []
        system = _AFLOW_PROPOSE_SYSTEM.format(
            n=max_candidates, category=category, node=node
        )
        user = _AFLOW_PROPOSE_USER.format(
            n=max_candidates,
            category=category,
            node=node,
            available=", ".join(available),
            current=", ".join(self._current_selection_names(node, category)),
        )
        data = await self._llm_json(system, user)
        raw = data.get("candidates") if isinstance(data, dict) else None
        if not isinstance(raw, list):
            return []
        valid = set(available)
        out: list[list[str]] = []
        for entry in raw:
            if not isinstance(entry, list):
                continue
            names = [n for n in entry if isinstance(n, str) and n in valid]
            if names:
                out.append(names)
            if len(out) >= max_candidates:
                break
        return out

    def _current_selection_names(self, node: str, category: str) -> list[str]:
        """The active policy names (aflow pointer) or the heuristic selection.

        Advisory context for the proposal prompt — names what the candidates are
        trying to beat. The COMPLEX tier is the typical AFlow target.
        """
        policy = self._store.current_policy(node, category)
        if policy:
            return policy
        try:
            selected = TechniqueSelector().select(
                complexity=TaskComplexity.COMPLEX, node=node, goal_pattern=category
            )
            return [t.name for t in selected]
        except Exception:  # noqa: BLE001 — advisory context only
            return []

    async def _llm_json(self, system: str, user: str) -> Any:
        """One gateway call returning a json_repair-parsed object (or ``{}``)."""
        response = await self._gateway.acompletion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
        content = (getattr(response, "content", "") or "").strip()
        if not content:
            return {}
        return json_repair.loads(content)
