"""Node optimization profiles for the DSPy/GEPA prompt optimizer (Phase 2 C2).

A profile is the DSPy-side specification the optimizer materializes for one
node: the student ``Signature`` (as a ``"input -> output"`` string), a seed
instruction (the node's current prompt), a small labeled trainset, and a CHEAP
proxy metric. GEPA/MIPROv2/COPRO search candidate instructions against the proxy
metric; only the winning candidate is then validated against the REAL golden
canary (full agent runs) in the engine — so the in-loop metric is cheap and cost
is bounded by the teleprompter's budget knobs, not by the metric.

Only ``classify`` ships in v1: it is the cleanest target (``goal -> complexity``,
exact-match metric against four labels). ``execute``/``verify``/``reflect`` are
deliberately NOT stubbed — :func:`get_profile` returns ``None`` for them so the
engine raises :class:`~src.optimizer.models.ConfigurationError` instead of
optimizing against an unvalidated proxy.

No ``dspy`` import here on purpose: the profile carries the signature as a plain
string and the metric as pure Python, so importing this module never requires
dspy. The engine imports dspy lazily at compile time (the optimizer container +
the test fakes are the only importers).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_DEFAULT_CLASSIFY_SEED = (
    "Classify the user's goal into exactly one complexity label: "
    "trivial, simple, complex, or critical. Judge by the number of steps, the "
    "tools and reasoning depth required, and whether it is multi-stage or "
    "high-stakes. Respond with only the lowercase label."
)

# A small, deterministic labeled trainset spanning all four complexity labels.
# These are realistic goal/complexity pairs (not synthetic) so the exact-match
# proxy metric teaches the optimizer the decision boundary.
_CLASSIFY_EXAMPLES: list[dict[str, str]] = [
    {"goal": "What is 2 plus 2?", "complexity": "trivial"},
    {"goal": "Sum the numbers in this list: 1, 2, 3, 4, 5", "complexity": "simple"},
    {
        "goal": "Find the cheapest flights from New York to London next month "
        "and summarize the top three options with prices.",
        "complexity": "simple",
    },
    {
        "goal": "Analyze this CSV of monthly sales, compute revenue trends, and "
        "generate a forecast report as markdown.",
        "complexity": "complex",
    },
    {
        "goal": "Build a REST API with authentication and a PostgreSQL schema, "
        "then deploy it to Kubernetes.",
        "complexity": "complex",
    },
    {
        "goal": "Design and implement a distributed, fault-tolerant real-time "
        "financial trading system with compliance auditing.",
        "complexity": "critical",
    },
]


def _classify_metric(
    example: Any,
    pred: Any,
    trace: Any = None,
    pred_name: str | None = None,
    pred_trace: Any = None,
) -> float:
    """Exact-match proxy metric for the classify signature.

    Matches GEPA's metric protocol ``(example, pred, trace, pred_name,
    pred_trace) -> float``. The optimizer validates the *winning* candidate
    against the real golden canary separately — this only steers the in-loop
    search cheaply.
    """
    gold = str(getattr(example, "complexity", "")).strip().lower()
    got = str(getattr(pred, "complexity", "")).strip().lower()
    if not gold:
        return 0.0
    return 1.0 if got == gold else 0.0


def _classify_seed() -> str:
    """The classify node's current system prompt, best-effort (fallback default).

    Anchors the search at the deployed prompt when the template renders without
    variables; any render failure falls back to the built-in default seed so the
    optimizer never hard-fails on a template plumbing detail.
    """
    try:
        from src.graph.prompts.loader import PromptTemplate

        text = str(PromptTemplate("classify_system")).strip()
        return text or _DEFAULT_CLASSIFY_SEED
    except Exception:  # noqa: BLE001 — seed is best-effort; never block the run
        return _DEFAULT_CLASSIFY_SEED


@dataclass(frozen=True)
class NodeProfile:
    """DSPy-side specification the optimizer materializes for one node."""

    node: str
    input_field: str
    output_field: str
    signature_def: str  # e.g. "goal -> complexity"
    seed_instruction: str
    examples: list[dict[str, str]]
    metric: Callable[..., float]


def get_profile(node: str) -> NodeProfile | None:
    """Resolve a node's optimization profile, or ``None`` when un-shipped.

    ``None`` is the deliberate signal for an un-shipped node (execute/verify/
    reflect) — the engine turns it into a :class:`ConfigurationError` so an
    unsupported target is never silently no-oped.
    """
    name = (node or "").strip().lower()
    if name == "classify":
        return NodeProfile(
            node="classify",
            input_field="goal",
            output_field="complexity",
            signature_def="goal -> complexity",
            seed_instruction=_classify_seed(),
            examples=list(_CLASSIFY_EXAMPLES),
            metric=_classify_metric,
        )
    return None
