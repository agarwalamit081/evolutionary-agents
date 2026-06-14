"""Per-call prompting-technique selection (§5).

Maps ``(complexity, node, goal-pattern)`` → an ordered list of injectable
technique bodies drawn from a curated registry. Selection is declarative
(metadata per technique), not hard-coded ``if/else``.

The ~57 ``.jinja2`` files under ``techniques/`` remain a reference library
of full prompts. This module holds the *curated subset* the agent actually
injects at runtime, each carrying its own short body text plus the metadata
(complexity fit, applicable nodes, goal patterns, token cost, priority)
needed for data-driven selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from src.graph.enums import TaskComplexity

# ── Node identifiers the selector keys on (string literals, kept typo-safe) ──
NODE_PLAN = "plan"
NODE_EXECUTE = "execute"
NODE_REFLECT = "reflect"
NODE_VERIFY = "verify"

# ── Goal-pattern keyword groups: match a goal's text to a technique family ──
_GOAL_PATTERNS: dict[str, tuple[str, ...]] = {
    "math": ("calculate", "compute", "solve", "equation", "math", "arithmetic",
             "formula", "algebra"),
    "reasoning": ("why", "reason", "deduce", "logic", "infer", "derive", "prove"),
    "code": ("code", "function", "implement", "debug", "refactor", "script",
             "program", "bug"),
    "research": ("search", "find", "investigate", "research", "look up",
                 "explore", "gather"),
    "writing": ("write", "draft", "compose", "summarize", "document", "article",
                "report"),
    "verification": ("verify", "audit", "check", "validate", "test", "review",
                     "confirm"),
    "analysis": ("analyze", "compare", "evaluate", "assess", "trade-off",
                 "tradeoff", "pros and cons"),
}


@dataclass(frozen=True)
class Technique:
    """A curated prompting technique injectable into a node's base prompt.

    Attributes:
        name: Stable identifier (matches the reference ``.jinja2`` file name
            where one exists).
        body: The injection text — concise reasoning guidance spliced above a
            node's JSON-schema footer. Must NOT contain its own
            ``<query>/<answer>`` scaffolding (the node supplies that).
        applies_to_complexities: Complexities this technique is suited to.
        nodes: Nodes where injecting this technique is meaningful.
        goal_patterns: Goal categories that make this technique especially
            apt. Empty means broadly applicable.
        token_cost_estimate: Rough body length in tokens, for budget capping.
        layer: Coaching category — ``reasoning`` / ``framing`` / ``verification``.
        priority: Tie-breaker; higher wins when several techniques qualify.
    """

    name: str
    body: str
    applies_to_complexities: frozenset[TaskComplexity]
    nodes: frozenset[str]
    goal_patterns: frozenset[str] = field(default_factory=frozenset)
    token_cost_estimate: int = 100
    layer: str = "reasoning"
    priority: int = 50


# ── Curated registry (data-driven; extend here, not in call sites) ──────────
TECHNIQUE_REGISTRY: list[Technique] = [
    Technique(
        name="chain_of_thought",
        body=(
            "Reason step by step before committing to an answer. Lay out each "
            "reasoning step explicitly, then derive the conclusion from those "
            "steps."
        ),
        applies_to_complexities=frozenset({TaskComplexity.COMPLEX, TaskComplexity.CRITICAL}),
        nodes=frozenset({NODE_PLAN, NODE_EXECUTE}),
        goal_patterns=frozenset({"math", "reasoning"}),
        token_cost_estimate=110,
        layer="reasoning",
        priority=80,
    ),
    Technique(
        name="self_ask",
        body=(
            "Break the query into 2-4 answerable sub-questions, answer each in "
            "turn, then synthesize the sub-answers into the final result."
        ),
        applies_to_complexities=frozenset({TaskComplexity.COMPLEX, TaskComplexity.CRITICAL}),
        nodes=frozenset({NODE_PLAN, NODE_EXECUTE}),
        goal_patterns=frozenset({"reasoning", "math"}),
        token_cost_estimate=90,
        layer="reasoning",
        priority=70,
    ),
    Technique(
        name="step_back",
        body=(
            "Before addressing the specifics, step back and state the general "
            "principle or framework that governs this problem, then apply it."
        ),
        applies_to_complexities=frozenset({TaskComplexity.COMPLEX, TaskComplexity.CRITICAL}),
        nodes=frozenset({NODE_PLAN, NODE_EXECUTE}),
        goal_patterns=frozenset({"reasoning", "code"}),
        token_cost_estimate=90,
        layer="reasoning",
        priority=65,
    ),
    Technique(
        name="generated_knowledge",
        body=(
            "Recall relevant facts, skills, or prior patterns from memory first; "
            "ground your answer in that recalled knowledge before proceeding."
        ),
        applies_to_complexities=frozenset({TaskComplexity.COMPLEX, TaskComplexity.CRITICAL}),
        nodes=frozenset({NODE_PLAN, NODE_EXECUTE}),
        goal_patterns=frozenset({"research", "code"}),
        token_cost_estimate=80,
        layer="reasoning",
        priority=60,
    ),
    Technique(
        name="role_prompting",
        body=(
            "Adopt the persona of a domain expert with deep, current knowledge "
            "of this field; reason and respond as that expert would."
        ),
        applies_to_complexities=frozenset({
            TaskComplexity.SIMPLE, TaskComplexity.COMPLEX, TaskComplexity.CRITICAL,
        }),
        nodes=frozenset({NODE_PLAN, NODE_EXECUTE}),
        goal_patterns=frozenset(),
        token_cost_estimate=70,
        layer="framing",
        priority=40,
    ),
    Technique(
        name="zero_shot",
        body=(
            "Answer directly and concisely — no need to decompose or show "
            "working unless the result is non-obvious."
        ),
        applies_to_complexities=frozenset({TaskComplexity.TRIVIAL, TaskComplexity.SIMPLE}),
        nodes=frozenset({NODE_PLAN, NODE_EXECUTE}),
        goal_patterns=frozenset(),
        token_cost_estimate=60,
        layer="framing",
        priority=85,
    ),
    Technique(
        name="reflection",
        body=(
            "After producing an initial assessment, critique it for accuracy, "
            "completeness, and clarity; then revise to address the weakest "
            "dimension."
        ),
        applies_to_complexities=frozenset({TaskComplexity.COMPLEX, TaskComplexity.CRITICAL}),
        nodes=frozenset({NODE_REFLECT}),
        goal_patterns=frozenset({"writing", "analysis"}),
        token_cost_estimate=90,
        layer="verification",
        priority=75,
    ),
    Technique(
        name="self_refine",
        body=(
            "Treat this as an improvement loop: assess the current state against "
            "the criteria, identify the lowest-scoring dimension, and refine "
            "specifically to lift it."
        ),
        applies_to_complexities=frozenset({TaskComplexity.COMPLEX, TaskComplexity.CRITICAL}),
        nodes=frozenset({NODE_REFLECT}),
        goal_patterns=frozenset({"writing", "code"}),
        token_cost_estimate=100,
        layer="verification",
        priority=80,
    ),
    Technique(
        name="chain_of_verification",
        body=(
            "After forming a verdict, independently verify each claim against "
            "the evidence; flag and correct any claim that does not hold up "
            "before declaring success."
        ),
        applies_to_complexities=frozenset({TaskComplexity.CRITICAL}),
        nodes=frozenset({NODE_VERIFY}),
        goal_patterns=frozenset({"verification", "math"}),
        token_cost_estimate=100,
        layer="verification",
        priority=85,
    ),
    Technique(
        name="checklist_prompting",
        body=(
            "Work through an explicit checklist of required dimensions; confirm "
            "each is satisfied before declaring completion."
        ),
        applies_to_complexities=frozenset({TaskComplexity.COMPLEX, TaskComplexity.CRITICAL}),
        nodes=frozenset({NODE_VERIFY}),
        goal_patterns=frozenset({"verification"}),
        token_cost_estimate=70,
        layer="verification",
        priority=70,
    ),
]


# Sentinel: the JSON-schema footer marker present in node system prompts.
# Technique bodies are spliced ABOVE this so the schema stays at the end and
# ``StructuredOutputManager.extract`` keeps working.
JSON_SCHEMA_MARKER = "Respond with a JSON object"


class TechniqueSelector:
    """Select prompting techniques for a single node call.

    The selector is deterministic for a given ``(complexity, node,
    goal_pattern, budget_tokens)`` tuple — same inputs always yield the same
    technique list (a §5 requirement).
    """

    def __init__(self, registry: list[Technique] | None = None) -> None:
        self._registry = registry if registry is not None else TECHNIQUE_REGISTRY

    @staticmethod
    def infer_goal_pattern(goal_text: str | None) -> str | None:
        """Map a goal's text to the first matching goal-pattern category.

        Returns ``None`` when no keyword group matches (techniques with empty
        ``goal_patterns`` still apply). Uses simple substring matching so the
        result is stable and explainable.
        """
        if not goal_text:
            return None
        lowered = goal_text.lower()
        for pattern, keywords in _GOAL_PATTERNS.items():
            if any(kw in lowered for kw in keywords):
                return pattern
        return None

    def select(
        self,
        complexity: TaskComplexity,
        node: str,
        goal_pattern: str | None = None,
        budget_tokens: int = 512,
    ) -> list[Technique]:
        """Return the ordered techniques to inject for this call.

        Filters by complexity + node fit, narrows to goal-pattern matches when
        any exist, ranks by priority (desc), and caps total token cost to
        ``budget_tokens``. Always returns at least the top-ranked candidate
        when any qualify (so a tight budget never zeroes out selection).
        """
        candidates = [
            t for t in self._registry
            if complexity in t.applies_to_complexities and node in t.nodes
        ]

        # Prefer goal-pattern matches, but fall back to the full candidate set
        # when none match the inferred pattern (broad techniques still apply).
        if goal_pattern:
            matched = [t for t in candidates if goal_pattern in t.goal_patterns]
            if matched:
                candidates = matched

        ranked = sorted(candidates, key=lambda t: t.priority, reverse=True)

        selected: list[Technique] = []
        spent = 0
        for technique in ranked:
            if spent + technique.token_cost_estimate <= budget_tokens:
                selected.append(technique)
                spent += technique.token_cost_estimate

        # Guarantee at least the strongest qualifier survives a tight budget.
        if not selected and ranked:
            selected = [ranked[0]]

        logger.info(
            f"Techniques selected for {node}/{complexity.value}/"
            f"{goal_pattern or 'none'}: {[t.name for t in selected]}"
        )
        return selected
