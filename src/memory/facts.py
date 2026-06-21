"""Semantic/fact memory tier — durable-fact extraction from episode text.

Facts are entity-ish knowledge ("orders.csv has 1,024 rows", "timestamps must be
UTC ISO-8601") distilled out of an episodic run. They live in warm memory as
``memory_type="fact"`` — distinct from skills/procedures (how-to) and from the
cold tier (raw episodes) — and are recalled alongside skills in
``retrieve_memory_node``.

This module owns only the *extraction* step (LLM → candidate facts); storage and
recall live on :class:`~src.memory.warm.WarmMemoryStore`. Extraction is
best-effort and never raises: a gateway/parse failure yields ``[]`` so it can
never break a fold.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import json_repair
from loguru import logger
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.llm.gateway import LLMGateway

# Maximum facts to keep per fold even if the model over-emits.
_DEFAULT_MAX_FACTS = 5


class FactCandidate(BaseModel):
    """A single extracted fact awaiting warm-memory persistence."""

    key: str = Field(..., description="Short stable identifier / entity name")
    value: str = Field(..., description="The durable fact value")
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="Extraction confidence")


def fact_extraction_prompt(text: str, max_facts: int = _DEFAULT_MAX_FACTS) -> str:
    """Build the fact-extraction prompt for an episode summary.

    The template body is brace-free so it is built via plain concatenation (not
    an f-string) — consistent with the jinja2-over-f-string rule for prompts.

    Args:
        text: The episode/summary text to mine for durable facts.
        max_facts: Upper bound on returned facts.

    Returns:
        The assembled prompt string.
    """
    return (
        "You are a fact extractor for a self-evolving agent's long-term memory.\n"
        "Read the run summary below and pull out DURABLE FACTS — entity-level\n"
        "knowledge that will remain true and useful on future runs (data schemas,\n"
        "row counts, invariants, canonical formats, environment quirks, proven\n"
        "recipes). Do NOT extract transient state, opinions, or narration.\n\n"
        f"Return at most {max_facts} facts as a JSON object:\n"
        '{"facts": [{"key": "<short stable id>", '
        '"value": "<the fact>", "confidence": 0.0-1.0}]}\n\n'
        "Return ONLY the JSON object — no prose, no markdown fences.\n\n"
        "RUN SUMMARY:\n"
        f"{text}"
    )


async def extract_facts(
    gateway: LLMGateway,
    text: str,
    *,
    max_facts: int = _DEFAULT_MAX_FACTS,
) -> list[FactCandidate]:
    """Mine ``text`` for durable facts via the gateway.

    Best-effort: returns ``[]`` on any gateway error, malformed JSON, or empty
    input. Never raises — extraction must not break the fold it runs inside.

    Args:
        gateway: An LLM gateway (provider config already resolved).
        text: The episode/summary text to mine.
        max_facts: Upper bound on returned candidates.

    Returns:
        Validated :class:`FactCandidate` list (possibly empty).
    """
    if not text or not text.strip():
        return []

    capped = max(1, max_facts)
    try:
        from src.graph.enums import TaskComplexity

        response = await gateway.acompletion(
            messages=[{"role": "user", "content": fact_extraction_prompt(text, capped)}],
            complexity=TaskComplexity.TRIVIAL,
            temperature=0.0,
        )
        content = response.content.strip()
    except Exception as exc:  # gateway outage must not break the fold
        logger.debug(f"Fact extraction gateway call failed: {exc}")
        return []

    # Strip a single markdown fence wrapper if present, then tolerate malformed
    # / truncated JSON via json_repair (mirrors MemoryFolder._generate_memory).
    if "```json" in content:
        content = content.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in content:
        content = content.split("```", 1)[1].split("```", 1)[0]

    try:
        parsed: Any = json_repair.loads(content.strip())
    except Exception as exc:  # defensive: json_repair rarely raises but never trust it
        logger.debug(f"Fact extraction JSON parse failed: {exc}")
        return []

    raw_facts = parsed.get("facts", []) if isinstance(parsed, dict) else []
    if not isinstance(raw_facts, list):
        return []

    candidates: list[FactCandidate] = []
    for item in raw_facts[:capped]:
        if not isinstance(item, dict):
            continue
        try:
            candidates.append(
                FactCandidate(
                    key=str(item.get("key", "")).strip(),
                    value=str(item.get("value", "")).strip(),
                    confidence=float(item.get("confidence", 0.5)),
                )
            )
        except (ValueError, TypeError):
            continue  # skip a single malformed fact, keep the well-formed ones

    # Drop empty-key/value entries — they carry no durable signal.
    return [c for c in candidates if c.key and c.value]
