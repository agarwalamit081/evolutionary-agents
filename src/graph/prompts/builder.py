"""Dynamic prompt construction — splice technique bodies into a base prompt (§5).

``build_messages`` assembles the ``[system, user]`` pair a node sends to the
gateway. When techniques are supplied, their bodies are injected into the
system prompt as a bulleted block:

- **Above** the JSON-schema footer marker when one is present (plan/verify/
  reflect), so the schema stays at the tail and ``StructuredOutputManager.extract``
  keeps working.
- **After the first paragraph** otherwise (e.g. the tool-calling execute
  prompt, which has no JSON schema), so the guidance still leads the response.

With no techniques the base prompt is passed through unchanged.
"""

from __future__ import annotations

from src.graph.prompts.technique_selector import JSON_SCHEMA_MARKER, Technique

_TECHNIQUE_HEADING = "Reasoning techniques to apply:"


def render_technique_block(techniques: list[Technique]) -> str:
    """Render techniques as a headed bulleted block (empty string when none)."""
    if not techniques:
        return ""
    lines = [_TECHNIQUE_HEADING]
    lines += [f"- {technique.body}" for technique in techniques]
    return "\n".join(lines)


def splice_techniques(base_system: str, techniques: list[Technique]) -> str:
    """Inject technique bodies into a base system prompt.

    Above the JSON-schema marker when present; else after the first paragraph
    (or prepended if there is no paragraph break). No-op for an empty list.
    """
    block = render_technique_block(techniques)
    if not block:
        return base_system

    marker_pos = base_system.find(JSON_SCHEMA_MARKER)
    if marker_pos != -1:
        return base_system[:marker_pos] + block + "\n\n" + base_system[marker_pos:]

    # No schema marker: lead with the guidance after the opening paragraph.
    paragraph_break = base_system.find("\n\n")
    if paragraph_break != -1:
        head = base_system[:paragraph_break]
        rest = base_system[paragraph_break:]
        return f"{head}\n\n{block}{rest}"
    return f"{block}\n\n{base_system}"


def build_messages(
    base_system: str,
    user_content: str,
    techniques: list[Technique] | None = None,
) -> list[dict[str, str]]:
    """Build the ``[system, user]`` message pair for a gateway call.

    Args:
        base_system: The node's rendered system prompt (already ``.format``-ed).
        user_content: The rendered user prompt.
        techniques: Optional techniques whose bodies are spliced into the
            system prompt. ``None`` or empty → pass-through.

    Returns:
        A two-element message list ready for ``gateway.acompletion``.
    """
    system = splice_techniques(base_system, techniques or [])
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
