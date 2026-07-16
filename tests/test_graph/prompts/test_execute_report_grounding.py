"""Execute prompt report-grounding instruction (Phase E, Fix B).

The battery-04 d-validation regression shipped a ``sales_report.md`` whose
prose numbers contradicted ``sales_summary.csv`` — the model transcribed figures
from memory instead of reading the computed data file back. Fix B added an
explicit grounding instruction to the execute system prompt so prose reports
cite figures that come from CODE reading the on-disk data artifact.

This is a characterization test: it locks the instruction against accidental
prompt deletion/rewrite so the regression does not silently return. Fix A's
``_cross_file_numeric_drift`` probe catches drift at verify; Fix B prevents the
model from producing it in the first place.
"""

from __future__ import annotations

from src.graph.prompts import EXECUTE_SYSTEM

# Phrases that MUST survive any execute-prompt edit. Each pins one half of the
# contract: the instruction to ground in code-read data, and the explicit
# fabrication-failure framing that motivates it.
_GROUNDING_MARKERS = (
    "Ground report figures in the computed data",
    "read the data file back inside code_executor",
    "recompute the figure there",
    "fabrication failure",
)


def test_execute_prompt_contains_report_grounding_instruction() -> None:
    """The execute system prompt instructs the agent to ground every report
    figure in code that reads the on-disk data artifact."""
    rendered = EXECUTE_SYSTEM.format(
        goal_text="Summarize sales by region into a report.",
        completed_count=1,
        total_steps=1,
        step_description="Write the report",
        memory_context="",
        tool_results_context="",
    )
    for marker in _GROUNDING_MARKERS:
        assert marker in rendered, (
            f"execute prompt lost its report-grounding instruction: missing {marker!r}"
        )


def test_grounding_instruction_applies_to_prose_reports_only() -> None:
    """The instruction scopes to PROSE reports/summaries — it must NOT forbid
    plain data deliverables (a .csv/.json is read straight from the tool that
    wrote it, no grounding prose). Assert the scoping clause is present so a
    future edit doesn't accidentally over-trigger it for data files."""
    rendered = EXECUTE_SYSTEM.format(
        goal_text="Write a CSV.",
        completed_count=0,
        total_steps=1,
        step_description="Emit rows",
        memory_context="",
        tool_results_context="",
    )
    # The "PROSE" qualifier and the scoping clause wrap across template lines
    # ("a PROSE\\n  report or summary (.md/.txt)"); assert the contiguous scope
    # token + the prose qualifier separately so a wrap edit doesn't flake this.
    assert "PROSE" in rendered
    assert "report or summary (.md/.txt)" in rendered
