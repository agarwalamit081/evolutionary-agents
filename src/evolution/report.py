"""Evolution report generator — human-readable summary of evolution cycles.

Produces a formatted text report from the cycle result dict that is logged
at INFO level and stored in the evolution record for post-run inspection.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def generate_report(
    cycle_result: dict[str, Any],
    generation: int,
    trigger: str = "reflection_recommended",
) -> str:
    """Generate a human-readable evolution report.

    Args:
        cycle_result: The dict returned by ``SelfEvolutionEngine.run_cycle()``.
        generation: Current evolution generation number.
        trigger: What triggered the evolution cycle.

    Returns:
        Formatted multi-line report string.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    deployed = cycle_result.get("deployed", False)
    proposal = cycle_result.get("proposal", {})
    validation = cycle_result.get("validation", {})
    sandbox_result = cycle_result.get("sandbox_result", {})
    ab_result = cycle_result.get("ab_result", {})
    deployment = cycle_result.get("deployment", {})

    lines: list[str] = []
    lines.append("══════════════════════════════════════════════════════")
    lines.append(f"  EVOLUTION REPORT — {now}")
    lines.append(f"  Generation: {generation} | Trigger: {trigger}")
    lines.append("══════════════════════════════════════════════════════")

    # ── Opportunity ───────────────────────────────────────────────────
    mut_type = _fmt_enum(proposal.get("mutation_type", "unknown"))
    description = proposal.get("description", "Unknown improvement")
    priority = proposal.get("priority", "unknown")
    patterns = proposal.get("patterns", [])

    lines.append("")
    lines.append("OPPORTUNITY IDENTIFIED:")
    lines.append(f"  Type: {mut_type} | Priority: {priority}")
    lines.append(f"  Description: {description}")
    if patterns:
        lines.append(f"  Patterns addressed: {', '.join(str(p) for p in patterns[:5])}")

    # ── Mutation ──────────────────────────────────────────────────────
    method = "LLM" if proposal.get("model_used") else "heuristic"
    target = proposal.get("target_path") or "evolution/latest_mutation.json"
    rationale = proposal.get("rationale", "")
    model_used = proposal.get("model_used") or "N/A (heuristic)"

    lines.append("")
    lines.append("MUTATION GENERATED:")
    lines.append(f"  Method: {method}")
    lines.append(f"  Model used: {model_used}")
    lines.append(f"  Target: {target}")
    lines.append(f"  Rationale: {rationale}")

    # ── Validation ────────────────────────────────────────────────────
    validation_passed = validation.get("passed", False)
    val_icon = "✅" if validation_passed else "❌"
    lines.append("")
    lines.append(f"VALIDATION: {val_icon} {'Passed' if validation_passed else 'Failed'}")

    if not validation_passed:
        reason = validation.get("reason", "Unknown")
        lines.append(f"  Reason: {reason}")
    else:
        safety_result = validation.get("safety_result", {})
        layers = safety_result.get("layers", {})
        if layers:
            passed_count = sum(1 for v in layers.values() if v.get("passed"))
            lines.append(f"  Safety layers: {passed_count}/{len(layers)} passed")

    # ── Sandbox ───────────────────────────────────────────────────────
    sandbox_passed = sandbox_result.get("passed", True)
    sandbox_note = sandbox_result.get("note", "")
    sb_icon = "✅" if sandbox_passed else "❌"

    lines.append("")
    if sandbox_note:
        lines.append(f"SANDBOX: {sb_icon} {sandbox_note}")
    else:
        sb_details = sandbox_result.get("sandbox_result", {})
        duration = sb_details.get("duration_seconds", 0)
        lines.append(f"SANDBOX: {sb_icon} Executed in {duration:.2f}s")
        if not sandbox_passed:
            lines.append(f"  Exit code: {sb_details.get('exit_code')}")
            stderr_preview = sb_details.get("stderr", "")[:200]
            if stderr_preview:
                lines.append(f"  Stderr: {stderr_preview}")

    # ── A/B Test ──────────────────────────────────────────────────────
    ab_significant = ab_result.get("is_significant", False)
    ab_note = ab_result.get("note", "")
    ab_icon = "✅" if ab_significant else "❌"

    lines.append("")
    if ab_note:
        lines.append(f"A/B TEST: {ab_icon} {ab_note}")
    else:
        control = ab_result.get("control_result")
        treatment = ab_result.get("treatment_result")
        if control and treatment:
            lines.append(
                f"A/B TEST: {ab_icon} "
                f"control={'pass' if control.get('success') else 'fail'} "
                f"({control.get('duration_seconds', 0):.2f}s) vs "
                f"treatment={'pass' if treatment.get('success') else 'fail'} "
                f"({treatment.get('duration_seconds', 0):.2f}s)"
            )
        else:
            lines.append(f"A/B TEST: {ab_icon} {'Significant' if ab_significant else 'Not significant'}")

    # ── Deployment ────────────────────────────────────────────────────
    deploy_icon = "✅" if deployed else "❌"
    lines.append("")
    if deployed:
        commit_hash = deployment.get("commit_hash", "")
        hash_display = commit_hash[:8] if commit_hash else "(no commit)"
        lines.append(f"DEPLOYMENT: {deploy_icon} Deployed (generation {generation})")
        lines.append(f"  Commit: {hash_display}")
    else:
        reason = cycle_result.get("reason", deployment.get("reason", "Unknown"))
        lines.append(f"DEPLOYMENT: {deploy_icon} Rejected")
        lines.append(f"  Reason: {reason}")

    # ── Effect on future runs ─────────────────────────────────────────
    lines.append("")
    if deployed:
        lines.append("EFFECT ON FUTURE RUNS:")
        lines.append("  ✅ Mutation stored as skill in warm memory (PostgreSQL).")
        lines.append("  The agent will load this improvement on the next run")
        lines.append("  via the retrieve_memory → execute context pipeline.")
    else:
        lines.append("EFFECT ON FUTURE RUNS: None (mutation was not deployed)")

    lines.append("══════════════════════════════════════════════════════")

    return "\n".join(lines)


def _fmt_enum(value: Any) -> str:
    """Format an enum or plain value for display."""
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)
