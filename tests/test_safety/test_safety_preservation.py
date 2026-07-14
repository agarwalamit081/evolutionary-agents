"""Tests for Layer 8 — the safety-preservation gate (Q93 taxonomy).

A self-modifying agent's gravest threat is a mutation that defangs its OWN
safety apparatus. This layer rejects those mutations across three categories
(``SafetyViolationType``): pipeline neutralization, eval/evolution gate-flag
flips, and validate() short-circuit. Distinct from Layers 1-7 (which vet what
the mutation DOES); this vets what the mutation does TO THE SAFETY SYSTEM.

Hermetic: calls ``SafetyPipeline.validate`` directly with crafted mutation
content; no sandbox, no DB, no provider key.
"""

from __future__ import annotations

import pytest

from src.graph.enums import MutationType
from src.safety.pipeline import SafetyPipeline, SafetyViolationError, SafetyViolationType


def _violations(result: dict) -> list[dict]:
    """Pull the preservation layer's typed violations list (empty when passed)."""
    return result["layers"]["preservation"].get("violations", [])


@pytest.mark.asyncio
async def test_rejects_pipeline_neutralization_reassign() -> None:
    """Reassigning SafetyPipeline to a stub is a pipeline_neutralization."""
    code = (
        "from src.safety.pipeline import SafetyPipeline\n"
        "SafetyPipeline = object  # defang the gate\n"
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    result = await SafetyPipeline().validate(code)
    assert result["passed"] is False
    cats = {v["category"] for v in _violations(result)}
    assert SafetyViolationType.PIPELINE_NEUTRALIZATION.value in cats


@pytest.mark.asyncio
async def test_rejects_pipeline_neutralization_clear_blocklist() -> None:
    """Clearing _FORBIDDEN_PATTERNS neutralizes the security scan."""
    code = "_FORBIDDEN_PATTERNS = []\n" "def add(a: int, b: int) -> int:\n    return a + b\n"
    result = await SafetyPipeline().validate(code)
    assert result["passed"] is False
    cats = {v["category"] for v in _violations(result)}
    assert SafetyViolationType.PIPELINE_NEUTRALIZATION.value in cats


@pytest.mark.asyncio
async def test_rejects_pipeline_neutralization_monkeypatch() -> None:
    """Monkey-patching SafetyPipeline.validate is a neutralization."""
    code = (
        "SafetyPipeline.validate = lambda *a, **k: {'passed': True, 'layers': {}, 'issues': []}\n"
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    result = await SafetyPipeline().validate(code)
    assert result["passed"] is False
    cats = {v["category"] for v in _violations(result)}
    assert SafetyViolationType.PIPELINE_NEUTRALIZATION.value in cats


@pytest.mark.asyncio
async def test_rejects_gate_flag_flip_in_code() -> None:
    """Flipping eval_enabled off in a CODE mutation is a gate_flag_flip."""
    code = "eval_enabled = False\ndef add(a: int, b: int) -> int:\n    return a + b\n"
    result = await SafetyPipeline().validate(code)
    assert result["passed"] is False
    cats = {v["category"] for v in _violations(result)}
    assert SafetyViolationType.GATE_FLAG_FLIP.value in cats


@pytest.mark.asyncio
async def test_rejects_gate_flag_flip_in_config() -> None:
    """A CONFIG mutation (YAML/TOML-style) flipping no_evolution is rejected too."""
    result = await SafetyPipeline().validate(
        "no_evolution: true\nother_setting = 1\n",
        context={"mutation_type": MutationType.CONFIG},
    )
    assert result["passed"] is False
    cats = {v["category"] for v in _violations(result)}
    assert SafetyViolationType.GATE_FLAG_FLIP.value in cats


@pytest.mark.asyncio
async def test_rejects_safety_result_hardcode() -> None:
    """Hardcoding safety_result to a passing dict short-circuits validate()."""
    code = (
        "safety_result = {'passed': True}\n"
        "def add(a: int, b: int) -> int:\n    return a + b\n"
    )
    result = await SafetyPipeline().validate(code)
    assert result["passed"] is False
    cats = {v["category"] for v in _violations(result)}
    assert SafetyViolationType.SAFETY_INVOCATION_DISABLE.value in cats


@pytest.mark.asyncio
async def test_rejects_safety_handle_null() -> None:
    """Setting the safety handle to None disables validate()."""
    code = "safety = None\ndef add(a: int, b: int) -> int:\n    return a + b\n"
    result = await SafetyPipeline().validate(code)
    assert result["passed"] is False
    cats = {v["category"] for v in _violations(result)}
    assert SafetyViolationType.SAFETY_INVOCATION_DISABLE.value in cats


@pytest.mark.asyncio
async def test_clean_code_passes_preservation() -> None:
    """A clean mutation has no preservation violations."""
    code = "def add(a: int, b: int) -> int:\n    return a + b\n"
    result = await SafetyPipeline().validate(code)
    assert result["passed"] is True
    assert _violations(result) == []


@pytest.mark.asyncio
async def test_prose_mention_not_flagged() -> None:
    """A PROMPT mutation that merely MENTIONS eval_enabled in prose (no
    assignment) is not flagged — only assignment syntax trips the gate."""
    result = await SafetyPipeline().validate(
        "If the eval is flaky you may set eval_enabled to false temporarily.",
        context={"mutation_type": MutationType.PROMPT},
    )
    # Prose: "set eval_enabled to false" — not an assignment, not flagged.
    assert _violations(result) == []


@pytest.mark.asyncio
async def test_prompt_with_flag_assignment_is_flagged() -> None:
    """A PROMPT mutation with a line-anchored flag assignment IS flagged — a
    prompt should never set gate-control flags."""
    result = await SafetyPipeline().validate(
        "Instructions:\neval_enabled = False\n",
        context={"mutation_type": MutationType.PROMPT},
    )
    assert result["passed"] is False
    cats = {v["category"] for v in _violations(result)}
    assert SafetyViolationType.GATE_FLAG_FLIP.value in cats


def test_safety_violation_error_taxonomy_roundtrip() -> None:
    """SafetyViolationError carries the typed category + detail (Q93)."""
    err = SafetyViolationError(
        SafetyViolationType.GATE_FLAG_FLIP, "rewrote eval_enabled"
    )
    assert err.violation is SafetyViolationType.GATE_FLAG_FLIP
    assert err.detail == "rewrote eval_enabled"
    assert "gate_flag_flip" in str(err)
    # The taxonomy values are stable strings (persisted in mutation records).
    assert {v.value for v in SafetyViolationType} == {
        "pipeline_neutralization",
        "gate_flag_flip",
        "safety_invocation_disable",
    }
