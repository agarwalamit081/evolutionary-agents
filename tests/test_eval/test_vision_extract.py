"""Phase 5c vision validation: the ``vision_extract`` GoalSpec + its threading.

Hermetic. Three contracts are pinned here:

1. The spec is registered (``lookup_goal_spec`` finds it), carries the committed
   fixture image, and is NOT in ``BATTERY04_GOALS`` (the nightly battery is
   unperturbed).
2. Its golden checks genuinely verify the extracted values — a correct
   ``values.json`` passes every check; a wrong/hallucinated one fails.
3. The goal→image path threads end-to-end: a local image ref resolves to a
   gateway-ready data-URI, and ``execute_node`` passes ``state["images"]`` into
   ``gateway.acompletion_with_tools`` (so a vision-capable model receives it).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.eval.checks import GoldenCheck
from src.eval.golden import BATTERY04_GOALS, GOLDEN_SPECS, lookup_goal_spec
from src.graph.factory import initial_state
from src.graph.models import PlanStep
from src.graph.nodes.execute import execute_node
from src.llm.gateway import resolve_image_refs
from src.llm.models import ToolCallResponse

_FIXTURE = "tests/fixtures/vision_sample.png"
_DELIVERABLE = "results/vision_extract/values.json"


def _patch_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the shared settings source at isolated tmp roots (mirrors verify)."""
    results = tmp_path / "results"
    workspace = tmp_path / "workspace"
    results.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    fake = SimpleNamespace(
        agent=SimpleNamespace(
            results_root=str(results),
            workspace_root=str(workspace),
            results_per_run_subdir=False,
        ),
        eval=SimpleNamespace(
            eval_enabled=False,
            eval_enforce=False,
            eval_llm_judge_enabled=False,
            eval_canary_min_score=0.8,
            eval_store_enabled=False,
        ),
    )
    monkeypatch.setattr("src.config.settings.get_settings", lambda: fake)
    return results


def _write_values(root: Path, payload: dict[str, Any]) -> Path:
    """Write a values.json deliverable under <root>/vision_extract/."""
    target = root / "vision_extract" / "values.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


# ── 1. Spec registration ────────────────────────────────────────────────


def test_vision_spec_registered_not_in_battery() -> None:
    """The vision spec is discoverable by lookup, carries the image, and is
    excluded from the nightly capability-curve battery."""
    spec = lookup_goal_spec("vision_extract")
    assert spec is not None
    assert spec.spec_id == "vision_extract"
    assert _FIXTURE in spec.images
    assert "vision_extract" in GOLDEN_SPECS
    assert "vision_extract" not in [s.spec_id for s in BATTERY04_GOALS]
    # Every declared check is a golden value/existence assertion.
    names = [c.name for c in spec.checks]
    assert "vision_deliverable_exists" in names
    assert "vision_revenue_matches_image" in names


def test_vision_fixture_image_exists() -> None:
    """The committed fixture image is present and decodes (the spec references it)."""
    from PIL import Image

    path = Path(_FIXTURE)
    assert path.is_file(), f"fixture missing: {path}"
    with Image.open(path) as img:
        assert img.format == "PNG" and img.size[0] > 0


# ── 2. Golden checks verify extracted values ────────────────────────────


async def _run_golden_checks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, payload: dict[str, Any]
) -> list[tuple[str, bool, float]]:
    """Write ``payload`` as values.json and run every vision golden check."""
    _patch_roots(monkeypatch, tmp_path)
    _write_values(tmp_path / "results", payload)
    spec = lookup_goal_spec("vision_extract")
    assert spec is not None
    checker = GoldenCheck()
    out: list[tuple[str, bool, float]] = []
    for cfg in spec.checks:
        result = await checker.check(cfg, [_DELIVERABLE], {})
        out.append((cfg.name, result.passed, result.score))
    return out


@pytest.mark.asyncio
async def test_golden_checks_pass_on_correct_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A values.json matching the fixture image passes every vision check."""
    results = await _run_golden_checks(
        monkeypatch,
        tmp_path,
        {"revenue": 42500, "units_sold": 1247, "top_region": "EMEA"},
    )
    assert results, "spec must declare checks"
    for name, passed, score in results:
        assert passed, f"{name} should pass on correct values (score={score})"
        assert score == 1.0


@pytest.mark.asyncio
async def test_golden_checks_fail_on_wrong_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A hallucinated values.json fails the value checks (existence still passes)."""
    results = dict(
        (name, (passed, score))
        for name, passed, score in await _run_golden_checks(
            monkeypatch,
            tmp_path,
            {"revenue": 99999, "units_sold": 0, "top_region": "APAC"},
        )
    )
    # The deliverable exists, so the existence check still passes.
    assert results["vision_deliverable_exists"][0] is True
    # But every value assertion fails — a model that ignored the image is caught.
    assert results["vision_revenue_matches_image"][0] is False
    assert results["vision_units_sold_matches_image"][0] is False
    assert results["vision_top_region_matches_image"][0] is False


# ── 3. Goal → image threading ───────────────────────────────────────────


def test_resolve_image_refs_embeds_local_file_as_data_uri() -> None:
    """A local fixture path is read once + embedded as a base64 data-URI."""
    payloads = resolve_image_refs([_FIXTURE])
    assert len(payloads) == 1
    assert payloads[0].startswith("data:image/png;base64,")


def test_resolve_image_refs_passes_urls_through_and_drops_garbage() -> None:
    """URLs/data-URIs pass through verbatim; non-existent/unreadable refs drop."""
    url = "https://example.com/img.png"
    data = "data:image/png;base64,AAAA"
    payloads = resolve_image_refs([url, data, "", "/no/such/file.png", None])  # type: ignore[list-item]
    assert url in payloads
    assert data in payloads
    # The missing file + falsy entries are dropped (never raised).
    assert all(not p.startswith("/no/such") for p in payloads)


@pytest.mark.asyncio
async def test_execute_node_threads_images_to_gateway() -> None:
    """execute_node passes state["images"] into acompletion_with_tools.

    Proves the goal→image path: a run whose spec resolved images into state
    surfaces them on the reasoning-loop call, where the gateway folds them into
    the last user message for a vision-capable model. Mirrors the proven
    TestExecuteNodeLLM harness (gateway + tools returning one code_executor
    call so the node reaches its single acompletion_with_tools call).
    """
    gateway = MagicMock()
    gateway.acompletion_with_tools = AsyncMock(
        return_value=ToolCallResponse(
            content="wrote values",
            tool_calls=[
                {
                    "id": "tc1",
                    "type": "function",
                    "function": {
                        "name": "code_executor",
                        "arguments": '{"code": "print(1)"}',
                    },
                }
            ],
            model="gpt-4o-mini-2024-07-18",
            provider="openai",
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            cost_usd=0.0001,
        )
    )
    async_handler = AsyncMock(return_value="1")
    tools = MagicMock()
    tools.list_tools = MagicMock(
        return_value=[
            {
                "type": "function",
                "function": {
                    "name": "code_executor",
                    "description": "Execute code",
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string"}},
                    },
                },
            }
        ]
    )
    tools.get_handler = MagicMock(return_value=async_handler)

    images = resolve_image_refs([_FIXTURE])
    state = dict(initial_state("Read the attached image and extract values", "vision-thread", 10))
    state["plan_steps"] = [
        PlanStep(id="step1", description="Extract values from the image", status="pending"),
    ]
    state["images"] = images

    await execute_node(state, gateway=gateway, tools=tools)

    gateway.acompletion_with_tools.assert_awaited()
    _, kwargs = gateway.acompletion_with_tools.call_args
    assert kwargs.get("images") == images
