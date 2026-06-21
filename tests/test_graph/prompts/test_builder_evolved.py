"""Phase 8 — evolved-prompt injection in ``src.graph.prompts.builder``.

The builder only loads promoted suffixes when ``evolution_promote_to_live`` is on
AND a PROMPT mutation was promoted for the node (pointer on disk). The canary's
in-process candidate override takes precedence and is cleared after. ``splice_evolved``
prepends a tagged ``[evolved]`` block; ``build_messages(node=...)`` wires it in.

``get_settings`` is monkeypatched (the builder reads it lazily) so promotion can
be toggled per-test without touching the lru_cache'd singleton.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from src.evolution.promote import PromotionGate
from src.graph.enums import MutationType
from src.graph.prompts.builder import (
    build_messages,
    clear_evolved_candidate,
    evolved_suffixes_for_node,
    set_evolved_candidate,
    splice_evolved,
)


def _settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    *,
    promote_on: bool = True,
) -> SimpleNamespace:
    fake = SimpleNamespace(
        evolution=SimpleNamespace(
            evolution_promote_to_live=promote_on,
            evolved_handlers_dir=str(tmp_path / "evolved"),
        ),
        eval=SimpleNamespace(eval_canary_min_score=0.8),
    )
    monkeypatch.setattr("src.config.settings.get_settings", lambda: fake)
    return fake


@pytest.fixture(autouse=True)
def _clear_override() -> Iterator[None]:
    """Ensure no in-process canary override leaks between tests."""
    clear_evolved_candidate()
    yield
    clear_evolved_candidate()


def _seed_promotion(monkeypatch: pytest.MonkeyPatch, tmp_path: Any, suffixes: list[str]) -> None:
    """Promote a PROMPT mutation so its suffixes land in the on-disk pointer."""

    async def _pass(_node: str, _s: list[str]) -> float | None:
        return 0.9

    _settings(monkeypatch, tmp_path, promote_on=True)
    gate = PromotionGate(canary=_pass)
    proposal = {
        "mutation_type": MutationType.PROMPT,
        "mutated_content": json.dumps({"target_node": "execute", "suffixes": suffixes}),
    }
    asyncio.run(gate.promote(proposal))


class TestEvolvedSuffixes:
    def test_empty_when_promotion_off(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
        _settings(monkeypatch, tmp_path, promote_on=False)
        set_evolved_candidate("execute", ["should-not-load"])
        # Toggle off short-circuits before the override is even consulted.
        assert evolved_suffixes_for_node("execute") == []

    def test_empty_when_node_none(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
        _settings(monkeypatch, tmp_path, promote_on=True)
        assert evolved_suffixes_for_node(None) == []

    def test_loads_promoted_suffixes_from_pointer(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _seed_promotion(monkeypatch, tmp_path, ["be strict about JSON", "verify types"])
        assert evolved_suffixes_for_node("execute") == ["be strict about JSON", "verify types"]
        # A node with no promotion is unaffected.
        assert evolved_suffixes_for_node("plan") == []

    def test_candidate_override_takes_precedence(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _seed_promotion(monkeypatch, tmp_path, ["from pointer"])
        set_evolved_candidate("execute", ["trial candidate"])
        assert evolved_suffixes_for_node("execute") == ["trial candidate"]
        clear_evolved_candidate("execute")
        # After clearing, the on-disk pointer is back in effect.
        assert evolved_suffixes_for_node("execute") == ["from pointer"]


class TestSpliceEvolved:
    def test_prepends_evolved_block_when_promoted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _seed_promotion(monkeypatch, tmp_path, ["always emit valid JSON"])
        out = splice_evolved("You are an executor.", "execute")
        assert out.startswith("[evolved]")
        assert "always emit valid JSON" in out
        assert out.endswith("You are an executor.")

    def test_passthrough_when_nothing_promoted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _settings(monkeypatch, tmp_path, promote_on=True)
        base = "You are a planner."
        assert splice_evolved(base, "plan") == base


class TestBuildMessagesNode:
    def test_node_arg_injects_promoted_guidance(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _seed_promotion(monkeypatch, tmp_path, ["emit UTC timestamps"])
        msgs = build_messages("base system", "user", node="execute")
        assert msgs[0]["role"] == "system"
        assert "[evolved]" in msgs[0]["content"]
        assert "emit UTC timestamps" in msgs[0]["content"]
        assert msgs[1] == {"role": "user", "content": "user"}

    def test_no_node_is_unchanged(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
        _seed_promotion(monkeypatch, tmp_path, ["secret"])
        msgs = build_messages("base system", "user")
        # Without a node, no evolved guidance is spliced (default-None path).
        assert msgs[0]["content"] == "base system"
