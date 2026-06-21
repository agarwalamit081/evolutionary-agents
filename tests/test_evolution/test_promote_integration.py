"""Integration: the promotion gate's on-disk write is consumed by the LIVE agent.

The unit suite (``test_promote.py``) exercises ``PromotionGate`` in isolation —
the SAME gate instance that writes the ``current.json`` pointer reads it back via
``current_suffixes``. The LIVE agent never touches that instance:
``builder.evolved_suffixes_for_node`` / ``splice_evolved`` construct a FRESH
``PromotionGate()`` from ``get_settings()`` and read the on-disk pointer. That
gap is exactly why O2 was "shipped but never exercised" — nothing proved the
write path is actually consumable end-to-end.

These tests drive the full promote → (fresh-instance read-back → builder
read-back → prompt splice) → rollback lifecycle against a real tmp
``.turing/evolved/prompts/``, with a deterministic in-process canary. No live
LLM/DB. They lock:
  - an INDEPENDENT ``PromotionGate`` instance reads what gate-A wrote (the
    pointer is durable on disk, not held in memory on one object);
  - ``builder.evolved_suffixes_for_node`` / ``splice_evolved`` surface it into
    the prompt the live agent actually sends;
  - the ``EVOLUTION_PROMOTE_TO_LIVE`` opt-in flag gates read-back (OFF → ``[]``
    even with a pointer present on disk);
  - ``rollback`` propagates across instances — the fresh reader sees the revert.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from src.evolution.promote import PromotionGate
from src.graph.enums import MutationType
from src.graph.prompts.builder import (
    clear_evolved_candidate,
    evolved_suffixes_for_node,
    splice_evolved,
)


# ---------------------------------------------------------------------------
# helpers — minimal local copies (keeps this integration test self-contained)
# ---------------------------------------------------------------------------


def _fake_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    *,
    promote_on: bool = True,
    min_score: float = 0.8,
) -> SimpleNamespace:
    """Point BOTH the gate and the builder at the same tmp handlers dir.

    The builder lazy-imports ``get_settings`` and constructs its own
    ``PromotionGate`` from it, so patching the single source is what makes the
    write (gate-A) and the read (builder's fresh instance) agree on a directory.
    """
    fake = SimpleNamespace(
        evolution=SimpleNamespace(
            evolution_promote_to_live=promote_on,
            evolved_handlers_dir=str(tmp_path / "evolved"),
        ),
        eval=SimpleNamespace(eval_canary_min_score=min_score),
    )
    monkeypatch.setattr("src.config.settings.get_settings", lambda: fake)
    return fake


def _proposal(suffixes: list[str], node: str = "execute") -> dict[str, Any]:
    return {
        "mutation_type": MutationType.PROMPT,
        "description": "address JSON mistakes",
        "rationale": "guide the model toward valid JSON",
        "model_used": "test-model",
        "mutated_content": json.dumps({"target_node": node, "suffixes": suffixes}),
    }


def _canary(score: float | None) -> Any:
    async def _fn(_node: str, _suffixes: list[str]) -> float | None:
        return score

    return _fn


@pytest.fixture(autouse=True)
def _clear_candidate_override() -> Iterator[None]:
    """The in-process canary override is a module global — keep it isolated."""
    clear_evolved_candidate()
    yield
    clear_evolved_candidate()


# ---------------------------------------------------------------------------
# integration: gate write → live read-back
# ---------------------------------------------------------------------------


class TestPromotionGateLiveReadBack:
    @pytest.mark.asyncio
    async def test_independent_instance_reads_promotion(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """gate-A writes; a FRESH PromotionGate() (no canary, settings-only —
        exactly how the builder instantiates it) reads the promotion back. This
        proves the pointer is durable on disk, not in-memory on one object."""
        _fake_settings(monkeypatch, tmp_path)
        writer = PromotionGate(canary=_canary(0.9))
        result = await writer.promote(_proposal(["always emit valid JSON"]))

        assert result["promoted"] is True
        # A brand-new instance, constructed the way the builder does it, sees it.
        reader = PromotionGate()
        assert reader.current_suffixes("execute") == ["always emit valid JSON"]

    @pytest.mark.asyncio
    async def test_builder_reads_promoted_suffixes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """The LIVE read path: builder.evolved_suffixes_for_node constructs its
        own PromotionGate from get_settings and surfaces the promotion."""
        _fake_settings(monkeypatch, tmp_path)
        await PromotionGate(canary=_canary(0.9)).promote(_proposal(["suffix one"]))

        assert evolved_suffixes_for_node("execute") == ["suffix one"]

    @pytest.mark.asyncio
    async def test_splice_evolved_prepends_tagged_block(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """The promoted guidance reaches the prompt the live agent sends:
        splice_evolved prepends a tagged [evolved] block ahead of the base."""
        _fake_settings(monkeypatch, tmp_path)
        await PromotionGate(canary=_canary(0.9)).promote(_proposal(["be concise"]))

        spliced = splice_evolved("BASE SYSTEM PROMPT", "execute")
        assert spliced.startswith("[evolved]")
        assert "be concise" in spliced
        # The base prompt is preserved after the block.
        assert "BASE SYSTEM PROMPT" in spliced

    @pytest.mark.asyncio
    async def test_opt_in_flag_gates_read_back(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """Safety: with EVOLUTION_PROMOTE_TO_LIVE off, the builder returns []
        even though a promotion IS on disk. Promotion must never reach the live
        agent until the operator opts in."""
        _fake_settings(monkeypatch, tmp_path, promote_on=True)
        await PromotionGate(canary=_canary(0.9)).promote(_proposal(["ghost"]))
        assert evolved_suffixes_for_node("execute") == ["ghost"]

        # Flip the opt-in off — the on-disk pointer is still there, but the
        # builder must not surface it.
        _fake_settings(monkeypatch, tmp_path, promote_on=False)
        assert evolved_suffixes_for_node("execute") == []
        assert splice_evolved("BASE", "execute") == "BASE"


# ---------------------------------------------------------------------------
# integration: rollback propagates across instances
# ---------------------------------------------------------------------------


class TestPromotionGateRollbackLiveReadBack:
    @pytest.mark.asyncio
    async def test_rollback_seen_by_fresh_reader(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """A rollback performed on the writer instance is visible to the live
        builder reader — the fresh PromotionGate() re-reads the pointer and sees
        the reverted active version."""
        _fake_settings(monkeypatch, tmp_path)
        gate = PromotionGate(canary=_canary(0.9))
        await gate.promote(_proposal(["first guidance"]))
        await gate.promote(_proposal(["second guidance"]))
        assert evolved_suffixes_for_node("execute") == ["second guidance"]

        rolled = gate.rollback("execute")
        assert rolled["rolled_back"] is True
        # The live reader sees the revert without re-creating the writer.
        assert evolved_suffixes_for_node("execute") == ["first guidance"]

        # Roll back again → no prior version → node dropped from the pointer →
        # the live reader sees nothing (no [evolved] block at all).
        gate.rollback("execute")
        assert evolved_suffixes_for_node("execute") == []
        assert splice_evolved("BASE", "execute") == "BASE"

    @pytest.mark.asyncio
    async def test_end_to_end_lifecycle(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """Headline: promote → fresh-instance read → builder read → splice →
        rollback → builder reads nothing, all against the on-disk pointer."""
        _fake_settings(monkeypatch, tmp_path)
        gate = PromotionGate(canary=_canary(0.85))

        # 1. Promote; the versioned artifact + pointer both land on disk.
        promoted = await gate.promote(_proposal(["end-to-end guidance"]))
        assert promoted["promoted"] is True
        version_file = gate.prompts_dir / promoted["version"]
        assert version_file.exists()
        pointer = json.loads((gate.prompts_dir / "current.json").read_text("utf-8"))
        assert pointer["execute"]["active"] == promoted["version"]

        # 2. An independent instance + the live builder both read it.
        assert PromotionGate().current_suffixes("execute") == ["end-to-end guidance"]
        assert evolved_suffixes_for_node("execute") == ["end-to-end guidance"]
        assert splice_evolved("BASE", "execute").startswith("[evolved]")

        # 3. Rollback drops the only version; the live reader sees the revert.
        assert gate.rollback("execute")["rolled_back"] is True
        assert evolved_suffixes_for_node("execute") == []
        # The immutable version artifact survives as an audit trail.
        assert version_file.exists()
