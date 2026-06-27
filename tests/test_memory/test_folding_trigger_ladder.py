"""Memory folding — the trigger ladder, RemoveMessage shrink, fold cap.

Companion to ``tests/test_memory/test_folding.py`` (which pins each trigger in
isolation, the JSON-repair salvage, fold-number sequencing, and serialization
helpers). This file covers the DISTINCT ladder-ordering + shrink-behavior angles
called out by the folding design, driving ``MemoryFolder`` directly with planted
message lists + a fake cost accumulator:

* the cap is the FIRST ladder rung — a capped run refuses even when every other
  trigger is screaming (so a 4th fold is refused);
* min-guard and cooldown fire in ladder order BEFORE the live-token/count/
  context triggers (a too-early or too-hot run does not fold);
* a successful fold emits ``RemoveMessage`` ops that SHRINK the message list
  (the reducer would drop the id'd messages);
* the live-token trigger reads the gateway cost-record accumulator (not state);
* ladder short-circuit: once the live-token trigger matches, the context-size
  trigger is not consulted (and vice-versa).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage, RemoveMessage

from src.memory.folding import MemoryFolder


def _mock_gateway(records: list[Any] | None = None) -> MagicMock:
    """A mock gateway whose ``get_cost_records`` returns the planted accumulator."""
    gateway = MagicMock()
    gateway.acompletion = AsyncMock()
    gateway.get_cost_records = MagicMock(return_value=list(records or []))
    return gateway


def _cost_record(input_tokens: int, output_tokens: int) -> MagicMock:
    rec = MagicMock()
    rec.input_tokens = input_tokens
    rec.output_tokens = output_tokens
    return rec


def _messages(count: int, *, chars: int = 50, with_id: bool = True) -> list[Any]:
    """Planted LangChain messages; id'd so RemoveMessage can target them."""
    msgs: list[Any] = []
    for i in range(count):
        msg = HumanMessage(content="x" * chars, id=f"msg-{i}" if with_id else None)
        msgs.append(msg)
    return msgs


def _state(
    *,
    iteration_count: int = 10,
    messages: list[Any] | None = None,
    fold_history: list[Any] | None = None,
    last_fold_iteration: int = 0,
) -> dict[str, Any]:
    return {
        "iteration_count": iteration_count,
        "messages": messages if messages is not None else [],
        "fold_history": fold_history if fold_history is not None else [],
        "last_fold_iteration": last_fold_iteration,
        "total_tokens_used": 0,
    }


# ─── Cap is the FIRST ladder rung ────────────────────────────────────


class TestFoldCapIsFirstRung:
    """``max_folds`` reached → should_fold False, even with every trigger screaming."""

    def test_fourth_fold_refused(self) -> None:
        """The default max_folds=3: a run with 3 folds refuses a 4th even when the
        live-token, message-count, AND context-size triggers are all exceeded."""
        gateway = _mock_gateway([_cost_record(10**6, 10**6)])  # tokens screaming
        folder = MemoryFolder(gateway)  # default max_folds=3
        state = _state(
            iteration_count=100,
            messages=_messages(50, chars=500),
            fold_history=[{"f": 1}, {"f": 2}, {"f": 3}],
        )
        assert folder.should_fold(state) is False

    def test_third_fold_allowed(self) -> None:
        """The 3rd fold (fold_history has 2) is allowed when a trigger is met."""
        gateway = _mock_gateway()
        folder = MemoryFolder(gateway, message_count_threshold=14)
        state = _state(
            iteration_count=30,
            messages=_messages(20),
            fold_history=[{"f": 1}, {"f": 2}],
            last_fold_iteration=20,
        )
        assert folder.should_fold(state) is True

    def test_custom_max_folds_one(self) -> None:
        """max_folds=1 → exactly one fold, then refused."""
        gateway = _mock_gateway()
        folder = MemoryFolder(gateway, max_folds=1, message_count_threshold=14)
        allowed = folder.should_fold(
            _state(iteration_count=10, messages=_messages(20))
        )
        refused = folder.should_fold(
            _state(
                iteration_count=20,
                messages=_messages(20),
                fold_history=[{"f": 1}],
                last_fold_iteration=10,
            )
        )
        assert allowed is True
        assert refused is False


# ─── Min-guard fires before substantive triggers ─────────────────────


class TestMinGuardPrecedesTriggers:
    """The min-guard (too-early / too-few-messages) fires BEFORE the live-token,
    count, and context triggers — so a screaming-token but 1-iteration run folds
    nothing."""

    def test_min_guard_iteration_blocks_even_with_huge_tokens(self) -> None:
        gateway = _mock_gateway([_cost_record(10**6, 10**6)])
        folder = MemoryFolder(gateway)
        state = _state(iteration_count=1, messages=_messages(50, chars=500))
        assert folder.should_fold(state) is False

    def test_min_guard_message_floor_blocks_even_with_huge_tokens(self) -> None:
        gateway = _mock_gateway([_cost_record(10**6, 10**6)])
        folder = MemoryFolder(gateway, message_count_floor=10)
        state = _state(iteration_count=10, messages=_messages(5))
        assert folder.should_fold(state) is False

    def test_min_guard_passes_at_boundary(self) -> None:
        """iteration_count=2 and messages=message_count_floor is the minimum that
        passes the guard (so a substantive trigger can then fire)."""
        gateway = _mock_gateway([_cost_record(10**6, 10**6)])
        folder = MemoryFolder(gateway, message_count_floor=10, token_threshold=1000)
        state = _state(iteration_count=2, messages=_messages(10))
        assert folder.should_fold(state) is True


# ─── Cooldown prevents back-to-back folds ────────────────────────────


class TestCooldownPreventsBackToBack:
    """Within ``fold_interval`` of the last fold, should_fold is False."""

    def test_cooldown_blocks_within_interval(self) -> None:
        gateway = _mock_gateway()
        folder = MemoryFolder(gateway, fold_interval=10, message_count_threshold=14)
        state = _state(
            iteration_count=12,
            messages=_messages(50),  # count trigger screaming
            last_fold_iteration=8,  # 12 - 8 = 4 < 10 → cooldown
        )
        assert folder.should_fold(state) is False

    def test_cooldown_clears_at_interval_boundary(self) -> None:
        gateway = _mock_gateway()
        folder = MemoryFolder(gateway, fold_interval=10, message_count_threshold=14)
        state = _state(
            iteration_count=20,
            messages=_messages(20),
            last_fold_iteration=10,  # 20 - 10 = 10 >= 10 → cleared
        )
        assert folder.should_fold(state) is True


# ─── Live-token trigger reads the gateway accumulator ────────────────


class TestLiveTokenTrigger:
    """The live-token trigger sums ``input+output`` across the gateway accumulator."""

    def test_live_token_above_threshold_fires(self) -> None:
        gateway = _mock_gateway([_cost_record(20_000, 20_000), _cost_record(15_000, 20_000)])
        folder = MemoryFolder(
            gateway, token_threshold=50_000, message_count_threshold=999
        )
        # 20+20+15+20 = 75_000 >= 50_000
        state = _state(iteration_count=10, messages=_messages(12))
        assert folder.should_fold(state) is True

    def test_live_token_below_threshold_does_not_fire(self) -> None:
        gateway = _mock_gateway([_cost_record(10_000, 10_000)])  # 20K < 50K
        folder = MemoryFolder(
            gateway,
            token_threshold=50_000,
            message_count_threshold=999,
            message_token_estimate=10**9,
        )
        state = _state(iteration_count=10, messages=_messages(12, chars=10))
        assert folder.should_fold(state) is False

    def test_live_token_gateway_exception_treated_as_zero(self) -> None:
        """If the gateway accumulator raises, the trigger reads 0 (fail-safe)."""
        gateway = MagicMock()
        gateway.get_cost_records = MagicMock(side_effect=RuntimeError("no gateway"))
        folder = MemoryFolder(
            gateway,
            token_threshold=1,
            message_count_threshold=999,
            message_token_estimate=10**9,
        )
        state = _state(iteration_count=10, messages=_messages(12, chars=10))
        # tokens=0 < threshold(1) won't fire; count/context suppressed → False.
        assert folder.should_fold(state) is False


# ─── Message-count + context-size triggers ───────────────────────────


class TestCountAndContextTriggers:
    """Message-count and context-size triggers fire independently."""

    def test_message_count_at_threshold_fires(self) -> None:
        gateway = _mock_gateway()
        folder = MemoryFolder(
            gateway,
            message_count_threshold=15,
            token_threshold=10**9,
            message_token_estimate=10**9,
        )
        state = _state(iteration_count=10, messages=_messages(15, chars=10))
        assert folder.should_fold(state) is True

    def test_context_size_above_estimate_fires(self) -> None:
        gateway = _mock_gateway()
        folder = MemoryFolder(
            gateway,
            message_token_estimate=100,
            message_count_threshold=999,
            token_threshold=10**9,
        )
        # 12 messages * 500 chars = 6000 chars // 4 = 1500 >= 100
        state = _state(iteration_count=10, messages=_messages(12, chars=500))
        assert folder.should_fold(state) is True


# ─── A successful fold SHRINKS the message list via RemoveMessage ────


class TestFoldShrinksMessages:
    """``build_removal_messages`` emits one RemoveMessage per id'd message —
    the add_messages reducer would DROP them, shrinking context."""

    def test_removals_target_every_idd_message(self) -> None:
        gateway = _mock_gateway()
        folder = MemoryFolder(gateway)
        msgs = _messages(8, with_id=True)
        removals = folder.build_removal_messages(_state(messages=msgs))
        assert len(removals) == 8
        assert all(isinstance(r, RemoveMessage) for r in removals)
        assert {r.id for r in removals} == {f"msg-{i}" for i in range(8)}

    def test_removals_skip_unidded_messages(self) -> None:
        """A message without an id cannot be removed by the reducer — skipped."""
        gateway = _mock_gateway()
        folder = MemoryFolder(gateway)
        msgs = _messages(5, with_id=True) + _messages(3, with_id=False)
        removals = folder.build_removal_messages(_state(messages=msgs))
        assert len(removals) == 5  # only the id'd ones

    @pytest.mark.asyncio
    async def test_removals_plus_summary_shrinks_context(self) -> None:
        """The fold contract: N old messages → N RemoveMessages + 1 summary. Under
        the add_messages reducer the net delta is ``1 - N`` messages, i.e. context
        SHRINKS for any N > 1."""
        gateway = _mock_gateway()
        gateway.acompletion = AsyncMock(return_value=MagicMock(content='{"a": 1}'))
        folder = MemoryFolder(gateway)
        msgs = _messages(12, with_id=True)
        state = _state(iteration_count=10, messages=msgs)

        removals = folder.build_removal_messages(state)
        summary = folder.build_summary_message(
            await folder.fold(state)
        )

        # Net message delta under the reducer = +1 summary, -N removals.
        net_delta = 1 - len(removals)
        assert net_delta < 0  # context shrinks
        assert isinstance(summary, HumanMessage)
        assert len(removals) == 12


# ─── Ladder short-circuit ordering ───────────────────────────────────


class TestLadderShortCircuit:
    """The ladder evaluates cap → guard → cooldown → token → count → context,
    first match wins; an earlier match short-circuits later rungs."""

    def test_token_trigger_short_circuits_before_context(self) -> None:
        """When the live-token trigger matches, the result is True regardless of
        the context-size estimate (the context rung is never reached)."""
        gateway = _mock_gateway([_cost_record(10**6, 10**6)])
        folder = MemoryFolder(
            gateway, token_threshold=1000, message_token_estimate=10**9
        )
        state = _state(iteration_count=10, messages=_messages(12, chars=10))
        assert folder.should_fold(state) is True

    def test_count_short_circuits_before_context(self) -> None:
        """Message-count trigger fires before the context-size rung is consulted."""
        gateway = _mock_gateway()
        folder = MemoryFolder(
            gateway,
            message_count_threshold=10,
            message_token_estimate=10**9,
            token_threshold=10**9,
        )
        state = _state(iteration_count=10, messages=_messages(10, chars=5))
        assert folder.should_fold(state) is True
