"""Retry-with-feedback for StructuredOutputManager.extract.

When the initial parse fails AND a gateway + the original messages are
supplied, extract re-prompts the model with the parse error + schema at
temperature=0. Without a gateway it is pure parse-or-fail (back-compat).

A FakeGateway scripts a sequence of response contents and records every
acompletion call (messages + kwargs) so the tests assert:
- bad-then-good recovers with exactly 1 retry;
- bad-then-bad returns None after max_retries;
- no-gateway path never calls the gateway;
- good JSON skips the gateway even when one is supplied;
- the feedback prompt carries the parse error + temperature=0.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from src.llm.structured_output import StructuredOutputManager


class SampleOutput(BaseModel):
    name: str
    value: int


class _Resp:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeGateway:
    """Records acompletion calls; returns scripted contents in order."""

    def __init__(self, contents: list[str]) -> None:
        self._contents = list(contents)
        self.calls: list[dict[str, Any]] = []

    async def acompletion(self, messages: list[dict[str, Any]] | None = None, **kwargs: Any) -> _Resp:
        self.calls.append({"messages": messages or [], "kwargs": kwargs})
        if not self._contents:
            return _Resp("")
        return _Resp(self._contents.pop(0))


# A raw response that fails EVERY parse stage (valid JSON object, but missing
# the required `value` field — json_repair cannot invent it).
_BAD = '{"name": "incomplete"}'
_GOOD = '{"name": "fixed", "value": 7}'
_MESSAGES = [{"role": "user", "content": "extract the sample"}]


class TestRetryWithFeedback:
    @pytest.mark.asyncio
    async def test_bad_then_good_recovers_with_one_retry(self) -> None:
        gw = FakeGateway([_GOOD])
        result = await StructuredOutputManager().extract(
            _BAD, SampleOutput, gateway=gw, messages=_MESSAGES, max_retries=1
        )
        assert result is not None
        assert result.name == "fixed"
        assert result.value == 7
        assert len(gw.calls) == 1  # exactly one retry call

    @pytest.mark.asyncio
    async def test_bad_then_bad_returns_none_after_max_retries(self) -> None:
        gw = FakeGateway([_BAD, _BAD])
        result = await StructuredOutputManager().extract(
            _BAD, SampleOutput, gateway=gw, messages=_MESSAGES, max_retries=2
        )
        assert result is None
        assert len(gw.calls) == 2  # exhausted both retries

    @pytest.mark.asyncio
    async def test_no_gateway_never_calls_and_returns_none(self) -> None:
        # Back-compat: pure parse-or-fail path, no LLM call.
        result = await StructuredOutputManager().extract(_BAD, SampleOutput)
        assert result is None

    @pytest.mark.asyncio
    async def test_gateway_without_messages_does_not_call(self) -> None:
        gw = FakeGateway([_GOOD])
        result = await StructuredOutputManager().extract(
            _BAD, SampleOutput, gateway=gw, messages=None, max_retries=1
        )
        assert result is None
        assert gw.calls == []

    @pytest.mark.asyncio
    async def test_good_json_skips_gateway_even_when_supplied(self) -> None:
        gw = FakeGateway([_GOOD])
        result = await StructuredOutputManager().extract(
            _GOOD, SampleOutput, gateway=gw, messages=_MESSAGES, max_retries=3
        )
        assert result is not None
        assert result.value == 7
        assert gw.calls == []  # first parse succeeded → no retry

    @pytest.mark.asyncio
    async def test_feedback_prompt_carries_error_and_temperature_zero(self) -> None:
        gw = FakeGateway([_GOOD])
        await StructuredOutputManager().extract(
            _BAD, SampleOutput, gateway=gw, messages=_MESSAGES, max_retries=1
        )
        assert len(gw.calls) == 1
        sent = gw.calls[0]["messages"]
        feedback = sent[-1]
        assert feedback["role"] == "user"
        # The correction prompt references JSON + the specific validation error.
        assert "JSON" in feedback["content"]
        assert "ValidationError" in feedback["content"]
        # The conversation preserves the original turn + the bad assistant reply.
        assert sent[0] == _MESSAGES[0]
        assert any(m.get("role") == "assistant" and m.get("content") == _BAD for m in sent)
        # temperature=0 is honored for determinism.
        assert gw.calls[0]["kwargs"].get("temperature") == 0.0

    @pytest.mark.asyncio
    async def test_fenced_json_from_retry_is_parsed(self) -> None:
        # The corrected response may still arrive inside code fences.
        gw = FakeGateway(["```json\n" + _GOOD + "\n```"])
        result = await StructuredOutputManager().extract(
            _BAD, SampleOutput, gateway=gw, messages=_MESSAGES, max_retries=1
        )
        assert result is not None
        assert result.value == 7
