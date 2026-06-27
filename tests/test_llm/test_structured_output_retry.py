"""``_parse_all`` json-repair salvage + retry-with-feedback edge matrix.

The retry-with-feedback loop is covered in ``test_structured_output_retry.py``
(bad→good, bad→bad, no-gateway, fenced retry). The native response_format /
tool_choice guard is covered in ``test_native_structured.py`` (gateway
integration). This file covers the NON-duplicative edges the spec calls out:

* MALFORMED JSON salvaged by ``json_repair`` on the PURE parse path (no
  gateway, no retry) — real LLM malformations: trailing commas, single-quoted
  keys/values, a prose preamble before the JSON object. These never reach the
  retry loop because ``_parse_all`` recovers them at the salvage stage.
* UNREPAIRABLE JSON (json_repair yields a dict missing a required field) →
  triggers the FEEDBACK retry, sending the parse error + schema back at
  temperature=0. A distinct malformation shape from the existing suite's
  ``{"name":"incomplete"}`` (schema-invalid dict): here the content is
  STRUCTURALLY unrepairable prose so json_repair cannot even coerce a dict.
* VALID JSON passes through UNCHANGED (the first parse stage short-circuits,
  no salvage, no retry, no gateway call).
* ``_capture_error`` reports a JSONDecodeError reason for unrepairable prose
  (the feedback prompt the model receives is actionable, not a generic
  "did not match").
* the native structured-output guard's DECISION helper
  (``is_anthropic_4x_or_newer``) — the structured_output module's own contract
  that decides whether the gateway must drop a competing ``tool_choice``. The
  gateway-side drop is integration-tested elsewhere; here the helper's
  pre-/post-4.x boundary is pinned so the guard fires for the right models.

No src/ file is modified.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from src.llm.structured_output import (
    StructuredOutputManager,
    build_native_response_format,
    is_anthropic_4x_or_newer,
)


class SampleOutput(BaseModel):
    name: str
    value: int


class _Resp:
    def __init__(self, content: str) -> None:
        self.content = content


class _ScriptedGateway:
    """Records acompletion calls; returns scripted contents in order."""

    def __init__(self, contents: list[str]) -> None:
        self._contents = list(contents)
        self.calls: list[dict[str, Any]] = []

    async def acompletion(
        self, messages: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> _Resp:
        self.calls.append({"messages": messages or [], "kwargs": kwargs})
        if not self._contents:
            return _Resp("")
        return _Resp(self._contents.pop(0))


_MESSAGES = [{"role": "user", "content": "extract the sample"}]


# ─── malformed JSON salvaged by json_repair (pure parse path) ────────────


class TestJsonRepairSalvage:
    """Real LLM malformations recovered by ``json_repair`` WITHOUT a gateway or
    retry — the salvage stage of ``_parse_all`` handles them before any
    feedback loop fires."""

    @pytest.mark.asyncio
    async def test_trailing_comma_salvaged(self) -> None:
        raw = '{"name": "trailing", "value": 3,}'
        result = await StructuredOutputManager().extract(raw, SampleOutput)
        assert result is not None
        assert result.name == "trailing"
        assert result.value == 3

    @pytest.mark.asyncio
    async def test_single_quoted_keys_salvaged(self) -> None:
        raw = "{'name': 'quoted', 'value': 5}"
        result = await StructuredOutputManager().extract(raw, SampleOutput)
        assert result is not None
        assert result.name == "quoted"
        assert result.value == 5

    @pytest.mark.asyncio
    async def test_prose_preamble_before_object_salvaged(self) -> None:
        """An LLM that emits a chatty preamble before the JSON object is
        salvaged — json_repair extracts the object from the surrounding prose."""
        raw = (
            "Here is the structured output you asked for:\n"
            '{"name": "preamble", "value": 11}\n'
            "Let me know if you need anything else."
        )
        result = await StructuredOutputManager().extract(raw, SampleOutput)
        assert result is not None
        assert result.name == "preamble"
        assert result.value == 11

    @pytest.mark.asyncio
    async def test_unclosed_brace_salvaged(self) -> None:
        """A truncated response with an unclosed object is repaired by
        json_repair (a common truncation failure mode)."""
        raw = '{"name": "unclosed", "value": 7'
        result = await StructuredOutputManager().extract(raw, SampleOutput)
        assert result is not None
        assert result.value == 7

    @pytest.mark.asyncio
    async def test_salvage_skips_gateway(self) -> None:
        """Salvaged JSON must NOT trigger a retry call even when a gateway IS
        supplied — the first salvage stage succeeds, so the feedback loop is
        never entered."""
        gw = _ScriptedGateway(['{"name": "unused", "value": 0}'])
        result = await StructuredOutputManager().extract(
            "{'name': 'salvage', 'value': 9}",
            SampleOutput,
            gateway=gw,
            messages=_MESSAGES,
            max_retries=3,
        )
        assert result is not None
        assert result.name == "salvage"
        assert gw.calls == []  # salvage succeeded → no retry


# ─── unrepairable JSON triggers feedback retry ───────────────────────────


class TestUnrepairableTriggersFeedbackRetry:
    """Content that ``json_repair`` CANNOT coerce into a schema-valid dict
    falls through to the feedback retry loop. The feedback prompt carries the
    parse error + the schema and is sent at temperature=0."""

    @pytest.mark.asyncio
    async def test_pure_prose_triggers_retry_then_recovers(self) -> None:
        """Pure prose (no object at all) is unrepairable → retry. The model's
        corrected reply is then parsed directly."""
        gw = _ScriptedGateway(['{"name": "fixed", "value": 4}'])
        result = await StructuredOutputManager().extract(
            "I cannot produce JSON for that request.",
            SampleOutput,
            gateway=gw,
            messages=_MESSAGES,
            max_retries=1,
        )
        assert result is not None
        assert result.name == "fixed"
        assert result.value == 4
        assert len(gw.calls) == 1
        # The feedback turn is the last message, sent to the model.
        feedback = gw.calls[0]["messages"][-1]
        assert feedback["role"] == "user"
        # The correction prompt carries a concrete parse-error reason + schema.
        assert "Parse error:" in feedback["content"]
        assert "Required JSON schema" in feedback["content"]

    @pytest.mark.asyncio
    async def test_unrepairable_then_still_bad_returns_none(self) -> None:
        """Retry still cannot produce valid JSON → None after max_retries."""
        gw = _ScriptedGateway(["still just prose, no object here"])
        result = await StructuredOutputManager().extract(
            "garbage prose no braces",
            SampleOutput,
            gateway=gw,
            messages=_MESSAGES,
            max_retries=1,
        )
        assert result is None
        assert len(gw.calls) == 1  # one retry, then give up

    @pytest.mark.asyncio
    async def test_feedback_prompt_carries_schema_and_temperature_zero(self) -> None:
        gw = _ScriptedGateway(['{"name": "ok", "value": 1}'])
        await StructuredOutputManager().extract(
            "nope", SampleOutput, gateway=gw, messages=_MESSAGES, max_retries=1
        )
        assert len(gw.calls) == 1
        feedback = gw.calls[0]["messages"][-1]["content"]
        # The schema is serialized into the correction prompt.
        assert "Required JSON schema" in feedback
        assert "name" in feedback  # the target model's field appears in the schema
        # temperature=0 is honored for determinism.
        assert gw.calls[0]["kwargs"].get("temperature") == 0.0

    @pytest.mark.asyncio
    async def test_retry_call_failure_is_swallowed(self) -> None:
        """If the retry ``acompletion`` itself raises, the manager returns None
        rather than propagating (a failed retry must not abort the caller)."""

        class _BoomGateway:
            def __init__(self) -> None:
                self.calls = 0

            async def acompletion(
                self, messages: list[dict[str, Any]] | None = None, **kw: Any
            ) -> _Resp:
                self.calls += 1
                raise RuntimeError("gateway exploded")

        gw = _BoomGateway()
        result = await StructuredOutputManager().extract(
            "nope", SampleOutput, gateway=gw, messages=_MESSAGES, max_retries=2
        )
        assert result is None
        assert gw.calls == 1  # the failing retry was attempted once


# ─── valid JSON passes through unchanged ─────────────────────────────────


class TestValidJsonPassThrough:
    @pytest.mark.asyncio
    async def test_valid_json_short_circuits_all_stages(self) -> None:
        """A valid JSON object matching the schema parses on the FIRST stage —
        no json_repair, no fence-strip, no retry, no gateway call."""
        gw = _ScriptedGateway(['{"name": "unused", "value": 0}'])
        raw = '{"name": "direct", "value": 42}'
        result = await StructuredOutputManager().extract(
            raw, SampleOutput, gateway=gw, messages=_MESSAGES, max_retries=3
        )
        assert result is not None
        assert result.name == "direct"
        assert result.value == 42
        assert gw.calls == []

    @pytest.mark.asyncio
    async def test_valid_json_extra_fields_ignored(self) -> None:
        """Extra unknown fields are silently dropped by Pydantic validation."""
        raw = '{"name": "x", "value": 1, "junk": true, "more": [1, 2]}'
        result = await StructuredOutputManager().extract(raw, SampleOutput)
        assert result is not None
        assert result.name == "x"
        assert result.value == 1


# ─── guarded-retry boundaries (no-gateway / no-messages / fenced retry) ────


class TestGuardedRetryBoundaries:
    """The ``extract`` guard (``gateway is None or not messages → None``) and
    the retry path's fence handling. These give-up branches were in the
    original suite and are restored here so a parse failure degrades gracefully
    and a fenced correction reply still parses."""

    @pytest.mark.asyncio
    async def test_no_gateway_yields_none_on_unrepairable(self) -> None:
        # No gateway → the feedback retry cannot fire; an unrepairable input
        # returns None immediately (the pure parse-or-fail back-compat path).
        result = await StructuredOutputManager().extract(
            "this is pure prose with no object", SampleOutput
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_gateway_without_messages_yields_none_and_never_calls(
        self,
    ) -> None:
        # A gateway is present but NO conversation messages → the guard short-
        # circuits to None WITHOUT invoking the gateway (the retry needs the
        # original messages to build the correction conversation).
        gw = _ScriptedGateway(['{"name": "unused", "value": 0}'])
        result = await StructuredOutputManager().extract(
            "pure prose no object", SampleOutput, gateway=gw, messages=None
        )
        assert result is None
        assert gw.calls == []

    @pytest.mark.asyncio
    async def test_fenced_json_returned_by_retry_is_parsed(self) -> None:
        # The retry call returns JSON wrapped in markdown fences; _parse_all
        # strips the fences before validating, so the recovered object parses.
        gw = _ScriptedGateway(['```json\n{"name": "late", "value": 8}\n```'])
        result = await StructuredOutputManager().extract(
            "nope", SampleOutput, gateway=gw, messages=_MESSAGES, max_retries=1
        )
        assert result is not None
        assert result.name == "late"
        assert result.value == 8
        assert len(gw.calls) == 1


# ─── native structured-output guard DECISION helper ──────────────────────


class TestNativeStructuredGuardDecision:
    """``is_anthropic_4x_or_newer`` is the structured_output module's own
    decision that gates the gateway's pre-emptive ``tool_choice`` drop: pre-4.x
    Anthropic forces tool-conversion for ``json_schema`` (conflicting with an
    explicit ``tool_choice`` → 400), while 4.x+ uses native ``output_format``
    (no conflict). The gateway-side drop is integration-tested in
    ``test_native_structured.py``; here the helper boundary is pinned."""

    def test_pre4x_anthropic_returns_false(self) -> None:
        assert not is_anthropic_4x_or_newer("claude-haiku-2-5")
        assert not is_anthropic_4x_or_newer("claude-sonnet-3-5-20241022")
        assert not is_anthropic_4x_or_newer("claude-opus-3")

    def test_4x_and_newer_returns_true(self) -> None:
        assert is_anthropic_4x_or_newer("claude-haiku-4-5-20251001")
        assert is_anthropic_4x_or_newer("claude-sonnet-4-6")
        assert is_anthropic_4x_or_newer("claude-opus-4-8")

    def test_non_anthropic_returns_false(self) -> None:
        assert not is_anthropic_4x_or_newer("gpt-4o-mini-2024-07-18")
        assert not is_anthropic_4x_or_newer("glm-4.7")
        assert not is_anthropic_4x_or_newer("deepseek-v4-flash")

    def test_no_schema_returns_json_object_mode(self) -> None:
        """The native response_format guard's other half: with NO schema, the
        builder returns provider-agnostic ``json_object`` mode (no
        ``tool_choice`` conflict to guard against)."""
        from types import SimpleNamespace

        out = build_native_response_format(None, "openai", SimpleNamespace(enabled=True))
        assert out == {"type": "json_object"}

    def test_disabled_returns_none(self) -> None:
        from types import SimpleNamespace

        assert (
            build_native_response_format(
                {"type": "object"}, "openai", SimpleNamespace(enabled=False)
            )
            is None
        )
