"""src.memory.facts — durable-fact extraction (Phase 5).

``extract_facts`` is the pure-ish extraction step (gateway call + json_repair +
Pydantic validation). It is exercised with a fake gateway (never a real LLM):
happy path, markdown-fence stripping, caps, malformed-item resilience, and the
never-raises contract (empty text / gateway outage / non-object JSON → []).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.memory.facts import FactCandidate, extract_facts, fact_extraction_prompt


class _FakeGateway:
    def __init__(self, content: str | Exception) -> None:
        if isinstance(content, Exception):
            self.acompletion = AsyncMock(side_effect=content)
        else:
            self.acompletion = AsyncMock(
                return_value=SimpleNamespace(content=content)
            )


class TestFactExtractionPrompt:
    def test_prompt_mentions_json_shape_and_limit(self) -> None:
        prompt = fact_extraction_prompt("some summary", max_facts=4)
        assert "facts" in prompt
        assert "RUN SUMMARY" in prompt
        assert "4" in prompt  # the cap is interpolated
        # No bare braces that would break if this were ever templated.
        assert "facts" in prompt


class TestExtractFacts:
    @pytest.mark.asyncio
    async def test_valid_json_returns_candidates(self) -> None:
        content = (
            '{"facts": ['
            '{"key": "row_count", "value": "1024 rows", "confidence": 0.9}, '
            '{"key": "tz", "value": "all UTC ISO-8601"}'
            "]}"
        )
        facts = await extract_facts(_FakeGateway(content), "summary", max_facts=5)
        assert [f.key for f in facts] == ["row_count", "tz"]
        assert facts[0].value == "1024 rows"
        assert facts[0].confidence == pytest.approx(0.9)
        # Missing confidence defaults to 0.5.
        assert facts[1].confidence == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_fenced_json_is_parsed(self) -> None:
        content = (
            "```json\n"
            '{"facts": [{"key": "schema", "value": "id,ts,amount", "confidence": 0.8}]}'
            "\n```"
        )
        facts = await extract_facts(_FakeGateway(content), "summary")
        assert len(facts) == 1
        assert facts[0].key == "schema"

    @pytest.mark.asyncio
    async def test_max_facts_caps_output(self) -> None:
        raw = [{"key": f"k{i}", "value": f"v{i}"} for i in range(7)]
        facts = await extract_facts(
            _FakeGateway('{"facts": ' + str(raw).replace("'", '"') + "}"),
            "summary",
            max_facts=3,
        )
        assert len(facts) == 3

    @pytest.mark.asyncio
    async def test_empty_text_returns_empty(self) -> None:
        gw = _FakeGateway('{"facts": [{"key": "x", "value": "y"}]}')
        assert await extract_facts(gw, "") == []
        assert await extract_facts(gw, "   ") == []

    @pytest.mark.asyncio
    async def test_gateway_error_returns_empty(self) -> None:
        # Never raises — a gateway outage yields no facts.
        facts = await extract_facts(
            _FakeGateway(RuntimeError("provider down")), "summary"
        )
        assert facts == []

    @pytest.mark.asyncio
    async def test_non_object_json_returns_empty(self) -> None:
        gw = _FakeGateway("[1, 2, 3]")  # a list, not an object with "facts"
        assert await extract_facts(gw, "summary") == []

    @pytest.mark.asyncio
    async def test_malformed_item_skipped_rest_kept(self) -> None:
        # 2nd item has no value → dropped; the valid ones survive.
        content = (
            '{"facts": ['
            '{"key": "good", "value": "ok"}, '
            '{"key": "bad"}, '
            '{"key": "also", "value": "fine", "confidence": "not-a-number"}'
            "]}"
        )
        facts = await extract_facts(_FakeGateway(content), "summary")
        assert [f.key for f in facts] == ["good"]


class TestFactCandidate:
    def test_candidate_validates_confidence_range(self) -> None:
        with pytest.raises(ValueError):
            FactCandidate(key="k", value="v", confidence=1.5)
        with pytest.raises(ValueError):
            FactCandidate(key="k", value="v", confidence=-0.1)
        # In-range is accepted.
        ok = FactCandidate(key="k", value="v", confidence=0.7)
        assert ok.confidence == pytest.approx(0.7)

    def test_candidate_defaults(self) -> None:
        c = FactCandidate(key="k", value="v")
        assert c.confidence == pytest.approx(0.5)
        assert isinstance(c, FactCandidate)
