"""#15 / D2 — gateway multimodal (vision) coverage.

Pins the D2 vision path at the gateway layer WITHOUT a live provider:

* the pure message-block builders — ``build_content_blocks``,
  ``_attach_images_to_last_user``, ``_content_char_len`` (block-list tolerant);
* the ``acompletion`` multimodal wiring — ``vision_enabled`` + ``images`` ⇒
  ``require_vision`` ⇒ the last user turn is folded into OpenAI ``image_url``
  content blocks BEFORE the cache lookup (so the payload participates in the
  cache key), and stays byte-identical to the text-only path when off/no images;
* the fail-safe fallback-chain filter — when a vision payload is attached, the
  chain is restricted to ``ModelSpec.supports_images`` models so a text-only
  fallback never silently drops the images; if the filter would empty the chain,
  the ORIGINAL chain is kept (degraded text attempt > immediate raise).

A real-provider ``@pytest.mark.e2e`` variant exercises the live vision path with
a tiny image (skips without a provider key).
"""

from __future__ import annotations

import base64
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config.settings import AgentSettings, Settings
from src.llm.gateway import (
    LLMGateway,
    _attach_images_to_last_user,
    _content_char_len,
    build_content_blocks,
)


# ─── Shared helpers (mirrors test_gateway.py's construction) ─────────


def _vision_settings(*, vision: bool = True) -> Settings:
    """A Settings whose agent.vision_enabled is pinned (robust to live .env)."""
    return Settings(agent=AgentSettings(vision_enabled=vision))


def _make_litellm_response(
    content: str | None = "ok",
    input_tokens: int = 4,
    output_tokens: int = 2,
) -> MagicMock:
    """A minimal litellm.ModelResponse-shaped mock."""
    message = MagicMock()
    message.content = content
    message.tool_calls = None
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"
    usage = MagicMock()
    usage.prompt_tokens = input_tokens
    usage.completion_tokens = output_tokens
    usage.total_tokens = input_tokens + output_tokens
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def _make_gateway(vision: bool = True) -> LLMGateway:
    """A gateway with mocked internals, no cache/cost-tracker/Redis touches."""
    with patch.object(LLMGateway, "_configure_litellm"):
        gw = LLMGateway(_vision_settings(vision=vision))
    # No-op rate limiter so tests never block; null cache + cost tracker so the
    # cache lookup / budget check are skipped (no Redis/DB in unit tests).
    gw._rate_limiter = MagicMock()
    gw._rate_limiter.acquire = AsyncMock(return_value=None)
    gw._cache = None
    gw._cost_tracker = None
    return gw


def _make_litellm_mock(mock_resp: MagicMock) -> MagicMock:
    """A litellm module double with a successful acompletion + error types set.

    Returned so each test can ``patch("src.llm.gateway.litellm", mock)`` and read
    ``mock.acompletion.call_args`` after the awaited call.
    """
    mock_litellm = MagicMock()
    mock_litellm.acompletion = AsyncMock(return_value=mock_resp)
    mock_litellm.Usage = MagicMock
    for err in (
        "RateLimitError",
        "Timeout",
        "ServiceUnavailableError",
        "APIConnectionError",
        "AuthenticationError",
        "BadRequestError",
    ):
        setattr(mock_litellm, err, Exception)
    return mock_litellm


# ─── build_content_blocks ────────────────────────────────────────────


class TestBuildContentBlocks:
    def test_no_images_returns_plain_text_unchanged(self) -> None:
        """The text-only path must be byte-identical — no wrapping, no list."""
        assert build_content_blocks("describe this", None) == "describe this"
        assert build_content_blocks("describe this", []) == "describe this"

    def test_single_image_produces_text_plus_image_block(self) -> None:
        blocks = build_content_blocks("describe this", ["data:image/png;base64,AAAA"])
        assert isinstance(blocks, list)
        assert blocks[0] == {"type": "text", "text": "describe this"}
        assert blocks[1] == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AAAA"},
        }

    def test_multiple_images_one_block_each_in_order(self) -> None:
        blocks = build_content_blocks("x", ["https://a/1.png", "https://a/2.png"])
        assert isinstance(blocks, list)
        image_urls = [b["image_url"]["url"] for b in blocks if b["type"] == "image_url"]
        assert image_urls == ["https://a/1.png", "https://a/2.png"]

    def test_falsy_and_non_str_entries_are_dropped(self) -> None:
        """None / "" / non-str entries are skipped; valid ones survive."""
        images: list[Any] = [None, "", 123, "https://keep.png"]
        blocks = build_content_blocks("x", images)
        assert isinstance(blocks, list)
        image_urls = [b["image_url"]["url"] for b in blocks if b["type"] == "image_url"]
        assert image_urls == ["https://keep.png"]


# ─── _attach_images_to_last_user ─────────────────────────────────────


class TestAttachImagesToLastUser:
    def test_folds_into_last_user_only(self) -> None:
        msgs: list[dict[str, Any]] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "describe this image"},
        ]
        out = _attach_images_to_last_user(msgs, ["data:image/png;base64,AAAA"])
        # Only the FINAL user turn gains the block form.
        assert isinstance(out[3]["content"], list)
        assert out[3]["content"][0]["type"] == "text"
        assert out[3]["content"][1]["type"] == "image_url"
        # Earlier turns are untouched.
        assert out[0]["content"] == "sys"
        assert out[1]["content"] == "first question"
        assert out[2]["content"] == "answer"

    def test_original_message_list_is_not_mutated(self) -> None:
        msgs: list[dict[str, Any]] = [{"role": "user", "content": "orig"}]
        _ = _attach_images_to_last_user(msgs, ["data:image/png;base64,AAAA"])
        assert msgs[0]["content"] == "orig"

    def test_no_user_message_returns_unchanged(self) -> None:
        """A vision request with no prompt is malformed — fail safe (drop, no raise)."""
        msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
        out = _attach_images_to_last_user(msgs, ["data:image/png;base64,AAAA"])
        assert out == msgs
        assert out[0]["content"] == "sys"

    def test_existing_list_content_collapsed_to_text_then_blocks(self) -> None:
        """A user message already in block-list form is flattened to text first so
        the image_url blocks append cleanly (no nested lists)."""
        msgs: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "part one"},
                    {"type": "text", "text": "part two"},
                ],
            }
        ]
        out = _attach_images_to_last_user(msgs, ["data:image/png;base64,AAAA"])
        content = out[0]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "part one\npart two"}
        assert content[1]["type"] == "image_url"


# ─── _content_char_len (block-list tolerant) ─────────────────────────


class TestContentCharLen:
    def test_string_content(self) -> None:
        assert _content_char_len("hello") == 5

    def test_block_list_counts_text_plus_flat_image_allowance(self) -> None:
        from src.llm.gateway import _IMAGE_FLAT_CHARS

        content = [
            {"type": "text", "text": "ab"},  # 2
            {"type": "image_url", "image_url": {"url": "x"}},  # flat
            {"type": "image_url", "image_url": {"url": "y"}},  # flat
        ]
        assert _content_char_len(content) == 2 + (_IMAGE_FLAT_CHARS * 2)

    def test_non_str_non_list_is_zero_and_never_raises(self) -> None:
        assert _content_char_len(None) == 0
        assert _content_char_len(12345) == 0


# ─── acompletion multimodal wiring (mocked litellm) ──────────────────


class TestAcompletionVisionWiring:
    @pytest.mark.asyncio
    async def test_vision_on_with_images_folds_blocks_into_last_user(
        self,
    ) -> None:
        gw = _make_gateway(vision=True)
        messages = [
            {"role": "system", "content": "you are a vision model"},
            {"role": "user", "content": "describe this"},
        ]
        mock_litellm = _make_litellm_mock(_make_litellm_response())
        with patch("src.llm.gateway.litellm", mock_litellm):
            await gw.acompletion(
                messages=messages,
                model="gpt-4o-mini-2024-07-18",
                images=["data:image/png;base64,AAAA"],
            )
        sent = mock_litellm.acompletion.call_args.kwargs["messages"]
        # The last user message reaching litellm is the multimodal block list.
        assert isinstance(sent[1]["content"], list)
        assert sent[1]["content"][-1] == {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AAAA"},
        }
        # The system turn is untouched.
        assert sent[0]["content"] == "you are a vision model"

    @pytest.mark.asyncio
    async def test_vision_off_leaves_messages_byte_identical(self) -> None:
        """vision_enabled=False (the default) must NOT touch the messages even when
        images are supplied — the text-only path is unchanged."""
        gw = _make_gateway(vision=False)
        messages = [{"role": "user", "content": "describe this"}]
        mock_litellm = _make_litellm_mock(_make_litellm_response())
        with patch("src.llm.gateway.litellm", mock_litellm):
            await gw.acompletion(
                messages=messages,
                model="gpt-4o-mini-2024-07-18",
                images=["data:image/png;base64,AAAA"],
            )
        sent = mock_litellm.acompletion.call_args.kwargs["messages"]
        assert sent == [{"role": "user", "content": "describe this"}]

    @pytest.mark.asyncio
    async def test_no_images_leaves_messages_text_only_even_when_vision_on(
        self,
    ) -> None:
        gw = _make_gateway(vision=True)
        messages = [{"role": "user", "content": "plain text prompt"}]
        mock_litellm = _make_litellm_mock(_make_litellm_response())
        with patch("src.llm.gateway.litellm", mock_litellm):
            await gw.acompletion(
                messages=messages,
                model="gpt-4o-mini-2024-07-18",
            )
        sent = mock_litellm.acompletion.call_args.kwargs["messages"]
        assert sent == [{"role": "user", "content": "plain text prompt"}]


# ─── fallback-chain vision filter (mocked litellm) ───────────────────


class TestVisionChainFilter:
    @pytest.mark.asyncio
    async def test_filter_restricts_to_vision_capable_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A text-only primary (deepseek-v4-flash) with a vision-capable fallback
        (gpt-4o-mini): with images+vision the chain is restricted so ONLY the
        vision model is attempted — the text-only primary is never tried (it
        would silently drop the images)."""
        import src.llm.gateway as gw_mod

        monkeypatch.setitem(
            gw_mod.FALLBACK_CHAINS,
            "deepseek-v4-flash",
            ["gpt-4o-mini-2024-07-18"],
        )
        gw = _make_gateway(vision=True)
        messages = [{"role": "user", "content": "describe"}]
        mock_litellm = _make_litellm_mock(_make_litellm_response())
        with patch("src.llm.gateway.litellm", mock_litellm):
            await gw.acompletion(
                messages=messages,
                model="deepseek-v4-flash",
                images=["data:image/png;base64,AAAA"],
            )
        attempted = [
            call.kwargs["model"] for call in mock_litellm.acompletion.call_args_list
        ]
        assert attempted, "expected at least one litellm attempt"
        # Every attempt must be the vision-capable model; deepseek never reached.
        assert all("deepseek" not in m for m in attempted)
        assert any("gpt-4o-mini" in m for m in attempted)

    @pytest.mark.asyncio
    async def test_filter_fail_safe_keeps_chain_when_no_vision_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A text-only primary with ONLY text-only fallbacks: the vision filter
        would empty the chain ⇒ fail-safe keeps the original chain and warns (a
        degraded text attempt is preferable to raising)."""
        import src.llm.gateway as gw_mod

        monkeypatch.setitem(
            gw_mod.FALLBACK_CHAINS,
            "deepseek-v4-flash",
            ["deepseek-v4-pro"],
        )
        gw = _make_gateway(vision=True)
        messages = [{"role": "user", "content": "describe"}]
        mock_litellm = _make_litellm_mock(_make_litellm_response())
        # Spy on the module-level loguru logger (caplog can't see loguru). The
        # fail-safe path emits a "No vision-capable model" WARNING.
        with (
            patch("src.llm.gateway.litellm", mock_litellm),
            patch.object(gw_mod.logger, "warning") as warn_spy,
        ):
            await gw.acompletion(
                messages=messages,
                model="deepseek-v4-flash",
                images=["data:image/png;base64,AAAA"],
            )
        attempted = [
            call.kwargs["model"] for call in mock_litellm.acompletion.call_args_list
        ]
        # Fail-safe: the original primary IS attempted (no vision model exists).
        assert attempted
        assert "deepseek" in attempted[0]
        # And the fail-safe path logged the warning.
        assert warn_spy.called
        assert any(
            "No vision-capable model" in str(call.args)
            for call in warn_spy.call_args_list
        )


# ─── E2E: real-provider vision call (skips without a key) ────────────


# A 1×1 transparent PNG (smallest valid image payload) as a data URI so the E2E
# test never depends on a network fetch for the image itself.
_PNG_1X1_B64 = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000d49444154789c63000100000005000100" "0d" "0a" "a1a6e600"
        "00000049454e44ae426082"
    )
).decode("ascii")
_PNG_1X1_DATA_URI = f"data:image/png;base64,{_PNG_1X1_B64}"


@pytest.mark.e2e
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="Requires OPENAI_API_KEY for the E2E vision call",
)
class TestVisionE2E:
    @pytest.mark.asyncio
    async def test_real_vision_call_returns_nonempty(self) -> None:
        """Live gpt-4o-mini vision round-trip: a tiny image + a trivial prompt
        must return a non-empty textual answer (semantic assertion, not exact)."""
        from src.config import get_settings

        settings = get_settings()
        settings.agent.vision_enabled = True
        gateway = LLMGateway(settings)
        response = await gateway.acompletion(
            messages=[{"role": "user", "content": "Reply with the single word OK."}],
            model="gpt-4o-mini-2024-07-18",
            images=[_PNG_1X1_DATA_URI],
            max_tokens=16,
        )

        assert response.content is not None
        assert str(response.content).strip() != ""
