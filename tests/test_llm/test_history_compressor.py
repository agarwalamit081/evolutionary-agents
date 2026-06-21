"""Tests for src.llm.history_compressor — message history truncation."""

from __future__ import annotations

from typing import Any

from src.llm.history_compressor import HistoryCompressor


def _make_messages(count: int, content_length: int = 50) -> list[dict[str, Any]]:
    """Create a list of simple text messages."""
    return [
        {"role": "user", "content": "x" * content_length}
        for _ in range(count)
    ]


class TestHistoryCompressorBasic:
    """Tests for basic compression behavior."""

    def test_short_history_unchanged(self) -> None:
        """Messages shorter than keep_recent are returned unchanged."""
        compressor = HistoryCompressor(keep_recent=10, interval=1)
        msgs = _make_messages(5)
        result = compressor.compress(msgs)
        assert result == msgs

    def test_single_message_unchanged(self) -> None:
        """A single message is never compressed."""
        compressor = HistoryCompressor(keep_recent=10, interval=1)
        msgs = [{"role": "user", "content": "hello"}]
        result = compressor.compress(msgs)
        assert result == msgs

    def test_empty_messages_unchanged(self) -> None:
        """Empty message list is returned unchanged."""
        compressor = HistoryCompressor(interval=1)
        result = compressor.compress([])
        assert result == []

    def test_compresses_older_messages(self) -> None:
        """Messages beyond keep_recent are truncated when too long."""
        compressor = HistoryCompressor(keep_recent=2, max_content_length=20, interval=1)
        msgs = _make_messages(5, content_length=200)
        result = compressor.compress(msgs)

        # Last 2 messages should be unchanged
        assert result[-1]["content"] == "x" * 200
        assert result[-2]["content"] == "x" * 200

        # First 3 messages should be truncated
        for i in range(3):
            assert len(result[i]["content"]) < 200
            assert "[Truncated]" in result[i]["content"]

    def test_does_not_modify_original(self) -> None:
        """Compression returns a new list, not modifying the original."""
        compressor = HistoryCompressor(keep_recent=1, max_content_length=10, interval=1)
        msgs = [{"role": "user", "content": "a" * 100}]
        original_content = msgs[0]["content"]
        compressor.compress(msgs)
        assert msgs[0]["content"] == original_content


class TestHistoryCompressorInterval:
    """Tests for interval-based compression."""

    def test_interval_controls_compression(self) -> None:
        """Compression only runs every N calls."""
        compressor = HistoryCompressor(keep_recent=1, max_content_length=10, interval=3)
        msgs = _make_messages(3, content_length=200)

        # Call 1: no compression
        result1 = compressor.compress(msgs)
        assert all(m["content"] == "x" * 200 for m in result1)

        # Call 2: no compression
        result2 = compressor.compress(msgs)
        assert all(m["content"] == "x" * 200 for m in result2)

        # Call 3: compression runs
        result3 = compressor.compress(msgs)
        assert result3[-1]["content"] == "x" * 200  # recent kept
        assert "[Truncated]" in result3[0]["content"]  # old compressed

    def test_reset_resets_counter(self) -> None:
        """reset() resets the call counter."""
        compressor = HistoryCompressor(interval=5)
        compressor._call_count = 4
        compressor.reset()
        assert compressor._call_count == 0


class TestHistoryCompressorBlocks:
    """Tests for structured content block compression."""

    def test_tool_result_content_truncated(self) -> None:
        """tool_result blocks with long content are truncated."""
        compressor = HistoryCompressor(keep_recent=0, max_content_length=20, interval=1)
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_result", "content": "y" * 200},
                ],
            }
        ]
        result = compressor.compress(msgs)
        assert "[Truncated]" in result[0]["content"][0]["content"]

    def test_tool_use_input_truncated(self) -> None:
        """tool_use blocks with long input values are truncated."""
        compressor = HistoryCompressor(keep_recent=0, max_content_length=20, interval=1)
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "input": {"code": "z" * 200},
                    },
                ],
            }
        ]
        result = compressor.compress(msgs)
        assert "[Truncated]" in result[0]["content"][0]["input"]["code"]

    def test_text_block_truncated(self) -> None:
        """Text blocks within structured content are truncated."""
        compressor = HistoryCompressor(keep_recent=0, max_content_length=20, interval=1)
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "w" * 200},
                ],
            }
        ]
        result = compressor.compress(msgs)
        assert "[Truncated]" in result[0]["content"][0]["text"]

    def test_short_content_not_truncated(self) -> None:
        """Content within max_content_length is not truncated."""
        compressor = HistoryCompressor(keep_recent=0, max_content_length=200, interval=1)
        msgs = _make_messages(3, content_length=50)
        result = compressor.compress(msgs)
        assert all(m["content"] == "x" * 50 for m in result)
