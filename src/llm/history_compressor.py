"""History compressor — truncates older messages to reduce token consumption.

Adapted from GenericAgent's ``compress_history_tags()`` pattern.
Keeps the N most recent messages intact and truncates content in older
messages to a configurable maximum length. Runs every ``interval`` calls
to avoid overhead on every gateway invocation.
"""

from __future__ import annotations

from typing import Any


class HistoryCompressor:
    """Compresses older messages in conversation history.

    Args:
        keep_recent: Number of recent messages to keep intact.
        max_content_length: Maximum character length for truncated content.
        interval: Run compression every N calls (1 = every call).
    """

    def __init__(
        self,
        keep_recent: int = 10,
        max_content_length: int = 800,
        interval: int = 5,
    ) -> None:
        self._keep_recent = keep_recent
        self._max_content_length = max_content_length
        self._interval = max(1, interval)
        self._call_count = 0

    def compress(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Compress older messages, keeping recent ones intact.

        Only runs every ``interval`` calls to amortize overhead. Short
        message lists (< ``keep_recent``) are returned unchanged.

        Args:
            messages: Chat messages in OpenAI format.

        Returns:
            Potentially compressed message list (new list, not in-place).
        """
        self._call_count += 1
        if self._call_count % self._interval != 0:
            return messages

        if len(messages) <= self._keep_recent:
            return messages

        compressed: list[dict[str, Any]] = []
        cutoff = len(messages) - self._keep_recent
        for i, msg in enumerate(messages):
            if i >= cutoff:
                compressed.append(msg)
            else:
                compressed.append(self._compress_message(msg))

        return compressed

    def _compress_message(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Truncate long content in a single message.

        Args:
            msg: A single chat message dict.

        Returns:
            A new dict with potentially truncated content.
        """
        msg = dict(msg)  # shallow copy
        content = msg.get("content")

        if isinstance(content, str):
            msg["content"] = self._truncate(content)
        elif isinstance(content, list):
            msg["content"] = [self._compress_block(b) for b in content]

        return msg

    def _compress_block(self, block: dict[str, Any]) -> dict[str, Any]:
        """Compress a content block (text, tool_result, tool_use).

        Args:
            block: A content block dict from a message.

        Returns:
            A new dict with potentially truncated values.
        """
        block = dict(block)
        btype = block.get("type", "")

        if btype == "text" and isinstance(block.get("text"), str):
            block["text"] = self._truncate(block["text"])
        elif btype == "tool_result":
            content = block.get("content")
            if isinstance(content, str):
                block["content"] = self._truncate(content)
        elif btype == "tool_use" and isinstance(block.get("input"), dict):
            input_dict = dict(block["input"])
            for k, v in input_dict.items():
                if isinstance(v, str) and len(v) > self._max_content_length:
                    input_dict[k] = self._truncate(v)
            block["input"] = input_dict

        return block

    def _truncate(self, text: str) -> str:
        """Truncate text to max_content_length with a marker.

        Args:
            text: The text to potentially truncate.

        Returns:
            Truncated text with ``...[Truncated]...`` marker, or original.
        """
        if len(text) <= self._max_content_length:
            return text
        half = self._max_content_length // 2
        return text[:half] + "\n...[Truncated]...\n" + text[-half:]

    def reset(self) -> None:
        """Reset the call counter (useful for testing)."""
        self._call_count = 0
