"""Regression tests for ``web_scraper`` extraction (Phase 1) + Bug A.

Bug A: ``markdownify`` (an OPTIONAL HTML→markdown fallback for pages trafilatura
can't parse) used to be imported EAGERLY at module top. When the dep was absent
from the image — which it was, having been omitted from ``requirements.txt`` —
importing the builtin-tool registry crashed graph build with ``ModuleNotFoundError``
on EVERY run, before any node executed (the worker's run executor died at iter 0).
The fix imports markdownify DEFENSIVELY so a missing optional fallback never
blocks the agent; these tests lock that the module imports + extracts without the
fallback, and that the fallback is only consulted when it is actually available.
"""

from __future__ import annotations

import pytest

import src.tools.builtin.web_scraper as ws


class TestMarkdownifyOptional:
    """The optional markdownify fallback must never crash graph build (Bug A)."""

    def test_module_imports_with_optional_fallback_attrs(self) -> None:
        """Importing the module exposes an availability flag + a (maybe-None) fn.

        ``_markdownify`` is the imported fn when installed, ``None`` when absent —
        either way the name exists, so no ``NameError`` reaches ``_extract_markdown``.
        """
        assert isinstance(ws._MARKDOWNIFY_AVAILABLE, bool)
        assert ws._markdownify is None or callable(ws._markdownify)

    def test_extract_markdown_works_with_fallback_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Simulate markdownify missing (flag off + fn None): ``_extract_markdown``
        must still return a str (trafilatura's result, or "" for unparseable input)
        and never raise — the exact regression for the eager-import crash."""
        monkeypatch.setattr(ws, "_MARKDOWNIFY_AVAILABLE", False)
        monkeypatch.setattr(ws, "_markdownify", None)

        # trafilatura-readable page → non-empty markdown via the PRIMARY path.
        good = (
            "<html><body><article><h1>Title</h1>"
            "<p>Hello readable main content here.</p></article></body></html>"
        )
        assert isinstance(ws._extract_markdown(good), str)

        # trafilatura-unreadable input → "" (fallback absent, no crash, no NameError).
        assert ws._extract_markdown("") == ""

    def test_fallback_not_invoked_when_flag_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the flag off, the (absent) fallback fn is never called even when
        trafilatura yields nothing — guarding against a bare ``markdownify(html)``
        reference that would ``NameError``/``TypeError`` on the missing dep."""
        monkeypatch.setattr(ws, "_MARKDOWNIFY_AVAILABLE", False)
        calls = {"n": 0}

        def _would_crash(_html: str) -> str:
            calls["n"] += 1
            return "fallback"

        monkeypatch.setattr(ws, "_markdownify", _would_crash)
        # Empty input → trafilatura yields nothing → reaches the fallback branch,
        # but the flag guard skips the call and returns "".
        assert ws._extract_markdown("") == ""
        assert calls["n"] == 0
