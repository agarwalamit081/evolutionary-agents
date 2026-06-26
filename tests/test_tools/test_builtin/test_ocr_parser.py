"""Tests for src.tools.builtin.ocr_parser — GLM-OCR (Z.AI layout_parsing) tool.

Covers the safety guards (missing key, exactly-one-input, path traversal,
missing file, unsupported extension) and the happy paths for both input
shapes (sandboxed file_path → base64, public url → passthrough), plus the
HTTP-failure and payload-edge cases. No real network: httpx.AsyncClient is
replaced with an in-test stand-in, and assert_public_host is stubbed for the
url path. Uses tmp_path as an isolated sandbox root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from src.config.settings import get_settings
from src.tools.builtin import ocr_parser as ocr_mod
from src.tools.builtin.ocr_parser import ocr_parser


class _FakeResponse:
    """Minimal stand-in for an httpx.Response (status_code + json())."""

    def __init__(self, status_code: int, payload: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, Any]:
        if self._payload is None:
            raise ValueError("non-JSON response")
        return self._payload


def _fake_async_client(response: _FakeResponse) -> tuple[type, dict[str, Any]]:
    """Build an httpx.AsyncClient stand-in returning ``response`` from .post.

    Returns (client_class, captured) where ``captured`` records the last
    (url, headers, json) posted, so a test can assert on the wire payload.
    """
    captured: dict[str, Any] = {}

    class _Client:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *exc: Any) -> bool:
            return False

        async def post(self, url: str, headers: Any = None, json: Any = None) -> _FakeResponse:
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return response

    return _Client, captured


def _fake_async_client_raising(exc: BaseException) -> type:
    """Build an httpx.AsyncClient stand-in whose .post raises ``exc``."""

    class _Client:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> "_Client":
            return self

        async def __aexit__(self, *exc_: Any) -> bool:
            return False

        async def post(self, url: str, headers: Any = None, json: Any = None) -> _FakeResponse:
            raise exc

    return _Client


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the OCR sandbox root at tmp_path and give it a fake ZAI key."""
    monkeypatch.setattr(get_settings().agent, "workspace_root", str(tmp_path))
    monkeypatch.setattr(get_settings().llm, "zai_api_key", "test-zai-key")
    return tmp_path


class TestOcrParserGuards:
    @pytest.mark.asyncio
    async def test_missing_key_returns_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(get_settings().llm, "zai_api_key", "")
        result = await ocr_parser(file_path="page.png")
        assert result.startswith("ERROR: ZAI_API_KEY")

    @pytest.mark.asyncio
    async def test_neither_input_returns_error(self, sandbox: Path) -> None:
        result = await ocr_parser()
        assert "exactly one of file_path or url" in result

    @pytest.mark.asyncio
    async def test_both_inputs_returns_error(self, sandbox: Path) -> None:
        result = await ocr_parser(file_path="page.png", url="https://example.com/a.png")
        assert "exactly one of file_path or url" in result

    @pytest.mark.asyncio
    async def test_file_traversal_blocked(self, sandbox: Path) -> None:
        result = await ocr_parser(file_path="../../escape.png")
        assert "path traversal" in result

    @pytest.mark.asyncio
    async def test_missing_file_returns_error(self, sandbox: Path) -> None:
        result = await ocr_parser(file_path="nope.png")
        assert "file not found" in result

    @pytest.mark.asyncio
    async def test_unsupported_extension_returns_error(self, sandbox: Path) -> None:
        (sandbox / "doc.txt").write_text("hello", encoding="utf-8")
        result = await ocr_parser(file_path="doc.txt")
        assert "unsupported type" in result


class TestOcrParserHappyPath:
    @pytest.mark.asyncio
    async def test_file_path_success_base64_payload(
        self, sandbox: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (sandbox / "page.png").write_bytes(b"\x89PNG\r\n\x1a\nfake-png-bytes")
        client_cls, captured = _fake_async_client(
            _FakeResponse(200, {"md_results": "# Heading\n\nOCR text", "layout_details": []})
        )
        monkeypatch.setattr(ocr_mod.httpx, "AsyncClient", client_cls)

        result = await ocr_parser(file_path="page.png")

        assert result == "# Heading\n\nOCR text"
        # The file input was base64-encoded into the API body, not passed as-is.
        import base64

        assert captured["json"]["model"] == "glm-ocr"
        assert captured["json"]["file"] == base64.b64encode(b"\x89PNG\r\n\x1a\nfake-png-bytes").decode()
        assert captured["headers"]["Authorization"] == "Bearer test-zai-key"
        assert captured["url"] == ocr_mod._ZAI_LAYOUT_PARSING_URL

    @pytest.mark.asyncio
    async def test_url_success_passes_url_through(
        self, sandbox: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Stub the SSRF guard so no real DNS resolution runs.
        monkeypatch.setattr(ocr_mod, "assert_public_host", lambda url: None)
        client_cls, captured = _fake_async_client(_FakeResponse(200, {"md_results": "url text"}))
        monkeypatch.setattr(ocr_mod.httpx, "AsyncClient", client_cls)

        result = await ocr_parser(url="https://example.com/scan.png")

        assert result == "url text"
        # A url input is forwarded verbatim, NOT base64-encoded.
        assert captured["json"]["file"] == "https://example.com/scan.png"

    @pytest.mark.asyncio
    async def test_truncates_long_markdown(self, sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (sandbox / "page.png").write_bytes(b"png")
        long_md = "x" * 5000
        client_cls, _ = _fake_async_client(_FakeResponse(200, {"md_results": long_md}))
        monkeypatch.setattr(ocr_mod.httpx, "AsyncClient", client_cls)

        result = await ocr_parser(file_path="page.png", max_chars=100)

        assert result.startswith("x" * 100)
        assert "truncated at 100 chars" in result
        assert len(result) < len(long_md)


class TestOcrParserFailures:
    @pytest.mark.asyncio
    async def test_non_200_returns_sanitized_error(
        self, sandbox: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (sandbox / "page.png").write_bytes(b"png")
        client_cls, _ = _fake_async_client(_FakeResponse(401, {"error": "bad key"}))
        monkeypatch.setattr(ocr_mod.httpx, "AsyncClient", client_cls)

        result = await ocr_parser(file_path="page.png")

        # Sanitized: status only, no body/headers leaked.
        assert result == "ERROR: GLM-OCR returned HTTP 401"

    @pytest.mark.asyncio
    async def test_empty_md_results_returns_error(
        self, sandbox: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (sandbox / "page.png").write_bytes(b"png")
        client_cls, _ = _fake_async_client(_FakeResponse(200, {"md_results": "   "}))
        monkeypatch.setattr(ocr_mod.httpx, "AsyncClient", client_cls)

        result = await ocr_parser(file_path="page.png")
        assert "no extractable text" in result

    @pytest.mark.asyncio
    async def test_timeout_returns_error(
        self, sandbox: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (sandbox / "page.png").write_bytes(b"png")
        monkeypatch.setattr(
            ocr_mod.httpx, "AsyncClient", _fake_async_client_raising(httpx.TimeoutException("slow"))
        )

        result = await ocr_parser(file_path="page.png")
        assert result.startswith("ERROR: GLM-OCR request timed out")

    @pytest.mark.asyncio
    async def test_non_json_response_returns_error(
        self, sandbox: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (sandbox / "page.png").write_bytes(b"png")
        client_cls, _ = _fake_async_client(_FakeResponse(200, payload=None))
        monkeypatch.setattr(ocr_mod.httpx, "AsyncClient", client_cls)

        result = await ocr_parser(file_path="page.png")
        assert "non-JSON response" in result
