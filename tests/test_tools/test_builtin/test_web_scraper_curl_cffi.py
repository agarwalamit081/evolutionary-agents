"""Unit tests for the curl_cffi TLS-impersonation anti-bot tier (S14, Gap 7).

The scraper fetches via ``httpx`` first; on an anti-bot signal (HTTP 403/429)
it retries once through a Chrome-impersonated ``curl_cffi`` session to bypass
Cloudflare/bot-WAF TLS blocking. These tests are deterministic — no network:
httpx is driven by ``httpx.MockTransport`` (returns 403/429/404) and curl_cffi
is replaced by a fake session object, so we assert *which path was taken* and
that the byte cap + error-surfacing contracts hold.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from src.tools.builtin import web_scraper as ws


# ── fakes + patches ───────────────────────────────────────────────────────


class _FakeResponse:
    """Stand-in for a curl_cffi Response (status/content/headers only)."""

    def __init__(self, status_code: int, content: bytes = b"", headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}


class _FakeSession:
    """Records its construction + the GET, returns a canned _FakeResponse."""

    def __init__(self, response: _FakeResponse, sink: list[dict[str, Any]], impersonate: str) -> None:
        self._response = response
        self._sink = sink
        self.impersonate = impersonate

    def __enter__(self) -> "_FakeSession":
        return self

    def __exit__(self, *_a: object) -> bool:
        return False

    def get(self, url: str, **kwargs: Any) -> _FakeResponse:
        self._sink.append({"url": url, "kwargs": kwargs})
        return self._response


def _install_curl_cffi(
    monkeypatch: pytest.MonkeyPatch,
    response: _FakeResponse,
    sink: list[dict[str, Any]],
    impersonate: str = "chrome",
) -> None:
    """Wire a fake curl_cffi.requests module + mark it available."""

    class _FakeRequests:
        @staticmethod
        def Session(impersonate: str = "chrome", **_kw: Any) -> _FakeSession:
            return _FakeSession(response, sink, impersonate)

    monkeypatch.setattr(ws, "_curl_cffi_requests", _FakeRequests)
    monkeypatch.setattr(ws, "_CURL_CFFI_AVAILABLE", True)


def _limits(monkeypatch: pytest.MonkeyPatch, **over: Any) -> None:
    base = dict(
        web_scraper_curl_cffi_enabled=True,
        web_scraper_curl_cffi_impersonate="chrome",
    )
    base.update(over)
    monkeypatch.setattr(ws, "_tool_limits", lambda: SimpleNamespace(**base))


def _patch_httpx(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    """Inject a MockTransport into every httpx.Client the scraper builds."""
    real_client = ws.httpx.Client

    def _factory(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(ws.httpx, "Client", _factory)


def _status_handler(status: int) -> Any:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status)

    return handler


# ── retry-triggered / bypass path ─────────────────────────────────────────


class TestAntiBotRetry:
    @pytest.mark.asyncio  # kept uniform with the suite's async style; runs sync code
    async def test_403_triggers_curl_cffi_and_returns_html(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_httpx(monkeypatch, _status_handler(403))
        sink: list[dict[str, Any]] = []
        _install_curl_cffi(monkeypatch, _FakeResponse(200, content=b"<html>OK</html>"), sink)
        _limits(monkeypatch)

        html = ws._fetch_html("https://example.com/page", timeout=5.0, max_bytes=1024)
        assert html == "<html>OK</html>"
        assert len(sink) == 1  # curl_cffi retried exactly once
        assert sink[0]["url"] == "https://example.com/page"
        # Chrome impersonation + timeout + redirect-following all reached the call.
        assert sink[0]["kwargs"]["timeout"] == 5.0
        assert sink[0]["kwargs"]["allow_redirects"] is True

    @pytest.mark.asyncio
    async def test_429_triggers_curl_cffi(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_httpx(monkeypatch, _status_handler(429))
        sink: list[dict[str, Any]] = []
        _install_curl_cffi(monkeypatch, _FakeResponse(200, content=b"<x/>"), sink)
        _limits(monkeypatch)

        assert ws._fetch_html("https://example.com/p", timeout=5.0, max_bytes=1024) == "<x/>"
        assert len(sink) == 1


# ── no-retry contracts ────────────────────────────────────────────────────


class TestNoRetry:
    @pytest.mark.asyncio
    async def test_404_does_not_invoke_curl_cffi(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A genuine 404 is not a TLS block → no anti-bot retry, error surfaces."""
        _patch_httpx(monkeypatch, _status_handler(404))
        sink: list[dict[str, Any]] = []
        _install_curl_cffi(monkeypatch, _FakeResponse(200, content=b"<x/>"), sink)
        _limits(monkeypatch)

        with pytest.raises(ws._FetchError) as exc:
            ws._fetch_html("https://example.com/missing", timeout=5.0, max_bytes=1024)
        assert "HTTP 404" in str(exc.value)
        assert sink == []  # curl_cffi never touched

    @pytest.mark.asyncio
    async def test_disabled_falls_through_without_retry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_httpx(monkeypatch, _status_handler(403))
        sink: list[dict[str, Any]] = []
        _install_curl_cffi(monkeypatch, _FakeResponse(200, content=b"<x/>"), sink)
        _limits(monkeypatch, web_scraper_curl_cffi_enabled=False)

        with pytest.raises(ws._FetchError) as exc:
            ws._fetch_html("https://example.com/p", timeout=5.0, max_bytes=1024)
        assert "HTTP 403" in str(exc.value)
        assert sink == []

    @pytest.mark.asyncio
    async def test_unavailable_falls_through_without_retry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """curl_cffi not importable → degrade to httpx-only, no crash, error surfaces."""
        _patch_httpx(monkeypatch, _status_handler(403))
        sink: list[dict[str, Any]] = []
        _install_curl_cffi(monkeypatch, _FakeResponse(200, content=b"<x/>"), sink)
        _limits(monkeypatch)
        monkeypatch.setattr(ws, "_CURL_CFFI_AVAILABLE", False)

        with pytest.raises(ws._FetchError) as exc:
            ws._fetch_html("https://example.com/p", timeout=5.0, max_bytes=1024)
        assert "HTTP 403" in str(exc.value)
        assert sink == []

    @pytest.mark.asyncio
    async def test_curl_cffi_still_blocked_surfaces_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If curl_cffi is ALSO blocked (403), surface the block — no silent masking."""
        _patch_httpx(monkeypatch, _status_handler(403))
        sink: list[dict[str, Any]] = []
        _install_curl_cffi(monkeypatch, _FakeResponse(403), sink)
        _limits(monkeypatch)

        with pytest.raises(ws._FetchError) as exc:
            ws._fetch_html("https://example.com/p", timeout=5.0, max_bytes=1024)
        assert "HTTP 403" in str(exc.value)
        assert len(sink) == 1  # retried, but the retry was also blocked


# ── byte-cap contracts on the curl_cffi path ──────────────────────────────


class TestCurlCffiByteCap:
    @pytest.mark.asyncio
    async def test_oversize_body_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_httpx(monkeypatch, _status_handler(403))
        sink: list[dict[str, Any]] = []
        _install_curl_cffi(monkeypatch, _FakeResponse(200, content=b"x" * 100), sink)
        _limits(monkeypatch)

        with pytest.raises(ws._FetchError) as exc:
            ws._fetch_html("https://example.com/p", timeout=5.0, max_bytes=10)
        assert "download cap" in str(exc.value)

    @pytest.mark.asyncio
    async def test_content_length_pre_check_rejects(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Content-Length > cap → rejected without trusting the (small) body."""
        _patch_httpx(monkeypatch, _status_handler(403))
        sink: list[dict[str, Any]] = []
        _install_curl_cffi(
            monkeypatch,
            _FakeResponse(200, content=b"tiny", headers={"Content-Length": "1000000"}),
            sink,
        )
        _limits(monkeypatch)

        with pytest.raises(ws._FetchError) as exc:
            ws._fetch_html("https://example.com/p", timeout=5.0, max_bytes=10)
        assert "download cap" in str(exc.value)
