"""Tests for src.tools.builtin.image_generator — text→image via litellm.

Covers the prompt guard, the base64 happy path (file written into the isolated
results root), the URL-only fallback, filename sanitization (traversal
stripped, ``.png`` appended), and the failure modes (provider error, empty
data, undecodable base64, neither b64 nor url). No real network: litellm's
``aimage_generation`` is replaced with an in-test coroutine that records the
posted kwargs. Uses ``tmp_path`` as an isolated results root.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pytest

from src.config.settings import get_settings
from src.tools.builtin import image_generator as image_gen
from src.tools.builtin.image_generator import image_generator


class _Image:
    """Minimal stand-in for a litellm ImageObject (b64_json / url)."""

    def __init__(self, b64_json: str | None = None, url: str | None = None) -> None:
        self.b64_json = b64_json
        self.url = url


class _Response:
    """Minimal stand-in for a litellm ImageResponse (.data)."""

    def __init__(self, data: list[_Image]) -> None:
        self.data = data


def _patch_generation(monkeypatch: pytest.MonkeyPatch, response: _Response) -> dict[str, Any]:
    """Replace litellm.aimage_generation with a recorder returning ``response``.

    Returns the captured kwargs dict so a test can assert on the wire payload.
    """
    captured: dict[str, Any] = {}

    async def _fake(**kwargs: Any) -> _Response:
        captured.update(kwargs)
        return response

    monkeypatch.setattr(image_gen.litellm, "aimage_generation", _fake)
    return captured


def _patch_raising(monkeypatch: pytest.MonkeyPatch, exc: BaseException) -> None:
    """Replace litellm.aimage_generation with a coroutine that raises ``exc``."""

    async def _fake(**kwargs: Any) -> _Response:
        raise exc

    monkeypatch.setattr(image_gen.litellm, "aimage_generation", _fake)


@pytest.fixture
def results_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate writes to tmp_path by pointing results_root at it."""
    monkeypatch.setattr(get_settings().agent, "results_root", str(tmp_path))
    return tmp_path


class TestImageGeneratorGuards:
    @pytest.mark.asyncio
    async def test_empty_prompt_returns_error(self, results_root: Path) -> None:
        assert (await image_generator("")).startswith("ERROR: prompt is required")
        assert (await image_generator("   ")).startswith("ERROR: prompt is required")

    @pytest.mark.asyncio
    async def test_provider_error_is_sanitized(
        self, results_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_raising(monkeypatch, RuntimeError("upstream said api_key=sk-leaked"))
        result = await image_generator("a red cube")
        assert result.startswith("ERROR: image generation failed (RuntimeError)")
        # The raw exception text must not leak.
        assert "sk-leaked" not in result


class TestImageGeneratorHappyPath:
    @pytest.mark.asyncio
    async def test_b64_success_writes_png_and_returns_path(
        self, results_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        png_bytes = b"\x89PNG\r\n\x1a\nfake-png"
        resp = _Response([_Image(b64_json=base64.b64encode(png_bytes).decode())])
        captured = _patch_generation(monkeypatch, resp)

        result = await image_generator("a blue circle", filename="circle")

        assert result.startswith("Wrote generated image")
        # The file exists under the isolated results root with the .png suffix.
        out = results_root / "circle.png"
        assert out.exists()
        assert out.read_bytes() == png_bytes
        # Wire payload: model/prompt/n/response_format forwarded, defaults applied.
        assert captured["prompt"] == "a blue circle"
        assert captured["n"] == 1
        assert captured["response_format"] == "b64_json"
        assert captured["model"] == get_settings().tools.image_gen_model
        assert captured["size"] == get_settings().tools.image_gen_default_size
        assert captured["quality"] == get_settings().tools.image_gen_default_quality

    @pytest.mark.asyncio
    async def test_explicit_size_and_quality_override_defaults(
        self, results_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        resp = _Response([_Image(b64_json=base64.b64encode(b"x").decode())])
        captured = _patch_generation(monkeypatch, resp)

        await image_generator("wide art", size="1536x1024", quality="high")

        assert captured["size"] == "1536x1024"
        assert captured["quality"] == "high"

    @pytest.mark.asyncio
    async def test_filename_without_extension_gets_png(
        self, results_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        resp = _Response([_Image(b64_json=base64.b64encode(b"x").decode())])
        _patch_generation(monkeypatch, resp)

        await image_generator("x", filename="diagram")

        assert (results_root / "diagram.png").exists()

    @pytest.mark.asyncio
    async def test_random_filename_when_omitted(
        self, results_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        resp = _Response([_Image(b64_json=base64.b64encode(b"x").decode())])
        _patch_generation(monkeypatch, resp)

        result = await image_generator("x")

        # No caller filename → a random image_<hex>.png lands under results root.
        pngs = list(results_root.glob("image_*.png"))
        assert len(pngs) == 1
        # The returned path references that very file (robust to the results root
        # being a tmp dir that need not contain the literal "results").
        assert str(pngs[0]) in result

    @pytest.mark.asyncio
    async def test_unsupported_params_retry_strips_and_succeeds(
        self, results_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """gpt-image-1 rejects ``response_format`` (UnsupportedParamsError).

        The tool must strip the flagged param and retry once — gpt-image-1 returns
        base64 by default, so the b64 happy path still fires. Regression for the
        live-smoke wire bug (raw response_format hard-failed gpt-image-1).
        """
        png_bytes = b"\x89PNG\r\n\x1a\nretry-png"
        resp = _Response([_Image(b64_json=base64.b64encode(png_bytes).decode())])
        calls: list[dict[str, Any]] = []

        # A stand-in exception whose type name is UnsupportedParamsError (litellm
        # raises its own; we duck-type by name to avoid stub noise in the tool).
        class _UnsupportedParamsError(Exception):
            pass

        _UnsupportedParamsError.__name__ = "UnsupportedParamsError"

        async def _fake(**kwargs: Any) -> _Response:
            calls.append(dict(kwargs))
            if len(calls) == 1:
                # litellm names the offending param in backticks.
                raise _UnsupportedParamsError(
                    "Setting `response_format` is not supported by openai, gpt-image-1"
                )
            return resp

        monkeypatch.setattr(image_gen.litellm, "aimage_generation", _fake)

        result = await image_generator("a red square")

        assert result.startswith("Wrote generated image")
        assert len(calls) == 2  # one failed, one retry
        # First call carried response_format; the retry dropped it.
        assert "response_format" in calls[0]
        assert "response_format" not in calls[1]
        # gpt-image-1 returns base64 by default → the PNG was still written.
        pngs = list(results_root.glob("*.png"))
        assert len(pngs) == 1
        assert pngs[0].read_bytes() == png_bytes


class TestImageGeneratorFilenameSafety:
    @pytest.mark.asyncio
    async def test_traversal_filename_is_confined(
        self, results_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        resp = _Response([_Image(b64_json=base64.b64encode(b"x").decode())])
        _patch_generation(monkeypatch, resp)

        await image_generator("x", filename="../../escape.png")

        # Nothing escapes the results root; a sanitized file lands inside it.
        escaped = results_root.parent.parent / "escape.png"
        assert not escaped.exists()
        inside = [p for p in results_root.rglob("*.png")]
        assert len(inside) == 1
        assert inside[0].resolve().is_relative_to(results_root.resolve())


class TestImageGeneratorFallbacksAndFailures:
    @pytest.mark.asyncio
    async def test_url_only_response_returns_url_no_file(
        self, results_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        resp = _Response([_Image(url="https://cdn.example.com/img.png")])
        _patch_generation(monkeypatch, resp)

        result = await image_generator("x")

        assert result.startswith("Generated image URL")
        assert "https://cdn.example.com/img.png" in result
        # No file written when only a URL came back.
        assert not list(results_root.glob("*.png"))

    @pytest.mark.asyncio
    async def test_empty_data_returns_error(
        self, results_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured = _patch_generation(monkeypatch, _Response([]))
        captured.clear()  # response carries no data

        result = await image_generator("x")

        assert result == "ERROR: image generation returned no data"

    @pytest.mark.asyncio
    async def test_undecodable_b64_returns_error(
        self, results_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        resp = _Response([_Image(b64_json="@@@not-base64@@@")])
        _patch_generation(monkeypatch, resp)

        result = await image_generator("x")

        assert result.startswith("ERROR: image generation returned undecodable")

    @pytest.mark.asyncio
    async def test_neither_b64_nor_url_returns_error(
        self, results_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        resp = _Response([_Image()])
        _patch_generation(monkeypatch, resp)

        result = await image_generator("x")

        assert result == "ERROR: image generation returned neither an inline image nor a URL"
