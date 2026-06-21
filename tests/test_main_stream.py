"""``--stream`` final-answer flag — streams the final deliverable token-by-token.

Pins:
* the ``main`` click command's ``stream`` param defaults to False,
* ``_stream_final_answer`` prints chunks in arrival order,
* ``_stream_final_answer`` falls back to the static ``final_output`` when the
  stream raises (generic exception OR ``asyncio.CancelledError``) and never
  propagates.
"""

from __future__ import annotations

import asyncio

import click.testing
import pytest

import main as main_mod


class _StreamGateway:
    """Minimal stand-in for ``LLMGateway`` exposing only ``astream``."""

    def __init__(self, chunks: list[str] | None = None, exc: BaseException | None = None):
        self._chunks = chunks
        self._exc = exc

    async def astream(self, messages):  # noqa: ANN001 — matches gateway signature shape
        if self._exc is not None:
            raise self._exc
        for chunk in self._chunks or []:
            yield chunk


class TestStreamFlagDefault:
    def test_stream_flag_defaults_off(self) -> None:
        """The click command's stream param default is False."""
        cmd = main_mod.main
        # click stores param defaults on the command's params list.
        stream_param = next(p for p in cmd.params if p.name == "stream")
        assert stream_param.is_flag is True
        assert stream_param.default is False


class TestStreamFinalAnswer:
    @pytest.mark.asyncio
    async def test_stream_final_answer_prints_chunks_in_order(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        gateway = _StreamGateway(chunks=["Hello", ", ", "world!"])
        result = {"final_output": "static fallback", "iteration_count": 3, "is_complete": True}

        await main_mod._stream_final_answer(gateway, "explain X", result)

        captured = capsys.readouterr()
        assert captured.out == "Hello, world!\n"

    @pytest.mark.asyncio
    async def test_stream_final_answer_falls_back_on_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        gateway = _StreamGateway(exc=RuntimeError("boom"))
        result = {"final_output": "static fallback", "iteration_count": 1, "is_complete": False}

        # Must NOT propagate.
        await main_mod._stream_final_answer(gateway, "explain X", result)

        captured = capsys.readouterr()
        assert captured.out == "static fallback\n"

    @pytest.mark.asyncio
    async def test_stream_final_answer_handles_cancelled_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        gateway = _StreamGateway(exc=asyncio.CancelledError())
        result = {"final_output": "static fallback", "iteration_count": 1, "is_complete": False}

        # Must NOT propagate the CancelledError.
        await main_mod._stream_final_answer(gateway, "explain X", result)

        captured = capsys.readouterr()
        assert captured.out == "static fallback\n"

    @pytest.mark.asyncio
    async def test_stream_final_answer_falls_back_when_gateway_none(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = {"final_output": "no gateway output", "iteration_count": 0, "is_complete": False}

        await main_mod._stream_final_answer(None, "explain X", result)

        captured = capsys.readouterr()
        assert captured.out == "no gateway output\n"

    @pytest.mark.asyncio
    async def test_stream_final_answer_falls_back_on_empty_stream(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # An AsyncMock-style gateway whose astream yields nothing.
        gateway = _StreamGateway(chunks=[])
        result = {"final_output": "fallback", "iteration_count": 0, "is_complete": False}

        await main_mod._stream_final_answer(gateway, "explain X", result)

        captured = capsys.readouterr()
        assert captured.out == "fallback\n"

    @pytest.mark.asyncio
    async def test_stream_final_answer_filters_empty_tokens(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Real providers emit empty-string delta chunks; they must not be printed
        # but also must not prevent the trailing newline from being emitted.
        gateway = _StreamGateway(chunks=["A", "", "B"])
        result = {"final_output": "fb", "iteration_count": 0, "is_complete": False}

        await main_mod._stream_final_answer(gateway, "explain X", result)

        captured = capsys.readouterr()
        assert captured.out == "AB\n"


def test_main_stream_option_is_registered() -> None:
    """``--stream`` is a registered flag on the click command and off by default."""
    runner = click.testing.CliRunner()
    # Invoke with --help so no agent run occurs; just confirm the option parses.
    result = runner.invoke(main_mod.main, ["--help"])
    assert result.exit_code == 0
    assert "--stream" in result.output
