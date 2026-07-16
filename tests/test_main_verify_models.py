"""SI-3 — ``--verify-models`` CLI flag (``main._run_verify_models``).

Smoke-tests each named model over the gateway's real routing
(``gateway.acompletion`` → registry ``_build_kwargs``), superseding the two
deleted scripts (``smoke_llm.py`` + ``verify_alibaba_models.py``). A passing
ping means the registry routing is live end-to-end, not merely unit-tested.

These tests mock the gateway (no real provider call) and assert:
  - the exit-code contract (0 = all healthy, 1 = any failed)
  - the default model set is used when none are named
  - failure output is sanitized (NO key value, generic ``sk-`` scrubbed)
  - ``_scrub_secrets`` redacts enumerated keys + generic sk-tokens + truncates
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Callable

import click.testing
import pytest

import main as main_mod
from main import _scrub_secrets


def _ok_resp(model: str) -> SimpleNamespace:
    return SimpleNamespace(
        provider="alibaba",
        model=model,
        input_tokens=4,
        output_tokens=1,
        cost_usd=0.00002,
        content="pong",
    )


def _patch_gateway(
    monkeypatch: pytest.MonkeyPatch, behavior: Callable[[str], object]
) -> None:
    """Replace ``src.llm.gateway.LLMGateway`` with a controllable fake.

    ``behavior(model)`` either returns a response object (healthy) or raises
    (failure); it is awaited inside the handler's ``acompletion`` call.
    """

    class _FakeGateway:
        def __init__(self, settings: object) -> None:
            self.settings = settings

        async def acompletion(  # noqa: PLR0913 — mirrors the real signature
            self,
            *,
            messages: list[dict[str, str]],
            model: str,
            temperature: float,
            max_tokens: int,
        ) -> object:
            return behavior(model)

    # The handler lazy-imports ``LLMGateway`` inside the function body, so
    # patching the module attribute is seen at import-execution time.
    monkeypatch.setattr("src.llm.gateway.LLMGateway", _FakeGateway)


class TestScrubSecrets:
    def test_redacts_enumerated_key(self) -> None:
        secret = "dashed-key-value-1234567890"
        out = _scrub_secrets(f"err: {secret} bad", secret)
        assert secret not in out
        assert "<redacted>" in out

    def test_redacts_generic_sk_token(self) -> None:
        out = _scrub_secrets("err: sk-AbCdEf1234567890 leaked", "not-the-key")
        assert "sk-AbCdEf1234567890" not in out
        assert "<redacted>" in out

    def test_truncates_and_flattens_newlines(self) -> None:
        out = _scrub_secrets("a\nb\n" + "x" * 300)
        assert "\n" not in out
        assert len(out) <= 200

    def test_plain_message_passthrough(self) -> None:
        assert _scrub_secrets("AuthenticationError: bad key") == (
            "AuthenticationError: bad key"
        )

    def test_empty_key_is_skipped_not_matched(self) -> None:
        # An empty/unset key must not cause a no-op replace to mask real scrubbing.
        out = _scrub_secrets("err: sk-Leak1234567890xyz", "")
        assert "sk-Leak1234567890xyz" not in out


class TestRunVerifyModels:
    def test_all_healthy_returns_0(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_gateway(monkeypatch, _ok_resp)
        result = click.testing.CliRunner().invoke(
            main_mod.main,
            ["--verify-models", "--verify-model", "qwen3.5-flash"],
        )
        assert result.exit_code == 0
        assert "[OK]" in result.output
        assert "1/1 healthy" in result.output
        # No secrets in the healthy line — only public provider/model/tokens.
        assert "provider=alibaba" in result.output

    def test_any_failure_returns_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fail(_model: str) -> object:
            raise RuntimeError("boom from provider")

        _patch_gateway(monkeypatch, fail)
        result = click.testing.CliRunner().invoke(
            main_mod.main,
            ["--verify-models", "--verify-model", "qwen3.5-flash"],
        )
        assert result.exit_code == 1
        assert "[FAIL]" in result.output
        assert "0/1 healthy" in result.output
        assert "boom from provider" in result.output

    def test_default_models_when_none_named(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The default set is derived from the live tier maps, not a hardcoded
        # tuple, so it tracks primary swaps automatically. Assert the real
        # routing primaries + the #783 cross-provider fallback are all smoked.
        defaults = main_mod._default_smoke_models()
        assert "glm-5.2" in defaults  # COMPLEX/CRITICAL + plan/execute primary
        assert "openrouter-glm-5-2" in defaults  # #783 cross-provider fallback
        assert len(defaults) >= 4  # ≥ one primary per tier + default + fallback

        seen: list[str] = []

        def record(model: str) -> object:
            seen.append(model)
            return _ok_resp(model)

        _patch_gateway(monkeypatch, record)
        result = click.testing.CliRunner().invoke(main_mod.main, ["--verify-models"])
        assert result.exit_code == 0
        assert set(seen) == set(defaults)
        assert f"{len(defaults)}/{len(defaults)} healthy" in result.output

    def test_mixed_pass_fail_returns_1(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        defaults = main_mod._default_smoke_models()
        # Fail exactly one model guaranteed to be in the derived default set.
        fail_model = "glm-5.2"
        assert fail_model in defaults  # guards the test premise

        def mixed(model: str) -> object:
            if model == fail_model:
                raise RuntimeError("timeout")
            return _ok_resp(model)

        _patch_gateway(monkeypatch, mixed)
        result = click.testing.CliRunner().invoke(main_mod.main, ["--verify-models"])
        assert result.exit_code == 1
        expected_ok = len(defaults) - 1
        assert f"{expected_ok}/{len(defaults)} healthy" in result.output

    def test_failure_output_scrubs_secrets(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        secret = "sk-SUPERSECRETtoken123456"

        def leak(_model: str) -> object:
            raise RuntimeError(f"provider said: {secret} invalid")

        _patch_gateway(monkeypatch, leak)
        result = click.testing.CliRunner().invoke(
            main_mod.main,
            ["--verify-models", "--verify-model", "qwen3.5-flash"],
        )
        assert result.exit_code == 1
        assert secret not in result.output
        assert "<redacted>" in result.output
