"""REAL-LLM provider-capability E2E tests (real gateway, real provider calls).

Each test exercises a provider-native opt-in capability through the REAL
``LLMGateway`` (no mocking). These are gated behind the relevant env flags and
are cheap (small ``max_tokens``, trivial prompts).

Requires a provider API key in the environment (``OPENAI_API_KEY`` or
``DEEPSEEK_API_KEY``). The entire module SKIPS CLEAN when no provider key is
set — it never fails for a missing key.

Run with: ``python -m pytest tests/test_e2e/test_provider_capabilities_real.py -v -m e2e``
"""

from __future__ import annotations

from typing import Any

import pytest

from src.llm.gateway import LLMGateway
from src.llm.models import LLMResponse


def _provider_keys() -> dict[str, str]:
    """Return the provider keys the gateway actually uses (from ``.env``/settings).

    Keys live in ``.env`` and are loaded by pydantic-settings; ``os.environ``
    does NOT see them (the gateway reads settings, not the shell). So the skip
    guard must consult the same source the gateway does, or it falsely runs
    without a key (or falsely skips with one). Only key PRESENCE is ever
    inspected — values are never logged or compared.
    """
    from src.config import get_settings

    llm = get_settings().llm
    keys: dict[str, str] = {}
    for attr in (
        "openai_api_key",
        "deepseek_api_key",
        "zai_api_key",
        "gemini_api_key",
        "google_api_key",
        "nvidia_api_key",
    ):
        val = getattr(llm, attr, None)
        if isinstance(val, str) and val:
            keys[attr] = val
    return keys


def _has_any_provider_key() -> bool:
    return bool(_provider_keys())


def _has_openai() -> bool:
    return "openai_api_key" in _provider_keys()


def _has_deepseek() -> bool:
    return "deepseek_api_key" in _provider_keys()


# Module is @pytest.mark.e2e and skips cleanly when NO provider key is
# configured in the gateway's settings source (``.env``). The guard consults
# the SAME source the gateway uses (pydantic-settings ``.env``), NOT
# ``os.environ`` — keys live in ``.env``, so ``os.environ`` is unreliable here.
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not _has_any_provider_key(),
        reason=(
            "Requires at least one provider API key in .env "
            "(OPENAI/DEEPSEEK/ZAI/GEMINI/GOOGLE/NVIDIA) for real-LLM E2E tests"
        ),
    ),
]


def _native_structured_model() -> str | None:
    """Pick a cheap model whose provider supports native JSON-schema output.

    DeepSeek + OpenAI both support strict ``json_schema`` response_format
    (``build_native_response_format`` emits that shape for both providers).
    Returns ``None`` if no supporting key is available.
    """
    if _has_openai():
        return "gpt-4o-mini-2024-07-18"
    if _has_deepseek():
        return "deepseek-v4-flash"
    return None


@pytest.fixture
def settings() -> Any:
    """Fresh Settings instance (mutable) per test.

    ``get_settings()`` caches a singleton; we deep-copy it so per-test flag
    flips (native structured / tool retrieval) never leak across tests or into
    the process-wide cached instance.
    """
    from src.config import get_settings

    base = get_settings()
    # pydantic-settings models are mutable (not frozen); use model_copy so a
    # flipped nested flag does not mutate the cached singleton.
    return base.model_copy(deep=True)


@pytest.fixture
def gateway(settings: Any) -> LLMGateway:
    """Real LLMGateway built from the per-test (mutable) settings."""
    return LLMGateway(settings)


@pytest.fixture
def tools() -> Any:
    """Default tool registry (built-in tools)."""
    from src.tools import create_default_registry

    return create_default_registry()


class TestNativeStructuredOutput:
    """(a) Native structured output returns parseable JSON for a schema request.

    Gated behind ``NATIVE_STRUCTURED_OUTPUT_ENABLED``. With the flag ON and a
    caller-supplied ``response_schema``, the gateway emits a provider-native
    ``response_format`` and the model returns schema-conformant JSON.
    """

    @pytest.mark.asyncio
    async def test_native_structured_returns_parseable_json(
        self, settings: Any, gateway: LLMGateway
    ) -> None:
        """A schema request returns JSON parseable into the requested shape."""
        model = _native_structured_model()
        if model is None:
            pytest.skip("no native-structured-capable provider key available")

        # Flip the feature ON for this test only.
        settings.native_structured.enabled = True

        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "is_even": {"type": "boolean"},
            },
            "required": ["answer", "is_even"],
            "additionalProperties": False,
        }

        response = await gateway.acompletion(
            messages=[
                {
                    "role": "user",
                    "content": "Is the number 4 even? Answer in one word.",
                }
            ],
            model=model,
            response_schema=schema,
            temperature=0.0,
            max_tokens=128,
        )

        assert isinstance(response, LLMResponse)
        assert response.content, "native structured output must be non-empty"

        # The response must be parseable as JSON with the requested keys.
        import json

        try:
            parsed = json.loads(response.content)
        except json.JSONDecodeError as exc:
            # Some providers wrap JSON; strip code fences and retry once.
            stripped = response.content.strip().strip("`")
            stripped = stripped[stripped.find("{") : stripped.rfind("}") + 1]
            parsed = json.loads(stripped) if stripped else pytest.fail(
                f"native structured output was not parseable JSON: {exc}"
            )

        assert isinstance(parsed, dict), "parsed structured output must be an object"
        assert "answer" in parsed, "schema-required key 'answer' must be present"
        assert "is_even" in parsed, "schema-required key 'is_even' must be present"

    @pytest.mark.asyncio
    async def test_native_structured_disabled_when_flag_off(
        self, settings: Any, gateway: LLMGateway
    ) -> None:
        """With the flag OFF, ``response_schema`` is ignored (no hard failure).

        The call still succeeds; behavior is unchanged from prompt-based JSON
        (graceful degradation — the documented default-off contract).
        """
        model = _native_structured_model()
        if model is None:
            pytest.skip("no native-structured-capable provider key available")

        settings.native_structured.enabled = False  # explicit default

        schema = {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        }

        response = await gateway.acompletion(
            messages=[{"role": "user", "content": "Reply with: OK"}],
            model=model,
            response_schema=schema,  # ignored when disabled
            temperature=0.0,
            max_tokens=32,
        )

        assert isinstance(response, LLMResponse)
        # The call must not crash and must return SOME content (flag-off is a
        # graceful no-op on response_schema, not an error).
        assert response.content


class TestToolRetrievalSelection:
    """(b) A tool-retrieval-enabled run narrows the tool set by relevance.

    Gated behind ``TOOL_RETRIEVAL_ENABLED``. ``select_tools_for_query`` with the
    flag ON returns the built-in set ∪ top-k dynamically-created tools nearest
    the query embedding; with the flag OFF it returns the full registered set
    (identical to pre-retrieval behavior). We assert the OFF→full behavior
    deterministically (no embedding dependency) and that the ON path is invoked
    without raising (graceful full-set fallback when no API embedding exists).
    """

    @pytest.mark.asyncio
    async def test_retrieval_disabled_returns_full_set(
        self, settings: Any, tools: Any
    ) -> None:
        """Flag OFF ⇒ ``select_tools_for_query`` returns the full registry."""
        from src.tools.selection import select_tools_for_query

        settings.agent.tool_retrieval_enabled = False

        selected = await select_tools_for_query(
            query="search the web for news",
            registry=tools,
            settings=settings,
        )

        full = tools.list_tools()
        assert isinstance(selected, list)
        assert len(selected) == len(full), (
            "tool retrieval OFF must return the FULL registered set "
            "(identical to pre-retrieval behavior)"
        )

    @pytest.mark.asyncio
    async def test_retrieval_enabled_never_raises(
        self, settings: Any, tools: Any
    ) -> None:
        """Flag ON ⇒ selection runs without raising (graceful full-set fallback).

        The ON path needs a real API embedding to rank; when one is unavailable
        (e.g. no embedding-provider key) it MUST fall back to the full set
        rather than raising — retrieval can never break a run.
        """
        from src.tools.selection import select_tools_for_query

        settings.agent.tool_retrieval_enabled = True

        # Must not raise regardless of embedding availability.
        selected = await select_tools_for_query(
            query="compute statistics on a dataset",
            registry=tools,
            settings=settings,
        )

        assert isinstance(selected, list)
        assert len(selected) >= 1, (
            "tool retrieval ON must return at least the built-in set (full fallback)"
        )


class TestSubAgentDelegation:
    """(c) Sub-agent spawn + delegate (guarded behind the delegation feature).

    A registered ``SubAgentSpec`` is spawned via ``SubAgentRegistry.spawn`` (no
    DB) and executed via ``SubAgentRunner.run``. The run returns the documented
    result dict shape. ``agent_selection_enabled`` is left at its default
    (OFF ⇒ all-fan-out), exercising the spawn+delegate seam without the
    embedding-based selector. This is a real-LLM delegation: the sub-agent's
    subgraph drives a real completion through the gateway.
    """

    @pytest.mark.asyncio
    async def test_sub_agent_spawn_and_run_returns_result_dict(
        self, settings: Any, gateway: LLMGateway, tools: Any
    ) -> None:
        """Spawn a registered sub-agent and run a trivial subtask through it."""
        from src.agents.registry import SubAgentRegistry
        from src.graph.models import SubAgentSpec

        registry = SubAgentRegistry()
        spec = SubAgentSpec(
            name="echo-explainer",
            description="Explains a concept in one short sentence.",
            goal="Explain the concept briefly.",
            parent_thread_id="thread-real-subagent-001",
            max_iterations=3,
        )
        registry.register(spec)

        runner = registry.spawn(
            name="echo-explainer",
            goal="Explain what an integer is in one short sentence.",
            parent_thread_id="thread-real-subagent-001",
            gateway=gateway,
            tools=tools,
            memory=None,
        )
        assert runner is not None, "spawn must return a runner for a registered active spec"

        result = await runner.run(
            goal="Explain what an integer is in one short sentence.",
            parent_thread_id="thread-real-subagent-001",
        )

        # Documented result-dict contract (runner.run return shape).
        assert isinstance(result, dict)
        for key in ("success", "result", "goal", "sub_agent_name"):
            assert key in result, f"runner.run result must include {key!r}"
        assert result["sub_agent_name"] == "echo-explainer"
        assert result["goal"]  # the subtask goal round-trips

    @pytest.mark.asyncio
    async def test_spawn_returns_none_for_unknown_agent(
        self, settings: Any, gateway: LLMGateway, tools: Any
    ) -> None:
        """spawn() returns None (not raise) for an unregistered sub-agent name."""
        from src.agents.registry import SubAgentRegistry

        registry = SubAgentRegistry()
        runner = registry.spawn(
            name="does-not-exist",
            goal="anything",
            parent_thread_id="thread-real-subagent-002",
            gateway=gateway,
            tools=tools,
            memory=None,
        )
        assert runner is None, "spawn of an unknown sub-agent must return None, not raise"
