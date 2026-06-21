"""Structured output extraction with multi-fallback JSON parsing."""

from __future__ import annotations

import re
from typing import Any, Protocol, Type, TypeVar

import json
import json_repair
from loguru import logger
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


# Claude major-version matcher: ``claude-{haiku|sonnet|opus}-{major}``. 4.x+
# supports native ``output_format`` JSON without forcing tool_choice; older
# families route through tool-conversion (which competes with tool_choice).
_ANTHROPIC_VERSION = re.compile(r"claude-(?:haiku|sonnet|opus)-(\d+)")


def is_anthropic_4x_or_newer(model: str) -> bool:
    """True for Claude 4.x+ (native ``output_format``; no forced tool_choice)."""
    match = _ANTHROPIC_VERSION.search(model)
    return bool(match) and int(match.group(1)) >= 4


def build_native_response_format(
    schema: dict[str, Any] | None,
    provider: str,
    settings: Any,
    *,
    schema_name: str = "response",
) -> dict[str, Any] | None:
    """Build a provider-native ``response_format`` for strict JSON output.

    Returns ``None`` when the feature is disabled
    (``NativeStructuredSettings.enabled``) so callers degrade gracefully to
    prompt-based JSON. With no schema, returns JSON-object mode
    (``{"type": "json_object"}``, provider-agnostic). With a schema, returns a
    ``json_schema`` response_format for the providers litellm translates
    (OpenAI/DeepSeek strict passthrough, Anthropic ``output_format``, Gemini
    ``response_schema``); other providers fall back to json_object.

    litellm routes the result per provider (verified against 1.83.14). The
    gateway's pre-emptive guard (``_execute_with_fallback``) drops a competing
    ``tool_choice`` for pre-4.x Anthropic, where json_schema forces
    tool-conversion; existing recovery catches any residual 400.
    """
    if not getattr(settings, "enabled", False):
        return None
    if not schema:
        return {"type": "json_object"}
    if provider in ("openai", "deepseek"):
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema,
                "strict": True,
            },
        }
    if provider in ("anthropic", "google"):
        # Anthropic output_format and Gemini response_schema have no "strict"
        # flag; omit it to avoid a provider-specific 400.
        return {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema},
        }
    return {"type": "json_object"}


class CompletionModel(Protocol):
    """Minimal async-completion contract ``extract`` needs for retry feedback.

    Structural (duck-typed): any object with an ``acompletion`` returning a
    response exposing ``.content`` conforms — notably ``LLMGateway``, but also
    lightweight test doubles. Avoids importing ``LLMGateway`` here (which would
    create an import cycle, since the gateway owns a ``StructuredOutputManager``).
    """

    async def acompletion(
        self, messages: list[dict[str, Any]], *, temperature: float | None = None
    ) -> Any: ...

# Instruction template appended to messages when requesting structured output
_STRUCTURED_INSTRUCTION = (
    "\n\nIMPORTANT: Respond with a valid JSON object matching this schema. "
    "Do NOT include markdown code fences. Output ONLY the JSON object."
)


class StructuredOutputManager:
    """Extracts typed Pydantic models from LLM responses.

    Strategy: JSON mode → json_repair → retry with feedback.
    """

    def __init__(self) -> None:
        self._repair_attempts = 0

    async def extract(
        self,
        raw_content: str,
        output_model: Type[T],
        *,
        gateway: CompletionModel | None = None,
        messages: list[dict[str, Any]] | None = None,
        max_retries: int = 1,
    ) -> T | None:
        """Extract and validate a Pydantic model from raw LLM content.

        Parse pipeline: direct ``model_validate_json`` → ``json.loads`` →
        ``json_repair`` salvage → strip markdown fences. If every parse stage
        fails AND a ``gateway`` plus the original ``messages`` are supplied, the
        manager re-prompts the model with the parse error and the target schema
        (retry-with-feedback) up to ``max_retries`` times at ``temperature=0``
        before giving up. Without a gateway this is a pure parse-or-fail
        returning ``None`` (back-compat with every existing caller).

        Args:
            raw_content: The raw text content from the LLM response.
            output_model: The Pydantic model class to parse into.
            gateway: Optional async-completion model for retry-with-feedback on
                parse failure. When ``None`` (the default), no LLM call is made.
            messages: The original prompt messages (OpenAI format). Required for
                retry-with-feedback; the bad response + a feedback turn are
                appended to build the correction conversation.
            max_retries: Max retry-with-feedback LLM calls (default 1).

        Returns:
            Validated Pydantic model instance, or None on failure.
        """
        result, error = self._parse_all(raw_content, output_model)
        if result is not None:
            return result

        # No gateway/conversation → pure parse-or-fail (back-compat path).
        if gateway is None or not messages:
            logger.warning(
                f"Failed to extract {output_model.__name__} from response "
                f"(first 200 chars): {raw_content[:200]}"
            )
            return None

        # Retry-with-feedback: send the parse error + schema back to the model.
        schema_str = json.dumps(output_model.model_json_schema())
        conversation = [*messages, {"role": "assistant", "content": raw_content}]
        attempts = max(1, int(max_retries))
        for attempt in range(1, attempts + 1):
            feedback = (
                "Your previous response could not be parsed as valid JSON for the "
                "target schema.\n\n"
                f"Parse error: {error}\n\n"
                f"Required JSON schema:\n{schema_str}\n\n"
                "Re-emit ONLY a single valid JSON object matching the schema. "
                "No prose, no explanations, no markdown code fences."
            )
            try:
                response = await gateway.acompletion(
                    messages=[*conversation, {"role": "user", "content": feedback}],
                    temperature=0.0,
                )
            except Exception as exc:  # noqa: BLE001 — a failed retry must not abort the caller
                logger.warning(f"Structured-output retry call failed on attempt {attempt}: {exc}")
                return None
            content = getattr(response, "content", "") or ""
            result, error = self._parse_all(content, output_model)
            if result is not None:
                logger.info(
                    f"Structured-output retry-with-feedback recovered {output_model.__name__} "
                    f"on attempt {attempt}"
                )
                return result

        logger.warning(
            f"Structured-output retry-with-feedback exhausted after {attempts} attempt(s) "
            f"for {output_model.__name__}"
        )
        return None

    def _parse_all(self, content: str, model: Type[T]) -> tuple[T | None, str | None]:
        """Run every parse stage; return (result, last_error).

        The error is the most informative failure reason (validation error or
        JSON-decode error) so the retry-with-feedback loop can show the model
        exactly what was wrong. Stages mirror the original ``extract`` logic:
        direct validate → ``json.loads`` → ``json_repair`` → strip fences.
        """
        result = self._try_parse(content, model)
        if result is not None:
            return result, None

        try:
            repaired = json_repair.loads(content)
            if isinstance(repaired, dict):
                result = self._try_parse_dict(repaired, model)
                if result is not None:
                    self._repair_attempts += 1
                    logger.debug(f"JSON repair succeeded (total repairs: {self._repair_attempts})")
                    return result, None
        except Exception as exc:  # noqa: BLE001 — json_repair is best-effort
            logger.debug(f"json_repair failed: {exc}")

        stripped = self._strip_code_fences(content)
        if stripped != content:
            result = self._try_parse(stripped, model)
            if result is not None:
                return result, None

        return None, self._capture_error(content, model)

    @staticmethod
    def _capture_error(content: str, model: Type[T]) -> str:
        """Summarize WHY content failed to parse, for the feedback prompt."""
        try:
            model.model_validate_json(content)
        except (ValidationError, ValueError) as exc:
            return f"{type(exc).__name__}: {str(exc)[:300]}"
        try:
            decoded = json.loads(content)
        except json.JSONDecodeError as exc:
            return f"JSONDecodeError: {exc.msg}"
        if not isinstance(decoded, dict):
            return f"expected a JSON object, got {type(decoded).__name__}"
        return "content did not match the target schema"

    def _try_parse(self, content: str, model: Type[T]) -> T | None:
        """Try to parse content string into a Pydantic model."""
        try:
            return model.model_validate_json(content)
        except (ValidationError, ValueError):
            pass

        # Try as raw dict
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                return model.model_validate(data)
        except (json.JSONDecodeError, ValidationError):
            pass

        return None

    def _try_parse_dict(self, data: dict[str, Any], model: Type[T]) -> T | None:
        """Try to validate a dict into a Pydantic model."""
        try:
            return model.model_validate(data)
        except ValidationError as exc:
            logger.debug(f"Validation failed for {model.__name__}: {exc.error_count()} errors")
            return None

    @staticmethod
    def _strip_code_fences(content: str) -> str:
        """Remove markdown code fences from content."""
        stripped = content.strip()
        if stripped.startswith("```"):
            lines = stripped.split("\n")
            # Remove first line (```json or ```) and last line (```)
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            return "\n".join(lines)
        return content

    @staticmethod
    def build_structured_prompt(system_prompt: str, output_model: Type[BaseModel]) -> str:
        """Append JSON schema instruction to a system prompt."""
        schema = output_model.model_json_schema()
        schema_str = json.dumps(schema, indent=2)
        return f"{system_prompt}{_STRUCTURED_INSTRUCTION}\n\nSchema:\n{schema_str}"
