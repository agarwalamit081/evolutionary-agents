"""Structured output extraction with multi-fallback JSON parsing."""

from __future__ import annotations

from typing import Any, Type, TypeVar

import json
import json_repair
from loguru import logger
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

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
    ) -> T | None:
        """Extract and validate a Pydantic model from raw LLM content.

        Args:
            raw_content: The raw text content from the LLM response.
            output_model: The Pydantic model class to parse into.
            max_retries: Number of json-repair attempts before giving up.

        Returns:
            Validated Pydantic model instance, or None on failure.
        """
        # Attempt 1: Direct parse
        result = self._try_parse(raw_content, output_model)
        if result is not None:
            return result

        # Attempt 2: json_repair salvage
        try:
            repaired = json_repair.loads(raw_content)
            if isinstance(repaired, dict):
                result = self._try_parse_dict(repaired, output_model)
                if result is not None:
                    self._repair_attempts += 1
                    logger.debug(f"JSON repair succeeded (total repairs: {self._repair_attempts})")
                    return result
        except Exception as exc:
            logger.debug(f"json_repair failed: {exc}")

        # Attempt 3: Strip markdown fences and retry
        stripped = self._strip_code_fences(raw_content)
        if stripped != raw_content:
            result = self._try_parse(stripped, output_model)
            if result is not None:
                return result

        logger.warning(
            f"Failed to extract {output_model.__name__} from response "
            f"(first 200 chars): {raw_content[:200]}"
        )
        return None

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
