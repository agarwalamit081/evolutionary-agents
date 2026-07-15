"""Jinja2-based prompt template loader and manager.

Provides a PromptTemplate wrapper with a .format() method so all existing
node imports continue working unchanged. Templates are loaded from the
templates/ subdirectory using jinja2.FileSystemLoader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jinja2

# Template directory: src/graph/prompts/templates/
_TEMPLATE_DIR = Path(__file__).parent / "templates"

# Shared jinja2 environment — cached for reuse
_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=jinja2.Undefined,  # strict-ish: raises on render if missing
)


class PromptTemplate:
    """Lazy-loading jinja2 template wrapper with .format() compatibility.

    Replaces raw string constants so all existing call sites using
    ``PROMPT.format(key=value)`` continue to work without modification.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._template: jinja2.Template | None = None

    def _get_template(self) -> jinja2.Template:
        """Lazily load and cache the jinja2 template."""
        if self._template is None:
            self._template = _env.get_template(f"{self._name}.j2")
        return self._template

    def format(self, **kwargs: Any) -> str:
        """Render the template with the given variables.

        Identical API to ``str.format()`` so existing node code works
        unchanged. Internally delegates to ``jinja2.Template.render()``.
        """
        return self._get_template().render(**kwargs)

    def __str__(self) -> str:
        """Render the template with no variables (for system prompts)."""
        return self._get_template().render()

    def __repr__(self) -> str:
        return f"PromptTemplate({self._name!r})"


class PromptManager:
    """Singleton manager for loading and rendering prompt templates.

    Provides a centralized registry for all prompts. Each template is
    loaded once and cached for the lifetime of the process.
    """

    def __init__(self) -> None:
        self._templates: dict[str, PromptTemplate] = {}

    def get(self, name: str) -> PromptTemplate:
        """Get or create a PromptTemplate by name."""
        if name not in self._templates:
            self._templates[name] = PromptTemplate(name)
        return self._templates[name]

    def render(self, name: str, **kwargs: Any) -> str:
        """Render a named template with the given variables."""
        return self.get(name).format(**kwargs)


# Module-level singleton
prompt_manager = PromptManager()
