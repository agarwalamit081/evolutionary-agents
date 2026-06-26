"""F3 — should_gate_destructive: the hint→HITL decision (duck-typed, fail-safe).

``should_gate_destructive`` answers the factual question "is this tool flagged
destructiveHint" via the registry's ``is_destructive``. The module is duck-typed
to avoid a safety→tools import cycle, and any registry error / missing registry
returns False (the lookup could not confirm destructiveness). The *policy*
safe-default — block a destructive tool when no human can resume the interrupt —
lives in the execute-node gate, not here.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.safety.pipeline import should_gate_destructive


def test_true_when_registry_flags_destructive() -> None:
    """A registry reporting destructiveHint=True gates the tool."""
    registry = MagicMock()
    registry.is_destructive.return_value = True
    assert should_gate_destructive("terminal_command", registry) is True


def test_false_when_registry_reports_not_destructive() -> None:
    """A non-destructive tool is not gated."""
    registry = MagicMock()
    registry.is_destructive.return_value = False
    assert should_gate_destructive("file_reader", registry) is False


def test_false_when_registry_is_none() -> None:
    """No registry ⇒ cannot confirm destructiveness ⇒ not gated here."""
    assert should_gate_destructive("terminal_command", None) is False


def test_swallows_registry_error_and_returns_false() -> None:
    """A malformed/duck-typed registry raises ⇒ the lookup fails open (False).

    This is the rare defensive path; the execute gate's interrupt-fallback is
    what guarantees a flagged destructive tool still never runs unapproved.
    """
    registry = MagicMock()
    registry.is_destructive.side_effect = RuntimeError("registry mis-shape")
    assert should_gate_destructive("anything", registry) is False


def test_real_registry_matches_annotation_map() -> None:
    """The default registry's is_destructive reflects the central annotation map."""
    from src.tools import create_default_registry

    registry = create_default_registry()
    assert should_gate_destructive("terminal_command", registry) is True
    assert should_gate_destructive("http_request", registry) is True
    assert should_gate_destructive("index_corpus", registry) is True
    assert should_gate_destructive("file_writer", registry) is False
    assert should_gate_destructive("code_executor", registry) is False
