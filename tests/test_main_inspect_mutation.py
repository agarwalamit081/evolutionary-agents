"""SI-6 — ``--inspect-mutation`` CLI flag (``main._run_inspect_mutation``).

Promoted from ``scripts/inspect_mutation.py``: the row-print is now CLI glue in
main.py, with the genuinely reusable logic (the JSON-vs-free-text SHAPE
heuristic) lifted to ``src.evolution.promote.classify_payload`` (covered in
``tests/test_evolution/test_promote.py``). These tests pin the WIRING:
option names → dispatch → handler → fetch → print contract, with a mocked
``get_session`` (no DB) and the exit-code contract (0 found / 1 not found).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import click.testing
import pytest

import main as main_mod


class _FakeSession:
    """Fake AsyncSession: execute() → result with scalars().first()."""

    def __init__(self, row: Any) -> None:
        self._row = row

    async def execute(self, stmt: object) -> SimpleNamespace:
        del stmt  # the exact SELECT is not asserted here
        return SimpleNamespace(scalars=lambda: SimpleNamespace(first=lambda: self._row))


class _FakeGetSession:
    """Async context manager yielding a _FakeSession (replaces get_session)."""

    def __init__(self, row: Any) -> None:
        self._row = row

    async def __aenter__(self) -> _FakeSession:
        return _FakeSession(self._row)

    async def __aexit__(self, *_a: object) -> bool:
        return False


def _row(content: str = "Be terse. Emit JSON only.") -> SimpleNamespace:
    return SimpleNamespace(
        id="4c5c11d4-1234-5678-9abc-def012345678",
        mutation_type="prompt",
        target_path="prompts/system_prompt.md",
        status="deployed",
        model_used="glm-4.7",
        description="rewrite execute system prompt",
        mutated_content=content,
    )


class TestRunInspectMutation:
    def test_no_mutation_returns_1(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "src.db.session.get_session", lambda: _FakeGetSession(None)
        )
        result = click.testing.CliRunner().invoke(main_mod.main, ["--inspect-mutation"])
        assert result.exit_code == 1
        assert "No mutation found" in result.output

    def test_found_row_prints_fields_and_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "src.db.session.get_session", lambda: _FakeGetSession(_row())
        )
        result = click.testing.CliRunner().invoke(main_mod.main, ["--inspect-mutation"])
        assert result.exit_code == 0
        # Row fields
        assert "mutation_type : prompt" in result.output
        assert "prompts/system_prompt.md" in result.output
        assert "status        : deployed" in result.output
        assert "content_len   :" in result.output
        # Shape label from classify_payload (free-text PROMPT → promotable)
        assert "shape" in result.output
        assert "free-text" in result.output
        assert "one promoted suffix" in result.output

    def test_long_content_truncates_unless_full(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        long_content = "x" * 1200
        monkeypatch.setattr(
            "src.db.session.get_session",
            lambda: _FakeGetSession(_row(long_content)),
        )
        truncated = click.testing.CliRunner().invoke(
            main_mod.main, ["--inspect-mutation"]
        )
        assert truncated.exit_code == 0
        assert "content_len   : 1200 chars" in result_out(truncated)
        assert "[truncated]" in result_out(truncated)

        full = click.testing.CliRunner().invoke(
            main_mod.main, ["--inspect-mutation", "--inspect-full"]
        )
        assert full.exit_code == 0
        assert "[truncated]" not in result_out(full)
        assert "x" * 1200 in result_out(full)


def result_out(result: Any) -> str:
    """Click may not flush output to .output in all setups; prefer stdout_bytes."""
    if result.output:
        return result.output
    return (result.stdout_bytes or b"").decode()
