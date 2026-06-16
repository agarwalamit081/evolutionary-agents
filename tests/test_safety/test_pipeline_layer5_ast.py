"""Layer 5 (behavioral) AST tests — file-write sandbox scoping (M4/B2).

The AST rewrite replaces the old ``"open(" in code and "write" in code``
substring test. New contract: an ``open()`` in a write mode is flagged only when
its literal path resolves outside the sandbox root. Relative-path writes
(legitimate workspace output) are allowed; dynamic paths are left to Layer 6.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.safety.pipeline import SafetyPipeline


@pytest.fixture
def pipeline() -> SafetyPipeline:
    return SafetyPipeline()


def _fn(body: str) -> str:
    """Wrap a statement in an async function (satisfies Layer 7)."""
    return f"async def save() -> None:\n    {body}\n"


class TestLayer5WriteScoping:
    @pytest.mark.asyncio
    async def test_relative_write_allowed(self, pipeline: SafetyPipeline) -> None:
        """A relative-path write resolves under cwd (sandbox) — not flagged.

        This is the regression the AST rewrite fixes: the old substring test
        false-positived any ``open(..., 'w')`` whose code never said 'sandbox'.
        """
        result = await pipeline.validate(_fn("open('output.json', 'w').write('x')"))
        assert result["layers"]["behavioral"]["passed"] is True

    @pytest.mark.asyncio
    async def test_absolute_write_outside_sandbox_flagged(
        self, pipeline: SafetyPipeline
    ) -> None:
        result = await pipeline.validate(
            _fn("open('/etc/malicious.txt', 'w').write('bad')")
        )
        layer = result["layers"]["behavioral"]
        assert layer["passed"] is False
        assert any("File write outside sandbox" in i for i in layer["issues"])

    @pytest.mark.asyncio
    async def test_explicit_sandbox_root_flagged_outside(
        self, pipeline: SafetyPipeline
    ) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="turing_safety_"))
        result = await pipeline.validate(
            _fn("open('/etc/x.txt', 'w').write('bad')"),
            context={"sandbox_root": str(tmp)},
        )
        assert result["layers"]["behavioral"]["passed"] is False

    @pytest.mark.asyncio
    async def test_write_inside_sandbox_root_allowed(
        self, pipeline: SafetyPipeline
    ) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="turing_safety_"))
        inner = str(tmp / "out.txt")
        result = await pipeline.validate(
            _fn(f"open({inner!r}, 'w').write('x')"),
            context={"sandbox_root": str(tmp)},
        )
        assert result["layers"]["behavioral"]["passed"] is True

    @pytest.mark.asyncio
    async def test_read_mode_not_flagged(self, pipeline: SafetyPipeline) -> None:
        """Read-only opens are never behavioral violations."""
        result = await pipeline.validate(
            _fn("open('/etc/hosts').read()")
        )
        assert result["layers"]["behavioral"]["passed"] is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["w", "a", "x", "wb", "w+", "a+", "x+"])
    async def test_all_write_modes_flagged_outside(
        self, pipeline: SafetyPipeline, mode: str
    ) -> None:
        result = await pipeline.validate(
            _fn(f"open('/etc/x.txt', {mode!r}).write('b')")
        )
        assert result["layers"]["behavioral"]["passed"] is False

    @pytest.mark.asyncio
    async def test_keyword_mode_arg_handled(
        self, pipeline: SafetyPipeline
    ) -> None:
        """``open(path, mode='w')`` (keyword) is detected as a write."""
        result = await pipeline.validate(
            _fn("open('/etc/x.txt', mode='w').write('b')")
        )
        assert result["layers"]["behavioral"]["passed"] is False

    @pytest.mark.asyncio
    async def test_dynamic_path_skipped(self, pipeline: SafetyPipeline) -> None:
        """A non-literal path can't be statically resolved — left to Layer 6."""
        result = await pipeline.validate(
            _fn("p = build_path()\nopen(p, 'w').write('b')")
        )
        assert result["layers"]["behavioral"]["passed"] is True

    @pytest.mark.asyncio
    async def test_infinite_loop_still_detected(
        self, pipeline: SafetyPipeline
    ) -> None:
        """The while-True heuristic is preserved alongside the AST write check."""
        result = await pipeline.validate(
            "def run_forever() -> None:\n    x = 1\n    while True:\n        x += 1\n"
        )
        layer = result["layers"]["behavioral"]
        assert layer["passed"] is False
        assert any("infinite loop" in i.lower() for i in layer["issues"])

    @pytest.mark.asyncio
    async def test_non_python_input_no_crash(self, pipeline: SafetyPipeline) -> None:
        """Syntax-broken code doesn't raise from Layer 5 (Layer 1 reports it)."""
        result = await pipeline.validate(_fn("open('/etc/x', 'w'"))
        # Layer 1 flags the syntax error; Layer 5 simply contributes nothing.
        assert "behavioral" in result["layers"]
