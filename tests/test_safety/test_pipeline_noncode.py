"""Non-code mutation handling in the safety pipeline (Bug C regression).

Root cause surfaced by a live run: ``validate()`` ran the AST-dependent layers
(syntax / imports / semantic) on every mutation's ``mutated_content``. A
PROMPT/CONFIG/MEMORY mutation's content is natural-language text, so
``ast.parse`` hard-failed at Layer 1 — meaning the most common evolution
opportunity (prompt refinement) was permanently rejected and could never
deploy. M4 made Layer 5 tolerate non-code; Layers 1/4/7 did not.

Fix: code-emitting mutation types (CODE, TOOL) and context-free callers
(dynamic-tool validation passes no ``mutation_type``) run every layer; non-code
types skip only the AST-dependent layers. Static size, security regex, and
behavioral (already non-code-tolerant) still apply to non-code mutations — this
is safe hardening, not a security bypass.
"""

from __future__ import annotations

import pytest

from src.graph.enums import MutationType
from src.safety.pipeline import SafetyPipeline


@pytest.fixture
def pipeline() -> SafetyPipeline:
    return SafetyPipeline()


_PROSE = (
    "Refine the verify prompt so the agent quotes the full pytest summary line "
    "and confirms an all-green result before declaring the task complete."
)


class TestNonCodeMutations:
    """Non-code mutations pass the AST layers but keep static/security/behavioral."""

    @pytest.mark.asyncio
    async def test_prompt_mutation_passes_safety(self, pipeline: SafetyPipeline) -> None:
        """A natural-language PROMPT mutation no longer dies at the syntax layer."""
        result = await pipeline.validate(
            code=_PROSE, context={"mutation_type": MutationType.PROMPT}
        )
        assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_config_mutation_passes_safety(self, pipeline: SafetyPipeline) -> None:
        """A YAML/config-text CONFIG mutation passes (not valid Python, not code)."""
        result = await pipeline.validate(
            code="max_iterations: 25\nverbose: true\n# runtime tuning",
            context={"mutation_type": MutationType.CONFIG},
        )
        assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_non_code_skips_syntax_imports_semantic(
        self, pipeline: SafetyPipeline
    ) -> None:
        """The three AST layers are marked skipped for non-code mutations."""
        result = await pipeline.validate(
            code=_PROSE, context={"mutation_type": MutationType.PROMPT}
        )
        for layer in ("syntax", "imports", "semantic"):
            assert result["layers"][layer]["passed"] is True
            assert "skipped" in result["layers"][layer].get("note", "")

    @pytest.mark.asyncio
    async def test_non_code_still_runs_security_regex(
        self, pipeline: SafetyPipeline
    ) -> None:
        """A forbidden pattern in non-code content still fails the security layer.

        Skipping the AST layers must not bypass the security regex scan — a
        prompt that injects ``os.system`` is still rejected.
        """
        result = await pipeline.validate(
            code="Next, call os.system('rm -rf /tmp/x') to clean up.",
            context={"mutation_type": MutationType.PROMPT},
        )
        assert result["passed"] is False
        assert result["layers"]["security"]["passed"] is False

    @pytest.mark.asyncio
    async def test_non_code_still_runs_static_size(
        self, pipeline: SafetyPipeline
    ) -> None:
        """An oversized non-code mutation still fails the static size check."""
        oversized = "Refine the verify prompt step by step.\n" * 600  # 600 lines
        result = await pipeline.validate(
            code=oversized, context={"mutation_type": MutationType.PROMPT}
        )
        assert result["passed"] is False
        assert result["layers"]["static"]["passed"] is False


class TestCodeMutationsUnchanged:
    """Code mutations and context-free callers still run the full pipeline."""

    @pytest.mark.asyncio
    async def test_code_bad_syntax_still_fails(self, pipeline: SafetyPipeline) -> None:
        """A CODE mutation with broken syntax still fails the syntax layer."""
        result = await pipeline.validate(
            code="async def f(\n    return 1",
            context={"mutation_type": MutationType.CODE},
        )
        assert result["passed"] is False
        assert result["layers"]["syntax"]["passed"] is False

    @pytest.mark.asyncio
    async def test_tool_clean_passes(self, pipeline: SafetyPipeline) -> None:
        """A clean TOOL handler passes all layers (TOOL is a code mutation)."""
        result = await pipeline.validate(
            code="async def t() -> int:\n    return 2\n",
            context={"mutation_type": MutationType.TOOL},
        )
        assert result["passed"] is True

    @pytest.mark.asyncio
    async def test_no_mutation_type_runs_all_layers(
        self, pipeline: SafetyPipeline
    ) -> None:
        """Dynamic-tool validation (no mutation_type) is unchanged: code is assumed."""
        result = await pipeline.validate(code="def g( :")  # no context
        assert result["passed"] is False
        assert result["layers"]["syntax"]["passed"] is False
