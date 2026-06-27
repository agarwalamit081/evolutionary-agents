"""Core tool generation, validation, materialization, and registration.

Generates missing tool code via LLM, validates through safety pipeline
and sandbox, materializes handler callables, and registers in ToolRegistry.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger
from pydantic import BaseModel, Field

from src.config import get_settings
from src.graph.enums import TaskComplexity
from src.tools.dynamic.allowlist import get_materializer_namespace

if TYPE_CHECKING:
    from src.llm.gateway import LLMGateway
    from src.safety.pipeline import SafetyPipeline
    from src.sandbox.executor import SandboxExecutor
    from src.tools.registry import ToolRegistry


class GeneratedTool(BaseModel):
    """Structured output from LLM describing a generated tool."""

    tool_name: str = Field(description="snake_case tool name, e.g. json_parser")
    description: str = Field(description="Human-readable description for LLM tool selection")
    input_schema: dict[str, Any] = Field(description="JSON Schema for tool parameters")
    handler_code: str = Field(description="Complete Python async function source code")
    test_code: str = Field(
        min_length=1,
        description=(
            "Self-contained test that calls the handler and asserts its result. "
            "Required (D9): a tool with no asserting test cannot be registered."
        ),
    )


class ToolGenerator:
    """Generates, validates, and registers tools at runtime.

    Security model:
        1. LLM generates tool code (untrusted)
        2. SafetyPipeline validates (static analysis)
        3. SandboxExecutor tests in isolation
        4. _materialize_handler() uses constrained namespace
        5. ToolRegistry.register() makes it available

    Rate-limited to ``AgentSettings.max_tools_per_run`` per instance.
    """

    def __init__(
        self,
        gateway: LLMGateway,
        safety_pipeline: SafetyPipeline,
        sandbox: SandboxExecutor | None = None,
    ) -> None:
        self._gateway = gateway
        self._safety = safety_pipeline
        self._sandbox = sandbox
        self._tools_created = 0

    async def generate(
        self,
        gap_description: str,
        context: dict[str, Any],
    ) -> GeneratedTool | None:
        """Ask LLM to generate tool code given a capability gap.

        Args:
            gap_description: What the tool should do (e.g. "fetch data from URLs").
            context: Execution context (goal, failed tools, existing tools).

        Returns:
            GeneratedTool if LLM produces valid output, None on failure.
        """
        max_tools = get_settings().agent.max_tools_per_run
        if self._tools_created >= max_tools:
            logger.warning(
                f"Max tools per run ({max_tools}) reached, skipping generation"
            )
            return None

        try:
            from src.llm.structured_output import StructuredOutputManager

            messages = self._build_messages(gap_description, context)

            # Route code generation at a code-strong model (configurable; default
            # deepseek-v4-pro) instead of the CHEAP tier (complexity=SIMPLE →
            # Haiku), which truncates non-trivial handlers → AST failure. An empty
            # setting preserves the legacy complexity-based routing.
            codegen_model = get_settings().agent.tool_generation_model

            # Force JSON mode so the model emits properly-escaped JSON. Without
            # it, a multi-line Python handler embedded as a JSON string value
            # breaks parsing on an unescaped quote/newline, and json_repair then
            # salvages a TRUNCATED handler (observed live: 156-char handlers,
            # "Syntax error at line 6: '(' was never closed") that fails the AST
            # safety gate and never registers. JSON mode (supported by both
            # deepseek-v4-pro and Haiku) eliminates that truncation at the
            # source. The system prompt already contains the token "JSON", which
            # DeepSeek/OpenAI JSON mode requires.
            json_mode = {"type": "json_object"}
            if codegen_model:
                response = await self._gateway.acompletion(
                    messages=messages,
                    model=codegen_model,
                    response_format=json_mode,
                    timeout=get_settings().llm.codegen_timeout,
                )
            else:
                response = await self._gateway.acompletion(
                    messages=messages,
                    complexity=TaskComplexity.SIMPLE,
                    response_format=json_mode,
                    timeout=get_settings().llm.codegen_timeout,
                )

            extractor = StructuredOutputManager()
            tool = await extractor.extract(response.content, GeneratedTool)
            if tool is None:
                logger.warning("Failed to extract GeneratedTool from LLM response")
                return None

            # Validate tool name format
            if not tool.tool_name.replace("_", "").isalnum():
                logger.warning(f"Invalid tool name: {tool.tool_name}")
                return None

            logger.info(
                f"Generated tool: {tool.tool_name} "
                f"({len(tool.handler_code)} chars handler, "
                f"{len(tool.test_code)} chars test)"
            )
            return tool

        except Exception as e:
            logger.warning(f"Tool generation failed: {e}")
            return None

    async def validate_and_register(
        self,
        tool: GeneratedTool,
        registry: ToolRegistry,
    ) -> dict[str, Any]:
        """Validate generated tool and register if safe.

        Runs safety pipeline (7 layers) with allowlisted modules,
        sandbox tests the handler, materializes the callable, and
        registers in the ToolRegistry.

        Args:
            tool: The generated tool specification.
            registry: ToolRegistry to register into.

        Returns:
            Result dict with 'success', 'reason', 'safety_result', 'sandbox_result'.
        """
        # Active-population cap (findings.md A3): the generated-tool population
        # must never grow past max_active_tools mid-run. A pre-register gate is
        # strictly safer than a post-register enforce_caps, which could retire a
        # tool the run is about to invoke (a race). Skip registration (return a
        # failure result) so the gap becomes a failed attempt, NOT a registry
        # entry. The per-run cap (max_tools_per_run) already bounds how many
        # generations reach this point.
        max_active_tools = get_settings().agent.max_active_tools
        if registry.generated_count >= max_active_tools:
            logger.warning(
                f"Active generated-tool population at cap "
                f"({registry.generated_count}/{max_active_tools}), "
                f"skipping registration of '{tool.tool_name}'"
            )
            return {
                "success": False,
                "reason": (
                    f"Active tool cap reached "
                    f"({registry.generated_count}/{max_active_tools})"
                ),
            }

        # Step 1: shared code gate (D9) — assertion presence + ruff lint + 7-layer
        # safety + optional sandbox smoke. Extracted to
        # ``src.tools.dynamic.validation`` so the operator-facing edit→approve
        # API (D10) applies the IDENTICAL bar; the two entry points cannot drift.
        from src.tools.dynamic.validation import dedupe_imports, validate_tool_code

        # Sanitize exact-duplicate top-level imports (F811): the dedupe inside
        # ``validate_tool_code`` makes the gate tolerant, but the registry
        # persists ``tool.handler_code`` directly (registry.register(...,
        # handler_code=...) below), so dedupe the field here too — the stored
        # handler is clean and registers on the first attempt. The retry-feedback
        # loop otherwise never converges on the model re-emitting the dup.
        tool.handler_code = dedupe_imports(tool.handler_code)
        tool.test_code = dedupe_imports(tool.test_code)

        validation = await validate_tool_code(
            handler_code=tool.handler_code,
            test_code=tool.test_code,
            tool_name=tool.tool_name,
            safety_pipeline=self._safety,
            sandbox=self._sandbox,
        )
        if not validation.passed:
            logger.warning(
                f"Tool '{tool.tool_name}' failed code validation: {validation.reason}"
            )
            return {
                "success": False,
                "reason": validation.reason,
                "safety_result": validation.safety_result,
                "lint_result": validation.lint_result,
                "sandbox_result": validation.sandbox_result,
            }

        safety_result = validation.safety_result
        sandbox_result = validation.sandbox_result

        # Step 3: Materialize handler
        try:
            handler = self._materialize_handler(tool.handler_code)
        except ValueError as e:
            logger.warning(f"Tool '{tool.tool_name}' materialization failed: {e}")
            return {
                "success": False,
                "reason": f"Handler materialization failed: {e}",
                "safety_result": safety_result,
                "sandbox_result": sandbox_result,
            }

        # Step 4: Register in ToolRegistry. ``generated=True`` + the source so
        # that, in a sandboxed code-exec mode (docker/runner), the execute node
        # routes this tool's invocation through that sandbox instead of calling
        # the in-process ``handler`` — the handler_code is untrusted LLM output
        # and must not run inside the worker with full DB/Redis/FS access.
        registry.register(
            name=tool.tool_name,
            handler=handler,
            description=tool.description,
            parameters=tool.input_schema,
            generated=True,
            handler_code=tool.handler_code,
            # F3 — tag as a runtime-generated tool (distinct from hand-written
            # builtins) for scope-injection / recall. Generated tools are never
            # flagged destructive (they run through the sandbox already).
            tags=["generated"],
        )

        self._tools_created += 1
        logger.info(
            f"Registered tool '{tool.tool_name}' "
            f"(total created this run: {self._tools_created})"
        )

        return {
            "success": True,
            "tool_name": tool.tool_name,
            "safety_result": safety_result,
            "lint_result": validation.lint_result,
            "sandbox_result": sandbox_result,
        }

    def _build_messages(
        self,
        gap_description: str,
        context: dict[str, Any],
    ) -> list[dict[str, str]]:
        """Build LLM messages for tool generation.

        Args:
            gap_description: What the tool should do.
            context: Execution context dict.

        Returns:
            List of message dicts for the LLM.
        """
        from src.graph.prompts import TOOL_GENERATE_SYSTEM, TOOL_GENERATE_USER

        existing_tools = context.get("existing_tools", [])
        existing_str = ", ".join(existing_tools) if existing_tools else "none"

        user_content = TOOL_GENERATE_USER.format(
            gap_description=gap_description,
            goal_text=context.get("goal_text", ""),
            failed_tools=context.get("failed_tools", "none"),
            error_details=context.get("error_details", ""),
            existing_tools=existing_str,
        )

        return [
            {"role": "system", "content": str(TOOL_GENERATE_SYSTEM)},
            {"role": "user", "content": user_content},
        ]

    def _materialize_handler(self, handler_code: str) -> Callable[..., Any]:
        """Create a callable from handler source code.

        Validates the code defines exactly one async function via AST,
        then compiles and executes in a constrained namespace containing
        only pre-imported safe modules.

        Args:
            handler_code: Python source code defining an async function.

        Returns:
            The async callable extracted from the code.

        Raises:
            ValueError: If code is invalid or doesn't define exactly one async function.
        """
        # Pre-validate with AST
        try:
            tree = ast.parse(handler_code)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in handler code: {e}") from e

        # Find async function definitions
        async_funcs = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef)
        ]

        if len(async_funcs) == 0:
            raise ValueError("Handler code must define exactly one async function, found 0")
        if len(async_funcs) > 1:
            raise ValueError(
                f"Handler code must define exactly one async function, found {len(async_funcs)}"
            )

        func_name = async_funcs[0].name

        # Compile and execute in constrained namespace
        namespace = get_materializer_namespace()
        code_obj = compile(handler_code, f"<generated_tool:{func_name}>", "exec")
        exec(code_obj, namespace)  # noqa: S102

        handler = namespace.get(func_name)
        if handler is None or not callable(handler):
            raise ValueError(f"Function '{func_name}' not found in materialized namespace")

        return handler
