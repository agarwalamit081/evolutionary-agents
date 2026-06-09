"""Tool create node — generates, validates, and registers missing tools at runtime.

When reflection detects tool gaps (e.g. the LLM called a tool that doesn't exist),
this node attempts to create the missing tool via LLM code generation, safety
validation, sandbox testing, and dynamic registration.

Flow:
    reflect (detects gap) → tool_create → plan (re-plan with new tool) → execute
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from src.graph.enums import Phase
from src.graph.state import AgentState

if TYPE_CHECKING:
    from src.llm.gateway import LLMGateway
    from src.tools.registry import ToolRegistry


async def tool_create_node(
    state: AgentState,
    *,
    gateway: LLMGateway | None = None,
    tools: ToolRegistry | None = None,
) -> dict[str, Any]:
    """Generate and register missing tools identified during reflection.

    Iterates over pending_tool_gaps, attempts LLM-based tool generation,
    validates through safety pipeline + sandbox, and registers successful
    tools in the ToolRegistry for immediate use.

    Args:
        state: Current agent state with pending_tool_gaps populated.
        gateway: Optional LLM gateway for code generation.
        tools: Optional ToolRegistry for dynamic registration.

    Returns:
        Partial state update clearing pending gaps and recording created tools.
    """
    pending_gaps = state.get("pending_tool_gaps", [])
    goal = state.get("current_goal")
    tool_results = state.get("tool_results", [])

    if not pending_gaps or gateway is None or tools is None:
        logger.info(
            f"Tool creation skipped: gaps={len(pending_gaps)}, "
            f"gateway={'yes' if gateway else 'no'}, tools={'yes' if tools else 'no'}"
        )
        return {
            "phase": Phase.EXECUTE,
            "pending_tool_gaps": [],
        }

    logger.info(f"Attempting to create {len(pending_gaps)} missing tool(s)")

    created_tools: list[dict[str, Any]] = []
    remaining_gaps: list[str] = []

    for gap_description in pending_gaps:
        result = await _create_single_tool(
            gateway=gateway,
            registry=tools,
            gap_description=gap_description,
            goal_text=goal.text if goal else "",
            tool_results=tool_results,
        )

        if result["success"]:
            created_tools.append(result)
            logger.info(f"Successfully created tool: {result['tool_name']}")
        else:
            remaining_gaps.append(gap_description)
            logger.warning(
                f"Failed to create tool for '{gap_description}': "
                f"{result.get('reason', 'unknown')}"
            )

    return {
        "phase": Phase.PLAN if created_tools else Phase.EXECUTE,
        "pending_tool_gaps": remaining_gaps,
        "tools_created": created_tools,
    }


async def _create_single_tool(
    gateway: LLMGateway,
    registry: ToolRegistry,
    gap_description: str,
    goal_text: str,
    tool_results: list[Any],
) -> dict[str, Any]:
    """Attempt to create a single tool for a specific capability gap.

    Args:
        gateway: LLM gateway for code generation.
        registry: ToolRegistry to register the tool into.
        gap_description: What capability the tool should provide.
        goal_text: Current goal text for context.
        tool_results: Previous tool results for error context.

    Returns:
        Result dict with 'success' bool and tool details or failure reason.
    """
    try:
        from src.safety.pipeline import SafetyPipeline
        from src.tools.dynamic.generator import ToolGenerator

        safety = SafetyPipeline()

        # Sandbox is optional — best-effort
        sandbox: Any = None
        try:
            from src.sandbox.executor import SandboxExecutor

            sandbox = SandboxExecutor()
        except Exception:
            logger.debug("SandboxExecutor not available for tool creation")

        generator = ToolGenerator(
            gateway=gateway,
            safety_pipeline=safety,
            sandbox=sandbox,
        )

        # Gather context about failed tools
        failed_tools: list[str] = []
        error_details: list[str] = []
        for tr in tool_results:
            if hasattr(tr, "error") and tr.error:
                if "Unknown tool" in str(tr.error):
                    failed_tools.append(f"{tr.tool_name}: {tr.error}")
                error_details.append(str(tr.error)[:200])

        context = {
            "goal_text": goal_text,
            "failed_tools": "; ".join(failed_tools[-3:]) or "none",
            "error_details": "; ".join(error_details[-3:]) or "none",
            "existing_tools": [t["name"] for t in registry.list_tools()],
        }

        generated = await generator.generate(gap_description, context)
        if generated is None:
            return {
                "success": False,
                "reason": "LLM generation failed or returned invalid output",
                "gap": gap_description,
            }

        result = await generator.validate_and_register(generated, registry)
        if not result["success"]:
            return {
                "success": False,
                "reason": result.get("reason", "validation failed"),
                "gap": gap_description,
            }

        # Best-effort persistence to DB (non-blocking, non-fatal)
        await _persist_tool(generated)

        return {
            "success": True,
            "tool_name": generated.tool_name,
            "description": generated.description,
            "safety_passed": True,
            "sandbox_passed": result.get("sandbox_result", {}).get("passed", True),
        }

    except Exception as e:
        logger.warning(f"Tool creation error for '{gap_description}': {e}")
        return {
            "success": False,
            "reason": str(e),
            "gap": gap_description,
        }


async def _persist_tool(generated: Any) -> None:
    """Best-effort persistence of a validated tool to the database.

    Non-fatal — if the DB is unavailable, the tool is still registered
    in-memory for the current run.

    Args:
        generated: GeneratedTool with handler_code and test_code.
    """
    try:
        from src.tools.dynamic.persister import ToolPersister

        persister = ToolPersister()
        await persister.persist(
            tool_name=generated.tool_name,
            description=generated.description,
            input_schema=generated.input_schema,
            handler_code=generated.handler_code,
            test_code=generated.test_code,
        )
        logger.info(f"Persisted tool '{generated.tool_name}' to database")
    except Exception as e:
        logger.debug(f"Tool persistence skipped (non-critical): {e}")
