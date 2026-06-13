"""Memory folding — compresses agent conversation history into structured summaries.

Adapted from DeepAgent's autonomous memory folding mechanism. When the
agent's reasoning context grows too large (triggered by iteration count,
token usage, or message count), three parallel LLM calls generate:

1. **Episode Memory**: key events, milestones, and decisions
2. **Working Memory**: immediate goals, challenges, and next actions
3. **Tool Memory**: tool usage patterns, success rates, derived rules

These replace the full message history with compressed summaries,
dramatically reducing token consumption on long-running tasks.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, RemoveMessage
from loguru import logger

from src.graph.enums import TaskComplexity

if TYPE_CHECKING:
    from src.llm.gateway import LLMGateway


class MemoryFoldResult:
    """Structured result of a memory fold operation."""

    __slots__ = (
        "episode_memory",
        "working_memory",
        "tool_memory",
        "folded_at",
        "tokens_saved_estimate",
        "fold_number",
    )

    def __init__(
        self,
        episode_memory: dict[str, Any],
        working_memory: dict[str, Any],
        tool_memory: dict[str, Any],
        fold_number: int,
        tokens_saved_estimate: int = 0,
    ) -> None:
        self.episode_memory = episode_memory
        self.working_memory = working_memory
        self.tool_memory = tool_memory
        self.folded_at = datetime.now(timezone.utc).isoformat()
        self.tokens_saved_estimate = tokens_saved_estimate
        self.fold_number = fold_number

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for state storage."""
        return {
            "episode_memory": self.episode_memory,
            "working_memory": self.working_memory,
            "tool_memory": self.tool_memory,
            "folded_at": self.folded_at,
            "tokens_saved_estimate": self.tokens_saved_estimate,
            "fold_number": self.fold_number,
        }


def _serialize_messages(messages: list[Any]) -> str:
    """Serialize a list of LangChain messages to a text summary.

    Args:
        messages: List of AnyMessage objects.

    Returns:
        A text representation suitable for LLM consumption.
    """
    parts: list[str] = []
    for msg in messages:
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            # Structured content blocks
            content = json.dumps(content, default=str, ensure_ascii=False)[:500]
        else:
            content = str(content)[:500]
        parts.append(f"[{role}] {content}")
    return "\n".join(parts)


def _serialize_tool_history(state: dict[str, Any]) -> str:
    """Serialize tool call history from agent state.

    Args:
        state: The current agent state dict.

    Returns:
        A text summary of tool calls and results.
    """
    tools_called = state.get("tools_called", [])
    tool_results = state.get("tool_results", [])

    lines: list[str] = []
    for tc in tools_called:
        name = tc.get("tool_name", tc.get("name", "unknown"))
        args = tc.get("arguments", tc.get("args", {}))
        lines.append(f"Called: {name}({json.dumps(args, default=str)[:200]})")

    for tr in tool_results:
        if hasattr(tr, "tool_name"):
            name = tr.tool_name
            success = getattr(tr, "success", None)
            output = getattr(tr, "output", "")
            error = getattr(tr, "error", "")
            status = "success" if success else "failed"
            detail = output[:200] if success else error[:200]
            lines.append(f"Result: {name} ({status}) — {detail}")
        elif isinstance(tr, dict):
            name = tr.get("tool_name", "unknown")
            status = "success" if tr.get("success") else "failed"
            detail = tr.get("output", tr.get("error", ""))[:200]
            lines.append(f"Result: {name} ({status}) — {detail}")

    return "\n".join(lines) if lines else "No tool calls recorded."


class MemoryFolder:
    """Compresses agent conversation history into structured summaries.

    Args:
        gateway: LLMGateway for making LLM calls. Also the source of live
            token totals via ``get_cost_records()`` (the token trigger reads
            the gateway's per-run accumulator, not graph state, because state's
            ``total_tokens_used`` is only flushed at the terminal store_memory
            node — long after folding needs to fire).
        fold_interval: Minimum iterations between folds (cooldown window).
        token_threshold: Trigger when accumulated gateway token usage reaches this.
        message_token_estimate: Tertiary trigger: fold when estimated message
            tokens (chars // 4) reach this.
        message_count_floor: Minimum messages before folding is considered.
        message_count_threshold: Primary trigger: fold when message count
            reaches this.
        max_folds: Maximum number of folds per agent run.
    """

    def __init__(
        self,
        gateway: LLMGateway,
        fold_interval: int = 6,
        token_threshold: int = 50_000,
        message_token_estimate: int = 8_000,
        message_count_floor: int = 10,
        message_count_threshold: int = 14,
        max_folds: int = 3,
    ) -> None:
        self._gateway = gateway
        self._fold_interval = fold_interval
        self._token_threshold = token_threshold
        self._message_token_estimate = message_token_estimate
        self._message_count_floor = message_count_floor
        self._message_count_threshold = message_count_threshold
        self._max_folds = max_folds
        self._fold_count = 0

    def should_fold(self, state: dict[str, Any]) -> bool:
        """Check if memory folding is needed based on state thresholds.

        Trigger ladder (evaluated in order, first match wins):

        1. **Cap** — already folded ``max_folds`` times this run → False.
        2. **Min guard** — too early (``iteration_count < 2``) or too few
           messages (``< message_count_floor``) → False.
        3. **Cooldown** — within ``fold_interval`` iterations of the last fold
           → False (prevents back-to-back folds).
        4. **Live-token trigger** — accumulated gateway token usage
           (``get_cost_records()``, the real per-run spend) reaches
           ``token_threshold`` → True.
        5. **Message-count trigger** — message count reaches
           ``message_count_threshold`` → True.
        6. **Context-size trigger** — estimated message tokens (chars // 4)
           reach ``message_token_estimate`` → True.

        Args:
            state: The current agent state dict.

        Returns:
            True if folding should be triggered.
        """
        # 1. Fold cap
        fold_history = state.get("fold_history", [])
        if len(fold_history) >= self._max_folds:
            return False

        iteration_count = state.get("iteration_count", 0)
        messages = state.get("messages", [])

        # 2. Minimum guard — don't fold too early or with too little history
        if iteration_count < 2 or len(messages) < self._message_count_floor:
            return False

        # 3. Cooldown — don't re-fold within fold_interval of the last fold
        last_fold = state.get("last_fold_iteration", 0) or 0
        if last_fold and (iteration_count - last_fold) < self._fold_interval:
            return False

        # 4. Live-token trigger — read real accumulated spend from the gateway
        #    (state["total_tokens_used"] is only flushed at store_memory, long
        #    after folding must fire, so it reads as 0 during the loop).
        total_tokens = 0
        try:
            for record in self._gateway.get_cost_records():
                total_tokens += (
                    getattr(record, "input_tokens", 0)
                    + getattr(record, "output_tokens", 0)
                )
        except Exception:
            total_tokens = 0
        if total_tokens >= self._token_threshold:
            return True

        # 5. Message-count trigger
        if len(messages) >= self._message_count_threshold:
            return True

        # 6. Context-size trigger (tertiary, for long-message scenarios)
        total_chars = sum(
            len(str(getattr(m, "content", ""))) for m in messages
        )
        if total_chars // 4 >= self._message_token_estimate:
            return True

        return False

    def build_removal_messages(self, state: dict[str, Any]) -> list[RemoveMessage]:
        """Build RemoveMessage entries for every id'd message in state.

        The ``messages`` field uses LangGraph's ``add_messages`` reducer, which
        appends and dedups by ID but never deletes. Returning
        ``RemoveMessage(id=...)`` alongside the fold summary instructs the
        reducer to actually drop the old messages, so folding shrinks context
        rather than just appending a summary on top of it.

        Args:
            state: The current agent state dict.

        Returns:
            One RemoveMessage per existing message that carries an id.
        """
        messages = state.get("messages", [])
        removal: list[RemoveMessage] = []
        for msg in messages:
            msg_id = getattr(msg, "id", None)
            if msg_id:
                removal.append(RemoveMessage(id=msg_id))
        return removal

    async def fold(self, state: dict[str, Any]) -> MemoryFoldResult:
        """Perform memory folding: compress history into 3 structured summaries.

        Args:
            state: The current agent state dict.

        Returns:
            MemoryFoldResult with episode, working, and tool memory.
        """
        from src.graph.prompts.memory_folding import (
            episode_memory_prompt,
            tool_memory_prompt,
            working_memory_prompt,
        )

        self._fold_count += 1
        messages = state.get("messages", [])
        goal_text = self._extract_goal(state)
        history_text = _serialize_messages(messages)
        tool_history_text = _serialize_tool_history(state)

        # Get available tools list
        tools_list = ""
        tools_obj = state.get("_tools")
        if tools_obj and hasattr(tools_obj, "list_names"):
            tools_list = ", ".join(tools_obj.list_names())

        # Estimate tokens being saved
        old_chars = sum(len(str(getattr(m, "content", ""))) for m in messages)
        tokens_saved = old_chars // 4

        logger.info(
            f"Memory fold #{self._fold_count}: compressing {len(messages)} messages "
            f"(~{tokens_saved} tokens) at iteration {state.get('iteration_count', 0)}"
        )

        # Run all 3 memory generation tasks in parallel
        episode, working, tool = await asyncio.gather(
            self._generate_memory(
                episode_memory_prompt(goal_text, history_text, tools_list),
                "episode",
            ),
            self._generate_memory(
                working_memory_prompt(goal_text, history_text, tools_list),
                "working",
            ),
            self._generate_memory(
                tool_memory_prompt(goal_text, history_text, tool_history_text, tools_list),
                "tool",
            ),
        )

        return MemoryFoldResult(
            episode_memory=episode,
            working_memory=working,
            tool_memory=tool,
            fold_number=self._fold_count,
            tokens_saved_estimate=tokens_saved,
        )

    def build_summary_message(self, result: MemoryFoldResult) -> HumanMessage:
        """Convert a fold result into a single summary HumanMessage.

        This message replaces the full conversation history in state.

        Args:
            result: The fold result to format.

        Returns:
            A HumanMessage with the compressed memory summary.
        """
        sections: list[str] = [
            f"[Memory Fold #{result.fold_number} — {result.folded_at}]",
            "",
            "## Episode Memory (key events and decisions)",
            json.dumps(result.episode_memory, indent=2, ensure_ascii=False),
            "",
            "## Working Memory (current goals and next actions)",
            json.dumps(result.working_memory, indent=2, ensure_ascii=False),
            "",
            "## Tool Memory (usage patterns and rules)",
            json.dumps(result.tool_memory, indent=2, ensure_ascii=False),
        ]
        return HumanMessage(content="\n".join(sections))

    async def _generate_memory(
        self,
        prompt: str,
        memory_type: str,
    ) -> dict[str, Any]:
        """Generate a single structured memory via LLM.

        Args:
            prompt: The full prompt for this memory type.
            memory_type: Label for logging ("episode", "working", "tool").

        Returns:
            Parsed JSON dict from the LLM response.
        """
        try:
            response = await self._gateway.acompletion(
                messages=[{"role": "user", "content": prompt}],
                complexity=TaskComplexity.TRIVIAL,
                temperature=0.1,
                max_tokens=1024,
            )
            content = response.content.strip()

            # Extract JSON from possible markdown fences
            if "```json" in content:
                content = content.split("```json", 1)[1].split("```", 1)[0]
            elif "```" in content:
                content = content.split("```", 1)[1].split("```", 1)[0]

            return json.loads(content.strip())

        except (json.JSONDecodeError, Exception) as exc:
            logger.warning(
                f"Failed to generate {memory_type} memory: {exc}. "
                f"Using fallback summary."
            )
            return {
                "error": f"Memory generation failed: {exc!s}",
                "memory_type": memory_type,
            }

    @staticmethod
    def _extract_goal(state: dict[str, Any]) -> str:
        """Extract the goal text from agent state.

        Args:
            state: The current agent state dict.

        Returns:
            Goal text string.
        """
        goal = state.get("current_goal")
        if goal and hasattr(goal, "text"):
            return goal.text
        return state.get("goal_text", "Unknown goal")
