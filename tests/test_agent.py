"""
Test suite for Turing Agent.

Tests are organised into layers:
  Unit    — individual classes, no LLM
  Integration — components working together, no LLM
  E2E     — full agent run (requires LLM API key)

Run all unit + integration tests (no API key needed):
  python -m pytest tests/ -v -k "not e2e"

Run everything including e2e:
  LLM_API_KEY=sk-ant-... python -m pytest tests/ -v
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def tmp_memory_dir(tmp_path):
    return str(tmp_path / "memory")


@pytest.fixture
def memory(tmp_memory_dir):
    from src.memory import Memory
    m = Memory(memory_dir=tmp_memory_dir)
    yield m
    m.close()


@pytest.fixture
def tool_registry(memory):
    from src.tools import create_registry
    return create_registry(source_dir=str(Path(__file__).parent.parent), memory=memory)


@pytest.fixture
def mock_llm():
    """LLM that returns canned JSON for planning."""
    llm = AsyncMock()
    llm.ainvoke = AsyncMock()
    return llm


# ─── Unit: State ─────────────────────────────────────────────────

class TestState:
    def test_initial_state_keys(self):
        from src.state import initial_state, Phase
        s = initial_state("test goal", agent_id="a1")
        assert s["agent_id"] == "a1"
        assert s["phase"] == Phase.PERCEIVE.value
        assert s["is_complete"] is False
        assert s["iteration_count"] == 0

    def test_goal_creation(self):
        from src.state import Goal, GoalStatus
        g = Goal(title="My goal", description="Do something", priority=1)
        assert g.status == GoalStatus.PENDING
        g.complete("done!")
        assert g.status == GoalStatus.COMPLETED
        assert g.result == "done!"
        assert g.completed_at is not None

    def test_goal_fail(self):
        from src.state import Goal, GoalStatus
        g = Goal(title="Fail goal", description="This will fail")
        g.fail("ran out of time")
        assert g.status == GoalStatus.FAILED

    def test_skill_success_rate(self):
        from src.state import Skill
        sk = Skill(name="my_skill", description="d", skill_type="code_module", content="x=1")
        assert sk.success_rate == sk.performance_score
        sk.usage_count = 10
        sk.success_count = 7
        assert abs(sk.success_rate - 0.7) < 0.001


# ─── Unit: Memory ────────────────────────────────────────────────

class TestMemory:
    def test_store_and_retrieve_experience(self, memory):
        eid = memory.store_experience(
            agent_id="a1", generation=0, goal="research AI",
            strategy="react", success=True, outcome="found papers",
            score=0.9, tokens_used=1500,
        )
        assert eid
        results = memory.search_experiences("research AI", limit=5)
        assert any(r["goal"] == "research AI" for r in results)

    def test_store_and_get_skill(self, memory):
        memory.store_skill(
            name="test_skill", description="a test skill",
            skill_type="prompt_template", content="Hello {name}",
            performance=0.7,
        )
        sk = memory.get_skill("test_skill")
        assert sk is not None
        assert sk["name"] == "test_skill"
        assert sk["skill_type"] == "prompt_template"

    def test_skill_update_on_conflict(self, memory):
        memory.store_skill("dup_skill", "v1", "new_tool", "code_v1")
        memory.store_skill("dup_skill", "v2", "new_tool", "code_v2")   # upsert
        sk = memory.get_skill("dup_skill")
        assert sk["description"] == "v2"

    def test_record_skill_use(self, memory):
        memory.store_skill("used_skill", "d", "strategy", "content")
        memory.record_skill_use("used_skill", True)
        memory.record_skill_use("used_skill", False)
        sk = memory.get_skill("used_skill")
        assert sk["usage_count"] == 2
        assert sk["success_count"] == 1

    def test_store_and_find_workflow(self, memory):
        memory.store_workflow(
            name="data_pipeline",
            description="ETL workflow",
            trigger_pattern="data pipeline",
            steps=[{"step": 1, "action": "extract"}],
        )
        wf = memory.find_workflow("data pipeline")
        assert wf is not None
        assert wf["name"] == "data_pipeline"

    def test_preferences(self, memory):
        memory.set_preference("a1", "verbosity", "high")
        memory.set_preference("a1", "language", "python")
        prefs = memory.get_preferences("a1")
        assert prefs["verbosity"] == "high"
        assert prefs["language"] == "python"

    def test_stats(self, memory):
        memory.store_experience("a1", 0, "g1", "react", True, score=0.8, tokens_used=100)
        memory.store_experience("a1", 0, "g2", "react", False, score=0.3, tokens_used=200)
        stats = memory.stats("a1")
        assert stats["experiences"]["total"] == 2
        assert stats["experiences"]["successes"] == 1

    def test_evolution_log(self, memory):
        eid = memory.log_evolution(
            agent_id="a1", from_gen=0, to_gen=1,
            skill_name="new_tool", change_type="new_tool",
            description="Added a new scraper", test_passed=True, deployed=True,
        )
        assert eid
        log = memory.get_evolution_log("a1")
        assert len(log) == 1
        assert log[0]["skill_name"] == "new_tool"


# ─── Unit: Tools ─────────────────────────────────────────────────

class TestTools:
    @pytest.mark.asyncio
    async def test_code_executor_success(self):
        from src.tools import CodeExecutor
        r = await CodeExecutor().execute(code="print('hello world')")
        assert r.success
        assert "hello world" in r.output

    @pytest.mark.asyncio
    async def test_code_executor_syntax_error(self):
        from src.tools import CodeExecutor
        r = await CodeExecutor().execute(code="def broken(:")
        assert not r.success
        assert "SyntaxError" in r.error

    @pytest.mark.asyncio
    async def test_code_executor_runtime_error(self):
        from src.tools import CodeExecutor
        r = await CodeExecutor().execute(code="raise ValueError('oops')")
        assert not r.success

    @pytest.mark.asyncio
    async def test_code_executor_timeout(self):
        from src.tools import CodeExecutor
        r = await CodeExecutor().execute(code="import time; time.sleep(100)", timeout=1)
        assert not r.success
        assert "Timed out" in r.error

    @pytest.mark.asyncio
    async def test_code_validator_clean(self):
        from src.tools import CodeValidator
        r = await CodeValidator().execute(code="x = [i**2 for i in range(10)]")
        assert r.success
        assert "No security issues" in r.output

    @pytest.mark.asyncio
    async def test_code_validator_warnings(self):
        from src.tools import CodeValidator
        r = await CodeValidator().execute(code="result = eval(user_input)")
        assert r.success   # still returns True — it's a warning not a block
        assert "eval" in r.output.lower()

    @pytest.mark.asyncio
    async def test_file_read_write(self, tmp_path):
        from src.tools import FileReader, FileWriter
        p = str(tmp_path / "test.txt")
        wr = await FileWriter().execute(path=p, content="hello\nworld\n")
        assert wr.success
        rd = await FileReader().execute(path=p)
        assert rd.success
        assert "hello" in rd.output

    @pytest.mark.asyncio
    async def test_file_reader_missing(self, tmp_path):
        from src.tools import FileReader
        r = await FileReader().execute(path=str(tmp_path / "nonexistent.txt"))
        assert not r.success

    @pytest.mark.asyncio
    async def test_file_writer_creates_dirs(self, tmp_path):
        from src.tools import FileWriter
        deep = str(tmp_path / "a" / "b" / "c" / "file.txt")
        r = await FileWriter().execute(path=deep, content="deep content")
        assert r.success
        assert Path(deep).exists()

    @pytest.mark.asyncio
    async def test_self_inspector_list(self, tmp_path):
        from src.tools import SelfInspector
        (tmp_path / "foo.py").write_text("x = 1")
        (tmp_path / "bar.py").write_text("y = 2")
        r = await SelfInspector(str(tmp_path)).execute(file="list")
        assert r.success
        assert "foo.py" in r.output

    @pytest.mark.asyncio
    async def test_memory_search_tool(self, memory):
        from src.tools import MemorySearchTool
        memory.store_experience("a1", 0, "quantum computing research", "react", True)
        tool = MemorySearchTool(memory)
        r = await tool.execute(query="quantum computing")
        assert r.success

    @pytest.mark.asyncio
    async def test_tool_registry_unknown(self, tool_registry):
        r = await tool_registry.call("totally_unknown_tool", x=1)
        assert not r.success
        assert "Unknown tool" in r.error


# ─── Unit: SkillRunner ────────────────────────────────────────────

class TestSkillRunner:
    @pytest.mark.asyncio
    async def test_prompt_template_skill(self, memory):
        from src.tools import CodeExecutor
        from src.skills import SkillRunner
        memory.store_skill(
            name="greeting_template",
            description="A greeting",
            skill_type="prompt_template",
            content="Hello, {name}! Welcome to {place}.",
        )
        runner = SkillRunner(memory, CodeExecutor())
        r = await runner.execute(
            skill_name="greeting_template",
            inputs={"name": "Alice", "place": "Turing"},
        )
        assert r.success
        assert "Alice" in r.output

    @pytest.mark.asyncio
    async def test_unknown_skill(self, memory):
        from src.tools import CodeExecutor
        from src.skills import SkillRunner
        runner = SkillRunner(memory, CodeExecutor())
        r = await runner.execute(skill_name="does_not_exist_xyz")
        assert not r.success

    @pytest.mark.asyncio
    async def test_code_skill(self, memory):
        from src.tools import CodeExecutor
        from src.skills import SkillRunner
        memory.store_skill(
            name="sum_printer",
            description="Prints a sum",
            skill_type="new_tool",
            content="print(sum(range(10)))",
        )
        runner = SkillRunner(memory, CodeExecutor())
        r = await runner.execute(skill_name="sum_printer")
        assert r.success
        assert "45" in r.output


# ─── Unit: PromptLibrary ─────────────────────────────────────────

class TestPromptLibrary:
    def test_seed_templates(self, memory):
        from src.skills import seed_prompt_library
        added = seed_prompt_library(memory)
        assert added >= 4
        sk = memory.get_skill("react_step_executor")
        assert sk is not None

    def test_seed_idempotent(self, memory):
        from src.skills import seed_prompt_library
        first = seed_prompt_library(memory)
        second = seed_prompt_library(memory)
        assert second == 0   # already seeded


# ─── Unit: ContextManager ────────────────────────────────────────

class TestContextManager:
    def test_trim_long_history(self, mock_llm):
        from src.context_manager import ContextManager
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        cm = ContextManager(mock_llm, max_messages=10)
        msgs = [SystemMessage(content="System")] + [
            HumanMessage(content=f"Message {i}") for i in range(20)
        ]
        trimmed = cm.trim(msgs)
        assert len(trimmed) <= 11

    def test_trim_long_observations(self, mock_llm):
        from src.context_manager import ContextManager
        from langchain_core.messages import HumanMessage
        cm = ContextManager(mock_llm, max_observation_len=100)
        long_msg = HumanMessage(content="x" * 500)
        trimmed = cm.trim([long_msg])
        assert len(trimmed[0].content) <= 200

    def test_context_block_format(self, mock_llm):
        from src.context_manager import ContextManager
        from src.state import initial_state
        cm = ContextManager(mock_llm)
        state = initial_state("Analyse quantum computing trends")
        block = cm.build_context_block(state)
        assert "Analyse quantum computing" in block

    def test_token_estimate(self, mock_llm):
        from src.context_manager import ContextManager
        cm = ContextManager(mock_llm)
        est = cm.estimate_tokens("a" * 400)
        assert 90 <= est <= 110


# ─── Integration: Nodes ──────────────────────────────────────────

class TestNodes:
    def _make_perceive_llm(self):
        """LLM that returns a valid perception analysis."""
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(content=json.dumps({
            "complexity": "moderate",
            "strategy": "react",
            "needs_tools": True,
            "should_spawn_sub_agents": False,
            "sub_agent_goals": [],
            "estimated_steps": 3,
            "key_challenges": ["getting current data"],
        })))
        return llm

    def _make_plan_llm(self):
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(content=json.dumps([
            {
                "id": "step_001",
                "description": "Search for information",
                "strategy": "react",
                "tools_needed": ["web_search"],
                "success_criteria": "Found relevant results",
                "fallback": "Use cached knowledge",
                "max_retries": 2,
            }
        ])))
        return llm

    @pytest.mark.asyncio
    async def test_perceive_node(self, memory, tool_registry):
        from src.nodes import perceive
        from src.state import initial_state
        llm = self._make_perceive_llm()
        state = initial_state("Research AI agent trends")
        result = await perceive(state=state, llm=llm, memory=memory, tool_registry=tool_registry)
        assert result["phase"] == "plan"
        assert "strategy" in result

    @pytest.mark.asyncio
    async def test_plan_node(self, memory, tool_registry):
        from src.nodes import plan
        from src.state import initial_state, Phase
        llm = self._make_plan_llm()
        state = initial_state("Research AI agent trends")
        state["phase"] = Phase.PLAN.value
        result = await plan(state=state, llm=llm, memory=memory, tool_registry=tool_registry)
        assert result["phase"] == Phase.ACT.value
        assert len(result["plan"]) >= 1
        assert result["current_step"] is not None

    @pytest.mark.asyncio
    async def test_act_node_no_tools(self, memory, tool_registry):
        from src.nodes import act
        from src.state import initial_state, Phase
        # LLM returns FINAL ANSWER immediately
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(
            content="THOUGHT: I can answer this directly.\nFINAL ANSWER: The answer is 42."
        ))
        state = initial_state("What is 6*7?")
        state["phase"] = Phase.ACT.value
        state["current_step"] = {
            "id": "step_001",
            "description": "Calculate 6*7",
            "tools_needed": [],
            "success_criteria": "Provides the answer",
            "fallback": "Manual calculation",
            "status": "pending",
            "goal_id": "g1",
        }
        result = await act(state=state, llm=llm, memory=memory, tool_registry=tool_registry)
        assert result["phase"] == Phase.OBSERVE.value
        assert result["iteration_count"] == 1

    @pytest.mark.asyncio
    async def test_observe_marks_completion(self, memory, tool_registry):
        from src.nodes import observe
        from src.state import initial_state, Phase
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(content=json.dumps({
            "success": True,
            "confidence": "high",
            "issues": [],
            "next_action": "continue",
        })))
        state = initial_state("Simple task")
        state["phase"] = Phase.OBSERVE.value
        step = {"id": "s1", "description": "Done step", "status": "done",
                "result": "completed", "tools_called": [], "success_criteria": "done"}
        state["current_step"] = step
        state["plan"] = [step]
        state["completed_steps"] = []
        state["pending_steps"] = []
        result = await observe(state=state, llm=llm, memory=memory, tool_registry=tool_registry)
        # No pending steps → should complete
        assert result.get("is_complete") is True


# ─── Integration: Evolution ───────────────────────────────────────

class TestEvolution:
    @pytest.mark.asyncio
    async def test_evolution_analyse(self, memory, tool_registry, tmp_path):
        from src.evolution import SelfEvolutionEngine
        from src.state import initial_state

        # LLM returns minimal proposals
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(content=json.dumps([
            {
                "type": "prompt_template",
                "name": "test_prompt",
                "description": "A better prompt",
                "rationale": "Current prompts are too verbose",
                "estimated_impact": 0.7,
            }
        ])))

        src_dir = str(Path(__file__).parent.parent / "src")
        engine = SelfEvolutionEngine(
            llm=llm, memory=memory, tool_registry=tool_registry,
            source_dir=src_dir, evolved_dir=str(tmp_path / "evolved"),
            agent_id="test_agent",
        )
        state = initial_state("test goal")
        analysis = await engine.analyse_codebase(state)
        assert "proposals_raw" in analysis
        assert len(analysis["proposals_raw"]) >= 1

    @pytest.mark.asyncio
    async def test_evolution_generates_prompt_skill(self, memory, tool_registry, tmp_path):
        from src.evolution import SelfEvolutionEngine
        from src.state import initial_state

        proposal_content = "Do this: {input}"
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(content=json.dumps({
            "code": proposal_content,
            "test": f"""
def test_my_prompt():
    template = \"\"\"{proposal_content}\"\"\"
    result = template.format(input="hello")
    assert result == "Do this: hello"
    return True
""",
        })))

        src_dir = str(Path(__file__).parent.parent / "src")
        engine = SelfEvolutionEngine(
            llm=llm, memory=memory, tool_registry=tool_registry,
            source_dir=src_dir, evolved_dir=str(tmp_path / "evolved"),
            agent_id="test_agent",
        )
        state = initial_state("improve prompts")
        raw = {"type": "prompt_template", "name": "my_prompt",
               "description": "test prompt", "rationale": "needs improvement",
               "estimated_impact": 0.6}
        proposal = await engine.generate_skill(raw, state)
        assert proposal.name == "my_prompt"
        assert proposal_content in proposal.content

        passed, output = await engine.test_proposal(proposal)
        assert passed, f"Test failed: {output}"


# ─── E2E: Full agent run (requires API key) ───────────────────────

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_e2e_simple_goal(tmp_path):
    """
    Run a full agent cycle with a simple goal.
    Requires LLM_API_KEY and LLM_PROVIDER env vars.
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic")
    api_key = os.getenv("LLM_API_KEY", os.getenv("ANTHROPIC_API_KEY", ""))
    model = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")

    if not api_key:
        pytest.skip("No API key set — skipping E2E test")

    # Create LLM
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model=model, api_key=api_key, temperature=0.3, max_tokens=4096)
    else:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=model, api_key=api_key, temperature=0.3)

    from src.memory import Memory
    from src.tools import create_registry
    from src.orchestrator import TuringAgent

    memory = Memory(memory_dir=str(tmp_path / "memory"))
    source_dir = str(Path(__file__).parent.parent)
    tool_registry = create_registry(source_dir=source_dir, memory=memory)

    agent = TuringAgent(
        llm=llm,
        memory=memory,
        tool_registry=tool_registry,
        source_dir=source_dir,
        evolved_dir=str(tmp_path / "evolved"),
        agent_id="e2e_test",
        max_iterations=10,
        max_retries=2,
        enable_evolution=False,   # keep test fast
        verbose=True,
    )

    result = await agent.run("Write and run a Python function that returns the first 5 Fibonacci numbers")
    assert result.get("final_output"), "No final output produced"
    assert result.get("iteration_count", 0) > 0, "No iterations ran"

    memory.close()
    print(f"\nE2E result: {result.get('final_output','')[:200]}")
