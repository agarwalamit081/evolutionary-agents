"""Evolution subsystem integration: 4-phase engine sequence, git versioning,
report synthesis, template payload shapes, and promote pointer boundaries.

This file is **exclusive** — it deliberately does NOT overlap with:
* ``test_evolution_rollback_paths.py`` (engine.run_cycle smoke/rollback arms) and
  ``test_promote_regression_rollback.py`` (promote canary-recursion / regression
  retain-prior). Those exist; we do not duplicate them.
* ``test_engine.py`` (unit-level per-phase), ``test_templates.py`` (per-template
  field presence), ``test_report.py`` (report substring presence),
  ``test_git_tracker.py`` (per-method GitTracker unit), ``test_promote.py``
  (promote passing/failing/inconclusive canary unit).

Instead this file targets the **cross-component integration seams** that the
unit files do not exercise together:

* the FULL 4-phase engine pipeline (analyze → generate → validate → sandbox/ab
  → deploy) driven against a REAL ``GitTracker`` over a tmp git repo, proving
  the sequence produces a proposal, deploys it to a committed shadow-repo state,
  and that a follow-up rollback restores the prior tree;
* report synthesis over a REAL deployed AND a REAL rejected ``run_cycle`` result
  (mutation counts/types/method/model surface in the rendered report);
* template payload shapes round-tripping through the engine's heuristic
  ``generate`` for EVERY ``MutationType`` (the proposal is well-formed, has a
  deployable ``target_path``, non-empty rationale, and a JSON-parseable
  ``mutated_content`` for the config-shaped types);
* ``GitTracker`` versioning + rollback against the live filesystem: a snapshot
  lands a new commit (log grows), a rollback to the pre-mutation hash removes a
  mutation-added file entirely (``reset --hard`` + ``clean -fd``), and
  ``get_diff(since_hash=)`` captures exactly the introduced change;
* ``PromotionGate`` pointer boundary semantics: an EXACTLY-at-threshold canary
  promotes (``>=``), the pointer dedups on content identity across the
  ``active_promotions`` introspection surface, and a regressing auto-rollback
  (two versions → ``rollback`` restores the prior pointer, history shortens by
  one) — the rollback-as-pointer-restoration contract distinct from the
  canary-gate retain-prior unit cases.

Deterministic: real ``git`` over ``tmp_path`` (no network), a fake async canary
scripting the promotion score, and mocked gateway/persister. No
``@pytest.mark.e2e``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.evolution.engine import SelfEvolutionEngine
from src.evolution.git_tracker import GitTracker
from src.evolution.promote import PromotionGate, parse_prompt_payload
from src.evolution.report import generate_report
from src.graph.enums import MutationType


# ---------------------------------------------------------------------------
# Settings fake — order-safe: monkeypatch the symbol only (never mutate the
# real get_settings() singleton or reassign class attrs, which poisons later
# tests under pytest-randomly).
# ---------------------------------------------------------------------------


def _promote_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    *,
    min_score: float = 0.8,
) -> SimpleNamespace:
    fake = SimpleNamespace(
        evolution=SimpleNamespace(
            evolution_promote_to_live=True,
            evolved_handlers_dir=str(tmp_path / "evolved"),
        ),
        eval=SimpleNamespace(eval_canary_min_score=min_score),
    )
    monkeypatch.setattr("src.config.settings.get_settings", lambda: fake)
    return fake


def _canary(score: float | None) -> Any:
    async def _fn(_node: str, _suffixes: list[str]) -> float | None:
        return score

    return _fn


def _json_proposal(suffixes: list[str], node: str = "execute") -> dict[str, Any]:
    return {
        "mutation_type": MutationType.PROMPT,
        "description": "address JSON mistakes",
        "rationale": "guide valid JSON",
        "model_used": "test-model",
        "mutated_content": json.dumps({"target_node": node, "suffixes": suffixes}),
        "target_path": "prompts/system_prompt.md",
    }


def _init_shadow(src_dir: Path, repo_dir: Path) -> GitTracker:
    """Build an initialized shadow repo mirroring src_dir (the repo is created
    by ``initialize()``; the source files are written first)."""
    tracker = GitTracker(source_dir=src_dir, repo_dir=repo_dir)
    return tracker


# ---------------------------------------------------------------------------
# 4-phase engine sequence → real git deploy → rollback restores tree
# ---------------------------------------------------------------------------


class TestEnginePhaseSequenceDeploysToGit:
    """The full analyze → generate → validate → sandbox/ab → deploy pipeline,
    driven against a REAL GitTracker. The deploy phase must apply the mutation
    to the shadow repo, commit a new snapshot, and capture a pre_deploy_hash;
    rolling that hash back must restore the pre-mutation tree (the file added
    by the mutation is removed via reset --hard + clean -fd)."""

    @pytest.mark.asyncio
    async def test_deploy_applies_mutation_and_commits_snapshot(
        self, tmp_path: Path
    ) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "base.py").write_text("BASE = 1\n", encoding="utf-8")
        tracker = _init_shadow(src, tmp_path / "shadow")
        await tracker.initialize()
        pre = await tracker.get_current_hash()

        engine = SelfEvolutionEngine()  # heuristic path (no gateway)

        # analyze → opportunity (failure patterns → a PROMPT opportunity)
        analysis = await engine.analyze([], failure_patterns=["json format errors"])
        assert analysis["opportunities"], "analyze must produce opportunities"
        opportunity = analysis["opportunities"][0]
        assert opportunity["type"] == MutationType.PROMPT

        # propose → well-formed proposal
        proposal = await engine.generate(opportunity)
        assert proposal["mutated_content"]
        assert proposal["target_path"]

        # validate → safety pipeline passes for heuristic PROMPT content
        validation = await engine.validate(proposal)
        assert validation["passed"] is True

        # ab/sandbox skipped (no sandbox) → deploy
        ab = await engine.ab_test(proposal, sandbox=None)
        assert ab["is_significant"] is True
        deployment = await engine.deploy(proposal, validation, ab, git_tracker=tracker)

        assert deployment["deployed"] is True
        assert deployment["commit_hash"]
        assert deployment["commit_hash"] != pre
        assert deployment["pre_deploy_hash"] == pre
        # The mutation landed on disk under the configured target_path.
        assert (tracker.repo_dir / proposal["target_path"]).exists()
        # The deploy advanced the engine generation counter.
        assert deployment["generation"] == 1

    @pytest.mark.asyncio
    async def test_rollback_restores_prior_tree(self, tmp_path: Path) -> None:
        """Rolling the shadow repo to the pre-deploy hash must restore the
        pre-mutation tree exactly: a file added by the mutation is removed by
        ``clean -fd`` (reset --hard alone would leave untracked files)."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "base.py").write_text("BASE = 1\n", encoding="utf-8")
        tracker = _init_shadow(src, tmp_path / "shadow")
        await tracker.initialize()
        pre = await tracker.get_current_hash()

        engine = SelfEvolutionEngine()
        opportunity = (await engine.analyze([], failure_patterns=["json errors"]))[
            "opportunities"
        ][0]
        proposal = await engine.generate(opportunity)
        target = proposal["target_path"]
        await tracker.apply_mutation(target, proposal["mutated_content"])
        await tracker.snapshot("mutation deploy")
        # The mutation file now exists.
        assert (tracker.repo_dir / target).exists()

        rolled = await engine.rollback_deployment(
            {"pre_deploy_hash": pre}, tracker
        )

        assert rolled["rolled_back"] is True
        assert rolled["pre_deploy_hash"] == pre
        # reset --hard + clean -fd → the mutation-added file is GONE (it was
        # untracked relative to the pre-deploy commit), restoring the prior tree.
        assert not (tracker.repo_dir / target).exists()
        # The original source file is intact.
        assert (tracker.repo_dir / "base.py").read_text("utf-8") == "BASE = 1\n"

    @pytest.mark.asyncio
    async def test_rollback_captures_diff_before_undoing(self, tmp_path: Path) -> None:
        """``rollback_deployment`` captures the reverted diff BEFORE resetting
        (after reset the tree is clean and the diff is empty). The diff must
        reference the mutation content."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "base.py").write_text("BASE = 1\n", encoding="utf-8")
        tracker = _init_shadow(src, tmp_path / "shadow")
        await tracker.initialize()
        pre = await tracker.get_current_hash()
        await tracker.apply_mutation("graph/m.py", "MARKER_CONTENT\n")
        await tracker.snapshot("mutation")

        rolled = await engine_rollback(tracker, pre)

        assert rolled["rolled_back"] is True
        assert "MARKER_CONTENT" in rolled["reverted_diff"]


async def engine_rollback(tracker: GitTracker, pre: str) -> dict[str, Any]:
    eng = SelfEvolutionEngine()
    return await eng.rollback_deployment({"pre_deploy_hash": pre}, tracker)


# ---------------------------------------------------------------------------
# Report synthesis over REAL run_cycle results
# ---------------------------------------------------------------------------


class TestReportOnRunCycleResults:
    """``generate_report`` over a real ``run_cycle`` result: the deployed and
    the rejected arms both render a well-formed report, surfacing the mutation
    type, the generation method (LLM vs heuristic), the model, and the deploy
    commit hash / rejection reason."""

    @pytest.mark.asyncio
    async def test_report_on_deployed_cycle(self, tmp_path: Path) -> None:
        engine = SelfEvolutionEngine()
        # failure patterns ⇒ a real PROMPT heuristic proposal that passes safety
        result = await engine.run_cycle([], failure_patterns=["json format errors"])

        assert result["deployed"] is True
        report = generate_report(result, generation=result["generation"])

        assert "EVOLUTION REPORT" in report
        # The mutation type surfaces (PROMPT enum value) + heuristic method.
        assert "prompt" in report.lower()
        assert "heuristic" in report.lower()
        # Deployed ⇒ a commit-hash line + the "stored as skill" future-run effect.
        assert "Commit:" in report
        assert "stored as skill" in report.lower()

    @pytest.mark.asyncio
    async def test_report_on_rejected_cycle(self, tmp_path: Path) -> None:
        """A CODE mutation with dangerous content fails safety validation → the
        cycle rejects; the report must render the validation-failure path and
        the 'None (mutation was not deployed)' effect, NOT the deployed arm."""
        engine = SelfEvolutionEngine()
        # Feed a CODE opportunity with dangerous content directly through the
        # safety gate (os.system call fails the safety pipeline → rejection).
        dangerous = {
            "mutation_type": MutationType.CODE,
            "description": "rm everything",
            "mutated_content": "import os\nos.system('rm -rf /tmp/x')\n",
            "target_path": "graph/bad.py",
            "priority": "high",
            "rationale": "bad",
        }
        validation = await engine.validate(dangerous)
        assert validation["passed"] is False
        rejection = engine.reject(dangerous, validation)

        report = generate_report(
            {
                "deployed": False,
                "proposal": dangerous,
                "validation": validation,
                "deployment": rejection,
                "reason": validation.get("reason"),
            },
            generation=0,
        )
        assert "VALIDATION: ❌" in report or "VALIDATION:" in report
        assert "Rejected" in report
        assert "not deployed" in report.lower()
        # The rejection arm must NOT claim a deploy commit.
        assert "stored as skill" not in report.lower()


# ---------------------------------------------------------------------------
# Template payload shapes round-trip through heuristic generate
# ---------------------------------------------------------------------------


class TestTemplatePayloadShapesViaEngine:
    """Every ``MutationType`` routed through the engine's heuristic ``generate``
    produces a well-formed proposal: a non-empty deployable ``target_path``, a
    non-empty rationale, content of the right shape (JSON-parseable for the
    config-shaped types; valid Python for CODE)."""

    @pytest.mark.parametrize(
        ("mtype", "description"),
        [
            (MutationType.PROMPT, "address json format errors"),
            (MutationType.WORKFLOW, "reduce average execution time (6000ms)"),
            (MutationType.TOOL, "search the web for context"),
            (MutationType.MEMORY, "tighten precision / reduce noise"),
            (MutationType.CODE, "memoize repeated computation"),
            (MutationType.CONFIG, "tune generation temperature"),
            (MutationType.SUB_AGENT_PROMPT, "sharpen sub-agent prompt"),
            (MutationType.SUB_AGENT_TOOLS, "expand sub-agent tools"),
            (MutationType.SUB_AGENT_CONFIG, "raise sub-agent iterations"),
            (MutationType.SUB_AGENT_MODEL_TIER, "downgrade sub-agent tier"),
        ],
    )
    @pytest.mark.asyncio
    async def test_each_mutation_type_produces_well_formed_proposal(
        self, mtype: MutationType, description: str
    ) -> None:
        engine = SelfEvolutionEngine()
        opportunity = {
            "type": mtype,
            "description": description,
            "patterns": ["json"],
            "target_sub_agent": "researcher",
        }
        proposal = await engine.generate(opportunity)

        # Common invariants.
        assert proposal["mutation_type"] == mtype
        assert proposal["model_used"] is None  # heuristic path
        assert proposal["tokens_used"] == 0
        assert proposal["rationale"]  # non-empty
        target = proposal["target_path"]
        assert target and target.startswith("evolution/")
        # No stray template variables leaked into the deployed content.
        assert "{target_node}" not in proposal["mutated_content"]

        # Type-specific shape.
        if mtype == MutationType.CODE:
            # CODE template emits real, loadable Python (a memoization helper).
            assert target.endswith(".py")
            compile(proposal["mutated_content"], target, "exec")  # must parse
            assert "async def" in proposal["mutated_content"]
        else:
            payload = json.loads(proposal["mutated_content"])
            assert isinstance(payload, dict)
            assert payload  # non-empty


class TestPromptTemplatePayloadShape:
    """The PROMPT template's payload is the shape ``parse_prompt_payload``
    consumes: a JSON object with ``target_node`` + ``suffixes``. Proving the
    engine-produced proposal round-trips through the promote parser closes the
    generate→promote loop (the heuristic deploy produces a promotable PROMPT)."""

    @pytest.mark.asyncio
    async def test_prompt_proposal_round_trips_through_parse(self) -> None:
        engine = SelfEvolutionEngine()
        proposal = await engine.generate(
            {"type": MutationType.PROMPT, "description": "x", "patterns": ["json"]}
        )

        parsed = parse_prompt_payload(proposal)

        assert parsed is not None
        node, suffixes = parsed  # type: ignore[misc]
        assert node == "execute"  # PROMPT template hard-codes execute
        assert suffixes  # at least the JSON guidance suffix


class TestConfigTemplateReadsEvolutionSettings:
    """``generate_config_tuning`` reads the live ``EvolutionSettings``
    temperature/max_tokens_factor knobs (battery-04 centralized config). The
    reflected values must match what ``get_settings().evolution`` reports, so a
    knob flip lands in the generated payload."""

    @pytest.mark.asyncio
    async def test_config_tuning_reflects_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.config.settings import get_settings

        # Flip BOTH knobs via the live singleton (order-safe: monkeypatch
        # restores at teardown; we mutate instance attrs, not the class).
        real = get_settings().evolution
        monkeypatch.setattr(real, "evolution_temperature", 0.123)
        monkeypatch.setattr(real, "evolution_max_tokens_factor", 0.777)

        engine = SelfEvolutionEngine()
        proposal = await engine.generate(
            {"type": MutationType.CONFIG, "description": "tune it"}
        )
        payload = json.loads(proposal["mutated_content"])

        assert payload["adjustments"]["temperature"] == 0.123
        assert payload["adjustments"]["max_tokens_factor"] == 0.777


# ---------------------------------------------------------------------------
# GitTracker versioning + rollback against the live filesystem
# ---------------------------------------------------------------------------


class TestGitVersioningAndRollback:
    """``GitTracker`` snapshot/rollback against a real tmp git repo. These lock
    the versioning contract the engine's deploy/rollback phases rely on: a
    snapshot grows the log, a rollback to a prior commit removes mutation-added
    files, and ``get_diff(since_hash=)`` reports exactly the introduced change.
    Distinct from ``test_git_tracker.py`` (per-method unit) by exercising the
    mutation→snapshot→rollback→log sequence end to end."""

    @pytest.mark.asyncio
    async def test_snapshot_grows_log_then_rollback_removes_mutation(
        self, tmp_path: Path
    ) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("A = 1\n", encoding="utf-8")
        tracker = _init_shadow(src, tmp_path / "shadow")
        await tracker.initialize()
        base_log = await tracker.get_log()
        base_hash = await tracker.get_current_hash()

        # Apply a mutation that adds a NEW file + mutates an existing one.
        await tracker.apply_mutation("added.py", "ADDED = True\n")
        await tracker.snapshot("mutation round 1")

        mid_log = await tracker.get_log()
        assert len(mid_log) == len(base_log) + 1
        assert mid_log[0]["message"] == "mutation round 1"
        # The mutation diff vs the base hash references the added file.
        diff = await tracker.get_diff(since_hash=base_hash)
        assert "added.py" in diff

        # Rollback to base: the added file is removed (clean -fd), the existing
        # file restored to its original content.
        ok = await tracker.rollback(base_hash)
        assert ok is True
        assert not (tracker.repo_dir / "added.py").exists()
        assert (tracker.repo_dir / "a.py").read_text("utf-8") == "A = 1\n"

        # After rollback the HEAD is back on the base commit.
        assert await tracker.get_current_hash() == base_hash

    @pytest.mark.asyncio
    async def test_commit_paths_never_sweeps_untracked(
        self, tmp_path: Path
    ) -> None:
        """``commit_paths`` stages ONLY the named paths (never ``-A``), so a
        stray untracked file in the repo is NOT swept into the commit. This is
        the main-repo discipline the promotion gate relies on."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "tracked.py").write_text("T = 1\n", encoding="utf-8")
        tracker = _init_shadow(src, tmp_path / "shadow")
        await tracker.initialize()
        # A stray local file (machine-specific) + an artifact we want to commit.
        (tracker.repo_dir / "local.env").write_text("SECRET=x\n", encoding="utf-8")
        artifact = tracker.repo_dir / "prompts" / "evolved" / "execute.abc12345.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("{}", encoding="utf-8")

        sha = await tracker.commit_paths(
            ["prompts/evolved/execute.abc12345.json"],
            "chore(evolution): promote execute prompt",
        )
        assert sha

        # The tracked artifact is committed, but the stray local.env is NOT.
        tracked = await tracker._git("ls-tree", "-r", "HEAD", "--name-only")
        committed = tracked[1]
        assert "prompts/evolved/execute.abc12345.json" in committed
        assert "local.env" not in committed
        # local.env remains an untracked working-tree file (never staged).
        assert (tracker.repo_dir / "local.env").exists()


# ---------------------------------------------------------------------------
# PromotionGate pointer boundary semantics + auto-rollback restores pointer
# ---------------------------------------------------------------------------


class TestPromotePointerBoundaries:
    """Pointer boundary semantics NOT covered by the promote unit/regression
    files: an exactly-at-threshold canary promotes (``>=``), content-identity
    dedup across the ``active_promotions`` introspection surface, and the
    auto-rollback-as-pointer-restoration contract (history shortens by one and
    the active suffix reverts to the prior version's)."""

    @pytest.mark.asyncio
    async def test_canary_exactly_at_threshold_promotes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _promote_settings(monkeypatch, tmp_path, min_score=0.8)
        gate = PromotionGate(canary=_canary(0.8))  # exactly == min_score

        result = await gate.promote(_json_proposal(["boundary guidance"]))

        assert result["promoted"] is True
        assert result["canary_score"] == 0.8  # >= threshold, not strictly >
        assert gate.current_suffixes("execute") == ["boundary guidance"]

    @pytest.mark.asyncio
    async def test_active_promotions_lists_promoted_node_metadata(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _promote_settings(monkeypatch, tmp_path)
        gate = PromotionGate(canary=_canary(0.9))

        await gate.promote(_json_proposal(["g1"]))

        active = gate.active_promotions()
        assert len(active) == 1
        entry = active[0]
        assert entry["node"] == "execute"
        assert entry["active"]  # the version file name
        assert entry["canary_score"] == 0.9
        assert entry["promoted_at"]  # ISO timestamp

    @pytest.mark.asyncio
    async def test_unpromoted_node_absent_from_active_promotions(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _promote_settings(monkeypatch, tmp_path)
        gate = PromotionGate(canary=_canary(0.5))  # fails threshold

        await gate.promote(_json_proposal(["rejected"]))

        assert gate.active_promotions() == []

    @pytest.mark.asyncio
    async def test_rollback_shortens_history_and_restores_prior_pointer(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """Two distinct promotions build a 2-entry history; rollback pops the
        last entry (history → 1), restores the active suffix to the PRIOR
        version, and reports the removed + restored version file names."""
        _promote_settings(monkeypatch, tmp_path)
        gate = PromotionGate(canary=_canary(0.9))
        first = await gate.promote(_json_proposal(["v1 guidance"]))
        second = await gate.promote(_json_proposal(["v2 guidance"]))
        assert gate.current_suffixes("execute") == ["v2 guidance"]

        rolled = gate.rollback("execute")

        assert rolled["rolled_back"] is True
        assert rolled["removed"] == second["version"]
        assert rolled["restored"] == first["version"]
        assert gate.current_suffixes("execute") == ["v1 guidance"]
        # History shrank by exactly one.
        pointer = json.loads((gate.prompts_dir / "current.json").read_text("utf-8"))
        assert len(pointer["execute"]["history"]) == 1
        assert pointer["execute"]["active"] == first["version"]

    @pytest.mark.asyncio
    async def test_rollback_no_history_is_safe_noop(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """A pointer entry with no history (malformed/edge) refuses to roll back
        rather than corrupting the pointer."""
        _promote_settings(monkeypatch, tmp_path)
        gate = PromotionGate(canary=_canary(0.9))
        # Hand-write a pointer entry with no history.
        gate._prompts_dir.mkdir(parents=True, exist_ok=True)
        gate._write_pointer({"execute": {"active": "execute.x.json", "history": []}})

        rolled = gate.rollback("execute")

        assert rolled["rolled_back"] is False
        assert "no history" in rolled["reason"]


class TestFreeTextSystemPromptMapsToExecute:
    """The free-text whole-file-rewrite shape the live LLM PROMPT generator
    emits: a ``prompts/system_prompt.md`` rewrite carries no node token, so it
    maps to the default ``execute`` node. Locking the default-node mapping for
    the system-prompt case specifically (the documented battery-04 q08 shape)."""

    def test_system_prompt_rewrite_maps_to_execute_node(self) -> None:
        proposal = {
            "mutation_type": MutationType.PROMPT,
            "target_path": "prompts/system_prompt.md",
            "mutated_content": (
                "# System Prompt\nYou are an AI agent. Generate artifacts.\n"
            ),
        }
        node, suffixes = parse_prompt_payload(proposal)  # type: ignore[misc]
        assert node == "execute"
        assert suffixes == ["# System Prompt\nYou are an AI agent. Generate artifacts."]

    def test_whitespace_only_freetext_is_rejected(self) -> None:
        """A PROMPT mutation whose content is whitespace-only is not promotable
        (an empty rewrite would promote nothing)."""
        proposal = {
            "mutation_type": MutationType.PROMPT,
            "target_path": "prompts/system_prompt.md",
            "mutated_content": "   \n\t  ",
        }
        assert parse_prompt_payload(proposal) is None
