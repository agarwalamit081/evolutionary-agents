"""#8 / G1 — CODE→core-src promotion-candidate scaffold (``src.evolution.promote``).

Covers the CODE counterpart of the PROMPT promotion gate. ``promote_code`` gates a
deployed CODE mutation on the graph-invariant shadow-verification and RECORDS it
as a versioned candidate under ``evolved/code/`` — it never reaches live core
``src/`` (merge-to-live is deferred). The canary is intentionally NOT invoked
(it splices PROMPT suffixes into the live builder and cannot exercise shadow
code). These tests pin: parsing, slugifying, gating (pass / invariant-fail /
fail-open), the shadow-only write (no live src), history dedup, the default-off
setting, and the ``run_cycle`` wiring (flag on ⇒ recorded; off ⇒ byte-identical).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config.settings import EvolutionSettings
from src.evolution.engine import SelfEvolutionEngine
from src.evolution.promote import (
    PromotionGate,
    _slugify_target,
    parse_code_payload,
)
from src.graph.enums import MutationType


# ---------------------------------------------------------------------------
# settings fake — PromotionGate reads get_settings() for evolved_handlers_dir.
# ---------------------------------------------------------------------------


def _fake_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> SimpleNamespace:
    fake = SimpleNamespace(
        evolution=SimpleNamespace(
            evolution_promote_to_live=True,
            evolved_handlers_dir=str(tmp_path / "evolved"),
            promotion_canary_timeout_s=180.0,
        ),
        eval=SimpleNamespace(eval_canary_min_score=0.8),
    )
    monkeypatch.setattr("src.config.settings.get_settings", lambda: fake)
    monkeypatch.setattr("src.config.get_settings", lambda: fake)
    return fake


def _code_proposal(
    target_path: str = "graph/nodes/execute.py",
    content: str = "def add(a: int, b: int) -> int:\n    return a + b\n",
) -> dict[str, Any]:
    """A CODE mutation proposal (whole-module rewrite + shadow-repo target)."""
    return {
        "mutation_type": MutationType.CODE,
        "description": "speed up the add helper",
        "rationale": "inline the addition",
        "model_used": "test-model",
        "target_path": target_path,
        "mutated_content": content,
    }


def _tracker(repo_dir: Path) -> SimpleNamespace:
    """A minimal git_tracker stand-in exposing only ``repo_dir`` (read-only)."""
    repo_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(repo_dir=repo_dir)


# ---------------------------------------------------------------------------
# parse_code_payload
# ---------------------------------------------------------------------------


class TestParseCodePayload:
    def test_parses_valid_code_proposal(self) -> None:
        assert parse_code_payload(_code_proposal()) == (
            "graph/nodes/execute.py",
            "def add(a: int, b: int) -> int:\n    return a + b\n",
        )

    def test_defaults_target_path_when_missing(self) -> None:
        """A pathless CODE candidate falls back to deploy()'s placeholder so it is
        recorded rather than silently dropped (mirrors the deploy() write path)."""
        proposal = {"mutation_type": MutationType.CODE, "mutated_content": "x = 1\n"}
        assert parse_code_payload(proposal) == ("evolution/latest_mutation.py", "x = 1\n")

    def test_rejects_non_code_mutation(self) -> None:
        assert (
            parse_code_payload(
                {"mutation_type": MutationType.PROMPT, "mutated_content": "guide"}
            )
            is None
        )

    def test_rejects_empty_content(self) -> None:
        assert (
            parse_code_payload(
                {"mutation_type": MutationType.CODE, "mutated_content": "   "}
            )
            is None
        )


# ---------------------------------------------------------------------------
# _slugify_target
# ---------------------------------------------------------------------------


class TestSlugifyTarget:
    def test_flattens_path_separators_and_extensions(self) -> None:
        assert _slugify_target("graph/nodes/execute.py") == "graph-nodes-execute-py"

    def test_collapses_runs_and_strips_edges(self) -> None:
        assert _slugify_target("//deep///path//") == "deep-path"
        assert _slugify_target("  spaced name.py  ") == "spaced-name-py"

    def test_empty_or_punctuation_only_falls_back(self) -> None:
        assert _slugify_target("") == "code"
        assert _slugify_target("///...") == "code"


# ---------------------------------------------------------------------------
# promote_code — gating + shadow-only recording
# ---------------------------------------------------------------------------


class TestPromoteCode:
    @pytest.mark.asyncio
    async def test_records_candidate_on_passing_invariants(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _fake_settings(monkeypatch, tmp_path)
        gate = PromotionGate()
        tracker = _tracker(tmp_path / "repo")  # empty repo ⇒ invariants skip → pass

        result = await gate.promote_code(_code_proposal(), tracker)

        assert result["recorded"] is True
        assert result["promoted"] is False  # honesty: NOT a live promotion
        assert result["live"] is False
        assert result["scope"] == "shadow-candidate"
        assert result["target_path"] == "graph/nodes/execute.py"
        assert "deferred" in result
        # Versioned artifact + pointer both written under evolved/code/.
        code_dir = gate.prompts_dir.parent / "code"
        version_file = code_dir / result["version"]
        assert version_file.exists()
        payload = json.loads(version_file.read_text("utf-8"))
        assert payload["kind"] == "code"
        assert payload["live"] is False
        assert payload["mutated_content"].startswith("def add")
        pointer = json.loads((code_dir / "current.json").read_text("utf-8"))
        assert pointer["graph/nodes/execute.py"]["active"] == result["version"]

    @pytest.mark.asyncio
    async def test_rejects_non_code_mutation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _fake_settings(monkeypatch, tmp_path)
        gate = PromotionGate()

        result = await gate.promote_code(
            {"mutation_type": MutationType.PROMPT, "mutated_content": "guide"}, None
        )

        assert result["recorded"] is False
        assert "not a promotable CODE mutation" in result["reason"]
        # Nothing written.
        assert not (gate.prompts_dir.parent / "code" / "current.json").exists()

    @pytest.mark.asyncio
    async def test_invariant_failure_rejects_candidate(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """A shadow repo with a syntax-broken module fails the compiles invariant
        → the candidate is rejected and NOTHING is recorded."""
        _fake_settings(monkeypatch, tmp_path)
        gate = PromotionGate()
        tracker = _tracker(tmp_path / "broken-repo")
        # A .py under src/ with a syntax error fails _check_compiles.
        (tmp_path / "broken-repo" / "src").mkdir(parents=True)
        (tmp_path / "broken-repo" / "src" / "broken.py").write_text(
            "def (\n", encoding="utf-8"
        )

        result = await gate.promote_code(_code_proposal(), tracker)

        assert result["recorded"] is False
        assert "invariant gate failed" in result["reason"]
        assert not (gate.prompts_dir.parent / "code" / "current.json").exists()

    @pytest.mark.asyncio
    async def test_fail_open_without_shadow_repo(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """No git_tracker ⇒ invariant re-check is skipped (fail-open) so promote_code
        is independently callable; the candidate is recorded."""
        _fake_settings(monkeypatch, tmp_path)
        gate = PromotionGate()

        result = await gate.promote_code(_code_proposal(), None)

        assert result["recorded"] is True
        assert (gate.prompts_dir.parent / "code" / "current.json").exists()

    @pytest.mark.asyncio
    async def test_never_writes_outside_evolved_tree(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """The candidate is confined to evolved/code/ — it must NOT touch live src
        (no file appears at the project root or under a src/ mirror)."""
        _fake_settings(monkeypatch, tmp_path)
        gate = PromotionGate()
        evolved_root = tmp_path / "evolved"
        before = {p for p in evolved_root.rglob("*")} if evolved_root.exists() else set()

        await gate.promote_code(_code_proposal(), None)

        after = {p for p in evolved_root.rglob("*")}
        new = after - before
        # Every newly-created path lives under the code/ sub-tree.
        assert new, "expected the candidate artifact + pointer to be written"
        assert all(Path(p).relative_to(evolved_root).parts[0] == "code" for p in new)

    @pytest.mark.asyncio
    async def test_identical_re_record_dedups_history(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _fake_settings(monkeypatch, tmp_path)
        gate = PromotionGate()
        proposal = _code_proposal()

        first = await gate.promote_code(proposal, None)
        second = await gate.promote_code(proposal, None)

        assert first["version"] == second["version"]  # same content ⇒ same sha
        pointer = json.loads(
            (gate.prompts_dir.parent / "code" / "current.json").read_text("utf-8")
        )
        assert len(pointer["graph/nodes/execute.py"]["history"]) == 1

    @pytest.mark.asyncio
    async def test_distinct_targets_keyed_separately(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        _fake_settings(monkeypatch, tmp_path)
        gate = PromotionGate()

        await gate.promote_code(_code_proposal("graph/nodes/plan.py", "a = 1\n"), None)
        await gate.promote_code(_code_proposal("graph/nodes/execute.py", "b = 2\n"), None)

        pointer = json.loads(
            (gate.prompts_dir.parent / "code" / "current.json").read_text("utf-8")
        )
        assert set(pointer) == {"graph/nodes/plan.py", "graph/nodes/execute.py"}


# ---------------------------------------------------------------------------
# default-off setting
# ---------------------------------------------------------------------------


class TestPromoteCodeSettings:
    def test_flag_defaults_off(self) -> None:
        """The code default is False — asserted via the Pydantic field default, so
        it is robust to whatever the live ``.env`` happens to set."""
        assert (
            EvolutionSettings.model_fields["evolution_promote_code_to_core"].default
            is False
        )


# ---------------------------------------------------------------------------
# run_cycle wiring (flag on ⇒ recorded; off ⇒ byte-identical)
# ---------------------------------------------------------------------------


def _settings(
    monkeypatch: pytest.MonkeyPatch, *, promote_code_on: bool
) -> None:
    """Point the engine's lazy ``from src.config import get_settings`` at a fake."""
    fake = SimpleNamespace(
        evolution=SimpleNamespace(
            evolution_promote_to_live=False,  # keep the PROMPT path off; CODE under test
            evolution_promote_code_to_core=promote_code_on,
            evolution_require_curve_clear=False,
        ),
    )
    monkeypatch.setattr("src.config.get_settings", lambda: fake)


def _code_engine_at_promotion_phase() -> tuple[SelfEvolutionEngine, MagicMock]:
    """An engine whose middle phases are mocked so run_cycle reaches the CODE
    promotion-candidate block. ``generate`` returns a CODE proposal; with no
    gateway/git_tracker/sandbox the invariant verify + post-deploy smoke are
    skipped → ``effective_deploy`` is True and the CODE wiring is reached."""
    engine = SelfEvolutionEngine()
    engine.validate = AsyncMock(return_value={"passed": True})
    engine.sandbox_test = AsyncMock(return_value={"passed": True})
    engine.ab_test = AsyncMock(return_value={"is_significant": True})
    engine.deploy = AsyncMock(
        return_value={
            "deployed": True,
            "pre_deploy_hash": None,
            "commit_hash": None,
            "mutation_type": MutationType.CODE,
            "description": "code fix",
            "target_path": "graph/nodes/execute.py",
            "rationale": "",
            "ab_result": None,
        }
    )
    engine.generate = AsyncMock(return_value=_code_proposal())
    gate = MagicMock()
    gate.promote = AsyncMock(return_value={"promoted": False, "reason": "not PROMPT"})
    gate.promote_code = AsyncMock(
        return_value={"recorded": True, "version": "v.py.deadbeef.json", "sha": "deadbeef"}
    )
    return engine, gate


class TestEngineCodePromotionWiring:
    async def test_flag_on_records_candidate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _settings(monkeypatch, promote_code_on=True)
        engine, gate = _code_engine_at_promotion_phase()

        result = await engine.run_cycle(
            [], failure_patterns=["slow add"], promotion_gate=gate
        )

        assert gate.promote_code.await_count == 1
        assert result["code_promotion"]["recorded"] is True
        assert result["code_promotion"]["sha"] == "deadbeef"

    async def test_flag_off_is_byte_identical(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _settings(monkeypatch, promote_code_on=False)
        engine, gate = _code_engine_at_promotion_phase()

        result = await engine.run_cycle(
            [], failure_patterns=["slow add"], promotion_gate=gate
        )

        assert gate.promote_code.await_count == 0
        assert result["code_promotion"] == {}  # untouched → byte-identical
