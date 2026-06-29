"""Tests for the experimental reasoning-technique scaffolds (#18).

Covers:
* the five technique modules ship genuine bodies + deferred-error ``apply``;
* ``ExperimentalTechniqueSettings`` defaults every flag off;
* with the flags off, the TechniqueSelector registry is byte-identical to the
  curated base (selection unchanged on a host run);
* master-off ignores per-technique flags;
* master-on + a per-technique flag surfaces that technique;
* a settings-resolution failure is fail-safe (base registry, never breaks).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.config.settings import ExperimentalTechniqueSettings
from src.graph.enums import TaskComplexity
from src.graph.prompts.technique_selector import (
    NODE_EXECUTE,
    TECHNIQUE_REGISTRY,
    TechniqueSelector,
)
from src.graph.techniques import (
    TechniqueDeferredError,
    absolute_zero,
    adversarial_debate,
    godel_agent,
    self_debugging,
    web_dreamer,
)

# (module, settings flag attr, name, a (complexity, node) it qualifies for).
EXPERIMENTAL: list[tuple[Any, str, str, TaskComplexity, str]] = [
    (self_debugging, "self_debugging_enabled", "self_debugging", TaskComplexity.COMPLEX, NODE_EXECUTE),
    (godel_agent, "godel_agent_enabled", "godel_agent", TaskComplexity.COMPLEX, NODE_EXECUTE),
    (web_dreamer, "web_dreamer_enabled", "web_dreamer", TaskComplexity.COMPLEX, "plan"),
    (absolute_zero, "absolute_zero_enabled", "absolute_zero", TaskComplexity.CRITICAL, NODE_EXECUTE),
    (adversarial_debate, "adversarial_debate_enabled", "adversarial_debate", TaskComplexity.CRITICAL, "verify"),
]


def _fake_settings(enabled: bool, **flags: bool) -> SimpleNamespace:
    """Build a stand-in for ``Settings.experimental_techniques``."""
    attrs = {
        "self_debugging_enabled": False,
        "godel_agent_enabled": False,
        "web_dreamer_enabled": False,
        "absolute_zero_enabled": False,
        "adversarial_debate_enabled": False,
    }
    attrs.update(flags)
    return SimpleNamespace(experimental_techniques=SimpleNamespace(enabled=enabled, **attrs))


def _patch(monkeypatch: pytest.MonkeyPatch, enabled: bool = False, **flags: bool) -> None:
    """Patch the lazily-read ``get_settings`` the selector resolves at construct time."""
    monkeypatch.setattr("src.config.settings.get_settings", lambda: _fake_settings(enabled, **flags))


# ── module + settings surface ────────────────────────────────────────────────


@pytest.mark.parametrize("module,flag,name,complexity,node", EXPERIMENTAL)
def test_techniques_carry_real_bodies_and_metadata(
    module: Any, flag: str, name: str, complexity: TaskComplexity, node: str
) -> None:
    """Each experimental technique has a non-empty body and qualifying metadata."""
    tech = module.TECHNIQUE
    assert tech.name == name
    # Real guidance, not a stub: at least a couple of sentences.
    assert len(tech.body.split()) >= 20
    assert complexity in tech.applies_to_complexities
    assert node in tech.nodes
    # The flag attr referenced by ENABLED_BY_FLAG must exist on the settings group.
    assert hasattr(ExperimentalTechniqueSettings(), flag)


@pytest.mark.parametrize("module,flag,name,complexity,node", EXPERIMENTAL)
async def test_apply_raises_documented_deferred_error(
    module: Any, flag: str, name: str, complexity: TaskComplexity, node: str
) -> None:
    """The full-algorithm entry is a documented deferred-error, never a silent no-op."""
    with pytest.raises(TechniqueDeferredError) as exc_info:
        await module.apply()
    message = str(exc_info.value)
    # Documents that the body-only injection works today + the full loop is deferred,
    # and names THIS technique's env var (flag.upper() e.g. SELF_DEBUGGING_ENABLED).
    assert "deferred" in message.lower()
    assert "TechniqueSelector" in message
    assert flag.upper() in message


def test_settings_default_all_off() -> None:
    """Every flag (master + per-technique) defaults to False."""
    settings = ExperimentalTechniqueSettings()
    assert settings.enabled is False
    for _module, flag, _name, _c, _n in EXPERIMENTAL:
        assert getattr(settings, flag) is False


# ── registry gating ──────────────────────────────────────────────────────────


def test_off_is_byte_identical_to_base_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flags off → the default registry equals the curated base exactly."""
    _patch(monkeypatch, enabled=False)
    selector = TechniqueSelector()
    assert selector._registry == TECHNIQUE_REGISTRY
    # And no experimental name is ever selectable.
    for _module, _flag, name, _c, _n in EXPERIMENTAL:
        assert name not in {t.name for t in selector._registry}


def test_master_off_ignores_per_technique_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Master off but a per-technique flag on → still the base registry only."""
    _patch(monkeypatch, enabled=False, self_debugging_enabled=True)
    selector = TechniqueSelector()
    assert selector._registry == TECHNIQUE_REGISTRY
    names = {t.name for t in selector._registry}
    assert "self_debugging" not in names


def test_enabled_master_plus_flag_surfaces_technique(monkeypatch: pytest.MonkeyPatch) -> None:
    """Master on + one flag on → only that technique is appended to the base registry."""
    _patch(monkeypatch, enabled=True, self_debugging_enabled=True)
    selector = TechniqueSelector()
    names = {t.name for t in selector._registry}
    assert "self_debugging" in names
    # The other four stay off.
    for _module, flag, name, _c, _n in EXPERIMENTAL:
        if flag == "self_debugging_enabled":
            continue
        assert name not in names

    # And it actually surfaces through select() for its qualifying call.
    selected = selector.select(TaskComplexity.COMPLEX, NODE_EXECUTE, goal_pattern="code")
    assert "self_debugging" in {t.name for t in selected}


def test_settings_resolution_failure_is_fail_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    """A get_settings() error must never break selection — fall back to the base registry."""
    def _boom() -> Any:
        raise RuntimeError("settings unavailable")

    monkeypatch.setattr("src.config.settings.get_settings", _boom)
    selector = TechniqueSelector()
    assert selector._registry == TECHNIQUE_REGISTRY
