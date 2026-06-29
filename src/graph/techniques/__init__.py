"""Experimental prompting techniques (Phase 2 #18, default-OFF scaffolds).

Five research reasoning frameworks surfaced as selectable prompting *bodies* in
the :class:`TechniqueSelector` registry, each behind its own opt-in flag (and a
master ``EXPERIMENTAL_TECHNIQUES_ENABLED``). When no flag is on the registry is
byte-identical to the curated base — these modules are imported lazily by
``technique_selector._experimental_techniques`` *only* when a flag is on, so a
host run with the flags off never pays the import or changes selection.

Each module ships two things:

* ``TECHNIQUE`` — the genuine prompting ``body`` spliced into node prompts when
  its flag is on (real reasoning guidance, not a stub);
* ``apply`` — the entry point for the *full* framework algorithm, which raises
  :class:`TechniqueDeferredError` today. The multi-turn controller / self-play /
  debate loop that constitutes the complete technique is deferred to a later
  phase; ``apply`` is a documented deferred-error, not an empty placeholder, so
  a caller that attempts to execute the full algorithm fails loudly instead of
  silently no-op'ing.

:class:`TechniqueDeferredError` lives in the leaf ``_errors`` module and the
submodules import it from there (``from ._errors import …``), never from this
``__init__``. That keeps the package graph one-directional
(``__init__`` → submodules), which avoids a real import cycle and lets static
analysis resolve each submodule's ``TECHNIQUE``.
"""

from __future__ import annotations

from src.graph.prompts.technique_selector import Technique
from ._errors import TechniqueDeferredError


# Relative import (``from .``) — the package importing its own submodules by
# absolute path confuses Pyright's self-resolution.
from . import (  # noqa: E402
    absolute_zero,
    adversarial_debate,
    godel_agent,
    self_debugging,
    web_dreamer,
)

# (ExperimentalTechniqueSettings field name, the Technique object). Append order
# = registry order. The field names MUST match ExperimentalTechniqueSettings.
ENABLED_BY_FLAG: list[tuple[str, Technique]] = [
    ("self_debugging_enabled", self_debugging.TECHNIQUE),
    ("godel_agent_enabled", godel_agent.TECHNIQUE),
    ("web_dreamer_enabled", web_dreamer.TECHNIQUE),
    ("absolute_zero_enabled", absolute_zero.TECHNIQUE),
    ("adversarial_debate_enabled", adversarial_debate.TECHNIQUE),
]

__all__ = [
    "ENABLED_BY_FLAG",
    "TechniqueDeferredError",
    "absolute_zero",
    "adversarial_debate",
    "godel_agent",
    "self_debugging",
    "web_dreamer",
]
