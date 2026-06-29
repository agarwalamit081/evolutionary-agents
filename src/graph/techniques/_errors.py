"""Shared exception for the experimental-technique package (leaf module).

Lives in its own module so the technique submodules import it as
``from ._errors import TechniqueDeferredError`` instead of ``from . import ...``
— that keeps the package ``__init__``→submodule graph one-directional (the
submodules never import back into ``__init__``), which both avoids a real
import cycle and lets static analysis resolve each submodule's ``TECHNIQUE``.
"""

from __future__ import annotations


class TechniqueDeferredError(Exception):
    """The full framework algorithm for an experimental technique is not wired yet.

    The opt-in flag injects ONLY the technique's prompting ``body`` into node
    prompts (via the :class:`TechniqueSelector` registry); the multi-turn
    controller / self-play / debate loop that constitutes the *full* technique is
    deferred to a later phase. Raised by each module's ``apply`` entry so a
    caller that attempts to execute the full algorithm gets a clear, documented
    failure instead of a silent no-op.
    """
