"""Graph search primitives (Phase 5 G3).

Reasoning-search execution primitives, distinct from the mutation engine
(which mutates prompts/code/tools) and the topology optimizer (G3b AFlow).
Each ships default-off / opt-in so a host run is byte-identical until the
corresponding gate is toggled.
"""

from __future__ import annotations

from src.graph.search.lats import lats_search_node

__all__ = ["lats_search_node"]
