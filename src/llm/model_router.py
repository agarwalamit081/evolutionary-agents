"""Complexity-based model routing with fallback chains."""

from __future__ import annotations

import json

from loguru import logger

from src.config.model_registry import FALLBACK_CHAINS, MODEL_REGISTRY, ModelTier
from src.config.settings import Settings, get_settings
from src.graph.enums import TaskComplexity


# ─────────────────────────────────────────────────────────────────────────────
# TEMPORARY — REVERT BY 2026-07-01 ────────────────────────────────────────────
# The Anthropic API key is under an account usage cap (blocked until 2026-07-01):
# every anthropic model call returns a 400/quota error and burns the fallback
# chain — the circuit breaker does NOT trip on 400/quota (only on transient
# 429/5xx/timeout), so the dead attempt repeats once per chain member. To
# eliminate the wasted calls ENTIRELY, anthropic is excluded from ALL routing
# until the cap resets, via two guards that both read this one set:
#   • ``_has_provider_key("anthropic")`` → False: drops anthropic from the
#     router's primary/chain/diverse selection AND from the gateway's chain
#     pre-filter (gateway.py ``_execute_with_fallback`` filters via this method).
#   • seeded into ``_exclude_providers``: covers the router's absolute-fallback
#     loop (route() :136-139) — the one path that consults ``excluded``, not keys.
# Anthropic ModelSpecs + FALLBACK_CHAINS entries are LEFT INTACT so reverting is
# trivial (delete this constant + the two guards + restore DEFAULT_COMPLEXITY_TIER)
# and the cross-provider fallback-invariant (FALLBACK_CHAINS still names anthropic)
# is preserved for the deferred test (#311). 2026-07-01 is the documented Anthropic
# cap-reset date, so this aligns 1:1 with when the funded key recovers.
# ─────────────────────────────────────────────────────────────────────────────
_TEMPORARY_DISABLED_PROVIDERS: frozenset[str] = frozenset({"anthropic"})


def _resolved_disabled_providers(settings: Settings) -> set[str]:
    """The effective disabled-provider set used by ALL routing guards.

    ``DISABLED_PROVIDERS`` env is AUTHORITATIVE when set to any value: a
    comma-list (e.g. ``"anthropic"`` / ``"anthropic,minimax"``); an EMPTY string
    means none disabled. When the env is UNSET (``None``), the curated
    ``_TEMPORARY_DISABLED_PROVIDERS`` baseline is used (anthropic under a quota
    cap until 2026-07-01). Authoritative-when-set (not merge) so the temporary
    Anthropic block can be cleared without a code change by setting
    ``DISABLED_PROVIDERS=`` once the cap resets — see the revert note above.
    """
    env_dp = settings.routing.routing_disabled_providers
    if env_dp is None:
        return set(_TEMPORARY_DISABLED_PROVIDERS)
    return {p.strip() for p in env_dp.split(",") if p.strip()}


# Mapping from TaskComplexity to model tier and fallback chain key — the
# DEFAULT used when no per-node override applies (see NODE_TIER_MAP) and by
# callers that pass no node identity.
#
# De-flat (findings-03 #1 / findings-05): SIMPLE and COMPLEX previously both
# resolved to deepseek-v4-flash (CHEAP), so every non-trivial task paid for —
# and got — a Cheap model. COMPLEX now maps to glm-4.7 (MODERATE) so a complex
# goal's no-node default is stronger than a simple one. Per-node overrides in
# NODE_TIER_MAP refine this further (e.g. execute stays CHEAP even on a complex
# goal — individual steps are simple tool-calling).
#
# SIMPLE primary is deepseek-v4-flash (Cheap) — swapped off
# claude-haiku-4-5-20251001 after that Anthropic key hit an account usage cap
# (blocked until 2026-07-01): the key stayed present so route() kept returning
# Haiku as primary, and every SIMPLE call failed before falling through the
# chain (89 wasted attempts in one run). deepseek-v4-flash is the registered
# CHEAP-tier peer and was already Haiku's first fallback, so this removes the
# dead attempt with no behavior change on a funded key. Haiku stays registered
# + as deepseek-v4-flash's first chain fallback.
COMPLEXITY_TIER_MAP: dict[TaskComplexity, tuple[ModelTier, str]] = {
    # TRIVIAL primary is the newest flash-tier model — qwen3.6-flash (the
    # successor to qwen3.5-flash), promoted so the rolling flash alias is the
    # trivial-tier workhorse. qwen3.5-flash stays registered and remains
    # qwen3.6-flash's first FALLBACK_CHAINS entry, so the swap is safe: the old
    # primary didn't disappear, it became the fallback. Both resolve to provider
    # "alibaba" (DashScope) via the registry, so the Alibaba api_base/key pin in
    # the gateway applies to whichever fires.
    TaskComplexity.TRIVIAL: (ModelTier.VERY_CHEAP, "qwen3.6-flash"),
    TaskComplexity.SIMPLE: (ModelTier.CHEAP, "deepseek-v4-flash"),
    TaskComplexity.COMPLEX: (ModelTier.MODERATE, "glm-4.7"),
    TaskComplexity.CRITICAL: (ModelTier.MODERATE, "glm-4.7"),
}

# Per-node routing overrides keyed by (TaskComplexity, node_name). A node-aware
# tier lets a complex goal use a stronger model for reasoning-heavy steps
# (plan/reflect/verify) while keeping execution CHEAP (cost discipline:
# individual execute steps are simple tool-calling). verify/reflect on
# complex/critical goals additionally prefer the reasoning model via
# route_reasoning() (chain-of-verification payoff) — handled in route(), not
# here. Missing (complexity, node) keys fall back to COMPLEXITY_TIER_MAP. All
# upgrades land in MODERATE (glm-4.7 / deepseek-v4-pro) — never a
# flagship/Opus/GPT-5 (guardrails).
NODE_TIER_MAP: dict[tuple[TaskComplexity, str], tuple[ModelTier, str]] = {
    # Planning a complex/critical goal benefits from a stronger model.
    (TaskComplexity.COMPLEX, "plan"): (ModelTier.MODERATE, "glm-4.7"),
    (TaskComplexity.CRITICAL, "plan"): (ModelTier.MODERATE, "glm-4.7"),
    # Execution steps stay CHEAP even on complex/critical goals — overrides the
    # de-flatted COMPLEX→MODERATE default so tool-calling steps don't overspend.
    (TaskComplexity.COMPLEX, "execute"): (ModelTier.CHEAP, "deepseek-v4-flash"),
    (TaskComplexity.CRITICAL, "execute"): (ModelTier.CHEAP, "deepseek-v4-flash"),
}


# Defensive default for an unmapped TaskComplexity (e.g. a future enum member
# added without a tier-map entry). Previously this tuple was duplicated as an
# inline ``.get()`` default at both routing call sites (route / route_diverse),
# so the two could silently drift. Centralized here as the single source.
# TEMPORARY (REVERT BY 2026-07-01): primary swapped off claude-haiku-4-5-20251001
# (Anthropic key quota-capped until 2026-07-01) to its registered CHEAP-tier peer
# deepseek-v4-flash — same swap rationale as SIMPLE (see comment above). Restore
# claude-haiku-4-5-20251001 when the cap resets.
# DEFAULT_COMPLEXITY_TIER: tuple[ModelTier, str] = (ModelTier.CHEAP, "claude-haiku-4-5-20251001")
DEFAULT_COMPLEXITY_TIER: tuple[ModelTier, str] = (ModelTier.CHEAP, "deepseek-v4-flash")


class ModelRouter:
    """Routes task complexity to appropriate model with fallback chain support."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # Seed the runtime excluded set from DISABLED_PROVIDERS (env) when set,
        # else the curated _TEMPORARY_DISABLED_PROVIDERS baseline. Runtime
        # mark_provider_unhealthy() additions land here too. Consulted by the
        # absolute-fallback loop in route() (the one path that uses `excluded`,
        # not _has_provider_key). See _resolved_disabled_providers.
        self._exclude_providers: set[str] = _resolved_disabled_providers(settings)

    def _effective_default_tier(self) -> tuple[ModelTier, str]:
        """Defensive default tier for an UNMAPPED TaskComplexity (the ``.get()``
        fallback in route()/route_diverse()).

        Operator can retune via ``ROUTING_DEFAULT_COMPLEXITY_TIER`` — a model_id
        present in MODEL_REGISTRY; its tier is re-derived from the ModelSpec so
        the absolute-fallback loop stays coherent. Empty/unknown → the curated
        ``DEFAULT_COMPLEXITY_TIER`` unchanged (routing never breaks on a bad env).
        """
        override = self._settings.routing.routing_default_complexity_tier.strip()
        if override:
            spec = MODEL_REGISTRY.get(override)
            if spec is not None:
                return spec.tier, override
            logger.warning(
                f"ROUTING_DEFAULT_COMPLEXITY_TIER='{override}' not in "
                f"MODEL_REGISTRY; using curated default"
            )
        return DEFAULT_COMPLEXITY_TIER

    def route(
        self,
        complexity: TaskComplexity,
        node: str | None = None,
        exclude_providers: set[str] | None = None,
    ) -> str:
        """Select the best model for a given complexity (optionally per-node).

        Args:
            complexity: The task complexity classification.
            node: Optional graph-node name (e.g. "plan", "execute", "verify").
                When set, NODE_TIER_MAP may override the complexity default, and
                verify/reflect on complex/critical goals prefer the reasoning
                model (``route_reasoning``).
            exclude_providers: Providers to skip (e.g., unhealthy ones).

        Returns:
            A model identifier string (litellm format).
        """
        # Verify/reflect on complex/critical goals → reasoning model. This also
        # resurrects route_reasoning() (previously zero callers): chain-of-
        # verification / self-reflection pay off most on hard goals. Falls back
        # to CRITICAL routing when the reasoning model's provider has no key
        # (and that fallback passes node=None, so it cannot recurse here).
        if (
            node in {"verify", "reflect"}
            and complexity in {TaskComplexity.COMPLEX, TaskComplexity.CRITICAL}
        ):
            return self.route_reasoning()

        # Per-node override, else the complexity default.
        if node is not None and (complexity, node) in NODE_TIER_MAP:
            tier, chain_key = NODE_TIER_MAP[(complexity, node)]
        else:
            tier, chain_key = COMPLEXITY_TIER_MAP.get(
                complexity, self._effective_default_tier()
            )
        # Layer operator env-knob overrides (F2) over the curated tier maps.
        # Reads get_settings().routing.* at call-time; empty/unparseable JSON
        # leaves (tier, chain_key) unchanged. See _apply_routing_overrides.
        tier, chain_key = self._apply_routing_overrides(complexity, node, tier, chain_key)
        excluded = (exclude_providers or set()) | self._exclude_providers

        # The complexity's primary model (the chain_key itself) is the intended
        # default for this tier. Try it FIRST. Previously ``_route_from_chain``
        # only walked ``FALLBACK_CHAINS[chain_key]``, which deliberately
        # excludes the primary — so e.g. COMPLEX→"deepseek-v4-flash" silently
        # resolved to its first fallback (claude-haiku-4-5-20251001) and the
        # named default was never selected even when its provider key was
        # present. (F15: this made a different Cheap model the de-facto default
        # for nearly every task, starving the battery of tool-calling
        # reliability.)
        primary_provider = self._extract_provider(chain_key)
        if primary_provider not in excluded and self._has_provider_key(primary_provider):
            return chain_key

        model = self._route_from_chain(chain_key, excluded)
        if model:
            return model

        # Absolute fallback: find any available model in the tier
        for model_id, spec in MODEL_REGISTRY.items():
            if spec.tier == tier and self._extract_provider(model_id) not in excluded:
                logger.warning(f"Using fallback model {model_id} for complexity {complexity}")
                return model_id

        # Last resort: first model in registry
        first = next(iter(MODEL_REGISTRY))
        logger.error(f"All models exhausted for complexity {complexity}, using {first}")
        return first

    def _apply_routing_overrides(
        self,
        complexity: TaskComplexity,
        node: str | None,
        tier: ModelTier,
        chain_key: str,
    ) -> tuple[ModelTier, str]:
        """Layer operator env-knob overrides over the curated tier maps (F2).

        Reads ``settings.routing.routing_node_tier_overrides_json`` /
        ``routing_complexity_tier_overrides_json`` at CALL time. Keys are
        ``"<COMPLEXITY>:<node>"`` (node tier, e.g. ``"COMPLEX:execute"``) or a
        bare ``"<COMPLEXITY>"`` (complexity tier); values are a litellm
        model_id. A node-tier override wins over a complexity-tier override.
        Empty / unparseable JSON, or a model_id absent from MODEL_REGISTRY, →
        the curated ``(tier, chain_key)`` unchanged (routing never breaks on a
        bad env). When an override applies, the tier is re-derived from the
        override model's ModelSpec so the absolute-fallback loop stays coherent.
        """
        override = self._routing_override_for(complexity, node)
        if not override:
            return tier, chain_key
        spec = MODEL_REGISTRY.get(override)
        if spec is None:
            logger.warning(
                f"Routing override '{override}' for {complexity.name}:{node} is "
                f"not in MODEL_REGISTRY; ignoring override"
            )
            return tier, chain_key
        logger.debug(
            f"Routing override applied for {complexity.name}:{node}: "
            f"{chain_key} -> {override} (tier {spec.tier.name})"
        )
        return spec.tier, override

    def _routing_override_for(
        self, complexity: TaskComplexity, node: str | None
    ) -> str | None:
        """Resolve the override model_id for (complexity, node), or None.

        Node-tier (``"COMPLEXITY:node"``) takes precedence over complexity-tier
        (bare ``"COMPLEXITY"``). Malformed JSON is logged at DEBUG and treated
        as no override.
        """
        cpx = complexity.name
        node_overrides = self._parse_json_overrides(
            self._settings.routing.routing_node_tier_overrides_json
        )
        if node is not None:
            hit = node_overrides.get(f"{cpx}:{node}")
            if hit:
                return hit
        cpx_overrides = self._parse_json_overrides(
            self._settings.routing.routing_complexity_tier_overrides_json
        )
        return cpx_overrides.get(cpx)

    @staticmethod
    def _parse_json_overrides(raw: str | None) -> dict[str, str]:
        """Parse a JSON ``{routing_key: model_id}`` override map, tolerantly."""
        if not raw or not raw.strip():
            return {}
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.debug(f"Routing override JSON unparseable, ignored: {exc}")
            return {}
        if not isinstance(parsed, dict):
            return {}
        return {str(k): str(v) for k, v in parsed.items() if v}

    def get_fallback_chain(self, model: str) -> list[str]:
        """Get the fallback chain for a specific model.

        Args:
            model: The primary model identifier.

        Returns:
            List of fallback model identifiers.
        """
        return FALLBACK_CHAINS.get(model, [])

    def get_fallback_tier0(self) -> str:
        """Get a cheap fallback model."""
        chain = FALLBACK_CHAINS.get("qwen3.5-flash", [])
        if chain:
            return chain[0]
        return "qwen3.5-flash"

    def route_reasoning(self) -> str:
        """Select the configured reasoning model.

        Uses the ``reasoning_llm_model`` from settings (e.g. deepseek-v4-pro).
        Falls back to ``route(CRITICAL)`` when the provider has no API key.

        Returns:
            A model identifier string (litellm format).
        """
        model = self._settings.llm.reasoning_llm_model
        provider = self._extract_provider(model)
        if self._has_provider_key(provider):
            return model
        logger.warning(
            f"Reasoning model {model} provider {provider} has no API key, "
            f"falling back to CRITICAL routing"
        )
        return self.route(TaskComplexity.CRITICAL)

    def route_diverse(
        self,
        n: int,
        complexity: TaskComplexity,
        node: str | None = None,
        exclude_providers: set[str] | None = None,
    ) -> list[str]:
        """Return *n* models from different providers for a given complexity.

        Used when spawning parallel sub-agents to spread load across
        providers and avoid rate limits.  Falls back to cycling through
        whatever providers are available.

        Args:
            n: Number of distinct models to return.
            complexity: Task complexity level for tier selection.
            node: Optional graph-node name for per-node tier override
                (mirrors ``route``). ``None`` for sub-agent fan-out, which is
                the common caller.
            exclude_providers: Providers to skip.

        Returns:
            List of *n* model identifiers, one per provider where possible.
        """
        excluded = (exclude_providers or set()) | self._exclude_providers
        # Per-node override, else the complexity default — mirrors route().
        if node is not None and (complexity, node) in NODE_TIER_MAP:
            tier, chain_key = NODE_TIER_MAP[(complexity, node)]
        else:
            tier, chain_key = COMPLEXITY_TIER_MAP.get(
                complexity, self._effective_default_tier()
            )

        # Collect one model per provider at the target tier
        provider_to_model: dict[str, str] = {}
        for model_id, spec in MODEL_REGISTRY.items():
            if spec.tier != tier:
                continue
            provider = self._extract_provider(model_id)
            if provider in excluded or provider in provider_to_model:
                continue
            if self._has_provider_key(provider):
                provider_to_model[provider] = model_id

        # Supplement from the fallback chain (may cross tiers)
        if len(provider_to_model) < n:
            for model_id in FALLBACK_CHAINS.get(chain_key, []):
                provider = self._extract_provider(model_id)
                if provider in excluded or provider in provider_to_model:
                    continue
                if self._has_provider_key(provider):
                    provider_to_model[provider] = model_id
                if len(provider_to_model) >= n:
                    break

        candidates = list(provider_to_model.values())

        if not candidates:
            # Absolute fallback: just repeat the default route (keyword so the
            # positional node slot is not accidentally filled by exclude_providers).
            return [self.route(complexity, exclude_providers=exclude_providers)] * max(1, n)

        # Cycle through candidates to fill n slots
        result: list[str] = []
        for i in range(n):
            result.append(candidates[i % len(candidates)])
        return result

    def mark_provider_unhealthy(self, provider: str) -> None:
        """Temporarily exclude a provider from routing."""
        self._exclude_providers.add(provider)
        logger.warning(f"Provider {provider} marked unhealthy, excluded from routing")

    def clear_provider_health(self, provider: str) -> None:
        """Re-enable a previously excluded provider."""
        self._exclude_providers.discard(provider)
        logger.info(f"Provider {provider} re-enabled for routing")

    def _route_from_chain(self, chain_key: str, exclude_providers: set[str]) -> str | None:
        """Try models in a fallback chain, skipping excluded providers."""
        chain = FALLBACK_CHAINS.get(chain_key, [])

        for model_id in chain:
            provider = self._extract_provider(model_id)
            if provider not in exclude_providers:
                # Verify we have API key for this provider
                if self._has_provider_key(provider):
                    return model_id
                logger.debug(f"Skipping {model_id}: no API key for {provider}")

        return None

    @staticmethod
    def _extract_provider(model: str) -> str:
        """Extract provider name from a model identifier (registry key or litellm id)."""
        # The registry is the source of truth for a registered model's provider —
        # consult it first so a key like "alibaba-deepseek-v4-flash" (provider
        # "alibaba", same model family served via DashScope) resolves correctly
        # even though no prefix heuristic below matches it. Without this, the
        # router logs "no API key for unknown" and silently skips the model in
        # every fallback chain it appears in — defeating provider-diverse routing.
        spec = MODEL_REGISTRY.get(model)
        if spec is not None:
            return spec.provider
        # litellm format: "provider/model-name" or "model-name"
        if "/" in model:
            return model.split("/")[0]
        # Registry key prefix for NVIDIA free-tier models
        if model.startswith("nvidia-"):
            return "nvidia"
        # Known model prefixes
        if model.startswith("gpt-") or model.startswith("text-embedding-"):
            return "openai"
        if model.startswith("claude-"):
            return "anthropic"
        if model.startswith("deepseek-"):
            return "deepseek"
        if model.startswith("gemini-"):
            return "google"
        if model.startswith("mistral-") or model.startswith("ministral-") or model.startswith("open-mistral-"):
            return "mistral"
        if model.startswith("qwen"):
            return "alibaba"
        if model.startswith("glm-"):
            return "zai"
        if model.startswith("kimi-") or model.startswith("moonshot-"):
            return "moonshot"
        if model.startswith("minimax-"):
            return "minimax"
        if model.startswith("llama-") or model.startswith("meta-llama/"):
            return "groq"
        return "unknown"

    @staticmethod
    def is_provider_disabled(provider: str) -> bool:
        """Whether a provider is temporarily excluded from all routing.

        Mirrors the gate ``_has_provider_key`` applies to router selection, so
        the gateway's budget-fallback search (``_get_cheaper_fallback``) can skip
        a disabled provider (e.g. anthropic under a quota cap) instead of
        selecting it as the cheaper model and burning the fallback chain on a 400.
        """
        return provider in _resolved_disabled_providers(get_settings())

    def _has_provider_key(self, provider: str) -> bool:
        """Check if an API key is available for a provider."""
        # Report a disabled provider (DISABLED_PROVIDERS env when set, else the
        # curated _TEMPORARY_DISABLED_PROVIDERS baseline) as key-less so it is
        # dropped from router primary/chain/diverse selection AND from the
        # gateway's fallback-chain pre-filter (gateway.py _execute_with_fallback
        # filters via this method). See _resolved_disabled_providers.
        if provider in _resolved_disabled_providers(self._settings):
            return False
        try:
            return self._settings.llm.has_provider_key(provider)
        except Exception:
            return False  # Skip provider if settings access fails
