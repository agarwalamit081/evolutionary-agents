---
name: api-cost-optimization
description: LLM API cost optimization — token usage minimization, smart caching, model routing, rate limit management, and cost-effective architecture decisions.
---

**When to Use**
- Optimizing LLM API costs and token usage.
- Choosing between models based on task complexity.
- Implementing caching to reduce duplicate API calls.
- Designing cost-aware LLM architectures.
- Setting up budget management and cost alerts.

**Core Principles**
1. **Measure First**: Track token usage and cost per feature/query before optimizing.
2. **Route by Complexity**: Cheap models (Haiku, GPT-4o-mini) for simple tasks; expensive models only when needed.
3. **Cache Aggressively**: Cache LLM responses for identical or semantically similar inputs.
4. **Minimize Tokens**: Compress prompts, use abbreviations, remove redundant context.
5. **Batch When Possible**: Combine small requests into single batch calls where supported.

**References**
- Load `reference.md` for model cost comparison, caching strategies, model routing, and budget management.
- Load `examples.md` for token counters, model routers, caching, and budget trackers.

**Scripts**
- `scripts/estimate_cost.py`: Estimate API cost for a prompt and expected output length.
