---
description: API Cost Optimization Reference
---

## Model Cost Comparison (approximate, per 1M tokens)

| Model | Input Cost | Output Cost | Best For |
|---|---|---|---|
| Claude Opus 4.8 | $15.00 | $75.00 | Complex reasoning, highest quality |
| Claude Sonnet 4.6 | $3.00 | $15.00 | General purpose, balanced |
| Claude Haiku 4.5 | $0.80 | $4.00 | Simple tasks, high volume |
| GPT-4o | $2.50 | $10.00 | General purpose |
| GPT-4o-mini | $0.15 | $0.60 | Simple classification, extraction |
| text-embedding-3-small | $0.02 | — | Embeddings |

## Token Counting Strategies

- **tiktoken** (OpenAI): Exact count per model family.
- **Anthropic tokenizer**: Use official tokenizer or ~3.5 chars/token estimate.
- **Rule of thumb**: Count before every API call; log counts for analysis.

## Caching Strategies

| Strategy | Hit Rate | Complexity | When to Use |
|---|---|---|---|
| Exact match cache | High for repeated queries | Low | FAQ bots, repeated lookups |
| Semantic cache | Medium (fuzzy matches) | Medium | Similar but not identical queries |
| Prompt cache / context caching | High for shared prefixes | Low | Anthropic/OpenAI native caching |
| Response cache with TTL | Varies | Low | Time-bounded queries (weather, prices) |

## Model Routing Patterns

- **Complexity classifier**: Small model classifies query as simple/medium/complex → routes accordingly.
- **Confidence escalation**: Try cheap model first, escalate to expensive if low confidence.
- **Feature-based routing**: Summarization → Haiku, code review → Sonnet, legal analysis → Opus.

## Prompt Compression Techniques

- Remove unnecessary whitespace and formatting.
- Use abbreviations in system prompts (e.g., "Ans in JSON" instead of "Please answer in JSON format").
- Omit few-shot examples for tasks the model already handles well.
- Summarize long context documents before injecting into prompt.

## Budget Management

- Per-request token caps (max_tokens).
- Daily/weekly/monthly budget limits with alerting.
- Per-feature cost tracking for cost allocation.
- Auto-downgrade when budget threshold reached (Opus → Sonnet → Haiku).

## Cost Monitoring

- Log token counts per request, per feature, per user.
- Dashboard: daily cost trend, cost by model, cost by feature.
- Alert on: daily cost spike (>2x average), single expensive request, budget threshold breach.
