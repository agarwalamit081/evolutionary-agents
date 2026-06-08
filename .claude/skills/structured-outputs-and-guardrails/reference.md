---
description: Structured Outputs and Guardrails Reference
---

## Structured Output Approaches

| Approach | Reliability | Speed | Use When |
|---|---|---|---|
| Native API (tool_use/functions) | Highest | Fast | Provider supports it |
| Instructor library | High | Medium | Python, Pydantic models |
| Outlines / Guidance | Very High | Slow | Constrained generation, regex grammars |
| Prompt-only JSON | Low | Fastest | Fallback, universal compatibility |

## Pydantic Model Design for LLM Outputs

- Use `Field(description=...)` for field-level guidance to the LLM.
- Use `Literal` types for enum constraints.
- Use `Optional` for fields the LLM might not fill.
- Include a `reasoning: str` field for chain-of-thought before the structured answer.

## Retry Strategies

- Max 3 retries with exponential backoff (1s, 2s, 4s).
- Include the validation error in the retry prompt so the LLM can self-correct.
- Fall back to a simpler schema if repeated failures.
- Log all failed attempts for prompt improvement.

## Semantic Validation

- **Hallucination check**: Extract claims from output, verify against source documents.
- **Toxicity**: Use classifier (Perspective API, local moderation model).
- **PII detection**: Regex + NER scan on LLM output before returning to user.
- **Factual grounding**: Cross-reference assertions with knowledge base.

## Error Taxonomy for Guardrails

| Error Type | Detection | Response |
|---|---|---|
| Schema violation | Pydantic/JSON Schema | Retry with error feedback |
| Hallucination | Claim extraction vs sources | Reject or flag for review |
| Toxicity | Classifier score > threshold | Reject with safe fallback |
| PII leak | NER/regex scan | Redact and warn |
| Empty/null fields | Schema validation | Retry with emphasis on completeness |
