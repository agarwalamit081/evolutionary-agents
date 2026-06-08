---
description: Prompt Engineering Reference
---

## System Prompt Anatomy

1. **Role**: Define who the AI is ("You are a senior code reviewer...").
2. **Context**: Background information the model needs.
3. **Instructions**: What the model should do, step by step.
4. **Constraints**: What the model must not do, boundaries, limits.
5. **Output Format**: Exact format expected (JSON schema, markdown, etc.).
6. **Examples**: 2-5 input/output pairs demonstrating desired behavior.

## Prompting Techniques

| Technique | When to Use | How |
|---|---|---|
| Zero-shot | Simple tasks, strong models | Just instructions, no examples |
| Few-shot | Specific format needed | 2-5 input/output examples in prompt |
| Chain-of-thought | Complex reasoning | "Think step by step" or `<thinking>` block |
| ReAct | Agent/tool use | Reason → Act → Observe loop |
| Self-consistency | High-stakes decisions | Run N times, take majority vote |
| Self-critique | Quality improvement | Generate → Critique → Refine |

## XML Tagging for Structured Prompts (Claude-specific)

```
<context>
  [Background information here]
</context>

<instructions>
  1. Analyze the input for [X].
  2. Extract [Y].
  3. Return in the specified format.
</instructions>

<output_format>
{
  "analysis": "...",
  "entities": [...],
  "confidence": 0.0-1.0
}
</output_format>
```

## Prompt Versioning Strategy

- Store prompts as files in git (e.g., `prompts/v2_summary.txt`).
- Include metadata header: model name, date, performance metrics, change description.
- Tag releases with model version (e.g., `prompt-v2.1-claude-sonnet`).
- A/B test: run both versions on golden dataset, compare metrics before switching.

## Common Pitfalls

- **Ambiguous instructions**: "Be helpful" → "Respond with exactly 3 bullet points, each under 50 words."
- **Conflicting constraints**: Don't say "be concise but comprehensive" — pick one or define boundaries.
- **Missing edge cases**: Always specify what to do with unexpected input.
- **Over-constraining**: Too many rules can confuse the model. Keep to 5-7 key instructions.
