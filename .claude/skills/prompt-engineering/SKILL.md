---
name: prompt-engineering
description: System prompt design, advanced prompting techniques (few-shot, chain-of-thought, ReAct, XML-tagging), structured output prompting, and prompt versioning.
---

**When to Use**
- Designing or optimizing system prompts.
- Implementing chain-of-thought, few-shot, or ReAct prompting.
- Creating prompt templates with variable interpolation.
- Versioning prompts for A/B testing or regression tracking.

**Core Principles**
1. **System Prompt First**: Invest in the system prompt — it's the highest-leverage optimization.
2. **Be Explicit**: State what you want, not what you don't. Provide examples of desired output.
3. **Structure with XML Tags**: Use `<context>`, `<instructions>`, `<output_format>` for clarity (especially with Claude).
4. **Chain-of-Thought**: Ask the model to reason step-by-step before answering. Improves accuracy on complex tasks.
5. **Few-Shot Examples**: Provide 2-5 input/output examples. More reliable than instructions alone.
6. **Version Everything**: Treat prompts like code. Version in git, tag with model name and date.

**References**
- Load `reference.md` for prompt anatomy, techniques, versioning strategy, and common pitfalls.
- Load `examples.md` for prompt templates, CoT patterns, and versioning examples.

**Scripts**
- `scripts/render_prompt.py`: Render a prompt template with variable substitution.
