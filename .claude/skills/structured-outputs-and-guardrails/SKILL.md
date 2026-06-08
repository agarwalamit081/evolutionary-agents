---
name: structured-outputs-and-guardrails
description: Reliable LLM structured output generation and semantic guardrails — JSON schema enforcement, function calling, retry logic, hallucination detection, and content safety.
---

**When to Use**
- Making LLMs return JSON or structured data reliably.
- Enforcing output schemas via Pydantic, Instructor, or Outlines.
- Implementing function/tool calling with LLMs.
- Preventing hallucinations, toxic output, or PII leakage.
- Building retry logic for malformed LLM outputs.

**Core Principles**
1. **Schema-First Design**: Define output schema before writing prompts. Use JSON Schema or Pydantic as the contract.
2. **Validate Everything**: Never trust raw LLM output. Always validate against schema before use.
3. **Graceful Retry**: On validation failure, retry with the error appended to the prompt (self-correction). Max 3 retries.
4. **Semantic Guardrails**: Beyond schema — check for hallucination, toxicity, and PII leakage.
5. **Use Native Features**: Prefer native structured outputs (Anthropic tool_use, OpenAI function_calling) over prompt-only approaches.

**References**
- Load `reference.md` for structured output approaches, Pydantic model design, retry strategies, and semantic validation.
- Load `examples.md` for extraction patterns, retry loops, and guardrail implementations.

**Scripts**
- `scripts/validate_llm_output.py`: Validate JSON output against a Pydantic schema with error reporting.
- `scripts/retry_with_feedback.py`: Generic retry wrapper that feeds validation errors back to the LLM.
