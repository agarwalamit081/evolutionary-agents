---
description: LLM Observability and Evals Reference
---

## Observability Stack

| Tool | Type | Key Features |
|---|---|---|
| LangSmith | Tracing + Eval | LangChain native, annotation, datasets |
| LangFuse | Tracing + Eval | Open-source, self-hostable, scoring |
| Weave | Tracing | Weights & Biases, experiment tracking |
| Arize Phoenix | Tracing | Open-source, local UI, OpenTelemetry |

**Key Spans to Trace**: retrieval, reranking, prompt construction, LLM call, post-processing, guardrails.

**Metrics to Track**: latency (p50/p95/p99), token usage, cost per query, error rate, user feedback.

## Evaluation Frameworks

| Framework | Speciality | Metrics |
|---|---|---|
| ragas | RAG-specific | Faithfulness, answer relevancy, context precision/recall |
| deepeval | General LLM | Hallucination, bias, toxicity, answer relevance, faithfulness |
| Custom LLM-as-judge | Any | Rubric-based scoring with calibrated model |

## Golden Dataset Design

- 50-200 diverse examples covering edge cases.
- Format per entry: `{"input": "...", "expected_output": "...", "context": "...", "metadata": {"difficulty": "hard", "category": "..."}}`
- Version alongside prompt changes in git.
- Split into development (tuning, 70%) and held-out (validation, 30%).

## LLM-as-a-Judge Pattern

1. Define scoring rubric (1-5 scale with descriptions per level).
2. Provide judge with input, output, and scoring criteria.
3. Run 3 independent judgments, take majority vote.
4. Calibrate: compare judge scores against 20+ human-labeled examples.
5. Report inter-annotator agreement (Cohen's kappa > 0.6 acceptable).

## CI/CD Integration

- Run evals on every PR that changes prompts or retrieval logic.
- Fail build if metrics drop below threshold (configurable per metric).
- Generate eval report as PR comment.
- Require eval approval for model upgrades.
