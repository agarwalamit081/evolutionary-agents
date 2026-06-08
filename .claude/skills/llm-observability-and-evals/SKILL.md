---
name: llm-observability-and-evals
description: LLM observability, tracing, and evaluation — LangSmith/LangFuse integration, evaluation scripts with ragas and deepeval, golden dataset testing, hallucination and faithfulness metrics, and LLM-as-a-judge frameworks.
---

**When to Use**
- Debugging LLM pipelines or agent workflows.
- Evaluating prompt changes against a baseline.
- Tracing multi-step LLM calls (retrieval → reranking → generation).
- Measuring hallucination rates, faithfulness, or answer quality.
- Building evaluation pipelines or A/B testing prompts.
- Setting up CI/CD gates for LLM output quality.

**Core Principles**
1. **Trace Everything**: Log every LLM call (input, output, latency, tokens, model) for debugging and optimization.
2. **Evaluate with Gold Standards**: Maintain a golden dataset of input-expected output pairs. Run evals on every prompt change.
3. **Multi-Dimensional Metrics**: Evaluate across faithfulness, relevance, correctness, toxicity, and latency — not just one metric.
4. **LLM-as-a-Judge**: Use a strong model to evaluate outputs. Define clear rubrics, calibrate against human judgments.
5. **Regression Testing**: Treat prompt/model changes like code changes — run evals before deploying.

**References**
- Load `reference.md` for observability stack, evaluation frameworks, golden dataset design, and CI/CD integration.
- Load `examples.md` for tracing setups, eval scripts, and LLM-as-a-judge patterns.

**Scripts**
- `scripts/run_evals.py`: Run evaluation suite against a golden dataset, output markdown report.
- `scripts/generate_golden_dataset_template.py`: Create JSONL template with sample entries.
