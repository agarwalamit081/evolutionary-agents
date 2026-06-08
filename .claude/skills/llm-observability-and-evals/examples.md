---
description: LLM Observability and Evals Examples
---

**Example 1: LangSmith Trace Setup for RAG Pipeline**

```python
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

@traceable(name="retrieval", run_type="retriever")
def retrieve_documents(query: str, k: int = 5) -> list[dict]:
    # Your retrieval logic here
    return vector_store.search(query, k=k)

@traceable(name="generation", run_type="llm")
def generate_answer(query: str, context: list[dict]) -> str:
    # Your generation logic here
    return llm.generate(prompt=query, context=context)

@traceable(name="rag_pipeline")
def rag_pipeline(query: str) -> str:
    docs = retrieve_documents(query)
    answer = generate_answer(query, docs)
    return answer
```

---

**Example 2: ragas Evaluation Script**

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# Your test data
data = {
    "question": ["What is refund policy?", "How to cancel?"],
    "answer": ["Refunds within 30 days...", "Navigate to settings..."],
    "contexts": [["Policy doc page 1"], ["Settings guide page 3"]],
    "ground_truth": ["30-day refund policy", "Settings > Subscription > Cancel"],
}

dataset = Dataset.from_dict(data)

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=ChatOpenAI(model="gpt-4o"),
    embeddings=OpenAIEmbeddings(),
)

print(results)
# {'faithfulness': 0.85, 'answer_relevancy': 0.92, ...}
```

---

**Example 3: deepeval Faithfulness Test**

```python
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric
from deepeval import assert_test

def test_rag_faithfulness():
    test_case = LLMTestCase(
        input="What is the return policy?",
        actual_output="Returns are accepted within 30 days of purchase with receipt.",
        retrieval_context=["Our return policy allows returns within 30 days..."],
    )
    metric = FaithfulnessMetric(threshold=0.7, model="gpt-4o")
    assert_test(test_case, [metric])
```

---

**Example 4: LLM-as-a-Judge Rubric and Scorer**

```python
import json
from anthropic import Anthropic

EVAL_RUBRIC = """
Score the answer 1-5:
5 = Perfect: Complete, accurate, well-structured, cites sources.
4 = Good: Mostly complete and accurate, minor issues.
3 = Adequate: Addresses the question but has gaps or minor errors.
2 = Poor: Significant errors or missing key information.
1 = Fail: Incorrect, irrelevant, or harmful.
"""

JUDGE_PROMPT = """{rubric}

Question: {question}
Context: {context}
Answer to evaluate: {answer}

Respond with JSON: {{"score": <1-5>, "reasoning": "<explanation>", "issues": [<list of specific issues>]}}"""

async def judge_answer(question: str, context: str, answer: str) -> dict:
    client = Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            rubric=EVAL_RUBRIC, question=question, context=context, answer=answer
        )}],
    )
    return json.loads(response.content[0].text)
```

---

**Example 5: Golden Dataset Format (JSONL)**

```jsonl
{"input": "What is the refund policy?", "expected_output": "30-day refund with receipt", "context": "Refund policy doc v2.1", "metadata": {"difficulty": "easy", "category": "policy"}}
{"input": "Can I return opened electronics?", "expected_output": "Opened electronics can be returned within 15 days with 15% restocking fee", "context": "Electronics return policy", "metadata": {"difficulty": "medium", "category": "policy"}}
{"input": "What happens if I lost my receipt?", "expected_output": "Store credit at current selling price with valid ID", "context": "No-receipt return policy", "metadata": {"difficulty": "hard", "category": "edge_case"}}
```

---

**Example 6: CI/CD Eval Workflow (GitHub Actions)**

```yaml
name: LLM Eval Gate
on:
  pull_request:
    paths:
      - 'prompts/**'
      - 'src/retrieval/**'

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install dependencies
        run: pip install -r requirements-eval.txt
      - name: Run evaluations
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: python scripts/run_evals.py --golden evals/golden_dataset.jsonl --threshold 0.75
      - name: Post results
        if: always()
        run: python scripts/post_eval_comment.py --report eval_report.md
```

---

**Example 7: Token Usage Monitoring Query**

```python
import sqlite3
from datetime import datetime, timedelta

def get_token_usage_last_7_days(db_path: str = "llm_logs.db"):
    conn = sqlite3.connect(db_path)
    week_ago = (datetime.now() - timedelta(days=7)).isoformat()

    rows = conn.execute("""
        SELECT
            DATE(timestamp) as date,
            model,
            SUM(input_tokens) as total_input,
            SUM(output_tokens) as total_output,
            COUNT(*) as request_count,
            AVG(latency_ms) as avg_latency
        FROM llm_calls
        WHERE timestamp > ?
        GROUP BY DATE(timestamp), model
        ORDER BY date DESC
    """, (week_ago,)).fetchall()

    for row in rows:
        date, model, inp, out, count, latency = row
        print(f"{date} | {model} | in:{inp:,} out:{out:,} | {count} calls | {latency:.0f}ms avg")
```

---

**Example 8: A/B Test Comparison for Prompt Variants**

```python
from statistics import mean
import json

def compare_variants(variant_a_file: str, variant_b_file: str):
    with open(variant_a_file) as f:
        a_results = [json.loads(line) for line in f]
    with open(variant_b_file) as f:
        b_results = [json.loads(line) for line in f]

    metrics = ["faithfulness", "relevance", "latency_ms"]

    print(f"{'Metric':<20} {'Variant A':>12} {'Variant B':>12} {'Delta':>10}")
    print("-" * 56)

    for metric in metrics:
        a_vals = [r[metric] for r in a_results if metric in r]
        b_vals = [r[metric] for r in b_results if metric in r]
        if a_vals and b_vals:
            a_avg, b_avg = mean(a_vals), mean(b_vals)
            delta = ((b_avg - a_avg) / a_avg) * 100
            print(f"{metric:<20} {a_avg:>12.3f} {b_avg:>12.3f} {delta:>+9.1f}%")
