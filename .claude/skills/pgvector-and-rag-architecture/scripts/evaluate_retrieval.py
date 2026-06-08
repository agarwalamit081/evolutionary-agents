"""Evaluate retrieval quality using a golden test set.
Usage: python evaluate_retrieval.py <test_set.jsonl>

Expected JSONL format per line:
{"query": "...", "relevant_ids": ["doc1", "doc2"], "retrieved_ids": ["doc3", "doc1", ...]}
"""
import json
import sys
from statistics import mean


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for i, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def evaluate(test_file: str, k_values: list[int] | None = None):
    if k_values is None:
        k_values = [1, 3, 5, 10]

    with open(test_file) as f:
        test_set = [json.loads(line) for line in f]

    metrics = {f"P@{k}": [] for k in k_values}
    metrics.update({f"R@{k}": [] for k in k_values})
    metrics["MRR"] = []

    for item in test_set:
        retrieved = item["retrieved_ids"]
        relevant = set(item["relevant_ids"])

        for k in k_values:
            metrics[f"P@{k}"].append(precision_at_k(retrieved, relevant, k))
            metrics[f"R@{k}"].append(recall_at_k(retrieved, relevant, k))
        metrics["MRR"].append(mrr(retrieved, relevant))

    print(f"--- Retrieval Evaluation ({len(test_set)} queries) ---")
    for metric, values in metrics.items():
        avg = mean(values)
        print(f"  {metric}: {avg:.4f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        evaluate(sys.argv[1])
    else:
        print("Usage: python evaluate_retrieval.py <test_set.jsonl>")
