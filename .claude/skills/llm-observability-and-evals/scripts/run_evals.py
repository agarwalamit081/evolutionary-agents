"""Run evaluation suite against a golden dataset, output markdown report.
Usage: python run_evals.py --golden evals/golden_dataset.jsonl [--threshold 0.75] [--output eval_report.md]
"""
import argparse
import json
from datetime import datetime
from statistics import mean


def compute_metrics(results: list[dict]) -> dict[str, float]:
    """Compute aggregate metrics from individual results."""
    metric_keys = [k for k in results[0] if k not in ("input", "expected", "actual")]
    aggregated = {}
    for key in metric_keys:
        values = [r[key] for r in results if key in r and isinstance(r[key], (int, float))]
        if values:
            aggregated[key] = mean(values)
    return aggregated


def generate_report(results: list[dict], metrics: dict[str, float], threshold: float) -> str:
    lines = [
        f"# Evaluation Report",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Total samples**: {len(results)}",
        f"**Threshold**: {threshold}",
        "",
        "## Aggregate Metrics",
        "",
        f"| Metric | Score | Status |",
        f"|---|---|---|",
    ]

    passed = True
    for metric, value in metrics.items():
        status = "PASS" if value >= threshold else "FAIL"
        if status == "FAIL":
            passed = False
        lines.append(f"| {metric} | {value:.4f} | {status} |")

    lines.append("")
    lines.append(f"**Overall**: {'PASS' if passed else 'FAIL'}")
    lines.append("")
    lines.append("## Per-Sample Details")
    lines.append("")
    for i, r in enumerate(results, 1):
        lines.append(f"### Sample {i}")
        lines.append(f"- **Input**: {r.get('input', 'N/A')[:100]}...")
        lines.append(f"- **Expected**: {r.get('expected', 'N/A')[:100]}...")
        lines.append(f"- **Actual**: {r.get('actual', 'N/A')[:100]}...")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run LLM evaluations")
    parser.add_argument("--golden", required=True, help="Path to golden dataset JSONL")
    parser.add_argument("--threshold", type=float, default=0.75, help="Minimum score to pass")
    parser.add_argument("--output", default="eval_report.md", help="Output report file")
    args = parser.parse_args()

    # Load golden dataset
    with open(args.golder) if hasattr(args, 'golden') else open(args.golden) as f:
        samples = [json.loads(line) for line in f]

    print(f"Loaded {len(samples)} samples from {args.golden}")

    # Placeholder: In production, run actual LLM + evaluation here
    # For now, show the structure
    print("NOTE: This is a scaffold. Implement your actual evaluation logic:")
    print("  1. Run each sample through your LLM pipeline")
    print("  2. Compute metrics (ragas, deepeval, or custom)")
    print("  3. Aggregate results")
    print()
    print(f"Run: python run_evals.py --golden {args.golden} --threshold {args.threshold}")

    # Report generation example (when you have results)
    # results = [...]  # Your evaluation results
    # metrics = compute_metrics(results)
    # report = generate_report(results, metrics, args.threshold)
    # with open(args.output, "w") as f:
    #     f.write(report)
    # print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
