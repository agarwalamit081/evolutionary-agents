"""Generate a JSONL golden dataset template with sample entries.
Usage: python generate_golden_dataset_template.py [--output golden_dataset.jsonl] [--samples 5]
"""
import argparse
import json


TEMPLATE_ENTRIES = [
    {
        "input": "What is the refund policy?",
        "expected_output": "Customers can request a full refund within 30 days of purchase with a valid receipt.",
        "context": "Refund policy document v2.1",
        "metadata": {"difficulty": "easy", "category": "policy"},
    },
    {
        "input": "Can I return opened electronics?",
        "expected_output": "Opened electronics can be returned within 15 days with a 15% restocking fee.",
        "context": "Electronics return policy section 3.2",
        "metadata": {"difficulty": "medium", "category": "policy"},
    },
    {
        "input": "What if I lost my receipt?",
        "expected_output": "Without a receipt, store credit is issued at the current selling price with valid photo ID.",
        "context": "No-receipt return policy",
        "metadata": {"difficulty": "hard", "category": "edge_case"},
    },
]


def main():
    parser = argparse.ArgumentParser(description="Generate golden dataset template")
    parser.add_argument("--output", default="golden_dataset.jsonl", help="Output file")
    parser.add_argument("--samples", type=int, default=3, help="Number of sample entries")
    args = parser.parse_args()

    with open(args.output, "w", encoding="utf-8") as f:
        for entry in TEMPLATE_ENTRIES[: args.samples]:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Generated {min(args.samples, len(TEMPLATE_ENTRIES))} entries → {args.output}")
    print("Add your own entries following the same JSONL format.")


if __name__ == "__main__":
    main()
