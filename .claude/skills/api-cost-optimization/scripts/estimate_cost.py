"""Estimate API cost for a prompt and expected output length.
Usage: python estimate_cost.py --model claude-sonnet-4-6 --input-tokens 1000 --output-tokens 500
       python estimate_cost.py --model gpt-4o --prompt-file my_prompt.txt --output-tokens 2000
"""
import argparse
import sys


COST_PER_MILLION = {
    "claude-opus-4-8":      {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6":    {"input": 3.00,  "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80,  "output": 4.00},
    "gpt-4o":               {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":          {"input": 0.15,  "output": 0.60},
    "text-embedding-3-small": {"input": 0.02, "output": 0.00},
}


def estimate(model: str, input_tokens: int, output_tokens: int) -> dict:
    costs = COST_PER_MILLION.get(model)
    if not costs:
        print(f"Unknown model: {model}")
        print(f"Supported: {', '.join(COST_PER_MILLION.keys())}")
        sys.exit(1)

    input_cost = (input_tokens / 1_000_000) * costs["input"]
    output_cost = (output_tokens / 1_000_000) * costs["output"]
    total = input_cost + output_cost

    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Estimate LLM API cost")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--input-tokens", type=int, help="Estimated input token count")
    parser.add_argument("--output-tokens", type=int, default=1000, help="Estimated output tokens")
    parser.add_argument("--prompt-file", help="Count tokens from a file (~4 chars/token)")
    parser.add_argument("--calls", type=int, default=1, help="Number of API calls")
    args = parser.parse_args()

    input_tokens = args.input_tokens or 0
    if args.prompt_file:
        with open(args.prompt_file) as f:
            input_tokens = len(f.read()) // 4

    if not input_tokens:
        print("Provide --input-tokens or --prompt-file")
        sys.exit(1)

    result = estimate(args.model, input_tokens, args.output_tokens)
    result["total_cost"] *= args.calls
    result["calls"] = args.calls

    print("--- Cost Estimate ---")
    print(f"Model:          {result['model']}")
    print(f"Input tokens:   {result['input_tokens']:,}")
    print(f"Output tokens:  {result['output_tokens']:,}")
    print(f"Calls:          {args.calls}")
    print(f"Input cost:     ${result['input_cost'] * args.calls:.6f}")
    print(f"Output cost:    ${result['output_cost'] * args.calls:.6f}")
    print(f"Total cost:     ${result['total_cost']:.6f}")


if __name__ == "__main__":
    main()
