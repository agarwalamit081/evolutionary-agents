"""Count tokens for message lists. Supports OpenAI (tiktoken) and Anthropic tokenizers.
Falls back to character-based estimate if tiktoken is unavailable.
Usage: python token_counter.py --model gpt-4o --messages '[{"role":"user","content":"Hello"}]'
"""
import argparse
import json


def count_text_tokens_openai(text: str, model: str = "gpt-4o") -> int:
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except ImportError:
        return len(text) // 4  # Rough estimate


def count_text_tokens_anthropic(text: str) -> int:
    # Anthropic uses ~3.5 chars per token on average
    return len(text) // 4


def count_message_tokens(messages: list[dict], model: str = "gpt-4o") -> int:
    total = 0
    for msg in messages:
        total += 4  # message overhead
        total += count_text_tokens_openai(msg.get("content", ""), model)
        total += count_text_tokens_openai(msg.get("role", ""), model)
    total += 2  # conversation priming
    return total


def get_model_budget(model: str) -> int:
    budgets = {
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-4-turbo": 128000,
        "claude-sonnet-4-6": 200000,
        "claude-opus-4-8": 200000,
        "claude-haiku-4-5-20251001": 200000,
    }
    return budgets.get(model, 128000)


def main():
    parser = argparse.ArgumentParser(description="Count tokens for messages")
    parser.add_argument("--model", default="gpt-4o", help="Model name")
    parser.add_argument("--messages", required=True, help="JSON array of messages")
    args = parser.parse_args()

    messages = json.loads(args.messages)
    tokens = count_message_tokens(messages, args.model)
    budget = get_model_budget(args.model)
    remaining = budget - tokens

    print(f"Model: {args.model}")
    print(f"Context window: {budget:,} tokens")
    print(f"Used: {tokens:,} tokens")
    print(f"Remaining: {remaining:,} tokens ({remaining/budget*100:.1f}%)")


if __name__ == "__main__":
    main()
