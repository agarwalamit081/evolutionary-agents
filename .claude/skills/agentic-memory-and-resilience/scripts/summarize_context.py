"""Compress conversation history into summaries.
Usage: python summarize_context.py --messages '[{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi!"}]'
"""
import argparse
import json


def summarize_heuristic(messages: list[dict]) -> str:
    """Compress messages using heuristics (no LLM needed)."""
    if not messages:
        return ""

    key_points = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system" or len(content) < 20:
            continue

        # Extract sentences that look like facts or decisions
        sentences = content.replace(". ", ".\n").split("\n")
        for s in sentences:
            s = s.strip()
            if any(kw in s.lower() for kw in ["decided", "confirmed", "result", "error", "found"]):
                key_points.append(s[:150])

    if not key_points:
        # Fallback: first 200 chars of last few messages
        return " | ".join(m["content"][:100] for m in messages[-3:] if m.get("content"))

    return f"[Summary of {len(messages)} messages] " + "; ".join(key_points[:5])


def compress_conversation(messages: list[dict], keep_recent: int = 4) -> list[dict]:
    """Compress old messages into a summary, keeping recent ones intact."""
    if len(messages) <= keep_recent + 1:
        return messages

    system_msg = messages[0] if messages[0]["role"] == "system" else None
    rest = messages[1:] if system_msg else messages

    old = rest[:-keep_recent]
    recent = rest[-keep_recent:]

    summary = summarize_heuristic(old)

    result = []
    if system_msg:
        result.append(system_msg)
    result.append({"role": "system", "content": f"[Previous context]\n{summary}"})
    result.extend(recent)
    return result


def main():
    parser = argparse.ArgumentParser(description="Summarize conversation context")
    parser.add_argument("--messages", required=True, help="JSON array of messages")
    parser.add_argument("--keep-recent", type=int, default=4, help="Number of recent messages to keep")
    args = parser.parse_args()

    messages = json.loads(args.messages)
    original_count = len(messages)
    original_chars = sum(len(m.get("content", "")) for m in messages)

    compressed = compress_conversation(messages, args.keep_recent)
    compressed_count = len(compressed)
    compressed_chars = sum(len(m.get("content", "")) for m in compressed)

    print(f"Original: {original_count} messages, {original_chars:,} chars")
    print(f"Compressed: {compressed_count} messages, {compressed_chars:,} chars")
    print(f"Reduction: {(1 - compressed_chars/original_chars)*100:.1f}%")
    print()
    print("Compressed messages:")
    print(json.dumps(compressed, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
