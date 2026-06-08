"""Generic retry wrapper that feeds validation errors back to the LLM.
Usage: python retry_with_feedback.py --prompt "Extract entities from: John works at Google" --schema schemas:EntityList
"""
import argparse
import json
import sys
import time

from pydantic import BaseModel, ValidationError


def retry_with_feedback(
    call_llm, prompt: str, model_class: type[BaseModel], max_retries: int = 3
) -> BaseModel:
    """Call LLM, validate output, retry with error feedback on failure."""
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(max_retries):
        print(f"Attempt {attempt + 1}/{max_retries}...")
        raw = call_llm(messages)

        try:
            data = json.loads(raw)
            result = model_class(**data)
            print("Success!")
            return result
        except (json.JSONDecodeError, ValidationError) as e:
            error_msg = str(e)
            print(f"  Validation error: {error_msg[:200]}")

            # Feed error back to LLM
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"Your output failed validation:\n{error_msg}\n\nPlease fix and return valid JSON."
            })

            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff

    raise RuntimeError(f"Failed after {max_retries} retries")


def main():
    parser = argparse.ArgumentParser(description="LLM retry with validation feedback")
    parser.add_argument("--prompt", required=True, help="The prompt to send")
    parser.add_argument("--schema", required=True, help="Pydantic model (module:Class)")
    args = parser.parse_args()

    # Load schema
    module_path, class_name = args.schema.rsplit(":", 1)
    module = __import__(module_path, fromlist=[class_name])
    model_class = getattr(module, class_name)

    # Placeholder LLM call — replace with your actual client
    def call_llm(messages):
        print(f"  Calling LLM with {len(messages)} messages...")
        return '{"result": "placeholder"}'  # Replace with actual LLM call

    try:
        result = retry_with_feedback(call_llm, args.prompt, model_class)
        print(json.dumps(result.model_dump(), indent=2))
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
