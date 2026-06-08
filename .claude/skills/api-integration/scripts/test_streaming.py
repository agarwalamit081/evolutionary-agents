"""Quick test script for streaming LLM API responses to stdout.
Usage: python test_streaming.py --provider openai --prompt "Tell me a story" --model gpt-4o-mini
"""
import argparse
import asyncio
import os
import time


async def test_openai(prompt: str, model: str):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    print(f"Streaming from OpenAI ({model}):\n")
    start = time.time()
    async for chunk in await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ):
        token = chunk.choices[0].delta.content or ""
        print(token, end="", flush=True)

    elapsed = time.time() - start
    print(f"\n\nCompleted in {elapsed:.2f}s")


async def test_anthropic(prompt: str, model: str):
    from anthropic import AsyncAnthropic
    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    print(f"Streaming from Anthropic ({model}):\n")
    start = time.time()
    async with client.messages.stream(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for text in stream.text_stream:
            print(text, end="", flush=True)

    elapsed = time.time() - start
    print(f"\n\nCompleted in {elapsed:.2f}s")


async def main():
    parser = argparse.ArgumentParser(description="Test streaming LLM API")
    parser.add_argument("--provider", choices=["openai", "anthropic"], required=True)
    parser.add_argument("--prompt", default="Explain async/await in 3 sentences.")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    if args.provider == "openai":
        model = args.model or "gpt-4o-mini"
        await test_openai(args.prompt, model)
    else:
        model = args.model or "claude-sonnet-4-6"
        await test_anthropic(args.prompt, model)


if __name__ == "__main__":
    asyncio.run(main())
