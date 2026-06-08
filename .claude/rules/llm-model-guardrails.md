# LLM Provider and Model Guardrails

## Core Directives
1. **Prefer LiteLLM or OpenAI client** as the unified wrapper for all LLM API calls. Use provider-specific SDKs (anthropic, google.generativeai) ONLY when explicitly requested.
2. **NEVER use expensive/flagship models** unless explicitly requested in the chat prompt. Always default to cost-effective models.
3. **NEVER hallucinate model names, base URLs, or API endpoints.** Use ONLY the exact model IDs listed in the reference table below.
4. **NEVER use deprecated model IDs.** Check the blocked list below.

---

## Allowed Model Reference

Use the exact model ID strings below. Do not guess or invent alternatives.

### Anthropic

| Model ID | Cost Tier | Context Window | Max Output | Input | Output | Tool Calling | JSON Mode | Streaming |
|---|---|---|---|---|---|---|---|---|
| `claude-haiku-4-5-20251001` | Cheap ($1/$5 per 1M) | 200K | 64K | Text, Image | Text | ✅ | ✅ | ✅ |
| `claude-sonnet-4-6` | Moderate ($3/$15 per 1M) | 1M | 128K | Text, Image | Text | ✅ | ✅ | ✅ |

### DeepSeek

| Model ID | Cost Tier | Context Window | Max Output | Input | Output | Tool Calling | JSON Mode | Streaming |
|---|---|---|---|---|---|---|---|---|
| `deepseek-v4-flash` | Very Cheap | ~128K | ~384K | Text, Image | Text | ✅ | ✅ | ✅ |
| `deepseek-v4-pro` | Moderate | ~128K | ~384K | Text, Image | Text | ✅ | ✅ | ✅ |

### Z.AI

| Model ID | Cost Tier | Context Window | Max Output | Input | Output | Tool Calling | JSON Mode | Streaming |
|---|---|---|---|---|---|---|---|---|
| `glm-4.7` | Moderate | 128K | 128K | Text, Image | Text | ✅ | ✅ | ✅ |
| `glm-5-turbo` | Moderate | 128K | 131K | Text, Image | Text | ✅ | ✅ | ✅ |
| `glm-4.5-air` | Very Cheap | 128K | 96K | Text | Text | ✅ | ✅ | ✅ |
| `glm-4.7-flash` | Very Cheap | 128K | 131K | Text | Text | ✅ | ✅ | ✅ |
| `glm-4-32b-0414-128k` | Cheap | 128K | 16K (~32K) | Text | Text | ✅ | ✅ | ✅ |
| `glm-4.6v` | Moderate | 128K | 32K | Text, Image | Text | ✅ | ✅ | ✅ |
| `glm-5V-turbo` | Moderate | 128K | 131K | Text, Image | Text | ✅ | ✅ | ✅ |

### MiniMax

| Model ID | Cost Tier | Context Window | Max Output | Input | Output | Tool Calling | JSON Mode | Streaming |
|---|---|---|---|---|---|---|---|---|
| `minimax-m2.5` | Moderate | 32K-128K | 196K | Text, Image | Text | ✅ | ✅ | ✅ |
| `minimax-m2.5-highspeed` | Cheap | 32K-128K | 131K (~196K) | Text, Image | Text | ✅ | ✅ | ✅ |
| `minimax-hailuo-2.3-fast` | Cheap | — | N/A (Video gen) | Text, Image | Text | — | — | — |
| `minimax-hailuo-02` | Very Cheap | — | N/A (Video gen) | Text | Text | — | — | — |

### Mistral

| Model ID | Cost Tier | Context Window | Max Output | Input | Output | Tool Calling | JSON Mode | Streaming |
|---|---|---|---|---|---|---|---|---|
| `mistral-medium-3-5` | Moderate | 128K | 128K-262K | Text, Image, Audio | Text, Audio | ✅ | ✅ | ✅ |
| `mistral-small-2603` | Cheap | 32K-64K | 256K | Text | Text | ✅ | ✅ | ✅ |
| `ministral-3b-2512` | Cheap | 128K | 131K | Text, Image | Text | ✅ | ✅ | ✅ |
| `open-mistral-nemo-2407` | Cheap | 128K | 16K | Text, Image | Text | ✅ | ✅ | ✅ |

### OpenAI

| Model ID | Cost Tier | Context Window | Max Output | Input | Output | Tool Calling | JSON Mode | Streaming |
|---|---|---|---|---|---|---|---|---|
| `gpt-4.1-mini-2025-04-14` | Cheap | 128K | 32K (~33K) | Text, Image | Text | ✅ | ✅ | ✅ |
| `gpt-4o-mini-2024-07-18` | Very Cheap | 128K | 16K | Text, Image | Text | ✅ | ✅ | ✅ |
| `gpt-5-nano-2025-08-07` | Cheap | 128K | 128K | Text, Image | Text | ✅ | ✅ | ✅ |
| `gpt-5.4-nano-2026-03-17` | Cheap | 128K | 128K | Text, Image | Text | ✅ | ✅ | ✅ |
| `gpt-5-mini-2025-08-07` | Moderate | 200K+ | 128K | Text, Image, Audio | Text, Audio | ✅ | ✅ | ✅ |
| `gpt-5-nano-2025-08-07` | Cheap | 128K | 128K | Text, Image | Text | ✅ | ✅ | ✅ |
| `text-embedding-3-large` | Very Cheap | 8K | N/A (Embedding) | Text | Embedding | ❌ | ❌ | ❌ |
| `text-embedding-3-small` | Very Cheap | 8K | N/A (Embedding) | Text | Embedding | ❌ | ❌ | ❌ |

### Google

| Model ID | Cost Tier | Context Window | Max Output | Input | Output | Tool Calling | JSON Mode | Streaming |
|---|---|---|---|---|---|---|---|---|
| `gemini-3-flash-preview` | Moderate | 1M | 65.5K | Text, Image, Audio, Video | Text, Audio, Video | ✅ | ✅ | ✅ |
| `gemini-3.1-flash-lite` | Cheap | 1M | 65.5K | Text, Image, Audio | Text, Audio | ✅ | ✅ | ✅ |
| `gemini-2.5-flash` | Cheap | 1M | 65.5K | Text, Image, Audio, Video | Text, Audio, Video | ✅ | ✅ | ✅ |
| `gemini-2.5-flash-lite` | Very Cheap | 1M | 65.5K | Text, Image | Text | ✅ | ✅ | ✅ |
| `gemini-embedding-2` | Very Cheap | Text Limit | N/A (Embedding) | Text | Embedding | ❌ | ❌ | ❌ |

### Moonshot

| Model ID | Cost Tier | Context Window | Max Output | Input | Output | Tool Calling | JSON Mode | Streaming |
|---|---|---|---|---|---|---|---|---|
| `kimi-k2.6` | Moderate | 128K | 131K-262K | Text, Image | Text | ✅ | ✅ | ✅ |
| `kimi-k2.5` | Moderate | 128K | 16K-131K | Text, Image | Text | ✅ | ✅ | ✅ |
| `moonshot-v1-8k` | Cheap | 8K | 8K | Text | Text | ✅ | ✅ | ✅ |
| `moonshot-v1-32k` | Cheap | 32K | 32K | Text | Text | ✅ | ✅ | ✅ |

### Qwen (Alibaba)

| Model ID | Cost Tier | Context Window | Max Output | Input | Output | Tool Calling | JSON Mode | Streaming |
|---|---|---|---|---|---|---|---|---|
| `qwen3.7-plus` | Cheap ($0.40/$1.60 per 1M) | 1M | 65.5K | Text, Image, Video | Text | ✅ | ✅ | ✅ |
| `qwen3.5-flash` | Very Cheap ($0.065/$0.26 per 1M) | 1M | 66K | Text, Image | Text | ✅ | ✅ | ✅ |

### Groq

| Model ID | Cost Tier | Context Window | Max Output | Input | Output | Tool Calling | JSON Mode | Streaming |
|---|---|---|---|---|---|---|---|---|
| `llama-3.1-8b-instant` | Very Cheap | 128K | 8K (Groq limit) | Text | Text | ✅ | ✅ | ✅ |
| `llama-3.3-70b-versatile` | Moderate | 128K | 32K (Groq limit) | Text, Image | Text | ✅ | ✅ | ✅ |
| `meta-llama/llama-4-scout-17b-16e-instruct` | Cheap | 128K | 8K | Text | Text | ✅ | ✅ | ✅ |
| `qwen/qwen3-32b` | Cheap | 128K | 131K | Text | Text | ✅ | ✅ | ✅ |

### OpenRouter (Free Tier)

| Model ID | Cost Tier | Context Window | Max Output |
|---|---|---|---|
| `openrouter/nvidia/nemotron-3-ultra-550b-a55b:free` | Free | 128K | 65K |
| `openrouter/openai/gpt-oss-120b:free` | Free | 128K | 131K |
| `openrouter/google/gemma-4-31b-it:free` | Free | 128K | 8K |
| `openrouter/moonshotai/kimi-k2.6:free` | Free | 128K | 131K-262K |
| `openrouter/qwen/qwen3-next-80b-a3b-instruct:free` | Free | 128K | 65K-262K |

### Ollama (Local)

| Model ID | Cost Tier | Context Window | Max Output |
|---|---|---|---|
| `phi4-mini:3.8b` | Local (Free) | 8K-32K | 16K |
| `qwen3:1.7b` | Local (Free) | 8K-32K | 32K (~38K) |
| `qwen2.5-coder:7b` | Local (Free) | 8K-32K | 8K |
| `qwen3.5:latest` | Local (Free) | 32K-128K | 131K-262K |
| `deepseek-r1:8b` | Local (Free) | 32K-64K | 8K-32K |
| `gemma4:latest` | Local (Free) | 32K-128K | 8K |
| `qwen3-vl:8b` | Local (Free) | 32K-128K | 33K |

---

## Blocked Models — NEVER Use Unless Explicitly Requested

### Anthropic
- `claude-opus-4-8`, `claude-opus-4-7`

### Z.AI
- `glm-5.1`, `glm-5`

### MiniMax
- `minimax-m2.7`

### Mistral
- `mistral-large-2512`, `devstral-2512`

### OpenAI
- `gpt-4.1-2025-04-14`, `gpt-5.2-2025-12-11`, `gpt-5.2-chat-latest`, `gpt-5.3-chat-latest`, `gpt-5.3-codex`
- `gpt-5.4`, `gpt-5.1-2025-11-13`, `gpt-5.5`, `gpt-5-2025-08-07`
- `o3-2025-04-16`, `gpt-4o-2024-08-06`

### Google
- `gemini-3.1-pro-preview`, `gemini-3-pro-image-preview`, `gemini-2.5-pro`

### x.ai
- `grok-4.3`

### Deprecated — NEVER Use
- `gpt-4-turbo`, `gpt-3.5-turbo`, `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`

---

## Implementation Examples

### Correct: LiteLLM unified wrapper with budget model
```python
import litellm

response = litellm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello world"}]
)
```

### Correct: OpenAI client (when LiteLLM not available)
```python
from openai import OpenAI

client = OpenAI()  # Uses OPENAI_API_KEY env var
response = client.chat.completions.create(
    model="gpt-4o-mini-2024-07-18",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Wrong: Expensive model without explicit request
```python
# BLOCKED: gpt-5.4 is expensive
response = litellm.completion(model="gpt-5.4", messages=[...])
```

### Wrong: Provider-specific SDK
```python
# BLOCKED: Use LiteLLM or OpenAI client instead
import anthropic
client = anthropic.Anthropic()
```
