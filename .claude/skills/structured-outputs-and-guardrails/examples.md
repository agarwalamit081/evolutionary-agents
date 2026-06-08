---
description: Structured Outputs and Guardrails Examples
---

**Example 1: Pydantic Model for Structured Extraction**

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional

class ExtractedEntity(BaseModel):
    name: str = Field(description="Full name of the person or organization")
    entity_type: Literal["person", "organization", "location"] = Field(description="Type of entity")
    confidence: float = Field(description="Confidence score 0.0-1.0", ge=0, le=1)
    aliases: list[str] = Field(default_factory=list, description="Alternative names or spellings")

class ExtractionResult(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning before extraction")
    entities: list[ExtractedEntity]
    summary: str = Field(description="One-sentence summary of the input text")
```

---

**Example 2: Instructor-Based Extraction with Retry**

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel

class SentimentResult(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    key_phrases: list[str]

client = instructor.from_openai(OpenAI())

result = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=SentimentResult,
    max_retries=3,  # Automatic retry on validation failure
    messages=[{"role": "user", "content": "Analyze: 'The product is amazing but shipping was slow'"}],
)
```

---

**Example 3: Anthropic tool_use for Structured Output**

```python
from anthropic import Anthropic
import json

client = Anthropic()

tools = [{
    "name": "extract_entities",
    "description": "Extract named entities from text",
    "input_schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": ["person", "org", "location"]},
                        "confidence": {"type": "number"}
                    },
                    "required": ["name", "type", "confidence"]
                }
            }
        },
        "required": ["entities"]
    }
}]

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "Apple announced Tim Cook will visit London next week."}],
)

for block in response.content:
    if block.type == "tool_use":
        result = block.input  # Already validated JSON
```

---

**Example 4: Retry Logic with Error Feedback Loop**

```python
import json
from pydantic import BaseModel, ValidationError

class AnalysisResult(BaseModel):
    topic: str
    confidence: float

def extract_with_retry(client, text: str, max_retries: int = 3) -> AnalysisResult:
    messages = [{"role": "user", "content": f"Analyze this text and return JSON with 'topic' and 'confidence':\n{text}"}]

    for attempt in range(max_retries):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            messages=messages,
        )
        try:
            raw = json.loads(response.content[0].text)
            return AnalysisResult(**raw)
        except (json.JSONDecodeError, ValidationError) as e:
            # Feed error back for self-correction
            messages.append({"role": "assistant", "content": response.content[0].text})
            messages.append({
                "role": "user",
                "content": f"Your output failed validation: {e}\nPlease fix and return valid JSON."
            })

    raise RuntimeError(f"Failed after {max_retries} retries")
```

---

**Example 5: Hallucination Detection Guardrail**

```python
import re

def check_hallucination(answer: str, context: str) -> dict:
    """Simple heuristic: check if key claims in answer appear in context."""
    answer_sentences = re.split(r'[.!?]+', answer)
    unsupported = []
    for sentence in answer_sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:
            continue
        words = set(sentence.lower().split())
        context_words = set(context.lower().split())
        overlap = len(words & context_words) / len(words) if words else 0
        if overlap < 0.3:
            unsupported.append(sentence)

    return {
        "hallucination_risk": "high" if len(unsupported) > 2 else "low",
        "unsupported_claims": unsupported,
    }
```

---

**Example 6: Content Moderation Pipeline**

```python
from typing import TypedDict

class ModerationResult(TypedDict):
    is_safe: bool
    flagged_categories: list[str]
    action: str  # "allow", "block", "flag_for_review"

MODERATION_CATEGORIES = ["violence", "hate_speech", "sexual_content", "self_harm", "pii"]

async def moderate_output(text: str) -> ModerationResult:
    flagged = []
    # In production: use Perspective API, Azure Content Safety, or local classifier
    # This is a simplified placeholder
    text_lower = text.lower()

    for category in MODERATION_CATEGORIES:
        # Placeholder — replace with actual classifier
        pass

    is_safe = len(flagged) == 0
    return {
        "is_safe": is_safe,
        "flagged_categories": flagged,
        "action": "allow" if is_safe else "block",
    }
```

---

**Example 7: JSON Schema Validation with Fallback**

```python
import json

def validate_and_parse(raw_output: str, schema: dict) -> dict | None:
    """Try to parse JSON from LLM output, handling markdown code blocks."""
    # Strip markdown code fences if present
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    # Validate required fields
    required = schema.get("required", [])
    for field in required:
        if field not in parsed:
            return None

    return parsed
```

---

**Example 8: Full Guardrail Pipeline**

```python
from pydantic import BaseModel

class GuardedResponse(BaseModel):
    answer: str
    sources: list[str]
    is_safe: bool
    hallucination_risk: str

async def generate_guarded_response(query: str, context: str, client) -> GuardedResponse:
    # 1. Generate
    response = await generate(client, query, context)

    # 2. Validate schema
    parsed = validate_and_parse(response, {"required": ["answer", "sources"]})

    # 3. Check hallucination
    hallu = check_hallucination(parsed["answer"], context)

    # 4. Moderate content
    moderation = await moderate_output(parsed["answer"])

    # 5. Return guarded result
    return GuardedResponse(
        answer=parsed["answer"] if moderation["is_safe"] else "[Content filtered]",
        sources=parsed.get("sources", []),
        is_safe=moderation["is_safe"],
        hallucination_risk=hallu["hallucination_risk"],
    )
```
