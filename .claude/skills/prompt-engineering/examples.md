---
description: Prompt Engineering Examples
---

**Example 1: System Prompt Template with XML Structure**

```xml
You are a senior data analyst. Analyze the provided dataset and generate insights.

<context>
{context}
</context>

<instructions>
1. Identify the top 3 trends in the data.
2. Flag any anomalies or outliers.
3. Provide actionable recommendations.
4. Use specific numbers from the data.
</instructions>

<constraints>
- Do not make up data points not present in the context.
- If data is insufficient, state what is missing.
- Keep each insight to 2-3 sentences.
</constraints>

<output_format>
Return a JSON object:
{
  "trends": [{"title": "...", "description": "...", "data_points": [...]}],
  "anomalies": [...],
  "recommendations": [...]
}
</output_format>
```

---

**Example 2: Few-Shot Classification Prompt**

```
Classify the customer feedback into one of: [billing, technical, feature_request, complaint, praise].

Examples:
Input: "I was charged twice for my subscription"
Output: {"category": "billing", "urgency": "high", "sentiment": "negative"}

Input: "The app crashes when I open settings"
Output: {"category": "technical", "urgency": "high", "sentiment": "negative"}

Input: "Would love a dark mode option"
Output: {"category": "feature_request", "urgency": "low", "sentiment": "neutral"}

Input: "Love the new dashboard design!"
Output: {"category": "praise", "urgency": "low", "sentiment": "positive"}

Now classify:
Input: "{user_input}"
Output:
```

---

**Example 3: Chain-of-Thought Reasoning Prompt**

```
Solve the following problem step by step.

<problem>
A store offers a 20% discount on all items. A customer buys 3 shirts at $25 each
and 2 pairs of pants at $45 each. Sales tax is 8.5%. What is the final total?
</problem>

<thinking>
Step 1: Calculate original subtotal
Step 2: Apply discount
Step 3: Add tax
Step 4: Final total
</thinking>

Show your work in the <thinking> block, then provide the final answer.
```

---

**Example 4: ReAct Agent Prompt**

```
You are an AI assistant that answers questions by reasoning and using tools.

For each step:
1. **Thought**: Think about what you need to do next.
2. **Action**: Call a tool (or decide you have the answer).
3. **Observation**: Review the tool result.

Repeat until you can provide the final answer.

Available tools:
- search_documents(query: str) -> Search internal knowledge base
- calculate(expression: str) -> Evaluate a mathematical expression
- get_user_info(user_id: str) -> Retrieve user profile data

Format each step as:
Thought: [your reasoning]
Action: [tool_name(args)]
Observation: [tool result]

When done:
Final Answer: [your response to the user]
```

---

**Example 5: Structured Output Prompt with Format Specification**

```
Extract all entities from the text below.

<rules>
- Each entity must have: name, type (person/organization/location/date), and confidence (0.0-1.0).
- Only include entities with confidence >= 0.7.
- Return valid JSON only, no other text.
</rules>

<text>
{input_text}
</text>

Return a JSON array:
[
  {"name": "Entity Name", "type": "person", "confidence": 0.95},
  ...
]
```

---

**Example 6: Prompt Template with Variable Interpolation**

```python
from string import Template

SUMMARY_PROMPT = Template("""
Summarize the following ${content_type} in ${max_points} bullet points.

<audience>${audience}</audience>
<content>${content}</content>

Each bullet point should be under ${max_words} words.
Focus on: ${focus_areas}
""")

# Usage
prompt = SUMMARY_PROMPT.safe_substitute(
    content_type="meeting transcript",
    max_points=5,
    audience="engineering team leads",
    content=transcript_text,
    max_words=30,
    focus_areas="decisions made, action items, blockers",
)
```

---

**Example 7: Self-Critique/Refinement Prompt**

```
You are a code reviewer. First, review the code. Then critique your own review.

<code>
{code_snippet}
</code>

Step 1: Write your initial review (bugs, improvements, style issues).
Step 2: Re-read your review and identify:
  - Any points that are subjective rather than objective?
  - Any suggestions that could introduce new bugs?
  - Did you miss any security concerns?
Step 3: Provide your final, refined review incorporating self-critique.
```

---

**Example 8: Prompt Version Metadata Header**

```yaml
# prompt: customer_support_v3
# model: claude-sonnet-4-6
# version: 3.2.1
# last_updated: 2025-01-15
# author: engineering-team
# performance:
#   satisfaction_score: 4.2/5 (up from 3.8 in v3.1)
#   resolution_rate: 78% (up from 72%)
#   avg_turns_to_resolve: 2.3
# changes:
#   - Added explicit escalation criteria for refund > $500
#   - Improved tone for frustrated customers
#   - Added product-specific troubleshooting paths
# a/b_test:
#   control: v3.1 (satisfaction: 3.8)
#   variant: v3.2 (satisfaction: 4.2)
#   winner: v3.2
---
```
