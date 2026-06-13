#!/bin/bash
# validate_llm_models.sh - Blocks expensive/deprecated LLM models and banned SDK imports
# PostToolUse hook for Edit/Write operations

# Read JSON input from stdin
INPUT=$(cat)

# Extract the file path from tool_input
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // .tool_input.filepath // empty' 2>/dev/null)

if [ -z "$FILE_PATH" ] || [ ! -f "$FILE_PATH" ]; then
  exit 0
fi

# Skip skills, rules, and hooks directories (reference material, not application code)
if echo "$FILE_PATH" | grep -E '(skills/|rules/|hooks/)' > /dev/null; then
  exit 0
fi

# Check for exemption comment
if grep -qE '#\s*EXPENSIVE_MODEL:\s*explicitly\s+requested' "$FILE_PATH" 2>/dev/null; then
  exit 0
fi

# Determine if file is in a tests directory (warn only, don't block)
IS_TEST=0
if echo "$FILE_PATH" | grep -E '(tests/|test/|__tests__/|spec/)' > /dev/null; then
  IS_TEST=1
fi

VIOLATION=0

# 1. Check for banned SDK imports
if grep -qE '(^import anthropic|^from anthropic import|^import google\.generativeai|^from google\.generativeai)' "$FILE_PATH" 2>/dev/null; then
  if [ "$IS_TEST" -eq 1 ]; then
    echo "WARNING: Provider-specific SDK import detected in test file. Prefer LiteLLM or OpenAI client." >&2
  else
    echo "BLOCKED: Provider-specific SDK import detected (anthropic/google.generativeai). Use LiteLLM or OpenAI client as the unified wrapper. Re-read rules/llm-model-guardrails.md for the correct approach." >&2
    VIOLATION=1
  fi
fi

# 2. Check for expensive model names
EXPENSIVE_MODELS=(
  'claude-opus-4-8'
  'claude-opus-4-7'
  'gpt-4\.1-2025-04-14'
  'gpt-5\.2([^a-zA-Z0-9._-]|$)'
  'gpt-5\.3([^a-zA-Z0-9._-]|$)'
  'gpt-5\.4([^a-zA-Z0-9._-]|$)'
  'gpt-5\.5([^a-zA-Z0-9._-]|$)'
  'gpt-5-2025'
  'o3-2025-04-16'
  'gpt-4o-2024-08-06'
  'gemini-2\.5-pro'
  'gemini-3\.1-pro'
  'gemini-3-pro-image'
  'mistral-large'
  'devstral'
  'grok-4\.3([^a-zA-Z0-9._-]|$)'
  'glm-5\.1'
  'glm-5([^a-zA-Z0-9._-]|$)'
  'minimax-m2\.7([^a-zA-Z0-9._-]|$)'
)

for model in "${EXPENSIVE_MODELS[@]}"; do
  if grep -qE "$model" "$FILE_PATH" 2>/dev/null; then
    if [ "$IS_TEST" -eq 1 ]; then
      echo "WARNING: Expensive model '$model' detected in test file. Ensure this is intentional." >&2
    else
      echo "BLOCKED: Expensive model '$model' detected. Default to cost-effective models (gpt-4o-mini, claude-haiku-4-5, deepseek-v4-flash). Use expensive models ONLY when explicitly requested. See rules/llm-model-guardrails.md." >&2
      VIOLATION=1
    fi
    break
  fi
done

# 3. Check for deprecated model references (warn only)
DEPRECATED_MODELS=(
  'gpt-4-turbo'
  'gpt-3\.5-turbo'
  'claude-3-opus'
  'claude-3-sonnet'
  'claude-3-haiku'
)

for model in "${DEPRECATED_MODELS[@]}"; do
  if grep -qE "$model" "$FILE_PATH" 2>/dev/null; then
    echo "WARNING: Deprecated model '$model' detected. Use current model IDs from rules/llm-model-guardrails.md." >&2
    break
  fi
done

# 4. Check for hardcoded max_tokens=4096
if grep -qE 'max_tokens\s*=\s*4096' "$FILE_PATH" 2>/dev/null; then
  echo "WARNING: Hardcoded max_tokens=4096 detected. Set max_tokens based on the model's actual capabilities per rules/llm-model-guardrails.md Max Output column." >&2
fi

if [ "$VIOLATION" -eq 1 ]; then
  exit 2
fi

exit 0
