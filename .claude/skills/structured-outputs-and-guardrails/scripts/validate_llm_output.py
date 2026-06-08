"""Validate LLM JSON output against a Pydantic schema file.
Usage: python validate_llm_output.py <json_file> <schema_module.ClassName>
Example: python validate_llm_output.py output.json schemas:ExtractionResult
"""
import importlib
import json
import sys

from pydantic import ValidationError


def validate(json_file: str, schema_ref: str):
    # Load JSON
    with open(json_file) as f:
        raw = f.read().strip()

    # Strip markdown code fences
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
        if raw.endswith("```"):
            raw = raw[:-3]
    data = json.loads(raw.strip())

    # Load Pydantic model
    module_path, class_name = schema_ref.rsplit(":", 1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)

    # Validate
    try:
        result = model_class(**data)
        print(f"Validation PASSED: {class_name}")
        print(f"Fields: {list(result.model_fields_set)}")
    except ValidationError as e:
        print(f"Validation FAILED: {class_name}")
        for error in e.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            print(f"  - {field}: {error['msg']} ({error['type']})")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        validate(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python validate_llm_output.py <json_file> <module:Class>")
        print("Example: python validate_llm_output.py output.json schemas:ExtractionResult")
