"""Validate a JSON tool schema file for completeness.
Usage: python validate_tool_schema.py <schema_file.json>
"""
import json
import sys


def validate_schema(schema: dict) -> list[str]:
    errors = []

    # Top-level required fields
    for field in ["name", "description", "input_schema"]:
        if field not in schema:
            errors.append(f"Missing required field: {field}")

    if errors:
        return errors

    # Name should be verb_noun format
    name = schema["name"]
    if "_" not in name and not any(c.isupper() for c in name):
        errors.append(f"Tool name '{name}' should follow verb_noun or camelCase convention")

    # Description should be substantive (>20 chars)
    if len(schema.get("description", "")) < 20:
        errors.append("Description too short — should explain when to use the tool")

    # Input schema validation
    input_schema = schema.get("input_schema", {})
    if input_schema.get("type") != "object":
        errors.append("input_schema.type should be 'object'")

    properties = input_schema.get("properties", {})
    if not properties:
        errors.append("input_schema should define at least one property")

    for prop_name, prop_def in properties.items():
        if "type" not in prop_def:
            errors.append(f"Property '{prop_name}' missing 'type'")
        if "description" not in prop_def:
            errors.append(f"Property '{prop_name}' missing 'description'")

    # Required fields should be listed
    required = input_schema.get("required", [])
    if not isinstance(required, list):
        errors.append("'required' should be a list of property names")

    return errors


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_tool_schema.py <schema_file.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        schema = json.load(f)

    errors = validate_schema(schema)

    if errors:
        print(f"Schema validation FAILED for {sys.argv[1]}:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print(f"Schema validation PASSED for {sys.argv[1]}")
        print(f"  Tool: {schema['name']}")
        print(f"  Properties: {list(schema.get('input_schema', {}).get('properties', {}).keys())}")
        print(f"  Required: {schema.get('input_schema', {}).get('required', [])}")


if __name__ == "__main__":
    main()
