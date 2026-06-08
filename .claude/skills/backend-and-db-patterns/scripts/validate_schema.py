"""Validate SQL schema files for enterprise standards.
Usage: python validate_schema.py <path_to_sql_file>
"""
import re
import sys


def validate_sql_file(filepath):
    with open(filepath, "r") as f:
        content = f.read().lower()

    errors = []
    if "created_at" not in content or "updated_at" not in content:
        errors.append("Missing standard audit columns (created_at, updated_at).")
    if re.search(r"\bfloat\b", content) or re.search(r"\breal\b", content):
        errors.append("Avoid FLOAT/REAL for precise data; use NUMERIC/DECIMAL.")
    if "select *" in content:
        errors.append("Avoid SELECT * — explicitly list columns.")

    if errors:
        print(f"Schema validation failed for {filepath}:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print(f"Schema {filepath} passed validation.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_schema.py <path_to_sql_file>")
        sys.exit(1)
    validate_sql_file(sys.argv[1])
