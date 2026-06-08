"""Check package.json for common JS anti-patterns or missing security practices.
Usage: python analyze_js_deps.py [path/to/package.json]
"""
import json
import sys


def check_package_json(filepath="package.json"):
    with open(filepath, "r") as f:
        pkg = json.load(f)

    deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
    warnings = []

    if "lodash" in deps and "lodash-es" not in deps:
        warnings.append("Consider 'lodash-es' for better tree-shaking.")
    if "moment" in deps:
        warnings.append("'moment' is legacy. Consider 'date-fns' or 'dayjs'.")
    if "rxjs" in deps and "rxjs" in pkg.get("devDependencies", {}):
        warnings.append("'rxjs' found in both deps and devDeps — deduplicate.")
    if "axios" in deps and "node-fetch" in deps:
        warnings.append("Both 'axios' and 'node-fetch' found — pick one.")

    if warnings:
        print("JS Dependency Warnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("JS dependencies look good.")


if __name__ == "__main__":
    check_package_json(sys.argv[1] if len(sys.argv) > 1 else "package.json")
