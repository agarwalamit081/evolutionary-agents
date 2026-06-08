"""
Basic heuristic linter to flag potential SOLID violations in Python files.
Usage: python solid_linter.py <path_to_file.py>
"""
import ast
import sys


def analyze(filepath):
    with open(filepath, "r") as f:
        tree = ast.parse(f.read())

    print(f"--- SOLID Heuristic Analysis for {filepath} ---")
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]

            # SRP Heuristic: Too many methods might indicate a God Class
            if len(methods) > 10:
                print(
                    f"  [SRP Risk] Class '{node.name}' has {len(methods)} methods. "
                    "Consider splitting responsibilities."
                )

            # DIP Heuristic: Direct instantiation of heavy dependencies in __init__
            for method in methods:
                if method.name == "__init__":
                    for child in ast.walk(method):
                        if isinstance(child, ast.Call) and isinstance(
                            child.func, ast.Name
                        ):
                            if child.func.id in [
                                "connect",
                                "MySQL",
                                "Postgres",
                                "Redis",
                                "requests",
                            ]:
                                print(
                                    f"  [DIP Risk] Class '{node.name}' directly "
                                    f"instantiates '{child.func.id}'. Prefer DI."
                                )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            analyze(sys.argv[1])
        except FileNotFoundError:
            print(f"File not found: {sys.argv[1]}")
        except SyntaxError:
            print("Invalid Python syntax in the provided file.")
    else:
        print("Usage: python solid_linter.py <path_to_file.py>")
