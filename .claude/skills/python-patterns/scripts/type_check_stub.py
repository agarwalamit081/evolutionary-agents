"""Scaffold type-checked function stubs from a simple spec.
Usage: python type_check_stub.py "def fetch_user(user_id: str) -> User" "def delete_user(user_id: str) -> bool"
"""
import argparse
import textwrap


def generate_stub(signature: str) -> str:
    """Convert a function signature into a type-checked stub."""
    # Extract function name
    name = signature.split("(")[0].replace("def ", "").strip()
    params = signature.split("(")[1].split(")")[0]
    return_type = signature.split("->")[-1].strip() if "->" in signature else "None"

    stub = textwrap.dedent(f"""\
    {signature}:
        \"\"\"TODO: Implement {name}.\"\"\"
        # Parameters: {params}
        # Returns: {return_type}
        raise NotImplementedError
    """)
    return stub


def main():
    parser = argparse.ArgumentParser(description="Generate type-checked function stubs")
    parser.add_argument("signatures", nargs="+", help="Function signatures")
    args = parser.parse_args()

    print("from __future__ import annotations")
    print()
    for sig in args.signatures:
        print(generate_stub(sig))
        print()


if __name__ == "__main__":
    main()
