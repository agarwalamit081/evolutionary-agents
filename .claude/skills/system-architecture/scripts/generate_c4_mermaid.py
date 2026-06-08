"""
Generates a C4 Context Mermaid diagram from a simple semicolon-delimited input.
Usage: python generate_c4_mermaid.py "System:Banking App; Person:Admin; Rel:Admin->Banking App:Manages"
"""
import sys


def generate_mermaid(input_str):
    parts = input_str.split(";")
    mermaid = ["C4Context", "  title Generated System Context Diagram"]

    for part in parts:
        part = part.strip()
        if part.startswith("System:"):
            name = part.split(":", 1)[1]
            mermaid.append(f'  System({name.lower().replace(" ", "_")}, "{name}")')
        elif part.startswith("Person:"):
            name = part.split(":", 1)[1]
            mermaid.append(f'  Person({name.lower().replace(" ", "_")}, "{name}")')
        elif part.startswith("Rel:"):
            rel = part.split(":", 1)[1]
            mermaid.append(f"  Rel({rel})")

    return "\n".join(mermaid)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(generate_mermaid(sys.argv[1]))
    else:
        print('Usage: python generate_c4_mermaid.py "System:App; Person:User; Rel:User->App:Uses"')
