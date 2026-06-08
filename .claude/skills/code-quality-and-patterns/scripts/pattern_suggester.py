"""
Suggests design patterns based on a keyword description of a problem.
Usage: python pattern_suggester.py "I have too many if-else statements for different algorithms"
"""
import sys

PATTERNS = {
    "if-else": "Strategy Pattern (encapsulate algorithms) or Factory Pattern (if creating objects).",
    "complex object": "Builder Pattern (step-by-step creation) or Factory.",
    "incompatible interface": "Adapter Pattern.",
    "add behavior dynamically": "Decorator Pattern or Middleware.",
    "notify changes": "Observer Pattern or Pub/Sub.",
    "simplify complex subsystem": "Facade Pattern.",
    "undo/redo": "Command Pattern.",
    "single instance": "Singleton (use with caution) or Dependency Injection scoped lifetime.",
    "too many responsibilities": "Single Responsibility Principle — split into focused classes.",
    "tight coupling": "Dependency Inversion Principle — depend on abstractions, not concretions.",
}


def suggest(problem):
    problem = problem.lower()
    suggestions = []
    for keyword, pattern in PATTERNS.items():
        if keyword in problem:
            suggestions.append(f"- {pattern}")

    if suggestions:
        print("Recommended Patterns:\n" + "\n".join(suggestions))
    else:
        print(
            "No direct pattern match found. "
            "Consider describing the structural or behavioral problem more specifically."
        )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        suggest(" ".join(sys.argv[1:]))
    else:
        print('Usage: python pattern_suggester.py "<description of code problem>"')
