"""Generate Mermaid diagram from a compiled LangGraph graph.
Usage: python visualize_graph.py --module myapp.graph --var app --output diagram.md
       python visualize_graph.py --template                   # Print a template diagram
"""
import argparse
import importlib
import sys


TEMPLATE = """```mermaid
graph TD
    Start([Start]) --> A[research_node]
    A --> B{{route_decision}}
    B -->|continue| C[analyze_node]
    B -->|error| D[error_correction_node]
    D --> A
    C --> E([End])
```"""


def from_graph(module_path: str, var_name: str) -> str:
    """Import a compiled graph and extract its Mermaid representation."""
    try:
        module = importlib.import_module(module_path)
        app = getattr(module, var_name)
        mermaid = app.get_graph().draw_mermaid()
        return f"```mermaid\n{mermaid}\n```"
    except ImportError as e:
        return f"Error: Could not import '{module_path}'. {e}"
    except AttributeError:
        return f"Error: Variable '{var_name}' not found in '{module_path}'."
    except Exception as e:
        return f"Error: {e}"


def main():
    parser = argparse.ArgumentParser(description="Visualize LangGraph graphs as Mermaid diagrams")
    parser.add_argument("--module", help="Python module path containing the graph (e.g., myapp.graph)")
    parser.add_argument("--var", default="app", help="Variable name of the compiled graph (default: app)")
    parser.add_argument("--template", action="store_true", help="Print a template Mermaid diagram")
    parser.add_argument("--output", help="Output file (default: stdout)")
    args = parser.parse_args()

    if args.template:
        result = TEMPLATE
    elif args.module:
        result = from_graph(args.module, args.var)
    else:
        print("Usage: python visualize_graph.py --module <module> --var <var> OR --template")
        sys.exit(1)

    if args.output:
        with open(args.output, "w") as f:
            f.write(result)
        print(f"Diagram saved to {args.output}")
    else:
        print(result)


if __name__ == "__main__":
    main()
