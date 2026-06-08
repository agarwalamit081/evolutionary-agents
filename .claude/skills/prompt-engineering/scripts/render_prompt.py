"""Render a prompt template with variable substitution.
Usage: python render_prompt.py template.txt --var key1=value1 --var key2=value2
"""
import argparse
import re
from pathlib import Path


def render_template(template_path: str, variables: dict[str, str]) -> str:
    template = Path(template_path).read_text(encoding="utf-8")

    # Support both {variable} and {{variable}} syntax
    def replacer(match):
        key = match.group(1) or match.group(2)
        if key in variables:
            return variables[key]
        return match.group(0)  # Leave unresolved placeholders

    result = re.sub(r"\{(\w+)\}|{{(\w+)}}", replacer, template)
    return result


def main():
    parser = argparse.ArgumentParser(description="Render prompt template")
    parser.add_argument("template", help="Path to prompt template file")
    parser.add_argument("--var", action="append", default=[], help="key=value pairs")
    parser.add_argument("--output", help="Output file (default: stdout)")
    args = parser.parse_args()

    variables = {}
    for pair in args.var:
        key, _, value = pair.partition("=")
        variables[key] = value

    rendered = render_template(args.template, variables)

    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
        print(f"Rendered prompt saved to {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
