"""Generate React component boilerplate with TypeScript interface.
Usage: python generate_component.py <ComponentName> [--dir ./src/components]
"""
import os
import argparse


def generate_component(name: str, output_dir: str) -> None:
    pascal = "".join(word.capitalize() for word in name.replace("-", " ").replace("_", " ").split())
    kebab = pascal[0].lower() + "".join(f"-{c.lower()}" if c.isupper() else c for c in pascal[1:])

    tsx = f"""import React from 'react';

interface {pascal}Props {{
  className?: string;
  children?: React.ReactNode;
}}

export const {pascal}: React.FC<{pascal}Props> = ({{ className, children }}) => {{
  return (
    <div className={{`{kebab} ${{className ?? ''}}`}}>
      {{children}}
    </div>
  );
}};
"""

    filepath = os.path.join(output_dir, f"{pascal}.tsx")
    os.makedirs(output_dir, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(tsx)
    print(f"Generated: {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate React component boilerplate")
    parser.add_argument("name", help="Component name (PascalCase or kebab-case)")
    parser.add_argument("--dir", default="./src/components", help="Output directory")
    args = parser.parse_args()
    generate_component(args.name, args.dir)
