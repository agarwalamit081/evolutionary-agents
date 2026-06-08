"""Generate Chrome extension project scaffold.
Usage: python generate_extension_scaffold.py --name "My Extension" --template react --features popup,content,background
"""
import argparse
import json
from pathlib import Path


def slugify(name: str) -> str:
    return name.lower().replace(" ", "-").replace("_", "-")


def generate_manifest(slug: str, description: str, features: list[str]) -> dict:
    manifest = {
        "manifest_version": 3,
        "name": slug.replace("-", " ").title(),
        "version": "1.0.0",
        "description": description or f"{slug.replace('-', ' ')} chrome extension",
        "permissions": ["storage"],
        "host_permissions": [],
        "icons": {
            "16": "assets/icons/icon-16.png",
            "48": "assets/icons/icon-48.png",
            "128": "assets/icons/icon-128.png",
        },
    }

    if "popup" in features:
        manifest["action"] = {"default_popup": "popup/index.html"}

    if "background" in features:
        manifest["background"] = {
            "service_worker": "background/index.ts",
            "type": "module",
        }

    if "content" in features:
        manifest["content_scripts"] = [
            {"matches": ["<all_urls>"], "js": ["content/index.ts"], "run_at": "document_idle"}
        ]

    if "options" in features:
        manifest["options_page"] = "options/index.html"

    return manifest


def create_scaffold(name: str, template: str, features: list[str], output_dir: str):
    slug = slugify(name)
    base = Path(output_dir) / slug

    # Create directory structure
    dirs = ["src/lib", "src/assets/icons", "public"]
    for feat in features:
        dirs.append(f"src/{feat}")

    for d in dirs:
        (base / d).mkdir(parents=True, exist_ok=True)

    # Generate manifest.json
    manifest = generate_manifest(slug, "", features)
    with open(base / "public" / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Generate package.json
    pkg = {
        "name": slug,
        "version": "1.0.0",
        "scripts": {"dev": "vite", "build": "tsc --noEmit && vite build"},
        "devDependencies": {
            "typescript": "^5.4.0",
            "vite": "^5.0.0",
            "@crxjs/vite-plugin": "^2.0.0",
        },
    }
    if template == "react":
        pkg["devDependencies"].update({
            "react": "^19.0.0",
            "react-dom": "^19.0.0",
            "@types/react": "^19.0.0",
            "@types/react-dom": "^19.0.0",
            "@vitejs/plugin-react": "^4.0.0",
        })

    with open(base / "package.json", "w") as f:
        json.dump(pkg, f, indent=2)

    # Generate tsconfig.json
    tsconfig = {
        "compilerOptions": {
            "target": "ES2020",
            "module": "ESNext",
            "moduleResolution": "bundler",
            "strict": True,
            "jsx": "react-jsx" if template == "react" else None,
            "outDir": "dist",
        },
        "include": ["src"],
    }
    tsconfig["compilerOptions"] = {k: v for k, v in tsconfig["compilerOptions"].items() if v is not None}
    with open(base / "tsconfig.json", "w") as f:
        json.dump(tsconfig, f, indent=2)

    # Generate .gitignore
    with open(base / ".gitignore", "w") as f:
        f.write("node_modules/\ndist/\n*.env\n")

    print(f"Generated: {base}/")
    print(f"  Features: {', '.join(features)}")
    print(f"  Template: {template}")
    print(f"\nNext steps:")
    print(f"  cd {slug} && npm install && npm run dev")


def main():
    parser = argparse.ArgumentParser(description="Generate Chrome extension scaffold")
    parser.add_argument("--name", required=True, help="Extension name")
    parser.add_argument("--template", choices=["react", "vanilla"], default="react")
    parser.add_argument("--features", default="popup,content,background",
                        help="Comma-separated: popup,content,background,options")
    parser.add_argument("--output", default=".", help="Output directory")
    args = parser.parse_args()

    features = [f.strip() for f in args.features.split(",")]
    create_scaffold(args.name, args.template, features, args.output)


if __name__ == "__main__":
    main()
