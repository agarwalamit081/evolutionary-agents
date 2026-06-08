#!/usr/bin/env python3
"""Generate Playwright Page Object Model class templates.

Usage:
    python generate_pom.py --name login
    python generate_pom.py --name login --url https://example.com/login
"""

import argparse
import re
import sys
import urllib.request
from html.parser import HTMLParser


class LocatorParser(HTMLParser):
    """Parse HTML to find interactive elements and suggest Playwright locators."""

    INTERACTIVE_TAGS = {"button", "a", "input", "select", "textarea"}

    def __init__(self):
        super().__init__()
        self.elements = []
        self._current = None

    def handle_starttag(self, tag, attrs):
        if tag not in self.INTERACTIVE_TAGS:
            return
        attr_dict = dict(attrs)
        self._current = {
            "tag": tag,
            "attrs": attr_dict,
            "text": "",
        }

    def handle_data(self, data):
        if self._current is not None:
            self._current["text"] += data.strip()

    def handle_endtag(self, tag):
        if self._current and tag == self._current["tag"]:
            self.elements.append(self._current)
            self._current = None


def _sanitize_name(text, tag):
    """Create a valid TS property name from element text or attributes."""
    source = text or tag
    name = re.sub(r"[^a-zA-Z0-9]+", " ", source).strip().lower()
    parts = name.split()[:4]
    return "_".join(parts) if parts else f"{tag}_element"


def suggest_locators(url):
    """Fetch a URL and suggest Playwright locators for interactive elements."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"Warning: Could not fetch {url}: {e}", file=sys.stderr)
        return []

    parser = LocatorParser()
    parser.feed(html)

    locators = []
    for el in parser.elements:
        tag = el["tag"]
        attrs = el["attrs"]
        text = el["text"][:30] if el["text"] else ""
        name = _sanitize_name(
            attrs.get("data-testid") or attrs.get("name") or attrs.get("id") or text,
            tag,
        )

        # Suggest locator following Playwright priority
        if "data-testid" in attrs:
            locator = f"getByTestId('{attrs['data-testid']}')"
        elif tag == "button" and text:
            locator = f"getByRole('button', {{ name: '{text}' }})"
        elif tag == "a" and text:
            locator = f"getByRole('link', {{ name: '{text}' }})"
        elif tag == "input" and attrs.get("type") in ("submit", "button"):
            locator = f"getByRole('button', {{ name: '{text or attrs.get('value', 'submit')}' }})"
        elif tag == "input" and "placeholder" in attrs:
            locator = f"getByPlaceholder('{attrs['placeholder']}')"
        elif tag == "input" and "aria-label" in attrs:
            locator = f"getByLabel('{attrs['aria-label']}')"
        elif tag == "input" and "name" in attrs:
            locator = f"getByLabel('{attrs['name']}')"
        elif tag == "select" and "name" in attrs:
            locator = f"getByLabel('{attrs['name']}')"
        elif text:
            locator = f"getByText('{text}')"
        else:
            continue

        is_action = tag in ("button", "a") or (
            tag == "input" and attrs.get("type") in ("submit", "button")
        )
        locators.append((name, locator, is_action))

    return locators


def generate_pom(page_name, locators=None):
    class_name = "".join(word.capitalize() for word in page_name.replace("-", " ").split())
    file_name = f"{class_name}Page.ts"

    if locators:
        fields = []
        assignments = []
        methods = []

        for name, locator, is_action in locators:
            ts_name = re.sub(r"[^a-zA-Z0-9]", "", name.title().replace(" ", ""))
            ts_name = ts_name[0].lower() + ts_name[1:] if ts_name else name

            fields.append(f"  readonly {ts_name}: Locator;")
            assignments.append(f"    this.{ts_name} = page.{locator};")

            if is_action:
                methods.append(
                    f"  async click{ts_name[0].upper() + ts_name[1:]}() {{\n"
                    f"    await this.{ts_name}.click();\n"
                    f"  }}"
                )
            elif "input" in name or "email" in name or "password" in name:
                methods.append(
                    f"  async fill{ts_name[0].upper() + ts_name[1:]}(value: string) {{\n"
                    f"    await this.{ts_name}.fill(value);\n"
                    f"  }}"
                )

        fields_str = "\n".join(fields)
        assign_str = "\n".join(assignments)
        methods_str = "\n\n".join(methods)
        methods_block = f"\n\n{methods_str}" if methods_str else ""

        content = f"""import {{ Page, Locator }} from '@playwright/test';

export class {class_name}Page {{
  readonly page: Page;

{fields_str}

  constructor(page: Page) {{
    this.page = page;
{assign_str}
  }}
{methods_block}
}}
"""
    else:
        content = f"""import {{ Page, Locator }} from '@playwright/test';

export class {class_name}Page {{
  readonly page: Page;
  // TODO: Add locators below using getByRole, getByTestId, etc.
  // readonly myButton: Locator;

  constructor(page: Page) {{
    this.page = page;
    // this.myButton = page.getByTestId('my-button');
  }}

  async navigate() {{
    await this.page.goto('/TODO');
  }}

  // TODO: Add interaction methods
  // async clickMyButton() {{
  //   await this.myButton.click();
  // }}
}}
"""

    with open(file_name, "w") as f:
        f.write(content)
    print(f"Generated: {file_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Playwright Page Object Model class templates."
    )
    parser.add_argument("--name", required=True, help="Page name (e.g. login, checkout)")
    parser.add_argument(
        "--url",
        help="Optional URL to scan for interactive elements and suggest locators",
    )
    args = parser.parse_args()

    locators = suggest_locators(args.url) if args.url else None
    generate_pom(args.name, locators)


if __name__ == "__main__":
    main()
