#!/usr/bin/env python3
"""Scan code for common security vulnerabilities."""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional


# ── Finding model ──────────────────────────────────────────────────────────

class Finding:
    def __init__(self, file: str, line: int, severity: str, category: str,
                 match: str, remediation: str):
        self.file = file
        self.line = line
        self.severity = severity
        self.category = category
        self.match = match
        self.remediation = remediation

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "line": self.line,
            "severity": self.severity,
            "category": self.category,
            "match": self.match,
            "remediation": self.remediation,
        }


# ── Scan rules ─────────────────────────────────────────────────────────────

RULES: list[tuple[str, str, str, str, str]] = [
    # (pattern, severity, category, display_tag, remediation)
    # 1. Hardcoded secrets
    (r'sk-[a-zA-Z0-9]{20,}', "Critical", "Hardcoded Secret",
     "OpenAI API key",
     "Move to environment variable. Rotate the leaked key immediately."),
    (r'AKIA[0-9A-Z]{16}', "Critical", "Hardcoded Secret",
     "AWS Access Key ID",
     "Use IAM roles or env vars. Rotate the leaked key immediately."),
    (r'ghp_[a-zA-Z0-9]{36}', "Critical", "Hardcoded Secret",
     "GitHub PAT",
     "Move to environment variable or GitHub Secrets. Rotate immediately."),
    (r'xoxb-[0-9]{10,}-[0-9]{10,}-[a-zA-Z0-9]{24}', "Critical",
     "Hardcoded Secret", "Slack Bot Token",
     "Move to environment variable. Rotate the leaked token."),
    (r'(?:api_key|apikey|api_secret|secret_key)\s*=\s*["\'][^"\']+["\']',
     "Critical", "Hardcoded Secret", "API key assignment",
     "Load from environment variable instead of hardcoding."),

    # 2. Raw SQL injection risks
    (r'f["\'].*\b(SELECT|INSERT|UPDATE|DELETE|DROP|ALTER)\b.*\{',
     "Critical", "SQL Injection", "f-string in SQL",
     "Use parameterized queries or ORM filters instead of f-strings in SQL."),
    (r'(?:execute|cursor\.execute)\s*\(\s*f["\']',
     "Critical", "SQL Injection", "Raw SQL with f-string",
     "Use parameterized queries: cursor.execute('... WHERE id=%s', (val,))"),

    # 3. Wildcard CORS
    (r'allow_origins\s*=\s*\[\s*["\']\*["\']\s*\]', "High",
     "CORS Misconfiguration", "Wildcard CORS origin",
     "Replace with explicit frontend domain origins."),
    (r'Access-Control-Allow-Origin:\s*\*', "High",
     "CORS Misconfiguration", "Wildcard CORS header",
     "Replace * with explicit allowed origins."),

    # 4. NEXT_PUBLIC_ / VITE_ sensitive keys
    (r'NEXT_PUBLIC_(?:API_KEY|SECRET|TOKEN|PASSWORD|PRIVATE)',
     "Critical", "Env Var Exposure", "NEXT_PUBLIC_ sensitive key",
     "Remove NEXT_PUBLIC_ prefix. These are bundled into client-side JS."),
    (r'VITE_(?:API_KEY|SECRET|TOKEN|PASSWORD|PRIVATE)',
     "Critical", "Env Var Exposure", "VITE_ sensitive key",
     "Remove VITE_ prefix. These are bundled into client-side JS."),

    # 5. eval / exec usage
    (r'\beval\s*\(', "Critical", "Code Injection", "eval() usage",
     "Remove eval(). Use ast.literal_eval() for literals or refactor."),
    (r'\bexec\s*\(', "Critical", "Code Injection", "exec() usage",
     "Remove exec(). Refactor to avoid dynamic code execution."),

    # 6. dangerouslySetInnerHTML without sanitization
    (r'dangerouslySetInnerHTML', "High", "XSS Risk",
     "dangerouslySetInnerHTML",
     "Ensure content is sanitized with DOMPurify before rendering."),

    # 7. Insecure password hashing
    (r'(?:hashlib\.)?(?:md5|sha1)\s*\(.+(?:password|passwd|pwd)',
     "High", "Insecure Hashing", "MD5/SHA1 for passwords",
     "Use bcrypt, scrypt, or argon2 for password hashing."),

    # 8. Unvalidated file uploads (heuristic)
    (r'UploadFile[^)]*\)[^:]*:', "Medium", "File Upload",
     "UploadFile without validation",
     "Validate file type by magic bytes and enforce a max size limit."),
]


# ── Scanner logic ───────────────────────────────────────────────────────────

def scan_file(filepath: str) -> list[Finding]:
    findings: list[Finding] = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except (OSError, PermissionError):
        return findings

    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("//"):
            continue  # skip comments

        for pattern, severity, category, tag, remediation in RULES:
            if re.search(pattern, stripped, re.IGNORECASE):
                matched = stripped[:80].strip()
                findings.append(Finding(
                    file=filepath,
                    line=lineno,
                    severity=severity,
                    category=category,
                    match=f"{tag}: {matched}",
                    remediation=remediation,
                ))

    return findings


def scan_path(target: str) -> list[Finding]:
    findings: list[Finding] = []
    path = Path(target)

    if path.is_file():
        findings.extend(scan_file(str(path)))
    elif path.is_dir():
        for root, _, files in os.walk(path):
            # Skip common non-source directories
            skip = {"node_modules", ".git", "__pycache__", ".venv", "venv",
                    "dist", "build", ".next", ".cache", "migrations"}
            if any(part in skip for part in Path(root).parts):
                continue
            for fname in files:
                ext = Path(fname).suffix.lower()
                if ext in {".py", ".js", ".ts", ".jsx", ".tsx", ".rb", ".go",
                           ".java", ".env", ".yaml", ".yml", ".json", ".toml",
                           ".cfg", ".ini", ".conf", ".sql"}:
                    findings.extend(scan_file(os.path.join(root, fname)))
    else:
        print(f"Error: {target} is not a file or directory.", file=sys.stderr)
        sys.exit(1)

    return findings


# ── Output formatters ───────────────────────────────────────────────────────

SEVERITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2}


def filter_by_severity(findings: list[Finding], severity: str) -> list[Finding]:
    if severity == "all":
        return findings
    allowed = {s for s, rank in SEVERITY_ORDER.items() if rank <= SEVERITY_ORDER.get(severity, 99)}
    return [f for f in findings if f.severity in allowed]


def format_text(findings: list[Finding]) -> str:
    if not findings:
        return "No security issues found."

    sorted_findings = sorted(
        findings,
        key=lambda f: (SEVERITY_ORDER.get(f.severity, 99), f.file, f.line),
    )

    output_lines: list[str] = []
    current_severity: Optional[str] = None

    for finding in sorted_findings:
        if finding.severity != current_severity:
            current_severity = finding.severity
            output_lines.append(f"\n{'=' * 60}")
            output_lines.append(f"  {current_severity.upper()}")
            output_lines.append(f"{'=' * 60}")

        output_lines.append(
            f"  {finding.file}:{finding.line}  [{finding.category}]"
        )
        output_lines.append(f"    {finding.match}")
        output_lines.append(f"    -> {finding.remediation}")
        output_lines.append("")

    output_lines.append(f"\nTotal: {len(findings)} finding(s)")
    return "\n".join(output_lines)


def format_json(findings: list[Finding]) -> str:
    return json.dumps(
        {"total": len(findings), "findings": [f.to_dict() for f in findings]},
        indent=2,
    )


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan code for common security vulnerabilities.",
    )
    parser.add_argument(
        "--path", required=True,
        help="File or directory to scan.",
    )
    parser.add_argument(
        "--severity", choices=["all", "critical", "high"], default="all",
        help="Minimum severity to report (default: all).",
    )
    parser.add_argument(
        "--report", choices=["text", "json"], default="text",
        help="Output format (default: text).",
    )
    args = parser.parse_args()

    # Map CLI severity to internal names
    severity_map = {"all": "all", "critical": "Critical", "high": "High"}

    findings = scan_path(args.path)
    findings = filter_by_severity(findings, severity_map[args.severity])

    if args.report == "json":
        print(format_json(findings))
    else:
        print(format_text(findings))

    if findings:
        sys.exit(1)  # Non-zero exit signals findings for CI pipelines
    sys.exit(0)


if __name__ == "__main__":
    main()
