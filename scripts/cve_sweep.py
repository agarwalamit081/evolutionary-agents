"""Periodic CVE sweep over pinned dependencies.

Audits ``requirements.txt`` / ``requirements.optimizer.txt`` against the PyPI /
OSV vulnerability databases using the ``pip-audit`` CLI, then writes a
machine-readable JSON report to ``logs/``. Intended to run on a periodic cadence
(there is no CI here, so this is the dependency-CVE signal). See the project
security rule: "ALWAYS verify package versions ... Check for known CVEs."

Non-fatal by design — a missing ``pip-audit`` binary, a network error, or a
parse hiccup is recorded in the report and the script still exits ``0`` (it is a
report tool, not a gate). Pass ``--strict`` to exit non-zero when any
vulnerability is found, e.g. for an optional pre-deploy check.

This script NEVER installs anything. If ``pip-audit`` is absent it surfaces the
install command as a hint in the report and exits ``0``.

Usage::

    python scripts/cve_sweep.py
    python scripts/cve_sweep.py --requirements requirements.txt --strict
    python scripts/cve_sweep.py --out logs/cve_sweep.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Make ``src`` importable when run as ``python scripts/cve_sweep.py``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REQUIREMENTS = ["requirements.txt", "requirements.optimizer.txt"]
INSTALL_HINT = "Install the scanner with: uv pip install pip-audit"


def _discover_requirements(requested: list[str] | None) -> list[Path]:
    """Resolve the requirements files to audit (default: the ones that exist)."""
    candidates = [REPO_ROOT / f for f in (requested or DEFAULT_REQUIREMENTS)]
    found = [p for p in candidates if p.is_file()]
    if not found:
        logger.warning("No requirements files found among: {}", candidates)
    return found


def _audit_file(audit_bin: str, req_file: Path) -> dict[str, Any]:
    """Run ``pip-audit`` on one requirements file, returning a result record.

    The command never touches the host environment — ``-r`` audits the pinned
    specs against the vuln DB without installing them.
    """
    proc = subprocess.run(  # noqa: S603 — trusted binary resolved via shutil.which
        [audit_bin, "-r", str(req_file), "-f", "json", "--disable-pip"],
        capture_output=True,
        text=True,
        check=False,
    )
    record: dict[str, Any] = {
        "file": str(req_file.relative_to(REPO_ROOT)),
        "returncode": proc.returncode,
        "vulnerabilities": [],
        "stderr": proc.stderr.strip()[-2000:] if proc.stderr else "",
    }
    if proc.stdout:
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            record["parse_error"] = "pip-audit stdout was not valid JSON"
            return record
        for dep in payload.get("dependencies", []):
            for vuln in dep.get("vulns", []):
                record["vulnerabilities"].append(
                    {
                        "package": dep.get("name"),
                        "version": dep.get("version"),
                        "id": vuln.get("id"),
                        "fix_versions": vuln.get("fix_versions"),
                        "description": (vuln.get("description") or "")[:500],
                    }
                )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--requirements",
        action="append",
        help="Requirements file to audit (repeatable; default auto-detected).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="JSON report path (default logs/cve_sweep_<timestamp>.json).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any vulnerability is found.",
    )
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else REPO_ROOT / "logs" / "cve_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scanner": "pip-audit",
        "status": "ok",
        "files": [],
    }

    audit_bin = shutil.which("pip-audit")
    if not audit_bin:
        report["status"] = "scanner-not-installed"
        report["install_hint"] = INSTALL_HINT
        logger.warning("pip-audit CLI not found on PATH — recording a hint, not installing.")
    else:
        for req_file in _discover_requirements(args.requirements):
            logger.info("Auditing {} ...", req_file.name)
            try:
                report["files"].append(_audit_file(audit_bin, req_file))
            except Exception as exc:  # noqa: BLE001 — non-fatal sweep
                report["files"].append(
                    {
                        "file": str(req_file.relative_to(REPO_ROOT)),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                report["status"] = "partial"

    total_vulns = sum(len(f.get("vulnerabilities", [])) for f in report["files"])
    report["total_vulnerabilities"] = total_vulns

    out_path.write_text(json.dumps(report, indent=2))
    logger.info("CVE report written to {} ({} vulnerabilities).", out_path, total_vulns)

    if args.strict and total_vulns:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
