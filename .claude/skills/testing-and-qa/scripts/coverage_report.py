"""Scaffold for coverage reporting. Adapt to your test runner (pytest, jest).
Usage: python coverage_report.py [--threshold 80]
"""
import argparse
import subprocess
import sys


def run_coverage(threshold: int):
    print(f"Running tests with coverage (threshold: {threshold}%)...")

    # Adapt these commands to your project:
    # Python/pytest:  pytest --cov=src --cov-report=term-missing --cov-fail-under={threshold}
    # Node/jest:      npx jest --coverage --coverageThreshold='{"global":{"branches":{threshold}}}'

    try:
        result = subprocess.run(
            ["pytest", "--cov=src", f"--cov-fail-under={threshold}",
             "--cov-report=term-missing", "-q"],
            capture_output=True, text=True, timeout=300
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"Coverage below {threshold}% or tests failed.")
            sys.exit(1)
        print(f"All tests passed with coverage >= {threshold}%.")
    except FileNotFoundError:
        print("pytest not found. Install with: pip install pytest pytest-cov")
        print("Or adapt this script for your test runner (jest, vitest, etc.).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, default=80, help="Minimum coverage %%")
    args = parser.parse_args()
    run_coverage(args.threshold)
