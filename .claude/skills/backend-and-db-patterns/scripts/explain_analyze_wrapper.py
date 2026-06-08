"""Conceptual scaffold for running EXPLAIN ANALYZE on a query file.
Adapt to your project's DB driver (psycopg2, asyncpg, SQLAlchemy).
Usage: python explain_analyze_wrapper.py <query_file.sql>
"""
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python explain_analyze_wrapper.py <query_file.sql>")
        sys.exit(1)

    query_file = sys.argv[1]
    with open(query_file, "r") as f:
        query = f.read().strip()

    if not query.upper().startswith("EXPLAIN"):
        query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) {query}"

    print("Execute this query via your project's DB CLI tool:")
    print(f"  psql -c \"{query[:80]}...\"")
    print()
    print("Or adapt this script to connect via psycopg2/asyncpg for automated analysis.")
    print("Key metrics to check:")
    print("  - Execution time")
    print("  - Sequential scans on large tables (should use index scans)")
    print("  - High row estimates vs actual (stale statistics → run ANALYZE)")


if __name__ == "__main__":
    main()
