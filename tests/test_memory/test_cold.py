"""src.memory.cold — JSONB tag-predicate regression (operator-type bug).

``ColdMemory.search_by_tags`` previously built its WHERE clause with
``context_tags.bool_op("?|")(tags)``. SQLAlchemy binds a Python list as jsonb,
and Postgres defines ``jsonb ?| text[]`` only — there is no ``jsonb ?| jsonb``
operator — so the query raised ``operator does not exist: jsonb ?| jsonb`` at
execution time (asyncpg). Every ``search_by_tags`` test mocked the method, so
the defect never surfaced until the live search smoke (which copied the same
expression) hit a real Postgres.

The fix routes the predicate through ``_tags_any_predicate``, which OR's the
JSONB existence operator ``?`` per tag (each binds a single text param). These
tests lock the fix by compiling the predicate under the postgresql dialect —
they fail if anyone reverts to the broken ``?|``-with-list form. The live-DB
proof is ``scripts/smoke_search_corpus.py`` (cleanup reports ``0 remain``).
"""

from __future__ import annotations

from sqlalchemy.dialects import postgresql

from src.memory.cold import _tags_any_predicate


def _compile(tags: list[str]) -> tuple[str, dict[str, object]]:
    compiled = _tags_any_predicate(tags).compile(dialect=postgresql.dialect())
    return str(compiled), dict(compiled.params)


class TestTagsAnyPredicate:
    def test_uses_jsonb_existence_operator_not_anyof(self) -> None:
        sql, _ = _compile(["url:https://example.invalid/x", "author:mark"])
        # The fixed form: per-tag JSONB existence (?) OR'd together.
        assert "?" in sql
        # The broken form (jsonb ?| jsonb) must never reappear.
        assert "?|" not in sql
        # One existence predicate per tag.
        assert sql.count("context_tags ?") == 2
        assert " OR " in sql

    def test_binds_each_tag_as_text_not_a_jsonb_blob(self) -> None:
        tags = ["url:https://example.invalid/x", "author:mark"]
        _, params = _compile(tags)
        # Each tag is a separate string param — not a single jsonb array, which
        # is what triggers the operator-type mismatch.
        assert set(params.values()) == set(tags)
        assert len(params) == len(tags)
        assert all(isinstance(v, str) for v in params.values())

    def test_empty_tags_matches_no_rows(self) -> None:
        # Prior empty-array ``?|`` semantics: no element → no match.
        sql = str(_tags_any_predicate([]).compile(dialect=postgresql.dialect()))
        assert sql.strip() == "false"

    def test_single_tag_compiles_without_or(self) -> None:
        sql, _ = _compile(["hash:abc"])
        assert "?|" not in sql
        assert sql.count("context_tags ?") == 1
        assert " OR " not in sql


class TestBrokenFormRejected:
    """Guard against re-introducing the ``?|``-with-list operator bug."""

    def test_anyof_with_list_does_not_match_fixed_operator(self) -> None:
        # The broken expression the bug used. Compiled here only to assert the
        # FIXED predicate does NOT render the same operator — they must differ.
        from src.db.models import ColdMemory as ColdMemoryModel

        broken = str(
            ColdMemoryModel.context_tags.bool_op("?|")(["a", "b"]).compile(
                dialect=postgresql.dialect()
            )
        )
        fixed, _ = _compile(["a", "b"])
        assert "?|" in broken  # the bug's signature
        assert "?|" not in fixed  # the fix does not share it
