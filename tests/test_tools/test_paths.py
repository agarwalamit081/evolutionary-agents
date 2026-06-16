"""Unit tests for the shared path resolver (``src.tools._paths``).

This is the single source of truth every file-touching tool resolves through,
so its de-nesting + traversal-guard semantics are tested directly (independent
of any one tool). Roots are monkeypatched at the single source the resolver
reads — ``src.config.settings.get_settings`` — so these stay hermetic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.tools import _paths


def _install_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the resolver at tmp_path/results + tmp_path/workspace.

    project_root then resolves to tmp_path (parent of results_root), matching
    the cwd alignment every file tool relies on.
    """
    results = tmp_path / "results"
    workspace = tmp_path / "workspace"
    fake = type(
        "S",
        (),
        {
            "agent": type(
                "A",
                (),
                {
                    "results_root": str(results),
                    "workspace_root": str(workspace),
                },
            )()
        },
    )()
    monkeypatch.setattr("src.config.settings.get_settings", lambda: fake)


class TestStripResultsPrefix:
    """``strip_results_prefix`` collapses redundant leading root components."""

    def test_strips_literal_results(self, monkeypatch, tmp_path) -> None:
        _install_roots(monkeypatch, tmp_path)
        assert _paths.strip_results_prefix(("results", "a.md")) == ("a.md",)

    def test_strips_double_nest(self, monkeypatch, tmp_path) -> None:
        """The original fabrication bug: results/results/x → x."""
        _install_roots(monkeypatch, tmp_path)
        assert _paths.strip_results_prefix(("results", "results", "x.md")) == ("x.md",)

    def test_strips_results_root_name(self, monkeypatch, tmp_path) -> None:
        """A leading component equal to the configured results_root name strips."""
        _install_roots(monkeypatch, tmp_path)
        # results_root dir name is "results" here, so this also covers the
        # name-based strip; verify it handles a subfolder beneath it.
        assert _paths.strip_results_prefix(("results", "sub", "b.md")) == ("sub", "b.md")

    def test_never_strips_lone_filename(self, monkeypatch, tmp_path) -> None:
        _install_roots(monkeypatch, tmp_path)
        assert _paths.strip_results_prefix(("x.md",)) == ("x.md",)

    def test_preserves_unrelated_prefix(self, monkeypatch, tmp_path) -> None:
        _install_roots(monkeypatch, tmp_path)
        assert _paths.strip_results_prefix(("reports", "x.md")) == ("reports", "x.md")


class TestNormalize:
    """``normalize`` de-nests, joins under the base, and guards traversal."""

    def test_results_base_resolves_under_results_root(self, monkeypatch, tmp_path) -> None:
        _install_roots(monkeypatch, tmp_path)
        assert _paths.normalize("a/b.md", base="results") == (tmp_path / "results" / "a" / "b.md")

    def test_results_prefix_is_de_nested(self, monkeypatch, tmp_path) -> None:
        _install_roots(monkeypatch, tmp_path)
        assert _paths.normalize("results/a.md", base="results") == (tmp_path / "results" / "a.md")

    def test_workspace_base(self, monkeypatch, tmp_path) -> None:
        _install_roots(monkeypatch, tmp_path)
        assert _paths.normalize("c.txt", base="workspace") == (tmp_path / "workspace" / "c.txt")

    def test_project_base(self, monkeypatch, tmp_path) -> None:
        _install_roots(monkeypatch, tmp_path)
        # base="project" (== results_root.parent); the leading "results" is in
        # the strip set, so it de-nests to project_root/d.md.
        assert _paths.normalize("results/d.md", base="project") == (tmp_path / "d.md")
        assert _paths.normalize("d.md", base="project") == (tmp_path / "d.md")

    def test_explicit_path_base_de_nests_own_name(self, monkeypatch, tmp_path) -> None:
        _install_roots(monkeypatch, tmp_path)
        sandbox = tmp_path / "sandbox"
        # An explicit base root strips its own leading name too.
        assert _paths.normalize("sandbox/x.md", base=sandbox) == (sandbox / "x.md")

    def test_traversal_blocked(self, monkeypatch, tmp_path) -> None:
        _install_roots(monkeypatch, tmp_path)
        with pytest.raises(ValueError):
            _paths.normalize("../../etc/passwd", base="results")

    def test_absolute_outside_root_blocked(self, monkeypatch, tmp_path) -> None:
        _install_roots(monkeypatch, tmp_path)
        with pytest.raises(ValueError):
            _paths.normalize("/etc/passwd", base="results")


class TestNormalizeSymlinks:
    """``normalize`` calls ``Path.resolve()`` so symlink targets are followed.

    These pin the real, observed behavior of the traversal guard against
    symlinks (not an aspirational contract): an inside-root link resolves and
    stays put; an escaping link resolves outside root and is blocked with
    ``ValueError``; a dangling link resolves to a non-existent path that is
    still inside root (``resolve()`` does not raise on Python 3.12) and is
    returned as-is. Roots are monkeypatched exactly as in ``TestNormalize``.
    """

    def test_symlink_inside_root_resolves(self, monkeypatch, tmp_path) -> None:
        """A symlink under results_root pointing to another in-root file resolves."""
        _install_roots(monkeypatch, tmp_path)
        root = _paths.results_root()
        root.mkdir(parents=True, exist_ok=True)
        target = root / "real.md"
        target.write_text("x")
        link = root / "link.md"
        link.symlink_to(target)

        resolved = _paths.normalize("link.md", base="results")

        # resolve() follows the link to the real target, which is still under root.
        assert resolved == target.resolve()
        assert resolved.is_relative_to(root)

    def test_symlink_escaping_root_blocked(self, monkeypatch, tmp_path) -> None:
        """A symlink under root pointing OUTSIDE root is blocked (raises ValueError).

        normalize resolves the link to its outside-root target, then the
        ``is_relative_to(root)`` guard rejects it with "Path traversal blocked".
        This is the clamp-free, strip-free behavior: it raises, matching how
        callers translate ValueError into their ERROR: strings.
        """
        _install_roots(monkeypatch, tmp_path)
        root = _paths.results_root()
        root.mkdir(parents=True, exist_ok=True)
        outside = tmp_path / "secret.txt"
        outside.write_text("y")
        (root / "escape.md").symlink_to(outside)

        with pytest.raises(ValueError, match="Path traversal blocked"):
            _paths.normalize("escape.md", base="results")

    def test_broken_symlink(self, monkeypatch, tmp_path) -> None:
        """A dangling symlink under root resolves safely (no crash, stays in root).

        resolve() on Python 3.12 returns the literal non-existent target path
        rather than raising, and that target is still under root, so normalize
        returns it. The link simply does not point at a real file (exists() is
        False) — normalize does not validate existence, only containment.
        """
        _install_roots(monkeypatch, tmp_path)
        root = _paths.results_root()
        root.mkdir(parents=True, exist_ok=True)
        (root / "dangling.md").symlink_to(root / "nope.md")

        resolved = _paths.normalize("dangling.md", base="results")

        assert resolved.is_relative_to(root)
        assert not resolved.exists()

    def test_normalize_idempotent(self, monkeypatch, tmp_path) -> None:
        """Normalizing an already-normalized path returns an equivalent path."""
        _install_roots(monkeypatch, tmp_path)
        once = _paths.normalize("sub/a/b.md", base="results")
        twice = _paths.normalize(str(once), base="results")

        # Re-normalizing the resolved absolute path de-nests the leading
        # results_root component and re-joins under root → same resolved target.
        assert once == twice
        assert twice == (tmp_path / "results" / "sub" / "a" / "b.md")


class TestRoots:
    """The resolved root helpers agree on the project_root = results_root.parent invariant."""

    def test_project_root_is_results_parent(self, monkeypatch, tmp_path) -> None:
        _install_roots(monkeypatch, tmp_path)
        assert _paths.project_root() == _paths.results_root().parent
        assert _paths.project_root() == tmp_path
