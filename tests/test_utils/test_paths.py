# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.utils.paths — extended-length path normalization."""

from __future__ import annotations

from pathlib import Path, PureWindowsPath

import pytest

from vane.utils.paths import (
    _normalize_extended_text,
    has_extended_prefix,
    normalize_extended_path,
    plain_path_text,
    resolve_path,
)


class TestNormalizeExtendedText:
    """String-level normalization (platform independent)."""

    def test_short_prefixed_path_loses_prefix_and_dot_dot(self) -> None:
        """A short extended-length path is returned plain with ``..`` collapsed."""
        text = r"\\?\D:\runs\cases\ws10\Monopile\..\Shared\Inflow.dat"
        assert _normalize_extended_text(text) == r"D:\runs\cases\ws10\Shared\Inflow.dat"

    def test_long_prefixed_path_keeps_prefix_but_collapses_dot_dot(self) -> None:
        """Beyond MAX_PATH the prefix stays (it is needed) but ``..`` is collapsed."""
        deep = "\\".join(["d" * 50] * 6)
        text = rf"\\?\D:\{deep}\case\..\Shared\file.dat"
        out = _normalize_extended_text(text)
        assert out.startswith("\\\\?\\D:\\")
        assert ".." not in out
        assert out.endswith(r"\Shared\file.dat")

    def test_unc_prefix_round_trips(self) -> None:
        """An extended UNC path becomes a plain UNC path when short."""
        text = r"\\?\UNC\server\share\a\..\b.dat"
        assert _normalize_extended_text(text) == r"\\server\share\b.dat"

    def test_plain_paths_are_unchanged(self) -> None:
        """Paths without the prefix pass through untouched (including ``..``)."""
        for text in (r"D:\a\..\b", "/home/user/../x", "relative/../y"):
            assert _normalize_extended_text(text) == text


class TestPathHelpers:
    """Path-level wrappers."""

    def test_plain_path_is_returned_as_is(self, tmp_path: Path) -> None:
        """A path with no prefix is returned unchanged (same object)."""
        assert normalize_extended_path(tmp_path) is tmp_path

    @pytest.mark.skipif(
        not isinstance(Path(), PureWindowsPath), reason="Windows path semantics"
    )
    def test_prefixed_windows_path_is_normalized(self) -> None:
        """On Windows a prefixed Path is normalized like its text."""
        path = Path(r"\\?\C:\x\y\..\z.dat")
        assert str(normalize_extended_path(path)) == r"C:\x\z.dat"

    def test_resolve_path_is_absolute(self, tmp_path: Path) -> None:
        """resolve_path returns an absolute, prefix-free path."""
        resolved = resolve_path(tmp_path / "a" / ".." / "b")
        assert resolved.is_absolute()
        assert not str(resolved).startswith("\\\\?\\")
        assert resolved.name == "b"


class TestPlainPathText:
    """Dropping the extended-length prefix for programs that are not long-path aware."""

    def test_long_prefixed_path_loses_prefix(self) -> None:
        """Unlike normalization, the prefix goes even beyond MAX_PATH."""
        deep = "\\".join(["d" * 50] * 6)
        text = rf"\\?\D:\{deep}\case\..\Shared\file.dat"
        out = plain_path_text(text)
        assert out == rf"D:\{deep}\Shared\file.dat"
        assert len(out) > 260

    def test_unc_prefix_becomes_unc_path(self) -> None:
        """An extended UNC path becomes a plain UNC path."""
        out = plain_path_text(r"\\?\UNC\server\share\a.dat")
        assert out == r"\\server\share\a.dat"

    def test_plain_text_is_unchanged(self) -> None:
        """A path without the prefix is returned verbatim (``..`` kept)."""
        assert plain_path_text(r"C:\case\..\x.dat") == r"C:\case\..\x.dat"
        assert plain_path_text(Path("rel") / "x.dat") == str(Path("rel") / "x.dat")

    def test_has_extended_prefix(self) -> None:
        """The prefix test looks at the path text only."""
        assert has_extended_prefix(r"\\?\C:\x")
        assert has_extended_prefix(PureWindowsPath(r"\\?\C:\x"))
        assert not has_extended_prefix(r"C:\x")
