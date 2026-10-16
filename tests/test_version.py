# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests that every file carrying the VANE version states the same release."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import vane
from vane._version import vane_version

_ROOT = Path(__file__).resolve().parent.parent


def _search(relative: str, pattern: str) -> str:
    """Return the first ``version`` group of ``pattern`` in a repository file."""
    path = _ROOT / relative
    if not path.is_file():
        pytest.skip(f"{relative} is not present (installed package without sources)")
    match = re.search(pattern, path.read_text(encoding="utf-8"), flags=re.MULTILINE)
    assert match is not None, f"no version found in {relative}"
    return match.group("version")


class TestPackageVersion:
    """Tests for the version exposed by the package."""

    def test_version_is_a_release_string(self) -> None:
        """``vane.__version__`` is a ``MAJOR.MINOR.PATCH`` string."""
        assert re.fullmatch(r"\d+\.\d+\.\d+", vane.__version__)

    def test_version_is_exported(self) -> None:
        """``__version__`` appears in ``vane.__all__``."""
        assert "__version__" in vane.__all__

    def test_provenance_version_is_the_package_version(self) -> None:
        """Provenance records carry the package version."""
        assert vane_version() == vane.__version__


class TestVersionAgreement:
    """The package, metadata, documentation, citation and changelog agree."""

    @pytest.mark.parametrize(
        ("relative", "pattern"),
        [
            ("pyproject.toml", r'^version\s*=\s*"(?P<version>[^"]+)"'),
            ("docs/conf.py", r'^release\s*=\s*"(?P<version>[^"]+)"'),
            ("CITATION.cff", r"^version:\s*(?P<version>\S+)"),
            ("CHANGELOG.md", r"^## \[(?P<version>\d+\.\d+\.\d+)\]"),
        ],
    )
    def test_file_states_the_package_version(self, relative: str, pattern: str) -> None:
        """Each file states the same version as ``vane.__version__``."""
        assert _search(relative, pattern) == vane.__version__

    def test_release_is_0_1_0_and_stable(self) -> None:
        """The release is 0.1.0 and is classified as production/stable."""
        assert vane.__version__ == "0.1.0"
        status = _search(
            "pyproject.toml", r'"Development Status :: (?P<version>[^"]+)"'
        )
        assert status == "5 - Production/Stable"

    def test_citation_date_matches_changelog(self) -> None:
        """The citation release date is the date of the latest changelog entry."""
        cited = _search("CITATION.cff", r"^date-released:\s*(?P<version>\S+)")
        logged = _search("CHANGELOG.md", r"^## \[\d+\.\d+\.\d+\] - (?P<version>\S+)")
        assert cited.strip("'\"") == logged
