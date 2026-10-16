# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.io.fst_writer — the format-preserving input editor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from vane.io.fst_writer import FastInputError, FastInputFile

if TYPE_CHECKING:
    from pathlib import Path

_DECK = """\
------- OpenFAST INPUT -------------------------------------------
        600   TMax            - Total run time (s)
---------------------- LINEARIZATION -----------------------------
False         Linearize       - Linearization analysis (flag)
False         CalcSteady      - Calculate a steady-state operating point (flag)
          1   NLinTimes       - Number of times to linearize (-) [>=1]
          0   LinTimes        - List of times at which to linearize (s)
"ES20.12E3"   OutFmt          - Format used for text tabular output (string)
"NRELOffshrBsline.dat"    EDFile          - ElastoDyn input file (quoted string)
"unused"      HydroFile       - HydroDyn input file (quoted string)
"""


def _deck(tmp_path: Path) -> Path:
    """Write the sample deck to a temp file and return its path."""
    path = tmp_path / "model.fst"
    path.write_text(_DECK, encoding="utf-8")
    return path


class TestRead:
    """Tests for reading and looking up keywords."""

    def test_get_and_contains(self, tmp_path: Path) -> None:
        """Get returns the first value token and __contains__ finds keywords."""
        deck = FastInputFile.read(_deck(tmp_path))
        assert "Linearize" in deck
        assert "NotAKeyword" not in deck
        assert deck.get("NLinTimes") == "1"
        assert deck.get("Linearize") == "False"
        assert deck.get("OutFmt") == '"ES20.12E3"'

    def test_missing_keyword_raises(self, tmp_path: Path) -> None:
        """Reading an absent keyword raises a clear error."""
        deck = FastInputFile.read(_deck(tmp_path))
        with pytest.raises(FastInputError, match="not found"):
            deck.get("DoesNotExist")

    def test_read_missing_file_raises(self, tmp_path: Path) -> None:
        """Reading a non-existent file raises FastInputError."""
        with pytest.raises(FastInputError, match="Cannot read"):
            FastInputFile.read(tmp_path / "nope.fst")


class TestSet:
    """Tests for editing values while preserving the line format."""

    def test_set_preserves_keyword_and_comment(self, tmp_path: Path) -> None:
        """Editing a value keeps the keyword and trailing comment intact."""
        deck = FastInputFile.read(_deck(tmp_path))
        deck.set("Linearize", True)
        deck.set("NLinTimes", 36)
        line = next(line for line in deck.text().splitlines() if "NLinTimes" in line)
        assert line.split(" - ")[0].split()[0] == "36"
        assert "Number of times to linearize" in line
        assert deck.get("Linearize") == "True"
        assert deck.get("NLinTimes") == "36"

    def test_set_bool_renders_true_false(self, tmp_path: Path) -> None:
        """Booleans render as True/False regardless of the original token."""
        deck = FastInputFile.read(_deck(tmp_path))
        deck.set("CalcSteady", True)
        assert deck.get("CalcSteady") == "True"
        deck.set("CalcSteady", False)
        assert deck.get("CalcSteady") == "False"

    def test_set_string_is_verbatim(self, tmp_path: Path) -> None:
        """A string value is written exactly as given (quotes preserved)."""
        deck = FastInputFile.read(_deck(tmp_path))
        deck.set("OutFmt", '"ES20.11E3"')
        assert deck.get("OutFmt") == '"ES20.11E3"'

    def test_set_array(self, tmp_path: Path) -> None:
        """set_array writes a space-separated value list for the keyword."""
        deck = FastInputFile.read(_deck(tmp_path))
        deck.set_array("LinTimes", [0.0, 30.0, 60.0])
        assert deck.tokens("LinTimes") == ["0.0", "30.0", "60.0"]

    def test_set_missing_keyword_raises(self, tmp_path: Path) -> None:
        """Editing an absent keyword raises a clear error."""
        deck = FastInputFile.read(_deck(tmp_path))
        with pytest.raises(FastInputError, match="not found"):
            deck.set("Nope", 1)

    def test_keyword_match_is_case_insensitive(self, tmp_path: Path) -> None:
        """A differently-cased keyword is found and edited in the deck's spelling."""
        path = tmp_path / "inflow.dat"
        path.write_text(
            "0.2           PLexp       - Power law exponent (-)\n", encoding="utf-8"
        )
        deck = FastInputFile.read(path)
        assert "PLExp" in deck
        deck.set("PLExp", 0.0)
        assert deck.get("PLExp") == "0.0"
        assert "PLexp" in deck.text()
        assert "PLExp" not in deck.text()

    def test_exact_keyword_match_is_preferred(self, tmp_path: Path) -> None:
        """An exact-case line wins over an earlier line that differs only in case."""
        path = tmp_path / "deck.dat"
        path.write_text(
            "1             tmax        - lower-case twin (-)\n"
            "2             TMax        - Total run time (s)\n",
            encoding="utf-8",
        )
        deck = FastInputFile.read(path)
        assert deck.get("TMax") == "2"
        deck.set("TMax", 600)
        assert deck.get("tmax") == "1"

    def test_long_value_stays_parseable(self, tmp_path: Path) -> None:
        """A value wider than the keyword column still parses (keyword shifts)."""
        deck = FastInputFile.read(_deck(tmp_path))
        deck.set("NLinTimes", "123456789012345678")
        assert deck.get("NLinTimes") == "123456789012345678"


class TestResolveAndWrite:
    """Tests for sub-file resolution and round-trip writing."""

    def test_resolve_subfile_path(self, tmp_path: Path) -> None:
        """Resolve returns the sub-file path relative to the deck directory."""
        deck = FastInputFile.read(_deck(tmp_path))
        resolved = deck.resolve("EDFile")
        assert resolved == tmp_path / "NRELOffshrBsline.dat"

    def test_quoted_path_containing_separator(self, tmp_path: Path) -> None:
        """A quoted path containing ``" - "`` is found, resolved, and rewritten."""
        path = tmp_path / "dash.fst"
        path.write_text(
            '"Run A - base/ED.dat"   EDFile   - ElastoDyn input (quoted string)\n',
            encoding="utf-8",
        )
        deck = FastInputFile.read(path)
        assert "EDFile" in deck
        assert deck.tokens("EDFile") == ['"Run A - base/ED.dat"']
        assert deck.resolve("EDFile") == tmp_path / "Run A - base" / "ED.dat"
        deck.set("EDFile", '"Run B - base/ED.dat"')
        assert deck.text() == (
            '"Run B - base/ED.dat"   EDFile   - ElastoDyn input (quoted string)\n'
        )

    def test_resolve_single_quoted_path(self, tmp_path: Path) -> None:
        """Fortran also delimits a string with apostrophes."""
        path = tmp_path / "apostrophe.fst"
        path.write_text("'Sub Dir/ED.dat'   EDFile   - ElastoDyn\n", encoding="utf-8")
        deck = FastInputFile.read(path)
        assert deck.resolve("EDFile") == tmp_path / "Sub Dir" / "ED.dat"

    def test_non_utf8_bytes_round_trip(self, tmp_path: Path) -> None:
        """A Windows-1252 byte in a comment is read and written back unchanged."""
        path = tmp_path / "cp1252.fst"
        path.write_bytes(
            b"0.0  BlPitch(1)  - Blade 1 initial pitch (\xb0)\n"
            b"  600   TMax  - Total run time (s)\n"
        )
        deck = FastInputFile.read(path)
        assert deck.get("TMax") == "600"
        deck.set("TMax", 700)
        deck.write()
        first, second = path.read_bytes().splitlines()
        assert first == b"0.0  BlPitch(1)  - Blade 1 initial pitch (\xb0)"
        assert second.split()[:2] == [b"700", b"TMax"]

    def test_resolve_unused_raises(self, tmp_path: Path) -> None:
        """An ``unused`` sub-file slot has no resolvable path."""
        deck = FastInputFile.read(_deck(tmp_path))
        with pytest.raises(FastInputError, match="no usable file path"):
            deck.resolve("HydroFile")

    def test_round_trip_preserves_other_lines(self, tmp_path: Path) -> None:
        """Writing after one edit changes only that line, not line count."""
        deck = FastInputFile.read(_deck(tmp_path))
        original_lines = deck.text().splitlines()
        deck.set("TMax", 700)
        out = deck.write(tmp_path / "out.fst")
        new_lines = out.read_text(encoding="utf-8").splitlines()
        assert len(new_lines) == len(original_lines)
        pairs = zip(original_lines, new_lines, strict=True)
        differing = [a for a, b in pairs if a != b]
        assert len(differing) == 1
        assert "TMax" in differing[0]

    def test_write_defaults_to_source_path(self, tmp_path: Path) -> None:
        """Write with no argument overwrites the source file."""
        path = _deck(tmp_path)
        deck = FastInputFile.read(path)
        deck.set("NLinTimes", 12)
        deck.write()
        assert FastInputFile.read(path).get("NLinTimes") == "12"

    def test_write_is_atomic_and_breaks_hard_links(self, tmp_path: Path) -> None:
        """Editing a hard-linked deck breaks the link, leaving the original intact."""
        import os

        original = _deck(tmp_path)
        linked = tmp_path / "linked.fst"
        os.link(original, linked)  # hard link: the two paths share one inode
        assert original.stat().st_ino == linked.stat().st_ino

        deck = FastInputFile.read(linked)
        deck.set("NLinTimes", 99)
        deck.write()

        # The atomic replace gives the edited path a new inode; the original (and any
        # other hard link to its inode, e.g. a shared template) is untouched.
        assert original.stat().st_ino != linked.stat().st_ino
        assert FastInputFile.read(linked).get("NLinTimes") == "99"
        assert FastInputFile.read(original).get("NLinTimes") == "1"

    def test_write_does_not_clobber_a_predictable_temp(self, tmp_path: Path) -> None:
        """A pre-existing ``<deck>.tmp`` hard link is never mutated by a write."""
        import os

        path = _deck(tmp_path)
        # A stale, predictable temp hard-linked from elsewhere (e.g. a shared template).
        keeper = tmp_path / "keeper.tmp"
        keeper.write_text("do not touch", encoding="utf-8")
        os.link(keeper, tmp_path / f"{path.name}.tmp")

        deck = FastInputFile.read(path)
        deck.set("NLinTimes", 7)
        deck.write()

        # The write used a freshly created unique temp, so the predictable hard-linked
        # ``.tmp`` (and its shared inode) is untouched, and the deck still updated.
        assert keeper.read_text(encoding="utf-8") == "do not touch"
        assert FastInputFile.read(path).get("NLinTimes") == "7"

    def test_write_preserves_destination_permissions(self, tmp_path: Path) -> None:
        """The atomic replace keeps the deck's mode (not mkstemp's 0o600)."""
        import stat

        path = _deck(tmp_path)
        path.chmod(0o640)  # POSIX: distinctive mode; Windows: only the RO bit
        expected = stat.S_IMODE(path.stat().st_mode)  # what the OS actually applied
        deck = FastInputFile.read(path)
        deck.set("NLinTimes", 5)
        deck.write()
        assert stat.S_IMODE(path.stat().st_mode) == expected
