# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Format-preserving editor for OpenFAST input files.

OpenFAST input decks use the convention ``<value(s)> <Keyword> - <description>``,
where the value(s) precede the keyword on each line. To drive a linearization,
VANE must *change* a handful of these values — enable ``Linearize``, set
``NLinTimes``, set the operating point's ``RotSpeed`` and ``HWindSpeed``, and so on
— without disturbing the rest of the deck. This module provides
:class:`FastInputFile`, a small editor that reads a deck, looks up a keyword,
replaces its value while preserving the keyword spelling, the trailing comment, and
the column layout, and writes the deck back. It is the write-side counterpart of
:mod:`vane.io.fst_reader`.

The editor operates on a single file. A deck spans several files (the primary
``.fst`` references ElastoDyn, InflowWind, ServoDyn, ...); :meth:`FastInputFile.
resolve` returns a referenced sub-file's path so the caller can open and edit it
the same way.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from vane.io.fst_reader import _split_field_line
from vane.utils.fs_retry import retry_transient
from vane.utils.paths import normalize_extended_path

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["FastInputError", "FastInputFile"]

# Bytes that are not UTF-8 are carried through a read/write round trip unchanged.
_ENCODING_ERRORS = "surrogateescape"


class FastInputError(ValueError):
    """Raised when an OpenFAST input file keyword cannot be found or edited."""


class FastInputFile:
    """An editable OpenFAST input deck file.

    The file is held as its original lines; :meth:`set` rewrites only the value of
    a single keyword's line, leaving every other character — including the keyword,
    the ``- description`` comment, and the surrounding whitespace — intact.

    Parameters
    ----------
    path : pathlib.Path
        Path the deck was read from (used to resolve referenced sub-files and as
        the default write destination).
    lines : list[str]
        The file's lines, without trailing newlines.
    """

    def __init__(self, path: Path, lines: list[str]) -> None:
        self.path = path
        self._lines = lines

    @classmethod
    def read(cls, path: str | Path) -> FastInputFile:
        """Read an OpenFAST input file.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the input file.

        Returns
        -------
        FastInputFile
            The editable deck.

        Raises
        ------
        FastInputError
            If the file cannot be read.

        Notes
        -----
        The file is decoded as UTF-8; a byte that is not UTF-8 (a Windows-1252
        ``°`` in a comment, say) is kept as is and written back unchanged by
        :meth:`write`, as OpenFAST itself reads the file byte by byte.
        """
        resolved = Path(path)
        try:
            text = resolved.read_text(encoding="utf-8", errors=_ENCODING_ERRORS)
        except OSError as exc:
            msg = f"Cannot read OpenFAST input file {resolved}: {exc}"
            raise FastInputError(msg) from exc
        return cls(resolved, text.splitlines())

    def __contains__(self, keyword: str) -> bool:
        """Return whether ``keyword`` labels a line in the deck."""
        return self._find(keyword) is not None

    def tokens(self, keyword: str) -> list[str]:
        """Return the value tokens preceding ``keyword`` on its line.

        Parameters
        ----------
        keyword : str
            The exact keyword (field name) to locate.

        Returns
        -------
        list[str]
            The whitespace-delimited value tokens before the keyword.

        Raises
        ------
        FastInputError
            If the keyword is not present.
        """
        index = self._find(keyword)
        if index is None:
            msg = f"Keyword {keyword!r} not found in {self.path}"
            raise FastInputError(msg)
        tokens, _ = _split_field_line(self._lines[index])
        return tokens[:-1]

    def get(self, keyword: str) -> str:
        """Return the first value token of ``keyword``.

        Parameters
        ----------
        keyword : str
            The keyword to read.

        Returns
        -------
        str
            The first value token (quotes preserved).

        Raises
        ------
        FastInputError
            If the keyword is not present or has no value.
        """
        tokens = self.tokens(keyword)
        if not tokens:
            msg = f"Keyword {keyword!r} has no value in {self.path}"
            raise FastInputError(msg)
        return tokens[0]

    def set(self, keyword: str, value: str | int | float | bool) -> None:
        """Replace the value of ``keyword`` in place.

        The keyword, its column, and the trailing ``- description`` comment are
        preserved; only the leading value field is rewritten. Booleans render as
        ``True``/``False``; ints and floats use ``str``; a ``str`` is written
        verbatim (so quoted paths and format codes such as ``'"ES20.11E3"'`` are
        the caller's responsibility).

        Parameters
        ----------
        keyword : str
            The keyword whose value to change.
        value : str or int or float or bool
            The new value.

        Raises
        ------
        FastInputError
            If the keyword is not present.
        """
        index = self._find(keyword)
        if index is None:
            msg = f"Keyword {keyword!r} not found in {self.path}"
            raise FastInputError(msg)
        self._lines[index] = self._rewrite(self._lines[index], keyword, _fmt(value))

    def set_array(self, keyword: str, values: Sequence[float]) -> None:
        """Replace ``keyword`` with a space-separated list of values.

        Used for array fields such as ``LinTimes``.

        Parameters
        ----------
        keyword : str
            The array keyword.
        values : Sequence[float]
            The values to write, joined with two spaces.

        Raises
        ------
        FastInputError
            If the keyword is not present.
        """
        index = self._find(keyword)
        if index is None:
            msg = f"Keyword {keyword!r} not found in {self.path}"
            raise FastInputError(msg)
        joined = "  ".join(_fmt(value) for value in values)
        self._lines[index] = self._rewrite(self._lines[index], keyword, joined)

    def resolve(self, keyword: str) -> Path:
        """Return the path of a referenced sub-file given its keyword.

        The value is treated as a path (quoted with ``"`` or ``'``, as Fortran
        accepts, or bare) relative to this deck's
        directory.

        Parameters
        ----------
        keyword : str
            The keyword whose value is a sub-file path (e.g. ``"EDFile"``).

        Returns
        -------
        pathlib.Path
            The resolved sub-file path.

        Raises
        ------
        FastInputError
            If the keyword is absent or its value is empty/``"unused"``.
        """
        tokens = self.tokens(keyword)
        value = " ".join(tokens).strip().strip("\"'")
        if not value or value.lower() == "unused":
            msg = f"Keyword {keyword!r} has no usable file path in {self.path}"
            raise FastInputError(msg)
        return normalize_extended_path(self.path.parent / value)

    def text(self) -> str:
        """Return the current deck as text with a trailing newline."""
        return "\n".join(self._lines) + "\n"

    def write(self, path: str | Path | None = None) -> Path:
        """Write the deck to disk.

        Parameters
        ----------
        path : str or pathlib.Path or None, optional
            Destination path; defaults to the path the deck was read from. Parent
            directories are created.

        Returns
        -------
        pathlib.Path
            The path written to.

        Notes
        -----
        The write is atomic: the text is written to a freshly created, uniquely named
        temporary file in the same directory and then :meth:`Path.replace`-d into
        place. This avoids leaving a partially written deck on failure and — because
        the replace swaps the directory entry to a new inode — editing a hard-linked
        copy never modifies the shared (e.g. template) inode, which lets a sweep
        hard-link its heavy read-only inputs while still editing the small deck files
        safely. The temporary file is created exclusively (a unique name), so it never
        collides with or mutates a pre-existing ``.tmp`` hard link, and an existing
        deck's permission bits are preserved across the replace.

        On Windows the temporary-file creation and the replace can fail transiently
        with an access-denied or sharing-violation error while another process (an
        antivirus scan, the search indexer, or an OpenFAST process still releasing its
        handles) briefly holds the deck or its directory. The whole write is then
        retried a few times with a short backoff (see
        :func:`vane.utils.fs_retry.retry_transient`) before the error is raised.
        """
        destination = Path(path) if path is not None else self.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        text = self.text()
        retry_transient(
            lambda: _write_atomic(destination, text),
            description=f"write {destination.name}",
        )
        return destination

    def _find(self, keyword: str) -> int | None:
        """Return the index of the line labeled ``keyword``, or ``None``.

        OpenFAST matches keywords case-insensitively, and decks from different
        generators spell some differently (``PLExp``/``PLexp``). An exact match is
        preferred; otherwise the first case-insensitive match is returned.
        """
        folded = keyword.casefold()
        fallback: int | None = None
        for index, line in enumerate(self._lines):
            parts, _ = _split_field_line(line)
            if len(parts) < 2:
                continue
            if parts[-1] == keyword:
                return index
            if fallback is None and parts[-1].casefold() == folded:
                fallback = index
        return fallback

    @staticmethod
    def _rewrite(line: str, keyword: str, new_value: str) -> str:
        """Return ``line`` with its value replaced by ``new_value``.

        The keyword's column is preserved when the new value fits; otherwise the
        keyword is shifted right with a two-space separator so the line still
        parses.
        """
        parts, head_end = _split_field_line(line)
        head, description = line[:head_end], line[head_end:]
        keyword = parts[-1]  # the deck's own spelling, which may differ in case
        keyword_col = head.rfind(keyword)
        tail = head[keyword_col:]  # keyword plus any spacing up to " - "
        if len(new_value) < keyword_col:
            new_head = new_value.ljust(keyword_col) + tail
        else:
            new_head = f"{new_value}  {tail}"
        return new_head + description


def _write_atomic(destination: Path, text: str) -> None:
    """Write ``text`` to ``destination`` through a temporary file and a replace."""
    descriptor, temp_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f"{destination.name}.", suffix=".tmp"
    )
    temp = Path(temp_name)
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", errors=_ENCODING_ERRORS
        ) as handle:
            handle.write(text)
        if destination.exists():
            # mkstemp creates the temp 0o600; keep the deck's original mode.
            shutil.copymode(destination, temp)
        temp.replace(destination)
    finally:
        # Remove the temp on any failure (write or replace); a no-op once the
        # successful replace has renamed it away.
        temp.unlink(missing_ok=True)


def _fmt(value: str | int | float | bool) -> str:
    """Format a value for an OpenFAST input field."""
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)
