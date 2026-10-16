# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Find and copy only the files an OpenFAST deck references.

A sweep copies the template case once per operating point so that each case can be
edited independently. When the template's sub-files live outside the deck's own folder
(``../5MW_Baseline/...``), ``template_root`` must be a higher directory -- for the
OpenFAST regression-test suite, a directory holding dozens of unrelated cases and their
outputs. Copying that whole tree per point costs hundreds of megabytes per case and
carries unrelated ``.lin`` files into the case directory.

:func:`deck_file_closure` instead walks the deck's references: starting from the
primary ``.fst`` it reads every referenced text input (ElastoDyn, its blade and tower
files, AeroDyn and its airfoil polars and ``@``-prefixed coordinate files, ServoDyn and
its controller DLL and controller input file, the controller's own performance file,
InflowWind wind files, SeaState, HydroDyn and its potential-flow data, SubDyn,
MoorDyn, structural-control files, ...) and follows them recursively. The walk does not
rely on a keyword list: every value token that looks like a path is resolved, relative
to the referencing file's directory (the OpenFAST convention) and to the deck's
directory (OpenFAST's working directory), and kept only if it names an existing file
under ``template_root``. Tokens that are numbers, ``"unused"``, ``"default"`` or do not
name a file are ignored, so an unknown keyword never stops the walk. Two kinds of
reference name more than one file and are expanded:

* a *root name* (HydroDyn ``PotFile``, SeaState ``WvKinFile``, InflowWind
  ``FilenameRoot`` and any other keyword ending in ``Root``) brings every file
  ``<root>.<ext>`` in its directory (``.1``, ``.3``, ``.hst``, ``.ss``, ``.ssexctn``,
  ``.12d``, ``.12s``, ``.Elev``, ...), except OpenFAST output files;
* a shared library (``.dll``, ``.so``, ``.dylib``) brings its platform and bitness
  variants of the same stem (``DISCON.dll``, ``DISCON.so``, ``DISCON_x64.dll``, ...).
  Other files next to the library are copied only when they are referenced.

:func:`copy_template_files` copies the closure into a case directory, preserving each
file's path relative to ``template_root`` so every relative reference still resolves.
When the walk cannot vouch for completeness -- a relative reference that escapes
``template_root`` or an input that cannot be read -- it logs a warning and copies the
whole tree instead, exactly as before; ``mode="tree"`` selects that conservative copy
outright. Correctness takes precedence over size. A reference given as an absolute path
outside ``template_root`` keeps working from the case copy and is left in place (an
*edited* input outside the case is still rejected by
:func:`~vane.run.require_inputs_within`).
"""

from __future__ import annotations

import logging
import ntpath
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from vane.io.fst_reader import fortran_float
from vane.utils.fs_retry import (
    copy2_with_retry,
    link_with_retry,
    rmtree_with_retry,
)
from vane.utils.paths import normalize_extended_path, plain_path_text, resolve_path

__all__ = [
    "DeckFileClosure",
    "DeckReferenceError",
    "TemplateCopyMode",
    "copy_template_files",
    "deck_file_closure",
    "infer_template_root",
]

logger = logging.getLogger(__name__)

TemplateCopyMode = Literal["referenced", "tree"]
"""How a template is copied into a case: its referenced files, or the whole tree."""

_COPY_MODES: tuple[TemplateCopyMode, ...] = ("referenced", "tree")

# Suffixes of files OpenFAST writes/overwrites per run. Such a file in the template is
# always copied (independent inode), never hard-linked, so a run cannot mutate an inode
# shared with the template or another concurrent case. A root-name expansion never
# selects them: they are results, not inputs.
OUTPUT_SUFFIXES = frozenset(
    {".out", ".outb", ".sum", ".ech", ".lin", ".vtk", ".vtp", ".chkp", ".dbg", ".log"}
)

# A quoted string (optionally ``@``-prefixed, the AirfoilInfo convention for "read from
# this file"), or a run of characters without blanks or commas.
_TOKEN = re.compile(r"""@?"[^"]*"|@?'[^']*'|[^\s,]+""")

# Value tokens that are never file names.
_SENTINELS = frozenset({"", "unused", "default", "none", "true", "false", "-"})

# Keywords whose value is a root name shared by several files (``<root>.<ext>``).
_ROOT_KEYWORDS = frozenset({"potfile", "wvkinfile"})

_SHARED_LIBRARY_SUFFIXES = (".dll", ".so", ".dylib")
# Bitness/platform tags controller builds append to the library stem.
_SHARED_LIBRARY_TAGS = ("", "_x64", "_x86", "_win64", "_win32", "_64", "_32")

# Files read only as data: copied when referenced, never scanned for references.
_DATA_SUFFIXES = frozenset(
    {
        *_SHARED_LIBRARY_SUFFIXES,
        ".exe",
        ".bts",
        ".wnd",
        ".bin",
        ".h5",
        ".hdf5",
        ".nc",
        ".npz",
        ".npy",
        ".zip",
        ".gz",
    }
)

# A referenced file larger than this is copied but not scanned (input decks are small;
# large files are wind or wave data).
_MAX_SCAN_BYTES = 16 * 1024 * 1024
_SNIFF_BYTES = 8192
_MAX_TOKEN_LENGTH = 1024


class DeckReferenceError(ValueError):
    """Raised when the referenced-file closure of a deck cannot be trusted.

    The walk raises it for a relative reference that resolves outside
    ``template_root`` (the case copy could not reproduce it) or for an input file that
    cannot be read. :func:`copy_template_files` catches it and falls back to copying
    the whole template tree.
    """


@dataclass(frozen=True)
class DeckFileClosure:
    r"""The files a deck references, found by :func:`deck_file_closure`.

    Parameters
    ----------
    root : pathlib.Path
        The template root every file in ``files`` lies under.
    files : tuple[pathlib.Path, ...]
        Absolute paths of the referenced files under ``root``, the primary deck
        included, sorted.
    external : tuple[pathlib.Path, ...]
        Absolute references to existing files outside ``root``. They are not copied;
        an absolute path keeps resolving from the case copy.
    open_paths : tuple[tuple[pathlib.Path, str], ...]
        For each referenced file (the deck included), the path string OpenFAST opens
        it by when run from the deck's directory on the deck's bare file name: the
        referencing file's directory *as OpenFAST spelled it*, joined to the reference
        but not normalized (``<deck dir>\.\..\Shared\Blade.dat``). Relative strings
        are prefixed with the deck's directory, so each is the full path the operating
        system receives. When a file is reached by several references, the longest
        spelling is kept.
    """

    root: Path
    files: tuple[Path, ...]
    external: tuple[Path, ...] = ()
    open_paths: tuple[tuple[Path, str], ...] = ()

    @property
    def relative_files(self) -> tuple[Path, ...]:
        """Return the referenced files relative to ``root``.

        Returns
        -------
        tuple[pathlib.Path, ...]
            Each file's path relative to the template root.
        """
        return tuple(path.relative_to(self.root) for path in self.files)

    @property
    def total_bytes(self) -> int:
        """Return the summed size of the referenced files in bytes.

        Returns
        -------
        int
            The total size of ``files`` on disk.
        """
        return sum(path.stat().st_size for path in self.files)


def deck_file_closure(
    fst_path: str | Path, template_root: str | Path
) -> DeckFileClosure:
    r"""Return every file under ``template_root`` that a deck references, recursively.

    Parameters
    ----------
    fst_path : str or pathlib.Path
        The primary ``.fst`` deck.
    template_root : str or pathlib.Path
        The directory the referenced files are collected from; must contain the deck.

    Returns
    -------
    DeckFileClosure
        The referenced files (the deck included) and any absolute references outside
        the root.

    Raises
    ------
    FileNotFoundError
        If ``fst_path`` is not an existing file.
    ValueError
        If ``fst_path`` is not inside ``template_root``.
    DeckReferenceError
        If a relative reference resolves to an existing file outside
        ``template_root``, or a referenced input cannot be read.

    Notes
    -----
    A value token is treated as a path when it contains ``.``, ``/`` or ``\`` and is
    not a number; root-name keywords are expanded whatever their spelling. A token is
    kept only if it names an existing file, so prose in a description or a stale
    keyword cannot add a missing file or stop the walk. Including an unneeded file is
    harmless; the walk errs on that side.
    """
    root = resolve_path(template_root)
    deck = resolve_path(fst_path)
    if not deck.is_file():
        msg = f"OpenFAST deck not found: {deck}"
        raise FileNotFoundError(msg)
    if not deck.is_relative_to(root):
        msg = f"fst_path {deck} is not inside template_root {root}"
        raise ValueError(msg)
    walker = _ClosureWalker(root, deck)
    walker.walk()
    return walker.result()


def infer_template_root(fst_path: str | Path) -> Path:
    r"""Return the smallest directory holding every file a deck references relatively.

    A deck that references shared inputs outside its own folder
    (``../5MW_Baseline/...``, ``../IEA-15-240-RWT/...``) must be copied per case from a
    directory that contains them, so that every relative reference still resolves in
    the copy and every edited sub-file lies inside it. This walks the deck's
    references (as :func:`deck_file_closure` does, without a root boundary) and
    returns the deepest common ancestor of the deck's directory and every file
    reached through a *relative* reference. References given as absolute paths keep
    working from any copy and do not widen the root.

    Parameters
    ----------
    fst_path : str or pathlib.Path
        The primary ``.fst`` deck.

    Returns
    -------
    pathlib.Path
        The inferred template root; the deck's own directory when every relative
        reference lies under it.

    Raises
    ------
    FileNotFoundError
        If ``fst_path`` is not an existing file.

    Notes
    -----
    A referenced input that cannot be read stops the walk; the files found up to
    that point still widen the root, and the reason is logged.
    """
    deck = resolve_path(fst_path)
    if not deck.is_file():
        msg = f"OpenFAST deck not found: {deck}"
        raise FileNotFoundError(msg)
    walker = _ClosureWalker(Path(deck.anchor), deck)
    try:
        walker.walk()
    except DeckReferenceError as error:
        logger.warning("Template-root inference could not walk %s: %s", deck, error)
    directories = [str(deck.parent)]
    directories.extend(str(path.parent) for path in walker.relative_targets())
    try:
        common = Path(os.path.commonpath(directories))
    except ValueError:  # references on another drive cannot be relative; ignore
        return deck.parent
    return normalize_extended_path(common)


def copy_template_files(
    fst_path: str | Path,
    template_root: str | Path,
    destination: str | Path,
    *,
    mode: TemplateCopyMode = "referenced",
    link_inputs: bool = True,
) -> TemplateCopyMode:
    """Copy a template case into ``destination``.

    Parameters
    ----------
    fst_path : str or pathlib.Path
        The primary ``.fst`` deck inside ``template_root``.
    template_root : str or pathlib.Path
        The directory copied from; file paths relative to it are preserved.
    destination : str or pathlib.Path
        The case directory to create. It must not exist yet.
    mode : {"referenced", "tree"}, optional
        ``"referenced"`` (the default) copies only the deck's referenced files
        (:func:`deck_file_closure`); ``"tree"`` copies the whole template tree.
    link_inputs : bool, optional
        When ``True`` (the default), input files are hard-linked rather than copied,
        so heavy read-only inputs share an inode with the template; OpenFAST output
        files (``.out``, ``.lin``, ...) are always copied. Linking falls back to a
        byte copy where it is unsupported (e.g. across filesystems).

    Returns
    -------
    {"referenced", "tree"}
        The copy that was made: ``"tree"`` when requested, or when the referenced-file
        walk could not vouch for completeness and fell back (logged as a warning).

    Raises
    ------
    ValueError
        If ``mode`` is not one of the supported copy modes.
    FileExistsError
        If ``destination`` already exists.
    """
    if mode not in _COPY_MODES:
        msg = f"template copy mode must be one of {_COPY_MODES}, got {mode!r}"
        raise ValueError(msg)
    source = resolve_path(template_root)
    target = Path(destination)
    if target.exists():
        msg = f"template copy destination already exists: {target}"
        raise FileExistsError(msg)
    if mode == "referenced":
        try:
            closure = deck_file_closure(fst_path, source)
        except DeckReferenceError as error:
            logger.warning(
                "Copying the whole template tree %s instead of the deck's referenced "
                "files: %s",
                source,
                error,
            )
        else:
            _copy_closure(closure, target, link_inputs=link_inputs)
            logger.info(
                "Copied %d referenced template files (%.1f MB) to %s",
                len(closure.files),
                closure.total_bytes / 1e6,
                target,
            )
            return "referenced"
    _copy_tree(source, target, link_inputs=link_inputs)
    return "tree"


def _copy_closure(closure: DeckFileClosure, target: Path, *, link_inputs: bool) -> None:
    """Copy the closure's files under ``target`` at their root-relative paths."""
    target.mkdir(parents=True)
    link = link_inputs
    for source in closure.files:
        destination = target / source.relative_to(closure.root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if link and source.suffix.lower() not in OUTPUT_SUFFIXES:
            try:
                link_with_retry(source, destination)
            except OSError as error:
                logger.info(
                    "Hard-linking template inputs failed (%s); copying instead", error
                )
                link = False
            else:
                continue
        copy2_with_retry(source, destination)


def _link_or_copy(source: str, destination: str) -> None:
    """Copy-function for copytree: hard-link inputs, copy files OpenFAST may rewrite."""
    if Path(source).suffix.lower() in OUTPUT_SUFFIXES:
        copy2_with_retry(source, destination)
    else:
        link_with_retry(source, destination)


def _copy_tree(source: Path, target: Path, *, link_inputs: bool) -> None:
    """Copy the whole template tree, hard-linking inputs when possible."""
    if link_inputs:
        try:
            shutil.copytree(source, target, copy_function=_link_or_copy)
        except OSError as error:
            logger.info(
                "Hard-linking template inputs failed (%s); copying instead", error
            )
            if target.exists():
                rmtree_with_retry(target)
        else:
            return
    shutil.copytree(source, target, copy_function=copy2_with_retry)


class _ClosureWalker:
    """Breadth-first walk over a deck's file references."""

    def __init__(self, root: Path, deck: Path) -> None:
        self.root = root
        self.deck = deck
        self._root_key = _key(root)
        self._files: dict[str, Path] = {}
        self._external: dict[str, Path] = {}
        self._escapes: list[str] = []
        self._queue: list[Path] = []
        self._listings: dict[str, list[Path]] = {}
        # OpenFAST opens the deck by its bare name from the deck's directory, so the
        # directory it prefixes to the deck's own references is ``.\`` (GetPath).
        working_dir = plain_path_text(deck.parent)
        self._deck_open_dir = f"{working_dir}{os.sep}.{os.sep}"
        self._open_paths: dict[str, tuple[Path, str]] = {}
        self._open_dirs: dict[str, str] = {}
        # Files a case copy must carry at the same relative position: the deck and
        # every file reached from one of them through a relative reference. A file
        # referenced by an absolute path stays where it is, and so do its own
        # relative references. See infer_template_root.
        self._relative: dict[str, Path] = {_key(deck): deck}

    def walk(self) -> None:
        """Visit the deck and every file it references, recursively."""
        deck_dir = plain_path_text(self.deck.parent)
        self._record_open(self.deck, f"{deck_dir}{os.sep}{self.deck.name}")
        self._open_dirs[_key(self.deck)] = self._deck_open_dir
        self._add(self.deck, scan=True)
        while self._queue:
            current = self._queue.pop()
            self._scan(current)
        if self._escapes:
            listed = "; ".join(self._escapes[:5])
            more = len(self._escapes) - 5
            suffix = f" (and {more} more)" if more > 0 else ""
            msg = (
                "relative references resolve outside template_root "
                f"{self.root}: {listed}{suffix}"
            )
            raise DeckReferenceError(msg)

    def result(self) -> DeckFileClosure:
        """Return the collected closure."""
        return DeckFileClosure(
            root=self.root,
            files=tuple(sorted(self._files.values(), key=_key)),
            external=tuple(sorted(self._external.values(), key=_key)),
            open_paths=tuple(self._open_paths[key] for key in sorted(self._open_paths)),
        )

    def relative_targets(self) -> tuple[Path, ...]:
        """Return the files (or library paths) reached by relative references."""
        return tuple(self._relative[key] for key in sorted(self._relative))

    def _record_open(self, path: Path, opened: str) -> None:
        """Keep the longest spelling OpenFAST opens ``path`` by."""
        key = _key(path)
        previous = self._open_paths.get(key)
        if previous is None or len(opened) > len(previous[1]):
            self._open_paths[key] = (path, opened)
        # The directory a scanned file prefixes to its own references is taken from
        # the first spelling it was reached by, as OpenFAST reads each file once.
        if key not in self._open_dirs:
            self._open_dirs[key] = _directory_part(opened)

    def _open_dir(self, source: Path) -> str:
        """Return the directory OpenFAST prefixes to ``source``'s references."""
        return self._open_dirs.get(_key(source), self._deck_open_dir)

    def _inside(self, path: Path) -> bool:
        """Return whether ``path`` lies strictly under the template root."""
        return _key(path).startswith(self._root_key.rstrip(os.sep) + os.sep)

    def _inside_or_root(self, directory: Path) -> bool:
        """Return whether ``directory`` is the template root or lies under it."""
        return _key(directory) == self._root_key or self._inside(directory)

    def _list_files(self, directory: Path) -> list[Path]:
        """Return the files directly in ``directory`` (cached per walk)."""
        key = _key(directory)
        cached = self._listings.get(key)
        if cached is None:
            cached = _list_files(directory)
            self._listings[key] = cached
        return cached

    def _add(self, path: Path, *, scan: bool) -> None:
        key = _key(path)
        if key in self._files:
            return
        self._files[key] = path
        if scan and _scannable(path):
            self._queue.append(path)

    def _scan(self, path: Path) -> None:
        try:
            with path.open("rb") as handle:
                data = handle.read(_MAX_SCAN_BYTES + 1)
        except OSError as error:
            msg = f"cannot read referenced input {path}: {error}"
            raise DeckReferenceError(msg) from error
        if len(data) > _MAX_SCAN_BYTES or b"\0" in data[:_SNIFF_BYTES]:
            return
        text = data.decode("utf-8", errors="replace")
        for line in text.splitlines():
            self._scan_line(line, path)

    def _scan_line(self, line: str, source: Path) -> None:
        tokens = [match.group(0) for match in _TOKEN.finditer(line)]
        if not tokens:
            return
        root_index = _root_keyword_index(tokens)
        for index, token in enumerate(tokens):
            value = _unquote(token)
            if root_index is not None and index < root_index:
                if _is_candidate_value(value):
                    self._resolve_root(value, source)
                continue
            if _looks_like_path(value):
                self._resolve_file(value, source)

    def _bases(self, source: Path) -> tuple[Path, ...]:
        if source.parent == self.deck.parent:
            return (source.parent,)
        return (source.parent, self.deck.parent)

    def _candidates(self, value: str, source: Path) -> list[tuple[Path, bool, str]]:
        """Return ``(path, relative, opened)`` resolutions of a reference.

        ``opened`` is the unnormalized string OpenFAST opens the file by: the
        referencing file's directory as OpenFAST spelled it, joined to ``value``.
        """
        text = value.replace("\\", "/") if os.sep == "/" else value
        if _is_absolute(text):
            # Joining keeps a full path as it is and puts a drive-less rooted path
            # on the deck's drive, where OpenFAST (run from the deck's directory)
            # opens it.
            return [(_normalize(self.deck.parent / text), False, value)]
        # Source last: when both bases coincide, the source's spelling wins.
        open_dirs = {
            _key(self.deck.parent): self._deck_open_dir,
            _key(source.parent): self._open_dir(source),
        }
        return [
            (_normalize(base / text), True, open_dirs[_key(base)] + value)
            for base in self._bases(source)
        ]

    def _resolve_file(self, value: str, source: Path) -> None:
        found_inside = False
        escaped: Path | None = None
        opened_any = False
        for candidate, relative, opened in self._candidates(value, source):
            if candidate.suffix.lower() in _SHARED_LIBRARY_SUFFIXES:
                variants = self._add_library_variants(candidate)
                found_inside |= variants
                if variants and relative and _key(source) in self._relative:
                    self._relative.setdefault(_key(candidate), candidate)
            if not _is_file(candidate):
                continue
            if relative and _key(source) in self._relative:
                self._relative.setdefault(_key(candidate), candidate)
            if not opened_any:
                # OpenFAST opens a reference relative to the referencing file first;
                # the working-directory spelling counts only when that one is absent.
                self._record_open(candidate, opened)
                opened_any = True
            if self._inside(candidate):
                self._add(candidate, scan=True)
                found_inside = True
            elif relative:
                escaped = escaped or candidate
            else:
                self._external.setdefault(_key(candidate), candidate)
        if escaped is not None and not found_inside:
            self._escapes.append(f"{value!r} in {source} -> {escaped}")

    def _add_library_variants(self, candidate: Path) -> bool:
        """Add the platform/bitness variants of a shared library; return if any."""
        directory = candidate.parent
        if not self._inside_or_root(directory) or not directory.is_dir():
            return False
        stem = candidate.stem.lower()
        wanted = {
            f"{stem}{tag}{suffix}"
            for tag in _SHARED_LIBRARY_TAGS
            for suffix in _SHARED_LIBRARY_SUFFIXES
        }
        added = False
        for entry in self._list_files(directory):
            if entry.name.lower() in wanted:
                self._add(entry, scan=False)
                added = True
        return added

    def _resolve_root(self, value: str, source: Path) -> None:
        found_inside = False
        escaped: Path | None = None
        opened_any = False
        for candidate, relative, opened in self._candidates(value, source):
            directory = candidate.parent
            if not directory.is_dir():
                continue
            prefix = candidate.name.lower() + "."
            matches = [
                entry
                for entry in self._list_files(directory)
                if entry.name.lower().startswith(prefix)
                and entry.suffix.lower() not in OUTPUT_SUFFIXES
            ]
            if _is_file(candidate):
                matches.append(candidate)
            if not matches:
                continue
            if not opened_any:
                for match in matches:
                    self._record_open(match, opened + match.name[len(candidate.name) :])
                opened_any = True
            if relative and _key(source) in self._relative:
                for match in matches:
                    self._relative.setdefault(_key(match), match)
            if self._inside(candidate):
                for match in matches:
                    self._add(match, scan=False)
                found_inside = True
            elif relative:
                escaped = escaped or candidate
            else:
                for match in matches:
                    self._external.setdefault(_key(match), match)
        if escaped is not None and not found_inside:
            self._escapes.append(f"root {value!r} in {source} -> {escaped}")


def _directory_part(opened: str) -> str:
    r"""Return ``opened`` up to and including its last ``/`` or ``\`` (GetPath)."""
    cut = max(opened.rfind("/"), opened.rfind("\\"))
    return opened[: cut + 1] if cut >= 0 else f".{os.sep}"


def _key(path: Path) -> str:
    """Return a comparison key for a path (case-folded where the OS folds case)."""
    return os.path.normcase(str(path))


def _normalize(path: Path) -> Path:
    r"""Collapse ``..`` lexically, keeping a needed ``\\?\`` prefix intact."""
    return normalize_extended_path(Path(os.path.normpath(path)))


def _is_absolute(text: str) -> bool:
    r"""Return whether a reference is absolute on this platform.

    On Windows a drive-less rooted path (``\Shared\File.dat``) names the same file
    from any directory on the drive, so it is absolute too (Python 3.13's
    ``ntpath.isabs`` no longer says so).
    """
    if os.name == "nt":
        return ntpath.isabs(text) or text.startswith(("\\", "/"))
    return text.startswith("/")


def _is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except (OSError, ValueError):
        return False


def _list_files(directory: Path) -> list[Path]:
    try:
        return [entry for entry in directory.iterdir() if entry.is_file()]
    except OSError:
        return []


def _scannable(path: Path) -> bool:
    """Return whether a referenced file may hold further references."""
    return path.suffix.lower() not in _DATA_SUFFIXES


def _unquote(token: str) -> str:
    """Strip the AirfoilInfo ``@`` marker and surrounding quotes from a token."""
    value = token.removeprefix("@")
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return value.strip()


def _is_number(value: str) -> bool:
    try:
        fortran_float(value)
    except ValueError:
        return False
    return True


def _is_candidate_value(value: str) -> bool:
    """Return whether a token could name a file at all."""
    return (
        value.lower() not in _SENTINELS
        and len(value) <= _MAX_TOKEN_LENGTH
        and not _is_number(value)
        and "\0" not in value
    )


def _looks_like_path(value: str) -> bool:
    """Return whether a token looks like a file reference."""
    return _is_candidate_value(value) and any(mark in value for mark in "./\\")


def _root_keyword_index(tokens: list[str]) -> int | None:
    """Return the index of a root-name keyword on the line, or ``None``.

    Only the line's head (values and keyword, before a ``-`` or ``!`` description) is
    searched, and the keyword must follow at least one value, so prose such as "at the
    blade root" in a description is never taken for a keyword.
    """
    for index, token in enumerate(tokens):
        if token == "-" or token.startswith("!"):
            return None
        if index == 0:
            continue
        name = token.lower()
        if name in _ROOT_KEYWORDS or (
            name.endswith("root") and name != "root" and name.isidentifier()
        ):
            return index
    return None
