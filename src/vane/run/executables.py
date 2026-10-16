# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Match an OpenFAST deck to an OpenFAST executable that can read it.

OpenFAST changes its input-file format between releases, and a deck written for one
release is rejected by another: a v4 primary file fails on v5 at ``ConvTol`` (v5 added
the tight-coupling lines), and a v4.0 deck fails on v4.2 (no ElastoDyn
``HubIner_Teeter``) and on v4.1 if it has HydroDyn members (the old ``NPropSets``
table). This module

* recognises a deck's input-format generation from the keywords its files carry
  (:func:`detect_deck_generation`);
* reads an executable's release from its version banner (``openfast -v``,
  :func:`openfast_version`), cached per executable file;
* chooses a compatible executable from the explicit argument, the
  ``VANE_OPENFAST_EXECUTABLES`` environment variable, a user configuration file and
  ``PATH`` (:func:`select_openfast_executable`), and fails before any case is copied
  when none can read the deck, naming the release the deck needs.

The keyword signatures follow the OpenFAST API-change notes for each release.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

from vane.io.fst_reader import fortran_float
from vane.io.fst_writer import FastInputError, FastInputFile
from vane.run.runner import OpenFastError, find_openfast

__all__ = [
    "OPENFAST_CONFIG_ENV",
    "OPENFAST_EXECUTABLES_ENV",
    "DeckGeneration",
    "OpenFastSelection",
    "candidate_executables",
    "detect_deck_generation",
    "openfast_version",
    "select_openfast_executable",
]

logger = logging.getLogger(__name__)

#: Environment variable listing OpenFAST executables, separated by ``os.pathsep``.
OPENFAST_EXECUTABLES_ENV = "VANE_OPENFAST_EXECUTABLES"
#: Environment variable naming a configuration file of OpenFAST executables.
OPENFAST_CONFIG_ENV = "VANE_OPENFAST_CONFIG"
# Default configuration file: one executable path per line, ``#`` starts a comment.
_DEFAULT_CONFIG = Path("~/.vane/openfast_executables.txt")
# Executable names looked up on PATH, in order.
_PATH_NAMES = ("openfast", "openfast_x64", "OpenFAST")
# The banner line ``OpenFAST-v4.2.1`` (a development build may append ``-<n>-g<sha>``).
_VERSION_RE = re.compile(r"OpenFAST-v(\d+)\.(\d+)\.(\d+)", re.IGNORECASE)
# Seconds allowed for ``openfast -v``; the banner prints in well under a second.
_PROBE_TIMEOUT = 60.0

Version = tuple[int, int, int]

# Primary-file keywords first written by OpenFAST v5.0 (tight coupling, multi-rotor).
_V5_PRIMARY_KEYS = (
    "ModCoupling",
    "RhoInf",
    "ConvTol",
    "MaxConvIter",
    "NRotors",
    "MirrorRotor",
)
# ElastoDyn keywords first written by v5.0 (blade-pitch DOF and pitch inertias).
_V5_ELASTODYN_KEYS = ("PitchDOF", "BlPIner(1)", "PBrIner(1)")

_version_cache: dict[tuple[str, int, int], Version | None] = {}
_cache_lock = threading.Lock()


@dataclass(frozen=True)
class DeckGeneration:
    """The OpenFAST release range whose input format a deck is written in.

    Parameters
    ----------
    label : str
        A short name of the format generation, e.g. ``"v5"``, ``"v4.2"`` or
        ``"v4.0-v4.1"``.
    min_version : tuple[int, int, int]
        The oldest release that reads the deck (inclusive).
    max_version : tuple[int, int, int] or None
        The first release that no longer reads it (exclusive); ``None`` when no
        upper bound is known.
    evidence : tuple[str, ...]
        The keyword findings the range was inferred from, for the log and the run
        provenance.
    """

    label: str
    min_version: Version
    max_version: Version | None
    evidence: tuple[str, ...] = ()

    def accepts(self, version: Version) -> bool:
        """Return whether an executable of ``version`` reads this deck's format.

        Parameters
        ----------
        version : tuple[int, int, int]
            The executable's release.

        Returns
        -------
        bool
            ``True`` when ``min_version <= version < max_version``.
        """
        if version < self.min_version:
            return False
        return self.max_version is None or version < self.max_version

    def describe(self) -> str:
        """Return the accepted release range as text, e.g. ``>=4.2.0, <5.0.0``.

        Returns
        -------
        str
            The range in requirement notation.
        """
        low = ".".join(str(part) for part in self.min_version)
        if self.max_version is None:
            return f">={low}"
        high = ".".join(str(part) for part in self.max_version)
        return f">={low}, <{high}"

    def to_dict(self) -> dict[str, object]:
        """Return the generation as a JSON-serializable dictionary.

        Returns
        -------
        dict[str, object]
            ``label``, ``requires`` (the release range) and ``evidence``.
        """
        return {
            "label": self.label,
            "requires": self.describe(),
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class OpenFastSelection:
    """The OpenFAST executable chosen for a deck, with what was considered.

    Parameters
    ----------
    executable : pathlib.Path
        The executable to run.
    version : tuple[int, int, int] or None
        Its release, or ``None`` when its banner could not be read.
    generation : DeckGeneration
        The deck's input-format generation.
    source : str
        Where the executable came from: ``"argument"``, the environment variable,
        the configuration file, or ``"PATH"``.
    considered : tuple[tuple[str, str], ...]
        Every candidate examined, as ``(path, version or "unknown")`` pairs.
    """

    executable: Path
    version: Version | None
    generation: DeckGeneration
    source: str
    considered: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        """Return the selection as a JSON-serializable dictionary.

        Returns
        -------
        dict[str, object]
            The executable, its version, the source, the deck generation and the
            candidates considered.
        """
        return {
            "executable": str(self.executable),
            "version": _version_text(self.version),
            "source": self.source,
            "deck_generation": self.generation.to_dict(),
            "considered": [list(pair) for pair in self.considered],
        }


def _version_text(version: Version | None) -> str:
    """Return ``version`` as ``"4.2.1"`` or ``"unknown"``."""
    return "unknown" if version is None else ".".join(str(part) for part in version)


def _has(file: FastInputFile | None, keyword: str) -> bool:
    """Return whether ``file`` has a ``keyword`` line (``False`` without a file)."""
    return file is not None and keyword in file


def _subfile(deck: FastInputFile, key: str, switch: str | None) -> FastInputFile | None:
    """Return the readable sub-file ``key`` of an enabled module, or ``None``."""
    if switch is not None and switch in deck:
        try:
            if fortran_float(deck.get(switch)) == 0.0:
                return None
        except (FastInputError, ValueError):
            pass
    if key not in deck:
        return None
    try:
        path = deck.resolve(key)
        return FastInputFile.read(path) if path.is_file() else None
    except FastInputError:
        return None


def detect_deck_generation(fst_path: str | Path) -> DeckGeneration:
    """Return the OpenFAST release range whose input format a deck uses.

    The decision reads keyword signatures the releases introduced or removed:

    * **v5** (``>=5.0``): the primary file has the tight-coupling or multi-rotor
      lines (``ModCoupling``, ``RhoInf``, ``ConvTol``, ``MaxConvIter``, ``NRotors``,
      ``MirrorRotor``), or ElastoDyn has ``PitchDOF``/``BlPIner(1)``/``PBrIner(1)``.
    * **unknown** (any release): the primary file has no ``CompHydro`` line or no
      readable ElastoDyn file, so it carries no format signal.
    * **v3** (``<4.0``): the primary file has ``CompHydro`` but no ``CompSeaSt``,
      which v4.0 added.
    * **v4.2** (``>=4.2, <5.0``): ElastoDyn has ``HubIner_Teeter``.
    * **v4.1** (``>=4.1, <4.2``): HydroDyn has the cylindrical/rectangular member
      tables (``NPropSetsCyl``) but ElastoDyn lacks ``HubIner_Teeter``.
    * **v4.0** (``>=4.0, <4.1``): HydroDyn has the single member table
      (``NPropSets``).
    * otherwise **v4.0-v4.1** (``>=4.0, <4.2``): a v4 deck without HydroDyn members
      and without ``HubIner_Teeter``.

    Parameters
    ----------
    fst_path : str or pathlib.Path
        The primary ``.fst`` deck.

    Returns
    -------
    DeckGeneration
        The release range and the keyword evidence for it.

    Raises
    ------
    vane.io.FastInputError
        If the primary deck cannot be read.
    """
    deck = FastInputFile.read(fst_path)
    elastodyn = _subfile(deck, "EDFile", None)
    hydrodyn = _subfile(deck, "HydroFile", "CompHydro")
    v5_keys = [key for key in _V5_PRIMARY_KEYS if key in deck]
    v5_keys += [
        f"ElastoDyn {key}" for key in _V5_ELASTODYN_KEYS if _has(elastodyn, key)
    ]
    if v5_keys:
        return DeckGeneration(
            "v5", (5, 0, 0), (6, 0, 0), (f"v5 keywords: {', '.join(v5_keys)}",)
        )
    if "CompHydro" not in deck or elastodyn is None:
        # Not a complete primary file (or no readable ElastoDyn): no format signal.
        return DeckGeneration(
            "unknown",
            (0, 0, 0),
            None,
            ("no CompHydro line or readable ElastoDyn file to identify the format",),
        )
    if "CompSeaSt" not in deck:
        return DeckGeneration(
            "v3", (0, 0, 0), (4, 0, 0), ("no CompSeaSt line (added in v4.0)",)
        )
    if _has(elastodyn, "HubIner_Teeter"):
        return DeckGeneration(
            "v4.2", (4, 2, 0), (5, 0, 0), ("ElastoDyn HubIner_Teeter (added in v4.2)",)
        )
    no_teeter = "ElastoDyn has no HubIner_Teeter (added in v4.2)"
    if _has(hydrodyn, "NPropSetsCyl"):
        return DeckGeneration(
            "v4.1",
            (4, 1, 0),
            (4, 2, 0),
            (no_teeter, "HydroDyn NPropSetsCyl (added in v4.1)"),
        )
    if _has(hydrodyn, "NPropSets"):
        return DeckGeneration(
            "v4.0",
            (4, 0, 0),
            (4, 1, 0),
            (no_teeter, "HydroDyn NPropSets (replaced in v4.1)"),
        )
    return DeckGeneration("v4.0-v4.1", (4, 0, 0), (4, 2, 0), (no_teeter,))


def openfast_version(executable: str | Path) -> Version | None:
    """Return an OpenFAST executable's release from its version banner.

    Runs ``<executable> -v`` once and parses ``OpenFAST-v<major>.<minor>.<patch>``.
    The result is cached per executable file (path, modification time and size), so
    repeated selections cost nothing.

    Parameters
    ----------
    executable : str or pathlib.Path
        The executable's path or a name on ``PATH``.

    Returns
    -------
    tuple[int, int, int] or None
        The release, or ``None`` when the executable cannot be found or run, or its
        output carries no version banner.
    """
    try:
        path = find_openfast(executable)
        stat = path.stat()
    except (OpenFastError, OSError):
        return None
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    with _cache_lock:
        if key in _version_cache:
            return _version_cache[key]
    version = _probe_version(path)
    with _cache_lock:
        _version_cache[key] = version
    return version


def _probe_version(path: Path) -> Version | None:
    """Run ``path -v`` and parse its banner (``None`` on any failure)."""
    try:
        completed = subprocess.run(
            [str(path), "-v"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_PROBE_TIMEOUT,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        logger.warning("Could not run %s -v to read its version: %s", path, error)
        return None
    match = _VERSION_RE.search(completed.stdout + completed.stderr)
    if match is None:
        logger.warning("%s -v printed no OpenFAST version banner", path)
        return None
    major, minor, patch = (int(group) for group in match.groups())
    return (major, minor, patch)


def _config_file() -> Path:
    """Return the configuration file listing OpenFAST executables."""
    configured = os.environ.get(OPENFAST_CONFIG_ENV, "").strip()
    return Path(configured).expanduser() if configured else _DEFAULT_CONFIG.expanduser()


def _expand_home(entry: str) -> str:
    """Return a listed executable path with a leading ``~`` expanded."""
    return str(Path(entry).expanduser()) if entry.startswith("~") else entry


def _config_entries(path: Path) -> list[str]:
    """Return the executable paths listed in a configuration file."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    entries = []
    for line in text.splitlines():
        entry = _expand_home(line.split("#", 1)[0].strip().strip('"'))
        if entry:
            entries.append(entry)
    return entries


def candidate_executables(
    executable: str | Path | None = None,
) -> list[tuple[str, str]]:
    """Return the OpenFAST executables to consider, in order of preference.

    The order is: the explicit ``executable`` argument; the entries of the
    ``VANE_OPENFAST_EXECUTABLES`` environment variable (separated by
    ``os.pathsep``); the lines of the configuration file named by
    ``VANE_OPENFAST_CONFIG`` (default ``~/.vane/openfast_executables.txt``; one path
    per line, ``#`` comments); and ``openfast``, ``openfast_x64`` and ``OpenFAST`` on
    ``PATH``. A leading ``~`` in a listed path (or in ``VANE_OPENFAST_CONFIG``) is
    the user's home directory. Duplicates (the same resolved file) are dropped;
    entries that do not exist are kept, so the caller can report them.

    Parameters
    ----------
    executable : str or pathlib.Path or None, optional
        An explicitly requested executable, listed first.

    Returns
    -------
    list[tuple[str, str]]
        ``(executable, source)`` pairs, where ``source`` says where it came from.
    """
    entries: list[tuple[str, str]] = []
    if executable is not None:
        entries.append((str(executable), "argument"))
    listed = os.environ.get(OPENFAST_EXECUTABLES_ENV, "")
    entries.extend(
        (_expand_home(item.strip().strip('"')), OPENFAST_EXECUTABLES_ENV)
        for item in listed.split(os.pathsep)
        if item.strip()
    )
    config = _config_file()
    entries.extend((item, str(config)) for item in _config_entries(config))
    entries.extend((name, "PATH") for name in _PATH_NAMES)
    unique: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item, source in entries:
        try:
            key = os.path.normcase(str(find_openfast(item)))
        except OpenFastError:
            if source == "PATH":
                continue  # a name simply not installed
            key = os.path.normcase(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append((item, source))
    return unique


def select_openfast_executable(
    fst_path: str | Path, executable: str | Path | None = None
) -> OpenFastSelection:
    """Choose an OpenFAST executable that can read a deck's input format.

    The deck's generation is detected (:func:`detect_deck_generation`) and each
    candidate (:func:`candidate_executables`) is probed for its release
    (:func:`openfast_version`).

    * An explicit ``executable`` is always the one used when it can read the deck or
      its version cannot be read (logged); if its version is known and outside the
      deck's range, the selection fails rather than silently running another one.
    * Otherwise the first candidate whose release the deck accepts is used; failing
      that, the first candidate whose version could not be read (with a warning).

    Parameters
    ----------
    fst_path : str or pathlib.Path
        The primary ``.fst`` deck.
    executable : str or pathlib.Path or None, optional
        An explicitly requested executable. ``None`` (the default) chooses among
        the configured ones.

    Returns
    -------
    OpenFastSelection
        The executable, its version, and the evidence of the choice.

    Raises
    ------
    vane.run.OpenFastError
        If no candidate can read the deck: the message names the release range the
        deck needs, the keyword evidence, and every executable considered with its
        version. Raised before any case is copied or launched.
    """
    generation = detect_deck_generation(fst_path)
    considered: list[tuple[str, str]] = []
    unknown: tuple[Path, str] | None = None
    for item, source in candidate_executables(executable):
        try:
            path = find_openfast(item)
        except OpenFastError:
            considered.append((item, "not found"))
            if source == "argument":
                msg = (
                    f"OpenFAST executable {item!r} (the executable argument) not found"
                )
                raise OpenFastError(msg) from None
            continue
        version = openfast_version(path)
        considered.append((str(path), _version_text(version)))
        if version is not None and generation.accepts(version):
            return _selected(path, version, generation, source, considered)
        if source == "argument":
            if version is None:
                logger.warning(
                    "Using %s without knowing its version (no banner); the deck needs "
                    "OpenFAST %s (%s)",
                    path,
                    generation.describe(),
                    generation.label,
                )
                return _selected(path, None, generation, source, considered)
            raise OpenFastError(_incompatible_message(fst_path, generation, considered))
        if version is None and unknown is None:
            unknown = (path, source)
    if unknown is not None:
        logger.warning(
            "No OpenFAST executable with a known compatible version was found; using "
            "%s, whose version could not be read (the deck needs OpenFAST %s)",
            unknown[0],
            generation.describe(),
        )
        return _selected(unknown[0], None, generation, unknown[1], considered)
    raise OpenFastError(_incompatible_message(fst_path, generation, considered))


def _selected(
    path: Path,
    version: Version | None,
    generation: DeckGeneration,
    source: str,
    considered: list[tuple[str, str]],
) -> OpenFastSelection:
    """Log and return a selection."""
    logger.info(
        "OpenFAST executable %s (version %s, from %s) selected for a %s deck (%s)",
        path,
        _version_text(version),
        source,
        generation.label,
        "; ".join(generation.evidence),
    )
    return OpenFastSelection(
        executable=path,
        version=version,
        generation=generation,
        source=source,
        considered=tuple(considered),
    )


def _incompatible_message(
    fst_path: str | Path,
    generation: DeckGeneration,
    considered: list[tuple[str, str]],
) -> str:
    """Return the error naming the release the deck needs and what was found."""
    found = (
        "; ".join(f"{path} ({version})" for path, version in considered)
        if considered
        else "none"
    )
    return (
        f"no OpenFAST executable can read {fst_path}: its input files are in the "
        f"{generation.label} format ({'; '.join(generation.evidence)}), which needs "
        f"OpenFAST {generation.describe()}. Executables considered: {found}. Pass "
        "executable=<path to a compatible OpenFAST>, or list compatible executables "
        f"in the {OPENFAST_EXECUTABLES_ENV} environment variable (separated by "
        f"{os.pathsep!r}) or in the file named by {OPENFAST_CONFIG_ENV} (default "
        f"{_DEFAULT_CONFIG})"
    )
