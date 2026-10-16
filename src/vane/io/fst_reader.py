# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Readers for OpenFAST model input decks (``.fst`` and ElastoDyn).

These parsers extract only the information VANE needs to drive linearization
post-processing: the linearization configuration block from the primary ``.fst``
file (so the expected set of ``.lin`` files is known), the active module
switches and referenced sub-file paths, and the rotor/tower geometry from
ElastoDyn (used to scale mode shapes).

OpenFAST input files use the convention ``<value(s)> <Keyword> - <description>``;
the value(s) precede the keyword on each line. Quoted string values and
multi-value array lines (e.g. ``LinTimes``) are both handled.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "FastModel",
    "FstFileError",
    "LinearizationConfig",
    "TurbineGeometry",
    "fortran_float",
    "read_elastodyn_geometry",
    "read_fst_file",
]

# One field-line token: a quoted string (which may contain blanks or " - ") or a
# run of non-blank characters.
_LINE_TOKEN = re.compile(r"\"[^\"]*\"|'[^']*'|\S+")

# Module switches read from the primary .fst file.
_COMP_SWITCHES = (
    "CompElast",
    "CompInflow",
    "CompAero",
    "CompServo",
    "CompSeaSt",
    "CompHydro",
    "CompSub",
    "CompMooring",
    "CompIce",
    "CompSoil",
)

# Referenced module sub-file keywords, in primary-file order.
_SUB_FILE_KEYWORDS = (
    "EDFile",
    "BDBldFile(1)",
    "BDBldFile(2)",
    "BDBldFile(3)",
    "InflowFile",
    "AeroFile",
    "ServoFile",
    "SeaStFile",
    "HydroFile",
    "SubFile",
    "MooringFile",
    "IceFile",
    "SoilFile",
)


class FstFileError(ValueError):
    """Raised when an OpenFAST input file is malformed or missing a field."""


@dataclass
class LinearizationConfig:
    """Linearization settings parsed from a primary ``.fst`` file.

    Parameters
    ----------
    linearize : bool
        Master linearization switch (``Linearize``).
    calc_steady : bool
        Whether a periodic steady-state trim precedes linearization
        (``CalcSteady``); when ``True``, ``n_lin_times`` azimuth snapshots are
        produced per operating point.
    trim_case : int
        Trim target ``{1: yaw, 2: torque, 3: pitch}`` (``TrimCase``).
    trim_tol : float
        Rotor-speed convergence tolerance (``TrimTol``).
    trim_gain : float
        Proportional trim gain (``TrimGain``).
    n_lin_times : int
        Number of linearization snapshots (``NLinTimes``); the number of ``.lin``
        files written per operating point.
    lin_times : list[float]
        Absolute linearization times in seconds (``LinTimes``); meaningful only
        when ``calc_steady`` is ``False``.
    lin_inputs : int
        Inputs included ``{0: none, 1: standard, 2: all}`` (``LinInputs``).
    lin_outputs : int
        Outputs included ``{0: none, 1: from OutList, 2: all}`` (``LinOutputs``).
    lin_out_jac : bool
        Whether full Jacobians are written (``LinOutJac``).
    lin_out_mod : bool
        Whether per-module ``.lin`` files are written (``LinOutMod``).
    """

    linearize: bool
    calc_steady: bool
    trim_case: int
    trim_tol: float
    trim_gain: float
    n_lin_times: int
    lin_times: list[float]
    lin_inputs: int
    lin_outputs: int
    lin_out_jac: bool
    lin_out_mod: bool


@dataclass
class TurbineGeometry:
    """Rotor and tower geometry parsed from an ElastoDyn input file.

    Parameters
    ----------
    num_blades : int
        Number of blades (``NumBl``).
    tip_radius : float
        Rotor apex-to-tip distance in metres (``TipRad``).
    hub_radius : float
        Rotor apex-to-blade-root distance in metres (``HubRad``).
    tower_height : float
        Tower-top height above its reference datum in metres (``TowerHt``).
    tower_base_height : float
        Tower-base height above the same datum in metres (``TowerBsHt``); ``0.0``
        when the ElastoDyn file does not define it.
    """

    num_blades: int
    tip_radius: float
    hub_radius: float
    tower_height: float
    tower_base_height: float

    @property
    def blade_length(self) -> float:
        """Return the flexible blade length ``TipRad - HubRad`` in metres."""
        return self.tip_radius - self.hub_radius

    @property
    def tower_length(self) -> float:
        """Return the flexible tower length ``TowerHt - TowerBsHt`` in metres."""
        return self.tower_height - self.tower_base_height


@dataclass
class FastModel:
    """Subset of a primary ``.fst`` model relevant to linearization.

    Parameters
    ----------
    path : pathlib.Path
        Path to the primary ``.fst`` file.
    comp : dict[str, int]
        Active-module switches keyed by name (e.g. ``"CompHydro"``).
    files : dict[str, pathlib.Path]
        Referenced sub-file paths resolved relative to ``path``'s directory,
        keyed by their keyword (e.g. ``"EDFile"``). Entries marked ``"unused"``
        in the deck are omitted.
    lin : LinearizationConfig
        Parsed linearization configuration.
    """

    path: Path
    comp: dict[str, int]
    files: dict[str, Path]
    lin: LinearizationConfig

    @property
    def ed_file(self) -> Path | None:
        """Return the resolved ElastoDyn input path, or ``None`` if unset."""
        return self.files.get("EDFile")

    @property
    def is_offshore(self) -> bool:
        """Return whether the model includes sea-state or hydrodynamic modules."""
        return self.comp.get("CompSeaSt", 0) >= 1 or self.comp.get("CompHydro", 0) >= 1

    @property
    def is_floating(self) -> bool:
        """Return a heuristic for a floating platform (has mooring, no SubDyn).

        Notes
        -----
        This is an approximation based on module switches: floating platforms use
        a mooring system (``CompMooring >= 1``) and no SubDyn substructure
        (``CompSub == 0``). Detailed system classification (spar/semi/TLP) is the
        responsibility of the configuration layer, not this reader.
        """
        return self.comp.get("CompMooring", 0) >= 1 and self.comp.get("CompSub", 0) == 0


def read_fst_file(path: Path) -> FastModel:
    """Read a primary OpenFAST ``.fst`` file.

    Parameters
    ----------
    path : pathlib.Path
        Path to the ``.fst`` file.

    Returns
    -------
    FastModel
        Parsed module switches, referenced file paths, and linearization config.

    Raises
    ------
    FstFileError
        If the file cannot be read. Missing module switches and linearization
        settings are tolerated and fall back to their defaults.
    """
    lines = _read_lines(path)
    # Module switches vary across OpenFAST versions and deck types (a newer deck
    # may add MHK or drop CompSoil), so every switch is optional and defaults to 0.
    comp = {name: _get_int(lines, name, path, default=0) for name in _COMP_SWITCHES}

    files: dict[str, Path] = {}
    for keyword in _SUB_FILE_KEYWORDS:
        tokens = _field_tokens(lines, keyword)
        if tokens is None:
            continue
        # A quoted path may contain spaces, so it can span several tokens; rejoin
        # them before stripping the surrounding quotes.
        value = " ".join(tokens).strip().strip("\"'")
        if value and value.lower() != "unused":
            files[keyword] = path.parent / value

    lin = _parse_linearization(lines, path)
    return FastModel(path=path, comp=comp, files=files, lin=lin)


def read_elastodyn_geometry(path: Path) -> TurbineGeometry:
    """Read rotor and tower geometry from an ElastoDyn input file.

    Parameters
    ----------
    path : pathlib.Path
        Path to the ElastoDyn ``.dat`` file.

    Returns
    -------
    TurbineGeometry
        The parsed rotor/tower geometry.

    Raises
    ------
    FstFileError
        If the file cannot be read or a required field is missing.

    Notes
    -----
    ``NumBl``, ``TipRad``, ``HubRad``, and ``TowerHt`` are required. ``TowerBsHt``
    is optional and defaults to ``0.0`` (tower base at the datum) when absent, as
    in ElastoDyn files that omit it.
    """
    lines = _read_lines(path)
    return TurbineGeometry(
        num_blades=_get_int(lines, "NumBl", path),
        tip_radius=_get_float(lines, "TipRad", path),
        hub_radius=_get_float(lines, "HubRad", path),
        tower_height=_get_float(lines, "TowerHt", path),
        tower_base_height=_get_float(lines, "TowerBsHt", path, default=0.0),
    )


def _parse_linearization(lines: list[str], path: Path) -> LinearizationConfig:
    """Build a :class:`LinearizationConfig` from primary-file lines.

    Every field is optional: a deck not configured for linearization (or written
    by a version that omits some settings) yields the natural defaults rather than
    a parse error.
    """
    return LinearizationConfig(
        linearize=_get_bool(lines, "Linearize", path, default=False),
        calc_steady=_get_bool(lines, "CalcSteady", path, default=False),
        trim_case=_get_int(lines, "TrimCase", path, default=0),
        trim_tol=_get_float(lines, "TrimTol", path, default=0.0),
        trim_gain=_get_float(lines, "TrimGain", path, default=0.0),
        n_lin_times=_get_int(lines, "NLinTimes", path, default=0),
        lin_times=_parse_lin_times(_field_tokens(lines, "LinTimes") or [], path),
        lin_inputs=_get_int(lines, "LinInputs", path, default=0),
        lin_outputs=_get_int(lines, "LinOutputs", path, default=0),
        lin_out_jac=_get_bool(lines, "LinOutJac", path, default=False),
        lin_out_mod=_get_bool(lines, "LinOutMod", path, default=False),
    )


def _parse_lin_times(tokens: list[str], path: Path) -> list[float]:
    """Parse a ``LinTimes`` value list into floats.

    Accepts space-separated (``30 60 90``), spaced-comma (``30, 60, 90``), and
    compact comma-only (``30,60,90``) forms by normalising commas to whitespace
    before splitting.

    Parameters
    ----------
    tokens : list[str]
        The raw value tokens preceding the ``LinTimes`` keyword.
    path : pathlib.Path
        The source file, used for error context.

    Returns
    -------
    list[float]
        The parsed linearization times.

    Raises
    ------
    FstFileError
        If a ``LinTimes`` token is not a number.
    """
    joined = " ".join(tokens).replace(",", " ")
    try:
        return [fortran_float(value) for value in joined.split()]
    except ValueError as exc:
        msg = f"LinTimes in {path} has a non-numeric value: {joined!r}"
        raise FstFileError(msg) from exc


def _read_lines(path: Path) -> list[str]:
    """Read a text input file into a list of lines.

    Parameters
    ----------
    path : pathlib.Path
        File to read.

    Returns
    -------
    list[str]
        The file's lines.

    Raises
    ------
    FstFileError
        If the file does not exist or cannot be read. Bytes that are not UTF-8
        (e.g. a Windows-1252 character in a comment) are replaced, not rejected.
    """
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        msg = f"Cannot read OpenFAST input file {path}: {exc}"
        raise FstFileError(msg) from exc


def _field_tokens(lines: list[str], keyword: str) -> list[str] | None:
    """Return the value tokens preceding ``keyword`` on its line, if present.

    Parameters
    ----------
    lines : list[str]
        File lines to search.
    keyword : str
        The keyword (field name) to locate. An exact match is preferred; otherwise
        the first case-insensitive match is used, as OpenFAST itself matches
        keywords case-insensitively (``PLExp``/``PLexp``).

    Returns
    -------
    list[str] or None
        The whitespace-delimited tokens before the keyword, or ``None`` if the
        keyword is not found.
    """
    folded = keyword.casefold()
    fallback: list[str] | None = None
    for line in lines:
        tokens, _ = _split_field_line(line)
        if len(tokens) < 2:
            continue
        if tokens[-1] == keyword:
            return tokens[:-1]
        if fallback is None and tokens[-1].casefold() == folded:
            fallback = tokens[:-1]
    return fallback


def _split_field_line(line: str) -> tuple[list[str], int]:
    """Split an input-file line into its value/keyword tokens and description.

    The line is tokenised with quoted strings kept whole, so a quoted path that
    contains blanks or the ``" - "`` description separator stays one token. The
    description starts at the first unquoted ``-`` token.

    Parameters
    ----------
    line : str
        One line of an OpenFAST input file.

    Returns
    -------
    tokens : list[str]
        The tokens before the description (values, then the keyword), with quotes
        preserved.
    head_end : int
        The character offset where the head ends: ``line[:head_end]`` is the
        value/keyword part and ``line[head_end:]`` the ``" - description"``
        separator and comment (empty when the line has no description).
    """
    tokens: list[str] = []
    for match in _LINE_TOKEN.finditer(line):
        if match.group(0) == "-":
            start = match.start()
            return tokens, start - 1 if start > 0 else start
        tokens.append(match.group(0))
    return tokens, len(line)


def _get_int(
    lines: list[str], keyword: str, path: Path, default: int | None = None
) -> int:
    """Return the integer value of ``keyword``, or ``default`` if it is absent.

    A ``default`` of ``None`` marks the field as required; any other value makes
    it optional, so module switches and linearization settings that a given
    OpenFAST version omits do not cause a parse failure.
    """
    tokens = _field_tokens(lines, keyword)
    if tokens is None:
        if default is None:
            msg = f"Keyword {keyword!r} not found in {path}"
            raise FstFileError(msg)
        return default
    value = _first_value(tokens)
    try:
        return int(value)
    except ValueError as exc:
        msg = f"Keyword {keyword!r} in {path} has a non-integer value {value!r}"
        raise FstFileError(msg) from exc


def _get_float(
    lines: list[str], keyword: str, path: Path, default: float | None = None
) -> float:
    """Return the float value of ``keyword``, or ``default`` if it is absent."""
    tokens = _field_tokens(lines, keyword)
    if tokens is None:
        if default is None:
            msg = f"Keyword {keyword!r} not found in {path}"
            raise FstFileError(msg)
        return default
    value = _first_value(tokens)
    try:
        return fortran_float(value)
    except ValueError as exc:
        msg = f"Keyword {keyword!r} in {path} has a non-numeric value {value!r}"
        raise FstFileError(msg) from exc


def _get_bool(
    lines: list[str], keyword: str, path: Path, default: bool | None = None
) -> bool:
    """Return the boolean value of ``keyword``, or ``default`` if it is absent.

    The value is read the way a Fortran list-directed ``READ`` reads a logical:
    an optional leading ``.`` is skipped and the next character decides, so
    ``True``, ``T``, ``.true.``, and even a misspelt ``Tru`` are true, while
    ``False``, ``F``, ``.false.``, and ``fale`` are false.
    """
    tokens = _field_tokens(lines, keyword)
    if tokens is None:
        if default is None:
            msg = f"Keyword {keyword!r} not found in {path}"
            raise FstFileError(msg)
        return default
    value = _first_value(tokens)
    initial = value.removeprefix(".")[:1].lower()
    if initial == "t":
        return True
    if initial == "f":
        return False
    msg = f"Cannot parse {keyword!r} value {value!r} as boolean in {path}"
    raise FstFileError(msg)


def _fortran_true(value: str) -> bool:
    """Return whether a flag value reads as a true Fortran logical.

    Fortran reads a logical from its first character after an optional ``.``, so
    ``True``, ``T`` and ``.true.`` are true (quotes, a list-directed comma and case
    are ignored). ``1`` is also accepted as true.
    """
    text = _first_value([value.strip().strip("\"'")]).strip()
    return text == "1" or text.removeprefix(".")[:1].lower() == "t"


_FORTRAN_EXPONENT = re.compile(r"(?<=[0-9.])[dD](?=[+-]?[0-9])")


def fortran_float(value: str) -> float:
    """Return a deck value read as a Fortran real.

    The token is cut at its first comma (a list-directed separator, as in
    ``1.21D+02, TipRad``), quotes are stripped, and a double-precision exponent
    (``1.21D+02``, ``5d-3``) is read like ``E``, as Fortran list-directed input does.

    Parameters
    ----------
    value : str
        The value token.

    Returns
    -------
    float
        The number.

    Raises
    ------
    ValueError
        If the token is not a number.
    """
    text = _first_value([value.strip()]).strip().strip("\"'")
    return float(_FORTRAN_EXPONENT.sub("E", text))


def _fortran_logical(value: str) -> bool | None:
    """Return a flag value read as a Fortran logical, or ``None`` if it is not one.

    ``True``/``T``/``.true.``/``.T.``/``1`` are true and ``False``/``F``/``.false.``/
    ``.F.``/``0`` false (see :func:`_fortran_true`); anything else is ``None``.
    """
    text = _first_value([value.strip().strip("\"'")]).strip()
    if text == "1":
        return True
    if text == "0":
        return False
    head = text.removeprefix(".")[:1].lower()
    return {"t": True, "f": False}.get(head)


def _first_value(tokens: list[str]) -> str:
    """Return the first list-directed value of a field's value tokens.

    Fortran list-directed input separates values by blanks *or* commas, so a line
    such as ``1,   0   LinInputs`` holds the value ``1``: the token is cut at its
    first comma.
    """
    return tokens[0].split(",", 1)[0]
