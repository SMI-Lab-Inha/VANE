# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Reader for OpenFAST linearization (``.lin``) files.

This module parses the *modern* linearization output produced by the OpenFAST
ModGlue engine into structured, in-memory objects. The modern format is
identified by the presence of a ``Derivative Order`` column in the
operating-point tables; the legacy format (which lacks that column and instead
emits ``StateRotation`` / mass-matrix blocks) is intentionally not supported and
raises :class:`LinFileFormatError`.

The on-disk layout consumed here is, in order:

1. A header block beginning at ``Simulation information:`` holding scalar
   metadata (simulation time, rotor speed in rad/s, azimuth in rad, wind speed,
   and channel counts) and terminated by ``Jacobians included in this file?``.
2. Operating-point tables (``Order of continuous states:`` and the analogous
   blocks for state derivatives, inputs, and outputs). Each row carries an
   index, an operating-point value, a rotating-frame flag, a derivative order,
   and a free-text description.
3. Matrix blocks (``A``, ``B``, ``C``, ``D`` under ``Linearized state
   matrices:`` and, when Jacobians are written, ``dUdu``/``dUdy``/``dXdy``/``J``
   under ``Jacobian matrices:``).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import os

    import numpy.typing as npt

__all__ = [
    "LinFile",
    "LinFileFormatError",
    "NonFiniteLinValue",
    "OperatingPointTable",
    "find_nonfinite_lin_value",
    "read_lin_file",
]

logger = logging.getLogger(__name__)

# Header markers and table labels used by the OpenFAST ModGlue writer.
_SIM_INFO_MARKER = "Simulation information:"
_JACOBIANS_MARKER = "Jacobians included in this file?"
_CONTINUOUS_STATES_LABEL = "Number of continuous states:"
_DERIV_ORDER_COLUMN = "Derivative Order"

# Maps an operating-point table label to the attribute name on :class:`LinFile`.
_OP_TABLE_LABELS: dict[str, str] = {
    "Order of continuous states:": "x",
    "Order of continuous state derivatives:": "xdot",
    "Order of inputs": "u",
    "Order of outputs": "y",
}

# Matrix header, e.g. ``A: 138 x 138``.
_MATRIX_HEADER_RE = re.compile(r"^([A-Za-z]+):\s+(\d+)\s+x\s+(\d+)\s*$")
# Module prefix at the start of a channel description, e.g. ``ED`` or ``BD_1``.
_MODULE_PREFIX_RE = re.compile(r"^([A-Za-z]+(?:_\d+)?)\s")


class LinFileFormatError(ValueError):
    """Raised when a ``.lin`` file cannot be parsed as the modern format."""


@dataclass
class OperatingPointTable:
    """Operating-point metadata for one channel group of a ``.lin`` file.

    A channel group is the set of continuous states, continuous-state
    derivatives, inputs, or outputs. All array attributes are parallel and share
    the channel ordering written in the file.

    Parameters
    ----------
    values : npt.NDArray[np.float64]
        Operating-point value of each channel, shape ``(n_channels,)``. For
        orientation channels that are written with multiple comma-separated
        components, only the first component is retained here.
    rotating_frame : npt.NDArray[np.bool_]
        Per-channel flag, shape ``(n_channels,)``. ``True`` marks a channel
        expressed in the blade-rotating frame (the hook used by the MBC3
        transform).
    derivative_order : npt.NDArray[np.int_]
        Per-channel derivative order (0, 1, or 2), shape ``(n_channels,)``.
    descriptions : list[str]
        Per-channel free-text description, length ``n_channels``. Each begins
        with the originating module prefix (e.g. ``"ED ..."``).
    """

    values: npt.NDArray[np.float64]
    rotating_frame: npt.NDArray[np.bool_]
    derivative_order: npt.NDArray[np.int_]
    descriptions: list[str]

    def __len__(self) -> int:
        """Return the number of channels in the table."""
        return int(self.values.shape[0])

    @property
    def modules(self) -> list[str]:
        """Return the originating module prefix of each channel.

        Returns
        -------
        list[str]
            The leading module token of every description (e.g. ``"ED"`` or
            ``"BD_1"``); an empty string is used when no prefix is present.
        """
        out: list[str] = []
        for desc in self.descriptions:
            match = _MODULE_PREFIX_RE.match(desc)
            out.append(match.group(1) if match else "")
        return out


@dataclass
class LinFile:
    """Parsed contents of a single OpenFAST ``.lin`` linearization file.

    Parameters
    ----------
    path : pathlib.Path
        Source file the data was read from.
    sim_time : float
        Simulation time of the linearization, in seconds.
    rotor_speed : float
        Rotor speed at the operating point, in rad/s (as written by OpenFAST).
    azimuth : float
        Rotor azimuth at the operating point, in rad.
    wind_speed : float
        Mean hub-height wind speed, in m/s. Zero when InflowWind is absent.
    jacobians_included : bool
        Whether the file contains the ``Jacobian matrices`` block.
    x, xdot, u, y : OperatingPointTable
        Operating-point tables for the continuous states, continuous-state
        derivatives, inputs, and outputs, respectively.
    a, b, c, d : npt.NDArray[np.float64] or None
        State-space matrices :math:`A` (n_x, n_x), :math:`B` (n_x, n_u),
        :math:`C` (n_y, n_x), and :math:`D` (n_y, n_u). ``None`` when the
        corresponding block is absent (e.g. ``B``/``D`` with zero inputs).
    jacobians : dict[str, npt.NDArray[np.float64]]
        Any additional matrices from the ``Jacobian matrices`` block, keyed by
        their label (e.g. ``"dUdu"``, ``"dUdy"``, ``"dXdy"``).
    """

    path: Path
    sim_time: float
    rotor_speed: float
    azimuth: float
    wind_speed: float
    jacobians_included: bool
    x: OperatingPointTable
    xdot: OperatingPointTable
    u: OperatingPointTable
    y: OperatingPointTable
    a: npt.NDArray[np.float64] | None
    b: npt.NDArray[np.float64] | None
    c: npt.NDArray[np.float64] | None
    d: npt.NDArray[np.float64] | None
    jacobians: dict[str, npt.NDArray[np.float64]]

    @property
    def n_x(self) -> int:
        """Return the number of continuous states."""
        return len(self.x)

    @property
    def n_u(self) -> int:
        """Return the number of inputs."""
        return len(self.u)

    @property
    def n_y(self) -> int:
        """Return the number of outputs."""
        return len(self.y)


def read_lin_file(path: str | os.PathLike[str]) -> LinFile:
    """Read and parse a modern OpenFAST ``.lin`` file.

    Parameters
    ----------
    path : str or os.PathLike
        Path to the ``.lin`` file to read.

    Returns
    -------
    LinFile
        The parsed linearization data.

    Raises
    ------
    LinFileFormatError
        If the file is empty, truncated, in the unsupported legacy format, or
        otherwise does not conform to the modern ModGlue layout.

    Notes
    -----
    Numeric fields containing Fortran overflow markers (runs of ``*``) are
    parsed as :data:`numpy.nan` and a warning is logged; the parse does not
    fail. Empty ``B``/``C``/``D`` blocks (no inputs/outputs) are not written by
    OpenFAST, so the corresponding :class:`LinFile` attribute is ``None``; the ``A``
    matrix, however, is required whenever the file declares continuous states.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        msg = f"Linearization file is empty: {path}"
        raise LinFileFormatError(msg)

    cursor = _Cursor(lines)
    scalars, jacobians_included, n_continuous = _parse_header(cursor, path)
    tables = _parse_operating_point_tables(cursor, path)
    matrices = _parse_matrices(cursor, path)

    lin = LinFile(
        path=path,
        sim_time=scalars["sim_time"],
        rotor_speed=scalars["rotor_speed"],
        azimuth=scalars["azimuth"],
        wind_speed=scalars["wind_speed"],
        jacobians_included=jacobians_included,
        x=tables["x"],
        xdot=tables["xdot"],
        u=tables["u"],
        y=tables["y"],
        a=matrices.pop("A", None),
        b=matrices.pop("B", None),
        c=matrices.pop("C", None),
        d=matrices.pop("D", None),
        jacobians=matrices,
    )
    _validate_lin_file(lin, declared_n_continuous=n_continuous)
    return lin


def _validate_lin_file(
    lin: LinFile, *, declared_n_continuous: int | None = None
) -> None:
    """Validate operating metadata and state-space matrix shapes after parsing.

    Overflow markers in matrix *data* are tolerated as ``nan`` (see
    :func:`read_lin_file`), but a non-finite operating-point scalar or baseline value
    (the ``x``/``xdot``/``u``/``y`` op tables) denotes a corrupt linearization, and a
    state-space matrix whose shape disagrees with the channel-table dimensions cannot be
    used downstream. All fail loudly here rather than surfacing as an opaque error deep
    in the analysis.

    Parameters
    ----------
    lin : LinFile
        The freshly parsed linearization file.
    declared_n_continuous : int or None, optional
        The continuous-state count from the file header. When given, the parsed ``x``
        operating-point table must contain exactly that many channels; a mismatch means
        the file is truncated (which would otherwise zero out ``n_x`` and bypass the
        ``A``-required check). ``None`` skips this cross-check (header lacks the count).

    Raises
    ------
    LinFileFormatError
        If an operating-point scalar or a baseline op-table value is non-finite, if the
        parsed continuous-state table disagrees with the header count, if the ``A``
        matrix is absent while the file declares continuous states, or a present
        ``A``/``B``/``C``/``D`` matrix does not match ``(n_x, n_x)``/``(n_x, n_u)``/
        ``(n_y, n_x)``/``(n_y, n_u)``.
    """
    for field_name, value in (
        ("rotor speed", lin.rotor_speed),
        ("azimuth", lin.azimuth),
        ("wind speed", lin.wind_speed),
        ("simulation time", lin.sim_time),
    ):
        if not np.isfinite(value):
            msg = f"{lin.path}: non-finite {field_name} ({value})"
            raise LinFileFormatError(msg)

    # The operating-point baselines (x/xdot/u/y op values) are the steady state the
    # linearization is taken about — a non-finite entry there (an overflow/NaN token in
    # the channel tables) corrupts the baseline downstream, so reject it. (Non-finite
    # values in the A/B/C/D matrix *data* are tolerated as ``nan``.)
    for table_name, table in (
        ("x", lin.x),
        ("xdot", lin.xdot),
        ("u", lin.u),
        ("y", lin.y),
    ):
        if table.values.size and not np.all(np.isfinite(table.values)):
            msg = f"{lin.path}: non-finite value in op table {table_name!r}"
            raise LinFileFormatError(msg)

    # Anchor the state count to the header's declaration: if the operating-point table
    # was truncated away, lin.n_x (from the parsed table) would read 0 and bypass the
    # A-required check below, so cross-check it against the header count first.
    if declared_n_continuous is not None and declared_n_continuous != lin.n_x:
        msg = (
            f"{lin.path}: header declares {declared_n_continuous} continuous states "
            f"but the op table has {lin.n_x}; the file is truncated or malformed"
        )
        raise LinFileFormatError(msg)

    # A is the load-bearing matrix for modal analysis: a full-system linearization with
    # continuous states must include it. Its absence (a truncated/partial .lin) is a
    # parse-time failure, not a silently A-less LinFile that only breaks later in MBC.
    if lin.a is None and lin.n_x > 0:
        msg = (
            f"{lin.path}: no A (state) matrix found, but the file declares "
            f"{lin.n_x} continuous states; the linearization is truncated or malformed"
        )
        raise LinFileFormatError(msg)

    expected = {
        "A": (lin.a, (lin.n_x, lin.n_x)),
        "B": (lin.b, (lin.n_x, lin.n_u)),
        "C": (lin.c, (lin.n_y, lin.n_x)),
        "D": (lin.d, (lin.n_y, lin.n_u)),
    }
    for name, (matrix, shape) in expected.items():
        if matrix is not None and matrix.shape != shape:
            msg = (
                f"{lin.path}: matrix {name} has shape {matrix.shape}, "
                f"expected {shape} from the channel tables"
            )
            raise LinFileFormatError(msg)


@dataclass(frozen=True)
class NonFiniteLinValue:
    """The first non-finite number found in a ``.lin`` file.

    Parameters
    ----------
    path : pathlib.Path
        The ``.lin`` file scanned.
    section : str
        Where the value sits: ``"header"`` for a header scalar, ``"x"``/``"xdot"``/
        ``"u"``/``"y"`` for an operating-point table, or the matrix label (``"A"``,
        ``"B"``, ``"C"``, ``"D"`` or a Jacobian such as ``"dUdu"``).
    variable : str
        What the value belongs to: the header field (``"Rotor Speed"``), the channel
        description of an operating-point row (``"AD x3 blade 1, node 59, -"``), or
        the matrix entry with its row and column channels
        (``"A[247, 1] (row: ...; column: ...)"``, 1-based).
    token : str
        The value as written in the file (``"NaN"``, ``"Infinity"``, ``"*****"``, ...).
    line_number : int
        The 1-based line of the file holding the value.
    """

    path: Path
    section: str
    variable: str
    token: str
    line_number: int

    @property
    def summary(self) -> str:
        """Return the location, variable and token as one line."""
        where = "header" if self.section == "header" else f"table {self.section!r}"
        if self.section not in {"header", *_OP_TABLE_LABELS.values()}:
            where = f"matrix {self.section!r}"
        return (
            f"{self.path.name} line {self.line_number}: {self.token} in {where} "
            f"for {self.variable}"
        )


# Lower-case substrings every spelling of a non-finite number contains (``NaN``,
# ``Inf``/``Infinity``, a Fortran ``*`` overflow run, an Intel/MSVC ``1.#QNAN``). Plain
# substring searches are used as the pre-filter because they scan a file of many
# megabytes in milliseconds, where a regular expression takes most of a second.
_NONFINITE_NEEDLES = ("nan", "inf", "*", "#")


def _has_candidate(lowered: str) -> bool:
    """Return whether lower-cased text holds anything that could be non-finite."""
    return any(needle in lowered for needle in _NONFINITE_NEEDLES)


def _has_nonfinite_token(text: str) -> bool:
    """Return whether any whitespace-delimited token of ``text`` is non-finite.

    Only the tokens around a needle are parsed, so a clean file costs a few substring
    searches. A word such as ``InflowWind`` contains a needle but is not a number.
    """
    lowered = text.lower()
    for needle in _NONFINITE_NEEDLES:
        start = lowered.find(needle)
        while start >= 0:
            begin = start
            while begin > 0 and not text[begin - 1].isspace():
                begin -= 1
            end = start + len(needle)
            while end < len(text) and not text[end].isspace():
                end += 1
            if _is_nonfinite_token(text[begin:end]):
                return True
            start = lowered.find(needle, end)
    return False


def _is_nonfinite_token(token: str) -> bool:
    """Return whether a numeric ``token`` reads as NaN or infinity.

    Uses the same spellings as :func:`_parse_float` (Fortran ``*`` overflow runs and
    Intel/MSVC ``1.#QNAN``-style markers included) but never logs. A non-numeric
    token is not a non-finite number.
    """
    token = token.rstrip(",")
    if not token:
        return False
    if "*" in token or _fortran_nonfinite(token) is not None:
        return True
    try:
        return not np.isfinite(float(token))
    except ValueError:
        return False


def find_nonfinite_lin_value(path: str | os.PathLike[str]) -> NonFiniteLinValue | None:
    """Return the first non-finite number in a ``.lin`` file, or ``None``.

    The scan covers the header scalars, the four operating-point tables (every
    component of an orientation channel) and every matrix block. :func:`read_lin_file`
    rejects a non-finite header scalar or operating-point value but accepts a
    non-finite matrix entry as ``nan``. This scan finds both, and names the channel
    the value belongs to.

    Parameters
    ----------
    path : str or os.PathLike
        The ``.lin`` file to scan.

    Returns
    -------
    NonFiniteLinValue or None
        The first non-finite value in file order, or ``None`` when every number in
        the file is finite.

    Raises
    ------
    OSError
        If the file cannot be read.

    Notes
    -----
    The file is first searched as a whole for a token that spells a non-finite
    number (``NaN``, ``Inf``, ``Infinity``, a Fortran ``*`` overflow run or an
    Intel/MSVC ``1.#QNAN``-style marker), which is fast even for a file of many
    megabytes. Only a file with such a token is then walked line by line, so that a
    non-numeric word (in a channel description, say) is never reported. Lines that do
    not follow the modern layout are skipped rather than rejected: the scan reports
    non-finite numbers and does not validate the format.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    if not _has_nonfinite_token(text):
        return None
    return _locate_nonfinite(path, text.splitlines())


def _locate_nonfinite(path: Path, lines: list[str]) -> NonFiniteLinValue | None:
    """Walk the lines of a ``.lin`` file and return its first non-finite value."""
    descriptions: dict[str, list[str]] = {}
    section = "header"
    table: list[str] = []
    matrix_rows: tuple[str, str] | None = None
    matrix_row = 0
    expect_header_rows = 0
    for number, line in enumerate(lines, start=1):
        label = _match_table_label(line)
        if label is not None:
            section = _OP_TABLE_LABELS[label]
            table = descriptions.setdefault(section, [])
            expect_header_rows = 2  # column labels and the dashed separator
            continue
        header = _MATRIX_HEADER_RE.match(line)
        if header is not None:
            section = header.group(1)
            matrix_rows = _MATRIX_CHANNELS.get(section)
            matrix_row = 0
            continue
        if expect_header_rows:
            expect_header_rows -= 1
            continue
        if not line.strip():
            continue
        if section == "header":
            found = _nonfinite_in_header_line(path, line, number)
        elif section in _OP_TABLE_LABELS.values():
            found = _nonfinite_in_table_row(path, section, line, number, table)
        else:
            matrix_row += 1
            found = _nonfinite_in_matrix_row(
                path, section, line, number, matrix_row, matrix_rows, descriptions
            )
        if found is not None:
            return found
    return None


# The operating-point tables naming a matrix's rows and columns: a row of A or B is
# the derivative of a state, a row of C or D an output.
_MATRIX_CHANNELS: dict[str, tuple[str, str]] = {
    "A": ("xdot", "x"),
    "B": ("xdot", "u"),
    "C": ("y", "x"),
    "D": ("y", "u"),
}


def _nonfinite_in_header_line(
    path: Path, line: str, number: int
) -> NonFiniteLinValue | None:
    """Return a non-finite header scalar on ``line`` (``Label: value unit``)."""
    if ":" not in line or not _has_candidate(line.lower()):
        return None
    field, _, rest = line.partition(":")
    tokens = rest.split()
    if tokens and _is_nonfinite_token(tokens[0]):
        return NonFiniteLinValue(path, "header", field.strip(), tokens[0], number)
    return None


def _nonfinite_in_table_row(
    path: Path, section: str, line: str, number: int, table: list[str]
) -> NonFiniteLinValue | None:
    """Record an operating-point row's description; return it if its value is bad."""
    tokens = line.split()
    rot_idx = next((i for i, tok in enumerate(tokens) if tok in ("T", "F")), -1)
    if rot_idx < 1:
        return None
    description = " ".join(tokens[rot_idx + 2 :])
    table.append(description)
    if not _has_candidate(line.lower()):
        return None
    for token in tokens[1:rot_idx]:
        if _is_nonfinite_token(token):
            return NonFiniteLinValue(path, section, description, token, number)
    return None


def _nonfinite_in_matrix_row(
    path: Path,
    section: str,
    line: str,
    number: int,
    row: int,
    channels: tuple[str, str] | None,
    descriptions: dict[str, list[str]],
) -> NonFiniteLinValue | None:
    """Return the first non-finite entry of a matrix row, naming its channels."""
    if not _has_candidate(line.lower()):
        return None
    for column, token in enumerate(line.split(), start=1):
        if not _is_nonfinite_token(token):
            continue
        variable = f"{section}[{row}, {column}]"
        if channels is not None:
            row_name = _channel_name(descriptions.get(channels[0], []), row)
            column_name = _channel_name(descriptions.get(channels[1], []), column)
            variable += f" (row: {row_name}; column: {column_name})"
        return NonFiniteLinValue(path, section, variable, token, number)
    return None


def _channel_name(table: list[str], index: int) -> str:
    """Return the 1-based ``index`` channel description, or a placeholder."""
    if 1 <= index <= len(table):
        return table[index - 1] or f"channel {index}"
    return f"channel {index}"


class _Cursor:
    """A forward-only line cursor over the contents of a ``.lin`` file."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self._index = 0

    @property
    def exhausted(self) -> bool:
        """Return whether every line has been consumed."""
        return self._index >= len(self._lines)

    def peek(self) -> str | None:
        """Return the current line without advancing, or ``None`` at the end."""
        if self.exhausted:
            return None
        return self._lines[self._index]

    def next(self) -> str:
        """Return the current line and advance the cursor."""
        line = self._lines[self._index]
        self._index += 1
        return line

    def advance_to(self, marker: str) -> bool:
        """Advance until a line containing ``marker``; consume that line.

        Parameters
        ----------
        marker : str
            Substring to search for.

        Returns
        -------
        bool
            ``True`` if the marker was found (cursor now sits just past it),
            ``False`` if the end of file was reached first.
        """
        while not self.exhausted:
            if marker in self.next():
                return True
        return False


def _fortran_nonfinite(token: str) -> float | None:
    """Map a Fortran/MSVC non-finite *spelling* to its value, or ``None`` if not one.

    Intel/MSVC runtimes write non-finite numbers as ``1.#QNAN``, ``1.#IND`` (NaN), and
    ``1.#INF`` / ``-1.#INF`` — none of which Python's ``float()`` accepts. These are
    distinct from a plain Fortran ``*`` overflow run (handled separately).
    """
    lowered = token.lower()
    if "#" not in lowered:
        return None
    if "qnan" in lowered or "ind" in lowered:
        return float("nan")
    if "inf" in lowered:
        return float("-inf") if lowered.lstrip().startswith("-") else float("inf")
    return None  # an unrecognised ``#`` token; let float() reject it below


def _parse_float(token: str) -> float:
    """Parse a Fortran-formatted float, mapping non-finite markers to nan/inf.

    Parameters
    ----------
    token : str
        A single whitespace-delimited numeric token.

    Returns
    -------
    float
        The parsed value. A Fortran ``*`` overflow run or an Intel/MSVC non-finite
        spelling (``1.#QNAN``/``1.#IND``/``1.#INF``) maps to ``nan``/``inf`` with a
        warning; a literal ``nan``/``inf`` is accepted but also warned.

    Raises
    ------
    LinFileFormatError
        If the token is neither numeric nor a recognised non-finite marker, instead of
        the raw ``ValueError`` ``float()`` would raise (which escapes the documented
        ``LinFileFormatError`` contract).
    """
    if "*" in token:
        logger.warning("Overflow marker %r parsed as NaN", token)
        return float("nan")
    nonfinite = _fortran_nonfinite(token)
    if nonfinite is not None:
        logger.warning("Non-finite marker %r parsed as %s", token, nonfinite)
        return nonfinite
    try:
        value = float(token)
    except ValueError as exc:
        msg = f"non-numeric value {token!r} in a linearization file"
        raise LinFileFormatError(msg) from exc
    if not np.isfinite(value):
        logger.warning("Non-finite value %r read from a linearization file", token)
    return value


def _normalize_scalar(key: str, value: float, unit: str) -> float:
    """Normalize a header scalar to rad/s (rotor speed) and rad (azimuth).

    OpenFAST writes rotor speed in rad/s and azimuth in rad, but its documentation
    describes rpm/deg. If a file's unit token reports those alternative units, the
    value is converted so downstream code always receives rad/s and rad.

    Parameters
    ----------
    key : str
        The scalar's attribute name (e.g. ``"rotor_speed"``).
    value : float
        The parsed numeric value.
    unit : str
        The unit token that followed the value in the header, if any.

    Returns
    -------
    float
        The value in internal units.
    """
    unit_lower = unit.lower()
    if key == "rotor_speed" and unit_lower == "rpm":
        return value * float(np.pi) / 30.0
    if key == "azimuth" and unit_lower.startswith("deg"):
        return value * float(np.pi) / 180.0
    return value


def _parse_header(
    cursor: _Cursor, path: Path
) -> tuple[dict[str, float], bool, int | None]:
    """Parse the ``Simulation information`` header block.

    Parameters
    ----------
    cursor : _Cursor
        Cursor positioned anywhere before the header.
    path : pathlib.Path
        Source path, used only for error messages.

    Returns
    -------
    tuple[dict[str, float], bool, int or None]
        A mapping with keys ``sim_time``, ``rotor_speed``, ``azimuth``, and
        ``wind_speed``; a flag indicating whether Jacobians are included; and the
        header's declared continuous-state count (``None`` if the header omits it).

    Raises
    ------
    LinFileFormatError
        If the header marker or a required field is missing.
    """
    if not cursor.advance_to(_SIM_INFO_MARKER):
        msg = f"Missing {_SIM_INFO_MARKER!r} header in {path}"
        raise LinFileFormatError(msg)

    scalars: dict[str, float] = {}
    # Both the space-separated header label and the underscore form used in the
    # OpenFAST documentation are accepted for rotor speed.
    field_map = {
        "Simulation time:": "sim_time",
        "Rotor Speed:": "rotor_speed",
        "Rotor_Speed:": "rotor_speed",
        "Azimuth:": "azimuth",
        "Wind Speed:": "wind_speed",
    }
    jacobians_included = False
    n_continuous: int | None = None
    while not cursor.exhausted:
        line = cursor.next()
        # The declared continuous-state count anchors the missing-A / truncated-table
        # check below: the file's own statement of how many states it should carry.
        if _CONTINUOUS_STATES_LABEL in line:
            tokens = line.split(_CONTINUOUS_STATES_LABEL)[1].split()
            if not tokens:
                # A present-but-empty label would otherwise leave the count ``None`` and
                # silently disable the missing-A / truncated-table cross-check below.
                msg = (
                    f"Header field {_CONTINUOUS_STATES_LABEL!r} in {path} has no "
                    f"value: {line.rstrip()!r}"
                )
                raise LinFileFormatError(msg)
            try:
                n_continuous = int(tokens[0])
            except ValueError as exc:
                msg = (
                    f"Header field {_CONTINUOUS_STATES_LABEL!r} in {path} has a "
                    f"non-integer value: {line.rstrip()!r}"
                )
                raise LinFileFormatError(msg) from exc
        for label, key in field_map.items():
            if label in line:
                fields = line.split(label)[1].split()
                if not fields:
                    msg = (
                        f"Header field {label!r} in {path} has no value: "
                        f"{line.rstrip()!r}"
                    )
                    raise LinFileFormatError(msg)
                unit = fields[1] if len(fields) > 1 else ""
                scalars[key] = _normalize_scalar(key, _parse_float(fields[0]), unit)
                break
        if _JACOBIANS_MARKER in line:
            jacobians_included = line.rsplit(maxsplit=1)[-1].strip().lower() == "yes"
            break

    missing = {"sim_time", "rotor_speed", "azimuth", "wind_speed"} - scalars.keys()
    if missing:
        msg = f"Header in {path} is missing fields: {sorted(missing)}"
        raise LinFileFormatError(msg)
    return scalars, jacobians_included, n_continuous


def _parse_operating_point_tables(
    cursor: _Cursor, path: Path
) -> dict[str, OperatingPointTable]:
    """Parse the four operating-point tables that follow the header.

    Parameters
    ----------
    cursor : _Cursor
        Cursor positioned just after the header block.
    path : pathlib.Path
        Source path, used only for error messages.

    Returns
    -------
    dict[str, OperatingPointTable]
        Tables keyed by ``"x"``, ``"xdot"``, ``"u"``, and ``"y"``. Missing
        tables (e.g. no inputs) yield empty tables.

    Raises
    ------
    LinFileFormatError
        If a table is present but in the unsupported legacy format.
    """
    tables: dict[str, OperatingPointTable] = {
        key: _empty_table() for key in ("x", "xdot", "u", "y")
    }
    while not cursor.exhausted:
        line = cursor.peek()
        if line is None:
            break
        label = _match_table_label(line)
        if label is None:
            if "Linearized state matrices:" in line or "Jacobian matrices:" in line:
                break
            cursor.next()
            continue
        cursor.next()  # consume the label line
        tables[_OP_TABLE_LABELS[label]] = _parse_one_table(cursor, path)
    return tables


def _match_table_label(line: str) -> str | None:
    """Return the operating-point table label contained in ``line``, if any."""
    for label in _OP_TABLE_LABELS:
        if line.startswith(label):
            return label
    return None


def _parse_one_table(cursor: _Cursor, path: Path) -> OperatingPointTable:
    """Parse a single operating-point table starting at its column header.

    Parameters
    ----------
    cursor : _Cursor
        Cursor positioned on the column-header line of the table.
    path : pathlib.Path
        Source path, used only for error messages.

    Returns
    -------
    OperatingPointTable
        The parsed table; empty when no data rows are present.

    Raises
    ------
    LinFileFormatError
        If the table lacks the modern ``Derivative Order`` column.
    """
    header = cursor.next()
    if _DERIV_ORDER_COLUMN not in header:
        msg = (
            f"{path} uses the unsupported legacy .lin format "
            f"(no {_DERIV_ORDER_COLUMN!r} column). Only the modern ModGlue "
            f"format is supported."
        )
        raise LinFileFormatError(msg)
    if not cursor.exhausted:
        cursor.next()  # consume the dashed separator line

    values: list[float] = []
    rotating: list[bool] = []
    deriv_order: list[int] = []
    descriptions: list[str] = []
    while not cursor.exhausted:
        line = cursor.peek()
        if line is None or not line.strip():
            break
        cursor.next()
        value, is_rot, order, desc = _parse_table_row(line, path)
        values.append(value)
        rotating.append(is_rot)
        deriv_order.append(order)
        descriptions.append(desc)

    return OperatingPointTable(
        values=np.asarray(values, dtype=np.float64),
        rotating_frame=np.asarray(rotating, dtype=np.bool_),
        derivative_order=np.asarray(deriv_order, dtype=np.int_),
        descriptions=descriptions,
    )


def _parse_table_row(line: str, path: Path) -> tuple[float, bool, int, str]:
    """Parse one operating-point table row.

    The row layout is ``index value[, value, value] {T|F} order description``,
    where orientation channels carry three comma-separated operating-point
    values (only the first is returned).

    Parameters
    ----------
    line : str
        A single data row from an operating-point table.
    path : pathlib.Path
        Source path, used only for error messages.

    Returns
    -------
    tuple[float, bool, int, str]
        The first operating-point value, the rotating-frame flag, the
        derivative order, and the description.

    Raises
    ------
    LinFileFormatError
        If the rotating-frame column cannot be located, or the derivative-order
        column after it is missing or non-integer (a truncated row).
    """
    tokens = line.split()
    rot_idx = next((i for i, tok in enumerate(tokens) if tok in ("T", "F")), -1)
    if rot_idx < 1:
        msg = f"Could not locate rotating-frame flag in {path} row: {line!r}"
        raise LinFileFormatError(msg)
    if rot_idx + 1 >= len(tokens):
        msg = f"Missing derivative-order column in {path} row: {line!r}"
        raise LinFileFormatError(msg)
    value = _parse_float(tokens[1].rstrip(","))
    is_rot = tokens[rot_idx] == "T"
    try:
        order = int(tokens[rot_idx + 1])
    except ValueError as exc:
        msg = f"Derivative-order column is not an integer in {path} row: {line!r}"
        raise LinFileFormatError(msg) from exc
    description = " ".join(tokens[rot_idx + 2 :])
    return value, is_rot, order, description


def _parse_matrices(cursor: _Cursor, path: Path) -> dict[str, npt.NDArray[np.float64]]:
    """Parse all matrix blocks remaining in the file.

    Parameters
    ----------
    cursor : _Cursor
        Cursor positioned after the operating-point tables.
    path : pathlib.Path
        Source path, used only for error messages.

    Returns
    -------
    dict[str, npt.NDArray[np.float64]]
        Matrices keyed by their label (``"A"``, ``"B"``, ``"dUdu"``, ...).

    Raises
    ------
    LinFileFormatError
        If a declared matrix has fewer data rows than its header specifies.
    """
    matrices: dict[str, npt.NDArray[np.float64]] = {}
    while not cursor.exhausted:
        line = cursor.next()
        match = _MATRIX_HEADER_RE.match(line)
        if match is None:
            continue
        name, n_rows, n_cols = match.group(1), int(match.group(2)), int(match.group(3))
        matrices[name] = _parse_matrix_body(cursor, name, n_rows, n_cols, path)
    return matrices


def _parse_matrix_body(
    cursor: _Cursor, name: str, n_rows: int, n_cols: int, path: Path
) -> npt.NDArray[np.float64]:
    """Read ``n_rows`` numeric rows of a matrix into an array.

    Parameters
    ----------
    cursor : _Cursor
        Cursor positioned just after the matrix header line.
    name : str
        Matrix label, used only for error messages.
    n_rows, n_cols : int
        Expected matrix dimensions from the header.
    path : pathlib.Path
        Source path, used only for error messages.

    Returns
    -------
    npt.NDArray[np.float64]
        The parsed matrix of shape ``(n_rows, n_cols)``.

    Raises
    ------
    LinFileFormatError
        If the block is truncated or a row has the wrong number of columns.
    """
    matrix = np.empty((n_rows, n_cols), dtype=np.float64)
    for row in range(n_rows):
        if cursor.exhausted:
            msg = f"Matrix {name!r} in {path} is truncated at row {row}"
            raise LinFileFormatError(msg)
        tokens = cursor.next().split()
        if len(tokens) != n_cols:
            msg = (
                f"Matrix {name!r} in {path}: row {row} has {len(tokens)} values, "
                f"expected {n_cols}"
            )
            raise LinFileFormatError(msg)
        matrix[row] = [_parse_float(tok) for tok in tokens]
    return matrix


def _empty_table() -> OperatingPointTable:
    """Return an :class:`OperatingPointTable` with no channels."""
    return OperatingPointTable(
        values=np.empty(0, dtype=np.float64),
        rotating_frame=np.empty(0, dtype=np.bool_),
        derivative_order=np.empty(0, dtype=np.int_),
        descriptions=[],
    )
