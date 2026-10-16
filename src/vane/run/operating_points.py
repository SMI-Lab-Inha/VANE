# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Operating points for a linearization sweep.

A linearization is taken at a single operating point — a wind speed, a rotor speed,
and a blade-pitch angle. A Campbell diagram or a load/QoI sweep needs a *series* of
them across the operating envelope. This module defines :class:`OperatingPoint` and
builds a sweep either from an explicit list or by interpolating a turbine's
steady-state operating curve (the rotor speed and pitch a controller would hold at
each wind speed), so a user need only choose the wind speeds to cover.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import PchipInterpolator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vane.run.steady_state import SettleConvergence, SettleCurveCheck

logger = logging.getLogger(__name__)

__all__ = [
    "OperatingPoint",
    "PlatformPosition",
    "operating_curve_from_csv",
    "operating_curve_from_performance",
    "operating_points_from_curve",
]


@dataclass(frozen=True)
class PlatformPosition:
    """Rigid-body floating-platform position (ElastoDyn platform DOFs).

    Up to six platform degrees of freedom used to start a floating linearization at its
    thrust-loaded equilibrium, so the steady-state trim does not have to drag the slow
    platform modes there from rest (which it cannot reliably do near peak thrust).
    Translations are in metres and rotations in degrees, matching ElastoDyn's
    ``PtfmSurge``/``PtfmSway``/``PtfmHeave``/``PtfmRoll``/``PtfmPitch``/``PtfmYaw``.

    Each field defaults to ``None``, meaning *leave that DOF at the deck's value*; only
    the DOFs given are written. This keeps a partially-known position (e.g. only the
    surge and pitch a deck happens to output) from overwriting the deck's other platform
    DOFs with fabricated zeros.

    Parameters
    ----------
    surge, sway, heave : float or None
        Platform translational displacements in metres, or ``None`` to leave unchanged.
    roll, pitch, yaw : float or None
        Platform rotational displacements in degrees, or ``None`` to leave unchanged.
    """

    surge: float | None = None
    sway: float | None = None
    heave: float | None = None
    roll: float | None = None
    pitch: float | None = None
    yaw: float | None = None

    def __post_init__(self) -> None:
        """Reject a non-finite platform displacement before it is written to a deck.

        Raises
        ------
        ValueError
            If any given (non-``None``) platform DOF is NaN or infinite.
        """
        for name in ("surge", "sway", "heave", "roll", "pitch", "yaw"):
            value = getattr(self, name)
            if value is not None and not np.isfinite(value):
                msg = f"platform {name} must be finite, got {value}"
                raise ValueError(msg)


@dataclass(frozen=True)
class OperatingPoint:
    """A single steady operating point for a linearization.

    Parameters
    ----------
    wind_speed : float
        Hub-height wind speed in m/s. ``0`` denotes a no-wind (standstill or
        still-air) point.
    rotor_speed_rpm : float
        Rotor speed in rev/min. ``0`` denotes a parked/standstill rotor.
    blade_pitch_deg : float
        Collective blade-pitch angle in degrees.
    name : str
        Optional label for the point (used to name a generated case directory).
    platform : PlatformPosition or None
        Initial floating-platform position to set in ElastoDyn. ``None`` (the default)
        leaves the platform degrees of freedom at the deck's values — appropriate for
        a bottom-fixed turbine or when the deck is already at equilibrium.
    settle_convergence : SettleConvergence or None
        Audited platform-velocity convergence from :func:`solve_steady_state`, when the
        point came from a settle run. Hand-authored curve points leave it ``None``.
    settle_curve_check : SettleCurveCheck or None
        For a settle run seeded from an operating curve, the settled rotor speed and
        pitch compared with the curve at this wind speed (see
        :func:`~vane.run.compare_settled_to_curve`); ``None`` otherwise.
    """

    wind_speed: float
    rotor_speed_rpm: float
    blade_pitch_deg: float
    name: str = ""
    platform: PlatformPosition | None = None
    settle_convergence: SettleConvergence | None = None
    settle_curve_check: SettleCurveCheck | None = None

    def __post_init__(self) -> None:
        """Reject non-finite or negative operating-point values.

        A non-finite value would otherwise flow into the rotating/still-air branch logic
        (``NaN > 0.0`` is ``False``, so a NaN wind speed would silently take the
        still-air path) and then be written into the OpenFAST deck, producing a
        plausible-looking but invalid operating condition.

        Raises
        ------
        ValueError
            If ``wind_speed``, ``rotor_speed_rpm``, or ``blade_pitch_deg`` is
            non-finite, or if ``wind_speed`` or ``rotor_speed_rpm`` is negative.
        """
        for name, value in (
            ("wind_speed", self.wind_speed),
            ("rotor_speed_rpm", self.rotor_speed_rpm),
            ("blade_pitch_deg", self.blade_pitch_deg),
        ):
            if not np.isfinite(value):
                msg = f"{name} must be finite, got {value}"
                raise ValueError(msg)
        if self.wind_speed < 0.0:
            msg = f"wind_speed must be non-negative, got {self.wind_speed}"
            raise ValueError(msg)
        if self.rotor_speed_rpm < 0.0:
            msg = f"rotor_speed_rpm must be non-negative, got {self.rotor_speed_rpm}"
            raise ValueError(msg)

    @property
    def is_rotating(self) -> bool:
        """Return whether the rotor turns under wind (a periodic operating point)."""
        return self.rotor_speed_rpm != 0.0 and self.wind_speed > 0.0


def operating_points_from_curve(
    wind_speeds: Sequence[float],
    *,
    curve_wind: Sequence[float],
    curve_rotor_speed: Sequence[float],
    curve_pitch: Sequence[float],
) -> list[OperatingPoint]:
    """Build operating points by interpolating a steady-state operating curve.

    The turbine's steady operating curve (rotor speed and collective pitch versus
    wind speed) is interpolated with a shape-preserving monotone cubic (PCHIP, which
    does not overshoot between samples) and evaluated at each requested wind speed.

    Parameters
    ----------
    wind_speeds : Sequence[float]
        Wind speeds (m/s) to produce operating points at.
    curve_wind : Sequence[float]
        Strictly increasing wind-speed samples of the operating curve.
    curve_rotor_speed : Sequence[float]
        Rotor speed (rpm) at each ``curve_wind`` sample.
    curve_pitch : Sequence[float]
        Collective blade pitch (deg) at each ``curve_wind`` sample.

    Returns
    -------
    list[OperatingPoint]
        One operating point per requested wind speed, named ``"ws<speed>"``.

    Raises
    ------
    ValueError
        If the curve arrays differ in length or have fewer than two points, the
        curve wind speeds are not strictly increasing, any curve value is
        non-finite, or a requested wind speed is non-finite or falls outside the
        curve's range (interpolation only — extrapolation is not performed).
    """
    wind = np.asarray(curve_wind, dtype=np.float64)
    rotor = np.asarray(curve_rotor_speed, dtype=np.float64)
    pitch = np.asarray(curve_pitch, dtype=np.float64)
    if not (wind.shape == rotor.shape == pitch.shape):
        msg = (
            f"operating-curve arrays must share a shape, got wind {wind.shape}, "
            f"rotor {rotor.shape}, pitch {pitch.shape}"
        )
        raise ValueError(msg)
    if wind.shape[0] < 2:
        msg = "operating curve needs at least two samples"
        raise ValueError(msg)
    if not np.all(np.diff(wind) > 0.0):
        msg = "curve_wind must be strictly increasing"
        raise ValueError(msg)
    for name, array in (("wind", wind), ("rotor", rotor), ("pitch", pitch)):
        if not np.all(np.isfinite(array)):
            msg = f"operating-curve {name} values must all be finite"
            raise ValueError(msg)

    requested = [float(speed) for speed in wind_speeds]
    low, high = float(wind[0]), float(wind[-1])
    for speed in requested:
        if not np.isfinite(speed) or not low <= speed <= high:
            msg = (
                f"requested wind speed {speed} is non-finite or outside the "
                f"operating-curve range [{low:g}, {high:g}]; extrapolation is "
                "not performed"
            )
            raise ValueError(msg)

    # extrapolate=False guards against producing values outside the sampled curve.
    rotor_of = PchipInterpolator(wind, rotor, extrapolate=False)
    pitch_of = PchipInterpolator(wind, pitch, extrapolate=False)
    return [
        OperatingPoint(
            wind_speed=speed,
            rotor_speed_rpm=float(rotor_of(speed)),
            blade_pitch_deg=float(pitch_of(speed)),
            name=f"ws{speed:g}",
        )
        for speed in requested
    ]


# The three required rotor-performance quantities and whether each must be non-negative.
# Wind speed and rotor speed are physically non-negative; collective blade pitch can be
# negative (a fine-pitch schedule), matching OperatingPoint and the curve API. Both the
# in-memory table path (operating_curve_from_performance) and the CSV path
# (operating_curve_from_csv) read this single policy so they cannot drift apart.
_PERFORMANCE_QUANTITIES: dict[str, bool] = {
    "wind_speed": True,
    "rotor_speed_rpm": True,
    "blade_pitch_deg": False,
}


def operating_curve_from_performance(
    *,
    performance_table: tuple[Sequence[float], Sequence[float], Sequence[float]]
    | None = None,
) -> tuple[list[float], list[float], list[float]]:
    """Derive a steady operating curve from a tabulated rotor-performance schedule.

    Returns the ``(curve_wind, curve_rotor_speed, curve_pitch)`` arrays that
    :func:`operating_points_from_curve` and :func:`~vane.run.run_operating_sweep`
    consume, from an explicit ``performance_table`` -- the turbine's tabulated steady
    operating schedule (wind speed, rotor speed, collective pitch), validated and
    returned sorted by ascending wind speed. The rotor-performance data is always
    supplied by the caller (e.g. read from a CSV with :func:`operating_curve_from_csv`);
    VANE does not compute it from blade geometry.

    Parameters
    ----------
    performance_table : tuple of three Sequence[float] or None, optional
        ``(wind_speed, rotor_speed_rpm, blade_pitch_deg)`` arrays of the turbine's
        steady operating schedule, of equal length (>= 2).

    Returns
    -------
    tuple of three list[float]
        ``(curve_wind, curve_rotor_speed, curve_pitch)``, sorted by ascending wind
        speed, ready to pass to :func:`operating_points_from_curve`.

    Raises
    ------
    ValueError
        If ``performance_table`` arrays differ in length, have fewer than two samples,
        contain a non-finite value, have a negative wind speed or rotor speed (negative
        collective blade pitch is allowed), or have a non-increasing wind-speed axis.
    RuntimeError
        If no ``performance_table`` is given, so there is no rotor-performance source to
        derive the curve from.
    """
    if performance_table is None:
        msg = (
            "rotor-performance data required: pass performance_table=(wind_speed, "
            "rotor_speed_rpm, blade_pitch_deg), or read a CSV with "
            "operating_curve_from_csv"
        )
        raise RuntimeError(msg)

    wind_in, rotor_in, pitch_in = performance_table
    wind = np.asarray(wind_in, dtype=np.float64)
    rotor = np.asarray(rotor_in, dtype=np.float64)
    pitch = np.asarray(pitch_in, dtype=np.float64)
    if not (wind.shape == rotor.shape == pitch.shape):
        msg = (
            f"performance_table arrays must share a shape, got wind {wind.shape}, "
            f"rotor {rotor.shape}, pitch {pitch.shape}"
        )
        raise ValueError(msg)
    if wind.ndim != 1 or wind.shape[0] < 2:
        msg = "performance_table needs at least two one-dimensional samples"
        raise ValueError(msg)
    # Wind speed and rotor speed must be finite and non-negative; blade pitch only
    # finite (negative collective pitch is physical). The same policy drives the CSV
    # path via ``_PERFORMANCE_QUANTITIES``, so the two cannot drift apart.
    for label, quantity, array in (
        ("wind", "wind_speed", wind),
        ("rotor", "rotor_speed_rpm", rotor),
        ("pitch", "blade_pitch_deg", pitch),
    ):
        if not np.all(np.isfinite(array)):
            msg = f"performance_table {label} values must all be finite"
            raise ValueError(msg)
        if _PERFORMANCE_QUANTITIES[quantity] and np.any(array < 0.0):
            msg = f"performance_table {label} values must be non-negative"
            raise ValueError(msg)
    order = np.argsort(wind, kind="stable")
    wind, rotor, pitch = wind[order], rotor[order], pitch[order]
    if not np.all(np.diff(wind) > 0.0):
        msg = "performance_table wind speeds must be distinct (no duplicate wind axis)"
        raise ValueError(msg)
    return wind.tolist(), rotor.tolist(), pitch.tolist()


# Canonical rotor-performance CSV column names and the header variants accepted for
# each. Header matching is case-insensitive and ignores surrounding whitespace, the unit
# bracket, and separators (``_``/``-``/space), so "wind_speed", "WindSpeed", and
# "V [m/s]" all resolve to the wind-speed column. The three structural columns are
# required; cp/ct/cq/gen_power and any other columns are tolerated and ignored.
_CSV_REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "wind_speed": ("windspeed", "wind", "v", "u", "vhub", "windspeedms"),
    "rotor_speed_rpm": ("rotorspeed", "rotspeed", "omega", "rpm", "rotorspeedrpm"),
    "blade_pitch_deg": ("bladepitch", "bldpitch", "pitch", "theta", "bladepitchdeg"),
}

# Units accepted in the bracket (``[...]`` or ``(...)``) of each required column, as
# normalized spelling -> multiplicative factor to the canonical unit (m/s, rpm, deg).
# A column without a unit bracket is taken to be in the canonical unit already; any
# other bracketed unit is rejected rather than silently misread.
_CSV_COLUMN_UNITS: dict[str, dict[str, float]] = {
    "wind_speed": {
        "m/s": 1.0,
        "mps": 1.0,
        "ms^-1": 1.0,
        "ms-1": 1.0,
        "m/sec": 1.0,
        "km/h": 1.0 / 3.6,
        "kmh": 1.0 / 3.6,
        "kph": 1.0 / 3.6,
        "km/hr": 1.0 / 3.6,
        "kmh^-1": 1.0 / 3.6,
        "kmh-1": 1.0 / 3.6,
    },
    "rotor_speed_rpm": {
        "rpm": 1.0,
        "rev/min": 1.0,
        "r/min": 1.0,
        "revs/min": 1.0,
        "rad/s": 60.0 / (2.0 * math.pi),
        "rad/sec": 60.0 / (2.0 * math.pi),
        "rads^-1": 60.0 / (2.0 * math.pi),
        "rads-1": 60.0 / (2.0 * math.pi),
    },
    "blade_pitch_deg": {
        "deg": 1.0,
        "degree": 1.0,
        "degrees": 1.0,
        "°": 1.0,
        "rad": 180.0 / math.pi,
        "radian": 180.0 / math.pi,
        "radians": 180.0 / math.pi,
    },
}

# Header names whose *name itself* states the canonical unit (e.g. "rotor_speed_rpm").
# A bracketed unit that converts from a different unit contradicts the name, and the
# column is rejected rather than guessed at.
_CSV_NAME_IMPLIES_CANONICAL_UNIT: frozenset[str] = frozenset(
    {"windspeedms", "rpm", "rotorspeedrpm", "bladepitchdeg"}
)


def _split_header(name: str) -> tuple[str, str | None]:
    """Split a CSV header into its name part and its bracketed unit, if any.

    The unit is the text inside the first ``[...]`` or ``(...)`` group; an unclosed
    bracket takes the rest of the header. A header without a bracket has no unit.
    """
    cut = min((i for i in (name.find("["), name.find("(")) if i >= 0), default=-1)
    if cut < 0:
        return name, None
    rest = name[cut + 1 :]
    close = min((i for i in (rest.find("]"), rest.find(")")) if i >= 0), default=-1)
    unit = rest if close < 0 else rest[:close]
    return name[:cut], unit.strip()


def _normalize_header(name: str) -> str:
    """Reduce a CSV header to a comparable key: lowercase, alphanumerics only.

    Drops whitespace, the unit bracket (``[m/s]``), and separators so a column name
    and its variants compare equal (e.g. ``"Wind Speed [m/s]"`` -> ``"windspeed"``).
    The unit itself is parsed separately by :func:`_csv_unit_factor`.
    """
    name_part, _unit = _split_header(name)
    return "".join(ch for ch in name_part.lower() if ch.isalnum())


def _normalize_unit(unit: str) -> str:
    """Reduce a unit spelling to a comparable key (case/whitespace-insensitive)."""
    key = "".join(unit.lower().split())
    for separator in ("·", "*", "."):
        key = key.replace(separator, "")
    return key.replace("⁻¹", "^-1")


def _csv_unit_factor(canonical: str, header_name: str) -> float:
    """Return the factor converting a required column to its canonical unit.

    Parameters
    ----------
    canonical : str
        Canonical column name (a key of ``_CSV_REQUIRED_COLUMNS``).
    header_name : str
        The CSV header cell matched to ``canonical``.

    Returns
    -------
    float
        Multiplicative factor to m/s, rpm, or deg; ``1.0`` when the header carries no
        unit or already states the canonical unit.

    Raises
    ------
    ValueError
        If the bracketed unit is not an accepted spelling for the column, or it
        converts from a unit the header name itself contradicts.
    """
    _name, unit = _split_header(header_name)
    if unit is None or not unit:
        return 1.0
    accepted = _CSV_COLUMN_UNITS[canonical]
    factor = accepted.get(_normalize_unit(unit))
    if factor is None:
        msg = (
            f"rotor-performance CSV column {header_name!r} ({canonical}) has "
            f"unsupported unit {unit!r}; accepted units: {sorted(accepted)} "
            "(or no unit, meaning the canonical unit)"
        )
        raise ValueError(msg)
    if factor != 1.0 and _normalize_header(header_name) in (
        _CSV_NAME_IMPLIES_CANONICAL_UNIT
    ):
        msg = (
            f"rotor-performance CSV column {header_name!r} ({canonical}) names its "
            f"canonical unit but is bracketed as {unit!r}; rename the column or fix "
            "the unit"
        )
        raise ValueError(msg)
    if factor != 1.0:
        logger.info(
            "rotor-performance CSV column %r (%s): converting %s by factor %.12g",
            header_name,
            canonical,
            unit,
            factor,
        )
    return factor


def _resolve_csv_columns(header: Sequence[str]) -> dict[str, int]:
    """Map each canonical required column to its index in ``header``.

    Raises
    ------
    ValueError
        If a required column cannot be matched, or two header columns match the same
        canonical column ambiguously.
    """
    normalized = [_normalize_header(name) for name in header]
    resolved: dict[str, int] = {}
    for canonical, variants in _CSV_REQUIRED_COLUMNS.items():
        accepted = {canonical.replace("_", ""), *variants}
        matches = [i for i, key in enumerate(normalized) if key in accepted]
        if not matches:
            msg = (
                f"rotor-performance CSV is missing the required {canonical!r} column; "
                f"accepted header variants include {sorted(accepted)}. Found columns: "
                f"{list(header)}"
            )
            raise ValueError(msg)
        if len(matches) > 1:
            cols = ", ".join(repr(header[i]) for i in matches)
            msg = (
                f"rotor-performance CSV has multiple columns matching {canonical!r} "
                f"({cols}); rename so only one matches"
            )
            raise ValueError(msg)
        resolved[canonical] = matches[0]
    return resolved


# The optional power- and thrust-coefficient columns of a rotor-performance CSV: the
# normalized header names accepted for each, the units a header bracket may declare
# (as the factor to a fraction) and the largest magnitude a fraction can have. A power
# coefficient cannot exceed the Betz limit 16/27; a thrust coefficient cannot exceed 2,
# its value at an axial induction of one under the high-induction correction of
# blade-element momentum theory. Both bounds lie far below the percentage values of a
# working rotor, which is how a percentage written without its unit is recognised.
_COEFFICIENT_HEADERS: dict[str, frozenset[str]] = {
    "cp": frozenset({"cp", "powercoefficient", "rtaerocp", "rtfldcp"}),
    "ct": frozenset({"ct", "thrustcoefficient", "rtaeroct", "rtfldct"}),
}
_COEFFICIENT_UNITS: dict[str, float] = {
    "": 1.0,
    "-": 1.0,
    "1": 1.0,
    "fraction": 1.0,
    "%": 0.01,
    "percent": 0.01,
    "pct": 0.01,
}
_COEFFICIENT_LIMITS: dict[str, float] = {"cp": 16.0 / 27.0, "ct": 2.0}
_COEFFICIENT_LIMIT_NAMES: dict[str, str] = {
    "cp": "the Betz limit 16/27",
    "ct": "2, the high-induction limit of blade-element momentum theory",
}


@dataclass(frozen=True)
class _CoefficientColumn:
    """A power- or thrust-coefficient column of a rotor-performance CSV.

    ``factor`` converts the column's values to a fraction; ``percent`` records that
    the header declared a percentage.
    """

    key: str
    index: int
    header: str
    factor: float
    percent: bool


def _coefficient_column(
    header: Sequence[str], key: str, source: object
) -> _CoefficientColumn | None:
    """Find the ``cp`` or ``ct`` column of a CSV header and read its declared unit.

    Parameters
    ----------
    header : Sequence[str]
        The CSV header row.
    key : str
        ``"cp"`` or ``"ct"``.
    source : object
        The CSV path, for error messages.

    Returns
    -------
    _CoefficientColumn or None
        The column, or ``None`` when the header has none.

    Raises
    ------
    ValueError
        If several columns match ``key``, or the column declares a unit other than a
        fraction (no unit, ``-``, ``1`` or ``fraction``) or a percentage (``%``,
        ``percent`` or ``pct``).
    """
    accepted = _COEFFICIENT_HEADERS[key]
    matches = [
        i for i, name in enumerate(header) if _normalize_header(name) in accepted
    ]
    if len(matches) > 1:
        names = ", ".join(repr(header[i]) for i in matches)
        msg = (
            f"rotor-performance CSV {source} has several {key} columns ({names}); "
            "keep one"
        )
        raise ValueError(msg)
    if not matches:
        return None
    index = matches[0]
    _name, unit = _split_header(header[index])
    unit_key = "" if unit is None else _normalize_unit(unit)
    factor = _COEFFICIENT_UNITS.get(unit_key)
    if factor is None:
        msg = (
            f"rotor-performance CSV {source} column {header[index]!r} has unsupported "
            f"unit {unit!r}; a {key} column is a fraction (no unit, [-], [1] or "
            "[fraction]) or a percentage ([%], [percent] or [pct])"
        )
        raise ValueError(msg)
    return _CoefficientColumn(
        key=key, index=index, header=header[index], factor=factor, percent=factor != 1.0
    )


def _coefficient_cell(row: Sequence[str], index: int) -> float:
    """Parse one optional coefficient cell; blank or non-numeric is ``NaN``."""
    if index >= len(row):
        return math.nan
    try:
        value = float(row[index].strip())
    except ValueError:
        return math.nan
    return value if math.isfinite(value) else math.nan


def _coefficient_fractions(
    column: _CoefficientColumn, values: Sequence[float], source: object
) -> list[float]:
    """Convert a coefficient column to fractions, rejecting an ambiguous scale.

    ``NaN`` (unknown) values pass through unchanged and are not checked.

    Raises
    ------
    ValueError
        If the column declares a percentage but every non-zero value is at most 1 in
        magnitude (fractions labelled as percent), or a converted value exceeds the
        coefficient's physical bound (a percentage without its unit, or a wrong one).
    """
    finite = [abs(v) for v in values if math.isfinite(v)]
    if column.percent and finite and 0.0 < max(finite) <= 1.0:
        msg = (
            f"rotor-performance CSV {source} column {column.header!r} declares a "
            f"percentage, but every {column.key} value is at most 1 in magnitude, "
            "which reads as a fraction; remove the percent unit or give the values "
            "in percent"
        )
        raise ValueError(msg)
    converted = [v * column.factor for v in values]
    limit = _COEFFICIENT_LIMITS[column.key]
    largest = max((abs(v) for v in converted if math.isfinite(v)), default=0.0)
    if largest > limit:
        hint = (
            "check the values and their unit"
            if column.percent
            else f"if the column is in percent, declare it as '{column.key} [%]'"
        )
        msg = (
            f"rotor-performance CSV {source} column {column.header!r} holds a "
            f"{column.key} of {largest:.6g} as a fraction, above "
            f"{_COEFFICIENT_LIMIT_NAMES[column.key]}; {hint}"
        )
        raise ValueError(msg)
    if column.percent:
        logger.info(
            "rotor-performance CSV column %r (%s): converting percent to a fraction",
            column.header,
            column.key,
        )
    return converted


def operating_curve_from_csv(
    path: str | Path,
) -> tuple[list[float], list[float], list[float]]:
    """Parse a rotor-performance CSV into a steady operating curve.

    Reads a header-row CSV of the turbine's steady operating schedule and returns the
    ``(curve_wind, curve_rotor_speed, curve_pitch)`` arrays
    :func:`~vane.run.run_operating_sweep` and :func:`operating_points_from_curve`
    consume, sorted by ascending wind speed.

    The CSV must have a header row naming its columns. Three columns are **required**
    (canonical names, with tolerated variants — header matching ignores case,
    whitespace, and separators):

    * ``wind_speed`` (m/s) -- also ``WindSpeed``, ``wind``, ``V [m/s]``, ``U``;
    * ``rotor_speed_rpm`` (rpm) -- also ``RotSpeed``, ``Omega [rpm]``, ``rpm``;
    * ``blade_pitch_deg`` (deg) -- also ``BldPitch``, ``pitch [deg]``, ``theta``.

    A unit written in square brackets or parentheses after a required column's name
    is parsed (case- and whitespace-insensitive) and the column converted to the
    canonical unit; a column without a unit is taken to be in the canonical unit:

    * wind speed accepts ``m/s`` (also ``mps``, ``m s^-1``) and converts ``km/h``
      (also ``kmh``, ``kph``) by 1/3.6;
    * rotor speed accepts ``rpm`` (also ``rev/min``) and converts ``rad/s`` by
      60/(2*pi);
    * blade pitch accepts ``deg`` (also ``degree``, ``degrees``) and converts
      ``rad`` by 180/pi.

    Any other unit on a required column raises :class:`ValueError` rather than being
    silently misread, as does a converting unit on a header whose name states the
    canonical unit (``rotor_speed_rpm [rad/s]``). Each conversion is logged at INFO.
    Units are read from the header only; a separate units row is not supported and
    fails as a non-numeric value. Any other columns (``cp``, ``ct``, ``cq``,
    ``gen_power``, ...) are ignored here, units included, so a standard
    rotor-performance export drops in unchanged;
    :func:`~vane.run.rotor_performance_from_csv` reads the ``cp`` and ``ct`` columns
    with their units.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the rotor-performance CSV file.

    Returns
    -------
    tuple of three list[float]
        ``(curve_wind, curve_rotor_speed, curve_pitch)``, sorted by ascending wind
        speed.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file is unreadable or empty, a required column is missing or carries
        an unsupported (or name-contradicting) unit, a value is
        non-numeric, NaN, or infinite, a wind speed or rotor speed is negative (negative
        collective blade pitch is allowed), there are fewer than two data rows (at least
        two operating points are needed to interpolate), or the wind speeds are
        duplicated or not strictly increasing once sorted.
    """
    import csv

    csv_path = Path(path)
    if not csv_path.is_file():
        msg = f"rotor-performance CSV not found: {csv_path}"
        raise FileNotFoundError(msg)
    try:
        text = csv_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as error:
        msg = f"rotor-performance CSV {csv_path} could not be read: {error}"
        raise ValueError(msg) from error

    reader = csv.reader(text.splitlines())
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        msg = f"rotor-performance CSV {csv_path} is empty (no header row)"
        raise ValueError(msg)
    header, data_rows = rows[0], rows[1:]
    columns = _resolve_csv_columns(header)
    if not data_rows:
        msg = f"rotor-performance CSV {csv_path} has a header but no data rows"
        raise ValueError(msg)
    if len(data_rows) < 2:
        msg = (
            f"rotor-performance CSV {csv_path} needs at least two operating points "
            f"(data rows) for interpolation; got {len(data_rows)}"
        )
        raise ValueError(msg)

    factors = {
        canonical: _csv_unit_factor(canonical, header[index])
        for canonical, index in columns.items()
    }
    wind, rotor, pitch = _read_csv_values(csv_path, header, data_rows, columns, factors)
    return _finalize_csv_curve(csv_path, wind, rotor, pitch)


def _read_csv_values(
    csv_path: Path,
    header: Sequence[str],
    data_rows: Sequence[Sequence[str]],
    columns: dict[str, int],
    factors: dict[str, float],
) -> tuple[list[float], list[float], list[float]]:
    """Extract, unit-convert, and validate the three required columns."""
    wind: list[float] = []
    rotor: list[float] = []
    pitch: list[float] = []
    n_cols = len(header)
    for line_no, row in enumerate(data_rows, start=2):
        if len(row) < n_cols:
            msg = (
                f"rotor-performance CSV {csv_path} line {line_no} has {len(row)} "
                f"fields but the header has {n_cols}"
            )
            raise ValueError(msg)
        for canonical, target in (
            ("wind_speed", wind),
            ("rotor_speed_rpm", rotor),
            ("blade_pitch_deg", pitch),
        ):
            raw = row[columns[canonical]].strip()
            try:
                value = float(raw)
            except ValueError as error:
                hint = (
                    "; a units row is not supported, so give each unit in the header "
                    "(e.g. 'wind_speed [m/s]')"
                    if line_no == 2
                    else ""
                )
                msg = (
                    f"rotor-performance CSV {csv_path} line {line_no}: {canonical} "
                    f"value {raw!r} is not numeric{hint}"
                )
                raise ValueError(msg) from error
            if not np.isfinite(value):
                msg = (
                    f"rotor-performance CSV {csv_path} line {line_no}: {canonical} "
                    f"value {raw!r} is not finite"
                )
                raise ValueError(msg)
            # Wind/rotor speed are physically non-negative; collective blade pitch can
            # be negative (a fine-pitch schedule). The same policy validates the table
            # path, via the shared _PERFORMANCE_QUANTITIES.
            if _PERFORMANCE_QUANTITIES[canonical] and value < 0.0:
                msg = (
                    f"rotor-performance CSV {csv_path} line {line_no}: {canonical} "
                    f"value {value} is negative"
                )
                raise ValueError(msg)
            with np.errstate(over="ignore"):
                converted = float(value * factors[canonical])
            if not np.isfinite(converted):
                msg = (
                    f"rotor-performance CSV {csv_path} line {line_no}: {canonical} "
                    f"value {raw!r} overflows on conversion to the canonical unit"
                )
                raise ValueError(msg)
            target.append(converted)
    return wind, rotor, pitch


def _finalize_csv_curve(
    csv_path: Path,
    wind: list[float],
    rotor: list[float],
    pitch: list[float],
) -> tuple[list[float], list[float], list[float]]:
    """Sort the parsed curve by wind and reject a duplicate/non-monotone wind axis.

    The caller has already enforced at least two data rows, so the strictly-increasing
    check below always has a gap to test.
    """
    wind_arr = np.asarray(wind, dtype=np.float64)
    order = np.argsort(wind_arr, kind="stable")
    if not np.all(np.diff(wind_arr[order]) > 0.0):
        msg = (
            f"rotor-performance CSV {csv_path} wind speeds must be distinct and "
            "strictly increasing once sorted (no duplicate wind axis)"
        )
        raise ValueError(msg)
    sorted_wind = [wind[i] for i in order]
    sorted_rotor = [rotor[i] for i in order]
    sorted_pitch = [pitch[i] for i in order]
    return sorted_wind, sorted_rotor, sorted_pitch
