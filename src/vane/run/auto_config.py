# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Infer a linearization sweep's turbine constants from the user's deck and curve.

:func:`~vane.run.run_operating_sweep` needs a few turbine constants that a deck does
not state in one place: the rated wind speed that separates torque trim from pitch
trim, the rated generator torque the trim holds above rated and derives its torque
gain from, a pitch-trim gain suited to the rotor's size, and a module configuration
OpenFAST can linearize. This module reads them from the **user's own inputs** -- the
operating curve they supplied and the deck's ElastoDyn, ServoDyn, AeroDyn, HydroDyn
and controller input files -- so a sweep runs with no other arguments. VANE still
computes no rotor performance: every value is a constant read from, or arithmetic on,
the user's data, and each one is logged and recorded with its source
(:class:`InferredValue`, collected in :class:`AutoConfiguration`) so the run
provenance says where it came from. A value the user sets explicitly is never
replaced.

The rules (see :func:`infer_rated_wind_speed`, :func:`infer_rated_generator_torque`
and :func:`scaled_trim_gain_pitch` for the evidence behind each):

* **rated wind speed** -- the first curve wind at which the rotor speed has reached
  its plateau and, when the curve has a ``cp`` column, the aerodynamic power
  ``cp * V**3`` has reached its plateau too. Cross-checked against the controller's
  rated speed and power (``DISCON.IN``) when present; a disagreement is logged.
* **rated generator torque** -- the ``DISCON.IN`` ``VS_RtTq`` (the file named by
  ServoDyn ``DLL_InFile``) when the deck runs a controller library; else ServoDyn
  ``VS_RtTq`` when ``VSContrl = 1``; else the curve's aerodynamic torque at rated
  from its ``cp`` column, ElastoDyn ``GBRatio``/``GBoxEff`` and the deck's air density.
* **pitch trim gain** -- ``1e-3 * (63 m / R)**3`` with ``R`` the ElastoDyn ``TipRad``.
* **module configuration** (only when no settings are passed) -- the
  :meth:`~vane.run.LinearizationSettings.offshore` preset adapted to the deck:
  a missing ``.ss`` radiation file is fitted from the WAMIT ``.1`` data in each case
  copy when the fit is accurate enough (``hydro_radiation_fit``), other missing
  state-space hydrodynamics files fall back to ``RdtnMod``/``ExctnMod`` 0, a
  deck's own linearizable unsteady-aerodynamics model is kept (with ``UAEndRad`` 0.99
  where the tip node would otherwise carry it), and modules the deck does not use are
  left untouched.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from vane.io.fst_reader import _fortran_true, fortran_float
from vane.io.fst_writer import FastInputError, FastInputFile
from vane.run.lin_config import (
    DLL_CONTROL_KEYS,
    FLEXIBILITY_DOFS,
    PERIODIC_TRIM_TOL,
    LinearizationSettings,
    is_size_scaled_default,
)
from vane.run.operating_points import (
    _coefficient_cell,
    _coefficient_column,
    _coefficient_fractions,
    _csv_unit_factor,
    _resolve_csv_columns,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "DEFAULT_TRIM_GAIN_PITCH",
    "DEFAULT_TRIM_TOL",
    "MAX_SCALED_TRIM_TOL",
    "REFERENCE_ROTOR_RADIUS",
    "AutoConfiguration",
    "ControllerConstants",
    "InferredValue",
    "auto_configure_sweep",
    "curve_power_coefficient",
    "infer_rated_generator_torque",
    "infer_rated_wind_speed",
    "read_controller_constants",
    "rotor_radius",
    "scaled_trim_gain_pitch",
    "scaled_trim_tol",
]

logger = logging.getLogger(__name__)

#: The class-default pitch trim gain (rad/(rad/s)); a value left here is scaled.
DEFAULT_TRIM_GAIN_PITCH = float(LinearizationSettings().trim_gain_pitch)
#: The class-default trim tolerance; a value left here is scaled.
DEFAULT_TRIM_TOL = float(LinearizationSettings().trim_tol)
#: Upper bound of the scaled trim tolerance. The MBC3 transform accepts a rotor-speed
#: spread of 1 % across the azimuth sweep, which a looser trim can exceed.
MAX_SCALED_TRIM_TOL = PERIODIC_TRIM_TOL
#: Rotor radius (m) at which the default trim gain and tolerance apply (NREL 5-MW).
REFERENCE_ROTOR_RADIUS = 63.0

# Relative band within which a curve value counts as on its plateau. The published
# schedules hold the rated rotor speed to six or more significant figures, so 0.5 %
# separates the plateau from the last region-2 point on every reference curve
# (closest: the NREL 5-MW published curve, 11.84 vs 12.14 rpm, 2.5 %).
PLATEAU_TOLERANCE = 0.005
# Disagreements between the curve and the controller file above these are logged.
_SPEED_AGREEMENT = 0.02
_POWER_AGREEMENT = 0.05
# UAEndRad written when a kept unsteady-aerodynamics model would reach the tip node.
_UA_END_RADIUS = 0.99
_LINEARIZABLE_UNSTEADY = frozenset({4, 5, 6})
_DEFAULT_AIR_DENSITY = 1.225


@dataclass(frozen=True)
class InferredValue:
    """One value VANE inferred or adjusted, with where it came from.

    Parameters
    ----------
    name : str
        What was inferred: a :class:`~vane.run.LinearizationSettings` field name
        (``rated_wind_speed``, ``hydro_radiation_mode``, ...) or ``template_root`` /
        ``executable``.
    value : float or int or str or None
        The value used.
    source : str
        The file and keyword, or the rule, it was taken from.
    detail : str, optional
        Further evidence (cross-checks, the numbers combined).
    """

    name: str
    value: float | int | str | None
    source: str
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        """Return the record as a JSON-serializable dictionary.

        Returns
        -------
        dict[str, object]
            ``name``, ``value``, ``source`` and ``detail``.
        """
        return {
            "name": self.name,
            "value": self.value,
            "source": self.source,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class AutoConfiguration:
    """Everything a sweep inferred or adjusted on its own, for the provenance.

    Parameters
    ----------
    inferred : tuple[InferredValue, ...]
        The inferred values, in the order they were decided.
    warnings : tuple[str, ...]
        Disagreements and fallbacks the user should know about (also logged).
    """

    inferred: tuple[InferredValue, ...] = ()
    warnings: tuple[str, ...] = ()

    def get(self, name: str) -> InferredValue | None:
        """Return the last record named ``name``, or ``None``.

        Parameters
        ----------
        name : str
            The record name.

        Returns
        -------
        InferredValue or None
            The record, when one was made.
        """
        for item in reversed(self.inferred):
            if item.name == name:
                return item
        return None

    def extended(
        self,
        inferred: Sequence[InferredValue] = (),
        warnings: Sequence[str] = (),
    ) -> AutoConfiguration:
        """Return a copy with more records and warnings appended.

        Parameters
        ----------
        inferred : Sequence[InferredValue], optional
            Records to append.
        warnings : Sequence[str], optional
            Warnings to append.

        Returns
        -------
        AutoConfiguration
            The extended configuration.
        """
        return AutoConfiguration(
            inferred=(*self.inferred, *inferred),
            warnings=(*self.warnings, *warnings),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the configuration as a JSON-serializable dictionary.

        Returns
        -------
        dict[str, object]
            ``inferred`` (a list of records) and ``warnings``.
        """
        return {
            "inferred": [item.to_dict() for item in self.inferred],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class ControllerConstants:
    """Rated constants read from a ``DISCON.IN``-style controller input file.

    Parameters
    ----------
    path : pathlib.Path
        The controller input file (ServoDyn ``DLL_InFile``).
    rated_power : float or None
        ``VS_RtPwr`` (W, electrical).
    rated_torque : float or None
        ``VS_RtTq`` (N·m, generator/HSS side).
    rated_generator_speed : float or None
        ``VS_RefSpd`` (else ``PC_RefSpd``), rad/s on the HSS.
    generator_efficiency : float or None
        ``VS_GenEff`` as a fraction (the file holds percent).
    blade_radius : float or None
        ``WE_BladeRadius`` (m).
    """

    path: Path
    rated_power: float | None = None
    rated_torque: float | None = None
    rated_generator_speed: float | None = None
    generator_efficiency: float | None = None
    blade_radius: float | None = None


def _float(file: FastInputFile | None, keyword: str) -> float | None:
    """Return a finite numeric keyword of ``file``, or ``None``."""
    if file is None or keyword not in file:
        return None
    try:
        value = fortran_float(file.get(keyword))
    except (FastInputError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _flag(file: FastInputFile | None, keyword: str) -> bool:
    """Return whether a flag keyword of ``file`` is present and true.

    The value is read as a Fortran logical (``True``, ``T``, ``.true.``, ...).
    """
    if file is None or keyword not in file:
        return False
    try:
        return _fortran_true(file.get(keyword))
    except FastInputError:
        return False


def _module_on(deck: FastInputFile, switch: str) -> bool:
    """Return whether a primary-deck module switch is not explicitly 0."""
    value = _float(deck, switch)
    return value is None or value != 0.0


def _subfile(
    deck: FastInputFile, key: str, switch: str | None = None
) -> FastInputFile | None:
    """Return a readable sub-file of an enabled module, or ``None``."""
    if switch is not None and not _module_on(deck, switch):
        return None
    if key not in deck:
        return None
    try:
        path = deck.resolve(key)
        return FastInputFile.read(path) if path.is_file() else None
    except FastInputError:
        return None


def _parse_controller_file(path: Path) -> dict[str, str]:
    """Return ``keyword -> first value`` of a ``value ! Keyword`` controller file."""
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return values
    for line in text.splitlines():
        head, separator, comment = line.partition("!")
        if not separator:
            continue
        tokens = head.split()
        names = comment.split()
        if tokens and names:
            values.setdefault(names[0], tokens[0].strip("\"'"))
    return values


def _positive(values: dict[str, str], keyword: str) -> float | None:
    """Return a positive finite controller value, or ``None``."""
    raw = values.get(keyword)
    if raw is None:
        return None
    try:
        value = fortran_float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) and value > 0.0 else None


def _uses_controller_library(servo: FastInputFile) -> bool:
    """Return whether ServoDyn runs any control through the Bladed-style library."""
    return any(_float(servo, key) == 5.0 for key in DLL_CONTROL_KEYS)


def read_controller_constants(fst_path: str | Path) -> ControllerConstants | None:
    """Return the rated constants of the deck's controller input file, if any.

    The file is ServoDyn's ``DLL_InFile`` (resolved relative to the ServoDyn file),
    read only when ServoDyn is enabled and runs a control channel through the
    Bladed-style library (``PCMode``, ``VSContrl``, ``HSSBrMode`` or ``YCMode`` = 5).
    The ``value ! Keyword - description`` layout of ``DISCON.IN`` is parsed;
    ``VS_RtPwr``, ``VS_RtTq``, ``VS_RefSpd`` (or ``PC_RefSpd``), ``VS_GenEff`` and
    ``WE_BladeRadius`` are read.

    Parameters
    ----------
    fst_path : str or pathlib.Path
        The primary ``.fst`` deck.

    Returns
    -------
    ControllerConstants or None
        The constants (each ``None`` when absent or not positive), or ``None`` when
        the deck has no readable controller input file or it holds none of them.
    """
    try:
        deck = FastInputFile.read(fst_path)
    except FastInputError:
        return None
    servo = _subfile(deck, "ServoFile", "CompServo")
    if servo is None or not _uses_controller_library(servo):
        return None
    if "DLL_InFile" not in servo:
        return None
    try:
        path = servo.resolve("DLL_InFile")
    except FastInputError:
        return None
    if not path.is_file():
        return None
    values = _parse_controller_file(path)
    efficiency = _positive(values, "VS_GenEff")
    constants = ControllerConstants(
        path=path,
        rated_power=_positive(values, "VS_RtPwr"),
        rated_torque=_positive(values, "VS_RtTq"),
        rated_generator_speed=_positive(values, "VS_RefSpd")
        or _positive(values, "PC_RefSpd"),
        generator_efficiency=None if efficiency is None else efficiency / 100.0,
        blade_radius=_positive(values, "WE_BladeRadius"),
    )
    if constants == ControllerConstants(path=path):
        return None
    return constants


def rotor_radius(fst_path: str | Path) -> float | None:
    """Return the rotor radius (ElastoDyn ``TipRad``, m) of a deck, if readable.

    Parameters
    ----------
    fst_path : str or pathlib.Path
        The primary ``.fst`` deck.

    Returns
    -------
    float or None
        ``TipRad``, or ``None`` when the deck has no readable positive value.
    """
    try:
        deck = FastInputFile.read(fst_path)
    except FastInputError:
        return None
    radius = _float(_subfile(deck, "EDFile"), "TipRad")
    return radius if radius is not None and radius > 0.0 else None


def _enabled_flexibility(
    fst_path: str | Path, settings: LinearizationSettings
) -> list[str]:
    """Return the flexibility DOFs the deck has off and the settings turn on."""
    try:
        deck = FastInputFile.read(fst_path)
    except FastInputError:
        return []
    elastodyn = _subfile(deck, "EDFile")
    if elastodyn is None:
        return []
    resolved = settings.resolved_structural_dofs()
    return [
        flag
        for flag in FLEXIBILITY_DOFS
        if resolved.get(flag) and flag in elastodyn and not _flag(elastodyn, flag)
    ]


def scaled_trim_gain_pitch(radius: float) -> float:
    """Return the default CalcSteady pitch trim gain for a rotor of ``radius``.

    The rule is ``2e-4 * (63 m / R)**3`` rad/(rad/s).

    Parameters
    ----------
    radius : float
        Rotor radius ``R`` (m), ElastoDyn ``TipRad``.

    Returns
    -------
    float
        The pitch trim gain (``TrimGain`` for ``TrimCase = 3``).

    Raises
    ------
    ValueError
        If ``radius`` is not finite and positive.

    Notes
    -----
    ServoDyn's pitch trim is a PI law on the rotor-speed error ``e`` (rad/s): each
    step adds ``TrimGain * e`` to an integrator and the pitch offset adds
    ``1000 * TrimGain * e`` on top, so the proportional gain is ``1000 * TrimGain``
    and the integral gain ``TrimGain / DT``. The aerodynamic torque a pitch change
    produces, ``dQ/dtheta``, scales with ``rho * R**3 * V**2`` at the similar rated
    wind speeds of utility turbines, so holding the trim's torque correction per unit
    speed error (``1000 * TrimGain * dQ/dtheta``) at its NREL 5-MW value gives
    ``TrimGain`` proportional to ``R**-3``. It is anchored at the class default 2e-4
    for R = 63 m. The ElastoDyn rotor inertia about the low-speed shaft (blades, hub
    and generator) grows nearly as fast, as ``R**3.4`` over the NREL 5-MW and IEA
    3.4/10/15/22-MW turbines (4.4e7 to 7.1e8 kg·m²), so the rule also keeps
    ``TrimGain`` times that inertia within a factor of two of the NREL 5-MW value on
    all five.

    Single-attempt CalcSteady trims give the evidence. The anchor was 1e-3 until the
    reference-turbine campaign showed that the r-test NREL 5-MW land deck (63 m,
    ``DT`` 6.25 ms) oscillates or motors above rated with it: at 16 m/s 1e-3 keeps
    oscillating while 6.3e-4, 5e-4 and 2e-4 converge, and at 13 m/s, just above
    rated, everything above 2e-4 motors. With the 2e-4 anchor every NREL 5-MW point,
    the IEA 3.4-MW (64.9 m) from 9 to 25 m/s and every IEA 15-MW (121.0 m) and
    22-MW (141.0 m) point above rated trim on the first attempt. Before that, the
    1e-3 anchor diverged on the IEA 10-, 15- and 22-MW, and its scaled 2.6e-4
    converged the IEA 10-MW (99.2 m) at 12 m/s.
    """
    if not (math.isfinite(radius) and radius > 0.0):
        msg = f"rotor radius must be finite and positive, got {radius}"
        raise ValueError(msg)
    return DEFAULT_TRIM_GAIN_PITCH * (REFERENCE_ROTOR_RADIUS / radius) ** 3


def scaled_trim_tol(radius: float) -> float:
    """Return the default CalcSteady trim tolerance for a rotor of ``radius``.

    The rule is ``1e-3 * (R / 63 m)**3``, kept within ``[1e-3, 1e-2]``.

    Parameters
    ----------
    radius : float
        Rotor radius ``R`` (m), ElastoDyn ``TipRad``.

    Returns
    -------
    float
        The trim tolerance (``TrimTol``).

    Raises
    ------
    ValueError
        If ``radius`` is not finite and positive.

    Notes
    -----
    ``TrimTol`` bounds the change of the trimmed outputs from one rotor revolution to
    the next. Without trim damping, the lightly damped tower and blade motion of a
    large, flexible turbine decays too slowly for a tight tolerance, while the rotor
    speed of a small, light rotor needs one: the MBC3 transform accepts a rotor-speed
    spread of at most 1 % across the azimuth sweep.

    Single-attempt trims on the reference-turbine campaign bracket the rule. At 1e-3
    the IEA 3.4-MW (64.9 m) and the NREL 5-MW (63 m) converge every point, but the
    IEA 15-MW (121.0 m) converges only 4 of 9 and the IEA 22-MW (141.0 m) 2 of 9. At
    1e-2 the IEA 15-MW converges 8 of 9 and the IEA 22-MW 9 of 9, while the IEA
    3.4-MW's below-rated points stop with a rotor-speed spread of 1.02--2.57 % that
    the MBC3 check rejects. Against the points that converge at both tolerances,
    1e-2 changes the frequencies below 3 Hz by at most 0.1 %, the damping ratios by
    at most 0.1 percentage point and the trimmed generator torque by at most 0.8 %.
    The rule gives 1e-3 to 1.1e-3 for the 63--65 m rotors, 3.9e-3 for the IEA 10-MW
    (99 m), 7.1e-3 for the IEA 15-MW and the 1e-2 bound for the IEA 22-MW. A trim
    that still does not converge is retried with a looser tolerance and a longer run.
    """
    if not (math.isfinite(radius) and radius > 0.0):
        msg = f"rotor radius must be finite and positive, got {radius}"
        raise ValueError(msg)
    scaled = DEFAULT_TRIM_TOL * (radius / REFERENCE_ROTOR_RADIUS) ** 3
    return min(max(scaled, DEFAULT_TRIM_TOL), MAX_SCALED_TRIM_TOL)


def curve_power_coefficient(
    csv_path: str | Path, curve_wind: Sequence[float]
) -> list[float] | None:
    """Return a performance CSV's ``cp`` column aligned with ``curve_wind``.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        The rotor-performance CSV (as given to :func:`~vane.run.run_operating_sweep`).
    curve_wind : Sequence[float]
        The curve's wind speeds as :func:`~vane.run.operating_curve_from_csv`
        returned them (sorted ascending).

    Returns
    -------
    list[float] or None
        The power coefficient (a fraction) at each curve wind, or ``None`` when the
        file cannot be read, has no ``cp`` column, its rows do not match
        ``curve_wind``, or any ``cp`` value is missing, non-numeric or negative.

    Raises
    ------
    ValueError
        If the ``cp`` column's unit or scale is wrong or ambiguous, as for
        :func:`~vane.run.rotor_performance_from_csv`: the header accepts the same
        column names (``cp``, ``power_coefficient``, ``RtAeroCp``, ``RtFldCp``, only
        one of them) and units (none, ``[-]``, ``[1]`` or ``[fraction]`` for a
        fraction; ``[%]``, ``[percent]`` or ``[pct]`` for a percentage, which is
        converted), a fraction above the Betz limit 16/27 is rejected as a percentage
        written without its unit, and a column declared in percent whose values are
        all at most 1 is rejected as fractions labelled as percent.
    """
    try:
        rows = [
            row
            for row in csv.reader(
                Path(csv_path).read_text(encoding="utf-8-sig").splitlines()
            )
            if any(cell.strip() for cell in row)
        ]
    except (OSError, UnicodeDecodeError):
        return None
    if len(rows) < 2:
        return None
    header = rows[0]
    # A wrong or ambiguous unit is an input error, raised as for the verification
    # reader; a column that is merely absent or incomplete is not used.
    column = _coefficient_column(header, "cp", csv_path)
    if column is None:
        return None
    try:
        wind_index = _resolve_csv_columns(header)["wind_speed"]
        factor = _csv_unit_factor("wind_speed", header[wind_index])
        pairs = sorted(
            (float(row[wind_index]) * factor, _coefficient_cell(row, column.index))
            for row in rows[1:]
        )
    except (ValueError, IndexError):
        return None
    values = _coefficient_fractions(column, [cp for _, cp in pairs], csv_path)
    if len(pairs) != len(curve_wind) or any(
        not math.isclose(wind, expected, rel_tol=1e-9, abs_tol=1e-12)
        for (wind, _), expected in zip(pairs, curve_wind, strict=True)
    ):
        return None
    # A zero cp (a zero-wind or parked row) is valid data; the rated-power level is
    # checked to be positive where it is used.
    if any(not math.isfinite(cp) or cp < 0.0 for cp in values):
        return None
    return values


@dataclass(frozen=True)
class _Plateau:
    """Where the curve reaches rated, and the rated rotor speed and power level."""

    index: int
    rotor_speed: float
    power_level: float | None  # final cp * V**3 of the rated plateau


def _rated_plateau(
    curve_wind: Sequence[float],
    curve_rotor_speed: Sequence[float],
    curve_cp: Sequence[float] | None,
) -> _Plateau:
    """Return the first curve index on both the rotor-speed and power plateaus."""
    wind = np.asarray(curve_wind, dtype=np.float64)
    speed = np.asarray(curve_rotor_speed, dtype=np.float64)
    lengths = {wind.size, speed.size}
    if curve_cp is not None:
        lengths.add(len(curve_cp))
    if len(lengths) != 1:
        msg = (
            "the operating curve's wind, rotor speed and cp sequences must have the "
            f"same length, got {sorted(lengths)}"
        )
        raise ValueError(msg)
    order = np.argsort(wind)
    wind, speed = wind[order], speed[order]
    top = float(np.max(speed)) if speed.size else 0.0
    if not math.isfinite(top) or top <= 0.0:
        msg = (
            "the operating curve has no positive rotor speed, so it has no rated point"
        )
        raise ValueError(msg)
    on_plateau = speed >= (1.0 - PLATEAU_TOLERANCE) * top
    # The rotor-speed plateau starts where every later sample stays on it: a curve
    # that touches its maximum and falls away again has not reached rated there.
    held = np.flip(np.logical_and.accumulate(np.flip(on_plateau)))
    first = int(np.argmax(held))
    if int(np.count_nonzero(held)) < 2:
        msg = (
            "the operating curve's rotor speed never levels off at its maximum (fewer "
            "than two final samples within "
            f"{100 * PLATEAU_TOLERANCE:g} % of {top:.6g} rpm), so its rated point "
            "cannot be located"
        )
        raise ValueError(msg)
    if curve_cp is None:
        return _Plateau(index=int(order[first]), rotor_speed=top, power_level=None)
    cp = np.asarray(curve_cp, dtype=np.float64)[order]
    power = cp * wind**3
    tail = power[first:]
    # The plateau level is the curve's final power: the median of the whole
    # constant-speed region would include a power ramp that precedes the plateau.
    level = float(tail[-1])
    band = np.abs(tail - level) <= PLATEAU_TOLERANCE * abs(level)
    # The power plateau starts where every later sample stays in the band: a power
    # that is still rising (or falling) through the last samples has no plateau, and
    # the median of a rising tail would otherwise be reached by one of its samples.
    settled = np.flip(np.logical_and.accumulate(np.flip(band)))
    if not math.isfinite(level) or level <= 0.0 or int(np.count_nonzero(settled)) < 2:
        msg = (
            "the operating curve's aerodynamic power (cp * V**3) never levels off "
            "after its rotor speed does (fewer than two final samples within "
            f"{100 * PLATEAU_TOLERANCE:g} % of its final value), so its "
            "rated point cannot be located; extend the curve beyond rated"
        )
        raise ValueError(msg)
    index = first + int(np.argmax(settled))
    return _Plateau(index=int(order[index]), rotor_speed=top, power_level=level)


def infer_rated_wind_speed(
    curve_wind: Sequence[float],
    curve_rotor_speed: Sequence[float],
    curve_cp: Sequence[float] | None = None,
) -> tuple[float, str]:
    """Return the rated wind speed of an operating curve, and how it was found.

    Rated is the first curve wind at which the rotor speed has reached its plateau
    (within 0.5 % of the curve's maximum, held for at least two samples) and, when
    ``curve_cp`` is given, the aerodynamic power ``cp * V**3`` has also reached its
    plateau: from that wind on, every curve sample is within 0.5 % of the curve's
    final power (at least two samples). A power still rising through the last
    samples has no plateau.

    Parameters
    ----------
    curve_wind, curve_rotor_speed : Sequence[float]
        The operating curve (m/s, rpm).
    curve_cp : Sequence[float] or None, optional
        The curve's power coefficients, aligned with ``curve_wind``.

    Returns
    -------
    tuple[float, str]
        The rated wind speed (m/s) and a description of the evidence.

    Raises
    ------
    ValueError
        If the curve has no rotor-speed plateau (e.g. it stops below rated), or
        its sequences differ in length.

    Notes
    -----
    The rated wind speed selects the CalcSteady trim: below it the generator torque
    is trimmed at the curve's pitch, at and above it the pitch is trimmed against a
    constant rated torque. The boundary that matters is therefore where the
    generator torque reaches rated, i.e. where the rotor speed is at its rated value
    *and* the power has stopped rising. The point where the pitch starts to rise is
    not used: a peak-shaving schedule pitches before rated (the published NREL 5-MW
    curve from 11.1 m/s, while its rotor speed and power level off at 12.12 m/s), and
    a fine-pitch schedule pitches below rated too. On the published reference curves
    the rule gives 12.12 m/s (NREL 5-MW published curve), 9.81 m/s (IEA 3.4-MW curve;
    published 9.8), 10.76 m/s (IEA 10-MW curve, whose rotor speed levels off at
    10.27 m/s while the power still rises to 10.76 m/s), 10.66 m/s (IEA 15-MW
    curve; published 10.66) and 11.13 m/s (IEA 22-MW curve; published 11.13). On a
    coarse curve without ``cp`` (an IEA 22-MW operating table on a 1 m/s grid) the
    first plateau sample (12 m/s) is returned, which errs to the safe side: a point
    between the true rated speed and it is torque-trimmed at the curve's pitch,
    which converges like any below-rated point.
    """
    plateau = _rated_plateau(curve_wind, curve_rotor_speed, curve_cp)
    rated = float(curve_wind[plateau.index])
    if plateau.power_level is None:
        detail = (
            f"first curve wind with the rotor speed on its {plateau.rotor_speed:.6g} "
            "rpm plateau (no cp column to locate the power plateau)"
        )
    else:
        detail = (
            f"first curve wind with the rotor speed on its {plateau.rotor_speed:.6g} "
            "rpm plateau and the aerodynamic power cp*V^3 on its plateau"
        )
    return rated, detail


def _air_density(deck: FastInputFile) -> tuple[float, str]:
    """Return the deck's air density and where it was read."""
    value = _float(deck, "AirDens")
    if value is not None and value > 0.0:
        return value, f"{deck.path.name} AirDens"
    aero = _subfile(deck, "AeroFile", "CompAero")
    value = _float(aero, "AirDens")
    if aero is not None and value is not None and value > 0.0:
        return value, f"{aero.path.name} AirDens"
    return _DEFAULT_AIR_DENSITY, f"default {_DEFAULT_AIR_DENSITY} kg/m^3"


def infer_rated_generator_torque(
    fst_path: str | Path,
    *,
    curve_wind: Sequence[float] | None = None,
    curve_rotor_speed: Sequence[float] | None = None,
    curve_cp: Sequence[float] | None = None,
    controller: ControllerConstants | None = None,
) -> InferredValue | None:
    """Return the rated generator torque (N·m, HSS side) from the deck or the curve.

    Sources, in order:

    1. the controller input file's ``VS_RtTq`` (``DISCON.IN``, the file named
       by ServoDyn ``DLL_InFile``) when the deck runs a controller library;
    2. ServoDyn ``VS_RtTq`` when ``VSContrl = 1`` (the simple variable-speed law
       actually uses it; under a library it is a placeholder);
    3. the curve's aerodynamic torque at rated, ``Q = cp * rho/2 * pi * R**2 * V**3
       / Omega`` averaged over the rated region, moved to the generator side with
       ElastoDyn ``GBoxEff`` and ``GBRatio`` (``Q * GBoxEff / GBRatio``), when the
       curve has a ``cp`` column.

    Parameters
    ----------
    fst_path : str or pathlib.Path
        The primary ``.fst`` deck.
    curve_wind, curve_rotor_speed, curve_cp : Sequence[float] or None, optional
        The operating curve and its power coefficients, for source 3.
    controller : ControllerConstants or None, optional
        The deck's controller constants, when already read
        (:func:`read_controller_constants` is called otherwise).

    Returns
    -------
    InferredValue or None
        The torque with its source, or ``None`` when no source has it.

    Notes
    -----
    The torque trim gain is derived from this value as a tenth of it (Jonkman's rule
    of thumb). On the low-speed shaft that loop's proportional gain is
    ``1000 * TrimGain * GBRatio = 100 * Q_rated``, so its time constant
    ``J / (100 * Q_rated)`` (``J`` the ElastoDyn rotor and drivetrain inertia) is
    0.10 s for the NREL 5-MW, 0.14 s for the IEA 3.4-MW, 0.19 s for the IEA 10-MW,
    0.18 s for the IEA 15-MW and 0.23 s for the IEA 22-MW: the rule scales with size
    by construction, since the rated torque grows with the rotor. Source 3 reproduces
    the published rated torques within 0.3 % (NREL 5-MW), 1.1 % (IEA 15-MW) and 3.9 %
    (IEA 22-MW).
    """
    if controller is None:
        controller = read_controller_constants(fst_path)
    if controller is not None and controller.rated_torque is not None:
        return InferredValue(
            "rated_generator_torque",
            controller.rated_torque,
            f"{controller.path.name} VS_RtTq",
            "controller input file (ServoDyn DLL_InFile)",
        )
    try:
        deck = FastInputFile.read(fst_path)
    except FastInputError:
        return None
    servo = _subfile(deck, "ServoFile", "CompServo")
    torque = _float(servo, "VS_RtTq")
    if (
        servo is not None
        and _float(servo, "VSContrl") == 1.0
        and torque is not None
        and torque > 0.0
    ):
        return InferredValue(
            "rated_generator_torque",
            torque,
            f"{servo.path.name} VS_RtTq",
            "ServoDyn simple variable-speed law (VSContrl=1)",
        )
    if curve_wind is None or curve_rotor_speed is None or curve_cp is None:
        return None
    return _curve_rated_torque(deck, curve_wind, curve_rotor_speed, curve_cp)


def _curve_rated_torque(
    deck: FastInputFile,
    curve_wind: Sequence[float],
    curve_rotor_speed: Sequence[float],
    curve_cp: Sequence[float],
) -> InferredValue | None:
    """Return the generator-side rated torque from the curve's power coefficient."""
    elastodyn = _subfile(deck, "EDFile")
    radius = _float(elastodyn, "TipRad")
    ratio = _float(elastodyn, "GBRatio")
    efficiency = _float(elastodyn, "GBoxEff")
    if radius is None or radius <= 0.0 or ratio is None or ratio <= 0.0:
        return None
    try:
        plateau = _rated_plateau(curve_wind, curve_rotor_speed, curve_cp)
    except ValueError:
        return None
    if plateau.power_level is None:
        return None
    density, density_source = _air_density(deck)
    gearbox = 1.0 if efficiency is None or efficiency <= 0.0 else efficiency / 100.0
    omega = plateau.rotor_speed * math.pi / 30.0
    power = 0.5 * density * math.pi * radius**2 * plateau.power_level
    torque = power / omega * gearbox / ratio
    return InferredValue(
        "rated_generator_torque",
        torque,
        "performance curve cp at rated",
        f"cp*rho/2*pi*R^2*V^3/Omega*GBoxEff/GBRatio with rho={density:g} "
        f"({density_source}), R={radius:g} m (ElastoDyn TipRad), "
        f"Omega={plateau.rotor_speed:.6g} rpm, GBoxEff={100 * gearbox:g} %, "
        f"GBRatio={ratio:g}",
    )


def _rated_cross_check(
    deck: FastInputFile,
    controller: ControllerConstants,
    curve_wind: Sequence[float],
    curve_rotor_speed: Sequence[float],
    curve_cp: Sequence[float] | None,
) -> list[str]:
    """Return disagreements between the curve's rated point and the controller."""
    warnings: list[str] = []
    try:
        plateau = _rated_plateau(curve_wind, curve_rotor_speed, curve_cp)
    except ValueError:
        return warnings
    elastodyn = _subfile(deck, "EDFile")
    ratio = _float(elastodyn, "GBRatio")
    if controller.rated_generator_speed is not None and ratio is not None and ratio > 0:
        rated_rpm = controller.rated_generator_speed / ratio * 30.0 / math.pi
        deviation = plateau.rotor_speed / rated_rpm - 1.0
        if abs(deviation) > _SPEED_AGREEMENT:
            warnings.append(
                f"the curve's rated rotor speed {plateau.rotor_speed:.4g} rpm differs "
                f"from the controller's {rated_rpm:.4g} rpm ({controller.path.name} "
                f"VS_RefSpd / ElastoDyn GBRatio) by {100 * deviation:+.1f} %"
            )
    radius = _float(elastodyn, "TipRad")
    if (
        plateau.power_level is not None
        and controller.rated_power is not None
        and radius is not None
    ):
        density, _ = _air_density(deck)
        efficiency = controller.generator_efficiency or 1.0
        gearbox = _float(elastodyn, "GBoxEff")
        gearbox_fraction = 1.0 if gearbox is None or gearbox <= 0 else gearbox / 100.0
        electrical = (
            0.5
            * density
            * math.pi
            * radius**2
            * plateau.power_level
            * efficiency
            * gearbox_fraction
        )
        deviation = electrical / controller.rated_power - 1.0
        if abs(deviation) > _POWER_AGREEMENT:
            warnings.append(
                f"the curve's rated power {electrical / 1e6:.4g} MW (cp, rho, "
                f"TipRad, VS_GenEff) differs from the controller's "
                f"{controller.rated_power / 1e6:.4g} MW ({controller.path.name} "
                f"VS_RtPwr) by {100 * deviation:+.1f} %"
            )
    return warnings


def _deck_module_settings(
    deck: FastInputFile,
) -> tuple[LinearizationSettings, list[InferredValue], list[str]]:
    """Return the offshore preset adapted to the deck's modules and files."""
    settings = LinearizationSettings.offshore()
    records: list[InferredValue] = [
        InferredValue(
            "module_preset",
            "offshore",
            "LinearizationSettings.offshore()",
            "still water, 2nd-order hydro off, frozen wake (DBEMT_Mod=-1), trim "
            "controller; adapted to the deck below",
        )
    ]
    warnings: list[str] = []
    hydro = _subfile(deck, "HydroFile", "CompHydro")
    if hydro is None or _float(hydro, "PotMod") != 1.0:
        settings = replace(
            settings, hydro_radiation_mode=None, hydro_excitation_mode=None
        )
    else:
        settings, hydro_records, hydro_warnings = _hydro_modes(settings, hydro, deck)
        records += hydro_records
        warnings += hydro_warnings
    aero = _subfile(deck, "AeroFile", "CompAero")
    if aero is not None and "AFAeroMod" not in aero:
        settings, aero_records, aero_warnings = _aero_modes(settings, aero)
        records += aero_records
        warnings += aero_warnings
    return settings, records, warnings


def _state_space_file_available(hydro: FastInputFile, suffix: str) -> bool:
    """Return whether the state-space file ``<PotFile><suffix>`` exists.

    The single place the sweep asks for a ``.ss`` (radiation) or ``.ssexctn``
    (excitation) file. A missing ``.ss`` may still be fitted from the WAMIT data (see
    :func:`_radiation_fit_plan`); a missing ``.ssexctn`` makes the sweep fall back to
    model 0.
    """
    from vane.run.preflight import _has_potfile

    return _has_potfile(hydro, suffix)


def _radiation_fit_plan(
    deck: FastInputFile,
) -> tuple[InferredValue | None, str]:
    """Return whether the missing ``<PotFile>.ss`` can be fitted from the ``.1`` data.

    The fit is tried here, in memory, before any case is copied, so a deck whose
    WAMIT data cannot be fitted within the acceptance limit falls back to
    ``RdtnMod = 0`` once, with a warning, instead of failing every case. Returns the
    provenance record of a usable fit (``None`` when there is none) and the reason
    there is none.
    """
    from vane.run.radiation_fit import RadiationFitError, preview_radiation_state_space

    try:
        record = preview_radiation_state_space(deck.path)
    except (RadiationFitError, FastInputError, OSError, ValueError) as error:
        return None, f"it cannot be fitted from the WAMIT data ({error})"
    if record is None:
        return None, "the deck has no WAMIT potential flow to fit it from"
    if not record.accepted:
        return None, (
            f"the fit to {record.source.name} is not accurate enough (max relative "
            f"error {record.max_relative_error:.3f})"
        )
    return (
        InferredValue(
            "hydro_radiation_fit",
            True,
            f"{record.source.name} (WAMIT radiation)",
            f"'<PotFile>.ss' is missing; it is fitted to {record.source.name} in each "
            f"case copy ({record.n_states} states, max relative fit error "
            f"{record.max_relative_error:.3f}), never next to the template's data",
        ),
        "",
    )


def _hydro_modes(
    settings: LinearizationSettings,
    hydro: FastInputFile,
    deck: FastInputFile,
) -> tuple[LinearizationSettings, list[InferredValue], list[str]]:
    """Choose linearizable HydroDyn radiation/excitation models for a WAMIT deck.

    A radiation model whose ``.ss`` file is missing keeps the state-space model
    (``RdtnMod = 2``) when the file can be fitted from the WAMIT ``.1`` data: the fit
    is then generated in each case copy (``hydro_radiation_fit``). Otherwise, as for a
    missing ``.ssexctn``, the model falls back to 0 with a warning.
    """
    records: list[InferredValue] = []
    warnings: list[str] = []
    name = hydro.path.name
    add_dofs = False
    if "NAddDOF" in hydro:
        try:
            add_dofs = any(int(fortran_float(t)) > 0 for t in hydro.tokens("NAddDOF"))
        except (FastInputError, ValueError):
            add_dofs = False
    choices: dict[str, int | None] = {}
    fit_record: InferredValue | None = None
    for field_name, keyword, suffix in (
        ("hydro_radiation_mode", "RdtnMod", ".ss"),
        ("hydro_excitation_mode", "ExctnMod", ".ssexctn"),
    ):
        current = _float(hydro, keyword)
        if current is None or current == 0.0:
            # The deck's own "none" is linearizable; a file without the line (an
            # older HydroDyn has no ExctnMod) or with a non-numeric value has no
            # model to switch, and is left unchanged for validate_deck to judge.
            choices[field_name] = None
            continue
        if add_dofs:
            choices[field_name] = 0
            warnings.append(
                f"{name} has additional body DOFs (NAddDOF>0), which cannot be "
                f"combined with a state-space model; {keyword}=0 is used"
            )
        elif _state_space_file_available(hydro, suffix):
            choices[field_name] = 2
        else:
            reason = ""
            if suffix == ".ss":
                fit_record, reason = _radiation_fit_plan(deck)
            if suffix == ".ss" and fit_record is not None:
                choices[field_name] = 2
            else:
                choices[field_name] = 0
                because = f" and {reason}" if reason else ""
                warnings.append(
                    f"{name} {keyword}={current:g} needs the state-space file "
                    f"'<PotFile>{suffix}', which is missing{because}; {keyword}=0 is "
                    "used (the "
                    f"{'radiation memory' if suffix == '.ss' else 'wave excitation'} "
                    "is omitted from the linearization)"
                )
        records.append(
            InferredValue(field_name, choices[field_name], f"{name} {keyword}")
        )
    if fit_record is not None:
        records.append(fit_record)
    return (
        replace(
            settings,
            hydro_radiation_mode=choices["hydro_radiation_mode"],
            hydro_excitation_mode=choices["hydro_excitation_mode"],
            hydro_radiation_fit=fit_record is not None,
        ),
        records,
        warnings,
    )


def _aero_modes(
    settings: LinearizationSettings, aero: FastInputFile
) -> tuple[LinearizationSettings, list[InferredValue], list[str]]:
    """Keep a modern AeroDyn deck's own linearizable unsteady-aerodynamics model."""
    name = aero.path.name
    unsteady = _float(aero, "UA_Mod")
    if unsteady is None or int(unsteady) not in _LINEARIZABLE_UNSTEADY:
        record = InferredValue(
            "unsteady_aero_mode",
            0,
            f"{name} UA_Mod",
            f"UA_Mod={unsteady if unsteady is None else int(unsteady)} replaced by "
            "quasi-steady aerodynamics (0)"
            if unsteady not in (None, 0.0)
            else "quasi-steady aerodynamics",
        )
        return settings, [record], []
    mode = int(unsteady)
    records = [
        InferredValue(
            "unsteady_aero_mode",
            mode,
            f"{name} UA_Mod",
            "the deck's own linearizable unsteady-aerodynamics model is kept",
        )
    ]
    end_radius = _float(aero, "UAEndRad")
    tip_reached = (
        _float(aero, "Wake_Mod") == 1.0
        and _flag(aero, "TipLoss")
        and (end_radius is None or end_radius >= 1.0)
    )
    if not tip_reached:
        return replace(settings, unsteady_aero_mode=None), records, []
    if "UAEndRad" not in aero:
        records[0] = InferredValue(
            "unsteady_aero_mode",
            0,
            f"{name} UA_Mod",
            f"UA_Mod={mode} would reach the blade-tip node under TipLoss and the file "
            "has no UAEndRad line to stop it inboard; quasi-steady aerodynamics (0)",
        )
        warning = (
            f"{name} has UA_Mod={mode} with TipLoss but no UAEndRad line; "
            "quasi-steady aerodynamics is used (add a UAEndRad line to keep the model)"
        )
        return settings, records, [warning]
    records.append(
        InferredValue(
            "ua_end_radius",
            _UA_END_RADIUS,
            f"{name} UAEndRad",
            f"UAEndRad={1.0 if end_radius is None else end_radius:g} with TipLoss "
            "would keep unsteady aerodynamics on the tip node, whose states can turn "
            "NaN in the linearization",
        )
    )
    return (
        replace(settings, unsteady_aero_mode=None, ua_end_radius=_UA_END_RADIUS),
        records,
        [],
    )


def auto_configure_sweep(
    fst_path: str | Path,
    settings: LinearizationSettings | None,
    *,
    curve_wind: Sequence[float] | None = None,
    curve_rotor_speed: Sequence[float] | None = None,
    curve_cp: Sequence[float] | None = None,
) -> tuple[LinearizationSettings, AutoConfiguration]:
    """Return sweep settings completed from the deck and curve, with the record.

    With ``settings=None`` the module configuration is chosen for the deck (the
    :meth:`~vane.run.LinearizationSettings.offshore` preset, adapted as described in
    the module docstring). In every case the turbine constants the settings leave
    unset are inferred: ``rated_wind_speed`` and ``rated_generator_torque`` when
    ``None``, and ``trim_gain_pitch`` and ``trim_tol`` when they were left at their
    class defaults (2e-4 and 1e-3, which apply to a 63 m rotor; see
    :func:`~vane.run.lin_config.is_size_scaled_default`). Values set explicitly are
    kept, even when they equal the defaults.

    Parameters
    ----------
    fst_path : str or pathlib.Path
        The primary ``.fst`` deck.
    settings : LinearizationSettings or None
        The user's settings, or ``None`` for a fully automatic configuration.
    curve_wind, curve_rotor_speed : Sequence[float] or None, optional
        The operating curve (m/s, rpm), when one was supplied.
    curve_cp : Sequence[float] or None, optional
        The curve's power coefficients, when its CSV has a ``cp`` column.

    Returns
    -------
    tuple[LinearizationSettings, AutoConfiguration]
        The completed settings and every inferred value with its source.

    Notes
    -----
    A constant that cannot be inferred is left unset; the sweep's boundary checks
    then fail before launch with the keyword to supply.
    """
    deck = FastInputFile.read(fst_path)
    records: list[InferredValue] = []
    warnings: list[str] = []
    if settings is None:
        resolved, module_records, module_warnings = _deck_module_settings(deck)
        records += module_records
        warnings += module_warnings
    else:
        resolved = settings.snapshot()
    controller = read_controller_constants(fst_path)
    curve = curve_wind is not None and curve_rotor_speed is not None
    if resolved.rated_wind_speed is None and curve:
        assert curve_wind is not None and curve_rotor_speed is not None
        try:
            rated, detail = infer_rated_wind_speed(
                curve_wind, curve_rotor_speed, curve_cp
            )
        except ValueError as error:
            warnings.append(f"rated_wind_speed could not be inferred: {error}")
        else:
            resolved = replace(resolved, rated_wind_speed=rated)
            records.append(
                InferredValue("rated_wind_speed", rated, "operating curve", detail)
            )
            if controller is not None:
                warnings += _rated_cross_check(
                    deck, controller, curve_wind, curve_rotor_speed, curve_cp
                )
    if resolved.rated_generator_torque is None:
        torque = infer_rated_generator_torque(
            fst_path,
            curve_wind=curve_wind,
            curve_rotor_speed=curve_rotor_speed,
            curve_cp=curve_cp,
            controller=controller,
        )
        if torque is not None:
            assert isinstance(torque.value, float)
            resolved = replace(resolved, rated_generator_torque=torque.value)
            records.append(torque)
    if is_size_scaled_default(resolved.trim_gain_pitch):
        radius = rotor_radius(fst_path)
        if radius is not None:
            gain = scaled_trim_gain_pitch(radius)
            resolved = replace(resolved, trim_gain_pitch=gain)
            records.append(
                InferredValue(
                    "trim_gain_pitch",
                    gain,
                    "ElastoDyn TipRad",
                    f"2e-4*(63/R)^3 with R={radius:g} m",
                )
            )
    if is_size_scaled_default(resolved.trim_tol):
        radius = rotor_radius(fst_path)
        if radius is not None:
            tolerance = scaled_trim_tol(radius)
            resolved = replace(resolved, trim_tol=tolerance)
            records.append(
                InferredValue(
                    "trim_tol",
                    tolerance,
                    "ElastoDyn TipRad",
                    f"1e-3*(R/63)^3 within [1e-3, 1e-2] with R={radius:g} m",
                )
            )
    enabled = _enabled_flexibility(fst_path, resolved)
    if enabled:
        warnings.append(
            f"ElastoDyn {', '.join(enabled)} {'is' if len(enabled) == 1 else 'are'} "
            "off in the deck and enabled for the linearization, so its blade and tower "
            "modes are linearized; pass structural_dofs={...: False} to keep "
            f"{'it' if len(enabled) == 1 else 'them'} off"
        )
    for record in records:
        logger.info(
            "auto-configured %s = %r from %s%s",
            record.name,
            record.value,
            record.source,
            f" ({record.detail})" if record.detail else "",
        )
    for warning in warnings:
        logger.warning("auto-configuration: %s", warning)
    return resolved, AutoConfiguration(tuple(records), tuple(warnings))
