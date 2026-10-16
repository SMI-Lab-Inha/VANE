# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Verify a linearization's trimmed operating point against rotor-performance data.

VANE does not compute rotor performance: the user supplies the steady operating
schedule (wind speed, rotor speed, blade pitch and, optionally, the power and thrust
coefficients) and OpenFAST's ``CalcSteady`` trims the turbine about it. Below rated it
trims the generator torque at the prescribed pitch and rotor speed; above rated it
trims the blade pitch at constant generator torque. Whether the trimmed point agrees
with the supplied curve is a direct consistency check on the curve and the deck: a
large mismatch points to a wrong curve, rotor radius, air density, gearbox ratio or
drivetrain loss, or a deck/curve inconsistency.

This module reads the trimmed operating point back out of the ``.lin`` files of each
linearization (inputs first, outputs as a fallback), derives the aerodynamic power
from the steady low-speed-shaft torque balance and compares rotor speed, pitch, power
coefficient and (when the outputs carry the rotor thrust) thrust coefficient with the
curve, each against a documented tolerance.

Torque balance
--------------
At a periodic steady state the azimuth-mean rotor acceleration vanishes, so the mean
aerodynamic torque on the low-speed shaft equals the shaft torque transmitted to the
gearbox. ElastoDyn relates the low-speed-shaft torque :math:`Q_{LSS}` to the
high-speed-shaft generator torque :math:`Q_g` through the gearbox ratio :math:`N` and
efficiency :math:`\eta`,

.. math::

    Q_{LSS} = N\,Q_g\,\eta^{-\operatorname{sgn}(Q_g \Omega)},

so that losses reduce the generator torque when power flows rotor to generator and
increase the required shaft torque when the generator motors the rotor. The
aerodynamic power and power coefficient are then

.. math::

    P = Q_{LSS}\,\Omega, \qquad
    C_p = \frac{P}{\tfrac12 \rho \pi R^2 U^3}.

The generator (electrical) efficiency does not enter: ``Q_g`` is the mechanical torque
on the high-speed shaft. Bearing friction and any other drivetrain loss not modelled by
``GBoxEff`` are neglected.

The balance also assumes nothing else acts on the rotor. ``CalcSteady``'s artificial
trim damping (``Twr_Kdmp``/``Bld_Kdmp``) does: on the IEA 10-MW sweep, trimmed with
5000 N/(m/s), it absorbed 8 % of the rotor torque at 4 m/s (OpenFAST v4.0.5: AeroDyn
power 436.6 kW against 396.7 kW from the balance; 1.3 % apart when the same point is
trimmed without damping, AeroDyn's power unchanged). When the deck beside the ``.lin``
files has non-zero trim damping, the aerodynamic power is therefore AeroDyn's own
rotor power (``RtFldPwr``/``RtAeroPwr``) when the outputs carry it.
"""

from __future__ import annotations

import csv
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.interpolate import PchipInterpolator

from vane.io.fst_reader import fortran_float
from vane.io.fst_writer import FastInputError, FastInputFile
from vane.io.lin_reader import LinFile, read_lin_file
from vane.run.operating_points import (
    _coefficient_cell,
    _coefficient_column,
    _coefficient_fractions,
    _csv_unit_factor,
    _resolve_csv_columns,
    operating_curve_from_csv,
)

if TYPE_CHECKING:
    import os
    from collections.abc import Sequence

    import numpy.typing as npt

    from vane.io.lin_reader import OperatingPointTable
    from vane.run.auto_sweep import OperatingSweepResult

logger = logging.getLogger(__name__)

__all__ = [
    "OperatingPointCheck",
    "OperatingPointVerification",
    "QuantityCheck",
    "RotorPerformanceCurve",
    "TrimTolerances",
    "TrimmedOperatingPoint",
    "TurbineConstants",
    "aerodynamic_power",
    "rotor_performance_from_csv",
    "trimmed_operating_point",
    "verify_operating_points",
    "verify_sweep_operating_points",
]

CheckStatus = Literal["pass", "flag", "unavailable"]
PointStatus = Literal["pass", "flag", "incomplete"]
RadiusConvention = Literal["projected", "tip"]

# ``.lin`` channel descriptions. Inputs are ``"<Mod> <text>, <unit>"``; outputs are
# ``"<Mod> <Channel>, (<unit>)"``.
_BLADE_PITCH_INPUT_RE = re.compile(
    r"^S?ED\s+Blade\s+(\d+)\s+pitch command,\s*rad\s*$", re.IGNORECASE
)
_COLLECTIVE_PITCH_INPUT_RE = re.compile(
    r"^S?ED\s+.*collective blade[- ]pitch command,\s*rad\s*$", re.IGNORECASE
)
_SINGLE_PITCH_INPUT_RE = re.compile(
    r"^S?ED\s+Blade pitch command,\s*rad\s*$", re.IGNORECASE
)
_GEN_TORQUE_INPUT_RE = re.compile(
    r"^S?ED\s+Generator torque,\s*N-?m\s*$", re.IGNORECASE
)
_OUTPUT_RE = re.compile(
    r"^(?P<module>\S+)\s+(?P<name>[^,]+),\s*\((?P<unit>[^)]*)\)\s*$"
)
_BLADE_PITCH_ANGLE_OUTPUT_RE = re.compile(
    r"^S?ED\s+Blade\s+(\d+)\s+pitch angle,\s*rad\s*$", re.IGNORECASE
)

# Unit factors to SI for the output channels this module reads.
_OUTPUT_UNIT_TO_SI: dict[str, float] = {
    "n": 1.0,
    "kn": 1.0e3,
    "nm": 1.0,
    "n-m": 1.0,
    "knm": 1.0e3,
    "kn-m": 1.0e3,
    "w": 1.0,
    "kw": 1.0e3,
    "deg": 1.0,
    "rad": 180.0 / math.pi,
}
# Rotor-thrust output channels, in order of preference. Only AeroDyn's rotor
# aerodynamic thrust is used: ElastoDyn's ``RotThrust`` is the low-speed-shaft thrust,
# which also carries the axial component of the rotor weight on a tilted shaft.
_THRUST_CHANNELS = ("RtFldFxh", "RtAeroFxh")
_AERO_POWER_CHANNELS = ("RtFldPwr", "RtAeroPwr")
# Relative spread across the azimuth files above which a set is reported as
# inconsistent. The files share one trimmed operating point, so the wind speed is
# constant; the rotor speed carries a physical once-per-revolution ripple (gravity and
# shear on the rotor) of a few tenths of a percent, which must not be reported.
_SET_SPREAD_RTOL = {"wind speed": 1.0e-3, "rotor speed": 0.02}
_ZERO_FLOOR = 1.0e-12


# --------------------------------------------------------------------------------------
# Rotor-performance curve
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RotorPerformanceCurve:
    """A user-supplied steady rotor-performance schedule.

    Parameters
    ----------
    wind_speed : tuple[float, ...]
        Strictly increasing hub-height wind speeds, in m/s.
    rotor_speed_rpm : tuple[float, ...]
        Rotor speed at each wind speed, in rpm.
    blade_pitch_deg : tuple[float, ...]
        Collective blade pitch at each wind speed, in degrees.
    cp : tuple[float, ...] or None, optional
        Aerodynamic power coefficient at each wind speed (``NaN`` where unknown), or
        ``None`` when the curve carries no power coefficient.
    ct : tuple[float, ...] or None, optional
        Rotor thrust coefficient at each wind speed (``NaN`` where unknown), or
        ``None`` when the curve carries no thrust coefficient.

    Raises
    ------
    ValueError
        If the arrays differ in length, have fewer than two samples, contain a
        non-finite wind, rotor speed or pitch, or the wind speeds are not strictly
        increasing.
    """

    wind_speed: tuple[float, ...]
    rotor_speed_rpm: tuple[float, ...]
    blade_pitch_deg: tuple[float, ...]
    cp: tuple[float, ...] | None = None
    ct: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        """Validate the curve's shape and monotonicity."""
        n = len(self.wind_speed)
        lengths = {
            "rotor_speed_rpm": len(self.rotor_speed_rpm),
            "blade_pitch_deg": len(self.blade_pitch_deg),
        }
        if self.cp is not None:
            lengths["cp"] = len(self.cp)
        if self.ct is not None:
            lengths["ct"] = len(self.ct)
        if any(length != n for length in lengths.values()):
            msg = f"curve arrays must share one length; wind has {n}, got {lengths}"
            raise ValueError(msg)
        if n < 2:
            msg = "a rotor-performance curve needs at least two samples"
            raise ValueError(msg)
        required = np.asarray(
            [self.wind_speed, self.rotor_speed_rpm, self.blade_pitch_deg],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(required)):
            msg = "curve wind speed, rotor speed and pitch must all be finite"
            raise ValueError(msg)
        if not np.all(np.diff(required[0]) > 0.0):
            msg = "curve wind speeds must be strictly increasing"
            raise ValueError(msg)

    @classmethod
    def from_arrays(
        cls,
        wind_speed: Sequence[float],
        rotor_speed_rpm: Sequence[float],
        blade_pitch_deg: Sequence[float],
        *,
        cp: Sequence[float] | None = None,
        ct: Sequence[float] | None = None,
    ) -> RotorPerformanceCurve:
        """Build a curve from array-likes, e.g. ``operating_curve_from_csv`` output.

        Parameters
        ----------
        wind_speed, rotor_speed_rpm, blade_pitch_deg : Sequence[float]
            The required schedule, in m/s, rpm and degrees.
        cp, ct : Sequence[float] or None, optional
            Optional power and thrust coefficients.

        Returns
        -------
        RotorPerformanceCurve
            The validated curve.
        """

        def _tuple(values: Sequence[float] | None) -> tuple[float, ...] | None:
            return None if values is None else tuple(float(v) for v in values)

        return cls(
            wind_speed=tuple(float(v) for v in wind_speed),
            rotor_speed_rpm=tuple(float(v) for v in rotor_speed_rpm),
            blade_pitch_deg=tuple(float(v) for v in blade_pitch_deg),
            cp=_tuple(cp),
            ct=_tuple(ct),
        )

    def at(self, wind_speed: float) -> dict[str, float]:
        """Interpolate the curve at one wind speed.

        The same shape-preserving monotone cubic (PCHIP) used to build sweep operating
        points from a curve is used here, without extrapolation.

        Parameters
        ----------
        wind_speed : float
            Hub-height wind speed, in m/s.

        Returns
        -------
        dict[str, float]
            ``rotor_speed_rpm``, ``blade_pitch_deg``, ``cp`` and ``ct``; a value is
            ``NaN`` outside the curve's range or when the curve lacks that quantity
            (a coefficient needs at least two finite samples).
        """
        wind = np.asarray(self.wind_speed, dtype=np.float64)
        out = {
            "rotor_speed_rpm": _pchip(wind, self.rotor_speed_rpm, wind_speed),
            "blade_pitch_deg": _pchip(wind, self.blade_pitch_deg, wind_speed),
            "cp": math.nan,
            "ct": math.nan,
        }
        for key, values in (("cp", self.cp), ("ct", self.ct)):
            if values is None:
                continue
            array = np.asarray(values, dtype=np.float64)
            finite = np.isfinite(array)
            if int(finite.sum()) >= 2:
                out[key] = _pchip(wind[finite], array[finite], wind_speed)
        return out


def _pchip(x: npt.ArrayLike, y: npt.ArrayLike, at: float) -> float:
    """Evaluate a non-extrapolating PCHIP interpolant (``NaN`` outside the range)."""
    if not math.isfinite(at):
        return math.nan
    return float(PchipInterpolator(x, y, extrapolate=False)(at))


def rotor_performance_from_csv(path: str | Path) -> RotorPerformanceCurve:
    """Read a rotor-performance CSV, including its optional coefficient columns.

    The required ``wind_speed``, ``rotor_speed_rpm`` and ``blade_pitch_deg`` columns
    are parsed, unit-converted and validated exactly as by
    :func:`~vane.run.operating_curve_from_csv` (unit-annotated headers such as
    ``wind_speed [m/s]`` are accepted). The optional power-coefficient column
    (``cp``, ``power_coefficient``, ``RtAeroCp``, ``RtFldCp``) and thrust-coefficient
    column (``ct``, ``thrust_coefficient``, ``RtAeroCt``, ``RtFldCt``) are read as
    well; a blank or non-numeric coefficient cell becomes ``NaN`` (unknown) rather
    than an error.

    The coefficients are returned as fractions. A coefficient column's unit is read
    from its header bracket, as for the required columns: no unit, ``[-]``, ``[1]`` or
    ``[fraction]`` means a fraction, and ``[%]``, ``[percent]`` or ``[pct]`` a
    percentage, which is divided by 100. Any other unit is rejected. So is a scale that
    contradicts the declared unit: a fraction above the physical bound (the Betz limit
    16/27 for ``cp``, 2 for ``ct``), which is what a percentage written without its
    unit looks like, and a column declared in percent whose non-zero values are all at
    most 1 in magnitude, which reads as fractions labelled as percent. A separate
    units row is not supported.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the rotor-performance CSV.

    Returns
    -------
    RotorPerformanceCurve
        The curve, sorted by ascending wind speed.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        As :func:`~vane.run.operating_curve_from_csv`; if more than one column matches
        the power or thrust coefficient; or if a coefficient column has an unsupported
        unit or a scale that contradicts its unit (see above).
    """
    wind, rotor, pitch = operating_curve_from_csv(path)
    text = Path(path).read_text(encoding="utf-8-sig")
    rows = [row for row in csv.reader(text.splitlines()) if any(c.strip() for c in row)]
    header, data_rows = rows[0], rows[1:]
    columns = _resolve_csv_columns(header)
    wind_index = columns["wind_speed"]
    wind_factor = _csv_unit_factor("wind_speed", header[wind_index])
    raw_wind = [float(row[wind_index]) * wind_factor for row in data_rows]
    order = np.argsort(np.asarray(raw_wind, dtype=np.float64), kind="stable")

    coefficients: dict[str, tuple[float, ...] | None] = {}
    for key in ("cp", "ct"):
        column = _coefficient_column(header, key, path)
        if column is None:
            coefficients[key] = None
            continue
        raw = [_coefficient_cell(row, column.index) for row in data_rows]
        values = _coefficient_fractions(column, raw, path)
        coefficients[key] = tuple(values[i] for i in order)
    return RotorPerformanceCurve(
        wind_speed=tuple(wind),
        rotor_speed_rpm=tuple(rotor),
        blade_pitch_deg=tuple(pitch),
        cp=coefficients["cp"],
        ct=coefficients["ct"],
    )


# --------------------------------------------------------------------------------------
# Turbine constants
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TurbineConstants:
    r"""Turbine constants the torque balance and coefficients need.

    Parameters
    ----------
    gearbox_ratio : float
        Gearbox ratio :math:`N` (high-speed over low-speed shaft speed), positive.
    gearbox_efficiency : float
        Gearbox efficiency :math:`\eta` as a fraction in ``(0, 1]`` (ElastoDyn's
        ``GBoxEff`` is in percent; :meth:`from_deck` converts it).
    rotor_radius : float
        Rotor radius :math:`R` used to normalize the coefficients, in m.
    air_density : float
        Air density :math:`\rho`, in kg/m^3.

    Raises
    ------
    ValueError
        If any constant is non-finite or non-positive, or the efficiency exceeds one.
    """

    gearbox_ratio: float
    gearbox_efficiency: float
    rotor_radius: float
    air_density: float

    def __post_init__(self) -> None:
        """Validate the constants."""
        for name in (
            "gearbox_ratio",
            "gearbox_efficiency",
            "rotor_radius",
            "air_density",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                msg = f"{name} must be finite and positive, got {value}"
                raise ValueError(msg)
        if self.gearbox_efficiency > 1.0:
            msg = (
                "gearbox_efficiency is a fraction in (0, 1], got "
                f"{self.gearbox_efficiency} (ElastoDyn's GBoxEff is in percent)"
            )
            raise ValueError(msg)

    @property
    def rotor_area(self) -> float:
        r"""Return the swept area :math:`\pi R^2`, in m^2."""
        return math.pi * self.rotor_radius**2

    @classmethod
    def from_deck(
        cls,
        fst_path: str | Path,
        *,
        radius_convention: RadiusConvention = "projected",
    ) -> TurbineConstants:
        """Read the constants from an OpenFAST deck.

        The gearbox ratio (``GBRatio``, or ``GBoxRatio`` for Simplified-ElastoDyn),
        gearbox efficiency (``GBoxEff``, percent; 100 % when absent), tip radius
        (``TipRad``) and precone (``PreCone(i)`` or ``PreCone``) are read from the
        structural (``EDFile``) input; this holds for BeamDyn decks too, whose rotor
        geometry still lives in ElastoDyn. The air density is AeroDyn's ``AirDens``,
        or the ``.fst`` ``AirDens`` when AeroDyn's is ``"default"`` or absent.

        Parameters
        ----------
        fst_path : str or pathlib.Path
            The primary ``.fst`` deck.
        radius_convention : {"projected", "tip"}, optional
            ``"projected"`` (default) normalizes with the rotor-plane radius
            ``TipRad * cos(PreCone)``, the swept-area radius most blade-element tools
            use; ``"tip"`` uses ``TipRad`` itself. Pass explicit constants when a curve
            was normalized with some other radius.

        Returns
        -------
        TurbineConstants
            The constants.

        Raises
        ------
        ValueError
            If a required value is missing or non-numeric, or ``radius_convention``
            is unknown.
        """
        if radius_convention not in ("projected", "tip"):
            msg = (
                "radius_convention must be 'projected' or 'tip', got "
                f"{radius_convention!r}"
            )
            raise ValueError(msg)
        try:
            deck = FastInputFile.read(fst_path)
            structure = FastInputFile.read(deck.resolve("EDFile"))
        except FastInputError as error:
            msg = f"cannot read the structural input of deck {fst_path}: {error}"
            raise ValueError(msg) from error
        ratio_key = "GBRatio" if "GBRatio" in structure else "GBoxRatio"
        gearbox_ratio = _deck_float(structure, ratio_key)
        if "GBoxEff" in structure:
            efficiency = _deck_float(structure, "GBoxEff") / 100.0
        else:
            logger.info("%s has no GBoxEff; taking a lossless gearbox", structure.path)
            efficiency = 1.0
        tip = _deck_float(structure, "TipRad")
        radius = tip
        if radius_convention == "projected":
            cones = [
                _deck_float(structure, f"PreCone({i})")
                for i in (1, 2, 3)
                if f"PreCone({i})" in structure
            ]
            if not cones and "PreCone" in structure:
                cones = [_deck_float(structure, "PreCone")]
            cone = float(np.mean(cones)) if cones else 0.0
            radius = tip * math.cos(math.radians(cone))
        return cls(
            gearbox_ratio=abs(gearbox_ratio),
            gearbox_efficiency=efficiency,
            rotor_radius=radius,
            air_density=_air_density(deck),
        )


def _deck_float(deck: FastInputFile, keyword: str) -> float:
    """Return a numeric deck value or raise a clear :class:`ValueError`."""
    try:
        token = deck.get(keyword)
    except FastInputError as error:
        msg = f"{deck.path}: {error}"
        raise ValueError(msg) from error
    try:
        return fortran_float(token)
    except ValueError as error:
        msg = f"{deck.path}: {keyword} value {token!r} is not numeric"
        raise ValueError(msg) from error


def _air_density(deck: FastInputFile) -> float:
    """Return AeroDyn's air density, falling back to the ``.fst`` value."""
    try:
        aero = FastInputFile.read(deck.resolve("AeroFile"))
    except FastInputError:
        aero = None
    if aero is not None and "AirDens" in aero:
        token = aero.get("AirDens").strip("\"'")
        if token.lower() != "default":
            return _deck_float(aero, "AirDens")
    if "AirDens" not in deck:
        msg = (
            f"no air density in {deck.path} or its AeroDyn input; pass "
            "TurbineConstants explicitly"
        )
        raise ValueError(msg)
    return _deck_float(deck, "AirDens")


# --------------------------------------------------------------------------------------
# Trimmed operating point from .lin files
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrimmedOperatingPoint:
    """The operating point OpenFAST trimmed to, read from one linearization's files.

    Every quantity is the mean over the azimuth files of the set (and, for pitch, over
    the blades).

    Parameters
    ----------
    wind_speed : float
        Hub-height wind speed, in m/s.
    rotor_speed_rpm : float
        Rotor speed, in rpm.
    blade_pitch_deg : float
        Blade pitch, mean over blades and azimuths, in degrees (``NaN`` if absent).
    blade_pitch_spread_deg : float
        Largest minus smallest pitch over blades and azimuths, in degrees (``NaN`` if
        absent). A trimmed collective pitch has zero spread.
    generator_torque : float
        High-speed-shaft generator torque, in N m (``NaN`` if absent).
    n_lin : int
        Number of ``.lin`` files in the set.
    pitch_source : str or None
        The channel the pitch was read from, or ``None``.
    generator_torque_source : str or None
        The channel the generator torque was read from, or ``None``.
    rotor_thrust : float or None
        AeroDyn rotor aerodynamic thrust along the shaft, in N, when the outputs carry
        it; else ``None``.
    rotor_thrust_source : str or None
        The output channel the thrust was read from.
    aerodyn_power : float or None
        AeroDyn rotor aerodynamic power, in W, when the outputs carry it; an
        independent check on the torque-balance power.
    issues : tuple[str, ...]
        Human-readable notes on anything missing or inconsistent.
    aerodyn_power_source : str or None
        The output channel ``aerodyn_power`` was read from.
    trim_damping : float or None
        The larger of the ``Twr_Kdmp`` and ``Bld_Kdmp`` trim-damping factors, in
        N/(m/s), of the ``<root>.fst`` deck beside the files (``0.0`` when
        ``CalcSteady`` is off); ``None`` when that deck cannot be read. Non-zero trim
        damping absorbs rotor torque, so the torque balance no longer gives the
        aerodynamic power (see :func:`verify_operating_points`).
    """

    wind_speed: float
    rotor_speed_rpm: float
    blade_pitch_deg: float
    blade_pitch_spread_deg: float
    generator_torque: float
    n_lin: int
    pitch_source: str | None = None
    generator_torque_source: str | None = None
    rotor_thrust: float | None = None
    rotor_thrust_source: str | None = None
    aerodyn_power: float | None = None
    issues: tuple[str, ...] = ()
    aerodyn_power_source: str | None = None
    trim_damping: float | None = None


def trimmed_operating_point(
    lin_files: Sequence[LinFile | str | os.PathLike[str]],
) -> TrimmedOperatingPoint:
    """Read the trimmed operating point from one linearization's ``.lin`` files.

    The blade pitch is read from the ElastoDyn pitch-command inputs (``ED Blade <i>
    pitch command``, the same for ElastoDyn and BeamDyn blades), or the collective
    pitch-command input; the generator torque from the ``ED Generator torque`` input.
    When the inputs are absent (``LinInputs = 0``) the ``ED BldPitch<i>`` and
    ``SrvD GenTq`` outputs are used instead. Wind and rotor speed come from each
    file's header. Every quantity is averaged over the azimuth files.

    Parameters
    ----------
    lin_files : Sequence[LinFile or str or os.PathLike]
        The full-system ``.lin`` files of one linearization (one per azimuth), parsed
        or as paths.

    Returns
    -------
    TrimmedOperatingPoint
        The trimmed point; a quantity that cannot be found is ``NaN`` and explained
        in :attr:`TrimmedOperatingPoint.issues`.

    Raises
    ------
    ValueError
        If ``lin_files`` is empty.
    """
    if len(lin_files) == 0:
        msg = "a linearization needs at least one .lin file"
        raise ValueError(msg)
    lins = [
        lin if isinstance(lin, LinFile) else read_lin_file(lin) for lin in lin_files
    ]
    issues: list[str] = []

    winds = np.asarray([lin.wind_speed for lin in lins], dtype=np.float64)
    speeds = np.asarray([lin.rotor_speed for lin in lins], dtype=np.float64) * (
        30.0 / math.pi
    )
    for name, values in (("wind speed", winds), ("rotor speed", speeds)):
        spread = float(np.ptp(values))
        scale = max(float(np.max(np.abs(values))), 1.0)
        if spread > _SET_SPREAD_RTOL[name] * scale:
            issues.append(
                f"{name} varies by {spread:.4g} across the {len(lins)} azimuth files"
            )

    pitch, pitch_spread, pitch_source = _pitch(lins)
    if pitch_source is None:
        issues.append(
            "blade pitch unavailable: no ED pitch-command input and no BldPitch "
            "output (write the inputs with LinInputs, or add BldPitch1 to OutList)"
        )
    torque, torque_source = _generator_torque(lins)
    if torque_source is None:
        issues.append(
            "generator torque unavailable: no ED generator-torque input and no SrvD "
            "GenTq output (write the inputs with LinInputs, or add GenTq to OutList)"
        )
    thrust, thrust_source = _output_mean(lins, _THRUST_CHANNELS)
    aero_power, power_source = _output_mean(lins, _AERO_POWER_CHANNELS)
    return TrimmedOperatingPoint(
        wind_speed=float(np.mean(winds)),
        rotor_speed_rpm=float(np.mean(speeds)),
        blade_pitch_deg=pitch,
        blade_pitch_spread_deg=pitch_spread,
        generator_torque=torque,
        n_lin=len(lins),
        pitch_source=pitch_source,
        generator_torque_source=torque_source,
        rotor_thrust=thrust,
        rotor_thrust_source=thrust_source,
        aerodyn_power=aero_power,
        issues=tuple(issues),
        aerodyn_power_source=power_source,
        trim_damping=_trim_damping(lins[0].path),
    )


_LIN_NAME_RE = re.compile(r"(?P<root>.+)\.\d+\.lin")


def _trim_damping(lin_path: Path) -> float | None:
    """Return the CalcSteady trim damping of the deck that wrote ``lin_path``.

    The deck is ``<root>.fst`` beside ``<root>.<k>.lin``. Returns the larger of
    ``Twr_Kdmp`` and ``Bld_Kdmp`` (``0.0`` when ``CalcSteady`` is off or the keys are
    absent), or ``None`` when the deck cannot be found or read.
    """
    match = _LIN_NAME_RE.fullmatch(lin_path.name)
    if match is None:
        return None
    fst = lin_path.with_name(f"{match.group('root')}.fst")
    try:
        if not fst.is_file():
            return None
        deck = FastInputFile.read(fst)
    except (FastInputError, OSError):
        return None
    if "CalcSteady" in deck and deck.get("CalcSteady").strip().lower() in {
        "false",
        "f",
        ".false.",
    }:
        return 0.0
    factors = []
    for keyword in ("Twr_Kdmp", "Bld_Kdmp"):
        if keyword in deck:
            try:
                factors.append(abs(fortran_float(deck.get(keyword))))
            except ValueError:
                return None
    return max(factors, default=0.0)


def _matching(table: OperatingPointTable, pattern: re.Pattern[str]) -> list[int]:
    """Return the indices of the channels whose description matches ``pattern``."""
    return [
        i for i, desc in enumerate(table.descriptions) if pattern.match(desc.strip())
    ]


def _output_channel(table: OperatingPointTable, name: str) -> list[tuple[int, float]]:
    """Return ``(index, factor-to-SI)`` for every output channel named ``name``."""
    found: list[tuple[int, float]] = []
    for i, desc in enumerate(table.descriptions):
        match = _OUTPUT_RE.match(desc.strip())
        if match is None or match.group("name").strip().lower() != name.lower():
            continue
        factor = _OUTPUT_UNIT_TO_SI.get(match.group("unit").strip().lower())
        if factor is not None:
            found.append((i, factor))
    return found


def _pitch(lins: Sequence[LinFile]) -> tuple[float, float, str | None]:
    """Return the mean and spread of the blade pitch (deg) and its source."""
    to_deg = 180.0 / math.pi
    described: tuple[tuple[str, OperatingPointTable, re.Pattern[str]], ...] = (
        ("input: ED Blade <i> pitch command", lins[0].u, _BLADE_PITCH_INPUT_RE),
        ("input: ED blade pitch command", lins[0].u, _SINGLE_PITCH_INPUT_RE),
        (
            "input: ED collective blade-pitch command",
            lins[0].u,
            _COLLECTIVE_PITCH_INPUT_RE,
        ),
        ("output: ED Blade <i> pitch angle", lins[0].y, _BLADE_PITCH_ANGLE_OUTPUT_RE),
    )
    for label, table, pattern in described[:3]:
        if _matching(table, pattern):
            values = to_deg * np.concatenate(
                [lin.u.values[_matching(lin.u, pattern)] for lin in lins]
            )
            return float(np.mean(values)), float(np.ptp(values)), label
    names = ("BldPitch1", "BldPitch2", "BldPitch3")
    if any(_output_channel(lins[0].y, name) for name in names):
        samples = [
            float(lin.y.values[index]) * factor
            for lin in lins
            for name in names
            for index, factor in _output_channel(lin.y, name)
        ]
        values = np.asarray(samples, dtype=np.float64)
        return float(np.mean(values)), float(np.ptp(values)), "output: ED BldPitch<i>"
    label, table, pattern = described[3]
    if _matching(table, pattern):
        values = to_deg * np.concatenate(
            [lin.y.values[_matching(lin.y, pattern)] for lin in lins]
        )
        return float(np.mean(values)), float(np.ptp(values)), label
    return math.nan, math.nan, None


def _generator_torque(lins: Sequence[LinFile]) -> tuple[float, str | None]:
    """Return the mean generator torque (N m) and its source."""
    if _matching(lins[0].u, _GEN_TORQUE_INPUT_RE):
        values = [
            float(np.mean(lin.u.values[_matching(lin.u, _GEN_TORQUE_INPUT_RE)]))
            for lin in lins
        ]
        return float(np.mean(values)), "input: ED Generator torque"
    mean, source = _output_mean(lins, ("GenTq",))
    if mean is None:
        return math.nan, None
    return mean, source


def _output_mean(
    lins: Sequence[LinFile], names: Sequence[str]
) -> tuple[float | None, str | None]:
    """Return the azimuth-mean (SI) of the first output channel found, and its name."""
    for name in names:
        channels = _output_channel(lins[0].y, name)
        if not channels:
            continue
        module = lins[0].y.descriptions[channels[0][0]].split()[0]
        values = [
            float(lin.y.values[found[0][0]]) * found[0][1]
            for lin in lins
            if (found := _output_channel(lin.y, name))
        ]
        return float(np.mean(values)), f"output: {module} {name}"
    return None, None


# --------------------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrimTolerances:
    """Tolerances of the trimmed-versus-curve comparison.

    The defaults reflect what a correct deck/curve pair achieves on the reference
    turbines (rotor speed exact, pitch within a few tenths of a degree, power
    coefficient within a few percent): a looser agreement is worth investigating.

    Parameters
    ----------
    rotor_speed_rel : float, optional
        Relative rotor-speed tolerance (default 0.01, i.e. 1 %).
    pitch_deg : float, optional
        Absolute blade-pitch tolerance, in degrees (default 0.5).
    cp_rel : float, optional
        Relative power-coefficient tolerance (default 0.05).
    ct_rel : float, optional
        Relative thrust-coefficient tolerance (default 0.05).

    Raises
    ------
    ValueError
        If a tolerance is negative or non-finite.
    """

    rotor_speed_rel: float = 0.01
    pitch_deg: float = 0.5
    cp_rel: float = 0.05
    ct_rel: float = 0.05

    def __post_init__(self) -> None:
        """Validate the tolerances."""
        for name in ("rotor_speed_rel", "pitch_deg", "cp_rel", "ct_rel"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                msg = f"tolerance {name} must be finite and non-negative, got {value}"
                raise ValueError(msg)


@dataclass(frozen=True)
class QuantityCheck:
    """One trimmed quantity compared with the curve.

    Parameters
    ----------
    trimmed : float
        The value at the trimmed operating point (``NaN`` if unavailable).
    curve : float
        The curve value at the trimmed wind speed (``NaN`` if unavailable).
    tolerance : float
        The tolerance applied.
    relative : bool
        Whether ``tolerance`` bounds the relative (``True``) or absolute deviation.
    """

    trimmed: float
    curve: float
    tolerance: float
    relative: bool

    @property
    def deviation(self) -> float:
        """Return ``trimmed - curve``."""
        return self.trimmed - self.curve

    @property
    def relative_deviation(self) -> float:
        """Return ``(trimmed - curve) / |curve|`` (``NaN`` when the curve is zero)."""
        if abs(self.curve) < _ZERO_FLOOR:
            return math.nan
        return self.deviation / abs(self.curve)

    @property
    def status(self) -> CheckStatus:
        """Return ``"pass"``, ``"flag"`` or ``"unavailable"``."""
        if not (math.isfinite(self.trimmed) and math.isfinite(self.curve)):
            return "unavailable"
        if self.relative:
            if abs(self.curve) < _ZERO_FLOOR:
                within = abs(self.trimmed) < _ZERO_FLOOR
            else:
                within = abs(self.relative_deviation) <= self.tolerance
        else:
            within = abs(self.deviation) <= self.tolerance
        return "pass" if within else "flag"


@dataclass(frozen=True)
class OperatingPointCheck:
    """The verification of one trimmed operating point.

    Parameters
    ----------
    label : str
        Identifies the operating point (a sweep case name or the set index).
    trimmed : TrimmedOperatingPoint
        The trimmed point read from the ``.lin`` files.
    aero_power : float
        Aerodynamic rotor power, in W, from the low-speed-shaft torque balance, or
        AeroDyn's own rotor power when trim damping makes the balance invalid (see
        ``power_source``).
    rotor_speed : QuantityCheck
        Rotor speed (rpm) against the curve.
    pitch : QuantityCheck
        Blade pitch (deg) against the curve.
    cp : QuantityCheck
        Power coefficient against the curve.
    ct : QuantityCheck
        Thrust coefficient against the curve (unavailable without a rotor-thrust
        output or a curve ``ct``).
    issues : tuple[str, ...]
        Notes on missing data, including the trimmed point's own issues.
    power_source : str
        Where ``aero_power`` comes from: ``"torque balance"`` or the AeroDyn output
        channel (``"output: AD RtFldPwr"``).
    """

    label: str
    trimmed: TrimmedOperatingPoint
    aero_power: float
    rotor_speed: QuantityCheck
    pitch: QuantityCheck
    cp: QuantityCheck
    ct: QuantityCheck
    issues: tuple[str, ...] = ()
    power_source: str = "torque balance"

    @property
    def checks(self) -> dict[str, QuantityCheck]:
        """Return the four checks keyed by quantity name."""
        return {
            "rotor_speed": self.rotor_speed,
            "pitch": self.pitch,
            "cp": self.cp,
            "ct": self.ct,
        }

    @property
    def status(self) -> PointStatus:
        """Return ``"flag"`` if any check fails, ``"incomplete"`` if data is missing.

        A point is ``"incomplete"`` when no check fails but its rotor speed or pitch
        could not be compared, or the trimmed data carries an issue. A curve that
        simply lacks ``cp``/``ct`` does not make a point incomplete.
        """
        statuses = [check.status for check in self.checks.values()]
        if "flag" in statuses:
            return "flag"
        if self.issues or "unavailable" in (self.rotor_speed.status, self.pitch.status):
            return "incomplete"
        return "pass"

    @property
    def flagged_quantities(self) -> tuple[str, ...]:
        """Return the names of the quantities outside tolerance."""
        return tuple(
            name for name, check in self.checks.items() if check.status == "flag"
        )


@dataclass(frozen=True)
class OperatingPointVerification:
    """The trimmed-versus-curve verification of a set of linearizations.

    Parameters
    ----------
    points : tuple[OperatingPointCheck, ...]
        One check per verified operating point, in input order.
    constants : TurbineConstants
        The turbine constants used.
    tolerances : TrimTolerances
        The tolerances applied.
    skipped : tuple[str, ...], optional
        Operating points that could not be verified (e.g. a non-converged sweep case),
        each with its reason.
    """

    points: tuple[OperatingPointCheck, ...]
    constants: TurbineConstants
    tolerances: TrimTolerances
    skipped: tuple[str, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        """Return whether every verified point passes and none was skipped."""
        return not self.skipped and all(point.status == "pass" for point in self.points)

    @property
    def flagged(self) -> tuple[OperatingPointCheck, ...]:
        """Return the points with at least one quantity outside tolerance."""
        return tuple(point for point in self.points if point.status == "flag")

    def to_records(self) -> list[dict[str, float | str | None]]:
        """Return one flat record per point, e.g. for a :class:`pandas.DataFrame`.

        Returns
        -------
        list[dict[str, float or str or None]]
            Keys: ``label``, ``wind_speed``, ``generator_torque``, ``aero_power``,
            ``power_source``, ``aerodyn_power``, ``trim_damping``, ``rotor_thrust``,
            ``status``, and for each of
            ``rotor_speed``, ``pitch``, ``cp`` and ``ct`` the ``<q>``,
            ``<q>_curve``, ``<q>_deviation`` and ``<q>_status`` entries.
        """
        records: list[dict[str, float | str | None]] = []
        for point in self.points:
            record: dict[str, float | str | None] = {
                "label": point.label,
                "wind_speed": point.trimmed.wind_speed,
                "generator_torque": point.trimmed.generator_torque,
                "aero_power": point.aero_power,
                "power_source": point.power_source,
                "aerodyn_power": point.trimmed.aerodyn_power,
                "trim_damping": point.trimmed.trim_damping,
                "rotor_thrust": point.trimmed.rotor_thrust,
                "status": point.status,
            }
            for name, check in point.checks.items():
                record[name] = check.trimmed
                record[f"{name}_curve"] = check.curve
                record[f"{name}_deviation"] = check.deviation
                record[f"{name}_status"] = check.status
            records.append(record)
        return records

    def to_table(self) -> str:
        """Return a plain-text table of the verification.

        Returns
        -------
        str
            One row per point: wind, rotor speed (trimmed/curve), pitch
            (trimmed/curve/deviation), generator torque, aerodynamic power, Cp
            (trimmed/curve/relative deviation), Ct (likewise) and the status.
        """
        columns = (
            ("label", 10),
            ("wind", 6),
            ("rpm", 7),
            ("rpm_c", 7),
            ("pitch", 6),
            ("pitch_c", 7),
            ("dpitch", 6),
            ("GenTq_kNm", 9),
            ("P_kW", 8),
            ("Cp", 6),
            ("Cp_c", 6),
            ("dCp%", 6),
            ("Ct", 6),
            ("Ct_c", 6),
            ("dCt%", 6),
        )
        header = " ".join(f"{name:>{width}}" for name, width in columns) + " status"
        lines = [header]
        lines.extend(_table_row(point) for point in self.points)
        lines.extend(f"skipped: {reason}" for reason in self.skipped)
        return "\n".join(lines)


def _table_row(p: OperatingPointCheck) -> str:
    """Format one point as a row of :meth:`OperatingPointVerification.to_table`."""
    flagged = f" [{', '.join(p.flagged_quantities)}]" if p.flagged_quantities else ""
    cells = (
        f"{p.label:>10}",
        f"{p.trimmed.wind_speed:6.2f}",
        f"{p.rotor_speed.trimmed:7.3f}",
        f"{p.rotor_speed.curve:7.3f}",
        f"{p.pitch.trimmed:6.2f}",
        f"{p.pitch.curve:7.2f}",
        f"{p.pitch.deviation:+6.2f}",
        f"{p.trimmed.generator_torque / 1e3:9.2f}",
        f"{p.aero_power / 1e3:8.1f}",
        f"{p.cp.trimmed:6.4f}",
        f"{p.cp.curve:6.4f}",
        f"{100.0 * p.cp.relative_deviation:+6.2f}",
        f"{p.ct.trimmed:6.4f}",
        f"{p.ct.curve:6.4f}",
        f"{100.0 * p.ct.relative_deviation:+6.2f}",
    )
    return " ".join(cells) + f" {p.status}{flagged}"


def aerodynamic_power(
    generator_torque: float, rotor_speed_rpm: float, constants: TurbineConstants
) -> float:
    r"""Return the aerodynamic rotor power from the steady torque balance.

    Parameters
    ----------
    generator_torque : float
        High-speed-shaft generator torque, in N m.
    rotor_speed_rpm : float
        Rotor speed, in rpm.
    constants : TurbineConstants
        Supplies the gearbox ratio and efficiency.

    Returns
    -------
    float
        :math:`N Q_g \Omega \eta^{-\operatorname{sgn}(Q_g\Omega)}`, in W: the
        gearbox loss is added back when power flows rotor to generator and subtracted
        when the generator motors the rotor.
    """
    omega = rotor_speed_rpm * math.pi / 30.0
    hss_power = generator_torque * constants.gearbox_ratio * omega
    if hss_power >= 0.0:
        return hss_power / constants.gearbox_efficiency
    return hss_power * constants.gearbox_efficiency


def _check_point(
    label: str,
    trimmed: TrimmedOperatingPoint,
    curve: RotorPerformanceCurve,
    constants: TurbineConstants,
    tolerances: TrimTolerances,
) -> OperatingPointCheck:
    """Compare one trimmed point with the curve."""
    issues = list(trimmed.issues)
    wind = trimmed.wind_speed
    reference = curve.at(wind)
    if not math.isfinite(reference["rotor_speed_rpm"]):
        issues.append(
            f"wind speed {wind:g} m/s is outside the curve range "
            f"[{curve.wind_speed[0]:g}, {curve.wind_speed[-1]:g}] m/s"
        )
    power = aerodynamic_power(
        trimmed.generator_torque, trimmed.rotor_speed_rpm, constants
    )
    power_source = "torque balance"
    if trimmed.trim_damping is not None and trimmed.trim_damping > 0.0:
        # CalcSteady's artificial blade/tower damping absorbs rotor torque, so the
        # trimmed generator torque no longer balances the aerodynamic torque.
        if trimmed.aerodyn_power is not None and math.isfinite(trimmed.aerodyn_power):
            power = trimmed.aerodyn_power
            power_source = trimmed.aerodyn_power_source or "AeroDyn rotor power"
        else:
            logger.warning(
                "%s: CalcSteady trim damping (Twr_Kdmp/Bld_Kdmp up to %g N/(m/s)) can "
                "absorb rotor torque, so the torque-balance power may be low; add "
                "RtFldPwr to the AeroDyn OutList to use AeroDyn's rotor power instead",
                label,
                trimmed.trim_damping,
            )
    cp = ct = math.nan
    if wind > 0.0:
        dynamic = 0.5 * constants.air_density * constants.rotor_area * wind**2
        cp = power / (dynamic * wind)
        if trimmed.rotor_thrust is not None:
            ct = trimmed.rotor_thrust / dynamic
    else:
        issues.append("zero wind speed: power and thrust coefficients are undefined")
    return OperatingPointCheck(
        label=label,
        trimmed=trimmed,
        aero_power=power,
        rotor_speed=QuantityCheck(
            trimmed.rotor_speed_rpm,
            reference["rotor_speed_rpm"],
            tolerances.rotor_speed_rel,
            relative=True,
        ),
        pitch=QuantityCheck(
            trimmed.blade_pitch_deg,
            reference["blade_pitch_deg"],
            tolerances.pitch_deg,
            relative=False,
        ),
        cp=QuantityCheck(cp, reference["cp"], tolerances.cp_rel, relative=True),
        ct=QuantityCheck(ct, reference["ct"], tolerances.ct_rel, relative=True),
        issues=tuple(issues),
        power_source=power_source,
    )


def _as_curve(
    curve: RotorPerformanceCurve
    | str
    | Path
    | tuple[Sequence[float], Sequence[float], Sequence[float]],
) -> RotorPerformanceCurve:
    """Coerce the accepted curve forms to a :class:`RotorPerformanceCurve`."""
    if isinstance(curve, RotorPerformanceCurve):
        return curve
    if isinstance(curve, (str, Path)):
        return rotor_performance_from_csv(curve)
    wind, rotor, pitch = curve
    return RotorPerformanceCurve.from_arrays(wind, rotor, pitch)


def _resolve_constants(
    deck: str | Path | None,
    constants: TurbineConstants | None,
    radius_convention: RadiusConvention,
) -> TurbineConstants:
    """Return explicit constants, or read them from the deck."""
    if constants is not None:
        return constants
    if deck is None:
        msg = "pass either deck= (an .fst path) or constants= (TurbineConstants)"
        raise ValueError(msg)
    return TurbineConstants.from_deck(deck, radius_convention=radius_convention)


def verify_operating_points(
    lin_sets: Sequence[Sequence[LinFile | str | os.PathLike[str]]],
    curve: RotorPerformanceCurve
    | str
    | Path
    | tuple[Sequence[float], Sequence[float], Sequence[float]],
    *,
    deck: str | Path | None = None,
    constants: TurbineConstants | None = None,
    radius_convention: RadiusConvention = "projected",
    tolerances: TrimTolerances | None = None,
    labels: Sequence[str] | None = None,
) -> OperatingPointVerification:
    r"""Verify the trimmed operating points of linearizations against a curve.

    For each linearization (its set of azimuth ``.lin`` files) the trimmed wind,
    rotor speed, blade pitch and generator torque are read back
    (:func:`trimmed_operating_point`), the aerodynamic power is derived from the steady
    low-speed-shaft torque balance (:func:`aerodynamic_power`) and the power
    coefficient :math:`C_p = P / (\tfrac12\rho\pi R^2U^3)` formed. The thrust
    coefficient is formed only when the outputs carry AeroDyn's rotor thrust
    (``RtFldFxh`` or ``RtAeroFxh`` in the AeroDyn ``OutList``). Each quantity is
    compared with the curve interpolated at the trimmed wind speed.

    Below rated, ``CalcSteady`` prescribes the rotor speed and pitch from the curve,
    so their agreement is by construction and :math:`C_p` (from the trimmed torque) is
    the informative check; above rated the trimmed pitch is.

    When the deck beside a set's files trimmed with artificial damping (non-zero
    ``Twr_Kdmp``/``Bld_Kdmp`` with ``CalcSteady``), the damping absorbs part of the
    rotor torque and the generator torque no longer balances the aerodynamic torque.
    The power is then AeroDyn's rotor power output when present
    (:attr:`OperatingPointCheck.power_source` says which was used); without it the
    balance is kept and a warning logged.

    Parameters
    ----------
    lin_sets : Sequence[Sequence[LinFile or str or os.PathLike]]
        One entry per operating point: the ``.lin`` files of that linearization.
    curve : RotorPerformanceCurve or str or pathlib.Path or tuple
        The user's rotor-performance data. A path is read with
        :func:`rotor_performance_from_csv` (so its ``cp``/``ct`` columns are used); a
        3-tuple such as :func:`~vane.run.operating_curve_from_csv` returns carries no
        coefficients, so only rotor speed and pitch are compared.
    deck : str or pathlib.Path or None, optional
        An ``.fst`` deck of the turbine, read for the constants
        (:meth:`TurbineConstants.from_deck`). Required unless ``constants`` is given.
    constants : TurbineConstants or None, optional
        Explicit constants; take precedence over ``deck``.
    radius_convention : {"projected", "tip"}, optional
        Radius used with ``deck`` (see :meth:`TurbineConstants.from_deck`).
    tolerances : TrimTolerances or None, optional
        Comparison tolerances (default :class:`TrimTolerances`).
    labels : Sequence[str] or None, optional
        A label per set; defaults to ``"ws<wind>"``.

    Returns
    -------
    OperatingPointVerification
        The per-point checks and overall outcome.

    Raises
    ------
    ValueError
        If neither ``deck`` nor ``constants`` is given, ``labels`` has the wrong
        length, a set is empty, or the deck/curve cannot be read.
    """
    curve_data = _as_curve(curve)
    turbine = _resolve_constants(deck, constants, radius_convention)
    limits = tolerances if tolerances is not None else TrimTolerances()
    if labels is not None and len(labels) != len(lin_sets):
        msg = f"got {len(labels)} labels for {len(lin_sets)} linearization sets"
        raise ValueError(msg)
    points = []
    for index, lin_set in enumerate(lin_sets):
        trimmed = trimmed_operating_point(lin_set)
        label = labels[index] if labels is not None else f"ws{trimmed.wind_speed:g}"
        points.append(_check_point(label, trimmed, curve_data, turbine, limits))
    report = OperatingPointVerification(
        points=tuple(points), constants=turbine, tolerances=limits
    )
    for point in report.flagged:
        logger.warning(
            "trimmed operating point %s deviates from the curve in %s",
            point.label,
            ", ".join(point.flagged_quantities),
        )
    return report


def verify_sweep_operating_points(
    sweep: OperatingSweepResult,
    curve: RotorPerformanceCurve
    | str
    | Path
    | tuple[Sequence[float], Sequence[float], Sequence[float]],
    *,
    constants: TurbineConstants | None = None,
    radius_convention: RadiusConvention = "projected",
    tolerances: TrimTolerances | None = None,
) -> OperatingPointVerification:
    """Verify the trimmed points of a :func:`~vane.run.run_operating_sweep` result.

    Every converged case with ``.lin`` files is verified, labelled by its operating
    point's name; a case that did not converge or produced no ``.lin`` files is listed
    in :attr:`OperatingPointVerification.skipped`. Unless ``constants`` is given, the
    turbine constants are read from the first verified case's configured deck.

    Parameters
    ----------
    sweep : OperatingSweepResult
        The sweep result.
    curve : RotorPerformanceCurve or str or pathlib.Path or tuple
        The rotor-performance data (see :func:`verify_operating_points`); pass the
        same CSV the sweep used to get the ``cp``/``ct`` comparison too.
    constants : TurbineConstants or None, optional
        Explicit constants.
    radius_convention : {"projected", "tip"}, optional
        Radius used when reading the constants from the deck.
    tolerances : TrimTolerances or None, optional
        Comparison tolerances.

    Returns
    -------
    OperatingPointVerification
        The per-point checks.

    Raises
    ------
    ValueError
        If no case of the sweep can be verified.
    """
    usable = []
    skipped = []
    for case in sweep.cases:
        name = case.operating_point.name or f"ws{case.operating_point.wind_speed:g}"
        if not case.converged:
            skipped.append(f"{name}: linearization did not converge")
        elif not case.lin_files:
            skipped.append(f"{name}: no .lin files")
        else:
            usable.append((name, case))
    if not usable:
        msg = "no converged sweep case with .lin files to verify; " + "; ".join(skipped)
        raise ValueError(msg)
    report = verify_operating_points(
        [case.lin_files for _, case in usable],
        curve,
        deck=usable[0][1].fst_path,
        constants=constants,
        radius_convention=radius_convention,
        tolerances=tolerances,
        labels=[name for name, _ in usable],
    )
    for reason in skipped:
        logger.warning("operating-point verification skipped %s", reason)
    return OperatingPointVerification(
        points=report.points,
        constants=report.constants,
        tolerances=report.tolerances,
        skipped=tuple(skipped),
    )
