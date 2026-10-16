# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Per-rev excitation lines and resonance detection.

A rotor turning at ``Omega`` excites the structure at integer multiples of the
rotational frequency: the ``nP`` lines, with frequency ``n * rpm / 60`` Hz. A
potential resonance occurs where a mode's natural-frequency curve crosses an ``nP``
line. For a three-bladed rotor the dominant excitations are 1P (rotor imbalance)
and the blade-passing 3P, 6P, and 9P lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from vane.campbell.builder import CampbellDiagram

__all__ = [
    "DEFAULT_HARMONICS",
    "ResonanceCrossing",
    "ResonanceSeverity",
    "excitation_frequencies",
    "find_resonances",
]

DEFAULT_HARMONICS: tuple[int, ...] = (1, 3, 6, 9)
_SECONDS_PER_MINUTE = 60.0

# Damping-ratio thresholds for resonance severity: a lightly damped mode crossing an
# excitation line is more dangerous than a well-damped one.
_SEVERITY_HIGH_BELOW_DAMPING = 0.01
_SEVERITY_MEDIUM_BELOW_DAMPING = 0.05


class ResonanceSeverity(Enum):
    """Damping-aware severity of a resonance crossing.

    A crossing's danger depends on how lightly damped the resonating mode is: a
    near-undamped mode at a crossing can build large response, while a well-damped one
    is benign.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


def _severity_from_damping(damping_ratio: float) -> ResonanceSeverity:
    """Classify a crossing's severity from the resonating mode's damping ratio."""
    if damping_ratio < _SEVERITY_HIGH_BELOW_DAMPING:
        return ResonanceSeverity.HIGH
    if damping_ratio < _SEVERITY_MEDIUM_BELOW_DAMPING:
        return ResonanceSeverity.MEDIUM
    return ResonanceSeverity.LOW


def _validate_harmonics(harmonics: Sequence[int]) -> list[int]:
    """Return the harmonics as validated positive, unique integers.

    Parameters
    ----------
    harmonics : Sequence[int]
        Candidate excitation harmonics.

    Returns
    -------
    list[int]
        The harmonics as a list of ``int`` (empty if none were given).

    Raises
    ------
    ValueError
        If any harmonic is not a finite positive integer, or the harmonics are not
        unique. An ``nP`` line is only physically meaningful for a positive integer
        ``n``; an empty sequence is allowed and simply requests no excitation lines.
    """
    values = list(harmonics)
    cleaned: list[int] = []
    for harmonic in values:
        number = float(harmonic)
        if not np.isfinite(number) or number != int(number) or int(number) < 1:
            msg = f"harmonics must be positive integers, got {harmonic!r}"
            raise ValueError(msg)
        cleaned.append(int(number))
    if len(set(cleaned)) != len(cleaned):
        msg = f"harmonics must be unique, got {values!r}"
        raise ValueError(msg)
    return cleaned


@dataclass(frozen=True)
class ResonanceCrossing:
    """A crossing of a mode curve with an ``nP`` excitation line.

    Parameters
    ----------
    track_index : int
        Index of the crossing mode track within the diagram.
    harmonic : int
        The excitation harmonic ``n`` (the ``nP`` line).
    rotor_speed_rpm : float
        Rotor speed at the crossing, interpolated between operating points.
    frequency_hz : float
        Frequency at the crossing, in hertz.
    damping_ratio : float
        Damping ratio of the resonating mode at the crossing, interpolated between
        operating points.
    severity : ResonanceSeverity
        Damping-aware severity of the crossing.
    """

    track_index: int
    harmonic: int
    rotor_speed_rpm: float
    frequency_hz: float
    damping_ratio: float
    severity: ResonanceSeverity


def excitation_frequencies(
    rotor_speed_rpm: npt.NDArray[np.float64], harmonics: Sequence[int]
) -> dict[int, npt.NDArray[np.float64]]:
    """Return the ``nP`` excitation-line frequencies at each rotor speed.

    Parameters
    ----------
    rotor_speed_rpm : npt.NDArray[np.float64]
        Rotor speed in rev/min, shape ``(n_points,)``.
    harmonics : Sequence[int]
        The harmonics ``n`` to evaluate.

    Returns
    -------
    dict[int, npt.NDArray[np.float64]]
        Mapping of each harmonic ``n`` to its ``nP`` frequency (Hz) at every rotor
        speed, ``n * rpm / 60``.

    Raises
    ------
    ValueError
        If ``harmonics`` contains a non-positive, fractional, non-finite, or
        duplicate value.
    """
    validated = _validate_harmonics(harmonics)
    rpm = np.asarray(rotor_speed_rpm, dtype=np.float64)
    return {n: n * rpm / _SECONDS_PER_MINUTE for n in validated}


def find_resonances(
    diagram: CampbellDiagram,
    harmonics: Sequence[int] = DEFAULT_HARMONICS,
    *,
    operating_range: tuple[float, float] | None = None,
) -> list[ResonanceCrossing]:
    """Find where mode curves cross the ``nP`` excitation lines.

    A crossing is detected wherever ``natural_frequency - n * rpm / 60`` changes
    sign (or is zero) between consecutive operating points along a track; the rotor
    speed, frequency, and damping ratio at the crossing are linearly interpolated, and
    each crossing is assigned a damping-aware :class:`ResonanceSeverity`.

    The rotor-speed samples need not be strictly increasing: each track's curve is
    sorted by rotor speed and duplicate rotor speeds are collapsed internally before the
    crossings are found, so a non-monotone or tied sweep is handled gracefully rather
    than producing spurious or zero-width crossings.

    Parameters
    ----------
    diagram : CampbellDiagram
        A Campbell diagram whose operating parameter is rotor speed in rev/min.
    harmonics : Sequence[int], optional
        Excitation harmonics to test (default 1P, 3P, 6P, 9P). Any positive-integer
        family may be supplied.
    operating_range : tuple[float, float] or None, optional
        If given, only crossings whose rotor speed falls within this
        ``(min_rpm, max_rpm)`` operating window are returned.

    Returns
    -------
    list[ResonanceCrossing]
        The detected crossings, in track-then-harmonic order.

    Raises
    ------
    ValueError
        If the diagram's operating parameter is not rotor speed in rev/min,
        ``harmonics`` contains a non-positive, fractional, non-finite, or duplicate
        value, or ``operating_range`` is given with ``min_rpm > max_rpm``.
    """
    if diagram.parameter_name != "rotor_speed_rpm":
        msg = (
            "find_resonances requires a rotor-speed (rpm) Campbell diagram; "
            f"got parameter '{diagram.parameter_name}'"
        )
        raise ValueError(msg)
    validated_harmonics = _validate_harmonics(harmonics)
    if operating_range is not None:
        low, high = operating_range
        if not (np.isfinite(low) and np.isfinite(high)):
            msg = f"operating_range must be finite, got {operating_range}"
            raise ValueError(msg)
        if low > high:
            msg = f"operating_range min exceeds max: {operating_range}"
            raise ValueError(msg)

    crossings: list[ResonanceCrossing] = []
    for track_index, track in enumerate(diagram.tracks):
        rpm, frequency, damping = _ordered_track_curve(*diagram.track_curve(track))
        if rpm.shape[0] < 2:
            continue
        for harmonic in validated_harmonics:
            difference = frequency - harmonic * rpm / _SECONDS_PER_MINUTE
            crossings.extend(
                _crossings_along_track(
                    track_index, harmonic, rpm, frequency, damping, difference
                )
            )
    if operating_range is not None:
        low, high = operating_range
        crossings = [c for c in crossings if low <= c.rotor_speed_rpm <= high]
    return crossings


def _ordered_track_curve(
    rpm: npt.NDArray[np.float64],
    frequency: npt.NDArray[np.float64],
    damping: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Sort a track's samples by rotor speed and drop duplicate-rpm points.

    Crossing detection walks consecutive samples and interpolates between them, which
    assumes the samples increase in rotor speed. A sweep whose rotor speed is not
    strictly increasing (e.g. two operating points at the same rpm, or a non-monotone
    wind-to-rpm map) would otherwise give a zero-width interpolation interval or a
    spurious crossing. Sorting by rpm and collapsing ties (keeping the first occurrence)
    makes the detection independent of the input order, so the caller need not
    pre-sort the sweep.

    Parameters
    ----------
    rpm, frequency, damping : npt.NDArray[np.float64]
        The track's rotor speed (rev/min), natural frequency (Hz), and damping ratio,
        in operating-point order.

    Returns
    -------
    tuple of npt.NDArray[np.float64]
        ``(rpm, frequency, damping)`` sorted by ascending rotor speed with duplicate
        rotor speeds removed.
    """
    order = np.argsort(rpm, kind="stable")
    rpm_sorted = rpm[order]
    keep = np.ones(rpm_sorted.shape[0], dtype=bool)
    if rpm_sorted.shape[0] > 1:
        keep[1:] = np.diff(rpm_sorted) > 0.0  # drop exact rpm ties, keep the first
    selected = order[keep]
    return rpm[selected], frequency[selected], damping[selected]


def _crossings_along_track(
    track_index: int,
    harmonic: int,
    rpm: npt.NDArray[np.float64],
    frequency: npt.NDArray[np.float64],
    damping: npt.NDArray[np.float64],
    difference: npt.NDArray[np.float64],
) -> list[ResonanceCrossing]:
    """Locate sign changes of ``difference`` along one track-harmonic pair."""
    found: list[ResonanceCrossing] = []
    for k in range(difference.shape[0] - 1):
        d0, d1 = float(difference[k]), float(difference[k + 1])
        if d0 == 0.0:
            found.append(
                _crossing_at(track_index, harmonic, rpm, frequency, damping, k, 0.0)
            )
        elif d0 * d1 < 0.0:
            found.append(
                _crossing_at(
                    track_index, harmonic, rpm, frequency, damping, k, d0 / (d0 - d1)
                )
            )
    if float(difference[-1]) == 0.0:
        last = difference.shape[0] - 1
        found.append(
            _crossing_at(track_index, harmonic, rpm, frequency, damping, last, 0.0)
        )
    return found


def _crossing_at(
    track_index: int,
    harmonic: int,
    rpm: npt.NDArray[np.float64],
    frequency: npt.NDArray[np.float64],
    damping: npt.NDArray[np.float64],
    k: int,
    t: float,
) -> ResonanceCrossing:
    """Build a crossing interpolated a fraction ``t`` past sample index ``k``."""

    def interp(values: npt.NDArray[np.float64]) -> float:
        nxt = values[k + 1] if k + 1 < values.shape[0] else values[k]
        return float(values[k] + t * (nxt - values[k]))

    damping_ratio = interp(damping)
    return ResonanceCrossing(
        track_index=track_index,
        harmonic=harmonic,
        rotor_speed_rpm=interp(rpm),
        frequency_hz=interp(frequency),
        damping_ratio=damping_ratio,
        severity=_severity_from_damping(damping_ratio),
    )
