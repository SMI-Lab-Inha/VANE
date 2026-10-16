# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Azimuth-density calibration of the averaged-model modal frequencies.

The azimuth-averaged (Coleman/MBC) model is only as good as its azimuth sampling: too
few azimuths and the average does not remove the rotor's periodic content, so the
Campbell frequencies are biased. VANE's "thin azimuth coverage" warning is otherwise an
empirical judgement; this module replaces it with a **calibration curve** — how the
averaged natural frequencies converge as the azimuth count grows — measured against the
densest available sampling as the reference.

Given a dense per-azimuth set, :func:`azimuth_density_sweep` sub-samples it to a series
of azimuth counts (1, 3, 6, 12, 36, ...), re-averages, and reports the natural-frequency
error of each coarse model relative to the full-density reference.
:func:`recommend_azimuth_count` then reads off the smallest azimuth count that brings
that error under a tolerance — a data-grounded answer to "how many azimuths are enough"
for a given turbine, in place of a hard-coded rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

__all__ = [
    "AzimuthConvergence",
    "azimuth_density_sweep",
    "recommend_azimuth_count",
]

_TWO_PI = 2.0 * np.pi
_DEFAULT_DENSITIES = (1, 3, 6, 12, 36)
_IMAG_TOLERANCE = 1e-8


@dataclass
class AzimuthConvergence:
    """How the averaged-model natural frequencies converge with azimuth count.

    Parameters
    ----------
    densities : npt.NDArray[np.int_]
        The azimuth counts evaluated, ascending.
    frequency_error_hz : npt.NDArray[np.float64]
        For each density, the maximum natural-frequency difference (Hz) between that
        coarse averaged model and the full-density reference.
    natural_frequencies_hz : npt.NDArray[np.float64]
        The sorted natural frequencies (Hz) at each density, shape
        ``(n_density, n_frequencies)``.
    reference_frequencies_hz : npt.NDArray[np.float64]
        The reference (full-density) sorted natural frequencies (Hz).
    """

    densities: npt.NDArray[np.int_]
    frequency_error_hz: npt.NDArray[np.float64]
    natural_frequencies_hz: npt.NDArray[np.float64]
    reference_frequencies_hz: npt.NDArray[np.float64]


def _natural_frequencies(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the sorted natural frequencies (Hz) of the under-damped modes.

    Only the under-damped modes (one conjugate representative each, positive imaginary
    part) are reported, matching :func:`vane.modal.compute_modes`; real/over-damped and
    rigid-body eigenvalues are excluded so they do not inflate the frequency list.
    """
    eigvals = np.linalg.eigvals(matrix)
    underdamped = eigvals[eigvals.imag > _IMAG_TOLERANCE]
    return np.sort(np.abs(underdamped) / _TWO_PI)


def azimuth_density_sweep(
    per_azimuth_a: npt.ArrayLike,
    azimuths: npt.ArrayLike,
    *,
    densities: Sequence[int] = _DEFAULT_DENSITIES,
) -> AzimuthConvergence:
    """Measure how the averaged model's frequencies converge with azimuth count.

    Sub-samples the (azimuth-sorted) per-azimuth matrices to each requested count, takes
    the mean as the averaged model, and compares its natural frequencies to those of the
    full-density average (the reference).

    Parameters
    ----------
    per_azimuth_a : npt.ArrayLike
        The frozen-time state matrices, shape ``(n_azimuth, n, n)`` (e.g.
        ``MBCResult.per_azimuth_a``).
    azimuths : npt.ArrayLike
        The rotor azimuth of each matrix; used only to order the samples, so any
        consistent unit works (degrees or radians, e.g. ``MBCResult.azimuths_deg``).
    densities : Sequence[int], optional
        Azimuth counts to evaluate; counts above the available number are skipped.
        Default ``(1, 3, 6, 12, 36)``.

    Returns
    -------
    AzimuthConvergence
        The per-density frequencies and their error versus the full-density reference.

    Raises
    ------
    ValueError
        If the matrices are not a consistent ``(n_azimuth, n, n)`` stack, the matrices
        or azimuths are non-finite, the azimuths do not match, a requested density is
        non-finite or not a whole number, or no requested density is achievable.
    """
    matrices = np.asarray(per_azimuth_a, dtype=np.float64)
    angles = np.asarray(azimuths, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        msg = f"per_azimuth_a must be (n_azimuth, n, n), got {matrices.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(matrices)):
        msg = "per_azimuth_a must be finite (no NaN or inf)"
        raise ValueError(msg)
    n_azimuth = matrices.shape[0]
    if angles.shape != (n_azimuth,):
        msg = f"azimuths must have {n_azimuth} angles, got {angles.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(angles)):
        msg = "azimuths must be finite"
        raise ValueError(msg)
    matrices = matrices[
        np.argsort(angles)
    ]  # order only; the azimuth unit is irrelevant

    # Densities are azimuth *counts*: reject a non-finite or fractional value rather
    # than silently truncating it (``int(2.9) == 2``), which would evaluate a different
    # density than the caller asked for.
    for d in densities:
        if not np.isfinite(d) or float(d) != int(d):
            msg = f"densities must be finite whole-number azimuth counts; got {d!r}"
            raise ValueError(msg)
    counts = sorted({int(d) for d in densities if 1 <= int(d) <= n_azimuth})
    if not counts:
        msg = (
            f"no requested density is achievable with {n_azimuth} azimuths "
            f"(requested {sorted(set(densities))})"
        )
        raise ValueError(msg)

    reference = _natural_frequencies(matrices.mean(axis=0))
    per_density = []
    errors = []
    for count in counts:
        indices = (np.arange(count) * n_azimuth) // count  # evenly spaced
        frequencies = _natural_frequencies(matrices[indices].mean(axis=0))
        if frequencies.shape != reference.shape:
            msg = (
                f"the averaged model at {count} azimuths has {frequencies.size} "
                f"under-damped modes vs {reference.size} at full density; the modal "
                f"structure changed across the sweep, so frequencies cannot be matched"
            )
            raise ValueError(msg)
        per_density.append(frequencies)
        errors.append(float(np.max(np.abs(frequencies - reference))))
    return AzimuthConvergence(
        densities=np.asarray(counts, dtype=np.int_),
        frequency_error_hz=np.asarray(errors, dtype=np.float64),
        natural_frequencies_hz=np.asarray(per_density, dtype=np.float64),
        reference_frequencies_hz=reference,
    )


def recommend_azimuth_count(
    convergence: AzimuthConvergence, *, tolerance_hz: float
) -> int:
    """Return the smallest evaluated azimuth count whose frequency error is within tol.

    Reads the convergence curve for the fewest azimuths at which the averaged model's
    natural-frequency error (and that of every denser model evaluated) stays at or below
    ``tolerance_hz`` — the data-grounded minimum azimuth coverage.

    Parameters
    ----------
    convergence : AzimuthConvergence
        A sweep from :func:`azimuth_density_sweep`.
    tolerance_hz : float
        The acceptable natural-frequency error (Hz).

    Returns
    -------
    int
        The recommended azimuth count.

    Raises
    ------
    ValueError
        If ``tolerance_hz`` is not positive, or no evaluated density meets it (the
        sampling does not converge within tolerance — use more azimuths).
    """
    if tolerance_hz <= 0.0:
        msg = f"tolerance_hz must be positive, got {tolerance_hz}"
        raise ValueError(msg)
    densities = convergence.densities
    errors = convergence.frequency_error_hz
    # Keep a count only if it and every denser evaluated model are within tolerance
    # (a suffix-AND over the ascending-density curve), so a lucky coarse hit is ignored.
    within = (errors <= tolerance_hz).astype(int)
    stays_within = np.flip(np.minimum.accumulate(np.flip(within))).astype(bool)
    candidates = densities[stays_within]
    if candidates.size == 0:
        msg = (
            f"no evaluated azimuth count reaches {tolerance_hz} Hz "
            f"(min error {float(errors.min()):.3g} Hz); use more azimuths"
        )
        raise ValueError(msg)
    return int(candidates.min())
