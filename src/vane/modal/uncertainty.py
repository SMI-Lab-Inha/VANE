# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Uncertainty quantification for modal estimates.

Two physically grounded uncertainty sources are combined into one per-mode
confidence:

- **Azimuth spread.** The MBC transform produces one non-rotating state matrix per
  azimuth; their average is the linear-time-invariant model. For an isotropic rotor
  every per-azimuth matrix is identical (azimuth-invariant to machine precision), so
  the eigenvalue spread across azimuths is ~0; for an anisotropic rotor the spread
  grows, quantifying how well the averaged model represents the periodic system.
- **Numerical conditioning and degeneracy** from the eigen-analysis.

The unified confidence multiplies independent reliability factors (each in ``[0, 1]``)
so that a mode is trusted only when it is numerically well posed, non-degenerate,
azimuth-stable, and (optionally) confidently tracked. It feeds the covariance
initialization of the state-space export.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from vane.modal.eigensolver import modes_from_mbc

if TYPE_CHECKING:
    import numpy.typing as npt

    from vane.io.mbc_transform import MBCResult
    from vane.modal.eigensolver import ModalSolution

__all__ = ["AzimuthSpread", "azimuth_spread", "unified_mode_confidence"]

_TWO_PI = 2.0 * np.pi


@dataclass
class AzimuthSpread:
    """Per-mode variation of the modal estimates across azimuth.

    Parameters
    ----------
    natural_frequency_std : npt.NDArray[np.float64]
        Standard deviation of each averaged mode's natural frequency (Hz) across the
        per-azimuth perturbations of the averaged model, one entry per mode of
        :func:`vane.modal.eigensolver.modes_from_mbc` of the same result, in its
        order.
    damping_ratio_std : npt.NDArray[np.float64]
        Standard deviation of each averaged mode's damping ratio across azimuths.
    n_azimuths : int
        Number of azimuth samples used.
    """

    natural_frequency_std: npt.NDArray[np.float64]
    damping_ratio_std: npt.NDArray[np.float64]
    n_azimuths: int


def azimuth_spread(result: MBCResult) -> AzimuthSpread:
    """Compute the per-mode azimuth spread of a retained MBC result.

    Rather than comparing per-azimuth spectra directly, this measures how each
    frozen-time matrix ``A(psi)`` perturbs the *averaged* model: for an isotropic rotor
    every ``A(psi) == avg_a`` and the spread is zero, while anisotropy makes the
    per-azimuth matrices (and their spectra) differ by an amount the spread captures.
    For each averaged mode the first-order eigenvalue perturbation
    ``dlambda = y^H (A(psi) - avg_a) x / (y^H x)`` (left eigenvector ``y``, right
    ``x``) is evaluated at every azimuth, and the standard deviation of the
    resulting natural frequency and damping is returned. It is zero for an isotropic
    rotor (``A(psi) == avg_a``) and grows with anisotropy.

    Parameters
    ----------
    result : MBCResult
        An MBC result computed with ``retain_per_azimuth=True``.

    Returns
    -------
    AzimuthSpread
        The per-mode frequency and damping spread.

    Raises
    ------
    ValueError
        If the per-azimuth matrices were not retained or ``avg_a`` is absent.
    """
    if result.per_azimuth_a is None:
        msg = (
            "per-azimuth matrices not retained; "
            "call mbc3_transform(retain_per_azimuth=True)"
        )
        raise ValueError(msg)
    if result.avg_a is None:
        msg = "MBCResult has no averaged A matrix"
        raise ValueError(msg)

    averaged = result.avg_a
    # The modes of the averaged model exactly as the eigensolver returns them (same
    # selection -- split rigid-body pairs dropped -- and order), so the spread is
    # aligned with the ModalSolution of this result mode for mode.
    solution = modes_from_mbc(result)
    lam = np.asarray(solution.eigenvalues, dtype=np.complex128)
    vl = np.asarray(solution.full_left_eigenvectors, dtype=np.complex128)
    vr = np.asarray(solution.full_eigenvectors, dtype=np.complex128)
    sensitivity = np.sum(np.conj(vl) * vr, axis=0)

    n_az = len(result.per_azimuth_a)
    n_modes = lam.shape[0]
    frequencies = np.empty((n_az, n_modes), dtype=np.float64)
    dampings = np.empty((n_az, n_modes), dtype=np.float64)
    for k, matrix in enumerate(result.per_azimuth_a):
        delta = matrix - averaged
        projection = np.einsum("ik,ij,jk->k", np.conj(vl), delta, vr)
        # A defective averaged eigenvalue has sensitivity ~0; first-order
        # perturbation theory breaks down there, so let it become nan cleanly.
        with np.errstate(divide="ignore", invalid="ignore"):
            perturbed = lam + projection / sensitivity
            magnitude = np.abs(perturbed)
            frequencies[k] = magnitude / _TWO_PI
            dampings[k] = np.where(magnitude > 0.0, -perturbed.real / magnitude, 0.0)

    empty = np.empty(0, dtype=np.float64)
    return AzimuthSpread(
        natural_frequency_std=frequencies.std(axis=0) if n_modes else empty,
        damping_ratio_std=dampings.std(axis=0) if n_modes else empty,
        n_azimuths=n_az,
    )


def unified_mode_confidence(
    solution: ModalSolution,
    *,
    frequency_spread: npt.NDArray[np.float64] | None = None,
    track_confidence: npt.NDArray[np.float64] | None = None,
    spread_scale: float = 0.05,
) -> npt.NDArray[np.float64]:
    """Combine the uncertainty sources into one per-mode confidence in ``[0, 1]``.

    The confidence is the product of independent reliability factors: numerical
    conditioning (``1 / kappa``), a degeneracy penalty, an azimuth-stability factor
    ``exp(-relative_spread / spread_scale)``, and an optional tracking confidence.

    Parameters
    ----------
    solution : ModalSolution
        The modal solution (provides condition numbers and degeneracy flags).
    frequency_spread : npt.NDArray[np.float64] or None, optional
        Per-mode natural-frequency standard deviation (e.g. from
        :func:`azimuth_spread`); ``nan`` entries are treated as zero spread.
    track_confidence : npt.NDArray[np.float64] or None, optional
        Per-mode tracking confidence in ``[0, 1]`` from the cross-operating-point
        tracker.
    spread_scale : float, optional
        Relative-spread scale at which the azimuth-stability factor falls to ``1/e``.

    Returns
    -------
    npt.NDArray[np.float64]
        The per-mode confidence.
    """
    n_modes = solution.n_modes
    if n_modes == 0:
        return np.empty(0, dtype=np.float64)

    if solution.condition_numbers.shape[0] == n_modes:
        with np.errstate(divide="ignore"):
            confidence = np.where(
                solution.condition_numbers > 0.0,
                1.0 / solution.condition_numbers,
                0.0,
            )
    else:
        confidence = np.ones(n_modes, dtype=np.float64)

    if solution.is_degenerate.shape[0] == n_modes:
        confidence = confidence * np.where(solution.is_degenerate, 0.5, 1.0)

    if frequency_spread is not None:
        relative = np.nan_to_num(
            frequency_spread / (np.abs(solution.natural_frequencies_hz) + 1e-12),
            nan=0.0,
        )
        confidence = confidence * np.exp(-relative / spread_scale)

    if track_confidence is not None:
        confidence = confidence * np.clip(track_confidence, 0.0, 1.0)

    return np.clip(confidence, 0.0, 1.0).astype(np.float64)
