# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Ground-truth benchmarking of modal extraction accuracy and robustness.

A synthetic system is built from *prescribed* natural frequencies, damping ratios,
and mode shapes, so the exact modal answer is known. Running the extraction on it
and scoring the result against that ground truth measures accuracy directly; adding
controlled noise to the system matrix and re-scoring measures robustness. This is
the quantitative evidence that the extraction is accurate and degrades gracefully,
which the reference tools do not provide.

A classically damped system ``x'' + C x' + K x = 0`` with ``M = I`` is assembled
from a modal matrix ``Phi`` and modal parameters via ``K = Phi diag(wn^2) Phi^-1``
and ``C = Phi diag(2 zeta wn) Phi^-1``; its state matrix ``[[0, I], [-K, -C]]`` then
has exactly the prescribed eigenvalues and (displacement) mode shapes ``Phi``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

from vane.modal.eigensolver import compute_modes
from vane.modal.mac import compute_mac

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from vane.modal.eigensolver import ModalSolution

__all__ = [
    "GroundTruthSystem",
    "ModeRecoveryScore",
    "robustness_curve",
    "score_recovery",
    "synthetic_system",
]

_TWO_PI = 2.0 * np.pi


@dataclass
class GroundTruthSystem:
    """A synthetic system with known modal parameters.

    Parameters
    ----------
    a : npt.NDArray[np.float64]
        State matrix of shape ``(2 n, 2 n)`` in ``[displacement, velocity]`` order.
    natural_frequencies_hz : npt.NDArray[np.float64]
        The prescribed natural frequencies (Hz), shape ``(n,)``.
    damping_ratios : npt.NDArray[np.float64]
        The prescribed damping ratios, shape ``(n,)``.
    mode_shapes : npt.NDArray[np.float64]
        The prescribed displacement mode shapes as columns, shape ``(n, n)``.
    ndof2 : int
        The number of second-order degrees of freedom ``n``.
    """

    a: npt.NDArray[np.float64]
    natural_frequencies_hz: npt.NDArray[np.float64]
    damping_ratios: npt.NDArray[np.float64]
    mode_shapes: npt.NDArray[np.float64]
    ndof2: int


@dataclass
class ModeRecoveryScore:
    """Accuracy of a recovered modal solution against a ground truth.

    Parameters
    ----------
    max_frequency_error : float
        Largest relative natural-frequency error over the matched modes.
    max_damping_error : float
        Largest absolute damping-ratio error over the matched modes.
    min_mac : float
        Smallest mode-shape MAC over the matched modes (1 = exact); ``0`` when any
        ground-truth mode was not recovered.
    mean_mac : float
        Mean mode-shape MAC over all ground-truth modes (unrecovered modes count
        as ``0``).
    n_matched : int
        Number of ground-truth modes matched to a recovered mode (less than the
        prescribed count when modes are missing).
    """

    max_frequency_error: float
    max_damping_error: float
    min_mac: float
    mean_mac: float
    n_matched: int


def synthetic_system(
    natural_frequencies_hz: Sequence[float],
    damping_ratios: Sequence[float],
    *,
    mode_shapes: npt.NDArray[np.float64] | None = None,
) -> GroundTruthSystem:
    """Build a classically damped system with the prescribed modal parameters.

    Parameters
    ----------
    natural_frequencies_hz : Sequence[float]
        Natural frequencies (Hz), all strictly positive.
    damping_ratios : Sequence[float]
        Damping ratios in ``[0, 1)``, parallel to ``natural_frequencies_hz``.
    mode_shapes : npt.NDArray[np.float64] or None, optional
        Invertible modal matrix ``Phi`` (columns are mode shapes), shape ``(n, n)``;
        the identity is used when omitted.

    Returns
    -------
    GroundTruthSystem
        The state matrix and the exact modal parameters it realizes.

    Raises
    ------
    ValueError
        If the inputs have mismatched lengths, non-positive frequencies, damping
        outside ``[0, 1)``, or a non-square / singular ``mode_shapes``.
    """
    frequencies = np.asarray(natural_frequencies_hz, dtype=np.float64)
    dampings = np.asarray(damping_ratios, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.shape != dampings.shape:
        msg = "natural_frequencies_hz and damping_ratios must be 1-D and equal length"
        raise ValueError(msg)
    if not (np.all(np.isfinite(frequencies)) and np.all(np.isfinite(dampings))):
        msg = "natural_frequencies_hz and damping_ratios must be finite"
        raise ValueError(msg)
    if frequencies.size == 0 or np.any(frequencies <= 0.0):
        msg = "natural_frequencies_hz must be non-empty and strictly positive"
        raise ValueError(msg)
    if np.any(dampings < 0.0) or np.any(dampings >= 1.0):
        msg = "damping_ratios must lie in [0, 1)"
        raise ValueError(msg)

    n = frequencies.shape[0]
    phi = (
        np.eye(n) if mode_shapes is None else np.asarray(mode_shapes, dtype=np.float64)
    )
    if phi.shape != (n, n):
        msg = f"mode_shapes must be square ({n}, {n}), got {phi.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(phi)):
        msg = "mode_shapes must be finite"
        raise ValueError(msg)
    try:
        inv_phi = np.linalg.inv(phi)
    except np.linalg.LinAlgError as exc:
        msg = "mode_shapes must be invertible"
        raise ValueError(msg) from exc

    omega = _TWO_PI * frequencies
    stiffness = phi @ np.diag(omega**2) @ inv_phi
    damping = phi @ np.diag(2.0 * dampings * omega) @ inv_phi
    a = np.block(
        [
            [np.zeros((n, n)), np.eye(n)],
            [-stiffness, -damping],
        ]
    )
    return GroundTruthSystem(
        a=a,
        natural_frequencies_hz=frequencies,
        damping_ratios=dampings,
        mode_shapes=phi,
        ndof2=n,
    )


def score_recovery(
    solution: ModalSolution, truth: GroundTruthSystem
) -> ModeRecoveryScore:
    """Score a recovered solution against the ground truth.

    Recovered modes are matched to ground-truth modes by maximum total MAC
    (Hungarian assignment), and the worst-case frequency and damping errors and the
    mode-shape correlations are reported over the matched pairs.

    Parameters
    ----------
    solution : ModalSolution
        The extracted modal solution.
    truth : GroundTruthSystem
        The ground-truth system.

    Returns
    -------
    ModeRecoveryScore
        The accuracy metrics.

    Raises
    ------
    ValueError
        If the solution has no modes to score.
    """
    if solution.n_modes == 0:
        msg = "solution has no modes to score"
        raise ValueError(msg)

    n_true = int(truth.natural_frequencies_hz.shape[0])
    truth_shapes = truth.mode_shapes.astype(np.complex128)
    mac = compute_mac(truth_shapes, solution.mode_shapes)
    truth_rows, solution_cols = linear_sum_assignment(mac, maximize=True)

    frequency_errors: list[float] = []
    damping_errors: list[float] = []
    macs: list[float] = []
    for truth_index, solution_index in zip(truth_rows, solution_cols, strict=True):
        true_frequency = float(truth.natural_frequencies_hz[truth_index])
        frequency_errors.append(
            abs(float(solution.natural_frequencies_hz[solution_index]) - true_frequency)
            / true_frequency
        )
        damping_errors.append(
            abs(
                float(solution.damping_ratios[solution_index])
                - float(truth.damping_ratios[truth_index])
            )
        )
        macs.append(float(mac[truth_index, solution_index]))

    # Ground-truth modes left unmatched (fewer modes recovered than prescribed) are
    # unrecovered: they drive the worst-case MAC to zero and dilute the mean.
    n_matched = len(macs)
    min_mac = 0.0 if n_matched < n_true else min(macs)
    return ModeRecoveryScore(
        max_frequency_error=max(frequency_errors),
        max_damping_error=max(damping_errors),
        min_mac=min_mac,
        mean_mac=float(np.sum(macs) / n_true),
        n_matched=n_matched,
    )


def robustness_curve(
    truth: GroundTruthSystem,
    noise_levels: Sequence[float],
    *,
    seed: int = 0,
) -> list[tuple[float, ModeRecoveryScore]]:
    """Score recovery as increasing noise is added to the system dynamics.

    One fixed seeded perturbation direction is drawn and, at each level ``sigma``,
    scaled by ``sigma * ||dynamics||`` and added to the stiffness/damping block of
    the state matrix (the kinematic ``[0, I]`` block is left exact). Reusing a single
    direction makes the sweep a pure amplitude sweep, so the error grows
    monotonically with ``sigma`` (to first order) rather than varying with an
    arbitrary fresh direction at each level.

    Parameters
    ----------
    truth : GroundTruthSystem
        The ground-truth system.
    noise_levels : Sequence[float]
        Relative noise amplitudes, each ``>= 0``.
    seed : int, optional
        Seed for the deterministic perturbation direction.

    Returns
    -------
    list[tuple[float, ModeRecoveryScore]]
        Each noise level paired with its recovery score.

    Raises
    ------
    ValueError
        If any noise level is negative or non-finite.
    """
    levels = np.asarray(noise_levels, dtype=np.float64)
    if levels.size and (not np.all(np.isfinite(levels)) or np.any(levels < 0.0)):
        msg = "noise_levels must be finite and non-negative"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    n = truth.ndof2
    scale = float(np.linalg.norm(truth.a[n:, :]))
    # Unit-normalize the direction so the perturbation Frobenius norm is exactly
    # sigma * ||dynamics||, i.e. sigma is a true relative amplitude independent of
    # system size (a raw standard-normal block has norm ~sqrt(2 n^2)).
    direction = rng.standard_normal((n, 2 * n))
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm > 0.0:
        direction = direction / direction_norm
    curve: list[tuple[float, ModeRecoveryScore]] = []
    for sigma in levels:
        noisy = truth.a.copy()
        noisy[n:, :] += direction * (float(sigma) * scale)
        solution = compute_modes(noisy, ndof2=n, ndof1=0)
        curve.append((float(sigma), score_recovery(solution, truth)))
    return curve
