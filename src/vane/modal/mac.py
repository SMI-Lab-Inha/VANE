# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Modal Assurance Criterion and mode-to-mode matching.

The Modal Assurance Criterion (MAC) measures the correlation between two mode
shapes; a value of 1 indicates perfect correlation and 0 indicates orthogonality
[1]_. For complex (state-space) modes the pole-weighted MACXP criterion [2]_ is
also provided. ``match_modes`` performs a one-to-one Hungarian assignment over a MAC
(optionally frequency-penalized) matrix between two operating points; this is a
pairwise building block. Cross-operating-point *tracking* over a whole sweep is done
separately by :func:`vane.modal.identify_modes`, which extracts globally optimal
continuity paths rather than chaining pairwise matches.

References
----------
.. [1] Allemang, R. J. (2003). The modal assurance criterion — twenty years of
   use and abuse. *Sound and Vibration*, 37(8), 14-23.
.. [2] Vacher, P., Jacquier, B., & Bucharles, A. (2010). Extensions of the MAC
   criterion to complex modes. *Proceedings of ISMA 2010*, 2713-2726.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = ["compute_mac", "compute_macxp", "match_modes"]


def compute_mac(
    phi_ref: npt.NDArray[np.complex128],
    phi_test: npt.NDArray[np.complex128],
) -> npt.NDArray[np.float64]:
    r"""Compute the Modal Assurance Criterion (MAC) matrix.

    Parameters
    ----------
    phi_ref : npt.NDArray[np.complex128]
        Reference mode-shape matrix, shape ``(n_dof, n_modes_ref)``.
    phi_test : npt.NDArray[np.complex128]
        Test mode-shape matrix, shape ``(n_dof, n_modes_test)``.

    Returns
    -------
    npt.NDArray[np.float64]
        MAC matrix of shape ``(n_modes_ref, n_modes_test)`` with values in
        ``[0, 1]``.

    Raises
    ------
    ValueError
        If the two mode-shape matrices have incompatible first dimensions.

    Notes
    -----
    The MAC between modes :math:`i` and :math:`j` is

    .. math::

        \text{MAC}_{ij} = \frac{|\phi_i^H \phi_j|^2}
        {(\phi_i^H \phi_i)(\phi_j^H \phi_j)}.
    """
    if phi_ref.shape[0] != phi_test.shape[0]:
        msg = (
            f"Mode shapes have incompatible DOF dimensions: "
            f"{phi_ref.shape[0]} vs {phi_test.shape[0]}"
        )
        raise ValueError(msg)
    cross = np.abs(phi_ref.conj().T @ phi_test) ** 2
    ref_norm = np.real(np.einsum("ij,ij->j", phi_ref.conj(), phi_ref))
    test_norm = np.real(np.einsum("ij,ij->j", phi_test.conj(), phi_test))
    denom = np.outer(ref_norm, test_norm)
    with np.errstate(divide="ignore", invalid="ignore"):
        mac = np.where(denom > 0.0, cross / denom, 0.0)
    # Clip to [0, 1] to absorb floating-point overshoot on near-collinear modes.
    return np.asarray(np.clip(mac, 0.0, 1.0), dtype=np.float64)


def compute_macxp(
    phi_ref: npt.NDArray[np.complex128],
    lambda_ref: npt.NDArray[np.complex128],
    phi_test: npt.NDArray[np.complex128],
    lambda_test: npt.NDArray[np.complex128],
) -> npt.NDArray[np.float64]:
    r"""Compute the pole-weighted MACXP matrix for complex modes.

    MACXP augments the MAC with the transpose product and weights both by the
    modal poles, making it more discriminating than MAC for complex state-space modes.

    Parameters
    ----------
    phi_ref, phi_test : npt.NDArray[np.complex128]
        Reference and test mode-shape matrices, shape ``(n_dof, n_modes)``.
    lambda_ref, lambda_test : npt.NDArray[np.complex128]
        Corresponding modal poles (eigenvalues), shape ``(n_modes,)``.

    Returns
    -------
    npt.NDArray[np.float64]
        MACXP matrix of shape ``(n_modes_ref, n_modes_test)``; a mode correlated
        with itself yields 1.

    Raises
    ------
    ValueError
        If shapes are inconsistent between mode matrices and pole vectors.

    Notes
    -----
    The criterion is defined for damped modes (poles with a non-zero real part);
    an undamped pole makes its self-correlation term infinite, so the **whole row**
    (reference pole) or **column** (test pole) of every undamped pole is returned as
    NaN rather than raising -- including the cross entries against damped modes,
    which would otherwise collapse to a misleading ``0.0``.
    """
    if phi_ref.shape[0] != phi_test.shape[0]:
        msg = (
            f"Mode shapes have incompatible DOF dimensions: "
            f"{phi_ref.shape[0]} vs {phi_test.shape[0]}"
        )
        raise ValueError(msg)
    if (
        phi_ref.shape[1] != lambda_ref.shape[0]
        or phi_test.shape[1] != lambda_test.shape[0]
    ):
        msg = "Number of poles must match the number of mode shapes"
        raise ValueError(msg)

    hermitian = np.abs(phi_ref.conj().T @ phi_test)
    transpose = np.abs(phi_ref.T @ phi_test)
    l_ref = lambda_ref[:, np.newaxis]
    l_test = lambda_test[np.newaxis, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = hermitian / np.abs(l_ref.conj() + l_test) + transpose / np.abs(
            l_ref + l_test
        )
        denom = np.outer(
            _macxp_self(phi_ref, lambda_ref), _macxp_self(phi_test, lambda_test)
        )
        result = numerator**2 / denom
    result[lambda_ref.real == 0.0, :] = np.nan
    result[:, lambda_test.real == 0.0] = np.nan
    return np.asarray(result, dtype=np.float64)


def match_modes(
    mac: npt.NDArray[np.float64],
    ref_frequencies: npt.NDArray[np.float64] | None = None,
    test_frequencies: npt.NDArray[np.float64] | None = None,
    frequency_weight: float = 0.0,
    *,
    forbidden: npt.NDArray[np.bool_] | None = None,
) -> list[tuple[int, int]]:
    """Match reference modes to test modes by maximum-weight assignment.

    A Hungarian (optimal) assignment maximizes the total correlation between the
    two mode sets. When frequencies are supplied, the MAC of each pair is scaled
    by ``1 - frequency_weight * |df| / span`` so that frequency proximity breaks
    ties between similar shapes (the strategy used for Campbell-diagram tracking).

    Parameters
    ----------
    mac : npt.NDArray[np.float64]
        Correlation matrix, shape ``(n_ref, n_test)`` (e.g. from
        :func:`compute_mac`).
    ref_frequencies, test_frequencies : npt.NDArray[np.float64] or None, optional
        Per-mode natural frequencies for the reference and test sets. Both must be
        given to apply the frequency penalty.
    frequency_weight : float, optional
        Strength of the frequency penalty in ``[0, 1]``; ``0`` (default) uses MAC
        alone.
    forbidden : npt.NDArray[np.bool_] or None, optional
        Pairs that must not be matched, shape ``(n_ref, n_test)``. This is a hard
        constraint: whenever an assignment avoiding every forbidden pair exists, it
        is chosen.

    Returns
    -------
    list[tuple[int, int]]
        ``(ref_index, test_index)`` pairs, one per matched mode (``min(n_ref,
        n_test)`` pairs), ordered by reference index.

    Raises
    ------
    ValueError
        If ``forbidden`` does not match the shape of ``mac``, or no assignment avoids
        every forbidden pair.
    """
    weight = np.array(mac, dtype=np.float64, copy=True)
    if (
        ref_frequencies is not None
        and test_frequencies is not None
        and frequency_weight > 0.0
    ):
        both = np.concatenate([ref_frequencies, test_frequencies])
        span = float(both.max() - both.min())
        if span > 0.0:
            delta = np.abs(
                ref_frequencies[:, np.newaxis] - test_frequencies[np.newaxis, :]
            )
            penalty = 1.0 - frequency_weight * delta / span
            weight = weight * np.clip(penalty, 0.0, 1.0)

    blocked: npt.NDArray[np.bool_] | None = None
    if forbidden is not None:
        blocked = np.asarray(forbidden, dtype=np.bool_)
        if blocked.shape != weight.shape:
            msg = f"forbidden must have shape {weight.shape}, got {blocked.shape}"
            raise ValueError(msg)
        # Any allowed pair scores at most max|weight|, so a forbidden pair costing more
        # than every allowed pair together can never be traded for a better total.
        scale = float(np.max(np.abs(weight), initial=0.0)) + 1.0
        weight = np.where(blocked, -scale * (min(weight.shape) + 1), weight)

    row_indices, col_indices = linear_sum_assignment(weight, maximize=True)
    pairs = sorted(zip(row_indices.tolist(), col_indices.tolist(), strict=True))
    if blocked is not None and any(blocked[r, c] for r, c in pairs):
        msg = "no mode assignment avoids every forbidden pair"
        raise ValueError(msg)
    return [(int(r), int(c)) for r, c in pairs]


def _macxp_self(
    phi: npt.NDArray[np.complex128], lam: npt.NDArray[np.complex128]
) -> npt.NDArray[np.float64]:
    """Return the MACXP self-correlation denominator term for each mode."""
    hermitian = np.real(np.einsum("ij,ij->j", phi.conj(), phi))
    transpose = np.abs(np.einsum("ij,ij->j", phi, phi))
    term = hermitian / (2.0 * np.abs(lam.real)) + transpose / (2.0 * np.abs(lam))
    return np.asarray(term, dtype=np.float64)
