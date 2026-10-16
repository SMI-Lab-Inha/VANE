# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Shared, unit-invariant validity checks for covariance matrices.

Every covariance entry point in :mod:`vane.sysid` (the Kalman-model builder, the
steady-state estimator assessment, process-noise discretization, disturbance
augmentation and the observability tools) decides whether a matrix is a valid
covariance through the predicates in this module, so the same ``Q``/``R``/``P0`` is
accepted or rejected consistently everywhere.

A covariance's channels routinely carry different physical units (a mooring tension in
newtons next to a platform angle in radians or a strain), so its variances can differ by
tens of orders of magnitude while the matrix is perfectly well posed. Validity must
therefore not depend on the unit of any channel: every test is invariant under
per-channel (diagonal, Jacobi) rescaling ``M -> S M S`` with ``S`` positive diagonal,
which in particular includes a global rescaling. Definiteness is judged on the
**correlation matrix** ``K = D^-1/2 M D^-1/2`` (``D = diag(M)``), whose unit diagonal
removes the units and leaves only the correlation structure. The tolerances are:

``SYMMETRY_RTOL = 1e-9``
    ``|M_ij - M_ji| <= SYMMETRY_RTOL * max(sqrt(|M_ii| |M_jj|), |M_ij|, |M_ji|)``.
    Every term of the pair scale rescales exactly like ``M_ij`` under ``S M S``, so the
    test is unit-invariant, including for a pair with a zero variance (where the pair
    is compared relative to its own entries). No global floor is used: one would tie a
    pair's tolerance to the units of unrelated channels. ``1e-9`` is loose enough
    for matrices assembled in floating point and tight enough to catch a transposition
    or data-entry error.
``PD_RTOL = 1e-12``
    Positive definite means every variance ``M_ii > 0``, the Cholesky factorization of
    ``K`` succeeds, and ``lambda_min(K) > PD_RTOL``. ``K`` has unit diagonal, so
    ``1 <= lambda_max(K) <= n`` and the bound caps the *unit-free* condition number of
    ``K`` near ``1e12``; by van der Sluis' theorem this Jacobi scaling is within
    a factor ``n`` of the best possible diagonal scaling, so it measures genuine
    near-dependence between channels, not unit disparity. At that limit a solve against
    ``K`` still keeps about four significant digits (``eps * 1e12 ~ 2e-4``); below it
    the correlation structure is numerically singular (e.g. two perfectly correlated
    channels) and a Kalman gain computed from it is unreliable. ``diag(1e8, 1e-6)`` and
    ``diag(1e-12)`` both have ``K = I`` and pass.
``PSD_RTOL = 1e-9``
    Positive semidefinite means: no variance is below ``-PSD_RTOL * max|M_ii|`` (a
    larger negative variance is invalid; a smaller one is floating-point residue of a
    structurally zero variance); every channel whose variance is not strictly positive
    has off-diagonal entries within the Cauchy-Schwarz bound
    ``|M_ij| <= sqrt(PSD_RTOL * |M_ii| * |M_jj|)``, which uses only the pair's own
    variances (so an exactly zero variance requires an exactly zero row); and
    the correlation matrix of the strictly positive channels satisfies
    ``lambda_min(K) >= -PSD_RTOL``. The margin tolerates round-off in computed matrices
    (a discretized ``Q_d``, a rank-deficient density) while rejecting any materially
    negative variance or correlation beyond one. The zero matrix is valid. Only the
    allowance for a tiny negative variance refers to the largest variance (a single
    channel has no scale of its own); every other part of the outcome is exactly
    invariant under per-channel rescaling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

SYMMETRY_RTOL = 1.0e-9
PSD_RTOL = 1.0e-9
PD_RTOL = 1.0e-12


def is_symmetric(matrix: npt.NDArray[np.float64]) -> bool:
    """Return whether a finite square matrix is symmetric, per channel pair.

    Parameters
    ----------
    matrix : npt.NDArray[np.float64]
        A finite, square matrix.

    Returns
    -------
    bool
        ``True`` when every ``|M_ij - M_ji|`` is within ``SYMMETRY_RTOL`` of the pair
        scale ``max(sqrt(|M_ii| |M_jj|), |M_ij|, |M_ji|)`` (the zero and empty matrices
        are symmetric).
    """
    if matrix.size == 0:
        return True
    root = np.sqrt(np.abs(np.diag(matrix)))
    magnitude = np.abs(matrix)
    pair_scale = np.maximum(np.outer(root, root), np.maximum(magnitude, magnitude.T))
    # An overflowing difference is infinite and fails the test, as it should.
    with np.errstate(over="ignore", invalid="ignore"):
        asymmetry = np.abs(matrix - matrix.T)
    return bool(np.all(asymmetry <= SYMMETRY_RTOL * pair_scale))


def symmetric_part(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return ``(M + M^T) / 2`` without overflow or underflow of exact entries.

    Callers use this matrix, not the raw input, after validation, so what they factor
    or solve with is exactly what the definiteness test judged.

    Entries that already equal their transpose (including the whole diagonal) are kept
    exactly, so a subnormal variance is not halved to zero; only the remaining pairs are
    averaged, with each term halved first so values near the float64 limit cannot
    overflow.
    """
    with np.errstate(under="ignore"):
        averaged = 0.5 * matrix + 0.5 * matrix.T
    return np.asarray(np.where(matrix == matrix.T, matrix, averaged), dtype=np.float64)


def _correlation(
    matrix: npt.NDArray[np.float64], variances: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Symmetric correlation matrix ``D^-1/2 M D^-1/2`` for strictly positive ``D``.

    An entry that overflows is left infinite (a correlation that large is far beyond
    one); callers treat any non-finite correlation as invalid.
    """
    inv_root = 1.0 / np.sqrt(variances)
    with np.errstate(over="ignore", invalid="ignore"):
        corr = inv_root[:, None] * symmetric_part(matrix) * inv_root[None, :]
    return np.asarray(corr, dtype=np.float64)


def is_positive_semidefinite(matrix: npt.NDArray[np.float64]) -> bool:
    """Return whether a symmetric matrix is positive semidefinite (unit-invariant).

    Parameters
    ----------
    matrix : npt.NDArray[np.float64]
        A square matrix, assumed already checked symmetric.

    Returns
    -------
    bool
        ``True`` when every entry is finite, no variance is materially negative, every
        non-positive-variance channel has a (numerically) zero row, and the correlation
        matrix of the positive-variance channels has ``lambda_min >= -PSD_RTOL``. The
        zero and empty matrices are positive semidefinite.
    """
    if not np.all(np.isfinite(matrix)):
        return False
    if matrix.size == 0:
        return True
    sym = symmetric_part(matrix)
    variances = np.diag(sym).copy()
    d_max = float(np.max(np.abs(variances)))
    if np.any(variances < -PSD_RTOL * d_max):
        return False
    positive = variances > 0.0
    zero = ~positive
    if np.any(zero):
        # Cauchy-Schwarz: |M_ij| <= sqrt(M_ii M_jj), with the non-positive variance's
        # own magnitude as its round-off scale. The bound uses only the pair's own
        # variances, so it rescales with the pair's units, and an exactly zero variance
        # requires an exactly zero row.
        # Product of square roots, so the bound stays finite for large variances.
        roots = np.sqrt(np.abs(variances))
        bound = np.sqrt(PSD_RTOL) * np.outer(roots[zero], roots)
        rows = np.abs(sym[zero, :])
        rows[np.arange(rows.shape[0]), np.flatnonzero(zero)] = 0.0
        if np.any(rows > bound):
            return False
    if not np.any(positive):
        return True
    sub = sym[np.ix_(positive, positive)]
    corr = _correlation(sub, variances[positive])
    if not np.all(np.isfinite(corr)):
        return False
    eigvals = np.linalg.eigvalsh(corr)
    return bool(eigvals[0] >= -PSD_RTOL)


def is_positive_definite(matrix: npt.NDArray[np.float64]) -> bool:
    """Return whether a symmetric matrix is numerically positive definite.

    Parameters
    ----------
    matrix : npt.NDArray[np.float64]
        A square matrix, assumed already checked symmetric.

    Returns
    -------
    bool
        ``True`` when every entry is finite, every variance is strictly positive, and
        the correlation matrix ``K`` has a Cholesky factor and
        ``lambda_min(K) > PD_RTOL``. The empty matrix is positive definite; the zero
        matrix is not.
    """
    if not np.all(np.isfinite(matrix)):
        return False
    if matrix.size == 0:
        return True
    variances = np.diag(matrix).copy()
    if not np.all(variances > 0.0):
        return False
    corr = _correlation(matrix, variances)
    if not np.all(np.isfinite(corr)):
        return False
    try:
        np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        return False
    return bool(np.linalg.eigvalsh(corr)[0] > PD_RTOL)


def validate_covariance(
    matrix: npt.NDArray[np.float64], name: str, *, definite: bool
) -> None:
    """Raise unless ``matrix`` is a finite, symmetric PSD (or PD) covariance.

    Parameters
    ----------
    matrix : npt.NDArray[np.float64]
        Candidate covariance; its shape is checked by the caller.
    name : str
        Name used in the error message.
    definite : bool
        Require positive definiteness instead of semidefiniteness.

    Raises
    ------
    ValueError
        If ``matrix`` is not finite, not symmetric, or not positive (semi)definite.
    """
    if not np.all(np.isfinite(matrix)):
        msg = f"{name} must be finite (no NaN or inf) to be a covariance"
        raise ValueError(msg)
    if not is_symmetric(matrix):
        msg = f"{name} must be symmetric to be a covariance"
        raise ValueError(msg)
    check = is_positive_definite if definite else is_positive_semidefinite
    valid = check(matrix)
    if not valid:
        adjective = "positive definite" if definite else "positive semidefinite"
        msg = f"{name} must be {adjective} to be a covariance"
        raise ValueError(msg)
