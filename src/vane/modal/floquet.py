# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Floquet / Hill modal analysis of a linear time-periodic (LTP) system.

A spinning rotor makes the linearized dynamics :math:`\dot{\mathbf x} = \mathbf A(\psi)
\mathbf x` **periodic** in the azimuth :math:`\psi = \Omega t` with period
:math:`T = 2\pi/\Omega`, not time-invariant. The multi-blade (Coleman) transform removes
that periodicity only for an *isotropic* rotor; for an **anisotropic** system — a
two-bladed rotor, or a floating turbine whose mooring/hydrodynamics break the rotational
symmetry — the azimuth-averaged model is an approximation whose error this module
quantifies and replaces.

Floquet theory writes each solution as :math:`\mathbf x(t) = e^{\lambda t}\mathbf p(t)`
with :math:`\mathbf p` periodic; the **Floquet exponents** :math:`\lambda` are the true
modal eigenvalues of the periodic system and :math:`\mathbf p(t) = \sum_m \mathbf p_m
e^{i m \Omega t}` carries the **harmonic mode-shape components**. Two independent routes
are provided:

* **Hill's method** (:func:`floquet_modes`): expand :math:`\mathbf A(\psi) = \sum_n
  \mathbf A_n e^{i n\psi}` and solve the eigenproblem of the Hill matrix with blocks
  :math:`\mathbf H_{kj} = \mathbf A_{k-j} - i k\Omega\,\delta_{kj}\mathbf I`; the
  principal exponents are the eigenvalues whose eigenvectors concentrate in the
  fundamental (:math:`m=0`) harmonic, and the blocks give :math:`\mathbf p_m`.
* **The monodromy matrix** (:func:`monodromy_exponents`): integrate the state-transition
  matrix over one period (piecewise-constant / matrix-exponential product); its
  eigenvalues are the **Floquet multipliers** :math:`\mu`, and
  :math:`\lambda = \ln\mu / T`. An independent cross-check of Hill.

For an isotropic rotor (every :math:`\mathbf A(\psi)` equal) both reduce to the ordinary
eigenvalues of that constant matrix, so the Floquet result degrades gracefully to the
averaged-model answer; the gap between them is the anisotropy error
(:func:`averaging_error`).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = [
    "HILL_UNIFORM_RTOL",
    "FloquetSolution",
    "averaging_error",
    "floquet_modes",
    "monodromy_exponents",
]

_TWO_PI = 2.0 * np.pi
# Two Hill eigenvalues closer than this (relative to the spectrum scale) are one
# degenerate eigenvalue whose eigenspace basis is rotated by ``_purify_degenerate``.
_DEGENERATE_RTOL = 1e-7
# A candidate is a shifted copy of an accepted exponent only if they agree modulo
# i*Omega to this relative tolerance (loose: truncation perturbs the outer copies) ...
_FAMILY_GAP_RTOL = 1e-3
# ... and its eigenvector is the accepted one shifted by k harmonic blocks.
_FAMILY_OVERLAP = 0.5
# Hill-vs-monodromy disagreement (modulo i*Omega, as a fraction of Omega) that warns.
_MONODROMY_ATOL = 1e-2
#: Relative tolerance on the azimuth intervals for Hill's method. The Fourier
#: coefficients of A(psi) come from the equal-weight discrete transform, which is exact
#: only on a uniform grid, so every interval must equal the mean interval to this
#: fraction of it (plus a 1e-9 rad absolute floor).
HILL_UNIFORM_RTOL = 1e-3
_HILL_UNIFORM_ATOL = 1e-9


@dataclass
class FloquetSolution:
    r"""The Floquet modes of a linear time-periodic system.

    Parameters
    ----------
    exponents : npt.NDArray[np.complex128]
        The principal Floquet exponents :math:`\lambda` (one per state), sorted by
        ascending natural frequency. Selected as the Hill eigenvalues that concentrate
        in the fundamental (:math:`m=0`) harmonic, one per distinct Floquet solution
        (harmonic-shifted copies :math:`\lambda + ik\Omega` of a kept exponent are
        skipped), so a near-time-invariant system
        recovers the ordinary eigenvalues **unfolded** (constant :math:`A` gives
        ``eig(A)`` exactly) — unlike :func:`monodromy_exponents`, whose imaginary part
        is known only modulo :math:`\Omega`.
    natural_frequencies_hz : npt.NDArray[np.float64]
        ``|lambda| / 2pi`` for each exponent.
    damping_ratios : npt.NDArray[np.float64]
        ``-Re(lambda) / |lambda|`` for each exponent.
    harmonics : npt.NDArray[np.int_]
        The harmonic indices ``m`` of the mode-shape blocks, from ``-n_harmonics`` to
        ``+n_harmonics``.
    harmonic_mode_shapes : npt.NDArray[np.complex128]
        The periodic mode-shape components ``p_m``, shape
        ``(n_harmonics_total, n_dof, n_modes)``; ``[harmonics == 0]`` is the
        fundamental (mean) mode shape.
    rotor_speed : float
        The rotor speed :math:`\Omega` (rad/s) used.
    """

    exponents: npt.NDArray[np.complex128]
    natural_frequencies_hz: npt.NDArray[np.float64]
    damping_ratios: npt.NDArray[np.float64]
    harmonics: npt.NDArray[np.int_]
    harmonic_mode_shapes: npt.NDArray[np.complex128]
    rotor_speed: float


def _stack(
    per_azimuth_a: npt.ArrayLike, azimuths: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the azimuth-sorted matrices and azimuths as arrays, validated."""
    matrices = np.asarray(per_azimuth_a, dtype=np.float64)
    angles = np.asarray(azimuths, dtype=np.float64) % _TWO_PI
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        msg = f"per_azimuth_a must be (n_azimuth, n, n), got {matrices.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(matrices)):
        msg = "per_azimuth_a must be finite (no NaN or inf)"
        raise ValueError(msg)
    if angles.shape != (matrices.shape[0],):
        msg = (
            f"azimuths must have one angle per matrix ({matrices.shape[0]}), "
            f"got {angles.shape}"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(angles)):
        msg = "azimuths must be finite"
        raise ValueError(msg)
    order = np.argsort(angles)
    sorted_angles = angles[order]
    # Each azimuth must be distinct once wrapped into [0, 2pi): a duplicate (including 0
    # and 2pi, which wrap to the same angle) makes a zero-width interval, so one
    # supplied matrix would get zero weight in the monodromy product and the Fourier
    # fit -- a quiet scientific error, not just malformed input.
    if sorted_angles.size > 1 and not np.all(np.diff(sorted_angles) > 0.0):
        msg = "azimuths must be distinct after wrapping into [0, 2pi)"
        raise ValueError(msg)
    return matrices[order], sorted_angles


def _fourier_coefficients(
    matrices: npt.NDArray[np.float64], angles: npt.NDArray[np.float64], n_max: int
) -> dict[int, npt.NDArray[np.complex128]]:
    """Return the Fourier coefficients ``A_n`` of ``A(psi)`` for ``|n| <= n_max``."""
    n_azimuth = matrices.shape[0]
    coeffs: dict[int, npt.NDArray[np.complex128]] = {}
    for n in range(-n_max, n_max + 1):
        phase = np.exp(-1j * n * angles)
        coeffs[n] = np.asarray(
            np.tensordot(phase, matrices, axes=(0, 0)) / n_azimuth,
            dtype=np.complex128,
        )
    return coeffs


def _family_gap(
    candidate: complex, accepted: complex, rotor_speed: float
) -> tuple[float, int]:
    """Return the gap between two exponents modulo ``i*Omega`` and the shift ``k``.

    ``candidate`` and ``accepted`` describe the same Floquet multiplier when
    ``candidate = accepted + i*k*Omega`` for an integer ``k``; the returned gap is
    ``|candidate - accepted - i*k*Omega|`` for the nearest such ``k``.
    """
    diff = candidate - accepted
    shift = round(diff.imag / rotor_speed)
    return float(abs(diff - 1j * shift * rotor_speed)), int(shift)


def _shifted_overlap(
    candidate: npt.NDArray[np.complex128],
    accepted: npt.NDArray[np.complex128],
    shift: int,
) -> float:
    """Return how well ``candidate`` is ``accepted``'s eigenvector shifted by ``shift``.

    Both are Hill eigenvectors shaped ``(n_blocks, n_dof)``. If the candidate exponent
    is ``accepted + i*shift*Omega`` *and* it is the same Floquet solution, its harmonic
    block ``m`` equals the accepted block ``m + shift`` (the truncation drops the
    blocks shifted past either edge). Returns the normalized inner product in
    ``[0, 1]``; ~1 for such a copy, ~0 for a different solution that merely shares the
    multiplier (e.g. a repeated eigenvalue, or two modes exactly ``Omega`` apart).
    """
    n_blocks = candidate.shape[0]
    if abs(shift) >= n_blocks:
        return 0.0
    if shift >= 0:
        cand_part = candidate[: n_blocks - shift]
        acc_part = accepted[shift:]
    else:
        cand_part = candidate[-shift:]
        acc_part = accepted[: n_blocks + shift]
    scale = float(np.linalg.norm(candidate) * np.linalg.norm(accepted))
    if scale == 0.0:
        return 0.0
    return float(abs(np.vdot(acc_part, cand_part))) / scale


def _purify_degenerate(
    values: npt.NDArray[np.complex128],
    vectors: npt.NDArray[np.complex128],
    n_dof: int,
    centre: int,
    tol: float,
) -> npt.NDArray[np.complex128]:
    """Rotate each degenerate Hill eigenspace so its basis ranks by m=0 energy.

    A Hill eigenvalue shared by two different Floquet solutions (a repeated eigenvalue,
    or an isotropic mode at exactly ``Omega/2`` whose conjugate's shifted copy lands on
    the same value) has an eigenspace that LAPACK returns in an arbitrary basis, mixing
    the fundamental of one solution with a sideband of the other. Rotating each such
    cluster's basis by the SVD of its fundamental (``m = 0``) block separates them, so
    the fundamental-energy ranking in :func:`_select_principal` sees clean vectors.
    """
    out = vectors.copy()
    assigned = np.zeros(values.size, dtype=bool)
    rows = slice(centre * n_dof, (centre + 1) * n_dof)
    for i in range(values.size):
        if assigned[i]:
            continue
        cluster = np.flatnonzero((np.abs(values - values[i]) <= tol) & ~assigned)
        assigned[cluster] = True
        if cluster.size < 2:
            continue
        basis, singular, _ = np.linalg.svd(vectors[:, cluster], full_matrices=False)
        rank = int(np.sum(singular > singular[0] * 1e-8))
        span = basis[:, :rank]
        _, _, rotation = np.linalg.svd(span[rows, :], full_matrices=True)
        out[:, cluster[:rank]] = span @ rotation.conj().T
    return np.asarray(out, dtype=np.complex128)


def _select_principal(
    values: npt.NDArray[np.complex128],
    shaped: npt.NDArray[np.complex128],
    central: npt.NDArray[np.float64],
    n_dof: int,
    rotor_speed: float,
) -> npt.NDArray[np.int_]:
    """Pick one Hill eigenvalue per distinct Floquet solution, ``n_dof`` in total.

    Every Floquet solution appears in the Hill spectrum as a *family* of copies
    ``lambda + i*k*Omega``, one per harmonic shift ``k``. Candidates are walked in
    descending fundamental (``m = 0``) energy and a candidate is rejected when it is a
    shifted copy of an already-accepted solution: its exponent equals the accepted one
    modulo ``i*Omega`` **and** its eigenvector is the accepted eigenvector shifted by
    the same number of harmonic blocks. The eigenvector test keeps genuinely distinct
    solutions that merely share a multiplier (repeated eigenvalues; an isotropic mode at
    exactly ``Omega/2`` and its conjugate). For a real system the conjugate of an
    exponent with ``Im(lambda) = Omega/2 (mod Omega)`` -- the parametric-resonance
    tongue, real negative multiplier -- *is* such a shifted copy, so only one of the
    pair is kept and the other solution family is not dropped. Ties in fundamental
    energy (a conjugate-symmetric pair) prefer the non-negative imaginary part.
    """
    order = np.lexsort((-values.imag, -np.round(central, 9)))
    accepted: list[int] = []
    for idx in order:
        candidate = complex(values[idx])
        duplicate = False
        for kept in accepted:
            reference = complex(values[kept])
            gap, shift = _family_gap(candidate, reference, rotor_speed)
            scale = max(rotor_speed, abs(reference))
            if gap <= _FAMILY_GAP_RTOL * scale and (
                _shifted_overlap(shaped[:, :, idx], shaped[:, :, kept], shift)
                >= _FAMILY_OVERLAP
            ):
                duplicate = True
                break
        if not duplicate:
            accepted.append(int(idx))
            if len(accepted) == n_dof:
                break
    if (
        len(accepted) < n_dof
    ):  # pragma: no cover - defensive; spectrum has n_dof families
        remaining = [int(i) for i in order if int(i) not in accepted]
        accepted.extend(remaining[: n_dof - len(accepted)])
    return np.asarray(accepted, dtype=np.int_)


def _monodromy(
    matrices: npt.NDArray[np.float64],
    angles: npt.NDArray[np.float64],
    rotor_speed: float,
) -> npt.NDArray[np.complex128]:
    """Return the Floquet multipliers of the validated, sorted azimuth stack."""
    # Azimuth interval each frozen matrix governs (wrapping past 2pi to close the loop).
    spans = np.diff(np.concatenate([angles, [angles[0] + _TWO_PI]]))
    monodromy = np.eye(matrices.shape[1], dtype=np.float64)
    for matrix, span in zip(matrices, spans, strict=True):
        monodromy = expm(matrix * (span / rotor_speed)) @ monodromy
    return np.asarray(np.linalg.eigvals(monodromy), dtype=np.complex128)


def _check_against_monodromy(
    exponents: npt.NDArray[np.complex128],
    matrices: npt.NDArray[np.float64],
    angles: npt.NDArray[np.float64],
    rotor_speed: float,
) -> None:
    """Warn when the Hill exponents do not reproduce the monodromy multipliers."""
    period = _TWO_PI / rotor_speed
    reference = np.log(_monodromy(matrices, angles, rotor_speed)) / period
    gaps = np.array(
        [[_family_gap(h, r, rotor_speed)[0] for r in reference] for h in exponents]
    )
    rows, cols = linear_sum_assignment(gaps)
    worst = float(np.max(gaps[rows, cols]))
    if worst > _MONODROMY_ATOL * rotor_speed:
        warnings.warn(
            "Hill exponents disagree with the monodromy multipliers by up to "
            f"{worst:.3g} rad/s (modulo i*Omega); increase n_harmonics and/or the "
            "azimuth sampling and cross-check with monodromy_exponents",
            RuntimeWarning,
            stacklevel=3,
        )


def _require_uniform_azimuths(angles: npt.NDArray[np.float64]) -> None:
    """Reject a sorted azimuth grid that is not uniform enough for Hill's method.

    Raises
    ------
    ValueError
        If any interval, including the one closing the revolution, differs from the
        mean interval by more than ``HILL_UNIFORM_RTOL`` of it. The message names the
        worst interval, its relative deviation and the monodromy alternative.
    """
    spacing = np.diff(np.concatenate([angles, [angles[0] + _TWO_PI]]))
    mean = float(spacing.mean())
    deviation = np.abs(spacing - mean)
    if np.all(deviation <= _HILL_UNIFORM_ATOL + HILL_UNIFORM_RTOL * mean):
        return
    worst = int(np.argmax(deviation))
    start = float(np.degrees(angles[worst]))
    end = float(np.degrees(angles[(worst + 1) % angles.size]))
    msg = (
        "floquet_modes (Hill's method) needs near-uniform azimuths: every azimuth "
        f"interval must equal the mean interval of {np.degrees(mean):.6g} deg within "
        f"a relative {HILL_UNIFORM_RTOL:g}, but the interval from {start:.6g} deg to "
        f"{end:.6g} deg is {np.degrees(spacing[worst]):.6g} deg, a relative deviation "
        f"of {deviation[worst] / mean:.3g}. Delivered OpenFAST azimuths are often "
        "uneven; use monodromy_exponents, which accepts any distinct azimuths"
    )
    raise ValueError(msg)


def floquet_modes(
    per_azimuth_a: npt.ArrayLike,
    azimuths: npt.ArrayLike,
    rotor_speed: float,
    *,
    n_harmonics: int = 4,
) -> FloquetSolution:
    r"""Compute the Floquet modes of an LTP system by Hill's harmonic-balance method.

    Parameters
    ----------
    per_azimuth_a : Sequence[npt.NDArray[np.float64]]
        The frozen-time state matrices :math:`\mathbf A(\psi_k)`, one per azimuth
        (e.g. ``MBCResult.per_azimuth_a`` with ``retain_per_azimuth=True``).
    azimuths : Sequence[float]
        The rotor azimuth (rad) of each matrix, spanning one revolution.
    rotor_speed : float
        The rotor speed :math:`\Omega` (rad/s); must be positive.
    n_harmonics : int, optional
        Harmonic truncation order ``M``; the Hill blocks need Fourier coefficients up to
        :math:`2M`, so the near-uniform azimuth sweep (see Notes) must have
        ``len(per_azimuth_a) >= 4*n_harmonics + 1`` samples so those coefficients are
        resolved without aliasing.

    Returns
    -------
    FloquetSolution
        The principal Floquet exponents, their frequencies/damping, and the harmonic
        mode-shape components.

    Raises
    ------
    ValueError
        If ``rotor_speed`` is not positive, the matrices are not square and consistent,
        the azimuth sweep is not near-uniform (see Notes; the message names the worst
        interval and its relative deviation), or it has too few samples to resolve the
        Hill coefficients for ``n_harmonics`` without aliasing.

    Warns
    -----
    RuntimeWarning
        If the selected exponents do not reproduce the monodromy multipliers
        (:math:`e^{\lambda T}`, compared modulo :math:`i\Omega`) to within
        :math:`10^{-2}\Omega` — the harmonic truncation or azimuth sampling is too
        coarse for this system.

    Notes
    -----
    **Limitation: near-uniform azimuths only.** The Fourier coefficients of
    :math:`\mathbf A(\psi)` are computed with the equal-weight discrete transform,
    which is exact only on a uniform grid. Every azimuth interval, including the one
    that closes the revolution, must equal the mean interval within a relative
    :data:`HILL_UNIFORM_RTOL` (``1e-3``); otherwise ``ValueError`` is raised. OpenFAST
    linearizes at the time steps nearest the requested azimuths, so a delivered grid is
    uneven unless the time step is commensurate with the azimuth spacing
    (:func:`~vane.modal.averaging_error_estimate.commensurate_time_step`) and the rotor
    runs at its trimmed speed. Analyse an uneven delivered grid with
    :func:`monodromy_exponents`, which weights each matrix by its own interval and
    accepts any distinct azimuths (its frequencies are known only modulo
    :math:`\Omega`). :func:`averaging_error` calls this function and has the same
    restriction.

    Every Floquet solution appears in the Hill spectrum as a family of copies
    :math:`\lambda + ik\Omega`. Exactly one representative per distinct solution is
    kept: candidates are walked in descending fundamental (:math:`m=0`) energy and a
    candidate that is a harmonic-shifted copy of an already-kept exponent (same
    exponent modulo :math:`i\Omega` *and* the same eigenvector shifted by :math:`k`
    blocks) is skipped. Inside a parametric-resonance tongue (real negative
    multipliers, :math:`\operatorname{Im}\lambda \equiv \Omega/2`) the conjugate of an
    exponent is such a copy, so both tongue families — including an unstable one — are
    kept; distinct solutions sharing a multiplier (an isotropic mode at exactly
    :math:`\Omega/2`) are both kept.
    """
    if not np.isfinite(rotor_speed) or rotor_speed <= 0.0:
        msg = (
            "rotor_speed must be positive and finite; Floquet analysis needs a "
            "spinning rotor"
        )
        raise ValueError(msg)
    if n_harmonics < 1:
        msg = f"n_harmonics must be at least 1, got {n_harmonics}"
        raise ValueError(msg)
    matrices, angles = _stack(per_azimuth_a, azimuths)
    n_azimuth, n_dof = matrices.shape[0], matrices.shape[1]
    required = 4 * n_harmonics + 1
    if n_azimuth < required:
        msg = (
            f"need at least {required} azimuths (4*n_harmonics+1) to resolve the Hill "
            f"coefficients up to 2*n_harmonics without aliasing for "
            f"n_harmonics={n_harmonics}, got {n_azimuth}"
        )
        raise ValueError(msg)
    _require_uniform_azimuths(angles)

    m = n_harmonics
    n_coeff = min(2 * m, (n_azimuth - 1) // 2)  # highest resolvable harmonic
    coeffs = _fourier_coefficients(matrices, angles, n_coeff)
    zero = np.zeros((n_dof, n_dof), dtype=np.complex128)

    blocks = 2 * m + 1
    size = blocks * n_dof
    hill = np.zeros((size, size), dtype=np.complex128)
    identity = np.eye(n_dof)
    for row in range(blocks):
        m_row = row - m
        for col in range(blocks):
            m_col = col - m
            block = coeffs.get(m_row - m_col, zero)
            sub = block.copy()
            if row == col:
                sub = sub - 1j * m_row * rotor_speed * identity
            hill[row * n_dof : (row + 1) * n_dof, col * n_dof : (col + 1) * n_dof] = sub

    values, vectors = np.linalg.eig(hill)
    tol = _DEGENERATE_RTOL * max(rotor_speed, float(np.max(np.abs(values))))
    vectors = _purify_degenerate(values, vectors, n_dof, m, tol)
    # Reshape each eigenvector into harmonic blocks and rank by fundamental energy.
    shaped = vectors.reshape(blocks, n_dof, size)  # (harmonic, dof, eigenindex)
    norms = np.linalg.norm(shaped, axis=1)  # (harmonic, eigenindex)
    total = np.linalg.norm(norms, axis=0)
    central = norms[m] / np.where(total > 0.0, total, 1.0)
    principal = _select_principal(values, shaped, central, n_dof, rotor_speed)

    exponents = values[principal]
    mode_shapes = shaped[:, :, principal]  # (harmonic, dof, n_modes)
    _check_against_monodromy(exponents, matrices, angles, rotor_speed)
    order = np.argsort(np.abs(exponents))
    exponents = exponents[order]
    mode_shapes = mode_shapes[:, :, order]
    magnitude = np.abs(exponents)
    with np.errstate(divide="ignore", invalid="ignore"):
        damping = np.where(magnitude > 0.0, -exponents.real / magnitude, 0.0)
    return FloquetSolution(
        exponents=np.asarray(exponents, dtype=np.complex128),
        natural_frequencies_hz=magnitude / _TWO_PI,
        damping_ratios=np.asarray(damping, dtype=np.float64),
        harmonics=np.arange(-m, m + 1, dtype=np.int_),
        harmonic_mode_shapes=np.asarray(mode_shapes, dtype=np.complex128),
        rotor_speed=rotor_speed,
    )


def monodromy_exponents(
    per_azimuth_a: npt.ArrayLike,
    azimuths: npt.ArrayLike,
    rotor_speed: float,
) -> npt.NDArray[np.complex128]:
    r"""Return the Floquet exponents via the monodromy matrix (independent of Hill).

    Integrates the state-transition matrix over one revolution as a product of
    matrix exponentials over the (piecewise-constant) azimuth intervals; the
    eigenvalues of that monodromy matrix are the Floquet multipliers :math:`\mu`,
    and the exponents are :math:`\ln\mu / T`.

    .. note::
       The multiplier :math:`\mu = e^{\lambda T}` is invariant under
       :math:`\lambda \mapsto \lambda + i k\Omega`, so this matrix determines the
       exponent's **real part (damping/growth rate) exactly but its imaginary part
       (frequency) only modulo** :math:`\Omega`: the principal branch of the logarithm
       folds it into the fundamental strip :math:`(-\Omega/2, \Omega/2]`. A mode whose
       damped frequency exceeds :math:`\Omega/2` is therefore reported aliased (e.g. a
       0.8 Hz mode with a 1 Hz rotor comes back at 0.2 Hz). Use :func:`floquet_modes`
       (Hill) for the unfolded frequencies; this helper is an exact cross-check of the
       damping and of frequencies within the strip.

    Parameters
    ----------
    per_azimuth_a : Sequence[npt.NDArray[np.float64]]
        The frozen-time state matrices, one per azimuth.
    azimuths : Sequence[float]
        The rotor azimuth (rad) of each matrix.
    rotor_speed : float
        The rotor speed :math:`\Omega` (rad/s); must be positive.

    Returns
    -------
    npt.NDArray[np.complex128]
        The Floquet exponents (real part exact; imaginary part folded into the
        principal strip :math:`(-\Omega/2, \Omega/2]`), sorted by natural frequency.

    Raises
    ------
    ValueError
        If ``rotor_speed`` is not positive and finite, the matrices are non-finite or
        inconsistently shaped, or the azimuths are non-finite or not distinct (after
        wrapping into ``[0, 2pi)``).
    """
    if not np.isfinite(rotor_speed) or rotor_speed <= 0.0:
        msg = (
            "rotor_speed must be positive and finite; Floquet analysis needs a "
            "spinning rotor"
        )
        raise ValueError(msg)
    matrices, angles = _stack(per_azimuth_a, azimuths)
    period = _TWO_PI / rotor_speed
    exponents = np.log(_monodromy(matrices, angles, rotor_speed)) / period
    order = np.argsort(np.abs(exponents))
    return np.asarray(exponents[order], dtype=np.complex128)


def averaging_error(
    per_azimuth_a: npt.ArrayLike,
    azimuths: npt.ArrayLike,
    rotor_speed: float,
    *,
    n_harmonics: int = 4,
) -> float:
    """Return the azimuth-averaging error: max frequency gap (Hz) Floquet vs averaged.

    Compares the Floquet natural frequencies to those of the azimuth-averaged model
    (the eigenvalues of the mean of ``per_azimuth_a``, which is what the averaged-LTI /
    MBC path uses). Zero for an isotropic rotor; it grows with anisotropy — the
    quantity that says when the averaged Campbell diagram is untrustworthy.

    Parameters
    ----------
    per_azimuth_a, azimuths, rotor_speed, n_harmonics
        As for :func:`floquet_modes`.

    Returns
    -------
    float
        The maximum absolute difference (Hz) between matched Floquet and averaged
        natural frequencies.

    Raises
    ------
    ValueError
        As :func:`floquet_modes`; in particular the azimuths must be near-uniform
        (relative :data:`HILL_UNIFORM_RTOL`), because the Floquet frequencies come from
        Hill's method.
    """
    # Reuse the validated, sorted stack (finite-checked) for the averaged model rather
    # than a fresh raw ``np.asarray`` that would let a non-finite matrix reach eigvals.
    matrices, _ = _stack(per_azimuth_a, azimuths)
    floquet = floquet_modes(
        per_azimuth_a, azimuths, rotor_speed, n_harmonics=n_harmonics
    )
    averaged = np.linalg.eigvals(np.mean(matrices, axis=0))
    averaged_hz = np.abs(averaged) / _TWO_PI
    floquet_hz = np.asarray(floquet.natural_frequencies_hz, dtype=np.float64)
    # Match each Floquet frequency to its nearest averaged frequency via an optimal
    # one-to-one assignment, not by sorted position: under strong anisotropy the two
    # spectra reorder, so a positional difference would compare different physical modes
    # and could inflate or mask the gap. The two sets have the same count (n_dof).
    cost = np.abs(floquet_hz[:, None] - averaged_hz[None, :])
    rows, cols = linear_sum_assignment(cost)
    return float(np.max(cost[rows, cols]))
