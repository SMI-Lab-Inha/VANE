# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Eigenvalue analysis of the azimuth-averaged state matrix.

Given the non-rotating (MBC-averaged) state matrix :math:`A` in the canonical
``[q2, q2dot, q1]`` layout, an eigenvalue analysis yields the full-system modes:
their natural frequencies, damping ratios, and (complex) mode shapes.

Each under-damped mode appears as a complex-conjugate eigenvalue pair; only the
representative with positive imaginary part is retained. For an eigenvalue
:math:`\lambda`, the modal quantities are

.. math::

    \omega_n = |\lambda|, \qquad
    \zeta = \frac{-\operatorname{Re}\lambda}{|\lambda|}, \qquad
    \omega_d = \operatorname{Im}\lambda,

so the natural and damped frequencies in hertz are
:math:`\omega_n / 2\pi` and :math:`\omega_d / 2\pi`. The velocity (``q2dot``)
rows are dropped from the returned mode shapes, leaving the second-order
displacements and first-order states; the full eigenvectors are also retained.

Notes
-----
OpenFAST exposes no full-system mass or stiffness matrix, so modal frequencies
are obtained directly from :math:`A` and vary with operating point.

**Numerically split rigid-body pairs.** A degree of freedom without restoring
stiffness -- the free rotor-azimuth (generator) DOF of a variable-speed turbine --
contributes a zero eigenvalue (the azimuth integrator) next to a small real one (the
aerodynamic damping of the rotor speed). Where the damping is small, the
finite-difference noise in the azimuth column of :math:`A` is enough to merge the two
into a complex pair of negligible frequency (periods of tens of minutes) instead of
two real eigenvalues. Such a pair is not an oscillation, and reporting it as one makes
a spurious mode appear at isolated operating points. A complex pair is therefore
classified as a *split rigid-body pair* -- dropped from the oscillatory modes and
counted in ``n_rigid_body_modes`` -- when both

1. its natural frequency is below ``rigid_body_frequency_hz`` (default
   :data:`RIGID_BODY_FREQUENCY_HZ`, a period of 333 s: well below every
   restoring-force mode of a wind turbine, including the 100-200 s surge and sway of
   moored floating platforms), and
2. its imaginary part is not resolved by the data: by first-order perturbation
   theory a relative error :math:`\delta` in :math:`A` moves the eigenvalue by up to
   :math:`\kappa\,\delta\,\rho(A)`, so a pair with

   .. math::

       |\operatorname{Im}\lambda| \le \delta\,\kappa(\lambda)\,\rho(A)

   is indistinguishable from two real eigenvalues. :math:`\kappa` is evaluated on the
   diagonally balanced matrix (the problem the eigenvalue solver actually solves, and
   independent of the state units), :math:`\rho` is the spectral radius and
   :math:`\delta` is ``split_pair_rtol`` (default :data:`SPLIT_PAIR_RTOL`).

Both conditions are needed. The first alone would drop a well-resolved slow
oscillation of an unusually soft model. The second alone is not enough because the
spectral radius of a model with stiff first-order states (MoorDyn line nodes) is
large, so well-resolved surge and sway modes of such a model can fall below the bound;
the frequency floor keeps them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import eig as _scipy_eig
from scipy.linalg import matrix_balance as _scipy_matrix_balance

if TYPE_CHECKING:
    import numpy.typing as npt

    from vane.io.mbc_transform import MBCResult

__all__ = [
    "RIGID_BODY_FREQUENCY_HZ",
    "SPLIT_PAIR_RTOL",
    "ModalSolution",
    "compute_modes",
    "modes_from_mbc",
]

_LOGGER = logging.getLogger(__name__)

_TWO_PI = 2.0 * np.pi

RIGID_BODY_FREQUENCY_HZ = 3e-3
"""Natural frequency (Hz) below which a numerically split pair may be rigid-body.

A period of 333 s. Moored floating platforms have their slowest restoring-force modes
(surge, sway) at periods of roughly 100-200 s (0.005-0.01 Hz), so no physical
oscillation of a wind turbine is this slow.
"""

SPLIT_PAIR_RTOL = 1e-4
r"""Relative accuracy :math:`\delta` of the state matrix assumed for split pairs.

OpenFAST computes its Jacobians by finite differences with perturbations of order
:math:`10^{-3}`, and the MBC transform averages them over azimuth, so the entries of
:math:`A` are not reliable to better than about this relative level. On the
reference-turbine sweeps the split rigid-body pairs sit 10-200 times inside the
resulting bound, and the platform modes of floating models without MoorDyn 2-50 times
outside it.
"""


@dataclass
class ModalSolution:
    r"""Modal properties extracted from a state matrix.

    All per-mode arrays share the same ordering (ascending natural frequency when
    the solver sorts). Only under-damped modes (positive imaginary eigenvalue)
    are represented.

    Parameters
    ----------
    eigenvalues : npt.NDArray[np.complex128]
        Retained eigenvalues :math:`\lambda`, shape ``(n_modes,)``.
    natural_frequencies_hz : npt.NDArray[np.float64]
        Natural frequencies :math:`|\lambda| / 2\pi`, shape ``(n_modes,)``.
    damping_ratios : npt.NDArray[np.float64]
        Damping ratios :math:`-\operatorname{Re}\lambda / |\lambda|`.
    damped_frequencies_hz : npt.NDArray[np.float64]
        Damped frequencies :math:`\operatorname{Im}\lambda / 2\pi`.
    mode_shapes : npt.NDArray[np.complex128]
        Reduced complex mode shapes (second-order displacements then first-order
        states), shape ``(ndof2 + ndof1, n_modes)``, phase-normalized so each
        mode's dominant component is real and positive.
    full_eigenvectors : npt.NDArray[np.complex128]
        Full eigenvectors over all states, shape ``(n_states, n_modes)``, sharing
        the same phase normalization.
    full_left_eigenvectors : npt.NDArray[np.complex128]
        Full left eigenvectors over all states, shape ``(n_states, n_modes)``.
        Together with ``full_eigenvectors`` these support scale-invariant
        biorthogonal participation and first-order eigenvalue sensitivity.
    n_rigid_body_modes : int
        Number of degrees of freedom without an under-damped representative
        (rigid-body or over-damped modes, including split rigid-body pairs).
    dof_descriptions : list[str]
        Descriptions of the degrees of freedom that index the rows of
        ``mode_shapes`` (the second-order displacements then first-order states).
        Empty when descriptions were not supplied to the solver.
    state_descriptions : list[str]
        Descriptions of every state indexing the full left and right eigenvectors.
        Empty when descriptions were not supplied to the solver.
    dof_mbc_coordinates : list[str]
        Multi-blade coordinate (``"collective"``/``"cosine"``/``"sine"``/``""``) of
        each ``mode_shapes`` row, aligned with ``dof_descriptions``. Empty when not
        supplied.
    condition_numbers : npt.NDArray[np.float64]
        Per-mode eigenvalue condition number :math:`\kappa = 1 / |y^H x|` (the
        reciprocal cosine of the angle between the left and right eigenvectors),
        shape ``(n_modes,)``. ``1`` is perfectly conditioned; large values flag
        eigenvalues that are numerically sensitive (highly non-normal system).
    is_degenerate : npt.NDArray[np.bool_]
        Per-mode flag marking modes whose eigenvalue is (near-)repeated by another
        retained mode, so the individual mode shape is defined only up to a
        rotation within the shared invariant subspace.
    n_unstable : int
        Number of eigenvalues with positive real part (unstable). Non-zero is a
        red flag for the operating point. Split rigid-body pairs are not counted:
        the sign of their real part is not resolved by the data.
    n_overdamped : int
        Number of non-oscillatory stable real eigenvalues (over-damped modes),
        which are not returned as oscillatory modes.
    n_split_rigid_body_pairs : int
        Number of complex-conjugate pairs classified as numerically split
        rigid-body eigenvalues (see the module notes) and therefore left out of the
        oscillatory modes; each also counts in ``n_rigid_body_modes``.
    rotor_speed_rpm : float
        Rotor speed (rev/min) of the operating point the state matrix was
        linearized at, or ``nan`` when unknown (e.g. a matrix given directly to
        :func:`compute_modes`). :func:`modes_from_mbc` copies it from the MBC
        result; the labeller uses it to read a blade whirl mode's damping in the
        rotating frame.
    """

    eigenvalues: npt.NDArray[np.complex128]
    natural_frequencies_hz: npt.NDArray[np.float64]
    damping_ratios: npt.NDArray[np.float64]
    damped_frequencies_hz: npt.NDArray[np.float64]
    mode_shapes: npt.NDArray[np.complex128]
    full_eigenvectors: npt.NDArray[np.complex128]
    n_rigid_body_modes: int
    dof_descriptions: list[str]
    condition_numbers: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    is_degenerate: npt.NDArray[np.bool_] = field(
        default_factory=lambda: np.empty(0, dtype=np.bool_)
    )
    n_unstable: int = 0
    n_overdamped: int = 0
    # Appended after the Stage-1 optional fields to keep positional construction
    # of the previously-existing fields stable.
    dof_mbc_coordinates: list[str] = field(default_factory=list)
    full_left_eigenvectors: npt.NDArray[np.complex128] = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.complex128)
    )
    state_descriptions: list[str] = field(default_factory=list)
    n_split_rigid_body_pairs: int = 0
    rotor_speed_rpm: float = float("nan")

    @property
    def n_modes(self) -> int:
        """Return the number of retained modes."""
        return int(self.eigenvalues.shape[0])

    @property
    def natural_frequencies_rad_s(self) -> npt.NDArray[np.float64]:
        """Return natural frequencies in rad/s."""
        return self.natural_frequencies_hz * _TWO_PI

    @property
    def max_condition_number(self) -> float:
        """Return the worst (largest) eigenvalue condition number, or 1 if none."""
        if self.condition_numbers.size == 0:
            return 1.0
        return float(self.condition_numbers.max())


def compute_modes(
    a: npt.NDArray[np.float64],
    ndof2: int,
    ndof1: int,
    *,
    sort_by_frequency: bool = True,
    descriptions: list[str] | None = None,
    mbc_coordinates: list[str] | None = None,
    rigid_body_frequency_hz: float = RIGID_BODY_FREQUENCY_HZ,
    split_pair_rtol: float = SPLIT_PAIR_RTOL,
) -> ModalSolution:
    """Compute modal properties from a state matrix.

    Parameters
    ----------
    a : npt.NDArray[np.float64]
        Square state matrix in ``[q2, q2dot, q1]`` order, shape
        ``(2 * ndof2 + ndof1, 2 * ndof2 + ndof1)``.
    ndof2 : int
        Number of second-order degrees of freedom (the ``q2`` block size).
    ndof1 : int
        Number of first-order states (the ``q1`` block size).
    sort_by_frequency : bool, optional
        Sort the returned modes by ascending natural frequency (the default).
    descriptions : list[str] or None, optional
        Descriptions of all ``2 * ndof2 + ndof1`` states, in the matrix order.
        When given, the subset that indexes the mode-shape rows is stored on the
        result as ``dof_descriptions``. A list of only the ``ndof2 + ndof1``
        mode-shape rows (displacements then first-order states) is also accepted
        and stored as is; ``state_descriptions`` is then left empty.
    mbc_coordinates : list[str] or None, optional
        Multi-blade coordinate of all states (or of the mode-shape rows), in the
        matrix order; the mode-shape subset is stored on the result as
        ``dof_mbc_coordinates``.
    rigid_body_frequency_hz : float, optional
        Natural frequency below which a complex pair whose imaginary part is not
        resolved by the data is classified as a split rigid-body pair (see the
        module notes). Default :data:`RIGID_BODY_FREQUENCY_HZ`; ``0.0`` keeps every
        complex pair as an oscillatory mode.
    split_pair_rtol : float, optional
        Assumed relative accuracy of ``a`` in the split-pair test. Default
        :data:`SPLIT_PAIR_RTOL`; ``0.0`` keeps every complex pair.

    Returns
    -------
    ModalSolution
        The modal frequencies, damping ratios, and mode shapes.

    Raises
    ------
    ValueError
        If ``a`` is not square, if ``ndof2``/``ndof1`` are not non-negative integers,
        if its size does not match ``2 * ndof2 + ndof1``, if ``descriptions`` or
        ``mbc_coordinates`` has neither one entry per state nor one per mode-shape
        row, or if
        ``rigid_body_frequency_hz`` or ``split_pair_rtol`` is negative or non-finite.
    """
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        msg = f"State matrix must be square, got shape {a.shape}"
        raise ValueError(msg)
    n_states = a.shape[0]
    # The partition sizes must be non-negative whole numbers before the count equation:
    # e.g. ndof2=-1, ndof1=3 satisfies 2*ndof2+ndof1 == 1 yet describes an impossible
    # layout, and would yield a degenerate (3, 0) mode-shape array from a 1x1 A.
    for name, value in (("ndof2", ndof2), ("ndof1", ndof1)):
        if not isinstance(value, (int, np.integer)) or value < 0:
            msg = f"{name} must be a non-negative integer, got {value!r}"
            raise ValueError(msg)
    if 2 * ndof2 + ndof1 != n_states:
        msg = (
            f"State count mismatch: 2*ndof2 + ndof1 = {2 * ndof2 + ndof1} "
            f"but A is {n_states}x{n_states}"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(a)):
        msg = "State matrix contains non-finite values (NaN or Inf)"
        raise ValueError(msg)
    # Per-state lists cover every state, or only the mode-shape rows (displacements
    # then first-order states). Any other length would silently attach descriptions
    # (and hence labels) to the wrong states, or fail later with an opaque IndexError.
    n_rows = ndof2 + ndof1
    for name, values in (
        ("descriptions", descriptions),
        ("mbc_coordinates", mbc_coordinates),
    ):
        if values is not None and len(values) not in (n_states, n_rows):
            msg = (
                f"{name} has {len(values)} entries but A has {n_states} states "
                f"({n_rows} mode-shape rows)"
            )
            raise ValueError(msg)
    for name, tolerance in (
        ("rigid_body_frequency_hz", rigid_body_frequency_hz),
        ("split_pair_rtol", split_pair_rtol),
    ):
        if not (np.isfinite(tolerance) and tolerance >= 0.0):
            msg = f"{name} must be a finite non-negative number, got {tolerance}"
            raise ValueError(msg)

    eigen = _scipy_eig(a, left=True, right=True)
    eigenvalues = np.asarray(eigen[0], dtype=np.complex128)
    left_vectors = np.asarray(eigen[1], dtype=np.complex128)
    right_vectors = np.asarray(eigen[2], dtype=np.complex128)

    split = _split_rigid_body_pairs(
        a,
        eigenvalues,
        left_vectors,
        right_vectors,
        rigid_body_frequency_hz=rigid_body_frequency_hz,
        rtol=split_pair_rtol,
    )
    n_split = int(np.count_nonzero(split & (eigenvalues.imag > 0.0)))
    if n_split:
        _LOGGER.debug(
            "Classified %d numerically split complex pair(s) below %g Hz as "
            "rigid-body eigenvalues",
            n_split,
            rigid_body_frequency_hz,
        )
    n_unstable, n_overdamped = _classify_spectrum(eigenvalues[~split])
    conditioning = _eigenvalue_conditioning(left_vectors, right_vectors)

    keep = np.flatnonzero((eigenvalues.imag > 0.0) & ~split)
    lam = eigenvalues[keep].astype(np.complex128)
    condition = conditioning[keep]
    degenerate = _degeneracy_flags(lam)

    omega_n = np.abs(lam)
    damping = -lam.real / omega_n
    omega_d = lam.imag

    displacement_rows = np.r_[0:ndof2, 2 * ndof2 : n_states]
    reduced = right_vectors[np.ix_(displacement_rows, keep)].astype(np.complex128)
    full = right_vectors[:, keep].astype(np.complex128)
    full_left = left_vectors[:, keep].astype(np.complex128)
    _phase_normalize(reduced, full)

    if sort_by_frequency:
        # Natural frequency first, then decay rate and damped frequency, so modes of
        # equal natural frequency are ordered the same way whatever order the
        # eigenvalue solver returned them in.
        order = np.lexsort((lam.imag, -lam.real, omega_n))
        lam, omega_n, damping, omega_d, condition, degenerate = (
            lam[order],
            omega_n[order],
            damping[order],
            omega_d[order],
            condition[order],
            degenerate[order],
        )
        reduced, full, full_left = (
            reduced[:, order],
            full[:, order],
            full_left[:, order],
        )

    reduced_rows = displacement_rows.tolist()
    dof_descriptions = _mode_shape_rows(descriptions, reduced_rows, n_states)
    dof_mbc_coordinates = _mode_shape_rows(mbc_coordinates, reduced_rows, n_states)
    full_descriptions = (
        list(descriptions)
        if descriptions is not None and len(descriptions) == n_states
        else []
    )
    return ModalSolution(
        eigenvalues=lam,
        natural_frequencies_hz=omega_n / _TWO_PI,
        damping_ratios=damping,
        damped_frequencies_hz=omega_d / _TWO_PI,
        mode_shapes=reduced,
        full_eigenvectors=full,
        n_rigid_body_modes=(ndof2 + ndof1) - int(keep.shape[0]),
        dof_descriptions=dof_descriptions,
        dof_mbc_coordinates=dof_mbc_coordinates,
        condition_numbers=condition,
        is_degenerate=degenerate,
        n_unstable=n_unstable,
        n_overdamped=n_overdamped,
        full_left_eigenvectors=full_left,
        state_descriptions=full_descriptions,
        n_split_rigid_body_pairs=n_split,
    )


def _mode_shape_rows(
    values: list[str] | None, rows: list[int], n_states: int
) -> list[str]:
    """Return the entries of a per-state list for the mode-shape rows.

    ``values`` covers every state (it is then indexed by ``rows``) or only the
    mode-shape rows (returned as is); ``None`` or empty gives an empty list.
    """
    if not values:
        return []
    if len(values) == n_states:
        return [values[i] for i in rows]
    return list(values)


def modes_from_mbc(
    result: MBCResult,
    *,
    rigid_body_frequency_hz: float = RIGID_BODY_FREQUENCY_HZ,
    split_pair_rtol: float = SPLIT_PAIR_RTOL,
) -> ModalSolution:
    """Compute modal properties from an MBC transform result.

    Parameters
    ----------
    result : MBCResult
        Output of :func:`vane.io.mbc_transform.mbc3_transform`; its ``avg_a`` must
        be present.
    rigid_body_frequency_hz : float, optional
        Frequency floor of the split rigid-body pair test; see
        :func:`compute_modes`.
    split_pair_rtol : float, optional
        Relative accuracy of the split rigid-body pair test; see
        :func:`compute_modes`.

    Returns
    -------
    ModalSolution
        The modal solution for the averaged state matrix.

    Raises
    ------
    ValueError
        If ``result.avg_a`` is ``None``.
    """
    if result.avg_a is None:
        msg = "MBCResult has no averaged A matrix to analyse"
        raise ValueError(msg)
    solution = compute_modes(
        result.avg_a,
        result.ndof2,
        result.ndof1,
        descriptions=result.state_descriptions,
        mbc_coordinates=result.mbc_coordinates or None,
        rigid_body_frequency_hz=rigid_body_frequency_hz,
        split_pair_rtol=split_pair_rtol,
    )
    solution.rotor_speed_rpm = float(result.rotor_speed_rpm)
    return solution


def _phase_normalize(
    reduced: npt.NDArray[np.complex128], full: npt.NDArray[np.complex128]
) -> None:
    """Rotate each mode's phase in place so its dominant component is real positive.

    For every column, the component of largest magnitude in ``reduced`` is made
    real and positive by dividing the column (in both ``reduced`` and ``full``) by
    that component's unit phase. This gives a deterministic, sign-stable mode
    shape across operating points.

    Parameters
    ----------
    reduced : npt.NDArray[np.complex128]
        Reduced mode-shape matrix, modified in place.
    full : npt.NDArray[np.complex128]
        Full eigenvector matrix, modified in place with the same rotation.
    """
    for col in range(reduced.shape[1]):
        dominant = reduced[np.argmax(np.abs(reduced[:, col])), col]
        magnitude = np.abs(dominant)
        if magnitude == 0.0:
            continue
        unit_phase = dominant / magnitude
        reduced[:, col] /= unit_phase
        full[:, col] /= unit_phase


def _classify_spectrum(eigenvalues: npt.NDArray[np.complex128]) -> tuple[int, int]:
    """Count unstable and over-damped eigenvalues in the full spectrum.

    Parameters
    ----------
    eigenvalues : npt.NDArray[np.complex128]
        All eigenvalues of the state matrix.

    Returns
    -------
    tuple[int, int]
        ``(n_unstable, n_overdamped)`` — eigenvalues with positive real part, and
        non-zero stable real (non-oscillatory) eigenvalues, using a tolerance
        scaled to the spectral radius.
    """
    if eigenvalues.size == 0:
        return 0, 0
    # Absolute numerical-zero floor (~1000x the eigenvalue round-off, eps*scale),
    # NOT a coarse fraction of the spectral radius: scaling the threshold by the
    # largest eigenvalue would hide slow instabilities in a stiff spectrum.
    scale = max(float(np.abs(eigenvalues).max()), 1.0)
    tolerance = 1e3 * float(np.finfo(np.float64).eps) * scale
    n_unstable = int(np.count_nonzero(eigenvalues.real > tolerance))
    is_real = np.abs(eigenvalues.imag) <= tolerance
    is_nonzero = np.abs(eigenvalues) > tolerance
    n_overdamped = int(
        np.count_nonzero(is_real & (eigenvalues.real < -tolerance) & is_nonzero)
    )
    return n_unstable, n_overdamped


def _split_rigid_body_pairs(
    a: npt.NDArray[np.float64],
    eigenvalues: npt.NDArray[np.complex128],
    left: npt.NDArray[np.complex128],
    right: npt.NDArray[np.complex128],
    *,
    rigid_body_frequency_hz: float,
    rtol: float,
) -> npt.NDArray[np.bool_]:
    r"""Flag complex eigenvalues that are numerically split rigid-body pairs.

    A complex eigenvalue (either member of the pair) is flagged when its natural
    frequency is below ``rigid_body_frequency_hz`` and
    :math:`|\operatorname{Im}\lambda| \le \delta\,\kappa_b(\lambda)\,\rho(A)`, with
    :math:`\kappa_b` the condition number on the diagonally balanced matrix
    :math:`D^{-1} A D`. Its right and left eigenvectors are :math:`D^{-1}x` and
    :math:`D y`, so :math:`\kappa_b = \|D^{-1}x\|\,\|D y\| / |y^H x|` follows from
    the eigenvectors of ``a`` without a second decomposition.

    Parameters
    ----------
    a : npt.NDArray[np.float64]
        The state matrix.
    eigenvalues : npt.NDArray[np.complex128]
        All eigenvalues of ``a``.
    left, right : npt.NDArray[np.complex128]
        The matching left and right eigenvectors as columns.
    rigid_body_frequency_hz : float
        Frequency floor of the test.
    rtol : float
        Relative accuracy :math:`\delta` of ``a``.

    Returns
    -------
    npt.NDArray[np.bool_]
        ``True`` for each eigenvalue of a split rigid-body pair.
    """
    flags = np.zeros(eigenvalues.shape[0], dtype=np.bool_)
    if eigenvalues.size == 0 or rigid_body_frequency_hz == 0.0 or rtol == 0.0:
        return flags
    candidates = np.flatnonzero(
        (eigenvalues.imag > 0.0)
        & (np.abs(eigenvalues) < _TWO_PI * rigid_body_frequency_hz)
    )
    if candidates.size == 0:
        return flags
    _, (scale, _) = _scipy_matrix_balance(a, permute=False, separate=True)
    scale = np.asarray(scale, dtype=np.float64)
    radius = float(np.abs(eigenvalues).max())
    for index in candidates:
        x = right[:, index]
        y = left[:, index]
        overlap = abs(np.vdot(y, x))
        if overlap > 0.0:
            kappa = np.linalg.norm(x / scale) * np.linalg.norm(y * scale) / overlap
            split = abs(eigenvalues[index].imag) <= rtol * float(kappa) * radius
        else:
            split = True  # a defective eigenvalue: the split is pure round-off
        if split:
            # Flag both members of the pair (the partner is the conjugate).
            partner = int(np.argmin(np.abs(eigenvalues - np.conj(eigenvalues[index]))))
            flags[index] = flags[partner] = True
    return flags


def _eigenvalue_conditioning(
    left: npt.NDArray[np.complex128], right: npt.NDArray[np.complex128]
) -> npt.NDArray[np.float64]:
    r"""Return each eigenvalue's condition number from its left/right eigenvectors.

    The condition number is :math:`\kappa = 1 / |y^H x|` with unit-norm left
    eigenvector ``y`` and right eigenvector ``x``; by Cauchy-Schwarz ``kappa >= 1``,
    equal to 1 for a normal matrix and large for an ill-conditioned (highly
    non-normal) eigenvalue.

    Parameters
    ----------
    left, right : npt.NDArray[np.complex128]
        Unit-norm left and right eigenvectors as columns, in matching order.

    Returns
    -------
    npt.NDArray[np.float64]
        The condition number per eigenvalue (``inf`` for a defective eigenvalue).
    """
    sensitivity = np.abs(np.sum(np.conj(left) * right, axis=0))
    with np.errstate(divide="ignore"):
        condition = np.where(sensitivity > 0.0, 1.0 / sensitivity, np.inf)
    return np.asarray(condition, dtype=np.float64)


# Relative tolerance below which two eigenvalues are treated as coincident.
DEGENERACY_RTOL = 1e-6


def _degeneracy_flags(
    eigenvalues: npt.NDArray[np.complex128], *, rtol: float = DEGENERACY_RTOL
) -> npt.NDArray[np.bool_]:
    """Flag eigenvalues that are (near-)repeated by another in the set.

    Parameters
    ----------
    eigenvalues : npt.NDArray[np.complex128]
        The retained eigenvalues.
    rtol : float, optional
        Relative tolerance for treating two eigenvalues as coincident.

    Returns
    -------
    npt.NDArray[np.bool_]
        ``True`` for each eigenvalue within ``rtol`` of another (its mode shape is
        then defined only up to a rotation within the shared invariant subspace).
    """
    n = eigenvalues.shape[0]
    flags = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        for j in range(i + 1, n):
            scale = max(abs(eigenvalues[i]), abs(eigenvalues[j]), 1e-30)
            if abs(eigenvalues[i] - eigenvalues[j]) <= rtol * scale:
                flags[i] = True
                flags[j] = True
    return flags
