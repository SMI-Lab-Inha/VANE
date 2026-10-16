# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Per-branch error estimates of the azimuth-averaged model.

The averaged (Coleman/MBC) analysis of a rotating turbine carries three errors that
are usually only guessed at. Each is predicted here from the averaged model and the
harmonic content of the samples, without a periodic (Floquet) solution:

* **Sampling.** The error of a coarse or hypothetical azimuth grid. Its matrix error
  is exactly a quadrature-weight error times the harmonic content
  (:func:`vane.modal.averaging.grid_error_matrix`), and its effect on each branch
  follows from eigenvalue perturbation of the averaged eigensolution: first or second
  order, or, when the rotor-azimuth state is present, a **structured** predictor that
  keeps the azimuth column as an exact rank-one term and the near-Jordan
  azimuth/rotor-speed modes exactly. The structured predictor also yields a
  predicted modal assurance against the reference branch, the **hybridisation flag**.
* **Quadrature.** The difference between the arithmetic sample mean (what the MBC
  transform applies) and the interval-weighted mean on the delivered grid.
* **Periodicity.** The second-order sideband estimate of the Floquet exponent minus
  the averaged eigenvalue (:func:`vane.modal.averaging.sideband_correction`).

All terms are reported in the same units: the relative change of the damped
frequency :math:`|\operatorname{Im}\lambda|` and of the damping ratio
:math:`\zeta=-\operatorname{Re}\lambda/|\lambda|`, in percent of the reference value.
:func:`estimate_averaging_error` assembles them per branch, together with the azimuth
consistency test. :func:`azimuth_count_table` designs the azimuth count from the
:math:`\operatorname{lcm}(N, B)` aliasing rule and the fitted harmonics, and
:func:`commensurate_time_step` suggests a time step that puts every target azimuth on
a whole number of steps.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from vane.modal.averaging import (
    AzimuthConsistency,
    AzimuthStates,
    EigenBasis,
    HarmonicModel,
    TowerPassageFit,
    _azimuth_array,
    _matrix_stack,
    _position,
    _positive_rate,
    _sideband_all,
    azimuth_consistency,
    eigen_basis,
    fit_harmonics,
    grid_error_matrix,
    interval_weights,
    uniform_azimuths,
    weight_kernel,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    import numpy.typing as npt

__all__ = [
    "AliasTail",
    "AveragingErrorEstimate",
    "AzimuthCountRow",
    "AzimuthCountTable",
    "BranchErrorEstimate",
    "CommensurateStep",
    "ErrorEstimate",
    "GridReport",
    "alias_free_azimuth_count",
    "azimuth_count_table",
    "commensurate_time_step",
    "estimate_averaging_error",
    "first_alias_order",
    "predict_branch_shift",
    "trial_branch_shift",
]

_TWO_PI = 2.0 * np.pi
# A branch whose predicted eigenvector has a MAC below this with the reference branch
# has hybridised (merged with another mode, typically the azimuth pendulum).
_DEFAULT_MAC_THRESHOLD = 0.9
# Modes whose azimuth plus rotor-speed participation exceeds this share are retained
# exactly by the structured predictor (the near-Jordan azimuth/speed pair).
_DEFAULT_SUBSPACE_SHARE = 0.05
# Conjugate-side candidates are excluded when tracking a branch.
_IMAG_TOL = 1e-12
# Above this eigenvalue condition number a branch is ill-conditioned: its first-order
# sensitivity to a matrix error is a thousand times that of a normal matrix, and on the
# floating reference turbines the perturbation predictors under-predict such branches.
_DEFAULT_MAX_CONDITION = 1e3
# Modes whose coupling to a branch exceeds this fraction of their detuning are coupled
# exactly by the general mode-mixing check; a weaker coupling moves the eigenvector MAC
# by less than its square (2.5e-5). Modes linked to a linked mode join as well, since
# on non-normal (mooring, flexible-blade) models many such modes move together.
_DEFAULT_MIXING_COUPLING = 0.005
# The alias tail is summed until a term falls below this share of the sum.
_TAIL_RTOL = 1e-12
# Hard stop of the tail sum (orders), far beyond any physical content.
_TAIL_MAX_ORDER = 100_000
# A tower-passage fit below this log R^2 (no clear decay of the column) gives no tail.
_MIN_TOWER_R2 = 0.5
# A branch's own sensitivity counts as decaying when its fitted trend falls at least to
# this fraction over the resolved blade multiples it is fitted on.
_MIN_OWN_DROP = 0.5
# A branch sensitivity below this fraction of the mean's norm is rounding, not content.
_NO_CONTENT = 1e-12
# A delivered azimuth matches a requested grid azimuth within this angle (rad).
_MATCH_TOL = 1e-9
_Predictor = Literal["first", "second", "structured"]
# "auto" maps to a placeholder that is resolved from the presence of the azimuth state.
_METHODS: dict[str, _Predictor] = {
    "auto": "second",
    "first": "first",
    "second": "second",
    "structured": "structured",
}


# --------------------------------------------------------------------------------------
# Branch estimates
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ErrorEstimate:
    r"""One predicted error term of a branch.

    Parameters
    ----------
    shift : complex
        Predicted eigenvalue change (1/s) relative to the reference branch.
    frequency_error_pct : float
        Relative change of the damped frequency, percent (``nan`` for a real
        reference eigenvalue).
    damping_error_pct : float
        Relative change of the damping ratio, percent (``nan`` for an undamped
        reference).
    mac : float
        Predicted modal assurance of the perturbed eigenvector with the reference one
        (``nan`` when the term predicts no eigenvector).
    hybridised : bool
        ``mac`` is below the hybridisation threshold.
    method : str
        ``"first"``, ``"second"``, ``"structured"``, ``"exact"`` or ``"sideband"``.
    mixing_mac : float, optional
        Modal assurance of the general mode-mixing check: the reference eigenvector
        against its counterpart in the perturbed model, from the exact trial
        eigenproblem or from an exact solve over the modes strongly coupled to the
        branch, independent of the azimuth subspace (``nan`` when not evaluated).
    mixed : bool, optional
        ``mixing_mac`` is below the hybridisation threshold: the branch has mixed with
        another mode, whatever that mode is.
    shift_bound : float, optional
        First-order conditioning bound :math:`\kappa_j\|\mathbf E\|_2` on
        :math:`|\delta\lambda|` (1/s), ``nan`` when not evaluated.
    """

    shift: complex
    frequency_error_pct: float
    damping_error_pct: float
    mac: float
    hybridised: bool
    method: str
    mixing_mac: float = math.nan
    mixed: bool = False
    shift_bound: float = math.nan

    def within(
        self, damping_tolerance_pct: float, frequency_tolerance_pct: float
    ) -> bool:
        """Return whether both errors are within the tolerances and nothing mixed.

        Parameters
        ----------
        damping_tolerance_pct : float
            Damping-ratio tolerance, percent of the reference value.
        frequency_tolerance_pct : float
            Frequency tolerance, percent of the reference value.

        Returns
        -------
        bool
            ``True`` when neither error exceeds its tolerance (an undefined, NaN error
            is not checked; an infinite one exceeds it) and the branch has neither
            hybridised nor mixed.
        """
        damping = self.damping_error_pct
        frequency = self.frequency_error_pct
        return not (
            self.hybridised
            or self.mixed
            or (not math.isnan(damping) and damping > damping_tolerance_pct)
            or (not math.isnan(frequency) and frequency > frequency_tolerance_pct)
        )


def _relative_errors(reference: complex, trial: complex) -> tuple[float, float]:
    """Return the relative damped-frequency and damping-ratio errors in percent."""
    if not (math.isfinite(trial.real) and math.isfinite(trial.imag)):
        return math.inf, math.inf  # an unbounded estimate (an exact resonance)
    f_ref, f_trial = abs(reference.imag), abs(trial.imag)
    frequency = 100.0 * abs(f_trial - f_ref) / f_ref if f_ref > 0.0 else math.nan
    z_ref = -reference.real / abs(reference) if reference != 0 else 0.0
    z_trial = -trial.real / abs(trial) if trial != 0 else 0.0
    damping = 100.0 * abs(z_trial - z_ref) / abs(z_ref) if z_ref != 0.0 else math.nan
    return frequency, damping


def _mac(
    reference: npt.NDArray[np.complex128], vectors: npt.NDArray[np.complex128]
) -> npt.NDArray[np.float64]:
    """Return the complex MAC of one vector against each column of ``vectors``."""
    numerator = np.abs(reference.conj() @ vectors) ** 2
    denominator = np.vdot(reference, reference).real * np.sum(
        np.abs(vectors) ** 2, axis=0
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denominator > 0.0, numerator / denominator, 0.0)
    return np.asarray(ratio, dtype=np.float64)


def _same_side(
    macs: npt.NDArray[np.float64],
    values: npt.NDArray[np.complex128],
    reference: complex,
) -> npt.NDArray[np.float64]:
    """Zero the MACs of candidates on the other side of the real axis."""
    out = np.array(macs, dtype=np.float64)
    if reference.imag > 0.0:
        out[values.imag < -_IMAG_TOL] = 0.0
    elif reference.imag < 0.0:
        out[values.imag > _IMAG_TOL] = 0.0
    return out


def _estimate(
    reference: complex,
    shift: complex,
    mac: float,
    method: str,
    mac_threshold: float,
    *,
    mixing_mac: float = math.nan,
    shift_bound: float = math.nan,
) -> ErrorEstimate:
    """Package a predicted shift as an :class:`ErrorEstimate`."""
    frequency, damping = _relative_errors(reference, reference + shift)
    return ErrorEstimate(
        shift=complex(shift),
        frequency_error_pct=frequency,
        damping_error_pct=damping,
        mac=float(mac),
        hybridised=bool(math.isfinite(mac) and mac < mac_threshold),
        method=method,
        mixing_mac=float(mixing_mac),
        mixed=bool(math.isfinite(mixing_mac) and mixing_mac < mac_threshold),
        shift_bound=float(shift_bound),
    )


def _perturbative(
    basis: EigenBasis,
    projected: npt.NDArray[np.complex128],
    picks: Sequence[int],
    order: int,
) -> list[tuple[complex, float]]:
    """First/second-order eigenvalue shifts and first-order eigenvector MACs."""
    values = basis.values
    out: list[tuple[complex, float]] = []
    for j in picks:
        detuning = values[j] - values
        detuning[j] = np.inf
        detuning = np.where(detuning == 0.0, np.inf, detuning)
        shift = complex(projected[j, j])
        if order >= 2:
            shift += complex(np.sum(projected[j, :] * projected[:, j] / detuning))
        coefficient = projected[:, j] / detuning
        coefficient[j] = 0.0
        vector = basis.right[:, j] + basis.right @ coefficient
        mac = float(_mac(basis.right[:, j], vector[:, None])[0])
        out.append((shift, mac))
    return out


def _azimuth_subspace(
    basis: EigenBasis, azimuth: AzimuthStates, share: float
) -> npt.NDArray[np.int_]:
    """Return the modes with appreciable azimuth plus rotor-speed participation."""
    part = basis.participation()
    weight = part[azimuth.position].copy()
    if azimuth.speed is not None:
        weight += part[azimuth.speed]
    total = np.maximum(part.sum(axis=0), np.finfo(np.float64).tiny)
    return np.asarray(np.flatnonzero(weight / total > share), dtype=np.int_)


def _structured(
    basis: EigenBasis,
    error: npt.NDArray[np.float64],
    picks: Sequence[int],
    azimuth: AzimuthStates,
    share: float,
) -> list[tuple[complex, float]]:
    """Azimuth-structured (rank-one plus near-Jordan subspace) prediction."""
    position = azimuth.position
    rest = error.copy()
    rest[:, position] = 0.0
    projected = basis.left.conj().T @ rest @ basis.right
    w = basis.left.conj().T @ error[:, position]
    z = basis.right[position, :]
    subspace = _azimuth_subspace(basis, azimuth, share)
    mask = np.zeros(projected.shape, dtype=bool)
    mask[subspace, :] = True
    mask[:, subspace] = True
    np.fill_diagonal(mask, val=True)
    reduced = np.diag(basis.values) + np.where(mask, projected, 0.0) + np.outer(w, z)
    values, coefficients = np.linalg.eig(reduced)
    vectors = basis.right @ coefficients
    out: list[tuple[complex, float]] = []
    for j in picks:
        reference = complex(basis.values[j])
        macs = _same_side(_mac(basis.right[:, j], vectors), values, reference)
        best = int(np.argmax(macs))
        out.append((complex(values[best]) - reference, float(macs[best])))
    return out


def _mixing(
    basis: EigenBasis,
    projected: npt.NDArray[np.complex128],
    picks: Sequence[int],
    coupling: float,
) -> list[float]:
    r"""Return the general mode-mixing MAC of each branch (cluster-exact solve).

    Two modes are linked when :math:`\sqrt{|P_{ik}P_{ki}|}` is at least ``coupling``
    times the detuning of their first-order shifted eigenvalues. The cluster of branch
    :math:`j` is every mode connected to it by a chain of links (so modes that move a
    linked mode are kept too); the eigenproblem of :math:`\boldsymbol\Lambda+\mathbf P`
    on the cluster is solved exactly and the MAC of the tracked eigenvector with the
    reference one is returned.
    """
    values = basis.values
    shifted = values + np.diag(projected)
    detuning = np.abs(shifted[:, None] - shifted[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        strength = np.sqrt(np.abs(projected * projected.T)) / detuning
    linked = np.nan_to_num(strength, nan=0.0, posinf=np.inf) >= coupling
    np.fill_diagonal(linked, val=False)
    out: list[float] = []
    for j in picks:
        member = np.zeros(values.size, dtype=bool)
        member[j] = True
        frontier = member.copy()
        while frontier.any():
            reached = linked[frontier].any(axis=0) & ~member
            member |= reached
            frontier = reached
        cluster = np.flatnonzero(member)
        if cluster.size == 1:
            out.append(1.0)
            continue
        reduced = np.diag(values[cluster]) + projected[np.ix_(cluster, cluster)]
        cluster_values, coefficients = np.linalg.eig(reduced)
        vectors = basis.right[:, cluster] @ coefficients
        reference = complex(values[j])
        macs = _same_side(_mac(basis.right[:, j], vectors), cluster_values, reference)
        out.append(float(np.max(macs)))
    return out


def predict_branch_shift(
    basis: EigenBasis,
    error: npt.ArrayLike,
    branches: Sequence[int],
    *,
    method: Literal["first", "second", "structured"] = "second",
    azimuth: AzimuthStates | int | None = None,
    subspace_share: float = _DEFAULT_SUBSPACE_SHARE,
    mac_threshold: float = _DEFAULT_MAC_THRESHOLD,
    mixing_coupling: float = _DEFAULT_MIXING_COUPLING,
) -> list[ErrorEstimate]:
    r"""Predict the change of reference branches caused by a matrix error.

    Parameters
    ----------
    basis : EigenBasis
        Eigentriples of the reference mean :math:`\mathbf A_0`.
    error : npt.ArrayLike
        Matrix error :math:`\mathbf E` (the perturbed model is
        :math:`\mathbf A_0+\mathbf E`).
    branches : Sequence[int]
        Mode indices into ``basis.values``.
    method : {"first", "second", "structured"}, optional
        ``"first"``: :math:`P_{jj}`; ``"second"`` (default): adds
        :math:`\sum_{k\neq j}P_{jk}P_{kj}/(\lambda_j-\lambda_k)`, with
        :math:`P=\mathbf U^H\mathbf E\mathbf V`; ``"structured"``: the azimuth column
        of :math:`\mathbf E` is kept as an exact rank-one term and the modes with
        azimuth or rotor-speed participation above ``subspace_share`` are coupled
        exactly; every other mode keeps its first-order shift. Required when the
        rotor-azimuth state is present with its delivered column, where the
        azimuth/speed pair is nearly defective and ordinary perturbation breaks down.
    azimuth : AzimuthStates or int or None, optional
        The rotor-azimuth states; required for ``"structured"``.
    subspace_share : float, optional
        Participation share above which a mode is retained exactly. Default 0.05.
    mac_threshold : float, optional
        Predicted MAC below which a branch is flagged hybridised or mixed. Default
        0.9.
    mixing_coupling : float, optional
        Coupling-to-detuning ratio above which a mode joins the branch's cluster in
        the general mode-mixing check. Default 0.005.

    Returns
    -------
    list[ErrorEstimate]
        One estimate per requested branch, in order. For the first- and second-order
        methods the MAC is that of the first-order eigenvector correction. Every
        estimate also carries the general mode-mixing MAC (``mixing_mac``,
        ``mixed``) and the first-order conditioning bound ``shift_bound``.

    Raises
    ------
    ValueError
        If the error matrix does not match the basis, an index is out of range, the
        method is unknown, ``mixing_coupling`` is not positive, or ``"structured"`` is
        requested without ``azimuth``.

    Notes
    -----
    The hybridisation flag of ``"structured"`` sees only mixing through the azimuth
    subspace. The mixing check is independent of it: two modes are linked when their
    coupling :math:`\sqrt{|P_{ik}P_{ki}|}` exceeds ``mixing_coupling`` times their
    detuning, and every mode connected to the branch by a chain of links is coupled
    exactly, so two branches that alias into each other (for example two heavily
    damped flap modes on a three-azimuth grid) are caught as well. On a coarse grid of
    a non-normal model the cluster can approach the whole model, and the cost that of
    the trial eigenproblem. The conditioning bound uses the eigenvalue condition number
    :math:`\kappa_j=\|\mathbf u_j\|\|\mathbf v_j\|/|\mathbf u_j^H\mathbf v_j|`:
    :math:`|\delta\lambda_j|\le\kappa_j\|\mathbf E\|_2` to first order (the
    pseudospectral radius). It is rigorous to first order but on multi-scale models it
    is far from tight; the branch-level condition number is what
    :func:`estimate_averaging_error` tests.
    """
    n = basis.values.size
    matrix = np.asarray(error, dtype=np.float64)
    if matrix.shape != (n, n) or not np.all(np.isfinite(matrix)):
        msg = f"error must be a finite ({n}, {n}) matrix, got shape {matrix.shape}"
        raise ValueError(msg)
    picks = [int(j) for j in branches]
    if any(not 0 <= j < n for j in picks):
        msg = f"branch indices must lie in [0, {n})"
        raise ValueError(msg)
    if not math.isfinite(mixing_coupling) or mixing_coupling <= 0.0:
        msg = f"mixing_coupling must be positive and finite, got {mixing_coupling!r}"
        raise ValueError(msg)
    if method not in ("first", "second", "structured"):
        msg = f"method must be 'first', 'second' or 'structured', got {method!r}"
        raise ValueError(msg)
    projected = basis.left.conj().T @ matrix @ basis.right
    if method == "structured":
        if azimuth is None:
            msg = "the structured predictor needs the rotor-azimuth state"
            raise ValueError(msg)
        states = (
            azimuth
            if isinstance(azimuth, AzimuthStates)
            else AzimuthStates(_position(azimuth, n))
        )
        _position(states, n)
        predictions = _structured(basis, matrix, picks, states, subspace_share)
    else:
        predictions = _perturbative(
            basis, projected, picks, 1 if method == "first" else 2
        )
    mixing = _mixing(basis, projected, picks, mixing_coupling)
    norm = float(np.linalg.norm(matrix, 2))
    return [
        _estimate(
            complex(basis.values[j]),
            shift,
            mac,
            method,
            mac_threshold,
            mixing_mac=mix,
            shift_bound=float(basis.condition[j]) * norm,
        )
        for j, (shift, mac), mix in zip(picks, predictions, mixing, strict=True)
    ]


def trial_branch_shift(
    basis: EigenBasis,
    trial: npt.ArrayLike,
    branches: Sequence[int],
    *,
    mac_threshold: float = _DEFAULT_MAC_THRESHOLD,
) -> list[ErrorEstimate]:
    r"""Solve a trial eigenproblem and track the reference branches in it.

    Parameters
    ----------
    basis : EigenBasis
        Eigentriples of the reference mean :math:`\mathbf A_0`.
    trial : npt.ArrayLike
        The trial (perturbed) state matrix :math:`\mathbf A_0+\mathbf E`, for example
        a coarse sub-grid mean.
    branches : Sequence[int]
        Mode indices into ``basis.values``.
    mac_threshold : float, optional
        Tracking MAC below which a branch is flagged mixed (and hybridised). Default
        0.9.

    Returns
    -------
    list[ErrorEstimate]
        One ``"exact"`` estimate per branch: the eigenvalue of the trial matrix whose
        eigenvector has the largest MAC with the reference eigenvector (on the same
        side of the real axis), with that MAC as both ``mac`` and ``mixing_mac``.

    Raises
    ------
    ValueError
        If the trial matrix does not match the basis or an index is out of range.

    Notes
    -----
    This is the trial-eigenproblem form of the mode-mixing check: the exact answer for
    the given matrix, at the cost of one dense eigen-solution, and the reference for
    the predicted terms.
    """
    n = basis.values.size
    matrix = np.asarray(trial, dtype=np.float64)
    if matrix.shape != (n, n) or not np.all(np.isfinite(matrix)):
        msg = f"trial must be a finite ({n}, {n}) matrix, got shape {matrix.shape}"
        raise ValueError(msg)
    picks = [int(j) for j in branches]
    if any(not 0 <= j < n for j in picks):
        msg = f"branch indices must lie in [0, {n})"
        raise ValueError(msg)
    values, vectors = np.linalg.eig(matrix)
    out: list[ErrorEstimate] = []
    for j in picks:
        value = complex(basis.values[j])
        macs = _same_side(_mac(basis.right[:, j], vectors), values, value)
        best = int(np.argmax(macs))
        mac = float(macs[best])
        out.append(
            _estimate(
                value,
                complex(values[best]) - value,
                mac,
                "exact",
                mac_threshold,
                mixing_mac=mac,
            )
        )
    return out


# --------------------------------------------------------------------------------------
# Error estimate
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GridReport:
    r"""Quadrature properties of the delivered azimuth grid.

    Parameters
    ----------
    n_azimuths : int
        Number of delivered azimuths.
    max_gap_deviation_pct : float
        Largest deviation of a gap from the uniform spacing, percent.
    kernel_orders : npt.NDArray[np.int_]
        Orders ``1 .. n_azimuths - 1``.
    kernel_magnitude : npt.NDArray[np.float64]
        :math:`|W_{\mathrm{ref}}(h)|` of the interval weights at those orders: the
        fraction of each harmonic that leaks into the reference mean (zero to rounding
        on a uniform grid).
    """

    n_azimuths: int
    max_gap_deviation_pct: float
    kernel_orders: npt.NDArray[np.int_]
    kernel_magnitude: npt.NDArray[np.float64]


@dataclass(frozen=True)
class BranchErrorEstimate:
    r"""Error budget of one branch of the averaged model.

    Parameters
    ----------
    label : str
        Branch label.
    index : int
        Index of the branch in the eigenbasis of the reference mean.
    eigenvalue : complex
        Reference eigenvalue (1/s).
    frequency_hz : float
        Reference damped frequency :math:`|\operatorname{Im}\lambda|/2\pi`.
    damping_ratio : float
        Reference damping ratio.
    sampling : ErrorEstimate or None
        Predicted error of the requested azimuth grid, ``None`` if no grid was given.
    sampling_dominant_order : int or None
        Harmonic order contributing most to the first-order sampling error.
    quadrature : ErrorEstimate
        Arithmetic mean versus the interval-weighted mean on the delivered grid.
    periodicity : ErrorEstimate
        Second-order sideband (Floquet) estimate.
    hybridised : bool
        The sampling term (exact when available) or the quadrature term has
        hybridised.
    within_tolerance : bool
        The branch is well conditioned, its sampling term is resolved, every term is
        within the tolerances and none has hybridised or mixed.
    condition : float
        Eigenvalue condition number :math:`\kappa_j=1/|\hat{\mathbf u}_j^H
        \hat{\mathbf v}_j|` of the unit-norm left and right eigenvectors.
    well_conditioned : bool
        :math:`\kappa_j` does not exceed ``max_condition``. Above it the perturbative
        terms (sampling prediction, quadrature, periodicity) cannot be trusted, and
        the branch is reported as ill-conditioned.
    sampling_exact : ErrorEstimate or None
        The sampling error from the exact trial eigenproblem of the grid mean (the
        delivered samples for a measured sub-grid, the harmonic model otherwise),
        ``None`` when not computed.
    mixed : bool
        The general mode-mixing check flagged the sampling term (exact when
        available) or the quadrature term.
    sampling_resolved : bool
        The sampling term is a measurement (a sub-grid of the delivered samples) or a
        prediction whose first alias order the reference resolves. ``False`` means the
        term is only the resolved part of the error and the branch cannot be
        accepted for that grid (see :attr:`AveragingErrorEstimate.sampling_status`).
    """

    label: str
    index: int
    eigenvalue: complex
    frequency_hz: float
    damping_ratio: float
    sampling: ErrorEstimate | None
    sampling_dominant_order: int | None
    quadrature: ErrorEstimate
    periodicity: ErrorEstimate
    hybridised: bool
    within_tolerance: bool
    condition: float = math.nan
    well_conditioned: bool = True
    sampling_exact: ErrorEstimate | None = None
    mixed: bool = False
    sampling_resolved: bool = True

    @property
    def sampling_term(self) -> ErrorEstimate | None:
        """Return the sampling term used for the assessment: exact when available.

        Returns
        -------
        ErrorEstimate or None
            ``sampling_exact`` if computed, else ``sampling``.
        """
        return self.sampling_exact if self.sampling_exact is not None else self.sampling


@dataclass(frozen=True)
class AveragingErrorEstimate:
    r"""Per-branch error estimates of an azimuth-averaged model.

    Parameters
    ----------
    branches : dict[str, BranchErrorEstimate]
        The assessed branches by label.
    azimuth : AzimuthConsistency or None
        The azimuth consistency test, when the rotor-azimuth state was given.
    grid : GridReport
        Quadrature properties of the delivered grid.
    reference_mean : npt.NDArray[np.float64]
        The interval-weighted reference mean.
    rotor_speed : float
        Rotor speed (rad/s).
    method : str
        Branch predictor used for the sampling and quadrature terms.
    damping_tolerance_pct : float
        Damping-ratio tolerance, percent of the reference value.
    frequency_tolerance_pct : float
        Frequency tolerance, percent of the reference value.
    max_condition : float, optional
        Eigenvalue condition number above which a branch is ill-conditioned.
    sampling_status : str or None, optional
        ``"measured"`` (a sub-grid of the delivered samples), ``"resolved"`` (a
        predicted grid whose first alias order :math:`\operatorname{lcm}(N,B)` lies
        within the resolved orders of the reference), ``"unresolved: needs denser
        reference ..."`` otherwise, ``None`` without a grid.
    sampling_first_alias : int or None, optional
        :math:`\operatorname{lcm}(N,B)` of a predicted grid of :math:`N` azimuths.
    reference_resolved_order : int or None, optional
        The highest harmonic order the reference samples resolve.
    """

    branches: dict[str, BranchErrorEstimate]
    azimuth: AzimuthConsistency | None
    grid: GridReport
    reference_mean: npt.NDArray[np.float64]
    rotor_speed: float
    method: str
    damping_tolerance_pct: float
    frequency_tolerance_pct: float
    max_condition: float = _DEFAULT_MAX_CONDITION
    sampling_status: str | None = None
    sampling_first_alias: int | None = None
    reference_resolved_order: int | None = None

    @property
    def azimuth_assessment(self) -> str | None:
        """Return the graded azimuth assessment, ``None`` without the azimuth state.

        Returns
        -------
        str or None
            ``"consistent"``, ``"inconsistent-but-harmless"`` or
            ``"inconsistent-and-risky"``.
        """
        return None if self.azimuth is None else self.azimuth.assessment

    @property
    def within_tolerance(self) -> bool:
        """Return whether every branch is within tolerance and the azimuth is not risky.

        A harmless azimuth inconsistency (see
        :func:`~vane.modal.averaging.azimuth_consistency`) does not void the branch
        budget; a risky one does.
        """
        risky = "inconsistent-and-risky"
        safe = self.azimuth is None or self.azimuth.assessment != risky
        return safe and all(b.within_tolerance for b in self.branches.values())


def _resolve_branches(
    basis: EigenBasis, branches: Mapping[str, int | complex] | None
) -> dict[str, int]:
    """Map branch labels to mode indices of ``basis``."""
    values = basis.values
    if branches is None:
        oscillatory = np.flatnonzero(values.imag > 0.0)
        ordered = oscillatory[np.argsort(values.imag[oscillatory])]
        return {f"mode_{k:02d}": int(i) for k, i in enumerate(ordered)}
    resolved: dict[str, int] = {}
    for label, target in branches.items():
        if isinstance(target, numbers.Integral) and not isinstance(target, bool):
            index = int(target)
            if not 0 <= index < values.size:
                msg = f"branch {label!r}: index {index} is out of range"
                raise ValueError(msg)
        else:
            value = complex(target)
            side = values.imag >= 0.0 if value.imag >= 0.0 else values.imag <= 0.0
            candidates = np.flatnonzero(side)
            index = int(candidates[np.argmin(np.abs(values[candidates] - value))])
        resolved[str(label)] = index
    if not resolved:
        msg = "branches must name at least one mode"
        raise ValueError(msg)
    if len(set(resolved.values())) != len(resolved):
        msg = "two branch labels resolve to the same mode of the reference mean"
        raise ValueError(msg)
    return resolved


def _grid_report(
    angles: npt.NDArray[np.float64], weights: npt.NDArray[np.float64]
) -> GridReport:
    """Return the quadrature report of the delivered grid."""
    n = angles.size
    ordered = np.sort(angles)
    gaps = np.diff(np.append(ordered, ordered[0] + _TWO_PI))
    deviation = 100.0 * float(np.max(np.abs(gaps / (_TWO_PI / n) - 1.0)))
    orders = np.arange(1, max(n, 2), dtype=np.int_)
    kernel = np.abs(weight_kernel(angles, weights, orders))
    return GridReport(
        n_azimuths=n,
        max_gap_deviation_pct=deviation,
        kernel_orders=orders,
        kernel_magnitude=np.asarray(kernel, dtype=np.float64),
    )


def _sampling_error(
    model: HarmonicModel,
    angles: npt.NDArray[np.float64],
    stack: npt.NDArray[np.float64],
    grid: int | npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128], int, bool]:
    """Return the sampling error, kernel difference, grid size and measured flag."""
    if isinstance(grid, numbers.Integral) and not isinstance(grid, bool):
        grid_angles = uniform_azimuths(int(grid), float(np.min(angles)))
    else:
        grid_angles = _azimuth_array(grid)
    grid_weights = interval_weights(grid_angles)
    orders = model.orders
    delta = weight_kernel(grid_angles, grid_weights, orders) - weight_kernel(
        model.azimuths, model.reference_weights, orders
    )
    distance = np.abs(np.angle(np.exp(1j * (grid_angles[:, None] - angles[None, :]))))
    nearest = np.argmin(distance, axis=1)
    measured = bool(np.all(distance[np.arange(grid_angles.size), nearest] < _MATCH_TOL))
    if measured:
        weights = np.zeros(angles.size, dtype=np.float64)
        weights[nearest] = grid_weights
        error = np.tensordot(weights - model.reference_weights, stack, axes=(0, 0))
    else:
        error = grid_error_matrix(model, grid_angles, grid_weights)
    return (
        np.asarray(error, dtype=np.float64),
        np.asarray(delta, dtype=np.complex128),
        int(grid_angles.size),
        measured,
    )


def _dominant_orders(
    model: HarmonicModel,
    basis: EigenBasis,
    picks: Sequence[int],
    delta: npt.NDArray[np.complex128],
) -> list[int | None]:
    """Return the harmonic order dominating each branch's first-order sampling error."""
    left = basis.left[:, picks].conj()
    right = basis.right[:, picks]
    contributions: dict[int, npt.NDArray[np.complex128]] = {}
    for position, h in enumerate(model.orders):
        order = int(h)
        if order == 0 or delta[position] == 0.0:
            continue
        term = delta[position] * np.sum(left * (model.coefficients[order] @ right), 0)
        key = abs(order)
        contributions[key] = contributions[key] + term if key in contributions else term
    if not contributions:
        return [None] * len(picks)
    orders = sorted(contributions)
    table = np.abs(np.vstack([contributions[h] for h in orders]))
    return [
        int(orders[int(np.argmax(table[:, k]))]) if np.any(table[:, k] > 0.0) else None
        for k in range(len(picks))
    ]


def _positive_finite(value: float, name: str) -> float:
    """Return ``value`` as a positive finite float or raise ``ValueError``."""
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        msg = f"{name} must be positive and finite, got {value!r}"
        raise ValueError(msg)
    return number


def estimate_averaging_error(
    azimuths: npt.ArrayLike,
    matrices: npt.ArrayLike,
    rotor_speed: float,
    *,
    branches: Mapping[str, int | complex] | None = None,
    azimuth: AzimuthStates | None = None,
    grid: int | npt.ArrayLike | None = None,
    method: Literal["auto", "first", "second", "structured"] = "auto",
    damping_tolerance_pct: float = 1.0,
    frequency_tolerance_pct: float = 0.1,
    periodicity_order: int | None = None,
    consistency_tolerance: float = 1e-4,
    mac_threshold: float = _DEFAULT_MAC_THRESHOLD,
    subspace_share: float = _DEFAULT_SUBSPACE_SHARE,
    exact: bool = True,
    max_condition: float = _DEFAULT_MAX_CONDITION,
    mixing_coupling: float = _DEFAULT_MIXING_COUPLING,
    blades: int = 3,
) -> AveragingErrorEstimate:
    r"""Estimate the error of each branch of an azimuth-averaged model from the samples.

    Parameters
    ----------
    azimuths : npt.ArrayLike
        Delivered azimuths in radians (e.g. ``np.deg2rad(result.azimuths_deg)``).
    matrices : npt.ArrayLike
        Fixed-frame state matrices, shape ``(n_azimuth, n, n)`` (e.g.
        ``result.per_azimuth_a`` of an MBC result kept with
        ``retain_per_azimuth=True``).
    rotor_speed : float
        Rotor speed :math:`\Omega` in rad/s.
    branches : Mapping[str, int | complex] or None, optional
        Branches to assess: label to mode index of the reference mean or to an
        approximate eigenvalue (matched to the nearest one, on the same side of the
        real axis). Default every oscillatory mode (``mode_00``, ... by frequency).
    azimuth : AzimuthStates or None, optional
        The rotor-azimuth states (:func:`~vane.modal.averaging.locate_azimuth_state`).
        Enables the consistency test and the structured predictor.
    grid : int or npt.ArrayLike or None, optional
        The azimuth grid whose sampling error is predicted: explicit azimuths (a
        sub-grid of the delivered ones is measured exactly; any other grid is
        predicted from the interpolating harmonic model) or a count ``N`` for a
        hypothetical uniform grid starting at the first delivered azimuth. For the
        worst phase over start azimuths use :func:`azimuth_count_table`. ``None``
        skips the sampling term.
    method : {"auto", "first", "second", "structured"}, optional
        Branch predictor for the sampling and quadrature terms. ``"auto"`` uses
        ``"structured"`` when ``azimuth`` is given and ``"second"`` otherwise.
    damping_tolerance_pct, frequency_tolerance_pct : float, optional
        Tolerances (percent of the reference value) for ``within_tolerance``. The
        damping tolerance, as a fraction, is also the scale tolerance of the azimuth
        consistency test.
    periodicity_order : int or None, optional
        Highest harmonic order of the sideband estimate; default ``(M - 1) // 2``.
    consistency_tolerance : float, optional
        Absolute tolerance on the azimuth residual exponent, 1/s, reported as
        ``azimuth.absolute_consistent``; the assessment uses the scaled test.
    mac_threshold : float, optional
        MAC below which a branch is flagged hybridised or mixed.
    subspace_share : float, optional
        Azimuth/speed participation share retained exactly by the structured
        predictor.
    exact : bool, optional
        Also solve the trial eigenproblem of the grid mean (one dense eigen-solution)
        for the exact sampling term and the trial form of the mode-mixing check.
        Default ``True``.
    max_condition : float, optional
        Eigenvalue condition number above which a branch is ill-conditioned. Default
        ``1e3``.
    mixing_coupling : float, optional
        Coupling-to-detuning ratio of the predicted mode-mixing check (see
        :func:`predict_branch_shift`).
    blades : int, optional
        Blade count :math:`B` of the :math:`\operatorname{lcm}(N,B)` aliasing rule that
        decides whether a predicted grid is resolved. Default 3; use 1 for an
        anisotropic rotor.

    Returns
    -------
    AveragingErrorEstimate
        The per-branch budget, the azimuth consistency test and the grid report.

    Raises
    ------
    ValueError
        If the inputs are inconsistent or a tolerance is not positive.

    Notes
    -----
    The sampling term of the analyst's own full grid cannot be measured from that
    grid; assess interleaved sub-grids (``grid=azimuths[::2]``) or extrapolate the
    harmonic model with :func:`azimuth_count_table`. The quadrature and periodicity
    terms are fully a priori.

    A grid that is not a sub-grid of the delivered samples is predicted from the
    interpolating harmonic model, which holds the orders up to the Nyquist order of the
    reference and nothing above. An :math:`N`-azimuth grid of a :math:`B`-bladed rotor
    is first aliased at :math:`\operatorname{lcm}(N,B)`. When that order lies above the
    resolved orders of the reference, the predicted term is only the resolved part of
    the error: the branch is reported with ``sampling_resolved=False`` and is not
    accepted, and ``sampling_status`` asks for a denser reference. On OpenFAST
    linearizations of two floating turbines, predictions from a 36-azimuth reference
    were exact for 12 azimuths (first alias 12) but recovered only 7 to 133 % of the
    error of 8, 10, 13 and 18 azimuths (first aliases 24 to 39 and the order-18 sine),
    while the same estimate built from a 72-azimuth reference recovered them all.

    The predicted terms are perturbative and assume a well-conditioned branch: the
    first-order sensitivity of :math:`\lambda_j` to a matrix error is
    :math:`\kappa_j` (the pseudospectral bound
    :math:`|\delta\lambda_j|\le\kappa_j\|\mathbf E\|_2`, reported as
    ``shift_bound``). A branch with :math:`\kappa_j>` ``max_condition`` is reported
    as ill-conditioned; with ``exact=True`` its sampling term is still exact, but its
    quadrature and periodicity terms are not trusted. On the IEA 22 MW
    semi-submersible at 8 m/s the platform sway branch (:math:`\kappa\approx1.2\times
    10^3`) is under-predicted six- to eight-fold by every predictor on some
    sub-grids, which this check flags and the exact term measures.
    """
    omega = _positive_rate(rotor_speed)
    _positive_finite(damping_tolerance_pct, "damping_tolerance_pct")
    _positive_finite(frequency_tolerance_pct, "frequency_tolerance_pct")
    _positive_finite(max_condition, "max_condition")
    b = _positive_int(blades, "blades")
    angles = _azimuth_array(azimuths)
    stack = _matrix_stack(matrices, angles.size)
    weights = interval_weights(angles)
    reference = np.asarray(np.tensordot(weights, stack, axes=(0, 0)), dtype=np.float64)
    basis = eigen_basis(reference)
    labels = _resolve_branches(basis, branches)
    picks = list(labels.values())
    if method not in _METHODS:
        msg = f"method must be one of {sorted(_METHODS)}, got {method!r}"
        raise ValueError(msg)
    chosen = _METHODS[method]
    if method == "auto":
        chosen = "structured" if azimuth is not None else "second"

    def predict(error: npt.NDArray[np.float64]) -> list[ErrorEstimate]:
        return predict_branch_shift(
            basis,
            error,
            picks,
            method=chosen,
            azimuth=azimuth,
            subspace_share=subspace_share,
            mac_threshold=mac_threshold,
            mixing_coupling=mixing_coupling,
        )

    quadrature = predict(np.asarray(stack.mean(axis=0) - reference, dtype=np.float64))
    sampling: list[ErrorEstimate | None] = [None] * len(picks)
    exact_terms: list[ErrorEstimate | None] = [None] * len(picks)
    dominant: list[int | None] = [None] * len(picks)
    status: str | None = None
    first_alias: int | None = None
    resolved_order: int | None = None
    resolved = True
    if grid is not None:
        model = fit_harmonics(angles, stack)
        resolved_order = model.resolved_order
        error, delta, count, measured = _sampling_error(model, angles, stack, grid)
        if measured:
            status = "measured"
        else:
            first_alias = math.lcm(count, b)
            resolved = first_alias <= resolved_order
            status = (
                "resolved"
                if resolved
                else (
                    "unresolved: needs denser reference (first alias "
                    f"lcm({count}, {b}) = {first_alias} above the resolved order "
                    f"{resolved_order} of {angles.size} reference azimuths)"
                )
            )
        sampling = list(predict(error))
        dominant = _dominant_orders(model, basis, picks, delta)
        if exact:
            exact_terms = list(
                trial_branch_shift(
                    basis, reference + error, picks, mac_threshold=mac_threshold
                )
            )
    periodic_model = fit_harmonics(
        angles, stack, method="piecewise_linear", max_order=periodicity_order
    )
    corrections = _sideband_all(periodic_model, basis, picks, omega, None)
    consistency = (
        azimuth_consistency(
            periodic_model,
            omega,
            azimuth,
            tolerance=consistency_tolerance,
            basis=basis,
            branches=picks,
            scale_tolerance=damping_tolerance_pct / 100.0,
        )
        if azimuth is not None
        else None
    )
    estimates: dict[str, BranchErrorEstimate] = {}
    for k, (label, j) in enumerate(labels.items()):
        value = complex(basis.values[j])
        periodicity = _estimate(value, corrections[k], math.nan, "sideband", 0.0)
        sampling_term = exact_terms[k] if exact_terms[k] is not None else sampling[k]
        terms = [
            t for t in (sampling_term, quadrature[k], periodicity) if t is not None
        ]
        condition = float(basis.condition[j])
        well_conditioned = condition <= max_condition
        estimates[label] = BranchErrorEstimate(
            label=label,
            index=j,
            eigenvalue=value,
            frequency_hz=abs(value.imag) / _TWO_PI,
            damping_ratio=-value.real / abs(value) if value != 0 else 0.0,
            sampling=sampling[k],
            sampling_dominant_order=dominant[k],
            quadrature=quadrature[k],
            periodicity=periodicity,
            hybridised=any(t.hybridised for t in terms),
            within_tolerance=well_conditioned
            and resolved
            and all(
                t.within(damping_tolerance_pct, frequency_tolerance_pct) for t in terms
            ),
            condition=condition,
            well_conditioned=well_conditioned,
            sampling_exact=exact_terms[k],
            mixed=any(t.mixed for t in terms),
            sampling_resolved=resolved,
        )
    return AveragingErrorEstimate(
        branches=estimates,
        azimuth=consistency,
        grid=_grid_report(angles, weights),
        reference_mean=reference,
        rotor_speed=omega,
        method=chosen,
        damping_tolerance_pct=float(damping_tolerance_pct),
        frequency_tolerance_pct=float(frequency_tolerance_pct),
        max_condition=float(max_condition),
        sampling_status=status,
        sampling_first_alias=first_alias,
        reference_resolved_order=resolved_order,
    )


# --------------------------------------------------------------------------------------
# Azimuth-count design
# --------------------------------------------------------------------------------------


def _positive_int(value: object, name: str) -> int:
    """Return ``value`` as a positive integer or raise ``ValueError``."""
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Integral)
        or int(value) < 1
    ):
        msg = f"{name} must be a positive integer, got {value!r}"
        raise ValueError(msg)
    return int(value)


def first_alias_order(count: int, blades: int) -> int:
    r"""Return the lowest harmonic order that aliases into a uniform ``count``-mean.

    Parameters
    ----------
    count : int
        Number of uniform azimuths :math:`N`.
    blades : int
        Blade count :math:`B` (use 1 for content at every order).

    Returns
    -------
    int
        :math:`\operatorname{lcm}(N, B)`. When the periodic content after MBC lies only
        at multiples of :math:`B` (an isotropic rotor), the :math:`N`-sample mean is
        aliased only by common multiples of :math:`N` and :math:`B`, so an
        :math:`N` that is not a multiple of :math:`B` pushes the first alias from
        :math:`N` to :math:`BN`.

    Raises
    ------
    ValueError
        If either argument is not a positive integer.
    """
    return math.lcm(_positive_int(count, "count"), _positive_int(blades, "blades"))


def alias_free_azimuth_count(blades: int, highest_order: int) -> int:
    r"""Return the smallest uniform azimuth count free of aliases up to an order.

    Parameters
    ----------
    blades : int
        Blade count :math:`B` (1 when the content is at every order).
    highest_order : int
        Highest harmonic order with significant content.

    Returns
    -------
    int
        The smallest :math:`N` with :math:`\operatorname{lcm}(N, B)` greater than
        ``highest_order``.

    Raises
    ------
    ValueError
        If ``blades`` or ``highest_order`` is not a positive integer.
    """
    b = _positive_int(blades, "blades")
    order = _positive_int(highest_order, "highest_order")
    count = 1
    while math.lcm(count, b) <= order:
        count += 1
    return count


@dataclass(frozen=True)
class AzimuthCountRow:
    r"""Predicted averaging error of one uniform azimuth count.

    Parameters
    ----------
    count : int
        Number of uniform azimuths :math:`N`.
    first_alias_order : int
        :math:`\operatorname{lcm}(N, B)`.
    multiple_of_blades : bool
        :math:`B \mid N`.
    resolved_damping_error_pct : float
        Worst damping-ratio error over branches and start azimuths from the resolved
        harmonics (exact eigenvalues of the model's uniform mean).
    resolved_frequency_error_pct : float
        The same for the damped frequency.
    unresolved_damping_bound_pct : float
        Worst first-order, worst-phase estimate of the aliases above the model's
        highest order, from the documented tail model (``nan`` when a branch's tail
        is unresolved or extrapolation is off). It is a model-based estimate, not a
        bound.
    total_damping_bound_pct : float
        Worst over branches of resolved plus tail estimate (``nan`` when any tail
        is unresolved).
    min_tracking_mac : float
        Lowest MAC with which a branch was tracked.
    hybridised : bool
        Some branch was tracked with a MAC below the hybridisation threshold.
    within_tolerance : bool
        ``total_damping_bound_pct`` is finite and within the tolerance: the
        *extrapolated* decision.
    resolved_damping_by_branch : dict[str, float]
        Worst resolved damping-ratio error per branch.
    unresolved_damping_by_branch : dict[str, float]
        Tail estimate per branch (``nan`` when unresolved).
    alias_resolved : bool, optional
        :math:`\operatorname{lcm}(N,B)` lies within the orders the model resolves, so
        the resolved error contains the first alias of the count.
    resolved_within_tolerance : bool, optional
        ``alias_resolved``, not hybridised and the resolved error within the
        tolerance: the decision supported by the samples.
    """

    count: int
    first_alias_order: int
    multiple_of_blades: bool
    resolved_damping_error_pct: float
    resolved_frequency_error_pct: float
    unresolved_damping_bound_pct: float
    total_damping_bound_pct: float
    min_tracking_mac: float
    hybridised: bool
    within_tolerance: bool
    resolved_damping_by_branch: dict[str, float]
    unresolved_damping_by_branch: dict[str, float]
    alias_resolved: bool = True
    resolved_within_tolerance: bool = False


@dataclass(frozen=True)
class AzimuthCountTable:
    """Azimuth-count design table and its two recommendations.

    Parameters
    ----------
    rows : tuple[AzimuthCountRow, ...]
        One row per evaluated count, ascending.
    recommended : int or None
        The recommendation supported by the samples, equal to
        ``recommended_resolved``.
    blades : int
        Blade count used for the lcm rule.
    damping_tolerance_pct : float
        The damping-ratio tolerance, percent.
    extrapolated : bool
        Whether a tail estimate was formed for at least one branch.
    recommended_resolved : int or None, optional
        The smallest count whose first alias the model resolves and whose resolved
        error meets the tolerance. The resolved error is exact for the content the
        samples resolve, so this is the recommended count.
    recommended_extrapolated : int or None, optional
        The smallest count whose resolved error plus tail estimate meets the
        tolerance. A model-based estimate, not a bound; ``None`` when a tail is
        unresolved.
    extrapolation_status : str, optional
        ``"extrapolated"``, ``"not requested"`` or ``"unresolved: needs denser
        linearization ..."`` naming the branches whose tail cannot be modelled.
    tails : dict[str, AliasTail], optional
        The tail model of each branch.
    resolved_order : int, optional
        The highest harmonic order the model resolves.
    """

    rows: tuple[AzimuthCountRow, ...]
    recommended: int | None
    blades: int
    damping_tolerance_pct: float
    extrapolated: bool
    recommended_resolved: int | None = None
    recommended_extrapolated: int | None = None
    extrapolation_status: str = "not requested"
    tails: dict[str, AliasTail] = field(default_factory=dict)
    resolved_order: int = 0

    def row(self, count: int) -> AzimuthCountRow:
        """Return the row of one azimuth count.

        Parameters
        ----------
        count : int
            An evaluated azimuth count.

        Returns
        -------
        AzimuthCountRow
            Its row.

        Raises
        ------
        KeyError
            If ``count`` was not evaluated.
        """
        for item in self.rows:
            if item.count == count:
                return item
        raise KeyError(count)


def _sensitivities(
    model: HarmonicModel, basis: EigenBasis, j: int, blades: int
) -> tuple[list[int], npt.NDArray[np.float64]]:
    """Return ``s_j(h)`` over the resolved multiples of the blade count."""
    orders = [
        h
        for h in range(blades, model.resolved_order + 1, blades)
        if h in model.coefficients and -h in model.coefficients
    ]
    u, v = basis.left[:, j], basis.right[:, j]
    values = np.array(
        [
            abs(u.conj() @ model.coefficients[h] @ v)
            + abs(u.conj() @ model.coefficients[-h] @ v)
            for h in orders
        ],
        dtype=np.float64,
    )
    return orders, values


@dataclass(frozen=True)
class AliasTail:
    r"""Model of one branch's harmonic sensitivity above the resolved orders.

    Parameters
    ----------
    model : str
        ``"own decay"`` (the branch's own sensitivity peaks inside the resolved
        multiples of :math:`B` and its fitted trend at least halves after),
        ``"tower law"`` (it does not, and the tower-passage decay of the azimuth
        column is used), ``"no content"`` (the sensitivity is rounding, the tail is
        zero) or ``"unresolved"`` (no tail can be justified from the samples).
    own_decay : float
        Fitted decay per order of the branch's own sensitivity (``nan`` if unused).
    tower_decay : float
        Tower-passage decay per order (``nan`` if unused).
    own_level : float
        Level of the smallest own-decay envelope through every fitted point, at
        ``anchor_order`` (1/s).
    peak_level : float
        Largest observed sensitivity :math:`\max_h s_j(h)` (1/s), the anchor of the
        tower-law envelope.
    anchor_order : int
        The highest resolved multiple of :math:`B`.
    peak_order : int
        The resolved order of the largest observed sensitivity.
    """

    model: str
    own_decay: float
    tower_decay: float
    own_level: float
    peak_level: float
    anchor_order: int
    peak_order: int

    def envelope(self, order: int) -> float:
        r"""Return the tail estimate of :math:`s_j(h)` at an order above the anchor.

        Parameters
        ----------
        order : int
            Harmonic order :math:`h`.

        Returns
        -------
        float
            The envelope of the tail model, ``nan`` when unresolved.
        """
        values: list[float] = []
        lift = order - self.anchor_order
        if self.model == "no content":
            return 0.0
        if math.isfinite(self.own_decay):
            values.append(self.own_level * self.own_decay**lift)
        elif math.isfinite(self.tower_decay):
            ratio = order / self.anchor_order
            values.append(self.peak_level * ratio * self.tower_decay**lift)
        return max(values) if values else math.nan


def _alias_tail(
    orders: list[int],
    sensitivity: npt.NDArray[np.float64],
    blades: int,
    tower: float | None,
    scale: float,
) -> AliasTail:
    """Choose the tail model of one branch (see :func:`azimuth_count_table`)."""
    if not orders:
        return AliasTail("unresolved", math.nan, math.nan, math.nan, math.nan, 0, 0)
    top = orders[-1]
    peak = orders[int(np.argmax(sensitivity))]
    peak_level = float(np.max(sensitivity))
    if peak_level <= _NO_CONTENT * scale:  # rounding only: nothing to extrapolate
        return AliasTail("no content", math.nan, math.nan, 0.0, peak_level, top, peak)
    own, own_level = math.nan, math.nan
    fitted = [k for k, h in enumerate(orders) if h >= 2 * blades]
    if len(fitted) >= 2 and peak < top:  # a line needs two points; the peak is seen
        h_arr = np.asarray([orders[k] for k in fitted], dtype=np.float64)
        s_arr = np.maximum(sensitivity[fitted], np.finfo(np.float64).tiny)
        slope = float(np.polyfit(h_arr, np.log(s_arr), 1)[0])
        # a decay is observed only if the fitted trend at least halves the sensitivity
        # over the fitted orders; a flatter trend is indistinguishable from no decay
        if slope < 0.0 and slope * (h_arr[-1] - h_arr[0]) <= math.log(_MIN_OWN_DROP):
            own = math.exp(slope)
            own_level = float(np.max(s_arr * own ** (top - h_arr)))
    if math.isfinite(own):  # the branch's own, observed decay comes first
        return AliasTail("own decay", own, math.nan, own_level, peak_level, top, peak)
    if tower is not None and 0.0 < tower < 1.0:
        return AliasTail("tower law", math.nan, tower, math.nan, peak_level, top, peak)
    return AliasTail("unresolved", math.nan, math.nan, math.nan, peak_level, top, peak)


def _tail_sum(tail: AliasTail, lcm: int, highest: int, limit: int | None) -> float:
    """Sum the tail envelope over the aliases ``m * lcm`` above ``highest``."""
    if tail.model == "unresolved":
        return math.nan
    total = 0.0
    h = lcm * max(1, highest // lcm + 1)
    while limit is None or h <= limit:
        term = tail.envelope(h)
        total += term
        if term <= _TAIL_RTOL * total or h > _TAIL_MAX_ORDER:
            break
        h += lcm
    return total


def _tower_rate(tail_decay: float | TowerPassageFit | None) -> float | None:
    """Return the tower-passage decay per order, or ``None``."""
    if tail_decay is None:
        return None
    if isinstance(tail_decay, TowerPassageFit):
        return tail_decay.decay if tail_decay.log_r2 >= _MIN_TOWER_R2 else None
    rate = float(tail_decay)
    if not math.isfinite(rate) or not 0.0 < rate < 1.0:
        msg = f"tail_decay must lie in (0, 1), got {tail_decay!r}"
        raise ValueError(msg)
    return rate


def azimuth_count_table(
    model: HarmonicModel,
    branches: Mapping[str, int | complex] | None = None,
    *,
    blades: int = 3,
    counts: Iterable[int] = range(5, 37),
    starts: int = 24,
    damping_tolerance_pct: float = 1.0,
    extrapolate: bool = True,
    max_alias_order: int | None = None,
    mac_threshold: float = _DEFAULT_MAC_THRESHOLD,
    tail_decay: float | TowerPassageFit | None = None,
) -> AzimuthCountTable:
    r"""Predict the averaging error of uniform azimuth counts and recommend one.

    Parameters
    ----------
    model : HarmonicModel
        Harmonic model of a dense set of samples (the interpolating trigonometric fit
        of :func:`~vane.modal.averaging.fit_harmonics`). Branch errors are measured
        against its zeroth harmonic.
    branches : Mapping[str, int | complex] or None, optional
        Branches to assess (as for :func:`estimate_averaging_error`, resolved against
        the eigenbasis of ``model.mean``). Default every oscillatory mode.
    blades : int, optional
        Blade count :math:`B` for the :math:`\operatorname{lcm}(N, B)` rule. Default 3.
    counts : Iterable[int], optional
        Uniform azimuth counts to evaluate. Default 5 to 36.
    starts : int, optional
        Start azimuths per count, spread over one grid spacing; the worst is kept.
    damping_tolerance_pct : float, optional
        Damping-ratio tolerance, percent of the reference value.
    extrapolate : bool, optional
        Form the tail estimate of the aliases above the model's highest order.
    max_alias_order : int or None, optional
        Highest alias order summed in the tail estimate; default ``None`` sums the
        geometric envelope to convergence, so the estimate does not depend on it.
    mac_threshold : float, optional
        Tracking MAC below which a count is flagged hybridised.
    tail_decay : float or TowerPassageFit or None, optional
        Tower-passage decay per order for branches whose own sensitivity does not
        decay within the resolved orders, either a number in :math:`(0,1)` or the
        fit of :func:`~vane.modal.averaging.tower_passage_decay` to the **delivered**
        samples (it is ignored when its :math:`R^2` in log is below 0.5). ``None``
        leaves such branches unresolved.

    Returns
    -------
    AzimuthCountTable
        One row per count, the resolved and the extrapolated recommendation, the
        tail model of every branch and the extrapolation status.

    Raises
    ------
    ValueError
        If an argument is invalid or the model is not of square matrices.

    Notes
    -----
    The **resolved** error is exact for the content the samples resolve: the exact
    eigenvalues of the uniform :math:`N`-mean of the fitted model, worst over start
    azimuths. It contains the first alias of :math:`N` only when
    :math:`\operatorname{lcm}(N,B)` lies within the resolved orders
    (``alias_resolved``); only such counts are accepted, and
    ``recommended_resolved`` is the smallest of them meeting the tolerance. A
    36-azimuth reference therefore confirms 5, 6, 9, 12 and 15 but not 7, 8, 10 or
    13, whose first aliases (21 to 39) it cannot observe: for those the resolved error
    is only a lower estimate, and a denser reference is needed to confirm them.

    The **tail** estimate is first order and worst phase,
    :math:`\sum_{m}\hat s_j(m\,\operatorname{lcm}(N,B))/\sigma_j` over the aliases above
    the model's highest order, with an envelope :math:`\hat s_j` of the sensitivity
    :math:`s_j(h)=|\mathbf u_j^H\mathbf A_h\mathbf v_j|+|\mathbf u_j^H\mathbf A_{-h}
    \mathbf v_j|`:

    * own decay: when :math:`s_j` peaks inside the resolved multiples of :math:`B` and
      its log-linear fit from :math:`2B` at least halves over the fitted orders, the
      smallest geometric envelope of that decay through every fitted point;
    * tower law: otherwise, :math:`\max_h s_j(h)\,(h/H)\,\rho^{h-H}` from the highest
      resolved multiple :math:`H`, with the tower-passage decay :math:`\rho` of the
      azimuth column (``tail_decay``), the law of the frozen-azimuth derivative of a
      localised inflow deficit;
    * neither: the branch is ``"unresolved"``, its tail and every total are ``nan``,
      and the table reports ``"unresolved: needs denser linearization"`` instead of a
      recommendation.

    Both envelopes are summable, so the estimate is independent of
    ``max_alias_order``. It is a model-based extrapolation of orders the samples cannot
    observe, **not a bound**: on new OpenFAST linearizations of two floating turbines
    this estimate from 36 azimuths was exceeded at 13 azimuths (and, on the delivered
    column, at 18) and was up to 30 times conservative elsewhere.
    ``recommended_extrapolated``
    is therefore an estimate, and ``recommended`` is the resolved count. With the
    delivered azimuth column the first-order tail is not valid (the near-Jordan
    azimuth/speed pair); evaluate the table on a pinned or neutralised model.
    """
    b = _positive_int(blades, "blades")
    n_starts = _positive_int(starts, "starts")
    alias_limit = (
        None
        if max_alias_order is None
        else _positive_int(max_alias_order, "max_alias_order")
    )
    if not math.isfinite(damping_tolerance_pct) or damping_tolerance_pct <= 0.0:
        msg = f"damping_tolerance_pct must be positive, got {damping_tolerance_pct!r}"
        raise ValueError(msg)
    grid_counts = sorted({_positive_int(c, "counts") for c in counts})
    if not grid_counts:
        msg = "counts must contain at least one azimuth count"
        raise ValueError(msg)
    tower = _tower_rate(tail_decay)
    basis = eigen_basis(model.mean)
    labels = _resolve_branches(basis, branches)
    tails: dict[str, AliasTail] = {}
    scale = max(float(np.linalg.norm(model.mean)), np.finfo(np.float64).tiny)
    if extrapolate:
        for label, j in labels.items():
            orders, sensitivity = _sensitivities(model, basis, j, b)
            tails[label] = _alias_tail(orders, sensitivity, b, tower, scale)
    resolved_order = model.resolved_order
    rows: list[AzimuthCountRow] = []
    for count in grid_counts:
        worst_damping = dict.fromkeys(labels, 0.0)
        worst_frequency = dict.fromkeys(labels, 0.0)
        min_mac = 1.0
        for start in np.arange(n_starts) * (_TWO_PI / count) / n_starts:
            values, vectors = np.linalg.eig(model.uniform_mean(count, float(start)))
            for label, j in labels.items():
                reference = complex(basis.values[j])
                macs = _same_side(_mac(basis.right[:, j], vectors), values, reference)
                best = int(np.argmax(macs))
                frequency, damping = _relative_errors(reference, complex(values[best]))
                worst_damping[label] = max(worst_damping[label], damping)
                worst_frequency[label] = max(worst_frequency[label], frequency)
                min_mac = min(min_mac, float(macs[best]))
        lcm = math.lcm(count, b)
        unresolved: dict[str, float] = {}
        for label, j in labels.items():
            if label not in tails:
                unresolved[label] = math.nan
                continue
            bound = _tail_sum(tails[label], lcm, model.max_order, alias_limit)
            sigma = abs(float(basis.values[j].real))
            unresolved[label] = 100.0 * bound / sigma if sigma > 0.0 else math.nan
        totals = [
            worst_damping[label] + (unresolved[label] if extrapolate else 0.0)
            for label in labels
        ]
        total = max(totals) if all(math.isfinite(t) for t in totals) else math.nan
        finite_unresolved = [v for v in unresolved.values() if math.isfinite(v)]
        resolved_damping = max(worst_damping.values())
        hybrid = min_mac < mac_threshold
        alias_resolved = lcm <= resolved_order
        rows.append(
            AzimuthCountRow(
                count=count,
                first_alias_order=lcm,
                multiple_of_blades=count % b == 0,
                resolved_damping_error_pct=resolved_damping,
                resolved_frequency_error_pct=max(worst_frequency.values()),
                unresolved_damping_bound_pct=(
                    max(finite_unresolved)
                    if finite_unresolved and len(finite_unresolved) == len(labels)
                    else math.nan
                ),
                total_damping_bound_pct=total,
                min_tracking_mac=min_mac,
                hybridised=hybrid,
                within_tolerance=bool(
                    math.isfinite(total) and total <= damping_tolerance_pct
                ),
                resolved_damping_by_branch=worst_damping,
                unresolved_damping_by_branch=unresolved,
                alias_resolved=alias_resolved,
                resolved_within_tolerance=bool(
                    alias_resolved
                    and not hybrid
                    and resolved_damping <= damping_tolerance_pct
                ),
            )
        )
    accepted = [row.count for row in rows if row.resolved_within_tolerance]
    estimated = [row.count for row in rows if row.within_tolerance]
    unresolved_labels = [k for k, t in tails.items() if t.model == "unresolved"]
    if not extrapolate:
        status = "not requested"
    elif unresolved_labels:
        status = (
            "unresolved: needs denser linearization (no decaying sensitivity and no "
            f"tower-passage decay for {', '.join(unresolved_labels)})"
        )
    else:
        status = "extrapolated"
    recommended = min(accepted) if accepted else None

    return AzimuthCountTable(
        rows=tuple(rows),
        recommended=recommended,
        blades=b,
        damping_tolerance_pct=float(damping_tolerance_pct),
        extrapolated=bool(
            any(t.model in ("own decay", "tower law") for t in tails.values())
        ),
        recommended_resolved=recommended,
        recommended_extrapolated=(
            min(estimated)
            if extrapolate and estimated and not unresolved_labels
            else None
        ),
        extrapolation_status=status,
        tails=tails,
        resolved_order=resolved_order,
    )


# --------------------------------------------------------------------------------------
# Time-step commensurability
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CommensurateStep:
    """A time step that places every target azimuth on a whole number of steps.

    Parameters
    ----------
    time_step : float
        Suggested time step (s), never larger than the requested one.
    steps_per_azimuth : int
        Whole number of steps between consecutive target azimuths.
    requested_time_step : float
        The requested time step (s).
    relative_change : float
        ``(requested - suggested) / requested``.
    azimuth_spacing_deg : float
        Target azimuth spacing, ``360 / count`` degrees.
    azimuth_per_step_deg : float
        Rotor rotation per suggested step, degrees.
    """

    time_step: float
    steps_per_azimuth: int
    requested_time_step: float
    relative_change: float
    azimuth_spacing_deg: float
    azimuth_per_step_deg: float


def commensurate_time_step(
    rotor_speed: float, count: int, requested_time_step: float
) -> CommensurateStep:
    r"""Suggest a time step close to a requested one that is commensurate with the grid.

    Parameters
    ----------
    rotor_speed : float
        Rotor speed :math:`\Omega` in rad/s.
    count : int
        Number of uniform target azimuths per revolution (e.g. ``NLinTimes``).
    requested_time_step : float
        The time step the analyst would use (s).

    Returns
    -------
    CommensurateStep
        :math:`\Delta t=T/(Nk)` with :math:`T=2\pi/\Omega` and :math:`k` the smallest
        whole number of steps per azimuth interval for which
        :math:`\Delta t\le` ``requested_time_step``.

    Raises
    ------
    ValueError
        If an argument is not positive and finite.

    Notes
    -----
    A linearization snapshot is taken at the first time step at or past its target
    azimuth, so on a time step that does not divide the azimuth interval the
    delivered grid is non-uniform by up to :math:`\Omega\Delta t`. With a
    commensurate step and the rotor at the trimmed speed every snapshot falls on its
    target and the interval and arithmetic means coincide.
    """
    omega = _positive_rate(rotor_speed)
    n = _positive_int(count, "count")
    if not math.isfinite(requested_time_step) or requested_time_step <= 0.0:
        msg = f"requested_time_step must be positive, got {requested_time_step!r}"
        raise ValueError(msg)
    interval = _TWO_PI / (omega * n)
    steps = max(1, math.ceil(interval / requested_time_step * (1.0 - 1e-12)))
    time_step = interval / steps
    return CommensurateStep(
        time_step=time_step,
        steps_per_azimuth=steps,
        requested_time_step=float(requested_time_step),
        relative_change=(requested_time_step - time_step) / requested_time_step,
        azimuth_spacing_deg=360.0 / n,
        azimuth_per_step_deg=math.degrees(omega * time_step),
    )
