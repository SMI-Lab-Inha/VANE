# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Azimuth averaging on delivered grids: quadrature, harmonics and the azimuth state.

The azimuth-averaged (Coleman/MBC) model of a rotating turbine is a weighted mean of the
fixed-frame state matrices :math:`\mathbf A(\psi_\ell)` sampled at the azimuths a
linearization actually delivered. Those azimuths are rarely uniform: each snapshot is
taken at the first time step at or past its target azimuth. This module provides the
building blocks for treating that mean exactly and for auditing it.

* **Quadrature.** :func:`interval_weights` gives the weights of the periodic
  piecewise-linear interpolant of the samples (the exact mean of that interpolant),
  and :func:`weight_kernel` their discrete Fourier transform
  :math:`W(h)=\sum_\ell w_\ell e^{\,jh\psi_\ell}`, the aliasing kernel of the grid.
* **Harmonics.** :func:`fit_harmonics` represents the samples by a harmonic model
  :math:`\mathbf A(\psi)=\sum_h \mathbf A_h e^{\,jh\psi}`. With the default
  interpolating trigonometric fit the error of *any* weighted mean of the samples is
  exactly :math:`\mathbf E=\sum_h[W_S(h)-W_{\mathrm{ref}}(h)]\mathbf A_h`
  (:func:`grid_error_matrix`).
* **The rotor-azimuth state.** :func:`locate_azimuth_state` finds the ElastoDyn
  generator-azimuth position (``DOF_GeAz``) and rotor-speed states from the state
  descriptions. For a linearization about a periodic orbit the time derivative of that
  orbit (the orbit tangent, whose azimuth component is one) solves the linearized
  equations, so one Floquet exponent is exactly zero. :func:`azimuth_consistency`
  tests this from the averaged model alone: the averaged azimuth eigenvalue plus its
  second-order sideband correction (:func:`sideband_correction`) must vanish.
* **Azimuth treatments.** :func:`pin_azimuth` replaces each sample by
  :math:`\mathbf A(\psi)(\mathbf I-\boldsymbol\xi\mathbf e_{\mathrm{az}}^T)`, which
  makes the tangent :math:`\boldsymbol\xi` an exact null vector of every sample and,
  when :math:`\boldsymbol\xi` is the right eigenvector of the reference mean
  (:func:`azimuth_tangent`), leaves every other reference eigenvalue unchanged. The
  simpler :func:`zero_azimuth_column` and :func:`delete_azimuth_state` are kept for
  comparison; they also remove the zero-eigenvalue artefact but **bias the reference**,
  because the mean of the azimuth column is physical (the rotation of the
  gravity-sagged rotor, cancelled in an exact linearization by the cyclic blade
  coupling).

The per-branch error budget built on these pieces is in
:mod:`vane.modal.averaging_error_estimate`. Everything here is pure NumPy/SciPy and
takes azimuths in radians.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.linalg import eig

from vane.config.dof_map import DofCategory, classify_dof

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

__all__ = [
    "AzimuthConsistency",
    "AzimuthStates",
    "BranchAzimuthRisk",
    "EigenBasis",
    "HarmonicModel",
    "TowerPassageFit",
    "azimuth_consistency",
    "azimuth_mean",
    "azimuth_tangent",
    "delete_azimuth_state",
    "eigen_basis",
    "fit_harmonics",
    "grid_error_matrix",
    "interval_weights",
    "locate_azimuth_state",
    "pin_azimuth",
    "sideband_correction",
    "subgrid_weights",
    "tower_passage_decay",
    "uniform_azimuths",
    "weight_kernel",
    "zero_azimuth_column",
]

_TWO_PI = 2.0 * np.pi
# A gap larger than half a revolution leaves the interpolant without support over most
# of the rotor and the interval rule extrapolates; such a grid is rejected.
_MAX_GAP = np.pi * (1.0 + 1e-12)
# Below this |h * gap| the hat-function integrals use their Taylor series, avoiding the
# cancellation of the closed form; twelve terms reach machine precision there.
_SERIES_LIMIT = 0.1
_SERIES_TERMS = 12
# Condition number above which a trigonometric design is rejected as unresolvable
# (azimuths too clustered for the requested orders).
_MAX_DESIGN_CONDITION = 1e10
# The neutral direction of a coarse mean is already contaminated by aliasing (the
# tangent of a 6- to 12-sample mean gives no improvement); warn below this count.
_MIN_TANGENT_SAMPLES = 18
# Relative imaginary part above which the azimuth eigenvalue is reported as complex.
_COMPLEX_RTOL = 1e-9


# --------------------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------------------


def _azimuth_array(azimuths: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return azimuths wrapped into ``[0, 2pi)``, validated as distinct and finite."""
    angles = np.asarray(azimuths, dtype=np.float64)
    if angles.ndim != 1 or angles.size == 0:
        msg = f"azimuths must be a non-empty 1-D array, got shape {angles.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(angles)):
        msg = "azimuths must be finite"
        raise ValueError(msg)
    wrapped = np.mod(angles, _TWO_PI)
    ordered = np.sort(wrapped)
    if ordered.size > 1:
        gaps = np.diff(np.append(ordered, ordered[0] + _TWO_PI))
        if np.min(gaps) <= 0.0:
            msg = "azimuths must be distinct after wrapping into [0, 2pi)"
            raise ValueError(msg)
    return np.asarray(wrapped, dtype=np.float64)


def _matrix_stack(
    matrices: npt.ArrayLike, n_azimuths: int, *, square: bool = True
) -> npt.NDArray[np.float64]:
    """Return a finite ``(n_azimuths, ...)`` stack, optionally of square matrices."""
    stack = np.asarray(matrices, dtype=np.float64)
    if stack.ndim < 2 or stack.shape[0] != n_azimuths:
        msg = (
            f"matrices must be a stack with one entry per azimuth ({n_azimuths}), "
            f"got shape {stack.shape}"
        )
        raise ValueError(msg)
    if square and (stack.ndim != 3 or stack.shape[1] != stack.shape[2]):
        msg = f"matrices must be (n_azimuth, n, n), got {stack.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(stack)):
        msg = "matrices must be finite (no NaN or inf)"
        raise ValueError(msg)
    return stack


def _square_matrix(matrix: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    """Return a finite square matrix."""
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        msg = f"{name} must be a square matrix, got shape {array.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must be finite (no NaN or inf)"
        raise ValueError(msg)
    return array


def _positive_rate(rotor_speed: float) -> float:
    """Return a validated positive, finite rotor speed in rad/s."""
    value = float(rotor_speed)
    if not math.isfinite(value) or value <= 0.0:
        msg = f"rotor_speed must be positive and finite (rad/s), got {rotor_speed!r}"
        raise ValueError(msg)
    return value


# --------------------------------------------------------------------------------------
# Quadrature weights and the aliasing kernel
# --------------------------------------------------------------------------------------


def interval_weights(azimuths: npt.ArrayLike) -> npt.NDArray[np.float64]:
    r"""Return the interval (periodic trapezoidal) weights of an azimuth grid.

    Parameters
    ----------
    azimuths : npt.ArrayLike
        Delivered azimuths in radians, any order, distinct modulo :math:`2\pi`.

    Returns
    -------
    npt.NDArray[np.float64]
        One weight per azimuth, in the input order, summing to one. Applied to the
        samples they give the exact revolution mean of the periodic piecewise-linear
        interpolant.

    Raises
    ------
    ValueError
        If the azimuths are empty, non-finite, duplicated after wrapping, or leave a
        gap larger than half a revolution (a single azimuth always does).

    Notes
    -----
    With the azimuths sorted, :math:`g_\ell=\psi_{\ell+1}-\psi_\ell` the periodic gaps
    and :math:`g_0 \equiv g_M`,

    .. math::

       w_\ell = \frac{g_{\ell-1}+g_\ell}{4\pi},

    which reduces to :math:`1/M` on a uniform grid.
    """
    angles = _azimuth_array(azimuths)
    order = np.argsort(angles)
    ordered = angles[order]
    gaps = np.diff(np.append(ordered, ordered[0] + _TWO_PI))
    if float(np.max(gaps)) > _MAX_GAP:
        msg = (
            f"the azimuth grid leaves a gap of {np.rad2deg(np.max(gaps)):.1f} deg; the "
            "interval rule needs every gap to be at most half a revolution"
        )
        raise ValueError(msg)
    sorted_weights = 0.5 * (np.roll(gaps, 1) + gaps) / _TWO_PI
    weights: npt.NDArray[np.float64] = np.empty_like(sorted_weights)
    weights[order] = sorted_weights
    return weights


def subgrid_weights(
    azimuths: npt.ArrayLike,
    indices: Sequence[int] | npt.ArrayLike,
    *,
    rule: Literal["interval", "arithmetic"] = "interval",
) -> npt.NDArray[np.float64]:
    """Return full-length weights of a sub-grid of the delivered azimuths.

    Parameters
    ----------
    azimuths : npt.ArrayLike
        All delivered azimuths in radians.
    indices : Sequence[int] or npt.ArrayLike
        Positions (into ``azimuths``) of the samples that form the sub-grid.
    rule : {"interval", "arithmetic"}, optional
        ``"interval"`` weights the sub-grid by its own :func:`interval_weights`;
        ``"arithmetic"`` gives each of its samples ``1/len(indices)``.

    Returns
    -------
    npt.NDArray[np.float64]
        A weight vector aligned with ``azimuths``, zero off the sub-grid, summing to
        one.

    Raises
    ------
    ValueError
        If ``indices`` is empty, out of range or repeated, or ``rule`` is unknown, or
        the sub-grid is invalid for :func:`interval_weights`.
    """
    angles = _azimuth_array(azimuths)
    picks = np.asarray(indices, dtype=np.int_).ravel()
    if picks.size == 0 or np.unique(picks).size != picks.size:
        msg = "indices must be a non-empty set of distinct positions"
        raise ValueError(msg)
    if np.min(picks) < 0 or np.max(picks) >= angles.size:
        msg = f"indices must lie in [0, {angles.size})"
        raise ValueError(msg)
    weights = np.zeros(angles.size, dtype=np.float64)
    if rule == "interval":
        weights[picks] = interval_weights(angles[picks])
    elif rule == "arithmetic":
        weights[picks] = 1.0 / picks.size
    else:
        msg = f"rule must be 'interval' or 'arithmetic', got {rule!r}"
        raise ValueError(msg)
    return weights


def weight_kernel(
    azimuths: npt.ArrayLike,
    weights: npt.ArrayLike,
    orders: Sequence[int] | npt.ArrayLike,
) -> npt.NDArray[np.complex128]:
    r"""Return the quadrature-weight DFT kernel of an azimuth grid.

    The kernel is :math:`W(h)=\sum_\ell w_\ell e^{jh\psi_\ell}`, the aliasing kernel of
    the grid.

    Parameters
    ----------
    azimuths : npt.ArrayLike
        Azimuths in radians.
    weights : npt.ArrayLike
        One weight per azimuth.
    orders : Sequence[int] or npt.ArrayLike
        Harmonic orders :math:`h` at which to evaluate the kernel.

    Returns
    -------
    npt.NDArray[np.complex128]
        :math:`W(h)` for each requested order. :math:`W(0)=\sum_\ell w_\ell`. On a
        uniform :math:`N`-grid starting at :math:`\psi_0` with equal weights,
        :math:`W(h)=e^{jh\psi_0}` when :math:`N\mid h` and zero otherwise.

    Raises
    ------
    ValueError
        If the weights do not match the azimuths or are non-finite.
    """
    angles = _azimuth_array(azimuths)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != angles.shape or not np.all(np.isfinite(w)):
        msg = f"weights must be finite with one per azimuth ({angles.size})"
        raise ValueError(msg)
    h = np.asarray(orders, dtype=np.float64).ravel()
    return np.asarray(np.exp(1j * np.outer(h, angles)) @ w, dtype=np.complex128)


def uniform_azimuths(count: int, start: float = 0.0) -> npt.NDArray[np.float64]:
    """Return a uniform grid of ``count`` azimuths (rad) starting at ``start``.

    Parameters
    ----------
    count : int
        Number of azimuths; must be at least one.
    start : float, optional
        First azimuth in radians.

    Returns
    -------
    npt.NDArray[np.float64]
        ``start + 2 pi k / count`` for ``k = 0 .. count - 1``, wrapped into
        ``[0, 2pi)``.

    Raises
    ------
    ValueError
        If ``count`` is not a positive integer or ``start`` is not finite.
    """
    if int(count) != count or count < 1:
        msg = f"count must be a positive integer, got {count!r}"
        raise ValueError(msg)
    if not math.isfinite(start):
        msg = "start must be finite"
        raise ValueError(msg)
    grid = float(start) + _TWO_PI * np.arange(int(count), dtype=np.float64) / count
    return np.asarray(np.mod(grid, _TWO_PI), dtype=np.float64)


def azimuth_mean(
    azimuths: npt.ArrayLike,
    matrices: npt.ArrayLike,
    *,
    rule: Literal["interval", "arithmetic"] = "interval",
) -> npt.NDArray[np.float64]:
    """Return the azimuth mean of a stack of sampled matrices.

    Parameters
    ----------
    azimuths : npt.ArrayLike
        Delivered azimuths in radians, one per sample.
    matrices : npt.ArrayLike
        Stack of samples, shape ``(n_azimuth, ...)`` (state, input or output matrices).
    rule : {"interval", "arithmetic"}, optional
        ``"interval"`` (default) uses :func:`interval_weights`; ``"arithmetic"`` the
        plain sample mean that the MBC transform applies.

    Returns
    -------
    npt.NDArray[np.float64]
        The weighted mean, with the shape of one sample.

    Raises
    ------
    ValueError
        If the inputs are inconsistent or the rule is unknown.
    """
    angles = _azimuth_array(azimuths)
    stack = _matrix_stack(matrices, angles.size, square=False)
    weights = subgrid_weights(angles, np.arange(angles.size), rule=rule)
    return np.asarray(np.tensordot(weights, stack, axes=(0, 0)), dtype=np.float64)


# --------------------------------------------------------------------------------------
# Harmonic model
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HarmonicModel:
    r"""Harmonic representation :math:`\mathbf A(\psi)=\sum_h\mathbf A_h e^{jh\psi}`.

    Parameters
    ----------
    azimuths : npt.NDArray[np.float64]
        The delivered azimuths (rad, wrapped into ``[0, 2pi)``) the model was fitted to.
    reference_weights : npt.NDArray[np.float64]
        The interval weights of those azimuths (the reference quadrature rule).
    reference_mean : npt.NDArray[np.float64]
        The reference mean :math:`\sum_\ell w_\ell^{\mathrm{ref}}\mathbf A(\psi_\ell)`
        computed from the samples.
    coefficients : dict[int, npt.NDArray[np.complex128]]
        :math:`\mathbf A_h` for every order present, with
        :math:`\mathbf A_{-h}=\overline{\mathbf A_h}`.
    method : str
        ``"trigonometric"`` (least-squares trigonometric series) or
        ``"piecewise_linear"`` (exact Fourier coefficients of the periodic
        piecewise-linear interpolant).
    interpolating : bool
        Whether the model reproduces every sample exactly (a square trigonometric
        design). Only an interpolating model makes :func:`grid_error_matrix` exact.
    nyquist_order : int or None
        The order represented by its cosine only (half the sample count of an
        interpolating fit to an even number of samples), else ``None``.
    blades : int or None
        When set, only orders that are multiples of ``blades`` were fitted.
    """

    azimuths: npt.NDArray[np.float64]
    reference_weights: npt.NDArray[np.float64]
    reference_mean: npt.NDArray[np.float64]
    coefficients: dict[int, npt.NDArray[np.complex128]]
    method: str
    interpolating: bool
    nyquist_order: int | None = None
    blades: int | None = None

    @property
    def orders(self) -> npt.NDArray[np.int_]:
        """Return every order present (negative and positive), ascending."""
        return np.asarray(sorted(self.coefficients), dtype=np.int_)

    @property
    def max_order(self) -> int:
        """Return the highest order present (including a cosine-only Nyquist order)."""
        return int(max(self.coefficients))

    @property
    def resolved_order(self) -> int:
        """Return the highest order whose cosine and sine are both resolved."""
        if self.nyquist_order is None:
            return self.max_order
        return int(
            max((h for h in self.coefficients if h < self.nyquist_order), default=0)
        )

    @property
    def mean(self) -> npt.NDArray[np.float64]:
        r"""Return the zeroth harmonic :math:`\mathbf A_0` (real)."""
        return np.asarray(self.coefficients[0].real, dtype=np.float64)

    def evaluate(self, azimuths: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Evaluate the model at the given azimuths.

        Parameters
        ----------
        azimuths : npt.ArrayLike
            Azimuths in radians (any finite values).

        Returns
        -------
        npt.NDArray[np.float64]
            The model matrices, shape ``(n, *sample_shape)``.
        """
        angles = np.atleast_1d(np.asarray(azimuths, dtype=np.float64))
        sample_shape = self.coefficients[0].shape
        total = np.zeros((angles.size, *sample_shape), dtype=np.complex128)
        expand = (angles.size,) + (1,) * len(sample_shape)
        for h, matrix in self.coefficients.items():
            total += np.exp(1j * h * angles).reshape(expand) * matrix[None]
        return np.asarray(total.real, dtype=np.float64)

    def uniform_mean(self, count: int, start: float = 0.0) -> npt.NDArray[np.float64]:
        r"""Return the model mean over a uniform ``count``-grid starting at ``start``.

        Parameters
        ----------
        count : int
            Number of uniform azimuths.
        start : float, optional
            First azimuth in radians.

        Returns
        -------
        npt.NDArray[np.float64]
            :math:`\mathbf A_0+\sum_{h\neq0,\ N\mid h}\mathbf A_h e^{jh\psi_0}`, the
            exact equal-weight mean of the model on that grid.
        """
        if int(count) != count or count < 1:
            msg = f"count must be a positive integer, got {count!r}"
            raise ValueError(msg)
        total = self.coefficients[0].astype(np.complex128).copy()
        for h, matrix in self.coefficients.items():
            if h != 0 and h % int(count) == 0:
                total += matrix * np.exp(1j * h * float(start))
        return np.asarray(total.real, dtype=np.float64)


def _fit_orders(max_order: int, blades: int | None) -> list[int]:
    """Return the positive orders ``1..max_order`` kept by an optional blade filter."""
    return [h for h in range(1, max_order + 1) if blades is None or h % blades == 0]


def _trigonometric_fit(
    angles: npt.NDArray[np.float64],
    stack: npt.NDArray[np.float64],
    max_order: int | None,
    blades: int | None,
) -> tuple[dict[int, npt.NDArray[np.complex128]], bool, int | None]:
    """Least-squares trigonometric fit; returns coefficients, interpolating, Nyquist."""
    n = angles.size
    nyquist: int | None = None
    if max_order is None:
        order = (n - 1) // 2
        if n % 2 == 0 and n >= 2:
            nyquist = n // 2
    else:
        order = int(max_order)
    if blades is not None and nyquist is not None and nyquist % blades:
        nyquist = None
    positive = _fit_orders(order, blades)
    columns = [np.ones_like(angles)]
    for h in positive:
        columns += [np.cos(h * angles), np.sin(h * angles)]
    if nyquist is not None:
        columns.append(np.cos(nyquist * angles))
    design = np.column_stack(columns)
    if design.shape[1] > n:
        msg = (
            f"{design.shape[1]} trigonometric unknowns cannot be fitted to {n} "
            "samples; "
            f"lower max_order to at most {(n - 1) // 2}"
        )
        raise ValueError(msg)
    condition = float(np.linalg.cond(design))
    if not math.isfinite(condition) or condition > _MAX_DESIGN_CONDITION:
        msg = (
            f"the trigonometric design is ill-conditioned (condition {condition:.3g}); "
            "the azimuths are too clustered for the requested orders"
        )
        raise ValueError(msg)
    shape = stack.shape[1:]
    flat = stack.reshape(n, -1)
    solution = np.linalg.lstsq(design, flat, rcond=None)[0]
    coefficients: dict[int, npt.NDArray[np.complex128]] = {
        0: solution[0].reshape(shape).astype(np.complex128)
    }
    for position, h in enumerate(positive):
        a = solution[1 + 2 * position].reshape(shape)
        b = solution[2 + 2 * position].reshape(shape)
        coefficients[h] = np.asarray(0.5 * (a - 1j * b), dtype=np.complex128)
        coefficients[-h] = np.asarray(0.5 * (a + 1j * b), dtype=np.complex128)
    if nyquist is not None:
        half = np.asarray(0.5 * solution[-1].reshape(shape), dtype=np.complex128)
        coefficients[nyquist] = half
        coefficients[-nyquist] = half.copy()
    return coefficients, design.shape[1] == n, nyquist


def _hat_integrals(
    u: npt.NDArray[np.complex128],
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    r"""Return :math:`\int_0^1 e^{ut}dt` and :math:`\int_0^1 t e^{ut}dt` elementwise."""
    small = np.abs(u) < _SERIES_LIMIT
    safe = np.where(small, 1.0, u)
    exp_u = np.exp(safe)
    i0 = (exp_u - 1.0) / safe
    i1 = (exp_u * (safe - 1.0) + 1.0) / safe**2
    if np.any(small):
        us = u[small]
        s0 = np.zeros_like(us)
        s1 = np.zeros_like(us)
        power = np.ones_like(us)
        for k in range(_SERIES_TERMS):
            s0 += power / math.factorial(k + 1)
            s1 += power / ((k + 2) * math.factorial(k))
            power = power * us
        i0[small] = s0
        i1[small] = s1
    return np.asarray(i0, dtype=np.complex128), np.asarray(i1, dtype=np.complex128)


def _piecewise_linear_fit(
    angles: npt.NDArray[np.float64],
    stack: npt.NDArray[np.float64],
    max_order: int | None,
    blades: int | None,
) -> dict[int, npt.NDArray[np.complex128]]:
    """Exact Fourier coefficients of the periodic piecewise-linear interpolant."""
    n = angles.size
    order = (n - 1) // 2 if max_order is None else int(max_order)
    sort = np.argsort(angles)
    nodes = angles[sort]
    values = stack[sort]
    gaps = np.diff(np.append(nodes, nodes[0] + _TWO_PI))
    following = np.roll(np.arange(n), -1)
    coefficients: dict[int, npt.NDArray[np.complex128]] = {}
    for h in [0, *_fit_orders(order, blades)]:
        u = np.asarray(-1j * h * gaps, dtype=np.complex128)
        i0, i1 = _hat_integrals(u)
        scale = gaps * np.exp(-1j * h * nodes) / _TWO_PI
        node_weights = np.zeros(n, dtype=np.complex128)
        np.add.at(node_weights, np.arange(n), scale * (i0 - i1))
        np.add.at(node_weights, following, scale * i1)
        coefficient = np.tensordot(node_weights, values, axes=(0, 0))
        coefficients[h] = np.asarray(coefficient, dtype=np.complex128)
        if h:
            coefficients[-h] = np.asarray(np.conj(coefficient), dtype=np.complex128)
    return coefficients


def fit_harmonics(
    azimuths: npt.ArrayLike,
    matrices: npt.ArrayLike,
    *,
    method: Literal["trigonometric", "piecewise_linear"] = "trigonometric",
    max_order: int | None = None,
    blades: int | None = None,
) -> HarmonicModel:
    r"""Fit a harmonic model :math:`\mathbf A(\psi)=\sum_h\mathbf A_h e^{jh\psi}`.

    Parameters
    ----------
    azimuths : npt.ArrayLike
        Delivered azimuths in radians, one per sample (any order).
    matrices : npt.ArrayLike
        Sample stack, shape ``(n_azimuth, ...)``.
    method : {"trigonometric", "piecewise_linear"}, optional
        ``"trigonometric"`` (default) solves the trigonometric design by least squares.
        With the default ``max_order`` it is square, orders ``0..(M-1)//2`` plus the
        cosine of order ``M/2`` for an even sample count ``M``, so the model
        interpolates every sample and the sampling-error identity of
        :func:`grid_error_matrix` holds exactly. ``"piecewise_linear"`` returns the
        exact Fourier coefficients of the periodic piecewise-linear interpolant of the
        samples; its :math:`\mathbf A_0` equals the interval-weighted mean, which makes
        it the natural model for the sideband (periodicity) estimate.
    max_order : int or None, optional
        Highest harmonic order. Default ``(M-1)//2`` (with the Nyquist cosine for the
        trigonometric fit to an even ``M``). A trigonometric fit with
        ``2*max_order + 1 > M`` is rejected.
    blades : int or None, optional
        Keep only orders that are multiples of ``blades`` (the content an isotropic
        ``blades``-bladed rotor leaves after MBC). The trigonometric fit is then a
        least-squares fit and does not interpolate.

    Returns
    -------
    HarmonicModel
        The fitted model, with the reference (interval) weights and mean.

    Raises
    ------
    ValueError
        If the inputs are inconsistent, ``method`` is unknown, ``max_order`` or
        ``blades`` is not a positive integer, the design has more unknowns than
        samples, or it is ill-conditioned.
    """
    angles = _azimuth_array(azimuths)
    stack = _matrix_stack(matrices, angles.size, square=False)
    if max_order is not None and (int(max_order) != max_order or max_order < 1):
        msg = f"max_order must be a positive integer, got {max_order!r}"
        raise ValueError(msg)
    if blades is not None and (int(blades) != blades or blades < 1):
        msg = f"blades must be a positive integer, got {blades!r}"
        raise ValueError(msg)
    weights = interval_weights(angles)
    reference_mean = np.asarray(np.tensordot(weights, stack, axes=(0, 0)))
    if method == "trigonometric":
        coefficients, interpolating, nyquist = _trigonometric_fit(
            angles, stack, max_order, blades
        )
    elif method == "piecewise_linear":
        coefficients = _piecewise_linear_fit(angles, stack, max_order, blades)
        interpolating, nyquist = False, None
    else:
        msg = f"method must be 'trigonometric' or 'piecewise_linear', got {method!r}"
        raise ValueError(msg)
    return HarmonicModel(
        azimuths=angles,
        reference_weights=weights,
        reference_mean=np.asarray(reference_mean, dtype=np.float64),
        coefficients=coefficients,
        method=method,
        interpolating=interpolating,
        nyquist_order=nyquist,
        blades=None if blades is None else int(blades),
    )


def grid_error_matrix(
    model: HarmonicModel,
    azimuths: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    r"""Return the error of a grid's mean relative to the reference mean.

    Parameters
    ----------
    model : HarmonicModel
        Harmonic model of the delivered samples.
    azimuths : npt.ArrayLike
        The azimuths (rad) of the grid whose mean is assessed: a subset of the
        delivered azimuths (a measured sub-grid) or any hypothetical grid (a design).
    weights : npt.ArrayLike or None, optional
        The grid's quadrature weights; default its :func:`interval_weights`.

    Returns
    -------
    npt.NDArray[np.float64]
        :math:`\mathbf E=\sum_h[W_{\mathrm{grid}}(h)-W_{\mathrm{ref}}(h)]\mathbf A_h`.

    Notes
    -----
    Any weighted mean of the samples is :math:`\sum_h W(h)\mathbf A_h` exactly when the
    model interpolates the samples, so for a sub-grid of the delivered azimuths and an
    interpolating model this equals :math:`\sum_\ell(w_\ell^{S}-w_\ell^{\mathrm{ref}})
    \mathbf A(\psi_\ell)` to rounding. The zeroth harmonic cancels because
    :math:`W(0)=1` for every consistent rule. On a uniform :math:`N`-grid
    :math:`W(h)` vanishes unless :math:`N\mid h` (the familiar aliasing rule); on a
    non-uniform grid it leaks at every order, which this form keeps. For a grid that
    is not a subset of the samples the result is a model-based prediction that
    ignores orders above :attr:`HarmonicModel.max_order`.
    """
    angles = _azimuth_array(azimuths)
    w = interval_weights(angles) if weights is None else np.asarray(weights, float)
    orders = model.orders
    w_grid = weight_kernel(angles, w, orders)
    w_ref = weight_kernel(model.azimuths, model.reference_weights, orders)
    total = np.zeros(model.coefficients[0].shape, dtype=np.complex128)
    for position, h in enumerate(orders):
        delta = w_grid[position] - w_ref[position]
        if delta != 0.0:
            total += delta * model.coefficients[int(h)]
    return np.asarray(total.real, dtype=np.float64)


# --------------------------------------------------------------------------------------
# Biorthonormal eigenbasis
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EigenBasis:
    r"""Biorthonormal eigentriples of a state matrix.

    Parameters
    ----------
    values : npt.NDArray[np.complex128]
        Every eigenvalue :math:`\lambda_k` (conjugates included), in solver order.
    left : npt.NDArray[np.complex128]
        Left eigenvectors as columns, scaled so that :math:`\mathbf u_k^H\mathbf v_k=1`.
    right : npt.NDArray[np.complex128]
        Unit-norm right eigenvectors as columns.
    condition : npt.NDArray[np.float64]
        Eigenvalue condition numbers :math:`1/|\hat{\mathbf u}_k^H\mathbf v_k|` of
        the unit-norm pair.
    """

    values: npt.NDArray[np.complex128]
    left: npt.NDArray[np.complex128]
    right: npt.NDArray[np.complex128]
    condition: npt.NDArray[np.float64]

    def participation(self) -> npt.NDArray[np.float64]:
        r"""Return the participation magnitudes :math:`|\bar u_{sk}v_{sk}|`.

        Returns
        -------
        npt.NDArray[np.float64]
            Shape ``(n_states, n_modes)``.
        """
        return np.asarray(np.abs(np.conj(self.left) * self.right), dtype=np.float64)

    def nearest(self, eigenvalue: complex) -> int:
        """Return the index of the eigenvalue closest to ``eigenvalue``.

        Parameters
        ----------
        eigenvalue : complex
            Target eigenvalue (1/s).

        Returns
        -------
        int
            Index into :attr:`values`.
        """
        return int(np.argmin(np.abs(self.values - complex(eigenvalue))))


def eigen_basis(matrix: npt.ArrayLike) -> EigenBasis:
    """Return the biorthonormal eigentriples of a square state matrix.

    Parameters
    ----------
    matrix : npt.ArrayLike
        Square, finite state matrix.

    Returns
    -------
    EigenBasis
        Eigenvalues with left and right eigenvectors, ``u_k^H v_k = 1``.

    Raises
    ------
    ValueError
        If the matrix is not square and finite, or an eigenvalue is defective (its
        left and right eigenvectors are orthogonal to working precision).
    """
    array = _square_matrix(matrix, "matrix")
    values, left, right = eig(array, left=True, right=True, check_finite=False)
    overlap = np.sum(np.conj(left) * right, axis=0)
    magnitude = np.abs(overlap)
    if np.any(magnitude <= np.finfo(np.float64).eps):
        msg = "the matrix has a defective eigenvalue; its eigentriples are undefined"
        raise ValueError(msg)
    left = left / np.conj(overlap)[None, :]
    return EigenBasis(
        values=np.asarray(values, dtype=np.complex128),
        left=np.asarray(left, dtype=np.complex128),
        right=np.asarray(right, dtype=np.complex128),
        condition=np.asarray(1.0 / magnitude, dtype=np.float64),
    )


# --------------------------------------------------------------------------------------
# Rotor-azimuth state
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AzimuthStates:
    """Positions of the rotor-azimuth position and rotor-speed states.

    Parameters
    ----------
    position : int
        0-based index of the rotor-azimuth position state (ElastoDyn ``DOF_GeAz``).
    speed : int or None, optional
        0-based index of its rate, the rotor-speed state, when known.
    """

    position: int
    speed: int | None = None


def locate_azimuth_state(
    state_descriptions: Sequence[str], *, ndof2: int | None = None
) -> AzimuthStates:
    """Locate the rotor-azimuth position and rotor-speed states from descriptions.

    Parameters
    ----------
    state_descriptions : Sequence[str]
        State descriptions in matrix order (e.g. ``MBCResult.state_descriptions``).
    ndof2 : int or None, optional
        Number of second-order DOFs of the ``[q2, q2dot, q1]`` layout. Used to place
        the rotor-speed state at ``position + ndof2`` when no velocity description of
        the generator DOF is present.

    Returns
    -------
    AzimuthStates
        The position index and, when found, the speed index.

    Raises
    ------
    ValueError
        If no state, or more than one non-velocity state, carries the generator
        azimuth token ``DOF_GeAz``, or ``ndof2`` is inconsistent with the layout.
    """
    positions: list[int] = []
    speeds: list[int] = []
    for index, description in enumerate(state_descriptions):
        info = classify_dof(str(description))
        if info.category is DofCategory.GENERATOR_AZIMUTH:
            (speeds if info.is_velocity else positions).append(index)
    if len(positions) != 1:
        found = "no" if not positions else f"{len(positions)}"
        msg = (
            f"found {found} rotor-azimuth position states (DOF_GeAz); exactly one is "
            "required (is the generator DOF enabled?)"
        )
        raise ValueError(msg)
    position = positions[0]
    speed: int | None = speeds[0] if len(speeds) == 1 else None
    if speed is None and ndof2 is not None:
        if not 0 <= position < ndof2 or position + ndof2 >= len(state_descriptions):
            msg = (
                f"ndof2={ndof2} is inconsistent with the azimuth state at {position} "
                f"among {len(state_descriptions)} states"
            )
            raise ValueError(msg)
        speed = position + ndof2
    return AzimuthStates(position=position, speed=speed)


def _position(azimuth: AzimuthStates | int, n: int) -> int:
    """Return the validated azimuth position index."""
    index = azimuth.position if isinstance(azimuth, AzimuthStates) else int(azimuth)
    if not 0 <= index < n:
        msg = f"azimuth state {index} is out of range for {n} states"
        raise ValueError(msg)
    return index


def _azimuth_mode(basis: EigenBasis, position: int) -> tuple[int, float]:
    """Return the mode with the largest azimuth-state share and that share."""
    part = basis.participation()
    share = part[position] / np.maximum(part.sum(axis=0), np.finfo(np.float64).tiny)
    index = int(np.argmax(share))
    return index, float(share[index])


def azimuth_tangent(
    mean_matrix: npt.ArrayLike,
    azimuth: AzimuthStates | int,
    *,
    n_samples: int | None = None,
) -> npt.NDArray[np.float64]:
    r"""Return the orbit-tangent estimate :math:`\boldsymbol\xi` from an averaged model.

    Parameters
    ----------
    mean_matrix : npt.ArrayLike
        The reference (azimuth-averaged) state matrix of a **dense** grid.
    azimuth : AzimuthStates or int
        The rotor-azimuth position state.
    n_samples : int or None, optional
        Number of azimuths that formed the mean. Below 18 a warning is issued: the
        neutral direction of a coarse mean is itself contaminated by aliasing.

    Returns
    -------
    npt.NDArray[np.float64]
        The real right eigenvector of the mode with the largest azimuth-state share,
        normalised to :math:`\xi_{\mathrm{az}}=1`.

    Raises
    ------
    ValueError
        If the matrix is invalid or the azimuth index is out of range.

    Warns
    -----
    RuntimeWarning
        If ``n_samples`` is below 18, or the azimuth eigenvalue is complex (an
        oscillatory azimuth pair; the real part of its eigenvector is returned and
        pinning then preserves the reference only to the size of its imaginary part).
    """
    basis = eigen_basis(mean_matrix)
    position = _position(azimuth, basis.values.size)
    if n_samples is not None and n_samples < _MIN_TANGENT_SAMPLES:
        warnings.warn(
            f"the tangent of a {n_samples}-azimuth mean is contaminated by aliasing; "
            f"estimate it from a grid of at least {_MIN_TANGENT_SAMPLES} azimuths",
            RuntimeWarning,
            stacklevel=2,
        )
    index, _share = _azimuth_mode(basis, position)
    value = complex(basis.values[index])
    if abs(value.imag) > _COMPLEX_RTOL * max(1.0, abs(value)):
        warnings.warn(
            f"the averaged azimuth eigenvalue {value:.3g} is complex; using the real "
            "part of its eigenvector as the tangent",
            RuntimeWarning,
            stacklevel=2,
        )
    vector = basis.right[:, index] / basis.right[position, index]
    return np.asarray(vector.real, dtype=np.float64)


def pin_azimuth(
    matrices: npt.ArrayLike,
    tangent: npt.ArrayLike,
    azimuth: AzimuthStates | int,
) -> npt.NDArray[np.float64]:
    r"""Apply tangent-pinned averaging to one matrix or a stack of samples.

    Each matrix is replaced by
    :math:`\mathbf A'=\mathbf A(\mathbf I-\boldsymbol\xi\mathbf e_{\mathrm{az}}^T)`.

    Parameters
    ----------
    matrices : npt.ArrayLike
        One state matrix ``(n, n)`` or a stack ``(n_azimuth, n, n)``.
    tangent : npt.ArrayLike
        Tangent :math:`\boldsymbol\xi` of length ``n``; it is rescaled so that its
        azimuth component is one.
    azimuth : AzimuthStates or int
        The rotor-azimuth position state.

    Returns
    -------
    npt.NDArray[np.float64]
        The pinned matrices, same shape as ``matrices``. Only the azimuth column
        changes, to :math:`\mathbf A\mathbf e_{\mathrm{az}}-\mathbf A\boldsymbol\xi`.

    Raises
    ------
    ValueError
        If shapes disagree, the inputs are non-finite, or the tangent's azimuth
        component is zero.

    Notes
    -----
    :math:`\mathbf I-\boldsymbol\xi\mathbf e_{\mathrm{az}}^T` is an oblique projector,
    so :math:`\mathbf A'(\psi)\boldsymbol\xi=\mathbf 0` for every sample: every
    sub-grid mean has an exact zero eigenvalue and the periodic model an exact unit
    Floquet multiplier. If :math:`\boldsymbol\xi` is a right eigenvector of the
    reference mean with eigenvalue :math:`\lambda_{\mathrm{az}}`, the pinned mean is
    :math:`\bar{\mathbf A}-\lambda_{\mathrm{az}}\boldsymbol\xi\mathbf e_{\mathrm{az}}^T`
    and, by Brauer's theorem, keeps every other reference eigenvalue unchanged.
    """
    stack = np.asarray(matrices, dtype=np.float64)
    single = stack.ndim == 2
    if single:
        stack = stack[None]
    if stack.ndim != 3 or stack.shape[1] != stack.shape[2]:
        msg = f"matrices must be (n, n) or (n_azimuth, n, n), got {np.shape(matrices)}"
        raise ValueError(msg)
    if not np.all(np.isfinite(stack)):
        msg = "matrices must be finite (no NaN or inf)"
        raise ValueError(msg)
    n = stack.shape[1]
    position = _position(azimuth, n)
    xi = np.asarray(tangent, dtype=np.float64)
    if xi.shape != (n,) or not np.all(np.isfinite(xi)):
        msg = f"tangent must be a finite vector of length {n}"
        raise ValueError(msg)
    if xi[position] == 0.0:
        msg = "the tangent's azimuth component must be non-zero"
        raise ValueError(msg)
    xi = xi / xi[position]
    pinned = stack.copy()
    pinned[:, :, position] = stack[:, :, position] - stack @ xi
    return np.asarray(pinned[0] if single else pinned, dtype=np.float64)


def zero_azimuth_column(
    matrices: npt.ArrayLike, azimuth: AzimuthStates | int
) -> npt.NDArray[np.float64]:
    r"""Zero the rotor-azimuth column (naive neutralisation, kept for comparison).

    Parameters
    ----------
    matrices : npt.ArrayLike
        One state matrix ``(n, n)`` or a stack ``(n_azimuth, n, n)``.
    azimuth : AzimuthStates or int
        The rotor-azimuth position state.

    Returns
    -------
    npt.NDArray[np.float64]
        A copy with the azimuth column set to zero.

    Raises
    ------
    ValueError
        If the matrices are not square or the index is out of range.

    Notes
    -----
    This is :func:`pin_azimuth` with :math:`\boldsymbol\xi=\mathbf e_{\mathrm{az}}`.
    It restores an exact zero eigenvalue and the perturbative regime of the sampling
    error, but :math:`\mathbf e_{\mathrm{az}}` is not an eigenvector of the mean, so
    it **changes the reference spectrum**: the mean of the azimuth column carries the
    rotation of the steady cyclic blade deflection, which is physical. On archived
    floating-turbine linearizations it moved platform damping by up to 7 % of its
    value. Prefer :func:`pin_azimuth` with :func:`azimuth_tangent` of a dense grid.
    """
    stack = np.array(matrices, dtype=np.float64)
    if stack.ndim not in (2, 3) or stack.shape[-1] != stack.shape[-2]:
        msg = f"matrices must be (n, n) or (n_azimuth, n, n), got {stack.shape}"
        raise ValueError(msg)
    position = _position(azimuth, stack.shape[-1])
    stack[..., :, position] = 0.0
    return stack


def delete_azimuth_state(
    matrices: npt.ArrayLike, azimuth: AzimuthStates | int
) -> npt.NDArray[np.float64]:
    """Delete the rotor-azimuth position state (row and column).

    Parameters
    ----------
    matrices : npt.ArrayLike
        One state matrix ``(n, n)`` or a stack ``(n_azimuth, n, n)``.
    azimuth : AzimuthStates or int
        The rotor-azimuth position state.

    Returns
    -------
    npt.NDArray[np.float64]
        Matrices of size ``n - 1``. The rotor-speed state is kept as a first-order
        state, so the ``[q2, q2dot, q1]`` pairing no longer holds.

    Raises
    ------
    ValueError
        If the matrices are not square or the index is out of range.

    Notes
    -----
    This is the conventional Campbell practice of dropping the zero-frequency
    generator mode. Its eigenvalues (and Floquet multipliers) are exactly those of
    :func:`zero_azimuth_column` without the zero, because a zero column makes the
    matrix block-triangular. It therefore **biases the reference** in the same way;
    see :func:`zero_azimuth_column`.
    """
    stack = np.asarray(matrices, dtype=np.float64)
    if stack.ndim not in (2, 3) or stack.shape[-1] != stack.shape[-2]:
        msg = f"matrices must be (n, n) or (n_azimuth, n, n), got {stack.shape}"
        raise ValueError(msg)
    position = _position(azimuth, stack.shape[-1])
    keep = np.setdiff1d(np.arange(stack.shape[-1]), [position])
    return np.asarray(stack[..., keep[:, None], keep[None, :]], dtype=np.float64)


# --------------------------------------------------------------------------------------
# Sideband (periodicity) correction and the azimuth consistency test
# --------------------------------------------------------------------------------------


def _sideband_all(
    model: HarmonicModel,
    basis: EigenBasis,
    indices: Sequence[int],
    rotor_speed: float,
    max_order: int | None,
) -> npt.NDArray[np.complex128]:
    """Return the second-order sideband correction for several modes at once."""
    picks = np.asarray(list(indices), dtype=np.int_)
    values = basis.values
    left_h = basis.left.conj().T
    total = np.zeros(picks.size, dtype=np.complex128)
    limit = model.max_order if max_order is None else int(max_order)
    for h in model.orders:
        order = int(h)
        if order == 0 or abs(order) > limit or -order not in model.coefficients:
            continue
        forcing = (
            basis.left[:, picks].conj().T @ model.coefficients[-order] @ basis.right
        )
        response = left_h @ model.coefficients[order] @ basis.right[:, picks]
        detuning = values[picks][None, :] + 1j * order * rotor_speed - values[:, None]
        numerator = forcing.T * response
        # an exact resonance without content contributes nothing; with content it
        # makes the second-order estimate infinite rather than NaN
        exact = detuning == 0.0
        safe = np.where(exact, 1.0, detuning)
        terms = np.where(
            exact, np.where(numerator == 0.0, 0.0, np.inf), numerator / safe
        )
        total += np.sum(terms, axis=0)
    return total


def sideband_correction(
    model: HarmonicModel,
    basis: EigenBasis,
    index: int,
    rotor_speed: float,
    *,
    max_order: int | None = None,
) -> complex:
    r"""Return the second-order sideband (periodicity) correction of one mode.

    Parameters
    ----------
    model : HarmonicModel
        Harmonic model of the samples (the ``"piecewise_linear"`` fit is consistent
        with the interval-weighted reference).
    basis : EigenBasis
        Eigentriples of the reference mean.
    index : int
        Mode index into ``basis.values``.
    rotor_speed : float
        Rotor speed :math:`\Omega` in rad/s.
    max_order : int or None, optional
        Highest harmonic order included; default every order of the model.

    Returns
    -------
    complex
        :math:`\widehat{\Delta\lambda}_j=\sum_{h\neq0}\sum_k
        \frac{(\mathbf u_j^H\mathbf A_{-h}\mathbf v_k)
        (\mathbf u_k^H\mathbf A_h\mathbf v_j)}{\lambda_j+jh\Omega-\lambda_k}`,
        the estimate of the Floquet exponent minus the
        averaged eigenvalue. Every mode is retained in the sum: truncating it to the
        modes nearest in detuning is not admissible.

    Raises
    ------
    ValueError
        If ``rotor_speed`` is not positive or ``index`` is out of range.
    """
    omega = _positive_rate(rotor_speed)
    if not 0 <= index < basis.values.size:
        msg = f"index {index} is out of range for {basis.values.size} modes"
        raise ValueError(msg)
    return complex(_sideband_all(model, basis, [index], omega, max_order)[0])


_ConsistencyAssessment = Literal[
    "consistent", "inconsistent-but-harmless", "inconsistent-and-risky"
]


@dataclass(frozen=True)
class BranchAzimuthRisk:
    r"""How strongly the azimuth residual can contaminate one branch.

    Parameters
    ----------
    index : int
        Index of the branch in the eigenbasis of the reference mean.
    eigenvalue : complex
        Its reference eigenvalue :math:`\lambda_j=-\sigma_j+j\omega_j` (1/s).
    decay_ratio : float
        :math:`\nu_j=r/\sigma_j`, the residual exponent over the branch decay rate
        (``inf`` for a branch that does not decay).
    coalescence_ratio : float
        :math:`\mu_j=|\kappa_r|/(\omega_j^2+b_\gamma^2/4)`, the residual azimuth
        stiffness over the coalescence threshold of the branch.
    risky : bool
        Both ratios exceed the scale tolerance.
    """

    index: int
    eigenvalue: complex
    decay_ratio: float
    coalescence_ratio: float
    risky: bool


@dataclass(frozen=True)
class AzimuthConsistency:
    r"""Result of the azimuth (periodic-orbit) consistency test.

    The azimuth position and rotor-speed states form a pair of modes, condensed
    :math:`s^2-bs-\kappa=0`. Both modes are tested: their sideband-corrected
    exponents :math:`\gamma_{1,2}` estimate the two Floquet exponents, one of which is
    exactly zero for a linearization about a periodic orbit.

    Parameters
    ----------
    eigenvalue : complex
        The averaged eigenvalue :math:`\lambda` of the **growing** mode of the pair,
        the one whose exponent has the larger real part (1/s).
    correction : complex
        Its sideband correction :math:`\widehat{\Delta\lambda}`.
    exponent : complex
        :math:`\gamma_1=\lambda+\widehat{\Delta\lambda}`, the growing exponent of the
        pair. In a saddle (:math:`\kappa>0`) this is the unstable root, not the stable
        one, whatever their azimuth shares.
    index : int
        Index of that mode in the eigenbasis of the reference mean.
    azimuth_share : float
        Participation share of the azimuth position state in that mode.
    condition : float
        Eigenvalue condition number of that mode (large for the near-Jordan pair).
    stiffness : float
        :math:`\kappa`, the azimuth stiffness of the static condensation of the
        reference mean onto (azimuth, rotor speed); ``nan`` without the speed state.
    speed_damping : float
        :math:`b`, the rotor-speed damping of that condensation (1/s); ``nan``
        without the speed state.
    tolerance : float
        Absolute tolerance on the residual exponent (1/s), kept for comparison.
    consistent : bool
        ``assessment == "consistent"``.
    pair_indices : tuple[int, int]
        Indices of the two modes of the pair, growing first.
    pair_eigenvalues : tuple[complex, complex]
        Their averaged eigenvalues.
    pair_exponents : tuple[complex, complex]
        Their sideband-corrected exponents :math:`(\gamma_1,\gamma_2)`,
        :math:`\operatorname{Re}\gamma_1\ge\operatorname{Re}\gamma_2`.
    residual_exponent : float
        :math:`r=\min(|\gamma_1|,|\gamma_2|)`, the distance of the pair from an exact
        zero exponent (1/s).
    residual_stiffness : float
        :math:`\kappa_r=-\operatorname{Re}(\gamma_1\gamma_2)`, the azimuth stiffness
        implied by the exponents (1/s^2); zero for a consistent linearization.
    absolute_consistent : bool
        ``residual_exponent <= tolerance``: a fixed absolute-tolerance test.
    scale_tolerance : float
        The dimensionless tolerance :math:`	au` of the scaled test.
    branch_risks : tuple[BranchAzimuthRisk, ...]
        The decay and coalescence ratios of every assessed branch.
    assessment : str
        ``"consistent"``, ``"inconsistent-but-harmless"`` or
        ``"inconsistent-and-risky"`` (see :func:`azimuth_consistency`).
    """

    eigenvalue: complex
    correction: complex
    exponent: complex
    index: int
    azimuth_share: float
    condition: float
    stiffness: float
    speed_damping: float
    tolerance: float
    consistent: bool
    pair_indices: tuple[int, int]
    pair_eigenvalues: tuple[complex, complex]
    pair_exponents: tuple[complex, complex]
    residual_exponent: float
    residual_stiffness: float
    absolute_consistent: bool
    scale_tolerance: float
    branch_risks: tuple[BranchAzimuthRisk, ...]
    assessment: _ConsistencyAssessment

    @property
    def growth_rate(self) -> float:
        r"""Return :math:`\operatorname{Re}\gamma_1`, the largest real part of the pair.

        With an open-loop rotor whose aerodynamic torque rises with speed
        (:math:`b>0`) this is the physical rotor-speed divergence, not a defect; the
        assessment therefore uses the residual, not the growth rate.
        """
        return float(self.exponent.real)

    @property
    def speed_damping_exponent(self) -> float:
        r"""Return :math:`b_\gamma=\operatorname{Re}(\gamma_1+\gamma_2)` (1/s)."""
        return float((self.pair_exponents[0] + self.pair_exponents[1]).real)

    @property
    def pendulum_frequency(self) -> float:
        r"""Return the azimuth pendulum frequency :math:`\sqrt{-\kappa-b^2/4}` (rad/s).

        A negative stiffness, for example created by aliasing on a coarse grid, makes
        the azimuth an oscillator that can hybridise with a platform branch near this
        frequency. Zero for a real pair or an unknown condensation.
        """
        discriminant = -self.stiffness - 0.25 * self.speed_damping**2
        if not math.isfinite(discriminant) or discriminant <= 0.0:
            return 0.0
        return math.sqrt(discriminant)

    @property
    def residual_pendulum_frequency(self) -> float:
        r"""Return :math:`|\operatorname{Im}\gamma_1|`, the residual pair frequency.

        Non-zero only when the residual makes the azimuth pair oscillatory.
        """
        return float(abs(self.exponent.imag))

    @property
    def worst_branch(self) -> BranchAzimuthRisk | None:
        """Return the assessed branch with the largest coalescence ratio.

        Returns
        -------
        BranchAzimuthRisk or None
            ``None`` when no branch was assessed.
        """
        if not self.branch_risks:
            return None
        return max(self.branch_risks, key=lambda risk: risk.coalescence_ratio)


def _condense(
    matrix: npt.NDArray[np.float64], position: int, speed: int
) -> tuple[float, float]:
    """Return ``(kappa, b)`` of the static condensation onto (azimuth, rotor speed)."""
    keep = np.array([position, speed])
    other = np.setdiff1d(np.arange(matrix.shape[0]), keep)
    block = matrix[np.ix_(keep, keep)]
    if other.size:
        try:
            coupling = np.linalg.solve(
                matrix[np.ix_(other, other)], matrix[np.ix_(other, keep)]
            )
        except np.linalg.LinAlgError:
            return math.nan, math.nan
        block = block - matrix[np.ix_(keep, other)] @ coupling
    return float(block[1, 0]), float(block[1, 1])


def _azimuth_pair(
    basis: EigenBasis, position: int, speed: int | None
) -> tuple[int, int]:
    """Return the azimuth mode and its partner in the azimuth/rotor-speed pair."""
    index, _share = _azimuth_mode(basis, position)
    values = basis.values
    value = complex(values[index])
    others = np.setdiff1d(np.arange(values.size), [index])
    if others.size == 0:
        return index, index
    if abs(value.imag) > _COMPLEX_RTOL * max(1.0, abs(value)):
        partner = int(others[np.argmin(np.abs(values[others] - value.conjugate()))])
        return index, partner
    part = basis.participation()
    weight = part[position].copy()
    if speed is not None:
        weight += part[speed]
    share = weight / np.maximum(part.sum(axis=0), np.finfo(np.float64).tiny)
    real = others[
        np.abs(values[others].imag)
        <= _COMPLEX_RTOL * np.maximum(1.0, np.abs(values[others]))
    ]
    pool = real if real.size else others
    return index, int(pool[np.argmax(share[pool])])


def _branch_risks(
    basis: EigenBasis,
    branches: Sequence[int],
    residual: float,
    stiffness: float,
    damping: float,
    scale: float,
) -> tuple[BranchAzimuthRisk, ...]:
    """Return the decay and coalescence ratios of each assessed branch."""
    risks: list[BranchAzimuthRisk] = []
    for j in branches:
        value = complex(basis.values[j])
        sigma = -value.real
        decay = residual / sigma if sigma > 0.0 else math.inf
        threshold = value.imag**2 + 0.25 * damping**2
        coalescence = abs(stiffness) / threshold if threshold > 0.0 else math.inf
        risks.append(
            BranchAzimuthRisk(
                index=int(j),
                eigenvalue=value,
                decay_ratio=float(decay),
                coalescence_ratio=float(coalescence),
                risky=bool(decay > scale and coalescence > scale),
            )
        )
    return tuple(risks)


def _assess(
    risks: tuple[BranchAzimuthRisk, ...], absolute: bool, scale: float
) -> _ConsistencyAssessment:
    """Return the graded assessment of the azimuth consistency test."""
    if not risks:
        return "consistent" if absolute else "inconsistent-but-harmless"
    if all(risk.decay_ratio <= scale for risk in risks):
        return "consistent"
    if any(risk.risky for risk in risks):
        return "inconsistent-and-risky"
    return "inconsistent-but-harmless"


def azimuth_consistency(
    model: HarmonicModel,
    rotor_speed: float,
    azimuth: AzimuthStates | int,
    *,
    tolerance: float = 1e-4,
    basis: EigenBasis | None = None,
    max_order: int | None = None,
    branches: Sequence[int] | None = None,
    scale_tolerance: float = 0.01,
) -> AzimuthConsistency:
    r"""Test whether the samples are a linearization about a periodic orbit.

    Parameters
    ----------
    model : HarmonicModel
        Harmonic model of the samples (``"piecewise_linear"`` recommended).
    rotor_speed : float
        Rotor speed :math:`\Omega` in rad/s.
    azimuth : AzimuthStates or int
        The rotor-azimuth states (see :func:`locate_azimuth_state`). The rotor-speed
        index is needed for the condensed stiffness and damping.
    tolerance : float, optional
        Absolute tolerance on the residual exponent in 1/s, reported as
        ``absolute_consistent``. Default ``1e-4``. It decides the assessment only when
        no branch is assessed.
    basis : EigenBasis or None, optional
        Eigentriples of ``model.reference_mean`` if already computed.
    max_order : int or None, optional
        Highest harmonic order in the sideband sum; default every order.
    branches : Sequence[int] or None, optional
        Indices (into the eigenbasis of the reference mean) of the branches the
        residual is assessed against. Default every oscillatory mode (positive
        imaginary part) outside the pair.
    scale_tolerance : float, optional
        Dimensionless tolerance :math:`\tau` of the decay and coalescence ratios.
        Default 0.01.
        :func:`~vane.modal.averaging_error_estimate.estimate_averaging_error` uses
        its damping tolerance, as a fraction.

    Returns
    -------
    AzimuthConsistency
        Both exponents of the azimuth/speed pair (the growing one first), the residual
        exponent and stiffness, the per-branch ratios and the graded assessment.

    Raises
    ------
    ValueError
        If the model is not of square matrices, ``rotor_speed``, ``tolerance`` or
        ``scale_tolerance`` is invalid, or an index is out of range.

    Notes
    -----
    The orbit tangent of an autonomous system linearized about a periodic orbit is a
    periodic solution of the linear time-periodic model, so one Floquet exponent is
    exactly zero. The averaged model then has an eigenvalue of second order in the
    harmonics whose sideband correction cancels it. Both modes of the
    azimuth/rotor-speed pair are corrected, giving :math:`\gamma_{1,2}`. In the
    condensation :math:`s^2-bs-\kappa=0` their sum is the speed damping
    :math:`b_\gamma` and their product is :math:`-\kappa_r`, so the defect is the
    residual stiffness :math:`\kappa_r=-\gamma_1\gamma_2` (zero when one root
    vanishes) and the residual exponent :math:`r=\min|\gamma_{1,2}|`. A positive
    :math:`b` with :math:`\kappa_r=0` is the physical open-loop rotor-speed divergence
    and is not a defect.

    The residual matters only for the branches it can reach. For a branch
    :math:`\lambda_j=-\sigma_j+j\omega_j` two ratios are formed,

    .. math::

       \nu_j=\frac{r}{\sigma_j},\qquad
       \mu_j=\frac{|\kappa_r|}{\omega_j^2+b_\gamma^2/4}.

    :math:`\nu_j\le\tau` means the residual root is slower than the branch decay by
    :math:`1/\tau`: even fully mixed into the branch it would change the decay rate by
    at most :math:`\tau\sigma_j`. :math:`\mu_j` is the coalescence condition of the
    two-state azimuth model: the azimuth pendulum, of decay rate :math:`|b|/2`, can be
    tuned to the branch frequency only when the azimuth stiffness reaches
    :math:`\omega_j^2+b^2/4`. :math:`\mu_j` is the share of that threshold the residual
    consumes on its own, and :math:`\sqrt{\mu_j}` is the natural frequency of the
    residual pair relative to it. The assessment is ``"consistent"`` when
    :math:`\nu_j\le\tau` for every branch; ``"inconsistent-and-risky"`` when some
    branch has both :math:`\nu_j>\tau` and :math:`\mu_j>\tau` (the residual is neither
    slow against the branch decay nor a decade below it in frequency, for
    :math:`\tau=0.01`); ``"inconsistent-but-harmless"`` otherwise. Without branches the
    absolute test decides between the first and the last. The test detects but does
    not locate the defect (for example a trim controller still acting during the
    linearization revolution).
    """
    omega = _positive_rate(rotor_speed)
    for name, value in (("tolerance", tolerance), ("scale_tolerance", scale_tolerance)):
        if not math.isfinite(value) or value <= 0.0:
            msg = f"{name} must be positive and finite, got {value!r}"
            raise ValueError(msg)
    reference = _square_matrix(model.reference_mean, "model.reference_mean")
    n = reference.shape[0]
    position = _position(azimuth, n)
    speed = azimuth.speed if isinstance(azimuth, AzimuthStates) else None
    if speed is not None and not 0 <= speed < n:
        msg = f"rotor-speed state {speed} is out of range for {n} states"
        raise ValueError(msg)
    if basis is None:
        basis = eigen_basis(reference)
    pair = _azimuth_pair(basis, position, speed)
    corrections = _sideband_all(model, basis, list(pair), omega, max_order)
    exponents = [
        complex(basis.values[k]) + complex(c)
        for k, c in zip(pair, corrections, strict=True)
    ]
    first, second = sorted(
        range(2), key=lambda k: (-exponents[k].real, -exponents[k].imag)
    )
    gamma = (exponents[first], exponents[second])
    if pair[0] == pair[1]:  # a one-state model has no partner
        residual, residual_stiffness = abs(gamma[0]), 0.0
    else:
        residual = min(abs(gamma[0]), abs(gamma[1]))
        residual_stiffness = -(gamma[0] * gamma[1]).real
    if branches is None:
        upper = np.flatnonzero(basis.values.imag > 0.0)
        picks = [int(k) for k in upper if int(k) not in pair]
    else:
        picks = [int(k) for k in branches]
        if any(not 0 <= k < n for k in picks):
            msg = f"branch indices must lie in [0, {n})"
            raise ValueError(msg)
    risks = _branch_risks(
        basis,
        picks,
        float(residual),
        float(residual_stiffness),
        float((gamma[0] + gamma[1]).real),
        float(scale_tolerance),
    )
    absolute = bool(residual <= tolerance)
    assessment = _assess(risks, absolute, float(scale_tolerance))
    stiffness, damping = (
        _condense(reference, position, speed)
        if speed is not None
        else (math.nan, math.nan)
    )
    growing = pair[first]
    part = basis.participation()
    shares = part[position] / np.maximum(part.sum(axis=0), np.finfo(np.float64).tiny)
    return AzimuthConsistency(
        eigenvalue=complex(basis.values[growing]),
        correction=complex(corrections[first]),
        exponent=gamma[0],
        index=int(growing),
        azimuth_share=float(shares[growing]),
        condition=float(basis.condition[growing]),
        stiffness=stiffness,
        speed_damping=damping,
        tolerance=float(tolerance),
        consistent=assessment == "consistent",
        pair_indices=(int(pair[first]), int(pair[second])),
        pair_eigenvalues=(
            complex(basis.values[pair[first]]),
            complex(basis.values[pair[second]]),
        ),
        pair_exponents=gamma,
        residual_exponent=float(residual),
        residual_stiffness=float(residual_stiffness),
        absolute_consistent=absolute,
        scale_tolerance=float(scale_tolerance),
        branch_risks=risks,
        assessment=assessment,
    )


# --------------------------------------------------------------------------------------
# Tower-passage spectrum of the azimuth column
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TowerPassageFit:
    r"""Fit of the tower-passage law to the azimuth-column harmonics.

    Parameters
    ----------
    orders : npt.NDArray[np.int_]
        The fitted orders (resolved multiples of the blade count).
    magnitudes : npt.NDArray[np.float64]
        :math:`|c_h|` at those orders.
    decay : float
        :math:`\rho`, the fitted decay per harmonic order.
    amplitude : float
        :math:`A_B`, the fitted magnitude at the blade order.
    log_r2 : float
        Coefficient of determination of the fit in :math:`\log|c_h|`.
    condensed : bool
        Whether :math:`c_h` is the condensed azimuth stiffness harmonic
        :math:`\boldsymbol\ell^T\mathbf A_h\boldsymbol\xi` (rotor-speed state known)
        or the azimuth-column norm :math:`\|\mathbf A_h\mathbf e_{\mathrm{az}}\|`.
    """

    orders: npt.NDArray[np.int_]
    magnitudes: npt.NDArray[np.float64]
    decay: float
    amplitude: float
    log_r2: float
    condensed: bool


def _condensation_vectors(
    matrix: npt.NDArray[np.float64], position: int, speed: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the static-condensation vectors ``(xi, l)`` onto (azimuth, speed)."""
    n = matrix.shape[0]
    keep = np.array([position, speed])
    other = np.setdiff1d(np.arange(n), keep)
    block = matrix[np.ix_(other, other)]
    right = np.linalg.solve(block, matrix[np.ix_(other, keep)])
    left = np.linalg.solve(block.T, matrix[np.ix_(keep, other)].T).T
    xi = np.zeros(n)
    xi[position] = 1.0
    xi[other] = -right[:, 0]
    ell = np.zeros(n)
    ell[speed] = 1.0
    ell[other] = -left[1, :]
    return xi, ell


def tower_passage_decay(
    model: HarmonicModel, azimuth: AzimuthStates | int, *, blades: int = 3
) -> TowerPassageFit:
    r"""Fit the tower-passage law to the harmonics of the azimuth column.

    Parameters
    ----------
    model : HarmonicModel
        Harmonic model of the delivered samples (the azimuth column as delivered, not
        pinned or zeroed).
    azimuth : AzimuthStates or int
        The rotor-azimuth states. With the rotor-speed state the condensed harmonics
        :math:`c_h=\boldsymbol\ell^T\mathbf A_h\boldsymbol\xi` are fitted, otherwise the
        column norms :math:`\|\mathbf A_h\mathbf e_{\mathrm{az}}\|`.
    blades : int, optional
        Blade count :math:`B`; only its multiples are fitted. Default 3.

    Returns
    -------
    TowerPassageFit
        :math:`|c_h|=A_B\,(h/B)\,\rho^{\,h-B}` fitted by least squares in
        :math:`\log` over the resolved multiples of :math:`B`.

    Raises
    ------
    ValueError
        If the model is not of square matrices, ``blades`` is not a positive integer,
        the condensation is singular, or fewer than two blade multiples are resolved
        with non-zero content.

    Notes
    -----
    A tower passage of peak deficit :math:`\delta` and harmonic decay :math:`\rho`
    makes the frozen-azimuth derivative of the rotor loads, which is the azimuth
    column, carry :math:`|c_h|\propto h\rho^{h-1}`: the derivative of a localised bump
    whose Fourier coefficients fall as :math:`\rho^h`. Every other load harmonic of the
    rotor comes from the same localised inflow, so :math:`\rho` sets the decay above
    the resolved orders that
    :func:`~vane.modal.averaging_error_estimate.azimuth_count_table` uses when a
    branch's own sensitivity does not decay.
    """
    if isinstance(blades, bool) or int(blades) != blades or blades < 1:
        msg = f"blades must be a positive integer, got {blades!r}"
        raise ValueError(msg)
    mean = _square_matrix(model.mean, "model.mean")
    n = mean.shape[0]
    position = _position(azimuth, n)
    speed = azimuth.speed if isinstance(azimuth, AzimuthStates) else None
    vectors: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None
    if speed is not None:
        try:
            vectors = _condensation_vectors(mean, position, speed)
        except np.linalg.LinAlgError as error:
            msg = "the condensation onto (azimuth, rotor speed) is singular"
            raise ValueError(msg) from error
    orders: list[int] = []
    magnitudes: list[float] = []
    b = int(blades)
    # content below rounding of the mean is no content
    floor = 1e3 * np.finfo(np.float64).eps * max(float(np.linalg.norm(mean)), 1.0)
    for h in range(b, model.resolved_order + 1, b):
        if h not in model.coefficients:
            continue
        coefficient = model.coefficients[h]
        if vectors is not None:
            xi, ell = vectors
            value = float(abs(ell @ coefficient @ xi))
        else:
            value = float(np.linalg.norm(coefficient[:, position]))
        if value > floor:
            orders.append(h)
            magnitudes.append(value)
    if len(orders) < 2:  # a line needs two points
        msg = "fewer than two resolved blade multiples carry azimuth-column content"
        raise ValueError(msg)
    h_arr = np.asarray(orders, dtype=np.float64)
    log_m = np.log(np.asarray(magnitudes))
    slope, intercept = np.polyfit(h_arr, log_m - np.log(h_arr / b), 1)
    predicted = intercept + slope * h_arr + np.log(h_arr / b)
    spread = float(np.sum((log_m - log_m.mean()) ** 2))
    r2 = 1.0 - float(np.sum((log_m - predicted) ** 2)) / spread if spread > 0 else 1.0
    return TowerPassageFit(
        orders=np.asarray(orders, dtype=np.int_),
        magnitudes=np.asarray(magnitudes, dtype=np.float64),
        decay=float(np.exp(slope)),
        amplitude=float(np.exp(intercept + slope * b)),
        log_r2=r2,
        condensed=vectors is not None,
    )
