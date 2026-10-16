# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Observability analysis, sensor placement, and the UQ-initialized reduced estimator.

The reduced-order model (:mod:`vane.sysid.rom`) keeps the modes that drive a load; this
module answers the two questions a load estimator then poses: *which sensors observe
those modes*, and *how to assemble the estimator with its uncertainty baked in*.

* :func:`observability_gramian` is the (infinite-horizon) observability Gramian
  :math:`\mathbf W_o` of a stable system — the Lyapunov solution of
  :math:`\mathbf A^{\mathsf T}\mathbf W_o + \mathbf W_o\mathbf A + \mathbf C^{\mathsf T}
  \mathbf C = 0` (continuous) — whose spectrum measures how strongly each state
  direction is seen at the outputs. With a measurement-noise covariance it becomes the
  noise-weighted (Fisher) information Gramian using :math:`\mathbf C^{\mathsf T}\mathbf
  R^{-1}\mathbf C`.
* :func:`finite_horizon_observability_gramian` is the observability Gramian accumulated
  over a finite sensing window :math:`[0, T]` — the relevant measure for short-window
  estimation and the only one defined when the system carries marginal or unstable modes
  (the infinite-horizon Gramian does not exist there). It is computed exactly: by the
  Van Loan block-matrix-exponential identity in continuous time, and by direct summation
  in discrete time.
* :func:`observability_scores` breaks the Gramian down into **per-output** scores (how
  much each candidate sensor sees) and **per-mode** scores (how strongly each identified
  mode is observed), tying the identified modal content to the sensor layout.
* :func:`load_reconstructability` answers the load-estimator question directly: given a
  *named load* output (not necessarily a sensor) and a sensor set, what fraction of that
  load lies in the observable subspace, and what is the a-priori (Cramér-Rao-type)
  variance of reconstructing it — the assessment a virtual-sensing twin needs before it
  runs a single time step.
* :func:`recommend_sensors` chooses, from a set of candidate output channels, the subset
  that maximises observability, by greedy D-optimal (log-determinant of :math:`\mathbf
  W_o`) selection — the standard sensor-placement objective, which rewards covering all
  observable directions rather than piling sensors on the same one.
* :func:`reduced_estimator` assembles a Kalman-ready model on the reduced system whose
  **initial covariance is inflated by each mode's confidence** — a less-trusted mode
  starts with larger variance — tying the reduction, the per-mode uncertainty, and the
  Kalman export into one auditable estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.linalg import (
    cho_factor,
    cho_solve,
    expm,
    solve_continuous_lyapunov,
    solve_discrete_lyapunov,
)

from vane.sysid._covariance import (
    is_positive_definite,
    is_symmetric,
    symmetric_part,
)
from vane.sysid.covariance_init import covariances_from_confidence
from vane.sysid.kalman_interface import build_kalman_model
from vane.sysid.rom import _resolve_output
from vane.sysid.state_space import StateSpace

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from vane.sysid.kalman_interface import KalmanModel
    from vane.sysid.rom import ModalReduction

__all__ = [
    "ConstrainedPlacement",
    "GroupConstraint",
    "LoadReconstructability",
    "ObservabilityScores",
    "SensorPlacement",
    "constrained_sensor_selection",
    "finite_horizon_observability_gramian",
    "load_reconstructability",
    "observability_gramian",
    "observability_scores",
    "recommend_sensors",
    "reduced_estimator",
]

# Largest |Re(eigenvalue)| * sub-interval for the continuous finite-horizon Van Loan
# exponential; the window is split into 2**k sub-intervals to respect it.
_RATE_PER_SUBINTERVAL = 1.0


@dataclass
class ObservabilityScores:
    """Per-output and per-mode breakdown of how strongly a system is observed.

    Parameters
    ----------
    per_output : npt.NDArray[np.float64]
        For each output row, the observability energy that single sensor provides
        (the trace of the Gramian built from that row alone), in the output order.
        A near-zero entry marks a sensor that sees almost nothing.
    output_names : list[str]
        The output names aligned with ``per_output`` (empty strings if unnamed).
    per_mode : npt.NDArray[np.float64]
        For each mode group in ``mode_states``, the observability energy of that mode
        (the trace of the Gramian's diagonal block over the mode's states), in the
        group order. Empty when no mode grouping was supplied.
    """

    per_output: npt.NDArray[np.float64]
    output_names: list[str]
    per_mode: npt.NDArray[np.float64]


@dataclass
class LoadReconstructability:
    """Whether and how well a target load can be reconstructed from a sensor set.

    Parameters
    ----------
    load_name : str
        The name (or ``"output[i]"``) of the target load output.
    reconstructable_fraction : float
        Fraction in ``[0, 1]`` of the load output direction that lies in the observable
        subspace of the sensor set, measured in the **Euclidean metric of the model's
        state coordinates**. ``1`` means the load is fully reconstructable; a value
        below ``1`` means a part of the load lives in the sensors' blind subspace and
        cannot be recovered at any noise level. Only that outcome (``1`` or not) is
        independent of the state basis: the *value* of a partial fraction depends on
        the state coordinates and units (a similarity transform or rescaling of the
        states changes it -- a purely blind load can read close to ``1`` in skewed
        coordinates), because splitting the load between the observable and blind
        subspaces requires a state inner product. Do not rank or threshold on a partial
        value; use ``noise_amplification`` (basis-independent) instead.
    blind_fraction : float
        ``1 - reconstructable_fraction`` — the irrecoverable share of the load, with the
        same coordinate dependence.
    observability : float
        The observed energy per unit load, ``c_load Wo c_load^T / (c_load c_load^T)``;
        larger means the load aligns with strongly observed directions. Like the
        fraction, it is normalised in the state coordinates and so is basis-dependent.
    noise_amplification : float
        The a-priori (Cramér-Rao-type) variance scale of reconstructing the load from
        the sensor information over the window, ``c_load Wo^+ c_load^T`` over the
        observable subspace. With a measurement-noise covariance supplied this is the
        best achievable static reconstruction variance; smaller is better. It is
        ``inf`` whenever ``reconstructable_fraction < 1`` — any blind component is
        irrecoverable at any noise level, so the variance is undefined (reported as
        infinite so a smaller-is-better ranking never prefers a blind load). It is
        invariant under any invertible change of state coordinates (``c_load Wo^+
        c_load^T`` is unchanged when ``x = T x'``), so it is the measure to compare
        sensor sets or models with.
    n_observable_directions : int
        The rank of the sensor observability Gramian (the dimension of the observable
        subspace).
    """

    load_name: str
    reconstructable_fraction: float
    blind_fraction: float
    observability: float
    noise_amplification: float
    n_observable_directions: int


@dataclass
class SensorPlacement:
    """A recommended set of sensors and the observability gained at each step.

    Parameters
    ----------
    selected : list[int]
        The chosen output (sensor) row indices, in the order they were selected.
    names : list[str]
        The output names of the chosen sensors (empty strings if unnamed).
    scores : list[float]
        The observability metric (log-determinant of the Gramian) after each
        selection — a non-decreasing record of the observability gained.
    """

    selected: list[int]
    names: list[str]
    scores: list[float]


@dataclass
class GroupConstraint:
    """A cap on how many sensors may be chosen from one group of channels.

    Parameters
    ----------
    members : Sequence[int]
        Output row indices that belong to the group (e.g. all channels on one component,
        or mutually redundant sensors).
    max_count : int
        The most sensors that may be selected from the group (``>= 0``).
    """

    members: Sequence[int]
    max_count: int


@dataclass
class ConstrainedPlacement:
    """A sensor selection that honours deployment constraints, and its cost.

    Parameters
    ----------
    selected : list[int]
        The chosen output (sensor) row indices: the required sensors first (in the order
        given), then the greedily added ones.
    names : list[str]
        The output names of the chosen sensors (empty strings if unnamed).
    scores : list[float]
        The observability metric (log-determinant of the Gramian) after each selection,
        aligned with ``selected``.
    total_cost : float
        The summed cost of the selected sensors.
    """

    selected: list[int]
    names: list[str]
    scores: list[float]
    total_cost: float


def _is_stable(a: npt.NDArray[np.float64], dt: float | None) -> bool:
    """Return whether ``a`` is asymptotically stable (continuous or discrete)."""
    eigvals = np.linalg.eigvals(a)
    if dt is None:
        return bool(eigvals.real.max() < 0.0)
    return bool(np.abs(eigvals).max() < 1.0)


def _output_information(
    c: npt.NDArray[np.float64],
    measurement_noise: npt.NDArray[np.float64] | None,
) -> npt.NDArray[np.float64]:
    """Return the output information matrix ``C^T R^-1 C`` (or ``C^T C`` if no noise).

    ``measurement_noise`` is the measurement-noise covariance ``R`` (shape
    ``(n_outputs, n_outputs)``); each output is weighted by its inverse noise, so a
    noisy channel contributes less observability.
    """
    if measurement_noise is None:
        return np.asarray(c.T @ c, dtype=np.float64)
    r = np.asarray(measurement_noise, dtype=np.float64)
    factor = _cholesky_of_noise(r, c.shape[0])
    # Symmetric solve C^T R^-1 C = (R^-1 C)^T C via the Cholesky factor (stable, and
    # never forms R^-1). The factorization already proved R positive-definite.
    r_inv_c = cho_solve(factor, c)
    return np.asarray(c.T @ r_inv_c, dtype=np.float64)


def _cholesky_of_noise(
    r: npt.NDArray[np.float64], n_outputs: int
) -> tuple[npt.NDArray[np.float64], bool]:
    """Validate ``R`` and return its Cholesky factor for a noise-weighted solve.

    A measurement-noise covariance must be square, symmetric, and positive-definite;
    otherwise ``C^T R^-1 C`` is undefined or indefinite and would silently corrupt the
    information Gramian, the sensor scores, and the reconstructability assessment.
    Reject a malformed ``R`` (wrong shape, asymmetric, or with a zero / negative /
    correlated singular variance) with a clear error rather than a raw ``LinAlgError``
    or a quietly wrong result.

    Parameters
    ----------
    r : npt.NDArray[np.float64]
        Candidate measurement-noise covariance ``R``.
    n_outputs : int
        Number of outputs the covariance must match.

    Returns
    -------
    tuple[npt.NDArray[np.float64], bool]
        The ``scipy.linalg.cho_factor`` result for ``R``.

    Raises
    ------
    ValueError
        If ``R`` is not ``(n_outputs, n_outputs)``, not symmetric, or not positive
        definite.
    """
    if r.shape != (n_outputs, n_outputs):
        msg = (
            f"measurement_noise must be ({n_outputs}, {n_outputs}) to match the "
            f"outputs, got {r.shape}"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(r)):
        msg = "measurement_noise must be finite (no NaN or inf)"
        raise ValueError(msg)
    if not is_symmetric(r):
        msg = "measurement_noise must be symmetric"
        raise ValueError(msg)
    if not is_positive_definite(r):
        msg = (
            "measurement_noise must be positive-definite (every channel needs a "
            "strictly positive, non-singular variance); got a singular or indefinite "
            "covariance"
        )
        raise ValueError(msg)
    # Factor the symmetric matrix that was validated, not one triangle of the input.
    factor = cho_factor(symmetric_part(r))
    return cast("tuple[npt.NDArray[np.float64], bool]", factor)


def _solve_gramian(
    a: npt.NDArray[np.float64],
    information: npt.NDArray[np.float64],
    dt: float | None,
) -> npt.NDArray[np.float64]:
    """Solve the observability Gramian Lyapunov equation for source ``information``.

    ``information`` is the output information matrix (``C^T C`` or ``C^T R^-1 C``).
    """
    if dt is None:
        gramian = solve_continuous_lyapunov(a.T, -information)
    else:
        gramian = solve_discrete_lyapunov(a.T, information)
    gramian = np.asarray(gramian, dtype=np.float64)
    return np.asarray(0.5 * (gramian + gramian.T), dtype=np.float64)


def observability_gramian(
    system: StateSpace,
    *,
    measurement_noise: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    r"""Return the infinite-horizon observability Gramian :math:`\mathbf W_o`.

    Solves the Lyapunov equation
    :math:`\mathbf A^{\mathsf T}\mathbf W_o + \mathbf W_o\mathbf A = -\mathbf M`
    (continuous) or
    :math:`\mathbf A^{\mathsf T}\mathbf W_o\mathbf A - \mathbf W_o = -\mathbf M`
    (discrete), where the source is :math:`\mathbf M = \mathbf C^{\mathsf T}\mathbf C`
    or, when ``measurement_noise`` is given, the noise-weighted information
    :math:`\mathbf M = \mathbf C^{\mathsf T}\mathbf R^{-1}\mathbf C`. The Gramian is
    symmetric positive semidefinite; a near-singular Gramian signals weakly- or
    un-observable state directions.

    Parameters
    ----------
    system : StateSpace
        A stable system with a ``C`` matrix.
    measurement_noise : npt.NDArray[np.float64] or None, optional
        Measurement-noise covariance ``R`` (shape ``(n_outputs, n_outputs)``). When
        given, each output is inverse-noise weighted, so the Gramian becomes the Fisher
        information available to an estimator. ``None`` uses unit weighting.

    Returns
    -------
    npt.NDArray[np.float64]
        The ``(n_states, n_states)`` observability Gramian.

    Raises
    ------
    ValueError
        If the system has no ``C`` matrix, is not asymptotically stable, or
        ``measurement_noise`` has the wrong shape.

    See Also
    --------
    finite_horizon_observability_gramian : the finite-window Gramian, defined even
        when the system has marginal or unstable modes.
    """
    if system.c is None:
        msg = "system has no C matrix; cannot form the observability Gramian"
        raise ValueError(msg)
    if not _is_stable(system.a, system.dt):
        msg = "observability Gramian requires an asymptotically stable system"
        raise ValueError(msg)
    information = _output_information(system.c, measurement_noise)
    return _solve_gramian(system.a, information, system.dt)


def finite_horizon_observability_gramian(
    system: StateSpace,
    horizon: float,
    *,
    measurement_noise: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    r"""Return the observability Gramian accumulated over a finite window ``[0, T]``.

    The finite-horizon observability Gramian is

    .. math::

        \mathbf W_o(T) = \int_0^T e^{\mathbf A^{\mathsf T}\tau}\,\mathbf M\,
        e^{\mathbf A\tau}\,\mathrm d\tau \quad\text{(continuous)}, \qquad
        \mathbf W_o(N) = \sum_{k=0}^{N-1}(\mathbf A^{\mathsf T})^k\,\mathbf M\,
        \mathbf A^k \quad\text{(discrete)},

    with source :math:`\mathbf M = \mathbf C^{\mathsf T}\mathbf C` (or its
    noise-weighted form :math:`\mathbf C^{\mathsf T}\mathbf R^{-1}\mathbf C`). Unlike
    the infinite-horizon Gramian it requires **no** stability assumption — it is the
    right observability measure for a short sensing window and for systems that retain
    marginal or unstable (e.g. aero) modes. The continuous integral is evaluated exactly
    by the Van Loan block-matrix-exponential identity; the discrete sum, directly.

    Parameters
    ----------
    system : StateSpace
        A system with a ``C`` matrix (need not be stable).
    horizon : float
        The sensing window length ``T`` in seconds (must be positive and finite). For a
        discrete
        system (``dt`` set) it is converted to ``N = round(T / dt)`` steps (at least 1).
    measurement_noise : npt.NDArray[np.float64] or None, optional
        Measurement-noise covariance ``R``; see :func:`observability_gramian`.

    Returns
    -------
    npt.NDArray[np.float64]
        The ``(n_states, n_states)`` finite-horizon observability Gramian.

    Raises
    ------
    ValueError
        If the system has no ``C`` matrix, ``horizon`` is not positive and finite,
        ``measurement_noise`` has the wrong shape, or the Gramian is not finite (an
        unstable mode growing beyond floating-point range over the window).

    Notes
    -----
    The continuous block ``[[-A^T, M], [0, A]]`` exponentiates ``-A``, which *grows* for
    every stable mode, so over a whole window it is the **fastest** mode (largest
    ``|Re(lambda)| T``) that overflows the intermediate exponential, not the slowest.
    The window is therefore split into ``2^k`` equal sub-intervals ``tau`` short enough
    that ``max |Re(lambda)| tau <= 1``; the sub-interval Gramian is recombined by
    doubling, ``W(2 tau) = W(tau) + Phi(tau)^T W(tau) Phi(tau)`` with
    ``Phi = expm(A tau)``, which stays finite for any stable mode and any window. A
    genuinely unstable mode can still make the Gramian itself overflow over a long
    window; that raises ``ValueError`` rather than returning ``inf``/``nan``. For a
    *stable* system over an effectively infinite horizon use
    :func:`observability_gramian`, which the finite-horizon Gramian converges to.
    """
    if system.c is None:
        msg = "system has no C matrix; cannot form the observability Gramian"
        raise ValueError(msg)
    if not (np.isfinite(horizon) and horizon > 0.0):
        # A finite-horizon Gramian needs a finite window: an infinite horizon makes the
        # block matrix exponential diverge. For a stable, effectively infinite horizon
        # use observability_gramian() (see Notes).
        msg = f"horizon must be a positive, finite time window, got {horizon}"
        raise ValueError(msg)
    a = np.asarray(system.a, dtype=np.float64)
    information = _output_information(system.c, measurement_noise)
    if system.dt is None:
        return _finite_horizon_continuous(a, information, horizon)
    n_steps = max(1, round(horizon / system.dt))
    return _finite_horizon_discrete(a, information, n_steps)


def _finite_horizon_continuous(
    a: npt.NDArray[np.float64],
    information: npt.NDArray[np.float64],
    horizon: float,
) -> npt.NDArray[np.float64]:
    r"""Continuous finite-horizon Gramian via the Van Loan block exponential.

    For ``M = [[-A^T, information], [0, A]]``, ``expm(M tau) = [[*, G], [0, E22]]`` with
    ``E22 = expm(A tau)``, and ``W_o(tau) = E22^T G`` (Van Loan, 1978). The window is
    split into ``2^k`` sub-intervals so ``max |Re(lambda)| tau <= 1`` (the ``-A^T``
    block grows for every stable mode) and recombined by doubling,
    ``W(2 tau) = W(tau) + Phi^T W(tau) Phi``, ``Phi = expm(A tau)``.

    Raises
    ------
    ValueError
        If the resulting Gramian is not finite.
    """
    n = a.shape[0]
    k = 0
    if n > 0:
        rate = float(np.max(np.abs(np.linalg.eigvals(a).real)))
        if rate * horizon > _RATE_PER_SUBINTERVAL:
            k = int(np.ceil(np.log2(rate * horizon / _RATE_PER_SUBINTERVAL)))
    tau = horizon / 2**k
    block = np.block([[-a.T, information], [np.zeros((n, n)), a]])
    exp_block = expm(block * tau)
    phi = np.asarray(exp_block[n:, n:], dtype=np.float64)  # expm(A tau)
    gramian = phi.T @ exp_block[:n, n:]
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(k):
            gramian = gramian + phi.T @ gramian @ phi
            phi = phi @ phi
    if not np.all(np.isfinite(gramian)):
        msg = (
            f"the finite-horizon observability Gramian over T = {horizon} is not "
            "finite (an unstable mode grows beyond floating-point range); shorten the "
            "horizon"
        )
        raise ValueError(msg)
    return np.asarray(0.5 * (gramian + gramian.T), dtype=np.float64)


def _finite_horizon_discrete(
    a: npt.NDArray[np.float64],
    information: npt.NDArray[np.float64],
    n_steps: int,
) -> npt.NDArray[np.float64]:
    """Discrete finite-horizon Gramian ``sum_k (A^T)^k M A^k`` by direct summation.

    Raises
    ------
    ValueError
        If the resulting Gramian is not finite.
    """
    gramian = np.zeros_like(information)
    a_power = np.eye(a.shape[0], dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(n_steps):
            gramian += a_power.T @ information @ a_power
            a_power = a_power @ a
    if not np.all(np.isfinite(gramian)):
        msg = (
            f"the finite-horizon observability Gramian over {n_steps} steps is not "
            "finite (an unstable mode grows beyond floating-point range); shorten the "
            "horizon"
        )
        raise ValueError(msg)
    return np.asarray(0.5 * (gramian + gramian.T), dtype=np.float64)


def _gramian_from_information(
    a: npt.NDArray[np.float64],
    information: npt.NDArray[np.float64],
    dt: float | None,
    horizon: float | None,
) -> npt.NDArray[np.float64]:
    """Dispatch to the infinite- or finite-horizon Gramian for an information source.

    ``horizon`` is ``None`` for the infinite-horizon (Lyapunov) Gramian, which assumes a
    stable system; otherwise the finite-horizon Gramian over ``[0, horizon]``.
    """
    if horizon is None:
        return _solve_gramian(a, information, dt)
    if dt is None:
        return _finite_horizon_continuous(a, information, horizon)
    n_steps = max(1, round(horizon / dt))
    return _finite_horizon_discrete(a, information, n_steps)


def _require_horizon_or_stable(
    a: npt.NDArray[np.float64], dt: float | None, horizon: float | None
) -> None:
    """Validate ``horizon`` and require stability for the infinite-horizon case.

    The infinite-horizon (Lyapunov) Gramian is requested with ``horizon=None``; a given
    ``horizon`` is a *finite* window and so must be positive and finite — an infinite
    value is not "infinite horizon" here, it would diverge the finite-horizon math
    (``expm`` of an unbounded window, or ``round(inf / dt)``).
    """
    if horizon is not None:
        if not (np.isfinite(horizon) and horizon > 0.0):
            msg = f"horizon must be a positive, finite time window, got {horizon}"
            raise ValueError(msg)
        return
    if not _is_stable(a, dt):
        msg = (
            "an infinite-horizon Gramian requires an asymptotically stable system; "
            "pass a finite horizon for a marginal or unstable system"
        )
        raise ValueError(msg)


def _resolve_output_row(system: StateSpace, output: int | str) -> int:
    """Resolve an output index or name to a ``C`` row (exact name, else substring).

    A substring matching several channels is rejected rather than resolved to the
    first, so an assessment is never issued for an unintended channel.
    """
    return _resolve_output(system, output)


def observability_scores(
    system: StateSpace,
    *,
    mode_states: Sequence[Sequence[int]] | None = None,
    measurement_noise: npt.NDArray[np.float64] | None = None,
    horizon: float | None = None,
) -> ObservabilityScores:
    r"""Break the observability Gramian down per output and per mode.

    Two diagnostics drawn from one Gramian:

    * **per output** — for each output row :math:`\mathbf c_i`, the trace of the Gramian
      built from that row alone, i.e. the observability energy that single sensor
      provides. This ranks candidate sensors by how much they see.
    * **per mode** — for each state group in ``mode_states``, the trace of the Gramian's
      diagonal block over those states. For the real modal realization of
      :func:`~vane.sysid.modal_reduction` (two states per mode, groups
      ``[[0, 1], [2, 3], ...]``) this diagonal block equals that mode's own
      single-mode Gramian exactly, so it measures how strongly each identified mode is
      observed — connecting the modal content to the sensor layout.

    Parameters
    ----------
    system : StateSpace
        A system with a ``C`` matrix. Must be asymptotically stable unless a finite
        ``horizon`` is given.
    mode_states : Sequence[Sequence[int]] or None, optional
        State-index groups, one per mode. ``None`` skips the per-mode breakdown.
    measurement_noise : npt.NDArray[np.float64] or None, optional
        Measurement-noise covariance ``R``; per-output scores use each channel's own
        variance ``R[i, i]``; per-mode scores use the full ``C^T R^-1 C`` information.
    horizon : float or None, optional
        Finite sensing window ``T``; ``None`` uses the infinite-horizon Gramian (which
        requires a stable system). See :func:`finite_horizon_observability_gramian`.

    Returns
    -------
    ObservabilityScores
        The per-output and (optional) per-mode observability energies.

    Raises
    ------
    ValueError
        If the system has no ``C`` matrix, ``horizon`` is non-positive, the system is
        unstable with no finite horizon, ``measurement_noise`` is the wrong shape, or a
        mode-state index is out of range.
    """
    if system.c is None:
        msg = "system has no C matrix; cannot score observability"
        raise ValueError(msg)
    a = np.asarray(system.a, dtype=np.float64)
    c = system.c
    _require_horizon_or_stable(a, system.dt, horizon)

    r = None
    if measurement_noise is not None:
        r = np.asarray(measurement_noise, dtype=np.float64)
        # Validate the full covariance here too: this per-output path weights each
        # channel by its own 1 / R[i, i] and never reaches _output_information, so
        # without this a singular/indefinite/asymmetric R would yield inf / negative /
        # silently-accepted scores. A positive-definite R guarantees R[i, i] > 0.
        _cholesky_of_noise(r, c.shape[0])

    per_output = np.empty(c.shape[0], dtype=np.float64)
    for i in range(c.shape[0]):
        row = c[i : i + 1]
        weight = 1.0 if r is None else 1.0 / float(r[i, i])
        info_i = weight * (row.T @ row)
        per_output[i] = float(
            np.trace(_gramian_from_information(a, info_i, system.dt, horizon))
        )

    per_mode: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
    if mode_states is not None:
        information = _output_information(c, measurement_noise)
        gramian = _gramian_from_information(a, information, system.dt, horizon)
        scores = []
        for group in mode_states:
            idx = list(group)
            if any(not 0 <= s < a.shape[0] for s in idx):
                msg = f"mode_states index group {idx} out of range [0, {a.shape[0]})"
                raise ValueError(msg)
            scores.append(float(np.trace(gramian[np.ix_(idx, idx)])))
        per_mode = np.asarray(scores, dtype=np.float64)

    names = [
        system.output_names[i] if i < len(system.output_names) else ""
        for i in range(c.shape[0])
    ]
    return ObservabilityScores(
        per_output=per_output, output_names=names, per_mode=per_mode
    )


def load_reconstructability(
    system: StateSpace,
    load_output: int | str,
    *,
    sensor_outputs: Sequence[int] | None = None,
    measurement_noise: npt.NDArray[np.float64] | None = None,
    horizon: float | None = None,
) -> LoadReconstructability:
    r"""Assess whether a target load is reconstructable from a sensor set.

    A load estimator measures a few sensor channels and reports a different output — the
    *load*. That load is recoverable only if its output direction
    :math:`\mathbf c_{\text{load}}` lies in the **observable subspace** of the sensors,
    the range of the sensor observability Gramian :math:`\mathbf W_o` (the orthogonal
    complement of the unobservable subspace, :math:`\ker\mathbf W_o`). This function
    reports that projection and the a-priori reconstruction variance — an assessment the
    downstream twin can check before running any filter.

    Whether the load is reconstructable, and its ``noise_amplification``, are
    independent of the state basis. The *size* of a partial projection
    (``reconstructable_fraction`` below ``1``) and ``observability`` are measured in
    the Euclidean metric of the state coordinates and change under a similarity
    transform or state rescaling; compare sensor sets by ``noise_amplification``.

    Parameters
    ----------
    system : StateSpace
        A system with a ``C`` matrix whose rows include both the sensors and the load.
        Must be asymptotically stable unless a finite ``horizon`` is given.
    load_output : int or str
        The target load output (row index, or a name matched exactly then by substring).
    sensor_outputs : Sequence[int] or None, optional
        The output rows used as measurements. ``None`` uses every output row **except
        the load row** (so the load is never measured to reconstruct itself). An
        explicit set must not contain ``load_output``.
    measurement_noise : npt.NDArray[np.float64] or None, optional
        Measurement-noise covariance ``R`` of the **sensor** rows (shape
        ``(n_sensors, n_sensors)``). When given, the Gramian is the Fisher information
        and ``noise_amplification`` is the best achievable static reconstruction error.
    horizon : float or None, optional
        Finite sensing window ``T``; ``None`` uses the infinite-horizon Gramian, which
        requires a stable system.

    Returns
    -------
    LoadReconstructability
        The reconstructable fraction, blind fraction, observed energy, a-priori variance
        scale, and the observable-subspace dimension.

    Raises
    ------
    ValueError
        If the system has no ``C`` matrix, the load row is zero, an index is out of
        range, ``sensor_outputs`` includes the load output or leaves no sensor rows,
        ``horizon`` is non-positive, the system is unstable with no finite horizon, or
        ``measurement_noise`` is the wrong shape.
    """
    if system.c is None:
        msg = "system has no C matrix; cannot assess reconstructability"
        raise ValueError(msg)
    a = np.asarray(system.a, dtype=np.float64)
    c = system.c
    _require_horizon_or_stable(a, system.dt, horizon)

    load_row = _resolve_output_row(system, load_output)
    c_load = np.asarray(c[load_row], dtype=np.float64)
    load_norm_sq = float(c_load @ c_load)
    if load_norm_sq <= 0.0:
        msg = (
            f"load output {load_output!r} has a zero output row; nothing to reconstruct"
        )
        raise ValueError(msg)

    if sensor_outputs is None:
        # Every output row *except the load itself* — measuring the target load to
        # "reconstruct" it is not virtual sensing and would trivially assess any load.
        rows = [i for i in range(c.shape[0]) if i != load_row]
    else:
        rows = list(dict.fromkeys(sensor_outputs))
        if any(not 0 <= i < c.shape[0] for i in rows):
            msg = f"sensor_outputs {rows} out of range [0, {c.shape[0]})"
            raise ValueError(msg)
        if load_row in rows:
            msg = (
                f"sensor_outputs must not include the load output (row {load_row}); "
                "reconstructing a load from a direct measurement of itself is not "
                "virtual sensing"
            )
            raise ValueError(msg)
    if not rows:
        msg = (
            "no sensor outputs remain after excluding the load output; provide at "
            "least one non-load output row"
        )
        raise ValueError(msg)
    information = _output_information(c[rows], measurement_noise)
    gramian = _gramian_from_information(a, information, system.dt, horizon)

    eigvals, eigvecs = np.linalg.eigh(gramian)
    tol = max(eigvals.max(), 0.0) * a.shape[0] * np.finfo(np.float64).eps
    observable = eigvals > tol
    proj = eigvecs[:, observable].T @ c_load  # load components in the observable basis
    reconstructable = float(proj @ proj) / load_norm_sq
    reconstructable = min(1.0, max(0.0, reconstructable))
    observability = float(c_load @ gramian @ c_load) / load_norm_sq
    # The variance is the inverse-information cost over the observable part, but any
    # component in the blind subspace is irrecoverable at any noise level (infinite
    # variance). Report inf so a smaller-is-better ranking never prefers a blind load.
    if reconstructable < 1.0 - 1e-9:
        noise_amplification = float("inf")
    else:
        noise_amplification = float(proj**2 @ (1.0 / eigvals[observable]))

    return LoadReconstructability(
        load_name=system.output_names[load_row]
        if load_row < len(system.output_names) and system.output_names[load_row]
        else f"output[{load_row}]",
        reconstructable_fraction=reconstructable,
        blind_fraction=1.0 - reconstructable,
        observability=observability,
        noise_amplification=noise_amplification,
        n_observable_directions=int(np.count_nonzero(observable)),
    )


def _logdet(gramian: npt.NDArray[np.float64], floor: float) -> float:
    """Regularised log-determinant of a symmetric PSD matrix."""
    eigvals = np.linalg.eigvalsh(gramian)
    return float(np.sum(np.log(np.maximum(eigvals, 0.0) + floor)))


def _placement_gramian(
    system: StateSpace,
    rows: Sequence[int],
    measurement_noise: npt.NDArray[np.float64] | None,
    horizon: float | None,
) -> npt.NDArray[np.float64]:
    """Observability Gramian of a sensor subset, noise-weighted and horizon-aware.

    ``measurement_noise`` is the full ``(n_outputs, n_outputs)`` measurement covariance;
    the block ``R[rows, rows]`` (the marginal covariance of the selected channels)
    weights their information, so a noisier sensor contributes less. ``horizon`` selects
    the finite-window Gramian (the only one defined for marginal/augmented models).
    """
    c = system.c
    if c is None:  # callers validate this; narrow the type for the row indexing below
        msg = "system has no C matrix"
        raise ValueError(msg)
    r_block = (
        None
        if measurement_noise is None
        else measurement_noise[np.ix_(list(rows), list(rows))]
    )
    information = _output_information(c[list(rows)], r_block)
    return _gramian_from_information(system.a, information, system.dt, horizon)


def recommend_sensors(
    system: StateSpace,
    *,
    n_sensors: int,
    candidates: Sequence[int] | None = None,
    measurement_noise: npt.NDArray[np.float64] | None = None,
    horizon: float | None = None,
) -> SensorPlacement:
    """Recommend sensors that best observe a system, by greedy D-optimal selection.

    Greedily adds, from the candidate output channels, the one that most increases the
    log-determinant of the observability Gramian built from the selected rows of ``C``.
    D-optimality rewards covering all observable directions, so it prefers complementary
    sensors over redundant ones. Intended for the reduced model (see
    :func:`~vane.sysid.modal_reduction`), where the candidates are the instrumentable
    channels and the modes are those that drive the load of interest.

    Parameters
    ----------
    system : StateSpace
        A system with a ``C`` matrix. Must be asymptotically stable unless a finite
        ``horizon`` is given.
    n_sensors : int
        Number of sensors to recommend (capped at the number of candidates).
    candidates : Sequence[int] or None, optional
        Output row indices that may be instrumented. ``None`` considers every output.
    measurement_noise : npt.NDArray[np.float64] or None, optional
        Full ``(n_outputs, n_outputs)`` measurement-noise covariance ``R``. When given,
        each selected sensor is weighted by its inverse noise (the marginal block
        ``R[rows, rows]``), so a noisy channel ranks below a calibrated one instead of
        scoring identically. ``None`` weights every channel equally.
    horizon : float or None, optional
        Finite sensing window ``T``. ``None`` uses the infinite-horizon Gramian (which
        requires a stable system); a finite horizon also admits marginal/augmented
        models whose infinite-horizon Gramian does not exist.

    Returns
    -------
    SensorPlacement
        The selected sensors (in order), their names, and the observability metric
        after each selection.

    Raises
    ------
    ValueError
        If the system has no ``C`` matrix, is unstable with no finite ``horizon``,
        ``n_sensors`` is not positive, a candidate index is out of range, or
        ``measurement_noise`` is not a symmetric positive-definite ``(n_outputs,
        n_outputs)`` matrix.
    """
    if system.c is None:
        msg = "system has no C matrix; cannot place sensors"
        raise ValueError(msg)
    _require_horizon_or_stable(system.a, system.dt, horizon)
    if n_sensors <= 0:
        msg = f"n_sensors must be positive, got {n_sensors}"
        raise ValueError(msg)
    c = system.c
    if measurement_noise is not None:
        measurement_noise = np.asarray(measurement_noise, dtype=np.float64)
        _cholesky_of_noise(measurement_noise, c.shape[0])
    # Deduplicate candidates while preserving order: a repeated index would otherwise
    # let the greedy loop select the same sensor twice and double-count its C row,
    # overstating observability instead of returning unique sensors.
    pool = (
        list(range(c.shape[0]))
        if candidates is None
        else list(dict.fromkeys(candidates))
    )
    if any(not 0 <= i < c.shape[0] for i in pool):
        msg = f"candidate indices {pool} out of range [0, {c.shape[0]})"
        raise ValueError(msg)

    # Scale-aware regulariser so the log-determinant is finite while rows are sparse.
    reference = _placement_gramian(system, pool, measurement_noise, horizon)
    floor = max(1e-12, 1e-9 * float(np.max(np.diag(reference))))

    selected: list[int] = []
    scores: list[float] = []
    remaining = list(pool)
    for _ in range(min(n_sensors, len(pool))):
        best_index, best_score = remaining[0], -np.inf
        for candidate in remaining:
            rows = [*selected, candidate]
            gramian = _placement_gramian(system, rows, measurement_noise, horizon)
            score = _logdet(gramian, floor)
            if score > best_score:
                best_score, best_index = score, candidate
        selected.append(best_index)
        remaining.remove(best_index)
        scores.append(best_score)

    names = [
        system.output_names[i] if i < len(system.output_names) else "" for i in selected
    ]
    return SensorPlacement(selected=selected, names=names, scores=scores)


def constrained_sensor_selection(
    system: StateSpace,
    *,
    candidates: Sequence[int] | None = None,
    required: Sequence[int] = (),
    forbidden: Sequence[int] = (),
    costs: Sequence[float] | None = None,
    budget: float | None = None,
    max_sensors: int | None = None,
    groups: Sequence[GroupConstraint] | None = None,
    measurement_noise: npt.NDArray[np.float64] | None = None,
    horizon: float | None = None,
) -> ConstrainedPlacement:
    """Greedy D-optimal sensor selection under deployment constraints.

    Extends :func:`recommend_sensors` with the constraints a real instrumentation plan
    faces: sensors that must or must not be used, a per-sensor cost and a total budget,
    a cap on the total count, and caps on how many sensors may come from each group
    (e.g. one component, or a set of redundant channels). Required sensors are taken
    first; the rest are added greedily by the largest log-determinant gain among the
    channels that keep every constraint satisfied, until none remain feasible.

    Parameters
    ----------
    system : StateSpace
        A system with a ``C`` matrix. Must be asymptotically stable unless a finite
        ``horizon`` is given.
    candidates : Sequence[int] or None, optional
        Output row indices that may be instrumented. ``None`` considers every output.
    required : Sequence[int], optional
        Output rows that must be selected (taken first, even if outside ``candidates``).
    forbidden : Sequence[int], optional
        Output rows that must never be selected.
    costs : Sequence[float] or None, optional
        Per-output cost (length ``n_outputs``), all non-negative. ``None`` treats every
        sensor as free (cost 0).
    budget : float or None, optional
        Maximum total cost of the selection. ``None`` imposes no budget.
    max_sensors : int or None, optional
        Maximum number of sensors. ``None`` imposes no count cap (the greedy loop then
        runs until no feasible candidate remains).
    groups : Sequence[GroupConstraint] or None, optional
        Per-group caps on the number of selected sensors.
    measurement_noise : npt.NDArray[np.float64] or None, optional
        Full ``(n_outputs, n_outputs)`` measurement-noise covariance ``R``; each
        selected sensor is weighted by its inverse noise so a noisy channel ranks below
        a calibrated one. ``None`` weights every channel equally.
    horizon : float or None, optional
        Finite sensing window ``T``. ``None`` uses the infinite-horizon Gramian (stable
        systems only); a finite horizon also admits marginal/augmented models.

    Returns
    -------
    ConstrainedPlacement
        The selected sensors (required first, then greedily added), their names, the
        observability metric after each selection, and the total cost.

    Raises
    ------
    ValueError
        If the system has no ``C`` matrix, is unstable with no finite ``horizon``, an
        index is out of range, ``required`` and ``forbidden`` overlap, a
        cost/budget/cap is negative, ``measurement_noise`` is not a symmetric
        positive-definite ``(n_outputs, n_outputs)`` matrix, or the required sensors
        alone already violate the budget, count cap, or a group cap.
    """
    if system.c is None:
        msg = "system has no C matrix; cannot place sensors"
        raise ValueError(msg)
    _require_horizon_or_stable(system.a, system.dt, horizon)
    c = system.c
    n_out = c.shape[0]
    if measurement_noise is not None:
        measurement_noise = np.asarray(measurement_noise, dtype=np.float64)
        _cholesky_of_noise(measurement_noise, n_out)
    required = list(dict.fromkeys(required))
    forbidden_set = set(forbidden)
    pool_in = (
        list(range(n_out)) if candidates is None else list(dict.fromkeys(candidates))
    )
    for label, indices in (
        ("candidates", pool_in),
        ("required", required),
        ("forbidden", list(forbidden_set)),
    ):
        if any(not 0 <= i < n_out for i in indices):
            msg = f"{label} indices out of range [0, {n_out})"
            raise ValueError(msg)
    if set(required) & forbidden_set:
        msg = "required and forbidden sensors overlap"
        raise ValueError(msg)

    cost_of = _build_costs(costs, n_out)
    if budget is not None and not budget >= 0.0:  # also rejects NaN
        msg = f"budget must be a non-negative number, got {budget}"
        raise ValueError(msg)
    if max_sensors is not None:
        _require_count_cap(max_sensors, "max_sensors", minimum=1)
    group_list = _validate_groups(groups, n_out)

    # The greedy pool excludes forbidden and already-required channels.
    pool = [i for i in pool_in if i not in forbidden_set and i not in required]

    # Required sensors are forced in; reject if they alone break a hard constraint.
    required_cost = float(sum(cost_of[i] for i in required))
    if budget is not None and required_cost > budget:
        msg = f"required sensors cost {required_cost} exceed the budget {budget}"
        raise ValueError(msg)
    if max_sensors is not None and len(required) > max_sensors:
        msg = f"required sensors ({len(required)}) exceed max_sensors ({max_sensors})"
        raise ValueError(msg)
    for group in group_list:
        used = len(group.members & set(required))
        if used > group.max_count:
            msg = f"required sensors use {used} of a group capped at {group.max_count}"
            raise ValueError(msg)

    reference_rows = sorted(set(required) | set(pool)) or [0]
    reference = _placement_gramian(system, reference_rows, measurement_noise, horizon)
    floor = max(1e-12, 1e-9 * float(np.max(np.diag(reference))))

    def logdet_of(rows: list[int]) -> float:
        gramian = _placement_gramian(system, rows, measurement_noise, horizon)
        return _logdet(gramian, floor)

    selected: list[int] = []
    scores: list[float] = []
    cost_so_far = 0.0
    for sensor in required:
        selected.append(sensor)
        cost_so_far += cost_of[sensor]
        scores.append(logdet_of(selected))

    def feasible(candidate: int) -> bool:
        if max_sensors is not None and len(selected) + 1 > max_sensors:
            return False
        if budget is not None and cost_so_far + cost_of[candidate] > budget:
            return False
        return all(
            not (
                candidate in group.members
                and len(group.members & set(selected)) + 1 > group.max_count
            )
            for group in group_list
        )

    remaining = list(pool)
    while True:
        feasible_now = [x for x in remaining if feasible(x)]
        if not feasible_now:
            break
        best_index, best_score = feasible_now[0], -np.inf
        for candidate in feasible_now:
            score = logdet_of([*selected, candidate])
            if score > best_score:
                best_score, best_index = score, candidate
        selected.append(best_index)
        remaining.remove(best_index)
        cost_so_far += cost_of[best_index]
        scores.append(best_score)

    names = [
        system.output_names[i] if i < len(system.output_names) else "" for i in selected
    ]
    return ConstrainedPlacement(
        selected=selected, names=names, scores=scores, total_cost=cost_so_far
    )


def _require_count_cap(value: float, name: str, *, minimum: int) -> None:
    """Validate that a count cap is an integer-valued number at or above ``minimum``."""
    if not (np.isfinite(value) and value >= minimum and float(value).is_integer()):
        msg = f"{name} must be an integer >= {minimum}, got {value}"
        raise ValueError(msg)


def _build_costs(costs: Sequence[float] | None, n_out: int) -> npt.NDArray[np.float64]:
    """Validate and return a length-``n_out`` non-negative per-output cost array."""
    if costs is None:
        return np.zeros(n_out, dtype=np.float64)
    values = np.asarray(costs, dtype=np.float64)
    if values.shape != (n_out,):
        msg = f"costs must have one value per output ({n_out}), got {values.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        msg = "costs must be finite and non-negative"
        raise ValueError(msg)
    return values


def _validate_groups(
    groups: Sequence[GroupConstraint] | None, n_out: int
) -> list[_Group]:
    """Validate group constraints and return them with member sets resolved."""
    resolved: list[_Group] = []
    for group in groups or ():
        _require_count_cap(group.max_count, "group max_count", minimum=0)
        members = set(group.members)
        if any(not 0 <= i < n_out for i in members):
            msg = f"group members {sorted(members)} out of range [0, {n_out})"
            raise ValueError(msg)
        resolved.append(_Group(members=members, max_count=group.max_count))
    return resolved


@dataclass
class _Group:
    """A group constraint with its members resolved to a set for fast membership."""

    members: set[int]
    max_count: int


def reduced_estimator(
    reduction: ModalReduction,
    *,
    sensor_outputs: Sequence[int],
    dt: float,
    confidences: Sequence[float] | None = None,
    process_noise: float = 1e-4,
    measurement_noise: float = 1e-2,
    base_variance: float = 1.0,
    uncertainty_scale: float = 10.0,
) -> KalmanModel:
    """Assemble a UQ-initialized Kalman model on a reduced modal system.

    Builds a discrete Kalman model whose measurement matrix is the chosen sensor rows of
    the reduced system and whose **initial covariance is inflated by each retained
    mode's confidence** (a less-trusted mode starts with larger variance, two states per
    mode). This is the reduced load estimator: reduction + per-mode uncertainty + Kalman
    export, in one object ready for a filter.

    .. note::

       ``process_noise`` here is a **hand-set discrete** ``Q`` diagonal (a sensible
       starting placeholder), *not* a Van Loan discretization of a continuous spectral
       density: unlike the steady-state assessment path
       (:func:`~vane.sysid.discretize_process_noise`), it is used directly as ``Q_d``,
       so it is not scaled to ``dt`` or coupled through ``A``. For a model-consistent
       ``Q`` (a modal spectral density discretized against ``A``), build the model from
       :func:`~vane.sysid.discretize_process_noise` instead and tune from there.

    Parameters
    ----------
    reduction : ModalReduction
        A reduced modal model from :func:`~vane.sysid.modal_reduction`.
    sensor_outputs : Sequence[int]
        Output row indices of the reduced system used as measurements (the sensors;
        e.g. from :func:`recommend_sensors`).
    dt : float
        Sample time for the discrete estimator.
    confidences : Sequence[float] or None, optional
        Per-mode confidence in ``[0, 1]`` (one per retained mode); a lower value
        inflates that mode's initial covariance. ``None`` trusts every mode (all ones).
    process_noise, measurement_noise, base_variance, uncertainty_scale : float, optional
        Passed to :func:`~vane.sysid.covariances_from_confidence`.

    Returns
    -------
    KalmanModel
        The dimension-checked, discretized Kalman model.

    Raises
    ------
    ValueError
        If a sensor index is out of range, or ``confidences`` has the wrong length.
    """
    reduced = reduction.system
    if reduced.c is None:
        msg = "reduced system has no C matrix"
        raise ValueError(msg)
    rows = list(sensor_outputs)
    if any(not 0 <= i < reduced.c.shape[0] for i in rows):
        msg = f"sensor_outputs {rows} out of range [0, {reduced.c.shape[0]})"
        raise ValueError(msg)
    n_modes = reduction.frequencies_hz.shape[0]
    if confidences is None:
        confidences = [1.0] * n_modes
    elif len(confidences) != n_modes:
        msg = f"confidences need one value per mode ({n_modes}), got {len(confidences)}"
        raise ValueError(msg)

    names = reduced.output_names
    measurement = StateSpace(
        a=reduced.a,
        b=reduced.b,
        c=reduced.c[rows],
        d=None if reduced.d is None else reduced.d[rows],
        state_names=list(reduced.state_names),
        input_names=list(reduced.input_names),
        output_names=[names[i] if i < len(names) else "" for i in rows],
        dt=None,
    )
    covariances = covariances_from_confidence(
        confidences,
        n_outputs=len(rows),
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        base_variance=base_variance,
        uncertainty_scale=uncertainty_scale,
    )
    return build_kalman_model(measurement.discretized(dt), covariances)
