# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Ready-to-use Kalman-filter model assembly.

A :class:`KalmanModel` bundles a state-space system with its process,
measurement, and initial covariances into the exact matrices a Kalman filter
consumes. The system can be discretized at a chosen sample time during assembly,
and the covariance dimensions are validated against the system so the resulting
model is internally consistent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from vane.sysid._covariance import (
    is_positive_definite,
    is_positive_semidefinite,
    is_symmetric,
    symmetric_part,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from vane.sysid.covariance_init import CovarianceSet
    from vane.sysid.state_space import StateSpace

__all__ = ["KalmanModel", "build_kalman_model"]


@dataclass
class KalmanModel:
    """A state-space system with Kalman covariances, ready for filtering.

    Parameters
    ----------
    a : npt.NDArray[np.float64]
        State matrix, shape ``(n_states, n_states)``.
    b, c, d : npt.NDArray[np.float64] or None
        Input, output, and feedthrough matrices.
    q, r, p0 : npt.NDArray[np.float64]
        Process-noise, measurement-noise, and initial state covariances.
    dt : float or None
        Sample time; ``None`` for a continuous-time model.
    state_names, output_names : list[str]
        State and output names carried from the source system.
    """

    a: npt.NDArray[np.float64]
    b: npt.NDArray[np.float64] | None
    c: npt.NDArray[np.float64] | None
    d: npt.NDArray[np.float64] | None
    q: npt.NDArray[np.float64]
    r: npt.NDArray[np.float64]
    p0: npt.NDArray[np.float64]
    dt: float | None = None
    state_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)


def build_kalman_model(
    state_space: StateSpace,
    covariances: CovarianceSet,
    *,
    dt: float | None = None,
) -> KalmanModel:
    """Assemble a Kalman model from a state-space system and covariances.

    Parameters
    ----------
    state_space : StateSpace
        The continuous- or discrete-time system.
    covariances : CovarianceSet
        The process, measurement, and initial covariances. ``Q`` is used exactly as
        given and must already be the discrete per-step covariance ``Q_d`` for the
        model's sample time.
    dt : float or None, optional
        When given and the system is continuous, discretize ``A``/``B`` at this
        sample time before assembly. ``Q`` is **not** discretized: convert a
        continuous-time ``Q_c`` (e.g. from :func:`vane.sysid.modal_process_noise`)
        with :func:`vane.sysid.discretize_process_noise` at the same ``dt`` first.

    Returns
    -------
    KalmanModel
        The assembled, dimension-checked model.

    Raises
    ------
    ValueError
        If ``dt`` is given for an already-discrete system, the covariance dimensions
        do not match the system, or a covariance is not a valid covariance matrix
        (``Q``/``P0`` must be finite, symmetric, and positive semidefinite; ``R`` must
        be finite, symmetric, and positive definite).
    """
    if dt is not None and state_space.is_discrete:
        msg = (
            f"state_space is already discrete (dt={state_space.dt}); cannot "
            f"re-discretize with dt={dt}"
        )
        raise ValueError(msg)
    system = state_space.discretized(dt) if dt is not None else state_space
    _validate_covariance_dimensions(system, covariances)
    _validate_covariance_values(covariances)
    return KalmanModel(
        a=system.a,
        b=system.b,
        c=system.c,
        d=system.d,
        # Store the symmetric matrices that were validated.
        q=symmetric_part(np.asarray(covariances.q, dtype=np.float64)),
        r=symmetric_part(np.asarray(covariances.r, dtype=np.float64)),
        p0=symmetric_part(np.asarray(covariances.p0, dtype=np.float64)),
        dt=system.dt,
        state_names=list(system.state_names),
        output_names=list(system.output_names),
    )


def _validate_covariance_dimensions(
    system: StateSpace, covariances: CovarianceSet
) -> None:
    """Validate that covariance shapes match the system dimensions."""
    n = system.n_states
    if covariances.q.shape != (n, n):
        msg = f"Q must have shape {(n, n)}, got {covariances.q.shape}"
        raise ValueError(msg)
    if covariances.p0.shape != (n, n):
        msg = f"P0 must have shape {(n, n)}, got {covariances.p0.shape}"
        raise ValueError(msg)
    # Validate R against the output count unconditionally: a system without C has
    # zero outputs, so a non-empty R is an inconsistency rather than skipped.
    p = system.n_outputs
    if covariances.r.shape != (p, p):
        msg = f"R must have shape {(p, p)}, got {covariances.r.shape}"
        raise ValueError(msg)


def _require_symmetric_finite(matrix: npt.NDArray[np.float64], name: str) -> None:
    """Raise unless ``matrix`` is finite and symmetric."""
    if not np.all(np.isfinite(matrix)):
        msg = f"{name} must be finite (no NaN or inf) to be a covariance"
        raise ValueError(msg)
    if not is_symmetric(matrix):
        msg = f"{name} must be symmetric to be a covariance"
        raise ValueError(msg)


def _validate_covariance_values(covariances: CovarianceSet) -> None:
    """Validate that ``Q``, ``R``, and ``P0`` are valid covariances, not just shaped.

    ``Q`` and ``P0`` must be finite, symmetric, and positive semidefinite; ``R`` must be
    finite, symmetric, and positive definite, since a measurement update factorizes the
    innovation covariance and a singular or indefinite ``R`` would corrupt it. A
    malformed covariance is rejected here rather than yielding invalid uncertainty or
    failing later inside the filter's Cholesky with a less actionable error.
    """
    for matrix, name in ((covariances.q, "Q"), (covariances.p0, "P0")):
        _require_symmetric_finite(matrix, name)
        if not is_positive_semidefinite(matrix):
            msg = f"{name} must be positive semidefinite to be a covariance"
            raise ValueError(msg)
    _require_symmetric_finite(covariances.r, "R")
    if not is_positive_definite(covariances.r):
        msg = (
            "R must be positive definite (every measurement channel needs a "
            "strictly positive, non-singular variance); got a singular or "
            "indefinite covariance"
        )
        raise ValueError(msg)
