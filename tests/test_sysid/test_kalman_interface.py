# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.sysid.kalman_interface — Kalman model assembly."""

from __future__ import annotations

import numpy as np
import pytest

from vane.sysid.covariance_init import CovarianceSet, initialize_covariances
from vane.sysid.kalman_interface import build_kalman_model
from vane.sysid.state_space import StateSpace


def _system() -> StateSpace:
    """Return a stable two-state, one-output system."""
    return StateSpace(
        a=np.array([[0.0, 1.0], [-4.0, -0.4]]),
        b=np.array([[0.0], [1.0]]),
        c=np.array([[1.0, 0.0]]),
    )


class TestBuildKalmanModel:
    """Tests for build_kalman_model."""

    def test_assembles_consistent_model(self) -> None:
        """A matching system and covariance set assemble into a model."""
        model = build_kalman_model(_system(), initialize_covariances(2, 1))
        assert model.a.shape == (2, 2)
        assert model.q.shape == (2, 2)
        assert model.r.shape == (1, 1)
        assert model.dt is None

    def test_discretizes_when_dt_given(self) -> None:
        """A sample time discretizes the system during assembly."""
        model = build_kalman_model(_system(), initialize_covariances(2, 1), dt=0.1)
        assert model.dt == 0.1
        # Discrete A differs from the continuous A.
        assert not np.allclose(model.a, _system().a)

    def test_dt_on_already_discrete_system_raises(self) -> None:
        """Passing dt for an already-discrete system raises instead of ignoring it."""
        discrete = _system().discretized(0.1)
        with pytest.raises(ValueError, match="already discrete"):
            build_kalman_model(discrete, initialize_covariances(2, 1), dt=0.05)

    def test_q_dimension_mismatch_raises(self) -> None:
        """A process-covariance dimension mismatch raises ValueError."""
        bad = CovarianceSet(q=np.eye(3), r=np.eye(1), p0=np.eye(2))
        with pytest.raises(ValueError, match="Q must have"):
            build_kalman_model(_system(), bad)

    def test_r_dimension_mismatch_raises(self) -> None:
        """A measurement-covariance dimension mismatch raises ValueError."""
        bad = CovarianceSet(q=np.eye(2), r=np.eye(2), p0=np.eye(2))
        with pytest.raises(ValueError, match="R must have"):
            build_kalman_model(_system(), bad)

    def test_r_validated_even_without_outputs(self) -> None:
        """A no-output system (C is None) still rejects a non-empty R."""
        no_output = StateSpace(a=np.array([[-1.0, 0.0], [0.0, -2.0]]))
        bad = CovarianceSet(q=np.eye(2), r=np.eye(1), p0=np.eye(2))
        with pytest.raises(ValueError, match="R must have"):
            build_kalman_model(no_output, bad)

    def test_negative_q_rejected(self) -> None:
        """A negative-(semi)definite Q is rejected, not just shape-checked."""
        bad = CovarianceSet(q=-np.eye(2), r=np.eye(1), p0=np.eye(2))
        with pytest.raises(ValueError, match="Q must be positive semidefinite"):
            build_kalman_model(_system(), bad)

    def test_asymmetric_p0_rejected(self) -> None:
        """An asymmetric P0 is rejected as not a covariance."""
        bad = CovarianceSet(
            q=np.eye(2), r=np.eye(1), p0=np.array([[1.0, 2.0], [0.0, 1.0]])
        )
        with pytest.raises(ValueError, match="P0 must be symmetric"):
            build_kalman_model(_system(), bad)

    def test_non_finite_q_rejected(self) -> None:
        """A non-finite Q is rejected."""
        q = np.eye(2)
        q[0, 0] = np.nan
        bad = CovarianceSet(q=q, r=np.eye(1), p0=np.eye(2))
        with pytest.raises(ValueError, match="Q must be finite"):
            build_kalman_model(_system(), bad)

    def test_singular_r_rejected(self) -> None:
        """A positive-semidefinite-but-singular R is rejected (needs PD)."""
        bad = CovarianceSet(q=np.eye(2), r=np.zeros((1, 1)), p0=np.eye(2))
        with pytest.raises(ValueError, match="R must be positive definite"):
            build_kalman_model(_system(), bad)

    def test_carries_names(self) -> None:
        """State and output names carry through to the model."""
        system = StateSpace(
            a=np.array([[-1.0]]),
            c=np.array([[1.0]]),
            state_names=["x"],
            output_names=["y"],
        )
        model = build_kalman_model(system, initialize_covariances(1, 1))
        assert model.state_names == ["x"]
        assert model.output_names == ["y"]
