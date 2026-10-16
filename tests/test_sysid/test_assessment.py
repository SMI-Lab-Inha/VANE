# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.sysid.assessment — steady-state estimator assessment."""

import numpy as np
import pytest
from scipy.linalg import block_diag

from vane.sysid import (
    EstimatorAssessment,
    StateSpace,
    assess_load,
    steady_state_estimator,
)


def _block(freq: float, zeta: float) -> np.ndarray:
    """Return a stable continuous 2x2 modal block."""
    wn = 2.0 * np.pi * freq
    sigma = -zeta * wn
    wd = wn * np.sqrt(1.0 - zeta**2)
    return np.array([[sigma, -wd], [wd, sigma]])


def _discrete_system() -> StateSpace:
    """Return a discrete 4-state, 1-output system (one sensor sees both modes)."""
    a = np.asarray(block_diag(_block(0.3, 0.05), _block(1.5, 0.05)))
    c = np.array([[1.0, 0.0, 1.0, 0.0]])
    return StateSpace(a=a, c=c, output_names=["y"]).discretized(0.05)


_QD = np.diag([1e-3, 1e-3, 1e-3, 1e-3])
_R = np.array([[2.5e-3]])


class TestSteadyStateEstimator:
    """Tests for steady_state_estimator."""

    def test_solves_the_riccati_fixed_point(self) -> None:
        """P- = Ad P+ Ad^T + Qd and P+ = (I - K C) P- at the steady state."""
        ss = _discrete_system()
        cert = steady_state_estimator(ss, process_noise=_QD, measurement_noise=_R)
        assert ss.c is not None
        ad, c = ss.a, ss.c
        np.testing.assert_allclose(
            cert.predicted_covariance,
            ad @ cert.error_covariance @ ad.T + _QD,
            atol=1e-10,
        )
        gain_check = (np.eye(4) - cert.gain @ c) @ cert.predicted_covariance
        np.testing.assert_allclose(cert.error_covariance, gain_check, atol=1e-10)

    def test_covariances_are_symmetric_psd(self) -> None:
        """Both steady-state covariances are symmetric positive-semidefinite."""
        cert = steady_state_estimator(
            _discrete_system(), process_noise=_QD, measurement_noise=_R
        )
        for cov in (cert.error_covariance, cert.predicted_covariance):
            np.testing.assert_allclose(cov, cov.T, atol=1e-12)
            assert np.all(np.linalg.eigvalsh(cov) >= -1e-12)

    def test_state_std_is_sqrt_diag(self) -> None:
        """The per-state std is the square root of the filtered-covariance diagonal."""
        cert = steady_state_estimator(
            _discrete_system(), process_noise=_QD, measurement_noise=_R
        )
        np.testing.assert_allclose(
            cert.state_std, np.sqrt(np.diag(cert.error_covariance)), atol=1e-12
        )

    def test_filtering_reduces_uncertainty(self) -> None:
        """The measurement update never increases the error covariance."""
        cert = steady_state_estimator(
            _discrete_system(), process_noise=_QD, measurement_noise=_R
        )
        gap = cert.predicted_covariance - cert.error_covariance
        assert np.all(np.linalg.eigvalsh(gap) >= -1e-12)

    def test_rejects_continuous_system(self) -> None:
        """A continuous system has no discrete steady-state filter here."""
        cont = StateSpace(a=_block(0.3, 0.05), c=np.eye(2))
        with pytest.raises(ValueError, match="discrete-time"):
            steady_state_estimator(
                cont, process_noise=np.eye(2) * 1e-3, measurement_noise=np.eye(2)
            )

    def test_rejects_missing_c(self) -> None:
        """A system without a C matrix is rejected."""
        ss = StateSpace(a=_block(0.3, 0.05)).discretized(0.05)
        with pytest.raises(ValueError, match="no C matrix"):
            steady_state_estimator(
                ss, process_noise=np.eye(2) * 1e-3, measurement_noise=np.eye(1)
            )

    def test_rejects_bad_process_noise_shape(self) -> None:
        """A process-noise of the wrong shape is rejected."""
        with pytest.raises(ValueError, match="process_noise must be"):
            steady_state_estimator(
                _discrete_system(), process_noise=np.eye(3), measurement_noise=_R
            )

    def test_rejects_non_pd_measurement_noise(self) -> None:
        """A singular (not positive-definite) R is rejected."""
        with pytest.raises(ValueError, match="positive definite"):
            steady_state_estimator(
                _discrete_system(),
                process_noise=_QD,
                measurement_noise=np.zeros((1, 1)),
            )

    def test_rejects_undetectable_system(self) -> None:
        """An unstable mode unseen by the sensors has no steady-state filter."""
        ad = np.diag([2.0, 0.5])  # state 0 is unstable
        c = np.array([[0.0, 1.0]])  # sensor only sees the stable state 1
        ss = StateSpace(a=ad, c=c, dt=0.1)
        with pytest.raises(ValueError, match="not detectable"):
            steady_state_estimator(
                ss, process_noise=np.eye(2) * 1e-3, measurement_noise=np.array([[1e-2]])
            )

    def test_failure_message_names_stabilisability_too(self) -> None:
        """The DARE failure names both causes: detectability and stabilisability.

        A missing stabilising solution can equally come from a marginal/unstable mode
        with no process noise ((A, Q^1/2) not stabilisable); the message must not point
        the user only at the sensors.
        """
        ss = StateSpace(a=np.diag([2.0, 0.5]), c=np.array([[0.0, 1.0]]), dt=0.1)
        with pytest.raises(ValueError, match=r"\(A, Q\^1/2\) is not stabilisable"):
            steady_state_estimator(
                ss, process_noise=np.eye(2) * 1e-3, measurement_noise=np.array([[1e-2]])
            )


class TestAssessLoad:
    """Tests for assess_load."""

    def _assessment(self) -> EstimatorAssessment:
        """Return an assessment for the shared discrete system."""
        return steady_state_estimator(
            _discrete_system(), process_noise=_QD, measurement_noise=_R
        )

    def test_error_std_matches_quadratic_form(self) -> None:
        """Each load error std is sqrt(L P+ L^T)."""
        cert = self._assessment()
        load = np.array([[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 1.0]])
        results = assess_load(cert, load)
        expected = np.sqrt(np.diag(load @ cert.error_covariance @ load.T))
        np.testing.assert_allclose([r.error_std for r in results], expected, atol=1e-12)
        assert [r.load_name for r in results] == ["load[0]", "load[1]"]

    def test_tolerance_decides_acceptance(self) -> None:
        """A load passes when its error std is within tolerance, fails otherwise."""
        cert = self._assessment()
        load = np.array([[1.0, 0.0, 0.0, 0.0]])
        std = assess_load(cert, load)[0].error_std
        assert assess_load(cert, load, tolerance=std * 1.1)[0].within_tolerance
        assert not assess_load(cert, load, tolerance=std * 0.9)[0].within_tolerance

    def test_per_load_tolerance_and_names(self) -> None:
        """Per-load tolerances and names are honoured."""
        cert = self._assessment()
        load = np.eye(4)
        results = assess_load(
            cert,
            load,
            tolerance=[1.0, 1.0, 1.0, 1e-9],
            names=["a", "b", "c", "d"],
        )
        assert [r.load_name for r in results] == ["a", "b", "c", "d"]
        assert results[0].within_tolerance
        assert not results[3].within_tolerance  # 1e-9 tolerance cannot be met

    def test_rejects_non_finite_load_map(self) -> None:
        """A non-finite load map is rejected rather than assessing a NaN error."""
        cert = self._assessment()
        load = np.array([[1.0, np.inf, 0.0, 0.0]])
        with pytest.raises(ValueError, match="load_map must be finite"):
            assess_load(cert, load)

    def test_no_tolerance_reports_without_accepting(self) -> None:
        """Without a tolerance the error is reported but not accepted (unchecked)."""
        cert = self._assessment()
        result = assess_load(cert, np.eye(4)[:1])[0]
        assert result.tolerance is None
        # Reported, not checked: an unchecked value must not be labelled accepted.
        assert not result.within_tolerance
        assert result.error_std >= 0.0

    def test_rejects_wrong_load_map_columns(self) -> None:
        """A load map whose columns do not match the states is rejected."""
        cert = self._assessment()
        with pytest.raises(ValueError, match="load_map must be"):
            assess_load(cert, np.ones((1, 3)))

    def test_rejects_wrong_names_length(self) -> None:
        """A names list that does not match the loads is rejected."""
        cert = self._assessment()
        with pytest.raises(ValueError, match="one entry per load"):
            assess_load(cert, np.eye(4), names=["only-one"])

    def test_rejects_wrong_tolerance_length(self) -> None:
        """A tolerance vector that does not match the loads is rejected."""
        cert = self._assessment()
        with pytest.raises(ValueError, match="scalar or have 2 values"):
            assess_load(cert, np.eye(2, 4), tolerance=[1.0, 2.0, 3.0])
