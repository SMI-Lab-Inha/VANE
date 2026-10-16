# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.sysid.kalman_filter -- the runnable Kalman estimator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from vane.sysid import (
    KalmanFilter,
    StateSpace,
    SteadyStateKalmanFilter,
    build_kalman_model,
    filter_consistency,
    initialize_covariances,
    reconstruct_load,
    rts_smoother,
    run_filter,
    steady_state_estimator,
)
from vane.sysid.covariance_init import CovarianceSet

if TYPE_CHECKING:
    from vane.sysid.kalman_interface import KalmanModel


def _oscillator_model(dt: float = 0.05) -> KalmanModel:
    """Return a stable 2-state oscillator discretized into a Kalman model."""
    wn, zeta = 2.0 * np.pi * 0.5, 0.05
    a = np.array([[-zeta * wn, -wn], [wn, -zeta * wn]])
    c = np.array([[1.0, 0.0]])
    ss = StateSpace(a=a, c=c, state_names=["q", "p"], output_names=["y"])
    cov = initialize_covariances(
        2, 1, process_noise=1e-3, measurement_noise=1e-2, initial_variance=1.0
    )
    return build_kalman_model(ss, cov, dt=dt)


def _simulate(
    model: KalmanModel, n_steps: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (truth, measurements) from the model's own noise covariances."""
    rng = np.random.default_rng(seed)
    assert model.c is not None
    x = np.array([1.0, 0.0])
    truth, meas = [], []
    for _ in range(n_steps):
        x = model.a @ x + rng.multivariate_normal(np.zeros(2), model.q)
        truth.append(x.copy())
        meas.append(model.c @ x + rng.multivariate_normal(np.zeros(1), model.r))
    return np.array(truth), np.array(meas)


class TestKalmanFilter:
    """Tests for the recursive KalmanFilter."""

    def test_recovers_the_state_from_noisy_measurements(self) -> None:
        """The filtered estimate tracks the truth to a small RMSE."""
        model = _oscillator_model()
        truth, meas = _simulate(model, 200)
        result = run_filter(model, meas)
        rmse = np.sqrt(np.mean((result.states - truth) ** 2))
        assert rmse < 0.2
        assert result.states.shape == (200, 2)
        assert result.covariances.shape == (200, 2, 2)

    def test_covariance_stays_symmetric_psd(self) -> None:
        """The Joseph-form update keeps every error covariance symmetric PSD."""
        model = _oscillator_model()
        _, meas = _simulate(model, 50)
        result = run_filter(model, meas)
        for p in result.covariances:
            np.testing.assert_allclose(p, p.T, atol=1e-12)
            assert np.linalg.eigvalsh(p).min() > -1e-9

    def test_rejects_continuous_model(self) -> None:
        """A continuous (dt=None) model cannot be filtered."""
        ss = StateSpace(a=-np.eye(2), c=np.eye(1, 2))
        cov = initialize_covariances(2, 1)
        model = build_kalman_model(ss, cov)  # dt stays None
        with pytest.raises(ValueError, match="discrete"):
            KalmanFilter(model)

    def test_rejects_bad_initial_state(self) -> None:
        """An initial state of the wrong length is rejected."""
        with pytest.raises(ValueError, match="initial_state must have length"):
            KalmanFilter(_oscillator_model(), initial_state=np.zeros(3))

    def test_control_with_no_b_or_d_is_rejected(self) -> None:
        """A control on a model with neither B nor D (no input path) is rejected."""
        kf = KalmanFilter(_oscillator_model())  # B and D both None
        with pytest.raises(ValueError, match="neither B nor D"):
            kf.predict(control=np.array([1.0]))

    def test_non_finite_control_is_rejected_in_predict(self) -> None:
        """A non-finite control is rejected in predict, even for a B-only model.

        Without this guard a NaN control poisons the state through ``B u`` and, with a
        fully masked measurement step, ``run_filter`` would return NaN states instead of
        raising. ``predict`` is also reused by the scheduled filter, so the check covers
        that path too.
        """
        a = np.array([[0.5, 0.0], [0.0, 0.3]])
        b = np.array([[1.0], [1.0]])
        c = np.array([[1.0, 0.0]])
        ss = StateSpace(a=a, b=b, c=c, dt=0.05)  # B present, D absent
        cov = initialize_covariances(2, 1)
        kf = KalmanFilter(build_kalman_model(ss, cov))
        with pytest.raises(ValueError, match="control has non-finite values"):
            kf.predict(control=np.array([np.nan]))

    def test_feedthrough_only_model_accepts_controls(self) -> None:
        """A model with D but no B still accepts the control (for the D u term)."""
        a = np.array([[0.5, 0.0], [0.0, 0.3]])
        c = np.array([[1.0, 0.0]])
        d = np.array([[2.0]])
        ss = StateSpace(a=a, c=c, d=d, dt=0.05)
        cov = initialize_covariances(2, 1, process_noise=1e-9, measurement_noise=1e-9)
        model = build_kalman_model(ss, cov)  # B is None, D present
        n = 20
        controls = np.ones((n, 1)) * 0.4
        x = np.zeros(2)
        truth_rows, meas_rows = [], []
        for k in range(n):
            x = a @ x  # no B: control does not drive the state
            truth_rows.append(x.copy())
            meas_rows.append(c @ x + d @ controls[k])
        result = run_filter(model, np.array(meas_rows), controls=controls)
        assert np.sqrt(np.mean((result.states - np.array(truth_rows)) ** 2)) < 1e-3

    def test_nonzero_feedthrough_requires_a_control(self) -> None:
        """A model with nonzero D must be given the control, not treated as zero."""
        a = np.array([[0.5, 0.0], [0.0, 0.3]])
        c = np.array([[1.0, 0.0]])
        d = np.array([[2.0]])
        ss = StateSpace(a=a, c=c, d=d, dt=0.05)
        cov = initialize_covariances(2, 1)
        model = build_kalman_model(ss, cov)
        with pytest.raises(ValueError, match="nonzero feedthrough D"):
            run_filter(model, np.zeros((5, 1)))  # no controls

    def test_feedthrough_is_subtracted_from_the_innovation(self) -> None:
        """A nonzero D requires the control subtracted from the measurement.

        With z = C x + D u exactly (no noise), the filter must recover x; if it ignored
        the feedthrough it would mis-correct by K D u.
        """
        a = np.array([[0.5, 0.0], [0.0, 0.3]])
        b = np.array([[1.0], [0.5]])
        c = np.array([[1.0, 0.0]])
        d = np.array([[2.0]])  # the control feeds straight into the output
        ss = StateSpace(a=a, b=b, c=c, d=d, dt=0.05)
        cov = initialize_covariances(2, 1, process_noise=1e-9, measurement_noise=1e-9)
        model = build_kalman_model(ss, cov)
        n = 30
        controls = np.ones((n, 1)) * 0.7
        x = np.zeros(2)
        truth_rows, meas_rows = [], []
        for k in range(n):
            x = a @ x + b @ controls[k]
            truth_rows.append(x.copy())
            meas_rows.append(c @ x + d @ controls[k])
        truth = np.array(truth_rows)
        meas = np.array(meas_rows)
        result = run_filter(model, meas, controls=controls)
        # Feedthrough handled -> the state is recovered to machine precision.
        assert np.sqrt(np.mean((result.states - truth) ** 2)) < 1e-3


class TestRunFilterValidation:
    """Input validation for run_filter."""

    def test_rejects_wrong_measurement_shape(self) -> None:
        """Measurements must be (n_steps, n_outputs)."""
        with pytest.raises(ValueError, match="measurements must be"):
            run_filter(_oscillator_model(), np.zeros((10, 2)))

    def test_rejects_mismatched_controls_length(self) -> None:
        """Controls must have one row per step."""
        model = _oscillator_model()
        _, meas = _simulate(model, 10)
        with pytest.raises(ValueError, match="controls must have"):
            run_filter(model, meas, controls=np.zeros((5, 1)))


class TestKalmanUpdateValidation:
    """Boundary validation on the direct KalmanFilter.update API."""

    def _ready_filter(self) -> KalmanFilter:
        """Return a predicted single-output filter ready for an update."""
        kf = KalmanFilter(_oscillator_model())
        kf.predict()
        return kf

    def test_rejects_wrong_measurement_length(self) -> None:
        """A measurement that is not length n_outputs raises before any matmul."""
        kf = self._ready_filter()
        with pytest.raises(ValueError, match="measurement must have length 1"):
            kf.update(np.array([0.1, 0.2]))

    def test_rejects_non_finite_available_measurement(self) -> None:
        """A non-finite value on a reporting channel is rejected."""
        kf = self._ready_filter()
        with pytest.raises(ValueError, match="non-finite"):
            kf.update(np.array([np.nan]))

    def test_rejects_non_positive_gate(self) -> None:
        """The NIS gate must be a positive finite threshold."""
        kf = self._ready_filter()
        with pytest.raises(ValueError, match="gate must be"):
            kf.update(np.array([0.1]), gate=0.0)

    def test_rejects_non_finite_gate(self) -> None:
        """An infinite gate is rejected rather than disabling the test silently."""
        kf = self._ready_filter()
        with pytest.raises(ValueError, match="gate must be"):
            kf.update(np.array([0.1]), gate=np.inf)

    def test_rejects_mis_sized_control_with_feedthrough(self) -> None:
        """With a nonzero D, a control of the wrong length is rejected."""
        a = np.array([[0.5, 0.0], [0.0, 0.3]])
        c = np.array([[1.0, 0.0]])
        d = np.array([[3.0]])
        ss = StateSpace(a=a, c=c, d=d, dt=0.05)
        cov = initialize_covariances(2, 1)
        kf = KalmanFilter(build_kalman_model(ss, cov))
        kf.predict(np.array([0.0]))
        with pytest.raises(ValueError, match="control must have length 1"):
            kf.update(np.array([0.1]), control=np.array([0.0, 0.0]))


class TestSensorDropoutAndGating:
    """Missing-sensor masking and outlier rejection."""

    def test_masked_steps_are_not_accepted(self) -> None:
        """A fully-masked step performs predict-only and is flagged not accepted."""
        model = _oscillator_model()
        _, meas = _simulate(model, 100)
        available = np.ones((100, 1), dtype=bool)
        available[40:50] = False
        result = run_filter(model, meas, available=available)
        assert int(result.accepted.sum()) == 90
        assert not result.accepted[40:50].any()
        assert np.isnan(result.nis[40:50]).all()

    def test_partial_mask_updates_on_the_reporting_channel(self) -> None:
        """With two sensors, masking one still updates on the other."""
        wn, zeta = 2.0 * np.pi * 0.5, 0.05
        a = np.array([[-zeta * wn, -wn], [wn, -zeta * wn]])
        c = np.eye(2)
        ss = StateSpace(a=a, c=c, output_names=["y0", "y1"])
        cov = initialize_covariances(2, 2, measurement_noise=1e-2)
        model = build_kalman_model(ss, cov, dt=0.05)
        kf = KalmanFilter(model)
        kf.predict()
        nis, ok, dof = kf.update(
            np.array([0.5, 0.5]), available=np.array([True, False])
        )
        assert ok
        assert np.isfinite(nis)
        assert dof == 1  # only one channel reported

    def test_outlier_is_gated_out(self) -> None:
        """A measurement whose NIS exceeds the gate is rejected."""
        model = _oscillator_model()
        _, meas = _simulate(model, 150)
        meas[100] = 1e6  # gross outlier
        result = run_filter(model, meas, gate=50.0)
        assert not result.accepted[100]


class TestFilterConsistency:
    """The NIS consistency check."""

    def test_well_tuned_filter_is_consistent(self) -> None:
        """A filter whose Q/R match the simulation passes the NIS test."""
        model = _oscillator_model()
        _, meas = _simulate(model, 400)
        result = run_filter(model, meas)
        report = filter_consistency(result)
        assert report.consistent
        assert report.lower <= report.mean_nis <= report.upper
        assert report.n_accepted == 400
        assert report.expected_nis == pytest.approx(1.0)  # one channel per step

    def test_consistency_uses_per_step_active_channel_count(self) -> None:
        """A partial mask lowers a step's dof; the NIS test accounts for it.

        With two sensors and one masked on half the steps, the average dof is 1.5, not
        2, so the chi-square band must be built from the summed active channels.
        """
        wn, zeta = 2.0 * np.pi * 0.5, 0.05
        a = np.array([[-zeta * wn, -wn], [wn, -zeta * wn]])
        ss = StateSpace(a=a, c=np.eye(2), output_names=["y0", "y1"])
        cov = initialize_covariances(2, 2, process_noise=1e-3, measurement_noise=1e-2)
        model = build_kalman_model(ss, cov, dt=0.05)
        rng = np.random.default_rng(1)
        x = np.array([1.0, 0.0])
        meas_rows = []
        for _ in range(400):
            x = model.a @ x + rng.multivariate_normal(np.zeros(2), model.q)
            noise = rng.multivariate_normal(np.zeros(2), model.r)
            meas_rows.append(model.c @ x + noise)
        meas = np.array(meas_rows)
        available = np.ones((400, 2), dtype=bool)
        available[::2, 1] = False  # drop the second sensor on every other step
        result = run_filter(model, meas, available=available)
        report = filter_consistency(result)
        # 200 steps use 2 channels, 200 use 1 -> average dof 1.5.
        assert report.expected_nis == pytest.approx(1.5)
        assert report.consistent

    def test_overconfident_filter_is_flagged(self) -> None:
        """Shrinking R far below the true noise makes the NIS blow past the band."""
        model = _oscillator_model()
        _, meas = _simulate(model, 400)
        model.r = model.r * 1e-4  # claim the sensor is far better than it is
        result = run_filter(model, meas)
        report = filter_consistency(result)
        assert not report.consistent
        assert report.mean_nis > report.upper

    def test_rejects_no_accepted_steps(self) -> None:
        """A run with no accepted steps cannot be tested."""
        model = _oscillator_model()
        _, meas = _simulate(model, 20)
        result = run_filter(model, meas, available=np.zeros((20, 1), dtype=bool))
        with pytest.raises(ValueError, match="no accepted steps"):
            filter_consistency(result)

    def test_rejects_bad_confidence(self) -> None:
        """A confidence outside (0, 1) is rejected."""
        model = _oscillator_model()
        _, meas = _simulate(model, 20)
        result = run_filter(model, meas)
        with pytest.raises(ValueError, match="confidence must be"):
            filter_consistency(result, confidence=1.5)


class TestRtsSmoother:
    """The Rauch-Tung-Striebel backward pass."""

    def test_smoother_reduces_error_and_variance(self) -> None:
        """The smoothed estimate is at least as accurate and certain as the filtered."""
        model = _oscillator_model()
        truth, meas = _simulate(model, 200)
        result = run_filter(model, meas)
        sm_states, sm_cov = rts_smoother(model, result)
        f_rmse = np.sqrt(np.mean((result.states - truth) ** 2))
        s_rmse = np.sqrt(np.mean((sm_states - truth) ** 2))
        assert s_rmse <= f_rmse + 1e-9
        # The smoothed trace is no larger than the filtered one (more information).
        f_trace = np.trace(result.covariances, axis1=1, axis2=2).sum()
        s_trace = np.trace(sm_cov, axis1=1, axis2=2).sum()
        assert s_trace <= f_trace + 1e-9
        assert sm_states.shape == result.states.shape

    def test_smoother_handles_a_singular_predicted_covariance(self) -> None:
        """Zero process noise and zero initial variance make the prior singular.

        The forward filter still completes (positive measurement noise), and the
        least-squares smoother gain must not raise on the singular prior covariance.
        """
        a = np.array([[0.6, 0.0], [0.0, 0.4]])
        c = np.eye(2)
        ss = StateSpace(a=a, c=c, dt=0.05)
        cov = initialize_covariances(
            2, 2, process_noise=0.0, measurement_noise=1e-2, initial_variance=0.0
        )
        model = build_kalman_model(ss, cov)
        result = run_filter(model, np.zeros((10, 2)))
        sm_states, sm_cov = rts_smoother(model, result)  # must not raise
        assert np.all(np.isfinite(sm_states))
        assert np.all(np.isfinite(sm_cov))


class TestReconstructLoad:
    """Reading a linear load off the estimated state."""

    def test_load_equals_map_times_state(self) -> None:
        """The reconstructed load is L x and its std is sqrt(diag(L P L^T))."""
        model = _oscillator_model()
        _, meas = _simulate(model, 100)
        result = run_filter(model, meas)
        load_map = np.array([[1.0, 2.0]])
        loads, stds = reconstruct_load(result, load_map)
        np.testing.assert_allclose(loads, result.states @ load_map.T, atol=1e-12)
        expected_var = np.einsum(
            "ij,tjk,ik->ti", load_map, result.covariances, load_map
        )
        np.testing.assert_allclose(stds, np.sqrt(expected_var), atol=1e-12)
        assert np.all(stds > 0.0)

    def test_smoothed_load_uses_the_smoothed_estimate(self) -> None:
        """Passing the smoothed pair reads the load off the smoothed state."""
        model = _oscillator_model()
        _, meas = _simulate(model, 100)
        result = run_filter(model, meas)
        smoothed = rts_smoother(model, result)
        load_map = np.array([[1.0, 0.0]])
        loads, _ = reconstruct_load(result, load_map, smoothed=smoothed)
        np.testing.assert_allclose(loads, smoothed[0] @ load_map.T, atol=1e-12)

    def test_rejects_mismatched_load_map(self) -> None:
        """A load map whose columns do not match the states is rejected."""
        model = _oscillator_model()
        _, meas = _simulate(model, 10)
        result = run_filter(model, meas)
        with pytest.raises(ValueError, match="load_map must be"):
            reconstruct_load(result, np.ones((1, 3)))

    def test_rejects_non_finite_load_map(self) -> None:
        """A non-finite load map is rejected (a NaN would silently poison the load)."""
        model = _oscillator_model()
        _, meas = _simulate(model, 10)
        result = run_filter(model, meas)
        with pytest.raises(ValueError, match="load_map must be finite"):
            reconstruct_load(result, np.array([[1.0, np.nan]]))


class TestSteadyStateKalmanFilter:
    """The constant-gain steady-state filter."""

    def test_recovers_the_state_with_the_steady_gain(self) -> None:
        """Run with the DARE steady-state gain and track the truth."""
        model = _oscillator_model()
        truth, meas = _simulate(model, 300)
        assert model.c is not None
        cert = steady_state_estimator(
            StateSpace(a=model.a, c=model.c, dt=model.dt),
            process_noise=model.q,
            measurement_noise=model.r,
        )
        ss_kf = SteadyStateKalmanFilter(
            model.a, model.c, cert.gain, state_names=model.state_names
        )
        states = ss_kf.run(meas)
        rmse = np.sqrt(np.mean((states - truth) ** 2))
        assert rmse < 0.2
        assert states.shape == (300, 2)

    def test_rejects_inconsistent_gain_shape(self) -> None:
        """A gain whose shape does not match (n_states, n_outputs) is rejected."""
        with pytest.raises(ValueError, match="gain must be"):
            SteadyStateKalmanFilter(np.eye(2), np.eye(1, 2), np.zeros((2, 2)))

    def test_nonzero_feedthrough_requires_a_control(self) -> None:
        """A steady-state filter with nonzero D must be given the control."""
        kf = SteadyStateKalmanFilter(
            np.eye(2), np.eye(2), np.zeros((2, 2)), d=np.eye(2)
        )
        with pytest.raises(ValueError, match="nonzero feedthrough D"):
            kf.step(np.zeros(2))  # no control

    def test_rejects_wrong_measurement_length(self) -> None:
        """A measurement of the wrong length is rejected, not silently broadcast."""
        kf = SteadyStateKalmanFilter(np.eye(2), np.eye(2), np.zeros((2, 2)))
        with pytest.raises(ValueError, match="measurement must have length 2"):
            kf.step(np.array([1.0]))  # one element for a two-output model

    def test_subtracts_feedthrough_with_d(self) -> None:
        """With a D matrix the steady-state filter subtracts D u from the innovation.

        On noise-free data the estimate must track the truth; without the feedthrough
        term it would be biased by the input.
        """
        a = np.array([[0.5, 0.0], [0.0, 0.3]])
        b = np.array([[1.0], [0.5]])
        c = np.array([[1.0, 0.0]])
        d = np.array([[3.0]])
        ss = StateSpace(a=a, b=b, c=c, d=d, dt=0.05)
        cov = initialize_covariances(2, 1, process_noise=1e-6, measurement_noise=1e-6)
        model = build_kalman_model(ss, cov)
        assert model.c is not None
        cert = steady_state_estimator(
            StateSpace(a=model.a, c=model.c, dt=model.dt),
            process_noise=model.q,
            measurement_noise=model.r,
        )
        kf = SteadyStateKalmanFilter(model.a, model.c, cert.gain, b=model.b, d=model.d)
        n = 40
        controls = np.ones((n, 1)) * 0.6
        x = np.zeros(2)
        truth_rows, meas_rows = [], []
        for k in range(n):
            x = a @ x + b @ controls[k]
            truth_rows.append(x.copy())
            meas_rows.append(c @ x + d @ controls[k])
        states = kf.run(np.array(meas_rows), controls=controls)
        assert np.sqrt(np.mean((states - np.array(truth_rows)) ** 2)) < 1e-2

    def test_rejects_wrong_initial_state_length(self) -> None:
        """An initial_state that does not match the state dimension is rejected."""
        with pytest.raises(ValueError, match="initial_state must have length 2"):
            SteadyStateKalmanFilter(
                np.eye(2), np.eye(2), np.zeros((2, 2)), initial_state=np.zeros(3)
            )

    def test_rejects_mis_sized_control_in_step(self) -> None:
        """A control of the wrong length is rejected before it reaches a matmul."""
        kf = SteadyStateKalmanFilter(
            np.eye(2), np.eye(2), np.zeros((2, 2)), b=np.eye(2)
        )
        with pytest.raises(ValueError, match="control must have length 2"):
            kf.step(np.zeros(2), np.zeros(3))

    def test_run_rejects_wrong_measurement_shape(self) -> None:
        """run() requires a 2-D (n_steps, n_outputs) measurement block."""
        kf = SteadyStateKalmanFilter(np.eye(2), np.eye(2), np.zeros((2, 2)))
        with pytest.raises(ValueError, match="measurements must be"):
            kf.run(np.zeros(2))  # 1-D, not (n_steps, n_outputs)

    def test_run_rejects_mismatched_controls_length(self) -> None:
        """run() requires one control row per measurement step."""
        kf = SteadyStateKalmanFilter(
            np.eye(2), np.eye(2), np.zeros((2, 2)), b=np.eye(2)
        )
        with pytest.raises(ValueError, match="controls must have"):
            kf.run(np.zeros((5, 2)), controls=np.zeros((3, 2)))


def _zoh_plant(
    *, noise: bool = False, u_before: float | None = 0.0, seed: int = 0, n: int = 400
) -> tuple[KalmanModel, np.ndarray, np.ndarray, np.ndarray]:
    """Return (model, truth, measurements, controls) of a ZOH-sampled oscillator.

    The continuous oscillator with nonzero ``B`` and ``D`` is discretized by
    :meth:`StateSpace.discretized`, which holds ``u_{k-1}`` over ``[t_{k-1}, t_k]``;
    the truth is therefore ``x_k = A x_{k-1} + B u_{k-1}``, ``y_k = C x_k + D u_k``,
    from ``x_{-1} = 0``. ``u_before`` is the input held before the record (``None``
    holds ``u_0`` backward). The input is a sinusoid that is nonzero at ``k = 0``.
    """
    rng = np.random.default_rng(seed)
    ac = np.array([[0.0, 1.0], [-4.0, -0.4]])
    bc = np.array([[0.0], [1.0]])
    c = np.array([[1.0, 0.0]])
    d = np.array([[0.3]])
    dt = 0.05
    ssd = StateSpace(a=ac, b=bc, c=c, d=d).discretized(dt)
    cov = initialize_covariances(
        2, 1, process_noise=1e-4, measurement_noise=1e-3, initial_variance=1e-12
    )
    model = build_kalman_model(ssd, cov)
    assert model.b is not None
    t = np.arange(n) * dt
    u = (5.0 * np.cos(2.0 * np.pi * 0.5 * t) + 1.0)[:, None]
    held = u[0] if u_before is None else np.array([u_before])
    x = np.zeros(2)
    truth, meas = [], []
    for k in range(n):
        x = model.a @ x + model.b @ held
        if noise:
            x = x + rng.multivariate_normal(np.zeros(2), model.q)
        truth.append(x.copy())
        y = c @ x + d @ u[k]
        if noise:
            y = y + rng.multivariate_normal(np.zeros(1), model.r)
        meas.append(y)
        held = u[k]
    return model, np.array(truth), np.array(meas), u


class TestZohControlTiming:
    """The filter follows the ZOH timing ``x_k = A x_{k-1} + B u_{k-1}``.

    The measurement is ``y_k = C x_k + D u_k`` with the control logged alongside it.

    Before the fix the prediction used ``u_k`` (one sample early), which biased a
    time-varying-input run by ``B (u_k - u_{k-1})`` every step.
    """

    def test_noise_free_sinusoid_is_tracked_exactly(self) -> None:
        """Noise-free ZOH data with B and D both nonzero is reconstructed to ~0 RMSE."""
        model, truth, meas, u = _zoh_plant()
        result = run_filter(model, meas, controls=u, initial_control=np.zeros(1))
        rmse = np.sqrt(np.mean((result.states - truth) ** 2, axis=0))
        assert np.all(rmse < 1e-6 * truth.std(axis=0))
        assert np.nanmax(result.nis) < 1e-6  # no systematic innovation left

    def test_noisy_sinusoid_is_nis_consistent(self) -> None:
        """With the model's own noise, the controlled run passes the NIS test."""
        model, truth, meas, u = _zoh_plant(noise=True, n=2000)
        result = run_filter(model, meas, controls=u, initial_control=np.zeros(1))
        assert filter_consistency(result).consistent
        rmse = np.sqrt(np.mean((result.states - truth) ** 2, axis=0))
        assert np.all(rmse < 0.1 * truth.std(axis=0))

    def test_first_prediction_defaults_to_holding_the_first_control(self) -> None:
        """Without initial_control the input is held at controls[0] before the run."""
        model, truth, meas, u = _zoh_plant(u_before=None)
        result = run_filter(model, meas, controls=u)
        np.testing.assert_allclose(result.states, truth, atol=1e-9)

    def test_initial_control_drives_the_first_prediction(self) -> None:
        """A plant at rest before the record needs initial_control = 0 to be exact."""
        model, truth, meas, u = _zoh_plant(u_before=0.0)
        exact = run_filter(model, meas, controls=u, initial_control=np.zeros(1))
        np.testing.assert_allclose(exact.states, truth, atol=1e-9)
        held = run_filter(model, meas, controls=u)  # holds u_0 = 6 backward instead
        assert np.abs(held.predicted_states[0] - truth[0]).max() > 1e-3

    def test_constant_input_matches_the_previous_convention(self) -> None:
        """A constant input gives the same run as predicting with the current input."""
        model, _, meas, _ = _zoh_plant()
        controls = np.full((meas.shape[0], 1), 0.7)
        result = run_filter(model, meas, controls=controls)
        kf = KalmanFilter(model)
        for k in range(meas.shape[0]):
            kf.predict(controls[k])  # the former timing: B u_k
            kf.update(meas[k], control=controls[k])
            np.testing.assert_allclose(result.states[k], kf.x, rtol=0, atol=1e-12)

    def test_step_matches_run_filter(self) -> None:
        """KalmanFilter.step applies the same timing as run_filter."""
        model, _, meas, u = _zoh_plant(noise=True, n=60)
        result = run_filter(model, meas, controls=u, initial_control=np.zeros(1))
        kf = KalmanFilter(model, initial_control=np.zeros(1))
        for k in range(meas.shape[0]):
            kf.step(meas[k], u[k])
            np.testing.assert_allclose(kf.x, result.states[k], atol=1e-12)
        default = KalmanFilter(model)  # no initial_control: holds u_0 backward
        default.step(meas[0], u[0])
        ref = run_filter(model, meas[:1], controls=u[:1])
        np.testing.assert_allclose(default.x, ref.states[0], atol=1e-12)

    def test_steady_state_filter_follows_the_zoh_timing(self) -> None:
        """The fixed-gain filter tracks noise-free ZOH data exactly too."""
        model, truth, meas, u = _zoh_plant()
        assert model.c is not None
        cert = steady_state_estimator(
            StateSpace(a=model.a, c=model.c, dt=model.dt),
            process_noise=model.q,
            measurement_noise=model.r,
        )
        kf = SteadyStateKalmanFilter(
            model.a,
            model.c,
            cert.gain,
            b=model.b,
            d=model.d,
            initial_control=np.zeros(1),
        )
        states = kf.run(meas, controls=u)
        np.testing.assert_allclose(states, truth, atol=1e-9)
        held = SteadyStateKalmanFilter(
            model.a, model.c, cert.gain, b=model.b, d=model.d
        )
        _, truth_held, meas_held, _ = _zoh_plant(u_before=None)
        np.testing.assert_allclose(
            held.run(meas_held, controls=u), truth_held, atol=1e-9
        )

    def test_rts_smoother_matches_the_batch_posterior(self) -> None:
        """The smoothed run equals the exact batch posterior under ZOH timing."""
        rng = np.random.default_rng(3)
        a = np.array([[0.9, 0.2, 0.0], [-0.2, 0.9, 0.1], [0.0, 0.0, 0.95]])
        b = np.array([[0.1], [0.0], [0.3]])
        c = np.array([[1.0, 0.0, 0.5]])
        d = np.array([[0.2]])
        q = np.diag([0.01, 0.02, 0.005])
        r = np.array([[0.05]])
        p0 = np.eye(3)
        ss = StateSpace(a=a, b=b, c=c, d=d, dt=0.1)
        model = build_kalman_model(ss, CovarianceSet(q=q, r=r, p0=p0))
        n_steps, n = 25, 3
        u = rng.standard_normal((n_steps, 1))
        z = rng.standard_normal((n_steps, 1))
        avail = rng.random((n_steps, 1)) > 0.2
        u_init = np.array([0.4])
        result = run_filter(
            model, z, controls=u, available=avail, initial_control=u_init
        )
        sm_states, sm_cov = rts_smoother(model, result)
        # Batch information form over x_0..x_{N-1}; x_{-1} ~ N(0, P0) and the input
        # held over interval k is u_{k-1} (u_{-1} = initial_control).
        info = np.zeros((n_steps * n, n_steps * n))
        vec = np.zeros(n_steps * n)
        prior0 = np.linalg.inv(a @ p0 @ a.T + q)
        info[:n, :n] += prior0
        vec[:n] += prior0 @ (b @ u_init)
        q_inv = np.linalg.inv(q)
        for k in range(1, n_steps):
            f = np.zeros((n, n_steps * n))
            f[:, k * n : (k + 1) * n] = np.eye(n)
            f[:, (k - 1) * n : k * n] = -a
            info += f.T @ q_inv @ f
            vec += f.T @ q_inv @ (b @ u[k - 1])
        r_inv = np.linalg.inv(r)
        for k in range(n_steps):
            if avail[k, 0]:
                g = np.zeros((1, n_steps * n))
                g[:, k * n : (k + 1) * n] = c
                info += g.T @ r_inv @ g
                vec += g.T @ r_inv @ (z[k] - d @ u[k])
        post_cov = np.linalg.inv(info)
        post_mean = (post_cov @ vec).reshape(n_steps, n)
        np.testing.assert_allclose(sm_states, post_mean, atol=1e-9)
        for k in range(n_steps):
            block = post_cov[k * n : (k + 1) * n, k * n : (k + 1) * n]
            np.testing.assert_allclose(sm_cov[k], block, atol=1e-9)

    def test_initial_control_requires_controls(self) -> None:
        """run_filter rejects an initial_control with no control sequence."""
        model, _, meas, _ = _zoh_plant(n=5)
        with pytest.raises(ValueError, match="initial_control requires controls"):
            run_filter(model, meas, initial_control=np.zeros(1))

    def test_rejects_mis_sized_initial_control(self) -> None:
        """A wrong-length initial_control is rejected at construction."""
        model, *_ = _zoh_plant(n=5)
        with pytest.raises(ValueError, match="control must have length 1"):
            KalmanFilter(model, initial_control=np.zeros(2))
        with pytest.raises(ValueError, match="control must have length 1"):
            SteadyStateKalmanFilter(
                np.eye(2),
                np.eye(1, 2),
                np.zeros((2, 1)),
                b=np.ones((2, 1)),
                initial_control=np.zeros(3),
            )


def _matched_two_channel_run(
    n_steps: int, seed: int = 0
) -> tuple[KalmanModel, np.ndarray]:
    """Return a matched 2-output model (P0 at steady state) and its own measurements."""
    rng = np.random.default_rng(seed)
    a = np.array([[0.95, 0.1], [-0.1, 0.95]])
    c = np.eye(2)
    q = 0.01 * np.eye(2)
    r = 0.1 * np.eye(2)
    ss = StateSpace(a=a, c=c, dt=0.1)
    cert = steady_state_estimator(ss, process_noise=q, measurement_noise=r)
    p0 = cert.error_covariance
    model = build_kalman_model(ss, CovarianceSet(q=q, r=r, p0=p0))
    x = rng.multivariate_normal(np.zeros(2), p0)
    meas = np.empty((n_steps, 2))
    for k in range(n_steps):
        x = a @ x + rng.multivariate_normal(np.zeros(2), q)
        meas[k] = c @ x + rng.multivariate_normal(np.zeros(2), r)
    return model, meas


class TestGatedConsistency:
    """A correctly tuned filter stays NIS-consistent with the outlier gate on."""

    def test_matched_model_with_a_99_percent_gate_is_consistent(self) -> None:
        """Gated-out NIS values count, so the gate does not bias the NIS test low."""
        from scipy.stats import chi2

        model, meas = _matched_two_channel_run(20000)
        result = run_filter(model, meas, gate=float(chi2.ppf(0.99, 2)))
        assert (~result.accepted).sum() > 0  # the gate did reject steps
        report = filter_consistency(result)
        assert report.consistent
        assert report.n_gated == int((~result.accepted).sum())
        assert report.n_accepted + report.n_gated == meas.shape[0]

    def test_gated_step_reports_its_degrees_of_freedom(self) -> None:
        """A rejected update still reports the channels in its innovation."""
        model, _ = _matched_two_channel_run(10)
        kf = KalmanFilter(model)
        kf.predict()
        nis, ok, dof = kf.update(np.array([1e3, 1e3]), gate=5.0)
        assert not ok
        assert nis > 5.0
        assert dof == 2


class TestGateProbability:
    """The probability gate scales its threshold with the channels that reported."""

    def test_rejection_rate_is_nominal_on_partial_steps(self) -> None:
        """One-channel steps are rejected at ~1 - p, not at the two-channel rate."""
        model, meas = _matched_two_channel_run(20000, seed=1)
        available = np.ones(meas.shape, dtype=bool)
        available[::2, 1] = False  # every other step reports one channel
        result = run_filter(model, meas, available=available, gate_probability=0.99)
        one = available.sum(axis=1) == 1
        rate_one = float(np.mean(~result.accepted[one]))
        rate_two = float(np.mean(~result.accepted[~one]))
        assert 0.006 < rate_one < 0.015
        assert 0.006 < rate_two < 0.015
        assert np.array_equal(result.n_active, available.sum(axis=1))
        assert filter_consistency(result).consistent

    def test_rejects_both_gates(self) -> None:
        """A fixed gate and a gate probability cannot both be given."""
        model, meas = _matched_two_channel_run(5)
        with pytest.raises(ValueError, match="not both"):
            run_filter(model, meas, gate=5.0, gate_probability=0.99)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, float("nan")])
    def test_rejects_invalid_probability(self, bad: float) -> None:
        """The gate probability must lie strictly inside (0, 1)."""
        model, meas = _matched_two_channel_run(5)
        with pytest.raises(ValueError, match="gate_probability must be in"):
            run_filter(model, meas, gate_probability=bad)

    def test_step_accepts_a_gate_probability(self) -> None:
        """KalmanFilter.step forwards the probability gate to the update."""
        model, _ = _matched_two_channel_run(5)
        kf = KalmanFilter(model)
        _, ok, dof = kf.step(np.array([1e3, 1e3]), gate_probability=0.99)
        assert not ok
        assert dof == 2
