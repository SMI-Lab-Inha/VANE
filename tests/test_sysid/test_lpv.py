# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.sysid.lpv -- the scheduled (LPV) estimator over a ROM family.

The families here are built by hand (two continuous oscillator realizations in a shared,
phase-aligned state basis) so the interpolation and runtime can be checked without
depending on the reduction/tracking pipeline that produces a real RomFamily.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from vane.sysid import (
    RomFamily,
    StateSpace,
    initialize_covariances,
    run_scheduled_filter,
    schedule_model,
)


def _oscillator(freq_hz: float, zeta: float = 0.03) -> StateSpace:
    """Return a continuous 2-state [position, velocity] oscillator at ``freq_hz``."""
    wn = 2.0 * np.pi * freq_hz
    a = np.array([[0.0, 1.0], [-(wn**2), -2.0 * zeta * wn]])
    return StateSpace(
        a=a,
        c=np.array([[1.0, 0.0]]),
        state_names=["pos", "vel"],
        output_names=["y"],
    )


def _family(macs: tuple[float, float] = (1.0, 0.98)) -> RomFamily:
    """Return a two-point continuous family at operating points 8 and 12 (one mode)."""
    return RomFamily(
        operating_points=np.array([8.0, 12.0]),
        systems=[_oscillator(0.3), _oscillator(0.5)],
        frequencies_hz=np.array([[0.3], [0.5]]),
        damping_ratios=np.array([[0.03], [0.03]]),
        tracking_mac=np.array([[macs[0]], [macs[1]]]),
    )


class TestScheduleModel:
    """Interpolating the family to a single operating point."""

    def test_exact_point_returns_that_member(self) -> None:
        """At an exact operating point the blend is that member, not extrapolated."""
        family = _family()
        scheduled = schedule_model(family, 8.0)
        np.testing.assert_allclose(scheduled.system.a, family.systems[0].a)
        assert not scheduled.extrapolated
        assert scheduled.min_tracking_mac == 1.0

    def test_midpoint_is_the_linear_blend(self) -> None:
        """Halfway between points, A is the average of the two members' A."""
        family = _family()
        scheduled = schedule_model(family, 10.0)
        expected = 0.5 * (family.systems[0].a + family.systems[1].a)
        np.testing.assert_allclose(scheduled.system.a, expected)
        assert not scheduled.extrapolated
        assert scheduled.min_tracking_mac == 0.98  # the step's tracking confidence

    def test_extrapolation_clamps_and_flags(self) -> None:
        """Outside the range the nearest member is used and flagged extrapolated."""
        family = _family()
        scheduled = schedule_model(family, 20.0)
        np.testing.assert_allclose(scheduled.system.a, family.systems[1].a)
        assert scheduled.extrapolated

    def test_descending_schedule_is_handled(self) -> None:
        """A descending operating-point schedule brackets the same way."""
        family = RomFamily(
            operating_points=np.array([12.0, 8.0]),
            systems=[_oscillator(0.5), _oscillator(0.3)],
            frequencies_hz=np.array([[0.5], [0.3]]),
            damping_ratios=np.array([[0.03], [0.03]]),
            tracking_mac=np.array([[1.0], [0.98]]),
        )
        scheduled = schedule_model(family, 10.0)
        expected = 0.5 * (_oscillator(0.5).a + _oscillator(0.3).a)
        np.testing.assert_allclose(scheduled.system.a, expected)
        assert not scheduled.extrapolated

    def test_descending_exact_interior_point_uses_that_members_mac(self) -> None:
        """On a descending sweep, an exact interior point reports its own MAC row."""
        # Build order [12, 10, 8]; tracking row k is point k vs point k-1.
        family = RomFamily(
            operating_points=np.array([12.0, 10.0, 8.0]),
            systems=[_oscillator(0.5), _oscillator(0.4), _oscillator(0.3)],
            frequencies_hz=np.array([[0.5], [0.4], [0.3]]),
            damping_ratios=np.array([[0.03], [0.03], [0.03]]),
            tracking_mac=np.array([[1.0], [0.90], [0.40]]),
        )
        # Landing exactly on 10 (build index 1) must report row 1 (0.90: 10-vs-12),
        # not row 2 (0.40: 8-vs-10).
        scheduled = schedule_model(family, 10.0)
        np.testing.assert_allclose(scheduled.system.a, family.systems[1].a)
        assert scheduled.min_tracking_mac == 0.90

    def test_mixed_feedthrough_presence_is_preserved(self) -> None:
        """A family mixing D presence keeps the present member's D (zeros, not None)."""
        # Member 0 has no D; member 1 has a feedthrough. Both share B (input dim 1).
        b = np.array([[0.0], [1.0]])
        member0 = StateSpace(a=_oscillator(0.3).a, b=b, c=np.array([[1.0, 0.0]]))
        member1 = StateSpace(
            a=_oscillator(0.5).a, b=b, c=np.array([[1.0, 0.0]]), d=np.array([[3.0]])
        )
        family = RomFamily(
            operating_points=np.array([8.0, 12.0]),
            systems=[member0, member1],
            frequencies_hz=np.array([[0.3], [0.5]]),
            damping_ratios=np.array([[0.03], [0.03]]),
            tracking_mac=np.array([[1.0], [0.98]]),
        )
        # Exactly on member 1 -> its D is kept, not dropped to None.
        at_12 = schedule_model(family, 12.0)
        assert at_12.system.d is not None
        np.testing.assert_allclose(at_12.system.d, [[3.0]])
        # Midpoint -> D is the zero-filled blend (half of member 1's D).
        mid = schedule_model(family, 10.0)
        assert mid.system.d is not None
        np.testing.assert_allclose(mid.system.d, [[1.5]])

    def test_low_mac_logs_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """A low tracking MAC at the interpolated step logs a warning."""
        family = _family(macs=(1.0, 0.5))
        with caplog.at_level(logging.WARNING, logger="vane.sysid.lpv"):
            scheduled = schedule_model(family, 10.0, mac_warn_threshold=0.8)
        assert scheduled.min_tracking_mac == 0.5
        assert "low tracking MAC" in caplog.text

    def test_extrapolation_logs_a_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An out-of-range operating point logs an extrapolation warning."""
        family = _family()
        with caplog.at_level(logging.WARNING, logger="vane.sysid.lpv"):
            schedule_model(family, 20.0)
        assert "extrapolation" in caplog.text

    def test_rejects_empty_family(self) -> None:
        """A family with no operating points is rejected."""
        empty = RomFamily(
            operating_points=np.empty(0),
            systems=[],
            frequencies_hz=np.empty((0, 0)),
            damping_ratios=np.empty((0, 0)),
            tracking_mac=np.empty((0, 0)),
        )
        with pytest.raises(ValueError, match="no operating points"):
            schedule_model(empty, 10.0)

    def test_rejects_non_finite_operating_point(self) -> None:
        """A non-finite operating point is rejected."""
        with pytest.raises(ValueError, match="operating_point must be finite"):
            schedule_model(_family(), np.inf)


def _simulate_constant_op(
    family: RomFamily, op: float, dt: float, n_steps: int, x0: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Noise-free truth/measurements from the scheduled model held at one op."""
    discrete = schedule_model(family, op).system.discretized(dt)
    assert discrete.c is not None
    x = x0.copy()
    truth = np.empty((n_steps, 2))
    meas = np.empty((n_steps, 1))
    for k in range(n_steps):
        x = discrete.a @ x
        truth[k] = x
        meas[k] = discrete.c @ x
    return truth, meas


class TestRunScheduledFilter:
    """The scheduled Kalman runtime over a drifting operating point."""

    def test_rejects_empty_family(self) -> None:
        """An empty family is a clear ValueError, not an IndexError."""
        empty = RomFamily(
            operating_points=np.empty(0),
            systems=[],
            frequencies_hz=np.empty((0, 0)),
            damping_ratios=np.empty((0, 0)),
            tracking_mac=np.empty((0, 0)),
        )
        cov = initialize_covariances(2, 1, measurement_noise=1e-2)
        with pytest.raises(ValueError, match="no operating points"):
            run_scheduled_filter(empty, cov, np.zeros((3, 1)), np.full(3, 8.0), dt=0.1)

    def test_tracks_at_a_constant_operating_point(self) -> None:
        """Held at one op, the scheduled filter recovers the (noise-free) truth."""
        family = _family()
        dt = 0.05
        n = 200
        x0 = np.array([1.0, 0.0])
        truth, meas = _simulate_constant_op(family, 8.0, dt, n, x0)
        cov = initialize_covariances(
            2, 1, process_noise=1e-9, measurement_noise=1e-6, initial_variance=1e-6
        )
        result = run_scheduled_filter(
            family, cov, meas, np.full(n, 8.0), dt=dt, initial_state=x0
        )
        rmse = np.sqrt(np.mean((result.states - truth) ** 2))
        assert rmse < 1e-2
        assert result.states.shape == (n, 2)
        assert result.accepted.all()
        np.testing.assert_allclose(result.min_tracking_mac, 1.0)  # op=8 is exact
        assert not result.extrapolated.any()

    def test_records_per_step_confidence_over_a_sweep(self) -> None:
        """A sweep from 8 to 12 records the interpolated-step tracking MAC per step."""
        family = _family()
        dt = 0.05
        n = 50
        ops = np.linspace(8.0, 12.0, n)
        _, meas = _simulate_constant_op(family, 10.0, dt, n, np.array([1.0, 0.0]))
        cov = initialize_covariances(2, 1, measurement_noise=1e-3)
        result = run_scheduled_filter(family, cov, meas, ops, dt=dt)
        assert result.min_tracking_mac[0] == 1.0  # starts exactly at op=8
        assert np.all(result.min_tracking_mac[1:] == 0.98)  # interior interpolation
        assert result.min_tracking_mac.shape == (n,)

    def test_extrapolation_is_flagged_and_warned(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Steps beyond the family range are flagged and a summary warning is logged."""
        family = _family()
        dt = 0.05
        n = 20
        ops = np.full(n, 15.0)  # all beyond the upper end (12)
        _, meas = _simulate_constant_op(family, 12.0, dt, n, np.array([1.0, 0.0]))
        cov = initialize_covariances(2, 1, measurement_noise=1e-3)
        with caplog.at_level(logging.WARNING, logger="vane.sysid.lpv"):
            result = run_scheduled_filter(family, cov, meas, ops, dt=dt)
        assert result.extrapolated.all()
        assert "extrapolated" in caplog.text

    def test_rejects_discrete_family(self) -> None:
        """A discrete family is rejected (members are discretized per step)."""
        discrete_member = _oscillator(0.3).discretized(0.05)
        family = RomFamily(
            operating_points=np.array([8.0, 12.0]),
            systems=[discrete_member, _oscillator(0.5).discretized(0.05)],
            frequencies_hz=np.array([[0.3], [0.5]]),
            damping_ratios=np.array([[0.03], [0.03]]),
            tracking_mac=np.array([[1.0], [0.98]]),
        )
        cov = initialize_covariances(2, 1)
        with pytest.raises(ValueError, match="continuous-time family"):
            run_scheduled_filter(
                family, cov, np.zeros((5, 1)), np.full(5, 8.0), dt=0.05
            )

    def test_rejects_non_positive_dt(self) -> None:
        """A non-positive dt is rejected."""
        cov = initialize_covariances(2, 1)
        with pytest.raises(ValueError, match="dt must be"):
            run_scheduled_filter(
                _family(), cov, np.zeros((5, 1)), np.full(5, 8.0), dt=0.0
            )

    def test_rejects_mismatched_operating_points_length(self) -> None:
        """operating_points must have one value per measurement step."""
        cov = initialize_covariances(2, 1)
        with pytest.raises(ValueError, match="operating_points must have"):
            run_scheduled_filter(
                _family(), cov, np.zeros((5, 1)), np.full(3, 8.0), dt=0.05
            )

    def test_rejects_wrong_measurement_shape(self) -> None:
        """Measurements must be (n_steps, n_outputs)."""
        cov = initialize_covariances(2, 1)
        with pytest.raises(ValueError, match="measurements must be"):
            run_scheduled_filter(
                _family(), cov, np.zeros((5, 2)), np.full(5, 8.0), dt=0.05
            )

    def test_empty_window_returns_empty_result(self) -> None:
        """A zero-step window returns an empty result (no IndexError on ops[0])."""
        cov = initialize_covariances(2, 1)
        result = run_scheduled_filter(
            _family(), cov, np.zeros((0, 1)), np.zeros(0), dt=0.05
        )
        assert result.states.shape == (0, 2)
        assert result.covariances.shape == (0, 2, 2)
        assert result.min_tracking_mac.shape == (0,)
        assert result.state_names == ["pos", "vel"]


class TestScheduledControlTiming:
    """The scheduled filter uses the same zero-order-hold timing as run_filter."""

    def _driven_family(self) -> RomFamily:
        system = StateSpace(
            a=_oscillator(0.4).a,
            b=np.array([[0.0], [1.0]]),
            c=np.array([[1.0, 0.0]]),
            d=np.array([[0.2]]),
            state_names=["pos", "vel"],
            input_names=["u"],
            output_names=["y"],
        )
        return RomFamily(
            operating_points=np.array([8.0, 12.0]),
            systems=[system, system],
            frequencies_hz=np.array([[0.4], [0.4]]),
            damping_ratios=np.array([[0.03], [0.03]]),
            tracking_mac=np.array([[1.0], [1.0]]),
        )

    @pytest.mark.parametrize("initial", [None, np.zeros(1)])
    def test_matches_run_filter_with_time_varying_input(
        self, initial: np.ndarray | None
    ) -> None:
        """A constant schedule reproduces run_filter exactly, input timing included."""
        from vane.sysid import build_kalman_model, run_filter

        family = self._driven_family()
        dt = 0.1
        n_steps = 60
        controls = np.sin(0.7 * np.arange(n_steps))[:, None]
        rng = np.random.default_rng(3)
        measurements = rng.normal(0.0, 0.1, (n_steps, 1))
        cov = initialize_covariances(2, 1, measurement_noise=1e-2)
        scheduled = run_scheduled_filter(
            family,
            cov,
            measurements,
            np.full(n_steps, 10.0),
            dt=dt,
            controls=controls,
            initial_control=initial,
        )
        model = build_kalman_model(family.systems[0], cov, dt=dt)
        reference = run_filter(
            model, measurements, controls=controls, initial_control=initial
        )
        np.testing.assert_allclose(scheduled.states, reference.states, atol=1e-12)
        np.testing.assert_allclose(scheduled.nis, reference.nis, atol=1e-10)

    def test_initial_control_requires_controls(self) -> None:
        """An initial input without a control record is rejected."""
        cov = initialize_covariances(2, 1, measurement_noise=1e-2)
        with pytest.raises(ValueError, match="initial_control requires controls"):
            run_scheduled_filter(
                _family(),
                cov,
                np.zeros((3, 1)),
                np.full(3, 8.0),
                dt=0.1,
                initial_control=np.zeros(1),
            )
