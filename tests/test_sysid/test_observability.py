# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.sysid.observability — Gramian, sensor placement, reduced estimator."""

from itertools import pairwise
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from scipy.integrate import quad_vec
from scipy.linalg import block_diag, expm

from vane.sysid import (
    GroupConstraint,
    ModalReduction,
    StateSpace,
    constrained_sensor_selection,
    finite_horizon_observability_gramian,
    load_reconstructability,
    modal_reduction,
    observability_gramian,
    observability_scores,
    recommend_sensors,
    reduced_estimator,
)


def _block(freq: float, zeta: float) -> np.ndarray:
    """Return a stable continuous 2x2 modal block."""
    wn = 2.0 * np.pi * freq
    sigma = -zeta * wn
    wd = wn * np.sqrt(1.0 - zeta**2)
    return np.array([[sigma, -wd], [wd, sigma]])


def _two_mode_a() -> np.ndarray:
    """Return a 4-state system matrix: mode 0 in states 0-1, mode 1 in states 2-3."""
    return np.asarray(block_diag(_block(0.3, 0.05), _block(1.5, 0.05)))


class TestObservabilityGramian:
    """Tests for observability_gramian."""

    def test_solves_the_lyapunov_equation(self) -> None:
        """The Gramian satisfies A^T Wo + Wo A + C^T C = 0."""
        a = _two_mode_a()
        c = np.array([[1.0, 0.0, 1.0, 0.0]])
        wo = observability_gramian(StateSpace(a=a, c=c))
        residual = a.T @ wo + wo @ a + c.T @ c
        np.testing.assert_allclose(residual, 0.0, atol=1e-9)
        assert np.all(np.linalg.eigvalsh(wo) > 0)  # observable -> positive definite

    def test_rejects_unstable_system(self) -> None:
        """An unstable system has no observability Gramian."""
        a = np.array([[0.5, -6.0], [6.0, 0.5]])  # unstable
        with pytest.raises(ValueError, match="stable"):
            observability_gramian(StateSpace(a=a, c=np.ones((1, 2))))

    def test_rejects_missing_c(self) -> None:
        """A system without a C matrix is rejected."""
        with pytest.raises(ValueError, match="no C matrix"):
            observability_gramian(StateSpace(a=_block(0.3, 0.05)))

    def test_noise_weighting_scales_the_information(self) -> None:
        """A larger measurement noise R lowers the (Fisher-information) Gramian."""
        a = _two_mode_a()
        c = np.eye(4)
        plain = observability_gramian(StateSpace(a=a, c=c))
        weighted = observability_gramian(
            StateSpace(a=a, c=c), measurement_noise=4.0 * np.eye(4)
        )
        # C^T (4 I)^-1 C = (1/4) C^T C, and the Lyapunov solve is linear in the source.
        np.testing.assert_allclose(weighted, 0.25 * plain, atol=1e-9)

    def test_rejects_bad_noise_shape(self) -> None:
        """A measurement-noise covariance of the wrong shape is rejected."""
        ss = StateSpace(a=_block(0.3, 0.05), c=np.eye(2))
        with pytest.raises(ValueError, match="measurement_noise"):
            observability_gramian(ss, measurement_noise=np.eye(3))

    def test_rejects_singular_noise_covariance(self) -> None:
        """A zero-variance (singular) channel is rejected, not silently inverted.

        ``measurement_noise_from_specs([..., 0.0, ...])`` yields a singular ``R``; the
        noise-weighted information would otherwise raise a raw ``LinAlgError`` or an
        undefined Gramian. A clear ``positive-definite`` error is raised instead.
        """
        ss = StateSpace(a=_block(0.3, 0.05), c=np.eye(2))
        singular = np.diag([1.0, 0.0])
        with pytest.raises(ValueError, match="positive-definite"):
            observability_gramian(ss, measurement_noise=singular)

    def test_rejects_indefinite_noise_covariance(self) -> None:
        """A negative variance (indefinite ``R``) is rejected before it hurts scores."""
        ss = StateSpace(a=_block(0.3, 0.05), c=np.eye(2))
        indefinite = np.diag([1.0, -2.0])
        with pytest.raises(ValueError, match="positive-definite"):
            observability_gramian(ss, measurement_noise=indefinite)

    def test_rejects_asymmetric_noise_covariance(self) -> None:
        """A non-symmetric ``R`` is rejected (a covariance must be symmetric)."""
        ss = StateSpace(a=_block(0.3, 0.05), c=np.eye(2))
        asymmetric = np.array([[1.0, 0.5], [0.0, 1.0]])
        with pytest.raises(ValueError, match="symmetric"):
            observability_gramian(ss, measurement_noise=asymmetric)


class TestFiniteHorizonGramian:
    """Tests for finite_horizon_observability_gramian."""

    def test_continuous_matches_numerical_integral(self) -> None:
        """The Van Loan result equals the direct integral of e^{A^T t} C^T C e^{A t}."""
        a = _two_mode_a()
        c = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        horizon = 2.0
        got = finite_horizon_observability_gramian(StateSpace(a=a, c=c), horizon)
        integral, _ = quad_vec(
            lambda t: expm(a.T * t) @ (c.T @ c) @ expm(a * t), 0.0, horizon
        )
        np.testing.assert_allclose(got, integral, atol=1e-8)
        np.testing.assert_allclose(got, got.T, atol=1e-12)  # symmetric

    def test_converges_to_infinite_horizon(self) -> None:
        """For a stable system a long window approaches the infinite-horizon Gramian.

        Uses a well-damped system so the window is long enough to converge while still
        inside the Van Loan method's numerically clean range (see the function's Notes).
        """
        ss = StateSpace(a=np.array([[0.0, 1.0], [-1.0, -1.0]]), c=np.eye(2))
        infinite = observability_gramian(ss)
        finite = finite_horizon_observability_gramian(ss, 18.0)
        np.testing.assert_allclose(finite, infinite, atol=1e-6)

    def test_defined_for_unstable_system(self) -> None:
        """The finite-horizon Gramian needs no stability and stays PSD."""
        a = np.array([[0.5, -6.0], [6.0, 0.5]])  # unstable
        gramian = finite_horizon_observability_gramian(
            StateSpace(a=a, c=np.eye(2)), 1.0
        )
        assert np.all(np.linalg.eigvalsh(gramian) >= -1e-9)  # PSD, finite

    def test_discrete_matches_direct_summation(self) -> None:
        """The discrete Gramian equals sum_k (A^T)^k C^T C A^k over the window."""
        ss = StateSpace(a=_two_mode_a(), c=np.eye(4)).discretized(0.02)
        assert ss.dt is not None and ss.c is not None  # narrow for the type checker
        ad, n_steps = ss.a, 50
        got = finite_horizon_observability_gramian(ss, n_steps * ss.dt)
        expected = np.zeros((4, 4))
        power = np.eye(4)
        for _ in range(n_steps):
            expected += power.T @ (ss.c.T @ ss.c) @ power
            power = power @ ad
        np.testing.assert_allclose(got, expected, atol=1e-9)

    def test_noise_weighting_scales(self) -> None:
        """Inverse-noise weighting scales the finite-horizon Gramian."""
        ss = StateSpace(a=_two_mode_a(), c=np.eye(4))
        plain = finite_horizon_observability_gramian(ss, 2.0)
        weighted = finite_horizon_observability_gramian(
            ss, 2.0, measurement_noise=4.0 * np.eye(4)
        )
        np.testing.assert_allclose(weighted, 0.25 * plain, atol=1e-9)

    @pytest.mark.parametrize("bad_horizon", [0.0, -1.0, float("inf"), float("nan")])
    def test_rejects_invalid_horizon(self, bad_horizon: float) -> None:
        """A non-positive, infinite, or NaN finite-horizon window is rejected.

        An infinite window would diverge the block matrix exponential; the
        infinite-horizon Gramian has its own (stability-checked) entry point.
        """
        ss = StateSpace(a=_block(0.3, 0.05), c=np.eye(2))
        with pytest.raises(ValueError, match="positive, finite time window"):
            finite_horizon_observability_gramian(ss, bad_horizon)

    def test_rejects_missing_c(self) -> None:
        """A system without a C matrix is rejected."""
        with pytest.raises(ValueError, match="no C matrix"):
            finite_horizon_observability_gramian(StateSpace(a=_block(0.3, 0.05)), 1.0)

    @pytest.mark.parametrize("horizon", [12.0, 20.0])
    def test_fast_stable_mode_matches_quadrature(self, horizon: float) -> None:
        """A fast stable mode does not overflow the Van Loan exponential.

        The block exponentiates ``-A``, so a -60 rad/s mode over a 12-20 s window
        overflowed the single-shot exponential (inf/NaN) before the window was split.
        """
        a = np.zeros((3, 3))
        a[0, 1] = 1.0
        a[1, 0] = -1.0
        a[1, 1] = -0.02
        a[2, 2] = -60.0
        c = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        got = finite_horizon_observability_gramian(StateSpace(a=a, c=c), horizon)
        integral, _ = quad_vec(
            lambda t: expm(a.T * t) @ (c.T @ c) @ expm(a * t),
            0.0,
            horizon,
            epsabs=1e-13,
            epsrel=1e-12,
            limit=2000,
        )
        np.testing.assert_allclose(got, integral, rtol=1e-9, atol=1e-11)

    def test_overflowing_unstable_gramian_raises(self) -> None:
        """An unstable mode that overflows over the window raises, never returns inf."""
        ss = StateSpace(a=np.array([[50.0]]), c=np.eye(1))
        with pytest.raises(ValueError, match="not finite"):
            finite_horizon_observability_gramian(ss, 20.0)
        discrete = StateSpace(a=np.array([[1e3]]), c=np.eye(1), dt=1.0)
        with pytest.raises(ValueError, match="not finite"):
            finite_horizon_observability_gramian(discrete, 200.0)


class TestObservabilityScores:
    """Tests for observability_scores."""

    def test_per_mode_equals_gramian_diagonal_blocks(self) -> None:
        """Each mode score equals the trace of the Gramian block over its states."""
        ss = StateSpace(a=_two_mode_a(), c=np.array([[1.0, 0.0, 2.0, 0.0]]))
        scores = observability_scores(ss, mode_states=[[0, 1], [2, 3]])
        wo = observability_gramian(ss)
        np.testing.assert_allclose(
            scores.per_mode, [np.trace(wo[:2, :2]), np.trace(wo[2:, 2:])], atol=1e-9
        )

    def test_per_output_ranks_a_loud_sensor_above_a_quiet_one(self) -> None:
        """A sensor with a larger output row carries more observability energy."""
        c = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.1, 0.0]])
        scores = observability_scores(
            StateSpace(a=_two_mode_a(), c=c, output_names=["loud", "quiet"])
        )
        assert scores.output_names == ["loud", "quiet"]
        assert scores.per_output[0] > scores.per_output[1]

    def test_no_mode_states_gives_empty_per_mode(self) -> None:
        """Without a mode grouping the per-mode breakdown is empty."""
        scores = observability_scores(StateSpace(a=_two_mode_a(), c=np.eye(4)))
        assert scores.per_mode.shape == (0,)
        assert scores.per_output.shape == (4,)

    @pytest.mark.parametrize(
        ("bad_noise", "match"),
        [
            (np.diag([1.0, 0.0, 1.0, 1.0]), "positive-definite"),  # singular channel
            (np.diag([1.0, -2.0, 1.0, 1.0]), "positive-definite"),  # negative variance
            (np.eye(4) + np.diag([0.5, 0.0, 0.0], 1), "symmetric"),  # asymmetric
        ],
    )
    def test_per_output_path_rejects_bad_noise(
        self, bad_noise: npt.NDArray[np.float64], match: str
    ) -> None:
        """The score-only path (mode_states=None) also validates the covariance.

        This per-output path weights each channel by ``1 / R[i, i]`` and never reaches
        ``_output_information``; a singular, indefinite, or asymmetric ``R`` must still
        fail with a clear error rather than yield ``inf`` / negative / accepted scores.
        """
        ss = StateSpace(a=_two_mode_a(), c=np.eye(4))
        with pytest.raises(ValueError, match=match):
            observability_scores(ss, measurement_noise=bad_noise)

    def test_noise_weighting_scales_per_mode(self) -> None:
        """A larger measurement noise lowers the per-mode information equally."""
        ss = StateSpace(a=_two_mode_a(), c=np.eye(4))
        plain = observability_scores(ss, mode_states=[[0, 1], [2, 3]])
        weighted = observability_scores(
            ss, mode_states=[[0, 1], [2, 3]], measurement_noise=4.0 * np.eye(4)
        )
        np.testing.assert_allclose(weighted.per_mode, 0.25 * plain.per_mode, atol=1e-9)

    def test_accepts_list_measurement_noise(self) -> None:
        """A nested-list covariance (as from YAML) is accepted like the array form."""
        ss = StateSpace(a=_two_mode_a(), c=np.eye(4))
        nested: Any = [[4.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
        as_list = observability_scores(ss, measurement_noise=nested)
        as_array = observability_scores(ss, measurement_noise=4.0 * np.eye(4))
        np.testing.assert_allclose(as_list.per_output, as_array.per_output, atol=1e-12)

    def test_finite_horizon_scores_an_unstable_system(self) -> None:
        """Per-mode/per-output scores are defined for an unstable finite window."""
        a = np.array([[0.5, -6.0], [6.0, 0.5]])  # unstable
        scores = observability_scores(
            StateSpace(a=a, c=np.eye(2)), mode_states=[[0, 1]], horizon=1.0
        )
        assert scores.per_mode[0] > 0.0
        assert np.all(scores.per_output > 0.0)

    def test_rejects_unstable_without_horizon(self) -> None:
        """An unstable system needs a finite horizon to be scored."""
        a = np.array([[0.5, -6.0], [6.0, 0.5]])
        with pytest.raises(ValueError, match="asymptotically stable"):
            observability_scores(StateSpace(a=a, c=np.eye(2)))

    @pytest.mark.parametrize("bad_horizon", [float("inf"), float("nan")])
    def test_rejects_non_finite_horizon(self, bad_horizon: float) -> None:
        """A non-finite horizon is rejected by the shared validator.

        ``horizon=inf`` is not "infinite horizon" (that is ``horizon=None``); a finite
        window of ``inf`` would diverge ``expm`` or ``round(inf / dt)`` downstream.
        """
        ss = StateSpace(a=_block(0.3, 0.05), c=np.eye(2))
        with pytest.raises(ValueError, match="positive, finite time window"):
            observability_scores(ss, horizon=bad_horizon)

    def test_rejects_out_of_range_mode_state(self) -> None:
        """A mode-state index outside the states is rejected."""
        ss = StateSpace(a=_two_mode_a(), c=np.eye(4))
        with pytest.raises(ValueError, match="out of range"):
            observability_scores(ss, mode_states=[[0, 9]])

    def test_rejects_missing_c(self) -> None:
        """A system without a C matrix is rejected."""
        with pytest.raises(ValueError, match="no C matrix"):
            observability_scores(StateSpace(a=_block(0.3, 0.05)))


class TestLoadReconstructability:
    """Tests for load_reconstructability."""

    def test_outcome_and_noise_amplification_are_basis_independent(self) -> None:
        """A similarity transform of the states keeps the outcome and the variance.

        ``reconstructable_fraction`` is coordinate-dependent for a partially blind load
        (documented), but whether a load is reconstructable and its
        ``noise_amplification`` are invariant under ``x = T x'``.
        """
        rng = np.random.default_rng(7)
        a0 = np.array([[-0.1, 2.0, 0.0], [-2.0, -0.1, 0.0], [0.0, 0.0, -0.5]])
        sensor = np.array([[1.0, 0.3, 0.0]])
        seen = np.array([[0.0, 1.0, 0.0]])
        blind = np.array([[0.0, 0.0, 1.0]])
        base = StateSpace(a=a0, c=np.vstack([sensor, seen, blind]))
        ref_seen = load_reconstructability(base, 1, sensor_outputs=[0])
        for _ in range(3):
            t = np.diag([1.0, 1e-2, 1e2]) @ rng.standard_normal((3, 3))
            t_inv = np.linalg.inv(t)
            moved = StateSpace(a=t_inv @ a0 @ t, c=np.vstack([sensor, seen, blind]) @ t)
            got_seen = load_reconstructability(moved, 1, sensor_outputs=[0])
            assert got_seen.reconstructable_fraction == pytest.approx(1.0, abs=1e-6)
            assert got_seen.noise_amplification == pytest.approx(
                ref_seen.noise_amplification, rel=1e-6
            )
            got_blind = load_reconstructability(moved, 2, sensor_outputs=[0])
            assert got_blind.reconstructable_fraction < 1.0
            assert got_blind.noise_amplification == np.inf

    def test_blind_load_is_unreconstructable(self) -> None:
        """A load living only in the sensors' blind subspace cannot be recovered."""
        # Sensor row 0 sees mode 0; the load row sees only mode 1 -> orthogonal.
        c = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        ss = StateSpace(a=_two_mode_a(), c=c, output_names=["sensor", "load"])
        result = load_reconstructability(ss, "load", sensor_outputs=[0])
        assert result.reconstructable_fraction == pytest.approx(0.0, abs=1e-9)
        assert result.blind_fraction == pytest.approx(1.0, abs=1e-9)
        assert result.load_name == "load"
        # A blind load is irrecoverable: its variance must be infinite, never a small
        # (best-looking) number a smaller-is-better ranking would pick.
        assert result.noise_amplification == np.inf

    def test_observed_load_is_fully_reconstructable(self) -> None:
        """A load a separate sensor fully observes has reconstructable fraction one."""
        # Sensor row 0 and the load row both see mode 1 (states 2,3): the sensor sees
        # the load's subspace without measuring the load output itself.
        c = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        ss = StateSpace(a=_two_mode_a(), c=c, output_names=["sensor", "load"])
        result = load_reconstructability(ss, "load", sensor_outputs=[0])
        assert result.reconstructable_fraction == pytest.approx(1.0, abs=1e-9)
        assert result.noise_amplification > 0.0

    def test_ambiguous_load_name_is_rejected(self) -> None:
        """A substring matching several channels never confirms the first one."""
        c = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        ss = StateSpace(
            a=_two_mode_a(), c=c, output_names=["sensor", "TwrBsMxt", "TwrBsMyt"]
        )
        with pytest.raises(ValueError, match="matches multiple channels"):
            load_reconstructability(ss, "TwrBsM", sensor_outputs=[0])

    def test_numpy_integer_load_index_is_accepted(self) -> None:
        """A NumPy integer output index resolves like a Python int."""
        c = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        ss = StateSpace(a=_two_mode_a(), c=c, output_names=["sensor", "load"])
        index: Any = np.int64(1)
        result = load_reconstructability(ss, index, sensor_outputs=[0])
        assert result.load_name == "load"

    def test_rejects_sensor_set_containing_the_load(self) -> None:
        """Measuring the load to reconstruct itself is rejected as a leak."""
        c = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        ss = StateSpace(a=_two_mode_a(), c=c, output_names=["sensor", "load"])
        with pytest.raises(ValueError, match="must not include the load"):
            load_reconstructability(ss, "load", sensor_outputs=[0, 1])

    def test_default_sensor_set_excludes_the_load(self) -> None:
        """The default sensor set omits the load row, so it cannot self-assess.

        The load lives in a subspace no other output sees, so the default (every row
        except the load) must report a *blind* load — not the falsely-perfect score the
        old "all rows" default produced by measuring the load itself.
        """
        c = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        ss = StateSpace(a=_two_mode_a(), c=c, output_names=["sensor", "load"])
        result = load_reconstructability(ss, "load")  # default sensor set
        assert result.reconstructable_fraction == pytest.approx(0.0, abs=1e-9)

    def test_partial_load_splits_between_observed_and_blind(self) -> None:
        """A load half in the observable subspace reports an intermediate fraction."""
        # Load row has equal weight on mode 0 (seen) and mode 1 (blind).
        c = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]])
        ss = StateSpace(a=_two_mode_a(), c=c, output_names=["sensor", "load"])
        result = load_reconstructability(ss, "load", sensor_outputs=[0])
        assert result.reconstructable_fraction == pytest.approx(0.5, abs=1e-9)
        assert result.blind_fraction == pytest.approx(0.5, abs=1e-9)
        assert result.noise_amplification == np.inf  # blind component -> undefined

    def test_noise_amplification_scales_with_measurement_noise(self) -> None:
        """Doubling the sensor variance scales the a-priori reconstruction variance."""
        c = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        ss = StateSpace(a=_two_mode_a(), c=c, output_names=["sensor", "load"])
        base = load_reconstructability(
            ss, "load", sensor_outputs=[0], measurement_noise=np.array([[1.0]])
        )
        noisy = load_reconstructability(
            ss, "load", sensor_outputs=[0], measurement_noise=np.array([[4.0]])
        )
        assert noisy.noise_amplification == pytest.approx(
            4.0 * base.noise_amplification, rel=1e-9
        )

    def test_reconstructs_a_named_load_from_a_reduced_model(self) -> None:
        """End to end: a reduced model's load is reconstructable from its sensors."""
        a = _two_mode_a()
        c = np.array([[1.0, 0.0, 0.5, 0.0], [0.2, 0.0, 1.0, 0.0]])
        ss = StateSpace(a=a, b=np.ones((4, 1)), c=c, output_names=["load", "sensor"])
        reduced = modal_reduction(ss, n_modes=None).system
        result = load_reconstructability(reduced, "load", sensor_outputs=[1])
        assert result.reconstructable_fraction == pytest.approx(1.0, abs=1e-9)
        assert result.n_observable_directions == 4

    def test_rejects_zero_load_row(self) -> None:
        """A load output whose row is all zeros has nothing to reconstruct."""
        c = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        ss = StateSpace(a=_two_mode_a(), c=c, output_names=["sensor", "load"])
        with pytest.raises(ValueError, match="zero output row"):
            load_reconstructability(ss, "load", sensor_outputs=[0])

    def test_rejects_out_of_range_sensor(self) -> None:
        """A sensor index outside the outputs is rejected."""
        ss = StateSpace(a=_two_mode_a(), c=np.eye(4))
        with pytest.raises(ValueError, match="out of range"):
            load_reconstructability(ss, 0, sensor_outputs=[9])

    def test_rejects_unstable_without_horizon(self) -> None:
        """An unstable system needs a finite horizon to be accepted."""
        a = np.array([[0.5, -6.0], [6.0, 0.5]])
        with pytest.raises(ValueError, match="asymptotically stable"):
            load_reconstructability(StateSpace(a=a, c=np.eye(2)), 0)


class TestRecommendSensors:
    """Tests for recommend_sensors."""

    def test_picks_complementary_sensors(self) -> None:
        """Greedy D-optimal placement covers both modes instead of doubling up."""
        a = _two_mode_a()
        # output 0 sees mode 0; output 1 sees mode 1; output 2 sees mode 0 (redundant).
        c = np.array(
            [
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.9, 0.9],
                [0.6, 0.6, 0.0, 0.0],
            ]
        )
        ss = StateSpace(a=a, c=c, output_names=["s0", "s1", "s2"])
        placement = recommend_sensors(ss, n_sensors=2, candidates=[0, 1, 2])
        assert set(placement.selected) == {0, 1}  # both modes observed
        assert placement.names == ["s0", "s1"]
        assert placement.scores[1] >= placement.scores[0]  # observability grows

    def test_caps_at_candidate_count(self) -> None:
        """Requesting more sensors than candidates returns all candidates."""
        ss = StateSpace(a=_two_mode_a(), c=np.eye(2, 4))
        placement = recommend_sensors(ss, n_sensors=5, candidates=[0, 1])
        assert len(placement.selected) == 2

    def test_rejects_nonpositive_n_sensors(self) -> None:
        """A non-positive sensor count is rejected."""
        ss = StateSpace(a=_two_mode_a(), c=np.eye(2, 4))
        with pytest.raises(ValueError, match="n_sensors must be positive"):
            recommend_sensors(ss, n_sensors=0)

    def test_duplicate_candidates_are_deduplicated(self) -> None:
        """Repeated candidate indices never select the same sensor twice."""
        ss = StateSpace(a=_two_mode_a(), c=np.eye(2, 4), output_names=["s0", "s1"])
        # A pool with a repeated index must behave like the unique set {0, 1}.
        placement = recommend_sensors(ss, n_sensors=2, candidates=[0, 0, 1])
        assert sorted(placement.selected) == [0, 1]
        assert len(set(placement.selected)) == len(placement.selected)

    def test_noise_weighting_prefers_the_calibrated_sensor(self) -> None:
        """With measurement noise, a low-noise sensor beats an identical noisy twin."""
        a = _two_mode_a()
        # Two candidates see the same mode; only their noise differs.
        c = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        ss = StateSpace(a=a, c=c, output_names=["clean", "noisy"])
        clean = recommend_sensors(
            ss, n_sensors=1, measurement_noise=np.diag([1.0, 100.0])
        )
        assert clean.selected == [0]  # the calibrated sensor wins
        # Flip the noise and the other sensor must win, proving noise drives the choice.
        flipped = recommend_sensors(
            ss, n_sensors=1, measurement_noise=np.diag([100.0, 1.0])
        )
        assert flipped.selected == [1]

    def test_finite_horizon_admits_a_marginal_system(self) -> None:
        """A finite horizon lets placement run on a non-stable (marginal) model."""
        a = np.array([[0.0, 1.0], [-1.0, 0.0]])  # pure oscillator: eigenvalues +/-i
        ss = StateSpace(a=a, c=np.eye(2), output_names=["p", "q"])
        with pytest.raises(ValueError, match="stable"):
            recommend_sensors(ss, n_sensors=1)  # infinite horizon needs stability
        placement = recommend_sensors(ss, n_sensors=1, horizon=5.0)
        assert len(placement.selected) == 1

    def test_rejects_bad_measurement_noise(self) -> None:
        """A measurement covariance of the wrong shape or not PD is rejected."""
        ss = StateSpace(a=_two_mode_a(), c=np.eye(2, 4), output_names=["s0", "s1"])
        with pytest.raises(ValueError, match="measurement_noise must be"):
            recommend_sensors(ss, n_sensors=1, measurement_noise=np.eye(3))
        with pytest.raises(ValueError, match="positive-definite"):
            recommend_sensors(ss, n_sensors=1, measurement_noise=np.diag([1.0, 0.0]))


class TestConstrainedSensorSelection:
    """Tests for constrained_sensor_selection."""

    @staticmethod
    def _system() -> StateSpace:
        """Return a system where 0,2 see mode 0 and 1,3 see mode 1 (2,3 redundant)."""
        c = np.array(
            [
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.6, 0.6, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.5],
            ]
        )
        return StateSpace(a=_two_mode_a(), c=c, output_names=["s0", "s1", "s2", "s3"])

    def test_picks_complementary_under_count_cap(self) -> None:
        """With a count cap the greedy pick covers both modes."""
        placement = constrained_sensor_selection(self._system(), max_sensors=2)
        assert set(placement.selected) == {0, 1}
        assert placement.scores[1] >= placement.scores[0]

    def test_accepts_noise_and_finite_horizon(self) -> None:
        """Noise weighting and a finite horizon are honoured under constraints.

        A high-noise mode-0 channel should lose to its clean redundant twin, and a
        finite horizon should let the selection run on a marginal model.
        """
        # Rows 0 and 2 both see mode 0; make row 0 noisy so the clean twin (2) wins.
        placement = constrained_sensor_selection(
            self._system(),
            candidates=[0, 2],
            max_sensors=1,
            measurement_noise=np.diag([100.0, 1.0, 1.0, 1.0]),
        )
        assert placement.selected == [2]
        marginal = StateSpace(a=np.array([[0.0, 1.0], [-1.0, 0.0]]), c=np.eye(2))
        with pytest.raises(ValueError, match="stable"):
            constrained_sensor_selection(marginal, max_sensors=1)
        ok = constrained_sensor_selection(marginal, max_sensors=1, horizon=5.0)
        assert len(ok.selected) == 1

    def test_required_sensors_come_first(self) -> None:
        """Required sensors are selected first, then the greedy complement."""
        placement = constrained_sensor_selection(
            self._system(), required=[2], max_sensors=2
        )
        assert placement.selected[0] == 2
        assert placement.selected[1] == 1  # the complementary mode

    def test_forbidden_sensors_are_excluded(self) -> None:
        """A forbidden sensor never appears in the selection."""
        placement = constrained_sensor_selection(
            self._system(), forbidden=[1], max_sensors=2
        )
        assert 1 not in placement.selected

    def test_budget_limits_total_cost(self) -> None:
        """The summed cost of the selection never exceeds the budget."""
        placement = constrained_sensor_selection(
            self._system(), costs=[1.0, 1.0, 1.0, 1.0], budget=2.0
        )
        assert placement.total_cost <= 2.0
        assert len(placement.selected) == 2

    def test_group_cap_limits_per_group(self) -> None:
        """A group cap limits how many sensors come from that group."""
        groups = [
            GroupConstraint(members=[0, 2], max_count=1),
            GroupConstraint(members=[1, 3], max_count=1),
        ]
        placement = constrained_sensor_selection(
            self._system(), groups=groups, max_sensors=4
        )
        assert len(set(placement.selected) & {0, 2}) <= 1
        assert len(set(placement.selected) & {1, 3}) <= 1

    def test_no_limit_selects_whole_pool_with_increasing_scores(self) -> None:
        """Without caps the greedy selection exhausts the pool, scores rising."""
        placement = constrained_sensor_selection(self._system())
        assert sorted(placement.selected) == [0, 1, 2, 3]
        assert all(a <= b + 1e-9 for a, b in pairwise(placement.scores))

    def test_rejects_required_over_budget(self) -> None:
        """Required sensors that alone exceed the budget are rejected."""
        with pytest.raises(ValueError, match="exceed the budget"):
            constrained_sensor_selection(
                self._system(),
                required=[0, 1],
                costs=[5.0, 5.0, 5.0, 5.0],
                budget=4.0,
            )

    def test_rejects_required_over_group_cap(self) -> None:
        """Required sensors that alone break a group cap are rejected."""
        with pytest.raises(ValueError, match="group capped"):
            constrained_sensor_selection(
                self._system(),
                required=[0, 2],
                groups=[GroupConstraint(members=[0, 2], max_count=1)],
            )

    def test_rejects_required_forbidden_overlap(self) -> None:
        """A sensor cannot be both required and forbidden."""
        with pytest.raises(ValueError, match="overlap"):
            constrained_sensor_selection(self._system(), required=[0], forbidden=[0])

    def test_rejects_negative_cost(self) -> None:
        """A negative cost is rejected."""
        with pytest.raises(ValueError, match="costs must be finite and non-negative"):
            constrained_sensor_selection(self._system(), costs=[1.0, -1.0, 1.0, 1.0])

    def test_rejects_non_finite_cost(self) -> None:
        """A NaN cost is rejected (it would otherwise bypass the budget cap)."""
        with pytest.raises(ValueError, match="costs must be finite and non-negative"):
            constrained_sensor_selection(
                self._system(), costs=[1.0, np.nan, 1.0, 1.0], budget=2.0
            )

    def test_rejects_non_finite_budget(self) -> None:
        """A NaN budget is rejected."""
        with pytest.raises(ValueError, match="budget must be a non-negative"):
            constrained_sensor_selection(self._system(), budget=np.nan)

    def test_rejects_non_finite_max_sensors(self) -> None:
        """A NaN count cap is rejected (it would otherwise disable the cap)."""
        bad_cap: Any = np.nan  # a NaN slipping in from a numeric config source
        with pytest.raises(ValueError, match="max_sensors must be an integer"):
            constrained_sensor_selection(self._system(), max_sensors=bad_cap)

    def test_rejects_fractional_max_sensors(self) -> None:
        """A fractional count cap is rejected rather than silently floored."""
        bad_cap: Any = 1.5
        with pytest.raises(ValueError, match="max_sensors must be an integer"):
            constrained_sensor_selection(self._system(), max_sensors=bad_cap)

    def test_rejects_non_finite_group_cap(self) -> None:
        """A NaN group cap is rejected (it would otherwise never be enforced)."""
        bad_count: Any = np.nan
        groups = [GroupConstraint(members=[0, 2], max_count=bad_count)]
        with pytest.raises(ValueError, match="group max_count must be an integer"):
            constrained_sensor_selection(self._system(), groups=groups)

    def test_rejects_unstable_system(self) -> None:
        """Constrained placement also needs a stable system."""
        a = np.array([[0.5, -6.0], [6.0, 0.5]])
        with pytest.raises(ValueError, match="asymptotically stable"):
            constrained_sensor_selection(StateSpace(a=a, c=np.eye(2)), max_sensors=1)


class TestReducedEstimator:
    """Tests for reduced_estimator."""

    def _reduction(self) -> ModalReduction:
        """Return a 2-mode reduction of a 2-output system."""
        a = _two_mode_a()
        c = np.array([[1.0, 0.0, 0.5, 0.0], [0.2, 0.0, 1.0, 0.0]])
        b = np.ones((4, 1))
        ss = StateSpace(a=a, b=b, c=c, output_names=["load", "other"])
        return modal_reduction(ss, n_modes=None)

    def test_assembles_a_discrete_kalman_model(self) -> None:
        """The estimator is a discrete Kalman model over the reduced states."""
        model = reduced_estimator(self._reduction(), sensor_outputs=[1], dt=0.05)
        assert model.dt == pytest.approx(0.05)
        assert model.a.shape == (4, 4)
        assert model.c is not None and model.c.shape == (1, 4)
        assert model.p0.shape == (4, 4)

    def test_low_confidence_inflates_initial_covariance(self) -> None:
        """A less-trusted mode starts with larger initial variance."""
        model = reduced_estimator(
            self._reduction(),
            sensor_outputs=[0],
            dt=0.05,
            confidences=[1.0, 0.2],
            base_variance=1.0,
            uncertainty_scale=10.0,
        )
        diag = np.diag(model.p0)
        assert diag[0] == pytest.approx(1.0)  # confident mode
        assert diag[2] == pytest.approx(9.0)  # 1 + 10 * (1 - 0.2)

    def test_rejects_wrong_confidence_length(self) -> None:
        """The confidence count must match the number of retained modes."""
        with pytest.raises(ValueError, match="one value per mode"):
            reduced_estimator(
                self._reduction(), sensor_outputs=[0], dt=0.05, confidences=[1.0]
            )

    def test_rejects_out_of_range_sensor(self) -> None:
        """A sensor index outside the outputs is rejected."""
        with pytest.raises(ValueError, match="out of range"):
            reduced_estimator(self._reduction(), sensor_outputs=[9], dt=0.05)
