# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.sysid.augmentation — unknown-input disturbance augmentation."""

import numpy as np
import pytest

from vane.sysid import (
    DisturbanceModel,
    StateSpace,
    augment_with_disturbance,
    combine_disturbance_models,
    discretize_process_noise,
    gauss_markov_disturbance,
    harmonic_disturbance,
    random_walk_disturbance,
    random_walk_harmonic_disturbance,
)

_ENTRY = np.array([[0.0], [1.0]])  # the unknown load enters on the velocity state


def _physical() -> StateSpace:
    """Return a 2-state oscillator with input on the velocity and a position output."""
    return StateSpace(
        a=np.array([[0.0, 1.0], [-4.0, -0.2]]),
        b=np.array([[0.0], [1.0]]),
        c=np.array([[1.0, 0.0]]),
        state_names=["pos", "vel"],
        input_names=["u"],
        output_names=["y"],
    )


class TestRandomWalkDisturbance:
    """Tests for random_walk_disturbance."""

    def test_builds_integrator_states(self) -> None:
        """A random walk has zero dynamics and identity output, one state per signal."""
        dist = random_walk_disturbance(1e-3, n_signals=2)
        np.testing.assert_array_equal(dist.a, np.zeros((2, 2)))
        np.testing.assert_array_equal(dist.c, np.eye(2))
        np.testing.assert_allclose(np.diag(dist.intensity), [1e-3, 1e-3])
        assert dist.state_names == ["load_0", "load_1"]

    def test_per_channel_intensity(self) -> None:
        """A per-channel intensity sets each random walk separately."""
        dist = random_walk_disturbance([1.0, 2.0], n_signals=2)
        np.testing.assert_allclose(np.diag(dist.intensity), [1.0, 2.0])

    def test_rejects_nonpositive_n_signals(self) -> None:
        """A non-positive channel count is rejected."""
        with pytest.raises(ValueError, match="n_signals must be positive"):
            random_walk_disturbance(1.0, n_signals=0)

    def test_rejects_wrong_intensity_length(self) -> None:
        """An intensity vector that does not match the channels is rejected."""
        with pytest.raises(ValueError, match="scalar or have 2 values"):
            random_walk_disturbance([1.0, 2.0, 3.0], n_signals=2)

    def test_rejects_negative_intensity(self) -> None:
        """A negative intensity is rejected."""
        with pytest.raises(ValueError, match="non-negative"):
            random_walk_disturbance(-1.0)


class TestGaussMarkovDisturbance:
    """Tests for bounded first-order load perturbations."""

    def test_builds_independent_mean_reverting_states(self) -> None:
        """Each signal has its declared decay rate and process intensity."""
        disturbance = gauss_markov_disturbance(
            [10.0, 20.0], [2.0, 3.0], n_signals=2, name="torque"
        )
        np.testing.assert_allclose(disturbance.a, np.diag([-0.1, -0.05]))
        np.testing.assert_array_equal(disturbance.c, np.eye(2))
        np.testing.assert_array_equal(np.diag(disturbance.intensity), [2.0, 3.0])
        assert disturbance.state_names == ["torque_0", "torque_1"]

    @pytest.mark.parametrize("time_constant", [0.0, -1.0, np.inf, np.nan])
    def test_rejects_nonpositive_or_nonfinite_time_constant(
        self, time_constant: float
    ) -> None:
        """Mean-reversion time scales must define stable finite poles."""
        with pytest.raises(ValueError, match="finite positive"):
            gauss_markov_disturbance(time_constant, 1.0)


class TestHarmonicDisturbance:
    """Tests for harmonic_disturbance."""

    def test_builds_oscillator_blocks(self) -> None:
        """Each frequency gives an undamped 2-state oscillator at omega = 2 pi f."""
        dist = harmonic_disturbance([0.1], 1e-2)
        omega = 2.0 * np.pi * 0.1
        np.testing.assert_allclose(dist.a, [[0.0, omega], [-omega, 0.0]])
        np.testing.assert_array_equal(dist.c, [[1.0, 0.0]])
        assert dist.state_names == ["wave_0_re", "wave_0_im"]
        # Eigenvalues are +/- i omega (a pure oscillator at the wave frequency).
        np.testing.assert_allclose(
            np.sort(np.linalg.eigvals(dist.a).imag), [-omega, omega], atol=1e-9
        )

    def test_multiple_frequencies(self) -> None:
        """Two frequencies give four states and two output channels."""
        dist = harmonic_disturbance([0.1, 0.2], 1.0)
        assert dist.a.shape == (4, 4)
        assert dist.c.shape == (2, 4)

    def test_rejects_empty_frequencies(self) -> None:
        """An empty frequency list is rejected."""
        with pytest.raises(ValueError, match="non-empty"):
            harmonic_disturbance([], 1.0)

    @pytest.mark.parametrize("frequency", [-0.2, 0.0, np.nan, np.inf, -np.inf])
    def test_rejects_invalid_frequency(self, frequency: float) -> None:
        """Every oscillator frequency must be finite and positive."""
        with pytest.raises(ValueError, match="finite and positive"):
            harmonic_disturbance([0.1, frequency], 1.0)

    def test_rejects_angular_frequency_overflow(self) -> None:
        """A finite hertz value cannot overflow the oscillator state matrix."""
        with pytest.raises(ValueError, match="angular frequencies must remain finite"):
            harmonic_disturbance([np.finfo(np.float64).max], 1.0)

    def test_accepts_frequency_below_angular_conversion_limit(self) -> None:
        """The finite conversion guard retains the largest representable range."""
        limit = np.finfo(np.float64).max / (2.0 * np.pi)
        frequency = np.nextafter(limit, 0.0)
        disturbance = harmonic_disturbance([frequency], 1.0)
        assert np.isfinite(disturbance.a).all()


class TestCombineDisturbanceModels:
    """Tests for independent mixed-prior assembly."""

    def test_combines_random_walk_and_harmonic_signal_channels(self) -> None:
        """Torque and wave keep distinct outputs in one block-diagonal model."""
        torque = random_walk_disturbance(2.0, name="torque")
        wave = harmonic_disturbance([0.1], 3.0, name="wave")
        combined = combine_disturbance_models([torque, wave])
        assert combined.a.shape == (3, 3)
        np.testing.assert_array_equal(combined.a[0], 0.0)
        np.testing.assert_allclose(combined.a[1:, 1:], wave.a)
        np.testing.assert_array_equal(
            combined.c,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )
        np.testing.assert_array_equal(np.diag(combined.intensity), [2.0, 3.0, 3.0])
        assert combined.state_names == ["torque_0", "wave_0_re", "wave_0_im"]

    def test_rejects_empty_collection(self) -> None:
        """An empty block union has no physical disturbance contract."""
        with pytest.raises(ValueError, match="at least one"):
            combine_disturbance_models([])


class TestAugmentWithDisturbance:
    """Tests for augment_with_disturbance."""

    def test_assembles_block_structure(self) -> None:
        """The augmented A/B/C have the expected block layout."""
        ss = _physical()
        dist = random_walk_disturbance(1e-3)
        aug = augment_with_disturbance(ss, dist, entry=_ENTRY)
        # Top-left is A; top-right is entry @ C_d; bottom-right is A_d (zero).
        np.testing.assert_array_equal(aug.system.a[:2, :2], ss.a)
        np.testing.assert_array_equal(aug.system.a[:2, 2:], _ENTRY @ dist.c)
        np.testing.assert_array_equal(aug.system.a[2:, 2:], dist.a)
        assert aug.system.b is not None
        np.testing.assert_array_equal(aug.system.b, [[0.0], [1.0], [0.0]])
        assert aug.system.c is not None
        np.testing.assert_array_equal(aug.system.c, [[1.0, 0.0, 0.0]])
        assert aug.system.state_names == ["pos", "vel", "load_0"]
        assert aug.n_disturbance_states == 1

    def test_disturbance_intensity_is_symmetrized(self) -> None:
        """An intensity asymmetric within tolerance enters the density symmetrized."""
        base = random_walk_disturbance(1.0, n_signals=2)
        intensity = np.array([[1.0, 0.0], [5.0e-10, 1.0]])
        dist = DisturbanceModel(
            a=base.a, c=base.c, intensity=intensity, state_names=base.state_names
        )
        aug = augment_with_disturbance(_physical(), dist, entry=np.eye(2))
        block = aug.spectral_density[2:, 2:]
        np.testing.assert_array_equal(block, block.T)
        np.testing.assert_allclose(block, 0.5 * (intensity + intensity.T))

    def test_recovery_reads_the_disturbance(self) -> None:
        """The recovery map reads the disturbance estimate off the augmented state."""
        ss = _physical()
        aug = augment_with_disturbance(ss, random_walk_disturbance(1.0), entry=_ENTRY)
        assert aug.recovery.shape == (1, 3)
        np.testing.assert_allclose(aug.recovery @ np.array([1.0, 2.0, 7.0]), [7.0])

    def test_direct_disturbance_output_is_promoted_into_c(self) -> None:
        """A removed D column becomes the appended disturbance-state C block."""
        ss = _physical()
        direct = np.array([[2.5]])
        aug = augment_with_disturbance(
            ss,
            random_walk_disturbance(1.0),
            entry=_ENTRY,
            output_entry=direct,
        )
        assert aug.system.c is not None
        np.testing.assert_array_equal(aug.system.c, [[1.0, 0.0, 2.5]])

    def test_rejects_wrong_output_entry_shape(self) -> None:
        """Direct feedthrough must align with physical outputs and signals."""
        with pytest.raises(ValueError, match="output_entry must be"):
            augment_with_disturbance(
                _physical(),
                random_walk_disturbance(1.0),
                entry=_ENTRY,
                output_entry=np.ones((2, 1)),
            )

    def test_spectral_density_combines_physical_and_disturbance(self) -> None:
        """The augmented Qc carries physical noise and the disturbance intensity."""
        ss = _physical()
        dist = random_walk_disturbance(1e-3)
        aug = augment_with_disturbance(
            ss, dist, entry=_ENTRY, process_noise=np.diag([0.0, 1e-4])
        )
        np.testing.assert_allclose(np.diag(aug.spectral_density), [0.0, 1e-4, 1e-3])
        # The augmented Qc is a valid covariance the discretizer accepts.
        qd = discretize_process_noise(aug.system.a, aug.spectral_density, 0.05)
        assert np.isfinite(qd).all()

    def test_harmonic_augmentation_adds_two_states(self) -> None:
        """Augmenting with a one-frequency harmonic model adds two states."""
        ss = _physical()
        aug = augment_with_disturbance(
            ss, harmonic_disturbance([0.15], 1.0), entry=_ENTRY
        )
        assert aug.system.n_states == 4
        assert aug.recovery.shape == (1, 4)

    def test_random_walk_harmonic_model_sums_one_physical_signal(self) -> None:
        """DC, 1P, and 3P real quadratures recover one load channel."""
        disturbance = random_walk_harmonic_disturbance(
            [0.1, 0.3],
            2.0,
            [3.0, 4.0],
            name="torque",
        )
        assert disturbance.a.shape == (5, 5)
        np.testing.assert_array_equal(disturbance.c, [[1.0, 1.0, 0.0, 1.0, 0.0]])
        np.testing.assert_array_equal(
            np.diag(disturbance.intensity), [2.0, 3.0, 3.0, 4.0, 4.0]
        )
        assert disturbance.state_names == [
            "torque_dc",
            "torque_0_re",
            "torque_0_im",
            "torque_1_re",
            "torque_1_im",
        ]

    def test_random_walk_harmonic_rejects_duplicate_frequencies(self) -> None:
        """Repeated oscillators cannot create an unobservable summed component."""
        with pytest.raises(ValueError, match="frequencies must be unique"):
            random_walk_harmonic_disturbance([0.1, 0.1], 1.0, 1.0)

    def test_random_walk_harmonic_rejects_angular_frequency_overflow(self) -> None:
        """The summed load path cannot retain an infinite oscillator matrix."""
        with pytest.raises(ValueError, match="angular frequencies must remain finite"):
            random_walk_harmonic_disturbance([np.finfo(np.float64).max], 1.0, 1.0)

    @pytest.mark.parametrize(
        "intensity",
        [np.nan, np.inf, [1.0, np.nan], [1.0, np.inf]],
    )
    def test_random_walk_harmonic_rejects_nonfinite_intensity(
        self, intensity: float | list[float]
    ) -> None:
        """Malformed harmonic noise fails before covariance construction."""
        with pytest.raises(ValueError, match="finite and non-negative"):
            random_walk_harmonic_disturbance([0.1, 0.3], 1.0, intensity)

    def test_rejects_discrete_system(self) -> None:
        """A discrete system cannot be augmented (the construction is continuous)."""
        ss = _physical().discretized(0.05)
        with pytest.raises(ValueError, match="continuous-time"):
            augment_with_disturbance(ss, random_walk_disturbance(1.0), entry=_ENTRY)

    def test_rejects_wrong_entry_shape(self) -> None:
        """An entry map that does not match the states/signals is rejected."""
        ss = _physical()
        with pytest.raises(ValueError, match="entry must be"):
            augment_with_disturbance(
                ss, random_walk_disturbance(1.0), entry=np.ones((3, 1))
            )

    def test_rejects_wrong_process_noise_shape(self) -> None:
        """A physical process-noise of the wrong shape is rejected."""
        ss = _physical()
        with pytest.raises(ValueError, match="process_noise must be"):
            augment_with_disturbance(
                ss, random_walk_disturbance(1.0), entry=_ENTRY, process_noise=np.eye(3)
            )

    def test_rejects_non_psd_process_noise(self) -> None:
        """A non-PSD physical process noise is rejected (not silently exported)."""
        ss = _physical()
        with pytest.raises(ValueError, match="positive semidefinite"):
            augment_with_disturbance(
                ss,
                random_walk_disturbance(1.0),
                entry=_ENTRY,
                process_noise=np.array([[1.0, 0.0], [0.0, -1.0]]),
            )

    def test_rejects_inconsistent_disturbance(self) -> None:
        """A hand-built disturbance with mismatched matrices is rejected."""
        bad = DisturbanceModel(
            a=np.zeros((2, 2)),
            c=np.ones((1, 3)),  # 3 columns but a is 2x2
            intensity=np.eye(2),
            state_names=[],
        )
        with pytest.raises(ValueError, match=r"disturbance.c must have 2 columns"):
            augment_with_disturbance(_physical(), bad, entry=np.ones((2, 1)))

    def test_rejects_non_psd_disturbance_intensity(self) -> None:
        """A disturbance intensity that is not PSD is rejected."""
        bad = DisturbanceModel(
            a=np.zeros((1, 1)),
            c=np.ones((1, 1)),
            intensity=np.array([[-1.0]]),
            state_names=[],
        )
        with pytest.raises(ValueError, match="positive semidefinite"):
            augment_with_disturbance(_physical(), bad, entry=np.ones((2, 1)))
