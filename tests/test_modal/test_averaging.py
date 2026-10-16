# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.modal.averaging — quadrature, harmonics and the azimuth state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.integrate import solve_ivp, trapezoid

from vane.modal import averaging as averaging_module
from vane.modal.averaging import (
    AzimuthConsistency,
    AzimuthStates,
    TowerPassageFit,
    azimuth_consistency,
    azimuth_mean,
    azimuth_tangent,
    delete_azimuth_state,
    eigen_basis,
    fit_harmonics,
    grid_error_matrix,
    interval_weights,
    locate_azimuth_state,
    pin_azimuth,
    sideband_correction,
    subgrid_weights,
    tower_passage_decay,
    uniform_azimuths,
    weight_kernel,
    zero_azimuth_column,
)
from vane.modal.floquet import monodromy_exponents

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

OMEGA = 1.0  # rotor speed of the synthetic systems, rad/s
AZIMUTH = AzimuthStates(position=1, speed=3)


def azimuth_system(
    eps: float = 0.05,
    delta: float = 0.0,
    *,
    k0: float = 0.09,
    cp: float = 0.02,
    b0: float = 0.2,
    cpw: float = 0.05,
    dwp: float = 0.01,
) -> Callable[[float], npt.NDArray[np.float64]]:
    """Return ``A(psi)`` of a four-state rotor model with an exact orbit tangent.

    States are ``[q, psi, q', w]`` (a platform-like oscillator ``q``, the rotor azimuth
    ``psi`` and their rates, so ``ndof2 = 2``). The periodic content (3P and 6P,
    scaled by ``eps``) is built so that ``xi(psi) = [p, 1, Omega p', 0]`` solves
    ``A(psi) xi = Omega xi'``: the model is the linearization about a periodic orbit
    and has one Floquet exponent at exactly zero. ``delta`` adds a constant azimuth
    stiffness to the rotor-speed row, which breaks that property.
    """

    def a_of(psi: float) -> npt.NDArray[np.float64]:
        p = eps * (0.8 * np.cos(3 * psi) + 0.5 * np.sin(3 * psi))
        dp = eps * 3.0 * (-0.8 * np.sin(3 * psi) + 0.5 * np.cos(3 * psi))
        ddp = -9.0 * p
        kp = k0 * (1.0 + eps * np.cos(3 * psi))
        base = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [-kp, 0.0, -cp, cpw],
                [0.02 * eps * np.sin(3 * psi), 0.0, dwp, -b0],
            ]
        )
        xi = np.array([p, 1.0, OMEGA * dp, 0.0])
        column = np.array([OMEGA**2 * ddp, 0.0]) - base[2:4] @ xi
        matrix = base.copy()
        matrix[2:4, 1] = column
        matrix[3, 1] += delta
        return matrix

    return a_of


def sample(
    a_of: Callable[[float], npt.NDArray[np.float64]], angles: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Sample a matrix function at the given azimuths."""
    return np.array([a_of(float(psi)) for psi in angles])


def jittered(count: int, amplitude: float, seed: int = 0) -> npt.NDArray[np.float64]:
    """Return a non-uniform grid: uniform azimuths moved by up to ``amplitude`` gaps."""
    rng = np.random.default_rng(seed)
    base = uniform_azimuths(count)
    return np.mod(
        base + amplitude * (2 * np.pi / count) * rng.uniform(size=count), 2 * np.pi
    )


def floquet_exponents(
    a_of: Callable[[float], npt.NDArray[np.float64]], n: int = 4
) -> npt.NDArray[np.complex128]:
    """Return accurate Floquet exponents by integrating the monodromy matrix."""
    period = 2 * np.pi / OMEGA

    def rhs(t: float, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return (a_of(OMEGA * t) @ y.reshape(n, n)).ravel()

    solution = solve_ivp(
        rhs, (0.0, period), np.eye(n).ravel(), method="DOP853", rtol=1e-12, atol=1e-14
    )
    monodromy = solution.y[:, -1].reshape(n, n)
    return np.log(np.linalg.eigvals(monodromy).astype(complex)) / period


def matched_exponent(exponents: npt.NDArray[np.complex128], target: complex) -> complex:
    """Return the exponent equal to ``target`` modulo ``i Omega`` most closely."""
    period = 2 * np.pi / OMEGA
    k = int(np.argmin(np.abs(np.exp((exponents - target) * period) - 1.0)))
    value = complex(exponents[k])
    return value + 1j * round((target - value).imag / OMEGA) * OMEGA


def platform_index(values: npt.NDArray[np.complex128]) -> int:
    """Return the oscillatory (platform) eigenvalue with positive imaginary part."""
    return int(np.argmax(values.imag))


class TestIntervalWeights:
    """Tests for interval_weights (the periodic trapezoidal weights)."""

    def test_uniform_grid_gives_equal_weights(self) -> None:
        """A uniform grid has weights 1/M."""
        weights = interval_weights(uniform_azimuths(12, 0.3))
        np.testing.assert_allclose(weights, 1.0 / 12, atol=1e-15)

    def test_sum_and_input_order(self) -> None:
        """Weights sum to one and follow the input order."""
        angles = jittered(36, 0.4)
        permutation = np.random.default_rng(3).permutation(36)
        weights = interval_weights(angles)
        assert abs(weights.sum() - 1.0) < 1e-12
        np.testing.assert_allclose(
            interval_weights(angles[permutation]), weights[permutation]
        )

    def test_exact_for_piecewise_linear_signals(self) -> None:
        """The weighted mean is the exact mean of the periodic linear interpolant."""
        angles = np.sort(jittered(17, 0.45, seed=5))
        values = np.random.default_rng(1).normal(size=17)
        dense = np.linspace(0.0, 2 * np.pi, 400_001)
        interpolant = np.interp(dense, angles, values, period=2 * np.pi)
        exact = trapezoid(interpolant, dense) / (2 * np.pi)
        assert abs(interval_weights(angles) @ values - exact) < 1e-9

    @pytest.mark.parametrize(
        "angles",
        [
            [0.0, 2 * np.pi],  # duplicate after wrapping
            [0.0],  # one azimuth leaves a full-revolution gap
            [0.0, 0.1, 0.2],  # gap larger than half a revolution
            [0.0, np.nan],
            [],
        ],
    )
    def test_rejects_invalid_grids(self, angles: list[float]) -> None:
        """Duplicates, large gaps and non-finite values are rejected."""
        with pytest.raises(ValueError, match="azimuth"):
            interval_weights(angles)


class TestSubgridWeights:
    """Tests for subgrid_weights."""

    def test_interval_and_arithmetic_rules(self) -> None:
        """Both rules give full-length weights summing to one, zero off the subset."""
        angles = jittered(12, 0.2)
        picks = [0, 3, 6, 9]
        for rule in ("interval", "arithmetic"):
            weights = subgrid_weights(angles, picks, rule=rule)  # type: ignore[arg-type]
            assert abs(weights.sum() - 1.0) < 1e-12
            assert np.all(weights[[1, 2, 4, 5, 7, 8, 10, 11]] == 0.0)
        np.testing.assert_allclose(
            subgrid_weights(angles, picks, rule="arithmetic")[picks], 0.25
        )

    @pytest.mark.parametrize("picks", [[], [0, 0], [12], [-1]])
    def test_rejects_bad_indices(self, picks: list[int]) -> None:
        """Empty, repeated or out-of-range indices are rejected."""
        with pytest.raises(ValueError, match="indices"):
            subgrid_weights(uniform_azimuths(12), picks)

    def test_rejects_unknown_rule(self) -> None:
        """An unknown rule is rejected."""
        with pytest.raises(ValueError, match="rule"):
            subgrid_weights(uniform_azimuths(4), [0, 1, 2, 3], rule="simpson")  # type: ignore[arg-type]


class TestWeightKernel:
    """Tests for weight_kernel."""

    @pytest.mark.parametrize("count", [5, 6, 7, 12])
    def test_uniform_alias_identity(self, count: int) -> None:
        """On a uniform grid W(h) = exp(i h psi0) when N divides h, else zero."""
        start = 0.37
        angles = uniform_azimuths(count, start)
        orders = np.arange(-40, 41)
        kernel = weight_kernel(angles, np.full(count, 1.0 / count), orders)
        expected = np.where(orders % count == 0, np.exp(1j * orders * start), 0.0)
        np.testing.assert_allclose(kernel, expected, atol=1e-13)

    def test_zeroth_order_is_weight_sum(self) -> None:
        """W(0) of the interval weights is one; leakage is non-zero off-uniform."""
        angles = jittered(36, 0.3)
        kernel = weight_kernel(angles, interval_weights(angles), [0, 3])
        assert abs(kernel[0] - 1.0) < 1e-12
        assert abs(kernel[1]) > 1e-6

    def test_rejects_mismatched_weights(self) -> None:
        """Weights must match the azimuths."""
        with pytest.raises(ValueError, match="weights"):
            weight_kernel([0.0, 1.0, 2.0], [0.5, 0.5], [0])


class TestUniformAzimuths:
    """Tests for uniform_azimuths."""

    def test_wraps_and_spaces(self) -> None:
        """The grid is uniform and wrapped into [0, 2pi)."""
        grid = uniform_azimuths(4, 7.0)
        assert np.all((grid >= 0.0) & (grid < 2 * np.pi))
        np.testing.assert_allclose(np.diff(np.sort(grid)), np.pi / 2)

    @pytest.mark.parametrize(("count", "start"), [(0, 0.0), (2.5, 0.0), (3, np.inf)])
    def test_rejects_invalid(self, count: float, start: float) -> None:
        """A non-positive or fractional count, or infinite start, is rejected."""
        with pytest.raises(ValueError, match=r"count|start"):
            uniform_azimuths(count, start)  # type: ignore[arg-type]


class TestAzimuthMean:
    """Tests for azimuth_mean."""

    def test_rules(self) -> None:
        """The interval mean matches the weights; the arithmetic one the plain mean."""
        angles = jittered(9, 0.3)
        stack = np.random.default_rng(2).normal(size=(9, 3, 2))
        np.testing.assert_allclose(
            azimuth_mean(angles, stack),
            np.tensordot(interval_weights(angles), stack, axes=(0, 0)),
        )
        np.testing.assert_allclose(
            azimuth_mean(angles, stack, rule="arithmetic"), stack.mean(axis=0)
        )

    def test_rejects_inconsistent_stack(self) -> None:
        """The stack must have one entry per azimuth and be finite."""
        with pytest.raises(ValueError, match="one entry per azimuth"):
            azimuth_mean(uniform_azimuths(4), np.zeros((3, 2, 2)))
        bad = np.zeros((4, 2, 2))
        bad[0, 0, 0] = np.inf
        with pytest.raises(ValueError, match="finite"):
            azimuth_mean(uniform_azimuths(4), bad)


class TestFitHarmonics:
    """Tests for fit_harmonics and HarmonicModel."""

    @pytest.mark.parametrize("count", [35, 36])
    def test_trigonometric_fit_interpolates(self, count: int) -> None:
        """The default trigonometric fit reproduces every sample."""
        angles = jittered(count, 0.3)
        stack = np.random.default_rng(4).normal(size=(count, 3, 3))
        model = fit_harmonics(angles, stack)
        assert model.interpolating
        assert model.nyquist_order == (count // 2 if count % 2 == 0 else None)
        assert model.resolved_order == (count - 1) // 2
        np.testing.assert_allclose(model.evaluate(angles), stack, atol=1e-10)
        np.testing.assert_allclose(
            model.coefficients[-5], np.conj(model.coefficients[5])
        )

    def test_recovers_exact_coefficients(self) -> None:
        """A trigonometric polynomial is recovered exactly from a non-uniform grid."""
        rng = np.random.default_rng(6)
        a0, a3 = rng.normal(size=(2, 2, 2))
        b3 = rng.normal(size=(2, 2))

        def a_of(psi: float) -> npt.NDArray[np.float64]:
            return a0 + a3 * np.cos(3 * psi) + b3 * np.sin(3 * psi)

        angles = jittered(36, 0.3)
        model = fit_harmonics(angles, sample(a_of, angles))
        np.testing.assert_allclose(model.mean, a0, atol=1e-11)
        np.testing.assert_allclose(
            model.coefficients[3], 0.5 * (a3 - 1j * b3), atol=1e-11
        )
        np.testing.assert_allclose(model.coefficients[4], 0.0, atol=1e-11)

    def test_blade_multiples_only(self) -> None:
        """A blade-multiple fit keeps only multiples of B and does not interpolate."""
        angles = uniform_azimuths(36)
        stack = np.random.default_rng(7).normal(size=(36, 2, 2))
        model = fit_harmonics(angles, stack, blades=3)
        assert not model.interpolating
        assert model.blades == 3
        assert all(h % 3 == 0 for h in model.coefficients)
        assert model.max_order == 18

    def test_piecewise_linear_coefficients(self) -> None:
        """The interpolant's Fourier coefficients: A0 is the interval mean."""
        angles = np.sort(jittered(11, 0.4, seed=9))
        values = np.random.default_rng(8).normal(size=11)
        model = fit_harmonics(angles, values[:, None], method="piecewise_linear")
        assert not model.interpolating
        assert abs(model.coefficients[0][0] - interval_weights(angles) @ values) < 1e-14
        dense = np.linspace(0.0, 2 * np.pi, 400_001)
        interpolant = np.interp(dense, angles, values, period=2 * np.pi)
        for h in (1, 3, 5):
            exact = trapezoid(interpolant * np.exp(-1j * h * dense), dense) / (
                2 * np.pi
            )
            assert abs(model.coefficients[h][0] - exact) < 1e-9

    def test_piecewise_linear_clustered_nodes_use_series(self) -> None:
        """Tiny gaps (the Taylor-series branch) give the same coefficients."""
        angles = np.array([0.0, 1e-4, 2.0, 2.0 + 2e-4, 4.0])
        values = np.array([1.0, 1.2, -0.5, -0.4, 0.3])
        model = fit_harmonics(angles, values[:, None], method="piecewise_linear")
        dense = np.linspace(0.0, 2 * np.pi, 2_000_001)
        interpolant = np.interp(dense, angles, values, period=2 * np.pi)
        exact = trapezoid(interpolant * np.exp(-2j * dense), dense) / (2 * np.pi)
        assert abs(model.coefficients[2][0] - exact) < 1e-8

    def test_uniform_mean_aliases_only_multiples(self) -> None:
        """The N-mean of a 3P/6P model is exact for N = 7 and aliases 6P for N = 6."""
        f = azimuth_system()
        model = fit_harmonics(uniform_azimuths(36), sample(f, uniform_azimuths(36)))
        np.testing.assert_allclose(model.uniform_mean(7), model.mean, atol=1e-14)
        start = 0.2
        expected = model.mean + 2.0 * (model.coefficients[6] * np.exp(6j * start)).real
        np.testing.assert_allclose(model.uniform_mean(6, start), expected, atol=1e-14)
        brute = sample(f, uniform_azimuths(6, start)).mean(axis=0)
        np.testing.assert_allclose(model.uniform_mean(6, start), brute, atol=1e-13)
        with pytest.raises(ValueError, match="count"):
            model.uniform_mean(0)

    def test_evaluate_vector_samples(self) -> None:
        """A model of vectors evaluates to the right shape."""
        angles = uniform_azimuths(8)
        model = fit_harmonics(angles, np.random.default_rng(0).normal(size=(8, 3)))
        assert model.evaluate([0.1, 0.2]).shape == (2, 3)

    def test_rejects_invalid_arguments(self) -> None:
        """Bad orders, blades, methods, designs and conditioning are rejected."""
        angles = uniform_azimuths(8)
        stack = np.zeros((8, 2, 2))
        with pytest.raises(ValueError, match="max_order"):
            fit_harmonics(angles, stack, max_order=0)
        with pytest.raises(ValueError, match="blades"):
            fit_harmonics(angles, stack, blades=0)
        with pytest.raises(ValueError, match="method"):
            fit_harmonics(angles, stack, method="spline")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="unknowns"):
            fit_harmonics(angles, stack, max_order=4)
        clustered = np.array([0.0, 1e-9, 2e-9, 2.0, 2.0 + 1e-9, 4.0, 4.0 + 1e-9])
        with pytest.raises(ValueError, match="ill-conditioned"):
            fit_harmonics(clustered, np.zeros((7, 1)))


class TestGridErrorMatrix:
    """Tests for grid_error_matrix (the quadrature-weight kernel identity)."""

    @pytest.mark.parametrize("step", [2, 3, 4])
    def test_kernel_identity_against_brute_force(self, step: int) -> None:
        """The kernel form equals the brute-force sub-grid mean error to 1e-10."""
        angles = jittered(36, 0.3, seed=11)
        stack = np.random.default_rng(12).normal(size=(36, 5, 5))
        model = fit_harmonics(angles, stack)
        reference = azimuth_mean(angles, stack)
        for offset in range(step):
            picks = np.arange(offset, 36, step)
            direct = np.tensordot(subgrid_weights(angles, picks), stack, axes=(0, 0))
            kernel = grid_error_matrix(model, angles[picks])
            relative = np.linalg.norm(kernel - (direct - reference)) / np.linalg.norm(
                direct - reference
            )
            assert relative < 1e-10

    def test_arbitrary_weights_and_subset(self) -> None:
        """The identity holds for any weights, e.g. the arithmetic rule on a subset."""
        angles = jittered(20, 0.25, seed=2)
        stack = np.random.default_rng(3).normal(size=(20, 3, 3))
        model = fit_harmonics(angles, stack)
        picks = np.array([0, 2, 5, 7, 11, 13, 17])
        weights = np.full(picks.size, 1.0 / picks.size)
        direct = stack[picks].mean(axis=0) - azimuth_mean(angles, stack)
        kernel = grid_error_matrix(model, angles[picks], weights)
        np.testing.assert_allclose(kernel, direct, atol=1e-10)

    def test_hypothetical_uniform_grid(self) -> None:
        """For a hypothetical uniform grid the error is the model's aliasing."""
        f = azimuth_system()
        model = fit_harmonics(uniform_azimuths(36), sample(f, uniform_azimuths(36)))
        grid = uniform_azimuths(6, 0.4)
        expected = sample(f, grid).mean(axis=0) - model.reference_mean
        np.testing.assert_allclose(grid_error_matrix(model, grid), expected, atol=1e-13)


class TestEigenBasis:
    """Tests for eigen_basis."""

    def test_biorthonormal(self) -> None:
        """Left and right vectors satisfy U^H V = I and reconstruct the matrix."""
        matrix = np.random.default_rng(5).normal(size=(6, 6))
        basis = eigen_basis(matrix)
        np.testing.assert_allclose(
            basis.left.conj().T @ basis.right, np.eye(6), atol=1e-10
        )
        reconstructed = basis.right @ np.diag(basis.values) @ basis.left.conj().T
        np.testing.assert_allclose(reconstructed, matrix, atol=1e-10)
        assert basis.nearest(basis.values[2]) == 2
        assert basis.participation().shape == (6, 6)
        assert np.all(basis.condition >= 1.0 - 1e-12)

    def test_rejects_invalid_and_defective(self) -> None:
        """Non-square, non-finite and defective matrices are rejected."""
        with pytest.raises(ValueError, match="square"):
            eigen_basis(np.zeros((2, 3)))
        with pytest.raises(ValueError, match="finite"):
            eigen_basis(np.array([[np.nan]]))
        with pytest.raises(ValueError, match="defective"):
            eigen_basis(np.array([[0.0, 1.0], [0.0, 0.0]]))


class TestLocateAzimuthState:
    """Tests for locate_azimuth_state."""

    _POSITION = "ED Variable speed generator DOF (internal DOF index = DOF_GeAz), rad"
    _SPEED = (
        "ED First time derivative of Variable speed generator DOF "
        "(internal DOF index = DOF_GeAz), rad/s"
    )
    _OTHER = (
        "ED Platform horizontal surge translation DOF (internal DOF index = DOF_Sg), m"
    )

    def test_finds_position_and_speed(self) -> None:
        """Both states are found from their descriptions, not a fixed index."""
        states = [self._OTHER, "ED Drivetrain (internal DOF index = DOF_DrTr), rad"]
        descriptions = [*states, self._POSITION, "a", "b", self._SPEED]
        assert locate_azimuth_state(descriptions) == AzimuthStates(2, 5)

    def test_speed_from_ndof2(self) -> None:
        """Without a velocity description the speed follows from ndof2."""
        descriptions = [self._OTHER, self._POSITION, "x", "y"]
        assert locate_azimuth_state(descriptions, ndof2=2) == AzimuthStates(1, 3)
        assert locate_azimuth_state(descriptions) == AzimuthStates(1, None)
        with pytest.raises(ValueError, match="ndof2"):
            locate_azimuth_state(descriptions, ndof2=1)

    def test_rejects_missing_or_ambiguous(self) -> None:
        """No or two azimuth position states are rejected."""
        with pytest.raises(ValueError, match="found no"):
            locate_azimuth_state([self._OTHER])
        with pytest.raises(ValueError, match="found 2"):
            locate_azimuth_state([self._POSITION, self._POSITION])


class TestAzimuthTreatments:
    """Tests for azimuth_tangent, pin_azimuth, zero_azimuth_column, deletion."""

    @staticmethod
    def _stack(delta: float = 1e-3) -> tuple[npt.NDArray[np.float64], ...]:
        angles = jittered(36, 0.3, seed=21)
        stack = sample(azimuth_system(delta=delta), angles)
        return angles, stack, azimuth_mean(angles, stack)

    def test_brauer_property(self) -> None:
        """Pinning moves the azimuth eigenvalue to 0 and keeps every other one."""
        angles, stack, mean = self._stack()
        tangent = azimuth_tangent(mean, AZIMUTH, n_samples=36)
        assert tangent[AZIMUTH.position] == 1.0
        pinned = pin_azimuth(stack, tangent, AZIMUTH)
        before = np.linalg.eigvals(mean)
        after = np.linalg.eigvals(azimuth_mean(angles, pinned))
        k_az = int(np.argmin(np.abs(before - 0.0048)))
        expected = before.copy()
        expected[k_az] = 0.0
        np.testing.assert_allclose(
            np.sort_complex(after), np.sort_complex(expected), atol=1e-10
        )

    def test_tangent_is_null_vector_of_every_sample(self) -> None:
        """A' xi = 0 for every sample; every sub-grid mean has a zero eigenvalue."""
        angles, stack, mean = self._stack()
        tangent = azimuth_tangent(mean, AZIMUTH)
        pinned = pin_azimuth(stack, tangent, AZIMUTH)
        residual = np.linalg.norm(pinned @ tangent, axis=1)
        assert np.max(residual / np.linalg.norm(stack, axis=(1, 2))) < 1e-12
        for step in (3, 4):
            sub = np.tensordot(
                subgrid_weights(angles, np.arange(0, 36, step)), pinned, 1
            )
            assert np.min(np.abs(np.linalg.eigvals(sub))) < 1e-12
        # unit Floquet multiplier of the pinned periodic model
        exponents = monodromy_exponents(pinned, angles, OMEGA)
        assert np.min(np.abs(exponents)) < 1e-10

    def test_single_matrix_and_normalisation(self) -> None:
        """A single matrix is pinned; the tangent is rescaled to xi_az = 1."""
        _angles, _stack, mean = self._stack()
        tangent = azimuth_tangent(mean, AZIMUTH.position)
        pinned_unit = pin_azimuth(mean, tangent, AZIMUTH)
        pinned_scaled = pin_azimuth(mean, 3.0 * tangent, 1)
        np.testing.assert_allclose(pinned_unit, pinned_scaled)
        e_az = np.zeros(4)
        e_az[1] = 1.0
        np.testing.assert_allclose(
            pin_azimuth(mean, e_az, AZIMUTH), zero_azimuth_column(mean, AZIMUTH)
        )

    def test_naive_neutralisation_biases_reference(self) -> None:
        """Zeroing the column changes the reference; deletion equals it minus zero."""
        _angles, stack, mean = self._stack(delta=0.0)
        zeroed = zero_azimuth_column(mean, AZIMUTH)
        deleted = delete_azimuth_state(mean, AZIMUTH)
        assert deleted.shape == (3, 3)
        assert delete_azimuth_state(stack, AZIMUTH).shape == (36, 3, 3)
        z_values = np.linalg.eigvals(zeroed)
        k_zero = int(np.argmin(np.abs(z_values)))
        np.testing.assert_allclose(
            np.sort_complex(np.delete(z_values, k_zero)),
            np.sort_complex(np.linalg.eigvals(deleted)),
            atol=1e-12,
        )
        platform = platform_index(np.linalg.eigvals(mean))
        reference = np.linalg.eigvals(mean)[platform]
        treated = z_values[platform_index(z_values)]
        assert abs(treated - reference) > 1e-6

    def test_warnings(self) -> None:
        """A coarse grid and a complex azimuth eigenvalue warn."""
        _angles, _stack, mean = self._stack()
        with pytest.warns(RuntimeWarning, match="contaminated"):
            azimuth_tangent(mean, AZIMUTH, n_samples=12)
        oscillatory = mean.copy()
        oscillatory[3, 1] = -0.05  # negative azimuth stiffness: a pendulum pair
        with pytest.warns(RuntimeWarning, match="complex"):
            azimuth_tangent(oscillatory, AZIMUTH)

    def test_rejects_invalid(self) -> None:
        """Bad shapes, non-finite input and zero azimuth components are rejected."""
        _angles, stack, mean = self._stack()
        with pytest.raises(ValueError, match="non-zero"):
            pin_azimuth(mean, np.array([1.0, 0.0, 0.0, 0.0]), AZIMUTH)
        with pytest.raises(ValueError, match="length"):
            pin_azimuth(mean, np.ones(3), AZIMUTH)
        with pytest.raises(ValueError, match=r"\(n, n\)"):
            pin_azimuth(np.zeros((2, 3)), np.ones(3), 1)
        with pytest.raises(ValueError, match="finite"):
            pin_azimuth(np.full((2, 2), np.nan), np.ones(2), 1)
        with pytest.raises(ValueError, match="out of range"):
            pin_azimuth(stack, np.ones(4), 7)
        with pytest.raises(ValueError, match=r"\(n, n\)"):
            zero_azimuth_column(np.zeros(3), 0)
        with pytest.raises(ValueError, match=r"\(n, n\)"):
            delete_azimuth_state(np.zeros((2, 3)), 0)


class TestSidebandCorrection:
    """Tests for sideband_correction against Floquet exponents."""

    def _errors(self, eps: float) -> tuple[float, float]:
        f = azimuth_system(eps=eps)
        angles = uniform_azimuths(36)
        model = fit_harmonics(angles, sample(f, angles))
        basis = eigen_basis(model.reference_mean)
        j = platform_index(basis.values)
        averaged = complex(basis.values[j])
        exact = matched_exponent(floquet_exponents(f), averaged)
        corrected = averaged + sideband_correction(model, basis, j, OMEGA)
        return abs(exact - averaged), abs(exact - corrected)

    def test_first_and_second_order_scaling(self) -> None:
        """The averaged error is O(eps^2); with the sideband correction, O(eps^4)."""
        first_large, second_large = self._errors(0.05)
        first_small, second_small = self._errors(0.025)
        assert 3.5 < first_large / first_small < 4.5
        assert 12.0 < second_large / second_small < 20.0
        assert second_large < 1e-3 * first_large

    def test_max_order_and_validation(self) -> None:
        """Truncating the harmonics changes the sum; bad arguments are rejected."""
        f = azimuth_system()
        angles = uniform_azimuths(36)
        model = fit_harmonics(angles, sample(f, angles))
        basis = eigen_basis(model.reference_mean)
        j = platform_index(basis.values)
        full = sideband_correction(model, basis, j, OMEGA)
        partial = sideband_correction(model, basis, j, OMEGA, max_order=3)
        assert full != partial
        with pytest.raises(ValueError, match="rotor_speed"):
            sideband_correction(model, basis, j, 0.0)
        with pytest.raises(ValueError, match="out of range"):
            sideband_correction(model, basis, 9, OMEGA)


class TestAzimuthConsistency:
    """Tests for azimuth_consistency."""

    def test_periodic_orbit_is_consistent(self) -> None:
        """An exact orbit tangent gives a zero azimuth exponent to O(eps^4)."""
        f = azimuth_system()
        angles = jittered(36, 0.2)
        model = fit_harmonics(angles, sample(f, angles), method="piecewise_linear")
        result = azimuth_consistency(model, OMEGA, AZIMUTH, tolerance=1e-5)
        assert isinstance(result, AzimuthConsistency)
        assert result.consistent
        assert abs(result.eigenvalue) > 10 * abs(result.exponent)
        assert result.azimuth_share > 0.4
        assert result.speed_damping < 0.0
        assert result.pendulum_frequency == 0.0

    def test_trigonometric_model_is_exact(self) -> None:
        """With exact harmonics the residual is fourth order in the content."""
        f = azimuth_system()
        model = fit_harmonics(uniform_azimuths(36), sample(f, uniform_azimuths(36)))
        result = azimuth_consistency(model, OMEGA, AZIMUTH)
        assert abs(result.exponent) < 1e-7
        assert abs(result.eigenvalue) > 1e-5

    def test_non_periodic_case_is_flagged(self) -> None:
        """A spurious azimuth stiffness is detected and matches the Floquet exponent."""
        f = azimuth_system(delta=1e-3)
        angles = uniform_azimuths(36)
        model = fit_harmonics(angles, sample(f, angles))
        result = azimuth_consistency(model, OMEGA, AZIMUTH)
        assert not result.consistent
        exact = floquet_exponents(f)
        floquet = complex(exact[np.argmin(np.abs(exact - result.exponent))])
        assert abs(floquet - result.exponent) < 1e-6
        assert result.stiffness > 0.0

    def test_pendulum_and_without_speed(self) -> None:
        """A negative stiffness reports a pendulum; no speed state gives NaN."""
        f = azimuth_system(delta=-0.05)
        angles = uniform_azimuths(36)
        model = fit_harmonics(angles, sample(f, angles))
        result = azimuth_consistency(model, OMEGA, AZIMUTH)
        assert result.pendulum_frequency > 0.1
        positional = azimuth_consistency(model, OMEGA, 1)
        assert np.isnan(positional.stiffness)
        assert positional.pendulum_frequency == 0.0

    def test_validation(self) -> None:
        """Invalid tolerance, rotor speed and indices are rejected."""
        f = azimuth_system()
        angles = uniform_azimuths(36)
        model = fit_harmonics(angles, sample(f, angles))
        with pytest.raises(ValueError, match="tolerance"):
            azimuth_consistency(model, OMEGA, AZIMUTH, tolerance=0.0)
        with pytest.raises(ValueError, match="rotor_speed"):
            azimuth_consistency(model, -1.0, AZIMUTH)
        with pytest.raises(ValueError, match="rotor-speed state"):
            azimuth_consistency(model, OMEGA, AzimuthStates(1, 9))
        vector_model = fit_harmonics(angles, np.zeros((36, 4)))
        with pytest.raises(ValueError, match="square"):
            azimuth_consistency(vector_model, OMEGA, AZIMUTH)

    def test_saddle_reports_the_growing_root(self) -> None:
        """In a saddle the growing exponent is reported, matching the Floquet one."""
        f = azimuth_system(b0=0.001, delta=1e-3)
        angles = uniform_azimuths(36)
        result = azimuth_consistency(
            fit_harmonics(angles, sample(f, angles)), OMEGA, AZIMUTH
        )
        growing, stable = result.pair_exponents
        assert growing.real > 0.02 > -0.02 > stable.real
        assert result.exponent == growing
        assert result.growth_rate == pytest.approx(growing.real)
        largest = max(floquet_exponents(f).real)
        assert result.growth_rate == pytest.approx(largest, abs=1e-6)
        assert result.residual_stiffness == pytest.approx(1e-3, rel=0.02)
        assert result.residual_exponent == pytest.approx(min(abs(growing), abs(stable)))
        assert result.pair_indices[0] == result.index
        assert result.pair_eigenvalues[0] == result.eigenvalue
        assert result.speed_damping_exponent == pytest.approx((growing + stable).real)

    def test_positive_speed_damping_is_not_a_defect(self) -> None:
        """An open-loop rotor-speed divergence grows but is consistent."""
        f = azimuth_system(b0=-0.05)
        angles = uniform_azimuths(36)
        result = azimuth_consistency(
            fit_harmonics(angles, sample(f, angles)), OMEGA, AZIMUTH
        )
        assert result.growth_rate == pytest.approx(0.05, rel=0.01)
        assert abs(result.residual_stiffness) < 1e-6
        assert result.assessment == "consistent"
        assert result.consistent
        assert result.absolute_consistent

    @pytest.mark.parametrize(
        ("delta", "assessment"),
        [
            (0.0, "consistent"),
            (1e-4, "inconsistent-but-harmless"),
            (1e-3, "inconsistent-and-risky"),
            (-2e-3, "inconsistent-and-risky"),
        ],
    )
    def test_graded_assessment(self, delta: float, assessment: str) -> None:
        """The scaled test grades the residual against the platform branch."""
        f = azimuth_system(delta=delta)
        angles = uniform_azimuths(36)
        result = azimuth_consistency(
            fit_harmonics(angles, sample(f, angles)), OMEGA, AZIMUTH
        )
        assert result.assessment == assessment
        (risk,) = result.branch_risks
        assert risk.decay_ratio == pytest.approx(
            result.residual_exponent / -risk.eigenvalue.real
        )
        threshold = risk.eigenvalue.imag**2 + 0.25 * result.speed_damping_exponent**2
        assert risk.coalescence_ratio == pytest.approx(
            abs(result.residual_stiffness) / threshold
        )
        assert risk.risky is (assessment == "inconsistent-and-risky")
        assert result.worst_branch == risk
        # the absolute test alone rejects every inconsistent case
        assert result.absolute_consistent is (delta == 0.0)

    def test_harmless_against_a_fast_branch(self) -> None:
        """The same residual is harmless for a branch a decade above it."""
        f = azimuth_system(delta=1e-3, k0=9.0, cp=0.2)
        angles = uniform_azimuths(36)
        result = azimuth_consistency(
            fit_harmonics(angles, sample(f, angles)), OMEGA, AZIMUTH
        )
        assert result.residual_stiffness > 5e-4
        assert result.assessment == "inconsistent-but-harmless"
        loose = azimuth_consistency(
            fit_harmonics(angles, sample(azimuth_system(delta=1e-3), angles)),
            OMEGA,
            AZIMUTH,
            scale_tolerance=0.9,
        )
        assert loose.assessment == "consistent"

    def test_explicit_branches_and_fallback(self) -> None:
        """No branch falls back to the absolute test; bad indices are rejected."""
        f = azimuth_system(delta=1e-3)
        angles = uniform_azimuths(36)
        model = fit_harmonics(angles, sample(f, angles))
        empty = azimuth_consistency(model, OMEGA, AZIMUTH, branches=[])
        assert empty.branch_risks == ()
        assert empty.worst_branch is None
        assert empty.assessment == "inconsistent-but-harmless"
        exact = azimuth_consistency(
            fit_harmonics(angles, sample(azimuth_system(), angles)),
            OMEGA,
            AZIMUTH,
            branches=[],
        )
        assert exact.assessment == "consistent"
        with pytest.raises(ValueError, match="branch indices"):
            azimuth_consistency(model, OMEGA, AZIMUTH, branches=[9])
        with pytest.raises(ValueError, match="scale_tolerance"):
            azimuth_consistency(model, OMEGA, AZIMUTH, scale_tolerance=0.0)

    def test_undamped_branch_and_single_state(self) -> None:
        """A non-decaying branch has an infinite decay ratio; one state has no pair."""
        basis = eigen_basis(
            np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1e-3]])
        )
        risks = averaging_module._branch_risks(basis, [0], 1e-3, 1e-6, 0.0, 0.01)
        assert risks[0].decay_ratio == np.inf
        single = fit_harmonics(uniform_azimuths(5), np.full((5, 1, 1), 2e-3))
        result = azimuth_consistency(single, OMEGA, 0)
        assert result.pair_indices == (0, 0)
        assert result.residual_stiffness == 0.0
        assert result.residual_exponent == pytest.approx(2e-3)
        assert result.residual_pendulum_frequency == pytest.approx(0.0, abs=1e-12)

    def test_singular_condensation_returns_nan(self) -> None:
        """A singular remaining block leaves the condensation undefined."""
        matrix = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, -0.1]])
        kappa, damping = averaging_module._condense(matrix, 0, 2)
        assert np.isnan(kappa)
        assert np.isnan(damping)
        kappa, damping = averaging_module._condense(
            matrix[np.ix_([0, 2], [0, 2])], 0, 1
        )
        assert (kappa, damping) == (0.0, -0.1)


def tower_column_stack(
    rho: float, count: int = 36, *, speed: bool = True
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return samples whose azimuth column follows the tower-passage law exactly."""
    angles = uniform_azimuths(count)

    def a_of(psi: float) -> npt.NDArray[np.float64]:
        matrix = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [-0.09, 0.0, -0.02, 0.01],
                [0.01, 0.0, 0.0, -0.2],
            ]
        )
        law = sum(
            (h / 3) * rho ** (h - 3) * np.cos(h * psi + 0.4 * h)
            for h in range(3, 61, 3)
        )
        matrix[3 if speed else 2, 1] = 1e-3 * law
        return matrix

    return angles, sample(a_of, angles)


class TestTowerPassageDecay:
    """Tests for tower_passage_decay."""

    @pytest.mark.parametrize("rho", [0.7, 0.85])
    def test_recovers_the_law(self, rho: float) -> None:
        """The condensed harmonics of an exact law give its decay."""
        angles, stack = tower_column_stack(rho)
        fit = tower_passage_decay(fit_harmonics(angles, stack), AZIMUTH)
        assert isinstance(fit, TowerPassageFit)
        assert fit.condensed
        assert fit.decay == pytest.approx(rho, rel=0.02)
        assert fit.log_r2 > 0.99
        assert list(fit.orders) == [3, 6, 9, 12, 15]
        assert fit.amplitude == pytest.approx(5e-4, rel=0.1)

    def test_column_norm_without_speed_state(self) -> None:
        """Without the speed state the column norms are fitted."""
        angles, stack = tower_column_stack(0.8, speed=False)
        fit = tower_passage_decay(fit_harmonics(angles, stack), 1)
        assert not fit.condensed
        assert fit.decay == pytest.approx(0.8, rel=0.02)

    def test_validation(self) -> None:
        """Bad blade counts, too few orders and a singular block are rejected."""
        angles, stack = tower_column_stack(0.8)
        model = fit_harmonics(angles, stack)
        with pytest.raises(ValueError, match="blades"):
            tower_passage_decay(model, AZIMUTH, blades=0)
        with pytest.raises(ValueError, match="fewer than two"):
            tower_passage_decay(model, AZIMUTH, blades=18)
        flat = fit_harmonics(angles, np.repeat(stack[:1], 36, axis=0))
        with pytest.raises(ValueError, match="fewer than two"):
            tower_passage_decay(flat, AZIMUTH)
        singular = np.zeros((36, 4, 4))
        with pytest.raises(ValueError, match="singular"):
            tower_passage_decay(fit_harmonics(angles, singular), AZIMUTH)
