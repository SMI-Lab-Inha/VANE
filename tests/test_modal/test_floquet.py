# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.modal.floquet — Floquet/Hill modal analysis of LTP systems."""

import warnings

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment

from vane.modal import floquet as floquet_module
from vane.modal.floquet import (
    averaging_error,
    floquet_modes,
    monodromy_exponents,
)

_OMEGA = 2.0 * np.pi * 1.0  # rotor at 1 Hz (60 rpm)


def _oscillator(freq: float, zeta: float) -> np.ndarray:
    """Return a 2x2 state matrix for a damped oscillator (slow vs the rotor)."""
    wn = 2.0 * np.pi * freq
    return np.array([[0.0, 1.0], [-(wn**2), -2.0 * zeta * wn]])


def _two_mode() -> np.ndarray:
    """Return a 4-state matrix with two slow modes (0.10 and 0.25 Hz)."""
    return np.block(
        [
            [_oscillator(0.10, 0.04), np.zeros((2, 2))],
            [np.zeros((2, 2)), _oscillator(0.25, 0.03)],
        ]
    )


def _samples(matrix_fn: object, n_azimuth: int) -> tuple[list[np.ndarray], np.ndarray]:
    """Sample a matrix-valued function of azimuth over one revolution."""
    angles = np.linspace(0.0, 2.0 * np.pi, n_azimuth, endpoint=False)
    return [matrix_fn(psi) for psi in angles], angles  # type: ignore[operator]


def _sortc(values: np.ndarray) -> np.ndarray:
    """Sort complex values by (imag, real) — stable across conjugate-pair ties."""
    return values[np.lexsort((values.real, values.imag))]


class TestLtiLimit:
    """An isotropic rotor (constant A) reduces to the ordinary eigenvalues."""

    def test_hill_recovers_eigenvalues(self) -> None:
        """Hill exponents equal eig(A) when A is azimuth-invariant."""
        a = _two_mode()
        per, az = _samples(lambda _psi: a, 21)
        sol = floquet_modes(per, az, _OMEGA, n_harmonics=3)
        np.testing.assert_allclose(
            _sortc(sol.exponents), _sortc(np.linalg.eigvals(a)), atol=1e-7
        )

    def test_monodromy_recovers_eigenvalues(self) -> None:
        """Monodromy exponents equal eig(A) when A is azimuth-invariant."""
        a = _two_mode()
        per, az = _samples(lambda _psi: a, 21)
        exponents = monodromy_exponents(per, az, _OMEGA)
        np.testing.assert_allclose(
            _sortc(exponents), _sortc(np.linalg.eigvals(a)), atol=1e-7
        )

    def test_mode_shapes_concentrate_in_the_fundamental(self) -> None:
        """For constant A the harmonic mode shape lives in the m=0 block."""
        a = _two_mode()
        per, az = _samples(lambda _psi: a, 21)
        sol = floquet_modes(per, az, _OMEGA, n_harmonics=3)
        fundamental = np.linalg.norm(sol.harmonic_mode_shapes[sol.harmonics == 0])
        sidebands = np.linalg.norm(sol.harmonic_mode_shapes[sol.harmonics != 0])
        assert sidebands < 1e-6 * fundamental


class TestHillVersusMonodromy:
    """Hill and the monodromy matrix agree on a genuinely periodic system."""

    def test_methods_agree(self) -> None:
        """Both methods agree for modes within the fundamental strip (|f| < Omega/2)."""
        a0, a1, a2 = _two_mode(), np.zeros((4, 4)), np.zeros((4, 4))
        a1[0, 2] = a1[2, 0] = 0.4  # azimuth-periodic inter-mode coupling
        a2[1, 3] = a2[3, 1] = 0.3

        def matrix(psi: float) -> np.ndarray:
            return np.asarray(a0 + a1 * np.cos(psi) + a2 * np.sin(psi))

        per, az = _samples(matrix, 180)
        hill = floquet_modes(per, az, _OMEGA, n_harmonics=3).exponents
        mono = monodromy_exponents(per, az, _OMEGA)
        np.testing.assert_allclose(_sortc(hill), _sortc(mono), atol=5e-3)

    def test_hill_unfolds_where_monodromy_folds(self) -> None:
        """Above Omega/2: Hill gives the true frequency; the monodromy folds it."""
        a = _oscillator(0.8, 0.01)  # 0.8 Hz, above Omega/2 = 0.5 Hz for the 1 Hz rotor
        per, az = _samples(lambda _psi: a, 9)
        hill = floquet_modes(per, az, _OMEGA, n_harmonics=2)
        mono = monodromy_exponents(per, az, _OMEGA)
        # Hill recovers the true 0.8 Hz; the monodromy aliases to |0.8 - 1.0| = 0.2 Hz.
        assert hill.natural_frequencies_hz.max() == pytest.approx(0.8, abs=2e-3)
        assert float(np.abs(mono.imag).max()) / (2.0 * np.pi) == pytest.approx(
            0.2, abs=2e-3
        )


class TestAveragingError:
    """The averaging error is zero for isotropy and grows with anisotropy."""

    def test_isotropic_has_zero_error(self) -> None:
        """A constant (isotropic) system has no averaging error."""
        a = _two_mode()
        per, az = _samples(lambda _psi: a, 21)
        assert averaging_error(per, az, _OMEGA, n_harmonics=3) < 1e-6

    def test_anisotropic_has_nonzero_error(self) -> None:
        """A strongly azimuth-dependent system deviates from the averaged model."""
        a0 = _two_mode()
        a1 = np.zeros((4, 4))
        a1[0, 2] = a1[2, 0] = 0.8

        def matrix(psi: float) -> np.ndarray:
            return np.asarray(a0 + a1 * np.cos(psi))

        per, az = _samples(matrix, 90)
        assert averaging_error(per, az, _OMEGA, n_harmonics=4) > 1e-3


_SLOW_OMEGA = 2.0 * np.pi * 12.0 / 60.0  # 12 rpm rotor


def _mathieu(ratio: float, eps: float, zeta: float) -> object:
    """Return ``A(psi)`` of a damped Mathieu oscillator, ``w0 = ratio * Omega``."""
    w0 = ratio * _SLOW_OMEGA

    def matrix(psi: float) -> np.ndarray:
        return np.array(
            [[0.0, 1.0], [-(w0**2) * (1.0 + eps * np.cos(psi)), -2.0 * zeta * w0]]
        )

    return matrix


def _ode_exponents(matrix_fn: object, omega: float) -> np.ndarray:
    """Return reference Floquet exponents from an ODE-integrated monodromy matrix."""
    period = 2.0 * np.pi / omega

    def rhs(t: float, x: np.ndarray) -> np.ndarray:
        a = matrix_fn(omega * t)  # type: ignore[operator]
        return np.asarray((a @ x.reshape(2, 2)).ravel())

    sol = solve_ivp(rhs, (0.0, period), np.eye(2).ravel(), rtol=1e-12, atol=1e-14)
    multipliers = np.linalg.eigvals(sol.y[:, -1].reshape(2, 2))
    return np.asarray(np.log(multipliers.astype(np.complex128)) / period)


def _family_mismatch(hill: np.ndarray, reference: np.ndarray, omega: float) -> float:
    """Return the worst matched exponent gap modulo ``i*omega`` (same multipliers)."""
    diff = hill[:, None] - reference[None, :]
    gaps = np.abs(diff - 1j * np.round(diff.imag / omega) * omega)
    rows, cols = linear_sum_assignment(gaps)
    return float(gaps[rows, cols].max())


class TestParametricResonance:
    """Principal exponents are one per Floquet family, matching the monodromy."""

    def test_unstable_tongue_keeps_both_families(self) -> None:
        """Inside the 0.5P tongue both real-multiplier families, one unstable, are kept.

        The two exponents are ``sigma_1 + i*Omega/2`` and ``sigma_2 + i*Omega/2``; the
        conjugate of each is a shifted copy of itself, so picking both conjugates of
        one family would drop the other (here the growing one).
        """
        matrix = _mathieu(0.5, 0.1, 0.005)
        per, az = _samples(matrix, 36)
        reference = _ode_exponents(matrix, _SLOW_OMEGA)
        assert reference.real.max() > 0.0  # the reference really is unstable
        sol = floquet_modes(per, az, _SLOW_OMEGA, n_harmonics=4)
        assert _family_mismatch(sol.exponents, reference, _SLOW_OMEGA) < 1e-6
        assert sol.exponents.real.max() == pytest.approx(reference.real.max(), abs=1e-6)
        assert sol.damping_ratios.min() < 0.0

    def test_stable_tongue_keeps_both_families(self) -> None:
        """A damped 0.5P tongue (both decaying at different rates) keeps both rates."""
        matrix = _mathieu(0.5, 0.02, 0.02)
        per, az = _samples(matrix, 36)
        reference = _ode_exponents(matrix, _SLOW_OMEGA)
        assert reference.real.max() < 0.0
        assert abs(reference.real[0] - reference.real[1]) > 1e-3
        sol = floquet_modes(per, az, _SLOW_OMEGA, n_harmonics=4)
        assert _family_mismatch(sol.exponents, reference, _SLOW_OMEGA) < 1e-6

    def test_isotropic_mode_at_half_omega_is_a_conjugate_pair(self) -> None:
        """An isotropic mode at exactly Omega/2 returns eig(A) = +-i*Omega/2, not twice.

        ``+i*Omega/2`` and ``-i*Omega/2`` share a multiplier (-1, double) but are two
        distinct solutions; the Hill eigenvalue there is degenerate.
        """
        matrix = _mathieu(0.5, 0.0, 0.0)
        a = matrix(0.0)  # type: ignore[operator]
        per, az = _samples(matrix, 36)
        sol = floquet_modes(per, az, _SLOW_OMEGA, n_harmonics=4)
        np.testing.assert_allclose(
            _sortc(sol.exponents), _sortc(np.linalg.eigvals(a)), atol=1e-9
        )
        reference = _ode_exponents(matrix, _SLOW_OMEGA)
        assert _family_mismatch(sol.exponents, reference, _SLOW_OMEGA) < 1e-6
        fundamental = np.linalg.norm(sol.harmonic_mode_shapes[sol.harmonics == 0])
        sidebands = np.linalg.norm(sol.harmonic_mode_shapes[sol.harmonics != 0])
        assert sidebands < 1e-6 * fundamental

    def test_mode_shapes_and_averaging_error_use_the_selected_set(self) -> None:
        """The mode shapes are the selected exponents' Hill eigenvectors.

        Each ``p_m`` stack must satisfy the Hill eigen-equation for its exponent, and
        :func:`averaging_error` must be computed from those same frequencies.
        """
        matrix = _mathieu(0.5, 0.1, 0.005)
        per, az = _samples(matrix, 36)
        sol = floquet_modes(per, az, _SLOW_OMEGA, n_harmonics=4)
        shapes = sol.harmonic_mode_shapes
        # Distinct families -> linearly independent fundamental mode shapes.
        fundamental = shapes[sol.harmonics == 0][0]
        assert np.linalg.matrix_rank(fundamental, tol=1e-6) == 2
        averaged_hz = np.abs(np.linalg.eigvals(np.mean(per, axis=0))) / (2.0 * np.pi)
        expected = np.max(np.abs(np.sort(sol.natural_frequencies_hz) - averaged_hz))
        assert averaging_error(per, az, _SLOW_OMEGA, n_harmonics=4) == pytest.approx(
            expected, abs=1e-12
        )

    def test_warns_when_hill_disagrees_with_monodromy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A Hill set that does not reproduce the monodromy multipliers warns."""
        matrix = _mathieu(0.3, 0.0, 0.01)
        per, az = _samples(matrix, 36)

        def shifted(*args: object) -> np.ndarray:
            return np.asarray([-1.0, 0.5], dtype=np.complex128)

        monkeypatch.setattr(floquet_module, "_monodromy", shifted)
        with pytest.warns(RuntimeWarning, match="monodromy"):
            floquet_modes(per, az, _SLOW_OMEGA, n_harmonics=4)

    def test_agreeing_case_does_not_warn(self) -> None:
        """A well-resolved anisotropic case passes the monodromy check silently."""
        matrix = _mathieu(0.52, 0.05, 0.005)
        per, az = _samples(matrix, 36)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            floquet_modes(per, az, _SLOW_OMEGA, n_harmonics=4)


class TestValidation:
    """Input validation."""

    def test_rejects_standstill(self) -> None:
        """A non-spinning rotor (Omega <= 0) is rejected."""
        per, az = _samples(lambda _psi: _two_mode(), 21)
        with pytest.raises(ValueError, match="rotor_speed must be positive"):
            floquet_modes(per, az, 0.0)

    def test_rejects_duplicate_azimuths(self) -> None:
        """Duplicate azimuths are rejected: a zero-width interval zero-weights a sample.

        Without this guard ``[0, 0, pi]`` would give the first matrix zero weight in the
        monodromy product and return a quietly wrong result instead of raising.
        """
        per = np.array([[[-1.0]], [[-2.0]], [[-3.0]]], dtype=np.float64)
        with pytest.raises(ValueError, match="distinct"):
            monodromy_exponents(per, [0.0, 0.0, np.pi], _OMEGA)

    def test_rejects_non_finite_rotor_speed(self) -> None:
        """A non-finite rotor speed is rejected (NaN slips past a bare ``<= 0``)."""
        per, az = _samples(lambda _psi: _two_mode(), 21)
        with pytest.raises(ValueError, match="positive and finite"):
            monodromy_exponents(per, az, np.nan)

    def test_rejects_non_finite_matrix(self) -> None:
        """A non-finite matrix entry is rejected before it reaches expm/eigvals."""
        per, az = _samples(lambda _psi: _two_mode(), 21)
        bad = np.array(per, dtype=np.float64)
        bad[0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="per_azimuth_a must be finite"):
            monodromy_exponents(bad, az, _OMEGA)

    def test_rejects_too_few_azimuths(self) -> None:
        """Too few azimuth samples for the requested harmonics is rejected."""
        per, az = _samples(lambda _psi: _two_mode(), 5)
        with pytest.raises(ValueError, match="need at least"):
            floquet_modes(per, az, _OMEGA, n_harmonics=4)

    def test_rejects_inconsistent_matrices(self) -> None:
        """Non-square or mismatched inputs are rejected."""
        with pytest.raises(ValueError, match="must be"):
            floquet_modes([np.ones((2, 3))], [0.0], _OMEGA, n_harmonics=1)

    def test_rejects_nonuniform_sweep(self) -> None:
        """A nonuniform azimuth sweep is rejected (Hill needs uniform spacing)."""
        angles = 2.0 * np.pi * np.linspace(0.0, 1.0, 20, endpoint=False) ** 2
        per = [_two_mode() for _ in angles]
        with pytest.raises(ValueError, match="near-uniform"):
            floquet_modes(per, angles, _OMEGA, n_harmonics=4)


class TestHillUniformGrid:
    """Hill's method needs near-uniform azimuths; the monodromy route does not."""

    @staticmethod
    def _uneven(n_azimuth: int, shift_deg: float) -> np.ndarray:
        """Return a uniform grid with one azimuth moved by ``shift_deg``."""
        angles = np.linspace(0.0, 2.0 * np.pi, n_azimuth, endpoint=False)
        angles[5] += np.radians(shift_deg)
        return angles

    def test_message_names_the_worst_interval(self) -> None:
        """The error states the tolerance, the worst interval and the alternative."""
        angles = self._uneven(36, 0.5)  # a 10 deg grid with one sample 0.5 deg late
        per = [_two_mode() for _ in angles]
        with pytest.raises(ValueError, match="near-uniform azimuths") as excinfo:
            floquet_modes(per, angles, _OMEGA, n_harmonics=4)
        message = str(excinfo.value)
        assert "relative 0.001" in message
        # The moved sample bounds the two worst intervals, 10.5 and 9.5 deg.
        assert "50.5 deg" in message
        assert "relative deviation of 0.05" in message
        assert "monodromy_exponents" in message

    def test_closing_interval_is_checked(self) -> None:
        """The interval that closes the revolution counts like any other."""
        angles = np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False) * (350.0 / 360.0)
        per = [_two_mode() for _ in angles]
        with pytest.raises(ValueError, match="near-uniform azimuths"):
            floquet_modes(per, angles, _OMEGA, n_harmonics=4)

    def test_averaging_error_has_the_same_restriction(self) -> None:
        """averaging_error uses Hill's method and rejects the same grid."""
        angles = self._uneven(36, 0.5)
        per = [_two_mode() for _ in angles]
        with pytest.raises(ValueError, match="near-uniform azimuths"):
            averaging_error(per, angles, _OMEGA, n_harmonics=4)

    def test_deviation_within_tolerance_is_accepted(self) -> None:
        """A deviation below the relative 1e-3 tolerance is accepted."""
        assert pytest.approx(1e-3) == floquet_module.HILL_UNIFORM_RTOL
        angles = self._uneven(36, 0.005)  # 5e-4 of the 10 deg interval
        a = _two_mode()
        sol = floquet_modes([a for _ in angles], angles, _OMEGA, n_harmonics=4)
        np.testing.assert_allclose(
            _sortc(sol.exponents), _sortc(np.linalg.eigvals(a)), atol=1e-7
        )

    def test_monodromy_handles_the_uneven_grid(self) -> None:
        """The monodromy route analyses the grid Hill rejects.

        For an azimuth-invariant matrix the multipliers are exactly ``exp(eig(A) T)``
        on any grid of distinct azimuths, however uneven.
        """
        angles = self._uneven(36, 0.5)
        a = _two_mode()
        exponents = monodromy_exponents([a for _ in angles], angles, _OMEGA)
        np.testing.assert_allclose(
            _sortc(exponents), _sortc(np.linalg.eigvals(a)), atol=1e-9
        )
