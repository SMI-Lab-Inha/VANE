# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.modal.calibration — azimuth-density convergence."""

import numpy as np
import pytest

from vane.modal.calibration import (
    AzimuthConvergence,
    azimuth_density_sweep,
    recommend_azimuth_count,
)


def _oscillator(freq: float, zeta: float) -> np.ndarray:
    """Return a 2x2 damped-oscillator state matrix."""
    wn = 2.0 * np.pi * freq
    return np.array([[0.0, 1.0], [-(wn**2), -2.0 * zeta * wn]])


def _base() -> np.ndarray:
    """Return a 4-state base matrix with two modes (0.10 and 0.25 Hz)."""
    return np.block(
        [
            [_oscillator(0.10, 0.04), np.zeros((2, 2))],
            [np.zeros((2, 2)), _oscillator(0.25, 0.03)],
        ]
    )


def _periodic_samples(n_azimuth: int) -> tuple[np.ndarray, np.ndarray]:
    """Return per-azimuth matrices with harmonics 1..6 of an inter-mode coupling."""
    base = _base()
    coupling = np.zeros((4, 4))
    coupling[0, 2] = coupling[2, 0] = 1.0
    angles = np.linspace(0.0, 2.0 * np.pi, n_azimuth, endpoint=False)
    matrices = np.array(
        [
            base + coupling * sum((0.3 / k) * np.cos(k * psi) for k in range(1, 7))
            for psi in angles
        ]
    )
    return matrices, angles


class TestAzimuthDensitySweep:
    """Tests for azimuth_density_sweep."""

    def test_converges_with_azimuth_count(self) -> None:
        """The averaged-model frequency error shrinks as more azimuths are used."""
        per, az = _periodic_samples(180)
        conv = azimuth_density_sweep(per, az, densities=(1, 3, 6, 12, 36))
        assert list(conv.densities) == [1, 3, 6, 12, 36]
        # Monotone non-increasing error, near-exact once the harmonics are resolved.
        assert np.all(np.diff(conv.frequency_error_hz) <= 1e-9)
        assert conv.frequency_error_hz[0] > conv.frequency_error_hz[2]
        assert conv.frequency_error_hz[conv.densities == 12][0] < 1e-9

    def test_reference_is_the_full_density_average(self) -> None:
        """The reference is the full-density mean's under-damped frequencies."""
        per, az = _periodic_samples(180)
        conv = azimuth_density_sweep(per, az)
        eigvals = np.linalg.eigvals(per.mean(axis=0))
        truth = np.sort(np.abs(eigvals[eigvals.imag > 1e-8]) / (2.0 * np.pi))
        np.testing.assert_allclose(conv.reference_frequencies_hz, truth, atol=1e-12)
        assert conv.reference_frequencies_hz.shape == (2,)  # two under-damped modes

    def test_skips_densities_above_available(self) -> None:
        """Requested counts beyond the available azimuths are skipped."""
        per, az = _periodic_samples(9)
        conv = azimuth_density_sweep(per, az, densities=(1, 3, 6, 12, 36))
        assert list(conv.densities) == [1, 3, 6]

    def test_azimuth_unit_is_irrelevant(self) -> None:
        """Degrees and radians give the same result (azimuths only order samples)."""
        per, az_rad = _periodic_samples(36)
        rad = azimuth_density_sweep(per, az_rad, densities=(3, 6, 12))
        deg = azimuth_density_sweep(per, np.degrees(az_rad), densities=(3, 6, 12))
        np.testing.assert_allclose(rad.frequency_error_hz, deg.frequency_error_hz)

    def test_rejects_no_achievable_density(self) -> None:
        """A request with no achievable density is rejected."""
        per, az = _periodic_samples(5)
        with pytest.raises(ValueError, match="no requested density"):
            azimuth_density_sweep(per, az, densities=(12, 36))

    def test_rejects_inconsistent_shape(self) -> None:
        """A non-square matrix stack is rejected."""
        with pytest.raises(ValueError, match="must be"):
            azimuth_density_sweep(np.zeros((3, 2, 4)), np.zeros(3))

    def test_rejects_fractional_density(self) -> None:
        """A fractional density is rejected, not silently truncated (int(2.9) == 2)."""
        per, az = _periodic_samples(36)
        with pytest.raises(ValueError, match="whole-number"):
            azimuth_density_sweep(per, az, densities=[2.9])  # type: ignore[list-item]

    def test_rejects_non_finite_density(self) -> None:
        """A non-finite density is rejected."""
        per, az = _periodic_samples(36)
        with pytest.raises(ValueError, match="whole-number"):
            azimuth_density_sweep(per, az, densities=[np.inf])  # type: ignore[list-item]

    def test_rejects_non_finite_matrix(self) -> None:
        """A non-finite matrix entry is rejected before it reaches eigvals."""
        per, az = _periodic_samples(36)
        bad = np.array(per, dtype=np.float64)
        bad[0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="per_azimuth_a must be finite"):
            azimuth_density_sweep(bad, az)

    def test_rejects_non_finite_azimuth(self) -> None:
        """A non-finite azimuth is rejected."""
        per, az = _periodic_samples(36)
        bad_az = np.array(az, dtype=np.float64)
        bad_az[0] = np.inf
        with pytest.raises(ValueError, match="azimuths must be finite"):
            azimuth_density_sweep(per, bad_az)


class TestRecommendAzimuthCount:
    """Tests for recommend_azimuth_count."""

    def test_returns_smallest_converged_count(self) -> None:
        """It returns the fewest azimuths that hold the error within tolerance."""
        per, az = _periodic_samples(180)
        conv = azimuth_density_sweep(per, az, densities=(1, 3, 6, 12, 36))
        assert recommend_azimuth_count(conv, tolerance_hz=1e-6) == 12
        assert recommend_azimuth_count(conv, tolerance_hz=10.0) == 1

    def test_raises_when_tolerance_unreachable(self) -> None:
        """A tolerance no evaluated density meets is rejected."""
        per, az = _periodic_samples(180)
        conv = azimuth_density_sweep(per, az, densities=(1, 3))
        with pytest.raises(ValueError, match="use more azimuths"):
            recommend_azimuth_count(conv, tolerance_hz=1e-9)

    def test_rejects_nonpositive_tolerance(self) -> None:
        """A non-positive tolerance is rejected."""
        per, az = _periodic_samples(36)
        conv = azimuth_density_sweep(per, az)
        with pytest.raises(ValueError, match="tolerance_hz must be positive"):
            recommend_azimuth_count(conv, tolerance_hz=0.0)


def test_convergence_dataclass_shapes() -> None:
    """The convergence record has aligned, well-shaped arrays."""
    per, az = _periodic_samples(36)
    conv = azimuth_density_sweep(per, az, densities=(3, 6, 12))
    assert isinstance(conv, AzimuthConvergence)
    assert conv.natural_frequencies_hz.shape == (3, 2)  # two under-damped modes
    assert conv.frequency_error_hz.shape == (3,)
