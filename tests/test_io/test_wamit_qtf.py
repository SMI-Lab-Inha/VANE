# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for WAMIT diagonal-QTF parsing and Newman's approximation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from vane.io import (
    DiagonalDifferenceQTF,
    newman_slow_drift_load,
    read_wamit_diagonal_difference_qtf,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_reads_and_dimensionalizes_complete_diagonal(tmp_path: Path) -> None:
    """Force and moment components use HydroDyn's respective length exponents."""
    path = tmp_path / "body.12d"
    rows = []
    for period in (10.0, 5.0):
        for component in range(1, 7):
            value = float(component) * period
            rows.append(f"{period} {period} 0 0 {component} {value} 0 {value} 0")
    path.write_text("\n".join(rows), encoding="utf-8")

    qtf = read_wamit_diagonal_difference_qtf(
        path,
        water_density_kg_m3=2.0,
        gravity_m_s2=3.0,
        characteristic_length_m=4.0,
    )

    np.testing.assert_allclose(
        qtf.angular_frequency_rad_s, [2 * np.pi / 10, 2 * np.pi / 5]
    )
    np.testing.assert_allclose(qtf.values_si[0, :3], 24.0 * np.arange(1, 4) * 10.0)
    np.testing.assert_allclose(qtf.values_si[0, 3:], 96.0 * np.arange(4, 7) * 10.0)
    with pytest.raises(ValueError, match="read-only"):
        qtf.values_si[0, 0] = 0.0


def test_qtf_constructor_rejects_nonfinite_heading() -> None:
    """A direct artifact cannot retain an undefined directional convention."""
    with pytest.raises(ValueError, match="wave heading must be finite"):
        DiagonalDifferenceQTF(
            angular_frequency_rad_s=np.array([0.1, 1.0]),
            values_si=np.ones((2, 6)),
            wave_heading_deg=np.nan,
        )


def test_newman_constant_positive_qtf_matches_two_component_solution() -> None:
    """The FFT implementation reproduces the analytic difference-frequency beat."""
    count = 512
    dt = 0.25
    time = np.arange(count) * dt
    omega_1 = 2.0 * np.pi * 8 / (count * dt)
    omega_2 = 2.0 * np.pi * 13 / (count * dt)
    amplitude_1 = 1.2
    amplitude_2 = 0.7
    elevation = amplitude_1 * np.cos(omega_1 * time) + amplitude_2 * np.cos(
        omega_2 * time
    )
    frequencies = np.array([0.01, 10.0])
    qtf_value = 5.0
    qtf = DiagonalDifferenceQTF(
        angular_frequency_rad_s=frequencies,
        values_si=np.full((2, 6), qtf_value),
        wave_heading_deg=0.0,
    )

    actual = newman_slow_drift_load(
        elevation,
        sample_interval_s=dt,
        qtf=qtf,
    )
    expected = qtf_value * (
        amplitude_1**2
        + amplitude_2**2
        + 2.0 * amplitude_1 * amplitude_2 * np.cos((omega_2 - omega_1) * time)
    )

    np.testing.assert_allclose(
        actual, np.repeat(expected[:, None], 6, axis=1), rtol=1.0e-11, atol=1.0e-11
    )


@pytest.mark.parametrize("qtf_value", [3.5, -3.5])
def test_newman_preserves_unique_nyquist_amplitude(qtf_value: float) -> None:
    """The even-record Nyquist bin is not doubled as a conjugate-pair member."""
    count = 32
    dt = 0.25
    amplitude = 1.7
    elevation = amplitude * (-1.0) ** np.arange(count)
    nyquist_omega = np.pi / dt
    qtf = DiagonalDifferenceQTF(
        angular_frequency_rad_s=np.array([0.01, 1.1 * nyquist_omega]),
        values_si=np.full((2, 6), qtf_value),
        wave_heading_deg=0.0,
    )

    actual = newman_slow_drift_load(elevation, sample_interval_s=dt, qtf=qtf)
    expected = np.full((count, 6), qtf_value * amplitude**2)

    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_newman_rejects_nonfinite_wave() -> None:
    """Runtime wave input fails closed on missing samples."""
    qtf = DiagonalDifferenceQTF(
        angular_frequency_rad_s=np.array([0.1, 1.0]),
        values_si=np.ones((2, 6)),
        wave_heading_deg=0.0,
    )
    with pytest.raises(ValueError, match="finite"):
        newman_slow_drift_load([0.0, 1.0, np.nan, 0.0], sample_interval_s=0.1, qtf=qtf)


def test_newman_rejects_nan_high_frequency_cutoff() -> None:
    """A NaN high cutoff cannot silently remove every incident-wave bin."""
    qtf = DiagonalDifferenceQTF(
        angular_frequency_rad_s=np.array([0.1, 1.0]),
        values_si=np.ones((2, 6)),
        wave_heading_deg=0.0,
    )
    with pytest.raises(ValueError, match="cutoffs"):
        newman_slow_drift_load(
            [0.0, 1.0, 0.0, -1.0],
            sample_interval_s=0.1,
            qtf=qtf,
            wave_high_cutoff_rad_s=np.nan,
        )


def test_newman_rejects_frequency_and_load_overflow() -> None:
    """Extreme finite inputs fail explicitly instead of returning NaN loads."""
    qtf = DiagonalDifferenceQTF(
        angular_frequency_rad_s=np.array([0.01, 100.0]),
        values_si=np.full((2, 6), np.finfo(np.float64).max),
        wave_heading_deg=0.0,
    )
    with pytest.raises(ValueError, match="too small for a finite frequency grid"):
        newman_slow_drift_load(
            [0.0, 1.0, 0.0, -1.0],
            sample_interval_s=np.nextafter(0.0, 1.0),
            qtf=qtf,
        )
    with pytest.raises(ValueError, match="finite numerical range"):
        newman_slow_drift_load(
            [0.0, 2.0, 0.0, -2.0],
            sample_interval_s=0.1,
            qtf=qtf,
        )


def test_reader_rejects_material_imaginary_diagonal(tmp_path: Path) -> None:
    """A complex diagonal beyond the explicit numerical tolerance fails closed."""
    path = tmp_path / "complex.12d"
    rows = []
    for period in (10.0, 5.0):
        for component in range(1, 7):
            imaginary = 0.1 if component == 1 else 0.0
            rows.append(f"{period} {period} 0 0 {component} 1 0 1 {imaginary}")
    path.write_text("\n".join(rows), encoding="utf-8")

    with pytest.raises(ValueError, match="not real"):
        read_wamit_diagonal_difference_qtf(
            path,
            water_density_kg_m3=1025.0,
            gravity_m_s2=9.81,
            characteristic_length_m=1.0,
        )


@pytest.mark.parametrize("tolerance", [np.nan, np.inf, -1.0])
def test_reader_rejects_nonfinite_or_negative_selection_tolerance(
    tmp_path: Path, tolerance: float
) -> None:
    """A malformed selection tolerance cannot admit off-diagonal QTF rows."""
    path = tmp_path / "off_diagonal.12d"
    path.write_text("10 5 30 60 1 1 0 1 0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="finite and nonnegative"):
        read_wamit_diagonal_difference_qtf(
            path,
            water_density_kg_m3=1025.0,
            gravity_m_s2=9.81,
            characteristic_length_m=1.0,
            tolerance=tolerance,
        )
