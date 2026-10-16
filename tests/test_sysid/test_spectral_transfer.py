# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for dynamic input-error propagation to state-space QoIs."""

from __future__ import annotations

import numpy as np
import pytest

from vane.sysid import (
    CrossSpectralDensity,
    FrequencyResponse,
    StateSpace,
    frequency_response,
    integrated_spectral_variance,
    propagate_cross_spectral_density,
    welch_cross_spectral_density,
)


def test_continuous_frequency_response_matches_first_order_solution() -> None:
    """The implementation includes both dynamic and direct-feedthrough terms."""
    system = StateSpace(
        a=np.array([[-2.0]]),
        b=np.array([[3.0]]),
        c=np.array([[4.0]]),
        d=np.array([[0.5]]),
        input_names=["torque"],
        output_names=["tension"],
    )
    frequencies = np.array([0.1, 0.5, 1.0])
    actual = frequency_response(system, frequencies)
    expected = 12.0 / (2j * np.pi * frequencies + 2.0) + 0.5
    np.testing.assert_allclose(actual.response[:, 0, 0], expected)
    assert actual.input_names == ("torque",)
    assert actual.output_names == ("tension",)
    with pytest.raises(ValueError, match="read-only"):
        actual.response[0, 0, 0] = 0.0


def test_frequency_response_rejects_angular_frequency_overflow() -> None:
    """A finite hertz value cannot overflow silently during radian conversion."""
    system = StateSpace(
        a=np.array([[-2.0]]),
        b=np.array([[1.0]]),
        c=np.array([[1.0]]),
        d=np.array([[0.0]]),
    )
    with pytest.raises(ValueError, match="finite angular frequencies"):
        frequency_response(system, np.array([np.finfo(np.float64).max]))


def test_discrete_response_uses_unit_circle_and_checks_nyquist() -> None:
    """A discrete transfer is evaluated at z=exp(j 2 pi f dt)."""
    system = StateSpace(
        a=np.array([[0.8]]),
        b=np.array([[1.0]]),
        c=np.array([[2.0]]),
        d=np.array([[0.0]]),
        dt=0.2,
    )
    frequencies = np.array([0.1, 1.0])
    actual = frequency_response(system, frequencies)
    z = np.exp(2j * np.pi * frequencies * 0.2)
    np.testing.assert_allclose(actual.response[:, 0, 0], 2.0 / (z - 0.8))
    with pytest.raises(ValueError, match="Nyquist"):
        frequency_response(system, np.array([0.1, 2.6]))


def test_discrete_response_scales_frequency_before_forming_phase() -> None:
    """A tiny finite sample time does not overflow an otherwise valid phase."""
    system = StateSpace(
        a=np.array([[0.8]]),
        b=np.array([[1.0]]),
        c=np.array([[2.0]]),
        d=np.array([[0.0]]),
        dt=float(np.nextafter(0.0, 1.0)),
    )
    actual = frequency_response(system, np.array([np.finfo(np.float64).max]))
    assert np.all(np.isfinite(actual.response))


def test_named_channel_selection_is_ordered_and_unambiguous() -> None:
    """Selected rows/columns retain the caller's requested order and names."""
    system = StateSpace(
        a=np.diag([-1.0, -2.0]),
        b=np.eye(2),
        c=np.eye(2),
        d=np.zeros((2, 2)),
        input_names=["aerodynamic torque", "hub thrust"],
        output_names=["surge", "fairlead tension"],
    )
    actual = frequency_response(
        system,
        np.array([0.1, 0.2]),
        input_channels=["thrust", "torque"],
        output_channels=["tension", "surge"],
    )
    assert actual.input_names == ("hub thrust", "aerodynamic torque")
    assert actual.output_names == ("fairlead tension", "surge")
    with pytest.raises(ValueError, match="unique"):
        frequency_response(system, np.array([0.1]), input_channels=["hub thrust", 1])


def test_csd_propagation_retains_correlated_cross_term() -> None:
    """H S H^H includes constructive correlation between two input errors."""
    frequencies = np.array([0.0, 1.0, 2.0])
    density = np.repeat(
        np.array([[[1.0, 0.5], [0.5, 4.0]]], dtype=np.complex128),
        frequencies.size,
        axis=0,
    )
    spectrum = CrossSpectralDensity(frequencies, density, ("qa", "thrust"))
    response = FrequencyResponse(
        frequencies,
        np.repeat(
            np.array([[[2.0, 3.0]]], dtype=np.complex128),
            frequencies.size,
            axis=0,
        ),
        ("qa", "thrust"),
        ("tension",),
    )
    propagated = propagate_cross_spectral_density(response, spectrum)
    # 2^2*1 + 3^2*4 + 2*(2*3*0.5) = 46.
    np.testing.assert_allclose(propagated.density[:, 0, 0], 46.0)
    np.testing.assert_allclose(
        integrated_spectral_variance(propagated, band_hz=(0.0, 2.0)), [92.0]
    )


def test_welch_csd_integrates_to_signal_variance() -> None:
    """The one-sided autospectrum has the expected density normalization."""
    rng = np.random.default_rng(42)
    signal = rng.normal(size=(16384, 2))
    signal[:, 1] = 0.4 * signal[:, 0] + 0.8 * signal[:, 1]
    spectrum = welch_cross_spectral_density(
        signal,
        sample_interval_s=0.1,
        segment_samples=1024,
        channel_names=("qa", "thrust"),
    )
    variance = integrated_spectral_variance(spectrum)
    np.testing.assert_allclose(variance, np.var(signal, axis=0), rtol=0.06)
    np.testing.assert_allclose(
        spectrum.density,
        spectrum.density.conj().transpose(0, 2, 1),
        atol=1.0e-14,
    )
    with pytest.raises(ValueError, match="read-only"):
        spectrum.density[0, 0, 0] = 0.0


def test_spectral_contract_rejects_fractional_counts_and_indefinite_density() -> None:
    """Welch sample counts stay discrete and CSD inputs remain physically valid."""
    signal = np.arange(16.0)
    with pytest.raises(TypeError, match="segment_samples must be an integer"):
        welch_cross_spectral_density(
            signal,
            sample_interval_s=0.1,
            segment_samples=8.5,  # type: ignore[arg-type]
        )
    indefinite = np.repeat(
        np.array([[[1.0, 2.0], [2.0, 1.0]]], dtype=np.complex128),
        2,
        axis=0,
    )
    with pytest.raises(ValueError, match="positive semidefinite"):
        CrossSpectralDensity(
            np.array([0.0, 1.0]),
            indefinite,
            ("torque", "thrust"),
        )


def test_csd_normalizes_admissible_roundoff_to_exact_hermitian_form() -> None:
    """Accepted floating-point asymmetry does not leak into later propagation."""
    density = np.array([[[1.0, 0.25 + 1.0e-14j], [0.25, 1.0]]])
    spectrum = CrossSpectralDensity(np.array([0.0]), density, ("a", "b"))
    np.testing.assert_array_equal(
        spectrum.density,
        spectrum.density.conj().transpose(0, 2, 1),
    )


def test_propagation_rejects_channel_or_grid_mismatch() -> None:
    """Silent input reordering is prohibited."""
    response = FrequencyResponse(
        np.array([0.0, 1.0]),
        np.ones((2, 1, 1), dtype=np.complex128),
        ("qa",),
        ("tension",),
    )
    wrong_name = CrossSpectralDensity(
        np.array([0.0, 1.0]),
        np.ones((2, 1, 1), dtype=np.complex128),
        ("thrust",),
    )
    with pytest.raises(ValueError, match="differ"):
        propagate_cross_spectral_density(response, wrong_name)
    wrong_grid = CrossSpectralDensity(
        np.array([0.0, 2.0]),
        np.ones((2, 1, 1), dtype=np.complex128),
        ("qa",),
    )
    with pytest.raises(ValueError, match="frequency grids"):
        propagate_cross_spectral_density(response, wrong_grid)


def _linear_autospectrum(frequencies: np.ndarray) -> CrossSpectralDensity:
    """Return a two-channel autospectrum S(f) = f and S(f) = 3."""
    density = np.zeros((frequencies.size, 2, 2), dtype=np.complex128)
    density[:, 0, 0] = frequencies
    density[:, 1, 1] = 3.0
    return CrossSpectralDensity(frequencies, density, ("ramp", "flat"))


def test_band_variance_includes_interpolated_band_edges() -> None:
    """Band limits between bins are integrated exactly, not truncated."""
    frequencies = np.arange(129) * 0.0390625
    spectrum = _linear_autospectrum(frequencies)

    variance = integrated_spectral_variance(spectrum, band_hz=(0.0, 0.1))
    # int_0^0.1 f df = 0.005 and int_0^0.1 3 df = 0.3.
    np.testing.assert_allclose(variance, [0.005, 0.3], rtol=1.0e-12)

    shifted = integrated_spectral_variance(spectrum, band_hz=(0.05, 0.2))
    np.testing.assert_allclose(
        shifted, [0.5 * (0.2**2 - 0.05**2), 3.0 * 0.15], rtol=1.0e-12
    )


def test_band_variance_between_two_bins_and_clipped_to_grid() -> None:
    """A narrow in-between band is valid; a band past the grid is clipped."""
    frequencies = np.linspace(0.0, 1.0, 11)
    spectrum = _linear_autospectrum(frequencies)

    narrow = integrated_spectral_variance(spectrum, band_hz=(0.42, 0.47))
    np.testing.assert_allclose(narrow, [0.5 * (0.47**2 - 0.42**2), 0.15])
    clipped = integrated_spectral_variance(spectrum, band_hz=(0.5, 5.0))
    np.testing.assert_allclose(clipped, [0.375, 1.5])
    with pytest.raises(ValueError, match="does not overlap"):
        integrated_spectral_variance(spectrum, band_hz=(2.0, 3.0))


def test_white_noise_band_variance_matches_analytic_value() -> None:
    """A white-noise band variance equals its flat one-sided PSD times the width."""
    rng = np.random.default_rng(7)
    sample_interval_s = 0.1
    signal = rng.normal(scale=2.0, size=2**17)
    spectrum = welch_cross_spectral_density(
        signal,
        sample_interval_s=sample_interval_s,
        segment_samples=128,
    )

    # Bins are 0.078125 Hz apart, so both band limits fall between bins away from
    # the detrended DC bin.
    variance = integrated_spectral_variance(spectrum, band_hz=(0.1, 0.35))
    # One-sided white-noise PSD 2 sigma^2 dt over a 0.25 Hz band.
    np.testing.assert_allclose(
        variance, [2.0 * 4.0 * sample_interval_s * 0.25], rtol=0.05
    )
