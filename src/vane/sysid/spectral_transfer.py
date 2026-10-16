# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Frequency-domain transfer and stochastic input-to-QoI propagation.

For a linear state-space model, the complex transfer matrix is

.. math::

   H(f) = C\left(sI-A\right)^{-1}B + D,

with :math:`s=j2\pi f` for a continuous model and
:math:`s=\exp(j2\pi f\Delta t)` for a discrete model.  If the input cross-spectral
density is :math:`S_u(f)`, the model-implied output spectrum is

.. math::

   S_y(f) = H(f)S_u(f)H(f)^{\mathsf H}.

This is the auditable bridge between an input-estimation error spectrum and the
motion/load quantities affected by that error.  It does not consult output truth and
does not claim nonlinear or out-of-support validity; those remain validation tasks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import csd

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from vane.sysid.state_space import StateSpace

__all__ = [
    "CrossSpectralDensity",
    "FrequencyResponse",
    "frequency_response",
    "integrated_spectral_variance",
    "propagate_cross_spectral_density",
    "welch_cross_spectral_density",
]


@dataclass(frozen=True)
class FrequencyResponse:
    """Named complex input-to-output response on a frequency grid.

    ``response`` has shape ``(n_frequencies, n_outputs, n_inputs)``.
    """

    frequencies_hz: npt.NDArray[np.float64]
    response: npt.NDArray[np.complex128]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate dimensions and freeze finite response data."""
        frequencies = _frequency_grid(self.frequencies_hz)
        response = np.array(self.response, dtype=np.complex128, copy=True)
        input_names = tuple(self.input_names)
        output_names = tuple(self.output_names)
        _validate_names(input_names, "input_names")
        _validate_names(output_names, "output_names")
        expected = (frequencies.size, len(output_names), len(input_names))
        if response.shape != expected:
            msg = f"frequency response has shape {response.shape}, expected {expected}"
            raise ValueError(msg)
        if not np.all(np.isfinite(response)):
            msg = "frequency response must contain only finite values"
            raise ValueError(msg)
        frequencies = np.array(frequencies, copy=True)
        frequencies.setflags(write=False)
        response.setflags(write=False)
        object.__setattr__(self, "frequencies_hz", frequencies)
        object.__setattr__(self, "response", response)
        object.__setattr__(self, "input_names", input_names)
        object.__setattr__(self, "output_names", output_names)


@dataclass(frozen=True)
class CrossSpectralDensity:
    """Named one-sided cross-spectral-density matrix.

    ``density`` has shape ``(n_frequencies, n_channels, n_channels)`` and follows
    the covariance convention ``S[i, j] = E[X_i conj(X_j)]``.
    """

    frequencies_hz: npt.NDArray[np.float64]
    density: npt.NDArray[np.complex128]
    channel_names: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate dimensions, Hermitian symmetry, and finite spectral values."""
        frequencies = _frequency_grid(self.frequencies_hz)
        density = np.array(self.density, dtype=np.complex128, copy=True)
        channel_names = tuple(self.channel_names)
        _validate_names(channel_names, "channel_names")
        expected = (frequencies.size, len(channel_names), len(channel_names))
        if density.shape != expected:
            msg = f"spectral density has shape {density.shape}, expected {expected}"
            raise ValueError(msg)
        if not np.all(np.isfinite(density)):
            msg = "spectral density must contain only finite values"
            raise ValueError(msg)
        hermitian = density.conj().transpose(0, 2, 1)
        if not np.allclose(density, hermitian, rtol=1.0e-10, atol=1.0e-12):
            msg = "spectral density must be Hermitian at every frequency"
            raise ValueError(msg)
        density = 0.5 * (density + hermitian)
        eigenvalues = np.linalg.eigvalsh(density)
        scale = np.maximum(np.max(np.abs(eigenvalues), axis=1), 1.0)
        if np.any(np.min(eigenvalues, axis=1) < -1.0e-10 * scale):
            msg = "spectral density must be positive semidefinite at every frequency"
            raise ValueError(msg)
        frequencies = np.array(frequencies, copy=True)
        frequencies.setflags(write=False)
        density.setflags(write=False)
        object.__setattr__(self, "frequencies_hz", frequencies)
        object.__setattr__(self, "density", density)
        object.__setattr__(self, "channel_names", channel_names)


def _frequency_grid(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    frequencies = np.asarray(values, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size == 0:
        msg = "frequencies_hz must be a non-empty one-dimensional array"
        raise ValueError(msg)
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies < 0.0):
        msg = "frequencies_hz must be finite and nonnegative"
        raise ValueError(msg)
    if np.any(np.diff(frequencies) <= 0.0):
        msg = "frequencies_hz must be strictly increasing"
        raise ValueError(msg)
    return frequencies


def _validate_names(names: tuple[str, ...], label: str) -> None:
    """Require a nonempty unique tuple of nonempty channel names."""
    if (
        not names
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != len(names)
    ):
        msg = f"{label} must contain unique nonempty strings"
        raise ValueError(msg)


def _sample_count(value: int, name: str) -> int:
    """Return an exact integer sample count without real-number truncation."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        msg = f"{name} must be an integer"
        raise TypeError(msg)
    return int(value)


def _resolve_channels(
    requested: Sequence[int | str] | None,
    names: list[str],
    count: int,
    label: str,
) -> tuple[list[int], tuple[str, ...]]:
    if requested is None:
        indices = list(range(count))
        resolved_names = (
            tuple(names) if names else tuple(f"{label}_{i}" for i in indices)
        )
        return indices, resolved_names
    indices = []
    resolved = []
    for channel in requested:
        if isinstance(channel, (int, np.integer)):
            index = int(channel)
            if not 0 <= index < count:
                msg = f"{label} index {index} out of range [0, {count})"
                raise ValueError(msg)
        elif isinstance(channel, str):
            if not names:
                msg = f"cannot resolve named {label} {channel!r}; model has no names"
                raise ValueError(msg)
            exact = [i for i, name in enumerate(names) if name == channel]
            matches = exact or [i for i, name in enumerate(names) if channel in name]
            if len(matches) != 1:
                msg = f"{label} {channel!r} resolved to {matches}"
                raise ValueError(msg)
            index = matches[0]
        else:
            msg = f"{label} selectors must be integer indices or strings"
            raise ValueError(msg)  # noqa: TRY004
        indices.append(index)
        resolved.append(names[index] if names else f"{label}_{index}")
    if len(set(indices)) != len(indices):
        msg = f"{label} selectors must resolve to unique channels"
        raise ValueError(msg)
    if not indices:
        msg = f"at least one {label} channel is required"
        raise ValueError(msg)
    return indices, tuple(resolved)


def frequency_response(
    system: StateSpace,
    frequencies_hz: npt.ArrayLike,
    *,
    input_channels: Sequence[int | str] | None = None,
    output_channels: Sequence[int | str] | None = None,
) -> FrequencyResponse:
    """Evaluate selected complex state-space transfers.

    Parameters
    ----------
    system
        Continuous or discrete state-space model with ``B`` and ``C`` matrices.
    frequencies_hz
        Strictly increasing nonnegative physical frequencies in Hz.  A discrete
        model is restricted to its Nyquist interval.
    input_channels, output_channels
        Optional unique indices or unambiguous names.  By default all channels are
        retained.
    """
    if system.b is None or system.c is None:
        msg = "frequency response requires both B and C matrices"
        raise ValueError(msg)
    frequencies = _frequency_grid(frequencies_hz)
    normalized_discrete_frequencies: npt.NDArray[np.float64] | None = None
    if system.dt is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            normalized_discrete_frequencies = frequencies * np.float64(system.dt)
        if (
            not np.all(np.isfinite(normalized_discrete_frequencies))
            or normalized_discrete_frequencies[-1] > 0.5
        ):
            msg = "frequency grid exceeds the discrete model's Nyquist frequency"
            raise ValueError(msg)
    inputs, input_names = _resolve_channels(
        input_channels, system.input_names, system.n_inputs, "input"
    )
    outputs, output_names = _resolve_channels(
        output_channels, system.output_names, system.n_outputs, "output"
    )
    b = system.b[:, inputs]
    c = system.c[outputs]
    if system.d is None:
        d: npt.NDArray[np.float64] = np.zeros(
            (len(outputs), len(inputs)), dtype=np.float64
        )
    else:
        d = system.d[np.ix_(outputs, inputs)]
    identity = np.eye(system.n_states)
    if system.dt is None:
        with np.errstate(over="ignore", invalid="ignore"):
            angular_frequencies = 2.0 * math.pi * frequencies
        if not np.all(np.isfinite(angular_frequencies)):
            msg = "frequency grid is too large for finite angular frequencies"
            raise ValueError(msg)
        spectral_points = 1j * angular_frequencies
    else:
        assert normalized_discrete_frequencies is not None
        spectral_points = np.exp(2j * math.pi * normalized_discrete_frequencies)
    response = np.empty(
        (frequencies.size, len(outputs), len(inputs)), dtype=np.complex128
    )
    for index, spectral_point in enumerate(spectral_points):
        response[index] = (
            c @ np.linalg.solve(spectral_point * identity - system.a, b) + d
        )
    return FrequencyResponse(frequencies, response, input_names, output_names)


def welch_cross_spectral_density(
    signals: npt.ArrayLike,
    *,
    sample_interval_s: float,
    segment_samples: int | None = None,
    overlap_samples: int | None = None,
    channel_names: Sequence[str] | None = None,
) -> CrossSpectralDensity:
    """Estimate a one-sided cross-spectral-density matrix using Welch averaging.

    Samples are rows and channels are columns.  A one-dimensional input is treated
    as one channel.  Cross terms are retained, so correlated torque/thrust errors are
    propagated without the independence assumption.
    """
    values = np.asarray(signals, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        msg = "signals must have shape (n_samples, n_channels) with at least 2 samples"
        raise ValueError(msg)
    if not np.all(np.isfinite(values)):
        msg = "signals must be finite"
        raise ValueError(msg)
    if not (math.isfinite(sample_interval_s) and sample_interval_s > 0.0):
        msg = "sample_interval_s must be positive and finite"
        raise ValueError(msg)
    n_samples, n_channels = values.shape
    nperseg = (
        min(256, n_samples)
        if segment_samples is None
        else _sample_count(segment_samples, "segment_samples")
    )
    if not 2 <= nperseg <= n_samples:
        msg = f"segment_samples must be in [2, {n_samples}]"
        raise ValueError(msg)
    noverlap = (
        nperseg // 2
        if overlap_samples is None
        else _sample_count(overlap_samples, "overlap_samples")
    )
    if not 0 <= noverlap < nperseg:
        msg = "overlap_samples must be nonnegative and smaller than segment_samples"
        raise ValueError(msg)
    if channel_names is None:
        names = tuple(f"channel_{i}" for i in range(n_channels))
    else:
        names = tuple(channel_names)
        if len(names) != n_channels or len(set(names)) != len(names):
            msg = "channel_names must be unique and match the signal columns"
            raise ValueError(msg)
    density: npt.NDArray[np.complex128] | None = None
    frequencies: npt.NDArray[np.float64] | None = None
    for row in range(n_channels):
        for column in range(n_channels):
            current_frequencies, current_density = csd(
                values[:, column],
                values[:, row],
                fs=1.0 / sample_interval_s,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                detrend="constant",
                return_onesided=True,
                scaling="density",
            )
            if frequencies is None:
                frequencies = np.asarray(current_frequencies, dtype=np.float64)
                density = np.empty(
                    (frequencies.size, n_channels, n_channels),
                    dtype=np.complex128,
                )
            assert density is not None and frequencies is not None
            density[:, row, column] = current_density
    assert density is not None and frequencies is not None
    density = 0.5 * (density + density.conj().transpose(0, 2, 1))
    return CrossSpectralDensity(frequencies, density, names)


def propagate_cross_spectral_density(
    response: FrequencyResponse,
    spectrum: CrossSpectralDensity,
) -> CrossSpectralDensity:
    """Propagate ``S_u`` through ``H`` as ``S_y = H S_u H^H``."""
    if response.input_names != spectrum.channel_names:
        msg = (
            "response inputs and spectrum channels differ: "
            f"{response.input_names!r} != {spectrum.channel_names!r}"
        )
        raise ValueError(msg)
    if not np.array_equal(response.frequencies_hz, spectrum.frequencies_hz):
        msg = "response and spectrum frequency grids must match exactly"
        raise ValueError(msg)
    expected = (
        spectrum.frequencies_hz.size,
        len(spectrum.channel_names),
        len(spectrum.channel_names),
    )
    if spectrum.density.shape != expected:
        msg = (
            f"spectrum density has shape {spectrum.density.shape}, expected {expected}"
        )
        raise ValueError(msg)
    output = np.einsum(
        "foi,fij,fpj->fop",
        response.response,
        spectrum.density,
        response.response.conj(),
        optimize=True,
    )
    output = 0.5 * (output + output.conj().transpose(0, 2, 1))
    return CrossSpectralDensity(
        response.frequencies_hz.copy(),
        np.asarray(output, dtype=np.complex128),
        response.output_names,
    )


def integrated_spectral_variance(
    spectrum: CrossSpectralDensity,
    *,
    band_hz: tuple[float, float] | None = None,
) -> npt.NDArray[np.float64]:
    """Integrate each autospectrum over a declared frequency band.

    Returns one nonnegative variance per channel.  The trapezoidal integral runs
    exactly from ``band_hz[0]`` to ``band_hz[1]``: the autospectra are linearly
    interpolated at band limits that fall between spectrum bins, so a band edge is
    not truncated to the nearest interior bin.  The band is clipped to the spectrum's
    frequency range, because the density is undefined outside it, and the clipped
    band must retain a positive width; this avoids silently reporting a zero-width
    integral.  Without ``band_hz`` the whole spectrum grid is integrated.
    """
    frequencies = _frequency_grid(spectrum.frequencies_hz)
    if frequencies.size < 2:
        msg = "spectral integration requires at least two frequency bins"
        raise ValueError(msg)
    diagonal = np.real(np.diagonal(spectrum.density, axis1=1, axis2=2))
    if band_hz is None:
        return np.asarray(
            np.maximum(trapezoid(diagonal, frequencies, axis=0), 0.0),
            dtype=np.float64,
        )
    low, high = (float(value) for value in band_hz)
    if not (math.isfinite(low) and math.isfinite(high) and 0.0 <= low < high):
        msg = "band_hz must contain finite bounds satisfying 0 <= low < high"
        raise ValueError(msg)
    low = max(low, float(frequencies[0]))
    high = min(high, float(frequencies[-1]))
    if not low < high:
        msg = "spectral integration band does not overlap the spectrum frequencies"
        raise ValueError(msg)
    interior = (frequencies > low) & (frequencies < high)
    grid = np.concatenate(([low], frequencies[interior], [high]))
    edges = np.stack(
        [
            np.interp([low, high], frequencies, diagonal[:, channel])
            for channel in range(diagonal.shape[1])
        ],
        axis=1,
    )
    values = np.concatenate((edges[:1], diagonal[interior], edges[1:]), axis=0)
    variance = trapezoid(values, grid, axis=0)
    return np.asarray(np.maximum(variance, 0.0), dtype=np.float64)
