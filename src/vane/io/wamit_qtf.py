# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""WAMIT diagonal-QTF parsing and Newman's slow-drift reconstruction.

The implementation follows OpenFAST HydroDyn's ``NewmanApp_InitCalc`` formulation:
the positive and negative parts of each real diagonal difference-frequency QTF are
square-root filtered separately, transformed into analytic time histories, squared,
and subtracted.  It is intentionally limited to a single wave heading so unsupported
directional assumptions fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = [
    "DiagonalDifferenceQTF",
    "newman_slow_drift_load",
    "read_wamit_diagonal_difference_qtf",
]


@dataclass(frozen=True)
class DiagonalDifferenceQTF:
    """Dimensional diagonal difference-frequency QTF for one wave heading.

    Parameters
    ----------
    angular_frequency_rad_s : npt.NDArray[np.float64]
        Strictly increasing angular-frequency grid, shape ``(n_frequency,)``.
    values_si : npt.NDArray[np.float64]
        Real diagonal QTF values, shape ``(n_frequency, 6)``.  Translational
        columns have units N/m² and rotational columns N-m/m².
    wave_heading_deg : float
        Common incident-wave heading of both QTF wave components.
    """

    angular_frequency_rad_s: npt.NDArray[np.float64]
    values_si: npt.NDArray[np.float64]
    wave_heading_deg: float

    def __post_init__(self) -> None:
        """Validate and normalize array storage."""
        frequencies = np.array(
            self.angular_frequency_rad_s, dtype=np.float64, copy=True
        )
        values = np.array(self.values_si, dtype=np.float64, copy=True)
        heading = float(self.wave_heading_deg)
        if frequencies.ndim != 1 or frequencies.size < 2:
            message = "QTF frequency grid must contain at least two points"
            raise ValueError(message)
        if values.shape != (frequencies.size, 6):
            message = "QTF values must have shape (n_frequency, 6)"
            raise ValueError(message)
        if not np.all(np.isfinite(frequencies)) or not np.all(np.isfinite(values)):
            message = "QTF data must be finite"
            raise ValueError(message)
        if np.any(frequencies <= 0.0) or np.any(np.diff(frequencies) <= 0.0):
            message = "QTF frequencies must be positive and strictly increasing"
            raise ValueError(message)
        if not np.isfinite(heading):
            message = "QTF wave heading must be finite"
            raise ValueError(message)
        frequencies.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, "angular_frequency_rad_s", frequencies)
        object.__setattr__(self, "values_si", values)
        object.__setattr__(self, "wave_heading_deg", heading)


def read_wamit_diagonal_difference_qtf(
    path: str | Path,
    *,
    water_density_kg_m3: float,
    gravity_m_s2: float,
    characteristic_length_m: float,
    wave_heading_deg: float = 0.0,
    tolerance: float = 1.0e-8,
    imaginary_tolerance: float = 1.0e-5,
) -> DiagonalDifferenceQTF:
    """Read and dimensionalize one heading's diagonal from a WAMIT ``.12d`` file.

    WAMIT stores period pairs, direction pairs, a one-based load component, and the
    complex QTF.  Newman's approximation uses only rows where both periods and both
    headings are equal, and discards the negligible imaginary diagonal component.
    HydroDyn dimensionalizes forces by ``rho*g*L`` and moments by ``rho*g*L**2``;
    this function applies the same convention.

    ``tolerance`` controls equality checks on periods/headings.  The separate
    ``imaginary_tolerance`` admits only the small diagonal roundoff that HydroDyn
    explicitly discards before applying Newman's approximation.

    Raises
    ------
    ValueError
        If physical scales are invalid, the requested diagonal is incomplete, or its
        nominally real values have a material imaginary component.
    """
    density = float(water_density_kg_m3)
    gravity = float(gravity_m_s2)
    length = float(characteristic_length_m)
    scales = (density, gravity, length)
    if not all(np.isfinite(value) and value > 0.0 for value in scales):
        message = "density, gravity, and characteristic length must be positive"
        raise ValueError(message)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        message = "diagonal-selection tolerance must be finite and nonnegative"
        raise ValueError(message)
    if not np.isfinite(wave_heading_deg):
        message = "wave heading must be finite"
        raise ValueError(message)
    if not np.isfinite(imaginary_tolerance) or imaginary_tolerance < 0.0:
        message = "imaginary tolerance must be finite and nonnegative"
        raise ValueError(message)
    raw = np.loadtxt(Path(path), ndmin=2)
    if raw.shape[1] < 9:
        message = "WAMIT .12d data must contain at least nine columns"
        raise ValueError(message)
    if not np.all(np.isfinite(raw)):
        message = "WAMIT .12d data must contain only finite values"
        raise ValueError(message)
    diagonal = (
        np.isclose(raw[:, 0], raw[:, 1], rtol=0.0, atol=tolerance)
        & np.isclose(raw[:, 2], wave_heading_deg, rtol=0.0, atol=tolerance)
        & np.isclose(raw[:, 3], wave_heading_deg, rtol=0.0, atol=tolerance)
        & (raw[:, 0] > 0.0)
    )
    component_data: list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []
    reference_frequency: npt.NDArray[np.float64] | None = None
    for component in range(1, 7):
        rows = raw[diagonal & (raw[:, 4] == component)]
        if rows.shape[0] < 2:
            message = f"QTF diagonal is incomplete for component {component}"
            raise ValueError(message)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            frequency = 2.0 * np.pi / rows[:, 0]
        if not np.all(np.isfinite(frequency)):
            message = (
                f"QTF periods for component {component} produce invalid frequencies"
            )
            raise ValueError(message)
        order = np.argsort(frequency)
        frequency = np.asarray(frequency[order], dtype=np.float64)
        real = np.asarray(rows[order, 7], dtype=np.float64)
        imaginary = np.asarray(rows[order, 8], dtype=np.float64)
        imaginary_limit = imaginary_tolerance * max(1.0, float(np.max(np.abs(real))))
        if float(np.max(np.abs(imaginary))) > imaginary_limit:
            message = f"QTF diagonal for component {component} is not real"
            raise ValueError(message)
        if reference_frequency is None:
            reference_frequency = frequency
        elif not np.allclose(frequency, reference_frequency, rtol=0.0, atol=tolerance):
            message = "QTF components do not share one frequency grid"
            raise ValueError(message)
        exponent = 1 if component <= 3 else 2
        try:
            dimensional_scale = density * gravity * length**exponent
        except OverflowError as exc:
            message = "QTF dimensional scales exceed the finite numerical range"
            raise ValueError(message) from exc
        if not np.isfinite(dimensional_scale):
            message = "QTF dimensional scales exceed the finite numerical range"
            raise ValueError(message)
        with np.errstate(over="ignore", invalid="ignore"):
            dimensional_values = real * dimensional_scale
        if not np.all(np.isfinite(dimensional_values)):
            message = "dimensional QTF values exceed the finite numerical range"
            raise ValueError(message)
        component_data.append((frequency, dimensional_values))
    assert reference_frequency is not None
    return DiagonalDifferenceQTF(
        angular_frequency_rad_s=reference_frequency,
        values_si=np.column_stack([values for _, values in component_data]),
        wave_heading_deg=float(wave_heading_deg),
    )


def newman_slow_drift_load(
    wave_elevation_m: npt.ArrayLike,
    *,
    sample_interval_s: float,
    qtf: DiagonalDifferenceQTF,
    wave_low_cutoff_rad_s: float = 0.0,
    wave_high_cutoff_rad_s: float = np.inf,
) -> npt.NDArray[np.float64]:
    """Evaluate Newman's mean- and slow-drift six-axis load on a periodic record.

    Parameters
    ----------
    wave_elevation_m : array-like
        Equally sampled, periodic incident-wave elevation.  The record should span
        the sea-state realization's period; using a cropped non-periodic record causes
        spectral-edge error and must be disclosed by the caller.
    sample_interval_s : float
        Wave-elevation sample interval in seconds.
    qtf : DiagonalDifferenceQTF
        Dimensional real diagonal QTF.
    wave_low_cutoff_rad_s, wave_high_cutoff_rad_s : float, optional
        Incident-wave frequency cutoffs.  These match SeaState's ``WvLowCOff`` and
        ``WvHiCOff`` rather than the resulting difference-frequency cutoffs.

    Returns
    -------
    npt.NDArray[np.float64]
        Six-axis force/moment history, shape ``(n_time, 6)``.
    """
    elevation = np.asarray(wave_elevation_m, dtype=np.float64)
    if elevation.ndim != 1 or elevation.size < 4:
        message = "wave elevation must be a one-dimensional record"
        raise ValueError(message)
    if not np.all(np.isfinite(elevation)):
        message = "wave elevation must be finite"
        raise ValueError(message)
    sample_interval = float(sample_interval_s)
    if not np.isfinite(sample_interval) or sample_interval <= 0.0:
        message = "sample interval must be positive"
        raise ValueError(message)
    if sample_interval < np.pi / np.finfo(np.float64).max:
        message = "sample interval is too small for a finite frequency grid"
        raise ValueError(message)
    if (
        not np.isfinite(wave_low_cutoff_rad_s)
        or wave_low_cutoff_rad_s < 0.0
        or not (
            np.isfinite(wave_high_cutoff_rad_s) or np.isposinf(wave_high_cutoff_rad_s)
        )
        or wave_high_cutoff_rad_s <= wave_low_cutoff_rad_s
    ):
        message = "wave-frequency cutoffs are invalid"
        raise ValueError(message)

    count = elevation.size
    spectrum = np.fft.rfft(elevation)
    analytic_scale = np.full(spectrum.shape, 2.0, dtype=np.float64)
    analytic_scale[0] = 1.0
    if count % 2 == 0:
        analytic_scale[-1] = 1.0
    analytic_spectrum = analytic_scale * spectrum
    omega = 2.0 * np.pi * np.fft.rfftfreq(count, sample_interval)
    within_cutoffs = (omega >= wave_low_cutoff_rad_s) & (
        omega <= wave_high_cutoff_rad_s
    )
    load = np.empty((count, 6), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        for component in range(6):
            diagonal = np.interp(
                omega,
                qtf.angular_frequency_rad_s,
                qtf.values_si[:, component],
                left=0.0,
                right=0.0,
            )
            diagonal[~within_cutoffs] = 0.0
            positive = diagonal > 0.0
            negative = diagonal < 0.0
            first_spectrum = np.zeros(count, dtype=np.complex128)
            second_spectrum = np.zeros(count, dtype=np.complex128)
            first_spectrum[: spectrum.size][positive] = analytic_spectrum[
                positive
            ] * np.sqrt(diagonal[positive])
            second_spectrum[: spectrum.size][negative] = analytic_spectrum[
                negative
            ] * np.sqrt(-diagonal[negative])
            first = np.fft.ifft(first_spectrum)
            second = np.fft.ifft(second_spectrum)
            load[:, component] = np.abs(first) ** 2 - np.abs(second) ** 2
    if not np.all(np.isfinite(load)):
        message = "Newman slow-drift reconstruction exceeded the finite numerical range"
        raise ValueError(message)
    return load
