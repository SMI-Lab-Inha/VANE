# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Unknown-input (disturbance) augmentation for load virtual sensing.

A reduced model reconstructs *states* from sensors, but the quantity a digital twin
ultimately wants is often an **unknown input** — the wave/wind load or mooring force
that drives the structure and is never measured. The standard way to estimate it is to
append a model of the disturbance to the state space and let the filter estimate the
augmented state; the disturbance estimate is then read straight off it.

Given a physical continuous model

.. math::

   \dot{\mathbf x} = \mathbf A\mathbf x + \mathbf B\mathbf u + \mathbf E\mathbf d,
   \qquad \mathbf y = \mathbf C\mathbf x + \mathbf D\mathbf u + \mathbf F\mathbf d,

and a disturbance shaping model
:math:`\dot{\mathbf z} = \mathbf A_d\mathbf z + \mathbf w`,
:math:`\mathbf d = \mathbf C_d\mathbf z` (white :math:`\mathbf w`), the augmented system
on :math:`[\mathbf x; \mathbf z]` is

.. math::

   \mathbf A_{\text{aug}} = \begin{bmatrix}\mathbf A & \mathbf E\mathbf C_d\\
   \mathbf 0 & \mathbf A_d\end{bmatrix},\quad
   \mathbf B_{\text{aug}} = \begin{bmatrix}\mathbf B\\ \mathbf 0\end{bmatrix},\quad
   \mathbf C_{\text{aug}} =
   \begin{bmatrix}\mathbf C & \mathbf F\mathbf C_d\end{bmatrix},

with the disturbance recovered from the augmented state by
:math:`\hat{\mathbf d} = \begin{bmatrix}\mathbf 0 & \mathbf C_d\end{bmatrix}
[\mathbf x; \mathbf z]`.

:func:`random_walk_disturbance` models a slowly-varying (quasi-constant) unknown load;
:func:`harmonic_disturbance` models a narrowband load at a known frequency — the natural
choice for wave-frequency mooring and platform loads. :func:`augment_with_disturbance`
assembles the augmented state space, the disturbance-recovery map, and the augmented
process-noise spectral density (ready for :func:`~vane.sysid.discretize_process_noise`).
This construction is deterministic and auditable, so it lives in the model factory and
not ad hoc in the estimator runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from vane.sysid._covariance import symmetric_part, validate_covariance
from vane.sysid.state_space import StateSpace

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

__all__ = [
    "AugmentedModel",
    "DisturbanceModel",
    "augment_with_disturbance",
    "combine_disturbance_models",
    "gauss_markov_disturbance",
    "harmonic_disturbance",
    "random_walk_disturbance",
    "random_walk_harmonic_disturbance",
]

_TWO_PI = 2.0 * np.pi


@dataclass
class DisturbanceModel:
    """A linear shaping model of an unknown disturbance/load.

    The disturbance ``d`` is the output of a small linear system driven by white noise,
    ``z' = a z + w``, ``d = c z``, with ``w`` a continuous white process of spectral
    density ``intensity``. Different ``a``/``c`` give different prior shapes (a random
    walk, a harmonic oscillator, ...).

    Parameters
    ----------
    a : npt.NDArray[np.float64]
        Disturbance internal dynamics, shape ``(n_dist_states, n_dist_states)``.
    c : npt.NDArray[np.float64]
        Disturbance output map, shape ``(n_signals, n_dist_states)``; ``d = c z`` has
        ``n_signals`` channels.
    intensity : npt.NDArray[np.float64]
        Continuous process-noise spectral density of the disturbance states, a symmetric
        positive-semidefinite ``(n_dist_states, n_dist_states)`` matrix.
    state_names : list[str]
        Names of the disturbance states (length ``n_dist_states``; may be empty).
    """

    a: npt.NDArray[np.float64]
    c: npt.NDArray[np.float64]
    intensity: npt.NDArray[np.float64]
    state_names: list[str]


@dataclass
class AugmentedModel:
    """A state space augmented with a disturbance model and the recovery map.

    Parameters
    ----------
    system : StateSpace
        The augmented continuous state space on ``[x; z]`` (physical states first, then
        disturbance states), with the physical inputs and outputs preserved.
    recovery : npt.NDArray[np.float64]
        The disturbance-recovery map, shape ``(n_signals, n_aug_states)``; the
        disturbance estimate is ``d_hat = recovery @ x_aug``.
    spectral_density : npt.NDArray[np.float64]
        The augmented continuous process-noise spectral density, shape
        ``(n_aug_states, n_aug_states)`` — the physical process noise (or zero) on the
        physical block and the disturbance ``intensity`` on the disturbance block.
    n_disturbance_states : int
        Number of appended disturbance states.
    """

    system: StateSpace
    recovery: npt.NDArray[np.float64]
    spectral_density: npt.NDArray[np.float64]
    n_disturbance_states: int


def combine_disturbance_models(
    disturbances: Sequence[DisturbanceModel],
) -> DisturbanceModel:
    """Combine independent disturbance models without changing their signals.

    The returned shaping model is the block-diagonal union of the supplied dynamics
    and process-noise intensities. Its output matrix is block diagonal as well, so the
    physical disturbance signals remain ordered by model and are not summed. This is
    useful for mixed priors such as aerodynamic torque as a random walk and incident
    wave elevation as a narrowband oscillator.

    Parameters
    ----------
    disturbances : Sequence[DisturbanceModel]
        Non-empty sequence of validated shaping models.

    Returns
    -------
    DisturbanceModel
        Independent block-diagonal union of every state and signal channel.
    """
    models = list(disturbances)
    if not models:
        msg = "at least one disturbance model is required"
        raise ValueError(msg)
    for disturbance in models:
        _validate_disturbance(disturbance)
    state_counts = [model.a.shape[0] for model in models]
    signal_counts = [model.c.shape[0] for model in models]
    total_states = sum(state_counts)
    total_signals = sum(signal_counts)
    a = np.zeros((total_states, total_states), dtype=np.float64)
    c = np.zeros((total_signals, total_states), dtype=np.float64)
    intensity = np.zeros((total_states, total_states), dtype=np.float64)
    state_offset = 0
    signal_offset = 0
    names: list[str] = []
    for model, n_states, n_signals in zip(
        models, state_counts, signal_counts, strict=True
    ):
        states = slice(state_offset, state_offset + n_states)
        signals = slice(signal_offset, signal_offset + n_signals)
        a[states, states] = model.a
        c[signals, states] = model.c
        intensity[states, states] = model.intensity
        names.extend(model.state_names)
        state_offset += n_states
        signal_offset += n_signals
    return DisturbanceModel(a=a, c=c, intensity=intensity, state_names=names)


def random_walk_disturbance(
    intensity: float | Sequence[float],
    *,
    n_signals: int = 1,
    name: str = "load",
) -> DisturbanceModel:
    """Model unknown loads as independent random walks (quasi-constant inputs).

    Each disturbance channel is integrated white noise (``z' = w``, ``d = z``), the
    standard prior for a slowly-varying or constant unknown input. One disturbance state
    per channel.

    Parameters
    ----------
    intensity : float or Sequence[float]
        Continuous process-noise intensity of each random walk; a scalar applies to all
        channels, or one value per channel. All values must be non-negative.
    n_signals : int, optional
        Number of independent disturbance channels (default 1).
    name : str, optional
        Base name for the disturbance states (``"{name}_{i}"``).

    Returns
    -------
    DisturbanceModel
        A random-walk disturbance model with ``n_signals`` states.

    Raises
    ------
    ValueError
        If ``n_signals`` is not positive, ``intensity`` has the wrong length, or any
        intensity is negative.
    """
    if n_signals <= 0:
        msg = f"n_signals must be positive, got {n_signals}"
        raise ValueError(msg)
    values = _broadcast_intensity(intensity, n_signals)
    a = np.zeros((n_signals, n_signals), dtype=np.float64)
    c = np.eye(n_signals, dtype=np.float64)
    names = [f"{name}_{i}" for i in range(n_signals)]
    return DisturbanceModel(a=a, c=c, intensity=np.diag(values), state_names=names)


def gauss_markov_disturbance(
    time_constants_s: float | Sequence[float],
    intensity: float | Sequence[float],
    *,
    n_signals: int = 1,
    name: str = "load",
) -> DisturbanceModel:
    """Model bounded load perturbations as independent first-order processes.

    Each signal follows ``z' = -z / tau + w`` and ``d = z``. The operating-point
    baseline is supplied by the surrounding linear model, so the prior mean remains
    zero in perturbation coordinates while ``tau`` prevents an unconstrained random
    walk from drifting indefinitely. This is suitable for scheduled controller-regulated
    loads such as above-rated aerodynamic torque.

    Parameters
    ----------
    time_constants_s : float or Sequence[float]
        Positive finite mean-reversion time constant per signal in seconds.
    intensity : float or Sequence[float]
        Non-negative continuous process-noise intensity per signal.
    n_signals : int, optional
        Number of independent signals; default one.
    name : str, optional
        Base name for the disturbance states.
    """
    if n_signals <= 0:
        msg = f"n_signals must be positive, got {n_signals}"
        raise ValueError(msg)
    constants = _broadcast_positive_values(
        time_constants_s, n_signals, "time_constants_s"
    )
    values = _broadcast_intensity(intensity, n_signals)
    return DisturbanceModel(
        a=-np.diag(1.0 / constants),
        c=np.eye(n_signals, dtype=np.float64),
        intensity=np.diag(values),
        state_names=[f"{name}_{index}" for index in range(n_signals)],
    )


def harmonic_disturbance(
    frequencies_hz: Sequence[float],
    intensity: float | Sequence[float],
    *,
    name: str = "wave",
) -> DisturbanceModel:
    """Model unknown loads as narrowband oscillators at known frequencies.

    Each frequency contributes a two-state undamped oscillator
    ``[[0, ω], [-ω, 0]]`` (``ω = 2π f``) driven by white noise, with the disturbance
    read from its first state. This is the natural prior for a wave-frequency load whose
    frequency is known but whose amplitude and phase are not.

    Parameters
    ----------
    frequencies_hz : Sequence[float]
        Oscillator frequencies in Hz, all finite and positive. One disturbance channel
        (and two states) per frequency.
    intensity : float or Sequence[float]
        Continuous process-noise intensity driving each oscillator (applied to both of
        its states); a scalar applies to all, or one value per frequency. Non-negative.
    name : str, optional
        Base name for the disturbance states (``"{name}_{i}_{re,im}"``).

    Returns
    -------
    DisturbanceModel
        A harmonic disturbance model with ``2 * len(frequencies_hz)`` states and one
        output channel per frequency.

    Raises
    ------
    ValueError
        If no frequencies are given, any frequency is non-finite or not positive,
        ``intensity`` has the wrong length, or any intensity is negative.
    """
    freqs = np.asarray(frequencies_hz, dtype=np.float64)
    if freqs.ndim != 1 or freqs.size == 0:
        msg = "frequencies_hz must be a non-empty 1-D sequence"
        raise ValueError(msg)
    if not np.all(np.isfinite(freqs)) or np.any(freqs <= 0.0):
        msg = "oscillator frequencies must be finite and positive"
        raise ValueError(msg)
    with np.errstate(over="ignore"):
        angular_frequencies = _TWO_PI * freqs
    if not np.all(np.isfinite(angular_frequencies)):
        msg = "oscillator angular frequencies must remain finite"
        raise ValueError(msg)
    n_freq = freqs.shape[0]
    values = _broadcast_intensity(intensity, n_freq)

    n_states = 2 * n_freq
    a = np.zeros((n_states, n_states), dtype=np.float64)
    c = np.zeros((n_freq, n_states), dtype=np.float64)
    diag = np.empty(n_states, dtype=np.float64)
    names: list[str] = []
    for i, omega in enumerate(angular_frequencies):
        block = 2 * i
        a[block, block + 1] = omega
        a[block + 1, block] = -omega
        c[i, block] = 1.0
        diag[block] = diag[block + 1] = values[i]
        names += [f"{name}_{i}_re", f"{name}_{i}_im"]
    return DisturbanceModel(a=a, c=c, intensity=np.diag(diag), state_names=names)


def random_walk_harmonic_disturbance(
    frequencies_hz: Sequence[float],
    random_walk_intensity: float,
    harmonic_intensity: float | Sequence[float],
    *,
    name: str = "load",
) -> DisturbanceModel:
    """Model one unknown load as a DC random walk plus scheduled harmonics.

    Unlike :func:`harmonic_disturbance`, whose frequencies are separate disturbance
    channels, this construction sums every component into one physical signal:
    ``d = z_dc + sum(z_h,re)``.  It is intended for a rotor load containing a slowly
    varying component together with known 1P/3P orders.  Amplitudes and phases remain
    estimated states; only the physics-derived frequencies are prescribed.

    Parameters
    ----------
    frequencies_hz : Sequence[float]
        Positive harmonic frequencies in hertz.
    random_walk_intensity : float
        Continuous process-noise intensity of the DC random-walk state.
    harmonic_intensity : float or Sequence[float]
        Continuous process-noise intensity per harmonic oscillator. A scalar
        broadcasts; each value is applied to both quadrature states.
    name : str, optional
        Base name for the DC and harmonic states.
    """
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size == 0:
        msg = "frequencies_hz must be a non-empty 1-D sequence"
        raise ValueError(msg)
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        msg = "harmonic frequencies must be finite and positive"
        raise ValueError(msg)
    with np.errstate(over="ignore"):
        angular_frequencies = _TWO_PI * frequencies
    if not np.all(np.isfinite(angular_frequencies)):
        msg = "harmonic angular frequencies must remain finite"
        raise ValueError(msg)
    if np.unique(frequencies).size != frequencies.size:
        msg = "harmonic frequencies must be unique"
        raise ValueError(msg)
    dc = float(random_walk_intensity)
    if not np.isfinite(dc) or dc < 0.0:
        msg = "random_walk_intensity must be finite and non-negative"
        raise ValueError(msg)
    harmonic_values = _broadcast_intensity(harmonic_intensity, frequencies.size)

    n_states = 1 + 2 * frequencies.size
    a = np.zeros((n_states, n_states), dtype=np.float64)
    c = np.zeros((1, n_states), dtype=np.float64)
    c[0, 0] = 1.0
    intensity = np.empty(n_states, dtype=np.float64)
    intensity[0] = dc
    names = [f"{name}_dc"]
    for index, omega in enumerate(angular_frequencies):
        block = 1 + 2 * index
        a[block, block + 1] = omega
        a[block + 1, block] = -omega
        c[0, block] = 1.0
        intensity[block : block + 2] = harmonic_values[index]
        names.extend((f"{name}_{index}_re", f"{name}_{index}_im"))
    return DisturbanceModel(
        a=a,
        c=c,
        intensity=np.diag(intensity),
        state_names=names,
    )


def augment_with_disturbance(
    system: StateSpace,
    disturbance: DisturbanceModel,
    *,
    entry: npt.NDArray[np.float64],
    output_entry: npt.NDArray[np.float64] | None = None,
    process_noise: npt.NDArray[np.float64] | None = None,
) -> AugmentedModel:
    """Append a disturbance model to a state space for unknown-input estimation.

    Builds the augmented continuous system on ``[x; z]`` with the disturbance entering
    the physical dynamics through ``entry`` (``x' += entry @ d``). An optional
    ``output_entry`` preserves direct disturbance-to-output feedthrough
    (``y += output_entry @ d``). The physical inputs and outputs are otherwise
    preserved, and the disturbance is recovered from the augmented state.

    Parameters
    ----------
    system : StateSpace
        A **continuous-time** physical model (``dt`` is ``None``).
    disturbance : DisturbanceModel
        The disturbance shaping model (e.g. from :func:`random_walk_disturbance` or
        :func:`harmonic_disturbance`).
    entry : npt.NDArray[np.float64]
        Input map of the disturbance into the physical state derivative, shape
        ``(n_states, n_signals)`` where ``n_signals`` is ``disturbance.c``'s row count.
        Often a column block of ``B`` (the load enters where a known input does).
    output_entry : npt.NDArray[np.float64] or None, optional
        Direct disturbance-to-output map, shape ``(n_outputs, n_signals)``. This is
        the disturbance column block removed from the physical system's ``D`` matrix
        when an originally known input is promoted to an augmented state. ``None``
        means there is no direct disturbance feedthrough.
    process_noise : npt.NDArray[np.float64] or None, optional
        Continuous process-noise spectral density of the physical states, shape
        ``(n_states, n_states)``. ``None`` uses zero physical process noise.

    Returns
    -------
    AugmentedModel
        The augmented system, the disturbance-recovery map, the augmented process-noise
        spectral density, and the disturbance-state count.

    Raises
    ------
    ValueError
        If ``system`` is discrete, ``entry``/``output_entry`` do not match the physical
        system and disturbance signals, or ``process_noise`` has the wrong shape.
    """
    if system.dt is not None:
        msg = "augment_with_disturbance requires a continuous-time system (dt is None)"
        raise ValueError(msg)
    _validate_disturbance(disturbance)
    a = np.asarray(system.a, dtype=np.float64)
    n = a.shape[0]
    n_signals = disturbance.c.shape[0]
    m = disturbance.a.shape[0]
    entry = np.asarray(entry, dtype=np.float64)
    if entry.shape != (n, n_signals):
        msg = (
            f"entry must be ({n}, {n_signals}) to map {n_signals} disturbance signals "
            f"into {n} states, got {entry.shape}"
        )
        raise ValueError(msg)
    direct_output = None
    if output_entry is not None:
        if system.c is None:
            msg = "output_entry requires a physical C matrix"
            raise ValueError(msg)
        direct_output = np.asarray(output_entry, dtype=np.float64)
        expected = (system.n_outputs, n_signals)
        if direct_output.shape != expected:
            msg = (
                f"output_entry must be {expected} to map {n_signals} disturbance "
                f"signals into {system.n_outputs} outputs, got {direct_output.shape}"
            )
            raise ValueError(msg)
        if not np.all(np.isfinite(direct_output)):
            msg = "output_entry must be finite (no NaN or inf)"
            raise ValueError(msg)
    if process_noise is not None:
        process_noise = np.asarray(process_noise, dtype=np.float64)
        if process_noise.shape != (n, n):
            msg = f"process_noise must be ({n}, {n}), got {process_noise.shape}"
            raise ValueError(msg)
        _validate_covariance(process_noise, "process_noise")
        process_noise = symmetric_part(process_noise)

    a_aug = np.zeros((n + m, n + m), dtype=np.float64)
    a_aug[:n, :n] = a
    a_aug[:n, n:] = entry @ disturbance.c
    a_aug[n:, n:] = disturbance.a

    b_aug = None
    if system.b is not None:
        b_aug = np.zeros((n + m, system.b.shape[1]), dtype=np.float64)
        b_aug[:n] = system.b

    c_aug = None
    if system.c is not None:
        c_aug = np.zeros((system.c.shape[0], n + m), dtype=np.float64)
        c_aug[:, :n] = system.c
        if direct_output is not None:
            c_aug[:, n:] = direct_output @ disturbance.c

    state_names = _augmented_state_names(system, disturbance, n, m)
    augmented = StateSpace(
        a=a_aug,
        b=b_aug,
        c=c_aug,
        d=system.d,
        state_names=state_names,
        input_names=list(system.input_names),
        output_names=list(system.output_names),
        output_mbc_coordinates=list(system.output_mbc_coordinates),
        dt=None,
    )

    recovery = np.zeros((n_signals, n + m), dtype=np.float64)
    recovery[:, n:] = disturbance.c

    spectral_density = np.zeros((n + m, n + m), dtype=np.float64)
    if process_noise is not None:
        spectral_density[:n, :n] = process_noise
    # The validated symmetric part, as for the physical process-noise block.
    spectral_density[n:, n:] = symmetric_part(
        np.asarray(disturbance.intensity, dtype=np.float64)
    )

    return AugmentedModel(
        system=augmented,
        recovery=recovery,
        spectral_density=spectral_density,
        n_disturbance_states=m,
    )


def _validate_disturbance(disturbance: DisturbanceModel) -> None:
    """Validate a disturbance model's matrices are mutually consistent covariances."""
    a = np.asarray(disturbance.a, dtype=np.float64)
    c = np.asarray(disturbance.c, dtype=np.float64)
    intensity = np.asarray(disturbance.intensity, dtype=np.float64)
    m = a.shape[0]
    if a.ndim != 2 or a.shape[1] != m:
        msg = f"disturbance.a must be square, got shape {a.shape}"
        raise ValueError(msg)
    if c.ndim != 2 or c.shape[1] != m:
        msg = f"disturbance.c must have {m} columns to match a, got {c.shape}"
        raise ValueError(msg)
    if intensity.shape != (m, m):
        msg = (
            f"disturbance.intensity must be ({m}, {m}) to match a, "
            f"got {intensity.shape}"
        )
        raise ValueError(msg)
    _validate_covariance(intensity, "disturbance.intensity")


def _validate_covariance(matrix: npt.NDArray[np.float64], name: str) -> None:
    """Validate that ``matrix`` is a symmetric positive-semidefinite covariance."""
    if not np.all(np.isfinite(matrix)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
    validate_covariance(matrix, name, definite=False)


def _broadcast_intensity(
    intensity: float | Sequence[float], n: int
) -> npt.NDArray[np.float64]:
    """Broadcast a scalar/length-``n`` intensity to a validated length-``n`` array."""
    values = np.atleast_1d(np.asarray(intensity, dtype=np.float64))
    if values.size == 1:
        values = np.repeat(values, n)
    if values.shape != (n,):
        msg = f"intensity must be a scalar or have {n} values, got {values.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        msg = "disturbance intensities must be finite and non-negative"
        raise ValueError(msg)
    return values


def _broadcast_positive_values(
    value: float | Sequence[float], n: int, name: str
) -> npt.NDArray[np.float64]:
    """Broadcast a scalar/length-``n`` vector of finite positive values."""
    values = np.atleast_1d(np.asarray(value, dtype=np.float64))
    if values.size == 1:
        values = np.repeat(values, n)
    if values.shape != (n,):
        msg = f"{name} must be a scalar or have {n} values, got {values.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        msg = f"{name} must contain finite positive values"
        raise ValueError(msg)
    return values


def _augmented_state_names(
    system: StateSpace, disturbance: DisturbanceModel, n: int, m: int
) -> list[str]:
    """Build the augmented state-name list, or empty if neither side is named."""
    if not system.state_names and not disturbance.state_names:
        return []
    physical = system.state_names or [f"state_{i}" for i in range(n)]
    dist = disturbance.state_names or [f"disturbance_{i}" for i in range(m)]
    return [*physical, *dist]
