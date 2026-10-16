# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Post-run acceptance checks for a linearized operating point.

``CalcSteady`` establishes periodicity according to OpenFAST's normalized output
criterion.  That criterion is necessary, but a deliberately loose ``TrimTol`` can
still stop while a floating platform carries a secular six-DOF drift.  Such a point
is parseable and can produce a finite MBC model, yet it is a poor equilibrium about
which to build a small-perturbation model.

This module provides an independent physical check.  It reads the ElastoDyn platform
rate states from every full-system ``.lin`` file and estimates the azimuth mean of each
rate component with quadrature weights that exactly annihilate the 1P, 2P and 3P
harmonics at the *actual* sample azimuths.  Periodic motion therefore cancels; a secular
drift does not.  The resulting translational and angular residuals can reject a
nominally successful trim without mistaking ordinary once-per-revolution motion for
non-convergence.

The weights are the minimum-norm solution of the harmonic moment conditions, which for
seven or more samples is exactly the constant term of a least-squares fit of a mean
plus 1P--3P harmonics.  On an exactly uniform sweep they reduce to the equal-weight
average.  They remain valid for the non-uniform azimuths ``CalcSteady`` actually
produces: OpenFAST takes each snapshot at the first time step at or past its target
azimuth, so real sweeps are uniform only to within ``RotSpeed * DT``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from vane.io.lin_reader import read_lin_file

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from vane.io.lin_reader import LinFile

__all__ = [
    "LinearizationAcceptancePolicy",
    "OperatingPointResidualAudit",
    "PeriodicStateResidualAudit",
    "StateDerivativeResidual",
    "audit_operating_point_residual",
    "audit_periodic_state_residual",
]

_TRANSLATIONAL_COMPONENTS = ("surge", "sway", "heave")
_ANGULAR_COMPONENTS = ("roll", "pitch", "yaw")
_ROTATING_SPEED_FLOOR_RAD_S = 1.0e-6
_DEFAULT_MIN_PERIODIC_AZIMUTHS = 12
_MAX_PLATFORM_RATE_HARMONIC = 3
# The harmonic moment conditions must be satisfiable to rounding accuracy; an
# inconsistent system means a 1P--3P harmonic aliases onto the mean (three samples).
_HARMONIC_MOMENT_ATOL = 1.0e-9
# Largest accepted variance inflation ``n * sum(w**2)`` of the mean weights relative to
# an equal-weight mean of the same samples.  A uniform sweep gives exactly 1, CalcSteady
# step quantisation 1 + O((RotSpeed*DT)**2), and a sweep missing one of nine samples
# 4/3; partial revolutions give 8 (three quarters) to thousands (one half), because the
# mean is then extrapolated from large alternating weights that amplify any unresolved
# 4P-and-higher content or numerical noise.
_MAX_MEAN_WEIGHT_INFLATION = 2.0


@dataclass(frozen=True)
class LinearizationAcceptancePolicy:
    """Physical residual limits for a floating operating point.

    Parameters
    ----------
    max_mean_translational_speed : float
        Maximum norm of the azimuth-mean platform translational velocity, in m/s.
    max_mean_angular_speed : float
        Maximum norm of the azimuth-mean platform angular velocity, in rad/s.
    min_rotating_azimuths : int
        Minimum number of distinct azimuth samples needed to distinguish periodic
        motion from secular drift for a rotating point. The default is 12 and values
        below four are prohibited so blade-passing motion cannot alias into the mean.
        A non-rotating point needs only one linearization.
    """

    max_mean_translational_speed: float = 1.0e-3
    max_mean_angular_speed: float = 1.0e-4
    min_rotating_azimuths: int = _DEFAULT_MIN_PERIODIC_AZIMUTHS

    def __post_init__(self) -> None:
        """Reject non-finite/non-positive limits or an undersized sweep."""
        for name, value in (
            ("max_mean_translational_speed", self.max_mean_translational_speed),
            ("max_mean_angular_speed", self.max_mean_angular_speed),
        ):
            if (
                not isinstance(value, int | float)
                or not math.isfinite(value)
                or value <= 0
            ):
                msg = f"{name} must be finite and positive, got {value!r}"
                raise ValueError(msg)
        if (
            isinstance(self.min_rotating_azimuths, bool)
            or not isinstance(self.min_rotating_azimuths, int)
            or self.min_rotating_azimuths < 4
        ):
            msg = (
                "min_rotating_azimuths must be at least 4 to resolve 3P motion at a "
                f"operating point, got {self.min_rotating_azimuths}"
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class OperatingPointResidualAudit:
    """Physical residual evidence extracted from one linearization sweep."""

    assessed: bool
    accepted: bool
    reason: str | None
    file_count: int
    distinct_azimuths: int
    rotating: bool
    translational_channels: tuple[str, ...]
    angular_channels: tuple[str, ...]
    mean_translational_velocity: tuple[float, ...] | None
    mean_angular_velocity: tuple[float, ...] | None
    mean_translational_speed: float | None
    mean_angular_speed: float | None
    max_instantaneous_translational_speed: float | None
    max_instantaneous_angular_speed: float | None
    max_mean_translational_speed: float
    max_mean_angular_speed: float


@dataclass(frozen=True)
class StateDerivativeResidual:
    """Azimuth statistics for one non-rotating continuous-state derivative.

    ``mean_to_rms_ratio`` is dimensionless and invariant to a change of state units.
    It approaches zero for a resolved zero-mean periodic derivative and one for a
    constant secular derivative.  It is ``None`` when the derivative is inactive at
    floating-point resolution; absolute ``mean_derivative`` and ``rms_derivative``
    values are retained so this ratio is never mistaken for an engineering limit.
    """

    state_description: str
    derivative_description: str
    module: str
    unit: str
    mean_derivative: float
    rms_derivative: float
    oscillatory_rms_derivative: float
    mean_to_rms_ratio: float | None
    active: bool


@dataclass(frozen=True)
class PeriodicStateResidualAudit:
    """State-derivative residuals for a periodic OpenFAST operating point.

    Rotating-frame states are excluded because their native derivatives contain the
    intended rotor-periodic coordinate transport.  The generator-azimuth coordinate
    is also excluded because its derivative is the intended rotor speed; its angular
    acceleration state remains in the audit.  No heterogeneous state units are
    combined into one physical norm.
    """

    assessed: bool
    reason: str | None
    file_count: int
    distinct_azimuths: int
    rotating: bool
    min_rotating_azimuths: int
    channels: tuple[StateDerivativeResidual, ...]
    excluded_rotating_channels: tuple[str, ...]
    excluded_cyclic_channels: tuple[str, ...]
    max_mean_to_rms_ratio: float | None


def audit_operating_point_residual(
    lin_files: Sequence[str | Path | LinFile],
    policy: LinearizationAcceptancePolicy | None = None,
) -> OperatingPointResidualAudit:
    """Audit secular platform drift across an OpenFAST azimuth sweep.

    Paths are parsed with VANE's strict ``.lin`` reader; already parsed
    :class:`~vane.io.lin_reader.LinFile` objects are accepted too.  If the model has no
    ElastoDyn platform-rate states (for example a fixed-bottom turbine), the check is
    not applicable and returns ``assessed=False, accepted=True``.  A rotating point
    with too few distinct azimuths is not accepted because an instantaneous periodic
    rate cannot be separated from secular drift.

    Parameters
    ----------
    lin_files : Sequence[path-like or LinFile]
        Full-system files from one operating point.
    policy : LinearizationAcceptancePolicy or None, optional
        Residual limits; defaults to :class:`LinearizationAcceptancePolicy`.

    Returns
    -------
    OperatingPointResidualAudit
        The component means, vector norms, thresholds, and acceptance decision.

    Raises
    ------
    ValueError
        If no files are supplied or their state descriptions are inconsistent.
    """
    if not lin_files:
        msg = "lin_files must contain at least one linearization"
        raise ValueError(msg)
    limits = policy or LinearizationAcceptancePolicy()
    files = [_as_lin_file(item) for item in lin_files]
    reference_descriptions = files[0].x.descriptions
    if any(item.x.descriptions != reference_descriptions for item in files[1:]):
        msg = "linearization files have inconsistent state descriptions"
        raise ValueError(msg)

    rotating = bool(
        np.mean(np.abs([item.rotor_speed for item in files]))
        > _ROTATING_SPEED_FLOOR_RAD_S
    )
    azimuths = _normalized_distinct_azimuths(files)
    distinct_azimuths = len(azimuths)
    translational = _component_indices(
        reference_descriptions, _TRANSLATIONAL_COMPONENTS
    )
    angular = _component_indices(reference_descriptions, _ANGULAR_COMPONENTS)
    translational_names = tuple(
        reference_descriptions[index] for index in translational
    )
    angular_names = tuple(reference_descriptions[index] for index in angular)

    if not translational and not angular:
        return OperatingPointResidualAudit(
            assessed=False,
            accepted=True,
            reason=None,
            file_count=len(files),
            distinct_azimuths=distinct_azimuths,
            rotating=rotating,
            translational_channels=(),
            angular_channels=(),
            mean_translational_velocity=None,
            mean_angular_velocity=None,
            mean_translational_speed=None,
            mean_angular_speed=None,
            max_instantaneous_translational_speed=None,
            max_instantaneous_angular_speed=None,
            max_mean_translational_speed=limits.max_mean_translational_speed,
            max_mean_angular_speed=limits.max_mean_angular_speed,
        )

    coverage_reason = (
        _rotating_azimuth_coverage_reason(
            len(files), azimuths, limits.min_rotating_azimuths
        )
        if rotating
        else None
    )
    if coverage_reason is not None:
        reason = f"rotating operating-point residual is unassessed: {coverage_reason}"
        return OperatingPointResidualAudit(
            assessed=False,
            accepted=False,
            reason=reason,
            file_count=len(files),
            distinct_azimuths=distinct_azimuths,
            rotating=True,
            translational_channels=translational_names,
            angular_channels=angular_names,
            mean_translational_velocity=None,
            mean_angular_velocity=None,
            mean_translational_speed=None,
            mean_angular_speed=None,
            max_instantaneous_translational_speed=None,
            max_instantaneous_angular_speed=None,
            max_mean_translational_speed=limits.max_mean_translational_speed,
            max_mean_angular_speed=limits.max_mean_angular_speed,
        )

    weights = _mean_weights(files, rotating)
    trans_values = _values(files, translational)
    angular_values = _values(files, angular)
    trans_mean, trans_residual, trans_max = _vector_metrics(trans_values, weights)
    angular_mean, angular_residual, angular_max = _vector_metrics(
        angular_values, weights
    )
    failures: list[str] = []
    if (
        trans_residual is not None
        and trans_residual > limits.max_mean_translational_speed
    ):
        failures.append(
            "mean platform translational drift "
            f"{trans_residual:.6g} m/s exceeds "
            f"{limits.max_mean_translational_speed:.6g} m/s"
        )
    if (
        angular_residual is not None
        and angular_residual > limits.max_mean_angular_speed
    ):
        failures.append(
            "mean platform angular drift "
            f"{angular_residual:.6g} rad/s exceeds "
            f"{limits.max_mean_angular_speed:.6g} rad/s"
        )
    final_reason = "; ".join(failures) or None
    return OperatingPointResidualAudit(
        assessed=True,
        accepted=not failures,
        reason=final_reason,
        file_count=len(files),
        distinct_azimuths=distinct_azimuths,
        rotating=rotating,
        translational_channels=translational_names,
        angular_channels=angular_names,
        mean_translational_velocity=trans_mean,
        mean_angular_velocity=angular_mean,
        mean_translational_speed=trans_residual,
        mean_angular_speed=angular_residual,
        max_instantaneous_translational_speed=trans_max,
        max_instantaneous_angular_speed=angular_max,
        max_mean_translational_speed=limits.max_mean_translational_speed,
        max_mean_angular_speed=limits.max_mean_angular_speed,
    )


def _normalized_distinct_azimuths(files: Sequence[LinFile]) -> tuple[float, ...]:
    """Return unique azimuths on ``[0, 2*pi)`` with parser-scale rounding."""
    return tuple(
        sorted({round(float(item.azimuth) % (2.0 * math.pi), 10) for item in files})
    )


def _harmonic_mean_weights(
    azimuths: Sequence[float],
) -> npt.NDArray[np.float64] | None:
    """Return mean-quadrature weights that annihilate the 1P--3P harmonics.

    The weights ``w`` solve ``sum(w) = 1`` and ``sum(w * cos(k*psi)) =
    sum(w * sin(k*psi)) = 0`` for ``k = 1..3`` with minimum Euclidean norm.  For seven
    or more samples this equals the constant term of the least-squares fit of a mean
    plus 1P--3P harmonics; on a uniform sweep every weight is ``1/n``.

    Returns
    -------
    numpy.ndarray or None
        One weight per sample in input order, or ``None`` when the moment conditions
        are inconsistent (a harmonic aliases onto the mean) or the weights inflate the
        mean's variance by more than ``_MAX_MEAN_WEIGHT_INFLATION``.
    """
    values = np.asarray(azimuths, dtype=np.float64)
    rows = [np.ones_like(values)]
    for harmonic in range(1, _MAX_PLATFORM_RATE_HARMONIC + 1):
        rows.extend((np.cos(harmonic * values), np.sin(harmonic * values)))
    moments = np.vstack(rows)
    target = np.zeros(moments.shape[0], dtype=np.float64)
    target[0] = 1.0
    weights = np.linalg.lstsq(moments, target, rcond=None)[0]
    if float(np.max(np.abs(moments @ weights - target))) > _HARMONIC_MOMENT_ATOL:
        return None
    inflation = float(values.size * np.sum(weights**2))
    if not math.isfinite(inflation) or inflation > _MAX_MEAN_WEIGHT_INFLATION:
        return None
    return np.asarray(weights, dtype=np.float64)


def _rotating_azimuth_coverage_reason(
    file_count: int,
    azimuths: Sequence[float],
    minimum: int,
) -> str | None:
    """Return the reason a rotating sweep cannot separate the mean from 1P--3P."""
    if len(azimuths) < minimum:
        return f"{len(azimuths)} distinct azimuth(s), need at least {minimum}"
    if len(azimuths) != file_count:
        return "duplicate azimuth samples would weight the residual mean"
    if _harmonic_mean_weights(azimuths) is None:
        return (
            "azimuth layout cannot separate the mean from the 1P through 3P harmonics"
        )
    return None


def _mean_weights(files: Sequence[LinFile], rotating: bool) -> npt.NDArray[np.float64]:
    """Return per-file weights for the azimuth mean (equal weights when parked)."""
    if rotating:
        weights = _harmonic_mean_weights([float(item.azimuth) for item in files])
        # The coverage check has already accepted this layout.
        assert weights is not None
        return weights
    return np.full(len(files), 1.0 / len(files), dtype=np.float64)


def audit_periodic_state_residual(
    lin_files: Sequence[str | Path | LinFile],
    *,
    min_rotating_azimuths: int = _DEFAULT_MIN_PERIODIC_AZIMUTHS,
) -> PeriodicStateResidualAudit:
    """Audit mean derivatives without mixing state units into a scalar norm.

    The derivative operating point in each ``.lin`` file is sampled over azimuth.
    For every non-rotating, non-cyclic continuous state, this function reports the
    azimuth mean, total RMS, zero-mean oscillatory RMS, and the scale-invariant ratio
    ``abs(mean) / RMS``.  A resolved periodic equilibrium has a small ratio, whereas a
    secular derivative survives the azimuth mean.  Absolute residuals must still be
    interpreted in their stated engineering units.

    Parameters
    ----------
    lin_files : Sequence[path-like or LinFile]
        Full-system files from one operating point.
    min_rotating_azimuths : int, optional
        Minimum number of distinct azimuths required before the periodic statistics
        are assessed.  The default is 12; publication campaigns should normally use
        36.  The denser requirement prevents a 3P component from aliasing into a
        three-azimuth mean.

    Returns
    -------
    PeriodicStateResidualAudit
        Per-state residual statistics and the explicit exclusions.

    Raises
    ------
    ValueError
        If no files are supplied, the requested sample count is smaller than four,
        or the state/derivative channel layouts disagree.
    """
    if not lin_files:
        msg = "lin_files must contain at least one linearization"
        raise ValueError(msg)
    if (
        isinstance(min_rotating_azimuths, bool)
        or not isinstance(min_rotating_azimuths, int)
        or min_rotating_azimuths < 4
    ):
        msg = (
            "min_rotating_azimuths must be at least 4 for a periodic state audit, "
            f"got {min_rotating_azimuths}"
        )
        raise ValueError(msg)

    files = [_as_lin_file(item) for item in lin_files]
    state_descriptions = files[0].x.descriptions
    derivative_descriptions = files[0].xdot.descriptions
    if len(state_descriptions) != len(derivative_descriptions):
        msg = (
            "continuous-state and derivative table lengths disagree: "
            f"{len(state_descriptions)} versus {len(derivative_descriptions)}"
        )
        raise ValueError(msg)
    for item in files[1:]:
        if item.x.descriptions != state_descriptions:
            msg = "linearization files have inconsistent state descriptions"
            raise ValueError(msg)
        if item.xdot.descriptions != derivative_descriptions:
            msg = "linearization files have inconsistent derivative descriptions"
            raise ValueError(msg)

    rotating = bool(
        np.mean(np.abs([item.rotor_speed for item in files]))
        > _ROTATING_SPEED_FLOOR_RAD_S
    )
    azimuths = _normalized_distinct_azimuths(files)
    distinct_azimuths = len(azimuths)
    rotating_mask = np.asarray(files[0].x.rotating_frame, dtype=np.bool_)
    if rotating_mask.shape != (len(state_descriptions),):
        msg = "continuous-state rotating-frame flags do not align with descriptions"
        raise ValueError(msg)

    excluded_rotating = tuple(
        description
        for description, is_rotating in zip(
            state_descriptions, rotating_mask, strict=True
        )
        if is_rotating
    )
    cyclic_mask = np.asarray(
        [_is_cyclic_coordinate(item) for item in state_descriptions], dtype=np.bool_
    )
    excluded_cyclic = tuple(
        description
        for description, is_cyclic in zip(state_descriptions, cyclic_mask, strict=True)
        if is_cyclic
    )

    coverage_reason = (
        _rotating_azimuth_coverage_reason(len(files), azimuths, min_rotating_azimuths)
        if rotating
        else None
    )
    if coverage_reason is not None:
        reason = f"periodic state residual is unassessed: {coverage_reason}"
        return PeriodicStateResidualAudit(
            assessed=False,
            reason=reason,
            file_count=len(files),
            distinct_azimuths=distinct_azimuths,
            rotating=True,
            min_rotating_azimuths=min_rotating_azimuths,
            channels=(),
            excluded_rotating_channels=excluded_rotating,
            excluded_cyclic_channels=excluded_cyclic,
            max_mean_to_rms_ratio=None,
        )

    retained = np.flatnonzero(~rotating_mask & ~cyclic_mask)
    modules = files[0].x.modules
    values = np.asarray(
        [[item.xdot.values[index] for index in retained] for item in files],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        msg = "continuous-state derivative operating points contain NaN or Inf"
        raise ValueError(msg)

    weights = _mean_weights(files, rotating)
    channels: list[StateDerivativeResidual] = []
    ratios: list[float] = []
    for column, index in enumerate(retained.tolist()):
        samples = values[:, column]
        mean = float(weights @ samples)
        rms = float(np.sqrt(np.mean(samples**2)))
        oscillatory_rms = float(np.sqrt(np.mean((samples - mean) ** 2)))
        activity_tolerance = (
            100.0
            * float(np.finfo(np.float64).eps)
            * max(float(np.max(np.abs(samples))), 1.0)
        )
        active = rms > activity_tolerance
        ratio = abs(mean) / rms if active else None
        if ratio is not None:
            ratios.append(ratio)
        channels.append(
            StateDerivativeResidual(
                state_description=state_descriptions[index],
                derivative_description=derivative_descriptions[index],
                module=modules[index],
                unit=_trailing_unit(derivative_descriptions[index]),
                mean_derivative=mean,
                rms_derivative=rms,
                oscillatory_rms_derivative=oscillatory_rms,
                mean_to_rms_ratio=ratio,
                active=active,
            )
        )

    return PeriodicStateResidualAudit(
        assessed=True,
        reason=None,
        file_count=len(files),
        distinct_azimuths=distinct_azimuths,
        rotating=rotating,
        min_rotating_azimuths=min_rotating_azimuths,
        channels=tuple(channels),
        excluded_rotating_channels=excluded_rotating,
        excluded_cyclic_channels=excluded_cyclic,
        max_mean_to_rms_ratio=max(ratios) if ratios else None,
    )


def _as_lin_file(item: str | Path | LinFile) -> LinFile:
    """Return ``item`` as a parsed ``LinFile``."""
    if isinstance(item, (str, Path)):
        return read_lin_file(Path(item))
    return item


def _is_cyclic_coordinate(description: str) -> bool:
    """Return whether a state is the advancing generator/rotor azimuth coordinate."""
    value = description.lower()
    return (
        "variable speed generator dof" in value
        and "first time derivative of" not in value
    )


def _trailing_unit(description: str) -> str:
    """Return a derivative description's trailing comma-delimited unit."""
    if "," not in description:
        return ""
    return description.rsplit(",", 1)[-1].strip()


def _component_indices(
    descriptions: Sequence[str], components: Sequence[str]
) -> tuple[int, ...]:
    """Locate enabled ElastoDyn platform rate states in physical component order."""
    indices: list[int] = []
    lowered = [description.lower() for description in descriptions]
    for component in components:
        matches = [
            index
            for index, description in enumerate(lowered)
            if "ed first time derivative of platform" in description
            and component in description
            and "dof" in description
        ]
        if len(matches) > 1:
            msg = f"ambiguous ElastoDyn platform-{component} rate states: {matches}"
            raise ValueError(msg)
        if matches:
            indices.append(matches[0])
    return tuple(indices)


def _values(
    files: Sequence[LinFile], indices: tuple[int, ...]
) -> npt.NDArray[np.float64]:
    """Stack the selected state operating-point values by azimuth."""
    if not indices:
        return np.empty((len(files), 0), dtype=np.float64)
    return np.asarray(
        [[item.x.values[index] for index in indices] for item in files],
        dtype=np.float64,
    )


def _vector_metrics(
    values: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> tuple[tuple[float, ...] | None, float | None, float | None]:
    """Return component mean, norm of that mean, and maximum instantaneous norm."""
    if values.shape[1] == 0:
        return None, None, None
    mean = weights @ values
    return (
        tuple(float(value) for value in mean),
        float(np.linalg.norm(mean)),
        float(np.max(np.linalg.norm(values, axis=1))),
    )
