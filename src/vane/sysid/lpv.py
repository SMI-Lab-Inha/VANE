# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Online scheduled (LPV) estimator over a :class:`~vane.sysid.RomFamily`.

:func:`~vane.sysid.build_rom_family` *packages* a turbine's linearizations into one
mode-tracked, phase-aligned reduced model per operating point -- the static half of a
linear-parameter-varying (LPV) estimator. This module is the runtime half: given a
scheduling parameter (wind speed, rotor speed, ...) it **selects and interpolates** the
family into a single model at that operating point, and runs a Kalman filter over a
record whose operating point drifts, carrying the state and covariance across the sweep.

Two things make the interpolation legitimate. The family's realizations are tracked into
a **consistent state order** (index ``k`` is the same physical mode everywhere),
**phase-aligned** and **scale-normalized** across points, so a model's state vector
means the same thing at
adjacent operating points and the matrices may be blended entry-by-entry; the running
estimate's state and covariance therefore stay valid as the schedule moves -- no
discontinuous re-basing at a switch. And each tracked mode carries a **tracking MAC**
against its
neighbour, so where the tracking is uncertain (a crossing or veering region, or an
operating point outside the family's range) the scheduler surfaces a low-confidence flag
rather than interpolating two unrelated modes silently.

This is the "run the estimator over a wind-speed sweep" product layer the ROM family was
built for.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from vane.sysid.kalman_filter import KalmanFilter
from vane.sysid.kalman_interface import build_kalman_model
from vane.sysid.state_space import StateSpace

if TYPE_CHECKING:
    import numpy.typing as npt

    from vane.sysid.covariance_init import CovarianceSet
    from vane.sysid.rom_family import RomFamily

__all__ = [
    "ScheduledFilterResult",
    "ScheduledModel",
    "run_scheduled_filter",
    "schedule_model",
]

_log = logging.getLogger(__name__)


@dataclass
class ScheduledModel:
    """The family interpolated to a single operating point, with a confidence flag.

    Parameters
    ----------
    system : StateSpace
        The model at ``operating_point`` -- a linear blend of the bracketing family
        members (or the nearest member when the point is outside the family range). Same
        time domain (continuous/discrete) and I/O layout as the family.
    operating_point : float
        The scheduling-parameter value this model was interpolated to.
    min_tracking_mac : float
        The lowest tracking MAC among the modes across the interpolated step -- a
        measure of how trustworthy the blend is. A low value means a crossing/veering
        region where the modes may not correspond.
    extrapolated : bool
        ``True`` when ``operating_point`` fell outside the family's range and the model
        is the clamped nearest member (an extrapolation, not an interpolation).
    """

    system: StateSpace
    operating_point: float
    min_tracking_mac: float
    extrapolated: bool


def _bracket(
    points: npt.NDArray[np.float64], value: float
) -> tuple[int, int, float, bool]:
    """Return ``(lo, hi, t, extrapolated)`` bracketing ``value`` in ``points``.

    ``points`` is strictly monotonic (ascending or descending, as the family was built).
    The interpolated quantity is ``(1 - t) * q[lo] + t * q[hi]`` with ``lo``/``hi`` the
    original family indices. Outside the range, ``lo == hi`` is the clamped nearest end
    and ``extrapolated`` is ``True``.
    """
    n = points.shape[0]
    ascending = bool(points[0] <= points[-1])
    ordered = points if ascending else points[::-1]

    def to_original(i: int) -> int:
        return i if ascending else n - 1 - i

    if value <= ordered[0]:
        return to_original(0), to_original(0), 0.0, bool(value < ordered[0])
    if value >= ordered[-1]:
        return to_original(n - 1), to_original(n - 1), 0.0, bool(value > ordered[-1])
    hi_ord = int(np.searchsorted(ordered, value, side="left"))
    lo_ord = hi_ord - 1
    span = float(ordered[hi_ord] - ordered[lo_ord])
    t = (value - float(ordered[lo_ord])) / span
    return to_original(lo_ord), to_original(hi_ord), t, False


def _blend(
    lower: npt.NDArray[np.float64] | None,
    upper: npt.NDArray[np.float64] | None,
    t: float,
) -> npt.NDArray[np.float64] | None:
    """Linearly interpolate two like-shaped matrices.

    Returns ``None`` only when *both* are absent. A family may mix optional-matrix
    presence (e.g. a ``D`` at one member but not its neighbour, allowed when ``B`` fixes
    the input dimension); an absent matrix is the zero matrix (no feedthrough/input
    coupling), so it is treated as zeros rather than dropping the present neighbour's
    contribution -- which would otherwise lose ``D``/``B`` even at that member's exact
    operating point (``t`` of 0 or 1).
    """
    if lower is None and upper is None:
        return None
    reference = lower if lower is not None else upper
    assert reference is not None  # at least one operand is present
    zeros = np.zeros_like(reference)
    low = zeros if lower is None else lower
    high = zeros if upper is None else upper
    blended: npt.NDArray[np.float64] = (1.0 - t) * low + t * high
    return blended


def _schedule_core(family: RomFamily, operating_point: float) -> ScheduledModel:
    """Interpolate the family to ``operating_point`` without emitting warnings.

    The shared worker for :func:`schedule_model` (which adds logging) and the runtime
    loop (which aggregates and warns once). Validates the family and operating point.
    """
    points = np.asarray(family.operating_points, dtype=np.float64)
    if points.size == 0:
        msg = "family has no operating points"
        raise ValueError(msg)
    if not np.isfinite(operating_point):
        msg = f"operating_point must be finite, got {operating_point}"
        raise ValueError(msg)
    lo, hi, t, extrapolated = _bracket(points, float(operating_point))
    low_system = family.systems[lo]
    high_system = family.systems[hi]
    system = StateSpace(
        a=_require(_blend(low_system.a, high_system.a, t)),
        b=_blend(low_system.b, high_system.b, t),
        c=_blend(low_system.c, high_system.c, t),
        d=_blend(low_system.d, high_system.d, t),
        state_names=low_system.state_names,
        input_names=low_system.input_names,
        output_names=low_system.output_names,
        dt=low_system.dt,
        output_mbc_coordinates=low_system.output_mbc_coordinates,
    )
    # The tracking MAC at build index ``k`` is mode k's correlation against the previous
    # build point. On an exact tabulated point (``t`` resolves to 0 or 1, including a
    # clamped end) the confidence is that selected member's own row; for a genuine
    # interpolation it is the higher *build* index of the bracketing pair (the
    # transition between them). Using ``max(lo, hi)`` unconditionally would, on a
    # descending sweep landing exactly on an interior point, read the wrong row.
    if t >= 1.0:
        mac_index = hi
    elif t <= 0.0:
        mac_index = lo
    else:
        mac_index = max(lo, hi)
    mac_row = family.tracking_mac[mac_index]
    min_tracking_mac = float(mac_row.min()) if mac_row.size else 1.0
    return ScheduledModel(
        system=system,
        operating_point=float(operating_point),
        min_tracking_mac=min_tracking_mac,
        extrapolated=extrapolated,
    )


def schedule_model(
    family: RomFamily,
    operating_point: float,
    *,
    mac_warn_threshold: float = 0.8,
) -> ScheduledModel:
    """Interpolate a ROM family to a single operating point.

    Brackets ``operating_point`` between two family members and linearly blends their
    ``A``/``B``/``C``/``D`` matrices (clamping to the nearest member outside the range).
    Because the family is mode-tracked and phase-aligned, the blended model is a valid
    state-space realization in the same state basis. The returned
    :attr:`ScheduledModel.min_tracking_mac` reports the lowest tracking MAC across the
    interpolated step; a value below ``mac_warn_threshold`` (or an extrapolation) logs a
    warning, because there the interpolation may mix modes that do not correspond.

    Parameters
    ----------
    family : RomFamily
        The mode-tracked operating-point family from
        :func:`~vane.sysid.build_rom_family`.
    operating_point : float
        The scheduling-parameter value to interpolate to.
    mac_warn_threshold : float, optional
        Tracking-MAC level below which a low-confidence warning is logged (default
        ``0.8``).

    Returns
    -------
    ScheduledModel
        The interpolated model and its confidence flags.

    Raises
    ------
    ValueError
        If the family is empty, or ``operating_point`` / ``mac_warn_threshold`` is not
        finite.
    """
    if not np.isfinite(mac_warn_threshold):
        msg = f"mac_warn_threshold must be finite, got {mac_warn_threshold}"
        raise ValueError(msg)
    scheduled = _schedule_core(family, operating_point)
    if scheduled.extrapolated:
        points = np.asarray(family.operating_points, dtype=np.float64)
        _log.warning(
            "operating_point %g is outside the family range [%g, %g]; using the "
            "clamped nearest model (extrapolation)",
            operating_point,
            float(points.min()),
            float(points.max()),
        )
    if scheduled.min_tracking_mac < mac_warn_threshold:
        _log.warning(
            "scheduled model at operating_point %g has a low tracking MAC (%.3f < "
            "%.3f): the interpolation crosses a region where modes may not correspond",
            operating_point,
            scheduled.min_tracking_mac,
            mac_warn_threshold,
        )
    return scheduled


def _require(matrix: npt.NDArray[np.float64] | None) -> npt.NDArray[np.float64]:
    """Return the matrix, asserting it is present (the ``A`` blend is never None)."""
    if matrix is None:  # pragma: no cover - A is always present in a valid family
        msg = "interpolated A matrix is unexpectedly None"
        raise ValueError(msg)
    return matrix


@dataclass
class ScheduledFilterResult:
    """The trajectory of a scheduled (LPV) Kalman run, with per-step confidence.

    Mirrors :class:`~vane.sysid.FilterResult` and adds the scheduling diagnostics.

    Parameters
    ----------
    states, covariances : npt.NDArray[np.float64]
        Filtered state means ``(n_steps, n_states)`` and covariances
        ``(n_steps, n_states, n_states)``.
    predicted_states, predicted_covariances : npt.NDArray[np.float64]
        A-priori means/covariances before each update.
    nis : npt.NDArray[np.float64]
        Normalized innovation squared per step (``nan`` where no channel was available).
    accepted : npt.NDArray[np.bool_]
        Whether each step's measurement was applied.
    n_active : npt.NDArray[np.int_]
        Number of channels used at each accepted step.
    min_tracking_mac : npt.NDArray[np.float64]
        The scheduled model's lowest tracking MAC at each step (per-step confidence).
    extrapolated : npt.NDArray[np.bool_]
        Whether each step's operating point fell outside the family range.
    state_names : list[str]
        Names of the states.
    """

    states: npt.NDArray[np.float64]
    covariances: npt.NDArray[np.float64]
    predicted_states: npt.NDArray[np.float64]
    predicted_covariances: npt.NDArray[np.float64]
    nis: npt.NDArray[np.float64]
    accepted: npt.NDArray[np.bool_]
    n_active: npt.NDArray[np.int_]
    min_tracking_mac: npt.NDArray[np.float64]
    extrapolated: npt.NDArray[np.bool_]
    state_names: list[str] = field(default_factory=list)


def run_scheduled_filter(
    family: RomFamily,
    covariances: CovarianceSet,
    measurements: npt.NDArray[np.float64],
    operating_points: npt.NDArray[np.float64],
    *,
    dt: float,
    controls: npt.NDArray[np.float64] | None = None,
    available: npt.NDArray[np.bool_] | None = None,
    gate: float | None = None,
    gate_probability: float | None = None,
    initial_state: npt.NDArray[np.float64] | None = None,
    initial_control: npt.NDArray[np.float64] | None = None,
    mac_warn_threshold: float = 0.8,
) -> ScheduledFilterResult:
    """Run a Kalman filter whose model is scheduled on a drifting operating point.

    At each step the family is interpolated to that step's ``operating_points`` value
    (:func:`schedule_model`), discretized with ``dt``, and used for one predict/update;
    the state and covariance carry across steps in the family's consistent,
    phase-aligned basis, so a moving operating point needs no discontinuous re-basing.
    Control timing is zero-order hold, as in :func:`~vane.sysid.run_filter`: the
    prediction into step ``k`` is driven by ``controls[k-1]`` (the first by
    ``initial_control``, defaulting to ``controls[0]``), and ``controls[k]`` enters
    only the feedthrough of the measurement logged with it.
    The filter tuning ``covariances`` (``Q``, ``R``, ``P0``) is held constant across the
    sweep -- only the plant model is scheduled.

    Parameters
    ----------
    family : RomFamily
        A **continuous-time** mode-tracked family (members are discretized per step).
    covariances : CovarianceSet
        The discrete process/measurement/initial covariances (``Q``, ``R``, ``P0``).
    measurements : npt.NDArray[np.float64]
        Measured outputs, shape ``(n_steps, n_outputs)``.
    operating_points : npt.NDArray[np.float64]
        The scheduling-parameter value at each step, shape ``(n_steps,)`` -- the
        time-varying operating condition (need not be monotonic).
    dt : float
        Discretization step for the scheduled model.
    controls : npt.NDArray[np.float64] or None, optional
        Control inputs, shape ``(n_steps, n_inputs)``; row ``k`` is the input logged
        with ``measurements[k]``. ``None`` for an autonomous model.
    available : npt.NDArray[np.bool_] or None, optional
        Per-step channel mask (sensor dropout), shape ``(n_steps, n_outputs)``.
    gate : float or None, optional
        Fixed chi-square NIS threshold for per-step outlier rejection.
    gate_probability : float or None, optional
        Gate probability ``p``; each step is gated at ``chi2.ppf(p, n_used)`` for the
        channels that reported. Mutually exclusive with ``gate``.
    initial_state : npt.NDArray[np.float64] or None, optional
        Initial state mean; defaults to zeros.
    initial_control : npt.NDArray[np.float64] or None, optional
        The input held before ``measurements[0]`` (``u_{-1}``), which drives the first
        prediction; defaults to ``controls[0]``. Requires ``controls``.
    mac_warn_threshold : float, optional
        Tracking-MAC level below which a step is counted low-confidence (default 0.8).

    Returns
    -------
    ScheduledFilterResult
        The filtered trajectory plus the per-step scheduling confidence.

    Raises
    ------
    ValueError
        If the family is empty or discrete, ``dt`` is not positive, ``measurements`` is
        not ``(n_steps, n_outputs)``, or ``operating_points`` / ``controls`` /
        ``available`` lengths disagree with the step count, ``initial_control`` is
        given without ``controls``, or both ``gate`` and ``gate_probability`` are set.
    """
    if initial_control is not None and controls is None:
        msg = "initial_control requires controls (it is the input before controls[0])"
        raise ValueError(msg)
    if gate is not None and gate_probability is not None:
        msg = "give either gate (a fixed NIS threshold) or gate_probability, not both"
        raise ValueError(msg)
    if not family.systems or np.asarray(family.operating_points).size == 0:
        msg = "family has no operating points"
        raise ValueError(msg)
    if family.systems[0].is_discrete:
        msg = (
            "run_scheduled_filter needs a continuous-time family (its members are "
            "discretized per step)"
        )
        raise ValueError(msg)
    if not np.isfinite(dt) or dt <= 0.0:
        msg = f"dt must be a positive finite step, got {dt}"
        raise ValueError(msg)
    if not np.isfinite(mac_warn_threshold):
        msg = f"mac_warn_threshold must be finite, got {mac_warn_threshold}"
        raise ValueError(msg)
    z = np.asarray(measurements, dtype=np.float64)
    reference_c = family.systems[0].c
    if reference_c is None:  # pragma: no cover - build_rom_family guarantees a C matrix
        msg = "family systems must have a C matrix"
        raise ValueError(msg)
    n_outputs = reference_c.shape[0]
    if z.ndim != 2 or z.shape[1] != n_outputs:
        msg = f"measurements must be (n_steps, {n_outputs}), got {z.shape}"
        raise ValueError(msg)
    n_steps = z.shape[0]
    ops = np.asarray(operating_points, dtype=np.float64).ravel()
    if ops.shape != (n_steps,):
        msg = f"operating_points must have {n_steps} values, got {ops.shape}"
        raise ValueError(msg)
    if controls is not None and len(controls) != n_steps:
        msg = f"controls must have {n_steps} rows, got {len(controls)}"
        raise ValueError(msg)
    if available is not None and len(available) != n_steps:
        msg = f"available must have {n_steps} rows, got {len(available)}"
        raise ValueError(msg)

    n_states = family.systems[0].a.shape[0]
    if n_steps == 0:
        # An empty window (as batch code may forward after segmentation/dropout) yields
        # an empty result, matching run_filter -- rather than indexing ops[0].
        return ScheduledFilterResult(
            states=np.empty((0, n_states), dtype=np.float64),
            covariances=np.empty((0, n_states, n_states), dtype=np.float64),
            predicted_states=np.empty((0, n_states), dtype=np.float64),
            predicted_covariances=np.empty((0, n_states, n_states), dtype=np.float64),
            nis=np.empty(0, dtype=np.float64),
            accepted=np.empty(0, dtype=bool),
            n_active=np.empty(0, dtype=int),
            min_tracking_mac=np.empty(0, dtype=np.float64),
            extrapolated=np.empty(0, dtype=bool),
            state_names=list(family.systems[0].state_names),
        )

    # Build the filter once on the first scheduled model, then re-point its plant
    # matrices each step; x and P carry across in the family's consistent basis.
    first = _schedule_core(family, float(ops[0]))
    model = build_kalman_model(first.system, covariances, dt=dt)
    kf = KalmanFilter(model, initial_state=initial_state)

    states = np.empty((n_steps, n_states), dtype=np.float64)
    covariances_out = np.empty((n_steps, n_states, n_states), dtype=np.float64)
    pred_states = np.empty((n_steps, n_states), dtype=np.float64)
    pred_covariances = np.empty((n_steps, n_states, n_states), dtype=np.float64)
    nis = np.empty(n_steps, dtype=np.float64)
    accepted = np.empty(n_steps, dtype=bool)
    n_active = np.empty(n_steps, dtype=int)
    min_mac = np.empty(n_steps, dtype=np.float64)
    extrapolated = np.empty(n_steps, dtype=bool)

    # ZOH: the prediction into step k is driven by u_{k-1}; the first by u_{-1}.
    held: npt.NDArray[np.float64] | None
    if initial_control is not None:
        held = np.asarray(initial_control, dtype=np.float64)
    elif controls is not None:
        held = controls[0]
    else:
        held = None
    for k in range(n_steps):
        # The core does not warn per step; the run reports the counts once below.
        scheduled = _schedule_core(family, float(ops[k]))
        discrete = scheduled.system.discretized(dt)
        kf.a = np.asarray(discrete.a, dtype=np.float64)
        kf.b = None if discrete.b is None else np.asarray(discrete.b, dtype=np.float64)
        kf.c = np.asarray(_require(discrete.c), dtype=np.float64)
        kf.d = None if discrete.d is None else np.asarray(discrete.d, dtype=np.float64)
        control_k = None if controls is None else controls[k]
        kf.predict(held)
        pred_states[k] = kf.x
        pred_covariances[k] = kf.p
        step_nis, step_ok, step_dof = kf.update(
            z[k],
            control=control_k,
            available=None if available is None else available[k],
            gate=gate,
            gate_probability=gate_probability,
        )
        held = control_k
        states[k] = kf.x
        covariances_out[k] = kf.p
        nis[k] = step_nis
        accepted[k] = step_ok
        n_active[k] = step_dof
        min_mac[k] = scheduled.min_tracking_mac
        extrapolated[k] = scheduled.extrapolated

    low_conf = int(np.sum(min_mac < mac_warn_threshold))
    n_extrapolated = int(np.sum(extrapolated))
    if low_conf or n_extrapolated:
        _log.warning(
            "scheduled run: %d/%d steps below the tracking-MAC threshold %.3f, %d "
            "step(s) extrapolated beyond the family range",
            low_conf,
            n_steps,
            mac_warn_threshold,
            n_extrapolated,
        )

    return ScheduledFilterResult(
        states=states,
        covariances=covariances_out,
        predicted_states=pred_states,
        predicted_covariances=pred_covariances,
        nis=nis,
        accepted=accepted,
        n_active=n_active,
        min_tracking_mac=min_mac,
        extrapolated=extrapolated,
        state_names=list(kf.state_names),
    )
