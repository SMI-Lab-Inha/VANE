# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Proactive per-operating-point trim auto-tuning for the linearization sweep.

The ``CalcSteady`` trim that finds a turbine's periodic steady state before linearizing
is hardest at and above rated wind speed: thrust is high and the platform/tower response
is lightly damped, so a tolerance and trim damping that converge an easy below-rated
point often stall a near-/above-rated one. :class:`~vane.run.LinearizationRetryPolicy`
recovers from that *reactively* — it adapts the tolerance, ``TMax``, trim gains and
damping to the classified failure after a run has already failed, costing one or more
wasted OpenFAST runs per hard point.

This module sets those same knobs *proactively*. Given an operating point and the
campaign-base :class:`~vane.run.LinearizationSettings`, :func:`tune_trim_settings`
returns the settings to start that point from, applying the documented above-rated trim
floors — a looser ``TrimTol`` and a trim-damping floor on ``Twr_Kdmp``/``Bld_Kdmp`` — to
points at or above rated so they are likely to converge on the first attempt. It only
ever *raises* tolerance/damping toward those floors, never tightens below the campaign
base, and below-rated and standstill points are returned unchanged, so the tuner can
only help. The reactive retry ladder still escalates from the tuned starting point for
any residual hard case.

The tuner adjusts only the three fields the retry ladder already treats as per-case
(``trim_tol``, ``twr_kdmp``, ``bld_kdmp``), so a sweep's run provenance — which records
those per case and compares the rest campaign-wide — is unaffected. The
below/above-rated boundary is the settings' own ``rated_wind_speed``, the same split
:func:`~vane.run.configure_linearization` uses to choose torque vs pitch trim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from vane.run.lin_config import MAX_TRIM_TOL

if TYPE_CHECKING:
    from vane.run.lin_config import LinearizationSettings
    from vane.run.operating_points import OperatingPoint

__all__ = ["TrimTuningPolicy", "tune_trim_settings"]


@dataclass
class TrimTuningPolicy:
    """Floors applied to the trim settings of an at/above-rated operating point.

    Parameters
    ----------
    above_rated_trim_tol : float
        Minimum rotor-speed trim tolerance (``TrimTol``) to use at or above rated wind
        speed. A point's tolerance is raised to this floor (never lowered below the
        campaign base); below-rated points keep the base tolerance. The default ``1e-2``
        is the loosened tolerance the above-rated trim needs (the base default is
        ``1e-3``). It must not exceed ``MAX_TRIM_TOL`` (``0.1``), and the retry ladder
        that escalates from it clamps there too.
    above_rated_damping : float
        Minimum artificial trim damping (``Twr_Kdmp`` and ``Bld_Kdmp``) to use at or
        above rated wind speed; the point's damping is raised to this floor. The damping
        acts only during the steady-state trim and is stripped from the linearized
        model, but it can bias the trimmed operating point slightly, so use the smallest
        floor that converges. The
        default ``5000`` matches the offshore preset's trim damping.
    """

    above_rated_trim_tol: float = 1e-2
    above_rated_damping: float = 5000.0

    def __post_init__(self) -> None:
        """Reject a non-finite or non-positive floor.

        Raises
        ------
        ValueError
            If ``above_rated_trim_tol`` or ``above_rated_damping`` is not finite and
            positive, or ``above_rated_trim_tol`` exceeds ``MAX_TRIM_TOL``.
        """
        for name, value in (
            ("above_rated_trim_tol", self.above_rated_trim_tol),
            ("above_rated_damping", self.above_rated_damping),
        ):
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0.0
            ):
                msg = f"{name} must be finite and positive, got {value!r}"
                raise ValueError(msg)
        if self.above_rated_trim_tol > MAX_TRIM_TOL:
            msg = (
                f"above_rated_trim_tol must not exceed {MAX_TRIM_TOL}, got "
                f"{self.above_rated_trim_tol!r}"
            )
            raise ValueError(msg)


def tune_trim_settings(
    operating_point: OperatingPoint,
    base: LinearizationSettings,
    policy: TrimTuningPolicy,
) -> LinearizationSettings:
    """Return the trim settings to start one operating point from.

    For a rotating point that trims (``CalcSteady``) at or above the base settings'
    ``rated_wind_speed``, the trim tolerance and tower/blade trim damping are raised to
    the policy's floors (whichever is larger — the campaign base is never tightened). A
    below-rated point, or a standstill/parked point that takes a single linearization
    without a trim, is returned unchanged.

    Parameters
    ----------
    operating_point : OperatingPoint
        The operating point whose starting trim settings to compute.
    base : LinearizationSettings
        The campaign-base settings.
    policy : TrimTuningPolicy
        The above-rated trim floors to apply.

    Returns
    -------
    LinearizationSettings
        A copy of ``base`` with the above-rated trim floors applied when they raise the
        knobs, or an unchanged copy otherwise.
    """
    # A point with no periodic trim (standstill/parked, or CalcSteady disabled) takes a
    # single linearization, so there is no trim to tune.
    if not (operating_point.is_rotating and base.calc_steady):
        return replace(base)
    # Below rated the trim is easy (torque trim, lower thrust); keep the base.
    if base.is_below_rated(operating_point.wind_speed):
        return replace(base)
    return replace(
        base,
        trim_tol=max(base.trim_tol, policy.above_rated_trim_tol),
        twr_kdmp=max(base.twr_kdmp, policy.above_rated_damping),
        bld_kdmp=max(base.bld_kdmp, policy.above_rated_damping),
    )
