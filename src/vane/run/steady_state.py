# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Solve a floating turbine's steady operating point by a fully-coupled settle run.

A CalcSteady linearization of a floating turbine needs good initial conditions — in
particular the platform's thrust-loaded equilibrium position — or the open-loop trim
diverges near peak thrust (the rotor decelerates into a stalled, motoring state). The
dependable way to obtain them is to run the *fully-coupled* model with its real
controller at a steady wind until it settles, then read the operating point off the
time series.

:func:`solve_steady_state` does exactly that: it configures the deck for a constant-wind
time-domain run (still water, uniform inflow with no shear by default -- the same
inflow the linearization then uses -- and the controller left active), runs OpenFAST,
discards the initial transient, and averages the steady window to return an
:class:`~vane.run.operating_points.OperatingPoint` carrying the settled rotor speed,
blade pitch, and :class:`~vane.run.operating_points.PlatformPosition`. Feeding that
point back into :func:`~vane.run.configure_linearization` (with
:meth:`LinearizationSettings.offshore`) starts the linearization at the true
equilibrium.

The deck is edited **in place**, so pass a writable working copy (the linearization
driver copies the case per operating point).

The settle run uses exactly the still water the linearization uses: SeaState is set to
``WaveMod = 0`` and the wave options OpenFAST rejects in still water are neutralised by
the same helper :func:`~vane.run.configure_linearization` applies.

Unlike the linearization, the settle run keeps the deck's Bladed-style controller
library (ServoDyn ``DLL_FileName``) active. Published decks often point it at a
library built for another platform (a Linux ``libdiscon.so`` path on Windows), so
the library is checked before OpenFAST launches and a missing one fails fast; pass
``controller_library`` to point the settle copy at a library built for this host.
"""

from __future__ import annotations

import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from vane.io.fst_reader import fortran_float
from vane.io.fst_writer import FastInputError, FastInputFile
from vane.run.lin_config import (
    DLL_CONTROL_KEYS,
    UNIFORM_INFLOW_SHEAR_EXPONENT,
    _deck_float,
    _make_still_water_consistent,
)
from vane.run.operating_points import OperatingPoint, PlatformPosition
from vane.run.runner import _validated_timeout, run_openfast
from vane.run.trim_verification import QuantityCheck, TrimTolerances
from vane.utils.fs_retry import retry_transient
from vane.utils.paths import resolve_path

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

    from vane.run.trim_verification import CheckStatus

__all__ = [
    "ConfirmTimeSource",
    "EquilibriumTolerances",
    "SettleAttempt",
    "SettleConfirmation",
    "SettleConvergence",
    "SettleCriterion",
    "SettleCurveCheck",
    "SettlePlatformDamping",
    "SettleStage",
    "SteadyStateError",
    "check_controller_library",
    "compare_settled_to_curve",
    "solve_steady_state",
    "validate_settle_schedule",
]

logger = logging.getLogger(__name__)

#: How a settle run is judged settled; see :func:`solve_steady_state`. ``"auto"``
#: starts with the velocity criterion and switches to the mean-drift criterion when a
#: sustained, non-decaying oscillation is detected; a convergence record always holds
#: the criterion that actually judged the run (``"velocity"`` or ``"mean_drift"``).
SettleCriterion = Literal["velocity", "mean_drift", "auto"]
#: Which run of a (two-stage) settle an attempt belongs to.
SettleStage = Literal["settle", "damped", "confirm"]
#: How the length of a two-stage settle's confirmation run was chosen.
ConfirmTimeSource = Literal["user", "platform_period", "damped_stage", "fallback"]

# ElastoDyn platform output channel for each PlatformPosition attribute.
_PLATFORM_CHANNELS = {
    "surge": "PtfmSurge",
    "sway": "PtfmSway",
    "heave": "PtfmHeave",
    "roll": "PtfmRoll",
    "pitch": "PtfmPitch",
    "yaw": "PtfmYaw",
}

# Default settle tolerance on the platform-velocity RMS (m/s and deg/s).
_DEFAULT_VELOCITY_TOL = 1e-2

# Automatic length of the undamped confirmation run of a two-stage settle. Its
# analysis window (the last three quarters; the first quarter is the restart
# transient) must hold this many periods of the slowest platform mode: a decaying
# residual's whole-period mean is then taken long after the restart, where its
# envelope is small, and the cycle-by-cycle decay is fitted over enough cycles to
# separate a sustained oscillation from a slowly decaying one. Eight periods of a
# ~100 s surge mode is the 1200 s the validated hand-tuned semi-submersible campaign
# needed; four (the old fixed 600 s) left a partial-cycle and slow-transient bias of
# about 0.1 m surge and 0.1 deg pitch on the IEA 15-MW semi.
_AUTO_CONFIRM_PERIODS = 8
_CONFIRM_ANALYSED_FRACTION = 0.75
# Length used when no platform period is known, and the floor of the automatic
# length (a short dominant period, e.g. a tower or blade mode still visible under the
# settle damping, says nothing about the slow platform modes).
_AUTO_CONFIRM_FALLBACK = 1200.0
# Ceiling of the automatic length when no ``max_settle_time`` bounds it: eight
# periods of a 340 s mode, beyond any moored platform's slowest rigid-body mode.
_AUTO_CONFIRM_CEILING = 3600.0

# ElastoDyn platform-velocity channels. At a true equilibrium these are ~zero, so their
# RMS over the steady window is the physical "has it stopped moving?" criterion.
_TRANSLATIONAL_VELOCITY_CHANNELS = ("PtfmTVxi", "PtfmTVyi", "PtfmTVzi")  # m/s
_ANGULAR_VELOCITY_CHANNELS = ("PtfmRVxi", "PtfmRVyi", "PtfmRVzi")  # deg/s

# ServoDyn control switches whose value 5 routes that control through the Bladed-style
# controller library; any one of them makes ServoDyn load ``DLL_FileName`` at start-up.
_DLL_SWITCH = 5.0

# ElastoDyn channels a settle run reads: the operating point and the equilibrium check.
_SETTLE_CHANNELS = (
    "RotSpeed",
    "BldPitch1",
    *_TRANSLATIONAL_VELOCITY_CHANNELS,
    *_ANGULAR_VELOCITY_CHANNELS,
    # The platform positions: the settled offset carried into the linearization,
    # and the means the mean-drift criterion must see before it can accept a point.
    *_PLATFORM_CHANNELS.values(),
)
# A quoted OutList entry, and the separators between channel names inside one.
_QUOTED = re.compile(r'"([^"]*)"')
_CHANNEL_SEPARATOR = re.compile(r"[\s,;]+")


@dataclass
class SettleAttempt:
    """Convergence evidence from one adaptive settle attempt.

    Each entry preserves the horizon and measured platform-velocity RMS values before
    a later adaptive run overwrites the OpenFAST output file.

    Parameters
    ----------
    translational_velocity_rms : float
        Largest RMS platform translational velocity (m/s) over the analysis window;
        ``nan`` when no translational-velocity channel was output.
    angular_velocity_rms : float
        Largest RMS platform angular velocity (deg/s) over the analysis window; ``nan``
        when no angular-velocity channel was output.
    assessed : bool
        Whether any platform-velocity channel was available to judge settling.
    converged : bool
        Whether every assessed velocity RMS was within tolerance.
    settle_time : float
        The ``TMax`` (s) of this attempt.
    iteration : int
        One-based attempt number.
    window_start, window_end : float or None, optional
        Bounds (s) of the trailing analysis window this attempt was judged over,
        ``[settle_time - (initial settle_time - discard_time), settle_time]``.
        ``None`` when not recorded.
    stage : {"settle", "damped", "confirm"}, optional
        Which run this was: an ordinary settle run (``"settle"``), a run of the
        temporarily damped first stage of a two-stage settle (``"damped"``), or its
        undamped confirmation run (``"confirm"``). See :func:`solve_steady_state`.
    mean_drift : dict[str, float] or None, optional
        Change of each measured operating-point mean between the first and second
        halves of the analysis window (second minus first): ``surge``/``sway``/
        ``heave`` (m), ``roll``/``pitch``/``yaw`` (deg), ``rotor_speed_rpm`` and
        ``blade_pitch_deg``. ``None`` when not computed.
    oscillation_period : float or None, optional
        Dominant period (s) of the platform-velocity channel with the largest RMS over
        the analysis window -- the period of whatever is still moving. ``None`` when
        it could not be estimated (too few samples, or no motion).
    criterion : {"velocity", "mean_drift"} or None, optional
        The criterion that judged this attempt (with ``criterion="auto"`` it may
        differ between attempts). ``None`` when not recorded.
    residual_damping_ratio : float or None, optional
        Damping ratio of the platform oscillation over the analysis window, from the
        cycle-by-cycle decay of the most active platform-velocity channel (positive:
        decaying; near zero: sustained; negative: growing). ``None`` when fewer than
        two whole cycles were output.
    """

    translational_velocity_rms: float
    angular_velocity_rms: float
    assessed: bool
    converged: bool
    settle_time: float
    iteration: int
    window_start: float | None = None
    window_end: float | None = None
    stage: SettleStage = "settle"
    mean_drift: dict[str, float] | None = None
    oscillation_period: float | None = None
    criterion: SettleCriterion | None = None
    residual_damping_ratio: float | None = None


@dataclass
class SettleConvergence:
    """Whether a settle run reached equilibrium, judged by platform-velocity RMS.

    Parameters
    ----------
    translational_velocity_rms : float
        Largest root-mean-square platform translational velocity (m/s) over the
        trailing analysis window, across the output velocity channels; ``nan`` when
        none were output.
    angular_velocity_rms : float
        Largest RMS platform angular velocity (deg/s) over the trailing analysis
        window; ``nan`` when no angular-velocity channel was output.
    assessed : bool
        Whether any platform-velocity channel was available to judge settling. When
        ``False`` the run cannot be confirmed settled (the deck output no velocity).
    converged : bool
        Whether every assessed velocity RMS is within tolerance (``True`` when not
        assessed — nothing measured cannot fail, but ``assessed`` records that).
    settle_time : float
        The ``TMax`` (s) of the run this convergence describes.
    iterations : int
        How many settle runs were performed (more than one only when adaptive).
    attempts : tuple of SettleAttempt
        Ordered convergence evidence for every attempted horizon. This makes an
        adaptive accept/reject decision reconstructible after intermediate OpenFAST
        output files have been overwritten.
    window_start, window_end : float or None, optional
        Bounds (s) of the trailing analysis window of the final run, over which both
        the convergence outcome and the returned operating-point means were computed.
        ``None`` when not recorded.
    criterion : {"velocity", "mean_drift"}, optional
        The settle criterion that judged the run: platform-velocity RMS within
        tolerance (``"velocity"``, the default) or operating-point means that no
        longer drift between the two halves of the analysis window
        (``"mean_drift"``). With ``criterion="auto"`` this is the criterion the
        automatic choice settled on. See :func:`solve_steady_state`.
    mean_drift : dict[str, float] or None, optional
        The final run's half-window mean drift (see :attr:`SettleAttempt.mean_drift`).
    oscillation_period : float or None, optional
        The final run's dominant platform-velocity period (s); see
        :attr:`SettleAttempt.oscillation_period`.
    confirmation : SettleConfirmation or None, optional
        For a two-stage settle (``platform_damping`` given), the damped stage's
        evidence and the undamped confirmation's outcome; ``None`` for an ordinary
        settle.
    requested_criterion : {"velocity", "mean_drift", "auto"}, optional
        The criterion the caller asked for. Differs from :attr:`criterion` only for
        ``"auto"``.
    criterion_switched : bool, optional
        ``True`` when ``criterion="auto"`` switched from the velocity to the
        mean-drift criterion because the platform carried a sustained, non-decaying
        oscillation the velocity criterion can never accept.
    prescribed_rotor : bool, optional
        ``True`` when the settle ran without a controller, with the rotor speed and
        blade pitch held at their initial values (see ``prescribed_rotor`` of
        :func:`solve_steady_state`); the settled rotor speed and pitch are then the
        prescribed ones and only the platform equilibrium was found.
    residual_damping_ratio : float or None, optional
        The final run's platform-oscillation damping ratio (see
        :attr:`SettleAttempt.residual_damping_ratio`).
    sustained_oscillation : bool, optional
        ``True`` when a two-stage settle was accepted with a non-decaying (sustained
        or growing) oscillation about its confirmed equilibrium (see
        :attr:`SettleConfirmation.sustained_oscillation`); a warning is logged.
    """

    translational_velocity_rms: float
    angular_velocity_rms: float
    assessed: bool
    converged: bool
    settle_time: float
    iterations: int
    attempts: tuple[SettleAttempt, ...] = ()
    window_start: float | None = None
    window_end: float | None = None
    criterion: SettleCriterion = "velocity"
    mean_drift: dict[str, float] | None = None
    oscillation_period: float | None = None
    confirmation: SettleConfirmation | None = None
    requested_criterion: SettleCriterion = "velocity"
    criterion_switched: bool = False
    prescribed_rotor: bool = False
    residual_damping_ratio: float | None = None
    sustained_oscillation: bool = False

    @property
    def growing_oscillation(self) -> bool:
        """Return whether the point was accepted with a growing oscillation.

        Returns
        -------
        bool
            :attr:`SettleConfirmation.growing_oscillation` of the confirmation;
            ``False`` for a single-stage settle.
        """
        return self.confirmation is not None and self.confirmation.growing_oscillation

    def to_dict(self) -> dict[str, object]:
        """Return the record as a dictionary, with ``growing_oscillation`` added.

        Returns
        -------
        dict[str, object]
            Every field (as :func:`dataclasses.asdict`), with ``confirmation`` from
            :meth:`SettleConfirmation.to_dict`, plus ``growing_oscillation``.
        """
        record = asdict(self)
        if self.confirmation is not None:
            record["confirmation"] = self.confirmation.to_dict()
        record["growing_oscillation"] = self.growing_oscillation
        return record


# The six platform DOFs, in HydroDyn's AddBLin row/column order.
_PLATFORM_DOFS = ("surge", "sway", "heave", "roll", "pitch", "yaw")
_TRANSLATION_DOFS = ("surge", "sway", "heave")
_ROTATION_DOFS = ("roll", "pitch", "yaw")


@dataclass(frozen=True)
class SettlePlatformDamping:
    """Temporary linear platform damping that speeds up a floating settle run.

    The damping is added to the diagonal of HydroDyn's ``AddBLin`` matrix (first
    body) in the **settle copy only**, for the first stage of a two-stage settle (see
    :func:`solve_steady_state`); it is removed again before the undamped
    confirmation run and never reaches a linearization case.

    Why the settled mean is unchanged: the added load is ``F = -B q_dot``. At a
    static equilibrium the platform velocity ``q_dot`` is zero, so the force is zero
    and the equilibrium equations -- rotor thrust against mooring, hydrostatic and
    gravity restoring loads -- are exactly those of the undamped model: the fixed
    point does not move. Over a stationary periodic response the mean of ``B q_dot``
    is ``B (q(t+T) - q(t)) / T = 0`` as well. Quadratic drag (``AddBQuad``, Morison
    drag) also vanishes at zero velocity, and the settle runs in still water, so
    there is no wave mean-drift force to change. Only the *transient* -- the slow,
    lightly damped surge/sway/yaw ringing -- decays faster. Caveats the undamped
    confirmation stage exists for: a model with more than one equilibrium (e.g. a
    controller region boundary) may be steered to a different one, and an
    oscillation the damping suppresses may be self-excited, in which case the
    undamped model has no stable equilibrium and its oscillation reappears.

    Parameters
    ----------
    surge, sway, heave : float, optional
        Added linear damping for the translational DOFs, N/(m/s). Default ``0``.
    roll, pitch, yaw : float, optional
        Added linear damping for the rotational DOFs, N m/(rad/s). Default ``0``.

    Raises
    ------
    ValueError
        If any value is negative or non-finite, or if all are zero.

    See Also
    --------
    SettlePlatformDamping.from_critical_fraction : size it as a fraction of critical.
    """

    surge: float = 0.0
    sway: float = 0.0
    heave: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def __post_init__(self) -> None:
        """Reject a negative, non-finite, or all-zero damping."""
        for name in _PLATFORM_DOFS:
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                msg = f"platform damping {name} must be finite and >= 0, got {value}"
                raise ValueError(msg)
        if not any(getattr(self, name) > 0.0 for name in _PLATFORM_DOFS):
            msg = "platform damping must be positive for at least one DOF"
            raise ValueError(msg)

    @classmethod
    def from_critical_fraction(
        cls,
        fraction: float | Mapping[str, float],
        *,
        mass: Mapping[str, float],
        natural_period: Mapping[str, float],
    ) -> SettlePlatformDamping:
        """Size the damping as a fraction of critical damping per DOF.

        For a single DOF with effective mass (or inertia) ``m`` and stiffness ``k``,
        critical damping is ``2 sqrt(k m) = 2 m omega_n = 4 pi m / T_n``, so the
        damping for a fraction ``zeta`` is ``4 pi zeta m / T_n``. The natural period
        stands in for the stiffness because it is what is usually known (published
        for reference platforms, or read off an earlier settle as its
        :attr:`SettleConvergence.oscillation_period`); the mooring stiffness alone is
        not. Precision is not needed: any positive damping leaves the equilibrium
        unchanged, and a fraction of about 0.3--0.7 decays the transient in one to
        two cycles while staying clear of the slow creep of an overdamped DOF.

        Parameters
        ----------
        fraction : float or Mapping[str, float]
            Fraction of critical damping, the same for every DOF given in ``mass``
            or one per DOF.
        mass : Mapping[str, float]
            Effective mass (kg) for ``surge``/``sway``/``heave`` or inertia
            (kg m^2) for ``roll``/``pitch``/``yaw``, *including* the hydrodynamic
            added mass, for each DOF to damp. DOFs absent from it are not damped.
        natural_period : Mapping[str, float]
            Natural period (s) of each DOF in ``mass``.

        Returns
        -------
        SettlePlatformDamping
            The absolute damping per DOF.

        Raises
        ------
        ValueError
            If a DOF name is unknown, a DOF in ``mass`` has no natural period (or
            fraction), or a mass, period or fraction is not positive and finite.

        Examples
        --------
        IEA 15-MW on the VolturnUS-S semi-submersible: surge mass about
        ``3.3e7`` kg (2.0e7 kg structure plus 1.3e7 kg added mass) with a surge period
        near 100 s under thrust; at half of critical damping

        >>> damping = SettlePlatformDamping.from_critical_fraction(
        ...     0.5, mass={"surge": 3.3e7}, natural_period={"surge": 100.0}
        ... )
        >>> round(damping.surge / 1e6, 2)
        2.07
        """
        values: dict[str, float] = {}
        for dof, dof_mass in mass.items():
            if dof not in _PLATFORM_DOFS:
                msg = f"unknown platform DOF {dof!r}; expected one of {_PLATFORM_DOFS}"
                raise ValueError(msg)
            if dof not in natural_period:
                msg = f"no natural_period given for platform DOF {dof!r}"
                raise ValueError(msg)
            if isinstance(fraction, Mapping):
                if dof not in fraction:
                    msg = f"no critical-damping fraction given for {dof!r}"
                    raise ValueError(msg)
                zeta = float(fraction[dof])
            else:
                zeta = float(fraction)
            period = float(natural_period[dof])
            for label, value in (
                ("mass", float(dof_mass)),
                ("natural_period", period),
                ("fraction", zeta),
            ):
                if not np.isfinite(value) or value <= 0.0:
                    msg = (
                        f"{label} for {dof!r} must be positive and finite, got {value}"
                    )
                    raise ValueError(msg)
            values[dof] = 4.0 * math.pi * zeta * float(dof_mass) / period
        return cls(**values)

    def diagonal(self) -> tuple[float, float, float, float, float, float]:
        """Return the damping in HydroDyn DOF order (surge ... yaw).

        Returns
        -------
        tuple of float
            ``(surge, sway, heave, roll, pitch, yaw)``.
        """
        return (self.surge, self.sway, self.heave, self.roll, self.pitch, self.yaw)


@dataclass(frozen=True)
class EquilibriumTolerances:
    """Tolerances on a change of the settled operating-point means.

    Used to judge whether the undamped confirmation of a two-stage settle reproduces
    the damped stage's equilibrium, and -- with ``criterion="mean_drift"`` -- whether
    a run's means still drift between the halves of its analysis window.

    Parameters
    ----------
    translation_m : float, optional
        Platform surge/sway/heave, m. Default ``0.1``.
    rotation_deg : float, optional
        Platform roll/pitch/yaw, deg. Default ``0.05``.
    rotor_speed_rel : float, optional
        Rotor speed, relative to ``max(|reference rotor speed|, 1 rpm)``. Default
        ``0.005``.
    pitch_deg : float, optional
        Collective blade pitch, deg. Default ``0.1``.
    residual_damping_ratio : float, optional
        Smallest damping ratio at which the residual platform oscillation of an
        undamped confirmation counts as *decaying* (see
        :attr:`SettleConfirmation.residual_damping_ratio`). Default ``0.005``.
    growing_damping_ratio : float, optional
        Classification threshold: an undamped confirmation whose residual damping
        ratio is below ``-growing_damping_ratio`` is *clearly growing* (its
        equilibrium is unstable under the deck's controller), not merely sustained.
        Default ``0.02``: the amplitude then grows by ``exp(2 pi 0.02) - 1``, about
        13 %, per cycle and by a factor of about 2.7 over the eight cycles an
        automatically sized confirmation analyses -- unmistakable growth, whereas a
        limit cycle, a beating of two slow modes or a controller re-exciting the
        platform after the restart stays above it (the IEA 15-MW semi at
        12-14 m/s gives -0.007 to -0.009). It does not reject anything: with the
        ``"auto"`` criterion a growing oscillation about a confirmed equilibrium is
        accepted like a sustained one, but flagged
        (:attr:`SettleConfirmation.growing_oscillation`) and logged with its own
        warning (see :func:`solve_steady_state`).
    oscillation_velocity_rms : float, optional
        Largest platform-velocity RMS (m/s for the translational channels, deg/s
        for the angular ones) of a sustained or growing oscillation the ``"auto"``
        criterion accepts about a confirmed equilibrium. Default ``0.2``, twenty
        times the default settle tolerance: at a 100 s surge period about +/-4.5 m
        of surge, at a 30 s pitch period about +/-1.4 deg of pitch -- small enough
        for its whole-period means to be a well-defined measurement of the
        equilibrium. A larger one is reported, not accepted.

    Raises
    ------
    ValueError
        If any tolerance is not positive and finite.
    """

    translation_m: float = 0.1
    rotation_deg: float = 0.05
    rotor_speed_rel: float = 0.005
    pitch_deg: float = 0.1
    residual_damping_ratio: float = 0.005
    growing_damping_ratio: float = 0.02
    oscillation_velocity_rms: float = 0.2

    def __post_init__(self) -> None:
        """Reject a non-positive or non-finite tolerance."""
        for name in (
            "translation_m",
            "rotation_deg",
            "rotor_speed_rel",
            "pitch_deg",
            "residual_damping_ratio",
            "growing_damping_ratio",
            "oscillation_velocity_rms",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                msg = f"{name} must be positive and finite, got {value}"
                raise ValueError(msg)

    def exceeded(
        self, changes: Mapping[str, float], reference_rotor_speed_rpm: float
    ) -> tuple[str, ...]:
        """Return the quantities whose change exceeds its tolerance.

        Parameters
        ----------
        changes : Mapping[str, float]
            Change of each mean, keyed as :attr:`SettleAttempt.mean_drift`.
        reference_rotor_speed_rpm : float
            The rotor speed the relative rotor-speed tolerance refers to.

        Returns
        -------
        tuple of str
            The names (in the order of ``changes``) outside tolerance.
        """
        limits: dict[str, float] = dict.fromkeys(_TRANSLATION_DOFS, self.translation_m)
        limits.update(dict.fromkeys(_ROTATION_DOFS, self.rotation_deg))
        limits["rotor_speed_rpm"] = self.rotor_speed_rel * max(
            abs(reference_rotor_speed_rpm), 1.0
        )
        limits["blade_pitch_deg"] = self.pitch_deg
        return tuple(
            name
            for name, change in changes.items()
            if name in limits and not abs(change) <= limits[name]
        )


@dataclass
class SettleConfirmation:
    """Evidence of a two-stage settle: damped stage, then undamped confirmation.

    Parameters
    ----------
    damping : SettlePlatformDamping
        The temporary platform damping of the first stage.
    damped_converged : bool
        Whether the damped stage met the settle criterion.
    damped_translational_velocity_rms, damped_angular_velocity_rms : float
        The damped stage's final platform-velocity RMS (m/s, deg/s; ``nan`` when not
        measured).
    damped_settle_time : float
        ``TMax`` (s) of the damped stage's final run.
    damped_means : dict[str, float]
        The damped stage's settled means (keys as :attr:`SettleAttempt.mean_drift`);
        the undamped confirmation starts from them.
    confirm_means : dict[str, float]
        The undamped confirmation run's means over its analysis window (averaged
        over whole periods of any residual oscillation).
    shift : dict[str, float]
        ``confirm_means - damped_means`` for every quantity measured in both.
    shift_flagged : tuple of str
        Quantities whose shift exceeds ``tolerances``.
    confirm_converged : bool
        Whether the undamped confirmation run met the settle criterion by itself.
    residual_damping_ratio : float or None
        Damping ratio of the undamped run's residual platform oscillation, from the
        decay of its cycle-by-cycle RMS: positive when it dies out, zero or negative
        when it is sustained or growing. ``None`` when fewer than two whole cycles
        were output.
    stable : bool
        Whether the undamped run is stable about the equilibrium: it met the
        criterion by itself, or (velocity criterion) its residual oscillation
        decays with a damping ratio of at least ``tolerances.residual_damping_ratio``.
    confirmed : bool
        ``True`` when the undamped run reproduced the damped equilibrium within
        ``tolerances`` *and* is ``stable``, or -- ``sustained_oscillation`` -- when
        it reproduced it while carrying a measurable sustained or growing
        oscillation (``"auto"``). Restarting an undamped model, even at its
        equilibrium, rings its slow modes (the controller's start-up transient
        alone does), and a lightly damped mode cannot ring down within a short
        confirmation; a decaying residual is accepted as confirmation of a stable
        equilibrium. ``False`` means the damping changed the answer, or the
        undamped model's oscillation could not be accepted (too large to measure
        its mean, too few cycles, or a criterion other than ``"auto"``).
    tolerances : EquilibriumTolerances
        The tolerances applied.
    confirm_time : float or None, optional
        ``TMax`` (s) of the confirmation run that was judged (the extended one when
        ``extended``). ``None`` when not recorded.
    confirm_time_source : {"user", "platform_period", "damped_stage", "fallback"} \
or None, optional
        How the first confirmation's length was chosen: given by the caller, or
        sized automatically from the slowest platform period (``platform_period``
        of :func:`solve_steady_state`, e.g. the standstill platform probe's), from
        the damped stage's dominant period, or the fallback length when neither
        was known. ``None`` when not recorded.
    extended : bool, optional
        ``True`` when the confirmation was decaying but its means had not yet
        returned to the damped equilibrium, so it was rerun once twice as long and
        that longer run was judged.
    sustained_oscillation : bool, optional
        ``True`` when the point was accepted although the undamped run keeps a
        non-decaying oscillation: its whole-period means reproduce the damped
        equilibrium within ``tolerances``, and either the mean-drift criterion
        accepted the run or -- ``"auto"`` criterion only -- the oscillation is
        measurable (residual damping ratio known, velocity RMS within
        ``tolerances.oscillation_velocity_rms``), whether sustained or growing.
        Linear damping cannot move an equilibrium (it acts on the velocity, zero
        there), so the damped stage's equilibrium is the undamped model's, stable
        or not; it is the operating point. A growing oscillation is additionally
        flagged by :attr:`growing_oscillation`.
    """

    damping: SettlePlatformDamping
    damped_converged: bool
    damped_translational_velocity_rms: float
    damped_angular_velocity_rms: float
    damped_settle_time: float
    damped_means: dict[str, float]
    confirm_means: dict[str, float]
    shift: dict[str, float]
    shift_flagged: tuple[str, ...]
    confirm_converged: bool
    residual_damping_ratio: float | None
    stable: bool
    confirmed: bool
    tolerances: EquilibriumTolerances
    confirm_time: float | None = None
    confirm_time_source: ConfirmTimeSource | None = None
    extended: bool = False
    sustained_oscillation: bool = False

    @property
    def growing_oscillation(self) -> bool:
        """Return whether the point was accepted with a *growing* oscillation.

        ``True`` when :attr:`sustained_oscillation` holds and the residual damping
        ratio is below ``-tolerances.growing_damping_ratio``: the equilibrium is
        unstable under the deck's controller (the closed-loop oscillation grows
        without the added damping). Linear damping does not move an equilibrium,
        and the shift test reproduced it, so it is linearized; the linearization
        trims without the controller, so the controller-in-the-loop instability is
        not part of the linear model.

        Returns
        -------
        bool
            Whether the accepted confirmation's oscillation is clearly growing.
        """
        residual = self.residual_damping_ratio
        return (
            self.sustained_oscillation
            and residual is not None
            and residual < -self.tolerances.growing_damping_ratio
        )

    def to_dict(self) -> dict[str, object]:
        """Return the confirmation as a dictionary, with :attr:`growing_oscillation`.

        Returns
        -------
        dict[str, object]
            Every field (as :func:`dataclasses.asdict`) plus
            ``growing_oscillation``.
        """
        return {**asdict(self), "growing_oscillation": self.growing_oscillation}


@dataclass(frozen=True)
class SettleCurveCheck:
    """A settled operating point compared with the user's operating curve.

    When a settle run is seeded from a rotor-performance curve, the controller -- not
    the curve -- decides where the turbine settles, and that settled point is what is
    linearized. Comparing the two surfaces a disagreement between the controller and
    the curve (a mistuned controller, a curve for a different model or a
    different air density, a settle that has not reached its steady state). A
    deviation outside tolerance is reported, never treated as a failure.

    Parameters
    ----------
    rotor_speed : QuantityCheck
        Settled rotor speed (rpm; ``trimmed`` holds the settled value) against the
        curve, with a relative tolerance.
    pitch : QuantityCheck
        Settled collective blade pitch (deg) against the curve, with an absolute
        tolerance.
    tolerances : TrimTolerances
        The tolerances applied (only ``rotor_speed_rel`` and ``pitch_deg`` are used).
    """

    rotor_speed: QuantityCheck
    pitch: QuantityCheck
    tolerances: TrimTolerances

    @property
    def status(self) -> CheckStatus:
        """Return ``"flag"`` if either quantity is outside tolerance.

        Returns
        -------
        {"pass", "flag", "unavailable"}
            ``"flag"`` when the rotor speed or pitch deviates beyond tolerance,
            ``"unavailable"`` when either could not be compared, else ``"pass"``.
        """
        statuses = (self.rotor_speed.status, self.pitch.status)
        if "flag" in statuses:
            return "flag"
        if "unavailable" in statuses:
            return "unavailable"
        return "pass"

    @property
    def flagged_quantities(self) -> tuple[str, ...]:
        """Return the names of the quantities outside tolerance."""
        checks = {"rotor_speed": self.rotor_speed, "pitch": self.pitch}
        return tuple(name for name, check in checks.items() if check.status == "flag")


def compare_settled_to_curve(
    settled: OperatingPoint,
    curve_point: OperatingPoint,
    tolerances: TrimTolerances | None = None,
) -> SettleCurveCheck:
    """Compare a settled operating point with the curve value at its wind speed.

    Parameters
    ----------
    settled : OperatingPoint
        The operating point a settle run returned (the controller's steady state).
    curve_point : OperatingPoint
        The operating curve evaluated at the same wind speed (e.g. the seed the settle
        run started from).
    tolerances : TrimTolerances or None, optional
        Rotor-speed (relative) and pitch (absolute) tolerances; ``None`` uses the
        :class:`~vane.run.TrimTolerances` defaults (1 % and 0.5 deg).

    Returns
    -------
    SettleCurveCheck
        The rotor-speed and pitch comparisons and their combined status.

    Raises
    ------
    ValueError
        If the two points are at different wind speeds.
    """
    if not math.isclose(settled.wind_speed, curve_point.wind_speed, abs_tol=1e-9):
        msg = (
            f"cannot compare a point settled at {settled.wind_speed} m/s with a curve "
            f"point at {curve_point.wind_speed} m/s"
        )
        raise ValueError(msg)
    tol = tolerances if tolerances is not None else TrimTolerances()
    return SettleCurveCheck(
        rotor_speed=QuantityCheck(
            settled.rotor_speed_rpm,
            curve_point.rotor_speed_rpm,
            tol.rotor_speed_rel,
            relative=True,
        ),
        pitch=QuantityCheck(
            settled.blade_pitch_deg,
            curve_point.blade_pitch_deg,
            tol.pitch_deg,
            relative=False,
        ),
        tolerances=tol,
    )


class SteadyStateError(RuntimeError):
    """Raised when a settle run fails, optionally carrying convergence evidence."""

    def __init__(
        self,
        message: str,
        *,
        convergence: SettleConvergence | None = None,
    ) -> None:
        super().__init__(message)
        self.convergence = convergence


def validate_settle_schedule(
    settle_time: float,
    discard_time: float,
    *,
    adaptive: bool = False,
    max_settle_time: float | None = None,
    settle_growth: float = 1.5,
    criterion: str = "auto",
    confirm_time: float | None = None,
) -> None:
    """Check a settle's schedule and criterion before anything is launched.

    :func:`solve_steady_state` runs these checks itself; a sweep calls them once up
    front so a bad schedule fails before its platform probe launches OpenFAST.

    Parameters
    ----------
    settle_time, discard_time : float
        The settle run length and the discarded start-up (s).
    adaptive : bool, optional
        Whether the run is grown until it settles.
    max_settle_time : float or None, optional
        The adaptive ceiling (s); required when ``adaptive``.
    settle_growth : float, optional
        The adaptive growth factor.
    criterion : str, optional
        ``"auto"``, ``"velocity"`` or ``"mean_drift"``.
    confirm_time : float or None, optional
        The confirmation length (s) of a two-stage settle; ``None`` is automatic.

    Raises
    ------
    ValueError
        If any value is out of range.
    """
    for pname, pvalue in (("settle_time", settle_time), ("discard_time", discard_time)):
        if not np.isfinite(pvalue):
            msg = f"{pname} must be finite, got {pvalue}"
            raise ValueError(msg)
    if discard_time < 0.0 or discard_time >= settle_time:
        msg = (
            f"discard_time must be in [0, settle_time); got discard_time="
            f"{discard_time}, settle_time={settle_time}"
        )
        raise ValueError(msg)
    if adaptive:
        if max_settle_time is None:
            msg = "adaptive settling requires max_settle_time"
            raise ValueError(msg)
        if not np.isfinite(max_settle_time):
            msg = f"max_settle_time must be finite, got {max_settle_time}"
            raise ValueError(msg)
        # ``inf`` passes ``> 1.0`` but would jump ``current_time`` straight to
        # ``max_settle_time`` on the first growth, so require it finite too.
        if not np.isfinite(settle_growth) or settle_growth <= 1.0:
            msg = (
                "settle_growth must be > 1 and finite for adaptive settling, got "
                f"{settle_growth}"
            )
            raise ValueError(msg)
        if max_settle_time < settle_time:
            msg = (
                f"max_settle_time ({max_settle_time}) must be >= settle_time "
                f"({settle_time}) for adaptive settling"
            )
            raise ValueError(msg)
    if criterion not in ("velocity", "mean_drift", "auto"):
        msg = f"criterion must be 'auto', 'velocity' or 'mean_drift', got {criterion!r}"
        raise ValueError(msg)
    if confirm_time is not None and (
        not np.isfinite(confirm_time) or confirm_time <= 0.0
    ):
        msg = f"confirm_time must be positive and finite, got {confirm_time}"
        raise ValueError(msg)


def solve_steady_state(
    fst_path: str | Path,
    wind_speed: float,
    *,
    settle_time: float = 1200.0,
    discard_time: float = 600.0,
    rotor_speed_rpm: float = 0.0,
    blade_pitch_deg: float = 0.0,
    platform_position: PlatformPosition | None = None,
    executable: str | Path = "openfast",
    timeout: float | None = None,
    name: str = "",
    velocity_tol: float = _DEFAULT_VELOCITY_TOL,
    angular_velocity_tol: float = _DEFAULT_VELOCITY_TOL,
    adaptive: bool = False,
    max_settle_time: float | None = None,
    settle_growth: float = 1.5,
    require_settled: bool = False,
    wind_shear_exponent: float | None = UNIFORM_INFLOW_SHEAR_EXPONENT,
    controller_library: str | Path | None = None,
    criterion: SettleCriterion = "auto",
    platform_damping: SettlePlatformDamping | None = None,
    confirm_time: float | None = None,
    equilibrium_tolerances: EquilibriumTolerances | None = None,
    prescribed_rotor: bool = False,
    platform_period: float | None = None,
) -> OperatingPoint:
    """Run the fully-coupled model to steady state and read the operating point.

    Rather than trust a fixed ``settle_time``, the run is checked for genuine
    equilibrium: the platform-velocity channels' RMS over the steady window must fall
    within tolerance (at equilibrium the platform is no longer moving). With
    ``adaptive``, ``TMax`` is grown and the run repeated until that criterion is met or
    ``max_settle_time`` is reached. The channels this needs (the platform velocities,
    plus ``RotSpeed`` and ``BldPitch1``) are appended to ElastoDyn's ``OutList`` in
    the working copy when the deck does not already output them.

    Every run is judged over a *trailing* window of fixed length
    ``settle_time - discard_time`` ending at that run's ``TMax``, i.e.
    ``[TMax - (settle_time - discard_time), TMax]`` (the first run's window is
    ``[discard_time, settle_time]``). A longer adaptive run therefore discards the
    early transient rather than diluting it, and the returned rotor speed, blade pitch
    and platform position are the means over that same trailing window. ``TStart`` is
    set to the window start, so only the window is written to the ``.out`` file.

    Parameters
    ----------
    fst_path : str or pathlib.Path
        Primary ``.fst`` deck (a writable working copy; edited in place).
    wind_speed : float
        Steady hub-height wind speed (m/s) to settle at.
    settle_time : float, optional
        Initial ``TMax`` for the settle run (s); should be long enough for the slow
        platform modes to settle past ``discard_time``.
    discard_time : float, optional
        Initial transient to discard (s) in the first run. Together with
        ``settle_time`` it fixes the analysis-window length
        ``settle_time - discard_time`` used by every (adaptive) run; output is written
        from the window start (``TStart``) and the window is averaged.
    rotor_speed_rpm, blade_pitch_deg : float, optional
        Initial rotor speed (rpm) and collective blade pitch (deg) for the run; the
        controller drives them to their steady values, so these only need to be
        reasonable starting guesses.
    platform_position : PlatformPosition or None, optional
        Initial six-DOF floating-platform position written to ElastoDyn. Fields left
        ``None`` retain the deck's values. This changes only initial conditions; the
        active controller and physical modules remain untouched.
    executable : str or pathlib.Path, optional
        OpenFAST executable name or path.
    timeout : float or None, optional
        Run timeout in seconds (per settle run).
    name : str, optional
        Label for the returned operating point.
    velocity_tol : float, optional
        Tolerance on the RMS platform translational velocity (m/s) for declaring the
        platform settled. Default ``1e-2``.
    angular_velocity_tol : float, optional
        Tolerance on the RMS platform angular velocity (deg/s). Default ``1e-2``.
    adaptive : bool, optional
        When ``True``, grow ``TMax`` by ``settle_growth`` and re-run until the platform
        settles within tolerance or ``max_settle_time`` is reached. Requires
        ``max_settle_time``. Default ``False`` (a single run at ``settle_time``).
    max_settle_time : float or None, optional
        The largest ``TMax`` (s) the adaptive search may use; required when
        ``adaptive`` is ``True``.
    settle_growth : float, optional
        Factor by which ``TMax`` grows between adaptive iterations (> 1). Default
        ``1.5``.
    require_settled : bool, optional
        When ``True``, raise :class:`SteadyStateError` if the platform is not confirmed
        settled (either the velocity RMS exceeds tolerance, or no platform-velocity
        channel was output to judge it). Default ``False`` (warn only).
    wind_shear_exponent : float or None, optional
        InflowWind power-law shear exponent (``PLExp``) for the settle run. The default
        ``0.0`` is steady uniform inflow, the same default
        :class:`~vane.run.LinearizationSettings` writes for the linearization, so the
        settled point is linearized in the inflow it settled in. ``None`` leaves the
        deck's ``PLExp`` unchanged. Keep it equal to the linearization's
        ``wind_shear_exponent``; :func:`~vane.run.run_operating_sweep` does.
    controller_library : str or pathlib.Path or None, optional
        Bladed-style controller library (e.g. a ``libdiscon.dll`` built for
        this host) to use for the settle run. When set, ServoDyn's ``DLL_FileName``
        in the deck being settled (a working copy) is rewritten to its absolute
        path. ``None`` (the default) keeps the deck's own ``DLL_FileName``, which
        must then exist. See :func:`check_controller_library`.
    criterion : {"auto", "velocity", "mean_drift"}, optional
        How a run is judged settled. ``"auto"`` (the default) applies the velocity
        criterion and switches to the mean-drift criterion only when the velocity
        criterion fails *and* the platform carries a sustained oscillation: one
        whose cycle-by-cycle RMS neither decays nor grows (its damping ratio, from
        the decay of the most active platform-velocity channel over the analysis
        window, lies within ``equilibrium_tolerances.residual_damping_ratio`` of
        zero). A still-decaying transient keeps the velocity criterion (an adaptive
        settle then runs longer), and a growing oscillation never switches it (a
        two-stage confirmation is judged as described under ``platform_damping``). The
        criterion that judged the run is recorded in
        :attr:`SettleConvergence.criterion` (and per attempt), and a switch in
        :attr:`SettleConvergence.criterion_switched`. ``"velocity"``: every
        platform-velocity RMS over the analysis window is within
        ``velocity_tol``/``angular_velocity_tol`` -- the platform has stopped
        moving. ``"mean_drift"``: the means of the platform position, rotor speed
        and blade pitch over the first and second halves of the analysis window
        agree within ``equilibrium_tolerances`` -- the *mean* no longer moves, even
        if a stationary oscillation about it persists. Use it for a point that
        carries a sustained (e.g. controller-platform) oscillation the velocity
        criterion can never accept; the operating point is then the mean about
        which it oscillates, and the oscillation is still reported (its velocity
        RMS and :attr:`SettleConvergence.oscillation_period`). The criterion used is
        recorded in :attr:`SettleConvergence.criterion`. Each half, and the
        returned mean, is averaged over its trailing whole number of oscillation
        periods so the phase of a partial cycle does not bias it; make each half of
        the window span at least a couple of periods.
    platform_damping : SettlePlatformDamping or None, optional
        Opt-in settle accelerator for floating platforms. When given, the settle
        runs in two stages. Stage 1 adds this linear damping to HydroDyn's
        ``AddBLin`` in the working copy and settles (adaptively, as configured)
        until the criterion is met; the slow, lightly damped platform modes then
        decay in a cycle or two instead of ringing for many minutes. Stage 2
        restores the deck's own ``AddBLin`` byte for byte, restarts from the stage-1
        means (rotor speed, blade pitch, platform position, and the tower-top and,
        when output, blade-tip deflections), and runs an undamped confirmation of
        ``confirm_time`` seconds, judged over its last three quarters. Every mean
        of the confirmation -- and of any settle run -- is averaged over the
        trailing whole number of the run's dominant oscillation period (aligned
        to the window end; the plain window mean when no period is detected or
        less than one fits), so a partial cycle does not bias it. The point is
        accepted when the undamped run's means reproduce the stage-1 means within
        ``equilibrium_tolerances`` (the *shift test*) *and* the undamped run is
        stable about them: it meets the criterion by itself or, for the velocity
        criterion, its residual oscillation decays with at least
        ``equilibrium_tolerances.residual_damping_ratio``. Restarting even at the
        equilibrium rings the slow modes (the controller's start-up transient
        does), and a lightly damped mode cannot ring down within a short run, so
        a decaying residual is accepted. A residual that decays but whose means
        have not yet returned (the shift test fails while the restart transient
        still biases them) gets one more confirmation run, twice as long (capped
        at ``max_settle_time`` when given), and that run is judged instead
        (:attr:`SettleConfirmation.extended`). With ``criterion="auto"`` a
        confirmation that keeps a non-decaying oscillation -- sustained *or
        growing* -- while passing the shift test is accepted at the stage-1
        equilibrium when the oscillation is measurable: its residual damping
        ratio is known and its velocity RMS is within
        ``equilibrium_tolerances.oscillation_velocity_rms``. A load linear in the
        platform velocity is zero at an equilibrium, so the damped stage's
        equilibrium is the undamped model's, stable or not, and the shift test
        has just reproduced it. :attr:`SettleConfirmation.sustained_oscillation`
        is set and a warning logged. When the residual is below
        ``-equilibrium_tolerances.growing_damping_ratio`` the point is also
        flagged :attr:`SettleConfirmation.growing_oscillation` with a distinct
        warning: the equilibrium is unstable under the deck's controller, but the
        linearization trims without the controller, so it is linearized and the
        controller-in-the-loop instability is not part of the linear model. A
        failed shift test, a too large oscillation, too few cycles to measure
        it, or a non-decaying oscillation under the ``"velocity"`` or
        ``"mean_drift"`` criterion is reported as not converged with its reason
        (and raises under ``require_settled``), never silently accepted. A
        single-stage settle never accepts a growing oscillation. A confirmed point is
        the stage-1 equilibrium; an unconfirmed one reports the undamped run's
        means. Both stages are recorded in :attr:`SettleConvergence.confirmation`
        and ``attempts``. See :class:`SettlePlatformDamping` for why a load
        proportional to velocity cannot move the equilibrium. Requires an active
        HydroDyn with an ``AddBLin`` matrix.
    confirm_time : float or None, optional
        ``TMax`` (s) of the undamped confirmation run of a two-stage settle; its
        analysis window is the last three quarters. ``None`` (the default) sizes it
        automatically so the analysis window holds eight periods of the slowest
        platform mode: ``8 T / 0.75`` rounded up to a whole second, with ``T``
        the ``platform_period`` when given, else the damped stage's dominant
        period. The length is never shorter than 1200 s (also used when no period
        is known) and never longer than ``max_settle_time`` when that is given,
        else 3600 s. The length and how it was chosen are recorded in
        :attr:`SettleConfirmation.confirm_time` and
        :attr:`SettleConfirmation.confirm_time_source`. An explicit value is used
        as given.
    platform_period : float or None, optional
        The slowest platform natural period (s), e.g. the longest period of the
        standstill platform probe (:attr:`~vane.run.PlatformProbe.slowest_period`;
        :func:`~vane.run.run_operating_sweep` passes it), used to size the
        automatic ``confirm_time``. ``None`` (the default) uses the damped stage's
        dominant period instead. Ignored without ``platform_damping`` or with an
        explicit ``confirm_time``.
    equilibrium_tolerances : EquilibriumTolerances or None, optional
        Tolerances on a change of the settled means: between the damped and the
        undamped stage of a two-stage settle, and between window halves for
        ``criterion="mean_drift"``. ``None`` uses the
        :class:`EquilibriumTolerances` defaults.
    prescribed_rotor : bool, optional
        Settle without a controller. ServoDyn is switched off in the working copy
        (``CompServo = 0``, so no controller library is loaded and none is
        checked), and ElastoDyn's generator and drivetrain DOFs are locked
        (``GenDOF = DrTrDOF = False``), so the rotor turns at exactly
        ``rotor_speed_rpm`` with the blades held at ``blade_pitch_deg`` while the
        platform finds its equilibrium under the resulting aerodynamic load. This is
        the fallback for a floating deck whose controller library is not available
        on this host: with a rotor-performance curve supplying the rotor speed and
        pitch, it yields the thrust-loaded platform position a CalcSteady
        linearization at that curve point starts from. The returned rotor speed and
        pitch are the prescribed ones; the record carries
        :attr:`SettleConvergence.prescribed_rotor`. ``controller_library`` must not
        be given with it. Default ``False``.

    Returns
    -------
    OperatingPoint
        The settled operating point: the averaged rotor speed and blade pitch, and a
        :class:`~vane.run.operating_points.PlatformPosition` when the deck outputs the
        platform channels (otherwise ``platform`` is ``None``).

    Raises
    ------
    SteadyStateError
        If the settle run fails, its output file cannot be read, or ``require_settled``
        is set and equilibrium could not be confirmed; or, before OpenFAST launches,
        if ServoDyn runs the controller library, no ``controller_library`` is given,
        and the deck's ``DLL_FileName`` does not exist.
    FileNotFoundError
        If ``controller_library`` is given but is not an existing file.
    ValueError
        If ``adaptive`` is set without ``max_settle_time`` or ``settle_growth <= 1``; if
        any deck-written numeric input (``wind_speed``, ``rotor_speed_rpm``,
        ``blade_pitch_deg``, ``settle_time``, ``discard_time``, and ``max_settle_time``
        when adaptive, and ``wind_shear_exponent`` when set) is non-finite; if
        ``wind_speed`` or ``rotor_speed_rpm`` is negative; if ``criterion`` is
        unknown; if ``prescribed_rotor`` is combined with ``controller_library``;
        if ``platform_period`` is given but not positive and finite; or, with
        ``platform_damping``, if ``confirm_time`` is given but not positive
        and finite or the deck has no active HydroDyn file with an ``AddBLin``
        matrix (checked before OpenFAST launches).
    """
    # Finite first, including the timing inputs written into the deck (TStart/TMax): a
    # NaN slips past every ``<`` / ``>=`` comparison below and would be written as a
    # real number. ``settle_growth`` and ``max_settle_time`` are validated in the
    # adaptive branch below where they are used.
    finite_inputs: tuple[tuple[str, float], ...] = (
        ("wind_speed", wind_speed),
        ("rotor_speed_rpm", rotor_speed_rpm),
        ("blade_pitch_deg", blade_pitch_deg),
        ("settle_time", settle_time),
        ("discard_time", discard_time),
    )
    if wind_shear_exponent is not None:
        finite_inputs = (*finite_inputs, ("wind_shear_exponent", wind_shear_exponent))
    for pname, pvalue in finite_inputs:
        if not np.isfinite(pvalue):
            msg = f"{pname} must be finite, got {pvalue}"
            raise ValueError(msg)
    if wind_speed < 0.0:
        msg = f"wind_speed must be non-negative, got {wind_speed}"
        raise ValueError(msg)
    if rotor_speed_rpm < 0.0:
        msg = f"rotor_speed_rpm must be non-negative, got {rotor_speed_rpm}"
        raise ValueError(msg)
    # Require finite positive: an ``inf`` tolerance passes ``> 0.0`` but would make
    # every run compare as settled (RMS <= inf), silently defeating the settle check.
    for tname, tvalue in (
        ("velocity_tol", velocity_tol),
        ("angular_velocity_tol", angular_velocity_tol),
    ):
        if not np.isfinite(tvalue) or tvalue <= 0.0:
            msg = f"{tname} must be positive and finite, got {tvalue}"
            raise ValueError(msg)
    validate_settle_schedule(
        settle_time,
        discard_time,
        adaptive=adaptive,
        max_settle_time=max_settle_time,
        settle_growth=settle_growth,
        criterion=criterion,
    )
    if prescribed_rotor and controller_library is not None:
        msg = (
            "prescribed_rotor settles without a controller; controller_library "
            "cannot be used with it"
        )
        raise ValueError(msg)
    if (
        platform_damping is not None
        and confirm_time is not None
        and (not np.isfinite(confirm_time) or confirm_time <= 0.0)
    ):
        msg = f"confirm_time must be positive and finite, got {confirm_time}"
        raise ValueError(msg)
    if platform_period is not None and (
        not np.isfinite(platform_period) or platform_period <= 0.0
    ):
        msg = f"platform_period must be positive and finite, got {platform_period}"
        raise ValueError(msg)
    tolerances = (
        equilibrium_tolerances
        if equilibrium_tolerances is not None
        else EquilibriumTolerances()
    )

    timeout = _validated_timeout(timeout)
    fst_path = Path(fst_path)
    # Fail at the boundary, before any deck is edited or OpenFAST is launched, when the
    # controller the settle run depends on cannot be loaded, or the damping of a
    # two-stage settle has no HydroDyn matrix to go into.
    library = (
        None
        if prescribed_rotor
        else check_controller_library(fst_path, controller_library)
    )
    if platform_damping is not None:
        _hydrodyn_with_damping_matrix(fst_path)
    run = _SettleRun(
        fst_path=fst_path,
        wind_speed=wind_speed,
        wind_shear_exponent=wind_shear_exponent,
        controller_library=library,
        executable=executable,
        timeout=timeout,
        judge=_SettleJudge(
            velocity_tol=velocity_tol,
            angular_velocity_tol=angular_velocity_tol,
            criterion=criterion,
            tolerances=tolerances,
        ),
        prescribed_rotor=prescribed_rotor,
    )
    start = _SettleStart(
        rotor_speed_rpm=rotor_speed_rpm,
        blade_pitch_deg=blade_pitch_deg,
        platform_position=platform_position,
    )
    schedule = _SettleSchedule(
        settle_time=settle_time,
        discard_time=discard_time,
        max_settle_time=max_settle_time if adaptive else None,
        settle_growth=settle_growth,
    )
    if platform_damping is None:
        stage = run.stage(start, schedule)
        convergence = stage.convergence
        means = stage.means
    else:
        convergence, means = _two_stage_settle(
            run,
            platform_damping,
            start,
            schedule,
            _ConfirmPlan(
                confirm_time=confirm_time,
                platform_period=platform_period,
                ceiling=(
                    max_settle_time
                    if max_settle_time is not None
                    and np.isfinite(max_settle_time)
                    and max_settle_time > 0.0
                    else None
                ),
            ),
            tolerances,
        )
    convergence.requested_criterion = criterion
    convergence.prescribed_rotor = prescribed_rotor
    if prescribed_rotor:
        # The rotor speed and pitch are held, not settled: report the prescribed
        # values exactly rather than their (numerically identical) output means.
        means = {**means, "RotSpeed": rotor_speed_rpm, "BldPitch1": blade_pitch_deg}

    _report_convergence(
        convergence, velocity_tol, angular_velocity_tol, require_settled
    )
    _require_rotor_means(means, fst_path)
    platform = _platform_from_means(means)
    return OperatingPoint(
        wind_speed=wind_speed,
        rotor_speed_rpm=means["RotSpeed"],
        blade_pitch_deg=means["BldPitch1"],
        name=name,
        platform=platform,
        settle_convergence=convergence,
    )


@dataclass(frozen=True)
class _SettleJudge:
    """The settle criterion and its tolerances."""

    velocity_tol: float
    angular_velocity_tol: float
    criterion: SettleCriterion
    tolerances: EquilibriumTolerances

    def judge(
        self,
        columns: list[str],
        window: npt.NDArray[np.float64],
        settle_time: float,
        iteration: int,
    ) -> SettleConvergence:
        """Return the outcome on one run's analysis window."""
        convergence = _settle_convergence(
            columns,
            window,
            self.velocity_tol,
            self.angular_velocity_tol,
            settle_time,
            iteration,
        )
        tolerances = (self.velocity_tol, self.angular_velocity_tol)
        period = _oscillation_period(columns, window, *tolerances)
        drift = _mean_drift(columns, window, period)
        residual = _residual_damping_ratio(columns, window, period, *tolerances)
        convergence.mean_drift = drift
        convergence.oscillation_period = period
        convergence.residual_damping_ratio = residual
        convergence.requested_criterion = self.criterion
        criterion: SettleCriterion = self.criterion
        if criterion == "auto":
            criterion = "velocity"
            if (
                not convergence.converged
                and convergence.assessed
                and _sustained(residual, self.tolerances.residual_damping_ratio)
                # The mean of the platform's own motion must be measurable: without
                # any position channel the drift test would judge only rotor speed
                # and pitch and could accept a drifting platform.
                and any(channel in columns for channel in _PLATFORM_CHANNELS.values())
            ):
                # The platform keeps oscillating at a constant amplitude: the
                # velocity criterion can never pass, so judge whether the *mean* of
                # the oscillation still moves.
                criterion = "mean_drift"
                convergence.criterion_switched = True
        convergence.criterion = criterion
        if criterion == "mean_drift":
            # Not assessed (nothing to compare) cannot fail, as for the velocity
            # criterion; ``assessed`` records it and require_settled rejects it.
            convergence.assessed = drift is not None
            reference = _channel_mean(columns, window, "RotSpeed")
            convergence.converged = drift is None or not self.tolerances.exceeded(
                drift, reference
            )
        return convergence


@dataclass(frozen=True)
class _SettleStart:
    """Initial conditions of a settle run (written to the ElastoDyn copy)."""

    rotor_speed_rpm: float
    blade_pitch_deg: float
    platform_position: PlatformPosition | None
    deflections: Mapping[str, float] | None = None


@dataclass(frozen=True)
class _SettleSchedule:
    """Run length and analysis window of a settle stage.

    ``max_settle_time`` ``None`` means a single run; otherwise ``TMax`` grows by
    ``settle_growth`` until the stage converges or reaches it.
    """

    settle_time: float
    discard_time: float
    max_settle_time: float | None
    settle_growth: float


@dataclass(frozen=True)
class _StageResult:
    """One settle stage's final outcome and channel means."""

    convergence: SettleConvergence
    means: dict[str, float]
    residual_damping_ratio: float | None = None


@dataclass(frozen=True)
class _SettleRun:
    """Everything a settle run needs that does not change between stages."""

    fst_path: Path
    wind_speed: float
    wind_shear_exponent: float | None
    controller_library: Path | None
    executable: str | Path
    timeout: float | None
    judge: _SettleJudge
    prescribed_rotor: bool = False

    def stage(
        self,
        start: _SettleStart,
        schedule: _SettleSchedule,
        *,
        stage: SettleStage = "settle",
        first_iteration: int = 1,
        extra_channels: tuple[str, ...] = (),
    ) -> _StageResult:
        """Run one (optionally adaptive) settle stage and judge it.

        Every run is judged, and averaged, over a trailing window of fixed length
        ``settle_time - discard_time`` ending at its ``TMax``, so a longer adaptive
        run discards the initial transient rather than diluting it.
        """
        current_time = schedule.settle_time
        window_length = schedule.settle_time - schedule.discard_time
        iteration = first_iteration - 1
        attempts: list[SettleAttempt] = []
        while True:
            iteration += 1
            window_start = current_time - window_length
            _configure_settle(
                self.fst_path,
                self.wind_speed,
                start.rotor_speed_rpm,
                start.blade_pitch_deg,
                start.platform_position,
                current_time,
                window_start,
                self.wind_shear_exponent,
                self.controller_library,
                deflections=start.deflections,
                extra_channels=extra_channels,
                prescribed_rotor=self.prescribed_rotor,
            )
            result = run_openfast(
                self.fst_path, executable=self.executable, timeout=self.timeout
            )
            if not result.succeeded:
                tail = "\n".join(result.output.strip().splitlines()[-15:])
                msg = (
                    f"steady-state settle run failed for {self.fst_path} "
                    f"(exit {result.returncode}):\n{tail}"
                )
                raise SteadyStateError(msg)

            out_path = self.fst_path.with_suffix(".out")
            columns, window = _read_channel_window(out_path, window_start)
            convergence = self.judge.judge(columns, window, current_time, iteration)
            convergence.window_start = window_start
            convergence.window_end = current_time
            attempts.append(
                SettleAttempt(
                    translational_velocity_rms=convergence.translational_velocity_rms,
                    angular_velocity_rms=convergence.angular_velocity_rms,
                    assessed=convergence.assessed,
                    converged=convergence.converged,
                    settle_time=convergence.settle_time,
                    iteration=iteration,
                    window_start=window_start,
                    window_end=current_time,
                    stage=stage,
                    mean_drift=convergence.mean_drift,
                    oscillation_period=convergence.oscillation_period,
                    criterion=convergence.criterion,
                    residual_damping_ratio=convergence.residual_damping_ratio,
                )
            )
            if convergence.converged or schedule.max_settle_time is None:
                break
            if current_time >= schedule.max_settle_time:
                break
            next_time = min(
                current_time * schedule.settle_growth, schedule.max_settle_time
            )
            logger.warning(
                "settle attempt %d at %.0f s did not converge: translational-velocity "
                "RMS %.6g m/s, angular RMS %.6g deg/s; retrying at %.0f s",
                iteration,
                current_time,
                convergence.translational_velocity_rms,
                convergence.angular_velocity_rms,
                next_time,
            )
            current_time = next_time

        convergence.attempts = tuple(attempts)
        # The point is the mean of whatever still oscillates: average it over the
        # trailing whole number of its periods so a partial cycle does not bias it
        # (the plain window mean when no period is detected or none fits).
        window = _whole_periods(columns, window, convergence.oscillation_period)
        means = {column: float(window[:, i].mean()) for i, column in enumerate(columns)}
        return _StageResult(
            convergence=convergence,
            means=means,
            residual_damping_ratio=_residual_damping_ratio(
                columns,
                window,
                convergence.oscillation_period,
                self.judge.velocity_tol,
                self.judge.angular_velocity_tol,
            ),
        )


# ElastoDyn tower-top deflection channels, added to the damped stage's output: they
# double as initial conditions, so the undamped confirmation starts with the tower
# already bent by the settled thrust and releasing it does not kick the platform.
_TOWER_TOP_CHANNELS = ("TTDspFA", "TTDspSS")
# Settled deflection channel -> the ElastoDyn initial condition it seeds. The blade-tip
# deflections are seeded only when the deck itself outputs them (they are ElastoDyn
# blade outputs, unavailable when BeamDyn models the blades), so they are never added.
_DEFLECTION_INITIAL_CONDITIONS = {
    "TTDspFA": "TTDspFA",
    "TTDspSS": "TTDspSS",
    "OoPDefl1": "OoPDefl",
    "IPDefl1": "IPDefl",
}


@dataclass(frozen=True)
class _ConfirmPlan:
    """How long the undamped confirmation of a two-stage settle runs.

    ``confirm_time`` ``None`` sizes it automatically (see :func:`_confirm_length`);
    ``ceiling`` is the caller's ``max_settle_time`` (``None``: not given).
    """

    confirm_time: float | None
    platform_period: float | None
    ceiling: float | None


def _confirm_length(
    plan: _ConfirmPlan, damped_period: float | None
) -> tuple[float, ConfirmTimeSource]:
    """Return the confirmation ``TMax`` (s) and how it was chosen.

    An explicit length wins. Otherwise the analysis window (the last
    ``_CONFIRM_ANALYSED_FRACTION`` of the run) must hold ``_AUTO_CONFIRM_PERIODS``
    periods of the slowest platform mode -- ``plan.platform_period`` when known, else
    the damped stage's dominant period -- rounded up to a whole second, floored at
    ``_AUTO_CONFIRM_FALLBACK`` (also the length without any period) and capped at
    ``plan.ceiling`` or, without one, ``_AUTO_CONFIRM_CEILING``.
    """
    if plan.confirm_time is not None:
        return plan.confirm_time, "user"
    source: ConfirmTimeSource
    period: float | None
    if plan.platform_period is not None:
        period, source = plan.platform_period, "platform_period"
    elif damped_period is not None:
        period, source = damped_period, "damped_stage"
    else:
        period, source = None, "fallback"
    length = _AUTO_CONFIRM_FALLBACK
    if period is not None:
        cycles = _AUTO_CONFIRM_PERIODS * period / _CONFIRM_ANALYSED_FRACTION
        length = max(float(math.ceil(cycles)), _AUTO_CONFIRM_FALLBACK)
    ceiling = plan.ceiling if plan.ceiling is not None else _AUTO_CONFIRM_CEILING
    return min(length, ceiling), source


# How a confirmation's length was chosen, for the log.
_CONFIRM_SOURCE_TEXT: dict[ConfirmTimeSource, str] = {
    "user": "as given",
    "platform_period": "sized from the slowest platform period",
    "damped_stage": "sized from the damped stage's dominant period",
    "fallback": "no platform period known",
}


@dataclass(frozen=True)
class _ConfirmOutcome:
    """The judgement of one undamped confirmation run against the damped stage."""

    result: _StageResult
    confirm_time: float
    means: dict[str, float]
    shift: dict[str, float]
    flagged: tuple[str, ...]
    residual: float | None
    decaying: bool
    stable: bool


def _judge_confirmation(
    confirm: _StageResult,
    confirm_time: float,
    damped_quantities: Mapping[str, float],
    tolerances: EquilibriumTolerances,
) -> _ConfirmOutcome:
    """Compare a confirmation run's whole-period means with the damped stage's."""
    quantities = _quantity_means(confirm.means)
    shift = {
        name: quantities[name] - value
        for name, value in damped_quantities.items()
        if name in quantities
    }
    flagged = tolerances.exceeded(shift, damped_quantities.get("rotor_speed_rpm", 0.0))
    residual = confirm.residual_damping_ratio
    decaying = (
        confirm.convergence.criterion == "velocity"
        and residual is not None
        and residual >= tolerances.residual_damping_ratio
    )
    return _ConfirmOutcome(
        result=confirm,
        confirm_time=confirm_time,
        means=quantities,
        shift=shift,
        flagged=flagged,
        residual=residual,
        decaying=decaying,
        stable=confirm.convergence.converged or decaying,
    )


def _bounded_oscillation(
    outcome: _ConfirmOutcome, tolerances: EquilibriumTolerances
) -> bool:
    """Return whether a confirmation's non-decaying oscillation is measurable.

    Its residual damping ratio is known (two or more whole cycles were analysed) and
    every measured platform-velocity RMS is within
    ``tolerances.oscillation_velocity_rms`` (at least one was measured), so its
    whole-period means are meaningful. A *growing* oscillation qualifies too: the
    added damping is linear in the platform velocity, which is zero at an
    equilibrium, so it cannot move the equilibrium, and the shift test (applied by
    the caller) shows the undamped run reproduces it. The growth is the closed loop
    with the deck's controller, which the linearization replaces by a trim.
    """
    if outcome.residual is None:
        return False
    return _within_oscillation_bound(outcome, tolerances)


def _within_oscillation_bound(
    outcome: _ConfirmOutcome, tolerances: EquilibriumTolerances
) -> bool:
    """Return whether every measured platform-velocity RMS is within the bound.

    ``tolerances.oscillation_velocity_rms`` limits the oscillation a confirmed
    equilibrium may carry: beyond it the whole-period mean is not a safe
    measurement, whichever criterion accepted the run.
    """
    convergence = outcome.result.convergence
    if not convergence.assessed:
        return False
    limit = tolerances.oscillation_velocity_rms
    return all(
        math.isnan(rms) or rms <= limit
        for rms in (
            convergence.translational_velocity_rms,
            convergence.angular_velocity_rms,
        )
    )


def _two_stage_settle(
    run: _SettleRun,
    damping: SettlePlatformDamping,
    start: _SettleStart,
    schedule: _SettleSchedule,
    plan: _ConfirmPlan,
    tolerances: EquilibriumTolerances,
) -> tuple[SettleConvergence, dict[str, float]]:
    """Settle with temporary platform damping, then confirm without it.

    Returns the combined convergence record and the channel means of the returned
    point (the damped stage's when confirmed, else the undamped run's).
    """
    hydro = _hydrodyn_with_damping_matrix(run.fst_path)
    original = hydro.read_bytes()
    # surrogateescape keeps any non-UTF-8 byte (a Windows-1252 comment) unchanged.
    damped_text = _add_platform_damping(
        original.decode("utf-8", "surrogateescape"), damping, hydro
    )
    _replace_file(hydro, damped_text.encode("utf-8", "surrogateescape"))
    try:
        damped = run.stage(
            start, schedule, stage="damped", extra_channels=_TOWER_TOP_CHANNELS
        )
    finally:
        # The deck's own damping, byte for byte, whatever happened in stage 1.
        _replace_file(hydro, original)
    _require_rotor_means(damped.means, run.fst_path)
    damped_quantities = _quantity_means(damped.means)
    confirm_time, source = _confirm_length(plan, damped.convergence.oscillation_period)
    logger.info(
        "damped settle stage %s after %.0f s; confirming without the added damping "
        "for %.0f s (%s)",
        "converged" if damped.convergence.converged else "did NOT converge",
        damped.convergence.settle_time,
        confirm_time,
        _CONFIRM_SOURCE_TEXT[source],
    )
    restart = _SettleStart(
        rotor_speed_rpm=max(damped.means["RotSpeed"], 0.0),
        blade_pitch_deg=damped.means["BldPitch1"],
        platform_position=_platform_from_means(damped.means),
        deflections={
            key: damped.means[channel]
            for channel, key in _DEFLECTION_INITIAL_CONDITIONS.items()
            if channel in damped.means
        },
    )

    def confirm_run(length: float, first_iteration: int) -> _ConfirmOutcome:
        result = run.stage(
            restart,
            _SettleSchedule(
                settle_time=length,
                # Skip the restart's first quarter; judge the free motion after it.
                discard_time=(1.0 - _CONFIRM_ANALYSED_FRACTION) * length,
                max_settle_time=None,
                settle_growth=schedule.settle_growth,
            ),
            stage="confirm",
            first_iteration=first_iteration,
            extra_channels=_TOWER_TOP_CHANNELS,
        )
        return _judge_confirmation(result, length, damped_quantities, tolerances)

    first_confirm = len(damped.convergence.attempts) + 1
    outcome = confirm_run(confirm_time, first_confirm)
    confirm_attempts = outcome.result.convergence.attempts
    extended = False
    longer = 2.0 * confirm_time
    if plan.ceiling is not None:
        longer = min(longer, plan.ceiling)
    if (
        damped.convergence.converged
        and outcome.decaying
        and outcome.flagged
        and longer > confirm_time
    ):
        # Still ringing down after the restart: its means are biased by the
        # transient, not moved by the damping. Give it one run twice as long.
        logger.info(
            "undamped confirmation is decaying (residual damping ratio %+.4f) but its "
            "means have not yet returned to the damped equilibrium (%s); extending "
            "it to %.0f s",
            outcome.residual,
            _format_changes({name: outcome.shift[name] for name in outcome.flagged}),
            longer,
        )
        outcome = confirm_run(longer, first_confirm + len(confirm_attempts))
        confirm_attempts = (*confirm_attempts, *outcome.result.convergence.attempts)
        extended = True

    confirm_convergence = outcome.result.convergence
    # Not velocity-quiet and not ringing down: the undamped run keeps oscillating.
    oscillating = not outcome.decaying and not (
        confirm_convergence.converged and confirm_convergence.criterion == "velocity"
    )
    # A measurable sustained or growing oscillation about the confirmed equilibrium
    # is accepted by "auto": linear damping cannot move an equilibrium, so the
    # damped point is the undamped model's, stable under the controller or not.
    bounded = (
        run.judge.criterion == "auto"
        and damped.convergence.converged
        and not outcome.stable
        and not outcome.flagged
        and _bounded_oscillation(outcome, tolerances)
    )
    # A run still oscillating is accepted only within the oscillation bound, also
    # when the mean-drift criterion (to which "auto" switches for a sustained
    # oscillation) found its means steady.
    stable = (
        outcome.stable
        and not outcome.flagged
        and (not oscillating or _within_oscillation_bound(outcome, tolerances))
    )
    confirmed = stable or bounded
    sustained = confirmed and oscillating and damped.convergence.converged
    residual = outcome.residual
    confirmation = SettleConfirmation(
        damping=damping,
        damped_converged=damped.convergence.converged,
        damped_translational_velocity_rms=(
            damped.convergence.translational_velocity_rms
        ),
        damped_angular_velocity_rms=damped.convergence.angular_velocity_rms,
        damped_settle_time=damped.convergence.settle_time,
        damped_means=damped_quantities,
        confirm_means=outcome.means,
        shift=outcome.shift,
        shift_flagged=outcome.flagged,
        confirm_converged=confirm_convergence.converged,
        residual_damping_ratio=residual,
        stable=outcome.stable,
        confirmed=confirmed,
        tolerances=tolerances,
        confirm_time=outcome.confirm_time,
        confirm_time_source=source,
        extended=extended,
        sustained_oscillation=sustained,
    )
    attempts = (*damped.convergence.attempts, *confirm_attempts)
    combined = SettleConvergence(
        translational_velocity_rms=confirm_convergence.translational_velocity_rms,
        angular_velocity_rms=confirm_convergence.angular_velocity_rms,
        assessed=damped.convergence.assessed and confirm_convergence.assessed,
        converged=damped.convergence.converged and confirmed,
        settle_time=confirm_convergence.settle_time,
        iterations=len(attempts),
        attempts=attempts,
        window_start=confirm_convergence.window_start,
        window_end=confirm_convergence.window_end,
        criterion=confirm_convergence.criterion,
        mean_drift=confirm_convergence.mean_drift,
        oscillation_period=confirm_convergence.oscillation_period,
        confirmation=confirmation,
        requested_criterion=confirm_convergence.requested_criterion,
        criterion_switched=(
            damped.convergence.criterion_switched
            or confirm_convergence.criterion_switched
        ),
        residual_damping_ratio=residual,
        sustained_oscillation=sustained,
    )
    # A confirmed point is the damped stage's equilibrium: the undamped run has just
    # reproduced it within tolerance, and the damped means carry no residual ringing.
    # An unconfirmed point reports the undamped model's own means.
    if combined.converged:
        return combined, damped.means
    return combined, outcome.result.means


def _require_rotor_means(means: Mapping[str, float], fst_path: Path) -> None:
    """Raise unless the settle output carried the rotor speed and blade pitch.

    The settled rotor speed and blade pitch are the whole point of the settle run, so
    they must come from the .out -- never silently fall back to the seed values, which
    would linearize at the wrong (often parked) operating point with no error.
    """
    missing = [channel for channel in ("RotSpeed", "BldPitch1") if channel not in means]
    if missing:
        msg = (
            f"steady-state settle run for {fst_path} did not output {missing}; the "
            "settled rotor speed and blade pitch are read from these channels, so add "
            "them to the OpenFAST OutList (else the operating point cannot be measured)"
        )
        raise SteadyStateError(msg)


# Operating-point quantity name -> its ElastoDyn channel.
_QUANTITY_CHANNELS = {
    **_PLATFORM_CHANNELS,
    "rotor_speed_rpm": "RotSpeed",
    "blade_pitch_deg": "BldPitch1",
}


def _quantity_means(channel_means: Mapping[str, float]) -> dict[str, float]:
    """Return the operating-point quantities among ``channel_means``."""
    return {
        name: channel_means[channel]
        for name, channel in _QUANTITY_CHANNELS.items()
        if channel in channel_means
    }


def _channel_mean(
    columns: list[str], window: npt.NDArray[np.float64], channel: str
) -> float:
    """Return a channel's window mean (``0`` when it was not output)."""
    if channel not in columns:
        return 0.0
    return float(window[:, columns.index(channel)].mean())


def _whole_periods(
    columns: list[str], rows: npt.NDArray[np.float64], period: float | None
) -> npt.NDArray[np.float64]:
    """Return the trailing whole number of ``period`` s of ``rows``.

    Averaging a stationary oscillation over a whole number of its periods removes the
    bias a partial cycle adds to the mean. ``rows`` is returned unchanged when the
    period is unknown, no ``Time`` column was output, or it spans less than one
    period.
    """
    if period is None or "Time" not in columns or rows.shape[0] < 2:
        return rows
    time = rows[:, columns.index("Time")]
    step = float(np.median(np.diff(time)))
    span = float(time[-1] - time[0]) + step
    cycles = math.floor(span / period)
    if cycles < 1:
        return rows
    kept = rows[time > time[-1] - cycles * period + 0.5 * step]
    return kept if kept.shape[0] >= 1 else rows


def _mean_drift(
    columns: list[str],
    window: npt.NDArray[np.float64],
    period: float | None = None,
) -> dict[str, float] | None:
    """Return the second-half minus first-half mean of each operating-point quantity.

    With a known oscillation ``period`` each half is averaged over its trailing whole
    number of periods, so a stationary oscillation reads as no drift whatever its
    phase. ``None`` when the window has fewer than two rows or no quantity was output.
    """
    if window.shape[0] < 2:
        return None
    first, second = (
        _whole_periods(columns, half, period) for half in np.array_split(window, 2)
    )
    drift = {
        name: float(
            second[:, columns.index(channel)].mean()
            - first[:, columns.index(channel)].mean()
        )
        for name, channel in _QUANTITY_CHANNELS.items()
        if channel in columns
    }
    return drift or None


def _sustained(residual: float | None, threshold: float) -> bool:
    """Return whether an oscillation neither decays nor grows.

    ``residual`` is the oscillation's damping ratio (see
    :func:`_residual_damping_ratio`); it is sustained when it lies strictly within
    ``threshold`` of zero. ``None`` (fewer than two whole cycles) is not sustained.
    """
    return residual is not None and abs(residual) < threshold


# Fewest samples for which a dominant oscillation period is estimated.
_MIN_PERIOD_SAMPLES = 8


def _most_active_velocity(
    columns: list[str],
    window: npt.NDArray[np.float64],
    velocity_tol: float = _DEFAULT_VELOCITY_TOL,
    angular_velocity_tol: float = _DEFAULT_VELOCITY_TOL,
) -> npt.NDArray[np.float64] | None:
    """Return the platform-velocity channel moving most relative to its tolerance.

    Translational (m/s) and angular (deg/s) channels compete on their de-meaned RMS
    divided by their settle tolerance, so a platform rocking in pitch, roll or yaw
    while its translations are quiet is analysed by its angular motion rather than
    by translational noise. ``None`` when no platform-velocity channel was output.
    """
    candidates = [
        (columns.index(name), tol)
        for group, tol in (
            (_TRANSLATIONAL_VELOCITY_CHANNELS, velocity_tol),
            (_ANGULAR_VELOCITY_CHANNELS, angular_velocity_tol),
        )
        for name in group
        if name in columns
    ]
    if not candidates:
        return None

    def activity(candidate: tuple[int, float]) -> float:
        values = window[:, candidate[0]]
        return float(np.sqrt(np.mean((values - values.mean()) ** 2))) / candidate[1]

    index, _ = max(candidates, key=activity)
    return np.asarray(window[:, index], dtype=np.float64)


def _oscillation_period(
    columns: list[str],
    window: npt.NDArray[np.float64],
    velocity_tol: float = _DEFAULT_VELOCITY_TOL,
    angular_velocity_tol: float = _DEFAULT_VELOCITY_TOL,
) -> float | None:
    """Return the dominant period (s) of the most active platform-velocity channel.

    The channel is chosen by :func:`_most_active_velocity` (translational and
    angular channels compete relative to their tolerances). The spectrum is a
    Hann-windowed, zero-padded FFT of the de-meaned signal. ``None`` when there are
    too few samples, no time step, or no motion at all.
    """
    if window.shape[0] < _MIN_PERIOD_SAMPLES or "Time" not in columns:
        return None
    chosen = _most_active_velocity(columns, window, velocity_tol, angular_velocity_tol)
    if chosen is None:
        return None
    signal = chosen - chosen.mean()
    if not np.any(signal):
        return None
    time = window[:, columns.index("Time")]
    step = float(np.median(np.diff(time)))
    if not np.isfinite(step) or step <= 0.0:
        return None
    padded = 8 * signal.size
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(signal.size), n=padded))
    frequencies = np.fft.rfftfreq(padded, step)
    peak = int(np.argmax(spectrum[1:])) + 1
    return float(1.0 / frequencies[peak])


def _residual_damping_ratio(
    columns: list[str],
    window: npt.NDArray[np.float64],
    period: float | None,
    velocity_tol: float = _DEFAULT_VELOCITY_TOL,
    angular_velocity_tol: float = _DEFAULT_VELOCITY_TOL,
) -> float | None:
    """Estimate the damping ratio of the residual platform oscillation.

    The most active platform-velocity channel (:func:`_most_active_velocity`) is split
    into whole periods counted back from the end of the window; the slope of the
    logarithm of their RMS per cycle is the logarithmic decrement ``delta``, and the
    damping ratio is ``delta / (2 pi)`` (light-damping approximation). Positive
    means decaying; zero or negative means sustained or growing. ``None`` when the
    period is unknown or fewer than two whole cycles are available.
    """
    if period is None or "Time" not in columns or window.shape[0] < 2:
        return None
    signal = _most_active_velocity(columns, window, velocity_tol, angular_velocity_tol)
    if signal is None:
        return None
    time = window[:, columns.index("Time")]
    step = float(np.median(np.diff(time)))
    cycles = math.floor((float(time[-1] - time[0]) + step) / period)
    if cycles < 2:
        return None
    rms = []
    for cycle in range(cycles, 0, -1):
        upper = time[-1] - (cycle - 1) * period
        mask = (time > upper - period + 0.5 * step) & (time <= upper + 0.5 * step)
        rms.append(float(np.sqrt(np.mean(signal[mask] ** 2))))
    if min(rms) <= 0.0:
        return None
    slope = float(np.polyfit(np.arange(cycles), np.log(rms), 1)[0])
    return -slope / (2.0 * math.pi)


def _hydrodyn_with_damping_matrix(fst_path: Path) -> Path:
    """Return the deck's active HydroDyn file, checking it has an ``AddBLin`` matrix.

    Raises
    ------
    ValueError
        If HydroDyn is off or missing, or its file has no ``AddBLin`` line.
    """
    deck = FastInputFile.read(fst_path)
    comp_hydro = _deck_float(deck, "CompHydro")
    path: Path | None = None
    if (comp_hydro is None or comp_hydro != 0.0) and "HydroFile" in deck:
        try:
            path = deck.resolve("HydroFile")
        except FastInputError:
            path = None
    if path is None or not path.is_file():
        msg = (
            f"platform_damping needs an active HydroDyn input in {fst_path} (CompHydro "
            "= 1 and an existing HydroFile); the damping is added to its AddBLin matrix"
        )
        raise ValueError(msg)
    text = path.read_text(encoding="utf-8", errors="surrogateescape")
    _damping_matrix_rows(text.splitlines(), path)
    return path


def _hydrodyn_int(lines: list[str], keyword: str, default: int) -> int:
    """Return the integer value of a HydroDyn ``value keyword`` line, or ``default``."""
    for line in lines:
        tokens = line.split()
        if len(tokens) >= 2 and tokens[1] == keyword:
            try:
                return int(fortran_float(tokens[0]))
            except ValueError:
                return default
    return default


def _leading_numbers(line: str) -> list[float]:
    """Return the numeric tokens that start ``line`` (Fortran ``D`` exponents too)."""
    numbers: list[float] = []
    for token in line.split():
        if _NUMBER.fullmatch(token) is None:
            break
        numbers.append(float(token.replace("D", "E").replace("d", "e")))
    return numbers


# A real number as Fortran list-directed input writes it (``E`` or ``D`` exponent).
_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][+-]?\d+)?")


def _damping_matrix_rows(lines: list[str], path: Path) -> tuple[range, int]:
    """Locate the ``AddBLin`` rows: return their line range and column count.

    HydroDyn writes ``AddBLin`` as ``6 NBody`` rows -- of ``6 NBody`` columns when
    ``NBodyMod = 1`` (one coupled matrix), else of 6 columns (one 6x6 matrix per
    body) -- with the ``AddBLin`` label on the first row; ``AddBQuad`` follows. The
    block is validated (enough numbers on every row, no other label inside it,
    ``AddBQuad`` in the block after it) so a deck laid out differently fails loudly
    instead of having the wrong numbers edited.
    """
    n_body = max(_hydrodyn_int(lines, "NBody", 1), 1)
    n_body_mod = _hydrodyn_int(lines, "NBodyMod", 1)
    n_rows = 6 * n_body
    n_cols = 6 * n_body if n_body_mod == 1 else 6
    label = next((i for i, line in enumerate(lines) if "AddBLin" in line.split()), None)
    if label is None:
        msg = f"no AddBLin matrix found in {path}; cannot add platform damping"
        raise ValueError(msg)
    rows = range(label, label + n_rows)
    following = lines[rows.stop : rows.stop + n_rows]
    valid = (
        rows.stop <= len(lines)
        and all(len(_leading_numbers(lines[i])) >= n_cols for i in rows)
        and not any("AddBQuad" in lines[i].split() for i in rows)
        and any("AddBQuad" in line.split() for line in following)
    )
    if not valid:
        msg = (
            f"cannot locate the {n_rows}x{n_cols} AddBLin matrix in {path} (expected "
            f"{n_rows} numeric rows starting at the AddBLin label, followed by "
            "AddBQuad); cannot add platform damping"
        )
        raise ValueError(msg)
    return rows, n_cols


def _add_platform_damping(text: str, damping: SettlePlatformDamping, path: Path) -> str:
    """Return ``text`` with ``damping`` added to the diagonal of body 1's ``AddBLin``.

    Only the six diagonal entries of the first body are rewritten; every other token
    and each row's trailing label/comment is kept verbatim.
    """
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()
    rows, _ = _damping_matrix_rows(lines, path)
    for dof, added in enumerate(damping.diagonal()):
        if added == 0.0:
            continue
        index = rows.start + dof
        line = lines[index]
        spans = [match.span() for match in re.finditer(r"\S+", line)]
        begin, stop = spans[dof]
        value = _leading_numbers(line)[dof] + added
        lines[index] = f"{line[:begin]}{value:.6E}{line[stop:]}"
    trailing = newline if text.endswith(("\n", "\r")) else ""
    return newline.join(lines) + trailing


def _replace_file(path: Path, content: bytes) -> None:
    """Write ``content`` to ``path`` through an atomic replace.

    The replace swaps in a new inode, so a hard-linked case copy never writes
    through to the template it shares the file with. A transient Windows
    access-denied or sharing-violation error retries the whole write (see
    :func:`vane.utils.fs_retry.retry_transient`).
    """

    def write_once() -> None:
        descriptor, temp_name = tempfile.mkstemp(
            dir=path.parent, prefix=f"{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
            Path(temp_name).replace(path)
        except BaseException:
            Path(temp_name).unlink(missing_ok=True)
            raise

    retry_transient(write_once, description=f"write {path.name}")


# Seconds a child process may take to load a candidate controller library.
_LOAD_CHECK_TIMEOUT = 60.0
# Load-check outcomes by (resolved path, size, modification time).
_LOAD_CHECKS: dict[tuple[str, int, int], str | None] = {}
_LOAD_CHECKS_LOCK = threading.Lock()
# Loads the library given as argv[1]; winmode=0 searches like a plain LoadLibrary
# (including PATH, where a controller's Fortran runtime often lives), as OpenFAST does.
_LOAD_CHECK_SCRIPT = (
    "import ctypes, sys\n"
    "kwargs = {'winmode': 0} if sys.platform == 'win32' else {}\n"
    "ctypes.CDLL(sys.argv[1], **kwargs)\n"
)


def _load_failure(path: Path) -> str | None:
    """Return why a child Python process cannot load the library at ``path``.

    A file with the host's library suffix may still be corrupt, built for another
    architecture or missing a dependency; OpenFAST would then fail to load it in the
    settle run. The library is loaded in a separate process (a crash there cannot
    affect this one), and the outcome is cached for the file's size and timestamp.
    ``None`` when it loads; otherwise the reason (the loader's last error line).
    """
    try:
        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    except OSError as error:
        return str(error)
    with _LOAD_CHECKS_LOCK:
        if key in _LOAD_CHECKS:
            return _LOAD_CHECKS[key]
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _LOAD_CHECK_SCRIPT, str(path)],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_LOAD_CHECK_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        loads, reason = False, str(error)
    else:
        loads = completed.returncode == 0
        lines = (completed.stderr or "").strip().splitlines()
        reason = lines[-1] if lines else ""
    failure = None if loads else (reason or "the loader reported no reason")
    with _LOAD_CHECKS_LOCK:
        _LOAD_CHECKS[key] = failure
    return failure


def _loads_on_host(path: Path) -> bool:
    """Return whether the library at ``path`` loads on this host (see above).

    A library that does not load is logged as a warning naming the reason.
    """
    failure = _load_failure(path)
    if failure is not None:
        logger.warning(
            "controller library %s exists but cannot be loaded on this host (%s)",
            path,
            failure,
        )
    return failure is None


def check_controller_library(
    fst_path: str | Path, controller_library: str | Path | None = None
) -> Path | None:
    """Check that a settle run can load the deck's Bladed-style controller library.

    A settle run keeps the deck's controller active, so when ServoDyn routes any
    control (``PCMode``, ``VSContrl``, ``HSSBrMode``, ``YCMode``, ``AfCmode`` or
    ``CCmode`` = 5) through the controller library, OpenFAST loads ``DLL_FileName``
    (relative to the ServoDyn file) at initialization. Published decks often carry a
    path valid only on the machine that generated them (e.g. a Linux
    ``libdiscon.so`` on Windows); this check turns that into a clear error before
    OpenFAST is launched. Nothing is written.

    Parameters
    ----------
    fst_path : str or pathlib.Path
        The primary ``.fst`` deck.
    controller_library : str or pathlib.Path or None, optional
        A replacement controller library for the settle run. When given it must be
        an existing file; the deck's own ``DLL_FileName`` is then not checked.

    Returns
    -------
    pathlib.Path or None
        The resolved ``controller_library`` to write into the settle copy's
        ``DLL_FileName``, or ``None`` when nothing is to be rewritten (no override
        given, or ServoDyn does not run the controller library).

    Raises
    ------
    FileNotFoundError
        If ``controller_library`` is given but does not exist.
    SteadyStateError
        If no override is given, ServoDyn runs the controller library, and its
        ``DLL_FileName`` does not name an existing file; or if the library to be
        used (the override, or the deck's own) exists but cannot be loaded on this
        host (wrong architecture, corrupt, or missing a dependency), checked by
        loading it in a child process.
    """
    library: Path | None = None
    if controller_library is not None:
        library = resolve_path(controller_library)
        if not library.is_file():
            msg = f"controller library not found: {library}"
            raise FileNotFoundError(msg)
    servo = _active_servodyn(FastInputFile.read(fst_path))
    if servo is None or not _uses_controller_library(servo):
        if library is not None:
            logger.warning(
                "controller_library %s is unused: %s does not run a ServoDyn "
                "controller library",
                library,
                fst_path,
            )
        return None
    if library is not None:
        _require_loadable(library, "controller_library")
        return library
    try:
        deck_library: Path | None = servo.resolve("DLL_FileName")
    except FastInputError:
        deck_library = None
    if deck_library is None or not deck_library.is_file():
        named = deck_library if deck_library is not None else "(none)"
        msg = (
            f"controller library not found: {named} (DLL_FileName in {servo.path}); "
            "pass a controller library path (controller_library=...) built for this "
            "platform"
        )
        raise SteadyStateError(msg)
    _require_loadable(deck_library, f"DLL_FileName in {servo.path}")
    return None


def _require_loadable(library: Path, origin: str) -> None:
    """Raise when an existing controller library does not load on this host."""
    failure = _load_failure(library)
    if failure is not None:
        msg = (
            f"controller library {library} ({origin}) cannot be loaded on this host "
            f"({failure}); pass a controller library built for this platform "
            "(controller_library=...)"
        )
        raise SteadyStateError(msg)


def _active_servodyn(deck: FastInputFile) -> FastInputFile | None:
    """Return the deck's ServoDyn input when ServoDyn runs, else ``None``."""
    comp_servo = _deck_float(deck, "CompServo")
    if comp_servo is not None and comp_servo == 0.0:
        return None
    if "ServoFile" not in deck:
        return None
    try:
        path = deck.resolve("ServoFile")
    except FastInputError:
        return None
    if not path.is_file():
        return None
    return FastInputFile.read(path)


def _uses_controller_library(servo: FastInputFile) -> bool:
    """Return whether any ServoDyn control is routed through the controller library."""
    return any(_deck_float(servo, key) == _DLL_SWITCH for key in DLL_CONTROL_KEYS)


def _settle_still_water(sea_state: FastInputFile) -> None:
    """Set still water exactly as the linearization does (``WaveMod = 0``).

    The wave options OpenFAST rejects in still water (stretching, MacCamy-Fuchs
    diffraction, a ``DEFAULT`` current heading, directional spreading, a constrained
    wave) are neutralised by the linearization's own helper, so the equilibrium is
    found in the same sea state it is then linearized in.
    """
    if "WaveMod" not in sea_state:
        return
    sea_state.set("WaveMod", 0)
    _make_still_water_consistent(sea_state)


def _settle_convergence(
    columns: list[str],
    window: npt.NDArray[np.float64],
    velocity_tol: float,
    angular_velocity_tol: float,
    settle_time: float,
    iteration: int,
) -> SettleConvergence:
    """Judge equilibrium from the platform-velocity RMS over the analysis window."""
    translational = _max_channel_rms(columns, window, _TRANSLATIONAL_VELOCITY_CHANNELS)
    angular = _max_channel_rms(columns, window, _ANGULAR_VELOCITY_CHANNELS)
    assessed = not (np.isnan(translational) and np.isnan(angular))
    within_translational = np.isnan(translational) or translational <= velocity_tol
    within_angular = np.isnan(angular) or angular <= angular_velocity_tol
    return SettleConvergence(
        translational_velocity_rms=translational,
        angular_velocity_rms=angular,
        assessed=assessed,
        converged=bool(within_translational and within_angular),
        settle_time=settle_time,
        iterations=iteration,
    )


def _max_channel_rms(
    columns: list[str],
    window: npt.NDArray[np.float64],
    channels: tuple[str, ...],
) -> float:
    """Return the largest RMS over the present ``channels`` (``nan`` if none exist)."""
    present = [columns.index(name) for name in channels if name in columns]
    if not present:
        return float("nan")
    values = [float(np.sqrt(np.mean(window[:, index] ** 2))) for index in present]
    return max(values)


def _report_convergence(
    convergence: SettleConvergence,
    velocity_tol: float,
    angular_velocity_tol: float,
    require_settled: bool,
) -> None:
    """Log the settling outcome and, when required, raise if it is not settled."""
    message = _unsettled_message(convergence, velocity_tol, angular_velocity_tol)
    if message is not None:
        if require_settled:
            raise SteadyStateError(message, convergence=convergence)
        logger.warning(message)
        return
    confirmation = convergence.confirmation
    if confirmation is not None and confirmation.growing_oscillation:
        logger.warning(
            "settled with a GROWING oscillation: the equilibrium is unstable under "
            "the deck's controller -- without the added damping the closed-loop "
            "oscillation grows (%s, residual damping ratio %+.4f, below -%g). Linear "
            "damping acts on the platform velocity, which is zero at an equilibrium, "
            "so it does not move the equilibrium, and the undamped run's whole-period "
            "mean reproduced it (shift test passed). The linearization trims without "
            "the controller, so this equilibrium is linearized and the "
            "controller-in-the-loop instability is not part of the linear model",
            _motion(convergence, velocity_tol, angular_velocity_tol),
            confirmation.residual_damping_ratio,
            confirmation.tolerances.growing_damping_ratio,
        )
        return
    if confirmation is not None and confirmation.sustained_oscillation:
        residual = confirmation.residual_damping_ratio
        logger.warning(
            "settled with a sustained oscillation: the undamped confirmation keeps a "
            "bounded oscillation (%s, residual damping ratio %s) whose whole-period "
            "mean reproduces the damped equilibrium; that equilibrium is the "
            "operating point, and its linearization exposes the weakly damped or "
            "unstable mode sustaining the oscillation",
            _motion(convergence, velocity_tol, angular_velocity_tol),
            "n/a" if residual is None else f"{residual:+.4f}",
        )
        return
    moving = (
        convergence.translational_velocity_rms > velocity_tol
        or convergence.angular_velocity_rms > angular_velocity_tol
    )
    if confirmation is not None and convergence.criterion == "velocity":
        # A confirmation still ringing down is accepted by its decay and its mean.
        if moving:
            logger.info(
                "settled: the undamped confirmation is still ringing down (%s) and "
                "its whole-period mean reproduces the damped equilibrium",
                _motion(convergence, velocity_tol, angular_velocity_tol),
            )
        return
    if moving:
        # Only the mean-drift criterion accepts a run that is still moving.
        logger.warning(
            "settled by the mean-drift criterion with a sustained oscillation about "
            "the mean: %s; the oscillation's mean is the operating point",
            _motion(convergence, velocity_tol, angular_velocity_tol),
        )


def _motion(
    convergence: SettleConvergence, velocity_tol: float, angular_velocity_tol: float
) -> str:
    """Describe the remaining platform motion of a settle run."""
    motion = (
        f"translational-velocity RMS {convergence.translational_velocity_rms:.3g} m/s "
        f"(tol {velocity_tol:.3g}), angular RMS "
        f"{convergence.angular_velocity_rms:.3g} deg/s "
        f"(tol {angular_velocity_tol:.3g})"
    )
    if convergence.oscillation_period is not None:
        motion += f", dominant period {convergence.oscillation_period:.0f} s"
    return motion


def _unsettled_message(
    convergence: SettleConvergence,
    velocity_tol: float = _DEFAULT_VELOCITY_TOL,
    angular_velocity_tol: float = _DEFAULT_VELOCITY_TOL,
) -> str | None:
    """Explain why a settle is not a confirmed equilibrium (``None`` when it is)."""
    if not convergence.assessed:
        if convergence.criterion == "mean_drift":
            return (
                "could not verify steady state: the settle output has too few rows or "
                "no operating-point channel to compare the analysis-window halves"
            )
        channels = ", ".join(
            (*_TRANSLATIONAL_VELOCITY_CHANNELS, *_ANGULAR_VELOCITY_CHANNELS)
        )
        return (
            "could not verify steady state: the deck output no platform-velocity "
            f"channel; add {channels} to the OutList to confirm settling"
        )
    if convergence.converged:
        return None
    message = (
        f"platform not settled after {convergence.settle_time:.0f} s "
        f"({convergence.iterations} run(s)): "
        f"{_motion(convergence, velocity_tol, angular_velocity_tol)}"
    )
    if convergence.criterion == "mean_drift" and convergence.mean_drift:
        message += f"; half-window mean drift {_format_changes(convergence.mean_drift)}"
    if convergence.confirmation is not None:
        message += "; " + _confirmation_failure(convergence, convergence.confirmation)
    return message


def _format_changes(changes: Mapping[str, float]) -> str:
    """Return ``name +x.xxx`` pairs for a log message."""
    return ", ".join(f"{name} {value:+.4g}" for name, value in changes.items())


def _confirmation_failure(
    convergence: SettleConvergence, confirmation: SettleConfirmation
) -> str:
    """Explain why a two-stage settle was not confirmed."""
    reasons: list[str] = []
    if not confirmation.damped_converged:
        reasons.append(
            "the damped stage did not settle within its time limit "
            f"({confirmation.damped_settle_time:.0f} s)"
        )
    if confirmation.shift_flagged:
        flagged = {
            name: confirmation.shift[name] for name in confirmation.shift_flagged
        }
        reasons.append(
            "removing the added damping moved the equilibrium "
            f"({_format_changes(flagged)}), so the damped stage's point is not the "
            "undamped model's"
        )
    if not confirmation.stable:
        reasons.append(_instability_reason(convergence, confirmation))
    if confirmation.extended:
        reasons.append(
            "(judged on the confirmation extended to "
            f"{confirmation.confirm_time or 0.0:.0f} s)"
        )
    return "; ".join(reasons) if reasons else "the two-stage settle was not confirmed"


def _instability_reason(
    convergence: SettleConvergence, confirmation: SettleConfirmation
) -> str:
    """Explain why an undamped confirmation's non-decaying motion was not accepted.

    A sustained or growing oscillation is not by itself a reason: the ``"auto"``
    criterion accepts it about a reproduced equilibrium. The reasons listed are the
    ones that actually blocked it.
    """
    tolerances = confirmation.tolerances
    residual = confirmation.residual_damping_ratio
    decay = ""
    if residual is not None:
        decay = f" (residual oscillation damping ratio {residual:+.4f})"
    if residual is not None and residual < -tolerances.growing_damping_ratio:
        motion = (
            f"the undamped confirmation run's oscillation is growing{decay}, below "
            f"-{tolerances.growing_damping_ratio:g} (the equilibrium is unstable "
            "under the deck's controller)"
        )
    else:
        motion = (
            "the undamped confirmation run is still moving and not decaying"
            f"{decay}: the motion the damping suppressed returns without it"
        )
    blockers: list[str] = []
    if convergence.requested_criterion != "auto":
        blockers.append(
            f"the {convergence.requested_criterion!r} criterion does not accept a "
            "non-decaying oscillation (only 'auto' accepts one about a reproduced "
            "equilibrium)"
        )
    if confirmation.shift_flagged:
        blockers.append("the equilibrium moved (see above)")
    if not confirmation.damped_converged:
        blockers.append("the damped stage did not settle")
    if residual is None:
        blockers.append(
            "fewer than two whole cycles were analysed, so the oscillation could not "
            "be measured"
        )
    limit = tolerances.oscillation_velocity_rms
    too_large = [
        f"{value:.3g} {unit}"
        for value, unit in (
            (convergence.translational_velocity_rms, "m/s"),
            (convergence.angular_velocity_rms, "deg/s"),
        )
        if not math.isnan(value) and value > limit
    ]
    if too_large:
        blockers.append(
            f"its platform-velocity RMS ({', '.join(too_large)}) exceeds "
            f"oscillation_velocity_rms {limit:g}, too large for its whole-period mean "
            "to measure the equilibrium"
        )
    if not blockers:
        blockers.append("the oscillation could not be accepted")
    return f"{motion}; not accepted because {'; and '.join(blockers)}"


def _configure_settle(
    fst_path: Path,
    wind_speed: float,
    rotor_speed_rpm: float,
    blade_pitch_deg: float,
    platform_position: PlatformPosition | None,
    settle_time: float,
    window_start: float,
    wind_shear_exponent: float | None = UNIFORM_INFLOW_SHEAR_EXPONENT,
    controller_library: Path | None = None,
    *,
    deflections: Mapping[str, float] | None = None,
    extra_channels: tuple[str, ...] = (),
    prescribed_rotor: bool = False,
) -> None:
    """Configure the deck for a constant-wind, still-water settle run, in place.

    ``TMax`` is ``settle_time`` and ``TStart`` is ``window_start``, so only the
    trailing analysis window is written to the output file. ``PLExp`` is set to
    ``wind_shear_exponent`` (uniform inflow by default) unless it is ``None``. When
    ``controller_library`` is given, ServoDyn's ``DLL_FileName`` is pointed at it.
    ``deflections`` sets ElastoDyn initial deflections by keyword (``TTDspFA``,
    ``TTDspSS``, ``OoPDefl``, ``IPDefl``) and ``extra_channels`` are output in
    addition to the settle channels. ``prescribed_rotor`` switches ServoDyn off
    (``CompServo = 0``) and locks the generator and drivetrain DOFs so the rotor turns
    at ``rotor_speed_rpm`` with the blades at ``blade_pitch_deg``.
    """
    deck = FastInputFile.read(fst_path)
    _set_if_present(deck, "Linearize", False)
    _set_if_present(deck, "CalcSteady", False)
    _set_if_present(deck, "WrVTK", 0)
    _set_if_present(deck, "OutFileFmt", 1)  # text output so it can be parsed
    # Coarse (only the mean is needed), but a whole multiple of DT: OpenFAST rejects
    # any other output step.
    _set_if_present(deck, "DT_Out", _settle_output_step(_deck_float(deck, "DT")))
    _set_if_present(deck, "TStart", window_start)  # write only the analysis window
    _set_if_present(deck, "TMax", settle_time)
    # Enable the inflow module so the requested wind is actually used; a deck copied
    # from a parked/still-air template may have CompInflow=0, which would otherwise let
    # the turbine settle in still air rather than at the requested wind speed.
    if wind_speed > 0.0:
        if _inflow_path(deck) is None:
            msg = (
                f"settle requests wind speed {wind_speed} m/s but {fst_path} has no "
                "usable InflowWind file to set it in; it would settle in still air"
            )
            raise SteadyStateError(msg)
        _set_if_present(deck, "CompInflow", 1)
    else:
        _set_if_present(deck, "CompInflow", 0)
    if prescribed_rotor:
        # No controller: nothing in ServoDyn may move the pitch or load a library.
        _set_if_present(deck, "CompServo", 0)
    deck.write()

    def set_elastodyn(elastodyn: FastInputFile) -> None:
        _set_if_present(elastodyn, "RotSpeed", rotor_speed_rpm)
        if prescribed_rotor:
            # A locked generator DOF holds the rotor at its initial speed.
            _set_if_present(elastodyn, "GenDOF", False)
            _set_if_present(elastodyn, "DrTrDOF", False)
        for key in ("BlPitch(1)", "BlPitch(2)", "BlPitch(3)"):
            _set_if_present(elastodyn, key, blade_pitch_deg)
        if platform_position is not None:
            for attribute, key in _PLATFORM_CHANNELS.items():
                value = getattr(platform_position, attribute)
                if value is not None:
                    _set_if_present(elastodyn, key, value)
        for key, value in (deflections or {}).items():
            _set_if_present(elastodyn, key, value)

    def set_inflow(inflow: FastInputFile) -> None:
        _set_if_present(inflow, "WindType", 1)
        _set_if_present(inflow, "HWindSpeed", wind_speed)
        if wind_shear_exponent is not None:
            _set_if_present(inflow, "PLExp", wind_shear_exponent)

    _edit_subfile(deck, "EDFile", set_elastodyn)
    _ensure_settle_channels(deck, (*_SETTLE_CHANNELS, *extra_channels))
    _edit_subfile(deck, "InflowFile", set_inflow)
    _edit_subfile(deck, "SeaStFile", _settle_still_water)
    if controller_library is not None:
        library_value = f'"{controller_library.as_posix()}"'
        _edit_subfile(
            deck,
            "ServoFile",
            lambda servo: _set_if_present(servo, "DLL_FileName", library_value),
        )


# Target output step (s) of a settle run.
_SETTLE_OUTPUT_STEP = 1.0


def _settle_output_step(time_step: float | None) -> float:
    """Return the settle ``DT_Out``: the whole multiple of ``DT`` nearest one second.

    OpenFAST requires ``DT_Out`` to be an integer multiple of ``DT`` (and at least
    ``DT``), so a fixed one-second step fails on a deck whose ``DT`` does not divide
    it (e.g. ``0.003`` or ``0.015`` s). An unknown or non-positive ``DT`` keeps one
    second.
    """
    if time_step is None or not math.isfinite(time_step) or time_step <= 0.0:
        return _SETTLE_OUTPUT_STEP
    steps = max(1, round(_SETTLE_OUTPUT_STEP / time_step))
    return float(f"{steps * time_step:.12g}")


def _ensure_settle_channels(
    deck: FastInputFile, channels: tuple[str, ...] = _SETTLE_CHANNELS
) -> None:
    """Add the channels the settle outcome needs to ElastoDyn's ``OutList``.

    The settled rotor speed and pitch (``RotSpeed``, ``BldPitch1``) and the platform
    velocities that judge equilibrium are appended, as one quoted line before the
    ``END`` of the first ``OutList`` block, when the deck does not already output
    them; published decks commonly omit the velocities, which would otherwise leave
    every settle unverifiable. A file without a recognisable ``OutList`` block is
    left untouched.
    """
    if "EDFile" not in deck:
        return
    try:
        path = deck.resolve("EDFile")
    except FastInputError:
        return
    if not path.is_file():
        return
    lines = path.read_text(encoding="utf-8", errors="surrogateescape").splitlines()
    header = next(
        (i for i, line in enumerate(lines) if line.split()[:1] == ["OutList"]), None
    )
    if header is None:
        return
    end = next(
        (i for i in range(header + 1, len(lines)) if lines[i][:3].upper() == "END"),
        None,
    )
    if end is None:
        return
    present = {
        name.upper()
        for line in lines[header + 1 : end]
        for quoted in _QUOTED.findall(line)
        for name in _CHANNEL_SEPARATOR.split(quoted)
        if name
    }
    missing = [name for name in channels if name.upper() not in present]
    if not missing:
        return
    lines.insert(end, '"' + ", ".join(missing) + '"')
    # Atomic replace: never write through a hard link into the template's ElastoDyn.
    _replace_file(path, ("\n".join(lines) + "\n").encode("utf-8", "surrogateescape"))


def _read_channel_window(
    out_path: Path, discard_time: float
) -> tuple[list[str], npt.NDArray[np.float64]]:
    """Return the channel names and the analysis-window data of an OpenFAST ``.out``.

    The ``.out`` file's channel-name row (the line starting with ``Time``) is followed
    by a units row and then the data. Rows with ``Time >= discard_time`` are kept (a
    safety net even though ``TStart`` already trims the output); if none qualify, all
    rows are kept. Returns the columns and the kept data as a ``(n_row, n_col)`` array.

    A kept sample that is NaN or infinite, or a field OpenFAST overflowed (written as
    ``*****``), means the run diverged: it is a :class:`SteadyStateError`. A NaN must
    not reach the outcome, where a velocity RMS of NaN reads as "channel not output"
    and the run would be judged settled.
    """
    if not out_path.is_file():
        msg = f"steady-state output file not found: {out_path}"
        raise SteadyStateError(msg)
    lines = out_path.read_text(errors="ignore").splitlines()
    header_index = next(
        (i for i, line in enumerate(lines) if line.split()[:1] == ["Time"]), None
    )
    if header_index is None:
        msg = f"no channel header (a 'Time' row) found in {out_path}"
        raise SteadyStateError(msg)
    columns = lines[header_index].split()
    rows: list[list[float]] = []
    for number, line in enumerate(lines[header_index + 2 :], start=header_index + 3):
        tokens = line.split()  # the units row is skipped above
        if len(tokens) != len(columns):
            continue
        try:
            rows.append([float(token) for token in tokens])
        except ValueError as error:
            msg = (
                f"line {number} of {out_path} holds a value that is not numeric (an "
                f"overflowed output field?): the settle run diverged: {line.strip()!r}"
            )
            raise SteadyStateError(msg) from error
    if not rows:
        msg = f"no data rows found in {out_path}"
        raise SteadyStateError(msg)
    data = np.asarray(rows, dtype=np.float64)
    time = data[:, columns.index("Time")]
    steady = data[time >= discard_time]
    if steady.shape[0] == 0:
        steady = data
    bad = ~np.isfinite(steady)
    if bad.any():
        channels = sorted({columns[i] for i in np.flatnonzero(bad.any(axis=0))})
        msg = (
            f"the settle output {out_path} holds non-finite values in its analysis "
            f"window ({', '.join(channels)}): the settle run diverged"
        )
        raise SteadyStateError(msg)
    return columns, steady


def _platform_from_means(means: dict[str, float]) -> PlatformPosition | None:
    """Build a PlatformPosition from the platform channels actually measured.

    Only the DOFs whose channel is present are set; the rest are left ``None`` so a
    partial platform output (e.g. only ``PtfmSurge``) never overwrites the deck's other
    platform DOFs with manufactured zeros. Returns ``None`` when no platform channel was
    output (an onshore deck).
    """
    measured = {
        attr: means[channel]
        for attr, channel in _PLATFORM_CHANNELS.items()
        if channel in means
    }
    if not measured:
        return None
    return PlatformPosition(**measured)


def _inflow_path(deck: FastInputFile) -> Path | None:
    """Return the deck's usable InflowWind file path, or ``None`` if there is none."""
    if "InflowFile" not in deck:
        return None
    try:
        path = deck.resolve("InflowFile")
    except FastInputError:
        return None
    return path if path.is_file() else None


def _edit_subfile(
    deck: FastInputFile, key: str, edit: Callable[[FastInputFile], None]
) -> None:
    """Apply ``edit`` to the sub-file referenced by ``key``, if it is present."""
    if key not in deck:
        return
    try:
        path = deck.resolve(key)
    except FastInputError:
        return
    if not path.is_file():
        return
    sub_file = FastInputFile.read(path)
    edit(sub_file)
    sub_file.write()


def _set_if_present(
    deck: FastInputFile, keyword: str, value: str | int | float | bool
) -> None:
    """Set ``keyword`` only when the deck contains it (tolerate deck variants)."""
    if keyword in deck:
        deck.set(keyword, value)
