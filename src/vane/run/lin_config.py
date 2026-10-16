# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Configure an OpenFAST deck for a linearization at one operating point.

Given a primary ``.fst`` deck and an :class:`~vane.run.operating_points.OperatingPoint`,
this sets the linearization controls in the primary file and the operating point in
the ElastoDyn and InflowWind sub-files, following the OpenFAST recommended practice:

* enable ``Linearize`` and write outputs/inputs at high precision (``OutFmt``);
* for a *rotating* point, trim to a periodic steady state (``CalcSteady = True``)
  with ``NLinTimes`` azimuth snapshots, choosing torque trim below rated wind speed
  and pitch trim at/above it (``TrimCase`` 2 vs 3) — the standard region split;
* when explicit absolute times are supplied, disable ``CalcSteady`` and linearize at
  those times; otherwise a *standstill / parked* point takes one snapshot at time zero;
* set the rotor speed and blade pitch in ElastoDyn and the steady wind speed in
  InflowWind, with uniform inflow (power-law shear exponent ``PLExp = 0``) by default
  -- the same steady uniform wind the settle run
  (:func:`~vane.run.solve_steady_state`) uses, so a settled operating point is
  linearized in the inflow it was settled in.

For a full floating (aero-hydro-servo-elastic) deck,
:meth:`LinearizationSettings.offshore` additionally makes the offshore modules
linearization-compatible — still water in SeaState, state-space hydrodynamics in
HydroDyn with the 2nd-order forces off,
quasi-steady aero with no dynamic wake in AeroDyn, and a trim-compatible ServoDyn (the
Bladed-style DLL controller is not linearizable, so CalcSteady's trim replaces it, with
region-aware generator torque — zero below rated, constant above). For a trimmed
rotating point the generator DOF (``GenDOF``) is also ensured on, since the trim
regulates rotor speed through it. These edits are skipped for any module a deck does not
have, so an onshore deck is unaffected.

The deck and its sub-files are edited **in place** with the format-preserving
:class:`~vane.io.FastInputFile`. For a multi-point sweep, copy the case to a
per-operating-point directory first so that shared sub-files (e.g. a common
InflowWind deck) are not mutated across points — the linearization driver does this.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from itertools import pairwise
from typing import TYPE_CHECKING

from vane.io.fst_reader import fortran_float
from vane.io.fst_writer import FastInputError, FastInputFile
from vane.utils.paths import resolve_path

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from vane.run.operating_points import OperatingPoint

__all__ = [
    "LinearizationSettings",
    "configure_linearization",
    "is_size_scaled_default",
    "needs_rated_wind_speed",
    "require_inputs_within",
]

# ElastoDyn structural-DOF flags disabled by default for a reliable linearization:
#   * YawDOF -- the free nacelle-yaw rigid-body mode. There is no yaw controller during
#     a linearization (the DLL is replaced by the trim), so leaving it free adds a
#     spurious zero-frequency rigid-body mode; disabling it is standard practice.
#   * DrTrDOF -- the drivetrain torsional-flexibility DOF (~0.62 Hz drivetrain-torsion
#     mode). Disabling it drops that mode from the linearization but markedly improves
#     CalcSteady trim convergence; this is a deliberate, documented tradeoff, fully
#     overridable for a study that needs the drivetrain-torsion mode.
#   * The blade and tower flexibility DOFs (FlapDOF1/2, EdgeDOF, TwFADOF1/2,
#     TwSSDOF1/2) are enabled: a deck shipped with them off (the published IEA 3.4-MW
#     deck has rigid blades) would otherwise linearize without its blade modes and with
#     tower modes stiffened by a rigid rotor, which no Campbell diagram can reveal.
FLEXIBILITY_DOFS: tuple[str, ...] = (
    "FlapDOF1",
    "FlapDOF2",
    "EdgeDOF",
    "TwFADOF1",
    "TwFADOF2",
    "TwSSDOF1",
    "TwSSDOF2",
)
_DEFAULT_STRUCTURAL_DOFS: dict[str, bool] = {
    **dict.fromkeys(FLEXIBILITY_DOFS, True),
    "YawDOF": False,
    "DrTrDOF": False,
}

# Below rated, the CalcSteady trim regulates rotor speed via generator torque; the gain
# ``TrimGain`` (Nm/(rad/s)) must be large enough or the trim never settles (Jonkman's
# most common non-convergence cause). His rule of thumb is ~a tenth of the rated
# generator torque, so the torque trim gain is derived from ``rated_generator_torque``
# with this fraction when given.
_TORQUE_TRIM_GAIN_FRACTION = 0.1

# Flattening ServoDyn's simple variable-speed law (VSContrl=1) to a constant torque:
# a tiny but strictly positive rated speed VS_RtGnSp and slip VS_SlPc (ServoDyn requires
# both > 0) with zero region-2 gain VS_Rgn2K, so torque == VS_RtTq at every speed (per
# Jonkman, OpenFAST issue #2985). VS_Rgn2K = 0 also makes ServoDyn's init-validation
# product VS_Rgn2K*VS_RtGnSp^2 exactly 0, satisfying its `<= VS_RtTq` check for any
# VS_RtTq >= 0 — which lets a below-rated trim neutralize a DLL-style dummy deck
# (VS_RtTq = 0) without a rated torque, the generator being held off regardless.
_SIMPLE_VS_SPEED_SENTINEL = 9.999e-9  # VS_RtGnSp / VS_SlPc: tiny but strictly positive
_REGION2_GAIN_OFF = 0.0  # VS_Rgn2K: no region-2 quadratic torque
_DEFAULT_TIME_BEYOND_TMAX = 9.999e4  # s; exceeds any realistic linearization TMax

# Shortest TMax (s) a CalcSteady trim is given when ``sim_time`` is not set. Template
# decks often carry a placeholder TMax of a few seconds; CalcSteady then reaches TMax
# before the rotor response is periodic and OpenFAST forces a single linearization.
# OpenFAST ends the run as soon as the trim converges, so the floor costs nothing for a
# point that converges early (a land-based turbine typically needs a few minutes).
MIN_CALC_STEADY_TMAX = 600.0

# InflowWind power-law shear exponent (PLExp) of steady uniform inflow. Both the settle
# run and the linearization write it by default, so the operating point is settled and
# linearized in the same (unsheared) wind.
UNIFORM_INFLOW_SHEAR_EXPONENT = 0.0

# Largest CalcSteady trim tolerance (TrimTol) VANE will configure. TrimTol bounds the
# normalized change of the trimmed response between successive rotor revolutions, so a
# value near 1 accepts revolutions that differ by the size of the response itself -- a
# near-vacuous periodicity test that can stop on a transient and still be recorded as
# converged. 0.1 (10 %) is an order of magnitude above the loosened tolerances the
# validated floating recipe needs (about 0.015) and the above-rated tuning floor (0.01),
# yet still distinguishes a periodic steady state from a drifting one. The retry ladder
# clamps its escalation here; an explicit larger setting is rejected.
MAX_TRIM_TOL = 0.1


class _SizeScaledDefault(float):
    """A trim default for a 63 m rotor that the sweep configuration scales.

    ``trim_tol`` and ``trim_gain_pitch`` default to instances of this ``float``
    subclass, so the automatic configuration can tell a value left at its default
    (scaled with the rotor radius) from the same number passed explicitly (a plain
    ``float``, always kept). Arithmetic on it returns a plain ``float``.
    """

    __slots__ = ()


def is_size_scaled_default(value: float) -> bool:
    """Return whether a trim setting is still the class default (not user-given).

    Parameters
    ----------
    value : float
        ``LinearizationSettings.trim_tol`` or ``trim_gain_pitch``.

    Returns
    -------
    bool
        ``True`` when the value is the untouched class default, which the automatic
        sweep configuration scales with the rotor radius; ``False`` for any value
        passed explicitly, even one equal to the default.
    """
    return isinstance(value, _SizeScaledDefault)


# ServoDyn control switches whose value 5 routes that control through the Bladed-style
# controller library (``DLL_FileName``): pitch, generator torque, HSS brake, yaw,
# active flaps and cable control. The one list every module reads.
DLL_CONTROL_KEYS = ("PCMode", "VSContrl", "HSSBrMode", "YCMode", "AfCmode", "CCmode")

# Loosest TrimTol that keeps the trimmed rotor speed periodic within the 1 % spread the
# MBC3 transform accepts on the reference-turbine campaign: at 1e-2 the IEA 3.4-MW below
# rated already stops with a 1.0-2.6 % spread, and at 0.021 the IEA 15-MW (monopile at
# 4 m/s, semi-submersible at 6 m/s) stops with up to 6.6 %. A trim that runs out of
# time is given more time rather than a tolerance beyond this.
PERIODIC_TRIM_TOL = 1e-2

# CalcSteady trim modes for an at/above-rated point (see ``above_rated_trim``).
_ABOVE_RATED_TRIM_MODES = frozenset({"pitch", "torque"})

_BLADE_PITCH_KEYS = ("BlPitch(1)", "BlPitch(2)", "BlPitch(3)")
_PITCH_NEUTRAL_KEYS = ("PitNeut(1)", "PitNeut(2)", "PitNeut(3)")

# Linearizable values for the OpenFAST module switches a linearization may set. ``None``
# (leave the deck unchanged) is always allowed and is checked separately.
_LIN_SWITCHES = frozenset({0, 1, 2})  # LinInputs / LinOutputs
_LINEARIZABLE_WAVE_MODS = frozenset({0})  # still water only
_LINEARIZABLE_RDTN_MODS = frozenset({0, 2})  # none or state-space (not convolution)
_LINEARIZABLE_EXCTN_MODS = frozenset({0, 2})  # none or state-space
_LINEARIZABLE_UA = frozenset({0, 4, 5, 6})  # quasi-steady or a linearizable UA model
_LINEARIZABLE_DBEMT = frozenset(
    {-1, 0, 3}
)  # frozen, none, or the continuous-state DBEMT
_LINEARIZABLE_GEN_MODELS = frozenset({1, 2})  # ServoDyn GenModel: simple or Thevenin


def _check_choice(name: str, value: int | None, allowed: frozenset[int]) -> None:
    """Reject ``value`` unless it is ``None`` or one of the ``allowed`` choices."""
    if value is not None and value not in allowed:
        choices = ", ".join(str(choice) for choice in sorted(allowed))
        msg = f"{name} must be one of {{{choices}}} for a linearization, got {value}"
        raise ValueError(msg)


@dataclass
class LinearizationSettings:
    """Tunable settings for configuring a linearization deck.

    Parameters
    ----------
    n_lin_times : int
        Number of linearizations per rotor revolution for a rotating point
        (``NLinTimes``); 36 (every 10 degrees of azimuth) is the OpenFAST
        recommendation.
    calc_steady : bool
        Whether a rotating point trims to a periodic steady state before
        linearizing (``CalcSteady``). A standstill/parked point always uses a
        single linearization regardless of this flag.
    trim_tol : float
        Rotor-speed convergence tolerance for the trim (``TrimTol``). Must be positive
        and at most ``MAX_TRIM_TOL`` (``0.1``): a larger tolerance no longer
        distinguishes a periodic steady state from a drifting transient. The default
        ``1e-3`` applies to a 63 m rotor; the automatic sweep configuration scales a
        value left at this default with the rotor radius
        (:func:`vane.run.auto_config.scaled_trim_tol`).
    trim_gain_torque : float
        Proportional trim gain used below rated wind speed (torque trim, Nm/(rad/s)).
        Used only when ``rated_generator_torque`` is not given; must be positive (as
        must ``trim_gain_pitch``). The default ``100.0`` is
        far too low for a utility-scale turbine — per Jonkman the torque trim gain
        should be ~a tenth of the rated generator torque (e.g. ~4000 for the NREL 5-MW),
        so supplying ``rated_generator_torque`` (which derives the gain) is strongly
        preferred over relying on this default.
    trim_gain_pitch : float
        Trim gain used at/above rated wind speed (pitch trim, rad/(rad/s)). ServoDyn's
        CalcSteady trim is a PI law: each time step adds ``TrimGain`` times the
        rotor-speed error to an integrator, and the pitch offset adds 1000 times that
        error on top, so the effective proportional gain is ``1000 * TrimGain`` and
        the integral gain ``TrimGain / DT``. The default ``2e-4`` applies to a 63 m
        rotor (the NREL 5-MW, which oscillates or motors above rated at ``1e-3``); the
        automatic sweep configuration scales a value left at this default with the
        rotor radius (:func:`vane.run.auto_config.scaled_trim_gain_pitch`).
    above_rated_trim : {"pitch", "torque"}
        How CalcSteady trims an at/above-rated point. ``"pitch"`` (default, the
        documented OpenFAST practice) holds the generator torque constant and trims
        blade pitch. ``"torque"`` holds the blade pitch at the operating point's value
        (the user's rotor-performance curve) and trims generator torque with the torque
        gain, exactly as below rated. Torque trim converges where a pitch trim
        oscillates (large rotors near rated, e.g. the IEA 15- and 22-MW); the
        linearized plant is the same open-loop model at the curve's pitch and rotor
        speed, and the trimmed torque's departure from rated measures how far the
        curve and the deck disagree (see :func:`~vane.run.verify_operating_points`).
        The retry ladder switches to ``"torque"`` as its final fallback. ServoDyn's
        flattened law stays engaged: in its region 3 the torque is the constant
        ``VS_RtTq`` with zero speed derivative, and the trim adds its offset on top, so
        the generator torque is one constant (``VS_RtTq`` + offset) exactly as below
        rated (0 + offset), the linearized plant is unchanged, and the trim starts from
        rated torque rather than from zero.
    rated_generator_torque : float or None
        Rated generator torque in N·m. When given, the **torque** trim gain is derived
        from it as ``rated_generator_torque / 10`` (Jonkman's rule of thumb), overriding
        ``trim_gain_torque``; this is the dependable way to get a torque trim that
        converges below rated. It also supplies the constant generator torque
        (``VS_RtTq``) the above-rated pitch trim needs when ``trim_controller`` is on.
        ``None`` (default) falls back to ``trim_gain_torque`` for the gain and to the
        deck's ``VS_RtTq`` for the torque -- but only when the deck's original
        ``VSContrl`` is ``1`` (the simple variable-speed law actually in use); a DLL or
        other controller deck carries a placeholder ``VS_RtTq``, so an above-rated
        trimmed point on it *requires* this value. This value is preserved across retry
        attempts.
    trim_gain_torque_override : float or None
        An explicit, already-resolved torque trim gain (Nm/(rad/s)) that takes
        precedence over both ``rated_generator_torque`` and ``trim_gain_torque`` in
        :meth:`resolved_trim_gain_torque`. Set by the retry ladder
        (:class:`LinearizationRetryPolicy`) to pin an escalated gain **without**
        discarding ``rated_generator_torque`` (which still drives ``VS_RtTq``); normally
        left ``None``.
    rated_wind_speed : float or None
        Wind speed (m/s) separating the below-rated (torque-trim) and above-rated
        (pitch-trim) control regions of the turbine being linearized (e.g. 11.4 for
        the NREL 5-MW, about 10.6 for the IEA 15-MW). There is no turbine-independent
        default: it is required for any rotating point whose trim or controller
        handling depends on the region (``calc_steady`` or ``trim_controller``), and a
        clear error is raised when it is missing.
    lin_inputs : int
        ``LinInputs`` switch (0=none, 1=standard, 2=all).
    lin_outputs : int
        ``LinOutputs`` switch (0=none, 1=from OutList, 2=all). ``1`` is required for
        the load/QoI channels to appear in ``C``/``D``.
    module_linearizations : bool
        Set ``LinOutMod`` so OpenFAST writes per-module linearization files alongside
        the full-system files. This is required for module-specific physics models,
        including MoorDyn driven directly by measured fairlead kinematics.
    full_jacobians : bool
        Set ``LinOutJac`` so OpenFAST also writes the unassembled module Jacobians.
        OpenFAST requires ``lin_inputs=lin_outputs=2`` when this debug export is
        enabled.  It is useful for auditing the exact module coupling paths used by a
        subsystem digital twin.
    out_fmt : str
        Output number format (``OutFmt``); a high precision such as ``"ES20.11E3"``
        is important for accurate eigenvalues.
    twr_kdmp, bld_kdmp : float
        Tower and blade artificial damping used only during the steady-state trim.
    sim_time : float or None
        ``TMax`` to set. ``None`` leaves the deck's value unchanged, except that a
        CalcSteady trim is given at least ``MIN_CALC_STEADY_TMAX`` (600 s): a
        template deck's placeholder TMax would end the trim before it converges.
        OpenFAST stops as soon as the trim converges, so the floor costs nothing.
    linearization_times : tuple[float, ...] or None
        Absolute OpenFAST times in seconds at which to linearize when
        ``calc_steady=False``. The values must be finite, non-negative, strictly
        increasing, no later than ``sim_time`` when it is supplied, and their count
        must equal ``n_lin_times``. ``None`` retains the single-snapshot untrimmed
        behaviour. This path is intended for a rotating plant that has already been
        demonstrated to be at periodic equilibrium by an independent physical check.
    wave_mod : int or None
        SeaState ``WaveMod`` to set (``0`` = still water, required for
        linearization). ``None`` leaves the deck unchanged. Setting still water also
        neutralises the wave options it leaves meaningless but that OpenFAST would
        reject: ``WaveDirMod`` and ``ConstWaveMod`` are set to 0 (required for any
        linearization), a wave-stretching model other than 0/1 (``WaveStMod``) is set
        to 0, MacCamy-Fuchs diffraction (``MCFD``) to 0, and a ``DEFAULT``
        sub-surface current heading (``CurrSSDir``) is replaced by the deck's
        ``WaveDir`` so the current keeps its direction.
    hydro_radiation_mode, hydro_excitation_mode : int or None
        HydroDyn ``RdtnMod`` / ``ExctnMod`` to set. Linearization rejects the
        convolution/DFT models, so use ``0`` (none) or ``2`` (state-space, which
        needs the ``.ss`` / ``.ssexctn`` files). ``None`` leaves the deck unchanged.
        When either is set and HydroDyn then requires ``RdtnDT`` to equal the
        glue-code ``DT`` (WAMIT potential flow with a radiation or state-space
        excitation model) but the deck holds another numeric step, ``RdtnDT`` is set
        to ``DEFAULT``.
    hydro_radiation_fit : bool
        Generate a missing ``<PotFile>.ss`` radiation file in each linearization case
        copy from the WAMIT ``.1`` data (:func:`vane.run.ensure_radiation_state_space`)
        before the case is validated and run; the template is never written, and the
        fit is recorded per case in the run provenance. Requires
        ``hydro_radiation_mode = 2``. Default ``False``.
        :func:`~vane.run.run_operating_sweep` sets it when its automatic configuration
        finds the file missing but the WAMIT data fittable within the acceptance limit.
    disable_second_order_hydro : bool
        Zero HydroDyn's 2nd-order force switches (``MnDrift``/``NewmanApp``/
        ``DiffQTF``/``SumQTF``) and switch off SeaState's 2nd-order wave kinematics
        (``WvDiffQTF``/``WvSumQTF``), which linearization does not support.
    unsteady_aero_mode : int or None
        AeroDyn ``UA_Mod`` to set; linearization requires ``0`` (quasi-steady) or
        ``4``/``5``/``6``. ``None`` leaves the deck unchanged. On a legacy-format
        AeroDyn deck (``AFAeroMod``/``UAMod``/``WakeMod`` keywords), ``0`` sets
        ``AFAeroMod = 1`` (steady airfoil aerodynamics), the only value that format
        can linearize; a linearizable unsteady model (``4``--``6``) cannot be expressed
        there and is reported by :func:`~vane.run.validate_deck`.
    dynamic_wake_mode : int or None
        AeroDyn ``DBEMT_Mod`` to set; linearization requires ``-1`` (frozen),
        ``0`` (none), or ``3``. ``None`` leaves the deck unchanged. On a legacy-format
        AeroDyn deck, ``-1``/``0`` replace a dynamic-BEMT wake (``WakeMod = 2``, not
        linearizable in that format) with plain BEMT (``WakeMod = 1``) and set the
        ``FrozenWake`` flag to ``True``/``False`` respectively; ``3`` cannot be
        expressed there and is reported by :func:`~vane.run.validate_deck`.
    ua_end_radius : float or None
        AeroDyn ``UAEndRad`` to set: the blade-radius fraction beyond which unsteady
        aerodynamics is switched off, in ``(0, 1]``. ``None`` (the default) leaves the
        deck unchanged. AeroDyn keeps unsteady aerodynamics on every node whose radius
        fraction is not *greater* than ``UAEndRad``, so ``UAEndRad = 1`` keeps it on at
        the blade-tip node. With the Prandtl tip loss model that node's inflow is
        fixed (full axial induction), and its unsteady-aerodynamics states can turn NaN
        or infinite in the linearization. A value below 1 (e.g. ``0.99``) ends
        unsteady aerodynamics inboard of the tip. The AeroDyn file must contain a
        ``UAEndRad`` line; a modern-format file without one is rejected before any file
        is written. Legacy-format files get the value only when they have the line.
    trim_controller : bool
        Make ServoDyn trim-compatible (``PCMode = 0``, ``VSContrl = 1``,
        ``HSSBrMode = 0``, ``YCMode = 0``, ``GenTiStr = GenTiStp = True``, and a
        ``GenModel`` other than 1/2 reset to 1): the Bladed-style DLL controller is not
        linearizable, so CalcSteady's own trim replaces it. For a trimmed rotating point
        the generator torque is then set up per region: below rated the generator is
        kept off (``GenTiStr``/``TimGenOn`` beyond TMax) so the torque trim owns it;
        at/above rated the simple variable-speed law is flattened to a constant torque
        (``VS_RtGnSp``/``VS_Rgn2K``/``VS_SlPc`` tiny, ``VS_RtTq`` set from
        ``rated_generator_torque`` when given, else the deck's existing ``VS_RtTq``) so
        the pitch trim converges. Flattening requires a positive constant torque; an
        above-rated point with neither ``rated_generator_torque`` nor a positive deck
        ``VS_RtTq`` is rejected (it would otherwise trim against zero torque). The deck
        ``VS_RtTq`` is trusted only when the deck's original ``VSContrl`` is ``1``:
        a DLL-controlled deck (``VSContrl = 5``) or any other non-simple law carries a
        placeholder (e.g. ``9999.9``), so an above-rated point on such a deck without
        ``rated_generator_torque`` is rejected rather than trimmed at a wrong torque.
    wind_shear_exponent : float or None
        InflowWind power-law shear exponent (``PLExp``) to write. The default ``0.0``
        is steady **uniform** inflow, the convention of the settle run
        (:func:`~vane.run.solve_steady_state`), so a settled point is linearized in the
        same inflow it settled in; :func:`~vane.run.run_operating_sweep` passes this
        value to its settle runs too. ``None`` leaves the deck's ``PLExp`` unchanged.
        Recorded with the other settings in the run provenance.
    structural_dofs : dict[str, bool] or None
        ElastoDyn structural-DOF flags to set in the ED sub-file (e.g.
        ``{"YawDOF": False, "DrTrDOF": False}``). ``None`` (the default) applies the
        linearization defaults: the blade and tower flexibility DOFs
        (``FLEXIBILITY_DOFS``: ``FlapDOF1``, ``FlapDOF2``, ``EdgeDOF``, ``TwFADOF1``,
        ``TwFADOF2``, ``TwSSDOF1``, ``TwSSDOF2``) are **enabled**, so a deck shipped
        with rigid blades or tower still linearizes its blade and tower modes, and
        ``YawDOF`` and ``DrTrDOF`` are **disabled**.
        Disabling ``YawDOF`` removes the spurious free nacelle-yaw rigid-body mode
        (there is no yaw controller in a linearization); disabling ``DrTrDOF`` drops the
        drivetrain-torsion mode (~0.62 Hz) but improves CalcSteady trim convergence — a
        deliberate tradeoff. Supply a mapping to override: it is **merged on top of**
        the defaults, so ``{"DrTrDOF": True}`` re-enables the drivetrain-torsion mode
        while still disabling ``YawDOF``, and ``{}`` keeps only the defaults. A flag the
        user explicitly lists but the ED deck lacks is an error; a default flag the deck
        lacks is silently skipped (the deck variant simply has no such DOF).

    Notes
    -----
    The module fields default to *leave the deck unchanged* (``None`` / ``False``),
    so an onshore or already-prepared deck is unaffected — except ``structural_dofs``,
    whose ``None`` default disables ``YawDOF``/``DrTrDOF`` for a reliable
    linearization. For a full floating (aero-hydro-servo-elastic) deck, build the
    linearization-compatible set in one call with :meth:`offshore`.
    """

    n_lin_times: int = 36
    calc_steady: bool = True
    trim_tol: float = _SizeScaledDefault(1e-3)
    trim_gain_torque: float = 100.0
    trim_gain_pitch: float = _SizeScaledDefault(2e-4)
    rated_wind_speed: float | None = None
    lin_inputs: int = 1
    lin_outputs: int = 1
    module_linearizations: bool = False
    full_jacobians: bool = False
    out_fmt: str = "ES20.11E3"
    twr_kdmp: float = 0.0
    bld_kdmp: float = 0.0
    sim_time: float | None = None
    linearization_times: tuple[float, ...] | None = None
    wave_mod: int | None = None
    hydro_radiation_mode: int | None = None
    hydro_excitation_mode: int | None = None
    disable_second_order_hydro: bool = False
    unsteady_aero_mode: int | None = None
    dynamic_wake_mode: int | None = None
    trim_controller: bool = False
    wind_shear_exponent: float | None = UNIFORM_INFLOW_SHEAR_EXPONENT
    structural_dofs: dict[str, bool] | None = field(default=None)
    rated_generator_torque: float | None = None
    trim_gain_torque_override: float | None = None
    above_rated_trim: str = "pitch"
    ua_end_radius: float | None = None
    hydro_radiation_fit: bool = False

    def __post_init__(self) -> None:
        """Reject settings that OpenFAST linearization cannot accept.

        Catches a misconfigured analysis at construction — before a deck is copied or
        OpenFAST is launched — rather than letting the run fail late: the linearization
        switches must select linearizable models, the counts and tolerances must be
        physical, and the output precision must be set.

        Raises
        ------
        ValueError
            If any field is out of its valid range for a linearization.
        """
        # Finite first: a NaN slips through every ``<= 0`` / ``< 0`` comparison below
        # (``NaN <= 0`` is ``False``) and would be written into a deck as a real number.
        finite_fields = {
            "trim_tol": self.trim_tol,
            "trim_gain_torque": self.trim_gain_torque,
            "trim_gain_pitch": self.trim_gain_pitch,
            "twr_kdmp": self.twr_kdmp,
            "bld_kdmp": self.bld_kdmp,
        }
        for fname, fvalue in finite_fields.items():
            if not math.isfinite(fvalue):
                msg = f"{fname} must be finite, got {fvalue}"
                raise ValueError(msg)
        if self.sim_time is not None and not math.isfinite(self.sim_time):
            msg = f"sim_time must be finite when set, got {self.sim_time}"
            raise ValueError(msg)
        if self.ua_end_radius is not None and (
            isinstance(self.ua_end_radius, bool)
            or not isinstance(self.ua_end_radius, int | float)
            or not math.isfinite(self.ua_end_radius)
            or not 0.0 < self.ua_end_radius <= 1.0
        ):
            msg = (
                "ua_end_radius must be a number in (0, 1] or None, got "
                f"{self.ua_end_radius!r}"
            )
            raise ValueError(msg)
        if self.wind_shear_exponent is not None and (
            isinstance(self.wind_shear_exponent, bool)
            or not isinstance(self.wind_shear_exponent, int | float)
            or not math.isfinite(self.wind_shear_exponent)
        ):
            msg = (
                "wind_shear_exponent must be a finite number or None, got "
                f"{self.wind_shear_exponent!r}"
            )
            raise ValueError(msg)
        # ``n_lin_times`` is an integer count but can arrive as a float from untyped
        # config; reject a non-finite or fractional value (which slips past ``< 1`` and
        # would be written as a non-integer ``NLinTimes``), then normalize a whole float
        # (e.g. ``36.0``) to ``int`` so ``[0.0] * n_lin_times`` later cannot raise.
        if not math.isfinite(self.n_lin_times):
            msg = f"n_lin_times must be finite, got {self.n_lin_times}"
            raise ValueError(msg)
        if self.n_lin_times != int(self.n_lin_times):
            msg = f"n_lin_times must be a whole number, got {self.n_lin_times}"
            raise ValueError(msg)
        self.n_lin_times = int(self.n_lin_times)
        if self.n_lin_times < 1:
            msg = f"n_lin_times must be >= 1, got {self.n_lin_times}"
            raise ValueError(msg)
        if self.above_rated_trim not in _ABOVE_RATED_TRIM_MODES:
            msg = (
                "above_rated_trim must be one of "
                f"{sorted(_ABOVE_RATED_TRIM_MODES)}, got {self.above_rated_trim!r}"
            )
            raise ValueError(msg)
        if self.trim_tol <= 0.0:
            msg = f"trim_tol must be positive, got {self.trim_tol}"
            raise ValueError(msg)
        # OpenFAST documents TrimGain as > 0 but does not check it: a zero gain never
        # trims and a negative one drives the rotor speed away from its target.
        for fname, fvalue in (
            ("trim_gain_torque", self.trim_gain_torque),
            ("trim_gain_pitch", self.trim_gain_pitch),
        ):
            if fvalue <= 0.0:
                msg = f"{fname} must be positive, got {fvalue}"
                raise ValueError(msg)
        if self.trim_tol > MAX_TRIM_TOL:
            msg = (
                f"trim_tol must not exceed {MAX_TRIM_TOL} (a looser CalcSteady "
                f"tolerance is not a periodicity test), got {self.trim_tol}"
            )
            raise ValueError(msg)
        if self.rated_wind_speed is not None and not (
            math.isfinite(self.rated_wind_speed) and self.rated_wind_speed > 0.0
        ):
            msg = (
                "rated_wind_speed must be finite and positive when set, got "
                f"{self.rated_wind_speed}"
            )
            raise ValueError(msg)
        if self.twr_kdmp < 0.0 or self.bld_kdmp < 0.0:
            msg = "twr_kdmp and bld_kdmp must be non-negative"
            raise ValueError(msg)
        if self.sim_time is not None and self.sim_time <= 0.0:
            msg = f"sim_time must be positive when set, got {self.sim_time}"
            raise ValueError(msg)
        if self.linearization_times is not None:
            if self.calc_steady:
                msg = "linearization_times requires calc_steady=False"
                raise ValueError(msg)
            times = tuple(float(value) for value in self.linearization_times)
            if len(times) != self.n_lin_times:
                msg = (
                    "linearization_times length must equal n_lin_times, got "
                    f"{len(times)} and {self.n_lin_times}"
                )
                raise ValueError(msg)
            if any(not math.isfinite(value) or value < 0.0 for value in times):
                msg = "linearization_times must be finite and non-negative"
                raise ValueError(msg)
            if any(later <= earlier for earlier, later in pairwise(times)):
                msg = "linearization_times must be strictly increasing"
                raise ValueError(msg)
            if self.sim_time is not None and times[-1] > self.sim_time:
                msg = (
                    "the final linearization time must not exceed sim_time, got "
                    f"{times[-1]} > {self.sim_time}"
                )
                raise ValueError(msg)
            self.linearization_times = times
        if self.rated_generator_torque is not None and (
            not math.isfinite(self.rated_generator_torque)
            or self.rated_generator_torque <= 0.0
        ):
            msg = (
                "rated_generator_torque must be finite and positive when set, got "
                f"{self.rated_generator_torque}"
            )
            raise ValueError(msg)
        if self.trim_gain_torque_override is not None and (
            not math.isfinite(self.trim_gain_torque_override)
            or self.trim_gain_torque_override <= 0.0
        ):
            msg = (
                "trim_gain_torque_override must be finite and positive when set, got "
                f"{self.trim_gain_torque_override}"
            )
            raise ValueError(msg)
        if not self.out_fmt.strip().strip('"'):
            msg = "out_fmt must be a non-empty output format"
            raise ValueError(msg)
        _check_choice("lin_inputs", self.lin_inputs, _LIN_SWITCHES)
        _check_choice("lin_outputs", self.lin_outputs, _LIN_SWITCHES)
        if self.full_jacobians and (self.lin_inputs != 2 or self.lin_outputs != 2):
            msg = "full_jacobians requires lin_inputs=2 and lin_outputs=2"
            raise ValueError(msg)
        _check_choice("wave_mod", self.wave_mod, _LINEARIZABLE_WAVE_MODS)
        _check_choice(
            "hydro_radiation_mode", self.hydro_radiation_mode, _LINEARIZABLE_RDTN_MODS
        )
        _check_choice(
            "hydro_excitation_mode",
            self.hydro_excitation_mode,
            _LINEARIZABLE_EXCTN_MODS,
        )
        if not isinstance(self.hydro_radiation_fit, bool):
            msg = (
                "hydro_radiation_fit must be a bool, got "
                f"{type(self.hydro_radiation_fit).__name__}"
            )
            raise ValueError(msg)  # noqa: TRY004
        if self.hydro_radiation_fit and self.hydro_radiation_mode != 2:
            msg = (
                "hydro_radiation_fit generates the state-space radiation file, so it "
                "requires hydro_radiation_mode=2, got "
                f"{self.hydro_radiation_mode!r}"
            )
            raise ValueError(msg)
        _check_choice("unsteady_aero_mode", self.unsteady_aero_mode, _LINEARIZABLE_UA)
        _check_choice("dynamic_wake_mode", self.dynamic_wake_mode, _LINEARIZABLE_DBEMT)
        if self.structural_dofs is not None:
            for key, value in self.structural_dofs.items():
                # Guard untyped config: keys must be non-empty strings, values bools.
                if not isinstance(key, str):
                    msg = f"structural_dofs keys must be strings, got {key!r}"
                    raise ValueError(msg)  # noqa: TRY004
                if not key.strip():
                    msg = f"structural_dofs keys must be non-empty, got {key!r}"
                    raise ValueError(msg)
                if not isinstance(value, bool):
                    msg = (
                        f"structural_dofs[{key!r}] must be a bool, got "
                        f"{type(value).__name__}"
                    )
                    raise ValueError(msg)  # noqa: TRY004

    def is_below_rated(self, wind_speed: float) -> bool:
        """Return whether ``wind_speed`` lies in the below-rated (torque-trim) region.

        Parameters
        ----------
        wind_speed : float
            Operating-point wind speed (m/s).

        Returns
        -------
        bool
            ``True`` below :attr:`rated_wind_speed`, ``False`` at or above it.

        Raises
        ------
        ValueError
            If :attr:`rated_wind_speed` is not set.
        """
        if self.rated_wind_speed is None:
            msg = (
                "rated_wind_speed is required to choose torque (below rated) or pitch "
                "(above rated) trim for a rotating operating point; set it for the "
                "turbine being linearized (e.g. 11.4 m/s for the NREL 5-MW)"
            )
            raise ValueError(msg)
        return wind_speed < self.rated_wind_speed

    def resolved_trim_gain_torque(self) -> float:
        """Return the torque trim gain (Nm/(rad/s)) to write into ``TrimGain``.

        An explicit ``trim_gain_torque_override`` (pinned by the retry ladder) takes
        precedence. Otherwise, when ``rated_generator_torque`` is given, the gain is
        derived from it as ``rated_generator_torque * 0.1`` (Jonkman's rule of thumb — a
        too-low torque trim gain is the most common cause of CalcSteady non-convergence
        below rated); otherwise the explicit ``trim_gain_torque`` is used.

        Returns
        -------
        float
            The torque trim gain to apply below rated wind speed.
        """
        if self.trim_gain_torque_override is not None:
            return self.trim_gain_torque_override
        if self.rated_generator_torque is not None:
            return self.rated_generator_torque * _TORQUE_TRIM_GAIN_FRACTION
        return self.trim_gain_torque

    def resolved_structural_dofs(self) -> dict[str, bool]:
        """Return the ElastoDyn DOF flags to apply, defaults merged with overrides.

        The linearization defaults (blade and tower flexibility enabled,
        ``YawDOF`` and ``DrTrDOF`` disabled) are the base; a user-supplied
        ``structural_dofs`` mapping is merged on top, so an override re-enables or adds
        flags while the un-overridden defaults remain.

        Returns
        -------
        dict[str, bool]
            The effective ElastoDyn DOF-flag map (flag name -> enabled).
        """
        resolved = dict(_DEFAULT_STRUCTURAL_DOFS)
        if self.structural_dofs is not None:
            resolved.update(self.structural_dofs)
        return resolved

    def explicit_structural_dofs(self) -> frozenset[str]:
        """Return the DOF-flag names the user explicitly requested (not just defaults).

        A user-listed flag must exist in the ED deck (its absence is an error); a flag
        present only via the defaults is skipped when the deck lacks it.

        Returns
        -------
        frozenset[str]
            The flag names supplied in ``structural_dofs`` (empty when ``None``).
        """
        if self.structural_dofs is None:
            return frozenset()
        return frozenset(self.structural_dofs)

    def snapshot(self) -> LinearizationSettings:
        """Return a fully independent copy, deep-copying the mutable fields.

        ``dataclasses.replace`` shallow-copies, so a plain ``replace`` snapshot would
        share the same ``structural_dofs`` dict object as the source. A later mutation
        of the source's (or the originally-supplied) dict would then also mutate the
        snapshot, breaking snapshot independence. This copies the mutable
        ``structural_dofs`` mapping so the returned settings are decoupled from any
        later mutation.

        Returns
        -------
        LinearizationSettings
            A copy whose mutable fields share no object with this instance.
        """
        dofs = None if self.structural_dofs is None else dict(self.structural_dofs)
        return replace(self, structural_dofs=dofs)

    @classmethod
    def offshore(cls, **overrides: object) -> LinearizationSettings:
        """Return settings prepared for a floating aero-hydro-servo-elastic deck.

        Enables the full linearization-compatible module set — still water, state-space
        hydrodynamics, quasi-steady aero with a frozen wake (``DBEMT_Mod = -1``, valid
        for linearization across OpenFAST versions), the 2nd-order forces off and the
        controller replaced by the trim — that lets a real FOWT deck linearize. No trim
        damping is set: ``Twr_Kdmp``/``Bld_Kdmp`` do work on the periodic blade and
        tower motion and bias the trimmed operating point (the IEA 10-MW's trimmed
        generator torque came out about 8 % low at 4 m/s with 5000 N/(m/s)), and on
        light blades they can destabilise the start of the trim. The retry ladder adds
        damping only when a trim keeps running out of time. Keyword ``overrides``
        replace any of these fields (e.g. ``twr_kdmp=5000.0, bld_kdmp=5000.0`` to trim
        with damping from the first attempt).

        Parameters
        ----------
        **overrides : object
            Field values to override on the returned settings (e.g.
            ``hydro_radiation_mode=0`` when no ``.ss`` file is available).

        Returns
        -------
        LinearizationSettings
            Settings with the offshore module configuration enabled.
        """
        base = cls(
            wave_mod=0,
            hydro_radiation_mode=2,
            hydro_excitation_mode=2,
            disable_second_order_hydro=True,
            unsteady_aero_mode=0,
            dynamic_wake_mode=-1,
            trim_controller=True,
        )
        for key, value in overrides.items():
            if not hasattr(base, key):
                msg = f"unknown LinearizationSettings field: {key!r}"
                raise TypeError(msg)
            setattr(base, key, value)
        base.__post_init__()  # re-validate after applying the overrides
        return base


# The primary-deck module switch that enables each sub-file a linearization may edit.
# A sub-file whose switch is explicitly 0 is inert: OpenFAST neither reads nor validates
# it, so VANE neither edits nor validates it either.
_SUBFILE_MODULE_SWITCH = {
    "AeroFile": "CompAero",
    "ServoFile": "CompServo",
    "HydroFile": "CompHydro",
    "SeaStFile": "CompSeaSt",
}


def _module_disabled(deck: FastInputFile, comp_key: str) -> bool:
    """Return whether the primary-deck module switch ``comp_key`` is explicitly 0.

    A switch that is absent (an older deck) or non-numeric counts as enabled, so the
    module is configured and validated rather than silently skipped.
    """
    switch = _deck_float(deck, comp_key)
    return switch is not None and switch == 0.0


def _subfile_disabled(deck: FastInputFile, key: str) -> bool:
    """Return whether the sub-file ``key`` belongs to a module the deck switches off."""
    comp_key = _SUBFILE_MODULE_SWITCH.get(key)
    return comp_key is not None and _module_disabled(deck, comp_key)


def _servo_module_disabled(deck: FastInputFile) -> bool:
    """Return whether ServoDyn is switched off in the primary deck (``CompServo = 0``).

    When off, ServoDyn is inert (OpenFAST does not run or validate it), so VANE skips
    configuring/validating the ServoFile even if the deck still points at one.
    """
    return _module_disabled(deck, "CompServo")


def _validate_operating_configuration(
    op: OperatingPoint, settings: LinearizationSettings, deck: FastInputFile
) -> None:
    """Reject (operating point, settings) combinations VANE cannot configure correctly.

    Called after the (non-mutating) deck read but before any deck is written, so a
    rejection never leaves a partially reconfigured case. Both ServoDyn-trim rejections
    are checked here (not deep inside the per-sub-file edits) for exactly that reason.

    Raises
    ------
    vane.io.FastInputError
        When ``trim_controller`` is on, ServoDyn is active, and the rotating point
        cannot be given a correct generator torque: **below rated** without CalcSteady
        (``calc_steady=False``) there is no torque trim and the steady torque is the
        region-2 law's rotor-performance-dependent value, which VANE does not compute
        (use ``calc_steady=True``); **above rated** the engaged generator needs a
        positive constant ``VS_RtTq`` (from ``rated_generator_torque`` or the deck) and
        none is available, or ``rated_generator_torque`` is absent and the deck's
        original ``VSContrl`` is not ``1`` (its ``VS_RtTq`` is then a placeholder that
        the engaged generator would apply). Also raised when explicit
        ``linearization_times`` extend past the effective ``TMax`` (``sim_time`` when
        set, else the deck's own value), since OpenFAST would end the run before the
        final requested snapshot. Also raised when ``ua_end_radius`` is set and the
        enabled, modern-format AeroDyn file has no ``UAEndRad`` line to write it to.
    """
    _check_ua_end_radius_line(deck, settings)
    if settings.linearization_times is not None:
        effective_tmax = (
            settings.sim_time
            if settings.sim_time is not None
            else _deck_float(deck, "TMax")
        )
        final_time = settings.linearization_times[-1]
        if effective_tmax is not None and final_time > effective_tmax:
            msg = (
                f"the final linearization time {final_time} s exceeds the effective "
                f"TMax {effective_tmax} s of {deck.path}; set sim_time or extend TMax"
            )
            raise FastInputError(msg)
    if (
        not settings.trim_controller
        or _servo_module_disabled(deck)
        or not op.is_rotating
    ):
        return
    below_rated = settings.is_below_rated(op.wind_speed)
    if below_rated and not settings.calc_steady:
        msg = (
            "a below-rated rotating point linearized without CalcSteady "
            "(calc_steady=False) has no torque trim to set the steady generator "
            "torque, and VANE does not compute the region-2 rotor performance to "
            "supply it; use calc_steady=True for a below-rated rotating operating point"
        )
        raise FastInputError(msg)
    if not below_rated:
        # Above rated the generator is engaged with a constant torque flattened from
        # VS_RtTq; reject up front (before any write) when no positive torque exists —
        # mirrors the backstop in _flatten_to_constant_torque without partial writes.
        servo = _read_subfile(deck, "ServoFile")
        if servo is not None and any(key in servo for key in _SIMPLE_VS_FIELDS):
            trust_deck_torque = _deck_uses_simple_vs_law(servo)
            if settings.rated_generator_torque is None and not trust_deck_torque:
                raise FastInputError(_untrusted_deck_torque_msg(servo))
            if _resolve_constant_torque(servo, settings, trust_deck_torque) is None:
                raise FastInputError(_NO_CONSTANT_TORQUE_MSG)


# Every sub-file a linearization may edit (the default confinement set).
_EDITED_SUBFILE_KEYS = (
    "EDFile",
    "InflowFile",
    "AeroFile",
    "ServoFile",
    "HydroFile",
    "SeaStFile",
)
# The sub-files a settle run may edit (ElastoDyn initial conditions, steady wind, still
# water, and ServoDyn's DLL_FileName when a controller library override is given);
# passed to :func:`require_inputs_within` for settle cases. ServoFile is edited only
# with the override, so callers drop it from the check otherwise.
SETTLE_EDITED_SUBFILE_KEYS = ("EDFile", "InflowFile", "SeaStFile", "ServoFile")


def _linearization_edited_subfile_keys(
    settings: LinearizationSettings, deck: FastInputFile
) -> tuple[str, ...]:
    """Return the sub-file keys :func:`configure_linearization` writes for ``settings``.

    ElastoDyn and InflowWind are always configured; the other modules only when the
    settings request a change there (each helper returns without writing otherwise)
    and the deck has the module enabled: a module switched off (``CompSeaSt``,
    ``CompHydro``, ``CompAero`` or ``CompServo`` = 0) leaves its file inert and
    untouched.
    """
    keys = ["EDFile", "InflowFile"]
    if settings.wave_mod is not None or settings.disable_second_order_hydro:
        keys.append("SeaStFile")
    if (
        settings.hydro_radiation_mode is not None
        or settings.hydro_excitation_mode is not None
        or settings.disable_second_order_hydro
    ):
        keys.append("HydroFile")
    if (
        settings.unsteady_aero_mode is not None
        or settings.dynamic_wake_mode is not None
        or settings.ua_end_radius is not None
    ):
        keys.append("AeroFile")
    if settings.trim_controller:
        keys.append("ServoFile")
    return tuple(key for key in keys if not _subfile_disabled(deck, key))


def needs_rated_wind_speed(
    settings: LinearizationSettings, deck_path: str | Path
) -> bool:
    """Return whether a rotating point on this deck needs ``rated_wind_speed``.

    The region split selects the CalcSteady trim case, and the ServoDyn generator
    handling under ``trim_controller``. It is therefore needed when ``calc_steady`` is
    on, or when ``trim_controller`` is on and the deck's ServoDyn is enabled; a disabled
    ServoDyn (``CompServo = 0``) is never configured, so it needs no region.

    Parameters
    ----------
    settings : LinearizationSettings
        The linearization settings.
    deck_path : str or pathlib.Path
        The primary ``.fst`` deck.

    Returns
    -------
    bool
        ``True`` when a rotating operating point requires the rated wind speed.
    """
    if settings.calc_steady:
        return True
    return settings.trim_controller and not _servo_module_disabled(
        FastInputFile.read(deck_path)
    )


def require_inputs_within(
    deck_path: str | Path,
    root: str | Path,
    *,
    keys: Sequence[str] | None = None,
) -> None:
    """Reject a deck whose edited sub-files resolve outside ``root``.

    Linearization and settle runs edit the primary deck and some of its sub-files. A
    sub-file referenced outside the per-case copy (e.g. ``../ElastoDyn.dat`` when the
    copy root is the deck's own directory) would be edited where it lives -- a file
    shared by every case and the template -- so such a deck is rejected before anything
    is written. A sub-file that is only read may stay shared.

    Parameters
    ----------
    deck_path : str or pathlib.Path
        The primary ``.fst`` deck.
    root : str or pathlib.Path
        Directory every edited input must resolve inside (the case copy).
    keys : Sequence[str] or None, optional
        The sub-file keywords that will be edited. ``None`` checks every file a
        linearization may edit (ElastoDyn, InflowWind, AeroDyn, ServoDyn, HydroDyn,
        SeaState), except the file of a module the deck switches off (``CompAero``,
        ``CompServo``, ``CompHydro`` or ``CompSeaSt`` = 0).

    Raises
    ------
    vane.io.FastInputError
        If the deck or an edited sub-file resolves outside ``root``.
    """
    boundary = resolve_path(root)
    deck = FastInputFile.read(deck_path)
    if not resolve_path(deck_path).is_relative_to(boundary):
        msg = f"deck {deck_path} is outside the case directory {boundary}"
        raise FastInputError(msg)
    if keys is None:
        # Every file a linearization may edit, except an inert (disabled) module's.
        keys = tuple(
            key for key in _EDITED_SUBFILE_KEYS if not _subfile_disabled(deck, key)
        )
    for key in keys:
        if key not in deck:
            continue
        try:
            path = deck.resolve(key)
        except FastInputError:
            continue
        if not resolve_path(path).is_relative_to(boundary):
            msg = (
                f"{key} of {deck.path} resolves to {path}, outside the case directory "
                f"{boundary}; configuring it would edit a file shared by every case. "
                "Set template_root to a directory that contains all referenced inputs"
            )
            raise FastInputError(msg)


def configure_linearization(
    deck_path: str | Path,
    operating_point: OperatingPoint,
    settings: LinearizationSettings | None = None,
    *,
    confine_to: str | Path | None = None,
) -> None:
    """Configure an OpenFAST deck for linearization at an operating point.

    Edits the primary ``.fst`` and its sub-files in place.

    Parameters
    ----------
    deck_path : str or pathlib.Path
        Path to the primary ``.fst`` deck (its sub-files are edited too).
    operating_point : OperatingPoint
        The operating point to set up.
    settings : LinearizationSettings or None, optional
        Linearization tuning; defaults are used when ``None``.
    confine_to : str or pathlib.Path or None, optional
        When given, every input these settings edit must resolve inside this
        directory (see :func:`require_inputs_within`); the sweep driver passes each
        case copy so a shared sub-file is never modified, while files that are only
        read may stay shared.

    Raises
    ------
    vane.io.FastInputError
        If a required keyword (e.g. ``Linearize``) is absent from the deck, the
        operating point requests a positive wind speed but the deck has no usable
        InflowWind file to set it in (which would silently linearize in still air), or
        an edited input lies outside ``confine_to``.

    Notes
    -----
    For a multi-point sweep, copy the case to a per-operating-point directory first
    so shared sub-files are not mutated across points. The template deck is assumed
    to already enable the structural/aero/control modules appropriate for the case
    (``CompElast``/``CompAero``/``CompServo``); this sets the linearization controls,
    the operating point, and — for a non-zero wind point — ``CompInflow`` so the
    configured wind is actually used.
    """
    settings = settings or LinearizationSettings()
    if operating_point.is_rotating and needs_rated_wind_speed(settings, deck_path):
        # Fails before any file is written when the region split is unknown.
        settings.is_below_rated(operating_point.wind_speed)
    if confine_to is not None:
        require_inputs_within(
            deck_path,
            confine_to,
            keys=_linearization_edited_subfile_keys(
                settings, FastInputFile.read(deck_path)
            ),
        )
    deck = FastInputFile.read(deck_path)
    # Reject unsupported (operating point, settings) combinations up front — after the
    # (non-mutating) read but before any deck is written — so a direct caller is never
    # left with a partially reconfigured case on a rejection.
    _validate_operating_configuration(operating_point, settings, deck)
    for keyword, requested in (
        ("LinOutJac", settings.full_jacobians),
        ("LinOutMod", settings.module_linearizations),
    ):
        if requested and keyword not in deck:
            msg = f"requested linearization export requires {keyword} in {deck.path}"
            raise FastInputError(msg)

    deck.set("Linearize", True)
    _set_if_present(deck, "LinInputs", settings.lin_inputs)
    _set_if_present(deck, "LinOutputs", settings.lin_outputs)
    _set_if_present(deck, "LinOutJac", settings.full_jacobians)
    _set_if_present(deck, "LinOutMod", settings.module_linearizations)
    _set_if_present(deck, "OutFmt", f'"{settings.out_fmt}"')
    _set_if_present(deck, "WrVTK", 0)
    if settings.sim_time is not None:
        _set_if_present(deck, "TMax", settings.sim_time)

    if operating_point.is_rotating and settings.calc_steady:
        _configure_calc_steady(deck, operating_point, settings)
    elif settings.linearization_times is not None:
        _configure_explicit_times(deck, settings.linearization_times)
    else:
        _configure_single(deck)

    # Match the inflow module to the operating point: enable it for a non-zero wind
    # point (with a usable wind file) so the configured wind is actually used, and
    # disable it for a zero-wind/standstill point so the case linearizes in true
    # still air rather than carrying aero-inflow states from an aero template.
    if operating_point.wind_speed > 0.0:
        if _inflow_path(deck) is None:
            msg = (
                f"operating point requests wind speed {operating_point.wind_speed} "
                f"m/s but {deck.path} has no usable InflowWind file to set it in; "
                "the linearization would run in still air"
            )
            raise FastInputError(msg)
        _set_if_present(deck, "CompInflow", 1)
    else:
        _set_if_present(deck, "CompInflow", 0)
    _keep_output_start_before_end(deck)
    deck.write()

    _configure_elastodyn(deck, operating_point, settings)
    _configure_inflow(deck, operating_point, settings.wind_shear_exponent)
    _configure_seastate(deck, settings)
    _configure_hydrodyn(deck, settings)
    _configure_aerodyn(deck, settings)
    _configure_servodyn(deck, operating_point, settings)


def _configure_calc_steady(
    deck: FastInputFile, op: OperatingPoint, settings: LinearizationSettings
) -> None:
    """Set the periodic-steady-state trim controls for a rotating point."""
    torque_trim = (
        settings.is_below_rated(op.wind_speed) or settings.above_rated_trim == "torque"
    )
    deck.set("CalcSteady", True)
    deck.set("TrimCase", 2 if torque_trim else 3)
    deck.set(
        "TrimGain",
        settings.resolved_trim_gain_torque()
        if torque_trim
        else settings.trim_gain_pitch,
    )
    deck.set("TrimTol", settings.trim_tol)
    _set_if_present(deck, "Twr_Kdmp", settings.twr_kdmp)
    _set_if_present(deck, "Bld_Kdmp", settings.bld_kdmp)
    deck.set("NLinTimes", settings.n_lin_times)
    if settings.sim_time is None:
        tmax = _deck_float(deck, "TMax")
        if tmax is not None and tmax < MIN_CALC_STEADY_TMAX:
            deck.set("TMax", MIN_CALC_STEADY_TMAX)
    # CalcSteady fills LinTimes itself; size the array to NLinTimes so the deck is
    # internally consistent.
    deck.set_array("LinTimes", [0.0] * settings.n_lin_times)


def _configure_single(deck: FastInputFile) -> None:
    """Set a single linearization for a standstill/parked point."""
    _set_if_present(deck, "CalcSteady", False)
    deck.set("NLinTimes", 1)
    deck.set_array("LinTimes", [0.0])


def _configure_explicit_times(
    deck: FastInputFile, linearization_times: tuple[float, ...]
) -> None:
    """Configure untrimmed linearizations at validated absolute times."""
    _set_if_present(deck, "CalcSteady", False)
    deck.set("NLinTimes", len(linearization_times))
    deck.set_array("LinTimes", linearization_times)


def _configure_elastodyn(
    deck: FastInputFile, op: OperatingPoint, settings: LinearizationSettings
) -> None:
    """Set the operating point, platform position, and structural DOF flags in ED.

    Raises
    ------
    vane.io.FastInputError
        If the operating point or a structural-DOF change is requested but the deck has
        no ElastoDyn sub-file to apply it to, or a structural-DOF flag the user listed
        explicitly is absent from the ED deck.
    """
    explicit = settings.explicit_structural_dofs()
    if "EDFile" not in deck:
        if explicit:
            msg = (
                f"structural_dofs requested {sorted(explicit)} but the deck has no "
                "ElastoDyn (EDFile) sub-file to set them in"
            )
            raise FastInputError(msg)
        return
    try:
        ed_path = deck.resolve("EDFile")
    except FastInputError:
        if explicit:
            raise
        return
    elastodyn = FastInputFile.read(ed_path)
    _set_if_present(elastodyn, "RotSpeed", op.rotor_speed_rpm)
    for key in _BLADE_PITCH_KEYS:
        _set_if_present(elastodyn, key, op.blade_pitch_deg)
    if op.platform is not None:
        platform = op.platform
        for key, value in (
            ("PtfmSurge", platform.surge),
            ("PtfmSway", platform.sway),
            ("PtfmHeave", platform.heave),
            ("PtfmRoll", platform.roll),
            ("PtfmPitch", platform.pitch),
            ("PtfmYaw", platform.yaw),
        ):
            if value is not None:  # leave unspecified DOFs at the deck's value
                _set_if_present(elastodyn, key, value)
    _apply_structural_dofs(elastodyn, settings, ed_path)
    # The trim regulates rotor speed through the generator DOF, so it must be free for a
    # trimmed rotating point; ensure it on unless the user explicitly set it.
    if op.is_rotating and settings.calc_steady and "GenDOF" not in explicit:
        _set_if_present(elastodyn, "GenDOF", True)
    elastodyn.write()


def _apply_structural_dofs(
    elastodyn: FastInputFile, settings: LinearizationSettings, ed_path: Path
) -> None:
    """Write the resolved ElastoDyn DOF flags, failing on an absent explicit flag.

    A flag the user explicitly listed in ``structural_dofs`` must exist in the deck;
    its absence is an error (the user asked for a change that cannot be made). A flag
    present only via the defaults is skipped when the deck lacks it.
    """
    explicit = settings.explicit_structural_dofs()
    for flag, enabled in settings.resolved_structural_dofs().items():
        if flag in elastodyn:
            elastodyn.set(flag, enabled)
        elif flag in explicit:
            msg = (
                f"structural_dofs requested {flag!r} but the ElastoDyn deck {ed_path} "
                "has no such DOF flag"
            )
            raise FastInputError(msg)


def _inflow_path(deck: FastInputFile) -> Path | None:
    """Return the deck's usable InflowWind file path, or ``None`` if there is none."""
    if "InflowFile" not in deck:
        return None
    try:
        path = deck.resolve("InflowFile")
    except FastInputError:
        return None
    return path if path.is_file() else None


def _configure_inflow(
    deck: FastInputFile, op: OperatingPoint, shear_exponent: float | None
) -> None:
    """Set the steady wind speed and shear in the InflowWind sub-file, if present.

    A zero wind speed is written too (still air), so the deck matches the requested
    operating point rather than retaining a previous non-zero wind field; only a
    missing or unusable ``InflowFile`` is skipped. ``shear_exponent`` is written to
    ``PLExp`` unless it is ``None``.
    """
    inflow_path = _inflow_path(deck)
    if inflow_path is None:
        return
    inflow = FastInputFile.read(inflow_path)
    _set_if_present(inflow, "WindType", 1)
    _set_if_present(inflow, "HWindSpeed", op.wind_speed)
    if shear_exponent is not None:
        _set_if_present(inflow, "PLExp", shear_exponent)
    inflow.write()


def _read_subfile(deck: FastInputFile, key: str) -> FastInputFile | None:
    """Return the sub-file referenced by ``key``, or ``None`` if it is not present.

    A missing keyword, an unresolvable path, or an absent file yields ``None`` (the deck
    variant simply does not have that module). The read does not mutate anything, so it
    is safe to use for validation before any deck is written.
    """
    if key not in deck:
        return None
    try:
        path = deck.resolve(key)
    except FastInputError:
        return None
    if not path.is_file():
        return None
    return FastInputFile.read(path)


def _edit_subfile(
    deck: FastInputFile, key: str, edit: Callable[[FastInputFile], None]
) -> None:
    """Apply ``edit`` to the sub-file referenced by ``key``, if it is present.

    A missing keyword, an unresolvable path, or an absent file is tolerated (the
    deck variant simply does not have that module), so the configuration is safe to
    apply to onshore as well as offshore decks. The file of a module the primary deck
    switches off (e.g. ``CompHydro = 0``) is inert and is left untouched.
    """
    if _subfile_disabled(deck, key):
        return
    sub_file = _read_subfile(deck, key)
    if sub_file is None:
        return
    edit(sub_file)
    sub_file.write()


def _configure_seastate(deck: FastInputFile, settings: LinearizationSettings) -> None:
    """Set still water and switch off 2nd-order wave kinematics for linearization."""
    wave_mod = settings.wave_mod
    second_order = settings.disable_second_order_hydro
    if wave_mod is None and not second_order:
        return

    def edit(sea_state: FastInputFile) -> None:
        if wave_mod is not None:
            _set_if_present(sea_state, "WaveMod", wave_mod)
            _make_still_water_consistent(sea_state)
        if second_order:
            for key in _SECOND_ORDER_WAVE_KEYS:
                _set_if_present(sea_state, key, False)

    _edit_subfile(deck, "SeaStFile", edit)


# SeaState 2nd-order wave-kinematics flags; linearization requires both False.
_SECOND_ORDER_WAVE_KEYS = ("WvDiffQTF", "WvSumQTF")
# SeaState wave-stretching models (WaveStMod) OpenFAST accepts in still water.
_STILL_WATER_WAVE_STRETCHING = frozenset({0, 1})


def _make_still_water_consistent(sea_state: FastInputFile) -> None:
    """Neutralise the SeaState wave options that still water leaves meaningless.

    With ``WaveMod = 0`` there are no incident waves, so directional spreading, a
    constrained (embedded) wave, wave stretching, and MacCamy-Fuchs diffraction have
    nothing to act on. OpenFAST nevertheless rejects some of them, either for any
    linearization (``WaveDirMod`` and ``ConstWaveMod`` must be 0) or in still water
    specifically (``WaveStMod`` must be 0 or 1, ``MCFD`` must be 0, and a ``DEFAULT``
    sub-surface current heading ``CurrSSDir`` -- which means "along the waves" -- is
    refused). They are reset here so the still-water request actually initializes. A
    ``DEFAULT`` current heading is replaced by the deck's ``WaveDir``, which preserves
    the current direction the deck described.
    """
    _set_if_present(sea_state, "WaveDirMod", 0)
    _set_if_present(sea_state, "ConstWaveMod", 0)
    stretching = _deck_float(sea_state, "WaveStMod")
    if stretching is not None and stretching not in _STILL_WATER_WAVE_STRETCHING:
        sea_state.set("WaveStMod", 0)
    diffraction = _deck_float(sea_state, "MCFD")
    if diffraction is not None and diffraction != 0.0:
        sea_state.set("MCFD", 0.0)
    if _current_heading_is_default(sea_state):
        wave_dir = _deck_float(sea_state, "WaveDir")
        sea_state.set("CurrSSDir", wave_dir if wave_dir is not None else 0.0)


def _current_heading_is_default(sea_state: FastInputFile) -> bool:
    """Return whether SeaState's sub-surface current heading follows the waves.

    ``CurrSSDir = DEFAULT`` aligns the sub-surface current with the incident waves; it
    is read only with the standard current model (``CurrMod = 1``) and is invalid in
    still water.

    Parameters
    ----------
    sea_state : vane.io.FastInputFile
        The SeaState input file.

    Returns
    -------
    bool
        ``True`` when ``CurrMod`` is 1 and ``CurrSSDir`` is ``DEFAULT``.
    """
    if _deck_float(sea_state, "CurrMod") != 1.0 or "CurrSSDir" not in sea_state:
        return False
    try:
        heading = sea_state.get("CurrSSDir")
    except FastInputError:
        return False
    return heading.strip("\"'").upper() == "DEFAULT"


def _configure_hydrodyn(deck: FastInputFile, settings: LinearizationSettings) -> None:
    """Set HydroDyn's radiation/excitation models and disable 2nd-order forces."""
    radiation = settings.hydro_radiation_mode
    excitation = settings.hydro_excitation_mode
    second_order = settings.disable_second_order_hydro
    if radiation is None and excitation is None and not second_order:
        return

    glue_dt = _deck_float(deck, "DT")

    def edit(hydro: FastInputFile) -> None:
        if excitation is not None:
            _set_if_present(hydro, "ExctnMod", excitation)
        if radiation is not None:
            _set_if_present(hydro, "RdtnMod", radiation)
        if (radiation is not None or excitation is not None) and (
            _radiation_step_mismatch(hydro, glue_dt)
        ):
            # A state-space model requested here makes HydroDyn insist on
            # RdtnDT == DT; DEFAULT is exactly the glue-code step.
            hydro.set("RdtnDT", '"DEFAULT"')
        if second_order:
            for key in _SECOND_ORDER_HYDRO_KEYS:
                _set_if_present(hydro, key, 0)

    _edit_subfile(deck, "HydroFile", edit)


# HydroDyn 2nd-order wave-force switches; linearization requires all of them 0.
_SECOND_ORDER_HYDRO_KEYS = ("MnDrift", "NewmanApp", "DiffQTF", "SumQTF")


def _radiation_step_mismatch(hydro: FastInputFile, glue_dt: float | None) -> bool:
    """Return whether HydroDyn will reject its ``RdtnDT`` against the glue ``DT``.

    With WAMIT potential flow (``PotMod = 1``), HydroDyn requires ``RdtnDT`` to equal
    the glue-code time step whenever a radiation model (``RdtnMod > 0``) or the
    state-space excitation model (``ExctnMod = 2``) is in use. ``DEFAULT`` always
    satisfies this; a numeric value must match ``DT``. Unknown values (absent keyword,
    unreadable ``DT``) are not reported.
    """
    if _deck_float(hydro, "PotMod") != 1.0 or glue_dt is None:
        return False
    radiation = _deck_float(hydro, "RdtnMod") or 0.0
    excitation = _deck_float(hydro, "ExctnMod") or 0.0
    if radiation <= 0.0 and excitation <= 1.0:
        return False
    step = _deck_float(hydro, "RdtnDT")  # None for DEFAULT (or absent)
    return step is not None and not math.isclose(step, glue_dt, rel_tol=1e-9)


def _is_legacy_aerodyn(aero: FastInputFile) -> bool:
    """Return whether an AeroDyn deck uses the legacy airfoil/wake keywords.

    Older AeroDyn input files select the airfoil model with ``AFAeroMod`` (1 = steady,
    2 = Beddoes-Leishman unsteady; linearization requires 1), the unsteady variant with
    ``UAMod``, and the wake with ``WakeMod`` (2 = dynamic BEMT and 3 = free vortex
    wake, neither linearizable), with frozen-wake linearization as the separate
    ``FrozenWake`` flag. Current files use ``UA_Mod``/``DBEMT_Mod`` instead.
    """
    return "AFAeroMod" in aero


# Legacy AeroDyn switch values (see ``_is_legacy_aerodyn``).
_LEGACY_STEADY_AIRFOIL = 1  # AFAeroMod: steady airfoil aerodynamics
_LEGACY_WAKE_BEMT = 1  # WakeMod: quasi-steady BEMT
_LEGACY_WAKE_DBEMT = 2  # WakeMod: dynamic BEMT (not linearizable)


def _check_ua_end_radius_line(
    deck: FastInputFile, settings: LinearizationSettings
) -> None:
    """Reject ``ua_end_radius`` on a modern AeroDyn file with no ``UAEndRad`` line.

    AeroDyn reads a missing ``UAEndRad`` as 1, so silently skipping the request would
    leave unsteady aerodynamics on at the blade tip.
    """
    if settings.ua_end_radius is None or _subfile_disabled(deck, "AeroFile"):
        return
    aero = _read_subfile(deck, "AeroFile")
    if aero is None or _is_legacy_aerodyn(aero) or "UAEndRad" in aero:
        return
    msg = (
        f"ua_end_radius={settings.ua_end_radius:g} is requested but {aero.path} has "
        "no UAEndRad line to write it to; add a UAEndRad line below UAStartRad"
    )
    raise FastInputError(msg)


def _configure_aerodyn(deck: FastInputFile, settings: LinearizationSettings) -> None:
    """Set AeroDyn's unsteady-aero and dynamic-wake models for linearization.

    Also writes ``UAEndRad`` when ``ua_end_radius`` is set (in a legacy-format file
    only when the line is present).
    """
    unsteady = settings.unsteady_aero_mode
    dynamic_wake = settings.dynamic_wake_mode
    ua_end_radius = settings.ua_end_radius
    if unsteady is None and dynamic_wake is None and ua_end_radius is None:
        return

    def edit(aero: FastInputFile) -> None:
        if ua_end_radius is not None:
            _set_if_present(aero, "UAEndRad", ua_end_radius)
        if _is_legacy_aerodyn(aero):
            _configure_legacy_aerodyn(aero, unsteady, dynamic_wake)
            return
        if unsteady is not None:
            _set_if_present(aero, "UA_Mod", unsteady)
        if dynamic_wake is not None:
            _set_if_present(aero, "DBEMT_Mod", dynamic_wake)

    _edit_subfile(deck, "AeroFile", edit)


def _configure_legacy_aerodyn(
    aero: FastInputFile, unsteady: int | None, dynamic_wake: int | None
) -> None:
    """Map the quasi-steady / frozen-wake request onto legacy AeroDyn keywords.

    Only unambiguous equivalents are written: quasi-steady aero (``UA_Mod = 0``) is
    ``AFAeroMod = 1``; a frozen (``-1``) or no (``0``) dynamic wake is plain BEMT
    (``WakeMod`` 2 -> 1) with ``FrozenWake`` on or off. Requests the legacy format
    cannot express (an unsteady model 4--6, continuous DBEMT 3) and a free vortex wake
    are left for :func:`~vane.run.validate_deck` to report.
    """
    if unsteady == 0:
        _set_if_present(aero, "AFAeroMod", _LEGACY_STEADY_AIRFOIL)
    if dynamic_wake in (-1, 0):
        if _deck_float(aero, "WakeMod") == float(_LEGACY_WAKE_DBEMT):
            aero.set("WakeMod", _LEGACY_WAKE_BEMT)
        _set_if_present(aero, "FrozenWake", dynamic_wake == -1)


def _time_beyond_tmax(settings: LinearizationSettings, deck: FastInputFile) -> float:
    """Return a time past the run's effective ``TMax`` for timed ServoDyn events.

    Used to neutralize ServoDyn's timed generator start/stop during a linearization:
    below rated a timed *start* this late means the generator never engages (so the
    torque trim owns the torque); above rated a timed *stop* this late means the
    generator stays engaged for the whole run (so the pitch trim keeps seeing the
    constant ``VS_RtTq``).

    The effective ``TMax`` is ``sim_time`` when set (it is written into the deck), else
    the primary deck's existing ``TMax`` — so the event is pushed past the *actual* run
    length, not a fixed constant that a longer-running template could exceed. Falls back
    to a large default only when neither is available.
    """
    effective_tmax = (
        settings.sim_time
        if settings.sim_time is not None
        else _deck_float(deck, "TMax")
    )
    if effective_tmax is not None and math.isfinite(effective_tmax):
        return effective_tmax + 1.0e3
    return _DEFAULT_TIME_BEYOND_TMAX


def _configure_servodyn(
    deck: FastInputFile, op: OperatingPoint, settings: LinearizationSettings
) -> None:
    """Make ServoDyn trim-compatible, with region-aware generator-torque handling.

    The Bladed-style DLL controller is not linearizable, so CalcSteady's own trim
    replaces it (``PCMode``/``VSContrl``/``HSSBrMode``/``YCMode``). OpenFAST's ServoDyn
    validation requires **timed** generator control for *any* linearization (rotating
    trim, parked, or standstill), so ``GenTiStr`` and ``GenTiStp`` are forced ``True``
    with the timed start/stop placed beyond ``TMax``.

    Disabling active pitch control exposes ServoDyn's passive pitch springs. Their
    ``PitNeut`` values are therefore moved to the requested operating-point pitch;
    otherwise a nonzero settled pitch silently relaxes back to the deck's original
    neutral (commonly zero) before a below-rated torque trim linearizes.

    By default the generator is held off — the right state for a parked/standstill point
    and for the below-rated torque trim (``TrimCase = 2``), which owns the torque itself
    so ServoDyn contributes zero. The simple variable-speed law is *neutralised* to a
    valid flat shape (``VS_Rgn2K = 0`` with a tiny ``VS_RtGnSp``/``VS_SlPc``) so the
    ``VSContrl = 1`` init validation passes even for a DLL-style dummy deck without a
    rated torque. An **above-rated rotating point** differs: above rated the steady
    generator torque is the constant rated torque, so for both an above-rated pitch trim
    (``TrimCase = 3``) and a single un-trimmed linearization (``calc_steady=False``) the
    law is flattened to ``VS_RtTq`` (from ``rated_generator_torque`` when set, else the
    deck's positive ``VS_RtTq``) and the generator is engaged for the whole run.

    (A *below-rated* rotating point with ``calc_steady=False`` is unsupported and is
    rejected up front by :func:`_validate_operating_configuration`, before any deck is
    written, so it never reaches here.)

    Raises
    ------
    vane.io.FastInputError
        Above rated, when no positive constant torque is available to flatten the law to
        (see :func:`_flatten_to_constant_torque`).
    """
    if not settings.trim_controller:
        return
    if _servo_module_disabled(deck):
        return  # ServoDyn off (CompServo = 0); the ServoFile is inert, leave it alone
    # The region matters only for a rotating point (a parked rotor keeps the generator
    # off), so a parked point needs no rated wind speed.
    below_rated = op.is_rotating and settings.is_below_rated(op.wind_speed)
    # The unsupported below-rated rotating calc_steady=False case is rejected up front
    # by configure_linearization (_validate_operating_configuration) before any write.
    # Above rated the steady generator torque is the (constant) rated torque, so the
    # generator is engaged for any rotating point linearized there — both a CalcSteady
    # pitch trim and a single un-trimmed linearization (calc_steady=False) need it
    # applied, since neither has a below-rated torque trim to supply the torque.
    engage_generator = op.is_rotating and not below_rated
    # Computed from the primary deck's effective TMax (see _time_beyond_tmax), once,
    # since it does not depend on the ServoDyn sub-file.
    beyond_tmax = _time_beyond_tmax(settings, deck)

    def edit(servo: FastInputFile) -> None:
        # Judge the deck's own VS_RtTq by its ORIGINAL VSContrl, before the trim
        # overwrites VSContrl with the simple law below.
        trust_deck_torque = _deck_uses_simple_vs_law(servo)
        for key, value in (
            ("PCMode", 0),
            ("VSContrl", 1),
            ("HSSBrMode", 0),
            ("YCMode", 0),
            # Active-flap and cable control: off, or a value of 5 would still load
            # the controller library, which cannot be linearized.
            ("AfCmode", 0),
            ("CCmode", 0),
        ):
            _set_if_present(servo, key, value)
        # GenModel is read only with VSContrl = 0, which the trim never uses, yet
        # ServoDyn rejects any value but 1 or 2 when linearizing: fall back to the
        # simple generator model (1) when the deck carries another one.
        gen_model = _deck_float(servo, "GenModel")
        if gen_model is not None and gen_model not in _LINEARIZABLE_GEN_MODELS:
            servo.set("GenModel", 1)
        for key in _PITCH_NEUTRAL_KEYS:
            _set_if_present(servo, key, op.blade_pitch_deg)
        # ServoDyn linearization validation requires timed generator control both ways
        # (GenTiStr/GenTiStp=True) for ANY linearization — rotating trim, parked, or
        # standstill. Set them and place the timed start/stop beyond TMax; the generator
        # is held off by default (right for a parked point and the below-rated torque
        # trim, which owns the torque). An above-rated rotating point overrides this.
        _set_if_present(servo, "GenTiStr", True)
        _set_if_present(servo, "GenTiStp", True)
        _set_if_present(servo, "TimGenOn", beyond_tmax)
        _set_if_present(servo, "TimGenOf", beyond_tmax)
        if engage_generator:
            # Above-rated rotating point: engage the generator the whole run (start at
            # 0) and flatten the law to the positive constant torque it applies.
            _flatten_to_constant_torque(servo, settings, trust_deck_torque)
            _set_if_present(servo, "TimGenOn", 0.0)
        else:
            # Parked/standstill or below-rated torque trim: generator stays off, so
            # ServoDyn returns zero torque. Neutralise the simple-VS law so VSContrl=1
            # init validation still passes for a DLL-style dummy deck.
            _neutralize_simple_vs(servo)

    _edit_subfile(deck, "ServoFile", edit)


def _keep_output_start_before_end(deck: FastInputFile) -> None:
    """Start the output at 0 s when the deck's ``TStart`` is not before ``TMax``.

    OpenFAST rejects a deck whose ``TMax`` is not greater than ``TStart`` (the first
    output time). A deck that skips a start-up transient in its output (``TStart`` =
    120 s, say) trips this once its run is shortened, so the output then starts at 0.
    """
    start = _deck_float(deck, "TStart")
    end = _deck_float(deck, "TMax")
    if start is not None and end is not None and start >= end:
        deck.set("TStart", 0.0)


def _deck_float(deck: FastInputFile, keyword: str) -> float | None:
    """Return a deck keyword parsed as ``float``, or ``None`` if absent/non-numeric."""
    if keyword not in deck:
        return None
    try:
        return fortran_float(deck.get(keyword))
    except (TypeError, ValueError):
        return None


_SIMPLE_VS_KEYS = ("VS_RtGnSp", "VS_Rgn2K", "VS_SlPc")
# Any simple variable-speed field, including the rated torque. The flatten/validation
# "is there a law to flatten?" check uses this superset so a deck that carries only
# ``VS_RtTq`` (and not the speed/slip fields) is still given its constant torque rather
# than left at a dummy value with the generator engaged.
_SIMPLE_VS_FIELDS = (*_SIMPLE_VS_KEYS, "VS_RtTq")


def _neutralize_simple_vs(servo: FastInputFile) -> None:
    """Set the simple variable-speed law to a valid, flat (constant-torque) shape.

    Writes a tiny positive rated speed/slip and zero region-2 gain so the simple-VS
    params pass ServoDyn's ``VSContrl = 1`` init validation: ``VS_RtGnSp`` > 0,
    ``VS_SlPc`` > 0, ``VS_Rgn2K`` >= 0, and the product ``VS_Rgn2K*VS_RtGnSp^2`` == 0
    (so ``<= VS_RtTq`` for any ``VS_RtTq`` >= 0). Torque is then flat at ``VS_RtTq``,
    left untouched here; callers needing a specific constant set it separately.
    """
    _set_if_present(servo, "VS_RtGnSp", _SIMPLE_VS_SPEED_SENTINEL)
    _set_if_present(servo, "VS_Rgn2K", _REGION2_GAIN_OFF)
    _set_if_present(servo, "VS_SlPc", _SIMPLE_VS_SPEED_SENTINEL)


_NO_CONSTANT_TORQUE_MSG = (
    "above-rated trim_controller flattens the ServoDyn variable-speed law to the "
    "constant torque VS_RtTq, but no positive generator torque is available: set "
    "rated_generator_torque (recommended), or supply a positive VS_RtTq in the "
    "ServoDyn deck. run_operating_sweep infers it from the controller input file "
    "(ServoDyn DLL_InFile, keyword VS_RtTq), from ServoDyn VS_RtTq with VSContrl=1, "
    "or from a cp column of performance_csv"
)


def _deck_uses_simple_vs_law(servo: FastInputFile) -> bool:
    """Return whether the deck's own ``VS_RtTq`` describes its active torque law.

    Only ServoDyn's simple variable-speed law (``VSContrl = 1``) uses ``VS_RtTq``. A
    DLL (``5``), user-subroutine (``3``), Simulink (``4``) or no-control (``0``) deck
    carries a placeholder there, commonly ``9999.9``, which must never become the
    constant torque of an above-rated trim. A deck without ``VSContrl`` is trusted.
    """
    vs_control = _deck_float(servo, "VSContrl")
    return vs_control is None or vs_control == 1.0


def _untrusted_deck_torque_msg(servo: FastInputFile) -> str:
    """Return the rejection message for a placeholder (non-simple-law) ``VS_RtTq``."""
    return (
        "above-rated trim_controller flattens the ServoDyn variable-speed law to a "
        f"constant torque, but {servo.path} has VSContrl={servo.get('VSContrl')} "
        "(not the simple law 1), so its VS_RtTq is a placeholder rather than the "
        "rated generator torque; set rated_generator_torque (run_operating_sweep "
        "infers it from the controller input file named by ServoDyn DLL_InFile, "
        "keyword VS_RtTq, or from a cp column of performance_csv; neither was "
        "available)"
    )


def _resolve_constant_torque(
    servo: FastInputFile,
    settings: LinearizationSettings,
    trust_deck_torque: bool,
) -> float | None:
    """Return the positive constant generator torque to flatten to, or ``None``.

    Prefers ``rated_generator_torque`` (validated positive at construction); otherwise
    falls back to the deck's existing ``VS_RtTq`` when ``trust_deck_torque`` (the deck's
    original ``VSContrl`` is the simple law). Returns ``None`` when neither yields a
    positive, finite value — the caller then rejects the configuration.
    """
    constant_torque = settings.rated_generator_torque
    if constant_torque is None and trust_deck_torque:
        constant_torque = _deck_float(servo, "VS_RtTq")
    if (
        constant_torque is None
        or not math.isfinite(constant_torque)
        or constant_torque <= 0.0
    ):
        return None
    return constant_torque


def _flatten_to_constant_torque(
    servo: FastInputFile,
    settings: LinearizationSettings,
    trust_deck_torque: bool,
) -> None:
    """Flatten ServoDyn's simple variable-speed law to a positive constant torque.

    For the above-rated pitch trim the generator is engaged and applies a *constant*
    torque. The law is neutralised (see :func:`_neutralize_simple_vs`) and ``VS_RtTq``
    is set to that constant: ``rated_generator_torque`` if given, else the deck's value
    when ``trust_deck_torque`` (its original ``VSContrl`` was the simple law).
    A non-positive torque (e.g. a DLL dummy ``VS_RtTq = 0``) would trim the engaged
    generator against zero torque; that case is rejected up front by
    :func:`_validate_operating_configuration` (before any deck write), so the guard here
    is a defensive backstop with the same contract.

    Raises
    ------
    vane.io.FastInputError
        If the law would be flattened but no positive, finite constant torque is
        available (neither ``rated_generator_torque`` nor a positive deck ``VS_RtTq``).
    """
    if not any(key in servo for key in _SIMPLE_VS_FIELDS):
        return  # this deck has no simple variable-speed law to flatten
    constant_torque = _resolve_constant_torque(servo, settings, trust_deck_torque)
    if constant_torque is None:
        if settings.rated_generator_torque is None and not trust_deck_torque:
            raise FastInputError(_untrusted_deck_torque_msg(servo))
        raise FastInputError(_NO_CONSTANT_TORQUE_MSG)
    _neutralize_simple_vs(servo)
    _set_if_present(servo, "VS_RtTq", constant_torque)


def _set_if_present(
    deck: FastInputFile, keyword: str, value: str | int | float | bool
) -> None:
    """Set ``keyword`` only when the deck contains it (tolerate deck variants)."""
    if keyword in deck:
        deck.set(keyword, value)
