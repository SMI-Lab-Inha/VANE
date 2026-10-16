# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Pre-flight validation that a deck will actually linearize, before running it.

An OpenFAST linearization fails late and obscurely when the deck holds a module
configuration linearization does not support — a Bladed-style DLL controller (which has
no Jacobian), wave excitation, convolution radiation, or a state-space hydrodynamics
model whose ``.ss`` / ``.ssexctn`` coefficient files are missing. A long settle and
trim run then aborts with a terse solver error.

:func:`validate_deck` checks the **effective** configuration — the deck as it will be
*after* the :class:`~vane.run.LinearizationSettings` edits are applied — and reports
every blocking problem at once, before any OpenFAST process is launched. It is a guard,
not a fixer: it never edits the deck. Settings that neutralise a problem (e.g.
``trim_controller`` replacing the DLL, or ``wave_mod=0`` forcing still water) clear the
corresponding check, so a raw FOWT template paired with
:meth:`LinearizationSettings.offshore` passes, while the same template run with default
settings is rejected with an actionable message.

The checks mirror the linearization restrictions OpenFAST enforces in its glue code and
modules (see the *Linearization restrictions* reference page). A module the primary
deck switches off (``CompServo``, ``CompSeaSt``, ``CompHydro``, ``CompAero``,
``CompMooring``, ``CompElast``, ``CompInflow`` = 0 or another module choice) leaves its
input file inert, so that file is not inspected.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

from vane.io.fst_reader import _fortran_true, fortran_float
from vane.io.fst_writer import FastInputError, FastInputFile
from vane.run.deck_files import DeckReferenceError, deck_file_closure
from vane.run.lin_config import (
    _LEGACY_STEADY_AIRFOIL,
    _LEGACY_WAKE_BEMT,
    _LEGACY_WAKE_DBEMT,
    _LINEARIZABLE_GEN_MODELS,
    _SECOND_ORDER_HYDRO_KEYS,
    _SECOND_ORDER_WAVE_KEYS,
    _STILL_WATER_WAVE_STRETCHING,
    DLL_CONTROL_KEYS,
    LinearizationSettings,
    _current_heading_is_default,
    _is_legacy_aerodyn,
    _radiation_step_mismatch,
)
from vane.utils.paths import resolve_path

__all__ = [
    "OPENFAST_MAX_PATH_LENGTH",
    "DeckPathLengthError",
    "DeckValidationError",
    "check_deck_path_lengths",
    "validate_deck",
]

_LOGGER = logging.getLogger(__name__)

# ServoDyn control switches whose value 5 selects the Bladed-style external DLL, which
# has no analytic Jacobian and cannot be linearized.
_DLL_SWITCH = 5
# ServoDyn switches linearization requires to be 0 (PCMode, HSSBrMode, YCMode) or in
# {0, 1} (VSContrl), and the timed generator start/stop it requires to be on.
_ZERO_CONTROL_KEYS = ("PCMode", "HSSBrMode", "YCMode")
_LINEARIZABLE_VS_CONTROL = frozenset({0, 1})
_TIMED_GENERATOR_KEYS = ("GenTiStr", "GenTiStp")

# AeroDyn unsteady-aero (UA_Mod), dynamic-wake (DBEMT_Mod), and wake (Wake_Mod) settings
# linearization accepts; any other value cannot be linearized.
_LINEARIZABLE_UA_MODES = frozenset({0, 4, 5, 6})
_LINEARIZABLE_DBEMT_MODES = frozenset({-1, 0, 3})
_LINEARIZABLE_WAKE_MODES = frozenset({0, 1})  # none or BEMT (not the OLAF free wake)
_BEMT_WAKE_MODE = 1
# AeroDyn reads a missing UAEndRad as 1 (unsteady aerodynamics out to the tip).
_DEFAULT_UA_END_RADIUS = 1.0

# Primary-deck module choices OpenFAST cannot linearize, by switch:
# (value, module name).
_UNLINEARIZABLE_MODULES: dict[str, tuple[tuple[int, str], ...]] = {
    "CompElast": ((3, "Simplified-ElastoDyn (SED)"),),
    "CompAero": ((1, "AeroDisk"), (3, "ExtLoads")),
    "CompMooring": ((2, "FEAMooring"), (4, "OrcaFlex")),
    "CompIce": ((1, "IceFloe"), (2, "IceDyn")),
    "CompSoil": ((1, "SoilDyn"),),
}
_COMP_ELAST_BEAMDYN = 2
_COMP_MOORING_MOORDYN = 3
_BEAMDYN_BLADE_KEYS = ("BDBldFile(1)", "BDBldFile(2)", "BDBldFile(3)")

# MoorDyn section headers (matched as substrings of a "---" header line, as MoorDyn
# itself does) for the line-type dictionary and the line list.
_MOORDYN_TYPE_HEADERS = ("LINE DICTIONARY", "LINE TYPES")
_MOORDYN_LINE_HEADERS = ("LINES", "LINE PROPERTIES", "LINE LIST")
_MOORDYN_MIN_VIV_COLUMNS = 11  # a line-type row with a lift coefficient (Cl) column
_MOORDYN_CL_COLUMN = 10
_MOORDYN_EA_COLUMN = 3  # bar-separated values select a viscoelastic line
_MOORDYN_LINE_TYPE_COLUMN = 1


class DeckValidationError(ValueError):
    """Raised when a deck cannot be linearized as configured.

    The message lists every blocking problem found, one per line, so the deck can be
    fixed in a single pass rather than one error per run.
    """


class DeckPathLengthError(DeckValidationError):
    """Raised when OpenFAST would open a deck file by a path that is too long.

    See :func:`check_deck_path_lengths`.
    """


OPENFAST_MAX_PATH_LENGTH = 258
r"""Longest full path, in characters, the Windows OpenFAST build opens a file by.

Windows' classic ``MAX_PATH`` is 260 characters including the terminating null. The
Intel-Fortran OpenFAST build is not long-path aware and does not collapse ``..``: on
OpenFAST v5.0.0 an input opened as ``<working dir>\.\..\5MW_Baseline\<file>`` of 258
characters was read and one of 259 characters failed with ``The input file ... was
not found``.
"""


def check_deck_path_lengths(
    deck_path: str | Path, *, max_length: int = OPENFAST_MAX_PATH_LENGTH
) -> None:
    r"""Reject a deck whose sub-files OpenFAST would open by an over-long path.

    OpenFAST builds every sub-file path by joining the referencing file's directory,
    as it spelled it, to the reference -- without collapsing ``..``. Started from the
    deck's directory on the deck's bare name (as :func:`~vane.run.run_openfast` does),
    a deck that keeps shared files next to it in ``../Shared/`` therefore opens them
    as ``<deck dir>\.\..\Shared\<file>``, and a nested reference inherits the whole
    chain. The operating system receives that string appended to the working
    directory. When it exceeds the Windows path limit the file cannot be opened and
    OpenFAST aborts at initialisation with ``The input file ... was not found``,
    although the file exists.

    This check walks the deck's references
    (:func:`~vane.run.deck_files.deck_file_closure`) and spells each one as OpenFAST
    will, before any process is launched.

    Parameters
    ----------
    deck_path : str or pathlib.Path
        The primary ``.fst`` deck, in the directory OpenFAST will be run from.
    max_length : int, optional
        Longest permitted full path in characters. Default
        :data:`OPENFAST_MAX_PATH_LENGTH`.

    Raises
    ------
    DeckPathLengthError
        If any file would be opened by a path longer than ``max_length``; the message
        names the longest such path and its length.
    FileNotFoundError
        If ``deck_path`` is not an existing file.

    Notes
    -----
    The limit applies to Windows builds that are not long-path aware, which is why
    :func:`~vane.run.run_openfast` runs this check only on Windows. A reference the
    walk cannot follow (an unreadable input) is skipped with a debug log rather than
    failing the check; OpenFAST will report it itself.
    """
    deck = resolve_path(deck_path)
    if not deck.is_file():
        msg = f"OpenFAST deck not found: {deck}"
        raise FileNotFoundError(msg)
    try:
        # Rooting the walk at the drive anchor keeps every reference "inside".
        closure = deck_file_closure(deck, Path(deck.anchor))
    except DeckReferenceError as error:
        _LOGGER.debug("Path-length check could not walk %s: %s", deck, error)
        return
    too_long = [opened for _, opened in closure.open_paths if len(opened) > max_length]
    if not too_long:
        return
    longest = max(too_long, key=len)
    msg = (
        f"OpenFAST cannot open {len(too_long)} file(s) of {deck}: the path it opens "
        f"them by exceeds the {max_length}-character Windows limit. Longest "
        f"({len(longest)} characters): {longest}. Use a shorter output_dir (or move "
        "the case closer to the drive root)."
    )
    raise DeckPathLengthError(msg)


def validate_deck(
    deck_path: str | Path,
    settings: LinearizationSettings | None = None,
) -> None:
    """Reject a deck that will not linearize as configured, before running it.

    Validates the configuration that *will be in force* once ``settings`` is applied:
    a problem the settings neutralise (the DLL controller replaced by the trim, waves
    forced to still water) does not fail the check, but a genuinely non-linearizable
    state that nothing fixes does. The checks cover the module choices OpenFAST cannot
    linearize (SED, AeroDisk, ExtLoads, FEAMooring, OrcaFlex, ice and soil modules),
    the CalcSteady azimuth step, InflowWind (lidar, upflow), ServoDyn (control modes,
    generator model, timed generator start/stop), SeaState (waves, spreading,
    constrained waves, 2nd-order kinematics, still-water consistency), HydroDyn
    (potential-flow model, radiation/excitation models and their files, 2nd-order
    forces, additional body DOFs, radiation time step), AeroDyn (wake, unsteady-aero,
    and dynamic-wake models, unsteady aerodynamics reaching the blade-tip node under
    tip loss, nacelle drag), BeamDyn (pitch actuator), and MoorDyn (VIV
    and viscoelastic lines).

    Parameters
    ----------
    deck_path : str or pathlib.Path
        Path to the primary ``.fst`` deck. Its sub-files are inspected when present
        and their module is enabled; an onshore deck without them is fine.
    settings : LinearizationSettings or None, optional
        The settings that will be applied before running. Defaults to
        :class:`LinearizationSettings` (all module fields "leave unchanged"), so a deck
        that is already linearization-ready passes and a raw one is judged as-is.

    Raises
    ------
    DeckValidationError
        If one or more blocking problems are found (all are reported together).
    vane.io.FastInputError
        If ``deck_path`` cannot be read at all.
    """
    settings = settings or LinearizationSettings()
    deck = FastInputFile.read(deck_path)
    problems: list[str] = []
    _check_module_choices(deck, problems)
    _check_calc_steady_step(deck, problems)
    _check_inflow(deck, problems)
    _check_controller(deck, settings, problems)
    _check_sea_state(deck, settings, problems)
    _check_hydrodyn(deck, settings, problems)
    _check_aerodyn(deck, settings, problems)
    _check_beamdyn(deck, problems)
    _check_moordyn(deck, problems)
    if problems:
        joined = "\n".join(f"  - {problem}" for problem in problems)
        msg = f"{deck.path} is not linearizable as configured:\n{joined}"
        raise DeckValidationError(msg)


def _module_enabled(deck: FastInputFile, comp_key: str) -> bool:
    """Return ``False`` only when the primary module switch is explicitly off.

    A deck may keep a sub-file path while disabling the module (``CompServo=0``,
    ``CompHydro=0``, ...); that sub-file is inert, so its contents must not fail the
    pre-flight. A switch that is absent (older deck) or non-zero counts as enabled.
    """
    return _read_int(deck, comp_key) != 0


def _check_module_choices(deck: FastInputFile, problems: list[str]) -> None:
    """Flag primary-deck module choices that OpenFAST cannot linearize."""
    for comp_key, choices in _UNLINEARIZABLE_MODULES.items():
        value = _read_int(deck, comp_key)
        for choice, name in choices:
            if value == choice:
                problems.append(
                    f"{comp_key}={choice} selects {name}, which OpenFAST cannot "
                    f"linearize; choose a linearizable module or set {comp_key}=0"
                )


def _check_calc_steady_step(deck: FastInputFile, problems: list[str]) -> None:
    """Flag a CalcSteady trim whose time step skips past an azimuth target.

    CalcSteady linearizes at ``NLinTimes`` equally spaced rotor azimuths and aborts
    ("The rotor is spinning too fast") when the rotor turns further than that spacing
    in one time step. Checked only on a deck that already requests a CalcSteady
    linearization (as configured by :func:`~vane.run.configure_linearization`), using
    its ``DT``, ``NLinTimes`` and the ElastoDyn initial ``RotSpeed``.
    """
    if not (_read_flag(deck, "Linearize") and _read_flag(deck, "CalcSteady")):
        return
    dt = _read_float(deck, "DT")
    n_lin_times = _read_int(deck, "NLinTimes")
    rot_speed = _subfile_float(deck, "EDFile", "RotSpeed")
    if dt is None or rot_speed is None or n_lin_times is None or n_lin_times < 1:
        return
    step_deg = abs(rot_speed) * 6.0 * dt  # rpm -> deg/s, times the step
    spacing_deg = 360.0 / n_lin_times
    if step_deg > spacing_deg:
        problems.append(
            f"CalcSteady: the rotor turns {step_deg:.3g} deg per time step "
            f"(RotSpeed={rot_speed:g} rpm, DT={dt:g} s), more than the "
            f"{spacing_deg:.3g} deg azimuth spacing of NLinTimes={n_lin_times}; "
            "reduce DT or n_lin_times"
        )


def _check_inflow(deck: FastInputFile, problems: list[str]) -> None:
    """Flag InflowWind options linearization rejects (lidar, upflow).

    The wind type is not checked: :func:`~vane.run.configure_linearization` always
    writes steady wind (``WindType = 1``).
    """
    if not _module_enabled(deck, "CompInflow"):
        return
    inflow = _subfile(deck, "InflowFile")
    if inflow is None:
        return
    sensor = _read_int(inflow, "SensorType")
    if sensor is not None and sensor != 0:
        problems.append(
            f"InflowWind has the lidar module enabled (SensorType={sensor}), which "
            "linearization does not support; set SensorType=0"
        )
    upflow = _read_float(inflow, "VFlowAng")
    if upflow is not None and upflow != 0.0:
        problems.append(
            f"InflowWind has a vertical flow (upflow) angle (VFlowAng={upflow:g}); "
            "linearization requires VFlowAng=0"
        )


def _check_controller(
    deck: FastInputFile, settings: LinearizationSettings, problems: list[str]
) -> None:
    """Flag ServoDyn control settings linearization rejects.

    ``trim_controller`` writes every one of them (control modes off or simple, timed
    generator start/stop on, a linearizable ``GenModel``), so it clears this check.
    """
    if settings.trim_controller:
        return  # the trim replaces the controller and sets every switch below
    if not _module_enabled(deck, "CompServo"):
        return  # ServoDyn is off; its file is inert
    servo = _subfile(deck, "ServoFile")
    if servo is None:
        return
    remedy = (
        "use LinearizationSettings.offshore() or set trim_controller=True so "
        "CalcSteady's trim replaces the controller"
    )
    dll = [key for key in DLL_CONTROL_KEYS if _read_int(servo, key) == _DLL_SWITCH]
    if dll:
        problems.append(
            f"ServoDyn uses the Bladed-style DLL controller ({', '.join(dll)}=5), "
            f"which is not linearizable; {remedy}"
        )
    other = [
        f"{key}={value}"
        for key in _ZERO_CONTROL_KEYS
        if key not in dll and (value := _read_int(servo, key)) not in (None, 0)
    ]
    vs_control = _read_int(servo, "VSContrl")
    if (
        "VSContrl" not in dll
        and vs_control is not None
        and vs_control not in _LINEARIZABLE_VS_CONTROL
    ):
        other.append(f"VSContrl={vs_control}")
    gen_model = _read_int(servo, "GenModel")
    if gen_model is not None and gen_model not in _LINEARIZABLE_GEN_MODELS:
        other.append(f"GenModel={gen_model}")
    other.extend(
        f"{key}=False"
        for key in _TIMED_GENERATOR_KEYS
        if key in servo and not _read_flag(servo, key)
    )
    if other:
        problems.append(
            f"ServoDyn settings are not linearizable ({', '.join(other)}); "
            "linearization requires PCMode=HSSBrMode=YCMode=0, VSContrl 0 or 1, "
            f"GenModel 1 or 2, and GenTiStr=GenTiStp=True; {remedy}"
        )


def _check_sea_state(
    deck: FastInputFile, settings: LinearizationSettings, problems: list[str]
) -> None:
    """Flag SeaState wave options that linearization cannot handle."""
    if not _module_enabled(deck, "CompSeaSt"):
        return  # SeaState is off; its file is inert
    sea_state = _subfile(deck, "SeaStFile")
    if sea_state is None:
        return
    effective = _effective(settings.wave_mod, _read_int(sea_state, "WaveMod"))
    if effective is not None and effective != 0:
        problems.append(
            f"SeaState has waves enabled (effective WaveMod={effective}); "
            f"linearization needs still water — set wave_mod=0"
        )
    if settings.wave_mod is None:
        # wave_mod (still water) resets these; without it the deck's values stand.
        _check_still_water_options(sea_state, effective, problems)
    if not settings.disable_second_order_hydro:
        active = [key for key in _SECOND_ORDER_WAVE_KEYS if _read_flag(sea_state, key)]
        if active:
            problems.append(
                f"SeaState has 2nd-order wave kinematics on ({', '.join(active)}), "
                f"which linearization does not support; set "
                f"disable_second_order_hydro=True to switch them off"
            )


def _check_still_water_options(
    sea_state: FastInputFile, wave_mod: int | None, problems: list[str]
) -> None:
    """Flag SeaState wave options a still-water linearization rejects."""
    active = [
        f"{key}={value}"
        for key in ("WaveDirMod", "ConstWaveMod")
        if (value := _read_int(sea_state, key)) not in (None, 0)
    ]
    if active:
        problems.append(
            f"SeaState has directional spreading or a constrained wave on "
            f"({', '.join(active)}), which linearization does not support; set "
            "wave_mod=0 (still water also resets WaveDirMod and ConstWaveMod)"
        )
    if wave_mod != 0:
        return
    still_water = []
    stretching = _read_int(sea_state, "WaveStMod")
    if stretching is not None and stretching not in _STILL_WATER_WAVE_STRETCHING:
        still_water.append(f"WaveStMod={stretching} (must be 0 or 1)")
    diffraction = _read_float(sea_state, "MCFD")
    if diffraction is not None and diffraction != 0.0:
        still_water.append(f"MCFD={diffraction:g} (must be 0)")
    if _current_heading_is_default(sea_state):
        still_water.append("CurrSSDir=DEFAULT (must be a heading)")
    if still_water:
        problems.append(
            f"SeaState still water (WaveMod=0) is inconsistent with "
            f"{', '.join(still_water)}; OpenFAST rejects these at initialization; "
            "set wave_mod=0 to have them reset"
        )


def _check_hydrodyn(
    deck: FastInputFile, settings: LinearizationSettings, problems: list[str]
) -> None:
    """Flag non-linearizable hydro models, missing state-space files, and conflicts."""
    if not _module_enabled(deck, "CompHydro"):
        return  # HydroDyn is off; its file is inert
    hydro = _subfile(deck, "HydroFile")
    if hydro is None:
        return
    pot_mod = _read_int(hydro, "PotMod")
    if pot_mod is not None and pot_mod > 1:
        problems.append(
            f"HydroDyn uses the FIT potential-flow model (PotMod={pot_mod}), which "
            "linearization does not support; set PotMod=0 or 1"
        )
    radiation = _effective(settings.hydro_radiation_mode, _read_int(hydro, "RdtnMod"))
    excitation = _effective(
        settings.hydro_excitation_mode, _read_int(hydro, "ExctnMod")
    )
    if pot_mod == 0:
        # No potential flow (e.g. a Morison-only monopile): HydroDyn ignores
        # RdtnMod/ExctnMod, so neither the model nor its .ss files matter.
        radiation = excitation = None
    if radiation == 1:
        problems.append(
            "HydroDyn uses convolution radiation (effective RdtnMod=1), which "
            "linearization does not support; use the state-space model (RdtnMod=2) "
            "or none (RdtnMod=0)"
        )
    if excitation == 1:
        problems.append(
            "HydroDyn uses DFT wave excitation (effective ExctnMod=1), which "
            "linearization does not support; use the state-space model (ExctnMod=2) "
            "or none (ExctnMod=0)"
        )
    if radiation == 2 and not _has_potfile(hydro, ".ss"):
        if not settings.hydro_radiation_fit:
            problems.append(
                "HydroDyn requests state-space radiation (effective RdtnMod=2) but the "
                "'.ss' coefficient file next to PotFile is missing"
            )
        elif not _has_potfile(hydro, ".1"):
            # hydro_radiation_fit generates the .ss file in each case copy, from .1.
            problems.append(
                "HydroDyn requests state-space radiation (effective RdtnMod=2) with "
                "hydro_radiation_fit, but neither the '.ss' coefficient file nor the "
                "WAMIT '.1' radiation file it is fitted to is next to PotFile"
            )
    if excitation == 2 and not _has_potfile(hydro, ".ssexctn"):
        problems.append(
            "HydroDyn requests state-space wave excitation (effective ExctnMod=2) but "
            "the '.ssexctn' coefficient file next to PotFile is missing"
        )
    _check_hydro_add_dofs(hydro, radiation, excitation, problems)
    if (
        settings.hydro_radiation_mode is None
        and settings.hydro_excitation_mode is None
        and _radiation_step_mismatch(hydro, _read_float(deck, "DT"))
    ):
        problems.append(
            f"HydroDyn RdtnDT={hydro.get('RdtnDT')} differs from the glue-code DT, "
            "which HydroDyn rejects with a radiation or state-space excitation model; "
            "set RdtnDT=DEFAULT"
        )
    if pot_mod is not None and pot_mod != 1:
        return  # HydroDyn switches the 2nd-order forces off without WAMIT flow
    if not settings.disable_second_order_hydro:
        active = [
            key
            for key in _SECOND_ORDER_HYDRO_KEYS
            if (value := _read_int(hydro, key)) is not None and value != 0
        ]
        if active:
            problems.append(
                f"HydroDyn has 2nd-order force switches active ({', '.join(active)}), "
                f"which linearization does not support; set "
                f"disable_second_order_hydro=True to zero them"
            )


def _check_hydro_add_dofs(
    hydro: FastInputFile,
    radiation: int | None,
    excitation: int | None,
    problems: list[str],
) -> None:
    """Flag additional (generalized) body DOFs combined with a state-space model.

    HydroDyn cannot use ``NAddDOF > 0`` with state-space radiation or excitation, and
    linearization cannot use the convolution/DFT alternatives, so such a body
    linearizes only with ``RdtnMod = ExctnMod = 0``.
    """
    if "NAddDOF" not in hydro:
        return
    try:
        counts = [int(fortran_float(token)) for token in hydro.tokens("NAddDOF")]
    except (FastInputError, ValueError):
        return
    if not any(count > 0 for count in counts):
        return
    state_space = [
        f"{name}=2"
        for name, value in (("RdtnMod", radiation), ("ExctnMod", excitation))
        if value == 2
    ]
    if state_space:
        problems.append(
            f"HydroDyn has additional body DOFs (NAddDOF>0), which cannot be "
            f"combined with the state-space model ({', '.join(state_space)}); "
            "a linearization of such a body needs RdtnMod=0 and ExctnMod=0"
        )


def _check_aerodyn(
    deck: FastInputFile, settings: LinearizationSettings, problems: list[str]
) -> None:
    """Flag AeroDyn wake, unsteady-aero, and dynamic-wake models and nacelle drag."""
    if not _module_enabled(deck, "CompAero"):
        return  # AeroDyn is off; its file is inert
    aero = _subfile(deck, "AeroFile")
    if aero is None:
        return
    if _is_legacy_aerodyn(aero):
        _check_legacy_aerodyn(aero, settings, problems)
        return
    wake = _read_int(aero, "Wake_Mod")
    if wake is not None and wake not in _LINEARIZABLE_WAKE_MODES:
        problems.append(
            f"AeroDyn wake model (Wake_Mod={wake}) is not linearizable; the free "
            "vortex wake (OLAF) has no linearization; set Wake_Mod=0 or 1 (BEMT)"
        )
    unsteady = _effective(settings.unsteady_aero_mode, _read_int(aero, "UA_Mod"))
    dynamic_wake = _effective(settings.dynamic_wake_mode, _read_int(aero, "DBEMT_Mod"))
    if unsteady is not None and unsteady not in _LINEARIZABLE_UA_MODES:
        problems.append(
            f"AeroDyn unsteady-aero model (effective UA_Mod={unsteady}) is not "
            f"linearizable; use quasi-steady (UA_Mod=0) via unsteady_aero_mode"
        )
    if dynamic_wake is not None and dynamic_wake not in _LINEARIZABLE_DBEMT_MODES:
        problems.append(
            f"AeroDyn dynamic-wake model (effective DBEMT_Mod={dynamic_wake}) is not "
            f"linearizable; use a frozen wake (DBEMT_Mod=-1) via dynamic_wake_mode"
        )
    if _read_flag(aero, "NacelleDrag"):
        problems.append(
            "AeroDyn nacelle drag (NacelleDrag=True) cannot be used for "
            "linearization; set NacelleDrag=False"
        )
    _check_unsteady_aero_tip(aero, settings, unsteady, wake, problems)


def _check_unsteady_aero_tip(
    aero: FastInputFile,
    settings: LinearizationSettings,
    unsteady: int | None,
    wake: int | None,
    problems: list[str],
) -> None:
    """Flag unsteady aerodynamics left on at the blade-tip node under tip loss.

    AeroDyn switches unsteady aerodynamics off only on nodes whose radius fraction
    (projected radius over the largest node radius) is *greater* than ``UAEndRad``, so
    with ``UAEndRad >= 1`` the tip node (fraction 1) keeps it. With the Prandtl tip
    loss model, BEMT's tip-loss constant is exactly zero at the last node, and BEMT
    then fixes that node's inductions (axial induction 1, tangential 0). The
    unsteady-aerodynamics states of that degenerate node can turn NaN or infinite in
    the linearization while the run otherwise converges. Checked for a linearizable
    unsteady model (``UA_Mod`` 4, 5 or 6) with BEMT induction (``Wake_Mod = 1``) and
    ``TipLoss`` on. A missing ``UAEndRad`` line reads as 1, as AeroDyn defaults it.
    """
    if unsteady not in _LINEARIZABLE_UA_MODES - {0}:
        return
    if wake != _BEMT_WAKE_MODE or not _read_flag(aero, "TipLoss"):
        return
    end_radius = settings.ua_end_radius
    if end_radius is None:
        deck_value = _read_float(aero, "UAEndRad")
        end_radius = _DEFAULT_UA_END_RADIUS if deck_value is None else deck_value
    if end_radius < _DEFAULT_UA_END_RADIUS:
        return
    problems.append(
        f"AeroDyn unsteady aerodynamics (effective UA_Mod={unsteady}) reaches the "
        f"blade-tip node (effective UAEndRad={end_radius:g}) with TipLoss=True; the "
        "tip node's inflow is fixed by the tip loss model and its unsteady-aero "
        "states can turn NaN in the linearization; set UAEndRad below 1 so unsteady "
        "aerodynamics stops inboard of the tip node, e.g. 0.99 (ua_end_radius=0.99)"
    )


def _check_legacy_aerodyn(
    aero: FastInputFile, settings: LinearizationSettings, problems: list[str]
) -> None:
    """Flag a legacy-format AeroDyn deck that will not linearize as configured.

    Mirrors the edits of :func:`~vane.run.configure_linearization` on a legacy deck:
    ``unsteady_aero_mode=0`` yields ``AFAeroMod=1`` and ``dynamic_wake_mode`` of
    ``-1``/``0`` turns a dynamic-BEMT ``WakeMod=2`` into plain BEMT.
    """
    unsteady = settings.unsteady_aero_mode
    dynamic_wake = settings.dynamic_wake_mode
    airfoil = _LEGACY_STEADY_AIRFOIL if unsteady == 0 else _read_int(aero, "AFAeroMod")
    if unsteady is not None and unsteady != 0:
        problems.append(
            f"unsteady_aero_mode={unsteady} cannot be applied to this legacy-format "
            "AeroDyn deck (AFAeroMod/UAMod keywords), which linearizes only with "
            "steady airfoil aerodynamics; use unsteady_aero_mode=0 (AFAeroMod=1)"
        )
    elif airfoil is not None and airfoil != _LEGACY_STEADY_AIRFOIL:
        problems.append(
            f"legacy AeroDyn airfoil model (effective AFAeroMod={airfoil}) is not "
            "linearizable; AFAeroMod must be 1 (steady); set unsteady_aero_mode=0"
        )
    wake = _read_int(aero, "WakeMod")
    if wake == _LEGACY_WAKE_DBEMT and dynamic_wake in (-1, 0):
        wake = _LEGACY_WAKE_BEMT
    if dynamic_wake == 3:
        problems.append(
            "dynamic_wake_mode=3 (continuous DBEMT) cannot be applied to this "
            "legacy-format AeroDyn deck, whose dynamic BEMT (WakeMod=2) is not "
            "linearizable; use dynamic_wake_mode=-1 (frozen BEMT wake)"
        )
    elif wake is not None and wake not in (0, _LEGACY_WAKE_BEMT):
        problems.append(
            f"legacy AeroDyn wake model (effective WakeMod={wake}) is not "
            "linearizable; use WakeMod=0 or 1; dynamic_wake_mode=-1 replaces a "
            "dynamic BEMT wake (WakeMod=2) with a frozen BEMT wake"
        )


def _check_beamdyn(deck: FastInputFile, problems: list[str]) -> None:
    """Flag a BeamDyn blade with the pitch actuator on (OpenFAST v4 rejects it)."""
    if _read_int(deck, "CompElast") != _COMP_ELAST_BEAMDYN:
        return  # BeamDyn blade files are read only with CompElast = 2
    seen: set[Path] = set()
    for key in _BEAMDYN_BLADE_KEYS:
        blade = _subfile(deck, key)
        if blade is None or blade.path in seen:
            continue
        seen.add(blade.path)
        if _read_flag(blade, "UsePitchAct"):
            problems.append(
                f"BeamDyn pitch actuator is on (UsePitchAct=True in {blade.path.name}),"
                " which cannot be used for linearization; set UsePitchAct=False"
            )


def _check_moordyn(deck: FastInputFile, problems: list[str]) -> None:
    """Flag MoorDyn lines using the VIV or viscoelastic model."""
    if _read_int(deck, "CompMooring") != _COMP_MOORING_MOORDYN:
        return
    path = _subfile_path(deck, "MooringFile")
    if path is None:
        return
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    unsupported = _moordyn_unlinearizable_lines(text)
    if unsupported:
        problems.append(
            "MoorDyn lines use the VIV (line-type Cl > 0) or viscoelastic "
            f"(bar-separated EA) model ({', '.join(unsupported)}), which "
            "linearization does not support"
        )


def _moordyn_unlinearizable_lines(text: str) -> list[str]:
    """Return the MoorDyn lines whose line type enables VIV or viscoelasticity.

    Parses the line-type dictionary and the line list the way MoorDyn does: a ``---``
    header names the section, the next two rows are column labels and units, and the
    data rows run until the next ``---`` header.
    """
    rows = [line.split("#", 1)[0].strip() for line in text.splitlines()]
    flagged_types: dict[str, str] = {}
    lines: list[tuple[str, str]] = []
    index = 0
    while index < len(rows):
        header = rows[index].upper()
        index += 1
        if "---" not in header:
            continue
        is_types = any(name in header for name in _MOORDYN_TYPE_HEADERS)
        is_lines = not is_types and any(
            name in header for name in _MOORDYN_LINE_HEADERS
        )
        if not (is_types or is_lines):
            continue
        index += 2  # column labels and units
        while index < len(rows) and "---" not in rows[index]:
            fields = rows[index].split()
            index += 1
            if not fields:
                continue
            if is_types:
                reason = _moordyn_type_reason(fields)
                if reason is not None:
                    flagged_types[fields[0]] = reason
            elif len(fields) > _MOORDYN_LINE_TYPE_COLUMN:
                lines.append((fields[0], fields[_MOORDYN_LINE_TYPE_COLUMN]))
    return [
        f"line {line_id}: {flagged_types[line_type]}"
        for line_id, line_type in lines
        if line_type in flagged_types
    ]


def _moordyn_type_reason(fields: list[str]) -> str | None:
    """Return why a MoorDyn line-type row is not linearizable, or ``None``."""
    if len(fields) > _MOORDYN_EA_COLUMN and "|" in fields[_MOORDYN_EA_COLUMN]:
        return f"type {fields[0]} viscoelastic"
    if len(fields) >= _MOORDYN_MIN_VIV_COLUMNS:
        try:
            lift = fortran_float(fields[_MOORDYN_CL_COLUMN])
        except ValueError:
            return None
        if lift > 0.0:
            return f"type {fields[0]} VIV"
    return None


def _has_potfile(hydro: FastInputFile, suffix: str) -> bool:
    """Return whether the WAMIT ``PotFile`` root has a sibling with ``suffix``."""
    if "PotFile" not in hydro:
        return False
    try:
        root = hydro.resolve("PotFile")
    except FastInputError:
        return False
    return root.with_name(root.name + suffix).is_file()


def _effective(override: int | None, deck_value: int | None) -> int | None:
    """Return the value in force: the settings override, else the deck's value."""
    return override if override is not None else deck_value


def _subfile_path(deck: FastInputFile, key: str) -> Path | None:
    """Return the existing sub-file path referenced by ``key``, or ``None``."""
    if key not in deck:
        return None
    try:
        path = deck.resolve(key)
    except FastInputError:
        return None
    return path if path.is_file() else None


def _subfile(deck: FastInputFile, key: str) -> FastInputFile | None:
    """Return the sub-file referenced by ``key``, or ``None`` if absent/unusable."""
    path = _subfile_path(deck, key)
    if path is None:
        return None
    try:
        return FastInputFile.read(path)
    except FastInputError:
        return None


def _subfile_float(deck: FastInputFile, key: str, keyword: str) -> float | None:
    """Return a numeric ``keyword`` from the sub-file referenced by ``key``."""
    sub_file = _subfile(deck, key)
    return None if sub_file is None else _read_float(sub_file, keyword)


def _read_flag(deck: FastInputFile, keyword: str) -> bool:
    """Return whether the flag ``keyword`` is present and true (a Fortran logical)."""
    if keyword not in deck:
        return False
    try:
        return _fortran_true(deck.get(keyword))
    except FastInputError:
        return False


def _read_float(deck: FastInputFile, keyword: str) -> float | None:
    """Return ``keyword`` as a finite float, or ``None`` if absent or non-numeric."""
    if keyword not in deck:
        return None
    try:
        value = fortran_float(deck.get(keyword))
    except (FastInputError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _read_int(deck: FastInputFile, keyword: str) -> int | None:
    """Return ``keyword`` as an int, or ``None`` if absent or non-numeric."""
    value = _read_float(deck, keyword)
    return None if value is None else int(value)
