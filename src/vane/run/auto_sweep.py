# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Automated operating-point sweep from rotor-performance data or settle runs."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from vane._json import write_json_atomic
from vane.io.fst_reader import _fortran_logical
from vane.io.fst_writer import FastInputError, FastInputFile
from vane.run.auto_config import (
    AutoConfiguration,
    InferredValue,
    auto_configure_sweep,
    curve_power_coefficient,
)
from vane.run.deck_files import infer_template_root
from vane.run.driver import (
    LinearizationDriver,
    LinearizationRetryPolicy,
    SweepCase,
    SweepCaseError,
    default_max_workers,
)
from vane.run.executables import select_openfast_executable
from vane.run.floating_auto import (
    FloatingSettlePlan,
    find_controller_library,
    plan_floating_settle,
)
from vane.run.lin_config import (
    MIN_CALC_STEADY_TMAX,
    SETTLE_EDITED_SUBFILE_KEYS,
    LinearizationSettings,
    _validate_operating_configuration,
    needs_rated_wind_speed,
    require_inputs_within,
)
from vane.run.operating_points import (
    OperatingPoint,
    PlatformPosition,
    operating_curve_from_csv,
    operating_points_from_curve,
)
from vane.run.preflight import validate_deck
from vane.run.steady_state import (
    SteadyStateError,
    _hydrodyn_with_damping_matrix,
    _unsettled_message,
    compare_settled_to_curve,
    solve_steady_state,
    validate_settle_schedule,
)
from vane.utils.fs_retry import rmtree_with_retry
from vane.utils.paths import resolve_path

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vane.run.deck_files import TemplateCopyMode
    from vane.run.steady_state import (
        EquilibriumTolerances,
        SettleConvergence,
        SettleCriterion,
        SettlePlatformDamping,
    )
    from vane.run.trim_tuning import TrimTuningPolicy
    from vane.run.trim_verification import TrimTolerances

__all__ = [
    "FLOATING_TRIM_PERIODS",
    "OperatingSweepResult",
    "SettleFailure",
    "SettleFailureReason",
    "SweepSettleError",
    "floating_trim_time",
    "run_operating_sweep",
]

logger = logging.getLogger(__name__)

#: File the sweep writes its auto-configuration record to, inside ``output_dir``.
AUTO_CONFIGURATION_FILE = "auto_configuration.json"
#: Periods of the slowest platform mode a floating linearization trim runs for.
FLOATING_TRIM_PERIODS = 20


def floating_trim_time(period: float, max_sim_time: float) -> float:
    """Return the CalcSteady run time of a floating linearization case.

    The trim must outlast the slow platform motion the settled initial conditions
    leave, so the run spans ``FLOATING_TRIM_PERIODS`` periods of the slowest platform
    mode, within ``[MIN_CALC_STEADY_TMAX, max_sim_time]``.

    Parameters
    ----------
    period : float
        Slowest platform natural period (s), from the standstill platform probe.
    max_sim_time : float
        Upper bound (s): the retry ladder's ``max_sim_time``.

    Returns
    -------
    float
        The run time ``TMax`` (s).

    Raises
    ------
    ValueError
        If ``period`` or ``max_sim_time`` is not finite and positive.

    Notes
    -----
    On the IEA 15-MW semi-submersible (slowest standstill period 125.5 s), settled
    cases linearized at the rotor-scaled ``TrimTol`` (7.1e-3) trimmed at 6, 8 and
    12 m/s within 2400 s (about 19 periods; rotor-speed spread 0.08--0.09 %), but
    only at 8 m/s within 1200 s and at none within 600 s. Loosening ``TrimTol`` to
    0.021 instead let the 6 m/s trim stop with a 6.6 % rotor-speed spread.
    """
    if not (math.isfinite(period) and period > 0.0):
        msg = f"platform period must be finite and positive, got {period}"
        raise ValueError(msg)
    if not (math.isfinite(max_sim_time) and max_sim_time > 0.0):
        msg = f"max_sim_time must be finite and positive, got {max_sim_time}"
        raise ValueError(msg)
    target = FLOATING_TRIM_PERIODS * period
    return min(
        max(target, MIN_CALC_STEADY_TMAX), max(max_sim_time, MIN_CALC_STEADY_TMAX)
    )


@dataclass(frozen=True)
class _SettleJudgement:
    """How each settle run is judged, and the optional two-stage accelerator."""

    criterion: SettleCriterion = "auto"
    platform_damping: SettlePlatformDamping | None = None
    confirm_time: float | None = None
    tolerances: EquilibriumTolerances | None = None
    prescribed_rotor: bool = False
    platform_period: float | None = None


#: Why a settle point is listed in :attr:`OperatingSweepResult.settle_failures`.
SettleFailureReason = Literal["not_settled", "failed", "error"]


def _json_safe(value: object) -> object:
    """Return ``value`` with non-finite floats as ``None`` (strict JSON)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class SettleFailure:
    """A settle point of a sweep that did not reach a confirmed equilibrium.

    A sweep never aborts because one wind does not settle: the point is recorded here
    with its diagnostics, the other winds carry on, and only the points that settled
    are linearized (unless ``require_settled=False``; see
    :func:`run_operating_sweep`).

    Parameters
    ----------
    wind_speed : float
        The wind speed (m/s) of the point.
    reason : {"not_settled", "failed", "error"}
        ``"not_settled"``: the settle ran to completion but its criterion was not met
        (or could not be assessed). ``"failed"``: the settle could not complete
        (:class:`~vane.run.SteadyStateError`, e.g. OpenFAST failed or its output
        could not be read). ``"error"``: an unexpected exception, which
        :func:`run_operating_sweep` re-raises once the sweep is complete.
    message : str
        A one-line description of what went wrong.
    linearized : bool
        Whether the point was linearized anyway (``"not_settled"`` with
        ``require_settled=False``, at the settle's last mean).
    operating_point : OperatingPoint or None, optional
        The settle's last mean operating point (with its ``settle_convergence``)
        when the settle ran to completion; ``None`` otherwise.
    convergence : SettleConvergence or None, optional
        The settle evidence: the last RMS, every attempt with its horizon and
        window, the criterion, and for a two-stage settle the confirmation.
        ``None`` when the settle failed before any run was judged.
    """

    wind_speed: float
    reason: SettleFailureReason
    message: str
    linearized: bool = False
    operating_point: OperatingPoint | None = None
    convergence: SettleConvergence | None = None

    def to_dict(self) -> dict[str, object]:
        """Return the failure as a strict-JSON dictionary (unmeasured RMS as null).

        Returns
        -------
        dict[str, object]
            ``wind_speed``, ``reason``, ``message``, ``linearized``, the settled
            ``rotor_speed_rpm``/``blade_pitch_deg``/``platform`` when measured, and
            the ``convergence`` evidence.
        """
        point = self.operating_point
        record: dict[str, object] = {
            "wind_speed": self.wind_speed,
            "reason": self.reason,
            "message": self.message,
            "linearized": self.linearized,
            "rotor_speed_rpm": None if point is None else point.rotor_speed_rpm,
            "blade_pitch_deg": None if point is None else point.blade_pitch_deg,
            "platform": None
            if point is None or point.platform is None
            else asdict(point.platform),
            "convergence": None
            if self.convergence is None
            else self.convergence.to_dict(),
        }
        safe = _json_safe(record)
        assert isinstance(safe, dict)
        return safe


class SweepSettleError(SteadyStateError):
    """Raised at the end of a sweep whose settle stage did not fully succeed.

    Raised only after every wind has been settled and every usable point linearized,
    so no completed work is lost: the full :class:`OperatingSweepResult` is attached
    as :attr:`result`. Raised when ``strict_settle=True`` and any point did not
    settle, or whenever a settle worker hit an unexpected exception (the first is
    chained as ``__cause__``).

    Parameters
    ----------
    message : str
        The error message.
    result : OperatingSweepResult
        The completed sweep result, including its ``settle_failures``.
    """

    def __init__(self, message: str, *, result: OperatingSweepResult) -> None:
        super().__init__(message)
        self.result = result


@dataclass
class OperatingSweepResult:
    """Result of a fully automated settle/linearize operating sweep.

    Parameters
    ----------
    operating_points : list[OperatingPoint]
        The linearized operating points, in ``wind_speeds`` order. Settle-derived
        points carry their ``settle_convergence`` and, when the settle runs were seeded
        from an operating curve, their ``settle_curve_check``. A settle point that was
        not linearized is absent here and listed in ``settle_failures``.
    cases : list[SweepCase]
        The linearization cases, one per operating point.
    settle_structural_dofs : dict[str, bool] or None, optional
        For a settle sweep, the template ElastoDyn flags the settle runs ran with for
        every structural DOF the linearization sets (the settle copy keeps the deck's
        flags; see :func:`run_operating_sweep`). ``None`` without settle runs.
    settle_failures : tuple of SettleFailure, optional
        The settle points that did not reach a confirmed equilibrium, in
        ``wind_speeds`` order, with their diagnostics. Empty when every point settled.
    floating_plan : FloatingSettlePlan or None, optional
        For a settle sweep, how the settle runs were set up automatically: the
        controller library used (or that none was loadable, so the rotor was
        prescribed from the operating curve), the settle damping and the platform
        probe it was sized from (see :func:`~vane.run.plan_floating_settle`).
        ``None`` without settle runs.
    auto_configuration : AutoConfiguration or None, optional
        Every value the sweep inferred from the deck and curve (template root,
        OpenFAST executable, rated wind speed and torque, pitch trim gain, module
        settings), each with its source, and the warnings raised while inferring
        them. ``None`` with ``auto_configure=False``.
    """

    operating_points: list[OperatingPoint]
    cases: list[SweepCase]
    settle_structural_dofs: dict[str, bool] | None = None
    settle_failures: tuple[SettleFailure, ...] = ()
    floating_plan: FloatingSettlePlan | None = None
    auto_configuration: AutoConfiguration | None = None


def _has_rotating_point(
    wind_speeds: Sequence[float],
    *,
    curve_wind: Sequence[float] | None,
    curve_rotor_speed: Sequence[float] | None,
    settle: bool,
) -> bool:
    """Return whether any requested point will have a spinning rotor.

    With an operating curve the rotor speed is read from the curve at each wind (a
    parked schedule is zero). A settle-derived point at positive wind is taken as
    rotating, since the controller sets its speed, unless the curve parks every
    point: a settle without a controller library prescribes the curve's (zero)
    speeds, and a controller that spins the rotor anyway is still caught when each
    case is configured.
    """
    winds = [float(wind) for wind in wind_speeds]
    if curve_wind is not None and curve_rotor_speed is not None:
        speeds = np.interp(
            winds,
            np.asarray(curve_wind, dtype=np.float64),
            np.asarray(curve_rotor_speed, dtype=np.float64),
        )
        if not settle or not np.any(speeds > 0.0):
            return bool(np.any(speeds > 0.0))
    return any(wind > 0.0 for wind in winds)


def run_operating_sweep(
    fst_path: str | Path,
    output_dir: str | Path,
    wind_speeds: Sequence[float],
    *,
    curve_wind: Sequence[float] | None = None,
    curve_rotor_speed: Sequence[float] | None = None,
    curve_pitch: Sequence[float] | None = None,
    performance_csv: str | Path | None = None,
    settle: bool = False,
    settle_time: float = 1200.0,
    discard_time: float = 600.0,
    max_settle_time: float | None = None,
    controller_library: str | Path | None = None,
    settle_initial_platform: PlatformPosition
    | Mapping[float, PlatformPosition]
    | None = None,
    settle_curve_tolerances: TrimTolerances | None = None,
    settle_criterion: SettleCriterion = "auto",
    settle_platform_damping: SettlePlatformDamping | None = None,
    settle_confirm_time: float | None = None,
    settle_equilibrium_tolerances: EquilibriumTolerances | None = None,
    settle_auto_damping: bool = True,
    controller_search_paths: Sequence[str | Path] = (),
    settings: LinearizationSettings | None = None,
    structural_dofs: dict[str, bool] | None = None,
    trim_tuning: TrimTuningPolicy | None = None,
    retry_policy: LinearizationRetryPolicy | None = None,
    executable: str | Path | None = None,
    template_root: str | Path | None = None,
    template_copy: TemplateCopyMode = "referenced",
    timeout: float | None = None,
    max_workers: int | None = None,
    reserve_cpus: int = 1,
    require_settled: bool = True,
    strict_settle: bool = False,
    strict_cases: bool = False,
    validate: bool = True,
    auto_configure: bool = True,
) -> OperatingSweepResult:
    """Derive operating points and run an OpenFAST linearization sweep.

    The function refuses to guess the rotor speed and blade pitch. It needs a
    rotor-performance source: an operating curve (``curve_wind``/``curve_rotor_speed``/
    ``curve_pitch``) or a rotor-performance CSV (``performance_csv``) -- not both --
    and/or ``settle=True`` to run the fully coupled controller first and read the
    operating point from the settled time series. Without any source, or with both a
    curve and a CSV, the sweep fails before any OpenFAST process is launched.

    Everything else is configured automatically by default (``auto_configure``):
    ``run_operating_sweep(fst_path, output_dir, wind_speeds, performance_csv=csv)``
    is a complete call. The copy root, the OpenFAST executable, the module settings
    and the turbine constants (rated wind speed, rated generator torque, pitch trim
    gain) are read from the deck and the curve, each logged and recorded with its
    source; anything that still cannot be made linearizable fails before launch
    with the keyword and its remedy.

    The combinations mean:

    * curve or CSV alone -- the curve, interpolated at each wind speed, is the
      linearization point.
    * ``settle=True`` alone -- each settle run starts from 0 rpm and 0 deg collective
      pitch (written to the settle copy's ElastoDyn ``RotSpeed``/``BlPitch``) and the
      deck's platform position; the settled point is linearized.
    * ``settle=True`` with a curve or CSV -- the curve is used **only as the settle
      runs' initial conditions** (``RotSpeed`` and ``BlPitch(1..3)`` at each wind
      speed), so the rotor does not have to spin up from rest and start-up transients
      are smaller. The controller still decides where the turbine settles and the
      **settled** point is linearized, never the curve value. After each settle the
      settled rotor speed and pitch are compared with the curve
      (``settle_curve_tolerances``); a deviation outside tolerance is logged as a
      warning and recorded on the point's ``settle_curve_check`` (and in the run
      provenance). It means the controller and the curve disagree and is not an
      error. Every requested wind speed must lie inside the curve's range.

    The platform offset is never estimated from the curve (VANE computes no thrust);
    it starts at the deck's values unless ``settle_initial_platform`` supplies it.

    Parameters
    ----------
    controller_library : str or pathlib.Path or None, optional
        Bladed-style controller library (e.g. a ``libdiscon.dll`` built for
        this host) for the ``settle=True`` runs. Each settle case's ServoDyn
        ``DLL_FileName`` is rewritten to it in the case copy only; the template
        deck is never edited. ``None`` (the default) finds one automatically (see
        :func:`~vane.run.find_controller_library`): the deck's own
        ``DLL_FileName`` when it loads on this host, else a host build of the same
        library next to it or in ``controller_search_paths``,
        ``VANE_CONTROLLER_LIBRARIES`` or the configuration file. When none exists
        the sweep does not fail: each settle runs with the rotor **prescribed** at
        the operating curve's speed and pitch (no controller; ``prescribed_rotor``
        of :func:`~vane.run.solve_steady_state`) and the point is linearized by
        CalcSteady from the platform equilibrium it finds -- which requires an
        operating curve or ``performance_csv``. The decision is recorded in
        :attr:`OperatingSweepResult.floating_plan`. Only valid with
        ``settle=True`` (the linearization replaces the controller by the trim and
        never loads the library).
    settle_initial_platform : PlatformPosition or Mapping or None, optional
        Initial platform position for the settle runs: one
        :class:`~vane.run.PlatformPosition` for every wind speed, or a mapping from
        wind speed to position (winds absent from it keep the deck's position). Only
        the DOFs given are written. It is an initial condition only: a rough estimate
        (e.g. the mean surge and pitch of an earlier run) shortens the platform's
        drift to its offset without changing where it settles. ``None`` (the default)
        keeps the deck's platform position. Only valid with ``settle=True``; every
        mapping key must be a requested wind speed.
    settle_curve_tolerances : TrimTolerances or None, optional
        Tolerances of the settled-versus-curve comparison (rotor speed relative,
        pitch absolute; see :func:`~vane.run.compare_settled_to_curve`). ``None``
        uses the :class:`~vane.run.TrimTolerances` defaults (1 %, 0.5 deg). Only
        valid with ``settle=True`` and a curve or CSV.
    settle_criterion : {"auto", "velocity", "mean_drift"}, optional
        How each settle run is judged settled (see
        :func:`~vane.run.solve_steady_state`): ``"auto"`` (the default) starts
        with the velocity criterion and switches to the mean-drift criterion when
        the platform carries a sustained, non-decaying oscillation; ``"velocity"``
        is platform-velocity RMS within tolerance; ``"mean_drift"`` is
        operating-point means that no longer drift between the halves of the
        analysis window. The criterion that judged each point is recorded in its
        ``settle_convergence`` and in the run provenance. Only valid with
        ``settle=True``.
    settle_platform_damping : SettlePlatformDamping or None, optional
        Settle accelerator for floating platforms. ``None`` (the default) sizes it
        automatically for a floating deck from a standstill platform probe (see
        :func:`~vane.run.plan_floating_settle`; ``settle_auto_damping=False``
        opts out). Each settle then runs in two
        stages, first with this temporary linear damping added to HydroDyn's
        ``AddBLin`` in the settle copy, then undamped from the stage-1 means to
        confirm the equilibrium is unchanged (see
        :func:`~vane.run.solve_steady_state` and
        :class:`~vane.run.SettlePlatformDamping`). The damping is only ever
        written to the settle case; every linearization case is copied afresh from
        the template and keeps the deck's own ``AddBLin``. The template must run
        HydroDyn with an ``AddBLin`` matrix; this is checked before any case is
        copied. Only valid with ``settle=True``.
    settle_confirm_time : float or None, optional
        ``TMax`` (s) of the undamped confirmation run of a two-stage settle. ``None``
        (the default) sizes it automatically so its analysis window holds eight
        periods of the slowest platform mode -- the platform probe's longest
        period when the probe ran, else the damped stage's dominant period --
        never shorter than 1200 s (the length without any period) and never
        longer than ``max_settle_time`` when given, else 3600 s (see
        ``confirm_time`` of :func:`~vane.run.solve_steady_state`). A decaying
        confirmation whose means have not yet returned to the damped equilibrium
        is rerun once twice as long. Used only with a settle damping.
    settle_equilibrium_tolerances : EquilibriumTolerances or None, optional
        Tolerances on a change of the settled means (damped versus undamped stage,
        and window halves for ``"mean_drift"``). ``None`` uses the
        :class:`~vane.run.EquilibriumTolerances` defaults. Only valid with
        ``settle=True``.
    settle_auto_damping : bool, optional
        For a floating deck without ``settle_platform_damping``, size the settle
        damping at half of critical from a standstill platform probe (two short
        linearizations in ``output_dir``; see :func:`~vane.run.probe_platform`).
        Default ``True``; ``False`` settles undamped.
    controller_search_paths : Sequence of str or pathlib.Path, optional
        Extra controller-library files or directories searched when
        ``controller_library`` is not given.
    performance_csv : str or pathlib.Path or None, optional
        Path to a rotor-performance CSV (see :func:`~vane.run.operating_curve_from_csv`
        for the column spec). When given it is parsed into the operating curve; it is
        mutually exclusive with the ``curve_*`` arrays. With ``settle=True`` it seeds
        the settle runs (see above).
    structural_dofs : dict[str, bool] or None, optional
        ElastoDyn structural-DOF flags to set in each case's ED sub-file. ``None`` (the
        default) keeps the ``settings`` object's own value, whose default disables
        ``YawDOF`` and ``DrTrDOF`` for a reliable trim (dropping the spurious
        free-yaw mode and the ~0.62 Hz drivetrain-torsion mode, the latter a deliberate
        trade-off for trim convergence). Pass a mapping to override — it is merged on
        top of the defaults, so ``{"DrTrDOF": True}`` re-enables the drivetrain-torsion
        mode while still disabling ``YawDOF``. A flag listed here but absent from the ED
        deck is an error. These flags apply to the linearization cases only: the
        settle copies keep the template deck's own ElastoDyn DOF flags, because a
        settle runs the deck's real controller on the model it was tuned for (a
        controller's filters assume the deck's drivetrain; the yaw drive holds the
        nacelle through ServoDyn). The settled mean rotor speed, pitch and platform
        position -- all that is carried to the linearization -- do not depend on the
        drivetrain or yaw flexibility. A difference between the two sets is logged,
        and the settle flags are recorded in
        :attr:`OperatingSweepResult.settle_structural_dofs`.
    retry_policy : LinearizationRetryPolicy or None, optional
        The convergence retry ladder for the linearization cases (see
        :class:`~vane.run.LinearizationRetryPolicy`), e.g. a larger ``max_sim_time``
        for a slowly settling floating platform. ``None`` (the default) uses the
        default policy.
    trim_tuning : TrimTuningPolicy or None, optional
        When set, each operating point's starting trim tolerance and damping are tuned
        for its wind-speed region before the convergence retry ladder runs (see
        :func:`~vane.run.tune_trim_settings`): at/above-rated points start from the
        looser-tolerance, higher-damping floors those harder trims need, so they tend to
        converge on the first attempt rather than after a reactive retry. ``None`` (the
        default) starts every point from ``settings`` unchanged. Tuning only raises the
        per-case ``trim_tol``/``twr_kdmp``/``bld_kdmp`` toward the floors, so the
        sweep's provenance is unaffected.
    settings : LinearizationSettings or None, optional
        The linearization settings. ``None`` (the default) configures the deck
        automatically (with ``auto_configure``): the
        :meth:`~vane.run.LinearizationSettings.offshore` module set, adapted to the
        deck -- a HydroDyn radiation model whose ``.ss`` file is missing keeps
        ``RdtnMod`` 2 when the file can be fitted from the WAMIT ``.1`` data (it is
        then generated in each case copy, ``hydro_radiation_fit``), a radiation or
        excitation model whose state-space file cannot be had falls back to
        ``RdtnMod``/``ExctnMod`` 0 with a recorded warning, a deck's own
        linearizable unsteady-aerodynamics model (``UA_Mod`` 4-6) is kept with
        ``UAEndRad`` 0.99 where it would reach the tip node under tip loss, and a
        module the deck does not use is left
        untouched. Settings passed explicitly are used as given, except that the
        turbine constants they leave unset are still inferred (see
        ``auto_configure``).
    executable : str or pathlib.Path or None, optional
        The OpenFAST executable. ``None`` (the default) chooses one whose release
        reads the deck's input-format generation, from the ``VANE_OPENFAST_EXECUTABLES``
        environment variable, the file named by ``VANE_OPENFAST_CONFIG`` (default
        ``~/.vane/openfast_executables.txt``) and ``PATH`` (see
        :func:`~vane.run.select_openfast_executable`); the sweep fails before any
        case is copied, naming the release the deck needs, when none can. An
        explicit executable is used as given, but is rejected up front when its
        version is known not to read the deck.
    template_root : str or pathlib.Path or None, optional
        Directory the deck's files are copied from per case. ``None`` (the default)
        uses the deck's own directory, or -- when the deck references files outside
        it (``../5MW_Baseline/...``) and ``auto_configure`` is on -- the deepest
        directory that holds the deck and every file it references by a relative
        path (:func:`~vane.run.deck_files.infer_template_root`). An explicit value
        always wins.
    template_copy : {"referenced", "tree"}, optional
        ``"referenced"`` (the default) copies only the files the deck references into
        each case (settle cases included); ``"tree"`` copies the whole
        ``template_root``. See :class:`~vane.run.LinearizationDriver`.
    require_settled : bool, optional
        What to do with a settle point that does not reach a confirmed equilibrium
        (its criterion not met, or not assessable). ``True`` (the default): do not
        linearize it. ``False``: linearize it at the settle's last mean. Either way
        the sweep carries on with the other winds -- one unsettled point never
        aborts it -- and the point is listed in
        :attr:`OperatingSweepResult.settle_failures` with its diagnostics (and, when
        linearized, its settle evidence is in the case provenance). A settle that
        cannot complete (OpenFAST fails, unreadable output) is likewise recorded and
        never linearized.
    strict_settle : bool, optional
        When ``True``, raise :class:`SweepSettleError` once the sweep is complete if
        any settle point is in ``settle_failures``; the full result is attached as
        its ``result``. Default ``False`` (the failures are logged and returned).
        An *unexpected* exception in a settle worker is always re-raised this way,
        after the other winds have been settled and linearized.
    strict_cases : bool, optional
        How a linearization case that raises an unexpected exception (a file-system
        error while its case is copied or configured, say) is reported. Such a case
        never aborts the sweep: it is recorded in :attr:`OperatingSweepResult.cases`
        as failed (``converged=False``, ``failure_mode == "case_error"``) and the
        other points run to completion (see
        :meth:`vane.run.LinearizationDriver.run_sweep`). Default ``False``: the
        failure is logged and the result returned. ``True``: raise
        :class:`vane.run.SweepCaseError` once the sweep is complete, with the full
        result attached as its ``result``.
    validate : bool, optional
        When ``True`` (the default) the template deck is checked under the effective
        settings (:func:`~vane.run.validate_deck`) and every rotating curve point's
        trim configuration is checked before any case is copied, and each
        configured case again before its OpenFAST run.
    auto_configure : bool, optional
        When ``True`` (the default) the sweep completes its configuration from the
        user's own deck and curve, logging every value and recording it with its
        source in :attr:`OperatingSweepResult.auto_configuration`, in the driver's
        run provenance and in ``output_dir/auto_configuration.json``
        (:func:`~vane.run.auto_configure_sweep`): the template root and executable
        (above), the module settings when ``settings`` is ``None``, and the turbine
        constants left unset -- ``rated_wind_speed`` (where the curve's rotor speed
        and power reach their rated plateau), ``rated_generator_torque`` (controller
        ``DISCON.IN`` ``VS_RtTq``, else ServoDyn ``VS_RtTq`` with ``VSContrl = 1``,
        else the curve's ``cp`` at rated) and a ``trim_gain_pitch`` left at its
        class default (scaled as ``2e-4 * (63 m / TipRad)**3``). VANE computes no
        rotor performance in doing so. ``False`` restores the explicit behaviour:
        nothing is inferred, ``settings=None`` means
        :meth:`~vane.run.LinearizationSettings.offshore`, and ``executable=None``
        means ``"openfast"`` on ``PATH``.

    Raises
    ------
    SweepSettleError
        After the sweep completes, with ``strict_settle=True`` and a settle failure,
        or when a settle worker raised an unexpected exception.
    SweepCaseError
        After the sweep completes, with ``strict_cases=True`` and a linearization
        case that raised an unexpected exception.
    ValueError
        Before any launch, for a malformed request or a rotating sweep whose rated
        wind speed is neither given nor inferable.
    vane.run.DeckValidationError
        Before any launch, when ``validate`` is on and the template cannot be
        linearized under the effective settings; every problem is listed with the
        keyword and its remedy.
    vane.io.FastInputError
        Before any launch, when a rotating curve point's trim cannot be configured
        (e.g. no rated generator torque for an above-rated point).
    vane.run.OpenFastError
        Before any launch, when no OpenFAST executable can read the deck.
    """
    workers = (
        default_max_workers(reserve_cpus=reserve_cpus)
        if max_workers is None
        else max_workers
    )
    if workers < 1:
        msg = f"max_workers must be >= 1, got {workers}"
        raise ValueError(msg)

    # A performance CSV is parsed to the operating curve up front. The CSV and the
    # curve arrays are two forms of the same source, so only one may be given; either
    # may be combined with settle=True, where it seeds the settle runs.
    curve_given = any(
        item is not None for item in (curve_wind, curve_rotor_speed, curve_pitch)
    )
    if performance_csv is not None:
        if curve_given:
            msg = (
                "provide one operating curve: performance_csv is mutually exclusive "
                "with curve_wind/curve_rotor_speed/curve_pitch"
            )
            raise ValueError(msg)
        curve_wind, curve_rotor_speed, curve_pitch = operating_curve_from_csv(
            performance_csv
        )
        curve_given = True
    if settle_initial_platform is not None and not settle:
        msg = (
            "settle_initial_platform applies only to settle=True runs; it is an "
            "initial condition of the settle, not of the linearization"
        )
        raise ValueError(msg)
    if settle_curve_tolerances is not None and not (settle and curve_given):
        msg = (
            "settle_curve_tolerances applies only to settle=True runs seeded from an "
            "operating curve or performance_csv"
        )
        raise ValueError(msg)

    judgement = _SettleJudgement(
        criterion=settle_criterion,
        platform_damping=settle_platform_damping,
        confirm_time=settle_confirm_time,
        tolerances=settle_equilibrium_tolerances,
    )
    if judgement != _SettleJudgement() and not settle:
        msg = (
            "settle_criterion, settle_platform_damping, settle_confirm_time and "
            "settle_equilibrium_tolerances apply only to settle=True runs"
        )
        raise ValueError(msg)

    inferred: list[InferredValue] = []
    if (
        auto_configure
        and template_root is None
        and Path(fst_path).is_file()
        and (root := infer_template_root(fst_path)) != resolve_path(fst_path).parent
    ):
        # The deck references inputs outside its folder: copy from their common
        # ancestor, so every relative reference and edited sub-file is in the case.
        if resolve_path(output_dir).is_relative_to(root):
            msg = (
                f"output_dir {output_dir} lies inside {root}, the directory the deck's "
                "cases must be copied from (it references files there by relative "
                "paths, e.g. ../); choose an output_dir outside it, or pass "
                "template_root explicitly"
            )
            raise ValueError(msg)
        template_root = root
        inferred.append(
            InferredValue(
                "template_root",
                str(root),
                "deck references",
                "the deepest directory holding the deck and every file it "
                "references by a relative path",
            )
        )
        logger.info(
            "auto-configured template_root = %s (the deck references files outside "
            "its own directory)",
            root,
        )
    _validate_sweep_inputs(
        Path(fst_path),
        wind_speeds,
        curve_wind=curve_wind,
        curve_rotor_speed=curve_rotor_speed,
        curve_pitch=curve_pitch,
        settle=settle,
        template_root=Path(template_root) if template_root is not None else None,
    )
    if controller_library is not None and not settle:
        msg = (
            "controller_library applies only to settle=True runs; the linearization "
            "replaces the controller by the trim and never loads the library"
        )
        raise ValueError(msg)
    if settle_platform_damping is not None:
        # The damping goes into each settle copy's HydroDyn AddBLin; fail before any
        # case is copied when the template has none.
        _hydrodyn_with_damping_matrix(Path(fst_path))

    auto: AutoConfiguration | None = None
    if auto_configure:
        effective_settings, auto = auto_configure_sweep(
            fst_path,
            settings,
            curve_wind=curve_wind,
            curve_rotor_speed=curve_rotor_speed,
            curve_cp=(
                curve_power_coefficient(performance_csv, curve_wind)
                if performance_csv is not None and curve_wind is not None
                else None
            ),
        )
        auto = AutoConfiguration(tuple(inferred)).extended(auto.inferred, auto.warnings)
    else:
        effective_settings = settings or LinearizationSettings.offshore()
    if (
        effective_settings.rated_wind_speed is None
        and needs_rated_wind_speed(effective_settings, fst_path)
        and _has_rotating_point(
            wind_speeds,
            curve_wind=curve_wind,
            curve_rotor_speed=curve_rotor_speed,
            settle=settle,
        )
    ):
        # Fail at the boundary, before any settle or linearization run is launched;
        # only a rotating point with trim or controller handling needs the region.
        reason = (
            ""
            if auto is None
            else " "
            + "; ".join(
                warning for warning in auto.warnings if "rated_wind_speed" in warning
            )
        )
        msg = (
            "rated_wind_speed is required for a rotating operating sweep (it selects "
            "torque or pitch trim) and could not be inferred: it is read from the "
            "operating curve (performance_csv or curve_*), where the rotor speed "
            "reaches its rated plateau, so supply a curve that extends past rated or "
            "pass settings with rated_wind_speed set for the turbine, e.g. "
            "LinearizationSettings.offshore(rated_wind_speed=10.59)." + reason
        )
        raise ValueError(msg.rstrip(" ."))
    if structural_dofs is not None:
        effective_settings = replace(
            effective_settings, structural_dofs=structural_dofs
        )
    if validate:
        # Every blocking linearization problem of the template under these settings
        # is reported now, before any case is copied or OpenFAST launched.
        validate_deck(fst_path, effective_settings)
    if not settle and curve_wind is not None and effective_settings.trim_controller:
        _precheck_curve_points(
            Path(fst_path),
            wind_speeds,
            effective_settings,
            curve_wind=curve_wind,
            curve_rotor_speed=curve_rotor_speed,
            curve_pitch=curve_pitch,
        )

    resolved_executable: str | Path
    if auto_configure:
        selection = select_openfast_executable(fst_path, executable)
        resolved_executable = selection.executable
        assert auto is not None
        auto = auto.extended(
            [
                InferredValue(
                    "executable",
                    str(selection.executable),
                    selection.source,
                    f"OpenFAST {'.'.join(map(str, selection.version or ())) or '?'}"
                    f" for a {selection.generation.label} deck "
                    f"({'; '.join(selection.generation.evidence)}); requires "
                    f"{selection.generation.describe()}",
                )
            ]
        )
    else:
        resolved_executable = executable if executable is not None else "openfast"

    driver = LinearizationDriver(
        fst_path=Path(fst_path),
        output_dir=Path(output_dir),
        settings=effective_settings,
        executable=resolved_executable,
        template_root=Path(template_root) if template_root is not None else None,
        template_copy=template_copy,
        validate=validate,
        trim_tuning=trim_tuning,
        retry_policy=(
            retry_policy if retry_policy is not None else LinearizationRetryPolicy()
        ),
        auto_configuration=None if auto is None else auto.to_dict(),
    )
    if settle and controller_library is None and not curve_given:
        # Without an operating curve the rotor cannot be prescribed, so a settle needs
        # a loadable controller library: check that before anything is written or a
        # platform probe is run.
        controller = find_controller_library(
            Path(fst_path), search_paths=controller_search_paths
        )
        if controller.status == "missing":
            msg = (
                "controller library not found on this host for "
                f"{controller.deck_library!r}, and no operating curve to prescribe "
                "the rotor from: pass controller_library=..., set "
                "VANE_CONTROLLER_LIBRARIES, or supply performance_csv / curve_*"
            )
            raise SteadyStateError(msg)
    if auto is not None:
        write_json_atomic(
            Path(output_dir) / AUTO_CONFIGURATION_FILE,
            _json_safe(auto.to_dict()),
            overwrite=True,
        )
    settle_dofs = (
        _settle_structural_dofs(Path(fst_path), effective_settings) if settle else None
    )
    floating_plan: FloatingSettlePlan | None = None
    settle_library: Path | None = None
    if settle:
        # Resolve the controller library and size the settle damping from the deck
        # itself (see vane.run.floating_auto); a missing library falls back to a
        # prescribed-rotor settle seeded from the operating curve.
        # The settle schedule fails here, before the platform probe launches
        # OpenFAST, rather than in every settle worker afterwards.
        validate_settle_schedule(
            settle_time,
            discard_time,
            adaptive=max_settle_time is not None,
            max_settle_time=max_settle_time,
            criterion=settle_criterion,
            confirm_time=settle_confirm_time,
        )
        floating_plan = plan_floating_settle(
            fst_path,
            driver.validated_workspace_dir("probe"),
            controller_library=controller_library,
            platform_damping=settle_platform_damping,
            auto_damping=settle_auto_damping,
            controller_search_paths=controller_search_paths,
            template_root=template_root,
            template_copy=template_copy,
            executable=resolved_executable,
            timeout=timeout,
        )
        if floating_plan.prescribed_rotor and not curve_given:
            msg = (
                "controller library not found on this host for "
                f"{floating_plan.controller.deck_library!r}, and no operating curve "
                "to prescribe the rotor from: pass controller_library=..., set "
                "VANE_CONTROLLER_LIBRARIES, or supply performance_csv / curve_*"
            )
            raise SteadyStateError(msg)
        settle_library = floating_plan.controller_library
        judgement = replace(
            judgement,
            platform_damping=floating_plan.platform_damping,
            prescribed_rotor=floating_plan.prescribed_rotor,
            # The probe's slowest platform period sizes the undamped confirmation.
            platform_period=(
                None
                if floating_plan.probe is None
                else floating_plan.probe.slowest_period
            ),
        )
        period = (
            None if floating_plan.probe is None else floating_plan.probe.slowest_period
        )
        if period is not None and driver.settings.sim_time is None:
            sim_time = floating_trim_time(period, driver.retry_policy.max_sim_time)
            driver.settings = replace(driver.settings, sim_time=sim_time)
            record = InferredValue(
                "sim_time",
                sim_time,
                "platform probe",
                f"{FLOATING_TRIM_PERIODS} periods of the slowest platform mode "
                f"({period:.4g} s), within [{MIN_CALC_STEADY_TMAX:g} s, "
                f"{driver.retry_policy.max_sim_time:g} s]",
            )
            logger.info(
                "auto-configured sim_time = %r from platform probe (%s)",
                sim_time,
                record.detail,
            )
            if auto is not None:
                auto = auto.extended((record,), ())
                driver.auto_configuration = auto.to_dict()
                write_json_atomic(
                    Path(output_dir) / AUTO_CONFIGURATION_FILE,
                    _json_safe(auto.to_dict()),
                    overwrite=True,
                )
    derived = _derive_operating_points(
        driver,
        wind_speeds,
        curve_wind=curve_wind,
        curve_rotor_speed=curve_rotor_speed,
        curve_pitch=curve_pitch,
        settle=settle,
        settle_time=settle_time,
        discard_time=discard_time,
        max_settle_time=max_settle_time,
        controller_library=settle_library,
        executable=resolved_executable,
        timeout=timeout,
        max_workers=workers,
        require_settled=require_settled,
        initial_platform=settle_initial_platform,
        curve_tolerances=settle_curve_tolerances,
        judgement=judgement,
    )
    case_error: SweepCaseError | None = None
    cases: list[SweepCase] = []
    if derived.points:
        try:
            cases = driver.run_sweep(
                derived.points,
                timeout=timeout,
                max_workers=workers,
                strict=strict_cases,
            )
        except SweepCaseError as error:
            # Every case has finished; keep them and raise once the result is built.
            case_error = error
            cases = error.cases
    result = OperatingSweepResult(
        operating_points=derived.points,
        cases=cases,
        settle_structural_dofs=settle_dofs,
        settle_failures=tuple(derived.failures),
        floating_plan=floating_plan,
        auto_configuration=auto,
    )
    if derived.failures:
        summary = "; ".join(
            f"{failure.wind_speed:g} m/s {failure.reason}"
            + (" (linearized at its last mean)" if failure.linearized else "")
            for failure in derived.failures
        )
        logger.warning(
            "%d of %d settle point(s) did not reach a confirmed equilibrium: %s",
            len(derived.failures),
            len(wind_speeds),
            summary,
        )
    if derived.errors:
        msg = (
            f"{len(derived.errors)} settle worker(s) raised an unexpected exception "
            "(the other points were settled and linearized; see .result): "
            + "; ".join(
                failure.message
                for failure in derived.failures
                if failure.reason == "error"
            )
        )
        raise SweepSettleError(msg, result=result) from derived.errors[0]
    if case_error is not None:
        raise SweepCaseError(
            str(case_error), cases=cases, result=result
        ) from case_error.__cause__
    if strict_settle and derived.failures:
        msg = (
            "strict_settle: settle point(s) did not reach a confirmed equilibrium: "
            + "; ".join(
                f"{failure.wind_speed:g} m/s: {failure.message}"
                for failure in derived.failures
            )
        )
        raise SweepSettleError(msg, result=result)
    return result


def _settle_structural_dofs(
    fst_path: Path, settings: LinearizationSettings
) -> dict[str, bool]:
    """Return the template's flags for the DOFs the linearization sets.

    The settle copy keeps the template deck's ElastoDyn DOF flags; the linearization
    writes ``settings.resolved_structural_dofs()``. Each such flag the template deck
    carries is returned as the settle runs see it, and a difference from the
    linearization's value is logged so the two DOF sets are never confused.
    """
    try:
        deck = FastInputFile.read(fst_path)
        elastodyn = FastInputFile.read(deck.resolve("EDFile"))
    except (OSError, FastInputError):
        return {}
    flags: dict[str, bool] = {}
    for flag, linearized in settings.resolved_structural_dofs().items():
        if flag not in elastodyn:
            continue
        settled = _fortran_logical(elastodyn.get(flag))
        if settled is None:
            continue
        flags[flag] = settled
        if settled != linearized:
            logger.info(
                "settle runs keep the deck's %s=%s for its controller; the "
                "linearization uses %s=%s",
                flag,
                settled,
                flag,
                linearized,
            )
    return flags


def _validate_sweep_inputs(
    fst_path: Path,
    wind_speeds: Sequence[float],
    *,
    curve_wind: Sequence[float] | None,
    curve_rotor_speed: Sequence[float] | None,
    curve_pitch: Sequence[float] | None,
    settle: bool,
    template_root: Path | None,
) -> None:
    """Reject a malformed sweep request before any OpenFAST process is launched.

    Validates the requested wind speeds and (when given) the operating curve at the
    boundary, raising one clear error per problem, so a misconfigured sweep fails
    immediately rather than part-way through a long campaign.

    Parameters
    ----------
    fst_path : pathlib.Path
        The template primary ``.fst`` deck.
    wind_speeds : Sequence[float]
        The requested sweep wind speeds (m/s).
    curve_wind, curve_rotor_speed, curve_pitch : Sequence[float] or None
        The operating-curve arrays, when the curve source is used.
    settle : bool
        Whether the settle source is requested.
    template_root : pathlib.Path or None
        The copy-root directory, when one is given.

    Raises
    ------
    ValueError
        If ``wind_speeds`` is empty, non-finite, negative, or contains duplicates; the
        curve arrays mismatch in length, are too short, are not strictly increasing
        in wind or do not cover every requested wind speed; or ``template_root`` is
        given but does not exist or does not contain ``fst_path``.
    FileNotFoundError
        If the template ``.fst`` deck does not exist.
    vane.io.FastInputError
        If a positive wind speed is requested but the template deck has no usable
        InflowWind file to set it in (which would silently linearize in still air).
    """
    if not fst_path.is_file():
        msg = f"template .fst deck not found: {fst_path}"
        raise FileNotFoundError(msg)

    requested = [float(w) for w in wind_speeds]
    if not requested:
        msg = "wind_speeds is empty; provide at least one wind speed to linearize"
        raise ValueError(msg)
    for wind in requested:
        if not math.isfinite(wind):
            msg = f"wind_speeds must all be finite, got {requested}"
            raise ValueError(msg)
        if wind < 0.0:
            msg = f"wind_speeds must be non-negative, got {wind}"
            raise ValueError(msg)
    duplicates = sorted({w for w in requested if requested.count(w) > 1})
    if duplicates:
        msg = f"wind_speeds contains duplicate wind speeds: {duplicates}"
        raise ValueError(msg)

    if template_root is not None:
        if not template_root.is_dir():
            msg = f"template_root does not exist or is not a directory: {template_root}"
            raise ValueError(msg)
        if not resolve_path(fst_path).is_relative_to(resolve_path(template_root)):
            msg = (
                f"fst_path {fst_path} is not inside template_root {template_root}; the "
                "copy-root must contain the deck"
            )
            raise ValueError(msg)

    curve_given = any(
        item is not None for item in (curve_wind, curve_rotor_speed, curve_pitch)
    )
    if curve_given:
        _validate_curve_arrays(curve_wind, curve_rotor_speed, curve_pitch)
        assert curve_wind is not None
        assert curve_rotor_speed is not None
        assert curve_pitch is not None
        # Interpolate now: a curve that is not strictly increasing, holds a
        # non-finite value, or does not cover every requested wind fails here --
        # before a settle sweep's platform probe launches OpenFAST.
        operating_points_from_curve(
            requested,
            curve_wind=curve_wind,
            curve_rotor_speed=curve_rotor_speed,
            curve_pitch=curve_pitch,
        )

    # Any positive wind point needs a usable InflowWind file in the template to write
    # the wind into; catch its absence here rather than after copying every case.
    if any(wind > 0.0 for wind in requested) and not _has_usable_inflow(fst_path):
        msg = (
            f"a positive wind speed was requested but {fst_path} has no usable "
            "InflowWind file to set it in; the linearization would run in still air"
        )
        raise FastInputError(msg)


def _validate_curve_arrays(
    curve_wind: Sequence[float] | None,
    curve_rotor_speed: Sequence[float] | None,
    curve_pitch: Sequence[float] | None,
) -> None:
    """Reject a partial or mismatched operating curve before launch."""
    if curve_wind is None or curve_rotor_speed is None or curve_pitch is None:
        msg = (
            "curve_wind, curve_rotor_speed, and curve_pitch must be supplied together; "
            "rotor speed and blade pitch are required"
        )
        raise ValueError(msg)
    lengths = (len(curve_wind), len(curve_rotor_speed), len(curve_pitch))
    if len(set(lengths)) != 1:
        msg = (
            "curve_wind, curve_rotor_speed, and curve_pitch must have the same length, "
            f"got {lengths}"
        )
        raise ValueError(msg)
    if lengths[0] < 2:
        msg = f"the operating curve needs at least two samples, got {lengths[0]}"
        raise ValueError(msg)


def _has_usable_inflow(fst_path: Path) -> bool:
    """Return whether the template deck resolves to an existing InflowWind file."""
    try:
        deck = FastInputFile.read(fst_path)
    except (OSError, FastInputError):
        return False
    if "InflowFile" not in deck:
        return False
    try:
        inflow = deck.resolve("InflowFile")
    except FastInputError:
        return False
    return inflow.is_file()


def _precheck_curve_points(
    fst_path: Path,
    wind_speeds: Sequence[float],
    settings: LinearizationSettings,
    *,
    curve_wind: Sequence[float],
    curve_rotor_speed: Sequence[float] | None,
    curve_pitch: Sequence[float] | None,
) -> None:
    """Reject curve points the trim cannot be configured for, before any launch.

    Each rotating curve point is checked against the template deck exactly as
    :func:`~vane.run.configure_linearization` checks it per case (the generator
    torque an above-rated trim needs, the ``UAEndRad`` line ``ua_end_radius``
    needs, ...), so a configuration error common to every case fails here, naming
    the keyword to supply, instead of once per case after the sweep has started.
    """
    if curve_rotor_speed is None or curve_pitch is None:
        return
    deck = FastInputFile.read(fst_path)
    points = operating_points_from_curve(
        wind_speeds,
        curve_wind=curve_wind,
        curve_rotor_speed=curve_rotor_speed,
        curve_pitch=curve_pitch,
    )
    for point in points:
        if point.is_rotating:
            _validate_operating_configuration(point, settings, deck)


def _derive_operating_points(
    driver: LinearizationDriver,
    wind_speeds: Sequence[float],
    *,
    curve_wind: Sequence[float] | None,
    curve_rotor_speed: Sequence[float] | None,
    curve_pitch: Sequence[float] | None,
    settle: bool,
    settle_time: float,
    discard_time: float,
    max_settle_time: float | None,
    controller_library: Path | None,
    executable: str | Path,
    timeout: float | None,
    max_workers: int,
    require_settled: bool,
    initial_platform: PlatformPosition | Mapping[float, PlatformPosition] | None = None,
    curve_tolerances: TrimTolerances | None = None,
    judgement: _SettleJudgement | None = None,
) -> _DerivedPoints:
    """Return the linearization points from the curve, settle runs, or both.

    With both, the curve points seed the settle runs and the settled points are
    returned, each carrying its comparison with the curve, together with the settle
    points that did not reach equilibrium.
    """
    curve_given = any(
        item is not None for item in (curve_wind, curve_rotor_speed, curve_pitch)
    )
    curve_points: list[OperatingPoint] | None = None
    if curve_given:
        if curve_wind is None or curve_rotor_speed is None or curve_pitch is None:
            msg = (
                "curve_wind, curve_rotor_speed, and curve_pitch must be supplied "
                "together; rotor speed and blade pitch are required"
            )
            raise ValueError(msg)
        curve_points = operating_points_from_curve(
            wind_speeds,
            curve_wind=curve_wind,
            curve_rotor_speed=curve_rotor_speed,
            curve_pitch=curve_pitch,
        )
        if not settle:
            return _DerivedPoints(points=curve_points)
    if not settle:
        msg = (
            "rotor-performance data is required: supply curve_wind/"
            "curve_rotor_speed/curve_pitch, or set settle=True to derive rotor speed "
            "and blade pitch from controller settle runs"
        )
        raise ValueError(msg)
    platforms = _settle_platforms(wind_speeds, initial_platform)
    return _settled_operating_points(
        driver,
        wind_speeds,
        settle_time=settle_time,
        discard_time=discard_time,
        max_settle_time=max_settle_time,
        controller_library=controller_library,
        executable=executable,
        timeout=timeout,
        max_workers=max_workers,
        require_settled=require_settled,
        seeds=curve_points,
        platforms=platforms,
        curve_tolerances=curve_tolerances,
        judgement=judgement,
    )


def _settle_platforms(
    wind_speeds: Sequence[float],
    initial_platform: PlatformPosition | Mapping[float, PlatformPosition] | None,
) -> list[PlatformPosition | None]:
    """Return each wind speed's initial settle platform position (``None``: deck's).

    Raises
    ------
    ValueError
        If a mapping names a wind speed that is not requested (a likely typo that
        would otherwise silently leave that point at the deck's position).
    TypeError
        If a value is not a :class:`~vane.run.PlatformPosition`.
    """
    winds = [float(wind) for wind in wind_speeds]
    if initial_platform is None:
        return [None] * len(winds)
    if isinstance(initial_platform, PlatformPosition):
        return [initial_platform] * len(winds)
    if not isinstance(initial_platform, Mapping):
        msg = (
            "settle_initial_platform must be a PlatformPosition or a mapping of wind "
            f"speed to PlatformPosition, got {type(initial_platform).__name__}"
        )
        raise TypeError(msg)
    by_wind: dict[float, PlatformPosition] = {}
    for key, position in initial_platform.items():
        if not isinstance(position, PlatformPosition):
            msg = (
                f"settle_initial_platform[{key!r}] must be a PlatformPosition, got "
                f"{type(position).__name__}"
            )
            raise TypeError(msg)
        by_wind[float(key)] = position
    unknown = sorted(set(by_wind) - set(winds))
    if unknown:
        msg = (
            f"settle_initial_platform names wind speeds {unknown} that are not in "
            f"wind_speeds {winds}"
        )
        raise ValueError(msg)
    return [by_wind.get(wind) for wind in winds]


def _settle_edited_subfile_keys(
    controller_library: Path | None, *, platform_damping: bool = False
) -> tuple[str, ...]:
    """Return the sub-files a settle case edits.

    ServoDyn is edited only with a controller-library override, and HydroDyn only
    when temporary platform damping is added to its ``AddBLin``.
    """
    keys: tuple[str, ...] = SETTLE_EDITED_SUBFILE_KEYS
    if controller_library is None:
        keys = tuple(key for key in keys if key != "ServoFile")
    if platform_damping:
        keys = (*keys, "HydroFile")
    return keys


def _settled_operating_points(
    driver: LinearizationDriver,
    wind_speeds: Sequence[float],
    *,
    settle_time: float,
    discard_time: float,
    max_settle_time: float | None,
    controller_library: Path | None = None,
    executable: str | Path,
    timeout: float | None,
    max_workers: int,
    require_settled: bool,
    seeds: Sequence[OperatingPoint] | None = None,
    platforms: Sequence[PlatformPosition | None] | None = None,
    curve_tolerances: TrimTolerances | None = None,
    judgement: _SettleJudgement | None = None,
) -> _DerivedPoints:
    """Settle each wind speed in its own copied case, parallelized by wind.

    ``seeds`` (the operating curve at each wind, in order) supply the initial rotor
    speed and pitch, and each settled point is then compared with its seed;
    ``platforms`` supply the initial platform positions. Without a seed a settle
    starts from 0 rpm and 0 deg pitch.

    One point's failure never stops the others: every wind is settled, each outcome
    is classified (settled, not settled, failed, unexpected error), and the points
    to linearize are returned in wind order with the failures and errors.
    """
    speeds = list(wind_speeds)
    seed_of = dict(zip(speeds, seeds, strict=True)) if seeds is not None else {}
    platform_of = (
        dict(zip(speeds, platforms, strict=True)) if platforms is not None else {}
    )
    # Reject any winds that map to the same workspace/operating-point label, not just
    # exact float duplicates. Two distinct winds that ``:g``-format to the same string
    # (e.g. 8.0000001 and 8.0000002 -> "ws8") would otherwise share one settle workspace
    # and clobber each other in parallel, and collide again as duplicate linearization
    # case names downstream. Failing fast here, naming the offenders, is unambiguous.
    label_to_winds: dict[str, list[float]] = {}
    for speed in speeds:
        label_to_winds.setdefault(f"ws{speed:g}", []).append(speed)
    colliding = {
        label: winds for label, winds in label_to_winds.items() if len(winds) > 1
    }
    if colliding:
        details = "; ".join(
            f"{sorted(winds)} -> {label!r}"
            for label, winds in sorted(colliding.items())
        )
        msg = (
            "wind_speeds map to colliding settle case names (distinct winds that "
            f"format to the same label would share a workspace): {details}"
        )
        raise ValueError(msg)

    # Validate every settle case directory up front, before any worker copies a tree, so
    # a name colliding with the template tree fails the whole sweep before a single
    # rmtree runs (preserving fail-before-launch under concurrency).
    settle_names = {speed: f"settle_ws{speed:g}" for speed in speeds}
    for name in settle_names.values():
        driver.validated_workspace_dir(name)
    # ServoDyn is edited (so must lie inside each case) only with a library override.
    judged = judgement if judgement is not None else _SettleJudgement()
    settle_keys = _settle_edited_subfile_keys(
        controller_library, platform_damping=judged.platform_damping is not None
    )
    # A sub-file a settle edits that lies outside the template root lies outside
    # every case copy too: a configuration error common to all winds, so it fails
    # here, before any case is copied, rather than once per settle worker.
    assert driver.template_root is not None
    require_inputs_within(driver.fst_path, driver.template_root, keys=settle_keys)

    def run_one(wind_speed: float) -> OperatingPoint:
        assert driver.template_root is not None
        # Route through the driver's overlap guard so a settle case whose name collides
        # with a template directory can never rmtree or be created inside the template.
        case_dir = driver.validated_workspace_dir(settle_names[wind_speed])
        if case_dir.exists():
            rmtree_with_retry(case_dir)
        driver._copy_template(driver.template_root, case_dir)
        deck = case_dir / driver.fst_path.relative_to(driver.template_root)
        # The settle run edits the case's sub-files; none may be shared.
        require_inputs_within(deck, case_dir, keys=settle_keys)
        seed = seed_of.get(wind_speed)
        settled = solve_steady_state(
            deck,
            wind_speed,
            rotor_speed_rpm=seed.rotor_speed_rpm if seed is not None else 0.0,
            blade_pitch_deg=seed.blade_pitch_deg if seed is not None else 0.0,
            platform_position=platform_of.get(wind_speed),
            settle_time=settle_time,
            discard_time=discard_time,
            executable=executable,
            timeout=timeout,
            name=f"ws{wind_speed:g}",
            adaptive=max_settle_time is not None,
            max_settle_time=max_settle_time,
            # The sweep decides what an unsettled point means (see _classify); the
            # settle itself only measures and reports.
            require_settled=False,
            # Settle in the same inflow the linearization will use.
            wind_shear_exponent=driver.settings.wind_shear_exponent,
            controller_library=controller_library,
            criterion=judged.criterion,
            platform_damping=judged.platform_damping,
            confirm_time=judged.confirm_time,
            equilibrium_tolerances=judged.tolerances,
            prescribed_rotor=judged.prescribed_rotor,
            platform_period=judged.platform_period,
        )
        if seed is None:
            return settled
        return _with_curve_check(settled, seed, curve_tolerances)

    def attempt(wind_speed: float) -> _SettleOutcome:
        try:
            point = run_one(wind_speed)
        except SteadyStateError as error:
            logger.error("settle at %g m/s failed: %s", wind_speed, error)  # noqa: TRY400
            first_line = str(error).strip().splitlines()[0] if str(error) else ""
            return _SettleOutcome(
                failure=SettleFailure(
                    wind_speed=wind_speed,
                    reason="failed",
                    message=first_line or type(error).__name__,
                    convergence=error.convergence,
                )
            )
        except Exception as error:  # recorded per point, re-raised after the sweep
            logger.exception("settle at %g m/s raised unexpectedly", wind_speed)
            return _SettleOutcome(
                failure=SettleFailure(
                    wind_speed=wind_speed,
                    reason="error",
                    message=f"{wind_speed:g} m/s: {type(error).__name__}: {error}",
                ),
                error=error,
            )
        # A prescribed-rotor settle only seeds the platform for the CalcSteady trim,
        # which finds the equilibrium itself: an unsettled seed is still linearized.
        # A parked point (or one without CalcSteady) gets a single untrimmed
        # snapshot instead, so it must have settled like any other.
        trimmed = point.is_rotating and driver.settings.calc_steady
        return _classify(
            point, require_settled and not (judged.prescribed_rotor and trimmed)
        )

    if max_workers == 1:
        outcomes = [attempt(speed) for speed in speeds]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            outcomes = list(pool.map(attempt, speeds))
    return _DerivedPoints(
        points=[outcome.point for outcome in outcomes if outcome.point is not None],
        failures=[
            outcome.failure for outcome in outcomes if outcome.failure is not None
        ],
        errors=[outcome.error for outcome in outcomes if outcome.error is not None],
    )


@dataclass(frozen=True)
class _SettleOutcome:
    """One settle point's result: the point to linearize and/or its failure."""

    point: OperatingPoint | None = None
    failure: SettleFailure | None = None
    error: Exception | None = None


@dataclass(frozen=True)
class _DerivedPoints:
    """The points to linearize, the settle failures, and unexpected errors."""

    points: list[OperatingPoint]
    failures: list[SettleFailure] = field(default_factory=list)
    errors: list[Exception] = field(default_factory=list)


def _classify(point: OperatingPoint, require_settled: bool) -> _SettleOutcome:
    """Accept a settled point; record an unsettled one (linearized if allowed)."""
    convergence = point.settle_convergence
    # A point without a convergence record carries no evidence against it (the
    # settle solver always attaches one); it is taken as the settle returned it.
    message = None if convergence is None else _unsettled_message(convergence)
    if message is None:
        return _SettleOutcome(point=point)
    failure = SettleFailure(
        wind_speed=point.wind_speed,
        reason="not_settled",
        message=message,
        linearized=not require_settled,
        operating_point=point,
        convergence=convergence,
    )
    return _SettleOutcome(point=None if require_settled else point, failure=failure)


def _with_curve_check(
    settled: OperatingPoint,
    seed: OperatingPoint,
    tolerances: TrimTolerances | None,
) -> OperatingPoint:
    """Attach the settled-versus-curve comparison, warning when it is flagged."""
    check = compare_settled_to_curve(settled, seed, tolerances)
    if check.status == "flag":
        logger.warning(
            "settled point at %g m/s disagrees with the operating curve (%s): rotor "
            "speed %.4g rpm vs curve %.4g rpm (%+.2f %%, tol %.2g %%), pitch %.3g deg "
            "vs curve %.3g deg (%+.2f deg, tol %.2g deg); the settled point is "
            "linearized, so check the controller against the curve",
            settled.wind_speed,
            ", ".join(check.flagged_quantities),
            check.rotor_speed.trimmed,
            check.rotor_speed.curve,
            100.0 * check.rotor_speed.relative_deviation,
            100.0 * check.tolerances.rotor_speed_rel,
            check.pitch.trimmed,
            check.pitch.curve,
            check.pitch.deviation,
            check.tolerances.pitch_deg,
        )
    return replace(settled, settle_curve_check=check)
