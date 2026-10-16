# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""End-to-end linearization driver: a turbine model and a sweep to ``.lin`` files.

The :class:`LinearizationDriver` ties the pieces together. Given a template OpenFAST
case and a list of operating points, it copies the case to a fresh directory per
point, configures each copy for linearization at that point
(:func:`vane.run.configure_linearization`), runs OpenFAST
(:func:`vane.run.run_openfast`), and collects the resulting ``.lin`` files — which
feed straight into the parsing and modal pipeline. This closes the loop from a
turbine model to a Campbell diagram / load (QoI) sweep / mode shapes, all from VANE.

A case is copied per operating point so that shared sub-files (a common InflowWind or
ElastoDyn deck) are never mutated across points. By default the case's own directory is
the copy root; pass ``template_root`` to a higher directory when the deck references
files outside it (e.g. ``../5MW_Baseline/...``). Only the files the deck references are
copied (:func:`vane.run.deck_files.copy_template_files`), at their paths relative to
``template_root``, so a broad root such as a whole regression-test suite costs only the
deck's own inputs per case.
"""

from __future__ import annotations

import logging
import math
import os
import re
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
from numbers import Integral
from typing import TYPE_CHECKING

from vane._json import write_json_atomic
from vane._version import vane_version
from vane.io.fst_reader import fortran_float
from vane.io.fst_writer import FastInputError, FastInputFile
from vane.io.lin_reader import find_nonfinite_lin_value
from vane.io.mbc_transform import ROTOR_SPEED_DRIFT_RTOL
from vane.run.acceptance import (
    LinearizationAcceptancePolicy,
    OperatingPointResidualAudit,
    audit_operating_point_residual,
)
from vane.run.deck_files import TemplateCopyMode, copy_template_files
from vane.run.lin_config import (
    MAX_TRIM_TOL,
    MIN_CALC_STEADY_TMAX,
    PERIODIC_TRIM_TOL,
    LinearizationSettings,
    configure_linearization,
    needs_rated_wind_speed,
)
from vane.run.preflight import DeckValidationError, validate_deck
from vane.run.radiation_fit import RadiationFitError, ensure_radiation_state_space
from vane.run.runner import (
    OpenFastDiagnosis,
    OpenFastFailureMode,
    OpenFastResult,
    OpenFastTimeoutError,
    _validated_timeout,
    classify_openfast_output,
    run_openfast,
)
from vane.run.trim_tuning import TrimTuningPolicy, tune_trim_settings
from vane.utils.fs_retry import rmtree_with_retry
from vane.utils.paths import resolve_path

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from vane.io.lin_reader import NonFiniteLinValue
    from vane.run.auto_sweep import OperatingSweepResult
    from vane.run.operating_points import OperatingPoint
    from vane.run.radiation_fit import RadiationFitRecord
    from vane.run.steady_state import SettleAttempt, SettleConfirmation

__all__ = [
    "CaseProvenance",
    "LinearizationAcceptancePolicy",
    "LinearizationAttempt",
    "LinearizationDriver",
    "LinearizationRetryPolicy",
    "RunProvenance",
    "SweepCase",
    "SweepCaseError",
    "default_max_workers",
]

logger = logging.getLogger(__name__)

_DEFAULT_RESERVED_CPUS = 1

# Failure mode recorded for an attempt whose run completed but whose operating point the
# physical acceptance check rejected (not an OpenFAST failure, so not an
# OpenFastFailureMode; that rejection is never retried).
_RESIDUAL_REJECTED_MODE = "physical_residual_rejected"

# Synthetic return code recorded for a case whose OpenFAST run timed out (the process is
# killed, so it has no real exit status). 124 is the conventional "timed out" code (cf.
# coreutils ``timeout``); it marks the case failed so the retry ladder escalates it and,
# on exhaustion, it is recorded as non-converged rather than aborting the whole sweep.
_TIMEOUT_RETURNCODE = 124

# Synthetic return code recorded for a case that raised an unexpected exception while
# it was generated, run or collected (OpenFastFailureMode.CASE_ERROR): there is no
# OpenFAST exit status to report, and a non-zero code marks the case failed.
_CASE_ERROR_RETURNCODE = -1

# Device names Windows reserves in every directory, with or without an extension
# (``CON``, ``nul.txt``, ...). A case directory with such a name cannot be created on
# Windows, so it is rejected on every platform to keep campaigns portable.
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
)


# The LinearizationSettings fields the retry ladder escalates per operating point (see
# LinearizationRetryPolicy.settings_for_attempt). They are intentionally NOT part of the
# campaign-wide configuration: a single hard point that needed a looser trim tolerance,
# higher trim damping, or a higher trim gain is a per-case escalation, recorded per case
# in the provenance, not a different campaign; provenance compares the campaign-wide
# settings while ignoring these. ``trim_gain_torque_override`` is included because gain
# escalation pins it per attempt; ``rated_generator_torque`` is NOT (it is campaign-wide
# configuration that the retry ladder now preserves, so it is compared like any other).
# ``sim_time`` is included because a trim that ran out of time is retried with a longer
# TMax.
_RETRY_ESCALATED_FIELDS = (
    "trim_tol",
    "twr_kdmp",
    "bld_kdmp",
    "trim_gain_torque_override",
    "trim_gain_pitch",
    "sim_time",
)


# Failure modes whose trim became unstable (rather than merely slow): the retry backs
# the gains off instead of escalating them.
_UNSTABLE_TRIM_MODES = frozenset(
    {
        OpenFastFailureMode.MOTORING,
        OpenFastFailureMode.ROTOR_OVERSPEED,
        OpenFastFailureMode.DIVERGENCE,
    }
)

# Every LinearizationSettings field name, for checking what a retry fallback changed.
_SETTINGS_FIELD_NAMES = tuple(item.name for item in fields(LinearizationSettings))


def above_rated_torque_trim(settings: LinearizationSettings) -> LinearizationSettings:
    """Return ``settings`` with at/above-rated points trimmed by generator torque.

    The default :attr:`LinearizationRetryPolicy.final_fallback`. A pitch trim that the
    gain and damping remedies cannot stabilise (large rotors near rated, where the
    trim's PI loop oscillates until the rotor motors) is replaced by a torque trim at
    the operating point's pitch, which converges like a below-rated point. A
    below-rated point already uses torque trim, so the switch leaves it unchanged.

    Parameters
    ----------
    settings : LinearizationSettings
        The settings the retry ladder chose for the final attempt.

    Returns
    -------
    LinearizationSettings
        A copy with ``above_rated_trim="torque"``.
    """
    return replace(settings, above_rated_trim="torque")


@dataclass
class LinearizationRetryPolicy:
    """Adaptive retry ladder for OpenFAST linearization failures.

    A case is retried when OpenFAST exits non-zero, when a rotating CalcSteady run
    returns fewer full-system ``.lin`` files than ``NLinTimes`` or logs that it forced
    the linearization, or when the run times out. Each failed attempt is classified
    (:func:`~vane.run.classify_openfast_output`) and the next attempt's settings are
    chosen from that failure, starting from the failed attempt's own settings so that
    successive remedies compose (:meth:`next_attempt_settings`):

    * **trim not converged** (the forced single ``.lin``): the trim ran out of time.
      ``TMax`` is extended to ``min(TMax * tmax_factor, max_sim_time)`` and
      ``TrimTol`` is loosened by ``trim_tol_factor``, but not beyond
      ``PERIODIC_TRIM_TOL`` (1e-2) unless it already was: a looser trim stops on a
      rotor speed that is not periodic enough for MBC3. The pitch trim gain is
      lowered by ``pitch_trim_gain_factor``: near rated, the IEA 15- and 22-MW pitch
      trims oscillate without settling at the default gain and converge at about
      1e-4 rad/(rad/s). The torque gain and the trim damping are left
      alone: raising the gain makes a slow trim diverge, and the damping biases the
      trimmed point and, on every reference deck it was tried on (IEA 3.4-MW, IEA 15-MW
      semi), made the rotor motor within 0.3 s.
    * **non-periodic sweep**: every ``.lin`` was written but the rotor speed varies
      across the azimuth files by more than MBC3 accepts (a loosened ``TrimTol`` let a
      drifting response through). ``TrimTol`` is tightened back to the base value and
      ``TMax`` is extended as for a trim that ran out of time.
    * **motoring** (``u%HSS_Spd is negative``): the trim overshot. The pitch gain is
      multiplied by ``pitch_trim_gain_factor`` and the torque gain divided by
      ``trim_gain_factor``. When the failure happens within ``early_failure_time`` of
      the start, the trim damping itself destabilised the (light) blades, so it is
      removed (``twr_kdmp = bld_kdmp = 0``); otherwise it is kept.
    * **rotor overspeed** and **divergence**: the gains are backed off as for motoring
      and the trim damping is multiplied by ``damping_backoff_factor`` (removed for an
      early failure).
    * **initialisation error**: an input problem. The case is not retried.
    * **timeout**, an incomplete explicit-time sweep, or an **unrecognised** non-zero
      exit: the generic escalation, which loosens ``TrimTol`` by ``trim_tol_factor``,
      raises the trim damping to ``max(previous, min_retry_damping) * damping_factor``,
      multiplies the torque trim gain by ``trim_gain_factor`` and the pitch trim gain
      by ``pitch_trim_gain_factor``.

    A too-low torque gain is a common cause of a below-rated trim failure (per
    Jonkman), which is why the generic step raises it. The pitch trim is a PI law whose
    proportional gain is ``1000 * TrimGain``, so an above-rated failure is usually
    overshoot into a growing oscillation, and every remedy except the time extension
    lowers that gain. Exhaustion returns the final failed result so callers still get
    the captured diagnostics.

    Parameters
    ----------
    max_attempts : int
        Total OpenFAST runs per operating point, including the first (default ``4``).
        Most failed attempts are cheap: a divergence fails within seconds, and only a
        trim that runs out of time uses its whole ``TMax``.
    trim_tol_factor : float
        Factor that loosens ``TrimTol`` on a trim-not-converged or generic retry.
    damping_factor : float
        Factor that raises the trim damping on a generic retry.
    min_retry_damping : float
        Floor the trim damping is raised to before ``damping_factor`` applies on a
        generic retry.
    trim_gain_factor : float
        Factor that raises the torque trim gain on a generic retry, and divides it on a
        motoring, overspeed or divergence retry.
    pitch_trim_gain_factor : float
        Factor (below one by default) that multiplies the pitch trim gain on every
        retry except a trim-not-converged one.
    tmax_factor : float
        Factor (at least one) that extends ``TMax`` on a trim-not-converged retry. The
        extended value is written as ``sim_time``; ``1`` disables the extension.
    max_sim_time : float
        Cap (s) on the extended ``TMax``. A deck already longer than the cap is never
        shortened.
    early_failure_time : float
        A motoring, overspeed or divergence failure at or before this simulation time
        (s) is attributed to the trim damping, which the next attempt removes.
    damping_backoff_factor : float
        Factor in ``[0, 1]`` that lowers the trim damping after a later overspeed or
        divergence.
    adaptive : bool
        ``True`` (the default) chooses each retry from the classified failure. ``False``
        uses the fixed schedule of :meth:`settings_for_attempt`, which applies the
        generic escalation on every retry whatever the failure. An initialisation error
        is never retried in either mode.
    final_fallback : callable or None
        A last resort for a trim that the gain and damping remedies cannot stabilise.
        The default, :func:`above_rated_torque_trim`, switches an at/above-rated point
        from pitch to torque trim at the curve's pitch; ``None`` disables it. When
        set, the **last** attempt (``max_attempts``) that follows a failure in
        ``fallback_modes`` runs with ``final_fallback(settings)``, where ``settings``
        is what the ladder would otherwise have chosen (so removed damping or an
        extended ``TMax`` carry over). It must return a :class:`LinearizationSettings`
        that differs from its argument only in retry-escalated fields and
        ``fallback_fields``. The attempt is flagged ``fallback=True`` in
        :attr:`SweepCase.attempt_history`.
    fallback_modes : frozenset[OpenFastFailureMode]
        The failures that trigger ``final_fallback``; by default the unstable-trim
        modes (motoring, rotor overspeed and divergence).
    fallback_fields : tuple[str, ...]
        The :class:`LinearizationSettings` fields ``final_fallback`` may change beyond
        the retry-escalated ones. They are treated as per-case values, excluded from
        the campaign-wide provenance comparison and recorded per case in
        :attr:`CaseProvenance.fallback_overrides`.
    """

    max_attempts: int = 4
    trim_tol_factor: float = 3.0
    damping_factor: float = 2.0
    min_retry_damping: float = 5000.0
    trim_gain_factor: float = 4.0
    pitch_trim_gain_factor: float = 0.2
    tmax_factor: float = 2.0
    max_sim_time: float = 3600.0
    early_failure_time: float = 1.0
    damping_backoff_factor: float = 0.5
    adaptive: bool = True
    final_fallback: Callable[[LinearizationSettings], LinearizationSettings] | None = (
        above_rated_torque_trim
    )
    fallback_modes: frozenset[OpenFastFailureMode] = _UNSTABLE_TRIM_MODES
    fallback_fields: tuple[str, ...] = ("above_rated_trim",)

    def __post_init__(self) -> None:
        """Reject a non-positive attempt count or an out-of-range factor."""
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, Integral)
            or self.max_attempts < 1
        ):
            msg = f"max_attempts must be >= 1, got {self.max_attempts}"
            raise ValueError(msg)
        self.max_attempts = int(self.max_attempts)
        for name in (
            "trim_tol_factor",
            "damping_factor",
            "min_retry_damping",
            "trim_gain_factor",
            "pitch_trim_gain_factor",
            "max_sim_time",
        ):
            checked = _checked_real(
                name,
                getattr(self, name),
                bound="positive",
                minimum=0.0,
                exclusive_minimum=True,
            )
            setattr(self, name, checked)
        self.tmax_factor = _checked_real(
            "tmax_factor", self.tmax_factor, bound=">= 1", minimum=1.0
        )
        self.early_failure_time = _checked_real(
            "early_failure_time",
            self.early_failure_time,
            bound="non-negative",
            minimum=0.0,
        )
        self.damping_backoff_factor = _checked_real(
            "damping_backoff_factor",
            self.damping_backoff_factor,
            bound="in [0, 1]",
            minimum=0.0,
            maximum=1.0,
        )
        if not isinstance(self.adaptive, bool):
            msg = f"adaptive must be a bool, got {self.adaptive!r}"
            raise TypeError(msg)
        if self.final_fallback is not None and not callable(self.final_fallback):
            msg = "final_fallback must be callable or None"
            raise TypeError(msg)
        self.fallback_modes = frozenset(
            OpenFastFailureMode(mode) for mode in self.fallback_modes
        )
        self.fallback_fields = tuple(self.fallback_fields)
        unknown = sorted(set(self.fallback_fields) - set(_SETTINGS_FIELD_NAMES))
        if unknown:
            msg = f"fallback_fields are not LinearizationSettings fields: {unknown}"
            raise ValueError(msg)

    def settings_for_attempt(
        self, base: LinearizationSettings, attempt: int
    ) -> LinearizationSettings:
        """Return the fixed (non-adaptive) settings for a 1-based attempt number.

        Attempt ``1`` is ``base`` unchanged. Attempt ``k > 1`` is ``k - 1`` generic
        escalation steps from ``base``:

        * ``trim_tol = min(base * trim_tol_factor**(k-1), MAX_TRIM_TOL)``; the clamp
          is logged and the as-run value is recorded per case in the provenance;
        * ``twr_kdmp``/``bld_kdmp = max(base, min_retry_damping) *
          damping_factor**(k-1)``, so every retry raises the damping above both the
          base and the floor;
        * torque trim gain ``= base * trim_gain_factor**(k-1)``;
        * pitch trim gain ``= base * pitch_trim_gain_factor**(k-1)``.

        The driver follows this schedule when ``adaptive`` is ``False``.

        Parameters
        ----------
        base : LinearizationSettings
            The operating point's starting settings.
        attempt : int
            The 1-based attempt number, at most ``max_attempts``.

        Returns
        -------
        LinearizationSettings
            The settings for that attempt.

        Raises
        ------
        ValueError
            If ``attempt`` is not an integer in ``[1, max_attempts]`` or the
            escalation leaves the finite positive range.
        """
        attempt = self._checked_attempt(attempt)
        if attempt == 1:
            return replace(base)
        try:
            scale = self.trim_tol_factor ** (attempt - 1)
            damping_scale = self.damping_factor ** (attempt - 1)
            gain_scale = self.trim_gain_factor ** (attempt - 1)
            pitch_gain_scale = self.pitch_trim_gain_factor ** (attempt - 1)
        except OverflowError as exc:
            msg = "retry escalation overflowed the finite numerical range"
            raise ValueError(msg) from exc
        _require_finite_positive(scale, damping_scale, gain_scale, pitch_gain_scale)
        # Pin the escalated torque gain via the explicit override so
        # resolved_trim_gain_torque() returns it rather than re-deriving the (too-low)
        # base gain. The override does NOT discard rated_generator_torque, which still
        # supplies the constant ServoDyn torque (VS_RtTq) the above-rated pitch trim
        # needs every attempt — clearing it would flatten retries to the wrong torque.
        return replace(
            base,
            trim_tol=_capped_trim_tol(base.trim_tol * scale, attempt),
            twr_kdmp=max(base.twr_kdmp, self.min_retry_damping) * damping_scale,
            bld_kdmp=max(base.bld_kdmp, self.min_retry_damping) * damping_scale,
            trim_gain_torque_override=base.resolved_trim_gain_torque() * gain_scale,
            trim_gain_pitch=base.trim_gain_pitch * pitch_gain_scale,
        )

    def next_attempt_settings(
        self,
        base: LinearizationSettings,
        previous: LinearizationSettings,
        diagnosis: OpenFastDiagnosis,
        *,
        attempt: int,
        previous_tmax: float | None = None,
    ) -> LinearizationSettings | None:
        """Return the settings for the retry that follows a classified failure.

        Parameters
        ----------
        base : LinearizationSettings
            The operating point's starting settings (used by the fixed schedule).
        previous : LinearizationSettings
            The settings of the attempt that failed; adaptive remedies start here.
        diagnosis : OpenFastDiagnosis
            The classified failure of that attempt.
        attempt : int
            The 1-based number of the attempt being configured, in
            ``[2, max_attempts]``.
        previous_tmax : float or None, optional
            The ``TMax`` the failed attempt actually ran with (read from its deck).
            The extension starts from it when ``previous.sim_time`` is unset, and from
            ``MIN_CALC_STEADY_TMAX`` when neither is known.

        Returns
        -------
        LinearizationSettings or None
            The next attempt's settings, or ``None`` when the failure is not
            retriable (an initialisation error or a non-finite linearization).

        Raises
        ------
        ValueError
            If ``attempt`` is not an integer in ``[2, max_attempts]``.
        """
        attempt = self._checked_attempt(attempt)
        if attempt == 1:
            msg = "attempt must be at least 2 for a retry"
            raise ValueError(msg)
        if not diagnosis.retriable:
            return None
        if self.adaptive:
            chosen = self._adaptive_step(
                base, previous, diagnosis, attempt, previous_tmax
            )
        else:
            chosen = self.settings_for_attempt(base, attempt)
        if not self.uses_fallback(diagnosis, attempt=attempt):
            return chosen
        assert self.final_fallback is not None  # checked by uses_fallback
        fallback = self.final_fallback(chosen)
        if not isinstance(fallback, LinearizationSettings):
            msg = (
                "final_fallback must return LinearizationSettings, got "
                f"{type(fallback).__name__}"
            )
            raise TypeError(msg)
        allowed = {*_RETRY_ESCALATED_FIELDS, *self.fallback_fields}
        changed = sorted(
            name
            for name in _SETTINGS_FIELD_NAMES
            if name not in allowed and getattr(fallback, name) != getattr(chosen, name)
        )
        if changed:
            msg = (
                f"final_fallback changed {changed}, which are not declared in "
                "fallback_fields; declare every field the fallback may change so "
                "provenance records it per case"
            )
            raise ValueError(msg)
        return fallback

    def uses_fallback(self, diagnosis: OpenFastDiagnosis, *, attempt: int) -> bool:
        """Return whether ``attempt`` runs with the ``final_fallback`` settings.

        Parameters
        ----------
        diagnosis : OpenFastDiagnosis
            The classified failure of the previous attempt.
        attempt : int
            The 1-based number of the attempt being configured.

        Returns
        -------
        bool
            ``True`` when a fallback is configured, ``attempt`` is the last one
            (``max_attempts``, at least 2) and the failure is retriable and one of
            ``fallback_modes``.
        """
        return (
            self.final_fallback is not None
            and attempt == self.max_attempts
            and attempt > 1
            and diagnosis.retriable
            and diagnosis.mode in self.fallback_modes
        )

    def _adaptive_step(
        self,
        base: LinearizationSettings,
        previous: LinearizationSettings,
        diagnosis: OpenFastDiagnosis,
        attempt: int,
        previous_tmax: float | None,
    ) -> LinearizationSettings:
        """Return the remedy for ``diagnosis``, starting from ``previous``."""
        mode = diagnosis.mode
        if mode is OpenFastFailureMode.NON_PERIODIC_SWEEP:
            # The tolerance let a drifting response through: tighten it back to the
            # base and give the trim longer to become periodic.
            return replace(
                previous,
                trim_tol=min(previous.trim_tol, base.trim_tol),
                sim_time=self._extended_sim_time(previous, previous_tmax),
            )
        if mode is OpenFastFailureMode.TRIM_NOT_CONVERGED:
            # A pitch trim that keeps oscillating near rated on a large rotor never
            # settles at the default gain (IEA 15/22-MW need ~1e-4 rad/(rad/s)), so the
            # pitch gain is backed off as well; below rated it is unused.
            loosened = min(
                previous.trim_tol * self.trim_tol_factor,
                max(previous.trim_tol, PERIODIC_TRIM_TOL),
            )
            return replace(
                previous,
                trim_tol=_capped_trim_tol(loosened, attempt),
                sim_time=self._extended_sim_time(previous, previous_tmax),
                trim_gain_pitch=previous.trim_gain_pitch * self.pitch_trim_gain_factor,
            )
        if mode in _UNSTABLE_TRIM_MODES:
            if (
                diagnosis.failure_time is not None
                and diagnosis.failure_time <= self.early_failure_time
            ):
                damping_scale = 0.0
            elif mode is OpenFastFailureMode.MOTORING:
                damping_scale = 1.0
            else:
                damping_scale = self.damping_backoff_factor
            return replace(
                previous,
                twr_kdmp=previous.twr_kdmp * damping_scale,
                bld_kdmp=previous.bld_kdmp * damping_scale,
                trim_gain_torque_override=(
                    previous.resolved_trim_gain_torque() / self.trim_gain_factor
                ),
                trim_gain_pitch=previous.trim_gain_pitch * self.pitch_trim_gain_factor,
            )
        return self._generic_step(previous, attempt)

    def _checked_attempt(self, attempt: int) -> int:
        """Return ``attempt`` as an int, rejecting one outside ``[1, max_attempts]``."""
        if (
            isinstance(attempt, bool)
            or not isinstance(attempt, Integral)
            or attempt < 1
            or attempt > self.max_attempts
        ):
            msg = f"attempt must be an integer in [1, {self.max_attempts}]"
            raise ValueError(msg)
        return int(attempt)

    def _generic_step(
        self, previous: LinearizationSettings, attempt: int
    ) -> LinearizationSettings:
        """Return one generic escalation step from ``previous``."""
        torque = previous.resolved_trim_gain_torque() * self.trim_gain_factor
        pitch = previous.trim_gain_pitch * self.pitch_trim_gain_factor
        twr = max(previous.twr_kdmp, self.min_retry_damping) * self.damping_factor
        bld = max(previous.bld_kdmp, self.min_retry_damping) * self.damping_factor
        _require_finite_positive(torque, pitch, twr, bld)
        return replace(
            previous,
            trim_tol=_capped_trim_tol(
                previous.trim_tol * self.trim_tol_factor, attempt
            ),
            twr_kdmp=twr,
            bld_kdmp=bld,
            trim_gain_torque_override=torque,
            trim_gain_pitch=pitch,
        )

    def _extended_sim_time(
        self, previous: LinearizationSettings, previous_tmax: float | None
    ) -> float | None:
        """Return the extended ``sim_time`` after a trim that ran out of time."""
        if previous.sim_time is not None:
            current = previous.sim_time
        elif previous_tmax is not None and math.isfinite(previous_tmax):
            current = previous_tmax
        else:
            current = MIN_CALC_STEADY_TMAX
        extended = min(current * self.tmax_factor, max(self.max_sim_time, current))
        return extended if extended > current else previous.sim_time


def _checked_real(
    name: str,
    value: object,
    *,
    bound: str,
    minimum: float,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    """Return ``value`` as a finite float within the given bounds, or raise."""
    msg = f"{name} must be {bound}, got {value}"
    if isinstance(value, bool):
        raise TypeError(msg)
    try:
        converted = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(msg) from exc
    if (
        not math.isfinite(converted)
        or converted < minimum
        or (exclusive_minimum and converted == minimum)
        or (maximum is not None and converted > maximum)
    ):
        raise ValueError(msg)
    return converted


def _require_finite_positive(*values: float) -> None:
    """Raise when an escalation left the finite positive range."""
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        msg = "retry escalation left the finite positive numerical range"
        raise ValueError(msg)


def _capped_trim_tol(trim_tol: float, attempt: int) -> float:
    """Clamp an escalated trim tolerance at ``MAX_TRIM_TOL``, logging the clamp."""
    if trim_tol > MAX_TRIM_TOL:
        logger.warning(
            "Retry attempt %d would loosen TrimTol to %.3g; capped at %.3g",
            attempt,
            trim_tol,
            MAX_TRIM_TOL,
        )
        return MAX_TRIM_TOL
    return trim_tol


def default_max_workers(*, reserve_cpus: int = _DEFAULT_RESERVED_CPUS) -> int:
    """Return a CPU-aware default worker count for independent OpenFAST cases."""
    if (
        isinstance(reserve_cpus, bool)
        or not isinstance(reserve_cpus, Integral)
        or reserve_cpus < 0
    ):
        msg = f"reserve_cpus must be non-negative, got {reserve_cpus}"
        raise ValueError(msg)
    reserve_cpus = int(reserve_cpus)
    count = os.cpu_count() or 1
    return max(1, count - reserve_cpus)


@dataclass(frozen=True)
class LinearizationAttempt:
    """The record of one OpenFAST run of an operating point's retry ladder.

    Parameters
    ----------
    attempt : int
        The 1-based attempt number.
    failure_mode : str or None
        The classified failure (an :class:`~vane.run.OpenFastFailureMode` value, or
        ``"physical_residual_rejected"`` when the acceptance check rejected a completed
        run). ``None`` when the attempt converged and was accepted.
    reason : str or None
        A one-line description of the failure; ``None`` on success.
    openfast_message : str
        The relevant excerpt of OpenFAST's output (its fatal error message, or its last
        lines); empty when none applies.
    failure_time : float or None
        The simulation time (s) of OpenFAST's fatal error, when it reported one.
    returncode : int
        The OpenFAST exit status (``124`` for a run killed by the timeout).
    n_lin : int
        The number of full-system ``.lin`` files the attempt wrote.
    log_file : str or None
        The file in the case directory holding the attempt's full OpenFAST standard
        output and error, ``<deck stem>.attempt<k>.log``; ``None`` if it could not be
        written.
    trim_tol, twr_kdmp, bld_kdmp, trim_gain_torque, trim_gain_pitch : float
        The attempt's trim tolerance, trim damping and effective trim gains.
    sim_time : float or None
        The attempt's ``sim_time`` setting (``None`` keeps the deck's ``TMax``).
    tmax : float or None
        The ``TMax`` the attempt's configured deck actually carried, when readable.
    fallback : bool
        Whether the attempt ran with the retry policy's ``final_fallback`` settings.
    """

    attempt: int
    failure_mode: str | None
    reason: str | None
    openfast_message: str
    failure_time: float | None
    returncode: int
    n_lin: int
    log_file: str | None
    trim_tol: float
    twr_kdmp: float
    bld_kdmp: float
    trim_gain_torque: float
    trim_gain_pitch: float
    sim_time: float | None
    tmax: float | None
    fallback: bool = False


@dataclass
class SweepCase:
    """The generated case and run outcome for one operating point.

    Parameters
    ----------
    operating_point : OperatingPoint
        The operating point this case was generated for.
    case_dir : pathlib.Path
        The per-operating-point directory the case was copied into.
    fst_path : pathlib.Path
        The configured primary ``.fst`` deck inside ``case_dir``.
    result : OpenFastResult
        The OpenFAST run outcome (return code, ``.lin`` files, output).
    settings : LinearizationSettings
        A snapshot of the *as-run* settings this case was configured with (the final
        retry attempt's, so a retried case carries its escalated trim/damping), frozen
        at run time so provenance reflects what actually produced the case even if the
        driver is later mutated.
    base_settings : LinearizationSettings
        A snapshot of the *campaign base* settings this case escalated from (the
        driver's settings at run time, before any per-case retry escalation). Frozen at
        run time so the campaign-wide configuration provenance records is independent of
        any later driver mutation. For an unretried case this equals ``settings``.
    executable : str
        The OpenFAST executable used for this case, captured at run time.
    attempts : int
        How many OpenFAST runs the retry ladder used to produce this result (``1`` when
        the first run converged). The case carries the final attempt's outcome.
    converged : bool
        Whether the final attempt converged: OpenFAST exited successfully, produced the
        requested number of full-system ``.lin`` files, did not force the
        linearization without finding the steady state, and every ``.lin`` file holds
        only finite numbers. ``False`` means the ladder was
        exhausted without a trustworthy linearization for this point.
    operating_point_residual : OperatingPointResidualAudit or None
        Optional independent six-DOF platform-drift audit. ``None`` when the driver's
        physical acceptance check was disabled.
    attempt_history : tuple[LinearizationAttempt, ...]
        One record per OpenFAST run, in order: the classified failure, the settings it
        ran with and the log file holding its full output. Empty for a case built by
        hand rather than by :meth:`LinearizationDriver.run_case`.
    fallback_fields : tuple[str, ...]
        The retry policy's ``fallback_fields`` when the case ran, snapshotted so
        provenance reads the case with the policy that produced it even if the
        driver's policy is replaced or mutated afterwards.
    radiation_fit : RadiationFitRecord or None
        With ``hydro_radiation_fit``, the state-space radiation file the final attempt
        generated in the case copy (:func:`vane.run.ensure_radiation_state_space`);
        ``None`` otherwise, or when the copy already had the file.
    """

    operating_point: OperatingPoint
    case_dir: Path
    fst_path: Path
    result: OpenFastResult
    settings: LinearizationSettings = field(default_factory=LinearizationSettings)
    base_settings: LinearizationSettings = field(default_factory=LinearizationSettings)
    executable: str = "openfast"
    attempts: int = 1
    converged: bool = True
    operating_point_residual: OperatingPointResidualAudit | None = None
    attempt_history: tuple[LinearizationAttempt, ...] = ()
    fallback_fields: tuple[str, ...] = ()
    radiation_fit: RadiationFitRecord | None = None

    @property
    def failure_mode(self) -> str | None:
        """Return the final attempt's classified failure, or ``None`` on success."""
        if not self.attempt_history:
            return None
        return self.attempt_history[-1].failure_mode

    @property
    def succeeded(self) -> bool:
        """Return whether OpenFAST exited successfully for this case."""
        return self.result.succeeded

    @property
    def lin_files(self) -> list[Path]:
        """Return the ``.lin`` files produced for this case."""
        return self.result.lin_files

    @property
    def n_lin(self) -> int:
        """Return the number of full-system ``.lin`` files produced."""
        return self.result.n_lin


class SweepCaseError(RuntimeError):
    """Raised by a strict sweep after a case raised an unexpected exception.

    A linearization sweep records an unexpected exception in one operating point (a
    file-system error while its case is copied or configured, say) as a failed
    :class:`SweepCase` whose ``failure_mode`` is ``"case_error"`` and carries on with
    the other points. With ``strict=True`` (:meth:`LinearizationDriver.run_sweep`) or
    ``strict_cases=True`` (:func:`vane.run.run_operating_sweep`) it then raises this
    error -- but only after every other case has finished, with all of them attached,
    so no completed work is lost. The first case's exception is chained as
    ``__cause__``.

    Parameters
    ----------
    message : str
        The error message.
    cases : list[SweepCase]
        Every case of the sweep, in input order, the failed ones included.
    result : OperatingSweepResult or None, optional
        The full sweep result, when raised by :func:`vane.run.run_operating_sweep`.

    Attributes
    ----------
    cases : list[SweepCase]
        As given.
    result : OperatingSweepResult or None
        As given.
    """

    def __init__(
        self,
        message: str,
        *,
        cases: list[SweepCase],
        result: OperatingSweepResult | None = None,
    ) -> None:
        super().__init__(message)
        self.cases = cases
        self.result = result


@dataclass(frozen=True)
class CaseProvenance:
    """Per-operating-point record within a sweep's provenance.

    Parameters
    ----------
    name : str
        The case-directory name for the operating point.
    wind_speed : float
        Hub-height wind speed (m/s).
    rotor_speed_rpm : float
        Rotor speed (rpm).
    blade_pitch_deg : float
        Collective blade pitch (deg).
    case_dir : str
        The case directory the run was written to.
    returncode : int
        OpenFAST exit status (0 on success).
    succeeded : bool
        Whether OpenFAST exited successfully.
    n_lin : int
        Number of full-system ``.lin`` files produced.
    attempts : int
        How many OpenFAST runs the retry ladder used for this point.
    converged : bool
        Whether the final attempt converged to a trustworthy linearization.
    trim_tol : float
        The as-run trim tolerance for this case (escalated above the campaign base when
        the case was retried).
    twr_kdmp, bld_kdmp : float
        The as-run tower/blade trim damping for this case (escalated on retry).
    trim_gain_torque, trim_gain_pitch : float
        The as-run torque/pitch trim gains for this case. ``trim_gain_torque`` is the
        *effective* torque gain (the escalated override on a retry, else
        ``rated_generator_torque / 10`` when derived, else the explicit value). Recorded
        per case because the gain is a per-case escalation ignored by the campaign-wide
        settings record.
    settle_* : optional
        Platform-velocity settle evidence carried by a point produced through
        :func:`vane.run.solve_steady_state`; ``None`` for hand-authored curve points.
    settle_criterion : str or None
        The criterion that accepted (or rejected) the settle: ``"velocity"`` or
        ``"mean_drift"`` (see :func:`vane.run.solve_steady_state`).
    settle_confirmation : SettleConfirmation or None
        For a two-stage settle with temporary platform damping, the damped stage's
        evidence and the undamped confirmation's outcome; ``None`` otherwise.
    settle_curve_rotor_speed_rpm, settle_curve_pitch_deg, settle_curve_status : optional
        For a settle seeded from an operating curve, the curve's rotor speed (rpm)
        and pitch (deg) at this wind speed and the settled-versus-curve status
        (``"pass"``, ``"flag"`` or ``"unavailable"``; see
        :class:`vane.run.SettleCurveCheck`); ``None`` otherwise.
    operating_point_residual : OperatingPointResidualAudit or None
        The physical acceptance audit, when the check was enabled.
    sim_time : float or None
        The as-run ``sim_time`` (``TMax``) setting, extended above the campaign base
        when a trim that ran out of time was retried; ``None`` keeps the deck's value.
    failure_mode : str or None
        The final attempt's classified failure; ``None`` when it converged.
    attempt_history : tuple[LinearizationAttempt, ...]
        The per-attempt records of the retry ladder (see :class:`LinearizationAttempt`).
    fallback_overrides : dict[str, object] or None
        For a case whose final attempt ran the retry policy's ``final_fallback``, the
        as-run values of its ``fallback_fields``; ``None`` otherwise.
    radiation_fit : dict[str, object] or None
        The state-space radiation file generated in the case copy
        (:meth:`vane.run.RadiationFitRecord.to_dict`: the ``.ss`` path, the WAMIT
        ``.1`` source, the state count and the per-entry fit errors); ``None`` when
        none was generated.
    """

    name: str
    wind_speed: float
    rotor_speed_rpm: float
    blade_pitch_deg: float
    case_dir: str
    returncode: int
    succeeded: bool
    n_lin: int
    attempts: int = 1
    converged: bool = True
    trim_tol: float = 0.0
    twr_kdmp: float = 0.0
    bld_kdmp: float = 0.0
    trim_gain_torque: float = 0.0
    trim_gain_pitch: float = 0.0
    settle_assessed: bool | None = None
    settle_converged: bool | None = None
    settle_translational_velocity_rms: float | None = None
    settle_angular_velocity_rms: float | None = None
    settle_time: float | None = None
    settle_iterations: int | None = None
    settle_attempts: tuple[SettleAttempt, ...] | None = None
    settle_criterion: str | None = None
    settle_confirmation: SettleConfirmation | None = None
    settle_curve_rotor_speed_rpm: float | None = None
    settle_curve_pitch_deg: float | None = None
    settle_curve_status: str | None = None
    operating_point_residual: OperatingPointResidualAudit | None = None
    sim_time: float | None = None
    failure_mode: str | None = None
    attempt_history: tuple[LinearizationAttempt, ...] = ()
    fallback_overrides: dict[str, object] | None = None
    radiation_fit: dict[str, object] | None = None


@dataclass(frozen=True)
class RunProvenance:
    """Reproducibility record for a linearization sweep.

    Captures the exact :class:`LinearizationSettings` applied to every case — so the
    settings a set of ``.lin`` files was produced under are recorded alongside them,
    not just remembered — together with the library version, the executable, and the
    per-case outcome.

    Parameters
    ----------
    vane_version : str
        Version of the library that drove the sweep.
    created_at : str
        ISO-8601 timestamp of when the record was made.
    executable : str
        The OpenFAST executable name or path used.
    settings : dict[str, object]
        Every :class:`LinearizationSettings` field and its value.
    cases : tuple[CaseProvenance, ...]
        Per-operating-point provenance, in run order.
    auto_configuration : dict[str, object] or None, optional
        The values :func:`vane.run.run_operating_sweep` inferred from the deck and
        curve, each with its source, and the warnings raised while inferring them
        (see :class:`vane.run.AutoConfiguration`); ``None`` when nothing was
        inferred.
    """

    vane_version: str
    created_at: str
    executable: str
    settings: dict[str, object]
    cases: tuple[CaseProvenance, ...]
    auto_configuration: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        """Return the record as a JSON-serializable dictionary."""
        record: dict[str, object] = {
            "vane_version": self.vane_version,
            "created_at": self.created_at,
            "executable": self.executable,
            "settings": self.settings,
            "cases": [_case_record(case) for case in self.cases],
        }
        if self.auto_configuration is not None:
            record["auto_configuration"] = self.auto_configuration
        return record

    def write(self, path: str | Path) -> Path:
        """Write the record to ``path`` as JSON and return the path written."""
        return write_json_atomic(path, self.to_dict(), overwrite=True)


_SETTLE_RMS_FIELDS = ("translational_velocity_rms", "angular_velocity_rms")


def _unmeasured_as_none(value: object) -> object:
    """Map an unmeasured (NaN) or diverged (infinite) settle RMS to ``None``.

    Provenance is written as strict JSON, which has no NaN or infinity.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _case_record(case: CaseProvenance) -> dict[str, object]:
    """Return a strict-JSON record of ``case``.

    A settle run whose output lacks a platform-velocity channel group reports that
    group's RMS as NaN ("not measured"); it is recorded as ``null`` rather than making
    the whole provenance record unwritable.
    """
    record = asdict(case)
    for field_name in _SETTLE_RMS_FIELDS:
        key = f"settle_{field_name}"
        record[key] = _unmeasured_as_none(record[key])
    attempts = record["settle_attempts"]
    if attempts is not None:
        for attempt in attempts:
            for field_name in _SETTLE_RMS_FIELDS:
                attempt[field_name] = _unmeasured_as_none(attempt[field_name])
    confirmation = record["settle_confirmation"]
    if confirmation is not None:
        assert case.settle_confirmation is not None
        confirmation["growing_oscillation"] = (
            case.settle_confirmation.growing_oscillation
        )
        for field_name in _SETTLE_RMS_FIELDS:
            key = f"damped_{field_name}"
            confirmation[key] = _unmeasured_as_none(confirmation[key])
    return record


@dataclass
class LinearizationDriver:
    """Generate, run, and collect linearizations across operating points.

    Parameters
    ----------
    fst_path : pathlib.Path
        The template case's primary ``.fst`` deck.
    output_dir : pathlib.Path
        Directory under which per-operating-point case directories are created.
        Must not be inside ``template_root`` (nor contain it), to avoid a recursive
        copy.
    settings : LinearizationSettings
        Linearization tuning applied to every case.
    executable : str or pathlib.Path
        OpenFAST executable name or path.
    template_root : pathlib.Path or None
        Directory the case is copied from per operating point; defaults to
        ``fst_path``'s parent. Set to a higher directory when the deck references files
        outside its own folder so those are copied too. File paths relative to it are
        preserved in each case directory.
    link_inputs : bool
        When ``True`` (the default), each case is created by **hard-linking** the
        template files rather than copying their bytes, so a sweep's heavy read-only
        inputs (e.g. WAMIT hydrodynamic data, airfoil tables) are shared at the inode
        level and a many-point sweep does not duplicate hundreds of megabytes per
        point. Each case still gets its own complete tree and its own outputs; the
        small deck files are edited via :meth:`FastInputFile.write`'s atomic replace,
        which breaks the link so the template is never modified. Falls back to a full
        byte copy automatically when hard-linking is unsupported (e.g. the output
        directory is on a different filesystem from the template). Set to ``False`` to
        always copy.
    template_copy : {"referenced", "tree"}
        What each case receives from ``template_root``. ``"referenced"`` (the default)
        copies only the files the deck references, found recursively from the primary
        deck (:func:`~vane.run.deck_files.deck_file_closure`); unrelated cases and stale
        outputs elsewhere under the root are left behind. When that walk cannot vouch
        for completeness (a relative reference outside ``template_root``, an unreadable
        input) the whole tree is copied instead, with a logged warning. ``"tree"``
        always copies the whole template tree.
    validate : bool
        When ``True`` (the default), each configured case is checked with
        :func:`~vane.run.validate_deck` before OpenFAST is launched, so a deck that
        cannot linearize as configured (a DLL controller, active waves, missing
        state-space files, ...) is rejected up front instead of failing late inside a
        long settle/trim run. Set to ``False`` to skip the pre-flight check.
    trim_tuning : TrimTuningPolicy or None
        When set, each operating point's *starting* trim settings are tuned for its
        wind-speed region (:func:`~vane.run.tune_trim_settings`) before the retry ladder
        runs: an at/above-rated point starts from the looser-tolerance, higher-damping
        floors that such points need, so it is more likely to converge on the first
        attempt instead of after a reactive retry. ``None`` (the default) starts every
        point from ``settings`` unchanged. Tuning only raises the per-case
        trim_tol/twr_kdmp/bld_kdmp toward the floors, so provenance is unaffected.
    acceptance_policy : LinearizationAcceptancePolicy or None
        Optional post-run physical check for a floating operating point. When set, VANE
        averages the ElastoDyn platform-rate states over the azimuth sweep and rejects
        a nominally successful trimmed or explicit-time result with excessive secular
        translational or angular drift. This rejection is not sent through the normal
        retry ladder because that ladder loosens ``TrimTol`` and would make a residual
        drift worse; the caller must tighten the base trim or improve the warm start.
    auto_configuration : dict[str, object] or None
        The record of values inferred from the deck and curve (set by
        :func:`vane.run.run_operating_sweep`), copied into :meth:`provenance`.

    Raises
    ------
    ValueError
        If ``fst_path`` is not within ``template_root``, or ``output_dir`` and
        ``template_root`` are nested in one another.
    """

    fst_path: Path
    output_dir: Path
    settings: LinearizationSettings = field(default_factory=LinearizationSettings)
    executable: str | Path = "openfast"
    template_root: Path | None = None
    link_inputs: bool = True
    template_copy: TemplateCopyMode = "referenced"
    validate: bool = True
    retry_policy: LinearizationRetryPolicy = field(
        default_factory=LinearizationRetryPolicy
    )
    trim_tuning: TrimTuningPolicy | None = None
    acceptance_policy: LinearizationAcceptancePolicy | None = None
    auto_configuration: dict[str, object] | None = None

    def __post_init__(self) -> None:
        """Resolve paths and validate the copy roots are non-nested."""
        self.fst_path = resolve_path(self.fst_path)
        self.output_dir = resolve_path(self.output_dir)
        root = (
            resolve_path(self.template_root)
            if self.template_root is not None
            else self.fst_path.parent
        )
        self.template_root = root
        if self.template_copy not in ("referenced", "tree"):
            msg = (
                "template_copy must be 'referenced' or 'tree', got "
                f"{self.template_copy!r}"
            )
            raise ValueError(msg)
        if not self.fst_path.is_relative_to(root):
            msg = f"fst_path {self.fst_path} is not inside template_root {root}"
            raise ValueError(msg)
        # Reject only output_dir inside template_root: that is the layout that makes
        # copytree(template_root -> output_dir/case) recurse. The reverse (template
        # under the output workspace) is fine — cases are written to sibling dirs.
        if self.output_dir.is_relative_to(root):
            msg = (
                f"output_dir {self.output_dir} must not be inside template_root "
                f"{root} (it would cause a recursive copy)"
            )
            raise ValueError(msg)

    def case_dir(self, operating_point: OperatingPoint) -> Path:
        """Return the case directory for an operating point."""
        return self.output_dir / _case_name(operating_point)

    def _op_base_settings(
        self, operating_point: OperatingPoint
    ) -> LinearizationSettings:
        """Return the per-operating-point starting settings the retry ladder escalates.

        With ``trim_tuning`` set, the campaign-base ``settings`` are tuned for this
        point's wind-speed region (an at/above-rated point gets the looser-tolerance,
        higher-damping trim floors proactively); without it, the campaign base is used
        unchanged. The campaign base recorded in provenance is always ``settings``; this
        only sets where the per-case escalation starts.
        """
        if self.trim_tuning is None:
            return self.settings
        return tune_trim_settings(operating_point, self.settings, self.trim_tuning)

    def validated_workspace_dir(self, name: str) -> Path:
        """Return ``output_dir/name``, rejecting one that overlaps the template tree.

        A workspace directory (a linearization case or an auxiliary settle case) that
        is, contains, or sits inside the template root would have the template deleted
        by the clean-copy step, which removes the destination first. This is possible
        when the template lives under the output workspace and a generated name collides
        with a template directory. Reject any such overlap up front with a clear error,
        so directory creation can never destroy the source tree.

        Parameters
        ----------
        name : str
            The directory name to create under ``output_dir``.

        Returns
        -------
        pathlib.Path
            The resolved, validated destination directory.

        Raises
        ------
        ValueError
            If the destination is, contains, or is inside the template root, or
            ``name`` is a device name Windows reserves (``CON``, ``NUL``, ``COM1``, ...,
            with or without an extension).
        """
        assert self.template_root is not None  # set in __post_init__
        if name.split(".", 1)[0].rstrip(" ").upper() in _WINDOWS_RESERVED_NAMES:
            msg = (
                f"workspace directory name {name!r} is a reserved Windows device name "
                "and cannot be created portably; rename the operating point"
            )
            raise ValueError(msg)
        destination = resolve_path(self.output_dir / name)
        root = self.template_root
        if root.is_relative_to(destination) or destination.is_relative_to(root):
            msg = (
                f"workspace directory {destination} would overlap the template root "
                f"{root} (it would delete or be created inside the template); choose a "
                "different output_dir or name"
            )
            raise ValueError(msg)
        return destination

    def _validated_case_dir(self, operating_point: OperatingPoint) -> Path:
        """Return the case directory, rejecting one that overlaps the template root."""
        return self.validated_workspace_dir(_case_name(operating_point))

    def generate_case(self, operating_point: OperatingPoint) -> Path:
        """Copy the template case and configure it for an operating point.

        Parameters
        ----------
        operating_point : OperatingPoint
            The operating point to set up.

        Returns
        -------
        pathlib.Path
            The configured primary ``.fst`` deck inside the new case directory.
        """
        deck, _ = self._generate_case_with_settings(operating_point, self.settings)
        return deck

    def _generate_case_with_settings(
        self, operating_point: OperatingPoint, settings: LinearizationSettings
    ) -> tuple[Path, RadiationFitRecord | None]:
        """Copy and configure a case with a per-attempt settings snapshot.

        Returns the configured deck and, with ``hydro_radiation_fit``, the record of
        the state-space radiation file generated in the case copy (``None`` when the
        copy already had one or none was requested).
        """
        assert self.template_root is not None  # set in __post_init__
        destination = self._validated_case_dir(operating_point)
        # Start from a clean copy of the template so a rerun cannot leave stale files
        # (e.g. outputs or a previously-configured deck) from an earlier sweep.
        if destination.exists():
            rmtree_with_retry(destination)
        self._copy_template(self.template_root, destination)
        deck = destination / self.fst_path.relative_to(self.template_root)
        configure_linearization(deck, operating_point, settings, confine_to=destination)
        radiation_fit: RadiationFitRecord | None = None
        if settings.hydro_radiation_fit:
            # Fitted into this case copy only, never next to the template's WAMIT data.
            radiation_fit = ensure_radiation_state_space(deck, confine_to=destination)
            if radiation_fit is not None and not radiation_fit.written:
                error = radiation_fit.max_relative_error
                msg = (
                    f"the state-space radiation fit to {radiation_fit.source} is not "
                    f"accurate enough (max relative error {error:.3f}); "
                    f"{radiation_fit.path.name} was not written, so linearize with "
                    "hydro_radiation_mode=0"
                )
                raise RadiationFitError(msg)
        return deck, radiation_fit

    def _copy_template(self, source: Path, destination: Path) -> None:
        """Copy the template case from ``source`` to ``destination``.

        Copies the deck's referenced files (or the whole tree, per ``template_copy``)
        with :func:`~vane.run.deck_files.copy_template_files`. With ``link_inputs`` the
        input files are hard-linked so the heavy read-only inputs are shared at the
        inode level (no per-case duplication); the small deck files are later edited via
        an atomic replace, which breaks the link so the template is untouched. Files
        OpenFAST overwrites in place (stale ``.out``/``.sum``/``.lin`` and other run
        artifacts) are *copied*, never hard-linked, so a run cannot mutate a shared
        inode across the template or concurrent cases. Hard-linking falls back to a
        byte copy where it is unsupported (e.g. a cross-filesystem output directory).
        """
        copy_template_files(
            self.fst_path,
            source,
            destination,
            mode=self.template_copy,
            link_inputs=self.link_inputs,
        )

    def run_case(
        self, operating_point: OperatingPoint, *, timeout: float | None = None
    ) -> SweepCase:
        """Generate, run, and collect one operating point.

        Parameters
        ----------
        operating_point : OperatingPoint
            The operating point to run.
        timeout : float or None, optional
            Per-run timeout in seconds.

        Returns
        -------
        SweepCase
            The generated case and its run outcome.

        Raises
        ------
        vane.run.DeckValidationError
            If ``validate`` is set and the configured deck cannot be linearized
            (raised before OpenFAST is launched).

        Notes
        -----
        Called directly, an unexpected exception (a file-system error while the case
        is copied or configured, say) propagates. Within :meth:`run_sweep` the same
        exception is instead recorded as a failed case (see there).
        """
        case, _ = self._run_case(operating_point, timeout=timeout, record_errors=False)
        return case

    def _run_case(
        self,
        operating_point: OperatingPoint,
        *,
        timeout: float | None,
        record_errors: bool,
    ) -> tuple[SweepCase, Exception | None]:
        """Run one operating point, optionally recording an unexpected exception.

        With ``record_errors`` an exception raised while an attempt is generated,
        run or collected ends the retry ladder and is returned alongside a failed
        case (``failure_mode == "case_error"``) that keeps the earlier attempts'
        history; otherwise it propagates. A :class:`~vane.run.DeckValidationError`
        and anything that is not an :class:`Exception` (a keyboard interrupt, say)
        always propagate.
        """
        timeout = _validated_timeout(timeout)
        policy = self.retry_policy
        name = _case_name(operating_point)
        result: OpenFastResult | None = None
        deck: Path | None = None
        # The per-operating-point starting settings the retry ladder escalates from
        # (tuned for the point's wind region when trim_tuning is set; the campaign base
        # otherwise). The campaign base snapshot below stays self.settings regardless.
        op_base = self._op_base_settings(operating_point)
        settings = policy.settings_for_attempt(op_base, 1)
        attempt = 0
        diagnosis: OpenFastDiagnosis | None = None
        reason: str | None = None
        residual_audit: OperatingPointResidualAudit | None = None
        history: list[LinearizationAttempt] = []
        fallback = False
        radiation_fit: RadiationFitRecord | None = None
        # Every attempt's full OpenFAST output, kept so it can be rewritten after the
        # next attempt regenerates (and so wipes) the case directory.
        logs: list[tuple[Path, str]] = []
        try:
            for attempt in range(1, policy.max_attempts + 1):
                deck, radiation_fit = self._generate_case_with_settings(
                    operating_point, settings
                )
                for log_path, text in logs:
                    _write_log(log_path, text)
                tmax = _deck_tmax(deck)
                if self.validate:
                    # The deck is already configured with the per-attempt settings, so
                    # validate it as it now stands (no further edits) before launching.
                    validate_deck(deck, LinearizationSettings())
                try:
                    result = run_openfast(
                        deck, executable=self.executable, timeout=timeout
                    )
                except OpenFastTimeoutError as exc:
                    # A timeout is a retriable hard-point failure (a slow trim), not
                    # a configuration error: synthesize a non-zero result so the
                    # ladder escalates and the case is recorded as non-converged,
                    # rather than the exception aborting the whole sweep and
                    # abandoning the other cases.
                    logger.warning(
                        "OpenFAST timed out for %s on attempt %d/%d: %s",
                        name,
                        attempt,
                        policy.max_attempts,
                        exc,
                    )
                    result = OpenFastResult(
                        fst_path=deck, returncode=_TIMEOUT_RETURNCODE, stderr=str(exc)
                    )
                    diagnosis = OpenFastDiagnosis(
                        mode=OpenFastFailureMode.TIMEOUT,
                        detail="run timed out",
                        openfast_message=str(exc),
                    )
                else:
                    diagnosis = _diagnose(operating_point, settings, result)
                reason = None if diagnosis is None else diagnosis.summary
                failure_mode = None if diagnosis is None else diagnosis.mode.value
                if diagnosis is None and self.acceptance_policy is not None:
                    residual_audit = audit_operating_point_residual(
                        result.lin_files, self.acceptance_policy
                    )
                    if not residual_audit.accepted:
                        # The normal retry ladder raises TrimTol. That is suitable for a
                        # trim that did not converge, but directionally wrong once a
                        # completed trim reveals excessive physical residual drift.
                        # Preserve the evidence and stop so the campaign can tighten the
                        # base trim or improve its initial conditions explicitly.
                        reason = residual_audit.reason or (
                            "operating-point physical residual was not accepted"
                        )
                        failure_mode = _RESIDUAL_REJECTED_MODE
                log_path = deck.with_name(f"{deck.stem}.attempt{attempt}.log")
                log_text = _attempt_log_text(attempt, settings, tmax, result, reason)
                log_written = _write_log(log_path, log_text)
                logs.append((log_path, log_text))
                history.append(
                    LinearizationAttempt(
                        attempt=attempt,
                        failure_mode=failure_mode,
                        reason=reason,
                        openfast_message=""
                        if diagnosis is None
                        else diagnosis.openfast_message,
                        failure_time=None
                        if diagnosis is None
                        else diagnosis.failure_time,
                        returncode=result.returncode,
                        n_lin=result.n_lin,
                        log_file=str(log_path) if log_written else None,
                        trim_tol=settings.trim_tol,
                        twr_kdmp=settings.twr_kdmp,
                        bld_kdmp=settings.bld_kdmp,
                        trim_gain_torque=settings.resolved_trim_gain_torque(),
                        trim_gain_pitch=settings.trim_gain_pitch,
                        sim_time=settings.sim_time,
                        tmax=tmax,
                        fallback=fallback,
                    )
                )
                if diagnosis is None:
                    break
                if not diagnosis.retriable:
                    if diagnosis.mode is OpenFastFailureMode.NON_FINITE_LINEARIZATION:
                        what = "produced a non-finite linearization of"
                        why = "a modelling degeneracy"
                    else:
                        what = "failed to initialise"
                        why = "an input problem"
                    logger.error(
                        "OpenFAST %s %s (%s; not retried): %s\n%s",
                        what,
                        name,
                        why,
                        diagnosis.detail,
                        diagnosis.openfast_message,
                    )
                    break
                if attempt == policy.max_attempts:
                    break
                next_settings = policy.next_attempt_settings(
                    op_base,
                    settings,
                    diagnosis,
                    attempt=attempt + 1,
                    previous_tmax=tmax,
                )
                # A retriable diagnosis always yields settings (None is non-retriable).
                assert next_settings is not None
                fallback = policy.uses_fallback(diagnosis, attempt=attempt + 1)
                logger.warning(
                    "Retrying linearization %s after attempt %d/%d failed (%s); next "
                    "attempt: TrimTol=%.3g, Twr_Kdmp=%.3g, Bld_Kdmp=%.3g, "
                    "torque gain=%.3g, pitch gain=%.3g, sim_time=%s%s",
                    name,
                    attempt,
                    policy.max_attempts,
                    reason,
                    next_settings.trim_tol,
                    next_settings.twr_kdmp,
                    next_settings.bld_kdmp,
                    next_settings.resolved_trim_gain_torque(),
                    next_settings.trim_gain_pitch,
                    next_settings.sim_time,
                    " (final fallback)" if fallback else "",
                )
                settings = next_settings
        except Exception as error:
            # A deck the pre-flight guard rejects is an input error found before
            # OpenFAST launches, not a case failure: it is raised, as before.
            if not record_errors or isinstance(error, DeckValidationError):
                raise
            case = self._case_error(
                operating_point,
                error,
                settings=settings,
                history=history,
                logs=logs,
                fallback=fallback,
            )
            return case, error
        assert deck is not None and result is not None
        converged = reason is None
        if not converged:
            # The ladder is exhausted without a trustworthy linearization. Do not pass
            # a forced 1-.lin result off as success: surface it loudly so callers act.
            logger.error(
                "Linearization %s did not converge after %d attempt(s): %s. The result "
                "is recorded as non-converged (converged=False); inspect %s",
                name,
                attempt,
                reason,
                self.case_dir(operating_point),
            )
        return SweepCase(
            operating_point=operating_point,
            case_dir=self.case_dir(operating_point),
            fst_path=deck,
            result=result,
            # Snapshot the as-run settings, the campaign base they escalated from, and
            # the executable now so provenance cannot be falsified by mutating the
            # driver later. snapshot() deep-copies the mutable structural_dofs dict so
            # the snapshots are fully independent of any later mutation.
            settings=settings.snapshot(),
            base_settings=self.settings.snapshot(),
            executable=str(self.executable),
            attempts=attempt,
            converged=converged,
            operating_point_residual=residual_audit,
            attempt_history=tuple(history),
            fallback_fields=tuple(self.retry_policy.fallback_fields),
            radiation_fit=radiation_fit,
        ), None

    def _case_error(
        self,
        operating_point: OperatingPoint,
        error: Exception,
        *,
        settings: LinearizationSettings,
        history: list[LinearizationAttempt],
        logs: list[tuple[Path, str]],
        fallback: bool,
    ) -> SweepCase:
        """Return the failed case recording an unexpected exception (``case_error``).

        The attempt that raised is appended to the earlier attempts' history, its
        traceback is written to its attempt log when the case directory exists, and the
        earlier attempts' logs are rewritten there (the failed regeneration may have
        wiped them).
        """
        assert self.template_root is not None  # set in __post_init__
        name = _case_name(operating_point)
        case_dir = self.case_dir(operating_point)
        deck = case_dir / self.fst_path.relative_to(self.template_root)
        number = len(history) + 1
        detail = f"{type(error).__name__}: {error}"
        reason = f"{OpenFastFailureMode.CASE_ERROR.value}: {detail}"
        trace = "".join(traceback.format_exception(error))
        logger.error(
            "Linearization %s raised an unexpected exception on attempt %d (%s); the "
            "case is recorded as failed (converged=False, failure_mode='case_error') "
            "and the sweep goes on with the other operating points",
            name,
            number,
            detail,
            exc_info=error,
        )
        log_written = False
        log_path = deck.with_name(f"{deck.stem}.attempt{number}.log")
        if deck.parent.is_dir():
            for earlier_path, text in logs:
                if not earlier_path.exists():
                    _write_log(earlier_path, text)
            log_written = _write_log(
                log_path, f"=== attempt {number}: {reason}\n\n{trace}"
            )
        history.append(
            LinearizationAttempt(
                attempt=number,
                failure_mode=OpenFastFailureMode.CASE_ERROR.value,
                reason=reason,
                openfast_message="",
                failure_time=None,
                returncode=_CASE_ERROR_RETURNCODE,
                n_lin=0,
                log_file=str(log_path) if log_written else None,
                trim_tol=settings.trim_tol,
                twr_kdmp=settings.twr_kdmp,
                bld_kdmp=settings.bld_kdmp,
                trim_gain_torque=settings.resolved_trim_gain_torque(),
                trim_gain_pitch=settings.trim_gain_pitch,
                sim_time=settings.sim_time,
                tmax=None,
                fallback=fallback,
            )
        )
        return SweepCase(
            operating_point=operating_point,
            case_dir=case_dir,
            fst_path=deck,
            result=OpenFastResult(
                fst_path=deck, returncode=_CASE_ERROR_RETURNCODE, stderr=trace
            ),
            settings=settings.snapshot(),
            base_settings=self.settings.snapshot(),
            executable=str(self.executable),
            attempts=number,
            converged=False,
            attempt_history=tuple(history),
            fallback_fields=tuple(self.retry_policy.fallback_fields),
        )

    def run_sweep(
        self,
        operating_points: Sequence[OperatingPoint],
        *,
        timeout: float | None = None,
        max_workers: int | None = None,
        strict: bool = False,
    ) -> list[SweepCase]:
        """Run a whole sweep of operating points.

        Each operating point is an independent OpenFAST run in its own case
        directory, so a sweep can run several points concurrently. OpenFAST itself is
        single-threaded here, so ``max_workers`` processes use ``max_workers`` cores;
        on a multi-core workstation the sweep wall-time approaches that of a single
        operating point rather than their sum.

        Parameters
        ----------
        operating_points : Sequence[OperatingPoint]
            The operating points to run, in order.
        timeout : float or None, optional
            Per-run timeout in seconds.
        max_workers : int or None, optional
            Number of operating points to run concurrently. ``None`` (the default)
            uses :func:`default_max_workers`, reserving one CPU for the machine.
            Results are returned in the input order regardless.
        strict : bool, optional
            When ``True``, raise :class:`SweepCaseError` once every case has finished
            if any case raised an unexpected exception; all the cases are attached as
            its ``cases``. Default ``False``: such a case is only recorded and logged.

        Returns
        -------
        list[SweepCase]
            One result per operating point, in the input order.

        Raises
        ------
        SweepCaseError
            With ``strict=True``, after all the cases have finished, when a case
            raised an unexpected exception.
        ValueError
            If two operating points map to the same case directory, a case
            directory would overlap the template root or use a reserved device name,
            or ``max_workers`` is less than 1. Names are compared case-insensitively
            (``casefold``), because the default Windows and macOS filesystems are case
            insensitive and two such cases would otherwise share -- and delete -- one
            directory. All such checks run up front, before any case is launched.
        vane.run.DeckValidationError
            If ``validate`` is set and a configured deck cannot be linearized: an
            input error found before that case's OpenFAST run is launched.

        Notes
        -----
        One operating point never discards the others. An OpenFAST failure is
        classified and retried (see :class:`LinearizationRetryPolicy`); an
        *unexpected* exception while a case is generated, run or collected -- a
        file-system error that outlasted the transient-error retry, an unreadable
        sub-file -- ends that point's retry ladder and is recorded as a failed case:
        ``converged`` is ``False``, ``failure_mode`` is ``"case_error"``
        (:attr:`~vane.run.OpenFastFailureMode.CASE_ERROR`), the final
        :class:`LinearizationAttempt` carries the exception type and message as its
        ``reason`` (and so does the provenance record), its attempt log holds the
        traceback, and the result's return code is ``-1``. The earlier attempts'
        records are kept. The argument checks listed under *Raises* are boundary
        errors and are still raised, the ``ValueError`` ones before any case is
        copied or launched.
        """
        timeout = _validated_timeout(timeout)
        if max_workers is None:
            max_workers = default_max_workers()
        if (
            isinstance(max_workers, bool)
            or not isinstance(max_workers, Integral)
            or max_workers < 1
        ):
            msg = f"max_workers must be >= 1, got {max_workers}"
            raise ValueError(msg)
        max_workers = int(max_workers)
        # A rotating point needs the rated wind speed for its trim region; fail before
        # any case is copied or any OpenFAST process starts.
        for op in operating_points:
            if op.is_rotating:
                settings = self._op_base_settings(op)
                if needs_rated_wind_speed(settings, self.fst_path):
                    settings.is_below_rated(op.wind_speed)
        by_key: dict[str, list[str]] = {}
        for op in operating_points:
            name = _case_name(op)
            by_key.setdefault(name.casefold(), []).append(name)
        duplicates = [names for names in by_key.values() if len(names) > 1]
        if duplicates:
            msg = (
                "operating points map to duplicate case names (compared "
                "case-insensitively, as on Windows/macOS filesystems): "
                f"{sorted(duplicates)}"
            )
            raise ValueError(msg)
        # Validate every destination up front so an invalid case fails the whole sweep
        # before any worker copies a case or launches OpenFAST (preserving the
        # fail-before-launch behaviour of the sequential path under concurrency).
        for op in operating_points:
            self._validated_case_dir(op)

        outcomes: list[tuple[SweepCase, Exception | None]]
        if max_workers == 1:
            outcomes = []
            for index, op in enumerate(operating_points, start=1):
                logger.info(
                    "Linearization sweep %d/%d: %s",
                    index,
                    len(operating_points),
                    _case_name(op),
                )
                outcomes.append(self._run_case(op, timeout=timeout, record_errors=True))
        else:
            logger.info(
                "Linearization sweep: %d operating points on %d workers",
                len(operating_points),
                max_workers,
            )
            # Each case writes to a per-point directory and spawns its own OpenFAST
            # process, so the cases are independent and safe to run concurrently.
            # Futures are collected in submission order to preserve the input order.
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(self._run_case, op, timeout=timeout, record_errors=True)
                    for op in operating_points
                ]
                outcomes = [future.result() for future in futures]
        cases = [case for case, _ in outcomes]
        errors = [(case, error) for case, error in outcomes if error is not None]
        if errors:
            summary = "; ".join(
                f"{_case_name(case.operating_point)}: {type(error).__name__}: {error}"
                for case, error in errors
            )
            logger.error(
                "%d of %d linearization case(s) raised an unexpected exception and "
                "were recorded as failed (failure_mode='case_error'): %s",
                len(errors),
                len(cases),
                summary,
            )
            if strict:
                msg = (
                    f"{len(errors)} linearization case(s) raised an unexpected "
                    "exception (the other cases finished; see .cases): " + summary
                )
                raise SweepCaseError(msg, cases=cases) from errors[0][1]
        return cases

    def provenance(
        self,
        cases: Sequence[SweepCase],
        *,
        timestamp: datetime | None = None,
    ) -> RunProvenance:
        """Build a reproducibility record for a completed sweep.

        Records the :class:`LinearizationSettings` every case was configured with, the
        library version and executable, and each case's outcome, so the conditions a
        set of ``.lin`` files was produced under travel with them.

        The recorded settings and executable are taken from the cases' own snapshots
        (frozen when each case ran), not from the driver's current attributes, so the
        record stays faithful even if the driver was reconfigured after the sweep.

        Parameters
        ----------
        cases : Sequence[SweepCase]
            The cases returned by :meth:`run_sweep` (or :meth:`run_case`).
        timestamp : datetime.datetime or None, optional
            The time to record; defaults to the current UTC time.

        Returns
        -------
        RunProvenance
            The settings, environment, and per-case record for the sweep.

        Raises
        ------
        ValueError
            If the cases differ on a campaign-wide setting (anything other than the
            per-case retry-escalated trim tolerance/damping) or on the executable, so no
            single run configuration describes them.
        """
        moment = timestamp if timestamp is not None else datetime.now(timezone.utc)
        settings, executable = self._run_configuration(cases)
        return RunProvenance(
            vane_version=vane_version(),
            created_at=moment.isoformat(),
            executable=executable,
            settings=asdict(settings),
            auto_configuration=self.auto_configuration,
            cases=tuple(
                CaseProvenance(
                    name=_case_name(case.operating_point),
                    wind_speed=case.operating_point.wind_speed,
                    rotor_speed_rpm=case.operating_point.rotor_speed_rpm,
                    blade_pitch_deg=case.operating_point.blade_pitch_deg,
                    case_dir=str(case.case_dir),
                    returncode=case.result.returncode,
                    succeeded=case.succeeded,
                    n_lin=case.result.n_lin,
                    attempts=case.attempts,
                    converged=case.converged,
                    trim_tol=case.settings.trim_tol,
                    twr_kdmp=case.settings.twr_kdmp,
                    bld_kdmp=case.settings.bld_kdmp,
                    trim_gain_torque=case.settings.resolved_trim_gain_torque(),
                    trim_gain_pitch=case.settings.trim_gain_pitch,
                    settle_assessed=None
                    if case.operating_point.settle_convergence is None
                    else case.operating_point.settle_convergence.assessed,
                    settle_converged=None
                    if case.operating_point.settle_convergence is None
                    else case.operating_point.settle_convergence.converged,
                    settle_translational_velocity_rms=None
                    if case.operating_point.settle_convergence is None
                    else (
                        case.operating_point.settle_convergence.translational_velocity_rms
                    ),
                    settle_angular_velocity_rms=None
                    if case.operating_point.settle_convergence is None
                    else case.operating_point.settle_convergence.angular_velocity_rms,
                    settle_time=None
                    if case.operating_point.settle_convergence is None
                    else case.operating_point.settle_convergence.settle_time,
                    settle_iterations=None
                    if case.operating_point.settle_convergence is None
                    else case.operating_point.settle_convergence.iterations,
                    settle_attempts=None
                    if case.operating_point.settle_convergence is None
                    else case.operating_point.settle_convergence.attempts,
                    settle_criterion=None
                    if case.operating_point.settle_convergence is None
                    else case.operating_point.settle_convergence.criterion,
                    settle_confirmation=None
                    if case.operating_point.settle_convergence is None
                    else case.operating_point.settle_convergence.confirmation,
                    settle_curve_rotor_speed_rpm=None
                    if (curve := case.operating_point.settle_curve_check) is None
                    else curve.rotor_speed.curve,
                    settle_curve_pitch_deg=None if curve is None else curve.pitch.curve,
                    settle_curve_status=None if curve is None else curve.status,
                    operating_point_residual=case.operating_point_residual,
                    sim_time=case.settings.sim_time,
                    failure_mode=case.failure_mode,
                    attempt_history=case.attempt_history,
                    fallback_overrides=self._fallback_overrides(case),
                    radiation_fit=None
                    if case.radiation_fit is None
                    else case.radiation_fit.to_dict(),
                )
                for case in cases
            ),
        )

    def _fallback_overrides(self, case: SweepCase) -> dict[str, object] | None:
        """Return the as-run fallback fields of a case that used the fallback."""
        if not case.attempt_history or not case.attempt_history[-1].fallback:
            return None
        return {
            name: _json_value(getattr(case.settings, name))
            for name in case.fallback_fields
        }

    def _run_configuration(
        self, cases: Sequence[SweepCase]
    ) -> tuple[LinearizationSettings, str]:
        """Return the campaign-wide settings/executable that produced ``cases``.

        Uses the cases' own run-time snapshots (faithful even if the driver was later
        mutated), falling back to the driver's current values only for an empty sweep.
        Cases may legitimately differ on the per-case retry-escalated fields
        (``trim_tol``/``twr_kdmp``/``bld_kdmp``) when one hard operating point needed a
        looser trim or higher damping; those are recorded per case, not campaign-wide.
        The campaign base for those fields is taken from each case's own
        ``base_settings`` snapshot (the un-escalated values frozen at run time), *not*
        from the live ``self.settings``, so the record is independent of any post-run
        driver mutation. Rejects cases differing on any campaign-wide field or base.
        """
        if not cases:
            return replace(self.settings), str(self.executable)
        # The campaign base comes from the cases' own run-time base snapshots, so a
        # later mutation of self.settings cannot change what provenance reports. All
        # cases of a sweep share one base; reject any that disagree (mixed base).
        base = replace(cases[0].base_settings)
        per_case_fields = {
            *_RETRY_ESCALATED_FIELDS,
            *(name for case in cases for name in case.fallback_fields),
        }
        base_escalated = {name: getattr(base, name) for name in per_case_fields}
        executable = cases[0].executable
        for case in cases:
            if replace(case.base_settings) != base:
                msg = (
                    "cases were produced with differing campaign-base "
                    "LinearizationSettings; provenance cannot record a single base"
                )
                raise ValueError(msg)
            # The as-run settings must match the base once the per-case escalated
            # fields are normalized away, so only a real campaign-wide diff is rejected.
            if replace(case.settings, **base_escalated) != base:
                msg = (
                    "cases were produced with differing campaign-wide "
                    "LinearizationSettings (beyond the per-case retry escalation); "
                    "provenance cannot record a single run configuration"
                )
                raise ValueError(msg)
            if case.executable != executable:
                msg = "cases were produced with differing executables"
                raise ValueError(msg)
        return base, executable


def _case_name(operating_point: OperatingPoint) -> str:
    """Return a filesystem-safe case-directory name for an operating point.

    Path separators and other special characters are replaced with underscores.
    Trailing dots are removed, as Windows silently drops them (``"ws8."`` and
    ``"ws8"`` would otherwise be distinct names for one directory). A name that
    sanitizes to nothing or a reserved path component (``"."``, ``".."``, or all
    dots) would resolve to the output directory itself or its parent, so it falls
    back to the operating-point values.
    """
    raw = operating_point.name or f"ws{operating_point.wind_speed:g}"
    sanitized = _sanitize(raw).rstrip(".")
    if not sanitized:
        op = operating_point
        sanitized = _sanitize(f"ws{op.wind_speed:g}_rpm{op.rotor_speed_rpm:g}")
    return sanitized.rstrip(".")


def _sanitize(name: str) -> str:
    """Replace path separators and special characters with underscores."""
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in name)


def _diagnose(
    operating_point: OperatingPoint,
    settings: LinearizationSettings,
    result: OpenFastResult,
) -> OpenFastDiagnosis | None:
    """Return why a run should be retried, or ``None`` when it converged acceptably.

    A non-zero exit is classified from OpenFAST's output
    (:func:`~vane.run.classify_openfast_output`). A rotating CalcSteady run that exited
    normally is still non-converged when it emits fewer full-system ``.lin`` files than
    the requested ``NLinTimes`` (the "forced 1 .lin" outcome) or logs that it forced
    the linearization without finding the steady state. An explicit-time run likewise
    requires every requested snapshot; equilibrium itself is assessed by the
    independent acceptance policy. A run that passes those checks is still rejected
    as :attr:`~vane.run.OpenFastFailureMode.NON_FINITE_LINEARIZATION` when any of its
    full-system ``.lin`` files holds a NaN or infinity (an operating-point value or a
    matrix entry); that failure is not retried.
    """
    if not result.succeeded:
        return classify_openfast_output(result.output, result.returncode)
    rotating_trim = operating_point.is_rotating and settings.calc_steady
    explicit_times = settings.linearization_times
    expected = (
        settings.n_lin_times
        if rotating_trim
        else len(explicit_times)
        if explicit_times is not None
        else 1
    )
    if result.n_lin < expected:
        found = f"expected {expected} full-system .lin files but found {result.n_lin}; "
        if rotating_trim:
            return OpenFastDiagnosis(
                mode=OpenFastFailureMode.TRIM_NOT_CONVERGED,
                detail=found
                + "CalcSteady did not converge to the requested azimuth sweep",
            )
        return OpenFastDiagnosis(
            mode=OpenFastFailureMode.INCOMPLETE_LINEARIZATION,
            detail=found + "the explicit-time linearization sweep is incomplete",
        )
    if rotating_trim:
        diagnosis = classify_openfast_output(result.output, result.returncode)
        if diagnosis is not None:
            return diagnosis
        diagnosis = _rotor_speed_spread_diagnosis(result)
        if diagnosis is not None:
            return diagnosis
    return _non_finite_diagnosis(result)


# Absolute floor of the rotor-speed spread test (rad/s), as in MBC3.
_ROTOR_SPEED_DRIFT_ATOL = 1e-6
_ROTOR_SPEED_HEADER_RE = re.compile(r"^\s*Rotor Speed:\s*(\S+)", re.MULTILINE)


def _header_rotor_speed(path: Path) -> float | None:
    """Return the ``Rotor Speed`` header value of a ``.lin`` file, or ``None``.

    Only the first few kilobytes are read: the header precedes the large tables.
    """
    try:
        with path.open(encoding="utf-8", errors="ignore") as handle:
            head = handle.read(8192)
    except OSError:
        return None
    match = _ROTOR_SPEED_HEADER_RE.search(head)
    if match is None:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _rotor_speed_spread_diagnosis(result: OpenFastResult) -> OpenFastDiagnosis | None:
    """Return a diagnosis when the sweep's rotor speed is not constant enough for MBC.

    CalcSteady stops when successive revolutions agree within ``TrimTol``; a loose
    tolerance can accept a response whose rotor speed still varies across the azimuth
    files by more than MBC3 allows (``ROTOR_SPEED_DRIFT_RTOL`` of the mean), which the
    post-processing would then reject. The spread is read from the file headers,
    exactly as :func:`~vane.io.mbc_transform.mbc3_transform` computes it.
    """
    speeds = [
        speed
        for path in result.lin_files
        if (speed := _header_rotor_speed(path)) is not None
    ]
    if len(speeds) < 2:
        return None
    spread = max(speeds) - min(speeds)
    mean = sum(abs(speed) for speed in speeds) / len(speeds)
    if spread <= max(ROTOR_SPEED_DRIFT_RTOL * mean, _ROTOR_SPEED_DRIFT_ATOL):
        return None
    return OpenFastDiagnosis(
        mode=OpenFastFailureMode.NON_PERIODIC_SWEEP,
        detail=(
            f"rotor speed varies {100.0 * spread / mean:.2f} % across the azimuth "
            f"sweep ({min(speeds):.4g} to {max(speeds):.4g} rad/s), more than the "
            f"{100.0 * ROTOR_SPEED_DRIFT_RTOL:.0f} % MBC3 accepts; the trim stopped "
            "on a response that is not periodic"
        ),
    )


# An AeroDyn unsteady-aerodynamics state (or its derivative) of one blade node, as
# OpenFAST describes it in the operating-point tables: ``AD x3 blade 1, node 59, -``.
_AERODYN_UA_STATE_RE = re.compile(
    r"^(?:First time derivative of )?AD x\d+ blade \d+, node \d+"
)
_UA_TIP_HINT = (
    "an AeroDyn unsteady-aerodynamics state is non-finite; when it is the blade-tip "
    "node, the tip loss model fixes the tip inflow there and unsteady aerodynamics "
    "should end inboard of it: set UAEndRad below 1 (e.g. 0.99, or "
    "LinearizationSettings(ua_end_radius=0.99))"
)
_NON_FINITE_HINT = (
    "a non-finite linearization is a modelling degeneracy that other trim settings "
    "cannot cure; fix the model input that produces the named variable"
)


def _non_finite_diagnosis(result: OpenFastResult) -> OpenFastDiagnosis | None:
    """Return a diagnosis when any full-system ``.lin`` file holds a NaN or infinity.

    Every file is scanned (:func:`~vane.io.find_nonfinite_lin_value`) so the
    diagnosis can say how many are affected; the first offending value names the
    variable. A file that cannot be read is skipped: the downstream reader reports it.
    """
    first: NonFiniteLinValue | None = None
    affected = 0
    for path in result.lin_files:
        try:
            found = find_nonfinite_lin_value(path)
        except OSError as error:
            logger.debug("Could not scan %s for non-finite values: %s", path, error)
            continue
        if found is None:
            continue
        affected += 1
        if first is None:
            first = found
    if first is None:
        return None
    hint = (
        _UA_TIP_HINT
        if _AERODYN_UA_STATE_RE.match(first.variable) is not None
        else _NON_FINITE_HINT
    )
    logger.warning(
        "%d of %d .lin files of %s hold non-finite values; the first is %s. Hint: %s",
        affected,
        result.n_lin,
        result.fst_path,
        first.summary,
        hint,
    )
    return OpenFastDiagnosis(
        mode=OpenFastFailureMode.NON_FINITE_LINEARIZATION,
        detail=(
            f"{affected} of {result.n_lin} full-system .lin files hold non-finite "
            f"values; first: {first.variable} = {first.token}"
        ),
        openfast_message=f"{first.summary}. Hint: {hint}",
    )


def _deck_tmax(deck: Path) -> float | None:
    """Return the ``TMax`` a configured deck carries, or ``None`` if unreadable."""
    try:
        value = fortran_float(FastInputFile.read(deck).get("TMax"))
    except (FastInputError, OSError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _attempt_log_text(
    attempt: int,
    settings: LinearizationSettings,
    tmax: float | None,
    result: OpenFastResult,
    reason: str | None,
) -> str:
    """Return the per-attempt log: a settings header and OpenFAST's full output."""
    header = [
        f"attempt: {attempt}",
        f"deck: {result.fst_path}",
        f"returncode: {result.returncode}",
        f"n_lin: {result.n_lin}",
        f"outcome: {'converged' if reason is None else reason}",
        f"TrimTol: {settings.trim_tol!r}",
        f"Twr_Kdmp: {settings.twr_kdmp!r}",
        f"Bld_Kdmp: {settings.bld_kdmp!r}",
        f"torque trim gain: {settings.resolved_trim_gain_torque()!r}",
        f"pitch trim gain: {settings.trim_gain_pitch!r}",
        f"sim_time: {settings.sim_time!r}",
        f"TMax: {tmax!r}",
    ]
    return "\n".join(
        [
            *header,
            "",
            "===== stdout =====",
            result.stdout,
            "===== stderr =====",
            result.stderr,
        ]
    )


def _write_log(path: Path, text: str) -> bool:
    """Write an attempt log, returning whether it was written.

    A log that cannot be written (e.g. a full disk) is reported but never fails the
    case: the run outcome is still returned and recorded.
    """
    try:
        path.write_text(text, encoding="utf-8", errors="replace")
    except OSError as error:
        logger.warning("Could not write the OpenFAST log %s: %s", path, error)
        return False
    return True


def _json_value(value: object) -> object:
    """Return a settings value in a JSON-friendly form (tuples become lists)."""
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return value
