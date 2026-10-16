# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""High-level modal-analysis pipeline.

``ModalPipeline`` runs the full workflow end to end: for each operating point it
applies the MBC3 transform, solves for modes, then links the modes across
operating points, assembles the Campbell diagram, and flags resonance crossings.
The :class:`PipelineResult` exposes every intermediate product and convenience
exports to a state-space system, so a complete analysis is a few lines of code.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vane.campbell.builder import build_campbell
from vane.campbell.excitation import (
    DEFAULT_HARMONICS,
    _validate_harmonics,
    find_resonances,
)
from vane.io.mbc_transform import mbc3_transform
from vane.modal.eigensolver import modes_from_mbc
from vane.modal.identifier import (
    _STRATEGIES,
    DEFAULT_HEAVILY_DAMPED_THRESHOLD,
    _validate_heavily_damped_threshold,
    identify_modes,
)
from vane.sysid.state_space import state_space_from_mbc

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vane.campbell.builder import CampbellDiagram
    from vane.campbell.excitation import ResonanceCrossing
    from vane.io.lin_reader import LinFile
    from vane.io.mbc_transform import MBCResult
    from vane.modal.eigensolver import ModalSolution
    from vane.modal.identifier import (
        IdentificationResult,
        ModeTrack,
        TrackingStrategy,
    )
    from vane.plugins import Tracker
    from vane.sysid.state_space import StateSpace

__all__ = ["DEFAULT_PARAMETER_RTOL", "ModalPipeline", "PipelineResult"]

_RPM_PARAMETER = "rotor_speed_rpm"
_WIND_PARAMETER = "wind_speed"
# The CLI spelling of each operating parameter, for actionable error messages.
_CLI_PARAMETER = {_RPM_PARAMETER: "rpm", _WIND_PARAMETER: "wind"}

#: Default relative tolerance below which two operating points' sweep-axis values
#: are treated as the same point (a plateau), scaled by the largest-magnitude value
#: on the axis. It sits well above the round-off that separates nominally equal
#: trimmed/settled values (``.lin`` headers carry a handful of significant digits,
#: and azimuth averages of the same schedule differ only in the trailing digits)
#: and well below any physically meaningful sweep step (typically 1e-2 or more of
#: the axis magnitude).
DEFAULT_PARAMETER_RTOL = 1e-4


@dataclass
class PipelineResult:
    """Outputs of a full modal-analysis run.

    Parameters
    ----------
    mbc_results : list[MBCResult]
        The MBC transform result for each operating point.
    solutions : list[ModalSolution]
        The modal solution for each operating point.
    identification : IdentificationResult
        The cross-operating-point mode tracks.
    campbell : CampbellDiagram
        The assembled Campbell diagram.
    resonances : list[ResonanceCrossing]
        Detected resonance crossings (empty for non-rotor-speed runs).
    parameter_name : str
        Name of the operating parameter used.
    """

    mbc_results: list[MBCResult]
    solutions: list[ModalSolution]
    identification: IdentificationResult
    campbell: CampbellDiagram
    resonances: list[ResonanceCrossing]
    parameter_name: str

    @property
    def tracks(self) -> list[ModeTrack]:
        """Return the identified mode tracks."""
        return self.identification.tracks

    def state_space(self, operating_point: int = 0) -> StateSpace:
        """Export the state-space system at one operating point.

        Parameters
        ----------
        operating_point : int, optional
            Index of the operating point to export (default the first).

        Returns
        -------
        StateSpace
            The continuous-time system from that operating point's MBC result.

        Raises
        ------
        IndexError
            If ``operating_point`` is negative or out of range.
        """
        if not 0 <= operating_point < len(self.mbc_results):
            msg = (
                f"operating_point {operating_point} out of range "
                f"[0, {len(self.mbc_results)})"
            )
            raise IndexError(msg)
        return state_space_from_mbc(self.mbc_results[operating_point])


class ModalPipeline:
    """End-to-end modal analysis from linearization files to a Campbell diagram."""

    def __init__(
        self,
        *,
        frequency_weight: float = 0.5,
        mac_threshold: float = 0.5,
        harmonics: Sequence[int] = DEFAULT_HARMONICS,
        tracking_strategy: TrackingStrategy = "single_path",
        branch_review: bool = False,
        resolve: bool = False,
        tracker: Tracker | None = None,
        heavily_damped_threshold: float | None = DEFAULT_HEAVILY_DAMPED_THRESHOLD,
    ) -> None:
        """Configure the pipeline.

        Parameters
        ----------
        frequency_weight : float, optional
            Frequency-continuity weight for cross-operating-point tracking.
        mac_threshold : float, optional
            Minimum MAC for a track to be extended.
        harmonics : Sequence[int], optional
            Excitation harmonics for resonance detection.
        tracking_strategy : {"single_path", "simultaneous"}, optional
            Cross-operating-point tracking strategy (see
            :func:`vane.modal.identifier.identify_modes`). ``"simultaneous"`` is more
            reliable for dense, crossing-rich spectra such as floating-platform
            sweeps.
        branch_review : bool, optional
            When ``True``, flag crossing/veering hotspots on the result's
            ``branch_groups`` for audit (the tracks are unchanged). Default ``False``.
        resolve : bool, optional
            When ``True``, also attach the smooth (minimum-curvature) alternative to
            each eligible hotspot (implies ``branch_review``). Default ``False``.
        tracker : Tracker or None, optional
            A custom cross-operating-point tracker (any
            :class:`~vane.plugins.Tracker`, e.g. a Floquet/LTP tracker). When given it
            replaces the built-in :func:`~vane.modal.identify_modes`, and the
            ``frequency_weight`` / ``mac_threshold`` / ``tracking_strategy`` /
            ``branch_review`` / ``resolve`` settings — which tune the built-in — no
            longer apply to tracking. Default ``None`` (use the built-in).
        heavily_damped_threshold : float or None, optional
            Damping ratio, in ``(0, 1]``, at or above which a mode is heavily damped
            (near-critical, measured in the rotating frame for a blade whirl mode)
            and is tracked apart from the lightly damped Campbell lines, into the
            identification's ``heavily_damped_tracks``. Default
            :data:`~vane.modal.identifier.DEFAULT_HEAVILY_DAMPED_THRESHOLD`;
            ``None`` tracks every mode together. See
            :func:`vane.modal.identifier.identify_modes`. Not used by a custom
            ``tracker``.

        Raises
        ------
        ValueError
            If ``frequency_weight`` or ``mac_threshold`` is not in ``[0, 1]``,
            ``heavily_damped_threshold`` is not in ``(0, 1]`` or ``None``,
            ``tracking_strategy`` is unknown, or ``harmonics`` contains a
            non-positive, fractional, non-finite, or duplicate value.
        """
        for name, value in (
            ("frequency_weight", frequency_weight),
            ("mac_threshold", mac_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                msg = f"{name} must be in [0, 1], got {value}"
                raise ValueError(msg)
        if tracking_strategy not in _STRATEGIES:
            msg = (
                f"tracking_strategy must be one of {sorted(_STRATEGIES)}, "
                f"got {tracking_strategy!r}"
            )
            raise ValueError(msg)
        _validate_heavily_damped_threshold(heavily_damped_threshold)
        self.heavily_damped_threshold = heavily_damped_threshold
        self.frequency_weight = frequency_weight
        self.mac_threshold = mac_threshold
        self.harmonics = tuple(_validate_harmonics(harmonics))
        self.tracking_strategy = tracking_strategy
        # Resolution implies review: it computes the smooth alternative for the
        # hotspots that the review pass detects.
        self.branch_review = branch_review or resolve
        self.resolve = resolve
        self.tracker = tracker

    def run(
        self,
        operating_points: Sequence[Sequence[LinFile]],
        *,
        parameter_name: str = _RPM_PARAMETER,
        parameter_rtol: float = DEFAULT_PARAMETER_RTOL,
    ) -> PipelineResult:
        """Run the full pipeline over a set of operating points.

        Parameters
        ----------
        operating_points : Sequence[Sequence[LinFile]]
            One entry per operating point, each a sequence of linearization files
            spanning that point's azimuth sweep.
        parameter_name : str, optional
            The operating parameter to plot against: ``"rotor_speed_rpm"`` (the
            default) or ``"wind_speed"``. Above rated the trimmed rotor speed is
            constant, so a sweep that extends past rated must use ``"wind_speed"``.
        parameter_rtol : float, optional
            Relative tolerance, scaled by the largest-magnitude parameter value,
            below which two operating points count as the same point on the sweep
            axis. Such a plateau is rejected because ordering by nearly equal values
            is arbitrary and would link non-adjacent operating points during
            tracking. Default :data:`DEFAULT_PARAMETER_RTOL`; must be a finite
            number in ``[0, 1)``.

        Returns
        -------
        PipelineResult
            Every intermediate product of the analysis. Operating points are
            ordered by ascending operating parameter regardless of input order, so
            tracking and resonance detection see a monotonic sweep.

        Raises
        ------
        ValueError
            If ``operating_points`` is empty, ``parameter_name`` is unknown,
            ``parameter_rtol`` is invalid, or the operating-parameter values are
            non-finite or not distinct (beyond ``parameter_rtol``) across points.
        """
        if not operating_points:
            msg = "operating_points must contain at least one operating point"
            raise ValueError(msg)
        _validate_parameter_name(parameter_name)
        _validate_parameter_rtol(parameter_rtol)

        mbc_results = [mbc3_transform(list(point)) for point in operating_points]
        solutions = [modes_from_mbc(result) for result in mbc_results]
        parameter_values = [
            self._parameter_value(result, parameter_name) for result in mbc_results
        ]
        _validate_sweep_axis(
            parameter_values,
            parameter_name,
            rtol=parameter_rtol,
            wind_speeds=[result.wind_speed for result in mbc_results],
        )

        # Sort by the operating parameter so cross-operating-point tracking and
        # resonance interpolation see a monotonic sweep regardless of input order.
        order = sorted(range(len(parameter_values)), key=parameter_values.__getitem__)
        mbc_results = [mbc_results[i] for i in order]
        solutions = [solutions[i] for i in order]
        parameter_values = [parameter_values[i] for i in order]

        if self.tracker is not None:
            # A custom tracker replaces the built-in; it receives the solutions and
            # the (sorted) sweep axis and is responsible for its own tracking logic.
            identification = self.tracker(solutions, parameter_values=parameter_values)
        else:
            identification = identify_modes(
                solutions,
                frequency_weight=self.frequency_weight,
                mac_threshold=self.mac_threshold,
                strategy=self.tracking_strategy,
                branch_review=self.branch_review,
                resolve=self.resolve,
                # The sweep axis (sorted to match the solutions) so resolution measures
                # curvature against the real, possibly non-uniform, parameter spacing.
                parameter_values=parameter_values,
                heavily_damped_threshold=self.heavily_damped_threshold,
                # Each point's rotor speed, so a blade whirl mode's damping is read
                # in the rotating frame where the blade mode is defined.
                rotor_speeds_rpm=_finite_rotor_speeds(mbc_results),
            )
        campbell = build_campbell(
            identification, parameter_values, parameter_name=parameter_name
        )
        resonances = (
            find_resonances(campbell, self.harmonics)
            if parameter_name == _RPM_PARAMETER
            else []
        )
        return PipelineResult(
            mbc_results=mbc_results,
            solutions=solutions,
            identification=identification,
            campbell=campbell,
            resonances=resonances,
            parameter_name=parameter_name,
        )

    @staticmethod
    def _parameter_value(result: MBCResult, parameter_name: str) -> float:
        """Return the operating-parameter value from an MBC result."""
        if parameter_name == _RPM_PARAMETER:
            return result.rotor_speed_rpm
        return result.wind_speed


def _finite_rotor_speeds(mbc_results: Sequence[MBCResult]) -> list[float] | None:
    """Return every point's rotor speed (rev/min), or None if any is not finite.

    Parameters
    ----------
    mbc_results : Sequence[MBCResult]
        The MBC results in sweep order.

    Returns
    -------
    list[float] or None
        The rotor speeds, or ``None`` (blade whirl modes are then not classified
        heavily damped) when a point carries no finite rotor speed.
    """
    speeds = [float(result.rotor_speed_rpm) for result in mbc_results]
    return speeds if all(math.isfinite(value) for value in speeds) else None


def _validate_parameter_name(parameter_name: str) -> None:
    """Reject an unknown operating-parameter name.

    Parameters
    ----------
    parameter_name : str
        The requested sweep axis.

    Raises
    ------
    ValueError
        If ``parameter_name`` is neither ``"rotor_speed_rpm"`` nor ``"wind_speed"``.
    """
    if parameter_name not in (_RPM_PARAMETER, _WIND_PARAMETER):
        msg = (
            f"parameter_name must be '{_RPM_PARAMETER}' or '{_WIND_PARAMETER}', "
            f"got '{parameter_name}'"
        )
        raise ValueError(msg)


def _validate_parameter_rtol(parameter_rtol: float) -> None:
    """Reject a non-finite, negative, or too-large sweep-axis plateau tolerance.

    Parameters
    ----------
    parameter_rtol : float
        The relative plateau tolerance.

    Raises
    ------
    ValueError
        If ``parameter_rtol`` is not a finite real number in ``[0, 1)``.
    """
    if (
        isinstance(parameter_rtol, bool)
        or not isinstance(parameter_rtol, (int, float))
        or not math.isfinite(parameter_rtol)
        or not 0.0 <= parameter_rtol < 1.0
    ):
        msg = (
            f"parameter_rtol must be a finite number in [0, 1), got {parameter_rtol!r}"
        )
        raise ValueError(msg)


def _validate_sweep_axis(
    values: Sequence[float],
    parameter_name: str,
    *,
    rtol: float = DEFAULT_PARAMETER_RTOL,
    wind_speeds: Sequence[float] | None = None,
) -> None:
    """Validate the sweep axis is finite and distinct beyond a plateau tolerance.

    Operating points are sorted by the sweep axis before tracking, so two points
    whose values differ only by round-off (e.g. trimmed above-rated rotor speeds of
    ``12.1000001`` and ``12.0999998`` rpm) would be ordered arbitrarily relative to
    the physical sweep, and tracking would link non-adjacent points. Exact and near
    duplicates are therefore both rejected rather than silently producing a
    scrambled Campbell diagram. The axis is never switched automatically.

    Parameters
    ----------
    values : Sequence[float]
        The operating-parameter value at each operating point.
    parameter_name : str
        The operating-parameter name, for the error message.
    rtol : float, optional
        Two values no further apart than ``rtol * max(abs(values))`` form a
        plateau. Default :data:`DEFAULT_PARAMETER_RTOL`.
    wind_speeds : Sequence[float] or None, optional
        The mean wind speed at each operating point. When given for a rejected
        rotor-speed axis, the error states whether the wind-speed axis would be
        valid for these operating points.

    Raises
    ------
    ValueError
        If any value is non-finite or two operating points' values lie within the
        plateau tolerance (including exact duplicates).
    """
    if not all(math.isfinite(value) for value in values):
        msg = f"operating parameter '{parameter_name}' has non-finite values: {values}"
        raise ValueError(msg)
    pair = _plateau_pair(values, rtol)
    if pair is None:
        return
    low, high = pair
    if low == high:
        reason = f"got duplicates ({low!r}) in {list(values)}"
    else:
        reason = (
            f"{low!r} and {high!r} differ by no more than parameter_rtol={rtol:g} of "
            "the axis magnitude, so the sweep axis has a plateau (e.g. a constant "
            "trimmed rotor speed above rated) and ordering by it would scramble the "
            "operating points and link non-adjacent points during tracking"
        )
    msg = (
        f"operating points must have distinct '{parameter_name}' values; {reason}"
        + _axis_advice(parameter_name, rtol=rtol, wind_speeds=wind_speeds)
    )
    raise ValueError(msg)


def _plateau_pair(values: Sequence[float], rtol: float) -> tuple[float, float] | None:
    """Return the first sorted-adjacent pair within the plateau tolerance.

    Parameters
    ----------
    values : Sequence[float]
        Finite sweep-axis values.
    rtol : float
        Relative tolerance, scaled by the largest-magnitude value.

    Returns
    -------
    tuple[float, float] or None
        The offending ``(lower, higher)`` pair, or ``None`` when every pair of
        values is separated by more than the tolerance.
    """
    ordered = sorted(values)
    threshold = rtol * max((abs(value) for value in ordered), default=0.0)
    for low, high in itertools.pairwise(ordered):
        if high - low <= threshold:
            return low, high
    return None


def _axis_advice(
    parameter_name: str, *, rtol: float, wind_speeds: Sequence[float] | None
) -> str:
    """Return an actionable hint for a rejected rotor-speed sweep axis.

    Parameters
    ----------
    parameter_name : str
        The rejected sweep axis.
    rtol : float
        The plateau tolerance in force.
    wind_speeds : Sequence[float] or None
        Per-point mean wind speeds, when known.

    Returns
    -------
    str
        A clause (with a leading separator) recommending the wind-speed axis for a
        rotor-speed axis, or an empty string for the wind-speed axis itself.
    """
    if parameter_name != _RPM_PARAMETER:
        return ""
    hint = (
        f"; use the wind-speed axis instead: parameter_name='{_WIND_PARAMETER}' "
        f"(CLI: --parameter {_CLI_PARAMETER[_WIND_PARAMETER]})"
    )
    if wind_speeds is None:
        return hint
    wind_valid = (
        all(math.isfinite(value) for value in wind_speeds)
        and _plateau_pair(wind_speeds, rtol) is None
    )
    if wind_valid:
        return hint + ", which is distinct across these operating points"
    return (
        hint + ", if the operating points carry distinct wind speeds (these do "
        f"not: {list(wind_speeds)})"
    )
