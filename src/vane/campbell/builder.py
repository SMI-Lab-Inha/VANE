# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Assembly of Campbell-diagram data from tracked modes.

A Campbell diagram plots modal natural frequency (and damping) against an operating
parameter — rotor speed or wind speed — with one curve per tracked mode. This
module pairs the mode tracks from :func:`vane.modal.identify_modes` with the
operating-parameter value of each operating point, producing a structure ready for
plotting or resonance analysis.

Heavily damped (near-critical) modes, which :func:`vane.modal.identify_modes` tracks
in a tier of their own, are carried separately in
:attr:`CampbellDiagram.heavily_damped_tracks`: they are drawn distinctly (or left
out) and never enter the resonance analysis, since a mode damped near critically
cannot resonate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from vane.modal.identifier import DEFAULT_HEAVILY_DAMPED_THRESHOLD, identify_modes

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from vane.modal.eigensolver import ModalSolution
    from vane.modal.identifier import IdentificationResult, ModeTrack

__all__ = ["CampbellDiagram", "build_campbell", "campbell_from_solutions"]


@dataclass
class CampbellDiagram:
    """Campbell-diagram data: tracked modes against an operating parameter.

    Parameters
    ----------
    parameter_values : npt.NDArray[np.float64]
        Operating-parameter value (e.g. rotor speed in rev/min) at each operating
        point, indexed by operating-point number, shape ``(n_operating_points,)``.
    parameter_name : str
        Name of the operating parameter (e.g. ``"rotor_speed_rpm"``).
    tracks : list[ModeTrack]
        The tracked modes, ordered by first natural frequency.
    n_operating_points : int
        Number of operating points.
    heavily_damped_tracks : list[ModeTrack]
        Tracks of heavily damped (near-critical) modes, kept apart from ``tracks``:
        plotted distinctly and left out of resonance detection. Empty when the
        identification had none or they were not included.
    """

    parameter_values: npt.NDArray[np.float64]
    parameter_name: str
    tracks: list[ModeTrack]
    n_operating_points: int
    heavily_damped_tracks: list[ModeTrack] = field(default_factory=list)

    def track_curve(
        self, track: ModeTrack
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        """Return the operating-parameter, frequency, and damping arrays of a track.

        Parameters
        ----------
        track : ModeTrack
            A track belonging to this diagram.

        Returns
        -------
        tuple of npt.NDArray[np.float64]
            ``(parameter_values, natural_frequencies_hz, damping_ratios)`` along the
            track, each of length equal to the track's span.
        """
        x = self.parameter_values[track.operating_points]
        return (
            x,
            np.asarray(track.natural_frequencies_hz, dtype=np.float64),
            np.asarray(track.damping_ratios, dtype=np.float64),
        )


def build_campbell(
    result: IdentificationResult,
    parameter_values: Sequence[float],
    *,
    parameter_name: str = "rotor_speed_rpm",
    include_heavily_damped: bool = True,
) -> CampbellDiagram:
    """Build a Campbell diagram from identified mode tracks.

    Parameters
    ----------
    result : IdentificationResult
        The cross-operating-point mode tracks.
    parameter_values : Sequence[float]
        Operating-parameter value at each operating point, in operating-point
        order; its length must equal ``result.n_operating_points``.
    parameter_name : str, optional
        Name of the operating parameter (default ``"rotor_speed_rpm"``).
    include_heavily_damped : bool, optional
        Carry the identification's heavily damped tracks into the diagram's
        ``heavily_damped_tracks`` (default ``True``); ``False`` omits them.

    Returns
    -------
    CampbellDiagram
        The assembled diagram.

    Raises
    ------
    ValueError
        If ``parameter_values`` length does not match the number of operating points,
        or the values are not a finite, one-dimensional, strictly increasing sweep axis
        (the excitation/resonance analysis interpolates against this axis, so duplicate,
        non-monotone, or non-finite values would give wrong crossings or raw errors).
    """
    if len(parameter_values) != result.n_operating_points:
        msg = (
            f"parameter_values has {len(parameter_values)} entries but there are "
            f"{result.n_operating_points} operating points"
        )
        raise ValueError(msg)
    values = np.asarray(parameter_values, dtype=np.float64)
    _validate_sweep_axis(values, parameter_name)
    return CampbellDiagram(
        parameter_values=values,
        parameter_name=parameter_name,
        tracks=result.tracks,
        n_operating_points=result.n_operating_points,
        heavily_damped_tracks=(
            list(result.heavily_damped_tracks) if include_heavily_damped else []
        ),
    )


def _validate_sweep_axis(values: npt.NDArray[np.float64], parameter_name: str) -> None:
    """Validate the Campbell sweep axis: finite, 1-D, and strictly increasing.

    The excitation lines and resonance crossings interpolate against this axis, so a
    non-finite, multi-dimensional, duplicate, or non-monotone value would silently
    produce a wrong diagram or a raw NumPy error. An empty or single-point sweep is
    trivially valid.
    """
    if values.ndim != 1:
        msg = (
            f"{parameter_name} values must be one-dimensional, got shape {values.shape}"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(values)):
        msg = f"{parameter_name} values must all be finite; got {values.tolist()}"
        raise ValueError(msg)
    if values.size > 1 and not np.all(np.diff(values) > 0.0):
        msg = (
            f"{parameter_name} values must be strictly increasing (a monotone sweep "
            f"axis with no duplicates); got {values.tolist()}"
        )
        raise ValueError(msg)


def campbell_from_solutions(
    solutions: Sequence[ModalSolution],
    parameter_values: Sequence[float],
    *,
    parameter_name: str = "rotor_speed_rpm",
    frequency_weight: float = 0.5,
    mac_threshold: float = 0.5,
    heavily_damped_threshold: float | None = DEFAULT_HEAVILY_DAMPED_THRESHOLD,
    rotor_speeds_rpm: Sequence[float] | None = None,
) -> CampbellDiagram:
    """Identify mode tracks from solutions and build a Campbell diagram.

    Parameters
    ----------
    solutions : Sequence[ModalSolution]
        Modal solutions, one per operating point, in operating-point order.
    parameter_values : Sequence[float]
        Operating-parameter value at each operating point.
    parameter_name : str, optional
        Name of the operating parameter.
    frequency_weight, mac_threshold : float, optional
        Passed through to :func:`vane.modal.identify_modes`.
    heavily_damped_threshold : float or None, optional
        Passed through to :func:`vane.modal.identify_modes`; ``None`` tracks every
        mode together.
    rotor_speeds_rpm : Sequence[float] or None, optional
        Rotor speed (rev/min) at each operating point, so blade whirl modes are
        classified by their rotating-frame damping. Defaults to
        ``parameter_values`` when ``parameter_name`` is ``"rotor_speed_rpm"``.

    Returns
    -------
    CampbellDiagram
        The assembled diagram.
    """
    if rotor_speeds_rpm is None and parameter_name == "rotor_speed_rpm":
        rotor_speeds_rpm = parameter_values
    result = identify_modes(
        solutions,
        frequency_weight=frequency_weight,
        mac_threshold=mac_threshold,
        heavily_damped_threshold=heavily_damped_threshold,
        rotor_speeds_rpm=rotor_speeds_rpm,
    )
    return build_campbell(result, parameter_values, parameter_name=parameter_name)
