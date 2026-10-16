# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Cross-operating-point mode identification (global tracking).

Modes computed independently at successive operating points (rotor speeds or wind
speeds) are linked into tracks — the lines of a Campbell diagram. Both tracking
strategies score a candidate link by the same affinity (the modal-assurance
criterion tempered by a frequency-continuity penalty) and forbid links below the
MAC threshold; they differ only in how that affinity is resolved into tracks:

``"single_path"`` (the default) extracts each mode line as the globally optimal
continuity path through the whole sweep: it repeatedly finds the single
best-continuity path over the not-yet-assigned modes (a deterministic dynamic
program over the affinity) and removes it, until every mode is assigned. It reads
the strongest line off first.

``"simultaneous"`` instead resolves each operating-point transition at once: all
modes on one side are matched to all modes on the next by a single maximum-weight
assignment (the Hungarian algorithm, with both sides free to stay unmatched). No
early high-affinity link can then claim a mode that a different pairing would use to
reach a higher overall total — the failure mode of a per-line extraction in dense
spectra or at near-degenerate crossings, where an earlier high-scoring path can
strand a node a later path would have preferred.

Both are deterministic — no random restarts or preset cluster counts — and each
track carries a confidence (its weakest link's correlation) and an ambiguity flag
(raised where a mode had two near-equally correlated continuations, i.e. a crossing
or veering hotspot). Every track is given a physical label.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from itertools import pairwise
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.optimize import linear_sum_assignment

from vane.modal.labeler import (
    _WHIRL_SHIFT,
    DEFAULT_HEAVILY_DAMPED_THRESHOLD,
    ModeLabel,
    _rotating_frame_damping,
    label_solution,
)
from vane.modal.mac import compute_mac

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from vane.modal.branch import BranchGroup
    from vane.modal.eigensolver import ModalSolution

__all__ = [
    "DEFAULT_HEAVILY_DAMPED_THRESHOLD",
    "DEFAULT_RIGID_BODY_FREQUENCY_HZ",
    "LABEL_MISMATCH_PENALTY",
    "NEAR_COINCIDENT_RTOL",
    "ContinuityMetric",
    "IdentificationResult",
    "ModeTrack",
    "TrackingStrategy",
    "heavily_damped_modes",
    "identify_modes",
    "rotor_frame_damping_ratios",
]

_LOGGER = logging.getLogger(__name__)

DEFAULT_RIGID_BODY_FREQUENCY_HZ = 1e-3
"""Natural frequency (Hz) below which a mode is rigid-body and is not tracked.

A period of 1000 s is far longer than any restrained mode of a turbine: the softest
moored floating-platform surge and sway modes sit near 0.005-0.01 Hz. Below it lie
only modes without restoring stiffness, such as the free rotor azimuth.
"""

LABEL_MISMATCH_PENALTY = 0.5
"""Factor applied to the affinity of a link between two differently labelled modes.

When both ends of a candidate link carry an unambiguous physical label (see
:attr:`vane.modal.labeler.ModeLabel.is_ambiguous`) and the labels differ -- another
family, mode number or multi-blade type -- the link's affinity is multiplied by this
factor. A confident label change between neighbouring operating points is strong
evidence that the link joins two different physical modes; ambiguous (hybrid) modes,
as at a veering, are not penalised.
"""

NEAR_COINCIDENT_RTOL = 0.01
r"""Relative eigenvalue distance within which modes of one point are near-coincident.

Two modes whose eigenvalues satisfy :math:`|\lambda_i - \lambda_j| \le
\mathrm{rtol} \max(|\lambda_i|, |\lambda_j|)` are mixed by any coupling of
comparable size, so their individual eigenvectors are an arbitrary basis of the
pair's subspace (e.g. the collective and regressive second flap of the IEA 22-MW
turbine, 0.07 % apart at 10 m/s). Their MAC with a neighbouring point's mode then
measures the mixing, not the identity. For a link between two modes carrying the
same unambiguous label, the subspace MAC of each end against the other end's
near-coincident cluster is used when it is higher; the labels (assigned one per
multi-blade type by :func:`vane.modal.label_solution`) then decide which cluster
member continues which line.
"""

TrackingStrategy = Literal["single_path", "simultaneous"]
_STRATEGIES: frozenset[str] = frozenset(("single_path", "simultaneous"))

ContinuityMetric = Literal["natural_frequency", "eigenvalue"]
_METRICS: frozenset[str] = frozenset(("natural_frequency", "eigenvalue"))

_NEG_INF = -np.inf
# A value standing in for a forbidden (or never-preferred) link in the assignment
# cost matrix: large enough in magnitude that the optimizer always prefers leaving a
# mode unmatched over selecting it.
_FORBIDDEN_VALUE = -1e9
# A small reward so any threshold-passing link is preferred over breaking the
# track, even when the frequency penalty zeroes its weight (e.g. frequency_weight=1
# at the span extreme). It does not change the ranking among allowed links.
_LINK_BONUS = 1e-3


@dataclass
class ModeTrack:
    """A mode followed across operating points.

    Parameters
    ----------
    operating_points : list[int]
        Indices of the operating points this track spans, in order.
    mode_indices : list[int]
        The mode index within each corresponding operating point.
    natural_frequencies_hz : list[float]
        Natural frequency at each operating point along the track.
    damping_ratios : list[float]
        Damping ratio at each operating point along the track.
    label : ModeLabel
        Representative physical label: the label carrying the largest summed
        confidence over the track's operating points (its most confident instance),
        counting only unambiguous point labels when there are any.
    confidence : float
        The weakest link correlation along the track (minimum MAC between
        consecutive points, or the subspace MAC across a near-coincident pair), in
        ``[0, 1]``; ``1.0`` for a single-point track.
    is_ambiguous : bool
        Whether the track's identity is in doubt: a link was a near-tie (a mode with
        two near-equally correlated continuations, i.e. a crossing or veering
        hotspot), an unambiguous point label disagrees with the track label, every
        point label is ambiguous, or the track is a single operating point of a
        multi-point sweep (no continuity confirms it).
    is_heavily_damped : bool
        Whether the track links heavily damped (near-critical) modes, which are
        tracked in their own tier (see :func:`identify_modes`).
    label_agreement : float
        Fraction of the track's labelled operating points whose own label equals
        the track label, in ``[0, 1]``.
    point_labels : list[ModeLabel]
        The label of the track's mode at each of its operating points, aligned with
        ``operating_points``; empty when any of those solutions carries no DOF
        descriptions (the agreement and ambiguity still use the labelled ones).
    """

    operating_points: list[int]
    mode_indices: list[int]
    natural_frequencies_hz: list[float]
    damping_ratios: list[float]
    label: ModeLabel
    confidence: float = 1.0
    is_ambiguous: bool = False
    is_heavily_damped: bool = False
    label_agreement: float = 1.0
    point_labels: list[ModeLabel] = field(default_factory=list)


@dataclass
class IdentificationResult:
    """Result of cross-operating-point mode identification.

    Parameters
    ----------
    tracks : list[ModeTrack]
        The identified mode tracks, ordered by their first natural frequency.
    n_operating_points : int
        Number of operating points processed.
    branch_groups : list[BranchGroup]
        Crossing/veering hotspots flagged by the branch-review pass (empty unless
        ``branch_review`` was requested). See :mod:`vane.modal.branch`.
    heavily_damped_tracks : list[ModeTrack]
        Tracks of the heavily damped (near-critical) modes, linked among themselves
        only, ordered by first natural frequency. Empty when the heavily damped tier
        is disabled or no mode qualifies.
    heavily_damped_modes : list[list[int]]
        For each operating point, the indices of the modes classified heavily
        damped, so they are reported even where they form no track. Empty when the
        tier is disabled.
    """

    tracks: list[ModeTrack]
    n_operating_points: int
    branch_groups: list[BranchGroup] = field(default_factory=list)
    heavily_damped_tracks: list[ModeTrack] = field(default_factory=list)
    heavily_damped_modes: list[list[int]] = field(default_factory=list)

    @property
    def n_heavily_damped(self) -> int:
        """Return the number of heavily damped modes over all operating points."""
        return sum(len(modes) for modes in self.heavily_damped_modes)


def identify_modes(
    solutions: Sequence[ModalSolution],
    *,
    frequency_weight: float = 0.5,
    mac_threshold: float = 0.5,
    ambiguity_margin: float = 0.2,
    strategy: TrackingStrategy = "single_path",
    branch_review: bool = False,
    resolve: bool = False,
    parameter_values: Sequence[float] | None = None,
    rigid_body_frequency_hz: float = DEFAULT_RIGID_BODY_FREQUENCY_HZ,
    heavily_damped_threshold: float | None = DEFAULT_HEAVILY_DAMPED_THRESHOLD,
    rotor_speeds_rpm: Sequence[float] | None = None,
    continuity_metric: ContinuityMetric = "natural_frequency",
) -> IdentificationResult:
    r"""Link modes across operating points into labeled tracks, globally.

    Parameters
    ----------
    solutions : Sequence[ModalSolution]
        Modal solutions, one per operating point, in operating-point order. All
        must share the same number of mode-shape degrees of freedom.
    frequency_weight : float, optional
        Weight of the frequency-continuity penalty in the link affinity, ``[0, 1]``.
    mac_threshold : float, optional
        Minimum modal assurance criterion for two modes to be linkable.
    ambiguity_margin : float, optional
        A link is flagged ambiguous when the best continuation's MAC exceeds the
        second best by less than this margin.
    strategy : {"single_path", "simultaneous"}, optional
        How links are resolved into tracks. ``"single_path"`` (the default) extracts
        the globally optimal continuity path one line at a time; ``"simultaneous"``
        resolves each operating-point transition with one maximum-weight assignment,
        which is more reliable where lines cross or veer in a dense spectrum. The two
        agree on well-separated spectra and differ only at contested transitions; see
        the module docstring.
    branch_review : bool, optional
        When ``True``, run the branch-review pass and attach the detected
        crossing/veering hotspots to the result's ``branch_groups`` (the tracks
        themselves are unchanged). Default ``False``. See :mod:`vane.modal.branch`.
    resolve : bool, optional
        When ``True``, also attach the smooth (minimum-curvature) alternative to each
        eligible branch group (implies ``branch_review``; the tracks are still not
        modified). Default ``False``.
    parameter_values : Sequence[float] or None, optional
        The operating-parameter value at each operating point, used only by
        ``resolve`` as the curvature abscissa so smoothness is measured against the
        real (possibly non-uniform) sweep spacing. Defaults to the operating-point
        index (uniform spacing).
    rigid_body_frequency_hz : float, optional
        Modes whose undamped natural frequency is below this value (Hz) are
        rigid-body modes with no restoring stiffness and are left out of every
        track. Default :data:`DEFAULT_RIGID_BODY_FREQUENCY_HZ`; ``0.0`` tracks
        every mode.
    heavily_damped_threshold : float or None, optional
        Damping ratio, in ``(0, 1]``, at or above which a mode is heavily damped
        (see :func:`heavily_damped_modes`). Heavily damped modes are kept out of the
        main tracks and linked among themselves only, into
        ``heavily_damped_tracks``, so they can neither take over nor split a
        lightly damped line; every one is listed per operating point in
        ``heavily_damped_modes``. Default
        :data:`DEFAULT_HEAVILY_DAMPED_THRESHOLD`. ``None`` disables the tier and
        tracks every mode together.
    rotor_speeds_rpm : Sequence[float] or None, optional
        Rotor speed (rev/min) at each operating point. Needed to measure the damping
        of a regressive or progressive blade whirl mode in the rotating frame (see
        :func:`rotor_frame_damping_ratios`). ``None`` (the default) uses each
        solution's own ``rotor_speed_rpm`` (set by
        :func:`vane.modal.eigensolver.modes_from_mbc`), as
        :func:`vane.modal.label_solution` does; at a point whose rotor speed is
        unknown (``nan``) such modes are never classified heavily damped, because
        their fixed-frame damping ratio is shifted by the whirl and is not a
        property of the mode.
    continuity_metric : {"natural_frequency", "eigenvalue"}, optional
        The distance the continuity penalty measures between two modes of the main
        tracks. ``"natural_frequency"`` (the default) is the difference of undamped
        natural frequencies; ``"eigenvalue"`` is the distance
        :math:`|\lambda_i - \lambda_j| / 2\pi` in the complex plane, which equals
        it for lightly damped modes and also weighs a change of decay rate. The
        heavily damped tier always uses ``"eigenvalue"``, since a near-critical
        mode's natural frequency mostly measures its decay rate.

    Returns
    -------
    IdentificationResult
        The identified tracks and the number of operating points.

    Raises
    ------
    ValueError
        If ``frequency_weight``, ``mac_threshold``, or ``ambiguity_margin`` is not
        in ``[0, 1]``, ``rigid_body_frequency_hz`` is negative or non-finite,
        ``heavily_damped_threshold`` is not in ``(0, 1]``, ``rotor_speeds_rpm`` has
        the wrong length or non-finite values, ``strategy`` or
        ``continuity_metric`` is unknown, or the solutions have inconsistent
        mode-shape DOF layout.

    Notes
    -----
    The free rotor-azimuth (generator) DOF has no stiffness: its azimuth and speed
    states give a zero and an aerodynamic-damping eigenvalue, which the eigensolver
    normally returns as real and drops. Where the two nearly coincide they can come
    back as a complex pair of negligible frequency (periods of tens of minutes), so
    the mode would appear at isolated operating points only. The
    ``rigid_body_frequency_hz`` floor removes it everywhere.

    A blade mode whose aerodynamic damping exceeds critical in the rotating frame
    has two real rotating-frame eigenvalues :math:`-\sigma_1, -\sigma_2`. The MBC
    transform maps each cyclic pair of them to a fixed-frame pair
    :math:`-\sigma_k \pm i\Omega`: an oscillation at the rotor frequency whose
    natural frequency :math:`\sqrt{\sigma_k^2 + \Omega^2}` measures the decay rate
    rather than any vibration, while the collective pair is real and is dropped
    by the eigensolver. Tracking those images by natural frequency against the
    lightly damped whirl lines either side of the transition draws spurious lines;
    the heavily damped tier keeps them apart.
    """
    for name, value in (
        ("frequency_weight", frequency_weight),
        ("mac_threshold", mac_threshold),
        ("ambiguity_margin", ambiguity_margin),
    ):
        if not 0.0 <= value <= 1.0:
            msg = f"{name} must be in [0, 1], got {value}"
            raise ValueError(msg)
    if not (np.isfinite(rigid_body_frequency_hz) and rigid_body_frequency_hz >= 0.0):
        msg = (
            "rigid_body_frequency_hz must be a finite non-negative number, "
            f"got {rigid_body_frequency_hz}"
        )
        raise ValueError(msg)
    _validate_heavily_damped_threshold(heavily_damped_threshold)
    if strategy not in _STRATEGIES:
        msg = f"strategy must be one of {sorted(_STRATEGIES)}, got {strategy!r}"
        raise ValueError(msg)
    if continuity_metric not in _METRICS:
        msg = (
            f"continuity_metric must be one of {sorted(_METRICS)}, "
            f"got {continuity_metric!r}"
        )
        raise ValueError(msg)
    speeds = _validate_rotor_speeds(rotor_speeds_rpm, len(solutions))
    if not solutions:
        return IdentificationResult(tracks=[], n_operating_points=0)
    _validate_consistent_dofs(solutions)
    # Without explicit speeds the labels read each point's own rotor speed (the
    # label_solution default, nan when unknown); the heavily damped tier must read
    # the same, or the two would disagree on which whirl modes are heavily damped.
    point_speeds = (
        [float(solution.rotor_speed_rpm) for solution in solutions]
        if speeds is None
        else speeds
    )

    rigid = [
        np.asarray(solution.natural_frequencies_hz, dtype=np.float64)
        < rigid_body_frequency_hz
        for solution in solutions
    ]
    n_excluded = sum(int(np.count_nonzero(mask)) for mask in rigid)
    if n_excluded:
        _LOGGER.debug(
            "Leaving %d rigid-body mode(s) below %g Hz out of tracking",
            n_excluded,
            rigid_body_frequency_hz,
        )
    labels = _SolutionLabelCache(solutions, speeds)
    heavy = [
        _heavily_damped_mask(
            solution,
            heavily_damped_threshold,
            point_speeds[op] if math.isfinite(point_speeds[op]) else None,
            labels.multiblade(op),
        )
        & ~rigid[op]
        for op, solution in enumerate(solutions)
    ]
    macs = _pairwise_macs(solutions, labels)
    main_excluded = [rigid[op] | heavy[op] for op in range(len(solutions))]
    main = _track_tier(
        solutions,
        macs,
        main_excluded,
        frequency_weight=frequency_weight,
        mac_threshold=mac_threshold,
        strategy=strategy,
        metric=continuity_metric,
        labels=labels,
    )
    result = _build_result(
        main, solutions, macs, ambiguity_margin, labels, excluded=main_excluded
    )
    if branch_review or resolve:
        # Local import avoids a module-load cycle (branch imports this module).
        from vane.modal.branch import attach_branch_groups

        result = attach_branch_groups(
            result,
            solutions,
            resolve=resolve,
            parameter_values=parameter_values,
            ambiguity_margin=ambiguity_margin,
        )
    if heavily_damped_threshold is None:
        return result
    n_heavy = sum(int(np.count_nonzero(mask)) for mask in heavy)
    heavy_tracks: list[ModeTrack] = []
    if n_heavy:
        _LOGGER.info(
            "Tracking %d heavily damped mode(s) (damping ratio >= %g) apart from "
            "the lightly damped lines",
            n_heavy,
            heavily_damped_threshold,
        )
        heavy_excluded = [~mask for mask in heavy]
        entries = _track_tier(
            solutions,
            macs,
            heavy_excluded,
            frequency_weight=frequency_weight,
            mac_threshold=mac_threshold,
            strategy=strategy,
            metric="eigenvalue",
            labels=labels,
        )
        heavy_tracks = [
            replace(track, is_heavily_damped=True)
            for track in _build_result(
                entries,
                solutions,
                macs,
                ambiguity_margin,
                labels,
                excluded=heavy_excluded,
            ).tracks
        ]
    return replace(
        result,
        heavily_damped_tracks=heavy_tracks,
        heavily_damped_modes=[np.flatnonzero(mask).tolist() for mask in heavy],
    )


def rotor_frame_damping_ratios(
    solution: ModalSolution,
    rotor_speed_rpm: float,
    multiblade: Sequence[str | None],
) -> npt.NDArray[np.float64]:
    r"""Return each mode's damping ratio in the frame in which the mode is defined.

    Parameters
    ----------
    solution : ModalSolution
        The modal solution of one operating point (in multi-blade coordinates).
    rotor_speed_rpm : float
        The rotor speed (rev/min) at that operating point.
    multiblade : Sequence[str or None]
        The multi-blade type of each mode (``"collective"``, ``"regressive"``,
        ``"progressive"``, ``"cyclic"`` or ``None``), e.g. the ``multiblade`` of
        :func:`vane.modal.label_solution`.

    Returns
    -------
    npt.NDArray[np.float64]
        Shape ``(n_modes,)``. For a regressive or progressive whirl mode, the
        damping ratio of the rotating-frame eigenvalue it is the image of; for
        every other mode (collective, tower, platform, drivetrain, unlabelled), the
        fixed-frame damping ratio. A ``"cyclic"`` mode of undetermined whirl takes
        the smaller of the two whirl readings.

    Raises
    ------
    ValueError
        If ``multiblade`` does not have one entry per mode or ``rotor_speed_rpm``
        is not finite.

    Notes
    -----
    The MBC transform shifts a cyclic blade mode's rotating-frame eigenvalue
    :math:`\lambda_r = -\sigma + i\omega_r` by the rotor frequency :math:`\Omega`:
    the progressive whirl appears at :math:`\omega_r + \Omega` and the regressive
    at :math:`|\omega_r - \Omega|`, with the same decay rate :math:`\sigma`. The
    fixed-frame damping ratio :math:`\sigma / |\lambda|` therefore differs between
    the two whirls of one blade mode and rises towards 1 on the regressive branch
    as :math:`\omega_r` approaches :math:`\Omega`, without the blade mode itself
    being any more damped. Undoing the shift, :math:`\omega_r = |\omega \mp
    \Omega|`, gives the damping ratio :math:`\sigma / \sqrt{\sigma^2 +
    \omega_r^2}` of the blade mode, which is the same for its regressive,
    collective and progressive members.
    """
    if len(multiblade) != solution.n_modes:
        msg = (
            f"multiblade has {len(multiblade)} entries but the solution has "
            f"{solution.n_modes} modes"
        )
        raise ValueError(msg)
    if not math.isfinite(rotor_speed_rpm):
        msg = f"rotor_speed_rpm must be finite, got {rotor_speed_rpm}"
        raise ValueError(msg)
    return _rotating_frame_damping(
        np.asarray(solution.eigenvalues, dtype=np.complex128),
        np.asarray(solution.damping_ratios, dtype=np.float64),
        rotor_speed_rpm,
        multiblade,
    )


def heavily_damped_modes(
    solution: ModalSolution,
    *,
    threshold: float = DEFAULT_HEAVILY_DAMPED_THRESHOLD,
    rotor_speed_rpm: float | None = None,
) -> npt.NDArray[np.bool_]:
    """Return which modes of one operating point are heavily damped.

    A mode is heavily damped (near-critical) when its damping ratio, measured in
    the frame in which the mode is defined (:func:`rotor_frame_damping_ratios`),
    is at least ``threshold``. Such a mode barely oscillates: its damped frequency
    is a small fraction of its natural frequency, which mostly measures the decay
    rate, and its shape correlates weakly with its neighbours across operating
    points. These are oscillatory (complex) modes; the eigensolver's
    ``n_overdamped`` instead counts the real, non-oscillatory eigenvalues, which
    are never returned as modes.

    Parameters
    ----------
    solution : ModalSolution
        The modal solution of one operating point.
    threshold : float, optional
        Damping ratio in ``(0, 1]`` at or above which a mode is heavily damped.
        Default :data:`DEFAULT_HEAVILY_DAMPED_THRESHOLD`.
    rotor_speed_rpm : float or None, optional
        The rotor speed (rev/min) at the operating point. ``None`` (the default) uses
        ``solution.rotor_speed_rpm``, as :func:`vane.modal.label_solution` does. When
        neither is known, regressive and progressive blade whirl modes are never
        classified, since their fixed-frame damping ratio is shifted by the whirl.

    Returns
    -------
    npt.NDArray[np.bool_]
        Shape ``(n_modes,)``; ``True`` for each heavily damped mode.

    Raises
    ------
    ValueError
        If ``threshold`` is not in ``(0, 1]`` or ``rotor_speed_rpm`` is not finite.
    """
    _validate_heavily_damped_threshold(threshold)
    if rotor_speed_rpm is not None and not math.isfinite(rotor_speed_rpm):
        msg = f"rotor_speed_rpm must be finite, got {rotor_speed_rpm}"
        raise ValueError(msg)
    speed = (
        float(solution.rotor_speed_rpm) if rotor_speed_rpm is None else rotor_speed_rpm
    )
    multiblade = (
        [label.multiblade for label in label_solution(solution, rotor_speed_rpm=speed)]
        if solution.dof_descriptions
        else [None] * solution.n_modes
    )
    return _heavily_damped_mask(
        solution, threshold, speed if math.isfinite(speed) else None, multiblade
    )


def _heavily_damped_mask(
    solution: ModalSolution,
    threshold: float | None,
    rotor_speed_rpm: float | None,
    multiblade: Sequence[str | None],
) -> npt.NDArray[np.bool_]:
    """Classify one point's modes as heavily damped (all ``False`` if disabled)."""
    if threshold is None:
        return np.zeros(solution.n_modes, dtype=np.bool_)
    if rotor_speed_rpm is None:
        ratios = np.asarray(solution.damping_ratios, dtype=np.float64)
        # Without the rotor speed a whirl mode's damping cannot be measured in its
        # own frame, so it is never classified.
        whirl = np.array(
            [entry in _WHIRL_SHIFT for entry in multiblade], dtype=np.bool_
        )
        return (ratios >= threshold) & ~whirl
    ratios = rotor_frame_damping_ratios(solution, rotor_speed_rpm, multiblade)
    return np.asarray(ratios >= threshold, dtype=np.bool_)


def _validate_heavily_damped_threshold(threshold: float | None) -> None:
    """Reject a heavily damped threshold outside ``(0, 1]``."""
    if threshold is None:
        return
    if not (math.isfinite(threshold) and 0.0 < threshold <= 1.0):
        msg = f"heavily_damped_threshold must be in (0, 1] or None, got {threshold}"
        raise ValueError(msg)


def _validate_rotor_speeds(
    rotor_speeds_rpm: Sequence[float] | None, n_points: int
) -> list[float] | None:
    """Return the rotor speeds as floats after checking length and finiteness."""
    if rotor_speeds_rpm is None:
        return None
    speeds = [float(value) for value in rotor_speeds_rpm]
    if len(speeds) != n_points:
        msg = (
            f"rotor_speeds_rpm has {len(speeds)} entries but there are "
            f"{n_points} operating points"
        )
        raise ValueError(msg)
    if not all(math.isfinite(value) for value in speeds):
        msg = f"rotor_speeds_rpm must all be finite, got {speeds}"
        raise ValueError(msg)
    return speeds


def _track_tier(
    solutions: Sequence[ModalSolution],
    macs: list[npt.NDArray[np.float64]],
    excluded: list[npt.NDArray[np.bool_]],
    *,
    frequency_weight: float,
    mac_threshold: float,
    strategy: TrackingStrategy,
    metric: ContinuityMetric,
    labels: _SolutionLabelCache | None = None,
) -> list[list[tuple[int, int]]]:
    """Track the modes not in ``excluded`` among themselves; return their paths."""
    affinities = _pairwise_affinities(
        solutions, macs, frequency_weight, mac_threshold, metric, labels
    )
    if strategy == "simultaneous":
        return _extract_tracks_simultaneous(solutions, affinities, excluded)
    return _extract_tracks(solutions, affinities, excluded)


def _pairwise_macs(
    solutions: Sequence[ModalSolution],
    labels: _SolutionLabelCache | None = None,
) -> list[npt.NDArray[np.float64]]:
    """Return the link MAC matrix between each consecutive operating-point pair.

    It is the MAC, raised to the subspace MAC across a near-coincident cluster
    (:data:`NEAR_COINCIDENT_RTOL`) where both ends carry the same unambiguous label.
    """
    macs = []
    for k in range(len(solutions) - 1):
        mac = compute_mac(solutions[k].mode_shapes, solutions[k + 1].mode_shapes)
        if labels is not None:
            mac = _with_cluster_macs(solutions[k], solutions[k + 1], mac, labels, k)
        macs.append(mac)
    return macs


def _near_coincident_clusters(solution: ModalSolution) -> list[list[int]]:
    """Group a point's modes whose eigenvalues lie within NEAR_COINCIDENT_RTOL."""
    lam = np.asarray(solution.eigenvalues, dtype=np.complex128)
    lam = lam.real + 1j * np.abs(lam.imag)
    order = np.argsort(np.abs(lam), kind="stable").tolist()
    parent = list(range(lam.shape[0]))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    # Sorted by modulus, near-coincident modes are close in the order; comparing
    # each mode with the following ones until the modulus gap exceeds the
    # tolerance keeps this near-linear on spectra with hundreds of modes.
    for position, first in enumerate(order):
        for second in order[position + 1 :]:
            scale = max(abs(lam[first]), abs(lam[second]))
            if abs(lam[second]) - abs(lam[first]) > NEAR_COINCIDENT_RTOL * scale:
                break
            if abs(lam[first] - lam[second]) <= NEAR_COINCIDENT_RTOL * scale:
                parent[find(second)] = find(first)
    groups: dict[int, list[int]] = {}
    for index in range(lam.shape[0]):
        groups.setdefault(find(index), []).append(index)
    return [members for members in groups.values() if len(members) > 1]


def _subspace_mac(
    basis: npt.NDArray[np.complex128], vectors: npt.NDArray[np.complex128]
) -> npt.NDArray[np.float64]:
    """Return the share of each column of ``vectors`` lying in span(``basis``)."""
    q, _ = np.linalg.qr(basis)
    projected = np.sum(np.abs(np.conj(q).T @ vectors) ** 2, axis=0)
    norms = np.sum(np.abs(vectors) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(norms > 0.0, projected / norms, 0.0)
    return np.asarray(np.clip(share, 0.0, 1.0), dtype=np.float64)


def _with_cluster_macs(
    first: ModalSolution,
    second: ModalSolution,
    mac: npt.NDArray[np.float64],
    labels: _SolutionLabelCache,
    op: int,
) -> npt.NDArray[np.float64]:
    """Raise same-label links across near-coincident clusters to the subspace MAC."""
    if not (labels.has_descriptions(op) and labels.has_descriptions(op + 1)):
        return mac
    result: npt.NDArray[np.float64] = np.array(mac, dtype=np.float64, copy=True)
    for a_side, cluster_solution, other_solution in (
        (True, first, second),
        (False, second, first),
    ):
        own_op, other_op = (op, op + 1) if a_side else (op + 1, op)
        other_keys: list[str | None] | None = None
        for cluster in _near_coincident_clusters(cluster_solution):
            keys = {m: _link_key(labels.label(own_op, m)) for m in cluster}
            if not any(keys.values()):
                continue
            if other_keys is None:
                other_keys = [
                    _link_key(labels.label(other_op, m))
                    for m in range(other_solution.n_modes)
                ]
            shares = _subspace_mac(
                cluster_solution.mode_shapes[:, cluster], other_solution.mode_shapes
            )
            for member in cluster:
                if keys[member] is None:
                    continue
                for other, other_key in enumerate(other_keys):
                    if other_key != keys[member]:
                        continue
                    row, col = (member, other) if a_side else (other, member)
                    result[row, col] = max(result[row, col], float(shares[other]))
    return result


def _link_key(label: ModeLabel) -> str | None:
    """Return the identity a label asserts for linking, or None if it asserts none.

    Ambiguous and unidentified labels assert no identity.
    """
    if label.is_ambiguous or label.label.startswith("Unidentified"):
        return None
    return label.label


def _pairwise_affinities(
    solutions: Sequence[ModalSolution],
    macs: list[npt.NDArray[np.float64]],
    frequency_weight: float,
    mac_threshold: float,
    metric: ContinuityMetric = "natural_frequency",
    labels: _SolutionLabelCache | None = None,
) -> list[npt.NDArray[np.float64]]:
    """Return the link affinity (MAC x frequency continuity x label match) per pair.

    Entries below ``mac_threshold`` are set to ``-inf`` so they cannot be linked.
    The continuity distance is the natural-frequency difference, or with
    ``metric="eigenvalue"`` the complex-plane eigenvalue distance in Hz; either is
    normalised by the span of natural frequencies at the two points. A link between
    two modes with different unambiguous labels is scaled by
    :data:`LABEL_MISMATCH_PENALTY`.
    """
    affinities: list[npt.NDArray[np.float64]] = []
    for k, mac in enumerate(macs):
        weight = mac.copy()
        freqs_a = solutions[k].natural_frequencies_hz
        freqs_b = solutions[k + 1].natural_frequencies_hz
        if frequency_weight > 0.0 and freqs_a.size and freqs_b.size:
            both = np.concatenate([freqs_a, freqs_b])
            span = float(both.max() - both.min())
            if span > 0.0:
                delta = _continuity_distance(solutions[k], solutions[k + 1], metric)
                penalty = np.clip(1.0 - frequency_weight * delta / span, 0.0, 1.0)
                weight = mac * penalty
        if labels is not None:
            weight = weight * _label_match_factors(labels, k, mac.shape)
        affinities.append(
            np.where(mac >= mac_threshold, weight + _LINK_BONUS, _NEG_INF)
        )
    return affinities


def _label_match_factors(
    labels: _SolutionLabelCache, op: int, shape: tuple[int, ...]
) -> npt.NDArray[np.float64]:
    """Return LABEL_MISMATCH_PENALTY where two unambiguous labels differ, else 1."""
    factors = np.ones(shape, dtype=np.float64)
    if not (labels.has_descriptions(op) and labels.has_descriptions(op + 1)):
        return factors
    codes: dict[str, int] = {}

    def encode(key: str | None) -> int:
        return -1 if key is None else codes.setdefault(key, len(codes))

    first = np.array(
        [encode(_link_key(labels.label(op, m))) for m in range(shape[0])], dtype=int
    )
    second = np.array(
        [encode(_link_key(labels.label(op + 1, m))) for m in range(shape[1])],
        dtype=int,
    )
    mismatch = (
        (first[:, np.newaxis] >= 0)
        & (second[np.newaxis, :] >= 0)
        & (first[:, np.newaxis] != second[np.newaxis, :])
    )
    factors[mismatch] = LABEL_MISMATCH_PENALTY
    return factors


def _continuity_distance(
    first: ModalSolution, second: ModalSolution, metric: ContinuityMetric
) -> npt.NDArray[np.float64]:
    """Return the pairwise continuity distance (Hz) between two points' modes."""
    if metric == "eigenvalue":
        lam_a = np.asarray(first.eigenvalues, dtype=np.complex128)
        lam_b = np.asarray(second.eigenvalues, dtype=np.complex128)
        # Compare the upper-half-plane member of each conjugate pair.
        lam_a = lam_a.real + 1j * np.abs(lam_a.imag)
        lam_b = lam_b.real + 1j * np.abs(lam_b.imag)
        return np.asarray(
            np.abs(lam_a[:, np.newaxis] - lam_b[np.newaxis, :]) / (2.0 * np.pi),
            dtype=np.float64,
        )
    return np.asarray(
        np.abs(
            first.natural_frequencies_hz[:, np.newaxis]
            - second.natural_frequencies_hz[np.newaxis, :]
        ),
        dtype=np.float64,
    )


def _extract_tracks(
    solutions: Sequence[ModalSolution],
    affinities: list[npt.NDArray[np.float64]],
    excluded: list[npt.NDArray[np.bool_]],
) -> list[list[tuple[int, int]]]:
    """Greedily extract globally-optimal continuity paths until all modes used.

    Modes flagged in ``excluded`` start out used, so no path starts at, enters, or
    leaves them.
    """
    used = [np.array(mask, dtype=np.bool_, copy=True) for mask in excluded]
    tracks: list[list[tuple[int, int]]] = []
    while True:
        path = _best_path(used, affinities)
        if path is None:
            return tracks
        for op, mode in path:
            used[op][mode] = True
        tracks.append(path)


def _extract_tracks_simultaneous(
    solutions: Sequence[ModalSolution],
    affinities: list[npt.NDArray[np.float64]],
    excluded: list[npt.NDArray[np.bool_]],
) -> list[list[tuple[int, int]]]:
    """Link modes by a simultaneous maximum-weight assignment at each transition.

    At every operating-point transition all modes are matched at once
    (:func:`_max_weight_matching`), so no early high-affinity link can strand a mode
    that a different assignment would have used. A mode left unmatched ends its
    track; a mode on the later side that receives no link begins a new one. Modes
    flagged in ``excluded`` are never linked and begin no track.
    """
    tracks: list[list[tuple[int, int]]] = []
    # head_track[mode] -> index in ``tracks`` of the active track ending at this mode
    # of the current operating point.
    head_track: dict[int, int] = {}
    for mode in np.flatnonzero(~excluded[0]).tolist():
        tracks.append([(0, mode)])
        head_track[mode] = len(tracks) - 1
    for k, affinity in enumerate(affinities):
        allowed = affinity.copy()
        allowed[excluded[k], :] = _NEG_INF
        allowed[:, excluded[k + 1]] = _NEG_INF
        next_head: dict[int, int] = {}
        for row, col in _max_weight_matching(allowed):
            track_index = head_track[row]
            tracks[track_index].append((k + 1, col))
            next_head[col] = track_index
        for mode in range(solutions[k + 1].n_modes):
            if mode not in next_head and not bool(excluded[k + 1][mode]):
                tracks.append([(k + 1, mode)])
                next_head[mode] = len(tracks) - 1
        head_track = next_head
    return tracks


def _max_weight_matching(
    affinity: npt.NDArray[np.float64],
) -> list[tuple[int, int]]:
    """Return the maximum-total-affinity one-to-one matching, allowing non-matches.

    Solves the balanced assignment between the rows and columns of ``affinity``
    (modes at two consecutive operating points) so the total linked affinity is
    globally maximal, with every mode free to remain unmatched. Links with
    non-finite affinity (below the MAC threshold) are never selected. Returns the
    matched ``(row, column)`` pairs that are joined by a finite-affinity link.

    The problem is padded to a square cost matrix whose off-diagonal blocks offer
    each mode a zero-value "stay unmatched" option, so a real link is taken only when
    its affinity beats opting out and a forbidden link never is. Unlike a greedy
    best-link-first match, no early high-affinity pairing can claim a mode that a
    different pairing would use to reach a higher overall total.
    """
    n_rows, n_cols = affinity.shape
    if n_rows == 0 or n_cols == 0:
        return []
    size = n_rows + n_cols
    value = np.zeros((size, size), dtype=np.float64)
    value[:n_rows, :n_cols] = np.where(
        np.isfinite(affinity), affinity, _FORBIDDEN_VALUE
    )
    # Row opt-out: each row may pair with its own dummy column at value 0.
    value[:n_rows, n_cols:] = _FORBIDDEN_VALUE
    value[np.arange(n_rows), n_cols + np.arange(n_rows)] = 0.0
    # Column opt-out: each column may pair with its own dummy row at value 0.
    value[n_rows:, :n_cols] = _FORBIDDEN_VALUE
    value[n_rows + np.arange(n_cols), np.arange(n_cols)] = 0.0
    rows, cols = linear_sum_assignment(value, maximize=True)
    return [
        (int(row), int(col))
        for row, col in zip(rows, cols, strict=True)
        if row < n_rows and col < n_cols and np.isfinite(affinity[row, col])
    ]


def _best_path(
    used: list[npt.NDArray[np.bool_]],
    affinities: list[npt.NDArray[np.float64]],
) -> list[tuple[int, int]] | None:
    """Return the highest-affinity contiguous path over unused modes, or None.

    A dynamic program over the layered graph (operating points as layers, modes as
    nodes): a path may start at any operating point and is extended only along
    links whose affinity is finite (MAC above threshold).
    """
    n_op = len(used)
    best: list[npt.NDArray[np.float64]] = []
    parent: list[npt.NDArray[np.int_]] = []
    for op in range(n_op):
        start = np.where(used[op], _NEG_INF, 0.0)
        link_parent = np.full(used[op].shape[0], -1, dtype=np.int_)
        if op > 0 and best[op - 1].size and start.size:
            candidate = best[op - 1][:, np.newaxis] + affinities[op - 1]
            from_prev = candidate.max(axis=0) if candidate.size else start
            best_i = candidate.argmax(axis=0) if candidate.size else link_parent
            # A used target mode can be neither started nor entered, so check the
            # extension on the target being available -- otherwise a later path
            # could route through an already-assigned (op, mode).
            take = (from_prev > start) & ~used[op]
            start = np.where(take, from_prev, start)
            link_parent = np.where(take, best_i, link_parent)
        best.append(start)
        parent.append(link_parent)

    end_op, end_mode, top = -1, -1, _NEG_INF
    for op in range(n_op):
        if best[op].size and float(best[op].max()) > top:
            top = float(best[op].max())
            end_op, end_mode = op, int(best[op].argmax())
    if top == _NEG_INF:
        return None

    path: list[tuple[int, int]] = []
    op, mode = end_op, end_mode
    while mode != -1:
        path.append((op, mode))
        previous = int(parent[op][mode])
        op, mode = op - 1, previous
    path.reverse()
    return path


def _build_result(
    tracks: list[list[tuple[int, int]]],
    solutions: Sequence[ModalSolution],
    macs: list[npt.NDArray[np.float64]],
    ambiguity_margin: float,
    solution_labels: _SolutionLabelCache | None = None,
    *,
    excluded: list[npt.NDArray[np.bool_]] | None = None,
) -> IdentificationResult:
    """Assemble ModeTrack objects with confidence and ambiguity, sorted by freq.

    ``excluded`` marks, per operating point, the modes outside the tier being built;
    they cannot continue a track, so they never make a link a near-tie.
    """
    # Label each operating point's modes once; a track only picks among them.
    if solution_labels is None:
        solution_labels = _SolutionLabelCache(solutions)
    built: list[ModeTrack] = []
    for entries in tracks:
        freqs = [float(solutions[op].natural_frequencies_hz[m]) for op, m in entries]
        damps = [float(solutions[op].damping_ratios[m]) for op, m in entries]
        confidence, ambiguous = _track_quality(
            entries, macs, ambiguity_margin, excluded
        )
        label = _track_label(entries, solution_labels)
        labelled = [
            solution_labels.label(op, m)
            for op, m in entries
            if solution_labels.has_descriptions(op)
        ]
        agreement = (
            sum(point.label == label.label for point in labelled) / len(labelled)
            if labelled
            else 1.0
        )
        contradicted = any(
            point.label != label.label and not point.is_ambiguous for point in labelled
        )
        # One label per operating point or none: a list compressed over the points
        # without descriptions would pair later labels with the wrong points.
        point_labels = labelled if len(labelled) == len(entries) else []
        unconfirmed = len(entries) == 1 and len(solutions) > 1
        built.append(
            ModeTrack(
                operating_points=[op for op, _ in entries],
                mode_indices=[m for _, m in entries],
                natural_frequencies_hz=freqs,
                damping_ratios=damps,
                label=label,
                confidence=confidence,
                is_ambiguous=(
                    ambiguous or contradicted or unconfirmed or label.is_ambiguous
                ),
                label_agreement=agreement,
                point_labels=point_labels,
            )
        )
    built.sort(key=lambda track: track.natural_frequencies_hz[0])
    return IdentificationResult(tracks=built, n_operating_points=len(solutions))


def _track_quality(
    entries: list[tuple[int, int]],
    macs: list[npt.NDArray[np.float64]],
    ambiguity_margin: float,
    excluded: list[npt.NDArray[np.bool_]] | None = None,
) -> tuple[float, bool]:
    """Return a track's weakest-link MAC and whether any link is a near-tie.

    The competing continuations of a link are the modes of the next point that the
    tier could link to; modes in ``excluded`` (rigid-body modes, or the other tier's
    modes) are not candidates and are ignored.
    """
    confidence = 1.0
    ambiguous = False
    for (op_a, mode_a), (op_b, mode_b) in pairwise(entries):
        mac_row = macs[op_a][mode_a]
        confidence = min(confidence, float(mac_row[mode_b]))
        if excluded is not None:
            mac_row = mac_row[~excluded[op_b]]
        if mac_row.shape[0] >= 2:
            ordered = np.sort(mac_row)[::-1]
            if float(ordered[0] - ordered[1]) < ambiguity_margin:
                ambiguous = True
    return confidence, ambiguous


class _SolutionLabelCache:
    """Label each operating point's modes at most once across all tracks."""

    def __init__(
        self,
        solutions: Sequence[ModalSolution],
        rotor_speeds_rpm: Sequence[float] | None = None,
    ) -> None:
        self._solutions = solutions
        self._speeds = rotor_speeds_rpm
        self._labels: dict[int, list[ModeLabel]] = {}

    def has_descriptions(self, op: int) -> bool:
        """Return whether operating point ``op`` carries DOF descriptions."""
        return bool(self._solutions[op].dof_descriptions)

    def _point(self, op: int) -> list[ModeLabel]:
        if op not in self._labels:
            solution = self._solutions[op]
            self._labels[op] = (
                label_solution(solution)
                if self._speeds is None
                else label_solution(solution, rotor_speed_rpm=self._speeds[op])
            )
        return self._labels[op]

    def label(self, op: int, mode: int) -> ModeLabel:
        """Return the (cached) label of ``mode`` at operating point ``op``."""
        return self._point(op)[mode]

    def multiblade(self, op: int) -> list[str | None]:
        """Return the multi-blade type of every mode at operating point ``op``."""
        if not self.has_descriptions(op):
            return [None] * self._solutions[op].n_modes
        return [label.multiblade for label in self._point(op)]


def _track_label(
    entries: list[tuple[int, int]], labels: _SolutionLabelCache
) -> ModeLabel:
    """Return the label a track's modes vote for, weighted by their confidence.

    Each mode is labelled by :func:`vane.modal.label_solution` (scale-invariant
    biorthogonal participation, with the multi-blade type), once per operating point
    for the whole result. Every distinct label (name plus multi-blade type) scores
    the summed confidence of the modes carrying it, and the highest score wins; its
    most confident instance represents the track. A label that holds at most
    operating points therefore names the track even where one point's mode is mixed
    (e.g. near a crossing, or a degenerate point without a whirl suffix), instead
    of whichever single point happened to be the most confident. Only unambiguous
    labels vote when the track has any: a hybrid point (see
    :attr:`vane.modal.labeler.ModeLabel.is_ambiguous`) does not name a line that
    is clearly identified elsewhere.
    """
    candidates: list[ModeLabel] = []
    for op, mode in entries:
        if not labels.has_descriptions(op):
            continue
        candidates.append(labels.label(op, mode))
    if not candidates:
        from vane.config.dof_map import DofCategory

        return ModeLabel(DofCategory.UNKNOWN, "Unidentified", 0.0, [])
    decided = [candidate for candidate in candidates if not candidate.is_ambiguous]
    if decided:
        candidates = decided
    scores: dict[str, float] = {}
    for candidate in candidates:
        scores[candidate.label] = scores.get(candidate.label, 0.0) + max(
            candidate.confidence, 0.0
        )
    best = max(scores.values())
    # Ties resolve to the label of the most confident mode among the tied labels.
    tied = [candidate for candidate in candidates if scores[candidate.label] == best]
    return max(tied, key=lambda label: label.confidence)


def _validate_consistent_dofs(solutions: Sequence[ModalSolution]) -> None:
    """Validate that every solution shares the reference mode-shape DOF layout.

    The DOF count must match, and when descriptions are available they must match
    too, so the modal assurance criterion compares corresponding degrees of
    freedom rather than silently mixing unrelated rows.
    """
    reference = solutions[0]
    ref_n = reference.mode_shapes.shape[0]
    ref_descriptions = reference.dof_descriptions
    for index, solution in enumerate(solutions[1:], start=1):
        if solution.mode_shapes.shape[0] != ref_n:
            msg = (
                f"Operating point {index} has {solution.mode_shapes.shape[0]} "
                f"mode-shape DOFs, expected {ref_n}"
            )
            raise ValueError(msg)
        if (
            ref_descriptions
            and solution.dof_descriptions
            and solution.dof_descriptions != ref_descriptions
        ):
            msg = (
                f"Operating point {index} has different DOF descriptions from the "
                f"reference; mode shapes are not comparable"
            )
            raise ValueError(msg)
