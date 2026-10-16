# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Branch review — detect crossing and veering hotspots among Campbell tracks.

Cross-operating-point tracking (:mod:`vane.modal.identifier`) links modes into the
lines of a Campbell diagram. Where two lines cross or *veer* (approach, exchange
character, then repel), a per-transition tracker is least certain which mode
continues which line. This module detects those contested clusters and surfaces them
as auditable :class:`BranchGroup` records — a *branch-review view* — rather than
quietly trusting the automatic assignment.

Detection is deterministic and conservative. A pair of tracks is flagged only where
**both** their frequencies approach (within a scale-adaptive tolerance) **and** their
mode shapes genuinely compete (an ambiguous link or elevated cross-correlation across
a transition). Frequency proximity alone is not enough: two modes with orthogonal
shapes that merely pass close in frequency are tracked unambiguously and are not a
hazard, so they are not flagged.

The pass adds no risk to the tracks themselves — it only reports. Re-drawing contested
clusters is a separate, opt-in step (see the ``resolve`` option of
:func:`vane.modal.identify_modes`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from itertools import permutations
from typing import TYPE_CHECKING

import numpy as np

from vane.modal.mac import compute_mac

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from vane.modal.eigensolver import ModalSolution
    from vane.modal.identifier import IdentificationResult, ModeTrack

__all__ = [
    "BranchGroup",
    "BranchKind",
    "BranchSubWindow",
    "attach_branch_groups",
    "detect_branch_groups",
    "resolve_branch_groups",
]

# Default veering tolerance as a fraction of the median adjacent natural-frequency
# spacing, so the absolute Hz tolerance scales with the turbine's spectrum.
_VEERING_FRACTION = 0.25
# Cross-MAC at or above this counts as two tracks "seeing" each other's continuation.
_DEFAULT_MAC_COMPETITION = 0.25
# Two continuations whose MAC differs by less than this (and are not negligible) are a
# near-tie: the link is ambiguous.
_DEFAULT_AMBIGUITY_MARGIN = 0.2
# A group of more than this many tracks is dense (surfaced, and resolved only up to
# the branch cap below).
_MAX_SIMPLE_GROUP = 2
_MIN_TRACKS = 2
# A smooth alternative needs at least three window points (curvature is a second
# difference); shorter windows are left unresolved.
_MIN_CURVATURE_POINTS = 3
# Resolution enumerates branch permutations, so the number of branches is capped;
# larger clusters are genuinely ambiguous and are surfaced for review only.
_MAX_RESOLVE_BRANCHES = 4

# A per-operating-point assignment of branch slots to member-track indices.
_Assignment = tuple[int, ...]


class BranchKind(Enum):
    """Whether a contested window is a frequency crossing or a veering."""

    CROSSING = "crossing"
    VEERING = "veering"


@dataclass(frozen=True)
class BranchSubWindow:
    """A contiguous operating-point window where two tracks are contested.

    Parameters
    ----------
    op_lo, op_hi : int
        First and last operating-point index of the window (``op_hi > op_lo``).
    kind : BranchKind
        Whether the two tracks cross (their frequency order swaps) or veer
        (approach and separate without swapping) over the window.
    truncated : bool
        Whether the closest approach is at a window endpoint, so the hotspot may
        extend beyond the swept range.
    """

    op_lo: int
    op_hi: int
    kind: BranchKind
    truncated: bool


@dataclass(frozen=True)
class BranchGroup:
    """A cluster of tracks contested at a crossing or veering hotspot.

    Parameters
    ----------
    track_indices : tuple[int, ...]
        Indices (ascending) into :attr:`IdentificationResult.tracks` of the tracks
        in the cluster.
    op_lo, op_hi : int
        Hull operating-point window spanning all of the cluster's contested windows.
    frequency_band_hz : tuple[float, float]
        Minimum and maximum natural frequency of the member tracks within the hull
        window.
    kind : BranchKind
        Group-level classification — :attr:`BranchKind.CROSSING` if any member pair
        crosses, otherwise :attr:`BranchKind.VEERING`.
    min_separation_hz : float
        Closest frequency approach between any member pair within the window.
    min_separation_op : int
        Operating-point index at which the closest approach occurs.
    sub_windows : tuple[BranchSubWindow, ...]
        The distinct contested windows that make up the group, ordered canonically.
    is_dense : bool
        Whether the group has more than two tracks. Dense groups are still surfaced
        for review; resolution handles them up to a small branch cap.
    resolved : bool
        Whether a resolution pass computed a smooth alternative for this group
        (always ``False`` for detection-only runs).
    smooth_branches : tuple[tuple[float, ...], ...] or None
        The minimum-curvature ("smooth", non-crossing) frequency alternative for the
        contested cluster, one tuple per branch aligned with the window operating
        points ``op_lo..op_hi``. Set by the resolution pass for an eligible group, else
        ``None``. The actual tracks are never modified — this is review-only metadata
        contrasting the smooth convention with the shape-following base tracks.
    """

    track_indices: tuple[int, ...]
    op_lo: int
    op_hi: int
    frequency_band_hz: tuple[float, float]
    kind: BranchKind
    min_separation_hz: float
    min_separation_op: int
    sub_windows: tuple[BranchSubWindow, ...]
    is_dense: bool
    resolved: bool = False
    smooth_branches: tuple[tuple[float, ...], ...] | None = None


@dataclass
class _TrackMaps:
    """Per-operating-point lookups for one track."""

    mode: dict[int, int]
    freq: dict[int, float]
    ops: frozenset[int]


@dataclass
class _PairWindow:
    """A contested window between two tracks, before grouping."""

    sub_window: BranchSubWindow
    min_separation_hz: float
    min_separation_op: int


@dataclass
class _ContestedEdge:
    """One contested window between a specific pair of tracks."""

    track_a: int
    track_b: int
    window: _PairWindow


def detect_branch_groups(
    result: IdentificationResult,
    solutions: Sequence[ModalSolution],
    *,
    veering_tol_hz: float | None = None,
    mac_competition: float = _DEFAULT_MAC_COMPETITION,
    ambiguity_margin: float = _DEFAULT_AMBIGUITY_MARGIN,
) -> list[BranchGroup]:
    """Find crossing/veering hotspots among a result's tracks.

    Parameters
    ----------
    result : IdentificationResult
        The tracking result whose tracks are reviewed. Not modified.
    solutions : Sequence[ModalSolution]
        The per-operating-point modal solutions the tracks were built from, in
        operating-point order (used for the mode-shape correlations).
    veering_tol_hz : float or None, optional
        Frequency proximity below which two tracks may be contested. ``None`` (the
        default) derives it deterministically as ``0.25`` times the median adjacent
        natural-frequency spacing across the sweep, so it scales with the spectrum.
    mac_competition : float, optional
        Cross-MAC at or above which two tracks are deemed to compete across a
        transition.
    ambiguity_margin : float, optional
        Two non-negligible continuations whose MAC differs by less than this are a
        near-tie (an ambiguous link).

    Returns
    -------
    list[BranchGroup]
        The contested clusters, ordered canonically by window then frequency. Empty
        when no tracks are contested.

    Raises
    ------
    ValueError
        If ``mac_competition`` or ``ambiguity_margin`` is not in ``[0, 1]``, or
        ``veering_tol_hz`` is negative.
    """
    _validate_thresholds(veering_tol_hz, mac_competition, ambiguity_margin)
    tracks = result.tracks
    if len(tracks) < _MIN_TRACKS or result.n_operating_points < 2:
        return []

    macs = [
        compute_mac(solutions[k].mode_shapes, solutions[k + 1].mode_shapes)
        for k in range(len(solutions) - 1)
    ]
    tolerance = (
        veering_tol_hz if veering_tol_hz is not None else _auto_tolerance(solutions)
    )
    maps = [_track_maps(track) for track in tracks]

    edges: dict[tuple[int, int], list[_PairWindow]] = {}
    for a in range(len(tracks)):
        for b in range(a + 1, len(tracks)):
            windows = _contested_windows(
                maps[a], maps[b], macs, tolerance, mac_competition, ambiguity_margin
            )
            if windows:
                edges[a, b] = windows

    return _build_groups(edges, maps)


def attach_branch_groups(
    result: IdentificationResult,
    solutions: Sequence[ModalSolution],
    *,
    resolve: bool = False,
    parameter_values: Sequence[float] | None = None,
    **kwargs: float | None,
) -> IdentificationResult:
    """Return a copy of ``result`` with its ``branch_groups`` populated.

    Detection does not alter the tracks; the returned result shares the same track
    objects and only carries the additional branch groups.

    Parameters
    ----------
    result : IdentificationResult
        The tracking result to review.
    solutions : Sequence[ModalSolution]
        The per-operating-point modal solutions, in operating-point order.
    resolve : bool, optional
        When ``True``, also compute the smooth (minimum-curvature) alternative for
        each eligible group (see :func:`resolve_branch_groups`); the tracks are still
        not modified. Default ``False``.
    parameter_values : Sequence[float] or None, optional
        The operating-parameter value at each operating point, forwarded to the
        resolution curvature (see :func:`resolve_branch_groups`).
    **kwargs
        Forwarded to :func:`detect_branch_groups` (``veering_tol_hz``,
        ``mac_competition``, ``ambiguity_margin``).

    Returns
    -------
    IdentificationResult
        A new result with the detected (and optionally resolved) branch groups.
    """
    groups = detect_branch_groups(result, solutions, **kwargs)  # type: ignore[arg-type]
    if resolve:
        groups = resolve_branch_groups(
            groups, result, parameter_values=parameter_values
        )
    # ``replace`` keeps every other field (e.g. the heavily damped tier).
    return replace(result, branch_groups=groups)


def resolve_branch_groups(
    groups: list[BranchGroup],
    result: IdentificationResult,
    *,
    parameter_values: Sequence[float] | None = None,
) -> list[BranchGroup]:
    """Annotate each eligible group with its smooth (minimum-curvature) alternative.

    For a group of two to four tracks spanning at least three operating points where
    every track is present throughout the window, computes the assignment of the
    group's modes to branches that minimizes the total squared frequency curvature —
    the smooth, non-crossing convention — and attaches it as ``smooth_branches`` with
    ``resolved=True``. The tracks are never modified; this is review-only metadata.
    Larger groups, short windows, and ragged membership are left unresolved.

    Parameters
    ----------
    groups : list[BranchGroup]
        The detected branch groups.
    result : IdentificationResult
        The tracking result the groups index (for the member-track frequencies).
    parameter_values : Sequence[float] or None, optional
        The operating-parameter value at each operating point, used as the abscissa
        for the curvature so smoothness is measured against the real (possibly
        non-uniform) sweep spacing. ``None`` (the default) falls back to the
        operating-point index (uniform spacing). When given, it must have one value
        per operating point and be strictly increasing.

    Returns
    -------
    list[BranchGroup]
        New groups, with the smooth alternative attached where eligible.
    """
    maps = [_track_maps(track) for track in result.tracks]
    abscissa = _resolution_abscissa(parameter_values, result.n_operating_points)
    resolved: list[BranchGroup] = []
    for group in groups:
        branches = _smooth_branches(group, maps, abscissa)
        if branches is None:
            resolved.append(group)
        else:
            resolved.append(replace(group, resolved=True, smooth_branches=branches))
    return resolved


def _resolution_abscissa(
    parameter_values: Sequence[float] | None, n_operating_points: int
) -> list[float]:
    """Return the curvature abscissa: the parameter values, or the OP index if unusable.

    Falls back to the operating-point index (uniform spacing) when no parameter values
    are supplied, the count does not match, or the values are not finite and strictly
    increasing (a non-monotone, duplicated, or non-finite axis would make the spacing
    zero, negative, or non-finite).
    """
    index = [float(op) for op in range(n_operating_points)]
    if parameter_values is None or len(parameter_values) != n_operating_points:
        return index
    values = [float(value) for value in parameter_values]
    usable = all(np.isfinite(values)) and all(
        values[i] < values[i + 1] for i in range(len(values) - 1)
    )
    return values if usable else index


def _smooth_branches(
    group: BranchGroup,
    maps: list[_TrackMaps],
    abscissa: list[float],
) -> tuple[tuple[float, ...], ...] | None:
    """Return the smooth (minimum-curvature) branch alternative, or None if ineligible.

    Eligible groups have between two and :data:`_MAX_RESOLVE_BRANCHES` member tracks,
    a window of at least three points, all members present throughout the window, and
    finite frequencies.
    """
    n_branches = len(group.track_indices)
    if not _MAX_SIMPLE_GROUP <= n_branches <= _MAX_RESOLVE_BRANCHES:
        return None
    window = list(range(group.op_lo, group.op_hi + 1))
    if len(window) < _MIN_CURVATURE_POINTS:
        return None
    tracks = group.track_indices
    if not all(op in maps[track].freq for track in tracks for op in window):
        return None
    freqs = [[maps[track].freq[op] for op in window] for track in tracks]
    if not all(np.isfinite(value) for branch in freqs for value in branch):
        return None
    x = [abscissa[op] for op in window]
    assignment = _min_curvature_branches(freqs, x)
    branches = [
        tuple(freqs[assignment[k][slot]][k] for k in range(len(window)))
        for slot in range(n_branches)
    ]
    branches.sort(key=lambda branch: branch[0])
    return tuple(branches)


def _min_curvature_branches(
    freqs: list[list[float]], x: list[float]
) -> list[_Assignment]:
    """Return, per window point, the assignment of branch slots to member tracks.

    Each branch slot follows one frequency at each point; the assignment minimizes the
    total squared second derivative of every branch's frequency with respect to the
    operating parameter ``x``. It is a deterministic second-order Viterbi over
    permutations of the member tracks (curvature couples three consecutive points),
    with the first point fixed to the identity since the branch labels are
    interchangeable.
    """
    perms = list(permutations(range(len(freqs))))
    identity = tuple(range(len(freqs)))
    n_points = len(x)
    # State (perm_{k-1}, perm_k) -> best cost; back[step][(cur, nxt)] = perm_{k-2}.
    cost: dict[tuple[_Assignment, _Assignment], float] = {
        (identity, cur): 0.0 for cur in perms
    }
    back: list[dict[tuple[_Assignment, _Assignment], _Assignment]] = []
    for k in range(2, n_points):
        nxt_cost: dict[tuple[_Assignment, _Assignment], float] = {}
        choice: dict[tuple[_Assignment, _Assignment], _Assignment] = {}
        for (prev, cur), base in cost.items():
            for nxt in perms:
                value = base + _branch_curvature_cost(freqs, x, k - 1, prev, cur, nxt)
                key = (cur, nxt)
                if key not in nxt_cost or value < nxt_cost[key]:
                    nxt_cost[key] = value
                    choice[key] = prev
        cost = nxt_cost
        back.append(choice)

    assignment: list[_Assignment] = [identity] * n_points
    prev, cur = min(cost, key=lambda state: (cost[state], state))
    assignment[n_points - 1], assignment[n_points - 2] = cur, prev
    for step in range(len(back) - 1, -1, -1):
        earlier = back[step][prev, cur]
        assignment[step] = earlier
        prev, cur = earlier, prev
    return assignment


def _branch_curvature_cost(
    freqs: list[list[float]],
    x: list[float],
    k: int,
    prev: _Assignment,
    cur: _Assignment,
    nxt: _Assignment,
) -> float:
    """Return the summed squared second derivative of every branch at point ``k``.

    The second derivative is taken with respect to the operating parameter ``x`` (the
    non-uniform three-point formula), so smoothness is measured against the actual
    sweep spacing rather than the operating-point index.
    """
    h_lo = x[k] - x[k - 1]
    h_hi = x[k + 1] - x[k]
    total = 0.0
    for slot in range(len(freqs)):
        curv = _second_derivative(
            freqs[prev[slot]][k - 1],
            freqs[cur[slot]][k],
            freqs[nxt[slot]][k + 1],
            h_lo,
            h_hi,
        )
        total += curv * curv
    return total


def _second_derivative(
    prev: float, cur: float, nxt: float, h_lo: float, h_hi: float
) -> float:
    """Three-point second derivative on a non-uniform grid (uniform: prev-2cur+nxt)."""
    return 2.0 * (
        prev / (h_lo * (h_lo + h_hi))
        - cur / (h_lo * h_hi)
        + nxt / (h_hi * (h_lo + h_hi))
    )


def _validate_thresholds(
    veering_tol_hz: float | None, mac_competition: float, ambiguity_margin: float
) -> None:
    """Validate the detection thresholds, raising ``ValueError`` on a bad value."""
    for name, value in (
        ("mac_competition", mac_competition),
        ("ambiguity_margin", ambiguity_margin),
    ):
        if not 0.0 <= value <= 1.0:
            msg = f"{name} must be in [0, 1], got {value}"
            raise ValueError(msg)
    if veering_tol_hz is not None and veering_tol_hz < 0.0:
        msg = f"veering_tol_hz must be non-negative, got {veering_tol_hz}"
        raise ValueError(msg)


def _auto_tolerance(solutions: Sequence[ModalSolution]) -> float:
    """Return the scale-adaptive default veering tolerance (Hz).

    The median over operating points of the median spacing between consecutive
    sorted natural frequencies, scaled by :data:`_VEERING_FRACTION`. Non-finite
    frequencies are dropped first, so a single NaN/inf cannot poison the tolerance
    (and silently disable detection). Returns ``0`` (only exactly equal frequencies
    are then proximate) when no spacing is available.
    """
    spacings: list[float] = []
    for solution in solutions:
        finite = solution.natural_frequencies_hz[
            np.isfinite(solution.natural_frequencies_hz)
        ]
        if finite.size >= 2:
            spacings.append(float(np.median(np.diff(np.sort(finite)))))
    if not spacings:
        return 0.0
    return _VEERING_FRACTION * float(np.median(spacings))


def _track_maps(track: ModeTrack) -> _TrackMaps:
    """Build per-operating-point mode and frequency lookups for one track."""
    return _TrackMaps(
        mode=dict(zip(track.operating_points, track.mode_indices, strict=True)),
        freq=dict(
            zip(track.operating_points, track.natural_frequencies_hz, strict=True)
        ),
        ops=frozenset(track.operating_points),
    )


def _contested_windows(
    track_a: _TrackMaps,
    track_b: _TrackMaps,
    macs: list[npt.NDArray[np.float64]],
    tolerance: float,
    mac_competition: float,
    ambiguity_margin: float,
) -> list[_PairWindow]:
    """Return the contested windows shared by two tracks (empty if none).

    Each maximal contiguous run of shared operating points is split into finite
    segments (so a single non-finite frequency only drops its own sample, not the
    whole run) and then into *local* hotspots — the bands where the frequency gap is
    within tolerance, each grown by one operating point on either side to capture the
    approach shoulders — so a close approach yields a window around that approach
    rather than the tracks' whole common lifetime.
    """
    shared = sorted(track_a.ops & track_b.ops)
    windows: list[_PairWindow] = []
    for run in _contiguous_runs(shared):
        freqs_a = np.array([track_a.freq[op] for op in run], dtype=np.float64)
        freqs_b = np.array([track_b.freq[op] for op in run], dtype=np.float64)
        finite = np.isfinite(freqs_a) & np.isfinite(freqs_b)
        for segment in _finite_segments(finite):
            if len(segment) < 2:
                continue
            ops = [run[i] for i in segment]
            windows.extend(
                _segment_windows(
                    ops,
                    freqs_a[segment],
                    freqs_b[segment],
                    track_a,
                    track_b,
                    macs,
                    tolerance,
                    mac_competition,
                    ambiguity_margin,
                )
            )
    return windows


def _segment_windows(
    ops: list[int],
    freqs_a: npt.NDArray[np.float64],
    freqs_b: npt.NDArray[np.float64],
    track_a: _TrackMaps,
    track_b: _TrackMaps,
    macs: list[npt.NDArray[np.float64]],
    tolerance: float,
    mac_competition: float,
    ambiguity_margin: float,
) -> list[_PairWindow]:
    """Return the contested windows within one all-finite contiguous segment."""
    gaps = np.abs(freqs_a - freqs_b)
    windows: list[_PairWindow] = []
    for lo, hi in _proximity_bands(gaps, tolerance):
        # Each band spans at least two operating points (a close index is grown by one
        # shoulder), so the window always has a transition to classify.
        band_ops = ops[lo : hi + 1]
        if not _has_competition(
            band_ops, track_a, track_b, macs, mac_competition, ambiguity_margin
        ):
            continue
        windows.append(
            _classify_window(band_ops, freqs_a[lo : hi + 1], freqs_b[lo : hi + 1])
        )
    return windows


def _finite_segments(finite: npt.NDArray[np.bool_]) -> list[list[int]]:
    """Split a boolean mask into the index lists of its maximal true runs."""
    segments: list[list[int]] = []
    current: list[int] = []
    for index, keep in enumerate(finite):
        if keep:
            current.append(index)
        elif current:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def _proximity_bands(
    gaps: npt.NDArray[np.float64], tolerance: float
) -> list[tuple[int, int]]:
    """Return index ranges of local approaches, each grown by one shoulder point.

    A band is a maximal run of indices whose gap is at or below ``tolerance``,
    expanded by one index on each side (clamped to the array) so the classifier sees
    the tracks separating again. Overlapping expanded bands are merged.
    """
    close = gaps <= tolerance
    last = gaps.shape[0] - 1
    bands: list[tuple[int, int]] = []
    index = 0
    while index <= last:
        if not close[index]:
            index += 1
            continue
        end = index
        while end + 1 <= last and close[end + 1]:
            end += 1
        lo, hi = max(0, index - 1), min(last, end + 1)
        if bands and lo <= bands[-1][1]:
            bands[-1] = (bands[-1][0], max(bands[-1][1], hi))
        else:
            bands.append((lo, hi))
        index = end + 1
    return bands


def _has_competition(
    run: list[int],
    track_a: _TrackMaps,
    track_b: _TrackMaps,
    macs: list[npt.NDArray[np.float64]],
    mac_competition: float,
    ambiguity_margin: float,
) -> bool:
    """Whether the two tracks compete across any transition within ``run``."""
    for op in run[:-1]:
        mac = macs[op]
        a_now, a_next = track_a.mode[op], track_a.mode[op + 1]
        b_now, b_next = track_b.mode[op], track_b.mode[op + 1]
        cross = max(float(mac[a_now, b_next]), float(mac[b_now, a_next]))
        if cross >= mac_competition:
            return True
        if _is_near_tie(
            float(mac[a_now, a_next]),
            float(mac[a_now, b_next]),
            mac_competition,
            ambiguity_margin,
        ) or _is_near_tie(
            float(mac[b_now, b_next]),
            float(mac[b_now, a_next]),
            mac_competition,
            ambiguity_margin,
        ):
            return True
    return False


def _is_near_tie(own: float, other: float, floor: float, margin: float) -> bool:
    """Whether a track's two candidate continuations are a non-negligible near-tie."""
    return max(own, other) >= floor and abs(own - other) < margin


def _classify_window(
    window: list[int],
    freqs_a: npt.NDArray[np.float64],
    freqs_b: npt.NDArray[np.float64],
) -> _PairWindow:
    """Classify a contested window as a crossing or veering."""
    signed = freqs_a - freqs_b
    gaps = np.abs(signed)
    # A crossing requires the frequency order to *swap* across the window — the
    # endpoints must have strictly opposite signs. A tie (zero gap) at an endpoint has
    # not swapped within the sampled window, so it is a truncated veering, not a
    # crossing.
    crossing = bool(signed[0] * signed[-1] < 0)
    closest = int(np.argmin(gaps))
    if crossing:
        kind, truncated = BranchKind.CROSSING, False
    else:
        kind = BranchKind.VEERING
        truncated = closest in (0, len(window) - 1)
    return _PairWindow(
        sub_window=BranchSubWindow(
            op_lo=window[0], op_hi=window[-1], kind=kind, truncated=truncated
        ),
        min_separation_hz=float(gaps[closest]),
        min_separation_op=window[closest],
    )


def _contiguous_runs(ops: list[int]) -> list[list[int]]:
    """Split a sorted operating-point list into maximal consecutive-integer runs."""
    runs: list[list[int]] = []
    for op in ops:
        if runs and op == runs[-1][-1] + 1:
            runs[-1].append(op)
        else:
            runs.append([op])
    return runs


def _build_groups(
    edges: dict[tuple[int, int], list[_PairWindow]],
    maps: list[_TrackMaps],
) -> list[BranchGroup]:
    """Assemble canonical branch groups from contested track-pair windows.

    Groups are connected components over the *contested windows*, not just track
    adjacency: two windows join only when they share a track **and** their
    operating-point ranges overlap. So two temporally separate hotspots that share a
    track (e.g. tracks A-B early and B-C late) stay distinct groups instead of merging
    into one long, falsely dense span.
    """
    items = [
        _ContestedEdge(pair[0], pair[1], window)
        for pair, windows in edges.items()
        for window in windows
    ]
    if not items:
        return []
    groups = [_assemble_group(component, maps) for component in _edge_components(items)]
    groups.sort(key=lambda g: (g.op_lo, g.frequency_band_hz[0], g.track_indices))
    return groups


def _edge_components(items: list[_ContestedEdge]) -> list[list[_ContestedEdge]]:
    """Group contested windows that share a track and overlap in time."""
    parent = list(range(len(items)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if _edges_linked(items[i], items[j]):
                parent[find(i)] = find(j)

    components: dict[int, list[_ContestedEdge]] = {}
    for index, item in enumerate(items):
        components.setdefault(find(index), []).append(item)
    return list(components.values())


def _edges_linked(first: _ContestedEdge, second: _ContestedEdge) -> bool:
    """Whether two contested windows share a track and overlap in operating points."""
    if not {first.track_a, first.track_b} & {second.track_a, second.track_b}:
        return False
    a, b = first.window.sub_window, second.window.sub_window
    return max(a.op_lo, b.op_lo) <= min(a.op_hi, b.op_hi)


def _assemble_group(edges: list[_ContestedEdge], maps: list[_TrackMaps]) -> BranchGroup:
    """Build one BranchGroup from a component of overlapping contested windows."""
    pair_windows = [edge.window for edge in edges]
    members = sorted({t for edge in edges for t in (edge.track_a, edge.track_b)})
    op_lo = min(w.sub_window.op_lo for w in pair_windows)
    op_hi = max(w.sub_window.op_hi for w in pair_windows)
    closest = min(pair_windows, key=lambda w: w.min_separation_hz)
    crossing = any(w.sub_window.kind is BranchKind.CROSSING for w in pair_windows)
    sub_windows = tuple(
        sorted(
            {w.sub_window for w in pair_windows},
            key=lambda s: (s.op_lo, s.op_hi, s.kind.value),
        )
    )
    return BranchGroup(
        track_indices=tuple(members),
        op_lo=op_lo,
        op_hi=op_hi,
        frequency_band_hz=_frequency_band(members, maps, op_lo, op_hi),
        kind=BranchKind.CROSSING if crossing else BranchKind.VEERING,
        min_separation_hz=closest.min_separation_hz,
        min_separation_op=closest.min_separation_op,
        sub_windows=sub_windows,
        is_dense=len(members) > _MAX_SIMPLE_GROUP,
    )


def _frequency_band(
    members: list[int], maps: list[_TrackMaps], op_lo: int, op_hi: int
) -> tuple[float, float]:
    """Return the (min, max) member-track frequency within the hull window."""
    values = [
        freq
        for track in members
        for op, freq in maps[track].freq.items()
        if op_lo <= op <= op_hi and np.isfinite(freq)
    ]
    return (min(values), max(values)) if values else (0.0, 0.0)
