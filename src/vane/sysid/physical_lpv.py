# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Physical-state LPV family export: named operating-point plants from ``.lin`` files.

:func:`~vane.sysid.build_rom_family` packages *reduced, mode-tracked* models for
modal/Campbell work. This module is the complementary export for **control design**: it
reads a directory of OpenFAST linearization files and assembles, per operating point, a
state-space model whose states/inputs/outputs are the **named physical DOFs** the caller
selects (e.g. ``"ED PtfmPitch"``, ``"GenTq"``, ``"BldPitch1"``), with a consistent
ordering across the operating-point sweep -- the linear-parameter-varying plant a
gain-scheduled controller schedules on.

The matrices are **azimuth-averaged in the physical (rotating) frame**, preserving the
original per-blade channel names; no MBC/Coleman transform is applied (for the
multi-blade coordinate representation use :func:`~vane.sysid.build_rom_family` /
:func:`~vane.sysid.state_space_from_mbc`). The result is one LTI model per operating
point -- the standard plant for LQR/LPV control synthesis. This is the single VANE entry
point a downstream controller needs: it never touches raw ``.lin`` files itself.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

from vane.sysid.state_space import StateSpace

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import numpy.typing as npt

    from vane.io.lin_reader import LinFile

__all__ = ["LPVFamily", "build_physical_lpv_family"]

_log = logging.getLogger(__name__)

# Default ceiling on the relative dropped-coupling norm of a state subset (see
# build_physical_lpv_family). 5 % admits a subset whose neglected feedback from the
# unselected states is a small perturbation of the selected dynamics, while rejecting
# the gross truncations (a DOF without its rate, or a platform DOF cut from the states
# it is coupled to) that change the plant qualitatively.
_DEFAULT_MAX_DROPPED_COUPLING = 0.05
# Number of dropped states the coupling-check error names as the largest contributors.
_N_CONTRIBUTORS = 3

# Scheduling-variable name (as a caller writes it) -> the LinFile header attribute.
_SCHEDULING_ATTR: dict[str, str] = {
    "windspeed": "wind_speed",
    "rotorspeed": "rotor_speed",
    "rotspeed": "rotor_speed",
}

# The physical unit of each scheduling variable's values, in the *raw linearization*
# convention: wind speed in m/s, rotor speed in rad/s (NOT rpm — the modal/Campbell path
# reports rpm). Surfaced on LPVFamily so a downstream scheduled controller cannot
# mis-schedule on the wrong unit.
_SCHEDULING_UNIT: dict[str, str] = {"wind_speed": "m/s", "rotor_speed": "rad/s"}

# Tolerances for the operating condition to be considered constant across one point's
# azimuth files. The two axes are checked with different tolerances on purpose:
#   * Wind speed is the *commanded* uniform inflow, identical across the azimuth files
#     of one operating point, so it stays TIGHT -- two same-root files at, say, 8.0 and
#     8.3 m/s are different operating points and must be rejected, not averaged.
#   * Rotor speed is a *response*: a trim/CalcSteady linearization captures the periodic
#     steady state, whose rotor speed physically ripples a fraction of a percent around
#     the revolution even in steady wind. That ripple is what azimuth-averaging folds
#     out, so the rotor-speed tolerance is relaxed to admit it while still catching
#     genuinely different points (a sweep step is tens of percent of rotor speed).
_WIND_RTOL = 1e-3
_ROTOR_RTOL = 5e-2
_CONDITION_ATOL = 1e-6
# Two azimuths (rad) closer than this are treated as the same -- a duplicate file.
# ~0.006 deg: well below any real sweep spacing, but above the ~4-decimal azimuth
# precision a .lin header records (so a wrapped 0 vs 2pi still reads as a duplicate).
_AZIMUTH_ATOL = 1e-4

# The full descriptions (with units) a request resolved to: (states, inputs, outputs).
# Compared across points so a name maps to the same channel everywhere.
_Resolved = tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]


@dataclass
class LPVFamily:
    """A parameter-indexed family of **named physical-state** plants for control design.

    Parameters
    ----------
    scheduling_points : npt.NDArray[np.float64]
        The scheduling-variable value of each operating point, shape ``(n_points,)``,
        sorted strictly ascending. In the raw linearization unit: ``WindSpeed`` in m/s,
        ``RotorSpeed`` in **rad/s** (not rpm).
    systems : list[StateSpace]
        The physical-state plant at each operating point, in the same order as
        ``scheduling_points``. Every system has the same selected states, inputs, and
        outputs, in the same order -- the consistency a scheduled controller relies on.
    state_names, input_names, output_names : list[str]
        The selected physical state, input, and output names, shared by every system.
    scheduling_variable : str
        The scheduling variable the family is indexed on (e.g. ``"WindSpeed"``).
    scheduling_unit : str
        The physical unit of ``scheduling_points`` in the raw linearization convention:
        ``"m/s"`` for ``WindSpeed``, ``"rad/s"`` for ``RotorSpeed`` (**not** rpm, unlike
        the modal/Campbell path). Carried explicitly so a scheduled controller cannot
        mis-schedule by assuming the wrong unit.
    auto_added_states : list[str]
        The states the builder added to the request to complete a position/velocity
        pair (see ``pair_derivatives`` in :func:`build_physical_lpv_family`), named by
        their unit-stripped linearization descriptions. Each also appears in
        ``state_names``. Empty when nothing was added.
    dropped_coupling_a : npt.NDArray[np.float64]
        Per operating point, the relative state-matrix coupling the state selection
        discards, ``||A[s, ~s]||_F / ||A[s, :]||_F`` (``s`` the selected states), shape
        ``(n_points,)``, in the order of ``scheduling_points``. Zero when every state
        is selected.
    dropped_coupling_c : npt.NDArray[np.float64]
        Per operating point, the relative output coupling the selection discards,
        ``||C[o, ~s]||_F / ||C[o, :]||_F`` (``o`` the selected outputs), shape
        ``(n_points,)``. Zero when no outputs are selected or every state is selected.
    """

    scheduling_points: npt.NDArray[np.float64]
    systems: list[StateSpace]
    state_names: list[str] = field(default_factory=list)
    input_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)
    scheduling_variable: str = "WindSpeed"
    scheduling_unit: str = "m/s"
    auto_added_states: list[str] = field(default_factory=list)
    dropped_coupling_a: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    dropped_coupling_c: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )

    @property
    def dropped_coupling(self) -> npt.NDArray[np.float64]:
        """Per-point relative dropped coupling, the larger of the ``A`` and ``C`` terms.

        Returns
        -------
        npt.NDArray[np.float64]
            ``max(dropped_coupling_a, dropped_coupling_c)`` element-wise, shape
            ``(n_points,)``. This is the quantity ``max_dropped_coupling`` checks.
        """
        combined: npt.NDArray[np.float64] = np.maximum(
            self.dropped_coupling_a, self.dropped_coupling_c
        )
        return combined


def _normalize_names(names: Sequence[str], label: str, *, required: bool) -> list[str]:
    """Materialize and validate a requested channel-name selection.

    Guards the common footguns: a single string (an easy mistake for a one-name list,
    which would otherwise iterate characters), a one-shot generator (consumed before the
    second use), duplicate names, and non-string entries.
    """
    # ValueError (not TypeError) keeps a single contract for this argument: a bad name
    # selection is a bad selection whether the type or the value is wrong.
    if isinstance(names, str):
        msg = f"{label} must be a sequence of names, not a single string {names!r}"
        raise ValueError(msg)  # noqa: TRY004
    materialized = list(names)
    if required and not materialized:
        msg = f"{label} must name at least one channel to select"
        raise ValueError(msg)
    for name in materialized:
        if not isinstance(name, str):
            msg = f"{label} entries must be strings, got a {type(name).__name__}"
            raise ValueError(msg)  # noqa: TRY004
    duplicates = sorted({n for n in materialized if materialized.count(n) > 1})
    if duplicates:
        msg = f"{label} has duplicate names: {duplicates}"
        raise ValueError(msg)
    return materialized


def _strip_units(description: str) -> str:
    """Return a channel description without its trailing ``, <units>`` comment."""
    return description.rsplit(",", 1)[0].strip()


# The phrase OpenFAST inserts to name a state's first derivative, e.g. a position DOF
# ``"... Platform pitch ..."`` and its rate ``"... First time derivative of Platform
# pitch ..."``. Both appear in a second-order plant's continuous-state list, so a
# request that omits the phrase substring-matches both; this prefers the position DOF.
_DERIVATIVE_PHRASE = "First time derivative of "


def _without_derivative_phrase(description: str) -> str:
    """Return ``description`` with the first-derivative phrase removed, if present."""
    return description.replace(_DERIVATIVE_PHRASE, "")


def _resolve_one(name: str, normalized: Sequence[str], kind: str) -> int:
    """Resolve one requested ``name`` to a single channel index.

    The matching tiers, in order of preference, are:

    1. an **exact** match against the unit-stripped description;
    2. a unique **substring** match;
    3. when a request substring-matches both a displacement DOF and its
       ``"First time derivative of <DOF>"`` twin (a second-order plant lists both), the
       **non-derivative** channel -- but only when the matches are exactly that DOF and
       its own derivatives (identical once the derivative phrase is stripped). Two
       genuinely different channels (e.g. ``"PtfmPitch"`` matching both ``"PtfmPitch"``
       and ``"PtfmPitch rate"``) stay ambiguous.

    A name that matches nothing, or that remains genuinely ambiguous after these tiers,
    raises ``ValueError`` listing the candidate channels rather than silently selecting
    one.
    """
    exact = [i for i, n in enumerate(normalized) if n == name]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:  # pragma: no cover - descriptions are unique within one file
        raise _ambiguous_error(name, normalized, exact, kind)

    partial = [i for i, n in enumerate(normalized) if name in n]
    if not partial:
        msg = (
            f"requested {kind} {name!r} was not found among the linearization "
            f"{kind} channels"
        )
        raise ValueError(msg)
    if len(partial) == 1:
        return partial[0]

    # A request can land on a displacement DOF and its rate twin together. They are the
    # same channel once the "First time derivative of " phrase is stripped, so the only
    # real choice is position-vs-rate: prefer the (non-derivative) position. If the
    # candidates are not one DOF and its own derivatives, the request is genuinely
    # ambiguous and must be rejected with the candidate list.
    stripped = {_without_derivative_phrase(normalized[i]) for i in partial}
    non_derivative = [i for i in partial if _DERIVATIVE_PHRASE not in normalized[i]]
    if len(stripped) == 1 and len(non_derivative) == 1:
        return non_derivative[0]
    raise _ambiguous_error(name, normalized, partial, kind)


def _ambiguous_error(
    name: str, normalized: Sequence[str], matches: Sequence[int], kind: str
) -> ValueError:
    """Return a ``ValueError`` listing the channels a request matched."""
    hits = ", ".join(repr(normalized[i]) for i in matches)
    msg = (
        f"requested {kind} {name!r} matches multiple channels ({hits}); "
        "use a more specific name"
    )
    return ValueError(msg)


def _select_indices(
    requested: Sequence[str], descriptions: Sequence[str], kind: str
) -> list[int]:
    """Return the column/row index of each requested name in ``descriptions``.

    Each name is resolved by :func:`_resolve_one` (exact, then whole-token, then
    shortest-description tiers). A name that is missing or ambiguous raises
    ``ValueError`` rather than silently selecting the wrong channel.
    """
    normalized = [_strip_units(d) for d in descriptions]
    indices = [_resolve_one(name, normalized, kind) for name in requested]
    if len(set(indices)) != len(indices):
        msg = (
            f"two requested {kind} names resolve to the same linearization channel; "
            "make the names distinct"
        )
        raise ValueError(msg)
    return indices


def _derivative_twin(index: int, normalized: Sequence[str]) -> int | None:
    """Return the index of ``normalized[index]``'s position/velocity twin, if unique.

    A second-order DOF ``"ED <DOF>"`` and its rate ``"ED First time derivative of
    <DOF>"`` are twins: identical once the derivative phrase is stripped, exactly one of
    them carrying the phrase. Returns ``None`` when the channel has no such twin.
    """
    name = normalized[index]
    is_rate = _DERIVATIVE_PHRASE in name
    base = _without_derivative_phrase(name)
    twins = [
        j
        for j, other in enumerate(normalized)
        if j != index
        and (_DERIVATIVE_PHRASE in other) != is_rate
        and _without_derivative_phrase(other) == base
    ]
    return twins[0] if len(twins) == 1 else None


def _pair_state_names(
    states: Sequence[str], x_desc: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Complete every selected second-order DOF with its position/velocity twin.

    Each requested state whose twin exists in the linearization but was not requested
    gets that twin inserted next to it -- a position immediately followed by its
    velocity -- so the selected block keeps the kinematic identity
    ``d(position)/dt = velocity``. Added twins are named by their unit-stripped
    descriptions (an exact match, so they resolve identically at every point).

    Returns
    -------
    tuple of (list[str], list[str])
        The final state-name list and the names that were added, in the order added.
    """
    normalized = [_strip_units(d) for d in x_desc]
    indices = _select_indices(states, x_desc, "state")
    selected = set(indices)
    final: list[str] = []
    added: list[str] = []
    for name, index in zip(states, indices, strict=True):
        twin = _derivative_twin(index, normalized)
        if twin is None or twin in selected:
            final.append(name)
            continue
        selected.add(twin)
        twin_name = normalized[twin]
        added.append(twin_name)
        if _DERIVATIVE_PHRASE in normalized[index]:  # a rate: its position goes first
            final.extend([twin_name, name])
        else:
            final.extend([name, twin_name])
    return final, added


@dataclass
class _Coupling:
    """One operating point's dropped-coupling measures (see ``_dropped_coupling``)."""

    a: float
    c: float
    # Per dropped state: (unit-stripped description, relative contribution).
    contributors: list[tuple[str, float]]


def _relative_block_norms(
    rows: npt.NDArray[np.float64], dropped: Sequence[int]
) -> tuple[float, npt.NDArray[np.float64]]:
    """Return ``||rows[:, dropped]||_F / ||rows||_F`` and each dropped column's part."""
    total = float(np.linalg.norm(rows))
    if total == 0.0 or not dropped:
        return 0.0, np.zeros(len(dropped), dtype=np.float64)
    columns: npt.NDArray[np.float64] = (
        np.linalg.norm(rows[:, list(dropped)], axis=0) / total
    )
    return float(np.linalg.norm(columns)), columns


def _dropped_coupling(
    a: npt.NDArray[np.float64],
    c: npt.NDArray[np.float64] | None,
    x_idx: Sequence[int],
    y_idx: Sequence[int],
    x_desc: Sequence[str],
) -> _Coupling:
    """Measure how much coupling to the unselected states a selection discards.

    The ``A`` term is ``||A[s, ~s]||_F / ||A[s, :]||_F`` -- the share of the selected
    states' dynamics driven by states the subset drops -- and the ``C`` term is the
    analogous ``||C[o, ~s]||_F / ||C[o, :]||_F`` for the selected outputs ``o``. Both
    are zero for a full-state selection.
    """
    chosen = set(x_idx)
    dropped = [j for j in range(a.shape[0]) if j not in chosen]
    a_ratio, a_cols = _relative_block_norms(a[list(x_idx), :], dropped)
    c_ratio, c_cols = 0.0, np.zeros(len(dropped), dtype=np.float64)
    if c is not None and y_idx:
        c_ratio, c_cols = _relative_block_norms(c[list(y_idx), :], dropped)
    shares = np.maximum(a_cols, c_cols)
    contributors = [
        (_strip_units(x_desc[j]), float(share))
        for j, share in zip(dropped, shares, strict=True)
    ]
    contributors.sort(key=lambda item: item[1], reverse=True)
    return _Coupling(a=a_ratio, c=c_ratio, contributors=contributors[:_N_CONTRIBUTORS])


def _validate_tolerance(max_dropped_coupling: float | None) -> float | None:
    """Validate the dropped-coupling ceiling: ``None`` or a finite float ``>= 0``."""
    if max_dropped_coupling is None:
        return None
    if isinstance(max_dropped_coupling, bool) or not isinstance(
        max_dropped_coupling, int | float
    ):
        msg = (
            "max_dropped_coupling must be a finite number >= 0 or None, got a "
            f"{type(max_dropped_coupling).__name__}"
        )
        raise ValueError(msg)  # noqa: TRY004
    value = float(max_dropped_coupling)
    if not math.isfinite(value) or value < 0.0:
        msg = f"max_dropped_coupling must be a finite number >= 0 or None; got {value}"
        raise ValueError(msg)
    return value


def _stack_mean(
    matrices: list[npt.NDArray[np.float64] | None], label: str
) -> npt.NDArray[np.float64] | None:
    """Average a matrix across azimuth files; ``None`` only if absent in all of them."""
    if all(m is None for m in matrices):
        return None
    if any(m is None for m in matrices):
        msg = f"the {label} matrix is present in some azimuth files but not others"
        raise ValueError(msg)
    present = [np.asarray(m, dtype=np.float64) for m in matrices if m is not None]
    shapes = {m.shape for m in present}
    if len(shapes) != 1:
        msg = f"the {label} matrix has inconsistent shapes across azimuths: {shapes}"
        raise ValueError(msg)
    averaged: npt.NDArray[np.float64] = np.mean(np.stack(present), axis=0)
    return averaged


def _physical_system(
    files: Sequence[LinFile],
    states: Sequence[str],
    inputs: Sequence[str],
    outputs: Sequence[str],
) -> tuple[StateSpace, _Resolved, _Coupling]:
    """Average one operating point's azimuth files and select the requested channels.

    Returns the selected ``StateSpace``, the **resolved full descriptions** (with
    units) the requested names matched, so the caller can verify that a name resolves to
    the *same* physical channel at every operating point, and the coupling the state
    selection drops at this point.
    """
    first = files[0]
    x_desc, u_desc, y_desc = (
        first.x.descriptions,
        first.u.descriptions,
        first.y.descriptions,
    )
    # The azimuth files of one operating point must share the *exact* channel layout
    # (same descriptions, same order): the matrices are averaged by row/column index, so
    # a different order would silently average together different physical channels.
    for lin in files:
        if (
            lin.x.descriptions != x_desc
            or lin.u.descriptions != u_desc
            or lin.y.descriptions != y_desc
        ):
            msg = "azimuth files of one operating point have different channel layouts"
            raise ValueError(msg)

    a = _stack_mean([lin.a for lin in files], "A")
    if a is None:
        msg = "linearization has no A matrix; cannot build a plant"
        raise ValueError(msg)
    b = _stack_mean([lin.b for lin in files], "B")
    c = _stack_mean([lin.c for lin in files], "C")
    d = _stack_mean([lin.d for lin in files], "D")

    x_idx = _select_indices(states, x_desc, "state")
    u_idx = _select_indices(inputs, u_desc, "input") if inputs else []
    y_idx = _select_indices(outputs, y_desc, "output") if outputs else []
    if inputs and b is None:
        msg = "inputs were requested but the linearization has no B matrix"
        raise ValueError(msg)
    if outputs and c is None:
        msg = "outputs were requested but the linearization has no C matrix"
        raise ValueError(msg)

    a_sel = a[np.ix_(x_idx, x_idx)]
    b_sel = b[np.ix_(x_idx, u_idx)] if inputs and b is not None else None
    c_sel = c[np.ix_(y_idx, x_idx)] if outputs and c is not None else None
    d_sel = d[np.ix_(y_idx, u_idx)] if inputs and outputs and d is not None else None
    system = StateSpace(
        a=a_sel,
        b=b_sel,
        c=c_sel,
        d=d_sel,
        state_names=list(states),
        input_names=list(inputs),
        output_names=list(outputs),
    )
    resolved: _Resolved = (
        tuple(x_desc[i] for i in x_idx),
        tuple(u_desc[i] for i in u_idx),
        tuple(y_desc[i] for i in y_idx),
    )
    coupling = _dropped_coupling(a, c if outputs else None, x_idx, y_idx, x_desc)
    return system, resolved, coupling


def _check_operating_condition(files: Sequence[LinFile]) -> None:
    """Validate that one point's azimuth files are a single operating condition.

    A real azimuth sweep holds **both** wind and rotor speed fixed and varies only the
    azimuth. If files grouped under one case root disagree on either (i.e. conditions
    sharing a name), the mean would fold them into a fabricated plant at a point that
    never existed -- so reject it, regardless of which axis the family is scheduled on.
    """
    for attr, label, rtol in (
        ("wind_speed", "WindSpeed", _WIND_RTOL),
        ("rotor_speed", "RotorSpeed", _ROTOR_RTOL),
    ):
        values = [float(getattr(lin, attr)) for lin in files]
        first = values[0]
        if not np.allclose(values, first, rtol=rtol, atol=_CONDITION_ATOL):
            msg = (
                f"the azimuth files of one operating point disagree on {label} "
                f"({values}); a discovered point must be a single operating condition"
            )
            raise ValueError(msg)
    # Each azimuth must appear once: a duplicated .lin (e.g. a repeated LinTimes or a
    # copied file) would be counted twice and bias the physical-frame average away from
    # the one-revolution mean. Azimuth is circular, so wrap into [0, 2pi) and also check
    # the gap across the 2pi boundary (e.g. 0 and 2pi are the same azimuth).
    two_pi = 2.0 * np.pi
    azimuths = sorted(float(lin.azimuth) % two_pi for lin in files)
    gaps = [hi - lo for lo, hi in pairwise(azimuths)]
    gaps.append(azimuths[0] + two_pi - azimuths[-1])  # the wrap-around gap
    if any(gap <= _AZIMUTH_ATOL for gap in gaps):
        msg = (
            f"an operating point has duplicate azimuth files (azimuths {azimuths}); "
            "each azimuth must appear once"
        )
        raise ValueError(msg)


def _scheduling_value(files: Sequence[LinFile], attr: str) -> float:
    """Return an operating point's scheduling value (mean over its azimuth files)."""
    return float(np.mean([float(getattr(lin, attr)) for lin in files]))


def build_physical_lpv_family(
    lin_dir: str | Path,
    *,
    states: Sequence[str],
    inputs: Sequence[str] = (),
    outputs: Sequence[str] = (),
    scheduling_var: str = "WindSpeed",
    pair_derivatives: bool = True,
    max_dropped_coupling: float | None = _DEFAULT_MAX_DROPPED_COUPLING,
) -> LPVFamily:
    """Build a named physical-state LPV plant family from a directory of ``.lin`` files.

    Discovers the operating points in ``lin_dir``, averages each point's azimuth files
    in the physical frame, and selects the named ``states``/``inputs``/``outputs`` into
    a state-space model -- one per operating point, ordered by ``scheduling_var``, with
    the same channel layout everywhere. No MBC transform is applied, so per-blade
    channel names are preserved (use :func:`~vane.sysid.build_rom_family` for the
    multi-blade coordinate representation).

    Selecting a state *subset* truncates the plant: rows and columns of the unselected
    states are removed, discarding their coupling into the selected ones. Two
    safeguards keep that truncation honest. Second-order DOFs are completed with their
    position/velocity twin (``pair_derivatives``), and the relative coupling the
    selection still drops is measured at every point and checked
    (``max_dropped_coupling``).

    Parameters
    ----------
    lin_dir : str or pathlib.Path
        Directory of OpenFAST ``.lin`` files, searched recursively, so a campaign laid
        out with one sub-directory per operating point is read as distinct points
        (grouped into operating points by case name).
    states : Sequence[str]
        Physical state names to select, e.g. ``["ED PtfmPitch", "ED RotSpeed"]``.
        Matched against the linearization's state descriptions (units stripped): an
        exact match is preferred, else a unique substring. When a name matches both a
        DOF and its ``"First time derivative of <DOF>"`` twin, the position DOF is
        chosen (and, with ``pair_derivatives``, its twin is then added); a name
        matching two genuinely different channels raises, listing them. The selected
        order is the state order, with any auto-added twin inserted next to its
        partner.
    inputs, outputs : Sequence[str], optional
        Input and output names to select (e.g. ``["GenTq", "BldPitch1"]``); empty
        selects none (no ``B`` / ``C``).
    scheduling_var : str, optional
        The scheduling variable: ``"WindSpeed"`` (default, m/s) or ``"RotorSpeed"``.
        ``RotorSpeed`` is the raw linearization rotor speed in **rad/s** (the unit
        OpenFAST writes), *not* rpm — unlike the modal/Campbell path, which reports
        rotor speed in rpm. Schedule a controller built against rad/s accordingly.
    pair_derivatives : bool, optional
        If true (default), every selected second-order DOF whose twin -- the
        ``"First time derivative of <DOF>"`` velocity of a selected position, or the
        position of a selected velocity -- exists in the linearization but was not
        requested gets that twin added, immediately after a position or immediately
        before a velocity, so each pair reads position then velocity. The added names
        (unit-stripped descriptions) are logged at INFO and recorded in
        :attr:`LPVFamily.auto_added_states`. A position without its velocity is a pure
        truncation artefact (``d(position)/dt`` is exactly the dropped velocity), so
        disable this only to reproduce a deliberate truncation.
    max_dropped_coupling : float or None, optional
        Ceiling on the relative coupling the state selection discards, default
        ``0.05``. At each operating point VANE computes
        ``||A[s, ~s]||_F / ||A[s, :]||_F`` over the selected states ``s`` and, when
        outputs are selected, ``||C[o, ~s]||_F / ||C[o, :]||_F`` over the selected
        outputs ``o``; the per-point value is the larger of the two
        (:attr:`LPVFamily.dropped_coupling`). If the largest value over the family
        exceeds the ceiling, ``ValueError`` is raised naming the worst point and the
        dropped states contributing most. The norms are taken in the linearization's
        own units, so the measure is a screening check, not a unit-invariant error
        bound. ``None`` disables the check (the norms are still recorded). Must be a
        finite number ``>= 0``.

    Returns
    -------
    LPVFamily
        The physical-state plant family, with ``scheduling_points`` sorted ascending and
        a ``StateSpace`` per point sharing the selected channel layout. Its
        ``state_names`` is the final (paired) state list; ``auto_added_states``,
        ``dropped_coupling_a`` and ``dropped_coupling_c`` record the pairing and the
        per-point truncation norms.

    Raises
    ------
    FileNotFoundError
        If ``lin_dir`` does not exist or contains no ``.lin`` files.
    ValueError
        If ``scheduling_var`` is unknown; ``max_dropped_coupling`` is not ``None`` or a
        finite number ``>= 0``; a name selection is invalid (a bare string, an empty
        ``states``, a duplicate, or a non-string entry); a requested name is
        missing/ambiguous; a name resolves to a different channel (or different units)
        across operating points; an operating point's azimuth files disagree on the
        channel layout or the operating condition (wind or rotor speed) or contain a
        duplicate azimuth; the schedule is non-finite or not increasing; or the state
        selection drops more coupling than ``max_dropped_coupling`` allows.
    """
    key = scheduling_var.lower().replace(" ", "").replace("_", "")
    attr = _SCHEDULING_ATTR.get(key)
    if attr is None:
        allowed = "WindSpeed, RotorSpeed"
        msg = f"scheduling_var must be one of {allowed}; got {scheduling_var!r}"
        raise ValueError(msg)
    tolerance = _validate_tolerance(max_dropped_coupling)
    states = _normalize_names(states, "states", required=True)
    inputs = _normalize_names(inputs, "inputs", required=False)
    outputs = _normalize_names(outputs, "outputs", required=False)

    # Imported lazily so the vane.sysid package does not pull in the study/pipeline
    # layer just to be imported; discovery is only needed when this function runs.
    # discover_operating_points raises FileNotFoundError for a missing/empty directory.
    from vane.study import discover_operating_points

    points = discover_operating_points(lin_dir)
    added: list[str] = []
    if pair_derivatives:
        # The pairing is decided once, on the first point's layout; the added twins are
        # exact descriptions, and the cross-point resolution check below guarantees
        # they denote the same channels everywhere.
        states, added = _pair_state_names(states, points[0].files[0].x.descriptions)
        if added:
            _log.info(
                "added the position/velocity twin state(s) %s to complete the "
                "requested second-order DOFs (pair_derivatives=False to opt out)",
                added,
            )

    entries: list[tuple[float, StateSpace, _Coupling]] = []
    reference: _Resolved | None = None
    for point in points:
        _check_operating_condition(point.files)
        system, resolved, coupling = _physical_system(
            point.files, states, inputs, outputs
        )
        # A name may resolve (by substring) to a different channel -- or the same
        # channel with different units -- in another point. The family advertises one
        # layout, so every point must resolve the names to the same full descriptions.
        if reference is None:
            reference = resolved
        elif resolved != reference:
            msg = (
                "requested names resolve to different linearization channels across "
                f"operating points: {reference} vs {resolved}"
            )
            raise ValueError(msg)
        entries.append((_scheduling_value(point.files, attr), system, coupling))
    entries.sort(key=lambda entry: entry[0])
    scheduling_points = np.asarray([entry[0] for entry in entries], dtype=np.float64)
    if not np.all(np.isfinite(scheduling_points)):
        msg = f"{scheduling_var} values must all be finite; got {scheduling_points}"
        raise ValueError(msg)
    if scheduling_points.size > 1 and not np.all(np.diff(scheduling_points) > 0.0):
        msg = (
            f"operating points must have distinct, increasing {scheduling_var}; got "
            f"{scheduling_points}"
        )
        raise ValueError(msg)

    couplings = [entry[2] for entry in entries]
    family = LPVFamily(
        scheduling_points=scheduling_points,
        systems=[entry[1] for entry in entries],
        state_names=list(states),
        input_names=list(inputs),
        output_names=list(outputs),
        scheduling_variable=scheduling_var,
        scheduling_unit=_SCHEDULING_UNIT[attr],
        auto_added_states=added,
        dropped_coupling_a=np.asarray([cp.a for cp in couplings], dtype=np.float64),
        dropped_coupling_c=np.asarray([cp.c for cp in couplings], dtype=np.float64),
    )
    _check_dropped_coupling(family, couplings, tolerance)
    return family


def _check_dropped_coupling(
    family: LPVFamily, couplings: Sequence[_Coupling], tolerance: float | None
) -> None:
    """Raise if the family's worst per-point dropped coupling exceeds ``tolerance``."""
    combined = family.dropped_coupling
    worst = int(np.argmax(combined))
    value = float(combined[worst])
    if tolerance is None or value <= tolerance:
        return
    contributors = ", ".join(
        f"{name!r} ({share:.3g})" for name, share in couplings[worst].contributors
    )
    msg = (
        f"the state selection drops {value:.3g} relative coupling at "
        f"{family.scheduling_variable} = {family.scheduling_points[worst]:g} "
        f"{family.scheduling_unit} (A: {couplings[worst].a:.3g}, "
        f"C: {couplings[worst].c:.3g}), above max_dropped_coupling={tolerance:g}; "
        f"the largest contributing dropped states are {contributors}. Select those "
        "states too, or pass max_dropped_coupling=None to accept the truncation"
    )
    raise ValueError(msg)
