# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Operating-point ROM family with consistently-tracked modes (LPV packaging).

A turbine is linearized at many operating points (wind speed, rotor speed, ...), and the
digital twin wants one reduced model *per* point so it can schedule the estimator on the
operating condition. Reducing each point independently is not enough: the reduction
ranks modes by their residue, so the order — and the meaning of "mode 2" — changes
from point to point as modes veer and cross. This module reduces every point and then
**tracks the modes across the family**, so a given index is the same physical mode at
every operating point: a proper linear-parameter-varying (LPV) packaging of the ROM.

Modes are matched between adjacent operating points by the Modal Assurance Criterion of
their **output residues** (how each mode appears at the outputs — the same quantity the
reduction ranks on), with a frequency-proximity tie-breaker, via an optimal (Hungarian)
assignment. Each tracked mode carries the MAC value against the previous point, so a low
value flags where the tracking is uncertain (a crossing or veering region) — the hook
the operating-envelope-consistency analysis needs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from vane.modal.mac import compute_mac, match_modes
from vane.sysid.rom import _resolve_input, _resolve_output, modal_reduction
from vane.sysid.state_space import StateSpace, static_gain

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

__all__ = ["RomFamily", "build_rom_family", "combine_rom_families"]

_log = logging.getLogger(__name__)
# A candidate whose joint transfer score is below this fraction of the strongest is
# numerically undriven by the required pairs and cannot fill a transfer-reserved slot.
_TRANSFER_REL_TOL = float(np.sqrt(np.finfo(np.float64).eps))


@dataclass
class RomFamily:
    """A parameter-indexed family of reduced models with tracked modes.

    Parameters
    ----------
    operating_points : npt.NDArray[np.float64]
        The scheduling-parameter value of each point, shape ``(n_points,)``, in the
        order given.
    systems : list[StateSpace]
        The reduced model at each operating point, all with the same mode (and state)
        ordering: index ``k`` is the same physical mode across the family.
    frequencies_hz : npt.NDArray[np.float64]
        Natural frequency of each tracked mode, shape ``(n_points, n_modes)``.
    damping_ratios : npt.NDArray[np.float64]
        **Physical** damping ratio of each tracked mode, shape ``(n_points, n_modes)``:
        the damping of the linearization's own eigenvalue, negative for an unstable
        mode, even where the stored realization reflected that pole (see
        ``stabilized``).
    tracking_mac : npt.NDArray[np.float64]
        The MAC of each tracked mode against the previous operating point, shape
        ``(n_points, n_modes)``; the first row is all ones (the reference). A low value
        marks a mode whose tracking across that step is uncertain.
    stabilized : npt.NDArray[np.bool_], optional
        ``True`` where the stored realization in ``systems`` reflected that mode's
        unstable pole into the stable half-plane, shape ``(n_points, n_modes)``. For
        such a mode the realized ``A`` block has damping ``-damping_ratios`` rather
        than the physical value. Defaults to all ``False`` (for hand-built families).
    """

    operating_points: npt.NDArray[np.float64]
    systems: list[StateSpace]
    frequencies_hz: npt.NDArray[np.float64]
    damping_ratios: npt.NDArray[np.float64]
    tracking_mac: npt.NDArray[np.float64]
    stabilized: npt.NDArray[np.bool_] = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.bool_)
    )

    def __post_init__(self) -> None:
        """Default an omitted ``stabilized`` mask to all ``False`` (modal shape)."""
        shape = np.shape(self.frequencies_hz)
        if np.size(self.stabilized) == 0 and np.shape(self.stabilized) != shape:
            self.stabilized = np.zeros(shape, dtype=np.bool_)


def _check_comparable_systems(systems: Sequence[StateSpace]) -> None:
    """Reject operating-point systems with mismatched I/O layout or time domain.

    Mode tracking compares **output residues** and LPV interpolation assembles
    like-for-like matrices, so every system must share the same output layout, input
    dimension, and time domain (continuous vs discrete). A mismatch would silently track
    the wrong modes or interpolate incompatible models.

    Raises
    ------
    ValueError
        If any system lacks a ``C`` matrix, or the output count/names, input count, or
        ``dt`` differ across the family.
    """
    reference = systems[0]
    if reference.c is None:
        msg = "system 0 has no C matrix; build_rom_family needs output residues"
        raise ValueError(msg)
    n_outputs = reference.c.shape[0]
    # Input count follows the StateSpace convention (B's width, or D's width for a
    # feedthrough-only model, or 0). A family mixing input dimensions (incl. B=None vs
    # an input matrix, or differing D widths) is inconsistent for LPV use.
    ref_inputs = reference.n_inputs
    for index, system in enumerate(systems):
        if system.c is None:
            msg = f"system {index} has no C matrix"
            raise ValueError(msg)
        if system.c.shape[0] != n_outputs:
            msg = (
                f"all systems must have the same number of outputs; system {index} has "
                f"{system.c.shape[0]}, expected {n_outputs}"
            )
            raise ValueError(msg)
        # Compare names directly (no short-circuit on emptiness): a family that mixes a
        # named system with an unnamed one cannot be confirmed to share an output
        # layout, so it is rejected; all-unnamed and all-identically-named pass.
        if system.output_names != reference.output_names:
            msg = (
                f"all systems must share the same output layout; system {index} output "
                "names differ from system 0 (a mix of named and unnamed is rejected)"
            )
            raise ValueError(msg)
        # The MBC coordinate labels (collective/cosine/sine vs plain rows) are part of
        # the output identity: same names but different MBC coordinates are not the same
        # physical outputs, so row-by-row residue matching would compare unlike rows.
        if system.output_mbc_coordinates != reference.output_mbc_coordinates:
            msg = (
                f"all systems must share the same output MBC coordinates; system "
                f"{index} differs from system 0 (a mix of MBC and plain rows fails)"
            )
            raise ValueError(msg)
        sys_inputs = system.n_inputs
        if sys_inputs != ref_inputs:
            msg = (
                f"all systems must have the same number of inputs (from B, or D for a "
                f"feedthrough-only model); system {index} has {sys_inputs}, expected "
                f"{ref_inputs}"
            )
            raise ValueError(msg)
        # Input names define the input layout (as output names do for outputs): a mix of
        # named and unnamed, or differing names, is an inconsistent layout.
        if system.input_names != reference.input_names:
            msg = (
                f"all systems must share the same input layout; system {index} input "
                "names differ from system 0 (a mix of named and unnamed is rejected)"
            )
            raise ValueError(msg)
        if system.dt != reference.dt:
            msg = (
                f"all systems must be the same time domain; system {index} has "
                f"dt={system.dt!r}, expected dt={reference.dt!r}"
            )
            raise ValueError(msg)


def _check_schedule(operating_points: Sequence[float]) -> npt.NDArray[np.float64]:
    """Return the schedule as an array, requiring finite, strictly monotonic values.

    The scheduling parameter must be finite and strictly monotonic (so the points are
    distinct and the LPV interpolation order is well-defined); a repeated or
    out-of-order value would make interpolation ambiguous.

    Raises
    ------
    ValueError
        If any value is non-finite, or the sequence is not strictly increasing or
        strictly decreasing.
    """
    points = np.asarray(operating_points, dtype=np.float64)
    if points.ndim != 1:
        msg = f"operating_points must be a 1-D sequence, got shape {points.shape}"
        raise ValueError(msg)
    if not np.all(np.isfinite(points)):
        msg = "operating_points must all be finite"
        raise ValueError(msg)
    if points.size > 1:
        diffs = np.diff(points)
        if not (np.all(diffs > 0.0) or np.all(diffs < 0.0)):
            msg = (
                "operating_points must be strictly monotonic (sorted ascending or "
                "descending) so the schedule is distinct and well-ordered; got "
                f"{points.tolist()}"
            )
            raise ValueError(msg)
    return points


def build_rom_family(
    systems: Sequence[StateSpace],
    operating_points: Sequence[float],
    *,
    n_modes: int,
    frequency_band_hz: tuple[float, float] | None = None,
    rank_output: int | str | None = None,
    rank_outputs: Sequence[int | str] | None = None,
    required_transfer_pairs: Sequence[tuple[int | str, int | str]] | None = None,
    n_required_transfer_modes: int = 0,
    frequency_weight: float = 0.5,
    match_dc_gain: bool = False,
    stabilize: bool = True,
) -> RomFamily:
    """Reduce a system at each operating point and track the modes across the family.

    The reference point's top-``n_modes`` modes (by residue) define the tracked set. At
    each later point the system is reduced to **all** candidate modes and the tracked
    modes are matched against that full set by output-residue MAC (with a frequency
    tie-breaker), so a tracked mode is recovered even when it drops below the
    top-``n_modes`` residue rank at that point; the matched modes are then selected and
    reordered to keep mode identity consistent.

    Parameters
    ----------
    systems : Sequence[StateSpace]
        One **continuous-time** linearization per operating point, with a ``C`` matrix
        and the same outputs throughout.
    operating_points : Sequence[float]
        The scheduling-parameter value of each system (same length as ``systems``).
    n_modes : int
        Number of modes to retain and track at every point. Each point must yield at
        least this many underdamped modes.
    frequency_band_hz : tuple[float, float] or None, optional
        Restrict the retained modes to this natural-frequency band (Hz).
    rank_output : int or str or None, optional
        Rank modes by their residue in this output when reducing (see
        :func:`~vane.sysid.modal_reduction`).
    rank_outputs : Sequence[int or str] or None, optional
        Rank modes by the equal-weight, unit-normalized joint residue of these outputs
        (see :func:`~vane.sysid.modal_reduction`). Mutually exclusive with
        ``rank_output``.
    required_transfer_pairs : sequence of tuple(output, input) or None, optional
        Observer input-output pairs whose strongest transfer-residue modes are reserved
        in the reference modal set (see :func:`~vane.sysid.modal_reduction`).
    n_required_transfer_modes : int, optional
        Number of the ``n_modes`` slots reserved for those transfer pairs. The reserved
        set is fixed at the reference point and then tracked by mode identity (joint
        MAC/frequency matching over all candidates), not re-selected per point, so a
        reshuffled transfer ranking cannot swap physical modes; a reserved mode is only
        barred from matching a candidate with negligible transfer residue.
    frequency_weight : float, optional
        Strength of the frequency-proximity tie-breaker in ``[0, 1]`` used when matching
        modes between points (default ``0.5``).
    match_dc_gain : bool, optional
        After the tracked modal subset is selected, correct each reduced feedthrough so
        its zero-frequency gain exactly equals its corresponding full system. This is
        applied after truncation (not to the all-candidate intermediate model), thereby
        preserving equilibrium stiffness for the actual stored family. Default
        ``False``.
    stabilize : bool, optional
        Reflect a retained unstable pole into the stable half-plane in the stored
        realizations (see :func:`~vane.sysid.modal_reduction`), so each member is a
        stable estimator basis. Default ``True``. The reported
        :attr:`RomFamily.damping_ratios` are always the **physical** (pre-reflection)
        values, :attr:`RomFamily.stabilized` marks every reflected mode, and a warning
        is logged when any mode was reflected. ``False`` stores the unstable poles.

    Returns
    -------
    RomFamily
        The reduced models in a consistent mode order, with per-point frequencies,
        physical damping ratios, the per-step tracking MAC, and the reflection mask.

    Raises
    ------
    ValueError
        If ``systems`` is empty, its length differs from ``operating_points``,
        ``n_modes`` is not positive, the systems do not share the same output layout,
        input dimension, and time domain, ``operating_points`` are not finite and
        strictly monotonic, or a point yields fewer than ``n_modes`` modes.
    """
    if len(systems) == 0:
        msg = "systems must contain at least one operating point"
        raise ValueError(msg)
    if len(systems) != len(operating_points):
        msg = (
            f"systems and operating_points must have the same length, got "
            f"{len(systems)} and {len(operating_points)}"
        )
        raise ValueError(msg)
    if n_modes <= 0:
        msg = f"n_modes must be positive, got {n_modes}"
        raise ValueError(msg)
    if n_required_transfer_modes > n_modes:
        msg = (
            f"n_required_transfer_modes={n_required_transfer_modes} exceeds "
            f"n_modes={n_modes}"
        )
        raise ValueError(msg)
    if not 0.0 <= frequency_weight <= 1.0:  # also rejects NaN
        msg = f"frequency_weight must be in [0, 1], got {frequency_weight}"
        raise ValueError(msg)
    _check_comparable_systems(systems)
    points = _check_schedule(operating_points)

    reduced_systems: list[StateSpace] = []
    freqs: list[npt.NDArray[np.float64]] = []
    damps: list[npt.NDArray[np.float64]] = []
    macs: list[npt.NDArray[np.float64]] = []
    reflected: list[npt.NDArray[np.bool_]] = []

    prev_residues: npt.NDArray[np.complex128] | None = None
    prev_freqs: npt.NDArray[np.float64] | None = None
    for index, system in enumerate(systems):
        # Reduce to ALL candidate modes (not just the top n_modes): a mode tracked from
        # the reference may fall below the top-n_modes residue rank at a later point, so
        # it must still be available to the assignment before any truncation.
        reduction = modal_reduction(
            system,
            n_modes=None,
            frequency_band_hz=frequency_band_hz,
            rank_output=rank_output,
            rank_outputs=rank_outputs,
            required_transfer_pairs=required_transfer_pairs,
            n_required_transfer_modes=n_required_transfer_modes,
            stabilize=stabilize,
        )
        n_candidates = reduction.frequencies_hz.shape[0]
        if n_candidates < n_modes:
            msg = (
                f"operating point {index} yields only {n_candidates} modes, "
                f"fewer than n_modes={n_modes}"
            )
            raise ValueError(msg)

        # Fix each mode's real scale (as the phase is fixed below): the eigenvector
        # normalization makes it depend on the arbitrary full-model basis, and blending
        # B/C entrywise across members of different scale would distort the gain.
        candidates = _normalize_modal_scales(reduction.system)
        residues = _output_residues(candidates)
        candidate_freqs = reduction.frequencies_hz
        if prev_residues is None:
            # Reference: track the top-n_modes (modal_reduction ranks by residue).
            order = list(range(n_modes))
            mac_row: npt.NDArray[np.float64] = np.ones(n_modes, dtype=np.float64)
        else:
            # Match ALL previously tracked modes jointly against the FULL candidate
            # set by output-residue MAC with the frequency tie-breaker, so a tracked
            # mode keeps its physical identity (and is recovered even outside this
            # point's top-n_modes) however the residue or transfer ranking reshuffles.
            # The transfer-reserved slots were fixed at the reference and follow their
            # modes by identity; the only extra constraint is that a reserved mode
            # cannot be matched to a candidate with negligible transfer residue in the
            # required pairs -- output-only MAC could otherwise swap an observer-
            # critical mode for a co-shaped structural mode the input does not drive.
            assert prev_freqs is not None
            mac = compute_mac(prev_residues, residues)
            reserved = n_required_transfer_modes
            forbidden = np.zeros(mac.shape, dtype=np.bool_)
            if reserved:
                assert required_transfer_pairs is not None  # validated by the reduction
                eligible = _transfer_eligible(candidates, required_transfer_pairs)
                if int(np.count_nonzero(eligible)) < reserved:
                    msg = (
                        f"operating point {index} has only "
                        f"{int(np.count_nonzero(eligible))} candidate modes with a "
                        "resolved transfer residue in required_transfer_pairs, fewer "
                        f"than n_required_transfer_modes={reserved}"
                    )
                    raise ValueError(msg)
                # A hard constraint, not a penalty: a reserved (observer-critical) mode
                # may never be assigned a candidate the required inputs do not drive.
                forbidden[:reserved, ~eligible] = True
            pairs = match_modes(
                mac, prev_freqs, candidate_freqs, frequency_weight, forbidden=forbidden
            )
            order = [candidate for _, candidate in pairs]
            mac_row = np.array(
                [mac[ref, candidate] for ref, candidate in pairs], dtype=np.float64
            )

        reordered = _reorder_modes(candidates, order)
        reordered_residues = residues[:, order]
        if prev_residues is None:
            aligned, aligned_residues = reordered, reordered_residues
        else:
            # The eigenvector phase np.linalg.eig assigns is arbitrary, so the same mode
            # can have a sign/rotation flip in its 2-state B/C block between points (the
            # MAC is phase-invariant and would not see it). Rotate each mode's block so
            # its residue aligns with the previous point — keeping the modal coordinates
            # continuous for LPV interpolation and cross-point estimator state.
            thetas = np.array(
                [
                    -np.angle(np.vdot(prev_residues[:, k], reordered_residues[:, k]))
                    for k in range(n_modes)
                ],
                dtype=np.float64,
            )
            aligned = _align_modal_phases(reordered, thetas)
            aligned_residues = _output_residues(aligned)

        if match_dc_gain:
            aligned = _match_dc_gain(system, aligned)
        reduced_systems.append(aligned)
        # ModalReduction reports the realized (post-reflection) damping; the family
        # reports the physical value, so undo the sign flip of every reflected mode.
        stabilized_mask = (
            np.zeros(n_candidates, dtype=np.bool_)
            if reduction.stabilized is None
            else np.asarray(reduction.stabilized, dtype=np.bool_)
        )[order]
        realized_damping = reduction.damping_ratios[order]
        freqs.append(candidate_freqs[order])
        damps.append(np.where(stabilized_mask, -realized_damping, realized_damping))
        macs.append(mac_row)
        reflected.append(stabilized_mask)
        prev_residues = aligned_residues
        prev_freqs = candidate_freqs[order]

    stabilized_all = np.vstack(reflected)
    if np.any(stabilized_all):
        flagged = [
            f"point {int(point)} mode {int(mode)}"
            for point, mode in np.argwhere(stabilized_all)
        ]
        _log.warning(
            "build_rom_family reflected %d unstable tracked mode(s) into the stable "
            "half-plane in the stored realizations (%s); RomFamily.damping_ratios "
            "reports their physical (negative) damping and RomFamily.stabilized marks "
            "them",
            len(flagged),
            ", ".join(flagged),
        )
    return RomFamily(
        operating_points=points,
        systems=reduced_systems,
        frequencies_hz=np.vstack(freqs),
        damping_ratios=np.vstack(damps),
        tracking_mac=np.vstack(macs),
        stabilized=stabilized_all,
    )


def _match_dc_gain(full: StateSpace, reduced: StateSpace) -> StateSpace:
    """Return ``reduced`` with feedthrough corrected to the full equilibrium gain."""
    if reduced.b is None or reduced.c is None:
        msg = "match_dc_gain requires systems with B and C matrices"
        raise ValueError(msg)
    try:
        reduced_equilibrium = np.linalg.solve(reduced.a, reduced.b)
    except np.linalg.LinAlgError as exc:
        msg = "match_dc_gain cannot correct a reduced system with singular A"
        raise ValueError(msg) from exc
    return StateSpace(
        a=reduced.a,
        b=reduced.b,
        c=reduced.c,
        d=np.asarray(
            static_gain(full) + reduced.c @ reduced_equilibrium,
            dtype=np.float64,
        ),
        state_names=list(reduced.state_names),
        input_names=list(reduced.input_names),
        output_names=list(reduced.output_names),
        dt=reduced.dt,
        output_mbc_coordinates=list(reduced.output_mbc_coordinates),
    )


def combine_rom_families(
    base: RomFamily,
    enrichment: RomFamily,
    *,
    duplicate_frequency_tolerance_hz: float = 1.0e-8,
    enrichment_outputs: Sequence[int | str] | None = None,
) -> RomFamily:
    """Compose two disjoint tracked modal families without re-reducing either.

    The base and enrichment must use the same operating-point schedule and I/O layout.
    Their modal state blocks are concatenated, their output contributions are summed
    through a horizontal ``C`` concatenation, and the shared feedthrough is retained
    once. A frequency-separation check rejects a mode present in both families, which
    would otherwise double-count its transfer residue.

    Parameters
    ----------
    base, enrichment : RomFamily
        Independently tracked, continuous- or discrete-time modal families with the
        same schedule, input/output names, MBC labels, feedthrough, and sample time.
    duplicate_frequency_tolerance_hz : float, optional
        Maximum absolute per-point frequency difference considered a duplicate.
        Default ``1e-8`` Hz.
    enrichment_outputs : sequence of int or str or None, optional
        Output rows to which the enrichment is allowed to contribute. ``None`` keeps
        every enrichment output. A declared subset zeros all other enrichment ``C``
        rows while leaving the base family untouched; this supports separated observer
        blocks, such as drivetrain modes used only by a rotor-speed measurement.

    Returns
    -------
    RomFamily
        A block-composed family with base modes first and enrichment modes second.

    Raises
    ------
    ValueError
        If the tolerance is invalid, schedules or I/O layouts differ, feedthroughs do
        not match exactly, or a duplicate modal branch is detected.
    """
    if (
        not np.isfinite(duplicate_frequency_tolerance_hz)
        or duplicate_frequency_tolerance_hz < 0.0
    ):
        msg = "duplicate_frequency_tolerance_hz must be finite and nonnegative"
        raise ValueError(msg)
    if not np.array_equal(base.operating_points, enrichment.operating_points):
        msg = "ROM families must have identical operating-point schedules"
        raise ValueError(msg)
    if len(base.systems) != len(enrichment.systems) or not base.systems:
        msg = "ROM families must contain the same nonzero number of systems"
        raise ValueError(msg)
    if base.frequencies_hz.shape[0] != len(base.systems) or (
        enrichment.frequencies_hz.shape[0] != len(enrichment.systems)
    ):
        msg = "ROM frequency metadata must align with the family systems"
        raise ValueError(msg)
    separation = np.abs(
        base.frequencies_hz[:, :, None] - enrichment.frequencies_hz[:, None, :]
    )
    duplicate = np.argwhere(separation <= duplicate_frequency_tolerance_hz)
    if duplicate.size:
        point, base_mode, enrichment_mode = duplicate[0]
        msg = (
            "ROM enrichment duplicates a base mode at operating-point index "
            f"{point}: base mode {base_mode}, enrichment mode {enrichment_mode}"
        )
        raise ValueError(msg)
    enrichment_rows = _resolve_enrichment_outputs(base.systems[0], enrichment_outputs)
    systems = [
        _combine_rom_systems(
            base_system,
            enrichment_system,
            enrichment_rows=enrichment_rows,
        )
        for base_system, enrichment_system in zip(
            base.systems, enrichment.systems, strict=True
        )
    ]
    return RomFamily(
        operating_points=np.array(base.operating_points, copy=True),
        systems=systems,
        frequencies_hz=np.hstack([base.frequencies_hz, enrichment.frequencies_hz]),
        damping_ratios=np.hstack([base.damping_ratios, enrichment.damping_ratios]),
        tracking_mac=np.hstack([base.tracking_mac, enrichment.tracking_mac]),
        stabilized=np.hstack([base.stabilized, enrichment.stabilized]),
    )


def _resolve_enrichment_outputs(
    system: StateSpace, requested: Sequence[int | str] | None
) -> tuple[int, ...] | None:
    """Resolve an optional, distinct set of enrichment output rows."""
    if requested is None:
        return None
    selections = list(requested)
    if not selections:
        msg = "enrichment_outputs must contain at least one output"
        raise ValueError(msg)
    rows = []
    for output in selections:
        if isinstance(output, (int, np.integer)):
            row = int(output)
            if not 0 <= row < system.n_outputs:
                msg = f"enrichment output index {row} is out of range"
                raise ValueError(msg)
        elif isinstance(output, str):
            exact = [
                index
                for index, name in enumerate(system.output_names)
                if name == output
            ]
            contains = [
                index
                for index, name in enumerate(system.output_names)
                if output in name
            ]
            matches = exact or contains
            if len(matches) != 1:
                msg = f"enrichment output {output!r} resolved to {matches}"
                raise ValueError(msg)
            row = matches[0]
        else:
            msg = f"enrichment output must be int or str, got {type(output).__name__}"
            raise ValueError(msg)  # noqa: TRY004
        rows.append(row)
    if len(set(rows)) != len(rows):
        msg = "enrichment_outputs must resolve to distinct rows"
        raise ValueError(msg)
    return tuple(rows)


def _combine_rom_systems(
    base: StateSpace,
    enrichment: StateSpace,
    *,
    enrichment_rows: tuple[int, ...] | None,
) -> StateSpace:
    """Return the disjoint modal union of two systems with one shared feedthrough."""
    if (
        base.input_names != enrichment.input_names
        or base.output_names != enrichment.output_names
        or base.output_mbc_coordinates != enrichment.output_mbc_coordinates
        or base.dt != enrichment.dt
    ):
        msg = "ROM enrichment systems must share the same I/O layout and time domain"
        raise ValueError(msg)
    if base.c is None or enrichment.c is None:
        msg = "ROM enrichment systems must both have C matrices"
        raise ValueError(msg)
    if (base.b is None) != (enrichment.b is None):
        msg = "ROM enrichment systems must both have B matrices or both omit B"
        raise ValueError(msg)
    if (base.d is None) != (enrichment.d is None) or (
        base.d is not None
        and enrichment.d is not None
        and not np.array_equal(base.d, enrichment.d)
    ):
        msg = "ROM enrichment systems must have identical feedthrough matrices"
        raise ValueError(msg)
    n_base = base.n_states
    n_enrichment = enrichment.n_states
    a = np.zeros((n_base + n_enrichment, n_base + n_enrichment))
    a[:n_base, :n_base] = base.a
    a[n_base:, n_base:] = enrichment.a
    b = (
        None
        if base.b is None or enrichment.b is None
        else np.vstack([base.b, enrichment.b])
    )
    enrichment_c = np.array(enrichment.c, copy=True)
    if enrichment_rows is not None:
        mask = np.ones(enrichment_c.shape[0], dtype=np.bool_)
        mask[list(enrichment_rows)] = False
        enrichment_c[mask] = 0.0
    return StateSpace(
        a=a,
        b=b,
        c=np.hstack([base.c, enrichment_c]),
        d=None if base.d is None else np.array(base.d, copy=True),
        state_names=[
            *[f"base:{name}" for name in base.state_names],
            *[f"enrichment:{name}" for name in enrichment.state_names],
        ],
        input_names=list(base.input_names),
        output_names=list(base.output_names),
        output_mbc_coordinates=list(base.output_mbc_coordinates),
        dt=base.dt,
    )


def _normalize_modal_scales(system: StateSpace) -> StateSpace:
    r"""Rescale each 2-state modal block so its input and output coordinates balance.

    A real scaling ``s`` of mode ``k``'s two states leaves its ``A`` block and transfer
    residue unchanged but maps ``b_k -> s b_k`` and ``c_k -> c_k / s`` (complex block
    coordinates ``b_k = B[2k] + i B[2k+1]`` and ``c_k = C[:, 2k] - i C[:, 2k+1]``).
    Choosing ``s = sqrt(|c_k| / |b_k|)`` gives ``|b_k| = |c_k| = sqrt(|b_k||c_k|)``,
    which depends only on the mode's transfer residue, not on the arbitrary scale of the
    full-model eigenvector. Members whose modes differ only by a state-space similarity
    then store identical blocks (after phase alignment), so an entrywise LPV blend of
    ``B``/``C`` has no spurious gain error. A mode without resolved input or output
    coordinates (no ``B``, or a zero ``b_k``/``c_k``) keeps its scale.
    """
    b, c = system.b, system.c
    if b is None or c is None:
        return system
    n_modes = system.n_states // 2
    scales = np.ones(system.n_states, dtype=np.float64)
    tiny = float(np.finfo(np.float64).tiny)
    for k in range(n_modes):
        b_norm = float(np.linalg.norm(b[2 * k : 2 * k + 2]))
        c_norm = float(np.linalg.norm(c[:, 2 * k : 2 * k + 2]))
        if b_norm > tiny and c_norm > tiny:
            scales[2 * k : 2 * k + 2] = np.sqrt(c_norm / b_norm)
    return StateSpace(
        a=system.a,
        b=np.asarray(b * scales[:, np.newaxis], dtype=np.float64),
        c=np.asarray(c / scales[np.newaxis, :], dtype=np.float64),
        d=system.d,
        state_names=list(system.state_names),
        input_names=list(system.input_names),
        output_names=list(system.output_names),
        output_mbc_coordinates=list(system.output_mbc_coordinates),
        dt=system.dt,
    )


def _transfer_eligible(
    system: StateSpace, pairs: Sequence[tuple[int | str, int | str]]
) -> npt.NDArray[np.bool_]:
    """Return which modes carry a resolved transfer residue in the required pairs.

    In the real modal realization, mode ``k``'s transfer residue for output ``i`` and
    input ``j`` is proportional to ``|c_k[i]| |b_k[j]|`` with the complex block
    coordinates ``c_k = C[:, 2k] - i C[:, 2k+1]`` and ``b_k = B[2k] + i B[2k+1]``. Each
    pair is normalized across the modes (as the reduction's transfer ranking does), and
    a mode is eligible when its joint score exceeds ``sqrt(eps)`` of the largest.
    """
    b, c = system.b, system.c
    assert b is not None and c is not None  # the reduction validated both
    n_modes = system.n_states // 2
    c_modal = c[:, 0::2] - 1j * c[:, 1::2]  # (n_outputs, n_modes)
    b_modal = b[0::2] + 1j * b[1::2]  # (n_modes, n_inputs)
    scores: npt.NDArray[np.float64] = np.zeros(n_modes, dtype=np.float64)
    for output, input_channel in pairs:
        row = _resolve_output(system, output)
        column = _resolve_input(system, input_channel)
        transfer = np.abs(c_modal[row]) * np.abs(b_modal[:, column])
        scale = float(np.linalg.norm(transfer))
        if scale > 0.0:
            scores += (transfer / scale) ** 2
    scores = np.sqrt(scores)
    peak = float(scores.max()) if scores.size else 0.0
    return np.asarray(scores > _TRANSFER_REL_TOL * peak, dtype=np.bool_)


def _output_residues(system: StateSpace) -> npt.NDArray[np.complex128]:
    """Return the complex output residue of each mode, shape ``(n_outputs, n_modes)``.

    In the real modal realization a mode occupies two columns of ``C``,
    ``[2 Re(C v), -2 Im(C v)]``; the complex residue ``C v`` (up to a real factor that
    cancels in the MAC) is recovered as ``C[:, 2k] - i C[:, 2k+1]``.
    """
    c = system.c
    assert c is not None  # the family builder only reduces systems with outputs
    n_modes = c.shape[1] // 2
    columns = [c[:, 2 * k] - 1j * c[:, 2 * k + 1] for k in range(n_modes)]
    return np.asarray(np.stack(columns, axis=1), dtype=np.complex128)


def _align_modal_phases(
    system: StateSpace, thetas: npt.NDArray[np.float64]
) -> StateSpace:
    """Rotate each 2-state modal block by its angle to align the modal phase.

    A rotation ``R(theta)`` of a mode's two states leaves the block-diagonal ``A`` block
    invariant (a rotation commutes with ``sigma I + omega J``) and multiplies the mode's
    complex residue by ``e^{i theta}``, realigning ``B``/``C`` without changing ``A``.
    """
    n_modes = thetas.shape[0]
    transform = np.eye(2 * n_modes, dtype=np.float64)
    for k in range(n_modes):
        cos_t, sin_t = np.cos(thetas[k]), np.sin(thetas[k])
        transform[2 * k : 2 * k + 2, 2 * k : 2 * k + 2] = [
            [cos_t, -sin_t],
            [sin_t, cos_t],
        ]
    # A commutes with the block rotations, so it is unchanged; B and C rotate.
    b = None if system.b is None else transform.T @ system.b
    c = None if system.c is None else system.c @ transform
    return StateSpace(
        a=system.a,
        b=None if b is None else np.asarray(b, dtype=np.float64),
        c=None if c is None else np.asarray(c, dtype=np.float64),
        d=system.d,
        state_names=list(system.state_names),
        input_names=list(system.input_names),
        output_names=list(system.output_names),
        output_mbc_coordinates=list(system.output_mbc_coordinates),
        dt=system.dt,
    )


def _reorder_modes(system: StateSpace, order: Sequence[int]) -> StateSpace:
    """Return ``system`` with its 2-state modal blocks reordered by ``order``."""
    state_index = [2 * mode + part for mode in order for part in (0, 1)]
    a = system.a[np.ix_(state_index, state_index)]
    b = None if system.b is None else system.b[state_index]
    c = None if system.c is None else system.c[:, state_index]
    names = [f"modal_{k}_{part}" for k in range(len(order)) for part in ("re", "im")]
    return StateSpace(
        a=np.asarray(a, dtype=np.float64),
        b=None if b is None else np.asarray(b, dtype=np.float64),
        c=None if c is None else np.asarray(c, dtype=np.float64),
        d=system.d,
        state_names=names,
        input_names=list(system.input_names),
        output_names=list(system.output_names),
        output_mbc_coordinates=list(system.output_mbc_coordinates),
        dt=system.dt,
    )
