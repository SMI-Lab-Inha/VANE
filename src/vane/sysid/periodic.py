# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Rotor-azimuth-periodic ROMs in one fixed modal coordinate system.

Reducing every azimuth independently produces arbitrarily re-phased modal states that
cannot be scheduled safely.  This module instead takes one accepted averaged modal
reduction and projects every retained per-azimuth MBC Jacobian through that same left/
right basis.  The resulting matrices form a linear time-periodic (LTP) family whose
state has a consistent meaning around the full revolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from vane.sysid.state_space import StateSpace

if TYPE_CHECKING:
    import numpy.typing as npt

    from vane.io.mbc_transform import MBCResult
    from vane.sysid.rom import ModalReduction

__all__ = ["PeriodicStateSpace", "periodic_state_space_from_mbc"]


@dataclass(frozen=True)
class PeriodicStateSpace:
    """A full-revolution matrix family in a fixed reduced-state basis."""

    azimuths_deg: npt.NDArray[np.float64]
    systems: tuple[StateSpace, ...]

    def __post_init__(self) -> None:
        """Validate complete periodic members with one physical channel layout."""
        azimuths = np.asarray(self.azimuths_deg, dtype=np.float64)
        if azimuths.ndim != 1 or azimuths.size < 2:
            msg = "azimuths_deg must contain at least two samples"
            raise ValueError(msg)
        if not np.all(np.isfinite(azimuths)):
            msg = "azimuths_deg must be finite"
            raise ValueError(msg)
        if np.any(azimuths < 0.0) or np.any(azimuths >= 360.0):
            msg = "azimuths_deg must lie in [0, 360)"
            raise ValueError(msg)
        if np.any(np.diff(azimuths) <= 0.0):
            msg = "azimuths_deg must be strictly increasing"
            raise ValueError(msg)
        if len(self.systems) != azimuths.size:
            msg = "one state-space member is required per azimuth"
            raise ValueError(msg)
        reference = self.systems[0]
        reference_layout = (
            reference.state_names,
            reference.input_names,
            reference.output_names,
            reference.output_mbc_coordinates,
        )
        for index, system in enumerate(self.systems[1:], start=1):
            shapes = (
                system.a.shape,
                _shape(system.b),
                _shape(system.c),
                _shape(system.d),
            )
            reference_shapes = (
                reference.a.shape,
                _shape(reference.b),
                _shape(reference.c),
                _shape(reference.d),
            )
            if shapes != reference_shapes:
                msg = f"periodic member {index} shapes {shapes} != {reference_shapes}"
                raise ValueError(msg)
            layout = (
                system.state_names,
                system.input_names,
                system.output_names,
                system.output_mbc_coordinates,
            )
            if layout != reference_layout:
                msg = f"periodic member {index} channel layout differs from member 0"
                raise ValueError(msg)
            if system.dt != reference.dt:
                msg = "periodic members must share one time domain"
                raise ValueError(msg)

    def at(self, azimuth_deg: float) -> StateSpace:
        """Linearly interpolate the periodic family, including the 360/0 seam."""
        angle = float(azimuth_deg)
        if not np.isfinite(angle):
            msg = "azimuth_deg must be finite"
            raise ValueError(msg)
        query = angle % 360.0
        upper = int(np.searchsorted(self.azimuths_deg, query, side="right"))
        if upper == len(self.systems):
            lower = upper - 1
            upper = 0
            low_angle = float(self.azimuths_deg[lower])
            high_angle = float(self.azimuths_deg[upper] + 360.0)
        else:
            lower = (upper - 1) % len(self.systems)
            low_angle = float(self.azimuths_deg[lower])
            high_angle = float(self.azimuths_deg[upper])
            if upper == 0:
                low_angle -= 360.0
        weight = (query - low_angle) / (high_angle - low_angle)
        low = self.systems[lower]
        high = self.systems[upper]
        return StateSpace(
            a=_blend(low.a, high.a, weight),
            b=_blend_optional(low.b, high.b, weight),
            c=_blend_optional(low.c, high.c, weight),
            d=_blend_optional(low.d, high.d, weight),
            state_names=list(low.state_names),
            input_names=list(low.input_names),
            output_names=list(low.output_names),
            dt=low.dt,
            output_mbc_coordinates=list(low.output_mbc_coordinates),
        )

    def discretized(self, dt: float) -> PeriodicStateSpace:
        """Exactly discretize every tabulated member before online interpolation."""
        return PeriodicStateSpace(
            azimuths_deg=np.array(self.azimuths_deg, copy=True),
            systems=tuple(system.discretized(dt) for system in self.systems),
        )


def _shape(matrix: npt.NDArray[np.float64] | None) -> tuple[int, int] | None:
    return None if matrix is None else matrix.shape


def _blend(
    lower: npt.NDArray[np.float64],
    upper: npt.NDArray[np.float64],
    weight: float,
) -> npt.NDArray[np.float64]:
    return (1.0 - weight) * lower + weight * upper


def _blend_optional(
    lower: npt.NDArray[np.float64] | None,
    upper: npt.NDArray[np.float64] | None,
    weight: float,
) -> npt.NDArray[np.float64] | None:
    if lower is None and upper is None:
        return None
    if lower is None or upper is None:
        msg = "periodic optional-matrix presence must be consistent"
        raise ValueError(msg)
    return _blend(lower, upper, weight)


def _stack(
    values: list[npt.NDArray[np.float64]] | None,
    *,
    name: str,
) -> npt.NDArray[np.float64] | None:
    if values is None:
        return None
    if not values:
        msg = f"per-azimuth {name} matrices must not be empty"
        raise ValueError(msg)
    result = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        msg = f"per-azimuth {name} matrices must be finite"
        raise ValueError(msg)
    return result


def _channel_map(
    reduced_names: list[str],
    reduced_coordinates: list[str],
    mbc_names: list[str],
    mbc_coordinates: list[str],
    *,
    n_reduced: int,
    n_mbc: int,
    kind: str,
) -> list[int]:
    """Return, per reduced channel, the index of the same channel in the MBC result.

    A channel is identified by its name plus, when either side records them, its MBC
    coordinate label. With both sides named, the reduced layout must be a permutation
    of the MBC layout (a reordered reduction is remapped); when either side is unnamed
    the channels can only be matched by position, so the counts must agree.

    Raises
    ------
    ValueError
        If the channel counts differ, the named layouts are not a permutation of one
        another, or a label is duplicated so the mapping is ambiguous.
    """
    if n_reduced != n_mbc:
        msg = (
            f"the reduction has {n_reduced} {kind}s but the MBC per-azimuth matrices "
            f"have {n_mbc}; the periodic deviations cannot be applied"
        )
        raise ValueError(msg)
    if not reduced_names or not mbc_names:
        return list(range(n_reduced))
    if len(reduced_names) != n_reduced or len(mbc_names) != n_mbc:
        msg = f"{kind} names do not cover every {kind} channel"
        raise ValueError(msg)
    use_coordinates = bool(reduced_coordinates) or bool(mbc_coordinates)
    if use_coordinates and (
        len(reduced_coordinates) != n_reduced or len(mbc_coordinates) != n_mbc
    ):
        msg = (
            f"{kind} MBC coordinates of the reduction and the MBC result do not both "
            f"label every {kind}; the channel identity cannot be confirmed"
        )
        raise ValueError(msg)

    def labels(names: list[str], coordinates: list[str]) -> list[tuple[str, str]]:
        return [
            (name, coordinates[i] if use_coordinates else "")
            for i, name in enumerate(names)
        ]

    reduced_labels = labels(reduced_names, reduced_coordinates)
    mbc_labels = labels(mbc_names, mbc_coordinates)
    if reduced_labels == mbc_labels:
        return list(range(n_reduced))
    if len(set(mbc_labels)) != n_mbc or len(set(reduced_labels)) != n_reduced:
        msg = f"duplicate {kind} labels make the reordered {kind} layout ambiguous"
        raise ValueError(msg)
    if set(reduced_labels) != set(mbc_labels):
        msg = (
            f"the reduction's {kind} layout {reduced_labels} does not match the MBC "
            f"result's {mbc_labels}"
        )
        raise ValueError(msg)
    position = {label: index for index, label in enumerate(mbc_labels)}
    return [position[label] for label in reduced_labels]


def periodic_state_space_from_mbc(
    result: MBCResult,
    reduction: ModalReduction,
) -> PeriodicStateSpace:
    """Project retained per-azimuth MBC matrices through one modal basis.

    The averaged reduced ``A`` is retained as the stable reference and projected
    per-azimuth deviations are added to it.  Thus any intentionally reflected unstable
    poles remain reflected while the measured cyclic Jacobian variation is preserved.

    The ``B``/``C``/``D`` deviations are mapped onto the reduction's channels by input
    name and by output name plus MBC coordinate, so a reduction whose outputs or inputs
    are a reordering of the MBC result's is remapped rather than corrupted. When either
    side is unnamed the channels are matched by position and only the counts are
    checked.

    Parameters
    ----------
    result : MBCResult
        An MBC result built with per-azimuth matrices retained.
    reduction : ModalReduction
        A modal reduction of the result's averaged system, with its projection maps.

    Returns
    -------
    PeriodicStateSpace
        One reduced member per azimuth in the reduction's fixed basis and channel order.

    Raises
    ------
    ValueError
        If required averaged or per-azimuth matrices are missing, dimensions disagree,
        or the reduction's input/output layout is not a permutation of the MBC result's
        input/output descriptions (and output MBC coordinates).
    """
    if result.avg_a is None:
        msg = "MBC result has no averaged A matrix"
        raise ValueError(msg)
    projection = reduction.state_projection
    reconstruction = reduction.state_reconstruction
    if projection is None or reconstruction is None:
        msg = "modal reduction does not retain state projection maps"
        raise ValueError(msg)
    if projection.shape[1] != result.avg_a.shape[0]:
        msg = "modal projection and MBC full-state dimensions do not match"
        raise ValueError(msg)
    per_a = _stack(result.per_azimuth_a, name="A")
    if per_a is None:
        msg = "MBC result did not retain per-azimuth A matrices"
        raise ValueError(msg)
    per_b = _stack(result.per_azimuth_b, name="B")
    per_c = _stack(result.per_azimuth_c, name="C")
    per_d = _stack(result.per_azimuth_d, name="D")
    count = result.azimuths_deg.size
    for name, values in (("A", per_a), ("B", per_b), ("C", per_c), ("D", per_d)):
        if values is not None and values.shape[0] != count:
            msg = f"per-azimuth {name} count does not match azimuths"
            raise ValueError(msg)

    # The per-azimuth deviations are indexed by the MBC result's channel order, the
    # reduced matrices by the reduction's; map each reduced channel to its MBC row or
    # column by name (and output MBC coordinate) so a reordered reduction is not
    # corrupted by a positional addition.
    reduced = reduction.system
    rows: list[int] = []
    if per_c is not None or per_d is not None:
        output_source = per_c if per_c is not None else per_d
        assert output_source is not None
        rows = _channel_map(
            reduced.output_names,
            reduced.output_mbc_coordinates,
            list(result.output_descriptions),
            list(result.output_mbc_coordinates),
            n_reduced=reduced.n_outputs,
            n_mbc=int(output_source.shape[1]),
            kind="output",
        )
    columns: list[int] = []
    if per_b is not None or per_d is not None:
        input_source = per_b if per_b is not None else per_d
        assert input_source is not None
        columns = _channel_map(
            reduced.input_names,
            [],
            list(result.input_descriptions),
            [],
            n_reduced=reduced.n_inputs,
            n_mbc=int(input_source.shape[2]),
            kind="input",
        )

    systems: list[StateSpace] = []
    for index in range(count):
        a = reduced.a + projection @ (per_a[index] - result.avg_a) @ reconstruction
        b = reduced.b
        if per_b is not None:
            if result.avg_b is None or reduced.b is None:
                msg = "per-azimuth B requires averaged and reduced B"
                raise ValueError(msg)
            b = reduced.b + projection @ (per_b[index] - result.avg_b)[:, columns]
        c = reduced.c
        if per_c is not None:
            if result.avg_c is None or reduced.c is None:
                msg = "per-azimuth C requires averaged and reduced C"
                raise ValueError(msg)
            c = reduced.c + (per_c[index] - result.avg_c)[rows] @ reconstruction
        d = reduced.d
        if per_d is not None:
            if result.avg_d is None or reduced.d is None:
                msg = "per-azimuth D requires averaged and reduced D"
                raise ValueError(msg)
            d = reduced.d + (per_d[index] - result.avg_d)[np.ix_(rows, columns)]
        systems.append(
            StateSpace(
                a=np.asarray(a, dtype=np.float64),
                b=None if b is None else np.asarray(b, dtype=np.float64),
                c=None if c is None else np.asarray(c, dtype=np.float64),
                d=None if d is None else np.asarray(d, dtype=np.float64),
                state_names=list(reduction.system.state_names),
                input_names=list(reduction.system.input_names),
                output_names=list(reduction.system.output_names),
                output_mbc_coordinates=list(reduction.system.output_mbc_coordinates),
            )
        )
    return PeriodicStateSpace(
        azimuths_deg=np.asarray(result.azimuths_deg, dtype=np.float64),
        systems=tuple(systems),
    )
