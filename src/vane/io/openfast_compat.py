# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Guarded compatibility corrections for known OpenFAST linearization defects.

The corrections in this module are deliberately signature based.  They do not infer
physics from arbitrary non-zero columns: a correction is applied only when the complete
matrix/name pattern produced by the corresponding OpenFAST implementation is present.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from vane.io.mbc_transform import MBCResult

__all__ = [
    "HydroDynWaveDofSignCorrection",
    "HydroDynWaveInputRemap",
    "StateSpaceExcitationMetadata",
    "apply_hydrodyn_wave_dof_signs",
    "read_state_space_excitation_metadata",
    "remap_hydrodyn_state_space_wave_input",
]

_WAVE = "HD Extended input: wave elevation at platform ref point, m"
_CURRENT = "HD Extended input: horizontal current speed (steady/uniform wind), m/s"
_SHEAR = "HD Extended input: vertical power-law shear exponent, -"
_DIRECTION = "HD Extended input: propagation direction, rad"


@dataclass(frozen=True)
class HydroDynWaveInputRemap:
    """Audit record for the OpenFAST HydroDyn state-space wave-input correction.

    ``source_wave_index`` is the zero column carrying the intended wave label;
    ``source_carrier_index`` is the column into which OpenFAST placed the state-space
    excitation ``B`` vector.  The corrected model swaps only these two descriptions,
    leaving every numeric matrix and operating-point value unchanged.
    """

    source_wave_index: int
    source_carrier_index: int
    wave_b_l2_before: float
    carrier_b_l2_before: float
    carrier_nonzero_rows: tuple[int, ...]
    state_space_signature: str


@dataclass(frozen=True)
class HydroDynWaveDofSignCorrection:
    """Audit record for an explicit ``.ssexctn`` DOF convention correction."""

    wave_input_index: int
    states_per_dof: tuple[int, int, int, int, int, int]
    dof_signs: tuple[int, int, int, int, int, int]
    corrected_rows: tuple[int, ...]


@dataclass(frozen=True)
class StateSpaceExcitationMetadata:
    """Header metadata required to drive a HydroDyn ``.ssexctn`` realization.

    HydroDyn evaluates the incident wave at ``t + time_offset_s``. The offset is part
    of the fitted realization and is therefore a required runtime input convention,
    not a tunable phase correction.
    """

    format_header: str
    wave_heading_deg: float
    time_offset_s: float
    state_count: int
    states_per_dof: tuple[int, int, int, int, int, int]


def read_state_space_excitation_metadata(
    path: str | Path,
) -> StateSpaceExcitationMetadata:
    """Read and validate the five-line header of a HydroDyn ``.ssexctn`` file."""
    source = Path(path)
    try:
        with source.open(encoding="utf-8") as stream:
            lines = [stream.readline().strip() for _ in range(5)]
    except OSError as error:
        msg = f"cannot read state-space excitation file {source}: {error}"
        raise ValueError(msg) from error
    if any(not line for line in lines):
        msg = f"state-space excitation file {source} has an incomplete header"
        raise ValueError(msg)
    header = lines[0]
    if not header.startswith("SS_Excitation_Fitting"):
        msg = f"unrecognized state-space excitation header {header!r}"
        raise ValueError(msg)
    try:
        heading = float(lines[1].split()[0])
        time_offset = float(lines[2].split()[0])
        state_count = int(lines[3].split()[0])
        counts = tuple(int(value) for value in lines[4].split("%")[0].split())
    except (IndexError, ValueError) as error:
        msg = f"cannot decode state-space excitation header in {source}"
        raise ValueError(msg) from error
    if not math.isfinite(heading) or not math.isfinite(time_offset):
        msg = "state-space excitation heading and time offset must be finite"
        raise ValueError(msg)
    if len(counts) != 6:
        msg = "state-space excitation header must declare six DOF state counts"
        raise ValueError(msg)
    if state_count <= 0 or any(count < 0 for count in counts):
        msg = "state-space excitation state counts must be non-negative and nonempty"
        raise ValueError(msg)
    if sum(counts) != state_count:
        msg = (
            "state-space excitation total state count does not match its six-DOF "
            "breakdown"
        )
        raise ValueError(msg)
    states_per_dof = (counts[0], counts[1], counts[2], counts[3], counts[4], counts[5])
    return StateSpaceExcitationMetadata(
        format_header=header,
        wave_heading_deg=heading,
        time_offset_s=time_offset,
        state_count=state_count,
        states_per_dof=states_per_dof,
    )


def apply_hydrodyn_wave_dof_signs(
    result: MBCResult,
    *,
    states_per_dof: tuple[int, int, int, int, int, int],
    dof_signs: tuple[int, int, int, int, int, int],
    nonzero_atol: float = 0.0,
) -> tuple[MBCResult, HydroDynWaveDofSignCorrection]:
    """Apply an explicit wave-excitation DOF sign convention to retained ``B`` data.

    This operation is intentionally not automatic. Callers must derive ``dof_signs``
    from an independent hydrodynamic convention audit (for example the phase of the
    WAMIT ``.3`` coefficients versus the preview-corrected ``.ssexctn`` transfer).
    The function enforces that the averaged and every retained azimuth-resolved wave
    column have the same contiguous, HydroDyn-only state block matching the six-DOF
    state counts in the fitted file.
    """
    if result.avg_b is None:
        msg = "HydroDyn wave DOF correction requires an averaged B matrix"
        raise ValueError(msg)
    if not math.isfinite(nonzero_atol) or nonzero_atol < 0.0:
        msg = "nonzero_atol must be finite and non-negative"
        raise ValueError(msg)
    if any(count < 0 for count in states_per_dof) or sum(states_per_dof) <= 0:
        msg = "states_per_dof must be non-negative and nonempty"
        raise ValueError(msg)
    if any(sign not in (-1, 1) for sign in dof_signs):
        msg = "every wave DOF sign must be either -1 or +1"
        raise ValueError(msg)
    if any(
        count == 0 and sign != 1
        for count, sign in zip(states_per_dof, dof_signs, strict=True)
    ):
        msg = "a zero-state wave DOF must retain sign +1"
        raise ValueError(msg)

    wave = _unique_index(result.input_descriptions, _WAVE)
    b = np.asarray(result.avg_b, dtype=np.float64)
    if not np.all(np.isfinite(b)):
        msg = "averaged B matrix contains non-finite values"
        raise ValueError(msg)
    if len(result.state_descriptions) != b.shape[0]:
        msg = "state descriptions do not align with the averaged B rows"
        raise ValueError(msg)
    expected_count = sum(states_per_dof)
    active = _active_hydrodyn_wave_rows(
        b[:, wave],
        state_descriptions=result.state_descriptions,
        expected_count=expected_count,
        nonzero_atol=nonzero_atol,
        label="averaged wave B column",
    )
    if result.avg_d is not None:
        _require_zero(result.avg_d[:, wave], 0.0, "wave-elevation D column")
    if result.input_operating_point is not None:
        _require_zero(
            result.input_operating_point[[wave]],
            0.0,
            "wave input operating point",
        )

    periodic_b: list[npt.NDArray[np.float64]] | None = None
    validated_periodic: list[npt.NDArray[np.float64]] = []
    if result.per_azimuth_b is not None:
        for index, matrix in enumerate(result.per_azimuth_b):
            values = np.asarray(matrix, dtype=np.float64)
            if values.shape != b.shape or not np.all(np.isfinite(values)):
                msg = (
                    f"per-azimuth B[{index}] must be finite with shape {b.shape}, "
                    f"got {values.shape}"
                )
                raise ValueError(msg)
            member_active = _active_hydrodyn_wave_rows(
                values[:, wave],
                state_descriptions=result.state_descriptions,
                expected_count=expected_count,
                nonzero_atol=nonzero_atol,
                label=f"per-azimuth B[{index}] wave column",
            )
            if not np.array_equal(member_active, active):
                msg = (
                    f"per-azimuth B[{index}] active wave rows "
                    f"{member_active.tolist()} do not match averaged block "
                    f"{active.tolist()}"
                )
                raise ValueError(msg)
            validated_periodic.append(values)

    corrected = np.array(b, copy=True)
    corrected_rows: list[int] = []
    offset = 0
    for count, sign in zip(states_per_dof, dof_signs, strict=True):
        rows = active[offset : offset + count]
        if sign == -1:
            corrected[rows, wave] *= -1.0
            corrected_rows.extend(int(row) for row in rows)
        offset += count
    audit = HydroDynWaveDofSignCorrection(
        wave_input_index=wave,
        states_per_dof=states_per_dof,
        dof_signs=dof_signs,
        corrected_rows=tuple(corrected_rows),
    )
    if result.per_azimuth_b is not None:
        periodic_b = []
        for values in validated_periodic:
            corrected_values = np.array(values, copy=True)
            corrected_values[corrected_rows, wave] *= -1.0
            periodic_b.append(corrected_values)
    return replace(result, avg_b=corrected, per_azimuth_b=periodic_b), audit


def _active_hydrodyn_wave_rows(
    column: npt.NDArray[np.float64],
    *,
    state_descriptions: list[str],
    expected_count: int,
    nonzero_atol: float,
    label: str,
) -> npt.NDArray[np.intp]:
    """Return a validated contiguous HydroDyn-only excitation block."""
    active = np.flatnonzero(np.abs(column) > nonzero_atol)
    foreign = [
        int(index)
        for index in active
        if not state_descriptions[int(index)].strip().casefold().startswith("hd")
    ]
    if foreign:
        msg = f"{label} excites non-HydroDyn states {foreign}"
        raise ValueError(msg)
    if active.size != expected_count:
        msg = (
            f"{label} has {active.size} active rows but .ssexctn declares "
            f"{expected_count} states"
        )
        raise ValueError(msg)
    if active.size and not np.array_equal(
        active,
        np.arange(active[0], active[0] + active.size),
    ):
        msg = f"{label} is not one contiguous block"
        raise ValueError(msg)
    return active


def remap_hydrodyn_state_space_wave_input(
    result: MBCResult,
    *,
    zero_atol: float = 0.0,
    nonzero_atol: float = 0.0,
) -> tuple[MBCResult, HydroDynWaveInputRemap]:
    """Return a copy with OpenFAST's HydroDyn wave-input label defect corrected.

    OpenFAST's HydroDyn state-space excitation Jacobian places ``SS_Exctn%B`` in
    ``m%Jac%Nu`` (the final HydroDyn extended-input column), while the module registers
    wave elevation before current speed, shear, and propagation direction.  The result
    is a non-zero wave-excitation column labelled ``propagation direction`` and a zero
    column labelled ``wave elevation``.

    This function fails closed unless all four HydroDyn extended inputs occur exactly
    once, the wave/current/shear columns are zero, the direction-labelled column is
    non-zero only in HydroDyn states, both affected direct-feedthrough columns are zero,
    and both operating-point inputs are zero.  On success it swaps the two descriptions
    and their MBC-coordinate labels; no matrix coefficient is changed.

    Parameters
    ----------
    result : MBCResult
        Azimuth-averaged OpenFAST model.
    zero_atol : float, optional
        Absolute tolerance used for columns that must be zero.  The default is exact.
    nonzero_atol : float, optional
        Minimum absolute coefficient required in the carrier column.

    Returns
    -------
    corrected, audit : tuple[MBCResult, HydroDynWaveInputRemap]
        Corrected copy and an immutable provenance record.

    Raises
    ------
    ValueError
        If the complete known-defect signature is not present.
    """
    if (
        not math.isfinite(zero_atol)
        or zero_atol < 0.0
        or not math.isfinite(nonzero_atol)
        or nonzero_atol < 0.0
    ):
        msg = "zero_atol and nonzero_atol must be finite and non-negative"
        raise ValueError(msg)
    if result.avg_b is None:
        msg = "HydroDyn wave-input remapping requires an averaged B matrix"
        raise ValueError(msg)

    indices = {
        name: _unique_index(result.input_descriptions, name)
        for name in (_WAVE, _CURRENT, _SHEAR, _DIRECTION)
    }
    wave = indices[_WAVE]
    carrier = indices[_DIRECTION]
    b = np.asarray(result.avg_b, dtype=np.float64)
    if not np.all(np.isfinite(b)):
        msg = "averaged B matrix contains non-finite values"
        raise ValueError(msg)
    for name in (_WAVE, _CURRENT, _SHEAR):
        _require_zero(b[:, indices[name]], zero_atol, f"{name} B column")

    carrier_column = b[:, carrier]
    active = np.flatnonzero(np.abs(carrier_column) > nonzero_atol)
    if active.size == 0:
        msg = "direction-labelled HydroDyn carrier B column is zero"
        raise ValueError(msg)
    if len(result.state_descriptions) != b.shape[0]:
        msg = "state descriptions do not align with the averaged B rows"
        raise ValueError(msg)
    foreign = [
        int(index)
        for index in active
        if not result.state_descriptions[int(index)].strip().casefold().startswith("hd")
    ]
    if foreign:
        msg = f"direction-labelled carrier excites non-HydroDyn states {foreign}"
        raise ValueError(msg)

    if result.per_azimuth_b is not None:
        if len(result.per_azimuth_b) != len(result.azimuths_deg):
            msg = "per-azimuth B matrices must align with the retained azimuths"
            raise ValueError(msg)
        for index, matrix in enumerate(result.per_azimuth_b):
            member_b = np.asarray(matrix, dtype=np.float64)
            if member_b.shape != b.shape or not np.all(np.isfinite(member_b)):
                msg = (
                    f"per-azimuth B[{index}] must be finite with shape {b.shape}, "
                    f"got {member_b.shape}"
                )
                raise ValueError(msg)
            for name in (_WAVE, _CURRENT, _SHEAR):
                _require_zero(
                    member_b[:, indices[name]],
                    zero_atol,
                    f"per-azimuth B[{index}] {name} column",
                )
            member_active = np.flatnonzero(np.abs(member_b[:, carrier]) > nonzero_atol)
            if member_active.size == 0:
                msg = f"per-azimuth B[{index}] direction-labelled carrier is zero"
                raise ValueError(msg)
            member_foreign = [
                int(row)
                for row in member_active
                if not result.state_descriptions[int(row)]
                .strip()
                .casefold()
                .startswith("hd")
            ]
            if member_foreign:
                msg = (
                    f"per-azimuth B[{index}] direction-labelled carrier excites "
                    f"non-HydroDyn states {member_foreign}"
                )
                raise ValueError(msg)

    if result.avg_d is not None:
        d = np.asarray(result.avg_d, dtype=np.float64)
        if not np.all(np.isfinite(d)):
            msg = "averaged D matrix contains non-finite values"
            raise ValueError(msg)
        _require_zero(d[:, wave], zero_atol, "wave-elevation D column")
        _require_zero(d[:, carrier], zero_atol, "direction-labelled carrier D column")
    else:
        d = None

    if result.per_azimuth_d is not None:
        if d is None:
            msg = "per-azimuth D matrices require an averaged D matrix"
            raise ValueError(msg)
        if len(result.per_azimuth_d) != len(result.azimuths_deg):
            msg = "per-azimuth D matrices must align with the retained azimuths"
            raise ValueError(msg)
        for index, matrix in enumerate(result.per_azimuth_d):
            member_d = np.asarray(matrix, dtype=np.float64)
            if member_d.shape != d.shape or not np.all(np.isfinite(member_d)):
                msg = (
                    f"per-azimuth D[{index}] must be finite with shape {d.shape}, "
                    f"got {member_d.shape}"
                )
                raise ValueError(msg)
            _require_zero(
                member_d[:, wave],
                zero_atol,
                f"per-azimuth D[{index}] wave-elevation column",
            )
            _require_zero(
                member_d[:, carrier],
                zero_atol,
                f"per-azimuth D[{index}] direction-labelled carrier column",
            )

    if result.input_operating_point is not None:
        operating_point = np.asarray(result.input_operating_point, dtype=np.float64)
        _require_zero(
            operating_point[[wave, carrier]],
            zero_atol,
            "wave/carrier input operating points",
        )

    # Per-azimuth baselines are consumed by remapped index downstream, so each retained
    # member must be zero too; nonzero entries could otherwise cancel in the average.
    if result.per_azimuth_input_operating_point is not None:
        if len(result.per_azimuth_input_operating_point) != len(result.azimuths_deg):
            msg = "per-azimuth input operating points must align with the azimuths"
            raise ValueError(msg)
        n_inputs = len(result.input_descriptions)
        for index, vector in enumerate(result.per_azimuth_input_operating_point):
            member_op = np.asarray(vector, dtype=np.float64)
            if member_op.shape != (n_inputs,) or not np.all(np.isfinite(member_op)):
                msg = (
                    f"per-azimuth input operating point {index} must be finite with "
                    f"shape ({n_inputs},), got {member_op.shape}"
                )
                raise ValueError(msg)
            _require_zero(
                member_op[[wave, carrier]],
                zero_atol,
                f"per-azimuth wave/carrier input operating points [{index}]",
            )

    descriptions = list(result.input_descriptions)
    descriptions[wave], descriptions[carrier] = (
        descriptions[carrier],
        descriptions[wave],
    )
    coordinates = list(result.input_mbc_coordinates)
    if coordinates:
        if len(coordinates) != len(descriptions):
            msg = "input MBC coordinates do not align with input descriptions"
            raise ValueError(msg)
        coordinates[wave], coordinates[carrier] = (
            coordinates[carrier],
            coordinates[wave],
        )

    audit = HydroDynWaveInputRemap(
        source_wave_index=wave,
        source_carrier_index=carrier,
        wave_b_l2_before=float(np.linalg.norm(b[:, wave])),
        carrier_b_l2_before=float(np.linalg.norm(carrier_column)),
        carrier_nonzero_rows=tuple(int(index) for index in active),
        state_space_signature=(
            "OpenFAST HydroDyn SS_Exctn%B stored in final extended-input column; "
            "numeric matrices unchanged and wave/direction descriptions swapped"
        ),
    )
    return (
        replace(
            result,
            input_descriptions=descriptions,
            input_mbc_coordinates=coordinates,
        ),
        audit,
    )


def _unique_index(names: list[str], requested: str) -> int:
    matches = [index for index, name in enumerate(names) if name == requested]
    if len(matches) != 1:
        msg = f"expected exactly one input {requested!r}, found {matches}"
        raise ValueError(msg)
    return matches[0]


def _require_zero(values: np.ndarray, atol: float, label: str) -> None:
    if not np.all(np.abs(values) <= atol):
        maximum = float(np.max(np.abs(values)))
        msg = f"{label} is not zero within atol={atol:g} (max={maximum:g})"
        raise ValueError(msg)
