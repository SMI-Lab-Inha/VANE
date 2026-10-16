# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for guarded compatibility corrections of OpenFAST linearization output."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from vane.io.mbc_transform import MBCResult
from vane.io.openfast_compat import (
    apply_hydrodyn_wave_dof_signs,
    read_state_space_excitation_metadata,
    remap_hydrodyn_state_space_wave_input,
)

if TYPE_CHECKING:
    from pathlib import Path

_INPUTS = [
    "HD Extended input: wave elevation at platform ref point, m",
    "HD Extended input: horizontal current speed (steady/uniform wind), m/s",
    "HD Extended input: vertical power-law shear exponent, -",
    "HD Extended input: propagation direction, rad",
]


def _result() -> MBCResult:
    b = np.zeros((3, 4))
    b[:2, 3] = [2.0, -3.0]
    return MBCResult(
        avg_a=np.eye(3),
        avg_b=b,
        avg_c=np.ones((1, 3)),
        avg_d=np.zeros((1, 4)),
        state_descriptions=["HD excitation 1", "HD excitation 2", "ED state"],
        ndof2=0,
        ndof1=3,
        n_blades=0,
        performed_transformation=False,
        rotor_speed_rpm=0.0,
        wind_speed=0.0,
        azimuths_deg=np.array([0.0]),
        input_descriptions=list(_INPUTS),
        input_mbc_coordinates=["", "", "", ""],
        input_operating_point=np.zeros(4),
    )


def test_remap_swaps_only_wave_and_carrier_descriptions() -> None:
    """The correction changes labels while preserving every numeric matrix."""
    source = _result()
    corrected, audit = remap_hydrodyn_state_space_wave_input(source)

    assert corrected.input_descriptions == [
        _INPUTS[3],
        _INPUTS[1],
        _INPUTS[2],
        _INPUTS[0],
    ]
    assert source.input_descriptions == _INPUTS
    assert corrected.avg_b is source.avg_b
    assert corrected.avg_d is source.avg_d
    assert audit.source_wave_index == 0
    assert audit.source_carrier_index == 3
    assert audit.wave_b_l2_before == 0.0
    assert audit.carrier_b_l2_before == pytest.approx(np.sqrt(13.0))
    assert audit.carrier_nonzero_rows == (0, 1)


@pytest.mark.parametrize(
    ("field", "row", "column"),
    [
        ("avg_b", 0, 0),
        ("avg_b", 2, 3),
        ("avg_d", 0, 3),
    ],
)
def test_remap_fails_closed_when_signature_is_violated(
    field: str, row: int, column: int
) -> None:
    """Any extra coupling in a protected column prevents the correction."""
    source = _result()
    matrix = np.array(getattr(source, field), copy=True)
    matrix[row, column] = 1.0
    changes: dict[str, Any] = {field: matrix}
    corrupted = replace(source, **changes)

    with pytest.raises(ValueError):
        remap_hydrodyn_state_space_wave_input(corrupted)


def test_remap_requires_zero_wave_and_carrier_operating_points() -> None:
    """A nonzero operating point makes a description-only swap unsafe."""
    source = replace(_result(), input_operating_point=np.array([1.0, 0.0, 0.0, 0.0]))

    with pytest.raises(ValueError, match="operating points"):
        remap_hydrodyn_state_space_wave_input(source)


@pytest.mark.parametrize(
    ("zero_atol", "nonzero_atol"),
    [
        (np.nan, 0.0),
        (np.inf, 0.0),
        (-1.0, 0.0),
        (0.0, np.nan),
        (0.0, np.inf),
        (0.0, -1.0),
    ],
)
def test_remap_requires_finite_nonnegative_tolerances(
    zero_atol: float, nonzero_atol: float
) -> None:
    """A malformed tolerance cannot weaken the complete defect signature."""
    with pytest.raises(ValueError, match="finite and non-negative"):
        remap_hydrodyn_state_space_wave_input(
            _result(), zero_atol=zero_atol, nonzero_atol=nonzero_atol
        )


@pytest.mark.parametrize(
    ("column", "message"),
    [
        (0, "wave elevation"),
        (3, "non-HydroDyn"),
    ],
)
def test_remap_rejects_cancelled_periodic_b_couplings(
    column: int, message: str
) -> None:
    """Protected member couplings cannot disappear behind the averaged B matrix."""
    source = _result()
    assert source.avg_b is not None
    first = np.array(source.avg_b, copy=True)
    second = np.array(source.avg_b, copy=True)
    first[2, column] = 1.0
    second[2, column] = -1.0
    source = replace(
        source,
        azimuths_deg=np.array([0.0, 180.0]),
        per_azimuth_b=[first, second],
    )

    with pytest.raises(ValueError, match=message):
        remap_hydrodyn_state_space_wave_input(source)


def test_remap_rejects_cancelled_periodic_d_couplings() -> None:
    """Averaging cannot hide a direct wave feedthrough in retained members."""
    source = _result()
    assert source.avg_d is not None
    first = np.array(source.avg_d, copy=True)
    second = np.array(source.avg_d, copy=True)
    first[0, 0] = 1.0
    second[0, 0] = -1.0
    source = replace(
        source,
        azimuths_deg=np.array([0.0, 180.0]),
        per_azimuth_d=[first, second],
    )

    with pytest.raises(ValueError, match="wave-elevation"):
        remap_hydrodyn_state_space_wave_input(source)


@pytest.mark.parametrize("column", [0, 3])
def test_remap_rejects_cancelled_periodic_input_baselines(column: int) -> None:
    """Member baselines that cancel in the average cannot survive a label swap."""
    first = np.zeros(4)
    second = np.zeros(4)
    first[column] = 0.5
    second[column] = -0.5
    source = replace(
        _result(),
        azimuths_deg=np.array([0.0, 180.0]),
        per_azimuth_input_operating_point=[first, second],
    )

    with pytest.raises(ValueError, match="per-azimuth wave/carrier input operating"):
        remap_hydrodyn_state_space_wave_input(source)


def test_remap_accepts_zero_periodic_input_baselines() -> None:
    """Zero wave/carrier member baselines keep the correction applicable."""
    baseline = np.array([0.0, 1.5, 0.14, 0.0])
    source = replace(
        _result(),
        azimuths_deg=np.array([0.0, 180.0]),
        per_azimuth_input_operating_point=[baseline, baseline.copy()],
    )

    corrected, _ = remap_hydrodyn_state_space_wave_input(source)
    assert corrected.input_descriptions[0] == _INPUTS[3]


def test_reads_state_space_excitation_preview_offset(tmp_path: Path) -> None:
    """The fitted time lead is recovered from the authoritative file header."""
    source = tmp_path / "platform.ssexctn"
    source.write_text(
        "SS_Excitation_Fitting v1.00.01: State-Spaces Matrices\n"
        "0 %Wave heading angle\n"
        "12 %time offset (tc)\n"
        "38 %total # of states\n"
        "20 0 10 0 8 0 %# states per degree of freedom\n",
        encoding="utf-8",
    )

    metadata = read_state_space_excitation_metadata(source)

    assert metadata.wave_heading_deg == 0.0
    assert metadata.time_offset_s == 12.0
    assert metadata.state_count == 38
    assert metadata.states_per_dof == (20, 0, 10, 0, 8, 0)


def test_rejects_inconsistent_state_space_excitation_header(tmp_path: Path) -> None:
    """A corrupt total cannot silently define the runtime wave convention."""
    source = tmp_path / "platform.ssexctn"
    source.write_text(
        "SS_Excitation_Fitting v1.00.01: State-Spaces Matrices\n"
        "0 %Wave heading angle\n"
        "12 %time offset (tc)\n"
        "39 %total # of states\n"
        "20 0 10 0 8 0 %# states per degree of freedom\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match"):
        read_state_space_excitation_metadata(source)


def test_applies_explicit_wave_dof_sign_to_contiguous_state_block() -> None:
    """Only the declared DOF rows in the corrected wave column change sign."""
    remapped, _ = remap_hydrodyn_state_space_wave_input(_result())
    assert remapped.avg_b is not None
    remapped = replace(remapped, per_azimuth_b=[np.array(remapped.avg_b, copy=True)])

    corrected, audit = apply_hydrodyn_wave_dof_signs(
        remapped,
        states_per_dof=(2, 0, 0, 0, 0, 0),
        dof_signs=(-1, 1, 1, 1, 1, 1),
    )

    assert remapped.avg_b is not None and corrected.avg_b is not None
    np.testing.assert_allclose(corrected.avg_b[:2, 3], [-2.0, 3.0])
    np.testing.assert_allclose(remapped.avg_b[:2, 3], [2.0, -3.0])
    assert corrected.per_azimuth_b is not None
    np.testing.assert_allclose(corrected.per_azimuth_b[0][:2, 3], [-2.0, 3.0])
    assert audit.corrected_rows == (0, 1)
    assert audit.wave_input_index == 3


def test_wave_dof_sign_correction_rejects_state_count_mismatch() -> None:
    """The fitted DOF layout must account for every active excitation state."""
    remapped, _ = remap_hydrodyn_state_space_wave_input(_result())

    with pytest.raises(ValueError, match="declares 3 states"):
        apply_hydrodyn_wave_dof_signs(
            remapped,
            states_per_dof=(3, 0, 0, 0, 0, 0),
            dof_signs=(-1, 1, 1, 1, 1, 1),
        )


@pytest.mark.parametrize(
    ("third_state", "message"),
    [
        ("ED state", "non-HydroDyn"),
        ("HD excitation 3", "declares 2 states"),
    ],
)
def test_wave_dof_sign_correction_rejects_cancelled_periodic_couplings(
    third_state: str, message: str
) -> None:
    """Illegal member couplings cannot disappear behind the azimuth average."""
    remapped, _ = remap_hydrodyn_state_space_wave_input(_result())
    assert remapped.avg_b is not None
    first = np.array(remapped.avg_b, copy=True)
    second = np.array(remapped.avg_b, copy=True)
    first[2, 3] = 1.0
    second[2, 3] = -1.0
    remapped = replace(
        remapped,
        state_descriptions=["HD excitation 1", "HD excitation 2", third_state],
        per_azimuth_b=[first, second],
    )

    with pytest.raises(ValueError, match=message):
        apply_hydrodyn_wave_dof_signs(
            remapped,
            states_per_dof=(2, 0, 0, 0, 0, 0),
            dof_signs=(-1, 1, 1, 1, 1, 1),
        )
