# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.export — tabular export of modal and Campbell results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from vane.benchmark import synthetic_system
from vane.campbell.builder import build_campbell
from vane.export import campbell_table, modes_table, write_table
from vane.modal.eigensolver import compute_modes
from vane.modal.identifier import identify_modes
from vane.modal.labeler import label_solution
from vane.modal.uncertainty import AzimuthSpread, unified_mode_confidence

if TYPE_CHECKING:
    from pathlib import Path

    from vane.modal.eigensolver import ModalSolution

_DESCS = [
    "ED 1st tower fore-aft (internal DOF index = DOF_TFA1), m",
    "ED drivetrain (internal DOF index = DOF_DrTr), rad",
    "ED tower fore-aft velocity",
    "ED drivetrain velocity",
]


def _solution(frequencies: tuple[float, float]) -> ModalSolution:
    truth = synthetic_system(list(frequencies), [0.02, 0.03])
    return compute_modes(truth.a, ndof2=2, ndof1=0, descriptions=_DESCS)


class TestModesTable:
    """Tests for modes_table."""

    def test_base_table_has_one_row_per_mode(self) -> None:
        """The base table has a row per mode and the core columns."""
        table = modes_table(_solution((1.0, 2.0)))
        assert len(table) == 2
        for column in (
            "mode",
            "natural_frequency_hz",
            "damping_ratio",
            "condition_number",
            "is_degenerate",
        ):
            assert column in table.columns
        np.testing.assert_allclose(
            np.sort(table["natural_frequency_hz"].to_numpy()), [1.0, 2.0], rtol=1e-9
        )

    def test_optional_columns_are_added(self) -> None:
        """Labels, confidence, and spread add their columns."""
        solution = _solution((1.0, 2.0))
        table = modes_table(
            solution,
            labels=label_solution(solution),
            confidence=unified_mode_confidence(solution),
            spread=AzimuthSpread(np.array([0.1, 0.2]), np.array([0.01, 0.02]), 6),
        )
        for column in (
            "label",
            "category",
            "multiblade",
            "confidence",
            "frequency_std_hz",
            "damping_std",
        ):
            assert column in table.columns

    def test_mismatched_label_length_raises(self) -> None:
        """A label list of the wrong length raises."""
        solution = _solution((1.0, 2.0))
        with pytest.raises(ValueError, match="labels has"):
            modes_table(solution, labels=label_solution(solution)[:1])


class TestCampbellTable:
    """Tests for campbell_table."""

    def test_long_format_rows_and_columns(self) -> None:
        """The table is long-format with the parameter column named accordingly."""
        solutions = [_solution((1.0, 2.0)), _solution((1.1, 2.1))]
        diagram = build_campbell(
            identify_modes(solutions), [10.0, 12.0], parameter_name="rotor_speed_rpm"
        )
        table = campbell_table(diagram)
        assert "rotor_speed_rpm" in table.columns
        assert set(table["track"]) == {0, 1}
        # two tracks, each spanning two operating points
        assert len(table) == 4

    def test_parameter_name_collision_raises(self) -> None:
        """A parameter name colliding with a fixed column is rejected."""
        solutions = [_solution((1.0, 2.0)), _solution((1.1, 2.1))]
        diagram = build_campbell(
            identify_modes(solutions), [10.0, 12.0], parameter_name="label"
        )
        with pytest.raises(ValueError, match="reserved export column"):
            campbell_table(diagram)

    def test_parameter_values_are_mapped(self) -> None:
        """Operating-point indices are mapped to their parameter values."""
        solutions = [_solution((1.0, 2.0)), _solution((1.1, 2.1))]
        diagram = build_campbell(
            identify_modes(solutions), [10.0, 12.0], parameter_name="rotor_speed_rpm"
        )
        table = campbell_table(diagram)
        assert set(table["rotor_speed_rpm"]) == {10.0, 12.0}


class TestWriteTable:
    """Tests for write_table."""

    def test_csv_round_trip(self, tmp_path: Path) -> None:
        """A CSV write reloads to the same content."""
        table = modes_table(_solution((1.0, 2.0)))
        path = tmp_path / "modes.csv"
        write_table(table, path)
        reloaded = pd.read_csv(path)
        np.testing.assert_allclose(
            np.sort(reloaded["natural_frequency_hz"].to_numpy()), [1.0, 2.0], rtol=1e-9
        )

    def test_json_round_trip(self, tmp_path: Path) -> None:
        """A JSON write reloads to the same number of rows."""
        table = modes_table(_solution((1.0, 2.0)))
        path = tmp_path / "modes.json"
        write_table(table, path)
        assert len(pd.read_json(path)) == 2

    def test_unsupported_suffix_raises(self, tmp_path: Path) -> None:
        """An unsupported file suffix raises."""
        table = modes_table(_solution((1.0, 2.0)))
        with pytest.raises(ValueError, match="unsupported export format"):
            write_table(table, tmp_path / "modes.txt")
