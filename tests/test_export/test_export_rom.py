# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.export — extra tables, the .mat format, and the ROM bundle."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from vane.benchmark import synthetic_system
from vane.campbell.excitation import ResonanceCrossing, ResonanceSeverity
from vane.export import (
    mode_shapes_table,
    participation_table,
    resonance_table,
    write_rom,
    write_table,
)
from vane.modal.eigensolver import compute_modes
from vane.modal.participation import participation_from_modes
from vane.sysid.rom import modal_reduction
from vane.sysid.state_space import StateSpace

if TYPE_CHECKING:
    from pathlib import Path

    from vane.modal.eigensolver import ModalSolution
    from vane.sysid.rom import ModalReduction

_DESCS = ["tower fore-aft, m", "drivetrain, rad", "tower fa vel", "drivetrain vel"]


def _solution() -> ModalSolution:
    """Return a 2-mode (4-state) solution with DOF descriptions."""
    truth = synthetic_system([0.3, 1.7], [0.02, 0.03])
    return compute_modes(truth.a, ndof2=2, ndof1=0, descriptions=_DESCS)


def _reduction() -> ModalReduction:
    """Return a modal reduction of a small observable system."""
    truth = synthetic_system([0.3, 1.7, 5.0], [0.02, 0.03, 0.05])
    n = truth.a.shape[0]
    c = np.zeros((1, n))
    c[0, 0] = 1.0
    system = StateSpace(a=truth.a, b=np.eye(n)[:, :1], c=c, d=np.zeros((1, 1)))
    return modal_reduction(system, n_modes=2)


def _cell_strings(stored: np.ndarray) -> list[str]:
    """Return the strings of a loaded MATLAB cell array (an empty cell is ``""``)."""
    assert stored.dtype == object  # a cell array, not a padded char matrix
    return [
        str(np.asarray(value).ravel()[0]) if np.asarray(value).size else ""
        for value in stored.ravel()
    ]


class TestModeShapesTable:
    """Tests for mode_shapes_table."""

    def test_one_row_per_mode_and_dof(self) -> None:
        """The table has n_dof x n_modes rows and the component columns."""
        solution = _solution()
        table = mode_shapes_table(solution)
        n_dof, n_modes = solution.mode_shapes.shape
        assert len(table) == n_dof * n_modes
        for column in ("mode", "dof", "dof_description", "real", "imag", "magnitude"):
            assert column in table.columns

    def test_magnitude_matches_complex_modulus(self) -> None:
        """The magnitude column equals the complex modulus of the mode shape."""
        solution = _solution()
        table = mode_shapes_table(solution)
        first = table[(table["mode"] == 0) & (table["dof"] == 0)].iloc[0]
        expected = abs(complex(solution.mode_shapes[0, 0]))
        assert first["magnitude"] == pytest.approx(expected)


class TestParticipationTable:
    """Tests for participation_table."""

    def test_one_row_per_mode_and_dof(self) -> None:
        """The table has a row per (mode, DOF) with the participation columns."""
        solution = _solution()
        participation = participation_from_modes(solution)
        table = participation_table(participation, descriptions=_DESCS)
        n_dof, n_modes = participation.magnitude.shape
        assert len(table) == n_dof * n_modes
        assert {"magnitude", "signed_magnitude", "phase_deg", "is_dominant"} <= set(
            table.columns
        )

    def test_one_dominant_dof_per_mode(self) -> None:
        """Exactly one DOF is flagged dominant for each mode."""
        solution = _solution()
        table = participation_table(participation_from_modes(solution))
        per_mode = table.groupby("mode")["is_dominant"].sum()
        assert (per_mode == 1).all()


class TestResonanceTable:
    """Tests for resonance_table."""

    def test_empty_has_columns(self) -> None:
        """No crossings still yields the columns (an empty frame)."""
        table = resonance_table([])
        assert len(table) == 0
        assert "severity" in table.columns

    def test_rows_carry_crossing_fields(self) -> None:
        """Each crossing becomes a row with its fields and named severity."""
        crossings = [
            ResonanceCrossing(
                track_index=1,
                harmonic=3,
                rotor_speed_rpm=10.0,
                frequency_hz=0.5,
                damping_ratio=0.01,
                severity=ResonanceSeverity.HIGH,
            )
        ]
        table = resonance_table(crossings)
        assert len(table) == 1
        assert table.iloc[0]["harmonic"] == 3
        assert table.iloc[0]["severity"] == "HIGH"


class TestMatExport:
    """Tests for the .mat table format."""

    def test_mat_roundtrips_columns(self, tmp_path: Path) -> None:
        """Each column is written as a MATLAB variable of the same (safe) name."""
        from scipy.io import loadmat

        table = mode_shapes_table(_solution())
        out = tmp_path / "shapes.mat"
        write_table(table, out)
        loaded = loadmat(out)
        assert "magnitude" in loaded
        assert loaded["magnitude"].size == len(table)
        # Columns are N x 1 MATLAB column vectors (the table orientation), not 1 x N.
        assert loaded["magnitude"].shape == (len(table), 1)

    def test_colliding_matlab_names_rejected(self, tmp_path: Path) -> None:
        """Two columns sanitizing to the same MATLAB name are rejected, not merged."""
        import pandas as pd

        # "a b" and "a-b" both sanitize to "a_b".
        table = pd.DataFrame({"a b": [1], "a-b": [2]})
        with pytest.raises(ValueError, match="same MATLAB variable name"):
            write_table(table, tmp_path / "clash.mat")

    def test_missing_values_in_string_columns_become_empty_cells(
        self, tmp_path: Path
    ) -> None:
        """An all-None or partly-None object column is written as a cell of strings.

        ``scipy.io.savemat`` cannot store ``None``, so without the mapping an all-None
        column (the multi-blade coordinate of a sweep without blade DOFs) aborted the
        whole export with a ``TypeError``.
        """
        import pandas as pd
        from scipy.io import loadmat

        table = pd.DataFrame(
            {
                "track": [0, 1],
                "label": ["tower FA", "blade flap"],
                "multiblade": [None, None],
                "partial": [None, "cos"],
                "gap": ["a", np.nan],
            }
        )
        out = tmp_path / "labels.mat"
        write_table(table, out)
        loaded = loadmat(out)

        def cells(name: str) -> list[str]:
            return [
                str(np.asarray(v).ravel()[0]) if np.asarray(v).size else ""
                for v in loaded[name].ravel()
            ]

        assert loaded["multiblade"].dtype == object  # a MATLAB cell array
        assert cells("multiblade") == ["", ""]
        assert cells("partial") == ["", "cos"]
        assert cells("gap") == ["a", ""]
        assert cells("label") == ["tower FA", "blade flap"]
        assert loaded["track"].ravel().tolist() == [0, 1]

    def test_unwritable_value_is_a_value_error(self, tmp_path: Path) -> None:
        """A value savemat cannot encode surfaces as a ValueError, not a TypeError."""
        import pandas as pd

        table = pd.DataFrame({"odd": [{1, 2}, {3}]})
        with pytest.raises(ValueError, match="MATLAB"):
            write_table(table, tmp_path / "odd.mat")

    def test_leading_underscore_column_is_preserved(self, tmp_path: Path) -> None:
        """A column whose name starts with '_' is not dropped by savemat."""
        import pandas as pd
        from scipy.io import loadmat

        # savemat ignores keys starting with '_', so the sanitizer must prefix them.
        table = pd.DataFrame({"_frequency": [1.0, 2.0]})
        out = tmp_path / "underscore.mat"
        write_table(table, out)
        loaded = loadmat(out)
        preserved = [k for k in loaded if not k.startswith("__") and "frequency" in k]
        assert preserved  # the column survived under a letter-prefixed name
        assert loaded[preserved[0]].size == 2


class TestWriteRom:
    """Tests for the write_rom bundle."""

    def test_writes_mat_bundle(self, tmp_path: Path) -> None:
        """The bundle has the matrices, modal table, and manifest."""
        from scipy.io import loadmat

        out = write_rom(_reduction(), tmp_path / "rom", matrix_format="mat")
        assert {p.name for p in out.iterdir()} == {
            "state_space.mat",
            "modes.csv",
            "manifest.json",
        }
        mat = loadmat(out / "state_space.mat")
        assert all(key in mat for key in ("A", "B", "C", "D"))
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["n_reduced_states"] == mat["A"].shape[0]
        assert manifest["n_retained_modes"] == 2
        assert manifest["vane_version"]

    def test_writes_npz_bundle(self, tmp_path: Path) -> None:
        """The npz loads without unsafe pickle, including the string name fields."""
        out = write_rom(_reduction(), tmp_path / "rom", matrix_format="npz")
        assert (out / "state_space.npz").is_file()
        # allow_pickle defaults to False: the bundle must be self-contained without it.
        loaded = np.load(out / "state_space.npz")
        assert loaded["A"].shape[0] == loaded["A"].shape[1]
        assert loaded["state_names"].dtype.kind in ("U", "S")  # string, not object

    @pytest.mark.parametrize("matrix_format", ["mat", "npz"])
    def test_output_mbc_coordinates_are_exported(
        self, tmp_path: Path, matrix_format: str
    ) -> None:
        """MBC output labels survive export so rows are not mistaken for blades."""
        from scipy.io import loadmat

        reduction = _reduction()
        reduction.system.output_mbc_coordinates = ["collective"]
        out = write_rom(reduction, tmp_path / "rom", matrix_format=matrix_format)
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["output_mbc_coordinates"] == ["collective"]
        if matrix_format == "mat":
            stored = loadmat(out / "state_space.mat")["output_mbc_coordinates"]
            assert _cell_strings(stored) == ["collective"]
        else:
            stored = np.load(out / "state_space.npz")["output_mbc_coordinates"]
            assert stored.tolist() == ["collective"]

    @pytest.mark.parametrize("matrix_format", ["mat", "npz"])
    def test_names_round_trip_exactly(self, tmp_path: Path, matrix_format: str) -> None:
        """Names of differing length (and an empty one) survive both formats.

        A unicode array was written to ``.mat`` as a blank-padded char matrix, so
        ``"a"`` came back as ``"a  "`` and an empty name was indistinguishable.
        """
        from scipy.io import loadmat

        reduction = _reduction()
        n = reduction.system.a.shape[0]
        state_names = ["x"] * n
        state_names[0] = "tower fore-aft"
        state_names[1] = ""
        reduction.system.state_names = state_names
        reduction.system.output_names = ["a"]
        out = write_rom(reduction, tmp_path / "rom", matrix_format=matrix_format)
        if matrix_format == "mat":
            loaded = loadmat(out / "state_space.mat")
            assert _cell_strings(loaded["state_names"]) == state_names
            assert _cell_strings(loaded["output_names"]) == ["a"]
        else:
            npz = np.load(out / "state_space.npz")  # allow_pickle stays False
            assert npz["state_names"].tolist() == state_names
            assert npz["output_names"].tolist() == ["a"]

    @pytest.mark.parametrize("matrix_format", ["mat", "npz"])
    def test_state_maps_are_exported(self, tmp_path: Path, matrix_format: str) -> None:
        """The full-state projection/reconstruction maps travel with the bundle."""
        from scipy.io import loadmat

        reduction = _reduction()
        assert reduction.state_projection is not None
        assert reduction.state_reconstruction is not None
        out = write_rom(reduction, tmp_path / "rom", matrix_format=matrix_format)
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["state_maps"] == ["state_projection", "state_reconstruction"]
        if matrix_format == "mat":
            stored = loadmat(out / "state_space.mat")
        else:
            stored = dict(np.load(out / "state_space.npz"))
        np.testing.assert_allclose(
            stored["state_projection"], reduction.state_projection
        )
        np.testing.assert_allclose(
            stored["state_reconstruction"], reduction.state_reconstruction
        )

    def test_absent_state_maps_are_omitted(self, tmp_path: Path) -> None:
        """A reduction without the maps writes neither and lists none."""
        reduction = _reduction()
        reduction.state_projection = None
        reduction.state_reconstruction = None
        out = write_rom(reduction, tmp_path / "rom", matrix_format="npz")
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["state_maps"] == []
        assert "state_projection" not in np.load(out / "state_space.npz")

    def test_modes_csv_has_one_row_per_mode(self, tmp_path: Path) -> None:
        """modes.csv lists each retained mode's frequency, damping, and residue."""
        import pandas as pd

        out = write_rom(_reduction(), tmp_path / "rom")
        modes = pd.read_csv(out / "modes.csv")
        assert len(modes) == 2
        assert {"natural_frequency_hz", "damping_ratio", "residue"} <= set(
            modes.columns
        )

    def test_invalid_format_rejected(self, tmp_path: Path) -> None:
        """An unsupported matrix format is rejected."""
        with pytest.raises(ValueError, match="matrix_format must be"):
            write_rom(_reduction(), tmp_path / "rom", matrix_format="parquet")

    def test_feedthrough_is_dimensionally_consistent(self, tmp_path: Path) -> None:
        """A missing D becomes a zero (n_outputs, n_inputs) block, not 0x0."""
        from scipy.io import loadmat

        reduction = _reduction()
        # Drop D to exercise the None path while B and C remain present.
        reduction.system.d = None
        out = write_rom(reduction, tmp_path / "rom")
        mat = loadmat(out / "state_space.mat")
        n_outputs, n_inputs = mat["C"].shape[0], mat["B"].shape[1]
        assert mat["D"].shape == (n_outputs, n_inputs)
        assert np.count_nonzero(mat["D"]) == 0
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["n_inputs"] == n_inputs
        assert manifest["n_outputs"] == n_outputs

    def test_feedthrough_only_b_omitted(self, tmp_path: Path) -> None:
        """D present, B omitted: B is zero-filled to D's input count."""
        from scipy.io import loadmat

        reduction = _reduction()
        system = reduction.system
        assert system.c is not None  # the reduced system always has C
        n_outputs = system.c.shape[0]
        # A feedthrough-only input: D carries two inputs, B is omitted.
        system.b = None
        system.d = np.zeros((n_outputs, 2))
        out = write_rom(reduction, tmp_path / "rom")
        mat = loadmat(out / "state_space.mat")
        n_states = mat["A"].shape[0]
        assert mat["B"].shape == (n_states, 2)  # B materialized to match D's inputs
        assert mat["D"].shape == (n_outputs, 2)
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["n_inputs"] == 2
