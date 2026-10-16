# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.io.fst_reader — OpenFAST .fst and ElastoDyn parsing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from vane.io.fst_reader import (
    FstFileError,
    read_elastodyn_geometry,
    read_fst_file,
)

if TYPE_CHECKING:
    from pathlib import Path

_FST = """------- OpenFAST primary input (synthetic) -----------------------------------
---------------------- FEATURE SWITCHES AND FLAGS ----------------------------
          1   CompElast   - Compute structural dynamics (switch)
          0   CompInflow  - Compute inflow wind velocities (switch)
          0   CompAero    - Compute aerodynamic loads (switch)
          1   CompServo   - Compute control dynamics (switch)
          1   CompSeaSt   - Compute sea state information (switch)
          1   CompHydro   - Compute hydrodynamic loads (switch)
          0   CompSub     - Compute sub-structural dynamics (switch)
          1   CompMooring - Compute mooring system (switch)
          0   CompIce     - Compute ice loads (switch)
          0   CompSoil    - Compute soil-structural dynamics (switch)
---------------------- INPUT FILES -------------------------------------------
"semi_ElastoDyn.dat"   EDFile      - ElastoDyn input (quoted string)
"semi_InflowWind.dat"  InflowFile  - InflowWind input (quoted string)
"semi_AeroDyn.dat"     AeroFile    - AeroDyn input (quoted string)
"semi_ServoDyn.dat"    ServoFile   - ServoDyn input (quoted string)
"semi_SeaState.dat"    SeaStFile   - SeaState input (quoted string)
"semi_HydroDyn.dat"    HydroFile   - HydroDyn input (quoted string)
"unused"               SubFile     - SubDyn input (quoted string)
"semi_MAP.dat"         MooringFile - Mooring input (quoted string)
---------------------- LINEARIZATION -----------------------------------------
True          Linearize   - Linearization analysis (flag)
False         CalcSteady  - Calculate a steady-state operating point (flag)
          3   TrimCase    - Controller parameter to be trimmed (-)
      0.001   TrimTol     - Tolerance for rotational speed convergence (-)
       0.01   TrimGain    - Proportional gain for the rotational speed error
          0   Twr_Kdmp    - Damping factor for the tower (N/(m/s))
          0   Bld_Kdmp    - Damping factor for the blades (N/(m/s))
          1   NLinTimes   - Number of times to linearize (-)
          0   LinTimes    - List of times at which to linearize (s)
          1   LinInputs   - Inputs included in linearization (switch)
          1   LinOutputs  - Outputs included in linearization (switch)
False         LinOutJac   - Include full Jacobians in linearization output (flag)
False         LinOutMod   - Write module-level linearization output files (flag)
"""

_ELASTODYN = """
------- ELASTODYN INPUT (synthetic) -----------------------------------
          3   NumBl       - Number of blades (-)
         63   TipRad      - Distance from rotor apex to blade tip (meters)
        1.5   HubRad      - Distance from rotor apex to blade root (meters)
       87.6   TowerHt     - Height of tower top relative to MSL (meters)
         10   TowerBsHt   - Height of tower base relative to MSL (meters)
"""

_FST_CALCSTEADY_MULTI = """
---------------------- LINEARIZATION -----------------------------------
True          Linearize   - Linearization analysis (flag)
True          CalcSteady  - Calculate a steady-state operating point (flag)
          2   TrimCase    - Controller parameter to be trimmed (-)
      0.001   TrimTol     - Tolerance (-)
       0.01   TrimGain    - Gain
          0   Twr_Kdmp    - tower damping
          0   Bld_Kdmp    - blade damping
          3   NLinTimes   - Number of times to linearize (-)
30, 60, 90    LinTimes    - List of times at which to linearize (s)
          2   LinInputs   - Inputs included (switch)
          2   LinOutputs  - Outputs included (switch)
True          LinOutJac   - Include full Jacobians (flag)
True          LinOutMod   - Write module-level files (flag)
          1   CompElast   - structural
          0   CompInflow  - inflow
          0   CompAero    - aero
          0   CompServo   - servo
          0   CompSeaSt   - seast
          0   CompHydro   - hydro
          0   CompSub     - sub
          0   CompMooring - mooring
          0   CompIce     - ice
          0   CompSoil    - soil
"""


@pytest.fixture
def fst_path(tmp_path: Path) -> Path:
    """Write the synthetic floating .fst deck and return its path."""
    path = tmp_path / "semi.fst"
    path.write_text(_FST, encoding="utf-8")
    (tmp_path / "semi_ElastoDyn.dat").write_text(_ELASTODYN, encoding="utf-8")
    return path


class TestReadFstFile:
    """Unit tests for read_fst_file using synthetic decks."""

    def test_comp_switches(self, fst_path: Path) -> None:
        """All module switches are parsed to their integer values."""
        model = read_fst_file(fst_path)
        assert model.comp["CompHydro"] == 1
        assert model.comp["CompSub"] == 0
        assert model.comp["CompMooring"] == 1
        assert model.comp["CompAero"] == 0

    def test_file_references_resolved(self, fst_path: Path) -> None:
        """Referenced files resolve relative to the .fst; 'unused' is dropped."""
        model = read_fst_file(fst_path)
        assert model.ed_file == fst_path.parent / "semi_ElastoDyn.dat"
        assert model.files["MooringFile"].name == "semi_MAP.dat"
        assert "SubFile" not in model.files

    def test_all_active_module_files_resolved(self, fst_path: Path) -> None:
        """Every referenced module sub-file is resolved, not only the first four."""
        files = read_fst_file(fst_path).files
        for keyword in ("InflowFile", "AeroFile", "ServoFile", "SeaStFile"):
            assert keyword in files
        assert files["AeroFile"].name == "semi_AeroDyn.dat"

    def test_linearization_config(self, fst_path: Path) -> None:
        """The linearization block is parsed with correct types."""
        lin = read_fst_file(fst_path).lin
        assert lin.linearize is True
        assert lin.calc_steady is False
        assert lin.trim_case == 3
        assert lin.n_lin_times == 1
        assert lin.lin_times == [0.0]
        assert lin.lin_out_jac is False

    def test_offshore_and_floating_flags(self, fst_path: Path) -> None:
        """The offshore/floating heuristics reflect the module switches."""
        model = read_fst_file(fst_path)
        assert model.is_offshore is True
        assert model.is_floating is True

    def test_multi_value_lin_times(self, tmp_path: Path) -> None:
        """A CalcSteady deck with a multi-value LinTimes line parses fully."""
        path = tmp_path / "cs.fst"
        path.write_text(_FST_CALCSTEADY_MULTI, encoding="utf-8")
        lin = read_fst_file(path).lin
        assert lin.calc_steady is True
        assert lin.n_lin_times == 3
        assert lin.lin_times == [30.0, 60.0, 90.0]
        assert lin.lin_out_mod is True

    def test_comma_only_lin_times(self, tmp_path: Path) -> None:
        """A compact comma-only LinTimes list (no spaces) parses correctly."""
        text = _FST_CALCSTEADY_MULTI.replace("30, 60, 90", "30,60,90")
        path = tmp_path / "cs_compact.fst"
        path.write_text(text, encoding="utf-8")
        assert read_fst_file(path).lin.lin_times == [30.0, 60.0, 90.0]

    def test_missing_fields_default_without_error(self, tmp_path: Path) -> None:
        """Absent module switches and linearization settings fall back to defaults."""
        path = tmp_path / "minimal.fst"
        path.write_text("          1   CompElast   - structural\n", encoding="utf-8")
        model = read_fst_file(path)
        assert model.comp["CompElast"] == 1
        assert model.comp["CompSoil"] == 0  # absent -> default
        assert model.lin.linearize is False
        assert model.lin.n_lin_times == 0
        assert model.lin.lin_times == []

    def test_unreadable_file_raises(self, tmp_path: Path) -> None:
        """A nonexistent file raises FstFileError."""
        with pytest.raises(FstFileError, match="Cannot read"):
            read_fst_file(tmp_path / "does_not_exist.fst")

    def test_non_integer_field_raises_fst_file_error(self, tmp_path: Path) -> None:
        """A non-integer int field raises FstFileError, not a raw ValueError."""
        path = tmp_path / "bad_int.fst"
        path.write_text("        abc   NLinTimes   - number\n", encoding="utf-8")
        with pytest.raises(FstFileError, match=r"NLinTimes.*non-integer"):
            read_fst_file(path)

    def test_non_numeric_float_field_raises_fst_file_error(
        self, tmp_path: Path
    ) -> None:
        """A non-numeric value for a float field raises FstFileError."""
        path = tmp_path / "bad_float.fst"
        path.write_text("        xyz   TrimTol   - tolerance\n", encoding="utf-8")
        with pytest.raises(FstFileError, match=r"TrimTol.*non-numeric"):
            read_fst_file(path)

    def test_non_numeric_lin_times_raises_fst_file_error(self, tmp_path: Path) -> None:
        """A non-numeric LinTimes token raises FstFileError, not a raw ValueError."""
        path = tmp_path / "bad_lintimes.fst"
        path.write_text("   30, oops, 90   LinTimes   - times\n", encoding="utf-8")
        with pytest.raises(FstFileError, match=r"LinTimes.*non-numeric"):
            read_fst_file(path)

    def test_list_directed_trailing_comma_values(self, tmp_path: Path) -> None:
        """A trailing comma (``1,   0   LinInputs``) ends the first value."""
        path = tmp_path / "comma.fst"
        path.write_text(
            "1,   0   LinInputs   - inputs\n"
            "  2,     NLinTimes   - count\n"
            " 0.5,    TrimTol     - tolerance\n",
            encoding="utf-8",
        )
        lin = read_fst_file(path).lin
        assert lin.lin_inputs == 1
        assert lin.n_lin_times == 2
        assert lin.trim_tol == pytest.approx(0.5)

    @pytest.mark.parametrize(
        ("token", "expected"),
        [
            ("True", True),
            ("T", True),
            (".true.", True),
            (".T.", True),
            ("t,", True),
            ("False", False),
            ("F", False),
            (".false.", False),
            ("fale", False),
        ],
    )
    def test_fortran_logical_values(
        self, tmp_path: Path, token: str, expected: bool
    ) -> None:
        """Logicals are read Fortran-style from the first character after ``.``."""
        path = tmp_path / "flag.fst"
        path.write_text(f"{token}   Linearize   - flag\n", encoding="utf-8")
        assert read_fst_file(path).lin.linearize is expected

    def test_unparseable_logical_raises(self, tmp_path: Path) -> None:
        """A logical that starts with neither T nor F raises FstFileError."""
        path = tmp_path / "flag.fst"
        path.write_text("yes   Linearize   - flag\n", encoding="utf-8")
        with pytest.raises(FstFileError, match="as boolean"):
            read_fst_file(path)

    def test_quoted_path_containing_separator(self, fst_path: Path) -> None:
        """A quoted path containing the ``" - "`` separator still resolves."""
        text = fst_path.read_text(encoding="utf-8").replace(
            '"semi_ElastoDyn.dat"', '"Run A - base/semi_ElastoDyn.dat"'
        )
        dashed = fst_path.parent / "dashed.fst"
        dashed.write_text(text, encoding="utf-8")
        model = read_fst_file(dashed)
        assert model.ed_file == dashed.parent / "Run A - base" / "semi_ElastoDyn.dat"

    def test_quoted_path_with_spaces(self, fst_path: Path) -> None:
        """A quoted sub-file path containing spaces resolves intact."""
        text = fst_path.read_text(encoding="utf-8").replace(
            '"semi_ElastoDyn.dat"', '"Sub Folder/semi_ElastoDyn.dat"'
        )
        spaced = fst_path.parent / "spaced.fst"
        spaced.write_text(text, encoding="utf-8")
        model = read_fst_file(spaced)
        assert model.ed_file == spaced.parent / "Sub Folder" / "semi_ElastoDyn.dat"

    def test_single_quoted_path_and_non_utf8_comment(self, tmp_path: Path) -> None:
        """An apostrophe-quoted path resolves; a Windows-1252 comment byte is read."""
        path = tmp_path / "legacy.fst"
        path.write_bytes(
            b"1 CompElast - structural dynamics (\xb0)\n"
            b"'Sub Folder/ED.dat' EDFile - ElastoDyn\n"
        )
        model = read_fst_file(path)
        assert model.comp["CompElast"] == 1
        assert model.ed_file == tmp_path / "Sub Folder" / "ED.dat"


class TestReadElastoDynGeometry:
    """Unit tests for read_elastodyn_geometry."""

    def test_geometry_values(self, tmp_path: Path) -> None:
        """Geometry fields and derived lengths are correct."""
        path = tmp_path / "ed.dat"
        path.write_text(_ELASTODYN, encoding="utf-8")
        geo = read_elastodyn_geometry(path)
        assert geo.num_blades == 3
        assert geo.tip_radius == pytest.approx(63.0)
        assert geo.hub_radius == pytest.approx(1.5)
        assert geo.blade_length == pytest.approx(61.5)
        assert geo.tower_length == pytest.approx(77.6)

    def test_missing_tower_base_height_defaults_to_zero(self, tmp_path: Path) -> None:
        """An ElastoDyn file without ``TowerBsHt`` reads a tower base at 0."""
        path = tmp_path / "ed.dat"
        text = "\n".join(
            line for line in _ELASTODYN.splitlines() if "TowerBsHt" not in line
        )
        path.write_text(text, encoding="utf-8")
        geo = read_elastodyn_geometry(path)
        assert geo.tower_base_height == 0.0
        assert geo.tower_length == pytest.approx(87.6)


class TestReadFstRTest:
    """Integration tests against real r-test decks (skipped if absent)."""

    def test_oc4semi_fst_and_geometry(self, rtest_dir: Path) -> None:
        """The 5MW OC4 semi deck parses to a floating model with 3 blades."""
        fst = (
            rtest_dir
            / "glue-codes"
            / "openfast"
            / "5MW_OC4Semi_Linear"
            / "5MW_OC4Semi_Linear.fst"
        )
        if not fst.is_file():
            pytest.skip(f"r-test case not found: {fst}")
        model = read_fst_file(fst)
        assert model.lin.linearize is True
        assert model.lin.n_lin_times == 1
        assert model.is_floating is True
        assert model.ed_file is not None and model.ed_file.is_file()

        geo = read_elastodyn_geometry(model.ed_file)
        assert geo.num_blades == 3
        assert geo.tip_radius == pytest.approx(63.0)
        assert geo.blade_length == pytest.approx(61.5)
        assert geo.tower_height == pytest.approx(87.6)


class TestFortranFloat:
    """Deck reals in Fortran's double-precision exponent notation."""

    @pytest.mark.parametrize(
        ("token", "expected"),
        [
            ("1.21D+02", 121.0),
            ("5d-3", 5e-3),
            ('"1.0D0"', 1.0),
            ("1.5e3", 1500.0),
            ("-2.0D-1", -0.2),
            ("63", 63.0),
            ("1.21D+02,", 121.0),
            ("0,", 0.0),
        ],
    )
    def test_reads_fortran_reals(self, token: str, expected: float) -> None:
        """D/d exponents read like E; quotes are ignored."""
        from vane.io.fst_reader import fortran_float

        assert fortran_float(token) == pytest.approx(expected)

    @pytest.mark.parametrize("token", ["DEFAULT", "D", "abc", "1.0D"])
    def test_non_numbers_are_rejected(self, token: str) -> None:
        """A word, or a dangling exponent marker, is not a number."""
        from vane.io.fst_reader import fortran_float

        with pytest.raises(ValueError):
            fortran_float(token)
