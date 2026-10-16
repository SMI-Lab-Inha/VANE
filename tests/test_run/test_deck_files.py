# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.run.deck_files — copying only the files a deck references."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from vane.run import (
    DeckFileClosure,
    DeckReferenceError,
    LinearizationDriver,
    OperatingPoint,
    copy_template_files,
    deck_file_closure,
    infer_template_root,
)

_FST = """\
------- OpenFAST INPUT FILE -------
False         Echo        - Echo input data to <RootName>.ech (flag)
     1.0      TMax        - Total run time (s)
"Turbine ED.dat"  EDFile  - ElastoDyn input parameters (quoted string)
""            BDBldFile(1) - unused BeamDyn blade
"../Shared/InflowWind.dat"  InflowFile  - inflow wind input parameters (quoted string)
"AeroDyn.dat" AeroFile    - aerodynamic input parameters (quoted string)
"ServoDyn.dat" ServoFile  - control input parameters (quoted string)
"HydroDyn.dat" HydroFile  - hydrodynamic input parameters (quoted string)
"none"        SubFile     - sub-structural input parameters (quoted string)
"unused"      MooringFile - mooring input parameters (quoted string)
"Missing.dat" IceFile     - a referenced file that does not exist
"""

_ED = """\
------- ELASTODYN INPUT FILE -------
     9.55     RotSpeed    - Initial rotor speed (rpm)
"../Shared/Blade.dat"    BldFile(1)  - blade 1 file (quoted string)
"../Shared/Blade.dat"    BldFile(2)  - blade 2 file (quoted string)
"../Shared/Blade.dat"    BldFile(3)  - blade 3 file (quoted string)
"Tower.dat"   TwrFile     - tower file (quoted string)
"""

_INFLOW = """\
------- InflowWind INPUT FILE -------
     1        WindType    - switch
"Wind/turb.bts"   FileName_BTS  - Name of the full-field wind file (quoted string)
"Wind/hawc"       FilenameRoot  - Rootname of the Bladed binary files (quoted string)
"""

_AERO = """\
------- AERODYN INPUT FILE -------
     2        NumAFfiles  - Number of airfoil files used (-)
"../Shared/Airfoils/AF0.dat"    AFNames  - Airfoil file names (quoted strings)
"../Shared/Airfoils/AF1.dat"
"../Shared/AD_blade.dat"    ADBlFile(1) - blade file
"../Shared/AD_blade.dat"    ADBlFile(2) - blade file
"../Shared/AD_blade.dat"    ADBlFile(3) - blade file
"""

_AIRFOIL = """\
! ------------ AirfoilInfo Input File ------------
DEFAULT       InterpOrd   ! Interpolation order
@"{coords}"   NumCoords   ! The number of coordinates in the airfoil shape file.
BL_missing.txt  BL_file   ! A boundary-layer file that does not exist
"""

_SERVO = """\
------- SERVODYN INPUT FILE -------
     5        PCMode      - Pitch control mode (switch)
     2        NumNStC     - Number of nacelle structural controllers (integer)
"StC_a.dat","StC_b.dat"   NStCfiles   - Name of the files (quoted strings)
"../Shared/Ctrl/DISCON.dll"    DLL_FileName - Name/location of the dynamic library
"../Shared/Ctrl/DISCON.IN"     DLL_InFile   - Name of input file sent to the DLL (-)
"DISCON"      DLL_ProcName - Name of procedure in DLL to be called (-)
"default"     DLL_DT       - Communication interval for dynamic library (s)
"""

_DISCON = """\
! Controller parameter input file
"Cp_Ct_Cq.txt"      ! PerfFileName  - File containing rotor performance tables
"unused"            ! OL_Filename   - Input file with open loop timeseries
"""

_HYDRO = """\
------- HydroDyn INPUT FILE -------
     1        PotMod      - Potential-flow model (switch)
"../Shared/Hydro/platform"    PotFile  - Root name of potential-flow model data
     0.0      PtfmRefzt   - The blade root and platform reference (m)
"""


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _build_tree(tmp_path: Path) -> tuple[Path, Path]:
    """Write a synthetic template tree; return (template root, primary deck)."""
    root = tmp_path / "suite"
    case = root / "Case"
    shared = root / "Shared"
    fst = _write(case / "Model.fst", _FST)
    _write(case / "Turbine ED.dat", _ED)
    _write(case / "Tower.dat", "tower data\n")
    _write(case / "AeroDyn.dat", _AERO)
    _write(case / "ServoDyn.dat", _SERVO)
    _write(case / "StC_a.dat", "nacelle StC a\n")
    _write(case / "StC_b.dat", "nacelle StC b\n")
    _write(case / "HydroDyn.dat", _HYDRO)
    # Unreferenced files beside the deck: a stale output and an unused input.
    _write(case / "Model.outb", "stale output\n")
    _write(case / "Unused.dat", "not referenced\n")
    _write(shared / "InflowWind.dat", _INFLOW)
    _write(shared / "Wind" / "turb.bts", "binary-ish wind\n")
    _write(shared / "Wind" / "hawc.wnd", "bladed wind\n")
    _write(shared / "Wind" / "hawc.sum", "bladed summary (an output suffix)\n")
    _write(shared / "Wind" / "other.bts", "another wind file\n")
    _write(shared / "Blade.dat", "blade data\n")
    _write(shared / "AD_blade.dat", "aero blade data\n")
    for index in range(2):
        _write(
            shared / "Airfoils" / f"AF{index}.dat",
            _AIRFOIL.format(coords=f"AF{index}_coords.txt"),
        )
        _write(shared / "Airfoils" / f"AF{index}_coords.txt", "0.0 0.0\n")
    _write(shared / "Airfoils" / "AF9.dat", "unreferenced airfoil\n")
    _write(shared / "Ctrl" / "DISCON.dll", "library\n")
    _write(shared / "Ctrl" / "DISCON_x64.dll", "64-bit library\n")
    _write(shared / "Ctrl" / "DISCON.IN", _DISCON)
    _write(shared / "Ctrl" / "Cp_Ct_Cq.txt", "performance tables\n")
    _write(shared / "Ctrl" / "OTHER.dll", "unrelated library\n")
    _write(shared / "Ctrl" / "notes.txt", "unrelated sidecar\n")
    for suffix in (".1", ".3", ".hst", ".ss", ".ssexctn", ".out"):
        _write(shared / "Hydro" / f"platform{suffix}", f"wamit {suffix}\n")
    _write(shared / "Hydro" / "other.hst", "another platform\n")
    _write(root / "OtherCase" / "Other.fst", "unrelated case\n")
    _write(root / "OtherCase" / "Other.1.lin", "unrelated linearization\n")
    return root, fst


_EXPECTED = {
    "Case/Model.fst",
    "Case/Turbine ED.dat",
    "Case/Tower.dat",
    "Case/AeroDyn.dat",
    "Case/ServoDyn.dat",
    "Case/StC_a.dat",
    "Case/StC_b.dat",
    "Case/HydroDyn.dat",
    "Shared/InflowWind.dat",
    "Shared/Wind/turb.bts",
    "Shared/Wind/hawc.wnd",
    "Shared/Blade.dat",
    "Shared/AD_blade.dat",
    "Shared/Airfoils/AF0.dat",
    "Shared/Airfoils/AF1.dat",
    "Shared/Airfoils/AF0_coords.txt",
    "Shared/Airfoils/AF1_coords.txt",
    "Shared/Ctrl/DISCON.dll",
    "Shared/Ctrl/DISCON_x64.dll",
    "Shared/Ctrl/DISCON.IN",
    "Shared/Ctrl/Cp_Ct_Cq.txt",
    "Shared/Hydro/platform.1",
    "Shared/Hydro/platform.3",
    "Shared/Hydro/platform.hst",
    "Shared/Hydro/platform.ss",
    "Shared/Hydro/platform.ssexctn",
}


def _relative(closure: DeckFileClosure) -> set[str]:
    return {path.as_posix() for path in closure.relative_files}


class TestDeckFileClosure:
    """Tests for the recursive referenced-file walk."""

    def test_collects_exactly_the_referenced_files(self, tmp_path: Path) -> None:
        """Every reference kind is followed and nothing unreferenced is included."""
        root, fst = _build_tree(tmp_path)
        closure = deck_file_closure(fst, root)
        assert _relative(closure) == _EXPECTED
        assert closure.external == ()
        assert closure.total_bytes == sum(path.stat().st_size for path in closure.files)

    def test_files_are_absolute_and_sorted(self, tmp_path: Path) -> None:
        """The closure lists absolute paths under the root, in a stable order."""
        root, fst = _build_tree(tmp_path)
        closure = deck_file_closure(fst, root)
        assert all(path.is_absolute() for path in closure.files)
        keys = [os.path.normcase(str(path)) for path in closure.files]
        assert keys == sorted(keys)

    def test_cyclic_references_terminate(self, tmp_path: Path) -> None:
        """Two files that reference each other are each visited once."""
        root = tmp_path / "root"
        fst = _write(root / "A.fst", '"B.dat" EDFile - b\n')
        _write(root / "B.dat", '"A.fst" Loop - back to the deck\n')
        assert _relative(deck_file_closure(fst, root)) == {"A.fst", "B.dat"}

    def test_binary_files_are_not_scanned(self, tmp_path: Path) -> None:
        """A referenced binary file is copied but its bytes are not followed."""
        root = tmp_path / "root"
        fst = _write(root / "A.fst", '"wind.dat" InflowFile - wind\n')
        (root / "wind.dat").write_bytes(b"\0\0 other.dat \0")
        _write(root / "other.dat", "would only be found by scanning the binary\n")
        assert _relative(deck_file_closure(fst, root)) == {"A.fst", "wind.dat"}

    def test_description_prose_is_not_a_root_keyword(self, tmp_path: Path) -> None:
        """A ``root`` word in a description does not expand the line's values."""
        root = tmp_path / "root"
        fst = _write(root / "A.fst", "Turbine  Name  - located at the blade root\n")
        _write(root / "Turbine.dat", "would match a root-name expansion\n")
        assert _relative(deck_file_closure(fst, root)) == {"A.fst"}

    def test_reference_relative_to_the_deck_directory(self, tmp_path: Path) -> None:
        """A sub-file path written relative to the deck's directory is also found."""
        root = tmp_path / "root"
        fst = _write(root / "Case" / "A.fst", '"Sub/Servo.dat" ServoFile - servo\n')
        _write(root / "Case" / "Sub" / "Servo.dat", '"perf.txt" PerfFile - perf\n')
        _write(root / "Case" / "perf.txt", "tables\n")
        closure = deck_file_closure(fst, root)
        assert _relative(closure) == {
            "Case/A.fst",
            "Case/Sub/Servo.dat",
            "Case/perf.txt",
        }

    def test_absolute_reference_outside_root_is_external(self, tmp_path: Path) -> None:
        """An absolute path outside the root is recorded, not copied, not an error."""
        outside = _write(tmp_path / "elsewhere" / "Shared.dat", "shared\n")
        root = tmp_path / "root"
        fst = _write(root / "A.fst", f'"{outside}" InflowFile - wind\n')
        closure = deck_file_closure(fst, root)
        assert _relative(closure) == {"A.fst"}
        assert [path.name for path in closure.external] == ["Shared.dat"]

    def test_relative_reference_escaping_root_raises(self, tmp_path: Path) -> None:
        """A ``../`` reference out of the root cannot be reproduced in a case copy."""
        _write(tmp_path / "Shared.dat", "shared\n")
        root = tmp_path / "root"
        fst = _write(root / "A.fst", '"../Shared.dat" InflowFile - wind\n')
        with pytest.raises(DeckReferenceError, match="outside template_root"):
            deck_file_closure(fst, root)

    def test_root_name_escaping_root_raises(self, tmp_path: Path) -> None:
        """A potential-flow root name out of the root is an escape too."""
        _write(tmp_path / "Hydro" / "platform.hst", "wamit\n")
        root = tmp_path / "root"
        fst = _write(root / "A.fst", '"../Hydro/platform" PotFile - root name\n')
        with pytest.raises(DeckReferenceError, match="root"):
            deck_file_closure(fst, root)

    def test_unreadable_input_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An input that cannot be read makes the closure untrustworthy."""
        root = tmp_path / "root"
        fst = _write(root / "A.fst", '"B.dat" EDFile - b\n')
        _write(root / "B.dat", "data\n")
        real_open = Path.open

        def failing_open(self: Path, *args: object, **kwargs: object) -> object:
            if self.name == "B.dat":
                msg = "permission denied"
                raise PermissionError(msg)
            return real_open(self, *args, **kwargs)  # type: ignore[call-overload]

        monkeypatch.setattr(Path, "open", failing_open)
        with pytest.raises(DeckReferenceError, match="cannot read"):
            deck_file_closure(fst, root)

    @pytest.mark.skipif(os.name != "nt", reason="extended-length paths are Windows")
    def test_extended_length_root_is_normalized(self, tmp_path: Path) -> None:
        r"""A ``\\?\`` template root still yields a closure with collapsed ``..``."""
        root, fst = _build_tree(tmp_path)
        closure = deck_file_closure("\\\\?\\" + str(fst), "\\\\?\\" + str(root))
        assert _relative(closure) == _EXPECTED
        assert not str(closure.root).startswith("\\\\?\\")

    def test_deck_outside_root_raises(self, tmp_path: Path) -> None:
        """The deck must lie under the template root."""
        fst = _write(tmp_path / "A.fst", "deck\n")
        (tmp_path / "root").mkdir()
        with pytest.raises(ValueError, match="not inside template_root"):
            deck_file_closure(fst, tmp_path / "root")

    def test_missing_deck_raises(self, tmp_path: Path) -> None:
        """A missing deck is reported as such."""
        with pytest.raises(FileNotFoundError):
            deck_file_closure(tmp_path / "A.fst", tmp_path)


class TestCopyTemplateFiles:
    """Tests for copying a template into a case directory."""

    def test_referenced_copy_preserves_layout(self, tmp_path: Path) -> None:
        """Only the referenced files are copied, at their root-relative paths."""
        root, fst = _build_tree(tmp_path)
        target = tmp_path / "out" / "case"
        assert copy_template_files(fst, root, target) == "referenced"
        copied = {
            path.relative_to(target).as_posix()
            for path in target.rglob("*")
            if path.is_file()
        }
        assert copied == _EXPECTED

    def test_inputs_are_hard_linked(self, tmp_path: Path) -> None:
        """With linking on, an unedited input shares the template's inode."""
        root, fst = _build_tree(tmp_path)
        target = tmp_path / "out" / "case"
        copy_template_files(fst, root, target, link_inputs=True)
        source = root / "Shared" / "Blade.dat"
        assert (target / "Shared" / "Blade.dat").stat().st_ino == source.stat().st_ino

    def test_output_files_are_never_linked(self, tmp_path: Path) -> None:
        """A referenced file with an output suffix is copied, not hard-linked."""
        root = tmp_path / "root"
        fst = _write(root / "A.fst", '"seed.out" InitFile - initial outputs\n')
        source = _write(root / "seed.out", "results\n")
        target = tmp_path / "out" / "case"
        copy_template_files(fst, root, target)
        assert (target / "seed.out").stat().st_ino != source.stat().st_ino

    def test_link_failure_falls_back_to_copy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Where hard links are unsupported, the files are byte-copied instead."""
        root, fst = _build_tree(tmp_path)

        def no_link(src: object, dst: object) -> None:
            msg = "Invalid cross-device link"
            raise OSError(msg)

        monkeypatch.setattr(os, "link", no_link)
        target = tmp_path / "out" / "case"
        copy_template_files(fst, root, target)
        source = root / "Shared" / "Blade.dat"
        assert (target / "Shared" / "Blade.dat").stat().st_ino != source.stat().st_ino
        assert (target / "Case" / "Model.fst").is_file()

    def test_no_link_copies_bytes(self, tmp_path: Path) -> None:
        """With linking off every file is an independent copy."""
        root, fst = _build_tree(tmp_path)
        target = tmp_path / "out" / "case"
        copy_template_files(fst, root, target, link_inputs=False)
        source = root / "Shared" / "Blade.dat"
        assert (target / "Shared" / "Blade.dat").stat().st_ino != source.stat().st_ino

    def test_tree_mode_copies_everything(self, tmp_path: Path) -> None:
        """The conservative mode copies the whole template tree."""
        root, fst = _build_tree(tmp_path)
        target = tmp_path / "out" / "case"
        assert copy_template_files(fst, root, target, mode="tree") == "tree"
        assert (target / "OtherCase" / "Other.1.lin").is_file()
        assert (target / "Case" / "Unused.dat").is_file()

    def test_escape_falls_back_to_tree_with_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An untrustworthy closure is replaced by a whole-tree copy and a warning."""
        _write(tmp_path / "Shared.dat", "shared\n")
        root = tmp_path / "root"
        fst = _write(root / "A.fst", '"../Shared.dat" InflowFile - wind\n')
        _write(root / "Unreferenced.dat", "only the tree copy carries this\n")
        target = tmp_path / "out" / "case"
        with caplog.at_level(logging.WARNING, logger="vane.run.deck_files"):
            assert copy_template_files(fst, root, target) == "tree"
        assert "whole template tree" in caplog.text
        assert (target / "Unreferenced.dat").is_file()

    def test_invalid_mode_raises(self, tmp_path: Path) -> None:
        """An unknown copy mode is rejected before anything is copied."""
        root, fst = _build_tree(tmp_path)
        target = tmp_path / "out" / "case"
        with pytest.raises(ValueError, match="copy mode"):
            copy_template_files(fst, root, target, mode="all")  # type: ignore[arg-type]
        assert not target.exists()

    def test_existing_destination_raises(self, tmp_path: Path) -> None:
        """The destination must be fresh."""
        root, fst = _build_tree(tmp_path)
        target = tmp_path / "out" / "case"
        target.mkdir(parents=True)
        with pytest.raises(FileExistsError):
            copy_template_files(fst, root, target)


def _standstill() -> OperatingPoint:
    return OperatingPoint(
        wind_speed=0.0, rotor_speed_rpm=0.0, blade_pitch_deg=0.0, name="pt"
    )


class TestRegressionSuiteClosure:
    """The walk on real regression-test decks (skipped when r-test is absent)."""

    def _deck(self, rtest_dir: Path, name: str) -> tuple[Path, Path]:
        root = rtest_dir / "glue-codes" / "openfast"
        fst = root / name / f"{name}.fst"
        if not fst.is_file():
            pytest.skip(f"r-test case not found: {fst}")
        return root, fst

    def test_floating_deck_closure(self, rtest_dir: Path) -> None:
        """The OC4 semi closure has its WAMIT data, controller and airfoils only."""
        root, fst = self._deck(rtest_dir, "5MW_OC4Semi_WSt_WavesWN")
        closure = deck_file_closure(fst, root)
        names = _relative(closure)
        assert "5MW_Baseline/HydroData/marin_semi.hst" in names
        assert "5MW_Baseline/HydroData/marin_semi.1" in names
        assert "5MW_Baseline/ServoData/DISCON_OC3Hywind.dll" in names
        assert any(name.startswith("5MW_Baseline/Airfoils/") for name in names)
        assert all(
            name.startswith(("5MW_OC4Semi_WSt_WavesWN/", "5MW_Baseline/"))
            for name in names
        )
        assert not any(name.endswith((".lin", ".outb")) for name in names)

    def test_driver_case_holds_only_the_deck(
        self, rtest_dir: Path, tmp_path: Path
    ) -> None:
        """A driver case under the whole suite root copies only its own deck tree."""
        root, fst = self._deck(rtest_dir, "5MW_Land_DLL_WTurb")
        driver = LinearizationDriver(
            fst_path=fst, output_dir=tmp_path / "out", template_root=root
        )
        deck = driver.generate_case(_standstill())
        case = tmp_path / "out" / "pt"
        assert deck == case / "5MW_Land_DLL_WTurb" / "5MW_Land_DLL_WTurb.fst"
        assert sorted(path.name for path in case.iterdir()) == [
            "5MW_Baseline",
            "5MW_Land_DLL_WTurb",
        ]
        assert not (case / "5MW_Land_DLL_WTurb" / "5MW_Land_DLL_WTurb.outb").exists()


class TestOpenPaths:
    """The unnormalized path strings OpenFAST opens each referenced file by."""

    def _open_paths(self, tmp_path: Path) -> tuple[Path, dict[str, str]]:
        root, fst = _build_tree(tmp_path)
        closure = deck_file_closure(fst, root)
        opened = {
            path.relative_to(root).as_posix(): text for path, text in closure.open_paths
        }
        return fst.parent, opened

    def test_every_referenced_file_has_an_open_path(self, tmp_path: Path) -> None:
        """Each referenced file carries the spelling OpenFAST uses.

        Library variants (``DISCON_x64.dll``) are copied but never opened.
        """
        _, opened = self._open_paths(tmp_path)
        assert set(opened) == _EXPECTED - {"Shared/Ctrl/DISCON_x64.dll"}

    def test_deck_is_opened_by_its_name_in_the_working_directory(
        self, tmp_path: Path
    ) -> None:
        """The deck itself is the working directory joined to its bare name."""
        case, opened = self._open_paths(tmp_path)
        assert opened["Case/Model.fst"] == f"{case}{os.sep}Model.fst"

    def test_deck_references_carry_the_dot_directory(self, tmp_path: Path) -> None:
        """The deck's references are prefixed with ``.`` and kept unnormalized."""
        case, opened = self._open_paths(tmp_path)
        prefix = f"{case}{os.sep}.{os.sep}"
        assert opened["Case/Turbine ED.dat"] == f"{prefix}Turbine ED.dat"
        assert opened["Shared/InflowWind.dat"] == f"{prefix}../Shared/InflowWind.dat"

    def test_nested_references_inherit_the_unnormalized_directory(
        self, tmp_path: Path
    ) -> None:
        """A sub-file's references keep the ``..`` chain of their parent."""
        case, opened = self._open_paths(tmp_path)
        prefix = f"{case}{os.sep}.{os.sep}"
        assert opened["Shared/Wind/turb.bts"] == f"{prefix}../Shared/Wind/turb.bts"
        assert opened["Shared/Airfoils/AF0_coords.txt"] == (
            f"{prefix}../Shared/Airfoils/AF0_coords.txt"
        )
        # Reached from the ElastoDyn file in the deck directory.
        assert opened["Shared/Blade.dat"] == f"{prefix}../Shared/Blade.dat"

    def test_root_name_expansion_keeps_the_suffix(self, tmp_path: Path) -> None:
        """Files expanded from a root name are spelled root plus suffix."""
        case, opened = self._open_paths(tmp_path)
        prefix = f"{case}{os.sep}.{os.sep}"
        assert opened["Shared/Hydro/platform.hst"] == (
            f"{prefix}../Shared/Hydro/platform.hst"
        )
        assert opened["Shared/Wind/hawc.wnd"] == f"{prefix}../Shared/Wind/hawc.wnd"

    def test_working_directory_fallback(self, tmp_path: Path) -> None:
        """A reference found only beside the deck is spelled from there."""
        root = tmp_path / "suite"
        case = root / "Case"
        fst = _write(case / "Model.fst", '"../Shared/Servo.dat" ServoFile - servo\n')
        _write(root / "Shared" / "Servo.dat", '"Ctrl.in" DLL_InFile - controller\n')
        _write(case / "Ctrl.in", "controller parameters\n")
        closure = deck_file_closure(fst, root)
        opened = {path.name: text for path, text in closure.open_paths}
        assert opened["Ctrl.in"] == f"{case}{os.sep}.{os.sep}Ctrl.in"

    def test_absolute_reference_is_spelled_as_written(self, tmp_path: Path) -> None:
        """An absolute reference is opened by its own text."""
        outside = _write(tmp_path / "elsewhere" / "Wind.dat", "wind\n")
        root = tmp_path / "suite"
        fst = _write(root / "Case" / "Model.fst", f'"{outside}" InflowFile - wind\n')
        closure = deck_file_closure(fst, root)
        opened = {path.name: text for path, text in closure.open_paths}
        assert opened["Wind.dat"] == str(outside)

    def test_longest_spelling_is_kept(self, tmp_path: Path) -> None:
        """A file reached by two references keeps its longer spelling."""
        root = tmp_path / "suite"
        case = root / "Case"
        fst = _write(
            case / "Model.fst",
            '"Shared.dat" EDFile - short\n"../Case/Shared.dat" AeroFile - long\n',
        )
        _write(case / "Shared.dat", "shared\n")
        closure = deck_file_closure(fst, root)
        opened = {path.name: text for path, text in closure.open_paths}
        assert opened["Shared.dat"] == f"{case}{os.sep}.{os.sep}../Case/Shared.dat"


class TestInferTemplateRoot:
    """The copy root inferred from where the deck's relative references lead."""

    def test_shared_inputs_widen_the_root(self, tmp_path: Path) -> None:
        """A deck reading ../Shared/... is copied from the common ancestor."""
        root, fst = _build_tree(tmp_path)
        assert infer_template_root(fst) == root.resolve()

    def test_self_contained_deck_keeps_its_directory(self, tmp_path: Path) -> None:
        """Without references outside its folder the deck's directory is the root."""
        fst = _write(tmp_path / "Case" / "Model.fst", '"ED.dat" EDFile - ElastoDyn\n')
        _write(tmp_path / "Case" / "ED.dat", '"Blade.dat" BldFile(1) - blade\n')
        _write(tmp_path / "Case" / "Blade.dat", "blade\n")
        _write(tmp_path / "Other.dat", "an unrelated file above the deck\n")
        assert infer_template_root(fst) == (tmp_path / "Case").resolve()

    def test_absolute_references_do_not_widen_the_root(self, tmp_path: Path) -> None:
        """An absolute reference, and its own relative references, stay in place."""
        shared = tmp_path / "Shared"
        _write(shared / "SD.dat", '"libdiscon.dll" DLL_FileName - library\n')
        _write(shared / "libdiscon.dll", "MZ")
        servo = (shared / "SD.dat").as_posix()
        fst = _write(
            tmp_path / "Case" / "Model.fst", f'"{servo}" ServoFile - ServoDyn\n'
        )
        assert infer_template_root(fst) == (tmp_path / "Case").resolve()

    def test_library_variants_count_as_references(self, tmp_path: Path) -> None:
        """A controller library found only as a bitness variant still counts."""
        _write(tmp_path / "Ctrl" / "DISCON_x64.dll", "MZ")
        fst = _write(
            tmp_path / "Case" / "Model.fst",
            '"../Ctrl/DISCON.dll" DLL_FileName - library\n',
        )
        assert infer_template_root(fst) == tmp_path.resolve()

    def test_unreadable_reference_keeps_what_was_found(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A walk that cannot read an input still uses the references found."""
        root, fst = _build_tree(tmp_path)
        original = Path.open

        def guarded(self: Path, *args: object, **kwargs: object) -> object:
            if self.name == "Turbine ED.dat":
                msg = "locked"
                raise PermissionError(msg)
            return original(self, *args, **kwargs)  # type: ignore[call-overload]

        monkeypatch.setattr(Path, "open", guarded)
        with caplog.at_level(logging.WARNING, logger="vane.run.deck_files"):
            inferred = infer_template_root(fst)
        assert "could not walk" in caplog.text
        assert inferred in (root.resolve(), fst.parent.resolve())

    def test_missing_deck(self, tmp_path: Path) -> None:
        """A missing deck is an error."""
        with pytest.raises(FileNotFoundError):
            infer_template_root(tmp_path / "absent.fst")

    @pytest.mark.skipif(os.name != "nt", reason="drive-less rooted paths are Windows")
    def test_drive_rooted_reference_is_absolute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        r"""``\Shared\...`` is absolute on the deck's drive and does not widen the root.

        Python 3.13's ``ntpath.isabs`` reports such a path as relative; the walk
        must not, whatever the interpreter.
        """
        import ntpath

        from vane.run import deck_files

        isabs = ntpath.isabs
        monkeypatch.setattr(
            deck_files.ntpath,
            "isabs",
            lambda text: ntpath.splitdrive(text)[0] != "" and isabs(text),
        )
        shared = tmp_path / "Shared" / "ED.dat"
        _write(shared, "ElastoDyn\n")
        rooted = str(shared)[len(shared.drive) :]
        fst = _write(tmp_path / "Case" / "Model.fst", f'"{rooted}" EDFile - ED\n')
        assert infer_template_root(fst) == (tmp_path / "Case").resolve()
        closure = deck_file_closure(fst, tmp_path / "Case")
        assert shared.resolve() in closure.external

    def test_regression_suite_deck(self, rtest_dir: Path) -> None:
        """The r-test land deck reads ../5MW_Baseline: the suite folder is the root."""
        suite = rtest_dir / "glue-codes" / "openfast"
        fst = suite / "5MW_Land_DLL_WTurb" / "5MW_Land_DLL_WTurb.fst"
        if not fst.is_file():
            pytest.skip(f"r-test case not found: {fst}")
        assert infer_template_root(fst) == suite.resolve()
