# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.run.driver — the end-to-end linearization driver."""

from __future__ import annotations

import json
from dataclasses import fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from vane.io.fst_writer import FastInputFile
from vane.run import (
    DeckValidationError,
    LinearizationDriver,
    LinearizationRetryPolicy,
    LinearizationSettings,
    OperatingPoint,
    SweepCase,
    TrimTuningPolicy,
    default_max_workers,
)
from vane.run.lin_config import MIN_CALC_STEADY_TMAX
from vane.run.runner import (
    OpenFastDiagnosis,
    OpenFastFailureMode,
    OpenFastResult,
    OpenFastTimeoutError,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_PRIMARY = """\
False Linearize - flag
False CalcSteady - flag
1 NLinTimes - -
0 LinTimes - s
3 TrimCase - -
0.01 TrimGain - -
0.001 TrimTol - -
1 CompInflow - flag
"ED.dat" EDFile - ElastoDyn file
"IW.dat" InflowFile - InflowWind file
"""

_ED = """\
9.55 RotSpeed - rpm
0.0 BlPitch(1) - deg
0.0 BlPitch(2) - deg
0.0 BlPitch(3) - deg
"""

# Minimal steady-wind InflowWind sub-file so a wind>0 operating point has a usable
# wind file to set its speed into (a zero-wind point simply leaves it untouched).
_IW = """\
1 WindType - switch
8 HWindSpeed - horizontal wind speed (m/s)
0 PLExp - power law exponent
"""


def _make_template(tmp_path: Path) -> Path:
    """Write a minimal self-contained template case and return its .fst path."""
    root = tmp_path / "template"
    root.mkdir()
    (root / "ED.dat").write_text(_ED, encoding="utf-8")
    (root / "IW.dat").write_text(_IW, encoding="utf-8")
    fst = root / "Model.fst"
    fst.write_text(_PRIMARY, encoding="utf-8")
    return fst


_SERVO_DLL = """\
5 PCMode - pitch control mode {5=DLL}
1 VSContrl - torque control mode
0 HSSBrMode - HSS brake mode
0 YCMode - yaw control mode
"""


def _make_dll_template(tmp_path: Path) -> Path:
    """Write a template whose ServoDyn uses the non-linearizable DLL controller."""
    root = tmp_path / "template"
    root.mkdir()
    (root / "ED.dat").write_text(_ED, encoding="utf-8")
    (root / "SD.dat").write_text(_SERVO_DLL, encoding="utf-8")
    fst = root / "Model.fst"
    fst.write_text(_PRIMARY + '"SD.dat" ServoFile - ServoDyn file\n', encoding="utf-8")
    return fst


def _standstill(name: str) -> OperatingPoint:
    """Return a standstill operating point with a given name."""
    return OperatingPoint(
        wind_speed=0.0, rotor_speed_rpm=0.0, blade_pitch_deg=0.0, name=name
    )


class TestSharedInputConfinement:
    """A case may never edit a sub-file shared outside its own copy."""

    def _outside_ed_template(self, tmp_path: Path) -> tuple[Path, Path]:
        root = tmp_path / "models" / "template"
        root.mkdir(parents=True)
        shared = tmp_path / "models" / "ED.dat"
        shared.write_text(_ED, encoding="utf-8")
        (root / "IW.dat").write_text(_IW, encoding="utf-8")
        fst = root / "Model.fst"
        fst.write_text(
            _PRIMARY.replace('"ED.dat" EDFile', '"../ED.dat" EDFile'), encoding="utf-8"
        )
        return fst, shared

    def test_driver_rejects_a_sub_file_outside_the_case(self, tmp_path: Path) -> None:
        """A ``../`` ElastoDyn file would be shared, so the case is rejected."""
        from vane.io.fst_writer import FastInputError

        fst, shared = self._outside_ed_template(tmp_path)
        before = shared.read_text(encoding="utf-8")
        driver = LinearizationDriver(fst_path=fst, output_dir=tmp_path / "out")
        with pytest.raises(FastInputError, match="outside the case directory"):
            driver.generate_case(_standstill("pt1"))
        assert shared.read_text(encoding="utf-8") == before

    def test_template_root_containing_inputs_is_accepted(self, tmp_path: Path) -> None:
        """With a copy root that contains the shared file, each case edits its copy."""
        fst, shared = self._outside_ed_template(tmp_path)
        before = shared.read_text(encoding="utf-8")
        driver = LinearizationDriver(
            fst_path=fst,
            output_dir=tmp_path / "out",
            template_root=tmp_path / "models",
        )
        deck = driver.generate_case(_standstill("pt1"))
        assert deck.is_file()
        assert shared.read_text(encoding="utf-8") == before

    def test_shared_read_only_module_file_is_allowed(self, tmp_path: Path) -> None:
        """An outside AeroDyn file is fine until the settings would edit it."""
        from vane.io.fst_writer import FastInputError

        root = tmp_path / "models" / "template"
        root.mkdir(parents=True)
        shared_aero = tmp_path / "models" / "AD.dat"
        shared_aero.write_text("0 UA_Mod - unsteady aero\n", encoding="utf-8")
        (root / "ED.dat").write_text(_ED, encoding="utf-8")
        (root / "IW.dat").write_text(_IW, encoding="utf-8")
        fst = root / "Model.fst"
        fst.write_text(_PRIMARY + '"../AD.dat" AeroFile - AeroDyn file\n', "utf-8")
        before = shared_aero.read_text(encoding="utf-8")

        reader = LinearizationDriver(fst_path=fst, output_dir=tmp_path / "out")
        assert reader.generate_case(_standstill("read")).is_file()

        writer = LinearizationDriver(
            fst_path=fst,
            output_dir=tmp_path / "out2",
            settings=LinearizationSettings(rated_wind_speed=11.4, unsteady_aero_mode=0),
        )
        with pytest.raises(FastInputError, match="AeroFile"):
            writer.generate_case(_standstill("write"))
        assert shared_aero.read_text(encoding="utf-8") == before

    @pytest.mark.parametrize(("comp_servo", "rejected"), [(0, False), (1, True)])
    def test_disabled_servodyn_file_may_stay_shared(
        self, tmp_path: Path, comp_servo: int, rejected: bool
    ) -> None:
        """With CompServo = 0 the trim never touches the (outside) ServoDyn file."""
        from vane.io.fst_writer import FastInputError

        root = tmp_path / "models" / "template"
        root.mkdir(parents=True)
        shared_servo = tmp_path / "models" / "SD.dat"
        shared_servo.write_text(_SERVO_DLL, encoding="utf-8")
        (root / "ED.dat").write_text(_ED, encoding="utf-8")
        (root / "IW.dat").write_text(_IW, encoding="utf-8")
        fst = root / "Model.fst"
        fst.write_text(
            _PRIMARY
            + f"{comp_servo} CompServo - servo switch\n"
            + '"../SD.dat" ServoFile - ServoDyn file\n',
            "utf-8",
        )
        driver = LinearizationDriver(
            fst_path=fst,
            output_dir=tmp_path / "out",
            settings=LinearizationSettings(rated_wind_speed=11.4, trim_controller=True),
        )
        if rejected:
            with pytest.raises(FastInputError, match="ServoFile"):
                driver.generate_case(_standstill("pt"))
        else:
            assert driver.generate_case(_standstill("pt")).is_file()
        assert shared_servo.read_text(encoding="utf-8") == _SERVO_DLL
        # The public helper's default key set honours the same switch.
        from vane.run import require_inputs_within

        if rejected:
            with pytest.raises(FastInputError, match="ServoFile"):
                require_inputs_within(fst, root)
        else:
            require_inputs_within(fst, root)

    def test_configure_linearization_confine_to(self, tmp_path: Path) -> None:
        """A direct call can opt into the same confinement check."""
        from vane.io.fst_writer import FastInputError
        from vane.run import configure_linearization

        fst, shared = self._outside_ed_template(tmp_path)
        before = shared.read_text(encoding="utf-8")
        with pytest.raises(FastInputError, match="outside the case directory"):
            configure_linearization(fst, _standstill("pt"), confine_to=fst.parent)
        assert shared.read_text(encoding="utf-8") == before


class TestGenerateCase:
    """Tests for copying and configuring a case."""

    def test_copies_tree_and_configures(self, tmp_path: Path) -> None:
        """generate_case copies the template and configures the copied deck."""
        fst = _make_template(tmp_path)
        driver = LinearizationDriver(fst_path=fst, output_dir=tmp_path / "out")
        deck_path = driver.generate_case(_standstill("pt1"))

        assert deck_path == (tmp_path / "out" / "pt1" / "Model.fst")
        assert deck_path.is_file()
        deck = FastInputFile.read(deck_path)
        assert deck.get("Linearize") == "True"
        assert deck.get("NLinTimes") == "1"
        # The copied ED sub-file is configured, leaving the template untouched.
        ed = FastInputFile.read(tmp_path / "out" / "pt1" / "ED.dat")
        assert ed.get("RotSpeed") == "0.0"
        assert FastInputFile.read(fst.parent / "ED.dat").get("RotSpeed") == "9.55"

    def test_regenerate_removes_stale_files(self, tmp_path: Path) -> None:
        """Re-generating a case starts from a clean copy, dropping stale files."""
        fst = _make_template(tmp_path)
        driver = LinearizationDriver(fst_path=fst, output_dir=tmp_path / "out")
        case_dir = driver.case_dir(_standstill("pt1"))
        driver.generate_case(_standstill("pt1"))
        stale = case_dir / "leftover.lin"
        stale.write_text("stale", encoding="utf-8")
        driver.generate_case(_standstill("pt1"))
        assert not stale.exists()
        assert (case_dir / "Model.fst").is_file()


def _make_template_with_heavy_file(tmp_path: Path) -> tuple[Path, Path]:
    """Write a template with a heavy read-only file; return (fst, heavy file)."""
    root = tmp_path / "template"
    root.mkdir()
    (root / "ED.dat").write_text(_ED, encoding="utf-8")
    heavy = root / "data.bin"
    heavy.write_text("heavy read-only payload", encoding="utf-8")
    fst = root / "Model.fst"
    # The deck references the heavy file, so the referenced-file copy includes it.
    fst.write_text(_PRIMARY + '"data.bin" DataFile - data\n', encoding="utf-8")
    return fst, heavy


class TestLinkInputs:
    """Tests for hard-linking shared inputs to avoid per-case duplication."""

    def test_shares_unedited_files_and_preserves_template(self, tmp_path: Path) -> None:
        """A heavy file is hard-linked; the edited deck stays independent."""
        fst, heavy = _make_template_with_heavy_file(tmp_path)
        driver = LinearizationDriver(
            fst_path=fst, output_dir=tmp_path / "out", link_inputs=True
        )
        driver.generate_case(_standstill("pt1"))
        case = tmp_path / "out" / "pt1"
        # The heavy, unedited input is hard-linked (shared inode), not duplicated.
        assert (case / "data.bin").stat().st_ino == heavy.stat().st_ino
        # The edited deck got a new inode (atomic write broke the link); the template
        # files are untouched.
        assert (case / "Model.fst").stat().st_ino != fst.stat().st_ino
        assert FastInputFile.read(fst).get("Linearize") == "False"
        assert FastInputFile.read(fst.parent / "ED.dat").get("RotSpeed") == "9.55"
        assert FastInputFile.read(case / "Model.fst").get("Linearize") == "True"

    def test_output_artifacts_are_copied_not_linked(self, tmp_path: Path) -> None:
        """Stale OpenFAST output files are copied, never hard-linked to the template."""
        fst, _heavy = _make_template_with_heavy_file(tmp_path)
        # A stale run artifact left in the template (same RootName OpenFAST will write).
        stale_out = fst.parent / "Model.out"
        stale_out.write_text("stale results", encoding="utf-8")
        driver = LinearizationDriver(
            fst_path=fst,
            output_dir=tmp_path / "out",
            link_inputs=True,
            template_copy="tree",
        )
        driver.generate_case(_standstill("pt1"))
        case_out = tmp_path / "out" / "pt1" / "Model.out"
        # The output artifact is an independent copy, not a hard link to the template.
        assert case_out.stat().st_ino != stale_out.stat().st_ino

    def test_referenced_copy_leaves_stale_outputs_behind(self, tmp_path: Path) -> None:
        """The default copy carries only referenced inputs, never stale outputs."""
        fst, _heavy = _make_template_with_heavy_file(tmp_path)
        (fst.parent / "Model.out").write_text("stale results", encoding="utf-8")
        (fst.parent / "Model.1.lin").write_text("stale lin", encoding="utf-8")
        driver = LinearizationDriver(fst_path=fst, output_dir=tmp_path / "out")
        driver.generate_case(_standstill("pt1"))
        case = tmp_path / "out" / "pt1"
        assert sorted(path.name for path in case.iterdir()) == [
            "ED.dat",
            "Model.fst",
            "data.bin",
        ]

    def test_tree_copy_falls_back_to_bytes_when_linking_unsupported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole-tree copy also falls back to a byte copy when linking fails."""
        import os

        fst, heavy = _make_template_with_heavy_file(tmp_path)

        def no_link(src: object, dst: object) -> None:
            msg = "Invalid cross-device link"
            raise OSError(msg)

        monkeypatch.setattr(os, "link", no_link)
        driver = LinearizationDriver(
            fst_path=fst, output_dir=tmp_path / "out", template_copy="tree"
        )
        driver.generate_case(_standstill("pt1"))
        copied = tmp_path / "out" / "pt1" / "data.bin"
        assert copied.stat().st_ino != heavy.stat().st_ino

    def test_invalid_template_copy_mode_is_rejected(self, tmp_path: Path) -> None:
        """An unknown template_copy value fails at construction."""
        fst = _make_template(tmp_path)
        with pytest.raises(ValueError, match="template_copy"):
            LinearizationDriver(
                fst_path=fst,
                output_dir=tmp_path / "out",
                template_copy="everything",  # type: ignore[arg-type]
            )

    def test_link_inputs_false_copies_bytes(self, tmp_path: Path) -> None:
        """With link_inputs disabled, every file is an independent byte copy."""
        fst, heavy = _make_template_with_heavy_file(tmp_path)
        driver = LinearizationDriver(
            fst_path=fst, output_dir=tmp_path / "out", link_inputs=False
        )
        driver.generate_case(_standstill("pt1"))
        copied = tmp_path / "out" / "pt1" / "data.bin"
        assert copied.stat().st_ino != heavy.stat().st_ino

    def test_falls_back_to_copy_when_linking_unsupported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If hard-linking fails (e.g. cross-filesystem), it copies bytes instead."""
        import os

        fst, heavy = _make_template_with_heavy_file(tmp_path)

        def no_link(src: object, dst: object) -> None:
            msg = "Invalid cross-device link"
            raise OSError(msg)

        # The driver calls os.link via copytree's copy_function; patch it globally.
        monkeypatch.setattr(os, "link", no_link)
        driver = LinearizationDriver(
            fst_path=fst, output_dir=tmp_path / "out", link_inputs=True
        )
        deck = driver.generate_case(_standstill("pt1"))  # must not raise
        assert deck.is_file()
        copied = tmp_path / "out" / "pt1" / "data.bin"
        assert copied.is_file()
        assert copied.stat().st_ino != heavy.stat().st_ino  # fell back to a real copy


class TestDriverValidation:
    """Tests for the driver's path validation."""

    def test_fst_outside_template_root_raises(self, tmp_path: Path) -> None:
        """A deck not inside template_root is rejected."""
        fst = _make_template(tmp_path)
        with pytest.raises(ValueError, match="not inside template_root"):
            LinearizationDriver(
                fst_path=fst,
                output_dir=tmp_path / "out",
                template_root=tmp_path / "elsewhere",
            )

    def test_output_inside_template_raises(self, tmp_path: Path) -> None:
        """An output_dir inside template_root would recurse and is rejected."""
        fst = _make_template(tmp_path)
        with pytest.raises(ValueError, match="recursive copy"):
            LinearizationDriver(fst_path=fst, output_dir=fst.parent / "out")

    def test_template_under_output_workspace_is_allowed(self, tmp_path: Path) -> None:
        """A template living under the output workspace is a valid sibling layout."""
        fst = _make_template(tmp_path)  # template_root == tmp_path/template
        # output_dir is the workspace that contains the template directory.
        driver = LinearizationDriver(fst_path=fst, output_dir=tmp_path)
        assert driver.generate_case(_standstill("ws8")).is_file()

    def test_case_name_colliding_with_template_raises(self, tmp_path: Path) -> None:
        """A case name that resolves to the template root is refused (not deleted)."""
        fst = _make_template(tmp_path)  # template dir is tmp_path/"template"
        driver = LinearizationDriver(fst_path=fst, output_dir=tmp_path)
        with pytest.raises(ValueError, match="overlap the template root"):
            driver.generate_case(_standstill("template"))
        # The template survives.
        assert (fst.parent / "Model.fst").is_file()

    @pytest.mark.parametrize("name", [".", "..", "...", "/", "../.."])
    def test_reserved_case_names_stay_inside_output(
        self, tmp_path: Path, name: str
    ) -> None:
        """A reserved/path-escaping case name resolves to a real subdirectory."""
        fst = _make_template(tmp_path)
        out = tmp_path / "out"
        driver = LinearizationDriver(fst_path=fst, output_dir=out)
        op = OperatingPoint(
            wind_speed=8.0, rotor_speed_rpm=9.55, blade_pitch_deg=0.0, name=name
        )
        case_dir = driver.case_dir(op)
        assert case_dir.parent == out.resolve()
        assert case_dir != out.resolve()
        # The name is a real directory component, not a reserved path element.
        assert case_dir.name not in ("", ".", "..")


class TestRunSweep:
    """Tests for sweep orchestration (with the run mocked)."""

    def test_runs_each_point_in_its_own_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_sweep generates and runs one case per point, in order."""
        from vane.run import driver as driver_module

        fst = _make_template(tmp_path)
        drv = LinearizationDriver(fst_path=fst, output_dir=tmp_path / "out")

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            return OpenFastResult(fst_path=deck_path, returncode=0)

        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        cases = drv.run_sweep([_standstill("a"), _standstill("b")])

        assert [c.operating_point.name for c in cases] == ["a", "b"]
        assert all(c.succeeded for c in cases)
        assert (tmp_path / "out" / "a" / "Model.fst").is_file()
        assert (tmp_path / "out" / "b" / "Model.fst").is_file()
        assert cases[0].case_dir == tmp_path / "out" / "a"

    def test_duplicate_case_names_raise(self, tmp_path: Path) -> None:
        """Two points mapping to the same case directory are rejected."""
        fst = _make_template(tmp_path)
        drv = LinearizationDriver(fst_path=fst, output_dir=tmp_path / "out")
        with pytest.raises(ValueError, match="duplicate case names"):
            drv.run_sweep([_standstill("same"), _standstill("same")])

    @pytest.mark.parametrize(
        ("first", "second"), [("Low", "low"), ("ws8.", "ws8"), ("CASE..", "case")]
    )
    def test_filesystem_equivalent_case_names_raise_before_any_run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        first: str,
        second: str,
    ) -> None:
        """Names one directory on Windows/macOS are rejected before any case runs.

        Case-insensitive filesystems (and Windows' trailing-dot stripping) map both
        names to one directory; the second case would rmtree the first case's
        results and its ``.lin`` files would be attributed to the wrong point.
        """
        from vane.run import driver as driver_module

        fst = _make_template(tmp_path)
        drv = LinearizationDriver(fst_path=fst, output_dir=tmp_path / "out")

        def fail_run(deck_path: Path, **_: object) -> OpenFastResult:
            pytest.fail("no case may run when case names collide")

        monkeypatch.setattr(driver_module, "run_openfast", fail_run)
        with pytest.raises(ValueError, match="duplicate case names"):
            drv.run_sweep([_standstill(first), _standstill(second)])
        assert not (tmp_path / "out").exists()

    def test_trailing_dots_are_dropped_from_case_names(self, tmp_path: Path) -> None:
        """The case directory is named as Windows would actually create it."""
        fst = _make_template(tmp_path)
        drv = LinearizationDriver(fst_path=fst, output_dir=tmp_path / "out")
        assert drv.case_dir(_standstill("ws8.")).name == "ws8"

    @pytest.mark.parametrize("name", ["CON", "nul", "com1", "LPT9.txt", "aux.lin"])
    def test_reserved_windows_device_names_raise(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
    ) -> None:
        """A case directory Windows cannot create is rejected on every platform."""
        from vane.run import driver as driver_module

        fst = _make_template(tmp_path)
        drv = LinearizationDriver(fst_path=fst, output_dir=tmp_path / "out")

        def fail_run(deck_path: Path, **_: object) -> OpenFastResult:
            pytest.fail("no case may run under a reserved device name")

        monkeypatch.setattr(driver_module, "run_openfast", fail_run)
        with pytest.raises(ValueError, match="reserved Windows device name"):
            drv.run_sweep([_standstill(name)])
        with pytest.raises(ValueError, match="reserved Windows device name"):
            drv.generate_case(_standstill(name))

    def test_parallel_sweep_runs_all_points_in_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A parallel sweep runs every point and returns results in input order."""
        from vane.run import driver as driver_module

        fst = _make_template(tmp_path)
        drv = LinearizationDriver(fst_path=fst, output_dir=tmp_path / "out")

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            return OpenFastResult(fst_path=deck_path, returncode=0)

        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        points = [_standstill(name) for name in ("a", "b", "c", "d")]
        cases = drv.run_sweep(points, max_workers=4)

        assert [c.operating_point.name for c in cases] == ["a", "b", "c", "d"]
        assert all(c.succeeded for c in cases)
        for name in ("a", "b", "c", "d"):
            assert (tmp_path / "out" / name / "Model.fst").is_file()

    def test_default_workers_uses_cpu_count_minus_reserve(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The sweep default parallelism is CPU-aware, not a hard-coded small count."""
        from vane.run import driver as driver_module

        fst = _make_template(tmp_path)
        drv = LinearizationDriver(fst_path=fst, output_dir=tmp_path / "out")
        seen: list[int] = []

        class RecordingPool:
            def __init__(self, max_workers: int) -> None:
                seen.append(max_workers)

            def __enter__(self) -> RecordingPool:
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def submit(self, fn: object, *args: object, **kwargs: object) -> object:
                class Future:
                    def result(self) -> tuple[SweepCase, None]:
                        op = cast("OperatingPoint", args[0])
                        deck = tmp_path / f"{op.name}.fst"
                        case = SweepCase(
                            operating_point=op,
                            case_dir=tmp_path / op.name,
                            fst_path=deck,
                            result=OpenFastResult(fst_path=deck, returncode=0),
                        )
                        return case, None

                return Future()

        monkeypatch.setattr("os.cpu_count", lambda: 8)
        monkeypatch.setattr(driver_module, "ThreadPoolExecutor", RecordingPool)
        drv.run_sweep([_standstill("a"), _standstill("b")])

        assert default_max_workers() == 7
        assert seen == [7]

    def test_missing_calcsteady_azimuth_files_retries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A rotating CalcSteady run that emits one .lin is treated as non-converged."""
        from vane.run import driver as driver_module

        fst = _make_template(tmp_path)
        drv = LinearizationDriver(
            fst_path=fst,
            output_dir=tmp_path / "out",
            settings=LinearizationSettings(rated_wind_speed=11.4, n_lin_times=3),
            retry_policy=LinearizationRetryPolicy(max_attempts=2),
            validate=False,
        )
        calls: list[Path] = []

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            calls.append(deck_path)
            lin_count = 1 if len(calls) == 1 else 3
            lin_files = [
                deck_path.with_suffix(f".{index}.lin")
                for index in range(1, lin_count + 1)
            ]
            return OpenFastResult(fst_path=deck_path, returncode=0, lin_files=lin_files)

        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        op = OperatingPoint(8.0, 9.55, 0.0, name="ws8")
        case = drv.run_case(op)

        assert len(calls) == 2
        assert case.succeeded
        assert case.result.n_lin == 3
        assert case.settings.trim_tol > drv.settings.trim_tol
        assert case.attempts == 2
        assert case.converged is True

    def test_incomplete_explicit_time_sweep_retries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An untrimmed rotating sweep requires every requested explicit snapshot."""
        from vane.run import driver as driver_module

        fst = _make_template(tmp_path)
        settings = LinearizationSettings(
            rated_wind_speed=11.4,
            n_lin_times=3,
            calc_steady=False,
            sim_time=30.0,
            linearization_times=(20.0, 21.0, 22.0),
        )
        drv = LinearizationDriver(
            fst_path=fst,
            output_dir=tmp_path / "out",
            settings=settings,
            retry_policy=LinearizationRetryPolicy(max_attempts=2),
            validate=False,
        )
        calls: list[Path] = []

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            calls.append(deck_path)
            lin_count = 1 if len(calls) == 1 else 3
            lin_files = [
                deck_path.with_suffix(f".{index}.lin")
                for index in range(1, lin_count + 1)
            ]
            return OpenFastResult(fst_path=deck_path, returncode=0, lin_files=lin_files)

        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        case = drv.run_case(OperatingPoint(18.0, 7.56, 15.46, name="ws18"))

        assert len(calls) == 2
        assert case.converged is True
        assert case.result.n_lin == 3

    def test_timeout_is_retried_then_succeeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run that times out is a retry reason, not a sweep-aborting exception."""
        from vane.run import driver as driver_module

        fst = _make_template(tmp_path)
        drv = LinearizationDriver(
            fst_path=fst,
            output_dir=tmp_path / "out",
            settings=LinearizationSettings(rated_wind_speed=11.4, n_lin_times=3),
            retry_policy=LinearizationRetryPolicy(max_attempts=2),
            validate=False,
        )
        calls: list[Path] = []

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            calls.append(deck_path)
            if len(calls) == 1:
                msg = "OpenFAST run timed out after 1 s"
                raise OpenFastTimeoutError(msg)
            lin_files = [deck_path.with_suffix(f".{i}.lin") for i in range(1, 4)]
            return OpenFastResult(fst_path=deck_path, returncode=0, lin_files=lin_files)

        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        case = drv.run_case(OperatingPoint(8.0, 9.55, 0.0, name="ws8"))

        assert len(calls) == 2  # timed out once, then retried and converged
        assert case.attempts == 2
        assert case.converged is True
        assert case.result.n_lin == 3

    def test_persistent_timeout_recorded_non_converged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run that times out every attempt is recorded non-converged, not raised."""
        from vane.run import driver as driver_module

        fst = _make_template(tmp_path)
        drv = LinearizationDriver(
            fst_path=fst,
            output_dir=tmp_path / "out",
            settings=LinearizationSettings(rated_wind_speed=11.4, n_lin_times=3),
            retry_policy=LinearizationRetryPolicy(max_attempts=2),
            validate=False,
        )

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            msg = "OpenFAST run timed out after 1 s"
            raise OpenFastTimeoutError(msg)

        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        # Must not raise: the sweep records the timeout rather than aborting.
        case = drv.run_case(OperatingPoint(16.0, 9.55, 12.0, name="ws16"))

        assert case.attempts == 2
        assert case.converged is False
        assert case.result.returncode == driver_module._TIMEOUT_RETURNCODE
        assert not case.succeeded

    def test_forced_linearization_text_is_non_converged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run that writes all .lin but logs a forced linearization is retried."""
        from vane.run import driver as driver_module

        fst = _make_template(tmp_path)
        drv = LinearizationDriver(
            fst_path=fst,
            output_dir=tmp_path / "out",
            settings=LinearizationSettings(rated_wind_speed=11.4, n_lin_times=3),
            retry_policy=LinearizationRetryPolicy(max_attempts=2),
            validate=False,
        )
        calls: list[Path] = []

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            calls.append(deck_path)
            lin_files = [
                deck_path.with_suffix(f".{index}.lin") for index in range(1, 4)
            ]
            # First attempt emits the requested files but logs that it forced the
            # linearization without finding the steady state; the second converges.
            stdout = (
                "[WARNING] Steady state not found before end of simulation. "
                "Forcing linearization."
                if len(calls) == 1
                else "Linearization complete."
            )
            return OpenFastResult(
                fst_path=deck_path,
                returncode=0,
                lin_files=lin_files,
                stdout=stdout,
            )

        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        op = OperatingPoint(8.0, 9.55, 0.0, name="ws8")
        case = drv.run_case(op)

        assert len(calls) == 2
        assert case.attempts == 2
        assert case.converged is True

    def test_exhausted_ladder_records_non_converged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A point that never converges is recorded converged=False, not as success."""
        from vane.run import driver as driver_module

        fst = _make_template(tmp_path)
        drv = LinearizationDriver(
            fst_path=fst,
            output_dir=tmp_path / "out",
            settings=LinearizationSettings(rated_wind_speed=11.4, n_lin_times=3),
            retry_policy=LinearizationRetryPolicy(max_attempts=3),
            validate=False,
        )
        calls: list[Path] = []

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            calls.append(deck_path)
            # Always forces a single .lin (the classic non-convergence outcome).
            return OpenFastResult(
                fst_path=deck_path,
                returncode=0,
                lin_files=[deck_path.with_suffix(".1.lin")],
            )

        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        op = OperatingPoint(8.0, 9.55, 0.0, name="ws8")
        case = drv.run_case(op)

        assert len(calls) == 3  # the full ladder was used
        assert case.attempts == 3
        assert case.converged is False
        assert case.result.n_lin == 1  # the forced result is not silently accepted

    def test_retry_ladder_escalates_then_stops(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each retry loosens TrimTol and raises trim damping, then stops on success."""
        from vane.run import driver as driver_module

        fst = _make_template(tmp_path)
        base = LinearizationSettings(
            rated_wind_speed=11.4, n_lin_times=3, trim_tol=1e-3
        )
        drv = LinearizationDriver(
            fst_path=fst,
            output_dir=tmp_path / "out",
            settings=base,
            retry_policy=LinearizationRetryPolicy(max_attempts=3),
            validate=False,
        )
        tols: list[float] = []
        dampings: list[float] = []

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            # The driver configured the deck with this attempt's settings; recover them
            # from the retry policy by attempt index so the escalation is observable
            # without the synthetic deck needing every trim field.
            attempt = len(tols) + 1
            settings = drv.retry_policy.settings_for_attempt(base, attempt)
            tols.append(settings.trim_tol)
            dampings.append(settings.twr_kdmp)
            lin_count = 1 if attempt < 3 else 3  # converge on the third attempt
            lin_files = [
                deck_path.with_suffix(f".{index}.lin")
                for index in range(1, lin_count + 1)
            ]
            return OpenFastResult(fst_path=deck_path, returncode=0, lin_files=lin_files)

        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        op = OperatingPoint(8.0, 9.55, 0.0, name="ws8")
        case = drv.run_case(op)

        assert len(tols) == 3  # escalated through the ladder, then stopped
        assert tols[0] < tols[1] < tols[2]  # TrimTol loosened each retry
        assert dampings[1] <= dampings[2]  # trim damping raised
        assert case.converged is True
        assert case.attempts == 3

    def test_retry_ladder_escalates_trim_gain(self) -> None:
        """The retry ladder raises the torque gain and lowers the pitch gain."""
        base = LinearizationSettings(
            rated_wind_speed=11.4, rated_generator_torque=40000.0
        )  # torque gain 4000
        policy = LinearizationRetryPolicy(max_attempts=3, trim_gain_factor=4.0)
        a1 = policy.settings_for_attempt(base, 1)
        a2 = policy.settings_for_attempt(base, 2)
        a3 = policy.settings_for_attempt(base, 3)
        assert a1.resolved_trim_gain_torque() == pytest.approx(4000.0)
        assert a2.resolved_trim_gain_torque() == pytest.approx(16000.0)  # 4000 * 4
        assert a3.resolved_trim_gain_torque() == pytest.approx(64000.0)  # 4000 * 16
        # The pitch trim is PI with proportional gain 1000*TrimGain; retries back off.
        assert a2.trim_gain_pitch == pytest.approx(base.trim_gain_pitch * 0.2)
        assert a3.trim_gain_pitch == pytest.approx(base.trim_gain_pitch * 0.04)
        with pytest.raises(ValueError, match="pitch_trim_gain_factor"):
            LinearizationRetryPolicy(pitch_trim_gain_factor=0.0)
        # Escalation pins the gain via the explicit override but PRESERVES
        # rated_generator_torque, which still supplies VS_RtTq for the above-rated
        # constant-torque ServoDyn on every retry attempt.
        assert a2.trim_gain_torque_override == pytest.approx(16000.0)
        assert a2.rated_generator_torque == base.rated_generator_torque

    def test_every_retry_raises_damping_above_the_offshore_base(self) -> None:
        """The first retry already raises damping above an at-floor base.

        A base at the 5000 retry floor (the former offshore preset) must not be left
        at exactly the base damping on attempt 2.
        """
        base = LinearizationSettings.offshore(
            rated_wind_speed=11.4, twr_kdmp=5000.0, bld_kdmp=5000.0
        )
        policy = LinearizationRetryPolicy(max_attempts=3)
        a2 = policy.settings_for_attempt(base, 2)
        a3 = policy.settings_for_attempt(base, 3)
        assert base.twr_kdmp == pytest.approx(5000.0)
        assert a2.twr_kdmp == pytest.approx(10000.0)
        assert a2.bld_kdmp == pytest.approx(10000.0)
        assert a3.twr_kdmp == pytest.approx(20000.0)

    def test_retry_ladder_caps_trim_tol(self, caplog: pytest.LogCaptureFixture) -> None:
        """Escalation from a tuned 1e-2 start stops at MAX_TRIM_TOL, not 1.0."""
        from vane.run.lin_config import MAX_TRIM_TOL

        base = LinearizationSettings(rated_wind_speed=11.4, trim_tol=1e-2)
        policy = LinearizationRetryPolicy(max_attempts=4)
        with caplog.at_level("WARNING"):
            tolerances = [
                policy.settings_for_attempt(base, attempt).trim_tol
                for attempt in (1, 2, 3, 4)
            ]
        assert tolerances == pytest.approx([1e-2, 3e-2, 9e-2, MAX_TRIM_TOL])
        assert "capped" in caplog.text

    def test_tuned_hard_point_is_not_recorded_converged_at_trim_tol_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A timed-out trim at the periodic limit gets more time, not a looser tol."""
        from vane.run import driver as driver_module
        from vane.run.lin_config import MAX_TRIM_TOL

        fst = _make_template(tmp_path)
        drv = LinearizationDriver(
            fst_path=fst,
            output_dir=tmp_path / "out",
            settings=LinearizationSettings(rated_wind_speed=11.4, n_lin_times=3),
            retry_policy=LinearizationRetryPolicy(max_attempts=3),
            trim_tuning=TrimTuningPolicy(),
            validate=False,
        )
        written: list[float] = []

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            written.append(float(FastInputFile.read(deck_path).get("TrimTol")))
            count = 3 if len(written) == 3 else 1
            lin_files = [deck_path.with_suffix(f".{n}.lin") for n in range(count)]
            return OpenFastResult(fst_path=deck_path, returncode=0, lin_files=lin_files)

        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        case = drv.run_case(OperatingPoint(16.0, 12.1, 12.0, name="ws16"))

        assert case.converged is True
        assert written == pytest.approx([1e-2, 1e-2, 1e-2])
        assert max(written) <= MAX_TRIM_TOL
        assert case.settings.trim_tol == pytest.approx(1e-2)
        assert drv.provenance([case]).cases[0].trim_tol == pytest.approx(1e-2)

    def test_explicit_trim_tol_above_cap_is_rejected(self) -> None:
        """A near-vacuous CalcSteady tolerance cannot be configured explicitly."""
        with pytest.raises(ValueError, match="trim_tol must not exceed"):
            LinearizationSettings(rated_wind_speed=11.4, trim_tol=0.5)

    def test_rejects_bad_trim_gain_factor(self) -> None:
        """A non-positive trim-gain escalation factor is rejected."""
        with pytest.raises(ValueError, match="trim_gain_factor must be positive"):
            LinearizationRetryPolicy(trim_gain_factor=0.0)

    def test_retry_policy_rejects_fractional_counts_and_invalid_attempts(self) -> None:
        """Retry indices and bounds remain discrete at the public boundary."""
        with pytest.raises(ValueError, match="max_attempts"):
            LinearizationRetryPolicy(max_attempts=2.5)  # type: ignore[arg-type]
        policy = LinearizationRetryPolicy(max_attempts=2)
        with pytest.raises(ValueError, match="attempt must be an integer"):
            policy.settings_for_attempt(LinearizationSettings(rated_wind_speed=11.4), 0)
        with pytest.raises(ValueError, match="attempt must be an integer"):
            policy.settings_for_attempt(LinearizationSettings(rated_wind_speed=11.4), 3)

    def test_invalid_max_workers_raises(self, tmp_path: Path) -> None:
        """A non-positive worker count is rejected."""
        fst = _make_template(tmp_path)
        drv = LinearizationDriver(fst_path=fst, output_dir=tmp_path / "out")
        with pytest.raises(ValueError, match="max_workers must be >= 1"):
            drv.run_sweep([_standstill("a")], max_workers=0)
        with pytest.raises(ValueError, match="max_workers must be >= 1"):
            drv.run_sweep(
                [_standstill("a")],
                max_workers=1.5,  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="reserve_cpus must be non-negative"):
            default_max_workers(reserve_cpus=1.5)  # type: ignore[arg-type]

    def test_invalid_case_dir_fails_before_launching_any(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A template-overlapping case dir fails the sweep before any case is run."""
        from vane.run import driver as driver_module

        fst = _make_template(tmp_path)  # template_root == tmp_path/template
        drv = LinearizationDriver(fst_path=fst, output_dir=tmp_path)

        launched: list[str] = []

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            launched.append(deck_path.parent.name)
            return OpenFastResult(fst_path=deck_path, returncode=0)

        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        # "template" collides with the template directory; it must be rejected up front
        # (under concurrency) before "ok" is ever generated or launched.
        with pytest.raises(ValueError, match="overlap the template root"):
            drv.run_sweep([_standstill("ok"), _standstill("template")], max_workers=2)
        assert launched == []
        assert not (tmp_path / "ok").exists()


class TestPreflightWiring:
    """The driver runs the deck pre-flight guard before launching OpenFAST."""

    def test_rejects_non_linearizable_deck_before_running(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A DLL-controller deck under default settings is rejected pre-launch."""
        from vane.run import driver as driver_module

        fst = _make_dll_template(tmp_path)
        drv = LinearizationDriver(fst_path=fst, output_dir=tmp_path / "out")
        launched: list[str] = []

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            launched.append(deck_path.parent.name)
            return OpenFastResult(fst_path=deck_path, returncode=0)

        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        with pytest.raises(DeckValidationError, match="DLL controller"):
            drv.run_case(_standstill("dll"))
        assert launched == []  # OpenFAST never launched

    def test_validate_false_skips_the_guard(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With validate=False the pre-flight check is skipped and the run proceeds."""
        from vane.run import driver as driver_module

        fst = _make_dll_template(tmp_path)
        drv = LinearizationDriver(
            fst_path=fst, output_dir=tmp_path / "out", validate=False
        )

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            return OpenFastResult(fst_path=deck_path, returncode=0)

        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        case = drv.run_case(_standstill("dll"))  # must not raise
        assert case.succeeded

    def test_linearizable_deck_passes_the_guard(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A plain (onshore) deck passes pre-flight and runs."""
        from vane.run import driver as driver_module

        fst = _make_template(tmp_path)
        drv = LinearizationDriver(fst_path=fst, output_dir=tmp_path / "out")

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            return OpenFastResult(fst_path=deck_path, returncode=0)

        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        assert drv.run_case(_standstill("ok")).succeeded


class TestDriverIntegration:
    """Real end-to-end sweep (skipped when OpenFAST or r-test data is absent)."""

    def test_closed_loop_model_to_modes(
        self, openfast_exe: Path, rtest_dir: Path, tmp_path: Path
    ) -> None:
        """Drive a real linearization from a turbine model through to modes."""
        import shutil

        from vane.io import read_lin_file
        from vane.io.mbc_transform import mbc3_transform
        from vane.modal.eigensolver import modes_from_mbc

        src = rtest_dir / "glue-codes" / "openfast"
        case = src / "WP_Stationary_Linear" / "WP_Stationary_Linear.fst"
        if not case.is_file():
            pytest.skip(f"r-test case not found: {case}")
        # Assemble a self-contained root: the case plus its sibling WP_Baseline,
        # preserving the ``../WP_Baseline`` reference structure.
        root = tmp_path / "turbine"
        shutil.copytree(src / "WP_Stationary_Linear", root / "WP_Stationary_Linear")
        shutil.copytree(src / "WP_Baseline", root / "WP_Baseline")
        fst = root / "WP_Stationary_Linear" / "WP_Stationary_Linear.fst"

        driver = LinearizationDriver(
            fst_path=fst,
            output_dir=tmp_path / "sweep",
            executable=openfast_exe,
            template_root=root,
        )
        cases = driver.run_sweep([_standstill("standstill")], timeout=300)
        assert len(cases) == 1
        assert cases[0].succeeded
        assert cases[0].result.n_lin >= 1
        solution = modes_from_mbc(
            mbc3_transform([read_lin_file(p) for p in cases[0].lin_files])
        )
        assert solution.n_modes > 0


class TestProvenance:
    """Tests for recording the run's LinearizationSettings in provenance."""

    def _sweep(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[LinearizationDriver, list[SweepCase]]:
        """Run a two-point sweep with a fake OpenFAST producing one .lin each."""
        from vane.run import driver as driver_module

        fst = _make_template(tmp_path)
        drv = LinearizationDriver(
            fst_path=fst,
            output_dir=tmp_path / "out",
            settings=LinearizationSettings(
                rated_wind_speed=11.4, n_lin_times=12, trim_tol=5e-4
            ),
            executable="openfast",
        )

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            lin = deck_path.with_suffix(".1.lin")
            lin.write_text("lin", encoding="utf-8")
            return OpenFastResult(fst_path=deck_path, returncode=0, lin_files=[lin])

        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        cases = drv.run_sweep([_standstill("a"), _standstill("b")])
        return drv, cases

    def test_records_settings_and_cases(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The provenance carries every settings field and a per-case record."""
        drv, cases = self._sweep(tmp_path, monkeypatch)
        moment = datetime(2026, 6, 12, 9, 0, tzinfo=timezone.utc)
        prov = drv.provenance(cases, timestamp=moment)

        assert prov.created_at == "2026-06-12T09:00:00+00:00"
        assert prov.executable == "openfast"
        # Every LinearizationSettings field is captured, including the overrides.
        assert prov.settings["n_lin_times"] == 12
        assert prov.settings["trim_tol"] == 5e-4
        assert set(prov.settings) == {f.name for f in fields(LinearizationSettings())}
        assert [c.name for c in prov.cases] == ["a", "b"]
        assert all(c.succeeded and c.n_lin == 1 for c in prov.cases)
        assert all(c.settle_converged is None for c in prov.cases)

    def test_records_settle_convergence_from_operating_point(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A settled point carries its velocity evidence into run provenance."""
        from vane.run import SettleAttempt, SettleConvergence

        drv, cases = self._sweep(tmp_path, monkeypatch)
        attempt = SettleAttempt(0.004, 0.002, True, True, 3600.0, 1)
        convergence = SettleConvergence(0.004, 0.002, True, True, 3600.0, 1, (attempt,))
        cases[0].operating_point = replace(
            cases[0].operating_point,
            settle_convergence=convergence,
        )
        recorded = drv.provenance(cases).cases[0]
        assert recorded.settle_converged is True
        assert recorded.settle_translational_velocity_rms == pytest.approx(0.004)
        assert recorded.settle_angular_velocity_rms == pytest.approx(0.002)
        assert recorded.settle_time == pytest.approx(3600.0)
        assert recorded.settle_iterations == 1
        assert recorded.settle_attempts == (attempt,)
        assert recorded.settle_curve_status is None

    def test_records_settle_curve_check_from_operating_point(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A curve-seeded settled point records the curve values and the outcome."""
        from vane.run import OperatingPoint, compare_settled_to_curve

        drv, cases = self._sweep(tmp_path, monkeypatch)
        point = cases[0].operating_point
        curve = OperatingPoint(point.wind_speed, 7.0, 3.0)
        settled = OperatingPoint(point.wind_speed, 7.5, 3.1)
        cases[0].operating_point = replace(
            point, settle_curve_check=compare_settled_to_curve(settled, curve)
        )
        path = drv.provenance(cases).write(tmp_path / "provenance.json")

        record = json.loads(path.read_text(encoding="utf-8"))["cases"]
        assert record[0]["settle_curve_rotor_speed_rpm"] == pytest.approx(7.0)
        assert record[0]["settle_curve_pitch_deg"] == pytest.approx(3.0)
        assert record[0]["settle_curve_status"] == "flag"
        assert record[1]["settle_curve_status"] is None

    def test_unmeasured_settle_rms_is_written_as_null(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A settle run lacking one velocity channel group still writes provenance."""
        from vane.run import SettleAttempt, SettleConvergence

        drv, cases = self._sweep(tmp_path, monkeypatch)
        nan = float("nan")
        attempt = SettleAttempt(nan, 0.002, True, True, 3600.0, 1)
        convergence = SettleConvergence(nan, 0.002, True, True, 3600.0, 1, (attempt,))
        cases[0].operating_point = replace(
            cases[0].operating_point,
            settle_convergence=convergence,
        )
        path = drv.provenance(cases).write(tmp_path / "provenance.json")

        record = json.loads(path.read_text(encoding="utf-8"))["cases"][0]
        assert record["settle_translational_velocity_rms"] is None
        assert record["settle_angular_velocity_rms"] == pytest.approx(0.002)
        assert record["settle_attempts"][0]["translational_velocity_rms"] is None
        assert record["settle_attempts"][0]["angular_velocity_rms"] == pytest.approx(
            0.002
        )

    def test_retry_escalated_case_still_records_provenance(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A sweep with one retried, escalated point still records valid provenance."""
        from vane.run import driver as driver_module

        fst = _make_template(tmp_path)
        base = LinearizationSettings(
            rated_wind_speed=11.4, n_lin_times=3, trim_tol=1e-3
        )
        drv = LinearizationDriver(
            fst_path=fst,
            output_dir=tmp_path / "out",
            settings=base,
            retry_policy=LinearizationRetryPolicy(max_attempts=2),
            validate=False,
        )
        # The first point (ws8) needs a retry: its first run emits a single forced
        # .lin, the second the requested three. The second point (ws10) converges first.
        seen: dict[str, int] = {}

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            key = deck_path.parent.name
            seen[key] = seen.get(key, 0) + 1
            forced = key == "ws8" and seen[key] == 1
            count = 1 if forced else 3
            lin_files = [
                deck_path.with_suffix(f".{i}.lin") for i in range(1, count + 1)
            ]
            return OpenFastResult(fst_path=deck_path, returncode=0, lin_files=lin_files)

        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        cases = drv.run_sweep(
            [
                OperatingPoint(8.0, 9.55, 0.0, name="ws8"),
                OperatingPoint(10.0, 9.55, 0.0, name="ws10"),
            ]
        )

        # The escalated case carries a looser trim_tol than the base; ws10 does not.
        ws8, ws10 = cases
        assert ws8.attempts == 2
        assert ws8.settings.trim_tol > base.trim_tol
        assert ws10.attempts == 1
        assert ws10.settings.trim_tol == base.trim_tol

        # Provenance no longer rejects the heterogeneous-by-escalation sweep: it records
        # the campaign base and surfaces the per-case escalation/attempts.
        prov = drv.provenance(cases)
        # The recorded campaign base is the un-escalated trim tolerance.
        assert prov.settings["trim_tol"] == base.trim_tol
        assert prov.settings["n_lin_times"] == 3
        by_name = {c.name: c for c in prov.cases}
        assert by_name["ws8"].attempts == 2
        assert by_name["ws8"].trim_tol > base.trim_tol  # per-case as-run value
        # ws8's trim ran out of time, so its retry extended TMax (recorded per case)
        # and left the trim gain alone; ws10 kept the deck's TMax.
        assert by_name["ws8"].sim_time == pytest.approx(1200.0)
        assert by_name["ws8"].failure_mode is None
        assert by_name["ws8"].attempt_history[0].failure_mode == "trim_not_converged"
        assert by_name["ws10"].sim_time is None
        assert by_name["ws8"].trim_gain_torque == by_name["ws10"].trim_gain_torque
        assert by_name["ws10"].trim_gain_torque == base.resolved_trim_gain_torque()
        assert by_name["ws10"].attempts == 1
        assert by_name["ws10"].trim_tol == base.trim_tol
        assert all(c.converged for c in prov.cases)

    def _retried_sweep(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[LinearizationDriver, list[SweepCase], LinearizationSettings]:
        """Run a sweep where ws8 needs an escalated retry and ws10 converges first."""
        from vane.run import driver as driver_module

        fst = _make_template(tmp_path)
        base = LinearizationSettings(
            rated_wind_speed=11.4, n_lin_times=3, trim_tol=1e-3
        )
        drv = LinearizationDriver(
            fst_path=fst,
            output_dir=tmp_path / "out",
            settings=base,
            retry_policy=LinearizationRetryPolicy(max_attempts=2),
            validate=False,
        )
        seen: dict[str, int] = {}

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            key = deck_path.parent.name
            seen[key] = seen.get(key, 0) + 1
            count = 1 if (key == "ws8" and seen[key] == 1) else 3
            lin_files = [
                deck_path.with_suffix(f".{i}.lin") for i in range(1, count + 1)
            ]
            return OpenFastResult(fst_path=deck_path, returncode=0, lin_files=lin_files)

        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        cases = drv.run_sweep(
            [
                OperatingPoint(8.0, 9.55, 0.0, name="ws8"),
                OperatingPoint(10.0, 9.55, 0.0, name="ws10"),
            ]
        )
        return drv, cases, base

    def test_provenance_base_survives_trim_tol_mutation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutating drv.settings.trim_tol after the sweep must not move the base."""
        drv, cases, base = self._retried_sweep(tmp_path, monkeypatch)
        # Mutate the live driver's escalated field after the cases ran. The recorded
        # base comes from the case snapshots, so this cannot change the provenance.
        drv.settings = replace(drv.settings, trim_tol=base.trim_tol * 50.0)
        prov = drv.provenance(cases)
        assert prov.settings["trim_tol"] == base.trim_tol  # the as-run base, not 0.05
        by_name = {c.name: c for c in prov.cases}
        assert by_name["ws8"].trim_tol > base.trim_tol  # per-case escalation preserved
        assert by_name["ws10"].trim_tol == base.trim_tol

    def test_provenance_still_rejects_campaign_wide_difference(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A genuinely-different campaign-wide setting is still rejected."""
        drv, cases, _base = self._retried_sweep(tmp_path, monkeypatch)
        # Forge a campaign-wide (non-escalated) difference on one case snapshot.
        cases[1].settings = replace(cases[1].settings, n_lin_times=99)
        cases[1].base_settings = replace(cases[1].base_settings, n_lin_times=99)
        drv.settings = replace(
            drv.settings, trim_tol=0.05
        )  # later mutation is irrelevant
        with pytest.raises(ValueError, match="differing campaign-base"):
            drv.provenance(cases)

    def test_writes_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The provenance round-trips through JSON on disk."""
        drv, cases = self._sweep(tmp_path, monkeypatch)
        moment = datetime(2026, 6, 12, 9, 0, tzinfo=timezone.utc)
        path = drv.provenance(cases, timestamp=moment).write(tmp_path / "prov.json")

        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["settings"]["n_lin_times"] == 12
        assert loaded["vane_version"]
        assert len(loaded["cases"]) == 2
        assert loaded["cases"][0]["name"] == "a"

    def test_snapshot_survives_driver_mutation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mutating the driver after the sweep does not falsify the provenance."""
        drv, cases = self._sweep(tmp_path, monkeypatch)
        drv.settings = LinearizationSettings(
            rated_wind_speed=11.4, n_lin_times=99
        )  # reconfigure afterwards
        drv.executable = "other-openfast"
        prov = drv.provenance(cases)
        # The record reflects what produced the cases (12), not the later mutation (99).
        assert prov.settings["n_lin_times"] == 12
        assert prov.executable == "openfast"

    def test_structural_dofs_snapshot_is_independent_of_later_mutation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A case's settings snapshot deep-copies the mutable structural_dofs dict."""
        from vane.run import driver as driver_module

        # The ED deck must carry the DOF flags the settings explicitly set, else the
        # explicit request for an absent flag is (correctly) rejected before launch.
        root = tmp_path / "template"
        root.mkdir()
        (root / "ED.dat").write_text(
            _ED + "True DrTrDOF - flag\nTrue YawDOF - flag\n", encoding="utf-8"
        )
        (root / "IW.dat").write_text(_IW, encoding="utf-8")
        fst = root / "Model.fst"
        fst.write_text(_PRIMARY, encoding="utf-8")
        original = {"YawDOF": False, "DrTrDOF": True}
        drv = LinearizationDriver(
            fst_path=fst,
            output_dir=tmp_path / "out",
            settings=LinearizationSettings(
                rated_wind_speed=11.4, structural_dofs=original
            ),
            validate=False,
        )

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            return OpenFastResult(fst_path=deck_path, returncode=0)

        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        case = drv.run_case(_standstill("a"))

        # Mutate both the live driver's dict and the originally-passed dict afterwards.
        assert drv.settings.structural_dofs is not None
        drv.settings.structural_dofs["DrTrDOF"] = False
        drv.settings.structural_dofs["YawDOF"] = True
        original["DrTrDOF"] = False
        original["new_flag"] = True

        # The case snapshots are unaffected: they hold their own dict objects.
        assert case.settings.structural_dofs == {"YawDOF": False, "DrTrDOF": True}
        assert case.base_settings.structural_dofs == {"YawDOF": False, "DrTrDOF": True}
        assert case.settings.structural_dofs is not drv.settings.structural_dofs
        assert case.settings.structural_dofs is not original

    def test_rejects_cases_with_differing_settings(self, tmp_path: Path) -> None:
        """A heterogeneous set of cases has no single run configuration."""
        fst = _make_template(tmp_path)
        drv = LinearizationDriver(fst_path=fst, output_dir=tmp_path / "out")
        deck = tmp_path / "M.fst"
        op = _standstill("a")
        case_a = SweepCase(
            operating_point=op,
            case_dir=tmp_path / "a",
            fst_path=deck,
            result=OpenFastResult(fst_path=deck, returncode=0),
            settings=LinearizationSettings(rated_wind_speed=11.4, n_lin_times=12),
        )
        case_b = SweepCase(
            operating_point=op,
            case_dir=tmp_path / "b",
            fst_path=deck,
            result=OpenFastResult(fst_path=deck, returncode=0),
            settings=LinearizationSettings(rated_wind_speed=11.4, n_lin_times=24),
        )
        with pytest.raises(
            ValueError, match="differing campaign-wide LinearizationSettings"
        ):
            drv.provenance([case_a, case_b])


def _diagnosis(
    mode: OpenFastFailureMode, failure_time: float | None = None
) -> OpenFastDiagnosis:
    """Return a synthetic diagnosis of the given failure mode."""
    return OpenFastDiagnosis(mode=mode, detail="synthetic", failure_time=failure_time)


class TestAdaptiveRetryPolicy:
    """The retry ladder picks the next attempt's settings from the failure mode."""

    _BASE = LinearizationSettings(
        rated_wind_speed=11.4,
        trim_tol=1e-3,
        twr_kdmp=1500.0,
        bld_kdmp=1500.0,
        trim_gain_torque=100.0,
        trim_gain_pitch=0.001,
    )

    def _next(
        self,
        diagnosis: OpenFastDiagnosis,
        *,
        policy: LinearizationRetryPolicy | None = None,
        previous: LinearizationSettings | None = None,
        previous_tmax: float | None = 600.0,
    ) -> LinearizationSettings | None:
        """Return the policy's settings for attempt 2 after ``diagnosis``."""
        policy = policy if policy is not None else LinearizationRetryPolicy()
        return policy.next_attempt_settings(
            self._BASE,
            previous if previous is not None else self._BASE,
            diagnosis,
            attempt=2,
            previous_tmax=previous_tmax,
        )

    def test_timed_out_trim_is_not_loosened_past_the_periodic_limit(self) -> None:
        """7.1e-3 becomes 1e-2, not 2.1e-2, which left a 6.6 % rotor-speed spread."""
        previous = replace(self._BASE, trim_tol=7.08e-3)
        nxt = self._next(
            _diagnosis(OpenFastFailureMode.TRIM_NOT_CONVERGED), previous=previous
        )
        assert nxt is not None
        assert nxt.trim_tol == pytest.approx(1e-2)
        assert nxt.sim_time == pytest.approx(1200.0)

    def test_a_looser_start_is_not_loosened_further(self) -> None:
        """A tolerance already beyond the periodic limit stays where it is."""
        previous = replace(self._BASE, trim_tol=2e-2)
        nxt = self._next(
            _diagnosis(OpenFastFailureMode.TRIM_NOT_CONVERGED), previous=previous
        )
        assert nxt is not None
        assert nxt.trim_tol == pytest.approx(2e-2)

    def test_default_allows_four_attempts(self) -> None:
        """Failed attempts fail fast, so the default ladder has four rungs."""
        assert LinearizationRetryPolicy().max_attempts == 4

    def test_trim_not_converged_extends_tmax_and_softens_the_pitch_trim(self) -> None:
        """A trim that ran out of time gets more time and a gentler pitch gain."""
        nxt = self._next(_diagnosis(OpenFastFailureMode.TRIM_NOT_CONVERGED))
        assert nxt is not None
        assert nxt.sim_time == pytest.approx(1200.0)
        assert nxt.trim_tol == pytest.approx(self._BASE.trim_tol * 3.0)
        assert nxt.twr_kdmp == self._BASE.twr_kdmp
        assert nxt.bld_kdmp == self._BASE.bld_kdmp
        assert nxt.resolved_trim_gain_torque() == pytest.approx(100.0)
        assert nxt.trim_gain_pitch == pytest.approx(self._BASE.trim_gain_pitch * 0.2)

    def test_tmax_extension_is_capped_and_never_shortens(self) -> None:
        """The extension stops at max_sim_time and keeps a deck already beyond it."""
        policy = LinearizationRetryPolicy(max_sim_time=2000.0)
        mode = _diagnosis(OpenFastFailureMode.TRIM_NOT_CONVERGED)
        from_sim_time = self._next(
            mode, policy=policy, previous=replace(self._BASE, sim_time=1500.0)
        )
        assert from_sim_time is not None
        assert from_sim_time.sim_time == pytest.approx(2000.0)
        long_deck = self._next(mode, policy=policy, previous_tmax=5000.0)
        assert long_deck is not None
        assert long_deck.sim_time is None  # the 5000 s deck TMax is kept
        unknown_tmax = self._next(mode, previous_tmax=None)
        assert unknown_tmax is not None
        assert unknown_tmax.sim_time == pytest.approx(2 * MIN_CALC_STEADY_TMAX)
        no_extension = self._next(
            mode, policy=LinearizationRetryPolicy(tmax_factor=1.0)
        )
        assert no_extension is not None
        assert no_extension.sim_time is None

    def test_late_motoring_backs_off_gains_and_keeps_damping(self) -> None:
        """Motoring well after the start is an overshooting trim: lower the gains."""
        nxt = self._next(_diagnosis(OpenFastFailureMode.MOTORING, failure_time=40.0))
        assert nxt is not None
        assert nxt.trim_gain_pitch == pytest.approx(self._BASE.trim_gain_pitch * 0.2)
        assert nxt.resolved_trim_gain_torque() == pytest.approx(100.0 / 4.0)
        assert nxt.twr_kdmp == pytest.approx(1500.0)
        assert nxt.bld_kdmp == pytest.approx(1500.0)
        assert nxt.trim_tol == pytest.approx(1e-3)
        assert nxt.sim_time is None

    def test_early_motoring_removes_trim_damping(self) -> None:
        """Motoring near t = 0 is damping destabilising light blades: remove it."""
        nxt = self._next(_diagnosis(OpenFastFailureMode.MOTORING, failure_time=0.06))
        assert nxt is not None
        assert nxt.twr_kdmp == 0.0
        assert nxt.bld_kdmp == 0.0
        assert nxt.trim_gain_pitch == pytest.approx(self._BASE.trim_gain_pitch * 0.2)

    @pytest.mark.parametrize(
        "mode",
        [OpenFastFailureMode.ROTOR_OVERSPEED, OpenFastFailureMode.DIVERGENCE],
    )
    def test_divergence_reduces_gains_and_damping(
        self, mode: OpenFastFailureMode
    ) -> None:
        """An overspeed or divergence lowers both the gains and the damping."""
        late = self._next(_diagnosis(mode, failure_time=120.0))
        assert late is not None
        assert late.twr_kdmp == pytest.approx(750.0)
        assert late.bld_kdmp == pytest.approx(750.0)
        assert late.trim_gain_pitch < self._BASE.trim_gain_pitch
        assert late.resolved_trim_gain_torque() < 100.0
        early = self._next(_diagnosis(mode, failure_time=0.1))
        assert early is not None
        assert early.twr_kdmp == 0.0
        unknown_time = self._next(_diagnosis(mode))
        assert unknown_time is not None
        assert unknown_time.twr_kdmp == pytest.approx(750.0)

    @pytest.mark.parametrize(
        "mode",
        [
            OpenFastFailureMode.UNKNOWN,
            OpenFastFailureMode.TIMEOUT,
            OpenFastFailureMode.INCOMPLETE_LINEARIZATION,
        ],
    )
    def test_unclassified_failure_takes_the_generic_step(
        self, mode: OpenFastFailureMode
    ) -> None:
        """Without a known cause the retry is one step of the blind schedule."""
        policy = LinearizationRetryPolicy()
        nxt = self._next(_diagnosis(mode), policy=policy)
        assert nxt == policy.settings_for_attempt(self._BASE, 2)

    def test_generic_steps_compose_like_the_blind_schedule(self) -> None:
        """Repeated generic steps from the previous attempt match the blind ladder."""
        policy = LinearizationRetryPolicy()
        unknown = _diagnosis(OpenFastFailureMode.UNKNOWN)
        second = policy.next_attempt_settings(
            self._BASE, self._BASE, unknown, attempt=2
        )
        assert second is not None
        third = policy.next_attempt_settings(self._BASE, second, unknown, attempt=3)
        assert third is not None
        expected = policy.settings_for_attempt(self._BASE, 3)
        assert third.trim_tol == pytest.approx(expected.trim_tol)
        assert third.twr_kdmp == pytest.approx(expected.twr_kdmp)
        assert third.resolved_trim_gain_torque() == pytest.approx(
            expected.resolved_trim_gain_torque()
        )
        assert third.trim_gain_pitch == pytest.approx(expected.trim_gain_pitch)

    def test_remedies_compose_from_the_previous_attempt(self) -> None:
        """A slow trim after an early motoring keeps the removed damping."""
        policy = LinearizationRetryPolicy()
        second = policy.next_attempt_settings(
            self._BASE,
            self._BASE,
            _diagnosis(OpenFastFailureMode.MOTORING, failure_time=0.05),
            attempt=2,
        )
        assert second is not None
        third = policy.next_attempt_settings(
            self._BASE,
            second,
            _diagnosis(OpenFastFailureMode.TRIM_NOT_CONVERGED),
            attempt=3,
            previous_tmax=600.0,
        )
        assert third is not None
        assert third.twr_kdmp == 0.0
        assert third.trim_gain_pitch == pytest.approx(second.trim_gain_pitch * 0.2)
        assert third.sim_time == pytest.approx(1200.0)

    def test_slow_trim_never_gains_damping(self) -> None:
        """A trim that runs out of time gets more time, never trim damping."""
        policy = LinearizationRetryPolicy()
        base = LinearizationSettings(rated_wind_speed=11.4)
        slow = _diagnosis(OpenFastFailureMode.TRIM_NOT_CONVERGED)
        second = policy.next_attempt_settings(
            base, base, slow, attempt=2, previous_tmax=600.0
        )
        assert second is not None
        third = policy.next_attempt_settings(
            base, second, slow, attempt=3, previous_tmax=1200.0
        )
        assert third is not None
        assert third.twr_kdmp == 0.0
        assert third.bld_kdmp == 0.0
        assert third.sim_time == pytest.approx(2400.0)

    def test_offshore_preset_has_no_trim_damping(self) -> None:
        """Trim damping biases the trimmed point, so offshore() starts undamped."""
        settings = LinearizationSettings.offshore(rated_wind_speed=11.4)
        assert settings.twr_kdmp == 0.0
        assert settings.bld_kdmp == 0.0

    def test_initialization_error_is_never_retried(self) -> None:
        """An input problem returns no retry, adaptive or not."""
        init = _diagnosis(OpenFastFailureMode.INITIALIZATION_ERROR)
        assert self._next(init) is None
        assert self._next(init, policy=LinearizationRetryPolicy(adaptive=False)) is None

    def test_non_adaptive_policy_follows_the_blind_schedule(self) -> None:
        """adaptive=False ignores the failure mode (except initialisation)."""
        policy = LinearizationRetryPolicy(adaptive=False)
        nxt = self._next(
            _diagnosis(OpenFastFailureMode.MOTORING, failure_time=0.06), policy=policy
        )
        assert nxt == policy.settings_for_attempt(self._BASE, 2)

    def test_rejects_invalid_knobs_and_attempts(self) -> None:
        """New knobs are range-checked and a retry is attempt 2 or later."""
        with pytest.raises(ValueError, match="tmax_factor must be >= 1"):
            LinearizationRetryPolicy(tmax_factor=0.5)
        with pytest.raises(ValueError, match="max_sim_time must be positive"):
            LinearizationRetryPolicy(max_sim_time=0.0)
        with pytest.raises(ValueError, match="damping_backoff_factor"):
            LinearizationRetryPolicy(damping_backoff_factor=1.5)
        with pytest.raises(ValueError, match="early_failure_time"):
            LinearizationRetryPolicy(early_failure_time=-1.0)
        with pytest.raises(ValueError, match="early_failure_time"):
            LinearizationRetryPolicy(early_failure_time="soon")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="tmax_factor"):
            LinearizationRetryPolicy(tmax_factor=True)
        with pytest.raises(TypeError, match="adaptive"):
            LinearizationRetryPolicy(adaptive=1)  # type: ignore[arg-type]
        policy = LinearizationRetryPolicy()
        unknown = _diagnosis(OpenFastFailureMode.UNKNOWN)
        with pytest.raises(ValueError, match="at least 2"):
            policy.next_attempt_settings(self._BASE, self._BASE, unknown, attempt=1)
        with pytest.raises(ValueError, match="attempt must be an integer"):
            policy.next_attempt_settings(self._BASE, self._BASE, unknown, attempt=5)


_MOTORING_AT_START = (
    "FAST_Solution:SrvD_UpdateStates:u%HSS_Spd is negative. Simple variable-speed "
    "control model is not valid for motoring situations.\n"
    " OpenFAST encountered an error at simulation time 0.06 of 600 seconds.\n"
)
_FORCED = (
    "[WARNING] Steady state not found before end of simulation. "
    "Forcing linearization.\n"
)


class TestAdaptiveDriver:
    """The driver classifies each attempt, logs it and adapts the next one."""

    def _driver(
        self, tmp_path: Path, *, max_attempts: int = 4, tmax: bool = True
    ) -> LinearizationDriver:
        """Return a driver over a template whose primary deck carries TMax."""
        fst = _make_template(tmp_path)
        if tmax:
            fst.write_text(_PRIMARY + "60 TMax - s\n", encoding="utf-8")
        return LinearizationDriver(
            fst_path=fst,
            output_dir=tmp_path / "out",
            settings=LinearizationSettings(
                rated_wind_speed=11.4, n_lin_times=3, twr_kdmp=1500.0, bld_kdmp=1500.0
            ),
            retry_policy=LinearizationRetryPolicy(max_attempts=max_attempts),
            validate=False,
        )

    def test_logs_every_attempt_and_records_the_failure_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each attempt's output survives in the case dir; modes go to provenance."""
        from vane.run import driver as driver_module

        drv = self._driver(tmp_path)
        outputs = [_MOTORING_AT_START, _FORCED, "OpenFAST terminated normally."]
        tmaxes: list[float] = []

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            index = len(tmaxes)
            tmaxes.append(float(FastInputFile.read(deck_path).get("TMax")))
            count = 3 if index != 1 else 1
            lin_files = [
                deck_path.with_suffix(f".{n}.lin") for n in range(1, count + 1)
            ]
            return OpenFastResult(
                fst_path=deck_path,
                returncode=1 if index == 0 else 0,
                lin_files=lin_files,
                stdout=outputs[index],
                stderr="stderr text" if index == 0 else "",
            )

        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        case = drv.run_case(OperatingPoint(8.0, 9.55, 0.0, name="ws8"))

        assert case.converged is True
        assert case.attempts == 3
        assert [a.failure_mode for a in case.attempt_history] == [
            "motoring",
            "trim_not_converged",
            None,
        ]
        first, second, third = case.attempt_history
        # Early motoring removed the trim damping; the slow trim then got more time.
        assert first.twr_kdmp == pytest.approx(1500.0)
        assert first.failure_time == pytest.approx(0.06)
        assert "HSS_Spd is negative" in first.openfast_message
        assert second.twr_kdmp == 0.0
        assert second.trim_gain_pitch < first.trim_gain_pitch
        assert tmaxes == pytest.approx([600.0, 600.0, 1200.0])
        assert third.sim_time == pytest.approx(1200.0)
        assert third.tmax == pytest.approx(1200.0)
        assert case.settings.sim_time == pytest.approx(1200.0)
        # The full output of every attempt is in the case directory, including the
        # attempts whose files the regenerated case directory would otherwise lose.
        for record, text in zip(case.attempt_history, outputs, strict=True):
            assert record.log_file is not None
            log = Path(record.log_file)
            assert log.parent == case.case_dir
            assert log.name == f"Model.attempt{record.attempt}.log"
            assert text in log.read_text(encoding="utf-8")
        assert "stderr text" in Path(first.log_file or "").read_text(encoding="utf-8")

        prov = drv.provenance([case])
        record = prov.cases[0]
        assert record.failure_mode is None
        assert record.sim_time == pytest.approx(1200.0)
        assert [a.failure_mode for a in record.attempt_history] == [
            "motoring",
            "trim_not_converged",
            None,
        ]
        # The campaign base is unchanged by the per-case TMax extension.
        assert prov.settings["sim_time"] is None
        path = prov.write(tmp_path / "prov.json")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        history = loaded["cases"][0]["attempt_history"]
        assert [a["failure_mode"] for a in history] == [
            "motoring",
            "trim_not_converged",
            None,
        ]
        assert history[0]["log_file"].endswith("Model.attempt1.log")

    def test_initialization_error_is_not_retried(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An input problem fails after one run, with OpenFAST's own message."""
        from vane.run import driver as driver_module

        drv = self._driver(tmp_path)
        calls: list[Path] = []

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            calls.append(deck_path)
            return OpenFastResult(
                fst_path=deck_path,
                returncode=1,
                stdout=(
                    "FAST_InitializeAll:InflowWind file not found.\n"
                    " OpenFAST encountered an error during module initialization.\n"
                ),
            )

        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        with caplog.at_level("ERROR"):
            case = drv.run_case(OperatingPoint(8.0, 9.55, 0.0, name="ws8"))

        assert len(calls) == 1
        assert case.attempts == 1
        assert case.converged is False
        assert case.failure_mode == "initialization_error"
        assert "InflowWind file not found" in caplog.text
        assert "not retried" in caplog.text

    def test_persistent_slow_trim_extends_tmax_each_attempt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A trim that keeps running out of time gets doubling TMax up to the cap."""
        from vane.run import driver as driver_module

        drv = self._driver(tmp_path)
        tmaxes: list[float] = []
        gains: list[str] = []

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            deck = FastInputFile.read(deck_path)
            tmaxes.append(float(deck.get("TMax")))
            gains.append(deck.get("TrimGain"))
            return OpenFastResult(
                fst_path=deck_path,
                returncode=0,
                lin_files=[deck_path.with_suffix(".1.lin")],
                stdout=_FORCED,
            )

        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        case = drv.run_case(OperatingPoint(8.0, 9.55, 0.0, name="ws8"))

        assert case.converged is False
        assert case.failure_mode == "trim_not_converged"
        assert tmaxes == pytest.approx([600.0, 1200.0, 2400.0, 3600.0])
        assert len(set(gains)) == 1  # the trim gain was never escalated

    def test_unreadable_tmax_still_extends_from_the_floor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A deck without TMax extends from the CalcSteady floor."""
        from vane.run import driver as driver_module

        drv = self._driver(tmp_path, max_attempts=2, tmax=False)

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            return OpenFastResult(
                fst_path=deck_path,
                returncode=0,
                lin_files=[deck_path.with_suffix(".1.lin")],
            )

        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        case = drv.run_case(OperatingPoint(8.0, 9.55, 0.0, name="ws8"))

        assert case.attempt_history[0].tmax is None
        assert case.attempt_history[1].sim_time == pytest.approx(
            2 * MIN_CALC_STEADY_TMAX
        )

    def test_unwritable_log_does_not_fail_the_case(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A log that cannot be written is reported and recorded as missing."""
        from vane.run import driver as driver_module

        drv = self._driver(tmp_path, max_attempts=1)

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            # A directory where the log should go makes writing it fail.
            deck_path.with_name("Model.attempt1.log").mkdir()
            lin_files = [deck_path.with_suffix(f".{n}.lin") for n in range(1, 4)]
            return OpenFastResult(fst_path=deck_path, returncode=0, lin_files=lin_files)

        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        with caplog.at_level("WARNING"):
            case = drv.run_case(OperatingPoint(8.0, 9.55, 0.0, name="ws8"))

        assert case.converged is True
        assert case.attempt_history[0].log_file is None
        assert "Could not write the OpenFAST log" in caplog.text


def _coarse_output(settings: LinearizationSettings) -> LinearizationSettings:
    """Stand-in fallback that changes one declared, harmless field."""
    return replace(settings, out_fmt="ES15.8E2")


class TestFinalFallback:
    """A pluggable last-resort settings change for an unstable trim."""

    _BASE = LinearizationSettings(rated_wind_speed=11.4, twr_kdmp=5000.0)
    _MOTORING = _diagnosis(OpenFastFailureMode.MOTORING, failure_time=41.0)

    def _policy(self, **kwargs: object) -> LinearizationRetryPolicy:
        """Return a three-attempt policy with the stand-in fallback."""
        options: dict[str, object] = {
            "max_attempts": 3,
            "final_fallback": _coarse_output,
            "fallback_fields": ("out_fmt",),
        }
        options.update(kwargs)
        return LinearizationRetryPolicy(**options)  # type: ignore[arg-type]

    def test_fallback_runs_only_on_the_last_attempt(self) -> None:
        """Earlier retries use the ladder; the last one applies the fallback to it."""
        policy = self._policy()
        second = policy.next_attempt_settings(
            self._BASE, self._BASE, self._MOTORING, attempt=2
        )
        assert second is not None
        assert second.out_fmt == self._BASE.out_fmt
        assert not policy.uses_fallback(self._MOTORING, attempt=2)
        third = policy.next_attempt_settings(
            self._BASE, second, self._MOTORING, attempt=3
        )
        assert third is not None
        assert policy.uses_fallback(self._MOTORING, attempt=3)
        assert third.out_fmt == "ES15.8E2"
        # The fallback starts from the ladder's own choice for that attempt.
        assert third.trim_gain_pitch == pytest.approx(second.trim_gain_pitch * 0.2)

    def test_fallback_skips_other_failure_modes(self) -> None:
        """A trim that merely ran out of time does not trigger the fallback."""
        policy = self._policy()
        slow = _diagnosis(OpenFastFailureMode.TRIM_NOT_CONVERGED)
        assert not policy.uses_fallback(slow, attempt=3)
        third = policy.next_attempt_settings(self._BASE, self._BASE, slow, attempt=3)
        assert third is not None
        assert third.out_fmt == self._BASE.out_fmt
        custom = self._policy(fallback_modes=frozenset({"trim_not_converged"}))
        assert custom.uses_fallback(slow, attempt=3)
        # The default falls back to above-rated torque trim; None disables it.
        default = LinearizationRetryPolicy()
        assert default.uses_fallback(self._MOTORING, attempt=4)
        fallback = default.next_attempt_settings(
            self._BASE, self._BASE, self._MOTORING, attempt=4
        )
        assert fallback is not None
        assert fallback.above_rated_trim == "torque"
        assert not LinearizationRetryPolicy(final_fallback=None).uses_fallback(
            self._MOTORING, attempt=4
        )

    def test_fallback_must_declare_what_it_changes(self) -> None:
        """An undeclared change or a wrong return type is rejected."""
        undeclared = self._policy(fallback_fields=())
        with pytest.raises(ValueError, match="out_fmt"):
            undeclared.next_attempt_settings(
                self._BASE, self._BASE, self._MOTORING, attempt=3
            )
        wrong_type = self._policy(final_fallback=lambda settings: {"x": 1})
        with pytest.raises(TypeError, match="must return LinearizationSettings"):
            wrong_type.next_attempt_settings(
                self._BASE, self._BASE, self._MOTORING, attempt=3
            )

    def test_rejects_invalid_fallback_configuration(self) -> None:
        """The fallback must be callable and its fields real settings fields."""
        with pytest.raises(TypeError, match="final_fallback must be callable"):
            self._policy(final_fallback="torque")
        with pytest.raises(ValueError, match="not LinearizationSettings fields"):
            self._policy(fallback_fields=("trim_mode_above_rated",))
        with pytest.raises(ValueError, match="is not a valid OpenFastFailureMode"):
            self._policy(fallback_modes=frozenset({"exploded"}))

    def test_driver_records_the_fallback_attempt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fallback attempt is flagged and its fields recorded per case."""
        from vane.run import driver as driver_module

        fst = _make_template(tmp_path)
        drv = LinearizationDriver(
            fst_path=fst,
            output_dir=tmp_path / "out",
            settings=LinearizationSettings(rated_wind_speed=11.4, n_lin_times=3),
            retry_policy=self._policy(max_attempts=2),
            validate=False,
        )
        seen: dict[str, int] = {}

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            key = deck_path.parent.name
            seen[key] = seen.get(key, 0) + 1
            if key == "ws16" and seen[key] == 1:
                return OpenFastResult(
                    fst_path=deck_path, returncode=1, stdout=_MOTORING_AT_START
                )
            lin_files = [deck_path.with_suffix(f".{n}.lin") for n in range(1, 4)]
            return OpenFastResult(fst_path=deck_path, returncode=0, lin_files=lin_files)

        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        cases = drv.run_sweep(
            [
                OperatingPoint(16.0, 12.1, 12.0, name="ws16"),
                OperatingPoint(18.0, 12.1, 15.0, name="ws18"),
            ],
            max_workers=1,
        )

        ws16, ws18 = cases
        assert ws18.attempts == 1
        assert ws16.converged is True
        assert [a.fallback for a in ws16.attempt_history] == [False, True]
        assert ws16.settings.out_fmt == "ES15.8E2"
        prov = drv.provenance(cases)
        by_name = {record.name: record for record in prov.cases}
        assert by_name["ws16"].fallback_overrides == {"out_fmt": "ES15.8E2"}
        assert by_name["ws18"].fallback_overrides is None
        assert prov.settings["out_fmt"] == LinearizationSettings().out_fmt
        loaded = json.loads(prov.write(tmp_path / "prov.json").read_text("utf-8"))
        assert loaded["cases"][0]["attempt_history"][1]["fallback"] is True

        # Provenance reads each case with the policy that produced it: replacing the
        # driver's policy afterwards must not change the record.
        assert ws16.fallback_fields == ("out_fmt",)
        drv.retry_policy = LinearizationRetryPolicy(final_fallback=None)
        later = drv.provenance(cases)
        later_by_name = {record.name: record for record in later.cases}
        assert later_by_name["ws16"].fallback_overrides == {"out_fmt": "ES15.8E2"}
        assert later.settings["out_fmt"] == LinearizationSettings().out_fmt


_TIP_UA_STATE = "AD x3 blade 1, node 59, -"


def _lin_text(*, tip_value: float) -> str:
    """Return a two-state ``.lin`` whose tip-node UA state has ``tip_value``."""
    import numpy as np

    from tests.conftest import build_lin_text

    return build_lin_text(
        states=[
            ("ED 1st flapwise bending-mode DOF of blade 1, m", True, 2),
            (_TIP_UA_STATE, True, 1),
        ],
        state_values=[0.5, tip_value],
        a=np.eye(2),
    )


class TestNonFiniteLinearization:
    """A converged run whose ``.lin`` files hold NaN is not accepted."""

    def _driver(self, tmp_path: Path) -> LinearizationDriver:
        fst = _make_template(tmp_path)
        return LinearizationDriver(
            fst_path=fst,
            output_dir=tmp_path / "out",
            settings=LinearizationSettings(rated_wind_speed=11.4, n_lin_times=3),
            retry_policy=LinearizationRetryPolicy(max_attempts=4),
            validate=False,
        )

    def _fake_run(
        self, calls: list[Path], bad_files: int
    ) -> Callable[..., OpenFastResult]:
        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            calls.append(deck_path)
            lin_files = []
            for index in range(1, 4):
                lin = deck_path.with_suffix(f".{index}.lin")
                tip = float("nan") if index > 3 - bad_files else 0.1
                lin.write_text(_lin_text(tip_value=tip), encoding="utf-8")
                lin_files.append(lin)
            return OpenFastResult(
                fst_path=deck_path,
                returncode=0,
                lin_files=lin_files,
                stdout="OpenFAST terminated normally.",
            )

        return fake_run

    def test_nan_lin_is_recorded_not_converged_and_not_retried(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Every requested file written is not enough: NaN content fails the case."""
        from vane.run import driver as driver_module

        drv = self._driver(tmp_path)
        calls: list[Path] = []
        monkeypatch.setattr(driver_module, "run_openfast", self._fake_run(calls, 2))
        with caplog.at_level("WARNING"):
            case = drv.run_case(OperatingPoint(8.0, 9.55, 0.0, name="ws8"))

        assert len(calls) == 1  # a modelling degeneracy: no trim retry
        assert case.n_lin == 3
        assert case.converged is False
        assert case.failure_mode == "non_finite_linearization"
        record = case.attempt_history[0]
        assert record.reason is not None
        assert "2 of 3 full-system .lin files hold non-finite values" in record.reason
        assert _TIP_UA_STATE in record.reason
        assert "UAEndRad below 1" in record.openfast_message
        assert "Model.2.lin" in record.openfast_message
        assert "not retried" in caplog.text
        assert "UAEndRad below 1" in caplog.text

        prov = drv.provenance([case])
        assert prov.cases[0].converged is False
        assert prov.cases[0].failure_mode == "non_finite_linearization"
        loaded = json.loads(prov.write(tmp_path / "prov.json").read_text("utf-8"))
        history = loaded["cases"][0]["attempt_history"]
        assert history[0]["failure_mode"] == "non_finite_linearization"
        log = Path(record.log_file or "").read_text(encoding="utf-8")
        assert "non_finite_linearization" in log

    def test_finite_lin_files_are_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same run with finite files converges on the first attempt."""
        from vane.run import driver as driver_module

        drv = self._driver(tmp_path)
        calls: list[Path] = []
        monkeypatch.setattr(driver_module, "run_openfast", self._fake_run(calls, 0))
        case = drv.run_case(OperatingPoint(8.0, 9.55, 0.0, name="ws8"))
        assert case.converged is True
        assert case.failure_mode is None
        assert len(calls) == 1

    def test_generic_hint_for_other_variables(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-finite structural value gets the generic modelling hint."""
        import numpy as np

        from tests.conftest import build_lin_text
        from vane.run import driver as driver_module

        drv = self._driver(tmp_path)

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            lin_files = []
            for index in range(1, 4):
                lin = deck_path.with_suffix(f".{index}.lin")
                lin.write_text(
                    build_lin_text(
                        states=[("ED tower fore-aft DOF, m", False, 2)],
                        state_values=[float("inf")],
                        a=np.eye(1),
                    ),
                    encoding="utf-8",
                )
                lin_files.append(lin)
            return OpenFastResult(fst_path=deck_path, returncode=0, lin_files=lin_files)

        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        case = drv.run_case(OperatingPoint(8.0, 9.55, 0.0, name="ws8"))
        message = case.attempt_history[0].openfast_message
        assert case.failure_mode == "non_finite_linearization"
        assert "ED tower fore-aft DOF, m" in message
        assert "modelling degeneracy" in message
        assert "UAEndRad" not in message

    def test_diagnosis_is_not_retriable(self) -> None:
        """The retry ladder gives up on a non-finite linearization."""
        diagnosis = OpenFastDiagnosis(
            mode=OpenFastFailureMode.NON_FINITE_LINEARIZATION, detail="NaN"
        )
        assert diagnosis.retriable is False
        policy = LinearizationRetryPolicy(max_attempts=3)
        base = LinearizationSettings(rated_wind_speed=11.4)
        assert policy.next_attempt_settings(base, base, diagnosis, attempt=2) is None


def _ok(deck_path: Path) -> OpenFastResult:
    """Return a successful fake run that wrote its one requested ``.lin`` file."""
    lin = deck_path.with_suffix(".1.lin")
    lin.write_text("lin", encoding="utf-8")
    return OpenFastResult(fst_path=deck_path, returncode=0, lin_files=[lin])


def _denied() -> PermissionError:
    """Return the transient Windows access-denied error a locked deck raises."""
    error = PermissionError(13, "Access is denied")
    error.winerror = 5  # type: ignore[misc]
    return error


class TestCaseErrorRecording:
    """One case's unexpected exception never discards the rest of the sweep."""

    def _driver(self, tmp_path: Path) -> LinearizationDriver:
        return LinearizationDriver(
            fst_path=_make_template(tmp_path), output_dir=tmp_path / "out"
        )

    @pytest.mark.parametrize("max_workers", [1, 3])
    def test_raising_case_is_recorded_and_others_complete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, max_workers: int
    ) -> None:
        """A case whose run raises is a failed case_error; the others succeed."""
        from vane.run import driver as driver_module

        drv = self._driver(tmp_path)

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            if deck_path.parent.name == "b":
                raise _denied()
            return _ok(deck_path)

        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        cases = drv.run_sweep(
            [_standstill("a"), _standstill("b"), _standstill("c")],
            max_workers=max_workers,
        )

        assert [case.operating_point.name for case in cases] == ["a", "b", "c"]
        assert cases[0].converged and cases[2].converged
        failed = cases[1]
        assert failed.converged is False
        assert failed.succeeded is False
        assert failed.failure_mode == "case_error"
        assert failed.failure_mode == OpenFastFailureMode.CASE_ERROR
        assert failed.result.returncode == -1
        assert failed.n_lin == 0
        assert failed.attempts == 1
        record = failed.attempt_history[-1]
        assert record.reason is not None
        assert "PermissionError" in record.reason
        assert "Access is denied" in record.reason
        assert record.log_file is not None
        log = Path(record.log_file).read_text(encoding="utf-8")
        assert "Traceback" in log and "PermissionError" in log
        assert "PermissionError" in failed.result.stderr

        provenance = drv.provenance(cases)
        assert provenance.cases[1].failure_mode == "case_error"
        assert provenance.cases[1].converged is False
        json.dumps(provenance.to_dict())  # still strict-JSON serializable

    def test_error_on_a_retry_keeps_the_earlier_attempts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An exception regenerating attempt 2 keeps attempt 1's record."""
        from vane.run import driver as driver_module

        drv = self._driver(tmp_path)
        original = LinearizationDriver._generate_case_with_settings
        generated: list[str] = []

        def flaky_generate(
            self: LinearizationDriver,
            operating_point: OperatingPoint,
            settings: LinearizationSettings,
        ) -> Path:
            generated.append(str(operating_point.name))
            if generated.count(str(operating_point.name)) == 2:
                raise _denied()
            return original(self, operating_point, settings)

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            msg = "timed out"
            raise OpenFastTimeoutError(msg)

        monkeypatch.setattr(
            LinearizationDriver, "_generate_case_with_settings", flaky_generate
        )
        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        monkeypatch.setattr("vane.utils.fs_retry._pause", lambda _: None)
        (case,) = drv.run_sweep([_standstill("a")], max_workers=1)

        assert case.converged is False
        assert case.attempts == 2
        assert [record.failure_mode for record in case.attempt_history] == [
            "timeout",
            "case_error",
        ]
        # Attempt 1's log survives (the failed regeneration did not wipe the case).
        first_log = case.attempt_history[0].log_file
        assert first_log is not None
        assert Path(first_log).is_file()

    def test_error_before_the_case_dir_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A copy that fails before the case directory exists records no log."""
        drv = self._driver(tmp_path)

        def broken_copy(
            self: LinearizationDriver, source: Path, destination: Path
        ) -> None:
            msg = "disk full"
            raise OSError(msg)

        monkeypatch.setattr(LinearizationDriver, "_copy_template", broken_copy)
        (case,) = drv.run_sweep([_standstill("a")], max_workers=1)
        assert case.failure_mode == "case_error"
        assert case.attempt_history[-1].log_file is None
        assert case.attempt_history[-1].reason == "case_error: OSError: disk full"
        assert case.fst_path == tmp_path / "out" / "a" / "Model.fst"

    def test_strict_raises_after_all_cases_with_them_attached(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """strict=True re-raises only after every case finished, cases attached."""
        from vane.run import SweepCaseError
        from vane.run import driver as driver_module

        drv = self._driver(tmp_path)
        ran: list[str] = []

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            ran.append(deck_path.parent.name)
            if deck_path.parent.name == "a":
                msg = "boom"
                raise RuntimeError(msg)
            return _ok(deck_path)

        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        with pytest.raises(SweepCaseError, match="boom") as excinfo:
            drv.run_sweep([_standstill("a"), _standstill("b")], strict=True)

        assert sorted(ran) == ["a", "b"]  # the case after the error still ran
        error = excinfo.value
        assert isinstance(error.__cause__, RuntimeError)
        assert [case.operating_point.name for case in error.cases] == ["a", "b"]
        assert error.cases[0].failure_mode == "case_error"
        assert error.cases[1].converged
        assert error.result is None

    def test_run_case_alone_still_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Called directly, run_case propagates an unexpected exception."""
        from vane.run import driver as driver_module

        drv = self._driver(tmp_path)

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            msg = "boom"
            raise RuntimeError(msg)

        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        with pytest.raises(RuntimeError, match="boom"):
            drv.run_case(_standstill("a"))

    def test_deck_validation_error_still_raises_from_a_sweep(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A deck the pre-flight guard rejects is an input error, not a case error."""
        from vane.run import driver as driver_module

        drv = LinearizationDriver(
            fst_path=_make_dll_template(tmp_path), output_dir=tmp_path / "out"
        )
        launched: list[str] = []

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            launched.append(deck_path.parent.name)
            return _ok(deck_path)

        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        with pytest.raises(DeckValidationError, match="DLL controller"):
            drv.run_sweep([_standstill("a")], max_workers=1)
        assert launched == []

    def test_keyboard_interrupt_propagates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An interrupt is never swallowed into a case record."""
        from vane.run import driver as driver_module

        drv = self._driver(tmp_path)

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            raise KeyboardInterrupt

        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        with pytest.raises(KeyboardInterrupt):
            drv.run_sweep([_standstill("a")], max_workers=1)

    def test_transient_deck_write_error_is_retried_not_recorded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A briefly denied deck replace is retried and the case converges."""
        from pathlib import Path as RealPath

        from vane.run import driver as driver_module

        drv = self._driver(tmp_path)
        real_replace = RealPath.replace
        denied: list[str] = []

        # Patch Path.replace, not os.replace: before Python 3.11 pathlib does not
        # route through os.replace, so an os-level patch would never fire there.
        def flaky_replace(self: RealPath, target: str | RealPath) -> RealPath:
            if str(target).endswith("ED.dat") and len(denied) < 2:
                denied.append(str(target))
                raise _denied()
            return real_replace(self, target)

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            return _ok(deck_path)

        monkeypatch.setattr(RealPath, "replace", flaky_replace)
        monkeypatch.setattr("vane.utils.fs_retry._pause", lambda _: None)
        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        (case,) = drv.run_sweep([_standstill("a")], max_workers=1)
        assert len(denied) == 2
        assert case.converged
        assert case.failure_mode is None

    def test_template_copy_retries_a_denied_copy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The referenced-file copy retries a briefly denied file copy."""
        import shutil

        from vane.run import driver as driver_module

        drv = LinearizationDriver(
            fst_path=_make_template(tmp_path),
            output_dir=tmp_path / "out",
            link_inputs=False,
        )
        real_copy2 = shutil.copy2
        denied: list[str] = []

        def flaky_copy2(source: str, target: str) -> object:
            if len(denied) < 1:
                denied.append(str(source))
                raise _denied()
            return real_copy2(source, target)

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            return _ok(deck_path)

        monkeypatch.setattr(shutil, "copy2", flaky_copy2)
        monkeypatch.setattr("vane.utils.fs_retry._pause", lambda _: None)
        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        (case,) = drv.run_sweep([_standstill("a")], max_workers=1)
        assert denied
        assert case.converged


class TestNonPeriodicSweep:
    """A CalcSteady sweep whose rotor speed drifts beyond MBC3's limit is retried."""

    @staticmethod
    def _lin(rotor_speed: float) -> str:
        import numpy as np

        from tests.conftest import build_lin_text

        return build_lin_text(
            states=[("ED 1st flapwise bending-mode DOF of blade 1, m", True, 2)],
            state_values=[0.5],
            a=np.eye(1),
            rotor_speed=rotor_speed,
        )

    def _fake_run(
        self, calls: list[Path], spreads: list[float]
    ) -> Callable[..., OpenFastResult]:
        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            spread = spreads[len(calls)]
            calls.append(deck_path)
            lin_files = []
            for index in range(1, 4):
                lin = deck_path.with_suffix(f".{index}.lin")
                speed = 0.8 * (1.0 + spread * (index - 1) / 2.0)
                lin.write_text(self._lin(speed), encoding="utf-8")
                lin_files.append(lin)
            return OpenFastResult(
                fst_path=deck_path,
                returncode=0,
                lin_files=lin_files,
                stdout="OpenFAST terminated normally.",
            )

        return fake_run

    def test_drifting_sweep_is_retried_with_tighter_tolerance_and_more_time(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 2.8 % rotor-speed spread fails the attempt; the retry tightens TrimTol."""
        from vane.run import driver as driver_module

        fst = _make_template(tmp_path)
        base = LinearizationSettings(
            rated_wind_speed=11.4, n_lin_times=3, trim_tol=1e-3
        )
        drv = LinearizationDriver(
            fst_path=fst,
            output_dir=tmp_path / "out",
            settings=base,
            retry_policy=LinearizationRetryPolicy(max_attempts=4),
            validate=False,
        )
        calls: list[Path] = []
        monkeypatch.setattr(
            driver_module, "run_openfast", self._fake_run(calls, [0.028, 0.002])
        )
        case = drv.run_case(OperatingPoint(8.0, 9.55, 0.0, name="ws8"))

        assert len(calls) == 2
        assert case.converged is True
        first = case.attempt_history[0]
        assert first.failure_mode == "non_periodic_sweep"
        assert first.reason is not None
        assert "2.76 %" in first.reason  # spread relative to the mean speed
        retry = case.attempt_history[1]
        assert retry.trim_tol == pytest.approx(base.trim_tol)
        assert retry.sim_time is not None
        assert retry.sim_time > 600.0

    def test_periodic_sweep_is_accepted(self, tmp_path: Path) -> None:
        """A spread inside MBC3's 1 % limit raises no diagnosis."""
        from vane.run.driver import _rotor_speed_spread_diagnosis

        files = []
        for index, speed in enumerate((0.800, 0.802, 0.804), start=1):
            lin = tmp_path / f"Model.{index}.lin"
            lin.write_text(self._lin(speed), encoding="utf-8")
            files.append(lin)
        result = OpenFastResult(
            fst_path=tmp_path / "Model.fst", returncode=0, lin_files=files
        )
        assert _rotor_speed_spread_diagnosis(result) is None


def test_provenance_maps_every_non_finite_settle_value_to_none() -> None:
    """A diverged (infinite) settle RMS, like an unmeasured (NaN) one, is null."""
    import math

    from vane.run.driver import _unmeasured_as_none

    assert _unmeasured_as_none(math.inf) is None
    assert _unmeasured_as_none(-math.inf) is None
    assert _unmeasured_as_none(math.nan) is None
    assert _unmeasured_as_none(0.25) == 0.25
