# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for the automatic configuration of run_operating_sweep (no arguments)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tests.test_run.test_radiation_fit import _write_wamit
from vane.io.fst_writer import FastInputError, FastInputFile
from vane.run import (
    DeckValidationError,
    LinearizationSettings,
    OpenFastError,
    run_operating_sweep,
)
from vane.run import auto_sweep as auto_module
from vane.run import driver as driver_module
from vane.run import executables as executables_module
from vane.run import radiation_fit as radiation_module
from vane.run.executables import (
    OPENFAST_CONFIG_ENV,
    OPENFAST_EXECUTABLES_ENV,
    DeckGeneration,
    OpenFastSelection,
)
from vane.run.runner import OpenFastResult

_PRIMARY = """\
600 TMax - run time
0.01 DT - time step
1 CompElast - structure
1 CompInflow - inflow
2 CompAero - aero
1 CompServo - servo
0 CompSeaSt - sea state
{hydro} CompHydro - hydro
False Linearize - flag
False CalcSteady - flag
3 TrimCase - case
0.01 TrimGain - gain
0.001 TrimTol - tolerance
1 NLinTimes - count
0 LinTimes - times
1.225 AirDens - air density
"ED.dat" EDFile - ElastoDyn
"../Shared/IW.dat" InflowFile - InflowWind
"AD.dat" AeroFile - AeroDyn
"SD.dat" ServoFile - ServoDyn
"HD.dat" HydroFile - HydroDyn
"""

_ED = """\
9.55 RotSpeed - rpm
0.0 BlPitch(1) - deg
0.0 BlPitch(2) - deg
0.0 BlPitch(3) - deg
True GenDOF - generator DOF
True DrTrDOF - drivetrain DOF
True YawDOF - yaw DOF
121.0 TipRad - rotor radius
1 GBRatio - gearbox ratio
100 GBoxEff - gearbox efficiency
0 HubIner_Teeter - teeter inertia
"""

_SERVO = """\
5 PCMode - pitch control
0 TPCOn - pitch on
5 VSContrl - torque control
1 GenModel - generator
True GenTiStr - timed start
True GenTiStp - timed stop
0 TimGenOn - start
9999 TimGenOf - stop
9999.9 VS_RtGnSp - rated speed
9999.9 VS_RtTq - placeholder torque
9999.9 VS_Rgn2K - region-2 gain
9999.9 VS_SlPc - slip
0 HSSBrMode - brake
0 YCMode - yaw
0 PitNeut(1) - neutral
"DISCON.IN" DLL_InFile - controller input
"""

_DISCON = """\
! DISCON-style controller input
15000000.0   ! VS_RtPwr   - Rated power [W]
19786800.0   ! VS_RtTq    - Rated torque [Nm]
0.78788      ! VS_RefSpd  - Rated generator speed [rad/s]
95.756       ! VS_GenEff  - Generator efficiency [%]
"""

_AERO = """\
1 Wake_Mod - BEMT
True TipLoss - tip loss
2 DBEMT_Mod - dynamic wake
3 UA_Mod - unsteady aero
1 UAEndRad - end radius
False NacelleDrag - nacelle drag
"""

_IW = """\
1 WindType - steady
8 HWindSpeed - wind
0.2 PLExp - shear
"""

_HYDRO = """\
1 PotMod - WAMIT
"Hydro/plat" PotFile - root
1 RdtnMod - convolution
2 ExctnMod - state space
0 NAddDOF - DOFs
"""

# An IEA 15-MW-like schedule: rotor speed and power level off at 10.66 m/s.
_CURVE = """\
wind_speed [m/s],rotor_speed_rpm [rpm],blade_pitch_deg [deg],cp,ct
4.0,5.0,3.0,0.40,0.8
8.0,5.7,0.0,0.48,0.8
10.0,7.13,0.0,0.48,0.75
10.66,7.5176,0.0,0.48,0.7
14.0,7.5176,10.0,{cp14},0.3
20.0,7.5176,17.7,{cp20},0.1
"""


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _make_suite(tmp_path: Path, *, hydro: bool = False) -> tuple[Path, Path]:
    """Write a deck that reads ../Shared/IW.dat; return (deck, curve CSV)."""
    suite = tmp_path / "suite"
    case = suite / "Case"
    fst = _write(case / "Model.fst", _PRIMARY.format(hydro=1 if hydro else 0))
    _write(case / "ED.dat", _ED)
    _write(case / "SD.dat", _SERVO)
    _write(case / "DISCON.IN", _DISCON)
    _write(case / "AD.dat", _AERO)
    if hydro:
        _write(case / "HD.dat", _HYDRO)
        _write(case / "Hydro" / "plat.1", "wamit\n")
        _write(case / "Hydro" / "plat.ssexctn", "state space\n")
    _write(suite / "Shared" / "IW.dat", _IW)
    level = 0.48 * 10.66**3
    curve = _write(
        tmp_path / "curve.csv",
        _CURVE.format(cp14=level / 14.0**3, cp20=level / 20.0**3),
    )
    return fst, curve


@pytest.fixture
def launches(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record OpenFAST launches (none really happen) and stub the executable."""
    calls: list[Path] = []

    def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
        calls.append(deck_path)
        return OpenFastResult(
            fst_path=deck_path,
            returncode=0,
            lin_files=[deck_path.with_suffix(f".{i}.lin") for i in range(1, 37)],
        )

    def select(fst_path: object, executable: object = None) -> OpenFastSelection:
        return OpenFastSelection(
            executable=Path("openfast-test"),
            version=(4, 2, 1),
            generation=DeckGeneration("v4.2", (4, 2, 0), (5, 0, 0), ("test",)),
            source="test",
        )

    monkeypatch.setattr(driver_module, "run_openfast", fake_run)
    monkeypatch.setattr(auto_module, "select_openfast_executable", select)
    return calls


def _case_value(case_dir: Path, file_name: str, keyword: str) -> str:
    matches = list(case_dir.rglob(file_name))
    assert len(matches) == 1, matches
    return FastInputFile.read(matches[0]).get(keyword)


class TestNoArgumentSweep:
    """run_operating_sweep(fst, out, winds, performance_csv=curve) is complete."""

    def test_defaults_configure_every_case(
        self, tmp_path: Path, launches: list[Path]
    ) -> None:
        """Root, rated wind/torque, trim gains and modules come from deck and curve."""
        fst, curve = _make_suite(tmp_path)
        out = tmp_path / "out"
        result = run_operating_sweep(
            fst, out, [8.0, 14.0], performance_csv=curve, max_workers=1
        )
        assert len(launches) == 2
        assert all(case.converged for case in result.cases)
        # The deck reads ../Shared, so each case carries Shared next to Case.
        below, above = (out / "ws8", out / "ws14")
        assert (below / "Shared" / "IW.dat").is_file()
        assert (below / "Case" / "Model.fst").is_file()
        # Below rated: torque trim with a tenth of the controller's rated torque.
        assert _case_value(below, "Model.fst", "TrimCase") == "2"
        assert float(_case_value(below, "Model.fst", "TrimGain")) == pytest.approx(
            1978680.0
        )
        # Above rated: pitch trim, gain scaled for R = 121 m, torque from DISCON.IN.
        assert _case_value(above, "Model.fst", "TrimCase") == "3"
        assert float(_case_value(above, "Model.fst", "TrimGain")) == pytest.approx(
            2e-4 * (63.0 / 121.0) ** 3
        )
        # Trim tolerance scaled for the rotor size, on both sides of rated.
        for case in (below, above):
            assert float(_case_value(case, "Model.fst", "TrimTol")) == pytest.approx(
                1e-3 * (121.0 / 63.0) ** 3
            )
        assert float(_case_value(above, "SD.dat", "VS_RtTq")) == pytest.approx(
            19786800.0
        )
        # Modules: quasi-steady aero (UA_Mod 3 is not linearizable), frozen wake.
        assert _case_value(above, "AD.dat", "UA_Mod") == "0"
        assert _case_value(above, "AD.dat", "DBEMT_Mod") == "-1"

        auto = result.auto_configuration
        assert auto is not None
        names = {item.name for item in auto.inferred}
        assert {
            "template_root",
            "rated_wind_speed",
            "rated_generator_torque",
            "trim_gain_pitch",
            "trim_tol",
            "executable",
        } <= names
        rated = auto.get("rated_wind_speed")
        assert rated is not None
        assert rated.value == pytest.approx(10.66)
        root = auto.get("template_root")
        assert root is not None
        assert Path(str(root.value)) == (tmp_path / "suite").resolve()

        written = json.loads(
            (out / auto_module.AUTO_CONFIGURATION_FILE).read_text(encoding="utf-8")
        )
        assert written == auto.to_dict()
        driver = driver_module.LinearizationDriver(
            fst_path=fst,
            output_dir=out,
            template_root=tmp_path / "suite",
            auto_configuration=auto.to_dict(),
        )
        provenance = driver.provenance(result.cases).to_dict()
        assert provenance["auto_configuration"] == auto.to_dict()
        assert provenance["executable"] == "openfast-test"

    def test_explicit_template_root_wins(
        self, tmp_path: Path, launches: list[Path]
    ) -> None:
        """An explicit root is used as given and not recorded as inferred."""
        fst, curve = _make_suite(tmp_path)
        result = run_operating_sweep(
            fst,
            tmp_path / "out",
            [8.0],
            performance_csv=curve,
            template_root=tmp_path / "suite",
            max_workers=1,
        )
        assert result.auto_configuration is not None
        assert result.auto_configuration.get("template_root") is None
        assert (tmp_path / "out" / "ws8" / "Shared" / "IW.dat").is_file()

    def test_output_inside_the_inferred_root_is_rejected(
        self, tmp_path: Path, launches: list[Path]
    ) -> None:
        """An output_dir under the inferred root fails before anything is copied."""
        fst, curve = _make_suite(tmp_path)
        out = tmp_path / "suite" / "runs"
        with pytest.raises(ValueError, match="choose an output_dir outside it"):
            run_operating_sweep(fst, out, [8.0], performance_csv=curve)
        assert launches == []
        assert not out.exists()

    def test_missing_state_space_radiation_falls_back(
        self, tmp_path: Path, launches: list[Path]
    ) -> None:
        """No .ss file: RdtnMod 0 with a recorded warning; ExctnMod 2 kept."""
        fst, curve = _make_suite(tmp_path, hydro=True)
        out = tmp_path / "out"
        result = run_operating_sweep(
            fst, out, [8.0], performance_csv=curve, max_workers=1
        )
        assert _case_value(out / "ws8", "HD.dat", "RdtnMod") == "0"
        assert _case_value(out / "ws8", "HD.dat", "ExctnMod") == "2"
        auto = result.auto_configuration
        assert auto is not None
        assert any("RdtnMod=1" in warning for warning in auto.warnings)

    def test_missing_state_space_radiation_is_fitted_per_case(
        self, tmp_path: Path, launches: list[Path]
    ) -> None:
        """No .ss but fittable WAMIT .1 data: RdtnMod 2, the fit in each case copy.

        The template is never written; each case copy gets its own ``.ss`` file,
        fitted before the case is validated and launched, and the fit is recorded in
        the automatic configuration, the settings and every case's provenance.
        """
        fst, curve = _make_suite(tmp_path, hydro=True)
        _write_wamit(fst.parent / "Hydro" / "plat.1")
        hydro = fst.parent / "HD.dat"
        hydro.write_text(
            hydro.read_text(encoding="utf-8") + "1025 WtrDens - water density\n",
            encoding="utf-8",
        )
        out = tmp_path / "out"
        result = run_operating_sweep(
            fst, out, [8.0, 14.0], performance_csv=curve, max_workers=2
        )
        assert len(launches) == 2
        assert not (fst.parent / "Hydro" / "plat.ss").exists()
        for case in result.cases:
            assert case.converged
            assert _case_value(case.case_dir, "HD.dat", "RdtnMod") == "2"
            fitted = case.case_dir / "Case" / "Hydro" / "plat.ss"
            assert fitted.is_file()
            assert case.radiation_fit is not None
            assert case.radiation_fit.written
            assert case.radiation_fit.path == fitted
            assert case.settings.hydro_radiation_fit
        auto = result.auto_configuration
        assert auto is not None
        record = auto.get("hydro_radiation_fit")
        assert record is not None
        assert record.value is True
        assert not any("RdtnMod" in warning for warning in auto.warnings)
        driver = driver_module.LinearizationDriver(
            fst_path=fst, output_dir=out, template_root=tmp_path / "suite"
        )
        provenance = driver.provenance(result.cases)
        assert provenance.settings["hydro_radiation_fit"] is True
        written = json.loads(
            provenance.write(out / "provenance.json").read_text(encoding="utf-8")
        )
        for case_record in written["cases"]:
            fit = case_record["radiation_fit"]
            assert fit["written"] is True
            assert fit["n_states"] > 0
            assert Path(fit["path"]).name == "plat.ss"
            assert Path(fit["source"]).name == "plat.1"

    def test_explicit_fit_that_fails_in_a_case_is_a_case_error(
        self, tmp_path: Path, launches: list[Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A requested per-case fit that is not accurate enough fails that case.

        Nothing is written and OpenFAST is not launched for it; the sweep records
        the failure (``case_error``) rather than linearizing without radiation.
        """
        monkeypatch.setattr(radiation_module, "_MAX_ACCEPTED_ERROR", 1e-12)
        fst, curve = _make_suite(tmp_path, hydro=True)
        _write_wamit(fst.parent / "Hydro" / "plat.1")
        hydro = fst.parent / "HD.dat"
        hydro.write_text(
            hydro.read_text(encoding="utf-8") + "1025 WtrDens - water density\n",
            encoding="utf-8",
        )
        settings = LinearizationSettings.offshore(
            hydro_radiation_fit=True, unsteady_aero_mode=0
        )
        result = run_operating_sweep(
            fst,
            tmp_path / "out",
            [8.0],
            performance_csv=curve,
            settings=settings,
            max_workers=1,
        )
        assert launches == []
        case = result.cases[0]
        assert not case.converged
        assert case.failure_mode == "case_error"
        reason = case.attempt_history[-1].reason
        assert reason is not None
        assert "RadiationFitError" in reason
        assert "not accurate enough" in reason
        assert not list((tmp_path / "out").rglob("*.ss"))
        assert not list(tmp_path.joinpath("suite").rglob("*.ss"))

    def test_unfittable_radiation_data_falls_back(
        self, tmp_path: Path, launches: list[Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fit outside the acceptance limit: RdtnMod 0, warned, nothing written."""
        monkeypatch.setattr(radiation_module, "_MAX_ACCEPTED_ERROR", 1e-12)
        fst, curve = _make_suite(tmp_path, hydro=True)
        _write_wamit(fst.parent / "Hydro" / "plat.1")
        hydro = fst.parent / "HD.dat"
        hydro.write_text(
            hydro.read_text(encoding="utf-8") + "1025 WtrDens - water density\n",
            encoding="utf-8",
        )
        out = tmp_path / "out"
        result = run_operating_sweep(
            fst, out, [8.0], performance_csv=curve, max_workers=1
        )
        assert _case_value(out / "ws8", "HD.dat", "RdtnMod") == "0"
        assert not list(out.rglob("*.ss"))
        assert result.cases[0].radiation_fit is None
        assert not result.cases[0].settings.hydro_radiation_fit
        auto = result.auto_configuration
        assert auto is not None
        assert auto.get("hydro_radiation_fit") is None
        assert any(
            "RdtnMod=1" in warning and "not accurate enough" in warning
            for warning in auto.warnings
        )

    def test_explicit_settings_keep_their_modules(
        self, tmp_path: Path, launches: list[Path]
    ) -> None:
        """User settings are used as given; only unset constants are inferred."""
        fst, curve = _make_suite(tmp_path)
        out = tmp_path / "out"
        settings = LinearizationSettings(
            unsteady_aero_mode=0, dynamic_wake_mode=0, trim_controller=True
        )
        run_operating_sweep(
            fst, out, [14.0], performance_csv=curve, settings=settings, max_workers=1
        )
        assert _case_value(out / "ws14", "AD.dat", "DBEMT_Mod") == "0"
        assert _case_value(out / "ws14", "Model.fst", "TrimCase") == "3"


class TestFailBeforeLaunch:
    """What still cannot be configured fails before any case is copied."""

    def test_auto_configure_off_restores_the_explicit_contract(
        self, tmp_path: Path, launches: list[Path]
    ) -> None:
        """Without inference the deck's ../ references and rated wind are errors."""
        fst, curve = _make_suite(tmp_path)
        with pytest.raises(ValueError, match="rated_wind_speed is required"):
            run_operating_sweep(
                fst,
                tmp_path / "out",
                [8.0],
                performance_csv=curve,
                template_root=tmp_path,
                auto_configure=False,
            )
        assert launches == []

    def test_uninferable_rated_wind_names_the_remedy(
        self, tmp_path: Path, launches: list[Path]
    ) -> None:
        """A curve that stops below rated: the error says why and what to pass."""
        fst, _ = _make_suite(tmp_path)
        with pytest.raises(ValueError, match="never levels off") as info:
            run_operating_sweep(
                fst,
                tmp_path / "out",
                [8.0],
                curve_wind=[4.0, 8.0, 10.0],
                curve_rotor_speed=[5.0, 5.7, 7.1],
                curve_pitch=[3.0, 0.0, 0.0],
            )
        assert "rated_wind_speed" in str(info.value)
        assert launches == []

    def test_no_rated_torque_source(self, tmp_path: Path, launches: list[Path]) -> None:
        """Library deck, no DISCON.IN, no cp: an above-rated point fails up front."""
        fst, _ = _make_suite(tmp_path)
        (fst.parent / "DISCON.IN").unlink()
        with pytest.raises(FastInputError, match="DLL_InFile") as info:
            run_operating_sweep(
                fst,
                tmp_path / "out",
                [14.0],
                curve_wind=[4.0, 10.66, 14.0, 20.0],
                curve_rotor_speed=[5.0, 7.5176, 7.5176, 7.5176],
                curve_pitch=[3.0, 0.0, 10.0, 17.7],
            )
        assert "rated_generator_torque" in str(info.value)
        assert launches == []

    def test_unlinearizable_option_names_keyword_and_remedy(
        self, tmp_path: Path, launches: list[Path]
    ) -> None:
        """Nacelle drag cannot be fixed automatically: a pre-launch hint."""
        fst, curve = _make_suite(tmp_path)
        aero = fst.parent / "AD.dat"
        aero.write_text(
            aero.read_text(encoding="utf-8").replace(
                "False NacelleDrag", "True NacelleDrag"
            ),
            encoding="utf-8",
        )
        with pytest.raises(DeckValidationError, match="set NacelleDrag=False"):
            run_operating_sweep(fst, tmp_path / "out", [8.0], performance_csv=curve)
        assert launches == []

    def test_no_compatible_executable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only a v5 OpenFAST for a v4.2 deck: fail naming the release needed."""
        launched: list[object] = []
        monkeypatch.setattr(
            driver_module, "run_openfast", lambda *a, **k: launched.append(a)
        )
        exe = _write(tmp_path / "bin" / "of5.exe", "MZ")

        def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(command, 0, "OpenFAST-v5.0.0\n", "")

        monkeypatch.setattr(executables_module.subprocess, "run", run)
        monkeypatch.setattr(executables_module, "_version_cache", {})
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        monkeypatch.setenv(OPENFAST_EXECUTABLES_ENV, str(exe))
        monkeypatch.setenv(OPENFAST_CONFIG_ENV, str(tmp_path / "none.txt"))
        fst, curve = _make_suite(tmp_path)
        out = tmp_path / "out"
        with pytest.raises(OpenFastError, match=r"OpenFAST >=4\.2\.0, <5\.0\.0"):
            run_operating_sweep(fst, out, [8.0], performance_csv=curve)
        assert launched == []
        assert not (out / "ws8").exists()


def _reference_root() -> Path:
    runs = os.environ.get("VANE_REFERENCE_RUNS_DIR")
    if not runs:
        pytest.skip("reference-turbine data unavailable; set VANE_REFERENCE_RUNS_DIR")
    root = Path(runs).parent
    if not (root / "_performance").is_dir():
        pytest.skip("reference-turbine data unavailable; set VANE_REFERENCE_RUNS_DIR")
    return root


class TestReferenceDecksWithoutLaunching:
    """The real decks configure with default arguments (OpenFAST stubbed out)."""

    def test_rtest_land_deck(
        self, rtest_dir: Path, tmp_path: Path, launches: list[Path]
    ) -> None:
        """The r-test deck (../5MW_Baseline) needs no template_root or constants."""
        curve = _reference_root() / "_performance" / "nrel5mw_curve.csv"
        deck = (
            rtest_dir / "glue-codes/openfast/5MW_Land_DLL_WTurb/5MW_Land_DLL_WTurb.fst"
        )
        out = tmp_path / "o"
        result = run_operating_sweep(
            deck, out, [8.0, 16.0], performance_csv=curve, max_workers=2
        )
        assert len(launches) == 2
        case = out / "ws16"
        assert (case / "5MW_Baseline").is_dir()
        assert _case_value(case, "5MW_Land_DLL_WTurb.fst", "TrimCase") == "3"
        torque = float(
            _case_value(case, "NRELOffshrBsline5MW_Onshore_ServoDyn.dat", "VS_RtTq")
        )
        assert torque == pytest.approx(43093.5, rel=0.005)
        assert result.auto_configuration is not None

    def test_iea15_monopile_deck(self, tmp_path: Path, launches: list[Path]) -> None:
        """The IEA 15-MW monopile (../IEA-15-240-RWT) configures automatically."""
        root = _reference_root()
        deck = (
            root
            / "IEA-15-240-RWT/OpenFAST/IEA-15-240-RWT-Monopile"
            / "IEA-15-240-RWT-Monopile.fst"
        )
        if not deck.is_file():
            pytest.skip("IEA 15-MW monopile deck unavailable")
        out = tmp_path / "o"
        run_operating_sweep(
            deck,
            out,
            [14.0],
            performance_csv=root / "_performance" / "iea15_curve.csv",
            max_workers=1,
        )
        case = out / "ws14"
        assert (case / "IEA-15-240-RWT" / "IEA-15-240-RWT_InflowFile.dat").is_file()
        gain = float(_case_value(case, "IEA-15-240-RWT-Monopile.fst", "TrimGain"))
        assert gain == pytest.approx(2e-4 * (63.0 / 120.97) ** 3)
        tolerance = float(_case_value(case, "IEA-15-240-RWT-Monopile.fst", "TrimTol"))
        assert tolerance == pytest.approx(1e-3 * (120.97 / 63.0) ** 3)
        torque = float(
            _case_value(case, "IEA-15-240-RWT-Monopile_ServoDyn.dat", "VS_RtTq")
        )
        assert torque == pytest.approx(19786800.0)
