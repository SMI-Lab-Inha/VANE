# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for the automated operating-point sweep helper."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from vane.io.fst_writer import FastInputError, FastInputFile
from vane.run import (
    LinearizationSettings,
    OperatingPoint,
    OperatingSweepResult,
    PlatformPosition,
    TrimTolerances,
    run_operating_sweep,
)
from vane.run.runner import OpenFastResult

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _stub_executable_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the sweeps with a placeholder executable (OpenFAST is stubbed out here)."""
    from pathlib import Path as _Path

    from vane.run import auto_sweep as _auto_sweep
    from vane.run.executables import DeckGeneration, OpenFastSelection

    def select(fst_path: object, executable: object = None) -> OpenFastSelection:
        return OpenFastSelection(
            executable=_Path(str(executable or "openfast")),
            version=None,
            generation=DeckGeneration("unknown", (0, 0, 0), None),
            source="test",
        )

    monkeypatch.setattr(_auto_sweep, "select_openfast_executable", select)


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
True DrTrDOF - Drivetrain rotational-flexibility DOF (flag)
True YawDOF - Yaw DOF (flag)
"""

# Minimal steady-wind InflowWind sub-file the curve path writes the wind speed into,
# mirroring how a real OpenFAST deck references its InflowWind input.
_IW = """\
1 WindType - switch
8 HWindSpeed - horizontal wind speed (m/s)
0 PLExp - power law exponent
"""


def _make_template(tmp_path: Path) -> Path:
    root = tmp_path / "template"
    root.mkdir()
    (root / "ED.dat").write_text(_ED, encoding="utf-8")
    (root / "IW.dat").write_text(_IW, encoding="utf-8")
    fst = root / "Model.fst"
    fst.write_text(_PRIMARY, encoding="utf-8")
    return fst


def test_requires_rotor_performance_or_settle(tmp_path: Path) -> None:
    """The helper fails before launch when rotor speed/pitch cannot be known."""
    fst = _make_template(tmp_path)
    with pytest.raises(ValueError, match="rotor-performance data is required"):
        run_operating_sweep(
            fst,
            tmp_path / "out",
            [8.0],
            settings=LinearizationSettings(rated_wind_speed=11.4),
            max_workers=1,
        )


def test_curve_drives_linearization_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A supplied operating curve is the source of rotor speed and pitch."""
    from vane.run import driver as driver_module

    fst = _make_template(tmp_path)

    def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
        return OpenFastResult(fst_path=deck_path, returncode=0)

    monkeypatch.setattr(driver_module, "run_openfast", fake_run)
    result = run_operating_sweep(
        fst,
        tmp_path / "out",
        [8.0, 12.0],
        curve_wind=[8.0, 12.0],
        curve_rotor_speed=[9.0, 12.0],
        curve_pitch=[0.0, 4.0],
        settings=LinearizationSettings(rated_wind_speed=11.4),
        max_workers=1,
        validate=False,
    )

    assert [point.wind_speed for point in result.operating_points] == [8.0, 12.0]
    assert [point.rotor_speed_rpm for point in result.operating_points] == [
        9.0,
        12.0,
    ]
    assert [point.blade_pitch_deg for point in result.operating_points] == [
        0.0,
        4.0,
    ]
    assert len(result.cases) == 2


def test_sweep_disables_yaw_and_drivetrain_dofs_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sweep's default linearization config disables YawDOF and DrTrDOF in ED."""
    from vane.io.fst_writer import FastInputFile
    from vane.run import driver as driver_module

    fst = _make_template(tmp_path)

    def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
        return OpenFastResult(fst_path=deck_path, returncode=0)

    monkeypatch.setattr(driver_module, "run_openfast", fake_run)
    run_operating_sweep(
        fst,
        tmp_path / "out",
        [8.0],
        curve_wind=[5.0, 12.0],
        curve_rotor_speed=[6.0, 12.0],
        curve_pitch=[0.0, 4.0],
        max_workers=1,
        validate=False,
        settings=LinearizationSettings.offshore(rated_wind_speed=11.4),
    )

    ed = FastInputFile.read(tmp_path / "out" / "ws8" / "ED.dat")
    assert ed.get("YawDOF") == "False"
    assert ed.get("DrTrDOF") == "False"


def test_sweep_structural_dofs_override_reaches_ed_deck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A structural_dofs override on run_operating_sweep is applied per case."""
    from vane.io.fst_writer import FastInputFile
    from vane.run import driver as driver_module

    fst = _make_template(tmp_path)

    def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
        return OpenFastResult(fst_path=deck_path, returncode=0)

    monkeypatch.setattr(driver_module, "run_openfast", fake_run)
    run_operating_sweep(
        fst,
        tmp_path / "out",
        [8.0],
        curve_wind=[5.0, 12.0],
        curve_rotor_speed=[6.0, 12.0],
        curve_pitch=[0.0, 4.0],
        structural_dofs={"DrTrDOF": True},  # re-enable drivetrain torsion
        max_workers=1,
        validate=False,
        settings=LinearizationSettings.offshore(rated_wind_speed=11.4),
    )

    ed = FastInputFile.read(tmp_path / "out" / "ws8" / "ED.dat")
    assert ed.get("DrTrDOF") == "True"  # override applied
    assert ed.get("YawDOF") == "False"  # default still in force


def test_performance_csv_drives_linearization_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A performance CSV is parsed to the curve and reaches the linearized ED deck."""
    from vane.io.fst_writer import FastInputFile
    from vane.run import driver as driver_module

    fst = _make_template(tmp_path)
    csv_path = tmp_path / "perf.csv"
    csv_path.write_text(
        "WindSpeed,RotSpeed,BldPitch,cp,gen_power\n"
        "5.0,6.0,0.0,0.48,1200.0\n"
        "12.0,12.0,4.0,0.45,15000.0\n",
        encoding="utf-8",
    )

    def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
        return OpenFastResult(fst_path=deck_path, returncode=0)

    monkeypatch.setattr(driver_module, "run_openfast", fake_run)
    result = run_operating_sweep(
        fst,
        tmp_path / "out",
        [8.0],
        performance_csv=csv_path,
        max_workers=1,
        validate=False,
        settings=LinearizationSettings.offshore(rated_wind_speed=11.4),
    )

    # The parsed curve set the rotor speed/pitch interpolated at 8 m/s.
    point = result.operating_points[0]
    assert point.wind_speed == 8.0
    assert 6.0 <= point.rotor_speed_rpm <= 12.0
    assert 0.0 <= point.blade_pitch_deg <= 4.0
    # And those values reached the configured ElastoDyn deck.
    ed = FastInputFile.read(tmp_path / "out" / "ws8" / "ED.dat")
    assert float(ed.get("RotSpeed")) == pytest.approx(point.rotor_speed_rpm)


def test_performance_csv_mutually_exclusive_with_curve(tmp_path: Path) -> None:
    """Passing both performance_csv and a curve fails before launch."""
    fst = _make_template(tmp_path)
    csv_path = tmp_path / "perf.csv"
    csv_path.write_text(
        "WindSpeed,RotSpeed,BldPitch\n5.0,6.0,0.0\n12.0,12.0,4.0\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_operating_sweep(
            fst,
            tmp_path / "out",
            [8.0],
            performance_csv=csv_path,
            curve_wind=[5.0, 12.0],
            curve_rotor_speed=[6.0, 12.0],
            curve_pitch=[0.0, 4.0],
            max_workers=1,
            validate=False,
            settings=LinearizationSettings.offshore(rated_wind_speed=11.4),
        )


def test_performance_csv_seeds_settle_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A performance CSV with settle=True seeds the settles; the settled point wins."""
    from vane.run import auto_sweep as auto_module
    from vane.run import driver as driver_module

    fst = _make_template(tmp_path)
    csv_path = tmp_path / "perf.csv"
    csv_path.write_text(
        "WindSpeed,RotSpeed,BldPitch\n5.0,6.0,0.0\n12.0,12.0,4.0\n", encoding="utf-8"
    )
    seeds: list[tuple[object, object]] = []

    def fake_settle(
        deck_path: Path, wind_speed: float, **kwargs: object
    ) -> OperatingPoint:
        seeds.append((kwargs["rotor_speed_rpm"], kwargs["blade_pitch_deg"]))
        return OperatingPoint(wind_speed, 9.0, 1.5, name=f"ws{wind_speed:g}")

    monkeypatch.setattr(auto_module, "solve_steady_state", fake_settle)
    monkeypatch.setattr(driver_module, "run_openfast", _fake_lin_run)
    result = run_operating_sweep(
        fst,
        tmp_path / "out",
        [8.0],
        performance_csv=csv_path,
        settle=True,
        max_workers=1,
        validate=False,
        settings=LinearizationSettings.offshore(rated_wind_speed=11.4),
    )

    assert len(seeds) == 1
    rotor_seed, pitch_seed = seeds[0]
    assert 6.0 < float(str(rotor_seed)) < 12.0
    assert 0.0 <= float(str(pitch_seed)) < 4.0
    point = result.operating_points[0]
    assert (point.rotor_speed_rpm, point.blade_pitch_deg) == (9.0, 1.5)
    assert point.settle_curve_check is not None
    assert point.settle_curve_check.rotor_speed.curve == pytest.approx(rotor_seed)


def _fake_lin_run(deck_path: Path, **_: object) -> OpenFastResult:
    return OpenFastResult(
        fst_path=deck_path,
        returncode=0,
        lin_files=[deck_path.with_suffix(".1.lin")],
    )


class TestSeededSettle:
    """settle=True combined with an operating curve used as initial conditions."""

    _CURVE: ClassVar[dict[str, list[float]]] = {
        "curve_wind": [5.0, 12.0],
        "curve_rotor_speed": [6.0, 12.0],
        "curve_pitch": [0.0, 4.0],
    }

    def _sweep(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        settled: dict[float, tuple[float, float]],
        **kwargs: object,
    ) -> tuple[OperatingSweepResult, list[dict[str, object]]]:
        from vane.run import auto_sweep as auto_module
        from vane.run import driver as driver_module

        fst = _make_template(tmp_path)
        calls: list[dict[str, object]] = []

        def fake_settle(
            deck_path: Path, wind_speed: float, **settle_kwargs: object
        ) -> OperatingPoint:
            calls.append({"wind_speed": wind_speed, **settle_kwargs})
            rpm, pitch = settled[wind_speed]
            return OperatingPoint(wind_speed, rpm, pitch, name=f"ws{wind_speed:g}")

        monkeypatch.setattr(auto_module, "solve_steady_state", fake_settle)
        monkeypatch.setattr(driver_module, "run_openfast", _fake_lin_run)
        options: dict[str, Any] = {
            "settle": True,
            "max_workers": 1,
            "validate": False,
            "settings": LinearizationSettings.offshore(rated_wind_speed=11.4),
            **self._CURVE,
            **kwargs,
        }
        result = run_operating_sweep(fst, tmp_path / "out", sorted(settled), **options)
        return result, calls

    def test_curve_seeds_rotor_speed_and_pitch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each settle starts from the curve; the settled point is linearized."""
        result, calls = self._sweep(
            tmp_path, monkeypatch, {5.0: (6.0, 0.0), 12.0: (12.0, 4.0)}
        )
        assert [(c["rotor_speed_rpm"], c["blade_pitch_deg"]) for c in calls] == [
            pytest.approx((6.0, 0.0)),
            pytest.approx((12.0, 4.0)),
        ]
        assert all(c["platform_position"] is None for c in calls)
        assert [p.rotor_speed_rpm for p in result.operating_points] == [6.0, 12.0]
        for point in result.operating_points:
            assert point.settle_curve_check is not None
            assert point.settle_curve_check.status == "pass"
        # The linearization cases run at the settled points.
        assert [c.operating_point.rotor_speed_rpm for c in result.cases] == [6.0, 12.0]

    def test_disagreement_is_recorded_and_warned_not_raised(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A settled point far from the curve is flagged, logged and still used."""
        with caplog.at_level(logging.WARNING, logger="vane.run.auto_sweep"):
            result, _ = self._sweep(tmp_path, monkeypatch, {12.0: (11.0, 6.0)})
        point = result.operating_points[0]
        assert point.rotor_speed_rpm == 11.0
        check = point.settle_curve_check
        assert check is not None
        assert check.status == "flag"
        assert check.flagged_quantities == ("rotor_speed", "pitch")
        assert check.pitch.deviation == pytest.approx(2.0)
        assert "disagrees with the operating curve" in caplog.text

    def test_custom_tolerances_apply(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """settle_curve_tolerances widens the comparison."""
        result, _ = self._sweep(
            tmp_path,
            monkeypatch,
            {12.0: (11.0, 6.0)},
            settle_curve_tolerances=TrimTolerances(rotor_speed_rel=0.1, pitch_deg=3.0),
        )
        check = result.operating_points[0].settle_curve_check
        assert check is not None
        assert check.status == "pass"

    def test_unseeded_settle_starts_from_rest_without_check(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a curve a settle starts from 0 rpm / 0 deg, as before."""
        result, calls = self._sweep(
            tmp_path,
            monkeypatch,
            {8.0: (9.0, 1.0)},
            curve_wind=None,
            curve_rotor_speed=None,
            curve_pitch=None,
        )
        assert (calls[0]["rotor_speed_rpm"], calls[0]["blade_pitch_deg"]) == (0.0, 0.0)
        assert result.operating_points[0].settle_curve_check is None

    def test_single_initial_platform_reaches_every_settle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One PlatformPosition seeds every settle run."""
        platform = PlatformPosition(surge=10.0, pitch=4.0)
        _, calls = self._sweep(
            tmp_path,
            monkeypatch,
            {8.0: (9.0, 1.0), 10.0: (10.0, 1.0)},
            settle_initial_platform=platform,
        )
        assert [c["platform_position"] for c in calls] == [platform, platform]

    def test_platform_mapping_by_wind(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A wind-keyed mapping seeds only the winds it names."""
        platform = PlatformPosition(surge=20.0)
        _, calls = self._sweep(
            tmp_path,
            monkeypatch,
            {8.0: (9.0, 1.0), 10.0: (10.0, 1.0)},
            settle_initial_platform={10: platform},
        )
        assert [c["platform_position"] for c in calls] == [None, platform]

    def test_platform_mapping_with_unknown_wind_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A mapping key that is not a requested wind fails before any settle."""
        with pytest.raises(ValueError, match="not in wind_speeds"):
            self._sweep(
                tmp_path,
                monkeypatch,
                {8.0: (9.0, 1.0)},
                settle_initial_platform={9.0: PlatformPosition(surge=1.0)},
            )

    @pytest.mark.parametrize("seed", [{8.0: (1.0, 2.0)}, [1.0]])
    def test_platform_of_wrong_type_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seed: object
    ) -> None:
        """A non-PlatformPosition seed is a type error."""
        with pytest.raises(TypeError, match="PlatformPosition"):
            self._sweep(
                tmp_path,
                monkeypatch,
                {8.0: (9.0, 1.0)},
                settle_initial_platform=seed,
            )

    def test_wind_outside_curve_rejected_before_settle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A requested wind outside the seeding curve fails before any settle."""
        with pytest.raises(ValueError, match="outside the operating-curve range"):
            self._sweep(tmp_path, monkeypatch, {13.0: (12.0, 5.0)})

    def test_settle_structural_dofs_recorded(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Settles keep the deck's DOFs; the linearization still disables them."""
        with caplog.at_level(logging.INFO, logger="vane.run.auto_sweep"):
            result, _ = self._sweep(tmp_path, monkeypatch, {8.0: (9.0, 1.0)})
        assert result.settle_structural_dofs == {"YawDOF": True, "DrTrDOF": True}
        assert "linearization uses DrTrDOF=False" in caplog.text
        ed = FastInputFile.read(result.cases[0].fst_path.parent / "ED.dat")
        assert ed.get("DrTrDOF").lower() == "false"
        assert ed.get("YawDOF").lower() == "false"


class TestSettleOnlyOptions:
    """Settle-only options are rejected without settle=True."""

    def test_initial_platform_requires_settle(self, tmp_path: Path) -> None:
        """settle_initial_platform without settle fails before launch."""
        fst = _make_template(tmp_path)
        with pytest.raises(ValueError, match="settle_initial_platform applies only"):
            run_operating_sweep(
                fst,
                tmp_path / "out",
                [8.0],
                curve_wind=[5.0, 12.0],
                curve_rotor_speed=[6.0, 12.0],
                curve_pitch=[0.0, 4.0],
                settle_initial_platform=PlatformPosition(surge=1.0),
                max_workers=1,
                validate=False,
                settings=LinearizationSettings.offshore(rated_wind_speed=11.4),
            )

    def test_curve_tolerances_require_seeded_settle(self, tmp_path: Path) -> None:
        """settle_curve_tolerances needs both settle=True and a curve."""
        fst = _make_template(tmp_path)
        with pytest.raises(ValueError, match="settle_curve_tolerances applies only"):
            run_operating_sweep(
                fst,
                tmp_path / "out",
                [8.0],
                settle=True,
                settle_curve_tolerances=TrimTolerances(),
                max_workers=1,
                validate=False,
                settings=LinearizationSettings.offshore(rated_wind_speed=11.4),
            )

    def test_curve_settle_without_structural_dofs_in_deck(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A deck without the DOF flags records an empty settle-DOF map."""
        from vane.run import auto_sweep as auto_module
        from vane.run import driver as driver_module

        fst = _make_template(tmp_path)
        (fst.parent / "ED.dat").write_text(
            "9.55 RotSpeed - rpm\n0.0 BlPitch(1) - deg\nmaybe DrTrDOF - flag\n",
            encoding="utf-8",
        )

        def fake_settle(deck_path: Path, wind_speed: float, **_: object) -> object:
            return OperatingPoint(wind_speed, 9.0, 1.0, name=f"ws{wind_speed:g}")

        monkeypatch.setattr(auto_module, "solve_steady_state", fake_settle)
        monkeypatch.setattr(driver_module, "run_openfast", _fake_lin_run)
        result = run_operating_sweep(
            fst,
            tmp_path / "out",
            [8.0],
            settle=True,
            max_workers=1,
            validate=False,
            settings=LinearizationSettings.offshore(rated_wind_speed=11.4),
        )
        assert result.settle_structural_dofs == {}


def test_settle_path_parallelizes_before_linearization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With settle=True, operating points come from settle runs before the sweep."""
    from vane.run import auto_sweep as auto_module
    from vane.run import driver as driver_module

    fst = _make_template(tmp_path)
    settled: list[float] = []

    def fake_settle(deck_path: Path, wind_speed: float, **_: object) -> OperatingPoint:
        settled.append(wind_speed)
        return OperatingPoint(
            wind_speed, 10.0 + wind_speed, 1.0, name=f"ws{wind_speed:g}"
        )

    def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
        return OpenFastResult(
            fst_path=deck_path,
            returncode=0,
            lin_files=[deck_path.with_suffix(".1.lin")],
        )

    monkeypatch.setattr(auto_module, "solve_steady_state", fake_settle)
    monkeypatch.setattr(driver_module, "run_openfast", fake_run)
    result = run_operating_sweep(
        fst,
        tmp_path / "out",
        [8.0, 10.0],
        settle=True,
        settings=LinearizationSettings(rated_wind_speed=11.4),
        max_workers=2,
        validate=False,
    )

    assert sorted(settled) == [8.0, 10.0]
    assert [point.rotor_speed_rpm for point in result.operating_points] == [
        18.0,
        20.0,
    ]


def test_settle_runs_use_the_linearization_shear_exponent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Settle runs receive the same PLExp convention the linearization writes."""
    from vane.run import auto_sweep as auto_module
    from vane.run import driver as driver_module

    fst = _make_template(tmp_path)
    received: list[object] = []

    def fake_settle(
        deck_path: Path, wind_speed: float, **kwargs: object
    ) -> OperatingPoint:
        received.append(kwargs["wind_shear_exponent"])
        return OperatingPoint(wind_speed, 10.0, 1.0, name=f"ws{wind_speed:g}")

    def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
        return OpenFastResult(
            fst_path=deck_path,
            returncode=0,
            lin_files=[deck_path.with_suffix(".1.lin")],
        )

    monkeypatch.setattr(auto_module, "solve_steady_state", fake_settle)
    monkeypatch.setattr(driver_module, "run_openfast", fake_run)
    result = run_operating_sweep(
        fst,
        tmp_path / "out",
        [8.0],
        settle=True,
        settings=LinearizationSettings(rated_wind_speed=11.4, wind_shear_exponent=0.14),
        max_workers=1,
        validate=False,
    )

    assert received == [0.14]
    inflow = result.cases[0].fst_path.parent / "IW.dat"
    assert "0.14" in inflow.read_text(encoding="utf-8")


def test_settle_case_name_colliding_with_template_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A settle case whose name collides with the template dir is rejected."""
    from vane.run import auto_sweep as auto_module

    # The template lives in a directory named like the wind-8 settle case, with the
    # parent as the output workspace, so an unguarded path would rmtree the template.
    root = tmp_path / "settle_ws8"
    root.mkdir()
    (root / "ED.dat").write_text(_ED, encoding="utf-8")
    (root / "IW.dat").write_text(_IW, encoding="utf-8")
    fst = root / "Model.fst"
    fst.write_text(_PRIMARY, encoding="utf-8")

    settled: list[float] = []

    def fake_settle(deck_path: Path, wind_speed: float, **_: object) -> OperatingPoint:
        settled.append(wind_speed)
        return OperatingPoint(wind_speed, 9.0, 0.0, name=f"ws{wind_speed:g}")

    monkeypatch.setattr(auto_module, "solve_steady_state", fake_settle)
    with pytest.raises(ValueError, match="overlap the template root"):
        run_operating_sweep(
            fst,
            tmp_path,  # output_dir is the parent of the template directory
            [8.0],
            settle=True,
            template_root=root,
            max_workers=1,
            validate=False,
            settings=LinearizationSettings.offshore(rated_wind_speed=11.4),
        )

    # The guard fired before any settle ran, and the template survives intact.
    assert settled == []
    assert fst.is_file()
    assert (root / "ED.dat").is_file()


def test_settle_winds_colliding_to_same_label_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Distinct winds that :g-collapse to one settle label are rejected, no clobber."""
    from vane.run import auto_sweep as auto_module

    fst = _make_template(tmp_path)
    settled: list[float] = []

    def fake_settle(deck_path: Path, wind_speed: float, **_: object) -> OperatingPoint:
        settled.append(wind_speed)
        return OperatingPoint(wind_speed, 9.0, 0.0, name=f"ws{wind_speed:g}")

    monkeypatch.setattr(auto_module, "solve_steady_state", fake_settle)
    # 8.0000001 and 8.0000002 are distinct floats but both format to "ws8" with :g, so
    # they would share one settle workspace and clobber each other in parallel.
    with pytest.raises(ValueError, match="colliding settle case names"):
        run_operating_sweep(
            fst,
            tmp_path / "out",
            [8.0000001, 8.0000002],
            settle=True,
            max_workers=2,
            validate=False,
            settings=LinearizationSettings.offshore(rated_wind_speed=11.4),
        )

    # The collision is caught before any settle worker runs, so nothing is clobbered.
    assert settled == []
    assert not (tmp_path / "out" / "settle_ws8").exists()


def _make_template_without_inflow(tmp_path: Path) -> Path:
    """Write a template whose primary deck references no InflowWind file."""
    root = tmp_path / "template"
    root.mkdir()
    (root / "ED.dat").write_text(_ED, encoding="utf-8")
    fst = root / "Model.fst"
    fst.write_text(
        '"ED.dat" EDFile - ElastoDyn file\nFalse Linearize - flag\n',
        encoding="utf-8",
    )
    return fst


class TestFailFastValidation:
    """run_operating_sweep validates inputs before launching any OpenFAST."""

    def test_missing_rated_wind_speed_rejected_before_launch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A rotating sweep without rated_wind_speed fails before any run."""
        from vane.run import driver as driver_module

        calls: list[object] = []
        monkeypatch.setattr(
            driver_module, "run_openfast", lambda *a, **k: calls.append(a)
        )
        fst = _make_template(tmp_path)
        with pytest.raises(ValueError, match="rated_wind_speed is required"):
            run_operating_sweep(
                fst,
                tmp_path / "out",
                [8.0],
                curve_wind=[5.0, 12.0],
                curve_rotor_speed=[6.0, 12.0],
                curve_pitch=[0.0, 4.0],
                max_workers=1,
                validate=False,
            )
        assert calls == []
        assert not (tmp_path / "out").exists()

    @pytest.mark.parametrize(
        ("rotor_speeds", "settings"),
        [
            ([0.0, 0.0], LinearizationSettings.offshore()),
            (
                [6.0, 12.0],
                LinearizationSettings.offshore(
                    calc_steady=False, trim_controller=False
                ),
            ),
        ],
        ids=["parked-curve", "untrimmed"],
    )
    def test_rated_wind_not_required_without_region_handling(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        rotor_speeds: list[float],
        settings: LinearizationSettings,
    ) -> None:
        """A parked or untrimmed sweep never chooses a trim region."""
        from vane.run import driver as driver_module

        monkeypatch.setattr(
            driver_module,
            "run_openfast",
            lambda deck, **_: OpenFastResult(fst_path=deck, returncode=0),
        )
        fst = _make_template(tmp_path)
        result = run_operating_sweep(
            fst,
            tmp_path / "out",
            [8.0],
            curve_wind=[5.0, 12.0],
            curve_rotor_speed=rotor_speeds,
            curve_pitch=[0.0, 4.0],
            max_workers=1,
            validate=False,
            settings=settings,
        )
        assert len(result.cases) == 1

    def test_empty_wind_speeds_rejected(self, tmp_path: Path) -> None:
        """An empty wind-speed list fails before launch."""
        fst = _make_template(tmp_path)
        with pytest.raises(ValueError, match="wind_speeds is empty"):
            run_operating_sweep(fst, tmp_path / "out", [], settle=True, max_workers=1)

    def test_duplicate_wind_speeds_rejected(self, tmp_path: Path) -> None:
        """Duplicate requested winds are rejected up front."""
        fst = _make_template(tmp_path)
        with pytest.raises(ValueError, match="duplicate wind speeds"):
            run_operating_sweep(
                fst,
                tmp_path / "out",
                [8.0, 8.0],
                curve_wind=[5.0, 12.0],
                curve_rotor_speed=[6.0, 12.0],
                curve_pitch=[0.0, 4.0],
                max_workers=1,
                validate=False,
                settings=LinearizationSettings.offshore(rated_wind_speed=11.4),
            )

    def test_non_finite_wind_speed_rejected(self, tmp_path: Path) -> None:
        """A non-finite requested wind is rejected up front."""
        fst = _make_template(tmp_path)
        with pytest.raises(ValueError, match="must all be finite"):
            run_operating_sweep(
                fst,
                tmp_path / "out",
                [float("nan")],
                settle=True,
                max_workers=1,
                settings=LinearizationSettings.offshore(rated_wind_speed=11.4),
            )

    def test_negative_wind_speed_rejected(self, tmp_path: Path) -> None:
        """A negative requested wind is rejected up front."""
        fst = _make_template(tmp_path)
        with pytest.raises(ValueError, match="must be non-negative"):
            run_operating_sweep(
                fst,
                tmp_path / "out",
                [-1.0],
                settle=True,
                max_workers=1,
                settings=LinearizationSettings.offshore(rated_wind_speed=11.4),
            )

    def test_mismatched_curve_lengths_rejected(self, tmp_path: Path) -> None:
        """Curve arrays of unequal length fail before launch."""
        fst = _make_template(tmp_path)
        with pytest.raises(ValueError, match="same length"):
            run_operating_sweep(
                fst,
                tmp_path / "out",
                [8.0],
                curve_wind=[5.0, 12.0],
                curve_rotor_speed=[6.0],
                curve_pitch=[0.0, 4.0],
                max_workers=1,
                validate=False,
                settings=LinearizationSettings.offshore(rated_wind_speed=11.4),
            )

    def test_short_curve_rejected(self, tmp_path: Path) -> None:
        """A one-sample curve is too short to interpolate."""
        fst = _make_template(tmp_path)
        with pytest.raises(ValueError, match="at least two samples"):
            run_operating_sweep(
                fst,
                tmp_path / "out",
                [8.0],
                curve_wind=[8.0],
                curve_rotor_speed=[9.0],
                curve_pitch=[0.0],
                max_workers=1,
                validate=False,
                settings=LinearizationSettings.offshore(rated_wind_speed=11.4),
            )

    def test_missing_template_root_rejected(self, tmp_path: Path) -> None:
        """A template_root that does not exist fails before launch."""
        fst = _make_template(tmp_path)
        with pytest.raises(ValueError, match="template_root does not exist"):
            run_operating_sweep(
                fst,
                tmp_path / "out",
                [8.0],
                settle=True,
                template_root=tmp_path / "nonexistent",
                max_workers=1,
                settings=LinearizationSettings.offshore(rated_wind_speed=11.4),
            )

    def test_missing_inflow_for_wind_point_rejected(self, tmp_path: Path) -> None:
        """A positive wind point with no usable InflowWind file fails before launch."""
        fst = _make_template_without_inflow(tmp_path)
        with pytest.raises(FastInputError, match="no usable InflowWind file"):
            run_operating_sweep(
                fst,
                tmp_path / "out",
                [8.0],
                curve_wind=[5.0, 12.0],
                curve_rotor_speed=[6.0, 12.0],
                curve_pitch=[0.0, 4.0],
                max_workers=1,
                validate=False,
                settings=LinearizationSettings.offshore(rated_wind_speed=11.4),
            )

    def test_missing_template_deck_rejected(self, tmp_path: Path) -> None:
        """A missing template .fst is rejected with FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match=r"template \.fst deck not found"):
            run_operating_sweep(
                tmp_path / "absent.fst",
                tmp_path / "out",
                [8.0],
                settle=True,
                max_workers=1,
                settings=LinearizationSettings.offshore(rated_wind_speed=11.4),
            )

    def test_template_root_not_containing_deck_rejected(self, tmp_path: Path) -> None:
        """A template_root that does not contain the deck fails before launch."""
        fst = _make_template(tmp_path)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        with pytest.raises(ValueError, match="not inside template_root"):
            run_operating_sweep(
                fst,
                tmp_path / "out",
                [8.0],
                settle=True,
                template_root=elsewhere,
                max_workers=1,
                settings=LinearizationSettings.offshore(rated_wind_speed=11.4),
            )

    def test_partial_curve_with_settle_rejected(self, tmp_path: Path) -> None:
        """A partial seeding curve with settle=True is rejected before launch."""
        fst = _make_template(tmp_path)
        with pytest.raises(ValueError, match="supplied together"):
            run_operating_sweep(
                fst,
                tmp_path / "out",
                [8.0],
                settle=True,
                curve_wind=[5.0, 12.0],
                curve_rotor_speed=[6.0, 12.0],
                max_workers=1,
                validate=False,
                settings=LinearizationSettings.offshore(rated_wind_speed=11.4),
            )

    def test_partial_curve_rejected(self, tmp_path: Path) -> None:
        """An operating curve missing one of its arrays is rejected before launch."""
        fst = _make_template(tmp_path)
        with pytest.raises(ValueError, match="must be supplied together"):
            run_operating_sweep(
                fst,
                tmp_path / "out",
                [8.0],
                curve_wind=[5.0, 12.0],
                curve_rotor_speed=[6.0, 12.0],
                # curve_pitch omitted
                max_workers=1,
                validate=False,
                settings=LinearizationSettings.offshore(rated_wind_speed=11.4),
            )


# ServoDyn routing pitch and torque through a controller library that is a Linux path.
_SERVO_DLL = """\
5 PCMode - pitch control mode
5 VSContrl - variable-speed control mode
"/home/runner/miniconda3/envs/test/lib/libdiscon.so" DLL_FileName - controller library
"""

_SETTLE_OUT = """\
OpenFAST results header line

Time\tRotSpeed\tBldPitch1
(s)\t(rpm)\t(deg)
600.0\t7.40\t9.0
601.0\t7.60\t11.0
"""


def _make_controlled_template(tmp_path: Path, servo_ref: str = '"SD.dat"') -> Path:
    """Write the sweep template plus a ServoDyn file using the controller library."""
    fst = _make_template(tmp_path)
    (fst.parent / "SD.dat").write_text(_SERVO_DLL, encoding="utf-8")
    fst.write_text(
        _PRIMARY + f"{servo_ref} ServoFile - ServoDyn file\n", encoding="utf-8"
    )
    return fst


class TestSettleControllerLibrary:
    """run_operating_sweep checks and overrides the settle runs' controller library."""

    def test_missing_library_fails_before_any_settle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A deck naming a missing library fails at the boundary; nothing is copied."""
        from vane.run import SteadyStateError
        from vane.run import auto_sweep as auto_module

        fst = _make_controlled_template(tmp_path)
        settled: list[float] = []

        def fake_settle(deck_path: Path, wind_speed: float, **_: object) -> None:
            settled.append(wind_speed)

        monkeypatch.setattr(auto_module, "solve_steady_state", fake_settle)
        with pytest.raises(SteadyStateError, match="controller library not found"):
            run_operating_sweep(
                fst,
                tmp_path / "out",
                [8.0],
                settle=True,
                settings=LinearizationSettings(rated_wind_speed=11.4),
                max_workers=1,
                validate=False,
            )
        assert settled == []
        assert not (tmp_path / "out").exists()

    def test_override_reaches_the_settle_copy_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The library is written into the settle case's ServoDyn, not the template."""
        from vane.io.fst_writer import FastInputFile
        from vane.run import driver as driver_module
        from vane.run import steady_state as steady_state_module

        fst = _make_controlled_template(tmp_path)
        library = tmp_path / "controller" / "libdiscon.dll"
        library.parent.mkdir()
        library.write_bytes(b"MZ")

        def fake_settle_run(fst_path: Path, **_: object) -> OpenFastResult:
            fst_path.with_suffix(".out").write_text(_SETTLE_OUT, encoding="utf-8")
            return OpenFastResult(fst_path=fst_path, returncode=0)

        def fake_lin_run(deck_path: Path, **_: object) -> OpenFastResult:
            return OpenFastResult(
                fst_path=deck_path,
                returncode=0,
                lin_files=[deck_path.with_suffix(".1.lin")],
            )

        monkeypatch.setattr(steady_state_module, "run_openfast", fake_settle_run)
        monkeypatch.setattr(driver_module, "run_openfast", fake_lin_run)
        result = run_operating_sweep(
            fst,
            tmp_path / "out",
            [8.0],
            settle=True,
            controller_library=library,
            settings=LinearizationSettings(rated_wind_speed=11.4),
            max_workers=1,
            validate=False,
            require_settled=False,  # the synthetic .out has no platform channels
        )

        assert result.operating_points[0].rotor_speed_rpm == pytest.approx(7.5)
        settle_servo = FastInputFile.read(tmp_path / "out" / "settle_ws8" / "SD.dat")
        assert settle_servo.resolve("DLL_FileName") == library.resolve()
        template_servo = (fst.parent / "SD.dat").read_text(encoding="utf-8")
        assert template_servo == _SERVO_DLL

    def test_override_confines_servodyn_to_the_case(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With an override, a shared ServoDyn outside the case copy is rejected."""
        from vane.run import auto_sweep as auto_module

        shared = tmp_path / "shared"
        shared.mkdir()
        (shared / "SD.dat").write_text(_SERVO_DLL, encoding="utf-8")
        fst = _make_controlled_template(
            tmp_path, servo_ref=f'"{(shared / "SD.dat").as_posix()}"'
        )
        library = tmp_path / "libdiscon.dll"
        library.write_bytes(b"MZ")
        settled: list[float] = []

        def fake_settle(deck_path: Path, wind_speed: float, **_: object) -> None:
            settled.append(wind_speed)

        monkeypatch.setattr(auto_module, "solve_steady_state", fake_settle)
        with pytest.raises(FastInputError, match="ServoFile"):
            run_operating_sweep(
                fst,
                tmp_path / "out",
                [8.0],
                settle=True,
                controller_library=library,
                settings=LinearizationSettings(rated_wind_speed=11.4),
                max_workers=1,
                validate=False,
            )
        assert settled == []
        assert (shared / "SD.dat").read_text(encoding="utf-8") == _SERVO_DLL

    def test_shared_servodyn_allowed_without_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without an override ServoDyn is only read, so it may stay shared."""
        from vane.run import auto_sweep as auto_module
        from vane.run import driver as driver_module

        shared = tmp_path / "shared"
        shared.mkdir()
        servo = _SERVO_DLL.replace(
            "/home/runner/miniconda3/envs/test/lib/libdiscon.so", "libdiscon.dll"
        )
        (shared / "SD.dat").write_text(servo, encoding="utf-8")
        (shared / "libdiscon.dll").write_bytes(b"MZ")
        fst = _make_controlled_template(
            tmp_path, servo_ref=f'"{(shared / "SD.dat").as_posix()}"'
        )
        received: list[object] = []

        def fake_settle(
            deck_path: Path, wind_speed: float, **kwargs: object
        ) -> OperatingPoint:
            received.append(kwargs["controller_library"])
            return OperatingPoint(wind_speed, 7.5, 10.0, name=f"ws{wind_speed:g}")

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            return OpenFastResult(
                fst_path=deck_path,
                returncode=0,
                lin_files=[deck_path.with_suffix(".1.lin")],
            )

        monkeypatch.setattr(auto_module, "solve_steady_state", fake_settle)
        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        run_operating_sweep(
            fst,
            tmp_path / "out",
            [8.0],
            settle=True,
            settings=LinearizationSettings(rated_wind_speed=11.4),
            max_workers=1,
            validate=False,
        )
        assert received == [None]

    def test_override_requires_settle(self, tmp_path: Path) -> None:
        """controller_library without settle=True is rejected before launch."""
        fst = _make_template(tmp_path)
        library = tmp_path / "libdiscon.dll"
        library.write_bytes(b"MZ")
        with pytest.raises(ValueError, match="only to settle=True"):
            run_operating_sweep(
                fst,
                tmp_path / "out",
                [8.0],
                curve_wind=[5.0, 12.0],
                curve_rotor_speed=[6.0, 12.0],
                curve_pitch=[0.0, 4.0],
                controller_library=library,
                settings=LinearizationSettings(rated_wind_speed=11.4),
                max_workers=1,
                validate=False,
            )


class TestCaseErrorsInOperatingSweep:
    """An unexpected exception in one linearization case never loses the sweep."""

    @staticmethod
    def _fake_run(deck_path: Path, **_: object) -> OpenFastResult:
        if deck_path.parent.name == "ws12":
            msg = "[WinError 5] Access is denied"
            raise PermissionError(msg)
        # A rotating CalcSteady point requests 36 azimuths.
        lins = [deck_path.with_suffix(f".{index}.lin") for index in range(1, 37)]
        for lin in lins:
            lin.write_text("lin", encoding="utf-8")
        return OpenFastResult(fst_path=deck_path, returncode=0, lin_files=lins)

    def _sweep(
        self, tmp_path: Path, *, strict_cases: bool = False
    ) -> OperatingSweepResult:
        return run_operating_sweep(
            _make_template(tmp_path),
            tmp_path / "out",
            [8.0, 12.0, 14.0],
            curve_wind=[8.0, 14.0],
            curve_rotor_speed=[9.0, 12.0],
            curve_pitch=[0.0, 8.0],
            settings=LinearizationSettings(rated_wind_speed=11.4),
            max_workers=3,
            validate=False,
            strict_cases=strict_cases,
        )

    def test_failed_case_is_recorded_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The result comes back with the failed case recorded as case_error."""
        from vane.run import driver as driver_module

        monkeypatch.setattr(driver_module, "run_openfast", self._fake_run)
        result = self._sweep(tmp_path)

        assert len(result.cases) == 3
        modes = [case.failure_mode for case in result.cases]
        assert modes == [None, "case_error", None]
        assert [case.converged for case in result.cases] == [True, False, True]

    def test_strict_cases_raises_with_the_full_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """strict_cases=True raises SweepCaseError carrying the whole result."""
        from vane.run import SweepCaseError
        from vane.run import driver as driver_module

        monkeypatch.setattr(driver_module, "run_openfast", self._fake_run)
        with pytest.raises(SweepCaseError, match="Access is denied") as excinfo:
            self._sweep(tmp_path, strict_cases=True)

        error = excinfo.value
        assert isinstance(error.__cause__, PermissionError)
        assert error.result is not None
        assert len(error.result.cases) == 3
        assert error.result.cases[0].converged
        assert error.result.cases[2].converged
        assert error.cases == error.result.cases


def test_operating_sweep_passes_the_retry_policy_to_the_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``retry_policy`` reaches the linearization driver unchanged."""
    from vane.run import LinearizationRetryPolicy
    from vane.run import auto_sweep as auto_sweep_module

    seen: list[object] = []

    class _StopError(Exception):
        pass

    real_driver = auto_sweep_module.LinearizationDriver

    def capture(*args: object, **kwargs: object) -> object:
        seen.append(kwargs.get("retry_policy"))
        raise _StopError

    monkeypatch.setattr(auto_sweep_module, "LinearizationDriver", capture)
    policy = LinearizationRetryPolicy(max_sim_time=7200.0)
    fst = _make_template(tmp_path)
    with pytest.raises(_StopError):
        auto_sweep_module.run_operating_sweep(
            fst,
            tmp_path / "out",
            [8.0],
            curve_wind=[4.0, 12.0],
            curve_rotor_speed=[7.0, 12.0],
            curve_pitch=[0.0, 0.0],
            settings=LinearizationSettings(rated_wind_speed=11.4),
            retry_policy=policy,
            validate=False,
        )
    assert seen == [policy]
    assert real_driver is not capture


def test_settle_records_convert_every_non_finite_float_for_strict_json() -> None:
    """Infinity, like NaN, becomes null so a settle failure stays strict JSON."""
    import json
    import math

    from vane.run.auto_sweep import _json_safe

    record = _json_safe({"rms": math.inf, "low": -math.inf, "nan": math.nan, "ok": 1.5})
    assert record == {"rms": None, "low": None, "nan": None, "ok": 1.5}
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize(
    ("winds", "curve_wind", "match"),
    [
        ([14.0], [4.0, 12.0], "outside the operating-curve range"),
        ([8.0], [12.0, 4.0], "strictly increasing"),
    ],
)
def test_bad_settle_curve_fails_before_the_platform_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    winds: list[float],
    curve_wind: list[float],
    match: str,
) -> None:
    """A seed curve that cannot be interpolated fails before OpenFAST is launched."""
    from vane.run import auto_sweep as auto_sweep_module

    def no_probe(*_: object, **__: object) -> None:
        msg = "the platform probe must not run for a malformed request"
        raise AssertionError(msg)

    monkeypatch.setattr(auto_sweep_module, "plan_floating_settle", no_probe)
    with pytest.raises(ValueError, match=match):
        run_operating_sweep(
            _make_template(tmp_path),
            tmp_path / "out",
            winds,
            curve_wind=curve_wind,
            curve_rotor_speed=[7.0, 12.0],
            curve_pitch=[0.0, 0.0],
            settle=True,
            settings=LinearizationSettings(rated_wind_speed=11.4),
            max_workers=1,
            validate=False,
        )
    assert not (tmp_path / "out").exists()


class TestHasRotatingPoint:
    """Which sweeps need a rated wind speed before anything is launched."""

    def test_parked_curve_with_settle_is_not_rotating(self) -> None:
        """A settle prescribing a parked curve's zero speeds linearizes no rotor.

        Settling used to count every positive wind as rotating, so the rated-wind
        check rejected a valid parked sweep before the prescribed-rotor fallback.
        """
        from vane.run.auto_sweep import _has_rotating_point

        parked = {"curve_wind": [4.0, 25.0], "curve_rotor_speed": [0.0, 0.0]}
        assert not _has_rotating_point([8.0, 12.0], settle=True, **parked)
        assert not _has_rotating_point([8.0, 12.0], settle=False, **parked)

    def test_spinning_curve_or_no_curve_is_rotating(self) -> None:
        """A spinning curve, or a settle without a curve, at positive wind rotates."""
        from vane.run.auto_sweep import _has_rotating_point

        spinning = {"curve_wind": [4.0, 25.0], "curve_rotor_speed": [6.0, 12.0]}
        assert _has_rotating_point([8.0], settle=True, **spinning)
        assert _has_rotating_point(
            [8.0], settle=True, curve_wind=None, curve_rotor_speed=None
        )
        assert not _has_rotating_point(
            [0.0], settle=True, curve_wind=None, curve_rotor_speed=None
        )
