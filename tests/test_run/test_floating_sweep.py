# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for the automatic floating settle setup inside run_operating_sweep."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tests.test_run.test_auto_sweep import (  # noqa: F401  (autouse fixture)
    _PRIMARY,
    _SERVO_DLL,
    _make_template,
    _stub_executable_selection,
)
from tests.test_run.test_settle_accelerator import _HD, _out
from vane.io.fst_writer import FastInputFile
from vane.run import LinearizationSettings, OperatingPoint, run_operating_sweep
from vane.run import auto_sweep as auto_module
from vane.run import driver as driver_module
from vane.run import floating_auto as floating_module
from vane.run import steady_state as steady_state_module
from vane.run.floating_auto import PlatformProbe
from vane.run.runner import OpenFastResult
from vane.run.steady_state import SettlePlatformDamping, SteadyStateError

if TYPE_CHECKING:
    from pathlib import Path

_CURVE: dict[str, Any] = {
    "curve_wind": [5.0, 20.0],
    "curve_rotor_speed": [6.0, 7.5],
    "curve_pitch": [0.0, 15.0],
}

_ED_EXTRA = """\
0 PtfmSurge - m
0 PtfmHeave - m
0 PtfmPitch - deg
True PtfmSgDOF - platform surge DOF
True PtfmPDOF - platform pitch DOF
"""


def _template(tmp_path: Path, *, floating: bool = False) -> Path:
    """Write a settle template whose ServoDyn names a Linux library."""
    fst = _make_template(tmp_path)
    root = fst.parent
    (root / "SD.dat").write_text(_SERVO_DLL, encoding="utf-8")
    primary = (
        "600 TMax - run time\n0 TStart - output start\n"
        + _PRIMARY
        + '1 CompServo - servo\n"SD.dat" ServoFile - ServoDyn\n'
    )
    if floating:
        (root / "HD.dat").write_text(_HD, encoding="utf-8")
        ed = root / "ED.dat"
        ed.write_text(ed.read_text(encoding="utf-8") + _ED_EXTRA, encoding="utf-8")
        primary += (
            '1 CompHydro - hydro\n"HD.dat" HydroFile - HydroDyn\n'
            "0 CompSub - sub\n3 CompMooring - mooring\n"
        )
    fst.write_text(primary, encoding="utf-8")
    return fst


def _fake_runs(
    monkeypatch: pytest.MonkeyPatch, **spec: float
) -> tuple[list[dict[str, str]], list[float]]:
    """Fake settle and linearization runs; return what each saw."""
    settles: list[dict[str, str]] = []
    linearized: list[float] = []

    def settle_run(fst_path: Path, **_: object) -> OpenFastResult:
        deck = FastInputFile.read(fst_path)
        servo = FastInputFile.read(fst_path.parent / "SD.dat")
        elastodyn = FastInputFile.read(fst_path.parent / "ED.dat")
        settles.append(
            {
                "comp_servo": deck.get("CompServo"),
                "library": servo.get("DLL_FileName"),
                "rotor": elastodyn.get("RotSpeed"),
                "pitch": elastodyn.get("BlPitch(1)"),
            }
        )
        out = _out(float(deck.get("TMax")), float(deck.get("TStart")), **spec)
        fst_path.with_suffix(".out").write_text(out, encoding="utf-8")
        return OpenFastResult(fst_path=fst_path, returncode=0)

    def lin_run(deck_path: Path, **_: object) -> OpenFastResult:
        inflow = FastInputFile.read(deck_path.parent / "IW.dat")
        linearized.append(float(inflow.get("HWindSpeed")))
        return OpenFastResult(
            fst_path=deck_path,
            returncode=0,
            lin_files=[deck_path.with_suffix(f".{i}.lin") for i in range(1, 37)],
        )

    monkeypatch.setattr(steady_state_module, "run_openfast", settle_run)
    monkeypatch.setattr(driver_module, "run_openfast", lin_run)
    return settles, linearized


def _sweep(fst: Path, tmp_path: Path, **kwargs: object) -> Any:  # noqa: ANN401
    options: dict[str, Any] = {
        "settle": True,
        "settle_time": 600.0,
        "discard_time": 300.0,
        "settings": LinearizationSettings(rated_wind_speed=10.6),
        "max_workers": 1,
        "validate": False,
        **kwargs,
    }
    return run_operating_sweep(fst, tmp_path / "out", [12.0], **options)


class TestControllerFallback:
    """No loadable controller library: the rotor is prescribed from the curve."""

    def test_prescribed_settle_then_linearization(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The settle runs without ServoDyn at the curve point; it is linearized."""
        fst = _template(tmp_path)
        settles, linearized = _fake_runs(monkeypatch)
        result = _sweep(fst, tmp_path, **_CURVE)
        plan = result.floating_plan
        assert plan is not None
        assert plan.prescribed_rotor
        assert plan.controller.status == "missing"
        assert settles[0]["comp_servo"] == "0"
        # The deck's own library path is left alone (ServoDyn is off).
        assert "libdiscon.so" in settles[0]["library"]
        point = result.operating_points[0]
        assert point.rotor_speed_rpm == pytest.approx(6.7)
        assert point.blade_pitch_deg == pytest.approx(7.0)
        assert point.settle_convergence.prescribed_rotor
        assert point.platform.surge == pytest.approx(17.0)
        assert linearized == [12.0]
        assert result.settle_failures == ()

    def test_unsettled_seed_is_still_linearized(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A prescribed settle only seeds the trim, so it is linearized anyway."""
        fst = _template(tmp_path)
        _, linearized = _fake_runs(monkeypatch, drift=0.05)
        result = _sweep(fst, tmp_path, **_CURVE)
        assert linearized == [12.0]
        assert len(result.settle_failures) == 1
        assert result.settle_failures[0].linearized

    def test_unsettled_parked_seed_is_not_linearized(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A parked prescribed point gets no trim, so it must have settled.

        Every prescribed point used to be exempt from require_settled, so an
        unsettled parked floater was linearized at its last transient mean.
        """
        fst = _template(tmp_path)
        _, linearized = _fake_runs(monkeypatch, drift=0.05)
        parked = {
            **_CURVE,
            "curve_rotor_speed": [0.0, 0.0],
            "curve_pitch": [90.0, 90.0],
        }
        result = _sweep(fst, tmp_path, **parked)
        assert linearized == []
        assert len(result.settle_failures) == 1
        assert not result.settle_failures[0].linearized

    def test_without_curve_fails_before_any_settle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a curve there is nothing to prescribe; fail before launch."""
        fst = _template(tmp_path)
        settles, _ = _fake_runs(monkeypatch)
        with pytest.raises(SteadyStateError, match="no operating curve"):
            _sweep(fst, tmp_path)
        assert settles == []

    def test_library_found_in_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A host build on VANE_CONTROLLER_LIBRARIES runs the real controller."""
        fst = _template(tmp_path)
        builds = tmp_path / "builds"
        builds.mkdir()
        suffix = floating_module._host_library_suffix()
        library = builds / f"libdiscon{suffix}"
        library.write_bytes(b"MZ")
        monkeypatch.setenv("VANE_CONTROLLER_LIBRARIES", str(builds))
        settles, _ = _fake_runs(monkeypatch)
        result = _sweep(fst, tmp_path, **_CURVE)
        assert not result.floating_plan.prescribed_rotor
        assert result.floating_plan.controller.status == "found"
        assert settles[0]["comp_servo"] == "1"
        assert library.name in settles[0]["library"]
        assert not result.operating_points[0].settle_convergence.prescribed_rotor


class TestAutomaticDamping:
    """A floating deck gets the settle damping sized from the platform probe."""

    _PROBE = PlatformProbe(
        effective_mass={"surge": 3.0e7, "pitch": 5.0e10},
        natural_period={"surge": 125.0, "pitch": 28.0},
        damping_ratio={"surge": 0.03, "pitch": 0.025},
        mode_frequency={"surge": 0.008, "pitch": 0.0357},
        probe_damping=SettlePlatformDamping(surge=1e6, pitch=1e9),
        n_states=14,
    )

    def _capture(
        self, monkeypatch: pytest.MonkeyPatch, probes: list[Path]
    ) -> list[dict[str, object]]:
        calls: list[dict[str, object]] = []

        def fake_probe(fst_path: Path, work_dir: Path, **_: object) -> PlatformProbe:
            probes.append(work_dir)
            return self._PROBE

        def fake_settle(
            deck: Path, wind_speed: float, **kwargs: object
        ) -> OperatingPoint:
            calls.append(kwargs)
            return OperatingPoint(wind_speed, 7.2, 8.0)

        monkeypatch.setattr(floating_module, "probe_platform", fake_probe)
        monkeypatch.setattr(auto_module, "solve_steady_state", fake_settle)
        monkeypatch.setattr(
            driver_module,
            "run_openfast",
            lambda deck_path, **_: OpenFastResult(fst_path=deck_path, returncode=0),
        )
        return calls

    def test_probe_sizes_the_damping(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The probe runs once and its damping reaches every settle."""
        fst = _template(tmp_path, floating=True)
        probes: list[Path] = []
        calls = self._capture(monkeypatch, probes)
        result = _sweep(fst, tmp_path, **_CURVE)
        assert probes == [tmp_path / "out" / "probe"]
        damping = calls[0]["platform_damping"]
        assert damping == self._PROBE.settle_damping(0.5)
        assert calls[0]["criterion"] == "auto"
        assert calls[0]["prescribed_rotor"] is True
        assert result.floating_plan.damping_source == "probe"
        assert result.floating_plan.to_dict()["probe"]["n_states"] == 14

    def test_probe_period_sizes_the_confirmation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The probe's slowest period reaches every settle; the length is automatic."""
        fst = _template(tmp_path, floating=True)
        probes: list[Path] = []
        calls = self._capture(monkeypatch, probes)
        _sweep(fst, tmp_path, **_CURVE)
        assert calls[0]["platform_period"] == 125.0
        assert calls[0]["confirm_time"] is None

    def test_probe_period_sizes_the_linearization_trim(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each case's first trim runs 20 slowest platform periods (125 s -> 2500 s)."""
        fst = _template(tmp_path, floating=True)
        probes: list[Path] = []
        self._capture(monkeypatch, probes)
        tmax: list[float] = []

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            tmax.append(float(FastInputFile.read(deck_path).get("TMax")))
            return OpenFastResult(fst_path=deck_path, returncode=0)

        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        result = _sweep(fst, tmp_path, **_CURVE)
        assert tmax[0] == pytest.approx(2500.0)
        record = result.auto_configuration.get("sim_time")
        assert record is not None
        assert record.value == pytest.approx(2500.0)
        assert record.source == "platform probe"

    def test_explicit_sim_time_is_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A sim_time the user set is not replaced by the probe-sized one."""
        fst = _template(tmp_path, floating=True)
        probes: list[Path] = []
        self._capture(monkeypatch, probes)
        tmax: list[float] = []

        def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
            tmax.append(float(FastInputFile.read(deck_path).get("TMax")))
            return OpenFastResult(fst_path=deck_path, returncode=0)

        monkeypatch.setattr(driver_module, "run_openfast", fake_run)
        settings = LinearizationSettings(rated_wind_speed=10.6, sim_time=900.0)
        result = _sweep(fst, tmp_path, settings=settings, **_CURVE)
        assert tmax[0] == pytest.approx(900.0)
        assert result.auto_configuration.get("sim_time") is None

    @pytest.mark.parametrize(
        "bad",
        [
            {"discard_time": 600.0},
            {"settle_criterion": "median"},
            {"settle_confirm_time": -1.0},
            {"max_settle_time": 100.0},
        ],
    )
    def test_bad_settle_schedule_fails_before_the_probe(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        bad: dict[str, object],
    ) -> None:
        """An invalid settle schedule is rejected before OpenFAST is launched.

        It used to be checked inside every settle worker, after the platform
        probe had already run OpenFAST twice.
        """
        fst = _template(tmp_path, floating=True)
        probes: list[Path] = []
        self._capture(monkeypatch, probes)
        with pytest.raises(ValueError, match="must"):
            _sweep(fst, tmp_path, **{**_CURVE, **bad})
        assert probes == []

    def test_opt_out(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """settle_auto_damping=False settles undamped and never probes."""
        fst = _template(tmp_path, floating=True)
        probes: list[Path] = []
        calls = self._capture(monkeypatch, probes)
        _sweep(fst, tmp_path, settle_auto_damping=False, **_CURVE)
        assert probes == []
        assert calls[0]["platform_damping"] is None

    def test_no_plan_without_settle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A curve-only sweep makes no settle plan."""
        fst = _template(tmp_path, floating=True)
        probes: list[Path] = []
        self._capture(monkeypatch, probes)
        result = run_operating_sweep(
            fst,
            tmp_path / "out",
            [12.0],
            settings=LinearizationSettings(rated_wind_speed=10.6),
            max_workers=1,
            validate=False,
            **_CURVE,
        )
        assert result.floating_plan is None
        assert probes == []


class TestFloatingTrimTime:
    """The CalcSteady run time of a floating linearization case."""

    def test_twenty_periods(self) -> None:
        """The IEA 15-MW semi's 125.5 s slowest period gives 2510 s."""
        assert auto_module.floating_trim_time(125.5, 3600.0) == pytest.approx(2510.0)

    def test_bounds(self) -> None:
        """Never below the CalcSteady floor, never beyond the ladder's ceiling."""
        assert auto_module.floating_trim_time(10.0, 3600.0) == pytest.approx(600.0)
        assert auto_module.floating_trim_time(400.0, 3600.0) == pytest.approx(3600.0)
        assert auto_module.floating_trim_time(400.0, 100.0) == pytest.approx(600.0)

    @pytest.mark.parametrize(
        ("period", "ceiling"),
        [(0.0, 3600.0), (-1.0, 3600.0), (float("nan"), 3600.0), (100.0, float("inf"))],
    )
    def test_invalid(self, period: float, ceiling: float) -> None:
        """Non-finite or non-positive inputs are rejected."""
        with pytest.raises(ValueError, match="finite and positive"):
            auto_module.floating_trim_time(period, ceiling)
