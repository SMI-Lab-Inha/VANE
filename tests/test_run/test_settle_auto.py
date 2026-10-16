# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for the automatic settle criterion and the prescribed-rotor settle."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.test_run.test_settle_accelerator import (
    _DAMPING,
    _FakeOpenFast,
    _make_case,
    _out,
)
from vane.io.fst_writer import FastInputFile
from vane.run import steady_state as steady_state_module
from vane.run.runner import OpenFastResult
from vane.run.steady_state import _sustained, solve_steady_state

if TYPE_CHECKING:
    from pathlib import Path

    from vane.run.operating_points import OperatingPoint


def _settle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **spec: float
) -> OperatingPoint:
    """Settle once (adaptively) against a synthetic ``.out`` with ``spec``."""
    fst = _make_case(tmp_path)

    def fake_run(fst_path: Path, **_: object) -> OpenFastResult:
        deck = FastInputFile.read(fst_path)
        out = _out(float(deck.get("TMax")), float(deck.get("TStart")), **spec)
        fst_path.with_suffix(".out").write_text(out, encoding="utf-8")
        return OpenFastResult(fst_path=fst_path, returncode=0)

    monkeypatch.setattr(steady_state_module, "run_openfast", fake_run)
    return solve_steady_state(
        fst,
        12.0,
        settle_time=1000.0,
        discard_time=400.0,
        adaptive=True,
        max_settle_time=1500.0,
    )


class TestAutoCriterion:
    """``criterion="auto"`` (the default) switches only for a sustained oscillation."""

    def test_quiet_platform_is_judged_by_velocity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A platform at rest settles by the velocity criterion; no switch."""
        op = _settle(tmp_path, monkeypatch)
        convergence = op.settle_convergence
        assert convergence is not None
        assert convergence.converged
        assert convergence.criterion == "velocity"
        assert convergence.requested_criterion == "auto"
        assert not convergence.criterion_switched

    def test_sustained_oscillation_switches_to_mean_drift(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A constant-amplitude oscillation is accepted about its steady mean."""
        op = _settle(tmp_path, monkeypatch, amplitude=2.0, period=110.0)
        convergence = op.settle_convergence
        assert convergence is not None
        assert convergence.converged
        assert convergence.criterion == "mean_drift"
        assert convergence.criterion_switched
        assert convergence.iterations == 1
        assert convergence.attempts[0].criterion == "mean_drift"
        assert convergence.residual_damping_ratio == pytest.approx(0.0, abs=2e-3)
        assert convergence.oscillation_period == pytest.approx(110.0, rel=0.1)
        assert op.platform is not None
        assert op.platform.surge == pytest.approx(17.0, abs=0.05)
        assert "sustained oscillation" in caplog.text

    def test_decaying_transient_keeps_the_velocity_criterion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A transient still ringing down is not accepted by its mean."""
        op = _settle(
            tmp_path, monkeypatch, amplitude=4.0, period=100.0, damping_ratio=0.01
        )
        convergence = op.settle_convergence
        assert convergence is not None
        assert not convergence.converged
        assert not convergence.criterion_switched
        assert [attempt.criterion for attempt in convergence.attempts] == [
            "velocity",
            "velocity",
        ]
        assert convergence.residual_damping_ratio == pytest.approx(0.01, rel=0.3)

    def test_no_switch_without_platform_positions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without position channels the mean-drift switch could not see the platform.

        The drift test would then judge only rotor speed and pitch and accept a
        drifting platform, so the sustained oscillation keeps the velocity outcome.
        """
        fst = _make_case(tmp_path)
        positions = {"PtfmSurge", "PtfmSway", "PtfmHeave", "PtfmRoll", "PtfmPitch"}
        positions.add("PtfmYaw")

        def fake_run(fst_path: Path, **_: object) -> OpenFastResult:
            deck = FastInputFile.read(fst_path)
            text = _out(
                float(deck.get("TMax")),
                float(deck.get("TStart")),
                amplitude=2.0,
                period=110.0,
            )
            lines = text.splitlines()
            keep = [i for i, n in enumerate(lines[2].split("\t")) if n not in positions]
            table = ["\t".join(line.split("\t")[i] for i in keep) for line in lines[2:]]
            fst_path.with_suffix(".out").write_text(
                "\n".join([*lines[:2], *table]) + "\n", encoding="utf-8"
            )
            return OpenFastResult(fst_path=fst_path, returncode=0)

        monkeypatch.setattr(steady_state_module, "run_openfast", fake_run)
        op = solve_steady_state(
            fst,
            12.0,
            settle_time=1000.0,
            discard_time=400.0,
            adaptive=True,
            max_settle_time=1500.0,
        )
        convergence = op.settle_convergence
        assert convergence is not None
        assert not convergence.criterion_switched
        assert convergence.criterion == "velocity"
        assert not convergence.converged

    def test_settle_requests_the_platform_positions(self) -> None:
        """The settle adds the position channels its outcome and offset need."""
        for channel in ("PtfmSurge", "PtfmPitch", "PtfmYaw"):
            assert channel in steady_state_module._SETTLE_CHANNELS

    def test_growing_oscillation_is_never_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A single-stage settle never accepts a growing oscillation.

        Without a damped stage there is no equilibrium to confirm, so "auto" keeps
        the velocity criterion; only a two-stage confirmation can accept one.
        """
        op = _settle(
            tmp_path, monkeypatch, amplitude=0.5, period=100.0, damping_ratio=-0.01
        )
        convergence = op.settle_convergence
        assert convergence is not None
        assert not convergence.converged
        assert convergence.criterion == "velocity"

    def test_drifting_mean_fails_after_the_switch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Switching criteria does not accept a mean that still creeps."""
        op = _settle(tmp_path, monkeypatch, amplitude=2.0, period=100.0, drift=0.002)
        convergence = op.settle_convergence
        assert convergence is not None
        assert convergence.criterion_switched
        assert not convergence.converged

    def test_two_stage_confirmation_with_sustained_oscillation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The undamped confirmation of a limit cycle is judged by its mean."""
        fst = _make_case(tmp_path)
        fake = _FakeOpenFast(
            damped={"surge": 17.0},
            confirm={"surge": 17.0, "amplitude": 2.0, "period": 100.0},
        )
        monkeypatch.setattr(steady_state_module, "run_openfast", fake)
        op = solve_steady_state(
            fst,
            12.0,
            settle_time=600.0,
            discard_time=300.0,
            platform_damping=_DAMPING,
            confirm_time=800.0,
            require_settled=True,
        )
        convergence = op.settle_convergence
        assert convergence is not None
        assert convergence.converged
        assert convergence.criterion == "mean_drift"
        assert convergence.criterion_switched
        assert convergence.requested_criterion == "auto"
        assert [attempt.criterion for attempt in convergence.attempts] == [
            "velocity",
            "mean_drift",
        ]
        assert convergence.confirmation is not None
        assert convergence.confirmation.confirmed
        # Accepted with a non-decaying oscillation: flagged as such.
        assert convergence.sustained_oscillation
        assert convergence.confirmation.sustained_oscillation

    @pytest.mark.parametrize(
        ("residual", "expected"),
        [(None, False), (0.0, True), (0.004, True), (0.006, False), (-0.006, False)],
    )
    def test_sustained(self, residual: float | None, expected: bool) -> None:
        """Sustained means neither decaying nor growing beyond the threshold."""
        assert _sustained(residual, 0.005) is expected


class TestPrescribedRotor:
    """Settling without a controller at a prescribed rotor speed and pitch."""

    def _controlled_case(self, tmp_path: Path) -> Path:
        """Write a settle deck whose ServoDyn runs a library missing here."""
        fst = _make_case(tmp_path)
        (tmp_path / "SD.dat").write_text(
            '5 PCMode - pitch\n"/home/x/libdiscon.so" DLL_FileName - library\n',
            encoding="utf-8",
        )
        text = fst.read_text(encoding="utf-8")
        fst.write_text(
            text + '1 CompServo - servo\n"SD.dat" ServoFile - ServoDyn\n',
            encoding="utf-8",
        )
        elastodyn = tmp_path / "ED.dat"
        elastodyn.write_text(
            "True GenDOF - gen\nTrue DrTrDOF - drivetrain\n"
            + elastodyn.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        return fst

    def test_runs_without_the_controller(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ServoDyn is off, the rotor locked, and the prescribed values returned."""
        fst = self._controlled_case(tmp_path)
        seen: dict[str, str] = {}

        def fake_run(fst_path: Path, **_: object) -> OpenFastResult:
            deck = FastInputFile.read(fst_path)
            elastodyn = FastInputFile.read(fst_path.parent / "ED.dat")
            seen.update(
                comp_servo=deck.get("CompServo"),
                gen=elastodyn.get("GenDOF"),
                drivetrain=elastodyn.get("DrTrDOF"),
                rotor=elastodyn.get("RotSpeed"),
            )
            out = _out(float(deck.get("TMax")), float(deck.get("TStart")))
            fst_path.with_suffix(".out").write_text(out, encoding="utf-8")
            return OpenFastResult(fst_path=fst_path, returncode=0)

        monkeypatch.setattr(steady_state_module, "run_openfast", fake_run)
        op = solve_steady_state(
            fst,
            12.0,
            settle_time=600.0,
            discard_time=300.0,
            rotor_speed_rpm=7.56,
            blade_pitch_deg=3.4,
            prescribed_rotor=True,
            require_settled=True,
        )
        assert seen == {
            "comp_servo": "0",
            "gen": "False",
            "drivetrain": "False",
            "rotor": "7.56",
        }
        assert op.rotor_speed_rpm == 7.56
        assert op.blade_pitch_deg == 3.4
        assert op.settle_convergence is not None
        assert op.settle_convergence.prescribed_rotor
        assert op.platform is not None
        assert op.platform.surge == pytest.approx(17.0)

    def test_rejects_a_controller_library(self, tmp_path: Path) -> None:
        """A prescribed rotor has no controller to give a library to."""
        fst = _make_case(tmp_path)
        library = tmp_path / "libdiscon.dll"
        library.write_bytes(b"MZ")
        with pytest.raises(ValueError, match="prescribed_rotor"):
            solve_steady_state(
                fst, 12.0, prescribed_rotor=True, controller_library=library
            )

    def test_unknown_criterion_lists_auto(self, tmp_path: Path) -> None:
        """The criterion error names every accepted value."""
        with pytest.raises(ValueError, match="'auto'"):
            solve_steady_state(
                _make_case(tmp_path),
                12.0,
                criterion="bogus",  # type: ignore[arg-type]
            )
