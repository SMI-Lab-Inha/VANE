# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Regression tests for the untuned floating settle (IEA 15-MW semi, 8-14 m/s).

The synthetic time series reproduce the settle records of an untouched sweep of the
IEA 15-MW VolturnUS-S semi-submersible: a damped stage that settled, followed by an
undamped confirmation that was

* still ringing down at 8 m/s (residual damping ratio +0.067, period 106 s), with a
  slow controller start-up transient in the blade pitch: over a 600 s confirmation
  its means read a false equilibrium shift (surge -0.11 m, pitch +0.10 deg);
* keeping a bounded 106 s oscillation at 12 m/s (residual damping ratio -0.007)
  about the damped equilibrium (shift -0.06 m), rejected as "sustained".
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.test_run.test_settle_accelerator import (
    _DAMPING,
    _damping_diagonal,
    _make_case,
    _out,
)
from vane.io.fst_writer import FastInputFile
from vane.run import (
    EquilibriumTolerances,
    PlatformProbe,
    SettlePlatformDamping,
    SteadyStateError,
    solve_steady_state,
)
from vane.run import steady_state as steady_state_module
from vane.run.auto_sweep import SettleFailure
from vane.run.runner import OpenFastResult
from vane.run.steady_state import _confirm_length, _ConfirmPlan

if TYPE_CHECKING:
    from pathlib import Path

    from vane.run.operating_points import OperatingPoint

_CHANNELS = (
    "Time",
    "RotSpeed",
    "BldPitch1",
    "PtfmSurge",
    "PtfmHeave",
    "PtfmPitch",
    "TTDspFA",
    "OoPDefl1",
    "IPDefl1",
    "PtfmTVxi",
    "PtfmRVyi",
)

# The 8 m/s point: damped-stage equilibrium and the undamped restart transient.
_EQ_8 = {"surge": 15.629, "pitch_deg": 0.00104, "rotor": 5.705, "ptfm_pitch": 2.596}
_RESTART_8 = {
    "period": 106.1,
    "zeta": 0.068,  # the record's residual damping ratio
    "amplitude": 3.1,  # surge envelope at the restart (1.5 m by t = 180 s)
    "offset": 6.1,  # surge offset of the slow transient (about -0.5 m at 200 s)
    "pitch": 5.0,  # blade-pitch transient (0.77 deg at 150 s, as recorded)
    "tau": 80.0,  # time constant of the slow controller transient (s)
}
# The 12 m/s point: damped-stage equilibrium and the sustained undamped oscillation.
_EQ_12 = {"surge": 17.770, "pitch_deg": 6.3325, "rotor": 7.560, "ptfm_pitch": 3.368}
# The IEA 22-MW semi at 12 m/s: damped-stage equilibrium; its undamped 1200 s
# confirmation grew (residual damping ratio -0.026, period 94 s, velocity RMS
# 0.061 m/s) about a reproduced mean (surge shift -0.017 m).
_EQ_22 = {"surge": 14.830, "pitch_deg": 7.4368, "rotor": 7.0611, "ptfm_pitch": 4.488}


def _row(
    time: float,
    rotor: float,
    pitch_deg: float,
    surge: float,
    ptfm_pitch: float,
    surge_velocity: float,
) -> str:
    values = (time, rotor, pitch_deg, surge, -0.46, ptfm_pitch, 0.9, 10.1, -1.2)
    values = (*values, surge_velocity, 0.0)
    return "\t".join(f"{value:.6f}" for value in values)


def _header() -> list[str]:
    return [
        "OpenFAST results header line",
        "",
        "\t".join(_CHANNELS),
        "\t".join("(-)" for _ in _CHANNELS),
    ]


def _quiet_out(tmax: float, tstart: float, eq: dict[str, float]) -> str:
    """Return a platform at rest at ``eq`` (a converged damped stage)."""
    lines = _header()
    for step in range(int(tstart), int(tmax) + 1):
        lines.append(
            _row(
                float(step),
                eq["rotor"],
                eq["pitch_deg"],
                eq["surge"],
                eq["ptfm_pitch"],
                0.0,
            )
        )
    return "\n".join(lines) + "\n"


def _decaying_out(tmax: float, tstart: float) -> str:
    """Return the 8 m/s restart: a decaying surge oscillation and a slow transient.

    ``surge = eq - c exp(-t/tau) + A exp(-zeta w t) sin(w t)`` and
    ``pitch = eq + p exp(-t/tau)``: both return to the damped equilibrium.
    """
    spec = _RESTART_8
    omega = 2.0 * math.pi / spec["period"]
    rate = spec["zeta"] * omega
    lines = _header()
    for step in range(int(tstart), int(tmax) + 1):
        t = float(step)
        slow = math.exp(-t / spec["tau"])
        envelope = spec["amplitude"] * math.exp(-rate * t)
        surge = _EQ_8["surge"] - spec["offset"] * slow + envelope * math.sin(omega * t)
        velocity = spec["offset"] / spec["tau"] * slow + envelope * (
            omega * math.cos(omega * t) - rate * math.sin(omega * t)
        )
        pitch = _EQ_8["pitch_deg"] + spec["pitch"] * slow
        lines.append(
            _row(t, _EQ_8["rotor"], pitch, surge, _EQ_8["ptfm_pitch"], velocity)
        )
    return "\n".join(lines) + "\n"


def _oscillating_out(
    tmax: float,
    tstart: float,
    *,
    amplitude: float,
    zeta: float,
    mean_surge: float = 17.711,
    period: float = 106.1,
    eq: dict[str, float] | None = None,
) -> str:
    """Return the 12 m/s run: an oscillation about (nearly) the damped mean."""
    eq = _EQ_12 if eq is None else eq
    omega = 2.0 * math.pi / period
    rate = zeta * omega
    lines = _header()
    for step in range(int(tstart), int(tmax) + 1):
        t = float(step)
        envelope = amplitude * math.exp(-rate * t)
        surge = mean_surge + envelope * math.sin(omega * t)
        velocity = envelope * (omega * math.cos(omega * t) - rate * math.sin(omega * t))
        pitch = eq["pitch_deg"] - 0.004 + 0.1 * envelope * math.sin(omega * t)
        lines.append(_row(t, eq["rotor"], pitch, surge, eq["ptfm_pitch"], velocity))
    return "\n".join(lines) + "\n"


class _Recorded:
    """Fake OpenFAST: a quiet damped stage, then ``confirm(tmax, tstart)``."""

    def __init__(self, eq: dict[str, float], confirm: object) -> None:
        self.eq = eq
        self.confirm = confirm
        self.launches: list[tuple[bool, float]] = []

    def __call__(self, fst_path: Path, **_: object) -> OpenFastResult:
        deck = FastInputFile.read(fst_path)
        tmax = float(deck.get("TMax"))
        tstart = float(deck.get("TStart"))
        damped = _damping_diagonal(fst_path.parent / "HD.dat")[0] > 0.0
        self.launches.append((damped, tmax))
        if damped:
            out = _quiet_out(tmax, tstart, self.eq)
        else:
            out = self.confirm(tmax, tstart)  # type: ignore[operator]
        fst_path.with_suffix(".out").write_text(out, encoding="utf-8")
        return OpenFastResult(fst_path=fst_path, returncode=0)


def _settle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake: _Recorded,
    **kwargs: object,
) -> OperatingPoint:
    fst = _make_case(tmp_path)
    monkeypatch.setattr(steady_state_module, "run_openfast", fake)
    options: dict[str, object] = {
        "settle_time": 1200.0,
        "discard_time": 600.0,
        "rotor_speed_rpm": 6.0,
        "blade_pitch_deg": 0.0,
        "platform_damping": _DAMPING,
        **kwargs,
    }
    return solve_steady_state(fst, 8.0, **options)  # type: ignore[arg-type]


class TestDecayingConfirmation:
    """8 m/s: a confirmation still ringing down is settled, not a false shift."""

    def test_automatic_confirmation_settles(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The automatic length outlasts the transient: the point is settled."""
        # Over the old fixed 600 s confirmation (window 150-600 s) the synthetic
        # restart reproduces the recorded false shift.
        columns, rows = _table(_decaying_out(600.0, 150.0))
        assert rows[:, columns.index("PtfmSurge")].mean() - _EQ_8["surge"] < -0.1
        assert rows[:, columns.index("BldPitch1")].mean() - _EQ_8["pitch_deg"] > 0.1
        caplog.set_level("INFO")
        fake = _Recorded(_EQ_8, _decaying_out)
        op = _settle(tmp_path, monkeypatch, fake, require_settled=True)
        convergence = op.settle_convergence
        assert convergence is not None
        assert convergence.converged
        confirmation = convergence.confirmation
        assert confirmation is not None
        # No platform period was known and the damped stage was quiet: fallback.
        assert confirmation.confirm_time == 1200.0
        assert confirmation.confirm_time_source == "fallback"
        assert fake.launches == [(True, 1200.0), (False, 1200.0)]
        assert confirmation.shift_flagged == ()
        assert abs(confirmation.shift["surge"]) < 0.05
        assert abs(confirmation.shift["blade_pitch_deg"]) < 0.02
        assert confirmation.residual_damping_ratio == pytest.approx(0.068, rel=0.3)
        assert not confirmation.extended
        assert not confirmation.sustained_oscillation
        assert op.platform is not None
        assert op.platform.surge == pytest.approx(_EQ_8["surge"], abs=1e-3)
        assert "still ringing down" in caplog.text

    def test_short_confirmation_is_extended_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 600 s confirmation flags the shift, so it is rerun twice as long."""
        fake = _Recorded(_EQ_8, _decaying_out)
        op = _settle(
            tmp_path, monkeypatch, fake, confirm_time=600.0, require_settled=True
        )
        convergence = op.settle_convergence
        assert convergence is not None
        assert convergence.converged
        assert fake.launches == [(True, 1200.0), (False, 600.0), (False, 1200.0)]
        assert [attempt.stage for attempt in convergence.attempts] == [
            "damped",
            "confirm",
            "confirm",
        ]
        assert [attempt.iteration for attempt in convergence.attempts] == [1, 2, 3]
        confirmation = convergence.confirmation
        assert confirmation is not None
        assert confirmation.extended
        assert confirmation.confirm_time == 1200.0
        assert confirmation.confirm_time_source == "user"
        assert confirmation.shift_flagged == ()

    def test_extension_that_still_shifts_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A decaying run whose mean really moved fails after its one extension."""

        def moved(tmax: float, tstart: float) -> str:
            return _out(tmax, tstart, surge=16.0, amplitude=1.0, damping_ratio=0.03)

        fake = _Recorded(_EQ_8, moved)
        with pytest.raises(SteadyStateError, match="moved the equilibrium") as error:
            _settle(
                tmp_path, monkeypatch, fake, confirm_time=600.0, require_settled=True
            )
        assert fake.launches == [(True, 1200.0), (False, 600.0), (False, 1200.0)]
        assert "extended to 1200 s" in str(error.value)
        convergence = error.value.convergence
        assert convergence is not None
        assert convergence.confirmation is not None
        assert convergence.confirmation.extended

    def test_extension_respects_max_settle_time(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The extension is capped at the caller's max_settle_time."""
        fake = _Recorded(_EQ_8, _decaying_out)
        op = _settle(
            tmp_path,
            monkeypatch,
            fake,
            confirm_time=600.0,
            adaptive=True,
            max_settle_time=900.0,
            settle_time=600.0,
            discard_time=300.0,
        )
        assert fake.launches == [(True, 600.0), (False, 600.0), (False, 900.0)]
        assert op.settle_convergence is not None
        assert op.settle_convergence.confirmation is not None
        assert op.settle_convergence.confirmation.extended
        assert op.settle_convergence.confirmation.confirm_time == 900.0

    def test_no_extension_at_the_ceiling(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A confirmation already at max_settle_time is judged as it is."""
        fake = _Recorded(_EQ_8, _decaying_out)
        op = _settle(
            tmp_path,
            monkeypatch,
            fake,
            adaptive=True,
            max_settle_time=600.0,
            settle_time=600.0,
            discard_time=300.0,
        )
        # The automatic length (1200 s fallback) is capped at 600 s; no extension.
        assert fake.launches == [(True, 600.0), (False, 600.0)]
        convergence = op.settle_convergence
        assert convergence is not None
        assert not convergence.converged
        assert convergence.confirmation is not None
        assert not convergence.confirmation.extended


class TestSustainedOscillation:
    """12 m/s: a bounded oscillation about a confirmed equilibrium is settled."""

    def test_bounded_oscillation_settles_with_a_warning(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The damped equilibrium is accepted and the oscillation flagged."""

        def record(tmax: float, tstart: float) -> str:
            return _oscillating_out(tmax, tstart, amplitude=1.6, zeta=-0.0067)

        fake = _Recorded(_EQ_12, record)
        op = _settle(tmp_path, monkeypatch, fake, require_settled=True)
        convergence = op.settle_convergence
        assert convergence is not None
        assert convergence.converged
        assert convergence.sustained_oscillation
        confirmation = convergence.confirmation
        assert confirmation is not None
        assert confirmation.sustained_oscillation
        assert confirmation.confirmed
        assert not confirmation.stable
        assert confirmation.shift_flagged == ()
        assert confirmation.residual_damping_ratio == pytest.approx(-0.0067, abs=3e-3)
        assert convergence.translational_velocity_rms > 0.05
        # The linearization point is the damped stage's equilibrium.
        assert op.platform is not None
        assert op.platform.surge == pytest.approx(_EQ_12["surge"], abs=1e-3)
        assert op.blade_pitch_deg == pytest.approx(_EQ_12["pitch_deg"], abs=1e-4)
        assert "settled with a sustained oscillation" in caplog.text
        record_dict = asdict(convergence)
        assert record_dict["sustained_oscillation"] is True
        assert record_dict["confirmation"]["sustained_oscillation"] is True

    @staticmethod
    def _growing_22(amplitude: float = 0.3, mean_surge: float = 14.813) -> _Recorded:
        """Return the IEA 22-MW semi at 12 m/s: growth at -0.026 about the mean."""

        def record(tmax: float, tstart: float) -> str:
            return _oscillating_out(
                tmax,
                tstart,
                amplitude=amplitude,
                zeta=-0.0259,
                mean_surge=mean_surge,
                period=93.6,
                eq=_EQ_22,
            )

        return _Recorded(_EQ_22, record)

    def test_growing_oscillation_with_confirmed_equilibrium_settles(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A growing closed-loop oscillation about a reproduced equilibrium settles.

        Linear damping cannot move an equilibrium, and the shift test reproduced
        it, so the damped point is linearized (the trim runs without the
        controller) and the controller-in-the-loop growth is flagged.
        """
        fake = self._growing_22()
        op = _settle(tmp_path, monkeypatch, fake, require_settled=True)
        convergence = op.settle_convergence
        assert convergence is not None
        assert convergence.converged
        assert convergence.sustained_oscillation
        assert convergence.growing_oscillation
        confirmation = convergence.confirmation
        assert confirmation is not None
        assert confirmation.growing_oscillation
        assert confirmation.shift_flagged == ()
        residual = confirmation.residual_damping_ratio
        assert residual is not None
        assert residual == pytest.approx(-0.0259, abs=4e-3)
        assert 0.03 < convergence.translational_velocity_rms < 0.2
        # The linearization point is the damped stage's equilibrium.
        assert op.platform is not None
        assert op.platform.surge == pytest.approx(_EQ_22["surge"], abs=1e-3)
        assert op.blade_pitch_deg == pytest.approx(_EQ_22["pitch_deg"], abs=1e-4)
        assert "settled with a GROWING oscillation" in caplog.text
        assert "unstable under the deck's controller" in caplog.text
        assert "not part of the linear model" in caplog.text
        # The flag reaches the JSON records.
        record = convergence.to_dict()
        assert record["growing_oscillation"] is True
        confirmation_record = record["confirmation"]
        assert isinstance(confirmation_record, dict)
        assert confirmation_record["growing_oscillation"] is True
        failure = SettleFailure(
            wind_speed=12.0, reason="not_settled", message="", convergence=convergence
        ).to_dict()
        assert failure["convergence"]["growing_oscillation"] is True  # type: ignore[index]

    def test_sustained_oscillation_is_not_flagged_growing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A residual above -growing_damping_ratio is sustained, not growing."""

        def record(tmax: float, tstart: float) -> str:
            return _oscillating_out(tmax, tstart, amplitude=1.6, zeta=-0.0067)

        op = _settle(tmp_path, monkeypatch, _Recorded(_EQ_12, record))
        convergence = op.settle_convergence
        assert convergence is not None
        assert convergence.sustained_oscillation
        assert not convergence.growing_oscillation
        assert convergence.to_dict()["growing_oscillation"] is False

    def test_growing_oscillation_above_the_rms_bound_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Growth is accepted only while its mean is measurable (RMS bound)."""
        # The recorded growth (RMS about 0.06 m/s) against a 0.03 bound: the mean is
        # still reproduced, so the RMS alone blocks the point.
        fake = self._growing_22()
        tolerances = EquilibriumTolerances(oscillation_velocity_rms=0.03)
        with pytest.raises(SteadyStateError) as error:
            _settle(
                tmp_path,
                monkeypatch,
                fake,
                require_settled=True,
                equilibrium_tolerances=tolerances,
            )
        message = str(error.value)
        assert "oscillation is growing" in message
        assert "not accepted because its platform-velocity RMS" in message
        assert "exceeds oscillation_velocity_rms 0.03" in message
        assert "moved the equilibrium" not in message
        convergence = error.value.convergence
        assert convergence is not None
        assert not convergence.growing_oscillation
        assert not convergence.sustained_oscillation

    def test_growing_oscillation_with_failed_shift_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A growing oscillation about a different mean fails on the shift."""
        fake = self._growing_22(mean_surge=14.5)
        with pytest.raises(SteadyStateError) as error:
            _settle(tmp_path, monkeypatch, fake, require_settled=True)
        message = str(error.value)
        assert "moved the equilibrium" in message
        assert "not accepted because the equilibrium moved" in message
        assert "exceeds oscillation_velocity_rms" not in message
        convergence = error.value.convergence
        assert convergence is not None
        assert not convergence.growing_oscillation

    def test_oscillation_above_the_rms_bound_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bounded but large oscillation is not accepted."""

        def record(tmax: float, tstart: float) -> str:
            return _oscillating_out(tmax, tstart, amplitude=1.6, zeta=-0.0067)

        fake = _Recorded(_EQ_12, record)
        tolerances = EquilibriumTolerances(oscillation_velocity_rms=0.02)
        with pytest.raises(SteadyStateError, match="not decaying"):
            _settle(
                tmp_path,
                monkeypatch,
                fake,
                require_settled=True,
                equilibrium_tolerances=tolerances,
            )

    def test_stationary_oscillation_above_the_rms_bound_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The mean-drift switch does not bypass the oscillation bound.

        A stationary oscillation (zero residual damping) makes "auto" switch to
        mean drift, whose steady means used to confirm the point without the RMS
        bound ever being applied.
        """

        def record(tmax: float, tstart: float) -> str:
            return _oscillating_out(tmax, tstart, amplitude=1.6, zeta=0.0)

        fake = _Recorded(_EQ_12, record)
        tolerances = EquilibriumTolerances(oscillation_velocity_rms=0.02)
        with pytest.raises(SteadyStateError):
            _settle(
                tmp_path,
                monkeypatch,
                fake,
                require_settled=True,
                equilibrium_tolerances=tolerances,
            )

    def test_shifted_mean_with_oscillation_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bounded oscillation about a different mean fails the shift test."""

        def record(tmax: float, tstart: float) -> str:
            return _oscillating_out(
                tmax, tstart, amplitude=1.6, zeta=-0.0067, mean_surge=17.4
            )

        fake = _Recorded(_EQ_12, record)
        with pytest.raises(SteadyStateError, match="moved the equilibrium") as error:
            _settle(tmp_path, monkeypatch, fake, require_settled=True)
        assert error.value.convergence is not None
        assert not error.value.convergence.sustained_oscillation

    @pytest.mark.parametrize(
        ("name", "value"),
        [("growing_damping_ratio", 0.0), ("oscillation_velocity_rms", math.inf)],
    )
    def test_new_tolerances_validated(self, name: str, value: float) -> None:
        """The new tolerances must be positive and finite."""
        with pytest.raises(ValueError, match=name):
            EquilibriumTolerances(**{name: value})


class TestIntegerPeriodMean:
    """Every reported mean spans a whole number of the dominant period."""

    def test_settled_point_is_a_whole_period_mean(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A small oscillation the velocity criterion accepts does not bias the mean.

        4.5 cycles of a 0.1 m, 100 s surge oscillation over 150-600 s bias the plain
        mean by -0.007 m; the trailing four whole periods carry no bias.
        """
        fst = _make_case(tmp_path)

        def fake_run(fst_path: Path, **_: object) -> OpenFastResult:
            deck = FastInputFile.read(fst_path)
            tmax, tstart = float(deck.get("TMax")), float(deck.get("TStart"))
            out = _out(tmax, tstart, amplitude=0.1, period=100.0)
            fst_path.with_suffix(".out").write_text(out, encoding="utf-8")
            return OpenFastResult(fst_path=fst_path, returncode=0)

        monkeypatch.setattr(steady_state_module, "run_openfast", fake_run)
        op = solve_steady_state(
            fst, 12.0, settle_time=600.0, discard_time=150.0, require_settled=True
        )
        assert op.settle_convergence is not None
        assert op.settle_convergence.criterion == "velocity"
        assert op.platform is not None
        plain = 17.0 - 0.1 * 100.0 / (2.0 * math.pi) * 2.0 / 450.0
        assert plain == pytest.approx(16.993, abs=1e-3)
        assert op.platform.surge == pytest.approx(17.0, abs=1.5e-3)


class TestAutomaticConfirmationLength:
    """The confirmation is sized from the slowest platform period."""

    @pytest.mark.parametrize(
        ("plan", "damped_period", "expected"),
        [
            # Eight periods in the last three quarters: 8 x 125.5 / 0.75, rounded up.
            (_ConfirmPlan(None, 125.5, None), None, (1339.0, "platform_period")),
            # The probe's period wins over the damped stage's.
            (_ConfirmPlan(None, 125.5, None), 300.0, (1339.0, "platform_period")),
            # Never shorter than the fallback length.
            (_ConfirmPlan(None, 60.0, None), None, (1200.0, "platform_period")),
            (_ConfirmPlan(None, None, None), 2.3, (1200.0, "damped_stage")),
            (_ConfirmPlan(None, None, None), 150.0, (1600.0, "damped_stage")),
            (_ConfirmPlan(None, None, None), None, (1200.0, "fallback")),
            # Capped by max_settle_time, else by the ceiling.
            (_ConfirmPlan(None, 125.5, 900.0), None, (900.0, "platform_period")),
            (_ConfirmPlan(None, None, 900.0), None, (900.0, "fallback")),
            (_ConfirmPlan(None, 500.0, None), None, (3600.0, "platform_period")),
            # An explicit length always wins.
            (_ConfirmPlan(450.0, 125.5, 300.0), 2.3, (450.0, "user")),
        ],
    )
    def test_rule(
        self,
        plan: _ConfirmPlan,
        damped_period: float | None,
        expected: tuple[float, str],
    ) -> None:
        """Explicit, else 8 periods / 0.75 in [1200 s, ceiling]."""
        assert _confirm_length(plan, damped_period) == expected

    def test_platform_period_sizes_the_confirmation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The probe's 125.5 s sway period gives a 1339 s confirmation."""
        fake = _Recorded(_EQ_8, lambda tmax, tstart: _quiet_out(tmax, tstart, _EQ_8))
        op = _settle(tmp_path, monkeypatch, fake, platform_period=125.5)
        assert fake.launches == [(True, 1200.0), (False, 1339.0)]
        convergence = op.settle_convergence
        assert convergence is not None
        assert convergence.converged
        assert convergence.confirmation is not None
        assert convergence.confirmation.confirm_time == 1339.0
        assert convergence.confirmation.confirm_time_source == "platform_period"
        # The confirmation discards its first quarter.
        assert convergence.window_start == pytest.approx(0.25 * 1339.0)

    @pytest.mark.parametrize("value", [0.0, -5.0, math.nan, math.inf])
    def test_invalid_platform_period_rejected(
        self, tmp_path: Path, value: float
    ) -> None:
        """The platform period must be positive and finite."""
        fst = _make_case(tmp_path)
        with pytest.raises(ValueError, match="platform_period"):
            solve_steady_state(
                fst, 12.0, platform_damping=_DAMPING, platform_period=value
            )

    def test_probe_slowest_period(self) -> None:
        """The probe reports its longest finite period."""
        probe = PlatformProbe(
            effective_mass={"surge": 3.0e7, "sway": 3.1e7, "pitch": 5.0e10},
            natural_period={"surge": 124.2, "sway": 125.5, "pitch": math.inf},
            damping_ratio={"surge": 0.03, "sway": 0.03, "pitch": 0.02},
            mode_frequency={"surge": 0.008, "sway": 0.008, "pitch": 0.0},
            probe_damping=SettlePlatformDamping(surge=1e6),
            n_states=14,
        )
        assert probe.slowest_period == 125.5
        assert probe.to_dict()["slowest_period"] == 125.5
        empty = PlatformProbe({}, {}, {}, {}, SettlePlatformDamping(surge=1e6), 0)
        assert empty.slowest_period is None


def _table(text: str) -> tuple[list[str], np.ndarray]:
    """Parse a synthetic ``.out`` into its columns and rows."""
    lines = text.splitlines()
    columns = lines[2].split("\t")
    rows = np.array([[float(v) for v in line.split("\t")] for line in lines[4:]])
    return columns, rows
