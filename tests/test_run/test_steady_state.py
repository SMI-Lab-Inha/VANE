# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.run.steady_state — settle a turbine to its operating point."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from vane.io.fst_writer import FastInputFile
from vane.run import PlatformPosition, SteadyStateError, solve_steady_state
from vane.run.runner import OpenFastResult

if TYPE_CHECKING:
    from pathlib import Path

# Platform-velocity channels at near-equilibrium (RMS ~1e-4): the platform has settled.
_OUT_SETTLED = """\
OpenFAST results header line

Time\tRotSpeed\tBldPitch1\tPtfmSurge\tPtfmTVxi\tPtfmTVyi\tPtfmTVzi\tPtfmRVxi\tPtfmRVyi\tPtfmRVzi
(s)\t(rpm)\t(deg)\t(m)\t(m/s)\t(m/s)\t(m/s)\t(deg/s)\t(deg/s)\t(deg/s)
600.0\t7.40\t9.0\t16.0\t1e-4\t1e-4\t1e-4\t1e-4\t1e-4\t1e-4
601.0\t7.60\t11.0\t18.0\t-1e-4\t-1e-4\t-1e-4\t-1e-4\t-1e-4\t-1e-4
"""

# Platform-velocity channels still large (RMS ~0.5): the platform has NOT settled.
_OUT_UNSETTLED = """\
OpenFAST results header line

Time\tRotSpeed\tBldPitch1\tPtfmSurge\tPtfmTVxi\tPtfmTVyi\tPtfmTVzi\tPtfmRVxi\tPtfmRVyi\tPtfmRVzi
(s)\t(rpm)\t(deg)\t(m)\t(m/s)\t(m/s)\t(m/s)\t(deg/s)\t(deg/s)\t(deg/s)
600.0\t7.40\t9.0\t16.0\t0.5\t0.5\t0.5\t0.5\t0.5\t0.5
601.0\t7.60\t11.0\t18.0\t-0.5\t-0.5\t-0.5\t-0.5\t-0.5\t-0.5
"""

_PRIMARY = """\
600 TMax - Total run time (s)
0 TStart - output start (s)
1 OutFileFmt - output format
"default" DT_Out - output step
0 CompInflow - inflow switch {0=still air; 1=InflowWind}
False Linearize - flag
False CalcSteady - flag
3 WrVTK - VTK output
"ED.dat" EDFile - ElastoDyn file
"IW.dat" InflowFile - InflowWind file
"""

# A floating .out that outputs only one platform channel (a partial OutList).
_OUT_PARTIAL = """\
OpenFAST results header line

Time\tRotSpeed\tBldPitch1\tPtfmSurge
(s)\t(rpm)\t(deg)\t(m)
600.0\t7.40\t9.0\t16.0
601.0\t7.60\t11.0\t18.0
"""

_ED = """\
9.55 RotSpeed - rpm
0.0 BlPitch(1) - deg
0.0 BlPitch(2) - deg
0.0 BlPitch(3) - deg
0 PtfmSurge - m
0 PtfmHeave - m
0 PtfmPitch - deg
"""

_IW = """\
3 WindType - switch
8 HWindSpeed - wind speed (m/s)
0.2 PLExp - power law exponent
"""

# A floating .out: header, channel-name row (starts with "Time"), units row, data.
_OUT_FLOATING = """\
OpenFAST results header line

Time\tRotSpeed\tBldPitch1\tPtfmSurge\tPtfmHeave\tPtfmPitch
(s)\t(rpm)\t(deg)\t(m)\t(m)\t(deg)
600.0\t7.40\t9.0\t16.0\t-0.5\t3.0
601.0\t7.60\t11.0\t18.0\t-0.5\t3.4
"""

# An onshore .out: no platform channels.
_OUT_ONSHORE = """\
OpenFAST results header line

Time\tRotSpeed\tBldPitch1
(s)\t(rpm)\t(deg)
600.0\t7.40\t9.0
601.0\t7.60\t11.0
"""

# A .out whose OutList omits RotSpeed/BldPitch1 (only platform velocities present).
_OUT_NO_ROTOR = """\
OpenFAST results header line

Time\tPtfmTVxi\tPtfmTVyi\tPtfmRVzi
(s)\t(m/s)\t(m/s)\t(deg/s)
600.0\t0.001\t0.001\t0.001
601.0\t0.001\t0.001\t0.001
"""


def _make_case(tmp_path: Path) -> Path:
    """Write a deck with ElastoDyn and InflowWind sub-files."""
    (tmp_path / "ED.dat").write_text(_ED, encoding="utf-8")
    (tmp_path / "IW.dat").write_text(_IW, encoding="utf-8")
    fst = tmp_path / "Model.fst"
    fst.write_text(_PRIMARY, encoding="utf-8")
    return fst


def _patch_run(
    monkeypatch: pytest.MonkeyPatch, out_text: str, *, succeed: bool = True
) -> None:
    """Patch run_openfast to write a synthetic .out and report success/failure."""
    from vane.run import steady_state as steady_state_module

    def fake_run(fst_path: Path, **_: object) -> OpenFastResult:
        if succeed:
            fst_path.with_suffix(".out").write_text(out_text, encoding="utf-8")
        return OpenFastResult(
            fst_path=fst_path, returncode=0 if succeed else 1, stdout="boom"
        )

    monkeypatch.setattr(steady_state_module, "run_openfast", fake_run)


class TestSolveSteadyState:
    """Tests for solve_steady_state (with the OpenFAST run patched)."""

    def test_non_finite_run_inputs_rejected_before_running(self) -> None:
        """A non-finite or negative wind/rotor input is rejected up front, no run.

        The check fires before any deck is touched, so a dummy path is fine — a NaN wind
        speed would otherwise take the still-air branch yet still write NaN to the deck.
        """
        with pytest.raises(ValueError, match="wind_speed must be finite"):
            solve_steady_state("nonexistent.fst", float("nan"))
        with pytest.raises(ValueError, match="rotor_speed_rpm must be finite"):
            solve_steady_state("nonexistent.fst", 12.0, rotor_speed_rpm=float("inf"))
        with pytest.raises(ValueError, match="wind_speed must be non-negative"):
            solve_steady_state("nonexistent.fst", -3.0)

    def test_non_finite_settle_times_rejected_before_running(self) -> None:
        """A non-finite settle/discard time is rejected before it is written as TMax.

        ``discard_time >= settle_time`` is ``False`` for a NaN ``settle_time``, so with
        no explicit finite check ``TMax = nan`` would reach the deck.
        """
        with pytest.raises(ValueError, match="settle_time must be finite"):
            solve_steady_state("nonexistent.fst", 12.0, settle_time=float("nan"))
        with pytest.raises(ValueError, match="discard_time must be finite"):
            solve_steady_state("nonexistent.fst", 12.0, discard_time=float("nan"))
        with pytest.raises(ValueError, match="max_settle_time must be finite"):
            solve_steady_state(
                "nonexistent.fst",
                12.0,
                adaptive=True,
                max_settle_time=float("nan"),
            )
        # inf passes ``> 1.0`` but must still be rejected (it would jump straight to
        # max_settle_time on the first growth).
        with pytest.raises(ValueError, match="settle_growth must be > 1 and finite"):
            solve_steady_state(
                "nonexistent.fst",
                12.0,
                adaptive=True,
                max_settle_time=2400.0,
                settle_growth=float("inf"),
            )

    def test_returns_averaged_operating_point(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The settled point averages the steady window, including the platform."""
        fst = _make_case(tmp_path)
        _patch_run(monkeypatch, _OUT_FLOATING)
        op = solve_steady_state(fst, 14.0, settle_time=1200.0, discard_time=600.0)

        assert op.wind_speed == 14.0
        assert op.rotor_speed_rpm == pytest.approx(7.50)  # (7.40 + 7.60) / 2
        assert op.blade_pitch_deg == pytest.approx(10.0)  # (9.0 + 11.0) / 2
        assert op.platform is not None
        assert op.platform.surge == pytest.approx(17.0)  # (16.0 + 18.0) / 2
        assert op.platform.heave == pytest.approx(-0.5)
        assert op.platform.pitch == pytest.approx(3.2)  # (3.0 + 3.4) / 2
        # DOFs the deck did not output stay None (not fabricated zeros).
        assert op.platform.sway is None
        assert op.platform.roll is None
        assert op.platform.yaw is None

    def test_configures_settle_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The deck is set up for a constant-wind, still-water time-domain run."""
        fst = _make_case(tmp_path)
        _patch_run(monkeypatch, _OUT_FLOATING)
        solve_steady_state(fst, 14.0, settle_time=1200.0, discard_time=600.0)

        deck = FastInputFile.read(fst)
        assert deck.get("Linearize") == "False"
        assert deck.get("CalcSteady") == "False"
        assert deck.get("TMax") == "1200.0"
        assert deck.get("TStart") == "600.0"
        # The inflow module is enabled so the requested wind is actually used.
        assert deck.get("CompInflow") == "1"
        inflow = FastInputFile.read(tmp_path / "IW.dat")
        assert inflow.get("WindType") == "1"
        assert inflow.get("HWindSpeed") == "14.0"
        assert inflow.get("PLExp") == "0.0"  # no shear

    def test_configures_explicit_platform_warm_start(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A platform warm start changes only supplied ElastoDyn initial DOFs."""
        fst = _make_case(tmp_path)
        _patch_run(monkeypatch, _OUT_FLOATING)
        platform = PlatformPosition(
            surge=17.7,
            sway=-0.04,  # absent from the synthetic deck: tolerated, not invented
            heave=-0.46,
            pitch=3.37,
        )
        solve_steady_state(
            fst,
            12.0,
            rotor_speed_rpm=7.56,
            blade_pitch_deg=6.33,
            platform_position=platform,
        )

        elastodyn = FastInputFile.read(tmp_path / "ED.dat")
        assert elastodyn.get("RotSpeed") == "7.56"
        assert elastodyn.get("BlPitch(1)") == "6.33"
        assert elastodyn.get("BlPitch(2)") == "6.33"
        assert elastodyn.get("BlPitch(3)") == "6.33"
        assert elastodyn.get("PtfmSurge") == "17.7"
        assert elastodyn.get("PtfmHeave") == "-0.46"
        assert elastodyn.get("PtfmPitch") == "3.37"
        assert "PtfmSway" not in elastodyn

    def test_partial_platform_output_sets_only_measured(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only platform DOFs the deck outputs are set; the rest stay None."""
        fst = _make_case(tmp_path)
        _patch_run(monkeypatch, _OUT_PARTIAL)
        op = solve_steady_state(fst, 14.0)
        assert op.platform is not None
        assert op.platform.surge == pytest.approx(17.0)
        # Unmeasured DOFs are not manufactured, so configure_linearization will not
        # overwrite the deck's heave/pitch with fabricated zeros.
        assert op.platform.heave is None
        assert op.platform.pitch is None

    def test_positive_wind_without_inflow_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A positive-wind settle with no usable InflowWind file is rejected."""
        (tmp_path / "ED.dat").write_text(_ED, encoding="utf-8")
        fst = tmp_path / "NoInflow.fst"
        fst.write_text(
            _PRIMARY.replace('"IW.dat" InflowFile', '"missing.dat" InflowFile'),
            encoding="utf-8",
        )
        _patch_run(monkeypatch, _OUT_FLOATING)
        with pytest.raises(SteadyStateError, match="no usable InflowWind file"):
            solve_steady_state(fst, 14.0)

    def test_onshore_output_has_no_platform(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A deck that outputs no platform channels yields ``platform=None``."""
        fst = _make_case(tmp_path)
        _patch_run(monkeypatch, _OUT_ONSHORE)
        op = solve_steady_state(fst, 12.0)
        assert op.platform is None
        assert op.rotor_speed_rpm == pytest.approx(7.50)

    def test_failed_run_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed settle run raises SteadyStateError."""
        fst = _make_case(tmp_path)
        _patch_run(monkeypatch, _OUT_FLOATING, succeed=False)
        with pytest.raises(SteadyStateError, match="settle run failed"):
            solve_steady_state(fst, 14.0)

    def test_missing_output_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A successful run that produced no .out raises SteadyStateError."""
        from vane.run import steady_state as steady_state_module

        fst = _make_case(tmp_path)

        def fake_run(fst_path: Path, **_: object) -> OpenFastResult:
            return OpenFastResult(fst_path=fst_path, returncode=0)  # writes no .out

        monkeypatch.setattr(steady_state_module, "run_openfast", fake_run)
        with pytest.raises(SteadyStateError, match="output file not found"):
            solve_steady_state(fst, 14.0)

    def test_missing_rotor_channels_raise_instead_of_seed_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A settle .out without RotSpeed/BldPitch1 raises, not a silent seed fallback.

        The settled rotor speed and pitch are the run's whole output; if the OutList
        omits them the function must fail clearly rather than return the (parked) seeds.
        """
        fst = _make_case(tmp_path)
        _patch_run(monkeypatch, _OUT_NO_ROTOR)
        with pytest.raises(SteadyStateError, match=r"RotSpeed.*BldPitch1"):
            solve_steady_state(fst, 14.0, require_settled=False)


class TestSettleConvergence:
    """Tests for the settle equilibrium criterion and adaptive TMax."""

    def test_settled_run_does_not_warn(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Velocities within tolerance settle cleanly with no warning."""
        fst = _make_case(tmp_path)
        _patch_run(monkeypatch, _OUT_SETTLED)
        with caplog.at_level("WARNING"):
            op = solve_steady_state(fst, 14.0, require_settled=True)
        assert op.platform is not None
        assert op.settle_convergence is not None
        assert op.settle_convergence.assessed
        assert op.settle_convergence.converged
        assert "not settled" not in caplog.text

    def test_unsettled_run_warns_but_returns(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Above tolerance, the point is still returned but a warning is logged."""
        fst = _make_case(tmp_path)
        _patch_run(monkeypatch, _OUT_UNSETTLED)
        with caplog.at_level("WARNING"):
            op = solve_steady_state(fst, 14.0)
        assert op.rotor_speed_rpm == pytest.approx(7.50)
        assert op.settle_convergence is not None
        assert not op.settle_convergence.converged
        assert "not settled" in caplog.text

    def test_unsettled_run_raises_when_required(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With require_settled, an unsettled run raises."""
        fst = _make_case(tmp_path)
        _patch_run(monkeypatch, _OUT_UNSETTLED)
        with pytest.raises(SteadyStateError, match="not settled"):
            solve_steady_state(fst, 14.0, require_settled=True)

    def test_no_velocity_channels_cannot_be_verified(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A deck without platform-velocity channels cannot confirm settling."""
        fst = _make_case(tmp_path)
        _patch_run(monkeypatch, _OUT_FLOATING)  # no PtfmTVxi/PtfmRVxi channels
        with pytest.raises(SteadyStateError, match="could not verify steady state"):
            solve_steady_state(fst, 14.0, require_settled=True)

    def test_adaptive_requires_max_settle_time(self, tmp_path: Path) -> None:
        """Adaptive settling without a cap is rejected."""
        fst = _make_case(tmp_path)
        with pytest.raises(ValueError, match="requires max_settle_time"):
            solve_steady_state(fst, 14.0, adaptive=True)

    def test_adaptive_cap_below_start_is_rejected(self, tmp_path: Path) -> None:
        """A cap below the initial settle_time is rejected before any run."""
        fst = _make_case(tmp_path)
        with pytest.raises(ValueError, match="must be >= settle_time"):
            solve_steady_state(
                fst, 14.0, settle_time=1500.0, adaptive=True, max_settle_time=1000.0
            )

    def test_non_positive_tolerance_is_rejected(self, tmp_path: Path) -> None:
        """A non-positive settle tolerance is rejected."""
        fst = _make_case(tmp_path)
        with pytest.raises(ValueError, match="must be positive"):
            solve_steady_state(fst, 14.0, velocity_tol=0.0)

    def test_discard_time_must_precede_settle_time(self, tmp_path: Path) -> None:
        """A discard window at or beyond settle_time is rejected."""
        fst = _make_case(tmp_path)
        with pytest.raises(ValueError, match="discard_time must be"):
            solve_steady_state(fst, 14.0, settle_time=600.0, discard_time=600.0)

    def test_adaptive_grows_tmax_until_settled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Adaptive runs extend TMax and stop once the platform settles."""
        from vane.run import steady_state as steady_state_module

        fst = _make_case(tmp_path)
        runs: list[float] = []

        def fake_run(fst_path: Path, **_: object) -> OpenFastResult:
            tmax = float(FastInputFile.read(fst_path).get("TMax"))
            runs.append(tmax)
            text = _OUT_SETTLED if tmax >= 1800.0 else _OUT_UNSETTLED
            fst_path.with_suffix(".out").write_text(text, encoding="utf-8")
            return OpenFastResult(fst_path=fst_path, returncode=0)

        monkeypatch.setattr(steady_state_module, "run_openfast", fake_run)
        op = solve_steady_state(
            fst,
            14.0,
            settle_time=1000.0,
            adaptive=True,
            max_settle_time=3000.0,
            require_settled=True,
        )
        assert op.platform is not None
        # 1000 -> 1500 -> 2250 (>= 1800, settles); TMax grew each iteration.
        assert len(runs) == 3
        assert runs == sorted(runs)
        assert runs[-1] >= 1800.0
        assert op.settle_convergence is not None
        assert op.settle_convergence.settle_time == pytest.approx(runs[-1])
        assert op.settle_convergence.iterations == 3
        assert [
            attempt.settle_time for attempt in op.settle_convergence.attempts
        ] == runs
        assert [attempt.converged for attempt in op.settle_convergence.attempts] == [
            False,
            False,
            True,
        ]

    def test_adaptive_rejection_carries_every_attempt(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A capped rejection exposes and logs all overwritten attempt metrics."""
        fst = _make_case(tmp_path)
        _patch_run(monkeypatch, _OUT_UNSETTLED)
        with (
            caplog.at_level("WARNING"),
            pytest.raises(SteadyStateError, match="not settled") as exc_info,
        ):
            solve_steady_state(
                fst,
                14.0,
                settle_time=1000.0,
                adaptive=True,
                max_settle_time=1500.0,
                require_settled=True,
            )

        convergence = exc_info.value.convergence
        assert convergence is not None
        assert convergence.iterations == 2
        assert [attempt.iteration for attempt in convergence.attempts] == [1, 2]
        assert [attempt.settle_time for attempt in convergence.attempts] == [
            1000.0,
            1500.0,
        ]
        assert all(not attempt.converged for attempt in convergence.attempts)
        assert "settle attempt 1 at 1000 s did not converge" in caplog.text
        assert "retrying at 1500 s" in caplog.text


class TestSettleConvergenceUnit:
    """Direct tests for the convergence helper."""

    def test_rms_and_outcome(self) -> None:
        """RMS is computed per channel and compared to the tolerances."""
        import numpy as np

        from vane.run.steady_state import _settle_convergence

        columns = ["Time", "PtfmTVxi", "PtfmRVxi"]
        window = np.array([[600.0, 0.2, 0.0], [601.0, -0.2, 0.0]])
        conv = _settle_convergence(columns, window, 1e-2, 1e-2, 1200.0, 1)
        assert conv.assessed
        assert not conv.converged  # 0.2 m/s translational exceeds 1e-2
        assert conv.translational_velocity_rms == pytest.approx(0.2)
        assert conv.angular_velocity_rms == pytest.approx(0.0)

    def test_unassessed_when_no_velocity_channels(self) -> None:
        """Without velocity channels the run is unassessed (and not failed)."""
        import numpy as np

        from vane.run.steady_state import _settle_convergence

        columns = ["Time", "RotSpeed"]
        window = np.array([[600.0, 7.4], [601.0, 7.6]])
        conv = _settle_convergence(columns, window, 1e-2, 1e-2, 1200.0, 1)
        assert not conv.assessed
        assert conv.converged  # nothing measured cannot fail


class TestReadChannelWindow:
    """Tests for the steady-window extraction of an OpenFAST .out file."""

    def test_discards_initial_transient(self, tmp_path: Path) -> None:
        """Only rows at or after the discard time are kept in the steady window."""
        from vane.run.steady_state import _read_channel_window

        out = tmp_path / "Model.out"
        out.write_text(
            "header\n\nTime\tRotSpeed\n(s)\t(rpm)\n"
            "0.0\t1.0\n300.0\t2.0\n600.0\t8.0\n900.0\t10.0\n",
            encoding="utf-8",
        )
        columns, window = _read_channel_window(out, discard_time=600.0)
        rot_speed = window[:, columns.index("RotSpeed")]
        assert rot_speed.mean() == pytest.approx(9.0)  # (8.0 + 10.0) / 2, not 0-300

    @pytest.mark.parametrize("token", ["NaN", "Infinity", "-Inf", "**********"])
    def test_non_finite_window_value_is_a_settle_failure(
        self, tmp_path: Path, token: str
    ) -> None:
        """A NaN, infinite or overflowed (``*****``) sample means the run diverged."""
        from vane.run.steady_state import _read_channel_window

        out = tmp_path / "Model.out"
        out.write_text(
            "header\n\nTime\tRotSpeed\tPtfmTVxi\n(s)\t(rpm)\t(m/s)\n"
            f"600.0\t7.0\t1e-4\n601.0\t7.0\t{token}\n",
            encoding="utf-8",
        )
        with pytest.raises(SteadyStateError, match=r"non-finite|not numeric"):
            _read_channel_window(out, discard_time=600.0)

    def test_non_finite_before_the_window_is_ignored(self, tmp_path: Path) -> None:
        """Only the analysis window is judged; the discarded transient may hold NaN."""
        from vane.run.steady_state import _read_channel_window

        out = tmp_path / "Model.out"
        out.write_text(
            "header\n\nTime\tRotSpeed\n(s)\t(rpm)\n0.0\tNaN\n600.0\t8.0\n900.0\t10.0\n",
            encoding="utf-8",
        )
        columns, window = _read_channel_window(out, discard_time=600.0)
        assert window[:, columns.index("RotSpeed")].tolist() == [8.0, 10.0]


# A diverged settle: one platform-velocity channel turned NaN in the window. The
# NaN channel is not the first one, so a plain ``max`` over the channel RMS values
# skipped it and the run was reported settled.
_OUT_NAN_VELOCITY = """\
OpenFAST results header line

Time\tRotSpeed\tBldPitch1\tPtfmSurge\tPtfmTVxi\tPtfmTVyi\tPtfmTVzi
(s)\t(rpm)\t(deg)\t(m)\t(m/s)\t(m/s)\t(m/s)
600.0\t7.40\t9.0\t16.0\t1e-4\tNaN\t1e-4
601.0\t7.60\t11.0\t18.0\t-1e-4\tNaN\t-1e-4
"""


class TestDivergedSettle:
    """A settle whose output turned non-finite is a failure, never settled."""

    def test_nan_velocity_is_not_reported_settled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The run raises instead of returning a converged NaN-free point."""
        fst = _make_case(tmp_path)
        _patch_run(monkeypatch, _OUT_NAN_VELOCITY)
        with pytest.raises(SteadyStateError, match="non-finite"):
            solve_steady_state(fst, 14.0, settle_time=1200.0, discard_time=600.0)


class TestSettleOutputStep:
    """The settle output step is a whole multiple of the deck's time step."""

    @pytest.mark.parametrize(
        ("dt", "expected"),
        [("0.0125", 1.0), ("0.003", 0.999), ("0.015", 1.005), ("2.5", 2.5)],
    )
    def test_dt_out_is_a_multiple_of_dt(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        dt: str,
        expected: float,
    ) -> None:
        """OpenFAST rejects a DT_Out that is not an integer multiple of DT."""
        fst = _make_case(tmp_path)
        fst.write_text(f"{dt} DT - time step\n" + _PRIMARY, encoding="utf-8")
        _patch_run(monkeypatch, _OUT_FLOATING)
        solve_steady_state(fst, 14.0, settle_time=1200.0, discard_time=600.0)
        deck = FastInputFile.read(fst)
        dt_out = float(deck.get("DT_Out"))
        assert dt_out == pytest.approx(expected, rel=1e-12)
        steps = dt_out / float(dt)
        assert steps == pytest.approx(round(steps), rel=1e-9)
        assert round(steps) >= 1

    def test_unreadable_dt_keeps_one_second(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a numeric DT the output step stays at one second."""
        fst = _make_case(tmp_path)
        _patch_run(monkeypatch, _OUT_FLOATING)
        solve_steady_state(fst, 14.0, settle_time=1200.0, discard_time=600.0)
        assert float(FastInputFile.read(fst).get("DT_Out")) == 1.0


def _transient_out(tmax: float, *, decay_time: float) -> str:
    """Return a full ``[0, tmax]`` .out whose platform moves until ``decay_time``.

    Before ``decay_time`` the platform velocities alternate at +/-0.5 and the surge,
    rotor speed and pitch sit at transient values; afterwards the platform is still at
    its equilibrium. Every row from t=0 is written (``TStart`` deliberately ignored),
    so the analysis window must also be enforced when the output is read back.
    """
    lines = [
        "OpenFAST results header line",
        "",
        "Time\tRotSpeed\tBldPitch1\tPtfmSurge\tPtfmTVxi\tPtfmRVyi",
        "(s)\t(rpm)\t(deg)\t(m)\t(m/s)\t(deg/s)",
    ]
    for step in range(int(tmax / 10.0) + 1):
        time = 10.0 * step
        if time < decay_time:
            velocity = 0.5 if step % 2 == 0 else -0.5
            lines.append(f"{time}\t3.0\t0.0\t30.0\t{velocity}\t{velocity}")
        else:
            lines.append(f"{time}\t7.5\t10.0\t17.0\t0.0\t0.0")
    return "\n".join(lines) + "\n"


class TestTrailingSettleWindow:
    """Adaptive retries judge and average a fixed-length trailing window."""

    def test_decayed_transient_converges_on_second_attempt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A transient over by the second run's window is excluded, not diluted.

        With ``settle_time=1000`` and ``discard_time=600`` the window is 400 s long, so
        the second run (``TMax=1500``) is judged over ``[1100, 1500]``: the transient
        that ends at 1050 s lies wholly outside it. Judging ``[600, TMax]`` instead
        keeps the transient in every window and never converges below the 2250 s cap.
        """
        import json
        from dataclasses import asdict

        from vane.run import steady_state as steady_state_module

        fst = _make_case(tmp_path)
        tstarts: list[float] = []

        def fake_run(fst_path: Path, **_: object) -> OpenFastResult:
            deck = FastInputFile.read(fst_path)
            tstarts.append(float(deck.get("TStart")))
            out = _transient_out(float(deck.get("TMax")), decay_time=1050.0)
            fst_path.with_suffix(".out").write_text(out, encoding="utf-8")
            return OpenFastResult(fst_path=fst_path, returncode=0)

        monkeypatch.setattr(steady_state_module, "run_openfast", fake_run)
        op = solve_steady_state(
            fst,
            14.0,
            settle_time=1000.0,
            discard_time=600.0,
            adaptive=True,
            max_settle_time=2250.0,
            require_settled=True,
            criterion="velocity",
        )

        convergence = op.settle_convergence
        assert convergence is not None
        assert convergence.converged
        assert convergence.iterations == 2
        # TStart trims the output to the trailing window of every run.
        assert tstarts == [600.0, 1100.0]
        assert (convergence.window_start, convergence.window_end) == (1100.0, 1500.0)
        assert [
            (attempt.window_start, attempt.window_end)
            for attempt in convergence.attempts
        ] == [(600.0, 1000.0), (1100.0, 1500.0)]
        assert [attempt.converged for attempt in convergence.attempts] == [False, True]
        assert convergence.translational_velocity_rms == pytest.approx(0.0)
        # The returned equilibrium is the trailing-window mean: no transient mixed in.
        assert op.rotor_speed_rpm == pytest.approx(7.5)
        assert op.blade_pitch_deg == pytest.approx(10.0)
        assert op.platform is not None
        assert op.platform.surge == pytest.approx(17.0)
        # The new provenance fields stay strict-JSON serializable.
        json.dumps(asdict(convergence), allow_nan=False)

    def test_first_attempt_window_is_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A single run keeps the ``[discard_time, settle_time]`` window."""
        fst = _make_case(tmp_path)
        _patch_run(monkeypatch, _OUT_SETTLED)
        op = solve_steady_state(fst, 14.0, settle_time=1200.0, discard_time=600.0)
        assert op.settle_convergence is not None
        assert op.settle_convergence.window_start == 600.0
        assert op.settle_convergence.window_end == 1200.0
        assert FastInputFile.read(fst).get("TStart") == "600.0"


def _make_lin_case(tmp_path: Path) -> Path:
    """Write the settle fixture plus the primary-deck linearization keywords."""
    fst = _make_case(tmp_path)
    fst.write_text(
        _PRIMARY
        + "1 NLinTimes - -\n0 LinTimes - s\n3 TrimCase - -\n"
        + "0.01 TrimGain - -\n0.001 TrimTol - -\n",
        encoding="utf-8",
    )
    return fst


class TestSettleAndLinearizationInflowAgree:
    """The settle run and the linearization write the same inflow shear."""

    @staticmethod
    def _settled_and_linearized_plexp(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exponent: float | None
    ) -> tuple[str, str]:
        """Return PLExp after a settle run and after linearization configuration."""
        from vane.run import LinearizationSettings, configure_linearization

        settle_dir = tmp_path / "settle"
        settle_dir.mkdir()
        fst = _make_case(settle_dir)
        _patch_run(monkeypatch, _OUT_FLOATING)
        point = solve_steady_state(
            fst,
            14.0,
            settle_time=1200.0,
            discard_time=600.0,
            wind_shear_exponent=exponent,
        )
        settled = FastInputFile.read(settle_dir / "IW.dat").get("PLExp")

        lin_dir = tmp_path / "lin"
        lin_dir.mkdir()
        lin_fst = _make_lin_case(lin_dir)
        configure_linearization(
            lin_fst,
            point,
            LinearizationSettings(rated_wind_speed=11.4, wind_shear_exponent=exponent),
        )
        linearized = FastInputFile.read(lin_dir / "IW.dat").get("PLExp")
        return settled, linearized

    def test_default_is_uniform_inflow_in_both(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A point settled in uniform inflow is linearized in uniform inflow.

        The template carries a sheared ``PLExp = 0.2``; the settle run removed it but
        the linearization used to keep it, so the settled equilibrium was linearized
        in a different (sheared) wind.
        """
        from vane.run import LinearizationSettings, configure_linearization

        fst = _make_case(tmp_path)
        _patch_run(monkeypatch, _OUT_FLOATING)
        point = solve_steady_state(fst, 14.0, settle_time=1200.0, discard_time=600.0)
        settled = FastInputFile.read(tmp_path / "IW.dat").get("PLExp")
        lin_dir = tmp_path / "lin"
        lin_dir.mkdir()
        lin_fst = _make_lin_case(lin_dir)
        configure_linearization(
            lin_fst, point, LinearizationSettings(rated_wind_speed=11.4)
        )
        linearized = FastInputFile.read(lin_dir / "IW.dat").get("PLExp")

        assert float(settled) == 0.0
        assert float(linearized) == float(settled)

    def test_explicit_exponent_is_used_by_both(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A requested shear exponent reaches both the settle and the linearization."""
        settled, linearized = self._settled_and_linearized_plexp(
            tmp_path, monkeypatch, 0.14
        )
        assert float(settled) == pytest.approx(0.14)
        assert float(linearized) == pytest.approx(0.14)

    def test_none_leaves_the_deck_shear_in_both(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``None`` keeps the template's own PLExp for both runs."""
        settled, linearized = self._settled_and_linearized_plexp(
            tmp_path, monkeypatch, None
        )
        assert float(settled) == pytest.approx(0.2)
        assert float(linearized) == pytest.approx(0.2)

    def test_non_finite_exponent_is_rejected(self, tmp_path: Path) -> None:
        """A NaN shear exponent is rejected before the deck is touched."""
        from vane.run import LinearizationSettings

        fst = _make_case(tmp_path)
        with pytest.raises(ValueError, match="wind_shear_exponent"):
            solve_steady_state(fst, 14.0, wind_shear_exponent=float("nan"))
        with pytest.raises(ValueError, match="wind_shear_exponent"):
            LinearizationSettings(
                rated_wind_speed=11.4, wind_shear_exponent=float("inf")
            )
        flag: Any = True
        with pytest.raises(ValueError, match="wind_shear_exponent"):
            LinearizationSettings(rated_wind_speed=11.4, wind_shear_exponent=flag)


# A SeaState carrying every option OpenFAST rejects in still water.
_SEA_STATE_WAVY = """\
2 WaveMod - incident wave model
3 WaveStMod - wave stretching model
30.0 WaveDir - incident wave heading (deg)
1 WaveDirMod - directional spreading
1 ConstWaveMod - constrained wave
1.5 MCFD - MacCamy-Fuchs diameter (m)
1 CurrMod - current model
DEFAULT CurrSSDir - sub-surface current heading
"""

# A ServoDyn driving pitch and torque through the Bladed-style controller library.
_SERVO_DLL = """\
5 PCMode - pitch control mode
5 VSContrl - variable-speed control mode
0 HSSBrMode - HSS brake model
0 YCMode - yaw control mode
"/home/runner/miniconda3/envs/test/lib/libdiscon.so" DLL_FileName - controller library
"DISCON.IN" DLL_InFile - controller input file
"""

# A ServoDyn with no control routed through the library (simple laws only).
_SERVO_SIMPLE = _SERVO_DLL.replace("5 PCMode", "0 PCMode").replace(
    "5 VSContrl", "1 VSContrl"
)


def _make_offshore_case(
    tmp_path: Path, servo: str = _SERVO_DLL, *, comp_servo: int = 1
) -> Path:
    """Write the settle fixture plus SeaState and ServoDyn sub-files."""
    fst = _make_case(tmp_path)
    (tmp_path / "SeaState.dat").write_text(_SEA_STATE_WAVY, encoding="utf-8")
    (tmp_path / "SD.dat").write_text(servo, encoding="utf-8")
    fst.write_text(
        _PRIMARY
        + f"{comp_servo} CompServo - ServoDyn switch\n"
        + '"SeaState.dat" SeaStFile - SeaState file\n'
        + '"SD.dat" ServoFile - ServoDyn file\n',
        encoding="utf-8",
    )
    return fst


def _patch_run_counting(monkeypatch: pytest.MonkeyPatch, out_text: str) -> list[Path]:
    """Patch run_openfast to write ``out_text`` and record every launched deck."""
    from vane.run import steady_state as steady_state_module

    launched: list[Path] = []

    def fake_run(fst_path: Path, **_: object) -> OpenFastResult:
        launched.append(fst_path)
        fst_path.with_suffix(".out").write_text(out_text, encoding="utf-8")
        return OpenFastResult(fst_path=fst_path, returncode=0, stdout="ok")

    monkeypatch.setattr(steady_state_module, "run_openfast", fake_run)
    return launched


class TestSettleStillWater:
    """The settle run uses exactly the linearization's still water."""

    def test_still_water_conflicts_are_neutralised(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """WaveStMod, MCFD, CurrSSDir, WaveDirMod and ConstWaveMod are made valid."""
        fst = _make_offshore_case(tmp_path, _SERVO_SIMPLE)
        _patch_run_counting(monkeypatch, _OUT_FLOATING)
        solve_steady_state(fst, 14.0, settle_time=1200.0, discard_time=600.0)

        sea = FastInputFile.read(tmp_path / "SeaState.dat")
        assert sea.get("WaveMod") == "0"
        assert sea.get("WaveStMod") == "0"
        assert float(sea.get("MCFD")) == 0.0
        assert float(sea.get("CurrSSDir")) == pytest.approx(30.0)
        assert sea.get("WaveDirMod") == "0"
        assert sea.get("ConstWaveMod") == "0"

    def test_matches_the_linearization_sea_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Settle and linearization leave identical SeaState files."""
        from vane.run import LinearizationSettings
        from vane.run.lin_config import _configure_seastate

        settle_dir = tmp_path / "settle"
        settle_dir.mkdir()
        lin_dir = tmp_path / "lin"
        lin_dir.mkdir()
        fst = _make_offshore_case(settle_dir, _SERVO_SIMPLE)
        lin_fst = _make_offshore_case(lin_dir, _SERVO_SIMPLE)
        _patch_run_counting(monkeypatch, _OUT_FLOATING)
        solve_steady_state(fst, 14.0, settle_time=1200.0, discard_time=600.0)
        settings = LinearizationSettings(rated_wind_speed=11.4, wave_mod=0)
        _configure_seastate(FastInputFile.read(lin_fst), settings)

        settled = (settle_dir / "SeaState.dat").read_text(encoding="utf-8")
        linearized = (lin_dir / "SeaState.dat").read_text(encoding="utf-8")
        assert settled == linearized

    def test_sea_state_without_wave_mod_is_left_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A SeaState variant without WaveMod is not edited."""
        fst = _make_offshore_case(tmp_path, _SERVO_SIMPLE)
        text = "3 WaveStMod - stretching\n"
        (tmp_path / "SeaState.dat").write_text(text, encoding="utf-8")
        _patch_run_counting(monkeypatch, _OUT_FLOATING)
        solve_steady_state(fst, 14.0, settle_time=1200.0, discard_time=600.0)
        assert (tmp_path / "SeaState.dat").read_text(encoding="utf-8") == text


class TestSettleControllerLibrary:
    """The settle run's controller library is checked, and can be replaced."""

    def test_missing_library_fails_before_launch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A DLL_FileName that does not exist fails fast; nothing runs or is edited."""
        fst = _make_offshore_case(tmp_path)
        primary_before = fst.read_text(encoding="utf-8")
        launched = _patch_run_counting(monkeypatch, _OUT_FLOATING)

        with pytest.raises(
            SteadyStateError,
            match=r"controller library not found: .*libdiscon\.so.*pass a controller",
        ):
            solve_steady_state(fst, 14.0, settle_time=1200.0, discard_time=600.0)

        assert launched == []
        assert fst.read_text(encoding="utf-8") == primary_before
        assert FastInputFile.read(tmp_path / "SeaState.dat").get("WaveMod") == "2"

    def test_existing_relative_library_runs_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A library that exists (relative to ServoDyn) runs; DLL_FileName untouched."""
        servo = _SERVO_DLL.replace(
            "/home/runner/miniconda3/envs/test/lib/libdiscon.so", "lib/libdiscon.dll"
        )
        fst = _make_offshore_case(tmp_path, servo)
        (tmp_path / "lib").mkdir()
        (tmp_path / "lib" / "libdiscon.dll").write_bytes(b"MZ")
        launched = _patch_run_counting(monkeypatch, _OUT_FLOATING)

        solve_steady_state(fst, 14.0, settle_time=1200.0, discard_time=600.0)

        assert launched == [fst]
        assert (tmp_path / "SD.dat").read_text(encoding="utf-8") == servo

    def test_override_rewrites_dll_filename_in_the_copy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """controller_library is written, absolute, into the settled deck's ServoDyn."""
        case = tmp_path / "case"
        case.mkdir()
        fst = _make_offshore_case(case)
        library = tmp_path / "controller" / "libdiscon.dll"
        library.parent.mkdir()
        library.write_bytes(b"MZ")
        launched = _patch_run_counting(monkeypatch, _OUT_FLOATING)

        solve_steady_state(
            fst,
            14.0,
            settle_time=1200.0,
            discard_time=600.0,
            controller_library=library,
        )

        assert launched == [fst]
        servo = FastInputFile.read(case / "SD.dat")
        assert servo.resolve("DLL_FileName") == library.resolve()
        assert servo.get("DLL_FileName").startswith('"')
        # The rest of ServoDyn (the controller modes) is not touched.
        assert servo.get("PCMode") == "5"
        assert servo.get("DLL_InFile") == '"DISCON.IN"'

    def test_missing_override_is_rejected(self, tmp_path: Path) -> None:
        """A controller_library that does not exist raises FileNotFoundError."""
        fst = _make_offshore_case(tmp_path)
        with pytest.raises(FileNotFoundError, match="controller library not found"):
            solve_steady_state(
                fst, 14.0, controller_library=tmp_path / "nowhere" / "libdiscon.dll"
            )

    def test_override_unused_without_library_control(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """With no control routed to the library the override is unused and warned."""
        fst = _make_offshore_case(tmp_path, _SERVO_SIMPLE)
        library = tmp_path / "libdiscon.dll"
        library.write_bytes(b"MZ")
        _patch_run_counting(monkeypatch, _OUT_FLOATING)

        with caplog.at_level("WARNING", logger="vane.run.steady_state"):
            solve_steady_state(
                fst,
                14.0,
                settle_time=1200.0,
                discard_time=600.0,
                controller_library=library,
            )

        assert "is unused" in caplog.text
        assert (tmp_path / "SD.dat").read_text(encoding="utf-8") == _SERVO_SIMPLE

    def test_disabled_servodyn_is_not_checked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CompServo = 0 never loads the library, so a missing one is not an error."""
        fst = _make_offshore_case(tmp_path, comp_servo=0)
        launched = _patch_run_counting(monkeypatch, _OUT_FLOATING)
        solve_steady_state(fst, 14.0, settle_time=1200.0, discard_time=600.0)
        assert launched == [fst]

    def test_unusable_dll_filename_reports_none(self, tmp_path: Path) -> None:
        """An ``unused`` DLL_FileName with library control is reported as missing."""
        from vane.run import check_controller_library

        servo = _SERVO_DLL.replace(
            '"/home/runner/miniconda3/envs/test/lib/libdiscon.so"', '"unused"'
        )
        fst = _make_offshore_case(tmp_path, servo)
        with pytest.raises(SteadyStateError, match=r"not found: \(none\)"):
            check_controller_library(fst)

    def test_deck_without_servodyn_needs_no_library(self, tmp_path: Path) -> None:
        """A deck with no, a missing, or an unused ServoDyn file needs no check."""
        from vane.run import check_controller_library

        fst = _make_case(tmp_path)
        assert check_controller_library(fst) is None
        fst.write_text(
            _PRIMARY + '"missing.dat" ServoFile - ServoDyn file\n', encoding="utf-8"
        )
        assert check_controller_library(fst) is None
        fst.write_text(
            _PRIMARY + '"unused" ServoFile - ServoDyn file\n', encoding="utf-8"
        )
        assert check_controller_library(fst) is None


_ED_OUTLIST = (
    _ED
    + """\
---------------------- OUTPUT --------------------------------------------------
                   OutList             - The next line(s) contains a list
"Azimuth"
"BldPitch1, RotSpeed"
"PtfmSurge"   - surge
"ptfmtvxi"
END of input file (the word "END" must appear in the first 3 columns)
---------------------- NODE OUTPUTS --------------------------------------------
              OutList     - The next line(s) contains a list of node outputs
"TDx"    - local flapwise deflection
END of input file (the word "END" must appear in the first 3 columns)
"""
)


class TestSettleOutputChannels:
    """The settle copy outputs every channel the settle outcome reads."""

    def test_missing_velocity_channels_are_appended(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only the absent channels are added, before the first OutList's END."""
        fst = _make_case(tmp_path)
        (tmp_path / "ED.dat").write_text(_ED_OUTLIST, encoding="utf-8")
        _patch_run_counting(monkeypatch, _OUT_SETTLED)
        solve_steady_state(fst, 14.0, settle_time=1200.0, discard_time=600.0)

        lines = (tmp_path / "ED.dat").read_text(encoding="utf-8").splitlines()
        added = (
            '"PtfmTVyi, PtfmTVzi, PtfmRVxi, PtfmRVyi, PtfmRVzi, '
            'PtfmSway, PtfmHeave, PtfmRoll, PtfmPitch, PtfmYaw"'
        )
        assert lines.count(added) == 1
        index = lines.index(added)
        assert lines[index - 1] == '"ptfmtvxi"'
        assert lines[index + 1].startswith("END")
        # The node-output OutList block is not touched.
        assert lines[-3].startswith("              OutList")
        assert lines[-2].startswith('"TDx"')

    def test_complete_outlist_is_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A deck already outputting every settle channel is not rewritten."""
        complete = _ED_OUTLIST.replace(
            '"ptfmtvxi"',
            '"PtfmTVxi PtfmTVyi PtfmTVzi"\n"PtfmRVxi; PtfmRVyi; PtfmRVzi"\n'
            '"PtfmSway, PtfmHeave, PtfmRoll, PtfmPitch, PtfmYaw"',
        )
        fst = _make_case(tmp_path)
        (tmp_path / "ED.dat").write_text(complete, encoding="utf-8")
        _patch_run_counting(monkeypatch, _OUT_SETTLED)
        solve_steady_state(fst, 14.0, settle_time=1200.0, discard_time=600.0)
        ed_after = (tmp_path / "ED.dat").read_text(encoding="utf-8")
        assert ed_after.split("OutList", 1)[1] == complete.split("OutList", 1)[1]

    def test_outlist_without_end_is_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An OutList block with no END line is not guessed at."""
        unterminated = _ED + '   OutList - list\n"RotSpeed"\n'
        fst = _make_case(tmp_path)
        (tmp_path / "ED.dat").write_text(unterminated, encoding="utf-8")
        _patch_run_counting(monkeypatch, _OUT_SETTLED)
        solve_steady_state(fst, 14.0, settle_time=1200.0, discard_time=600.0)
        ed_after = (tmp_path / "ED.dat").read_text(encoding="utf-8")
        assert ed_after.endswith('OutList - list\n"RotSpeed"\n')

    def test_missing_elastodyn_reference_is_tolerated(self, tmp_path: Path) -> None:
        """A deck whose EDFile is unusable or absent is left alone."""
        from vane.run.steady_state import _ensure_settle_channels

        fst = _make_case(tmp_path)
        fst.write_text('"unused" EDFile - ElastoDyn file\n', encoding="utf-8")
        _ensure_settle_channels(FastInputFile.read(fst))
        fst.write_text('"gone.dat" EDFile - ElastoDyn file\n', encoding="utf-8")
        _ensure_settle_channels(FastInputFile.read(fst))
        fst.write_text("1 CompServo - switch\n", encoding="utf-8")
        _ensure_settle_channels(FastInputFile.read(fst))
        assert (tmp_path / "ED.dat").read_text(encoding="utf-8") == _ED


class TestCompareSettledToCurve:
    """The settled-versus-curve comparison of a curve-seeded settle."""

    def test_agreement_passes(self) -> None:
        """A settled point within tolerance of the curve passes."""
        from vane.run import OperatingPoint, compare_settled_to_curve

        check = compare_settled_to_curve(
            OperatingPoint(12.0, 7.55, 6.9), OperatingPoint(12.0, 7.52, 6.76)
        )
        assert check.status == "pass"
        assert check.flagged_quantities == ()
        assert check.rotor_speed.relative
        assert not check.pitch.relative

    def test_disagreement_is_flagged(self) -> None:
        """Rotor speed beyond 1 % and pitch beyond 0.5 deg are flagged."""
        from vane.run import OperatingPoint, compare_settled_to_curve

        check = compare_settled_to_curve(
            OperatingPoint(12.0, 7.0, 8.0), OperatingPoint(12.0, 7.52, 6.76)
        )
        assert check.status == "flag"
        assert check.flagged_quantities == ("rotor_speed", "pitch")
        assert check.pitch.deviation == pytest.approx(1.24)

    def test_tolerances_are_applied(self) -> None:
        """Caller tolerances replace the defaults."""
        from vane.run import OperatingPoint, TrimTolerances, compare_settled_to_curve

        check = compare_settled_to_curve(
            OperatingPoint(12.0, 7.0, 8.0),
            OperatingPoint(12.0, 7.52, 6.76),
            TrimTolerances(rotor_speed_rel=0.1, pitch_deg=2.0),
        )
        assert check.status == "pass"
        assert check.tolerances.pitch_deg == 2.0

    def test_parked_curve_is_unavailable_for_a_spinning_rotor(self) -> None:
        """A zero-speed curve value with a spinning settle cannot be judged relatively.

        The relative rotor-speed check flags a non-zero settled speed against a zero
        curve speed; a fully zero pair passes.
        """
        from vane.run import OperatingPoint, compare_settled_to_curve

        parked = compare_settled_to_curve(
            OperatingPoint(0.0, 0.0, 90.0), OperatingPoint(0.0, 0.0, 90.0)
        )
        assert parked.status == "pass"
        spinning = compare_settled_to_curve(
            OperatingPoint(3.0, 5.0, 0.0), OperatingPoint(3.0, 0.0, 0.0)
        )
        assert spinning.status == "flag"

    def test_different_wind_speeds_rejected(self) -> None:
        """Points at different winds cannot be compared."""
        from vane.run import OperatingPoint, compare_settled_to_curve

        with pytest.raises(ValueError, match="cannot compare"):
            compare_settled_to_curve(
                OperatingPoint(12.0, 7.0, 8.0), OperatingPoint(11.0, 7.0, 8.0)
            )

    def test_unavailable_when_a_value_is_missing(self) -> None:
        """A non-finite compared value makes the check unavailable, not flagged."""
        from vane.run import QuantityCheck, SettleCurveCheck, TrimTolerances

        check = SettleCurveCheck(
            rotor_speed=QuantityCheck(7.5, 7.5, 0.01, relative=True),
            pitch=QuantityCheck(float("nan"), 3.0, 0.5, relative=False),
            tolerances=TrimTolerances(),
        )
        assert check.status == "unavailable"
