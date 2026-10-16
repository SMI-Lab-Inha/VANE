# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for the two-stage damped settle and the mean-drift settle criterion."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pytest

from vane.io.fst_writer import FastInputFile
from vane.run import (
    EquilibriumTolerances,
    LinearizationSettings,
    SettlePlatformDamping,
    SteadyStateError,
    SweepSettleError,
    run_operating_sweep,
    solve_steady_state,
)
from vane.run import steady_state as steady_state_module
from vane.run.runner import OpenFastResult
from vane.run.steady_state import (
    _add_platform_damping,
    _mean_drift,
    _oscillation_period,
    _residual_damping_ratio,
    _whole_periods,
)

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
600 TMax - Total run time (s)
0 TStart - output start (s)
1 OutFileFmt - output format
"default" DT_Out - output step
1 CompInflow - inflow switch
1 CompHydro - HydroDyn switch
False Linearize - flag
False CalcSteady - flag
0 WrVTK - VTK output
"ED.dat" EDFile - ElastoDyn file
"IW.dat" InflowFile - InflowWind file
"HD.dat" HydroFile - HydroDyn file
"""

_ED = """\
0.0 RotSpeed - rpm
0.0 BlPitch(1) - deg
0.0 BlPitch(2) - deg
0.0 BlPitch(3) - deg
0.0 TTDspFA - m
0.0 TTDspSS - m
0.0 OoPDefl - m
0.0 IPDefl - m
0 PtfmSurge - m
0 PtfmSway - m
0 PtfmHeave - m
0 PtfmRoll - deg
0 PtfmPitch - deg
0 PtfmYaw - deg
OutList - outputs
"RotSpeed"
END of OutList
"""

_IW = """\
1 WindType - switch
8 HWindSpeed - wind speed (m/s)
0.0 PLExp - power law exponent
"""

# A HydroDyn excerpt laid out as published decks are: AddF0 vector, AddCLin, AddBLin
# (with a non-zero roll entry to check it is added to, not overwritten) and AddBQuad.
_ROW_ZERO = "  0  0  0  0  0  0"
_HD = f"""\
------- HydroDyn Input File ----
             1   NBody       - Number of WAMIT bodies
             1   NBodyMod    - Body coupling model
---------------------- PLATFORM ADDITIONAL STIFFNESS AND DAMPING ------
           0.0   AddF0    - Additional preload
           0.0
           0.0
           0.0
           0.0
           0.0
{_ROW_ZERO}   AddCLin  - Additional linear stiffness
{_ROW_ZERO}
{_ROW_ZERO}
{_ROW_ZERO}
{_ROW_ZERO}
{_ROW_ZERO}
{_ROW_ZERO}   AddBLin  - Additional linear damping
{_ROW_ZERO}
{_ROW_ZERO}
  0  0  0  5.0E+08  0  0
{_ROW_ZERO}
{_ROW_ZERO}
9.23E+05 0 0 0 -8.92E+06 0 AddBQuad - Additional quadratic drag
{_ROW_ZERO}
{_ROW_ZERO}
{_ROW_ZERO}
{_ROW_ZERO}
{_ROW_ZERO}
---------------------- STRIP THEORY OPTIONS ------
"""

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


def _out(
    tmax: float,
    tstart: float,
    *,
    surge: float = 17.0,
    amplitude: float = 0.0,
    period: float = 100.0,
    drift: float = 0.0,
    damping_ratio: float = 0.0,
) -> str:
    """Return a ``.out`` from ``tstart`` to ``tmax`` (1 s output step).

    The surge oscillates about ``surge`` (plus a linear ``drift`` per second) with
    ``amplitude`` and ``period``, its envelope decaying as ``exp(-zeta omega t)`` for
    ``damping_ratio`` ``zeta`` (growing when negative); the surge velocity is its
    derivative.
    """
    lines = [
        "OpenFAST results header line",
        "",
        "\t".join(_CHANNELS),
        "\t".join("(-)" for _ in _CHANNELS),
    ]
    omega = 2.0 * math.pi / period
    for step in range(int(tstart), int(tmax) + 1):
        time = float(step)
        envelope = amplitude * math.exp(-damping_ratio * omega * time)
        position = surge + drift * time + envelope * math.sin(omega * time)
        velocity = drift + envelope * omega * (
            math.cos(omega * time) - damping_ratio * math.sin(omega * time)
        )
        row = (time, 7.5, 6.3, position, -0.46, 3.4, 0.9, 10.1, -1.2, velocity, 0.0)
        lines.append("\t".join(f"{value:.6f}" for value in row))
    return "\n".join(lines) + "\n"


def _make_case(tmp_path: Path) -> Path:
    """Write a floating settle deck with ElastoDyn, InflowWind and HydroDyn."""
    (tmp_path / "ED.dat").write_text(_ED, encoding="utf-8")
    (tmp_path / "IW.dat").write_text(_IW, encoding="utf-8")
    (tmp_path / "HD.dat").write_text(_HD, encoding="utf-8")
    fst = tmp_path / "Model.fst"
    fst.write_text(_PRIMARY, encoding="utf-8")
    return fst


def _damping_diagonal(hydro: Path) -> list[float]:
    """Return the diagonal of the ``AddBLin`` matrix in ``hydro``."""
    lines = hydro.read_text(encoding="utf-8", errors="replace").splitlines()
    start = next(i for i, line in enumerate(lines) if "AddBLin" in line.split())
    return [float(lines[start + i].split()[i]) for i in range(6)]


class _FakeOpenFast:
    """Fake OpenFAST: records each launch and writes a stage-dependent ``.out``."""

    def __init__(self, damped: dict[str, float], confirm: dict[str, float]) -> None:
        self.damped = damped
        self.confirm = confirm
        self.launches: list[dict[str, object]] = []

    def __call__(self, fst_path: Path, **_: object) -> OpenFastResult:
        deck = FastInputFile.read(fst_path)
        tmax = float(deck.get("TMax"))
        tstart = float(deck.get("TStart"))
        diagonal = _damping_diagonal(fst_path.parent / "HD.dat")
        elastodyn = FastInputFile.read(fst_path.parent / "ED.dat")
        damped = diagonal[0] > 0.0
        self.launches.append(
            {
                "damped": damped,
                "diagonal": diagonal,
                "tmax": tmax,
                "rotor": float(elastodyn.get("RotSpeed")),
                "surge": float(elastodyn.get("PtfmSurge")),
                "ttdspfa": float(elastodyn.get("TTDspFA")),
                "oopdefl": float(elastodyn.get("OoPDefl")),
                "ipdefl": float(elastodyn.get("IPDefl")),
                "outlist": (fst_path.parent / "ED.dat").read_text(encoding="utf-8"),
            }
        )
        spec = self.damped if damped else self.confirm
        out = _out(tmax, tstart, **spec)
        fst_path.with_suffix(".out").write_text(out, encoding="utf-8")
        return OpenFastResult(fst_path=fst_path, returncode=0)


_DAMPING = SettlePlatformDamping(surge=2.0e6, sway=2.0e6, roll=1.0e8, yaw=4.0e9)


def _settle(
    fst: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake: _FakeOpenFast,
    **kwargs: object,
) -> object:
    monkeypatch.setattr(steady_state_module, "run_openfast", fake)
    options: dict[str, object] = {
        "settle_time": 600.0,
        "discard_time": 300.0,
        "rotor_speed_rpm": 7.0,
        "blade_pitch_deg": 5.0,
        "platform_damping": _DAMPING,
        "confirm_time": 400.0,
        **kwargs,
    }
    return solve_steady_state(fst, 12.0, **options)  # type: ignore[arg-type]


class TestSettlePlatformDamping:
    """Validation and sizing of the temporary platform damping."""

    def test_diagonal_order(self) -> None:
        """The diagonal follows HydroDyn's surge..yaw order."""
        damping = SettlePlatformDamping(surge=1.0, heave=3.0, yaw=6.0)
        assert damping.diagonal() == (1.0, 0.0, 3.0, 0.0, 0.0, 6.0)

    @pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
    def test_invalid_value_rejected(self, value: float) -> None:
        """A negative or non-finite damping is rejected."""
        with pytest.raises(ValueError, match="surge"):
            SettlePlatformDamping(surge=value)

    def test_all_zero_rejected(self) -> None:
        """Damping that damps nothing is rejected."""
        with pytest.raises(ValueError, match="at least one"):
            SettlePlatformDamping()

    def test_from_critical_fraction(self) -> None:
        """Damping is 4 pi zeta m / T per DOF; DOFs not given stay undamped."""
        damping = SettlePlatformDamping.from_critical_fraction(
            0.5,
            mass={"surge": 3.3e7, "yaw": 5.0e10},
            natural_period={"surge": 100.0, "yaw": 80.0},
        )
        assert damping.surge == pytest.approx(4 * math.pi * 0.5 * 3.3e7 / 100.0)
        assert damping.yaw == pytest.approx(4 * math.pi * 0.5 * 5.0e10 / 80.0)
        assert damping.sway == 0.0
        # Equivalent to 2 zeta sqrt(k m) with k = m (2 pi / T)^2.
        stiffness = 3.3e7 * (2 * math.pi / 100.0) ** 2
        assert damping.surge == pytest.approx(2 * 0.5 * math.sqrt(stiffness * 3.3e7))

    def test_from_critical_fraction_per_dof(self) -> None:
        """A mapping gives each DOF its own fraction."""
        damping = SettlePlatformDamping.from_critical_fraction(
            {"surge": 0.2, "sway": 0.4},
            mass={"surge": 1.0e7, "sway": 1.0e7},
            natural_period={"surge": 100.0, "sway": 100.0},
        )
        assert damping.sway == pytest.approx(2.0 * damping.surge)

    @pytest.mark.parametrize(
        ("fraction", "mass", "period", "match"),
        [
            (0.5, {"bogus": 1.0}, {"bogus": 1.0}, "unknown platform DOF"),
            (0.5, {"surge": 1.0}, {}, "no natural_period"),
            ({"sway": 0.5}, {"surge": 1.0}, {"surge": 1.0}, "no critical-damping"),
            (0.5, {"surge": -1.0}, {"surge": 1.0}, "mass"),
            (0.5, {"surge": 1.0}, {"surge": 0.0}, "natural_period"),
            (float("nan"), {"surge": 1.0}, {"surge": 1.0}, "fraction"),
        ],
    )
    def test_from_critical_fraction_rejects_bad_input(
        self,
        fraction: float | dict[str, float],
        mass: dict[str, float],
        period: dict[str, float],
        match: str,
    ) -> None:
        """Unknown DOFs, missing periods and non-positive inputs are rejected."""
        with pytest.raises(ValueError, match=match):
            SettlePlatformDamping.from_critical_fraction(
                fraction, mass=mass, natural_period=period
            )


class TestEquilibriumTolerances:
    """Tolerances on a change of the settled means."""

    def test_exceeded(self) -> None:
        """Each quantity is compared against its own tolerance."""
        tolerances = EquilibriumTolerances()
        changes = {
            "surge": 0.05,
            "pitch": 0.06,
            "rotor_speed_rpm": 0.03,
            "blade_pitch_deg": -0.2,
            "unrelated": 99.0,
        }
        assert tolerances.exceeded(changes, 7.5) == ("pitch", "blade_pitch_deg")
        # The rotor-speed tolerance is relative, floored at 1 rpm.
        assert tolerances.exceeded({"rotor_speed_rpm": 0.004}, 0.0) == ()
        assert tolerances.exceeded({"rotor_speed_rpm": 0.006}, 0.0) == (
            "rotor_speed_rpm",
        )

    def test_nan_change_is_exceeded(self) -> None:
        """An unmeasurable change is never treated as within tolerance."""
        assert EquilibriumTolerances().exceeded({"surge": float("nan")}, 7.5) == (
            "surge",
        )

    @pytest.mark.parametrize("value", [0.0, -1.0, float("inf")])
    def test_invalid_tolerance_rejected(self, value: float) -> None:
        """Tolerances must be positive and finite."""
        with pytest.raises(ValueError, match="translation_m"):
            EquilibriumTolerances(translation_m=value)


class TestAddPlatformDamping:
    """Editing HydroDyn's AddBLin diagonal."""

    def test_adds_to_the_diagonal_and_keeps_everything_else(
        self, tmp_path: Path
    ) -> None:
        """Only the diagonal entries change; existing damping is added to."""
        path = tmp_path / "HD.dat"
        edited = _add_platform_damping(_HD, _DAMPING, path)
        path.write_text(edited, encoding="utf-8")
        assert _damping_diagonal(path) == pytest.approx(
            [2.0e6, 2.0e6, 0.0, 6.0e8, 0.0, 4.0e9]
        )
        original = _HD.splitlines()
        changed = [
            i for i, line in enumerate(edited.splitlines()) if line != original[i]
        ]
        start = next(i for i, line in enumerate(original) if "AddBLin" in line)
        assert changed == [start, start + 1, start + 3, start + 5]
        # The label and its description survive on the first row.
        assert edited.splitlines()[start].endswith(
            "AddBLin  - Additional linear damping"
        )

    def test_crlf_line_endings_preserved(self, tmp_path: Path) -> None:
        """A Windows-style deck keeps its line endings."""
        text = _HD.replace("\n", "\r\n")
        edited = _add_platform_damping(text, _DAMPING, tmp_path / "HD.dat")
        assert edited.count("\r\n") == text.count("\r\n")

    def test_multi_body_damps_the_first_body_only(self, tmp_path: Path) -> None:
        """With NBodyMod > 1 there are NBody 6x6 matrices; body 1 gets the damping."""
        body_rows = "\n".join([_ROW_ZERO] * 11)
        text = (
            "  2   NBody - bodies\n  2   NBodyMod - coupling\n"
            f"{_ROW_ZERO}   AddBLin - damping\n{body_rows}\n"
            f"{_ROW_ZERO}   AddBQuad - drag\n" + "\n".join([_ROW_ZERO] * 11) + "\n"
        )
        edited = _add_platform_damping(
            text, SettlePlatformDamping(surge=1.0), tmp_path / "HD.dat"
        )
        rows = edited.splitlines()[2:14]
        assert float(rows[0].split()[0]) == 1.0
        assert all(float(row.split()[0]) == 0.0 for row in rows[1:])

    def test_coupled_multi_body_needs_wide_rows(self, tmp_path: Path) -> None:
        """NBodyMod = 1 with two bodies needs 12x12; 6-column rows are rejected."""
        text = (
            "  2   NBody - bodies\n  1   NBodyMod - coupling\n"
            + _HD.split("---------------------- PLATFORM", 1)[1].split("\n", 1)[1]
        )
        with pytest.raises(ValueError, match="12x12 AddBLin"):
            _add_platform_damping(text, _DAMPING, tmp_path / "HD.dat")

    def test_missing_matrix_rejected(self, tmp_path: Path) -> None:
        """A HydroDyn file without AddBLin cannot be damped."""
        with pytest.raises(ValueError, match="no AddBLin"):
            _add_platform_damping("1 NBody - x\n", _DAMPING, tmp_path / "HD.dat")

    def test_label_not_on_first_row_rejected(self, tmp_path: Path) -> None:
        """A layout whose label is not on the first row fails instead of misediting."""
        lines = _HD.splitlines()
        start = next(i for i, line in enumerate(lines) if "AddBLin" in line.split())
        lines[start] = _ROW_ZERO
        lines[start + 3] = f"{_ROW_ZERO}   AddBLin - damping"
        with pytest.raises(ValueError, match="cannot locate"):
            _add_platform_damping("\n".join(lines), _DAMPING, tmp_path / "HD.dat")


class TestTwoStageSettle:
    """Settle with temporary damping, then confirm the equilibrium without it."""

    def test_non_utf8_hydrodyn_comment_is_preserved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A Windows-1252 byte in HydroDyn does not stop the damped stage.

        The damping was injected through a strict UTF-8 decode, so such a deck
        raised ``UnicodeDecodeError`` and lost the accelerator.
        """
        fst = _make_case(tmp_path)
        hydro = tmp_path / "HD.dat"
        hydro.write_bytes(b"! water at 15 \xb0C\n" + hydro.read_bytes())
        original = hydro.read_bytes()
        fake = _FakeOpenFast(damped={"surge": 17.0}, confirm={"surge": 17.02})
        _settle(fst, monkeypatch, fake)
        assert [launch["damped"] for launch in fake.launches] == [True, False]
        assert hydro.read_bytes() == original

    def test_confirmed_equilibrium(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Stage 1 is damped, stage 2 is not and starts from the stage-1 means."""
        fst = _make_case(tmp_path)
        original = (tmp_path / "HD.dat").read_bytes()
        fake = _FakeOpenFast(damped={"surge": 17.0}, confirm={"surge": 17.02})
        op = _settle(fst, monkeypatch, fake)

        assert [launch["damped"] for launch in fake.launches] == [True, False]
        assert fake.launches[0]["diagonal"] == pytest.approx(
            [2.0e6, 2.0e6, 0.0, 6.0e8, 0.0, 4.0e9]
        )
        assert fake.launches[1]["diagonal"] == pytest.approx(
            [0.0, 0.0, 0.0, 5.0e8, 0.0, 0.0]
        )
        # The deck's own AddBLin is restored byte for byte.
        assert (tmp_path / "HD.dat").read_bytes() == original
        # Stage 2 starts from the stage-1 means, tower-top deflection included.
        confirm = fake.launches[1]
        assert confirm["tmax"] == 400.0
        assert confirm["rotor"] == pytest.approx(7.5)
        assert confirm["surge"] == pytest.approx(17.0, abs=1e-3)
        assert confirm["ttdspfa"] == pytest.approx(0.9)
        assert confirm["oopdefl"] == pytest.approx(10.1)
        assert confirm["ipdefl"] == pytest.approx(-1.2)
        assert "TTDspFA" in str(confirm["outlist"])
        # Blade-tip deflections are seeded when output but never added to the OutList.
        assert "OoPDefl1" not in str(confirm["outlist"])

        convergence = op.settle_convergence  # type: ignore[attr-defined]
        assert convergence.converged
        assert convergence.criterion == "velocity"
        assert [a.stage for a in convergence.attempts] == ["damped", "confirm"]
        assert [a.iteration for a in convergence.attempts] == [1, 2]
        assert convergence.iterations == 2
        # The confirmation skips its first quarter (the restart transient).
        assert (convergence.window_start, convergence.window_end) == (100.0, 400.0)
        confirmation = convergence.confirmation
        assert confirmation is not None
        assert confirmation.confirmed
        assert confirmation.damping == _DAMPING
        assert confirmation.shift["surge"] == pytest.approx(0.02, abs=1e-3)
        assert confirmation.shift_flagged == ()
        assert confirmation.stable
        # A confirmed point is the damped stage's equilibrium.
        assert op.platform.surge == pytest.approx(17.0, abs=1e-3)  # type: ignore[attr-defined]
        json.dumps(asdict(convergence), allow_nan=False)

    def test_shifted_equilibrium_is_reported_not_accepted(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An undamped mean that moved beyond tolerance fails the confirmation."""
        fst = _make_case(tmp_path)
        fake = _FakeOpenFast(damped={"surge": 17.0}, confirm={"surge": 17.5})
        op = _settle(fst, monkeypatch, fake)
        convergence = op.settle_convergence  # type: ignore[attr-defined]
        assert not convergence.converged
        assert convergence.confirmation.confirm_converged
        assert convergence.confirmation.shift_flagged == ("surge",)
        assert "moved the equilibrium" in caplog.text

    def test_shifted_equilibrium_raises_when_required(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """require_settled turns a failed confirmation into an error."""
        fst = _make_case(tmp_path)
        fake = _FakeOpenFast(damped={"surge": 17.0}, confirm={"surge": 17.5})
        with pytest.raises(SteadyStateError, match="moved the equilibrium") as error:
            _settle(fst, monkeypatch, fake, require_settled=True)
        assert error.value.convergence is not None
        assert error.value.convergence.confirmation is not None

    def test_returning_oscillation_is_reported(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A sustained oscillation that reappears without damping is not accepted."""
        fst = _make_case(tmp_path)
        fake = _FakeOpenFast(
            damped={"surge": 17.0},
            confirm={"surge": 17.0, "amplitude": 2.0, "period": 100.0},
        )
        op = _settle(fst, monkeypatch, fake, criterion="velocity")
        convergence = op.settle_convergence  # type: ignore[attr-defined]
        assert not convergence.converged
        assert not convergence.confirmation.confirm_converged
        assert convergence.translational_velocity_rms > 0.05
        assert convergence.oscillation_period == pytest.approx(100.0, rel=0.1)
        confirmation = convergence.confirmation
        assert not confirmation.stable
        assert confirmation.residual_damping_ratio == pytest.approx(0.0, abs=2e-3)
        assert "still moving and not decaying" in caplog.text
        assert "dominant period" in caplog.text
        # An unconfirmed point reports the undamped run's own (whole-period) mean.
        assert op.platform.surge == pytest.approx(17.0, abs=0.05)  # type: ignore[attr-defined]

    def test_decaying_residual_confirms_a_stable_equilibrium(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A restart transient that rings down counts as a stable equilibrium."""
        fst = _make_case(tmp_path)
        fake = _FakeOpenFast(
            damped={"surge": 17.0},
            confirm={"surge": 17.0, "amplitude": 1.0, "damping_ratio": 0.03},
        )
        op = _settle(fst, monkeypatch, fake, require_settled=True)
        convergence = op.settle_convergence  # type: ignore[attr-defined]
        assert convergence.converged
        assert convergence.translational_velocity_rms > 0.01
        confirmation = convergence.confirmation
        assert not confirmation.confirm_converged
        assert confirmation.stable
        assert confirmation.confirmed
        assert confirmation.residual_damping_ratio == pytest.approx(0.03, rel=0.25)

    def test_growing_residual_is_accepted_and_flagged_by_auto(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Accept, under "auto", a growing residual about the reproduced equilibrium.

        Linear damping cannot move an equilibrium, so the damped point is the
        undamped model's even when the closed loop with the controller is unstable;
        the point is flagged ``growing_oscillation`` instead of being dropped.
        """
        fst = _make_case(tmp_path)
        fake = _FakeOpenFast(
            damped={"surge": 17.0},
            confirm={"surge": 17.0, "amplitude": 0.5, "damping_ratio": -0.04},
        )
        op = _settle(fst, monkeypatch, fake, require_settled=True)
        convergence = op.settle_convergence  # type: ignore[attr-defined]
        assert convergence.converged
        assert convergence.growing_oscillation
        assert convergence.confirmation.growing_oscillation

    def test_growing_residual_fails_the_velocity_criterion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The explicit "velocity" criterion still rejects a growing residual."""
        fst = _make_case(tmp_path)
        fake = _FakeOpenFast(
            damped={"surge": 17.0},
            confirm={"surge": 17.0, "amplitude": 0.5, "damping_ratio": -0.04},
        )
        with pytest.raises(SteadyStateError, match=r"damping ratio -0\.0"):
            _settle(fst, monkeypatch, fake, require_settled=True, criterion="velocity")

    def test_slow_growth_fails_outside_the_auto_criterion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only "auto" accepts a bounded oscillation; "velocity" still rejects it."""
        fst = _make_case(tmp_path)
        fake = _FakeOpenFast(
            damped={"surge": 17.0},
            confirm={"surge": 17.0, "amplitude": 0.5, "damping_ratio": -0.01},
        )
        with pytest.raises(SteadyStateError, match="only 'auto' accepts one"):
            _settle(fst, monkeypatch, fake, require_settled=True, criterion="velocity")

    def test_decay_does_not_count_for_the_mean_drift_criterion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With mean drift the confirmation must meet that criterion itself."""
        fst = _make_case(tmp_path)
        fake = _FakeOpenFast(
            damped={"surge": 17.0},
            confirm={"surge": 17.0, "drift": 0.003, "damping_ratio": 0.03},
        )
        op = _settle(fst, monkeypatch, fake, criterion="mean_drift")
        confirmation = op.settle_convergence.confirmation  # type: ignore[attr-defined]
        assert not confirmation.stable
        assert not confirmation.confirmed

    def test_unconverged_damped_stage_is_reported(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A damped stage that never settles fails the settle even if stage 2 holds."""
        fst = _make_case(tmp_path)
        fake = _FakeOpenFast(
            damped={"surge": 17.0, "amplitude": 1.0}, confirm={"surge": 17.0}
        )
        op = _settle(fst, monkeypatch, fake, criterion="velocity")
        convergence = op.settle_convergence  # type: ignore[attr-defined]
        assert not convergence.converged
        assert not convergence.confirmation.damped_converged
        assert "damped stage did not settle" in caplog.text

    def test_adaptive_damped_stage_then_one_confirmation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The damped stage grows adaptively; the confirmation is a single run."""
        fst = _make_case(tmp_path)
        fake = _FakeOpenFast(
            damped={"surge": 17.0, "amplitude": 1.0}, confirm={"surge": 17.0}
        )
        op = _settle(
            fst,
            monkeypatch,
            fake,
            adaptive=True,
            max_settle_time=900.0,
            criterion="velocity",
        )
        convergence = op.settle_convergence  # type: ignore[attr-defined]
        assert [launch["tmax"] for launch in fake.launches] == [600.0, 900.0, 400.0]
        assert [a.stage for a in convergence.attempts] == [
            "damped",
            "damped",
            "confirm",
        ]
        assert convergence.confirmation.damped_settle_time == 900.0

    def test_damping_removed_when_stage_one_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed damped run still restores the deck's own AddBLin."""
        fst = _make_case(tmp_path)
        original = (tmp_path / "HD.dat").read_bytes()

        def failing_run(fst_path: Path, **_: object) -> OpenFastResult:
            assert _damping_diagonal(fst_path.parent / "HD.dat")[0] > 0.0
            return OpenFastResult(fst_path=fst_path, returncode=1, stdout="boom")

        monkeypatch.setattr(steady_state_module, "run_openfast", failing_run)
        with pytest.raises(SteadyStateError, match="settle run failed"):
            solve_steady_state(fst, 12.0, platform_damping=_DAMPING)
        assert (tmp_path / "HD.dat").read_bytes() == original

    def test_hard_linked_template_is_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The damping edit replaces the file, never writing through a hard link."""
        template = tmp_path / "template_HD.dat"
        template.write_text(_HD, encoding="utf-8")
        fst = _make_case(tmp_path)
        (tmp_path / "HD.dat").unlink()
        try:
            (tmp_path / "HD.dat").hardlink_to(template)
        except OSError:  # pragma: no cover - filesystem without hard links
            pytest.skip("hard links unsupported")
        fake = _FakeOpenFast(damped={"surge": 17.0}, confirm={"surge": 17.0})
        _settle(fst, monkeypatch, fake)
        assert template.read_text(encoding="utf-8") == _HD
        assert fake.launches[0]["damped"]

    def test_requires_hydrodyn_before_launch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without an active HydroDyn the damping is rejected before any run."""
        fst = _make_case(tmp_path)
        fst.write_text(_PRIMARY.replace("1 CompHydro", "0 CompHydro"), encoding="utf-8")
        fake = _FakeOpenFast(damped={}, confirm={})
        with pytest.raises(ValueError, match="active HydroDyn"):
            _settle(fst, monkeypatch, fake)
        assert fake.launches == []

    def test_matrix_layout_checked_before_launch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A HydroDyn without an AddBLin matrix is rejected before any run."""
        fst = _make_case(tmp_path)
        (tmp_path / "HD.dat").write_text("1 NBody - x\n", encoding="utf-8")
        fake = _FakeOpenFast(damped={}, confirm={})
        with pytest.raises(ValueError, match="no AddBLin"):
            _settle(fst, monkeypatch, fake)
        assert fake.launches == []

    @pytest.mark.parametrize("value", [0.0, float("nan")])
    def test_invalid_confirm_time_rejected(self, tmp_path: Path, value: float) -> None:
        """The confirmation run length must be positive and finite."""
        fst = _make_case(tmp_path)
        with pytest.raises(ValueError, match="confirm_time"):
            solve_steady_state(fst, 12.0, platform_damping=_DAMPING, confirm_time=value)


class TestMeanDriftCriterion:
    """A stationary oscillation about a steady mean, judged by mean drift."""

    def _run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        **spec: float,
    ) -> object:
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
            criterion="mean_drift",
            equilibrium_tolerances=EquilibriumTolerances(translation_m=0.2),
        )

    def test_stationary_oscillation_accepted_and_reported(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The mean is the operating point; the oscillation is still reported."""
        op = self._run(tmp_path, monkeypatch, amplitude=2.0, period=100.0)
        convergence = op.settle_convergence  # type: ignore[attr-defined]
        assert convergence.converged
        assert convergence.criterion == "mean_drift"
        assert convergence.translational_velocity_rms > 0.05
        assert convergence.oscillation_period == pytest.approx(100.0, rel=0.1)
        assert abs(convergence.mean_drift["surge"]) < 0.2
        assert convergence.attempts[0].mean_drift == convergence.mean_drift
        assert op.platform.surge == pytest.approx(17.0, abs=0.1)  # type: ignore[attr-defined]
        assert "sustained oscillation" in caplog.text

    def test_drifting_mean_is_not_settled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A mean still creeping between window halves is rejected."""
        op = self._run(tmp_path, monkeypatch, drift=0.002)
        convergence = op.settle_convergence  # type: ignore[attr-defined]
        assert not convergence.converged
        assert convergence.mean_drift["surge"] == pytest.approx(0.6, rel=0.02)

    def test_velocity_criterion_rejects_the_same_oscillation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The velocity criterion never accepts a sustained oscillation."""
        fst = _make_case(tmp_path)

        def fake_run(fst_path: Path, **_: object) -> OpenFastResult:
            deck = FastInputFile.read(fst_path)
            out = _out(
                float(deck.get("TMax")), float(deck.get("TStart")), amplitude=2.0
            )
            fst_path.with_suffix(".out").write_text(out, encoding="utf-8")
            return OpenFastResult(fst_path=fst_path, returncode=0)

        monkeypatch.setattr(steady_state_module, "run_openfast", fake_run)
        op = solve_steady_state(
            fst, 12.0, settle_time=1000.0, discard_time=400.0, criterion="velocity"
        )
        assert op.settle_convergence is not None
        assert not op.settle_convergence.converged
        assert op.settle_convergence.criterion == "velocity"

    def test_unknown_criterion_rejected(self, tmp_path: Path) -> None:
        """Only the two documented criteria are accepted."""
        fst = _make_case(tmp_path)
        with pytest.raises(ValueError, match="criterion"):
            solve_steady_state(fst, 12.0, criterion="bogus")  # type: ignore[arg-type]

    def test_single_row_window_is_unassessed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Too short a window cannot judge drift; require_settled rejects it."""
        fst = _make_case(tmp_path)

        def fake_run(fst_path: Path, **_: object) -> OpenFastResult:
            fst_path.with_suffix(".out").write_text(
                _out(600.0, 600.0), encoding="utf-8"
            )
            return OpenFastResult(fst_path=fst_path, returncode=0)

        monkeypatch.setattr(steady_state_module, "run_openfast", fake_run)
        with pytest.raises(SteadyStateError, match="compare the analysis-window"):
            solve_steady_state(fst, 12.0, criterion="mean_drift", require_settled=True)


class TestWholePeriodAveraging:
    """Means of an oscillation over whole periods are free of phase bias."""

    _COLUMNS: ClassVar[list[str]] = ["Time", "PtfmSurge", "PtfmTVxi"]

    def _window(self, period: float) -> np.ndarray:
        time = np.arange(300.0, 900.0)
        omega = 2 * np.pi / period
        return np.column_stack(
            [
                time,
                17.0 + 2.0 * np.sin(omega * time),
                2.0 * omega * np.cos(omega * time),
            ]
        )

    def test_half_means_use_whole_periods(self) -> None:
        """A stationary oscillation shows no drift once halves span whole cycles."""
        window = self._window(130.0)
        naive = _mean_drift(self._COLUMNS, window)
        synchronous = _mean_drift(self._COLUMNS, window, 130.0)
        assert naive is not None
        assert synchronous is not None
        assert abs(naive["surge"]) > 0.2
        assert abs(synchronous["surge"]) < 0.02

    def test_whole_periods_trims_to_the_trailing_cycles(self) -> None:
        """Two whole 130 s cycles are kept from a 600 s window."""
        window = self._window(130.0)
        kept = _whole_periods(self._COLUMNS, window, 520.0 / 4)
        assert kept.shape[0] == 520
        assert kept[-1, 0] == window[-1, 0]
        assert float(kept[:, 1].mean()) == pytest.approx(17.0, abs=1e-6)

    def test_unknown_or_long_period_keeps_every_row(self) -> None:
        """No period, a period longer than the window, or no time column: unchanged."""
        window = self._window(130.0)
        assert _whole_periods(self._COLUMNS, window, None) is window
        assert _whole_periods(self._COLUMNS, window, 1000.0) is window
        assert _whole_periods(["Surge"], window[:, 1:2], 130.0).shape[0] == 600


class TestResidualDampingRatio:
    """Damping ratio of the residual oscillation from its cycle-by-cycle decay."""

    def test_unavailable_cases(self) -> None:
        """No period, under two cycles, no velocity channel or no motion: ``None``."""
        time = np.arange(0.0, 300.0)
        moving = np.column_stack([time, np.sin(2 * np.pi * time / 100.0)])
        still = np.column_stack([time, np.zeros_like(time)])
        columns = ["Time", "PtfmTVxi"]
        assert _residual_damping_ratio(columns, moving, None) is None
        assert _residual_damping_ratio(columns, moving, 200.0) is None
        assert _residual_damping_ratio(["Time", "RotSpeed"], moving, 100.0) is None
        assert _residual_damping_ratio(columns, still, 100.0) is None

    def test_angular_motion_wins_over_translational_noise(self) -> None:
        """A rocking platform with quiet translations is judged by its rotation.

        The translational group always exists, so it used to be chosen even when
        it carried only numerical noise: the period and damping then came from the
        noise and a sustained pitch oscillation could not be recognised.
        """
        rng = np.random.default_rng(0)
        time = np.arange(0.0, 1200.0, 0.5)
        pitch_rate = 0.05 * np.sin(2 * np.pi * time / 30.0)  # deg/s, 30 s period
        noise = 1e-7 * rng.standard_normal((time.size, 3))  # m/s
        columns = ["Time", "PtfmTVxi", "PtfmTVyi", "PtfmTVzi", "PtfmRVyi"]
        window = np.column_stack([time, noise, pitch_rate])
        period = _oscillation_period(columns, window)
        assert period == pytest.approx(30.0, rel=0.02)
        residual = _residual_damping_ratio(columns, window, period)
        assert residual is not None
        assert abs(residual) < 1e-3  # sustained, not noise-driven

    def test_angular_channel_used_without_translational(self) -> None:
        """An angular-velocity decay is measured when no translation is output."""
        time = np.arange(0.0, 400.0)
        omega = 2 * np.pi / 50.0
        signal = np.exp(-0.02 * omega * time) * np.cos(omega * time)
        ratio = _residual_damping_ratio(
            ["Time", "PtfmRVyi"], np.column_stack([time, signal]), 50.0
        )
        assert ratio == pytest.approx(0.02, rel=0.1)


class TestOscillationPeriod:
    """Dominant period of the platform motion."""

    def test_angular_channels_used_without_translational(self) -> None:
        """With only angular velocities output, they give the period."""
        time = np.arange(0.0, 400.0)
        window = np.column_stack([time, np.sin(2 * np.pi * time / 25.0)])
        assert _oscillation_period(["Time", "PtfmRVyi"], window) == pytest.approx(
            25.0, rel=0.05
        )

    def test_unavailable_cases(self) -> None:
        """No motion, no velocity channel, or too few rows give ``None``."""
        time = np.arange(0.0, 50.0)
        still = np.column_stack([time, np.zeros_like(time)])
        assert _oscillation_period(["Time", "PtfmTVxi"], still) is None
        assert _oscillation_period(["Time", "RotSpeed"], still) is None
        assert _oscillation_period(["Time", "PtfmTVxi"], still[:3]) is None
        frozen = np.column_stack([np.zeros(20), np.arange(20.0)])
        assert _oscillation_period(["Time", "PtfmTVxi"], frozen) is None


# --- run_operating_sweep -------------------------------------------------------------

_SWEEP_PRIMARY = """\
False Linearize - flag
False CalcSteady - flag
1 NLinTimes - -
0 LinTimes - s
3 TrimCase - -
0.01 TrimGain - -
0.001 TrimTol - -
1 CompInflow - flag
1 CompHydro - flag
600 TMax - s
0 TStart - s
"ED.dat" EDFile - ElastoDyn file
"IW.dat" InflowFile - InflowWind file
"HD.dat" HydroFile - HydroDyn file
"""


def _make_template(tmp_path: Path) -> Path:
    root = tmp_path / "template"
    root.mkdir()
    (root / "ED.dat").write_text(_ED, encoding="utf-8")
    (root / "IW.dat").write_text(_IW, encoding="utf-8")
    (root / "HD.dat").write_text(_HD, encoding="utf-8")
    fst = root / "Model.fst"
    fst.write_text(_SWEEP_PRIMARY, encoding="utf-8")
    return fst


class TestSweepSettleAccelerator:
    """run_operating_sweep threads the accelerator to the settle copies only."""

    def test_damping_never_reaches_the_linearization_case(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The settle copy is damped; the linearization case keeps the deck's file."""
        from vane.run import driver as driver_module

        fst = _make_template(tmp_path)
        fake = _FakeOpenFast(damped={"surge": 17.0}, confirm={"surge": 17.0})
        lin_hydro: list[bytes] = []

        def fake_lin_run(deck_path: Path, **_: object) -> OpenFastResult:
            lin_hydro.append((deck_path.parent / "HD.dat").read_bytes())
            return OpenFastResult(
                fst_path=deck_path,
                returncode=0,
                lin_files=[deck_path.with_suffix(".1.lin")],
            )

        monkeypatch.setattr(steady_state_module, "run_openfast", fake)
        monkeypatch.setattr(driver_module, "run_openfast", fake_lin_run)
        result = run_operating_sweep(
            fst,
            tmp_path / "out",
            [12.0],
            settle=True,
            settle_time=600.0,
            discard_time=300.0,
            settle_platform_damping=_DAMPING,
            settle_confirm_time=400.0,
            settle_criterion="mean_drift",
            settle_equilibrium_tolerances=EquilibriumTolerances(translation_m=0.2),
            settings=LinearizationSettings(rated_wind_speed=10.6),
            max_workers=1,
            validate=False,
        )

        assert [launch["damped"] for launch in fake.launches] == [True, False]
        template_bytes = (fst.parent / "HD.dat").read_bytes()
        # Every linearization attempt (retries included) sees the deck's own file.
        assert lin_hydro
        assert all(hydro == template_bytes for hydro in lin_hydro)
        assert _damping_diagonal(fst.parent / "HD.dat")[0] == 0.0
        convergence = result.operating_points[0].settle_convergence
        assert convergence is not None
        assert convergence.criterion == "mean_drift"
        assert convergence.confirmation is not None
        assert convergence.confirmation.confirmed

        provenance = result.cases[0]
        assert provenance.operating_point.settle_convergence is convergence
        record = tmp_path / "out" / "provenance.json"
        driver = driver_module.LinearizationDriver(
            fst_path=fst,
            output_dir=tmp_path / "out",
            settings=LinearizationSettings(rated_wind_speed=10.6),
        )
        written = driver.provenance(result.cases).write(record)
        data = json.loads(written.read_text(encoding="utf-8"))
        case = data["cases"][0]
        assert case["settle_criterion"] == "mean_drift"
        assert case["settle_confirmation"]["confirmed"] is True
        assert case["settle_confirmation"]["damping"]["surge"] == 2.0e6
        assert [a["stage"] for a in case["settle_attempts"]] == ["damped", "confirm"]

    def test_template_without_hydrodyn_rejected_before_copy(
        self, tmp_path: Path
    ) -> None:
        """A template that cannot take the damping fails before any case exists."""
        fst = _make_template(tmp_path)
        fst.write_text(
            _SWEEP_PRIMARY.replace("1 CompHydro", "0 CompHydro"), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="active HydroDyn"):
            run_operating_sweep(
                fst,
                tmp_path / "out",
                [12.0],
                settle=True,
                settle_platform_damping=_DAMPING,
                settings=LinearizationSettings(rated_wind_speed=10.6),
                max_workers=1,
                validate=False,
            )
        assert not (tmp_path / "out").exists()

    @pytest.mark.parametrize(
        "option",
        [
            {"settle_platform_damping": _DAMPING},
            {"settle_criterion": "mean_drift"},
            {"settle_confirm_time": 300.0},
            {"settle_equilibrium_tolerances": EquilibriumTolerances()},
        ],
    )
    def test_settle_options_require_settle(
        self, tmp_path: Path, option: dict[str, object]
    ) -> None:
        """The accelerator options are settle-only."""
        fst = _make_template(tmp_path)
        with pytest.raises(ValueError, match="apply only to settle=True"):
            run_operating_sweep(
                fst,
                tmp_path / "out",
                [12.0],
                curve_wind=[5.0, 20.0],
                curve_rotor_speed=[6.0, 7.5],
                curve_pitch=[0.0, 15.0],
                max_workers=1,
                validate=False,
                **option,  # type: ignore[arg-type]
            )


class TestSweepSettleFailures:
    """One wind that never settles (or fails) does not abort the sweep."""

    _WINDS: ClassVar[list[float]] = [8.0, 12.0, 16.0]

    def _sweep(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        behaviour: str,
        winds: list[float] | None = None,
        **kwargs: object,
    ) -> tuple[Any, list[float]]:
        """Run a settle sweep in which 12 m/s misbehaves as ``behaviour``."""
        from vane.run import driver as driver_module

        fst = _make_template(tmp_path)
        linearized: list[float] = []

        def wind_of(deck_path: Path) -> float:
            inflow = FastInputFile.read(deck_path.parent / "IW.dat")
            return float(inflow.get("HWindSpeed"))

        def fake_settle_run(fst_path: Path, **_: object) -> OpenFastResult:
            wind = wind_of(fst_path)
            deck = FastInputFile.read(fst_path)
            tmax, tstart = float(deck.get("TMax")), float(deck.get("TStart"))
            if wind == 12.0 and behaviour == "fails":
                return OpenFastResult(fst_path=fst_path, returncode=1, stdout="boom")
            if wind == 12.0 and behaviour == "raises":
                msg = "unexpected worker bug"
                raise RuntimeError(msg)
            amplitude = 2.0 if wind == 12.0 and behaviour == "oscillates" else 0.0
            out = _out(tmax, tstart, surge=wind, amplitude=amplitude)
            fst_path.with_suffix(".out").write_text(out, encoding="utf-8")
            return OpenFastResult(fst_path=fst_path, returncode=0)

        def fake_lin_run(deck_path: Path, **_: object) -> OpenFastResult:
            linearized.append(wind_of(deck_path))
            return OpenFastResult(
                fst_path=deck_path,
                returncode=0,
                lin_files=[deck_path.with_suffix(".1.lin")],
            )

        monkeypatch.setattr(steady_state_module, "run_openfast", fake_settle_run)
        monkeypatch.setattr(driver_module, "run_openfast", fake_lin_run)
        options: dict[str, Any] = {
            "settle": True,
            "settle_time": 600.0,
            "discard_time": 300.0,
            "max_settle_time": 900.0,
            "settings": LinearizationSettings(rated_wind_speed=10.6),
            "max_workers": 1,
            "validate": False,
            # The sustained oscillation at 12 m/s must stay unsettled.
            "settle_criterion": "velocity",
            **kwargs,
        }
        result = run_operating_sweep(
            fst, tmp_path / "out", winds or self._WINDS, **options
        )
        return result, linearized

    @pytest.mark.parametrize("workers", [1, 3])
    def test_unsettled_point_is_skipped_and_recorded(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        workers: int,
    ) -> None:
        """The other winds are linearized; the unsettled one is listed, not raised."""
        result, linearized = self._sweep(
            tmp_path, monkeypatch, "oscillates", max_workers=workers
        )
        assert [p.wind_speed for p in result.operating_points] == [8.0, 16.0]
        assert sorted(set(linearized)) == [8.0, 16.0]
        assert len(result.cases) == 2
        (failure,) = result.settle_failures
        assert failure.wind_speed == 12.0
        assert failure.reason == "not_settled"
        assert not failure.linearized
        assert "platform not settled after 900 s" in failure.message
        assert failure.convergence is not None
        assert [a.settle_time for a in failure.convergence.attempts] == [600.0, 900.0]
        assert failure.convergence.translational_velocity_rms > 0.05
        assert failure.operating_point is not None
        record = json.loads(json.dumps(failure.to_dict(), allow_nan=False))
        assert record["reason"] == "not_settled"
        assert record["platform"]["surge"] == pytest.approx(12.0, abs=0.5)
        assert len(record["convergence"]["attempts"]) == 2
        assert "did not reach a confirmed equilibrium" in caplog.text

    def test_unsettled_point_linearized_when_not_required(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """require_settled=False linearizes it at its last mean and still lists it."""
        result, linearized = self._sweep(
            tmp_path, monkeypatch, "oscillates", require_settled=False
        )
        assert [p.wind_speed for p in result.operating_points] == self._WINDS
        assert 12.0 in linearized
        (failure,) = result.settle_failures
        assert failure.linearized
        unsettled = result.operating_points[1].settle_convergence
        assert unsettled is not None
        assert not unsettled.converged

    def test_strict_settle_raises_after_completing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """strict_settle raises only once every other point is linearized."""
        with pytest.raises(SweepSettleError, match="12 m/s") as error:
            self._sweep(tmp_path, monkeypatch, "oscillates", strict_settle=True)
        assert isinstance(error.value, SteadyStateError)
        assert len(error.value.result.cases) == 2
        assert error.value.result.settle_failures[0].wind_speed == 12.0

    def test_failed_settle_run_is_recorded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An OpenFAST failure in one settle is a recorded failure, not an abort."""
        result, linearized = self._sweep(tmp_path, monkeypatch, "fails")
        assert sorted(set(linearized)) == [8.0, 16.0]
        (failure,) = result.settle_failures
        assert failure.reason == "failed"
        assert "settle run failed" in failure.message
        assert failure.operating_point is None
        assert failure.to_dict()["platform"] is None

    @pytest.mark.parametrize("workers", [1, 3])
    def test_unexpected_error_keeps_the_other_points(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: int
    ) -> None:
        """A worker bug is re-raised at the end, with the completed work attached."""
        with pytest.raises(SweepSettleError, match="unexpected") as error:
            self._sweep(tmp_path, monkeypatch, "raises", max_workers=workers)
        assert isinstance(error.value.__cause__, RuntimeError)
        result = error.value.result
        assert [p.wind_speed for p in result.operating_points] == [8.0, 16.0]
        assert len(result.cases) == 2
        (failure,) = result.settle_failures
        assert failure.reason == "error"
        assert "RuntimeError: unexpected worker bug" in failure.message

    def test_every_point_unsettled_linearizes_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With nothing settled the sweep returns no cases rather than failing."""
        result, linearized = self._sweep(
            tmp_path, monkeypatch, "oscillates", winds=[12.0]
        )
        assert result.operating_points == []
        assert result.cases == []
        assert linearized == []
        assert len(result.settle_failures) == 1
