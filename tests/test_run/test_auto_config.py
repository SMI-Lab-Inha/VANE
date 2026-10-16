# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.run.auto_config -- turbine constants inferred from deck and curve."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from vane.run import LinearizationSettings
from vane.run.auto_config import (
    DEFAULT_TRIM_GAIN_PITCH,
    DEFAULT_TRIM_TOL,
    MAX_SCALED_TRIM_TOL,
    AutoConfiguration,
    InferredValue,
    auto_configure_sweep,
    curve_power_coefficient,
    infer_rated_generator_torque,
    infer_rated_wind_speed,
    read_controller_constants,
    rotor_radius,
    scaled_trim_gain_pitch,
    scaled_trim_tol,
)
from vane.run.operating_points import operating_curve_from_csv

# A peak-shaving schedule shaped like the published NREL 5-MW one: the pitch rises from
# 11.1 m/s while the rotor speed and power level off only at 12.12 m/s.
_WIND = [3.0, 6.0, 9.77, 11.116, 12.121, 14.43, 18.39, 25.0]
_SPEED = [6.9, 6.9, 10.4, 11.835, 12.1376, 12.1376, 12.1376, 12.1376]
_PITCH = [1.6, 0.0, 0.0, 2.47, 4.12, 9.55, 15.45, 23.12]
# cp * V**3 constant (697.9) from 12.121 m/s on.
_CP = [0.47, 0.47, 0.47, 0.4358, 697.9 / 12.121**3, 697.9 / 14.43**3]
_CP += [697.9 / 18.39**3, 697.9 / 25.0**3]


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _deck(
    tmp_path: Path,
    *,
    servo: str | None = None,
    discon: str | None = None,
    aero: str | None = None,
    hydro: str | None = None,
    tip_radius: float = 63.0,
) -> Path:
    """Write a synthetic deck: ElastoDyn always, the other modules on request."""
    root = tmp_path / "deck"
    _write(
        root / "ED.dat",
        f"{tip_radius} TipRad - rotor radius\n97 GBRatio - ratio\n"
        "100 GBoxEff - efficiency (%)\n0 HubIner_Teeter - teeter inertia\n",
    )
    lines = [
        "1.225 AirDens - air density",
        f"{1 if servo else 0} CompServo - switch",
        f"{2 if aero else 0} CompAero - switch",
        f"{1 if hydro else 0} CompHydro - switch",
        '"ED.dat" EDFile - ElastoDyn',
        f'"{"SD.dat" if servo else "unused"}" ServoFile - ServoDyn',
        f'"{"AD.dat" if aero else "unused"}" AeroFile - AeroDyn',
        f'"{"HD.dat" if hydro else "unused"}" HydroFile - HydroDyn',
    ]
    if servo:
        _write(root / "SD.dat", servo)
    if discon:
        _write(root / "DISCON.IN", discon)
    if aero:
        _write(root / "AD.dat", aero)
    if hydro:
        _write(root / "HD.dat", hydro)
    return _write(root / "Model.fst", "\n".join(lines) + "\n")


_SERVO_DLL = """\
5 PCMode - pitch control
5 VSContrl - variable-speed control
9999.9 VS_RtTq - placeholder
"DISCON.IN" DLL_InFile - controller input file
"""

_DISCON = """\
! Controller parameter input file
5000000.0          ! VS_RtPwr          - Wind turbine rated power [W]
43093.55           ! VS_RtTq           - Rated torque, [Nm].
122.90967          ! VS_RefSpd         - Rated generator speed [rad/s]
94.4               ! VS_GenEff         - Generator efficiency [%]
63.0               ! WE_BladeRadius    - Blade length [m]
"""

_SERVO_SIMPLE = """\
0 PCMode - pitch control
1 VSContrl - variable-speed control
43093.55 VS_RtTq - rated torque
"""


class TestInferRatedWindSpeed:
    """Rated wind speed from the curve's rotor-speed and power plateaus."""

    def test_peak_shaving_rated_is_the_plateau_not_the_pitch_rise(self) -> None:
        """The pitch rises from 11.1 m/s but rated is where speed/power level off."""
        rated, detail = infer_rated_wind_speed(_WIND, _SPEED)
        assert rated == pytest.approx(12.121)
        assert "plateau" in detail
        assert "no cp column" in detail

    def test_power_plateau_moves_rated_past_the_speed_plateau(self) -> None:
        """At constant rated speed the power may still rise (region 2.5)."""
        wind = [8.0, 9.0, 10.27, 10.66, 10.76, 12.0, 14.0]
        speed = [7.0, 8.0, 8.676, 8.676, 8.676, 8.676, 8.676]
        level = 0.4568 * 10.76**3
        cp = [0.455, 0.455, 0.456, 0.4568, level / 10.76**3]
        cp += [level / 12.0**3, level / 14.0**3]
        assert infer_rated_wind_speed(wind, speed)[0] == 10.27
        rated, detail = infer_rated_wind_speed(wind, speed, cp)
        assert rated == 10.76
        assert "cp*V^3" in detail

    def test_speed_falling_off_its_maximum_is_not_rated(self) -> None:
        """Two samples touching the maximum, then falling away, are not a plateau."""
        with pytest.raises(ValueError, match="never levels off"):
            infer_rated_wind_speed(
                [4.0, 6.0, 8.0, 10.0, 12.0], [5.0, 10.0, 6.0, 10.0, 6.0]
            )

    def test_power_ramp_before_a_held_plateau_is_rated_at_the_plateau(self) -> None:
        """A ramp after the speed plateau does not hide the final held power.

        The median of the whole constant-speed region (here 45) lay below every
        final sample, so a valid three-sample plateau at 100 was rejected.
        """
        wind = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        speed = [5.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        power = [5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 100.0, 100.0, 100.0]
        cp = [p / v**3 for p, v in zip(power, wind, strict=True)]
        rated, _ = infer_rated_wind_speed(wind, speed, cp)
        assert rated == 10.0

    def test_power_still_rising_has_no_rated_point(self) -> None:
        """Rotor speed levelled but cp*V^3 still rising: no rated wind is invented.

        The median of a rising tail is always reached by one of its samples, so the
        old test returned a rated wind for a curve that never reaches rated power.
        """
        wind = [4.0, 6.0, 8.0, 9.0, 10.0, 11.0]
        speed = [6.0, 8.0, 10.0, 10.0, 10.0, 10.0]
        cp = [0.48, 0.48, 0.48, 0.47, 0.46, 0.45]  # cp*V^3 rises to the last sample
        with pytest.raises(ValueError, match="aerodynamic power"):
            infer_rated_wind_speed(wind, speed, cp)

    def test_curve_ending_below_rated_is_rejected(self) -> None:
        """A rotor speed still rising at the last sample has no plateau."""
        with pytest.raises(ValueError, match="never levels off"):
            infer_rated_wind_speed([4.0, 6.0, 8.0], [5.0, 6.0, 7.0])

    def test_parked_curve_is_rejected(self) -> None:
        """A curve with no positive rotor speed has no rated point."""
        with pytest.raises(ValueError, match="no positive rotor speed"):
            infer_rated_wind_speed([4.0, 8.0], [0.0, 0.0])

    def test_unsorted_curve_is_handled(self) -> None:
        """The curve order does not matter."""
        order = [3, 0, 7, 5, 1, 6, 2, 4]
        rated, _ = infer_rated_wind_speed(
            [_WIND[i] for i in order], [_SPEED[i] for i in order]
        )
        assert rated == pytest.approx(12.121)


class TestCurvePowerCoefficient:
    """The optional cp column of a performance CSV."""

    def _csv(self, tmp_path: Path, header: str, rows: list[str]) -> Path:
        return _write(tmp_path / "curve.csv", header + "\n" + "\n".join(rows) + "\n")

    def test_cp_is_read_and_aligned(self, tmp_path: Path) -> None:
        """The cp column comes back sorted like operating_curve_from_csv's winds."""
        path = self._csv(
            tmp_path,
            "wind_speed [m/s],rotor_speed_rpm,blade_pitch_deg,cp,ct",
            ["12,12,4,0.39,0.5", "8,9,0,0.47,0.7"],
        )
        wind, _, _ = operating_curve_from_csv(path)
        assert curve_power_coefficient(path, wind) == [0.47, 0.39]

    def test_missing_cp_column(self, tmp_path: Path) -> None:
        """Without a cp column there is nothing to read."""
        path = self._csv(
            tmp_path, "wind_speed,rotor_speed_rpm,blade_pitch_deg", ["8,9,0", "12,12,4"]
        )
        assert curve_power_coefficient(path, [8.0, 12.0]) is None

    def test_empty_cp_values(self, tmp_path: Path) -> None:
        """A cp column left empty (a table that publishes none) is ignored."""
        path = self._csv(
            tmp_path,
            "wind_speed,rotor_speed_rpm,blade_pitch_deg,cp,ct",
            ["8,9,0,,", "12,12,4,,"],
        )
        assert curve_power_coefficient(path, [8.0, 12.0]) is None

    def test_mismatched_winds_and_missing_file(self, tmp_path: Path) -> None:
        """A cp column that does not match the curve, or no file, gives None."""
        path = self._csv(
            tmp_path,
            "wind_speed,rotor_speed_rpm,blade_pitch_deg,cp",
            ["8,9,0,0.4", "12,12,4,0.3"],
        )
        assert curve_power_coefficient(path, [8.0, 13.0]) is None
        assert curve_power_coefficient(path, [8.0]) is None
        assert curve_power_coefficient(tmp_path / "absent.csv", [8.0]) is None

    def test_short_or_non_positive(self, tmp_path: Path) -> None:
        """A header-only file or a non-positive cp gives None."""
        header_only = self._csv(tmp_path, "wind_speed,rotor_speed_rpm,cp", [])
        assert curve_power_coefficient(header_only, []) is None
        path = self._csv(
            tmp_path,
            "wind_speed,rotor_speed_rpm,blade_pitch_deg,cp",
            ["8,9,0,0.4", "12,12,4,-0.1"],
        )
        assert curve_power_coefficient(path, [8.0, 12.0]) is None


class TestControllerConstants:
    """DISCON.IN constants found through ServoDyn DLL_InFile."""

    def test_discon_file_is_read(self, tmp_path: Path) -> None:
        """Every rated constant is parsed, VS_GenEff as a fraction."""
        fst = _deck(tmp_path, servo=_SERVO_DLL, discon=_DISCON)
        constants = read_controller_constants(fst)
        assert constants is not None
        assert constants.rated_torque == pytest.approx(43093.55)
        assert constants.rated_power == pytest.approx(5e6)
        assert constants.rated_generator_speed == pytest.approx(122.90967)
        assert constants.generator_efficiency == pytest.approx(0.944)
        assert constants.blade_radius == pytest.approx(63.0)
        assert constants.path.name == "DISCON.IN"

    def test_pc_refspd_is_the_fallback_speed(self, tmp_path: Path) -> None:
        """Without VS_RefSpd the pitch controller's reference speed is used."""
        discon = _DISCON.replace("VS_RefSpd", "PC_RefSpd")
        constants = read_controller_constants(
            _deck(tmp_path, servo=_SERVO_DLL, discon=discon)
        )
        assert constants is not None
        assert constants.rated_generator_speed == pytest.approx(122.90967)

    @pytest.mark.parametrize(
        ("servo", "discon"),
        [
            (_SERVO_SIMPLE + '"DISCON.IN" DLL_InFile - file\n', _DISCON),
            (_SERVO_DLL, None),
            (_SERVO_DLL.replace('"DISCON.IN" DLL_InFile', '"" Other'), _DISCON),
            (_SERVO_DLL, "! nothing here\n0 ! VS_RtTq - zero\n"),
        ],
        ids=["no-library", "missing-file", "no-dll-infile", "no-constants"],
    )
    def test_no_usable_controller_file(
        self, tmp_path: Path, servo: str, discon: str | None
    ) -> None:
        """No library, no file, or no positive constants: nothing is read."""
        fst = _deck(tmp_path, servo=servo, discon=discon)
        assert read_controller_constants(fst) is None

    def test_servodyn_switched_off(self, tmp_path: Path) -> None:
        """A disabled ServoDyn is inert."""
        fst = _deck(tmp_path, servo=_SERVO_DLL, discon=_DISCON)
        text = fst.read_text(encoding="utf-8").replace("1 CompServo", "0 CompServo")
        fst.write_text(text, encoding="utf-8")
        assert read_controller_constants(fst) is None
        assert read_controller_constants(tmp_path / "absent.fst") is None


class TestRatedGeneratorTorque:
    """The rated-torque source chain: controller file, ServoDyn, curve cp."""

    def test_controller_file_first(self, tmp_path: Path) -> None:
        """A DISCON.IN VS_RtTq wins over everything else."""
        fst = _deck(tmp_path, servo=_SERVO_DLL, discon=_DISCON)
        torque = infer_rated_generator_torque(
            fst, curve_wind=_WIND, curve_rotor_speed=_SPEED, curve_cp=_CP
        )
        assert torque is not None
        assert torque.value == pytest.approx(43093.55)
        assert torque.source == "DISCON.IN VS_RtTq"

    @pytest.mark.parametrize("switch", ["AfCmode", "CCmode"])
    def test_flap_or_cable_control_through_the_library(
        self, tmp_path: Path, switch: str
    ) -> None:
        """Active-flap or cable control at 5 also loads the library and its file."""
        servo = (
            f"0 PCMode - pitch\n0 VSContrl - torque\n5 {switch} - routed to the DLL\n"
            '"DISCON.IN" DLL_InFile - controller input file\n'
        )
        torque = infer_rated_generator_torque(
            _deck(tmp_path, servo=servo, discon=_DISCON)
        )
        assert torque is not None
        assert torque.source == "DISCON.IN VS_RtTq"

    def test_controller_constant_in_fortran_d_notation(self, tmp_path: Path) -> None:
        """``4.309355D+04 ! VS_RtTq`` in DISCON.IN is the rated torque, not absent."""
        discon = _DISCON.replace(
            "43093.55           ! VS_RtTq", "4.309355D+04 ! VS_RtTq"
        )
        assert "D+04" in discon
        torque = infer_rated_generator_torque(
            _deck(tmp_path, servo=_SERVO_DLL, discon=discon)
        )
        assert torque is not None
        assert torque.value == pytest.approx(43093.55)
        assert torque.source == "DISCON.IN VS_RtTq"

    def test_servodyn_simple_law(self, tmp_path: Path) -> None:
        """ServoDyn's VS_RtTq is used when the simple law (VSContrl=1) runs."""
        torque = infer_rated_generator_torque(_deck(tmp_path, servo=_SERVO_SIMPLE))
        assert torque is not None
        assert torque.value == pytest.approx(43093.55)
        assert "VSContrl=1" in torque.detail

    def test_library_placeholder_is_never_used(self, tmp_path: Path) -> None:
        """Under a library ServoDyn's VS_RtTq is a placeholder; no curve, no value."""
        fst = _deck(tmp_path, servo=_SERVO_DLL)
        assert infer_rated_generator_torque(fst) is None

    def test_curve_cp_fallback(self, tmp_path: Path) -> None:
        """The curve's aerodynamic torque at rated, moved to the generator side."""
        fst = _deck(tmp_path, servo=_SERVO_DLL)
        torque = infer_rated_generator_torque(
            fst, curve_wind=_WIND, curve_rotor_speed=_SPEED, curve_cp=_CP
        )
        assert torque is not None
        omega = 12.1376 * math.pi / 30.0
        expected = 0.5 * 1.225 * math.pi * 63.0**2 * 697.9 / omega / 97.0
        assert torque.value == pytest.approx(expected, rel=1e-9)
        assert torque.value == pytest.approx(43093.5, rel=0.01)
        assert torque.source == "performance curve cp at rated"

    def test_curve_without_cp_or_gearbox(self, tmp_path: Path) -> None:
        """No cp column, or no ElastoDyn gearbox ratio: no estimate."""
        fst = _deck(tmp_path, servo=_SERVO_DLL)
        assert (
            infer_rated_generator_torque(
                fst, curve_wind=_WIND, curve_rotor_speed=_SPEED
            )
            is None
        )
        ed = fst.parent / "ED.dat"
        ed.write_text("63 TipRad - radius\n", encoding="utf-8")
        assert (
            infer_rated_generator_torque(
                fst, curve_wind=_WIND, curve_rotor_speed=_SPEED, curve_cp=_CP
            )
            is None
        )

    def test_air_density_from_aerodyn_and_default(self, tmp_path: Path) -> None:
        """An older deck keeps AirDens in AeroDyn; with none the default is used."""
        aero = "1.0 AirDens - density\n0 UA_Mod - quasi-steady\n"
        fst = _deck(tmp_path, servo=_SERVO_DLL, aero=aero)
        text = fst.read_text(encoding="utf-8").replace("1.225 AirDens", "0 Gravity")
        fst.write_text(text, encoding="utf-8")
        torque = infer_rated_generator_torque(
            fst, curve_wind=_WIND, curve_rotor_speed=_SPEED, curve_cp=_CP
        )
        assert torque is not None
        assert "rho=1 (AD.dat AirDens)" in torque.detail
        (fst.parent / "AD.dat").write_text("0 UA_Mod - none\n", encoding="utf-8")
        torque = infer_rated_generator_torque(
            fst, curve_wind=_WIND, curve_rotor_speed=_SPEED, curve_cp=_CP
        )
        assert torque is not None
        assert "default 1.225" in torque.detail

    def test_unreadable_deck(self, tmp_path: Path) -> None:
        """A missing deck gives no torque."""
        assert infer_rated_generator_torque(tmp_path / "absent.fst") is None


class TestPitchTrimGain:
    """The rotor-size scaling of the pitch trim gain."""

    @pytest.mark.parametrize(
        ("radius", "converged_gain"),
        [(63.0, 2e-4), (64.9, 2e-4), (121.0, 2.83e-5), (141.0, 1.75e-5)],
        ids=["nrel5mw", "iea3p4", "iea15", "iea22"],
    )
    def test_rule_matches_the_campaign_evidence(
        self, radius: float, converged_gain: float
    ) -> None:
        """Within a factor of 1.5 of a gain that converged the turbine above rated.

        Single-attempt trims of the reference-turbine campaign: every NREL 5-MW point,
        the IEA 3.4-MW from 9 to 25 m/s, and the IEA 15- and 22-MW above rated.
        """
        gain = scaled_trim_gain_pitch(radius)
        assert converged_gain / 1.5 <= gain <= converged_gain * 1.5

    def test_reference_value_and_cubic_scaling(self) -> None:
        """2e-4 at 63 m, an eighth of it at twice the radius."""
        assert scaled_trim_gain_pitch(63.0) == pytest.approx(DEFAULT_TRIM_GAIN_PITCH)
        assert pytest.approx(2e-4) == DEFAULT_TRIM_GAIN_PITCH
        assert scaled_trim_gain_pitch(126.0) == pytest.approx(2e-4 / 8.0)

    @pytest.mark.parametrize("radius", [0.0, -1.0, math.inf, math.nan])
    def test_invalid_radius(self, radius: float) -> None:
        """Only a finite positive radius has a gain."""
        with pytest.raises(ValueError, match="rotor radius"):
            scaled_trim_gain_pitch(radius)


class TestTrimTolerance:
    """The rotor-size scaling of the trim tolerance."""

    @pytest.mark.parametrize(
        ("radius", "expected"),
        [
            (63.0, 1e-3),
            (64.9, 1e-3 * (64.9 / 63.0) ** 3),
            (99.2, 1e-3 * (99.2 / 63.0) ** 3),
            (121.0, 1e-3 * (121.0 / 63.0) ** 3),
            (141.0, 1e-2),
        ],
        ids=["nrel5mw", "iea3p4", "iea10", "iea15", "iea22"],
    )
    def test_rule_on_the_reference_turbines(
        self, radius: float, expected: float
    ) -> None:
        """Cubic in the radius, anchored at 1e-3 for 63 m and capped at 1e-2."""
        assert scaled_trim_tol(radius) == pytest.approx(expected)

    def test_bracketed_by_the_campaign_evidence(self) -> None:
        """Tight where 1e-2 broke periodicity, loose where 1e-3 did not trim.

        At 1e-2 the IEA 3.4-MW below rated stopped with a 1.02--2.57 % rotor-speed
        spread; at 1e-3 the IEA 22-MW trimmed only 2 of 9 points within ``TMax``.
        """
        assert scaled_trim_tol(64.9) < 1.2e-3
        assert scaled_trim_tol(141.0) == pytest.approx(MAX_SCALED_TRIM_TOL)

    def test_bounds(self) -> None:
        """Never tighter than the default, never looser than the MBC3-safe cap."""
        assert scaled_trim_tol(10.0) == pytest.approx(DEFAULT_TRIM_TOL)
        assert scaled_trim_tol(500.0) == pytest.approx(MAX_SCALED_TRIM_TOL)

    @pytest.mark.parametrize("radius", [0.0, -1.0, math.inf, math.nan])
    def test_invalid_radius(self, radius: float) -> None:
        """Only a finite positive radius has a tolerance."""
        with pytest.raises(ValueError, match="rotor radius"):
            scaled_trim_tol(radius)

    def test_rotor_radius_reads_tipradius(self, tmp_path: Path) -> None:
        """TipRad comes from ElastoDyn; absent decks give None."""
        assert rotor_radius(_deck(tmp_path, tip_radius=120.97)) == 120.97
        assert rotor_radius(tmp_path / "absent.fst") is None


_AERO_UA4 = """\
1 Wake_Mod - BEMT
True TipLoss - tip loss
-1 DBEMT_Mod - frozen
4 UA_Mod - unsteady
1 UAEndRad - end radius
"""

_HYDRO = """\
1 PotMod - WAMIT
"pot/plat" PotFile - root
1 RdtnMod - convolution
1 ExctnMod - DFT
0 NAddDOF - additional DOFs
"""


class TestAutoConfigureSweep:
    """Settings completed from the deck and curve, with the provenance record."""

    def test_all_constants_inferred_with_sources(self, tmp_path: Path) -> None:
        """settings=None: every constant is inferred and recorded with its source."""
        fst = _deck(tmp_path, servo=_SERVO_DLL, discon=_DISCON, tip_radius=121.0)
        settings, auto = auto_configure_sweep(
            fst, None, curve_wind=_WIND, curve_rotor_speed=_SPEED, curve_cp=_CP
        )
        assert settings.rated_wind_speed == pytest.approx(12.121)
        assert settings.rated_generator_torque == pytest.approx(43093.55)
        assert settings.trim_gain_pitch == pytest.approx(2e-4 * (63 / 121) ** 3)
        assert settings.trim_tol == pytest.approx(1e-3 * (121 / 63) ** 3)
        assert settings.trim_controller
        names = [item.name for item in auto.inferred]
        assert names[0] == "module_preset"
        assert {
            "rated_wind_speed",
            "rated_generator_torque",
            "trim_gain_pitch",
            "trim_tol",
        } <= set(names)
        record = auto.get("rated_generator_torque")
        assert record is not None
        assert record.source == "DISCON.IN VS_RtTq"
        assert auto.get("absent") is None
        json.dumps(auto.to_dict())

    def test_explicit_default_valued_trim_settings_are_kept(
        self, tmp_path: Path
    ) -> None:
        """Passing the default numbers explicitly is not the same as omitting them.

        The equality test used to treat ``trim_gain_pitch=2e-4, trim_tol=1e-3``
        as unset and rescale them for the 121 m rotor.
        """
        fst = _deck(tmp_path, servo=_SERVO_DLL, discon=_DISCON, tip_radius=121.0)
        given = LinearizationSettings(trim_gain_pitch=2e-4, trim_tol=1e-3)
        settings, auto = auto_configure_sweep(
            fst, given, curve_wind=_WIND, curve_rotor_speed=_SPEED
        )
        assert settings.trim_gain_pitch == 2e-4
        assert settings.trim_tol == 1e-3
        assert auto.get("trim_gain_pitch") is None
        assert auto.get("trim_tol") is None
        # Omitted, the same fields are still scaled.
        scaled, _ = auto_configure_sweep(
            fst, LinearizationSettings(), curve_wind=_WIND, curve_rotor_speed=_SPEED
        )
        assert scaled.trim_tol == pytest.approx(1e-3 * (121.0 / 63.0) ** 3)

    def test_fortran_exponent_tip_radius_is_read(self, tmp_path: Path) -> None:
        """``1.2097D+02 TipRad`` is a 121 m rotor, not an unreadable value.

        ``float`` rejected the D exponent, so the 63 m trim defaults stayed on.
        """
        fst = _deck(tmp_path, tip_radius=120.97)
        ed = fst.parent / "ED.dat"
        ed.write_text(
            ed.read_text(encoding="utf-8").replace(
                "120.97 TipRad", "1.2097D+02 TipRad"
            ),
            encoding="utf-8",
        )
        assert rotor_radius(fst) == pytest.approx(120.97)
        settings, _ = auto_configure_sweep(fst, LinearizationSettings())
        assert settings.trim_tol == pytest.approx(1e-3 * (120.97 / 63.0) ** 3)

    def test_list_directed_comma_after_a_value(self, tmp_path: Path) -> None:
        """``1.21D+02, TipRad``: the comma ends the value, it does not void it."""
        fst = _deck(tmp_path, tip_radius=121.0)
        ed = fst.parent / "ED.dat"
        ed.write_text(
            ed.read_text(encoding="utf-8").replace("121.0 TipRad", "1.21D+02, TipRad"),
            encoding="utf-8",
        )
        assert rotor_radius(fst) == pytest.approx(121.0)

    def test_default_marker_survives_replace_and_arithmetic(self) -> None:
        """``replace`` keeps an untouched default; arithmetic gives a plain float."""
        from dataclasses import replace

        from vane.run.lin_config import is_size_scaled_default

        default = LinearizationSettings()
        assert is_size_scaled_default(default.trim_tol)
        kept = replace(default, rated_wind_speed=11.4)
        assert is_size_scaled_default(kept.trim_tol)
        assert not is_size_scaled_default(default.trim_tol * 3.0)
        assert not is_size_scaled_default(replace(default, trim_tol=1e-3).trim_tol)

    def test_explicit_values_win(self, tmp_path: Path) -> None:
        """Explicit settings are kept; only their unset constants are filled."""
        fst = _deck(tmp_path, servo=_SERVO_DLL, discon=_DISCON, tip_radius=121.0)
        given = LinearizationSettings(
            rated_wind_speed=11.4,
            trim_gain_pitch=5e-4,
            trim_tol=4e-3,
            unsteady_aero_mode=None,
        )
        settings, auto = auto_configure_sweep(
            fst, given, curve_wind=_WIND, curve_rotor_speed=_SPEED
        )
        assert settings.rated_wind_speed == 11.4
        assert settings.trim_gain_pitch == 5e-4
        assert settings.trim_tol == 4e-3
        assert auto.get("trim_tol") is None
        assert settings.rated_generator_torque == pytest.approx(43093.55)
        assert not settings.trim_controller  # the user's settings, not the preset
        assert auto.get("rated_wind_speed") is None
        assert auto.get("module_preset") is None
        assert given.rated_generator_torque is None  # the input is not mutated

    def test_rigid_blades_are_reported(self, tmp_path: Path) -> None:
        """Flexibility DOFs a deck ships off are named in a warning; overrides win."""
        fst = _deck(tmp_path)
        ed = fst.parent / "ED.dat"
        ed.write_text(
            ed.read_text(encoding="utf-8")
            + "False FlapDOF1 - flap\nFalse EdgeDOF - edge\nTrue TwFADOF1 - tower\n",
            encoding="utf-8",
        )
        _, auto = auto_configure_sweep(fst, None)
        warning = next(w for w in auto.warnings if "FlapDOF1" in w)
        assert "EdgeDOF" in warning
        assert "TwFADOF1" not in warning

        kept = LinearizationSettings(structural_dofs={"FlapDOF1": False})
        _, auto = auto_configure_sweep(fst, kept)
        warning = next(w for w in auto.warnings if "EdgeDOF" in w)
        assert "FlapDOF1" not in warning

    def test_uninferable_constants_are_left_unset(self, tmp_path: Path) -> None:
        """A curve without a plateau leaves rated_wind_speed unset, with a warning."""
        fst = _deck(tmp_path)
        settings, auto = auto_configure_sweep(
            fst, None, curve_wind=[4.0, 8.0], curve_rotor_speed=[5.0, 7.0]
        )
        assert settings.rated_wind_speed is None
        assert settings.rated_generator_torque is None
        assert any("rated_wind_speed could not be inferred" in w for w in auto.warnings)

    def test_controller_disagreement_is_warned(self, tmp_path: Path) -> None:
        """A curve whose rated speed or power differs from the controller warns."""
        discon = _DISCON.replace("122.90967", "140.0").replace("5000000.0", "4e6")
        fst = _deck(tmp_path, servo=_SERVO_DLL, discon=discon)
        _, auto = auto_configure_sweep(
            fst, None, curve_wind=_WIND, curve_rotor_speed=_SPEED, curve_cp=_CP
        )
        assert any("rated rotor speed" in w for w in auto.warnings)
        assert any("rated power" in w for w in auto.warnings)

    def test_agreeing_controller_is_not_warned(self, tmp_path: Path) -> None:
        """The NREL 5-MW-like curve agrees with its controller within tolerance."""
        fst = _deck(tmp_path, servo=_SERVO_DLL, discon=_DISCON)
        _, auto = auto_configure_sweep(
            fst, None, curve_wind=_WIND, curve_rotor_speed=_SPEED, curve_cp=_CP
        )
        assert auto.warnings == ()

    def test_missing_state_space_files_fall_back_to_none(self, tmp_path: Path) -> None:
        """No .ss/.ssexctn: RdtnMod/ExctnMod 0, each with a recorded warning."""
        fst = _deck(tmp_path, hydro=_HYDRO)
        settings, auto = auto_configure_sweep(fst, None)
        assert settings.hydro_radiation_mode == 0
        assert settings.hydro_excitation_mode == 0
        assert any("RdtnMod=1" in w and ".ss'" in w for w in auto.warnings)
        assert any("ExctnMod=1" in w and ".ssexctn'" in w for w in auto.warnings)

    def test_fittable_wamit_data_keep_state_space_radiation(
        self, tmp_path: Path
    ) -> None:
        """No .ss but fittable .1 data: RdtnMod 2 with a per-case fit, recorded.

        The template is only read: the fit is generated in each case copy.
        """
        from tests.test_run.test_radiation_fit import _write_wamit

        fst = _deck(tmp_path, hydro=_HYDRO + "1025 WtrDens - water density\n")
        (fst.parent / "pot").mkdir()
        _write_wamit(fst.parent / "pot" / "plat.1")
        settings, auto = auto_configure_sweep(fst, None)
        assert settings.hydro_radiation_mode == 2
        assert settings.hydro_radiation_fit
        assert settings.hydro_excitation_mode == 0
        record = auto.get("hydro_radiation_fit")
        assert record is not None
        assert record.value is True
        assert "plat.1" in record.detail
        assert not any("RdtnMod" in w for w in auto.warnings)
        assert any("ExctnMod=1" in w for w in auto.warnings)
        assert not (fst.parent / "pot" / "plat.ss").exists()
        json.dumps(auto.to_dict())

    def test_unreadable_wamit_data_fall_back_with_the_reason(
        self, tmp_path: Path
    ) -> None:
        """A malformed .1 file cannot be fitted: RdtnMod 0, the warning says why."""
        fst = _deck(tmp_path, hydro=_HYDRO + "1025 WtrDens - water density\n")
        _write(fst.parent / "pot" / "plat.1", "not wamit data\n")
        settings, auto = auto_configure_sweep(fst, None)
        assert settings.hydro_radiation_mode == 0
        assert not settings.hydro_radiation_fit
        assert any(
            "RdtnMod=1" in w and "cannot be fitted" in w and "malformed" in w
            for w in auto.warnings
        )

    def test_present_state_space_files_are_used(self, tmp_path: Path) -> None:
        """With the files present the state-space models are selected."""
        fst = _deck(tmp_path, hydro=_HYDRO)
        _write(fst.parent / "pot" / "plat.ss", "ss")
        _write(fst.parent / "pot" / "plat.ssexctn", "ss")
        settings, auto = auto_configure_sweep(fst, None)
        assert settings.hydro_radiation_mode == 2
        assert settings.hydro_excitation_mode == 2
        assert auto.warnings == ()

    def test_deck_none_models_and_additional_dofs(self, tmp_path: Path) -> None:
        """A deck's own 0 is kept; additional body DOFs force 0 with a warning."""
        hydro = _HYDRO.replace("1 ExctnMod", "0 ExctnMod").replace(
            "0 NAddDOF", "2 NAddDOF"
        )
        fst = _deck(tmp_path, hydro=hydro)
        settings, auto = auto_configure_sweep(fst, None)
        assert settings.hydro_excitation_mode is None
        assert settings.hydro_radiation_mode == 0
        assert any("NAddDOF" in w for w in auto.warnings)

    def test_additional_dofs_with_a_list_directed_comma(self, tmp_path: Path) -> None:
        """``2, NAddDOF`` is two extra DOFs, not an unreadable (zero) count."""
        hydro = _HYDRO.replace("0 NAddDOF", "2, NAddDOF")
        settings, auto = auto_configure_sweep(_deck(tmp_path, hydro=hydro), None)
        assert settings.hydro_radiation_mode == 0
        assert any("NAddDOF" in w for w in auto.warnings)

    def test_morison_only_hydrodyn_is_untouched(self, tmp_path: Path) -> None:
        """Without WAMIT flow the radiation/excitation switches are irrelevant."""
        fst = _deck(tmp_path, hydro=_HYDRO.replace("1 PotMod", "0 PotMod"))
        settings, _ = auto_configure_sweep(fst, None)
        assert settings.hydro_radiation_mode is None
        assert settings.hydro_excitation_mode is None

    def test_linearizable_unsteady_aero_is_kept_off_the_tip(
        self, tmp_path: Path
    ) -> None:
        """UA_Mod 4 is kept; UAEndRad 0.99 keeps it off the tip node."""
        settings, auto = auto_configure_sweep(_deck(tmp_path, aero=_AERO_UA4), None)
        assert settings.unsteady_aero_mode is None
        assert settings.ua_end_radius == 0.99
        assert settings.dynamic_wake_mode == -1
        record = auto.get("ua_end_radius")
        assert record is not None
        assert record.source == "AD.dat UAEndRad"

    def test_unsteady_aero_inboard_needs_nothing(self, tmp_path: Path) -> None:
        """A deck already ending UA inboard (or without tip loss) is left alone."""
        aero = _AERO_UA4.replace("1 UAEndRad", "0.95 UAEndRad")
        settings, _ = auto_configure_sweep(_deck(tmp_path, aero=aero), None)
        assert settings.unsteady_aero_mode is None
        assert settings.ua_end_radius is None

    def test_unsteady_aero_without_uaendrad_line(self, tmp_path: Path) -> None:
        """No UAEndRad line to write: quasi-steady aerodynamics, with a warning."""
        aero = _AERO_UA4.replace("1 UAEndRad - end radius\n", "")
        settings, auto = auto_configure_sweep(_deck(tmp_path, aero=aero), None)
        assert settings.unsteady_aero_mode == 0
        assert settings.ua_end_radius is None
        assert any("no UAEndRad line" in w for w in auto.warnings)

    def test_unlinearizable_unsteady_aero_is_replaced(self, tmp_path: Path) -> None:
        """UA_Mod 3 becomes quasi-steady (0), recorded with the deck's value."""
        aero = _AERO_UA4.replace("4 UA_Mod", "3 UA_Mod")
        settings, auto = auto_configure_sweep(_deck(tmp_path, aero=aero), None)
        assert settings.unsteady_aero_mode == 0
        record = auto.get("unsteady_aero_mode")
        assert record is not None
        assert "UA_Mod=3" in record.detail

    def test_records_extend_and_serialize(self) -> None:
        """AutoConfiguration is append-only and JSON-ready."""
        base = AutoConfiguration((InferredValue("a", 1.0, "x"),), ("w",))
        more = base.extended([InferredValue("a", 2.0, "y", "d")], ["v"])
        record = more.get("a")
        assert record is not None
        assert record.value == 2.0
        assert more.to_dict() == {
            "inferred": [
                {"name": "a", "value": 1.0, "source": "x", "detail": ""},
                {"name": "a", "value": 2.0, "source": "y", "detail": "d"},
            ],
            "warnings": ["w", "v"],
        }


def _reference_turbines() -> Path:
    """Return the reference-turbine directory (parent of the runs), or skip."""
    runs = os.environ.get("VANE_REFERENCE_RUNS_DIR")
    if not runs:
        pytest.skip("reference-turbine data unavailable; set VANE_REFERENCE_RUNS_DIR")
    root = Path(runs).parent
    if not (root / "_performance" / "turbines.json").is_file():
        pytest.skip("reference-turbine data unavailable; set VANE_REFERENCE_RUNS_DIR")
    return root


_REFERENCE = {
    # turbine key, deck, curve, inferred rated wind (m/s)
    "iea3p4": (
        "IEA-3.4-130-RWT/openfast/IEA-3.4-130-RWT.fst",
        "iea3p4_curve.csv",
        9.8127,
    ),
    "iea10": (
        "IEA-10.0-198-RWT/openfast/IEA-10.0-198-RWT.fst",
        "iea10_curve.csv",
        10.7577,
    ),
    "iea15": (
        "IEA-15-240-RWT/OpenFAST/IEA-15-240-RWT-Monopile/IEA-15-240-RWT-Monopile.fst",
        "iea15_curve.csv",
        10.6594,
    ),
}


@pytest.mark.parametrize("turbine", sorted(_REFERENCE))
def test_reference_turbine_constants(turbine: str) -> None:
    """The published rated torque and wind are recovered from the real inputs."""
    root = _reference_turbines()
    deck, curve, rated_wind = _REFERENCE[turbine]
    published = json.loads(
        (root / "_performance" / "turbines.json").read_text(encoding="utf-8")
    )[turbine]
    csv_path = root / "_performance" / curve
    wind, speed, _ = operating_curve_from_csv(csv_path)
    settings, auto = auto_configure_sweep(
        root / deck,
        None,
        curve_wind=wind,
        curve_rotor_speed=speed,
        curve_cp=curve_power_coefficient(csv_path, wind),
    )
    assert settings.rated_wind_speed == pytest.approx(rated_wind, abs=1e-3)
    assert settings.rated_wind_speed == pytest.approx(
        published["rated_wind_speed_mps"]["value"], abs=0.8
    )
    assert settings.rated_generator_torque == pytest.approx(
        published["rated_generator_torque_Nm"]["value"], rel=1e-6
    )
    radius = published["rotor_radius_m"]["value"]
    assert settings.trim_gain_pitch == pytest.approx(scaled_trim_gain_pitch(radius))
    assert settings.unsteady_aero_mode == 0  # UA_Mod 3 decks
    assert auto.get("rated_wind_speed") is not None


def test_nrel5mw_rtest_torque_from_the_curve(rtest_dir: Path) -> None:
    """The r-test land deck's legacy DLL has no DISCON.IN: cp gives the torque."""
    root = _reference_turbines()
    csv_path = root / "_performance" / "nrel5mw_curve.csv"
    wind, speed, _ = operating_curve_from_csv(csv_path)
    deck = rtest_dir / "glue-codes/openfast/5MW_Land_DLL_WTurb/5MW_Land_DLL_WTurb.fst"
    settings, auto = auto_configure_sweep(
        deck,
        None,
        curve_wind=wind,
        curve_rotor_speed=speed,
        curve_cp=curve_power_coefficient(csv_path, wind),
    )
    assert settings.rated_wind_speed == pytest.approx(12.121)
    assert settings.rated_generator_torque == pytest.approx(43093.5, rel=0.005)
    record = auto.get("rated_generator_torque")
    assert record is not None
    assert record.source == "performance curve cp at rated"


class TestReviewedEdgeCases:
    """Unusual decks and curves the automatic configuration must survive."""

    def test_hydrodyn_without_an_excitation_line(self, tmp_path: Path) -> None:
        """An older HydroDyn file has no ExctnMod line: nothing to switch, no crash."""
        hydro = _HYDRO.replace("1 ExctnMod - DFT\n", "")
        settings, auto = auto_configure_sweep(_deck(tmp_path, hydro=hydro), None)
        assert settings.hydro_excitation_mode is None
        assert settings.hydro_radiation_mode == 0
        assert auto.get("hydro_excitation_mode") is None
        assert not any("ExctnMod" in w for w in auto.warnings)

    def test_tip_loss_written_as_a_fortran_logical(self, tmp_path: Path) -> None:
        """``.true.`` turns tip loss on, so UAEndRad is still moved off the tip."""
        aero = _AERO_UA4.replace("True TipLoss", ".true. TipLoss")
        settings, _ = auto_configure_sweep(_deck(tmp_path, aero=aero), None)
        assert settings.ua_end_radius == 0.99

    def test_mismatched_cp_is_rejected(self) -> None:
        """A cp sequence not aligned with the curve is an error, not a broadcast."""
        with pytest.raises(ValueError, match="same length"):
            infer_rated_wind_speed(_WIND, _SPEED, [0.45])
        with pytest.raises(ValueError, match="same length"):
            infer_rated_wind_speed(_WIND, _SPEED[:-1])


class TestCurvePowerCoefficientUnits:
    """A cp column given in percent must not inflate the inferred rated torque."""

    def _csv(self, tmp_path: Path, cp: str, first: str, second: str) -> Path:
        text = (
            f"wind_speed,rotor_speed_rpm,blade_pitch_deg,{cp}\n"
            f"8,9,0,{first}\n12,12,4,{second}\n"
        )
        return _write(tmp_path / "curve.csv", text)

    def test_percent_unit_is_converted(self, tmp_path: Path) -> None:
        """``cp [%]`` holds percentages; they come back as fractions."""
        path = self._csv(tmp_path, "cp [%]", "47", "39")
        values = curve_power_coefficient(path, [8.0, 12.0])
        assert values == pytest.approx([0.47, 0.39])

    def test_dimensionless_units_are_accepted(self, tmp_path: Path) -> None:
        """``[-]`` and ``(1)`` mean a fraction."""
        for unit in ("cp [-]", "Cp (1)"):
            path = self._csv(tmp_path, unit, "0.47", "0.39")
            assert curve_power_coefficient(path, [8.0, 12.0]) == [0.47, 0.39]

    def test_zero_cp_row_is_kept(self, tmp_path: Path) -> None:
        """A zero-wind row with cp = 0 is valid data; the column is kept."""
        path = _write(
            tmp_path / "curve.csv",
            "wind_speed,rotor_speed_rpm,blade_pitch_deg,cp\n"
            "0,0,90,0\n8,9,0,0.47\n12,12,4,0.39\n",
        )
        assert curve_power_coefficient(path, [0.0, 8.0, 12.0]) == [0.0, 0.47, 0.39]

    def test_negative_cp_is_rejected(self, tmp_path: Path) -> None:
        """A negative power coefficient is not usable data."""
        path = self._csv(tmp_path, "cp", "-0.1", "0.39")
        assert curve_power_coefficient(path, [8.0, 12.0]) is None

    def test_percent_without_a_unit_is_rejected(self, tmp_path: Path) -> None:
        """Values above the Betz limit cannot be power coefficients."""
        path = self._csv(tmp_path, "cp", "47", "39")
        with pytest.raises(ValueError, match="Betz limit"):
            curve_power_coefficient(path, [8.0, 12.0])

    def test_percent_forms_are_converted(self, tmp_path: Path) -> None:
        """``[percent]`` and ``(pct)`` are percentages, like ``[%]``."""
        for unit in ("cp [percent]", "Cp (pct)"):
            path = self._csv(tmp_path, unit, "47", "39")
            values = curve_power_coefficient(path, [8.0, 12.0])
            assert values == pytest.approx([0.47, 0.39])

    def test_fractions_declared_as_percent_are_rejected(self, tmp_path: Path) -> None:
        """Fractions under a percent unit would give a 100 times smaller cp."""
        path = self._csv(tmp_path, "cp [%]", "0.47", "0.39")
        with pytest.raises(ValueError, match="declares a percentage"):
            curve_power_coefficient(path, [8.0, 12.0])

    def test_same_columns_as_the_verification_reader(self, tmp_path: Path) -> None:
        """The long cp column name is read, and two cp columns are an error."""
        path = self._csv(tmp_path, "power_coefficient [%]", "47", "39")
        assert curve_power_coefficient(path, [8.0, 12.0]) == pytest.approx([0.47, 0.39])
        two = _write(
            tmp_path / "two.csv",
            "wind_speed,rotor_speed_rpm,blade_pitch_deg,cp,RtAeroCp\n"
            "8,9,0,0.47,0.47\n12,12,4,0.39,0.39\n",
        )
        with pytest.raises(ValueError, match="several cp columns"):
            curve_power_coefficient(two, [8.0, 12.0])

    def test_unknown_unit_is_rejected(self, tmp_path: Path) -> None:
        """A unit that is not a fraction or a percentage is an error, not a guess."""
        for unit in ("cp [W]", "cp [kW]"):
            path = self._csv(tmp_path, unit, "0.47", "0.39")
            with pytest.raises(ValueError, match="unsupported unit"):
                curve_power_coefficient(path, [8.0, 12.0])
