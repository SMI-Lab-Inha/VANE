# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.run.lin_config — configuring a deck for linearization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from vane.io.fst_writer import FastInputError, FastInputFile
from vane.run import (
    LinearizationRetryPolicy,
    LinearizationSettings,
    OperatingPoint,
    PlatformPosition,
    configure_linearization,
)
from vane.run.lin_config import FLEXIBILITY_DOFS

if TYPE_CHECKING:
    from pathlib import Path

# Default settings for the NREL 5-MW-like fixture decks (rated wind 11.4 m/s).
_SETTINGS = LinearizationSettings(rated_wind_speed=11.4)

_PRIMARY = """\
        600   TMax        - Total run time (s)
          0   CompInflow  - inflow switch {0=still air; 1=InflowWind}
False         Linearize   - Linearization analysis (flag)
False         CalcSteady  - steady-state operating point (flag)
          3   TrimCase    - {1:yaw; 2:torque; 3:pitch}
      0.001   TrimTol     - tolerance
       0.01   TrimGain    - gain
          0   Twr_Kdmp    - tower damping
          0   Bld_Kdmp    - blade damping
          1   NLinTimes   - number of linearizations
          0   LinTimes    - times (s)
          0   LinInputs   - {0;1;2}
          0   LinOutputs  - {0;1;2}
False         LinOutJac   - full Jacobian debug output
False         LinOutMod   - module linearizations
"ES20.12E3"   OutFmt      - format (string)
          3   WrVTK       - VTK output
"ED.dat"      EDFile      - ElastoDyn file
"IW.dat"      InflowFile  - InflowWind file
"""

_ED = """\
       9.55   RotSpeed    - Initial rotor speed (rpm)
     0.0121   BlPitch(1)  - Blade 1 pitch (deg)
     0.0121   BlPitch(2)  - Blade 2 pitch (deg)
     0.0121   BlPitch(3)  - Blade 3 pitch (deg)
True          GenDOF      - Generator DOF (flag)
True          DrTrDOF     - Drivetrain rotational-flexibility DOF (flag)
True          YawDOF      - Yaw DOF (flag)
"""

# An ElastoDyn deck with neither YawDOF nor DrTrDOF flag (an onshore/simple variant).
_ED_NO_DOF_FLAGS = """\
       9.55   RotSpeed    - Initial rotor speed (rpm)
     0.0121   BlPitch(1)  - Blade 1 pitch (deg)
     0.0121   BlPitch(2)  - Blade 2 pitch (deg)
     0.0121   BlPitch(3)  - Blade 3 pitch (deg)
True          GenDOF      - Generator DOF (flag)
"""

_IW = """\
          3   WindType    - switch
          8   HWindSpeed  - horizontal wind speed (m/s)
          0   PLExp       - power law exponent
"""


def _make_case(tmp_path: Path) -> Path:
    """Write a self-contained primary deck with ED and InflowWind sub-files."""
    (tmp_path / "ED.dat").write_text(_ED, encoding="utf-8")
    (tmp_path / "IW.dat").write_text(_IW, encoding="utf-8")
    fst = tmp_path / "Model.fst"
    fst.write_text(_PRIMARY, encoding="utf-8")
    return fst


class TestConfigureRotating:
    """Tests for configuring a rotating (CalcSteady) operating point."""

    def test_below_rated_uses_torque_trim(self, tmp_path: Path) -> None:
        """Below rated wind speed selects torque trim and sets the operating point."""
        fst = _make_case(tmp_path)
        op = OperatingPoint(wind_speed=8.0, rotor_speed_rpm=9.55, blade_pitch_deg=0.5)
        configure_linearization(
            fst, op, LinearizationSettings(rated_wind_speed=11.4, n_lin_times=36)
        )

        deck = FastInputFile.read(fst)
        assert deck.get("Linearize") == "True"
        assert deck.get("CalcSteady") == "True"
        assert deck.get("TrimCase") == "2"  # torque
        assert deck.get("TrimGain") == "100.0"
        assert deck.get("NLinTimes") == "36"
        assert len(deck.tokens("LinTimes")) == 36
        assert deck.get("LinOutputs") == "1"
        assert deck.get("LinOutJac") == "False"
        assert deck.get("LinOutMod") == "False"
        assert deck.get("OutFmt") == '"ES20.11E3"'
        assert deck.get("WrVTK") == "0"

        elastodyn = FastInputFile.read(tmp_path / "ED.dat")
        assert elastodyn.get("RotSpeed") == "9.55"
        assert elastodyn.get("BlPitch(1)") == "0.5"
        inflow = FastInputFile.read(tmp_path / "IW.dat")
        assert inflow.get("WindType") == "1"
        assert inflow.get("HWindSpeed") == "8.0"
        # The inflow module is enabled so the configured wind is actually used.
        assert deck.get("CompInflow") == "1"

    def test_short_template_tmax_is_raised_for_calc_steady(
        self, tmp_path: Path
    ) -> None:
        """A placeholder TMax is raised so CalcSteady can converge before it."""
        fst = _make_case(tmp_path)
        fst.write_text(_PRIMARY.replace("600   TMax", "1.0   TMax"), "utf-8")
        op = OperatingPoint(wind_speed=8.0, rotor_speed_rpm=9.55, blade_pitch_deg=0.5)
        configure_linearization(fst, op, _SETTINGS)
        assert float(FastInputFile.read(fst).get("TMax")) == 600.0

    def test_long_tmax_and_explicit_sim_time_are_kept(self, tmp_path: Path) -> None:
        """A longer deck TMax, or an explicit sim_time, is never overridden."""
        op = OperatingPoint(wind_speed=8.0, rotor_speed_rpm=9.55, blade_pitch_deg=0.5)
        long_case = tmp_path / "long"
        long_case.mkdir()
        fst = _make_case(long_case)
        fst.write_text(_PRIMARY.replace("600   TMax", "2000   TMax"), "utf-8")
        configure_linearization(fst, op, _SETTINGS)
        assert float(FastInputFile.read(fst).get("TMax")) == 2000.0

        explicit = tmp_path / "explicit"
        explicit.mkdir()
        fst = _make_case(explicit)
        settings = LinearizationSettings(rated_wind_speed=11.4, sim_time=90.0)
        configure_linearization(fst, op, settings)
        assert float(FastInputFile.read(fst).get("TMax")) == 90.0

    def test_output_start_beyond_a_shortened_run_is_reset(self, tmp_path: Path) -> None:
        """A TStart the shortened run would not reach starts the output at 0 s."""
        op = OperatingPoint(wind_speed=8.0, rotor_speed_rpm=9.55, blade_pitch_deg=0.5)
        with_start = _PRIMARY.replace(
            "600   TMax", "600   TMax\n        120   TStart      - first output (s)"
        )
        short = tmp_path / "short"
        short.mkdir()
        fst = _make_case(short)
        fst.write_text(with_start, "utf-8")
        configure_linearization(
            fst, op, LinearizationSettings(rated_wind_speed=11.4, sim_time=90.0)
        )
        deck = FastInputFile.read(fst)
        assert float(deck.get("TMax")) == 90.0
        assert float(deck.get("TStart")) == 0.0

        kept = tmp_path / "kept"
        kept.mkdir()
        fst = _make_case(kept)
        fst.write_text(with_start, "utf-8")
        configure_linearization(fst, op, _SETTINGS)
        assert float(FastInputFile.read(fst).get("TStart")) == 120.0

    def test_differently_cased_shear_keyword_is_set(self, tmp_path: Path) -> None:
        """An InflowWind deck spelling ``PLexp`` still gets uniform inflow."""
        fst = _make_case(tmp_path)
        (tmp_path / "IW.dat").write_text(
            _IW.replace("0   PLExp", "0.2 PLexp"), encoding="utf-8"
        )
        op = OperatingPoint(wind_speed=8.0, rotor_speed_rpm=9.55, blade_pitch_deg=0.5)
        configure_linearization(fst, op, _SETTINGS)
        inflow = FastInputFile.read(tmp_path / "IW.dat")
        assert float(inflow.get("PLexp")) == 0.0

    def test_can_request_module_linearization_files(self, tmp_path: Path) -> None:
        """LinOutMod is an explicit campaign setting for subsystem model export."""
        fst = _make_case(tmp_path)
        op = OperatingPoint(wind_speed=8.0, rotor_speed_rpm=9.55, blade_pitch_deg=0.5)
        configure_linearization(
            fst,
            op,
            LinearizationSettings(rated_wind_speed=11.4, module_linearizations=True),
        )
        assert FastInputFile.read(fst).get("LinOutMod") == "True"

    def test_can_request_full_module_jacobians(self, tmp_path: Path) -> None:
        """LinOutJac is explicit and paired with OpenFAST's all-variable switches."""
        fst = _make_case(tmp_path)
        op = OperatingPoint(wind_speed=8.0, rotor_speed_rpm=9.55, blade_pitch_deg=0.5)
        settings = LinearizationSettings(
            rated_wind_speed=11.4,
            lin_inputs=2,
            lin_outputs=2,
            module_linearizations=True,
            full_jacobians=True,
        )
        configure_linearization(fst, op, settings)
        deck = FastInputFile.read(fst)
        assert deck.get("LinInputs") == "2"
        assert deck.get("LinOutputs") == "2"
        assert deck.get("LinOutJac") == "True"
        assert deck.get("LinOutMod") == "True"

    @pytest.mark.parametrize(
        ("keyword", "settings"),
        [
            (
                "LinOutJac",
                LinearizationSettings(
                    rated_wind_speed=11.4,
                    lin_inputs=2,
                    lin_outputs=2,
                    full_jacobians=True,
                ),
            ),
            (
                "LinOutMod",
                LinearizationSettings(
                    rated_wind_speed=11.4, module_linearizations=True
                ),
            ),
        ],
    )
    def test_requested_export_requires_deck_keyword(
        self, tmp_path: Path, keyword: str, settings: LinearizationSettings
    ) -> None:
        """An explicitly requested export cannot disappear on an older deck."""
        fst = _make_case(tmp_path)
        lines = [line for line in _PRIMARY.splitlines() if keyword not in line]
        fst.write_text("\n".join(lines) + "\n", encoding="utf-8")
        op = OperatingPoint(wind_speed=8.0, rotor_speed_rpm=9.55, blade_pitch_deg=0.5)

        with pytest.raises(FastInputError, match=keyword):
            configure_linearization(fst, op, settings)
        assert FastInputFile.read(fst).get("Linearize") == "False"

    def test_above_rated_uses_pitch_trim(self, tmp_path: Path) -> None:
        """At/above rated wind speed selects pitch trim with the pitch gain."""
        fst = _make_case(tmp_path)
        op = OperatingPoint(wind_speed=16.0, rotor_speed_rpm=12.1, blade_pitch_deg=12.0)
        configure_linearization(fst, op, _SETTINGS)
        deck = FastInputFile.read(fst)
        assert deck.get("TrimCase") == "3"  # pitch
        assert float(deck.get("TrimGain")) == pytest.approx(2e-4)

    def test_above_rated_torque_trim_holds_pitch(self, tmp_path: Path) -> None:
        """``above_rated_trim="torque"`` trims torque at the curve pitch above rated."""
        fst = _make_case(tmp_path)
        op = OperatingPoint(wind_speed=16.0, rotor_speed_rpm=12.1, blade_pitch_deg=12.0)
        settings = LinearizationSettings(
            rated_wind_speed=11.4,
            rated_generator_torque=43_093.5,
            above_rated_trim="torque",
        )
        configure_linearization(fst, op, settings)
        deck = FastInputFile.read(fst)
        assert deck.get("TrimCase") == "2"  # torque
        assert float(deck.get("TrimGain")) == pytest.approx(4309.35)
        assert FastInputFile.read(tmp_path / "ED.dat").get("BlPitch(1)") == "12.0"

    def test_invalid_above_rated_trim_rejected(self) -> None:
        """Only pitch and torque trims exist."""
        with pytest.raises(ValueError, match="above_rated_trim"):
            LinearizationSettings(rated_wind_speed=11.4, above_rated_trim="yaw")

    def test_below_rated_torque_gain_from_rated_torque(self, tmp_path: Path) -> None:
        """rated_generator_torque sets the torque trim gain to rated/10 (Jonkman)."""
        fst = _make_case(tmp_path)
        op = OperatingPoint(wind_speed=8.0, rotor_speed_rpm=9.55, blade_pitch_deg=0.5)
        settings = LinearizationSettings(
            rated_wind_speed=11.4, rated_generator_torque=40000.0
        )
        configure_linearization(fst, op, settings)
        deck = FastInputFile.read(fst)
        assert deck.get("TrimCase") == "2"  # torque
        assert float(deck.get("TrimGain")) == pytest.approx(4000.0)  # 40000 / 10

    def test_platform_initial_conditions_are_set(self, tmp_path: Path) -> None:
        """A platform position writes the ElastoDyn platform DOFs."""
        (tmp_path / "ED.dat").write_text(
            _ED + "0 PtfmSurge - m\n0 PtfmHeave - m\n0 PtfmPitch - deg\n",
            encoding="utf-8",
        )
        (tmp_path / "IW.dat").write_text(_IW, encoding="utf-8")
        fst = tmp_path / "Model.fst"
        fst.write_text(_PRIMARY, encoding="utf-8")
        op = OperatingPoint(
            wind_speed=14.0,
            rotor_speed_rpm=7.5,
            blade_pitch_deg=10.0,
            platform=PlatformPosition(surge=15.0, heave=-0.5, pitch=3.0),
        )
        configure_linearization(fst, op, _SETTINGS)
        ed = FastInputFile.read(tmp_path / "ED.dat")
        assert ed.get("PtfmSurge") == "15.0"
        assert ed.get("PtfmHeave") == "-0.5"
        assert ed.get("PtfmPitch") == "3.0"

    def test_platform_leaves_unset_dofs_unchanged(self, tmp_path: Path) -> None:
        """Only the platform DOFs given are written; the rest keep the deck's value."""
        (tmp_path / "ED.dat").write_text(
            _ED + "1.0 PtfmSurge - m\n2.0 PtfmHeave - m\n3.0 PtfmPitch - deg\n",
            encoding="utf-8",
        )
        (tmp_path / "IW.dat").write_text(_IW, encoding="utf-8")
        fst = tmp_path / "Model.fst"
        fst.write_text(_PRIMARY, encoding="utf-8")
        op = OperatingPoint(
            wind_speed=14.0,
            rotor_speed_rpm=7.5,
            blade_pitch_deg=10.0,
            platform=PlatformPosition(surge=15.0),  # only surge known
        )
        configure_linearization(fst, op, _SETTINGS)
        ed = FastInputFile.read(tmp_path / "ED.dat")
        assert ed.get("PtfmSurge") == "15.0"  # set
        assert ed.get("PtfmHeave") == "2.0"  # left at the deck's value
        assert ed.get("PtfmPitch") == "3.0"  # left at the deck's value


class TestConfigureStandstill:
    """Tests for a standstill / parked operating point."""

    def test_standstill_is_single_linearization(self, tmp_path: Path) -> None:
        """A standstill point disables CalcSteady and uses one linearization."""
        fst = _make_case(tmp_path)
        op = OperatingPoint(wind_speed=0.0, rotor_speed_rpm=0.0, blade_pitch_deg=0.0)
        configure_linearization(fst, op, _SETTINGS)
        deck = FastInputFile.read(fst)
        assert deck.get("Linearize") == "True"
        assert deck.get("CalcSteady") == "False"
        assert deck.get("NLinTimes") == "1"
        assert len(deck.tokens("LinTimes")) == 1

    def test_standstill_keeps_a_short_tmax(self, tmp_path: Path) -> None:
        """The CalcSteady TMax floor does not apply to a single snapshot."""
        fst = _make_case(tmp_path)
        fst.write_text(_PRIMARY.replace("600   TMax", "1.0   TMax"), "utf-8")
        op = OperatingPoint(wind_speed=0.0, rotor_speed_rpm=0.0, blade_pitch_deg=0.0)
        configure_linearization(fst, op, _SETTINGS)
        assert float(FastInputFile.read(fst).get("TMax")) == 1.0

    def test_zero_wind_sets_inflow_to_still_air(self, tmp_path: Path) -> None:
        """A zero-wind point writes steady still air, not the deck's old wind field."""
        fst = _make_case(tmp_path)  # IW.dat starts at HWindSpeed 8, WindType 3
        op = OperatingPoint(wind_speed=0.0, rotor_speed_rpm=0.0, blade_pitch_deg=0.0)
        configure_linearization(fst, op, _SETTINGS)
        inflow = FastInputFile.read(tmp_path / "IW.dat")
        assert inflow.get("WindType") == "1"
        assert inflow.get("HWindSpeed") == "0.0"
        # Still air: the inflow module stays off (CompInflow not forced on).
        assert FastInputFile.read(fst).get("CompInflow") == "0"

    def test_zero_wind_clears_compinflow_from_aero_template(
        self, tmp_path: Path
    ) -> None:
        """A standstill from a CompInflow=1 template disables the inflow module."""
        (tmp_path / "ED.dat").write_text(_ED, encoding="utf-8")
        (tmp_path / "IW.dat").write_text(_IW, encoding="utf-8")
        fst = tmp_path / "Aero.fst"
        fst.write_text(_PRIMARY.replace("0   CompInflow", "1   CompInflow"), "utf-8")
        op = OperatingPoint(wind_speed=0.0, rotor_speed_rpm=0.0, blade_pitch_deg=0.0)
        configure_linearization(fst, op, _SETTINGS)
        assert FastInputFile.read(fst).get("CompInflow") == "0"

    def test_no_inflow_file_is_skipped(self, tmp_path: Path) -> None:
        """A zero-wind deck whose InflowFile is unused configures without error."""
        fst = tmp_path / "NoWind.fst"
        fst.write_text(
            "False Linearize - flag\nFalse CalcSteady - flag\n"
            "1 NLinTimes - -\n0 LinTimes - s\n"
            '"unused" InflowFile - file\n',
            encoding="utf-8",
        )
        op = OperatingPoint(wind_speed=0.0, rotor_speed_rpm=0.0, blade_pitch_deg=0.0)
        configure_linearization(fst, op)  # must not raise
        assert FastInputFile.read(fst).get("Linearize") == "True"

    def test_positive_wind_without_inflow_raises(self, tmp_path: Path) -> None:
        """A positive-wind point with no usable InflowWind file is rejected."""
        from vane.io.fst_writer import FastInputError

        fst = tmp_path / "NoInflow.fst"
        fst.write_text(
            "False Linearize - flag\nFalse CalcSteady - flag\n"
            "1 NLinTimes - -\n0 LinTimes - s\n3 TrimCase - -\n"
            "0.01 TrimGain - -\n0.001 TrimTol - -\n"
            '"unused" InflowFile - file\n',
            encoding="utf-8",
        )
        op = OperatingPoint(wind_speed=8.0, rotor_speed_rpm=9.55, blade_pitch_deg=0.0)
        with pytest.raises(FastInputError, match="no usable InflowWind file"):
            configure_linearization(fst, op, _SETTINGS)


class TestConfigureToleratesVariants:
    """Tests that optional/absent keywords are handled gracefully."""

    def test_missing_optional_keys_are_skipped(self, tmp_path: Path) -> None:
        """A minimal deck without optional keys still configures the essentials."""
        (tmp_path / "IW.dat").write_text(_IW, encoding="utf-8")
        fst = tmp_path / "Min.fst"
        fst.write_text(
            "False Linearize - flag\n"
            "False CalcSteady - flag\n"
            "1 NLinTimes - -\n"
            "0 LinTimes - s\n"
            "3 TrimCase - -\n"
            "0.01 TrimGain - -\n"
            "0.001 TrimTol - -\n"
            '"IW.dat" InflowFile - file\n',
            encoding="utf-8",
        )
        op = OperatingPoint(wind_speed=8.0, rotor_speed_rpm=9.55, blade_pitch_deg=0.0)
        configure_linearization(fst, op, _SETTINGS)
        deck = FastInputFile.read(fst)
        assert deck.get("Linearize") == "True"
        assert deck.get("NLinTimes") == "36"
        assert "WrVTK" not in deck  # optional key absent, skipped without error


_SEASTATE = (
    "2 WaveMod - wave kinematics model\n"
    "True WvDiffQTF - full difference-frequency 2nd-order wave kinematics (flag)\n"
    "True WvSumQTF - full summation-frequency 2nd-order wave kinematics (flag)\n"
)
_HYDRODYN = (
    "1 ExctnMod - wave excitation\n"
    "1 RdtnMod - radiation memory\n"
    "12 NewmanApp - 2nd-order Newman\n"
    "0 MnDrift - mean drift\n"
)
_AERODYN = (
    "1 Wake_Mod - wake model\n2 DBEMT_Mod - dynamic BEMT\n3 UA_Mod - unsteady aero\n"
)
_SERVODYN = (
    "5 PCMode - pitch control mode\n"
    "0.0 PitNeut(1) - blade 1 passive pitch neutral (deg)\n"
    "0.0 PitNeut(2) - blade 2 passive pitch neutral (deg)\n"
    "0.0 PitNeut(3) - blade 3 passive pitch neutral (deg)\n"
    "5 VSContrl - variable-speed control\n"
    "0 HSSBrMode - HSS brake\n"
    "0 YCMode - yaw control\n"
    "False GenTiStr - generator start by time (flag)\n"
    "0.0 TimGenOn - generator-on time (s)\n"
    "True GenTiStp - generator stop by time (flag)\n"
    "500.0 TimGenOf - generator-off time (s, below the 600 s TMax in the fixture)\n"
    "100.0 VS_RtGnSp - rated generator speed (rpm)\n"
    "2.5 VS_Rgn2K - region-2 torque constant (Nm/(rad/s)^2)\n"
    "10.0 VS_SlPc - rated-torque slip percentage (%)\n"
    "0.0 VS_RtTq - rated generator torque (Nm)\n"
)


def _make_offshore_case(tmp_path: Path) -> Path:
    """Write a deck with SeaState/HydroDyn/AeroDyn/ServoDyn offshore sub-files."""
    (tmp_path / "ED.dat").write_text(_ED, encoding="utf-8")
    (tmp_path / "IW.dat").write_text(_IW, encoding="utf-8")
    (tmp_path / "SeaState.dat").write_text(_SEASTATE, encoding="utf-8")
    (tmp_path / "HydroDyn.dat").write_text(_HYDRODYN, encoding="utf-8")
    (tmp_path / "AeroDyn.dat").write_text(_AERODYN, encoding="utf-8")
    (tmp_path / "ServoDyn.dat").write_text(_SERVODYN, encoding="utf-8")
    fst = tmp_path / "Model.fst"
    fst.write_text(
        _PRIMARY
        + '"SeaState.dat" SeaStFile - SeaState file\n'
        + '"HydroDyn.dat" HydroFile - HydroDyn file\n'
        + '"AeroDyn.dat" AeroFile - AeroDyn file\n'
        + '"ServoDyn.dat" ServoFile - ServoDyn file\n',
        encoding="utf-8",
    )
    return fst


class TestConfigureOffshore:
    """Tests for the offshore (aero-hydro-servo-elastic) module configuration."""

    def test_offshore_makes_modules_linearization_compatible(
        self, tmp_path: Path
    ) -> None:
        """``offshore()`` rewrites the hydro/aero/servo modules for linearization."""
        fst = _make_offshore_case(tmp_path)
        op = OperatingPoint(wind_speed=16.0, rotor_speed_rpm=7.5, blade_pitch_deg=12.0)
        # Above rated, the constant-torque flattening needs a rated torque for VS_RtTq.
        configure_linearization(
            fst,
            op,
            LinearizationSettings.offshore(
                rated_wind_speed=11.4, rated_generator_torque=19_624_000.0
            ),
        )

        sea_state = FastInputFile.read(tmp_path / "SeaState.dat")
        assert sea_state.get("WaveMod") == "0"
        assert sea_state.get("WvDiffQTF") == "False"
        assert sea_state.get("WvSumQTF") == "False"
        hydro = FastInputFile.read(tmp_path / "HydroDyn.dat")
        assert hydro.get("ExctnMod") == "2"
        assert hydro.get("RdtnMod") == "2"
        assert hydro.get("NewmanApp") == "0"
        aero = FastInputFile.read(tmp_path / "AeroDyn.dat")
        assert aero.get("UA_Mod") == "0"
        assert aero.get("DBEMT_Mod") == "-1"  # frozen wake, valid for linearization
        servo = FastInputFile.read(tmp_path / "ServoDyn.dat")
        assert servo.get("PCMode") == "0"
        assert servo.get("VSContrl") == "1"

    def test_offshore_overrides_apply(self, tmp_path: Path) -> None:
        """Keyword overrides on ``offshore()`` reach the deck (e.g. no .ss file)."""
        fst = _make_offshore_case(tmp_path)
        op = OperatingPoint(wind_speed=16.0, rotor_speed_rpm=7.5, blade_pitch_deg=12.0)
        settings = LinearizationSettings.offshore(
            rated_wind_speed=11.4,
            hydro_radiation_mode=0,
            hydro_excitation_mode=0,
            rated_generator_torque=19_624_000.0,
        )
        configure_linearization(fst, op, settings)
        hydro = FastInputFile.read(tmp_path / "HydroDyn.dat")
        assert hydro.get("RdtnMod") == "0"
        assert hydro.get("ExctnMod") == "0"

    def test_default_settings_leave_modules_untouched(self, tmp_path: Path) -> None:
        """Default (onshore) settings do not modify the offshore module files."""
        fst = _make_offshore_case(tmp_path)
        op = OperatingPoint(wind_speed=16.0, rotor_speed_rpm=7.5, blade_pitch_deg=12.0)
        configure_linearization(fst, op, _SETTINGS)  # module fields None/False
        assert FastInputFile.read(tmp_path / "SeaState.dat").get("WaveMod") == "2"
        assert FastInputFile.read(tmp_path / "SeaState.dat").get("WvSumQTF") == "True"
        assert FastInputFile.read(tmp_path / "HydroDyn.dat").get("RdtnMod") == "1"
        assert FastInputFile.read(tmp_path / "AeroDyn.dat").get("UA_Mod") == "3"
        assert FastInputFile.read(tmp_path / "ServoDyn.dat").get("PCMode") == "5"

    def test_offshore_configures_legacy_aerodyn_keywords(self, tmp_path: Path) -> None:
        """A legacy AeroDyn deck gets steady airfoils and a frozen BEMT wake.

        Legacy files select the airfoil model with ``AFAeroMod`` (must be 1 when
        linearizing) and the wake with ``WakeMod`` (dynamic BEMT = 2 cannot be
        linearized); ``UA_Mod`` is absent, so editing only it left unsteady aero on
        and OpenFAST failed late.
        """
        fst = _make_offshore_case(tmp_path)
        legacy = (
            "2 WakeMod - wake/induction model {0=none; 1=BEMT; 2=DBEMT; 3=OLAF}\n"
            "2 AFAeroMod - airfoil aerodynamics {1=steady; 2=Beddoes-Leishman}\n"
            "False FrozenWake - assume frozen wake during linearization (flag)\n"
            "1 DBEMT_Mod - dynamic BEMT model {used only when WakeMod=2}\n"
            "3 UAMod - unsteady aero model switch {used only when AFAeroMod=2}\n"
        )
        (tmp_path / "AeroDyn.dat").write_text(legacy, encoding="utf-8")
        op = OperatingPoint(wind_speed=8.0, rotor_speed_rpm=9.0, blade_pitch_deg=0.0)

        configure_linearization(
            fst, op, LinearizationSettings.offshore(rated_wind_speed=11.4)
        )

        aero = FastInputFile.read(tmp_path / "AeroDyn.dat")
        assert aero.get("AFAeroMod") == "1"
        assert aero.get("WakeMod") == "1"
        assert aero.get("FrozenWake") == "True"
        # Inert legacy switches are left alone rather than given modern values.
        assert aero.get("DBEMT_Mod") == "1"
        assert aero.get("UAMod") == "3"

    def test_legacy_aerodyn_without_frozen_wake_request(self, tmp_path: Path) -> None:
        """``dynamic_wake_mode=0`` keeps plain BEMT with FrozenWake off."""
        fst = _make_offshore_case(tmp_path)
        (tmp_path / "AeroDyn.dat").write_text(
            "2 WakeMod - wake model\n1 AFAeroMod - airfoil model\n"
            "True FrozenWake - frozen wake (flag)\n",
            encoding="utf-8",
        )
        op = OperatingPoint(wind_speed=8.0, rotor_speed_rpm=9.0, blade_pitch_deg=0.0)

        configure_linearization(
            fst,
            op,
            LinearizationSettings(
                rated_wind_speed=11.4, unsteady_aero_mode=0, dynamic_wake_mode=0
            ),
        )

        aero = FastInputFile.read(tmp_path / "AeroDyn.dat")
        assert aero.get("WakeMod") == "1"
        assert aero.get("FrozenWake") == "False"

    def test_offshore_tolerates_onshore_deck(self, tmp_path: Path) -> None:
        """``offshore()`` on a deck without hydro/aero/servo files does not raise."""
        fst = _make_case(tmp_path)  # only ED + InflowWind sub-files
        op = OperatingPoint(wind_speed=8.0, rotor_speed_rpm=9.55, blade_pitch_deg=0.0)
        configure_linearization(
            fst, op, LinearizationSettings.offshore(rated_wind_speed=11.4)
        )
        assert FastInputFile.read(fst).get("Linearize") == "True"


class TestConfigureServodynTorque:
    """Region-aware generator-torque handling for a trimmed ServoDyn deck."""

    def test_below_rated_keeps_generator_off(self, tmp_path: Path) -> None:
        """Below rated the generator is kept off so the torque trim owns the torque."""
        fst = _make_offshore_case(tmp_path)
        op = OperatingPoint(wind_speed=8.0, rotor_speed_rpm=9.55, blade_pitch_deg=0.5)
        configure_linearization(
            fst, op, LinearizationSettings(rated_wind_speed=11.4, trim_controller=True)
        )

        servo = FastInputFile.read(tmp_path / "ServoDyn.dat")
        # Linearization requires timed generator control both ways; the start and stop
        # are pushed beyond TMax so the generator never engages (ServoDyn returns 0).
        assert servo.get("GenTiStr") == "True"
        assert servo.get("GenTiStp") == "True"
        assert float(servo.get("TimGenOn")) > 600.0
        assert float(servo.get("TimGenOf")) > 600.0
        # The simple-VS law is neutralised (Rgn2K=0, tiny RtGnSp) so VSContrl=1 init
        # validation passes for the dummy deck; no rated torque is needed below rated.
        assert float(servo.get("VS_RtGnSp")) == pytest.approx(9.999e-9)
        assert float(servo.get("VS_Rgn2K")) == pytest.approx(0.0)
        # VS_RtTq is left at the deck's dummy value (the generator never applies it).
        assert float(servo.get("VS_RtTq")) == pytest.approx(0.0)
        # With PCMode disabled, the passive pitch spring must hold the settled pitch;
        # leaving its original zero neutral would erase this operating point.
        assert float(servo.get("PitNeut(1)")) == pytest.approx(0.5)
        assert float(servo.get("PitNeut(2)")) == pytest.approx(0.5)
        assert float(servo.get("PitNeut(3)")) == pytest.approx(0.5)

    def test_below_rated_generator_off_time_follows_sim_time(
        self, tmp_path: Path
    ) -> None:
        """With an explicit TMax the generator-on time clears it (sim_time + 1000 s)."""
        fst = _make_offshore_case(tmp_path)
        op = OperatingPoint(wind_speed=8.0, rotor_speed_rpm=9.55, blade_pitch_deg=0.5)
        settings = LinearizationSettings(
            rated_wind_speed=11.4, trim_controller=True, sim_time=1200.0
        )
        configure_linearization(fst, op, settings)

        servo = FastInputFile.read(tmp_path / "ServoDyn.dat")
        assert float(servo.get("TimGenOn")) == pytest.approx(2200.0)  # 1200 + 1000

    def test_generator_event_time_follows_deck_tmax(self, tmp_path: Path) -> None:
        """With no sim_time, the timed event clears the deck's *actual* TMax.

        A fixed constant could fall inside a long-running template; the event time must
        exceed the deck's TMax (here a large 250000 s) so the timed start/stop never
        fires during the linearization.
        """
        big_tmax = 250_000.0
        (tmp_path / "ED.dat").write_text(_ED, encoding="utf-8")
        (tmp_path / "IW.dat").write_text(_IW, encoding="utf-8")
        (tmp_path / "SeaState.dat").write_text(_SEASTATE, encoding="utf-8")
        (tmp_path / "HydroDyn.dat").write_text(_HYDRODYN, encoding="utf-8")
        (tmp_path / "AeroDyn.dat").write_text(_AERODYN, encoding="utf-8")
        (tmp_path / "ServoDyn.dat").write_text(_SERVODYN, encoding="utf-8")
        fst = tmp_path / "Model.fst"
        fst.write_text(
            _PRIMARY.replace("600   TMax", f"{big_tmax}   TMax")
            + '"SeaState.dat" SeaStFile - SeaState file\n'
            + '"HydroDyn.dat" HydroFile - HydroDyn file\n'
            + '"AeroDyn.dat" AeroFile - AeroDyn file\n'
            + '"ServoDyn.dat" ServoFile - ServoDyn file\n',
            encoding="utf-8",
        )
        op = OperatingPoint(wind_speed=8.0, rotor_speed_rpm=9.55, blade_pitch_deg=0.5)
        configure_linearization(
            fst, op, LinearizationSettings(rated_wind_speed=11.4, trim_controller=True)
        )

        servo = FastInputFile.read(tmp_path / "ServoDyn.dat")
        # The generator-on time exceeds the (large) deck TMax, not a fixed 99990 s.
        assert float(servo.get("TimGenOn")) > big_tmax

    def test_above_rated_uses_constant_torque(self, tmp_path: Path) -> None:
        """At/above rated the variable-speed law is flattened to a constant torque."""
        fst = _make_offshore_case(tmp_path)
        op = OperatingPoint(wind_speed=16.0, rotor_speed_rpm=7.5, blade_pitch_deg=12.0)
        settings = LinearizationSettings(
            rated_wind_speed=11.4,
            trim_controller=True,
            rated_generator_torque=19_624_000.0,
        )
        configure_linearization(fst, op, settings)

        servo = FastInputFile.read(tmp_path / "ServoDyn.dat")
        # Flattened law: tiny rated speed/slip, zero region-2 gain, constant VS_RtTq.
        assert float(servo.get("VS_RtGnSp")) == pytest.approx(9.999e-9)
        assert float(servo.get("VS_SlPc")) == pytest.approx(9.999e-9)
        assert float(servo.get("VS_Rgn2K")) == pytest.approx(0.0)
        assert float(servo.get("VS_RtTq")) == pytest.approx(19_624_000.0)
        # Generator engaged from the start; the timed stop (500 s < 600 s TMax in the
        # fixture) is pushed beyond TMax (with GenTiStp=True) so it stays on all run.
        assert servo.get("GenTiStr") == "True"
        assert servo.get("GenTiStp") == "True"
        assert float(servo.get("TimGenOn")) == pytest.approx(0.0)
        assert float(servo.get("TimGenOf")) > 600.0

    def test_above_rated_dummy_rttq_no_rated_torque_raises(
        self, tmp_path: Path
    ) -> None:
        """Flattening with no rated torque and a zero VS_RtTq is rejected up front.

        Even on a simple-law deck (``VSContrl = 1``), a zero ``VS_RtTq`` would trim
        against zero generator torque, so it must fail clearly rather than silently
        produce a wrong operating point — and, because the rejection is validated before
        any write, the deck must be left byte-for-byte unchanged.
        """
        fst = _make_offshore_case(tmp_path)
        servo_path = tmp_path / "ServoDyn.dat"
        servo_path.write_text(
            _SERVODYN.replace("5 VSContrl", "1 VSContrl"), encoding="utf-8"
        )
        before = {p: p.read_bytes() for p in sorted(tmp_path.glob("*.dat"))}
        before[fst] = fst.read_bytes()
        op = OperatingPoint(wind_speed=16.0, rotor_speed_rpm=7.5, blade_pitch_deg=12.0)
        settings = LinearizationSettings(rated_wind_speed=11.4, trim_controller=True)
        with pytest.raises(FastInputError, match="no positive generator torque"):
            configure_linearization(fst, op, settings)
        # No partial reconfiguration: every deck file is untouched after the rejection.
        for path, content in before.items():
            assert path.read_bytes() == content

    def test_above_rated_uses_positive_deck_rttq_when_no_rated(
        self, tmp_path: Path
    ) -> None:
        """With no rated torque but a positive deck VS_RtTq, that deck value is used."""
        (tmp_path / "ED.dat").write_text(_ED, encoding="utf-8")
        (tmp_path / "IW.dat").write_text(_IW, encoding="utf-8")
        (tmp_path / "SeaState.dat").write_text(_SEASTATE, encoding="utf-8")
        (tmp_path / "HydroDyn.dat").write_text(_HYDRODYN, encoding="utf-8")
        (tmp_path / "AeroDyn.dat").write_text(_AERODYN, encoding="utf-8")
        # A simple-controller template (VSContrl=1) that already carries a real
        # positive VS_RtTq.
        (tmp_path / "ServoDyn.dat").write_text(
            _SERVODYN.replace(
                "0.0 VS_RtTq - rated generator torque (Nm)",
                "18000.0 VS_RtTq - rated generator torque (Nm)",
            ).replace("5 VSContrl", "1 VSContrl"),
            encoding="utf-8",
        )
        fst = tmp_path / "Model.fst"
        fst.write_text(
            _PRIMARY
            + '"SeaState.dat" SeaStFile - SeaState file\n'
            + '"HydroDyn.dat" HydroFile - HydroDyn file\n'
            + '"AeroDyn.dat" AeroFile - AeroDyn file\n'
            + '"ServoDyn.dat" ServoFile - ServoDyn file\n',
            encoding="utf-8",
        )
        op = OperatingPoint(wind_speed=16.0, rotor_speed_rpm=7.5, blade_pitch_deg=12.0)
        configure_linearization(
            fst, op, LinearizationSettings(rated_wind_speed=11.4, trim_controller=True)
        )

        servo = FastInputFile.read(tmp_path / "ServoDyn.dat")
        assert float(servo.get("VS_RtGnSp")) == pytest.approx(9.999e-9)
        assert float(servo.get("VS_RtTq")) == pytest.approx(18000.0)

    @pytest.mark.parametrize("vs_control", ["5", "0", "3"])
    def test_above_rated_placeholder_rttq_on_non_simple_law_deck_raises(
        self, tmp_path: Path, vs_control: str
    ) -> None:
        """A non-simple-law deck's positive VS_RtTq placeholder is never trusted.

        DLL-controlled decks carry a placeholder such as ``9999.9`` in ``VS_RtTq``.
        Above rated without ``rated_generator_torque`` that would become the engaged
        generator's constant torque and trim a wrong operating point, so the
        configuration is rejected before any deck file is written.
        """
        fst = _make_offshore_case(tmp_path)
        (tmp_path / "ServoDyn.dat").write_text(
            _SERVODYN.replace(
                "0.0 VS_RtTq - rated generator torque (Nm)",
                "9999.9 VS_RtTq - rated generator torque (Nm)",
            ).replace("5 VSContrl", f"{vs_control} VSContrl"),
            encoding="utf-8",
        )
        before = {p: p.read_bytes() for p in sorted(tmp_path.glob("*.dat"))}
        before[fst] = fst.read_bytes()
        op = OperatingPoint(wind_speed=16.0, rotor_speed_rpm=7.5, blade_pitch_deg=12.0)

        with pytest.raises(FastInputError, match="set rated_generator_torque"):
            configure_linearization(
                fst, op, LinearizationSettings.offshore(rated_wind_speed=11.4)
            )
        for path, content in before.items():
            assert path.read_bytes() == content

    def test_placeholder_rttq_deck_accepts_explicit_rated_torque(
        self, tmp_path: Path
    ) -> None:
        """With rated_generator_torque the DLL deck's placeholder is overwritten."""
        fst = _make_offshore_case(tmp_path)
        (tmp_path / "ServoDyn.dat").write_text(
            _SERVODYN.replace(
                "0.0 VS_RtTq - rated generator torque (Nm)",
                "9999.9 VS_RtTq - rated generator torque (Nm)",
            ),
            encoding="utf-8",
        )
        op = OperatingPoint(wind_speed=16.0, rotor_speed_rpm=7.5, blade_pitch_deg=12.0)
        configure_linearization(
            fst,
            op,
            LinearizationSettings.offshore(
                rated_wind_speed=11.4, rated_generator_torque=43093.55
            ),
        )

        servo = FastInputFile.read(tmp_path / "ServoDyn.dat")
        assert servo.get("VSContrl") == "1"
        assert float(servo.get("VS_RtTq")) == pytest.approx(43093.55)

    def test_below_rated_placeholder_rttq_deck_needs_no_rated_torque(
        self, tmp_path: Path
    ) -> None:
        """Below rated the generator is held off, so the placeholder is harmless."""
        fst = _make_offshore_case(tmp_path)
        op = OperatingPoint(wind_speed=8.0, rotor_speed_rpm=9.0, blade_pitch_deg=0.0)
        configure_linearization(
            fst, op, LinearizationSettings.offshore(rated_wind_speed=11.4)
        )

        servo = FastInputFile.read(tmp_path / "ServoDyn.dat")
        assert servo.get("VSContrl") == "1"

    def test_flatten_backstop_uses_original_vscontrl(self, tmp_path: Path) -> None:
        """The in-edit backstop judges the deck torque by the original VSContrl."""
        from vane.run.lin_config import _flatten_to_constant_torque

        servo_path = tmp_path / "ServoDyn.dat"
        servo_path.write_text(_SERVODYN, encoding="utf-8")
        servo = FastInputFile.read(servo_path)
        with pytest.raises(FastInputError, match="placeholder"):
            _flatten_to_constant_torque(
                servo,
                LinearizationSettings(rated_wind_speed=11.4, trim_controller=True),
                False,
            )

    def test_above_rated_retry_still_writes_rated_torque(self, tmp_path: Path) -> None:
        """A retry attempt keeps writing VS_RtTq from the rated torque.

        Regression: the retry ladder escalates the trim gain via an override and must
        not drop ``rated_generator_torque``; otherwise an above-rated retry would still
        flatten the variable-speed law but leave ``VS_RtTq`` at the deck's dummy value,
        trimming to the wrong constant torque.
        """
        fst = _make_offshore_case(tmp_path)
        op = OperatingPoint(wind_speed=16.0, rotor_speed_rpm=7.5, blade_pitch_deg=12.0)
        base = LinearizationSettings(
            rated_wind_speed=11.4,
            trim_controller=True,
            rated_generator_torque=19_624_000.0,
        )
        retry = LinearizationRetryPolicy().settings_for_attempt(base, 2)
        configure_linearization(fst, op, retry)

        servo = FastInputFile.read(tmp_path / "ServoDyn.dat")
        assert float(servo.get("VS_RtTq")) == pytest.approx(19_624_000.0)
        assert float(servo.get("VS_RtGnSp")) == pytest.approx(9.999e-9)

    def test_above_rated_single_linearization_engages_generator(
        self, tmp_path: Path
    ) -> None:
        """An un-trimmed above-rated point (calc_steady=False) still engages it.

        With no CalcSteady torque trim, nothing else supplies the steady generator
        torque, so an above-rated single linearization must apply the constant rated
        torque itself — not run with the generator off.
        """
        fst = _make_offshore_case(tmp_path)
        op = OperatingPoint(wind_speed=16.0, rotor_speed_rpm=7.5, blade_pitch_deg=12.0)
        settings = LinearizationSettings(
            rated_wind_speed=11.4,
            trim_controller=True,
            calc_steady=False,
            rated_generator_torque=19_624_000.0,
        )
        configure_linearization(fst, op, settings)

        servo = FastInputFile.read(tmp_path / "ServoDyn.dat")
        assert float(servo.get("VS_RtTq")) == pytest.approx(19_624_000.0)
        assert servo.get("GenTiStr") == "True"
        assert servo.get("GenTiStp") == "True"
        assert float(servo.get("TimGenOn")) == pytest.approx(0.0)  # generator engaged
        assert float(servo.get("TimGenOf")) > 600.0

    def test_above_rated_explicit_times_write_complete_sweep(
        self, tmp_path: Path
    ) -> None:
        """A settled rotating point may use an explicit full-revolution time set."""
        fst = _make_offshore_case(tmp_path)
        op = OperatingPoint(
            wind_speed=18.0, rotor_speed_rpm=7.56, blade_pitch_deg=15.46
        )
        times = tuple(4492.0 + index * 0.22 for index in range(36))
        settings = LinearizationSettings(
            rated_wind_speed=11.4,
            n_lin_times=36,
            calc_steady=False,
            sim_time=4500.0,
            linearization_times=times,
            trim_controller=True,
            rated_generator_torque=19_786_800.0,
        )
        configure_linearization(fst, op, settings)

        deck = FastInputFile.read(fst)
        assert deck.get("CalcSteady") == "False"
        assert deck.get("NLinTimes") == "36"
        assert tuple(float(value) for value in deck.tokens("LinTimes")) == times

    def test_explicit_times_beyond_deck_tmax_are_rejected(self, tmp_path: Path) -> None:
        """Without sim_time, explicit times are bounded by the deck's own TMax."""
        fst = _make_offshore_case(tmp_path)
        before = fst.read_text()
        op = OperatingPoint(
            wind_speed=18.0, rotor_speed_rpm=7.56, blade_pitch_deg=15.46
        )
        settings = LinearizationSettings(
            rated_wind_speed=11.4,
            n_lin_times=2,
            calc_steady=False,
            linearization_times=(599.0, 601.0),
            trim_controller=True,
            rated_generator_torque=19_786_800.0,
        )
        with pytest.raises(FastInputError, match="exceeds the effective TMax"):
            configure_linearization(fst, op, settings)
        assert fst.read_text() == before

    def test_below_rated_single_linearization_is_rejected(self, tmp_path: Path) -> None:
        """A below-rated rotating point without CalcSteady is rejected, not zero-torque.

        Below rated the steady generator torque is the region-2 (rotor-performance)
        law, which VANE does not compute; without a torque trim nothing supplies it, so
        the unsupported combination fails clearly rather than linearizing at zero.
        """
        fst = _make_offshore_case(tmp_path)
        before = fst.read_bytes()
        op = OperatingPoint(wind_speed=8.0, rotor_speed_rpm=9.55, blade_pitch_deg=0.5)
        settings = LinearizationSettings(
            rated_wind_speed=11.4, trim_controller=True, calc_steady=False
        )
        with pytest.raises(FastInputError, match="calc_steady=True"):
            configure_linearization(fst, op, settings)
        assert fst.read_bytes() == before  # rejected before any deck write

    def test_standstill_holds_generator_off_and_validates(self, tmp_path: Path) -> None:
        """A parked/standstill linearization still satisfies ServoDyn validation.

        A standstill point is not a rotating trim, but it is still a linearization, so
        ServoDyn requires GenTiStr/GenTiStp=True and valid VSContrl=1 params. The
        generator is held off (no constant torque flattened) and the simple-VS law is
        neutralised so a DLL-style dummy deck passes init validation.
        """
        fst = _make_offshore_case(tmp_path)
        op = OperatingPoint(wind_speed=0.0, rotor_speed_rpm=0.0, blade_pitch_deg=0.0)
        configure_linearization(
            fst, op, LinearizationSettings(rated_wind_speed=11.4, trim_controller=True)
        )

        servo = FastInputFile.read(tmp_path / "ServoDyn.dat")
        assert servo.get("PCMode") == "0"
        assert servo.get("GenTiStr") == "True"
        assert servo.get("GenTiStp") == "True"
        assert float(servo.get("TimGenOn")) > 600.0  # generator held off
        assert float(servo.get("TimGenOf")) > 600.0
        # Simple-VS law neutralised (not flattened to a rated torque): no rated needed.
        assert float(servo.get("VS_RtGnSp")) == pytest.approx(9.999e-9)
        assert float(servo.get("VS_Rgn2K")) == pytest.approx(0.0)
        assert float(servo.get("VS_RtTq")) == pytest.approx(0.0)  # deck dummy untouched

    def test_trim_controller_off_leaves_torque_keys(self, tmp_path: Path) -> None:
        """With trim_controller off the ServoDyn torque keys are not touched."""
        fst = _make_offshore_case(tmp_path)
        op = OperatingPoint(wind_speed=16.0, rotor_speed_rpm=7.5, blade_pitch_deg=12.0)
        configure_linearization(fst, op, _SETTINGS)  # trim_controller False

        servo = FastInputFile.read(tmp_path / "ServoDyn.dat")
        assert servo.get("PCMode") == "5"  # untouched
        assert float(servo.get("PitNeut(1)")) == pytest.approx(0.0)
        assert servo.get("GenTiStr") == "False"
        assert float(servo.get("VS_RtGnSp")) == pytest.approx(100.0)

    def test_servo_disabled_deck_is_left_untouched(self, tmp_path: Path) -> None:
        """With CompServo=0 the inert ServoFile is neither configured nor validated.

        An above-rated trim_controller point with a dummy VS_RtTq=0 would normally be
        rejected by the flattening, but when ServoDyn is switched off the ServoFile is
        inert (OpenFAST never validates it), so configuration must skip it, not fail.
        """
        (tmp_path / "ED.dat").write_text(_ED, encoding="utf-8")
        (tmp_path / "IW.dat").write_text(_IW, encoding="utf-8")
        (tmp_path / "ServoDyn.dat").write_text(_SERVODYN, encoding="utf-8")
        fst = tmp_path / "Model.fst"
        fst.write_text(
            _PRIMARY.replace(
                "0   CompInflow  - inflow switch {0=still air; 1=InflowWind}",
                "0   CompInflow  - inflow switch {0=still air; 1=InflowWind}\n"
                "          0   CompServo   - servo switch {0=off}",
            )
            + '"ServoDyn.dat" ServoFile - ServoDyn file\n',
            encoding="utf-8",
        )
        op = OperatingPoint(wind_speed=16.0, rotor_speed_rpm=7.5, blade_pitch_deg=12.0)
        # Would raise in _flatten_to_constant_torque if the inert file were edited.
        configure_linearization(
            fst, op, LinearizationSettings(rated_wind_speed=11.4, trim_controller=True)
        )

        servo = FastInputFile.read(tmp_path / "ServoDyn.dat")
        assert servo.get("PCMode") == "5"  # untouched — ServoDyn is off
        assert servo.get("GenTiStr") == "False"
        assert float(servo.get("VS_RtGnSp")) == pytest.approx(100.0)
        assert float(servo.get("VS_RtTq")) == pytest.approx(0.0)


class TestRatedWindSpeedRequired:
    """The trim region split needs the turbine's own rated wind speed."""

    def test_rotating_point_without_rated_is_rejected_before_writing(
        self, tmp_path: Path
    ) -> None:
        """No turbine-independent default: a rotating trim needs rated_wind_speed."""
        fst = _make_case(tmp_path)
        before = fst.read_text(encoding="utf-8")
        op = OperatingPoint(wind_speed=11.0, rotor_speed_rpm=9.0, blade_pitch_deg=0.0)
        with pytest.raises(ValueError, match="rated_wind_speed is required"):
            configure_linearization(fst, op, LinearizationSettings())
        assert fst.read_text(encoding="utf-8") == before

    def test_standstill_point_needs_no_rated(self, tmp_path: Path) -> None:
        """A parked point has no trim region, so no rated wind is needed."""
        fst = _make_case(tmp_path)
        op = OperatingPoint(wind_speed=0.0, rotor_speed_rpm=0.0, blade_pitch_deg=0.0)
        configure_linearization(fst, op, LinearizationSettings())
        assert FastInputFile.read(fst).get("Linearize") == "True"

    @pytest.mark.parametrize(("comp_servo", "required"), [(0, False), (1, True)])
    def test_disabled_servodyn_untrimmed_point_needs_no_rated(
        self, tmp_path: Path, comp_servo: int, required: bool
    ) -> None:
        """Without CalcSteady and with ServoDyn off, nothing picks a trim region."""
        fst = _make_case(tmp_path)
        fst.write_text(
            fst.read_text(encoding="utf-8")
            + f"{comp_servo} CompServo - servo switch\n",
            encoding="utf-8",
        )
        op = OperatingPoint(wind_speed=11.0, rotor_speed_rpm=9.0, blade_pitch_deg=0.0)
        settings = LinearizationSettings(
            calc_steady=False, trim_controller=True, rated_generator_torque=4.0e4
        )
        if required:
            with pytest.raises(ValueError, match="rated_wind_speed is required"):
                configure_linearization(fst, op, settings)
        else:
            configure_linearization(fst, op, settings)
            assert FastInputFile.read(fst).get("Linearize") == "True"

    def test_region_split_uses_the_given_rated(self) -> None:
        """An IEA 15-MW rated speed puts 11 m/s above rated, unlike the 5-MW value."""
        iea15 = LinearizationSettings(rated_wind_speed=10.59)
        assert not iea15.is_below_rated(11.0)
        assert LinearizationSettings(rated_wind_speed=11.4).is_below_rated(11.0)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -3.0])
    def test_invalid_rated_rejected(self, value: float) -> None:
        """A rated wind speed must be finite and positive when set."""
        with pytest.raises(ValueError, match="rated_wind_speed must be finite"):
            LinearizationSettings(rated_wind_speed=value)


class TestSettingsValidation:
    """LinearizationSettings rejects settings a linearization cannot accept."""

    def test_defaults_and_offshore_are_valid(self) -> None:
        """The built-in configurations construct without error."""
        LinearizationSettings(rated_wind_speed=11.4)
        LinearizationSettings.offshore(rated_wind_speed=11.4)

    def test_rejects_non_positive_counts_and_tolerances(self) -> None:
        """Counts and tolerances must be physical."""
        with pytest.raises(ValueError, match="n_lin_times must be >= 1"):
            LinearizationSettings(rated_wind_speed=11.4, n_lin_times=0)
        with pytest.raises(ValueError, match="trim_tol must be positive"):
            LinearizationSettings(rated_wind_speed=11.4, trim_tol=0.0)
        with pytest.raises(
            ValueError, match="rated_wind_speed must be finite and positive"
        ):
            LinearizationSettings(rated_wind_speed=-1.0)
        with pytest.raises(ValueError, match="sim_time must be positive"):
            LinearizationSettings(rated_wind_speed=11.4, sim_time=0.0)
        with pytest.raises(ValueError, match="non-negative"):
            LinearizationSettings(rated_wind_speed=11.4, twr_kdmp=-1.0)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        (
            ({"linearization_times": (1.0,)}, "requires calc_steady=False"),
            (
                {
                    "calc_steady": False,
                    "n_lin_times": 2,
                    "linearization_times": (1.0,),
                },
                "length must equal n_lin_times",
            ),
            (
                {
                    "calc_steady": False,
                    "n_lin_times": 2,
                    "linearization_times": (1.0, 1.0),
                },
                "strictly increasing",
            ),
            (
                {
                    "calc_steady": False,
                    "n_lin_times": 2,
                    "linearization_times": (-1.0, 1.0),
                },
                "finite and non-negative",
            ),
            (
                {
                    "calc_steady": False,
                    "n_lin_times": 2,
                    "sim_time": 1.5,
                    "linearization_times": (1.0, 2.0),
                },
                "must not exceed sim_time",
            ),
        ),
    )
    def test_rejects_invalid_explicit_times(
        self, kwargs: dict[str, object], message: str
    ) -> None:
        """Explicit times fail before any OpenFAST deck can be changed."""
        with pytest.raises(ValueError, match=message):
            LinearizationSettings(**kwargs)  # type: ignore[arg-type]

    def test_rejects_non_finite_numeric_fields(self) -> None:
        """A NaN/inf numeric setting is rejected (it slips past every < / <= check)."""
        nan = float("nan")
        with pytest.raises(ValueError, match="trim_tol must be finite"):
            LinearizationSettings(rated_wind_speed=11.4, trim_tol=nan)
        with pytest.raises(
            ValueError, match="rated_wind_speed must be finite and positive"
        ):
            LinearizationSettings(rated_wind_speed=nan)
        with pytest.raises(ValueError, match="twr_kdmp must be finite"):
            LinearizationSettings(rated_wind_speed=11.4, twr_kdmp=nan)
        with pytest.raises(ValueError, match="trim_gain_torque must be finite"):
            LinearizationSettings(rated_wind_speed=11.4, trim_gain_torque=float("inf"))
        with pytest.raises(ValueError, match="sim_time must be finite"):
            LinearizationSettings(rated_wind_speed=11.4, sim_time=nan)

    def test_rejects_non_positive_trim_gains(self) -> None:
        """A zero gain never trims and a negative one diverges: both are rejected."""
        for name in ("trim_gain_torque", "trim_gain_pitch"):
            for bad in (0.0, -1e-3):
                with pytest.raises(ValueError, match=f"{name} must be positive"):
                    LinearizationSettings(rated_wind_speed=11.4, **{name: bad})

    def test_rejects_bad_rated_generator_torque(self) -> None:
        """rated_generator_torque, when set, must be finite and positive."""
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="rated_generator_torque must be"):
                LinearizationSettings(rated_wind_speed=11.4, rated_generator_torque=bad)

    def test_rejects_bad_trim_gain_torque_override(self) -> None:
        """trim_gain_torque_override, when set, must be finite and positive."""
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="trim_gain_torque_override must be"):
                LinearizationSettings(
                    rated_wind_speed=11.4, trim_gain_torque_override=bad
                )

    def test_resolved_trim_gain_torque(self) -> None:
        """The torque gain comes from rated/10 when given, else trim_gain_torque."""
        assert (
            LinearizationSettings(rated_wind_speed=11.4).resolved_trim_gain_torque()
            == 100.0
        )
        explicit = LinearizationSettings(rated_wind_speed=11.4, trim_gain_torque=250.0)
        assert explicit.resolved_trim_gain_torque() == 250.0
        derived = LinearizationSettings(
            rated_wind_speed=11.4, rated_generator_torque=40000.0
        )
        assert derived.resolved_trim_gain_torque() == pytest.approx(4000.0)
        # rated_generator_torque takes precedence over an explicit trim_gain_torque.
        both = LinearizationSettings(
            rated_wind_speed=11.4,
            trim_gain_torque=250.0,
            rated_generator_torque=40000.0,
        )
        assert both.resolved_trim_gain_torque() == pytest.approx(4000.0)
        # trim_gain_torque_override takes precedence over both.
        overridden = LinearizationSettings(
            rated_wind_speed=11.4,
            trim_gain_torque=250.0,
            rated_generator_torque=40000.0,
            trim_gain_torque_override=12345.0,
        )
        assert overridden.resolved_trim_gain_torque() == pytest.approx(12345.0)

    def test_rejects_non_finite_or_fractional_n_lin_times(self) -> None:
        """n_lin_times (an integer count) rejects NaN/inf and fractional values.

        It can arrive as a float from untyped config; a NaN slips past ``< 1`` and would
        be written as a non-integer ``NLinTimes``.
        """
        with pytest.raises(ValueError, match="n_lin_times must be finite"):
            LinearizationSettings(rated_wind_speed=11.4, n_lin_times=float("nan"))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="n_lin_times must be a whole number"):
            LinearizationSettings(rated_wind_speed=11.4, n_lin_times=2.5)  # type: ignore[arg-type]

    def test_whole_float_n_lin_times_is_normalized_to_int(self) -> None:
        """A whole float (e.g. from JSON config) is coerced to int, not left a float."""
        settings = LinearizationSettings(rated_wind_speed=11.4, n_lin_times=36.0)  # type: ignore[arg-type]
        assert settings.n_lin_times == 36
        assert isinstance(settings.n_lin_times, int)

    def test_rejects_empty_out_fmt(self) -> None:
        """An empty output format is rejected."""
        with pytest.raises(ValueError, match="out_fmt"):
            LinearizationSettings(rated_wind_speed=11.4, out_fmt='""')

    def test_rejects_non_linearizable_switches(self) -> None:
        """Module switches must select linearizable models."""
        with pytest.raises(ValueError, match="lin_inputs"):
            LinearizationSettings(rated_wind_speed=11.4, lin_inputs=3)
        with pytest.raises(ValueError, match="full_jacobians requires"):
            LinearizationSettings(rated_wind_speed=11.4, full_jacobians=True)
        with pytest.raises(ValueError, match="full_jacobians requires"):
            LinearizationSettings(
                rated_wind_speed=11.4, lin_inputs=2, lin_outputs=1, full_jacobians=True
            )
        with pytest.raises(ValueError, match="wave_mod"):
            LinearizationSettings(rated_wind_speed=11.4, wave_mod=2)
        with pytest.raises(ValueError, match="hydro_radiation_mode"):
            LinearizationSettings(
                rated_wind_speed=11.4, hydro_radiation_mode=1
            )  # convolution
        with pytest.raises(ValueError, match="hydro_excitation_mode"):
            LinearizationSettings(rated_wind_speed=11.4, hydro_excitation_mode=1)
        with pytest.raises(ValueError, match="unsteady_aero_mode"):
            LinearizationSettings(rated_wind_speed=11.4, unsteady_aero_mode=3)
        with pytest.raises(ValueError, match="dynamic_wake_mode"):
            LinearizationSettings(rated_wind_speed=11.4, dynamic_wake_mode=2)

    def test_none_switches_are_allowed(self) -> None:
        """``None`` (leave the deck unchanged) is always valid for a module switch."""
        settings = LinearizationSettings(
            rated_wind_speed=11.4,
            wave_mod=None,
            hydro_radiation_mode=None,
            unsteady_aero_mode=None,
        )
        assert settings.wave_mod is None

    def test_offshore_override_is_validated(self) -> None:
        """An invalid override through ``offshore()`` is rejected too."""
        with pytest.raises(ValueError, match="hydro_radiation_mode"):
            LinearizationSettings.offshore(
                rated_wind_speed=11.4, hydro_radiation_mode=1
            )


class TestConfigureStructuralDofs:
    """Tests for ElastoDyn structural-DOF flag control during linearization."""

    def test_default_disables_yaw_and_drivetrain_dofs(self, tmp_path: Path) -> None:
        """By default YawDOF and DrTrDOF are written False in the ED deck."""
        fst = _make_case(tmp_path)
        op = OperatingPoint(wind_speed=8.0, rotor_speed_rpm=9.55, blade_pitch_deg=0.5)
        configure_linearization(fst, op, LinearizationSettings(rated_wind_speed=11.4))

        ed = FastInputFile.read(tmp_path / "ED.dat")
        assert ed.get("YawDOF") == "False"
        assert ed.get("DrTrDOF") == "False"
        # An unrelated DOF flag is untouched.
        assert ed.get("GenDOF") == "True"

    def test_offshore_settings_also_disable_the_dofs(self, tmp_path: Path) -> None:
        """The offshore preset carries the same DOF defaults."""
        fst = _make_case(tmp_path)
        op = OperatingPoint(wind_speed=8.0, rotor_speed_rpm=9.55, blade_pitch_deg=0.5)
        configure_linearization(
            fst, op, LinearizationSettings.offshore(rated_wind_speed=11.4)
        )

        ed = FastInputFile.read(tmp_path / "ED.dat")
        assert ed.get("YawDOF") == "False"
        assert ed.get("DrTrDOF") == "False"

    def test_override_re_enables_drivetrain_dof(self, tmp_path: Path) -> None:
        """An override merges on top of the defaults: DrTrDOF back on, YawDOF off."""
        fst = _make_case(tmp_path)
        op = OperatingPoint(wind_speed=8.0, rotor_speed_rpm=9.55, blade_pitch_deg=0.5)
        configure_linearization(
            fst,
            op,
            LinearizationSettings(
                rated_wind_speed=11.4, structural_dofs={"DrTrDOF": True}
            ),
        )

        ed = FastInputFile.read(tmp_path / "ED.dat")
        assert ed.get("DrTrDOF") == "True"  # re-enabled by the override
        assert ed.get("YawDOF") == "False"  # default still applies

    def test_override_can_enable_an_arbitrary_dof(self, tmp_path: Path) -> None:
        """A general flag (e.g. GenDOF) can be set through structural_dofs."""
        fst = _make_case(tmp_path)
        op = OperatingPoint(wind_speed=8.0, rotor_speed_rpm=9.55, blade_pitch_deg=0.5)
        configure_linearization(
            fst,
            op,
            LinearizationSettings(
                rated_wind_speed=11.4, structural_dofs={"GenDOF": False}
            ),
        )

        ed = FastInputFile.read(tmp_path / "ED.dat")
        assert ed.get("GenDOF") == "False"

    def test_default_is_a_no_op_when_deck_lacks_the_flags(self, tmp_path: Path) -> None:
        """A default-disable on a deck without the flag is skipped, not an error."""
        (tmp_path / "ED.dat").write_text(_ED_NO_DOF_FLAGS, encoding="utf-8")
        (tmp_path / "IW.dat").write_text(_IW, encoding="utf-8")
        fst = tmp_path / "Model.fst"
        fst.write_text(_PRIMARY, encoding="utf-8")
        op = OperatingPoint(wind_speed=8.0, rotor_speed_rpm=9.55, blade_pitch_deg=0.5)

        configure_linearization(
            fst, op, LinearizationSettings(rated_wind_speed=11.4)
        )  # no error

        ed = FastInputFile.read(tmp_path / "ED.dat")
        assert "YawDOF" not in ed
        assert "DrTrDOF" not in ed

    def test_explicit_flag_absent_from_deck_raises(self, tmp_path: Path) -> None:
        """An explicitly-requested flag the ED deck lacks fails clearly."""
        (tmp_path / "ED.dat").write_text(_ED_NO_DOF_FLAGS, encoding="utf-8")
        (tmp_path / "IW.dat").write_text(_IW, encoding="utf-8")
        fst = tmp_path / "Model.fst"
        fst.write_text(_PRIMARY, encoding="utf-8")
        op = OperatingPoint(wind_speed=8.0, rotor_speed_rpm=9.55, blade_pitch_deg=0.5)

        with pytest.raises(FastInputError, match="DrTrDOF"):
            configure_linearization(
                fst,
                op,
                LinearizationSettings(
                    rated_wind_speed=11.4, structural_dofs={"DrTrDOF": False}
                ),
            )

    def test_non_bool_value_rejected(self) -> None:
        """A non-bool DOF value is rejected at settings construction."""
        with pytest.raises(ValueError, match="must be a bool"):
            LinearizationSettings(rated_wind_speed=11.4, structural_dofs={"YawDOF": 1})  # type: ignore[dict-item]

    def test_empty_key_rejected(self) -> None:
        """An empty DOF flag name is rejected at settings construction."""
        with pytest.raises(ValueError, match="non-empty"):
            LinearizationSettings(rated_wind_speed=11.4, structural_dofs={"  ": False})

    def test_gendof_forced_on_for_trimmed_rotating_point(self, tmp_path: Path) -> None:
        """A trimmed rotating point forces GenDOF on (the trim regulates via it)."""
        ed = _ED.replace("True          GenDOF", "False         GenDOF")
        (tmp_path / "ED.dat").write_text(ed, encoding="utf-8")
        (tmp_path / "IW.dat").write_text(_IW, encoding="utf-8")
        fst = tmp_path / "Model.fst"
        fst.write_text(_PRIMARY, encoding="utf-8")
        op = OperatingPoint(wind_speed=8.0, rotor_speed_rpm=9.55, blade_pitch_deg=0.5)

        configure_linearization(fst, op, LinearizationSettings(rated_wind_speed=11.4))

        assert FastInputFile.read(tmp_path / "ED.dat").get("GenDOF") == "True"

    def test_explicit_gendof_off_is_respected_when_rotating(
        self, tmp_path: Path
    ) -> None:
        """An explicit GenDOF=False wins over the trim's GenDOF enforcement."""
        fst = _make_case(tmp_path)  # ED deck has GenDOF True
        op = OperatingPoint(wind_speed=8.0, rotor_speed_rpm=9.55, blade_pitch_deg=0.5)
        configure_linearization(
            fst,
            op,
            LinearizationSettings(
                rated_wind_speed=11.4, structural_dofs={"GenDOF": False}
            ),
        )

        assert FastInputFile.read(tmp_path / "ED.dat").get("GenDOF") == "False"

    def test_gendof_not_forced_for_standstill(self, tmp_path: Path) -> None:
        """A standstill point is not trimmed, so GenDOF is left at the deck's value."""
        ed = _ED.replace("True          GenDOF", "False         GenDOF")
        (tmp_path / "ED.dat").write_text(ed, encoding="utf-8")
        (tmp_path / "IW.dat").write_text(_IW, encoding="utf-8")
        fst = tmp_path / "Model.fst"
        fst.write_text(_PRIMARY, encoding="utf-8")
        op = OperatingPoint(wind_speed=0.0, rotor_speed_rpm=0.0, blade_pitch_deg=0.0)

        configure_linearization(fst, op, LinearizationSettings(rated_wind_speed=11.4))

        assert FastInputFile.read(tmp_path / "ED.dat").get("GenDOF") == "False"

    def test_resolved_and_explicit_helpers(self) -> None:
        """resolved_structural_dofs merges defaults; explicit lists only overrides."""
        flexible = dict.fromkeys(FLEXIBILITY_DOFS, True)
        default = LinearizationSettings(rated_wind_speed=11.4)
        assert default.resolved_structural_dofs() == {
            **flexible,
            "YawDOF": False,
            "DrTrDOF": False,
        }
        assert default.explicit_structural_dofs() == frozenset()

        override = LinearizationSettings(
            rated_wind_speed=11.4, structural_dofs={"DrTrDOF": True, "EdgeDOF": False}
        )
        assert override.resolved_structural_dofs() == {
            **flexible,
            "EdgeDOF": False,
            "YawDOF": False,
            "DrTrDOF": True,
        }
        assert override.explicit_structural_dofs() == frozenset({"DrTrDOF", "EdgeDOF"})

    def test_rigid_deck_is_made_flexible(self, tmp_path: Path) -> None:
        """Blade and tower DOFs a deck ships off are enabled; an override keeps one off.

        The published IEA 3.4-MW deck has FlapDOF1/FlapDOF2/EdgeDOF off, so it
        linearized without any blade mode until the defaults enabled them.
        """
        fst = _make_case(tmp_path)
        ed = tmp_path / "ED.dat"
        rigid = "".join(
            f"False         {flag}    - blade DOF (flag)\n"
            for flag in ("FlapDOF1", "FlapDOF2", "EdgeDOF")
        )
        ed.write_text(_ED + rigid, encoding="utf-8")
        op = OperatingPoint(wind_speed=8.0, rotor_speed_rpm=9.55, blade_pitch_deg=0.5)
        configure_linearization(fst, op, _SETTINGS)
        deck = FastInputFile.read(ed)
        for flag in ("FlapDOF1", "FlapDOF2", "EdgeDOF"):
            assert deck.get(flag) == "True"

        configure_linearization(
            fst,
            op,
            LinearizationSettings(
                rated_wind_speed=11.4, structural_dofs={"EdgeDOF": False}
            ),
        )
        assert FastInputFile.read(ed).get("EdgeDOF") == "False"


_ABOVE_RATED = OperatingPoint(
    wind_speed=16.0, rotor_speed_rpm=7.5, blade_pitch_deg=12.0
)
_OFFSHORE = LinearizationSettings.offshore(
    rated_wind_speed=11.4, rated_generator_torque=19_624_000.0
)


class TestLinearizationRestrictionFixes:
    """Auto-fixes for OpenFAST linearization restrictions behind existing flags."""

    def test_still_water_resets_options_openfast_rejects(self, tmp_path: Path) -> None:
        """wave_mod=0 also clears the wave options still water cannot carry."""
        fst = _make_offshore_case(tmp_path)
        (tmp_path / "SeaState.dat").write_text(
            _SEASTATE + "1 WaveDirMod - directional spreading\n"
            "1 ConstWaveMod - constrained wave\n"
            "2 WaveStMod - wave stretching\n"
            "9.0 MCFD - MacCamy-Fuchs diameter (m)\n"
            "30 WaveDir - incident wave heading (deg)\n"
            "1 CurrMod - current model\n"
            '"DEFAULT" CurrSSDir - sub-surface current heading (deg)\n',
            encoding="utf-8",
        )
        configure_linearization(fst, _ABOVE_RATED, _OFFSHORE)

        sea_state = FastInputFile.read(tmp_path / "SeaState.dat")
        assert sea_state.get("WaveMod") == "0"
        assert sea_state.get("WaveDirMod") == "0"
        assert sea_state.get("ConstWaveMod") == "0"
        assert sea_state.get("WaveStMod") == "0"
        assert float(sea_state.get("MCFD")) == 0.0
        assert float(sea_state.get("CurrSSDir")) == 30.0

    def test_still_water_keeps_compatible_options(self, tmp_path: Path) -> None:
        """Options already valid in still water are left exactly as they were."""
        fst = _make_offshore_case(tmp_path)
        (tmp_path / "SeaState.dat").write_text(
            _SEASTATE + "1 WaveStMod - wave stretching\n"
            "0 MCFD - MacCamy-Fuchs diameter (m)\n"
            "0 CurrMod - current model\n"
            '"DEFAULT" CurrSSDir - unused without CurrMod=1\n',
            encoding="utf-8",
        )
        configure_linearization(fst, _ABOVE_RATED, _OFFSHORE)

        sea_state = FastInputFile.read(tmp_path / "SeaState.dat")
        assert sea_state.get("WaveStMod") == "1"
        assert sea_state.get("MCFD") == "0"
        assert sea_state.get("CurrSSDir") == '"DEFAULT"'

    @pytest.mark.parametrize(
        ("rdtn_dt", "expected"), [("0.025", '"DEFAULT"'), ("0.0125", "0.0125")]
    )
    def test_state_space_hydro_aligns_radiation_step(
        self, tmp_path: Path, rdtn_dt: str, expected: str
    ) -> None:
        """A numeric RdtnDT that differs from DT becomes DEFAULT under state space."""
        fst = _make_offshore_case(tmp_path)
        fst.write_text(
            "0.0125 DT - time step (s)\n" + fst.read_text("utf-8"), encoding="utf-8"
        )
        (tmp_path / "HydroDyn.dat").write_text(
            "1 PotMod - WAMIT\n0 ExctnMod - excitation\n0 RdtnMod - radiation\n"
            f"{rdtn_dt} RdtnDT - radiation step (s)\n",
            encoding="utf-8",
        )
        configure_linearization(fst, _ABOVE_RATED, _OFFSHORE)
        assert FastInputFile.read(tmp_path / "HydroDyn.dat").get("RdtnDT") == expected

    def test_radiation_step_left_alone_without_wamit(self, tmp_path: Path) -> None:
        """Without WAMIT potential flow HydroDyn ignores RdtnDT, so it is kept."""
        fst = _make_offshore_case(tmp_path)
        fst.write_text(
            "0.0125 DT - time step (s)\n" + fst.read_text("utf-8"), encoding="utf-8"
        )
        (tmp_path / "HydroDyn.dat").write_text(
            "0 PotMod - none\n0 ExctnMod - excitation\n0 RdtnMod - radiation\n"
            "0.025 RdtnDT - radiation step (s)\n",
            encoding="utf-8",
        )
        configure_linearization(fst, _ABOVE_RATED, _OFFSHORE)
        assert FastInputFile.read(tmp_path / "HydroDyn.dat").get("RdtnDT") == "0.025"

    @pytest.mark.parametrize(("deck_value", "expected"), [("3", "1"), ("2", "2")])
    def test_trim_controller_makes_generator_model_linearizable(
        self, tmp_path: Path, deck_value: str, expected: str
    ) -> None:
        """A user generator model (3) falls back to the simple model; 2 is kept."""
        fst = _make_offshore_case(tmp_path)
        servo = tmp_path / "ServoDyn.dat"
        servo.write_text(
            servo.read_text("utf-8") + f"{deck_value} GenModel - generator model\n",
            encoding="utf-8",
        )
        configure_linearization(fst, _ABOVE_RATED, _OFFSHORE)
        assert FastInputFile.read(servo).get("GenModel") == expected

    def test_trim_controller_turns_off_flap_and_cable_control(
        self, tmp_path: Path
    ) -> None:
        """AfCmode/CCmode at 5 would still load the controller library: zeroed."""
        fst = _make_offshore_case(tmp_path)
        servo = tmp_path / "ServoDyn.dat"
        servo.write_text(
            servo.read_text("utf-8")
            + "5 AfCmode - flap control\n5 CCmode - cable control\n",
            encoding="utf-8",
        )
        configure_linearization(fst, _ABOVE_RATED, _OFFSHORE)
        deck = FastInputFile.read(servo)
        assert deck.get("AfCmode") == "0"
        assert deck.get("CCmode") == "0"


class TestDisabledModulesAreInert:
    """A module switched off in the primary deck is neither edited nor confined."""

    _SWITCHES = (
        "0 CompAero - aero switch\n"
        "0 CompServo - servo switch\n"
        "0 CompHydro - hydro switch\n"
        "0 CompSeaSt - sea-state switch\n"
    )

    def test_disabled_module_files_are_not_edited(self, tmp_path: Path) -> None:
        """offshore() leaves the files of switched-off modules byte-identical."""
        fst = _make_offshore_case(tmp_path)
        fst.write_text(self._SWITCHES + fst.read_text("utf-8"), encoding="utf-8")
        names = ("SeaState.dat", "HydroDyn.dat", "AeroDyn.dat", "ServoDyn.dat")
        before = {name: (tmp_path / name).read_text("utf-8") for name in names}

        configure_linearization(fst, _ABOVE_RATED, _OFFSHORE)

        for name in names:
            assert (tmp_path / name).read_text("utf-8") == before[name]

    def test_shared_file_of_disabled_module_is_not_confined(
        self, tmp_path: Path
    ) -> None:
        """A disabled module's file may live outside the case directory."""
        case = tmp_path / "case"
        case.mkdir()
        fst = _make_offshore_case(case)
        shared = tmp_path / "shared"
        shared.mkdir()
        for name in ("SeaState.dat", "HydroDyn.dat", "AeroDyn.dat"):
            (case / name).replace(shared / name)
        text = (
            fst.read_text("utf-8")
            .replace('"SeaState.dat"', '"../shared/SeaState.dat"')
            .replace('"HydroDyn.dat"', '"../shared/HydroDyn.dat"')
            .replace('"AeroDyn.dat"', '"../shared/AeroDyn.dat"')
        )
        fst.write_text(text, encoding="utf-8")
        with pytest.raises(FastInputError, match="outside the case directory"):
            configure_linearization(fst, _ABOVE_RATED, _OFFSHORE, confine_to=case)

        fst.write_text(self._SWITCHES + text, encoding="utf-8")
        configure_linearization(fst, _ABOVE_RATED, _OFFSHORE, confine_to=case)
        assert FastInputFile.read(shared / "HydroDyn.dat").get("RdtnMod") == "1"


class TestUnsteadyAeroEndRadius:
    """``ua_end_radius`` ends unsteady aerodynamics inboard of the blade tip."""

    _OP = OperatingPoint(wind_speed=8.0, rotor_speed_rpm=9.0, blade_pitch_deg=0.0)
    _AERO = (
        "1 Wake_Mod - wake model\n"
        "4 UA_Mod - unsteady aero\n"
        "True TipLoss - Prandtl tip loss (flag)\n"
        "0.05 UAStartRad - starting radius for UA\n"
        "1 UAEndRad - ending radius for UA\n"
    )

    def _case(self, tmp_path: Path, aero: str) -> Path:
        fst = _make_offshore_case(tmp_path)
        (tmp_path / "AeroDyn.dat").write_text(aero, encoding="utf-8")
        return fst

    def test_writes_uaendrad(self, tmp_path: Path) -> None:
        """The requested radius fraction replaces the deck's ``UAEndRad``."""
        fst = self._case(tmp_path, self._AERO)
        configure_linearization(
            fst,
            self._OP,
            LinearizationSettings(rated_wind_speed=11.4, ua_end_radius=0.99),
        )
        aero = FastInputFile.read(tmp_path / "AeroDyn.dat")
        assert float(aero.get("UAEndRad")) == pytest.approx(0.99)
        assert aero.get("UA_Mod") == "4"  # the model itself is left alone

    def test_default_leaves_uaendrad(self, tmp_path: Path) -> None:
        """Without the setting the deck's ``UAEndRad`` stands."""
        fst = self._case(tmp_path, self._AERO)
        configure_linearization(fst, self._OP, _SETTINGS)
        assert FastInputFile.read(tmp_path / "AeroDyn.dat").get("UAEndRad") == "1"

    def test_missing_line_is_rejected_before_writing(self, tmp_path: Path) -> None:
        """A modern file without ``UAEndRad`` (read as 1) cannot take the request."""
        aero = self._AERO.replace("1 UAEndRad - ending radius for UA\n", "")
        fst = self._case(tmp_path, aero)
        before = fst.read_text(encoding="utf-8")
        with pytest.raises(FastInputError, match="no UAEndRad line"):
            configure_linearization(
                fst,
                self._OP,
                LinearizationSettings(rated_wind_speed=11.4, ua_end_radius=0.99),
            )
        assert fst.read_text(encoding="utf-8") == before

    def test_legacy_deck_without_line_is_tolerated(self, tmp_path: Path) -> None:
        """A legacy file (steady airfoils only) has no tip state to protect."""
        fst = self._case(tmp_path, "1 WakeMod - wake model\n1 AFAeroMod - airfoil\n")
        configure_linearization(
            fst,
            self._OP,
            LinearizationSettings(rated_wind_speed=11.4, ua_end_radius=0.99),
        )
        assert "UAEndRad" not in FastInputFile.read(tmp_path / "AeroDyn.dat")

    def test_disabled_aerodyn_is_not_checked(self, tmp_path: Path) -> None:
        """With ``CompAero = 0`` the AeroDyn file is inert and not required to match."""
        fst = self._case(tmp_path, "1 Wake_Mod - wake model\n")
        fst.write_text(
            "0 CompAero - aero switch\n" + fst.read_text("utf-8"), encoding="utf-8"
        )
        configure_linearization(
            fst,
            self._OP,
            LinearizationSettings(rated_wind_speed=11.4, ua_end_radius=0.99),
        )
        assert "UAEndRad" not in FastInputFile.read(tmp_path / "AeroDyn.dat")

    def test_aerodyn_file_is_confined(self, tmp_path: Path) -> None:
        """The setting edits AeroDyn, so a shared AeroDyn file is refused."""
        case = tmp_path / "case"
        case.mkdir()
        fst = self._case(case, self._AERO)
        shared = tmp_path / "shared"
        shared.mkdir()
        (case / "AeroDyn.dat").replace(shared / "AeroDyn.dat")
        fst.write_text(
            fst.read_text("utf-8").replace('"AeroDyn.dat"', '"../shared/AeroDyn.dat"'),
            encoding="utf-8",
        )
        settings = LinearizationSettings(rated_wind_speed=11.4, ua_end_radius=0.99)
        with pytest.raises(FastInputError, match="outside the case directory"):
            configure_linearization(fst, self._OP, settings, confine_to=case)
        # Without the setting AeroDyn is not edited, so it may stay shared.
        configure_linearization(fst, self._OP, _SETTINGS, confine_to=case)

    @pytest.mark.parametrize("value", [0.0, -0.1, 1.01, float("nan"), True, "0.9"])
    def test_invalid_values_are_rejected(self, value: object) -> None:
        """Only a number in (0, 1] is a radius fraction."""
        with pytest.raises(ValueError, match="ua_end_radius"):
            LinearizationSettings(ua_end_radius=value)  # type: ignore[arg-type]

    def test_one_is_accepted(self) -> None:
        """``1`` is AeroDyn's own default and a valid explicit value."""
        assert LinearizationSettings(ua_end_radius=1).ua_end_radius == 1


class TestHydroRadiationFit:
    """``hydro_radiation_fit`` generates the state-space radiation file per case."""

    def test_requires_state_space_radiation(self) -> None:
        """The fit only makes sense for RdtnMod = 2."""
        assert LinearizationSettings(
            hydro_radiation_mode=2, hydro_radiation_fit=True
        ).hydro_radiation_fit
        assert LinearizationSettings.offshore(hydro_radiation_fit=True)
        for mode in (None, 0):
            with pytest.raises(ValueError, match="hydro_radiation_mode=2"):
                LinearizationSettings(
                    hydro_radiation_mode=mode, hydro_radiation_fit=True
                )

    def test_must_be_a_bool(self) -> None:
        """A truthy non-bool is rejected rather than silently accepted."""
        with pytest.raises(ValueError, match="must be a bool"):
            LinearizationSettings(
                hydro_radiation_mode=2,
                hydro_radiation_fit=1,  # type: ignore[arg-type]
            )
