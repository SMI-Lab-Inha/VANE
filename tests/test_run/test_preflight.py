# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.run.preflight — pre-flight deck validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from vane.run import (
    OPENFAST_MAX_PATH_LENGTH,
    DeckPathLengthError,
    DeckReferenceError,
    DeckValidationError,
    LinearizationSettings,
    check_deck_path_lengths,
    deck_file_closure,
    validate_deck,
)

_PRIMARY = """\
"SD.dat" ServoFile - ServoDyn file
"SeaSt.dat" SeaStFile - SeaState file
"HD.dat" HydroFile - HydroDyn file
"""

_SERVO_DLL = """\
5 PCMode - pitch control mode {0=none; 5=DLL}
5 VSContrl - torque control mode {1=simple; 5=DLL}
0 HSSBrMode - HSS brake mode
0 YCMode - yaw control mode
"""

_SERVO_SIMPLE = """\
0 PCMode - pitch control mode
1 VSContrl - torque control mode
0 HSSBrMode - HSS brake mode
0 YCMode - yaw control mode
"""

_SEASTATE_WAVES = """\
2 WaveMod - wave model {0=still water; 2=JONSWAP}
"""

_SEASTATE_STILL = """\
0 WaveMod - wave model
"""

_HYDRO_SS = """\
2 RdtnMod - radiation memory model {0=none; 1=conv; 2=state-space}
2 ExctnMod - excitation model {0=none; 2=state-space}
"hydro/marin" PotFile - root of the potential-flow data files
"""

_HYDRO_CONV = """\
1 RdtnMod - radiation memory model
0 ExctnMod - excitation model
"hydro/marin" PotFile - root
"""

# State-space hydro (needs the .ss/.ssexctn files) with a 2nd-order force switch active.
_HYDRO_SS_SECOND_ORDER = """\
2 RdtnMod - radiation memory model
2 ExctnMod - excitation model
"hydro/marin" PotFile - root of the potential-flow data files
1 MnDrift - mean-drift 2nd-order forces
0 NewmanApp - Newman approximation
0 DiffQTF - difference-frequency QTF
0 SumQTF - sum-frequency QTF
"""

_AERO_BAD = """\
3 UA_Mod - unsteady aerodynamics model {non-linearizable}
2 DBEMT_Mod - dynamic wake model {non-linearizable}
"""

_AERO_GOOD = """\
0 UA_Mod - unsteady aerodynamics model {quasi-steady}
-1 DBEMT_Mod - dynamic wake model {frozen}
"""


def _write_deck(
    tmp_path: Path,
    *,
    servo: str | None = None,
    seastate: str | None = None,
    hydro: str | None = None,
    aero: str | None = None,
) -> Path:
    """Write a primary deck plus only the requested sub-files."""
    lines = []
    if servo is not None:
        (tmp_path / "SD.dat").write_text(servo, encoding="utf-8")
        lines.append('"SD.dat" ServoFile - ServoDyn file')
    if seastate is not None:
        (tmp_path / "SeaSt.dat").write_text(seastate, encoding="utf-8")
        lines.append('"SeaSt.dat" SeaStFile - SeaState file')
    if hydro is not None:
        (tmp_path / "HD.dat").write_text(hydro, encoding="utf-8")
        lines.append('"HD.dat" HydroFile - HydroDyn file')
    if aero is not None:
        (tmp_path / "AD.dat").write_text(aero, encoding="utf-8")
        lines.append('"AD.dat" AeroFile - AeroDyn file')
    fst = tmp_path / "Model.fst"
    fst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fst


def _make_ss_files(tmp_path: Path) -> None:
    """Create the state-space coefficient files the PotFile root points at."""
    (tmp_path / "hydro").mkdir(exist_ok=True)
    (tmp_path / "hydro" / "marin.ss").write_text("ss", encoding="utf-8")
    (tmp_path / "hydro" / "marin.ssexctn").write_text("ssx", encoding="utf-8")


class TestController:
    """DLL-controller checks."""

    def test_dll_without_trim_is_rejected(self, tmp_path: Path) -> None:
        """A Bladed/DLL controller with no trim replacement is rejected."""
        fst = _write_deck(tmp_path, servo=_SERVO_DLL)
        with pytest.raises(DeckValidationError, match="DLL controller"):
            validate_deck(fst)

    def test_dll_with_trim_controller_passes(self, tmp_path: Path) -> None:
        """trim_controller replaces the DLL, so the deck passes."""
        fst = _write_deck(tmp_path, servo=_SERVO_DLL)
        validate_deck(
            fst, LinearizationSettings(rated_wind_speed=11.4, trim_controller=True)
        )

    def test_simple_controller_passes(self, tmp_path: Path) -> None:
        """A non-DLL controller is fine even without trim replacement."""
        fst = _write_deck(tmp_path, servo=_SERVO_SIMPLE)
        validate_deck(fst)

    def test_disabled_servo_module_is_skipped(self, tmp_path: Path) -> None:
        """A DLL ServoDyn file is inert when CompServo=0, so it must not be rejected."""
        (tmp_path / "SD.dat").write_text(_SERVO_DLL, encoding="utf-8")
        fst = tmp_path / "Model.fst"
        fst.write_text(
            "0 CompServo - compute control {0=off}\n"
            '"SD.dat" ServoFile - ServoDyn file\n',
            encoding="utf-8",
        )
        validate_deck(fst)  # CompServo=0 → the DLL file is not inspected


class TestSeaState:
    """Wave-model checks."""

    def test_waves_rejected(self, tmp_path: Path) -> None:
        """Active waves prevent linearization."""
        fst = _write_deck(tmp_path, seastate=_SEASTATE_WAVES)
        with pytest.raises(DeckValidationError, match="waves enabled"):
            validate_deck(fst)

    def test_waves_overridden_by_settings(self, tmp_path: Path) -> None:
        """wave_mod=0 in settings clears the wave check."""
        fst = _write_deck(tmp_path, seastate=_SEASTATE_WAVES)
        validate_deck(fst, LinearizationSettings(rated_wind_speed=11.4, wave_mod=0))

    def test_second_order_wave_kinematics_rejected(self, tmp_path: Path) -> None:
        """WvDiffQTF/WvSumQTF on is rejected unless the settings switch them off."""
        fst = _write_deck(
            tmp_path,
            seastate=_SEASTATE_STILL + "True WvSumQTF - sum-frequency kinematics\n",
        )
        with pytest.raises(DeckValidationError, match="WvSumQTF"):
            validate_deck(fst)
        validate_deck(
            fst,
            LinearizationSettings(
                rated_wind_speed=11.4, disable_second_order_hydro=True
            ),
        )

    def test_still_water_passes(self, tmp_path: Path) -> None:
        """Still water is fine."""
        fst = _write_deck(tmp_path, seastate=_SEASTATE_STILL)
        validate_deck(fst)


class TestHydroDyn:
    """Radiation/excitation and state-space file checks."""

    def test_convolution_radiation_rejected(self, tmp_path: Path) -> None:
        """Convolution radiation cannot be linearized."""
        fst = _write_deck(tmp_path, hydro=_HYDRO_CONV)
        with pytest.raises(DeckValidationError, match="convolution radiation"):
            validate_deck(fst)

    def test_state_space_without_files_rejected(self, tmp_path: Path) -> None:
        """State-space hydro with missing .ss/.ssexctn files is rejected."""
        fst = _write_deck(tmp_path, hydro=_HYDRO_SS)
        with pytest.raises(DeckValidationError, match=r"'\.ss' coefficient file"):
            validate_deck(fst)

    def test_state_space_with_files_passes(self, tmp_path: Path) -> None:
        """State-space hydro passes when the coefficient files exist."""
        _make_ss_files(tmp_path)
        fst = _write_deck(tmp_path, hydro=_HYDRO_SS)
        validate_deck(fst)

    def test_no_potential_flow_ignores_radiation_models(self, tmp_path: Path) -> None:
        """With PotMod=0, RdtnMod/ExctnMod are unused and need no .ss files."""
        fst = _write_deck(
            tmp_path, hydro="0 PotMod - potential-flow model\n" + _HYDRO_SS
        )
        validate_deck(fst)
        validate_deck(fst, LinearizationSettings.offshore(rated_wind_speed=10.0))

    def test_potential_flow_still_needs_state_space_files(self, tmp_path: Path) -> None:
        """With PotMod=1 the missing .ss file is still reported."""
        fst = _write_deck(
            tmp_path, hydro="1 PotMod - potential-flow model\n" + _HYDRO_SS
        )
        with pytest.raises(DeckValidationError, match=r"'\.ss' coefficient file"):
            validate_deck(fst)

    def test_radiation_fit_accepts_a_missing_ss_with_wamit_data(
        self, tmp_path: Path
    ) -> None:
        """hydro_radiation_fit generates the .ss per case, so .1 data suffice."""
        (tmp_path / "hydro").mkdir()
        (tmp_path / "hydro" / "marin.ssexctn").write_text("ssx", encoding="utf-8")
        fst = _write_deck(tmp_path, hydro="1 PotMod - potential flow\n" + _HYDRO_SS)
        fitting = LinearizationSettings(
            hydro_radiation_mode=2, hydro_radiation_fit=True
        )
        with pytest.raises(DeckValidationError, match=r"WAMIT '\.1' radiation file"):
            validate_deck(fst, fitting)
        (tmp_path / "hydro" / "marin.1").write_text("wamit", encoding="utf-8")
        validate_deck(fst, fitting)
        with pytest.raises(DeckValidationError, match=r"'\.ss' coefficient file"):
            validate_deck(fst)

    def test_disabled_hydro_module_is_skipped(self, tmp_path: Path) -> None:
        """A non-linearizable HydroDyn file is inert when CompHydro=0."""
        (tmp_path / "HD.dat").write_text(_HYDRO_CONV, encoding="utf-8")
        fst = tmp_path / "Model.fst"
        fst.write_text(
            "0 CompHydro - compute hydrodynamics {0=off}\n"
            '"HD.dat" HydroFile - HydroDyn file\n',
            encoding="utf-8",
        )
        validate_deck(fst)  # CompHydro=0 → the convolution-radiation file is ignored

    def test_second_order_switch_rejected(self, tmp_path: Path) -> None:
        """An active 2nd-order force switch is rejected under default settings."""
        _make_ss_files(tmp_path)
        fst = _write_deck(tmp_path, hydro=_HYDRO_SS_SECOND_ORDER)
        with pytest.raises(DeckValidationError, match="2nd-order force switches"):
            validate_deck(fst)

    def test_second_order_cleared_by_settings(self, tmp_path: Path) -> None:
        """disable_second_order_hydro clears the 2nd-order check."""
        _make_ss_files(tmp_path)
        fst = _write_deck(tmp_path, hydro=_HYDRO_SS_SECOND_ORDER)
        validate_deck(
            fst,
            LinearizationSettings(
                rated_wind_speed=11.4, disable_second_order_hydro=True
            ),
        )


class TestAeroDyn:
    """Unsteady-aero and dynamic-wake checks."""

    def test_bad_modes_rejected(self, tmp_path: Path) -> None:
        """Non-linearizable UA_Mod/DBEMT_Mod are rejected."""
        fst = _write_deck(tmp_path, aero=_AERO_BAD)
        with pytest.raises(DeckValidationError) as excinfo:
            validate_deck(fst)
        message = str(excinfo.value)
        assert "UA_Mod=3" in message
        assert "DBEMT_Mod=2" in message

    def test_modes_overridden_by_settings(self, tmp_path: Path) -> None:
        """Settings overriding the aero modes clear the check."""
        fst = _write_deck(tmp_path, aero=_AERO_BAD)
        validate_deck(
            fst,
            LinearizationSettings(
                rated_wind_speed=11.4, unsteady_aero_mode=0, dynamic_wake_mode=-1
            ),
        )

    def test_good_modes_pass(self, tmp_path: Path) -> None:
        """Linearizable aero modes pass as-is."""
        fst = _write_deck(tmp_path, aero=_AERO_GOOD)
        validate_deck(fst)


# Legacy-format AeroDyn: airfoil model AFAeroMod (2 = unsteady B-L), dynamic BEMT wake
# (WakeMod = 2), and a DBEMT_Mod that only applies when WakeMod = 2.
_AERO_LEGACY = """\
2 WakeMod - wake/induction model {0=none; 1=BEMT; 2=DBEMT; 3=OLAF}
2 AFAeroMod - airfoil aerodynamics model {1=steady; 2=Beddoes-Leishman unsteady}
False FrozenWake - assume frozen wake during linearization (flag)
1 DBEMT_Mod - dynamic BEMT model {used only when WakeMod=2}
3 UAMod - unsteady aero model switch {used only when AFAeroMod=2}
"""


class TestLegacyAeroDyn:
    """Legacy AeroDyn keywords (AFAeroMod/UAMod/WakeMod) are checked, not ignored."""

    def test_legacy_unsteady_deck_is_rejected(self, tmp_path: Path) -> None:
        """Unsteady airfoils and dynamic BEMT would fail late inside OpenFAST."""
        fst = _write_deck(tmp_path, aero=_AERO_LEGACY)
        with pytest.raises(DeckValidationError) as excinfo:
            validate_deck(fst)
        message = str(excinfo.value)
        assert "AFAeroMod=2" in message
        assert "WakeMod=2" in message

    def test_offshore_settings_make_legacy_deck_pass(self, tmp_path: Path) -> None:
        """Quasi-steady aero and a frozen wake map onto the legacy keywords."""
        fst = _write_deck(tmp_path, aero=_AERO_LEGACY)
        validate_deck(fst, LinearizationSettings.offshore(rated_wind_speed=11.4))

    def test_legacy_free_vortex_wake_is_rejected(self, tmp_path: Path) -> None:
        """A free vortex wake (WakeMod=3) has no linearizable legacy replacement."""
        fst = _write_deck(tmp_path, aero=_AERO_LEGACY.replace("2 WakeMod", "3 WakeMod"))
        with pytest.raises(DeckValidationError, match="WakeMod=3"):
            validate_deck(fst, LinearizationSettings.offshore(rated_wind_speed=11.4))

    @pytest.mark.parametrize(
        ("overrides", "fragment"),
        [
            ({"unsteady_aero_mode": 5}, "unsteady_aero_mode=5"),
            ({"unsteady_aero_mode": 0, "dynamic_wake_mode": 3}, "dynamic_wake_mode=3"),
        ],
    )
    def test_requests_the_legacy_format_cannot_express_are_rejected(
        self, tmp_path: Path, overrides: dict[str, Any], fragment: str
    ) -> None:
        """A modern-only model request is reported rather than silently dropped."""
        fst = _write_deck(tmp_path, aero=_AERO_LEGACY)
        settings = LinearizationSettings(**overrides)
        with pytest.raises(DeckValidationError, match=fragment):
            validate_deck(fst, settings)

    def test_linearizable_legacy_deck_passes_as_is(self, tmp_path: Path) -> None:
        """Steady airfoils with plain BEMT pass without any settings override."""
        fst = _write_deck(
            tmp_path,
            aero=_AERO_LEGACY.replace("2 WakeMod", "1 WakeMod").replace(
                "2 AFAeroMod", "1 AFAeroMod"
            ),
        )
        validate_deck(fst)


class TestComposite:
    """Whole-deck behaviour."""

    def test_onshore_deck_passes(self, tmp_path: Path) -> None:
        """A deck with no offshore/control sub-files is trivially fine."""
        fst = _write_deck(tmp_path)
        validate_deck(fst)

    def test_offshore_settings_make_raw_fowt_pass(self, tmp_path: Path) -> None:
        """A raw FOWT template passes under LinearizationSettings.offshore()."""
        _make_ss_files(tmp_path)
        fst = _write_deck(
            tmp_path,
            servo=_SERVO_DLL,
            seastate=_SEASTATE_WAVES,
            hydro=_HYDRO_SS_SECOND_ORDER,
            aero=_AERO_BAD,
        )
        # offshore() neutralises the DLL, waves, 2nd-order forces, and the aero modes.
        validate_deck(fst, LinearizationSettings.offshore(rated_wind_speed=11.4))

    def test_all_problems_reported_together(self, tmp_path: Path) -> None:
        """The raw FOWT template under defaults reports every problem at once."""
        fst = _write_deck(
            tmp_path, servo=_SERVO_DLL, seastate=_SEASTATE_WAVES, hydro=_HYDRO_CONV
        )
        with pytest.raises(DeckValidationError) as excinfo:
            validate_deck(fst)
        message = str(excinfo.value)
        assert "DLL controller" in message
        assert "waves enabled" in message
        assert "convolution radiation" in message


def _write_files(tmp_path: Path, primary: str, **files: str) -> Path:
    """Write ``Model.fst`` with ``primary`` and each named sub-file beside it."""
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    fst = tmp_path / "Model.fst"
    fst.write_text(primary, encoding="utf-8")
    return fst


def _problems(fst: Path, settings: LinearizationSettings | None = None) -> str:
    """Return the validation message for ``fst`` ('' when the deck passes)."""
    try:
        validate_deck(fst, settings)
    except DeckValidationError as exc:
        return str(exc)
    return ""


class TestModuleChoices:
    """Primary-deck module choices OpenFAST cannot linearize."""

    @pytest.mark.parametrize(
        ("switch", "value", "name"),
        [
            ("CompElast", 3, "Simplified-ElastoDyn"),
            ("CompAero", 1, "AeroDisk"),
            ("CompAero", 3, "ExtLoads"),
            ("CompMooring", 2, "FEAMooring"),
            ("CompMooring", 4, "OrcaFlex"),
            ("CompIce", 1, "IceFloe"),
            ("CompIce", 2, "IceDyn"),
            ("CompSoil", 1, "SoilDyn"),
        ],
    )
    def test_unlinearizable_module_is_rejected(
        self, tmp_path: Path, switch: str, value: int, name: str
    ) -> None:
        """Each module OpenFAST has no linearization for is named in the message."""
        fst = _write_files(tmp_path, f"{value} {switch} - module switch\n")
        with pytest.raises(
            DeckValidationError, match=f"{switch}={value} selects {name}"
        ):
            validate_deck(fst)

    @pytest.mark.parametrize(
        ("switch", "value"),
        [
            ("CompElast", 1),
            ("CompElast", 2),
            ("CompAero", 2),
            ("CompMooring", 1),
            ("CompMooring", 3),
            ("CompSub", 2),
            ("CompIce", 0),
            ("CompSoil", 0),
        ],
    )
    def test_linearizable_module_passes(
        self, tmp_path: Path, switch: str, value: int
    ) -> None:
        """ElastoDyn, BeamDyn, AeroDyn, MAP, MoorDyn, and ExtPtfm are not flagged."""
        fst = _write_files(tmp_path, f"{value} {switch} - module switch\n")
        validate_deck(fst)


_CALC_STEADY_PRIMARY = """\
{dt} DT - time step (s)
True Linearize - linearization analysis (flag)
True CalcSteady - steady-state periodic operating point (flag)
36 NLinTimes - number of linearizations
"ED.dat" EDFile - ElastoDyn file
"""


class TestCalcSteadyStep:
    """The CalcSteady azimuth spacing versus the rotor travel per time step."""

    def test_time_step_too_long_for_azimuth_spacing(self, tmp_path: Path) -> None:
        """12.1 rpm with DT=0.5 s turns 36 deg per step, past the 10 deg spacing."""
        fst = _write_files(
            tmp_path,
            _CALC_STEADY_PRIMARY.format(dt=0.5),
            **{"ED.dat": "12.1 RotSpeed - initial rotor speed (rpm)\n"},
        )
        with pytest.raises(DeckValidationError, match=r"rotor turns 36\.3 deg"):
            validate_deck(fst)

    def test_regular_time_step_passes(self, tmp_path: Path) -> None:
        """A realistic DT is far inside the azimuth spacing."""
        fst = _write_files(
            tmp_path,
            _CALC_STEADY_PRIMARY.format(dt=0.0125),
            **{"ED.dat": "12.1 RotSpeed - initial rotor speed (rpm)\n"},
        )
        validate_deck(fst)

    def test_not_checked_without_calc_steady(self, tmp_path: Path) -> None:
        """A deck that does not request CalcSteady is not judged on its step."""
        primary = _CALC_STEADY_PRIMARY.format(dt=0.5).replace(
            "True CalcSteady", "False CalcSteady"
        )
        fst = _write_files(
            tmp_path, primary, **{"ED.dat": "12.1 RotSpeed - rotor speed (rpm)\n"}
        )
        validate_deck(fst)


class TestInflowWind:
    """Lidar and upflow restrictions."""

    _PRIMARY = '1 CompInflow - inflow switch\n"IW.dat" InflowFile - InflowWind file\n'

    def test_lidar_and_upflow_rejected(self, tmp_path: Path) -> None:
        """An active lidar and a non-zero upflow angle are both reported."""
        fst = _write_files(
            tmp_path,
            self._PRIMARY,
            **{
                "IW.dat": "1 WindType - wind type\n8 VFlowAng - upflow angle (deg)\n"
                "1 SensorType - lidar switch\n"
            },
        )
        message = _problems(fst)
        assert "SensorType=1" in message
        assert "VFlowAng=8" in message

    def test_clean_inflow_passes(self, tmp_path: Path) -> None:
        """No lidar and zero upflow pass."""
        fst = _write_files(
            tmp_path,
            self._PRIMARY,
            **{"IW.dat": "0 VFlowAng - upflow angle (deg)\n0 SensorType - lidar\n"},
        )
        validate_deck(fst)

    def test_disabled_inflow_is_skipped(self, tmp_path: Path) -> None:
        """CompInflow=0 leaves the InflowWind file inert."""
        fst = _write_files(
            tmp_path,
            self._PRIMARY.replace("1 CompInflow", "0 CompInflow"),
            **{"IW.dat": "1 SensorType - lidar switch\n"},
        )
        validate_deck(fst)


class TestServoDynRestrictions:
    """ServoDyn settings beyond the DLL controller."""

    _PRIMARY = '"SD.dat" ServoFile - ServoDyn file\n'

    @pytest.mark.parametrize(
        ("line", "fragment"),
        [
            ("3 PCMode - pitch control (user routine)\n", "PCMode=3"),
            ("3 VSContrl - torque control (user routine)\n", "VSContrl=3"),
            ("1 HSSBrMode - HSS brake\n", "HSSBrMode=1"),
            ("4 YCMode - yaw control (Simulink)\n", "YCMode=4"),
            ("5 AfCmode - flap control (controller library)\n", "AfCmode=5"),
            ("5 CCmode - cable control (controller library)\n", "CCmode=5"),
            ("3 GenModel - generator model (user)\n", "GenModel=3"),
            ("False GenTiStr - timed generator start (flag)\n", "GenTiStr=False"),
            ("False GenTiStp - timed generator stop (flag)\n", "GenTiStp=False"),
        ],
    )
    def test_non_linearizable_setting_rejected(
        self, tmp_path: Path, line: str, fragment: str
    ) -> None:
        """Each ServoDyn switch OpenFAST rejects is reported, and the trim clears it."""
        fst = _write_files(tmp_path, self._PRIMARY, **{"SD.dat": line})
        with pytest.raises(DeckValidationError, match=fragment):
            validate_deck(fst)
        validate_deck(
            fst, LinearizationSettings(rated_wind_speed=11.4, trim_controller=True)
        )

    def test_linearizable_servo_passes(self, tmp_path: Path) -> None:
        """The settings OpenFAST accepts pass without the trim."""
        fst = _write_files(
            tmp_path,
            self._PRIMARY,
            **{
                "SD.dat": _SERVO_SIMPLE + "2 GenModel - Thevenin\n"
                "True GenTiStr - timed start\nTrue GenTiStp - timed stop\n"
            },
        )
        validate_deck(fst)

    def test_fortran_logical_spellings_are_true(self, tmp_path: Path) -> None:
        """``.true.`` and ``T`` are true flags, not reported as False."""
        fst = _write_files(
            tmp_path,
            self._PRIMARY,
            **{
                "SD.dat": _SERVO_SIMPLE + ".true. GenTiStr - timed start\n"
                "T GenTiStp - timed stop\n"
            },
        )
        validate_deck(fst)


class TestSeaStateRestrictions:
    """Directional spreading, constrained waves, and still-water consistency."""

    _PRIMARY = '1 CompSeaSt - SeaState switch\n"SeaSt.dat" SeaStFile - SeaState file\n'

    def test_spreading_and_constrained_wave_rejected(self, tmp_path: Path) -> None:
        """WaveDirMod/ConstWaveMod are rejected even in still water."""
        fst = _write_files(
            tmp_path,
            self._PRIMARY,
            **{
                "SeaSt.dat": _SEASTATE_STILL + "1 WaveDirMod - spreading\n"
                "2 ConstWaveMod - constrained wave\n"
            },
        )
        with pytest.raises(DeckValidationError, match="WaveDirMod=1, ConstWaveMod=2"):
            validate_deck(fst)
        # wave_mod=0 resets both (see configure_linearization).
        validate_deck(fst, LinearizationSettings(wave_mod=0))

    def test_still_water_inconsistencies_rejected(self, tmp_path: Path) -> None:
        """Still water with stretching 2, diffraction, or a DEFAULT current fails."""
        fst = _write_files(
            tmp_path,
            self._PRIMARY,
            **{
                "SeaSt.dat": _SEASTATE_STILL + "2 WaveStMod - stretching\n"
                "8.0 MCFD - MacCamy-Fuchs diameter (m)\n"
                "1 CurrMod - current model\n"
                '"DEFAULT" CurrSSDir - sub-surface current heading\n'
            },
        )
        message = _problems(fst)
        assert "WaveStMod=2" in message
        assert "MCFD=8" in message
        assert "CurrSSDir=DEFAULT" in message
        validate_deck(fst, LinearizationSettings(wave_mod=0))

    def test_waves_with_stretching_are_not_still_water_problems(
        self, tmp_path: Path
    ) -> None:
        """With waves on, only the waves are reported (wave_mod=0 fixes the rest)."""
        fst = _write_files(
            tmp_path,
            self._PRIMARY,
            **{"SeaSt.dat": _SEASTATE_WAVES + "2 WaveStMod - stretching\n"},
        )
        message = _problems(fst)
        assert "waves enabled" in message
        assert "WaveStMod" not in message

    def test_disabled_seastate_is_skipped(self, tmp_path: Path) -> None:
        """CompSeaSt=0 leaves the SeaState file inert."""
        fst = _write_files(
            tmp_path,
            self._PRIMARY.replace("1 CompSeaSt", "0 CompSeaSt"),
            **{"SeaSt.dat": _SEASTATE_WAVES + "1 WaveDirMod - spreading\n"},
        )
        validate_deck(fst)


class TestHydroDynRestrictions:
    """Potential-flow model, DFT excitation, add-DOFs, and radiation step."""

    _PRIMARY = (
        "0.0125 DT - time step (s)\n"
        '1 CompHydro - HydroDyn switch\n"HD.dat" HydroFile - HydroDyn file\n'
    )

    def test_fit_potential_flow_rejected(self, tmp_path: Path) -> None:
        """PotMod=2 (FIT) is not linearizable."""
        fst = _write_files(
            tmp_path, self._PRIMARY, **{"HD.dat": "2 PotMod - potential flow\n"}
        )
        with pytest.raises(DeckValidationError, match="PotMod=2"):
            validate_deck(fst)

    def test_dft_excitation_rejected(self, tmp_path: Path) -> None:
        """ExctnMod=1 (DFT) is rejected; the state-space override clears it."""
        _make_ss_files(tmp_path)
        fst = _write_files(
            tmp_path,
            self._PRIMARY,
            **{
                "HD.dat": "1 PotMod - WAMIT\n0 RdtnMod - radiation\n"
                "1 ExctnMod - DFT excitation\n"
                '"hydro/marin" PotFile - potential-flow root\n'
            },
        )
        with pytest.raises(DeckValidationError, match="DFT wave excitation"):
            validate_deck(fst)
        validate_deck(fst, LinearizationSettings(hydro_excitation_mode=2))

    def test_second_order_ignored_without_wamit(self, tmp_path: Path) -> None:
        """HydroDyn switches 2nd-order forces off when PotMod is not 1."""
        fst = _write_files(
            tmp_path,
            self._PRIMARY,
            **{"HD.dat": "0 PotMod - none\n12 DiffQTF - full QTF\n"},
        )
        validate_deck(fst)

    def test_add_dofs_conflict_with_state_space(self, tmp_path: Path) -> None:
        """NAddDOF>0 with state-space radiation/excitation is reported."""
        _make_ss_files(tmp_path)
        fst = _write_files(
            tmp_path,
            self._PRIMARY,
            **{
                "HD.dat": "1 PotMod - WAMIT\n0 RdtnMod - radiation\n"
                "0 ExctnMod - excitation\n6 NAddDOF - additional body DOFs\n"
                '"hydro/marin" PotFile - potential-flow root\n'
            },
        )
        validate_deck(fst)  # no state-space model: fine
        message = _problems(
            fst,
            LinearizationSettings(hydro_radiation_mode=2, hydro_excitation_mode=2),
        )
        assert "NAddDOF>0" in message
        assert "RdtnMod=2, ExctnMod=2" in message

    def test_radiation_step_mismatch(self, tmp_path: Path) -> None:
        """A numeric RdtnDT other than DT is rejected; a hydro override fixes it."""
        _make_ss_files(tmp_path)
        fst = _write_files(
            tmp_path,
            self._PRIMARY,
            **{
                "HD.dat": "1 PotMod - WAMIT\n2 RdtnMod - radiation\n"
                "0 ExctnMod - excitation\n0.025 RdtnDT - radiation step (s)\n"
                '"hydro/marin" PotFile - potential-flow root\n'
            },
        )
        with pytest.raises(DeckValidationError, match=r"RdtnDT=0\.025"):
            validate_deck(fst)
        validate_deck(fst, LinearizationSettings(hydro_radiation_mode=2))


class TestAeroDynRestrictions:
    """Wake model and nacelle drag."""

    _PRIMARY = '2 CompAero - AeroDyn\n"AD.dat" AeroFile - AeroDyn file\n'

    def test_olaf_wake_rejected(self, tmp_path: Path) -> None:
        """Wake_Mod=3 (OLAF) cannot be linearized."""
        fst = _write_files(
            tmp_path, self._PRIMARY, **{"AD.dat": _AERO_GOOD + "3 Wake_Mod - OLAF\n"}
        )
        with pytest.raises(DeckValidationError, match="Wake_Mod=3"):
            validate_deck(fst)

    def test_nacelle_drag_rejected(self, tmp_path: Path) -> None:
        """NacelleDrag=True cannot be linearized."""
        fst = _write_files(
            tmp_path,
            self._PRIMARY,
            **{"AD.dat": _AERO_GOOD + "True NacelleDrag - nacelle drag (flag)\n"},
        )
        with pytest.raises(DeckValidationError, match="NacelleDrag"):
            validate_deck(fst)

    def test_bemt_without_drag_passes(self, tmp_path: Path) -> None:
        """BEMT with nacelle drag off passes."""
        fst = _write_files(
            tmp_path,
            self._PRIMARY,
            **{"AD.dat": _AERO_GOOD + "1 Wake_Mod - BEMT\nFalse NacelleDrag - off\n"},
        )
        validate_deck(fst)


class TestBeamDyn:
    """The BeamDyn pitch actuator."""

    _PRIMARY = (
        "{elast} CompElast - structural module\n"
        '"BD.dat" BDBldFile(1) - blade 1\n'
        '"BD.dat" BDBldFile(2) - blade 2\n'
        '"BD.dat" BDBldFile(3) - blade 3\n'
    )

    def test_pitch_actuator_rejected_once(self, tmp_path: Path) -> None:
        """A shared blade file with the actuator on is reported once."""
        fst = _write_files(
            tmp_path,
            self._PRIMARY.format(elast=2),
            **{"BD.dat": "True UsePitchAct - pitch actuator (flag)\n"},
        )
        message = _problems(fst)
        assert message.count("UsePitchAct=True") == 1

    def test_elastodyn_blades_ignore_beamdyn_files(self, tmp_path: Path) -> None:
        """With CompElast=1 the BeamDyn files are inert."""
        fst = _write_files(
            tmp_path,
            self._PRIMARY.format(elast=1),
            **{"BD.dat": "True UsePitchAct - pitch actuator (flag)\n"},
        )
        validate_deck(fst)


_MOORDYN = """\
--------------------- MoorDyn Input File ------------------------------------
Mooring system for a test deck
----------------------- LINE TYPES ------------------------------------------
TypeName  Diam  Mass/m  EA          BA/-zeta  EI  Cd   Ca   CdAx  CaAx  Cl
(name)    (m)   (kg/m)  (N)         (N-s/-)   (-) (-)  (-)  (-)   (-)   (-)
visco     0.09  77.7    3.8E8|1E9   -0.8      0   1.6  1.0  0.1   0.0   0.0
viv       0.09  77.7    3.8E8       -0.8      0   1.6  1.0  0.1   0.0   1.0
plain     0.09  77.7    3.8E8       -0.8      0   1.6  1.0  0.1   0.0
---------------------- POINTS --------------------------------
ID  Attachment  X  Y  Z  M  V  CdA  CA
(-) (-) (m) (m) (m) (kg) (m^3) (m^2) (-)
1   Fixed  0  0  -320  0  0  0  0
2   Vessel 0  0  -70   0  0  0  0
---------------------- LINES --------------------------------------
ID  LineType  AttachA  AttachB  UnstrLen  NumSegs  Outputs
(-) (-)       (-)      (-)      (m)       (-)      (-)
1   {line1}   1        2        835.35    20       -
2   plain     1        2        835.35    20       -  # a trailing comment
---------------------- OPTIONS -----------------------------------------
"""


class TestMoorDyn:
    """MoorDyn VIV and viscoelastic line models."""

    _PRIMARY = (
        '{mooring} CompMooring - mooring module\n"MD.dat" MooringFile - MoorDyn\n'
    )

    @pytest.mark.parametrize(
        ("line_type", "fragment"), [("visco", "viscoelastic"), ("viv", "VIV")]
    )
    def test_used_unsupported_line_type_rejected(
        self, tmp_path: Path, line_type: str, fragment: str
    ) -> None:
        """A line whose type enables VIV or viscoelasticity is reported."""
        fst = _write_files(
            tmp_path,
            self._PRIMARY.format(mooring=3),
            **{"MD.dat": _MOORDYN.format(line1=line_type)},
        )
        with pytest.raises(
            DeckValidationError, match=f"line 1: type {line_type} {fragment}"
        ):
            validate_deck(fst)

    def test_unused_unsupported_types_pass(self, tmp_path: Path) -> None:
        """Defined but unused VIV/viscoelastic types do not matter."""
        fst = _write_files(
            tmp_path,
            self._PRIMARY.format(mooring=3),
            **{"MD.dat": _MOORDYN.format(line1="plain")},
        )
        validate_deck(fst)

    def test_other_mooring_module_ignores_moordyn_file(self, tmp_path: Path) -> None:
        """With MAP (CompMooring=1) the MoorDyn-format file is not parsed."""
        fst = _write_files(
            tmp_path,
            self._PRIMARY.format(mooring=1),
            **{"MD.dat": _MOORDYN.format(line1="visco")},
        )
        validate_deck(fst)


class TestCheckDeckPathLengths:
    """Tests for the Windows path-length pre-flight check."""

    @staticmethod
    def _deck(tmp_path: Path) -> tuple[Path, str]:
        """Write a deck with a shared sub-file; return it and the longest open path."""
        case = tmp_path / "Case"
        case.mkdir()
        shared = tmp_path / "Shared"
        shared.mkdir()
        fst = case / "Model.fst"
        fst.write_text('"../Shared/InflowWind.dat" InflowFile - wind\n', "utf-8")
        (shared / "InflowWind.dat").write_text(
            '"Wind/turbulent_wind.bts" FileName - w\n', "utf-8"
        )
        (shared / "Wind").mkdir()
        (shared / "Wind" / "turbulent_wind.bts").write_text("wind\n", "utf-8")
        longest = f"{case}{os.sep}.{os.sep}../Shared/Wind/turbulent_wind.bts"
        return fst, longest

    def test_short_paths_pass(self, tmp_path: Path) -> None:
        """A deck whose longest open path fits passes."""
        fst, longest = self._deck(tmp_path)
        check_deck_path_lengths(fst, max_length=len(longest))

    def test_overlong_path_is_named_with_its_length(self, tmp_path: Path) -> None:
        """One character over the limit fails, naming the path and its length."""
        fst, longest = self._deck(tmp_path)
        with pytest.raises(DeckPathLengthError) as excinfo:
            check_deck_path_lengths(fst, max_length=len(longest) - 1)
        message = str(excinfo.value)
        assert f"({len(longest)} characters): {longest}" in message
        assert "shorter output_dir" in message
        assert "1 file(s)" in message

    def test_error_is_a_deck_validation_error(self, tmp_path: Path) -> None:
        """Callers catching DeckValidationError also catch the length error."""
        fst, _ = self._deck(tmp_path)
        with pytest.raises(DeckValidationError):
            check_deck_path_lengths(fst, max_length=10)

    def test_unnormalized_length_is_what_counts(self, tmp_path: Path) -> None:
        """The ``..`` segments count although the normalized path is shorter."""
        fst, longest = self._deck(tmp_path)
        normalized = os.path.normpath(longest)
        assert len(normalized) < len(longest)
        with pytest.raises(DeckPathLengthError):
            check_deck_path_lengths(fst, max_length=len(normalized))

    def test_missing_deck_raises(self, tmp_path: Path) -> None:
        """A deck that does not exist is reported as such."""
        with pytest.raises(FileNotFoundError):
            check_deck_path_lengths(tmp_path / "absent.fst")

    def test_unwalkable_deck_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A reference walk that cannot complete does not fail the check."""
        from vane.run import preflight

        def broken(*_args: object) -> object:
            msg = "cannot read referenced input"
            raise DeckReferenceError(msg)

        fst, _ = self._deck(tmp_path)
        monkeypatch.setattr(preflight, "deck_file_closure", broken)
        check_deck_path_lengths(fst, max_length=10)

    def test_default_limit(self) -> None:
        """The default is the limit measured on the Windows OpenFAST build."""
        assert OPENFAST_MAX_PATH_LENGTH == 258

    def test_regression_deck_in_place_passes(self, rtest_dir: Path) -> None:
        """The r-test land deck's longest spelling is its shared InflowWind file."""
        fst = (
            rtest_dir
            / "glue-codes"
            / "openfast"
            / "5MW_Land_DLL_WTurb"
            / "5MW_Land_DLL_WTurb.fst"
        )
        if not fst.is_file():
            pytest.skip(f"r-test case not found: {fst}")
        check_deck_path_lengths(fst)
        closure = deck_file_closure(fst, Path(fst.anchor))
        longest = max((text for _, text in closure.open_paths), key=len)
        assert longest.endswith(
            "../5MW_Baseline/NRELOffshrBsline5MW_InflowWind_12mps.dat"
        )
        assert longest.startswith(f"{fst.parent}{os.sep}.{os.sep}")


class TestUnsteadyAeroAtTheTip:
    """Unsteady aerodynamics must end inboard of the tip node under tip loss."""

    _PRIMARY = '2 CompAero - AeroDyn\n"AD.dat" AeroFile - AeroDyn file\n'
    # The IEA 22-MW rotor-only sweep setting whose tip-node UA state went NaN.
    _AERO = (
        "1 Wake_Mod - BEMT\n"
        "4 UA_Mod - unsteady aerodynamics {HGM}\n"
        "0 DBEMT_Mod - no dynamic wake\n"
        "True TipLoss - Prandtl tip loss (flag)\n"
        "True HubLoss - Prandtl hub loss (flag)\n"
        "0.05 UAStartRad - starting radius for UA\n"
        "1 UAEndRad - ending radius for UA\n"
    )

    def _deck(self, tmp_path: Path, aero: str) -> Path:
        return _write_files(tmp_path, self._PRIMARY, **{"AD.dat": aero})

    def test_ua_to_the_tip_with_tip_loss_is_rejected(self, tmp_path: Path) -> None:
        """The failing campaign deck is caught with the remedy."""
        message = _problems(self._deck(tmp_path, self._AERO))
        assert "UAEndRad=1" in message
        assert "TipLoss=True" in message
        assert "set UAEndRad below 1" in message
        assert "0.99" in message

    def test_tip_loss_as_a_fortran_logical(self, tmp_path: Path) -> None:
        """``.TRUE.`` tip loss is caught like ``True``."""
        aero = self._AERO.replace("True TipLoss", ".TRUE. TipLoss")
        assert "TipLoss=True" in _problems(self._deck(tmp_path, aero))

    def test_missing_uaendrad_reads_as_one(self, tmp_path: Path) -> None:
        """AeroDyn defaults a missing ``UAEndRad`` to 1, so the check still applies."""
        aero = self._AERO.replace("1 UAEndRad - ending radius for UA\n", "")
        assert "UAEndRad=1" in _problems(self._deck(tmp_path, aero))

    def test_inboard_end_radius_passes(self, tmp_path: Path) -> None:
        """``UAEndRad < 1`` switches the tip node off (fraction 1 > UAEndRad)."""
        aero = self._AERO.replace("1 UAEndRad", "0.99 UAEndRad")
        assert _problems(self._deck(tmp_path, aero)) == ""

    def test_setting_clears_the_check(self, tmp_path: Path) -> None:
        """``ua_end_radius`` is the effective value once the settings apply."""
        fst = self._deck(tmp_path, self._AERO)
        assert _problems(fst, LinearizationSettings(ua_end_radius=0.99)) == ""
        assert "UAEndRad=1" in _problems(fst, LinearizationSettings(ua_end_radius=1.0))

    def test_without_tip_loss_passes(self, tmp_path: Path) -> None:
        """Without tip loss the tip node's inductions are not fixed."""
        aero = self._AERO.replace("True TipLoss", "False TipLoss")
        assert _problems(self._deck(tmp_path, aero)) == ""

    def test_quasi_steady_aero_passes(self, tmp_path: Path) -> None:
        """No unsteady aerodynamics, in the deck or forced by the settings, no check."""
        fst = self._deck(tmp_path, self._AERO.replace("4 UA_Mod", "0 UA_Mod"))
        assert _problems(fst) == ""
        fst = self._deck(tmp_path, self._AERO)
        assert _problems(fst, LinearizationSettings(unsteady_aero_mode=0)) == ""

    def test_settings_turning_ua_on_are_checked(self, tmp_path: Path) -> None:
        """A quasi-steady deck linearized with ``unsteady_aero_mode=5`` is judged."""
        fst = self._deck(tmp_path, self._AERO.replace("4 UA_Mod", "0 UA_Mod"))
        message = _problems(fst, LinearizationSettings(unsteady_aero_mode=5))
        assert "effective UA_Mod=5" in message

    def test_no_induction_passes(self, tmp_path: Path) -> None:
        """``Wake_Mod = 0`` computes no induction, so tip loss is unused."""
        aero = self._AERO.replace("1 Wake_Mod", "0 Wake_Mod")
        assert _problems(self._deck(tmp_path, aero)) == ""

    def test_legacy_deck_is_not_checked_for_the_tip(self, tmp_path: Path) -> None:
        """A legacy deck linearizes only with steady airfoils, so no tip state."""
        aero = (
            "1 WakeMod - BEMT\n"
            "1 AFAeroMod - steady airfoils\n"
            "True TipLoss - Prandtl tip loss (flag)\n"
            "1 UAEndRad - ending radius for UA\n"
        )
        assert _problems(self._deck(tmp_path, aero)) == ""
