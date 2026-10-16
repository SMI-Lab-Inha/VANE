# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.run.trim_verification — trimmed operating point versus curve."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import build_lin_text
from vane.io.lin_reader import LinFile, OperatingPointTable
from vane.run import (
    LinearizationSettings,
    OperatingPoint,
    OperatingSweepResult,
    QuantityCheck,
    RotorPerformanceCurve,
    SweepCase,
    TrimTolerances,
    TurbineConstants,
    aerodynamic_power,
    find_lin_files,
    rotor_performance_from_csv,
    trimmed_operating_point,
    verify_operating_points,
    verify_sweep_operating_points,
)
from vane.run.runner import OpenFastResult

RHO = 1.225
GB_RATIO = 97.0
GB_EFF = 0.944
TIP_RAD = 63.0
PRECONE_DEG = -2.5
RADIUS = TIP_RAD * math.cos(math.radians(PRECONE_DEG))

CURVE_WIND = (4.0, 8.0, 12.0, 16.0, 20.0)
CURVE_RPM = (7.0, 9.5, 12.1, 12.1, 12.1)
CURVE_PITCH = (0.0, 0.0, 4.0, 10.0, 15.0)
CURVE_CP = (0.40, 0.45, 0.35, 0.20, 0.10)
CURVE_CT = (0.80, 0.75, 0.50, 0.30, 0.15)

CONSTANTS = TurbineConstants(
    gearbox_ratio=GB_RATIO,
    gearbox_efficiency=GB_EFF,
    rotor_radius=RADIUS,
    air_density=RHO,
)


def _curve() -> RotorPerformanceCurve:
    return RotorPerformanceCurve(
        wind_speed=CURVE_WIND,
        rotor_speed_rpm=CURVE_RPM,
        blade_pitch_deg=CURVE_PITCH,
        cp=CURVE_CP,
        ct=CURVE_CT,
    )


def _table(values: list[float], descriptions: list[str]) -> OperatingPointTable:
    count = len(values)
    return OperatingPointTable(
        values=np.asarray(values, dtype=np.float64),
        rotating_frame=np.zeros(count, dtype=np.bool_),
        derivative_order=np.zeros(count, dtype=np.int_),
        descriptions=descriptions,
    )


def _generator_torque(wind: float, rpm: float, cp: float) -> float:
    """Return the HSS torque that yields ``cp`` through the lossy torque balance."""
    power = cp * 0.5 * RHO * math.pi * RADIUS**2 * wind**3
    return power * GB_EFF / (GB_RATIO * rpm * math.pi / 30.0)


def _thrust(wind: float, ct: float) -> float:
    return ct * 0.5 * RHO * math.pi * RADIUS**2 * wind**2


def _inputs(pitch_deg: list[float], torque: float) -> tuple[list[float], list[str]]:
    values = [12.0]
    names = [
        "IfW Extended input: horizontal wind speed (steady/uniform wind) (hub), m/s"
    ]
    for blade, pitch in enumerate(pitch_deg, start=1):
        values.append(math.radians(pitch))
        names.append(f"ED Blade {blade} pitch command, rad")
    values += [0.0, torque, math.radians(float(np.mean(pitch_deg)))]
    names += [
        "ED Blade 1 pitch moment, Nm",
        "ED Generator torque, Nm",
        "ED Extended input: collective blade-pitch command, rad",
    ]
    return values, names


def _lin(
    *,
    wind: float,
    rpm: float,
    pitch_deg: float | list[float],
    torque: float,
    azimuth: float = 0.0,
    inputs: bool = True,
    outputs: dict[str, float] | None = None,
) -> LinFile:
    pitches = pitch_deg if isinstance(pitch_deg, list) else [pitch_deg] * 3
    u_values, u_names = _inputs(pitches, torque) if inputs else ([], [])
    y_values = list((outputs or {}).values())
    y_names = list((outputs or {}).keys())
    return LinFile(
        path=Path(f"synthetic.{azimuth}.lin"),
        sim_time=0.0,
        rotor_speed=rpm * math.pi / 30.0,
        azimuth=azimuth,
        wind_speed=wind,
        jacobians_included=False,
        x=_table([], []),
        xdot=_table([], []),
        u=_table(u_values, u_names),
        y=_table(y_values, y_names),
        a=None,
        b=None,
        c=None,
        d=None,
        jacobians={},
    )


def _consistent_set(
    index: int, *, n_azimuth: int = 4, cp_scale: float = 1.0
) -> list[LinFile]:
    """Return azimuth files trimmed exactly onto curve sample ``index``."""
    wind, rpm, pitch = CURVE_WIND[index], CURVE_RPM[index], CURVE_PITCH[index]
    torque = _generator_torque(wind, rpm, CURVE_CP[index] * cp_scale)
    outputs = {"AD RtAeroFxh, (N)": _thrust(wind, CURVE_CT[index])}
    return [
        _lin(
            wind=wind,
            rpm=rpm,
            pitch_deg=pitch,
            torque=torque,
            azimuth=2.0 * math.pi * k / n_azimuth,
            outputs=outputs,
        )
        for k in range(n_azimuth)
    ]


def _write_deck(
    root: Path,
    *,
    ad_air_density: str = "1.225",
    fst_air_density: str | None = "1.0",
    simplified: bool = False,
) -> Path:
    """Write a minimal deck (fst + structural + AeroDyn) and return the fst path."""
    root.mkdir(parents=True, exist_ok=True)
    if simplified:
        structure = [
            "------- SIMPLIFIED ELASTODYN INPUT FILE -------",
            f"{TIP_RAD}   TipRad      - tip radius (m)",
            f"{PRECONE_DEG}   PreCone     - cone (deg)",
            f"{GB_RATIO}   GBoxRatio   - Gearbox ratio (-)",
        ]
    else:
        structure = [
            "------- ELASTODYN INPUT FILE -------",
            f"{TIP_RAD}   TipRad      - tip radius (m)",
            *(f"{PRECONE_DEG}   PreCone({i})  - cone (deg)" for i in (1, 2, 3)),
            f"{100.0 * GB_EFF}   GBoxEff     - Gearbox efficiency (%)",
            f"{GB_RATIO}   GBRatio     - Gearbox ratio (-)",
        ]
    (root / "structure.dat").write_text("\n".join(structure) + "\n", encoding="utf-8")
    (root / "aero.dat").write_text(
        f"------- AERODYN -------\n{ad_air_density}   AirDens   - Air density\n",
        encoding="utf-8",
    )
    fst = [
        "------- OpenFAST INPUT FILE -------",
        '"structure.dat"    EDFile    - structural file',
        '"aero.dat"    AeroFile    - aero file',
    ]
    if fst_air_density is not None:
        fst.append(f"{fst_air_density}   AirDens   - Air density (kg/m^3)")
    path = root / "turbine.fst"
    path.write_text("\n".join(fst) + "\n", encoding="utf-8")
    return path


class TestRotorPerformanceCurve:
    """Tests for RotorPerformanceCurve and rotor_performance_from_csv."""

    def test_interpolates_at_samples_and_nan_outside(self) -> None:
        """Curve samples are reproduced; outside the range everything is NaN."""
        curve = _curve()
        at = curve.at(12.0)
        assert at["rotor_speed_rpm"] == pytest.approx(12.1)
        assert at["cp"] == pytest.approx(0.35)
        assert at["ct"] == pytest.approx(0.50)
        outside = curve.at(25.0)
        assert all(math.isnan(v) for v in outside.values())
        assert math.isnan(curve.at(math.nan)["blade_pitch_deg"])

    def test_coefficients_need_two_finite_samples(self) -> None:
        """A coefficient with fewer than two finite samples is unavailable."""
        curve = RotorPerformanceCurve.from_arrays(
            [4.0, 8.0, 12.0],
            [7.0, 9.0, 12.0],
            [0.0, 0.0, 3.0],
            cp=[0.4, math.nan, math.nan],
            ct=[0.8, math.nan, 0.6],
        )
        at = curve.at(8.0)
        assert math.isnan(at["cp"])
        assert 0.6 < at["ct"] < 0.8

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"rotor_speed_rpm": (1.0,)}, "one length"),
            ({"cp": (0.1,)}, "one length"),
            ({"ct": (0.1,)}, "one length"),
            (
                {
                    "wind_speed": (4.0,),
                    "rotor_speed_rpm": (1.0,),
                    "blade_pitch_deg": (0.0,),
                },
                "at least two",
            ),
            ({"blade_pitch_deg": (0.0, math.nan)}, "finite"),
            ({"wind_speed": (8.0, 4.0)}, "strictly increasing"),
        ],
    )
    def test_rejects_invalid_curves(
        self, kwargs: dict[str, object], match: str
    ) -> None:
        """Malformed curves raise ValueError."""
        base: dict[str, object] = {
            "wind_speed": (4.0, 8.0),
            "rotor_speed_rpm": (7.0, 9.0),
            "blade_pitch_deg": (0.0, 0.0),
        }
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            RotorPerformanceCurve(**base)  # type: ignore[arg-type]

    def test_reads_unit_annotated_csv_with_coefficients(self, tmp_path: Path) -> None:
        """Unit-annotated headers parse; cp/ct are read and sorted with the wind."""
        path = tmp_path / "curve.csv"
        path.write_text(
            "wind_speed [m/s],rotor_speed_rpm [rpm],blade_pitch_deg [deg],cp,ct [-]\n"
            "8,9.5,0,0.45,\n"
            "4,7,0,0.40,0.8\n"
            "12,12.1,4,n/a,0.5\n",
            encoding="utf-8",
        )
        curve = rotor_performance_from_csv(path)
        assert curve.wind_speed == (4.0, 8.0, 12.0)
        assert curve.cp is not None
        assert curve.cp[:2] == (0.40, 0.45)
        assert math.isnan(curve.cp[2])
        assert curve.ct is not None
        assert math.isnan(curve.ct[1])

    def test_csv_without_coefficients(self, tmp_path: Path) -> None:
        """A CSV with only the required columns yields no coefficients."""
        path = tmp_path / "curve.csv"
        path.write_text(
            "WindSpeed,RotSpeed,BldPitch\n4,7,0\n8,9.5,0\n", encoding="utf-8"
        )
        curve = rotor_performance_from_csv(path)
        assert curve.cp is None
        assert curve.ct is None

    def test_csv_with_two_cp_columns_is_rejected(self, tmp_path: Path) -> None:
        """Ambiguous coefficient columns raise ValueError."""
        path = tmp_path / "curve.csv"
        path.write_text(
            "wind_speed,rotor_speed_rpm,blade_pitch_deg,cp,Cp [-]\n"
            "4,7,0,0.4,0.4\n8,9.5,0,0.45,0.45\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="several cp"):
            rotor_performance_from_csv(path)


class TestCoefficientUnits:
    """Units of the cp and ct columns of a rotor-performance CSV."""

    @staticmethod
    def _csv(tmp_path: Path, cp: str, ct: str, cps: tuple[str, str]) -> Path:
        """Write a two-row CSV with the given cp/ct headers and cp values."""
        path = tmp_path / "curve.csv"
        path.write_text(
            f"wind_speed,rotor_speed_rpm,blade_pitch_deg,{cp},{ct}\n"
            f"4,7,0,{cps[0]},80\n8,9.5,0,{cps[1]},75\n",
            encoding="utf-8",
        )
        return path

    @pytest.mark.parametrize("unit", ["", " [-]", " [1]", " (fraction)", " [ - ]"])
    def test_fraction_forms(self, tmp_path: Path, unit: str) -> None:
        """No unit, [-], [1] and [fraction] all mean a fraction."""
        path = tmp_path / "curve.csv"
        path.write_text(
            f"wind_speed,rotor_speed_rpm,blade_pitch_deg,cp{unit},ct{unit}\n"
            "4,7,0,0.40,0.80\n8,9.5,0,0.45,0.75\n",
            encoding="utf-8",
        )
        curve = rotor_performance_from_csv(path)
        assert curve.cp == (0.40, 0.45)
        assert curve.ct == (0.80, 0.75)

    @pytest.mark.parametrize("unit", ["[%]", "(%)", "[percent]", "[Pct]"])
    def test_percent_forms_are_converted(self, tmp_path: Path, unit: str) -> None:
        """A declared percentage is divided by 100; NaN cells stay unknown."""
        path = self._csv(tmp_path, f"cp {unit}", f"ct {unit}", ("40", ""))
        curve = rotor_performance_from_csv(path)
        assert curve.cp is not None
        assert curve.cp[0] == pytest.approx(0.40)
        assert math.isnan(curve.cp[1])
        assert curve.ct == pytest.approx((0.80, 0.75))

    def test_named_coefficient_columns(self, tmp_path: Path) -> None:
        """The long column names take units the same way."""
        path = tmp_path / "curve.csv"
        path.write_text(
            "wind_speed,rotor_speed_rpm,blade_pitch_deg,"
            "Power Coefficient [%],thrust_coefficient [-]\n"
            "4,7,0,40,0.8\n8,9.5,0,45,0.75\n",
            encoding="utf-8",
        )
        curve = rotor_performance_from_csv(path)
        assert curve.cp == pytest.approx((0.40, 0.45))
        assert curve.ct == (0.8, 0.75)

    @pytest.mark.parametrize("unit", ["[W]", "[kW]", "[deg]", "[ratio]"])
    def test_unsupported_unit_is_rejected(self, tmp_path: Path, unit: str) -> None:
        """Any other unit on a coefficient column is an error naming the forms."""
        path = self._csv(tmp_path, f"cp {unit}", "ct [%]", ("0.4", "0.45"))
        with pytest.raises(ValueError, match=r"unsupported unit .*fraction.*percent"):
            rotor_performance_from_csv(path)

    def test_percent_without_unit_is_rejected(self, tmp_path: Path) -> None:
        """Undeclared percentages exceed the Betz limit and are not guessed at."""
        path = self._csv(tmp_path, "cp", "ct [%]", ("40", "45"))
        with pytest.raises(ValueError, match=r"Betz limit.*declare it as 'cp \[%\]'"):
            rotor_performance_from_csv(path)

    def test_thrust_percent_without_unit_is_rejected(self, tmp_path: Path) -> None:
        """A ct column of percentages without a unit exceeds the bound of 2."""
        path = self._csv(tmp_path, "cp", "ct", ("0.40", "0.45"))
        with pytest.raises(ValueError, match=r"ct of 80 .*declare it as 'ct \[%\]'"):
            rotor_performance_from_csv(path)

    def test_fractions_declared_as_percent_are_rejected(self, tmp_path: Path) -> None:
        """Values of at most 1 under a percent unit are fractions mislabelled."""
        path = self._csv(tmp_path, "cp [%]", "ct [%]", ("0.40", "0.45"))
        with pytest.raises(ValueError, match="declares a percentage"):
            rotor_performance_from_csv(path)

    def test_percent_above_bound_is_rejected(self, tmp_path: Path) -> None:
        """A declared percentage above the Betz limit is still impossible."""
        path = self._csv(tmp_path, "cp [%]", "ct [%]", ("40", "75"))
        with pytest.raises(ValueError, match="check the values and their unit"):
            rotor_performance_from_csv(path)

    def test_units_row_is_reported(self, tmp_path: Path) -> None:
        """A separate units row fails with a hint to use the header."""
        path = tmp_path / "curve.csv"
        path.write_text(
            "wind_speed,rotor_speed_rpm,blade_pitch_deg,cp\n"
            "m/s,rpm,deg,-\n4,7,0,0.4\n8,9.5,0,0.45\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="units row is not supported"):
            rotor_performance_from_csv(path)


class TestTurbineConstants:
    """Tests for TurbineConstants."""

    def test_from_deck_reads_elastodyn_and_aerodyn(self, tmp_path: Path) -> None:
        """Gearbox, efficiency (percent), projected radius and AeroDyn density."""
        constants = TurbineConstants.from_deck(_write_deck(tmp_path))
        assert constants.gearbox_ratio == pytest.approx(GB_RATIO)
        assert constants.gearbox_efficiency == pytest.approx(GB_EFF)
        assert constants.rotor_radius == pytest.approx(RADIUS)
        assert constants.air_density == pytest.approx(1.225)
        assert constants.rotor_area == pytest.approx(math.pi * RADIUS**2)

    def test_tip_radius_convention(self, tmp_path: Path) -> None:
        """The tip convention uses TipRad unprojected."""
        constants = TurbineConstants.from_deck(
            _write_deck(tmp_path), radius_convention="tip"
        )
        assert constants.rotor_radius == pytest.approx(TIP_RAD)

    def test_default_air_density_falls_back_to_fst(self, tmp_path: Path) -> None:
        """AeroDyn ``"default"`` density defers to the .fst AirDens."""
        deck = _write_deck(tmp_path, ad_air_density='"default"', fst_air_density="1.1")
        assert TurbineConstants.from_deck(deck).air_density == pytest.approx(1.1)

    def test_simplified_elastodyn(self, tmp_path: Path) -> None:
        """Simplified-ElastoDyn keys (GBoxRatio, PreCone) and a lossless gearbox."""
        constants = TurbineConstants.from_deck(_write_deck(tmp_path, simplified=True))
        assert constants.gearbox_ratio == pytest.approx(GB_RATIO)
        assert constants.gearbox_efficiency == 1.0
        assert constants.rotor_radius == pytest.approx(RADIUS)

    def test_missing_air_density_is_an_error(self, tmp_path: Path) -> None:
        """No density in AeroDyn or the fst raises a clear error."""
        deck = _write_deck(tmp_path, ad_air_density='"default"', fst_air_density=None)
        with pytest.raises(ValueError, match="no air density"):
            TurbineConstants.from_deck(deck)

    def test_missing_aerodyn_file_uses_fst(self, tmp_path: Path) -> None:
        """An unreadable AeroDyn file falls back to the fst density."""
        deck = _write_deck(tmp_path, fst_air_density="1.2")
        (tmp_path / "aero.dat").unlink()
        assert TurbineConstants.from_deck(deck).air_density == pytest.approx(1.2)

    def test_non_numeric_value_is_an_error(self, tmp_path: Path) -> None:
        """A non-numeric constant raises ValueError naming the keyword."""
        deck = _write_deck(tmp_path)
        structure = tmp_path / "structure.dat"
        structure.write_text(
            structure.read_text(encoding="utf-8").replace(
                f"{GB_RATIO}   GBRatio", "abc   GBRatio"
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="GBRatio"):
            TurbineConstants.from_deck(deck)

    def test_missing_structural_file_is_an_error(self, tmp_path: Path) -> None:
        """An unreadable structural file raises ValueError."""
        deck = _write_deck(tmp_path)
        (tmp_path / "structure.dat").unlink()
        with pytest.raises(ValueError, match="structural input"):
            TurbineConstants.from_deck(deck)

    def test_missing_keyword_is_an_error(self, tmp_path: Path) -> None:
        """A missing TipRad raises ValueError."""
        deck = _write_deck(tmp_path)
        structure = tmp_path / "structure.dat"
        lines = structure.read_text(encoding="utf-8").splitlines()
        structure.write_text(
            "\n".join(line for line in lines if "TipRad" not in line) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="TipRad"):
            TurbineConstants.from_deck(deck)

    def test_unknown_radius_convention(self, tmp_path: Path) -> None:
        """An unknown radius convention is rejected."""
        with pytest.raises(ValueError, match="radius_convention"):
            TurbineConstants.from_deck(
                _write_deck(tmp_path),
                radius_convention="hub",  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"gearbox_ratio": 0.0},
            {"gearbox_efficiency": 94.4},
            {"rotor_radius": math.nan},
            {"air_density": -1.0},
        ],
    )
    def test_validation(self, kwargs: dict[str, float]) -> None:
        """Non-positive, non-finite or percent-valued constants are rejected."""
        base = {
            "gearbox_ratio": 1.0,
            "gearbox_efficiency": 1.0,
            "rotor_radius": 1.0,
            "air_density": 1.0,
        }
        base.update(kwargs)
        with pytest.raises(ValueError, match=r"gearbox|rotor_radius|air_density"):
            TurbineConstants(**base)


class TestAerodynamicPower:
    """Tests for the torque-balance power and its loss sign convention."""

    def test_generating_adds_gearbox_loss(self) -> None:
        """Rotor-to-generator power: P = N Q_g Omega / eta."""
        power = aerodynamic_power(1000.0, 30.0 / math.pi, CONSTANTS)
        assert power == pytest.approx(1000.0 * GB_RATIO / GB_EFF)

    def test_motoring_subtracts_gearbox_loss(self) -> None:
        """Generator-to-rotor power: P = N Q_g Omega * eta (less negative)."""
        power = aerodynamic_power(-1000.0, 30.0 / math.pi, CONSTANTS)
        assert power == pytest.approx(-1000.0 * GB_RATIO * GB_EFF)


class TestTrimmedOperatingPoint:
    """Tests for trimmed_operating_point."""

    def test_averages_blades_and_azimuths(self) -> None:
        """Pitch is the mean over blades and azimuths; spread is reported."""
        lins = [
            _lin(wind=12.0, rpm=12.0, pitch_deg=[4.0, 4.2, 4.1], torque=100.0),
            _lin(wind=12.0, rpm=12.2, pitch_deg=[4.3, 4.0, 4.1], torque=110.0),
        ]
        point = trimmed_operating_point(lins)
        assert point.blade_pitch_deg == pytest.approx(
            np.mean([4.0, 4.2, 4.1, 4.3, 4.0, 4.1])
        )
        assert point.blade_pitch_spread_deg == pytest.approx(0.3)
        assert point.generator_torque == pytest.approx(105.0)
        assert point.rotor_speed_rpm == pytest.approx(12.1)
        assert point.n_lin == 2
        assert point.pitch_source == "input: ED Blade <i> pitch command"
        assert point.generator_torque_source == "input: ED Generator torque"
        assert point.rotor_thrust is None
        assert point.issues == ()

    def test_collective_pitch_input(self) -> None:
        """Without per-blade commands the collective command is used."""
        lin = _lin(wind=12.0, rpm=12.0, pitch_deg=5.0, torque=1.0)
        keep = [i for i, d in enumerate(lin.u.descriptions) if "Blade" not in d]
        lin.u = _table(
            [float(lin.u.values[i]) for i in keep],
            [lin.u.descriptions[i] for i in keep],
        )
        point = trimmed_operating_point([lin])
        assert point.blade_pitch_deg == pytest.approx(5.0)
        assert point.pitch_source == "input: ED collective blade-pitch command"

    def test_outputs_fallback_without_inputs(self) -> None:
        """With LinInputs = 0 the BldPitch and GenTq outputs (kN-m) are used."""
        outputs = {
            "SrvD GenTq, (kN-m)": 43.0,
            "ED BldPitch1, (deg)": 10.0,
            "ED BldPitch2, (deg)": 10.2,
            "ED BldPitch3, (deg)": 10.4,
            "AD RtFldFxh, (N)": 5.0e5,
            "AD RtFldPwr, (W)": 5.0e6,
            "ED RotThrust, (kN)": 900.0,
        }
        lin = _lin(
            wind=16.0,
            rpm=12.1,
            pitch_deg=0.0,
            torque=0.0,
            inputs=False,
            outputs=outputs,
        )
        point = trimmed_operating_point([lin])
        assert point.blade_pitch_deg == pytest.approx(10.2)
        assert point.generator_torque == pytest.approx(43.0e3)
        assert point.pitch_source == "output: ED BldPitch<i>"
        assert point.generator_torque_source == "output: SrvD GenTq"
        assert point.rotor_thrust == pytest.approx(5.0e5)
        assert point.rotor_thrust_source == "output: AD RtFldFxh"
        assert point.aerodyn_power == pytest.approx(5.0e6)

    def test_pitch_angle_output_fallback(self) -> None:
        """The ElastoDyn ``Blade <i> pitch angle, rad`` output is a last resort."""
        lin = _lin(
            wind=8.0,
            rpm=9.0,
            pitch_deg=0.0,
            torque=0.0,
            inputs=False,
            outputs={"ED Blade 1 pitch angle, rad": math.radians(2.0)},
        )
        point = trimmed_operating_point([lin])
        assert point.blade_pitch_deg == pytest.approx(2.0)
        assert point.pitch_source == "output: ED Blade <i> pitch angle"

    def test_missing_everything_is_reported(self) -> None:
        """Without inputs or outputs the quantities are NaN and explained."""
        lin = _lin(wind=8.0, rpm=9.0, pitch_deg=0.0, torque=0.0, inputs=False)
        point = trimmed_operating_point([lin])
        assert math.isnan(point.blade_pitch_deg)
        assert math.isnan(point.generator_torque)
        assert any("blade pitch unavailable" in issue for issue in point.issues)
        assert any("generator torque unavailable" in issue for issue in point.issues)

    def test_ignores_elastodyn_rotor_thrust(self) -> None:
        """ElastoDyn's shaft thrust (weight-contaminated) is not used for Ct."""
        lin = _lin(
            wind=8.0,
            rpm=9.0,
            pitch_deg=0.0,
            torque=1.0,
            outputs={"ED RotThrust, (kN)": 500.0},
        )
        assert trimmed_operating_point([lin]).rotor_thrust is None

    def test_inconsistent_wind_is_reported(self) -> None:
        """Azimuth files at different winds are flagged as an issue."""
        lins = [
            _lin(wind=8.0, rpm=9.0, pitch_deg=0.0, torque=1.0),
            _lin(wind=9.0, rpm=9.0, pitch_deg=0.0, torque=1.0),
        ]
        point = trimmed_operating_point(lins)
        assert any("wind speed varies" in issue for issue in point.issues)

    def test_empty_set_is_rejected(self) -> None:
        """An empty set raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            trimmed_operating_point([])

    def test_reads_paths(self, tmp_path: Path) -> None:
        """Paths are parsed with the .lin reader."""
        state = ("ED Variable speed generator DOF, rad", False, 2)
        inputs = [
            ("ED Blade 1 pitch command, rad", True, 0),
            ("ED Blade 2 pitch command, rad", True, 0),
            ("ED Blade 3 pitch command, rad", True, 0),
            ("ED Generator torque, Nm", False, 0),
        ]
        text = build_lin_text(
            states=[
                state,
                ("ED First time derivative of Variable speed, rad/s", False, 2),
            ],
            inputs=inputs,
            a=np.zeros((2, 2)),
            b=np.zeros((2, 4)),
            rotor_speed=1.2,
            wind_speed=11.0,
            input_values=[0.1, 0.1, 0.1, 4.0e4],
        )
        path = tmp_path / "case.1.lin"
        path.write_text(text, encoding="utf-8")
        point = trimmed_operating_point([path])
        assert point.blade_pitch_deg == pytest.approx(math.degrees(0.1))
        assert point.generator_torque == pytest.approx(4.0e4)
        assert point.wind_speed == pytest.approx(11.0)


class TestQuantityCheck:
    """Tests for QuantityCheck."""

    def test_relative_and_absolute(self) -> None:
        """Relative and absolute tolerances are applied as documented."""
        assert QuantityCheck(1.04, 1.0, 0.05, relative=True).status == "pass"
        assert QuantityCheck(1.06, 1.0, 0.05, relative=True).status == "flag"
        assert QuantityCheck(4.4, 4.0, 0.5, relative=False).status == "pass"
        assert QuantityCheck(4.6, 4.0, 0.5, relative=False).status == "flag"
        assert QuantityCheck(math.nan, 4.0, 0.5, relative=False).status == "unavailable"

    def test_zero_curve_value(self) -> None:
        """A zero curve value passes only a zero trimmed value (relative check)."""
        assert QuantityCheck(0.0, 0.0, 0.01, relative=True).status == "pass"
        check = QuantityCheck(0.1, 0.0, 0.01, relative=True)
        assert check.status == "flag"
        assert math.isnan(check.relative_deviation)


class TestVerifyOperatingPoints:
    """Tests for verify_operating_points."""

    def test_consistent_points_pass(self) -> None:
        """Points trimmed exactly onto the curve pass with near-zero deviations."""
        report = verify_operating_points(
            [_consistent_set(i) for i in range(len(CURVE_WIND))],
            _curve(),
            constants=CONSTANTS,
        )
        assert report.passed
        assert report.flagged == ()
        for point in report.points:
            assert point.cp.relative_deviation == pytest.approx(0.0, abs=1e-12)
            assert point.ct.relative_deviation == pytest.approx(0.0, abs=1e-12)
            assert point.pitch.deviation == pytest.approx(0.0, abs=1e-12)
        assert [p.label for p in report.points] == [
            "ws4",
            "ws8",
            "ws12",
            "ws16",
            "ws20",
        ]

    def test_wrong_gearbox_efficiency_is_flagged(self) -> None:
        """Ignoring a 5.6 % gearbox loss biases Cp low and is flagged at 5 %."""
        lossless = TurbineConstants(
            gearbox_ratio=GB_RATIO,
            gearbox_efficiency=1.0,
            rotor_radius=RADIUS,
            air_density=RHO,
        )
        report = verify_operating_points(
            [_consistent_set(2)], _curve(), constants=lossless
        )
        (point,) = report.points
        assert point.cp.relative_deviation == pytest.approx(GB_EFF - 1.0)
        assert point.status == "flag"
        assert point.flagged_quantities == ("cp",)
        assert not report.passed
        assert "flag [cp]" in report.to_table()

    def test_tolerances_are_configurable(self) -> None:
        """A loose Cp tolerance accepts the biased point."""
        report = verify_operating_points(
            [_consistent_set(2, cp_scale=1.08)],
            _curve(),
            constants=CONSTANTS,
            tolerances=TrimTolerances(cp_rel=0.1),
        )
        assert report.passed

    def test_deck_supplies_constants(self, tmp_path: Path) -> None:
        """Constants are read from the deck when not given."""
        report = verify_operating_points(
            [_consistent_set(1)], _curve(), deck=_write_deck(tmp_path)
        )
        assert report.constants == TurbineConstants.from_deck(tmp_path / "turbine.fst")
        assert report.passed

    def test_tuple_curve_compares_speed_and_pitch_only(self) -> None:
        """A (wind, rpm, pitch) curve has no Cp/Ct; the point still passes."""
        report = verify_operating_points(
            [_consistent_set(3)],
            (CURVE_WIND, CURVE_RPM, CURVE_PITCH),
            constants=CONSTANTS,
            labels=["above-rated"],
        )
        (point,) = report.points
        assert point.label == "above-rated"
        assert point.cp.status == "unavailable"
        assert point.status == "pass"

    def test_csv_curve_path(self, tmp_path: Path) -> None:
        """A CSV path is read including its coefficients."""
        path = tmp_path / "curve.csv"
        rows = zip(CURVE_WIND, CURVE_RPM, CURVE_PITCH, CURVE_CP, CURVE_CT, strict=True)
        path.write_text(
            "wind_speed [m/s],rotor_speed_rpm [rpm],blade_pitch_deg [deg],cp,ct\n"
            + "".join(",".join(str(v) for v in row) + "\n" for row in rows),
            encoding="utf-8",
        )
        report = verify_operating_points(
            [_consistent_set(0)], path, constants=CONSTANTS
        )
        assert report.points[0].cp.status == "pass"
        assert report.points[0].ct.status == "pass"

    def test_out_of_range_and_missing_data_are_incomplete(self) -> None:
        """A wind outside the curve, or missing pitch, makes a point incomplete."""
        outside = [_lin(wind=25.0, rpm=12.1, pitch_deg=20.0, torque=1.0)]
        missing = [_lin(wind=8.0, rpm=9.5, pitch_deg=0.0, torque=1.0, inputs=False)]
        report = verify_operating_points(
            [outside, missing], _curve(), constants=CONSTANTS
        )
        assert [p.status for p in report.points] == ["incomplete", "incomplete"]
        assert any("outside the curve range" in i for i in report.points[0].issues)
        assert not report.passed

    def test_zero_wind_has_no_coefficients(self) -> None:
        """At zero wind the coefficients are undefined and reported as such."""
        curve = RotorPerformanceCurve.from_arrays([0.0, 4.0], [0.0, 7.0], [90.0, 0.0])
        report = verify_operating_points(
            [[_lin(wind=0.0, rpm=0.0, pitch_deg=90.0, torque=0.0)]],
            curve,
            constants=CONSTANTS,
        )
        (point,) = report.points
        assert math.isnan(point.cp.trimmed)
        assert any("zero wind" in issue for issue in point.issues)

    def test_records(self) -> None:
        """to_records exposes every compared quantity."""
        report = verify_operating_points(
            [_consistent_set(2)], _curve(), constants=CONSTANTS
        )
        (record,) = report.to_records()
        assert record["label"] == "ws12"
        assert record["status"] == "pass"
        assert record["cp_curve"] == pytest.approx(0.35)
        assert record["ct_status"] == "pass"

    def test_requires_deck_or_constants(self) -> None:
        """Without a deck or constants there is nothing to normalize with."""
        with pytest.raises(ValueError, match="deck="):
            verify_operating_points([_consistent_set(0)], _curve())

    def test_label_count_must_match(self) -> None:
        """Labels must match the number of sets."""
        with pytest.raises(ValueError, match="labels"):
            verify_operating_points(
                [_consistent_set(0)], _curve(), constants=CONSTANTS, labels=["a", "b"]
            )

    @pytest.mark.parametrize(
        "name", ["rotor_speed_rel", "pitch_deg", "cp_rel", "ct_rel"]
    )
    def test_tolerances_validated(self, name: str) -> None:
        """Negative tolerances are rejected."""
        with pytest.raises(ValueError, match=name):
            TrimTolerances(**{name: -1.0})


def _sweep_case(
    tmp_path: Path, name: str, lins: list[LinFile], *, converged: bool = True
) -> SweepCase:
    case_dir = tmp_path / name
    fst = _write_deck(case_dir)
    paths = []
    for k, lin in enumerate(lins, start=1):
        names_u = lin.u.descriptions
        text = build_lin_text(
            states=[
                ("ED Variable speed generator DOF, rad", False, 2),
                (
                    "ED First time derivative of Variable speed generator DOF, rad/s",
                    False,
                    2,
                ),
            ],
            inputs=[(desc, False, 0) for desc in names_u],
            outputs=[(desc, False, 0) for desc in lin.y.descriptions],
            a=np.zeros((2, 2)),
            b=np.zeros((2, len(names_u))),
            c=np.zeros((len(lin.y.descriptions), 2)),
            d=np.zeros((len(lin.y.descriptions), len(names_u))),
            rotor_speed=lin.rotor_speed,
            wind_speed=lin.wind_speed,
            azimuth=lin.azimuth,
            input_values=list(lin.u.values),
            output_values=list(lin.y.values),
        )
        path = case_dir / f"turbine.{k}.lin"
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    point = lins[0]
    return SweepCase(
        operating_point=OperatingPoint(
            wind_speed=point.wind_speed,
            rotor_speed_rpm=point.rotor_speed * 30.0 / math.pi,
            blade_pitch_deg=0.0,
            name=name,
        ),
        case_dir=case_dir,
        fst_path=fst,
        result=OpenFastResult(fst, 0, lin_files=paths if converged else []),
        settings=LinearizationSettings(),
        converged=converged,
    )


class TestVerifySweepOperatingPoints:
    """Tests for verify_sweep_operating_points."""

    def test_verifies_converged_cases_and_skips_the_rest(self, tmp_path: Path) -> None:
        """Converged cases are verified (deck constants); others are skipped."""
        good = _sweep_case(tmp_path, "ws8", _consistent_set(1))
        bad = _sweep_case(tmp_path, "ws12", _consistent_set(2), converged=False)
        empty = _sweep_case(tmp_path, "ws16", _consistent_set(3))
        empty.result.lin_files.clear()
        sweep = OperatingSweepResult(
            operating_points=[c.operating_point for c in (good, bad, empty)],
            cases=[good, bad, empty],
        )
        report = verify_sweep_operating_points(sweep, _curve())
        assert [p.label for p in report.points] == ["ws8"]
        assert report.points[0].status == "pass"
        assert report.skipped == (
            "ws12: linearization did not converge",
            "ws16: no .lin files",
        )
        assert not report.passed
        assert "skipped: ws12" in report.to_table()

    def test_nothing_to_verify(self, tmp_path: Path) -> None:
        """A sweep without a usable case raises ValueError."""
        bad = _sweep_case(tmp_path, "ws12", _consistent_set(2), converged=False)
        sweep = OperatingSweepResult(
            operating_points=[bad.operating_point], cases=[bad]
        )
        with pytest.raises(ValueError, match="no converged"):
            verify_sweep_operating_points(sweep, _curve())


def _reference_runs() -> Path | None:
    """Return the local reference-turbine sweep directory, if present."""
    env = os.environ.get("VANE_REFERENCE_RUNS_DIR")
    candidates = [Path(env)] if env else []
    return next((path for path in candidates if path.is_dir()), None)


@pytest.mark.parametrize(
    ("key", "deck", "csv_name", "cp_rtol"),
    [
        (
            "nrel5mw_land",
            "5MW_Land_DLL_WTurb/5MW_Land_DLL_WTurb.fst",
            "nrel5mw_curve.csv",
            0.03,
        ),
        ("iea3p4_land", "IEA-3.4-130-RWT.fst", "iea3p4_curve.csv", 0.025),
    ],
)
def test_reference_turbine_sweeps(
    key: str, deck: str, csv_name: str, cp_rtol: float
) -> None:
    """Real CalcSteady sweeps agree with their rotor-performance curves.

    Skips unless the (never committed) reference-turbine sweep output is available,
    through ``VANE_REFERENCE_RUNS_DIR``.
    """
    runs = _reference_runs()
    if runs is None or not (runs / key / "cases").is_dir():
        pytest.skip("reference-turbine sweep data unavailable")
    curve = runs.parent / "_performance" / csv_name
    sets, labels = [], []
    for case_dir in sorted((runs / key / "cases").iterdir()):
        lins = find_lin_files(case_dir / deck)
        if lins:
            sets.append(lins)
            labels.append(case_dir.name)
    report = verify_operating_points(
        sets, curve, deck=Path(sets[0][0]).parent / Path(deck).name, labels=labels
    )
    assert report.passed, report.to_table()
    for point in report.points:
        assert abs(point.cp.relative_deviation) < cp_rtol
        assert abs(point.pitch.deviation) < 0.5
        assert point.trimmed.n_lin == 36


def _damped_set(
    root: Path, *, calc_steady: str = "True", kdmp: str | None = "5000.0"
) -> list[LinFile]:
    """Return curve point 1 with AeroDyn power, written beside a CalcSteady deck.

    The generator torque is trimmed 10 % low, as if trim damping took the rest of
    the aerodynamic torque; ``AD RtFldPwr`` carries the true aerodynamic power.
    """
    root.mkdir(parents=True, exist_ok=True)
    lines = [f"{calc_steady}   CalcSteady  - trim (flag)"]
    if kdmp is not None:
        lines += [f"{kdmp}   Twr_Kdmp  - tower damping", "0.0   Bld_Kdmp  - blade"]
    (root / "case.fst").write_text("\n".join(lines) + "\n", encoding="utf-8")
    wind, rpm = CURVE_WIND[1], CURVE_RPM[1]
    power = CURVE_CP[1] * 0.5 * RHO * math.pi * RADIUS**2 * wind**3
    torque = _generator_torque(wind, rpm, CURVE_CP[1] * 0.9)
    lins = []
    for k in range(3):
        lin = _lin(
            wind=wind,
            rpm=rpm,
            pitch_deg=CURVE_PITCH[1],
            torque=torque,
            outputs={"AD RtFldPwr, (W)": power},
        )
        lin.path = root / f"case.{k + 1}.lin"
        lins.append(lin)
    return lins


class TestTrimDamping:
    """CalcSteady trim damping absorbs rotor torque; AeroDyn's power is used then."""

    def test_trim_damping_is_read_from_the_deck(self, tmp_path: Path) -> None:
        """The larger Twr_Kdmp/Bld_Kdmp of the deck beside the files is reported."""
        point = trimmed_operating_point(_damped_set(tmp_path))
        assert point.trim_damping == pytest.approx(5000.0)
        assert point.aerodyn_power_source == "output: AD RtFldPwr"

    def test_damped_point_uses_aerodyn_power(self, tmp_path: Path) -> None:
        """With trim damping the power and Cp come from AeroDyn, which matches."""
        report = verify_operating_points(
            [_damped_set(tmp_path)], _curve(), constants=CONSTANTS
        )
        point = report.points[0]
        assert point.power_source == "output: AD RtFldPwr"
        assert point.cp.relative_deviation == pytest.approx(0.0, abs=1e-9)
        assert report.to_records()[0]["power_source"] == "output: AD RtFldPwr"
        assert report.to_records()[0]["trim_damping"] == pytest.approx(5000.0)

    @pytest.mark.parametrize(
        ("calc_steady", "kdmp"), [("False", "5000.0"), ("True", "0.0"), ("True", None)]
    )
    def test_undamped_point_uses_the_torque_balance(
        self, tmp_path: Path, calc_steady: str, kdmp: str | None
    ) -> None:
        """Without trim damping the torque balance is used (here 10 % low)."""
        lins = _damped_set(tmp_path, calc_steady=calc_steady, kdmp=kdmp)
        point = verify_operating_points([lins], _curve(), constants=CONSTANTS).points[0]
        assert point.trimmed.trim_damping == 0.0
        assert point.power_source == "torque balance"
        assert point.cp.relative_deviation == pytest.approx(-0.1, rel=1e-6)

    def test_missing_deck_gives_unknown_damping(self, tmp_path: Path) -> None:
        """Files without a readable deck beside them keep the torque balance."""
        lins = _damped_set(tmp_path)
        (tmp_path / "case.fst").unlink()
        point = verify_operating_points([lins], _curve(), constants=CONSTANTS).points[0]
        assert point.trimmed.trim_damping is None
        assert point.power_source == "torque balance"

    def test_non_numeric_damping_gives_unknown(self, tmp_path: Path) -> None:
        """A damping value that is not a number is treated as unknown."""
        lins = _damped_set(tmp_path, kdmp="lots")
        assert trimmed_operating_point(lins).trim_damping is None

    def test_damping_without_aerodyn_power_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Without RtFldPwr the torque balance is kept and a warning logged."""
        lins = _damped_set(tmp_path)
        for lin in lins:
            lin.y = _table([], [])
        with caplog.at_level("WARNING"):
            report = verify_operating_points([lins], _curve(), constants=CONSTANTS)
        assert report.points[0].power_source == "torque balance"
        assert "trim damping" in caplog.text


def test_iea10_damped_sweep_uses_aerodyn_power() -> None:
    """The IEA 10-MW sweep was trimmed with damping; its power is AeroDyn's.

    Skips unless the reference-turbine sweep output is available. Its trimmed
    generator torque is 1.5-10 % below AeroDyn's rotor torque because the
    CalcSteady trim damping absorbs the difference.
    """
    runs = _reference_runs()
    if runs is None or not (runs / "iea10_monopile" / "cases").is_dir():
        pytest.skip("reference-turbine sweep data unavailable")
    deck = "IEA-10.0-198-RWT.fst"
    sets = [
        lins
        for case_dir in sorted((runs / "iea10_monopile" / "cases").iterdir())
        if (lins := find_lin_files(case_dir / deck))
    ]
    if not sets:
        pytest.skip("IEA 10-MW sweep has no linearizations")
    report = verify_operating_points(
        sets,
        runs.parent / "_performance" / "iea10_curve.csv",
        deck=Path(sets[0][0]).parent / deck,
    )
    for point in report.points:
        assert point.trimmed.trim_damping is not None
        assert point.trimmed.trim_damping > 0.0
        assert point.power_source == "output: AD RtFldPwr"
        assert point.aero_power == pytest.approx(point.trimmed.aerodyn_power)
        balance = aerodynamic_power(
            point.trimmed.generator_torque,
            point.trimmed.rotor_speed_rpm,
            report.constants,
        )
        assert 0.005 < point.aero_power / balance - 1.0 < 0.12
        assert abs(point.cp.relative_deviation) < 0.06
