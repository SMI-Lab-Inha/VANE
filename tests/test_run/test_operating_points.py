# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.run.operating_points — operating points and sweeps."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from vane.run import (
    OperatingPoint,
    operating_curve_from_csv,
    operating_curve_from_performance,
    operating_points_from_curve,
)
from vane.run.operating_points import PlatformPosition

if TYPE_CHECKING:
    from pathlib import Path


class TestOperatingPoint:
    """Tests for the OperatingPoint value type."""

    def test_is_rotating_requires_wind_and_speed(self) -> None:
        """A point is rotating only with both non-zero rotor speed and wind."""
        assert OperatingPoint(8.0, 9.5, 0.0).is_rotating
        assert not OperatingPoint(0.0, 0.0, 0.0).is_rotating  # standstill
        assert not OperatingPoint(8.0, 0.0, 0.0).is_rotating  # parked in wind
        assert not OperatingPoint(0.0, 9.5, 0.0).is_rotating  # spun, no wind

    def test_non_finite_values_rejected(self) -> None:
        """A NaN/inf operating-point value is rejected at construction."""
        with pytest.raises(ValueError, match="wind_speed must be finite"):
            OperatingPoint(np.nan, 9.5, 0.0)
        with pytest.raises(ValueError, match="rotor_speed_rpm must be finite"):
            OperatingPoint(8.0, np.inf, 0.0)
        with pytest.raises(ValueError, match="blade_pitch_deg must be finite"):
            OperatingPoint(8.0, 9.5, np.nan)

    def test_negative_wind_or_rotor_speed_rejected(self) -> None:
        """A negative wind or rotor speed is rejected (pitch may be negative)."""
        with pytest.raises(ValueError, match="wind_speed must be non-negative"):
            OperatingPoint(-1.0, 9.5, 0.0)
        with pytest.raises(ValueError, match="rotor_speed_rpm must be non-negative"):
            OperatingPoint(8.0, -9.5, 0.0)
        OperatingPoint(8.0, 9.5, -2.0)  # negative pitch is allowed

    def test_non_finite_platform_position_rejected(self) -> None:
        """A non-finite platform DOF is rejected; None DOFs are left unchecked."""
        with pytest.raises(ValueError, match="platform pitch must be finite"):
            PlatformPosition(pitch=np.nan)
        PlatformPosition(surge=1.0)  # a finite, partially-specified position is fine


class TestOperatingPointsFromCurve:
    """Tests for building a sweep from a steady-state operating curve."""

    def test_interpolates_rotor_speed_and_pitch(self) -> None:
        """Each requested wind speed gets the curve's interpolated speed/pitch."""
        points = operating_points_from_curve(
            [4.0, 12.0],
            curve_wind=[4.0, 8.0, 12.0, 16.0],
            curve_rotor_speed=[6.0, 9.0, 12.0, 12.0],
            curve_pitch=[0.0, 0.0, 4.0, 14.0],
        )
        assert [p.wind_speed for p in points] == [4.0, 12.0]
        # At the sample knots the interpolant returns the sample values exactly.
        assert points[0].rotor_speed_rpm == pytest.approx(6.0)
        assert points[1].rotor_speed_rpm == pytest.approx(12.0)
        assert points[1].blade_pitch_deg == pytest.approx(4.0)
        assert points[0].name == "ws4"

    def test_monotone_interpolation_does_not_overshoot(self) -> None:
        """PCHIP stays within the bracketing sample values (no overshoot)."""
        points = operating_points_from_curve(
            [10.0],
            curve_wind=[8.0, 12.0],
            curve_rotor_speed=[9.0, 12.0],
            curve_pitch=[0.0, 4.0],
        )
        assert 9.0 <= points[0].rotor_speed_rpm <= 12.0
        assert 0.0 <= points[0].blade_pitch_deg <= 4.0

    def test_negative_pitch_curve_is_allowed(self) -> None:
        """A curve with a negative collective blade pitch is accepted."""
        points = operating_points_from_curve(
            [4.0],
            curve_wind=[4.0, 8.0],
            curve_rotor_speed=[5.0, 7.5],
            curve_pitch=[-1.5, 0.0],
        )
        assert points[0].blade_pitch_deg == pytest.approx(-1.5)

    @pytest.mark.parametrize(
        ("wind", "rotor", "pitch", "match"),
        [
            ([4.0, 8.0], [1.0], [1.0, 2.0], "share a shape"),
            ([4.0], [1.0], [1.0], "at least two"),
            ([8.0, 4.0], [1.0, 2.0], [1.0, 2.0], "strictly increasing"),
            ([4.0, 8.0], [1.0, float("nan")], [1.0, 2.0], "finite"),
        ],
    )
    def test_invalid_curve_raises(
        self,
        wind: list[float],
        rotor: list[float],
        pitch: list[float],
        match: str,
    ) -> None:
        """Malformed operating curves are rejected with a clear message."""
        with pytest.raises(ValueError, match=match):
            operating_points_from_curve(
                [6.0], curve_wind=wind, curve_rotor_speed=rotor, curve_pitch=pitch
            )

    @pytest.mark.parametrize("speed", [2.0, 20.0, float("nan")])
    def test_out_of_range_speed_raises(self, speed: float) -> None:
        """A requested speed outside the curve range is rejected (no extrapolation)."""
        with pytest.raises(ValueError, match="outside the operating-curve range"):
            operating_points_from_curve(
                [speed],
                curve_wind=[4.0, 8.0, 12.0],
                curve_rotor_speed=[6.0, 9.0, 12.0],
                curve_pitch=[0.0, 0.0, 4.0],
            )


class TestOperatingCurveFromPerformance:
    """Tests for deriving an operating curve from a rotor-performance source."""

    def test_table_is_returned_sorted(self) -> None:
        """A supplied performance table is validated and sorted by wind speed."""
        wind, rotor, pitch = operating_curve_from_performance(
            performance_table=([12.0, 4.0, 8.0], [12.0, 6.0, 9.0], [4.0, 0.0, 0.0])
        )
        assert wind == [4.0, 8.0, 12.0]
        assert rotor == [6.0, 9.0, 12.0]
        assert pitch == [0.0, 0.0, 4.0]

    def test_negative_pitch_in_table_is_allowed(self) -> None:
        """A negative collective blade pitch in the table is accepted."""
        wind, _rotor, pitch = operating_curve_from_performance(
            performance_table=([4.0, 8.0], [5.0, 7.5], [-1.5, 0.0])
        )
        assert wind == [4.0, 8.0]
        assert pitch == [-1.5, 0.0]

    def test_negative_wind_in_table_raises(self) -> None:
        """A negative wind speed in the table is rejected."""
        with pytest.raises(ValueError, match="wind values must be non-negative"):
            operating_curve_from_performance(
                performance_table=([-4.0, 8.0], [5.0, 7.5], [0.0, 1.0])
            )

    def test_negative_rotor_in_table_raises(self) -> None:
        """A negative rotor speed in the table is rejected."""
        with pytest.raises(ValueError, match="rotor values must be non-negative"):
            operating_curve_from_performance(
                performance_table=([4.0, 8.0], [-5.0, 7.5], [0.0, 1.0])
            )

    def test_table_feeds_operating_points_from_curve(self) -> None:
        """The returned arrays drive operating_points_from_curve directly."""
        wind, rotor, pitch = operating_curve_from_performance(
            performance_table=([4.0, 8.0, 12.0], [6.0, 9.0, 12.0], [0.0, 0.0, 4.0])
        )
        points = operating_points_from_curve(
            [8.0], curve_wind=wind, curve_rotor_speed=rotor, curve_pitch=pitch
        )
        assert points[0].rotor_speed_rpm == pytest.approx(9.0)

    def test_no_source_raises_clear_error(self) -> None:
        """With no table supplied, a clear rotor-performance error is raised."""
        with pytest.raises(RuntimeError, match="rotor-performance data required"):
            operating_curve_from_performance()

    def test_mismatched_table_lengths_raise(self) -> None:
        """A table whose arrays differ in length is rejected."""
        with pytest.raises(ValueError, match="must share a shape"):
            operating_curve_from_performance(
                performance_table=([4.0, 8.0], [6.0], [0.0, 0.0])
            )

    def test_short_table_raises(self) -> None:
        """A one-sample table is too short to interpolate."""
        with pytest.raises(ValueError, match="at least two"):
            operating_curve_from_performance(performance_table=([4.0], [6.0], [0.0]))

    def test_non_finite_table_raises(self) -> None:
        """A non-finite table value is rejected."""
        with pytest.raises(ValueError, match="must all be finite"):
            operating_curve_from_performance(
                performance_table=([4.0, float("nan")], [6.0, 9.0], [0.0, 0.0])
            )

    def test_duplicate_wind_axis_raises(self) -> None:
        """A duplicated wind speed in the table is rejected."""
        with pytest.raises(ValueError, match="distinct"):
            operating_curve_from_performance(
                performance_table=([8.0, 8.0], [6.0, 9.0], [0.0, 0.0])
            )


# A well-formed rotor-performance CSV with the optional cp/ct/cq/gen_power columns
# present and rows out of wind order, to exercise sorting and tolerated extra columns.
_SAMPLE_CSV = (
    "WindSpeed,RotSpeed,BldPitch,cp,ct,cq,gen_power\n"
    "12.0,7.5,4.0,0.45,0.70,0.06,15000.0\n"
    "4.0,5.0,0.0,0.48,0.80,0.07,1200.0\n"
    "8.0,7.5,0.0,0.49,0.78,0.08,9000.0\n"
)


class TestOperatingCurveFromCsv:
    """Tests for the rotor-performance CSV parser."""

    def _write(self, tmp_path: Path, text: str, name: str = "perf.csv") -> Path:
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_well_formed_csv_with_optional_columns(self, tmp_path: Path) -> None:
        """A CSV with cp/ct/cq/gen_power parses to the sorted curve, extras ignored."""
        wind, rotor, pitch = operating_curve_from_csv(
            self._write(tmp_path, _SAMPLE_CSV)
        )
        assert wind == [4.0, 8.0, 12.0]  # sorted by wind ascending
        assert rotor == [5.0, 7.5, 7.5]
        assert pitch == [0.0, 0.0, 4.0]

    def test_header_name_variants_tolerated(self, tmp_path: Path) -> None:
        """Case/whitespace/unit-bracket header variants all resolve."""
        text = "V [m/s], Omega [rpm], pitch [deg]\n4.0, 5.0, 0.0\n8.0, 7.5, 1.0\n"
        wind, rotor, pitch = operating_curve_from_csv(self._write(tmp_path, text))
        assert wind == [4.0, 8.0]
        assert rotor == [5.0, 7.5]
        assert pitch == [0.0, 1.0]

    def test_underscore_header_variants_tolerated(self, tmp_path: Path) -> None:
        """The canonical underscore names also parse."""
        text = "wind_speed,rotor_speed_rpm,blade_pitch_deg\n4.0,5.0,0.0\n8.0,7.5,1.0\n"
        wind, _rotor, _pitch = operating_curve_from_csv(self._write(tmp_path, text))
        assert wind == [4.0, 8.0]

    def test_feeds_run_curve(self, tmp_path: Path) -> None:
        """The parsed curve drives operating_points_from_curve."""
        wind, rotor, pitch = operating_curve_from_csv(
            self._write(tmp_path, _SAMPLE_CSV)
        )
        points = operating_points_from_curve(
            [8.0], curve_wind=wind, curve_rotor_speed=rotor, curve_pitch=pitch
        )
        assert points[0].rotor_speed_rpm == pytest.approx(7.5)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """A missing CSV file fails clearly."""
        with pytest.raises(FileNotFoundError, match="not found"):
            operating_curve_from_csv(tmp_path / "absent.csv")

    def test_missing_required_column_raises(self, tmp_path: Path) -> None:
        """A CSV missing the pitch column fails with a clear message."""
        text = "WindSpeed,RotSpeed\n4.0,5.0\n8.0,7.5\n"
        with pytest.raises(ValueError, match="blade_pitch_deg"):
            operating_curve_from_csv(self._write(tmp_path, text))

    def test_non_numeric_value_raises(self, tmp_path: Path) -> None:
        """A non-numeric cell fails with the line number and column."""
        text = "WindSpeed,RotSpeed,BldPitch\n4.0,fast,0.0\n8.0,7.5,1.0\n"
        with pytest.raises(ValueError, match="not numeric"):
            operating_curve_from_csv(self._write(tmp_path, text))

    def test_nan_value_raises(self, tmp_path: Path) -> None:
        """A NaN cell is rejected."""
        text = "WindSpeed,RotSpeed,BldPitch\n4.0,nan,0.0\n8.0,7.5,1.0\n"
        with pytest.raises(ValueError, match="not finite"):
            operating_curve_from_csv(self._write(tmp_path, text))

    def test_negative_rotor_speed_raises(self, tmp_path: Path) -> None:
        """A negative rotor speed is rejected."""
        text = "WindSpeed,RotSpeed,BldPitch\n4.0,-5.0,0.0\n8.0,7.5,1.0\n"
        with pytest.raises(ValueError, match="negative"):
            operating_curve_from_csv(self._write(tmp_path, text))

    def test_negative_wind_speed_raises(self, tmp_path: Path) -> None:
        """A negative wind speed is rejected."""
        text = "WindSpeed,RotSpeed,BldPitch\n-4.0,5.0,0.0\n8.0,7.5,1.0\n"
        with pytest.raises(ValueError, match="negative"):
            operating_curve_from_csv(self._write(tmp_path, text))

    def test_negative_pitch_is_allowed(self, tmp_path: Path) -> None:
        """A negative collective blade pitch is physically valid and accepted."""
        text = "WindSpeed,RotSpeed,BldPitch\n4.0,5.0,-1.5\n8.0,7.5,0.0\n"
        wind, rotor, pitch = operating_curve_from_csv(self._write(tmp_path, text))
        assert wind == [4.0, 8.0]
        assert rotor == [5.0, 7.5]
        assert pitch == [-1.5, 0.0]  # the negative pitch sample survived

    def test_no_data_rows_raises(self, tmp_path: Path) -> None:
        """A header with no data rows fails clearly."""
        text = "WindSpeed,RotSpeed,BldPitch\n"
        with pytest.raises(ValueError, match="no data rows"):
            operating_curve_from_csv(self._write(tmp_path, text))

    def test_single_data_row_raises(self, tmp_path: Path) -> None:
        """A one-row CSV is unusable (two points needed to interpolate); reject it."""
        text = "WindSpeed,RotSpeed,BldPitch\n8.0,7.5,1.0\n"
        with pytest.raises(ValueError, match="at least two operating points"):
            operating_curve_from_csv(self._write(tmp_path, text))

    def test_two_data_rows_is_the_minimum_usable(self, tmp_path: Path) -> None:
        """Exactly two rows is the minimum and parses successfully."""
        text = "WindSpeed,RotSpeed,BldPitch\n4.0,5.0,0.0\n8.0,7.5,1.0\n"
        wind, rotor, pitch = operating_curve_from_csv(self._write(tmp_path, text))
        assert wind == [4.0, 8.0]
        assert rotor == [5.0, 7.5]
        assert pitch == [0.0, 1.0]

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        """A fully empty file fails clearly."""
        with pytest.raises(ValueError, match="empty"):
            operating_curve_from_csv(self._write(tmp_path, ""))

    def test_duplicate_wind_raises(self, tmp_path: Path) -> None:
        """Duplicate wind speeds are rejected (non-monotone axis)."""
        text = "WindSpeed,RotSpeed,BldPitch\n8.0,5.0,0.0\n8.0,7.5,1.0\n"
        with pytest.raises(ValueError, match="strictly increasing"):
            operating_curve_from_csv(self._write(tmp_path, text))

    def test_short_row_raises(self, tmp_path: Path) -> None:
        """A data row with fewer fields than the header fails clearly."""
        text = "WindSpeed,RotSpeed,BldPitch\n4.0,5.0\n8.0,7.5,1.0\n"
        with pytest.raises(ValueError, match="fields but the header"):
            operating_curve_from_csv(self._write(tmp_path, text))

    def test_ambiguous_duplicate_column_raises(self, tmp_path: Path) -> None:
        """Two header columns matching the same required column are rejected."""
        # Both "WindSpeed" and "wind" normalize to the wind-speed column.
        text = "WindSpeed,wind,RotSpeed,BldPitch\n4.0,4.0,5.0,0.0\n8.0,8.0,7.5,1.0\n"
        with pytest.raises(ValueError, match="multiple columns matching"):
            operating_curve_from_csv(self._write(tmp_path, text))


def _write_csv(tmp_path: Path, header: str, rows: str) -> Path:
    path = tmp_path / "units.csv"
    path.write_text(f"{header}\n{rows}", encoding="utf-8")
    return path


class TestOperatingCurveCsvUnits:
    """Tests for unit-aware parsing of the rotor-performance CSV header."""

    _ROWS = "4.0,1.0,0.1\n8.0,2.0,0.2\n"

    def test_rad_per_s_rotor_speed_and_rad_pitch_are_not_misread(
        self, tmp_path: Path
    ) -> None:
        """rad/s and rad columns are converted, not silently read as rpm/deg."""
        path = _write_csv(
            tmp_path,
            "Wind Speed [m/s],Rotor Speed [rad/s],Blade Pitch [rad]",
            self._ROWS,
        )
        wind, rotor, pitch = operating_curve_from_csv(path)
        assert wind == [4.0, 8.0]
        assert rotor == pytest.approx([60.0 / (2.0 * np.pi), 120.0 / (2.0 * np.pi)])
        assert pitch == pytest.approx([np.degrees(0.1), np.degrees(0.2)])

    def test_conversion_overflow_is_rejected(self, tmp_path: Path) -> None:
        """A finite value that overflows on unit conversion is rejected."""
        path = _write_csv(
            tmp_path,
            "Wind Speed [m/s],Rotor Speed [rad/s],Blade Pitch [deg]",
            "4.0,1e308,0.1\n8.0,2.0,0.2\n",
        )
        with pytest.raises(ValueError, match="overflows on conversion"):
            operating_curve_from_csv(path)

    def test_km_per_h_wind_converted(self, tmp_path: Path) -> None:
        """A km/h wind column is converted to m/s."""
        path = _write_csv(
            tmp_path, "wind (km/h),rpm,pitch", "36.0,5.0,0.0\n72.0,7.5,1.0\n"
        )
        wind, rotor, pitch = operating_curve_from_csv(path)
        assert wind == pytest.approx([10.0, 20.0])
        assert rotor == [5.0, 7.5]
        assert pitch == [0.0, 1.0]

    def test_conversion_logged_at_info(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Applying a unit conversion is logged at INFO."""
        path = _write_csv(tmp_path, "V,Omega [rad/s],theta", self._ROWS)
        with caplog.at_level("INFO", logger="vane.run.operating_points"):
            operating_curve_from_csv(path)
        assert any("rad/s" in record.getMessage() for record in caplog.records)

    @pytest.mark.parametrize(
        ("wind_unit", "rotor_unit", "pitch_unit"),
        [
            ("[m/s]", "[rpm]", "[deg]"),
            ("[ M/S ]", "(RPM)", "(Deg)"),
            ("(mps)", "[rev/min]", "[degree]"),
            ("[m s^-1]", "[ Rev / Min ]", "[degrees]"),
            ("", "", ""),
            ("[]", "()", "[ ]"),
            ("[m/s", "(rpm", "[deg"),  # unclosed bracket takes the rest
            (
                "[m·s⁻¹]",
                "[r/min]",
                "[°]",
            ),
        ],
    )
    def test_canonical_spellings_not_converted(
        self, tmp_path: Path, wind_unit: str, rotor_unit: str, pitch_unit: str
    ) -> None:
        """Every accepted canonical-unit spelling leaves values unchanged."""
        header = f"WindSpeed {wind_unit},RotSpeed {rotor_unit},BldPitch {pitch_unit}"
        wind, rotor, pitch = operating_curve_from_csv(
            _write_csv(tmp_path, header, self._ROWS)
        )
        assert wind == [4.0, 8.0]
        assert rotor == [1.0, 2.0]
        assert pitch == [0.1, 0.2]

    @pytest.mark.parametrize(
        ("header", "column", "unit"),
        [
            ("V [ft/s],rpm,pitch", "wind_speed", "ft/s"),
            ("V,Omega [Hz],pitch", "rotor_speed_rpm", "Hz"),
            ("V,rpm,pitch [grad]", "blade_pitch_deg", "grad"),
        ],
    )
    def test_unknown_unit_rejected(
        self, tmp_path: Path, header: str, column: str, unit: str
    ) -> None:
        """An unsupported unit names the column, the unit, and the accepted units."""
        with pytest.raises(ValueError, match="accepted units") as info:
            operating_curve_from_csv(_write_csv(tmp_path, header, self._ROWS))
        assert column in str(info.value)
        assert repr(unit) in str(info.value)

    def test_name_contradicting_unit_rejected(self, tmp_path: Path) -> None:
        """A header naming rpm but bracketed rad/s is ambiguous and rejected."""
        path = _write_csv(
            tmp_path, "wind_speed,rotor_speed_rpm [rad/s],blade_pitch_deg", self._ROWS
        )
        with pytest.raises(ValueError, match="names its canonical unit"):
            operating_curve_from_csv(path)

    def test_ignored_columns_keep_any_unit(self, tmp_path: Path) -> None:
        """Units on tolerated columns (cp, gen_power) are still ignored."""
        path = _write_csv(
            tmp_path,
            "V,rpm,pitch,cp [furlongs],gen_power [kW]",
            "4.0,1.0,0.1,0.4,100\n8.0,2.0,0.2,0.5,200\n",
        )
        wind, _rotor, _pitch = operating_curve_from_csv(path)
        assert wind == [4.0, 8.0]
