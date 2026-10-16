# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for :mod:`vane.run.trim_tuning` and its wiring into the driver."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

import pytest

from vane.run.driver import (
    _RETRY_ESCALATED_FIELDS,
    LinearizationDriver,
    SweepCase,
)
from vane.run.lin_config import LinearizationSettings
from vane.run.operating_points import OperatingPoint
from vane.run.runner import OpenFastResult
from vane.run.trim_tuning import TrimTuningPolicy, tune_trim_settings

if TYPE_CHECKING:
    from pathlib import Path

# Below-rated and above-rated rotating points (default rated_wind_speed is 11.4).
BELOW = OperatingPoint(wind_speed=8.0, rotor_speed_rpm=7.0, blade_pitch_deg=0.0)
ABOVE = OperatingPoint(wind_speed=16.0, rotor_speed_rpm=7.55, blade_pitch_deg=12.0)
AT_RATED = OperatingPoint(wind_speed=11.4, rotor_speed_rpm=7.55, blade_pitch_deg=0.0)


class TestTrimTuningPolicy:
    """Construction and validation of the tuning policy."""

    def test_defaults_are_positive(self) -> None:
        """The default floors are finite and positive."""
        policy = TrimTuningPolicy()
        assert policy.above_rated_trim_tol > 0.0
        assert policy.above_rated_damping > 0.0

    def test_rejects_trim_tol_above_cap(self) -> None:
        """A tuning floor looser than MAX_TRIM_TOL is not a periodicity test."""
        with pytest.raises(ValueError, match="must not exceed"):
            TrimTuningPolicy(above_rated_trim_tol=0.2)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_bad_trim_tol(self, bad: float) -> None:
        """A non-positive or non-finite tolerance floor is rejected."""
        with pytest.raises(ValueError, match="above_rated_trim_tol must be"):
            TrimTuningPolicy(above_rated_trim_tol=bad)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_bad_damping(self, bad: float) -> None:
        """A non-positive or non-finite damping floor is rejected."""
        with pytest.raises(ValueError, match="above_rated_damping must be"):
            TrimTuningPolicy(above_rated_damping=bad)


class TestTuneTrimSettings:
    """The per-operating-point trim tuner."""

    def test_below_rated_is_unchanged(self) -> None:
        """A below-rated point keeps the campaign-base trim settings."""
        base = LinearizationSettings(rated_wind_speed=11.4)
        tuned = tune_trim_settings(BELOW, base, TrimTuningPolicy())
        assert tuned.trim_tol == base.trim_tol
        assert tuned.twr_kdmp == base.twr_kdmp
        assert tuned.bld_kdmp == base.bld_kdmp

    def test_above_rated_raises_to_floors(self) -> None:
        """An above-rated point starts from the policy's tolerance/damping floors."""
        base = LinearizationSettings(
            rated_wind_speed=11.4
        )  # trim_tol 1e-3, twr/bld_kdmp 0
        policy = TrimTuningPolicy(above_rated_trim_tol=1e-2, above_rated_damping=5000.0)
        tuned = tune_trim_settings(ABOVE, base, policy)
        assert tuned.trim_tol == pytest.approx(1e-2)
        assert tuned.twr_kdmp == pytest.approx(5000.0)
        assert tuned.bld_kdmp == pytest.approx(5000.0)

    def test_at_rated_is_treated_as_above(self) -> None:
        """Exactly at rated wind uses the above-rated floors (matches trim split)."""
        base = LinearizationSettings(rated_wind_speed=11.4)
        tuned = tune_trim_settings(AT_RATED, base, TrimTuningPolicy())
        assert tuned.trim_tol == pytest.approx(1e-2)
        assert tuned.twr_kdmp == pytest.approx(5000.0)

    def test_never_tightens_below_the_base(self) -> None:
        """A base already looser/stiffer than the floors is kept (tuner only raises)."""
        base = LinearizationSettings(
            rated_wind_speed=11.4, trim_tol=0.05, twr_kdmp=8000.0, bld_kdmp=9000.0
        )
        tuned = tune_trim_settings(ABOVE, base, TrimTuningPolicy())
        assert tuned.trim_tol == pytest.approx(0.05)
        assert tuned.twr_kdmp == pytest.approx(8000.0)
        assert tuned.bld_kdmp == pytest.approx(9000.0)

    def test_non_rotating_point_is_unchanged(self) -> None:
        """A parked point (rotor stopped) takes a single lin, so no trim is tuned."""
        parked = OperatingPoint(
            wind_speed=16.0, rotor_speed_rpm=0.0, blade_pitch_deg=90.0
        )
        base = LinearizationSettings(rated_wind_speed=11.4)
        tuned = tune_trim_settings(parked, base, TrimTuningPolicy())
        assert tuned.trim_tol == base.trim_tol
        assert tuned.twr_kdmp == base.twr_kdmp

    def test_calc_steady_disabled_is_unchanged(self) -> None:
        """With CalcSteady off there is no trim to tune, even above rated."""
        base = LinearizationSettings(rated_wind_speed=11.4, calc_steady=False)
        tuned = tune_trim_settings(ABOVE, base, TrimTuningPolicy())
        assert tuned.trim_tol == base.trim_tol
        assert tuned.twr_kdmp == base.twr_kdmp

    def test_only_trim_fields_change(self) -> None:
        """Tuning leaves every non-trim field identical to the base."""
        base = LinearizationSettings(n_lin_times=24, rated_wind_speed=10.0)
        tuned = tune_trim_settings(ABOVE, base, TrimTuningPolicy())
        assert tuned.n_lin_times == base.n_lin_times
        assert tuned.rated_wind_speed == base.rated_wind_speed
        assert tuned.trim_gain_torque == base.trim_gain_torque
        assert tuned.trim_gain_pitch == base.trim_gain_pitch
        assert tuned.calc_steady == base.calc_steady

    def test_returns_a_copy(self) -> None:
        """The tuner returns a fresh settings object, not the base instance."""
        base = LinearizationSettings(rated_wind_speed=11.4)
        assert tune_trim_settings(BELOW, base, TrimTuningPolicy()) is not base
        assert tune_trim_settings(ABOVE, base, TrimTuningPolicy()) is not base


def _make_driver(tmp_path: Path, **kwargs: object) -> LinearizationDriver:
    """Build a driver over a throwaway template tree (no OpenFAST is launched)."""
    template_root = tmp_path / "tmpl"
    template_root.mkdir()
    fst = template_root / "main.fst"
    fst.touch()
    kwargs.setdefault("settings", LinearizationSettings(rated_wind_speed=11.4))
    return LinearizationDriver(
        fst_path=fst,
        output_dir=tmp_path / "out",
        template_root=template_root,
        **kwargs,  # type: ignore[arg-type]
    )


class TestDriverWiring:
    """The driver's per-operating-point starting settings."""

    def test_no_tuning_returns_campaign_base(self, tmp_path: Path) -> None:
        """Without a policy the starting settings are the campaign base itself."""
        driver = _make_driver(tmp_path)
        assert driver._op_base_settings(ABOVE) is driver.settings

    def test_tuning_leaves_below_rated_at_base(self, tmp_path: Path) -> None:
        """With a policy, a below-rated point still starts from the base values."""
        driver = _make_driver(tmp_path, trim_tuning=TrimTuningPolicy())
        base = driver._op_base_settings(BELOW)
        assert base.trim_tol == driver.settings.trim_tol
        assert base.twr_kdmp == driver.settings.twr_kdmp

    def test_tuning_applies_floors_above_rated(self, tmp_path: Path) -> None:
        """With a policy, an above-rated point starts from the tuned floors."""
        policy = TrimTuningPolicy(above_rated_trim_tol=1e-2, above_rated_damping=5000.0)
        driver = _make_driver(tmp_path, trim_tuning=policy)
        base = driver._op_base_settings(ABOVE)
        assert base.trim_tol == pytest.approx(1e-2)
        assert base.twr_kdmp == pytest.approx(5000.0)
        assert base.bld_kdmp == pytest.approx(5000.0)


class TestProvenanceCompatibility:
    """Per-OP tuning must not break the sweep's single-base provenance."""

    def test_tuning_only_touches_retry_escalated_fields(self) -> None:
        """Tuning changes only fields provenance treats per case (and changes some)."""
        base = LinearizationSettings(rated_wind_speed=11.4)
        tuned = tune_trim_settings(ABOVE, base, TrimTuningPolicy())
        base_d, tuned_d = asdict(base), asdict(tuned)
        differing = {key for key in base_d if base_d[key] != tuned_d[key]}
        assert differing  # the above-rated point really was tuned
        assert differing <= set(_RETRY_ESCALATED_FIELDS)

    def test_provenance_records_a_single_base_for_tuned_cases(
        self, tmp_path: Path
    ) -> None:
        """A mix of a base below-rated and a tuned above-rated case is recordable."""
        policy = TrimTuningPolicy()
        driver = _make_driver(tmp_path, trim_tuning=policy)
        cases = [
            self._case(driver, BELOW),  # tuner leaves this at the campaign base
            self._case(driver, ABOVE),  # tuner raises trim_tol/damping
        ]
        # Must not raise: tuning only moved the per-case retry-escalated fields, so the
        # campaign base (driver.settings) still describes every case.
        prov = driver.provenance(cases)
        assert prov.settings["trim_tol"] == driver.settings.trim_tol
        assert prov.cases[0].trim_tol == pytest.approx(driver.settings.trim_tol)
        assert prov.cases[1].trim_tol == pytest.approx(policy.above_rated_trim_tol)
        assert prov.cases[1].twr_kdmp == pytest.approx(policy.above_rated_damping)

    @staticmethod
    def _case(driver: LinearizationDriver, op: OperatingPoint) -> SweepCase:
        """Build a SweepCase as run_case would: shared base, per-OP as-run settings."""
        as_run = driver._op_base_settings(op)
        deck = driver.case_dir(op) / "main.fst"
        return SweepCase(
            operating_point=op,
            case_dir=driver.case_dir(op),
            fst_path=deck,
            result=OpenFastResult(fst_path=deck, returncode=0),
            settings=as_run.snapshot(),
            base_settings=driver.settings.snapshot(),
            executable=str(driver.executable),
        )
