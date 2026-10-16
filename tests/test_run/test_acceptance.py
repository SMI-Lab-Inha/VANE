# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Physical operating-point acceptance tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vane.io.lin_reader import LinFile, OperatingPointTable
from vane.run import (
    LinearizationAcceptancePolicy,
    LinearizationDriver,
    LinearizationRetryPolicy,
    LinearizationSettings,
    OperatingPoint,
    OperatingPointResidualAudit,
    audit_operating_point_residual,
    audit_periodic_state_residual,
)
from vane.run.runner import OpenFastResult

_RATE_DESCRIPTIONS = [
    "ED First time derivative of Platform horizontal surge translation DOF, m/s",
    "ED First time derivative of Platform horizontal sway translation DOF, m/s",
    "ED First time derivative of Platform vertical heave translation DOF, m/s",
    "ED First time derivative of Platform roll tilt rotation DOF, rad/s",
    "ED First time derivative of Platform pitch tilt rotation DOF, rad/s",
    "ED First time derivative of Platform yaw rotation DOF, rad/s",
]


def _table(values: list[float], descriptions: list[str]) -> OperatingPointTable:
    """Return a minimal operating-point table."""
    count = len(values)
    return OperatingPointTable(
        values=np.asarray(values, dtype=np.float64),
        rotating_frame=np.zeros(count, dtype=np.bool_),
        derivative_order=np.zeros(count, dtype=np.int_),
        descriptions=descriptions,
    )


def _lin(azimuth: float, rates: list[float]) -> LinFile:
    """Return a synthetic rotating linearization carrying platform rates."""
    count = len(rates)
    empty = _table([], [])
    return LinFile(
        path=Path(f"case-{azimuth}.lin"),
        sim_time=azimuth,
        rotor_speed=1.0,
        azimuth=azimuth,
        wind_speed=12.0,
        jacobians_included=False,
        x=_table(rates, _RATE_DESCRIPTIONS[:count]),
        xdot=_table([0.0] * count, _RATE_DESCRIPTIONS[:count]),
        u=empty,
        y=empty,
        a=np.zeros((count, count)),
        b=None,
        c=None,
        d=None,
        jacobians={},
    )


def _derivative_lin(
    azimuth: float,
    derivatives: list[float],
    *,
    rotating: list[bool] | None = None,
) -> LinFile:
    """Return a synthetic file carrying structural derivative operating points."""
    descriptions = [
        "ED Platform surge translation DOF, m",
        "ED Variable speed generator DOF, rad",
        "ED First time derivative of Variable speed generator DOF, rad/s",
        "ED 1st flapwise bending-mode DOF of blade 1, m",
    ][: len(derivatives)]
    derivative_descriptions = [
        "First time derivative of ED Platform surge translation DOF, m/s",
        "First time derivative of ED Variable speed generator DOF, rad/s",
        "First time derivative of ED First time derivative of Variable speed "
        "generator DOF, rad/s/s",
        "First time derivative of ED 1st flapwise bending-mode DOF of blade 1, m/s",
    ][: len(derivatives)]
    count = len(derivatives)
    state_table = _table([0.0] * count, descriptions)
    state_table.rotating_frame = np.asarray(
        rotating if rotating is not None else [False] * count, dtype=np.bool_
    )
    derivative_table = _table(derivatives, derivative_descriptions)
    empty = _table([], [])
    return LinFile(
        path=Path(f"derivative-{azimuth}.lin"),
        sim_time=azimuth,
        rotor_speed=1.0,
        azimuth=azimuth,
        wind_speed=12.0,
        jacobians_included=False,
        x=state_table,
        xdot=derivative_table,
        u=empty,
        y=empty,
        a=np.zeros((count, count)),
        b=None,
        c=None,
        d=None,
        jacobians={},
    )


class TestOperatingPointResidualAudit:
    """Azimuth-mean platform-drift check."""

    def test_periodic_motion_cancels_without_false_rejection(self) -> None:
        """Large 1P rates are accepted when their azimuth mean is zero."""
        files = [
            _lin(
                azimuth,
                [
                    float(np.cos(azimuth)),
                    float(np.sin(azimuth)),
                    0.0,
                    0.01 * float(np.cos(azimuth)),
                    0.0,
                    0.0,
                ],
            )
            for azimuth in (2.0 * np.pi * index / 12.0 for index in range(12))
        ]

        audit = audit_operating_point_residual(files)

        assert audit.assessed is True
        assert audit.accepted is True
        assert audit.mean_translational_speed == pytest.approx(0.0, abs=1e-15)
        assert audit.mean_angular_speed == pytest.approx(0.0, abs=1e-15)
        assert audit.max_instantaneous_translational_speed == pytest.approx(1.0)

    def test_secular_drift_is_rejected(self) -> None:
        """A constant drift component survives averaging and fails the check."""
        files = [
            _lin(azimuth, [0.0, 0.002, 0.0, 0.0, 0.0, 0.0])
            for azimuth in (2.0 * np.pi * index / 12.0 for index in range(12))
        ]

        audit = audit_operating_point_residual(files)

        assert audit.assessed is True
        assert audit.accepted is False
        assert audit.mean_translational_speed == pytest.approx(0.002)
        assert audit.reason is not None
        assert "translational drift" in audit.reason

    def test_rotating_single_azimuth_is_not_accepted(self) -> None:
        """One instantaneous rate cannot establish a periodic mean."""
        audit = audit_operating_point_residual(
            [_lin(0.0, [0.0] * 6)],
            LinearizationAcceptancePolicy(min_rotating_azimuths=4),
        )

        assert audit.assessed is False
        assert audit.accepted is False
        assert audit.reason is not None
        assert "need at least 4" in audit.reason

    def test_unbalanced_azimuth_layout_is_not_assessed(self) -> None:
        """A dense partial-revolution sweep cannot masquerade as a periodic mean."""
        files = [_lin(2.0 * np.pi * index / 24.0, [0.0] * 6) for index in range(12)]

        audit = audit_operating_point_residual(files)

        assert audit.assessed is False
        assert audit.accepted is False
        assert audit.reason is not None
        assert "cannot separate the mean from the 1P through 3P harmonics" in (
            audit.reason
        )

    def test_quantised_calcsteady_azimuths_recover_known_mean(self) -> None:
        """Step-quantised azimuths still separate a secular mean from 1P--3P motion.

        CalcSteady snapshots land on the first time step at or past each target
        azimuth, so a real 36-point sweep is uniform only to within one step. The
        fitted mean must recover the drift exactly despite large periodic rates.
        """
        rng = np.random.default_rng(7)
        azimuths = 2.0 * np.pi * np.arange(36) / 36.0 + rng.uniform(0.0, 0.02, 36)
        drift = 4.0e-4
        files = [
            _lin(
                float(azimuth),
                [
                    drift
                    + 0.3 * float(np.cos(azimuth))
                    + 0.2 * float(np.sin(2.0 * azimuth))
                    + 0.1 * float(np.cos(3.0 * azimuth + 0.4)),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            )
            for azimuth in azimuths
        ]

        audit = audit_operating_point_residual(files)

        assert audit.assessed is True
        assert audit.accepted is True
        assert audit.mean_translational_velocity is not None
        assert audit.mean_translational_velocity[0] == pytest.approx(drift, rel=1e-9)
        assert audit.mean_translational_speed == pytest.approx(drift, rel=1e-9)

    def test_quantised_azimuths_still_reject_secular_drift(self) -> None:
        """A drift above the limit is rejected on a non-uniform sweep too."""
        azimuths = 2.0 * np.pi * np.arange(36) / 36.0 + np.linspace(0.0, 0.015, 36)
        files = [
            _lin(float(azimuth), [0.0, 0.0, 0.0, 0.0, 0.0, 2.0e-4 + np.sin(azimuth)])
            for azimuth in azimuths
        ]

        audit = audit_operating_point_residual(files)

        assert audit.assessed is True
        assert audit.accepted is False
        assert audit.mean_angular_speed == pytest.approx(2.0e-4, rel=1e-9)

    def test_fixed_model_without_platform_states_is_not_applicable(self) -> None:
        """A fixed-bottom model is accepted without claiming an assessment."""
        item = _lin(0.0, [])
        item.rotor_speed = 0.0

        audit = audit_operating_point_residual([item])

        assert audit.assessed is False
        assert audit.accepted is True
        assert audit.reason is None

    @pytest.mark.parametrize(
        ("keyword", "value"),
        [
            ("max_mean_translational_speed", 0.0),
            ("max_mean_angular_speed", float("nan")),
            ("min_rotating_azimuths", 3),
        ],
    )
    def test_policy_rejects_invalid_limits(self, keyword: str, value: float) -> None:
        """Physical limits must be usable."""
        overrides: dict[str, Any] = {keyword: value}
        with pytest.raises(ValueError):
            LinearizationAcceptancePolicy(**overrides)


class TestPeriodicStateResidualAudit:
    """Unit-preserving derivative residual audit."""

    def test_reports_periodic_and_secular_derivatives_without_unit_mixing(self) -> None:
        """The scale-free ratio separates a periodic derivative from constant drift."""
        files = [
            _derivative_lin(2.0 * np.pi * index / 12.0, [value, 1.0, 0.002])
            for index, value in enumerate(np.cos(2.0 * np.pi * np.arange(12) / 12.0))
        ]

        audit = audit_periodic_state_residual(files)

        assert audit.assessed is True
        assert len(audit.channels) == 2
        periodic, acceleration = audit.channels
        assert periodic.unit == "m/s"
        assert periodic.mean_derivative == pytest.approx(0.0, abs=1e-15)
        assert periodic.mean_to_rms_ratio == pytest.approx(0.0, abs=1e-15)
        assert acceleration.unit == "rad/s/s"
        assert acceleration.mean_derivative == pytest.approx(0.002)
        assert acceleration.mean_to_rms_ratio == pytest.approx(1.0)
        assert audit.max_mean_to_rms_ratio == pytest.approx(1.0)

    def test_excludes_advancing_azimuth_and_rotating_blade_state(self) -> None:
        """Intended coordinate transport is not reported as disequilibrium."""
        files = [
            _derivative_lin(
                2.0 * np.pi * index / 12.0,
                [0.0, 0.8, 0.0, 0.2],
                rotating=[False, False, False, True],
            )
            for index in range(12)
        ]

        audit = audit_periodic_state_residual(files)

        assert len(audit.channels) == 2
        assert len(audit.excluded_cyclic_channels) == 1
        assert "generator DOF" in audit.excluded_cyclic_channels[0]
        assert len(audit.excluded_rotating_channels) == 1
        assert "blade 1" in audit.excluded_rotating_channels[0]

    def test_non_uniform_sweep_uses_harmonic_mean(self) -> None:
        """Quantised azimuths are assessed and the fitted mean removes 1P--3P."""
        azimuths = 2.0 * np.pi * np.arange(36) / 36.0 + np.linspace(0.0, 0.02, 36)
        files = [
            _derivative_lin(
                float(azimuth),
                [0.001 + float(np.cos(azimuth)) + 0.5 * float(np.sin(3.0 * azimuth))],
            )
            for azimuth in azimuths
        ]

        audit = audit_periodic_state_residual(files)

        assert audit.assessed is True
        (channel,) = audit.channels
        assert channel.mean_derivative == pytest.approx(0.001, rel=1e-9)

    def test_three_azimuths_are_not_accepted_as_periodic_by_default(self) -> None:
        """A 3P component can alias into a three-sample azimuth mean."""
        files = [
            _derivative_lin(2.0 * np.pi * index / 3.0, [0.001]) for index in range(3)
        ]

        audit = audit_periodic_state_residual(files)

        assert audit.assessed is False
        assert audit.reason is not None
        assert "need at least 12" in audit.reason


def _driver_template(tmp_path: Path) -> Path:
    """Write the minimum rotating driver template used by the check wiring test."""
    root = tmp_path / "template"
    root.mkdir()
    (root / "ED.dat").write_text(
        "7.5 RotSpeed - rpm\n"
        "6.0 BlPitch(1) - deg\n"
        "6.0 BlPitch(2) - deg\n"
        "6.0 BlPitch(3) - deg\n",
        encoding="utf-8",
    )
    (root / "IW.dat").write_text(
        "1 WindType - switch\n12 HWindSpeed - m/s\n0 PLExp - -\n",
        encoding="utf-8",
    )
    fst = root / "Main.fst"
    fst.write_text(
        "False Linearize - flag\n"
        "False CalcSteady - flag\n"
        "3 NLinTimes - -\n"
        "0 0 0 LinTimes - s\n"
        "3 TrimCase - -\n"
        "0.01 TrimGain - -\n"
        "0.001 TrimTol - -\n"
        "1 CompInflow - flag\n"
        '"ED.dat" EDFile - -\n'
        '"IW.dat" InflowFile - -\n',
        encoding="utf-8",
    )
    return fst


def test_driver_records_physical_rejection_without_loose_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Residual failure stops rather than entering the TrimTol-loosening ladder."""
    from vane.run import driver as driver_module

    fst = _driver_template(tmp_path)
    driver = LinearizationDriver(
        fst_path=fst,
        output_dir=tmp_path / "out",
        settings=LinearizationSettings(rated_wind_speed=11.4, n_lin_times=3),
        retry_policy=LinearizationRetryPolicy(max_attempts=3),
        acceptance_policy=LinearizationAcceptancePolicy(),
        validate=False,
    )
    calls: list[Path] = []

    def fake_run(deck_path: Path, **_: object) -> OpenFastResult:
        calls.append(deck_path)
        return OpenFastResult(
            fst_path=deck_path,
            returncode=0,
            lin_files=[deck_path.with_suffix(f".{index}.lin") for index in range(1, 4)],
        )

    rejected = OperatingPointResidualAudit(
        assessed=True,
        accepted=False,
        reason="mean platform translational drift exceeds check",
        file_count=3,
        distinct_azimuths=3,
        rotating=True,
        translational_channels=("surge", "sway", "heave"),
        angular_channels=("roll", "pitch", "yaw"),
        mean_translational_velocity=(0.0, 0.002, 0.0),
        mean_angular_velocity=(0.0, 0.0, 0.0),
        mean_translational_speed=0.002,
        mean_angular_speed=0.0,
        max_instantaneous_translational_speed=0.002,
        max_instantaneous_angular_speed=0.0,
        max_mean_translational_speed=0.001,
        max_mean_angular_speed=0.0001,
    )
    monkeypatch.setattr(driver_module, "run_openfast", fake_run)
    monkeypatch.setattr(
        driver_module, "audit_operating_point_residual", lambda *_: rejected
    )

    case = driver.run_case(OperatingPoint(12.0, 7.5, 6.0, name="ws12"))

    assert len(calls) == 1
    assert case.attempts == 1
    assert case.converged is False
    assert case.operating_point_residual is rejected
    assert driver.provenance([case]).cases[0].operating_point_residual is rejected
