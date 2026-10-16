# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.pipeline — the high-level modal pipeline."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from vane.io.lin_reader import LinFile, read_lin_file
from vane.pipeline import ModalPipeline

if TYPE_CHECKING:
    from tests.conftest import LinFileFactory

_STATES: list[tuple[str, bool, int]] = [
    ("ED 1st tower fore-aft (internal DOF index = DOF_TFA1), m", False, 2),
    (
        "ED First time derivative of 1st tower fore-aft "
        "(internal DOF index = DOF_TFA1), m/s",
        False,
        1,
    ),
]
_A = np.array([[0.0, 1.0], [-4.0, -0.2]], dtype=np.float64)


def _operating_point(
    make_lin_file: LinFileFactory, *, rotor_speed: float = 0.0, wind_speed: float = 0.0
) -> list[LinFile]:
    """Build a one-azimuth operating point at a given rotor or wind speed."""
    path = make_lin_file(
        states=_STATES, a=_A, rotor_speed=rotor_speed, wind_speed=wind_speed
    )
    return [read_lin_file(path)]


class TestModalPipeline:
    """Tests for the end-to-end pipeline."""

    def test_runs_end_to_end(self, make_lin_file: LinFileFactory) -> None:
        """The pipeline produces solutions, tracks, and a Campbell diagram."""
        operating_points = [
            _operating_point(make_lin_file, rotor_speed=1.0),
            _operating_point(make_lin_file, rotor_speed=1.5),
        ]
        result = ModalPipeline().run(operating_points)
        assert len(result.solutions) == 2
        assert len(result.mbc_results) == 2
        assert result.parameter_name == "rotor_speed_rpm"
        assert result.campbell.parameter_values.shape == (2,)
        assert (
            result.campbell.parameter_values[0] != result.campbell.parameter_values[1]
        )
        assert len(result.tracks) >= 1

    def test_state_space_export(self, make_lin_file: LinFileFactory) -> None:
        """The result exports a state-space system at an operating point."""
        result = ModalPipeline().run([_operating_point(make_lin_file, rotor_speed=1.0)])
        system = result.state_space(0)
        assert system.n_states == 2
        assert system.state_names

    def test_wind_speed_parameter(self, make_lin_file: LinFileFactory) -> None:
        """A wind-speed run records wind values and skips resonance detection."""
        operating_points = [
            _operating_point(make_lin_file, wind_speed=8.0),
            _operating_point(make_lin_file, wind_speed=10.0),
        ]
        result = ModalPipeline().run(operating_points, parameter_name="wind_speed")
        assert result.parameter_name == "wind_speed"
        assert result.resonances == []
        np.testing.assert_allclose(result.campbell.parameter_values, [8.0, 10.0])

    def test_operating_points_sorted_by_parameter(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Operating points given out of order are sorted by the parameter."""
        operating_points = [
            _operating_point(make_lin_file, rotor_speed=1.5),
            _operating_point(make_lin_file, rotor_speed=1.0),
        ]
        result = ModalPipeline().run(operating_points)
        values = result.campbell.parameter_values
        assert values[0] < values[1]

    def test_state_space_out_of_range_raises(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """A negative or too-large operating-point index raises IndexError."""
        result = ModalPipeline().run([_operating_point(make_lin_file, rotor_speed=1.0)])
        for bad_index in (-1, 5):
            with pytest.raises(IndexError, match="out of range"):
                result.state_space(bad_index)

    @pytest.mark.parametrize(
        ("frequency_weight", "mac_threshold"),
        [(1.5, 0.5), (0.5, -0.1), (float("nan"), 0.5)],
    )
    def test_invalid_weights_raise(
        self, frequency_weight: float, mac_threshold: float
    ) -> None:
        """Out-of-range or NaN tuning weights are rejected at construction."""
        with pytest.raises(ValueError, match="must be in"):
            ModalPipeline(
                frequency_weight=frequency_weight, mac_threshold=mac_threshold
            )

    def test_unknown_tracking_strategy_raises(self) -> None:
        """An unrecognized tracking strategy is rejected at construction."""
        with pytest.raises(ValueError, match="tracking_strategy"):
            ModalPipeline(tracking_strategy="greedy")  # type: ignore[arg-type]

    def test_simultaneous_strategy_runs_end_to_end(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """The simultaneous tracking strategy threads through to the result."""
        operating_points = [
            _operating_point(make_lin_file, rotor_speed=1.0),
            _operating_point(make_lin_file, rotor_speed=1.5),
        ]
        pipeline = ModalPipeline(tracking_strategy="simultaneous")
        assert pipeline.tracking_strategy == "simultaneous"
        result = pipeline.run(operating_points)
        assert len(result.tracks) >= 1

    def test_branch_review_flag_threads_through(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """The branch_review flag reaches identification and yields a group list."""
        operating_points = [
            _operating_point(make_lin_file, rotor_speed=1.0),
            _operating_point(make_lin_file, rotor_speed=1.5),
        ]
        pipeline = ModalPipeline(branch_review=True)
        assert pipeline.branch_review is True
        result = pipeline.run(operating_points)
        assert isinstance(result.identification.branch_groups, list)

    def test_resolve_implies_branch_review(self, make_lin_file: LinFileFactory) -> None:
        """The resolve flag turns on review and threads through to identification."""
        operating_points = [
            _operating_point(make_lin_file, rotor_speed=1.0),
            _operating_point(make_lin_file, rotor_speed=1.5),
        ]
        pipeline = ModalPipeline(resolve=True)
        assert pipeline.resolve is True
        assert pipeline.branch_review is True
        result = pipeline.run(operating_points)
        assert isinstance(result.identification.branch_groups, list)

    def test_empty_operating_points_raises(self) -> None:
        """An empty operating-point list raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            ModalPipeline().run([])

    def test_unknown_parameter_raises(self, make_lin_file: LinFileFactory) -> None:
        """An unknown parameter name raises ValueError."""
        with pytest.raises(ValueError, match="parameter_name"):
            ModalPipeline().run(
                [_operating_point(make_lin_file, rotor_speed=1.0)],
                parameter_name="bogus",
            )

    def test_duplicate_operating_parameter_raises(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Two operating points at the same parameter value are rejected."""
        operating_points = [
            _operating_point(make_lin_file, rotor_speed=1.0),
            _operating_point(make_lin_file, rotor_speed=1.0),
        ]
        with pytest.raises(ValueError, match="distinct"):
            ModalPipeline().run(operating_points)

    @pytest.mark.parametrize(
        ("harmonics", "match"),
        [([0], "positive integers"), ([1.5], "positive integers"), ([3, 3], "unique")],
    )
    def test_invalid_harmonics_rejected(self, harmonics: list[int], match: str) -> None:
        """Non-positive, fractional, or duplicate harmonics are rejected early."""
        with pytest.raises(ValueError, match=match):
            ModalPipeline(harmonics=harmonics)


# Above rated the trimmed rotor speed is constant: nominally 12.1 rpm at every wind
# speed, differing only by round-off, while the wind speed rises monotonically.
_RATED_RAD_S = 12.1 * np.pi / 30.0
_PLATEAU_WINDS = (12.0, 14.0, 16.0, 18.0)
_PLATEAU_ROUND_OFF = (1e-9, -2e-9, 3e-9, -1e-9)


def _plateau_sweep(make_lin_file: LinFileFactory) -> list[list[LinFile]]:
    """Build an above-rated sweep: distinct winds, rotor speeds equal to round-off.

    The points are given out of wind order so that ordering them requires a
    meaningful sweep axis.
    """
    points = []
    for wind, round_off in zip(_PLATEAU_WINDS, _PLATEAU_ROUND_OFF, strict=True):
        (lin,) = _operating_point(make_lin_file, wind_speed=wind)
        points.append([dataclasses.replace(lin, rotor_speed=_RATED_RAD_S + round_off)])
    return [points[2], points[0], points[3], points[1]]


class TestSweepAxisPlateau:
    """A plateaued sweep axis is rejected with an actionable message."""

    def test_rotor_speed_plateau_is_rejected(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Round-off-distinct rotor speeds are a plateau, not a sortable axis."""
        with pytest.raises(ValueError, match="plateau") as excinfo:
            ModalPipeline().run(_plateau_sweep(make_lin_file))
        message = str(excinfo.value)
        assert "parameter_name='wind_speed'" in message
        assert "--parameter wind" in message
        assert "which is distinct across these operating points" in message

    def test_wind_axis_orders_a_plateau_sweep(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """The same above-rated sweep runs on the wind axis, in wind order."""
        result = ModalPipeline().run(
            _plateau_sweep(make_lin_file), parameter_name="wind_speed"
        )
        np.testing.assert_allclose(
            result.campbell.parameter_values, sorted(_PLATEAU_WINDS)
        )
        np.testing.assert_allclose(
            [mbc.wind_speed for mbc in result.mbc_results], sorted(_PLATEAU_WINDS)
        )

    def test_advice_notes_when_wind_is_not_distinct(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Without distinct winds, the hint says the wind axis would not help."""
        points = [
            [dataclasses.replace(lin, rotor_speed=_RATED_RAD_S + round_off)]
            for round_off in (0.0, 1e-9)
            for lin in _operating_point(make_lin_file, wind_speed=12.0)
        ]
        with pytest.raises(ValueError, match="these do not"):
            ModalPipeline().run(points)

    def test_wind_axis_plateau_is_rejected(self, make_lin_file: LinFileFactory) -> None:
        """The plateau check applies to the wind axis too, without the rpm hint."""
        points = [
            _operating_point(make_lin_file, wind_speed=10.0),
            [
                dataclasses.replace(lin, wind_speed=10.0 + 1e-9)
                for lin in _operating_point(make_lin_file, wind_speed=10.0)
            ],
        ]
        with pytest.raises(ValueError, match="plateau") as excinfo:
            ModalPipeline().run(points, parameter_name="wind_speed")
        assert "--parameter" not in str(excinfo.value)

    def test_duplicate_message_recommends_wind_axis(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Exactly equal rotor speeds with distinct winds point at the wind axis."""
        points = [
            _operating_point(make_lin_file, rotor_speed=1.0, wind_speed=12.0),
            _operating_point(make_lin_file, rotor_speed=1.0, wind_speed=14.0),
        ]
        with pytest.raises(ValueError, match="duplicates") as excinfo:
            ModalPipeline().run(points)
        assert "--parameter wind" in str(excinfo.value)

    def test_close_but_meaningful_steps_are_accepted(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """A 0.1 % rotor-speed step is well above the default plateau tolerance."""
        points = [
            _operating_point(make_lin_file, rotor_speed=1.0),
            _operating_point(make_lin_file, rotor_speed=1.001),
        ]
        result = ModalPipeline().run(points)
        assert result.campbell.parameter_values.shape == (2,)

    def test_zero_tolerance_restores_exact_distinctness(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """parameter_rtol=0 rejects only exact duplicates."""
        result = ModalPipeline().run(_plateau_sweep(make_lin_file), parameter_rtol=0.0)
        assert result.campbell.parameter_values.shape == (4,)

    def test_larger_tolerance_rejects_a_coarser_plateau(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """A user-raised tolerance rejects steps the default would accept."""
        points = [
            _operating_point(make_lin_file, rotor_speed=1.0),
            _operating_point(make_lin_file, rotor_speed=1.001),
        ]
        with pytest.raises(ValueError, match="plateau"):
            ModalPipeline().run(points, parameter_rtol=1e-2)

    @pytest.mark.parametrize("bad", [-1e-6, 1.0, float("nan"), float("inf"), True, "x"])
    def test_invalid_tolerance_rejected(
        self, make_lin_file: LinFileFactory, bad: object
    ) -> None:
        """A negative, too-large, non-finite, or non-numeric tolerance is rejected."""
        rtol: Any = bad
        with pytest.raises(ValueError, match="parameter_rtol"):
            ModalPipeline().run(
                [_operating_point(make_lin_file, rotor_speed=1.0)],
                parameter_rtol=rtol,
            )
