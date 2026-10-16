# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.sysid.physical_lpv -- named physical-state LPV plant export."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from vane.export import LPVFamily, build_physical_lpv_family
from vane.sysid.physical_lpv import _stack_mean

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

    from tests.conftest import LinFileFactory

# (description, rotating_frame, derivative_order) channel specs.
_STATES = [
    ("ED PtfmPitch, (deg)", False, 2),
    ("ED PtfmPitch rate, (deg/s)", False, 2),
    ("ED RotSpeed, (rpm)", False, 2),
]
_INPUTS = [("GenTq, (Nm)", False, 0), ("BldPitch1, (rad)", True, 0)]
_OUTPUTS = [("RotSpeed, (rpm)", False, 0), ("BldPitch1, (rad)", True, 0)]

# A second-order layout: platform pitch, its "First time derivative of" twin, and an
# independent generator DOF with its own rate twin.
_POS = "ED Platform pitch tilt rotation DOF (DOF_P)"
_VEL = "ED First time derivative of Platform pitch tilt rotation DOF (DOF_P)"
_GEN = "ED Variable speed generator DOF (DOF_GeAz)"
_GEN_RATE = "ED First time derivative of Variable speed generator DOF (DOF_GeAz)"
_PAIRED_STATES = [
    (f"{_POS}, rad", False, 2),
    (f"{_VEL}, rad/s", False, 2),
    ("ED RotSpeed, rpm", False, 2),
]
_FOUR_STATES = [
    (f"{_POS}, rad", False, 2),
    (f"{_GEN}, rad", False, 2),
    (f"{_VEL}, rad/s", False, 2),
    (f"{_GEN_RATE}, rad/s", False, 2),
]
# Two decoupled second-order DOFs: d(pos)/dt = vel, d(vel)/dt = -k pos - c vel.
_A4 = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [-0.5, 0.0, -0.1, 0.0],
        [0.0, 0.0, 0.0, -0.2],
    ],
    dtype=np.float64,
)

_A = np.arange(9, dtype=np.float64).reshape(3, 3)
_B = np.arange(6, dtype=np.float64).reshape(3, 2)
_C = np.arange(6, dtype=np.float64).reshape(2, 3)
_D = np.arange(4, dtype=np.float64).reshape(2, 2)


def _write_point(
    make_lin_file: LinFileFactory, *, wind_speed: float, scale: float = 1.0
) -> Path:
    """Write one single-azimuth operating point and return its path."""
    return make_lin_file(
        states=_STATES,
        inputs=_INPUTS,
        outputs=_OUTPUTS,
        a=_A * scale,
        b=_B * scale,
        c=_C * scale,
        d=_D * scale,
        wind_speed=wind_speed,
        rotor_speed=0.5 + 0.05 * wind_speed,
    )


class TestBuildPhysicalLpvFamily:
    """The public physical-state LPV exporter."""

    def test_builds_named_family_sorted_by_wind_speed(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Three operating points become a wind-speed-sorted named family."""
        lin_dir = _write_point(make_lin_file, wind_speed=9.0, scale=2.0).parent
        _write_point(make_lin_file, wind_speed=5.0, scale=1.0)
        _write_point(make_lin_file, wind_speed=13.0, scale=3.0)
        family = build_physical_lpv_family(
            lin_dir,
            states=["ED PtfmPitch", "ED PtfmPitch rate", "ED RotSpeed"],
            inputs=["GenTq", "BldPitch1"],
            outputs=["RotSpeed", "BldPitch1"],
        )
        assert isinstance(family, LPVFamily)
        np.testing.assert_array_equal(family.scheduling_points, [5.0, 9.0, 13.0])
        assert len(family.systems) == 3
        assert family.scheduling_variable == "WindSpeed"
        assert family.scheduling_unit == "m/s"
        assert family.state_names == [
            "ED PtfmPitch",
            "ED PtfmPitch rate",
            "ED RotSpeed",
        ]
        # Sorted ascending: scale 1 at 5 m/s, scale 3 at 13 m/s.
        np.testing.assert_allclose(family.systems[0].a, _A)
        np.testing.assert_allclose(family.systems[-1].a, _A * 3.0)
        assert family.systems[0].state_names == family.state_names
        assert family.systems[0].input_names == ["GenTq", "BldPitch1"]
        assert family.systems[0].output_names == ["RotSpeed", "BldPitch1"]

    def test_selects_and_reorders_a_subset(self, make_lin_file: LinFileFactory) -> None:
        """Selecting a reordered subset slices A/B/C/D by name accordingly."""
        lin_dir = _write_point(make_lin_file, wind_speed=8.0).parent
        family = build_physical_lpv_family(
            lin_dir,
            states=["ED RotSpeed", "ED PtfmPitch"],  # indices [2, 0]
            inputs=["GenTq"],  # index [0]
            outputs=["RotSpeed"],  # index [0]
            max_dropped_coupling=None,  # a deliberate truncation (drops the rate)
        )
        system = family.systems[0]
        # A[ix_([2,0],[2,0])] = [[8, 6], [2, 0]]
        np.testing.assert_allclose(system.a, [[8.0, 6.0], [2.0, 0.0]])
        assert system.b is not None
        np.testing.assert_allclose(system.b, _B[np.ix_([2, 0], [0])])
        assert system.c is not None
        np.testing.assert_allclose(system.c, _C[np.ix_([0], [2, 0])])
        assert system.state_names == ["ED RotSpeed", "ED PtfmPitch"]

    def test_recurses_nested_lin_directory_tree(self, tmp_path: Path) -> None:
        """A campaign laid out one sub-directory per point is read recursively."""
        from tests.conftest import build_lin_text

        root = tmp_path / "campaign"
        for folder, wind, scale in (("ws5", 5.0, 1.0), ("ws9", 9.0, 2.0)):
            point_dir = root / folder
            point_dir.mkdir(parents=True)
            text = build_lin_text(
                states=_STATES,
                inputs=_INPUTS,
                outputs=_OUTPUTS,
                a=_A * scale,
                b=_B * scale,
                c=_C * scale,
                d=_D * scale,
                wind_speed=wind,
                rotor_speed=0.5 + 0.05 * wind,
            )
            (point_dir / "case.1.lin").write_text(text, encoding="utf-8")

        family = build_physical_lpv_family(
            root, states=["ED PtfmPitch", "ED RotSpeed"], max_dropped_coupling=None
        )

        np.testing.assert_array_equal(family.scheduling_points, [5.0, 9.0])
        assert len(family.systems) == 2

    def test_substring_match_resolves_a_unique_channel(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """A unique substring (e.g. 'RotSpeed') matches the full description."""
        lin_dir = _write_point(make_lin_file, wind_speed=8.0).parent
        family = build_physical_lpv_family(
            lin_dir, states=["RotSpeed"], max_dropped_coupling=None
        )
        assert family.systems[0].a.shape == (1, 1)
        np.testing.assert_allclose(family.systems[0].a, [[_A[2, 2]]])

    def test_rotor_speed_scheduling(self, make_lin_file: LinFileFactory) -> None:
        """The family can be scheduled on rotor speed instead of wind speed."""
        lin_dir = _write_point(make_lin_file, wind_speed=5.0).parent
        _write_point(make_lin_file, wind_speed=12.0)
        family = build_physical_lpv_family(
            lin_dir,
            states=["ED PtfmPitch"],
            scheduling_var="RotorSpeed",
            max_dropped_coupling=None,
        )
        assert family.scheduling_variable == "RotorSpeed"
        # The unit is part of the contract: rotor speed is in rad/s (the raw .lin unit),
        # NOT rpm — a scheduled controller must be able to check this, not guess it.
        assert family.scheduling_unit == "rad/s"
        # rotor_speed = 0.5 + 0.05*wind rad/s, so wind 5.0 -> 0.75, wind 12.0 -> 1.1.
        np.testing.assert_allclose(family.scheduling_points, [0.75, 1.10], atol=1e-9)

    def test_missing_name_raises(self, make_lin_file: LinFileFactory) -> None:
        """A requested name absent from the linearization is rejected."""
        lin_dir = _write_point(make_lin_file, wind_speed=8.0).parent
        with pytest.raises(ValueError, match="was not found"):
            build_physical_lpv_family(lin_dir, states=["ED Nonexistent"])

    def test_ambiguous_name_raises(self, make_lin_file: LinFileFactory) -> None:
        """A name matching several channels is rejected as ambiguous."""
        lin_dir = _write_point(make_lin_file, wind_speed=8.0).parent
        # 'PtfmPitch' is a substring of both 'ED PtfmPitch' and 'ED PtfmPitch rate'.
        with pytest.raises(ValueError, match="matches multiple channels"):
            build_physical_lpv_family(lin_dir, states=["PtfmPitch"])

    def test_ambiguous_name_lists_the_candidates(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """The ambiguity error names the channels the request matched."""
        lin_dir = _write_point(make_lin_file, wind_speed=8.0).parent
        with pytest.raises(ValueError, match=r"ED PtfmPitch.*ED PtfmPitch rate"):
            build_physical_lpv_family(lin_dir, states=["PtfmPitch"])

    def test_dof_disambiguated_from_its_first_time_derivative(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Without pairing, a name matching a DOF and its rate selects the position."""
        # A real second-order plant lists both the DOF and its "First time derivative
        # of <DOF>"; a request for the bare DOF must resolve to the position channel.
        a = np.eye(3, dtype=np.float64)
        lin_dir = make_lin_file(
            states=_PAIRED_STATES, a=a, wind_speed=8.0, rotor_speed=1.0
        ).parent
        family = build_physical_lpv_family(
            lin_dir,
            states=["Platform pitch tilt rotation DOF (DOF_P)"],
            pair_derivatives=False,
        )
        # Resolved to row 0 (the position DOF), not row 1 (its time derivative).
        assert family.state_names == ["Platform pitch tilt rotation DOF (DOF_P)"]
        assert family.auto_added_states == []
        np.testing.assert_allclose(family.systems[0].a, [[a[0, 0]]])

    def test_explicit_derivative_request_still_resolves(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Asking for the derivative explicitly selects the derivative channel."""
        a = np.array([[2.0, 0.0], [0.0, 5.0]], dtype=np.float64)
        lin_dir = make_lin_file(
            states=_PAIRED_STATES[:2], a=a, wind_speed=8.0, rotor_speed=1.0
        ).parent
        family = build_physical_lpv_family(
            lin_dir,
            states=["First time derivative of Platform pitch tilt rotation DOF"],
            pair_derivatives=False,
        )
        np.testing.assert_allclose(family.systems[0].a, [[5.0]])

    def test_unknown_scheduling_var_raises(self, make_lin_file: LinFileFactory) -> None:
        """An unsupported scheduling variable is rejected."""
        lin_dir = _write_point(make_lin_file, wind_speed=8.0).parent
        with pytest.raises(ValueError, match="scheduling_var must be one of"):
            build_physical_lpv_family(
                lin_dir, states=["ED PtfmPitch"], scheduling_var="Azimuth"
            )

    def test_empty_states_raises(self, make_lin_file: LinFileFactory) -> None:
        """At least one state must be selected."""
        lin_dir = _write_point(make_lin_file, wind_speed=8.0).parent
        with pytest.raises(ValueError, match="at least one channel"):
            build_physical_lpv_family(lin_dir, states=[])

    def test_inconsistent_azimuth_layout_raises(self, tmp_path: Path) -> None:
        """Same-OP azimuth files with a different channel order are rejected."""
        from tests.conftest import build_lin_text

        swapped = [_STATES[1], _STATES[0], _STATES[2]]  # first two states reordered
        (tmp_path / "mixed.1.lin").write_text(
            build_lin_text(states=_STATES, a=_A, wind_speed=8.0, azimuth=0.0),
            encoding="utf-8",
        )
        (tmp_path / "mixed.2.lin").write_text(
            build_lin_text(states=swapped, a=_A, wind_speed=8.0, azimuth=1.0),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="different channel layouts"):
            build_physical_lpv_family(tmp_path, states=["ED PtfmPitch"])

    def test_mixed_operating_condition_raises(self, tmp_path: Path) -> None:
        """Same-case files from different wind speeds are rejected, not averaged."""
        from tests.conftest import build_lin_text

        (tmp_path / "mix.1.lin").write_text(
            build_lin_text(states=_STATES, a=_A, wind_speed=8.0), encoding="utf-8"
        )
        (tmp_path / "mix.2.lin").write_text(
            build_lin_text(states=_STATES, a=_A, wind_speed=10.0), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="disagree on WindSpeed"):
            build_physical_lpv_family(tmp_path, states=["ED PtfmPitch"])

    def test_name_resolving_to_different_channels_across_points_raises(
        self, tmp_path: Path
    ) -> None:
        """A name resolving to different channels/units per point is rejected."""
        from tests.conftest import build_lin_text

        rpm = [("ED RotSpeed, (rpm)", False, 2)]
        rad = [("ED RotSpeed, (rad/s)", False, 2)]  # same name, different units
        (tmp_path / "p1.1.lin").write_text(
            build_lin_text(states=rpm, a=np.array([[1.0]]), wind_speed=8.0),
            encoding="utf-8",
        )
        (tmp_path / "p2.1.lin").write_text(
            build_lin_text(states=rad, a=np.array([[2.0]]), wind_speed=10.0),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="resolve to different"):
            build_physical_lpv_family(tmp_path, states=["ED RotSpeed"])

    def test_duplicate_azimuth_raises(self, tmp_path: Path) -> None:
        """A duplicated azimuth file would double-count in the mean and is rejected."""
        from tests.conftest import build_lin_text

        for index in (1, 2):  # two files at the SAME azimuth (a duplicate)
            (tmp_path / f"dup.{index}.lin").write_text(
                build_lin_text(states=_STATES, a=_A, wind_speed=8.0, azimuth=0.5),
                encoding="utf-8",
            )
        with pytest.raises(ValueError, match="duplicate azimuth"):
            build_physical_lpv_family(tmp_path, states=["ED PtfmPitch"])

    def test_wrap_around_duplicate_azimuth_raises(self, tmp_path: Path) -> None:
        """Azimuths 0 and 2pi are the same physical azimuth and are rejected."""
        from tests.conftest import build_lin_text

        for index, azimuth in ((1, 0.0), (2, 2.0 * np.pi)):
            (tmp_path / f"wrap.{index}.lin").write_text(
                build_lin_text(states=_STATES, a=_A, wind_speed=8.0, azimuth=azimuth),
                encoding="utf-8",
            )
        with pytest.raises(ValueError, match="duplicate azimuth"):
            build_physical_lpv_family(tmp_path, states=["ED PtfmPitch"])

    def test_mixed_rotor_speed_raises(self, tmp_path: Path) -> None:
        """Same wind speed but different rotor speed is still a mixed condition."""
        from tests.conftest import build_lin_text

        for index, rotor in ((1, 0.9), (2, 1.2)):
            (tmp_path / f"mix.{index}.lin").write_text(
                build_lin_text(states=_STATES, a=_A, wind_speed=8.0, rotor_speed=rotor),
                encoding="utf-8",
            )
        with pytest.raises(ValueError, match="disagree on RotorSpeed"):
            build_physical_lpv_family(tmp_path, states=["ED PtfmPitch"])

    def test_close_wind_speeds_still_rejected(self, tmp_path: Path) -> None:
        """Wind speed is the commanded inflow, so its tolerance stays tight.

        Two same-root files at 8.0 and 8.3 m/s (within the relaxed rotor-speed tolerance
        but well above the tight wind tolerance) are different operating points and must
        be rejected, not averaged into a fabricated plant.
        """
        from tests.conftest import build_lin_text

        for index, wind in ((1, 8.0), (2, 8.3)):  # 3.75% apart -> within 5%, above 0.1%
            (tmp_path / f"near.{index}.lin").write_text(
                build_lin_text(
                    states=_STATES, a=_A, wind_speed=wind, azimuth=float(index)
                ),
                encoding="utf-8",
            )
        with pytest.raises(ValueError, match="disagree on WindSpeed"):
            build_physical_lpv_family(tmp_path, states=["ED PtfmPitch"])

    def test_periodic_ripple_within_tolerance_is_accepted(self, tmp_path: Path) -> None:
        """A small per-azimuth rotor-speed ripple (a CalcSteady periodic state) is kept.

        The azimuth files of one trim/CalcSteady operating point carry a sub-percent
        rotor-speed ripple around the revolution; that physical periodicity is what
        azimuth-averaging folds out, so it must not be rejected as a mixed condition.
        """
        from tests.conftest import build_lin_text

        for index, (rotor, azimuth) in enumerate(
            ((0.790, 0.0), (0.795, 1.0), (0.785, 2.0)), start=1
        ):
            (tmp_path / f"ws8.{index}.lin").write_text(
                build_lin_text(
                    states=_STATES,
                    a=_A,
                    wind_speed=8.0,
                    rotor_speed=rotor,
                    azimuth=azimuth,
                ),
                encoding="utf-8",
            )
        family = build_physical_lpv_family(
            tmp_path, states=["ED PtfmPitch"], max_dropped_coupling=None
        )
        assert family.scheduling_points.shape == (1,)
        np.testing.assert_allclose(family.scheduling_points, [8.0])

    def test_empty_directory_raises_file_not_found(self, tmp_path: Path) -> None:
        """A directory with no .lin files raises FileNotFoundError, as documented."""
        with pytest.raises(FileNotFoundError):
            build_physical_lpv_family(tmp_path, states=["ED PtfmPitch"])

    def test_states_as_a_bare_string_raises(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """A single string for states (not a list) is rejected, not iterated by char."""
        lin_dir = _write_point(make_lin_file, wind_speed=8.0).parent
        with pytest.raises(ValueError, match="must be a sequence of names"):
            build_physical_lpv_family(lin_dir, states="ED PtfmPitch")

    def test_duplicate_state_names_raise(self, make_lin_file: LinFileFactory) -> None:
        """Duplicate requested names are rejected."""
        lin_dir = _write_point(make_lin_file, wind_speed=8.0).parent
        with pytest.raises(ValueError, match="duplicate names"):
            build_physical_lpv_family(lin_dir, states=["ED PtfmPitch", "ED PtfmPitch"])

    def test_distinct_names_to_same_channel_raise(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Two different names resolving to the same channel are rejected."""
        lin_dir = _write_point(make_lin_file, wind_speed=8.0).parent
        # 'RotSpeed' (substring) and 'ED RotSpeed' (exact) both resolve to the same row.
        with pytest.raises(ValueError, match="resolve to the same"):
            build_physical_lpv_family(lin_dir, states=["ED RotSpeed", "RotSpeed"])

    def test_states_from_a_generator_are_consumed_once(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """A generator of names is materialized, not exhausted by the first use."""
        lin_dir = _write_point(make_lin_file, wind_speed=8.0).parent
        names = (n for n in ["ED PtfmPitch", "ED RotSpeed"])
        family = build_physical_lpv_family(
            lin_dir,
            states=names,  # type: ignore[arg-type]
            max_dropped_coupling=None,
        )
        assert family.state_names == ["ED PtfmPitch", "ED RotSpeed"]
        assert family.systems[0].a.shape == (2, 2)

    def test_inputs_without_b_matrix_raises(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Requesting inputs from a linearization that has no B matrix is rejected."""
        # Input channels present (so the names resolve), but no B matrix block.
        path = make_lin_file(states=_STATES, inputs=_INPUTS, a=_A, wind_speed=8.0)
        with pytest.raises(ValueError, match="no B matrix"):
            build_physical_lpv_family(
                path.parent, states=["ED PtfmPitch"], inputs=["GenTq"]
            )


def _write_four_state_point(
    make_lin_file: LinFileFactory,
    *,
    wind_speed: float,
    coupling: float = 0.0,
    c: npt.NDArray[np.float64] | None = None,
) -> Path:
    """Write a two-DOF point; ``coupling`` feeds the generator DOF into pitch accel."""
    a = _A4.copy()
    a[2, 1] = coupling  # d(pitch rate)/dt depends on the generator position
    outputs = [("PtfmPitch out, (rad)", False, 0)] if c is not None else []
    return make_lin_file(
        states=_FOUR_STATES,
        outputs=outputs,
        a=a,
        c=c,
        wind_speed=wind_speed,
        rotor_speed=0.5 + 0.05 * wind_speed,
    )


class TestDerivativePairing:
    """Auto-completion of second-order DOFs with their position/velocity twin."""

    def test_position_request_adds_its_velocity(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """A position alone gains its rate twin, so d(pos)/dt = vel is kept."""
        lin_dir = _write_four_state_point(make_lin_file, wind_speed=8.0).parent
        family = build_physical_lpv_family(lin_dir, states=[_POS])
        assert family.state_names == [_POS, _VEL]
        assert family.auto_added_states == [_VEL]
        assert family.systems[0].state_names == [_POS, _VEL]
        # Previously this truncated to A = [[0]] (a pure integrator with no dynamics).
        np.testing.assert_allclose(family.systems[0].a, [[0.0, 1.0], [-0.5, -0.1]])
        np.testing.assert_allclose(family.dropped_coupling_a, [0.0])

    def test_velocity_request_adds_its_position_first(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """A rate alone gains its position twin, inserted before it."""
        lin_dir = _write_four_state_point(make_lin_file, wind_speed=8.0).parent
        family = build_physical_lpv_family(lin_dir, states=[_VEL])
        assert family.state_names == [_POS, _VEL]
        assert family.auto_added_states == [_POS]

    def test_pairs_are_inserted_next_to_their_partner(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Each twin lands beside its partner, keeping the request order otherwise."""
        lin_dir = _write_four_state_point(make_lin_file, wind_speed=8.0).parent
        family = build_physical_lpv_family(lin_dir, states=[_GEN_RATE, _POS])
        assert family.state_names == [_GEN, _GEN_RATE, _POS, _VEL]
        assert family.auto_added_states == [_GEN, _VEL]
        order = [1, 3, 0, 2]  # file indices of the final state order
        np.testing.assert_allclose(family.systems[0].a, _A4[np.ix_(order, order)])

    def test_complete_pairs_add_nothing(self, make_lin_file: LinFileFactory) -> None:
        """A request that already holds both twins is left unchanged."""
        lin_dir = _write_four_state_point(make_lin_file, wind_speed=8.0).parent
        family = build_physical_lpv_family(lin_dir, states=[_VEL, _POS])
        assert family.state_names == [_VEL, _POS]
        assert family.auto_added_states == []

    def test_bare_dof_name_selects_both_twins(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """A name matching a DOF and its rate yields the position then the velocity."""
        lin_dir = _write_four_state_point(make_lin_file, wind_speed=8.0).parent
        family = build_physical_lpv_family(
            lin_dir, states=["Platform pitch tilt rotation DOF"]
        )
        assert family.state_names == ["Platform pitch tilt rotation DOF", _VEL]

    def test_added_states_are_logged(
        self, make_lin_file: LinFileFactory, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The auto-added twins are reported at INFO."""
        lin_dir = _write_four_state_point(make_lin_file, wind_speed=8.0).parent
        with caplog.at_level(logging.INFO, logger="vane.sysid.physical_lpv"):
            build_physical_lpv_family(lin_dir, states=[_POS])
        assert any(_VEL in record.getMessage() for record in caplog.records)

    def test_opt_out_keeps_the_bare_truncation(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """pair_derivatives=False reproduces the requested subset exactly."""
        lin_dir = _write_four_state_point(make_lin_file, wind_speed=8.0).parent
        family = build_physical_lpv_family(
            lin_dir, states=[_POS], pair_derivatives=False, max_dropped_coupling=None
        )
        assert family.state_names == [_POS]
        np.testing.assert_allclose(family.systems[0].a, [[0.0]])
        # Row [0, 0, 1, 0]: all of it lives in the dropped velocity column.
        np.testing.assert_allclose(family.dropped_coupling_a, [1.0])


class TestDroppedCouplingGate:
    """The relative dropped-coupling measure and its tolerance check."""

    def test_unpaired_position_is_rejected_by_default(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Without pairing, dropping a DOF's rate trips the default check."""
        lin_dir = _write_four_state_point(make_lin_file, wind_speed=8.0).parent
        with pytest.raises(ValueError, match="relative coupling"):
            build_physical_lpv_family(lin_dir, states=[_POS], pair_derivatives=False)

    def test_check_names_the_worst_point_and_contributors(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """The error reports the worst scheduling point and the dropped culprit."""
        lin_dir = _write_four_state_point(
            make_lin_file, wind_speed=12.0, coupling=2.0
        ).parent
        _write_four_state_point(make_lin_file, wind_speed=6.0, coupling=0.01)
        with pytest.raises(ValueError, match=r"WindSpeed = 12 m/s") as info:
            build_physical_lpv_family(lin_dir, states=[_POS])
        assert _GEN in str(info.value)
        assert "max_dropped_coupling=0.05" in str(info.value)

    def test_disabled_check_records_per_point_norms(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """With the check off, the per-point norms are recorded in schedule order."""
        lin_dir = _write_four_state_point(
            make_lin_file, wind_speed=12.0, coupling=2.0
        ).parent
        _write_four_state_point(make_lin_file, wind_speed=6.0, coupling=0.01)
        family = build_physical_lpv_family(
            lin_dir, states=[_POS], max_dropped_coupling=None
        )
        np.testing.assert_array_equal(family.scheduling_points, [6.0, 12.0])

        def expected(k: float) -> float:
            # Selected rows [0, 0, 1, 0] and [-0.5, k, -0.1, 0]; dropped columns 1, 3.
            return abs(k) / float(np.sqrt(1.0 + 0.25 + k**2 + 0.01))

        np.testing.assert_allclose(
            family.dropped_coupling_a, [expected(0.01), expected(2.0)]
        )
        np.testing.assert_allclose(family.dropped_coupling_c, [0.0, 0.0])
        np.testing.assert_allclose(family.dropped_coupling, family.dropped_coupling_a)

    def test_small_coupling_passes_the_default_check(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """A weak link to a dropped state stays under the default tolerance."""
        lin_dir = _write_four_state_point(
            make_lin_file, wind_speed=8.0, coupling=0.01
        ).parent
        family = build_physical_lpv_family(lin_dir, states=[_POS])
        assert 0.0 < float(family.dropped_coupling[0]) < 0.05

    def test_custom_tolerance_is_honoured(self, make_lin_file: LinFileFactory) -> None:
        """A looser ceiling admits what a tighter one rejects."""
        lin_dir = _write_four_state_point(
            make_lin_file, wind_speed=8.0, coupling=0.01
        ).parent
        with pytest.raises(ValueError, match=r"max_dropped_coupling=0.001"):
            build_physical_lpv_family(
                lin_dir, states=[_POS], max_dropped_coupling=0.001
            )
        family = build_physical_lpv_family(
            lin_dir, states=[_POS], max_dropped_coupling=0.5
        )
        assert family.state_names == [_POS, _VEL]

    def test_output_coupling_to_dropped_states_is_gated(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """A selected output reading a dropped state trips the C term of the check."""
        c = np.array([[1.0, 1.0, 0.0, 0.0]], dtype=np.float64)  # reads the gen DOF
        lin_dir = _write_four_state_point(make_lin_file, wind_speed=8.0, c=c).parent
        with pytest.raises(ValueError, match=r"C: 0\.707"):
            build_physical_lpv_family(lin_dir, states=[_POS], outputs=["PtfmPitch out"])
        family = build_physical_lpv_family(
            lin_dir,
            states=[_POS],
            outputs=["PtfmPitch out"],
            max_dropped_coupling=None,
        )
        np.testing.assert_allclose(family.dropped_coupling_a, [0.0])
        np.testing.assert_allclose(family.dropped_coupling_c, [np.sqrt(0.5)])

    def test_full_selection_has_zero_coupling(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Selecting every state drops nothing, even under a zero tolerance."""
        lin_dir = _write_point(make_lin_file, wind_speed=8.0).parent
        family = build_physical_lpv_family(
            lin_dir,
            states=["ED PtfmPitch", "ED PtfmPitch rate", "ED RotSpeed"],
            outputs=["RotSpeed", "BldPitch1"],
            max_dropped_coupling=0.0,
        )
        np.testing.assert_array_equal(family.dropped_coupling_a, [0.0])
        np.testing.assert_array_equal(family.dropped_coupling_c, [0.0])
        assert family.auto_added_states == []

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, True, "0.1"])
    def test_invalid_tolerance_raises(
        self, make_lin_file: LinFileFactory, bad: object
    ) -> None:
        """The tolerance must be None or a finite number >= 0."""
        lin_dir = _write_point(make_lin_file, wind_speed=8.0).parent
        tolerance: Any = bad
        with pytest.raises(ValueError, match="max_dropped_coupling must be"):
            build_physical_lpv_family(
                lin_dir, states=["ED RotSpeed"], max_dropped_coupling=tolerance
            )

    def test_integer_tolerance_is_accepted(self, make_lin_file: LinFileFactory) -> None:
        """An int ceiling is a valid number (1 admits any truncation)."""
        lin_dir = _write_point(make_lin_file, wind_speed=8.0).parent
        family = build_physical_lpv_family(
            lin_dir, states=["ED RotSpeed"], max_dropped_coupling=1
        )
        assert family.state_names == ["ED RotSpeed"]


class TestLPVFamilyDefaults:
    """The provenance fields default to empty for a hand-built family."""

    def test_defaults_are_empty(self) -> None:
        """A family built without the new fields has empty pairing/coupling records."""
        family = LPVFamily(scheduling_points=np.zeros(0), systems=[])
        assert family.auto_added_states == []
        assert family.dropped_coupling.shape == (0,)


class TestStackMean:
    """The azimuth-averaging helper."""

    def test_averages_present_matrices(self) -> None:
        """Matrices present in every file are averaged element-wise."""
        result = _stack_mean([np.array([[2.0]]), np.array([[4.0]])], "A")
        assert result is not None
        np.testing.assert_allclose(result, [[3.0]])

    def test_all_absent_returns_none(self) -> None:
        """A matrix absent from every file averages to None."""
        assert _stack_mean([None, None], "B") is None

    def test_partial_presence_raises(self) -> None:
        """A matrix present in some files but not others is rejected."""
        with pytest.raises(ValueError, match="present in some azimuth files"):
            _stack_mean([np.array([[1.0]]), None], "A")

    def test_shape_mismatch_raises(self) -> None:
        """Matrices that disagree in shape across azimuths are rejected."""
        with pytest.raises(ValueError, match="inconsistent shapes"):
            _stack_mean([np.array([[1.0]]), np.array([[1.0, 2.0]])], "A")
