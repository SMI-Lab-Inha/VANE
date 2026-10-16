# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.io.mbc_transform — MBC3 multi-blade coordinate transform."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pytest

from vane.io.lin_reader import read_lin_file
from vane.io.mbc_transform import (
    _has_rate_unit,
    _pair_key,
    _tt_inverse,
    find_blade_triplets,
    mbc3_transform,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import ChannelSpec, LinFileFactory
    from vane.io.lin_reader import LinFile

# Modal parameters for the synthetic system: a non-rotating tower DOF plus three
# identical (isotropic), uncoupled blade flap DOFs.
_K = np.array([4.0, 9.0, 9.0, 9.0])
_C = np.array([0.2, 0.3, 0.3, 0.3])


def _blade_states() -> list[ChannelSpec]:
    """Return 8 channel specs: 4 displacement DOFs then their 4 velocities."""
    disp: list[ChannelSpec] = [
        (
            "ED 1st tower fore-aft bending mode DOF (internal DOF index = DOF_TFA1), m",
            False,
            2,
        ),
        (
            "ED 1st flapwise bending-mode DOF of blade 1 "
            "(internal DOF index = DOF_BF(1,1)), m",
            True,
            2,
        ),
        (
            "ED 1st flapwise bending-mode DOF of blade 2 "
            "(internal DOF index = DOF_BF(2,1)), m",
            True,
            2,
        ),
        (
            "ED 1st flapwise bending-mode DOF of blade 3 "
            "(internal DOF index = DOF_BF(3,1)), m",
            True,
            2,
        ),
    ]
    vel: list[ChannelSpec] = [
        (f"ED First time derivative of {desc}", rot, 2) for desc, rot, _ in disp
    ]
    return disp + vel


def _blade_a() -> np.ndarray:
    """Return the 8x8 state matrix [[0, I], [-K, -C]] for the synthetic system."""
    a = np.zeros((8, 8), dtype=np.float64)
    a[0:4, 4:8] = np.eye(4)
    a[4:8, 0:4] = -np.diag(_K)
    a[4:8, 4:8] = -np.diag(_C)
    return a


def _sorted_eigs(matrix: np.ndarray) -> list[complex]:
    """Return the eigenvalues sorted by a rounded key (stable for degeneracies)."""
    eigs = np.linalg.eigvals(matrix)
    return sorted(eigs, key=lambda v: (round(v.real, 6), round(v.imag, 6)))


def _isotropic_blade_states() -> list[ChannelSpec]:
    """Return states for three identical, uncoupled blade flap DOFs (q2 then q2dot)."""
    disp: list[ChannelSpec] = [
        (
            f"ED 1st flapwise bending-mode DOF of blade {b} "
            f"(internal DOF index = DOF_BF({b},1)), m",
            True,
            2,
        )
        for b in (1, 2, 3)
    ]
    vel: list[ChannelSpec] = [
        (f"ED First time derivative of {desc}", True, 2) for desc, _, _ in disp
    ]
    return disp + vel


def _isotropic_blade_a(w0: float = 2.0, zeta: float = 0.02) -> np.ndarray:
    """Return the rotating-frame A for three identical uncoupled blade oscillators.

    Each blade is a second-order oscillator with undamped frequency ``w0`` and
    damping ratio ``zeta``; the matrix is azimuth-independent and isotropic, so
    the MBC transform must render it azimuth-invariant.
    """
    a = np.zeros((6, 6), dtype=np.float64)
    a[0:3, 3:6] = np.eye(3)
    a[3:6, 0:3] = -(w0**2) * np.eye(3)
    a[3:6, 3:6] = -2.0 * zeta * w0 * np.eye(3)
    return a


def _avg_a_at(
    make_lin_file: LinFileFactory, omega: float, azimuth: float
) -> np.ndarray:
    """Transform the isotropic rotor at one azimuth and return its averaged A."""
    path = make_lin_file(
        states=_isotropic_blade_states(),
        a=_isotropic_blade_a(),
        rotor_speed=omega,
        azimuth=azimuth,
    )
    result = mbc3_transform([read_lin_file(path)])
    assert result.avg_a is not None
    avg_a: np.ndarray = result.avg_a
    return avg_a


class TestFindBladeTriplets:
    """Unit tests for blade-triplet detection from descriptions."""

    def test_finds_complete_triplet_ordered(self) -> None:
        """Three blade channels group into one triplet ordered by blade number."""
        descs = [
            "ED 1st tower fore-aft bending mode DOF (internal DOF index = DOF_TFA1), m",
            "ED 1st flapwise bending-mode DOF of blade 3 "
            "(internal DOF index = DOF_BF(3,1)), m",
            "ED 1st flapwise bending-mode DOF of blade 1 "
            "(internal DOF index = DOF_BF(1,1)), m",
            "ED 1st flapwise bending-mode DOF of blade 2 "
            "(internal DOF index = DOF_BF(2,1)), m",
        ]
        rotating = np.array([False, True, True, True])
        assert find_blade_triplets(descs, rotating) == [(2, 3, 1)]

    def test_incomplete_group_is_ignored(self) -> None:
        """A group missing a blade does not form a triplet."""
        descs = [
            "ED flap blade 1, m",
            "ED flap blade 2, m",
        ]
        rotating = np.array([True, True])
        assert find_blade_triplets(descs, rotating) == []

    def test_non_rotating_channels_ignored(self) -> None:
        """Channels not in the rotating frame are never grouped."""
        descs = [
            "ED flap blade 1, m",
            "ED flap blade 2, m",
            "ED flap blade 3, m",
        ]
        rotating = np.array([False, False, False])
        assert find_blade_triplets(descs, rotating) == []

    def test_distinguishes_nodes_within_blade(self) -> None:
        """AeroDyn-style per-node channels form one triplet per node, not one."""
        descs = [
            "AD B1N001 axial induction, -",
            "AD B2N001 axial induction, -",
            "AD B3N001 axial induction, -",
            "AD B1N002 axial induction, -",
            "AD B2N002 axial induction, -",
            "AD B3N002 axial induction, -",
        ]
        rotating = np.ones(6, dtype=bool)
        assert sorted(find_blade_triplets(descs, rotating)) == [(0, 1, 2), (3, 4, 5)]

    def test_separates_blade_modes(self) -> None:
        """Flap modes 1 and 2 of each blade form separate triplets."""
        descs = [
            "ED 1st flapwise DOF of blade 1 (internal DOF index = DOF_BF(1,1)), m",
            "ED 1st flapwise DOF of blade 2 (internal DOF index = DOF_BF(2,1)), m",
            "ED 1st flapwise DOF of blade 3 (internal DOF index = DOF_BF(3,1)), m",
            "ED 2nd flapwise DOF of blade 1 (internal DOF index = DOF_BF(1,2)), m",
            "ED 2nd flapwise DOF of blade 2 (internal DOF index = DOF_BF(2,2)), m",
            "ED 2nd flapwise DOF of blade 3 (internal DOF index = DOF_BF(3,2)), m",
        ]
        rotating = np.ones(6, dtype=bool)
        assert sorted(find_blade_triplets(descs, rotating)) == [(0, 1, 2), (3, 4, 5)]

    def test_suffix_style_blade_channels(self) -> None:
        """Suffix-style output/input channel names form triplets."""
        descs = [
            "ED RootMyb1, (kN-m)",
            "ED RootMyb2, (kN-m)",
            "ED RootMyb3, (kN-m)",
            "ED BldPitch1, (rad)",
            "ED BldPitch2, (rad)",
            "ED BldPitch3, (rad)",
        ]
        rotating = np.ones(6, dtype=bool)
        assert sorted(find_blade_triplets(descs, rotating)) == [(0, 1, 2), (3, 4, 5)]

    def test_beamdyn_instance_prefix_triplet(self) -> None:
        """Real BeamDyn channels (BD_<blade> + word-form node) form a triplet."""
        descs = [
            "BD_1 finite element node 2 (number of elements = 1; "
            "element order = 5) translational displacement in X, m",
            "BD_2 finite element node 2 (number of elements = 1; "
            "element order = 5) translational displacement in X, m",
            "BD_3 finite element node 2 (number of elements = 1; "
            "element order = 5) translational displacement in X, m",
        ]
        rotating = np.ones(3, dtype=bool)
        assert find_blade_triplets(descs, rotating) == [(0, 1, 2)]

    def test_multiple_blade_tokens_per_channel(self) -> None:
        """A channel with two blade tokens (BD_<b> and B<b>N) still groups."""
        descs = [
            "BD_1 node B1N001 displacement, m",
            "BD_2 node B2N001 displacement, m",
            "BD_3 node B3N001 displacement, m",
        ]
        rotating = np.ones(3, dtype=bool)
        assert find_blade_triplets(descs, rotating) == [(0, 1, 2)]


class TestTtInverse:
    """Unit tests for the analytic 3x3 multi-blade matrix inverse."""

    @pytest.mark.parametrize("azimuth", [0.0, 0.3, 1.7, 5.5])
    def test_inverse_recovers_identity(self, azimuth: float) -> None:
        """The analytic inverse times the forward matrix is the identity."""
        blades = azimuth + 2.0 * np.pi / 3.0 * np.arange(3)
        cos_b, sin_b = np.cos(blades), np.sin(blades)
        tt = np.column_stack((np.ones(3), cos_b, sin_b))
        ttv = _tt_inverse(cos_b, sin_b)
        np.testing.assert_allclose(ttv @ tt, np.eye(3), atol=1e-12)


class TestSweepValidation:
    """mbc3_transform validates the files form one coherent azimuth sweep."""

    def test_duplicate_azimuth_raises(self, make_lin_file: LinFileFactory) -> None:
        """Two files sampling the same azimuth are rejected."""
        states, a = _blade_states(), _blade_a()
        files = [
            read_lin_file(
                make_lin_file(states=states, a=a, rotor_speed=1.0, azimuth=0.5)
            )
            for _ in range(2)
        ]
        with pytest.raises(ValueError, match="duplicate azimuths"):
            mbc3_transform(files)

    def test_wraparound_duplicate_azimuth_raises(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Azimuths near 0 and near 2*pi are circular duplicates and rejected."""
        states, a = _blade_states(), _blade_a()
        files = [
            read_lin_file(
                make_lin_file(states=states, a=a, rotor_speed=1.0, azimuth=0.0)
            ),
            read_lin_file(
                make_lin_file(
                    states=states, a=a, rotor_speed=1.0, azimuth=2.0 * np.pi - 1e-6
                )
            ),
        ]
        with pytest.raises(ValueError, match="duplicate azimuths"):
            mbc3_transform(files)

    def test_rotor_speed_drift_raises(self, make_lin_file: LinFileFactory) -> None:
        """A sweep whose rotor speed drifts between files is rejected."""
        states, a = _blade_states(), _blade_a()
        files = [
            read_lin_file(
                make_lin_file(states=states, a=a, rotor_speed=1.0, azimuth=0.0)
            ),
            read_lin_file(
                make_lin_file(states=states, a=a, rotor_speed=2.0, azimuth=1.0)
            ),
        ]
        with pytest.raises(ValueError, match="drifts across"):
            mbc3_transform(files)

    def test_consistent_sweep_passes(self, make_lin_file: LinFileFactory) -> None:
        """Distinct azimuths at a constant rotor speed transform without error."""
        states, a = _blade_states(), _blade_a()
        files = [
            read_lin_file(
                make_lin_file(states=states, a=a, rotor_speed=1.0, azimuth=az)
            )
            for az in (0.0, 1.0, 2.0)
        ]
        assert mbc3_transform(files).n_blades == 3


class TestMbc3Synthetic:
    """Unit tests on synthetic systems with a known answer."""

    def test_zero_speed_preserves_eigenvalues(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """At zero rotor speed MBC3 is a similarity transform (eigenvalues kept)."""
        a = _blade_a()
        path = make_lin_file(states=_blade_states(), a=a, rotor_speed=0.0, azimuth=0.3)
        result = mbc3_transform([read_lin_file(path)])
        assert result.performed_transformation is True
        assert result.n_blades == 3
        assert (result.ndof2, result.ndof1) == (4, 0)
        assert result.avg_a is not None
        np.testing.assert_allclose(
            _sorted_eigs(result.avg_a), _sorted_eigs(a), atol=1e-9
        )

    def test_multi_azimuth_average_preserves_eigenvalues(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Averaging an isotropic rotor over azimuths preserves eigenvalues."""
        a = _blade_a()
        states = _blade_states()
        files = [
            read_lin_file(
                make_lin_file(states=states, a=a, rotor_speed=0.0, azimuth=az)
            )
            for az in (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0)
        ]
        result = mbc3_transform(files)
        assert result.performed_transformation is True
        assert result.azimuths_deg.tolist() == pytest.approx(
            [0.0, 120.0, 240.0], abs=0.01
        )
        assert result.avg_a is not None
        np.testing.assert_allclose(
            _sorted_eigs(result.avg_a), _sorted_eigs(a), atol=1e-9
        )

    def test_reorders_states_q2_first(self, make_lin_file: LinFileFactory) -> None:
        """Transformed state order is [q2, q2dot, q1] with q2 displacements first."""
        path = make_lin_file(states=_blade_states(), a=_blade_a(), azimuth=0.1)
        result = mbc3_transform([read_lin_file(path)])
        assert len(result.state_descriptions) == 8
        first_four = result.state_descriptions[:4]
        assert all("First time derivative" not in d for d in first_four)

    def test_transforms_b_c_d_shapes(self, make_lin_file: LinFileFactory) -> None:
        """B, C, D are transformed and keep their dimensions."""
        states = _blade_states()
        inputs: list[ChannelSpec] = [("ED generator torque, Nm", False, 0)]
        outputs: list[ChannelSpec] = [("ED rotor speed, rad/s", False, 0)]
        path = make_lin_file(
            states=states,
            inputs=inputs,
            outputs=outputs,
            a=_blade_a(),
            b=np.ones((8, 1)),
            c=np.ones((1, 8)),
            d=np.array([[2.0]]),
            rotor_speed=1.0,
            azimuth=0.4,
        )
        result = mbc3_transform([read_lin_file(path)], retain_per_azimuth=True)
        assert result.avg_b is not None and result.avg_b.shape == (8, 1)
        assert result.avg_c is not None and result.avg_c.shape == (1, 8)
        assert result.avg_d is not None and result.avg_d.shape == (1, 1)
        assert result.per_azimuth_a is not None
        assert result.per_azimuth_b is not None
        assert result.per_azimuth_c is not None
        assert result.per_azimuth_d is not None
        assert result.per_azimuth_output_operating_point is not None
        assert result.per_azimuth_input_operating_point is not None
        assert len(result.per_azimuth_a) == 1
        assert result.avg_a is not None
        np.testing.assert_allclose(result.per_azimuth_a[0], result.avg_a)
        np.testing.assert_allclose(result.per_azimuth_b[0], result.avg_b)
        np.testing.assert_allclose(result.per_azimuth_c[0], result.avg_c)
        np.testing.assert_allclose(result.per_azimuth_d[0], result.avg_d)
        # With no input/output triplets, D is unchanged by the transform.
        np.testing.assert_allclose(result.avg_d, [[2.0]])

    def test_untransformed_per_azimuth_d_does_not_alias_input(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Retained per-azimuth D of an untransformed average is a copy."""
        states: list[ChannelSpec] = [
            ("ED tower displacement, m", False, 2),
            ("ED First time derivative of tower displacement, m/s", False, 2),
        ]
        lin = read_lin_file(
            make_lin_file(
                states=states,
                inputs=[("ED torque, Nm", False, 0)],
                outputs=[("ED PtfmSurge, (m)", False, 0)],
                a=np.array([[0.0, 1.0], [-4.0, -0.2]]),
                b=np.ones((2, 1)),
                c=np.ones((1, 2)),
                d=np.array([[2.0]]),
            )
        )
        result = mbc3_transform([lin], retain_per_azimuth=True)
        assert result.performed_transformation is False
        assert result.per_azimuth_d is not None and lin.d is not None
        result.per_azimuth_d[0][0, 0] = 99.0
        np.testing.assert_array_equal(lin.d, [[2.0]])

    def test_carries_untransformed_io_operating_points(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Plain output/input baselines survive the azimuth average."""
        states: list[ChannelSpec] = [
            ("ED tower displacement, m", False, 2),
            ("ED First time derivative of tower displacement, m/s", False, 2),
        ]
        inputs: list[ChannelSpec] = [("ED torque, Nm", False, 0)]
        outputs: list[ChannelSpec] = [("ED PtfmSurge, (m)", False, 0)]
        files = [
            read_lin_file(
                make_lin_file(
                    states=states,
                    inputs=inputs,
                    outputs=outputs,
                    a=np.array([[0.0, 1.0], [-4.0, -0.2]]),
                    b=np.ones((2, 1)),
                    c=np.ones((1, 2)),
                    d=np.zeros((1, 1)),
                    input_values=[value],
                    output_values=[10.0 * value],
                    azimuth=azimuth,
                )
            )
            for value, azimuth in ((2.0, 0.0), (4.0, 1.0))
        ]
        result = mbc3_transform(files)
        assert result.input_operating_point is not None
        assert result.output_operating_point is not None
        np.testing.assert_allclose(result.input_operating_point, [3.0])
        np.testing.assert_allclose(result.output_operating_point, [30.0])

    def test_transforms_rotating_io_operating_points(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Rotating blade baselines become collective/cosine/sine coordinates."""
        states: list[ChannelSpec] = [
            ("ED tower displacement, m", False, 2),
            ("ED First time derivative of tower displacement, m/s", False, 2),
        ]
        inputs: list[ChannelSpec] = [
            (f"ED BldPitch{blade}, rad", True, 0) for blade in (1, 2, 3)
        ]
        outputs: list[ChannelSpec] = [
            (f"ED RootMxb{blade}, (kN-m)", True, 0) for blade in (1, 2, 3)
        ]
        azimuth = 0.0
        blades = azimuth + 2.0 * np.pi / 3.0 * np.arange(3)
        transform = np.column_stack((np.ones(3), np.cos(blades), np.sin(blades)))
        input_mbc = np.array([0.2, 0.03, -0.04])
        output_mbc = np.array([100.0, 5.0, -7.0])
        path = make_lin_file(
            states=states,
            inputs=inputs,
            outputs=outputs,
            a=np.array([[0.0, 1.0], [-4.0, -0.2]]),
            b=np.ones((2, 3)),
            c=np.ones((3, 2)),
            d=np.zeros((3, 3)),
            input_values=transform @ input_mbc,
            output_values=transform @ output_mbc,
            azimuth=azimuth,
        )
        result = mbc3_transform([read_lin_file(path)], retain_per_azimuth=True)
        assert result.input_operating_point is not None
        assert result.output_operating_point is not None
        np.testing.assert_allclose(result.input_operating_point, input_mbc, atol=1e-10)
        np.testing.assert_allclose(
            result.output_operating_point, output_mbc, atol=1e-10
        )
        assert result.per_azimuth_input_operating_point is not None
        assert result.per_azimuth_output_operating_point is not None
        np.testing.assert_allclose(
            result.per_azimuth_input_operating_point[0], input_mbc, atol=1e-10
        )
        np.testing.assert_allclose(
            result.per_azimuth_output_operating_point[0], output_mbc, atol=1e-10
        )

    def test_no_rotating_states_skips_transform(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """A model without blade triplets averages without transforming."""
        states: list[ChannelSpec] = [
            ("ED 1st tower fore-aft bending mode DOF (DOF_TFA1), m", False, 2),
            ("ED First time derivative of tower fore-aft (DOF_TFA1), m/s", False, 2),
        ]
        a = np.array([[0.0, 1.0], [-4.0, -0.2]])
        path = make_lin_file(states=states, a=a, azimuth=0.2)
        result = mbc3_transform([read_lin_file(path)])
        assert result.performed_transformation is False
        assert result.n_blades == 0
        assert result.avg_a is not None
        np.testing.assert_allclose(result.avg_a, a)

    def test_io_only_triplets_still_transform(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Rotating blade I/O with no rotating blade states still transforms B."""
        states: list[ChannelSpec] = [
            (
                "ED 1st tower fore-aft bending mode DOF "
                "(internal DOF index = DOF_TFA1), m",
                False,
                2,
            ),
            (
                "ED First time derivative of 1st tower fore-aft bending mode DOF "
                "(internal DOF index = DOF_TFA1), m/s",
                False,
                2,
            ),
        ]
        inputs: list[ChannelSpec] = [
            ("ED BldPitch1, rad", True, 0),
            ("ED BldPitch2, rad", True, 0),
            ("ED BldPitch3, rad", True, 0),
        ]
        path = make_lin_file(
            states=states,
            inputs=inputs,
            a=np.array([[0.0, 1.0], [-4.0, -0.2]]),
            b=np.arange(6.0).reshape(2, 3),
            azimuth=0.3,
        )
        result = mbc3_transform([read_lin_file(path)])
        assert result.performed_transformation is True
        assert result.n_blades == 3
        assert result.avg_b is not None and result.avg_b.shape == (2, 3)

    def test_velocity_row_with_order_one_is_second_order(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """A 'First time derivative' row is q2dot even when its order is 1."""
        states: list[ChannelSpec] = [
            (
                "ED 1st tower fore-aft bending mode DOF "
                "(internal DOF index = DOF_TFA1), m",
                False,
                2,
            ),
            (
                "ED First time derivative of 1st tower fore-aft bending mode DOF "
                "(internal DOF index = DOF_TFA1), m/s",
                False,
                1,
            ),
        ]
        path = make_lin_file(states=states, a=np.array([[0.0, 1.0], [-4.0, -0.2]]))
        result = mbc3_transform([read_lin_file(path)])
        assert (result.ndof2, result.ndof1) == (1, 0)

    def test_untransformed_result_uses_canonical_state_order(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Without a transform, matrices are still reordered to [q2, q2dot, q1]."""
        states: list[ChannelSpec] = [
            ("ED tower fore-aft (internal DOF index = DOF_TFA1), m", False, 2),
            (
                "ED First time derivative of tower fore-aft "
                "(internal DOF index = DOF_TFA1), m/s",
                False,
                2,
            ),
            ("BD_1 finite element node 2 translational X, m", False, 2),
            (
                "BD_1 First time derivative of finite element node 2 "
                "translational X, m/s",
                False,
                2,
            ),
        ]
        a = np.arange(16.0).reshape(4, 4)
        result = mbc3_transform([read_lin_file(make_lin_file(states=states, a=a))])
        assert result.performed_transformation is False
        # Native order is [disp_ED, vel_ED, disp_BD, vel_BD]; canonical groups
        # displacements then velocities -> permutation [0, 2, 1, 3].
        perm = [0, 2, 1, 3]
        assert result.avg_a is not None
        np.testing.assert_allclose(result.avg_a, a[np.ix_(perm, perm)])
        assert "First time derivative" not in result.state_descriptions[0]
        assert "First time derivative" not in result.state_descriptions[1]
        assert "First time derivative" in result.state_descriptions[2]
        assert "First time derivative" in result.state_descriptions[3]

    def test_empty_input_raises(self) -> None:
        """An empty file list raises ValueError."""
        with pytest.raises(ValueError, match="at least one"):
            mbc3_transform([])

    def test_inconsistent_matrices_across_azimuths_raises(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Same-dimension azimuth files with a different matrix set are rejected.

        (A file missing the load-bearing A matrix is now rejected earlier, at parse
        time, by read_lin_file; here both files have A and matching dimensions, but one
        carries a feedthrough D matrix and the other does not.)
        """
        states = _blade_states()
        inputs: list[ChannelSpec] = [("ED generator torque, Nm", False, 0)]
        outputs: list[ChannelSpec] = [("ED tower-base moment, N-m", False, 0)]
        b = np.zeros((8, 1))
        c = np.zeros((1, 8))
        first = read_lin_file(
            make_lin_file(
                states=states,
                inputs=inputs,
                outputs=outputs,
                a=_blade_a(),
                b=b,
                c=c,
                d=np.zeros((1, 1)),  # D present
                azimuth=0.0,
            )
        )
        second = read_lin_file(
            make_lin_file(
                states=states,
                inputs=inputs,
                outputs=outputs,
                a=_blade_a(),
                b=b,
                c=c,  # no D
                azimuth=0.1,
            )
        )
        with pytest.raises(ValueError, match="Inconsistent matrices"):
            mbc3_transform([first, second])

    def test_inconsistent_channel_layout_raises(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Files with matching dimensions but different descriptions are rejected."""
        a = np.array([[0.0, 1.0], [-4.0, -0.2]])
        states_a: list[ChannelSpec] = [
            ("ED tower fore-aft (internal DOF index = DOF_TFA1), m", False, 2),
            (
                "ED First time derivative of tower fore-aft "
                "(internal DOF index = DOF_TFA1), m/s",
                False,
                2,
            ),
        ]
        states_b: list[ChannelSpec] = [
            ("ED generator azimuth (internal DOF index = DOF_GeAz), rad", False, 2),
            (
                "ED First time derivative of generator azimuth "
                "(internal DOF index = DOF_GeAz), rad/s",
                False,
                2,
            ),
        ]
        first = read_lin_file(make_lin_file(states=states_a, a=a, azimuth=0.0))
        second = read_lin_file(make_lin_file(states=states_b, a=a, azimuth=0.1))
        with pytest.raises(ValueError, match="channel layout"):
            mbc3_transform([first, second])


class TestMbc3Mathematics:
    """Rigorous checks of the MBC3 transform's defining mathematical properties."""

    @pytest.mark.parametrize("omega", [0.0, 0.5, 1.5])
    def test_a_is_azimuth_invariant(
        self, make_lin_file: LinFileFactory, omega: float
    ) -> None:
        """For an isotropic rotor the transformed A is identical at every azimuth.

        This is the defining property of the Coleman/MBC transform and holds at
        every rotor speed; it fails if the block-diagonal inverse omits the
        velocity-coupling block, which would corrupt A only when ``omega != 0``.
        """
        azimuths = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
        mats = [_avg_a_at(make_lin_file, omega, psi) for psi in azimuths]
        for mat in mats[1:]:
            np.testing.assert_allclose(mat, mats[0], atol=1e-9)

    def test_regressive_progressive_frequency_split(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """The cosine/sine modes split by +/- Omega; the collective stays at w0."""
        omega, w0 = 1.0, 2.0
        path = make_lin_file(
            states=_isotropic_blade_states(),
            a=_isotropic_blade_a(w0=w0),
            rotor_speed=omega,
            azimuth=0.3,
        )
        result = mbc3_transform([read_lin_file(path)])
        assert result.avg_a is not None
        damped = np.unique(np.round(np.abs(np.linalg.eigvals(result.avg_a).imag), 2))
        np.testing.assert_allclose(
            np.sort(damped), [w0 - omega, w0, w0 + omega], atol=0.02
        )

    def test_average_equals_single_azimuth_for_isotropic_rotor(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Averaging an azimuth-invariant system reproduces the single-azimuth A."""
        states, a = _isotropic_blade_states(), _isotropic_blade_a()
        files = [
            read_lin_file(make_lin_file(states=states, a=a, rotor_speed=1.0, azimuth=p))
            for p in (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0)
        ]
        averaged = mbc3_transform(files).avg_a
        single = mbc3_transform([files[0]]).avg_a
        assert averaged is not None and single is not None
        np.testing.assert_allclose(averaged, single, atol=1e-9)

    def test_b_transform_is_azimuth_invariant(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """B for isotropic blade inputs is azimuth-invariant, like A."""
        inputs: list[ChannelSpec] = [
            (f"ED BldPitch{b}, rad", True, 0) for b in (1, 2, 3)
        ]
        b = np.zeros((6, 3))
        b[3:6, :] = np.eye(3)
        mats = []
        for psi in np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False):
            path = make_lin_file(
                states=_isotropic_blade_states(),
                inputs=inputs,
                a=_isotropic_blade_a(),
                b=b,
                rotor_speed=1.0,
                azimuth=psi,
            )
            avg_b = mbc3_transform([read_lin_file(path)]).avg_b
            assert avg_b is not None
            mats.append(avg_b)
        for mat in mats[1:]:
            np.testing.assert_allclose(mat, mats[0], atol=1e-9)

    def test_tt_inverse_matches_numerical_inverse(self) -> None:
        """The analytic t-tilde inverse matches a numerical matrix inverse."""
        for psi in (0.0, 0.4, 2.1, 5.0):
            blades = psi + 2.0 * np.pi / 3.0 * np.arange(3)
            cos_b, sin_b = np.cos(blades), np.sin(blades)
            tt = np.column_stack((np.ones(3), cos_b, sin_b))
            np.testing.assert_allclose(
                _tt_inverse(cos_b, sin_b), np.linalg.inv(tt), atol=1e-12
            )

    def test_transformed_matrices_are_real_and_finite(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """The transformed A is purely real and finite for real input matrices."""
        path = make_lin_file(
            states=_isotropic_blade_states(),
            a=_isotropic_blade_a(),
            rotor_speed=1.2,
            azimuth=0.7,
        )
        result = mbc3_transform([read_lin_file(path)])
        assert result.avg_a is not None
        assert np.isrealobj(result.avg_a)
        assert np.all(np.isfinite(result.avg_a))


class TestMbc3RTest:
    """Integration test on the real CalcSteady azimuth triple (skipped if absent)."""

    def test_calcsteady_runs(self, calcsteady_lin_files: list[Path]) -> None:
        """MBC3 on the 3-azimuth 5MW land case yields a real, finite LTI system."""
        files = [read_lin_file(p) for p in calcsteady_lin_files]
        result = mbc3_transform(files)
        assert result.performed_transformation is True
        assert result.n_blades == 3
        assert result.avg_a is not None
        n_x = files[0].n_x
        assert result.avg_a.shape == (n_x, n_x)
        assert np.all(np.isfinite(result.avg_a))
        assert 2 * result.ndof2 + result.ndof1 == n_x
        # CalcSteady rotor speed is ~1.0003 rad/s -> ~9.55 rev/min.
        assert result.rotor_speed_rpm == pytest.approx(9.55, abs=0.2)
        assert result.azimuths_deg.tolist() == pytest.approx(
            [0.0, 120.0, 240.0], abs=2.0
        )


class TestStateLayoutMultiModule:
    """Velocity-state detection across modules (ElastoDyn marker + MoorDyn V/P)."""

    def test_has_rate_unit(self) -> None:
        """A trailing rate unit (.../s) marks a velocity; a plain unit does not."""
        assert _has_rate_unit("MD Line 1 node 1 Vx, m/s")
        assert _has_rate_unit("ED First time derivative of ..., rad/s")
        assert not _has_rate_unit("MD Line 1 node 1 Px, m")
        assert not _has_rate_unit("ED Platform surge ..., m")

    def test_pair_key_links_moordyn_position_and_velocity(self) -> None:
        """A MoorDyn position and its velocity share a pairing key."""
        assert _pair_key("MD Line 1 node 1 Px, m") == _pair_key(
            "MD Line 1 node 1 Vx, m/s"
        )

    def test_pair_key_does_not_link_unrelated_rate_state(self) -> None:
        """An AeroDyn induced velocity (no position partner) is not collapsed."""
        # "vind" has no upper-case P/V component, so its key is the bare body and
        # cannot match a MoorDyn position key.
        vind = _pair_key("AD vind (axial) at blade 1, node 1, m/s")
        assert vind == "AD vind (axial) at blade 1, node 1"
        assert vind != _pair_key("MD Line 1 node 1 Px, m")

    def test_moordyn_velocity_states_form_second_order_pairs(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """MoorDyn Px/Vx node states pair into q2/q2dot alongside ElastoDyn DOFs."""
        states: list[ChannelSpec] = [
            ("ED 1st tower fore-aft (internal DOF index = DOF_TFA1), m", False, 2),
            (
                "ED First time derivative of 1st tower fore-aft (DOF_TFA1), m/s",
                False,
                1,
            ),
            ("MD Line 1 node 1 Px, m", False, 2),
            ("MD Line 1 node 1 Vx, m/s", False, 2),
        ]
        a = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [-4.0, -0.2, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, -9.0, -0.3],
            ],
            dtype=np.float64,
        )
        result = mbc3_transform([read_lin_file(make_lin_file(states=states, a=a))])
        # One ElastoDyn DOF and one MoorDyn node DOF, each a displacement/velocity
        # pair, and no first-order states.
        assert result.ndof2 == 2
        assert result.ndof1 == 0
        assert result.avg_a is not None

    def test_pair_key_strips_bare_unit_and_leibniz_derivative(self) -> None:
        """A StC position and its ``dX/dt`` velocity (no comma) share a key."""
        pos = _pair_key("SrvD Nacelle StC 1 local displacement state X m")
        vel = _pair_key("SrvD Nacelle StC 1 local displacement state dX/dt m/s")
        assert pos == vel == "SrvD Nacelle StC 1 local displacement state X"
        # A trailing channel name that is not a unit is left untouched.
        assert _pair_key("HD 1 RdtnPtfmS") == "HD 1 RdtnPtfmS"

    def test_servodyn_stc_states_form_second_order_pairs(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """ServoDyn StC ``X m``/``dX/dt m/s`` states pair into q2/q2dot."""
        stc = "SrvD Tower StC 1 local displacement state"
        states: list[ChannelSpec] = [
            (f"{stc} X m", False, 2),
            (f"{stc} Y m", False, 2),
            (f"{stc} dX/dt m/s", False, 2),
            (f"{stc} dY/dt m/s", False, 2),
            ("ED Platform pitch tilt rotation DOF (DOF_P), rad", False, 2),
            (
                "ED First time derivative of Platform pitch tilt rotation DOF (DOF_P), "
                "rad/s",
                False,
                2,
            ),
        ]
        n = len(states)
        a = -np.eye(n)
        result = mbc3_transform([read_lin_file(make_lin_file(states=states, a=a))])
        assert result.ndof2 == 3
        assert result.ndof1 == 0
        # Each displacement is followed (ndof2 later) by its own velocity.
        descs = result.state_descriptions
        assert descs[:3] == [f"{stc} X m", f"{stc} Y m", states[4][0]]
        assert descs[3:] == [f"{stc} dX/dt m/s", f"{stc} dY/dt m/s", states[5][0]]

    def test_order_zero_subdyn_rigid_body_pair_is_second_order(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """SubDyn order-0 rigid-body displacement/velocity pairs become q2/q2dot."""
        states: list[ChannelSpec] = [
            ("ED Platform pitch tilt rotation DOF (DOF_P), rad", False, 2),
            ("SD Rigid-body mode 1 displacement, m or rad", False, 0),
            ("SD Rigid-body mode 2 displacement, m or rad", False, 0),
            (
                "ED First time derivative of Platform pitch tilt rotation DOF (DOF_P), "
                "rad/s",
                False,
                2,
            ),
            (
                "SD First time derivative of rigid-body mode 1 displacement, "
                "(m or rad)/s",
                False,
                0,
            ),
            (
                "SD First time derivative of rigid-body mode 2 displacement, "
                "(m or rad)/s",
                False,
                0,
            ),
            ("HD 1 RdtnPtfmS", False, 1),
        ]
        a = -np.eye(len(states))
        result = mbc3_transform([read_lin_file(make_lin_file(states=states, a=a))])
        assert result.ndof2 == 3
        assert result.ndof1 == 1
        assert result.state_descriptions[-1] == "HD 1 RdtnPtfmS"

    def test_order_zero_marker_without_partner_stays_first_order(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """An order-0 marker state with no displacement partner is first-order."""
        states: list[ChannelSpec] = [
            ("ED Platform pitch tilt rotation DOF (DOF_P), rad", False, 2),
            (
                "ED First time derivative of Platform pitch tilt rotation DOF (DOF_P), "
                "rad/s",
                False,
                2,
            ),
            ("XX First time derivative of an orphan quantity, m/s", False, 0),
        ]
        a = -np.eye(len(states))
        result = mbc3_transform([read_lin_file(make_lin_file(states=states, a=a))])
        assert result.ndof2 == 1
        assert result.ndof1 == 1


def _iso_file(
    make_lin_file: LinFileFactory,
    omega: float,
    azimuth_deg: float,
    wind_speed: float = 12.0,
) -> LinFile:
    """Return an isotropic-rotor lin file (blade triplets) at one azimuth/speed."""
    return read_lin_file(
        make_lin_file(
            states=_isotropic_blade_states(),
            a=_isotropic_blade_a(),
            rotor_speed=omega,
            wind_speed=wind_speed,
            azimuth=np.deg2rad(azimuth_deg),
        )
    )


class TestAzimuthCoverage:
    """Tests for the rotating-case azimuth-coverage warning and drift override."""

    def test_single_azimuth_rotating_warns(
        self, make_lin_file: LinFileFactory, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A single azimuth at nonzero rotor speed warns of insufficient coverage."""
        with caplog.at_level(logging.WARNING, logger="vane.io.mbc_transform"):
            mbc3_transform([_iso_file(make_lin_file, omega=1.0, azimuth_deg=0.0)])
        assert "full-revolution sweep" in caplog.text

    def test_standstill_single_azimuth_does_not_warn(
        self, make_lin_file: LinFileFactory, caplog: pytest.LogCaptureFixture
    ) -> None:
        """At standstill a single azimuth is valid, so no coverage warning is logged."""
        with caplog.at_level(logging.WARNING, logger="vane.io.mbc_transform"):
            mbc3_transform([_iso_file(make_lin_file, omega=0.0, azimuth_deg=0.0)])
        assert "full-revolution sweep" not in caplog.text
        assert "non-uniform" not in caplog.text

    def test_nonuniform_azimuths_warn(
        self, make_lin_file: LinFileFactory, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Three clustered azimuths (a large gap) warn that the average is biased."""
        files = [_iso_file(make_lin_file, 1.0, d) for d in (0.0, 10.0, 20.0)]
        with caplog.at_level(logging.WARNING, logger="vane.io.mbc_transform"):
            mbc3_transform(files)
        assert "non-uniform" in caplog.text

    def test_clustered_azimuths_warn(
        self, make_lin_file: LinFileFactory, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Near-duplicate azimuths (distinct enough for the duplicate check) warn."""
        # 0, 0.01, 180 deg: effectively two samples, but 0.01 deg > the 1e-3 deg
        # duplicate tolerance, so it is caught only by the clustering (min-gap) check.
        files = [_iso_file(make_lin_file, 1.0, d) for d in (0.0, 0.01, 180.0)]
        with caplog.at_level(logging.WARNING, logger="vane.io.mbc_transform"):
            mbc3_transform(files)
        assert "non-uniform" in caplog.text

    def test_uniform_three_azimuths_do_not_warn(
        self, make_lin_file: LinFileFactory, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Three uniform azimuths (the minimum) cover a revolution, so no warning."""
        files = [_iso_file(make_lin_file, 1.0, d) for d in (0.0, 120.0, 240.0)]
        with caplog.at_level(logging.WARNING, logger="vane.io.mbc_transform"):
            mbc3_transform(files)
        assert "full-revolution sweep" not in caplog.text
        assert "non-uniform" not in caplog.text

    def test_rotor_speed_drift_default_rejects_then_override_accepts(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Rotor-speed drift ~3%: rejected by the 1% default, allowed by 5%."""
        files = [
            _iso_file(make_lin_file, 1.00, 0.0),
            _iso_file(make_lin_file, 1.03, 120.0),
        ]
        with pytest.raises(ValueError, match="Rotor speed drifts"):
            mbc3_transform(files)
        result = mbc3_transform(files, rotor_speed_drift_rtol=0.05)
        assert result.avg_a is not None

    def test_wind_speed_drift_default_rejects_then_override_accepts(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Wind drift is not allowed to masquerade as azimuthal periodicity."""
        files = [
            _iso_file(make_lin_file, 1.0, 0.0, wind_speed=12.0),
            _iso_file(make_lin_file, 1.0, 120.0, wind_speed=12.4),
        ]
        with pytest.raises(ValueError, match="Wind speed drifts"):
            mbc3_transform(files)
        result = mbc3_transform(files, wind_speed_drift_rtol=0.05)
        assert result.avg_a is not None

    @pytest.mark.parametrize("value", [-1.0, np.nan, np.inf])
    @pytest.mark.parametrize(
        "keyword", ["rotor_speed_drift_rtol", "wind_speed_drift_rtol"]
    )
    def test_invalid_drift_tolerance_raises(
        self,
        make_lin_file: LinFileFactory,
        keyword: str,
        value: float,
    ) -> None:
        """A non-finite or negative tolerance cannot disable coherence checks."""
        files = [_iso_file(make_lin_file, 1.0, 0.0)]
        with pytest.raises(ValueError, match=f"{keyword} must be finite"):
            if keyword == "rotor_speed_drift_rtol":
                mbc3_transform(files, rotor_speed_drift_rtol=value)
            else:
                mbc3_transform(files, wind_speed_drift_rtol=value)

    def test_rotor_speed_drift_just_inside_tolerance_passes(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """A spread just under the 1% default is accepted (boundary, low side)."""
        # 1.000 / 1.008: spread 0.008, mean 1.004 -> 0.80% < 1%.
        files = [
            _iso_file(make_lin_file, 1.000, 0.0),
            _iso_file(make_lin_file, 1.008, 120.0),
        ]
        result = mbc3_transform(files)
        assert result.avg_a is not None

    def test_rotor_speed_drift_just_outside_tolerance_raises(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """A spread just over the 1% default is rejected (boundary, high side)."""
        # 1.000 / 1.013: spread 0.013, mean 1.0065 -> 1.29% > 1%.
        files = [
            _iso_file(make_lin_file, 1.000, 0.0),
            _iso_file(make_lin_file, 1.013, 120.0),
        ]
        with pytest.raises(ValueError, match="Rotor speed drifts"):
            mbc3_transform(files)


def _n_blade_files(
    make_lin_file: LinFileFactory, n_blades: int, omega: float
) -> list[LinFile]:
    """Return three azimuths of an isotropic ``n_blades`` flap rotor."""
    disp: list[ChannelSpec] = [
        (
            f"ED 1st flapwise bending-mode DOF of blade {b} "
            f"(internal DOF index = DOF_BF({b},1)), m",
            True,
            2,
        )
        for b in range(1, n_blades + 1)
    ]
    states = disp + [
        (f"ED First time derivative of {desc}", True, 2) for desc, _, _ in disp
    ]
    a = np.zeros((2 * n_blades, 2 * n_blades), dtype=np.float64)
    a[:n_blades, n_blades:] = np.eye(n_blades)
    a[n_blades:, :n_blades] = -4.0 * np.eye(n_blades)
    a[n_blades:, n_blades:] = -0.08 * np.eye(n_blades)
    return [
        read_lin_file(make_lin_file(states=states, a=a, rotor_speed=omega, azimuth=psi))
        for psi in (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0)
    ]


class TestRotorBladeCount:
    """The three-blade transform is never applied to another rotor silently."""

    def test_four_bladed_rotor_is_rejected(self, make_lin_file: LinFileFactory) -> None:
        """Blades 1-3 of a four-bladed rotor used to be transformed as a triplet.

        The fourth blade stayed in the rotating frame while the other three were
        transformed with a 120-degree blade spacing, a silently wrong model.
        """
        with pytest.raises(ValueError, match="three-bladed rotors only"):
            mbc3_transform(_n_blade_files(make_lin_file, 4, 1.0))

    @pytest.mark.parametrize("channel", ["inputs", "outputs"])
    def test_rigid_four_bladed_rotor_io_is_rejected(
        self, make_lin_file: LinFileFactory, channel: str
    ) -> None:
        """A rigid rotor has blade inputs/outputs but no blade states: still checked.

        Blades 1-3 of its I/O channels formed a triplet and were transformed with a
        120-degree spacing while blade 4 stayed in the rotating frame.
        """
        states = [
            ("ED 1st tower fore-aft bending mode DOF, m", False, 2),
            ("ED First time derivative of 1st tower fore-aft bending mode DOF, m/s",
             False, 2),
        ]  # fmt: skip
        blades = [(f"ED Blade {b} pitch command, rad", True, 0) for b in range(1, 5)]
        a = np.array([[0.0, 1.0], [-4.0, -0.1]])
        files = [
            read_lin_file(
                make_lin_file(
                    states=states,
                    a=a,
                    rotor_speed=1.0,
                    azimuth=psi,
                    **{channel: blades},
                )
            )
            for psi in (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0)
        ]
        with pytest.raises(ValueError, match=f"Rotating {channel} name blade 4"):
            mbc3_transform(files)

    def test_two_bladed_rotating_rotor_warns(
        self, make_lin_file: LinFileFactory, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Rotating states without a triplet stay in the rotating frame: warned."""
        with caplog.at_level(logging.WARNING, logger="vane.io.mbc_transform"):
            result = mbc3_transform(_n_blade_files(make_lin_file, 2, 1.0))
        assert not result.performed_transformation
        assert "form no three-blade triplet" in caplog.text

    def test_parked_two_bladed_rotor_does_not_warn(
        self, make_lin_file: LinFileFactory, caplog: pytest.LogCaptureFixture
    ) -> None:
        """At standstill the rotating frame is the fixed frame: no warning."""
        with caplog.at_level(logging.WARNING, logger="vane.io.mbc_transform"):
            mbc3_transform(_n_blade_files(make_lin_file, 2, 0.0))
        assert "form no three-blade triplet" not in caplog.text

    def test_three_bladed_rotor_does_not_warn(
        self, make_lin_file: LinFileFactory, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Every rotating state of a three-bladed rotor is in a triplet."""
        with caplog.at_level(logging.WARNING, logger="vane.io.mbc_transform"):
            result = mbc3_transform(_n_blade_files(make_lin_file, 3, 1.0))
        assert result.performed_transformation
        assert "form no three-blade triplet" not in caplog.text
