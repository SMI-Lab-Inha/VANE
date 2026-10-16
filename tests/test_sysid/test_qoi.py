# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.sysid.qoi — the quantity-of-interest (load) map."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from vane.config.output_map import OutputCategory
from vane.modal.eigensolver import ModalSolution
from vane.sysid.qoi import build_qoi_map, qoi_map_from_mbc
from vane.sysid.state_space import StateSpace

# A 4-state, 2-input, 3-output toy system: a fairlead-tension load, a tower-base
# moment load, and a (non-load) platform-motion channel.
_OUTPUTS = ["MAP T[1], [N]", "ED TwrBsMyt, (kN-m)", "ED PtfmSurge, (m)"]
_INPUTS = ["HD wave elevation, m", "ED Generator torque, Nm"]
_STATES = ["s0", "s1", "s2", "s3"]


def _system() -> StateSpace:
    """Return a small state-space system with named loads and inputs."""
    c = np.array(
        [
            [1.0, 0.0, 2.0, 0.0],  # fairlead tension: states 0 and 2
            [0.0, 3.0, 0.0, 0.0],  # tower-base moment: state 1
            [0.0, 0.0, 0.0, 1.0],  # platform surge: state 3
        ],
        dtype=np.float64,
    )
    d = np.array([[0.5, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
    return StateSpace(
        a=np.eye(4),
        b=np.ones((4, 2)),
        c=c,
        d=d,
        state_names=_STATES,
        input_names=_INPUTS,
        output_names=_OUTPUTS,
    )


def _solution() -> ModalSolution:
    """Return a two-mode solution whose full eigenvectors span the 4 states."""
    phi = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.0],
            [0.0, 0.2],
        ],
        dtype=np.complex128,
    )
    freq = np.array([0.1, 0.8], dtype=np.float64)
    return ModalSolution(
        eigenvalues=(-0.01 + 1j * freq).astype(np.complex128),
        natural_frequencies_hz=freq,
        damping_ratios=np.array([0.02, 0.05]),
        damped_frequencies_hz=freq,
        mode_shapes=phi,
        full_eigenvectors=phi,
        n_rigid_body_modes=0,
        dof_descriptions=_STATES,
    )


class TestBuildQoiMap:
    """Tests for build_qoi_map."""

    def test_loads_only_filters_non_loads(self) -> None:
        """By default only the load channels are included."""
        qoi = build_qoi_map(_system())
        names = [s.name for s in qoi.sensitivities]
        assert names == ["MAP T[1], [N]", "ED TwrBsMyt, (kN-m)"]
        assert all(s.is_load for s in qoi.sensitivities)

    def test_include_all_channels(self) -> None:
        """loads_only=False keeps the non-load channels too."""
        qoi = build_qoi_map(_system(), loads_only=False)
        assert len(qoi.sensitivities) == 3
        surge = qoi.get("ED PtfmSurge, (m)")
        assert surge.info.category is OutputCategory.PLATFORM_MOTION
        assert not surge.is_load

    def test_state_and_input_sensitivities(self) -> None:
        """The C and D rows are exposed for each output."""
        qoi = build_qoi_map(_system())
        fair = qoi.get("MAP T[1], [N]")
        np.testing.assert_allclose(fair.state_sensitivity, [1.0, 0.0, 2.0, 0.0])
        assert fair.input_sensitivity is not None
        np.testing.assert_allclose(fair.input_sensitivity, [0.5, 0.0])

    def test_modal_residues_equal_c_phi(self) -> None:
        """Modal residues are exactly the rows of C @ Phi."""
        system, solution = _system(), _solution()
        assert system.c is not None
        qoi = build_qoi_map(system, solution)
        expected = system.c @ solution.full_eigenvectors  # (3 outputs, 2 modes)
        fair = qoi.get("MAP T[1], [N]")
        assert fair.modal_residues is not None
        np.testing.assert_allclose(fair.modal_residues, expected[0])
        moment = qoi.get("ED TwrBsMyt, (kN-m)")
        assert moment.modal_residues is not None
        np.testing.assert_allclose(moment.modal_residues, expected[1])

    def test_modal_residues_match_hand_computed_values(self) -> None:
        """The residues equal the analytically-known C @ Phi, load and non-load.

        With C and Phi fixed by the fixtures, C @ Phi is known exactly, so this checks
        the "which modes drive this load" residue independently of the implementation's
        eigenvectors:

        * fairlead tension C=[1, 0, 2, 0] -> mode0 = 1*1 + 2*0.5 = 2.0, mode1 = 0;
        * tower-base moment C=[0, 3, 0, 0] -> mode0 = 0, mode1 = 3*1 = 3.0;
        * platform surge (a *non-load* output) C=[0, 0, 0, 1] -> mode0 = 0,
          mode1 = 1*0.2 = 0.2.
        """
        qoi = build_qoi_map(_system(), _solution(), loads_only=False)
        fair = qoi.get("MAP T[1], [N]")
        assert fair.modal_residues is not None
        np.testing.assert_allclose(fair.modal_residues, [2.0, 0.0])
        moment = qoi.get("ED TwrBsMyt, (kN-m)")
        assert moment.modal_residues is not None
        np.testing.assert_allclose(moment.modal_residues, [0.0, 3.0])
        surge = qoi.get("ED PtfmSurge, (m)")  # the non-load channel
        assert surge.modal_residues is not None
        np.testing.assert_allclose(surge.modal_residues, [0.0, 0.2])

    def test_dominant_modes_ranking(self) -> None:
        """dominant_modes orders modes by residue magnitude."""
        qoi = build_qoi_map(_system(), _solution())
        # Fairlead row C=[1,0,2,0]; Phi cols -> residue mode0 = 1*1 + 2*0.5 = 2.0,
        # mode1 = 0 -> mode 0 dominates.
        fair = qoi.get("MAP T[1], [N]")
        assert fair.dominant_modes(1).tolist() == [0]
        # Tower moment C=[0,3,0,0]; residue mode0=0, mode1=3 -> mode 1 dominates.
        moment = qoi.get("ED TwrBsMyt, (kN-m)")
        assert moment.dominant_modes(1).tolist() == [1]

    def test_no_modal_solution_leaves_residues_none(self) -> None:
        """Without a solution the residues and dominant modes are empty/None."""
        qoi = build_qoi_map(_system())
        fair = qoi.get("MAP T[1], [N]")
        assert fair.modal_residues is None
        assert fair.dominant_modes().size == 0
        assert qoi.mode_frequencies_hz is None


class TestModalParticipationTable:
    """Tests for the long-format modal participation table."""

    def test_table_columns_and_normalization(self) -> None:
        """The table has one row per (load, mode) with a per-load normalization."""
        qoi = build_qoi_map(_system(), _solution())
        table = qoi.modal_participation_table()
        assert set(table.columns) >= {
            "load",
            "category",
            "mode",
            "frequency_hz",
            "residue_magnitude",
            "normalized",
        }
        # 2 loads x 2 modes.
        assert len(table) == 4
        # Each load's strongest mode is normalized to 1.
        for _, group in table.groupby("load"):
            np.testing.assert_allclose(group["normalized"].max(), 1.0)

    def test_table_empty_without_residues(self) -> None:
        """Without modal residues the table is empty."""
        table = build_qoi_map(_system()).modal_participation_table()
        assert table.empty


class TestQoiMapValidation:
    """Tests for argument validation."""

    def test_missing_c_raises(self) -> None:
        """A system without C cannot produce a QoI map."""
        system = StateSpace(a=np.eye(2))
        with pytest.raises(ValueError, match="no C matrix"):
            build_qoi_map(system)

    def test_missing_output_names_raises(self) -> None:
        """A system without output names cannot classify channels."""
        system = StateSpace(a=np.eye(2), c=np.ones((1, 2)))
        with pytest.raises(ValueError, match="no output_names"):
            build_qoi_map(system)

    def test_eigenvector_state_mismatch_raises(self) -> None:
        """A solution whose eigenvectors do not span the states is rejected."""
        bad = _solution()
        bad_vectors: npt.NDArray[np.complex128] = np.ones((3, 2), dtype=np.complex128)
        bad.full_eigenvectors = bad_vectors
        with pytest.raises(ValueError, match="span"):
            build_qoi_map(_system(), bad)


class TestRotatingOutputTriplet:
    """A transformed blade-load output triplet must not be labeled per-blade."""

    def test_transformed_outputs_relabeled(self, make_lin_file: object) -> None:
        """MBC output coordinates clear the misleading per-blade component."""
        from vane.io.lin_reader import read_lin_file
        from vane.io.mbc_transform import mbc3_transform
        from vane.sysid.state_space import state_space_from_mbc

        disp = [
            (
                f"ED 1st flapwise bending-mode DOF of blade {b} "
                f"(internal DOF index = DOF_BF({b},1)), m",
                True,
                2,
            )
            for b in (1, 2, 3)
        ]
        vel = [(f"ED First time derivative of {d}", True, 2) for d, _, _ in disp]
        states = disp + vel
        a = np.zeros((6, 6), dtype=np.float64)
        a[0:3, 3:6] = np.eye(3)
        a[3:6, 0:3] = -4.0 * np.eye(3)
        a[3:6, 3:6] = -0.1 * np.eye(3)
        outputs = [(f"ED RootMxb{b}, (kN-m)", True, 0) for b in (1, 2, 3)]
        c = np.zeros((3, 6), dtype=np.float64)
        c[0:3, 0:3] = np.eye(3)

        path = make_lin_file(  # type: ignore[operator]
            states=states, outputs=outputs, a=a, c=c
        )
        mbc = mbc3_transform([read_lin_file(path)])
        assert mbc.performed_transformation
        assert mbc.output_mbc_coordinates == ["collective", "cosine", "sine"]

        # state_space_from_mbc carries the coordinates, so build_qoi_map is safe by
        # default — no explicit output_mbc_coordinates argument required.
        system = state_space_from_mbc(mbc)
        assert system.output_mbc_coordinates == ["collective", "cosine", "sine"]
        qoi = build_qoi_map(system)
        assert {s.mbc_coordinate for s in qoi.loads()} == {
            "collective",
            "cosine",
            "sine",
        }
        assert all(s.info.component is None for s in qoi.loads())
        # An explicit override is still honored.
        plain = build_qoi_map(system, output_mbc_coordinates=[""] * 3)
        assert any(s.info.component is not None for s in plain.loads())

    def test_coordinate_length_mismatch_raises(self) -> None:
        """A mismatched output_mbc_coordinates length is rejected."""
        with pytest.raises(ValueError, match="output_mbc_coordinates length"):
            build_qoi_map(_system(), output_mbc_coordinates=["collective"])

    def test_qoi_map_from_mbc_is_default_safe(self, make_lin_file: object) -> None:
        """qoi_map_from_mbc forwards the coordinates without a manual argument."""
        from vane.io.lin_reader import read_lin_file
        from vane.io.mbc_transform import mbc3_transform

        disp = [
            (
                f"ED 1st flapwise bending-mode DOF of blade {b} "
                f"(internal DOF index = DOF_BF({b},1)), m",
                True,
                2,
            )
            for b in (1, 2, 3)
        ]
        vel = [(f"ED First time derivative of {d}", True, 2) for d, _, _ in disp]
        a = np.zeros((6, 6), dtype=np.float64)
        a[0:3, 3:6] = np.eye(3)
        a[3:6, 0:3] = -4.0 * np.eye(3)
        a[3:6, 3:6] = -0.1 * np.eye(3)
        outputs = [(f"ED RootMxc{b}, (kN-m)", True, 0) for b in (1, 2, 3)]
        c = np.zeros((3, 6), dtype=np.float64)
        c[0:3, 0:3] = np.eye(3)
        path = make_lin_file(  # type: ignore[operator]
            states=disp + vel, outputs=outputs, a=a, c=c
        )
        mbc = mbc3_transform([read_lin_file(path)])
        # No manual output_mbc_coordinates: the convenience forwards them.
        qoi = qoi_map_from_mbc(mbc)
        assert {s.mbc_coordinate for s in qoi.loads()} == {
            "collective",
            "cosine",
            "sine",
        }
        assert all(s.info.component is None for s in qoi.loads())


class TestThreadingFromMbc:
    """Output descriptions thread through MBC into the QoI map (synthetic .lin)."""

    def test_outputs_reach_qoi_map(self, make_lin_file: object) -> None:
        """A synthetic .lin with load outputs yields a populated QoI map."""
        from vane.io.lin_reader import read_lin_file
        from vane.io.mbc_transform import mbc3_transform
        from vane.modal.eigensolver import modes_from_mbc
        from vane.sysid.state_space import state_space_from_mbc

        states = [
            ("ED 1st tower fore-aft (internal DOF index = DOF_TFA1), m", False, 2),
            (
                "ED First time derivative of 1st tower fore-aft (DOF_TFA1), m/s",
                False,
                1,
            ),
        ]
        inputs = [("ED Generator torque, Nm", False, 0)]
        outputs = [("ED TwrBsMyt, (kN-m)", False, 0), ("ED PtfmSurge, (m)", False, 0)]
        a = np.array([[0.0, 1.0], [-4.0, -0.2]], dtype=np.float64)
        b = np.array([[0.0], [1.0]], dtype=np.float64)
        c = np.array([[2.0, 0.0], [0.0, 0.0]], dtype=np.float64)
        d = np.zeros((2, 1), dtype=np.float64)
        path = make_lin_file(  # type: ignore[operator]
            states=states, inputs=inputs, outputs=outputs, a=a, b=b, c=c, d=d
        )
        mbc = mbc3_transform([read_lin_file(path)])
        assert mbc.output_descriptions == [o[0] for o in outputs]
        system = state_space_from_mbc(mbc)
        assert system.output_names == [o[0] for o in outputs]
        qoi = build_qoi_map(system, modes_from_mbc(mbc))
        # Only the tower-base moment is a load.
        assert [s.name for s in qoi.loads()] == ["ED TwrBsMyt, (kN-m)"]
