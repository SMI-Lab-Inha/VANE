# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.io.state_selection: keeping and removing states by pattern."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from vane.config.dof_map import DofCategory
from vane.io import mbc3_transform, read_lin_file, select_states
from vane.modal.eigensolver import modes_from_mbc

if TYPE_CHECKING:
    import numpy.typing as npt

    from tests.conftest import ChannelSpec, LinFileFactory
    from vane.io.lin_reader import LinFile

_ED = "ED 1st "
_DISPLACEMENTS: list[ChannelSpec] = [
    (_ED + "tower fore-aft bending mode DOF (internal DOF index = DOF_TFA1), m",
     False, 2),
    (_ED + "tower side-to-side bending mode DOF (internal DOF index = DOF_TSS1), m",
     False, 2),
    (_ED + "flapwise bending-mode DOF of blade 1 (internal DOF index = DOF_BF(1,1)), m",
     True, 2),
    (_ED + "flapwise bending-mode DOF of blade 2 (internal DOF index = DOF_BF(2,1)), m",
     True, 2),
    (_ED + "flapwise bending-mode DOF of blade 3 (internal DOF index = DOF_BF(3,1)), m",
     True, 2),
]  # fmt: skip
_FIRST_ORDER: ChannelSpec = ("HD Radiation state 1, -", False, 1)
_STATES: list[ChannelSpec] = [
    *_DISPLACEMENTS,
    *[
        (
            d.replace("ED ", "ED First time derivative of ", 1).replace(", m", ", m/s"),
            rot,
            2,
        )
        for d, rot, _ in _DISPLACEMENTS
    ],
    _FIRST_ORDER,
]
_INPUTS: list[ChannelSpec] = [("ED Generator torque, Nm", False, 0)]
_OUTPUTS: list[ChannelSpec] = [
    ("ED TwrBsMyt, (kN-m)", False, 0),
    ("ED RootMyc1, (kN-m)", True, 0),
]
_N = len(_STATES)
_K = np.array([4.0, 4.5, 9.0, 9.0, 9.0])


def _matrices() -> tuple[npt.NDArray[np.float64], ...]:
    """Return a stable 11-state A and dense B, C, D with distinct entries."""
    a = np.zeros((_N, _N))
    a[0:5, 5:10] = np.eye(5)
    a[5:10, 0:5] = -np.diag(_K)
    a[5:10, 5:10] = -0.2 * np.eye(5)
    a[10, 10] = -3.0
    a[10, 0] = 0.1  # the radiation state is driven by the tower displacement
    b = np.arange(1.0, _N + 1.0).reshape(_N, 1)
    c = np.arange(1.0, 2 * _N + 1.0).reshape(2, _N)
    d = np.array([[0.5], [0.25]])
    return a, b, c, d


def _write(make_lin_file: LinFileFactory, azimuth: float = 0.0) -> LinFile:
    """Write and read one synthetic azimuth of the 11-state model."""
    a, b, c, d = _matrices()
    path = make_lin_file(
        states=_STATES,
        inputs=_INPUTS,
        outputs=_OUTPUTS,
        a=a,
        b=b,
        c=c,
        d=d,
        rotor_speed=1.0,
        azimuth=azimuth,
    )
    return read_lin_file(path)


class TestSelectStatesOnLinFile:
    """Selection on a raw ``.lin`` file, before the MBC3 transform."""

    def test_drop_by_pattern_removes_rows_and_columns(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Dropping the HD state removes its row and column of A, B and C."""
        lin = _write(make_lin_file)
        reduced = select_states(lin, drop=r"^HD ")
        keep = np.arange(_N - 1)
        a, b, c, d = _matrices()
        assert reduced.n_x == _N - 1
        np.testing.assert_array_equal(reduced.a, a[np.ix_(keep, keep)])
        np.testing.assert_array_equal(reduced.b, b[keep, :])
        np.testing.assert_array_equal(reduced.c, c[:, keep])
        np.testing.assert_array_equal(reduced.d, d)
        assert reduced.x.descriptions == lin.x.descriptions[:-1]
        assert len(reduced.xdot) == _N - 1
        assert reduced.n_u == lin.n_u and reduced.n_y == lin.n_y
        assert lin.n_x == _N  # the input is not modified

    def test_second_order_pairs_are_selected_together(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """A pattern matching only the velocity description removes both states."""
        lin = _write(make_lin_file)
        reduced = select_states(lin, drop=r"First time derivative of 1st tower side")
        assert reduced.n_x == _N - 2
        assert not any("DOF_TSS1" in desc for desc in reduced.x.descriptions)

    def test_keep_by_category(self, make_lin_file: LinFileFactory) -> None:
        """Keeping the tower categories retains the four tower states only."""
        lin = _write(make_lin_file)
        reduced = select_states(
            lin,
            keep_categories=[DofCategory.TOWER_FORE_AFT_1, "tower_side_side_1"],
        )
        assert reduced.n_x == 4
        assert all("DOF_T" in desc for desc in reduced.x.descriptions)
        assert reduced.b is not None and reduced.b.shape == (4, 1)
        assert reduced.c is not None and reduced.c.shape == (2, 4)

    def test_keep_then_drop(self, make_lin_file: LinFileFactory) -> None:
        """``drop`` is applied after ``keep``."""
        lin = _write(make_lin_file)
        reduced = select_states(lin, keep=r"\bED\b", drop=r"DOF_BF\(3")
        assert reduced.n_x == 8


class TestSelectStatesOnMBCResult:
    """Selection on the azimuth-averaged model, after the MBC3 transform."""

    def test_layout_and_eigenvalues_after_mbc(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Removing a DOF after MBC keeps the q2/q2dot/q1 layout and other modes."""
        sweep = [_write(make_lin_file, azimuth=psi) for psi in (0.0, 2.0, 4.0)]
        result = mbc3_transform(sweep, retain_per_azimuth=True)
        reduced = select_states(result, drop_categories=["tower_side_side_1"])
        assert (reduced.ndof2, reduced.ndof1) == (result.ndof2 - 1, result.ndof1)
        n = len(reduced.state_descriptions)
        assert reduced.avg_a is not None and reduced.avg_a.shape == (n, n)
        assert reduced.avg_b is not None and reduced.avg_b.shape == (n, 1)
        assert reduced.avg_c is not None and reduced.avg_c.shape == (2, n)
        np.testing.assert_array_equal(reduced.avg_d, result.avg_d)
        assert reduced.per_azimuth_a is not None
        assert all(m.shape == (n, n) for m in reduced.per_azimuth_a)
        assert len(reduced.mbc_coordinates) == len(result.mbc_coordinates) - 2
        solution = modes_from_mbc(reduced)
        frequencies = np.sort(solution.natural_frequencies_hz)
        expected = np.sqrt(_K[[0, 2]]) / (2 * np.pi)
        for value in expected:
            assert np.min(np.abs(frequencies - value)) < 0.02 * value


class TestSelectStatesErrors:
    """Invalid selections are rejected with a specific error."""

    def test_no_match_raises(self, make_lin_file: LinFileFactory) -> None:
        """A pattern that matches no state is an error, not a silent no-op."""
        lin = _write(make_lin_file)
        with pytest.raises(ValueError, match="matches no state"):
            select_states(lin, drop=r"BeamDyn")
        with pytest.raises(ValueError, match="matches no state"):
            select_states(lin, keep_categories=["platform_surge"])

    def test_empty_and_invalid_selections(self, make_lin_file: LinFileFactory) -> None:
        """Missing selectors, bad patterns, unknown classes and empty results fail."""
        lin = _write(make_lin_file)
        with pytest.raises(ValueError, match="at least one"):
            select_states(lin)
        with pytest.raises(ValueError, match="invalid regular expression"):
            select_states(lin, drop="(")
        with pytest.raises(ValueError, match="unknown DOF category"):
            select_states(lin, drop_categories=["not_a_category"])
        with pytest.raises(ValueError, match="removes every state"):
            select_states(lin, drop=".")
        with pytest.raises(TypeError, match="sequence"):
            select_states(lin, drop=[1])  # type: ignore[list-item]
        with pytest.raises(TypeError, match="LinFile or an MBCResult"):
            select_states(object(), drop="x")  # type: ignore[call-overload]
