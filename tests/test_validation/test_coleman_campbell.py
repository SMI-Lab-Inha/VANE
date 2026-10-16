# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""End-to-end validation against the analytical Coleman (MBC) frequency split.

For an isotropic three-bladed rotor of identical blade-flap oscillators at natural
frequency ``w0``, the multi-blade-coordinate transform yields, at rotor speed
``Omega``, a collective mode at ``w0`` and a regressive/progressive pair at
``w0 -/+ Omega`` (rad/s). Sweeping ``Omega`` therefore produces three Campbell lines:
a flat collective line, a falling regressive line, and a rising progressive line.

This exercises the whole pipeline — MBC transform, eigen-analysis, cross-operating
-point tracking, and Campbell assembly — and checks the result against that closed
-form prediction, not just internal consistency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vane.campbell.builder import build_campbell
from vane.io.lin_reader import read_lin_file
from vane.io.mbc_transform import mbc3_transform
from vane.modal.eigensolver import modes_from_mbc
from vane.modal.identifier import identify_modes
from vane.modal.labeler import label_solution

if TYPE_CHECKING:
    from tests.conftest import LinFileFactory
    from vane.modal.eigensolver import ModalSolution

_TWO_PI = 2.0 * np.pi
_W0 = 3.0  # blade-flap natural frequency, rad/s
_ROTOR_SPEEDS = (1.0, 1.5, 2.0)  # rad/s


def _blade_states() -> list[tuple[str, bool, int]]:
    disp: list[tuple[str, bool, int]] = [
        (
            f"ED 1st flapwise bending-mode DOF of blade {b} "
            f"(internal DOF index = DOF_BF({b},1)), m",
            True,
            2,
        )
        for b in (1, 2, 3)
    ]
    return disp + [
        (f"ED First time derivative of {desc}", True, 2) for desc, _, _ in disp
    ]


def _blade_a() -> np.ndarray:
    a = np.zeros((6, 6), dtype=np.float64)
    a[0:3, 3:6] = np.eye(3)
    a[3:6, 0:3] = -(_W0**2) * np.eye(3)
    a[3:6, 3:6] = -2.0 * 0.02 * _W0 * np.eye(3)
    return a


def _solution_at(make_lin_file: LinFileFactory, omega: float) -> ModalSolution:
    states = _blade_states()
    a = _blade_a()
    files = [
        read_lin_file(make_lin_file(states=states, a=a, rotor_speed=omega, azimuth=psi))
        for psi in (0.0, 2.0, 4.0)
    ]
    return modes_from_mbc(mbc3_transform(files))


class TestColemanCampbell:
    """Validate the Campbell diagram against the analytical Coleman split."""

    def test_split_matches_coleman_prediction(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """At each rotor speed the modes are at {w0 - Omega, w0, w0 + Omega}."""
        solutions = [_solution_at(make_lin_file, omega) for omega in _ROTOR_SPEEDS]
        diagram = build_campbell(
            identify_modes(solutions),
            list(_ROTOR_SPEEDS),
            parameter_name="rotor_speed_rad_s",
        )
        assert len(diagram.tracks) == 3
        assert all(len(track.operating_points) == 3 for track in diagram.tracks)

        for index, omega in enumerate(_ROTOR_SPEEDS):
            frequencies = sorted(
                track.natural_frequencies_hz[index] for track in diagram.tracks
            )
            expected = [
                (_W0 - omega) / _TWO_PI,
                _W0 / _TWO_PI,
                (_W0 + omega) / _TWO_PI,
            ]
            np.testing.assert_allclose(frequencies, expected, rtol=2e-2)

    def test_line_trends_are_regressive_collective_progressive(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """The three lines fall, stay flat, and rise with rotor speed."""
        solutions = [_solution_at(make_lin_file, omega) for omega in _ROTOR_SPEEDS]
        diagram = build_campbell(identify_modes(solutions), list(_ROTOR_SPEEDS))
        regressive, collective, progressive = sorted(
            diagram.tracks, key=lambda track: track.natural_frequencies_hz[0]
        )
        assert (
            regressive.natural_frequencies_hz[-1] < regressive.natural_frequencies_hz[0]
        )
        assert (
            progressive.natural_frequencies_hz[-1]
            > progressive.natural_frequencies_hz[0]
        )
        np.testing.assert_allclose(
            collective.natural_frequencies_hz,
            _W0 / _TWO_PI,
            rtol=2e-2,
        )

    def test_modes_are_labeled_collective_regressive_progressive(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """The MBC taxonomy labels the three modes at a sampled rotor speed."""
        labels = label_solution(_solution_at(make_lin_file, _ROTOR_SPEEDS[-1]))
        assert {label.multiblade for label in labels} == {
            "collective",
            "regressive",
            "progressive",
        }
