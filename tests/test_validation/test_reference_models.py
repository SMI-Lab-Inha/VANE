# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Reference-model validation: a canonical land turbine through the study pipeline.

A multi-degree-of-freedom system with documented baseline modes — a tower fore-aft
mode, a drivetrain mode, and an isotropic three-bladed flap triplet — is generated
across a rotor-speed sweep and run end to end through the provenance-tracked study
API. The recovered Campbell diagram is checked against the documented baselines: the
flat tower and drivetrain lines, and the blade collective/regressive/progressive
split. This exercises discovery-to-provenance on a realistic, reproducible case
without any proprietary turbine data (the model is generated programmatically).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from vane.io.lin_reader import read_lin_file
from vane.io.mbc_transform import mbc3_transform
from vane.modal.eigensolver import modes_from_mbc
from vane.modal.labeler import label_solution
from vane.study import run_study

if TYPE_CHECKING:
    from tests.conftest import ChannelSpec, LinFileFactory
    from vane.io.lin_reader import LinFile

_TWO_PI = 2.0 * np.pi

# Documented baseline natural frequencies (Hz).
_F_TOWER = 0.32
_F_DRIVE = 1.70
_F_BLADE = 0.69
_ROTOR_SPEEDS = (0.5, 1.0, 1.5)  # rad/s


def _reference_states() -> list[ChannelSpec]:
    """Tower, drivetrain, and three rotating blade-flap DOFs, plus derivatives."""
    disp: list[ChannelSpec] = [
        ("ED 1st tower fore-aft (internal DOF index = DOF_TFA1), m", False, 2),
        ("ED drivetrain torsion (internal DOF index = DOF_DrTr), rad", False, 2),
    ]
    disp += [
        (
            f"ED 1st flapwise bending-mode DOF of blade {b} "
            f"(internal DOF index = DOF_BF({b},1)), m",
            True,
            2,
        )
        for b in (1, 2, 3)
    ]
    vel: list[ChannelSpec] = [
        (f"ED First time derivative of {desc}", rot, 2) for desc, rot, _ in disp
    ]
    return disp + vel


def _reference_a() -> np.ndarray:
    """Return the 10x10 state matrix of the canonical land turbine."""
    omega = _TWO_PI * np.array(
        [_F_TOWER, _F_DRIVE, _F_BLADE, _F_BLADE, _F_BLADE], dtype=np.float64
    )
    zeta = np.array([0.02, 0.02, 0.01, 0.01, 0.01], dtype=np.float64)
    n = omega.shape[0]
    a = np.zeros((2 * n, 2 * n), dtype=np.float64)
    a[:n, n:] = np.eye(n)
    a[n:, :n] = -np.diag(omega**2)
    a[n:, n:] = -np.diag(2.0 * zeta * omega)
    return a


def _operating_point(make_lin_file: LinFileFactory, omega: float) -> list[LinFile]:
    """Build a three-azimuth operating point at a given rotor speed."""
    states = _reference_states()
    a = _reference_a()
    return [
        read_lin_file(make_lin_file(states=states, a=a, rotor_speed=omega, azimuth=psi))
        for psi in (0.0, 2.0, 4.0)
    ]


class TestReferenceLandTurbine:
    """Validate the canonical land turbine end to end."""

    def test_baseline_frequencies_and_provenance(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """The study recovers the documented baselines and records provenance."""
        operating_points = [
            _operating_point(make_lin_file, omega) for omega in _ROTOR_SPEEDS
        ]
        study = run_study(operating_points)

        # Provenance: 3 operating points, 3 azimuths each.
        assert len(study.provenance.operating_points) == 3
        assert len(study.provenance.source_files) == 9
        assert study.provenance.vane_version

        diagram = study.pipeline.campbell
        assert len(diagram.tracks) == 5
        assert all(len(track.operating_points) == 3 for track in diagram.tracks)

        # At every rotor speed the five modes are the documented baselines: flat
        # tower and drivetrain lines, and the blade collective with the
        # regressive/progressive pair split by the rotor speed.
        for index, omega in enumerate(_ROTOR_SPEEDS):
            split = omega / _TWO_PI
            expected = sorted(
                [_F_TOWER, _F_BLADE - split, _F_BLADE, _F_BLADE + split, _F_DRIVE]
            )
            recovered = sorted(
                track.natural_frequencies_hz[index] for track in diagram.tracks
            )
            np.testing.assert_allclose(recovered, expected, rtol=2e-2)

    def test_blade_modes_are_taxonomically_labelled(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """The blade triplet is labelled collective/regressive/progressive."""
        solution = modes_from_mbc(
            mbc3_transform(_operating_point(make_lin_file, _ROTOR_SPEEDS[-1]))
        )
        multiblades = {
            label.multiblade
            for label in label_solution(solution)
            if label.multiblade is not None
        }
        assert {"collective", "regressive", "progressive"} <= multiblades
