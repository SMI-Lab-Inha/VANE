# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.campbell.builder — Campbell-diagram assembly."""

from __future__ import annotations

import numpy as np
import pytest

from vane.campbell.builder import build_campbell, campbell_from_solutions
from vane.modal.eigensolver import ModalSolution
from vane.modal.identifier import IdentificationResult, identify_modes

_DESC = "ED 1st tower fore-aft (internal DOF index = DOF_TFA1), m"


def _solution(frequency: float) -> ModalSolution:
    """Return a single-mode solution at a given natural frequency."""
    shapes = np.array([[1.0]], dtype=np.complex128)
    return ModalSolution(
        eigenvalues=np.array([-0.01 + 1j * frequency], dtype=np.complex128),
        natural_frequencies_hz=np.array([frequency]),
        damping_ratios=np.array([0.02]),
        damped_frequencies_hz=np.array([frequency]),
        mode_shapes=shapes,
        full_eigenvectors=shapes,
        n_rigid_body_modes=0,
        dof_descriptions=[_DESC],
    )


class TestBuildCampbell:
    """Tests for build_campbell and campbell_from_solutions."""

    def test_pairs_tracks_with_parameter(self) -> None:
        """A diagram pairs the track frequencies with the operating parameter."""
        diagram = campbell_from_solutions(
            [_solution(1.0), _solution(1.1), _solution(1.2)], [6.0, 9.0, 12.0]
        )
        assert len(diagram.tracks) == 1
        x, freq, damping = diagram.track_curve(diagram.tracks[0])
        np.testing.assert_allclose(x, [6.0, 9.0, 12.0])
        np.testing.assert_allclose(freq, [1.0, 1.1, 1.2])
        assert damping.shape == (3,)

    def test_parameter_name_default(self) -> None:
        """The operating parameter defaults to rotor speed in rpm."""
        diagram = campbell_from_solutions([_solution(1.0)], [10.0])
        assert diagram.parameter_name == "rotor_speed_rpm"
        assert diagram.n_operating_points == 1

    def test_custom_parameter_name(self) -> None:
        """A wind-speed Campbell diagram records the parameter name."""
        diagram = campbell_from_solutions(
            [_solution(1.0), _solution(1.0)],
            [8.0, 10.0],
            parameter_name="wind_speed",
        )
        assert diagram.parameter_name == "wind_speed"

    def test_length_mismatch_raises(self) -> None:
        """A parameter-value/operating-point count mismatch raises ValueError."""
        result = identify_modes([_solution(1.0), _solution(1.1)])
        with pytest.raises(ValueError, match="operating points"):
            build_campbell(result, [10.0])

    def test_empty_result(self) -> None:
        """An empty identification yields an empty diagram."""
        diagram = build_campbell(IdentificationResult([], 0), [])
        assert diagram.tracks == []
        assert diagram.n_operating_points == 0

    def test_non_finite_sweep_axis_rejected(self) -> None:
        """A non-finite parameter value is rejected before resonance analysis."""
        with pytest.raises(ValueError, match="must all be finite"):
            campbell_from_solutions([_solution(1.0), _solution(1.1)], [10.0, np.inf])

    def test_non_monotone_sweep_axis_rejected(self) -> None:
        """A non-increasing (or duplicate) sweep axis is rejected."""
        with pytest.raises(ValueError, match="strictly increasing"):
            campbell_from_solutions([_solution(1.0), _solution(1.1)], [10.0, 5.0])
        with pytest.raises(ValueError, match="strictly increasing"):
            campbell_from_solutions([_solution(1.0), _solution(1.1)], [10.0, 10.0])
