# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Shared fixtures for the viz tests (with a headless Matplotlib backend)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from typing import TYPE_CHECKING

import numpy as np
import pytest

from vane.campbell.builder import (
    CampbellDiagram,
    build_campbell,
    campbell_from_solutions,
)
from vane.modal.eigensolver import ModalSolution
from vane.modal.identifier import identify_modes

if TYPE_CHECKING:
    from vane.modal.branch import BranchGroup

_DESCS = [
    "ED 1st tower fore-aft (internal DOF index = DOF_TFA1), m",
    "ED drivetrain (internal DOF index = DOF_DrTr), rad",
]


def _solution(frequencies: list[float], shapes: np.ndarray) -> ModalSolution:
    """Return a ModalSolution with the given frequencies and mode-shape columns."""
    freq = np.array(frequencies, dtype=np.float64)
    n = freq.shape[0]
    return ModalSolution(
        eigenvalues=(-0.01 + 1j * freq).astype(np.complex128),
        natural_frequencies_hz=freq,
        damping_ratios=np.full(n, 0.02),
        damped_frequencies_hz=freq,
        mode_shapes=shapes.astype(np.complex128),
        full_eigenvectors=shapes.astype(np.complex128),
        n_rigid_body_modes=0,
        dof_descriptions=_DESCS,
    )


@pytest.fixture
def campbell_diagram() -> CampbellDiagram:
    """Return a two-track rotor-speed Campbell diagram with a 3P crossing."""
    shapes = np.array([[1.0, 0.0], [0.0, 1.0]])
    solutions = [_solution([1.5, 2.0], shapes), _solution([1.5, 2.1], shapes)]
    return campbell_from_solutions(solutions, [0.0, 60.0])


@pytest.fixture
def mode_solution() -> ModalSolution:
    """Return a single-mode solution with complex DOF components."""
    shapes = np.array([[1.0 + 0.2j], [0.3 - 0.1j]], dtype=np.complex128)
    return _solution([1.5], shapes)


@pytest.fixture
def branch_review_case() -> tuple[CampbellDiagram, list[BranchGroup]]:
    """Return a Campbell diagram and the branch groups of a frequency crossing."""
    distinct = np.array([[1.0, 0.0], [0.0, 1.0]])
    mixed = np.array([[1.0, 0.9], [0.9, 1.0]])
    solutions = [
        _solution([1.0, 2.0], distinct),
        _solution([1.5, 1.5], mixed),
        _solution([2.0, 1.0], distinct),
    ]
    result = identify_modes(solutions, strategy="simultaneous", branch_review=True)
    diagram = build_campbell(result, [0.0, 30.0, 60.0])
    return diagram, result.branch_groups


@pytest.fixture
def resolved_branch_case() -> tuple[CampbellDiagram, list[BranchGroup]]:
    """Return a diagram and a resolved group carrying a smooth alternative."""
    distinct = np.array([[1.0, 0.0], [0.0, 1.0]])
    mixed = np.array([[1.0, 0.7], [0.7, 1.0]])
    solutions = [
        _solution([1.0, 2.0], distinct),
        _solution([1.5, 1.5], mixed),
        _solution([1.0, 2.0], distinct),
    ]
    result = identify_modes(solutions, strategy="simultaneous", resolve=True)
    diagram = build_campbell(result, [0.0, 30.0, 60.0])
    return diagram, result.branch_groups
