# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Scale-invariant biorthogonal modal-participation tests."""

from __future__ import annotations

import numpy as np
import pytest

from vane.modal.biorthogonal import (
    biorthogonal_participation,
    classify_state_subsystem,
)
from vane.modal.eigensolver import compute_modes


def _oscillator() -> np.ndarray:
    """Return a damped, non-normal second-order oscillator."""
    return np.array([[0.0, 1.0], [-4.0, -0.2]], dtype=np.float64)


def test_complex_factors_close_and_magnitudes_sum_to_one() -> None:
    """Left--right factors satisfy their complex and magnitude normalisations."""
    solution = compute_modes(
        _oscillator(),
        ndof2=1,
        ndof1=0,
        descriptions=["ED Platform pitch DOF", "ED Platform pitch rate DOF"],
    )

    result = biorthogonal_participation(solution)

    assert np.all(result.valid)
    np.testing.assert_allclose(np.sum(result.complex_factors, axis=0), 1.0, atol=1e-12)
    np.testing.assert_allclose(
        np.sum(result.magnitude_fractions, axis=0), 1.0, atol=1e-12
    )
    np.testing.assert_allclose(result.closure_error, 0.0, atol=1e-12)
    assert result.subsystem_names == ("platform",)
    np.testing.assert_allclose(result.subsystem_fractions, 1.0, atol=1e-12)


def test_participation_is_invariant_to_diagonal_state_scaling() -> None:
    """Changing state units does not alter left--right participation shares."""
    a = _oscillator()
    descriptions = ["ED Platform pitch DOF", "ED Platform pitch rate DOF"]
    reference = biorthogonal_participation(
        compute_modes(a, 1, 0, descriptions=descriptions)
    )
    scaling = np.diag([1000.0, 0.01])
    scaled_a = np.linalg.solve(scaling, a @ scaling)
    scaled = biorthogonal_participation(
        compute_modes(scaled_a, 1, 0, descriptions=descriptions)
    )

    np.testing.assert_allclose(
        scaled.complex_factors, reference.complex_factors, rtol=1e-11, atol=1e-11
    )
    np.testing.assert_allclose(
        scaled.magnitude_fractions,
        reference.magnitude_fractions,
        rtol=1e-11,
        atol=1e-11,
    )


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("ED Platform horizontal surge translation DOF", "platform"),
        ("ED 1st tower fore-aft bending mode DOF", "tower"),
        ("ED Variable speed generator DOF", "drivetrain"),
        (
            "BD_1 finite element node 2 (number of elements = 1; element order = 5) "
            "translational displacement in X, m",
            "blades",
        ),
        ("BD 1st flap mode", "blades"),
        (
            "First time derivative of BD_2 finite element node 3 (number of "
            "elements = 1; element order = 5) rotational displacement in Z, rad/s",
            "blades",
        ),
        ("First time derivative of ED 1st tower fore-aft bending mode DOF", "tower"),
        ("AD DBEMT induced velocity state", "aerodynamics"),
        ("HD RdtnPtfmSg7", "hydrodynamics"),
        ("MD Line1 internal state", "mooring"),
        ("SrvD pitch-controller state", "controller"),
    ],
)
def test_state_subsystem_classification(description: str, expected: str) -> None:
    """Common OpenFAST state prefixes map to engineering subsystems."""
    assert classify_state_subsystem(description) == expected


def test_missing_left_vectors_is_rejected() -> None:
    """Old externally constructed solutions fail explicitly rather than mislead."""
    solution = compute_modes(_oscillator(), 1, 0, descriptions=["a", "b"])
    solution.full_left_eigenvectors = np.empty((0, 0), dtype=np.complex128)
    with pytest.raises(ValueError, match="no left eigenvectors"):
        biorthogonal_participation(solution)
