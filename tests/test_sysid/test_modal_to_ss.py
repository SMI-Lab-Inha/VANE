# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.sysid.modal_to_ss — real modal realization."""

from __future__ import annotations

import numpy as np
import pytest

from vane.sysid.modal_to_ss import modal_state_space, modal_state_space_from_solution


class TestModalStateSpace:
    """Tests for modal_state_space."""

    def test_block_eigenvalues_match_modes(self) -> None:
        """The realization's eigenvalues are the modes and their conjugates."""
        eigenvalues = np.array([-0.2 + 2.0j, -0.5 + 5.0j], dtype=np.complex128)
        system = modal_state_space(eigenvalues)
        realized = np.sort_complex(np.linalg.eigvals(system.a))
        expected = np.sort_complex(np.concatenate([eigenvalues, eigenvalues.conj()]))
        np.testing.assert_allclose(realized, expected, atol=1e-10)

    def test_state_count_is_twice_modes(self) -> None:
        """Each mode contributes two real states."""
        system = modal_state_space(np.array([-0.1 + 1j, -0.2 + 3j]))
        assert system.n_states == 4

    def test_output_matrix_shape(self) -> None:
        """With mode shapes, C maps modal states to outputs."""
        eigenvalues = np.array([-0.2 + 2.0j])
        shapes = np.array([[1.0 + 0.5j], [0.5 - 0.2j], [0.1j]], dtype=np.complex128)
        system = modal_state_space(eigenvalues, shapes)
        assert system.c is not None
        assert system.c.shape == (3, 2)

    def test_mode_shape_mismatch_raises(self) -> None:
        """A mode-shape/eigenvalue count mismatch raises ValueError."""
        with pytest.raises(ValueError, match="columns"):
            modal_state_space(
                np.array([-0.2 + 2j]), np.ones((3, 2), dtype=np.complex128)
            )

    def test_empty_modes(self) -> None:
        """No modes yields an empty system."""
        system = modal_state_space(np.empty(0, dtype=np.complex128))
        assert system.n_states == 0

    def test_two_dimensional_eigenvalues_raises(self) -> None:
        """A 2-D eigenvalue array is rejected."""
        with pytest.raises(ValueError, match="1-D"):
            modal_state_space(np.zeros((2, 2), dtype=np.complex128))

    def test_one_dimensional_mode_shapes_raises(self) -> None:
        """A 1-D mode-shape array is rejected."""
        with pytest.raises(ValueError, match="2-D"):
            modal_state_space(np.array([-0.2 + 2j]), np.ones(3, dtype=np.complex128))


class TestModalFromSolution:
    """Tests for modal_state_space_from_solution."""

    def test_from_solution_round_trips_eigenvalues(self) -> None:
        """The realization recovers the solution's modal eigenvalues."""
        from vane.modal.eigensolver import compute_modes

        a = np.array([[0.0, 1.0], [-4.0, -0.4]])
        solution = compute_modes(a, ndof2=1, ndof1=0, descriptions=["x", "v"])
        system = modal_state_space_from_solution(solution)
        realized = np.linalg.eigvals(system.a)
        # The single under-damped mode and its conjugate reproduce A's spectrum.
        np.testing.assert_allclose(
            np.sort_complex(realized),
            np.sort_complex(np.linalg.eigvals(a)),
            atol=1e-10,
        )
        assert system.output_names == ["x"]
