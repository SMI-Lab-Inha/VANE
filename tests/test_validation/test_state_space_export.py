# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Validation of state-space export fidelity (the digital-twin round trip).

A modal state-space realization must carry exactly the modes it was built from: a set
of prescribed natural frequencies and damping ratios, realized as a block-diagonal
state-space model, must re-extract to the same frequencies and damping when its state
matrix is solved again. This guarantees the exported digital-twin model is faithful to
the identified modes.
"""

from __future__ import annotations

import numpy as np

from vane.benchmark import synthetic_system
from vane.modal.eigensolver import compute_modes
from vane.sysid.modal_to_ss import modal_state_space, modal_state_space_from_solution

_FREQUENCIES_HZ = np.array([0.5, 1.5, 3.0])
_DAMPING_RATIOS = np.array([0.01, 0.02, 0.05])


def _conjugate_representatives(
    frequencies_hz: np.ndarray, damping_ratios: np.ndarray
) -> np.ndarray:
    """Return the upper-half-plane eigenvalues for the given modal parameters."""
    omega = 2.0 * np.pi * frequencies_hz
    return -damping_ratios * omega + 1j * omega * np.sqrt(1.0 - damping_ratios**2)


class TestStateSpaceExportFidelity:
    """The exported state space preserves the modal parameters."""

    def test_realization_preserves_eigenvalues(self) -> None:
        """A modal realization re-extracts the frequencies and damping it encodes."""
        eigenvalues = _conjugate_representatives(_FREQUENCIES_HZ, _DAMPING_RATIOS)
        state_space = modal_state_space(eigenvalues)

        recovered = compute_modes(state_space.a, ndof2=_FREQUENCIES_HZ.size, ndof1=0)
        order = np.argsort(recovered.natural_frequencies_hz)
        np.testing.assert_allclose(
            recovered.natural_frequencies_hz[order], _FREQUENCIES_HZ, rtol=1e-9
        )
        np.testing.assert_allclose(
            recovered.damping_ratios[order], _DAMPING_RATIOS, rtol=1e-6
        )

    def test_round_trip_from_solution(self) -> None:
        """Solution -> state space -> solution preserves the modes."""
        truth = synthetic_system(_FREQUENCIES_HZ.tolist(), _DAMPING_RATIOS.tolist())
        solution = compute_modes(truth.a, ndof2=3, ndof1=0)
        state_space = modal_state_space_from_solution(solution)

        recovered = compute_modes(state_space.a, ndof2=3, ndof1=0)
        np.testing.assert_allclose(
            np.sort(recovered.natural_frequencies_hz),
            np.sort(solution.natural_frequencies_hz),
            rtol=1e-9,
        )
        np.testing.assert_allclose(
            recovered.damping_ratios[np.argsort(recovered.natural_frequencies_hz)],
            solution.damping_ratios[np.argsort(solution.natural_frequencies_hz)],
            rtol=1e-6,
        )

    def test_realization_state_matrix_is_real(self) -> None:
        """The modal realization is a real state-space model."""
        eigenvalues = _conjugate_representatives(_FREQUENCIES_HZ, _DAMPING_RATIOS)
        state_space = modal_state_space(eigenvalues)
        assert np.isrealobj(state_space.a)
        assert state_space.a.shape == (
            2 * _FREQUENCIES_HZ.size,
            2 * _FREQUENCIES_HZ.size,
        )
