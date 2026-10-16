# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Validation against the closed-form Euler-Bernoulli cantilever beam.

A uniform cantilever beam has the analytical natural frequencies

.. math::

    f_n = \\frac{(\\beta_n L)^2}{2\\pi} \\sqrt{\\frac{EI}{\\rho A L^4}},

with :math:`\\beta_n L = 1.875104, 4.694091, 7.854757, \\dots` the roots of
:math:`\\cos(\\beta L)\\cosh(\\beta L) + 1 = 0`. A consistent-mass finite-element model
of the beam is assembled here, cast into the state matrix ``[[0, I], [-M^-1 K, 0]]``,
and handed to the VANE eigen-core; the recovered lowest frequencies must match the
analytical values. This is the canonical structural-dynamics benchmark, independent
of any VANE-internal construction.
"""

from __future__ import annotations

import numpy as np

from vane.modal.eigensolver import compute_modes

# Roots (beta_n * L) of the cantilever characteristic equation.
_BETA_L = (1.8751040687, 4.6940911330, 7.8547574382)
_N_ELEMENTS = 24
_LENGTH = 1.0
_FLEXURAL_RIGIDITY = 1.0  # EI
_MASS_PER_LENGTH = 1.0  # rho * A


def _element_matrices(length: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the 4x4 stiffness and consistent-mass matrices of a beam element."""
    le = length
    stiffness = (_FLEXURAL_RIGIDITY / le**3) * np.array(
        [
            [12.0, 6.0 * le, -12.0, 6.0 * le],
            [6.0 * le, 4.0 * le**2, -6.0 * le, 2.0 * le**2],
            [-12.0, -6.0 * le, 12.0, -6.0 * le],
            [6.0 * le, 2.0 * le**2, -6.0 * le, 4.0 * le**2],
        ]
    )
    mass = (_MASS_PER_LENGTH * le / 420.0) * np.array(
        [
            [156.0, 22.0 * le, 54.0, -13.0 * le],
            [22.0 * le, 4.0 * le**2, 13.0 * le, -3.0 * le**2],
            [54.0, 13.0 * le, 156.0, -22.0 * le],
            [-13.0 * le, -3.0 * le**2, -22.0 * le, 4.0 * le**2],
        ]
    )
    return stiffness, mass


def _assemble_cantilever() -> tuple[np.ndarray, np.ndarray]:
    """Assemble the global stiffness/mass of a clamped-free beam (root fixed)."""
    n_nodes = _N_ELEMENTS + 1
    n_dof = 2 * n_nodes
    stiffness = np.zeros((n_dof, n_dof), dtype=np.float64)
    mass = np.zeros((n_dof, n_dof), dtype=np.float64)
    element_k, element_m = _element_matrices(_LENGTH / _N_ELEMENTS)
    for element in range(_N_ELEMENTS):
        base = 2 * element
        dofs = np.array([base, base + 1, base + 2, base + 3])
        grid = np.ix_(dofs, dofs)
        stiffness[grid] += element_k
        mass[grid] += element_m
    # Clamp the root node: drop its transverse and rotational DOFs (0 and 1).
    free = slice(2, n_dof)
    return stiffness[free, free], mass[free, free]


def _analytical_frequencies_hz() -> np.ndarray:
    coefficient = np.sqrt(_FLEXURAL_RIGIDITY / (_MASS_PER_LENGTH * _LENGTH**4)) / (
        2.0 * np.pi
    )
    return np.array([(beta**2) * coefficient for beta in _BETA_L])


class TestCantileverBeam:
    """Validate the eigen-core against the analytical cantilever beam."""

    def test_lowest_frequencies_match_closed_form(self) -> None:
        """The lowest finite-element frequencies match the analytical beam."""
        stiffness, mass = _assemble_cantilever()
        n_free = stiffness.shape[0]
        a = np.zeros((2 * n_free, 2 * n_free), dtype=np.float64)
        a[:n_free, n_free:] = np.eye(n_free)
        a[n_free:, :n_free] = -np.linalg.solve(mass, stiffness)

        solution = compute_modes(a, ndof2=n_free, ndof1=0)
        recovered = np.sort(solution.natural_frequencies_hz)[:3]
        analytical = _analytical_frequencies_hz()
        # Consistent-mass FEM converges from above; 24 elements give < 1% on the
        # first three modes.
        np.testing.assert_allclose(recovered, analytical, rtol=1e-2)

    def test_modes_are_undamped(self) -> None:
        """An undamped beam yields zero damping ratios."""
        stiffness, mass = _assemble_cantilever()
        n_free = stiffness.shape[0]
        a = np.zeros((2 * n_free, 2 * n_free), dtype=np.float64)
        a[:n_free, n_free:] = np.eye(n_free)
        a[n_free:, :n_free] = -np.linalg.solve(mass, stiffness)

        solution = compute_modes(a, ndof2=n_free, ndof1=0)
        np.testing.assert_allclose(solution.damping_ratios, 0.0, atol=1e-9)
