# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Real modal state-space realization from modal parameters.

A set of under-damped modes is realized as a real, block-diagonal state-space
system. Each mode with eigenvalue :math:`\lambda = \sigma + j\omega` contributes a
``2x2`` block

.. math::

    \begin{bmatrix} \sigma & -\omega \\ \omega & \sigma \end{bmatrix},

whose eigenvalues are :math:`\sigma \pm j\omega`, acting on the real and imaginary
parts of the modal coordinate. When mode shapes are supplied, the output matrix
maps the modal states to physical responses via :math:`[\,2\operatorname{Re}\phi,
\,-2\operatorname{Im}\phi\,]`. The result is a compact reduced-order model of the
identified modes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import block_diag

from vane.sysid.state_space import StateSpace

if TYPE_CHECKING:
    import numpy.typing as npt

    from vane.modal.eigensolver import ModalSolution

__all__ = ["modal_state_space", "modal_state_space_from_solution"]


def modal_state_space(
    eigenvalues: npt.NDArray[np.complex128],
    mode_shapes: npt.NDArray[np.complex128] | None = None,
    *,
    output_names: list[str] | None = None,
) -> StateSpace:
    """Build a real modal state-space realization.

    Parameters
    ----------
    eigenvalues : npt.NDArray[np.complex128]
        Modal eigenvalues, shape ``(n_modes,)``.
    mode_shapes : npt.NDArray[np.complex128] or None, optional
        Complex mode shapes, shape ``(n_outputs, n_modes)``; when given, the output
        matrix maps modal states to physical responses.
    output_names : list[str] or None, optional
        Names of the outputs (the rows of ``mode_shapes``).

    Returns
    -------
    StateSpace
        A continuous-time system with ``2 * n_modes`` states.

    Raises
    ------
    ValueError
        If ``mode_shapes`` column count does not match the number of eigenvalues.
    """
    if eigenvalues.ndim != 1:
        msg = f"eigenvalues must be a 1-D array, got a {eigenvalues.ndim}-D array"
        raise ValueError(msg)
    n_modes = int(eigenvalues.shape[0])
    if mode_shapes is not None:
        if mode_shapes.ndim != 2:
            msg = f"mode_shapes must be a 2-D array, got a {mode_shapes.ndim}-D array"
            raise ValueError(msg)
        if mode_shapes.shape[1] != n_modes:
            msg = (
                f"mode_shapes has {mode_shapes.shape[1]} columns but there are "
                f"{n_modes} eigenvalues"
            )
            raise ValueError(msg)

    blocks = [
        np.array(
            [[value.real, -value.imag], [value.imag, value.real]], dtype=np.float64
        )
        for value in eigenvalues
    ]
    a = (
        np.asarray(block_diag(*blocks), dtype=np.float64)
        if blocks
        else np.zeros((0, 0), dtype=np.float64)
    )

    state_names = [
        f"mode_{mode}_{part}" for mode in range(n_modes) for part in ("re", "im")
    ]
    c = _modal_output_matrix(mode_shapes, n_modes)
    return StateSpace(
        a=a,
        c=c,
        state_names=state_names,
        output_names=list(output_names) if output_names is not None else [],
    )


def modal_state_space_from_solution(solution: ModalSolution) -> StateSpace:
    """Build a modal realization from a :class:`ModalSolution`.

    Parameters
    ----------
    solution : ModalSolution
        The modal solution whose eigenvalues and mode shapes are realized.

    Returns
    -------
    StateSpace
        The real modal state-space system, with outputs named by the solution's
        DOF descriptions when available.
    """
    return modal_state_space(
        solution.eigenvalues,
        solution.mode_shapes,
        output_names=solution.dof_descriptions or None,
    )


def _modal_output_matrix(
    mode_shapes: npt.NDArray[np.complex128] | None, n_modes: int
) -> npt.NDArray[np.float64] | None:
    """Return the output matrix mapping modal states to physical responses."""
    if mode_shapes is None:
        return None
    n_outputs = mode_shapes.shape[0]
    c = np.zeros((n_outputs, 2 * n_modes), dtype=np.float64)
    for mode in range(n_modes):
        c[:, 2 * mode] = 2.0 * mode_shapes[:, mode].real
        c[:, 2 * mode + 1] = -2.0 * mode_shapes[:, mode].imag
    return c
