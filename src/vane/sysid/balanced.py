# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Balanced truncation — gramian-based reduction with an a-priori error bound.

:func:`~vane.sysid.modal_reduction` keeps whole modes chosen by their residue in a load.
Balanced truncation is the complementary, input-output-optimal reduction: it balances
the controllability and observability Gramians so that each state is equally hard to
reach and to observe, then drops the states with the smallest **Hankel singular values**
— the ones that contribute least to the transfer function. Unlike modal reduction it
comes with a guarantee: the truncated model's worst-case frequency-response error is
bounded a priori by twice the sum of the discarded Hankel singular values.

For a stable system the Gramians solve the Lyapunov equations
:math:`\mathbf A\mathbf W_c+\mathbf W_c\mathbf A^{\mathsf T}+\mathbf B\mathbf B^{\mathsf
T}=\mathbf 0` and :math:`\mathbf A^{\mathsf T}\mathbf W_o+\mathbf W_o\mathbf A+
\mathbf C^{\mathsf T}\mathbf C=\mathbf 0` (discrete analogues for a sampled system). The
Hankel singular values are :math:`\sigma_i=\sqrt{\lambda_i(\mathbf W_c\mathbf W_o)}`,
and in the balanced realization both Gramians equal
:math:`\operatorname{diag}(\sigma)`. This reduction requires asymptotic stability (the
Gramians exist) and that every *retained* Hankel singular value be resolved above
rounding level (the retained order is controllable and observable). The Gramians are
factored from their symmetric eigendecomposition with rounding-level negative
eigenvalues clipped, so the numerically indefinite Gramians of a lightly damped minimal
system do not cause a spurious rejection. It suits the structural ROM, while the
unstable aero poles of a raw turbine linearization are handled by
:func:`~vane.sysid.modal_reduction` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import (
    eigh,
    solve_continuous_lyapunov,
    solve_discrete_lyapunov,
    svd,
)

from vane.sysid.state_space import StateSpace

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = ["BalancedReduction", "balanced_truncation"]


@dataclass
class BalancedReduction:
    r"""A balanced-truncation reduction and its Hankel-singular-value spectrum.

    Parameters
    ----------
    system : StateSpace
        The reduced (or, when no order is given, balanced full-order) system, with the
        original inputs and outputs and the feed-through ``D`` unchanged.
    hankel_singular_values : npt.NDArray[np.float64]
        All Hankel singular values of the full system, in descending order. The retained
        states correspond to the largest ones; the tail is what the error bound sums.
    error_bound : float
        The a-priori worst-case frequency-response error
        ``2 * sum(discarded Hankel singular values)`` — an upper bound on the
        :math:`\mathcal H_\infty` error
        :math:`\lVert\mathbf G-\mathbf G_r\rVert_\infty`. Zero when nothing is dropped.
    n_full_states : int
        Number of states in the original system.
    """

    system: StateSpace
    hankel_singular_values: npt.NDArray[np.float64]
    error_bound: float
    n_full_states: int


def balanced_truncation(
    system: StateSpace,
    *,
    n_states: int | None = None,
    tol: float | None = None,
) -> BalancedReduction:
    """Reduce a system by balanced truncation, with an a-priori error bound.

    Balances the controllability and observability Gramians, then keeps the states with
    the largest Hankel singular values. The order is set by ``n_states`` (an explicit
    count) or ``tol`` (drop singular values at or below ``tol`` times the largest); with
    neither, the system is balanced but not truncated.

    Parameters
    ----------
    system : StateSpace
        A stable system with both ``B`` and ``C`` matrices. May be continuous (``dt`` is
        ``None``) or discrete.
    n_states : int or None, optional
        Number of states to retain (the largest Hankel singular values). Takes
        precedence over ``tol``. Must be in ``[1, n_states_full]``.
    tol : float or None, optional
        Relative threshold: retain the states whose Hankel singular value exceeds
        ``tol * max(sigma)``. Must be non-negative.

    Returns
    -------
    BalancedReduction
        The reduced system, the full Hankel-singular-value spectrum, and the bound.

    Raises
    ------
    ValueError
        If the system lacks ``B`` or ``C``, is not asymptotically stable, the retained
        order includes a Hankel singular value at rounding level (the system is not
        minimal at that order), or ``n_states``/``tol`` is out of range.
    """
    if system.b is None or system.c is None:
        msg = "balanced truncation needs both B and C matrices"
        raise ValueError(msg)
    if not system.is_stable():
        msg = "balanced truncation requires an asymptotically stable system"
        raise ValueError(msg)
    a, b, c = system.a, system.b, system.c
    n = a.shape[0]

    if system.dt is None:
        w_c = solve_continuous_lyapunov(a, -(b @ b.T))
        w_o = solve_continuous_lyapunov(a.T, -(c.T @ c))
    else:
        w_c = solve_discrete_lyapunov(a, b @ b.T)
        w_o = solve_discrete_lyapunov(a.T, c.T @ c)

    l_c = _square_root_factor(np.asarray(w_c, dtype=np.float64))
    l_o = _square_root_factor(np.asarray(w_o, dtype=np.float64))

    left, sigma, right_t = svd(l_o.T @ l_c)
    sigma = np.asarray(sigma, dtype=np.float64)
    if not float(sigma[0]) > 0.0:
        msg = (
            "every Hankel singular value is zero; the system has no controllable and "
            "observable dynamics to balance"
        )
        raise ValueError(msg)

    keep = _resolve_order(sigma, n, n_states, tol)
    error_bound = float(2.0 * np.sum(sigma[keep:]))

    # The balancing transform divides by sqrt of each *retained* Hankel singular value.
    # A lightly damped minimal system has numerically indefinite Gramians and a long
    # tail of HSVs at rounding level; those are harmless when truncated away (their
    # columns are dropped), so only a retained near-zero HSV -- whose 1/sqrt would blow
    # up the retained block -- signals a non-minimal order and is rejected.
    floor = float(np.finfo(np.float64).eps) * float(sigma[0])
    if float(sigma[keep - 1]) <= floor:
        msg = (
            "the retained balanced order includes a near-zero Hankel singular value "
            f"({float(sigma[keep - 1]):.3e} <= {floor:.3e}); the system is not "
            "minimal at this order (uncontrollable or unobservable states) — request "
            "fewer states (smaller n_states), pass a tolerance, or reduce it modally "
            "first"
        )
        raise ValueError(msg)

    inv_sqrt = 1.0 / np.sqrt(sigma[:keep])
    transform = (l_c @ right_t[:keep].T) * inv_sqrt[np.newaxis, :]
    inverse = inv_sqrt[:, np.newaxis] * (left[:, :keep].T @ l_o.T)

    reduced = StateSpace(
        a=np.asarray(inverse @ a @ transform, dtype=np.float64),
        b=np.asarray(inverse @ b, dtype=np.float64),
        c=np.asarray(c @ transform, dtype=np.float64),
        d=system.d,
        state_names=[f"balanced_{i}" for i in range(keep)],
        input_names=list(system.input_names),
        output_names=list(system.output_names),
        output_mbc_coordinates=list(system.output_mbc_coordinates),
        dt=system.dt,
    )
    return BalancedReduction(
        system=reduced,
        hankel_singular_values=sigma,
        error_bound=error_bound,
        n_full_states=n,
    )


def _square_root_factor(gramian: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    r"""Return a square-root factor ``L`` with ``L L^T`` equal to the PSD part of W.

    The Lyapunov solution of a lightly damped system is positive semi-definite in exact
    arithmetic but numerically indefinite (tiny negative eigenvalues at rounding level),
    so a Cholesky factorization fails even for a minimal system. The Gramian is
    symmetrized and factored from its symmetric eigendecomposition with the eigenvalues
    clipped at zero, ``L = V \operatorname{diag}(\sqrt{\max(\lambda, 0)})``.
    """
    symmetric = 0.5 * (gramian + gramian.T)
    eigenvalues, eigenvectors = eigh(symmetric)
    root = np.sqrt(np.clip(eigenvalues, 0.0, None))
    return np.asarray(eigenvectors * root[np.newaxis, :], dtype=np.float64)


def _resolve_order(
    sigma: npt.NDArray[np.float64],
    n: int,
    n_states: int | None,
    tol: float | None,
) -> int:
    """Resolve how many states to keep from ``n_states``/``tol`` (default: all)."""
    if n_states is not None:
        if not 1 <= n_states <= n:
            msg = f"n_states must be in [1, {n}], got {n_states}"
            raise ValueError(msg)
        return n_states
    if tol is not None:
        if tol < 0.0:
            msg = f"tol must be non-negative, got {tol}"
            raise ValueError(msg)
        threshold = tol * float(sigma[0])
        return max(1, int(np.count_nonzero(sigma > threshold)))
    return n
