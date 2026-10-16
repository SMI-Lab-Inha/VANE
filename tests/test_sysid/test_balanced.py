# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.sysid.balanced — balanced truncation with an error bound."""

import numpy as np
import pytest
from scipy.linalg import solve_continuous_lyapunov

from vane.sysid import StateSpace, balanced_truncation, observability_gramian


def _stable_system(seed: int = 3, n: int = 6) -> StateSpace:
    """Return a random stable MIMO system of order ``n``."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    a = a - (np.abs(np.linalg.eigvals(a).real).max() + 0.5) * np.eye(n)
    b = rng.standard_normal((n, 2))
    c = rng.standard_normal((1, n))
    return StateSpace(a=a, b=b, c=c, input_names=["u0", "u1"], output_names=["y"])


def _max_response_error(
    full: StateSpace, reduced: StateSpace, frequencies: np.ndarray
) -> float:
    """Return the largest spectral-norm transfer-function error over ``frequencies``."""
    assert full.b is not None and full.c is not None
    assert reduced.b is not None and reduced.c is not None
    n_full, n_red = full.a.shape[0], reduced.a.shape[0]
    error = 0.0
    for w in frequencies:
        g_full = full.c @ np.linalg.solve(1j * w * np.eye(n_full) - full.a, full.b)
        g_red = reduced.c @ np.linalg.solve(
            1j * w * np.eye(n_red) - reduced.a, reduced.b
        )
        error = max(error, float(np.linalg.norm(g_full - g_red, 2)))
    return error


class TestBalancedTruncation:
    """Tests for balanced_truncation."""

    def test_hankel_values_match_gramian_product(self) -> None:
        """The Hankel singular values are sqrt of the eigenvalues of Wc Wo."""
        ss = _stable_system()
        result = balanced_truncation(ss)
        assert ss.b is not None and ss.c is not None
        w_c = solve_continuous_lyapunov(ss.a, -(ss.b @ ss.b.T))
        w_o = solve_continuous_lyapunov(ss.a.T, -(ss.c.T @ ss.c))
        expected = np.sort(np.sqrt(np.linalg.eigvals(w_c @ w_o).real))[::-1]
        np.testing.assert_allclose(result.hankel_singular_values, expected, atol=1e-8)

    def test_full_order_is_balanced(self) -> None:
        """With no truncation the realization is balanced: Wo = diag(sigma)."""
        result = balanced_truncation(_stable_system())
        assert result.system.n_states == 6
        w_o = observability_gramian(result.system)
        np.testing.assert_allclose(
            w_o, np.diag(result.hankel_singular_values), atol=1e-7
        )

    def test_truncation_respects_the_error_bound(self) -> None:
        """The reduced model's response error stays under 2 * sum(tail sigma)."""
        ss = _stable_system()
        result = balanced_truncation(ss, n_states=3)
        assert result.system.n_states == 3
        frequencies = np.logspace(-2, 2, 200)
        error = _max_response_error(ss, result.system, frequencies)
        assert error <= result.error_bound + 1e-9

    def test_error_bound_is_twice_the_discarded_tail(self) -> None:
        """The bound equals twice the sum of the discarded Hankel singular values."""
        result = balanced_truncation(_stable_system(), n_states=2)
        tail = result.hankel_singular_values[2:]
        assert result.error_bound == pytest.approx(2.0 * float(np.sum(tail)))

    def test_no_truncation_has_zero_error_bound(self) -> None:
        """Balancing without truncation discards nothing, so the bound is zero."""
        result = balanced_truncation(_stable_system())
        assert result.error_bound == pytest.approx(0.0, abs=1e-9)

    def test_tolerance_selects_the_order(self) -> None:
        """A relative tolerance keeps the singular values above the threshold."""
        result = balanced_truncation(_stable_system(), tol=0.01)
        sigma = result.hankel_singular_values
        expected = int(np.count_nonzero(sigma > 0.01 * sigma[0]))
        assert result.system.n_states == expected

    def test_discrete_system_reduces(self) -> None:
        """A discrete system is reduced and stays discrete."""
        result = balanced_truncation(_stable_system().discretized(0.05), n_states=3)
        assert result.system.n_states == 3
        assert result.system.dt == pytest.approx(0.05)

    def test_rejects_missing_b_or_c(self) -> None:
        """A system without B or C cannot be balanced."""
        ss = _stable_system()
        with pytest.raises(ValueError, match="both B and C"):
            balanced_truncation(StateSpace(a=ss.a, c=ss.c))

    def test_rejects_unstable_system(self) -> None:
        """An unstable system has no Gramians to balance."""
        a = np.array([[0.5, -6.0], [6.0, 0.5]])
        ss = StateSpace(a=a, b=np.ones((2, 1)), c=np.ones((1, 2)))
        with pytest.raises(ValueError, match="asymptotically stable"):
            balanced_truncation(ss)

    def test_rejects_non_minimal_system(self) -> None:
        """A system with an uncontrollable state has a singular Gramian."""
        # State 1 is stable but unreachable (B has no second row) and unobservable.
        a = np.diag([-1.0, -2.0])
        b = np.array([[1.0], [0.0]])
        c = np.array([[1.0, 0.0]])
        with pytest.raises(ValueError, match="not minimal"):
            balanced_truncation(StateSpace(a=a, b=b, c=c))

    def test_lightly_damped_chain_reduces_within_the_bound(self) -> None:
        """A minimal 25-DOF spring-mass chain is reduced, not misdiagnosed.

        Its Lyapunov solutions are numerically indefinite (a Cholesky factorization
        fails), yet the system is minimal; truncating the rounding-level HSV tail must
        succeed and respect the a-priori bound.
        """
        n_dof, k = 25, 1.0e3
        stiffness = np.zeros((n_dof, n_dof))
        for i in range(n_dof):
            stiffness[i, i] += 2.0 * k * (1.0 + 0.3 * i)
            if i + 1 < n_dof:
                stiffness[i, i + 1] -= k
                stiffness[i + 1, i] -= k
        damping = 0.5 / k * stiffness * 0.01 + 0.05 * np.eye(n_dof)
        a = np.block(
            [[np.zeros((n_dof, n_dof)), np.eye(n_dof)], [-stiffness, -damping]]
        )
        b = np.zeros((2 * n_dof, 1))
        b[-1, 0] = 1.0
        c = np.zeros((2, 2 * n_dof))
        c[0, 0] = 1.0
        c[1, n_dof - 1] = k
        chain = StateSpace(a=a, b=b, c=c, d=np.zeros((2, 1)))
        result = balanced_truncation(chain, tol=1e-8)
        assert 0 < result.system.n_states < 2 * n_dof
        assert result.system.is_stable()
        error = _max_response_error(chain, result.system, np.linspace(0.1, 100.0, 400))
        assert error <= result.error_bound + 1e-9 * result.hankel_singular_values[0]

    def test_rejects_out_of_range_n_states(self) -> None:
        """An order outside [1, n] is rejected."""
        ss = _stable_system()
        with pytest.raises(ValueError, match="n_states must be in"):
            balanced_truncation(ss, n_states=99)

    def test_rejects_negative_tol(self) -> None:
        """A negative tolerance is rejected."""
        with pytest.raises(ValueError, match="tol must be non-negative"):
            balanced_truncation(_stable_system(), tol=-1.0)
