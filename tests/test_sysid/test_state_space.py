# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.sysid.state_space — LTI state-space representation."""

from __future__ import annotations

import numpy as np
import pytest

from vane.sysid.state_space import StateSpace, static_gain


def _stable_second_order() -> StateSpace:
    """Return a stable single-DOF oscillator state-space system."""
    a = np.array([[0.0, 1.0], [-4.0, -0.4]])
    return StateSpace(a=a, b=np.array([[0.0], [1.0]]), c=np.array([[1.0, 0.0]]))


class TestStateSpaceValidation:
    """Tests for shape and name validation."""

    def test_non_square_a_raises(self) -> None:
        """A non-square A raises ValueError."""
        with pytest.raises(ValueError, match="square"):
            StateSpace(a=np.zeros((2, 3)))

    def test_b_row_mismatch_raises(self) -> None:
        """A B with the wrong number of rows raises ValueError."""
        with pytest.raises(ValueError, match="B must have"):
            StateSpace(a=np.zeros((2, 2)), b=np.zeros((3, 1)))

    def test_c_column_mismatch_raises(self) -> None:
        """A C with the wrong number of columns raises ValueError."""
        with pytest.raises(ValueError, match="C must have"):
            StateSpace(a=np.zeros((2, 2)), c=np.zeros((1, 3)))

    def test_d_shape_mismatch_raises(self) -> None:
        """A D inconsistent with B and C raises ValueError."""
        with pytest.raises(ValueError, match="D must have"):
            StateSpace(
                a=np.zeros((2, 2)),
                b=np.zeros((2, 1)),
                c=np.zeros((1, 2)),
                d=np.zeros((1, 2)),
            )

    def test_state_name_length_mismatch_raises(self) -> None:
        """A wrong-length state-name list raises ValueError."""
        with pytest.raises(ValueError, match="state_names"):
            StateSpace(a=np.zeros((2, 2)), state_names=["only_one"])

    def test_non_finite_a_raises(self) -> None:
        """A NaN/inf in A is rejected at construction (the foundational guard)."""
        a = np.eye(2)
        a[0, 0] = np.nan
        with pytest.raises(ValueError, match="A must be finite"):
            StateSpace(a=a)

    def test_non_finite_b_raises(self) -> None:
        """A non-finite B is rejected at construction."""
        with pytest.raises(ValueError, match="B must be finite"):
            StateSpace(a=np.zeros((2, 2)), b=np.full((2, 1), np.inf))

    def test_non_finite_c_raises(self) -> None:
        """A non-finite C is rejected at construction."""
        with pytest.raises(ValueError, match="C must be finite"):
            StateSpace(a=np.zeros((2, 2)), c=np.full((1, 2), np.inf))

    def test_non_finite_d_raises(self) -> None:
        """A non-finite D is rejected at construction."""
        with pytest.raises(ValueError, match="D must be finite"):
            StateSpace(
                a=np.zeros((2, 2)),
                b=np.zeros((2, 1)),
                c=np.zeros((1, 2)),
                d=np.full((1, 1), np.nan),
            )

    def test_one_dimensional_b_raises(self) -> None:
        """A 1-D B (even with the right length) is rejected as not a matrix."""
        with pytest.raises(ValueError, match="2-D matrix"):
            StateSpace(a=np.zeros((2, 2)), b=np.zeros(2))

    def test_one_dimensional_c_raises(self) -> None:
        """A 1-D C is rejected as not a matrix."""
        with pytest.raises(ValueError, match="2-D matrix"):
            StateSpace(a=np.zeros((2, 2)), c=np.zeros(2))

    @pytest.mark.parametrize(
        "bad_dt", [0.0, -1.0, float("nan"), float("inf"), float("-inf")]
    )
    def test_invalid_constructor_dt_raises(self, bad_dt: float) -> None:
        """A zero, negative, NaN, or infinite sample time is rejected at build time."""
        with pytest.raises(ValueError, match="positive and finite"):
            StateSpace(a=np.zeros((1, 1)), dt=bad_dt)

    def test_d_validated_against_inputs_when_b_known(self) -> None:
        """D is rejected when its column count disagrees with B, even without C."""
        with pytest.raises(ValueError, match="inputs"):
            StateSpace(a=np.zeros((2, 2)), b=np.zeros((2, 1)), d=np.zeros((1, 2)))

    def test_d_validated_against_outputs_when_c_known(self) -> None:
        """D is rejected when its row count disagrees with C, even without B."""
        with pytest.raises(ValueError, match="outputs"):
            StateSpace(a=np.zeros((2, 2)), c=np.zeros((1, 2)), d=np.zeros((2, 1)))

    def test_input_names_without_b_raises(self) -> None:
        """Input names without a B (zero inputs) are rejected."""
        with pytest.raises(ValueError, match="input_names"):
            StateSpace(a=np.zeros((2, 2)), input_names=["u"])


class TestStateSpaceProperties:
    """Tests for dimension and stability properties."""

    def test_dimensions(self) -> None:
        """State, input, and output counts reflect the matrices."""
        system = _stable_second_order()
        assert system.n_states == 2
        assert system.n_inputs == 1
        assert system.n_outputs == 1

    def test_feedthrough_only_dimensions_follow_d(self) -> None:
        """A system with only D reports D's columns/rows as inputs/outputs."""
        system = StateSpace(a=np.zeros((2, 2)), d=np.zeros((3, 4)))
        assert system.n_inputs == 4
        assert system.n_outputs == 3
        # The feedthrough inputs/outputs can now be named (length matches n_*).
        named = StateSpace(
            a=np.zeros((2, 2)),
            d=np.zeros((3, 4)),
            input_names=["u0", "u1", "u2", "u3"],
            output_names=["y0", "y1", "y2"],
        )
        assert named.n_inputs == 4
        assert named.n_outputs == 3

    def test_continuous_stability(self) -> None:
        """A system with negative-real-part eigenvalues is stable."""
        assert _stable_second_order().is_stable() is True

    def test_unstable_system(self) -> None:
        """A system with a positive-real-part eigenvalue is unstable."""
        system = StateSpace(a=np.array([[1.0, 0.0], [0.0, -1.0]]))
        assert system.is_stable() is False


class TestDiscretization:
    """Tests for zero-order-hold discretization."""

    def test_scalar_discretization(self) -> None:
        """A scalar system discretizes to exp(a*dt)."""
        system = StateSpace(a=np.array([[-1.0]]))
        discrete = system.discretized(0.5)
        assert discrete.is_discrete
        assert discrete.a[0, 0] == pytest.approx(np.exp(-0.5))

    def test_discrete_eigenvalues_are_exponentials(self) -> None:
        """Discrete eigenvalues equal the exponential of continuous eigenvalues."""
        system = _stable_second_order()
        dt = 0.1
        discrete = system.discretized(dt)
        continuous_eigs = np.sort_complex(np.linalg.eigvals(system.a))
        discrete_eigs = np.sort_complex(np.linalg.eigvals(discrete.a))
        np.testing.assert_allclose(
            discrete_eigs, np.exp(continuous_eigs * dt), rtol=1e-8
        )
        assert discrete.is_stable() is True

    @pytest.mark.parametrize("bad_dt", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_discretization_dt_raises(self, bad_dt: float) -> None:
        """A non-positive, NaN, or infinite sample time raises ValueError."""
        with pytest.raises(ValueError, match="positive and finite"):
            _stable_second_order().discretized(bad_dt)

    def test_already_discrete_raises(self) -> None:
        """Discretizing a discrete system raises ValueError."""
        discrete = _stable_second_order().discretized(0.1)
        with pytest.raises(ValueError, match="already discrete"):
            discrete.discretized(0.1)


class TestStaticGain:
    """Zero-frequency gain is explicit and fails on undefined contracts."""

    def test_computes_gain_with_implicit_zero_feedthrough(self) -> None:
        """G(0) is D - C A^-1 B, with absent D interpreted as zero."""
        system = _stable_second_order()
        expected = -system.c @ np.linalg.solve(system.a, system.b)
        np.testing.assert_allclose(static_gain(system), expected)

    def test_rejects_discrete_missing_io_and_singular_systems(self) -> None:
        """Undefined static-gain cases fail with specific errors."""
        with pytest.raises(ValueError, match="continuous-time"):
            static_gain(_stable_second_order().discretized(0.1))
        with pytest.raises(ValueError, match="both B and C"):
            static_gain(StateSpace(a=-np.eye(2)))
        with pytest.raises(ValueError, match="A is singular"):
            static_gain(
                StateSpace(
                    a=np.zeros((1, 1)),
                    b=np.ones((1, 1)),
                    c=np.ones((1, 1)),
                )
            )
