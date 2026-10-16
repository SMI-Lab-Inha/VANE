# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.modal.participation — per-DOF mode participation."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pytest

from vane.config.dof_map import classify_dof
from vane.io import read_lin_file
from vane.io.mbc_transform import mbc3_transform
from vane.modal.eigensolver import modes_from_mbc
from vane.modal.labeler import label_solution
from vane.modal.participation import (
    compute_participation,
    participation_from_modes,
    participation_scale_factors,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestComputeParticipation:
    """Unit tests for compute_participation."""

    def test_normalized_to_dominant_component(self) -> None:
        """Each mode is normalized so its dominant component has magnitude one."""
        modes = np.array([[3.0], [1.0], [0.5]], dtype=np.complex128)
        result = compute_participation(modes)
        np.testing.assert_allclose(result.magnitude[:, 0], [1.0, 1.0 / 3.0, 1.0 / 6.0])
        assert result.dominant_state[0] == 0

    def test_out_of_phase_component_is_negated(self) -> None:
        """Components more than 90 degrees out of phase get a negative sign."""
        modes = np.array([[1.0], [-1.0], [1.0j]], dtype=np.complex128)
        result = compute_participation(modes)
        # Component 1 is 180 deg out of phase -> negative; component 2 is 90 deg.
        assert result.signed_magnitude[0, 0] == pytest.approx(1.0)
        assert result.signed_magnitude[1, 0] == pytest.approx(-1.0)

    def test_in_phase_components_stay_positive(self) -> None:
        """A real, in-phase mode has all-positive signed magnitudes."""
        modes = np.array([[2.0], [1.0], [0.5]], dtype=np.complex128)
        result = compute_participation(modes)
        assert np.all(result.signed_magnitude[:, 0] >= 0.0)

    def test_phase_is_reported_in_degrees(self) -> None:
        """The phase array reports each component's angle in degrees."""
        modes = np.array([[1.0], [1.0j]], dtype=np.complex128)
        result = compute_participation(modes)
        assert result.phase_deg[1, 0] == pytest.approx(90.0)

    def test_scale_factors_applied_before_normalization(self) -> None:
        """Scale factors reweight DOFs before the per-mode normalization."""
        modes = np.array([[1.0], [1.0]], dtype=np.complex128)
        result = compute_participation(modes, scale_factors=np.array([0.1, 1.0]))
        # After scaling the second DOF dominates; the first becomes 0.1 of it.
        assert result.dominant_state[0] == 1
        np.testing.assert_allclose(result.magnitude[:, 0], [0.1, 1.0])

    def test_multiple_modes_independent(self) -> None:
        """Each mode column is normalized independently."""
        modes = np.array([[10.0, 1.0], [5.0, 4.0]], dtype=np.complex128)
        result = compute_participation(modes)
        np.testing.assert_allclose(result.magnitude[:, 0], [1.0, 0.5])
        np.testing.assert_allclose(result.magnitude[:, 1], [0.25, 1.0])

    def test_scale_factor_length_mismatch_raises(self) -> None:
        """A scale_factors length mismatch raises ValueError."""
        modes = np.ones((3, 2), dtype=np.complex128)
        with pytest.raises(ValueError, match="scale_factors"):
            compute_participation(modes, scale_factors=np.array([1.0, 1.0]))

    def test_empty_dof_returns_empty_without_error(self) -> None:
        """A zero-DOF mode matrix returns empty arrays instead of raising."""
        result = compute_participation(np.zeros((0, 2), dtype=np.complex128))
        assert result.magnitude.shape == (0, 2)
        assert result.dominant_state.shape == (2,)

    def test_zero_mode_normalizes_to_zero(self) -> None:
        """An all-zero mode column normalizes to zeros rather than NaN."""
        modes = np.zeros((3, 1), dtype=np.complex128)
        result = compute_participation(modes)
        assert np.all(result.magnitude == 0.0)


class TestParticipationFromModes:
    """Tests for the ModalSolution convenience wrapper."""

    def test_participation_from_modes(self) -> None:
        """participation_from_modes uses the solution's mode shapes."""
        from vane.modal.eigensolver import compute_modes

        a = np.zeros((4, 4), dtype=np.float64)
        a[0:2, 2:4] = np.eye(2)
        a[2:4, 0:2] = -np.diag([4.0, 25.0])
        a[2:4, 2:4] = -np.diag([0.2, 0.5])
        solution = compute_modes(a, ndof2=2, ndof1=0)
        result = participation_from_modes(solution)
        assert result.magnitude.shape == solution.mode_shapes.shape
        assert np.all(result.magnitude.max(axis=0) == pytest.approx(1.0))


# Descriptions carrying the internal DOF tokens classify_dof recognizes.
_BLADE_FLAP = "ED 1st flapwise bending-mode DOF of blade 1, DOF_BF(1,1), m"
_BLADE_EDGE = "ED 1st edgewise bending-mode DOF of blade 2, DOF_BE(2,1), m"
_TOWER_FA1 = "ED 1st tower fore-aft bending mode, DOF_TFA1, m"
_TOWER_SS2 = "ED 2nd tower side-to-side bending mode, DOF_TSS2, m"
_PLATFORM_PITCH = "ED platform pitch tilt rotation, DOF_P, rad"
_PLATFORM_SURGE = "ED platform horizontal surge translation, DOF_Sg, m"
_BLADE_PITCH = "SrvD blade 1 pitch, DOF_BP(1), rad"
_GEN_AZIMUTH = "ED variable speed generator, DOF_GeAz, rad"
_TOWER_FA1_VEL = "ED First time derivative of 1st tower fore-aft mode, DOF_TFA1, m/s"
_UNKNOWN = "AD blade 1 inflow state, m/s"


class TestParticipationScaleFactors:
    """Deriving per-DOF scale factors from state descriptions and geometry."""

    def test_blade_and_tower_use_their_lengths(self) -> None:
        """Blade deflections scale by 1/blade_length, tower by 1/tower_length."""
        factors = participation_scale_factors(
            [_BLADE_FLAP, _BLADE_EDGE, _TOWER_FA1, _TOWER_SS2],
            blade_length=100.0,
            tower_length=120.0,
        )
        np.testing.assert_allclose(
            factors, [1 / 100.0, 1 / 100.0, 1 / 120.0, 1 / 120.0]
        )

    def test_rotational_and_unknown_are_unscaled(self) -> None:
        """Rotations, rigid-body platform DOFs, and unknown states stay at 1.0."""
        factors = participation_scale_factors(
            [_PLATFORM_PITCH, _PLATFORM_SURGE, _BLADE_PITCH, _GEN_AZIMUTH, _UNKNOWN],
            blade_length=100.0,
            tower_length=120.0,
        )
        np.testing.assert_allclose(factors, [1.0, 1.0, 1.0, 1.0, 1.0])

    def test_velocity_state_scales_like_its_dof(self) -> None:
        """A velocity state shares its DOF's category, so it scales the same."""
        factors = participation_scale_factors(
            [_TOWER_FA1_VEL], blade_length=100.0, tower_length=120.0
        )
        np.testing.assert_allclose(factors, [1 / 120.0])

    def test_length_matches_descriptions(self) -> None:
        """One factor is returned per description, in order."""
        descriptions = [_TOWER_FA1, _BLADE_FLAP, _UNKNOWN]
        factors = participation_scale_factors(
            descriptions, blade_length=80.0, tower_length=140.0
        )
        assert factors.shape == (3,)
        np.testing.assert_allclose(factors, [1 / 140.0, 1 / 80.0, 1.0])

    def test_empty_descriptions(self) -> None:
        """No descriptions yields an empty factor array."""
        factors = participation_scale_factors(
            [], blade_length=100.0, tower_length=120.0
        )
        assert factors.shape == (0,)

    def test_feeds_compute_participation(self) -> None:
        """The derived factors plug straight into compute_participation."""
        descriptions = [_BLADE_FLAP, _TOWER_FA1, _PLATFORM_PITCH]
        modes = np.array([[2.0], [3.0], [1.0]], dtype=np.complex128)
        factors = participation_scale_factors(
            descriptions, blade_length=100.0, tower_length=10.0
        )
        result = compute_participation(modes, factors)
        # Tower row scaled by 1/10 (0.3) now dominates the unscaled platform row (1.0)?
        # Scaled components: blade 2/100=0.02, tower 3/10=0.3, platform 1.0 -> platform.
        assert result.dominant_state[0] == 2

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_bad_blade_length(self, bad: float) -> None:
        """A non-positive or non-finite blade length is rejected."""
        with pytest.raises(ValueError, match="blade_length must be"):
            participation_scale_factors(
                [_BLADE_FLAP], blade_length=bad, tower_length=120.0
            )

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_bad_tower_length(self, bad: float) -> None:
        """A non-positive or non-finite tower length is rejected."""
        with pytest.raises(ValueError, match="tower_length must be"):
            participation_scale_factors(
                [_TOWER_FA1], blade_length=100.0, tower_length=bad
            )


_BD_PREFIX = "BD_1 finite element node 7 (number of elements = 1; element order = 5)"
_BD_TX = f"{_BD_PREFIX} translational displacement in X, m"
_BD_TY = f"{_BD_PREFIX} translational displacement in Y, m"
_BD_TZ = f"{_BD_PREFIX} translational displacement in Z, m"
_BD_RX = f"{_BD_PREFIX} rotational displacement in X, rad"
_BD_RY = f"{_BD_PREFIX} rotational displacement in Y, rad"
_BD_RZ = f"{_BD_PREFIX} rotational displacement in Z, rad"
_BD_TX_VEL = f"First time derivative of {_BD_TX}/s"


class TestBeamDynScaleFactors:
    """BeamDyn nodal translations are scaled by the blade length."""

    def test_translations_scale_by_blade_length(self) -> None:
        """TX/TY/TZ nodal states get 1/blade_length, like ElastoDyn deflections."""
        factors = participation_scale_factors(
            [_BD_TX, _BD_TY, _BD_TZ, _BD_TX_VEL],
            blade_length=60.0,
            tower_length=90.0,
        )
        np.testing.assert_allclose(factors, 1.0 / 60.0)

    def test_rotations_stay_unscaled(self) -> None:
        """The nodal rotation parameters (radians) are left at 1.0."""
        factors = participation_scale_factors(
            [_BD_RX, _BD_RY, _BD_RZ], blade_length=60.0, tower_length=90.0
        )
        np.testing.assert_allclose(factors, 1.0)

    def test_scaled_translation_no_longer_outweighs_rotation(self) -> None:
        """A tip deflection of L/2 per unit rotation becomes comparable to it."""
        modes = np.array([[30.0], [1.0]], dtype=np.complex128)
        factors = participation_scale_factors(
            [_BD_TX, _BD_RY], blade_length=60.0, tower_length=90.0
        )
        raw = compute_participation(modes)
        scaled = compute_participation(modes, factors)
        assert raw.dominant_state[0] == 0
        assert scaled.dominant_state[0] == 1
        np.testing.assert_allclose(scaled.magnitude[:, 0], [0.5, 1.0])


def test_rtest_beamdyn_translations_become_comparable(rtest_dir: Path) -> None:
    """On the NREL 5-MW BeamDyn r-test model the scaling removes the unit bias.

    Unscaled, the nodal translations (metres) of the first flap and edge modes are
    20-50 times the nodal rotations; scaled by the 61.5 m blade length, the tip
    deflection ratio is of the order of the tip rotation, as for a cantilever.
    """
    lin = (
        rtest_dir
        / "glue-codes"
        / "openfast"
        / "5MW_Land_BD_Linear"
        / "5MW_Land_BD_Linear.1.lin"
    )
    if not lin.is_file():
        pytest.skip(f"r-test BeamDyn linearization not found: {lin}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solution = modes_from_mbc(mbc3_transform([read_lin_file(lin)]))
    components = [classify_dof(d).component for d in solution.dof_descriptions]
    translation = np.array([c in ("TX", "TY", "TZ") for c in components])
    rotation = np.array([c in ("RX", "RY", "RZ") for c in components])
    assert translation.any()
    assert rotation.any()
    factors = participation_scale_factors(
        solution.dof_descriptions, blade_length=61.5, tower_length=87.6
    )
    np.testing.assert_allclose(factors[translation], 1.0 / 61.5)
    np.testing.assert_allclose(factors[rotation], 1.0)
    raw = compute_participation(solution.mode_shapes).magnitude
    scaled = compute_participation(solution.mode_shapes, factors).magnitude
    labels = label_solution(solution)
    for name in ("1st Blade Flap (Collective)", "1st Blade Edge (Collective)"):
        k = next(i for i, label in enumerate(labels) if label.label == name)
        raw_ratio = raw[translation, k].max() / raw[rotation, k].max()
        scaled_ratio = scaled[translation, k].max() / scaled[rotation, k].max()
        assert raw_ratio > 10.0, (name, raw_ratio)
        assert 0.1 < scaled_ratio < 2.0, (name, scaled_ratio)
