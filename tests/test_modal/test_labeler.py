# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.modal.labeler — physics-aware mode labeling."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from vane.config.dof_map import DofCategory
from vane.modal.labeler import ModeLabel, label_mode, label_modes, label_solution

if TYPE_CHECKING:
    from tests.conftest import LinFileFactory
    from vane.modal.eigensolver import ModalSolution

_TOWER_FA = "ED 1st tower fore-aft (internal DOF index = DOF_TFA1), m"
_TOWER_SS = "ED 1st tower side-side (internal DOF index = DOF_TSS1), m"
_BLADE_FLAP = "ED 1st flapwise blade 1 (internal DOF index = DOF_BF(1,1)), m"
_DRIVETRAIN = "ED drivetrain (internal DOF index = DOF_DrTr), rad"
_AERO = "AD B1N001 axial induction, -"


def _isotropic_rotor_states() -> list[tuple[str, bool, int]]:
    """Return three identical rotating blade-flap DOFs and their velocities."""
    disp: list[tuple[str, bool, int]] = [
        (
            f"ED 1st flapwise bending-mode DOF of blade {b} "
            f"(internal DOF index = DOF_BF({b},1)), m",
            True,
            2,
        )
        for b in (1, 2, 3)
    ]
    return disp + [
        (f"ED First time derivative of {desc}", True, 2) for desc, _, _ in disp
    ]


def _isotropic_rotor_a(w0: float) -> np.ndarray:
    """Return the rotating-frame A for three identical blade oscillators."""
    a = np.zeros((6, 6), dtype=np.float64)
    a[0:3, 3:6] = np.eye(3)
    a[3:6, 0:3] = -(w0**2) * np.eye(3)
    a[3:6, 3:6] = -2.0 * 0.02 * w0 * np.eye(3)
    return a


class TestBladeMultiblade:
    """Tests for collective/regressive/progressive multi-blade identification."""

    def test_isotropic_rotor_taxonomy(self, make_lin_file: LinFileFactory) -> None:
        """Isotropic rotor: collective at w0, regressive/progressive at w0 -/+ Omega."""
        from vane.io.lin_reader import read_lin_file
        from vane.io.mbc_transform import mbc3_transform
        from vane.modal.eigensolver import modes_from_mbc

        w0, omega = 2.0, 1.5
        states = _isotropic_rotor_states()
        a = _isotropic_rotor_a(w0)
        files = [
            read_lin_file(
                make_lin_file(states=states, a=a, rotor_speed=omega, azimuth=psi)
            )
            for psi in (0.0, 2.0, 4.0)
        ]
        solution = modes_from_mbc(mbc3_transform(files))
        labels = label_solution(solution)

        assert solution.n_modes == 3
        # Sorted ascending by frequency: regressive (w0-Omega), collective, progressive.
        assert [label.multiblade for label in labels] == [
            "regressive",
            "collective",
            "progressive",
        ]
        assert all(label.category is DofCategory.BLADE_FLAP_1 for label in labels)
        assert "Regressive" in labels[0].label

    def test_multiblade_type_classification(self) -> None:
        """The whirl sign and collective dominance are classified correctly."""
        from vane.modal.labeler import _multiblade_type

        coords = ["collective", "cosine", "sine"]
        collective = np.array([1.0, 0.1, 0.1], dtype=np.complex128)
        regressive = np.array([0.0, 1.0, 1.0j], dtype=np.complex128)  # qs = +i qc
        progressive = np.array([0.0, 1.0, -1.0j], dtype=np.complex128)  # qs = -i qc
        assert _multiblade_type(collective, coords) == "collective"
        assert _multiblade_type(regressive, coords) == "regressive"
        assert _multiblade_type(progressive, coords) == "progressive"

    def test_standing_cyclic_mode_is_not_progressive(self) -> None:
        """A cosine/sine pair in phase (no whirl) is 'cyclic', not 'progressive'."""
        from vane.modal.labeler import _multiblade_type

        coords = ["collective", "cosine", "sine"]
        shape = np.array([0.0, 1.0, 1.0], dtype=np.complex128)  # in phase, no whirl
        assert _multiblade_type(shape, coords) == "cyclic"

    def test_leaked_partner_coordinate_is_cyclic(self) -> None:
        """One coordinate dominating with a leaked partner is 'cyclic', not whirling."""
        from vane.modal.labeler import _multiblade_type

        coords = ["collective", "cosine", "sine"]
        # Cosine dominates; sine is only numerical leakage -> standing, no whirl.
        shape = np.array([0.0, 1.0, 1.0e-12j], dtype=np.complex128)
        assert _multiblade_type(shape, coords) == "cyclic"

    def test_no_multiblade_when_blade_participation_minor(self) -> None:
        """A non-blade-dominated mode (e.g. tower) gets no multi-blade type."""
        from vane.modal.labeler import _multiblade_type

        coords = ["", "collective", "cosine", "sine"]
        shape = np.array([10.0, 0.1, 0.1, 0.1], dtype=np.complex128)
        assert _multiblade_type(shape, coords) is None

    def test_multiblade_is_phase_invariant(self) -> None:
        """The whirl classification is invariant to the mode's overall phase."""
        from vane.modal.labeler import _multiblade_type

        coords = ["collective", "cosine", "sine"]
        shape = np.array([0.0, 1.0, 1.0j], dtype=np.complex128)
        rotated = shape * np.exp(1j * 0.7)
        assert _multiblade_type(rotated, coords) == _multiblade_type(shape, coords)

    def test_no_multiblade_without_mbc_coordinates(self) -> None:
        """Without MBC coordinates, every mode's multiblade type is None."""
        from vane.modal.eigensolver import compute_modes

        a = np.zeros((4, 4), dtype=np.float64)
        a[0:2, 2:4] = np.eye(2)
        a[2:4, 0:2] = -np.diag([4.0, 9.0])
        a[2:4, 2:4] = -np.diag([0.2, 0.3])
        solution = compute_modes(
            a, ndof2=2, ndof1=0, descriptions=[_TOWER_FA, _DRIVETRAIN, "v1", "v2"]
        )
        assert all(label.multiblade is None for label in label_solution(solution))


class TestLabelMode:
    """Unit tests for label_mode."""

    def test_dominant_category_names_the_mode(self) -> None:
        """The category carrying the most participation labels the mode."""
        magnitudes = np.array([1.0, 0.1, 0.05])
        label = label_mode(magnitudes, [_TOWER_FA, _TOWER_SS, _BLADE_FLAP])
        assert label.category is DofCategory.TOWER_FORE_AFT_1
        assert label.label == "1st Tower Fore-Aft"

    def test_confidence_is_dominant_fraction(self) -> None:
        """Confidence equals the dominant category's share of participation."""
        magnitudes = np.array([0.75, 0.25])
        label = label_mode(magnitudes, [_TOWER_FA, _DRIVETRAIN])
        assert label.confidence == pytest.approx(0.75)

    def test_mixed_mode_has_low_confidence(self) -> None:
        """A two-way split yields a confidence near 0.5."""
        label = label_mode(np.array([0.5, 0.5]), [_TOWER_FA, _DRIVETRAIN])
        assert label.confidence == pytest.approx(0.5)

    def test_blade_dominated_mode(self) -> None:
        """A blade-dominated mode is labeled as a blade flap mode."""
        label = label_mode(np.array([0.1, 1.0]), [_TOWER_FA, _BLADE_FLAP])
        assert label.category is DofCategory.BLADE_FLAP_1
        assert "Blade Flap" in label.label

    def test_dominant_dofs_listed_in_order(self) -> None:
        """The top contributing DOFs are listed by descending magnitude."""
        label = label_mode(np.array([0.2, 1.0]), [_TOWER_FA, _DRIVETRAIN])
        assert label.dominant_dofs[0] == _DRIVETRAIN

    def test_all_zero_participation_is_unidentified(self) -> None:
        """A mode with no participation is labeled Unidentified."""
        label = label_mode(np.zeros(2), [_TOWER_FA, _DRIVETRAIN])
        assert label.category is DofCategory.UNKNOWN
        assert label.confidence == 0.0

    def test_aero_dominated_mode_is_unidentified(self) -> None:
        """An aerodynamic-state-dominated mode is Unidentified."""
        label = label_mode(np.array([0.1, 1.0]), [_TOWER_FA, _AERO])
        assert label.category is DofCategory.UNKNOWN

    def test_length_mismatch_raises(self) -> None:
        """A magnitude/description length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="does not match"):
            label_mode(np.ones(3), [_TOWER_FA, _DRIVETRAIN])


class TestLabelModesAndSolution:
    """Tests for the participation- and solution-level wrappers."""

    def test_label_modes_one_per_column(self) -> None:
        """label_modes returns one label per participation column."""
        from vane.modal.participation import compute_participation

        modes = np.array([[1.0, 0.1], [0.1, 1.0]], dtype=np.complex128)
        participation = compute_participation(modes)
        labels = label_modes(participation, [_TOWER_FA, _DRIVETRAIN])
        assert [label.category for label in labels] == [
            DofCategory.TOWER_FORE_AFT_1,
            DofCategory.DRIVETRAIN_TORSION,
        ]

    def test_label_solution_uses_dof_descriptions(self) -> None:
        """label_solution labels a real modal solution via its descriptions."""
        from vane.modal.eigensolver import compute_modes

        # Two uncoupled oscillators: a tower DOF and a drivetrain DOF.
        a = np.zeros((4, 4), dtype=np.float64)
        a[0:2, 2:4] = np.eye(2)
        a[2:4, 0:2] = -np.diag([4.0, 9.0])
        a[2:4, 2:4] = -np.diag([0.2, 0.3])
        solution = compute_modes(
            a, ndof2=2, ndof1=0, descriptions=[_TOWER_FA, _DRIVETRAIN, "v1", "v2"]
        )
        labels = label_solution(solution)
        assert {label.category for label in labels} == {
            DofCategory.TOWER_FORE_AFT_1,
            DofCategory.DRIVETRAIN_TORSION,
        }

    def test_label_solution_without_descriptions_raises(self) -> None:
        """label_solution raises when the solution has no DOF descriptions."""
        from vane.modal.eigensolver import compute_modes

        solution = compute_modes(np.array([[0.0, 1.0], [-4.0, -0.2]]), ndof2=1, ndof1=0)
        with pytest.raises(ValueError, match="dof_descriptions"):
            label_solution(solution)


_TOWER_FA_VEL = f"First time derivative of {_TOWER_FA}"
_N_MODULE = 30


def _swamped_tower_solution() -> ModalSolution:
    """Return a tower mode driving many one-way-coupled module states.

    Thirty first-order mooring-like states are driven hard by the tower but do not
    feed back, so they dominate the tower mode's raw right-eigenvector magnitude
    while their left-eigenvector components -- and so their participation -- are
    exactly zero.
    """
    from vane.modal.eigensolver import compute_modes

    n = 2 + _N_MODULE
    a = np.zeros((n, n), dtype=np.float64)
    a[0, 1] = 1.0
    a[1, 0] = -((2.0 * np.pi * 0.5) ** 2)
    a[1, 1] = -0.05
    a[2:, 2:] = -2.0 * np.eye(_N_MODULE)
    a[2:, 0] = 100.0
    descriptions = [_TOWER_FA, _TOWER_FA_VEL] + [
        f"MD Line1 Node{i} position X, m" for i in range(_N_MODULE)
    ]
    return compute_modes(a, ndof2=1, ndof1=_N_MODULE, descriptions=descriptions)


def _module_dominated_solution() -> ModalSolution:
    """Return a module-state oscillator weakly coupled to a tower DOF."""
    from vane.modal.eigensolver import compute_modes

    a = np.zeros((4, 4), dtype=np.float64)
    a[0, 1] = 1.0
    a[1, 0] = -((2.0 * np.pi * 0.5) ** 2)
    a[1, 1] = -0.05
    # A lightly damped first-order oscillator at 2 rad/s (e.g. radiation states).
    a[2:4, 2:4] = np.array([[-0.1, 2.0], [-2.0, -0.1]])
    a[1, 2] = 0.05
    a[2, 0] = 0.05
    descriptions = [
        _TOWER_FA,
        _TOWER_FA_VEL,
        "HD Radiation state 1, -",
        "HD Radiation state 2, -",
    ]
    return compute_modes(a, ndof2=1, ndof1=2, descriptions=descriptions)


class TestParticipationLabelling:
    """Labels come from scale-invariant biorthogonal participation."""

    def test_swamping_module_states_do_not_hide_tower_mode(self) -> None:
        """Large raw module amplitudes with no participation still give tower FA1."""
        solution = _swamped_tower_solution()
        assert solution.n_modes == 1
        raw = np.abs(solution.mode_shapes[:, 0])
        # Precondition: the raw magnitude vote would be swamped by module states.
        assert raw[1:].sum() > 10.0 * raw[0]
        assert label_mode(raw, solution.dof_descriptions).category is (
            DofCategory.UNKNOWN
        )

        label = label_solution(solution)[0]
        assert label.category is DofCategory.TOWER_FORE_AFT_1
        assert label.label == "1st Tower Fore-Aft"
        assert label.confidence == pytest.approx(1.0)
        assert label.dominant_dofs == [_TOWER_FA]

    def test_module_dominated_mode_stays_unidentified(self) -> None:
        """A mode whose participation sits in unclassified states is Unidentified."""
        solution = _module_dominated_solution()
        # Sorted by frequency: the ~0.32 Hz module oscillator, then the 0.5 Hz tower.
        np.testing.assert_allclose(
            solution.natural_frequencies_hz, [1.0 / np.pi, 0.5], rtol=1e-2
        )
        module, tower = label_solution(solution)
        assert module.category is DofCategory.UNKNOWN
        assert module.label == "Unidentified"
        assert module.confidence > 0.9
        assert tower.category is DofCategory.TOWER_FORE_AFT_1

    def test_track_labels_use_participation(self) -> None:
        """Identified tracks are labelled by participation, not raw magnitude."""
        from vane.modal.identifier import identify_modes

        solution = _swamped_tower_solution()
        result = identify_modes([solution, solution])
        assert len(result.tracks) == 1
        assert result.tracks[0].label.category is DofCategory.TOWER_FORE_AFT_1

    def test_track_labels_are_computed_once_per_operating_point(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every track reuses its operating point's labels (no per-track relabel)."""
        from vane.modal import identifier, labeler

        solution = _module_dominated_solution()
        calls: list[int] = []
        original = labeler.label_solution

        def counting(sol: ModalSolution) -> list[ModeLabel]:
            calls.append(1)
            return original(sol)

        monkeypatch.setattr("vane.modal.identifier.label_solution", counting)
        result = identifier.identify_modes([solution, solution, solution])
        assert len(result.tracks) == 2
        assert len(calls) == 3

    def test_degenerate_pair_shares_a_subspace_label(self) -> None:
        """A repeated eigenvalue is labelled from its subspace, not an arbitrary basis.

        Identical fore-aft and side-side oscillators form a degenerate pair; each
        eigenvector is an arbitrary rotation within the pair, so both modes must get
        the subspace's label with a confidence that exposes the even split.
        """
        from vane.modal.eigensolver import compute_modes

        omega = 2.0 * np.pi * 0.5
        a = np.zeros((4, 4))
        a[:2, 2:] = np.eye(2)
        a[2:, :2] = -(omega**2) * np.eye(2)
        a[2:, 2:] = -0.02 * omega * np.eye(2)
        solution = compute_modes(
            a, ndof2=2, ndof1=0, descriptions=[_TOWER_FA, _TOWER_SS]
        )
        assert bool(np.all(solution.is_degenerate))
        first, second = label_solution(solution)
        assert first.category is second.category
        assert first.confidence == pytest.approx(0.5, abs=1e-6)
        assert second.confidence == pytest.approx(0.5, abs=1e-6)

    def test_degenerate_cyclic_pair_gets_no_whirl_suffix(self) -> None:
        """An arbitrary basis of a repeated cyclic pair defines no whirl type."""
        from vane.modal.eigensolver import compute_modes

        flap = "ED 1st flapwise bending-mode DOF of blade {} (DOF_BF({},1)), m"
        descriptions = [flap.format(b, b) for b in (1, 2, 3)]
        omega = 2.0 * np.pi * np.array([0.7, 0.9, 0.9])
        a = np.zeros((6, 6))
        a[:3, 3:] = np.eye(3)
        a[3:, :3] = -np.diag(omega**2)
        a[3:, 3:] = -np.diag(0.02 * omega)
        solution = compute_modes(
            a,
            ndof2=3,
            ndof1=0,
            descriptions=descriptions,
            mbc_coordinates=["collective", "cosine", "sine"],
        )
        np.testing.assert_array_equal(solution.is_degenerate, [False, True, True])
        collective, first, second = label_solution(solution)
        assert collective.multiblade == "collective"
        assert first.multiblade is None
        assert second.multiblade is None

    def test_degenerate_clusters_follow_tolerance_chains(self) -> None:
        """A near B and B near C form one cluster even when A and C are not near."""
        from vane.modal.eigensolver import DEGENERACY_RTOL
        from vane.modal.labeler import _degenerate_clusters

        step = 0.9 * DEGENERACY_RTOL
        eigenvalues = np.array([1.0, 1.0 + step, 1.0 + 2.0 * step, 5.0], dtype=complex)
        flags = np.array([True, True, True, False])
        assert _degenerate_clusters(eigenvalues, flags) == [[0, 1, 2]]

    def test_threshold_controls_acceptance(self) -> None:
        """Known participation below the threshold leaves the mode Unidentified."""
        weights = np.array([0.3, 0.1, 0.6])
        descriptions = [_TOWER_FA, _DRIVETRAIN, _AERO]
        strict = label_mode(weights, descriptions, min_known_fraction=0.5)
        assert strict.category is DofCategory.UNKNOWN
        assert strict.confidence == pytest.approx(0.6)
        loose = label_mode(weights, descriptions, min_known_fraction=0.4)
        assert loose.category is DofCategory.TOWER_FORE_AFT_1
        assert loose.confidence == pytest.approx(0.3)
        # The plain plurality vote (the default) lets the unclassified state win.
        assert label_mode(weights, descriptions).category is DofCategory.UNKNOWN

    def test_solution_threshold_keyword(self) -> None:
        """label_solution forwards its threshold to the vote."""
        solution = _module_dominated_solution()
        labels = label_solution(solution, min_known_participation=0.0)
        assert all(label.category is not DofCategory.UNKNOWN for label in labels)

    def test_label_modes_forwards_threshold(self) -> None:
        """label_modes applies the classified-share threshold when given."""
        from vane.modal.participation import compute_participation

        modes = np.array([[0.2], [1.0]], dtype=np.complex128)
        participation = compute_participation(modes)
        plain = label_modes(participation, [_TOWER_FA, _AERO])
        filtered = label_modes(
            participation, [_TOWER_FA, _AERO], min_known_fraction=0.1
        )
        assert plain[0].category is DofCategory.UNKNOWN
        assert filtered[0].category is DofCategory.TOWER_FORE_AFT_1

    @pytest.mark.parametrize("value", [-0.1, 1.5, float("nan"), float("inf")])
    def test_out_of_range_threshold_raises(self, value: float) -> None:
        """A threshold outside [0, 1] or non-finite is rejected."""
        with pytest.raises(ValueError, match="participation threshold"):
            label_solution(_swamped_tower_solution(), min_known_participation=value)
        with pytest.raises(ValueError, match="participation threshold"):
            label_mode(np.ones(1), [_TOWER_FA], min_known_fraction=value)

    @pytest.mark.parametrize("value", ["0.5", True, None])
    def test_non_numeric_threshold_raises(self, value: object) -> None:
        """A non-numeric (or boolean) solution threshold is a TypeError."""
        bad: Any = value
        with pytest.raises(TypeError, match="participation threshold"):
            label_solution(_swamped_tower_solution(), min_known_participation=bad)

    def test_missing_left_eigenvectors_fall_back_to_magnitude(self) -> None:
        """Without left eigenvectors the raw-magnitude plurality vote is used."""
        solution = _swamped_tower_solution()
        stripped = replace(
            solution,
            full_left_eigenvectors=np.empty((0, 0), dtype=np.complex128),
        )
        label = label_solution(stripped)[0]
        expected = label_mode(
            np.abs(solution.mode_shapes[:, 0]), solution.dof_descriptions
        )
        assert label == expected

    def test_unresolvable_mode_falls_back_to_magnitude(self) -> None:
        """A mode whose left-right product vanishes uses the magnitude vote."""
        solution = _swamped_tower_solution()
        orthogonal = np.zeros_like(solution.full_left_eigenvectors)
        label = label_solution(replace(solution, full_left_eigenvectors=orthogonal))[0]
        assert label.category is DofCategory.UNKNOWN

    def test_tower_mode_drops_raw_magnitude_whirl_type(self) -> None:
        """A tower-participating mode is not given a blade whirl type."""
        from vane.modal.eigensolver import ModalSolution

        blade = "ED 1st flapwise blade 1 (internal DOF index = DOF_BF(1,1)), m"
        right = np.array([[0.1], [1.0], [0.1j], [1.0j]], dtype=np.complex128)
        left = np.array([[1.0], [0.01], [1.0], [0.01]], dtype=np.complex128)
        solution = ModalSolution(
            eigenvalues=np.array([1.0j], dtype=np.complex128),
            natural_frequencies_hz=np.array([1.0 / (2.0 * np.pi)]),
            damping_ratios=np.zeros(1),
            damped_frequencies_hz=np.array([1.0 / (2.0 * np.pi)]),
            mode_shapes=right[:2].copy(),
            full_eigenvectors=right,
            n_rigid_body_modes=0,
            dof_descriptions=[_TOWER_FA, blade],
            dof_mbc_coordinates=["", "collective"],
            full_left_eigenvectors=left,
        )
        label = label_solution(solution)[0]
        assert label.category is DofCategory.TOWER_FORE_AFT_1
        assert label.multiblade is None
        assert label.label == "1st Tower Fore-Aft"

        # With the participation moved onto the blade, the whirl type is kept.
        blade_left = np.array([[0.01], [1.0], [0.01], [1.0]], dtype=np.complex128)
        blade_label = label_solution(
            replace(solution, full_left_eigenvectors=blade_left)
        )[0]
        assert blade_label.category is DofCategory.BLADE_FLAP_1
        assert blade_label.multiblade == "collective"
        assert blade_label.label == "1st Blade Flap (Collective)"

    def test_unidentified_mode_keeps_whirl_type(self) -> None:
        """An Unidentified rotating mode keeps its multi-blade suffix."""
        from vane.modal.eigensolver import ModalSolution

        rotating = "AD B1N001 axial induction, -"
        right = np.array([[0.1], [1.0], [0.1j], [1.0j]], dtype=np.complex128)
        left = np.array([[0.01], [1.0], [0.01], [1.0]], dtype=np.complex128)
        solution = ModalSolution(
            eigenvalues=np.array([1.0j], dtype=np.complex128),
            natural_frequencies_hz=np.array([1.0 / (2.0 * np.pi)]),
            damping_ratios=np.zeros(1),
            damped_frequencies_hz=np.array([1.0 / (2.0 * np.pi)]),
            mode_shapes=right[:2].copy(),
            full_eigenvectors=right,
            n_rigid_body_modes=0,
            dof_descriptions=[_TOWER_FA, rotating],
            dof_mbc_coordinates=["", "collective"],
            full_left_eigenvectors=left,
        )
        label = label_solution(solution)[0]
        assert label.category is DofCategory.UNKNOWN
        assert label.label == "Unidentified (Collective)"
        assert label.multiblade == "collective"


_SURGE = "ED Platform horizontal surge translation DOF (internal DOF index = DOF_Sg), m"
_PITCH = "ED Platform pitch tilt rotation DOF (internal DOF index = DOF_P), rad"
_SWAY = "ED Platform horizontal sway translation DOF (internal DOF index = DOF_Sw), m"
_TOWER_FA2 = "ED 2nd tower fore-aft bending mode (internal DOF index = DOF_TFA2), m"
_TOWER_SS2 = "ED 2nd tower side-to-side bending (internal DOF index = DOF_TSS2), m"
_FLAP2 = "ED 2nd flapwise of blade 1 (internal DOF index = DOF_BF(1,2)), m"
_EDGE = "ED 1st edgewise of blade {} (internal DOF index = DOF_BE({},1)), m"


def _solution_with_participation(
    frequencies_hz: list[float],
    participation: np.ndarray,
    descriptions: list[str],
    *,
    shapes: np.ndarray | None = None,
    mbc_coordinates: list[str] | None = None,
) -> ModalSolution:
    """Return a solution whose biorthogonal participation is ``participation``.

    Every DOF is second-order; the right and left eigenvectors carry the square
    root of the target participation on the displacement states (zero on the
    velocities), so the normalized left-right product reproduces ``participation``
    column by column. ``shapes`` overrides the reported (displacement) mode shapes.
    """
    from vane.modal.eigensolver import ModalSolution

    weights = np.asarray(participation, dtype=np.float64)
    n_modes = weights.shape[1]
    root = np.sqrt(weights).astype(np.complex128)
    full = np.vstack([root, np.zeros_like(root)])
    frequency = np.asarray(frequencies_hz, dtype=np.float64)
    omega = 2.0 * np.pi * frequency
    return ModalSolution(
        eigenvalues=(-0.01 * omega + 1j * omega).astype(np.complex128),
        natural_frequencies_hz=frequency,
        damping_ratios=np.full(n_modes, 0.01),
        damped_frequencies_hz=frequency,
        mode_shapes=root.copy() if shapes is None else shapes.astype(np.complex128),
        full_eigenvectors=full,
        n_rigid_body_modes=0,
        dof_descriptions=descriptions,
        is_degenerate=np.zeros(n_modes, dtype=np.bool_),
        dof_mbc_coordinates=mbc_coordinates or [],
        full_left_eigenvectors=full.copy(),
    )


class TestSupportPlaneLabels:
    """Tower modes of a fixed-bottom support are tower modes, not platform modes."""

    _DOFS = (_SURGE, _PITCH, _TOWER_FA, _TOWER_FA2, _FLAP2)

    def test_monopile_second_tower_mode_is_tower(self) -> None:
        """Platform translation plus tower bending in one plane names a tower mode.

        The participation is that of the OC3 monopile's 2.42 Hz mode: platform
        surge alone out-votes each tower DOF, but tower bending carries almost half
        of the fore-aft support plane, which as a whole out-votes the blade.
        """
        weights = np.array([0.30, 0.06, 0.09, 0.21, 0.29])
        label = label_mode(weights, self._DOFS, min_known_fraction=0.5)
        assert label.category is DofCategory.TOWER_FORE_AFT_2
        assert label.label == "2nd Tower Fore-Aft"
        assert label.confidence == pytest.approx(0.66 / 0.95)

    def test_floating_rigid_body_mode_stays_platform(self) -> None:
        """A rigid-body platform mode has no tower share and is not regrouped."""
        weights = np.array([0.97, 0.02, 0.01, 0.0, 0.0])
        label = label_mode(weights, self._DOFS, min_known_fraction=0.5)
        assert label.category is DofCategory.PLATFORM_SURGE
        assert label.label == "Platform Surge"

    def test_substructure_mode_with_minor_tower_share_stays_platform(self) -> None:
        """Tower bending below a third of the support plane leaves a platform mode."""
        weights = np.array([0.05, 0.76, 0.12, 0.06, 0.01])
        label = label_mode(weights, self._DOFS, min_known_fraction=0.5)
        assert label.category is DofCategory.PLATFORM_PITCH

    def test_support_plane_must_out_vote_the_rest(self) -> None:
        """A blade-dominated mode stays a blade mode despite tower co-participation."""
        weights = np.array([0.10, 0.02, 0.05, 0.13, 0.70])
        label = label_mode(weights, self._DOFS, min_known_fraction=0.5)
        assert label.category is DofCategory.BLADE_FLAP_2

    def test_side_side_plane(self) -> None:
        """Sway with side-side tower bending names a side-side tower mode."""
        weights = np.array([0.47, 0.27, 0.17, 0.09])
        descriptions = [_SWAY, _TOWER_SS2, _TOWER_SS, _EDGE.format(1, 1)]
        label = label_mode(weights, descriptions, min_known_fraction=0.5)
        assert label.category is DofCategory.TOWER_SIDE_SIDE_2
        assert label.label == "2nd Tower Side-Side"

    def test_plurality_path_is_unchanged(self) -> None:
        """The raw-magnitude plurality vote (mixed units) never regroups."""
        weights = np.array([0.30, 0.06, 0.09, 0.21, 0.29])
        assert label_mode(weights, self._DOFS).category is DofCategory.PLATFORM_SURGE

    def test_fore_aft_modes_are_numbered_by_frequency(self) -> None:
        """The third global fore-aft bending mode is the 3rd, not a 2nd or platform.

        Participation patterned on the OC3 monopile: 1st bending (TFA1 + pitch),
        2nd (surge + TFA2), 3rd (surge + TFA2 again), then a substructure rocking
        mode dominated by platform pitch.
        """
        weights = np.array(
            [
                [0.06, 0.30, 0.39, 0.05],
                [0.27, 0.06, 0.11, 0.76],
                [0.64, 0.09, 0.10, 0.12],
                [0.00, 0.21, 0.37, 0.06],
                [0.03, 0.34, 0.03, 0.01],
            ]
        )
        weights = weights / weights.sum(axis=0)
        solution = _solution_with_participation(
            [0.28, 2.42, 5.26, 14.6], weights, list(self._DOFS)
        )
        labels = label_solution(solution)
        assert [label.label for label in labels] == [
            "1st Tower Fore-Aft",
            "2nd Tower Fore-Aft",
            "3rd Tower Fore-Aft",
            "Platform Pitch",
        ]
        # The category stays the dominant tower DOF.
        assert labels[2].category is DofCategory.TOWER_FORE_AFT_2

    def test_tower_dof_number_is_a_lower_bound(self) -> None:
        """A lone 2nd-tower mode keeps its number when no 1st mode is labelled."""
        weights = np.array([[0.1], [0.0], [0.0], [0.8], [0.1]])
        solution = _solution_with_participation([2.9], weights, list(self._DOFS))
        assert label_solution(solution)[0].label == "2nd Tower Fore-Aft"

    def test_degenerate_pair_shares_a_number(self) -> None:
        """Both modes of a repeated tower eigenvalue get the same label."""
        from vane.modal.eigensolver import compute_modes

        omega = 2.0 * np.pi * 0.5
        a = np.zeros((4, 4))
        a[:2, 2:] = np.eye(2)
        a[2:, :2] = -(omega**2) * np.eye(2)
        a[2:, 2:] = -0.02 * omega * np.eye(2)
        solution = compute_modes(
            a, ndof2=2, ndof1=0, descriptions=[_TOWER_FA, _TOWER_SS]
        )
        assert bool(np.all(solution.is_degenerate))
        first, second = label_solution(solution)
        assert first.label == second.label

    @pytest.mark.parametrize(
        ("number", "text"),
        [
            (1, "1st"),
            (2, "2nd"),
            (3, "3rd"),
            (4, "4th"),
            (11, "11th"),
            (12, "12th"),
            (13, "13th"),
            (21, "21st"),
            (22, "22nd"),
            (111, "111th"),
        ],
    )
    def test_ordinal(self, number: int, text: str) -> None:
        """Ordinals follow English suffix rules."""
        from vane.modal.labeler import _ordinal

        assert _ordinal(number) == text


class TestMultibladeShare:
    """The collective/cyclic split weighs coordinates by energy or participation."""

    def test_mostly_collective_mode_is_collective(self) -> None:
        """A cyclic pair is not counted twice against the collective coordinate.

        Blade energy is ``3|q0|^2 + 1.5(|qc|^2 + |qs|^2)``: here 3.0 collective vs
        1.92 cyclic, although the plain magnitude sum (1.0 vs 1.6) favours cyclic.
        """
        from vane.modal.labeler import _multiblade_type

        coords = ["collective", "cosine", "sine"]
        shape = np.array([1.0, 0.8, 0.8j], dtype=np.complex128)
        assert _multiblade_type(shape, coords) == "collective"

    def test_participation_decides_the_split(self) -> None:
        """Given participation, it decides collective vs cyclic over the shape."""
        from vane.modal.labeler import _multiblade_type

        coords = ["collective", "cosine", "sine"]
        shape = np.array([0.1, 1.0, 1.0j], dtype=np.complex128)
        assert _multiblade_type(shape, coords) == "regressive"
        weights = np.array([0.6, 0.2, 0.2])
        assert _multiblade_type(shape, coords, weights) == "collective"
        # Unusable participation falls back to the blade energy of the shape.
        assert _multiblade_type(shape, coords, np.full(3, np.nan)) == "regressive"

    def test_blade_mode_keeps_type_when_blades_are_a_minority(self) -> None:
        """A blade-labelled mode keeps its type with tower/drivetrain co-participation.

        Patterned on the collective edge / drivetrain / 2nd tower side-side mode of
        a floating turbine: the edge category wins the vote with 0.44 although the
        blade coordinates hold less than half the participation, and the raw shape
        is tower-dominated.
        """
        descriptions = [_TOWER_SS2, _DRIVETRAIN] + [
            _EDGE.format(b, b) for b in (1, 2, 3)
        ]
        weights = np.array([[0.37], [0.19], [0.42], [0.01], [0.01]])
        shapes = np.array([[10.0], [1.0], [1.0], [0.01], [0.01j]])
        solution = _solution_with_participation(
            [3.6],
            weights,
            descriptions,
            shapes=shapes,
            mbc_coordinates=["", "", "collective", "cosine", "sine"],
        )
        label = label_solution(solution)[0]
        assert label.category is DofCategory.BLADE_EDGE_1
        assert label.multiblade == "collective"
        assert label.label == "1st Blade Edge (Collective)"

    def test_unidentified_mode_needs_a_rotor_majority(self) -> None:
        """An Unidentified mode gets no type from a minor rotor share."""
        descriptions = [_AERO, "HD Radiation state 1, -", _EDGE.format(1, 1)]
        weights = np.array([[0.2], [0.7], [0.1]])
        solution = _solution_with_participation(
            [0.5],
            weights,
            descriptions,
            mbc_coordinates=["collective", "", "collective"],
        )
        label = label_solution(solution)[0]
        assert label.category is DofCategory.UNKNOWN
        assert label.multiblade is None
