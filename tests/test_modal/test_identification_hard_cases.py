# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Synthetic tests of hard mode-identification cases (labels, whirl slots, tracking).

Each test reproduces, with a few prescribed degrees of freedom, one failure class
found on the reference-turbine linearizations (see
``test_reference_label_tables.py`` for the real data):

* a hybrid of two families (a monopile tower mode half progressive flap, a
  near-critical collective flap half rotor speed, a flap half unsteady-aero states)
  was named confidently after whichever share was a few percent larger;
* near-coincident members of one blade family mixed and were both given the same
  multi-blade type, a fourth claimant took a whirl type from a purer member, and
  standing (unwhirled) patterns had no type;
* a BeamDyn drivetrain-torsion mode was named a collective edgewise or rotor mode;
* near-coincident modes mixed so strongly that their MAC with the neighbouring
  points fell below the link threshold, splitting a Campbell line;
* a single-point "line" and a track named by its hybrid points were reported as
  confident.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vane.config.dof_map import DofCategory
from vane.modal.eigensolver import ModalSolution, compute_modes
from vane.modal.identifier import heavily_damped_modes, identify_modes
from vane.modal.labeler import (
    LABEL_AMBIGUITY_RATIO,
    label_mode,
    label_solution,
)

_FLAP2 = "ED 2nd flapwise of blade {} (internal DOF index = DOF_BF({},2)), m"
_FLAP1 = "ED 1st flapwise of blade {} (internal DOF index = DOF_BF({},1)), m"
_TFA1 = "ED 1st tower fore-aft bending mode DOF (internal DOF index = DOF_TFA1), m"
_TFA2 = "ED 2nd tower fore-aft bending mode DOF (internal DOF index = DOF_TFA2), m"
_TSS1 = "ED 1st tower side-to-side bending mode DOF (internal DOF index = DOF_TSS1), m"
_SURGE = "ED Platform horizontal surge translation DOF (internal DOF index = DOF_Sg), m"
_PITCH = "ED Platform pitch tilt rotation DOF (internal DOF index = DOF_P), rad"
_GEN = "ED Variable speed generator DOF (internal DOF index = DOF_GeAz), rad"
_DRTR = "ED Drivetrain rotational-flexibility DOF (internal DOF index = DOF_DrTr), rad"
_AERO = "AD B1N001 axial induction, -"

# Second flap of a three-bladed rotor in multi-blade coordinates, then the support.
_FAMILY_DOFS = [
    _FLAP2.format(1, 1),
    _FLAP2.format(2, 2),
    _FLAP2.format(3, 3),
    _TFA2,
    _SURGE,
    _TFA1,
]
_FAMILY_COORDS = ["collective", "cosine", "sine", "", "", ""]

_ROOT2 = math.sqrt(2.0)
# Unit multi-blade shapes: collective, regressive (backward whirl) and progressive.
_COLLECTIVE = np.array([1, 0, 0, 0, 0, 0], dtype=np.complex128)
_REGRESSIVE = np.array([0, 1 / _ROOT2, 1j / _ROOT2, 0, 0, 0], dtype=np.complex128)
_PROGRESSIVE = np.array([0, 1 / _ROOT2, -1j / _ROOT2, 0, 0, 0], dtype=np.complex128)


def _solution(
    frequencies_hz: list[float],
    participation: list[list[float]],
    shapes: list[np.ndarray],
    descriptions: list[str],
    coordinates: list[str],
    *,
    damping: list[float] | None = None,
    rotor_speed_rpm: float = float("nan"),
) -> ModalSolution:
    """Return a solution with prescribed participation, mode shapes and frequencies.

    Every DOF is second-order; the right and left eigenvectors carry the square
    root of the target participation on the displacement states, so the
    biorthogonal participation reproduces ``participation`` (one row per mode).
    """
    weights = np.asarray(participation, dtype=np.float64).T
    n_modes = weights.shape[1]
    root = np.sqrt(weights).astype(np.complex128)
    full = np.vstack([root, np.zeros_like(root)])
    frequency = np.asarray(frequencies_hz, dtype=np.float64)
    zeta = np.full(n_modes, 0.02) if damping is None else np.asarray(damping)
    omega = 2.0 * np.pi * frequency
    eigenvalues = -zeta * omega + 1j * omega * np.sqrt(1.0 - zeta**2)
    return ModalSolution(
        eigenvalues=eigenvalues.astype(np.complex128),
        natural_frequencies_hz=frequency,
        damping_ratios=zeta.astype(np.float64),
        damped_frequencies_hz=frequency * np.sqrt(1.0 - zeta**2),
        mode_shapes=np.column_stack(shapes).astype(np.complex128),
        full_eigenvectors=full,
        n_rigid_body_modes=0,
        dof_descriptions=descriptions,
        is_degenerate=np.zeros(n_modes, dtype=np.bool_),
        dof_mbc_coordinates=coordinates,
        full_left_eigenvectors=full.copy(),
        rotor_speed_rpm=rotor_speed_rpm,
    )


def _family(
    frequencies_hz: list[float],
    participation: list[list[float]],
    shapes: list[np.ndarray],
    **kwargs: object,
) -> ModalSolution:
    return _solution(
        frequencies_hz,
        participation,
        shapes,
        _FAMILY_DOFS,
        _FAMILY_COORDS,
        **kwargs,  # type: ignore[arg-type]
    )


def _mix(collective: float, cyclic: np.ndarray) -> np.ndarray:
    """Return a unit shape ``sqrt(c) * collective + sqrt(1 - c) * cyclic``."""
    return math.sqrt(collective) * _COLLECTIVE + math.sqrt(1.0 - collective) * cyclic


def _whirl(circularity: float) -> np.ndarray:
    """Return a unit cyclic shape with the given whirl circularity in [-1, 1]."""
    # q_c = 1, q_s = a * i: circularity = 2a / (1 + a^2).
    a = (1.0 - math.sqrt(1.0 - circularity**2)) / circularity if circularity else 0.0
    shape = np.array([0, 1, 1j * a, 0, 0, 0], dtype=np.complex128)
    return shape / np.linalg.norm(shape)


class TestCategoryAmbiguity:
    """A hybrid of two families is reported ambiguous with its runner-up."""

    _DOFS = (_FLAP2.format(1, 1), _SURGE, _TFA2, _TFA1, _PITCH)

    def test_tower_blade_hybrid_is_ambiguous(self) -> None:
        """The IEA 10-MW monopile 2nd fore-aft mode below rated (1.71 Hz)."""
        label = label_mode(
            np.array([0.51, 0.20, 0.16, 0.06, 0.03]), self._DOFS, min_known_fraction=0.5
        )
        assert label.label == "2nd Blade Flap"
        assert label.is_ambiguous
        assert label.alternative == "2nd Tower Fore-Aft"
        assert label.alternative_category is DofCategory.TOWER_FORE_AFT_2
        assert label.alternative_confidence == pytest.approx(0.45 / 0.96)

    def test_clean_mode_is_not_ambiguous(self) -> None:
        """A mode carried mostly by one family is named with confidence."""
        label = label_mode(
            np.array([0.92, 0.02, 0.03, 0.02, 0.01]), self._DOFS, min_known_fraction=0.5
        )
        assert label.label == "2nd Blade Flap"
        assert not label.is_ambiguous
        assert label.alternative == "2nd Tower Fore-Aft"

    def test_threshold_is_the_runner_up_ratio(self) -> None:
        """The flag flips where the runner-up reaches LABEL_AMBIGUITY_RATIO."""
        winner = 0.5
        below = label_mode(
            np.array([winner, 0.0, winner * LABEL_AMBIGUITY_RATIO * 0.99, 0.0, 0.0]),
            self._DOFS,
            min_known_fraction=0.0,
        )
        at = label_mode(
            np.array([winner, 0.0, winner * LABEL_AMBIGUITY_RATIO, 0.0, 0.0]),
            self._DOFS,
            min_known_fraction=0.0,
        )
        assert not below.is_ambiguous
        assert at.is_ambiguous

    def test_flap_and_rotor_speed_hybrid_is_ambiguous(self) -> None:
        """IEA 15-MW at 25 m/s: 40 % collective flap, 39 % generator azimuth."""
        label = label_mode(
            np.array([0.40, 0.39, 0.07, 0.14]),
            [_FLAP1.format(1, 1), _GEN, _TFA1, _AERO],
            min_known_fraction=0.5,
        )
        assert label.category is DofCategory.BLADE_FLAP_1
        assert label.is_ambiguous
        assert label.alternative == "Rotor/Generator"

    def test_aero_state_hybrid_is_ambiguous(self) -> None:
        """IEA 22-MW rotor with unsteady aero: 54 % flap, 45 % unclassified states."""
        label = label_mode(
            np.array([0.54, 0.45, 0.01]),
            [_FLAP1.format(1, 1), _AERO, _TFA1],
            min_known_fraction=0.5,
        )
        assert label.category is DofCategory.BLADE_FLAP_1
        assert label.is_ambiguous
        assert label.alternative == "Unidentified"

    def test_unidentified_mode_with_a_strong_structural_share(self) -> None:
        """An aero-dominated mode with a near-equal flap share is ambiguous too."""
        label = label_mode(
            np.array([0.45, 0.55]), [_FLAP1.format(1, 1), _AERO], min_known_fraction=0.5
        )
        assert label.label == "Unidentified"
        assert label.is_ambiguous
        assert label.alternative == "1st Blade Flap"


class TestWhirlSlots:
    """Each blade family has one collective, regressive and progressive member."""

    def test_clean_family_is_untouched(self) -> None:
        """A well-separated triplet keeps its types and is not ambiguous."""
        solution = _family(
            [1.78, 2.01, 2.07],
            [
                [0, 0.45, 0.45, 0.1, 0, 0],
                [0.95, 0, 0, 0.05, 0, 0],
                [0, 0.45, 0.45, 0.1, 0, 0],
            ],
            [_REGRESSIVE, _COLLECTIVE, _PROGRESSIVE],
        )
        labels = label_solution(solution)
        assert [label.multiblade for label in labels] == [
            "regressive",
            "collective",
            "progressive",
        ]
        assert not any(label.is_ambiguous for label in labels)

    def test_mixed_collective_pair_is_split(self) -> None:
        """NREL 5-MW OC3 at 4-6 m/s: two near-coincident modes both 'collective'.

        The 2nd flap collective and progressive members 0.9 % apart mix almost
        evenly (collective share 0.48 and 0.51); the joint assignment gives the
        collective type to one and the progressive to the other, both ambiguous.
        """
        solution = _family(
            [1.78, 2.008, 2.026],
            [
                [0, 0.40, 0.37, 0.14, 0.04, 0.05],
                [0.48, 0.05, 0.38, 0.05, 0.02, 0.02],
                [0.51, 0.09, 0.35, 0.03, 0.01, 0.01],
            ],
            [_REGRESSIVE, _mix(0.53, _whirl(-0.59)), _mix(0.54, _whirl(-0.75))],
        )
        labels = label_solution(solution)
        assert [label.label for label in labels] == [
            "2nd Blade Flap (Regressive)",
            "2nd Blade Flap (Collective)",
            "2nd Blade Flap (Progressive)",
        ]
        assert labels[1].is_ambiguous
        assert labels[2].is_ambiguous
        assert labels[2].alternative == "2nd Blade Flap (Collective)"

    def test_fourth_claimant_takes_its_tower_alternative(self) -> None:
        """IEA 10-MW at 4-8 m/s: two 'progressive' 2nd flap modes.

        The purer one (69 % flap) keeps the progressive type; the hybrid is renamed
        after its tower alternative and flagged, so the monopile's 2nd fore-aft line
        keeps its name. The fixture puts the hybrid exactly on the boundary (fore-aft
        vote 3/7, flap 4/7: a ratio of exactly ``LABEL_AMBIGUITY_RATIO``); a tie counts
        as reaching the ratio on every NumPy build (it once flipped with NumPy 2.5).
        """
        solution = _family(
            [1.419, 1.593, 1.594, 1.713],
            [
                [0, 0.48, 0.48, 0.02, 0.01, 0.01],
                [0.91, 0.01, 0.06, 0.01, 0.01, 0.0],
                [0.06, 0.26, 0.40, 0.13, 0.09, 0.02],
                [0.02, 0.45, 0.09, 0.16, 0.20, 0.06],
            ],
            [_REGRESSIVE, _COLLECTIVE, _whirl(-0.41), _whirl(-0.55)],
        )
        labels = label_solution(solution)
        assert labels[2].label == "2nd Blade Flap (Progressive)"
        assert labels[3].label == "2nd Tower Fore-Aft"
        assert labels[3].category is DofCategory.TOWER_FORE_AFT_2
        assert labels[3].is_ambiguous
        assert labels[3].alternative == "2nd Blade Flap (Progressive)"
        names = [label.label for label in labels]
        assert len(set(names)) == len(names)

    def test_regressive_above_progressive_is_reordered(self) -> None:
        """A weak whirl sign that puts the regressive above the progressive."""
        solution = _family(
            [1.0, 1.1, 1.2],
            [[0, 0.5, 0.5, 0, 0, 0], [1.0, 0, 0, 0, 0, 0], [0, 0.5, 0.5, 0, 0, 0]],
            [_whirl(-0.3), _COLLECTIVE, _whirl(0.3)],
        )
        labels = label_solution(solution)
        assert labels[0].multiblade == "regressive"
        assert labels[2].multiblade == "progressive"
        assert labels[0].is_ambiguous
        assert labels[2].is_ambiguous

    def test_standing_pattern_gets_the_free_whirl_type(self) -> None:
        """IEA 3.4-MW at 4-6 m/s: a standing 2nd flap pattern above the regressive."""
        solution = _family(
            [2.17, 2.36, 2.44],
            [
                [0, 0.48, 0.48, 0.04, 0, 0],
                [0.30, 0.12, 0.48, 0.10, 0, 0],
                [0.92, 0.04, 0.04, 0, 0, 0],
            ],
            [_REGRESSIVE, _mix(0.30, _whirl(0.11)), _COLLECTIVE],
        )
        labels = label_solution(solution)
        assert labels[1].label == "2nd Blade Flap (Progressive)"
        assert labels[1].is_ambiguous
        assert labels[1].alternative == "2nd Blade Flap (Cyclic)"

    def test_undecided_split_without_conflict_is_flagged(self) -> None:
        """A collective share within the ambiguity ratio of the cyclic one."""
        solution = _family(
            [1.0],
            [[0.50, 0.22, 0.22, 0.06, 0, 0]],
            [_mix(0.5, _whirl(-0.8))],
        )
        (label,) = label_solution(solution)
        assert label.multiblade == "collective"
        assert label.is_ambiguous
        assert label.alternative == "2nd Blade Flap (Progressive)"

    def test_heavily_damped_images_keep_their_whirl(self) -> None:
        """The fixed-frame images of an overdamped blade mode are not reassigned.

        IEA 15-MW above rated: both images of the overdamped 1st flap whirl
        forward at the rotor speed; they are not members of a whirl pair.
        """
        rpm = 7.5
        omega = rpm * math.pi / 30.0
        sigmas = (2.0, 11.0)
        frequencies = [math.hypot(s, omega) / (2.0 * math.pi) for s in sigmas]
        damping = [s / math.hypot(s, omega) for s in sigmas]
        solution = _family(
            frequencies,
            [[0, 0.5, 0.5, 0, 0, 0], [0, 0.5, 0.5, 0, 0, 0]],
            [_PROGRESSIVE, _PROGRESSIVE],
            damping=damping,
            rotor_speed_rpm=rpm,
        )
        labels = label_solution(solution)
        assert [label.multiblade for label in labels] == ["progressive", "progressive"]
        assert not any(label.is_ambiguous for label in labels)


_BD_NODES = (2, 3, 4, 5, 6)


def _bd_rx(node: int) -> str:
    return (
        f"BD_1 finite element node {node} (number of elements = 1; element order = 5)"
        " rotational displacement in X, rad"
    )


def _beamdyn_solution(
    profile: list[float], *, generator: bool, generator_share: float = 0.0
) -> ModalSolution:
    """Return one collective edgewise BeamDyn mode with a drivetrain DOF.

    ``profile`` is the root-to-tip collective section rotation about X.
    """
    descriptions = [_bd_rx(node) for node in _BD_NODES] + [_DRTR]
    coordinates = ["collective"] * len(_BD_NODES) + [""]
    shape = [*profile, 0.01]
    blade = [abs(value) for value in profile]
    blade_total = sum(blade)
    participation = [b / blade_total * (1.0 - generator_share - 0.02) for b in blade]
    participation.append(0.02)
    if generator:
        descriptions.append(_GEN)
        coordinates.append("")
        shape.append(0.2)
        participation.append(generator_share)
    return _solution(
        [0.66],
        [participation],
        [np.asarray(shape, dtype=np.complex128)],
        descriptions,
        coordinates,
    )


class TestBeamDynDrivetrain:
    """A BeamDyn drivetrain-torsion mode is named as such."""

    def test_locked_generator_rotor_swing_is_drivetrain(self) -> None:
        """NREL 5-MW BeamDyn, DrTrDOF on, GenDOF off: the 0.663 Hz mode.

        The blade roots rotate in phase with the tips by half the tip rotation:
        the whole rotor swings against the shaft spring.
        """
        (label,) = label_solution(
            _beamdyn_solution([-0.50, -0.69, -0.90, -0.99, -1.0], generator=False)
        )
        assert label.label == "Drivetrain Torsion"
        assert label.category is DofCategory.DRIVETRAIN_TORSION

    def test_counter_rotating_hub_is_edge_bending(self) -> None:
        """The 2.71 Hz mode of the same model: the hub counter-rotates."""
        (label,) = label_solution(
            _beamdyn_solution([0.24, 0.06, -0.50, -0.95, -1.0], generator=False)
        )
        assert label.label == "1st Blade Edge (Collective)"

    def test_free_generator_elastic_mode_is_drivetrain(self) -> None:
        """NREL 5-MW BeamDyn with aero, generator free: the 1.68 Hz mode."""
        (label,) = label_solution(
            _beamdyn_solution(
                [0.21, -0.19, -0.70, -0.96, -1.0], generator=True, generator_share=0.68
            )
        )
        assert label.label == "Drivetrain Torsion"

    def test_rigid_rotor_mode_stays_rotor(self) -> None:
        """A rigid rotor rotation with the generator is the rotor-azimuth mode."""
        (label,) = label_solution(
            _beamdyn_solution(
                [-1.0, -1.0, -1.0, -1.0, -1.0], generator=True, generator_share=0.68
            )
        )
        assert label.label == "Rotor/Generator"


def _cluster_sweep() -> list[ModalSolution]:
    """Return three points whose 2nd flap collective/regressive pair mixes in between.

    At the middle point the two members are 0.07 % apart (IEA 22-MW monopile at
    10 m/s) and each is a 46/54 mix of the pure shapes, so the plain MAC with the
    neighbouring points' pure members is below 0.5 for the right partner.
    """
    mixed_angle = math.acos(math.sqrt(0.46))
    first = math.cos(mixed_angle) * _COLLECTIVE + math.sin(mixed_angle) * _REGRESSIVE
    second = -math.sin(mixed_angle) * _COLLECTIVE + math.cos(mixed_angle) * _REGRESSIVE
    pure = [[0.95, 0, 0, 0.05, 0, 0], [0, 0.47, 0.47, 0.06, 0, 0]]
    mixed = [[0.55, 0.20, 0.20, 0.05, 0, 0], [0.38, 0.28, 0.29, 0.05, 0, 0]]
    return [
        _family([3.387, 3.432], pure, [_COLLECTIVE, _REGRESSIVE]),
        _family([3.3110, 3.3133], mixed, [first, second]),
        _family([3.338, 3.375], pure[::-1], [_REGRESSIVE, _COLLECTIVE]),
    ]


class TestNearCoincidentTracking:
    """A line keeps its identity through a near-coincident, mixed pair."""

    def test_mixed_pair_is_labelled_one_per_type(self) -> None:
        """The mixed members still get one collective and one regressive label."""
        labels = label_solution(_cluster_sweep()[1])
        assert [label.multiblade for label in labels] == ["collective", "regressive"]

    @pytest.mark.parametrize("strategy", ["single_path", "simultaneous"])
    def test_each_line_is_one_track(self, strategy: str) -> None:
        """The collective and regressive lines each span all three points."""
        result = identify_modes(_cluster_sweep(), strategy=strategy)  # type: ignore[arg-type]
        by_label = {track.label.label: track for track in result.tracks}
        assert len(result.tracks) == 2
        collective = by_label["2nd Blade Flap (Collective)"]
        regressive = by_label["2nd Blade Flap (Regressive)"]
        assert collective.operating_points == [0, 1, 2]
        assert collective.mode_indices == [0, 0, 1]
        assert regressive.mode_indices == [1, 1, 0]
        assert collective.label_agreement == 1.0
        assert not collective.is_ambiguous


def _support_sweep(extra: bool) -> ModalSolution:
    """Return tower fore-aft and side-side modes, plus a flap mode when ``extra``."""
    dofs = [_TFA1, _TSS1, _FLAP1.format(1, 1)]
    frequencies = [0.32, 0.33]
    participation = [[0.95, 0.03, 0.02], [0.03, 0.95, 0.02]]
    shapes = [np.array([1, 0, 0], dtype=complex), np.array([0, 1, 0], dtype=complex)]
    if extra:
        frequencies.append(0.7)
        participation.append([0.02, 0.02, 0.96])
        shapes.append(np.array([0, 0, 1], dtype=complex))
    return _solution(frequencies, participation, shapes, dofs, ["", "", ""])


class TestTrackFlags:
    """A track's identity is reported in doubt when nothing confirms it."""

    def test_single_point_line_is_ambiguous(self) -> None:
        """A mode seen at one point of a multi-point sweep forms no confirmed line."""
        result = identify_modes(
            [_support_sweep(False), _support_sweep(True), _support_sweep(False)]
        )
        lone = [t for t in result.tracks if len(t.operating_points) == 1]
        assert len(lone) == 1
        assert lone[0].is_ambiguous
        assert all(
            not t.is_ambiguous for t in result.tracks if len(t.operating_points) == 3
        )

    def test_single_point_sweep_is_not_flagged(self) -> None:
        """With one operating point every mode is a single-point track."""
        result = identify_modes([_support_sweep(True)])
        assert not any(track.is_ambiguous for track in result.tracks)

    def test_hybrid_points_do_not_name_a_clear_line(self) -> None:
        """One clean tower point outweighs two hybrid points of the same line."""
        dofs = (_TFA2, _SURGE, _FLAP2.format(1, 1), _TFA1)
        clean = _solution(
            [1.65],
            [[0.60, 0.25, 0.10, 0.05]],
            [np.array([1, 0.3, 0.2, 0.1])],
            list(dofs),
            [""] * 4,
        )
        hybrid = _solution(
            [1.71],
            [[0.25, 0.15, 0.55, 0.05]],
            [np.array([1, 0.3, 0.3, 0.1])],
            list(dofs),
            [""] * 4,
        )
        result = identify_modes([hybrid, hybrid, clean])
        (track,) = result.tracks
        assert track.label.label == "2nd Tower Fore-Aft"
        assert track.label_agreement == pytest.approx(1 / 3)
        assert [point.is_ambiguous for point in track.point_labels] == [
            True,
            True,
            False,
        ]

    def test_point_labels_stay_aligned_with_operating_points(self) -> None:
        """A point without DOF descriptions empties the list rather than shifting it.

        The labelled points were compressed together, so the last label was
        paired with the second operating point.
        """
        from dataclasses import replace

        dofs = (_TFA2, _SURGE, _FLAP2.format(1, 1), _TFA1)
        clean = _solution(
            [1.65],
            [[0.60, 0.25, 0.10, 0.05]],
            [np.array([1, 0.3, 0.2, 0.1])],
            list(dofs),
            [""] * 4,
        )
        bare = replace(clean, dof_descriptions=[])
        result = identify_modes([clean, bare, clean])
        (track,) = result.tracks
        assert track.operating_points == [0, 1, 2]
        assert track.point_labels == []
        assert track.label.label == "2nd Tower Fore-Aft"
        assert track.label_agreement == pytest.approx(1.0)


def _stiffness_model(order: list[int]) -> tuple[np.ndarray, list[str]]:
    """Return a 3-DOF state matrix and descriptions with the DOFs in ``order``."""
    dofs = [_TFA1, _TSS1, _FLAP1.format(1, 1)]
    stiffness = np.array([[4.0, 0.3, 0.1], [0.3, 4.4, 0.2], [0.1, 0.2, 19.0]])
    damping = 0.02 * stiffness + 0.01 * np.eye(3)
    p = np.eye(3)[order]
    k = p @ stiffness @ p.T
    c = p @ damping @ p.T
    a = np.zeros((6, 6))
    a[:3, 3:] = np.eye(3)
    a[3:, :3] = -k
    a[3:, 3:] = -c
    names = [dofs[i] for i in order]
    velocities = [f"ED First time derivative of {name}" for name in names]
    return a, names + velocities


class TestDeterminism:
    """Results do not depend on the state ordering or the solver's mode order."""

    def test_state_permutation_gives_the_same_modes_and_labels(self) -> None:
        """Reordering the DOFs leaves frequencies, damping and labels unchanged."""
        results = []
        for order in ([0, 1, 2], [2, 0, 1], [1, 2, 0]):
            a, descriptions = _stiffness_model(order)
            solution = compute_modes(a, 3, 0, descriptions=descriptions)
            results.append(
                (
                    solution.natural_frequencies_hz,
                    solution.damping_ratios,
                    [label.label for label in label_solution(solution)],
                )
            )
        for frequencies, damping, labels in results[1:]:
            np.testing.assert_allclose(frequencies, results[0][0], rtol=1e-12)
            np.testing.assert_allclose(damping, results[0][1], rtol=1e-9)
            assert labels == results[0][2]

    def test_equal_natural_frequencies_are_ordered_by_decay(self) -> None:
        """Two modes of equal natural frequency come out in decay-rate order."""
        omega = 2.0
        blocks = [
            np.array([[0.0, 1.0], [-(omega**2), -2.0 * zeta * omega]])
            for zeta in (0.30, 0.05)
        ]
        for first, second in ((0, 1), (1, 0)):
            a = np.zeros((4, 4))
            a[:2, :2] = blocks[first]
            a[2:, 2:] = blocks[second]
            solution = compute_modes(a, 0, 4)
            assert solution.damping_ratios[0] == pytest.approx(0.05)
            assert solution.damping_ratios[1] == pytest.approx(0.30)

    def test_modes_from_mbc_carries_the_rotor_speed(self) -> None:
        """The labeller can read whirl damping in the rotating frame."""
        from vane.io.mbc_transform import MBCResult
        from vane.modal.eigensolver import modes_from_mbc

        a, descriptions = _stiffness_model([0, 1, 2])
        result = MBCResult(
            avg_a=a,
            avg_b=None,
            avg_c=None,
            avg_d=None,
            ndof2=3,
            ndof1=0,
            state_descriptions=descriptions,
            n_blades=0,
            performed_transformation=False,
            rotor_speed_rpm=9.5,
            wind_speed=8.0,
            azimuths_deg=np.zeros(1),
        )
        assert modes_from_mbc(result).rotor_speed_rpm == pytest.approx(9.5)


class TestWhirlFit:
    """The per-point whirl assignment never gives a mode a type it does not carry."""

    def test_pure_whirl_is_never_collective(self) -> None:
        """A third regressive-whirling member is left untyped, not called collective.

        The free collective slot used to go to a pure whirl (no collective content)
        because an assignment adding zero fit tied with leaving the member out.
        """
        cyclic = [0, 0.48, 0.48, 0.04, 0, 0]
        solution = _family(
            [1.78, 2.07, 2.20],
            [cyclic, cyclic, cyclic],
            [_REGRESSIVE, _PROGRESSIVE, _mix(0.01, _REGRESSIVE)],
        )
        labels = label_solution(solution)
        assert labels[0].label == "2nd Blade Flap (Regressive)"
        assert labels[1].label == "2nd Blade Flap (Progressive)"
        assert labels[2].label == "2nd Blade Flap"
        assert labels[2].multiblade is None
        assert labels[2].is_ambiguous
        assert labels[2].alternative == "2nd Blade Flap (Regressive)"

    def test_minor_share_does_not_rename_a_fourth_claimant(self) -> None:
        """A 96 % flap claimant keeps its family name, not its 4 % tower runner-up."""
        cyclic = [0, 0.48, 0.48, 0.04, 0, 0]
        solution = _family(
            [1.78, 2.0, 2.07, 2.30],
            [cyclic, [0.96, 0, 0, 0.04, 0, 0], cyclic, cyclic],
            [_REGRESSIVE, _COLLECTIVE, _PROGRESSIVE, _whirl(-0.9)],
        )
        labels = label_solution(solution)
        assert [label.label for label in labels[:3]] == [
            "2nd Blade Flap (Regressive)",
            "2nd Blade Flap (Collective)",
            "2nd Blade Flap (Progressive)",
        ]
        assert labels[3].label == "2nd Blade Flap"
        assert labels[3].category is DofCategory.BLADE_FLAP_2
        assert labels[3].is_ambiguous
        assert labels[3].alternative == "2nd Blade Flap (Progressive)"


_BD_SPAN = np.array([(node - 1) / 5 for node in _BD_NODES])


def _bd_ry(blade: int, node: int) -> str:
    return (
        f"BD_{blade} finite element node {node} (number of elements = 1; element "
        "order = 5) rotational displacement in Y, rad"
    )


# BeamDyn flapwise section rotations of the three multi-blade coordinates per node.
_BD_DOFS = [_bd_ry(blade, node) for node in _BD_NODES for blade in (1, 2, 3)]
_BD_COORDS = ["collective", "cosine", "sine"] * len(_BD_NODES)


def _bd_shape(order: int, kind: str, circularity: float = 1.0) -> np.ndarray:
    """Return a BeamDyn flap shape of the given spanwise order and multi-blade type."""
    profile = np.sin((order - 0.5) * np.pi * _BD_SPAN)
    shape = np.zeros(3 * len(_BD_NODES), dtype=np.complex128)
    if kind == "collective":
        shape[0::3] = profile
    else:
        a = (1.0 - math.sqrt(1.0 - circularity**2)) / circularity
        shape[1::3] = profile
        shape[2::3] = 1j * a * profile
    return shape / np.linalg.norm(shape)


def _bd_solution(
    frequencies_hz: list[float], shapes: list[np.ndarray]
) -> ModalSolution:
    participation = [
        list(np.abs(shape) ** 2 / np.sum(np.abs(shape) ** 2)) for shape in shapes
    ]
    return _solution(frequencies_hz, participation, shapes, _BD_DOFS, _BD_COORDS)


class TestBeamDynWhirlNumbering:
    """A BeamDyn member with the wrong whirl type does not shift the numbering."""

    def test_weak_whirl_member_is_retyped_in_its_family(self) -> None:
        """The 1st flap's progressive member, read as regressive, stays 1st flap.

        BeamDyn modes are numbered within their multi-blade type, so the mis-typed
        member used to be counted as the second regressive mode: it became
        "2nd Blade Flap (Regressive)" and the true 2nd flap regressive the 3rd.
        """
        labels = label_solution(
            _bd_solution(
                [0.6, 0.7, 0.8, 1.8, 2.0, 2.1],
                [
                    _bd_shape(1, "cyclic", 1.0),
                    _bd_shape(1, "collective"),
                    _bd_shape(1, "cyclic", 0.3),
                    _bd_shape(2, "cyclic", 1.0),
                    _bd_shape(2, "collective"),
                    _bd_shape(2, "cyclic", -1.0),
                ],
            )
        )
        assert [label.label for label in labels] == [
            "1st Blade Flap (Regressive)",
            "1st Blade Flap (Collective)",
            "1st Blade Flap (Progressive)",
            "2nd Blade Flap (Regressive)",
            "2nd Blade Flap (Collective)",
            "2nd Blade Flap (Progressive)",
        ]
        assert labels[2].is_ambiguous
        assert labels[2].alternative == "1st Blade Flap (Regressive)"
        assert not any(label.is_ambiguous for k, label in enumerate(labels) if k != 2)

    def test_families_sharing_a_shape_order_are_split_by_frequency(self) -> None:
        """A 3rd flap with an underestimated (2nd-order) shape keeps its own family."""
        labels = label_solution(
            _bd_solution(
                [1.8, 2.0, 2.1, 4.6, 5.0, 5.1],
                [
                    _bd_shape(2, "cyclic", 1.0),
                    _bd_shape(2, "collective"),
                    _bd_shape(2, "cyclic", 0.3),
                    _bd_shape(2, "cyclic", 1.0),
                    _bd_shape(2, "collective"),
                    _bd_shape(2, "cyclic", -1.0),
                ],
            )
        )
        assert [label.label for label in labels] == [
            "2nd Blade Flap (Regressive)",
            "2nd Blade Flap (Collective)",
            "2nd Blade Flap (Progressive)",
            "3rd Blade Flap (Regressive)",
            "3rd Blade Flap (Collective)",
            "3rd Blade Flap (Progressive)",
        ]

    def test_underestimated_order_keeps_its_own_type(self) -> None:
        """A 2nd flap regressive with a 1st-order shape is not forced into slot 1."""
        labels = label_solution(
            _bd_solution(
                [0.6, 0.7, 1.8, 2.0, 2.1],
                [
                    _bd_shape(1, "cyclic", 1.0),
                    _bd_shape(1, "collective"),
                    _bd_shape(1, "cyclic", 1.0),
                    _bd_shape(2, "collective"),
                    _bd_shape(2, "cyclic", -1.0),
                ],
            )
        )
        assert [label.label for label in labels] == [
            "1st Blade Flap (Regressive)",
            "1st Blade Flap (Collective)",
            "2nd Blade Flap (Regressive)",
            "2nd Blade Flap (Collective)",
            "2nd Blade Flap (Progressive)",
        ]
        assert not any(label.is_ambiguous for label in labels)


def _overdamped_images(rpm: float) -> ModalSolution:
    """Return a lightly damped regressive flap and two overdamped whirl images."""
    omega = rpm * math.pi / 30.0
    sigmas = (2.0, 11.0)
    frequencies = [0.6] + [math.hypot(s, omega) / (2.0 * math.pi) for s in sigmas]
    damping = [0.03] + [s / math.hypot(s, omega) for s in sigmas]
    cyclic = [0, 0.5, 0.5, 0, 0, 0]
    return _family(
        frequencies,
        [cyclic, cyclic, cyclic],
        [_REGRESSIVE, _PROGRESSIVE, _PROGRESSIVE],
        damping=damping,
        rotor_speed_rpm=rpm,
    )


class TestRotorSpeedDefaults:
    """Without explicit rotor speeds each point's own speed is used everywhere."""

    def test_identify_modes_reads_the_solution_rotor_speed(self) -> None:
        """The heavily damped tier matches the one given explicit rotor speeds.

        The labels already read the solutions' rotor speed while the tier did not,
        so overdamped whirl images were tracked with the lightly damped lines.
        """
        sweep = [_overdamped_images(7.5), _overdamped_images(7.6)]
        explicit = identify_modes(sweep, rotor_speeds_rpm=[7.5, 7.6])
        implicit = identify_modes(sweep)
        assert explicit.heavily_damped_modes == [[1, 2], [1, 2]]
        assert implicit.heavily_damped_modes == explicit.heavily_damped_modes
        assert [t.mode_indices for t in implicit.tracks] == [
            t.mode_indices for t in explicit.tracks
        ]

    def test_heavily_damped_modes_reads_the_solution_rotor_speed(self) -> None:
        """The single-point classifier uses the solution's rotor speed by default."""
        solution = _overdamped_images(7.5)
        assert heavily_damped_modes(solution).tolist() == [False, True, True]
        assert (
            heavily_damped_modes(solution, rotor_speed_rpm=7.5).tolist()
            == heavily_damped_modes(solution).tolist()
        )


class TestTrackAmbiguity:
    """Only modes the tier can link to compete for a track's continuation."""

    def test_excluded_twin_does_not_make_a_line_ambiguous(self) -> None:
        """A heavily damped mode of the same shape is no competing continuation."""
        dofs = [_TFA1, _TSS1, _FLAP1.format(1, 1)]
        shape = np.array([1, 0, 0], dtype=complex)
        point = _solution(
            [0.32, 0.45],
            [[0.95, 0.03, 0.02], [0.95, 0.03, 0.02]],
            [shape, shape],
            dofs,
            ["", "", ""],
            damping=[0.02, 0.995],
        )
        result = identify_modes([point, point])
        (track,) = result.tracks
        assert track.mode_indices == [0, 0]
        assert not track.is_ambiguous
        assert result.heavily_damped_modes == [[1], [1]]


class TestRatioTies:
    """Share comparisons against the ambiguity ratio do not depend on rounding."""

    def test_exact_and_rounded_ties_reach_the_ratio(self) -> None:
        """A share at the ratio, or one ulp below it, counts as reaching it."""
        from vane.modal.labeler import LABEL_AMBIGUITY_RATIO, _reaches

        winner = 4.0 / 7.0
        tie = LABEL_AMBIGUITY_RATIO * winner
        assert _reaches(tie, LABEL_AMBIGUITY_RATIO, winner)
        assert _reaches(np.nextafter(tie, 0.0), LABEL_AMBIGUITY_RATIO, winner)
        assert not _reaches(0.99 * tie, LABEL_AMBIGUITY_RATIO, winner)
