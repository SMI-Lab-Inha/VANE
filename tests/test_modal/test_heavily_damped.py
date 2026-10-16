# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for the heavily damped (near-critical) tracking tier of vane.modal.identifier.

Synthetic cases pin the classification and the tier separation; real-data cases
(skipped when the locally produced reference-turbine runs are absent) reproduce the
IEA 15-MW monopile sweep, whose first-flap modes pass critical damping in the
rotating frame above about 9 m/s and used to draw spurious Campbell lines.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest

from vane.campbell.builder import build_campbell, campbell_from_solutions
from vane.campbell.excitation import find_resonances
from vane.io import LinFile, read_lin_file
from vane.io.mbc_transform import mbc3_transform
from vane.modal.eigensolver import ModalSolution, modes_from_mbc
from vane.modal.identifier import (
    DEFAULT_HEAVILY_DAMPED_THRESHOLD,
    _heavily_damped_mask,
    heavily_damped_modes,
    identify_modes,
    rotor_frame_damping_ratios,
)
from vane.modal.labeler import label_solution
from vane.pipeline import ModalPipeline
from vane.viz.campbell_plot import plot_campbell

_TWO_PI = 2.0 * math.pi


def _solution(
    eigenvalues: list[complex], shapes: list[list[float]] | np.ndarray
) -> ModalSolution:
    """Build a ModalSolution from explicit eigenvalues and mode-shape columns."""
    lam = np.asarray(eigenvalues, dtype=np.complex128)
    modulus = np.abs(lam)
    matrix = np.asarray(shapes, dtype=np.complex128)
    return ModalSolution(
        eigenvalues=lam,
        natural_frequencies_hz=modulus / _TWO_PI,
        damping_ratios=-lam.real / modulus,
        damped_frequencies_hz=np.abs(lam.imag) / _TWO_PI,
        mode_shapes=matrix,
        full_eigenvectors=matrix,
        n_rigid_body_modes=0,
        dof_descriptions=[f"DOF {i}" for i in range(matrix.shape[0])],
    )


def _mode(frequency_hz: float, zeta: float) -> complex:
    """Return the upper eigenvalue of a mode with natural frequency and damping."""
    omega = _TWO_PI * frequency_hz
    return complex(-zeta * omega, omega * math.sqrt(1.0 - zeta**2))


class TestRotorFrameDampingRatios:
    """The damping of a whirl mode is read in the rotating frame."""

    def test_whirl_shift_is_undone(self) -> None:
        """Progressive and regressive images of one blade mode give its own ratio."""
        omega = 12.0 * math.pi / 30.0  # 12 rpm
        sigma, omega_r = 2.0, 3.0
        solution = _solution(
            [
                complex(-sigma, omega_r + omega),
                complex(-sigma, omega_r),
                complex(-sigma, omega_r - omega),
            ],
            np.eye(3),
        )
        ratios = rotor_frame_damping_ratios(
            solution, 12.0, ["progressive", "collective", "regressive"]
        )
        expected = sigma / math.hypot(sigma, omega_r)
        np.testing.assert_allclose(ratios, expected, rtol=1e-12)
        # The fixed-frame ratios differ between the three images.
        assert np.ptp(solution.damping_ratios) > 0.05

    def test_non_whirl_modes_keep_the_fixed_frame_ratio(self) -> None:
        """Tower, collective and unlabelled modes are not shifted."""
        solution = _solution([_mode(0.3, 0.1), _mode(0.8, 0.5)], np.eye(2))
        ratios = rotor_frame_damping_ratios(solution, 10.0, [None, "collective"])
        np.testing.assert_allclose(ratios, solution.damping_ratios)

    def test_cyclic_mode_takes_the_smaller_reading(self) -> None:
        """A whirl of undetermined direction is classified conservatively."""
        omega = 10.0 * math.pi / 30.0
        solution = _solution([complex(-1.0, omega)], np.eye(1))
        (ratio,) = rotor_frame_damping_ratios(solution, 10.0, ["cyclic"])
        assert ratio == pytest.approx(1.0 / math.hypot(1.0, 2.0 * omega))

    def test_overdamped_blade_mode_image_reads_critical(self) -> None:
        """A real rotating-frame root seen at the rotor frequency is past critical."""
        omega = 7.5 * math.pi / 30.0
        solution = _solution([complex(-1.5, omega)], np.eye(1))
        assert solution.damping_ratios[0] < 0.9
        (ratio,) = rotor_frame_damping_ratios(solution, 7.5, ["progressive"])
        assert ratio == pytest.approx(1.0)

    def test_length_mismatch_raises(self) -> None:
        """One multi-blade entry per mode is required."""
        solution = _solution([_mode(0.3, 0.1)], np.eye(1))
        with pytest.raises(ValueError, match="multiblade"):
            rotor_frame_damping_ratios(solution, 10.0, [None, None])

    def test_non_finite_speed_raises(self) -> None:
        """A non-finite rotor speed is rejected."""
        solution = _solution([_mode(0.3, 0.1)], np.eye(1))
        with pytest.raises(ValueError, match="rotor_speed_rpm"):
            rotor_frame_damping_ratios(solution, math.nan, [None])


class TestHeavilyDampedModes:
    """Classification of one operating point's modes."""

    def test_threshold_on_fixed_frame_ratio_without_whirl(self) -> None:
        """A non-whirl mode at or above the threshold is heavily damped."""
        solution = _solution(
            [_mode(0.3, 0.05), _mode(0.5, 0.995), _mode(0.7, 0.6)], np.eye(3)
        )
        np.testing.assert_array_equal(
            heavily_damped_modes(solution), [False, True, False]
        )
        np.testing.assert_array_equal(
            heavily_damped_modes(solution, threshold=0.5), [False, True, True]
        )

    def test_whirl_modes_need_the_rotor_speed(self) -> None:
        """Without the rotor speed a whirl mode is never classified."""
        solution = _solution([_mode(0.5, 0.995), _mode(0.6, 0.995)], np.eye(2))
        mask = _heavily_damped_mask(solution, 0.99, None, ["regressive", None])
        np.testing.assert_array_equal(mask, [False, True])

    def test_disabled_threshold_classifies_nothing(self) -> None:
        """A ``None`` threshold leaves every mode in the main tier."""
        solution = _solution([_mode(0.5, 0.999)], np.eye(1))
        assert not _heavily_damped_mask(solution, None, None, [None]).any()

    @pytest.mark.parametrize("threshold", [0.0, -0.1, 1.5, math.nan])
    def test_invalid_threshold_raises(self, threshold: float) -> None:
        """The threshold must lie in (0, 1]."""
        solution = _solution([_mode(0.5, 0.2)], np.eye(1))
        with pytest.raises(ValueError, match="heavily_damped_threshold"):
            heavily_damped_modes(solution, threshold=threshold)

    def test_non_finite_speed_raises(self) -> None:
        """A non-finite rotor speed is rejected."""
        solution = _solution([_mode(0.5, 0.2)], np.eye(1))
        with pytest.raises(ValueError, match="rotor_speed_rpm"):
            heavily_damped_modes(solution, rotor_speed_rpm=math.inf)

    def test_default_threshold_is_near_critical(self) -> None:
        """The documented default sits within 1 % of critical damping."""
        assert pytest.approx(0.99) == DEFAULT_HEAVILY_DAMPED_THRESHOLD


def _steal_sweep() -> list[ModalSolution]:
    """Return a lightly damped line whose continuation meets a near-critical mode.

    The near-critical mode at the second point correlates better with the line's
    first point than the true (slightly rotated) continuation does.
    """
    first = _solution([_mode(1.0, 0.02)], [[1.0], [0.0], [0.0]])
    second = _solution(
        [_mode(1.0, 0.02), _mode(1.0, 0.995)],
        [[0.8, 1.0], [0.6, 0.05], [0.0, 0.0]],
    )
    third = _solution([_mode(1.0, 0.02)], [[0.8], [0.6], [0.0]])
    return [first, second, third]


def _fade_sweep() -> list[ModalSolution]:
    """Return a lightly damped line that passes critical damping after point 2."""
    shape = [[1.0], [0.2]]
    return [
        _solution([_mode(0.65, 0.80)], shape),
        _solution([_mode(0.66, 0.85)], shape),
        _solution([_mode(0.36, 0.995)], shape),
        _solution([_mode(0.30, 0.996)], shape),
    ]


class TestHeavilyDampedTier:
    """identify_modes keeps near-critical modes out of the lightly damped lines."""

    def test_near_critical_mode_cannot_steal_a_line(self) -> None:
        """The lightly damped line runs through all points; the intruder is apart."""
        result = identify_modes(_steal_sweep(), frequency_weight=0.0)
        assert len(result.tracks) == 1
        assert result.tracks[0].mode_indices == [0, 0, 0]
        assert result.heavily_damped_modes == [[], [1], []]
        assert len(result.heavily_damped_tracks) == 1
        heavy = result.heavily_damped_tracks[0]
        assert heavy.is_heavily_damped
        assert heavy.operating_points == [1]
        assert not result.tracks[0].is_heavily_damped

    def test_opt_out_restores_joint_tracking(self) -> None:
        """With the tier disabled the near-critical mode takes over the line again."""
        result = identify_modes(
            _steal_sweep(), frequency_weight=0.0, heavily_damped_threshold=None
        )
        assert len(result.tracks) == 2
        assert [0, 1, 0] in [track.mode_indices for track in result.tracks]
        assert result.heavily_damped_tracks == []
        assert result.heavily_damped_modes == []
        assert result.n_heavily_damped == 0

    def test_line_ends_where_the_mode_passes_critical(self) -> None:
        """A line is not continued into near-critical modes; those form their own."""
        result = identify_modes(_fade_sweep())
        assert [t.operating_points for t in result.tracks] == [[0, 1]]
        assert [t.operating_points for t in result.heavily_damped_tracks] == [[2, 3]]
        assert result.n_heavily_damped == 2
        joint = identify_modes(_fade_sweep(), heavily_damped_threshold=None)
        assert [t.operating_points for t in joint.tracks] == [[0, 1, 2, 3]]

    def test_simultaneous_strategy_also_separates(self) -> None:
        """The tier applies to the simultaneous strategy too."""
        result = identify_modes(
            _steal_sweep(), frequency_weight=0.0, strategy="simultaneous"
        )
        assert result.tracks[0].mode_indices == [0, 0, 0]
        assert len(result.heavily_damped_tracks) == 1

    def test_branch_review_keeps_the_tier(self) -> None:
        """Attaching branch groups does not drop the heavily damped tier."""
        result = identify_modes(_fade_sweep(), branch_review=True)
        assert len(result.heavily_damped_tracks) == 1
        assert result.n_heavily_damped == 2

    def test_rotor_speeds_are_validated(self) -> None:
        """Rotor speeds must match the points and be finite."""
        with pytest.raises(ValueError, match="rotor_speeds_rpm"):
            identify_modes(_fade_sweep(), rotor_speeds_rpm=[10.0])
        with pytest.raises(ValueError, match="rotor_speeds_rpm"):
            identify_modes(_fade_sweep(), rotor_speeds_rpm=[10.0, 10.0, 10.0, np.nan])

    def test_invalid_threshold_raises(self) -> None:
        """identify_modes rejects a threshold outside (0, 1]."""
        with pytest.raises(ValueError, match="heavily_damped_threshold"):
            identify_modes(_fade_sweep(), heavily_damped_threshold=1.2)


class TestContinuityMetric:
    """The optional complex-eigenvalue continuity distance."""

    @staticmethod
    def _sweep() -> list[ModalSolution]:
        # P keeps the decay rate; Q is closer in natural frequency but decays
        # six times faster. Both correlate equally with the first point.
        first = _solution([complex(-0.5, 6.0)], [[1.0], [1.0]])
        second = _solution(
            [complex(-0.5, 6.3), complex(-3.0, 5.3)], [[1.0, 0.5], [0.5, 1.0]]
        )
        return [first, second]

    def test_natural_frequency_metric_prefers_the_closer_frequency(self) -> None:
        """The default metric links the closer natural frequency."""
        result = identify_modes(self._sweep(), frequency_weight=1.0)
        linked = next(t for t in result.tracks if len(t.operating_points) == 2)
        assert linked.mode_indices == [0, 1]

    def test_eigenvalue_metric_prefers_the_closer_eigenvalue(self) -> None:
        """The eigenvalue metric links the mode with the same decay rate."""
        result = identify_modes(
            self._sweep(), frequency_weight=1.0, continuity_metric="eigenvalue"
        )
        linked = next(t for t in result.tracks if len(t.operating_points) == 2)
        assert linked.mode_indices == [0, 0]

    def test_unknown_metric_raises(self) -> None:
        """An unknown continuity metric is rejected."""
        with pytest.raises(ValueError, match="continuity_metric"):
            identify_modes(self._sweep(), continuity_metric="damped")  # type: ignore[arg-type]


class TestCampbellHeavilyDamped:
    """The Campbell diagram carries, plots and isolates the heavily damped tier."""

    def test_builder_carries_the_tier(self) -> None:
        """build_campbell keeps the heavily damped tracks apart, or omits them."""
        result = identify_modes(_fade_sweep())
        diagram = build_campbell(result, [4.0, 6.0, 8.0, 10.0])
        assert len(diagram.tracks) == 1
        assert len(diagram.heavily_damped_tracks) == 1
        omitted = build_campbell(
            result, [4.0, 6.0, 8.0, 10.0], include_heavily_damped=False
        )
        assert omitted.heavily_damped_tracks == []

    def test_resonances_ignore_heavily_damped_tracks(self) -> None:
        """A near-critical mode crossing an nP line is not a resonance."""
        # 1P at 30-60 rpm is 0.5-1.0 Hz; the near-critical mode sits at 0.7 Hz.
        sweep = [_solution([_mode(0.7, 0.995)], [[1.0]]) for _ in range(2)]
        diagram = campbell_from_solutions(sweep, [30.0, 60.0])
        assert len(diagram.heavily_damped_tracks) == 1
        assert find_resonances(diagram, [1]) == []
        joint = campbell_from_solutions(
            sweep, [30.0, 60.0], heavily_damped_threshold=None
        )
        assert len(find_resonances(joint, [1])) == 1

    def test_plot_draws_the_tier_distinctly(self) -> None:
        """Heavily damped tracks are dotted, named, and can be hidden."""
        diagram = build_campbell(identify_modes(_fade_sweep()), [4.0, 6.0, 8.0, 10.0])
        figure = plot_campbell(diagram, show_excitation=False)
        heavy = [
            trace
            for trace in figure.data
            if isinstance(trace, go.Scatter) and "(heavily damped)" in trace.name
        ]
        assert len(heavy) == 1
        assert heavy[0].line.dash == "dot"
        assert heavy[0].legendgroup == "heavily damped"
        # Hover data: damping and damped frequency.
        np.testing.assert_allclose(
            heavy[0].customdata[:, 1],
            np.asarray(heavy[0].y) * np.sqrt(1.0 - heavy[0].customdata[:, 0] ** 2),
        )
        hidden = plot_campbell(
            diagram, show_excitation=False, show_heavily_damped=False
        )
        assert not any("(heavily damped)" in trace.name for trace in hidden.data)


class TestPipelineHeavilyDamped:
    """ModalPipeline exposes and validates the threshold."""

    def test_invalid_threshold_raises(self) -> None:
        """The pipeline rejects a threshold outside (0, 1]."""
        with pytest.raises(ValueError, match="heavily_damped_threshold"):
            ModalPipeline(heavily_damped_threshold=0.0)

    def test_threshold_is_stored(self) -> None:
        """The default and the opt-out are kept on the pipeline."""
        assert ModalPipeline().heavily_damped_threshold == pytest.approx(0.99)
        assert ModalPipeline(
            heavily_damped_threshold=None
        ).heavily_damped_threshold is (None)


# --- real data ----------------------------------------------------------------------


def _runs_dir() -> Path:
    """Return the reference-turbine linearization folder, or skip when unavailable."""
    candidates = []
    env = os.environ.get("VANE_REFERENCE_RUNS_DIR")
    if env:
        candidates.append(Path(env))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    pytest.skip("reference-turbine runs unavailable; set VANE_REFERENCE_RUNS_DIR")


def _lin_files(key: str, wind: str, stem: str, *, nested: bool = True) -> list[Path]:
    """Return one operating point's azimuth-ordered ``.lin`` files, or skip."""
    case = _runs_dir() / key / "cases" / f"ws{wind}"
    if nested:
        case = case / stem
    pattern = re.compile(rf"{re.escape(stem)}\.(\d+)\.lin")
    files = sorted(
        (p for p in case.glob(f"{stem}.*.lin") if pattern.fullmatch(p.name)),
        key=lambda p: int(pattern.fullmatch(p.name).group(1)),  # type: ignore[union-attr]
    )
    if len(files) < 3:
        pytest.skip(f"incomplete azimuth sweep in {case}")
    return files


_IEA15 = ("iea15_monopile", "IEA-15-240-RWT-Monopile")


@pytest.fixture(scope="module")
def iea15_points() -> list[list[LinFile]]:
    """Return the IEA 15-MW monopile linearizations at 6, 8, 10 and 12 m/s."""
    key, stem = _IEA15
    return [
        [read_lin_file(p) for p in _lin_files(key, wind, stem)]
        for wind in ("6", "8", "10", "12")
    ]


@pytest.fixture(scope="module")
def iea15_result(iea15_points: list[list[LinFile]]) -> object:
    """Run the pipeline on the IEA 15-MW monopile sweep."""
    return ModalPipeline().run(iea15_points, parameter_name="wind_speed")


def _flap_tracks(tracks: list) -> list:  # type: ignore[type-arg]
    return [t for t in tracks if t.label.label.startswith("1st Blade Flap")]


class TestIea15FirstFlap:
    """IEA 15-MW: first-flap whirl lines end where the blade mode passes critical."""

    def test_whirl_lines_end_at_8_mps(self, iea15_result: object) -> None:
        """The three first-flap lines span 6-8 m/s and do not continue above."""
        result = iea15_result
        flap = _flap_tracks(result.tracks)  # type: ignore[attr-defined]
        assert sorted(t.label.multiblade for t in flap) == [
            "collective",
            "progressive",
            "regressive",
        ]
        for track in flap:
            assert track.operating_points == [0, 1], track.label.label

    def test_near_critical_modes_form_their_own_tier(
        self, iea15_result: object
    ) -> None:
        """Above rated the two flap images are reported and tracked apart."""
        identification = iea15_result.identification  # type: ignore[attr-defined]
        assert [len(modes) for modes in identification.heavily_damped_modes] == [
            0,
            0,
            2,
            2,
        ]
        heavy = identification.heavily_damped_tracks
        assert len(heavy) == 2
        assert all(t.operating_points == [2, 3] for t in heavy)
        assert all(t.is_heavily_damped for t in heavy)
        # One image decays slowly (about 0.3 Hz), the other fast (1.4-1.8 Hz).
        means = sorted(float(np.mean(t.natural_frequencies_hz)) for t in heavy)
        assert means[0] < 0.4
        assert means[1] > 1.3
        campbell = iea15_result.campbell  # type: ignore[attr-defined]
        assert len(campbell.heavily_damped_tracks) == 2

    def test_old_joint_tracking_draws_the_spurious_lines(
        self, iea15_points: list[list[LinFile]]
    ) -> None:
        """With the tier disabled, flap lines jump into the near-critical modes."""
        joint = ModalPipeline(heavily_damped_threshold=None).run(
            iea15_points[1:3], parameter_name="wind_speed"
        )
        flap = _flap_tracks(joint.tracks)
        assert any(t.operating_points == [0, 1] for t in flap)


def test_regressive_flap_near_unit_fixed_frame_damping_stays_a_line() -> None:
    """IEA 3.4-MW at 25 m/s: fixed-frame 0.996 regressive flap is not near-critical.

    The regressive image's fixed-frame damping ratio is inflated by the whirl
    shift; in the rotating frame the blade mode is at about 0.95, so it stays in
    the lightly damped tier.
    """
    stem = "IEA-3.4-130-RWT"
    mbc = mbc3_transform(
        [read_lin_file(p) for p in _lin_files("iea3p4_land", "25", stem, nested=False)]
    )
    solution = modes_from_mbc(mbc)
    labels = label_solution(solution)
    regressive = next(
        k
        for k, label in enumerate(labels)
        if label.label.startswith("1st Blade Flap") and label.multiblade == "regressive"
    )
    assert solution.damping_ratios[regressive] > DEFAULT_HEAVILY_DAMPED_THRESHOLD
    rotor = rotor_frame_damping_ratios(
        solution, mbc.rotor_speed_rpm, [label.multiblade for label in labels]
    )
    assert rotor[regressive] < 0.97
    assert not heavily_damped_modes(solution, rotor_speed_rpm=mbc.rotor_speed_rpm).any()
