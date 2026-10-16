# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.viz.modeshape_fowt — physical full-system mode-shape plots."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import pytest

from vane.config.system_config import SystemType, default_geometry, semisub_geometry
from vane.modal.eigensolver import ModalSolution
from vane.viz.modeshape_fowt import (
    _cantilever_shape,
    _cantilever_tip_slope,
    deform_mode,
    infer_geometry,
    plot_mode_fowt,
    plot_mode_fowt_animation,
)

# A floating-turbine state list: platform DOFs, a tower mode, and an MBC blade triplet.
_FLOATING_DESCS = [
    "ED platform horizontal surge translation DOF (internal DOF index = DOF_Sg), m",
    "ED platform pitch tilt rotation DOF (internal DOF index = DOF_P), rad",
    "ED platform heave translation DOF (internal DOF index = DOF_Hv), m",
    "ED 1st tower fore-aft bending mode DOF (internal DOF index = DOF_TFA1), m",
    "ED 1st tower side-side bending mode DOF (internal DOF index = DOF_TSS1), m",
    "ED 1st flapwise bending-mode DOF of blade 1 (internal DOF index = DOF_BF(1,1)), m",
    "ED 1st flapwise bending-mode DOF of blade 2 (internal DOF index = DOF_BF(2,1)), m",
    "ED 1st flapwise bending-mode DOF of blade 3 (internal DOF index = DOF_BF(3,1)), m",
]
_FLOATING_MBC = ["", "", "", "", "", "collective", "cosine", "sine"]

# Offshore support types draw a still-water plane; onshore / legacy-fixed do not.
_OFFSHORE_TYPES = frozenset(
    {
        SystemType.SEMI,
        SystemType.SPAR,
        SystemType.TLP,
        SystemType.MONOPILE,
        SystemType.JACKET,
    }
)


def _solution(
    shapes: npt.NDArray[np.complex128],
    *,
    descriptions: list[str] | None = None,
    mbc: list[str] | None = None,
) -> ModalSolution:
    """Build a ModalSolution from explicit mode-shape columns."""
    n_modes = shapes.shape[1]
    freq = np.linspace(0.05, 0.5, n_modes)
    return ModalSolution(
        eigenvalues=(-0.01 + 1j * freq).astype(np.complex128),
        natural_frequencies_hz=freq.astype(np.float64),
        damping_ratios=np.full(n_modes, 0.03),
        damped_frequencies_hz=freq.astype(np.float64),
        mode_shapes=shapes,
        full_eigenvectors=shapes,
        n_rigid_body_modes=0,
        dof_descriptions=descriptions if descriptions is not None else _FLOATING_DESCS,
        dof_mbc_coordinates=mbc if mbc is not None else _FLOATING_MBC,
    )


def _platform_pitch_mode() -> ModalSolution:
    """Return a real, platform-pitch-dominated mode (surge and a little tower)."""
    shapes = np.zeros((8, 1), dtype=np.complex128)
    shapes[0, 0] = 0.5  # surge
    shapes[1, 0] = 1.0  # pitch
    shapes[3, 0] = 0.3  # tower fore-aft
    return _solution(shapes)


class TestCantileverShapes:
    """Tests for the clamped-free bending shape functions."""

    def test_clamped_root_free_tip(self) -> None:
        """Both modes vanish at the clamped root and equal 1 at the free tip."""
        for mode in (1, 2):
            assert _cantilever_shape(mode, np.array([0.0]))[0] == pytest.approx(0.0)
            assert _cantilever_shape(mode, np.array([1.0]))[0] == pytest.approx(1.0)

    def test_first_mode_is_monotonic(self) -> None:
        """The first bending mode grows monotonically from root to tip."""
        values = _cantilever_shape(1, np.linspace(0.0, 1.0, 50))
        assert np.all(np.diff(values) > 0.0)

    def test_second_mode_has_internal_node(self) -> None:
        """The second bending mode changes sign once along the interior span."""
        # Exclude the clamped root (exactly 0) so the count reflects interior nodes.
        values = _cantilever_shape(2, np.linspace(0.0, 1.0, 200))[1:]
        sign_changes = int(np.count_nonzero(np.diff(np.sign(values)) != 0))
        assert sign_changes == 1

    def test_tip_slopes_are_positive(self) -> None:
        """The unit-tip-normalized shapes rise toward the tip (positive slope)."""
        assert _cantilever_tip_slope(1) > 0.0
        assert _cantilever_tip_slope(2) > 0.0


class TestDeformMode:
    """Tests for the modal-coordinate to displacement-field mapping."""

    def test_displacements_align_with_members(self) -> None:
        """One complex displacement field is returned per geometry member."""
        solution = _platform_pitch_mode()
        geometry = semisub_geometry()
        deformation = deform_mode(solution, 0, geometry)
        assert len(deformation.displacements) == len(geometry.members)
        for member, field in zip(
            geometry.members, deformation.displacements, strict=True
        ):
            assert field.shape == member.nodes.shape

    def test_scale_normalizes_peak_deflection(self) -> None:
        """The peak nodal deflection matches the requested fraction of the size."""
        solution = _platform_pitch_mode()
        geometry = semisub_geometry()
        deformation = deform_mode(solution, 0, geometry, deflection_ratio=0.1)
        peak = max(
            float(np.sqrt((np.abs(field) ** 2).sum(axis=1)).max())
            for field in deformation.displacements
            if field.size
        )
        np.testing.assert_allclose(
            peak, 0.1 * geometry.characteristic_length, rtol=1e-6
        )

    def test_platform_pitch_moves_tower_top_downwind(self) -> None:
        """A platform-pitch mode tilts the whole turbine, moving the tower top in x."""
        solution = _platform_pitch_mode()
        geometry = semisub_geometry()
        deformation = deform_mode(solution, 0, geometry)
        phase = deformation.peak_phase_rad()
        tower_index = geometry.members.index(geometry.members_of_kind("tower")[0])
        tower_top = deformation.deformed_nodes(tower_index, phase)[-1]
        # Significant downwind (x) motion; the tower top is undeflected in the ghost.
        assert abs(float(tower_top[0])) > 1.0
        np.testing.assert_allclose(tower_top[2], geometry.tower_top[2], atol=1e-6)

    def test_mooring_anchor_stays_fixed(self) -> None:
        """The seabed anchor end of a mooring line does not move."""
        solution = _platform_pitch_mode()
        geometry = semisub_geometry()
        deformation = deform_mode(solution, 0, geometry)
        for index, member in enumerate(geometry.members):
            if member.kind != "mooring":
                continue
            # Last node is the anchor: its complex displacement is zero.
            np.testing.assert_allclose(
                np.abs(deformation.displacements[index][-1]), 0.0, atol=1e-12
            )
            # First node is the fairlead: it moves with the platform.
            assert float(np.abs(deformation.displacements[index][0]).sum()) > 0.0

    def test_inverse_mbc_differs_across_blades(self) -> None:
        """A cosine/sine blade mode deflects the three blades differently."""
        shapes = np.zeros((8, 1), dtype=np.complex128)
        shapes[5, 0] = 1.0  # collective flap
        shapes[6, 0] = 1.0  # cosine flap
        deformation = deform_mode(_solution(shapes), 0, semisub_geometry())
        geometry = deformation.geometry
        tips = []
        for index, member in enumerate(geometry.members):
            if member.kind == "blade":
                # Flapwise tip amplitude (magnitude of the complex tip displacement).
                tips.append(float(np.abs(deformation.displacements[index][-1]).sum()))
        assert len(tips) == 3
        # The three blades are not all identical (cosine component breaks symmetry).
        assert max(tips) - min(tips) > 1e-6

    def test_sine_cyclic_flap_moves_negative_y_blade_downwind(self) -> None:
        """A positive sine-cyclic flap pushes the blade at +90 deg (-y) downwind.

        With the OpenFAST azimuth (right-hand about downwind +x, blade 1 up) blade
        2 sits at 120 deg, on the -y side, where ``sin(psi) > 0``.
        """
        shapes = np.zeros((8, 1), dtype=np.complex128)
        shapes[7, 0] = 1.0  # sine-cyclic flap
        deformation = deform_mode(_solution(shapes), 0, semisub_geometry())
        blades = {
            member.blade_index: (member, deformation.displacements[index][-1].real)
            for index, member in enumerate(deformation.geometry.members)
            if member.kind == "blade"
        }
        blade2, tip2 = blades[2]
        blade3, tip3 = blades[3]
        assert blade2.nodes[-1][1] < 0.0
        assert tip2[0] > 0.0  # -y blade flaps downwind
        assert blade3.nodes[-1][1] > 0.0
        assert tip3[0] < 0.0  # +y blade flaps upwind
        np.testing.assert_allclose(blades[1][1][0], 0.0, atol=1e-9)

    def test_tower_fore_aft_and_collective_flap_share_direction(self) -> None:
        """In-phase positive tower fore-aft and collective flap both point +x."""
        shapes = np.zeros((8, 2), dtype=np.complex128)
        shapes[3, 0] = 1.0  # tower fore-aft only
        shapes[5, 1] = 1.0  # collective flap only
        solution = _solution(shapes)
        geometry = semisub_geometry()
        tower_index = geometry.members.index(geometry.members_of_kind("tower")[0])
        tower = deform_mode(solution, 0, geometry)
        assert tower.displacements[tower_index][-1].real[0] > 0.0
        flap = deform_mode(solution, 1, geometry)
        for index, member in enumerate(geometry.members):
            if member.kind == "blade":
                assert flap.displacements[index][-1].real[0] > 0.0

    def test_zero_mode_gives_zero_field(self) -> None:
        """A mode with no recognized participation produces no deflection."""
        shapes = np.zeros((8, 1), dtype=np.complex128)
        deformation = deform_mode(_solution(shapes), 0, semisub_geometry())
        assert deformation.scale == 0.0
        for field in deformation.displacements:
            np.testing.assert_allclose(np.abs(field), 0.0)

    def test_peak_phase_of_real_mode_is_displacement_extreme(self) -> None:
        """A real mode peaks at phase 0 (its displacement extreme)."""
        deformation = deform_mode(_platform_pitch_mode(), 0, semisub_geometry())
        assert deformation.peak_phase_rad() == pytest.approx(0.0, abs=1e-9)

    def test_velocity_and_unknown_rows_ignored(self) -> None:
        """Velocity-derivative and unrecognized states do not deform the structure."""
        descs = [
            "ED First time derivative of platform pitch (internal DOF index = DOF_P)",
            "SrvD generator torque control state, -",
        ]
        shapes = np.array([[1.0], [1.0]], dtype=np.complex128)
        deformation = deform_mode(
            _solution(shapes, descriptions=descs, mbc=["", ""]),
            0,
            semisub_geometry(),
        )
        assert deformation.scale == 0.0


class TestDeformModeValidation:
    """Tests for argument validation in deform_mode."""

    def test_empty_descriptions_raises(self) -> None:
        """A solution without DOF descriptions cannot be deformed."""
        shapes = np.ones((1, 1), dtype=np.complex128)
        solution = _solution(shapes, descriptions=[], mbc=[])
        with pytest.raises(ValueError, match="no dof_descriptions"):
            deform_mode(solution, 0)

    def test_out_of_range_mode_raises(self) -> None:
        """An out-of-range mode index is rejected."""
        with pytest.raises(ValueError, match="out of range"):
            deform_mode(_platform_pitch_mode(), 5)

    def test_non_positive_deflection_ratio_raises(self) -> None:
        """A non-positive deflection ratio is rejected."""
        with pytest.raises(ValueError, match="deflection_ratio must be positive"):
            deform_mode(_platform_pitch_mode(), 0, deflection_ratio=0.0)


class TestInferGeometry:
    """Tests for inferring a schematic geometry from a solution."""

    def test_platform_dofs_give_floating(self) -> None:
        """A solution with platform DOFs infers a floating (semi) geometry."""
        geometry = infer_geometry(_platform_pitch_mode())
        assert geometry.system_type is SystemType.SEMI

    def test_no_platform_dofs_give_fixed(self) -> None:
        """A tower-only solution infers a bottom-fixed geometry."""
        descs = ["ED 1st tower fore-aft (internal DOF index = DOF_TFA1), m"]
        shapes = np.array([[1.0]], dtype=np.complex128)
        geometry = infer_geometry(_solution(shapes, descriptions=descs, mbc=[""]))
        assert geometry.system_type is SystemType.FIXED


class TestPlotModeFowt:
    """Tests for the static and animated figures."""

    def test_static_returns_figure_with_traces(self) -> None:
        """The static plot returns a populated 3-D figure."""
        figure = plot_mode_fowt(_platform_pitch_mode(), 0)
        assert isinstance(figure, go.Figure)
        # Ghost + deformed traces for every member, plus the water plane.
        assert len(figure.data) > len(semisub_geometry().members)

    def test_static_honours_explicit_phase(self) -> None:
        """Passing a phase freezes the deformation there without error."""
        figure = plot_mode_fowt(_platform_pitch_mode(), 0, phase_deg=45.0)
        assert isinstance(figure, go.Figure)

    def test_title_reports_frequency(self) -> None:
        """The title carries the mode's natural frequency."""
        figure = plot_mode_fowt(_platform_pitch_mode(), 0)
        assert "Hz" in figure.layout.title.text

    def test_animation_has_frames(self) -> None:
        """The animation builds the requested number of frames and a play button."""
        figure = plot_mode_fowt_animation(_platform_pitch_mode(), 0, n_frames=10)
        assert len(figure.frames) == 10
        assert figure.layout.updatemenus

    def test_animation_has_a_scrub_slider(self) -> None:
        """The animation carries a frame slider with one step per frame for replay."""
        figure = plot_mode_fowt_animation(_platform_pitch_mode(), 0, n_frames=8)
        assert figure.layout.sliders
        assert len(figure.layout.sliders[0].steps) == 8

    def test_swing_motion_oscillates_symmetrically(self) -> None:
        """The swing motion reaches the same peak deflection with opposite sign.

        Frame 0 (cos=+1) and the half-period frame (cos=-1) are mirror images about the
        undeformed geometry, unlike the complex-orbit cycle.
        """
        figure = plot_mode_fowt_animation(
            _platform_pitch_mode(), 0, motion="swing", n_frames=8, scale=3.0
        )
        first = np.asarray(figure.frames[0].data[0].z)
        half = np.asarray(figure.frames[4].data[0].z)  # cos(pi) = -1
        undeformed = 0.5 * (first + half)
        assert np.allclose(first - undeformed, -(half - undeformed), atol=1e-9)

    def test_animation_rejects_zero_frames(self) -> None:
        """A non-positive frame count is rejected."""
        with pytest.raises(ValueError, match="n_frames must be at least 1"):
            plot_mode_fowt_animation(_platform_pitch_mode(), 0, n_frames=0)

    def test_initial_frame_matches_the_slider_zero_step(self) -> None:
        """The loaded figure shows frame 0, so it agrees with the slider's 0 step.

        For a complex mode the peak phase is non-zero; the static base data must still
        equal frame 0 (fraction 0), or the figure would display one phase while the
        slider reads 0 and the first scrub would jump.
        """
        figure = plot_mode_fowt_animation(_platform_pitch_mode(), 0, n_frames=12)
        base_deformed = np.asarray(figure.data[-1].z)
        frame0_deformed = np.asarray(figure.frames[0].data[-1].z)
        assert figure.layout.sliders[0].active == 0
        np.testing.assert_allclose(base_deformed, frame0_deformed)

    def test_animation_rejects_unknown_motion(self) -> None:
        """A typo'd motion is rejected rather than silently falling back to cycle."""
        # An Any-typed value defeats the Literal check without a brittle inline ignore
        # (which reads as "unused" when mypy checks this test file in isolation).
        bad_motion: Any = "swng"
        with pytest.raises(ValueError, match="motion must be 'cycle' or 'swing'"):
            plot_mode_fowt_animation(_platform_pitch_mode(), 0, motion=bad_motion)

    def test_fixed_system_has_no_water_plane(self) -> None:
        """A bottom-fixed inferred geometry omits the still-water plane."""
        descs = ["ED 1st tower fore-aft (internal DOF index = DOF_TFA1), m"]
        shapes = np.array([[1.0]], dtype=np.complex128)
        solution = _solution(shapes, descriptions=descs, mbc=[""])
        figure = plot_mode_fowt(solution, 0)
        assert all(trace.type != "mesh3d" for trace in figure.data)

    @pytest.mark.parametrize("system_type", list(SystemType))
    def test_every_support_type_renders_static_and_animated(
        self, system_type: SystemType
    ) -> None:
        """Each balance-of-plant preset renders end to end with finite coordinates.

        A geometry can construct correctly yet still feed the deformation engine a
        member the renderer mishandles. Drive every support type through both the
        static and the animated path and assert real, finite figures, so a broken
        preset cannot slip past the construction-only unit tests.
        """
        geometry = default_geometry(system_type)
        mode = _platform_pitch_mode()

        static = plot_mode_fowt(mode, 0, geometry=geometry)
        animated = plot_mode_fowt_animation(mode, 0, geometry=geometry, n_frames=4)

        assert len(static.data) > 0
        assert len(animated.frames) == 4
        traces = list(static.data) + [
            datum for frame in animated.frames for datum in frame.data
        ]
        for trace in traces:
            for axis in ("x", "y", "z"):
                values = getattr(trace, axis, None)
                if values is not None and len(values):
                    assert np.all(np.isfinite(np.asarray(values, dtype=float)))

        has_water = any(trace.type == "mesh3d" for trace in static.data)
        assert has_water is (system_type in _OFFSHORE_TYPES)
