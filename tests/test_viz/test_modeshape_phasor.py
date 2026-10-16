# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.viz.modeshape_phasor — complex phasor mode-shape views."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from vane.modal.eigensolver import ModalSolution
from vane.viz.modeshape_phasor import (
    plot_blade_components,
    plot_mode_phasor,
    plot_mode_phasor_animation,
)


def _blade_solution() -> ModalSolution:
    """Build a one-mode solution whose three DOFs are MBC blade coordinates."""
    shape = np.array([[0.2 + 0j], [0.8 + 0j], [0.6 + 0.4j]], dtype=np.complex128)
    return ModalSolution(
        eigenvalues=np.array([-0.1 + 2.0j], dtype=np.complex128),
        natural_frequencies_hz=np.array([0.32], dtype=np.float64),
        damping_ratios=np.array([0.05], dtype=np.float64),
        damped_frequencies_hz=np.array([0.31], dtype=np.float64),
        mode_shapes=shape,
        full_eigenvectors=shape,
        n_rigid_body_modes=0,
        dof_descriptions=[
            "blade collective flap",
            "blade cosine flap",
            "blade sine flap",
        ],
        dof_mbc_coordinates=["collective", "cosine", "sine"],
    )


class TestPlotModePhasor:
    """Tests for plot_mode_phasor."""

    def test_returns_polar_figure(self, mode_solution: ModalSolution) -> None:
        """A polar phasor figure with arrow and marker traces is returned."""
        figure = plot_mode_phasor(mode_solution, 0)
        assert isinstance(figure, go.Figure)
        assert any(isinstance(trace, go.Scatterpolar) for trace in figure.data)

    def test_marker_count_respects_top_n(self, mode_solution: ModalSolution) -> None:
        """The marker trace shows at most top_n degrees of freedom."""
        figure = plot_mode_phasor(mode_solution, 0, top_n=1)
        markers = [
            trace
            for trace in figure.data
            if trace.mode == "markers" and isinstance(trace, go.Scatterpolar)
        ]
        assert markers
        assert len(markers[0].r) == 1

    def test_out_of_range_mode_raises(self, mode_solution: ModalSolution) -> None:
        """An out-of-range mode index raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            plot_mode_phasor(mode_solution, 9)

    def test_nonpositive_top_n_raises(self, mode_solution: ModalSolution) -> None:
        """A non-positive top_n raises ValueError."""
        with pytest.raises(ValueError, match="top_n"):
            plot_mode_phasor(mode_solution, 0, top_n=0)


class TestPlotModePhasorAnimation:
    """Tests for plot_mode_phasor_animation."""

    def test_has_frames_over_the_period(self, mode_solution: ModalSolution) -> None:
        """The animation carries one frame per requested step."""
        figure = plot_mode_phasor_animation(mode_solution, 0, n_frames=12)
        assert len(figure.frames) == 12
        assert figure.layout.updatemenus  # a play button is present

    def test_frames_rotate_the_phasors(self, mode_solution: ModalSolution) -> None:
        """Later frames rotate the phasor angles relative to the first."""
        figure = plot_mode_phasor_animation(mode_solution, 0, n_frames=4)
        base = np.asarray(figure.frames[0].data[1].theta, dtype=float)
        rotated = np.asarray(figure.frames[1].data[1].theta, dtype=float)
        np.testing.assert_allclose(rotated, base + 90.0)

    def test_nonpositive_frames_raises(self, mode_solution: ModalSolution) -> None:
        """A non-positive frame count raises ValueError."""
        with pytest.raises(ValueError, match="n_frames"):
            plot_mode_phasor_animation(mode_solution, 0, n_frames=0)


class TestPlotBladeComponents:
    """Tests for plot_blade_components."""

    def test_groups_by_multiblade_coordinate(self) -> None:
        """Each multi-blade coordinate becomes its own coloured bar group."""
        figure = plot_blade_components(_blade_solution(), 0)
        assert isinstance(figure, go.Figure)
        assert {trace.name for trace in figure.data} == {
            "collective",
            "cosine",
            "sine",
        }

    def test_solution_without_blade_coordinates_raises(
        self, mode_solution: ModalSolution
    ) -> None:
        """A solution without MBC coordinates cannot be decomposed."""
        with pytest.raises(ValueError, match="multi-blade"):
            plot_blade_components(mode_solution, 0)

    def test_out_of_range_mode_raises(self) -> None:
        """An out-of-range mode index raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            plot_blade_components(_blade_solution(), 5)
