# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.viz.diagnostics — MAC, tracking-confidence, azimuth-coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go

from vane.io.lin_reader import read_lin_file
from vane.study import run_study
from vane.viz.diagnostics import (
    plot_azimuth_coverage,
    plot_branch_review,
    plot_mac_matrix,
    plot_tracking_confidence,
)

if TYPE_CHECKING:
    from tests.conftest import LinFileFactory
    from vane.campbell.builder import CampbellDiagram
    from vane.modal.branch import BranchGroup
    from vane.modal.eigensolver import ModalSolution

_STATES: list[tuple[str, bool, int]] = [
    ("ED 1st tower fore-aft (internal DOF index = DOF_TFA1), m", False, 2),
    (
        "ED First time derivative of 1st tower fore-aft "
        "(internal DOF index = DOF_TFA1), m/s",
        False,
        1,
    ),
]
_A = np.array([[0.0, 1.0], [-4.0, -0.2]], dtype=np.float64)


class TestPlotMacMatrix:
    """Tests for plot_mac_matrix."""

    def test_self_mac_diagonal_is_one(self, mode_solution: ModalSolution) -> None:
        """A solution correlated with itself has a unit-MAC diagonal."""
        figure = plot_mac_matrix(mode_solution, mode_solution)
        assert isinstance(figure, go.Figure)
        diagonal = np.diag(np.asarray(figure.data[0].z))
        np.testing.assert_allclose(diagonal, 1.0, atol=1e-8)

    def test_cells_use_distinct_positions(self, mode_solution: ModalSolution) -> None:
        """Cells sit at distinct positions so same-frequency modes do not merge."""
        figure = plot_mac_matrix(mode_solution, mode_solution)
        x = figure.data[0].x
        y = figure.data[0].y
        assert len(set(x)) == len(x)
        assert len(set(y)) == len(y)


class TestPlotTrackingConfidence:
    """Tests for plot_tracking_confidence."""

    def test_one_bar_per_track(self, campbell_diagram: CampbellDiagram) -> None:
        """There is one confidence bar per tracked mode."""
        figure = plot_tracking_confidence(campbell_diagram)
        assert isinstance(figure, go.Figure)
        assert len(figure.data[0].x) == len(campbell_diagram.tracks)

    def test_ambiguous_track_gets_distinct_colour(
        self, campbell_diagram: CampbellDiagram
    ) -> None:
        """An ambiguous track is coloured differently from a confident one."""
        campbell_diagram.tracks[0].is_ambiguous = True
        campbell_diagram.tracks[1].is_ambiguous = False
        figure = plot_tracking_confidence(campbell_diagram)
        colors = figure.data[0].marker.color
        assert colors[0] != colors[1]

    def test_same_label_tracks_stay_distinct(
        self, campbell_diagram: CampbellDiagram
    ) -> None:
        """Two tracks sharing a physical label remain separate bars."""
        campbell_diagram.tracks[1].label = campbell_diagram.tracks[0].label
        figure = plot_tracking_confidence(campbell_diagram)
        x = figure.data[0].x
        assert len(x) == len(campbell_diagram.tracks)
        assert len(set(x)) == len(campbell_diagram.tracks)


class TestPlotBranchReview:
    """Tests for plot_branch_review."""

    def test_contested_groups_render_competing_tracks(
        self, branch_review_case: tuple[CampbellDiagram, list[BranchGroup]]
    ) -> None:
        """A detected crossing renders a panel with both competing tracks."""
        diagram, groups = branch_review_case
        assert groups  # the fixture's crossing must be detected
        figure = plot_branch_review(diagram, groups)
        assert isinstance(figure, go.Figure)
        assert len(figure.data) >= 2

    def test_empty_groups_render_placeholder(
        self, campbell_diagram: CampbellDiagram
    ) -> None:
        """With no contested groups the figure is a labelled placeholder."""
        figure = plot_branch_review(campbell_diagram, [])
        assert isinstance(figure, go.Figure)
        assert figure.layout.annotations[0].text == "No contested branches detected"

    def test_resolved_group_overlays_smooth_alternative(
        self, resolved_branch_case: tuple[CampbellDiagram, list[BranchGroup]]
    ) -> None:
        """A resolved group adds the dashed smooth-alternative traces."""
        diagram, groups = resolved_branch_case
        assert any(group.resolved for group in groups)
        figure = plot_branch_review(diagram, groups)
        assert any(trace.name == "smooth alternative" for trace in figure.data)


class TestPlotAzimuthCoverage:
    """Tests for plot_azimuth_coverage."""

    def test_one_trace_per_operating_point(self, make_lin_file: LinFileFactory) -> None:
        """Each operating point contributes one azimuth-coverage trace."""
        operating_points = [
            [
                read_lin_file(
                    make_lin_file(states=_STATES, a=_A, rotor_speed=speed, azimuth=az)
                )
                for az in (0.0, 1.0)
            ]
            for speed in (1.0, 1.5)
        ]
        result = run_study(operating_points)
        figure = plot_azimuth_coverage(result.provenance)
        assert isinstance(figure, go.Figure)
        assert len(figure.data) == len(result.provenance.operating_points)
