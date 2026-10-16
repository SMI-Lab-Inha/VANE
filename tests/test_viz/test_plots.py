# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for the vane.viz plotting functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
import pytest

from vane.viz.campbell_plot import plot_campbell
from vane.viz.damping_plot import plot_damping
from vane.viz.modeshape_3d import plot_mode_3d
from vane.viz.modeshape_static import plot_mode_participation, plot_mode_shape

if TYPE_CHECKING:
    from vane.campbell.builder import CampbellDiagram
    from vane.modal.eigensolver import ModalSolution


class TestPlotCampbell:
    """Tests for plot_campbell."""

    def test_returns_figure_with_track_lines(
        self, campbell_diagram: CampbellDiagram
    ) -> None:
        """The figure includes one trace per track plus excitation/resonance."""
        figure = plot_campbell(campbell_diagram)
        assert isinstance(figure, go.Figure)
        names = [trace.name for trace in figure.data]
        # Two tracks, plus 1P/3P/6P/9P lines, plus severity-grouped resonance markers.
        assert sum(name.endswith("P") for name in names) == 4
        assert any(name.startswith("Resonance") for name in names)

    def test_excitation_skipped_for_non_rpm(
        self, campbell_diagram: CampbellDiagram
    ) -> None:
        """A non-rotor-speed diagram omits nP lines and resonance markers."""
        campbell_diagram.parameter_name = "wind_speed"
        figure = plot_campbell(campbell_diagram)
        names = [trace.name for trace in figure.data]
        assert not any(name.startswith("Resonance") for name in names)
        assert not any(name.endswith("P") for name in names)

    def test_toggles_disable_overlays(self, campbell_diagram: CampbellDiagram) -> None:
        """Disabling excitation and resonance leaves only the track lines."""
        figure = plot_campbell(
            campbell_diagram, show_excitation=False, show_resonances=False
        )
        assert len(figure.data) == len(campbell_diagram.tracks)

    def test_confidence_scales_track_opacity(
        self, campbell_diagram: CampbellDiagram
    ) -> None:
        """A low-confidence track renders more transparent than a confident one."""
        campbell_diagram.tracks[0].confidence = 1.0
        campbell_diagram.tracks[1].confidence = 0.0
        figure = plot_campbell(
            campbell_diagram, show_excitation=False, show_resonances=False
        )
        assert figure.data[0].opacity == pytest.approx(1.0)
        assert figure.data[1].opacity < figure.data[0].opacity

    def test_ambiguous_track_is_flagged(
        self, campbell_diagram: CampbellDiagram
    ) -> None:
        """An ambiguous track gets an annotated legend name and diamond markers."""
        campbell_diagram.tracks[0].is_ambiguous = True
        figure = plot_campbell(
            campbell_diagram, show_excitation=False, show_resonances=False
        )
        assert "(ambiguous)" in figure.data[0].name
        assert figure.data[0].marker.symbol == "diamond-open"

    def test_ambiguous_points_of_a_clear_track_are_marked(
        self, campbell_diagram: CampbellDiagram
    ) -> None:
        """A hybrid point of a clear line gets its own marker and hover text."""
        from dataclasses import replace

        track = campbell_diagram.tracks[0]
        clear = replace(track.label, is_ambiguous=False)
        hybrid = replace(track.label, is_ambiguous=True, alternative="2nd Tower")
        track.point_labels = [clear] * len(track.operating_points)
        track.point_labels[-1] = hybrid
        figure = plot_campbell(
            campbell_diagram, show_excitation=False, show_resonances=False
        )
        symbols = list(figure.data[0].marker.symbol)
        assert symbols[-1] == "diamond-open"
        assert set(symbols[:-1]) == {"circle"}
        assert "or 2nd Tower (ambiguous)" in figure.data[0].customdata[-1][2]
        assert figure.data[1].marker.symbol == "circle"

    def test_resonance_markers_grouped_by_severity(
        self, campbell_diagram: CampbellDiagram
    ) -> None:
        """Resonance traces are named and coloured by severity."""
        figure = plot_campbell(campbell_diagram, show_excitation=False)
        resonance = [t for t in figure.data if t.name.startswith("Resonance")]
        assert resonance
        for trace in resonance:
            severity = trace.name.split("(")[1].rstrip(")")
            assert severity in {"high", "medium", "low"}

    def test_draws_into_supplied_figure(
        self, campbell_diagram: CampbellDiagram
    ) -> None:
        """A caller-supplied figure is reused rather than replaced."""
        existing = go.Figure()
        existing.add_trace(go.Scatter(x=[0.0], y=[0.0], name="prior"))
        figure = plot_campbell(
            campbell_diagram,
            figure=existing,
            show_excitation=False,
            show_resonances=False,
        )
        assert figure is existing
        assert figure.data[0].name == "prior"

    def test_operating_range_filters_resonances(
        self, campbell_diagram: CampbellDiagram
    ) -> None:
        """An empty operating window yields no resonance markers."""
        figure = plot_campbell(
            campbell_diagram, show_excitation=False, operating_range=(1e9, 2e9)
        )
        assert not any(t.name.startswith("Resonance") for t in figure.data)


class TestPlotDamping:
    """Tests for plot_damping."""

    def test_one_trace_per_track(self, campbell_diagram: CampbellDiagram) -> None:
        """Damping plot has one line per track."""
        figure = plot_damping(campbell_diagram)
        assert isinstance(figure, go.Figure)
        assert len(figure.data) == len(campbell_diagram.tracks)

    def test_fraction_axis(self, campbell_diagram: CampbellDiagram) -> None:
        """Plotting as a fraction labels the axis accordingly."""
        figure = plot_damping(campbell_diagram, as_percent=False)
        assert "(-)" in figure.layout.yaxis.title.text

    def test_stability_bands_are_drawn(self, campbell_diagram: CampbellDiagram) -> None:
        """Two stability bands (unstable + caution) are shaded by default."""
        figure = plot_damping(campbell_diagram)
        assert len(figure.layout.shapes) == 2
        no_bands = plot_damping(campbell_diagram, show_stability_bands=False)
        assert len(no_bands.layout.shapes) == 0

    def test_instability_is_flagged(self, campbell_diagram: CampbellDiagram) -> None:
        """A negative-damping point adds a flag trace and names the mode."""
        track = campbell_diagram.tracks[0]
        track.damping_ratios = [-0.01, *track.damping_ratios[1:]]
        figure = plot_damping(campbell_diagram)
        # One extra (flag) trace beyond the per-track lines.
        assert len(figure.data) == len(campbell_diagram.tracks) + 1
        assert "UNSTABLE" in figure.layout.title.text
        assert track.label.label in figure.layout.title.text

    def test_nan_damping_does_not_shrink_the_unstable_band(
        self, campbell_diagram: CampbellDiagram
    ) -> None:
        """A NaN point on a track must not hide that track's negative damping."""
        track = campbell_diagram.tracks[0]
        assert len(track.damping_ratios) >= 2
        track.damping_ratios = [-0.05, float("nan"), *track.damping_ratios[2:]]
        figure = plot_damping(campbell_diagram, as_percent=True)
        unstable_band = figure.layout.shapes[0]
        # The band reaches below the lowest finite damping (-5 %).
        assert unstable_band.y0 <= -5.0

    def test_all_nan_track_keeps_the_default_band(
        self, campbell_diagram: CampbellDiagram
    ) -> None:
        """A track with no finite damping neither fails nor deepens the band."""
        track = campbell_diagram.tracks[0]
        track.damping_ratios = [float("nan")] * len(track.damping_ratios)
        figure = plot_damping(campbell_diagram, as_percent=True, caution_damping=0.01)
        assert np.isfinite(figure.layout.shapes[0].y0)

    def test_confidence_scales_opacity(self, campbell_diagram: CampbellDiagram) -> None:
        """Low confidence fades the line; full confidence is opaque."""
        campbell_diagram.tracks[0].confidence = 0.0
        campbell_diagram.tracks[1].confidence = 1.0
        figure = plot_damping(campbell_diagram)
        assert figure.data[0].opacity == pytest.approx(0.35)
        assert figure.data[1].opacity == pytest.approx(1.0)

    def test_ambiguous_track_is_marked_and_named(
        self, campbell_diagram: CampbellDiagram
    ) -> None:
        """An ambiguous track gets open diamonds (as in the Campbell plot) and a flag.

        The line keeps its family dash, so the dash still identifies the DOF.
        """
        campbell_diagram.tracks[0].is_ambiguous = True
        figure = plot_damping(campbell_diagram)
        assert figure.data[0].marker.symbol == "diamond-open"
        assert figure.data[0].line.dash == "solid"  # 1st tower fore-aft
        assert "ambiguous" in figure.data[0].name

    def test_negative_caution_rejected(self, campbell_diagram: CampbellDiagram) -> None:
        """A negative caution threshold is rejected."""
        with pytest.raises(ValueError, match="caution_damping must be non-negative"):
            plot_damping(campbell_diagram, caution_damping=-0.1)


class TestPlotModeShapeStatic:
    """Tests for plot_mode_shape."""

    def test_returns_matplotlib_figure(self, mode_solution: ModalSolution) -> None:
        """A matplotlib figure with one bar axis is returned."""
        from matplotlib.figure import Figure

        figure = plot_mode_shape(mode_solution, 0)
        assert isinstance(figure, Figure)

    def test_out_of_range_mode_raises(self, mode_solution: ModalSolution) -> None:
        """An out-of-range mode index raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            plot_mode_shape(mode_solution, 5)

    def test_nonpositive_top_n_raises(self, mode_solution: ModalSolution) -> None:
        """A non-positive top_n raises ValueError."""
        with pytest.raises(ValueError, match="top_n"):
            plot_mode_shape(mode_solution, 0, top_n=0)

    def test_without_descriptions_raises(self) -> None:
        """A solution without DOF descriptions cannot be plotted."""
        from vane.modal.eigensolver import compute_modes

        solution = compute_modes(np.array([[0.0, 1.0], [-4.0, -0.2]]), ndof2=1, ndof1=0)
        with pytest.raises(ValueError, match="dof_descriptions"):
            plot_mode_shape(solution, 0)

    def test_participation_is_the_primary_name(
        self, mode_solution: ModalSolution
    ) -> None:
        """plot_mode_participation is the accurate name; plot_mode_shape aliases it."""
        from matplotlib.figure import Figure

        assert isinstance(plot_mode_participation(mode_solution, 0), Figure)

    def test_draws_into_supplied_axes(self, mode_solution: ModalSolution) -> None:
        """Supplying axes reuses their parent figure instead of creating a new one."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        returned = plot_mode_participation(mode_solution, 0, ax=ax)
        assert returned is fig


class TestPlotMode3d:
    """Tests for plot_mode_3d."""

    def test_returns_figure_with_phasors(self, mode_solution: ModalSolution) -> None:
        """A 3-D figure with one phasor per shown DOF is returned."""
        figure = plot_mode_3d(mode_solution, 0)
        assert isinstance(figure, go.Figure)
        assert len(figure.data) == 2  # two DOFs in the fixture

    def test_out_of_range_mode_raises(self, mode_solution: ModalSolution) -> None:
        """An out-of-range mode index raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            plot_mode_3d(mode_solution, 9)
