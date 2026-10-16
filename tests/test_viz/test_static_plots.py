# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for the Campbell and damping plots: encoding, labels, legends, panels."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex
from matplotlib.text import Annotation

from tests.test_viz.test_modeshape_fowt import _solution as _fowt_solution
from vane.campbell.builder import CampbellDiagram
from vane.config.dof_map import DofCategory
from vane.config.system_config import SystemType, default_geometry
from vane.modal.identifier import ModeTrack
from vane.modal.labeler import ModeLabel
from vane.viz import (
    FAMILY_COLORS,
    ModeFamily,
    plot_blade_components_static,
    plot_branch_review_static,
    plot_campbell,
    plot_campbell_panels,
    plot_campbell_static,
    plot_damping,
    plot_damping_static,
    plot_mac_matrix_static,
    plot_mode_fowt_static,
    plot_mode_phasor_static,
)
from vane.viz._style import track_styles
from vane.viz.campbell_plot import HEAVILY_DAMPED_COLOR
from vane.viz.modeshape_fowt import _within

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from vane.modal.eigensolver import ModalSolution


def _track(
    category: DofCategory,
    text: str,
    frequency: float,
    *,
    multiblade: str | None = None,
    slope: float = 0.0,
    damping: float = 0.02,
) -> ModeTrack:
    """Return a three-point track with a fixed label."""
    label = ModeLabel(category, text, 0.9, [], multiblade)
    rpm = np.array([0.0, 6.0, 12.0])
    return ModeTrack(
        operating_points=[0, 1, 2],
        mode_indices=[0, 0, 0],
        natural_frequencies_hz=list(frequency + slope * rpm / 60.0),
        damping_ratios=[damping, damping, damping],
        label=label,
    )


@pytest.fixture
def family_diagram() -> CampbellDiagram:
    """Return a rotor-speed diagram with tracks of several families, out of order."""
    tracks = [
        _track(
            DofCategory.BLADE_FLAP_1,
            "1st Blade Flap (Regressive)",
            0.7,
            multiblade="regressive",
            slope=-1.0,
        ),
        _track(DofCategory.TOWER_FORE_AFT_1, "1st Tower Fore-Aft", 0.32),
        _track(DofCategory.PLATFORM_PITCH, "Platform Pitch", 0.03),
        _track(DofCategory.UNKNOWN, "Unidentified", 1.5),
        _track(DofCategory.DRIVETRAIN_TORSION, "Drivetrain Torsion", 0.6),
        _track(
            DofCategory.BLADE_EDGE_1,
            "1st Blade Edge (Progressive)",
            1.1,
            multiblade="progressive",
            slope=1.0,
        ),
    ]
    heavy = _track(DofCategory.BLADE_FLAP_1, "1st Blade Flap", 0.8, damping=0.9)
    return CampbellDiagram(
        parameter_values=np.array([0.0, 6.0, 12.0]),
        parameter_name="rotor_speed_rpm",
        tracks=tracks,
        n_operating_points=3,
        heavily_damped_tracks=[heavy],
    )


def _key_texts(figure: Figure) -> list[str]:
    """Return the entries of the figure's key (figure legend or axes legend)."""
    legends = figure.legends or [figure.axes[0].get_legend()]
    return [text.get_text() for text in legends[0].get_texts()]


def _annotations(figure: Figure) -> list[str]:
    """Return the texts of every annotation of the figure's axes."""
    return [
        child.get_text()
        for axes in figure.axes
        for child in axes.get_children()
        if isinstance(child, Annotation) and child.get_text()
    ]


class TestPlotCampbellStatic:
    """Tests for plot_campbell_static."""

    def test_key_lists_families_then_overlays(
        self, family_diagram: CampbellDiagram
    ) -> None:
        """The compact key lists the families, Blade Passing and the overlays."""
        figure = plot_campbell_static(family_diagram)
        texts = _key_texts(figure)
        families = ["Blades", "Tower", "Platform", "Drivetrain", "Other"]
        assert texts[:5] == families
        assert texts[5] == "Blade Passing"
        assert {"Regressive", "Progressive", "Heavily damped"} <= set(texts)
        assert not any("1st Tower" in text for text in texts)
        plt.close(figure)

    def test_direct_labels_and_inline_tags(
        self, family_diagram: CampbellDiagram
    ) -> None:
        """Every mode is named at its line end; the nP lines are tagged inline."""
        figure = plot_campbell_static(family_diagram, harmonics=(1, 3, 6))
        texts = _annotations(figure)
        for track in family_diagram.tracks:
            assert track.label.label in texts
        assert {"1P", "3P", "6P"} <= set(texts)
        labels = [
            child
            for child in figure.axes[0].get_children()
            if isinstance(child, Annotation) and child.get_text().startswith("1st")
        ]
        # Labels sit outside the right edge of the axes, at distinct heights.
        assert all(label.xy[0] == pytest.approx(1.0) for label in labels)
        figure.draw_without_rendering()
        boxes = [label.get_window_extent() for label in labels]
        for index, first in enumerate(boxes):
            for second in boxes[index + 1 :]:
                assert not first.overlaps(second)
        plt.close(figure)

    def test_tracks_take_the_family_style(
        self, family_diagram: CampbellDiagram
    ) -> None:
        """Each track has its family colour, cycled dash and marker; opacity fades."""
        family_diagram.tracks[1].confidence = 0.0
        figure = plot_campbell_static(
            family_diagram, show_excitation=False, show_resonances=False
        )
        lines = figure.axes[0].lines
        styles = track_styles(family_diagram.tracks)
        for line, style in zip(lines, styles, strict=False):
            assert to_hex(line.get_color()) == style.color
            assert line.get_marker() == style.mpl_marker
        assert to_hex(lines[1].get_color()) == FAMILY_COLORS[ModeFamily.TOWER]
        assert lines[0].get_marker() == "v"
        assert lines[1].get_alpha() == pytest.approx(0.35)
        assert "Low confidence" in _key_texts(figure)
        plt.close(figure)

    def test_heavily_damped_tracks_drawn_grey(
        self, family_diagram: CampbellDiagram
    ) -> None:
        """Heavily damped tracks are dotted grey lines, and can be left out."""
        figure = plot_campbell_static(
            family_diagram, show_excitation=False, show_resonances=False
        )
        grey = [
            line
            for line in figure.axes[0].lines
            if to_hex(line.get_color()) == HEAVILY_DAMPED_COLOR
        ]
        assert len(grey) == 1
        assert grey[0].get_linestyle() == ":"
        plt.close(figure)
        hidden = plot_campbell_static(
            family_diagram,
            show_excitation=False,
            show_resonances=False,
            show_heavily_damped=False,
        )
        assert len(hidden.axes[0].lines) == len(family_diagram.tracks)
        plt.close(hidden)

    def test_resonances_and_operating_range(
        self, campbell_diagram: CampbellDiagram
    ) -> None:
        """Resonances are filled crosses in the key; the window is shaded and named."""
        figure = plot_campbell_static(campbell_diagram, operating_range=(10.0, 60.0))
        assert any(text.startswith("Resonance") for text in _key_texts(figure))
        crosses = [line for line in figure.axes[0].lines if line.get_marker() == "X"]
        assert crosses
        assert "Operating Speed Range" in _annotations(figure)
        assert figure.axes[0].patches
        plt.close(figure)
        hidden = plot_campbell_static(
            campbell_diagram, operating_range=(10.0, 60.0), show_operating_range=False
        )
        assert "Operating Speed Range" not in _annotations(hidden)
        plt.close(hidden)

    def test_non_rpm_diagram(self, campbell_diagram: CampbellDiagram) -> None:
        """A wind-speed diagram draws neither nP lines nor resonances."""
        campbell_diagram.parameter_name = "wind_speed"
        figure = plot_campbell_static(campbell_diagram, operating_range=(0.0, 30.0))
        texts = _key_texts(figure)
        assert "Blade Passing" not in texts
        assert not any(text.startswith("Resonance") for text in texts)
        assert figure.axes[0].get_xlabel() == "Wind speed (m/s)"
        assert figure.axes[0].get_ylabel() == "Frequency (Hz)"
        assert "Operating Wind Range" in _annotations(figure)
        plt.close(figure)

    def test_ambiguous_track_and_points(
        self, campbell_diagram: CampbellDiagram
    ) -> None:
        """An ambiguous track gets open diamonds; a hybrid point gets an overlay."""
        campbell_diagram.tracks[0].is_ambiguous = True
        track = campbell_diagram.tracks[1]
        clear = replace(track.label, is_ambiguous=False)
        hybrid = replace(track.label, is_ambiguous=True, alternative="other")
        track.point_labels = [clear] * len(track.operating_points)
        track.point_labels[-1] = hybrid
        figure = plot_campbell_static(
            campbell_diagram, show_excitation=False, show_resonances=False
        )
        lines = figure.axes[0].lines
        assert lines[0].get_marker() == "D"
        assert lines[0].get_markerfacecolor() == "none"
        assert lines[1].get_markevery() == [0]
        assert lines[2].get_marker() == "D"
        assert any(text.endswith("(?)") for text in _annotations(figure))
        assert "Ambiguous" in _key_texts(figure)
        plt.close(figure)

    def test_draws_into_supplied_axes(self, campbell_diagram: CampbellDiagram) -> None:
        """Supplied axes are reused, with the labels inside and the key above."""
        figure, axes = plt.subplots()
        returned = plot_campbell_static(campbell_diagram, ax=axes)
        assert returned is figure
        assert axes.get_legend() is not None
        assert not figure.legends
        names = [
            child
            for child in axes.get_children()
            if isinstance(child, Annotation) and "Tower" in child.get_text()
        ]
        assert names[0].get_horizontalalignment() == "right"
        plt.close(figure)

    def test_per_track_legend(self, family_diagram: CampbellDiagram) -> None:
        """direct_labels=False draws a legend with one entry per track, by family."""
        figure = plot_campbell_static(family_diagram, direct_labels=False)
        texts = _key_texts(figure)
        positions = [texts.index(t) for t in ("Blades", "Tower", "Platform", "Other")]
        assert positions == sorted(positions)
        assert texts[texts.index("Tower") + 1] == "1st Tower Fore-Aft"
        assert "3P" in texts
        plt.close(figure)
        figure, axes = plt.subplots()
        plot_campbell_static(family_diagram, ax=axes, direct_labels=False, title="T")
        assert axes.get_legend() is not None
        assert axes.get_title() == "T"
        plt.close(figure)

    def test_narrow_figures_use_short_names_and_grow(
        self, family_diagram: CampbellDiagram
    ) -> None:
        """Below 12 cm the labels are short; a crowded figure grows in height."""
        figure = plot_campbell_static(family_diagram, size=6.0)
        texts = _annotations(figure)
        assert "1st flap reg." in texts
        width, height = figure.get_size_inches()
        assert width == pytest.approx(6.0 / 2.54)
        assert height >= 6.0 * 0.75 / 2.54
        plt.close(figure)

    def test_empty_diagram(self) -> None:
        """A diagram without tracks draws empty axes and no key."""
        diagram = CampbellDiagram(np.array([], dtype=np.float64), "x", [], 0)
        figure = plot_campbell_static(diagram)
        assert not figure.legends
        plt.close(figure)


class TestPlotCampbellPanels:
    """Tests for plot_campbell_panels."""

    @pytest.mark.parametrize(
        ("size", "layout", "shape"),
        [
            ("double", "auto", (1, 2)),
            ("single", "auto", (2, 1)),
            ("double", "column", (2, 1)),
            ("single", "row", (1, 2)),
        ],
    )
    def test_layouts(
        self,
        family_diagram: CampbellDiagram,
        size: str,
        layout: str,
        shape: tuple[int, int],
    ) -> None:
        """Two aligned panels, labelled (a) and (b), share the parameter label."""
        figure = plot_campbell_panels(
            family_diagram,
            size=size,
            layout=layout,  # type: ignore[arg-type]
            operating_range=(3.0, 12.0),
        )
        panels = figure.axes
        assert len(panels) == 2
        positions = [panel.get_position() for panel in panels]
        if shape == (1, 2):
            assert positions[0].x0 < positions[1].x0
            assert figure.get_supxlabel() == "Rotor speed (rpm)"
        else:
            assert positions[0].y0 > positions[1].y0
            assert panels[1].get_xlabel() == "Rotor speed (rpm)"
        texts = _annotations(figure)
        assert "(a)" in texts
        assert "(b)" in texts
        assert panels[0].get_ylabel() == "Frequency (Hz)"
        assert panels[1].get_ylabel() == "Damping ratio (%)"
        assert panels[0].get_shared_x_axes().joined(panels[0], panels[1])
        assert figure.legends
        plt.close(figure)

    def test_per_track_legend_and_bad_layout(
        self, family_diagram: CampbellDiagram
    ) -> None:
        """direct_labels=False uses a per-track legend; a bad layout is rejected."""
        figure = plot_campbell_panels(family_diagram, direct_labels=False)
        assert "1st Tower Fore-Aft" in _key_texts(figure)
        plt.close(figure)
        with pytest.raises(ValueError, match="layout must be"):
            plot_campbell_panels(family_diagram, layout="grid")  # type: ignore[arg-type]


class TestPlotlyCampbell:
    """Encoding and legend of the interactive Campbell and damping plots."""

    def test_traces_grouped_by_family(self, family_diagram: CampbellDiagram) -> None:
        """Track traces carry family legend groups, ranks and one title per group."""
        figure = plot_campbell(family_diagram)
        tracks = figure.data[: len(family_diagram.tracks)]
        groups = [trace.legendgroup for trace in tracks]
        assert groups == ["blade", "tower", "platform", "other", "drivetrain", "blade"]
        ranks = {trace.legendgroup: trace.legendrank for trace in tracks}
        assert ranks["blade"] < ranks["tower"] < ranks["platform"]
        assert ranks["platform"] < ranks["drivetrain"] < ranks["other"]
        assert tracks[0].legendgrouptitle.text == "Blades"
        assert tracks[5].legendgrouptitle.text is None
        assert figure.layout.legend.traceorder == "grouped"
        assert figure.layout.xaxis.title.text == "Rotor speed (rpm)"
        assert figure.layout.yaxis.title.text == "Frequency (Hz)"

    def test_blade_passing_and_inline_tags(
        self, family_diagram: CampbellDiagram
    ) -> None:
        """One Blade Passing legend entry; the nP lines are tagged with annotations."""
        figure = plot_campbell(family_diagram, harmonics=(1, 3, 6))
        legend_names = [
            trace.name for trace in figure.data if trace.showlegend is not False
        ]
        assert "Blade Passing" in legend_names
        assert not {"1P", "3P", "6P"} & set(legend_names)
        tags = {a.text for a in figure.layout.annotations}
        assert {"1P", "3P", "6P"} <= tags
        assert figure.layout.yaxis.range[1] == pytest.approx(1.3 * 1.5)

    def test_track_style_and_hover(self, family_diagram: CampbellDiagram) -> None:
        """Trace colour, dash and marker follow the family style; hover has units."""
        figure = plot_campbell(family_diagram)
        styles = track_styles(family_diagram.tracks)
        for trace, style in zip(figure.data, styles, strict=False):
            assert trace.line.color == style.color
            assert trace.line.dash == style.dash
            assert trace.marker.symbol == style.marker
        hover = figure.data[0].hovertemplate
        for part in ("Rotor speed (rpm)", "Operating point", "Frequency", "Damping"):
            assert part in hover

    def test_operating_range_is_shaded(self, family_diagram: CampbellDiagram) -> None:
        """An operating window is shaded and named; it can be switched off."""
        figure = plot_campbell(family_diagram, operating_range=(3.0, 9.0))
        assert len(figure.layout.shapes) == 1
        assert "Operating Speed Range" in {a.text for a in figure.layout.annotations}
        hidden = plot_campbell(
            family_diagram, operating_range=(3.0, 9.0), show_operating_range=False
        )
        assert len(hidden.layout.shapes) == 0

    def test_damping_traces_grouped_by_family(
        self, family_diagram: CampbellDiagram
    ) -> None:
        """The damping plot uses the same family styling and grouping."""
        figure = plot_damping(family_diagram)
        styles = track_styles(family_diagram.tracks)
        for trace, style in zip(figure.data, styles, strict=False):
            assert trace.legendgroup == style.family.value
            assert trace.line.color == style.color
            assert trace.line.dash == style.dash
        assert figure.layout.title.text is None


class TestPlotDampingStatic:
    """Tests for plot_damping_static."""

    def test_bands_key_and_labels(self, family_diagram: CampbellDiagram) -> None:
        """Stability bands, the compact key and direct labels are drawn."""
        figure = plot_damping_static(family_diagram)
        axes = figure.axes[0]
        assert len(axes.patches) == 2
        assert _key_texts(figure)[:2] == ["Blades", "Tower"]
        texts = _annotations(figure)
        assert "unstable" in texts
        assert "1st Tower Fore-Aft" in texts
        assert axes.get_ylabel() == "Damping ratio (%)"
        plt.close(figure)

    def test_bands_can_be_turned_off(self, family_diagram: CampbellDiagram) -> None:
        """Without bands no patch is drawn; a zero caution band is skipped."""
        figure = plot_damping_static(family_diagram, show_stability_bands=False)
        assert len(figure.axes[0].patches) == 0
        plt.close(figure)
        figure = plot_damping_static(family_diagram, caution_damping=0.0)
        assert len(figure.axes[0].patches) == 1
        plt.close(figure)

    def test_instability_is_flagged(self, family_diagram: CampbellDiagram) -> None:
        """A negative-damping point is crossed and listed in the key."""
        track = family_diagram.tracks[1]
        track.damping_ratios = [0.01, -0.01, float("nan")]
        figure = plot_damping_static(family_diagram, as_percent=False, title="T")
        axes = figure.axes[0]
        crosses = [line for line in axes.lines if line.get_marker() == "X"]
        assert len(crosses) == 1
        assert "Unstable" in _key_texts(figure)
        assert axes.get_ylabel() == "Damping ratio (-)"
        assert axes.get_title() == "T"
        plt.close(figure)

    def test_ambiguity_markers(self, campbell_diagram: CampbellDiagram) -> None:
        """Ambiguous tracks and points get open diamonds, as in the Campbell plot."""
        campbell_diagram.tracks[0].is_ambiguous = True
        track = campbell_diagram.tracks[1]
        clear = replace(track.label, is_ambiguous=False)
        hybrid = replace(track.label, is_ambiguous=True, alternative="other")
        track.point_labels = [clear] * len(track.operating_points)
        track.point_labels[0] = hybrid
        figure = plot_damping_static(campbell_diagram)
        lines = figure.axes[0].lines
        assert lines[0].get_marker() == "D"
        assert lines[2].get_marker() == "D"
        plt.close(figure)

    def test_negative_caution_rejected(self, family_diagram: CampbellDiagram) -> None:
        """A negative caution threshold is rejected."""
        with pytest.raises(ValueError, match="caution_damping must be non-negative"):
            plot_damping_static(family_diagram, caution_damping=-0.1)

    def test_supplied_axes_and_per_track_legend(
        self, family_diagram: CampbellDiagram
    ) -> None:
        """Supplied axes are reused; direct_labels=False gives a per-track legend."""
        figure, axes = plt.subplots()
        assert plot_damping_static(family_diagram, ax=axes) is figure
        assert axes.get_legend() is not None
        plt.close(figure)
        figure = plot_damping_static(family_diagram, direct_labels=False)
        assert "1st Tower Fore-Aft" in _key_texts(figure)
        plt.close(figure)


class TestOtherStaticViews:
    """Branches of the other static views."""

    def test_mac_values_written_or_not(self, mode_solution: ModalSolution) -> None:
        """Small matrices carry their values; annotate=False leaves them out."""
        figure = plot_mac_matrix_static(mode_solution, mode_solution, title="T")
        assert any(t.get_text() == "1.00" for t in figure.axes[0].texts)
        plt.close(figure)
        figure = plot_mac_matrix_static(mode_solution, mode_solution, annotate=False)
        assert not figure.axes[0].texts
        plt.close(figure)

    def test_branch_review_without_groups(
        self, campbell_diagram: CampbellDiagram
    ) -> None:
        """No contested branches gives a single placeholder panel."""
        figure = plot_branch_review_static(campbell_diagram, [])
        assert "No contested branches" in figure.axes[0].texts[0].get_text()
        plt.close(figure)

    def test_blade_components_need_mbc(self, mode_solution: ModalSolution) -> None:
        """A solution without multi-blade coordinates is rejected."""
        with pytest.raises(ValueError, match="multi-blade coordinates"):
            plot_blade_components_static(mode_solution, 0)

    def test_phasor_many_dofs(self, mode_solution: ModalSolution) -> None:
        """The phasor wheel names its DOFs in a legend titled with the mode."""
        figure = plot_mode_phasor_static(mode_solution, 0, title="T")
        legend = figure.axes[0].get_legend()
        assert legend is not None
        assert "Hz" in legend.get_title().get_text()
        plt.close(figure)

    def test_turbine_view_onshore_and_trimmed_mooring(self) -> None:
        """An onshore view has no water plane; mooring lines are cut at the view."""
        shapes = np.zeros((8, 1), dtype=np.complex128)
        shapes[3, 0] = 1.0
        solution = _fowt_solution(shapes)
        geometry = default_geometry(SystemType.LAND)
        figure = plot_mode_fowt_static(solution, 0, geometry, title="T", phase_deg=0.0)
        assert figure.axes[0].get_zlabel() == "z (m)"
        plt.close(figure)
        nodes = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
        assert _within(nodes, 10.0).shape == (2, 3)
        assert _within(nodes[::-1].copy(), 10.0).shape == (2, 3)
        assert _within(nodes[:2], 10.0).shape == (2, 3)
