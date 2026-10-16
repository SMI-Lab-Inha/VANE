# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.viz._style — the shared style and the mode-family encoding."""

from __future__ import annotations

import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import pytest
from matplotlib.colors import to_hex
from matplotlib.lines import TICKRIGHT, TICKUP, Line2D

from vane.config.dof_map import DofCategory
from vane.modal.identifier import ModeTrack
from vane.modal.labeler import ModeLabel
from vane.viz import (
    FAMILY_COLORS,
    FAMILY_TITLES,
    PLOTLY_TEMPLATE,
    STANDARD_COLORS,
    STANDARD_LINES,
    VANE_STYLE,
    ModeFamily,
    apply_style,
    mode_family,
    mode_line_style,
    plot_style,
    plotly_template,
    standard_colorway,
    style_rc,
)
from vane.viz._style import (
    _DASHES,
    SANS_SERIF_FONTS,
    _spread,
    excitation_tag_positions,
    resolve_size,
    resolve_template,
    series_colors,
    short_dof_name,
    short_mode_name,
    static_style,
    track_styles,
)
from vane.viz.publication import figure_size

# The standard engineering-paper line colours, as specified.
_STANDARD = (
    (0.0, 0.0, 0.0),
    (0.85, 0.0, 0.0),
    (0.0, 0.0, 0.85),
    (0.0, 0.6, 0.0),
    (0.8, 0.0, 0.8),
    (1.0, 0.5, 0.0),
    (0.0, 0.7, 0.7),
)
_FRAME = to_hex((0.15, 0.15, 0.15))

_EXPECTED_FAMILY = {
    DofCategory.PLATFORM_SURGE: ModeFamily.PLATFORM,
    DofCategory.PLATFORM_SWAY: ModeFamily.PLATFORM,
    DofCategory.PLATFORM_HEAVE: ModeFamily.PLATFORM,
    DofCategory.PLATFORM_ROLL: ModeFamily.PLATFORM,
    DofCategory.PLATFORM_PITCH: ModeFamily.PLATFORM,
    DofCategory.PLATFORM_YAW: ModeFamily.PLATFORM,
    DofCategory.TOWER_FORE_AFT_1: ModeFamily.TOWER,
    DofCategory.TOWER_SIDE_SIDE_1: ModeFamily.TOWER,
    DofCategory.TOWER_FORE_AFT_2: ModeFamily.TOWER,
    DofCategory.TOWER_SIDE_SIDE_2: ModeFamily.TOWER,
    DofCategory.GENERATOR_AZIMUTH: ModeFamily.DRIVETRAIN,
    DofCategory.DRIVETRAIN_TORSION: ModeFamily.DRIVETRAIN,
    DofCategory.BLADE_FLAP_1: ModeFamily.BLADE,
    DofCategory.BLADE_FLAP_2: ModeFamily.BLADE,
    DofCategory.BLADE_FLAP: ModeFamily.BLADE,
    DofCategory.BLADE_EDGE_1: ModeFamily.BLADE,
    DofCategory.BLADE_EDGE: ModeFamily.BLADE,
    DofCategory.BLADE_PITCH: ModeFamily.BLADE,
    DofCategory.BLADE_TORSION: ModeFamily.BLADE,
    DofCategory.BLADE_AXIAL: ModeFamily.BLADE,
    DofCategory.NACELLE_YAW: ModeFamily.OTHER,
    DofCategory.ROTOR_FURL: ModeFamily.OTHER,
    DofCategory.TAIL_FURL: ModeFamily.OTHER,
    DofCategory.TEETER: ModeFamily.OTHER,
    DofCategory.UNKNOWN: ModeFamily.OTHER,
}

_MULTIBLADE_TYPES = (None, "collective", "regressive", "progressive", "cyclic")


def _label(
    category: DofCategory, text: str = "Mode", multiblade: str | None = None
) -> ModeLabel:
    """Return a minimal mode label."""
    return ModeLabel(category, text, 0.9, [], multiblade)


def _track(category: DofCategory) -> ModeTrack:
    """Return a one-point track of a category."""
    return ModeTrack([0], [0], [1.0], [0.01], _label(category))


class TestPalette:
    """The standard line colours."""

    def test_order(self) -> None:
        """Black, red, blue, green, magenta, orange, cyan, in that order."""
        assert STANDARD_LINES == _STANDARD
        assert tuple(to_hex(rgb) for rgb in _STANDARD) == STANDARD_COLORS
        assert standard_colorway() == list(STANDARD_COLORS)


class TestStyleRc:
    """The Matplotlib rcParams of the style."""

    def test_colour_cycle(self) -> None:
        """The colour cycle is the standard line colours."""
        colors = [entry["color"] for entry in style_rc()["axes.prop_cycle"]]
        assert colors == list(STANDARD_COLORS)

    def test_fonts(self) -> None:
        """Helvetica, Arial, Liberation Sans, DejaVu Sans; 10 pt with 9 pt ticks."""
        rc = style_rc()
        assert rc["font.sans-serif"] == [
            "Helvetica",
            "Arial",
            "Liberation Sans",
            "DejaVu Sans",
        ]
        assert rc["font.family"][-1] == "sans-serif"
        assert all(f in (*SANS_SERIF_FONTS, "sans-serif") for f in rc["font.family"])
        assert rc["font.size"] == pytest.approx(10.0)
        assert rc["axes.labelsize"] == pytest.approx(10.0)
        assert rc["xtick.labelsize"] == rc["ytick.labelsize"] == pytest.approx(9.0)
        assert rc["legend.fontsize"] == pytest.approx(9.0)
        assert rc["axes.titlesize"] == pytest.approx(11.0)
        assert rc["axes.titleweight"] == "normal"

    def test_frame_and_ticks(self) -> None:
        """Boxed axes in soft dark grey, ticks in on all four sides."""
        rc = style_rc()
        for side in ("left", "right", "top", "bottom"):
            assert rc[f"axes.spines.{side}"] is True
        assert rc["axes.edgecolor"] == _FRAME
        assert rc["axes.linewidth"] == pytest.approx(0.75)
        assert rc["xtick.direction"] == rc["ytick.direction"] == "in"
        for key in ("xtick.top", "xtick.bottom", "ytick.left", "ytick.right"):
            assert rc[key] is True
        assert rc["xtick.major.size"] == pytest.approx(4.0)
        assert rc["xtick.minor.size"] == pytest.approx(2.0)
        assert rc["xtick.major.width"] == pytest.approx(0.75)
        assert rc["xtick.minor.width"] == pytest.approx(0.5)

    def test_lines_grid_legend_output(self) -> None:
        """1.0 pt lines, 5 pt markers, no grid, square opaque legend, 300 dpi."""
        rc = style_rc()
        assert rc["lines.linewidth"] == pytest.approx(1.0)
        assert rc["lines.markersize"] == pytest.approx(5.0)
        assert rc["axes.grid"] is False
        assert rc["grid.linestyle"] == "--"
        assert rc["legend.frameon"] is True
        assert rc["legend.framealpha"] == pytest.approx(1.0)
        assert rc["legend.fancybox"] is False
        assert rc["legend.facecolor"] == "white"
        assert rc["legend.edgecolor"] == _FRAME
        assert rc["savefig.dpi"] == 300
        assert rc["savefig.bbox"] == "tight"


class TestApplyStyle:
    """apply_style and the plot_style context manager."""

    def test_apply_style_sets_the_rcparams(self) -> None:
        """After apply_style the global rcParams carry the style values."""
        with mpl.rc_context():
            apply_style()
            for key, value in style_rc().items():
                if key == "axes.prop_cycle":
                    colors = [c["color"] for c in mpl.rcParams[key]]
                    assert colors == list(STANDARD_COLORS)
                elif isinstance(value, list):
                    assert list(mpl.rcParams[key]) == value, key
                else:
                    assert mpl.rcParams[key] == value, key

    def test_context_applies_and_restores(self) -> None:
        """The style is in effect inside the block and the rcParams are restored."""
        with mpl.rc_context({"lines.linewidth": 3.3, "xtick.direction": "out"}):
            with plot_style():
                assert mpl.rcParams["lines.linewidth"] == pytest.approx(1.0)
                assert mpl.rcParams["xtick.direction"] == "in"
            assert mpl.rcParams["lines.linewidth"] == pytest.approx(3.3)
            assert mpl.rcParams["xtick.direction"] == "out"

    def test_context_restores_after_an_exception(self) -> None:
        """An exception inside the block still restores the rcParams."""
        before = mpl.rcParams["lines.linewidth"]
        message = "boom"
        with pytest.raises(RuntimeError, match=message), plot_style():
            raise RuntimeError(message)
        assert mpl.rcParams["lines.linewidth"] == before

    def test_figure_made_inside_has_box_and_inward_ticks(self) -> None:
        """Axes created inside the block carry the box and the inward ticks."""
        with plot_style():
            figure, axes = plt.subplots()
            axes.plot([0.0, 1.0], [0.0, 1.0])
        assert all(spine.get_visible() for spine in axes.spines.values())
        xtick = axes.xaxis.get_major_ticks()[0]
        ytick = axes.yaxis.get_major_ticks()[0]
        assert xtick.tick1line.get_marker() == TICKUP
        assert ytick.tick1line.get_marker() == TICKRIGHT
        assert xtick.tick2line.get_visible()
        assert to_hex(axes.lines[0].get_color()) == STANDARD_COLORS[0]
        plt.close(figure)

    def test_restyles_an_existing_figure_and_axes(self) -> None:
        """A figure or axes made elsewhere gets the box, ticks and colour cycle."""
        with mpl.rc_context({"axes.spines.top": False, "xtick.direction": "out"}):
            figure, axes = plt.subplots()
        apply_style(figure)
        assert axes.spines["top"].get_visible()
        assert axes.xaxis.get_tick_params(which="major")["direction"] == "in"
        assert to_hex(axes.plot([0.0, 1.0])[0].get_color()) == STANDARD_COLORS[0]
        plt.close(figure)
        with mpl.rc_context({"axes.spines.right": False}):
            figure, axes = plt.subplots()
        apply_style(axes)
        assert axes.spines["right"].get_visible()
        plt.close(figure)

    def test_plotly_figure_gets_the_template(self) -> None:
        """A Plotly figure gets the vane template; other types are rejected."""
        figure = go.Figure()
        apply_style(figure)
        assert figure.layout.template.layout.colorway == STANDARD_COLORS
        with pytest.raises(TypeError, match="apply_style expects"):
            apply_style("figure")  # type: ignore[arg-type]


class TestStaticStyle:
    """Resolution of the static plots' style and size arguments."""

    def test_vane_style_and_size(self) -> None:
        """The VANE style with a size preset sets both."""
        with static_style(VANE_STYLE, figure_size("single")):
            assert mpl.rcParams["xtick.direction"] == "in"
            width, _ = mpl.rcParams["figure.figsize"]
            assert width == pytest.approx(8.5 / 2.54)
            assert mpl.rcParams["axes.labelsize"] == pytest.approx(9.0)

    def test_none_keeps_the_caller_rcparams(self) -> None:
        """None leaves the current rcParams in effect; a dict is applied."""
        with mpl.rc_context({"lines.linewidth": 4.2}), static_style(None):
            assert mpl.rcParams["lines.linewidth"] == pytest.approx(4.2)
        with static_style({"lines.linewidth": 2.7}):
            assert mpl.rcParams["lines.linewidth"] == pytest.approx(2.7)

    def test_resolve_size(self) -> None:
        """Explicit sizes always apply; defaults only to new figures in a style."""
        assert resolve_size(None, "single", ax=None, style=None) is None
        figure, axes = plt.subplots()
        assert resolve_size(None, "single", ax=axes, style=VANE_STYLE) is None
        plt.close(figure)
        double = resolve_size(
            None, "double", ax=None, style=VANE_STYLE, aspect=(1.0, 2.0)
        )
        assert double is not None
        assert double.height_cm == pytest.approx(34.0)
        single = resolve_size(
            "single", "double", ax=None, style=None, aspect=(1.0, 2.0)
        )
        assert single is not None
        assert single.height_cm == pytest.approx(8.5)


class TestPlotlyTemplate:
    """The registered vane Plotly template."""

    def test_layout_keys(self) -> None:
        """Palette, fonts, white backgrounds, mirrored box, inward ticks, no grid."""
        assert PLOTLY_TEMPLATE == "vane"
        layout = pio.templates[PLOTLY_TEMPLATE].layout
        assert layout.colorway == STANDARD_COLORS
        assert layout.font.family.startswith("Helvetica, Arial")
        assert layout.paper_bgcolor == layout.plot_bgcolor == "white"
        for axis in (layout.xaxis, layout.yaxis):
            assert axis.mirror == "ticks"
            assert axis.ticks == "inside"
            assert axis.showline is True
            assert axis.showgrid is False
            assert axis.linecolor == _FRAME
        assert layout.legend.bordercolor == _FRAME
        assert layout.legend.bgcolor == "white"
        assert layout.scene.xaxis.ticks == "inside"
        assert layout.polar.radialaxis.ticks == "inside"

    def test_builder_and_resolution(self) -> None:
        """plotly_template builds a copy; None resolves to vane, re-registered."""
        built = plotly_template()
        assert built.to_plotly_json() == pio.templates[PLOTLY_TEMPLATE].to_plotly_json()
        assert resolve_template("plotly_white") == "plotly_white"
        saved = pio.templates[PLOTLY_TEMPLATE]
        del pio.templates[PLOTLY_TEMPLATE]
        try:
            assert resolve_template(None) == PLOTLY_TEMPLATE
            assert PLOTLY_TEMPLATE in pio.templates
        finally:
            pio.templates[PLOTLY_TEMPLATE] = saved


class TestModeFamily:
    """The mode-family encoding of the Campbell and damping plots."""

    def test_every_category_has_the_expected_family(self) -> None:
        """Every DofCategory maps to its family; the table covers the whole enum."""
        assert set(_EXPECTED_FAMILY) == set(DofCategory)
        for category, family in _EXPECTED_FAMILY.items():
            assert mode_family(category) is family

    def test_family_colours_and_titles(self) -> None:
        """Blades green, tower black, platform red, drivetrain magenta, other cyan."""
        assert FAMILY_COLORS[ModeFamily.BLADE] == STANDARD_COLORS[3]
        assert FAMILY_COLORS[ModeFamily.TOWER] == STANDARD_COLORS[0]
        assert FAMILY_COLORS[ModeFamily.PLATFORM] == STANDARD_COLORS[1]
        assert FAMILY_COLORS[ModeFamily.DRIVETRAIN] == STANDARD_COLORS[4]
        assert FAMILY_COLORS[ModeFamily.OTHER] == STANDARD_COLORS[6]
        assert FAMILY_TITLES[ModeFamily.BLADE] == "Blades"
        assert set(FAMILY_TITLES) == set(ModeFamily)

    def test_line_style_cycles_within_a_family(self) -> None:
        """The index sets the dash; the cycle wraps; the colour stays the family's."""
        label = _label(DofCategory.TOWER_FORE_AFT_1)
        dashes = [mode_line_style(label, k).dash for k in range(len(_DASHES) + 1)]
        assert dashes[:4] == ["solid", "dash", "dashdot", "dot"]
        assert dashes[len(_DASHES)] == dashes[0]
        assert {mode_line_style(label, k).color for k in range(3)} == {
            FAMILY_COLORS[ModeFamily.TOWER]
        }

    def test_track_styles_count_per_family(self) -> None:
        """Each family's tracks cycle the line styles independently."""
        tracks = [
            _track(DofCategory.TOWER_FORE_AFT_1),
            _track(DofCategory.BLADE_FLAP_1),
            _track(DofCategory.TOWER_SIDE_SIDE_1),
            _track(DofCategory.BLADE_EDGE_1),
        ]
        dashes = [style.dash for style in track_styles(tracks)]
        assert dashes == ["solid", "solid", "dash", "dash"]

    def test_multiblade_type_sets_the_marker(self) -> None:
        """Collective circle, regressive and progressive triangles, cyclic square."""
        markers = {
            kind: mode_line_style(_label(DofCategory.BLADE_FLAP_1, "Flap", kind))
            for kind in _MULTIBLADE_TYPES
        }
        assert markers[None].marker == markers["collective"].marker == "circle"
        assert markers["regressive"].marker == "triangle-down"
        assert markers["progressive"].marker == "triangle-up"
        assert markers["cyclic"].marker == "square"
        assert markers["regressive"].mpl_marker == "v"
        assert markers["progressive"].mpl_marker == "^"

    def test_mapping_is_deterministic(self) -> None:
        """Every category and multi-blade type gets the same style on every call."""
        for category in DofCategory:
            for kind in _MULTIBLADE_TYPES:
                label = _label(category, "Mode", kind)
                first = mode_line_style(label, 2)
                assert mode_line_style(label, 2) == first
                assert first.color == FAMILY_COLORS[mode_family(category)]

    def test_dashes_are_valid_for_both_libraries(self) -> None:
        """Every dash is accepted by Plotly and by Matplotlib."""
        for plotly_dash, linestyle in _DASHES:
            go.Scatter(line={"dash": plotly_dash})
            Line2D([], [], linestyle=linestyle)


class TestNamesAndPlacement:
    """Short names, series colours and the label and tag placement."""

    def test_short_mode_names(self) -> None:
        """Mode labels shorten for narrow figures."""
        assert short_mode_name("1st Blade Flap (Regressive)") == "1st flap reg."
        assert short_mode_name("1st Tower Side-Side") == "1st tower SS"
        assert short_mode_name("Platform Pitch") == "Pitch"

    def test_short_dof_names(self) -> None:
        """DOF descriptions shorten to their token, or to their first characters."""
        tower = "ED 1st tower fore-aft (internal DOF index = DOF_TFA1), m"
        flap = "ED 1st flapwise DOF of blade 2 (internal DOF index = DOF_BF(2,1)), m"
        pitch = "ED pitch of blade 3 (internal DOF index = DOF_BP(3)), rad"
        assert short_dof_name(tower) == "Tower FA1"
        assert short_dof_name(flap) == "Blade 2 flap 1"
        assert short_dof_name(flap, "cosine") == "Flap 1 cos."
        assert short_dof_name(pitch) == "Blade 3 pitch"
        assert short_dof_name("s0") == "s0"
        long = "A description far longer than the limit of twenty-eight characters"
        assert short_dof_name(long).endswith("…")
        assert len(short_dof_name(long)) <= 28

    def test_series_colours(self) -> None:
        """Up to seven series use the palette; more use a colour map, all distinct."""
        assert series_colors(3) == list(STANDARD_COLORS[:3])
        assert len(set(series_colors(10))) == 10

    def test_spread_keeps_order_and_gap(self) -> None:
        """Labels keep their order, stay a gap apart, and stay within bounds."""
        placed = _spread([5.0, 5.0, 5.2, 50.0], 2.0, 0.0, 60.0)
        ordered = sorted(placed)
        gaps = [b - a for a, b in itertools.pairwise(ordered)]
        assert min(gaps) >= 2.0 - 1e-9
        assert placed[3] == pytest.approx(50.0)
        crowded = _spread([10.0] * 5, 5.0, 0.0, 8.0)
        assert min(crowded) >= 0.0
        assert max(crowded) <= 8.0
        assert max(_spread([9.0, 9.5], 2.0, 0.0, 10.0)) <= 10.0

    def test_excitation_tags_lie_on_their_lines(self) -> None:
        """Tags lie on their lines inside the plot; lines outside get none."""
        tags = excitation_tag_positions(
            [0, 1, 3, 6],
            (0.0, 12.0),
            (0.0, 1.0),
            [([0.0, 6.0], [0.5, 0.5]), ([7.0], [0.7]), ([], [])],
        )
        assert [order for order, _, _ in tags] == [1, 3, 6]
        for order, x, y in tags:
            assert y == pytest.approx(order * x / 60.0)
            assert 0.0 <= x <= 12.0
            assert 0.0 <= y <= 1.0
        assert excitation_tag_positions([100], (0.0, 1.0), (5.0, 6.0)) == []
