# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.viz.palette — the standard colours and the Okabe-Ito option."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go

from vane.viz import STANDARD_COLORS as PUBLIC_COLORS
from vane.viz import mode_line_style
from vane.viz.campbell_plot import plot_campbell
from vane.viz.damping_plot import plot_damping
from vane.viz.palette import (
    OKABE_ITO,
    STANDARD_COLORS,
    apply_colorblind_palette,
    apply_standard_palette,
    colorblind_colorway,
    standard_colorway,
)

if TYPE_CHECKING:
    from vane.campbell.builder import CampbellDiagram


def test_palette_is_the_style_colour_order() -> None:
    """The palette module re-exports the one colour order of the style."""
    assert STANDARD_COLORS is PUBLIC_COLORS
    assert standard_colorway() == list(STANDARD_COLORS)


def test_apply_standard_palette_sets_colorway() -> None:
    """apply_standard_palette sets the figure's colorway and returns it."""
    figure = go.Figure()
    assert apply_standard_palette(figure) is figure
    assert tuple(figure.layout.colorway) == STANDARD_COLORS


def test_okabe_ito_has_eight_distinct_hex_colors() -> None:
    """The Okabe-Ito palette is eight distinct 6-digit hex colours."""
    assert len(OKABE_ITO) == 8
    assert len(set(OKABE_ITO)) == 8
    assert all(c.startswith("#") and len(c) == 7 for c in OKABE_ITO)
    assert colorblind_colorway() == list(OKABE_ITO)


def test_campbell_and_damping_use_the_colour_order(
    campbell_diagram: CampbellDiagram,
) -> None:
    """The template colourway is the standard colours; tracks take family colours."""
    figure = plot_campbell(campbell_diagram)
    assert tuple(figure.layout.template.layout.colorway) == STANDARD_COLORS
    for trace, track in zip(figure.data, campbell_diagram.tracks, strict=False):
        assert trace.line.color == mode_line_style(track.label).color
    damping = plot_damping(campbell_diagram)
    assert tuple(damping.layout.template.layout.colorway) == STANDARD_COLORS


def test_apply_colorblind_palette_swaps_by_hue(
    campbell_diagram: CampbellDiagram,
) -> None:
    """Standard colours become the Okabe-Ito colour of the nearest hue."""
    figure = plot_campbell(campbell_diagram)
    returned = apply_colorblind_palette(figure)
    assert returned is figure
    assert tuple(figure.layout.colorway) == OKABE_ITO
    # The fixture's first track is a tower mode (black), which stays black.
    assert figure.data[0].line.color == "#000000"
    # The blue excitation lines become the Okabe-Ito blue.
    excitation = [t for t in figure.data if t.name and t.name.endswith("P")]
    assert {t.line.color for t in excitation} == {"#0072B2"}


def test_apply_colorblind_palette_handles_colour_lists() -> None:
    """Per-point colour lists are swapped element by element."""
    figure = go.Figure(
        go.Bar(
            x=[0, 1, 2], y=[1, 2, 3], marker_color=[*STANDARD_COLORS[1:3], "#123456"]
        )
    )
    apply_colorblind_palette(figure)
    assert list(figure.data[0].marker.color) == ["#D55E00", "#0072B2", "#123456"]
