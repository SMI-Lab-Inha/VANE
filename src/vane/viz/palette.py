# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Colour palettes of the figures.

Every VANE figure uses one palette, the standard engineering-paper line colours
(:data:`STANDARD_COLORS`: black, red, blue, green, magenta, orange, cyan), defined once
with the rest of the figure style (see :func:`vane.viz.apply_style`) and re-exported
here. :func:`apply_standard_palette` sets it as a Plotly figure's trace colour cycle.

For readers with a colour-vision deficiency, the Okabe-Ito qualitative palette [1]_
(:data:`OKABE_ITO`), an eight-colour set designed to stay distinguishable across the
common colour-vision deficiencies, is available as an opt-in alternative:
:func:`apply_colorblind_palette` replaces each standard colour of a finished figure by
the Okabe-Ito colour of the nearest hue (black stays black, red becomes vermillion,
blue stays blue, green becomes bluish green, and so on), so that the mode families
stay distinct.

References
----------
.. [1] Okabe, M. and Ito, K. (2008). Color Universal Design (CUD): how to make figures
   and presentations that are friendly to colorblind people.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vane.viz._style import STANDARD_COLORS, standard_colorway

if TYPE_CHECKING:
    import plotly.graph_objects as go

__all__ = [
    "OKABE_ITO",
    "STANDARD_COLORS",
    "apply_colorblind_palette",
    "apply_standard_palette",
    "colorblind_colorway",
    "standard_colorway",
]

# The Okabe-Ito colourblind-safe qualitative palette (hex). Black is the canonical
# eighth colour and is ordered last so the chromatic hues cycle first, keeping it clear
# of the black axes/annotations on a white background.
OKABE_ITO: tuple[str, ...] = (
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black
)

# Standard colour (lower-case hex) -> the Okabe-Ito colour of the nearest hue.
_TO_OKABE_ITO = dict(
    zip(
        (color.lower() for color in STANDARD_COLORS),
        (
            OKABE_ITO[7],  # black -> black
            OKABE_ITO[5],  # red -> vermillion
            OKABE_ITO[4],  # blue -> blue
            OKABE_ITO[2],  # green -> bluish green
            OKABE_ITO[6],  # magenta -> reddish purple
            OKABE_ITO[0],  # orange -> orange
            OKABE_ITO[1],  # cyan -> sky blue
        ),
        strict=True,
    )
)


def colorblind_colorway() -> list[str]:
    """Return the Okabe-Ito palette as a Plotly ``colorway`` list.

    Returns
    -------
    list[str]
        The eight colours of :data:`OKABE_ITO`.
    """
    return list(OKABE_ITO)


def apply_standard_palette(figure: go.Figure) -> go.Figure:
    """Set a figure's trace colour cycle to the standard line colours.

    Parameters
    ----------
    figure : plotly.graph_objects.Figure
        The figure; its ``layout.colorway`` is set so that traces added without an
        explicit colour cycle through :data:`STANDARD_COLORS`.

    Returns
    -------
    plotly.graph_objects.Figure
        The same figure, for chaining.
    """
    figure.update_layout(colorway=standard_colorway())
    return figure


def _swap(color: object) -> object:
    """Return the Okabe-Ito colour for a standard colour; other values are unchanged."""
    if isinstance(color, str):
        return _TO_OKABE_ITO.get(color.lower(), color)
    if isinstance(color, (list, tuple)):
        return [_swap(item) for item in color]
    return color


def apply_colorblind_palette(figure: go.Figure) -> go.Figure:
    """Recolour a figure with the colourblind-safe Okabe-Ito palette.

    The figure's ``layout.colorway`` becomes :data:`OKABE_ITO`, and every explicit line
    or marker colour that is one of :data:`STANDARD_COLORS` (such as the mode-family
    colours of the Campbell diagram) is replaced by the Okabe-Ito colour of the
    nearest hue. Other explicit colours (greys, colour scales) are left unchanged.

    Parameters
    ----------
    figure : plotly.graph_objects.Figure
        The figure to recolour, in place.

    Returns
    -------
    plotly.graph_objects.Figure
        The same figure, for chaining.
    """
    figure.update_layout(colorway=colorblind_colorway())
    for trace in figure.data:
        for part in ("line", "marker"):
            if part not in trace:
                continue
            style = trace[part]
            if "color" in style and style.color is not None:
                style.color = _swap(style.color)
    return figure
