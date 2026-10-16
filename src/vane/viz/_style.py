# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""The figure style shared by every plot in :mod:`vane.viz`.

All VANE figures follow the plotting conventions of the Marine Structural Mechanics
and Integrity Lab (SMI Lab), Inha University, defined once here:

* the standard engineering-paper line colours, in order: black, red, blue, green,
  magenta, orange, cyan (:data:`STANDARD_LINES`);
* Helvetica or Arial (falling back to Liberation Sans, then DejaVu Sans), 10 pt text
  with 9 pt tick labels and legends, normal-weight titles;
* boxed axes with a soft dark-grey frame, ticks pointing inward on all four sides, a
  white background, 1.0 pt lines and 5 pt markers; no grid unless switched on, and
  then a dashed light grey one;
* legends with a thin grey border and an opaque white fill;
* saved figures at 300 dpi, trimmed to their content.

For Matplotlib the style is a set of ``rcParams`` (:func:`style_rc`), applied
permanently by :func:`apply_style` or temporarily by the :func:`plot_style` context
manager. For Plotly it is the registered template ``"vane"`` (:data:`PLOTLY_TEMPLATE`,
built by :func:`plotly_template`). Every static plotting function draws inside
:func:`plot_style` and every interactive one applies the template, unless the caller
passes another style or template.

The module also holds the per-track encoding of the Campbell and damping plots: the
mode family sets the colour (blades green, tower black, platform red, drivetrain
magenta, other modes cyan; the ``nP`` excitation lines are blue), the line style
cycles within a family, the multi-blade type sets the marker, and line opacity scales
with identification confidence (:func:`mode_line_style`, :func:`confidence_opacity`).
"""

from __future__ import annotations

import contextlib
import functools
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.colors import to_hex, to_rgb
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.rcsetup import cycler
from matplotlib.transforms import offset_copy

from vane.config.dof_map import DofCategory
from vane.viz.publication import FigureSize, figure_size

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from matplotlib.artist import Artist
    from matplotlib.legend import Legend
    from matplotlib.text import Annotation

    from vane.modal.identifier import ModeTrack
    from vane.modal.labeler import ModeLabel

__all__ = [
    "FAMILY_COLORS",
    "FAMILY_TITLES",
    "PLOTLY_TEMPLATE",
    "STANDARD_COLORS",
    "STANDARD_LINES",
    "VANE_STYLE",
    "ModeFamily",
    "ModeLineStyle",
    "apply_style",
    "mode_family",
    "mode_line_style",
    "plot_style",
    "plotly_template",
    "standard_colorway",
    "style_rc",
]

#: The standard engineering-paper line colours, in order, as RGB triples in ``[0, 1]``:
#: black, red, blue, green, magenta, orange, cyan.
STANDARD_LINES: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),
    (0.85, 0.0, 0.0),
    (0.0, 0.0, 0.85),
    (0.0, 0.6, 0.0),
    (0.8, 0.0, 0.8),
    (1.0, 0.5, 0.0),
    (0.0, 0.7, 0.7),
)

#: The same colours as hex strings.
STANDARD_COLORS: tuple[str, ...] = tuple(to_hex(rgb) for rgb in STANDARD_LINES)

#: Name of the registered Plotly template that every interactive figure applies.
PLOTLY_TEMPLATE = "vane"

#: Value of the static plots' ``style=`` argument that selects the VANE style.
VANE_STYLE = "vane"

# Neutral colours: the soft dark-grey frame and text, the dashed grid, reference lines.
AXES_COLOR = to_hex((0.15, 0.15, 0.15))
GRID_COLOR = to_hex((0.65, 0.65, 0.65))
REFERENCE_COLOR = "#808080"
# Shade of the operating range.
OPERATING_RANGE_COLOR = to_hex((0.85, 0.85, 0.85))
# Colour of the nP excitation ("blade passing") lines.
EXCITATION_COLOR = STANDARD_COLORS[2]
# Colour of alarms (instability, high-severity resonance).
ALERT_COLOR = STANDARD_COLORS[1]

# Preferred fonts, in order; the system sans-serif font is the last resort.
SANS_SERIF_FONTS: tuple[str, ...] = (
    "Helvetica",
    "Arial",
    "Liberation Sans",
    "DejaVu Sans",
)
PLOTLY_FONT_FAMILY = "Helvetica, Arial, 'Liberation Sans', sans-serif"

_AXES_LINEWIDTH = 0.75
_LINE_WIDTH = 1.0
_MARKER_SIZE = 5.0
_FONT_SIZE = 10.0
_SMALL_FONT_SIZE = 9.0
_TITLE_SIZE = 11.0
# Plotly sizes in CSS pixels (96 per inch, so 1 pt = 4/3 px).
_PX_PER_PT = 96.0 / 72.0

# Lowest line opacity for a zero-confidence track; full confidence renders opaque.
MIN_CONFIDENCE_OPACITY = 0.35
# Opacity below which a track counts as visibly faded (low confidence) in the key.
LOW_CONFIDENCE_OPACITY = 0.85

# Axis labels (with units) of the operating parameters VANE sweeps.
_PARAMETER_LABELS = {
    "rotor_speed_rpm": "Rotor speed (rpm)",
    "wind_speed": "Wind speed (m/s)",
}


def standard_colorway() -> list[str]:
    """Return the standard line colours as a list of hex strings.

    Returns
    -------
    list[str]
        The seven colours of :data:`STANDARD_COLORS`, for a Plotly ``colorway`` or a
        Matplotlib colour cycle.
    """
    return list(STANDARD_COLORS)


@functools.cache
def _installed_font() -> str | None:
    """Return the first preferred font installed on this system, if any."""
    installed = {entry.name for entry in font_manager.fontManager.ttflist}
    return next((name for name in SANS_SERIF_FONTS if name in installed), None)


def style_rc() -> dict[str, Any]:
    """Return the Matplotlib ``rcParams`` of the VANE style.

    Returns
    -------
    dict[str, Any]
        A new dictionary of ``rcParams`` values: the standard line colours as the
        colour cycle; a Helvetica or Arial font at 10 pt, with 9 pt tick labels and
        legends and 11 pt normal-weight titles; boxed axes with a 0.75 pt soft
        dark-grey frame and inward ticks on all four sides (4 pt major, 2 pt minor);
        1.0 pt lines and 5 pt markers; no grid (a dashed light grey one when switched
        on); a legend with a thin grey border, opaque white fill and square corners;
        and saved figures at 300 dpi, trimmed to their content.

    Notes
    -----
    The font family names the installed preferred font explicitly, followed by the
    generic ``"sans-serif"`` (whose list is Helvetica, Arial, Liberation Sans, DejaVu
    Sans), so a figure keeps its font after a :func:`plot_style` block has closed and
    Matplotlib does not warn about a missing font.
    """
    preferred = _installed_font()
    family = ["sans-serif"] if preferred is None else [preferred, "sans-serif"]
    rc: dict[str, Any] = {
        "font.family": family,
        "font.sans-serif": list(SANS_SERIF_FONTS),
        "mathtext.fontset": "dejavusans",
        "mathtext.default": "regular",
        "font.size": _FONT_SIZE,
        "axes.titlesize": _TITLE_SIZE,
        "axes.titleweight": "normal",
        "axes.labelsize": _FONT_SIZE,
        "axes.labelweight": "normal",
        "xtick.labelsize": _SMALL_FONT_SIZE,
        "ytick.labelsize": _SMALL_FONT_SIZE,
        "legend.fontsize": _SMALL_FONT_SIZE,
        "legend.title_fontsize": _SMALL_FONT_SIZE,
        "figure.titlesize": _TITLE_SIZE,
        "figure.titleweight": "normal",
        "axes.prop_cycle": cycler(color=standard_colorway()),
        "lines.linewidth": _LINE_WIDTH,
        "lines.markersize": _MARKER_SIZE,
        "lines.solid_capstyle": "round",
        "axes.linewidth": _AXES_LINEWIDTH,
        "axes.edgecolor": AXES_COLOR,
        "axes.labelcolor": AXES_COLOR,
        "text.color": AXES_COLOR,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": False,
        "grid.color": GRID_COLOR,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.edgecolor": AXES_COLOR,
        "legend.facecolor": "white",
        "legend.fancybox": False,
        "legend.borderpad": 0.4,
        "axes.formatter.useoffset": False,
        "axes.formatter.use_mathtext": True,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
    for side in ("left", "right", "top", "bottom"):
        rc[f"axes.spines.{side}"] = True
    for axis, sides in (("xtick", ("bottom", "top")), ("ytick", ("left", "right"))):
        rc[f"{axis}.direction"] = "in"
        rc[f"{axis}.color"] = AXES_COLOR
        rc[f"{axis}.major.size"] = 4.0
        rc[f"{axis}.minor.size"] = 2.0
        rc[f"{axis}.major.width"] = _AXES_LINEWIDTH
        rc[f"{axis}.minor.width"] = 0.5
        for side in sides:
            rc[f"{axis}.{side}"] = True
    return rc


@contextlib.contextmanager
def plot_style() -> Iterator[None]:
    """Apply the VANE Matplotlib style inside a ``with`` block.

    The ``rcParams`` of :func:`style_rc` are in effect for the figures and axes
    created inside the block; the previous ``rcParams`` are restored when the block
    exits, so the global configuration is not changed.

    Yields
    ------
    None
        Control returns to the block with the style in effect.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from vane.viz import plot_style
    >>> with plot_style():
    ...     fig, ax = plt.subplots()
    ...     lines = ax.plot([0.0, 1.0], [0.0, 1.0])
    """
    with mpl.rc_context(rc=cast("dict[Any, Any]", style_rc())):
        yield


def _restyle_axes(axes: Axes) -> None:
    """Give an existing axes the box, inward ticks and colour cycle of the style."""
    axes.set_facecolor("white")
    for spine in axes.spines.values():
        spine.set_visible(True)
        spine.set_color(AXES_COLOR)
        spine.set_linewidth(_AXES_LINEWIDTH)
    axes.tick_params(
        which="both",
        direction="in",
        bottom=True,
        top=True,
        left=True,
        right=True,
        color=AXES_COLOR,
        labelcolor=AXES_COLOR,
    )
    axes.tick_params(which="major", length=4.0, width=_AXES_LINEWIDTH)
    axes.tick_params(which="minor", length=2.0, width=0.5)
    axes.set_prop_cycle(color=standard_colorway())


def apply_style(target: Figure | Axes | go.Figure | None = None) -> None:
    """Apply the VANE style permanently, globally or to one figure.

    Parameters
    ----------
    target : matplotlib.figure.Figure, matplotlib.axes.Axes, plotly Figure or None
        What to style. ``None`` (the default) updates the global Matplotlib
        ``rcParams`` with :func:`style_rc`, so every later figure in the session uses
        the style (use :func:`plot_style` for a temporary change instead). A
        Matplotlib figure or axes is restyled in place: white background, box, inward
        ticks on all sides, and the standard colour cycle for lines added afterwards
        (fonts and the colours of existing lines are left as they are). A Plotly
        figure gets the ``"vane"`` template.

    Raises
    ------
    TypeError
        If ``target`` is none of the supported types.
    """
    if target is None:
        mpl.rcParams.update(cast("dict[Any, Any]", style_rc()))
    elif isinstance(target, go.Figure):
        target.update_layout(template=resolve_template(None))
    elif isinstance(target, Figure):
        target.set_facecolor("white")
        for axes in target.axes:
            _restyle_axes(axes)
    elif isinstance(target, Axes):
        _restyle_axes(target)
    else:
        msg = (
            "apply_style expects a Matplotlib Figure or Axes, a Plotly Figure or "
            f"None, got {type(target).__name__}"
        )
        raise TypeError(msg)


@contextlib.contextmanager
def static_style(
    style: str | dict[str, Any] | Path | None,
    size: FigureSize | None = None,
) -> Iterator[None]:
    """Apply a static plot's ``style`` and ``size`` inside a ``with`` block.

    ``"vane"`` selects :func:`plot_style`, ``None`` keeps the caller's current
    ``rcParams``, and anything else is passed to :func:`matplotlib.style.context` (a
    style name, a path or a dictionary of ``rcParams``). A ``size`` then sets the
    figure size and font sizes on top.
    """
    with contextlib.ExitStack() as stack:
        if style is not None:
            if isinstance(style, str) and style == VANE_STYLE:
                stack.enter_context(plot_style())
            else:
                stack.enter_context(matplotlib.style.context(style))
        if size is not None:
            stack.enter_context(mpl.rc_context(rc=cast("dict[Any, Any]", size.rc())))
        yield


def resolve_size(
    size: str | float | FigureSize | None,
    default: str | float | FigureSize,
    *,
    ax: Axes | None,
    style: object,
    aspect: float | tuple[float, float] | None = None,
) -> FigureSize | None:
    """Return the size a static plot applies, or ``None`` to apply none.

    An explicit ``size`` always applies. Without one, the function's ``default`` size
    applies to a new figure in a VANE or named style, but not to the caller's axes or
    when ``style=None`` asks for the caller's own ``rcParams``. ``aspect`` is the
    height-to-width ratio, or a ``(narrow, wide)`` pair used below and above 12 cm.
    """
    chosen = size if size is not None else default
    if size is None and (ax is not None or style is None):
        return None
    if isinstance(aspect, tuple):
        width = figure_size(chosen).width_cm
        aspect = aspect[0] if width < _SHORT_LABEL_CM else aspect[1]
    return figure_size(chosen, aspect=aspect)


def _cartesian_axis() -> dict[str, Any]:
    """Return the 2-D Cartesian axis settings of the Plotly template."""
    return {
        "showline": True,
        "linecolor": AXES_COLOR,
        "linewidth": 1,
        "mirror": "ticks",
        "ticks": "inside",
        "ticklen": 4 * _PX_PER_PT,
        "tickwidth": 1,
        "tickcolor": AXES_COLOR,
        "showgrid": False,
        "gridcolor": GRID_COLOR,
        "griddash": "dash",
        "zeroline": False,
        "automargin": True,
        "title": {"standoff": 8},
    }


def plotly_template() -> go.layout.Template:
    """Build the VANE Plotly template.

    The template is registered at import as ``plotly.io.templates["vane"]``. This
    function returns a new copy, which can be modified and passed as a ``template=``
    argument.

    Returns
    -------
    plotly.graph_objects.layout.Template
        The template: the standard line colours as the ``colorway``; Helvetica or
        Arial at 10 pt (9 pt ticks and legend); white backgrounds; boxed axes with the
        soft dark-grey frame mirrored on all four sides and inward ticks; no grid
        (dashed light grey when switched on); a legend with a thin grey border and an
        opaque white fill; and the same conventions for 3-D scenes, polar plots and
        colour bars.
    """
    small = {"size": _SMALL_FONT_SIZE * _PX_PER_PT}
    scene_axis = {
        "showline": True,
        "linecolor": AXES_COLOR,
        "mirror": True,
        "ticks": "inside",
        "tickcolor": AXES_COLOR,
        "tickfont": small,
        "showgrid": True,
        "gridcolor": "#e0e0e0",
        "showbackground": True,
        "backgroundcolor": "white",
        "zeroline": False,
    }
    polar_axis = {
        "showline": True,
        "linecolor": AXES_COLOR,
        "ticks": "inside",
        "tickcolor": AXES_COLOR,
        "tickfont": small,
        "showgrid": True,
        "gridcolor": "#e0e0e0",
    }
    colorbar = {
        "outlinecolor": AXES_COLOR,
        "outlinewidth": 1,
        "ticks": "inside",
        "tickcolor": AXES_COLOR,
        "tickfont": small,
    }
    axis = _cartesian_axis()
    axis["tickfont"] = small
    layout = go.Layout(
        colorway=standard_colorway(),
        font={
            "family": PLOTLY_FONT_FAMILY,
            "size": _FONT_SIZE * _PX_PER_PT,
            "color": AXES_COLOR,
        },
        title={
            "font": {"size": _TITLE_SIZE * _PX_PER_PT},
            "x": 0.5,
            "xanchor": "center",
        },
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=axis,
        yaxis=dict(axis),
        scene={
            "bgcolor": "white",
            "xaxis": scene_axis,
            "yaxis": scene_axis,
            "zaxis": scene_axis,
        },
        polar={
            "bgcolor": "white",
            "radialaxis": polar_axis,
            "angularaxis": polar_axis,
        },
        legend={
            "bgcolor": "white",
            "bordercolor": AXES_COLOR,
            "borderwidth": 1,
            "font": small,
            "tracegroupgap": 6,
        },
        hoverlabel={"font": {"family": PLOTLY_FONT_FAMILY}},
        coloraxis={"colorbar": colorbar},
    )
    data = {
        "scatter": [go.Scatter(line={"width": 1.5 * _PX_PER_PT}, marker={"size": 6})],
        "heatmap": [go.Heatmap(colorbar=colorbar)],
    }
    return go.layout.Template(layout=layout, data=data)


def resolve_template(
    template: str | go.layout.Template | None,
) -> str | go.layout.Template:
    """Return the template to apply: the caller's, or ``"vane"`` for ``None``."""
    if template is not None:
        return template
    if PLOTLY_TEMPLATE not in pio.templates:
        pio.templates[PLOTLY_TEMPLATE] = plotly_template()
    return PLOTLY_TEMPLATE


pio.templates[PLOTLY_TEMPLATE] = plotly_template()


def parameter_label(parameter_name: str) -> str:
    """Return the axis label, with units, of an operating parameter."""
    return _PARAMETER_LABELS.get(parameter_name, parameter_name)


def operating_range_label(parameter_name: str) -> str:
    """Return the name of the operating window drawn on a plot of this parameter."""
    if parameter_name == "wind_speed":
        return "Operating Wind Range"
    return "Operating Speed Range"


def rc_points(key: str) -> float:
    """Return a font-size ``rcParams`` entry in points (``"medium"`` is resolved)."""
    return float(FontProperties(size=mpl.rcParams[cast("Any", key)]).get_size())


def text_color(color: str, shade: float = 0.8) -> str:
    """Return a darker shade of a colour, for text on a white background."""
    red, green, blue = to_rgb(color)
    return to_hex((shade * red, shade * green, shade * blue))


# --- Per-track encoding of the Campbell and damping plots ---------------------------


class ModeFamily(Enum):
    """Mode family of a tracked mode, which sets its line colour.

    The families are the blade modes (flap, edge, pitch, torsion and axial), tower
    bending (fore-aft and side-side), the platform rigid-body modes (surge, sway,
    heave, roll, pitch, yaw), the drivetrain (rotor/generator azimuth and drivetrain
    torsion), and every other category (nacelle yaw, furl, teeter, unidentified).
    """

    BLADE = "blade"
    TOWER = "tower"
    PLATFORM = "platform"
    DRIVETRAIN = "drivetrain"
    OTHER = "other"


#: Legend title of each mode family.
FAMILY_TITLES: dict[ModeFamily, str] = {
    ModeFamily.BLADE: "Blades",
    ModeFamily.TOWER: "Tower",
    ModeFamily.PLATFORM: "Platform",
    ModeFamily.DRIVETRAIN: "Drivetrain",
    ModeFamily.OTHER: "Other",
}

#: Line colour of each mode family: blades green, tower black, platform red,
#: drivetrain magenta, other modes cyan (the ``nP`` lines are blue).
FAMILY_COLORS: dict[ModeFamily, str] = {
    ModeFamily.BLADE: STANDARD_COLORS[3],
    ModeFamily.TOWER: STANDARD_COLORS[0],
    ModeFamily.PLATFORM: STANDARD_COLORS[1],
    ModeFamily.DRIVETRAIN: STANDARD_COLORS[4],
    ModeFamily.OTHER: STANDARD_COLORS[6],
}

_CATEGORY_FAMILY: dict[DofCategory, ModeFamily] = {
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

# Line styles cycled within a family: (Plotly dash, Matplotlib linestyle).
_DASHES: tuple[tuple[str, str | tuple[float, tuple[float, ...]]], ...] = (
    ("solid", "-"),
    ("dash", "--"),
    ("dashdot", "-."),
    ("dot", ":"),
    ("6px,2px,2px,2px", (0.0, (3.0, 1.0, 1.0, 1.0))),
    ("10px,4px", (0.0, (5.0, 2.0))),
    ("2px,2px", (0.0, (1.0, 1.0))),
)

# Marker of each multi-blade type: (Plotly symbol, Matplotlib marker). The triangles
# point the way the whirling branches slope on a rotor-speed Campbell diagram.
_MULTIBLADE_MARKERS: dict[str | None, tuple[str, str]] = {
    None: ("circle", "o"),
    "collective": ("circle", "o"),
    "regressive": ("triangle-down", "v"),
    "progressive": ("triangle-up", "^"),
    "cyclic": ("square", "s"),
}

# Marker of an ambiguous track or point (Plotly symbol, Matplotlib marker drawn open).
AMBIGUOUS_SYMBOL = "diamond-open"
AMBIGUOUS_MPL_MARKER = "D"

#: Line width of the Campbell and damping tracks (pt).
TRACK_LINEWIDTH = 1.5
#: Line width of the ``nP`` excitation lines (pt).
EXCITATION_LINEWIDTH = 0.8


@dataclass(frozen=True)
class ModeLineStyle:
    """Line style of a tracked mode in the Campbell and damping plots.

    Parameters
    ----------
    family : ModeFamily
        The mode family, which sets the colour.
    color : str
        Hex line and marker colour (the family colour).
    dash : str
        Plotly line dash.
    marker : str
        Plotly marker symbol.
    linestyle : str or tuple
        Matplotlib linestyle equivalent to ``dash``.
    mpl_marker : str
        Matplotlib marker equivalent to ``marker``.
    """

    family: ModeFamily
    color: str
    dash: str
    marker: str
    linestyle: str | tuple[float, tuple[float, ...]]
    mpl_marker: str


def mode_family(category: DofCategory) -> ModeFamily:
    """Return the mode family of a DOF category.

    Parameters
    ----------
    category : DofCategory
        A category from :mod:`vane.config.dof_map`.

    Returns
    -------
    ModeFamily
        The family. Nacelle yaw, rotor and tail furl, teeter and unidentified modes
        are :attr:`ModeFamily.OTHER`.
    """
    return _CATEGORY_FAMILY.get(category, ModeFamily.OTHER)


def mode_line_style(label: ModeLabel, index: int = 0) -> ModeLineStyle:
    """Return the line style of a mode label in the Campbell and damping plots.

    The colour is the family colour (:data:`FAMILY_COLORS`). The line style cycles
    within the family by ``index``, the position of the track among the tracks of its
    family in the figure: solid, dashed, dash-dotted, dotted, then three finer
    patterns. The marker is set by the multi-blade type: a circle for collective and
    non-blade modes, a downward triangle for regressive, an upward triangle for
    progressive and a square for a standing cyclic mode.

    Parameters
    ----------
    label : ModeLabel
        The track's label (``track.label``).
    index : int, optional
        Position of the track within its family (default ``0``, a solid line).

    Returns
    -------
    ModeLineStyle
        Colour, dash and marker for Plotly and Matplotlib.
    """
    family = mode_family(label.category)
    plotly_dash, linestyle = _DASHES[index % len(_DASHES)]
    symbol, mpl_marker = _MULTIBLADE_MARKERS.get(
        label.multiblade, _MULTIBLADE_MARKERS[None]
    )
    return ModeLineStyle(
        family=family,
        color=FAMILY_COLORS[family],
        dash=plotly_dash,
        marker=symbol,
        linestyle=linestyle,
        mpl_marker=mpl_marker,
    )


def track_styles(tracks: Sequence[ModeTrack]) -> list[ModeLineStyle]:
    """Return the style of each track, cycling the line style within each family."""
    counts: dict[ModeFamily, int] = {}
    styles = []
    for track in tracks:
        family = mode_family(track.label.category)
        index = counts.get(family, 0)
        counts[family] = index + 1
        styles.append(mode_line_style(track.label, index))
    return styles


def family_rank(family: ModeFamily) -> int:
    """Return the legend rank of a family (families are listed in enum order)."""
    return 100 * (list(ModeFamily).index(family) + 1)


def confidence_opacity(track: ModeTrack) -> float:
    """Map a track's confidence in ``[0, 1]`` to a line opacity."""
    confidence = min(max(track.confidence, 0.0), 1.0)
    return MIN_CONFIDENCE_OPACITY + (1.0 - MIN_CONFIDENCE_OPACITY) * confidence


def track_name(track: ModeTrack, ambiguous: bool) -> str:
    """Return the legend name for a track, flagged when ambiguous."""
    return f"{track.label.label} (ambiguous)" if ambiguous else track.label.label


def ambiguous_points(track: ModeTrack) -> list[bool] | None:
    """Return which points of a track carry an ambiguous label, or ``None`` if none."""
    labels = track.point_labels
    if len(labels) != len(track.operating_points) or not any(
        label.is_ambiguous for label in labels
    ):
        return None
    return [label.is_ambiguous for label in labels]


# Legend ranks of the groups listed after the mode families.
HEAVILY_DAMPED_RANK = 700
EXCITATION_RANK = 800
RESONANCE_RANK = 900


def plotly_legend_group(
    key: str, title: str, rank: int, titled: set[str]
) -> dict[str, Any]:
    """Return a trace's legend-group settings; the group's first trace has the title."""
    settings: dict[str, Any] = {"legendgroup": key, "legendrank": rank}
    if key not in titled:
        titled.add(key)
        settings["legendgrouptitle"] = {"text": title}
    return settings


def plotly_family_legend(family: ModeFamily, titled: set[str]) -> dict[str, Any]:
    """Return the legend-group settings of a trace of a mode family."""
    return plotly_legend_group(
        family.value, FAMILY_TITLES[family], family_rank(family), titled
    )


def plotly_point_symbols(
    track: ModeTrack, symbol: str, ambiguous: bool, show_ambiguity: bool
) -> str | list[str]:
    """Return a track's marker symbol(s): open diamonds where the label is in doubt.

    A whole ambiguous track uses open diamonds throughout; otherwise each point whose
    own label is ambiguous (a hybrid point of a clear line) gets one, and the other
    points keep the track's ``symbol``.
    """
    if ambiguous:
        return AMBIGUOUS_SYMBOL
    flags = ambiguous_points(track) if show_ambiguity else None
    if flags is None:
        return symbol
    return [AMBIGUOUS_SYMBOL if flag else symbol for flag in flags]


def new_axes(ax: Axes | None) -> tuple[Figure, Axes]:
    """Return the figure and axes to draw into: the caller's, or a new figure.

    A new figure takes its size from the ``rcParams`` in effect and uses the
    constrained layout, so labels and legends outside the axes get their own room.
    """
    if ax is not None:
        return cast("Figure", ax.get_figure()), ax
    figure, axes = plt.subplots(layout="constrained")
    return figure, axes


# Legend entries per column of the per-track legends, and the length of a legend line
# (in font sizes), long enough to show the dash pattern.
_LEGEND_ROWS = 18
_HANDLE_LENGTH = 2.5


class LegendGroups:
    """Legend entries collected by group and drawn as one legend in group order."""

    def __init__(self) -> None:
        self._groups: dict[str, tuple[int, str, list[tuple[Artist, str]]]] = {}

    def add(self, key: str, title: str, rank: int, handle: Artist, label: str) -> None:
        """Add one entry to a group, creating the group on first use."""
        self._groups.setdefault(key, (rank, title, []))[2].append((handle, label))

    def add_family(self, family: ModeFamily, handle: Artist, label: str) -> None:
        """Add one entry to a mode family's group."""
        self.add(
            family.value, FAMILY_TITLES[family], family_rank(family), handle, label
        )

    def entries(self) -> tuple[list[Artist], list[str], list[int]]:
        """Return the handles, the labels and the positions of the group titles.

        Groups are ordered by rank; each starts with a title entry (an empty handle).
        """
        handles: list[Artist] = []
        labels: list[str] = []
        headers: list[int] = []
        for _, title, items in sorted(self._groups.values(), key=lambda g: g[0]):
            headers.append(len(handles))
            handles.append(Line2D([], [], linestyle="none", marker=""))
            labels.append(title)
            for handle, label in items:
                handles.append(handle)
                labels.append(label)
        return handles, labels, headers

    def draw(self, axes: Axes, *, own_figure: bool) -> Legend | None:
        """Draw one legend with a titled block per group, to the right of the axes.

        When VANE created the figure (``own_figure``) it is a figure legend outside
        the axes, for which the constrained layout makes room; in a caller's axes it
        is anchored just outside the axes' right edge. Returns the legend, or ``None``
        when there are no entries.
        """
        handles, labels, headers = self.entries()
        if not handles:
            return None
        ncols = -(-len(handles) // _LEGEND_ROWS)
        figure = cast("Figure", axes.get_figure())
        if own_figure:
            legend = figure.legend(
                handles,
                labels,
                loc="outside right upper",
                ncols=ncols,
                handlelength=_HANDLE_LENGTH,
            )
        else:
            legend = axes.legend(
                handles,
                labels,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                ncols=ncols,
                handlelength=_HANDLE_LENGTH,
            )
        texts = legend.get_texts()
        for index in headers:
            texts[index].set_fontweight("bold")
        return legend


_MULTIBLADE_NAMES = {
    "collective": "Collective",
    "regressive": "Regressive",
    "progressive": "Progressive",
    "cyclic": "Cyclic",
}


class EncodingKey:
    """The compact legend of a Campbell or damping plot.

    Instead of one entry per track (the tracks carry direct labels), the key lists
    the mode families present (colour), the ``nP`` lines ("Blade Passing"), the
    multi-blade types present (marker), and the overlays used: ambiguity, low
    confidence, heavily damped tracks and resonance or instability markers.
    """

    def __init__(self) -> None:
        self.families: list[ModeFamily] = []
        self.multiblade: list[str] = []
        self.ambiguous = False
        self.low_confidence = False
        self.heavily_damped = False
        self.excitation = False
        self.markers: list[tuple[str, str]] = []

    def add_track(
        self, label: ModeLabel, *, ambiguous: bool, low_confidence: bool
    ) -> None:
        """Record the encoding a track uses."""
        family = mode_family(label.category)
        if family not in self.families:
            self.families.append(family)
        kind = label.multiblade
        if kind in _MULTIBLADE_NAMES and kind not in self.multiblade:
            self.multiblade.append(kind)
        self.ambiguous = self.ambiguous or ambiguous
        self.low_confidence = self.low_confidence or low_confidence

    def merge(self, other: EncodingKey) -> None:
        """Add the entries of another key (for a figure with several panels)."""
        for family in other.families:
            if family not in self.families:
                self.families.append(family)
        for kind in other.multiblade:
            if kind not in self.multiblade:
                self.multiblade.append(kind)
        self.ambiguous = self.ambiguous or other.ambiguous
        self.low_confidence = self.low_confidence or other.low_confidence
        self.heavily_damped = self.heavily_damped or other.heavily_damped
        self.excitation = self.excitation or other.excitation
        for entry in other.markers:
            if entry not in self.markers:
                self.markers.append(entry)

    def entries(self) -> tuple[list[Artist], list[str]]:
        """Return the legend handles and labels, families first in family order."""
        width = 2.0
        handles: list[Artist] = []
        labels: list[str] = []
        for family in ModeFamily:
            if family in self.families:
                handles.append(
                    Line2D([], [], color=FAMILY_COLORS[family], linewidth=width)
                )
                labels.append(FAMILY_TITLES[family])
        if self.excitation:
            handles.append(Line2D([], [], color=EXCITATION_COLOR, linewidth=width))
            labels.append("Blade Passing")
        kinds = [kind for kind in _MULTIBLADE_NAMES if kind in self.multiblade]
        if any(kind != "collective" for kind in kinds):
            for kind in kinds:
                marker = _MULTIBLADE_MARKERS[kind][1]
                handles.append(
                    Line2D([], [], linestyle="none", marker=marker, color=AXES_COLOR)
                )
                labels.append(_MULTIBLADE_NAMES[kind])
        if self.ambiguous:
            handles.append(
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker=AMBIGUOUS_MPL_MARKER,
                    markerfacecolor="none",
                    color=AXES_COLOR,
                )
            )
            labels.append("Ambiguous")
        if self.low_confidence:
            handles.append(Line2D([], [], color=AXES_COLOR, linewidth=width, alpha=0.4))
            labels.append("Low confidence")
        if self.heavily_damped:
            handles.append(
                Line2D(
                    [],
                    [],
                    color=REFERENCE_COLOR,
                    linestyle=":",
                    marker="o",
                    markerfacecolor="none",
                )
            )
            labels.append("Heavily damped")
        resonance = [(n, c) for n, c in self.markers if n.startswith("Resonance (")]
        for name, color in self.markers:
            if (name, color) in resonance:
                continue
            handles.append(_cross(color))
            labels.append(name)
        if resonance:
            severities = ", ".join(
                name[len("Resonance (") : -1] for name, _ in resonance
            )
            handles.append(tuple(_cross(color) for _, color in resonance))  # type: ignore[arg-type]
            labels.append(
                "Resonance" if len(resonance) > 1 else f"Resonance ({severities})"
            )
        return handles, labels

    def draw(self, axes: Axes, *, own_figure: bool) -> Legend | None:
        """Draw the key above the plot, in as many columns as the width allows.

        When VANE created the figure (``own_figure``) the key is a figure legend above
        the axes, for which the constrained layout makes room; in a caller's axes it
        sits just above the axes. Returns the legend, or ``None`` when there is
        nothing to explain.
        """
        handles, labels = self.entries()
        if not handles:
            return None
        figure = cast("Figure", axes.get_figure())
        width_px = float(figure.get_size_inches()[0]) * figure.dpi
        legend: Legend | None = None
        # Use as many columns as fit across the figure (at most four).
        for ncols in range(min(len(handles), _KEY_MAX_COLUMNS), 0, -1):
            if legend is not None:
                legend.remove()
            options: dict[str, Any] = {
                "ncols": ncols,
                "handlelength": _HANDLE_LENGTH,
                "columnspacing": 1.0,
                "handler_map": {tuple: HandlerTuple(ndivide=None, pad=0.6)},
            }
            if own_figure:
                legend = figure.legend(
                    handles, labels, loc="outside upper center", **options
                )
            else:
                legend = axes.legend(
                    handles,
                    labels,
                    loc="lower left",
                    bbox_to_anchor=(0.0, 1.02),
                    borderaxespad=0.0,
                    **options,
                )
            if legend.get_window_extent().width <= 0.98 * width_px:
                break
        return legend


def _cross(color: str) -> Line2D:
    """Return the legend handle of a filled-cross marker of a colour."""
    return Line2D(
        [],
        [],
        linestyle="none",
        marker="X",
        markersize=7.0,
        markeredgecolor=AXES_COLOR,
        markeredgewidth=0.5,
        color=color,
    )


# Most columns of the compact key.
_KEY_MAX_COLUMNS = 4


# Abbreviations used by the direct labels of narrow figures.
_SHORT_NAMES = (
    (" (Collective)", " coll."),
    (" (Regressive)", " reg."),
    (" (Progressive)", " prog."),
    (" (Cyclic)", " cyc."),
    ("Blade Flap", "flap"),
    ("Blade Edge", "edge"),
    ("Blade Torsion", "torsion"),
    ("Tower Fore-Aft", "tower FA"),
    ("Tower Side-Side", "tower SS"),
    ("Drivetrain Torsion", "drivetrain"),
    ("Rotor/Generator", "rotor"),
    ("Platform ", ""),
)

# Below this figure width (cm) the direct labels use the short mode names.
_SHORT_LABEL_CM = 12.0


def short_mode_name(name: str) -> str:
    """Return the short form of a mode label used for direct labels in narrow figures.

    ``"1st Blade Flap (Regressive)"`` becomes ``"1st flap reg."``,
    ``"1st Tower Side-Side"`` becomes ``"1st tower SS"`` and ``"Platform Pitch"``
    becomes ``"Pitch"``.
    """
    for long, short in _SHORT_NAMES:
        name = name.replace(long, short)
    return name.strip()


def direct_label_text(axes: Axes, name: str) -> str:
    """Return a mode name for a direct label: shortened in a narrow figure."""
    figure = cast("Figure", axes.get_figure())
    width_cm = float(figure.get_size_inches()[0]) * 2.54
    return short_mode_name(name) if width_cm < _SHORT_LABEL_CM else name


# Vertical pitch of the direct labels, in font sizes, and their offset from the axes.
_LABEL_PITCH = 1.25
_LABEL_OFFSET_PT = 6.0


def _spread(targets: list[float], gap: float, low: float, high: float) -> list[float]:
    """Return label positions near ``targets``, at least ``gap`` apart, within bounds.

    The labels keep their order; when they do not fit between ``low`` and ``high``
    they are spaced evenly over that range.
    """
    order = sorted(range(len(targets)), key=targets.__getitem__)
    positions = [min(max(targets[i], low), high) for i in order]
    for k in range(1, len(positions)):
        positions[k] = max(positions[k], positions[k - 1] + gap)
    if positions and positions[-1] > high:
        positions[-1] = high
        for k in range(len(positions) - 2, -1, -1):
            positions[k] = min(positions[k], positions[k + 1] - gap)
    if positions and positions[0] < low:
        step = (high - low) / max(len(positions) - 1, 1)
        positions = [low + k * step for k in range(len(positions))]
    result = [0.0] * len(targets)
    for rank, index in enumerate(order):
        result[index] = positions[rank]
    return result


def label_line_ends(
    axes: Axes,
    ends: Sequence[tuple[float, float, str, str]],
    *,
    inside: bool = False,
    grow: bool = False,
) -> list[Annotation]:
    """Name lines at their right ends without overlaps.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        The axes; its data limits must be final.
    ends : Sequence[tuple[float, float, str, str]]
        ``(x, y, text, colour)`` of each line end, in data coordinates.
    inside : bool, optional
        Place the column of names just inside the right edge of the axes, reading
        leftwards (for axes the caller laid out, where text outside the axes could be
        cut off). By default the names form a column just outside the right edge.
    grow : bool, optional
        Make the figure taller when the axes are too short to stack all the names
        without overlap (for a figure the plot function created).

    Returns
    -------
    list[matplotlib.text.Annotation]
        The labels, each at the height of its line end where there is room and moved
        vertically just enough to clear its neighbours otherwise, with a thin leader
        line back to the height of the line.
    """
    if not ends:
        return []
    figure = cast("Figure", axes.get_figure())
    size = rc_points("legend.fontsize") - 1.0
    offset = -_LABEL_OFFSET_PT if inside else _LABEL_OFFSET_PT
    if grow:
        figure.draw_without_rendering()
        height_pt = axes.get_window_extent().height * 72.0 / figure.dpi
        needed = (len(ends) + 0.5) * _LABEL_PITCH * size
        if needed > height_pt:
            width, height = figure.get_size_inches()
            figure.set_size_inches(width, height + (needed - height_pt) / 72.0)
    labels = [
        axes.annotate(
            text,
            xy=(1.0, 0.5),
            xycoords="axes fraction",
            xytext=(offset, 0.0),
            textcoords="offset points",
            ha="right" if inside else "left",
            va="center",
            fontsize=size,
            color=color,
            annotation_clip=False,
            zorder=5,
            arrowprops={
                "arrowstyle": "-",
                "linewidth": 0.6,
                "color": color,
                "alpha": 0.6,
                "shrinkA": 1.0,
                "shrinkB": 0.0,
            },
        )
        for _, _, text, color in ends
    ]
    # Place, let the layout make room for the labels, then place again at the final
    # axes height.
    for _ in range(2):
        _place_end_labels(axes, labels, ends, size, offset)
        figure.draw_without_rendering()
    _place_end_labels(axes, labels, ends, size, offset)
    return labels


def _place_end_labels(
    axes: Axes,
    labels: list[Annotation],
    ends: Sequence[tuple[float, float, str, str]],
    size: float,
    offset: float,
) -> None:
    """Position the direct labels for the current axes height."""
    figure = cast("Figure", axes.get_figure())
    to_axes = axes.transData + axes.transAxes.inverted()
    height_pt = axes.get_window_extent().height * 72.0 / figure.dpi
    fractions = [
        min(max(float(to_axes.transform((x, y))[1]), 0.0), 1.0) for x, y, _, _ in ends
    ]
    targets = [fraction * height_pt for fraction in fractions]
    gap = _LABEL_PITCH * size
    placed = _spread(targets, gap, 0.5 * gap, height_pt - 0.5 * gap)
    for label, fraction, target, position in zip(
        labels, fractions, targets, placed, strict=True
    ):
        label.xy = (1.0, fraction)
        label.set_position((offset, position - target))


def excitation_tag_positions(
    harmonics: Sequence[int],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    obstacles: Sequence[tuple[Sequence[float], Sequence[float]]] = (),
) -> list[tuple[int, float, float]]:
    """Return where to tag each ``nP`` line inline, clear of the data.

    Each line ``f = n x / 60`` is sampled along the part of it that lies inside the
    plot, between 20 % and 90 % of that length; the tag goes to the sample farthest
    from the ``obstacles`` (the track polylines, in data coordinates) and from the
    tags already placed, distances measured in plot-fraction units.

    Parameters
    ----------
    harmonics : Sequence[int]
        The excitation orders; non-positive orders are skipped.
    x_range, y_range : tuple[float, float]
        The plot limits.
    obstacles : Sequence[tuple[Sequence[float], Sequence[float]]], optional
        ``(x, y)`` polylines the tags should keep clear of.

    Returns
    -------
    list[tuple[int, float, float]]
        ``(order, x, y)`` of each tag, for the lines that enter the plot.
    """
    x_low, x_high = x_range
    y_low, y_high = y_range
    width = (x_high - x_low) or 1.0
    height = (y_high - y_low) or 1.0
    points: list[tuple[float, float]] = []
    for xs, ys in obstacles:
        x_arr = np.asarray(xs, dtype=np.float64)
        y_arr = np.asarray(ys, dtype=np.float64)
        keep = np.isfinite(x_arr) & np.isfinite(y_arr)
        x_arr, y_arr = x_arr[keep], y_arr[keep]
        if x_arr.size == 0:
            continue
        if x_arr.size > 1:
            dense = np.linspace(0.0, x_arr.size - 1.0, 12 * (x_arr.size - 1) + 1)
            index = np.arange(x_arr.size, dtype=np.float64)
            x_arr = np.interp(dense, index, x_arr)
            y_arr = np.interp(dense, index, y_arr)
        points.extend(
            zip(
                ((x_arr - x_low) / width).tolist(),
                ((y_arr - y_low) / height).tolist(),
                strict=True,
            )
        )
    placed: list[tuple[int, float, float]] = []
    for order in harmonics:
        if order <= 0:
            continue
        start = max(x_low, y_low * 60.0 / order)
        end = min(x_high, y_high * 60.0 / order)
        if not end > start:
            continue
        best: tuple[float, float, float] | None = None
        for fraction in np.linspace(0.2, 0.9, 15).tolist():
            x = start + fraction * (end - start)
            y = order * x / 60.0
            u, v = (x - x_low) / width, (y - y_low) / height
            others = points + [
                ((px - x_low) / width, (py - y_low) / height) for _, px, py in placed
            ]
            clearance = min(
                (float(np.hypot(u - pu, v - pv)) for pu, pv in others), default=1.0
            )
            if best is None or clearance > best[0] + 1e-9:
                best = (clearance, x, y)
        if best is not None:
            placed.append((order, best[1], best[2]))
    return placed


def label_excitation_lines(
    axes: Axes,
    harmonics: Sequence[int],
    *,
    obstacles: Sequence[tuple[Sequence[float], Sequence[float]]] = (),
    color: str = EXCITATION_COLOR,
) -> list[Annotation]:
    """Tag the ``nP`` lines inline (``3P``) where they are clear of the data.

    The axes limits must be final. The positions come from
    :func:`excitation_tag_positions`; a white backing keeps the line from running
    through the text.
    """
    size = rc_points("legend.fontsize")
    return [
        axes.annotate(
            f"{order}P",
            xy=(x, y),
            ha="center",
            va="center",
            fontsize=size,
            color=color,
            zorder=4,
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
            },
        )
        for order, x, y in excitation_tag_positions(
            harmonics, axes.get_xlim(), axes.get_ylim(), obstacles
        )
    ]


def mark_operating_range(axes: Axes, window: tuple[float, float], label: str) -> None:
    """Shade an operating window grey and mark it with a labelled double arrow.

    The arrow and its label sit near the top of the axes; the caller leaves room for
    them above the data.
    """
    low, high = sorted(float(value) for value in window)
    x_low, x_high = axes.get_xlim()
    low, high = max(low, x_low), min(high, x_high)
    if not high > low:
        return
    axes.axvspan(low, high, color=OPERATING_RANGE_COLOR, linewidth=0, zorder=0)
    figure = cast("Figure", axes.get_figure())
    size = rc_points("legend.fontsize")
    # The arrow sits 1.9 font sizes below the top of the axes, its label just above.
    shifted = offset_copy(
        axes.get_xaxis_transform(), fig=figure, y=-1.9 * size, units="points"
    )
    axes.annotate(
        "",
        xy=(high, 1.0),
        xytext=(low, 1.0),
        xycoords=shifted,
        textcoords=shifted,
        arrowprops={"arrowstyle": "<->", "color": "0.3", "linewidth": 1.0},
        zorder=5,
    )
    # Keep the label inside the axes: centred over the window, or aligned with its
    # inner end when the window touches either side of the axes.
    span = x_high - x_low
    if high > x_low + 0.7 * span:
        anchor, align, nudge = high, "right", -2.0
    elif low < x_low + 0.3 * span:
        anchor, align, nudge = low, "left", 2.0
    else:
        anchor, align, nudge = 0.5 * (low + high), "center", 0.0
    axes.annotate(
        label,
        xy=(anchor, 1.0),
        xycoords=shifted,
        xytext=(nudge, 0.3 * size),
        textcoords="offset points",
        ha=align,
        va="bottom",
        fontsize=size,
        color="0.2",
        zorder=5,
    )


# --- Degree-of-freedom names and series colours of the mode-shape views ------------

_DOF_NAMES = {
    "Sg": "Surge",
    "Sw": "Sway",
    "Hv": "Heave",
    "R": "Roll",
    "P": "Pitch",
    "Y": "Platform yaw",
    "TFA1": "Tower FA1",
    "TSS1": "Tower SS1",
    "TFA2": "Tower FA2",
    "TSS2": "Tower SS2",
    "Yaw": "Nacelle yaw",
    "GeAz": "Rotor azimuth",
    "DrTr": "Drivetrain",
    "RFrl": "Rotor furl",
    "TFrl": "Tail furl",
    "Teet": "Teeter",
}
_BLADE_DOF_RE = re.compile(r"DOF_B([FEP])\((\d+)(?:,(\d+))?\)")
_FIXED_DOF_RE = re.compile(r"DOF_(\w+)")
_BLADE_DOF_KIND = {"F": "flap", "E": "edge", "P": "pitch"}
_MBC_SHORT = {"collective": "coll.", "cosine": "cos.", "sine": "sin."}
# Longest name kept in full when a description carries no known DOF token.
_MAX_DOF_NAME = 28


def short_dof_name(description: str, coordinate: str = "") -> str:
    """Return a short name of a degree of freedom for tick labels and legends.

    ElastoDyn descriptions are named after their ``DOF_`` token (``"Tower FA1"``,
    ``"Surge"``, ``"Blade 2 flap 1"``); after the MBC transform a blade DOF is named by
    its multi-blade coordinate instead of its blade (``"Flap 1 coll."``). Other
    descriptions are shortened to their first characters.

    Parameters
    ----------
    description : str
        The state description from the linearization file.
    coordinate : str, optional
        The multi-blade coordinate of the DOF (``"collective"``, ``"cosine"`` or
        ``"sine"``), or ``""``.

    Returns
    -------
    str
        The short name.
    """
    blade = _BLADE_DOF_RE.search(description)
    if blade is not None:
        kind = _BLADE_DOF_KIND[blade.group(1)]
        order = f" {blade.group(3)}" if blade.group(3) else ""
        if coordinate in _MBC_SHORT:
            return f"{kind.capitalize()}{order} {_MBC_SHORT[coordinate]}"
        return f"Blade {blade.group(2)} {kind}{order}"
    fixed = _FIXED_DOF_RE.search(description)
    if fixed is not None and fixed.group(1) in _DOF_NAMES:
        return _DOF_NAMES[fixed.group(1)]
    text = description.strip()
    if len(text) <= _MAX_DOF_NAME:
        return text
    return text[: _MAX_DOF_NAME - 1].rstrip() + "…"


def series_colors(count: int) -> list[str]:
    """Return ``count`` distinct line colours: the standard colours, then a colour map.

    Up to seven series take the standard line colours in order; more series sample a
    perceptually ordered continuous colour map instead, so that no two share a hue.
    """
    if count <= len(STANDARD_COLORS):
        return list(STANDARD_COLORS[:count])
    cmap = plt.get_cmap("turbo")
    return [to_hex(cmap(index / max(count - 1, 1))) for index in range(count)]
