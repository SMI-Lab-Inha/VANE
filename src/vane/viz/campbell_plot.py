# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Campbell-diagram plotting, interactive (Plotly) and static (Matplotlib).

Renders a :class:`~vane.campbell.builder.CampbellDiagram`: one line per tracked mode
(natural frequency versus the operating parameter), the ``nP`` excitation lines, and
markers at the detected resonance crossings. :func:`plot_campbell` returns an
interactive Plotly figure, :func:`plot_campbell_static` a Matplotlib figure for
publication, and :func:`plot_campbell_panels` a two-panel Matplotlib figure with the
frequencies and the damping ratios side by side.

The figures follow the lab's Campbell conventions. Each track is coloured by its mode
family (blades green, tower black, platform red, drivetrain magenta, other modes cyan)
and the line style cycles within a family; the marker shows the multi-blade type. The
``nP`` lines ("Blade Passing") are thin and blue, tagged inline (``3P``) and kept out
of the legend. An optional operating window is shaded grey and marked with a labelled
double arrow. The static figures name each mode at its line end, in a column at the
right edge with a collision-free vertical placement, and keep the legend to a compact
key of the families and overlays. VANE adds resonance crossings (filled crosses
coloured by severity), confidence (reduced opacity), ambiguity (open diamonds) and
heavily damped tracks (dotted grey), each explained in the key. The interactive figure
has a legend grouped by family and a hover label on every point (mode, operating
point, frequency, damping, confidence).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from vane.campbell.excitation import (
    DEFAULT_HARMONICS,
    ResonanceSeverity,
    find_resonances,
)
from vane.viz._style import (
    AMBIGUOUS_MPL_MARKER,
    AXES_COLOR,
    EXCITATION_COLOR,
    EXCITATION_LINEWIDTH,
    EXCITATION_RANK,
    HEAVILY_DAMPED_RANK,
    LOW_CONFIDENCE_OPACITY,
    OPERATING_RANGE_COLOR,
    RESONANCE_RANK,
    STANDARD_COLORS,
    TRACK_LINEWIDTH,
    VANE_STYLE,
    EncodingKey,
    LegendGroups,
    ambiguous_points,
    confidence_opacity,
    direct_label_text,
    excitation_tag_positions,
    label_excitation_lines,
    label_line_ends,
    mark_operating_range,
    new_axes,
    operating_range_label,
    parameter_label,
    plotly_family_legend,
    plotly_legend_group,
    plotly_point_symbols,
    resolve_size,
    resolve_template,
    static_style,
    text_color,
    track_name,
    track_styles,
)
from vane.viz.damping_plot import draw_damping_static
from vane.viz.publication import add_panel_labels, figure_size

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from vane.campbell.builder import CampbellDiagram
    from vane.campbell.excitation import ResonanceCrossing
    from vane.modal.identifier import ModeTrack
    from vane.viz._style import ModeLineStyle
    from vane.viz.publication import FigureSize

__all__ = [
    "HEAVILY_DAMPED_COLOR",
    "SEVERITY_COLORS",
    "plot_campbell",
    "plot_campbell_panels",
    "plot_campbell_static",
]

#: Line colour of the heavily damped (near-critical) tracks.
HEAVILY_DAMPED_COLOR = "#7f7f7f"

_RPM_PARAMETER = "rotor_speed_rpm"
_FREQUENCY_LABEL = "Frequency (Hz)"
# Headroom above the highest mode: the nP lines run off the top, and the key and the
# operating-range marker sit in the space left above the modes.
_HEADROOM = 1.3
# Narrowest figure (cm) whose two panels go side by side in the "auto" layout.
_ROW_PANELS_CM = 12.0
# Margin around the operating-parameter range, as a fraction of the range.
_X_MARGIN = 0.02

#: Marker colour for each resonance severity (more damped, less alarming): red,
#: orange and grey.
SEVERITY_COLORS: dict[ResonanceSeverity, str] = {
    ResonanceSeverity.HIGH: STANDARD_COLORS[1],
    ResonanceSeverity.MEDIUM: STANDARD_COLORS[5],
    ResonanceSeverity.LOW: "#8c8c8c",
}

# Severities in drawing order: the most alarming is drawn last (on top).
_SEVERITY_ORDER = (
    ResonanceSeverity.LOW,
    ResonanceSeverity.MEDIUM,
    ResonanceSeverity.HIGH,
)


def plot_campbell(
    diagram: CampbellDiagram,
    *,
    harmonics: Sequence[int] = DEFAULT_HARMONICS,
    show_excitation: bool = True,
    show_resonances: bool = True,
    show_confidence: bool = True,
    show_ambiguity: bool = True,
    show_heavily_damped: bool = True,
    operating_range: tuple[float, float] | None = None,
    show_operating_range: bool = True,
    figure: go.Figure | None = None,
    template: str | go.layout.Template | None = None,
    title: str | None = None,
) -> go.Figure:
    """Plot a Campbell diagram as an interactive figure.

    Parameters
    ----------
    diagram : CampbellDiagram
        The diagram to plot.
    harmonics : Sequence[int], optional
        Excitation harmonics for the ``nP`` lines and resonance markers (only used
        for rotor-speed diagrams).
    show_excitation : bool, optional
        Draw the ``nP`` excitation lines, tagged inline ``1P``, ``3P``, ...
        (rotor-speed diagrams only). The legend has one "Blade Passing" entry for
        them.
    show_resonances : bool, optional
        Mark detected resonance crossings, coloured by severity (rotor-speed diagrams
        only).
    show_confidence : bool, optional
        Scale each track's line opacity by its identification confidence.
    show_ambiguity : bool, optional
        Flag ambiguous tracks (open-diamond markers and an annotated legend name) and
        the ambiguous points of otherwise clear tracks (open-diamond markers at those
        points). The hover names each point's own label, with its alternative when
        ambiguous.
    show_heavily_damped : bool, optional
        Draw the diagram's heavily damped tracks: dotted grey lines with open
        markers, named "(heavily damped)", whose hover adds the damped frequency.
        Default ``True``; ``False`` leaves them out.
    operating_range : tuple[float, float] or None, optional
        The ``(min, max)`` operating window of the parameter. Resonance detection is
        restricted to it, and it is shaded grey (see ``show_operating_range``).
    show_operating_range : bool, optional
        Shade the operating window and name it ("Operating Speed Range" or
        "Operating Wind Range"). Default ``True``; it has no effect without
        ``operating_range``.
    figure : plotly.graph_objects.Figure or None, optional
        Draw into this figure instead of creating a new one (for composing subplots or
        overlays). The figure is returned.
    template : str, plotly.graph_objects.layout.Template or None, optional
        Plotly layout template to apply. ``None`` (the default) applies the VANE
        template ``"vane"``; pass another name (for example ``"plotly_white"``) or a
        template object to override it.
    title : str or None, optional
        Figure title. ``None`` (the default) draws none, leaving the caption to the
        document.

    Returns
    -------
    plotly.graph_objects.Figure
        The Campbell-diagram figure, with the legend grouped by mode family.
    """
    figure = go.Figure() if figure is None else figure
    hover = _track_hovertemplate(diagram.parameter_name)
    titled: set[str] = set()
    for track, style in zip(diagram.tracks, track_styles(diagram.tracks), strict=True):
        x, frequency, damping = diagram.track_curve(track)
        ambiguous = show_ambiguity and track.is_ambiguous
        point_names = _point_names(track, frequency.shape[0])
        figure.add_trace(
            go.Scatter(
                x=x,
                y=frequency,
                mode="lines+markers",
                name=track_name(track, ambiguous),
                opacity=confidence_opacity(track) if show_confidence else 1.0,
                line={"color": style.color, "dash": style.dash},
                marker={
                    "symbol": plotly_point_symbols(
                        track, style.marker, ambiguous, show_ambiguity
                    ),
                    "color": style.color,
                },
                customdata=[
                    [float(d), float(track.confidence), name, int(point)]
                    for d, name, point in zip(
                        damping, point_names, track.operating_points, strict=True
                    )
                ],
                hovertemplate=hover,
                **plotly_family_legend(style.family, titled),
            )
        )

    if show_heavily_damped:
        _add_heavily_damped_tracks(figure, diagram, titled)

    top = _frequency_top(diagram, show_heavily_damped)
    span = _x_span(diagram)
    is_rpm = diagram.parameter_name == _RPM_PARAMETER
    if show_excitation and is_rpm and span is not None:
        _add_excitation_lines(figure, diagram, harmonics, titled, span, top)
    if show_resonances and is_rpm:
        _add_resonance_markers(figure, diagram, harmonics, operating_range, titled)
    if show_operating_range and operating_range is not None:
        figure.add_vrect(
            x0=min(operating_range),
            x1=max(operating_range),
            fillcolor=OPERATING_RANGE_COLOR,
            opacity=0.5,
            line_width=0,
            layer="below",
            annotation_text=operating_range_label(diagram.parameter_name),
            annotation_position="top",
        )

    layout: dict[str, Any] = {
        "xaxis_title": parameter_label(diagram.parameter_name),
        "yaxis_title": _FREQUENCY_LABEL,
        "legend": {"traceorder": "grouped"},
        "template": resolve_template(template),
    }
    if title is not None:
        layout["title"] = title
    if top is not None:
        layout["yaxis_range"] = [0.0, top]
    if span is not None:
        layout["xaxis_range"] = list(span)
    figure.update_layout(**layout)
    return figure


def plot_campbell_static(
    diagram: CampbellDiagram,
    *,
    harmonics: Sequence[int] = DEFAULT_HARMONICS,
    show_excitation: bool = True,
    show_resonances: bool = True,
    show_confidence: bool = True,
    show_ambiguity: bool = True,
    show_heavily_damped: bool = True,
    operating_range: tuple[float, float] | None = None,
    show_operating_range: bool = True,
    ax: Axes | None = None,
    style: str | dict[str, Any] | Path | None = VANE_STYLE,
    size: str | float | FigureSize | None = None,
    direct_labels: bool = True,
    title: str | None = None,
) -> Figure:
    """Plot a Campbell diagram as a static Matplotlib figure for publication.

    The static counterpart of :func:`plot_campbell`, with the same encoding. Each mode
    is named at its line end, in a column just right of the axes (inside the right
    edge when drawing into the caller's axes), moved vertically only as far as needed
    to keep the names apart, with a thin leader back to the line. The frequency axis
    ends 30 % above the highest mode, so the steep ``nP`` lines run off the top; they
    are tagged inline at staggered heights. A compact key inside the axes explains
    the families, "Blade Passing" and any overlays used.

    Parameters
    ----------
    diagram : CampbellDiagram
        The diagram to plot.
    harmonics : Sequence[int], optional
        Excitation harmonics for the ``nP`` lines and resonance markers (only used
        for rotor-speed diagrams).
    show_excitation : bool, optional
        Draw the ``nP`` excitation lines (rotor-speed diagrams only).
    show_resonances : bool, optional
        Mark detected resonance crossings, coloured by severity (rotor-speed diagrams
        only).
    show_confidence : bool, optional
        Scale each track's line opacity by its identification confidence.
    show_ambiguity : bool, optional
        Draw ambiguous tracks, and the ambiguous points of clear tracks, with open
        diamonds, and mark ambiguous tracks with ``?`` in their labels.
    show_heavily_damped : bool, optional
        Draw the heavily damped tracks as dotted grey lines with open markers.
    operating_range : tuple[float, float] or None, optional
        The ``(min, max)`` operating window: restricts resonance detection and is
        shaded grey with a labelled double arrow (see ``show_operating_range``).
    show_operating_range : bool, optional
        Shade and mark the operating window. Default ``True``.
    ax : matplotlib.axes.Axes or None, optional
        Draw into these axes instead of creating a new figure. The axes keep their own
        appearance (use :func:`vane.viz.apply_style` to restyle them); the parent
        figure is returned.
    style : str, dict, pathlib.Path or None, optional
        Matplotlib style in effect while drawing. ``"vane"`` (the default) is the VANE
        style (:func:`vane.viz.plot_style`); ``None`` keeps the current ``rcParams``
        (for example a ``plt.style.context`` opened by the caller); any other value is
        passed to :func:`matplotlib.style.context`.
    size : str, float, FigureSize or None, optional
        Figure size and fonts (see :func:`vane.viz.figure_size`): ``"single"``
        (8.5 cm wide), ``"double"`` (17 cm) or a width in cm. ``None`` uses
        ``"double"`` for a new figure. In a figure narrower than 12 cm the direct
        labels use short mode names (``"1st flap reg."``).
    direct_labels : bool, optional
        Name the modes at their line ends with a compact key (the default). ``False``
        draws a legend with one entry per track instead, grouped and titled by family,
        to the right of the axes.
    title : str or None, optional
        Axes title; ``None`` (the default) draws none.

    Returns
    -------
    matplotlib.figure.Figure
        The Campbell-diagram figure.
    """
    resolved = resolve_size(size, "double", ax=ax, style=style)
    with static_style(style, resolved):
        figure, axes = new_axes(ax)
        legend = None if direct_labels else LegendGroups()
        key, ends = draw_campbell_static(
            axes,
            diagram,
            harmonics=harmonics,
            show_excitation=show_excitation,
            show_resonances=show_resonances,
            show_confidence=show_confidence,
            show_ambiguity=show_ambiguity,
            show_heavily_damped=show_heavily_damped,
            operating_range=operating_range if show_operating_range else None,
            resonance_range=operating_range,
            legend=legend,
        )
        if title is not None:
            axes.set_title(title)
        if legend is not None:
            legend.draw(axes, own_figure=ax is None)
        else:
            key.draw(axes, own_figure=ax is None)
            label_line_ends(axes, ends, inside=ax is not None, grow=ax is None)
    return figure


def plot_campbell_panels(
    diagram: CampbellDiagram,
    *,
    harmonics: Sequence[int] = DEFAULT_HARMONICS,
    layout: Literal["auto", "row", "column"] = "auto",
    as_percent: bool = True,
    show_excitation: bool = True,
    show_resonances: bool = True,
    show_confidence: bool = True,
    show_ambiguity: bool = True,
    show_heavily_damped: bool = True,
    show_stability_bands: bool = True,
    caution_damping: float = 0.01,
    operating_range: tuple[float, float] | None = None,
    show_operating_range: bool = True,
    style: str | dict[str, Any] | Path | None = VANE_STYLE,
    size: str | float | FigureSize | None = None,
    direct_labels: bool = True,
) -> Figure:
    """Plot frequencies and damping ratios of a Campbell diagram as two panels.

    Panel (a) is the Campbell diagram (:func:`plot_campbell_static`) and panel (b) the
    damping ratio (:func:`vane.viz.damping_plot.plot_damping_static`) against the same
    operating parameter. The panels share the operating-parameter axis and its label
    and carry ``(a)`` and ``(b)`` outside their top-left corners; one compact key in
    panel (a) explains the encoding of both. The mode names stand at the right edge:
    of panel (b) in the ``"row"`` layout and of both panels in the ``"column"``
    layout.

    Parameters
    ----------
    diagram : CampbellDiagram
        The diagram to plot.
    harmonics : Sequence[int], optional
        Excitation harmonics for panel (a).
    layout : {"auto", "row", "column"}, optional
        ``"row"`` places the panels side by side; ``"column"`` stacks them, frequency
        on top. ``"auto"`` (the default) uses ``"row"`` for figures at least 12 cm wide
        and ``"column"`` for narrower ones.
    as_percent : bool, optional
        Plot damping in percent (the default) rather than as a fraction.
    show_excitation, show_resonances, show_confidence, show_ambiguity : bool, optional
        As for :func:`plot_campbell_static`.
    show_heavily_damped : bool, optional
        Draw the heavily damped tracks in panel (a).
    show_stability_bands : bool, optional
        Shade the unstable and caution bands in panel (b).
    caution_damping : float, optional
        Upper edge of the caution band as a damping fraction (default ``0.01``).
    operating_range : tuple[float, float] or None, optional
        The operating window: restricts resonance detection and is shaded in both
        panels (see ``show_operating_range``).
    show_operating_range : bool, optional
        Shade the operating window and mark it in panel (a). Default ``True``.
    style : str, dict, pathlib.Path or None, optional
        Matplotlib style, as for :func:`plot_campbell_static`.
    size : str, float, FigureSize or None, optional
        Figure size and fonts; ``None`` uses ``"double"`` (17 cm wide), with a height
        suited to the layout.
    direct_labels : bool, optional
        Name the modes at their line ends (the default); ``False`` draws a per-track
        legend instead.

    Returns
    -------
    matplotlib.figure.Figure
        The two-panel figure.

    Raises
    ------
    ValueError
        If ``layout`` is not ``"auto"``, ``"row"`` or ``"column"``, or
        ``caution_damping`` is negative.
    """
    if layout not in ("auto", "row", "column"):
        msg = f"layout must be 'auto', 'row' or 'column', got {layout!r}"
        raise ValueError(msg)
    if layout == "auto":
        width = figure_size(size if size is not None else "double").width_cm
        row = width >= _ROW_PANELS_CM
    else:
        row = layout == "row"
    resolved = resolve_size(
        size, "double", ax=None, style=style, aspect=0.47 if row else 0.95
    )
    window = operating_range if show_operating_range else None
    with static_style(style, resolved):
        shape = (1, 2) if row else (2, 1)
        figure, grid = plt.subplots(*shape, layout="constrained", sharex=True)
        panels: list[Axes] = list(np.ravel(grid))
        legend = None if direct_labels else LegendGroups()
        key, frequency_ends = draw_campbell_static(
            panels[0],
            diagram,
            harmonics=harmonics,
            show_excitation=show_excitation,
            show_resonances=show_resonances,
            show_confidence=show_confidence,
            show_ambiguity=show_ambiguity,
            show_heavily_damped=show_heavily_damped,
            operating_range=window,
            resonance_range=operating_range,
            legend=legend,
        )
        damping_key, damping_ends = draw_damping_static(
            panels[1],
            diagram,
            as_percent=as_percent,
            show_stability_bands=show_stability_bands,
            show_confidence=show_confidence,
            caution_damping=caution_damping,
            legend=None,
        )
        if window is not None:
            panels[1].axvspan(
                min(window),
                max(window),
                color=OPERATING_RANGE_COLOR,
                linewidth=0,
                zorder=0,
            )
        key.merge(damping_key)
        x_label = parameter_label(diagram.parameter_name)
        for panel in panels:
            panel.set_xlabel("")
        if row:
            figure.supxlabel(x_label, fontsize=plt.rcParams["axes.labelsize"])
        else:
            panels[1].set_xlabel(x_label)
        add_panel_labels(panels)
        if legend is not None:
            legend.draw(panels[-1], own_figure=True)
        else:
            key.draw(panels[0], own_figure=True)
            if not row:
                label_line_ends(panels[0], frequency_ends, grow=True)
            label_line_ends(panels[1], damping_ends, grow=True)
    return figure


def draw_campbell_static(
    axes: Axes,
    diagram: CampbellDiagram,
    *,
    harmonics: Sequence[int],
    show_excitation: bool,
    show_resonances: bool,
    show_confidence: bool,
    show_ambiguity: bool,
    show_heavily_damped: bool,
    operating_range: tuple[float, float] | None,
    resonance_range: tuple[float, float] | None,
    legend: LegendGroups | None,
) -> tuple[EncodingKey, list[tuple[float, float, str, str]]]:
    """Draw a Campbell diagram on Matplotlib axes.

    ``operating_range`` is the window to shade (``None`` for none) and
    ``resonance_range`` the window of the resonance search. Returns the encoding key
    and the line ends ``(x, y, text, colour)`` for the direct labels; when ``legend``
    is given, every track is also added to it.
    """
    key = EncodingKey()
    ends: list[tuple[float, float, str, str]] = []
    for track, style in zip(diagram.tracks, track_styles(diagram.tracks), strict=True):
        end = _draw_static_track(
            axes, key, legend, diagram, track, style, show_confidence, show_ambiguity
        )
        if end is not None:
            ends.append(end)
    if show_heavily_damped:
        for track in diagram.heavily_damped_tracks:
            x, frequency, _ = diagram.track_curve(track)
            (line,) = axes.plot(
                x,
                frequency,
                color=HEAVILY_DAMPED_COLOR,
                linestyle=":",
                linewidth=TRACK_LINEWIDTH,
                marker="o",
                markerfacecolor="none",
            )
            key.heavily_damped = True
            if legend is not None:
                legend.add(
                    "heavily damped",
                    "Heavily damped",
                    HEAVILY_DAMPED_RANK,
                    line,
                    f"{track.label.label} (heavily damped)",
                )
    span = _x_span(diagram)
    if span is not None:
        axes.set_xlim(*span)
    top = _frequency_top(diagram, show_heavily_damped)
    if top is not None:
        axes.set_ylim(0.0, top)
    is_rpm = diagram.parameter_name == _RPM_PARAMETER
    if show_excitation and is_rpm and span is not None:
        ray = np.linspace(span[0], span[1], 64)
        for harmonic in harmonics:
            (line,) = axes.plot(
                ray,
                harmonic * ray / 60.0,
                color=EXCITATION_COLOR,
                linewidth=EXCITATION_LINEWIDTH,
                zorder=1,
            )
            if legend is not None:
                legend.add(
                    "excitation", "Blade Passing", EXCITATION_RANK, line, f"{harmonic}P"
                )
        key.excitation = True
        label_excitation_lines(axes, harmonics, obstacles=_track_lines(diagram))
    if show_resonances and is_rpm:
        crossings = find_resonances(diagram, harmonics, operating_range=resonance_range)
        _draw_static_resonances(axes, key, legend, crossings)
    if operating_range is not None:
        mark_operating_range(
            axes, operating_range, operating_range_label(diagram.parameter_name)
        )
    axes.set_xlabel(parameter_label(diagram.parameter_name))
    axes.set_ylabel(_FREQUENCY_LABEL)
    return key, ends


def _draw_static_track(
    axes: Axes,
    key: EncodingKey,
    legend: LegendGroups | None,
    diagram: CampbellDiagram,
    track: ModeTrack,
    style: ModeLineStyle,
    show_confidence: bool,
    show_ambiguity: bool,
) -> tuple[float, float, str, str] | None:
    """Draw one track with its ambiguous points marked; return its labelled end."""
    x, frequency, _ = diagram.track_curve(track)
    ambiguous = show_ambiguity and track.is_ambiguous
    alpha = confidence_opacity(track) if show_confidence else 1.0
    flags = ambiguous_points(track) if show_ambiguity and not ambiguous else None
    clear = None if flags is None else [i for i, flag in enumerate(flags) if not flag]
    (line,) = axes.plot(
        x,
        frequency,
        color=style.color,
        linestyle=style.linestyle,
        linewidth=TRACK_LINEWIDTH,
        marker=AMBIGUOUS_MPL_MARKER if ambiguous else style.mpl_marker,
        markerfacecolor="none" if ambiguous else style.color,
        markevery=clear,
        alpha=alpha,
        zorder=3,
    )
    if flags is not None:
        mask = np.asarray(flags, dtype=np.bool_)
        axes.plot(
            x[mask],
            frequency[mask],
            linestyle="none",
            marker=AMBIGUOUS_MPL_MARKER,
            markerfacecolor="none",
            markeredgecolor=style.color,
            alpha=alpha,
            zorder=3,
        )
    key.add_track(
        track.label,
        ambiguous=ambiguous or flags is not None,
        low_confidence=alpha < LOW_CONFIDENCE_OPACITY,
    )
    if legend is not None:
        legend.add_family(style.family, line, track_name(track, ambiguous))
    finite = np.flatnonzero(np.isfinite(frequency))
    if finite.size == 0:
        return None
    last = int(finite[-1])
    text = direct_label_text(axes, track.label.label) + (" (?)" if ambiguous else "")
    return float(x[last]), float(frequency[last]), text, text_color(style.color)


def _x_span(diagram: CampbellDiagram) -> tuple[float, float] | None:
    """Return the operating-parameter axis range, with a small margin."""
    values = np.asarray(diagram.parameter_values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    low, high = float(values.min()), float(values.max())
    margin = _X_MARGIN * (high - low) if high > low else 0.5
    return low - margin, high + margin


def _frequency_top(diagram: CampbellDiagram, include_heavy: bool) -> float | None:
    """Return the top of the frequency axis: 30 % above the highest mode."""
    tracks = list(diagram.tracks)
    if include_heavy:
        tracks.extend(diagram.heavily_damped_tracks)
    values = [
        float(value)
        for track in tracks
        for value in track.natural_frequencies_hz
        if np.isfinite(value)
    ]
    if not values or max(values) <= 0.0:
        return None
    return _HEADROOM * max(values)


def _draw_static_resonances(
    axes: Axes,
    key: EncodingKey,
    legend: LegendGroups | None,
    crossings: list[ResonanceCrossing],
) -> None:
    """Draw the resonance crossings as filled crosses, one set per severity."""
    by_severity = _group_by_severity(crossings)
    for severity in _SEVERITY_ORDER:
        group = by_severity.get(severity)
        if not group:
            continue
        name = f"Resonance ({severity.value})"
        (markers,) = axes.plot(
            [crossing.rotor_speed_rpm for crossing in group],
            [crossing.frequency_hz for crossing in group],
            linestyle="none",
            marker="X",
            markersize=7.0,
            markeredgecolor=AXES_COLOR,
            markeredgewidth=0.5,
            color=SEVERITY_COLORS[severity],
            zorder=4,
        )
        key.markers.append((name, SEVERITY_COLORS[severity]))
        if legend is not None:
            legend.add("resonance", "Resonance", RESONANCE_RANK, markers, name)


def _group_by_severity(
    crossings: list[ResonanceCrossing],
) -> dict[ResonanceSeverity, list[ResonanceCrossing]]:
    """Group resonance crossings by severity."""
    by_severity: dict[ResonanceSeverity, list[ResonanceCrossing]] = {}
    for crossing in crossings:
        by_severity.setdefault(crossing.severity, []).append(crossing)
    return by_severity


def _track_hovertemplate(parameter_name: str) -> str:
    """Hover template naming the mode, operating point, frequency and damping."""
    return (
        "<b>%{fullData.name}</b><br>"
        f"{parameter_label(parameter_name)}: %{{x:.4g}}<br>"
        "Operating point: %{customdata[3]}<br>"
        "Frequency: %{y:.4g} Hz<br>"
        "Damping ratio: %{customdata[0]:.2%}<br>"
        "Confidence: %{customdata[1]:.2f}<br>%{customdata[2]}<extra></extra>"
    )


def _point_names(track: ModeTrack, n_points: int) -> list[str]:
    """Return the hover text naming the track's mode at each point.

    An ambiguous point label is shown with its alternative; a point without a label
    (no DOF descriptions) shows nothing.
    """
    labels = track.point_labels
    if len(labels) != n_points:
        return [""] * n_points
    return [
        f"point: {label.label} or {label.alternative} (ambiguous)"
        if label.is_ambiguous
        else f"point: {label.label}"
        for label in labels
    ]


def _add_heavily_damped_tracks(
    figure: go.Figure, diagram: CampbellDiagram, titled: set[str]
) -> None:
    """Add the heavily damped tracks, styled apart from the lightly damped lines."""
    hover = (
        "<b>%{fullData.name}</b><br>"
        f"{parameter_label(diagram.parameter_name)}: %{{x:.4g}}<br>"
        "Frequency: %{y:.4g} Hz<br>Damped frequency: %{customdata[1]:.4g} Hz<br>"
        "Damping ratio: %{customdata[0]:.2%}<extra></extra>"
    )
    for track in diagram.heavily_damped_tracks:
        x, frequency, damping = diagram.track_curve(track)
        damped = frequency * np.sqrt(np.clip(1.0 - damping**2, 0.0, None))
        figure.add_trace(
            go.Scatter(
                x=x,
                y=frequency,
                mode="lines+markers",
                name=f"{track.label.label} (heavily damped)",
                line={"dash": "dot", "color": HEAVILY_DAMPED_COLOR},
                marker={"symbol": "circle-open", "color": HEAVILY_DAMPED_COLOR},
                customdata=np.column_stack([damping, damped]),
                hovertemplate=hover,
                **plotly_legend_group(
                    "heavily damped", "Heavily damped", HEAVILY_DAMPED_RANK, titled
                ),
            )
        )


def _track_lines(diagram: CampbellDiagram) -> list[tuple[list[float], list[float]]]:
    """Return the ``(x, frequency)`` polyline of every track, for tag placement."""
    lines = []
    for track in diagram.tracks:
        x, frequency, _ = diagram.track_curve(track)
        lines.append((x.tolist(), frequency.tolist()))
    return lines


def _add_excitation_lines(
    figure: go.Figure,
    diagram: CampbellDiagram,
    harmonics: Sequence[int],
    titled: set[str],
    span: tuple[float, float],
    top: float | None,
) -> None:
    """Add the thin blue nP lines, one "Blade Passing" legend entry, inline tags."""
    x = [span[0], span[1]]
    group = plotly_legend_group("excitation", "", EXCITATION_RANK, titled)
    group.pop("legendgrouptitle", None)
    figure.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name="Blade Passing",
            line={"color": EXCITATION_COLOR, "width": 2},
            hoverinfo="skip",
            **group,
        )
    )
    for harmonic in harmonics:
        figure.add_trace(
            go.Scatter(
                x=x,
                y=[harmonic * value / 60.0 for value in x],
                mode="lines",
                line={"color": EXCITATION_COLOR, "width": 1},
                name=f"{harmonic}P",
                showlegend=False,
                legendgroup="excitation",
                hovertemplate=f"{harmonic}P excitation<extra></extra>",
            )
        )
    if top is None:
        return
    for order, x_tag, y_tag in excitation_tag_positions(
        harmonics, span, (0.0, top), _track_lines(diagram)
    ):
        figure.add_annotation(
            x=x_tag,
            y=y_tag,
            text=f"{order}P",
            showarrow=False,
            font={"color": EXCITATION_COLOR},
            bgcolor="rgba(255, 255, 255, 0.85)",
        )


def _add_resonance_markers(
    figure: go.Figure,
    diagram: CampbellDiagram,
    harmonics: Sequence[int],
    operating_range: tuple[float, float] | None,
    titled: set[str],
) -> None:
    """Add resonance crossing markers, one trace per severity, coloured by severity."""
    crossings = find_resonances(diagram, harmonics, operating_range=operating_range)
    by_severity = _group_by_severity(crossings)
    for severity in _SEVERITY_ORDER:
        group = by_severity.get(severity)
        if not group:
            continue
        figure.add_trace(
            go.Scatter(
                x=[crossing.rotor_speed_rpm for crossing in group],
                y=[crossing.frequency_hz for crossing in group],
                mode="markers",
                marker={
                    "symbol": "x",
                    "size": 11,
                    "color": SEVERITY_COLORS[severity],
                    "line": {"width": 1, "color": AXES_COLOR},
                },
                name=f"Resonance ({severity.value})",
                customdata=[
                    [crossing.harmonic, crossing.damping_ratio] for crossing in group
                ],
                hovertemplate=(
                    f"<b>Resonance ({severity.value})</b><br>"
                    "%{customdata[0]}P crossing<br>Rotor speed: %{x:.4g} rpm<br>"
                    "Frequency: %{y:.4g} Hz<br>Damping ratio: %{customdata[1]:.2%}"
                    "<extra></extra>"
                ),
                **plotly_legend_group("resonance", "Resonance", RESONANCE_RANK, titled),
            )
        )
