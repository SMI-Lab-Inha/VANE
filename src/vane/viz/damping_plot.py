# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Damping-ratio plotting against the operating parameter, with stability signals.

Renders the damping ratio of each tracked mode as a function of the operating
parameter (rotor speed or wind speed), the companion to the Campbell diagram for
assessing stability margins across the operating envelope. :func:`plot_damping` returns
an interactive Plotly figure and :func:`plot_damping_static` a Matplotlib figure for
publication.

The plot makes stability legible at a glance: the negative-damping region is shaded as
**unstable** and a thin band above zero as a low-margin **caution** zone, and any point
that crosses into negative damping is flagged. Tracks use the same styling as the
Campbell diagram: colour, dash and marker by mode family
(:func:`vane.viz.mode_line_style`), line opacity scaled by identification confidence,
and open-diamond markers on ambiguous tracks and points. The static figure names each
track at its line end and explains the encoding in a compact key; the interactive one
has a legend grouped by family.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import numpy as np
import plotly.graph_objects as go

from vane.viz._style import (
    ALERT_COLOR,
    AMBIGUOUS_MPL_MARKER,
    LOW_CONFIDENCE_OPACITY,
    STANDARD_COLORS,
    TRACK_LINEWIDTH,
    VANE_STYLE,
    EncodingKey,
    LegendGroups,
    ambiguous_points,
    confidence_opacity,
    direct_label_text,
    label_line_ends,
    new_axes,
    parameter_label,
    plotly_family_legend,
    plotly_point_symbols,
    resolve_size,
    resolve_template,
    static_style,
    text_color,
    track_name,
    track_styles,
)

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from vane.campbell.builder import CampbellDiagram
    from vane.viz.publication import FigureSize

__all__ = ["plot_damping", "plot_damping_static"]

# Default low-damping caution threshold (fraction): modes below this are lightly damped.
_DEFAULT_CAUTION_DAMPING = 0.01
_UNSTABLE_COLOR = ALERT_COLOR
_CAUTION_COLOR = STANDARD_COLORS[5]
_BAND_OPACITY = 0.08


def _check_caution(caution_damping: float) -> None:
    """Reject a negative caution threshold."""
    if caution_damping < 0.0:
        msg = f"caution_damping must be non-negative, got {caution_damping}"
        raise ValueError(msg)


def _minimum_damping(damping: npt.NDArray[np.float64], current: float) -> float:
    """Return the running minimum damping, ignoring NaN (failed) points.

    A plain ``min()`` would return NaN, and ``min(0.0, NaN)`` keeps ``0.0``, hiding the
    track's genuine negative damping.
    """
    finite = damping[np.isfinite(damping)]
    return min(current, float(finite.min())) if finite.size else current


def _band_floor(minimum_damping: float, caution_damping: float, scale: float) -> float:
    """Return the lower edge of the unstable band, in plotted units.

    The band reaches below the lowest plotted damping but stays visible (a 0.5 % floor)
    even when every mode is stable, so the danger zone always shows.
    """
    depth = max(-minimum_damping, caution_damping, 0.005)
    return -depth * scale * 1.05


def _unstable_title(unstable_labels: list[str]) -> str | None:
    """Return a warning title naming the modes that go unstable, or ``None``."""
    if not unstable_labels:
        return None
    unique = ", ".join(dict.fromkeys(unstable_labels))
    return f"UNSTABLE: {unique}"


def _damping_label(as_percent: bool) -> str:
    """Return the damping axis label with its unit."""
    return "Damping ratio (%)" if as_percent else "Damping ratio (-)"


def plot_damping(
    diagram: CampbellDiagram,
    *,
    as_percent: bool = True,
    show_stability_bands: bool = True,
    show_confidence: bool = True,
    caution_damping: float = _DEFAULT_CAUTION_DAMPING,
    template: str | go.layout.Template | None = None,
    title: str | None = None,
) -> go.Figure:
    """Plot damping ratio versus the operating parameter, with stability signals.

    Parameters
    ----------
    diagram : CampbellDiagram
        The diagram whose tracks are plotted.
    as_percent : bool, optional
        Plot damping as a percentage (the default) rather than a fraction.
    show_stability_bands : bool, optional
        Shade the unstable (damping < 0) region and the low-margin caution band above
        it. Default ``True``.
    show_confidence : bool, optional
        Scale each track's line opacity by its identification confidence and mark an
        ambiguous track (and the ambiguous points of a clear track) with open
        diamonds. Default ``True``.
    caution_damping : float, optional
        Upper edge of the caution band as a damping *fraction* (default ``0.01``, i.e.
        1%); modes below it are flagged as lightly damped.
    template : str, plotly.graph_objects.layout.Template or None, optional
        Plotly layout template to apply. ``None`` (the default) applies the VANE
        template ``"vane"``.
    title : str or None, optional
        Figure title. ``None`` (the default) draws a title only as a warning, naming
        the modes that go unstable.

    Returns
    -------
    plotly.graph_objects.Figure
        The damping figure: one line per tracked mode (styled by mode family, with the
        legend grouped by family), unstable points flagged, and, when enabled, the
        unstable and caution stability bands.

    Raises
    ------
    ValueError
        If ``caution_damping`` is negative.
    """
    _check_caution(caution_damping)
    scale = 100.0 if as_percent else 1.0
    figure = go.Figure()
    unstable_labels: list[str] = []
    minimum_damping = 0.0
    titled: set[str] = set()
    axis_label = parameter_label(diagram.parameter_name)
    unit = " %" if as_percent else ""
    hover = (
        "<b>%{fullData.name}</b><br>"
        f"{axis_label}: %{{x:.4g}}<br>Operating point: %{{customdata}}<br>"
        f"Damping ratio: %{{y:.3g}}{unit}<extra></extra>"
    )
    for track, style in zip(diagram.tracks, track_styles(diagram.tracks), strict=True):
        x, _, damping = diagram.track_curve(track)
        minimum_damping = _minimum_damping(damping, minimum_damping)
        ambiguous = track.is_ambiguous
        flagged = show_confidence and ambiguous
        figure.add_trace(
            go.Scatter(
                x=x,
                y=damping * scale,
                mode="lines+markers",
                name=track_name(track, ambiguous),
                opacity=confidence_opacity(track) if show_confidence else 1.0,
                line={"color": style.color, "dash": style.dash},
                marker={
                    "symbol": plotly_point_symbols(
                        track, style.marker, flagged, show_confidence
                    ),
                    "color": style.color,
                },
                customdata=list(track.operating_points),
                hovertemplate=hover,
                **plotly_family_legend(style.family, titled),
            )
        )
        unstable = damping < 0.0
        if bool(unstable.any()):
            unstable_labels.append(track.label.label)
            figure.add_trace(
                go.Scatter(
                    x=x[unstable],
                    y=damping[unstable] * scale,
                    mode="markers",
                    marker={"symbol": "x", "size": 10, "color": _UNSTABLE_COLOR},
                    name=f"{track.label.label} (unstable)",
                    legendgroup=style.family.value,
                    showlegend=False,
                    hovertemplate="UNSTABLE<br>damping=%{y:.3g}<extra></extra>",
                )
            )

    if show_stability_bands and diagram.tracks:
        _add_stability_bands(figure, minimum_damping, caution_damping, scale)

    layout: dict[str, Any] = {
        "xaxis_title": axis_label,
        "yaxis_title": _damping_label(as_percent),
        "legend": {"traceorder": "grouped"},
        "template": resolve_template(template),
    }
    shown = _unstable_title(unstable_labels) if title is None else title
    if shown is not None:
        layout["title"] = shown
    figure.update_layout(**layout)
    return figure


def plot_damping_static(
    diagram: CampbellDiagram,
    *,
    as_percent: bool = True,
    show_stability_bands: bool = True,
    show_confidence: bool = True,
    caution_damping: float = _DEFAULT_CAUTION_DAMPING,
    ax: Axes | None = None,
    style: str | dict[str, Any] | Path | None = VANE_STYLE,
    size: str | float | FigureSize | None = None,
    direct_labels: bool = True,
    title: str | None = None,
) -> Figure:
    """Plot damping ratio versus the operating parameter as a Matplotlib figure.

    The static counterpart of :func:`plot_damping`: the same tracks, family styling,
    stability bands and instability flags (red filled crosses). Each mode is named at
    its line end; a compact key inside the axes explains the encoding.

    Parameters
    ----------
    diagram : CampbellDiagram
        The diagram whose tracks are plotted.
    as_percent : bool, optional
        Plot damping as a percentage (the default) rather than a fraction.
    show_stability_bands : bool, optional
        Shade the unstable (damping < 0) region and the low-margin caution band above
        it. Default ``True``.
    show_confidence : bool, optional
        Scale each track's line opacity by its identification confidence and mark
        ambiguous tracks and points with open diamonds. Default ``True``.
    caution_damping : float, optional
        Upper edge of the caution band as a damping *fraction* (default ``0.01``).
    ax : matplotlib.axes.Axes or None, optional
        Draw into these axes instead of creating a new figure; the parent figure is
        returned.
    style : str, dict, pathlib.Path or None, optional
        Matplotlib style in effect while drawing: ``"vane"`` (the default),
        ``None`` for the current ``rcParams``, or any value accepted by
        :func:`matplotlib.style.context`.
    size : str, float, FigureSize or None, optional
        Figure size and fonts (see :func:`vane.viz.figure_size`); ``None`` uses
        ``"double"`` for a new figure.
    direct_labels : bool, optional
        Name the tracks at their line ends with a compact key (the default);
        ``False`` draws a per-track legend grouped by family instead.
    title : str or None, optional
        Axes title; ``None`` (the default) draws none.

    Returns
    -------
    matplotlib.figure.Figure
        The damping figure.

    Raises
    ------
    ValueError
        If ``caution_damping`` is negative.
    """
    _check_caution(caution_damping)
    resolved = resolve_size(size, "double", ax=ax, style=style)
    with static_style(style, resolved):
        figure, axes = new_axes(ax)
        legend = None if direct_labels else LegendGroups()
        key, ends = draw_damping_static(
            axes,
            diagram,
            as_percent=as_percent,
            show_stability_bands=show_stability_bands,
            show_confidence=show_confidence,
            caution_damping=caution_damping,
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


def draw_damping_static(
    axes: Axes,
    diagram: CampbellDiagram,
    *,
    as_percent: bool,
    show_stability_bands: bool,
    show_confidence: bool,
    caution_damping: float,
    legend: LegendGroups | None,
) -> tuple[EncodingKey, list[tuple[float, float, str, str]]]:
    """Draw the damping ratios on Matplotlib axes.

    Returns the encoding key and the line ends ``(x, y, text, colour)`` for the direct
    labels; when ``legend`` is given, every track is also added to it.

    Raises
    ------
    ValueError
        If ``caution_damping`` is negative.
    """
    _check_caution(caution_damping)
    scale = 100.0 if as_percent else 1.0
    key = EncodingKey()
    ends: list[tuple[float, float, str, str]] = []
    minimum_damping = 0.0
    any_unstable = False
    styles = track_styles(diagram.tracks)
    for track, line_style in zip(diagram.tracks, styles, strict=True):
        x, _, damping = diagram.track_curve(track)
        minimum_damping = _minimum_damping(damping, minimum_damping)
        ambiguous = show_confidence and track.is_ambiguous
        alpha = confidence_opacity(track) if show_confidence else 1.0
        flags = ambiguous_points(track) if show_confidence else None
        clear = (
            None
            if flags is None or ambiguous
            else [i for i, flag in enumerate(flags) if not flag]
        )
        (line,) = axes.plot(
            x,
            damping * scale,
            color=line_style.color,
            linestyle=line_style.linestyle,
            linewidth=TRACK_LINEWIDTH,
            marker=AMBIGUOUS_MPL_MARKER if ambiguous else line_style.mpl_marker,
            markerfacecolor="none" if ambiguous else line_style.color,
            markevery=clear,
            alpha=alpha,
        )
        if clear is not None and flags is not None:
            mask = np.asarray(flags, dtype=np.bool_)
            axes.plot(
                x[mask],
                damping[mask] * scale,
                linestyle="none",
                marker=AMBIGUOUS_MPL_MARKER,
                markerfacecolor="none",
                markeredgecolor=line_style.color,
                alpha=alpha,
            )
        key.add_track(
            track.label,
            ambiguous=ambiguous or clear is not None,
            low_confidence=alpha < LOW_CONFIDENCE_OPACITY,
        )
        if legend is not None:
            legend.add_family(
                line_style.family, line, track_name(track, track.is_ambiguous)
            )
        finite = np.flatnonzero(np.isfinite(damping))
        if finite.size:
            last = int(finite[-1])
            shade = text_color(line_style.color)
            text = direct_label_text(axes, track.label.label)
            ends.append((float(x[last]), float(damping[last] * scale), text, shade))
        unstable = damping < 0.0
        if bool(unstable.any()):
            any_unstable = True
            axes.plot(
                x[unstable],
                damping[unstable] * scale,
                linestyle="none",
                marker="X",
                markersize=7.0,
                markeredgecolor="#262626",
                markeredgewidth=0.5,
                color=_UNSTABLE_COLOR,
                zorder=4,
            )
    if any_unstable:
        key.markers.append(("Unstable", _UNSTABLE_COLOR))
    if show_stability_bands and diagram.tracks:
        floor = _band_floor(minimum_damping, caution_damping, scale)
        axes.axhspan(floor, 0.0, color=_UNSTABLE_COLOR, alpha=_BAND_OPACITY, lw=0)
        axes.annotate(
            "unstable",
            xy=(1.0, 0.0),
            xycoords=("axes fraction", "data"),
            xytext=(-3.0, -2.0),
            textcoords="offset points",
            color=_UNSTABLE_COLOR,
            ha="right",
            va="top",
            fontsize=mpl.rcParams["xtick.labelsize"],
        )
        if caution_damping > 0.0:
            axes.axhspan(
                0.0,
                caution_damping * scale,
                color=_CAUTION_COLOR,
                alpha=_BAND_OPACITY,
                linewidth=0,
            )
    axes.axhline(0.0, color="0.5", linewidth=0.6, zorder=1)
    axes.set_xlabel(parameter_label(diagram.parameter_name))
    axes.set_ylabel(_damping_label(as_percent))
    return key, ends


def _add_stability_bands(
    figure: go.Figure,
    minimum_damping: float,
    caution_damping: float,
    scale: float,
) -> None:
    """Shade the unstable (damping < 0) region and the low-margin caution band."""
    figure.add_hrect(
        y0=_band_floor(minimum_damping, caution_damping, scale),
        y1=0.0,
        fillcolor=_UNSTABLE_COLOR,
        opacity=_BAND_OPACITY,
        line_width=0,
        layer="below",
        annotation_text="unstable",
        annotation_position="bottom left",
    )
    if caution_damping > 0.0:
        figure.add_hrect(
            y0=0.0,
            y1=caution_damping * scale,
            fillcolor=_CAUTION_COLOR,
            opacity=_BAND_OPACITY,
            line_width=0,
            layer="below",
        )
