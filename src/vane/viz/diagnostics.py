# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Diagnostic plots for the identification and tracking quality.

These are the "show your work" visuals that turn the pipeline's internal quality
signals into inspectable figures: the Modal Assurance Criterion (MAC) correlation
between adjacent operating points (the basis of mode tracking), the per-track
identification confidence, the zoomed review of contested crossings and veerings, and
the azimuth coverage behind each operating point's averaged model. Each plot has an
interactive (Plotly) form and a static (Matplotlib) form for publication, the latter
named ``*_static``. They complement the reference plots and feed the audit report.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.patches import Rectangle
from plotly.subplots import make_subplots

from vane.modal.mac import compute_mac
from vane.viz._style import (
    AXES_COLOR,
    REFERENCE_COLOR,
    STANDARD_COLORS,
    TRACK_LINEWIDTH,
    VANE_STYLE,
    new_axes,
    parameter_label,
    rc_points,
    resolve_size,
    resolve_template,
    static_style,
    track_styles,
)
from vane.viz.publication import add_panel_labels

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from vane.campbell.builder import CampbellDiagram
    from vane.modal.branch import BranchGroup
    from vane.modal.eigensolver import ModalSolution
    from vane.study import Provenance
    from vane.viz.publication import FigureSize

__all__ = [
    "plot_azimuth_coverage",
    "plot_azimuth_coverage_static",
    "plot_branch_review",
    "plot_branch_review_static",
    "plot_mac_matrix",
    "plot_mac_matrix_static",
    "plot_tracking_confidence",
    "plot_tracking_confidence_static",
]

# Bar colour for confident versus ambiguous tracks: blue and orange.
_CONFIDENT_COLOR = STANDARD_COLORS[2]
_AMBIGUOUS_COLOR = STANDARD_COLORS[5]
# Distinct colours cycled across a branch group's member tracks.
_BRANCH_COLORS = STANDARD_COLORS
# Most hotspot panels drawn before the rest are summarized rather than shown.
_MAX_BRANCH_PANELS = 9
_BRANCH_PANEL_COLUMNS = 3
# Identification confidence below which the audit report flags a track for review.
_REVIEW_THRESHOLD = 0.5
_MAC_LABEL = "MAC (-)"
_CONFIDENCE_LABEL = "Identification confidence (-)"
_FREQUENCY_LABEL = "Frequency (Hz)"
_AZIMUTH_LABEL = "Azimuth (deg)"


def _with_title(layout: dict[str, Any], title: str | None) -> dict[str, Any]:
    """Add a title to a layout update when one is given."""
    if title is not None:
        layout["title"] = title
    return layout


def _mac_labels(
    reference: ModalSolution, test: ModalSolution
) -> tuple[list[str], list[str]]:
    """Return the frequency tick labels of the two solutions."""
    return (
        [f"{freq:.3g}" for freq in reference.natural_frequencies_hz],
        [f"{freq:.3g}" for freq in test.natural_frequencies_hz],
    )


def plot_mac_matrix(
    reference: ModalSolution,
    test: ModalSolution,
    *,
    template: str | go.layout.Template | None = None,
    title: str | None = None,
) -> go.Figure:
    """Plot the MAC matrix between two operating points as a heatmap.

    The MAC between mode *i* of ``reference`` and mode *j* of ``test`` is the squared
    correlation of their shapes; a near-diagonal pattern means the modes track cleanly,
    while off-diagonal energy flags crossings or veering.

    Parameters
    ----------
    reference, test : ModalSolution
        The two solutions to correlate (e.g. adjacent operating points). Their mode
        shapes must share the same degrees of freedom.
    template : str, plotly.graph_objects.layout.Template or None, optional
        Plotly layout template to apply. ``None`` (the default) applies the VANE
        template ``"vane"``.
    title : str or None, optional
        Figure title; ``None`` (the default) draws none.

    Returns
    -------
    plotly.graph_objects.Figure
        The MAC heatmap, with values in ``[0, 1]``; the axes are labelled with the
        natural frequencies of the modes.
    """
    mac = compute_mac(reference.mode_shapes, test.mode_shapes)
    ref_labels, test_labels = _mac_labels(reference, test)
    n_ref = len(ref_labels)
    n_test = len(test_labels)
    # Cells sit at integer positions with the frequencies as tick text, so two modes
    # at the same (rounded) frequency stay distinct rows/columns rather than collapsing
    # onto one categorical position.
    customdata = [
        [[ref_labels[i], test_labels[j]] for j in range(n_test)] for i in range(n_ref)
    ]
    figure = go.Figure(
        go.Heatmap(
            z=mac,
            x=list(range(n_test)),
            y=list(range(n_ref)),
            customdata=customdata,
            zmin=0.0,
            zmax=1.0,
            colorscale="Greys",
            colorbar={"title": {"text": _MAC_LABEL}},
            hovertemplate=(
                "Reference mode: %{customdata[0]} Hz<br>"
                "Test mode: %{customdata[1]} Hz<br>MAC: %{z:.3f}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        **_with_title(
            {
                "xaxis": {
                    "title": {"text": "Test mode frequency (Hz)"},
                    "tickmode": "array",
                    "tickvals": list(range(n_test)),
                    "ticktext": test_labels,
                },
                "yaxis": {
                    "title": {"text": "Reference mode frequency (Hz)"},
                    "tickmode": "array",
                    "tickvals": list(range(n_ref)),
                    "ticktext": ref_labels,
                },
                "template": resolve_template(template),
            },
            title,
        )
    )
    return figure


def plot_mac_matrix_static(
    reference: ModalSolution,
    test: ModalSolution,
    *,
    ax: Axes | None = None,
    style: str | dict[str, Any] | Path | None = VANE_STYLE,
    size: str | float | FigureSize | None = None,
    annotate: bool | None = None,
    title: str | None = None,
) -> Figure:
    """Plot the MAC matrix between two operating points as a static heatmap.

    Parameters
    ----------
    reference, test : ModalSolution
        The two solutions to correlate; their mode shapes must share the same degrees
        of freedom.
    ax : matplotlib.axes.Axes or None, optional
        Draw into these axes instead of creating a new figure.
    style : str, dict, pathlib.Path or None, optional
        Matplotlib style in effect while drawing (``"vane"`` by default; ``None`` keeps
        the current ``rcParams``).
    size : str, float, FigureSize or None, optional
        Figure size and fonts (see :func:`vane.viz.figure_size`); ``None`` uses
        ``"single"`` for a new figure, square.
    annotate : bool or None, optional
        Write the MAC value in each cell. ``None`` (the default) does so for matrices
        of up to 10 by 10 modes.
    title : str or None, optional
        Axes title; ``None`` (the default) draws none.

    Returns
    -------
    matplotlib.figure.Figure
        The heatmap on a grey scale from white (0) to black (1), with a colour bar;
        the axes are labelled with the natural frequencies of the modes.
    """
    mac = compute_mac(reference.mode_shapes, test.mode_shapes)
    ref_labels, test_labels = _mac_labels(reference, test)
    resolved = resolve_size(size, "single", ax=ax, style=style, aspect=(0.85, 0.5))
    write_values = max(mac.shape) <= 10 if annotate is None else annotate
    with static_style(style, resolved):
        figure, axes = new_axes(ax)
        image = axes.imshow(
            mac, cmap="Greys", vmin=0.0, vmax=1.0, origin="lower", aspect="auto"
        )
        axes.set_xticks(range(len(test_labels)), test_labels, rotation=90)
        axes.set_yticks(range(len(ref_labels)), ref_labels)
        axes.tick_params(top=False, right=False)
        axes.set_xlabel("Test mode frequency (Hz)")
        axes.set_ylabel("Reference mode frequency (Hz)")
        colorbar = figure.colorbar(image, ax=axes)
        colorbar.set_label(_MAC_LABEL)
        if write_values:
            size_pt = rc_points("xtick.labelsize") - 1.0
            for (row, col), value in np.ndenumerate(mac):
                axes.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=size_pt,
                    color="white" if value > 0.5 else AXES_COLOR,
                )
        if title is not None:
            axes.set_title(title)
    return figure


def plot_tracking_confidence(
    diagram: CampbellDiagram,
    *,
    template: str | go.layout.Template | None = None,
    title: str | None = None,
) -> go.Figure:
    """Plot each tracked mode's identification confidence as a bar chart.

    Ambiguous tracks are coloured distinctly, and a reference line marks the level
    below which the audit report flags a track for review.

    Parameters
    ----------
    diagram : CampbellDiagram
        The diagram whose tracks are shown.
    template : str, plotly.graph_objects.layout.Template or None, optional
        Plotly layout template to apply. ``None`` (the default) applies the VANE
        template ``"vane"``.
    title : str or None, optional
        Figure title; ``None`` (the default) draws none.

    Returns
    -------
    plotly.graph_objects.Figure
        One bar per track, confidence on ``[0, 1]``.
    """
    tracks = diagram.tracks
    # Bars sit at integer positions, not at the labels themselves, so two tracks that
    # share a physical label remain two distinct bars (a categorical x-axis would
    # merge same-label bars into one).
    positions = list(range(len(tracks)))
    labels = [track.label.label for track in tracks]
    confidence = [track.confidence for track in tracks]
    colors = [
        _AMBIGUOUS_COLOR if track.is_ambiguous else _CONFIDENT_COLOR for track in tracks
    ]
    customdata = [
        [label, "yes" if track.is_ambiguous else "no"]
        for label, track in zip(labels, tracks, strict=True)
    ]
    figure = go.Figure(
        go.Bar(
            x=positions,
            y=confidence,
            marker_color=colors,
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Confidence: %{y:.2f}<br>"
                "Ambiguous: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        **_with_title(
            {
                "yaxis_title": _CONFIDENCE_LABEL,
                "yaxis_range": [0.0, 1.05],
                "xaxis": {
                    "title": {"text": "Mode"},
                    "tickmode": "array",
                    "tickvals": positions,
                    "ticktext": labels,
                },
                "template": resolve_template(template),
            },
            title,
        )
    )
    figure.add_hline(
        y=_REVIEW_THRESHOLD,
        line_dash="dash",
        line_color=REFERENCE_COLOR,
        annotation_text="review threshold",
    )
    return figure


def plot_tracking_confidence_static(
    diagram: CampbellDiagram,
    *,
    ax: Axes | None = None,
    style: str | dict[str, Any] | Path | None = VANE_STYLE,
    size: str | float | FigureSize | None = None,
    title: str | None = None,
) -> Figure:
    """Plot each tracked mode's identification confidence as a static bar chart.

    Parameters
    ----------
    diagram : CampbellDiagram
        The diagram whose tracks are shown.
    ax : matplotlib.axes.Axes or None, optional
        Draw into these axes instead of creating a new figure.
    style : str, dict, pathlib.Path or None, optional
        Matplotlib style in effect while drawing (``"vane"`` by default).
    size : str, float, FigureSize or None, optional
        Figure size and fonts; ``None`` uses ``"single"`` for a new figure.
    title : str or None, optional
        Axes title; ``None`` (the default) draws none.

    Returns
    -------
    matplotlib.figure.Figure
        Horizontal bars, one per track in diagram order (top to bottom), blue for a
        clear track and orange for an ambiguous one, with the review threshold
        dashed.
    """
    tracks = diagram.tracks
    count = len(tracks)
    aspect = (
        min(1.2, max(0.5, 0.07 * count + 0.25)),
        min(0.7, max(0.3, 0.035 * count + 0.15)),
    )
    resolved = resolve_size(size, "single", ax=ax, style=style, aspect=aspect)
    with static_style(style, resolved):
        figure, axes = new_axes(ax)
        positions = np.arange(len(tracks))
        axes.barh(
            positions,
            [track.confidence for track in tracks],
            color=[
                _AMBIGUOUS_COLOR if track.is_ambiguous else _CONFIDENT_COLOR
                for track in tracks
            ],
            height=0.7,
        )
        axes.set_yticks(positions, [track.label.label for track in tracks])
        axes.invert_yaxis()
        axes.set_xlim(0.0, 1.0)
        axes.axvline(
            _REVIEW_THRESHOLD, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8
        )
        axes.set_xlabel(_CONFIDENCE_LABEL)
        if any(track.is_ambiguous for track in tracks):
            handles = [
                Rectangle((0, 0), 1, 1, color=_CONFIDENT_COLOR),
                Rectangle((0, 0), 1, 1, color=_AMBIGUOUS_COLOR),
            ]
            axes.legend(
                handles,
                ["Clear", "Ambiguous"],
                loc="lower left",
                bbox_to_anchor=(0.0, 1.02),
                borderaxespad=0.0,
                ncols=2,
            )
        if title is not None:
            axes.set_title(title)
    return figure


def plot_branch_review(
    diagram: CampbellDiagram,
    branch_groups: Sequence[BranchGroup],
    *,
    template: str | go.layout.Template | None = None,
    title: str | None = None,
) -> go.Figure:
    """Plot each contested branch group as a zoomed Campbell panel.

    One small-multiple panel per crossing/veering hotspot shows the competing tracks
    over the group's operating-point window, with a guide at the closest approach, so
    an analyst can review where the automatic tracking is least certain.

    Parameters
    ----------
    diagram : CampbellDiagram
        The diagram the branch groups index (its tracks and parameter axis).
    branch_groups : Sequence[BranchGroup]
        The contested groups to render (e.g. ``result.identification.branch_groups``).
    template : str, plotly.graph_objects.layout.Template or None, optional
        Plotly layout template to apply. ``None`` (the default) applies the VANE
        template ``"vane"``.
    title : str or None, optional
        Figure title. ``None`` (the default) draws one only to say how many groups
        were left out when there are more than nine.

    Returns
    -------
    plotly.graph_objects.Figure
        One panel per group (capped at nine, with the remainder noted), or a single
        "none detected" panel when ``branch_groups`` is empty.
    """
    if not branch_groups:
        return _empty_branch_figure(template, title)

    shown = list(branch_groups[:_MAX_BRANCH_PANELS])
    n_cols = min(_BRANCH_PANEL_COLUMNS, len(shown))
    n_rows = -(-len(shown) // n_cols)  # ceil division
    titles = [_branch_panel_title(group) for group in shown]
    figure = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles)
    for index, group in enumerate(shown):
        row, col = divmod(index, n_cols)
        _draw_branch_panel(figure, diagram, group, row + 1, col + 1)

    dropped = len(branch_groups) - len(shown)
    if title is None and dropped:
        title = f"Showing {len(shown)} of {len(branch_groups)} contested groups"
    figure.update_layout(
        **_with_title(
            {"showlegend": False, "template": resolve_template(template)}, title
        )
    )
    figure.update_xaxes(title_text=parameter_label(diagram.parameter_name))
    figure.update_yaxes(title_text=_FREQUENCY_LABEL)
    return figure


def _branch_panel_title(group: BranchGroup) -> str:
    """Return the title of a branch-review panel."""
    return f"{group.kind.value} (Δf min {group.min_separation_hz:.3g} Hz)"


def plot_branch_review_static(
    diagram: CampbellDiagram,
    branch_groups: Sequence[BranchGroup],
    *,
    style: str | dict[str, Any] | Path | None = VANE_STYLE,
    size: str | float | FigureSize | None = None,
) -> Figure:
    """Plot each contested branch group as a zoomed static Campbell panel.

    The static counterpart of :func:`plot_branch_review`: up to nine panels, three
    per row, labelled (a), (b), ... when there are several, each with a legend above
    it that names the kind of hotspot, the closest approach and the competing tracks
    (drawn in their family colours and line styles), the
    smooth alternative dashed when resolution computed it, and a dotted guide at the
    closest approach.

    Parameters
    ----------
    diagram : CampbellDiagram
        The diagram the branch groups index.
    branch_groups : Sequence[BranchGroup]
        The contested groups to render.
    style : str, dict, pathlib.Path or None, optional
        Matplotlib style in effect while drawing (``"vane"`` by default).
    size : str, float, FigureSize or None, optional
        Figure size and fonts; ``None`` uses ``"double"``.

    Returns
    -------
    matplotlib.figure.Figure
        The panels, or a single panel saying that no contested branches were found.
    """
    shown = list(branch_groups[:_MAX_BRANCH_PANELS])
    n_cols = max(1, min(_BRANCH_PANEL_COLUMNS, len(shown)))
    n_rows = max(1, -(-len(shown) // n_cols))
    resolved = resolve_size(
        size, "double", ax=None, style=style, aspect=0.33 * n_rows + 0.12
    )
    styles = track_styles(diagram.tracks)
    with static_style(style, resolved):
        figure, grid = plt.subplots(n_rows, n_cols, layout="constrained", squeeze=False)
        panels: list[Axes] = list(np.ravel(grid))
        if not shown:
            panels[0].text(
                0.5,
                0.5,
                "No contested branches detected",
                ha="center",
                va="center",
                transform=panels[0].transAxes,
            )
            panels[0].set_xticks([])
            panels[0].set_yticks([])
            return figure
        for panel, group in zip(panels, shown, strict=False):
            for track_index in group.track_indices:
                track = diagram.tracks[track_index]
                style_of = styles[track_index]
                x, freqs, _ = diagram.track_curve(track)
                ops = np.array(track.operating_points)
                mask = (ops >= group.op_lo) & (ops <= group.op_hi)
                panel.plot(
                    x[mask],
                    freqs[mask],
                    color=style_of.color,
                    linestyle=style_of.linestyle,
                    marker=style_of.mpl_marker,
                    linewidth=TRACK_LINEWIDTH,
                    label=track.label.label,
                )
            if group.smooth_branches is not None:
                window = diagram.parameter_values[group.op_lo : group.op_hi + 1]
                for branch in group.smooth_branches:
                    panel.plot(
                        window, list(branch), color=AXES_COLOR, linestyle="--", lw=0.8
                    )
            panel.axvline(
                float(diagram.parameter_values[group.min_separation_op]),
                color=REFERENCE_COLOR,
                linestyle=":",
                linewidth=0.8,
            )
            small = rc_points("legend.fontsize") - 1.0
            panel.legend(
                loc="lower left",
                bbox_to_anchor=(0.0, 1.02),
                borderaxespad=0.0,
                ncols=2,
                title=_branch_panel_title(group),
                fontsize=small,
                title_fontsize=small,
                alignment="left",
            )
        for panel in panels[len(shown) :]:
            panel.set_visible(False)
        figure.supxlabel(
            parameter_label(diagram.parameter_name),
            fontsize=plt.rcParams["axes.labelsize"],
        )
        figure.supylabel(_FREQUENCY_LABEL, fontsize=plt.rcParams["axes.labelsize"])
        if len(shown) > 1:
            add_panel_labels(panels[: len(shown)])
    return figure


def _draw_branch_panel(
    figure: go.Figure,
    diagram: CampbellDiagram,
    group: BranchGroup,
    row: int,
    col: int,
) -> None:
    """Draw one branch group's competing tracks into a subplot cell."""
    for offset, track_index in enumerate(group.track_indices):
        track = diagram.tracks[track_index]
        x, freqs, _ = diagram.track_curve(track)
        ops = np.array(track.operating_points)
        mask = (ops >= group.op_lo) & (ops <= group.op_hi)
        color = _BRANCH_COLORS[offset % len(_BRANCH_COLORS)]
        figure.add_trace(
            go.Scatter(
                x=x[mask],
                y=freqs[mask],
                mode="lines+markers",
                name=track.label.label,
                line={"color": color},
                hovertemplate=(
                    f"<b>{track.label.label}</b><br>"
                    f"{parameter_label(diagram.parameter_name)}: %{{x:.4g}}<br>"
                    "Frequency: %{y:.4g} Hz<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )
    figure.add_vline(
        x=float(diagram.parameter_values[group.min_separation_op]),
        line_dash="dot",
        line_color=REFERENCE_COLOR,
        row=row,
        col=col,
    )
    _draw_smooth_branches(figure, diagram, group, row, col)


def _draw_smooth_branches(
    figure: go.Figure,
    diagram: CampbellDiagram,
    group: BranchGroup,
    row: int,
    col: int,
) -> None:
    """Overlay the smooth (minimum-curvature) alternative, if resolution computed it."""
    if group.smooth_branches is None:
        return
    window_x = diagram.parameter_values[group.op_lo : group.op_hi + 1]
    for offset, branch in enumerate(group.smooth_branches):
        figure.add_trace(
            go.Scatter(
                x=window_x,
                y=list(branch),
                mode="lines",
                name="smooth alternative",
                line={"color": AXES_COLOR, "dash": "dash", "width": 1},
                hovertemplate=(
                    "<b>smooth alternative</b><br>"
                    f"{parameter_label(diagram.parameter_name)}: %{{x:.4g}}<br>"
                    "Frequency: %{y:.4g} Hz<extra></extra>"
                ),
                showlegend=offset == 0,
            ),
            row=row,
            col=col,
        )


def _empty_branch_figure(
    template: str | go.layout.Template | None, title: str | None
) -> go.Figure:
    """Return a placeholder figure when no branch groups were detected."""
    figure = go.Figure()
    figure.update_layout(
        **_with_title(
            {
                "annotations": [
                    {
                        "text": "No contested branches detected",
                        "showarrow": False,
                        "xref": "paper",
                        "yref": "paper",
                        "x": 0.5,
                        "y": 0.5,
                    }
                ],
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "template": resolve_template(template),
            },
            title,
        )
    )
    return figure


def plot_azimuth_coverage(
    provenance: Provenance,
    *,
    template: str | go.layout.Template | None = None,
    title: str | None = None,
) -> go.Figure:
    """Plot the azimuth samples behind each operating point's averaged model.

    Sparse or clustered azimuths weaken the azimuth average; this shows, per operating
    point, exactly which rotor azimuths were sampled.

    Parameters
    ----------
    provenance : Provenance
        A study's provenance, carrying each operating point's azimuth list.
    template : str, plotly.graph_objects.layout.Template or None, optional
        Plotly layout template to apply. ``None`` (the default) applies the VANE
        template ``"vane"``.
    title : str or None, optional
        Figure title; ``None`` (the default) draws none.

    Returns
    -------
    plotly.graph_objects.Figure
        Azimuth (degrees) versus operating-parameter value, one marker per sample.
    """
    figure = go.Figure()
    axis_label = parameter_label(provenance.parameter_name)
    for index, point in enumerate(provenance.operating_points):
        name = point.name or f"OP {index}"
        figure.add_trace(
            go.Scatter(
                x=np.full(len(point.azimuths_deg), point.parameter_value),
                y=list(point.azimuths_deg),
                mode="markers",
                name=name,
                marker={"color": AXES_COLOR},
                hovertemplate=(
                    f"<b>{name}</b><br>{axis_label}: %{{x:.4g}}<br>"
                    "Azimuth: %{y:.1f}&deg;<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        **_with_title(
            {
                "xaxis_title": axis_label,
                "yaxis_title": _AZIMUTH_LABEL,
                "yaxis": {"range": [-10.0, 370.0], "dtick": 90},
                "showlegend": False,
                "template": resolve_template(template),
            },
            title,
        )
    )
    return figure


def plot_azimuth_coverage_static(
    provenance: Provenance,
    *,
    ax: Axes | None = None,
    style: str | dict[str, Any] | Path | None = VANE_STYLE,
    size: str | float | FigureSize | None = None,
    title: str | None = None,
) -> Figure:
    """Plot the sampled azimuths of each operating point as a static figure.

    Parameters
    ----------
    provenance : Provenance
        A study's provenance, carrying each operating point's azimuth list.
    ax : matplotlib.axes.Axes or None, optional
        Draw into these axes instead of creating a new figure.
    style : str, dict, pathlib.Path or None, optional
        Matplotlib style in effect while drawing (``"vane"`` by default).
    size : str, float, FigureSize or None, optional
        Figure size and fonts; ``None`` uses ``"single"`` for a new figure.
    title : str or None, optional
        Axes title; ``None`` (the default) draws none.

    Returns
    -------
    matplotlib.figure.Figure
        Azimuth (0 to 360 degrees, ticks every 90 degrees) versus the operating
        parameter, one black marker per sample.
    """
    resolved = resolve_size(size, "single", ax=ax, style=style)
    with static_style(style, resolved):
        figure, axes = new_axes(ax)
        for point in provenance.operating_points:
            axes.plot(
                np.full(len(point.azimuths_deg), point.parameter_value),
                list(point.azimuths_deg),
                linestyle="none",
                marker="o",
                markersize=3.5,
                color=AXES_COLOR,
            )
        axes.set_ylim(-15.0, 375.0)
        axes.set_yticks([0.0, 90.0, 180.0, 270.0, 360.0])
        axes.set_xlabel(parameter_label(provenance.parameter_name))
        axes.set_ylabel(_AZIMUTH_LABEL)
        if title is not None:
            axes.set_title(title)
    return figure
