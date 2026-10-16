# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Complex mode-shape composition views.

A mode shape is complex, and after the multi-blade-coordinate transform the blade
degrees of freedom split into collective, cosine, and sine coordinates. These views
expose that structure directly, without needing turbine geometry: a **phasor wheel**
plots each dominant DOF as a vector (length = participation magnitude, angle = phase),
its **animated** variant rotates the phasors through one modal period so the relative
phasing reads as motion, and the **blade-component** bar chart shows how a mode's
participation divides among the collective/cosine/sine multi-blade coordinates. The
phasor wheel and the bar chart also have static (Matplotlib) forms for publication.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import plotly.graph_objects as go

from vane.viz._style import (
    AXES_COLOR,
    STANDARD_COLORS,
    VANE_STYLE,
    resolve_size,
    resolve_template,
    series_colors,
    short_dof_name,
    static_style,
)

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt
    from matplotlib.figure import Figure
    from matplotlib.projections.polar import PolarAxes

    from vane.modal.eigensolver import ModalSolution
    from vane.viz.publication import FigureSize

__all__ = [
    "plot_blade_components",
    "plot_blade_components_static",
    "plot_mode_phasor",
    "plot_mode_phasor_animation",
    "plot_mode_phasor_static",
]

_MARKER_COLOR = STANDARD_COLORS[2]
_SPOKE_COLOR = "#c8c8c8"
_RADIAL_LABEL = "Normalized |participation| (-)"

# Bar colour for each multi-blade coordinate: black, red, blue.
_BLADE_COMPONENT_COLORS = {
    "collective": STANDARD_COLORS[0],
    "cosine": STANDARD_COLORS[1],
    "sine": STANDARD_COLORS[2],
}


def _polar_layout() -> dict[str, Any]:
    """Return the polar-axis layout shared by the phasor figures."""
    return {
        "radialaxis": {"range": [0.0, 1.05], "title": {"text": _RADIAL_LABEL}},
        "angularaxis": {"ticksuffix": "°", "direction": "counterclockwise"},
    }


def _title(default: str, title: str | None) -> str:
    """Return the title: the caller's, or the informative default."""
    return default if title is None else title


def plot_mode_phasor(
    solution: ModalSolution,
    mode_index: int,
    *,
    top_n: int = 10,
    template: str | go.layout.Template | None = None,
    title: str | None = None,
) -> go.Figure:
    """Plot a mode's dominant DOFs as phasors on a polar wheel.

    Each degree of freedom is drawn as a vector whose length is its participation
    magnitude (normalized to the most active DOF) and whose angle is its phase. Two
    DOFs that are in phase point the same way; a 180-degree spread is anti-phase; a
    spread of intermediate angles indicates a travelling (whirling) mode.

    Parameters
    ----------
    solution : ModalSolution
        A solution whose ``dof_descriptions`` are populated.
    mode_index : int
        Index of the mode to plot.
    top_n : int, optional
        Number of highest-participation degrees of freedom to show.
    template : str, plotly.graph_objects.layout.Template or None, optional
        Plotly layout template to apply. ``None`` (the default) applies the VANE
        template ``"vane"``.
    title : str or None, optional
        Figure title. ``None`` (the default) names the mode and its natural
        frequency; ``""`` draws none.

    Returns
    -------
    plotly.graph_objects.Figure
        The polar phasor figure.

    Raises
    ------
    ValueError
        If the solution has no DOF descriptions, ``mode_index`` is out of range, or
        ``top_n`` is not positive.
    """
    radius, magnitude, phase_deg, labels = _phasor_data(solution, mode_index, top_n)
    figure = go.Figure(_phasor_traces(radius, magnitude, phase_deg, labels))
    figure.update_layout(
        title=_title(_phasor_title(solution, mode_index), title),
        polar=_polar_layout(),
        showlegend=False,
        template=resolve_template(template),
    )
    return figure


def plot_mode_phasor_static(
    solution: ModalSolution,
    mode_index: int,
    *,
    top_n: int = 6,
    style: str | dict[str, Any] | Path | None = VANE_STYLE,
    size: str | float | FigureSize | None = None,
    title: str | None = None,
) -> Figure:
    """Plot a mode's dominant DOFs as phasors on a static polar wheel.

    Each degree of freedom is an arrow from the centre, its length the normalized
    participation magnitude and its angle the phase relative to the dominant DOF
    (counter-clockwise, in degrees). Each arrow has its own colour (the standard line
    colours, in order of participation), and the legend names the DOFs with their
    magnitudes.

    Parameters
    ----------
    solution : ModalSolution
        A solution whose ``dof_descriptions`` are populated.
    mode_index : int
        Index of the mode to plot.
    top_n : int, optional
        Number of highest-participation degrees of freedom to show (default 6).
    style : str, dict, pathlib.Path or None, optional
        Matplotlib style in effect while drawing (``"vane"`` by default).
    size : str, float, FigureSize or None, optional
        Figure size and fonts; ``None`` uses ``"single"``.
    title : str or None, optional
        Axes title; ``None`` (the default) draws none (the legend title names the
        mode and its frequency).

    Returns
    -------
    matplotlib.figure.Figure
        The polar figure.

    Raises
    ------
    ValueError
        If the solution has no DOF descriptions, ``mode_index`` is out of range, or
        ``top_n`` is not positive.
    """
    import matplotlib.pyplot as plt

    radius, _, phase_deg, _ = _phasor_data(solution, mode_index, top_n)
    names = _short_labels(solution, mode_index, top_n)
    colors = series_colors(len(names))
    resolved = resolve_size(size, "single", ax=None, style=style, aspect=(0.62, 0.4))
    with static_style(style, resolved):
        figure = plt.figure(layout="constrained")
        axes = figure.add_subplot(projection="polar")
        for value, angle, name, color in zip(
            radius.tolist(), phase_deg.tolist(), names, colors, strict=True
        ):
            theta = float(np.radians(angle))
            axes.annotate(
                "",
                xy=(theta, value),
                xytext=(0.0, 0.0),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "linewidth": 1.5,
                    "shrinkA": 0.0,
                    "shrinkB": 0.0,
                    "mutation_scale": 9.0,
                },
            )
            axes.plot([], [], color=color, linewidth=1.5, label=f"{name} ({value:.2f})")
        axes.set_ylim(0.0, 1.05)
        axes.set_yticks([0.25, 0.5, 0.75, 1.0])
        cast("PolarAxes", axes).set_rlabel_position(_clear_angle(phase_deg.tolist()))
        axes.grid(visible=True)
        frequency = float(solution.natural_frequencies_hz[mode_index])
        axes.legend(
            loc="center left",
            bbox_to_anchor=(1.12, 0.5),
            title=f"Mode {mode_index} ({frequency:.4g} Hz)",
            handlelength=1.5,
        )
        axes.text(
            np.radians(270.0),
            1.45,
            _RADIAL_LABEL,
            ha="center",
            va="top",
            fontsize=plt.rcParams["xtick.labelsize"],
            color=AXES_COLOR,
        )
        if title is not None:
            axes.set_title(title)
    return figure


def plot_mode_phasor_animation(
    solution: ModalSolution,
    mode_index: int,
    *,
    top_n: int = 10,
    n_frames: int = 24,
    template: str | go.layout.Template | None = None,
    title: str | None = None,
) -> go.Figure:
    """Animate a mode's phasors through one modal period.

    The phasors rotate rigidly (all DOFs share the modal frequency), so the animation
    makes the *relative* phasing visible as motion: an in-phase mode sweeps as a rigid
    fan, while a whirling mode shows a rotating spread.

    Parameters
    ----------
    solution : ModalSolution
        A solution whose ``dof_descriptions`` are populated.
    mode_index : int
        Index of the mode to plot.
    top_n : int, optional
        Number of highest-participation degrees of freedom to show.
    n_frames : int, optional
        Number of animation frames over one period (must be positive).
    template : str, plotly.graph_objects.layout.Template or None, optional
        Plotly layout template to apply. ``None`` (the default) applies the VANE
        template ``"vane"``.
    title : str or None, optional
        Figure title. ``None`` (the default) names the mode and its natural
        frequency; ``""`` draws none.

    Returns
    -------
    plotly.graph_objects.Figure
        The animated polar phasor figure, with a play button.

    Raises
    ------
    ValueError
        If the solution has no DOF descriptions, ``mode_index`` is out of range, or
        ``top_n`` or ``n_frames`` is not positive.
    """
    if n_frames < 1:
        msg = f"n_frames must be at least 1, got {n_frames}"
        raise ValueError(msg)
    radius, magnitude, phase_deg, labels = _phasor_data(solution, mode_index, top_n)
    frames = [
        go.Frame(
            data=_phasor_traces(
                radius, magnitude, phase_deg + 360.0 * k / n_frames, labels
            ),
            name=str(k),
        )
        for k in range(n_frames)
    ]
    figure = go.Figure(
        data=_phasor_traces(radius, magnitude, phase_deg, labels),
        frames=frames,
    )
    figure.update_layout(
        title=_title(_phasor_title(solution, mode_index), title),
        polar=_polar_layout(),
        showlegend=False,
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 80, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    }
                ],
            }
        ],
        template=resolve_template(template),
    )
    return figure


def _blade_component_data(
    solution: ModalSolution, mode_index: int
) -> tuple[list[int], npt.NDArray[np.float64]]:
    """Return the indices of the multi-blade DOFs and the mode's magnitudes."""
    if not solution.dof_descriptions:
        msg = "ModalSolution has no dof_descriptions; cannot plot blade components"
        raise ValueError(msg)
    if not 0 <= mode_index < solution.n_modes:
        msg = f"mode_index {mode_index} out of range [0, {solution.n_modes})"
        raise ValueError(msg)
    coordinates = solution.dof_mbc_coordinates
    blade_indices = [
        i for i, coord in enumerate(coordinates) if coord in _BLADE_COMPONENT_COLORS
    ]
    if not blade_indices:
        msg = (
            "solution carries no multi-blade coordinates; apply the MBC transform "
            "before plotting blade components"
        )
        raise ValueError(msg)
    magnitude = np.abs(solution.mode_shapes[:, mode_index]).astype(np.float64)
    return blade_indices, magnitude


def plot_blade_components(
    solution: ModalSolution,
    mode_index: int,
    *,
    template: str | go.layout.Template | None = None,
    title: str | None = None,
) -> go.Figure:
    """Plot how a mode's participation divides among the multi-blade coordinates.

    After the MBC transform each blade degree of freedom is a collective, cosine, or
    sine coordinate. This bar chart shows the participation magnitude of each such DOF,
    coloured by coordinate: a collective-dominated bar set marks a collective mode,
    while comparable cosine and sine sets mark a cyclic (whirling) mode.

    Parameters
    ----------
    solution : ModalSolution
        A solution carrying ``dof_mbc_coordinates`` (i.e. produced from an MBC-
        transformed model) and ``dof_descriptions``.
    mode_index : int
        Index of the mode to plot.
    template : str, plotly.graph_objects.layout.Template or None, optional
        Plotly layout template to apply. ``None`` (the default) applies the VANE
        template ``"vane"``.
    title : str or None, optional
        Figure title. ``None`` (the default) names the mode and its natural
        frequency; ``""`` draws none.

    Returns
    -------
    plotly.graph_objects.Figure
        One bar per multi-blade DOF, grouped and coloured by coordinate.

    Raises
    ------
    ValueError
        If the solution has no DOF descriptions, ``mode_index`` is out of range, or the
        solution carries no multi-blade coordinates.
    """
    blade_indices, magnitude = _blade_component_data(solution, mode_index)
    coordinates = solution.dof_mbc_coordinates
    positions = list(range(len(blade_indices)))
    # The MBC transform reorders the original per-blade descriptions but does not
    # rename them, so the row described as "Blade 1 ..." is actually the collective (or
    # cosine/sine) coordinate of its triplet. Name the bars by their coordinate.
    labels = [
        short_dof_name(solution.dof_descriptions[i], coordinates[i])
        for i in blade_indices
    ]
    figure = go.Figure()
    for component, color in _BLADE_COMPONENT_COLORS.items():
        # Bars sit at integer positions (with the DOF names as tick text) so two DOFs
        # sharing a description stay distinct.
        group = [
            (position, float(magnitude[index]), solution.dof_descriptions[index])
            for position, index in zip(positions, blade_indices, strict=True)
            if coordinates[index] == component
        ]
        if not group:
            continue
        figure.add_trace(
            go.Bar(
                x=[item[0] for item in group],
                y=[item[1] for item in group],
                name=component,
                marker_color=color,
                customdata=[item[2] for item in group],
                hovertemplate=(
                    "%{customdata}<br>Magnitude: %{y:.3g}"
                    f"<extra>{component}</extra>"
                ),
            )
        )
    frequency = solution.natural_frequencies_hz[mode_index]
    figure.update_layout(
        title=_title(f"Mode {mode_index} ({frequency:.4g} Hz)", title),
        yaxis_title="Participation magnitude (-)",
        xaxis={
            "title": {"text": "Multi-blade degree of freedom"},
            "tickmode": "array",
            "tickvals": positions,
            "ticktext": labels,
        },
        legend_title="MBC coordinate",
        template=resolve_template(template),
    )
    return figure


def plot_blade_components_static(
    solution: ModalSolution,
    mode_index: int,
    *,
    style: str | dict[str, Any] | Path | None = VANE_STYLE,
    size: str | float | FigureSize | None = None,
    title: str | None = None,
) -> Figure:
    """Plot how a mode's participation divides among the multi-blade coordinates.

    The static counterpart of :func:`plot_blade_components`: one bar per multi-blade
    DOF, coloured by coordinate (collective black, cosine red, sine blue), with the
    legend titled by the mode and its natural frequency.

    Parameters
    ----------
    solution : ModalSolution
        A solution carrying ``dof_mbc_coordinates`` and ``dof_descriptions``.
    mode_index : int
        Index of the mode to plot.
    style : str, dict, pathlib.Path or None, optional
        Matplotlib style in effect while drawing (``"vane"`` by default).
    size : str, float, FigureSize or None, optional
        Figure size and fonts; ``None`` uses ``"single"``.
    title : str or None, optional
        Axes title; ``None`` (the default) draws none.

    Returns
    -------
    matplotlib.figure.Figure
        The bar chart.

    Raises
    ------
    ValueError
        If the solution has no DOF descriptions, ``mode_index`` is out of range, or the
        solution carries no multi-blade coordinates.
    """
    import matplotlib.pyplot as plt

    blade_indices, magnitude = _blade_component_data(solution, mode_index)
    coordinates = solution.dof_mbc_coordinates
    resolved = resolve_size(size, "single", ax=None, style=style)
    with static_style(style, resolved):
        figure, axes = plt.subplots(layout="constrained")
        for component, color in _BLADE_COMPONENT_COLORS.items():
            group = [
                (position, float(magnitude[index]))
                for position, index in enumerate(blade_indices)
                if coordinates[index] == component
            ]
            if group:
                axes.bar(
                    [item[0] for item in group],
                    [item[1] for item in group],
                    color=color,
                    width=0.7,
                    label=component.capitalize(),
                )
        axes.set_xticks(
            range(len(blade_indices)),
            [
                short_dof_name(solution.dof_descriptions[i], coordinates[i])
                for i in blade_indices
            ],
            rotation=45,
            ha="right",
        )
        axes.tick_params(axis="x", top=False)
        axes.set_ylim(0.0, 1.15 * max(float(magnitude[blade_indices].max()), 1e-12))
        axes.set_ylabel("Participation magnitude (-)")
        frequency = float(solution.natural_frequencies_hz[mode_index])
        axes.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            borderaxespad=0.0,
            title=f"Mode {mode_index} ({frequency:.4g} Hz)",
            ncols=3,
        )
        if title is not None:
            axes.set_title(title)
    return figure


def _clear_angle(angles_deg: list[float]) -> float:
    """Return the polar angle (deg) farthest from every phasor, for tick labels."""
    candidates = [22.5 * k for k in range(16)]

    def clearance(candidate: float) -> float:
        return min(
            (abs((candidate - angle + 180.0) % 360.0 - 180.0) for angle in angles_deg),
            default=180.0,
        )

    return max(candidates, key=clearance)


def _phasor_data(
    solution: ModalSolution, mode_index: int, top_n: int
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    list[str],
]:
    """Return ``(radius, magnitude, phase_deg, labels)`` for a mode's top DOFs."""
    if not solution.dof_descriptions:
        msg = "ModalSolution has no dof_descriptions; cannot plot mode phasors"
        raise ValueError(msg)
    if not 0 <= mode_index < solution.n_modes:
        msg = f"mode_index {mode_index} out of range [0, {solution.n_modes})"
        raise ValueError(msg)
    if top_n < 1:
        msg = f"top_n must be at least 1, got {top_n}"
        raise ValueError(msg)

    # The eigensolver's mode shapes are already phase-normalized (the dominant DOF is
    # real and positive), so the phasor angles read as phases relative to that DOF.
    shape = solution.mode_shapes[:, mode_index]
    magnitude = np.abs(shape)
    order = _phasor_order(magnitude, top_n)
    selected = magnitude[order]
    peak = float(selected.max()) if selected.size and selected.max() > 0 else 1.0
    radius = selected / peak
    phase_deg = np.degrees(np.angle(shape[order]))
    labels = [solution.dof_descriptions[i] for i in order.tolist()]
    return radius, selected, phase_deg, labels


def _phasor_order(
    magnitude: npt.NDArray[np.float64], top_n: int
) -> npt.NDArray[np.intp]:
    """Return the indices of the ``top_n`` largest magnitudes, largest first."""
    return np.argsort(magnitude)[::-1][: min(top_n, magnitude.shape[0])]


def _short_labels(solution: ModalSolution, mode_index: int, top_n: int) -> list[str]:
    """Return the short DOF names of a mode's top DOFs, largest first."""
    order = _phasor_order(np.abs(solution.mode_shapes[:, mode_index]), top_n)
    coordinates = solution.dof_mbc_coordinates
    return [
        short_dof_name(
            solution.dof_descriptions[i],
            coordinates[i] if i < len(coordinates) else "",
        )
        for i in order.tolist()
    ]


def _phasor_traces(
    radius: npt.NDArray[np.float64],
    magnitude: npt.NDArray[np.float64],
    phase_deg: npt.NDArray[np.float64],
    labels: list[str],
) -> list[go.Scatterpolar]:
    """Build the arrow and marker polar traces for a set of phasors."""
    arrow_r: list[float | None] = []
    arrow_theta: list[float | None] = []
    for value, angle in zip(radius.tolist(), phase_deg.tolist(), strict=True):
        arrow_r.extend([0.0, value, None])
        arrow_theta.extend([angle, angle, None])
    arrows = go.Scatterpolar(
        r=arrow_r,
        theta=arrow_theta,
        mode="lines",
        line={"color": _SPOKE_COLOR},
        hoverinfo="skip",
        showlegend=False,
    )
    markers = go.Scatterpolar(
        r=radius,
        theta=phase_deg,
        mode="markers",
        marker={"size": 9, "color": _MARKER_COLOR},
        customdata=np.stack([magnitude, np.array(range(len(labels)))], axis=-1),
        text=labels,
        hovertemplate=(
            "%{text}<br>|participation|: %{customdata[0]:.3g}<br>"
            "Phase: %{theta:.0f}&deg;<extra></extra>"
        ),
        showlegend=False,
    )
    return [arrows, markers]


def _phasor_title(solution: ModalSolution, mode_index: int) -> str:
    """Title naming the mode and its natural frequency."""
    frequency = solution.natural_frequencies_hz[mode_index]
    return f"Mode {mode_index} phasors ({frequency:.4g} Hz)"
