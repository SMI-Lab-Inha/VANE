# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Three-dimensional complex mode-shape plotting.

A complex mode shape carries both magnitude and phase per degree of freedom. This
view renders the most active degrees of freedom as phasors in three dimensions:
the horizontal axis orders the degrees of freedom by participation while the other
two axes are the real and imaginary parts of each component, so the spread out of
the real plane shows how non-proportional (complex) the mode is.
:func:`plot_mode_3d` is interactive (Plotly) and :func:`plot_mode_3d_static` static
(Matplotlib).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go

from vane.viz._style import (
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

    from vane.modal.eigensolver import ModalSolution
    from vane.viz.publication import FigureSize

__all__ = ["plot_mode_3d", "plot_mode_3d_static"]

_RANK_LABEL = "DOF rank (-)"
_REAL_LABEL = "Real part (-)"
_IMAG_LABEL = "Imaginary part (-)"


def _top_dofs(
    solution: ModalSolution, mode_index: int, top_n: int
) -> tuple[npt.NDArray[np.complex128], list[int]]:
    """Return the mode shape and the indices of its ``top_n`` largest components."""
    if not solution.dof_descriptions:
        msg = "ModalSolution has no dof_descriptions; cannot plot mode shape"
        raise ValueError(msg)
    if not 0 <= mode_index < solution.n_modes:
        msg = f"mode_index {mode_index} out of range [0, {solution.n_modes})"
        raise ValueError(msg)
    if top_n < 1:
        msg = f"top_n must be at least 1, got {top_n}"
        raise ValueError(msg)
    shape = np.asarray(solution.mode_shapes[:, mode_index], dtype=np.complex128)
    order = np.argsort(np.abs(shape))[::-1][:top_n]
    return shape, [int(k) for k in order.tolist()]


def plot_mode_3d(
    solution: ModalSolution,
    mode_index: int,
    *,
    top_n: int = 12,
    template: str | go.layout.Template | None = None,
    title: str | None = None,
) -> go.Figure:
    """Plot a mode's most active degrees of freedom as 3-D complex phasors.

    Parameters
    ----------
    solution : ModalSolution
        A solution whose ``dof_descriptions`` are populated.
    mode_index : int
        Index of the mode to plot.
    top_n : int, optional
        Number of highest-magnitude degrees of freedom to show.
    template : str, plotly.graph_objects.layout.Template or None, optional
        Plotly layout template to apply. ``None`` (the default) applies the VANE
        template ``"vane"``.
    title : str or None, optional
        Figure title. ``None`` (the default) names the mode and its natural
        frequency; ``""`` draws none.

    Returns
    -------
    plotly.graph_objects.Figure
        The 3-D phasor figure.

    Raises
    ------
    ValueError
        If the solution has no DOF descriptions, ``mode_index`` is out of range, or
        ``top_n`` is not positive.
    """
    shape, order = _top_dofs(solution, mode_index, top_n)
    figure = go.Figure()
    for rank, dof in enumerate(order):
        component = shape[dof]
        figure.add_trace(
            go.Scatter3d(
                x=[float(rank), float(rank)],
                y=[0.0, float(component.real)],
                z=[0.0, float(component.imag)],
                mode="lines+markers",
                name=solution.dof_descriptions[dof],
            )
        )
    frequency = solution.natural_frequencies_hz[mode_index]
    default = f"Mode {mode_index} ({frequency:.4g} Hz)"
    figure.update_layout(
        title=default if title is None else title,
        scene={
            "xaxis_title": _RANK_LABEL,
            "yaxis_title": _REAL_LABEL,
            "zaxis_title": _IMAG_LABEL,
        },
        template=resolve_template(template),
    )
    return figure


def plot_mode_3d_static(
    solution: ModalSolution,
    mode_index: int,
    *,
    top_n: int = 6,
    style: str | dict[str, Any] | Path | None = VANE_STYLE,
    size: str | float | FigureSize | None = None,
    title: str | None = None,
) -> Figure:
    """Plot a mode's most active degrees of freedom as static 3-D complex phasors.

    Each degree of freedom is a stem from the axis of DOF ranks to its complex
    component (real and imaginary parts), in its own colour; the legend names the
    DOFs and its title the mode and its natural frequency.

    Parameters
    ----------
    solution : ModalSolution
        A solution whose ``dof_descriptions`` are populated.
    mode_index : int
        Index of the mode to plot.
    top_n : int, optional
        Number of highest-magnitude degrees of freedom to show (default 6).
    style : str, dict, pathlib.Path or None, optional
        Matplotlib style in effect while drawing (``"vane"`` by default).
    size : str, float, FigureSize or None, optional
        Figure size and fonts; ``None`` uses ``"single"``.
    title : str or None, optional
        Axes title; ``None`` (the default) draws none.

    Returns
    -------
    matplotlib.figure.Figure
        The 3-D figure.

    Raises
    ------
    ValueError
        If the solution has no DOF descriptions, ``mode_index`` is out of range, or
        ``top_n`` is not positive.
    """
    import matplotlib.pyplot as plt

    shape, order = _top_dofs(solution, mode_index, top_n)
    colors = series_colors(len(order))
    coordinates = solution.dof_mbc_coordinates
    resolved = resolve_size(size, "single", ax=None, style=style, aspect=(0.85, 0.55))
    with static_style(style, resolved):
        figure = plt.figure(layout="constrained")
        axes = figure.add_subplot(projection="3d")
        for rank, (dof, color) in enumerate(zip(order, colors, strict=True)):
            component = shape[dof]
            name = short_dof_name(
                solution.dof_descriptions[dof],
                coordinates[dof] if dof < len(coordinates) else "",
            )
            axes.plot(
                [rank, rank],
                [0.0, float(component.real)],
                [0.0, float(component.imag)],
                color=color,
                linewidth=1.5,
                marker="o",
                markevery=[1],
                label=name,
            )
        axes.plot(
            [0.0, max(len(order) - 1, 1)], [0.0, 0.0], [0.0, 0.0], color="0.5", lw=0.6
        )
        axes.set_xlabel(_RANK_LABEL)
        axes.set_ylabel(_REAL_LABEL)
        axes.set_zlabel(_IMAG_LABEL)
        axes.set_xticks(range(len(order)))
        axes.view_init(elev=20.0, azim=-60.0)
        frequency = float(solution.natural_frequencies_hz[mode_index])
        axes.set_box_aspect(None, zoom=0.82)
        figure.legend(
            loc="outside upper center",
            ncols=min(3, len(order)),
            title=f"Mode {mode_index} ({frequency:.4g} Hz)",
        )
        if title is not None:
            axes.set_title(title)
    return figure
