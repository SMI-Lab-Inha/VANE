# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Static mode-shape (participation) plotting.

Since OpenFAST mode shapes are expressed over modal degrees of freedom rather than
spatial coordinates, a mode is visualized as a horizontal bar chart of the signed
participation of its most active degrees of freedom: bar length is the relative
contribution and the sign encodes the in-/out-of-phase relationship to the dominant
degree of freedom. The plot is therefore modal *participation*, not spatial geometry —
hence the primary name :func:`plot_mode_participation`. Following the lab's mode-shape
conventions, the mode is named in the legend with its natural frequency, a thin grey
line marks zero, and ``short_names=True`` names the degrees of freedom briefly
(``"Tower FA1"``, ``"Flap 1 coll."``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from vane.modal.participation import participation_from_modes
from vane.viz._style import (
    STANDARD_COLORS,
    VANE_STYLE,
    new_axes,
    resolve_size,
    short_dof_name,
    static_style,
)

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from vane.modal.eigensolver import ModalSolution
    from vane.viz.publication import FigureSize

__all__ = ["plot_mode_participation", "plot_mode_shape"]

_BAR_COLOR = STANDARD_COLORS[2]


def plot_mode_participation(
    solution: ModalSolution,
    mode_index: int,
    *,
    top_n: int = 10,
    ax: Axes | None = None,
    color: str = _BAR_COLOR,
    style: str | dict[str, Any] | Path | None = VANE_STYLE,
    size: str | float | FigureSize | None = None,
    short_names: bool = False,
    title: str | None = None,
) -> Figure:
    """Plot the signed participation of a mode's most active degrees of freedom.

    Parameters
    ----------
    solution : ModalSolution
        A solution whose ``dof_descriptions`` are populated.
    mode_index : int
        Index of the mode to plot.
    top_n : int, optional
        Number of highest-participation degrees of freedom to show.
    ax : matplotlib.axes.Axes or None, optional
        Draw into these axes instead of creating a new figure (for composing a
        multi-panel report). The axes keep their own appearance; the parent figure is
        returned.
    color : str, optional
        Bar colour; blue, the third standard line colour, by default.
    style : str, dict, pathlib.Path or None, optional
        Matplotlib style in effect while drawing. ``"vane"`` (the default) is the VANE
        style (:func:`vane.viz.plot_style`); ``None`` keeps the current ``rcParams``;
        any other value is passed to :func:`matplotlib.style.context`.
    size : str, float, FigureSize or None, optional
        Figure size and fonts (see :func:`vane.viz.figure_size`); ``None`` uses
        ``"single"`` for a new figure.
    short_names : bool, optional
        Label the bars with short DOF names (``"Tower FA1"``, ``"Flap 1 coll."``)
        instead of the full state descriptions (the default). Short names suit a
        printed figure.
    title : str or None, optional
        Axes title; ``None`` (the default) draws none. The legend names the mode and
        its natural frequency.

    Returns
    -------
    matplotlib.figure.Figure
        The bar-chart figure.

    Raises
    ------
    ValueError
        If the solution has no DOF descriptions, ``mode_index`` is out of range, or
        ``top_n`` is not positive.
    """
    if not solution.dof_descriptions:
        msg = "ModalSolution has no dof_descriptions; cannot plot mode shape"
        raise ValueError(msg)
    if not 0 <= mode_index < solution.n_modes:
        msg = f"mode_index {mode_index} out of range [0, {solution.n_modes})"
        raise ValueError(msg)
    if top_n < 1:
        msg = f"top_n must be at least 1, got {top_n}"
        raise ValueError(msg)

    participation = participation_from_modes(solution)
    signed = participation.signed_magnitude[:, mode_index]
    order = np.argsort(np.abs(signed))[::-1][:top_n]
    coordinates = solution.dof_mbc_coordinates
    labels = [
        short_dof_name(
            solution.dof_descriptions[i],
            coordinates[i] if i < len(coordinates) else "",
        )
        if short_names
        else solution.dof_descriptions[i]
        for i in order.tolist()
    ]
    values = signed[order]
    frequency = float(solution.natural_frequencies_hz[mode_index])
    count = len(order)
    aspect = (
        min(1.0, max(0.5, 0.075 * count + 0.25)),
        min(0.6, max(0.3, 0.04 * count + 0.15)),
    )

    resolved = resolve_size(size, "single", ax=ax, style=style, aspect=aspect)
    with static_style(style, resolved):
        figure, axes = new_axes(ax)
        positions = np.arange(len(order))
        axes.barh(
            positions,
            values,
            color=color,
            height=0.7,
            label=f"Mode {mode_index} ({frequency:.4g} Hz)",
        )
        axes.set_yticks(positions)
        axes.set_yticklabels(labels)
        axes.invert_yaxis()
        axes.axvline(0.0, color="0.5", linewidth=0.6)
        limit = max(1.0, float(np.max(np.abs(values)))) if values.size else 1.0
        axes.set_xlim(-1.08 * limit, 1.08 * limit)
        axes.set_xlabel("Signed participation (-)")
        axes.legend(
            loc="lower left",
            bbox_to_anchor=(0.0, 1.02),
            borderaxespad=0.0,
            handlelength=1.2,
        )
        if title is not None:
            axes.set_title(title)
    return figure


def plot_mode_shape(
    solution: ModalSolution,
    mode_index: int,
    *,
    top_n: int = 10,
    ax: Axes | None = None,
    color: str = _BAR_COLOR,
    style: str | dict[str, Any] | Path | None = VANE_STYLE,
    size: str | float | FigureSize | None = None,
    short_names: bool = False,
    title: str | None = None,
) -> Figure:
    """Compatibility alias for :func:`plot_mode_participation`.

    The plot shows signed modal **participation**, not spatial mode-shape geometry;
    :func:`plot_mode_participation` is the accurate name. This alias forwards all
    arguments unchanged.
    """
    return plot_mode_participation(
        solution,
        mode_index,
        top_n=top_n,
        ax=ax,
        color=color,
        style=style,
        size=size,
        short_names=short_names,
        title=title,
    )
