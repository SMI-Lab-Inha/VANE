# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Visualization of the load quantity-of-interest map.

A :class:`~vane.sysid.qoi.QoiMap` records, for each structural load, how strongly
each identified mode contributes to it (the modal residue :math:`C_i \Phi`). This
view renders that as a **load-versus-mode heatmap**: rows are the load channels,
columns are the modes (labelled by natural frequency), and the colour is the
per-load normalized residue magnitude. It reads as a load-monitoring map — at a
glance, which modes drive which loads (e.g. the low-frequency platform modes that
dominate mooring tension versus the structural modes that dominate the yaw-bearing
moment). :func:`plot_load_mode_heatmap` is interactive (Plotly) and
:func:`plot_load_mode_heatmap_static` static (Matplotlib).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go

from vane.viz._style import (
    VANE_STYLE,
    new_axes,
    resolve_size,
    resolve_template,
    static_style,
)

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from vane.sysid.qoi import QoiMap, QoiSensitivity
    from vane.viz.publication import FigureSize

__all__ = ["plot_load_mode_heatmap", "plot_load_mode_heatmap_static"]

_COLORBAR_LABEL = "Normalized contribution (-)"
_MODE_AXIS_LABEL = "Mode frequency (Hz)"
_LOAD_AXIS_LABEL = "Load channel"


def _heatmap_data(
    qoi_map: QoiMap, min_normalized: float
) -> tuple[npt.NDArray[np.float64], list[str], list[str]]:
    """Return the kept matrix, the mode tick labels and the load names."""
    loads = qoi_map.loads()
    if not loads:
        msg = "QoiMap has no load channels to plot"
        raise ValueError(msg)
    if any(load.modal_residues is None for load in loads):
        msg = "QoiMap has no modal residues; build it with a ModalSolution"
        raise ValueError(msg)

    matrix = _normalized_matrix(loads)
    if min_normalized <= 0.0:
        # The documented escape hatch: keep every mode, including any that are
        # completely dormant (zero contribution to every load).
        keep = np.arange(matrix.shape[1])
    else:
        keep = np.flatnonzero(matrix.max(axis=0) > min_normalized)
        if keep.size == 0:
            keep = np.arange(matrix.shape[1])
    frequencies = qoi_map.mode_frequencies_hz
    mode_labels = [
        f"{float(frequencies[int(k)]):.3g}" if frequencies is not None else f"m{k}"
        for k in keep.tolist()
    ]
    return matrix[:, keep], mode_labels, [_load_label(load.name) for load in loads]


def _load_label(name: str) -> str:
    """Return a load channel name without its module prefix: ``"TwrBsMyt (kN-m)"``."""
    text = re.sub(r"^[A-Za-z]+\s+", "", name.strip(), count=1) or name
    return text.replace(", (", " (")


def plot_load_mode_heatmap(
    qoi_map: QoiMap,
    *,
    min_normalized: float = 0.05,
    template: str | go.layout.Template | None = None,
    title: str | None = None,
) -> go.Figure:
    """Plot a load-versus-mode modal-contribution heatmap.

    Parameters
    ----------
    qoi_map : QoiMap
        A map built with a modal solution, so each load carries modal residues.
    min_normalized : float, optional
        Keep only the modes whose per-load normalized contribution exceeds this
        threshold for at least one load, so dormant modes are dropped and the map
        stays legible. Set to ``0`` to keep every mode.
    template : str, plotly.graph_objects.layout.Template or None, optional
        Plotly layout template to apply. ``None`` (the default) applies the VANE
        template ``"vane"``.
    title : str or None, optional
        Figure title; ``None`` (the default) draws none.

    Returns
    -------
    plotly.graph_objects.Figure
        The heatmap (loads on the y-axis, contributing modes on the x-axis, labelled
        by natural frequency).

    Raises
    ------
    ValueError
        If the map has no load channels or was built without a modal solution.
    """
    matrix, mode_labels, load_labels = _heatmap_data(qoi_map, min_normalized)
    # Place the rows and columns at integer positions and label them with tick text:
    # categorical values would merge duplicate names (or equal rounded frequencies)
    # and let numeric-looking names turn the axis numeric.
    rows = list(range(len(load_labels)))
    cols = list(range(len(mode_labels)))
    customdata = np.array(
        [[[name, mode] for mode in mode_labels] for name in load_labels], dtype=object
    )

    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=cols,
            y=rows,
            customdata=customdata,
            colorscale="YlOrRd",
            zmin=0.0,
            zmax=1.0,
            colorbar={"title": {"text": _COLORBAR_LABEL, "side": "right"}},
            hovertemplate=(
                "Load: %{customdata[0]}<br>Mode: %{customdata[1]} Hz<br>"
                "Contribution: %{z:.3f}<extra></extra>"
            ),
        )
    )
    layout: dict[str, Any] = {
        "xaxis": {
            "title": {"text": _MODE_AXIS_LABEL},
            "tickmode": "array",
            "tickvals": cols,
            "ticktext": mode_labels,
        },
        "yaxis": {
            "title": {"text": _LOAD_AXIS_LABEL},
            "tickmode": "array",
            "tickvals": rows,
            "ticktext": load_labels,
        },
        "template": resolve_template(template),
    }
    if title is not None:
        layout["title"] = title
    figure.update_layout(**layout)
    return figure


def plot_load_mode_heatmap_static(
    qoi_map: QoiMap,
    *,
    min_normalized: float = 0.05,
    ax: Axes | None = None,
    style: str | dict[str, Any] | Path | None = VANE_STYLE,
    size: str | float | FigureSize | None = None,
    title: str | None = None,
) -> Figure:
    """Plot a load-versus-mode modal-contribution heatmap as a static figure.

    Parameters
    ----------
    qoi_map : QoiMap
        A map built with a modal solution, so each load carries modal residues.
    min_normalized : float, optional
        Keep only the modes whose per-load normalized contribution exceeds this
        threshold for at least one load. Set to ``0`` to keep every mode.
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
        The heatmap with a colour bar; modes on the x-axis labelled by natural
        frequency, load channels on the y-axis.

    Raises
    ------
    ValueError
        If the map has no load channels or was built without a modal solution.
    """
    matrix, mode_labels, load_labels = _heatmap_data(qoi_map, min_normalized)
    count = len(load_labels)
    aspect = (
        min(1.0, max(0.45, 0.1 * count + 0.3)),
        min(0.7, max(0.35, 0.06 * count + 0.2)),
    )
    resolved = resolve_size(size, "single", ax=ax, style=style, aspect=aspect)
    with static_style(style, resolved):
        figure, axes = new_axes(ax)
        image = axes.imshow(
            matrix, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto", origin="upper"
        )
        axes.set_xticks(range(len(mode_labels)), mode_labels, rotation=90)
        axes.set_yticks(range(len(load_labels)), load_labels)
        axes.tick_params(top=False, right=False)
        axes.set_xlabel(_MODE_AXIS_LABEL)
        axes.set_ylabel(_LOAD_AXIS_LABEL)
        colorbar = figure.colorbar(image, ax=axes)
        colorbar.set_label(_COLORBAR_LABEL)
        if title is not None:
            axes.set_title(title)
    return figure


def _normalized_matrix(
    loads: list[QoiSensitivity],
) -> npt.NDArray[np.float64]:
    """Return the per-load normalized residue-magnitude matrix (loads x modes)."""
    rows: list[npt.NDArray[np.float64]] = []
    for load in loads:
        residues = load.modal_residues
        if residues is None:  # pragma: no cover - caller guarantees residues exist
            continue
        magnitude = np.abs(residues)
        peak = float(magnitude.max()) if magnitude.size else 1.0
        rows.append(magnitude / (peak if peak > 0.0 else 1.0))
    return np.asarray(np.vstack(rows), dtype=np.float64)
