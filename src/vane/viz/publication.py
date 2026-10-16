# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Publication figure sizes, export and layout helpers.

Journal figures are drawn at the width they are printed: a single column is about
8.5 cm wide and a double column (full page width) about 17 cm. :func:`figure_size`
turns a size request (``"single"``, ``"double"`` or a width in centimetres) into a
:class:`FigureSize` with the figure size and the matching font sizes (axis labels 9 pt,
ticks and legends 8 pt), so that a figure is readable without scaling. Every static
plot function of :mod:`vane.viz` takes such a ``size=`` argument.

:func:`save_figure` writes a figure at a size preset: vector PDF and SVG, and PNG at
600 dpi, for Matplotlib figures; static images for Plotly figures when the optional
``kaleido`` package is installed. :func:`add_panel_labels` puts ``(a)``, ``(b)``, ...
outside the top-left corner of the panels of a multi-panel figure.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import plotly.graph_objects as go
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.text import Text

__all__ = [
    "SIZE_PRESETS",
    "FigureSize",
    "add_panel_labels",
    "figure_size",
    "save_figure",
]

_CM_PER_INCH = 2.54
# Plotly lays out in CSS pixels (96 per inch); fonts are given in pixels.
_PLOTLY_PX_PER_INCH = 96.0
_PX_PER_PT = _PLOTLY_PX_PER_INCH / 72.0
# Widths (cm) below which the fonts step down by 1 pt, and the default aspect ratios.
_NARROW_CM = 8.0
_WIDE_CM = 12.0
_ASPECT_NARROW = 0.75
_ASPECT_WIDE = 0.47

_FORMATS = ("pdf", "svg", "png", "eps")


@dataclass(frozen=True)
class FigureSize:
    """A printed figure size with the matching font sizes.

    Parameters
    ----------
    width_cm : float
        Figure width in centimetres.
    height_cm : float
        Figure height in centimetres.
    label_pt : float, optional
        Font size of axis labels and titles, in points.
    tick_pt : float, optional
        Font size of tick labels, in points.
    legend_pt : float, optional
        Font size of legends and direct labels, in points.
    """

    width_cm: float
    height_cm: float
    label_pt: float = 9.0
    tick_pt: float = 8.0
    legend_pt: float = 8.0

    @property
    def width_in(self) -> float:
        """Figure width in inches."""
        return self.width_cm / _CM_PER_INCH

    @property
    def height_in(self) -> float:
        """Figure height in inches."""
        return self.height_cm / _CM_PER_INCH

    def rc(self) -> dict[str, Any]:
        """Return the Matplotlib ``rcParams`` of this size (figure size and fonts).

        Returns
        -------
        dict[str, Any]
            ``figure.figsize`` and the font sizes of labels, titles, ticks and
            legends.
        """
        return {
            "figure.figsize": (self.width_in, self.height_in),
            "font.size": self.label_pt,
            "axes.labelsize": self.label_pt,
            "axes.titlesize": self.label_pt,
            "figure.titlesize": self.label_pt,
            "xtick.labelsize": self.tick_pt,
            "ytick.labelsize": self.tick_pt,
            "legend.fontsize": self.legend_pt,
            "legend.title_fontsize": self.legend_pt,
        }


#: The named size presets: one journal column (8.5 cm) and the full page width (17 cm).
SIZE_PRESETS: dict[str, FigureSize] = {
    "single": FigureSize(8.5, 8.5 * _ASPECT_NARROW),
    "double": FigureSize(17.0, 17.0 * _ASPECT_WIDE),
}


def figure_size(
    size: str | float | FigureSize, *, aspect: float | None = None
) -> FigureSize:
    """Resolve a size request into a :class:`FigureSize`.

    Parameters
    ----------
    size : str, float or FigureSize
        ``"single"`` (8.5 cm wide), ``"double"`` (17 cm wide), a width in centimetres,
        or a :class:`FigureSize`, returned unchanged unless ``aspect`` is given.
    aspect : float or None, optional
        Height-to-width ratio. ``None`` uses 0.75 below 12 cm width and 0.47 above
        (the presets' ratios).

    Returns
    -------
    FigureSize
        Width, height and font sizes. Below 8 cm width the fonts are 1 pt smaller
        (8 pt labels, 7 pt ticks and legends).

    Raises
    ------
    ValueError
        If ``size`` is an unknown name or a non-positive width, or ``aspect`` is not
        positive.
    """
    if aspect is not None and not aspect > 0.0:
        msg = f"aspect must be positive, got {aspect}"
        raise ValueError(msg)
    if isinstance(size, FigureSize):
        resolved = size
    elif isinstance(size, str):
        if size not in SIZE_PRESETS:
            names = ", ".join(repr(name) for name in SIZE_PRESETS)
            msg = f"unknown figure size {size!r}; use {names} or a width in cm"
            raise ValueError(msg)
        resolved = SIZE_PRESETS[size]
    else:
        width = float(size)
        if not width > 0.0:
            msg = f"figure width must be positive, got {size} cm"
            raise ValueError(msg)
        ratio = _ASPECT_NARROW if width < _WIDE_CM else _ASPECT_WIDE
        if width < _NARROW_CM:
            resolved = FigureSize(width, width * ratio, 8.0, 7.0, 7.0)
        else:
            resolved = FigureSize(width, width * ratio)
    if aspect is not None:
        resolved = replace(resolved, height_cm=resolved.width_cm * aspect)
    return resolved


def add_panel_labels(
    axes: Sequence[Axes], labels: Sequence[str] | None = None
) -> list[Text]:
    """Label the panels of a figure ``(a)``, ``(b)``, ... outside their top-left corner.

    Parameters
    ----------
    axes : Sequence[matplotlib.axes.Axes]
        The panels, in reading order.
    labels : Sequence[str] or None, optional
        The labels; ``None`` gives ``(a)``, ``(b)``, ``(c)``, ...

    Returns
    -------
    list[matplotlib.text.Text]
        The label texts.

    Raises
    ------
    ValueError
        If ``labels`` does not have one entry per panel.
    """
    names = (
        [f"({chr(ord('a') + index)})" for index in range(len(axes))]
        if labels is None
        else list(labels)
    )
    if len(names) != len(axes):
        msg = f"got {len(names)} panel labels for {len(axes)} panels"
        raise ValueError(msg)
    return [
        panel.annotate(
            name,
            xy=(0.0, 1.0),
            xycoords="axes fraction",
            xytext=(-3.0, 4.0),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontweight="bold",
        )
        for panel, name in zip(axes, names, strict=True)
    ]


def _kaleido_available() -> bool:
    """Return whether the optional ``kaleido`` package (Plotly export) is installed."""
    return importlib.util.find_spec("kaleido") is not None


def _output_paths(path: Path, formats: Sequence[str]) -> list[Path]:
    """Return the files to write: the path's own suffix, or one per format."""
    if path.suffix:
        suffix = path.suffix.lower().lstrip(".")
        if suffix not in _FORMATS:
            msg = f"unsupported figure format {path.suffix!r}; use one of {_FORMATS}"
            raise ValueError(msg)
        return [path]
    unknown = [fmt for fmt in formats if fmt.lower() not in _FORMATS]
    if unknown:
        msg = f"unsupported figure format(s) {unknown}; use one of {_FORMATS}"
        raise ValueError(msg)
    return [path.with_suffix(f".{fmt.lower()}") for fmt in formats]


def _plotly_sized(figure: go.Figure, size: FigureSize) -> go.Figure:
    """Return a copy of a Plotly figure laid out at a printed size."""
    sized = go.Figure(figure)
    tick = {"tickfont": {"size": size.tick_pt * _PX_PER_PT}}
    sized.update_layout(
        width=round(size.width_in * _PLOTLY_PX_PER_INCH),
        height=round(size.height_in * _PLOTLY_PX_PER_INCH),
        font={"size": size.label_pt * _PX_PER_PT},
        legend={"font": {"size": size.legend_pt * _PX_PER_PT}},
        margin={"l": 50, "r": 10, "t": 30, "b": 45},
    )
    sized.update_xaxes(**tick)
    sized.update_yaxes(**tick)
    return sized


def save_figure(
    figure: Figure | go.Figure,
    path: str | Path,
    *,
    size: str | float | FigureSize | None = None,
    dpi: int = 600,
    formats: Sequence[str] = ("pdf", "svg", "png"),
) -> list[Path]:
    """Write a figure for publication at a size preset.

    Parameters
    ----------
    figure : matplotlib.figure.Figure or plotly.graph_objects.Figure
        The figure to write.
    path : str or pathlib.Path
        Destination. With a suffix (``.pdf``, ``.svg``, ``.png`` or ``.eps``) only that
        file is written; without one, one file per entry of ``formats``. Parent
        directories are created.
    size : str, float, FigureSize or None, optional
        Size preset (see :func:`figure_size`). ``None`` keeps the figure's own size.
        The file has exactly this size (the page is not trimmed to the content).
        A Matplotlib figure is resized; its fonts keep the sizes it was drawn with,
        so pass the same ``size=`` to the plot function to get matching fonts. A
        Plotly figure is laid out at the preset size and fonts.
    dpi : int, optional
        Resolution of raster (PNG) output. Default 600.
    formats : Sequence[str], optional
        Formats written when ``path`` has no suffix. Default PDF, SVG and PNG.

    Returns
    -------
    list[pathlib.Path]
        The files written.

    Raises
    ------
    ValueError
        If a format is not supported or the size is invalid.
    ImportError
        If ``figure`` is a Plotly figure and the ``kaleido`` package, which Plotly
        needs for static images, is not installed.
    TypeError
        If ``figure`` is neither a Matplotlib nor a Plotly figure.
    """
    targets = _output_paths(Path(path), formats)
    resolved = None if size is None else figure_size(size)
    if isinstance(figure, Figure):
        if resolved is not None:
            figure.set_size_inches(resolved.width_in, resolved.height_in)
        # The whole figure, at exactly its size: a "tight" bounding box (the style's
        # default for savefig) would change the printed size.
        width, height = figure.get_size_inches()
        page = Bbox.from_bounds(0.0, 0.0, float(width), float(height))
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(
                target,
                dpi=dpi,
                format=target.suffix.lstrip(".").lower(),
                bbox_inches=page,
            )
        return targets
    if isinstance(figure, go.Figure):
        if not _kaleido_available():
            msg = (
                "static export of Plotly figures needs the optional 'kaleido' "
                "package; install it with: pip install kaleido"
            )
            raise ImportError(msg)
        sized = figure if resolved is None else _plotly_sized(figure, resolved)
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
            fmt = target.suffix.lstrip(".").lower()
            scale = dpi / _PLOTLY_PX_PER_INCH if fmt == "png" else 1.0
            sized.write_image(str(target), format=fmt, scale=scale)
        return targets
    msg = (
        "save_figure expects a Matplotlib or Plotly figure, got "
        f"{type(figure).__name__}"
    )
    raise TypeError(msg)
