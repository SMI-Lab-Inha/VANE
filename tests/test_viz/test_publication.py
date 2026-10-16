# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.viz.publication — size presets, export and panel labels."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pytest

from vane.viz import (
    SIZE_PRESETS,
    FigureSize,
    add_panel_labels,
    figure_size,
    plot_campbell,
    plot_campbell_static,
    publication,
    save_figure,
)

if TYPE_CHECKING:
    from pathlib import Path

    from vane.campbell.builder import CampbellDiagram

_CM = 1.0 / 2.54


def _png_size(path: Path) -> tuple[int, int]:
    """Return the pixel width and height of a PNG file."""
    header = path.read_bytes()[16:24]
    width, height = struct.unpack(">II", header)
    return int(width), int(height)


class TestFigureSize:
    """The size presets and their fonts."""

    def test_presets(self) -> None:
        """Single column 8.5 cm and double column 17 cm, 9/8/8 pt fonts."""
        single = figure_size("single")
        double = figure_size("double")
        assert single.width_in == pytest.approx(8.5 * _CM)
        assert double.width_in == pytest.approx(17.0 * _CM)
        assert single.height_in == pytest.approx(8.5 * 0.75 * _CM)
        assert double.height_in == pytest.approx(17.0 * 0.47 * _CM)
        for size in (single, double):
            assert (size.label_pt, size.tick_pt, size.legend_pt) == (9.0, 8.0, 8.0)
        assert set(SIZE_PRESETS) == {"single", "double"}

    def test_explicit_widths(self) -> None:
        """A width in cm, with smaller fonts below 8 cm and a wide aspect above 12."""
        narrow = figure_size(6.0)
        assert narrow.width_cm == pytest.approx(6.0)
        assert (narrow.label_pt, narrow.tick_pt) == (8.0, 7.0)
        wide = figure_size(14.0)
        assert wide.height_cm == pytest.approx(14.0 * 0.47)
        assert figure_size(10.0, aspect=1.0).height_cm == pytest.approx(10.0)
        custom = FigureSize(5.0, 4.0)
        assert figure_size(custom) is custom

    def test_rc_sets_size_and_fonts(self) -> None:
        """The rcParams of a size carry the figure size and the fonts."""
        rc = figure_size("single").rc()
        assert rc["figure.figsize"][0] == pytest.approx(8.5 * _CM)
        assert rc["axes.labelsize"] == 9.0
        assert rc["xtick.labelsize"] == rc["legend.fontsize"] == 8.0

    @pytest.mark.parametrize("bad", ["column", 0.0, -3.0])
    def test_invalid_sizes(self, bad: str | float) -> None:
        """Unknown names and non-positive widths are rejected."""
        with pytest.raises(ValueError, match="figure"):
            figure_size(bad)

    def test_invalid_aspect(self) -> None:
        """A non-positive aspect is rejected."""
        with pytest.raises(ValueError, match="aspect"):
            figure_size("single", aspect=0.0)

    @pytest.mark.parametrize(("size", "width_cm"), [("single", 8.5), ("double", 17.0)])
    def test_plot_functions_take_the_size(
        self, campbell_diagram: CampbellDiagram, size: str, width_cm: float
    ) -> None:
        """A static plot drawn at a preset has its width and its fonts."""
        figure = plot_campbell_static(campbell_diagram, size=size)
        width, _ = figure.get_size_inches()
        assert width == pytest.approx(width_cm * _CM)
        axes = figure.axes[0]
        assert axes.xaxis.label.get_fontsize() == pytest.approx(9.0)
        tick = axes.xaxis.get_major_ticks()[0].label1
        assert tick.get_fontsize() == pytest.approx(8.0)
        plt.close(figure)


class TestSaveFigure:
    """Writing figures for publication."""

    def test_writes_pdf_svg_png_at_size_and_dpi(
        self, campbell_diagram: CampbellDiagram, tmp_path: Path
    ) -> None:
        """Without a suffix, PDF, SVG and PNG are written at the requested size."""
        figure = plot_campbell_static(campbell_diagram, size="single")
        written = save_figure(figure, tmp_path / "out" / "campbell", size="single")
        assert [path.suffix for path in written] == [".pdf", ".svg", ".png"]
        assert all(path.exists() and path.stat().st_size > 0 for path in written)
        size = figure_size("single")
        width, height = _png_size(written[2])
        assert width == pytest.approx(size.width_in * 600, abs=1)
        assert height == pytest.approx(size.height_in * 600, abs=1)
        assert written[0].read_bytes().startswith(b"%PDF")
        assert b"<svg" in written[1].read_bytes()
        plt.close(figure)

    def test_suffix_selects_one_format_and_dpi(
        self, campbell_diagram: CampbellDiagram, tmp_path: Path
    ) -> None:
        """A path with a suffix writes that file only, at the given dpi."""
        figure = plot_campbell_static(campbell_diagram, size=10.0)
        written = save_figure(figure, tmp_path / "c.png", dpi=100)
        assert written == [tmp_path / "c.png"]
        width, _ = _png_size(written[0])
        assert width == pytest.approx(10.0 * _CM * 100, abs=1)
        plt.close(figure)

    def test_unsupported_formats(self, tmp_path: Path) -> None:
        """Unknown suffixes and formats are rejected."""
        figure = plt.figure()
        with pytest.raises(ValueError, match="unsupported figure format"):
            save_figure(figure, tmp_path / "c.jpg")
        with pytest.raises(ValueError, match="unsupported figure format"):
            save_figure(figure, tmp_path / "c", formats=("tiff",))
        plt.close(figure)

    def test_other_objects_are_rejected(self, tmp_path: Path) -> None:
        """Only Matplotlib and Plotly figures are accepted."""
        with pytest.raises(TypeError, match="save_figure expects"):
            save_figure("figure", tmp_path / "c.png")  # type: ignore[arg-type]

    def test_plotly_without_kaleido(
        self,
        campbell_diagram: CampbellDiagram,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A Plotly figure without kaleido raises a clear ImportError."""
        monkeypatch.setattr(publication, "_kaleido_available", lambda: False)
        with pytest.raises(ImportError, match="kaleido"):
            save_figure(plot_campbell(campbell_diagram), tmp_path / "c.png")

    def test_plotly_with_kaleido_uses_the_preset(
        self,
        campbell_diagram: CampbellDiagram,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A Plotly figure is laid out at the preset and written at the dpi."""
        calls: list[dict[str, Any]] = []

        def fake_write(self: go.Figure, path: str, **kwargs: object) -> None:
            calls.append(
                {"path": path, "width": self.layout.width, "font": self.layout.font}
                | kwargs
            )

        monkeypatch.setattr(publication, "_kaleido_available", lambda: True)
        monkeypatch.setattr(go.Figure, "write_image", fake_write)
        figure = plot_campbell(campbell_diagram)
        written = save_figure(figure, tmp_path / "c", size="single")
        assert [path.suffix for path in written] == [".pdf", ".svg", ".png"]
        assert calls[0]["width"] == round(8.5 * _CM * 96)
        assert calls[0]["font"].size == pytest.approx(12.0)
        assert calls[2]["scale"] == pytest.approx(600 / 96)
        assert calls[0]["scale"] == pytest.approx(1.0)
        # The original figure is left untouched.
        assert figure.layout.width is None
        save_figure(figure, tmp_path / "d.svg")
        assert calls[-1]["width"] is None


class TestPanelLabels:
    """Panel labels of multi-panel figures."""

    def test_labels_outside_the_corner(self) -> None:
        """The panels get (a), (b), ... bold, right-aligned at the corner."""
        figure, grid = plt.subplots(1, 3)
        texts = add_panel_labels(list(grid))
        assert [text.get_text() for text in texts] == ["(a)", "(b)", "(c)"]
        assert all(text.get_fontweight() == "bold" for text in texts)
        assert add_panel_labels([grid[0]], ["A"])[0].get_text() == "A"
        with pytest.raises(ValueError, match="panel labels"):
            add_panel_labels(list(grid), ["(a)"])
        plt.close(figure)
