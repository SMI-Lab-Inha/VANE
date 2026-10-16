# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.viz.vtk_export — legacy VTK deformed-mesh export."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

import numpy as np
import pytest

from vane.viz.vtk_export import write_mode_animation_vtk, write_mode_shape_vtk

if TYPE_CHECKING:
    from pathlib import Path

_POINTS = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
_DISP = np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]])
_IMAG = np.array([[0.0, 0.05, 0.0], [0.0, 0.05, 0.0], [0.0, 0.05, 0.0]])


def _section(lines: list[str], header: str) -> int:
    """Return the index of the line whose first token matches ``header``."""
    return next(i for i, line in enumerate(lines) if line.startswith(header))


def _read_points(path: Path) -> np.ndarray:
    """Read the deformed POINTS array (n_nodes, 3) from a legacy VTK file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = _section(lines, "POINTS 3 float") + 1
    return np.array([[float(v) for v in lines[start + i].split()] for i in range(3)])


class TestWriteModeShapeVtk:
    """Tests for write_mode_shape_vtk."""

    def test_writes_deformed_points(self, tmp_path: Path) -> None:
        """POINTS holds points + scale * displacement."""
        out = write_mode_shape_vtk(_POINTS, _DISP, tmp_path / "m.vtk", scale=2.0)
        lines = out.read_text(encoding="utf-8").splitlines()
        assert lines[0].startswith("# vtk DataFile Version")
        assert "DATASET POLYDATA" in lines
        start = _section(lines, "POINTS 3 float") + 1
        coords = np.array(
            [[float(v) for v in lines[start + i].split()] for i in range(3)]
        )
        np.testing.assert_allclose(coords, _POINTS + 2.0 * _DISP)

    def test_writes_displacement_vectors(self, tmp_path: Path) -> None:
        """A scaled displacement vector field is written as POINT_DATA."""
        out = write_mode_shape_vtk(_POINTS, _DISP, tmp_path / "m.vtk", scale=3.0)
        lines = out.read_text(encoding="utf-8").splitlines()
        assert f"POINT_DATA {len(_POINTS)}" in lines
        start = _section(lines, "VECTORS displacement float") + 1
        vectors = np.array(
            [[float(v) for v in lines[start + i].split()] for i in range(3)]
        )
        np.testing.assert_allclose(vectors, 3.0 * _DISP)

    def test_writes_line_connectivity(self, tmp_path: Path) -> None:
        """A polyline is written as a VTK LINES cell."""
        out = write_mode_shape_vtk(
            _POINTS, _DISP, tmp_path / "m.vtk", lines=[[0, 1, 2]]
        )
        text = out.read_text(encoding="utf-8")
        assert "LINES 1 4" in text  # one polyline, 3 nodes -> count word + 3 indices
        assert "3 0 1 2" in text

    def test_points_only_without_lines(self, tmp_path: Path) -> None:
        """Without connectivity no LINES section is written."""
        out = write_mode_shape_vtk(_POINTS, _DISP, tmp_path / "m.vtk")
        assert "LINES" not in out.read_text(encoding="utf-8")

    def test_rejects_non_3d_points(self, tmp_path: Path) -> None:
        """Points must be (n_nodes, 3)."""
        with pytest.raises(ValueError, match=r"n_nodes, 3"):
            write_mode_shape_vtk(np.zeros((3, 2)), np.zeros((3, 2)), tmp_path / "m.vtk")

    def test_rejects_mismatched_displacement(self, tmp_path: Path) -> None:
        """Displacement must match the points shape."""
        with pytest.raises(ValueError, match="match points"):
            write_mode_shape_vtk(np.zeros((3, 3)), np.zeros((2, 3)), tmp_path / "m.vtk")

    def test_rejects_out_of_range_line_index(self, tmp_path: Path) -> None:
        """A line referencing a non-existent node is rejected."""
        with pytest.raises(ValueError, match="node index"):
            write_mode_shape_vtk(_POINTS, _DISP, tmp_path / "m.vtk", lines=[[0, 9]])

    def test_rejects_non_finite_displacement(self, tmp_path: Path) -> None:
        """A non-finite displacement is rejected."""
        bad = _DISP.copy()
        bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="displacement must be finite"):
            write_mode_shape_vtk(_POINTS, bad, tmp_path / "m.vtk")

    def test_rejects_non_finite_points(self, tmp_path: Path) -> None:
        """A non-finite node coordinate is rejected."""
        bad = _POINTS.copy()
        bad[1, 2] = np.inf
        with pytest.raises(ValueError, match="points must be finite"):
            write_mode_shape_vtk(bad, _DISP, tmp_path / "m.vtk")

    def test_rejects_complex_displacement(self, tmp_path: Path) -> None:
        """A genuinely complex mode is rejected instead of losing its imaginary part."""
        with pytest.raises(ValueError, match="imaginary part"):
            write_mode_shape_vtk(_POINTS, _DISP + 1j * _IMAG, tmp_path / "m.vtk")

    @pytest.mark.parametrize("scale", [np.inf, -np.inf, np.nan])
    def test_rejects_non_finite_scale(self, scale: float, tmp_path: Path) -> None:
        """A non-finite scale is rejected instead of writing inf/nan coordinates."""
        with pytest.raises(ValueError, match="scale must be finite"):
            write_mode_shape_vtk(_POINTS, _DISP, tmp_path / "m.vtk", scale=scale)

    def test_multiline_title_stays_on_the_header_line(self, tmp_path: Path) -> None:
        """Line breaks in the title cannot displace the ASCII/DATASET keywords."""
        path = write_mode_shape_vtk(
            _POINTS, _DISP, tmp_path / "m.vtk", title="line1\nline2\r\nline3"
        )
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[1] == "line1 line2  line3"
        assert lines[2] == "ASCII"
        assert lines[3] == "DATASET POLYDATA"

    def test_long_title_is_truncated_to_the_header_limit(self, tmp_path: Path) -> None:
        """The header line is capped so it fits the 256-character legacy limit."""
        path = write_mode_shape_vtk(_POINTS, _DISP, tmp_path / "m.vtk", title="x" * 400)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines[1]) == 255
        assert lines[2] == "ASCII"

    def test_animation_frames_sanitize_the_title(self, tmp_path: Path) -> None:
        """Every animation frame gets the same single-line header."""
        frames = write_mode_animation_vtk(
            _POINTS, _DISP, tmp_path / "m.pvd", n_frames=2, title="a\nb"
        )
        for frame in frames:
            lines = frame.read_text(encoding="utf-8").splitlines()
            assert lines[1] == "a b"
            assert lines[2] == "ASCII"

    def test_rejects_non_finite_imaginary_part(self, tmp_path: Path) -> None:
        """A NaN hidden in the imaginary part is not dropped with it."""
        bad = _DISP + 0j
        bad[0, 1] = complex(0.0, np.nan)
        with pytest.raises(ValueError, match="displacement must be finite"):
            write_mode_shape_vtk(_POINTS, bad, tmp_path / "m.vtk")

    def test_accepts_complex_with_round_off_imaginary_part(
        self, tmp_path: Path
    ) -> None:
        """A complex array whose imaginary part is round-off is written as real."""
        path = write_mode_shape_vtk(_POINTS, _DISP + 1e-14j * _IMAG, tmp_path / "m.vtk")
        np.testing.assert_allclose(_read_points(path), _POINTS + _DISP, atol=1e-9)

    def test_phase_rotated_real_part_is_writable(self, tmp_path: Path) -> None:
        """The documented recipe writes the motion at a chosen phase."""
        complex_disp = _DISP + 1j * _IMAG
        field = np.real(complex_disp * np.exp(1j * np.pi / 2))
        path = write_mode_shape_vtk(_POINTS, field, tmp_path / "m.vtk")
        np.testing.assert_allclose(_read_points(path), _POINTS - _IMAG, atol=1e-9)


class TestWriteModeAnimationVtk:
    """Tests for write_mode_animation_vtk (a VTK .pvd time series)."""

    def test_writes_one_file_per_frame_and_a_pvd(self, tmp_path: Path) -> None:
        """n_frames VTK files are written plus a .pvd collection at the given path."""
        pvd = tmp_path / "m.pvd"
        frames = write_mode_animation_vtk(_POINTS, _DISP, pvd, n_frames=6)
        assert len(frames) == 6
        assert all(frame.exists() for frame in frames)
        assert frames[0].name == "m.0000.vtk"
        assert frames[5].name == "m.0005.vtk"
        assert pvd.exists()

    def test_pvd_is_a_collection_referencing_every_frame(self, tmp_path: Path) -> None:
        """The .pvd is a VTK Collection with one timestepped DataSet per frame."""
        pvd = tmp_path / "m.pvd"
        write_mode_animation_vtk(_POINTS, _DISP, pvd, n_frames=4)
        text = pvd.read_text(encoding="utf-8")
        assert 'type="Collection"' in text
        assert text.count("<DataSet ") == 4
        assert 'timestep="0"' in text
        assert 'file="m.0000.vtk"' in text
        assert 'file="m.0003.vtk"' in text

    def test_first_frame_is_the_real_part(self, tmp_path: Path) -> None:
        """Frame 0 (phase 0) is points + scale * Re(displacement)."""
        frames = write_mode_animation_vtk(
            _POINTS, _DISP + 1j * _IMAG, tmp_path / "m.pvd", n_frames=4, scale=2.0
        )
        np.testing.assert_allclose(_read_points(frames[0]), _POINTS + 2.0 * _DISP)

    def test_quarter_period_is_minus_the_imaginary_part(self, tmp_path: Path) -> None:
        """At a quarter period the motion is -scale * Im(displacement)."""
        frames = write_mode_animation_vtk(
            _POINTS, _DISP + 1j * _IMAG, tmp_path / "m.pvd", n_frames=4
        )
        np.testing.assert_allclose(_read_points(frames[1]), _POINTS - _IMAG, atol=1e-9)

    def test_half_period_reverses_a_real_mode(self, tmp_path: Path) -> None:
        """At half a period a real mode is fully reversed (points - scale * disp)."""
        frames = write_mode_animation_vtk(
            _POINTS, _DISP, tmp_path / "m.pvd", n_frames=4
        )
        np.testing.assert_allclose(_read_points(frames[2]), _POINTS - _DISP, atol=1e-9)

    def test_lines_are_written_into_each_frame(self, tmp_path: Path) -> None:
        """Connectivity is written into every frame."""
        frames = write_mode_animation_vtk(
            _POINTS, _DISP, tmp_path / "m.pvd", n_frames=3, lines=[[0, 1, 2]]
        )
        for frame in frames:
            assert "LINES 1 4" in frame.read_text(encoding="utf-8")

    def test_default_frame_count(self, tmp_path: Path) -> None:
        """The default sweep uses 24 frames."""
        frames = write_mode_animation_vtk(_POINTS, _DISP, tmp_path / "m.pvd")
        assert len(frames) == 24

    def test_pvd_is_valid_xml_with_metacharacter_filename(self, tmp_path: Path) -> None:
        """A stem with XML metacharacters yields well-formed, parseable .pvd XML."""
        pvd = tmp_path / "mode & tower.pvd"
        frames = write_mode_animation_vtk(_POINTS, _DISP, pvd, n_frames=2)
        tree = ET.parse(pvd)  # raises on malformed XML (the unescaped-'&' bug)
        # The parser un-escapes the attribute back to the real on-disk filename.
        files = [dataset.get("file") for dataset in tree.iter("DataSet")]
        assert files == [frame.name for frame in frames]
        assert files[0] is not None and "&" in files[0]

    @pytest.mark.parametrize("bad", [0, -1, 2.5])
    def test_rejects_bad_frame_count(self, bad: float, tmp_path: Path) -> None:
        """A non-positive or fractional frame count is rejected."""
        with pytest.raises(ValueError, match="n_frames"):
            write_mode_animation_vtk(
                _POINTS,
                _DISP,
                tmp_path / "m.pvd",
                n_frames=bad,  # type: ignore[arg-type]
            )

    def test_rejects_non_finite_scale(self, tmp_path: Path) -> None:
        """A non-finite scale is rejected."""
        with pytest.raises(ValueError, match="scale must be finite"):
            write_mode_animation_vtk(_POINTS, _DISP, tmp_path / "m.pvd", scale=np.inf)

    def test_rejects_mismatched_displacement(self, tmp_path: Path) -> None:
        """Displacement must match the points shape."""
        with pytest.raises(ValueError, match="match points"):
            write_mode_animation_vtk(
                np.zeros((3, 3)), np.zeros((2, 3)), tmp_path / "m.pvd"
            )

    def test_rejects_out_of_range_line_index(self, tmp_path: Path) -> None:
        """A line referencing a non-existent node is rejected."""
        with pytest.raises(ValueError, match="node index"):
            write_mode_animation_vtk(_POINTS, _DISP, tmp_path / "m.pvd", lines=[[0, 9]])
