# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Export a mode shape as a deformed mesh in the legacy VTK format.

A mode shape is most legible in 3-D over the real structure. This writer turns a mode's
nodal displacements and the undeformed node coordinates into a legacy ASCII VTK
``POLYDATA`` file — the deformed node positions, the displacement vector field, and an
optional line connectivity — that opens directly in a VTK viewer and can be compared
against OpenFAST's own ``-VTKLin`` linearization output.

:func:`write_mode_animation_vtk` extends this to a VTK-native **animation**: it
sweeps a (real or complex) mode shape through one oscillation period into one VTK file
per frame plus a ``.pvd`` collection, so a VTK viewer plays the mode's motion directly
rather than the deformed shape being a single still frame. Both write the format by
hand, so they need no ``vtk`` package.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from xml.sax.saxutils import quoteattr

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

__all__ = ["write_mode_animation_vtk", "write_mode_shape_vtk"]

# The legacy VTK header line holds at most 256 characters including its newline.
_MAX_TITLE_CHARS = 255

# Relative size of an imaginary part, against the largest displacement magnitude,
# below which a complex displacement is treated as real (round-off from a solver).
_IMAG_RTOL = 1e-9


def _sanitize_title(title: str) -> str:
    """Return ``title`` as a single legacy-VTK header line.

    The header is one line of at most 256 characters; an embedded line break would
    push the ``ASCII`` / ``DATASET`` keywords out of place and corrupt the file, so
    every line break or other non-printable character becomes a space and the result
    is truncated to fit.
    """
    single_line = "".join(char if char.isprintable() else " " for char in title)
    return single_line[:_MAX_TITLE_CHARS]


def _real_displacement(displacement: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return a real displacement array, rejecting a genuinely complex one.

    A complex input whose imaginary part is only round-off (relative to the largest
    displacement magnitude) is accepted as its real part; anything larger raises,
    because a single static frame cannot represent a complex (phase-varying) mode.
    """
    values = np.asarray(displacement)
    if not np.iscomplexobj(values):
        return np.asarray(values, dtype=np.float64)
    values = np.asarray(values, dtype=np.complex128)
    if not np.all(np.isfinite(values)):
        msg = "displacement must be finite (no NaN or inf)"
        raise ValueError(msg)
    imag = np.abs(values.imag)
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    if values.size and float(np.max(imag)) > _IMAG_RTOL * peak:
        msg = (
            "displacement is complex with a non-negligible imaginary part; a static "
            "frame needs a real field. Pass the real motion at a chosen phase, "
            "np.real(displacement * np.exp(1j * phase)), or animate the full "
            "complex mode with write_mode_animation_vtk"
        )
        raise ValueError(msg)
    return np.asarray(values.real, dtype=np.float64)


def _validate_node_shapes(
    node_points: npt.NDArray[np.float64],
    node_displacement: npt.NDArray[np.float64] | npt.NDArray[np.complex128],
) -> None:
    """Validate node coordinates and displacement: both ``(n_nodes, 3)`` and finite.

    ``node_points`` is real; ``node_displacement`` matches its shape and may be complex
    (a complex mode shape, animated as its real orbit). Each caller converts to its own
    dtype first, so this only checks shapes and finiteness.
    """
    if node_points.ndim != 2 or node_points.shape[1] != 3:
        msg = f"points must have shape (n_nodes, 3), got {node_points.shape}"
        raise ValueError(msg)
    if node_displacement.shape != node_points.shape:
        msg = (
            f"displacement must match points shape {node_points.shape}, "
            f"got {node_displacement.shape}"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(node_points)):
        msg = "points must be finite (no NaN or inf)"
        raise ValueError(msg)
    if not np.all(np.isfinite(node_displacement)):
        msg = "displacement must be finite (no NaN or inf)"
        raise ValueError(msg)


def _validate_lines(lines: Sequence[Sequence[int]], n_nodes: int) -> list[list[int]]:
    """Return the polylines as index lists, rejecting any out-of-range node index."""
    polylines = [list(line) for line in lines]
    for line in polylines:
        if any(not 0 <= index < n_nodes for index in line):
            msg = f"a line references a node index outside [0, {n_nodes})"
            raise ValueError(msg)
    return polylines


def _polydata_text(
    deformed: npt.NDArray[np.float64],
    field: npt.NDArray[np.float64],
    polylines: list[list[int]] | None,
    title: str,
) -> str:
    """Serialize deformed points + a displacement field as legacy VTK POLYDATA."""
    n_nodes = deformed.shape[0]
    out_lines = [
        "# vtk DataFile Version 3.0",
        _sanitize_title(title),
        "ASCII",
        "DATASET POLYDATA",
        f"POINTS {n_nodes} float",
        *(f"{x:.9g} {y:.9g} {z:.9g}" for x, y, z in deformed),
    ]
    if polylines is not None:
        total = sum(len(line) + 1 for line in polylines)
        out_lines.append(f"LINES {len(polylines)} {total}")
        out_lines.extend(
            " ".join(str(value) for value in (len(line), *line)) for line in polylines
        )
    out_lines.append(f"POINT_DATA {n_nodes}")
    out_lines.append("VECTORS displacement float")
    out_lines.extend(f"{u:.9g} {v:.9g} {w:.9g}" for u, v, w in field)
    return "\n".join(out_lines) + "\n"


def write_mode_shape_vtk(
    points: npt.ArrayLike,
    displacement: npt.ArrayLike,
    path: str | Path,
    *,
    scale: float = 1.0,
    lines: Sequence[Sequence[int]] | None = None,
    title: str = "VANE mode shape",
) -> Path:
    """Write a mode's deformed mesh to a legacy ASCII VTK ``POLYDATA`` file.

    Parameters
    ----------
    points : npt.ArrayLike
        Undeformed node coordinates, shape ``(n_nodes, 3)``.
    displacement : npt.ArrayLike
        Real nodal displacement vectors of the mode, shape ``(n_nodes, 3)``. A complex
        array is accepted only when its imaginary part is round-off (below ``1e-9``
        of the largest displacement magnitude); a genuinely complex mode is rejected
        rather than silently reduced to its real part. To write the motion at phase
        ``phase``, pass ``np.real(displacement * np.exp(1j * phase))``; to show the
        whole oscillation, use :func:`write_mode_animation_vtk`.
    path : str or pathlib.Path
        Destination ``.vtk`` file path.
    scale : float, optional
        Factor applied to ``displacement`` before adding it to ``points`` to form the
        deformed geometry (visual exaggeration). Must be finite. Default ``1.0``.
    lines : Sequence[Sequence[int]] or None, optional
        Polyline connectivity as sequences of node indices (e.g. one polyline per beam).
        Written as the VTK ``LINES`` cells; ``None`` writes points only.
    title : str, optional
        The VTK header title line. Line breaks and other non-printable characters
        are replaced by spaces and the title is truncated to 255 characters, so the
        header always stays a single valid line.

    Returns
    -------
    pathlib.Path
        The file written: deformed ``POINTS``, a ``displacement`` point-vector field,
        and the ``LINES`` cells when ``lines`` is given.

    Raises
    ------
    ValueError
        If ``points`` / ``displacement`` are not ``(n_nodes, 3)`` and equal in shape,
        non-finite, ``displacement`` has a non-negligible imaginary part, ``scale``
        is non-finite, or a line index is out of range.
    """
    if not np.isfinite(scale):
        msg = f"scale must be finite, got {scale}"
        raise ValueError(msg)
    node_points = np.asarray(points, dtype=np.float64)
    node_displacement = _real_displacement(displacement)
    _validate_node_shapes(node_points, node_displacement)
    polylines = (
        _validate_lines(lines, node_points.shape[0]) if lines is not None else None
    )
    field = scale * node_displacement
    text = _polydata_text(node_points + field, field, polylines, title)
    destination = Path(path)
    destination.write_text(text, encoding="utf-8")
    return destination


def write_mode_animation_vtk(
    points: npt.ArrayLike,
    displacement: npt.ArrayLike,
    path: str | Path,
    *,
    n_frames: int = 24,
    scale: float = 1.0,
    lines: Sequence[Sequence[int]] | None = None,
    title: str = "VANE mode shape",
) -> list[Path]:
    """Write a mode's oscillation as a VTK animation (frames + ``.pvd``).

    A mode oscillates: the physical motion at phase ``theta`` is the real part of the
    complex mode shape rotated by ``theta``, ``Re(displacement) * cos(theta) -
    Im(displacement) * sin(theta)``. This sweeps ``theta`` over one full period in
    ``n_frames`` equal steps, writing each frame's deformed mesh as a legacy VTK
    ``POLYDATA`` file (via the same serializer as :func:`write_mode_shape_vtk`) and a
    a ``.pvd`` collection at ``path`` that ties the frames into one time series.
    Opening the ``.pvd`` in a VTK viewer plays the mode's motion directly — the
    VTK-native counterpart of the interactive Plotly animation.

    A real ``displacement`` animates as a pure ``cos(theta)`` standing-wave swing; a
    complex one animates its travelling/orbiting motion (non-zero imaginary part).

    Parameters
    ----------
    points : npt.ArrayLike
        Undeformed node coordinates, shape ``(n_nodes, 3)``.
    displacement : npt.ArrayLike
        Nodal mode-shape displacement vectors, shape ``(n_nodes, 3)``; real or complex.
    path : str or pathlib.Path
        Destination ``.pvd`` collection path. One VTK file per frame is written next to
        it, named ``<stem>.<frame>.vtk``.
    n_frames : int, optional
        Number of frames over one oscillation period. Default ``24``.
    scale : float, optional
        Visual exaggeration applied to the displacement. Default ``1.0``.
    lines : Sequence[Sequence[int]] or None, optional
        Polyline connectivity (e.g. one polyline per beam), written into every frame.
    title : str, optional
        The VTK header title line written into every frame, sanitized to a single
        line of at most 255 characters as in :func:`write_mode_shape_vtk`.

    Returns
    -------
    list[pathlib.Path]
        The per-frame VTK files in time order. The ``.pvd`` collection is written at
        ``path`` (open that in a VTK viewer).

    Raises
    ------
    ValueError
        If ``points`` / ``displacement`` are not ``(n_nodes, 3)`` and equal in shape,
        non-finite, ``scale`` is non-finite, ``n_frames`` is not a whole number >= 1, or
        a line index is out of range.
    """
    if not np.isfinite(scale):
        msg = f"scale must be finite, got {scale}"
        raise ValueError(msg)
    if not np.isfinite(n_frames) or n_frames != int(n_frames):
        msg = f"n_frames must be a whole number, got {n_frames!r}"
        raise ValueError(msg)
    n_frames = int(n_frames)
    if n_frames < 1:
        msg = f"n_frames must be >= 1, got {n_frames}"
        raise ValueError(msg)

    node_points = np.asarray(points, dtype=np.float64)
    node_displacement = np.asarray(displacement, dtype=np.complex128)
    _validate_node_shapes(node_points, node_displacement)
    polylines = (
        _validate_lines(lines, node_points.shape[0]) if lines is not None else None
    )
    real_part = np.asarray(node_displacement.real, dtype=np.float64)
    imag_part = np.asarray(node_displacement.imag, dtype=np.float64)

    destination = Path(path)
    stem = destination.stem
    angles = 2.0 * np.pi * np.arange(n_frames) / n_frames
    frame_files: list[Path] = []
    collection: list[str] = []
    for index, angle in enumerate(angles):
        field = scale * (real_part * np.cos(angle) - imag_part * np.sin(angle))
        text = _polydata_text(node_points + field, field, polylines, title)
        frame_name = f"{stem}.{index:04d}.vtk"
        frame_path = destination.with_name(frame_name)
        frame_path.write_text(text, encoding="utf-8")
        frame_files.append(frame_path)
        # quoteattr XML-escapes the filename (and adds quotes) so a stem with XML
        # metacharacters (&, <, ">, ...) — legal in a filename — still yields valid PVD.
        collection.append(
            f'    <DataSet timestep="{index}" group="" part="0" '
            f"file={quoteattr(frame_name)}/>"
        )

    pvd_text = "\n".join(
        [
            '<?xml version="1.0"?>',
            '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
            "  <Collection>",
            *collection,
            "  </Collection>",
            "</VTKFile>",
            "",
        ]
    )
    destination.write_text(pvd_text, encoding="utf-8")
    return frame_files
