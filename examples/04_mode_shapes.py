# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tutorial 4: visualize the mode shapes of one operating point.

The script writes one synthetic operating point (8 m/s), finds and labels its modes,
and visualizes them three ways: a static bar chart of the participation of each
degree of freedom (matplotlib), interactive 3-D views of the deformed turbine and of
the complex mode-shape phasors (Plotly HTML), and a VTK animation of the
regressive flap mode (legacy VTK files and a ``.pvd`` collection).
"""

from __future__ import annotations

import sys

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from _synthetic import (
    SyntheticTurbine,
    figure_style,
    output_directories,
    output_parser,
    save_figure,
)

from vane.io import mbc3_transform, read_lin_file
from vane.modal import label_solution, modes_from_mbc
from vane.viz import (
    deform_mode,
    plot_mode_3d,
    plot_mode_fowt,
    plot_mode_participation,
    write_mode_animation_vtk,
)

WIND_SPEED = 8.0  # m/s


def main(argv: list[str] | None = None) -> int:
    """Run the tutorial.

    Parameters
    ----------
    argv : list[str] or None, optional
        Command-line arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        The exit status.
    """
    args = output_parser(__doc__).parse_args(argv)
    with output_directories(args.output) as (data, figures):
        # 1. One operating point, its modes and their labels.
        paths = SyntheticTurbine().write_operating_point(data / "lin", WIND_SPEED)
        solution = modes_from_mbc(mbc3_transform([read_lin_file(p) for p in paths]))
        labels = label_solution(solution)
        for k, label in enumerate(labels):
            frequency = solution.natural_frequencies_hz[k]
            print(f"mode {k}: {frequency:.4f} Hz  {label.label}")
        tower = next(k for k, lab in enumerate(labels) if "Fore-Aft" in lab.label)
        flap = next(k for k, lab in enumerate(labels) if "Regressive" in lab.label)

        # 2. Static participation charts of two modes. Short names label the bars
        # with the degree of freedom and, after MBC3, its multi-blade coordinate
        # (collective, cosine or sine) rather than blades 1, 2 and 3.
        with figure_style():
            figure, axes = plt.subplots(2, 1, figsize=(17.0 / 2.54, 4.4))
            for ax, k in zip(axes, (tower, flap), strict=True):
                plot_mode_participation(solution, k, top_n=4, ax=ax, short_names=True)
                frequency = solution.natural_frequencies_hz[k]
                ax.set_title(f"{labels[k].label}, {frequency:.3f} Hz")
            figure.tight_layout()
            save_figure(figure, figures / "tutorial_mode_participation.png")

        # 3. Interactive 3-D views (open the HTML files in a browser).
        for k in (tower, flap):
            plot_mode_fowt(solution, k).write_html(data / f"mode_{k}_turbine.html")
            plot_mode_3d(solution, k).write_html(data / f"mode_{k}_phasors.html")
        print(f"3-D views written: mode_{tower}_*.html, mode_{flap}_*.html")

        # 4. A VTK animation of the regressive flap mode.
        deformation = deform_mode(solution, flap)
        members = deformation.geometry.members
        points = np.vstack([member.nodes for member in members])
        displacement = np.vstack(deformation.displacements)
        lines, start = [], 0
        for member in members:
            count = len(member.nodes)
            lines.append(list(range(start, start + count)))
            start += count
        written = write_mode_animation_vtk(
            points, displacement, data / "regressive_flap.pvd", lines=lines
        )
        frames = [path for path in written if path.suffix == ".vtk"]
        print(
            f"VTK animation: regressive_flap.pvd and {len(frames)} frames, "
            f"{points.shape[0]} nodes on {len(members)} members"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
