# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tutorial 2: read the .lin files of one operating point, apply MBC3, find the modes.

The script writes the twelve azimuth files of one synthetic operating point (8 m/s),
reads them, applies the multi-blade coordinate transform with azimuth averaging,
solves the eigenvalue problem, and prints the frequencies, damping ratios and
participation of each mode. It compares the result with the eigenvalues of one
frozen-azimuth matrix in OpenFAST's mixed frame, and draws both in one figure.
"""

from __future__ import annotations

import re
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
from vane.modal import (
    azimuth_spread,
    compute_modes,
    modes_from_mbc,
    participation_from_modes,
)

WIND_SPEED = 8.0  # m/s
N_AZIMUTHS = 12


def short_name(description: str, coordinate: str) -> str:
    """Return a short label for a degree of freedom.

    Parameters
    ----------
    description : str
        The ``.lin`` state description.
    coordinate : str
        Its multi-blade coordinate (``""`` for a fixed-frame state).

    Returns
    -------
    str
        For example ``"TFA1"`` or ``"BF1 cosine"``.
    """
    match = re.search(r"DOF_(\w+)", description)
    token = match.group(1) if match else description
    if token == "BF":
        token = "BF1"
    return f"{token} {coordinate}" if coordinate else token


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
        # 1. Write one operating point: twelve azimuths over one revolution.
        turbine = SyntheticTurbine()
        paths = turbine.write_operating_point(
            data / "ws080", WIND_SPEED, n_azimuths=N_AZIMUTHS
        )

        # 2. Read the files.
        files = [read_lin_file(path) for path in paths]
        first = files[0]
        print(f"Read {len(files)} files, {paths[0].name} to {paths[-1].name}")
        print(
            f"  {first.n_x} states, {first.n_u} inputs, {first.n_y} outputs; "
            f"rotor speed {first.rotor_speed:.4f} rad/s, wind {first.wind_speed} m/s"
        )
        rotating = [
            d
            for d, r in zip(first.x.descriptions, first.x.rotating_frame, strict=True)
            if r
        ]
        print(f"  {len(rotating)} states in the rotating frame, e.g. {rotating[0]!r}")

        # 3. MBC3 transform and azimuth average.
        mbc = mbc3_transform(files, retain_per_azimuth=True)
        print()
        print("MBC3 transform")
        print(f"  performed: {mbc.performed_transformation}, blades: {mbc.n_blades}")
        print(f"  rotor speed {mbc.rotor_speed_rpm:.3f} rpm, wind {mbc.wind_speed} m/s")
        print(f"  azimuths (deg): {np.round(mbc.azimuths_deg, 1).tolist()}")
        print(f"  ndof2 = {mbc.ndof2}, ndof1 = {mbc.ndof1}")
        names = [
            short_name(desc, coord)
            for desc, coord in zip(
                mbc.state_descriptions[: mbc.ndof2],
                mbc.mbc_coordinates[: mbc.ndof2],
                strict=True,
            )
        ]
        print(f"  displacement states: {names}")

        # 4. Eigenanalysis of the averaged matrix.
        solution = modes_from_mbc(mbc)
        print()
        print(
            f"Eigenanalysis: {solution.n_modes} oscillatory modes, "
            f"{solution.n_rigid_body_modes} rigid-body DOF, "
            f"{solution.n_overdamped} real eigenvalues"
        )
        print("  mode   f_n (Hz)   f_d (Hz)   damping (-)")
        for k in range(solution.n_modes):
            print(
                f"  {k:4d}   {solution.natural_frequencies_hz[k]:8.4f}   "
                f"{solution.damped_frequencies_hz[k]:8.4f}   "
                f"{solution.damping_ratios[k]:10.4f}"
            )

        # 5. Participation of each degree of freedom in each mode.
        participation = participation_from_modes(solution)
        print()
        print("Participation (each column scaled to a maximum of 1)")
        print(
            "  DOF            "
            + "".join(f"  mode {k}" for k in range(solution.n_modes))
        )
        for row, name in enumerate(names):
            values = "".join(f"  {v:6.3f}" for v in participation.magnitude[row])
            print(f"  {name:<14}{values}")

        # 6. Per-azimuth spread left by the periodic (3P) content.
        spread = azimuth_spread(mbc)
        print()
        print("Azimuth spread of the natural frequency (Hz)")
        print("  " + ", ".join(f"{s:.2e}" for s in spread.natural_frequency_std))

        # 7. For comparison: one frozen-azimuth matrix in the mixed frame.
        assert first.a is not None
        frozen = compute_modes(first.a, ndof2=mbc.ndof2, ndof1=0)
        print()
        print("Frozen azimuth (0 deg), mixed frame, no MBC")
        print(
            "  f_n (Hz): "
            + ", ".join(f"{f:.4f}" for f in frozen.natural_frequencies_hz)
        )

        with figure_style():
            figure, (left, right) = plt.subplots(
                1, 2, figsize=(6.5, 4.0), gridspec_kw={"width_ratios": [1.0, 1.3]}
            )
            for x, freqs in (
                (0, frozen.natural_frequencies_hz),
                (1, solution.natural_frequencies_hz),
            ):
                left.hlines(freqs, x - 0.3, x + 0.3)
            left.set_xticks([0, 1], ["frozen\nazimuth", "MBC3\naverage"])
            left.set_xlim(-0.6, 1.6)
            left.set_ylabel("Natural frequency (Hz)")
            left.set_title("Frequencies")
            image = right.imshow(
                participation.magnitude, vmin=0.0, vmax=1.0, aspect="auto"
            )
            right.set_xticks(range(solution.n_modes))
            right.set_xlabel("Mode")
            right.set_yticks(range(len(names)), names)
            right.set_title("Participation")
            right.grid(visible=False)
            figure.colorbar(image, ax=right)
            figure.tight_layout()
            save_figure(figure, figures / "tutorial_mbc_modes.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
