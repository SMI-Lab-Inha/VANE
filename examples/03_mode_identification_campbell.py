# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tutorial 3: identify and track the modes of a sweep and draw its Campbell diagram.

The script writes a synthetic sweep of ten operating points (4 to 15 m/s, twelve
azimuths each), labels the modes of one point, runs the modal pipeline over the
points below rated on the rotor-speed axis, lists the tracks and the resonance
crossings, writes the Campbell and resonance tables (CSV, and Excel when ``openpyxl``
is installed) and an interactive Campbell diagram, and draws a static one. It then
shows why a sweep that extends past rated must use the wind-speed axis.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap

import matplotlib as mpl

mpl.use("Agg")

import numpy as np
from _synthetic import (
    SyntheticTurbine,
    output_directories,
    output_parser,
    save_figure,
)

from vane import ModalPipeline, discover_operating_points
from vane.export import campbell_table, resonance_table, write_table
from vane.io import mbc3_transform
from vane.modal import label_solution, modes_from_mbc
from vane.viz import plot_campbell, plot_campbell_static

WIND_SPEEDS = (4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 13.0, 15.0)  # m/s
HARMONICS = (1, 3, 6)


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
        # 1. Write the sweep: the .lin files of every wind speed, in one directory.
        turbine = SyntheticTurbine()
        turbine.write_sweep(data / "sweep", WIND_SPEEDS, n_azimuths=12)
        points = discover_operating_points(data / "sweep")
        print(f"Discovered {len(points)} operating points:")
        for point in points:
            first = point[0]
            print(
                f"  {point.name}: {len(point)} azimuths, wind {first.wind_speed:4.1f} "
                f"m/s, rotor {first.rotor_speed * 30 / np.pi:6.3f} rpm"
            )

        # 2. Physical labels at one operating point.
        at_8 = next(p for p in points if p.name == "ws080")
        solution = modes_from_mbc(mbc3_transform(at_8))
        print()
        print("Labels at 8 m/s")
        print("  f_n (Hz)  label                          category             conf.")
        for k, label in enumerate(label_solution(solution)):
            print(
                f"  {solution.natural_frequencies_hz[k]:8.4f}  {label.label:<30} "
                f"{label.category.name:<20} {label.confidence:5.2f}"
            )

        # 3. Track the points below rated on the rotor-speed axis.
        rated = turbine.rated_wind_speed
        below = [p for p in points if p[0].wind_speed <= rated + 0.5]
        pipeline = ModalPipeline(harmonics=HARMONICS)
        result = pipeline.run(below, parameter_name="rotor_speed_rpm")
        diagram = result.campbell
        rpm = np.asarray(diagram.parameter_values)
        print()
        print(f"Campbell diagram: {len(diagram.tracks)} tracks over {rpm.size} points")
        print(
            "  track  label                          f(first)  f(last)  conf.  ambig."
        )
        for index, track in enumerate(diagram.tracks):
            print(
                f"  {index:5d}  {track.label.label:<30} "
                f"{track.natural_frequencies_hz[0]:8.4f} "
                f"{track.natural_frequencies_hz[-1]:8.4f}  {track.confidence:5.2f}  "
                f"{track.is_ambiguous}"
            )
        print()
        print("Resonance crossings")
        print("  track  harmonic  rotor (rpm)  f (Hz)  damping  severity")
        for crossing in result.resonances:
            print(
                f"  {crossing.track_index:5d}  {crossing.harmonic:7d}P  "
                f"{crossing.rotor_speed_rpm:11.2f}  {crossing.frequency_hz:6.3f}  "
                f"{crossing.damping_ratio:7.4f}  {crossing.severity.value}"
            )

        # 4. Tables: CSV always, Excel when openpyxl is installed.
        tracks = campbell_table(diagram)
        crossings = resonance_table(result.resonances)
        write_table(tracks, data / "campbell.csv")
        write_table(crossings, data / "resonances.csv")
        print()
        print(f"Campbell table: {len(tracks)} rows, columns {list(tracks.columns)}")
        if importlib.util.find_spec("openpyxl") is not None:
            write_table(tracks, data / "campbell.xlsx")
            print("Tables written: campbell.csv, resonances.csv, campbell.xlsx")
        else:
            print(
                "Tables written: campbell.csv, resonances.csv (no openpyxl: no .xlsx)"
            )

        # 5. The interactive diagram (Plotly HTML) and a static one (Matplotlib).
        plot_campbell(diagram).write_html(data / "campbell.html")
        figure = plot_campbell_static(diagram, harmonics=HARMONICS, size="double")
        save_figure(figure, figures / "tutorial_campbell.png")

        # 6. Past rated the rotor speed is constant: use the wind-speed axis.
        print()
        try:
            pipeline.run(points, parameter_name="rotor_speed_rpm")
        except ValueError as error:
            print("Rotor-speed axis over the full sweep raises ValueError:")
            print(
                textwrap.fill(
                    str(error), width=86, initial_indent="  ", subsequent_indent="  "
                )
            )
        by_wind = pipeline.run(points, parameter_name="wind_speed")
        print(
            f"Wind-speed axis: {len(by_wind.campbell.tracks)} tracks over "
            f"{len(by_wind.campbell.parameter_values)} points, "
            f"{len(by_wind.resonances)} resonances reported"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
