# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tutorial 1: run an OpenFAST linearization sweep over an operating curve.

This tutorial needs an OpenFAST executable and a local clone of the public OpenFAST
``r-test`` repository. Set ``VANE_RTEST_DIR`` to the clone and ``VANE_OPENFAST`` to
the executable (or put ``openfast`` on ``PATH``). Without them the script prints what
is missing and exits with status 0.

The script linearizes the NREL 5 MW land-based deck
``glue-codes/openfast/5MW_Land_DLL_WTurb`` at a few wind speeds, with the rotor
speed and pitch of ``data/nrel5mw_operating_curve.csv``. It prints the settings VANE
inferred from the deck and the curve, the outcome of each operating point, and the
modes of the resulting ``.lin`` files on the wind-speed axis. ``--configure-only``
reads the deck and the curve and prints the inferred settings without launching
OpenFAST.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from _synthetic import output_parser

from vane import ModalPipeline, discover_operating_points
from vane.run import (
    auto_configure_sweep,
    curve_power_coefficient,
    detect_deck_generation,
    operating_curve_from_csv,
    run_operating_sweep,
)

CURVE = Path(__file__).resolve().parent / "data" / "nrel5mw_operating_curve.csv"
DECK = Path("glue-codes", "openfast", "5MW_Land_DLL_WTurb", "5MW_Land_DLL_WTurb.fst")


def find_inputs() -> tuple[Path | None, Path | None, list[str]]:
    """Return the deck, the OpenFAST executable and what is missing.

    Returns
    -------
    tuple
        The deck path (or ``None``), the executable path (or ``None``) and a list of
        messages describing each missing input.
    """
    missing = []
    deck = None
    clone = os.environ.get("VANE_RTEST_DIR")
    if not clone:
        missing.append("VANE_RTEST_DIR is not set (a clone of OpenFAST/r-test)")
    elif not (Path(clone) / DECK).is_file():
        missing.append(f"deck not found: {Path(clone) / DECK}")
    else:
        deck = Path(clone) / DECK
    found = os.environ.get("VANE_OPENFAST") or shutil.which("openfast")
    executable = Path(found) if found and Path(found).is_file() else None
    if executable is None:
        missing.append("no OpenFAST executable: set VANE_OPENFAST or put it on PATH")
    return deck, executable, missing


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
    parser = output_parser(
        __doc__,
        output_help=(
            "directory for the sweep results (default: a new temporary directory, "
            "which is kept and printed)"
        ),
    )
    parser.add_argument(
        "--winds",
        type=float,
        nargs="+",
        default=[6.0, 10.0, 14.0, 18.0],
        help="wind speeds to linearize, m/s (default: 6 10 14 18)",
    )
    parser.add_argument(
        "--configure-only",
        action="store_true",
        help="print the inferred settings and stop before launching OpenFAST",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="parallel OpenFAST runs (default: CPU count less one)",
    )
    args = parser.parse_args(argv)

    deck, executable, missing = find_inputs()
    if deck is None or (executable is None and not args.configure_only):
        print("This tutorial needs OpenFAST and the OpenFAST r-test deck:")
        for message in missing:
            print(f"  - {message}")
        print("Nothing was run.")
        return 0

    # 1. The operating curve: wind speed, rotor speed and pitch, with cp.
    wind, rotor_speed, pitch = operating_curve_from_csv(CURVE)
    print(f"Operating curve {CURVE.name}: {len(wind)} points")
    for row in zip(wind, rotor_speed, pitch, strict=True):
        print("  wind {:6.3f} m/s   rotor {:7.4f} rpm   pitch {:5.2f} deg".format(*row))

    # 2. What VANE reads from the deck and the curve.
    generation = detect_deck_generation(deck)
    print()
    print(
        f"Deck {deck.name}: input format {generation.label}, needs OpenFAST "
        f"{generation.describe()}"
    )
    settings, auto = auto_configure_sweep(
        deck,
        None,
        curve_wind=wind,
        curve_rotor_speed=rotor_speed,
        curve_cp=curve_power_coefficient(CURVE, wind),
    )
    print("Inferred settings:")
    for item in auto.inferred:
        print(f"  {item.name} = {item.value!r}  ({item.source})")
    for warning in auto.warnings:
        print(f"  warning: {warning}")
    print(f"  NLinTimes = {settings.n_lin_times}, CalcSteady = {settings.calc_steady}")
    if args.configure_only:
        return 0

    # 3. The sweep. Each operating point takes minutes; the points run in parallel.
    # The results are kept: in --output, or in a new temporary directory.
    root = args.output or Path(tempfile.mkdtemp(prefix="vane_sweep_"))
    sweep_dir = root / "sweep"
    print()
    print(f"Running OpenFAST {executable}; results in {sweep_dir}")
    result = run_operating_sweep(
        deck,
        sweep_dir,
        args.winds,
        performance_csv=CURVE,
        executable=executable,
        max_workers=args.max_workers,
    )

    # One row per operating point.
    print()
    print("  wind (m/s)  rotor (rpm)  pitch (deg)  converged  attempts  .lin files")
    for case in result.cases:
        point = case.operating_point
        print(
            f"  {point.wind_speed:10.1f}  {point.rotor_speed_rpm:11.3f}  "
            f"{point.blade_pitch_deg:11.2f}  {case.converged!s:>9}  "
            f"{case.attempts:8d}  {case.n_lin:10d}"
        )
    if result.auto_configuration is not None:
        for item in result.auto_configuration.inferred:
            if item.name in ("executable", "template_root"):
                print(f"  {item.name} = {item.value}")

    # 4. The modes of the linearized points, on the wind-speed axis.
    if not all(case.converged for case in result.cases):
        print("Some points did not converge; see the attempt logs in each case.")
        return 0
    pipeline = ModalPipeline().run(
        discover_operating_points(sweep_dir), parameter_name="wind_speed"
    )
    print()
    for track in pipeline.campbell.tracks:
        values = ", ".join(f"{f:.3f}" for f in track.natural_frequencies_hz)
        print(f"  {track.label.label:<34} {values} Hz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
