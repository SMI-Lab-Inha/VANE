# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Generate the figures of the plot gallery (docs/user_guide/plot_gallery.rst).

The figures are drawn from synthetic data only, so the script is deterministic and runs
in a few seconds:

* a sweep of the synthetic five-megawatt-class turbine of the tutorials
  (``examples/_synthetic.py``: tower, rotor and blade-flap modes), written to a
  temporary directory and analysed with :func:`vane.study.run_study`;
* a hand-built floating-turbine Campbell diagram with every mode family, to show the
  full encoding;
* a small floating-turbine modal solution for the whole-turbine view, and a crossing
  of two modes for the branch review.

Each plot is written at the single-column (8.5 cm) and double-column (17 cm) size as a
PNG at 150 dpi, small enough for the documentation. Run from the repository root::

    python docs/gallery/make_gallery.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from vane.campbell.builder import CampbellDiagram, build_campbell
from vane.config.dof_map import DofCategory
from vane.modal.eigensolver import ModalSolution
from vane.modal.identifier import ModeTrack, identify_modes
from vane.modal.labeler import ModeLabel
from vane.study import discover_operating_points, run_study
from vane.sysid.qoi import build_qoi_map
from vane.sysid.state_space import state_space_from_mbc
from vane.viz import (
    plot_azimuth_coverage_static,
    plot_blade_components_static,
    plot_branch_review_static,
    plot_campbell_panels,
    plot_campbell_static,
    plot_damping_static,
    plot_load_mode_heatmap_static,
    plot_mac_matrix_static,
    plot_mode_3d_static,
    plot_mode_fowt_static,
    plot_mode_participation,
    plot_mode_phasor_static,
    plot_tracking_confidence_static,
    save_figure,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    import numpy.typing as npt
    from matplotlib.figure import Figure

    from vane.modal.branch import BranchGroup
    from vane.study import StudyResult

ROOT = Path(__file__).resolve().parents[2]
FIGURES = Path(__file__).resolve().parent / "figures"
DPI = 150
SIZES = ("single", "double")
WIND_SPEEDS = (4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0)
HARMONICS = (1, 3, 6)
#: Index of the operating point used for the single-point figures (8 m/s).
POINT = 4

FLOATING_DESCRIPTIONS = [
    "ED platform horizontal surge translation DOF (internal DOF index = DOF_Sg), m",
    "ED platform pitch tilt rotation DOF (internal DOF index = DOF_P), rad",
    "ED platform heave translation DOF (internal DOF index = DOF_Hv), m",
    "ED 1st tower fore-aft bending mode DOF (internal DOF index = DOF_TFA1), m",
    "ED 1st tower side-side bending mode DOF (internal DOF index = DOF_TSS1), m",
    "ED 1st flapwise bending-mode DOF of blade 1 (internal DOF index = DOF_BF(1,1)), m",
    "ED 1st flapwise bending-mode DOF of blade 2 (internal DOF index = DOF_BF(2,1)), m",
    "ED 1st flapwise bending-mode DOF of blade 3 (internal DOF index = DOF_BF(3,1)), m",
]
FLOATING_MBC = ["", "", "", "", "", "collective", "cosine", "sine"]


def _synthetic_module() -> ModuleType:
    """Import the tutorials' synthetic-turbine module from its path."""
    path = ROOT / "examples" / "_synthetic.py"
    spec = importlib.util.spec_from_file_location("vane_gallery_synthetic", path)
    if spec is None or spec.loader is None:
        msg = f"cannot import {path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def turbine_study(directory: Path) -> StudyResult:
    """Write the synthetic sweep below rated and run a rotor-speed study on it."""
    turbine = _synthetic_module().SyntheticTurbine()
    turbine.write_sweep(directory, WIND_SPEEDS, n_azimuths=12)
    points = discover_operating_points(directory)
    return run_study(
        points,
        harmonics=HARMONICS,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _track(
    category: DofCategory,
    name: str,
    frequency: npt.NDArray[np.float64],
    damping: float,
    *,
    multiblade: str | None = None,
    confidence: float = 1.0,
    ambiguous: bool = False,
) -> ModeTrack:
    """Return a full-span track with a fixed label."""
    text = name if multiblade is None else f"{name} ({multiblade.capitalize()})"
    label = ModeLabel(category, text, 0.9, [], multiblade)
    n = frequency.size
    return ModeTrack(
        operating_points=list(range(n)),
        mode_indices=[0] * n,
        natural_frequencies_hz=[float(f) for f in frequency],
        damping_ratios=[damping] * n,
        label=label,
        confidence=confidence,
        is_ambiguous=ambiguous,
    )


def floating_diagram() -> CampbellDiagram:
    """Return a floating-turbine Campbell diagram with every mode family."""
    rpm = np.linspace(0.0, 12.0, 9)
    omega = rpm / 60.0
    flat = np.ones_like(rpm)
    flap = np.sqrt(0.67**2 + 1.6 * omega**2)
    tracks = [
        _track(DofCategory.PLATFORM_SURGE, "Platform Surge", 0.008 * flat, 0.08),
        _track(DofCategory.PLATFORM_HEAVE, "Platform Heave", 0.032 * flat, 0.06),
        _track(DofCategory.PLATFORM_PITCH, "Platform Pitch", 0.037 * flat, 0.05),
        _track(
            DofCategory.PLATFORM_YAW, "Platform Yaw", 0.12 * flat, 0.03, confidence=0.8
        ),
        _track(DofCategory.TOWER_FORE_AFT_1, "1st Tower Fore-Aft", 0.48 * flat, 0.04),
        _track(
            DofCategory.TOWER_SIDE_SIDE_1, "1st Tower Side-Side", 0.46 * flat, 0.006
        ),
        _track(DofCategory.DRIVETRAIN_TORSION, "Drivetrain Torsion", 0.62 * flat, 0.02),
        _track(
            DofCategory.BLADE_FLAP_1,
            "1st Blade Flap",
            flap - omega,
            0.12,
            multiblade="regressive",
        ),
        _track(
            DofCategory.BLADE_FLAP_1,
            "1st Blade Flap",
            flap + 0.03,
            0.14,
            multiblade="collective",
        ),
        _track(
            DofCategory.BLADE_FLAP_1,
            "1st Blade Flap",
            flap + omega,
            0.12,
            multiblade="progressive",
        ),
        _track(
            DofCategory.BLADE_EDGE_1,
            "1st Blade Edge",
            1.08 * flat - omega,
            0.005,
            multiblade="regressive",
            confidence=0.55,
        ),
        _track(
            DofCategory.BLADE_EDGE_1,
            "1st Blade Edge",
            1.08 * flat + omega,
            0.005,
            multiblade="progressive",
            ambiguous=True,
        ),
        _track(
            DofCategory.NACELLE_YAW, "Nacelle Yaw", 1.45 * flat, 0.01, confidence=0.7
        ),
    ]
    return CampbellDiagram(rpm, "rotor_speed_rpm", tracks, rpm.size)


def floating_solution() -> ModalSolution:
    """Return a floating-turbine solution with a coupled pitch-tower-flap mode."""
    shape = np.array(
        [[0.35], [1.0], [0.05], [0.45], [0.02], [0.6 + 0.1j], [0.05], [0.04]],
        dtype=np.complex128,
    )
    return ModalSolution(
        eigenvalues=np.array([-0.01 + 0.25j], dtype=np.complex128),
        natural_frequencies_hz=np.array([0.04], dtype=np.float64),
        damping_ratios=np.array([0.05], dtype=np.float64),
        damped_frequencies_hz=np.array([0.04], dtype=np.float64),
        mode_shapes=shape,
        full_eigenvectors=shape,
        n_rigid_body_modes=0,
        dof_descriptions=FLOATING_DESCRIPTIONS,
        dof_mbc_coordinates=FLOATING_MBC,
    )


def crossing_case() -> tuple[CampbellDiagram, list[BranchGroup]]:
    """Return a Campbell diagram and branch groups of two modes that cross."""
    descriptions = [
        "ED 1st tower fore-aft (internal DOF index = DOF_TFA1), m",
        "ED drivetrain (internal DOF index = DOF_DrTr), rad",
    ]

    def solution(frequencies: list[float], shapes: list[list[float]]) -> ModalSolution:
        freq = np.array(frequencies, dtype=np.float64)
        modes = np.array(shapes, dtype=np.complex128)
        return ModalSolution(
            eigenvalues=(-0.01 + 1j * freq).astype(np.complex128),
            natural_frequencies_hz=freq,
            damping_ratios=np.full(freq.size, 0.02),
            damped_frequencies_hz=freq,
            mode_shapes=modes,
            full_eigenvectors=modes,
            n_rigid_body_modes=0,
            dof_descriptions=descriptions,
        )

    distinct = [[1.0, 0.0], [0.0, 1.0]]
    mixed = [[1.0, 0.9], [0.9, 1.0]]
    solutions = [
        solution([0.50, 0.70], distinct),
        solution([0.55, 0.66], distinct),
        solution([0.60, 0.61], mixed),
        solution([0.65, 0.56], distinct),
        solution([0.70, 0.52], distinct),
    ]
    result = identify_modes(
        solutions, strategy="simultaneous", branch_review=True, resolve=True
    )
    diagram = build_campbell(result, [0.0, 3.0, 6.0, 9.0, 12.0])
    return diagram, list(result.branch_groups)


def builders(study: StudyResult) -> dict[str, Callable[[str], Figure]]:
    """Return the figure builder of every gallery plot, by file name."""
    diagram = study.pipeline.campbell
    floating = floating_diagram()
    solution = study.pipeline.solutions[POINT]
    flap = next(
        k
        for k, label in enumerate(study.pipeline.identification.tracks)
        if "Regressive" in label.label.label
    )
    flap_mode = study.pipeline.identification.tracks[flap].mode_indices[POINT]
    qoi = build_qoi_map(
        state_space_from_mbc(study.pipeline.mbc_results[POINT]), solution
    )
    crossing, groups = crossing_case()
    fowt = floating_solution()
    return {
        "campbell": lambda size: plot_campbell_static(
            diagram, harmonics=HARMONICS, size=size
        ),
        "campbell_floating": lambda size: plot_campbell_static(
            floating, harmonics=HARMONICS, operating_range=(6.9, 12.1), size=size
        ),
        "damping": lambda size: plot_damping_static(diagram, size=size),
        "panels": lambda size: plot_campbell_panels(
            diagram, harmonics=HARMONICS, size=size
        ),
        "participation": lambda size: plot_mode_participation(
            solution, flap_mode, top_n=5, short_names=True, size=size
        ),
        "phasor": lambda size: plot_mode_phasor_static(
            solution, flap_mode, top_n=5, size=size
        ),
        "blade_components": lambda size: plot_blade_components_static(
            solution, flap_mode, size=size
        ),
        "mode_3d": lambda size: plot_mode_3d_static(
            solution, flap_mode, top_n=5, size=size
        ),
        "turbine_mode": lambda size: plot_mode_fowt_static(fowt, 0, size=size),
        "mac": lambda size: plot_mac_matrix_static(
            study.pipeline.solutions[POINT - 1], solution, size=size
        ),
        "confidence": lambda size: plot_tracking_confidence_static(floating, size=size),
        "azimuth_coverage": lambda size: plot_azimuth_coverage_static(
            study.provenance, size=size
        ),
        "branch_review": lambda size: plot_branch_review_static(
            crossing, groups, size=size
        ),
        "qoi": lambda size: plot_load_mode_heatmap_static(qoi, size=size),
    }


def write(name: str, builder: Callable[[str], Figure], output: Path) -> None:
    """Draw a figure at each size and write it as a PNG."""
    for size in SIZES:
        figure = builder(size)
        path = output / f"{name}_{size}.png"
        save_figure(figure, path, dpi=DPI)
        plt.close(figure)
        print(f"{path.name}  {path.stat().st_size // 1024} kB")


def main(argv: list[str] | None = None) -> int:
    """Write every gallery figure.

    Parameters
    ----------
    argv : list[str] or None, optional
        Command-line arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        The exit status.
    """
    parser = argparse.ArgumentParser(description="Generate the plot gallery.")
    parser.add_argument("--output", type=Path, default=FIGURES)
    parser.add_argument("--only", nargs="*", default=None, help="figure names")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vane_gallery_") as scratch:
        study = turbine_study(Path(scratch) / "sweep")
        for name, builder in builders(study).items():
            if args.only is None or name in args.only:
                write(name, builder, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
