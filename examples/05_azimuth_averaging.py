# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tutorial 5: choose the azimuth count and estimate the error of the averaged model.

The script works on a synthetic periodic model with an exact orbit tangent and
blade-passing content at orders 3 to 24. It applies the blade-number rule for the
azimuth count, locates the rotor-azimuth state and pins its direction, compares the
exact error of uniform azimuth grids with the rule, estimates the sampling,
quadrature and periodicity errors of a non-uniform 36-azimuth linearization, checks
the periodicity term against Floquet exponents from the monodromy matrix, and
suggests a commensurate time step.
"""

from __future__ import annotations

import sys

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from _synthetic import (
    PERIODIC_STATES,
    figure_style,
    output_directories,
    output_parser,
    periodic_rotor_matrix,
    save_figure,
)

from vane.modal.averaging import (
    azimuth_mean,
    azimuth_tangent,
    eigen_basis,
    fit_harmonics,
    locate_azimuth_state,
    pin_azimuth,
    uniform_azimuths,
)
from vane.modal.averaging_error_estimate import (
    alias_free_azimuth_count,
    azimuth_count_table,
    commensurate_time_step,
    estimate_averaging_error,
    first_alias_order,
)
from vane.modal.floquet import monodromy_exponents

ROTOR_SPEED = 1.0  # rad/s (9.55 rpm)
BLADES = 3


def sample(azimuths: np.ndarray) -> np.ndarray:
    """Return the model's state matrices at the given azimuths.

    Parameters
    ----------
    azimuths : numpy.ndarray
        Rotor azimuths, rad.

    Returns
    -------
    numpy.ndarray
        The matrices, shape ``(n_azimuths, 4, 4)``.
    """
    return np.array(
        [periodic_rotor_matrix(float(psi), rotor_speed=ROTOR_SPEED) for psi in azimuths]
    )


def delivered_grid(count: int, jitter: float, seed: int = 4) -> np.ndarray:
    """Return a non-uniform grid like the one a CalcSteady linearization delivers.

    Parameters
    ----------
    count : int
        Number of target azimuths.
    jitter : float
        Largest delay of a snapshot past its target, as a fraction of the spacing.
    seed : int, optional
        Seed of the random delays.

    Returns
    -------
    numpy.ndarray
        Sorted azimuths, rad.
    """
    delays = (
        jitter * (2 * np.pi / count) * np.random.default_rng(seed).uniform(size=count)
    )
    return np.sort(np.mod(uniform_azimuths(count) + delays, 2 * np.pi))


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
    with output_directories(args.output) as (_data, figures):
        # 1. The blade-number rule: an N-azimuth mean is first aliased at lcm(N, B).
        print("Azimuth count N and its first alias order lcm(N, 3)")
        counts = range(3, 14)
        print("  N      " + "".join(f"{n:4d}" for n in counts))
        print(
            "  alias  " + "".join(f"{first_alias_order(n, BLADES):4d}" for n in counts)
        )
        print(
            "Smallest alias-free count for content up to order 18: "
            f"{alias_free_azimuth_count(BLADES, 18)}; up to order 24: "
            f"{alias_free_azimuth_count(BLADES, 24)}"
        )

        # 2. A dense reference: 72 uniform azimuths resolve orders up to 35.
        dense = uniform_azimuths(72)
        samples = sample(dense)
        azimuth = locate_azimuth_state(list(PERIODIC_STATES), ndof2=2)
        reference = azimuth_mean(dense, samples)
        basis = eigen_basis(reference)
        platform = complex(basis.values[int(np.argmax(basis.values.imag))])
        print()
        print(f"Azimuth state {azimuth.position}, rotor-speed state {azimuth.speed}")
        print(f"Reference eigenvalues: {np.round(np.sort_complex(basis.values), 6)}")

        # 3. Pin the azimuth direction with the tangent of the dense mean.
        tangent = azimuth_tangent(reference, azimuth, n_samples=dense.size)
        pinned = pin_azimuth(samples, tangent, azimuth)
        coarse = uniform_azimuths(9, 0.1)
        raw_9 = sample(coarse).mean(axis=0)
        pinned_9 = pin_azimuth(sample(coarse), tangent, azimuth).mean(axis=0)
        print(f"Orbit tangent: {np.round(tangent, 5)}")
        print("Mean of 9 uniform azimuths, eigenvalues")
        raw_values = np.round(np.sort_complex(np.linalg.eigvals(raw_9)), 5)
        print(f"  delivered column: {raw_values}")
        print(
            "  pinned:           "
            f"{np.round(np.sort_complex(np.linalg.eigvals(pinned_9)), 5)}"
        )

        # 4. The resolved error of every uniform count, worst over start azimuths.
        table = azimuth_count_table(
            fit_harmonics(dense, pinned),
            {"platform": platform},
            blades=BLADES,
            counts=range(4, 37),
            damping_tolerance_pct=0.05,
        )
        print()
        print("Uniform azimuth counts, platform damping error (resolved, worst start)")
        print("  N   alias  mult. of 3  damping err. (%)  frequency err. (%)")
        for row in table.rows:
            if row.count <= 18:
                print(
                    f"  {row.count:2d}  {row.first_alias_order:5d}  "
                    f"{row.multiple_of_blades!s:>10}  "
                    f"{row.resolved_damping_error_pct:16.3g}  "
                    f"{row.resolved_frequency_error_pct:18.3g}"
                )
        print(
            f"Recommended count (0.05 % damping): {table.recommended_resolved}; "
            f"resolved orders: up to {table.resolved_order}"
        )

        # 5. Error estimate of a delivered, non-uniform 36-azimuth linearization.
        delivered = delivered_grid(36, jitter=0.3)
        stack = sample(delivered)
        delivered_tangent = azimuth_tangent(
            azimuth_mean(delivered, stack), azimuth, n_samples=delivered.size
        )
        print()
        print("Error estimate of a 36-azimuth linearization (percent of reference)")
        print(
            "  samples    grid  status    sampling      quadrature  periodicity  accept"
        )
        print("                             zeta     f    zeta        zeta")
        for name, matrices in (
            ("delivered", stack),
            ("pinned", pin_azimuth(stack, delivered_tangent, azimuth)),
        ):
            for grid_name, grid in (("18", delivered[::2]), ("9", delivered[::4])):
                budget = estimate_averaging_error(
                    delivered,
                    matrices,
                    ROTOR_SPEED,
                    branches={"platform": platform},
                    azimuth=azimuth,
                    grid=grid,
                    damping_tolerance_pct=1.0,
                    frequency_tolerance_pct=0.1,
                )
                branch = budget.branches["platform"]
                term = branch.sampling_term
                assert term is not None
                print(
                    f"  {name:<9}  {grid_name:>4}  {budget.sampling_status:<8}  "
                    f"{term.damping_error_pct:6.3g} {term.frequency_error_pct:6.3g}  "
                    f"{branch.quadrature.damping_error_pct:8.3g}  "
                    f"{branch.periodicity.damping_error_pct:11.3g}  "
                    f"{branch.within_tolerance}"
                )
        print(f"Azimuth assessment: {budget.azimuth_assessment}")
        print(f"Platform eigenvalue of the 36-azimuth mean: {branch.eigenvalue:.6f}")
        print(f"Platform eigenvalue of the 72-azimuth mean: {platform:.6f}")

        # 6. Periodicity term against the Floquet exponent of the monodromy matrix.
        uniform = estimate_averaging_error(
            dense, samples, ROTOR_SPEED, branches={"platform": platform}
        ).branches["platform"]
        fine = uniform_azimuths(720)
        exponents = monodromy_exponents(sample(fine), fine, ROTOR_SPEED)
        floquet = complex(exponents[np.argmax(exponents.imag)])
        corrected = uniform.eigenvalue + uniform.periodicity.shift
        print()
        print("Platform branch of the 72-azimuth model (1/s)")
        print(f"  averaged eigenvalue:         {uniform.eigenvalue:.7f}")
        print(f"  averaged + periodicity term: {corrected:.7f}")
        print(f"  Floquet exponent (monodromy): {floquet:.7f}")

        # 7. A time step that puts each of 36 azimuths on a whole number of steps.
        step = commensurate_time_step(ROTOR_SPEED, 36, requested_time_step=0.025)
        print()
        print(
            f"Commensurate time step: {step.time_step:.6f} s "
            f"({step.steps_per_azimuth} steps per {step.azimuth_spacing_deg:g} deg)"
        )

        with figure_style():
            figure, axes = plt.subplots(figsize=(6.5, 4.0))
            floor = 1e-8
            for multiple, marker, name in (
                (True, "s", "N a multiple of 3"),
                (False, "o", "N not a multiple of 3"),
            ):
                rows = [r for r in table.rows if r.multiple_of_blades is multiple]
                axes.semilogy(
                    [r.count for r in rows],
                    [max(r.resolved_damping_error_pct, floor) for r in rows],
                    marker=marker,
                    linestyle="none",
                    label=name,
                )
            axes.axhline(0.05, color="0.4", linestyle="--", linewidth=1.0)
            axes.annotate("tolerance 0.05 %", (30, 0.06), fontsize=8)
            axes.annotate("rounding level (drawn at 1e-8 %)", (11, 2e-8), fontsize=8)
            axes.set_ylim(floor / 3, 2.0)
            axes.set_xlabel("Number of uniform azimuths N")
            axes.set_ylabel("Platform damping error (%)")
            axes.legend(loc="center right")
            figure.tight_layout()
            save_figure(figure, figures / "tutorial_azimuth_count.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
