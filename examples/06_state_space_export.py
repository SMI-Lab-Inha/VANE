# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tutorial 6: export state-space, reduced, Kalman and LPV models.

The script writes a synthetic sweep (4 to 15 m/s), runs the modal pipeline on the
wind-speed axis, and exports the model of one operating point as a state-space
system. It reduces that system to its structural modes, builds a Kalman filter on two
tower-top accelerometers, runs it on a simulated record and reconstructs the
tower-base moment. It then builds a mode-tracked family over the sweep, schedules it
between two operating points, saves and reloads it, and builds a named
physical-state family for control design directly from the ``.lin`` files.
"""

from __future__ import annotations

import sys
import textwrap

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

from vane import ModalPipeline, discover_operating_points
from vane.export import (
    build_physical_lpv_family,
    load_lpv_family,
    save_lpv_family,
    write_rom,
)
from vane.sysid import (
    CovarianceSet,
    StateSpace,
    build_kalman_model,
    build_rom_family,
    discretize_process_noise,
    filter_consistency,
    load_rom_family,
    measurement_noise_from_specs,
    modal_process_noise,
    modal_reduction,
    reconstruct_load,
    run_filter,
    save_rom_family,
    schedule_model,
)

WIND_SPEEDS = (4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 13.0, 15.0)  # m/s
BAND_HZ = (0.1, 2.0)
SENSORS = ("YawBrTAxp", "YawBrTAyp")
SENSOR_STD = (0.02, 0.02)  # m/s^2
DT = 0.05  # s
N_STEPS = 6000  # 300 s


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
        # 1. The sweep and the modal pipeline on the wind-speed axis.
        SyntheticTurbine().write_sweep(data / "sweep", WIND_SPEEDS, n_azimuths=12)
        points = discover_operating_points(data / "sweep")
        result = ModalPipeline().run(points, parameter_name="wind_speed")
        winds = [mbc.wind_speed for mbc in result.mbc_results]

        # 2. The full-order model at 8 m/s.
        index = winds.index(8.0)
        system = result.state_space(index)
        print(
            f"State space at {winds[index]} m/s: {system.n_states} states, "
            f"{system.n_inputs} inputs, {system.n_outputs} outputs, "
            f"stable: {system.is_stable()}"
        )
        print(f"  inputs:  {system.input_names}")
        print(f"  outputs: {system.output_names}")

        # 3. Modal reduction to the structural modes that drive the tower-base load.
        rom = modal_reduction(
            system, n_modes=4, frequency_band_hz=BAND_HZ, rank_output="TwrBsMyt"
        )
        reduced = rom.system
        print()
        print(
            f"Modal reduction: {rom.n_full_states} -> {reduced.n_states} states, "
            f"energy retained {rom.energy_retained:.4f}"
        )
        for f, z in zip(rom.frequencies_hz, rom.damping_ratios, strict=True):
            print(f"  {f:.4f} Hz, damping {z:.4f}")
        bundle = write_rom(rom, data / "rom_8ms")
        print(f"  written: {sorted(p.name for p in bundle.iterdir())}")

        # 4. A Kalman filter on the two tower-top accelerometers.
        rows = [
            i
            for i, name in enumerate(reduced.output_names)
            if any(tag in name for tag in SENSORS)
        ]
        assert reduced.b is not None and reduced.c is not None and reduced.d is not None
        plant = StateSpace(
            a=reduced.a,
            b=reduced.b,
            c=reduced.c[rows],
            d=reduced.d[rows],
            state_names=reduced.state_names,
            input_names=reduced.input_names,
            output_names=[reduced.output_names[i] for i in rows],
        )
        q_c = modal_process_noise([1.0e-4] * len(rom.frequencies_hz))
        q_d = discretize_process_noise(plant.a, q_c, DT)
        r = measurement_noise_from_specs(SENSOR_STD)
        model = build_kalman_model(
            plant, CovarianceSet(q=q_d, r=r, p0=np.eye(plant.n_states)), dt=DT
        )
        assert model.c is not None

        # A simulated record from the same model: process and sensor noise only.
        rng = np.random.default_rng(1)
        truth = np.zeros((N_STEPS, plant.n_states))
        state = np.zeros(plant.n_states)
        noise_q = np.linalg.cholesky(q_d + 1e-15 * np.eye(plant.n_states))
        for k in range(N_STEPS):
            state = model.a @ state + noise_q @ rng.standard_normal(plant.n_states)
            truth[k] = state
        measurements = truth @ model.c.T + rng.standard_normal((N_STEPS, len(rows))) * (
            np.asarray(SENSOR_STD)
        )
        controls = np.zeros((N_STEPS, plant.n_inputs))  # no input perturbation
        run = run_filter(model, measurements, controls=controls)
        consistency = filter_consistency(run)
        tower = reduced.c[[reduced.output_names.index("ED TwrBsMyt, (kN-m)")]]
        load, load_std = reconstruct_load(run, tower)
        true_load = truth @ tower.T
        error = load[:, 0] - true_load[:, 0]
        settled = slice(N_STEPS // 10, None)
        rmse = float(np.sqrt(np.mean(error[settled] ** 2)))
        inside = float(np.mean(np.abs(error[settled]) <= 2.0 * load_std[settled, 0]))
        print()
        print(f"Kalman filter on {plant.output_names}, dt = {DT} s, {N_STEPS} steps")
        print(
            f"  mean NIS {consistency.mean_nis:.3f} (expected "
            f"{consistency.expected_nis:.1f}), consistent: {consistency.consistent}"
        )
        print(
            f"  TwrBsMyt: true std {np.std(true_load[settled]):.1f} kN-m, "
            f"RMSE {rmse:.1f} kN-m, within 2 sigma {100 * inside:.1f} %"
        )

        # 5. A mode-tracked family over the sweep, scheduled on wind speed.
        systems = [result.state_space(i) for i in range(len(winds))]
        family = build_rom_family(
            systems,
            winds,
            n_modes=4,
            frequency_band_hz=BAND_HZ,
            rank_output="TwrBsMyt",
        )
        print()
        print("Mode-tracked family: frequency (Hz) of each mode")
        print("  wind   " + "".join(f"  mode {k}" for k in range(4)))
        for wind, freqs in zip(
            family.operating_points, family.frequencies_hz, strict=True
        ):
            print(f"  {wind:5.1f}  " + "".join(f"  {f:6.4f}" for f in freqs))
        lowest = np.round(family.tracking_mac.min(axis=0), 4)
        print(f"  lowest tracking MAC per mode: {lowest}")
        scheduled = schedule_model(family, 9.5)
        print(
            f"Scheduled at 9.5 m/s: {scheduled.system.n_states} states, "
            f"min tracking MAC {scheduled.min_tracking_mac:.4f}, "
            f"extrapolated {scheduled.extrapolated}"
        )
        save_rom_family(family, data / "family.npz")
        reloaded = load_rom_family(data / "family.npz")
        same = np.array_equal(reloaded.systems[3].a, family.systems[3].a)
        print(f"Saved and reloaded family.npz: matrices identical {same}")

        # 6. A named physical-state family straight from the .lin files.
        print()
        try:
            build_physical_lpv_family(
                data / "sweep",
                states=["1st tower fore-aft", "Variable speed generator"],
                inputs=["horizontal wind speed", "collective blade-pitch"],
                outputs=["YawBrTAxp", "GenSpeed"],
            )
        except ValueError as error:
            print("Tower and generator only raises ValueError:")
            text = textwrap.shorten(str(error), width=300)
            print(
                textwrap.fill(
                    text, width=86, initial_indent="  ", subsequent_indent="  "
                )
            )
        plant_family = build_physical_lpv_family(
            data / "sweep",
            states=[
                "1st tower fore-aft",
                "Variable speed generator",
                "1st flapwise bending-mode DOF of blade 1",
                "1st flapwise bending-mode DOF of blade 2",
                "1st flapwise bending-mode DOF of blade 3",
            ],
            inputs=["horizontal wind speed", "collective blade-pitch"],
            outputs=["YawBrTAxp", "GenSpeed"],
        )
        print(
            f"Physical family: {len(plant_family.systems)} points, "
            f"{len(plant_family.state_names)} states, scheduled on "
            f"{plant_family.scheduling_variable} ({plant_family.scheduling_unit})"
        )
        print(f"  twins added: {len(plant_family.auto_added_states)}")
        print(
            "  dropped coupling: "
            f"max {float(np.max(plant_family.dropped_coupling)):.2e}"
        )
        save_lpv_family(plant_family, data / "physical_family.npz")
        reloaded_plant = load_lpv_family(data / "physical_family.npz")
        same = reloaded_plant.state_names == plant_family.state_names and all(
            np.array_equal(getattr(x, m), getattr(y, m))
            for x, y in zip(reloaded_plant.systems, plant_family.systems, strict=True)
            for m in ("a", "b", "c", "d")
        )
        print(f"  saved and reloaded physical_family.npz: matrices identical {same}")

        with figure_style():
            figure, axes = plt.subplots(figsize=(6.5, 4.0))
            time = DT * np.arange(N_STEPS)
            window = (time >= 200.0) & (time <= 260.0)
            axes.plot(time[window], true_load[window, 0], label="true")
            axes.plot(time[window], load[window, 0], label="estimate")
            axes.fill_between(
                time[window],
                load[window, 0] - 2.0 * load_std[window, 0],
                load[window, 0] + 2.0 * load_std[window, 0],
                alpha=0.25,
                linewidth=0.0,
                label="estimate +/- 2 sigma",
            )
            axes.set_xlabel("Time (s)")
            axes.set_ylabel("TwrBsMyt perturbation (kN-m)")
            axes.legend(loc="upper right", fontsize=8)
            figure.tight_layout()
            save_figure(figure, figures / "tutorial_kalman_load.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
