# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Generate the figures of the verification and validation record.

Every figure is computed from a small synthetic system with a known answer, using the
same constructions as the test suite:

* ``floquet_sideband_order.png``: the averaged eigenvalue misses the Floquet exponent
  at second order in the modulation amplitude, and the sideband-corrected value at
  fourth order (``tests/test_modal/test_averaging.py::TestSidebandCorrection``).
* ``azimuth_count_aliasing.png``: the uniform azimuth mean of a three-bladed rotor is
  first aliased at ``lcm(N, 3)``
  (``tests/test_modal/test_averaging.py::TestWeightKernel``
  and ``tests/test_modal/test_averaging_error_estimate.py::TestAzimuthCountRule``).
* ``tangent_pinning_spectrum.png``: tangent pinning moves only the azimuth eigenvalue,
  while zeroing the azimuth column changes the reference
  (``tests/test_modal/test_averaging.py::TestAzimuthTreatments``).
* ``mbc3_coleman_split.png``: MBC3 on an isotropic rotor recovers ``omega_d`` and
  ``omega_d -/+ Omega`` (``tests/test_io/test_mbc_transform.py::TestMbc3Mathematics``).

Run from the repository root, with the package importable::

    PYTHONPATH=src python docs/validation/make_figures.py

The output is deterministic. The figures are written next to this script, in
``figures/``.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from vane.io.lin_reader import LinFile, OperatingPointTable
from vane.io.mbc_transform import mbc3_transform
from vane.modal.averaging import (
    AzimuthStates,
    azimuth_mean,
    azimuth_tangent,
    eigen_basis,
    fit_harmonics,
    pin_azimuth,
    sideband_correction,
    uniform_azimuths,
    zero_azimuth_column,
)
from vane.modal.averaging_error_estimate import first_alias_order

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    import numpy.typing as npt
    from matplotlib.figure import Figure

    MatrixFunction = Callable[[float], npt.NDArray[np.float64]]

OUT = Path(__file__).resolve().parent / "figures"
OMEGA = 1.0  # rotor speed of the synthetic periodic systems, rad/s
AZIMUTH = AzimuthStates(position=1, speed=3)

SIZE = (6.5, 4.0)
DPI = 110


def _figure_style() -> AbstractContextManager[object]:
    """Return the package's plotting style context, or a neutral one."""
    try:
        from vane.viz import plot_style  # type: ignore[attr-defined,unused-ignore]
    except ImportError:
        return contextlib.nullcontext()
    return plot_style()  # type: ignore[no-any-return,unused-ignore]


def _save(fig: Figure, name: str) -> None:
    """Write a figure as a small PNG without volatile metadata."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=DPI, metadata={"Software": None})
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Synthetic periodic rotor with an exact orbit tangent (tests/test_modal/test_averaging)
# --------------------------------------------------------------------------------------


def azimuth_system(eps: float = 0.05, delta: float = 0.0) -> MatrixFunction:
    """Return ``A(psi)`` of the four-state rotor model with an exact orbit tangent.

    States are ``[q, psi, q', w]``. The 3P and 6P content scales with ``eps`` and is
    built so that the model is the linearization about a periodic orbit, with one
    Floquet exponent exactly zero; ``delta`` adds a constant azimuth stiffness.
    """
    k0, cp, b0, cpw, dwp = 0.09, 0.02, 0.2, 0.05, 0.01

    def a_of(psi: float) -> npt.NDArray[np.float64]:
        p = eps * (0.8 * np.cos(3 * psi) + 0.5 * np.sin(3 * psi))
        dp = eps * 3.0 * (-0.8 * np.sin(3 * psi) + 0.5 * np.cos(3 * psi))
        ddp = -9.0 * p
        kp = k0 * (1.0 + eps * np.cos(3 * psi))
        base = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [-kp, 0.0, -cp, cpw],
                [0.02 * eps * np.sin(3 * psi), 0.0, dwp, -b0],
            ]
        )
        xi = np.array([p, 1.0, OMEGA * dp, 0.0])
        matrix: npt.NDArray[np.float64] = base.copy()
        matrix[2:4, 1] = np.array([OMEGA**2 * ddp, 0.0]) - base[2:4] @ xi
        matrix[3, 1] += delta
        return matrix

    return a_of


def _sample(a_of: MatrixFunction, angles: npt.NDArray[np.float64]) -> np.ndarray:
    return np.array([a_of(float(psi)) for psi in angles])


def _floquet_exponents(a_of: MatrixFunction, n: int = 4) -> npt.NDArray[np.complex128]:
    """Return Floquet exponents from an integrated monodromy matrix."""
    period = 2 * np.pi / OMEGA
    solution = solve_ivp(
        lambda t, y: (a_of(OMEGA * t) @ y.reshape(n, n)).ravel(),
        (0.0, period),
        np.eye(n).ravel(),
        method="DOP853",
        rtol=1e-13,
        atol=1e-15,
    )
    monodromy = solution.y[:, -1].reshape(n, n)
    multipliers = np.linalg.eigvals(monodromy).astype(np.complex128)
    return np.asarray(np.log(multipliers) / period, dtype=np.complex128)


def _matched(exponents: npt.NDArray[np.complex128], target: complex) -> complex:
    """Return the exponent equal to ``target`` modulo ``i Omega`` most closely."""
    period = 2 * np.pi / OMEGA
    k = int(np.argmin(np.abs(np.exp((exponents - target) * period) - 1.0)))
    value = complex(exponents[k])
    return value + 1j * round((target - value).imag / OMEGA) * OMEGA


def figure_floquet_order() -> None:
    """Averaged and sideband-corrected eigenvalue error against modulation amplitude."""
    amplitudes = np.geomspace(0.005, 0.4, 13)
    averaged_error, corrected_error = [], []
    angles = uniform_azimuths(36)
    for eps in amplitudes:
        a_of = azimuth_system(eps=float(eps))
        model = fit_harmonics(angles, _sample(a_of, angles))
        basis = eigen_basis(model.reference_mean)
        j = int(np.argmax(basis.values.imag))
        averaged = complex(basis.values[j])
        exact = _matched(_floquet_exponents(a_of), averaged)
        corrected = averaged + sideband_correction(model, basis, j, OMEGA)
        averaged_error.append(abs(exact - averaged))
        corrected_error.append(abs(exact - corrected))

    fig, ax = plt.subplots(figsize=SIZE, layout="constrained")
    ax.loglog(
        amplitudes, averaged_error, "o-", color="C0", ms=5, label="averaged eigenvalue"
    )
    ax.loglog(
        amplitudes,
        corrected_error,
        "s-",
        color="C1",
        ms=5,
        label="averaged + sideband correction",
    )
    for errors, power in ((averaged_error, 2), (corrected_error, 4)):
        anchor = len(amplitudes) // 2
        guide = errors[anchor] * (amplitudes / amplitudes[anchor]) ** power
        ax.loglog(amplitudes, guide, "k--", lw=0.8)
        ax.annotate(
            f"slope {power}",
            (amplitudes[-3], guide[-3]),
            xytext=(6, -12),
            textcoords="offset points",
        )
    ax.set_xlabel(r"Modulation amplitude $\varepsilon$ (-), 3P and 6P content")
    ax.set_ylabel(r"$|\lambda - \gamma_{\mathrm{Floquet}}|$ (1/s)")
    ax.set_title("Platform branch against its Floquet exponent", loc="left")
    ax.legend(loc="upper left")
    _save(fig, "floquet_sideband_order.png")


# --------------------------------------------------------------------------------------
# Aliasing of the uniform azimuth mean of a three-bladed rotor
# --------------------------------------------------------------------------------------


def _blade_harmonic_system(ratio: float = 0.4, orders: int = 30) -> MatrixFunction:
    """Return ``A(psi)`` with geometric content at 3P, 6P, ..., ``3 orders`` P."""

    def a_of(psi: float) -> npt.NDArray[np.float64]:
        mod = sum(ratio**m * np.cos(3 * m * psi + 0.3 * m) for m in range(1, orders))
        return np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [-0.09 * (1.0 + 0.05 * mod), 0.02 * mod, -0.02, 0.0],
                [0.01 * mod, -0.25, 0.0, -0.03],
            ]
        )

    return a_of


def figure_azimuth_aliasing() -> None:
    """Azimuth-mean eigenvalue error against the azimuth count N, B = 3."""
    a_of = _blade_harmonic_system()
    # 101 uniform azimuths: no multiple of 101 lies in the content (orders <= 87).
    dense = uniform_azimuths(101)
    reference = np.linalg.eigvals(azimuth_mean(dense, _sample(a_of, dense)))
    target = complex(reference[int(np.argmax(reference.imag))])
    floor = 1e-16
    counts = np.arange(2, 25)
    errors = []
    for count in counts:
        worst = 0.0
        for start in np.linspace(0.0, 2 * np.pi / count, 12, endpoint=False):
            grid = uniform_azimuths(int(count), float(start))
            values = np.linalg.eigvals(azimuth_mean(grid, _sample(a_of, grid)))
            nearest = values[int(np.argmin(np.abs(values - target)))]
            worst = max(worst, abs(nearest - target) / abs(target))
        errors.append(max(worst, floor))
    errors_arr = np.asarray(errors)
    alias = np.array([first_alias_order(int(n), 3) for n in counts])
    multiple = counts % 3 == 0

    fig, (left, right) = plt.subplots(1, 2, figsize=SIZE, layout="constrained")
    for ax, x in ((left, counts), (right, alias)):
        ax.semilogy(
            x[multiple],
            errors_arr[multiple],
            "o",
            color="C1",
            ms=6,
            label="N a multiple of 3: first alias at N",
        )
        ax.semilogy(
            x[~multiple],
            errors_arr[~multiple],
            "s",
            color="C0",
            ms=6,
            label="other N: first alias at 3N",
        )
        ax.axhline(2e-15, color="k", ls=":", lw=0.8)
        ax.set_ylim(floor * 0.3, 1e-2)
    left.text(1.5, 2.6e-15, "round-off", ha="left", va="bottom")
    left.set_xlabel("Azimuth count $N$ (-)")
    left.set_ylabel(r"Worst relative eigenvalue error (-)")
    left.set_xticks([3, 6, 9, 12, 15, 18, 21, 24])
    left.set_title("Error against N", loc="left")
    right.set_xlabel(r"First alias order lcm$(N, 3)$ (-)")
    right.set_title("Against lcm(N, 3)", loc="left")
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2, fontsize=8)
    _save(fig, "azimuth_count_aliasing.png")


# --------------------------------------------------------------------------------------
# Tangent pinning against zeroing the azimuth column
# --------------------------------------------------------------------------------------


def figure_tangent_pinning() -> None:
    """Eigenvalues of the reference mean before and after the azimuth treatments."""
    rng = np.random.default_rng(21)
    base = uniform_azimuths(36)
    angles = np.mod(base + 0.3 * (2 * np.pi / 36) * rng.uniform(size=36), 2 * np.pi)
    samples = _sample(azimuth_system(delta=1e-3), angles)
    mean = azimuth_mean(angles, samples)
    tangent = azimuth_tangent(mean, AZIMUTH, n_samples=36)
    pinned = azimuth_mean(angles, pin_azimuth(samples, tangent, AZIMUTH))
    zeroed = zero_azimuth_column(mean, AZIMUTH)

    before = np.linalg.eigvals(mean).astype(np.complex128)
    order = np.argsort(before.real + 1e-3 * before.imag)
    before = before[order]
    names = []
    for value in before:
        if abs(value.imag) > 1e-9:
            names.append("platform +" if value.imag > 0 else "platform -")
        elif value.real > 0:
            names.append("azimuth")
        else:
            names.append("rotor speed")

    def matched(values: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
        out = np.empty_like(before)
        free = list(values)
        for k, value in enumerate(before):
            if names[k] == "azimuth":
                continue
            i = int(np.argmin([abs(v - value) for v in free]))
            out[k] = free.pop(i)
        out[names.index("azimuth")] = free[0]
        return out

    after_pin, after_zero = (
        matched(np.linalg.eigvals(pinned).astype(np.complex128)),
        matched(np.linalg.eigvals(zeroed).astype(np.complex128)),
    )
    floor = 1e-17

    fig, (left, right) = plt.subplots(
        1, 2, figsize=SIZE, layout="constrained", width_ratios=(1.15, 1.0)
    )
    left.plot(
        before.real,
        before.imag,
        "o",
        mfc="none",
        mec="C0",
        mew=1.6,
        ms=10,
        label="reference mean",
    )
    left.plot(
        after_pin.real,
        after_pin.imag,
        "x",
        color="C1",
        mew=1.8,
        ms=8,
        label="tangent pinned",
    )
    left.axhline(0.0, color="k", lw=0.5)
    left.axvline(0.0, color="k", lw=0.5)
    for value, name in zip(before, names, strict=True):
        left.annotate(
            name,
            (value.real, value.imag),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )
    left.margins(x=0.12, y=0.15)
    left.set_xlabel(r"Re $\lambda$ (1/s)")
    left.set_ylabel(r"Im $\lambda$ (rad/s)")
    left.set_title("Eigenvalues", loc="left")
    left.legend(loc="lower left")

    positions = np.arange(len(before))
    shift_pin = np.maximum(np.abs(after_pin - before), floor)
    shift_zero = np.maximum(np.abs(after_zero - before), floor)
    right.semilogy(
        positions - 0.12,
        shift_pin,
        "x",
        color="C1",
        mew=1.8,
        ms=8,
        label="tangent pinned",
    )
    right.semilogy(
        positions + 0.12,
        shift_zero,
        "D",
        color="C2",
        ms=6,
        label="azimuth column zeroed",
    )
    right.axhline(1e-15, color="k", ls=":", lw=0.8)
    right.text(len(before) - 0.5, 1.6e-15, "round-off", ha="right", va="bottom")
    right.set_ylim(floor * 0.3, 1.0)
    right.set_xlim(-0.5, len(before) - 0.5)
    right.set_xticks(positions, names, rotation=20)
    right.set_ylabel(r"$|\Delta\lambda|$ from the reference (1/s)")
    right.set_title("Eigenvalue change", loc="left")
    right.legend(loc="center left")
    _save(fig, "tangent_pinning_spectrum.png")


# --------------------------------------------------------------------------------------
# MBC3 on an isotropic rotor (tests/test_io/test_mbc_transform)
# --------------------------------------------------------------------------------------


def _table(descriptions: list[str], rotating: bool) -> OperatingPointTable:
    count = len(descriptions)
    return OperatingPointTable(
        values=np.zeros(count),
        rotating_frame=np.full(count, rotating, dtype=bool),
        derivative_order=np.full(count, 2, dtype=int),
        descriptions=descriptions,
    )


def _isotropic_rotor(
    a: npt.NDArray[np.float64], rotor_speed: float, azimuth: float
) -> LinFile:
    """Return an in-memory linearization of three identical blade-flap oscillators."""
    displacement = [
        f"ED 1st flapwise bending-mode DOF of blade {b} "
        f"(internal DOF index = DOF_BF({b},1)), m"
        for b in (1, 2, 3)
    ]
    states = _table(
        displacement + [f"ED First time derivative of {d}" for d in displacement], True
    )
    empty = _table([], False)
    return LinFile(
        path=Path("isotropic_rotor.lin"),
        sim_time=0.0,
        rotor_speed=rotor_speed,
        azimuth=azimuth,
        wind_speed=0.0,
        jacobians_included=False,
        x=states,
        xdot=states,
        u=empty,
        y=empty,
        a=a,
        b=None,
        c=None,
        d=None,
        jacobians={},
    )


def figure_coleman_split() -> None:
    """Damped frequencies of the MBC3-averaged isotropic rotor against rotor speed."""
    w0, zeta = 2.0, 0.02
    a = np.zeros((6, 6))
    a[0:3, 3:6] = np.eye(3)
    a[3:6, 0:3] = -(w0**2) * np.eye(3)
    a[3:6, 3:6] = -2.0 * zeta * w0 * np.eye(3)
    damped = w0 * np.sqrt(1.0 - zeta**2)
    speeds = np.linspace(0.0, 1.5, 16)
    lines = np.empty((speeds.size, 3))
    for i, speed in enumerate(speeds):
        files = [_isotropic_rotor(a, float(speed), psi) for psi in (0.0, 2.0, 4.0)]
        result = mbc3_transform(files)
        assert result.avg_a is not None
        values = np.linalg.eigvals(result.avg_a)
        lines[i] = np.sort(values.imag[values.imag > 0])
    exact = np.column_stack(
        (damped - speeds, np.full_like(speeds, damped), damped + speeds)
    )
    worst = float(np.max(np.abs(np.sort(lines, axis=1) - np.sort(exact, axis=1))))

    fig, ax = plt.subplots(figsize=SIZE, layout="constrained")
    dense = np.linspace(0.0, 1.5, 200)
    for formula, offset in (
        (r"$\omega_d-\Omega$", -1),
        (r"$\omega_d$", 0),
        (r"$\omega_d+\Omega$", 1),
    ):
        ax.plot(dense, damped + offset * dense, "k-", lw=0.8)
        ax.annotate(
            formula,
            (1.5, damped + offset * 1.5),
            xytext=(6, -3),
            textcoords="offset points",
        )
    regressive = np.min(lines, axis=1)
    progressive = np.max(lines, axis=1)
    collective = np.median(lines, axis=1)
    ax.plot(speeds, regressive, "o", color="C0", ms=6, label="regressive (VANE)")
    ax.plot(speeds, collective, "s", color="C1", ms=6, label="collective (VANE)")
    ax.plot(speeds, progressive, "^", color="C2", ms=6, label="progressive (VANE)")
    ax.set_xlim(-0.05, 1.75)
    ax.set_xlabel(r"Rotor speed $\Omega$ (rad/s)")
    ax.set_ylabel("Damped frequency (rad/s)")
    ax.set_title(
        rf"Isotropic rotor, $\omega_0$ = 2 rad/s: largest deviation {worst:.0e} rad/s",
        loc="left",
    )
    ax.legend(loc="upper left")
    _save(fig, "mbc3_coleman_split.png")


def main() -> None:
    """Write every figure."""
    for figure in (
        figure_floquet_order,
        figure_azimuth_aliasing,
        figure_tangent_pinning,
        figure_coleman_split,
    ):
        with _figure_style():
            figure()
    for path in sorted(OUT.glob("*.png")):
        print(f"{path.name}: {path.stat().st_size / 1024:.1f} kB")


if __name__ == "__main__":
    main()
