# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Small synthetic OpenFAST linearization files for the VANE tutorials.

Real OpenFAST ``.lin`` files are not distributed with VANE. The tutorials instead
write small synthetic files in the current (ModGlue) ``.lin`` format, with
ElastoDyn-style channel descriptions and rotating-frame flags, so that every step of
the analysis runs in seconds on any machine.

The synthetic turbine has six structural degrees of freedom:

* the first tower fore-aft and side-side bending modes (fixed frame);
* the generator azimuth (``DOF_GeAz``), whose velocity is the rotor speed;
* the first flapwise bending mode of each of the three blades (rotating frame).

Each blade is a damped oscillator in its own rotating frame, with the centrifugal
(Southwell) stiffening :math:`\omega_r^2=\omega_0^2+K_{sp}\Omega^2`. The tower
fore-aft mode couples to the three blades equally, and the tower side-side mode
couples to each blade through the sine of its azimuth. A small 3P modulation of the
tower fore-aft stiffness makes the model periodic in azimuth. The matrices are
written in OpenFAST's mixed frame (tower states fixed, blade states rotating), so the
MBC3 transform, not this module, produces the collective, regressive and progressive
blade modes.

The numbers are chosen to resemble a 5 MW-class turbine but describe no real
machine. Two inputs (hub-height wind speed and collective pitch) and eight outputs
(tower-top accelerations, generator speed, tower-base moments and the three blade
root flap moments) give the model a ``B``, ``C`` and ``D`` matrix for the estimation
tutorial.
"""

from __future__ import annotations

import argparse
import math
import tempfile
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import numpy.typing as npt
    from matplotlib.figure import Figure

#: Channel specification: ``(description, rotating_frame)``.
Channel = tuple[str, bool]

BLADES = 3
#: Where the tutorial figures go when ``--output`` is not given.
FIGURES = Path(__file__).resolve().parent.parent / "docs" / "tutorials" / "figures"
_DERIVATIVE = "First time derivative of "

_DISPLACEMENTS: tuple[Channel, ...] = (
    (
        "ED 1st tower fore-aft bending mode DOF (internal DOF index = DOF_TFA1), m",
        False,
    ),
    (
        "ED 1st tower side-to-side bending mode DOF (internal DOF index = DOF_TSS1), m",
        False,
    ),
    ("ED Variable speed generator DOF (internal DOF index = DOF_GeAz), rad", False),
    *(
        (
            f"ED 1st flapwise bending-mode DOF of blade {k} "
            f"(internal DOF index = DOF_BF({k},1)), m",
            True,
        )
        for k in range(1, BLADES + 1)
    ),
)

#: The continuous states: six displacements, then their first time derivatives.
STATES: tuple[Channel, ...] = (
    *_DISPLACEMENTS,
    *(
        (desc.replace("ED ", "ED " + _DERIVATIVE, 1) + "/s", rot)
        for desc, rot in _DISPLACEMENTS
    ),
)

#: The inputs: hub-height wind speed and collective blade pitch.
INPUTS: tuple[Channel, ...] = (
    (
        "IfW Extended input: horizontal wind speed (steady/uniform wind) (hub), m/s",
        False,
    ),
    ("ED Extended input: collective blade-pitch command, rad", False),
)

#: The outputs: two sensors, generator speed, two loads, three blade loads.
OUTPUTS: tuple[Channel, ...] = (
    ("ED YawBrTAxp, (m/s^2)", False),
    ("ED YawBrTAyp, (m/s^2)", False),
    ("ED GenSpeed, (rpm)", False),
    ("ED TwrBsMyt, (kN-m)", False),
    ("ED TwrBsMxt, (kN-m)", False),
    *((f"ED RootMyc{k}, (kN-m)", True) for k in range(1, BLADES + 1)),
)


@dataclass(frozen=True)
class SyntheticTurbine:
    """A synthetic three-bladed turbine with tower and blade-flap modes.

    Parameters
    ----------
    tower_fore_aft_hz, tower_side_side_hz : float
        Natural frequencies of the first tower modes, Hz.
    flap_hz : float
        Natural frequency of the first blade flap mode at standstill, Hz.
    southwell : float
        Centrifugal stiffening coefficient of the blade flap mode.
    rotor_radius : float
        Rotor radius, m, used by the operating schedule.
    tip_speed_ratio : float
        Tip-speed ratio held below rated.
    rated_rpm : float
        Rated rotor speed, rpm.
    gearbox_ratio : float
        Gearbox ratio, for the generator-speed output.
    fore_aft_coupling, side_side_coupling : float
        Tower-to-blade stiffness coupling, as a fraction of the geometric mean of the
        two squared natural frequencies.
    harmonic_3p : float
        Relative 3P modulation of the tower fore-aft stiffness.
    """

    tower_fore_aft_hz: float = 0.336
    tower_side_side_hz: float = 0.303
    flap_hz: float = 0.68
    southwell: float = 1.6
    rotor_radius: float = 63.0
    tip_speed_ratio: float = 7.55
    rated_rpm: float = 12.1
    gearbox_ratio: float = 97.0
    fore_aft_coupling: float = 0.15
    side_side_coupling: float = 0.10
    harmonic_3p: float = 0.03

    @property
    def rated_wind_speed(self) -> float:
        """Return the wind speed at which the rotor reaches its rated speed, m/s.

        Returns
        -------
        float
            Rated wind speed of the schedule.
        """
        omega = self.rated_rpm * math.pi / 30.0
        return omega * self.rotor_radius / self.tip_speed_ratio

    def rotor_speed_rpm(self, wind_speed: float) -> float:
        """Return the scheduled rotor speed at a wind speed.

        Parameters
        ----------
        wind_speed : float
            Hub-height wind speed, m/s.

        Returns
        -------
        float
            Rotor speed, rpm: a constant tip-speed ratio below rated, the rated speed
            above it.
        """
        omega = self.tip_speed_ratio * wind_speed / self.rotor_radius
        return min(omega * 30.0 / math.pi, self.rated_rpm)

    def blade_pitch_deg(self, wind_speed: float) -> float:
        """Return the scheduled collective pitch at a wind speed.

        Parameters
        ----------
        wind_speed : float
            Hub-height wind speed, m/s.

        Returns
        -------
        float
            Pitch, deg: zero below rated, then an illustrative 2 deg per m/s.
        """
        return max(0.0, 2.0 * (wind_speed - self.rated_wind_speed))

    def matrices(
        self, wind_speed: float, azimuth: float
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Return the mixed-frame ``A``, ``B``, ``C`` and ``D`` at one azimuth.

        Parameters
        ----------
        wind_speed : float
            Hub-height wind speed, m/s; sets the rotor speed and the damping.
        azimuth : float
            Rotor azimuth of blade 1, rad.

        Returns
        -------
        tuple of npt.NDArray[np.float64]
            ``A`` (12 x 12), ``B`` (12 x 2), ``C`` (8 x 12) and ``D`` (8 x 2), in the
            order of :data:`STATES`, :data:`INPUTS` and :data:`OUTPUTS`.
        """
        omega = self.rotor_speed_rpm(wind_speed) * math.pi / 30.0
        w_fa = 2.0 * math.pi * self.tower_fore_aft_hz
        w_ss = 2.0 * math.pi * self.tower_side_side_hz
        w_rest = 2.0 * math.pi * self.flap_hz
        w_flap = math.sqrt(w_rest**2 + self.southwell * omega**2)
        blades = azimuth + 2.0 * math.pi / BLADES * np.arange(BLADES)
        n = 3 + BLADES
        flaps = slice(3, n)

        stiffness = np.zeros((n, n))
        stiffness[0, 0] = w_fa**2 * (1.0 + self.harmonic_3p * math.cos(3.0 * azimuth))
        stiffness[1, 1] = w_ss**2
        stiffness[flaps, flaps] = w_flap**2 * np.eye(BLADES)
        kappa_fa = self.fore_aft_coupling * w_fa * w_flap
        kappa_ss = self.side_side_coupling * w_ss * w_flap * np.sin(blades)
        stiffness[0, flaps] = stiffness[flaps, 0] = kappa_fa
        stiffness[1, flaps] = stiffness[flaps, 1] = kappa_ss

        # Aerodynamic damping grows with wind speed; the rotor speed is damped by
        # the slope of the aerodynamic torque.
        damping = np.diag(
            [
                2.0 * (0.02 + 0.004 * wind_speed) * w_fa,
                2.0 * 0.008 * w_ss,
                0.05,
                *([2.0 * (0.08 + 0.004 * wind_speed) * w_flap] * BLADES),
            ]
        )
        a = np.block([[np.zeros((n, n)), np.eye(n)], [-stiffness, -damping]])

        # Wind speed and pitch act on the tower, the rotor speed and each blade.
        b = np.zeros((2 * n, 2))
        b[n + 0] = [0.17, -5.7]
        b[n + 2] = [0.011, -0.34]
        b[n + 3 : 2 * n] = [0.5, -15.0]

        c = np.zeros((len(OUTPUTS), 2 * n))
        d = np.zeros((len(OUTPUTS), 2))
        c[0], d[0] = a[n + 0], b[n + 0]  # tower-top fore-aft acceleration
        c[1], d[1] = a[n + 1], b[n + 1]  # tower-top side-side acceleration
        c[2, n + 2] = self.gearbox_ratio * 30.0 / math.pi  # generator speed, rpm
        c[3, 0] = 2.0e5  # tower-base fore-aft moment, kN-m per m
        c[4, 1] = -2.0e5  # tower-base side-side moment, kN-m per m
        c[5:, 3:n] = 2.0e3 * np.eye(BLADES)  # blade-root flap moments, kN-m per m
        return a, b, c, d

    def write_operating_point(
        self,
        directory: str | Path,
        wind_speed: float,
        *,
        n_azimuths: int = 12,
        name: str | None = None,
    ) -> list[Path]:
        """Write the ``.lin`` files of one operating point.

        Parameters
        ----------
        directory : str or pathlib.Path
            Directory to write into (created when missing).
        wind_speed : float
            Hub-height wind speed, m/s.
        n_azimuths : int, optional
            Number of uniformly spaced azimuths over one revolution.
        name : str or None, optional
            Case name; the files are ``<name>.1.lin``, ``<name>.2.lin``, ...
            Default ``"ws<10 x wind speed>"``, e.g. ``"ws080"`` at 8 m/s.

        Returns
        -------
        list[pathlib.Path]
            The written files, in azimuth order.
        """
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        case = name or f"ws{round(10 * wind_speed):03d}"
        rpm = self.rotor_speed_rpm(wind_speed)
        omega = rpm * math.pi / 30.0
        pitch = math.radians(self.blade_pitch_deg(wind_speed))
        paths = []
        for k in range(n_azimuths):
            azimuth = 2.0 * math.pi * k / n_azimuths
            a, b, c, d = self.matrices(wind_speed, azimuth)
            states = [0.0] * len(STATES)
            states[2], states[6 + 2] = azimuth, omega
            outputs = [0.0] * len(OUTPUTS)
            outputs[2] = self.gearbox_ratio * rpm
            path = folder / f"{case}.{k + 1}.lin"
            write_lin_file(
                path,
                a=a,
                b=b,
                c=c,
                d=d,
                rotor_speed=omega,
                azimuth=azimuth,
                wind_speed=wind_speed,
                sim_time=60.0 + azimuth / omega,
                state_values=states,
                input_values=[wind_speed, pitch],
                output_values=outputs,
            )
            paths.append(path)
        return paths

    def write_sweep(
        self,
        directory: str | Path,
        wind_speeds: Sequence[float],
        *,
        n_azimuths: int = 12,
    ) -> list[list[Path]]:
        """Write the ``.lin`` files of several wind speeds into one directory.

        Parameters
        ----------
        directory : str or pathlib.Path
            Directory of the sweep. Each operating point is a case named
            ``ws<10 x wind speed>`` (``ws080.1.lin``, ``ws080.2.lin``, ...).
        wind_speeds : Sequence[float]
            Hub-height wind speeds, m/s.
        n_azimuths : int, optional
            Number of azimuths per operating point.

        Returns
        -------
        list[list[pathlib.Path]]
            The files of each operating point.
        """
        return [
            self.write_operating_point(directory, wind, n_azimuths=n_azimuths)
            for wind in wind_speeds
        ]


#: State descriptions of :func:`periodic_rotor_matrix`, in matrix order.
PERIODIC_STATES: tuple[str, ...] = (
    "ED Platform pitch tilt rotation DOF (internal DOF index = DOF_P), rad",
    "ED Variable speed generator DOF (internal DOF index = DOF_GeAz), rad",
    "ED First time derivative of Platform pitch tilt rotation DOF "
    "(internal DOF index = DOF_P), rad/s",
    "ED First time derivative of Variable speed generator DOF "
    "(internal DOF index = DOF_GeAz), rad/s",
)


def periodic_rotor_matrix(
    azimuth: float,
    *,
    rotor_speed: float = 1.0,
    amplitude: float = 0.05,
    decay: float = 0.6,
    n_harmonics: int = 8,
) -> npt.NDArray[np.float64]:
    r"""Return the state matrix of a periodic platform-rotor model at one azimuth.

    The four states are a platform pitch rotation :math:`q`, the rotor azimuth
    :math:`\psi` and their rates (``ndof2 = 2``; see :data:`PERIODIC_STATES`). The
    model has periodic content at the blade-passing orders :math:`3, 6, \ldots, 3M`
    with amplitudes falling geometrically by ``decay`` per order, as the MBC model of
    an isotropic three-bladed rotor does. It is built as the linearization about a
    periodic orbit: the orbit tangent :math:`\boldsymbol\xi(\psi)=[p, 1, \Omega p',
    0]` solves :math:`\mathbf A(\psi)\boldsymbol\xi=\Omega\boldsymbol\xi'`, so one
    Floquet exponent is exactly zero.

    Parameters
    ----------
    azimuth : float
        Rotor azimuth :math:`\psi`, rad.
    rotor_speed : float, optional
        Rotor speed :math:`\Omega`, rad/s.
    amplitude : float, optional
        Relative size of the periodic content.
    decay : float, optional
        Ratio of the content at order :math:`3(m+1)` to that at order :math:`3m`.
    n_harmonics : int, optional
        Number of blade-passing orders :math:`M`.

    Returns
    -------
    npt.NDArray[np.float64]
        The 4 x 4 state matrix :math:`\mathbf A(\psi)`.
    """
    omega = rotor_speed
    orders = 3.0 * np.arange(1, n_harmonics + 1)
    weights = amplitude * decay ** np.arange(n_harmonics)
    cosines, sines = np.cos(orders * azimuth), np.sin(orders * azimuth)
    p = float(np.sum(weights * (0.8 * cosines + 0.5 * sines)))
    dp = float(np.sum(weights * orders * (-0.8 * sines + 0.5 * cosines)))
    ddp = float(np.sum(-weights * orders**2 * (0.8 * cosines + 0.5 * sines)))
    modulation = float(np.sum(weights * cosines))
    stiffness = 0.09 * omega**2 * (1.0 + modulation)
    matrix = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-stiffness, 0.0, -0.02 * omega, 0.05 * omega],
            [0.02 * omega**2 * modulation, 0.0, 0.01 * omega, -0.2 * omega],
        ]
    )
    # The azimuth column is the one that makes the orbit tangent an exact solution.
    tangent = np.array([p, 1.0, omega * dp, 0.0])
    matrix[2:4, 1] = np.array([omega**2 * ddp, 0.0]) - matrix[2:4] @ tangent
    return matrix


def lin_file_text(
    *,
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    c: npt.NDArray[np.float64],
    d: npt.NDArray[np.float64],
    rotor_speed: float,
    azimuth: float,
    wind_speed: float,
    sim_time: float,
    state_values: Sequence[float],
    input_values: Sequence[float],
    output_values: Sequence[float],
    states: Sequence[Channel] = STATES,
    inputs: Sequence[Channel] = INPUTS,
    outputs: Sequence[Channel] = OUTPUTS,
) -> str:
    """Return the text of a ``.lin`` file in the current OpenFAST format.

    Parameters
    ----------
    a, b, c, d : npt.NDArray[np.float64]
        The state-space matrices in the order of ``states``, ``inputs`` and
        ``outputs``.
    rotor_speed : float
        Rotor speed, rad/s.
    azimuth : float
        Rotor azimuth, rad.
    wind_speed : float
        Hub-height wind speed, m/s.
    sim_time : float
        Simulation time of the snapshot, s.
    state_values, input_values, output_values : Sequence[float]
        Operating-point values of the states, inputs and outputs.
    states, inputs, outputs : Sequence[Channel], optional
        Channel descriptions and rotating-frame flags.

    Returns
    -------
    str
        The file contents.
    """
    lines = [
        "",
        "Linearized model: synthetic example written by the VANE tutorials",
        " linked with  NWTC Subroutine Library; ElastoDyn; InflowWind",
        "",
        "Description from the FAST input file: synthetic three-bladed turbine",
        "",
        "Simulation information:",
        f"   Simulation time:                {sim_time:10.4f} s",
        f"   Rotor Speed:                    {rotor_speed:10.4f} rad/s",
        f"   Azimuth:                        {azimuth:10.4f} rad",
        f"   Wind Speed:                     {wind_speed:10.4f} m/s",
        f"   Number of continuous states:    {len(states):5d}",
        "   Number of discrete states:           0",
        "   Number of constraint states:         0",
        f"   Number of inputs:               {len(inputs):5d}",
        f"   Number of outputs:              {len(outputs):5d}",
        "   Jacobians included in this file?    No",
        "",
    ]
    derivatives = [(f"{_DERIVATIVE}{desc}/s", rot) for desc, rot in states]
    rates = [0.0] * len(states)
    lines += _table("Order of continuous states:", states, state_values, 2)
    lines += _table("Order of continuous state derivatives:", derivatives, rates, 2)
    lines += _table("Order of inputs:", inputs, input_values, 0)
    lines += _table("Order of outputs:", outputs, output_values, 0)
    lines += ["Linearized state matrices:", ""]
    for label, matrix in (("A", a), ("B", b), ("C", c), ("D", d)):
        rows, cols = matrix.shape
        lines.append(f"{label}: {rows} x {cols}")
        lines += ["  " + "  ".join(f"{v: .12E}" for v in row) for row in matrix]
        lines.append("")
    return "\n".join(lines) + "\n"


def write_lin_file(path: str | Path, **content: object) -> Path:
    """Write a ``.lin`` file built by :func:`lin_file_text`.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination file.
    **content : object
        Keyword arguments of :func:`lin_file_text`.

    Returns
    -------
    pathlib.Path
        The written file.
    """
    target = Path(path)
    target.write_text(lin_file_text(**content), encoding="utf-8")  # type: ignore[arg-type]
    return target


def _table(
    label: str, channels: Sequence[Channel], values: Sequence[float], order: int
) -> list[str]:
    """Render one operating-point table of a ``.lin`` file."""
    rows = [
        label,
        "   Row/Column Operating Point         Rotating Frame? Derivative Order "
        "Description",
        "   ---------- ---------------         --------------- ---------------- "
        "-----------",
    ]
    for k, ((desc, rotating), value) in enumerate(
        zip(channels, values, strict=True), start=1
    ):
        flag = "T" if rotating else "F"
        rows.append(f"   {k:8d}    {value: .12E}        {flag}   {order:13d}   {desc}")
    rows.append("")
    return rows


def output_parser(
    description: str, *, output_help: str | None = None
) -> argparse.ArgumentParser:
    """Return the command-line parser shared by the tutorial scripts.

    Parameters
    ----------
    description : str
        The script description shown by ``--help``.
    output_help : str or None, optional
        Help text of ``--output``; the default describes
        :func:`output_directories`.

    Returns
    -------
    argparse.ArgumentParser
        A parser with the ``--output`` option.
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=output_help
        or (
            "directory for every file the script writes; by default the generated "
            "data go to a temporary directory that is deleted at the end and the "
            "figures to docs/tutorials/figures"
        ),
    )
    return parser


@contextmanager
def output_directories(output: Path | None) -> Iterator[tuple[Path, Path]]:
    """Yield the data and figure directories of a tutorial run.

    Parameters
    ----------
    output : pathlib.Path or None
        The ``--output`` directory. When given, the data go to ``output / "data"``
        and the figures to ``output``. When ``None``, the data go to a temporary
        directory, removed on exit, and the figures to :data:`FIGURES`.

    Yields
    ------
    tuple[pathlib.Path, pathlib.Path]
        The data directory and the figure directory, both existing.
    """
    if output is None:
        FIGURES.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="vane_tutorial_") as scratch:
            yield Path(scratch), FIGURES
    else:
        data = output / "data"
        data.mkdir(parents=True, exist_ok=True)
        yield data, output


def figure_style() -> AbstractContextManager[object]:
    """Return the context manager that styles the tutorial figures.

    Returns
    -------
    contextlib.AbstractContextManager
        :func:`vane.viz.plot_style` when the installed VANE provides it, otherwise
        a context manager that leaves the matplotlib defaults unchanged.
    """
    try:
        from vane.viz import plot_style
    except ImportError:
        return nullcontext()
    return plot_style()


def save_figure(figure: Figure, path: Path) -> None:
    """Save a tutorial figure as a small PNG and close it.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure.
    path : pathlib.Path
        Destination ``.png`` file.
    """
    import matplotlib.pyplot as plt

    figure.savefig(path, dpi=110)
    plt.close(figure)
    print(f"Figure written: {path.name}")
