# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""State-space radiation fits generated from WAMIT first-order radiation data.

A linearization of a floating platform needs HydroDyn's state-space radiation model
(``RdtnMod = 2``): the convolution model is not linearizable. That model reads a
``<PotFile>.ss`` file, which some published decks do not ship (e.g. the IEA 22-MW
semi-submersible ships only the WAMIT ``.1``/``.3``/``.hst`` data). Without it the
radiation memory must be switched off, leaving only the infinite-frequency added mass
and no wave-radiation damping. This module generates the missing file from the
WAMIT ``.1`` data itself.

**The fit.** The radiation impulse-response function ``K_ij(t)`` relates the
radiation force on DOF ``i`` to the velocity of DOF ``j``; its transfer function is

.. math::

    K_{ij}(j\omega) = B_{ij}(\omega) + j\omega \left[A_{ij}(\omega) -
    A_{ij}(\infty)\right]

with ``A`` and ``B`` the frequency-dependent added mass and radiation damping. Each
significant entry (every diagonal entry, and an off-diagonal one when its peak
exceeds ``coupling_threshold`` of the geometric mean of the two diagonal peaks) is
fitted in the frequency domain by a strictly proper rational function with stable
poles, by vector fitting [Gustavsen1999]_: the poles are relocated iteratively
(unstable ones reflected into the left half-plane), and the residues of the final fit
satisfy ``K_ij(0) = 0`` exactly (the radiation force has no static part) by a
constrained least-squares solve. The order grows until the relative error over the
WAMIT frequencies, ``||K_fit - K|| / ||K||``, falls below ``tolerance``. Each entry is
realised in real modal form, the blocks grouped by input DOF as HydroDyn expects, and
the output sign is that of the radiation *force* (``F = -K * v``).

**Validation.** The IEA 15-MW VolturnUS-S ships both WAMIT data and an ``.ss`` file.
The generated fit matches the WAMIT data within 3.5--5 % per entry (the shipped fit
within 10--21 %), its impulse responses deviate from the WAMIT cosine transform of
the damping by 2--9 % of the peak (the shipped fit by 9--25 %), and a standstill
linearization with it reproduces the platform frequencies of one with the shipped
file within 1 % and their damping ratios within 0.003.

Wave-excitation fits (``.ssexctn``) are *not* generated. A linearization runs in
still water, where the excitation states are driven by nothing and leave the
platform dynamics unchanged, so ``ExctnMod = 0`` loses only the wave-elevation input
channel of the linear model.

References
----------
.. [Gustavsen1999] Gustavsen, B., & Semlyen, A. (1999). Rational approximation of
   frequency domain responses by vector fitting. *IEEE Transactions on Power
   Delivery*, 14(3), 1052-1061.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg

from vane.io.fst_writer import FastInputError, FastInputFile
from vane.run.lin_config import _deck_float
from vane.run.steady_state import _replace_file
from vane.utils.paths import resolve_path

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = [
    "RadiationFitError",
    "RadiationFitRecord",
    "RadiationPairFit",
    "RadiationStateSpace",
    "WamitRadiation",
    "ensure_radiation_state_space",
    "fit_radiation_state_space",
    "preview_radiation_state_space",
    "read_radiation_state_space",
    "read_wamit_radiation",
    "write_radiation_state_space",
]

logger = logging.getLogger(__name__)

_N_DOF = 6
# Largest acceptable relative fit error for a generated file to be written.
_MAX_ACCEPTED_ERROR = 0.2


class RadiationFitError(ValueError):
    """Raised when radiation data cannot be read or fitted."""


@dataclass(frozen=True)
class WamitRadiation:
    """Dimensional frequency-dependent radiation coefficients of one body.

    Parameters
    ----------
    frequencies : npt.NDArray[np.float64]
        Angular frequencies (rad/s), positive and ascending, shape ``(n,)``.
    added_mass : npt.NDArray[np.float64]
        Added mass ``A(omega)``, shape ``(n, 6, 6)`` (kg, kg m, kg m^2).
    damping : npt.NDArray[np.float64]
        Radiation damping ``B(omega)``, shape ``(n, 6, 6)``.
    added_mass_infinite : npt.NDArray[np.float64]
        Infinite-frequency added mass ``A(inf)``, shape ``(6, 6)``.
    """

    frequencies: npt.NDArray[np.float64]
    added_mass: npt.NDArray[np.float64]
    damping: npt.NDArray[np.float64]
    added_mass_infinite: npt.NDArray[np.float64]

    def kernel(self) -> npt.NDArray[np.complex128]:
        """Return the radiation kernel transfer ``B + j omega (A - A_inf)``.

        Returns
        -------
        npt.NDArray[np.complex128]
            Shape ``(n, 6, 6)``.
        """
        omega = self.frequencies[:, None, None]
        return self.damping + 1j * omega * (
            self.added_mass - self.added_mass_infinite[None, :, :]
        )


def _wamit_scale(row: int, column: int, length: float) -> float:
    """Return WAMIT's ``L^k`` (k = 3, 4 or 5) for a 0-based DOF pair."""
    rotations = (row >= 3) + (column >= 3)
    return float(length ** (3 + rotations))


def read_wamit_radiation(
    path: str | Path, *, water_density: float, length_scale: float = 1.0
) -> WamitRadiation:
    """Read a WAMIT ``.1`` file (single body) into dimensional coefficients.

    Each row is ``PER I J Abar [Bbar]``. ``PER = 0`` is the infinite-frequency
    (zero-period) added mass and ``PER < 0`` the zero-frequency one; other rows are
    at ``omega = 2 pi / PER``. The non-dimensional values are scaled by
    ``rho L^k`` (added mass) and ``rho omega L^k`` (damping), ``k = 3`` for two
    translations, ``4`` for mixed and ``5`` for two rotations.

    Parameters
    ----------
    path : str or pathlib.Path
        The ``.1`` file.
    water_density : float
        Water density ``rho`` (kg/m^3).
    length_scale : float, optional
        WAMIT's characteristic length ``L`` (HydroDyn ``WAMITULEN``). Default ``1``.

    Returns
    -------
    WamitRadiation
        The coefficients at the finite positive frequencies, and ``A(inf)``.

    Raises
    ------
    RadiationFitError
        If the file has DOF indices beyond 6 (more than one body), no
        infinite-frequency rows, fewer than four finite frequencies, or a malformed
        row; or if ``water_density``/``length_scale`` is not positive and finite.
    """
    for name, value in (("water_density", water_density), ("length", length_scale)):
        if not math.isfinite(value) or value <= 0.0:
            msg = f"{name} must be positive and finite, got {value}"
            raise RadiationFitError(msg)
    infinite = np.zeros((_N_DOF, _N_DOF))
    has_infinite = False
    added: dict[float, npt.NDArray[np.float64]] = {}
    damped: dict[float, npt.NDArray[np.float64]] = {}
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        tokens = line.split()
        if not tokens:
            continue
        try:
            period = float(tokens[0])
            row, column = int(tokens[1]) - 1, int(tokens[2]) - 1
            a_bar = float(tokens[3])
            b_bar = float(tokens[4]) if len(tokens) > 4 else 0.0
        except (IndexError, ValueError) as error:
            msg = f"malformed WAMIT .1 row {number} in {path}: {line!r}"
            raise RadiationFitError(msg) from error
        if not (0 <= row < _N_DOF and 0 <= column < _N_DOF):
            msg = (
                f"WAMIT .1 row {number} in {path} names DOF ({row + 1}, {column + 1}); "
                "only a single body (DOFs 1-6) is supported"
            )
            raise RadiationFitError(msg)
        scale = water_density * _wamit_scale(row, column, length_scale)
        if period == 0.0:
            infinite[row, column] = a_bar * scale
            has_infinite = True
        elif period > 0.0:
            omega = 2.0 * math.pi / period
            added.setdefault(omega, np.zeros((_N_DOF, _N_DOF)))[row, column] = (
                a_bar * scale
            )
            damped.setdefault(omega, np.zeros((_N_DOF, _N_DOF)))[row, column] = (
                b_bar * scale * omega
            )
    if not has_infinite:
        msg = f"{path} has no infinite-frequency (PER = 0) added-mass rows"
        raise RadiationFitError(msg)
    omegas = sorted(added)
    if len(omegas) < 4:
        msg = f"{path} has {len(omegas)} finite frequencies; at least 4 are needed"
        raise RadiationFitError(msg)
    return WamitRadiation(
        frequencies=np.asarray(omegas, dtype=np.float64),
        added_mass=np.stack([added[w] for w in omegas]),
        damping=np.stack([damped[w] for w in omegas]),
        added_mass_infinite=infinite,
    )


@dataclass(frozen=True)
class RadiationPairFit:
    """The rational fit of one radiation-kernel entry ``K_ij``.

    Parameters
    ----------
    row, column : int
        The 0-based force DOF ``i`` and velocity DOF ``j``.
    poles : tuple of complex
        The stable poles (rad/s); one of each complex-conjugate pair.
    coefficients : tuple of float
        Real partial-fraction coefficients: one per real pole, two (real and
        imaginary part of the residue) per complex pair.
    n_states : int
        The order of the fit (states in its realisation).
    relative_error : float
        ``||K_fit - K|| / ||K||`` over the WAMIT frequencies.
    """

    row: int
    column: int
    poles: tuple[complex, ...]
    coefficients: tuple[float, ...]
    n_states: int
    relative_error: float

    def evaluate(
        self, frequencies: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.complex128]:
        """Return the fitted ``K_ij(j omega)`` at ``frequencies`` (rad/s).

        Parameters
        ----------
        frequencies : npt.NDArray[np.float64]
            Angular frequencies (rad/s).

        Returns
        -------
        npt.NDArray[np.complex128]
            The fitted kernel transfer.
        """
        s = np.asarray(1j * np.asarray(frequencies, dtype=np.float64), np.complex128)
        basis = _basis(s, self.poles)
        return np.asarray(basis @ np.asarray(self.coefficients), dtype=np.complex128)

    def realization(
        self,
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        """Return a real modal realisation ``(A, b, c)`` of the fitted ``K_ij``.

        Returns
        -------
        tuple of npt.NDArray[np.float64]
            ``A`` (``n x n``), ``b`` (``n``) and ``c`` (``n``) with
            ``K_ij(s) = c (sI - A)^-1 b``.
        """
        blocks: list[npt.NDArray[np.float64]] = []
        b: list[float] = []
        c: list[float] = []
        index = 0
        for pole in self.poles:
            if _is_real(pole):
                blocks.append(np.array([[pole.real]]))
                b.append(1.0)
                c.append(self.coefficients[index])
                index += 1
            else:
                sigma, omega = pole.real, pole.imag
                blocks.append(np.array([[sigma, -omega], [omega, sigma]]))
                b.extend((1.0, 0.0))
                c.extend(
                    (
                        2.0 * self.coefficients[index],
                        -2.0 * self.coefficients[index + 1],
                    )
                )
                index += 2
        return (
            np.asarray(scipy.linalg.block_diag(*blocks), dtype=np.float64),
            np.asarray(b, dtype=np.float64),
            np.asarray(c, dtype=np.float64),
        )


@dataclass(frozen=True)
class RadiationStateSpace:
    """A HydroDyn state-space radiation model (the content of a ``.ss`` file).

    The model maps the six platform velocities to the six radiation forces:
    ``x' = A x + B v``, ``F = C x``.

    Parameters
    ----------
    a : npt.NDArray[np.float64]
        State matrix, ``(n, n)``.
    b : npt.NDArray[np.float64]
        Input matrix, ``(n, 6)``.
    c : npt.NDArray[np.float64]
        Output matrix, ``(6, n)``.
    states_per_dof : tuple of int
        States belonging to each input DOF (the blocks are grouped by input DOF).
    pairs : tuple of RadiationPairFit, optional
        The per-entry fits (empty for a model read from a file).
    """

    a: npt.NDArray[np.float64]
    b: npt.NDArray[np.float64]
    c: npt.NDArray[np.float64]
    states_per_dof: tuple[int, ...]
    pairs: tuple[RadiationPairFit, ...] = ()

    @property
    def n_states(self) -> int:
        """Return the number of radiation states."""
        return int(self.a.shape[0])

    @property
    def max_relative_error(self) -> float:
        """Return the largest per-entry relative fit error (``nan`` if unknown)."""
        if not self.pairs:
            return float("nan")
        return max(pair.relative_error for pair in self.pairs)

    def force_response(
        self, frequencies: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.complex128]:
        """Return the velocity-to-radiation-force transfer ``C (j omega I - A)^-1 B``.

        For an exact model this equals ``-K(j omega)``.

        Parameters
        ----------
        frequencies : npt.NDArray[np.float64]
            Angular frequencies (rad/s).

        Returns
        -------
        npt.NDArray[np.complex128]
            Shape ``(n_frequencies, 6, 6)``.
        """
        identity = np.eye(self.n_states)
        return np.stack(
            [
                self.c @ np.linalg.solve(1j * omega * identity - self.a, self.b)
                for omega in np.asarray(frequencies, dtype=np.float64)
            ]
        )


def _is_real(pole: complex) -> bool:
    """Return whether a pole is (numerically) real."""
    return abs(pole.imag) <= 1e-12 * max(abs(pole), 1.0)


def _basis(
    s: npt.NDArray[np.complex128], poles: tuple[complex, ...] | list[complex]
) -> npt.NDArray[np.complex128]:
    """Return the real-coefficient partial-fraction basis at ``s``.

    A real pole ``p`` gives ``1/(s - p)``; a complex pair ``(p, p*)`` gives
    ``1/(s-p) + 1/(s-p*)`` and ``j/(s-p) - j/(s-p*)``.
    """
    columns: list[npt.NDArray[np.complex128]] = []
    for pole in poles:
        if _is_real(pole):
            columns.append(1.0 / (s - pole.real))
        else:
            columns.append(1.0 / (s - pole) + 1.0 / (s - pole.conjugate()))
            columns.append(1j / (s - pole) - 1j / (s - pole.conjugate()))
    return np.column_stack(columns)


def _relocated_poles(
    poles: list[complex], sigma_coefficients: npt.NDArray[np.float64]
) -> list[complex]:
    """Return the zeros of ``sigma(s) = 1 + sum c_k basis_k(s)``, stabilised."""
    blocks: list[npt.NDArray[np.float64]] = []
    b: list[float] = []
    for pole in poles:
        if _is_real(pole):
            blocks.append(np.array([[pole.real]]))
            b.append(1.0)
        else:
            blocks.append(np.array([[pole.real, pole.imag], [-pole.imag, pole.real]]))
            b.extend((2.0, 0.0))
    a = scipy.linalg.block_diag(*blocks)
    zeros = np.linalg.eigvals(a - np.outer(np.asarray(b), sigma_coefficients))
    relocated: list[complex] = []
    for zero in zeros:
        if zero.imag < -1e-12 * max(abs(zero), 1.0):
            continue  # keep one of each conjugate pair
        pole = complex(zero.real, max(zero.imag, 0.0))
        if pole.real >= 0.0:
            # Reflect into the left half-plane; a pole on the axis is nudged left.
            pole = complex(-abs(pole.real) or -1e-6 * max(abs(pole), 1.0), pole.imag)
        relocated.append(pole)
    return relocated


def _fit_entry(
    frequencies: npt.NDArray[np.float64],
    kernel: npt.NDArray[np.complex128],
    n_pairs: int,
    iterations: int,
) -> tuple[list[complex], npt.NDArray[np.float64], float]:
    """Vector-fit one kernel entry with ``n_pairs`` starting pole pairs.

    Returns the poles, the real basis coefficients and the relative error, all in
    the original (unscaled) units.
    """
    omega_scale = float(frequencies.max())
    value_scale = float(np.abs(kernel).max())
    s = np.asarray(1j * frequencies / omega_scale, dtype=np.complex128)
    y = kernel / value_scale
    start = np.linspace(frequencies.min() / omega_scale, 1.0, n_pairs)
    poles: list[complex] = [complex(-beta / 100.0, beta) for beta in start]
    rows_real = np.concatenate([y.real, y.imag])
    for _ in range(iterations):
        basis = _basis(s, poles)
        system = np.hstack([basis, -y[:, None] * basis])
        stacked = np.vstack([system.real, system.imag])
        norms = np.linalg.norm(stacked, axis=0)
        norms[norms == 0.0] = 1.0
        solution = np.linalg.lstsq(stacked / norms, rows_real, rcond=None)[0] / norms
        poles = _relocated_poles(poles, solution[basis.shape[1] :])
    # Final residues: least squares with the constraint K(0) = 0.
    basis = _basis(s, poles)
    at_zero = _basis(np.zeros(1, dtype=np.complex128), poles)[0].real
    stacked = np.vstack([basis.real, basis.imag])
    size = stacked.shape[1]
    kkt = np.zeros((size + 1, size + 1))
    kkt[:size, :size] = stacked.T @ stacked
    kkt[:size, size] = at_zero
    kkt[size, :size] = at_zero
    rhs = np.concatenate([stacked.T @ rows_real, [0.0]])
    coefficients = np.linalg.lstsq(kkt, rhs, rcond=None)[0][:size]
    fitted = basis @ coefficients
    error = float(np.linalg.norm(fitted - y) / np.linalg.norm(y))
    scaled_poles = [pole * omega_scale for pole in poles]
    return scaled_poles, coefficients * value_scale * omega_scale, error


def _fit_pair(
    frequencies: npt.NDArray[np.float64],
    kernel: npt.NDArray[np.complex128],
    row: int,
    column: int,
    tolerance: float,
    max_states: int,
    iterations: int,
) -> RadiationPairFit:
    """Fit one entry at the lowest order meeting ``tolerance`` (else the best)."""
    best: RadiationPairFit | None = None
    for n_pairs in range(1, max_states // 2 + 1):
        poles, coefficients, error = _fit_entry(
            frequencies, kernel, n_pairs, iterations
        )
        if not np.all(np.isfinite(coefficients)) or not math.isfinite(error):
            continue
        n_states = sum(1 if _is_real(pole) else 2 for pole in poles)
        fit = RadiationPairFit(
            row=row,
            column=column,
            poles=tuple(poles),
            coefficients=tuple(float(value) for value in coefficients),
            n_states=n_states,
            relative_error=error,
        )
        if best is None or fit.relative_error < best.relative_error:
            best = fit
        if error <= tolerance:
            break
    if best is None:
        msg = f"could not fit radiation kernel entry K{row + 1}{column + 1}"
        raise RadiationFitError(msg)
    return best


def fit_radiation_state_space(
    radiation: WamitRadiation,
    *,
    tolerance: float = 0.05,
    max_states_per_entry: int = 20,
    coupling_threshold: float = 0.05,
    iterations: int = 20,
) -> RadiationStateSpace:
    """Fit a state-space radiation model to WAMIT radiation coefficients.

    See the module notes for the method.

    Parameters
    ----------
    radiation : WamitRadiation
        The dimensional radiation coefficients.
    tolerance : float, optional
        Target relative error of each fitted entry over the WAMIT frequencies; the
        order grows (two states at a time) until it is met. Default ``0.05``.
    max_states_per_entry : int, optional
        Largest order tried per entry. Default ``20``.
    coupling_threshold : float, optional
        An off-diagonal entry is fitted when its peak magnitude is at least this
        fraction of the geometric mean of the two diagonal peaks. Default ``0.05``.
    iterations : int, optional
        Pole-relocation iterations per order. Default ``20``.

    Returns
    -------
    RadiationStateSpace
        The assembled model, its blocks grouped by input DOF, with the per-entry
        fits.

    Raises
    ------
    RadiationFitError
        If an argument is out of range or an entry cannot be fitted.
    """
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        msg = f"tolerance must be positive and finite, got {tolerance}"
        raise RadiationFitError(msg)
    if max_states_per_entry < 2:
        msg = f"max_states_per_entry must be >= 2, got {max_states_per_entry}"
        raise RadiationFitError(msg)
    kernel = radiation.kernel()
    peaks = np.abs(kernel).max(axis=0)
    pairs: list[RadiationPairFit] = []
    for column in range(_N_DOF):
        for row in range(_N_DOF):
            peak = peaks[row, column]
            if peak <= 0.0:
                continue
            if row != column:
                scale = math.sqrt(peaks[row, row] * peaks[column, column])
                if peak < coupling_threshold * scale:
                    continue
            pairs.append(
                _fit_pair(
                    radiation.frequencies,
                    kernel[:, row, column],
                    row,
                    column,
                    tolerance,
                    max_states_per_entry,
                    iterations,
                )
            )
    if not pairs:
        msg = "the radiation data are zero; nothing to fit"
        raise RadiationFitError(msg)
    blocks: list[npt.NDArray[np.float64]] = []
    b_columns: list[tuple[int, npt.NDArray[np.float64]]] = []
    c_rows: list[tuple[int, npt.NDArray[np.float64]]] = []
    states_per_dof = [0] * _N_DOF
    for pair in pairs:  # already grouped by input DOF (column)
        a_pair, b_pair, c_pair = pair.realization()
        blocks.append(a_pair)
        b_columns.append((pair.column, b_pair))
        # HydroDyn's output is the radiation force, -K * velocity.
        c_rows.append((pair.row, -c_pair))
        states_per_dof[pair.column] += pair.n_states
    a = np.asarray(scipy.linalg.block_diag(*blocks), dtype=np.float64)
    n_states = a.shape[0]
    b = np.zeros((n_states, _N_DOF))
    c = np.zeros((_N_DOF, n_states))
    offset = 0
    for (column, b_pair), (row, c_pair) in zip(b_columns, c_rows, strict=True):
        size = b_pair.size
        b[offset : offset + size, column] = b_pair
        c[row, offset : offset + size] = c_pair
        offset += size
    return RadiationStateSpace(
        a=a,
        b=b,
        c=c,
        states_per_dof=tuple(states_per_dof),
        pairs=tuple(pairs),
    )


def write_radiation_state_space(
    model: RadiationStateSpace, path: str | Path, *, header: str = ""
) -> Path:
    """Write a model in HydroDyn's ``.ss`` format.

    The file holds a header line, the six enabled-DOF flags, the state count, the
    states per DOF, then the rows of ``A``, ``B`` and ``C``. It is written through
    an atomic replace, so a hard-linked copy never writes through to a template.

    Parameters
    ----------
    model : RadiationStateSpace
        The model to write.
    path : str or pathlib.Path
        Destination ``.ss`` file.
    header : str, optional
        Text of the first line.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    destination = Path(path)
    lines = [
        header.replace("\n", " ") or "State-space radiation model fitted to WAMIT data",
        " ".join("1" for _ in range(_N_DOF)) + "\t%Enabled DoFs",
        f"{model.n_states}\t%Radiation States",
        " ".join(str(count) for count in model.states_per_dof)
        + "\t%Radiation States per DoF",
    ]
    for matrix in (model.a, model.b, model.c):
        lines.extend(" ".join(f"{value:.9e}" for value in row) for row in matrix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _replace_file(destination, ("\n".join(lines) + "\n").encode("utf-8"))
    return destination


def _leading_integers(line: str) -> list[int]:
    """Return the integers that start ``line``, up to its first non-integer token."""
    values: list[int] = []
    for token in line.split():
        if not re.fullmatch(r"[+-]?\d+", token):
            break
        values.append(int(token))
    return values


def read_radiation_state_space(path: str | Path) -> RadiationStateSpace:
    """Read a HydroDyn ``.ss`` radiation file.

    Parameters
    ----------
    path : str or pathlib.Path
        The ``.ss`` file.

    Returns
    -------
    RadiationStateSpace
        The model (with no per-entry fits).

    Raises
    ------
    RadiationFitError
        If the file is malformed.
    """
    lines = [line for line in Path(path).read_text().splitlines() if line.strip()]
    try:
        n_states = int(lines[2].split()[0])
        per_dof = tuple(_leading_integers(lines[3]))
        rows = [
            [float(token) for token in line.split()]
            for line in lines[4 : 4 + 2 * n_states + _N_DOF]
        ]
        a = np.asarray(rows[:n_states], dtype=np.float64)
        b = np.asarray(rows[n_states : 2 * n_states], dtype=np.float64)
        c = np.asarray(rows[2 * n_states :], dtype=np.float64)
    except (IndexError, ValueError) as error:
        msg = f"malformed state-space radiation file {path}"
        raise RadiationFitError(msg) from error
    if (
        a.shape != (n_states, n_states)
        or b.shape != (n_states, _N_DOF)
        or c.shape != (_N_DOF, n_states)
    ):
        msg = f"state-space radiation file {path} has inconsistent matrix sizes"
        raise RadiationFitError(msg)
    # The per-DOF counts split the states among the platform-velocity inputs; HydroDyn
    # orders the states by DOF, so they must be six non-negative counts that add up.
    if (
        len(per_dof) != _N_DOF
        or any(count < 0 for count in per_dof)
        or sum(per_dof) != n_states
    ):
        msg = (
            f"state-space radiation file {path} lists radiation states per DOF "
            f"{list(per_dof)}, which are not {_N_DOF} non-negative counts adding up to "
            f"its {n_states} states"
        )
        raise RadiationFitError(msg)
    return RadiationStateSpace(a=a, b=b, c=c, states_per_dof=per_dof)


@dataclass(frozen=True)
class RadiationFitRecord:
    """Provenance of a generated ``.ss`` radiation file.

    Parameters
    ----------
    path : pathlib.Path
        The ``.ss`` file (written only when ``written``).
    source : pathlib.Path
        The WAMIT ``.1`` file it was fitted to.
    n_states : int
        Radiation states of the model.
    max_relative_error : float
        Largest per-entry relative fit error.
    entry_errors : dict[str, float]
        Relative fit error per fitted entry (``"K11"``, ``"K15"``, ...).
    written : bool
        Whether the file was written (the fit met the acceptance limit).
    """

    path: Path
    source: Path
    n_states: int
    max_relative_error: float
    entry_errors: dict[str, float]
    written: bool

    @property
    def accepted(self) -> bool:
        """Return whether every fitted entry is within the acceptance limit (0.2).

        Returns
        -------
        bool
            ``True`` when the fit is accurate enough to be written.
        """
        return self.max_relative_error <= _MAX_ACCEPTED_ERROR

    def to_dict(self) -> dict[str, object]:
        """Return the record as a JSON-ready dictionary.

        Returns
        -------
        dict[str, object]
            The fields, with the paths as strings.
        """
        return {
            "path": str(self.path),
            "source": str(self.source),
            "n_states": self.n_states,
            "max_relative_error": self.max_relative_error,
            "entry_errors": dict(self.entry_errors),
            "written": self.written,
        }


def _water_density(deck: FastInputFile, hydro: FastInputFile) -> float:
    """Return the water density from the primary deck, HydroDyn or SeaState."""
    for source in (deck, hydro):
        value = _deck_float(source, "WtrDens")
        if value is not None and value > 0.0:
            return value
    if "SeaStFile" in deck:
        try:
            sea_state = FastInputFile.read(deck.resolve("SeaStFile"))
        except (OSError, FastInputError):
            sea_state = None
        if sea_state is not None:
            value = _deck_float(sea_state, "WtrDens")
            if value is not None and value > 0.0:
                return value
    msg = f"no positive WtrDens found for {deck.path}"
    raise RadiationFitError(msg)


def _wamit_radiation_paths(
    fst_path: str | Path,
) -> tuple[FastInputFile, FastInputFile, Path, Path] | None:
    """Return the deck, HydroDyn file, ``.ss`` target and ``.1`` source of a deck.

    ``None`` when HydroDyn is off, unreadable or uses no WAMIT potential flow.
    """
    deck = FastInputFile.read(fst_path)
    if (_deck_float(deck, "CompHydro") or 0.0) == 0.0 or "HydroFile" not in deck:
        return None
    try:
        hydro = FastInputFile.read(deck.resolve("HydroFile"))
    except (OSError, FastInputError):
        return None
    if _deck_float(hydro, "PotMod") != 1.0:
        return None
    n_body = _deck_float(hydro, "NBody") or 1.0
    if n_body != 1.0:
        msg = f"{hydro.path} uses {n_body:g} WAMIT bodies; only one is supported"
        raise RadiationFitError(msg)
    try:
        root = hydro.resolve("PotFile")
    except FastInputError as error:
        msg = f"{hydro.path} names no usable WAMIT PotFile: {error}"
        raise RadiationFitError(msg) from error
    return (
        deck,
        hydro,
        root.with_name(root.name + ".ss"),
        root.with_name(root.name + ".1"),
    )


def _fit_record(
    deck: FastInputFile,
    hydro: FastInputFile,
    target: Path,
    source: Path,
    tolerance: float,
) -> tuple[RadiationStateSpace, RadiationFitRecord]:
    """Fit the deck's ``.1`` data; return the model and an unwritten record."""
    if not source.is_file():
        msg = f"cannot fit {target.name}: WAMIT radiation file {source} not found"
        raise RadiationFitError(msg)
    radiation = read_wamit_radiation(
        source,
        water_density=_water_density(deck, hydro),
        length_scale=_deck_float(hydro, "WAMITULEN") or 1.0,
    )
    model = fit_radiation_state_space(radiation, tolerance=tolerance)
    record = RadiationFitRecord(
        path=target,
        source=source,
        n_states=model.n_states,
        max_relative_error=model.max_relative_error,
        entry_errors={
            f"K{pair.row + 1}{pair.column + 1}": pair.relative_error
            for pair in model.pairs
        },
        written=False,
    )
    return model, record


def preview_radiation_state_space(
    fst_path: str | Path, *, tolerance: float = 0.05
) -> RadiationFitRecord | None:
    """Fit a deck's WAMIT ``.1`` radiation data without writing anything.

    The fit :func:`ensure_radiation_state_space` would write, for deciding before any
    case is copied whether the ``.ss`` file can be generated
    (:attr:`RadiationFitRecord.accepted`). An existing ``.ss`` file is ignored.

    Parameters
    ----------
    fst_path : str or pathlib.Path
        The primary ``.fst`` deck (only read).
    tolerance : float, optional
        Per-entry fit tolerance (see :func:`fit_radiation_state_space`).

    Returns
    -------
    RadiationFitRecord or None
        The fit's record (``written`` is ``False``), or ``None`` when HydroDyn is off
        or uses no WAMIT potential flow.

    Raises
    ------
    RadiationFitError
        As :func:`ensure_radiation_state_space`.
    """
    located = _wamit_radiation_paths(fst_path)
    if located is None:
        return None
    _, record = _fit_record(*located, tolerance)
    return record


def ensure_radiation_state_space(
    fst_path: str | Path,
    *,
    tolerance: float = 0.05,
    overwrite: bool = False,
    confine_to: str | Path | None = None,
) -> RadiationFitRecord | None:
    """Generate the ``<PotFile>.ss`` radiation file a deck lacks, from its ``.1``.

    Call it on a working copy before a linearization with ``RdtnMod = 2``. Nothing
    is done when HydroDyn is off, uses no WAMIT potential flow (``PotMod != 1``),
    or the ``.ss`` file already exists (unless ``overwrite``). The file is written
    only when every fitted entry is within ``0.2`` relative error; otherwise the
    record says so, nothing is written, and the caller should linearize with the
    radiation memory off (``hydro_radiation_mode=0``).

    Parameters
    ----------
    fst_path : str or pathlib.Path
        The primary ``.fst`` deck (a working copy; the file is written next to the
        WAMIT data it references).
    tolerance : float, optional
        Per-entry fit tolerance (see :func:`fit_radiation_state_space`).
    overwrite : bool, optional
        Refit and replace an existing ``.ss`` file. Default ``False``.
    confine_to : str or pathlib.Path or None, optional
        A directory the ``.ss`` file must lie in (a case copy): a ``PotFile`` that
        resolves outside it, e.g. into a shared template, is rejected before
        anything is fitted or written. ``None`` (the default) does not check.

    Returns
    -------
    RadiationFitRecord or None
        The fit's provenance, or ``None`` when nothing needed to be generated.

    Raises
    ------
    RadiationFitError
        If the deck uses more than one WAMIT body, the ``.1`` file is missing or
        cannot be fitted, or the ``.ss`` file would lie outside ``confine_to``.
    """
    located = _wamit_radiation_paths(fst_path)
    if located is None:
        return None
    deck, hydro, target, source = located
    if confine_to is not None and not resolve_path(target).is_relative_to(
        resolve_path(confine_to)
    ):
        msg = (
            f"the state-space radiation file {target} would be written outside "
            f"{confine_to}; the PotFile of a working copy must lie inside it"
        )
        raise RadiationFitError(msg)
    if target.is_file() and not overwrite:
        return None
    model, record = _fit_record(deck, hydro, target, source, tolerance)
    accepted = record.accepted
    if accepted:
        write_radiation_state_space(
            model,
            target,
            header=(
                f"State-space radiation model fitted by VANE to {source.name} "
                f"(vector fitting, {model.n_states} states, max relative error "
                f"{model.max_relative_error:.3f})"
            ),
        )
        logger.warning(
            "generated the missing state-space radiation file %s from %s (%d states, "
            "max relative fit error %.3f); wave-excitation fits are not generated",
            target,
            source.name,
            model.n_states,
            model.max_relative_error,
        )
    else:
        logger.warning(
            "the state-space radiation fit to %s is not accurate enough (max relative "
            "error %.3f > %.2f); %s was not written, so linearize with the radiation "
            "memory off",
            source,
            model.max_relative_error,
            _MAX_ACCEPTED_ERROR,
            target.name,
        )
    return replace(record, written=accepted)
