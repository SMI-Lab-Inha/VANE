# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Clean linear-time-invariant state-space representation.

A :class:`StateSpace` holds the validated ``A``, ``B``, ``C``, ``D`` matrices of a
continuous- or discrete-time system together with the state, input, and output
names. The averaged matrices of an MBC transform are exported directly into this
form, and a zero-order-hold discretization is provided for use in a digital twin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import expm

if TYPE_CHECKING:
    import numpy.typing as npt

    from vane.io.mbc_transform import MBCResult

__all__ = ["StateSpace", "state_space_from_mbc", "static_gain"]


@dataclass
class StateSpace:
    """A linear time-invariant state-space system.

    Parameters
    ----------
    a : npt.NDArray[np.float64]
        State matrix, shape ``(n_states, n_states)``.
    b, c, d : npt.NDArray[np.float64] or None
        Input, output, and feedthrough matrices, shapes ``(n_states, n_inputs)``,
        ``(n_outputs, n_states)``, ``(n_outputs, n_inputs)``; any may be ``None``.
    state_names, input_names, output_names : list[str]
        Names of the states, inputs, and outputs; each is either empty or matches
        the corresponding dimension.
    dt : float or None
        Sample time; ``None`` for a continuous-time system, otherwise the discrete
        sample time in seconds.
    output_mbc_coordinates : list[str]
        Multi-blade coordinate of each output (``"collective"``/``"cosine"``/
        ``"sine"`` for a transformed blade-triplet output, else ``""``), aligned
        with ``output_names``. Empty when unknown. Carried from an MBC transform so
        a load map can tell a transformed multi-blade output apart from a literal
        per-blade channel.

    Raises
    ------
    ValueError
        If the matrix shapes or name lengths are inconsistent.
    """

    a: npt.NDArray[np.float64]
    b: npt.NDArray[np.float64] | None = None
    c: npt.NDArray[np.float64] | None = None
    d: npt.NDArray[np.float64] | None = None
    state_names: list[str] = field(default_factory=list)
    input_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)
    dt: float | None = None
    output_mbc_coordinates: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate matrix shapes, the sample time, and name lengths."""
        if self.a.ndim != 2 or self.a.shape[0] != self.a.shape[1]:
            msg = f"A must be square, got shape {self.a.shape}"
            raise ValueError(msg)
        if not np.all(np.isfinite(self.a)):
            msg = "A must be finite (no NaN or inf)"
            raise ValueError(msg)
        n = self.a.shape[0]
        if self.b is not None:
            self._require_matrix(self.b, "B")
            if self.b.shape[0] != n:
                msg = f"B must have {n} rows, got {self.b.shape}"
                raise ValueError(msg)
        if self.c is not None:
            self._require_matrix(self.c, "C")
            if self.c.shape[1] != n:
                msg = f"C must have {n} columns, got {self.c.shape}"
                raise ValueError(msg)
        self._validate_feedthrough()
        if self.dt is not None and not (math.isfinite(self.dt) and self.dt > 0.0):
            msg = f"Discrete sample time must be positive and finite, got {self.dt}"
            raise ValueError(msg)
        self._check_names(self.state_names, n, "state_names")
        self._check_names(self.input_names, self.n_inputs, "input_names")
        self._check_names(self.output_names, self.n_outputs, "output_names")
        self._check_names(
            self.output_mbc_coordinates, self.n_outputs, "output_mbc_coordinates"
        )

    def _validate_feedthrough(self) -> None:
        """Validate D against whichever input/output dimensions are known."""
        if self.d is None:
            return
        self._require_matrix(self.d, "D")
        if self.b is not None and self.d.shape[1] != self.b.shape[1]:
            msg = f"D must have {self.b.shape[1]} columns (inputs), got {self.d.shape}"
            raise ValueError(msg)
        if self.c is not None and self.d.shape[0] != self.c.shape[0]:
            msg = f"D must have {self.c.shape[0]} rows (outputs), got {self.d.shape}"
            raise ValueError(msg)

    @staticmethod
    def _require_matrix(matrix: npt.NDArray[np.float64], label: str) -> None:
        """Raise if ``matrix`` is not a two-dimensional, finite matrix."""
        if matrix.ndim != 2:
            msg = f"{label} must be a 2-D matrix, got a {matrix.ndim}-D array"
            raise ValueError(msg)
        if not np.all(np.isfinite(matrix)):
            msg = f"{label} must be finite (no NaN or inf)"
            raise ValueError(msg)

    @staticmethod
    def _check_names(names: list[str], expected: int, label: str) -> None:
        """Raise if a non-empty name list does not match the expected length."""
        if names and len(names) != expected:
            msg = f"{label} has {len(names)} entries, expected {expected}"
            raise ValueError(msg)

    @property
    def n_states(self) -> int:
        """Return the number of states."""
        return int(self.a.shape[0])

    @property
    def n_inputs(self) -> int:
        """Return the number of inputs (from B, else D, else 0).

        A feedthrough-only system (``B is None`` but ``D`` present) still has
        ``D.shape[1]`` inputs; the feedthrough columns define them.
        """
        if self.b is not None:
            return int(self.b.shape[1])
        if self.d is not None:
            return int(self.d.shape[1])
        return 0

    @property
    def n_outputs(self) -> int:
        """Return the number of outputs (from C, else D, else 0).

        A feedthrough-only system (``C is None`` but ``D`` present) still has
        ``D.shape[0]`` outputs; the feedthrough rows define them.
        """
        if self.c is not None:
            return int(self.c.shape[0])
        if self.d is not None:
            return int(self.d.shape[0])
        return 0

    @property
    def is_discrete(self) -> bool:
        """Return whether the system is discrete-time."""
        return self.dt is not None

    def is_stable(self) -> bool:
        """Return whether the system is asymptotically stable.

        Returns
        -------
        bool
            For a continuous system, whether every eigenvalue of ``A`` has
            negative real part; for a discrete system, whether every eigenvalue
            lies strictly inside the unit circle.
        """
        eigenvalues = np.linalg.eigvals(self.a)
        if self.is_discrete:
            return bool(np.all(np.abs(eigenvalues) < 1.0))
        return bool(np.all(eigenvalues.real < 0.0))

    def discretized(self, dt: float) -> StateSpace:
        """Return a zero-order-hold discretization at sample time ``dt``.

        Parameters
        ----------
        dt : float
            Sample time in seconds; must be positive.

        Returns
        -------
        StateSpace
            The discrete-time system; ``C`` and ``D`` are unchanged.

        Raises
        ------
        ValueError
            If ``dt`` is not a positive, finite number, or the system is already
            discrete.
        """
        if not (math.isfinite(dt) and dt > 0.0):
            msg = f"Sample time must be positive and finite, got {dt}"
            raise ValueError(msg)
        if self.is_discrete:
            msg = "System is already discrete"
            raise ValueError(msg)
        ad, bd = _zero_order_hold(self.a, self.b, dt)
        return StateSpace(
            a=ad,
            b=bd,
            c=self.c,
            d=self.d,
            state_names=list(self.state_names),
            input_names=list(self.input_names),
            output_names=list(self.output_names),
            dt=dt,
            output_mbc_coordinates=list(self.output_mbc_coordinates),
        )


def state_space_from_mbc(result: MBCResult) -> StateSpace:
    """Export the averaged MBC matrices as a continuous-time state-space system.

    Parameters
    ----------
    result : MBCResult
        An MBC transform result whose ``avg_a`` is present.

    Returns
    -------
    StateSpace
        The continuous-time system with state names from the result.

    Raises
    ------
    ValueError
        If ``result.avg_a`` is ``None``.
    """
    if result.avg_a is None:
        msg = "MBCResult has no averaged A matrix to export"
        raise ValueError(msg)
    return StateSpace(
        a=result.avg_a,
        b=result.avg_b,
        c=result.avg_c,
        d=result.avg_d,
        state_names=list(result.state_descriptions),
        input_names=list(result.input_descriptions),
        output_names=list(result.output_descriptions),
        output_mbc_coordinates=list(result.output_mbc_coordinates),
    )


def static_gain(system: StateSpace) -> npt.NDArray[np.float64]:
    r"""Return the zero-frequency input-output gain ``D - C A^-1 B``.

    This quantity is especially important for module-level OpenFAST models: a very
    large direct stiffness can be cancelled by the equilibrium response of internal
    states.  Preserving ``D`` alone during reduction then gives the wrong static
    physics even when the retained modes are otherwise accurate.

    Raises
    ------
    ValueError
        If the system is discrete, lacks ``B`` or ``C``, or ``A`` is singular.
    """
    if system.is_discrete:
        msg = "static_gain requires a continuous-time system"
        raise ValueError(msg)
    if system.b is None or system.c is None:
        msg = "static_gain requires both B and C matrices"
        raise ValueError(msg)
    direct = (
        np.zeros((system.n_outputs, system.n_inputs), dtype=np.float64)
        if system.d is None
        else system.d
    )
    try:
        equilibrium = np.linalg.solve(system.a, system.b)
    except np.linalg.LinAlgError as exc:
        msg = "static_gain is undefined because A is singular"
        raise ValueError(msg) from exc
    return np.asarray(direct - system.c @ equilibrium, dtype=np.float64)


def _zero_order_hold(
    a: npt.NDArray[np.float64], b: npt.NDArray[np.float64] | None, dt: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64] | None]:
    """Zero-order-hold discretization of ``(A, B)`` via the augmented exponential."""
    n = a.shape[0]
    if b is None:
        ad: npt.NDArray[np.float64] = np.asarray(expm(a * dt), dtype=np.float64)
        return ad, None
    m = b.shape[1]
    augmented = np.zeros((n + m, n + m), dtype=np.float64)
    augmented[:n, :n] = a
    augmented[:n, n:] = b
    exponential = np.asarray(expm(augmented * dt), dtype=np.float64)
    return exponential[:n, :n], exponential[:n, n:]
