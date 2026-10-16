# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Scale-invariant left--right modal participation.

Raw right-eigenvector amplitudes depend on the units chosen for the states.  A metre,
a radian, and an internal aerodynamic or hydrodynamic state therefore cannot be
compared by magnitude alone.  For a simple eigenvalue, the element-wise left--right
product

.. math::

   p_{ki}=\frac{\overline{w}_{ki}v_{ki}}
                 {\mathbf{w}_i^H\mathbf{v}_i}

is invariant to any nonsingular diagonal rescaling of the states.  Its complex
components sum to one.  This module reports those factors and a non-negative share
formed by normalising their magnitudes, both per state and aggregated into physical
subsystems.

Individual factors are not physically unique for a repeated eigenvalue because the
eigenvectors may rotate within its invariant subspace.  The caller should therefore
use :attr:`~vane.modal.eigensolver.ModalSolution.is_degenerate` as an acceptance check
before interpreting an individual degenerate mode.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from vane.modal.eigensolver import ModalSolution

__all__ = [
    "BiorthogonalParticipation",
    "biorthogonal_participation",
    "classify_state_subsystem",
]

_STANDARD_SUBSYSTEM_ORDER = (
    "platform",
    "tower",
    "blades",
    "drivetrain",
    "aerodynamics",
    "hydrodynamics",
    "mooring",
    "controller",
    "inflow",
    "substructure",
    "other",
)


@dataclass(frozen=True)
class BiorthogonalParticipation:
    """Left--right participation factors for every retained oscillatory mode.

    Parameters
    ----------
    complex_factors : npt.NDArray[np.complex128]
        Complex state factors, shape ``(n_states, n_modes)``.  Each valid column
        sums to one.
    magnitude_fractions : npt.NDArray[np.float64]
        Absolute factors normalised to sum to one per valid mode.
    closure_error : npt.NDArray[np.float64]
        Absolute difference between each complex column sum and one.
    valid : npt.NDArray[np.bool_]
        Whether the left--right inner product is numerically resolvable.  Invalid
        columns contain ``NaN`` and must not be interpreted.
    subsystem_names : tuple[str, ...]
        Ordered subsystem labels indexing ``subsystem_fractions``.
    subsystem_fractions : npt.NDArray[np.float64]
        Magnitude fractions summed by subsystem, shape
        ``(n_subsystems, n_modes)``.
    state_subsystems : tuple[str, ...]
        Subsystem assigned to each state.
    """

    complex_factors: npt.NDArray[np.complex128]
    magnitude_fractions: npt.NDArray[np.float64]
    closure_error: npt.NDArray[np.float64]
    valid: npt.NDArray[np.bool_]
    subsystem_names: tuple[str, ...]
    subsystem_fractions: npt.NDArray[np.float64]
    state_subsystems: tuple[str, ...]


def biorthogonal_participation(
    solution: ModalSolution,
    *,
    subsystem_labels: Sequence[str] | None = None,
) -> BiorthogonalParticipation:
    """Compute scale-invariant participation from a modal solution.

    Parameters
    ----------
    solution : ModalSolution
        Modal solution retaining matching full left and right eigenvectors.
    subsystem_labels : sequence of str or None, optional
        Explicit subsystem label for every state.  When omitted, labels are
        inferred from ``solution.state_descriptions``.

    Returns
    -------
    BiorthogonalParticipation
        State and subsystem participation factors.

    Raises
    ------
    ValueError
        If left eigenvectors are unavailable, shapes disagree, or supplied labels
        do not match the state count.
    """
    right = np.asarray(solution.full_eigenvectors, dtype=np.complex128)
    left = np.asarray(solution.full_left_eigenvectors, dtype=np.complex128)
    if left.size == 0:
        msg = (
            "ModalSolution has no left eigenvectors; recompute it with the current "
            "VANE eigensolver before requesting biorthogonal participation"
        )
        raise ValueError(msg)
    if left.shape != right.shape:
        msg = (
            f"left/right eigenvector shape mismatch: {left.shape} versus {right.shape}"
        )
        raise ValueError(msg)

    n_states, n_modes = right.shape
    labels = _state_subsystems(solution, subsystem_labels, n_states)
    denominator = np.sum(np.conj(left) * right, axis=0)
    scale = np.linalg.norm(left, axis=0) * np.linalg.norm(right, axis=0)
    tolerance = 100.0 * np.finfo(np.float64).eps * np.maximum(scale, 1.0)
    valid = np.abs(denominator) > tolerance

    factors = np.full((n_states, n_modes), np.nan + 1j * np.nan, dtype=np.complex128)
    factors[:, valid] = (
        np.conj(left[:, valid]) * right[:, valid] / denominator[np.newaxis, valid]
    )
    closure = np.full(n_modes, np.inf, dtype=np.float64)
    if np.any(valid):
        closure[valid] = np.abs(np.sum(factors[:, valid], axis=0) - 1.0)

    magnitude = np.abs(factors)
    magnitude_sum = np.nansum(magnitude, axis=0)
    fractions = np.full((n_states, n_modes), np.nan, dtype=np.float64)
    usable = valid & (magnitude_sum > 0.0)
    fractions[:, usable] = magnitude[:, usable] / magnitude_sum[np.newaxis, usable]

    subsystem_names = _ordered_subsystems(labels)
    subsystem = np.full((len(subsystem_names), n_modes), np.nan, dtype=np.float64)
    for row, name in enumerate(subsystem_names):
        indices = np.asarray([label == name for label in labels], dtype=np.bool_)
        if np.any(usable):
            subsystem[row, usable] = np.sum(fractions[indices][:, usable], axis=0)

    return BiorthogonalParticipation(
        complex_factors=factors,
        magnitude_fractions=fractions,
        closure_error=closure,
        valid=np.asarray(valid, dtype=np.bool_),
        subsystem_names=subsystem_names,
        subsystem_fractions=subsystem,
        state_subsystems=labels,
    )


_BEAMDYN_PREFIX_RE = re.compile(r"(bd(_\d+)?\s|beamdyn)")
_DERIVATIVE_PREFIX = "first time derivative of "


def classify_state_subsystem(description: str) -> str:
    """Classify an OpenFAST state description into an engineering subsystem."""
    value = description.strip().lower()
    # A leading derivative marker (the state-derivative spelling) names the same
    # state, so it is classified by the module prefix that follows it.
    if value.startswith(_DERIVATIVE_PREFIX):
        value = value[len(_DERIVATIVE_PREFIX) :].lstrip()
    if value.startswith("ed "):
        if "platform" in value:
            return "platform"
        if "tower" in value:
            return "tower"
        if "blade" in value or "flap" in value or "edge" in value:
            return "blades"
        if any(token in value for token in ("generator", "drivetrain", "shaft")):
            return "drivetrain"
    # BeamDyn prefixes carry the blade instance, e.g. ``BD_1 finite element node 2``.
    if _BEAMDYN_PREFIX_RE.match(value):
        return "blades"
    if value.startswith(("ad ", "aerodyn", "bemt", "dbemt", "ua ")):
        return "aerodynamics"
    if value.startswith(("hd ", "hydrodyn", "ss ", "seastate")):
        return "hydrodynamics"
    if value.startswith(("md ", "moordyn", "map ")):
        return "mooring"
    if value.startswith(("srvd ", "servodyn", "controller", "discon")):
        return "controller"
    if value.startswith(("ifw ", "inflowwind")):
        return "inflow"
    if value.startswith(("sd ", "subdyn")):
        return "substructure"
    return "other"


def _state_subsystems(
    solution: ModalSolution,
    explicit: Sequence[str] | None,
    n_states: int,
) -> tuple[str, ...]:
    """Return validated explicit labels or labels inferred from descriptions."""
    if explicit is not None:
        if len(explicit) != n_states:
            msg = (
                f"subsystem_labels has {len(explicit)} entries but there are "
                f"{n_states} states"
            )
            raise ValueError(msg)
        labels = tuple(str(label).strip() for label in explicit)
        if any(not label for label in labels):
            msg = "subsystem_labels must not contain empty labels"
            raise ValueError(msg)
        return labels
    descriptions = solution.state_descriptions
    if len(descriptions) != n_states:
        msg = (
            "state descriptions are required to infer subsystems: "
            f"got {len(descriptions)} descriptions for {n_states} states"
        )
        raise ValueError(msg)
    return tuple(classify_state_subsystem(item) for item in descriptions)


def _ordered_subsystems(labels: Sequence[str]) -> tuple[str, ...]:
    """Return stable engineering order followed by any caller-defined labels."""
    present = set(labels)
    standard = [name for name in _STANDARD_SUBSYSTEM_ORDER if name in present]
    custom = sorted(present.difference(_STANDARD_SUBSYSTEM_ORDER))
    return tuple(standard + custom)
