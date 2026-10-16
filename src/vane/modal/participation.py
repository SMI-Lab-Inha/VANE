# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Per-degree-of-freedom participation in each mode.

OpenFAST exposes no full-system mass matrix, so a classical mass-normalized modal
participation factor is unavailable. Instead, participation is quantified from the
right eigenvector: the magnitude of each state's component in a mode measures how
strongly that degree of freedom participates. Each mode is normalized so its
dominant component has magnitude one, and a sign is assigned from the phase of
each component relative to the dominant one (components more than 90 degrees out
of phase are negated), giving the signed contribution used for physical mode
identification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from vane.config.dof_map import DofCategory, classify_dof

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from vane.modal.eigensolver import ModalSolution

__all__ = [
    "ParticipationResult",
    "compute_participation",
    "participation_from_modes",
    "participation_scale_factors",
]

# Categories whose ElastoDyn state is a translational deflection (metres) tied to a
# characteristic length, so dividing by that length makes them comparable to the
# dimensionless rotational/generalized DOFs. Everything else (platform rigid-body DOFs,
# rotations such as blade pitch / drivetrain torsion / platform roll-pitch-yaw, and
# unrecognized states) is left unscaled at 1.0.
_BLADE_DEFLECTION = frozenset(
    {
        DofCategory.BLADE_FLAP_1,
        DofCategory.BLADE_FLAP_2,
        DofCategory.BLADE_EDGE_1,
    }
)
_TOWER_DEFLECTION = frozenset(
    {
        DofCategory.TOWER_FORE_AFT_1,
        DofCategory.TOWER_SIDE_SIDE_1,
        DofCategory.TOWER_FORE_AFT_2,
        DofCategory.TOWER_SIDE_SIDE_2,
    }
)
# BeamDyn nodal translations (metres) along the blade-frame axes. The nodal rotations
# ("RX"/"RY"/"RZ") are rotation parameters, in radians for small rotations, and stay
# unscaled.
_BEAMDYN_TRANSLATION = frozenset({"TX", "TY", "TZ"})


@dataclass
class ParticipationResult:
    """Per-state participation in each mode.

    Parameters
    ----------
    magnitude : npt.NDArray[np.float64]
        Normalized participation magnitude, shape ``(n_dof, n_modes)``; each
        column's maximum is 1.
    signed_magnitude : npt.NDArray[np.float64]
        ``magnitude`` with the sign of components more than 90 degrees out of
        phase with the dominant component flipped negative.
    phase_deg : npt.NDArray[np.float64]
        Phase of each component in degrees, shape ``(n_dof, n_modes)``.
    dominant_state : npt.NDArray[np.int_]
        Index of the largest-magnitude state in each mode, shape ``(n_modes,)``.
    """

    magnitude: npt.NDArray[np.float64]
    signed_magnitude: npt.NDArray[np.float64]
    phase_deg: npt.NDArray[np.float64]
    dominant_state: npt.NDArray[np.int_]


def compute_participation(
    mode_shapes: npt.NDArray[np.complex128],
    scale_factors: npt.NDArray[np.float64] | None = None,
) -> ParticipationResult:
    """Compute per-state participation from complex mode shapes.

    Parameters
    ----------
    mode_shapes : npt.NDArray[np.complex128]
        Complex mode-shape matrix, shape ``(n_dof, n_modes)``.
    scale_factors : npt.NDArray[np.float64] or None, optional
        Per-DOF multipliers applied before normalization (e.g. ``1/blade_length``
        for translational blade DOFs) so heterogeneous DOFs are comparable. When
        ``None``, no scaling is applied.

    Returns
    -------
    ParticipationResult
        Normalized magnitudes, signed magnitudes, phases, and dominant states.

    Raises
    ------
    ValueError
        If ``scale_factors`` length does not match the number of DOFs.
    """
    phi = mode_shapes
    if scale_factors is not None:
        if scale_factors.shape[0] != phi.shape[0]:
            msg = (
                f"scale_factors length {scale_factors.shape[0]} does not match "
                f"{phi.shape[0]} DOFs"
            )
            raise ValueError(msg)
        phi = phi * scale_factors[:, np.newaxis]

    magnitude = np.abs(phi)
    n_dof, n_modes = phi.shape
    if n_dof == 0:
        empty = np.zeros((0, n_modes), dtype=np.float64)
        return ParticipationResult(
            magnitude=empty,
            signed_magnitude=empty,
            phase_deg=empty,
            dominant_state=np.zeros(n_modes, dtype=np.int_),
        )
    column_max = magnitude.max(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.where(column_max > 0.0, magnitude / column_max, 0.0)

    phase = np.angle(phi, deg=True)
    dominant = np.argmax(magnitude, axis=0)
    dominant_phase = phase[dominant, np.arange(n_modes)]
    relative = np.mod(phase - dominant_phase[np.newaxis, :], 360.0)
    relative = np.where(relative > 180.0, relative - 360.0, relative)
    sign = np.where(np.abs(relative) > 90.0, -1.0, 1.0)

    return ParticipationResult(
        magnitude=np.asarray(normalized, dtype=np.float64),
        signed_magnitude=np.asarray(normalized * sign, dtype=np.float64),
        phase_deg=np.asarray(phase, dtype=np.float64),
        dominant_state=np.asarray(dominant, dtype=np.int_),
    )


def participation_from_modes(
    solution: ModalSolution,
    scale_factors: npt.NDArray[np.float64] | None = None,
) -> ParticipationResult:
    """Compute participation from a :class:`ModalSolution`'s mode shapes.

    Parameters
    ----------
    solution : ModalSolution
        Modal solution whose ``mode_shapes`` are analysed.
    scale_factors : npt.NDArray[np.float64] or None, optional
        Per-DOF multipliers, as in :func:`compute_participation`.

    Returns
    -------
    ParticipationResult
        The participation result for the solution's mode shapes.
    """
    return compute_participation(solution.mode_shapes, scale_factors)


def participation_scale_factors(
    descriptions: Sequence[str],
    *,
    blade_length: float,
    tower_length: float,
) -> npt.NDArray[np.float64]:
    """Derive per-DOF participation scale factors from the state descriptions.

    A turbine's states are a mix of units: blade and tower deflections are in metres
    while the other degrees of freedom are rotations (radians) or generalized
    coordinates. Comparing their raw eigenvector components over-weights whichever has
    the larger numeric units. Scaling each translational deflection by its
    characteristic length — blade flap/edge by the flexible blade length, tower
    fore-aft/side-side by the flexible tower length — turns it into a dimensionless
    tip-deflection ratio comparable to the rest. A BeamDyn blade's nodal translations
    (along the blade-frame ``X``, ``Y`` and ``Z`` axes, in metres) are scaled by the
    blade length too, so they become spanwise deflection ratios comparable with the
    nodal rotations, which are in radians. It builds the
    ``scale_factors`` array that :func:`compute_participation` accepts (replacing a
    hand-made one), by classifying each state with :func:`~vane.config.classify_dof`.

    Every other category — the platform rigid-body DOFs (surge/sway/heave and
    roll/pitch/yaw), blade pitch, drivetrain torsion, generator azimuth, furl/teeter,
    BeamDyn nodal rotations, and any unrecognized state — is left at ``1.0`` (no
    length scaling). Velocity states scale by the same factor as their position DOF,
    since they share its classification.

    Only the raw-magnitude participation (:func:`compute_participation`) depends on
    these factors. The physical labels of :func:`vane.modal.label_solution` use the
    biorthogonal participation, which is invariant to any rescaling of the states
    and needs none.

    Parameters
    ----------
    descriptions : Sequence[str]
        The state descriptions, one per mode-shape row and in the same order (e.g. from
        the linearization file / MBC result), as classified by
        :func:`~vane.config.classify_dof`.
    blade_length : float
        Flexible blade length in metres (e.g. ``TurbineGeometry.blade_length`` from
        :func:`~vane.io.read_elastodyn_geometry`, ``TipRad - HubRad``, which the
        ElastoDyn file defines for BeamDyn blades too). Must be finite and positive.
    tower_length : float
        Flexible tower length in metres (e.g. ``TurbineGeometry.tower_length``). Must be
        finite and positive.

    Returns
    -------
    npt.NDArray[np.float64]
        The per-DOF scale factors, shape ``(len(descriptions),)``: ``1/blade_length``
        for ElastoDyn blade flap/edge deflections and BeamDyn nodal translations,
        ``1/tower_length`` for tower deflections, ``1.0`` otherwise. Pass it straight
        to :func:`compute_participation` / :func:`participation_from_modes`.

    Raises
    ------
    ValueError
        If ``blade_length`` or ``tower_length`` is not finite and positive.
    """
    for name, value in (("blade_length", blade_length), ("tower_length", tower_length)):
        if not math.isfinite(value) or value <= 0.0:
            msg = f"{name} must be finite and positive, got {value}"
            raise ValueError(msg)

    factors = np.ones(len(descriptions), dtype=np.float64)
    for index, description in enumerate(descriptions):
        info = classify_dof(description)
        category = info.category
        if category in _BLADE_DEFLECTION or info.component in _BEAMDYN_TRANSLATION:
            factors[index] = 1.0 / blade_length
        elif category in _TOWER_DEFLECTION:
            factors[index] = 1.0 / tower_length
    return factors
