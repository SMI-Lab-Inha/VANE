# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Classification of OpenFAST degree-of-freedom descriptions.

OpenFAST embeds an internal degree-of-freedom (DOF) token in each ElastoDyn state
description (e.g. ``DOF_TFA1`` for the first tower fore-aft mode or ``DOF_BF(1,1)``
for the first flapwise mode of blade 1). These tokens are stable across versions and
are used here to map a state description onto a physical DOF category, the
originating module, the blade index for rotating DOFs, and whether the state is a
velocity. This classification underpins physics-aware mode labeling.

BeamDyn states carry no such token: they are finite-element nodal displacements and
rotations, e.g. ``BD_1 finite element node 2 (number of elements = 1; element order =
5) translational displacement in X, m`` and its velocity twin (``BD_1 First time
derivative of finite element node 2 ...``, also spelled with the derivative marker
before the ``BD_<n>`` prefix). ``BD_<n>`` names the blade, and the node index runs
from the root (node 1, a constraint that is never a state) to the tip.

In the linearization the nodal states are perturbations expressed in the BeamDyn
blade reference frame *frozen at the operating point*: the blade-root orientation
that ElastoDyn passes to BeamDyn, i.e. ElastoDyn's pitched blade coordinate system.
``Z`` runs along the pitch axis toward the tip, ``X`` is normal to the rotor plane at
zero pitch (the flapwise direction) and ``Y`` lies in the rotor plane at zero pitch
(the edgewise direction). Because the frame follows the blade pitch, ``X``/``Y`` stay
flapwise/edgewise at any pitch angle; only the local structural twist (largest at the
root) mixes them, so the split is the pitch-axis flap/edge split ElastoDyn uses for
its own tip deflections. A translation along ``X`` and a rotation about ``Y`` are
therefore :attr:`DofCategory.BLADE_FLAP`, a translation along ``Y`` and a rotation
about ``X`` :attr:`DofCategory.BLADE_EDGE`, a rotation about ``Z``
:attr:`DofCategory.BLADE_TORSION`, and a translation along ``Z``
:attr:`DofCategory.BLADE_AXIAL`.

The root boundary condition of the linearized beam is the root-motion input measured
from the same frozen frame, so a nodal state is the *total* nodal perturbation: it
includes the rigid motion the blade root carries from the rest of the structure (a
rotor-azimuth perturbation appears as a rigid rotation of every node), and the nodal
velocity states are total velocities as well. These categories therefore say which
blade *direction* a state measures, not which bending mode it belongs to: the mode
order (1st/2nd/3rd) is resolved from the mode shape along the span by
:func:`vane.modal.labeler.label_solution`, which also separates near-rigid blade
motion from elastic bending.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

__all__ = ["DofCategory", "DofInfo", "classify_dof"]


class DofCategory(Enum):
    """Physical category of a degree of freedom (string-valued, 3.10 compatible)."""

    PLATFORM_SURGE = "platform_surge"
    PLATFORM_SWAY = "platform_sway"
    PLATFORM_HEAVE = "platform_heave"
    PLATFORM_ROLL = "platform_roll"
    PLATFORM_PITCH = "platform_pitch"
    PLATFORM_YAW = "platform_yaw"
    TOWER_FORE_AFT_1 = "tower_fore_aft_1"
    TOWER_SIDE_SIDE_1 = "tower_side_side_1"
    TOWER_FORE_AFT_2 = "tower_fore_aft_2"
    TOWER_SIDE_SIDE_2 = "tower_side_side_2"
    NACELLE_YAW = "nacelle_yaw"
    GENERATOR_AZIMUTH = "generator_azimuth"
    DRIVETRAIN_TORSION = "drivetrain_torsion"
    ROTOR_FURL = "rotor_furl"
    TAIL_FURL = "tail_furl"
    TEETER = "teeter"
    BLADE_FLAP_1 = "blade_flap_1"
    BLADE_FLAP_2 = "blade_flap_2"
    BLADE_EDGE_1 = "blade_edge_1"
    BLADE_PITCH = "blade_pitch"
    UNKNOWN = "unknown"
    # BeamDyn finite-element categories, appended after ``UNKNOWN`` so every
    # pre-existing member keeps its position (and category-indexed features their
    # order). They name the direction a nodal state measures, not a mode number.
    BLADE_FLAP = "blade_flap"
    BLADE_EDGE = "blade_edge"
    BLADE_TORSION = "blade_torsion"
    BLADE_AXIAL = "blade_axial"


@dataclass(frozen=True)
class DofInfo:
    """Classification of a single state description.

    Parameters
    ----------
    category : DofCategory
        The physical DOF category (``UNKNOWN`` if not recognized).
    module : str
        The originating module prefix (e.g. ``"ED"`` or ``"BD_1"``), or an empty
        string if absent.
    blade : int or None
        The 1-based blade index for rotating blade DOFs, otherwise ``None``.
    is_velocity : bool
        Whether the state is the time derivative (velocity) of a DOF.
    node : int or None, optional
        For a BeamDyn nodal state, the 1-based finite-element node index (node 1 is
        the blade root, so state nodes start at 2); otherwise ``None``.
    component : str or None, optional
        For a BeamDyn nodal state, the measured component in the blade reference
        frame: ``"TX"``, ``"TY"``, ``"TZ"`` for a translation along ``X``, ``Y``,
        ``Z`` and ``"RX"``, ``"RY"``, ``"RZ"`` for a rotation about it; otherwise
        ``None``.
    """

    category: DofCategory
    module: str
    blade: int | None
    is_velocity: bool
    node: int | None = None
    component: str | None = None


_FIXED_TOKENS: dict[str, DofCategory] = {
    "DOF_Sg": DofCategory.PLATFORM_SURGE,
    "DOF_Sw": DofCategory.PLATFORM_SWAY,
    "DOF_Hv": DofCategory.PLATFORM_HEAVE,
    "DOF_R": DofCategory.PLATFORM_ROLL,
    "DOF_P": DofCategory.PLATFORM_PITCH,
    "DOF_Y": DofCategory.PLATFORM_YAW,
    "DOF_TFA1": DofCategory.TOWER_FORE_AFT_1,
    "DOF_TSS1": DofCategory.TOWER_SIDE_SIDE_1,
    "DOF_TFA2": DofCategory.TOWER_FORE_AFT_2,
    "DOF_TSS2": DofCategory.TOWER_SIDE_SIDE_2,
    "DOF_Yaw": DofCategory.NACELLE_YAW,
    "DOF_GeAz": DofCategory.GENERATOR_AZIMUTH,
    "DOF_DrTr": DofCategory.DRIVETRAIN_TORSION,
    "DOF_RFrl": DofCategory.ROTOR_FURL,
    "DOF_TFrl": DofCategory.TAIL_FURL,
    "DOF_Teet": DofCategory.TEETER,
}

# Blade DOF tokens: DOF_BF(blade, mode), DOF_BE(blade, mode), DOF_BP(blade).
_BLADE_FLAP_RE = re.compile(r"DOF_BF\((\d+),(\d+)\)")
_BLADE_EDGE_RE = re.compile(r"DOF_BE\((\d+),(\d+)\)")
_BLADE_PITCH_RE = re.compile(r"DOF_BP\((\d+)\)")
# A fixed token is ``DOF_`` followed by word characters, stopping before any '('.
_FIXED_TOKEN_RE = re.compile(r"DOF_(\w+)")
_MODULE_RE = re.compile(r"^([A-Za-z]+(?:_\d+)?)\s")
_DERIV_MARKER = "First time derivative of"
# BeamDyn nodal state: the ``BD`` / ``BD_<n>`` module prefix (``<n>`` is the blade)
# with the derivative marker before or after it, the finite-element node index, then
# a translational or rotational nodal quantity along/about X, Y, or Z.
_BEAMDYN_STATE_RE = re.compile(
    r"^\s*(?:First time derivative of\s+)?BD(?:_(\d+))?\s+"
    r"(?:First time derivative of\s+)?finite element node\s+(\d+)\b.*?"
    r"\b(translational|rotational)\s+(?:displacement|parameters?)\s+"
    r"(?:in|about|along)\s+([XYZ])\b",
    re.IGNORECASE,
)
_BEAMDYN_CATEGORY = {
    "TX": DofCategory.BLADE_FLAP,
    "RY": DofCategory.BLADE_FLAP,
    "TY": DofCategory.BLADE_EDGE,
    "RX": DofCategory.BLADE_EDGE,
    "RZ": DofCategory.BLADE_TORSION,
    "TZ": DofCategory.BLADE_AXIAL,
}

_FLAP_MODE = {1: DofCategory.BLADE_FLAP_1, 2: DofCategory.BLADE_FLAP_2}


def classify_dof(description: str) -> DofInfo:
    """Classify a state description into a :class:`DofInfo`.

    Parameters
    ----------
    description : str
        A state description from a linearization file (e.g.
        ``"ED 1st tower fore-aft ... (internal DOF index = DOF_TFA1), m"``).

    Returns
    -------
    DofInfo
        The classification; ``category`` is :attr:`DofCategory.UNKNOWN` when the
        description carries neither a known ElastoDyn ``DOF_`` token nor a
        recognized BeamDyn nodal quantity. For a BeamDyn state, ``module`` and
        ``blade`` come from its ``BD_<n>`` prefix (``blade`` is ``None`` for a bare
        ``BD`` prefix) and ``node``/``component`` locate the state on the blade.
    """
    module_match = _MODULE_RE.match(description)
    module = module_match.group(1) if module_match is not None else ""
    is_velocity = _DERIV_MARKER in description

    beamdyn = _BEAMDYN_STATE_RE.match(description)
    if beamdyn is not None:
        blade_text = beamdyn.group(1)
        component = beamdyn.group(3)[0].upper() + beamdyn.group(4).upper()
        return DofInfo(
            category=_BEAMDYN_CATEGORY[component],
            module="BD" if blade_text is None else f"BD_{blade_text}",
            blade=None if blade_text is None else int(blade_text),
            is_velocity=is_velocity,
            node=int(beamdyn.group(2)),
            component=component,
        )

    category, blade = _classify_token(description)
    return DofInfo(
        category=category, module=module, blade=blade, is_velocity=is_velocity
    )


def _classify_token(description: str) -> tuple[DofCategory, int | None]:
    """Return the DOF category and blade index encoded in ``description``."""
    flap = _BLADE_FLAP_RE.search(description)
    if flap is not None:
        blade, mode = int(flap.group(1)), int(flap.group(2))
        return _FLAP_MODE.get(mode, DofCategory.UNKNOWN), blade
    edge = _BLADE_EDGE_RE.search(description)
    if edge is not None:
        return DofCategory.BLADE_EDGE_1, int(edge.group(1))
    pitch = _BLADE_PITCH_RE.search(description)
    if pitch is not None:
        return DofCategory.BLADE_PITCH, int(pitch.group(1))

    fixed = _FIXED_TOKEN_RE.search(description)
    if fixed is not None:
        category = _FIXED_TOKENS.get(f"DOF_{fixed.group(1)}")
        if category is not None:
            return category, None
    return DofCategory.UNKNOWN, None
