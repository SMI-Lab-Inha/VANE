# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Reference geometry of a wind-turbine system for mode-shape visualisation.

A linearization file carries *modal* degrees of freedom — generalized coordinates
such as platform surge, tower fore-aft bending, or a blade flap mode — not nodal
positions. To draw a mode on the physical structure, a reference (undeformed)
geometry is required: where the tower, nacelle, rotor, floating platform, and
mooring lines sit in space. This module provides that geometry as a small,
parametric, fully auditable model.

The geometry is a *schematic* assembled from a turbine's principal dimensions
(tower height, blade length, hub overhang, platform draft and radius, mooring
fairlead and anchor positions). It is deliberately not a high-fidelity hull mesh:
every node position follows from the documented dimensions, so the drawing can be
audited line by line. The deformation that animates it (see
:mod:`vane.viz.modeshape_fowt`) is driven by the *actual* complex mode shape, so
the relative motion across the structure is physically meaningful even though the
hull outline is stylised.

Coordinate convention
----------------------
A right-handed inertial frame with origin at the intersection of the undeflected
tower centreline and the still-water level (SWL), matching OpenFAST's platform
reference point: ``+x`` downwind, ``+y`` to the left looking downwind, ``+z`` up.
Onshore systems place the origin at the tower base instead; the convention is
otherwise identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "FowtGeometry",
    "Member",
    "SystemType",
    "default_geometry",
    "jacket_geometry",
    "monopile_geometry",
    "onshore_geometry",
    "semisub_geometry",
    "spar_geometry",
    "tlp_geometry",
]

_DEG = np.pi / 180.0


class SystemType(Enum):
    """Support structure of a wind-turbine system (string-valued, 3.10 safe)."""

    FIXED = "fixed"
    SPAR = "spar"
    SEMI = "semi"
    TLP = "tlp"
    LAND = "land"
    MONOPILE = "monopile"
    JACKET = "jacket"


@dataclass(frozen=True)
class Member:
    r"""One polyline component of the reference structure.

    A member is an ordered list of node positions forming a polyline (the tower
    centreline, one blade, a pontoon, a mooring line, ...). The optional fields
    tell the deformation engine how the member responds to the modal degrees of
    freedom; members that only ride the platform rigid-body motion (a pontoon, a
    column) need none of them.

    Parameters
    ----------
    name : str
        Human-readable label, shown in the figure legend.
    kind : str
        Component class, one of ``"tower"``, ``"blade"``, ``"platform"``,
        ``"mooring"``, ``"nacelle"``, or ``"hub"``. Governs the deformation rule.
    nodes : npt.NDArray[np.float64]
        Undeformed node positions, shape ``(n_nodes, 3)``.
    span_fraction : npt.NDArray[np.float64] or None
        For ``"tower"`` and ``"blade"`` members, the normalized arc length
        :math:`\xi \in [0, 1]` of each node from the clamped end (tower base /
        blade root), shape ``(n_nodes,)``. The bending shape functions are
        evaluated at these values.
    blade_index : int or None
        For ``"blade"`` members, the 1-based blade number; its reference azimuth
        is ``2*pi*(blade_index - 1)/n_blades``, measured as a right-hand rotation
        about the downwind shaft axis from vertical (the OpenFAST rotor azimuth
        used by the MBC3 transform), so blade 2 of a three-bladed rotor points to
        ``-y``. ``None`` otherwise.
    flap_dir, edge_dir : npt.NDArray[np.float64] or None
        For ``"blade"`` members, the unit flapwise (out-of-rotor-plane) and
        edgewise (in-rotor-plane) directions in the inertial frame, shape
        ``(3,)``. Blade bending displaces nodes along these. They follow the
        ElastoDyn blade-frame signs: positive flap points nominally downwind and
        positive edge points against the direction of rotation (toward the
        trailing edge).
    rides_tower_top : bool
        Whether the member is rigidly attached to the tower top (nacelle, hub,
        and blades) and therefore inherits the tower-top translation and bending
        rotation in addition to the platform rigid-body motion.
    """

    name: str
    kind: str
    nodes: npt.NDArray[np.float64]
    span_fraction: npt.NDArray[np.float64] | None = None
    blade_index: int | None = None
    flap_dir: npt.NDArray[np.float64] | None = None
    edge_dir: npt.NDArray[np.float64] | None = None
    rides_tower_top: bool = False


@dataclass(frozen=True)
class FowtGeometry:
    r"""Reference geometry of a complete wind-turbine system.

    Parameters
    ----------
    system_type : SystemType
        The support-structure class this geometry represents.
    members : list[Member]
        All polyline components (tower, blades, nacelle, hub, platform members,
        mooring lines).
    reference_point : npt.NDArray[np.float64]
        The platform reference point about which the rigid-body platform degrees
        of freedom (surge/sway/heave/roll/pitch/yaw) act, shape ``(3,)``.
    tower_top : npt.NDArray[np.float64]
        Position of the tower top (yaw bearing), shape ``(3,)``; the origin of the
        tower-top rotation that carries the rotor-nacelle assembly.
    tower_length : float
        Undeformed tower length, used to convert a bending slope into a tower-top
        rotation.
    n_blades : int
        Number of rotor blades.
    characteristic_length : float
        A representative length (typically the hub height) used to scale the
        animated deflection to a visible, dimensionless fraction.

    Notes
    -----
    The geometry is immutable; presets are produced by the builder functions in
    this module. ``members`` is stored as a tuple-backed list to keep the
    dataclass hashable-friendly while remaining ergonomic to iterate.
    """

    system_type: SystemType
    members: list[Member]
    reference_point: npt.NDArray[np.float64]
    tower_top: npt.NDArray[np.float64]
    tower_length: float
    n_blades: int
    characteristic_length: float
    metadata: dict[str, float] = field(default_factory=dict)

    def members_of_kind(self, kind: str) -> list[Member]:
        """Return the members whose ``kind`` equals ``kind``.

        Parameters
        ----------
        kind : str
            The component class to select.

        Returns
        -------
        list[Member]
            The matching members, in their stored order.
        """
        return [member for member in self.members if member.kind == kind]


def _line(
    start: npt.NDArray[np.float64], end: npt.NDArray[np.float64], n: int
) -> npt.NDArray[np.float64]:
    """Return ``n`` evenly spaced node positions from ``start`` to ``end``."""
    fractions = np.linspace(0.0, 1.0, n)[:, np.newaxis]
    return start[np.newaxis, :] * (1.0 - fractions) + end[np.newaxis, :] * fractions


def _rotor_frame(tilt_deg: float) -> npt.NDArray[np.float64]:
    """Return the rotor-frame basis rotated by a shaft tilt about ``+y``.

    The columns are the rotor-local axes expressed in the inertial frame: column 0
    is the shaft (axial / nominal flap) direction, columns 1 and 2 span the rotor
    plane. A positive tilt raises the downwind shaft end (the OpenFAST ``ShftTilt``
    sign), so an upwind rotor tilted nose-up takes a negative tilt.

    Parameters
    ----------
    tilt_deg : float
        Shaft tilt angle in degrees.

    Returns
    -------
    npt.NDArray[np.float64]
        The ``(3, 3)`` rotation matrix mapping rotor-local axes to the inertial
        frame.
    """
    tilt = tilt_deg * _DEG
    cos_t, sin_t = np.cos(tilt), np.sin(tilt)
    # Rotation about +y by -tilt, so the axial x basis (column 0) maps to
    # (cos, 0, +sin): a positive tilt raises the downwind shaft end, as documented.
    return np.array(
        [
            [cos_t, 0.0, -sin_t],
            [0.0, 1.0, 0.0],
            [sin_t, 0.0, cos_t],
        ],
        dtype=np.float64,
    )


def _build_blades(
    hub_center: npt.NDArray[np.float64],
    *,
    n_blades: int,
    hub_radius: float,
    blade_length: float,
    tilt_deg: float,
    precone_deg: float,
    n_nodes: int,
) -> list[Member]:
    r"""Construct the blade members radiating from the hub.

    Each blade is a straight radial polyline from the hub radius to the tip at its
    reference azimuth, tilted out of the rotor plane by the precone and carried by
    the shaft-tilted rotor frame. The flapwise (out-of-plane) and edgewise
    (in-plane) unit directions are stored so the deformation engine can displace
    the nodes along them.

    Parameters
    ----------
    hub_center : npt.NDArray[np.float64]
        Hub centre position, shape ``(3,)``.
    n_blades : int
        Number of blades.
    hub_radius : float
        Radial offset of the blade root from the hub centre.
    blade_length : float
        Length of each blade from root to tip.
    tilt_deg : float
        Shaft tilt in degrees, OpenFAST ``ShftTilt`` sign: positive raises the
        downwind shaft end, so an upwind rotor tilted nose-up has a negative tilt.
    precone_deg : float
        Blade precone in degrees, OpenFAST ``PreCone`` sign: positive cones the
        blades downwind, negative cones them upwind (away from the tower of an
        upwind rotor).
    n_nodes : int
        Number of nodes per blade.

    Returns
    -------
    list[Member]
        One :class:`Member` per blade.
    """
    rotor = _rotor_frame(tilt_deg)
    axial, plane_y, plane_z = rotor[:, 0], rotor[:, 1], rotor[:, 2]
    precone = precone_deg * _DEG
    cos_c, sin_c = np.cos(precone), np.sin(precone)

    members: list[Member] = []
    radius = hub_radius + np.linspace(0.0, blade_length, n_nodes)
    span = np.linspace(0.0, 1.0, n_nodes)
    for blade in range(1, n_blades + 1):
        azimuth = 2.0 * np.pi * (blade - 1) / n_blades
        cos_a, sin_a = np.cos(azimuth), np.sin(azimuth)
        # In-plane radial unit vector at this azimuth. The rotor turns positively
        # about the downwind shaft axis, so azimuth 0 points up (+z) and a quarter
        # turn later the blade points to -y (right, looking downwind).
        radial_plane = cos_a * plane_z - sin_a * plane_y
        # Precone tilts the radial direction along the downwind shaft axis; a
        # negative precone cones the blade upwind.
        radial = cos_c * radial_plane + sin_c * axial
        # Edgewise: in-plane, perpendicular to the blade, pointing against the
        # direction of rotation (toward the trailing edge), edge = radial x axial.
        edge = np.cross(radial, axial)
        edge /= np.linalg.norm(edge)
        # Flapwise: out of plane, completing the right-handed (flap, edge, radial)
        # triad, so it points nominally downwind.
        flap = np.cross(edge, radial)
        flap /= np.linalg.norm(flap)
        nodes = (
            hub_center[np.newaxis, :] + radius[:, np.newaxis] * radial[np.newaxis, :]
        )
        members.append(
            Member(
                name=f"Blade {blade}",
                kind="blade",
                nodes=nodes.astype(np.float64),
                span_fraction=span.astype(np.float64),
                blade_index=blade,
                flap_dir=flap.astype(np.float64),
                edge_dir=edge.astype(np.float64),
                rides_tower_top=True,
            )
        )
    return members


def _nacelle_and_hub(
    tower_top: npt.NDArray[np.float64],
    hub_center: npt.NDArray[np.float64],
) -> list[Member]:
    """Return the nacelle/shaft line from the tower top to the hub as one member."""
    nacelle = Member(
        name="Nacelle",
        kind="nacelle",
        nodes=_line(tower_top, hub_center, 2),
        rides_tower_top=True,
    )
    return [nacelle]


def _rotor_nacelle_tower(
    *,
    tower_base_z: float,
    tower_top_z: float,
    hub_height: float,
    overhang: float,
    hub_radius: float,
    blade_length: float,
    tilt_deg: float,
    precone_deg: float,
    n_blades: int,
    n_tower: int,
    n_blade_nodes: int,
) -> tuple[list[Member], npt.NDArray[np.float64], float]:
    """Build the shared tower + nacelle + rotor members above the support structure.

    Returns
    -------
    tuple
        ``(members, tower_top, tower_length)`` — the structural members, the
        tower-top position, and the undeformed tower length.
    """
    tower_base = np.array([0.0, 0.0, tower_base_z], dtype=np.float64)
    tower_top = np.array([0.0, 0.0, tower_top_z], dtype=np.float64)
    tower_length = float(tower_top_z - tower_base_z)
    span = np.linspace(0.0, 1.0, n_tower)
    tower = Member(
        name="Tower",
        kind="tower",
        nodes=_line(tower_base, tower_top, n_tower),
        span_fraction=span.astype(np.float64),
    )
    # Hub sits a shaft overhang upwind of the tower top, at hub height.
    hub_center = np.array([-overhang, 0.0, hub_height], dtype=np.float64)
    members: list[Member] = [tower]
    members += _nacelle_and_hub(tower_top, hub_center)
    members += _build_blades(
        hub_center,
        n_blades=n_blades,
        hub_radius=hub_radius,
        blade_length=blade_length,
        tilt_deg=tilt_deg,
        precone_deg=precone_deg,
        n_nodes=n_blade_nodes,
    )
    return members, tower_top, tower_length


def _ring(radius: float, z: float, n: int = 37) -> npt.NDArray[np.float64]:
    """Return a closed horizontal ring polyline of ``radius`` at height ``z``."""
    angle = np.linspace(0.0, 2.0 * np.pi, n)
    return np.column_stack(
        [radius * np.cos(angle), radius * np.sin(angle), np.full(n, z)]
    ).astype(np.float64)


def _vertical_column(
    x: float, y: float, z_bottom: float, z_top: float, n: int = 6
) -> npt.NDArray[np.float64]:
    """Return a vertical column polyline between two heights at ``(x, y)``."""
    return _line(
        np.array([x, y, z_bottom], dtype=np.float64),
        np.array([x, y, z_top], dtype=np.float64),
        n,
    )


def _catenary_line(
    fairlead: npt.NDArray[np.float64],
    anchor: npt.NDArray[np.float64],
    *,
    sag: float,
    n: int = 16,
) -> npt.NDArray[np.float64]:
    r"""Return a sagging mooring-line polyline from ``fairlead`` to ``anchor``.

    The horizontal path is the straight chord; the vertical profile adds a
    parabolic sag (zero at both ends, ``sag`` at mid-span) to evoke a slack
    catenary. This is a drawing aid for the undeformed line, not a tension
    solution.

    Parameters
    ----------
    fairlead, anchor : npt.NDArray[np.float64]
        The line endpoints (platform attachment and seabed anchor), shape ``(3,)``.
    sag : float
        Maximum downward sag at mid-span.
    n : int, optional
        Number of nodes along the line.

    Returns
    -------
    npt.NDArray[np.float64]
        The mooring-line node positions, shape ``(n, 3)``.
    """
    fraction = np.linspace(0.0, 1.0, n)
    line = (
        fairlead[np.newaxis, :] * (1.0 - fraction[:, np.newaxis])
        + anchor[np.newaxis, :] * fraction[:, np.newaxis]
    )
    line[:, 2] -= sag * 4.0 * fraction * (1.0 - fraction)
    return line.astype(np.float64)


def _mooring_lines(
    *,
    fairlead_radius: float,
    fairlead_z: float,
    anchor_radius: float,
    anchor_z: float,
    n_lines: int,
    sag: float,
    phase_deg: float = 0.0,
) -> list[Member]:
    """Return ``n_lines`` evenly spaced catenary mooring members.

    The fairlead end is stored first (node 0) so the deformation engine can fade
    the platform motion from the moving fairlead to the fixed anchor.
    """
    members: list[Member] = []
    for line in range(n_lines):
        angle = phase_deg * _DEG + 2.0 * np.pi * line / n_lines
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        fairlead = np.array(
            [fairlead_radius * cos_a, fairlead_radius * sin_a, fairlead_z],
            dtype=np.float64,
        )
        anchor = np.array(
            [anchor_radius * cos_a, anchor_radius * sin_a, anchor_z], dtype=np.float64
        )
        members.append(
            Member(
                name=f"Mooring {line + 1}",
                kind="mooring",
                nodes=_catenary_line(fairlead, anchor, sag=sag),
            )
        )
    return members


def semisub_geometry(
    *,
    hub_height: float = 90.0,
    tower_base_z: float = 10.0,
    blade_length: float = 60.0,
    column_radius: float = 6.0,
    offset_radius: float = 28.0,
    draft: float = 20.0,
    freeboard: float = 10.0,
    fairlead_radius: float = 28.0,
    fairlead_depth: float = 14.0,
    anchor_radius: float = 420.0,
    water_depth: float = 200.0,
    n_blades: int = 3,
    shaft_tilt_deg: float = -5.0,
    precone_deg: float = -2.5,
    overhang: float = 7.0,
) -> FowtGeometry:
    """Build a three-column semi-submersible FOWT reference geometry.

    A central column carries the tower; three outer columns on an offset radius are
    joined to it by pontoons and braces, and three catenary mooring lines run from
    the outer-column fairleads to seabed anchors. Defaults are representative of a
    5-MW-class semi-submersible and may be overridden per turbine.

    Parameters
    ----------
    hub_height : float, optional
        Hub height above SWL.
    tower_base_z : float, optional
        Tower-base height above SWL (top of the central column / freeboard).
    blade_length : float, optional
        Blade length from root to tip.
    column_radius : float, optional
        Radius of each buoyancy column.
    offset_radius : float, optional
        Radial distance of the outer columns from the centreline.
    draft : float, optional
        Column draft below SWL.
    freeboard : float, optional
        Column height above SWL.
    fairlead_radius : float, optional
        Radius of the mooring fairleads.
    fairlead_depth : float, optional
        Depth of the fairleads below SWL.
    anchor_radius : float, optional
        Radius of the seabed anchors.
    water_depth : float, optional
        Water depth (anchors sit on the seabed at ``-water_depth``).
    n_blades : int, optional
        Number of rotor blades.
    shaft_tilt_deg : float, optional
        Shaft tilt angle in degrees, with the OpenFAST ``ShftTilt`` sign (positive
        raises the downwind shaft end). The default ``-5`` tilts the upwind rotor
        nose-up, as for the NREL 5-MW turbine.
    precone_deg : float, optional
        Blade precone angle in degrees, with the OpenFAST ``PreCone`` sign
        (positive cones the blades downwind). The default ``-2.5`` cones the
        blades upwind, away from the tower.
    overhang : float, optional
        Upwind shaft overhang of the hub from the tower top.

    Returns
    -------
    FowtGeometry
        The assembled semi-submersible geometry.
    """
    tower_top_z = hub_height
    members, tower_top, tower_length = _rotor_nacelle_tower(
        tower_base_z=tower_base_z,
        tower_top_z=tower_top_z,
        hub_height=hub_height,
        overhang=overhang,
        hub_radius=0.05 * blade_length,
        blade_length=blade_length,
        tilt_deg=shaft_tilt_deg,
        precone_deg=precone_deg,
        n_blades=n_blades,
        n_tower=9,
        n_blade_nodes=11,
    )

    z_bottom, z_top = -draft, freeboard
    platform: list[Member] = [
        Member(
            name="Central column",
            kind="platform",
            nodes=_vertical_column(0.0, 0.0, z_bottom, tower_base_z),
        )
    ]
    for column in range(3):
        angle = 2.0 * np.pi * column / 3.0
        cx, cy = offset_radius * np.cos(angle), offset_radius * np.sin(angle)
        platform.append(
            Member(
                name=f"Offset column {column + 1}",
                kind="platform",
                nodes=_vertical_column(cx, cy, z_bottom, z_top),
            )
        )
        # Pontoon from the central column base to this offset column base.
        platform.append(
            Member(
                name=f"Pontoon {column + 1}",
                kind="platform",
                nodes=_line(
                    np.array([0.0, 0.0, z_bottom], dtype=np.float64),
                    np.array([cx, cy, z_bottom], dtype=np.float64),
                    4,
                ),
            )
        )
    # A waterline ring through the three offset columns ties the raft together.
    platform.append(
        Member(name="Deck ring", kind="platform", nodes=_ring(offset_radius, 0.0))
    )

    mooring = _mooring_lines(
        fairlead_radius=fairlead_radius,
        fairlead_z=-fairlead_depth,
        anchor_radius=anchor_radius,
        anchor_z=-water_depth,
        n_lines=3,
        sag=0.18 * water_depth,
    )
    return FowtGeometry(
        system_type=SystemType.SEMI,
        members=members + platform + mooring,
        reference_point=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        tower_top=tower_top,
        tower_length=tower_length,
        n_blades=n_blades,
        characteristic_length=hub_height,
        metadata={"water_depth": water_depth, "draft": draft},
    )


def spar_geometry(
    *,
    hub_height: float = 90.0,
    tower_base_z: float = 10.0,
    blade_length: float = 60.0,
    spar_radius: float = 4.7,
    draft: float = 120.0,
    fairlead_radius: float = 5.2,
    fairlead_depth: float = 70.0,
    anchor_radius: float = 850.0,
    water_depth: float = 320.0,
    n_blades: int = 3,
    shaft_tilt_deg: float = -5.0,
    precone_deg: float = -2.5,
    overhang: float = 7.0,
) -> FowtGeometry:
    """Build a single-column spar-buoy FOWT reference geometry (OC3-Hywind class).

    A deep slender cylinder provides buoyancy and ballast stability; three catenary
    mooring lines spread from fairleads near the waterline. Parameters mirror
    :func:`semisub_geometry`.

    Returns
    -------
    FowtGeometry
        The assembled spar geometry.
    """
    tower_top_z = hub_height
    members, tower_top, tower_length = _rotor_nacelle_tower(
        tower_base_z=tower_base_z,
        tower_top_z=tower_top_z,
        hub_height=hub_height,
        overhang=overhang,
        hub_radius=0.05 * blade_length,
        blade_length=blade_length,
        tilt_deg=shaft_tilt_deg,
        precone_deg=precone_deg,
        n_blades=n_blades,
        n_tower=9,
        n_blade_nodes=11,
    )
    platform = [
        Member(
            name="Spar",
            kind="platform",
            nodes=_vertical_column(0.0, 0.0, -draft, tower_base_z, n=10),
        ),
        Member(name="Waterline", kind="platform", nodes=_ring(spar_radius, 0.0)),
        Member(name="Base", kind="platform", nodes=_ring(spar_radius * 1.6, -draft)),
    ]
    mooring = _mooring_lines(
        fairlead_radius=fairlead_radius,
        fairlead_z=-fairlead_depth,
        anchor_radius=anchor_radius,
        anchor_z=-water_depth,
        n_lines=3,
        sag=0.2 * water_depth,
    )
    return FowtGeometry(
        system_type=SystemType.SPAR,
        members=members + platform + mooring,
        reference_point=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        tower_top=tower_top,
        tower_length=tower_length,
        n_blades=n_blades,
        characteristic_length=hub_height,
        metadata={"water_depth": water_depth, "draft": draft},
    )


def monopile_geometry(
    *,
    hub_height: float = 90.0,
    tower_base_z: float = 20.0,
    blade_length: float = 60.0,
    monopile_radius: float = 3.0,
    water_depth: float = 20.0,
    embedment: float = 25.0,
    mudline_z: float = 0.0,
    n_blades: int = 3,
    shaft_tilt_deg: float = -5.0,
    precone_deg: float = -2.5,
    overhang: float = 7.0,
) -> FowtGeometry:
    """Build a bottom-fixed monopile reference geometry (no mooring lines).

    The monopile runs from below the mudline up to the tower base above the
    waterline. There are no mooring lines; the platform reference point sits at the
    mudline, so platform degrees of freedom (if any) act there.

    Parameters
    ----------
    mudline_z : float, optional
        Seabed elevation. The default ``0.0`` keeps the plain bottom-fixed
        convention (seabed at the reference plane, no water column). Offshore
        presets pass ``-water_depth`` so the still-water plane at ``z = 0`` sits
        above the seabed and the pile crosses the waterline.

    Returns
    -------
    FowtGeometry
        The assembled monopile geometry.
    """
    tower_top_z = hub_height
    members, tower_top, tower_length = _rotor_nacelle_tower(
        tower_base_z=tower_base_z,
        tower_top_z=tower_top_z,
        hub_height=hub_height,
        overhang=overhang,
        hub_radius=0.05 * blade_length,
        blade_length=blade_length,
        tilt_deg=shaft_tilt_deg,
        precone_deg=precone_deg,
        n_blades=n_blades,
        n_tower=9,
        n_blade_nodes=11,
    )
    platform = [
        Member(
            name="Monopile",
            kind="platform",
            nodes=_vertical_column(0.0, 0.0, mudline_z - embedment, tower_base_z, n=8),
        ),
        Member(
            name="Mudline",
            kind="platform",
            nodes=_ring(monopile_radius * 2.0, mudline_z),
        ),
    ]
    return FowtGeometry(
        system_type=SystemType.FIXED,
        members=members + platform,
        reference_point=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        tower_top=tower_top,
        tower_length=tower_length,
        n_blades=n_blades,
        characteristic_length=hub_height,
        metadata={"water_depth": water_depth},
    )


def onshore_geometry(
    *,
    hub_height: float = 90.0,
    blade_length: float = 60.0,
    foundation_radius: float = 8.0,
    n_blades: int = 3,
    shaft_tilt_deg: float = -5.0,
    precone_deg: float = -2.5,
    overhang: float = 7.0,
) -> FowtGeometry:
    """Build a land-based (onshore) reference geometry: tower + rotor on the ground.

    The tower rises from the ground plane (``z = 0``) to the hub; there is no
    substructure, water, or mooring. A foundation ring marks the base.

    Returns
    -------
    FowtGeometry
        The assembled onshore geometry.
    """
    members, tower_top, tower_length = _rotor_nacelle_tower(
        tower_base_z=0.0,
        tower_top_z=hub_height,
        hub_height=hub_height,
        overhang=overhang,
        hub_radius=0.05 * blade_length,
        blade_length=blade_length,
        tilt_deg=shaft_tilt_deg,
        precone_deg=precone_deg,
        n_blades=n_blades,
        n_tower=9,
        n_blade_nodes=11,
    )
    foundation = [
        Member(name="Foundation", kind="platform", nodes=_ring(foundation_radius, 0.0))
    ]
    return FowtGeometry(
        system_type=SystemType.LAND,
        members=members + foundation,
        reference_point=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        tower_top=tower_top,
        tower_length=tower_length,
        n_blades=n_blades,
        characteristic_length=hub_height,
        metadata={},
    )


def tlp_geometry(
    *,
    hub_height: float = 90.0,
    tower_base_z: float = 10.0,
    blade_length: float = 60.0,
    column_radius: float = 6.0,
    pontoon_radius: float = 20.0,
    draft: float = 20.0,
    freeboard: float = 10.0,
    water_depth: float = 200.0,
    n_tendons: int = 3,
    n_blades: int = 3,
    shaft_tilt_deg: float = -5.0,
    precone_deg: float = -2.5,
    overhang: float = 7.0,
) -> FowtGeometry:
    """Build a tension-leg-platform (TLP) reference geometry.

    A central buoyancy column carries the tower; spoke pontoons reach out to the
    tendon-attachment corners, and **vertical taut tendons** run straight down from each
    corner to a seabed anchor directly below it — the defining TLP feature, unlike a
    semi-submersible's spread catenary mooring.

    Returns
    -------
    FowtGeometry
        The assembled TLP geometry.
    """
    tower_top_z = hub_height
    members, tower_top, tower_length = _rotor_nacelle_tower(
        tower_base_z=tower_base_z,
        tower_top_z=tower_top_z,
        hub_height=hub_height,
        overhang=overhang,
        hub_radius=0.05 * blade_length,
        blade_length=blade_length,
        tilt_deg=shaft_tilt_deg,
        precone_deg=precone_deg,
        n_blades=n_blades,
        n_tower=9,
        n_blade_nodes=11,
    )
    z_bottom = -draft
    platform: list[Member] = [
        Member(
            name="Central column",
            kind="platform",
            nodes=_vertical_column(0.0, 0.0, z_bottom, freeboard),
        )
    ]
    tendons: list[Member] = []
    for corner in range(n_tendons):
        angle = 2.0 * np.pi * corner / n_tendons
        cx, cy = pontoon_radius * np.cos(angle), pontoon_radius * np.sin(angle)
        platform.append(
            Member(
                name=f"Pontoon {corner + 1}",
                kind="platform",
                nodes=_line(
                    np.array([0.0, 0.0, z_bottom], dtype=np.float64),
                    np.array([cx, cy, z_bottom], dtype=np.float64),
                    4,
                ),
            )
        )
        # A taut vertical tendon: fairlead at the corner base, anchor straight below.
        tendons.append(
            Member(
                name=f"Tendon {corner + 1}",
                kind="mooring",
                nodes=_line(
                    np.array([cx, cy, z_bottom], dtype=np.float64),
                    np.array([cx, cy, -water_depth], dtype=np.float64),
                    6,
                ),
            )
        )
    platform.append(
        Member(name="Deck ring", kind="platform", nodes=_ring(column_radius, freeboard))
    )
    return FowtGeometry(
        system_type=SystemType.TLP,
        members=members + platform + tendons,
        reference_point=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        tower_top=tower_top,
        tower_length=tower_length,
        n_blades=n_blades,
        characteristic_length=hub_height,
        metadata={"water_depth": water_depth, "draft": draft},
    )


def jacket_geometry(
    *,
    hub_height: float = 110.0,
    tower_base_z: float = 20.0,
    blade_length: float = 60.0,
    base_radius: float = 15.0,
    top_radius: float = 7.0,
    water_depth: float = 50.0,
    embedment: float = 0.0,
    n_legs: int = 4,
    n_bays: int = 4,
    n_blades: int = 3,
    shaft_tilt_deg: float = -5.0,
    precone_deg: float = -2.5,
    overhang: float = 7.0,
) -> FowtGeometry:
    """Build a bottom-fixed lattice-jacket reference geometry.

    ``n_legs`` battered legs run from the mudline up to the transition piece, tied by
    ``n_bays`` levels of X-braces; a transition-piece deck carries the tower. The legs
    taper from ``base_radius`` at the seabed to ``top_radius`` at the transition piece.

    Returns
    -------
    FowtGeometry
        The assembled jacket geometry.
    """
    members, tower_top, tower_length = _rotor_nacelle_tower(
        tower_base_z=tower_base_z,
        tower_top_z=hub_height,
        hub_height=hub_height,
        overhang=overhang,
        hub_radius=0.05 * blade_length,
        blade_length=blade_length,
        tilt_deg=shaft_tilt_deg,
        precone_deg=precone_deg,
        n_blades=n_blades,
        n_tower=9,
        n_blade_nodes=11,
    )
    # The seabed (mudline) is at -water_depth; the braced bays span from there up
    # to the transition piece. Any embedment drives only the pile toe *below* the
    # mudline, so the displayed seabed and the lowest bay stay at the true seabed.
    z_seabed, z_top = -water_depth, tower_base_z
    levels = np.linspace(z_seabed, z_top, n_bays + 1)
    radii = np.linspace(base_radius, top_radius, n_bays + 1)
    angles = [2.0 * np.pi * leg / n_legs + np.pi / n_legs for leg in range(n_legs)]

    def corner(level: int, leg: int) -> npt.NDArray[np.float64]:
        return np.array(
            [
                radii[level] * np.cos(angles[leg]),
                radii[level] * np.sin(angles[leg]),
                levels[level],
            ],
            dtype=np.float64,
        )

    def leg_nodes(leg: int) -> npt.NDArray[np.float64]:
        bays = [corner(level, leg) for level in range(n_bays + 1)]
        if embedment > 0.0:  # prepend the embedded pile toe below the mudline
            base = bays[0]
            toe = np.array([base[0], base[1], z_seabed - embedment], dtype=np.float64)
            return np.array([toe, *bays], dtype=np.float64)
        return np.array(bays, dtype=np.float64)

    structure: list[Member] = [
        Member(name=f"Leg {leg + 1}", kind="platform", nodes=leg_nodes(leg))
        for leg in range(n_legs)
    ]
    for bay in range(n_bays):
        for leg in range(n_legs):
            nxt = (leg + 1) % n_legs
            structure.append(
                Member(
                    name=f"Brace {bay + 1}-{leg + 1}a",
                    kind="platform",
                    nodes=_line(corner(bay, leg), corner(bay + 1, nxt), 2),
                )
            )
            structure.append(
                Member(
                    name=f"Brace {bay + 1}-{leg + 1}b",
                    kind="platform",
                    nodes=_line(corner(bay, nxt), corner(bay + 1, leg), 2),
                )
            )
    structure.append(
        Member(name="Mudline", kind="platform", nodes=_ring(base_radius, z_seabed))
    )
    structure.append(
        Member(name="Transition piece", kind="platform", nodes=_ring(top_radius, z_top))
    )
    return FowtGeometry(
        system_type=SystemType.JACKET,
        members=members + structure,
        reference_point=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        tower_top=tower_top,
        tower_length=tower_length,
        n_blades=n_blades,
        characteristic_length=hub_height,
        metadata={"water_depth": water_depth},
    )


def _monopile_offshore() -> FowtGeometry:
    """Return an offshore monopile: seabed at ``-water_depth``, pile crossing z=0.

    Unlike the plain bottom-fixed preset (mudline at the reference plane), the
    offshore type draws a still-water plane at ``z = 0``; the mudline therefore
    drops to ``-water_depth`` so the seabed sits below the water surface rather
    than coinciding with it.
    """
    water_depth = 30.0
    return replace(
        monopile_geometry(water_depth=water_depth, mudline_z=-water_depth),
        system_type=SystemType.MONOPILE,
    )


_BUILDERS: dict[SystemType, Callable[[], FowtGeometry]] = {
    SystemType.SEMI: semisub_geometry,
    SystemType.SPAR: spar_geometry,
    SystemType.TLP: tlp_geometry,
    SystemType.FIXED: monopile_geometry,
    SystemType.LAND: onshore_geometry,
    SystemType.MONOPILE: _monopile_offshore,
    SystemType.JACKET: jacket_geometry,
}


def default_geometry(system_type: SystemType) -> FowtGeometry:
    """Return a default reference geometry for a system type.

    Parameters
    ----------
    system_type : SystemType
        The support-structure class. ``TLP`` uses the tension-leg
        schematic of :func:`tlp_geometry`.

    Returns
    -------
    FowtGeometry
        A representative geometry built with the preset's default dimensions.
    """
    return _BUILDERS[system_type]()
