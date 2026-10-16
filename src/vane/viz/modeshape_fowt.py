# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Physical full-system mode-shape visualization of a wind-turbine.

The other mode-shape views in :mod:`vane.viz` are deliberately abstract — phasor
wheels and participation bars — because an OpenFAST mode shape is expressed over
*modal* degrees of freedom, not spatial coordinates. This module closes the gap to
a picture an engineer can read at a glance: it draws the whole structure — floating
platform, mooring lines, tower, nacelle, and the three blades — and deforms it with
the actual complex mode shape, then animates it through one modal cycle.

How the modal coordinates become motion
----------------------------------------
The reference geometry (:mod:`vane.config.system_config`) supplies undeformed node
positions and, for the tower and blades, a normalized arc length per node. Each
modal degree of freedom is then mapped to a physical displacement field:

* **Platform** surge/sway/heave/roll/pitch/yaw move the *entire* structure as a
  rigid body about the platform reference point.
* **Tower** fore-aft and side-side bending modes add a lateral deflection along the
  tower via clamped-free (cantilever) bending shape functions; the resulting
  tower-top translation and slope carry the rotor-nacelle assembly.
* **Blade** flap and edge modes deflect each blade along its flapwise and edgewise
  directions. After the multi-blade-coordinate transform the blade modes are
  collective/cosine/sine; the inverse transform recovers the per-blade amplitude at
  each blade's azimuth.
* **Mooring** lines are dragged kinematically by the fairlead motion, fading to the
  fixed anchor.

Honesty
-------
The hull and mooring outlines are a *schematic* (see
:mod:`vane.config.system_config`); the bending shape functions are the standard
Euler-Bernoulli cantilever modes, not the turbine's exact polynomial shapes. What
is faithful is the *relative* motion across the structure, which is set by the
computed complex mode shape. The deflection magnitude is exaggerated by a scale
factor for visibility (universal practice for mode-shape plots) — it is not a
physical amplitude.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from matplotlib.ticker import MaxNLocator

from vane.config.dof_map import DofCategory, classify_dof
from vane.config.system_config import (
    FowtGeometry,
    Member,
    SystemType,
    default_geometry,
)
from vane.viz._style import (
    FAMILY_COLORS,
    STANDARD_COLORS,
    VANE_STYLE,
    ModeFamily,
    resolve_size,
    resolve_template,
    static_style,
)

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.figure import Figure

    from vane.modal.eigensolver import ModalSolution
    from vane.viz.publication import FigureSize

__all__ = [
    "FowtModeDeformation",
    "deform_mode",
    "infer_geometry",
    "plot_mode_fowt",
    "plot_mode_fowt_animation",
    "plot_mode_fowt_static",
]

# First two clamped-free (cantilever) Euler-Bernoulli eigenvalues beta*L.
_CANTILEVER_BETA = (1.8751040687119611, 4.694091132974175)

_PLATFORM_TRANSLATION = {
    DofCategory.PLATFORM_SURGE: 0,
    DofCategory.PLATFORM_SWAY: 1,
    DofCategory.PLATFORM_HEAVE: 2,
}
_PLATFORM_ROTATION = {
    DofCategory.PLATFORM_ROLL: 0,
    DofCategory.PLATFORM_PITCH: 1,
    DofCategory.PLATFORM_YAW: 2,
}
_TOWER_FORE_AFT = (DofCategory.TOWER_FORE_AFT_1, DofCategory.TOWER_FORE_AFT_2)
_TOWER_SIDE_SIDE = (DofCategory.TOWER_SIDE_SIDE_1, DofCategory.TOWER_SIDE_SIDE_2)
_BLADE_FLAP = (DofCategory.BLADE_FLAP_1, DofCategory.BLADE_FLAP_2)
_BLADE_EDGE = DofCategory.BLADE_EDGE_1

# Colour per member kind for the deformed structure; the undeformed ghost is grey.
# The structural members take the colours of their mode families in the Campbell
# diagram (tower black, blades green, platform red); hub, nacelle and mooring are
# neutral greys.
_KIND_COLORS = {
    "tower": FAMILY_COLORS[ModeFamily.TOWER],
    "nacelle": "#555555",
    "hub": "#555555",
    "blade": FAMILY_COLORS[ModeFamily.BLADE],
    "platform": FAMILY_COLORS[ModeFamily.PLATFORM],
    "mooring": "#7f7f7f",
}
_GHOST_COLOR = "#cfcfcf"
_WATER_COLOR = STANDARD_COLORS[6]


@dataclass
class FowtModeDeformation:
    r"""A mode's scaled complex displacement field over a reference geometry.

    Parameters
    ----------
    geometry : FowtGeometry
        The reference geometry that was deformed.
    displacements : list[npt.NDArray[np.complex128]]
        Per-member complex displacement fields, aligned with ``geometry.members``;
        element ``m`` has shape ``(n_nodes_m, 3)`` and is already multiplied by the
        visualization scale. The instantaneous real displacement at modal phase
        :math:`\theta` is ``(displacements[m] * exp(1j * theta)).real``.
    scale : float
        The length multiplier applied so the peak nodal deflection equals the
        requested fraction of the geometry's characteristic length.
    mode_index : int
        The mode that was visualized.
    frequency_hz : float
        Natural frequency of the mode, for labelling.
    damping_ratio : float
        Damping ratio of the mode, for labelling.
    """

    geometry: FowtGeometry
    displacements: list[npt.NDArray[np.complex128]]
    scale: float
    mode_index: int
    frequency_hz: float
    damping_ratio: float

    def peak_phase_rad(self) -> float:
        r"""Return the modal phase of maximum total deflection.

        A complex mode passes through different deformed configurations over a
        cycle; the instantaneous field at phase :math:`\theta` is
        :math:`\operatorname{Re}(d\,e^{i\theta})`. This returns the
        :math:`\theta` that maximizes the field's total squared norm — the most
        legible single snapshot, and the displacement extreme for a real (in-phase)
        mode.

        Returns
        -------
        float
            The peak-deflection phase in radians, in ``[-pi/2, pi/2]`` (a half-angle
            from the field's double-angle norm, so it is defined modulo ``pi``).
        """
        p = q = r = 0.0
        for vec in self.displacements:
            if not vec.size:
                continue
            p += float((vec.real**2).sum())
            q += float((vec.imag**2).sum())
            r += float((vec.real * vec.imag).sum())
        if p == 0.0 and q == 0.0:
            return 0.0
        return float(-0.5 * np.arctan2(r, 0.5 * (p - q)))

    def deformed_nodes(
        self, member_index: int, phase_rad: float
    ) -> npt.NDArray[np.float64]:
        r"""Return a member's node positions at a modal phase.

        Parameters
        ----------
        member_index : int
            Index into ``geometry.members``.
        phase_rad : float
            Modal phase :math:`\theta` in radians.

        Returns
        -------
        npt.NDArray[np.float64]
            The deformed node positions, shape ``(n_nodes, 3)``.
        """
        member = self.geometry.members[member_index]
        offset = (self.displacements[member_index] * np.exp(1j * phase_rad)).real
        return np.asarray(member.nodes + offset, dtype=np.float64)


@dataclass
class _Amplitudes:
    """Complex modal amplitudes of a single mode, grouped by physical role."""

    platform: dict[DofCategory, complex] = field(default_factory=dict)
    tower: dict[DofCategory, complex] = field(default_factory=dict)
    blade_mbc: dict[DofCategory, dict[str, complex]] = field(default_factory=dict)
    blade_indexed: dict[tuple[DofCategory, int], complex] = field(default_factory=dict)
    has_mbc: bool = False


def _cantilever_shape(
    mode: int, xi: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    r"""Return the clamped-free bending shape ``mode`` evaluated at ``xi``.

    The Euler-Bernoulli cantilever mode shape, normalized to unit deflection at the
    free tip (``xi = 1``):

    .. math::

        \psi_n(\xi) = \cosh(\beta_n \xi) - \cos(\beta_n \xi)
        - \sigma_n[\sinh(\beta_n \xi) - \sin(\beta_n \xi)],

    with :math:`\sigma_n = (\cosh\beta_n + \cos\beta_n)/(\sinh\beta_n + \sin\beta_n)`,
    the clamped-free (cantilever) constant that enforces a zero-moment, zero-shear
    free end.

    Parameters
    ----------
    mode : int
        Bending mode number, 1 or 2.
    xi : npt.NDArray[np.float64]
        Normalized arc length in ``[0, 1]``.

    Returns
    -------
    npt.NDArray[np.float64]
        The shape value at each ``xi``, normalized so the tip value is 1.
    """
    beta = _CANTILEVER_BETA[mode - 1]
    sigma = (np.cosh(beta) + np.cos(beta)) / (np.sinh(beta) + np.sin(beta))

    def raw(s: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.asarray(
            np.cosh(beta * s)
            - np.cos(beta * s)
            - sigma * (np.sinh(beta * s) - np.sin(beta * s)),
            dtype=np.float64,
        )

    tip = raw(np.array([1.0]))[0]
    return np.asarray(raw(xi) / tip, dtype=np.float64)


def _cantilever_tip_slope(mode: int) -> float:
    r"""Return the tip slope :math:`\mathrm{d}\psi_n/\mathrm{d}\xi` at ``xi = 1``.

    Parameters
    ----------
    mode : int
        Bending mode number, 1 or 2.

    Returns
    -------
    float
        The slope of the unit-tip-normalized shape at the free end.
    """
    beta = _CANTILEVER_BETA[mode - 1]
    sigma = (np.cosh(beta) + np.cos(beta)) / (np.sinh(beta) + np.sin(beta))
    tip = np.cosh(beta) - np.cos(beta) - sigma * (np.sinh(beta) - np.sin(beta))
    slope = beta * (np.sinh(beta) + np.sin(beta)) - sigma * beta * (
        np.cosh(beta) - np.cos(beta)
    )
    return float(slope / tip)


def _extract_amplitudes(solution: ModalSolution, mode_index: int) -> _Amplitudes:
    """Group a mode's complex DOF amplitudes by physical role.

    Parameters
    ----------
    solution : ModalSolution
        A solution carrying ``dof_descriptions`` (and ``dof_mbc_coordinates`` when
        the model has been MBC-transformed).
    mode_index : int
        The mode column to read.

    Returns
    -------
    _Amplitudes
        Platform, tower, and blade amplitudes ready for the deformation map.
    """
    shape = solution.mode_shapes[:, mode_index]
    coordinates = solution.dof_mbc_coordinates
    amplitudes = _Amplitudes(has_mbc=bool(coordinates))
    for row, description in enumerate(solution.dof_descriptions):
        info = classify_dof(description)
        if info.is_velocity or info.category is DofCategory.UNKNOWN:
            continue
        value = complex(shape[row])
        category = info.category
        if category in _PLATFORM_TRANSLATION or category in _PLATFORM_ROTATION:
            amplitudes.platform[category] = value
        elif category in _TOWER_FORE_AFT or category in _TOWER_SIDE_SIDE:
            amplitudes.tower[category] = value
        elif category in _BLADE_FLAP or category == _BLADE_EDGE:
            mbc = coordinates[row] if row < len(coordinates) else ""
            if mbc in ("collective", "cosine", "sine"):
                amplitudes.blade_mbc.setdefault(category, {})[mbc] = value
            elif info.blade is not None:
                amplitudes.blade_indexed[(category, info.blade)] = value
    return amplitudes


def _platform_motion(
    amplitudes: _Amplitudes,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """Return the rigid-body ``(translation, rotation)`` complex vectors."""
    translation = np.zeros(3, dtype=np.complex128)
    rotation = np.zeros(3, dtype=np.complex128)
    for category, axis in _PLATFORM_TRANSLATION.items():
        translation[axis] = amplitudes.platform.get(category, 0.0)
    for category, axis in _PLATFORM_ROTATION.items():
        rotation[axis] = amplitudes.platform.get(category, 0.0)
    return translation, rotation


def _rigid_body(
    nodes: npt.NDArray[np.float64],
    translation: npt.NDArray[np.complex128],
    rotation: npt.NDArray[np.complex128],
    reference: npt.NDArray[np.float64],
) -> npt.NDArray[np.complex128]:
    """Return the small-displacement rigid-body field at ``nodes``.

    Parameters
    ----------
    nodes : npt.NDArray[np.float64]
        Node positions, shape ``(n, 3)``.
    translation, rotation : npt.NDArray[np.complex128]
        Complex translation and small-angle rotation vectors, shape ``(3,)``.
    reference : npt.NDArray[np.float64]
        The point the rotation acts about, shape ``(3,)``.

    Returns
    -------
    npt.NDArray[np.complex128]
        The complex displacement at each node, shape ``(n, 3)``.
    """
    arm = nodes - reference[np.newaxis, :]
    spin = np.cross(rotation[np.newaxis, :], arm)
    return (translation[np.newaxis, :] + spin).astype(np.complex128)


def _tower_top_motion(
    amplitudes: _Amplitudes, tower_length: float
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """Return the tower-top bending ``(translation, rotation)`` complex vectors.

    The shapes are normalized to unit tip deflection, so the tip translation is the
    sum of the fore-aft (and side-side) modal amplitudes; the rotation follows from
    the bending slopes at the tip.
    """
    fa1 = amplitudes.tower.get(DofCategory.TOWER_FORE_AFT_1, 0.0)
    fa2 = amplitudes.tower.get(DofCategory.TOWER_FORE_AFT_2, 0.0)
    ss1 = amplitudes.tower.get(DofCategory.TOWER_SIDE_SIDE_1, 0.0)
    ss2 = amplitudes.tower.get(DofCategory.TOWER_SIDE_SIDE_2, 0.0)
    translation = np.array([fa1 + fa2, ss1 + ss2, 0.0], dtype=np.complex128)
    slope_scale = 1.0 / tower_length if tower_length > 0.0 else 0.0
    slope_x = (fa1 * _cantilever_tip_slope(1) + fa2 * _cantilever_tip_slope(2)) * (
        slope_scale
    )
    slope_y = (ss1 * _cantilever_tip_slope(1) + ss2 * _cantilever_tip_slope(2)) * (
        slope_scale
    )
    # A fore-aft slope rotates about +y; a side-side slope rotates about -x.
    rotation = np.array([-slope_y, slope_x, 0.0], dtype=np.complex128)
    return translation, rotation


def _tower_bending(
    member: Member, amplitudes: _Amplitudes
) -> npt.NDArray[np.complex128]:
    """Return the tower's flexible (bending) displacement along its span."""
    xi = member.span_fraction
    assert xi is not None  # tower members always carry span fractions
    fa1 = amplitudes.tower.get(DofCategory.TOWER_FORE_AFT_1, 0.0)
    fa2 = amplitudes.tower.get(DofCategory.TOWER_FORE_AFT_2, 0.0)
    ss1 = amplitudes.tower.get(DofCategory.TOWER_SIDE_SIDE_1, 0.0)
    ss2 = amplitudes.tower.get(DofCategory.TOWER_SIDE_SIDE_2, 0.0)
    phi1, phi2 = _cantilever_shape(1, xi), _cantilever_shape(2, xi)
    bending = np.zeros((xi.shape[0], 3), dtype=np.complex128)
    bending[:, 0] = fa1 * phi1 + fa2 * phi2
    bending[:, 1] = ss1 * phi1 + ss2 * phi2
    return bending


def _blade_amplitude(
    amplitudes: _Amplitudes, category: DofCategory, blade: int, n_blades: int
) -> complex:
    """Return a blade's complex amplitude for a DOF category via inverse MBC."""
    if amplitudes.has_mbc and category in amplitudes.blade_mbc:
        components = amplitudes.blade_mbc[category]
        azimuth = 2.0 * np.pi * (blade - 1) / n_blades
        return complex(
            components.get("collective", 0.0)
            + components.get("cosine", 0.0) * np.cos(azimuth)
            + components.get("sine", 0.0) * np.sin(azimuth)
        )
    return amplitudes.blade_indexed.get((category, blade), 0.0)


def _blade_bending(
    member: Member, amplitudes: _Amplitudes, n_blades: int
) -> npt.NDArray[np.complex128]:
    """Return a blade's flap+edge bending displacement along its span."""
    xi = member.span_fraction
    blade = member.blade_index
    flap_dir, edge_dir = member.flap_dir, member.edge_dir
    assert xi is not None and blade is not None
    assert flap_dir is not None and edge_dir is not None
    flap1 = _blade_amplitude(amplitudes, DofCategory.BLADE_FLAP_1, blade, n_blades)
    flap2 = _blade_amplitude(amplitudes, DofCategory.BLADE_FLAP_2, blade, n_blades)
    edge1 = _blade_amplitude(amplitudes, DofCategory.BLADE_EDGE_1, blade, n_blades)
    phi1, phi2 = _cantilever_shape(1, xi), _cantilever_shape(2, xi)
    flap = flap1 * phi1 + flap2 * phi2
    edge = edge1 * phi1
    return (
        flap[:, np.newaxis] * flap_dir[np.newaxis, :]
        + edge[:, np.newaxis] * edge_dir[np.newaxis, :]
    ).astype(np.complex128)


def _member_displacement(
    member: Member,
    amplitudes: _Amplitudes,
    geometry: FowtGeometry,
    rigid: tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]],
    top: tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]],
) -> npt.NDArray[np.complex128]:
    """Return the unscaled complex displacement field of one member."""
    translation, rotation = rigid
    top_translation, top_rotation = top
    nodes = member.nodes
    base = _rigid_body(nodes, translation, rotation, geometry.reference_point)

    if member.kind == "mooring":
        # The fairlead (node 0) rides the platform; fade to the fixed anchor.
        weight = np.linspace(1.0, 0.0, nodes.shape[0])[:, np.newaxis]
        fairlead = _rigid_body(
            nodes[:1], translation, rotation, geometry.reference_point
        )
        return (weight * fairlead).astype(np.complex128)

    if member.kind == "tower":
        return (base + _tower_bending(member, amplitudes)).astype(np.complex128)

    if member.rides_tower_top:
        arm = nodes - geometry.tower_top[np.newaxis, :]
        carry = top_translation[np.newaxis, :] + np.cross(
            top_rotation[np.newaxis, :], arm
        )
        displacement = base + carry
        if member.kind == "blade":
            displacement = displacement + _blade_bending(
                member, amplitudes, geometry.n_blades
            )
        return displacement.astype(np.complex128)

    return base


def deform_mode(
    solution: ModalSolution,
    mode_index: int,
    geometry: FowtGeometry | None = None,
    *,
    deflection_ratio: float = 0.12,
    scale: float = 1.0,
) -> FowtModeDeformation:
    """Map a mode shape onto a reference geometry as a complex displacement field.

    Parameters
    ----------
    solution : ModalSolution
        A solution whose ``dof_descriptions`` are populated.
    mode_index : int
        Index of the mode to visualize.
    geometry : FowtGeometry or None, optional
        The reference geometry. When ``None`` a schematic is inferred from the
        solution's degrees of freedom (see :func:`infer_geometry`).
    deflection_ratio : float, optional
        Target peak nodal deflection as a fraction of the geometry's characteristic
        length. The displacement field is scaled so its largest node reaches this.
    scale : float, optional
        Additional user exaggeration multiplier applied on top of
        ``deflection_ratio``.

    Returns
    -------
    FowtModeDeformation
        The scaled per-member complex displacement field.

    Raises
    ------
    ValueError
        If the solution has no DOF descriptions, ``mode_index`` is out of range, or
        ``deflection_ratio`` is not positive.
    """
    if not solution.dof_descriptions:
        msg = "ModalSolution has no dof_descriptions; cannot build a mode shape"
        raise ValueError(msg)
    if not 0 <= mode_index < solution.n_modes:
        msg = f"mode_index {mode_index} out of range [0, {solution.n_modes})"
        raise ValueError(msg)
    if deflection_ratio <= 0.0:
        msg = f"deflection_ratio must be positive, got {deflection_ratio}"
        raise ValueError(msg)

    geometry = geometry if geometry is not None else infer_geometry(solution)
    amplitudes = _extract_amplitudes(solution, mode_index)
    rigid = _platform_motion(amplitudes)
    top = _tower_top_motion(amplitudes, geometry.tower_length)

    raw = [
        _member_displacement(member, amplitudes, geometry, rigid, top)
        for member in geometry.members
    ]
    peak = max(
        (
            float(np.sqrt((np.abs(vec) ** 2).sum(axis=1)).max())
            for vec in raw
            if vec.size
        ),
        default=0.0,
    )
    target = deflection_ratio * geometry.characteristic_length * scale
    factor = target / peak if peak > 0.0 else 0.0
    displacements = [(vec * factor).astype(np.complex128) for vec in raw]
    return FowtModeDeformation(
        geometry=geometry,
        displacements=displacements,
        scale=factor,
        mode_index=mode_index,
        frequency_hz=float(solution.natural_frequencies_hz[mode_index]),
        damping_ratio=float(solution.damping_ratios[mode_index]),
    )


def infer_geometry(solution: ModalSolution) -> FowtGeometry:
    """Infer a schematic reference geometry from a solution's degrees of freedom.

    A floating (semi-submersible) schematic is chosen when platform degrees of
    freedom are present, otherwise a bottom-fixed monopile. The number of blades is
    taken from the highest blade index seen, defaulting to three. Principal
    dimensions use the preset defaults — pass an explicit
    :class:`~vane.config.system_config.FowtGeometry` to match a specific turbine.

    Parameters
    ----------
    solution : ModalSolution
        The solution whose ``dof_descriptions`` are inspected.

    Returns
    -------
    FowtGeometry
        A default geometry matching the detected support structure.
    """
    has_platform = False
    max_blade = 0
    for description in solution.dof_descriptions:
        info = classify_dof(description)
        if info.category in _PLATFORM_TRANSLATION or info.category in (
            _PLATFORM_ROTATION
        ):
            has_platform = True
        if info.blade is not None:
            max_blade = max(max_blade, info.blade)
    geometry = default_geometry(SystemType.SEMI if has_platform else SystemType.FIXED)
    if max_blade and max_blade != geometry.n_blades:
        # Fall through with the preset blade count; presets are 3-bladed and the
        # MBC check enforces three blades elsewhere, so this is informational only.
        pass
    return geometry


def _ghost_trace(member: Member) -> go.Scatter3d:
    """Return the faint undeformed outline trace for a member."""
    nodes = member.nodes
    return go.Scatter3d(
        x=nodes[:, 0],
        y=nodes[:, 1],
        z=nodes[:, 2],
        mode="lines",
        line={"color": _GHOST_COLOR, "width": 2},
        hoverinfo="skip",
        showlegend=False,
    )


def _deformed_trace(
    member: Member,
    nodes: npt.NDArray[np.float64],
    *,
    show_legend: bool,
) -> go.Scatter3d:
    """Return the coloured deformed trace for a member."""
    color = _KIND_COLORS.get(member.kind, "#333333")
    return go.Scatter3d(
        x=nodes[:, 0],
        y=nodes[:, 1],
        z=nodes[:, 2],
        mode="lines",
        line={"color": color, "width": 6},
        name=member.kind,
        legendgroup=member.kind,
        showlegend=show_legend,
        text=member.name,
        hovertemplate="%{text}<extra></extra>",
    )


_OFFSHORE = frozenset(
    {
        SystemType.SEMI,
        SystemType.SPAR,
        SystemType.TLP,
        SystemType.MONOPILE,
        SystemType.JACKET,
    }
)


def _water_trace(geometry: FowtGeometry) -> go.Mesh3d | None:
    """Return a translucent still-water plane for offshore systems, else ``None``."""
    if geometry.system_type not in _OFFSHORE:
        return None
    extent = max(
        (float(np.abs(member.nodes[:, :2]).max()) for member in geometry.members),
        default=geometry.characteristic_length,
    )
    span = 1.05 * extent
    corners = np.array(
        [[-span, -span], [span, -span], [span, span], [-span, span]], dtype=np.float64
    )
    return go.Mesh3d(
        x=corners[:, 0],
        y=corners[:, 1],
        z=np.zeros(4),
        i=[0, 0],
        j=[1, 2],
        k=[2, 3],
        color=_WATER_COLOR,
        opacity=0.12,
        hoverinfo="skip",
        showlegend=False,
        name="Still water",
    )


def _first_of_kind(members: list[Member]) -> set[int]:
    """Return the indices that are the first occurrence of each member kind."""
    seen: set[str] = set()
    firsts: set[int] = set()
    for index, member in enumerate(members):
        if member.kind not in seen:
            seen.add(member.kind)
            firsts.add(index)
    return firsts


def _scene_layout(title: str) -> dict[str, object]:
    """Return the 3-D scene layout dict shared by the static and animated figures."""
    return {
        "title": {"text": title},
        "scene": {
            "xaxis_title": "x, downwind (m)",
            "yaxis_title": "y, lateral (m)",
            "zaxis_title": "z, up (m)",
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.6, "y": 1.6, "z": 0.6}},
        },
        "margin": {"l": 0, "r": 0, "t": 40, "b": 0},
    }


def _mode_title(deformation: FowtModeDeformation) -> str:
    """Return a one-line title naming the mode, frequency, and damping."""
    return (
        f"Mode {deformation.mode_index} - "
        f"{deformation.frequency_hz:.3f} Hz, "
        f"damping {deformation.damping_ratio:.3f} "
        f"(schematic; deflection exaggerated {deformation.scale:.2g}x for visibility)"
    )


def plot_mode_fowt(
    solution: ModalSolution,
    mode_index: int,
    geometry: FowtGeometry | None = None,
    *,
    deflection_ratio: float = 0.12,
    scale: float = 1.0,
    phase_deg: float | None = None,
    template: str | go.layout.Template | None = None,
    title: str | None = None,
) -> go.Figure:
    """Plot a mode on the full wind-turbine structure at a single phase.

    The undeformed structure is drawn as a faint grey outline; the deformed
    structure — platform, mooring lines, tower, nacelle, and blades — is drawn in
    colour at the requested modal phase.

    Parameters
    ----------
    solution : ModalSolution
        A solution whose ``dof_descriptions`` are populated.
    mode_index : int
        Index of the mode to plot.
    geometry : FowtGeometry or None, optional
        Reference geometry; inferred from the solution when ``None``.
    deflection_ratio : float, optional
        Peak deflection as a fraction of the characteristic length.
    scale : float, optional
        Extra exaggeration multiplier.
    phase_deg : float or None, optional
        Modal phase in degrees at which to freeze the deformation. When ``None``
        (the default) the phase of maximum total deflection is used, which is the
        displacement extreme for a real mode and the most legible snapshot for a
        complex one.
    template : str, plotly.graph_objects.layout.Template or None, optional
        Plotly layout template to apply. ``None`` (the default) applies the VANE
        template ``"vane"``.
    title : str or None, optional
        Figure title. ``None`` (the default) names the mode, its frequency and
        damping and the deflection scale; ``""`` draws none.

    Returns
    -------
    plotly.graph_objects.Figure
        The 3-D mode-shape figure.

    Raises
    ------
    ValueError
        Propagated from :func:`deform_mode` for an empty, out-of-range, or
        non-positively-scaled request.
    """
    deformation = deform_mode(
        solution,
        mode_index,
        geometry,
        deflection_ratio=deflection_ratio,
        scale=scale,
    )
    geometry = deformation.geometry
    firsts = _first_of_kind(geometry.members)
    phase = (
        deformation.peak_phase_rad()
        if phase_deg is None
        else float(np.radians(phase_deg))
    )

    traces: list[go.Scatter3d | go.Mesh3d] = []
    water = _water_trace(geometry)
    if water is not None:
        traces.append(water)
    traces.extend(_ghost_trace(member) for member in geometry.members)
    for index, member in enumerate(geometry.members):
        nodes = deformation.deformed_nodes(index, phase)
        traces.append(_deformed_trace(member, nodes, show_legend=index in firsts))

    figure = go.Figure(data=traces)
    figure.update_layout(
        **_scene_layout(_mode_title(deformation) if title is None else title)
    )
    figure.update_layout(template=resolve_template(template))
    return figure


def plot_mode_fowt_animation(
    solution: ModalSolution,
    mode_index: int,
    geometry: FowtGeometry | None = None,
    *,
    motion: Literal["cycle", "swing"] = "cycle",
    deflection_ratio: float = 0.12,
    scale: float = 1.0,
    n_frames: int = 36,
    template: str | go.layout.Template | None = None,
    title: str | None = None,
) -> go.Figure:
    """Animate a mode on the full wind-turbine structure (Play, scrub, swing/orbit).

    The faint undeformed outline and the still-water plane stay fixed while the coloured
    structure oscillates, so the relative phasing across platform, tower, and blades
    reads as physical motion. The figure carries a Play/Pause control and a **frame
    slider** to scrub or replay any instant; the deformation **magnification** is set
    manually with ``deflection_ratio`` x ``scale`` (an engineering deformation-scale
    factor, as in an FE post-processor).

    Two motions are offered. ``"cycle"`` sweeps the full modal phase, so a whirling
    (complex) mode traces its true orbit. ``"swing"`` oscillates the real peak shape
    symmetrically about the undeformed state (``cos`` amplitude from ``+peak`` through
    ``0`` to ``-peak``) — the standing-wave animation a modal analyst expects for a
    single mode.

    Parameters
    ----------
    solution : ModalSolution
        A solution whose ``dof_descriptions`` are populated.
    mode_index : int
        Index of the mode to animate.
    geometry : FowtGeometry or None, optional
        Reference geometry; inferred from the solution when ``None``.
    motion : {"cycle", "swing"}, optional
        ``"cycle"`` sweeps the full complex phase (orbit); ``"swing"`` oscillates the
        real peak shape about the undeformed state. Default ``"cycle"``.
    deflection_ratio : float, optional
        Peak deflection as a fraction of the characteristic length.
    scale : float, optional
        Manual magnification multiplier on top of ``deflection_ratio``.
    n_frames : int, optional
        Number of frames over one period (must be positive).
    template : str, plotly.graph_objects.layout.Template or None, optional
        Plotly layout template to apply. ``None`` (the default) applies the VANE
        template ``"vane"``.
    title : str or None, optional
        Figure title. ``None`` (the default) names the mode, its frequency and
        damping and the deflection scale; ``""`` draws none.

    Returns
    -------
    plotly.graph_objects.Figure
        The animated 3-D mode-shape figure with a Play/Pause control and a scrub slider.

    Raises
    ------
    ValueError
        If ``n_frames`` is not positive, or as propagated from :func:`deform_mode`.
    """
    if n_frames < 1:
        msg = f"n_frames must be at least 1, got {n_frames}"
        raise ValueError(msg)
    if motion not in ("cycle", "swing"):
        msg = f"motion must be 'cycle' or 'swing', got {motion!r}"
        raise ValueError(msg)
    deformation = deform_mode(
        solution,
        mode_index,
        geometry,
        deflection_ratio=deflection_ratio,
        scale=scale,
    )
    geometry = deformation.geometry
    firsts = _first_of_kind(geometry.members)
    n_members = len(geometry.members)

    static: list[go.Scatter3d | go.Mesh3d] = []
    water = _water_trace(geometry)
    if water is not None:
        static.append(water)
    static.extend(_ghost_trace(member) for member in geometry.members)

    # The real peak-deflection field; "swing" scales this by a cosine, "cycle" sweeps
    # the complex phase to show the true orbit.
    peak_phase = deformation.peak_phase_rad()
    peak_offsets = [
        np.asarray((vec * np.exp(1j * peak_phase)).real, dtype=np.float64)
        for vec in deformation.displacements
    ]

    def member_nodes(index: int, fraction: float) -> npt.NDArray[np.float64]:
        """Return a member's node positions at a cycle fraction in ``[0, 1)``."""
        if motion == "swing":
            amplitude = float(np.cos(2.0 * np.pi * fraction))
            nodes = geometry.members[index].nodes + amplitude * peak_offsets[index]
            return np.asarray(nodes, dtype=np.float64)
        return deformation.deformed_nodes(index, 2.0 * np.pi * fraction)

    def deformed_traces(fraction: float) -> list[go.Scatter3d]:
        """Return the coloured deformed traces at one cycle fraction."""
        return [
            _deformed_trace(
                geometry.members[index],
                member_nodes(index, fraction),
                show_legend=index in firsts,
            )
            for index in range(n_members)
        ]

    deformed_indices = list(range(len(static), len(static) + n_members))
    # The loaded figure must show frame 0 so it agrees with the slider's active
    # step (0) and its "0°" label; both motions therefore start at fraction 0
    # (swing: cos 0 = +1 peak; cycle: the start of the orbit sweep).
    base = static + deformed_traces(0.0)
    frames = [
        go.Frame(
            data=deformed_traces(k / n_frames),
            traces=deformed_indices,
            name=str(k),
        )
        for k in range(n_frames)
    ]
    figure = go.Figure(data=base, frames=frames)
    figure.update_layout(
        **_scene_layout(_mode_title(deformation) if title is None else title)
    )
    figure.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.02,
                "y": 0.05,
                "xanchor": "left",
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 70, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "❚❚ Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[_frame_slider(n_frames)],
    )
    figure.update_layout(template=resolve_template(template))
    return figure


def _frame_slider(n_frames: int) -> dict[str, object]:
    """Return a Plotly slider that scrubs/replays the animation frame by frame."""
    return {
        "active": 0,
        "x": 0.12,
        "len": 0.84,
        "xanchor": "left",
        "y": 0.02,
        "currentvalue": {"prefix": "phase: ", "visible": True, "xanchor": "right"},
        "pad": {"t": 0, "b": 8},
        "steps": [
            {
                "label": f"{round(360 * k / n_frames):d}°",
                "method": "animate",
                "args": [
                    [str(k)],
                    {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    },
                ],
            }
            for k in range(n_frames)
        ],
    }


def plot_mode_fowt_static(
    solution: ModalSolution,
    mode_index: int,
    geometry: FowtGeometry | None = None,
    *,
    deflection_ratio: float = 0.12,
    scale: float = 1.0,
    phase_deg: float | None = None,
    elevation_deg: float = 15.0,
    azimuth_deg: float = -50.0,
    style: str | dict[str, Any] | Path | None = VANE_STYLE,
    size: str | float | FigureSize | None = None,
    title: str | None = None,
) -> Figure:
    """Plot a mode on the full wind-turbine structure as a static 3-D figure.

    The static counterpart of :func:`plot_mode_fowt`: the undeformed structure as a
    light grey outline and the deformed one coloured by member (tower black, blades
    green, platform red, as the mode families of the Campbell diagram; nacelle, hub
    and mooring grey), on equal axes, with a translucent still-water plane for
    offshore systems. The legend title notes that the deflection is exaggerated.

    Parameters
    ----------
    solution : ModalSolution
        A solution whose ``dof_descriptions`` are populated.
    mode_index : int
        Index of the mode to plot.
    geometry : FowtGeometry or None, optional
        Reference geometry; inferred from the solution when ``None``.
    deflection_ratio : float, optional
        Peak deflection as a fraction of the characteristic length.
    scale : float, optional
        Extra exaggeration multiplier.
    phase_deg : float or None, optional
        Modal phase in degrees; ``None`` uses the phase of maximum total deflection.
    elevation_deg, azimuth_deg : float, optional
        Viewing angles of the 3-D axes, in degrees.
    style : str, dict, pathlib.Path or None, optional
        Matplotlib style in effect while drawing (``"vane"`` by default).
    size : str, float, FigureSize or None, optional
        Figure size and fonts; ``None`` uses ``"single"``, nearly square.
    title : str or None, optional
        Axes title. ``None`` (the default) draws none; the legend title names the mode
        and its natural frequency.

    Returns
    -------
    matplotlib.figure.Figure
        The 3-D mode-shape figure.

    Raises
    ------
    ValueError
        Propagated from :func:`deform_mode` for an empty, out-of-range, or
        non-positively-scaled request.
    """
    import matplotlib.pyplot as plt

    deformation = deform_mode(
        solution,
        mode_index,
        geometry,
        deflection_ratio=deflection_ratio,
        scale=scale,
    )
    geometry = deformation.geometry
    phase = (
        deformation.peak_phase_rad()
        if phase_deg is None
        else float(np.radians(phase_deg))
    )
    firsts = _first_of_kind(geometry.members)
    resolved = resolve_size(size, "single", ax=None, style=style, aspect=(0.95, 0.6))
    with static_style(style, resolved):
        figure = plt.figure(layout="constrained")
        axes = figure.add_subplot(projection="3d")
        deformed = [
            deformation.deformed_nodes(index, phase)
            for index in range(len(geometry.members))
        ]
        extent = _view_extent(geometry, deformed)
        if geometry.system_type in _OFFSHORE:
            grid = np.array([-extent, extent])
            plane_x, plane_y = np.meshgrid(grid, grid)
            axes.plot_surface(
                plane_x,
                plane_y,
                np.zeros_like(plane_x),
                color=_WATER_COLOR,
                alpha=0.12,
                linewidth=0,
                shade=False,
            )
        for member in geometry.members:
            nodes = _within(member.nodes, extent)
            axes.plot(
                nodes[:, 0], nodes[:, 1], nodes[:, 2], color=_GHOST_COLOR, linewidth=0.8
            )
        for index, (member, moved) in enumerate(
            zip(geometry.members, deformed, strict=True)
        ):
            shown = _within(moved, extent)
            axes.plot(
                shown[:, 0],
                shown[:, 1],
                shown[:, 2],
                color=_KIND_COLORS.get(member.kind, "#333333"),
                linewidth=1.8,
                label=member.kind.capitalize() if index in firsts else None,
            )
        points = np.vstack(
            [_within(nodes, extent) for nodes in deformed]
            + [_within(m.nodes, extent) for m in geometry.members]
        )
        z_low, z_high = float(points[:, 2].min()), float(points[:, 2].max())
        axes.set_xlim(-extent, extent)
        axes.set_ylim(-extent, extent)
        axes.set_zlim(z_low, z_high)
        axes.set_box_aspect(
            (2.0 * extent, 2.0 * extent, max(z_high - z_low, 0.1 * extent)),
            zoom=0.78,
        )
        for axis in (axes.xaxis, axes.yaxis):
            axis.set_major_locator(MaxNLocator(4))
        axes.set_xlabel("x (m)")
        axes.set_ylabel("y (m)")
        axes.set_zlabel("z (m)")
        axes.view_init(elev=elevation_deg, azim=azimuth_deg)
        figure.legend(
            loc="outside upper center",
            ncols=3,
            title=(
                f"Mode {deformation.mode_index} ({deformation.frequency_hz:.4g} Hz); "
                "schematic, deflection exaggerated"
            ),
        )
        if title is not None:
            axes.set_title(title)
    return figure


def _view_extent(
    geometry: FowtGeometry, deformed: list[npt.NDArray[np.float64]]
) -> float:
    """Return the half-width of the static view: the structure without its mooring.

    The mooring spreads far wider than the turbine, so the view is sized on the other
    members (with a 25 % margin) and the mooring lines are cut at its edge.
    """
    members = [
        nodes
        for member, nodes in zip(geometry.members, deformed, strict=True)
        if member.kind != "mooring"
    ] or deformed
    reach = max(float(np.abs(nodes[:, :2]).max()) for nodes in members)
    height = max(float(np.ptp(nodes[:, 2])) for nodes in members)
    return 1.25 * max(reach, 0.25 * height, 1.0)


def _within(nodes: npt.NDArray[np.float64], extent: float) -> npt.NDArray[np.float64]:
    """Return the leading nodes of a polyline that lie within the horizontal view."""
    inside = np.all(np.abs(nodes[:, :2]) <= extent, axis=1)
    if bool(inside.all()):
        return nodes
    outside = np.flatnonzero(~inside)
    first = int(outside[0])
    if first == 0:
        keep = np.flatnonzero(inside)
        return nodes[keep] if keep.size else nodes[:0]
    return nodes[:first]
