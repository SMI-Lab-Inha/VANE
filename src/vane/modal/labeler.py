# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Physics-aware labeling of system modes.

Each mode is labeled by the physical degree-of-freedom category that dominates
it. Participation is aggregated per :class:`DofCategory` (via
:func:`vane.config.dof_map.classify_dof`), the dominant category names the mode
(e.g. "1st Tower Fore-Aft"), and the fraction of participation it carries is
reported as a confidence — a low value flags an ambiguous, mixed mode.

:func:`label_solution` votes with the scale-invariant left--right (biorthogonal)
participation of :mod:`vane.modal.biorthogonal`, so the many unclassified module
states of a coupled model (mooring line nodes, hydrodynamic radiation states,
aerodynamic states) cannot out-vote the structural degrees of freedom merely by
having large raw eigenvector amplitudes in their own units.

BeamDyn blades have no modal DOFs: their nodal states vote as flapwise, edgewise,
torsional, or axial blade motion (see :mod:`vane.config.dof_map`), and
:func:`label_solution` numbers the resulting flap, edge, and torsion modes from the
spanwise shape of their section rotation and their frequency order.

Ambiguity is reported, never resolved silently. A mode whose runner-up category
carries at least :data:`LABEL_AMBIGUITY_RATIO` of the winner's weight (a hybrid, e.g.
a monopile's second tower fore-aft mode mixed with the progressive second flap), or
whose multi-blade type is not decided by its collective/cyclic split and whirl, is
returned with ``is_ambiguous=True`` and the runner-up as ``alternative``. Within one
operating point each blade mode family (e.g. the second flap) has exactly one
collective, one regressive and one progressive member; :func:`label_solution`
enforces this (see :func:`label_solution`, Notes).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from vane.config.dof_map import DofCategory, classify_dof
from vane.modal.biorthogonal import biorthogonal_participation
from vane.modal.eigensolver import DEGENERACY_RTOL

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from vane.modal.eigensolver import ModalSolution
    from vane.modal.participation import ParticipationResult

__all__ = [
    "BEAMDYN_RIGID_RANGE",
    "BEAMDYN_SHAPE_TOLERANCE",
    "DEFAULT_HEAVILY_DAMPED_THRESHOLD",
    "DEFAULT_MIN_KNOWN_PARTICIPATION",
    "DRIVETRAIN_ROOT_SHARE",
    "LABEL_AMBIGUITY_RATIO",
    "TOWER_SUPPORT_SHARE",
    "WHIRL_CIRCULARITY_THRESHOLD",
    "ModeLabel",
    "category_to_label",
    "label_mode",
    "label_modes",
    "label_solution",
]

LABEL_AMBIGUITY_RATIO = 0.75
"""Runner-up to winner weight ratio at or above which a label is ambiguous.

The category vote (and, for a blade mode, the collective-versus-cyclic split) names
a mode after its largest share. When the runner-up carries at least this fraction of
the winner's weight -- e.g. 0.40 against 0.53 -- the mode is a hybrid of the two, and
which name it gets is decided by a few percent of participation that move with the
operating point. The label is then reported as ambiguous with the runner-up as its
alternative. ``0.75`` is set from the reference-turbine sweeps: the clean modes of
the NREL 5-MW, IEA 3.4-, 10-, 15- and 22-MW turbines have runner-up ratios below
0.4, while the hybrids that flip labels between neighbouring operating points
(monopile tower/flap, flap/edge veering, flap/rotor-speed, aero-state/flap) sit
between 0.8 and 1.0.
"""

WHIRL_CIRCULARITY_THRESHOLD = 0.2
"""Circularity below which a cyclic blade mode's whirl direction is undetermined.

The circularity :math:`2\\operatorname{Im}(q_s \\overline{q_c}) / (|q_c|^2 +
|q_s|^2)` is ``+1``/``-1`` for a pure backward/forward whirl and ``0`` for a
standing pattern; below this magnitude the mode is a standing (``"cyclic"``) pattern,
e.g. a mixture of the regressive and progressive members of a near-coincident pair.
"""

DEFAULT_HEAVILY_DAMPED_THRESHOLD = 0.99
"""Damping ratio at or above which a mode is heavily damped (near-critical).

The ratio is measured in the frame in which the mode is defined: the rotating frame
for a regressive or progressive blade whirl mode (see
:func:`vane.modal.identifier.rotor_frame_damping_ratios`), the fixed frame otherwise.
``0.99`` means within 1 % of critical damping, where the damped frequency is below
15 % of the natural frequency and the natural frequency :math:`|\\lambda|` is
essentially a decay rate. It is set from the reference-turbine sweeps (NREL 5-MW land
and OC3 monopile, IEA 3.4-MW, IEA 10-MW and IEA 15-MW monopile, CalcSteady with
quasi-steady aero): every first-flap Campbell line that tracks cleanly stays at or
below 0.95 in this frame (its fixed-frame value reaches 0.998 on the regressive
branch), while the modes that broke the IEA 15-MW lines above rated sit at 1.000,
i.e. past critical damping in the rotating frame. Lightly damped lines that the
community still draws (first flap at 0.25-0.78 in published IEA Campbell diagrams)
are far below it.
"""

DRIVETRAIN_ROOT_SHARE = 0.25
"""Minimum in-phase root rotation of a BeamDyn drivetrain-torsion mode.

With the drivetrain flexible (``DrTrDOF``) and the generator locked (no ``GenDOF``),
the lowest collective edgewise BeamDyn mode is the drivetrain torsion: the whole
rotor swings against the shaft spring, so the blade roots rotate in phase with the
tips by a large fraction of the tip rotation (one half on the NREL 5-MW). In a
collective edgewise bending mode the hub instead stays put or counter-rotates. A
collective edgewise mode whose root section rotation is in phase with its peak
rotation and at least this fraction of it is named ``"Drivetrain Torsion"``.
"""

_WHIRL_TYPES = ("collective", "regressive", "progressive")

# Smallest fit (:meth:`_MultibladeInfo.score`, the share of a mode's multi-blade
# content compatible with a type) for which the per-point whirl assignment may give a
# mode that type. A mode carrying less than a quarter of its blade motion in a
# collective (or in the whirl direction) is not that type, however free its slot is.
_MIN_WHIRL_FIT = 1.0 - LABEL_AMBIGUITY_RATIO

# Relative slack of the share comparisons against LABEL_AMBIGUITY_RATIO and
# _MIN_WHIRL_FIT. A share exactly at the ratio is a tie that floating-point rounding
# (which differs between NumPy builds) must not decide: it always counts as reaching it.
_RATIO_SLACK = 1e-9


def _reaches(share: float, ratio: float, reference: float) -> bool:
    """Return whether ``share`` is at least ``ratio * reference``, ties included."""
    return share >= ratio * reference * (1.0 - _RATIO_SLACK)


DEFAULT_MIN_KNOWN_PARTICIPATION = 0.5
"""Default minimum share of participation that classified DOFs must carry.

A mode is named after a physical DOF category only when the classified
(structural) degrees of freedom carry at least half of its total participation,
i.e. when the structure -- not unclassified module states -- is the majority
participant. Below this the mode is reported as ``"Unidentified"``.
"""

_LABELS: dict[DofCategory, str] = {
    DofCategory.PLATFORM_SURGE: "Platform Surge",
    DofCategory.PLATFORM_SWAY: "Platform Sway",
    DofCategory.PLATFORM_HEAVE: "Platform Heave",
    DofCategory.PLATFORM_ROLL: "Platform Roll",
    DofCategory.PLATFORM_PITCH: "Platform Pitch",
    DofCategory.PLATFORM_YAW: "Platform Yaw",
    DofCategory.TOWER_FORE_AFT_1: "1st Tower Fore-Aft",
    DofCategory.TOWER_SIDE_SIDE_1: "1st Tower Side-Side",
    DofCategory.TOWER_FORE_AFT_2: "2nd Tower Fore-Aft",
    DofCategory.TOWER_SIDE_SIDE_2: "2nd Tower Side-Side",
    DofCategory.NACELLE_YAW: "Nacelle Yaw",
    DofCategory.GENERATOR_AZIMUTH: "Rotor/Generator",
    DofCategory.DRIVETRAIN_TORSION: "Drivetrain Torsion",
    DofCategory.ROTOR_FURL: "Rotor Furl",
    DofCategory.TAIL_FURL: "Tail Furl",
    DofCategory.TEETER: "Teeter",
    DofCategory.BLADE_FLAP_1: "1st Blade Flap",
    DofCategory.BLADE_FLAP_2: "2nd Blade Flap",
    DofCategory.BLADE_EDGE_1: "1st Blade Edge",
    DofCategory.BLADE_PITCH: "Blade Pitch",
    DofCategory.BLADE_FLAP: "Blade Flap",
    DofCategory.BLADE_EDGE: "Blade Edge",
    DofCategory.BLADE_TORSION: "Blade Torsion",
    DofCategory.BLADE_AXIAL: "Blade Axial",
    DofCategory.UNKNOWN: "Unidentified",
}


TOWER_SUPPORT_SHARE = 1.0 / 3.0
"""Minimum share of a support plane's participation that tower bending must carry.

A support plane groups the platform translation and tilt with the tower bending DOFs
of one plane (fore-aft: surge, pitch and the tower fore-aft modes; side-side: sway,
roll and the tower side-side modes). On a fixed-bottom support whose substructure
flexibility is carried by the platform DOFs (e.g. a monopile), the global tower
bending modes put a large share of their participation on platform translation, so a
per-category vote would call them platform modes. When tower bending carries at least
this share of the plane's participation and the plane as a whole out-votes every other
category, the mode is a tower bending mode of that plane. A rigid-body platform mode
of a floating turbine has (near-)zero tower participation and is never regrouped.
"""

_PLATFORM_FORE_AFT = (DofCategory.PLATFORM_SURGE, DofCategory.PLATFORM_PITCH)
_PLATFORM_SIDE_SIDE = (DofCategory.PLATFORM_SWAY, DofCategory.PLATFORM_ROLL)
_TOWER_FORE_AFT = (DofCategory.TOWER_FORE_AFT_1, DofCategory.TOWER_FORE_AFT_2)
_TOWER_SIDE_SIDE = (DofCategory.TOWER_SIDE_SIDE_1, DofCategory.TOWER_SIDE_SIDE_2)
_SUPPORT_PLANES = (
    (_PLATFORM_FORE_AFT, _TOWER_FORE_AFT),
    (_PLATFORM_SIDE_SIDE, _TOWER_SIDE_SIDE),
)
# The tower-bending mode number implied by the dominant tower DOF category.
_TOWER_DOF_ORDINAL = {
    DofCategory.TOWER_FORE_AFT_1: 1,
    DofCategory.TOWER_FORE_AFT_2: 2,
    DofCategory.TOWER_SIDE_SIDE_1: 1,
    DofCategory.TOWER_SIDE_SIDE_2: 2,
}

# Energy weight of a multi-blade coordinate: for N >= 3 blades the summed blade
# energy is N |q0|^2 + (N/2)(|qc|^2 + |qs|^2), so a cyclic coordinate counts half.
_MBC_ENERGY_WEIGHT = {"collective": 1.0, "cosine": 0.5, "sine": 0.5}

# Categories that may carry a multi-blade (collective/cyclic whirl) type.
_MULTIBLADE_CATEGORIES = frozenset(
    {
        DofCategory.BLADE_FLAP_1,
        DofCategory.BLADE_FLAP_2,
        DofCategory.BLADE_EDGE_1,
        DofCategory.BLADE_PITCH,
        DofCategory.BLADE_FLAP,
        DofCategory.BLADE_EDGE,
        DofCategory.BLADE_TORSION,
        DofCategory.BLADE_AXIAL,
        DofCategory.UNKNOWN,
    }
)

# BeamDyn bending/torsion families and the nodal rotation that resolves each one's
# spanwise shape: a flapwise deflection (along X) turns the section about Y, an
# edgewise deflection (along Y) turns it about X, and torsion is the rotation about Z.
_BEAMDYN_FAMILY_ROTATION = {
    DofCategory.BLADE_FLAP: "RY",
    DofCategory.BLADE_EDGE: "RX",
    DofCategory.BLADE_TORSION: "RZ",
}

BEAMDYN_SHAPE_TOLERANCE = 0.05
"""Relative size below which a spanwise rotation increment is ignored.

The order of a BeamDyn bending or torsion mode is one plus the number of sign changes
of the increments of its nodal section rotation along the span; increments smaller
than this fraction of the largest one are treated as numerical noise and skipped.
"""

BEAMDYN_RIGID_RANGE = 0.1
"""Spanwise rotation spread below which a BeamDyn blade mode counts as rigid.

The linearized BeamDyn nodal states carry the rigid motion of the blade root. When
the section rotation varies along the span by less than this fraction of its largest
value, the blade turns as a rigid body and the mode is not an elastic bending mode;
a rigid collective edgewise rotation (all blades turning together in the rotor
plane) is the rotor-azimuth mode.
"""


def category_to_label(category: DofCategory) -> str:
    """Return the human-readable label for a DOF category.

    Parameters
    ----------
    category : DofCategory
        The category to name.

    Returns
    -------
    str
        The display label (``"Unidentified"`` for unknown categories).
    """
    return _LABELS.get(category, _LABELS[DofCategory.UNKNOWN])


@dataclass(frozen=True)
class ModeLabel:
    """Physical label for a single mode.

    Parameters
    ----------
    category : DofCategory
        The dominant DOF category.
    label : str
        Human-readable name of the dominant category.
    confidence : float
        Fraction of total participation carried by the dominant category, in
        ``[0, 1]``; low values indicate a mixed or ambiguous mode. For an
        ``"Unidentified"`` mode it is the share carried by unclassified states.
    dominant_dofs : list[str]
        Descriptions of the highest-participation degrees of freedom (up to three).
    multiblade : str or None
        For a blade mode after the MBC transform, the multi-blade type
        ``"collective"``, ``"regressive"``, ``"progressive"``, or ``"cyclic"`` (a
        standing cyclic mode with no whirl direction); ``None`` for non-blade modes
        or when no multi-blade information is available.
    is_ambiguous : bool, optional
        Whether the label is not decided by the mode's participation: the runner-up
        category carries at least :data:`LABEL_AMBIGUITY_RATIO` of the winner's
        weight, the multi-blade type is not decided (collective and cyclic shares
        within that ratio, or a standing cyclic pattern), or the per-point
        multi-blade assignment of :func:`label_solution` overrode the mode's own
        type. Default ``False``.
    alternative : str or None, optional
        The strongest competing label: the runner-up category of the vote, the
        competing multi-blade type of an undecided blade mode, or the label the mode
        had before a multi-blade reassignment; ``None`` when nothing competes. It is
        given for unambiguous labels too (as the runner-up); ``is_ambiguous`` says
        whether it is a close competitor.
    alternative_category : DofCategory or None, optional
        The category of ``alternative``.
    alternative_confidence : float, optional
        The share of the participation carried by ``alternative`` (``0.0`` when
        there is none).
    """

    category: DofCategory
    label: str
    confidence: float
    dominant_dofs: list[str]
    multiblade: str | None = None
    is_ambiguous: bool = False
    alternative: str | None = None
    alternative_category: DofCategory | None = None
    alternative_confidence: float = 0.0


def label_mode(
    magnitudes: npt.NDArray[np.float64],
    descriptions: Sequence[str],
    *,
    multiblade: str | None = None,
    min_known_fraction: float | None = None,
) -> ModeLabel:
    """Label one mode from its per-DOF participation weights.

    Parameters
    ----------
    magnitudes : npt.NDArray[np.float64]
        Non-negative participation weight of each DOF, shape ``(n_dof,)``: raw
        mode-shape magnitudes, or (preferably) the scale-invariant participation
        fractions used by :func:`label_solution`.
    descriptions : Sequence[str]
        DOF descriptions parallel to ``magnitudes``.
    multiblade : str or None, optional
        Multi-blade type (``"collective"``/``"regressive"``/``"progressive"``) for a
        blade mode; appended to the label and stored on the result.
    min_known_fraction : float or None, optional
        When ``None`` (the default), the plain plurality vote: the category with
        the largest summed weight wins, :attr:`DofCategory.UNKNOWN` included. When
        a number in ``[0, 1]`` is given, the dominant category is the largest
        *classified* (non-``UNKNOWN``) category, accepted only if the classified
        categories together carry at least this fraction of the total weight;
        otherwise the mode is ``"Unidentified"``. On this (participation) path a
        support plane -- the platform translation and tilt plus the tower bending
        DOFs of one plane -- votes as one group when tower bending carries at least
        :data:`TOWER_SUPPORT_SHARE` of it; if the group out-votes every other
        category the mode is named after its dominant tower DOF.

    Returns
    -------
    ModeLabel
        The dominant category, its label, the confidence, and the top DOFs. The
        confidence is the dominant category's share of the total weight (the whole
        support plane's share for a regrouped tower mode); for an
        ``"Unidentified"`` mode it is the share carried by unclassified states.

    Raises
    ------
    ValueError
        If ``magnitudes`` and ``descriptions`` have different lengths, or if
        ``min_known_fraction`` is outside ``[0, 1]`` or non-finite.
    TypeError
        If ``min_known_fraction`` is not a real number.
    """
    if magnitudes.shape[0] != len(descriptions):
        msg = (
            f"magnitudes length {magnitudes.shape[0]} does not match "
            f"{len(descriptions)} descriptions"
        )
        raise ValueError(msg)
    threshold = (
        None if min_known_fraction is None else _validate_fraction(min_known_fraction)
    )

    totals: dict[DofCategory, float] = {}
    for magnitude, description in zip(magnitudes.tolist(), descriptions, strict=True):
        category = classify_dof(description).category
        totals[category] = totals.get(category, 0.0) + float(magnitude)

    total = sum(totals.values())
    if not totals or total <= 0.0:
        return ModeLabel(
            DofCategory.UNKNOWN, _LABELS[DofCategory.UNKNOWN], 0.0, [], multiblade
        )

    order = np.argsort(magnitudes)[::-1]
    dominant_dofs = [descriptions[i] for i in order[:3].tolist() if magnitudes[i] > 0.0]
    members: tuple[DofCategory, ...]
    if threshold is None:
        dominant = max(totals, key=lambda category: totals[category])
        weight = totals[dominant]
        members = (dominant,)
    else:
        known: dict[DofCategory, float] = {
            c: v for c, v in totals.items() if c is not DofCategory.UNKNOWN
        }
        known_total = sum(known.values())
        if known_total <= 0.0 or known_total < threshold * total:
            dominant = DofCategory.UNKNOWN
            weight = totals.get(dominant, 0.0)
            members = (dominant,)
        else:
            dominant = max(known, key=lambda category: known[category])
            weight = known[dominant]
            members = (dominant,)
            support = _support_plane_vote(known)
            if support is not None:
                dominant, weight, members = support
    confidence = weight / total
    label = _LABELS[dominant]
    if multiblade is not None:
        label = f"{label} ({multiblade.capitalize()})"
    rival = _rival_vote(totals, members)
    if rival is None:
        return ModeLabel(dominant, label, confidence, dominant_dofs, multiblade)
    return ModeLabel(
        dominant,
        label,
        confidence,
        dominant_dofs,
        multiblade,
        is_ambiguous=weight > 0.0 and _reaches(rival[1], LABEL_AMBIGUITY_RATIO, weight),
        alternative=_LABELS[rival[0]],
        alternative_category=rival[0],
        alternative_confidence=rival[1] / total,
    )


def _validate_fraction(value: object) -> float:
    """Return ``value`` as a float after checking it is a finite number in [0, 1]."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        msg = f"participation threshold must be a number in [0, 1], got {value!r}"
        raise TypeError(msg)
    fraction = float(value)
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        msg = (
            f"participation threshold must be a finite number in [0, 1], got {value!r}"
        )
        raise ValueError(msg)
    return fraction


def _support_plane_vote(
    known: dict[DofCategory, float],
) -> tuple[DofCategory, float, tuple[DofCategory, ...]] | None:
    """Return the tower category, weight and members of a winning support plane.

    For each plane the platform translation/tilt and tower bending weights are
    pooled. A plane qualifies only when tower bending carries at least
    :data:`TOWER_SUPPORT_SHARE` of the pool, and it wins only when the pool exceeds
    every category outside it. The mode is then named after the plane's dominant
    tower DOF. Returns ``None`` when no plane wins.
    """
    best: tuple[DofCategory, float, tuple[DofCategory, ...]] | None = None
    for dominant_tower, pool, members in _qualifying_planes(known):
        rival = max(
            (weight for c, weight in known.items() if c not in members), default=0.0
        )
        if pool > rival and (best is None or pool > best[1]):
            best = (dominant_tower, pool, members)
    return best


def _qualifying_planes(
    weights: dict[DofCategory, float],
) -> list[tuple[DofCategory, float, tuple[DofCategory, ...]]]:
    """Return ``(dominant tower DOF, pooled weight, members)`` of each tower plane.

    Only planes whose tower bending carries at least :data:`TOWER_SUPPORT_SHARE` of
    the plane's pooled platform-plus-tower weight are returned.
    """
    planes: list[tuple[DofCategory, float, tuple[DofCategory, ...]]] = []
    for platform, tower in _SUPPORT_PLANES:
        tower_weight = sum(weights.get(category, 0.0) for category in tower)
        pool = tower_weight + sum(weights.get(category, 0.0) for category in platform)
        if tower_weight <= 0.0 or tower_weight < TOWER_SUPPORT_SHARE * pool:
            continue
        dominant_tower = max(tower, key=lambda category: weights.get(category, 0.0))
        planes.append((dominant_tower, pool, (*platform, *tower)))
    return planes


def _rival_vote(
    totals: dict[DofCategory, float], members: tuple[DofCategory, ...]
) -> tuple[DofCategory, float] | None:
    """Return the strongest competitor of the winning category group.

    Every category outside ``members`` competes on its own weight (unclassified
    states included, as ``UNKNOWN``); a qualifying support plane disjoint from
    ``members`` competes with its pooled weight under its dominant tower DOF. Ties
    resolve to the earlier category in :class:`DofCategory` order. Returns ``None``
    when there is no competitor with positive weight.
    """
    candidates = [
        (category, weight)
        for category, weight in totals.items()
        if category not in members and weight > 0.0
    ]
    candidates.extend(
        (tower, pool)
        for tower, pool, plane in _qualifying_planes(totals)
        if not set(plane) & set(members)
    )
    if not candidates:
        return None
    order = list(DofCategory)
    return max(candidates, key=lambda item: (item[1], -order.index(item[0])))


def _multiblade_type(
    shape: npt.NDArray[np.complex128],
    mbc_coordinates: Sequence[str],
    weights: npt.NDArray[np.float64] | None = None,
    *,
    min_blade_share: float = 0.5,
) -> str | None:
    r"""Classify a mode as collective/regressive/progressive from its MBC coordinates.

    Collective vs cyclic is decided by which multi-blade coordinates carry the mode;
    for a cyclic mode the whirl direction is the sign of
    :math:`\operatorname{Im}(q_s \, \overline{q_c})` for the dominant cosine/sine
    pair, which is invariant to the mode's overall phase. Returns ``None`` when the
    mode carries no multi-blade participation.

    Parameters
    ----------
    shape : npt.NDArray[np.complex128]
        Complex mode shape over the degrees of freedom.
    mbc_coordinates : Sequence[str]
        Multi-blade coordinate of each DOF (``"collective"``/``"cosine"``/``"sine"``
        or ``""``), aligned with ``shape``.
    weights : npt.NDArray[np.float64] or None, optional
        Scale-invariant participation of each DOF, aligned with ``shape``. When
        given (and finite) it decides the blade share and collective-vs-cyclic
        split. Otherwise the blade energy is used: :math:`|q_0|^2` for a collective
        and :math:`|q_c|^2/2`, :math:`|q_s|^2/2` for the cyclic coordinates, which
        is each coordinate's share of :math:`\sum_b |q_b|^2` over the blades. (A
        plain magnitude sum would count the cyclic pair twice and call a mainly
        collective mode cyclic.)
    min_blade_share : float, optional
        Minimum share of the mode that the multi-blade coordinates must carry for
        it to receive a type (default ``0.5``), so a non-blade mode with minor
        coupling into the rotor is not given one.

    Returns
    -------
    str or None
        ``"collective"``, ``"regressive"``, ``"progressive"``, ``"cyclic"`` (a
        standing cyclic mode), or ``None``.
    """
    return _multiblade_info(
        shape, mbc_coordinates, weights, min_blade_share=min_blade_share
    ).kind


@dataclass(frozen=True)
class _MultibladeInfo:
    """The multi-blade decomposition of one mode.

    ``kind`` is the multi-blade type (or ``None``), ``collective`` and ``cyclic`` the
    shares carried by the collective and the cosine/sine coordinates, and
    ``circularity`` the whirl circularity of the dominant cosine/sine pair in
    ``[-1, 1]`` (positive for a backward, i.e. regressive, whirl; ``0`` when there
    is no cyclic pair).
    """

    kind: str | None
    collective: float = 0.0
    cyclic: float = 0.0
    circularity: float = 0.0

    @property
    def is_undecided(self) -> bool:
        """Return whether the collective/cyclic split or the whirl is undecided."""
        if self.kind == "cyclic":
            return True
        if self.kind not in _WHIRL_TYPES:
            return False
        larger = max(self.collective, self.cyclic)
        smaller = min(self.collective, self.cyclic)
        return larger > 0.0 and _reaches(smaller, LABEL_AMBIGUITY_RATIO, larger)

    def score(self, kind: str) -> float:
        """Return how well the mode fits multi-blade type ``kind``, in ``[0, 1]``."""
        total = self.collective + self.cyclic
        if total <= 0.0:
            return 0.0
        if kind == "collective":
            return self.collective / total
        direction = 0.5 * (1.0 + self.circularity)
        if kind == "progressive":
            direction = 1.0 - direction
        return self.cyclic / total * direction


def _multiblade_info(
    shape: npt.NDArray[np.complex128],
    mbc_coordinates: Sequence[str],
    weights: npt.NDArray[np.float64] | None = None,
    *,
    min_blade_share: float = 0.5,
) -> _MultibladeInfo:
    """Return the multi-blade decomposition of a mode (see :func:`_multiblade_type`)."""
    if not mbc_coordinates:
        return _MultibladeInfo(None)
    magnitude = np.abs(shape)
    if weights is not None and weights.shape == magnitude.shape:
        share = np.asarray(weights, dtype=np.float64)
        if not bool(np.all(np.isfinite(share))):
            share = _mbc_energy(magnitude, mbc_coordinates)
    else:
        share = _mbc_energy(magnitude, mbc_coordinates)
    total = float(share.sum())
    if not total > 0.0:
        return _MultibladeInfo(None)
    collective = float(
        sum(share[i] for i, c in enumerate(mbc_coordinates) if c == "collective")
    )
    cyclic_indices = [
        i for i, c in enumerate(mbc_coordinates) if c in ("cosine", "sine")
    ]
    cyclic = float(sum(share[i] for i in cyclic_indices))
    # Only a blade-dominated mode gets a multi-blade type: a non-blade mode (e.g.
    # tower) with tiny coupling into the blade coordinates must not be mislabeled.
    if collective + cyclic <= 0.0 or collective + cyclic < min_blade_share * total:
        return _MultibladeInfo(None, collective, cyclic)
    circularity = _circularity(shape, mbc_coordinates, share, cyclic_indices)
    if collective >= cyclic:
        return _MultibladeInfo("collective", collective, cyclic, circularity or 0.0)
    if circularity is None:
        return _MultibladeInfo(None, collective, cyclic)
    # Circularity in [-1, 1]: +/-1 for a pure forward/backward whirl, ~0 for a
    # standing pattern OR when one coordinate is mere numerical leakage (both
    # whirl and intensity then collapse). Only a genuinely whirling mode -- both
    # coordinates significant AND in quadrature -- is given a whirl direction.
    if abs(circularity) < WHIRL_CIRCULARITY_THRESHOLD:
        return _MultibladeInfo("cyclic", collective, cyclic, circularity)
    # Sign calibrated against the known Coleman split: the lower-frequency backward
    # whirl (w0 - Omega) is regressive, the higher one progressive.
    kind = "regressive" if circularity > 0.0 else "progressive"
    return _MultibladeInfo(kind, collective, cyclic, circularity)


def _circularity(
    shape: npt.NDArray[np.complex128],
    mbc_coordinates: Sequence[str],
    share: npt.NDArray[np.float64],
    cyclic_indices: list[int],
) -> float | None:
    """Return the whirl circularity of the dominant cosine/sine pair.

    ``None`` when the dominant cyclic coordinate has no partner; ``0.0`` when the
    pair carries no amplitude or there are no cyclic coordinates.
    """
    if not cyclic_indices:
        return 0.0
    dominant = max(cyclic_indices, key=lambda i: float(share[i]))
    cosine_index, sine_index = (
        (dominant, dominant + 1)
        if mbc_coordinates[dominant] == "cosine"
        else (dominant - 1, dominant)
    )
    if not (0 <= cosine_index < shape.shape[0] and 0 <= sine_index < shape.shape[0]):
        return None
    cosine_value, sine_value = shape[cosine_index], shape[sine_index]
    intensity = float(np.abs(cosine_value) ** 2 + np.abs(sine_value) ** 2)
    if intensity <= 0.0:
        return 0.0
    whirl = float((sine_value * np.conj(cosine_value)).imag)
    return 2.0 * whirl / intensity


def _mbc_energy(
    magnitude: npt.NDArray[np.float64], mbc_coordinates: Sequence[str]
) -> npt.NDArray[np.float64]:
    """Return each DOF's energy weight: squared magnitude, cyclic coordinates halved."""
    factors = np.array(
        [_MBC_ENERGY_WEIGHT.get(coordinate, 1.0) for coordinate in mbc_coordinates],
        dtype=np.float64,
    )
    if factors.shape != magnitude.shape:
        factors = np.ones_like(magnitude)
    energy: npt.NDArray[np.float64] = factors * magnitude**2
    return energy


def _ordinal(number: int) -> str:
    """Return ``number`` as an English ordinal (``1st``, ``2nd``, ``3rd``, ``4th``)."""
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def _number_tower_modes(
    solution: ModalSolution, labels: list[ModeLabel]
) -> list[ModeLabel]:
    """Return the labels with each plane's tower modes numbered by frequency.

    The dominant tower DOF gives only a lower bound on a mode's number: on a
    fixed-bottom support the global bending modes outnumber the two tower DOFs per
    plane (the substructure adds platform translation and tilt), so the third
    fore-aft mode is still dominated by the second tower DOF. Within a plane the
    tower modes are therefore numbered in frequency order, each at least its DOF's
    number and above the previous mode's; the modes of one degenerate cluster share
    a number. The category is left as the dominant DOF category.
    """
    frequencies = np.asarray(solution.natural_frequencies_hz, dtype=np.float64)
    eigenvalues = np.asarray(solution.eigenvalues, dtype=np.complex128)
    flags = np.zeros(len(labels), dtype=np.bool_)
    if np.shape(solution.is_degenerate) == flags.shape:
        flags = np.asarray(solution.is_degenerate, dtype=np.bool_)
    if frequencies.shape != flags.shape or eigenvalues.shape != flags.shape:
        return labels
    result = list(labels)
    for tower in (_TOWER_FORE_AFT, _TOWER_SIDE_SIDE):
        members = sorted(
            (k for k, label in enumerate(labels) if label.category in tower),
            key=lambda k: float(frequencies[k]),
        )
        number, previous = 0, -1
        for k in members:
            dof_number = _TOWER_DOF_ORDINAL[labels[k].category]
            if previous >= 0 and _coincident(eigenvalues, flags, previous, k):
                number = max(dof_number, number)
            else:
                number = max(dof_number, number + 1)
            previous = k
            if number != dof_number:
                plane = "Fore-Aft" if tower is _TOWER_FORE_AFT else "Side-Side"
                text = f"{_ordinal(number)} Tower {plane}"
                multiblade = labels[k].multiblade
                if multiblade is not None:
                    text = f"{text} ({multiblade.capitalize()})"
                result[k] = replace(labels[k], label=text)
    return result


def _coincident(
    eigenvalues: npt.NDArray[np.complex128],
    flags: npt.NDArray[np.bool_],
    first: int,
    second: int,
) -> bool:
    """Return whether two degenerate-flagged modes share a repeated eigenvalue."""
    if not (bool(flags[first]) and bool(flags[second])):
        return False
    scale = max(abs(eigenvalues[first]), abs(eigenvalues[second]), 1e-30)
    return bool(
        abs(eigenvalues[first] - eigenvalues[second]) <= DEGENERACY_RTOL * scale
    )


def _beamdyn_rotation_profile(
    shape: npt.NDArray[np.complex128],
    rows: dict[tuple[str, int | None], list[tuple[int, int]]],
) -> npt.NDArray[np.float64] | None:
    """Return a mode's real spanwise section-rotation profile, root to tip.

    ``rows`` maps a ``(multi-blade coordinate, blade)`` key to its ``(node, row)``
    pairs for one rotation component. The key carrying the most energy is used (for
    a cyclic mode, its dominant cosine or sine coordinate; without MBC, its dominant
    blade). The complex profile is turned to its principal real direction, the phase
    that maximises the real part's energy, so a lightly damped (nearly real) shape
    keeps its node pattern. Returns ``None`` when fewer than three nodes are present.
    """
    best: npt.NDArray[np.complex128] | None = None
    best_energy = 0.0
    for members in rows.values():
        if len(members) < 3:
            continue
        ordered = [row for _, row in sorted(members)]
        values = np.asarray(shape[ordered], dtype=np.complex128)
        energy = float(np.sum(np.abs(values) ** 2))
        if energy > best_energy:
            best, best_energy = values, energy
    if best is None:
        return None
    phase = 0.5 * float(np.angle(np.sum(best**2)))
    profile: npt.NDArray[np.float64] = np.real(best * np.exp(-1j * phase))
    return profile


def _is_rigid_profile(profile: npt.NDArray[np.float64]) -> bool:
    """Return whether a section-rotation profile barely varies along the span."""
    peak = float(np.max(np.abs(profile)))
    if peak <= 0.0:
        return False
    spread = float(np.max(profile) - np.min(profile))
    return spread < BEAMDYN_RIGID_RANGE * peak


def _shape_order(profile: npt.NDArray[np.float64]) -> int:
    """Return one plus the sign changes of the spanwise rotation increments.

    The increment between neighbouring nodes is the section rotation accumulated
    over that span segment, so its sign is the sign of the mean bending (or
    twisting) curvature there. A cantilevered blade's ``n``-th mode changes the
    curvature sign ``n - 1`` times; the increments are also blind to the rigid root
    rotation every node carries. Increments below :data:`BEAMDYN_SHAPE_TOLERANCE`
    of the largest are skipped.
    """
    increments = np.diff(profile)
    largest = float(np.max(np.abs(increments))) if increments.size else 0.0
    if largest <= 0.0:
        return 1
    signs = np.sign(increments[np.abs(increments) > BEAMDYN_SHAPE_TOLERANCE * largest])
    return 1 + int(np.count_nonzero(signs[1:] != signs[:-1]))


def _beamdyn_rotation_rows(
    solution: ModalSolution,
) -> dict[str, dict[tuple[str, int | None], list[tuple[int, int]]]]:
    """Group the BeamDyn rotation rows of the mode shapes by component and key.

    Returns ``component -> (multi-blade coordinate, blade) -> [(node, row)]`` for the
    displacement-level rotation rows (``"RX"``, ``"RY"``, ``"RZ"``).
    """
    coordinates = solution.dof_mbc_coordinates
    grouped: dict[str, dict[tuple[str, int | None], list[tuple[int, int]]]] = {}
    for row, description in enumerate(solution.dof_descriptions):
        info = classify_dof(description)
        if info.node is None or info.component is None or info.is_velocity:
            continue
        if not info.component.startswith("R"):
            continue
        coordinate = coordinates[row] if row < len(coordinates) else ""
        grouped.setdefault(info.component, {}).setdefault(
            (coordinate, info.blade), []
        ).append((info.node, row))
    return grouped


def _beamdyn_drivetrain_modes(
    solution: ModalSolution,
    labels: list[ModeLabel],
    edge_rows: dict[tuple[str, int | None], list[tuple[int, int]]],
) -> list[ModeLabel]:
    """Return the labels with BeamDyn drivetrain-torsion modes named as such.

    Only a model with BeamDyn blades and a flexible drivetrain (an ElastoDyn
    ``DOF_DrTr`` state) is affected. With BeamDyn the rotor's rotation sits in the
    blades' nodal states, so the drivetrain-torsion mode does not vote for the
    drivetrain DOF:

    * with the generator free, the generator swings against the rotor and the mode
      votes ``GENERATOR_AZIMUTH``; it is the drivetrain torsion when the blades'
      collective edgewise rotation is elastic (the rigid-body rotor-azimuth mode
      rotates them rigidly);
    * with the generator locked, the whole rotor swings against the shaft spring
      and the mode votes collective edgewise; it is the drivetrain torsion when the
      root section rotates in phase with the peak rotation by at least
      :data:`DRIVETRAIN_ROOT_SHARE` of it (an edgewise bending mode keeps the hub
      still or counter-rotates it).
    """
    categories = {classify_dof(d).category for d in solution.dof_descriptions}
    if DofCategory.DRIVETRAIN_TORSION not in categories or not edge_rows:
        return labels
    generator_free = DofCategory.GENERATOR_AZIMUTH in categories
    result = list(labels)
    for k, label in enumerate(labels):
        votes_generator = label.category is DofCategory.GENERATOR_AZIMUTH
        votes_edge = (
            not generator_free
            and label.category is DofCategory.BLADE_EDGE
            and label.multiblade == "collective"
        )
        if not (votes_generator or votes_edge):
            continue
        profile = _beamdyn_rotation_profile(solution.mode_shapes[:, k], edge_rows)
        if profile is None or _is_rigid_profile(profile):
            continue
        if votes_edge and not _root_in_phase(profile):
            continue
        result[k] = replace(
            label,
            category=DofCategory.DRIVETRAIN_TORSION,
            label=_LABELS[DofCategory.DRIVETRAIN_TORSION],
            multiblade=None,
        )
    return result


def _root_in_phase(profile: npt.NDArray[np.float64]) -> bool:
    """Return whether the root rotation follows the peak by DRIVETRAIN_ROOT_SHARE."""
    peak = float(profile[int(np.argmax(np.abs(profile)))])
    root = float(profile[0])
    return peak != 0.0 and root / peak >= DRIVETRAIN_ROOT_SHARE


def _number_blade_modes(
    solution: ModalSolution,
    labels: list[ModeLabel],
    infos: list[_MultibladeInfo],
    heavy: npt.NDArray[np.bool_],
) -> list[ModeLabel]:
    """Return the labels with BeamDyn flap, edge, and torsion modes numbered.

    A BeamDyn mode is first checked for rigid blade motion (see
    :data:`BEAMDYN_RIGID_RANGE`): a rigid collective edgewise mode is the rotor-azimuth
    mode and is renamed ``"Rotor/Generator"``; any other rigid blade mode keeps its
    unnumbered family label. Each remaining mode gets a lower bound on its order from
    its spanwise shape (:func:`_shape_order`). Within a family, the modes of one
    multi-blade type (collective, regressive, progressive) are then numbered in order
    of frequency, each at least its shape order and above the previous mode's, so a
    regressive/collective/progressive triplet of one bending mode shares a number and
    a blade mode missed below (e.g. an aerodynamically swamped one) does not shift
    the numbers of the modes above it. Modes without a whirl type (standing cyclic
    or degenerate modes, or a model without MBC) are numbered the same way but share
    each number among up to two modes (three without MBC), one per cyclic coordinate
    (per blade). Before numbering, a shape-order family whose members break the
    one-mode-per-type rule has its types reassigned (see
    :func:`_beamdyn_whirl_types`); a reassigned mode is marked ambiguous with its
    previous type as the alternative.
    """
    frequencies = np.asarray(solution.natural_frequencies_hz, dtype=np.float64)
    shapes = solution.mode_shapes
    if frequencies.shape != (len(labels),) or shapes.ndim != 2:
        return labels
    rows = _beamdyn_rotation_rows(solution)
    labels = _beamdyn_drivetrain_modes(solution, labels, rows.get("RX", {}))
    targets = [
        k
        for k, label in enumerate(labels)
        if label.category in _BEAMDYN_FAMILY_ROTATION
    ]
    if not targets:
        return labels
    result = list(labels)
    bounds: dict[int, int] = {}
    for k in targets:
        label = labels[k]
        component = _BEAMDYN_FAMILY_ROTATION[label.category]
        profile = _beamdyn_rotation_profile(shapes[:, k], rows.get(component, {}))
        if profile is None:
            bounds[k] = 1
        elif _is_rigid_profile(profile):
            if label.category is DofCategory.BLADE_EDGE and (
                label.multiblade == "collective"
            ):
                result[k] = replace(
                    label,
                    category=DofCategory.GENERATOR_AZIMUTH,
                    label=_LABELS[DofCategory.GENERATOR_AZIMUTH],
                    multiblade=None,
                )
        else:
            bounds[k] = _shape_order(profile)
    retyped = _beamdyn_whirl_types(labels, bounds, infos, heavy, frequencies)
    kinds = {k: retyped.get(k, labels[k].multiblade) for k in bounds}
    capacity = 2 if any(solution.dof_mbc_coordinates) else 3
    groups: dict[tuple[DofCategory, str], list[int]] = {}
    for k in bounds:
        kind = kinds[k]
        whirl = kind if kind in _WHIRL_TYPES else "standing"
        groups.setdefault((labels[k].category, whirl), []).append(k)
    for (category, whirl), members in groups.items():
        number, used = 0, 0
        for k in sorted(members, key=lambda index: float(frequencies[index])):
            bound = bounds[k]
            if whirl != "standing" or number == 0 or bound > number or used >= capacity:
                number, used = max(bound, number + 1), 1
            else:
                used += 1
            family = f"{_ordinal(number)} {_LABELS[category]}"
            kind = kinds[k]
            text = family if kind is None else f"{family} ({kind.capitalize()})"
            if k not in retyped:
                result[k] = replace(labels[k], label=text)
                continue
            previous = labels[k].multiblade or ""
            result[k] = replace(
                labels[k],
                label=text,
                multiblade=kind,
                is_ambiguous=True,
                alternative=f"{family} ({previous.capitalize()})",
                alternative_category=category,
                alternative_confidence=labels[k].confidence,
            )
    return result


def _beamdyn_whirl_types(
    labels: list[ModeLabel],
    bounds: dict[int, int],
    infos: list[_MultibladeInfo],
    heavy: npt.NDArray[np.bool_],
    frequencies: npt.NDArray[np.float64],
) -> dict[int, str]:
    """Return the BeamDyn modes whose multi-blade type changes before numbering.

    A BeamDyn mode's number is counted within its multi-blade type, so a member that
    carries the wrong type (near-coincident members that mix, or a weak whirl) would
    be counted in another type's sequence and shift the numbers of every mode above
    it. The family of a mode is not known before numbering, so it is taken from the
    shape order: the lightly damped modes of one category and shape order (see
    :func:`_shape_order`) are split into ``ceil(n / 3)`` families at their largest
    frequency gaps, and a family that breaks the one-mode-per-type rule is assigned
    its types jointly, as :func:`_resolve_whirl_slots` does for a numbered family.
    Members the assignment leaves without a type keep theirs, so a mode of a higher
    family whose shape order is underestimated still numbers in its own sequence.
    """
    groups: dict[tuple[DofCategory, int], list[int]] = {}
    for k, bound in bounds.items():
        if labels[k].multiblade in (*_WHIRL_TYPES, "cyclic") and not bool(heavy[k]):
            groups.setdefault((labels[k].category, bound), []).append(k)
    retyped: dict[int, str] = {}
    for members in groups.values():
        for family in _frequency_families(members, frequencies):
            if _family_is_consistent(family, labels, frequencies):
                continue
            assignment = _best_whirl_assignment(family, labels, infos, frequencies)
            retyped.update(
                (k, kind)
                for k, kind in assignment.items()
                if kind != labels[k].multiblade
            )
    return retyped


def _frequency_families(
    members: list[int], frequencies: npt.NDArray[np.float64]
) -> list[list[int]]:
    """Split modes into ``ceil(n / 3)`` frequency-ordered groups at the widest gaps."""
    ordered = sorted(members, key=lambda k: float(frequencies[k]))
    n_families = -(-len(ordered) // len(_WHIRL_TYPES))
    gaps = np.diff(frequencies[ordered]) if len(ordered) > 1 else np.empty(0)
    # The widest gaps separate the families (stable order breaks ties by position).
    cuts = sorted(
        np.argsort(-gaps, kind="stable")[: n_families - 1].tolist(), reverse=True
    )
    families = [ordered]
    for cut in cuts:
        head = families.pop(0)
        families[:0] = [head[: cut + 1], head[cut + 1 :]]
    return families


def label_modes(
    participation: ParticipationResult,
    descriptions: Sequence[str],
    *,
    min_known_fraction: float | None = None,
) -> list[ModeLabel]:
    """Label every mode of a participation result.

    Parameters
    ----------
    participation : ParticipationResult
        Per-DOF participation (its ``magnitude`` columns are one mode each).
    descriptions : Sequence[str]
        DOF descriptions parallel to the participation rows.
    min_known_fraction : float or None, optional
        Forwarded to :func:`label_mode`; ``None`` (the default) keeps the plain
        plurality vote.

    Returns
    -------
    list[ModeLabel]
        One label per mode (column).
    """
    return [
        label_mode(
            participation.magnitude[:, col],
            descriptions,
            min_known_fraction=min_known_fraction,
        )
        for col in range(participation.magnitude.shape[1])
    ]


def label_solution(
    solution: ModalSolution,
    *,
    min_known_participation: float = DEFAULT_MIN_KNOWN_PARTICIPATION,
    rotor_speed_rpm: float | None = None,
) -> list[ModeLabel]:
    r"""Label every mode of a modal solution by its biorthogonal participation.

    For mode :math:`k` with right and left eigenvectors :math:`\mathbf{x}_k` and
    :math:`\mathbf{y}_k` of the full state matrix, the participation of state
    :math:`i` is :math:`p_{ik} \propto |\overline{y}_{ik} x_{ik}|`, normalised so
    that :math:`\sum_i p_{ik} = 1` (see
    :func:`vane.modal.biorthogonal.biorthogonal_participation`). The displacement
    and velocity states of a second-order DOF are summed onto that DOF, and the
    shares are voted per :class:`DofCategory` by :func:`label_mode` with
    ``min_known_fraction=min_known_participation``: the largest classified category
    names the mode if the classified DOFs carry at least ``min_known_participation``
    of it, otherwise the mode is ``"Unidentified"``. The confidence is the dominant
    category's share of the mode's participation.

    Platform translation/tilt and tower bending of one plane vote together when
    tower bending carries at least :data:`TOWER_SUPPORT_SHARE` of that plane's
    participation, so the global tower modes of a fixed-bottom support (whose
    substructure flexibility sits in the platform DOFs) are named tower modes, not
    platform modes. The tower modes of each plane are then numbered in order of
    frequency (``"3rd Tower Fore-Aft"`` for the third), never below the number of
    their dominant tower DOF.

    A non-degenerate rotor mode carries its multi-blade type (collective,
    regressive, progressive, or a standing cyclic pattern), decided from the
    collective/cyclic split of its participation and the whirl of its dominant
    cosine/sine pair.

    A BeamDyn blade mode (one voted :attr:`~DofCategory.BLADE_FLAP`,
    :attr:`~DofCategory.BLADE_EDGE`, or :attr:`~DofCategory.BLADE_TORSION`) is then
    numbered, e.g. ``"2nd Blade Flap (Collective)"``. Its shape order is one plus
    the number of sign changes of the spanwise increments of its section rotation
    (about ``Y`` for flap, ``X`` for edge, ``Z`` for torsion), which is blind to
    the rigid root rotation the nodal states carry. Within a family the modes of one
    multi-blade type are numbered in frequency order, each at least its shape order.
    A mode whose section rotation barely varies along the span
    (:data:`BEAMDYN_RIGID_RANGE`) is rigid blade motion, not bending: a rigid
    collective edgewise mode is the rotor-azimuth mode (``"Rotor/Generator"``), and
    any other rigid mode keeps its unnumbered family label.

    Parameters
    ----------
    solution : ModalSolution
        A solution whose ``dof_descriptions`` are populated (i.e. produced with
        descriptions, as by :func:`vane.modal.eigensolver.modes_from_mbc`).
    min_known_participation : float, optional
        Minimum share of the participation, in ``[0, 1]``, that classified DOFs
        must carry for a mode to receive a physical label (default
        :data:`DEFAULT_MIN_KNOWN_PARTICIPATION`, ``0.5``: a majority).
    rotor_speed_rpm : float or None, optional
        Rotor speed (rev/min) of the operating point, used to read a whirl mode's
        damping in the rotating frame so heavily damped modes stay out of the
        per-point multi-blade assignment. Default ``None`` uses
        ``solution.rotor_speed_rpm`` (set by
        :func:`vane.modal.eigensolver.modes_from_mbc`).

    Returns
    -------
    list[ModeLabel]
        One label per mode.

    Raises
    ------
    ValueError
        If the solution carries no DOF descriptions, or ``min_known_participation``
        is outside ``[0, 1]`` or non-finite.
    TypeError
        If ``min_known_participation`` is not a real number.

    Notes
    -----
    A solution without usable left eigenvectors (e.g. constructed by hand), or an
    individual mode whose left--right product is numerically unresolvable, falls
    back to the plurality vote over raw mode-shape magnitudes. The multi-blade type
    is decided from the mode's MBC coordinates -- the collective/cyclic split from
    its participation (or, without it, from the blade energy of the mode shape) and
    the whirl from the shape's cosine/sine phase; on the participation path it is
    attached only to a blade or ``"Unidentified"`` mode, never to a mode the vote
    assigns to the tower, drivetrain, or platform.

    **Ambiguity.** A label whose runner-up category carries at least
    :data:`LABEL_AMBIGUITY_RATIO` of the winner's weight is returned with
    ``is_ambiguous=True`` and the runner-up as ``alternative``; so is a blade mode
    whose collective and cyclic shares are within that ratio, or whose whirl is a
    standing pattern (``"cyclic"``).

    **One mode per multi-blade type.** Each blade mode family (e.g. the second flap)
    of a three-bladed rotor has one collective, one regressive and one progressive
    member, the regressive below the progressive. Where near-coincident members mix
    and break this at one point, the family's lightly damped members are assigned
    to the three types jointly (best summed fit subject to the whirl frequency
    order); a reassigned member is marked ambiguous, and a member left over is
    renamed after its non-blade alternative (e.g. a tower or platform mode it is
    mixed with) or keeps its unsuffixed family name, ambiguous either way.
    Heavily damped modes (:data:`DEFAULT_HEAVILY_DAMPED_THRESHOLD`, read in the
    rotating frame for a whirl mode) are left out of the assignment: the
    fixed-frame images of an overdamped blade mode whirl at the rotor speed and
    are not members of a whirl pair.

    **BeamDyn drivetrain.** With a flexible drivetrain, a BeamDyn model's
    drivetrain-torsion mode votes for the generator azimuth (generator free) or for
    collective edgewise motion (generator locked); it is renamed
    ``"Drivetrain Torsion"`` from its blade-root rotation (see
    :data:`DRIVETRAIN_ROOT_SHARE`).

    The tower modes of each plane are numbered after the multi-blade assignment, so
    a tower/blade hybrid that loses its blade type to a purer family member (and is
    renamed after its tower alternative) is counted among the tower modes.
    """
    if not solution.dof_descriptions:
        msg = "ModalSolution has no dof_descriptions; cannot label modes"
        raise ValueError(msg)
    threshold = _validate_fraction(min_known_participation)
    weights = _participation_weights(solution)
    mbc_coordinates = solution.dof_mbc_coordinates
    # A degenerate mode's eigenvector is an arbitrary rotation within its subspace,
    # so its multi-blade (whirl) type is not defined; it gets no suffix.
    degenerate = np.zeros(solution.mode_shapes.shape[1], dtype=np.bool_)
    if np.shape(solution.is_degenerate) == degenerate.shape:
        degenerate = np.asarray(solution.is_degenerate, dtype=np.bool_)
    columns = [
        _label_column(
            solution,
            weights,
            col,
            threshold,
            mbc_coordinates if not bool(degenerate[col]) else [],
        )
        for col in range(solution.mode_shapes.shape[1])
    ]
    labels = [label for label, _ in columns]
    infos = [info for _, info in columns]
    speed = solution.rotor_speed_rpm if rotor_speed_rpm is None else rotor_speed_rpm
    # Numbering only renames collective or untyped modes (rotor, drivetrain), whose
    # damping is read in the fixed frame either way, so it does not change this.
    heavy = _heavily_damped(solution, labels, speed)
    labels = _number_blade_modes(solution, labels, infos, heavy)
    labels = _resolve_whirl_slots(solution, labels, infos, heavy)
    return _number_tower_modes(solution, labels)


def _heavily_damped(
    solution: ModalSolution, labels: list[ModeLabel], rotor_speed_rpm: float
) -> npt.NDArray[np.bool_]:
    """Flag the heavily damped modes of one point (rotating frame for whirl modes).

    Without a finite rotor speed a regressive/progressive mode's damping cannot be
    read in its own frame, and it is never flagged.
    """
    multiblade = [label.multiblade for label in labels]
    ratios = np.asarray(solution.damping_ratios, dtype=np.float64)
    if ratios.shape != (len(labels),):
        return np.zeros(len(labels), dtype=np.bool_)
    if not math.isfinite(rotor_speed_rpm):
        whirl = np.array([m in _WHIRL_SHIFT for m in multiblade], dtype=np.bool_)
        return np.asarray(
            (ratios >= DEFAULT_HEAVILY_DAMPED_THRESHOLD) & ~whirl, dtype=np.bool_
        )
    rotating = _rotating_frame_damping(
        np.asarray(solution.eigenvalues, dtype=np.complex128),
        ratios,
        rotor_speed_rpm,
        multiblade,
    )
    return np.asarray(rotating >= DEFAULT_HEAVILY_DAMPED_THRESHOLD, dtype=np.bool_)


_WHIRL_SHIFT = {"progressive": (-1.0,), "regressive": (1.0,), "cyclic": (-1.0, 1.0)}


def _rotating_frame_damping(
    eigenvalues: npt.NDArray[np.complex128],
    damping_ratios: npt.NDArray[np.float64],
    rotor_speed_rpm: float,
    multiblade: Sequence[str | None],
) -> npt.NDArray[np.float64]:
    """Return each mode's damping ratio in its own frame.

    See :func:`vane.modal.identifier.rotor_frame_damping_ratios`: a regressive or
    progressive mode's fixed-frame eigenvalue is shifted back by the rotor speed
    (both shifts for a standing ``"cyclic"`` mode, keeping the smaller ratio);
    every other mode keeps its fixed-frame damping ratio.
    """
    ratios: npt.NDArray[np.float64] = np.array(damping_ratios, dtype=np.float64)
    omega = abs(rotor_speed_rpm) * math.pi / 30.0
    for mode, whirl in enumerate(multiblade):
        shifts = _WHIRL_SHIFT.get(whirl or "")
        if shifts is None:
            continue
        sigma = -float(eigenvalues[mode].real)
        damped = abs(float(eigenvalues[mode].imag))
        moduli = [math.hypot(sigma, damped + shift * omega) for shift in shifts]
        ratios[mode] = min(
            sigma / modulus if modulus > 0.0 else 0.0 for modulus in moduli
        )
    return ratios


def _family_name(label: ModeLabel) -> str:
    """Return a blade label without its multi-blade suffix (e.g. ``2nd Blade Flap``)."""
    if label.multiblade is None:
        return label.label
    suffix = f" ({label.multiblade.capitalize()})"
    return label.label.removesuffix(suffix)


def _resolve_whirl_slots(
    solution: ModalSolution,
    labels: list[ModeLabel],
    infos: list[_MultibladeInfo],
    heavy: npt.NDArray[np.bool_],
) -> list[ModeLabel]:
    """Give each blade mode family one collective, regressive and progressive mode.

    A rotating-frame blade mode of an N >= 3 bladed rotor appears in the fixed frame
    as exactly one collective and one regressive/progressive whirl pair, the
    regressive below the progressive. Wherever one point's labels break this -- two
    modes claiming a type, a standing ``"cyclic"`` mode, or a regressive member
    above the progressive one (typical of near-coincident modes that mix) -- the
    family's lightly damped, non-degenerate members are assigned to the three types
    jointly. The assignment maximises the summed fit, each member scoring its family
    share (``confidence``) times its fit to the type (``_MultibladeInfo.score``),
    subject to the frequency order of the whirl pair. A member whose type changes is
    marked ambiguous with its previous label as the alternative. A member left
    without a type (a fourth claimant, or a member that fits no free type) is renamed
    after its alternative category when that is a non-blade close competitor (e.g.
    the tower or platform it is mixed with), and otherwise keeps its family name
    without a type; it is marked ambiguous either way. Families that already satisfy
    the rule are left unchanged, except that an undecided multi-blade split is marked
    ambiguous.
    """
    frequencies = np.asarray(solution.natural_frequencies_hz, dtype=np.float64)
    result = list(labels)
    families: dict[tuple[DofCategory, str], list[int]] = {}
    for k, label in enumerate(labels):
        if (
            label.category in _MULTIBLADE_CATEGORIES
            and label.category is not DofCategory.UNKNOWN
            and label.multiblade in (*_WHIRL_TYPES, "cyclic")
            and not bool(heavy[k])
        ):
            families.setdefault((label.category, _family_name(label)), []).append(k)
    for (_, family), members in families.items():
        if _family_is_consistent(members, labels, frequencies):
            for k in members:
                if infos[k].is_undecided and not result[k].is_ambiguous:
                    result[k] = _mark_undecided(result[k], infos[k], family)
            continue
        assignment = _best_whirl_assignment(members, labels, infos, frequencies)
        for k in members:
            kind = assignment.get(k)
            if kind is None:
                result[k] = _demote(labels[k], family)
            elif kind == labels[k].multiblade:
                if infos[k].is_undecided and not labels[k].is_ambiguous:
                    result[k] = _mark_undecided(labels[k], infos[k], family)
            else:
                result[k] = replace(
                    labels[k],
                    label=f"{family} ({kind.capitalize()})",
                    multiblade=kind,
                    is_ambiguous=True,
                    alternative=labels[k].label,
                    alternative_category=labels[k].category,
                    alternative_confidence=labels[k].confidence,
                )
    return result


def _family_is_consistent(
    members: list[int], labels: list[ModeLabel], frequencies: npt.NDArray[np.float64]
) -> bool:
    """Return whether a family has at most one mode per type and ordered whirls."""
    kinds = [labels[k].multiblade for k in members]
    if "cyclic" in kinds or any(kinds.count(kind) > 1 for kind in _WHIRL_TYPES):
        return False
    by_kind = {labels[k].multiblade: k for k in members}
    regressive, progressive = by_kind.get("regressive"), by_kind.get("progressive")
    if regressive is None or progressive is None:
        return True
    return bool(frequencies[regressive] < frequencies[progressive])


def _best_whirl_assignment(
    members: list[int],
    labels: list[ModeLabel],
    infos: list[_MultibladeInfo],
    frequencies: npt.NDArray[np.float64],
) -> dict[int, str]:
    """Return the best-fitting mode-to-type assignment of one family.

    Every injective assignment of the three multi-blade types to the members (each
    type to at most one member) is scored; a regressive member must lie below the
    progressive one, and a member is assigned only a type it fits by at least
    :data:`_MIN_WHIRL_FIT`. Ties resolve to the first assignment found in member order.
    """
    candidates: list[int | None] = [*members, None, None, None]
    best: dict[int, str] = {}
    best_score = -1.0
    seen: set[tuple[int | None, ...]] = set()
    for chosen in itertools.permutations(candidates, len(_WHIRL_TYPES)):
        if chosen in seen:
            continue
        seen.add(chosen)
        slot = dict(zip(_WHIRL_TYPES, chosen, strict=True))
        regressive, progressive = slot["regressive"], slot["progressive"]
        if (
            regressive is not None
            and progressive is not None
            and not frequencies[regressive] < frequencies[progressive]
        ):
            continue
        # A member is never given a type its shape does not carry (e.g. "collective"
        # for a pure whirl): it is left without a type instead.
        if any(
            not _reaches(infos[k].score(kind), _MIN_WHIRL_FIT, 1.0)
            for kind, k in slot.items()
            if k is not None
        ):
            continue
        score = sum(
            max(labels[k].confidence, 0.0) * infos[k].score(kind)
            for kind, k in slot.items()
            if k is not None
        )
        if score > best_score + 1e-12:
            best_score = score
            best = {k: kind for kind, k in slot.items() if k is not None}
    return best


def _mark_undecided(label: ModeLabel, info: _MultibladeInfo, family: str) -> ModeLabel:
    """Return ``label`` marked ambiguous because its multi-blade type is undecided."""
    if info.kind in ("collective", "cyclic"):
        rival = "regressive" if info.circularity > 0.0 else "progressive"
    else:
        rival = "collective"
    return replace(
        label,
        is_ambiguous=True,
        alternative=f"{family} ({rival.capitalize()})",
        alternative_category=label.category,
        alternative_confidence=label.confidence,
    )


def _demote(label: ModeLabel, family: str) -> ModeLabel:
    """Return the label of a blade mode left without a multi-blade type.

    The mode is renamed after its non-blade alternative only when that alternative
    is a close competitor of the vote (a hybrid, see :data:`LABEL_AMBIGUITY_RATIO`);
    a minor share of another category does not name a blade-dominated mode.
    """
    alternative = label.alternative_category
    mixed = _reaches(
        label.alternative_confidence, LABEL_AMBIGUITY_RATIO, label.confidence
    )
    if (
        alternative is not None
        and alternative not in _MULTIBLADE_CATEGORIES
        and label.confidence > 0.0
        and mixed
    ):
        return ModeLabel(
            category=alternative,
            label=_LABELS[alternative],
            confidence=label.alternative_confidence,
            dominant_dofs=label.dominant_dofs,
            multiblade=None,
            is_ambiguous=True,
            alternative=label.label,
            alternative_category=label.category,
            alternative_confidence=label.confidence,
        )
    return replace(
        label,
        label=family,
        multiblade=None,
        is_ambiguous=True,
        alternative=label.label,
        alternative_category=label.category,
        alternative_confidence=label.confidence,
    )


def _label_column(
    solution: ModalSolution,
    weights: npt.NDArray[np.float64] | None,
    col: int,
    threshold: float,
    mbc_coordinates: Sequence[str],
) -> tuple[ModeLabel, _MultibladeInfo]:
    """Label one mode by participation, or by raw magnitude when unavailable.

    On the participation path the multi-blade type is attached only to a blade or
    ``"Unidentified"`` mode: rotating module states (e.g. aerodynamic blade-node
    states) carry multi-blade coordinates too, and can give a whirl type to a mode
    the participation vote assigns to the tower, drivetrain, or platform. A blade
    mode always gets its type -- the vote has already found it blade-dominated, even
    when the tower or drivetrain share pulls the blade coordinates below half of the
    participation -- while an ``"Unidentified"`` mode gets one only when the rotor
    coordinates carry at least half of it. Empty ``mbc_coordinates`` (no MBC data,
    or a degenerate mode) give no multi-blade type.
    """
    shape = solution.mode_shapes[:, col]
    if weights is not None and bool(np.all(np.isfinite(weights[:, col]))):
        label = label_mode(
            weights[:, col],
            solution.dof_descriptions,
            min_known_fraction=threshold,
        )
        if label.category not in _MULTIBLADE_CATEGORIES:
            return label, _MultibladeInfo(None)
        min_share = 0.5 if label.category is DofCategory.UNKNOWN else 0.0
        info = _multiblade_info(
            shape, mbc_coordinates, weights[:, col], min_blade_share=min_share
        )
        if info.kind is None:
            return label, info
        return (
            replace(
                label,
                label=f"{label.label} ({info.kind.capitalize()})",
                multiblade=info.kind,
            ),
            info,
        )
    info = _multiblade_info(shape, mbc_coordinates)
    return (
        label_mode(np.abs(shape), solution.dof_descriptions, multiblade=info.kind),
        info,
    )


def _participation_weights(
    solution: ModalSolution,
) -> npt.NDArray[np.float64] | None:
    """Return biorthogonal participation summed onto the mode-shape rows.

    The full states are ordered ``[q2, q2dot, q1]`` while the mode-shape rows are
    ``[q2, q1]``; the velocity participation of each second-order DOF is added to
    its displacement row. Returns ``None`` when left eigenvectors are unavailable
    or the eigenvector layout does not match the mode shapes; the column of a mode
    whose participation is unresolvable is ``NaN``.
    """
    right = solution.full_eigenvectors
    left = solution.full_left_eigenvectors
    n_rows, n_modes = solution.mode_shapes.shape
    if (
        left.size == 0
        or left.ndim != 2
        or right.ndim != 2
        or left.shape != right.shape
        or right.shape[1] != n_modes
    ):
        return None
    n_states = right.shape[0]
    ndof2 = n_states - n_rows
    ndof1 = n_rows - ndof2
    if ndof2 < 0 or ndof1 < 0 or len(solution.dof_descriptions) != n_rows:
        return None
    # Only the per-state fractions are needed, so every state is put in a single
    # placeholder subsystem; this avoids requiring the full state descriptions.
    fractions = biorthogonal_participation(
        solution, subsystem_labels=("state",) * n_states
    ).magnitude_fractions
    fractions = _with_degenerate_cluster_fractions(solution, fractions)
    weights: npt.NDArray[np.float64] = np.concatenate(
        (fractions[:ndof2] + fractions[ndof2 : 2 * ndof2], fractions[2 * ndof2 :])
    )
    return weights


def _degenerate_clusters(
    eigenvalues: npt.NDArray[np.complex128], flags: npt.NDArray[np.bool_]
) -> list[list[int]]:
    """Group flagged modes into connected components of coinciding eigenvalues.

    Two modes are linked when their eigenvalues coincide within the eigensolver's
    degeneracy tolerance; clusters are the transitive closure, so a tolerance chain
    (A near B, B near C) forms one cluster even if A and C are further apart.
    """
    members = np.flatnonzero(flags).tolist()
    parent = {index: index for index in members}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for position, first in enumerate(members):
        for second in members[position + 1 :]:
            scale = max(abs(eigenvalues[first]), abs(eigenvalues[second]), 1e-30)
            if abs(eigenvalues[first] - eigenvalues[second]) <= DEGENERACY_RTOL * scale:
                parent[find(second)] = find(first)
    groups: dict[int, list[int]] = {}
    for index in members:
        groups.setdefault(find(index), []).append(index)
    return [cluster for cluster in groups.values() if len(cluster) > 1]


def _with_degenerate_cluster_fractions(
    solution: ModalSolution, fractions: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    r"""Replace each degenerate mode's participation by its subspace's participation.

    Within a (near-)repeated eigenvalue the individual eigenvectors are an arbitrary
    basis of the shared invariant subspace, so their per-state factors are not
    meaningful. The spectral projector onto the subspace,
    :math:`P = X_c (Y_c^H X_c)^{-1} Y_c^H`, is basis-invariant; the normalized
    magnitudes of its diagonal give the subspace's state participation, and every
    mode of the cluster is labelled from it (so a symmetric pair shares one label
    whose confidence reflects the combined participation).
    """
    flags = np.asarray(solution.is_degenerate, dtype=np.bool_)
    if flags.shape != (fractions.shape[1],) or not np.any(flags):
        return fractions
    right = np.asarray(solution.full_eigenvectors, dtype=np.complex128)
    left = np.asarray(solution.full_left_eigenvectors, dtype=np.complex128)
    eigenvalues = np.asarray(solution.eigenvalues, dtype=np.complex128)
    result = np.array(fractions, dtype=np.float64, copy=True)
    for cluster in _degenerate_clusters(eigenvalues, flags):
        x_c = right[:, cluster]
        y_c = left[:, cluster]
        gram = np.conj(y_c).T @ x_c
        try:
            coupling = np.linalg.solve(gram, np.conj(y_c).T)
        except np.linalg.LinAlgError:
            result[:, cluster] = np.nan  # unresolvable: fall back per mode
            continue
        diagonal = np.abs(np.einsum("ia,ai->i", x_c, coupling))
        total = float(np.sum(diagonal))
        if not (np.isfinite(total) and total > 0.0):
            result[:, cluster] = np.nan
            continue
        result[:, cluster] = (diagonal / total)[:, np.newaxis]
    return result
