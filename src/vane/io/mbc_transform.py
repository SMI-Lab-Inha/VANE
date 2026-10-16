# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Multi-blade coordinate (MBC3 / Coleman) transformation.

OpenFAST writes linearized state matrices in a mixed frame in which the blade
degrees of freedom rotate with the rotor, making the system matrices
azimuth-dependent. The MBC3 transform projects the three per-blade rotating
coordinates onto non-rotating multi-blade coordinates (collective, cosine, and
sine), and averaging the transformed matrices over one revolution yields the
azimuth-invariant linear time-invariant system used for modal analysis.

This module implements the transform for three-bladed rotors directly from the
parsed :class:`~vane.io.lin_reader.LinFile` objects of a single operating point
(one file per azimuth). Continuous states are reordered into the canonical
``[q2, q2dot, q1]`` layout (second-order displacements, their velocities, then
first-order states) and blade triplets are detected from the channel
descriptions. The transformed ``A``, ``B``, ``C``, and ``D`` matrices are
averaged over azimuth.

Notes
-----
At zero rotor speed the transform reduces to a similarity transform of the state
matrix and therefore preserves its eigenvalues; this property is used to verify
the implementation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import block_diag as _scipy_block_diag

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from vane.io.lin_reader import LinFile, OperatingPointTable

__all__ = [
    "ROTOR_SPEED_DRIFT_RTOL",
    "MBCResult",
    "find_blade_triplets",
    "mbc3_transform",
]

logger = logging.getLogger(__name__)

_N_BLADES = 3
_DERIV_MARKER = "First time derivative of"
_DEG_PER_RAD = 180.0 / np.pi
_RPM_PER_RAD_S = 30.0 / np.pi


def _has_rate_unit(description: str) -> bool:
    """Return whether a state description's trailing unit is a rate (``.../s``)."""
    return description.rsplit(",", 1)[-1].strip().endswith("/s")


# A MoorDyn position/velocity component, e.g. the ``Px``/``Vx`` in
# ``MD Line 1 node 1 Px``; both map to the same partner key (``...x``).
_POS_VEL_COMPONENT = re.compile(r"[PV]([XYZxyz])\b")
# A whitespace-separated trailing unit on a description written without the usual
# ``", unit"`` separator, e.g. ServoDyn StC ``... local displacement state dX/dt  m/s``.
# Only recognised physical units are stripped, so a trailing channel name (HydroDyn
# ``RdtnPtfmS``, MoorDyn ``Px``) is never mistaken for a unit.
_BARE_TRAILING_UNIT = re.compile(
    r"\s+\(?(?:m|rad|deg|N|kN|N-m|kN-m|Nm|kg|-)\)?(?:/s)*$"
)
# A Leibniz time-derivative name, e.g. the ``dX/dt`` of a ServoDyn StC velocity state,
# which pairs with the position state named ``X``.
_LEIBNIZ_DERIVATIVE = re.compile(r"\bd(\w+)/dt\b")
_DERIV_MARKER_RE = re.compile(re.escape(_DERIV_MARKER) + r"\s*")


def _pair_key(description: str) -> str:
    """Return a key shared by a displacement and its velocity partner.

    The trailing unit is dropped (whether written after a comma or, as ServoDyn
    structural-control states do, after whitespace only), a position/velocity
    component letter (MoorDyn ``Px``/``Vx``) is collapsed, and a Leibniz derivative
    name ``dX/dt`` is reduced to ``X``, so ``MD Line 1 node 1 Px`` and ``... Vx`` (or
    StC ``... state X  m`` and ``... state dX/dt  m/s``) share a key. Used to
    confirm that a rate-unit state is the time derivative of an actual position
    state rather than an unrelated first-order rate (e.g. AeroDyn induced velocity
    ``vind``, which has no position partner).
    """
    if "," in description:
        body = description.rsplit(",", 1)[0]
    else:
        body = _BARE_TRAILING_UNIT.sub("", description)
    body = _LEIBNIZ_DERIVATIVE.sub(r"\1", body)
    return _POS_VEL_COMPONENT.sub(r"\1", body)


def _marker_partner_key(description: str) -> str:
    """Return the pairing key of the displacement a marker velocity differentiates.

    The ``First time derivative of`` marker is removed (it follows the module
    prefix, e.g. ``ED First time derivative of Platform ...``) and the key is
    case-folded, because some modules capitalise the displacement differently in the
    two descriptions (SubDyn ``Rigid-body mode 1`` versus ``... of rigid-body mode 1``).
    """
    return _pair_key(_DERIV_MARKER_RE.sub("", description)).casefold()


# Two azimuths closer than this (after wrapping to [0, 360) deg) are duplicates.
_AZIMUTH_DUPLICATE_TOL_DEG = 1e-3
# A single operating point must hold rotor speed roughly constant across azimuth. The
# default is tight (1%) so gyroscopic/centrifugal effects and damping are not smeared
# across a too-broad "operating point"; mbc3_transform exposes an override.
ROTOR_SPEED_DRIFT_RTOL = 0.01
_ROTOR_SPEED_DRIFT_RTOL = ROTOR_SPEED_DRIFT_RTOL
_ROTOR_SPEED_DRIFT_ATOL = 1e-6
_WIND_SPEED_DRIFT_RTOL = 0.01
_WIND_SPEED_DRIFT_ATOL = 1e-6

# Azimuth-coverage check for a *rotating* MBC average (warned, not enforced). Below this
# rotor speed the rotor is effectively at standstill, where the transform reduces to a
# similarity transform and a single azimuth is valid. A rotating, anisotropic rotor
# needs a near-uniform full-revolution sweep to average out the periodic content; three
# uniform azimuths is the minimum for a three-blade rotor; 36 is the OpenFAST norm.
_ROTATING_RPM_TOL = 1e-3
_MIN_AZIMUTHS_FOR_MBC = 3
_NONUNIFORM_GAP_FACTOR = 1.5

# Patterns for locating blade-index tokens in a channel description; the blade
# digit is the last character of each match. These specific patterns are matched
# before falling back to a single digit, so node/span numbers are not mistaken
# for the blade index.
_BLADE_TOKEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"[Bb]lade \d"),
    re.compile(r"[Bb]lade [Rr]oot \d"),
    re.compile(r"BD_\d"),
    re.compile(r"[Bb]\d"),
    re.compile(r"[Bb]lade\d"),
    re.compile(r"PitchBearing\d"),
)
_ANY_DIGIT = re.compile(r"\d")

Triplet = tuple[int, ...]


@dataclass
class MBCResult:
    """Result of an MBC3 transform over one operating point's azimuth sweep.

    Parameters
    ----------
    avg_a, avg_b, avg_c, avg_d : npt.NDArray[np.float64] or None
        Azimuth-averaged state-space matrices in the non-rotating frame. Each is
        ``None`` when the corresponding input matrix was absent. When the
        transform is performed, rows/columns follow the reordered
        ``[q2, q2dot, q1]`` state layout; otherwise the original order is kept.
    state_descriptions : list[str]
        State descriptions in the same order as the rows/columns of ``avg_a``.
    ndof2 : int
        Number of second-order degrees of freedom.
    ndof1 : int
        Number of first-order states.
    n_blades : int
        Number of blades inferred from the rotating triplets (3, or 0 when no
        rotating triplets were found).
    performed_transformation : bool
        Whether the MBC3 transform was applied. ``False`` when no three-blade
        triplets were detected, in which case the matrices are plain azimuth
        averages in the original state order.
    rotor_speed_rpm : float
        Mean rotor speed over the azimuth sweep, in rev/min.
    wind_speed : float
        Mean wind speed over the azimuth sweep, in m/s.
    azimuths_deg : npt.NDArray[np.float64]
        Sorted rotor azimuths used, in degrees.
    mbc_coordinates : list[str]
        Multi-blade coordinate of each state, aligned with ``state_descriptions``:
        ``"collective"``, ``"cosine"``, or ``"sine"`` for a transformed blade-triplet
        state, and ``""`` for non-rotating states or when the transform was not
        performed.
    output_descriptions : list[str]
        Output channel descriptions aligned with the rows of ``avg_c``/``avg_d``
        (in the transformed output order). Empty when no output matrix is present.
    input_descriptions : list[str]
        Input channel descriptions aligned with the columns of ``avg_b``/``avg_d``
        (in the transformed input order). Empty when no input matrix is present.
    output_mbc_coordinates : list[str]
        Multi-blade coordinate of each output (``"collective"``/``"cosine"``/
        ``"sine"`` for a transformed blade-triplet output, else ``""``), aligned
        with ``output_descriptions``. A transformed output row is a multi-blade
        combination, not the per-blade channel its description names.
    input_mbc_coordinates : list[str]
        Multi-blade coordinate of each input, aligned with ``input_descriptions``.
    output_operating_point : npt.NDArray[np.float64] or None
        Azimuth-averaged output operating point in the transformed output order. This
        is the baseline that must be subtracted from absolute measurements before a
        perturbation-state estimator is run.
    input_operating_point : npt.NDArray[np.float64] or None
        Azimuth-averaged input operating point in the transformed input order.
    per_azimuth_a, per_azimuth_b, per_azimuth_c, per_azimuth_d : list of array or None
        The transformed (non-rotating) state-space matrices at each azimuth, in the
        same state/input/output order as their averaged counterparts. They are retained
        only when ``retain_per_azimuth=True`` and the corresponding matrix exists. The
        complete ensemble supports azimuth-periodic input/output simulation; the ``A``
        stack also underpins Floquet and azimuth-spread analyses.
    per_azimuth_output_operating_point : list of array or None
        Transformed output operating-point vectors at each azimuth, aligned with the
        output descriptions. These preserve deterministic periodic loads that an
        azimuth average would otherwise erase.
    per_azimuth_input_operating_point : list of array or None
        Transformed input operating-point vectors at each azimuth, aligned with the
        input descriptions. These preserve deterministic periodic controls.
    """

    avg_a: npt.NDArray[np.float64] | None
    avg_b: npt.NDArray[np.float64] | None
    avg_c: npt.NDArray[np.float64] | None
    avg_d: npt.NDArray[np.float64] | None
    state_descriptions: list[str]
    ndof2: int
    ndof1: int
    n_blades: int
    performed_transformation: bool
    rotor_speed_rpm: float
    wind_speed: float
    azimuths_deg: npt.NDArray[np.float64]
    mbc_coordinates: list[str] = field(default_factory=list)
    # per_azimuth_a stays directly after mbc_coordinates to preserve the positional
    # field order of earlier releases; the channel-description fields are appended.
    per_azimuth_a: list[npt.NDArray[np.float64]] | None = None
    output_descriptions: list[str] = field(default_factory=list)
    input_descriptions: list[str] = field(default_factory=list)
    output_mbc_coordinates: list[str] = field(default_factory=list)
    input_mbc_coordinates: list[str] = field(default_factory=list)
    output_operating_point: npt.NDArray[np.float64] | None = None
    input_operating_point: npt.NDArray[np.float64] | None = None
    per_azimuth_b: list[npt.NDArray[np.float64]] | None = None
    per_azimuth_c: list[npt.NDArray[np.float64]] | None = None
    per_azimuth_d: list[npt.NDArray[np.float64]] | None = None
    per_azimuth_output_operating_point: list[npt.NDArray[np.float64]] | None = None
    per_azimuth_input_operating_point: list[npt.NDArray[np.float64]] | None = None


def _sibling_members(
    cleaned: Sequence[str],
    available: Sequence[bool],
    matcher: re.Pattern[str],
    *,
    anchor: int,
    blade: int,
) -> dict[int, int]:
    """Return ``blade number -> channel index`` for the anchor and its blade siblings.

    A sibling is an available rotating channel matching ``matcher`` (the anchor's
    description with all blade digits tied to one captured value), so it differs only by
    a consistent blade number. The first match per blade wins.
    """
    members = {blade: anchor}
    for index, other in enumerate(cleaned):
        if index == anchor or not available[index]:
            continue
        match = matcher.fullmatch(other)
        if match is None:
            continue
        sibling_blade = int(match.group(1))
        if 1 <= sibling_blade <= _N_BLADES:
            members.setdefault(sibling_blade, index)
    return members


def find_blade_triplets(
    descriptions: Sequence[str], rotating_frame: npt.NDArray[np.bool_]
) -> list[Triplet]:
    """Group rotating channels into per-blade triplets by description.

    For each rotating channel, every blade-index token is located (prose
    ``blade 1``, BeamDyn ``BD_1``, blade-coordinate/AeroDyn ``b1``/``B1``,
    ``PitchBearing1``, or a fallback single digit). A sibling matcher is built in
    which all blade digits are tied to one captured value, so a triplet is
    confirmed only when the other rotating channels are identical except for a
    consistent blade number — node, mode, and span indices stay literal and are
    never mistaken for the blade index, even when a channel carries more than one
    blade token. The ``(internal DOF index = ...)`` parenthetical, which holds a
    second blade-numbered token, is removed before matching.

    Parameters
    ----------
    descriptions : Sequence[str]
        Channel descriptions for one channel group.
    rotating_frame : npt.NDArray[np.bool_]
        Per-channel rotating-frame flags, parallel to ``descriptions``.

    Returns
    -------
    list[tuple[int, ...]]
        Triplets of indices into ``descriptions``, each ordered by blade number;
        channels that do not form a complete blade triplet are omitted.
    """
    cleaned = [_strip_dof_index(desc) for desc in descriptions]
    available = [bool(flag) for flag in rotating_frame]
    triplets: list[Triplet] = []
    for i, desc in enumerate(cleaned):
        if not available[i]:
            continue
        spans = _blade_token_spans(desc)
        if not spans:
            continue
        blade = int(desc[spans[0][1] - 1])
        if not 1 <= blade <= _N_BLADES:
            continue
        matcher = _build_sibling_matcher(desc, spans)
        members = _sibling_members(cleaned, available, matcher, anchor=i, blade=blade)
        if set(members) == set(range(1, _N_BLADES + 1)):
            triplet = tuple(members[b] for b in range(1, _N_BLADES + 1))
            triplets.append(triplet)
            for idx in triplet:
                available[idx] = False
    return triplets


def mbc3_transform(
    lin_files: Sequence[LinFile],
    *,
    omega_dot: float = 0.0,
    retain_per_azimuth: bool = False,
    rotor_speed_drift_rtol: float = _ROTOR_SPEED_DRIFT_RTOL,
    wind_speed_drift_rtol: float = _WIND_SPEED_DRIFT_RTOL,
) -> MBCResult:
    """Apply the MBC3 transform and azimuth-average a set of linearizations.

    Parameters
    ----------
    lin_files : Sequence[LinFile]
        Linearization files for a single operating point, one per azimuth. They
        must share the same channel layout; they are sorted by azimuth here.
    omega_dot : float, optional
        Rotor angular acceleration in rad/s^2, assumed zero (appropriate for a
        steady-state periodic operating point).
    retain_per_azimuth : bool, optional
        Keep the transformed ``A``/``B``/``C``/``D`` matrices at each azimuth for
        Floquet, azimuth-spread, and periodic input/output analyses. Off by default to
        save memory.
    rotor_speed_drift_rtol : float, optional
        Relative rotor-speed spread allowed across the azimuth sweep before the files
        are rejected as not one operating point. Defaults to a tight 1%; raise it to
        accept a looser definition of "the same operating point".
    wind_speed_drift_rtol : float, optional
        Relative wind-speed spread allowed across the azimuth sweep. Defaults to 1%
        so wind-dependent Jacobian changes cannot be interpreted as rotor-periodic
        content. Both drift tolerances must be finite and nonnegative.

    Warns
    -----
    For a *rotating* operating point with blade triplets, a single/too-few or
    non-uniform azimuth sweep is logged as a warning: the MBC azimuth average only
    removes the rotor's periodic content when it spans a full revolution
    near-uniformly, so a sparse sweep yields a misleading averaged model. A rotating
    state that forms no blade triplet (e.g. of a one- or two-bladed rotor) is logged
    as a warning too, since it stays in the rotating frame.

    Returns
    -------
    MBCResult
        The azimuth-averaged non-rotating matrices and associated metadata.

    Raises
    ------
    ValueError
        If ``lin_files`` is empty, the files have inconsistent state counts, a file
        is missing the ``A`` matrix, two files share an azimuth, a drift tolerance is
        invalid, rotor/wind speed drifts across the sweep (the files are not a
        single operating point), or a rotating state names a blade numbered above
        three (the transform is the three-blade one).
    """
    if not lin_files:
        msg = "mbc3_transform requires at least one linearization file"
        raise ValueError(msg)

    files = sorted(lin_files, key=lambda lf: lf.azimuth)
    reference = files[0]
    _validate_consistent(files)
    _validate_sweep(files, rotor_speed_drift_rtol, wind_speed_drift_rtol)

    layout = _StateLayout.from_table(reference.x)
    triplets2 = _local_triplets(reference.x, layout.disp2)
    triplets1 = _local_triplets(reference.x, layout.first1)
    inp_triplets = find_blade_triplets(
        reference.u.descriptions, reference.u.rotating_frame
    )
    out_triplets = find_blade_triplets(
        reference.y.descriptions, reference.y.rotating_frame
    )

    n_blades = (
        _N_BLADES if (triplets2 or triplets1 or inp_triplets or out_triplets) else 0
    )
    omega = np.array([lf.rotor_speed for lf in files], dtype=np.float64)
    azimuths_deg = np.array(
        [lf.azimuth * _DEG_PER_RAD for lf in files], dtype=np.float64
    )
    mean_rpm = float(np.mean(omega) * _RPM_PER_RAD_S)
    mean_ws = float(np.mean([lf.wind_speed for lf in files]))
    _check_rotating_states(reference.x, layout, triplets2, triplets1, mean_rpm)
    _reject_extra_blades(reference.u, "inputs")
    _reject_extra_blades(reference.y, "outputs")

    if n_blades != _N_BLADES:
        logger.info("No three-blade triplets found; averaging without MBC transform")
        return _averaged_without_transform(
            files,
            layout,
            mean_rpm,
            mean_ws,
            azimuths_deg,
            retain_per_azimuth=retain_per_azimuth,
        )

    _check_azimuth_coverage(azimuths_deg, mean_rpm)

    plan = _TransformPlan(
        layout=layout,
        seq2=_new_seq(layout.ndof2, triplets2),
        seq1=_new_seq(layout.ndof1, triplets1),
        inp=_new_seq(reference.n_u, inp_triplets),
        out=_new_seq(reference.n_y, out_triplets),
        n_trip2=len(triplets2),
        n_trip1=len(triplets1),
        n_trip_inp=len(inp_triplets),
        n_trip_out=len(out_triplets),
    )

    accum = _Accumulator(reference)
    per_azimuth_a: list[npt.NDArray[np.float64]] | None = (
        [] if retain_per_azimuth else None
    )
    per_azimuth_b: list[npt.NDArray[np.float64]] | None = (
        [] if retain_per_azimuth and reference.b is not None else None
    )
    per_azimuth_c: list[npt.NDArray[np.float64]] | None = (
        [] if retain_per_azimuth and reference.c is not None else None
    )
    per_azimuth_d: list[npt.NDArray[np.float64]] | None = (
        [] if retain_per_azimuth and reference.d is not None else None
    )
    per_azimuth_output_operating_point: list[npt.NDArray[np.float64]] | None = (
        [] if retain_per_azimuth else None
    )
    per_azimuth_input_operating_point: list[npt.NDArray[np.float64]] | None = (
        [] if retain_per_azimuth else None
    )
    for lf, omega_i in zip(files, omega, strict=True):
        transformed = plan.transform_one(lf, omega_i, omega_dot)
        accum.add(transformed)
        if per_azimuth_a is not None and transformed["a"] is not None:
            per_azimuth_a.append(transformed["a"])
        if per_azimuth_b is not None and transformed["b"] is not None:
            per_azimuth_b.append(transformed["b"])
        if per_azimuth_c is not None and transformed["c"] is not None:
            per_azimuth_c.append(transformed["c"])
        if per_azimuth_d is not None and transformed["d"] is not None:
            per_azimuth_d.append(transformed["d"])
        if (
            per_azimuth_output_operating_point is not None
            and transformed["y_op"] is not None
        ):
            per_azimuth_output_operating_point.append(transformed["y_op"])
        if (
            per_azimuth_input_operating_point is not None
            and transformed["u_op"] is not None
        ):
            per_azimuth_input_operating_point.append(transformed["u_op"])
    avg = accum.mean()

    return MBCResult(
        avg_a=avg["a"],
        avg_b=avg["b"],
        avg_c=avg["c"],
        avg_d=avg["d"],
        state_descriptions=[reference.x.descriptions[i] for i in plan.state_perm],
        ndof2=layout.ndof2,
        ndof1=layout.ndof1,
        n_blades=n_blades,
        performed_transformation=True,
        rotor_speed_rpm=mean_rpm,
        wind_speed=mean_ws,
        azimuths_deg=azimuths_deg,
        mbc_coordinates=plan.mbc_coordinates,
        output_descriptions=[
            reference.y.descriptions[i] for i in plan.out.order.tolist()
        ],
        input_descriptions=[
            reference.u.descriptions[i] for i in plan.inp.order.tolist()
        ],
        output_mbc_coordinates=plan.output_mbc_coordinates,
        input_mbc_coordinates=plan.input_mbc_coordinates,
        output_operating_point=avg["y_op"],
        input_operating_point=avg["u_op"],
        per_azimuth_a=per_azimuth_a,
        per_azimuth_b=per_azimuth_b,
        per_azimuth_c=per_azimuth_c,
        per_azimuth_d=per_azimuth_d,
        per_azimuth_output_operating_point=per_azimuth_output_operating_point,
        per_azimuth_input_operating_point=per_azimuth_input_operating_point,
    )


@dataclass
class _StateLayout:
    """Native-index partition of continuous states into q2, q2dot, and q1."""

    disp2: list[int]
    vel2: list[int]
    first1: list[int]

    @property
    def ndof2(self) -> int:
        return len(self.disp2)

    @property
    def ndof1(self) -> int:
        return len(self.first1)

    @classmethod
    def from_table(cls, table: OperatingPointTable) -> _StateLayout:
        """Partition a continuous-state table into q2/q2dot/q1 native indices.

        A second-order velocity (q2dot) state is identified either by ElastoDyn's
        ``First time derivative of`` description marker, or — for modules that name
        their velocity states differently (e.g. MoorDyn ``Vx``/``Vy``/``Vz`` line-node
        velocities) — by a rate unit (``.../s``) on a second-order state *whose
        position partner is present* (so an unrelated first-order rate such as an
        AeroDyn induced velocity ``vind`` is not mistaken for a structural velocity).
        The marker is checked first so it applies even when OpenFAST writes the
        velocity with a derivative order of 1. Displacement and velocity states are
        collected in their native order, so the k-th displacement pairs with the
        k-th velocity (each module lists positions and velocities in the same order).

        Some modules write a derivative order of 0 for a displacement/velocity pair
        (SubDyn rigid-body modes). A marker state whose displacement partner is
        present is therefore a velocity, and that partner a displacement, whatever
        the recorded order; an order-0 marker state *without* a partner stays
        first-order.
        """
        orders = table.derivative_order.tolist()
        descriptions = table.descriptions
        is_marker = [_DERIV_MARKER in desc for desc in descriptions]
        # Case-folded keys of every non-marker state, and the subset of those that a
        # marker velocity differentiates (its displacement partner).
        plain_keys = {
            _pair_key(desc).casefold()
            for desc, marker in zip(descriptions, is_marker, strict=True)
            if not marker
        }
        marker_keys = {
            _marker_partner_key(desc)
            for desc, marker in zip(descriptions, is_marker, strict=True)
            if marker
        }
        # Position-state keys: second-order, non-marker, non-rate states. A rate-unit
        # state is a velocity only if its key matches one of these (it has a position
        # partner whose time derivative it is).
        position_keys = {
            _pair_key(desc)
            for order, desc, marker in zip(orders, descriptions, is_marker, strict=True)
            if order == 2 and not marker and not _has_rate_unit(desc)
        }
        disp2: list[int] = []
        vel2: list[int] = []
        first1: list[int] = []
        for i, (order, desc) in enumerate(zip(orders, descriptions, strict=True)):
            if is_marker[i]:
                has_partner = _marker_partner_key(desc) in plain_keys
                (vel2 if has_partner or order != 0 else first1).append(i)
                continue
            paired_velocity = (
                order == 2 and _has_rate_unit(desc) and _pair_key(desc) in position_keys
            )
            if paired_velocity:
                vel2.append(i)
            elif order == 2 or _pair_key(desc).casefold() in marker_keys:
                disp2.append(i)
            else:
                first1.append(i)
        if len(disp2) != len(vel2):
            msg = (
                f"Second-order displacement/velocity mismatch: {len(disp2)} vs "
                f"{len(vel2)}; cannot form q2/q2dot pairs"
            )
            raise ValueError(msg)
        return cls(disp2=disp2, vel2=vel2, first1=first1)


def _local_triplets(
    table: OperatingPointTable, native_indices: list[int]
) -> list[Triplet]:
    """Find blade triplets within a subset of states, in local index space."""
    descriptions = [table.descriptions[i] for i in native_indices]
    rotating = table.rotating_frame[native_indices]
    return find_blade_triplets(descriptions, rotating)


@dataclass
class _NewSequence:
    """A reordering that places non-rotating channels before blade triplets."""

    order: npt.NDArray[np.int_]
    n_fixed: int


def _new_seq(n_total: int, triplets: list[Triplet]) -> _NewSequence:
    """Build a ``[non-rotating, triplet0, triplet1, ...]`` index ordering."""
    in_triplet = {idx for triplet in triplets for idx in triplet}
    fixed = [i for i in range(n_total) if i not in in_triplet]
    order = fixed + [idx for triplet in triplets for idx in triplet]
    return _NewSequence(order=np.array(order, dtype=np.int_), n_fixed=len(fixed))


# The three transformed states of each blade triplet, in their reordered order.
_MBC_COORDINATE_NAMES = ("collective", "cosine", "sine")


def _label_triplet_block(
    coords: list[str], offset: int, n_fixed: int, n_trip: int
) -> None:
    """Tag each triplet's three states collective/cosine/sine within one block."""
    for k in range(n_trip):
        base = offset + n_fixed + _N_BLADES * k
        for j in range(_N_BLADES):
            coords[base + j] = _MBC_COORDINATE_NAMES[j]


@dataclass
class _TransformPlan:
    """Reordering and triplet counts needed to transform every azimuth."""

    layout: _StateLayout
    seq2: _NewSequence
    seq1: _NewSequence
    inp: _NewSequence
    out: _NewSequence
    n_trip2: int
    n_trip1: int
    n_trip_inp: int
    n_trip_out: int

    @property
    def state_perm(self) -> list[int]:
        """Native state indices in reordered ``[q2, q2dot, q1]`` order."""
        q2 = [self.layout.disp2[k] for k in self.seq2.order]
        q2dot = [self.layout.vel2[k] for k in self.seq2.order]
        q1 = [self.layout.first1[k] for k in self.seq1.order]
        return q2 + q2dot + q1

    @property
    def mbc_coordinates(self) -> list[str]:
        """Multi-blade coordinate of each reordered state ``[q2, q2dot, q1]``.

        Each transformed blade-triplet state is ``"collective"``, ``"cosine"``, or
        ``"sine"``; non-rotating states are ``""``.
        """
        ndof2, ndof1 = self.layout.ndof2, self.layout.ndof1
        coords = [""] * (2 * ndof2 + ndof1)
        _label_triplet_block(coords, 0, self.seq2.n_fixed, self.n_trip2)
        _label_triplet_block(coords, ndof2, self.seq2.n_fixed, self.n_trip2)
        _label_triplet_block(coords, 2 * ndof2, self.seq1.n_fixed, self.n_trip1)
        return coords

    @property
    def output_mbc_coordinates(self) -> list[str]:
        """Multi-blade coordinate of each transformed output, in row order.

        A rotating output triplet's three rows become ``"collective"``,
        ``"cosine"``, ``"sine"`` after the transform; non-rotating outputs are
        ``""``. This lets a load downstream of the output equation be told apart
        from a literal per-blade channel.
        """
        coords = [""] * int(self.out.order.shape[0])
        _label_triplet_block(coords, 0, self.out.n_fixed, self.n_trip_out)
        return coords

    @property
    def input_mbc_coordinates(self) -> list[str]:
        """Multi-blade coordinate of each transformed input, in column order."""
        coords = [""] * int(self.inp.order.shape[0])
        _label_triplet_block(coords, 0, self.inp.n_fixed, self.n_trip_inp)
        return coords

    def transform_one(
        self, lin: LinFile, omega: float, omega_dot: float
    ) -> dict[str, npt.NDArray[np.float64] | None]:
        """Transform one azimuth's matrices into the non-rotating frame."""
        ndof2, ndof1 = self.layout.ndof2, self.layout.ndof1
        blades = lin.azimuth + 2.0 * np.pi / _N_BLADES * np.arange(_N_BLADES)
        cos_b, sin_b = np.cos(blades), np.sin(blades)

        tt = np.column_stack((np.ones(_N_BLADES), cos_b, sin_b))
        ttv = _tt_inverse(cos_b, sin_b)
        tt2 = np.column_stack((np.zeros(_N_BLADES), -sin_b, cos_b))
        tt3 = np.column_stack((np.zeros(_N_BLADES), -cos_b, -sin_b))

        t1 = _expand(tt, self.seq2.n_fixed, self.n_trip2, identity=True)
        t1v = _expand(ttv, self.seq2.n_fixed, self.n_trip2, identity=True)
        t2 = _expand(tt2, self.seq2.n_fixed, self.n_trip2, identity=False)
        t3 = _expand(tt3, self.seq2.n_fixed, self.n_trip2, identity=False)
        t1q = _expand(tt, self.seq1.n_fixed, self.n_trip1, identity=True)
        t1qv = _expand(ttv, self.seq1.n_fixed, self.n_trip1, identity=True)
        t2q = _expand(tt2, self.seq1.n_fixed, self.n_trip1, identity=False)
        t1c = _expand(tt, self.inp.n_fixed, self.n_trip_inp, identity=True)
        t1cv = _expand(ttv, self.inp.n_fixed, self.n_trip_inp, identity=True)
        t1ov = _expand(ttv, self.out.n_fixed, self.n_trip_out, identity=True)

        lmat = _block_lower(ndof2, ndof1, t1, omega * t2, t1q)
        rmat = _block_lower(
            ndof2,
            ndof1,
            omega * t2,
            omega**2 * t3 + omega_dot * t2,
            omega * t2q,
            corner=2.0 * omega * t2,
        )
        tinv = _block_diag(t1v, t1v, t1qv)

        state_perm = np.array(self.state_perm, dtype=np.int_)
        out: dict[str, npt.NDArray[np.float64] | None] = {
            "a": None,
            "b": None,
            "c": None,
            "d": None,
            "u_op": t1cv @ lin.u.values[self.inp.order],
            "y_op": t1ov @ lin.y.values[self.out.order],
        }
        if lin.a is not None:
            a_perm = lin.a[np.ix_(state_perm, state_perm)]
            out["a"] = tinv @ (a_perm @ lmat - rmat)
        if lin.b is not None:
            b_perm = lin.b[np.ix_(state_perm, self.inp.order)]
            out["b"] = tinv @ b_perm @ t1c
        if lin.c is not None:
            c_perm = lin.c[np.ix_(self.out.order, state_perm)]
            out["c"] = t1ov @ c_perm @ lmat
        if lin.d is not None:
            d_perm = lin.d[np.ix_(self.out.order, self.inp.order)]
            out["d"] = t1ov @ d_perm @ t1c
        return out


class _Accumulator:
    """Running sum of transformed matrices for azimuth averaging."""

    def __init__(self, reference: LinFile) -> None:
        self._sums: dict[str, npt.NDArray[np.float64] | None] = {
            "a": None,
            "b": None,
            "c": None,
            "d": None,
            "u_op": None,
            "y_op": None,
        }
        self._count = 0
        self._present = {
            "a": reference.a is not None,
            "b": reference.b is not None,
            "c": reference.c is not None,
            "d": reference.d is not None,
        }

    def add(self, matrices: dict[str, npt.NDArray[np.float64] | None]) -> None:
        """Add one azimuth's transformed matrices to the running totals."""
        for key, value in matrices.items():
            if value is None:
                continue
            current = self._sums[key]
            self._sums[key] = value if current is None else current + value
        self._count += 1

    def mean(self) -> dict[str, npt.NDArray[np.float64] | None]:
        """Return the azimuth means of each accumulated matrix."""
        out: dict[str, npt.NDArray[np.float64] | None] = {}
        for key, total in self._sums.items():
            out[key] = None if total is None else total / self._count
        return out


def _averaged_without_transform(
    files: Sequence[LinFile],
    layout: _StateLayout,
    mean_rpm: float,
    mean_ws: float,
    azimuths_deg: npt.NDArray[np.float64],
    *,
    retain_per_azimuth: bool = False,
) -> MBCResult:
    """Average matrices over azimuth without applying the MBC transform.

    The averaged matrices are still reordered into the canonical
    ``[q2, q2dot, q1]`` state layout so that ``ndof2``/``ndof1`` are consistent
    with the matrix ordering regardless of whether the transform was performed.
    """
    accum = _Accumulator(files[0])
    for lf in files:
        accum.add(
            {
                "a": lf.a,
                "b": lf.b,
                "c": lf.c,
                "d": lf.d,
                "u_op": lf.u.values,
                "y_op": lf.y.values,
            }
        )
    avg = accum.mean()
    state_perm = np.array(layout.disp2 + layout.vel2 + layout.first1, dtype=np.int_)
    descriptions = files[0].x.descriptions
    per_azimuth_a = (
        [lf.a[np.ix_(state_perm, state_perm)] for lf in files if lf.a is not None]
        if retain_per_azimuth
        else None
    )
    per_azimuth_b = (
        [lf.b[state_perm, :] for lf in files if lf.b is not None]
        if retain_per_azimuth and files[0].b is not None
        else None
    )
    per_azimuth_c = (
        [lf.c[:, state_perm] for lf in files if lf.c is not None]
        if retain_per_azimuth and files[0].c is not None
        else None
    )
    per_azimuth_d = (
        [np.array(lf.d, copy=True) for lf in files if lf.d is not None]
        if retain_per_azimuth and files[0].d is not None
        else None
    )
    per_azimuth_output_operating_point = (
        [np.array(lf.y.values, copy=True) for lf in files]
        if retain_per_azimuth
        else None
    )
    per_azimuth_input_operating_point = (
        [np.array(lf.u.values, copy=True) for lf in files]
        if retain_per_azimuth
        else None
    )
    return MBCResult(
        avg_a=None if avg["a"] is None else avg["a"][np.ix_(state_perm, state_perm)],
        avg_b=None if avg["b"] is None else avg["b"][state_perm, :],
        avg_c=None if avg["c"] is None else avg["c"][:, state_perm],
        avg_d=avg["d"],
        state_descriptions=[descriptions[i] for i in state_perm],
        ndof2=layout.ndof2,
        ndof1=layout.ndof1,
        n_blades=0,
        performed_transformation=False,
        rotor_speed_rpm=mean_rpm,
        wind_speed=mean_ws,
        azimuths_deg=azimuths_deg,
        mbc_coordinates=[""] * state_perm.shape[0],
        output_descriptions=list(files[0].y.descriptions),
        input_descriptions=list(files[0].u.descriptions),
        output_mbc_coordinates=[""] * len(files[0].y.descriptions),
        input_mbc_coordinates=[""] * len(files[0].u.descriptions),
        output_operating_point=avg["y_op"],
        input_operating_point=avg["u_op"],
        per_azimuth_a=per_azimuth_a,
        per_azimuth_b=per_azimuth_b,
        per_azimuth_c=per_azimuth_c,
        per_azimuth_d=per_azimuth_d,
        per_azimuth_output_operating_point=per_azimuth_output_operating_point,
        per_azimuth_input_operating_point=per_azimuth_input_operating_point,
    )


def _tt_inverse(
    cos_col: npt.NDArray[np.float64], sin_col: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Return the analytic inverse of the 3x3 multi-blade matrix ``tt``."""
    c1, c2, c3 = cos_col
    s1, s2, s3 = sin_col
    ttv: npt.NDArray[np.float64] = np.array(
        [
            [c2 * s3 - s2 * c3, c3 * s1 - s3 * c1, c1 * s2 - s1 * c2],
            [s2 - s3, s3 - s1, s1 - s2],
            [c3 - c2, c1 - c3, c2 - c1],
        ],
        dtype=np.float64,
    )
    scaled: npt.NDArray[np.float64] = ttv / (1.5 * 3.0**0.5)
    return scaled


def _expand(
    block: npt.NDArray[np.float64], n_fixed: int, n_triplets: int, *, identity: bool
) -> npt.NDArray[np.float64]:
    """Build a block-diagonal of a fixed-frame base plus repeated 3x3 blocks."""
    base = np.eye(n_fixed) if identity else np.zeros((n_fixed, n_fixed))
    return _block_diag(base, *([block] * n_triplets))


def _block_diag(*blocks: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Assemble a block-diagonal matrix from the given blocks.

    A thin typed wrapper over :func:`scipy.linalg.block_diag`; zero-sized blocks
    are permitted and contribute no rows or columns.

    Parameters
    ----------
    *blocks : npt.NDArray[np.float64]
        Two-dimensional blocks to place along the diagonal.

    Returns
    -------
    npt.NDArray[np.float64]
        The assembled block-diagonal matrix.
    """
    result: npt.NDArray[np.float64] = _scipy_block_diag(*blocks)
    return result


def _block_lower(
    ndof2: int,
    ndof1: int,
    top_left: npt.NDArray[np.float64],
    mid_left: npt.NDArray[np.float64],
    bottom_right: npt.NDArray[np.float64],
    *,
    corner: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """Assemble the lower-block-triangular ``L``/``R`` matrices of Eq. 29.

    The layout over the ``[q2, q2dot, q1]`` blocks is::

        [[top_left, 0, 0], [mid_left, center, 0], [0, 0, bottom_right]]

    where ``center`` is ``top_left`` for ``L`` and ``corner`` for ``R``.
    """
    size = 2 * ndof2 + ndof1
    mat = np.zeros((size, size), dtype=np.float64)
    mat[:ndof2, :ndof2] = top_left
    mat[ndof2 : 2 * ndof2, :ndof2] = mid_left
    mat[ndof2 : 2 * ndof2, ndof2 : 2 * ndof2] = top_left if corner is None else corner
    mat[2 * ndof2 :, 2 * ndof2 :] = bottom_right
    return mat


def _blade_token_spans(description: str) -> list[tuple[int, int]]:
    """Return non-overlapping spans of blade-index tokens (blade digit last).

    All specific blade patterns are collected and overlaps resolved greedily by
    position; if none match, the first single digit is used as a fallback so
    suffix-style names such as ``BldPitch1`` are still handled.

    Parameters
    ----------
    description : str
        The channel description.

    Returns
    -------
    list[tuple[int, int]]
        ``(start, end)`` spans, sorted by position; the blade digit of each token
        is at ``end - 1``.
    """
    spans: list[tuple[int, int]] = []
    for pattern in _BLADE_TOKEN_PATTERNS:
        spans.extend(match.span() for match in pattern.finditer(description))
    spans.sort()
    chosen: list[tuple[int, int]] = []
    last_end = -1
    for start, end in spans:
        if start >= last_end:
            chosen.append((start, end))
            last_end = end
    if chosen:
        return chosen
    fallback = _ANY_DIGIT.search(description)
    return [fallback.span()] if fallback is not None else []


def _build_sibling_matcher(
    description: str, spans: list[tuple[int, int]]
) -> re.Pattern[str]:
    """Compile a regex matching the same channel with a consistent other blade.

    Every blade-token digit is replaced by one capture group (the first
    occurrence) or a backreference to it (subsequent occurrences), so a matching
    sibling channel must repeat the same blade number at every blade token while
    keeping all other text — including node and span indices — literal.

    Parameters
    ----------
    description : str
        The reference channel description.
    spans : list[tuple[int, int]]
        Blade-token spans from :func:`_blade_token_spans`.

    Returns
    -------
    re.Pattern[str]
        A compiled pattern whose first group captures the sibling's blade digit.
    """
    parts: list[str] = []
    last = 0
    for index, (_, end) in enumerate(spans):
        parts.append(re.escape(description[last : end - 1]))
        parts.append(r"(\d)" if index == 0 else r"\1")
        last = end
    parts.append(re.escape(description[last:]))
    return re.compile("".join(parts))


def _strip_dof_index(description: str) -> str:
    """Remove the ``(internal DOF index = ...)`` parenthetical from a description.

    That parenthetical carries a second blade-numbered token (e.g.
    ``DOF_BF(1,1)``) that would otherwise vary across a blade triplet and prevent
    the sibling channels from matching.

    Parameters
    ----------
    description : str
        The channel description.

    Returns
    -------
    str
        The description with the DOF-index parenthetical removed, if present.
    """
    start = description.find("(internal DOF index = ")
    if start < 0:
        return description
    end = description.find(")", start)
    if end < 0:
        return description
    return description[:start] + description[end + 1 :]


def _present_matrices(lin: LinFile) -> frozenset[str]:
    """Return the set of state-space matrices present (not ``None``) in a file."""
    pairs = (("A", lin.a), ("B", lin.b), ("C", lin.c), ("D", lin.d))
    return frozenset(name for name, matrix in pairs if matrix is not None)


def _validate_sweep(
    files: Sequence[LinFile],
    rotor_speed_drift_rtol: float = _ROTOR_SPEED_DRIFT_RTOL,
    wind_speed_drift_rtol: float = _WIND_SPEED_DRIFT_RTOL,
) -> None:
    """Validate the files form one coherent azimuth sweep of a single operating point.

    The files must sample distinct azimuths — duplicate azimuths bias the azimuth
    average toward the repeated sample and corrupt the MBC integral — and hold the
    rotor and wind speeds roughly constant, since an averaged non-rotating model is
    only meaningful for one periodic operating point. These are quality checks the
    average silently violates otherwise.

    Parameters
    ----------
    files : Sequence[LinFile]
        The azimuth files of one operating point.
    rotor_speed_drift_rtol : float, optional
        Relative rotor-speed spread allowed across the sweep before the files are
        rejected as not one operating point.
    wind_speed_drift_rtol : float, optional
        Relative wind-speed spread allowed across the sweep before rejection.

    Raises
    ------
    ValueError
        If a relative tolerance is non-finite or negative, two files share an azimuth
        (within a small tolerance), or rotor/wind-speed spread exceeds its allowed
        drift.
    """
    for name, tolerance in (
        ("rotor_speed_drift_rtol", rotor_speed_drift_rtol),
        ("wind_speed_drift_rtol", wind_speed_drift_rtol),
    ):
        if not np.isfinite(tolerance) or tolerance < 0.0:
            msg = f"{name} must be finite and nonnegative"
            raise ValueError(msg)
    azimuths_deg = np.sort(
        np.array([lf.azimuth for lf in files], dtype=np.float64) * _DEG_PER_RAD % 360.0
    )
    if azimuths_deg.size >= 2:
        # Gaps around the full circle, including the wrap-around between the last and
        # first azimuth, so a 0 deg / ~360 deg pair is caught as a duplicate too.
        wrap_gap = 360.0 - float(azimuths_deg[-1] - azimuths_deg[0])
        min_gap = min(float(np.diff(azimuths_deg).min()), wrap_gap)
        if min_gap < _AZIMUTH_DUPLICATE_TOL_DEG:
            msg = (
                "Azimuth sweep has duplicate azimuths; each file must sample a "
                "distinct rotor azimuth"
            )
            raise ValueError(msg)

    wind = np.array([lf.wind_speed for lf in files], dtype=np.float64)
    if wind.size >= 2:
        spread = float(wind.max() - wind.min())
        tolerance = max(
            wind_speed_drift_rtol * float(np.mean(np.abs(wind))),
            _WIND_SPEED_DRIFT_ATOL,
        )
        if spread > tolerance:
            msg = (
                f"Wind speed drifts across the azimuth sweep "
                f"({wind.min():.4g} to {wind.max():.4g} m/s); a single operating "
                f"point must hold wind speed constant"
            )
            raise ValueError(msg)

    omega = np.array([lf.rotor_speed for lf in files], dtype=np.float64)
    if omega.size >= 2:
        spread = float(omega.max() - omega.min())
        tolerance = max(
            rotor_speed_drift_rtol * float(np.mean(np.abs(omega))),
            _ROTOR_SPEED_DRIFT_ATOL,
        )
        if spread > tolerance:
            msg = (
                f"Rotor speed drifts across the azimuth sweep "
                f"({omega.min():.4g} to {omega.max():.4g} rad/s); a single operating "
                f"point must hold rotor speed constant"
            )
            raise ValueError(msg)


def _check_azimuth_coverage(
    azimuths_deg: npt.NDArray[np.float64], mean_rpm: float
) -> None:
    """Warn when a rotating MBC average has too few or non-uniform azimuths.

    The MBC transform only removes the rotor's periodic content from an *anisotropic*
    rotating system when the azimuth average spans a full revolution near-uniformly.
    With too few or unevenly spaced azimuths the averaged LTI model is biased, which is
    easy to do unintentionally for a floating turbine. At standstill the transform is a
    similarity transform and a single azimuth is fine, so the check applies only when
    the rotor is spinning. This warns (it does not reject) — a single azimuth is
    legitimate for an isotropic-rotor mechanics check.
    """
    if abs(mean_rpm) < _ROTATING_RPM_TOL:
        return
    count = int(azimuths_deg.size)
    if count < _MIN_AZIMUTHS_FOR_MBC:
        logger.warning(
            "MBC averaged over %d azimuth(s) at %.3g rev/min: a rotating operating "
            "point needs a near-uniform full-revolution sweep (%d uniform azimuths is "
            "the bare minimum for a three-blade rotor; the OpenFAST norm is "
            "NLinTimes=36). The averaged model may be misleading — a single azimuth is "
            "only valid at standstill or for an isotropic rotor.",
            count,
            mean_rpm,
            _MIN_AZIMUTHS_FOR_MBC,
        )
        return
    wrapped = np.sort(azimuths_deg % 360.0)
    gaps = np.diff(np.append(wrapped, wrapped[0] + 360.0))
    largest_gap = float(gaps.max())
    smallest_gap = float(gaps.min())
    uniform_gap = 360.0 / count
    # Flag both a too-large gap (part of the revolution uncovered) and a too-small one
    # (azimuths clustered/near-duplicate, e.g. 0, 0.01, 180 deg — distinct enough to
    # pass the duplicate check yet effectively one fewer sample), since either biases
    # the azimuth average away from uniform full-revolution coverage.
    if (
        largest_gap > _NONUNIFORM_GAP_FACTOR * uniform_gap
        or smallest_gap < uniform_gap / _NONUNIFORM_GAP_FACTOR
    ):
        logger.warning(
            "MBC azimuths are non-uniform at %.3g rev/min (gaps span %.2f to %.0f deg "
            "vs %.0f deg for uniform coverage); the azimuth average is biased toward "
            "the densely sampled arc.",
            mean_rpm,
            smallest_gap,
            largest_gap,
            uniform_gap,
        )


def _blade_numbers(description: str) -> set[int]:
    """Return the blade numbers named by the specific blade tokens of a description.

    Only the explicit blade patterns count (``blade 2``, ``BD_2``, ``B2``, ...), not
    the single-digit fallback, which could read a node or mode number.
    """
    text = _strip_dof_index(description)
    return {
        int(text[match.end() - 1])
        for pattern in _BLADE_TOKEN_PATTERNS
        for match in pattern.finditer(text)
    }


def _reject_extra_blades(table: OperatingPointTable, kind: str) -> None:
    """Reject rotating ``kind`` (states, inputs or outputs) that name a fourth blade.

    The transform is the three-blade Coleman transform: blades 1-3 of a larger rotor
    would be transformed with a 120-degree spacing while the others stayed in the
    rotating frame. Rotating inputs and outputs are checked as well as states, since a
    rigid rotor has blade inputs and outputs but no blade states.

    Raises
    ------
    ValueError
        If a rotating entry of ``table`` names a blade numbered above three.
    """
    rotating = np.asarray(table.rotating_frame, dtype=np.bool_)
    blades = {
        number
        for description, flag in zip(table.descriptions, rotating, strict=True)
        if flag
        for number in _blade_numbers(description)
    }
    if blades and max(blades) > _N_BLADES:
        msg = (
            f"Rotating {kind} name blade {max(blades)}: the MBC3 transform supports "
            f"three-bladed rotors only"
        )
        raise ValueError(msg)


def _check_rotating_states(
    table: OperatingPointTable,
    layout: _StateLayout,
    triplets2: list[Triplet],
    triplets1: list[Triplet],
    mean_rpm: float,
) -> None:
    """Reject a rotor of more than three blades; warn of untransformed rotating states.

    The transform is the three-blade Coleman transform. A fourth blade's states would
    match no triplet while blades 1-3 were transformed with the 120-degree blade
    spacing of a three-bladed rotor, a silently wrong model, so it is rejected. A
    rotating state left outside every triplet (a one- or two-bladed rotor, or a
    blade-1-only state) stays in the rotating frame; while the rotor turns, its
    azimuth average is not a non-rotating model, which is logged as a warning.

    Raises
    ------
    ValueError
        If a rotating state names a blade numbered above three.
    """
    rotating = np.asarray(table.rotating_frame, dtype=np.bool_)
    _reject_extra_blades(table, "states")
    if abs(mean_rpm) < _ROTATING_RPM_TOL:
        return
    covered = {layout.disp2[i] for triplet in triplets2 for i in triplet}
    covered |= {layout.vel2[i] for triplet in triplets2 for i in triplet}
    covered |= {layout.first1[i] for triplet in triplets1 for i in triplet}
    left = [
        index for index in np.flatnonzero(rotating).tolist() if index not in covered
    ]
    if left:
        logger.warning(
            "%d rotating state(s) form no three-blade triplet (e.g. %r) and stay in "
            "the rotating frame at %.3g rev/min; their azimuth average is not a "
            "non-rotating model",
            len(left),
            table.descriptions[left[0]],
            mean_rpm,
        )


def _validate_consistent(files: Sequence[LinFile]) -> None:
    """Validate that every file shares the reference dimensions and matrices.

    Every azimuth file must share the reference's channel counts, present
    state-space matrices, and channel layout (state/input/output descriptions), so
    the azimuth average combines matching rows and columns rather than silently
    mixing channels from differently ordered or unrelated files.
    """
    ref = files[0]
    ref_present = _present_matrices(ref)
    if "A" not in ref_present:
        msg = "Reference linearization file has no A matrix"
        raise ValueError(msg)
    for lf in files[1:]:
        if (lf.n_x, lf.n_u, lf.n_y) != (ref.n_x, ref.n_u, ref.n_y):
            msg = (
                f"Inconsistent dimensions across azimuth files: "
                f"{lf.path.name} has ({lf.n_x}, {lf.n_u}, {lf.n_y}), "
                f"expected ({ref.n_x}, {ref.n_u}, {ref.n_y})"
            )
            raise ValueError(msg)
        present = _present_matrices(lf)
        if present != ref_present:
            msg = (
                f"Inconsistent matrices across azimuth files: "
                f"{lf.path.name} has {sorted(present)}, "
                f"expected {sorted(ref_present)}"
            )
            raise ValueError(msg)
        if (
            lf.x.descriptions != ref.x.descriptions
            or lf.u.descriptions != ref.u.descriptions
            or lf.y.descriptions != ref.y.descriptions
        ):
            msg = (
                f"Inconsistent channel layout across azimuth files: "
                f"{lf.path.name} has different state/input/output descriptions "
                f"from the reference file"
            )
            raise ValueError(msg)
