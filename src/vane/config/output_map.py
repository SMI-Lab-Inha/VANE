# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Classification of OpenFAST linearization output channels.

The output equation ``y = C x + D u`` of a linearization maps the system state and
inputs onto OpenFAST output channels. Many of those channels are *loads* — mooring
fairlead and anchor tensions, tower-base and tower-top (yaw-bearing) moments and
forces, blade-root moments, shaft torque — which are the quantities of interest for
structural load and health monitoring. This module classifies an output channel
description (e.g. ``"MAP T[1], [N]"`` or ``"ED YawBrMzp, (kN-m)"``) into a physical
:class:`OutputCategory`, recording the originating module, the load type, the
component index (mooring line or blade number), the measured quantity, the axis, and
the unit. It is the output-side counterpart of :func:`vane.config.classify_dof`.

The catalog covers the common MAP++/MoorDyn mooring, ElastoDyn, and BeamDyn load
channels; unrecognized channels are returned as :attr:`OutputCategory.UNKNOWN` and
are simply not treated as loads.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "LOAD_CATEGORIES",
    "OutputCategory",
    "OutputInfo",
    "classify_output",
]


class OutputCategory(Enum):
    """Physical category of a linearization output channel (string-valued)."""

    MOORING_FAIRLEAD_TENSION = "mooring_fairlead_tension"
    MOORING_ANCHOR_TENSION = "mooring_anchor_tension"
    MOORING_LINE_TENSION = "mooring_line_tension"
    TOWER_BASE_MOMENT = "tower_base_moment"
    TOWER_BASE_FORCE = "tower_base_force"
    TOWER_TOP_MOMENT = "tower_top_moment"
    TOWER_TOP_FORCE = "tower_top_force"
    BLADE_ROOT_MOMENT = "blade_root_moment"
    BLADE_ROOT_FORCE = "blade_root_force"
    SHAFT_TORQUE = "shaft_torque"
    PLATFORM_MOTION = "platform_motion"
    DEFLECTION = "deflection"
    WAVE_ELEVATION = "wave_elevation"
    UNKNOWN = "unknown"


#: Categories that represent a structural load (a force, moment, or tension).
LOAD_CATEGORIES: frozenset[OutputCategory] = frozenset(
    {
        OutputCategory.MOORING_FAIRLEAD_TENSION,
        OutputCategory.MOORING_ANCHOR_TENSION,
        OutputCategory.MOORING_LINE_TENSION,
        OutputCategory.TOWER_BASE_MOMENT,
        OutputCategory.TOWER_BASE_FORCE,
        OutputCategory.TOWER_TOP_MOMENT,
        OutputCategory.TOWER_TOP_FORCE,
        OutputCategory.BLADE_ROOT_MOMENT,
        OutputCategory.BLADE_ROOT_FORCE,
        OutputCategory.SHAFT_TORQUE,
    }
)


@dataclass(frozen=True)
class OutputInfo:
    """Classification of a single output channel description.

    Parameters
    ----------
    category : OutputCategory
        The physical category (``UNKNOWN`` if not recognized).
    module : str
        The originating module prefix (e.g. ``"ED"``, ``"MAP"``, ``"MD"``), or an
        empty string if absent.
    is_load : bool
        Whether the channel is a structural load (a member of
        :data:`LOAD_CATEGORIES`).
    component : int or None
        The 1-based mooring-line or blade index for component-indexed channels,
        otherwise ``None``.
    quantity : str
        The measured quantity (``"tension"``, ``"force"``, ``"moment"``,
        ``"displacement"``, ``"rotation"``, ``"elevation"``, or ``""``).
    axis : str or None
        The axis label (``"x"``, ``"y"``, ``"z"``) for directional forces and
        moments, otherwise ``None``.
    unit : str
        The unit token parsed from the trailing ``(...)`` or ``[...]`` of the
        description (e.g. ``"kN-m"``, ``"N"``, ``"m"``), or an empty string.
    """

    category: OutputCategory
    module: str
    is_load: bool
    component: int | None
    quantity: str
    axis: str | None
    unit: str


_MODULE_RE = re.compile(r"^([A-Za-z]+(?:_\d+)?)\s")
# Trailing unit token in parentheses or brackets, e.g. ", (kN-m)" or ", [N]".
_UNIT_RE = re.compile(r"[([]([^)\]]*)[)\]]\s*$")


@dataclass(frozen=True)
class _Pattern:
    """One channel-name pattern and the classification it yields."""

    regex: re.Pattern[str]
    category: OutputCategory
    quantity: str
    component_group: int | None
    axis_group: int | None
    component_from_module: bool = False


def _p(
    pattern: str,
    category: OutputCategory,
    quantity: str,
    *,
    component: int | None = None,
    axis: int | None = None,
    component_from_module: bool = False,
) -> _Pattern:
    """Build a case-insensitive :class:`_Pattern` (concise table helper)."""
    return _Pattern(
        regex=re.compile(pattern, re.IGNORECASE),
        category=category,
        quantity=quantity,
        component_group=component,
        axis_group=axis,
        component_from_module=component_from_module,
    )


# Channel-name patterns, matched in order (most specific first). The name is the
# token after the module prefix and before the trailing comma/unit.
_PATTERNS: tuple[_Pattern, ...] = (
    # MAP++ quasi-static mooring: T[i] = fairlead (top) tension, T_a[i] = anchor.
    _p(
        r"^T_a\[(\d+)\]$", OutputCategory.MOORING_ANCHOR_TENSION, "tension", component=1
    ),
    _p(
        r"^T\[(\d+)\]$", OutputCategory.MOORING_FAIRLEAD_TENSION, "tension", component=1
    ),
    # MoorDyn fairlead/anchor and per-node line tensions.
    _p(
        r"^ANCHTEN(\d+)$", OutputCategory.MOORING_ANCHOR_TENSION, "tension", component=1
    ),
    _p(
        r"^FAIRTEN(\d+)$",
        OutputCategory.MOORING_FAIRLEAD_TENSION,
        "tension",
        component=1,
    ),
    # MoorDyn line tensions: per-node ``L#N#T``/``LINE#N#T`` and end tensions
    # ``LINE#TENA``/``LINE#TENB`` (MoorDyn v2 uses the spelled-out ``LINE`` prefix).
    _p(
        r"^(?:LINE|L)(\d+)N\d+T$",
        OutputCategory.MOORING_LINE_TENSION,
        "tension",
        component=1,
    ),
    _p(
        r"^(?:LINE|L)(\d+)TEN[AB]$",
        OutputCategory.MOORING_LINE_TENSION,
        "tension",
        component=1,
    ),
    # ElastoDyn tower base moments/forces.
    _p(r"^TwrBsM([xyz])t$", OutputCategory.TOWER_BASE_MOMENT, "moment", axis=1),
    _p(r"^TwrBsF([xyz])t$", OutputCategory.TOWER_BASE_FORCE, "force", axis=1),
    # ElastoDyn yaw-bearing (tower-top) moments/forces.
    _p(r"^YawBrM([xyz])[pn]?$", OutputCategory.TOWER_TOP_MOMENT, "moment", axis=1),
    _p(r"^YawBrF([xyz])[pn]?$", OutputCategory.TOWER_TOP_FORCE, "force", axis=1),
    # Blade-root moments/forces. ElastoDyn directional channels exist in both the
    # blade (b) and coned (c) reference frames (RootMxb1/RootMxc1); also the
    # in-plane/out-of-plane/flap/edge aliases; and the BeamDyn B<blade>Root...r form.
    _p(
        r"^RootM([xyz])[bc](\d+)$",
        OutputCategory.BLADE_ROOT_MOMENT,
        "moment",
        component=2,
        axis=1,
    ),
    _p(
        r"^RootM(?:IP|OoP|Flp|Edg)(\d+)$",
        OutputCategory.BLADE_ROOT_MOMENT,
        "moment",
        component=1,
    ),
    _p(
        r"^B(\d+)RootM([xyz])r?$",
        OutputCategory.BLADE_ROOT_MOMENT,
        "moment",
        component=1,
        axis=2,
    ),
    _p(
        r"^RootF([xyz])[bc](\d+)$",
        OutputCategory.BLADE_ROOT_FORCE,
        "force",
        component=2,
        axis=1,
    ),
    _p(
        r"^B(\d+)RootF([xyz])r?$",
        OutputCategory.BLADE_ROOT_FORCE,
        "force",
        component=1,
        axis=2,
    ),
    # BeamDyn linearization outputs name the blade in the module prefix instead of
    # the channel (``BD_1 RootMxr``); the component is taken from ``BD_<n>``.
    _p(
        r"^RootM([xyz])r$",
        OutputCategory.BLADE_ROOT_MOMENT,
        "moment",
        axis=1,
        component_from_module=True,
    ),
    _p(
        r"^RootF([xyz])r$",
        OutputCategory.BLADE_ROOT_FORCE,
        "force",
        axis=1,
        component_from_module=True,
    ),
    # Drivetrain / low-speed shaft torque.
    _p(r"^(?:RotTorq|LSShftTq|LSShftMxa)$", OutputCategory.SHAFT_TORQUE, "moment"),
    # Non-load reference channels.
    _p(
        r"^Ptfm(?:Surge|Sway|Heave|Roll|Pitch|Yaw)$",
        OutputCategory.PLATFORM_MOTION,
        "displacement",
    ),
    _p(r"^TTDsp(?:FA|SS)$", OutputCategory.DEFLECTION, "displacement"),
    _p(r"^Wave\d*Elev1?$", OutputCategory.WAVE_ELEVATION, "elevation"),
)


def classify_output(description: str) -> OutputInfo:
    """Classify a linearization output channel description.

    Parameters
    ----------
    description : str
        An output channel description from a ``.lin`` file (e.g.
        ``"MAP T[1], [N]"`` or ``"ED YawBrMzp, (kN-m)"``).

    Returns
    -------
    OutputInfo
        The classification; ``category`` is :attr:`OutputCategory.UNKNOWN` when
        the channel name matches no known pattern.
    """
    module_match = _MODULE_RE.match(description)
    module = module_match.group(1) if module_match is not None else ""
    unit_match = _UNIT_RE.search(description)
    unit = unit_match.group(1).strip() if unit_match is not None else ""

    name = _channel_name(description, module)
    for pattern in _PATTERNS:
        match = pattern.regex.match(name)
        if match is None:
            continue
        component = (
            int(match.group(pattern.component_group))
            if pattern.component_group is not None
            else None
        )
        if pattern.component_from_module:
            component = _module_index(module)
        axis = (
            match.group(pattern.axis_group).lower()
            if pattern.axis_group is not None
            else None
        )
        return OutputInfo(
            category=pattern.category,
            module=module,
            is_load=pattern.category in LOAD_CATEGORIES,
            component=component,
            quantity=pattern.quantity,
            axis=axis,
            unit=unit,
        )
    return OutputInfo(
        category=OutputCategory.UNKNOWN,
        module=module,
        is_load=False,
        component=None,
        quantity="",
        axis=None,
        unit=unit,
    )


def _module_index(module: str) -> int | None:
    """Return the instance index of an indexed module prefix (``BD_2`` -> 2)."""
    _, sep, index = module.rpartition("_")
    return int(index) if sep and index.isdigit() else None


def _channel_name(description: str, module: str) -> str:
    """Return the bare channel name (module prefix and trailing unit removed)."""
    body = description[len(module) :].strip() if module else description.strip()
    # Drop the trailing ", (unit)" / ", [unit]" and any following text.
    body = re.split(r"\s*,\s*", body, maxsplit=1)[0]
    return body.strip()
