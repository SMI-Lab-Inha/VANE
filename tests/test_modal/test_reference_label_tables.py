# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Reference mode-label tables of the reference-turbine linearizations.

Each table below locks the labels and natural frequencies that
:func:`vane.modal.label_solution` and :func:`vane.modal.identify_modes` give on the
locally produced CalcSteady linearizations (36 azimuths per operating point) of the
NREL 5-MW (land, OC3 monopile), IEA 3.4-MW (land), IEA 10-MW, 15-MW and 22-MW
(monopile) reference turbines, and -- when ``VANE_REFERENCE_FULL=1`` -- the IEA 15-MW
VolturnUS-S semi-submersible and the IEA 22-MW rotor-only sweep, whose files are
large (0.5-1.7 GB per operating point). The values were checked against the physics
before being locked in: regressive and progressive whirl split by about the rotor
frequency either side of the rotating-frame blade mode, tower modes softened by the
monopile (NREL 5-MW first tower 0.32 -> 0.28 Hz, second 2.9 -> 2.4 Hz), and the
VolturnUS-S rigid-body periods (surge/sway 100-150 s, heave 20.6 s, roll/pitch
26-29 s, yaw about 90 s).

Every entry is an unambiguous label that must appear exactly once among the lightly
damped modes of its operating point, within 1 % of the listed frequency. The
``hybrids`` are modes known to mix two families (e.g. a monopile tower mode with a
blade whirl mode, or the near-critical collective flap with the rotor-speed mode);
they must be reported as ambiguous rather than given a confident label.

The data are never committed: the location is taken from ``VANE_REFERENCE_RUNS_DIR``,
and the tests skip when it is not set.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from vane.io import read_lin_file
from vane.io.mbc_transform import mbc3_transform
from vane.modal.eigensolver import ModalSolution, modes_from_mbc
from vane.modal.identifier import (
    DEFAULT_HEAVILY_DAMPED_THRESHOLD,
    identify_modes,
    rotor_frame_damping_ratios,
)
from vane.modal.labeler import label_solution

if TYPE_CHECKING:
    from collections.abc import Callable

    from vane.modal.labeler import ModeLabel

_RTOL = 0.01
_WHIRL = re.compile(
    r"^(?P<family>.*) \((?P<whirl>Regressive|Collective|Progressive)\)$"
)


@dataclass(frozen=True)
class _Case:
    """One turbine's reference table."""

    run: str
    stem: str
    subdir: str
    winds: tuple[float, ...]
    expected: dict[float, dict[str, float]]
    hybrids: dict[float, tuple[float, ...]] = field(default_factory=dict)
    f_max: float = 4.0
    heavy: bool = False
    case_dirs: Callable[[Path, float], Path] | None = None


def _ws(wind: float) -> str:
    return f"ws{wind:g}"


def _iea22_monopile_dir(runs: Path, wind: float) -> Path:
    # The 12 m/s point of the main sweep is incomplete; it was rerun separately.
    run = "iea22_monopile_g1e4" if wind == 12 else "iea22_monopile"
    return runs / run / "cases" / _ws(wind) / "IEA-22-280-RWT-Monopile"


def _iea22_rotor_dir(runs: Path, wind: float) -> Path:
    run = "iea22_rotoronly_ws9" if wind == 9 else "iea22_rotoronly"
    return runs / run / "cases" / _ws(wind) / "IEA-22-280-RWT-Monopile"


def _iea15_semi_dir(runs: Path, wind: float) -> Path:
    summary = runs / "iea15_semi__final" / "summary.json"
    if not summary.is_file():
        pytest.skip(f"sweep summary not found: {summary}")
    for row in json.loads(summary.read_text(encoding="utf-8")):
        if math.isclose(float(row["wind"]), wind):
            folder = Path(row["case_dir"])
            # The summary records absolute paths; re-root them on the runs folder.
            return (
                runs
                / folder.parent.parent.name
                / "cases"
                / folder.name
                / ("IEA-15-240-RWT-UMaineSemi")
            )
    pytest.skip(f"no {wind} m/s point in {summary}")


_CASES: dict[str, _Case] = {
    "nrel5mw_land": _Case(
        run="nrel5mw_land",
        stem="5MW_Land_DLL_WTurb",
        subdir="5MW_Land_DLL_WTurb",
        winds=(6.0, 13.0, 25.0),
        f_max=3.1,
        expected={
            6.0: {
                "1st Tower Side-Side": 0.3211,
                "1st Tower Fore-Aft": 0.3312,
                "1st Blade Flap (Regressive)": 0.5867,
                "1st Blade Flap (Collective)": 0.7095,
                "1st Blade Flap (Progressive)": 0.7960,
                "1st Blade Edge (Regressive)": 0.9759,
                "1st Blade Edge (Progressive)": 1.2091,
                "2nd Blade Flap (Regressive)": 1.8367,
                "2nd Blade Flap (Collective)": 2.0120,
                "2nd Blade Flap (Progressive)": 2.0670,
                "1st Blade Edge (Collective)": 2.1795,
                "2nd Tower Fore-Aft": 2.9142,
                "2nd Tower Side-Side": 2.9553,
            },
            13.0: {
                "1st Tower Side-Side": 0.3212,
                "1st Tower Fore-Aft": 0.3372,
                "1st Blade Flap (Regressive)": 0.6182,
                "1st Blade Flap (Collective)": 0.7611,
                "1st Blade Edge (Regressive)": 0.8880,
                "1st Blade Flap (Progressive)": 0.9092,
                "1st Blade Edge (Progressive)": 1.2962,
                "2nd Blade Flap (Regressive)": 1.7756,
                "2nd Blade Flap (Collective)": 2.0295,
                "2nd Blade Flap (Progressive)": 2.1663,
                "1st Blade Edge (Collective)": 2.1756,
                "2nd Tower Fore-Aft": 2.9090,
                "2nd Tower Side-Side": 2.9575,
            },
            25.0: {
                "1st Tower Side-Side": 0.3215,
                "1st Tower Fore-Aft": 0.3320,
                "1st Blade Flap (Regressive)": 0.6130,
                "1st Blade Flap (Collective)": 0.7084,
                "1st Blade Flap (Progressive)": 0.8684,
                "1st Blade Edge (Regressive)": 0.8842,
                "1st Blade Edge (Progressive)": 1.2910,
                "2nd Blade Flap (Regressive)": 1.7855,
                "2nd Blade Flap (Progressive)": 2.1781,
                "2nd Tower Fore-Aft": 2.9019,
                "2nd Tower Side-Side": 2.9775,
            },
        },
        # At 25 m/s (23 deg pitch) the collective 1st edge and 2nd flap exchange
        # character: 48 % edge against 44 % flap.
        hybrids={25.0: (1.8800,)},
    ),
    "nrel5mw_oc3monopile": _Case(
        run="nrel5mw_oc3monopile",
        stem="5MW_OC3Mnpl_DLL_WTurb_WavesIrr",
        subdir="5MW_OC3Mnpl_DLL_WTurb_WavesIrr",
        winds=(6.0, 13.0, 25.0),
        f_max=2.7,
        expected={
            6.0: {
                "1st Tower Side-Side": 0.2759,
                "1st Tower Fore-Aft": 0.2775,
                "1st Blade Flap (Regressive)": 0.5822,
                "1st Blade Flap (Collective)": 0.7099,
                "1st Blade Flap (Progressive)": 0.7907,
                "1st Blade Edge (Regressive)": 0.9748,
                "1st Blade Edge (Progressive)": 1.2081,
                "2nd Blade Flap (Regressive)": 1.7759,
                "2nd Tower Fore-Aft": 2.4224,
            },
            13.0: {
                "1st Tower Side-Side": 0.2760,
                "1st Tower Fore-Aft": 0.2826,
                "1st Blade Flap (Regressive)": 0.6129,
                "1st Blade Flap (Collective)": 0.7621,
                "1st Blade Edge (Regressive)": 0.8863,
                "1st Blade Flap (Progressive)": 0.9051,
                "1st Blade Edge (Progressive)": 1.2943,
                "2nd Blade Flap (Regressive)": 1.7295,
                "2nd Blade Flap (Collective)": 2.0292,
                "2nd Blade Flap (Progressive)": 2.1172,
                "2nd Tower Fore-Aft": 2.3949,
                "2nd Tower Side-Side": 2.5184,
            },
            25.0: {
                "1st Tower Side-Side": 0.2761,
                "1st Tower Fore-Aft": 0.2760,
                "1st Blade Flap (Regressive)": 0.6070,
                "1st Blade Flap (Collective)": 0.7093,
                "1st Blade Flap (Progressive)": 0.8643,
                "1st Blade Edge (Regressive)": 0.8787,
                "1st Blade Edge (Progressive)": 1.2829,
                "2nd Blade Flap (Regressive)": 1.7477,
                "2nd Blade Flap (Progressive)": 2.1399,
                "2nd Tower Fore-Aft": 2.3839,
                "2nd Tower Side-Side": 2.5589,
            },
        },
        # 6 m/s: the 2nd flap collective and progressive members are 0.6 % apart
        # and mix almost evenly; the collective 1st edge mixes with the 2nd tower
        # side-side mode of the monopile at every wind speed.
        hybrids={
            6.0: (2.0066, 2.0193, 2.0497),
            13.0: (2.0450,),
            25.0: (1.8877, 2.1026),
        },
    ),
    "iea3p4_land": _Case(
        run="iea3p4_land",
        stem="IEA-3.4-130-RWT",
        subdir="",
        winds=(6.0, 11.0, 20.0),
        f_max=2.8,
        expected={
            6.0: {
                "1st Tower Side-Side": 0.4066,
                "1st Tower Fore-Aft": 0.4114,
                "1st Blade Flap (Regressive)": 0.7053,
                "1st Blade Flap (Collective)": 0.8526,
                "1st Blade Flap (Progressive)": 0.9191,
                "1st Blade Edge (Regressive)": 0.9420,
                "1st Blade Edge (Progressive)": 1.1826,
                "1st Blade Edge (Collective)": 1.6017,
                "2nd Blade Flap (Regressive)": 2.1668,
                "2nd Blade Flap (Collective)": 2.4509,
                "2nd Tower Side-Side": 2.6843,
            },
            11.0: {
                "1st Tower Side-Side": 0.4066,
                "1st Tower Fore-Aft": 0.4223,
                "1st Blade Flap (Regressive)": 0.7906,
                "1st Blade Edge (Regressive)": 0.8603,
                "1st Blade Flap (Collective)": 0.9160,
                "1st Blade Flap (Progressive)": 1.0280,
                "1st Blade Edge (Progressive)": 1.2507,
                "1st Blade Edge (Collective)": 1.5992,
                "2nd Blade Flap (Regressive)": 2.0261,
                "2nd Blade Flap (Collective)": 2.3044,
                "2nd Blade Flap (Progressive)": 2.3802,
                "2nd Tower Fore-Aft": 2.6222,
                "2nd Tower Side-Side": 2.6842,
            },
            20.0: {
                "1st Tower Side-Side": 0.4069,
                "1st Tower Fore-Aft": 0.4207,
                "1st Blade Flap (Regressive)": 0.7962,
                "1st Blade Edge (Regressive)": 0.8542,
                "1st Blade Flap (Collective)": 0.8659,
                "1st Blade Flap (Progressive)": 0.9809,
                "1st Blade Edge (Progressive)": 1.2418,
                "1st Blade Edge (Collective)": 1.5574,
                "2nd Blade Flap (Regressive)": 2.0248,
                "2nd Blade Flap (Collective)": 2.3026,
                "2nd Blade Flap (Progressive)": 2.3880,
                "2nd Tower Fore-Aft": 2.6363,
                "2nd Tower Side-Side": 2.6849,
            },
        },
        # 6 m/s: the 2nd flap progressive member is a standing (unwhirled) pattern,
        # and the 2nd tower fore-aft mode mixes with the 2nd flap.
        hybrids={6.0: (2.3721, 2.7468)},
    ),
    "iea10_monopile": _Case(
        run="iea10_monopile",
        stem="IEA-10.0-198-RWT",
        subdir="",
        winds=(6.0, 9.5, 17.0),
        f_max=1.8,
        expected={
            6.0: {
                "1st Tower Side-Side": 0.2752,
                "1st Tower Fore-Aft": 0.2792,
                "1st Blade Flap (Regressive)": 0.3498,
                "1st Blade Flap (Collective)": 0.4459,
                "1st Blade Flap (Progressive)": 0.5225,
                "1st Blade Edge (Regressive)": 0.5670,
                "1st Blade Edge (Progressive)": 0.7688,
                "1st Blade Edge (Collective)": 1.3861,
                "2nd Blade Flap (Regressive)": 1.4103,
                "2nd Blade Flap (Collective)": 1.5821,
                "2nd Blade Flap (Progressive)": 1.5841,
                "2nd Tower Side-Side": 1.7498,
            },
            9.5: {
                "1st Tower Side-Side": 0.2752,
                "1st Tower Fore-Aft": 0.2812,
                "1st Blade Flap (Regressive)": 0.3535,
                "1st Blade Flap (Collective)": 0.4679,
                "1st Blade Edge (Regressive)": 0.5344,
                "1st Blade Flap (Progressive)": 0.5656,
                "1st Blade Edge (Progressive)": 0.8115,
                "2nd Blade Flap (Regressive)": 1.3546,
                "1st Blade Edge (Collective)": 1.3929,
                "2nd Blade Flap (Collective)": 1.5467,
                "2nd Blade Flap (Progressive)": 1.6306,
                "2nd Tower Fore-Aft": 1.6581,
                "2nd Tower Side-Side": 1.7488,
            },
            17.0: {
                "1st Tower Side-Side": 0.2753,
                "1st Tower Fore-Aft": 0.2820,
                "1st Blade Flap (Regressive)": 0.3766,
                "1st Blade Flap (Collective)": 0.4474,
                "1st Blade Edge (Regressive)": 0.5216,
                "1st Blade Flap (Progressive)": 0.5481,
                "1st Blade Edge (Progressive)": 0.8120,
                "1st Blade Edge (Collective)": 1.3405,
                "2nd Blade Flap (Regressive)": 1.3835,
                "2nd Blade Flap (Collective)": 1.6263,
                "2nd Tower Fore-Aft": 1.6384,
                "2nd Blade Flap (Progressive)": 1.6773,
                "2nd Tower Side-Side": 1.7493,
            },
        },
        # Below rated the monopile's 2nd fore-aft mode is half progressive 2nd flap.
        hybrids={6.0: (1.7190,)},
    ),
    "iea15_monopile": _Case(
        run="iea15_monopile",
        stem="IEA-15-240-RWT-Monopile",
        subdir="IEA-15-240-RWT-Monopile",
        winds=(6.0, 8.0, 12.0, 25.0),
        f_max=1.8,
        expected={
            6.0: {
                "1st Tower Side-Side": 0.1759,
                "1st Tower Fore-Aft": 0.1802,
                "1st Blade Flap (Regressive)": 0.5595,
                "1st Blade Flap (Collective)": 0.6285,
                "1st Blade Flap (Progressive)": 0.6492,
                "1st Blade Edge (Regressive)": 0.6641,
                "1st Blade Edge (Progressive)": 0.8332,
                "2nd Tower Fore-Aft": 1.2057,
                "2nd Tower Side-Side": 1.2970,
                "2nd Blade Flap (Regressive)": 1.3360,
                "2nd Blade Flap (Collective)": 1.4655,
                "2nd Blade Flap (Progressive)": 1.4893,
                "1st Blade Edge (Collective)": 1.7602,
            },
            8.0: {
                "1st Tower Side-Side": 0.1758,
                "1st Tower Fore-Aft": 0.1816,
                "1st Blade Flap (Regressive)": 0.5861,
                "1st Blade Flap (Collective)": 0.6534,
                "1st Blade Edge (Regressive)": 0.6575,
                "1st Blade Flap (Progressive)": 0.6624,
                "1st Blade Edge (Progressive)": 0.8505,
                "2nd Tower Fore-Aft": 1.2186,
                "2nd Blade Flap (Regressive)": 1.2688,
                "2nd Tower Side-Side": 1.2964,
                "2nd Blade Flap (Collective)": 1.4268,
                "2nd Blade Flap (Progressive)": 1.4518,
                "1st Blade Edge (Collective)": 1.7629,
            },
            # Above rated the 1st flap is past critical damping in the rotating
            # frame: its images form the heavily damped tier, not Campbell lines.
            12.0: {
                "1st Tower Side-Side": 0.1758,
                "1st Tower Fore-Aft": 0.1842,
                "1st Blade Edge (Regressive)": 0.6204,
                "1st Blade Edge (Progressive)": 0.8702,
                "2nd Blade Flap (Regressive)": 1.1119,
                "2nd Tower Fore-Aft": 1.2452,
                "2nd Tower Side-Side": 1.2966,
                "2nd Blade Flap (Collective)": 1.3132,
                "2nd Blade Flap (Progressive)": 1.3639,
                "1st Blade Edge (Collective)": 1.7602,
            },
            25.0: {
                "1st Tower Side-Side": 0.1762,
                "1st Tower Fore-Aft": 0.1793,
                "1st Blade Edge (Regressive)": 0.6128,
                "1st Blade Edge (Progressive)": 0.8573,
                "2nd Blade Flap (Regressive)": 1.1329,
                "2nd Tower Fore-Aft": 1.2622,
                "2nd Blade Flap (Collective)": 1.2976,
                "2nd Tower Side-Side": 1.2979,
                "2nd Blade Flap (Progressive)": 1.3808,
                "1st Blade Edge (Collective)": 1.7837,
            },
        },
        # 25 m/s: the near-critical collective flap couples with the rotor speed
        # (40 % flap against 39 % generator azimuth) and appears at this point only.
        hybrids={25.0: (0.1478,)},
    ),
    "iea22_monopile": _Case(
        run="iea22_monopile",
        stem="IEA-22-280-RWT-Monopile",
        subdir="IEA-22-280-RWT-Monopile",
        winds=(6.0, 10.0, 12.0, 24.0),
        case_dirs=_iea22_monopile_dir,
        expected={
            6.0: {
                "1st Tower Side-Side": 0.1582,
                "1st Tower Fore-Aft": 0.1615,
                "1st Blade Flap (Regressive)": 0.3314,
                "1st Blade Flap (Collective)": 0.4197,
                "1st Blade Flap (Progressive)": 0.4391,
                "1st Blade Edge (Regressive)": 0.4971,
                "1st Blade Edge (Progressive)": 0.6260,
                "2nd Tower Fore-Aft": 0.9715,
                "2nd Tower Side-Side": 1.0711,
                "1st Blade Edge (Collective)": 1.3178,
                "3rd Tower Fore-Aft": 2.1571,
                "3rd Tower Side-Side": 2.6650,
                "2nd Blade Flap (Collective)": 3.3872,
            },
            10.0: {
                "1st Tower Side-Side": 0.1582,
                "1st Tower Fore-Aft": 0.1662,
                "1st Blade Flap (Regressive)": 0.3519,
                "1st Blade Flap (Collective)": 0.4365,
                "1st Blade Edge (Regressive)": 0.4572,
                "1st Blade Flap (Progressive)": 0.4597,
                "1st Blade Edge (Progressive)": 0.6641,
                "2nd Tower Fore-Aft": 0.9633,
                "2nd Tower Side-Side": 1.0696,
                "1st Blade Edge (Collective)": 1.3152,
                "3rd Tower Fore-Aft": 2.1441,
                "3rd Tower Side-Side": 2.6531,
                # 0.07 % apart: one collective and one regressive member each.
                "2nd Blade Flap (Collective)": 3.3110,
                "2nd Blade Flap (Regressive)": 3.3133,
                "2nd Blade Flap (Progressive)": 3.6571,
            },
            12.0: {
                "1st Tower Side-Side": 0.1582,
                "1st Tower Fore-Aft": 0.1676,
                "1st Blade Edge (Regressive)": 0.4333,
                "1st Blade Edge (Progressive)": 0.6693,
                "2nd Tower Fore-Aft": 0.9664,
                "2nd Tower Side-Side": 1.0703,
                "1st Blade Edge (Collective)": 1.2935,
                "3rd Tower Fore-Aft": 2.1526,
                "3rd Tower Side-Side": 2.6675,
                "2nd Blade Flap (Regressive)": 3.3376,
                "2nd Blade Flap (Collective)": 3.3750,
                "2nd Blade Flap (Progressive)": 3.6469,
            },
            24.0: {
                "1st Tower Side-Side": 0.1585,
                "1st Tower Fore-Aft": 0.1640,
                "1st Blade Edge (Regressive)": 0.4288,
                "1st Blade Edge (Progressive)": 0.6621,
                "2nd Tower Fore-Aft": 0.9826,
                "2nd Tower Side-Side": 1.0716,
                "1st Blade Edge (Collective)": 1.2006,
                "3rd Tower Fore-Aft": 2.1790,
                "3rd Tower Side-Side": 2.6917,
                "2nd Blade Flap (Regressive)": 3.3976,
                "2nd Blade Flap (Collective)": 3.5483,
                "2nd Blade Flap (Progressive)": 3.6882,
            },
        },
        # The monopile's torsion (platform yaw) mixes with the 2nd flap regressive
        # at every wind speed; at 24 m/s the near-critical collective flap couples
        # with the rotor speed.
        hybrids={
            6.0: (3.0690,),
            10.0: (3.0860,),
            12.0: (3.1532,),
            24.0: (0.1229, 3.2061),
        },
    ),
    "iea15_semi": _Case(
        run="iea15_semi",
        stem="IEA-15-240-RWT-UMaineSemi",
        subdir="IEA-15-240-RWT-UMaineSemi",
        # 18 m/s is kept so the platform pitch line (0.039 -> 0.032 Hz above
        # rated) is sampled finely enough to be linked by its MAC.
        winds=(6.0, 12.0, 18.0, 24.0),
        heavy=True,
        case_dirs=_iea15_semi_dir,
        f_max=3.7,
        expected={
            6.0: {
                "Platform Sway": 0.0067,
                "Platform Surge": 0.0089,
                "Platform Yaw": 0.0112,
                "Platform Roll": 0.0346,
                "Platform Pitch": 0.0374,
                "Platform Heave": 0.0485,
                "1st Tower Side-Side": 0.5184,
                "1st Tower Fore-Aft": 0.5400,
                "1st Blade Flap (Regressive)": 0.5651,
                "1st Blade Flap (Collective)": 0.6282,
                "1st Blade Flap (Progressive)": 0.6543,
                "1st Blade Edge (Regressive)": 0.6712,
                "1st Blade Edge (Progressive)": 0.8381,
                "2nd Blade Flap (Regressive)": 1.3319,
                "2nd Blade Flap (Collective)": 1.4669,
                "2nd Blade Flap (Progressive)": 1.4931,
                "1st Blade Edge (Collective)": 1.7604,
                "2nd Tower Fore-Aft": 2.6068,
                "2nd Tower Side-Side": 3.5949,
            },
            12.0: {
                "Platform Sway": 0.0063,
                "Platform Surge": 0.0098,
                "Platform Yaw": 0.0118,
                "Platform Roll": 0.0344,
                "Platform Pitch": 0.0385,
                "Platform Heave": 0.0484,
                "1st Tower Side-Side": 0.5195,
                "1st Tower Fore-Aft": 0.5383,
                "1st Blade Edge (Regressive)": 0.6267,
                "1st Blade Edge (Progressive)": 0.8784,
                "2nd Blade Flap (Regressive)": 1.1397,
                "2nd Blade Flap (Collective)": 1.3116,
                "2nd Blade Flap (Progressive)": 1.3755,
                "1st Blade Edge (Collective)": 1.7602,
                "2nd Tower Fore-Aft": 2.5870,
                "2nd Tower Side-Side": 3.5904,
            },
            18.0: {
                "Platform Sway": 0.0064,
                "Platform Surge": 0.0085,
                "Platform Yaw": 0.0109,
                "Platform Pitch": 0.0313,
                "Platform Roll": 0.0345,
                "Platform Heave": 0.0484,
                "1st Tower Side-Side": 0.5185,
                "1st Tower Fore-Aft": 0.5402,
                "1st Blade Edge (Regressive)": 0.6278,
                "1st Blade Edge (Progressive)": 0.8768,
                "2nd Blade Flap (Regressive)": 1.1688,
                "2nd Blade Flap (Collective)": 1.3314,
                "2nd Blade Flap (Progressive)": 1.4051,
                "1st Blade Edge (Collective)": 1.7643,
                "2nd Tower Fore-Aft": 2.5876,
                "2nd Tower Side-Side": 3.5935,
            },
            24.0: {
                "Platform Sway": 0.0065,
                "Platform Surge": 0.0086,
                "Platform Yaw": 0.0101,
                "Platform Pitch": 0.0321,
                "Platform Roll": 0.0346,
                "Platform Heave": 0.0484,
                "1st Tower Side-Side": 0.5180,
                "1st Tower Fore-Aft": 0.5389,
                "1st Blade Edge (Regressive)": 0.6284,
                "1st Blade Edge (Progressive)": 0.8751,
                "2nd Blade Flap (Regressive)": 1.1607,
                "2nd Blade Flap (Collective)": 1.3044,
                "2nd Blade Flap (Progressive)": 1.3975,
                "1st Blade Edge (Collective)": 1.7791,
                "2nd Tower Fore-Aft": 2.5830,
                "2nd Tower Side-Side": 3.5944,
            },
        },
        hybrids={24.0: (0.1526,)},
    ),
    "iea22_rotoronly": _Case(
        run="iea22_rotoronly",
        stem="IEA-22-280-RWT-Monopile",
        subdir="IEA-22-280-RWT-Monopile",
        winds=(6.0, 12.0, 20.0),
        heavy=True,
        case_dirs=_iea22_rotor_dir,
        f_max=3.8,
        expected={
            6.0: {
                "1st Blade Flap (Regressive)": 0.4533,
                "1st Blade Edge (Regressive)": 0.4972,
                "1st Blade Flap (Collective)": 0.5198,
                "1st Blade Flap (Progressive)": 0.5404,
                "1st Blade Edge (Progressive)": 0.6202,
                "1st Blade Edge (Collective)": 1.3250,
                "2nd Blade Flap (Regressive)": 3.3697,
                "2nd Blade Flap (Collective)": 3.4685,
                "2nd Blade Flap (Progressive)": 3.4927,
            },
            12.0: {
                "1st Blade Edge (Regressive)": 0.4338,
                "1st Blade Edge (Progressive)": 0.6689,
                "1st Blade Edge (Collective)": 1.3101,
                "2nd Blade Flap (Regressive)": 3.3261,
                "2nd Blade Flap (Collective)": 3.5115,
                "2nd Blade Flap (Progressive)": 3.5596,
            },
            20.0: {
                "1st Blade Edge (Regressive)": 0.4299,
                "1st Blade Edge (Progressive)": 0.6653,
                "1st Blade Edge (Collective)": 1.2434,
                "2nd Blade Flap (Regressive)": 3.4195,
                "2nd Blade Flap (Progressive)": 3.6537,
                "2nd Blade Flap (Collective)": 3.7116,
            },
        },
        # With unsteady aerodynamics the 1st flap triplet near 1.1 Hz is about half
        # unsteady-aero states (unclassified) at 12 m/s.
        hybrids={12.0: (1.0684, 1.1119, 1.1571)},
    ),
}


def _runs_dir() -> Path:
    """Return the reference-turbine linearization folder, or skip when unavailable."""
    candidates = []
    env = os.environ.get("VANE_REFERENCE_RUNS_DIR")
    if env:
        candidates.append(Path(env))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    pytest.skip("reference-turbine runs unavailable; set VANE_REFERENCE_RUNS_DIR")


def _case_dir(case: _Case, runs: Path, wind: float) -> Path:
    if case.case_dirs is not None:
        return case.case_dirs(runs, wind)
    folder = runs / case.run / "cases" / _ws(wind)
    return folder / case.subdir if case.subdir else folder


def _point(case_dir: Path, stem: str) -> tuple[ModalSolution, float]:
    """Return the MBC modal solution and rotor speed of one operating point."""
    pattern = re.compile(rf"{re.escape(stem)}\.(\d+)\.lin")
    files = sorted(
        (p for p in case_dir.glob(f"{stem}.*.lin") if pattern.fullmatch(p.name)),
        key=lambda p: int(pattern.fullmatch(p.name).group(1)),  # type: ignore[union-attr]
    )
    if len(files) != 36:
        pytest.skip(f"incomplete azimuth sweep in {case_dir}")
    mbc = mbc3_transform([read_lin_file(p) for p in files])
    return modes_from_mbc(mbc), float(mbc.rotor_speed_rpm)


@dataclass
class _Loaded:
    case: _Case
    solutions: list[ModalSolution]
    speeds: list[float]
    labels: list[list[ModeLabel]]
    heavy: list[np.ndarray]


@pytest.fixture(scope="module", params=sorted(_CASES))
def loaded(request: pytest.FixtureRequest) -> _Loaded:
    """Return one turbine's reference table with its modal solutions and labels."""
    case = _CASES[request.param]
    if case.heavy and os.environ.get("VANE_REFERENCE_FULL") != "1":
        pytest.skip(f"{request.param}: large files; set VANE_REFERENCE_FULL=1 to run")
    runs = _runs_dir()
    solutions, speeds = [], []
    for wind in case.winds:
        folder = _case_dir(case, runs, wind)
        if not folder.is_dir():
            pytest.skip(f"reference-turbine case not found: {folder}")
        solution, speed = _point(folder, case.stem)
        solutions.append(solution)
        speeds.append(speed)
    labels = [label_solution(solution) for solution in solutions]
    heavy = [
        rotor_frame_damping_ratios(solution, speed, [lab.multiblade for lab in point])
        >= DEFAULT_HEAVILY_DAMPED_THRESHOLD
        for solution, speed, point in zip(solutions, speeds, labels, strict=True)
    ]
    return _Loaded(case, solutions, speeds, labels, heavy)


def _lightly_damped(loaded: _Loaded, index: int) -> list[tuple[float, ModeLabel]]:
    """Return ``(frequency, label)`` of a point's lightly damped modes below f_max."""
    solution = loaded.solutions[index]
    return [
        (float(frequency), label)
        for k, (frequency, label) in enumerate(
            zip(solution.natural_frequencies_hz, loaded.labels[index], strict=True)
        )
        if 1e-3 < frequency < loaded.case.f_max and not bool(loaded.heavy[index][k])
    ]


class TestPointLabels:
    """Per-operating-point labels against the reference tables."""

    def test_every_reference_label_is_found_once(self, loaded: _Loaded) -> None:
        """Each reference label names exactly one unambiguous mode at its frequency."""
        for index, wind in enumerate(loaded.case.winds):
            modes = _lightly_damped(loaded, index)
            for name, frequency in loaded.case.expected[wind].items():
                matches = [
                    f
                    for f, label in modes
                    if label.label == name and not label.is_ambiguous
                ]
                assert len(matches) == 1, (wind, name, matches)
                assert matches[0] == pytest.approx(frequency, rel=_RTOL), (wind, name)

    def test_hybrids_are_reported_ambiguous(self, loaded: _Loaded) -> None:
        """A mode that mixes two families is flagged, never confidently named."""
        for index, wind in enumerate(loaded.case.winds):
            solution = loaded.solutions[index]
            for frequency in loaded.case.hybrids.get(wind, ()):
                k = int(np.argmin(np.abs(solution.natural_frequencies_hz - frequency)))
                assert solution.natural_frequencies_hz[k] == pytest.approx(
                    frequency, rel=_RTOL
                )
                label = loaded.labels[index][k]
                assert label.is_ambiguous, (wind, frequency, label.label)
                assert label.alternative is not None

    def test_no_label_is_given_twice(self, loaded: _Loaded) -> None:
        """No unambiguous physical label repeats among one point's modes."""
        for index, wind in enumerate(loaded.case.winds):
            names = [
                label.label
                for _, label in _lightly_damped(loaded, index)
                if not label.is_ambiguous and not label.label.startswith("Unidentified")
            ]
            repeated = [name for name, count in Counter(names).items() if count > 1]
            assert not repeated, (wind, repeated)

    def test_regressive_lies_below_progressive(self, loaded: _Loaded) -> None:
        """Each blade family's regressive whirl is below its progressive whirl."""
        for index, wind in enumerate(loaded.case.winds):
            whirls: dict[str, dict[str, float]] = {}
            for frequency, label in _lightly_damped(loaded, index):
                match = _WHIRL.match(label.label)
                if match and not label.is_ambiguous:
                    whirls.setdefault(match["family"], {})[match["whirl"]] = frequency
            for family, members in whirls.items():
                if {"Regressive", "Progressive"} <= members.keys():
                    assert members["Regressive"] < members["Progressive"], (
                        wind,
                        family,
                    )


class TestTracks:
    """Campbell lines across the reference operating points."""

    def test_each_reference_line_is_one_track(self, loaded: _Loaded) -> None:
        """A label listed at every point forms one track through all of them."""
        result = identify_modes(loaded.solutions, rotor_speeds_rpm=loaded.speeds)
        common = set.intersection(
            *(set(table) for table in loaded.case.expected.values())
        )
        for name in sorted(common):
            tracks = [t for t in result.tracks if t.label.label == name]
            assert len(tracks) == 1, (name, [t.operating_points for t in tracks])
            assert tracks[0].operating_points == list(range(len(loaded.case.winds)))
            for index, frequency in enumerate(tracks[0].natural_frequencies_hz):
                wind = loaded.case.winds[index]
                assert frequency == pytest.approx(
                    loaded.case.expected[wind][name], rel=_RTOL
                ), (name, wind)

    def test_confident_tracks_agree_with_their_points(self, loaded: _Loaded) -> None:
        """A track not flagged ambiguous carries no contradicting point label."""
        result = identify_modes(loaded.solutions, rotor_speeds_rpm=loaded.speeds)
        for track in result.tracks:
            if track.is_ambiguous or track.label.label.startswith("Unidentified"):
                continue
            if float(np.mean(track.natural_frequencies_hz)) > loaded.case.f_max:
                continue
            for point in track.point_labels:
                assert point.is_ambiguous or point.label == track.label.label, (
                    track.label.label,
                    point.label,
                )
