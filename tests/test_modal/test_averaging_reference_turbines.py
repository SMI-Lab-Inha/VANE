# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Regression of the averaging error estimate on reference-turbine linearizations.

These tests apply the error estimate to operating points of the NREL 5 MW, IEA 3.4 MW,
IEA 22 MW and IEA 15 MW reference turbines, linearized by
:func:`vane.run.run_operating_sweep` with its default settings (36 azimuths each,
``CalcSteady`` trims). The ``.lin`` files are not part of the repository. Set
``VANE_REFERENCE_TURBINES`` to a directory holding ``<configuration>/ws<wind>/``
(for example ``nrel5mw_land/ws8/``) with the ``.lin`` files of each point. Every test
skips when the data are not available. The floating points carry about 1000 states
and take about a minute each to read.
"""

from __future__ import annotations

import functools
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from vane.io import read_lin_file
from vane.io.mbc_transform import mbc3_transform
from vane.modal.averaging import (
    AzimuthStates,
    azimuth_consistency,
    azimuth_mean,
    azimuth_tangent,
    eigen_basis,
    fit_harmonics,
    locate_azimuth_state,
    pin_azimuth,
    subgrid_weights,
    tower_passage_decay,
)
from vane.modal.averaging_error_estimate import (
    azimuth_count_table,
    estimate_averaging_error,
    predict_branch_shift,
    trial_branch_shift,
)
from vane.modal.eigensolver import compute_modes
from vane.modal.labeler import label_solution

if TYPE_CHECKING:
    import numpy.typing as npt

_ENV = "VANE_REFERENCE_TURBINES"
_FIXED = (
    "1st Tower Fore-Aft",
    "1st Tower Side-Side",
    "1st Blade Flap (Regressive)",
    "1st Blade Flap (Collective)",
    "1st Blade Flap (Progressive)",
    "1st Blade Edge (Regressive)",
    "1st Blade Edge (Progressive)",
)
_FLOATING = (
    "Platform Surge",
    "Platform Sway",
    "Platform Heave",
    "Platform Roll",
    "Platform Pitch",
    "Platform Yaw",
    "1st Tower Fore-Aft",
    "1st Tower Side-Side",
)


@functools.cache
def _point(
    config: str, wind: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, AzimuthStates,
           dict[str, int]]:  # fmt: skip
    """Return azimuths, matrices, Omega, azimuth states and labelled branch indices."""
    env = os.environ.get(_ENV)
    folder = None if not env else Path(env) / config / f"ws{wind:g}"
    if folder is None or not folder.is_dir():
        pytest.skip(f"reference-turbine linearizations unavailable; set {_ENV}")
    paths = [p for p in folder.rglob("*.lin") if re.fullmatch(r".+\.\d+\.lin", p.name)]
    paths.sort(key=lambda p: int(re.findall(r"\.(\d+)\.lin$", p.name)[0]))
    result = mbc3_transform([read_lin_file(p) for p in paths], retain_per_azimuth=True)
    angles = np.deg2rad(result.azimuths_deg)
    samples = np.asarray(result.per_azimuth_a, dtype=np.float64)
    omega = result.rotor_speed_rpm * 2.0 * np.pi / 60.0
    azimuth = locate_azimuth_state(result.state_descriptions, ndof2=result.ndof2)
    reference = azimuth_mean(angles, samples)
    solution = compute_modes(
        reference,
        result.ndof2,
        result.ndof1,
        descriptions=result.state_descriptions,
        mbc_coordinates=result.mbc_coordinates,
    )
    labels = label_solution(solution, rotor_speed_rpm=result.rotor_speed_rpm)
    wanted = _FLOATING if "semi" in config else _FIXED
    basis = eigen_basis(reference)
    upper = np.flatnonzero(basis.values.imag > 0)
    picks: dict[str, int] = {}
    for k in np.argsort(solution.natural_frequencies_hz):
        name = labels[int(k)].label
        if name in wanted and name not in picks:
            value = complex(solution.eigenvalues[int(k)])
            picks[name] = int(upper[np.argmin(np.abs(basis.values[upper] - value))])
    return angles, samples, omega, azimuth, picks


def _consistency(config: str, wind: float) -> object:
    angles, samples, omega, azimuth, picks = _point(config, wind)
    model = fit_harmonics(angles, samples, method="piecewise_linear")
    return azimuth_consistency(model, omega, azimuth, branches=list(picks.values()))


def test_saddle_reports_growing_root_and_is_harmless() -> None:
    """NREL 5 MW land, 8 m/s: the saddle's growing root, harmless on the tower."""
    result = _consistency("nrel5mw_land", 8.0)
    growing, stable = result.pair_exponents  # type: ignore[attr-defined]
    assert growing.real == pytest.approx(0.0312, abs=2e-4)  # monodromy +0.0312
    assert stable.real == pytest.approx(-0.0287, abs=2e-4)  # the former report
    assert result.assessment == "inconsistent-but-harmless"  # type: ignore[attr-defined]
    assert not result.absolute_consistent  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("config", "wind", "assessment"),
    [
        ("iea3p4_land", 4.0, "consistent"),
        ("iea22_monopile", 17.0, "consistent"),
        ("iea22_monopile", 10.0, "inconsistent-but-harmless"),
        ("iea15_semi", 12.0, "inconsistent-and-risky"),
        ("iea22_semi", 8.0, "inconsistent-and-risky"),
    ],
)
def test_graded_assessments(config: str, wind: float, assessment: str) -> None:
    """Fixed-bottom residuals are harmless; floating near-rated ones are risky."""
    assert _consistency(config, wind).assessment == assessment  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("config", "wind", "expected"),
    [("iea3p4_land", 25.0, 10), ("iea22_monopile", 10.0, 13)],
)
def test_flap_mixing_at_three_azimuths_is_caught(
    config: str, wind: float, expected: int
) -> None:
    """The general check catches every three-azimuth flap mixing; no false alarm."""
    angles, samples, _omega, azimuth, picks = _point(config, wind)
    reference = azimuth_mean(angles, samples)
    basis = eigen_basis(reference)
    indices = list(picks.values())
    actual = caught = missed_by_structured = false_alarms = 0
    for offset in range(12):
        mean = np.tensordot(
            subgrid_weights(angles, np.arange(offset, 36, 12)), samples, axes=(0, 0)
        )
        exact = trial_branch_shift(basis, mean, indices)
        predicted = predict_branch_shift(
            basis, mean - reference, indices, method="structured", azimuth=azimuth
        )
        for ex, pr in zip(exact, predicted, strict=True):
            actual += ex.mixed
            caught += ex.mixed and pr.mixed
            missed_by_structured += ex.mixed and not pr.hybridised
            false_alarms += pr.mixed and not ex.mixed
    assert actual == expected
    assert caught == expected
    assert missed_by_structured == expected  # the azimuth-structured flag sees none
    assert false_alarms == 0


def test_ill_conditioned_sway_is_flagged_and_measured() -> None:
    """IEA 22 MW semi 8 m/s: sway (condition about 1178) is not well_conditioned."""
    angles, samples, omega, azimuth, picks = _point("iea22_semi", 8.0)
    budget = estimate_averaging_error(
        angles,
        samples,
        omega,
        branches=picks,
        azimuth=azimuth,
        grid=angles[2::6],
        damping_tolerance_pct=1.0,
        frequency_tolerance_pct=0.1,
    )
    sway = budget.branches["Platform Sway"]
    assert sway.condition == pytest.approx(1178, rel=0.01)
    assert not sway.well_conditioned
    assert not sway.within_tolerance
    assert sway.sampling is not None
    assert sway.sampling_exact is not None
    # every predictor under-predicts the measured 4.1 % six- to eight-fold
    assert sway.sampling.damping_error_pct < 0.7
    assert sway.sampling_exact.damping_error_pct == pytest.approx(4.08, abs=0.05)
    assert sway.sampling.shift_bound >= abs(sway.sampling_exact.shift)
    assert budget.branches["Platform Surge"].well_conditioned


def test_count_recommendations_are_labelled_and_stable() -> None:
    """NREL 5 MW land 13 m/s: growing flap sensitivity no longer gives 31."""
    angles, samples, _omega, azimuth, picks = _point("nrel5mw_land", 13.0)
    reference = azimuth_mean(angles, samples)
    basis = eigen_basis(reference)
    pinned = pin_azimuth(samples, azimuth_tangent(reference, azimuth), azimuth)
    model = fit_harmonics(angles, pinned)
    pinned_basis = eigen_basis(model.mean)
    branches = {
        label: complex(
            pinned_basis.values[
                np.argmin(np.abs(pinned_basis.values - basis.values[j]))
            ]
        )
        for label, j in picks.items()
    }
    tower = tower_passage_decay(fit_harmonics(angles, samples), azimuth)
    assert 0.8 < tower.decay < 0.9
    table = azimuth_count_table(
        model, branches, counts=range(5, 19), starts=6, tail_decay=tower
    )
    assert table.extrapolation_status == "extrapolated"
    assert table.recommended == table.recommended_resolved
    for row in table.rows:
        assert row.alias_resolved is (row.first_alias_order <= 17)
    assert table.recommended_extrapolated is not None
    assert table.recommended_extrapolated == 5  # the first run gave 31
    assert {t.model for t in table.tails.values()} <= {"own decay", "tower law"}
    assert table.recommended == 5
