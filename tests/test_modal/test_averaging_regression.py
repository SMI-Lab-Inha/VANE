# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Regression of the averaging error estimate on stored linearizations.

These tests reproduce reference values of the averaging error budget from stored
36-azimuth fixed-frame state matrices of two floating turbines, the IEA 15 MW
semi-submersible (``iea15``) and the OC3 spar (``oc3``), at eight operating points.
The matrices are not part of the repository. Set ``VANE_AVERAGING_ARCHIVE`` to a
directory holding ``<system>_u<wind:02d>/azimuth_matrices.npz`` (for example
``iea15_u12/azimuth_matrices.npz``); each file stores the delivered azimuths in
degrees (``azimuths_deg``) and the fixed-frame state matrices (``state_matrices``,
shape ``(n_azimuths, n, n)``). A second set, further linearizations of two of those
points on other azimuth grids (8 to 18 azimuths and a 72-azimuth reference), is read
from ``VANE_AVERAGING_RUNS`` as ``<point>/N<count>_off<offset>/azimuth_matrices.npz``.
Every test skips when its data are not available.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from vane.modal.averaging import (
    AzimuthStates,
    EigenBasis,
    azimuth_consistency,
    azimuth_mean,
    azimuth_tangent,
    eigen_basis,
    fit_harmonics,
    locate_azimuth_state,
    pin_azimuth,
    sideband_correction,
    subgrid_weights,
    zero_azimuth_column,
)
from vane.modal.averaging_error_estimate import (
    azimuth_count_table,
    estimate_averaging_error,
    predict_branch_shift,
)

if TYPE_CHECKING:
    import numpy.typing as npt

_ENV = "VANE_AVERAGING_ARCHIVE"
_LABELS = ("surge", "sway", "heave", "roll", "pitch", "yaw")
_FILE = "azimuth_matrices.npz"
# Rotor speed (rpm) of each archived operating point.
_RPM = {
    ("iea15", 6): 5.000011692174983,
    ("iea15", 10): 7.074569849553994,
    ("iea15", 12): 7.559806745217331,
    ("iea15", 18): 7.559753693569634,
    ("iea15", 24): 7.559939374336575,
    ("oc3", 8): 8.94,
    ("oc3", 12): 12.09,
    ("oc3", 18): 12.10,
}
# ElastoDyn DOF tokens in state order (blade DOFs omitted); the drivetrain DOF is
# present only in the semi-submersible deck.
_DOFS = ("Sg", "Sw", "Hv", "R", "P", "Y", "TFA1", "TSS1", "TFA2", "TSS2", "GeAz")


def _archive_root() -> Path | None:
    """Return the stored-matrix directory, or ``None`` when it is not available."""
    env = os.environ.get(_ENV)
    if not env:
        return None
    root = Path(env)
    return root if root.is_dir() else None


@functools.cache
def _load(
    system: str, wind: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], AzimuthStates, float]:
    """Return azimuths, matrices, the located azimuth states and Omega of a case."""
    root = _archive_root()
    path = None if root is None else root / f"{system}_u{wind:02d}" / _FILE
    if path is None or not path.is_file():
        pytest.skip(f"stored matrices unavailable; set {_ENV} to enable")
    with np.load(path, allow_pickle=False) as handle:
        angles = np.deg2rad(np.asarray(handle["azimuths_deg"], dtype=float))
        matrices = np.asarray(handle["state_matrices"], dtype=float)
    ndof2 = _second_order_size(matrices[0])
    descriptions = [f"HD radiation state {k}" for k in range(matrices.shape[1])]
    tokens = [*_DOFS, "DrTr"] if system == "iea15" else list(_DOFS)
    for k, token in enumerate(tokens):
        descriptions[k] = f"ED DOF {k} (internal DOF index = DOF_{token}), m"
        descriptions[k + ndof2] = (
            f"ED First time derivative of DOF {k} "
            f"(internal DOF index = DOF_{token}), m/s"
        )
    for k in range(len(tokens), ndof2):
        descriptions[k] = f"ED blade DOF {k}"
        descriptions[k + ndof2] = f"ED First time derivative of blade DOF {k}"
    azimuth = locate_azimuth_state(descriptions)
    return angles, matrices, azimuth, _RPM[(system, wind)] * 2 * np.pi / 60


def _second_order_size(matrix: npt.NDArray[np.float64]) -> int:
    """Return ``ndof2`` of a ``[q2, q2dot, q1]`` state matrix."""
    n = matrix.shape[0]
    for size in range(1, n // 2 + 1):
        expected = np.zeros((size, n))
        expected[:, size : 2 * size] = np.eye(size)
        if np.allclose(matrix[:size], expected, atol=1e-9):
            return size
    msg = "not a second-order layout"
    raise AssertionError(msg)


def _platform(basis: EigenBasis, ndof2: int) -> dict[str, int]:
    """Identify the six platform branches by left-right participation."""
    values = basis.values
    candidates = np.flatnonzero((values.imag > 0) & (values.imag / (2 * np.pi) < 0.12))
    part = basis.participation()
    shares = np.array(
        [
            (part[:6, i] + part[ndof2 : ndof2 + 6, i]) / part[:, i].sum()
            for i in candidates
        ]
    ).T
    rows, cols = linear_sum_assignment(-shares)
    return {_LABELS[r]: int(candidates[c]) for r, c in zip(rows, cols, strict=True)}


def _tracked(
    reference: npt.NDArray[np.complex128], matrix: npt.NDArray[np.float64]
) -> tuple[complex, float]:
    """Return the eigenvalue tracking a reference vector by MAC, and the MAC."""
    values, vectors = np.linalg.eig(matrix)
    macs = np.abs(reference.conj() @ vectors) ** 2 / (
        np.vdot(reference, reference).real * np.sum(np.abs(vectors) ** 2, axis=0)
    )
    macs[values.imag < -1e-12] = 0.0
    k = int(np.argmax(macs))
    return complex(values[k]), float(macs[k])


def _damping_error(reference: complex, trial: complex) -> float:
    """Return the relative damping-ratio change, percent."""
    z_ref = -reference.real / abs(reference)
    return 100.0 * abs(-trial.real / abs(trial) - z_ref) / abs(z_ref)


# Averaged azimuth eigenvalue (real part), its consistency residual (lambda plus the
# sideband correction; order 12, piecewise-linear harmonics) and the monodromy
# Floquet exponent.
_CONSISTENCY = {
    ("iea15", 6): (0.005628211877650916, 0.005686964245564221, 0.00568659514591145),
    ("iea15", 10): (-0.0002288282172532457, -0.00023783408059522708, -0.0002378916654),
    ("iea15", 12): (-3.8006999706438937e-06, 7.0943709101262925e-06, 9.750851056e-06),
    ("iea15", 18): (8.38832323188587e-06, 6.183410983427227e-05, 6.288502878055307e-05),
    ("iea15", 24): (0.00013843394126791337, 0.0001975980383234329, 0.0001984019430007),
    ("oc3", 8): (0.0196872734562588, 0.019871767157772825, 0.019862333615565192),
    ("oc3", 12): (0.018606338249185284, 0.018841435563832228, 0.018841353838460403),
    ("oc3", 18): (0.0008594244185792663, 0.0017755195544536429, 0.001769848912934908),
}


# Graded assessment of the scaled test against the six platform branches (1 % scale).
_ASSESSMENTS = {
    ("iea15", 6): "inconsistent-and-risky",
    ("iea15", 10): "inconsistent-but-harmless",
    ("iea15", 12): "consistent",
    ("iea15", 18): "inconsistent-but-harmless",
    ("iea15", 24): "inconsistent-but-harmless",
    ("oc3", 8): "inconsistent-and-risky",
    ("oc3", 12): "inconsistent-and-risky",
    ("oc3", 18): "inconsistent-and-risky",
}


@pytest.mark.parametrize("case", sorted(_CONSISTENCY))
def test_azimuth_consistency_reproduces_floquet(case: tuple[str, int]) -> None:
    """lambda_az plus its sideband correction gives the Floquet exponent to 1e-5."""
    angles, matrices, azimuth, omega = _load(*case)
    assert azimuth.position == 10
    model = fit_harmonics(angles, matrices, method="piecewise_linear", max_order=12)
    reference = azimuth_mean(angles, matrices)
    basis = eigen_basis(reference)
    branches = _platform(basis, azimuth.speed - azimuth.position)  # type: ignore[operator]
    result = azimuth_consistency(
        model, omega, azimuth, basis=basis, branches=list(branches.values())
    )
    averaged, residual, floquet = _CONSISTENCY[case]
    # every stored pair has its growing root on the azimuth mode
    assert result.eigenvalue.real == pytest.approx(averaged, rel=1e-6, abs=1e-10)
    assert result.exponent.real == pytest.approx(residual, rel=1e-6, abs=1e-10)
    assert abs(result.exponent.real - floquet) < 1e-5
    assert result.assessment == _ASSESSMENTS[case]
    # a fixed absolute tolerance of 1e-4 1/s accepts only the two smallest residuals
    assert result.absolute_consistent is (case in (("iea15", 12), ("iea15", 18)))
    if case in (("oc3", 8), ("oc3", 12)):
        surge = next(r for r in result.branch_risks if r.index == branches["surge"])
        assert surge.risky
        assert surge.coalescence_ratio > 0.15
        assert surge.decay_ratio > 2.0


# Structured damping-ratio predictions, percent, for the N = 12 sub-grid at
# offset 0 with the delivered azimuth column.
_STRUCTURED_N12 = {
    ("iea15", 12): (9.826856956579485, 0.16178013554787954, 0.025492327269440562,
                    0.08886515452024953, 0.19203343023260877, 0.04775564908131567),
    ("oc3", 8): (1.1380248315648034, 0.041004550734019894, 0.08565978358556196,
                 0.15940526711862765, 0.705481615890861, 0.06433768339089989),
}  # fmt: skip


@pytest.mark.parametrize("case", sorted(_STRUCTURED_N12))
def test_sampling_estimate_n12(case: tuple[str, int]) -> None:
    """The estimate reproduces the structured predictions and the actual errors."""
    angles, matrices, azimuth, omega = _load(*case)
    basis = eigen_basis(azimuth_mean(angles, matrices))
    ndof2 = azimuth.speed - azimuth.position if azimuth.speed else 0
    branches = _platform(basis, ndof2)
    budget = estimate_averaging_error(
        angles, matrices, omega, branches=branches, azimuth=azimuth, grid=angles[::3]
    )
    coarse = np.tensordot(subgrid_weights(angles, np.arange(0, 36, 3)), matrices, 1)
    for label, expected in zip(_LABELS, _STRUCTURED_N12[case], strict=True):
        sampling = budget.branches[label].sampling
        assert sampling is not None
        assert sampling.damping_error_pct == pytest.approx(expected, abs=1e-6)
        j = branches[label]
        actual, _mac = _tracked(basis.right[:, j], coarse)
        assert (
            abs(sampling.damping_error_pct - _damping_error(basis.values[j], actual))
            < 0.3
        )
    # Neutral treatment: ordinary second order is accurate to 0.05 damping points.
    neutral = zero_azimuth_column(matrices, azimuth)
    reference = azimuth_mean(angles, neutral)
    nbasis = eigen_basis(reference)
    nbranches = _platform(nbasis, ndof2)
    ncoarse = np.tensordot(subgrid_weights(angles, np.arange(0, 36, 3)), neutral, 1)
    for estimate, j in zip(
        predict_branch_shift(nbasis, ncoarse - reference, list(nbranches.values())),
        nbranches.values(),
        strict=True,
    ):
        actual, _mac = _tracked(nbasis.right[:, j], ncoarse)
        assert (
            abs(estimate.damping_error_pct - _damping_error(nbasis.values[j], actual))
            < 0.05
        )


def test_structured_predictor_flags_every_hybrid() -> None:
    """Over 8 cases x 15 sub-grids: 13 hybrids, all flagged, no false alarm."""
    actual_hybrids = flagged = false_alarms = 0
    worst_residual = 0.0
    for case in sorted(_CONSISTENCY):
        angles, matrices, azimuth, _omega = _load(*case)
        reference = azimuth_mean(angles, matrices)
        basis = eigen_basis(reference)
        branches = _platform(basis, azimuth.speed - azimuth.position)  # type: ignore[operator]
        picks = list(branches.values())
        for count in (6, 9, 12, 18):
            step = 36 // count
            for offset in range(step):
                weights = subgrid_weights(angles, np.arange(offset, 36, step))
                coarse = np.tensordot(weights, matrices, 1)
                estimates = predict_branch_shift(
                    basis,
                    coarse - reference,
                    picks,
                    method="structured",
                    azimuth=azimuth,
                )
                for j, estimate in zip(picks, estimates, strict=True):
                    actual, mac = _tracked(basis.right[:, j], coarse)
                    hybrid = mac < 0.9
                    actual_hybrids += hybrid
                    flagged += hybrid and estimate.hybridised
                    false_alarms += (not hybrid) and estimate.hybridised
                    error = _damping_error(basis.values[j], actual)
                    worst_residual = max(
                        worst_residual, abs(estimate.damping_error_pct - error)
                    )
    assert (actual_hybrids, flagged, false_alarms) == (13, 13, 0)
    assert worst_residual < 0.3


def test_quadrature_and_periodicity_terms() -> None:
    """Quadrature (first order) and the sideband correction reproduce the budget."""
    angles, matrices, azimuth, omega = _load("iea15", 12)
    reference = azimuth_mean(angles, matrices)
    basis = eigen_basis(reference)
    branches = _platform(basis, azimuth.speed - azimuth.position)  # type: ignore[operator]
    first = predict_branch_shift(
        basis, matrices.mean(axis=0) - reference, [branches["surge"]], method="first"
    )[0]
    assert first.damping_error_pct == pytest.approx(0.7274056785666687, rel=1e-6)
    actual, _mac = _tracked(basis.right[:, branches["surge"]], matrices.mean(axis=0))
    exact = _damping_error(basis.values[branches["surge"]], actual)
    assert abs(first.damping_error_pct - exact) < 0.01 * exact
    model = fit_harmonics(angles, matrices, method="piecewise_linear", max_order=12)
    expected = {
        "surge": 5.149989318841189e-06 + 1.2745583221778996e-06j,
        "pitch": 4.018343576469201e-06 - 1.4879551674112204e-06j,
        "roll": -3.116822551080362e-07 + 2.1252426651156218e-07j,
    }
    for label, value in expected.items():
        shift = sideband_correction(model, basis, branches[label], omega)
        assert abs(shift - value) < 1e-4 * abs(value)


@pytest.mark.parametrize(
    ("case", "worst"),
    [(("iea15", 12), 0.27188619156313226), (("oc3", 8), 0.11310009468817789)],
)
def test_tangent_pinning(case: tuple[str, int], worst: float) -> None:
    """Pinning keeps the reference (< 1e-6 %) and cures the N = 12 sampling error."""
    angles, matrices, azimuth, _omega = _load(*case)
    reference = azimuth_mean(angles, matrices)
    basis = eigen_basis(reference)
    ndof2 = azimuth.speed - azimuth.position  # type: ignore[operator]
    branches = _platform(basis, ndof2)
    pinned = pin_azimuth(matrices, azimuth_tangent(reference, azimuth), azimuth)
    pbasis = eigen_basis(azimuth_mean(angles, pinned))
    pbranches = _platform(pbasis, ndof2)
    for label, j in branches.items():
        k = pbranches[label]
        assert _damping_error(basis.values[j], pbasis.values[k]) < 1e-6
    largest = 0.0
    for offset in range(3):
        coarse = np.tensordot(
            subgrid_weights(angles, np.arange(offset, 36, 3)), pinned, 1
        )
        for j in pbranches.values():
            actual, _mac = _tracked(pbasis.right[:, j], coarse)
            largest = max(largest, _damping_error(pbasis.values[j], actual))
    assert largest == pytest.approx(worst, rel=1e-6)


def test_azimuth_count_table_neutral() -> None:
    """The design table reproduces the resolved and extrapolated count errors."""
    angles, matrices, azimuth, _omega = _load("iea15", 12)
    model = fit_harmonics(angles, zero_azimuth_column(matrices, azimuth))
    basis = eigen_basis(model.mean)
    branches = _platform(basis, azimuth.speed - azimuth.position)  # type: ignore[operator]
    table = azimuth_count_table(
        model,
        {k: complex(basis.values[v]) for k, v in branches.items()},
        counts=(7, 12),
    )
    seven, twelve = table.row(7), table.row(12)
    assert seven.resolved_damping_by_branch["surge"] == pytest.approx(
        0.01504941157039617, rel=1e-6
    )
    assert seven.unresolved_damping_by_branch["surge"] == pytest.approx(
        0.12380030953856165, rel=1e-6
    )
    assert table.tails["surge"].model == "own decay"
    assert twelve.resolved_damping_by_branch["surge"] == pytest.approx(
        0.2749942847358352, rel=1e-6
    )
    assert seven.total_damping_bound_pct < twelve.total_damping_bound_pct
    # 7 azimuths alias first at 21, above the 17 orders 36 samples resolve: only 12
    # is accepted, 7 is the extrapolated estimate
    assert not seven.alias_resolved
    assert twelve.alias_resolved
    assert table.recommended == table.recommended_resolved == 12
    assert table.recommended_extrapolated == 7


# --------------------------------------------------------------------------------------
# Further linearizations on azimuth grids outside the stored set
# --------------------------------------------------------------------------------------

_RUNS_ENV = "VANE_AVERAGING_RUNS"
_RUN_POINTS = {"iea15_u12": ("iea15", 12), "oc3_u08": ("oc3", 8)}
_RUN_COUNTS = (8, 10, 12, 13, 18)


def _run(point: str, run: str) -> tuple[npt.NDArray[np.float64], ...]:
    """Return the azimuths and matrices of one further linearization, or skip."""
    env = os.environ.get(_RUNS_ENV)
    path = None if not env else Path(env) / point / run / _FILE
    if path is None or not path.is_file():
        pytest.skip(f"further linearizations unavailable; set {_RUNS_ENV}")
    with np.load(path, allow_pickle=False) as handle:
        angles = np.deg2rad(np.asarray(handle["azimuths_deg"], dtype=float))
        matrices = np.asarray(handle["state_matrices"], dtype=float)
    return angles, matrices


@pytest.mark.parametrize("treatment", ["delivered", "pinned"])
@pytest.mark.parametrize("point", sorted(_RUN_POINTS))
def test_estimate_never_accepts_an_out_of_tolerance_branch(
    point: str, treatment: str
) -> None:
    """No branch of 90 further linearizations is accepted beyond 1 % / 0.1 %.

    The estimate is built from the stored 36-azimuth model only and evaluated on each
    run's own delivered azimuths. The actual error is the run's mean against the
    72-azimuth run. Without the alias-resolution check, 9 of 180 delivered cases would
    be accepted wrongly (all at 8, 10, 13 or 18 azimuths, whose first aliases 36
    samples cannot observe).
    """
    a72, m72 = _run(point, "N72_off0")
    angles, matrices, azimuth, omega = _load(*_RUN_POINTS[point])
    ndof2 = azimuth.speed - azimuth.position  # type: ignore[operator]
    if treatment == "pinned":
        tangent = azimuth_tangent(azimuth_mean(a72, m72), azimuth)
        matrices = pin_azimuth(
            matrices, azimuth_tangent(azimuth_mean(angles, matrices), azimuth), azimuth
        )
        m72 = pin_azimuth(m72, tangent, azimuth)
    reference = eigen_basis(azimuth_mean(a72, m72))
    truth = _platform(reference, ndof2)
    stored = _platform(eigen_basis(azimuth_mean(angles, matrices)), ndof2)
    wrong = accepted = unresolved = 0
    for count in _RUN_COUNTS:
        for offset in range(3):
            run_angles, run = _run(point, f"N{count}_off{offset}")
            if treatment == "pinned":
                run = pin_azimuth(run, tangent, azimuth)
            budget = estimate_averaging_error(
                angles,
                matrices,
                omega,
                branches=stored,
                azimuth=azimuth if treatment == "delivered" else None,
                grid=run_angles,
                damping_tolerance_pct=1.0,
                frequency_tolerance_pct=0.1,
            )
            mean = azimuth_mean(run_angles, run)
            for label, j in truth.items():
                actual, mac = _tracked(reference.right[:, j], mean)
                value = complex(reference.values[j])
                damping = _damping_error(value, actual)
                frequency = 100 * abs(abs(actual.imag) / abs(value.imag) - 1)
                outside = mac < 0.9 or damping > 1.0 or frequency > 0.1
                branch = budget.branches[label]
                wrong += branch.within_tolerance and outside
                accepted += branch.within_tolerance
                unresolved += not branch.sampling_resolved
            if count != 12 and budget.sampling_status != "measured":
                assert budget.sampling_status is not None
                assert budget.sampling_status.startswith("unresolved")
    assert wrong == 0
    assert accepted > 0  # the check is not vacuous: resolved counts are accepted
    assert unresolved > 0
