# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.modal.averaging_error_estimate — the per-branch error budget."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.test_modal.test_averaging import (
    AZIMUTH,
    OMEGA,
    azimuth_system,
    floquet_exponents,
    jittered,
    matched_exponent,
    platform_index,
    sample,
)
from vane.modal.averaging import (
    AzimuthStates,
    EigenBasis,
    TowerPassageFit,
    azimuth_mean,
    eigen_basis,
    fit_harmonics,
    subgrid_weights,
    uniform_azimuths,
)
from vane.modal.averaging_error_estimate import (
    AveragingErrorEstimate,
    ErrorEstimate,
    alias_free_azimuth_count,
    azimuth_count_table,
    commensurate_time_step,
    estimate_averaging_error,
    first_alias_order,
    predict_branch_shift,
    trial_branch_shift,
)

if TYPE_CHECKING:
    import numpy.typing as npt


def tracked(
    reference: npt.NDArray[np.complex128], matrix: npt.NDArray[np.float64]
) -> tuple[complex, float]:
    """Return the eigenvalue of ``matrix`` best matching a reference vector, and MAC."""
    values, vectors = np.linalg.eig(matrix)
    macs = np.abs(reference.conj() @ vectors) ** 2 / (
        np.vdot(reference, reference).real * np.sum(np.abs(vectors) ** 2, axis=0)
    )
    macs[values.imag < -1e-12] = 0.0
    k = int(np.argmax(macs))
    return complex(values[k]), float(macs[k])


def geometric_model(ratio: float = 0.4, count: int = 36) -> object:
    """Return a harmonic model with geometrically decaying 3P, 6P, ... content."""

    def a_of(psi: float) -> npt.NDArray[np.float64]:
        modulation = sum(ratio**m * np.cos(3 * m * psi + 0.3 * m) for m in range(1, 6))
        return np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [-0.09 * (1.0 + 0.05 * modulation), 0.02 * modulation, -0.02, 0.0],
                [0.01 * modulation, -0.25, 0.0, -0.03],
            ]
        )

    angles = uniform_azimuths(count)
    return fit_harmonics(angles, sample(a_of, angles))


class TestPredictBranchShift:
    """Tests for predict_branch_shift."""

    @staticmethod
    def _errors(scale: float) -> tuple[float, float]:
        rng = np.random.default_rng(0)
        base = rng.normal(size=(6, 6)) - 3.0 * np.eye(6)
        error = scale * rng.normal(size=(6, 6))
        basis = eigen_basis(base)
        j = platform_index(basis.values)
        exact, _mac = tracked(basis.right[:, j], base + error)
        first, second = (
            predict_branch_shift(basis, error, [j], method=m)[0]  # type: ignore[arg-type]
            for m in ("first", "second")
        )
        reference = complex(basis.values[j])
        return (
            abs(reference + first.shift - exact),
            abs(reference + second.shift - exact),
        )

    def test_perturbation_orders(self) -> None:
        """First order errs at O(E^2) and second order at O(E^3)."""
        first_large, second_large = self._errors(1e-3)
        first_small, second_small = self._errors(5e-4)
        assert 3.5 < first_large / first_small < 4.5
        assert 7.0 < second_large / second_small < 9.0

    def test_structured_exact_for_azimuth_column(self) -> None:
        """A pure azimuth-column error, however large, is predicted exactly."""
        mean = azimuth_mean(
            uniform_azimuths(36), sample(azimuth_system(), uniform_azimuths(36))
        )
        basis = eigen_basis(mean)
        error = np.zeros((4, 4))
        error[:, 1] = [0.0, 0.0, 0.03, -0.04]
        j = platform_index(basis.values)
        estimate = predict_branch_shift(
            basis, error, [j], method="structured", azimuth=AZIMUTH
        )[0]
        exact, mac = tracked(basis.right[:, j], mean + error)
        assert abs(basis.values[j] + estimate.shift - exact) < 1e-10
        assert abs(estimate.mac - mac) < 1e-10
        assert estimate.method == "structured"

    def test_hybridisation_flag(self) -> None:
        """An aliased negative azimuth stiffness hybridises the platform branch."""
        f = azimuth_system(cpw=0.2, dwp=0.2, b0=0.02)
        mean = azimuth_mean(uniform_azimuths(36), sample(f, uniform_azimuths(36)))
        basis = eigen_basis(mean)
        j = platform_index(basis.values)
        for kappa, hybrid in ((0.01, False), (0.05, True)):
            error = np.zeros((4, 4))
            error[3, 1] = -kappa
            structured = predict_branch_shift(
                basis, error, [j], method="structured", azimuth=AZIMUTH
            )[0]
            _exact, mac = tracked(basis.right[:, j], mean + error)
            assert structured.hybridised is hybrid
            assert (mac < 0.9) is hybrid
            assert abs(structured.mac - mac) < 1e-9
        second = predict_branch_shift(basis, error, [j], method="second")[0]
        assert not second.hybridised  # ordinary perturbation misses it

    def test_structured_with_index_only(self) -> None:
        """An integer azimuth index (no speed state) is accepted."""
        mean = azimuth_mean(
            uniform_azimuths(36), sample(azimuth_system(), uniform_azimuths(36))
        )
        basis = eigen_basis(mean)
        error = 1e-4 * np.ones((4, 4))
        estimate = predict_branch_shift(
            basis, error, [0, 1], method="structured", azimuth=1
        )
        assert len(estimate) == 2
        assert all(isinstance(e, ErrorEstimate) for e in estimate)

    def test_validation(self) -> None:
        """Shapes, indices, methods and the azimuth requirement are validated."""
        basis = eigen_basis(np.diag([-1.0, -2.0]))
        with pytest.raises(ValueError, match="finite"):
            predict_branch_shift(basis, np.zeros((3, 3)), [0])
        with pytest.raises(ValueError, match="indices"):
            predict_branch_shift(basis, np.zeros((2, 2)), [2])
        with pytest.raises(ValueError, match="method"):
            predict_branch_shift(basis, np.zeros((2, 2)), [0], method="third")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="azimuth"):
            predict_branch_shift(basis, np.zeros((2, 2)), [0], method="structured")

    def test_real_reference_has_nan_frequency(self) -> None:
        """A real reference eigenvalue has no frequency error."""
        basis = eigen_basis(np.diag([-1.0, -2.0]))
        error = np.array([[0.1, 0.0], [0.0, 0.0]])
        estimate = predict_branch_shift(basis, error, [0])[0]
        assert math.isnan(estimate.frequency_error_pct)
        assert estimate.damping_error_pct == 0.0
        assert estimate.within(1.0, 1.0)


class TestAveragingErrorEstimate:
    """Tests for estimate_averaging_error."""

    @staticmethod
    def _case(delta: float = 0.0) -> tuple[npt.NDArray[np.float64], ...]:
        angles = jittered(36, 0.3, seed=4)
        return angles, sample(azimuth_system(delta=delta), angles)

    def test_terms_against_exact_values(self) -> None:
        """Sampling, quadrature and periodicity terms match exact eigen-solutions."""
        angles, stack = self._case()
        reference = azimuth_mean(angles, stack)
        basis = eigen_basis(reference)
        j = platform_index(basis.values)
        budget = estimate_averaging_error(
            angles,
            stack,
            OMEGA,
            branches={"platform": complex(basis.values[j])},
            azimuth=AZIMUTH,
            grid=angles[::3],
        )
        branch = budget.branches["platform"]
        assert branch.index == j
        assert budget.method == "structured"
        lam = complex(basis.values[j])
        # sampling: measured sub-grid
        picks = np.arange(0, 36, 3)
        coarse = np.tensordot(subgrid_weights(angles, picks), stack, axes=(0, 0))
        exact, _mac = tracked(basis.right[:, j], coarse)
        assert branch.sampling is not None
        assert abs(branch.sampling.shift - (exact - lam)) < 0.05 * abs(exact - lam)
        assert branch.sampling_dominant_order in (3, 6, 9, 12)
        # quadrature: arithmetic mean
        exact, _mac = tracked(basis.right[:, j], stack.mean(axis=0))
        assert abs(branch.quadrature.shift - (exact - lam)) < 0.05 * abs(exact - lam)
        # periodicity: Floquet exponent (uniform grid, so the reference is exact)
        uniform = uniform_azimuths(36)
        periodic = estimate_averaging_error(
            uniform, sample(azimuth_system(), uniform), OMEGA, branches={"p": lam}
        ).branches["p"]
        floquet = matched_exponent(
            floquet_exponents(azimuth_system()), periodic.eigenvalue
        )
        expected = floquet - periodic.eigenvalue
        assert abs(periodic.periodicity.shift - expected) < 0.1 * abs(expected)
        assert math.isnan(branch.periodicity.mac)
        assert branch.frequency_hz == pytest.approx(abs(lam.imag) / (2 * np.pi))
        assert budget.azimuth is not None
        assert budget.azimuth.consistent

    def test_hypothetical_grids(self) -> None:
        """A hypothetical grid is predicted from the interpolating model."""
        angles, stack = self._case()
        basis = eigen_basis(azimuth_mean(angles, stack))
        j = platform_index(basis.values)
        lam = complex(basis.values[j])
        grid = uniform_azimuths(6, 0.2)
        budget = estimate_averaging_error(
            angles, stack, OMEGA, branches={"p": j}, azimuth=AZIMUTH, grid=grid
        )
        coarse = sample(azimuth_system(), grid).mean(axis=0)
        exact, _mac = tracked(basis.right[:, j], coarse)
        sampling = budget.branches["p"].sampling
        assert sampling is not None
        assert abs(sampling.shift - (exact - lam)) < 0.05 * abs(exact - lam)
        counted = estimate_averaging_error(
            angles, stack, OMEGA, branches={"p": j}, grid=6
        )
        assert counted.branches["p"].sampling is not None
        assert counted.method == "second"

    def test_default_branches_and_assessments(self) -> None:
        """Default branches cover every oscillatory mode; tolerances decide."""
        angles, stack = self._case()
        loose = estimate_averaging_error(
            angles,
            stack,
            OMEGA,
            azimuth=AZIMUTH,
            damping_tolerance_pct=50.0,
            frequency_tolerance_pct=50.0,
        )
        assert isinstance(loose, AveragingErrorEstimate)
        assert list(loose.branches) == ["mode_00"]
        assert loose.branches["mode_00"].sampling is None
        assert loose.within_tolerance
        tight = estimate_averaging_error(
            angles,
            stack,
            OMEGA,
            azimuth=AZIMUTH,
            damping_tolerance_pct=1e-6,
            frequency_tolerance_pct=1e-6,
        )
        assert not tight.within_tolerance
        assert loose.grid.n_azimuths == 36
        assert loose.grid.kernel_magnitude.shape == (35,)
        assert loose.grid.max_gap_deviation_pct > 1.0
        inconsistent = estimate_averaging_error(
            *self._case(delta=1e-3),
            OMEGA,
            azimuth=AZIMUTH,
            damping_tolerance_pct=50.0,
            frequency_tolerance_pct=50.0,
        )
        assert inconsistent.azimuth is not None
        assert not inconsistent.azimuth.consistent
        # the residual is graded against the platform branch, not a fixed 1e-4 1/s
        assert inconsistent.azimuth_assessment in (
            "inconsistent-but-harmless",
            "inconsistent-and-risky",
        )
        assert inconsistent.within_tolerance is (
            inconsistent.azimuth_assessment == "inconsistent-but-harmless"
        )
        risky = estimate_averaging_error(
            *self._case(delta=-2e-3),
            OMEGA,
            azimuth=AZIMUTH,
            damping_tolerance_pct=1.0,
            frequency_tolerance_pct=50.0,
        )
        assert risky.azimuth_assessment == "inconsistent-and-risky"
        assert not risky.within_tolerance
        harmless = estimate_averaging_error(
            *self._case(delta=1e-4),
            OMEGA,
            azimuth=AZIMUTH,
            damping_tolerance_pct=50.0,
            frequency_tolerance_pct=50.0,
        )
        assert harmless.azimuth is not None
        assert not harmless.azimuth.absolute_consistent
        assert (
            harmless.within_tolerance
        )  # a harmless residual no longer voids the budget

    def test_branch_resolution_errors(self) -> None:
        """Bad branch specifications are rejected."""
        angles, stack = self._case()
        with pytest.raises(ValueError, match="out of range"):
            estimate_averaging_error(angles, stack, OMEGA, branches={"x": 9})
        with pytest.raises(ValueError, match="same mode"):
            estimate_averaging_error(angles, stack, OMEGA, branches={"a": 0, "b": 0})
        with pytest.raises(ValueError, match="at least one"):
            estimate_averaging_error(angles, stack, OMEGA, branches={})
        with pytest.raises(ValueError, match="damping_tolerance_pct"):
            estimate_averaging_error(angles, stack, OMEGA, damping_tolerance_pct=0.0)
        with pytest.raises(ValueError, match="method"):
            estimate_averaging_error(angles, stack, OMEGA, method="exact")  # type: ignore[arg-type]
        resolved = estimate_averaging_error(
            angles, stack, OMEGA, branches={"lower": -0.01 - 0.3j}
        )
        assert resolved.branches["lower"].eigenvalue.imag < 0.0


class TestAzimuthCountRule:
    """Tests for the lcm(N, B) rule and the azimuth-count table."""

    def test_first_alias_order(self) -> None:
        """The first alias of an N-mean is lcm(N, B)."""
        assert first_alias_order(8, 3) == 24
        assert first_alias_order(12, 3) == 12
        assert first_alias_order(7, 1) == 7
        for bad in (0, 2.5, True):
            with pytest.raises(ValueError, match="positive integer"):
                first_alias_order(bad, 3)  # type: ignore[arg-type]

    def test_alias_free_count(self) -> None:
        """The smallest N whose first alias lies above the content."""
        assert alias_free_azimuth_count(3, 18) == 7
        assert alias_free_azimuth_count(3, 2) == 1
        assert alias_free_azimuth_count(1, 18) == 19
        assert alias_free_azimuth_count(2, 9) == 5
        with pytest.raises(ValueError, match="highest_order"):
            alias_free_azimuth_count(3, 0)

    def test_lcm_rule_on_model(self) -> None:
        """Counts with no common multiple of B in the content are alias-free."""
        model = geometric_model()
        table = azimuth_count_table(model, counts=[3, 6, 7, 8, 9, 10, 12, 36], starts=8)
        resolved = {row.count: row.resolved_damping_error_pct for row in table.rows}
        assert resolved[7] < 1e-9
        assert resolved[8] < 1e-9
        assert resolved[10] < 1e-9
        for count in (3, 6, 9, 12):
            assert resolved[count] > 1e-3
        assert table.row(7).first_alias_order == 21
        assert table.row(9).multiple_of_blades
        assert not table.row(8).multiple_of_blades
        assert table.extrapolated
        assert table.row(7).unresolved_damping_bound_pct > 0.0
        assert (
            table.row(36).unresolved_damping_bound_pct
            < table.row(7).unresolved_damping_bound_pct
        )
        # only counts whose first alias the 36 samples resolve are accepted
        for row in table.rows:
            assert row.alias_resolved is (row.first_alias_order <= 17)
            if not row.alias_resolved:
                assert not row.resolved_within_tolerance
        assert table.resolved_order == 17
        assert table.recommended == table.recommended_resolved == 6
        assert table.row(6).resolved_within_tolerance
        assert table.recommended_extrapolated == min(
            r.count for r in table.rows if r.within_tolerance
        )
        assert table.extrapolation_status == "extrapolated"
        assert {t.model for t in table.tails.values()} <= {"own decay", "no content"}
        assert any(t.model == "no content" for t in table.tails.values())
        silent = next(t for t in table.tails.values() if t.model == "no content")
        assert silent.envelope(21) == 0.0
        with pytest.raises(KeyError):
            table.row(5)

    def test_without_extrapolation(self) -> None:
        """Without extrapolation the bound is NaN and the total is the resolved one."""
        model = geometric_model()
        basis = eigen_basis(model.mean)
        j = platform_index(basis.values)
        table = azimuth_count_table(
            model, {"p": complex(basis.values[j])}, counts=[6, 7], extrapolate=False
        )
        row = table.row(6)
        assert math.isnan(row.unresolved_damping_bound_pct)
        assert row.total_damping_bound_pct == row.resolved_damping_error_pct
        assert not table.extrapolated
        assert table.extrapolation_status == "not requested"
        assert table.recommended_extrapolated is None
        assert set(row.resolved_damping_by_branch) == {"p"}

    def test_no_count_meets_tolerance(self) -> None:
        """A tolerance below every error gives no recommendation."""
        table = azimuth_count_table(
            geometric_model(), counts=[6], damping_tolerance_pct=1e-9, starts=2
        )
        assert table.recommended is None
        assert not table.row(6).within_tolerance

    def test_too_few_orders_to_extrapolate(self) -> None:
        """With fewer than two resolved blade multiples nothing is extrapolated."""
        angles = uniform_azimuths(12)
        model = fit_harmonics(angles, sample(azimuth_system(), angles))
        table = azimuth_count_table(model, counts=[7], starts=2)
        assert not table.extrapolated

    def test_validation(self) -> None:
        """Invalid design arguments are rejected."""
        model = geometric_model()
        with pytest.raises(ValueError, match="blades"):
            azimuth_count_table(model, blades=0)
        with pytest.raises(ValueError, match="starts"):
            azimuth_count_table(model, starts=0)
        with pytest.raises(ValueError, match="damping_tolerance_pct"):
            azimuth_count_table(model, damping_tolerance_pct=-1.0)
        with pytest.raises(ValueError, match="at least one"):
            azimuth_count_table(model, counts=[])
        with pytest.raises(ValueError, match="counts"):
            azimuth_count_table(model, counts=[True])


class TestCommensurateTimeStep:
    """Tests for commensurate_time_step."""

    def test_whole_steps_per_azimuth(self) -> None:
        """The step divides the azimuth interval and does not exceed the request."""
        omega = 2 * np.pi / 10.0  # 10 s revolution
        step = commensurate_time_step(omega, 36, 0.025)
        assert step.steps_per_azimuth == 12
        assert step.time_step <= 0.025
        assert step.time_step * step.steps_per_azimuth * 36 == pytest.approx(10.0)
        assert step.azimuth_spacing_deg == pytest.approx(10.0)
        assert step.azimuth_per_step_deg * step.steps_per_azimuth == pytest.approx(10.0)
        assert 0.0 <= step.relative_change < 0.1

    def test_exact_divisor_and_large_request(self) -> None:
        """An exact divisor is kept; a request above the interval gives one step."""
        omega = 2 * np.pi / 9.0
        exact = commensurate_time_step(omega, 36, 0.25 / 8)
        assert exact.steps_per_azimuth == 8
        assert exact.relative_change == pytest.approx(0.0, abs=1e-12)
        large = commensurate_time_step(omega, 36, 1.0)
        assert large.steps_per_azimuth == 1
        assert large.time_step == pytest.approx(0.25)

    def test_validation(self) -> None:
        """Invalid rotor speed, count and time step are rejected."""
        with pytest.raises(ValueError, match="rotor_speed"):
            commensurate_time_step(0.0, 36, 0.01)
        with pytest.raises(ValueError, match="count"):
            commensurate_time_step(1.0, 0, 0.01)
        with pytest.raises(ValueError, match="requested_time_step"):
            commensurate_time_step(1.0, 36, math.inf)


def test_azimuth_states_reexported() -> None:
    """The azimuth states type is shared with the averaging module."""
    assert AzimuthStates(1, 3) == AZIMUTH


def oscillator(sigma: float, omega: float) -> npt.NDArray[np.float64]:
    """Return the first-order block of a damped oscillator ``-sigma +- j omega``."""
    return np.array([[-sigma, omega], [-omega, -sigma]])


def rebuild(basis: EigenBasis) -> npt.NDArray[np.float64]:
    """Return the real matrix of an eigenbasis."""
    return np.asarray(
        np.real((basis.right * basis.values) @ basis.left.conj().T), dtype=np.float64
    )


class TestConditioning:
    """Tests for the eigenvalue-conditioning check of the error estimate."""

    @staticmethod
    def _case(spread: float) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Two close oscillators whose eigenvectors are ``spread`` apart."""
        blocks = np.zeros((4, 4))
        blocks[:2, :2] = oscillator(0.01, 1.0)
        blocks[2:, 2:] = oscillator(0.012, 1.002)
        transform = np.eye(4)
        transform[:2, 2:] = np.eye(2)
        transform[2:, 2:] = spread * np.eye(2)
        mean = transform @ blocks @ np.linalg.inv(transform)
        angles = jittered(36, 0.2, seed=2)
        rng = np.random.default_rng(3)
        harmonic = 1e-4 * rng.normal(size=(4, 4))
        stack = np.array([mean + harmonic * np.cos(3 * psi) for psi in angles])
        return angles, stack

    def test_ill_conditioned_branch_is_not_well_conditioned(self) -> None:
        """A branch above max_condition is flagged whatever its error."""
        angles, stack = self._case(1e-4)
        budget = estimate_averaging_error(
            angles,
            stack,
            OMEGA,
            grid=angles[::3],
            damping_tolerance_pct=50.0,
            frequency_tolerance_pct=50.0,
        )
        basis = eigen_basis(azimuth_mean(angles, stack))
        for branch in budget.branches.values():
            assert branch.condition == pytest.approx(basis.condition[branch.index])
            assert branch.condition > 1e3
            assert not branch.well_conditioned
            assert not branch.within_tolerance
            assert branch.sampling is not None
            assert branch.sampling_exact is not None
            # the first-order pseudospectral bound covers the exact shift
            assert branch.sampling.shift_bound >= abs(branch.sampling_exact.shift)
        assert not budget.within_tolerance
        assert budget.max_condition == 1e3
        relaxed = estimate_averaging_error(
            angles,
            stack,
            OMEGA,
            grid=angles[::3],
            damping_tolerance_pct=50.0,
            frequency_tolerance_pct=50.0,
            max_condition=1e9,
        )
        assert all(b.well_conditioned for b in relaxed.branches.values())

    def test_well_conditioned_branch_is_well_conditioned(self) -> None:
        """A normal pair keeps a condition number near one."""
        angles, stack = self._case(1.0)
        budget = estimate_averaging_error(
            angles,
            stack,
            OMEGA,
            damping_tolerance_pct=50.0,
            frequency_tolerance_pct=50.0,
        )
        assert all(b.condition < 10.0 for b in budget.branches.values())
        assert all(b.well_conditioned for b in budget.branches.values())
        with pytest.raises(ValueError, match="max_condition"):
            estimate_averaging_error(angles, stack, OMEGA, max_condition=0.0)


class TestModeMixing:
    """Tests for the general mode-mixing check."""

    @staticmethod
    def _model(coupling: float) -> tuple[EigenBasis, npt.NDArray[np.float64]]:
        """Two close oscillators plus a decoupled azimuth pair, and a coupling error."""
        mean = np.zeros((6, 6))
        mean[:2, :2] = oscillator(0.3, 1.0)
        mean[2:4, 2:4] = oscillator(0.3, 1.02)
        mean[4, 5] = 1.0
        mean[5, 5] = -0.2
        error = np.zeros((6, 6))
        error[1, 2] = error[3, 0] = coupling
        return eigen_basis(mean), error

    def test_mixing_outside_the_azimuth_subspace_is_caught(self) -> None:
        """Two branches aliasing into each other: the old flag misses it."""
        basis, error = self._model(0.05)
        j = int(np.argmin(np.abs(basis.values - (-0.3 + 1.0j))))
        trial = trial_branch_shift(basis, rebuild(basis) + error, [j])[0]
        assert trial.method == "exact"
        assert trial.mixed
        assert trial.mixing_mac == trial.mac < 0.9
        structured = predict_branch_shift(
            basis, error, [j], method="structured", azimuth=AzimuthStates(4, 5)
        )[0]
        assert not structured.hybridised  # the azimuth-structured flag misses it
        assert structured.mixed
        assert structured.mixing_mac == pytest.approx(trial.mac, abs=1e-3)
        second = predict_branch_shift(basis, error, [j], method="second")[0]
        assert second.mixed
        assert not second.within(100.0, 100.0)

    def test_weak_coupling_is_no_false_alarm(self) -> None:
        """A weak coupling is not flagged as mixing."""
        basis, error = self._model(1e-4)
        picks = [int(k) for k in np.flatnonzero(basis.values.imag > 0)]
        for estimate in predict_branch_shift(basis, error, picks):
            assert not estimate.mixed
            assert estimate.mixing_mac > 0.999
        uncoupled = predict_branch_shift(basis, np.zeros((6, 6)), picks)
        assert all(e.mixing_mac == 1.0 for e in uncoupled)

    def test_estimate_reports_mixing(self) -> None:
        """The exact sampling term carries the trial-eigenproblem mixing check."""
        basis, error = self._model(0.05)
        mean = rebuild(basis)
        angles = uniform_azimuths(36)
        stack = np.array([mean + 2.0 * error * np.cos(3 * psi) for psi in angles])
        budget = estimate_averaging_error(
            angles,
            stack,
            OMEGA,
            grid=angles[::12],
            damping_tolerance_pct=100.0,
            frequency_tolerance_pct=100.0,
        )
        assert budget.sampling_status == "measured"
        mixed = [b for b in budget.branches.values() if b.mixed]
        assert mixed
        assert all(not b.within_tolerance for b in mixed)
        assert all(b.sampling_term is b.sampling_exact for b in mixed)

    def test_validation(self) -> None:
        """Bad coupling, trial shapes and indices are rejected."""
        basis, error = self._model(0.05)
        with pytest.raises(ValueError, match="mixing_coupling"):
            predict_branch_shift(basis, error, [0], mixing_coupling=0.0)
        with pytest.raises(ValueError, match="trial"):
            trial_branch_shift(basis, np.zeros((2, 2)), [0])
        with pytest.raises(ValueError, match="indices"):
            trial_branch_shift(basis, np.zeros((6, 6)), [7])


class TestUnresolvedGrids:
    """Tests for the alias-resolution check of predicted grids."""

    @staticmethod
    def _case() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        angles = jittered(36, 0.2, seed=5)
        return angles, sample(azimuth_system(), angles)

    def test_unresolved_grid_is_never_accepted(self) -> None:
        """A count whose first alias exceeds the resolved order is not accepted."""
        angles, stack = self._case()
        budget = estimate_averaging_error(
            angles,
            stack,
            OMEGA,
            azimuth=AZIMUTH,
            grid=7,
            damping_tolerance_pct=100.0,
            frequency_tolerance_pct=100.0,
        )
        assert budget.sampling_first_alias == 21
        assert budget.reference_resolved_order == 17
        assert budget.sampling_status is not None
        assert budget.sampling_status.startswith("unresolved: needs denser reference")
        for branch in budget.branches.values():
            assert not branch.sampling_resolved
            assert not branch.within_tolerance
        assert not budget.within_tolerance

    def test_resolved_and_measured_grids(self) -> None:
        """A resolved count and a measured sub-grid are accepted as usual."""
        angles, stack = self._case()
        resolved = estimate_averaging_error(
            angles,
            stack,
            OMEGA,
            azimuth=AZIMUTH,
            grid=12,
            damping_tolerance_pct=100.0,
            frequency_tolerance_pct=100.0,
        )
        assert resolved.sampling_status == "resolved"
        assert resolved.sampling_first_alias == 12
        assert resolved.within_tolerance
        measured = estimate_averaging_error(
            angles,
            stack,
            OMEGA,
            azimuth=AZIMUTH,
            grid=angles[::2],
            damping_tolerance_pct=100.0,
            frequency_tolerance_pct=100.0,
        )
        assert measured.sampling_status == "measured"
        assert measured.sampling_first_alias is None
        anisotropic = estimate_averaging_error(
            angles,
            stack,
            OMEGA,
            azimuth=AZIMUTH,
            grid=19,
            blades=1,
            damping_tolerance_pct=100.0,
            frequency_tolerance_pct=100.0,
        )
        assert anisotropic.sampling_first_alias == 19
        assert not anisotropic.within_tolerance
        bare = estimate_averaging_error(angles, stack, OMEGA)
        assert bare.sampling_status is None
        assert bare.azimuth_assessment is None
        with pytest.raises(ValueError, match="blades"):
            estimate_averaging_error(angles, stack, OMEGA, grid=12, blades=0)


class TestAliasTail:
    """Tests for the tail model of the azimuth-count table."""

    def test_growing_sensitivity_is_unresolved_without_a_tail(self) -> None:
        """Growing sensitivity is reported unresolved instead of a flat tail."""
        model = geometric_model(ratio=1.3)
        table = azimuth_count_table(model, counts=[5, 7, 8], starts=4)
        assert {t.model for t in table.tails.values()} - {"no content"} == {
            "unresolved"
        }
        assert table.extrapolation_status.startswith(
            "unresolved: needs denser linearization"
        )
        assert table.recommended_extrapolated is None
        assert not table.extrapolated
        for row in table.rows:
            assert math.isnan(row.total_damping_bound_pct)
            assert math.isnan(row.unresolved_damping_bound_pct)
            assert not row.within_tolerance
        for tail in table.tails.values():
            if tail.model == "unresolved":
                assert tail.peak_order == tail.anchor_order == 15
                assert math.isnan(tail.envelope(21))

    def test_tower_tail_converges_and_ignores_the_alias_limit(self) -> None:
        """With the tower-passage decay the estimate is finite and limit-free."""
        model = geometric_model(ratio=1.3)
        tables = [
            azimuth_count_table(
                model, counts=[7, 8], starts=4, tail_decay=0.8, max_alias_order=limit
            )
            for limit in (None, 360, 100_000)
        ]
        totals = [[r.total_damping_bound_pct for r in t.rows] for t in tables]
        assert all(math.isfinite(v) for v in totals[0])
        np.testing.assert_allclose(totals[0], totals[1], rtol=1e-9)
        np.testing.assert_allclose(totals[0], totals[2], rtol=1e-12)
        assert {t.model for t in tables[0].tails.values()} - {"no content"} == {
            "tower law"
        }
        assert tables[0].extrapolation_status == "extrapolated"
        tail = next(iter(tables[0].tails.values()))
        assert tail.envelope(21) == pytest.approx(tail.peak_level * (21 / 15) * 0.8**6)

    def test_own_decay_takes_precedence(self) -> None:
        """A decaying branch keeps its own envelope whatever the tower decay."""
        model = geometric_model(ratio=0.4)
        own = azimuth_count_table(model, counts=[7], starts=2)
        with_tower = azimuth_count_table(model, counts=[7], starts=2, tail_decay=0.95)
        decaying = [k for k, t in own.tails.items() if t.model == "own decay"]
        assert decaying
        for label in decaying:
            tail = with_tower.tails[label]
            assert tail.model == "own decay"
            assert math.isnan(tail.tower_decay)
            assert tail.envelope(21) == pytest.approx(own.tails[label].envelope(21))
            assert tail.envelope(21) == pytest.approx(
                tail.own_level * tail.own_decay ** (21 - tail.anchor_order)
            )

    def test_tower_fit_input(self) -> None:
        """A fit object is used when its quality allows; bad decays are rejected."""
        model = geometric_model(ratio=1.3)
        good = TowerPassageFit(
            orders=np.array([3, 6]),
            magnitudes=np.array([1.0, 0.5]),
            decay=0.8,
            amplitude=1.0,
            log_r2=0.95,
            condensed=True,
        )
        poor = TowerPassageFit(
            orders=good.orders,
            magnitudes=good.magnitudes,
            decay=0.8,
            amplitude=1.0,
            log_r2=0.1,
            condensed=True,
        )
        used = azimuth_count_table(model, counts=[7], starts=2, tail_decay=good)
        assert {t.model for t in used.tails.values()} - {"no content"} == {"tower law"}
        ignored = azimuth_count_table(model, counts=[7], starts=2, tail_decay=poor)
        assert {t.model for t in ignored.tails.values()} - {"no content"} == {
            "unresolved"
        }
        for bad in (1.0, 0.0, math.nan):
            with pytest.raises(ValueError, match="tail_decay"):
                azimuth_count_table(model, counts=[7], tail_decay=bad)


def test_unbounded_estimate_exceeds_the_tolerance() -> None:
    """An infinite (resonant) estimate exceeds it; an undefined (NaN) one is skipped."""
    basis = eigen_basis(np.diag([-1.0, -2.0]))
    error = np.zeros((2, 2))
    estimate = predict_branch_shift(basis, error, [0])[0]
    assert estimate.within(1.0, 1.0)
    unbounded = ErrorEstimate(
        shift=complex(math.inf, 0.0),
        frequency_error_pct=math.inf,
        damping_error_pct=math.inf,
        mac=math.nan,
        hybridised=False,
        method="sideband",
    )
    assert not unbounded.within(1.0, 1.0)
