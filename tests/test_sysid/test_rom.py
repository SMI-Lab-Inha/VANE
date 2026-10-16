# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.sysid.rom — modal reduction and load-observability ranking."""

import numpy as np
import pytest
from scipy.linalg import block_diag

from vane.sysid import (
    StateSpace,
    modal_reduction,
    output_mode_residues,
)

# A 3-mode synthetic system (all underdamped) so an all-mode modal realization is exact.
_MODES = [(0.30, 0.05), (2.00, 0.02), (5.00, 0.10)]  # (freq Hz, damping)


def _modal_block(freq: float, zeta: float) -> np.ndarray:
    """Return a continuous 2x2 block: natural frequency ``freq``, damping ``zeta``."""
    wn = 2.0 * np.pi * freq
    sigma = -zeta * wn
    wd = wn * np.sqrt(1.0 - zeta**2)
    return np.array([[sigma, -wd], [wd, sigma]])


def _system() -> StateSpace:
    """Return a 6-state, 2-input, 2-output system; output 0 is driven by mode 0."""
    a = block_diag(*[_modal_block(f, z) for f, z in _MODES])
    rng = np.random.default_rng(0)
    b = rng.standard_normal((6, 2))
    c = np.zeros((2, 6))
    c[0] = [1.0, 0.0, 0.05, 0.0, 0.02, 0.0]  # "load" dominated by mode 0
    c[1] = [0.3, 0.1, 0.8, 0.2, 0.5, 0.1]  # general output
    return StateSpace(
        a=a, b=b, c=c, d=np.zeros((2, 2)), output_names=["TwrBsMyt", "GenSpeed"]
    )


def _freq_response(ss: StateSpace, omega: float) -> np.ndarray:
    """Transfer-function matrix C (jw I - A)^-1 B + D at angular frequency omega."""
    assert ss.b is not None and ss.c is not None and ss.d is not None
    n = ss.a.shape[0]
    resolvent = np.linalg.inv(1j * omega * np.eye(n) - ss.a)
    return np.asarray(ss.c @ resolvent @ ss.b + ss.d)


class TestModalReduction:
    """Tests for modal_reduction."""

    def test_all_modes_preserves_eigenvalues(self) -> None:
        """Keeping every mode preserves the natural frequencies and eigenvalues."""
        red = modal_reduction(_system(), n_modes=None)
        assert red.system.n_states == 6
        assert red.n_full_states == 6
        np.testing.assert_allclose(
            np.sort(red.frequencies_hz), [0.3, 2.0, 5.0], atol=1e-6
        )
        got = np.sort_complex(np.linalg.eigvals(red.system.a))
        want = np.sort_complex(np.linalg.eigvals(_system().a))
        np.testing.assert_allclose(got, want, atol=1e-8)

    def test_all_mode_realization_matches_transfer_function(self) -> None:
        """The all-mode realization reproduces the full transfer function exactly."""
        ss = _system()
        red = modal_reduction(ss, n_modes=None).system
        for f in (0.1, 0.3, 1.0, 5.0):
            w = 2.0 * np.pi * f
            np.testing.assert_allclose(
                _freq_response(red, w), _freq_response(ss, w), atol=1e-8
            )

    def test_near_degenerate_modes_reduce_without_failure(self) -> None:
        """Veering (nearly coalesced) modes reduce without an inversion blow-up.

        Two underdamped modes a hair apart in frequency are mixed by an
        ill-conditioned similarity so their right eigenvectors nearly align — the
        case ``np.linalg.inv(eigvecs)`` raises or amplifies badly. The similarity
        preserves the eigenvalues, so the modal realization must still reproduce the
        full transfer function.
        """
        modal = block_diag(_modal_block(2.0000, 0.01), _modal_block(2.0006, 0.01))
        # A near-singular mixer (condition number ~1e6) that couples the two close
        # modes across blocks, collapsing their eigenvector directions together.
        mixer = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1e-6, 0.0],
                [0.0, 0.0, 0.0, 1e-6],
            ]
        )
        a = mixer @ modal @ np.linalg.inv(mixer)
        rng = np.random.default_rng(2)
        ss = StateSpace(
            a=a,
            b=rng.standard_normal((4, 1)),
            c=rng.standard_normal((1, 4)),
            d=np.zeros((1, 1)),
            output_names=["y"],
        )
        red = modal_reduction(ss, n_modes=None).system
        assert red.n_states == 4
        for f in (0.5, 2.0, 4.0):
            w = 2.0 * np.pi * f
            np.testing.assert_allclose(
                _freq_response(red, w), _freq_response(ss, w), atol=1e-6
            )

    def test_repeated_modes_preserve_transfer_function(self) -> None:
        """Exactly repeated modes keep a bi-orthogonal projection (the cluster dual).

        Two identical underdamped modes give a repeated eigenvalue of multiplicity
        two, then a similarity mixes the degenerate blocks so the cluster's right
        eigenvectors are non-orthogonal — normalizing each left vector independently
        would leak one mode's input residue into the other and change the response.
        """
        modal = block_diag(_modal_block(2.0, 0.03), _modal_block(2.0, 0.03))
        mixer = np.array(
            [
                [1.0, 0.0, 0.5, 0.0],
                [0.0, 1.0, 0.0, 0.5],
                [0.3, 0.0, 1.0, 0.0],
                [0.0, 0.3, 0.0, 1.0],
            ]
        )
        a = mixer @ modal @ np.linalg.inv(mixer)
        rng = np.random.default_rng(3)
        ss = StateSpace(
            a=a,
            b=rng.standard_normal((4, 2)),
            c=rng.standard_normal((1, 4)),
            d=np.zeros((1, 2)),
            output_names=["y"],
        )
        red = modal_reduction(ss, n_modes=None).system
        assert red.n_states == 4
        for f in (0.5, 2.0, 4.0):
            w = 2.0 * np.pi * f
            np.testing.assert_allclose(
                _freq_response(red, w), _freq_response(ss, w), atol=1e-8
            )

    def test_defective_repeated_mode_is_rejected(self) -> None:
        """A defective (Jordan-chain) repeated mode is rejected, not silently reduced.

        The real Jordan form of a complex pair with a size-two chain has the eigenvalue
        repeated with geometric multiplicity one, so its eigenvectors do not span the
        eigenspace. A block-diagonal modal realization cannot represent the generalized-
        eigenvector dynamics, so reduction must raise rather than return a model with a
        different transfer function.
        """
        sigma, omega = -0.1, 2.0 * np.pi * 2.0
        a = np.array(
            [
                [sigma, -omega, 1.0, 0.0],
                [omega, sigma, 0.0, 1.0],
                [0.0, 0.0, sigma, -omega],
                [0.0, 0.0, omega, sigma],
            ]
        )
        ss = StateSpace(
            a=a,
            b=np.ones((4, 1)),
            c=np.ones((1, 4)),
            d=np.zeros((1, 1)),
            output_names=["y"],
        )
        with pytest.raises(ValueError, match="defective"):
            modal_reduction(ss, n_modes=None)

    def test_load_observability_keeps_the_driving_mode(self) -> None:
        """Ranking by a load keeps the mode that drives it."""
        red = modal_reduction(_system(), n_modes=1, rank_output="TwrBsMyt")
        assert red.system.n_states == 2
        assert red.frequencies_hz[0] == pytest.approx(0.3, abs=1e-6)
        assert red.damping_ratios[0] == pytest.approx(0.05, abs=1e-6)
        assert 0.0 < red.energy_retained <= 1.0

    def test_joint_qoi_ranking_is_unit_normalized(self) -> None:
        """Several QoIs vote equally even when their raw units differ by orders."""
        a = np.asarray(block_diag(_modal_block(0.4, 0.05), _modal_block(1.2, 0.05)))
        c = np.zeros((3, 4))
        c[0, 0] = 1.0e6  # one newton-scale output votes for mode 0
        c[1, 2] = 1.0  # two metre/radian-scale outputs vote for mode 1
        c[2, 2] = 2.0
        system = StateSpace(a=a, c=c, output_names=["tension", "surge", "pitch"])
        reduction = modal_reduction(
            system,
            n_modes=1,
            rank_outputs=["tension", "surge", "pitch"],
        )
        assert reduction.frequencies_hz[0] == pytest.approx(1.2, abs=1e-6)

    def test_required_transfer_mode_is_reserved_before_output_ranking(self) -> None:
        """An unknown-input mode cannot be crowded out by structural QoI ranking."""
        a = np.asarray(block_diag(*[_modal_block(f, z) for f, z in _MODES]))
        b = np.zeros((6, 1))
        b[4, 0] = 1.0
        c = np.zeros((2, 6))
        c[0, 0] = 1.0
        c[0, 2] = 0.5
        c[1, 4] = 1.0
        system = StateSpace(
            a=a,
            b=b,
            c=c,
            input_names=["unknown torque"],
            output_names=["structural load", "rotor speed"],
        )
        reduction = modal_reduction(
            system,
            n_modes=2,
            rank_output="structural load",
            required_transfer_pairs=[("rotor speed", "unknown torque")],
            n_required_transfer_modes=1,
        )
        np.testing.assert_allclose(
            np.sort(reduction.frequencies_hz), [0.3, 5.0], atol=1e-6
        )
        assert reduction.system.b is not None and reduction.system.c is not None
        assert system.b is not None and system.c is not None
        full_moment = system.c[1] @ system.b[:, 0]
        reduced_moment = reduction.system.c[1] @ reduction.system.b[:, 0]
        assert reduced_moment == pytest.approx(full_moment)

    def test_frequency_band_filters_candidates(self) -> None:
        """A frequency band restricts the candidate modes."""
        red = modal_reduction(_system(), frequency_band_hz=(1.0, 3.0))
        np.testing.assert_allclose(red.frequencies_hz, [2.0], atol=1e-6)

    def test_stabilizes_unstable_poles(self) -> None:
        """An unstable retained pole is reflected into the stable half-plane."""
        a = block_diag(_modal_block(0.3, 0.05), np.array([[0.5, -6.0], [6.0, 0.5]]))
        ss = StateSpace(a=a, b=np.ones((4, 1)), c=np.ones((1, 4)), output_names=["y"])
        red = modal_reduction(ss, n_modes=None, stabilize=True)
        assert red.n_stabilized == 1
        assert np.linalg.eigvals(red.system.a).real.max() < 0.0
        # The reported damping must match the realized (stabilized) dynamics, not the
        # original unstable sign: every ratio is positive and consistent with the stable
        # reduced A. For the stabilized 0.5 +/- 6j pole, zeta = 0.5 / |0.5 + 6j|.
        assert np.all(red.damping_ratios > 0.0)
        # The stabilized pole's zeta (0.083) exceeds the other mode's (0.05), so it is
        # the maximum; pre-fix it would have been reported as -0.083.
        assert red.damping_ratios.max() == pytest.approx(
            0.5 / abs(complex(0.5, 6.0)), rel=1e-6
        )
        # The per-mode mask identifies exactly the reflected mode.
        assert red.stabilized is not None
        assert int(np.count_nonzero(red.stabilized)) == 1
        flipped = int(np.flatnonzero(red.stabilized)[0])
        assert red.frequencies_hz[flipped] == pytest.approx(
            abs(complex(0.5, 6.0)) / (2.0 * np.pi), rel=1e-9
        )

    def test_reduced_system_keeps_io_labels(self) -> None:
        """The reduced system preserves the input/output labels and stays continuous."""
        red = modal_reduction(_system(), n_modes=2).system
        assert red.output_names == ["TwrBsMyt", "GenSpeed"]
        assert red.b is not None and red.b.shape == (4, 2)
        assert red.dt is None

    def test_optional_dc_matching_preserves_static_gain(self) -> None:
        """A reduced module retains equilibrium stiffness despite modal truncation."""
        full = _system()
        assert full.b is not None and full.c is not None
        full.d = np.full((full.n_outputs, full.n_inputs), 1.0e6)
        reduced = modal_reduction(
            full, n_modes=1, rank_output="TwrBsMyt", match_dc_gain=True
        ).system
        assert reduced.b is not None and reduced.c is not None
        full_gain = full.d - full.c @ np.linalg.solve(full.a, full.b)
        reduced_gain = reduced.d - reduced.c @ np.linalg.solve(reduced.a, reduced.b)
        np.testing.assert_allclose(reduced_gain, full_gain, rtol=1e-12, atol=1e-10)


class TestValidation:
    """Tests for modal_reduction input validation."""

    def test_rejects_discrete_system(self) -> None:
        """A discrete-time system is rejected."""
        ss = _system()
        ss.dt = 0.01
        with pytest.raises(ValueError, match="continuous-time"):
            modal_reduction(ss)

    def test_rejects_missing_c(self) -> None:
        """A system without a C matrix is rejected."""
        ss = StateSpace(a=np.eye(2))
        with pytest.raises(ValueError, match="no C matrix"):
            modal_reduction(ss)

    def test_rejects_nonpositive_n_modes(self) -> None:
        """A non-positive n_modes is rejected."""
        with pytest.raises(ValueError, match="n_modes must be positive"):
            modal_reduction(_system(), n_modes=0)

    def test_rejects_empty_band(self) -> None:
        """A band that excludes every mode is rejected."""
        with pytest.raises(ValueError, match="no underdamped modes"):
            modal_reduction(_system(), frequency_band_hz=(20.0, 30.0))

    def test_rejects_conflicting_ranking_arguments(self) -> None:
        """Single- and multi-output ranking cannot both define the objective."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            modal_reduction(
                _system(), rank_output=0, rank_outputs=["TwrBsMyt", "GenSpeed"]
            )

    def test_rejects_empty_or_duplicate_joint_ranking(self) -> None:
        """A joint ranking must contain at least one distinct physical row."""
        with pytest.raises(ValueError, match="at least one output"):
            modal_reduction(_system(), rank_outputs=[])
        with pytest.raises(ValueError, match="distinct output rows"):
            modal_reduction(_system(), rank_outputs=[0, "TwrBsMyt"])

    def test_rejects_joint_outputs_with_no_modal_residue(self) -> None:
        """An all-zero joint objective cannot rank modes meaningfully."""
        system = _system()
        assert system.c is not None
        system.c[:] = 0.0
        with pytest.raises(ValueError, match="zero residue"):
            modal_reduction(system, rank_outputs=[0, 1])

    def test_rejects_inconsistent_required_transfer_configuration(self) -> None:
        """Transfer reservation requires pairs, an input matrix, and modal budget."""
        with pytest.raises(ValueError, match="need n_required_transfer_modes"):
            modal_reduction(_system(), required_transfer_pairs=[("GenSpeed", 0)])
        with pytest.raises(ValueError, match="needs at least one"):
            modal_reduction(_system(), n_required_transfer_modes=1)
        with pytest.raises(ValueError, match="exceeds n_modes"):
            modal_reduction(
                _system(),
                n_modes=1,
                required_transfer_pairs=[("GenSpeed", 0)],
                n_required_transfer_modes=2,
            )
        system = _system()
        system.b = None
        with pytest.raises(ValueError, match="B matrix"):
            modal_reduction(
                system,
                required_transfer_pairs=[("GenSpeed", 0)],
                n_required_transfer_modes=1,
            )


class TestOutputModeResidues:
    """Tests for output_mode_residues."""

    def test_ranks_by_descending_residue(self) -> None:
        """Modes are returned ranked by descending residue in the chosen output."""
        freqs, damping, residues = output_mode_residues(_system(), "TwrBsMyt")
        assert np.all(np.diff(residues) <= 0)
        assert freqs[0] == pytest.approx(0.3, abs=1e-6)
        assert damping[0] == pytest.approx(0.05, abs=1e-6)

    def test_resolves_output_by_index(self) -> None:
        """An output can be selected by integer index."""
        freqs, _, _ = output_mode_residues(_system(), 0)
        assert freqs[0] == pytest.approx(0.3, abs=1e-6)

    def test_rejects_unknown_output(self) -> None:
        """An unresolvable output name is rejected."""
        with pytest.raises(ValueError, match="no output matches"):
            output_mode_residues(_system(), "nonexistent")

    def test_rejects_ambiguous_output_substring(self) -> None:
        """A substring matching several outputs is rejected, not first-matched."""
        ss = _system()
        ss.output_names = ["Bld1RootMxb", "Bld2RootMxb"]
        with pytest.raises(ValueError, match="matches multiple channels"):
            output_mode_residues(ss, "RootMxb")

    def test_rejects_discrete_system(self) -> None:
        """A discrete system is rejected (continuous-time modal metadata only)."""
        ss = _system()
        ss.dt = 0.01
        with pytest.raises(ValueError, match="continuous-time"):
            output_mode_residues(ss, "TwrBsMyt")
