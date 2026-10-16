# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.sysid.rom_family — operating-point ROM family with mode tracking."""

import logging
from typing import Any

import numpy as np
import pytest
from scipy.linalg import block_diag

from vane.sysid import (
    RomFamily,
    StateSpace,
    build_rom_family,
    combine_rom_families,
    schedule_model,
    static_gain,
)


def _modal_block(freq: float, zeta: float) -> np.ndarray:
    """Return a stable continuous 2x2 modal block at ``freq`` Hz."""
    wn = 2.0 * np.pi * freq
    sigma = -zeta * wn
    wd = wn * np.sqrt(1.0 - zeta**2)
    return np.array([[sigma, -wd], [wd, sigma]])


def _system(freq_a: float, freq_b: float, res_a: float, res_b: float) -> StateSpace:
    """Two modes: A shows in output 0, B in output 1, with the given residues."""
    a = np.asarray(block_diag(_modal_block(freq_a, 0.05), _modal_block(freq_b, 0.05)))
    c = np.zeros((3, 4))
    c[0, 0] = res_a  # mode A -> output 0
    c[1, 2] = res_b  # mode B -> output 1
    c[2, 0], c[2, 2] = 0.3 * res_a, 0.3 * res_b
    return StateSpace(a=a, c=c, output_names=["o0", "o1", "o2"])


def _three_mode_system(
    freqs: tuple[float, float, float], residues: tuple[float, float, float]
) -> StateSpace:
    """Three modes A/B/C in outputs 0/1/2 with the given frequencies and residues."""
    a = np.asarray(
        block_diag(*(_modal_block(f, 0.05) for f in freqs)),
    )
    c = np.zeros((3, 6))
    for mode, res in enumerate(residues):
        c[mode, 2 * mode] = res
    return StateSpace(a=a, c=c, output_names=["o0", "o1", "o2"])


class TestBuildRomFamily:
    """Tests for build_rom_family."""

    def test_family_reserves_and_tracks_required_transfer_mode(self) -> None:
        """The observer mode reserved at the reference remains a tracked branch."""
        systems = [
            _three_mode_system((0.3, 1.5, 2.5), (1.0, 0.8, 0.1)),
            _three_mode_system((0.4, 1.4, 2.6), (0.9, 0.7, 0.1)),
        ]
        for system in systems:
            system.b = np.zeros((6, 1))
            system.b[4, 0] = 1.0
            system.input_names = ["unknown torque"]
        family = build_rom_family(
            systems,
            [8.0, 10.0],
            n_modes=2,
            rank_output="o0",
            required_transfer_pairs=[("o2", "unknown torque")],
            n_required_transfer_modes=1,
        )
        np.testing.assert_allclose(family.frequencies_hz[:, 0], [2.5, 2.6], atol=1e-6)
        np.testing.assert_allclose(family.frequencies_hz[:, 1], [0.3, 0.4], atol=1e-6)

    def test_dc_gain_match_is_applied_after_tracked_truncation(self) -> None:
        """The stored one-mode family, not the all-mode intermediate, closes DC."""
        systems = [
            _system(0.3, 1.5, 1.0, 0.8),
            _system(0.4, 1.6, 0.9, 0.7),
        ]
        for system in systems:
            system.b = np.ones((4, 1))
            system.d = np.zeros((3, 1))
            system.input_names = ["load"]
        family = build_rom_family(
            systems,
            [8.0, 10.0],
            n_modes=1,
            rank_output="o0",
            match_dc_gain=True,
        )
        for full, reduced in zip(systems, family.systems, strict=True):
            np.testing.assert_allclose(static_gain(reduced), static_gain(full))

    def test_transfer_partition_prevents_output_mac_substitution(self) -> None:
        """A co-shaped but uncontrollable mode cannot replace an observer mode."""

        def system(
            structural: float,
            first: float,
            second: float,
            driven_mode: int,
        ) -> StateSpace:
            a = np.asarray(
                block_diag(
                    _modal_block(structural, 0.05),
                    _modal_block(first, 0.05),
                    _modal_block(second, 0.05),
                )
            )
            b = np.zeros((6, 1))
            b[2 * driven_mode, 0] = 1.0
            c = np.zeros((2, 6))
            c[0, 0] = 1.0
            c[1, 2] = 1.0
            c[1, 4] = 1.0
            return StateSpace(
                a=a,
                b=b,
                c=c,
                input_names=["unknown torque"],
                output_names=["load", "rotor speed"],
            )

        reference = system(0.3, 1.0, 2.0, driven_mode=2)
        later = system(0.4, 1.1, 2.1, driven_mode=1)
        family = build_rom_family(
            [reference, later],
            [8.0, 10.0],
            n_modes=2,
            rank_output="load",
            required_transfer_pairs=[("rotor speed", "unknown torque")],
            n_required_transfer_modes=1,
        )
        np.testing.assert_allclose(family.frequencies_hz[:, 0], [2.0, 1.1], atol=1e-6)
        np.testing.assert_allclose(family.frequencies_hz[:, 1], [0.3, 0.4], atol=1e-6)

    def test_reserved_mode_keeps_identity_when_transfer_ranking_swaps(self) -> None:
        """A reshuffled transfer ranking cannot pair two different physical modes.

        Both points share the same A and output shapes; only the unknown-input gains of
        the 2 rad/s and 5 rad/s modes swap, so the transfer-strongest mode changes. The
        reserved slot was fixed at the reference and must follow its mode by identity.
        """

        def system(gain_a: float, gain_b: float) -> StateSpace:
            a = np.zeros((6, 6))
            for k, omega in enumerate((2.0, 5.0, 9.0)):
                a[2 * k : 2 * k + 2, 2 * k : 2 * k + 2] = [
                    [-0.05, -omega],
                    [omega, -0.05],
                ]
            b = np.zeros((6, 2))
            b[0, 0], b[2, 0], b[4, 0] = gain_a, gain_b, 0.01
            b[0, 1] = b[2, 1] = b[4, 1] = 1.0
            c = np.zeros((3, 6))
            c[0, 0] = c[0, 2] = c[0, 4] = 1.0
            c[1, 0] = c[2, 2] = 1.0
            c[1, 4] = c[2, 4] = 0.5
            return StateSpace(a=a, b=b, c=c)

        family = build_rom_family(
            [system(1.0, 0.2), system(0.2, 1.0)],
            [1.0, 2.0],
            n_modes=2,
            required_transfer_pairs=[(0, 0)],
            n_required_transfer_modes=1,
        )
        np.testing.assert_allclose(
            family.frequencies_hz[1], family.frequencies_hz[0], rtol=1e-12
        )
        np.testing.assert_allclose(family.tracking_mac[1], 1.0, atol=1e-9)

    def test_blend_of_similarity_transformed_members_is_exact(self) -> None:
        """Modal scale is fixed, so blending equivalent members keeps the response.

        The middle member is a random similarity transform of the same 5-DOF chain, so
        its eigenvectors (and hence unnormalized modal B/C scales) differ. Every blend
        must reproduce the common frequency response.
        """
        n_dof, k = 5, 1.0e3
        stiffness = np.zeros((n_dof, n_dof))
        for i in range(n_dof):
            stiffness[i, i] += 2.0 * k * (1.0 + 0.3 * i)
            if i + 1 < n_dof:
                stiffness[i, i + 1] -= k
                stiffness[i + 1, i] -= k
        damping = 0.5 / k * stiffness * 0.01 + 0.05 * np.eye(n_dof)
        a = np.block(
            [[np.zeros((n_dof, n_dof)), np.eye(n_dof)], [-stiffness, -damping]]
        )
        b = np.zeros((2 * n_dof, 1))
        b[-1, 0] = 1.0
        c = np.zeros((2, 2 * n_dof))
        c[0, 0] = 1.0
        c[1, n_dof - 1] = k
        chain = StateSpace(a=a, b=b, c=c, d=np.zeros((2, 1)))
        transform = np.random.default_rng(3).normal(size=(2 * n_dof, 2 * n_dof))
        similar = StateSpace(
            a=np.asarray(np.linalg.solve(transform, a @ transform), dtype=np.float64),
            b=np.asarray(np.linalg.solve(transform, b), dtype=np.float64),
            c=c @ transform,
            d=np.zeros((2, 1)),
        )
        family = build_rom_family([chain, similar, chain], [0.0, 1.0, 2.0], n_modes=5)

        def response(system: StateSpace, omega: float) -> np.ndarray:
            assert system.b is not None and system.c is not None
            identity = np.eye(system.n_states)
            return np.asarray(
                system.c @ np.linalg.solve(1j * omega * identity - system.a, system.b)
            )

        omegas = np.linspace(0.5, 60.0, 200)
        truth = np.array([response(chain, w) for w in omegas])
        for point in (0.5, 1.5):
            blended = schedule_model(family, point).system
            model = np.array([response(blended, w) for w in omegas])
            error = np.abs(model - truth).max() / np.abs(truth).max()
            assert error < 1e-8

    def test_combines_disjoint_modal_families_without_double_feedthrough(self) -> None:
        """Composition preserves both tracked blocks and counts shared D only once."""
        base_systems = [
            _system(0.3, 1.5, 1.0, 0.1),
            _system(0.4, 1.6, 1.0, 0.1),
        ]
        enrichment_systems = [
            _system(2.0, 3.0, 1.0, 0.1),
            _system(2.1, 3.1, 1.0, 0.1),
        ]
        for system in [*base_systems, *enrichment_systems]:
            system.d = np.ones((3, 1))
        base = build_rom_family(base_systems, [8.0, 10.0], n_modes=1, rank_output="o0")
        enrichment = build_rom_family(
            enrichment_systems, [8.0, 10.0], n_modes=1, rank_output="o0"
        )
        combined = combine_rom_families(base, enrichment)
        np.testing.assert_allclose(combined.frequencies_hz, [[0.3, 2.0], [0.4, 2.1]])
        assert combined.systems[0].n_states == 4
        assert combined.systems[0].d is not None
        np.testing.assert_array_equal(combined.systems[0].d, np.ones((3, 1)))
        np.testing.assert_allclose(combined.systems[0].a[:2, 2:], 0.0)
        np.testing.assert_allclose(combined.systems[0].a[2:, :2], 0.0)

    def test_combination_rejects_duplicate_modal_branch(self) -> None:
        """A branch already present in the base cannot be counted twice."""
        family = build_rom_family(
            [_system(0.3, 1.5, 1.0, 0.4)], [8.0], n_modes=1, rank_output="o0"
        )
        with pytest.raises(ValueError, match="duplicates a base mode"):
            combine_rom_families(family, family)

    def test_combination_can_restrict_enrichment_to_observer_output(self) -> None:
        """A separated observer block contributes only to its declared sensor row."""
        base = build_rom_family(
            [_system(0.3, 1.5, 1.0, 0.4)], [8.0], n_modes=1, rank_output="o0"
        )
        enrichment = build_rom_family(
            [_system(2.0, 3.0, 0.2, 1.0)], [8.0], n_modes=1, rank_output="o1"
        )
        combined = combine_rom_families(
            base,
            enrichment,
            enrichment_outputs=["o1"],
        )
        assert combined.systems[0].c is not None
        np.testing.assert_allclose(combined.systems[0].c[[0, 2], 2:], 0.0)
        assert np.linalg.norm(combined.systems[0].c[1, 2:]) > 0.0

    def test_tracks_mode_identity_across_a_crossing(self) -> None:
        """A tracked index stays the same physical mode even when the ranking flips.

        At point 1 mode B has the larger residue, so a standalone reduction would order
        the modes oppositely; tracking by shape must keep mode 0 the output-0 mode.
        """
        systems = [
            _system(0.3, 1.5, 1.0, 0.4),
            _system(0.9, 0.85, 0.4, 1.0),  # residue ranking flips here
            _system(1.5, 0.3, 0.3, 1.0),
        ]
        family = build_rom_family(systems, [8.0, 10.0, 12.0], n_modes=2)
        # Mode 0 is the output-0 mode at every point: its residue is in output 0, not 1.
        for system in family.systems:
            assert system.c is not None
            residue0 = np.abs(system.c[:, 0] - 1j * system.c[:, 1])
            assert residue0[0] > residue0[1]  # output 0 dominates for tracked mode 0
        # Its frequency rises monotonically (0.3 -> 0.9 -> 1.5) through the crossing.
        np.testing.assert_allclose(
            family.frequencies_hz[:, 0], [0.3, 0.9, 1.5], atol=1e-6
        )

    def test_tracks_a_mode_that_leaves_the_top_n_modes(self) -> None:
        """A tracked mode is recovered even when it drops below the top-n_modes rank.

        Tracking the two strongest modes of the reference: at the second point a third
        mode becomes dominant, so mode B falls to residue rank 3. Reducing to all
        candidates (not just the top 2) keeps B available, so it stays tracked.
        """
        reference = _three_mode_system((0.3, 1.5, 2.5), (1.0, 0.9, 0.1))
        # Mode C (output 2) now dominates; B (output 1) drops to the weakest residue.
        later = _three_mode_system((0.35, 1.55, 2.55), (0.9, 0.1, 1.0))
        family = build_rom_family([reference, later], [8.0, 10.0], n_modes=2)
        # Tracked modes 0 and 1 are A (output 0) and B (output 1) at BOTH points.
        for system in family.systems:
            assert system.c is not None
            res0 = np.abs(system.c[:, 0] - 1j * system.c[:, 1])
            res1 = np.abs(system.c[:, 2] - 1j * system.c[:, 3])
            assert np.argmax(res0) == 0  # mode 0 lives in output 0
            assert np.argmax(res1) == 1  # mode 1 lives in output 1 (recovered, not C)
        np.testing.assert_allclose(family.frequencies_hz[:, 1], [1.5, 1.55], atol=1e-6)

    def test_modal_phase_is_aligned_across_points(self) -> None:
        """Consecutive tracked modes share a phase, so B/C vary continuously.

        The eigenvector phase from ``np.linalg.eig`` is arbitrary, so a tracked mode's
        residue can rotate between points; alignment makes the complex correlation of
        consecutive residues real (no sign/rotation flip in the stored realization).
        """
        systems = [
            _system(0.3, 1.5, 1.0, 0.4),
            _system(0.9, 0.85, 0.4, 1.0),
            _system(1.5, 0.3, 0.3, 1.0),
        ]
        family = build_rom_family(systems, [8.0, 10.0, 12.0], n_modes=2)

        def residues(system: StateSpace) -> np.ndarray:
            assert system.c is not None
            return np.stack(
                [system.c[:, 2 * k] - 1j * system.c[:, 2 * k + 1] for k in range(2)],
                axis=1,
            )

        for prev, curr in zip(family.systems, family.systems[1:], strict=False):
            corr = np.einsum("ik,ik->k", residues(prev).conj(), residues(curr))
            np.testing.assert_allclose(corr.imag, 0.0, atol=1e-9)  # phase aligned

    def test_shapes_and_reference_row(self) -> None:
        """The arrays are (n_points, n_modes) and the first tracking row is all ones."""
        systems = [_system(0.3, 1.5, 1.0, 0.5), _system(0.4, 1.4, 1.0, 0.5)]
        family = build_rom_family(systems, [8.0, 10.0], n_modes=2)
        assert family.frequencies_hz.shape == (2, 2)
        assert family.damping_ratios.shape == (2, 2)
        assert family.tracking_mac.shape == (2, 2)
        np.testing.assert_array_equal(family.tracking_mac[0], [1.0, 1.0])
        assert np.all((family.tracking_mac >= 0.0) & (family.tracking_mac <= 1.0))
        np.testing.assert_array_equal(family.operating_points, [8.0, 10.0])

    def test_reduced_systems_preserve_eigenvalues(self) -> None:
        """Each tracked model's eigenvalue frequencies match the reported ones."""
        systems = [_system(0.3, 1.5, 1.0, 0.5), _system(0.4, 1.4, 1.0, 0.5)]
        family = build_rom_family(systems, [8.0, 10.0], n_modes=2)
        for system, freqs in zip(family.systems, family.frequencies_hz, strict=True):
            eigvals = np.linalg.eigvals(system.a)
            pair = eigvals[eigvals.imag > 0]  # one eigenvalue per conjugate pair
            eig_freqs = np.sort(np.abs(pair) / (2.0 * np.pi))
            np.testing.assert_allclose(eig_freqs, np.sort(freqs), atol=1e-6)

    def test_single_point_family(self) -> None:
        """A one-point family is the plain reduction with a unit tracking row."""
        family = build_rom_family([_system(0.3, 1.5, 1.0, 0.5)], [8.0], n_modes=2)
        assert len(family.systems) == 1
        np.testing.assert_array_equal(family.tracking_mac, [[1.0, 1.0]])

    def test_rejects_empty_systems(self) -> None:
        """An empty family is rejected."""
        with pytest.raises(ValueError, match="at least one operating point"):
            build_rom_family([], [], n_modes=2)

    def test_rejects_length_mismatch(self) -> None:
        """Mismatched systems and operating points are rejected."""
        with pytest.raises(ValueError, match="same length"):
            build_rom_family([_system(0.3, 1.5, 1.0, 0.5)], [8.0, 10.0], n_modes=2)

    def test_rejects_nonpositive_n_modes(self) -> None:
        """A non-positive mode count is rejected."""
        with pytest.raises(ValueError, match="n_modes must be positive"):
            build_rom_family([_system(0.3, 1.5, 1.0, 0.5)], [8.0], n_modes=0)

    def test_rejects_out_of_range_frequency_weight(self) -> None:
        """A frequency_weight outside [0, 1] is rejected (it changes the semantics)."""
        with pytest.raises(ValueError, match="frequency_weight must be in"):
            build_rom_family(
                [_system(0.3, 1.5, 1.0, 0.5)], [8.0], n_modes=2, frequency_weight=2.0
            )

    def test_rejects_too_few_modes_at_a_point(self) -> None:
        """A point with fewer underdamped modes than requested is rejected."""
        systems = [_system(0.3, 1.5, 1.0, 0.5)]  # only 2 modes available
        with pytest.raises(ValueError, match="fewer than n_modes"):
            build_rom_family(systems, [8.0], n_modes=3)

    def test_rejects_mismatched_output_count(self) -> None:
        """Systems with different numbers of outputs cannot be tracked together."""
        good = _system(0.3, 1.5, 1.0, 0.5)
        assert good.c is not None
        odd = StateSpace(a=good.a, c=good.c[:2], output_names=["o0", "o1"])
        with pytest.raises(ValueError, match="same number of outputs"):
            build_rom_family([good, odd], [8.0, 10.0], n_modes=2)

    def test_rejects_mismatched_output_names(self) -> None:
        """Systems whose output layout differs are rejected."""
        good = _system(0.3, 1.5, 1.0, 0.5)
        relabelled = StateSpace(a=good.a, c=good.c, output_names=["x0", "x1", "x2"])
        with pytest.raises(ValueError, match="same output layout"):
            build_rom_family([good, relabelled], [8.0, 10.0], n_modes=2)

    def test_rejects_mixed_named_and_unnamed_outputs(self) -> None:
        """A family mixing a named system with an unnamed one is rejected.

        Output names define the output layout; if one system labels its outputs and
        another does not, the shared layout cannot be confirmed.
        """
        named = _system(0.3, 1.5, 1.0, 0.5)
        unnamed = StateSpace(a=named.a, c=named.c)  # no output_names
        with pytest.raises(ValueError, match="same output layout"):
            build_rom_family([named, unnamed], [8.0, 10.0], n_modes=2)

    def test_rejects_mismatched_output_mbc_coordinates(self) -> None:
        """Same output names but different MBC coordinates are not the same outputs.

        An MBC-derived system (collective/cosine/sine rows) and a plain-output system
        can share output names yet label different physical rows, so row-by-row residue
        matching would compare unlike outputs.
        """
        plain = _system(0.3, 1.5, 1.0, 0.5)
        assert plain.c is not None
        mbc = StateSpace(
            a=plain.a,
            c=plain.c,
            output_names=plain.output_names,
            output_mbc_coordinates=["collective", "cosine", "sine"],
        )
        with pytest.raises(ValueError, match="output MBC coordinates"):
            build_rom_family([plain, mbc], [8.0, 10.0], n_modes=2)

    def test_rejects_missing_c_matrix(self) -> None:
        """A system without a C matrix has no output residues to track."""
        good = _system(0.3, 1.5, 1.0, 0.5)
        no_c = StateSpace(a=good.a)
        with pytest.raises(ValueError, match="no C matrix"):
            build_rom_family([good, no_c], [8.0, 10.0], n_modes=2)

    def test_rejects_mismatched_input_dimension(self) -> None:
        """A family mixing B=None (zero inputs) with an input matrix is rejected.

        StateSpace treats ``b=None`` as zero inputs, so an interpolated family mixing
        the two would have inconsistent input dimensions.
        """
        good = _system(0.3, 1.5, 1.0, 0.5)  # b is None -> zero inputs
        with_input = StateSpace(
            a=good.a,
            b=np.ones((good.a.shape[0], 2)),
            c=good.c,
            output_names=good.output_names,
        )
        with pytest.raises(ValueError, match="same number of inputs"):
            build_rom_family([good, with_input], [8.0, 10.0], n_modes=2)

    def test_rejects_mismatched_feedthrough_input_dimension(self) -> None:
        """Feedthrough-only models with different D widths are inconsistent.

        With ``B`` absent the input dimension comes from ``D`` (matching ``write_rom``),
        so a family with ``D`` of width 1 at one point and 2 at another is rejected.
        """
        good = _system(0.3, 1.5, 1.0, 0.5)
        assert good.c is not None
        d1 = StateSpace(
            a=good.a, c=good.c, d=np.zeros((3, 1)), output_names=good.output_names
        )
        d2 = StateSpace(
            a=good.a, c=good.c, d=np.zeros((3, 2)), output_names=good.output_names
        )
        with pytest.raises(ValueError, match="same number of inputs"):
            build_rom_family([d1, d2], [8.0, 10.0], n_modes=2)

    def test_rejects_mismatched_time_domain(self) -> None:
        """Mixing continuous and discrete systems is rejected."""
        good = _system(0.3, 1.5, 1.0, 0.5)
        discrete = StateSpace(
            a=good.a, c=good.c, output_names=good.output_names, dt=0.01
        )
        with pytest.raises(ValueError, match="same time domain"):
            build_rom_family([good, discrete], [8.0, 10.0], n_modes=2)

    def test_rejects_non_monotonic_schedule(self) -> None:
        """A non-monotonic (or repeated) scheduling parameter is rejected."""
        systems = [_system(0.3, 1.5, 1.0, 0.5) for _ in range(3)]
        with pytest.raises(ValueError, match="strictly monotonic"):
            build_rom_family(systems, [8.0, 12.0, 10.0], n_modes=2)

    def test_rejects_non_finite_schedule(self) -> None:
        """A non-finite scheduling value is rejected."""
        systems = [_system(0.3, 1.5, 1.0, 0.5) for _ in range(2)]
        with pytest.raises(ValueError, match="must all be finite"):
            build_rom_family(systems, [8.0, np.inf], n_modes=2)

    def test_rejects_two_dimensional_schedule(self) -> None:
        """A non-1-D scheduling array is rejected before the monotonic check."""
        systems = [_system(0.3, 1.5, 1.0, 0.5) for _ in range(2)]
        # An Any-typed value passes the 2-D array through the Sequence[float] type.
        two_d: Any = np.array([[8.0], [10.0]])
        with pytest.raises(ValueError, match="1-D sequence"):
            build_rom_family(systems, two_d, n_modes=2)

    def test_reports_physical_damping_of_reflected_unstable_modes(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An unstable mode keeps its negative damping; the reflection is flagged.

        The 5 rad/s mode's real part runs -0.1, +0.05, +0.2 across three points. The
        stored realizations are stabilized (default), but the reported damping is the
        physical one, the mask marks the two reflected entries, and a warning is logged.
        """

        def two_mode(sigma: float) -> StateSpace:
            a = np.zeros((4, 4))
            for k, lam in enumerate((complex(-0.05, 2.0), complex(sigma, 5.0))):
                a[2 * k : 2 * k + 2, 2 * k : 2 * k + 2] = [
                    [lam.real, -lam.imag],
                    [lam.imag, lam.real],
                ]
            c = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
            return StateSpace(a=a, b=np.ones((4, 1)), c=c)

        sigmas = (-0.1, 0.05, 0.2)
        with caplog.at_level(logging.WARNING, logger="vane.sysid.rom_family"):
            family = build_rom_family(
                [two_mode(s) for s in sigmas], [5.0, 10.0, 15.0], n_modes=2
            )
        fast = int(np.argmin(np.abs(family.frequencies_hz[0] - 5.0 / (2 * np.pi))))
        expected = [-s / abs(complex(s, 5.0)) for s in sigmas]
        np.testing.assert_allclose(family.damping_ratios[:, fast], expected, rtol=1e-9)
        assert family.damping_ratios[1, fast] < 0.0
        assert family.damping_ratios[2, fast] < 0.0
        np.testing.assert_array_equal(family.stabilized[:, fast], [False, True, True])
        np.testing.assert_array_equal(family.stabilized[:, 1 - fast], False)
        # The stored realizations are still stable (default stabilize=True).
        for system in family.systems:
            assert np.linalg.eigvals(system.a).real.max() < 0.0
        assert "reflected 2 unstable tracked mode" in caplog.text

    def test_stabilize_false_keeps_unstable_poles(self) -> None:
        """With stabilize=False the realization keeps the growing pole, unflagged."""
        a = np.asarray(
            block_diag(_modal_block(0.3, 0.05), np.array([[0.2, -5.0], [5.0, 0.2]]))
        )
        system = StateSpace(a=a, b=np.ones((4, 1)), c=np.ones((1, 4)))
        family = build_rom_family([system], [5.0], n_modes=2, stabilize=False)
        assert np.linalg.eigvals(family.systems[0].a).real.max() > 0.0
        assert family.damping_ratios.min() < 0.0
        assert not np.any(family.stabilized)

    def test_hand_built_family_defaults_to_unstabilized_mask(self) -> None:
        """A RomFamily constructed without a mask gets an all-False one."""
        family = RomFamily(
            operating_points=np.array([1.0, 2.0]),
            systems=[],
            frequencies_hz=np.ones((2, 3)),
            damping_ratios=np.ones((2, 3)),
            tracking_mac=np.ones((2, 3)),
        )
        assert family.stabilized.shape == (2, 3)
        assert family.stabilized.dtype == np.bool_
        assert not np.any(family.stabilized)

    def test_rejects_mismatched_input_names(self) -> None:
        """Systems with the same input count but different input layout are rejected."""
        good = _system(0.3, 1.5, 1.0, 0.5)
        assert good.c is not None
        b = np.ones((good.a.shape[0], 1))
        named = StateSpace(
            a=good.a, b=b, c=good.c, output_names=good.output_names, input_names=["u0"]
        )
        unnamed = StateSpace(a=good.a, b=b, c=good.c, output_names=good.output_names)
        with pytest.raises(ValueError, match="same input layout"):
            build_rom_family([named, unnamed], [8.0, 10.0], n_modes=2)
