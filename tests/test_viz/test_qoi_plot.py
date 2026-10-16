# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.viz.qoi_plot — the load-versus-mode heatmap."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from vane.modal.eigensolver import ModalSolution
from vane.sysid.qoi import build_qoi_map
from vane.sysid.state_space import StateSpace
from vane.viz.qoi_plot import plot_load_mode_heatmap

_OUTPUTS = ["MAP T[1], [N]", "ED TwrBsMyt, (kN-m)"]


def _system() -> StateSpace:
    """Return a 3-state system with two load outputs."""
    c = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], dtype=np.float64)
    return StateSpace(
        a=np.eye(3),
        c=c,
        state_names=["s0", "s1", "s2"],
        output_names=_OUTPUTS,
    )


def _solution() -> ModalSolution:
    """Return a three-mode solution spanning the 3 states (one near-dormant)."""
    phi = np.array(
        [[1.0, 0.0, 0.01], [0.0, 1.0, 0.0], [0.5, 0.0, 0.0]],
        dtype=np.complex128,
    )
    freq = np.array([0.1, 0.8, 2.0], dtype=np.float64)
    return ModalSolution(
        eigenvalues=(-0.01 + 1j * freq).astype(np.complex128),
        natural_frequencies_hz=freq,
        damping_ratios=np.full(3, 0.03),
        damped_frequencies_hz=freq,
        mode_shapes=phi,
        full_eigenvectors=phi,
        n_rigid_body_modes=0,
        dof_descriptions=["s0", "s1", "s2"],
    )


class TestPlotLoadModeHeatmap:
    """Tests for plot_load_mode_heatmap."""

    def test_returns_heatmap_figure(self) -> None:
        """The plot returns a single heatmap trace over loads and modes."""
        figure = plot_load_mode_heatmap(build_qoi_map(_system(), _solution()))
        assert isinstance(figure, go.Figure)
        assert figure.data[0].type == "heatmap"
        # Two load rows.
        assert figure.data[0].z.shape[0] == 2

    def test_min_normalized_drops_dormant_modes(self) -> None:
        """A high threshold prunes weakly contributing modes from the columns."""
        qoi = build_qoi_map(_system(), _solution())
        full = plot_load_mode_heatmap(qoi, min_normalized=0.0)
        pruned = plot_load_mode_heatmap(qoi, min_normalized=0.5)
        assert pruned.data[0].z.shape[1] < full.data[0].z.shape[1]

    def test_zero_threshold_keeps_fully_dormant_modes(self) -> None:
        """min_normalized=0 keeps every mode, even one with zero contribution."""
        # A fourth mode with an eigenvector orthogonal to every load row: it
        # contributes exactly zero, yet the zero-threshold escape hatch keeps it.
        phi = np.array(
            [
                [1.0, 0.0, 0.01, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0],
            ],
            dtype=np.complex128,
        )
        freq = np.array([0.1, 0.8, 2.0, 3.0], dtype=np.float64)
        solution = ModalSolution(
            eigenvalues=(-0.01 + 1j * freq).astype(np.complex128),
            natural_frequencies_hz=freq,
            damping_ratios=np.full(4, 0.03),
            damped_frequencies_hz=freq,
            mode_shapes=phi,
            full_eigenvectors=phi,
            n_rigid_body_modes=0,
            dof_descriptions=["s0", "s1", "s2"],
        )
        figure = plot_load_mode_heatmap(
            build_qoi_map(_system(), solution), min_normalized=0.0
        )
        assert figure.data[0].z.shape[1] == 4

    def test_duplicate_load_names_keep_separate_rows(self) -> None:
        """Two loads sharing a name are drawn as two rows, not merged into one."""
        names = ["ED TwrBsMyt, (kN-m)", "ED TwrBsMyt, (kN-m)", "ED RootMyc1, (kN-m)"]
        system = StateSpace(
            a=np.eye(3),
            c=np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]]),
            state_names=["s0", "s1", "s2"],
            output_names=names,
        )
        figure = plot_load_mode_heatmap(build_qoi_map(system, _solution()))
        heatmap = figure.data[0]
        assert heatmap.z.shape[0] == 3
        # Distinct integer row positions, labelled with the load names.
        assert list(heatmap.y) == [0, 1, 2]
        assert len(set(heatmap.y)) == 3
        assert list(figure.layout.yaxis.tickvals) == [0, 1, 2]
        # The tick labels drop the module prefix: "TwrBsMyt (kN-m)".
        tidy = ["TwrBsMyt (kN-m)", "TwrBsMyt (kN-m)", "RootMyc1 (kN-m)"]
        assert list(figure.layout.yaxis.ticktext) == tidy
        # The hover still names each row's load.
        assert [row[0][0] for row in heatmap.customdata] == tidy

    def test_no_loads_raises(self) -> None:
        """A map with no load channels cannot be plotted."""
        system = StateSpace(
            a=np.eye(1), c=np.ones((1, 1)), output_names=["ED PtfmSurge, (m)"]
        )
        qoi = build_qoi_map(system, loads_only=False)
        with pytest.raises(ValueError, match="no load channels"):
            plot_load_mode_heatmap(qoi)

    def test_no_residues_raises(self) -> None:
        """A map built without a modal solution has no residues to plot."""
        with pytest.raises(ValueError, match="no modal residues"):
            plot_load_mode_heatmap(build_qoi_map(_system()))
