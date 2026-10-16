# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Every public vane.viz plotting function draws in the VANE style, with units."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.io as pio
import pytest
from matplotlib.colors import to_hex
from matplotlib.lines import TICKUP

from tests.test_viz.test_modeshape_fowt import _platform_pitch_mode
from tests.test_viz.test_modeshape_phasor import _blade_solution
from tests.test_viz.test_qoi_plot import _solution as _qoi_solution
from tests.test_viz.test_qoi_plot import _system as _qoi_system
from vane.io.lin_reader import read_lin_file
from vane.study import run_study
from vane.sysid.qoi import build_qoi_map
from vane.viz import (
    PLOTLY_TEMPLATE,
    STANDARD_COLORS,
    plot_azimuth_coverage,
    plot_azimuth_coverage_static,
    plot_blade_components,
    plot_blade_components_static,
    plot_branch_review,
    plot_branch_review_static,
    plot_campbell,
    plot_campbell_panels,
    plot_campbell_static,
    plot_damping,
    plot_damping_static,
    plot_load_mode_heatmap,
    plot_load_mode_heatmap_static,
    plot_mac_matrix,
    plot_mac_matrix_static,
    plot_mode_3d,
    plot_mode_3d_static,
    plot_mode_fowt,
    plot_mode_fowt_animation,
    plot_mode_fowt_static,
    plot_mode_participation,
    plot_mode_phasor,
    plot_mode_phasor_animation,
    plot_mode_phasor_static,
    plot_tracking_confidence,
    plot_tracking_confidence_static,
    study_report_html,
    style_rc,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import plotly.graph_objects as go
    from matplotlib.figure import Figure

    from tests.conftest import LinFileFactory
    from vane.campbell.builder import CampbellDiagram
    from vane.modal.branch import BranchGroup
    from vane.modal.eigensolver import ModalSolution
    from vane.study import StudyResult

_STATES: list[tuple[str, bool, int]] = [
    ("ED 1st tower fore-aft (internal DOF index = DOF_TFA1), m", False, 2),
    (
        "ED First time derivative of 1st tower fore-aft "
        "(internal DOF index = DOF_TFA1), m/s",
        False,
        1,
    ),
]

# rcParams the style sets and that must not leak into the global configuration.
_WATCHED = (
    "axes.prop_cycle",
    "font.family",
    "font.size",
    "xtick.direction",
    "xtick.top",
    "axes.spines.top",
    "lines.linewidth",
    "figure.figsize",
)
_UNIT = re.compile(r"\(.+\)")


@pytest.fixture
def study(make_lin_file: LinFileFactory) -> StudyResult:
    """Return a two-point study of a one-mode system."""
    a = np.array([[0.0, 1.0], [-4.0, -0.2]], dtype=np.float64)
    operating_points = [
        [
            read_lin_file(
                make_lin_file(states=_STATES, a=a, rotor_speed=speed, azimuth=az)
            )
            for az in (0.0, 1.0)
        ]
        for speed in (1.0, 1.5)
    ]
    return run_study(operating_points)


def _uses_vane_template(figure: go.Figure) -> bool:
    """Return whether a Plotly figure carries the vane template."""
    return bool(
        figure.layout.template.to_plotly_json()
        == pio.templates[PLOTLY_TEMPLATE].to_plotly_json()
    )


def _plotly_builders(
    campbell_diagram: CampbellDiagram,
    mode_solution: ModalSolution,
    branch_review_case: tuple[CampbellDiagram, list[BranchGroup]],
) -> dict[str, Callable[..., go.Figure]]:
    """Return a builder per public Plotly plotting function (template as keyword)."""
    diagram, groups = branch_review_case
    qoi_map = build_qoi_map(_qoi_system(), _qoi_solution())
    fowt = _platform_pitch_mode()
    return {
        "campbell": lambda **kw: plot_campbell(campbell_diagram, **kw),
        "damping": lambda **kw: plot_damping(campbell_diagram, **kw),
        "mac": lambda **kw: plot_mac_matrix(mode_solution, mode_solution, **kw),
        "confidence": lambda **kw: plot_tracking_confidence(campbell_diagram, **kw),
        "branch": lambda **kw: plot_branch_review(diagram, groups, **kw),
        "branch_empty": lambda **kw: plot_branch_review(diagram, [], **kw),
        "qoi": lambda **kw: plot_load_mode_heatmap(qoi_map, **kw),
        "mode_3d": lambda **kw: plot_mode_3d(mode_solution, 0, **kw),
        "phasor": lambda **kw: plot_mode_phasor(mode_solution, 0, **kw),
        "phasor_anim": lambda **kw: plot_mode_phasor_animation(
            mode_solution, 0, n_frames=2, **kw
        ),
        "blade": lambda **kw: plot_blade_components(_blade_solution(), 0, **kw),
        "fowt": lambda **kw: plot_mode_fowt(fowt, 0, **kw),
        "fowt_anim": lambda **kw: plot_mode_fowt_animation(fowt, 0, n_frames=2, **kw),
    }


_PLOTLY_NAMES = (
    "campbell",
    "damping",
    "mac",
    "confidence",
    "branch",
    "branch_empty",
    "qoi",
    "mode_3d",
    "phasor",
    "phasor_anim",
    "blade",
    "fowt",
    "fowt_anim",
)

# Numeric axis titles each Plotly figure must label with a unit.
_PLOTLY_UNITS: dict[str, Callable[[go.Figure], list[str]]] = {
    "campbell": lambda f: [f.layout.xaxis.title.text, f.layout.yaxis.title.text],
    "damping": lambda f: [f.layout.xaxis.title.text, f.layout.yaxis.title.text],
    "mac": lambda f: [f.layout.xaxis.title.text, f.layout.yaxis.title.text],
    "confidence": lambda f: [f.layout.yaxis.title.text],
    "branch": lambda f: [f.layout.xaxis.title.text, f.layout.yaxis.title.text],
    "qoi": lambda f: [f.layout.xaxis.title.text],
    "mode_3d": lambda f: [
        f.layout.scene.xaxis.title.text,
        f.layout.scene.yaxis.title.text,
        f.layout.scene.zaxis.title.text,
    ],
    "phasor": lambda f: [f.layout.polar.radialaxis.title.text],
    "blade": lambda f: [f.layout.yaxis.title.text],
    "fowt": lambda f: [
        f.layout.scene.xaxis.title.text,
        f.layout.scene.zaxis.title.text,
    ],
}


@pytest.mark.parametrize("name", _PLOTLY_NAMES)
def test_plotly_function_applies_the_template(
    name: str,
    campbell_diagram: CampbellDiagram,
    mode_solution: ModalSolution,
    branch_review_case: tuple[CampbellDiagram, list[BranchGroup]],
) -> None:
    """Each interactive plot applies vane by default, honours template=, has units."""
    builders = _plotly_builders(campbell_diagram, mode_solution, branch_review_case)
    figure = builders[name]()
    assert _uses_vane_template(figure)
    custom = builders[name](template="plotly_white")
    assert (
        custom.layout.template.to_plotly_json()
        == pio.templates["plotly_white"].to_plotly_json()
    )
    for title in _PLOTLY_UNITS.get(name, lambda f: [])(figure):
        assert title is not None
        assert _UNIT.search(title), title


@pytest.mark.parametrize("name", ["campbell", "damping", "mac", "confidence", "qoi"])
def test_plotly_titles_are_optional(
    name: str,
    campbell_diagram: CampbellDiagram,
    mode_solution: ModalSolution,
    branch_review_case: tuple[CampbellDiagram, list[BranchGroup]],
) -> None:
    """No default title; a title= argument sets one."""
    builders = _plotly_builders(campbell_diagram, mode_solution, branch_review_case)
    assert builders[name]().layout.title.text is None
    assert builders[name](title="Figure").layout.title.text == "Figure"


def test_azimuth_coverage_and_report(study: StudyResult) -> None:
    """The azimuth plot applies the template with units; the report embeds it."""
    figure = plot_azimuth_coverage(study.provenance)
    assert _uses_vane_template(figure)
    assert _UNIT.search(figure.layout.xaxis.title.text)
    assert figure.layout.yaxis.title.text == "Azimuth (deg)"
    report = study_report_html(study)
    assert re.search(r'"colorway":\s*\["#000000"', report)
    assert "Helvetica, Arial" in report
    assert "Mode-tracking confidence" in report


def _snapshot() -> dict[str, Any]:
    """Return the watched global rcParams."""
    return {key: mpl.rcParams[key] for key in _WATCHED}


def _static_builders(
    campbell_diagram: CampbellDiagram,
    mode_solution: ModalSolution,
    branch_review_case: tuple[CampbellDiagram, list[BranchGroup]],
    study: StudyResult,
) -> dict[str, Callable[..., Figure]]:
    """Return a builder per public Matplotlib plotting function."""
    diagram, groups = branch_review_case
    qoi_map = build_qoi_map(_qoi_system(), _qoi_solution())
    fowt = _platform_pitch_mode()
    return {
        "participation": lambda **kw: plot_mode_participation(mode_solution, 0, **kw),
        "campbell": lambda **kw: plot_campbell_static(campbell_diagram, **kw),
        "damping": lambda **kw: plot_damping_static(campbell_diagram, **kw),
        "panels": lambda **kw: plot_campbell_panels(campbell_diagram, **kw),
        "mac": lambda **kw: plot_mac_matrix_static(mode_solution, mode_solution, **kw),
        "confidence": lambda **kw: plot_tracking_confidence_static(
            campbell_diagram, **kw
        ),
        "azimuth": lambda **kw: plot_azimuth_coverage_static(study.provenance, **kw),
        "branch": lambda **kw: plot_branch_review_static(diagram, groups, **kw),
        "qoi": lambda **kw: plot_load_mode_heatmap_static(qoi_map, **kw),
        "phasor": lambda **kw: plot_mode_phasor_static(mode_solution, 0, **kw),
        "blade": lambda **kw: plot_blade_components_static(_blade_solution(), 0, **kw),
        "mode_3d": lambda **kw: plot_mode_3d_static(mode_solution, 0, **kw),
        "fowt": lambda **kw: plot_mode_fowt_static(fowt, 0, **kw),
    }


_STATIC_NAMES = (
    "participation",
    "campbell",
    "damping",
    "panels",
    "mac",
    "confidence",
    "azimuth",
    "branch",
    "qoi",
    "phasor",
    "blade",
    "mode_3d",
    "fowt",
)


def _numeric_labels(figure: Figure) -> list[str]:
    """Return the non-empty axis labels of every axes, and the figure's sup-labels."""
    labels = []
    for axes in figure.axes:
        for getter in ("get_xlabel", "get_ylabel", "get_zlabel"):
            if hasattr(axes, getter):
                text = getattr(axes, getter)()
                if text:
                    labels.append(text)
    labels.extend(
        text for text in (figure.get_supxlabel(), figure.get_supylabel()) if text
    )
    # A polar plot names its radial quantity in a text below the wheel.
    labels.extend(
        text.get_text()
        for axes in figure.axes
        if axes.name == "polar"
        for text in axes.texts
    )
    return labels


@pytest.mark.parametrize("name", _STATIC_NAMES)
@pytest.mark.parametrize("size", ["single", "double"])
def test_static_function_draws_in_the_style(
    name: str,
    size: str,
    campbell_diagram: CampbellDiagram,
    mode_solution: ModalSolution,
    branch_review_case: tuple[CampbellDiagram, list[BranchGroup]],
    study: StudyResult,
) -> None:
    """Each static plot draws in the style at the size, labels units, leaks nothing."""
    builders = _static_builders(
        campbell_diagram, mode_solution, branch_review_case, study
    )
    before = _snapshot()
    with mpl.rc_context({"xtick.direction": "out", "axes.spines.top": False}):
        inner = _snapshot()
        figure = builders[name](size=size)
        assert _snapshot() == inner
    assert _snapshot() == before
    width_cm = figure.get_size_inches()[0] * 2.54
    assert width_cm == pytest.approx(8.5 if size == "single" else 17.0)
    assert to_hex(figure.get_facecolor()) == "#ffffff"
    axes = figure.axes[0]
    if axes.name == "rectilinear":
        assert axes.spines["top"].get_visible()
        assert axes.xaxis.get_major_ticks()[0].tick1line.get_marker() == TICKUP
    labels = _numeric_labels(figure)
    assert labels
    unit_free = {"Load channel", "Multi-blade degree of freedom"}
    assert any(_UNIT.search(label) for label in labels if label not in unit_free)
    assert axes.xaxis.label.get_fontfamily() == style_rc()["font.family"]
    plt.close(figure)


@pytest.mark.parametrize("name", ["participation", "campbell", "damping", "mac"])
def test_static_function_honours_the_caller_style(
    name: str,
    campbell_diagram: CampbellDiagram,
    mode_solution: ModalSolution,
    branch_review_case: tuple[CampbellDiagram, list[BranchGroup]],
    study: StudyResult,
) -> None:
    """style=None keeps the caller's rcParams; a dictionary is applied."""
    builder = _static_builders(
        campbell_diagram, mode_solution, branch_review_case, study
    )[name]
    with mpl.rc_context({"xtick.direction": "out", "axes.spines.top": False}):
        figure = builder(style=None)
    axes = figure.axes[0]
    assert not axes.spines["top"].get_visible()
    assert axes.xaxis.get_major_ticks()[0].tick1line.get_marker() != TICKUP
    plt.close(figure)
    figure = builder(style={"axes.spines.right": False})
    assert not figure.axes[0].spines["right"].get_visible()
    plt.close(figure)


def test_participation_bar_colour_and_legend(mode_solution: ModalSolution) -> None:
    """The bars are blue, named in the legend with the mode and its frequency."""
    figure = plot_mode_participation(mode_solution, 0)
    axes = figure.axes[0]
    assert to_hex(axes.patches[0].get_facecolor()) == STANDARD_COLORS[2]
    legend = axes.get_legend()
    assert legend is not None
    assert "Hz" in legend.get_texts()[0].get_text()
    assert axes.get_title() == ""
    plt.close(figure)
    full = plot_mode_participation(mode_solution, 0, title="T")
    labels = [tick.get_text() for tick in full.axes[0].get_yticklabels()]
    assert any("DOF_TFA1" in label for label in labels)
    short = plot_mode_participation(mode_solution, 0, short_names=True)
    names = [tick.get_text() for tick in short.axes[0].get_yticklabels()]
    assert "Tower FA1" in names
    plt.close(short)
    assert full.axes[0].get_title() == "T"
    plt.close(full)
