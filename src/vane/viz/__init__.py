# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Visualization — Campbell plots, mode shape plots, damping plots.

Every figure follows one style, the plotting conventions of the Marine Structural
Mechanics and Integrity Lab (SMI Lab), Inha University: Matplotlib figures are drawn
inside :func:`plot_style`, Plotly figures with the registered ``"vane"`` template.
Static figures take a ``size=`` (``"single"``, ``"double"`` or a width in cm), and
:func:`save_figure` writes them for publication.
"""

from vane.viz._style import (
    FAMILY_COLORS,
    FAMILY_TITLES,
    PLOTLY_TEMPLATE,
    STANDARD_COLORS,
    STANDARD_LINES,
    VANE_STYLE,
    ModeFamily,
    ModeLineStyle,
    apply_style,
    mode_family,
    mode_line_style,
    plot_style,
    plotly_template,
    standard_colorway,
    style_rc,
)
from vane.viz.campbell_plot import (
    plot_campbell,
    plot_campbell_panels,
    plot_campbell_static,
)
from vane.viz.damping_plot import plot_damping, plot_damping_static
from vane.viz.diagnostics import (
    plot_azimuth_coverage,
    plot_azimuth_coverage_static,
    plot_branch_review,
    plot_branch_review_static,
    plot_mac_matrix,
    plot_mac_matrix_static,
    plot_tracking_confidence,
    plot_tracking_confidence_static,
)
from vane.viz.modeshape_3d import plot_mode_3d, plot_mode_3d_static
from vane.viz.modeshape_fowt import (
    FowtModeDeformation,
    deform_mode,
    infer_geometry,
    plot_mode_fowt,
    plot_mode_fowt_animation,
    plot_mode_fowt_static,
)
from vane.viz.modeshape_phasor import (
    plot_blade_components,
    plot_blade_components_static,
    plot_mode_phasor,
    plot_mode_phasor_animation,
    plot_mode_phasor_static,
)
from vane.viz.modeshape_static import plot_mode_participation, plot_mode_shape
from vane.viz.palette import (
    OKABE_ITO,
    apply_colorblind_palette,
    apply_standard_palette,
    colorblind_colorway,
)
from vane.viz.publication import (
    SIZE_PRESETS,
    FigureSize,
    add_panel_labels,
    figure_size,
    save_figure,
)
from vane.viz.qoi_plot import plot_load_mode_heatmap, plot_load_mode_heatmap_static
from vane.viz.report import study_report_html, write_report, write_report_pdf
from vane.viz.vtk_export import write_mode_animation_vtk, write_mode_shape_vtk

__all__ = [
    "FAMILY_COLORS",
    "FAMILY_TITLES",
    "OKABE_ITO",
    "PLOTLY_TEMPLATE",
    "SIZE_PRESETS",
    "STANDARD_COLORS",
    "STANDARD_LINES",
    "VANE_STYLE",
    "FigureSize",
    "FowtModeDeformation",
    "ModeFamily",
    "ModeLineStyle",
    "add_panel_labels",
    "apply_colorblind_palette",
    "apply_standard_palette",
    "apply_style",
    "colorblind_colorway",
    "deform_mode",
    "figure_size",
    "infer_geometry",
    "mode_family",
    "mode_line_style",
    "plot_azimuth_coverage",
    "plot_azimuth_coverage_static",
    "plot_blade_components",
    "plot_blade_components_static",
    "plot_branch_review",
    "plot_branch_review_static",
    "plot_campbell",
    "plot_campbell_panels",
    "plot_campbell_static",
    "plot_damping",
    "plot_damping_static",
    "plot_load_mode_heatmap",
    "plot_load_mode_heatmap_static",
    "plot_mac_matrix",
    "plot_mac_matrix_static",
    "plot_mode_3d",
    "plot_mode_3d_static",
    "plot_mode_fowt",
    "plot_mode_fowt_animation",
    "plot_mode_fowt_static",
    "plot_mode_participation",
    "plot_mode_phasor",
    "plot_mode_phasor_animation",
    "plot_mode_phasor_static",
    "plot_mode_shape",
    "plot_style",
    "plot_tracking_confidence",
    "plot_tracking_confidence_static",
    "plotly_template",
    "save_figure",
    "standard_colorway",
    "study_report_html",
    "style_rc",
    "write_mode_animation_vtk",
    "write_mode_shape_vtk",
    "write_report",
    "write_report_pdf",
]
