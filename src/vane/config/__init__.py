# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Configuration — DOF/output index mapping and system type definitions."""

from vane.config.dof_map import DofCategory, DofInfo, classify_dof
from vane.config.output_map import (
    LOAD_CATEGORIES,
    OutputCategory,
    OutputInfo,
    classify_output,
)
from vane.config.system_config import (
    FowtGeometry,
    Member,
    SystemType,
    default_geometry,
    jacket_geometry,
    monopile_geometry,
    onshore_geometry,
    semisub_geometry,
    spar_geometry,
    tlp_geometry,
)

__all__ = [
    "LOAD_CATEGORIES",
    "DofCategory",
    "DofInfo",
    "FowtGeometry",
    "Member",
    "OutputCategory",
    "OutputInfo",
    "SystemType",
    "classify_dof",
    "classify_output",
    "default_geometry",
    "jacket_geometry",
    "monopile_geometry",
    "onshore_geometry",
    "semisub_geometry",
    "spar_geometry",
    "tlp_geometry",
]
