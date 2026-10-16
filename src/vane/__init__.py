# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""VANE — Visualization and Automated Natural-frequency Extraction.

Automated OpenFAST linearization post-processing, physics-aware mode
identification, and state-space export for digital twin development of
fixed-bottom and floating offshore wind turbines.
"""

from vane.pipeline import ModalPipeline, PipelineResult
from vane.plugins import Labeler, Tracker
from vane.study import (
    Provenance,
    StudyResult,
    discover_operating_points,
    run_study,
)

# The release version. ``pyproject.toml``, ``docs/conf.py``, ``CITATION.cff`` and
# ``CHANGELOG.md`` carry the same number; ``tests/test_version.py`` checks this.
__version__ = "0.1.0"

__all__ = [
    "Labeler",
    "ModalPipeline",
    "PipelineResult",
    "Provenance",
    "StudyResult",
    "Tracker",
    "__version__",
    "discover_operating_points",
    "run_study",
]
