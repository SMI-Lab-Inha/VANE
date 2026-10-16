# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Report the VANE version for provenance records."""

from __future__ import annotations


def vane_version() -> str:
    """Return the VANE version string recorded in provenance metadata.

    Returns
    -------
    str
        The package version, ``vane.__version__``.
    """
    from vane import __version__

    return __version__
