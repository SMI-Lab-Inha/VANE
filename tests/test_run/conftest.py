# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Shared fixtures for the ``vane.run`` tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from vane.run import steady_state

if TYPE_CHECKING:
    from pathlib import Path


def pytest_configure(config: pytest.Config) -> None:
    """Register the marker that keeps the real controller-library load check."""
    config.addinivalue_line(
        "markers",
        "real_library_load: run the real child-process controller-library load check",
    )


@pytest.fixture(autouse=True)
def _libraries_load(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Treat placeholder controller-library files as loadable.

    The tests write empty files named like controller libraries; the real check
    (loading each in a child process) is exercised by ``real_library_load`` tests.
    """
    if request.node.get_closest_marker("real_library_load") is None:

        def loads(path: Path) -> str | None:
            return None

        monkeypatch.setattr(steady_state, "_load_failure", loads)
