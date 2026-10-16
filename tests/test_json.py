# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane._json — strict JSON parsing."""

from __future__ import annotations

import pytest

from vane._json import loads_json_strict


class TestLoadsJsonStrict:
    """Tests for loads_json_strict."""

    def test_parses_standard_document(self) -> None:
        """A standard document with finite numbers parses unchanged."""
        assert loads_json_strict('{"a": 1.5, "b": [1, -2e3]}') == {
            "a": 1.5,
            "b": [1, -2000.0],
        }

    @pytest.mark.parametrize("literal", ["1e999", "-1e400", "1.0E+309"])
    def test_overflowing_number_is_rejected(self, literal: str) -> None:
        """A literal that overflows to infinity is rejected, not read as inf."""
        with pytest.raises(ValueError, match="non-finite"):
            loads_json_strict(f'{{"a": {literal}}}')

    def test_nonstandard_constant_is_rejected(self) -> None:
        """The NaN/Infinity constants remain rejected."""
        with pytest.raises(ValueError, match="nonstandard JSON numeric constant"):
            loads_json_strict('{"a": NaN}')

    def test_duplicate_name_is_rejected(self) -> None:
        """A duplicate object name is rejected."""
        with pytest.raises(ValueError, match="duplicate JSON object name"):
            loads_json_strict('{"a": 1, "a": 2}')
