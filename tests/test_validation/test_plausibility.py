# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.validation — non-binding plausibility comparison."""

import numpy as np
import pytest

from vane.validation import PlausibilityComparison, compare_to_reference


class TestCompareToReference:
    """Tests for compare_to_reference."""

    def test_exact_match_has_zero_delta(self) -> None:
        """Identical frequencies give zero absolute and relative delta."""
        freqs = [0.28, 0.31, 0.62]
        cmp = compare_to_reference(freqs, freqs)
        np.testing.assert_allclose(cmp.absolute_delta_hz, 0.0, atol=1e-15)
        np.testing.assert_allclose(cmp.relative_delta, 0.0, atol=1e-15)
        assert cmp.max_abs_relative_delta == pytest.approx(0.0, abs=1e-15)

    def test_pairs_each_reference_with_nearest_extracted(self) -> None:
        """Each reference is paired with the closest extracted frequency."""
        computed = [0.65, 0.275, 0.305, 1.10]  # extra, out of order
        reference = [0.28, 0.31, 0.62]
        cmp = compare_to_reference(computed, reference)
        np.testing.assert_allclose(cmp.matched_hz, [0.275, 0.305, 0.65])
        np.testing.assert_allclose(cmp.reference_hz, reference)
        np.testing.assert_allclose(
            cmp.absolute_delta_hz, [-0.005, -0.005, 0.03], atol=1e-12
        )

    def test_assignment_is_unique(self) -> None:
        """Two references never claim the same extracted mode."""
        # Both references sit nearest 0.50 greedily, but the unique assignment must
        # split them across the two available extracted modes.
        cmp = compare_to_reference([0.50, 0.51], [0.499, 0.502])
        assert sorted(cmp.matched_hz.tolist()) == [0.50, 0.51]
        assert len(set(cmp.matched_hz.tolist())) == 2

    def test_signed_delta_direction(self) -> None:
        """Delta is matched minus reference (positive when VANE is higher)."""
        cmp = compare_to_reference([0.33], [0.30])
        assert cmp.absolute_delta_hz[0] == pytest.approx(0.03)
        assert cmp.relative_delta[0] == pytest.approx(0.10)

    def test_large_discrepancy_does_not_raise(self) -> None:
        """A big delta is quantified, never checked (the whole point of B1)."""
        cmp = compare_to_reference([1.0], [0.5])
        assert cmp.max_abs_relative_delta == pytest.approx(1.0)

    def test_default_labels(self) -> None:
        """Without labels, modes are numbered from one."""
        cmp = compare_to_reference([0.1, 0.2], [0.1, 0.2])
        assert cmp.labels == ["mode 1", "mode 2"]

    def test_custom_labels_and_source_in_report(self) -> None:
        """Custom labels and a source name appear in the report."""
        cmp = compare_to_reference(
            [0.275, 0.62],
            [0.28, 0.62],
            labels=["1st tower fore-aft", "1st tower side-side"],
            source="OC4 reference",
        )
        text = cmp.report()
        assert "OC4 reference" in text
        assert "1st tower fore-aft" in text
        assert "non-binding" in text.lower()

    def test_returns_dataclass(self) -> None:
        """The result is a PlausibilityComparison with aligned shapes."""
        cmp = compare_to_reference([0.1, 0.2, 0.3], [0.1, 0.3])
        assert isinstance(cmp, PlausibilityComparison)
        assert cmp.reference_hz.shape == (2,)
        assert cmp.matched_hz.shape == (2,)
        assert cmp.absolute_delta_hz.shape == (2,)


class TestValidation:
    """Input validation."""

    def test_rejects_too_few_extracted(self) -> None:
        """Fewer extracted than reference frequencies cannot be paired."""
        with pytest.raises(ValueError, match="at least as many"):
            compare_to_reference([0.1], [0.1, 0.2])

    def test_rejects_label_length_mismatch(self) -> None:
        """Labels must match the reference count."""
        with pytest.raises(ValueError, match="one entry per reference"):
            compare_to_reference([0.1, 0.2], [0.1, 0.2], labels=["only one"])

    def test_rejects_non_positive(self) -> None:
        """Non-positive frequencies are rejected."""
        with pytest.raises(ValueError, match="strictly positive"):
            compare_to_reference([0.1, -0.2], [0.1])

    def test_rejects_non_finite(self) -> None:
        """Non-finite frequencies are rejected."""
        with pytest.raises(ValueError, match="finite"):
            compare_to_reference([0.1, np.inf], [0.1])

    def test_rejects_non_1d(self) -> None:
        """A 2-D input is rejected."""
        with pytest.raises(ValueError, match="1-D"):
            compare_to_reference(np.ones((2, 2)), [0.1])

    def test_rejects_empty(self) -> None:
        """An empty reference is rejected."""
        with pytest.raises(ValueError, match="non-empty"):
            compare_to_reference([0.1], [])
