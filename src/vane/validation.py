# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Plausibility comparison of extracted frequencies against published references.

VANE is a *post-processor* of an OpenFAST linearization: its correctness question is
whether it faithfully recovers the modal content of the matrices OpenFAST produced,
which is settled against version-unambiguous references (analytical solutions, exact
MBC invariants, prescribed-mode systems, and a second independent implementation on the
*same* matrices). Published reference-turbine frequencies are deliberately **not** used
as a pass/fail check: they are typically produced by other tools (isolated-beam modal
codes or design-stage aero-structural tools) at a particular model version, and they
routinely disagree with a coupled OpenFAST linearization because they answer a
*different* question and drift with model/tool version. Requiring agreement with them
would test the *model*, not VANE.

This module offers the published comparison in its correct role instead — a
**plausibility check and reporting feature**. :func:`compare_to_reference` matches the
frequencies VANE extracted to a set of literature values, reports the per-mode
differences, and leaves the outcome to the analyst, so a discrepancy can be *attributed*
to its cause (isolated-component vs full-system coupled modes, design-stage vs as-built,
tool/version drift) rather than silently failing a tolerance. It raises no error on
large deltas: it quantifies them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

__all__ = [
    "PlausibilityComparison",
    "compare_to_reference",
]


@dataclass
class PlausibilityComparison:
    """A non-binding comparison of extracted frequencies to published references.

    Each published reference frequency is paired with the closest extracted frequency
    (a unique assignment), and the signed differences are recorded. No tolerance is
    applied; the arrays simply quantify how the extracted model sits relative to the
    literature so the analyst can attribute the deltas.

    Parameters
    ----------
    labels : list[str]
        A label for each reference mode, parallel to ``reference_hz``.
    reference_hz : npt.NDArray[np.float64]
        The published reference natural frequencies (Hz), shape ``(n_ref,)``.
    matched_hz : npt.NDArray[np.float64]
        The extracted natural frequency (Hz) paired with each reference, same shape.
    absolute_delta_hz : npt.NDArray[np.float64]
        Signed difference ``matched_hz - reference_hz`` (Hz), same shape.
    relative_delta : npt.NDArray[np.float64]
        Signed relative difference ``(matched_hz - reference_hz) / reference_hz``,
        same shape (a fraction, not a percentage).
    source : str or None
        An optional name for the reference set (e.g. the publication), for reports.
    """

    labels: list[str]
    reference_hz: npt.NDArray[np.float64]
    matched_hz: npt.NDArray[np.float64]
    absolute_delta_hz: npt.NDArray[np.float64]
    relative_delta: npt.NDArray[np.float64]
    source: str | None

    @property
    def max_abs_relative_delta(self) -> float:
        """Return the largest absolute relative difference over the matched modes."""
        if self.relative_delta.size == 0:
            return 0.0
        return float(np.max(np.abs(self.relative_delta)))

    def report(self) -> str:
        """Return a human-readable plausibility table.

        Returns
        -------
        str
            A multi-line table of reference vs extracted frequency and the per-mode
            delta, headed by a note that this is a non-binding plausibility comparison.
        """
        title = "Plausibility comparison (non-binding)"
        if self.source is not None:
            title += f" vs {self.source}"
        header = (
            f"{'mode':<24}{'ref [Hz]':>12}{'VANE [Hz]':>12}"
            f"{'delta [Hz]':>12}{'delta [%]':>10}"
        )
        rows = [
            f"{label:<24}{ref:>12.4f}{got:>12.4f}{dabs:>12.4f}{drel * 100.0:>10.2f}"
            for label, ref, got, dabs, drel in zip(
                self.labels,
                self.reference_hz,
                self.matched_hz,
                self.absolute_delta_hz,
                self.relative_delta,
                strict=True,
            )
        ]
        return "\n".join([title, header, *rows])


def compare_to_reference(
    computed_hz: npt.ArrayLike,
    reference_hz: npt.ArrayLike,
    *,
    labels: Sequence[str] | None = None,
    source: str | None = None,
) -> PlausibilityComparison:
    """Compare extracted natural frequencies to published references (non-binding).

    Each reference frequency is paired with the closest extracted frequency under a
    unique assignment that minimizes the total absolute frequency distance (so two
    references never claim the same extracted mode), and the signed per-mode
    differences are reported. This is a **plausibility check**, not a correctness check:
    no tolerance is applied and no error is raised on large differences — they are
    quantified so they can be attributed to their cause (isolated-component vs coupled
    model, design-stage vs as-built, tool/version drift).

    Parameters
    ----------
    computed_hz : npt.ArrayLike
        The natural frequencies VANE extracted (Hz), shape ``(n_computed,)``; must be
        at least as many as the reference frequencies so every reference can be paired.
    reference_hz : npt.ArrayLike
        The published reference natural frequencies (Hz), shape ``(n_ref,)``.
    labels : Sequence[str] or None, optional
        A label for each reference mode, parallel to ``reference_hz`` (e.g.
        ``"1st tower fore-aft"``). Defaults to ``"mode 1"``, ``"mode 2"``, ...
    source : str or None, optional
        An optional name for the reference set, carried into :meth:`report`.

    Returns
    -------
    PlausibilityComparison
        The per-mode reference/extracted frequencies and their differences.

    Raises
    ------
    ValueError
        If either input is not a 1-D array of finite, strictly positive frequencies,
        if ``labels`` does not match ``reference_hz`` in length, or if fewer
        frequencies were extracted than references (so a pairing is impossible).
    """
    computed = np.asarray(computed_hz, dtype=np.float64)
    reference = np.asarray(reference_hz, dtype=np.float64)
    for name, array in (("computed_hz", computed), ("reference_hz", reference)):
        if array.ndim != 1 or array.size == 0:
            msg = f"{name} must be a non-empty 1-D array, got shape {array.shape}"
            raise ValueError(msg)
        if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
            msg = f"{name} must be finite and strictly positive"
            raise ValueError(msg)
    if reference.size > computed.size:
        msg = (
            f"need at least as many extracted frequencies as references; got "
            f"{computed.size} extracted vs {reference.size} references"
        )
        raise ValueError(msg)

    if labels is None:
        mode_labels = [f"mode {i + 1}" for i in range(reference.size)]
    else:
        mode_labels = list(labels)
        if len(mode_labels) != reference.size:
            msg = (
                f"labels must have one entry per reference frequency; got "
                f"{len(mode_labels)} labels vs {reference.size} references"
            )
            raise ValueError(msg)

    cost = np.abs(reference[:, None] - computed[None, :])
    _ref_rows, computed_cols = linear_sum_assignment(cost)
    # ``linear_sum_assignment`` returns the reference rows already in ascending order,
    # so the paired arrays stay aligned with ``reference``/``labels``.
    matched = computed[computed_cols]
    absolute_delta = matched - reference
    relative_delta = absolute_delta / reference
    return PlausibilityComparison(
        labels=mode_labels,
        reference_hz=reference,
        matched_hz=matched,
        absolute_delta_hz=absolute_delta,
        relative_delta=relative_delta,
        source=source,
    )
