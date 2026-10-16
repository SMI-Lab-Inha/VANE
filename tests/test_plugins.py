# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.plugins: the Tracker and Labeler extension protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from vane.modal import identify_modes, label_solution
from vane.pipeline import ModalPipeline
from vane.plugins import Labeler, Tracker

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tests.conftest import LinFileFactory
    from vane.io.lin_reader import LinFile
    from vane.modal.eigensolver import ModalSolution
    from vane.modal.identifier import IdentificationResult

from vane.io.lin_reader import read_lin_file

_STATES: list[tuple[str, bool, int]] = [
    ("ED 1st tower fore-aft (internal DOF index = DOF_TFA1), m", False, 2),
    (
        "ED First time derivative of 1st tower fore-aft "
        "(internal DOF index = DOF_TFA1), m/s",
        False,
        1,
    ),
]
_A = np.array([[0.0, 1.0], [-4.0, -0.2]], dtype=np.float64)


def _operating_point(
    make_lin_file: LinFileFactory, *, rotor_speed: float
) -> list[LinFile]:
    """Build a two-azimuth operating point at a given rotor speed."""
    return [
        read_lin_file(
            make_lin_file(states=_STATES, a=_A, rotor_speed=rotor_speed, azimuth=az)
        )
        for az in (0.0, 1.0)
    ]


class TestCallableProtocolsAreStaticOnly:
    """Tracker/Labeler are deliberately not runtime_checkable.

    A callable-only Protocol's ``isinstance`` would only check callability, which would
    accept a non-tracker; conformance is enforced statically and proven by the
    injection points instead.
    """

    def test_isinstance_is_disallowed(self) -> None:
        """An isinstance check against the callable protocols raises, by design."""
        for protocol in (Tracker, Labeler):
            with pytest.raises(TypeError):
                isinstance(identify_modes, protocol)

    def test_builtins_are_callable(self) -> None:
        """The built-ins present the callable interface the protocols require."""
        assert callable(identify_modes)
        assert callable(label_solution)


class TestPipelineTrackerInjection:
    """ModalPipeline uses an injected tracker in place of the built-in."""

    def test_custom_tracker_is_used(self, make_lin_file: LinFileFactory) -> None:
        """A custom tracker runs and its result flows through the pipeline."""
        points = [
            _operating_point(make_lin_file, rotor_speed=1.0),
            _operating_point(make_lin_file, rotor_speed=1.5),
        ]
        calls: list[int] = []

        def spy_tracker(
            solutions: Sequence[ModalSolution],
            *,
            parameter_values: Sequence[float] | None = None,
        ) -> IdentificationResult:
            calls.append(len(solutions))
            return identify_modes(solutions, parameter_values=parameter_values)

        result = ModalPipeline(tracker=spy_tracker).run(points)
        assert calls == [2]  # the custom tracker was called with both solutions
        assert result.identification.tracks  # and its output is the pipeline's result

    def test_builtin_used_without_a_tracker(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Without a tracker the built-in identify_modes is used."""
        points = [
            _operating_point(make_lin_file, rotor_speed=1.0),
            _operating_point(make_lin_file, rotor_speed=1.5),
        ]
        result = ModalPipeline().run(points)
        assert result.identification.tracks
