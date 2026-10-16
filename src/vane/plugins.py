# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Extension protocols for VANE's pluggable analysis stages.

VANE's two model-dependent stages, cross-operating-point **tracking** and physical
**labelling**, are each defined here as a :class:`typing.Protocol`, so an alternative
implementation can be used wherever the built-in is used, without subclassing
anything. These are *static* (type-checker) contracts; the consumers accept the
protocol type, so a static type checker verifies that a custom implementation fits.

``Tracker`` and ``Labeler`` are callable-only protocols and are deliberately **not**
``runtime_checkable``: an ``isinstance`` check against a callable Protocol only confirms
that the object is callable (not its signature or return type), so it would accept a
non-tracker and fail later. Use static typing, or simply the injection points, for
them.

The built-ins satisfy these contracts structurally:

* :func:`vane.modal.identify_modes` is a :class:`Tracker`;
* :func:`vane.modal.label_solution` is a :class:`Labeler`.

The consumers accept the protocol type, not the concrete class:
:class:`vane.ModalPipeline` takes an optional ``tracker``. An LTP/Floquet tracker or
an alternative rule-based labeller can therefore be used without changing the
pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vane.modal.eigensolver import ModalSolution
    from vane.modal.identifier import IdentificationResult
    from vane.modal.labeler import ModeLabel

__all__ = ["Labeler", "Tracker"]


class Labeler(Protocol):
    """Assign a physical label to every mode of a solution.

    Any callable taking a :class:`~vane.modal.ModalSolution` and returning one
    :class:`~vane.modal.ModeLabel` per mode (in mode order) is a labeler.
    :func:`vane.modal.label_solution` is the built-in. A static (callable) contract:
    not ``runtime_checkable`` (see the module docstring).
    """

    def __call__(self, solution: ModalSolution) -> list[ModeLabel]:
        """Return one label per mode of ``solution``, in mode order."""
        ...


class Tracker(Protocol):
    """Link modes across operating points into labeled tracks.

    Any callable taking the per-operating-point solutions (with the optional sweep
    axis) and returning an :class:`~vane.modal.IdentificationResult` is a tracker.
    :func:`vane.modal.identify_modes` is the built-in; a Floquet/LTP tracker conforms
    by presenting the same call. A static (callable) contract: not
    ``runtime_checkable`` (see the module docstring).
    """

    def __call__(
        self,
        solutions: Sequence[ModalSolution],
        *,
        parameter_values: Sequence[float] | None = None,
    ) -> IdentificationResult:
        """Return the cross-operating-point tracks for ``solutions``."""
        ...
