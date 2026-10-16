# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Quantity-of-interest (load) map from a linearized output equation.

A linearization's output equation :math:`y = C x + D u` already contains the
sensitivity of every requested output — including structural loads such as mooring
fairlead/anchor tensions, tower-base and yaw-bearing moments, and blade-root moments
— to the system state and inputs. This module turns that into an explicit,
auditable **quantity-of-interest (QoI) map**: for each load output it exposes

* the **state sensitivity** (its row of :math:`C`): how each state moves the load;
* the **input sensitivity** (its row of :math:`D`): the direct feed-through from the
  inputs (wind, wave, control) to the load, bypassing the dynamics; and
* the **modal residues** :math:`C_i \Phi`: how each identified mode contributes to
  the load. Writing the state in modal coordinates :math:`x = \Phi q` gives
  :math:`y_i = (C_i \Phi)\, q`, so the residue of mode :math:`k` is the complex gain
  with which that mode appears in load :math:`i`. This is the bridge from a modal
  identification to load monitoring: it answers *which modes drive this load*.

The modal residues are taken over the retained (under-damped) modes of a
:class:`~vane.modal.eigensolver.ModalSolution`; over-damped/rigid-body modes and the
direct :math:`D u` feed-through are not part of that modal sum and are reported
separately (the input sensitivity). The map is a static, deterministic property of
the linear model — the reduced-order foundation a downstream estimator consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from vane.config.output_map import OutputCategory, OutputInfo, classify_output

if TYPE_CHECKING:
    import numpy.typing as npt
    import pandas as pd

    from vane.io.mbc_transform import MBCResult
    from vane.modal.eigensolver import ModalSolution
    from vane.sysid.state_space import StateSpace

__all__ = ["QoiMap", "QoiSensitivity", "build_qoi_map", "qoi_map_from_mbc"]


@dataclass(frozen=True)
class QoiSensitivity:
    r"""Sensitivity of one output channel to the states, inputs, and modes.

    Parameters
    ----------
    name : str
        The output channel description.
    info : OutputInfo
        Its classification (category, load flag, component, unit, ...).
    output_index : int
        Row index of this channel in ``C``/``D``.
    state_sensitivity : npt.NDArray[np.float64]
        The channel's row of ``C``, shape ``(n_states,)``: the linear sensitivity
        of the output to each state.
    input_sensitivity : npt.NDArray[np.float64] or None
        The channel's row of ``D``, shape ``(n_inputs,)``, or ``None`` when the
        system has no feed-through matrix. The direct input-to-output gain.
    modal_residues : npt.NDArray[np.complex128] or None
        The complex modal residues :math:`C_i \Phi`, shape ``(n_modes,)``, or
        ``None`` when no modal solution was supplied. The contribution of each
        retained mode to this output.
    mbc_coordinate : str
        The multi-blade coordinate of the channel (``"collective"``, ``"cosine"``,
        or ``"sine"``) when it is a transformed blade-triplet output, else ``""``.
        A non-empty value means this row is a multi-blade combination rather than
        the literal per-blade channel its name suggests, so ``info.component`` is
        cleared to ``None``.
    """

    name: str
    info: OutputInfo
    output_index: int
    state_sensitivity: npt.NDArray[np.float64]
    input_sensitivity: npt.NDArray[np.float64] | None
    modal_residues: npt.NDArray[np.complex128] | None
    mbc_coordinate: str = ""

    @property
    def is_load(self) -> bool:
        """Return whether this output is a structural load."""
        return self.info.is_load

    def dominant_modes(self, top_n: int = 3) -> npt.NDArray[np.int_]:
        """Return the indices of the modes that contribute most to this output.

        Parameters
        ----------
        top_n : int
            Number of leading modes to return.

        Returns
        -------
        npt.NDArray[np.int_]
            Mode indices ordered by descending residue magnitude (empty when no
            modal residues are available).
        """
        if self.modal_residues is None:
            return np.empty(0, dtype=np.int_)
        order = np.argsort(np.abs(self.modal_residues))[::-1]
        return np.asarray(order[:top_n], dtype=np.int_)


@dataclass
class QoiMap:
    """A map of output channels to their state/input/modal sensitivities.

    Parameters
    ----------
    sensitivities : list[QoiSensitivity]
        One entry per included output channel.
    state_names : list[str]
        State descriptions indexing ``state_sensitivity``.
    input_names : list[str]
        Input descriptions indexing ``input_sensitivity``.
    mode_frequencies_hz : npt.NDArray[np.float64] or None
        Natural frequencies of the modes indexing ``modal_residues``, or ``None``
        when no modal solution was supplied.
    """

    sensitivities: list[QoiSensitivity]
    state_names: list[str]
    input_names: list[str]
    mode_frequencies_hz: npt.NDArray[np.float64] | None = field(default=None)

    def loads(self) -> list[QoiSensitivity]:
        """Return only the structural-load sensitivities."""
        return [s for s in self.sensitivities if s.is_load]

    def by_category(self, category: OutputCategory) -> list[QoiSensitivity]:
        """Return the sensitivities of a given output category."""
        return [s for s in self.sensitivities if s.info.category is category]

    def get(self, name: str) -> QoiSensitivity:
        """Return the sensitivity whose channel name equals ``name``.

        Raises
        ------
        KeyError
            If no channel has that exact name.
        """
        for sensitivity in self.sensitivities:
            if sensitivity.name == name:
                return sensitivity
        msg = f"no output channel named {name!r}"
        raise KeyError(msg)

    def modal_participation_table(self) -> pd.DataFrame:
        """Return a long-format table of modal contributions to each load.

        Each row is one (load, mode) pair with the residue magnitude and phase and
        the per-load normalized magnitude (the fraction relative to that load's
        strongest mode). Empty when no modal residues are available.

        Returns
        -------
        pandas.DataFrame
            Columns ``load``, ``category``, ``unit``, ``mode``, ``frequency_hz``,
            ``residue_magnitude``, ``residue_phase_deg``, ``normalized``.
        """
        import pandas as pd

        rows: list[dict[str, object]] = []
        frequencies = self.mode_frequencies_hz
        for sensitivity in self.loads():
            residues = sensitivity.modal_residues
            if residues is None or residues.size == 0:
                continue
            magnitude = np.abs(residues)
            peak = float(magnitude.max())
            normalizer = peak if peak > 0.0 else 1.0
            rows.extend(
                {
                    "load": sensitivity.name,
                    "category": sensitivity.info.category.value,
                    "mbc_coordinate": sensitivity.mbc_coordinate,
                    "unit": sensitivity.info.unit,
                    "mode": mode,
                    "frequency_hz": (
                        float(frequencies[mode])
                        if frequencies is not None
                        else float("nan")
                    ),
                    "residue_magnitude": float(magnitude[mode]),
                    "residue_phase_deg": float(np.degrees(np.angle(residues[mode]))),
                    "normalized": float(magnitude[mode] / normalizer),
                }
                for mode in range(residues.shape[0])
            )
        return pd.DataFrame(rows)


def build_qoi_map(
    system: StateSpace,
    solution: ModalSolution | None = None,
    *,
    output_mbc_coordinates: list[str] | None = None,
    loads_only: bool = True,
) -> QoiMap:
    r"""Build the quantity-of-interest map from a state-space system.

    .. note::

        This is a **modal-residue screening** map — which modes drive which load
        channels in the linearized model — not a validated load ranking. The residues
        are in the linearization's output units (and an MBC-averaged frame for rotating
        loads); they are **not** fatigue/DEL, design-load, or extreme-load magnitudes
        and must not be used as a substitute for a load-analysis or assessment
        calculation. Use it to decide which modes a downstream estimator should retain,
        then validate the estimator against time-domain or measured loads.

    Parameters
    ----------
    system : StateSpace
        A system carrying an output matrix ``C`` and named outputs. The optional
        feed-through ``D`` supplies the input sensitivities.
    solution : ModalSolution or None, optional
        A modal solution of the same system; when given, the modal residues
        :math:`C_i \Phi` are computed from its full eigenvectors so each load's
        modal composition is available.
    output_mbc_coordinates : list[str] or None, optional
        The multi-blade coordinate of each output, aligned with the system outputs.
        When ``None`` (the default) the system's own
        :attr:`~vane.sysid.state_space.StateSpace.output_mbc_coordinates` are used,
        which :func:`~vane.sysid.state_space.state_space_from_mbc` populates — so an
        MBC-transformed system is handled correctly without any extra argument. A
        rotating blade-load output is a collective/cyclic combination, so when a
        channel's coordinate is multi-blade its per-blade component index is cleared
        and the coordinate is recorded on the sensitivity. Pass an explicit list to
        override. Non-rotating loads (mooring, tower, platform) are unaffected.
    loads_only : bool, optional
        When ``True`` (the default) include only channels classified as structural
        loads; when ``False`` include every output channel.

    Returns
    -------
    QoiMap
        The per-output sensitivities.

    Raises
    ------
    ValueError
        If the system has no ``C`` matrix or no output names, the output-name count
        disagrees with ``C``, ``output_mbc_coordinates`` has the wrong length, or
        the modal solution's eigenvectors do not span the system states.
    """
    if system.c is None:
        msg = "StateSpace has no C matrix; cannot build a QoI map"
        raise ValueError(msg)
    if not system.output_names:
        msg = (
            "StateSpace has no output_names; the QoI map needs output descriptions "
            "to classify load channels (use state_space_from_mbc on a recent "
            "MBCResult)"
        )
        raise ValueError(msg)
    c = system.c
    if len(system.output_names) != c.shape[0]:
        msg = (
            f"output_names length {len(system.output_names)} does not match C rows "
            f"{c.shape[0]}"
        )
        raise ValueError(msg)
    # Default to the system's own MBC coordinates so a system produced by
    # state_space_from_mbc is handled correctly without an explicit argument.
    if output_mbc_coordinates is None:
        output_mbc_coordinates = system.output_mbc_coordinates or None
    if output_mbc_coordinates is not None and len(output_mbc_coordinates) != c.shape[0]:
        msg = (
            f"output_mbc_coordinates length {len(output_mbc_coordinates)} does not "
            f"match C rows {c.shape[0]}"
        )
        raise ValueError(msg)

    residue_matrix = _modal_residues(c, solution)
    frequencies = solution.natural_frequencies_hz if solution is not None else None

    sensitivities: list[QoiSensitivity] = []
    for index, name in enumerate(system.output_names):
        coordinate = (
            output_mbc_coordinates[index] if output_mbc_coordinates is not None else ""
        )
        info = _classify_with_mbc(name, coordinate)
        if loads_only and not info.is_load:
            continue
        sensitivities.append(
            QoiSensitivity(
                name=name,
                info=info,
                output_index=index,
                state_sensitivity=np.asarray(c[index], dtype=np.float64),
                input_sensitivity=(
                    None
                    if system.d is None
                    else np.asarray(system.d[index], dtype=np.float64)
                ),
                modal_residues=(
                    None if residue_matrix is None else residue_matrix[index]
                ),
                mbc_coordinate=coordinate,
            )
        )
    return QoiMap(
        sensitivities=sensitivities,
        state_names=list(system.state_names),
        input_names=list(system.input_names),
        mode_frequencies_hz=frequencies,
    )


def qoi_map_from_mbc(
    result: MBCResult,
    solution: ModalSolution | None = None,
    *,
    loads_only: bool = True,
) -> QoiMap:
    """Build a QoI map directly from an MBC transform result.

    A thin convenience over ``build_qoi_map(state_space_from_mbc(result), ...)``.
    The exported state-space carries the output multi-blade coordinates, so a
    rotating blade-load output is correctly identified as a multi-blade coordinate
    rather than mislabeled as a literal per-blade channel.

    Parameters
    ----------
    result : MBCResult
        An MBC transform result with an averaged output matrix.
    solution : ModalSolution or None, optional
        A modal solution of the same averaged system, for the modal residues.
    loads_only : bool, optional
        Restrict to structural-load channels (the default).

    Returns
    -------
    QoiMap
        The per-output sensitivities, with transformed blade outputs tagged by
        their multi-blade coordinate.
    """
    from vane.sysid.state_space import state_space_from_mbc

    return build_qoi_map(state_space_from_mbc(result), solution, loads_only=loads_only)


_MBC_COORDINATES = frozenset({"collective", "cosine", "sine"})


def _classify_with_mbc(name: str, coordinate: str) -> OutputInfo:
    """Classify an output, clearing the per-blade component if MBC-transformed.

    A transformed blade-triplet output row is a collective/cyclic combination, not
    the per-blade channel its description names, so the literal blade component is
    dropped to avoid reporting the wrong physical channel.
    """
    info = classify_output(name)
    if coordinate in _MBC_COORDINATES and info.component is not None:
        return replace(info, component=None)
    return info


def _modal_residues(
    c: npt.NDArray[np.float64], solution: ModalSolution | None
) -> npt.NDArray[np.complex128] | None:
    """Return the ``(n_outputs, n_modes)`` residue matrix ``C @ Phi``, or ``None``."""
    if solution is None:
        return None
    phi = solution.full_eigenvectors
    if phi.shape[0] != c.shape[1]:
        msg = (
            f"modal eigenvectors span {phi.shape[0]} states but C has "
            f"{c.shape[1]} columns; the solution must match the system"
        )
        raise ValueError(msg)
    return np.asarray(c.astype(np.complex128) @ phi, dtype=np.complex128)
