# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tabular and reduced-order-model export of the analysis products.

The analysis products — a per-operating-point modal solution, its mode shapes and
participation factors, a Campbell diagram of tracked modes, and detected resonance
crossings — are flattened into :class:`pandas.DataFrame` tables and written to CSV,
JSON, Excel, or ``.mat`` for downstream use (spreadsheets, reports, notebooks and
numerical environments that read ``.mat`` files). The tables optionally carry the
physical labels, per-mode confidence, and azimuth-spread uncertainty produced by the
rest of the library.

:func:`write_rom` packages a reduced-order model (the reduced state space plus its
modal metadata) as a self-contained, auditable bundle ready for a digital-twin /
state-estimation workflow. :func:`build_physical_lpv_family` exports the named
physical-state operating-point plant family a gain-scheduled controller schedules on;
:func:`~vane.sysid.serialize.save_lpv_family` and
:func:`~vane.sysid.serialize.load_lpv_family` archive it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from vane._version import vane_version

# Re-exported here so a controller imports the LPV plant and its archive functions
# from the export surface (``from vane.export import build_physical_lpv_family``); the
# implementation lives in vane.sysid, alongside the StateSpace and ROM-family builders
# and their serialization.
from vane.sysid.physical_lpv import LPVFamily, build_physical_lpv_family
from vane.sysid.serialize import load_lpv_family, save_lpv_family

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from vane.campbell.builder import CampbellDiagram
    from vane.campbell.excitation import ResonanceCrossing
    from vane.modal.eigensolver import ModalSolution
    from vane.modal.labeler import ModeLabel
    from vane.modal.participation import ParticipationResult
    from vane.modal.uncertainty import AzimuthSpread
    from vane.sysid.rom import ModalReduction

__all__ = [
    "LPVFamily",
    "build_physical_lpv_family",
    "campbell_table",
    "load_lpv_family",
    "mode_shapes_table",
    "modes_table",
    "participation_table",
    "resonance_table",
    "save_lpv_family",
    "write_rom",
    "write_table",
]

_FORMAT_SUFFIXES = (".csv", ".json", ".xlsx", ".mat")
_CAMPBELL_RESERVED_COLUMNS = frozenset(
    {
        "track",
        "label",
        "multiblade",
        "natural_frequency_hz",
        "damping_ratio",
        "confidence",
        "is_ambiguous",
    }
)


def modes_table(
    solution: ModalSolution,
    *,
    labels: Sequence[ModeLabel] | None = None,
    confidence: npt.NDArray[np.float64] | Sequence[float] | None = None,
    spread: AzimuthSpread | None = None,
) -> pd.DataFrame:
    """Flatten a modal solution into a per-mode table.

    Parameters
    ----------
    solution : ModalSolution
        The modal solution to export.
    labels : Sequence[ModeLabel] or None, optional
        Per-mode physical labels; their name, category, and multi-blade type are
        added as columns.
    confidence : npt.NDArray[np.float64] or Sequence[float] or None, optional
        Per-mode unified confidence, added as a column.
    spread : AzimuthSpread or None, optional
        Per-mode azimuth spread; its frequency and damping standard deviations are
        added as columns.

    Returns
    -------
    pandas.DataFrame
        One row per mode, ordered as in ``solution``.

    Raises
    ------
    ValueError
        If any optional argument's length does not match the number of modes.
    """
    n_modes = solution.n_modes
    if labels is not None and len(labels) != n_modes:
        msg = f"labels has {len(labels)} entries, expected {n_modes}"
        raise ValueError(msg)
    if confidence is not None and len(confidence) != n_modes:
        msg = f"confidence has {len(confidence)} entries, expected {n_modes}"
        raise ValueError(msg)
    if spread is not None and spread.natural_frequency_std.shape[0] != n_modes:
        msg = (
            f"spread has {spread.natural_frequency_std.shape[0]} entries, "
            f"expected {n_modes}"
        )
        raise ValueError(msg)

    data: dict[str, object] = {
        "mode": list(range(n_modes)),
        "natural_frequency_hz": solution.natural_frequencies_hz,
        "damped_frequency_hz": solution.damped_frequencies_hz,
        "damping_ratio": solution.damping_ratios,
    }
    if solution.condition_numbers.shape[0] == n_modes:
        data["condition_number"] = solution.condition_numbers
    if solution.is_degenerate.shape[0] == n_modes:
        data["is_degenerate"] = solution.is_degenerate
    if labels is not None:
        data["label"] = [label.label for label in labels]
        data["category"] = [label.category.name for label in labels]
        data["multiblade"] = [label.multiblade for label in labels]
    if confidence is not None:
        data["confidence"] = np.asarray(confidence, dtype=np.float64)
    if spread is not None:
        data["frequency_std_hz"] = spread.natural_frequency_std
        data["damping_std"] = spread.damping_ratio_std
    return pd.DataFrame(data)


def campbell_table(diagram: CampbellDiagram) -> pd.DataFrame:
    """Flatten a Campbell diagram into a long-format table of tracked modes.

    Parameters
    ----------
    diagram : CampbellDiagram
        The Campbell diagram to export.

    Returns
    -------
    pandas.DataFrame
        One row per (track, operating point): the track index and label, the
        operating-parameter value, natural frequency, damping ratio, track
        confidence, and ambiguity flag. The operating-parameter column is named
        after ``diagram.parameter_name``.

    Raises
    ------
    ValueError
        If ``diagram.parameter_name`` collides with a fixed export column.
    """
    parameter = diagram.parameter_name
    if parameter in _CAMPBELL_RESERVED_COLUMNS:
        msg = (
            f"parameter_name '{parameter}' collides with a reserved export column; "
            f"rename it (reserved: {', '.join(sorted(_CAMPBELL_RESERVED_COLUMNS))})"
        )
        raise ValueError(msg)
    rows: list[dict[str, object]] = []
    for track_id, track in enumerate(diagram.tracks):
        for point, frequency, damping in zip(
            track.operating_points,
            track.natural_frequencies_hz,
            track.damping_ratios,
            strict=True,
        ):
            rows.append(
                {
                    "track": track_id,
                    "label": track.label.label,
                    "multiblade": track.label.multiblade,
                    parameter: float(diagram.parameter_values[point]),
                    "natural_frequency_hz": frequency,
                    "damping_ratio": damping,
                    "confidence": track.confidence,
                    "is_ambiguous": track.is_ambiguous,
                }
            )
    columns = [
        "track",
        "label",
        "multiblade",
        parameter,
        "natural_frequency_hz",
        "damping_ratio",
        "confidence",
        "is_ambiguous",
    ]
    return pd.DataFrame(rows, columns=columns)


def mode_shapes_table(solution: ModalSolution) -> pd.DataFrame:
    """Flatten a solution's complex mode shapes into a long-format table.

    Parameters
    ----------
    solution : ModalSolution
        The modal solution whose ``mode_shapes`` (one column per mode) are exported.

    Returns
    -------
    pandas.DataFrame
        One row per (mode, degree of freedom): the mode index, DOF index and
        description, and the mode-shape component's real part, imaginary part,
        magnitude, and phase (degrees).
    """
    shapes = np.asarray(solution.mode_shapes)
    n_dof, n_modes = shapes.shape
    descriptions = _dof_descriptions(solution.dof_descriptions, n_dof)
    rows: list[dict[str, object]] = []
    for mode in range(n_modes):
        column = shapes[:, mode]
        for dof in range(n_dof):
            value = complex(column[dof])
            rows.append(
                {
                    "mode": mode,
                    "dof": dof,
                    "dof_description": descriptions[dof],
                    "real": value.real,
                    "imag": value.imag,
                    "magnitude": abs(value),
                    "phase_deg": float(np.degrees(np.angle(value))),
                }
            )
    columns = [
        "mode",
        "dof",
        "dof_description",
        "real",
        "imag",
        "magnitude",
        "phase_deg",
    ]
    return pd.DataFrame(rows, columns=columns)


def participation_table(
    participation: ParticipationResult,
    *,
    descriptions: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Flatten participation factors into a long-format per-(mode, DOF) table.

    Parameters
    ----------
    participation : ParticipationResult
        The per-DOF participation to export (``magnitude`` has one column per mode).
    descriptions : Sequence[str] or None, optional
        DOF descriptions parallel to the participation rows; generic ``dof_i`` names
        are used when omitted or mismatched in length.

    Returns
    -------
    pandas.DataFrame
        One row per (mode, DOF): the magnitude, signed magnitude, phase (degrees), and
        whether the DOF is the mode's dominant participant.
    """
    magnitude = np.asarray(participation.magnitude)
    n_dof, n_modes = magnitude.shape
    names = _dof_descriptions(list(descriptions or []), n_dof)
    dominant = np.asarray(participation.dominant_state)
    signed = np.asarray(participation.signed_magnitude)
    phase = np.asarray(participation.phase_deg)
    rows: list[dict[str, object]] = [
        {
            "mode": mode,
            "dof": dof,
            "dof_description": names[dof],
            "magnitude": float(magnitude[dof, mode]),
            "signed_magnitude": float(signed[dof, mode]),
            "phase_deg": float(phase[dof, mode]),
            "is_dominant": bool(dominant[mode] == dof),
        }
        for mode in range(n_modes)
        for dof in range(n_dof)
    ]
    columns = [
        "mode",
        "dof",
        "dof_description",
        "magnitude",
        "signed_magnitude",
        "phase_deg",
        "is_dominant",
    ]
    return pd.DataFrame(rows, columns=columns)


def resonance_table(resonances: Sequence[ResonanceCrossing]) -> pd.DataFrame:
    """Flatten detected resonance crossings into a table.

    Parameters
    ----------
    resonances : Sequence[ResonanceCrossing]
        The crossings to export (e.g. ``PipelineResult.resonances``).

    Returns
    -------
    pandas.DataFrame
        One row per crossing: the track index, excitation harmonic, rotor speed
        (rpm), crossing frequency (Hz), damping ratio, and severity. Empty (with the
        columns present) when there are no crossings.
    """
    columns = [
        "track",
        "harmonic",
        "rotor_speed_rpm",
        "frequency_hz",
        "damping_ratio",
        "severity",
    ]
    rows = [
        {
            "track": crossing.track_index,
            "harmonic": crossing.harmonic,
            "rotor_speed_rpm": crossing.rotor_speed_rpm,
            "frequency_hz": crossing.frequency_hz,
            "damping_ratio": crossing.damping_ratio,
            "severity": crossing.severity.name,
        }
        for crossing in resonances
    ]
    return pd.DataFrame(rows, columns=columns)


def _dof_descriptions(descriptions: list[str], n_dof: int) -> list[str]:
    """Return DOF descriptions of length ``n_dof``, falling back to generic names."""
    if len(descriptions) == n_dof:
        return descriptions
    return [f"dof_{index}" for index in range(n_dof)]


def write_table(table: pd.DataFrame, path: str | Path) -> None:
    """Write a table to CSV, JSON, Excel, or MATLAB, inferred from the file suffix.

    Parameters
    ----------
    table : pandas.DataFrame
        The table to write.
    path : str or pathlib.Path
        Destination path; the suffix selects the format (``.csv``, ``.json``,
        ``.xlsx``, or ``.mat``). A ``.mat`` file stores each column as a MATLAB
        variable of the same name.

    Raises
    ------
    ValueError
        If the suffix is not a supported format, Excel is requested without the
        optional ``openpyxl`` dependency installed, or a ``.mat`` table holds a value
        MATLAB cannot store. Missing values (``None``/``NaN``) in object/string
        columns are written to ``.mat`` as empty strings in a cell array.
    """
    destination = Path(path)
    suffix = destination.suffix.lower()
    if suffix == ".csv":
        table.to_csv(destination, index=False)
    elif suffix == ".json":
        table.to_json(destination, orient="records", indent=2)
    elif suffix == ".xlsx":
        try:
            import openpyxl  # noqa: F401
        except ImportError as exc:
            msg = (
                "Excel export requires the optional 'openpyxl' package; "
                'install it with: pip install "vane[excel]"'
            )
            raise ValueError(msg) from exc
        table.to_excel(destination, index=False)
    elif suffix == ".mat":
        from scipy.io import savemat

        # oned_as="column": write each column as an N x 1 MATLAB column vector (the
        # table orientation), not the 1 x N row that savemat defaults to.
        try:
            savemat(destination, _table_to_matlab(table), oned_as="column")
        except TypeError as exc:
            # savemat rejects values it cannot map to a MATLAB class; surface it as an
            # input error (a clean CLI message) instead of an internal TypeError.
            msg = f"cannot write the table to MATLAB .mat: {exc}"
            raise ValueError(msg) from exc
    else:
        msg = (
            f"unsupported export format '{suffix}'; "
            f"use one of {', '.join(_FORMAT_SUFFIXES)}"
        )
        raise ValueError(msg)


def _table_to_matlab(table: pd.DataFrame) -> dict[str, Any]:
    """Map a DataFrame to a MATLAB-savable dict, one array variable per column.

    Column names are sanitized to valid MATLAB identifiers (non-alphanumeric
    characters become ``_``, a leading digit is prefixed), and object/string columns
    are written as cell arrays of strings, with a missing value (``None``/``NaN``)
    written as the empty string ``""`` -- ``scipy.io.savemat`` cannot store ``None``,
    so an all-``None`` column (e.g. the multi-blade coordinate of a sweep without blade
    DOFs) would otherwise abort the export.
    """
    payload: dict[str, Any] = {}
    for column in table.columns:
        name = _matlab_name(str(column))
        if name in payload:
            msg = (
                f"two columns map to the same MATLAB variable name {name!r}; "
                f"rename a column before exporting to .mat"
            )
            raise ValueError(msg)
        series = table[column]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            cells = np.empty(len(series), dtype=object)
            cells[:] = [_matlab_cell(value) for value in series.tolist()]
            payload[name] = cells
        else:
            payload[name] = series.to_numpy()
    return payload


def _matlab_cell(value: object) -> object:
    """Return one object-column entry for a MATLAB cell (missing -> ``""``)."""
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return value


def _cell_array(values: list[str]) -> npt.NDArray[np.object_]:
    """Return ``values`` as a 1-D object array (a MATLAB cell array of strings)."""
    cells = np.empty(len(values), dtype=object)
    cells[:] = values
    return cells


def _matlab_name(name: str) -> str:
    """Return a valid MATLAB variable name for a column label.

    A MATLAB variable name must begin with a letter, so a name that does not (a leading
    digit, underscore, or other non-letter) is prefixed; this also avoids the leading
    underscore that ``scipy.io.savemat`` would otherwise silently drop.
    """
    cleaned = "".join(char if char.isalnum() else "_" for char in name)
    if not cleaned or not cleaned[0].isalpha():
        cleaned = f"v_{cleaned}"
    return cleaned


def write_rom(
    reduction: ModalReduction,
    output_dir: str | Path,
    *,
    matrix_format: str = "mat",
) -> Path:
    """Write a reduced-order model as a self-contained, auditable bundle.

    The bundle is a directory holding the reduced state-space matrices, a per-mode
    modal table, and a JSON manifest, so the reduced model a digital-twin / state
    estimator consumes travels with the provenance of how it was reduced.

    Parameters
    ----------
    reduction : ModalReduction
        A reduced-order model from :func:`vane.sysid.modal_reduction`.
    output_dir : str or pathlib.Path
        Destination directory; created if it does not exist.
    matrix_format : {"mat", "npz"}, optional
        Format for the state-space matrices — ``.mat`` (the default) or NumPy
        ``.npz``.

    Returns
    -------
    pathlib.Path
        The bundle directory written: ``state_space.{mat,npz}`` (the ``A``/``B``/``C``/
        ``D`` matrices, ``dt``, the state/input/output names and each output's
        multi-blade coordinate, empty when unknown -- MATLAB cell arrays of strings in
        ``.mat``, unicode arrays loadable without ``allow_pickle`` in ``.npz``; plus
        ``state_projection`` (full state to modal coordinates) and
        ``state_reconstruction`` (modal coordinates to full state) when the reduction
        carries them, listed under ``state_maps`` in the manifest),
        ``modes.csv`` (per
        retained mode: natural frequency, damping ratio, ranking residue), and
        ``manifest.json`` (the reduction metadata and library version).

    Raises
    ------
    ValueError
        If ``matrix_format`` is not ``"mat"`` or ``"npz"``.
    """
    if matrix_format not in ("mat", "npz"):
        msg = f"matrix_format must be 'mat' or 'npz', got {matrix_format!r}"
        raise ValueError(msg)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    system = reduction.system

    # Emit dimensionally consistent matrices: a missing B/C/D becomes a correctly
    # shaped zero block so a downstream loader can always reconstruct y = C x + D u
    # without a shape mismatch. The input/output counts use the StateSpace convention
    # (B/D for inputs, C/D for outputs) — a feedthrough-only system (D set, B omitted)
    # still reports its true input count, and B is zero-filled to match.
    a = np.asarray(system.a, dtype=np.float64)
    n_states = a.shape[0]
    d_given = None if system.d is None else np.asarray(system.d, dtype=np.float64)
    n_inputs = system.n_inputs
    n_outputs = system.n_outputs
    b = (
        np.asarray(system.b, dtype=np.float64)
        if system.b is not None
        else np.zeros((n_states, n_inputs), dtype=np.float64)
    )
    c = (
        np.asarray(system.c, dtype=np.float64)
        if system.c is not None
        else np.zeros((n_outputs, n_states), dtype=np.float64)
    )
    d = (
        d_given
        if d_given is not None
        else np.zeros((n_outputs, n_inputs), dtype=np.float64)
    )

    payload: dict[str, Any] = {
        "A": a,
        "B": b,
        "C": c,
        "D": d,
        "dt": np.float64(system.dt if system.dt is not None else 0.0),
    }
    names: dict[str, list[str]] = {
        "state_names": list(system.state_names),
        "input_names": list(system.input_names),
        "output_names": list(system.output_names),
        # Distinguishes MBC collective/cosine/sine rows from literal per-blade rows.
        "output_mbc_coordinates": list(system.output_mbc_coordinates),
    }
    # The full-state <-> modal-coordinate maps, when the reduction carries them, so a
    # consumer can project a full physical state (or reconstruct one) consistently.
    state_maps: list[str] = []
    for key in ("state_projection", "state_reconstruction"):
        value = getattr(reduction, key)
        if value is not None:
            payload[key] = np.asarray(value, dtype=np.float64)
            state_maps.append(key)
    if matrix_format == "mat":
        from scipy.io import savemat

        # Object arrays become MATLAB cell arrays, so every name round-trips exactly;
        # a unicode array would be stored as a blank-padded char matrix instead.
        for key, values in names.items():
            payload[key] = _cell_array(values)
        savemat(destination / "state_space.mat", payload, oned_as="column")
        matrix_file = "state_space.mat"
    else:
        # Unicode string arrays (not object arrays) so the .npz loads without the
        # unsafe ``allow_pickle=True``.
        for key, values in names.items():
            payload[key] = np.array(values, dtype=str)
        np.savez(destination / "state_space.npz", **payload)
        matrix_file = "state_space.npz"

    n_modes = int(np.asarray(reduction.frequencies_hz).shape[0])
    modes = pd.DataFrame(
        {
            "mode": list(range(n_modes)),
            "natural_frequency_hz": np.asarray(reduction.frequencies_hz),
            "damping_ratio": np.asarray(reduction.damping_ratios),
            "residue": np.asarray(reduction.residues),
        }
    )
    write_table(modes, destination / "modes.csv")

    manifest: dict[str, object] = {
        "vane_version": vane_version(),
        "matrix_file": matrix_file,
        "n_full_states": reduction.n_full_states,
        "n_reduced_states": int(np.asarray(system.a).shape[0]),
        "n_retained_modes": n_modes,
        "n_stabilized": reduction.n_stabilized,
        "energy_retained": reduction.energy_retained,
        "is_discrete": system.dt is not None,
        "dt": system.dt,
        "n_inputs": n_inputs,
        "n_outputs": n_outputs,
        "state_names": system.state_names,
        "input_names": system.input_names,
        "output_names": system.output_names,
        "output_mbc_coordinates": system.output_mbc_coordinates,
        # Names of the optional full-state maps stored in the matrix file.
        "state_maps": state_maps,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return destination
