# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Safe, provenance-stamped save/reload for estimator results and model families.

An estimator's outputs are evidence: a filtered trajectory or a scheduled run must be
reproducible and archivable, not recomputed from memory. This
module serializes the :mod:`vane.sysid` result dataclasses, the tracked
:class:`~vane.sysid.rom_family.RomFamily` and the physical-state
:class:`~vane.sysid.physical_lpv.LPVFamily` to a single ``.npz`` file each and loads
them back exactly.

The format is deliberately **safe and transparent**: the numeric arrays are stored as
plain NumPy arrays and every other field as a JSON metadata blob, so a result is loaded
with ``allow_pickle=False`` -- it can carry no executable payload, unlike a pickle or
joblib artifact. Each file also records the result type and the **VANE version** that
produced it, so an archived result is traceable to the code that made it.
"""

from __future__ import annotations

import json
import os
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np

from vane._version import vane_version
from vane.sysid.kalman_filter import ConsistencyReport, FilterResult
from vane.sysid.lpv import ScheduledFilterResult
from vane.sysid.physical_lpv import LPVFamily
from vane.sysid.rom_family import RomFamily
from vane.sysid.state_space import StateSpace

__all__ = [
    "LPV_FAMILY_SCHEMA_VERSION",
    "load_lpv_family",
    "load_result",
    "load_rom_family",
    "save_lpv_family",
    "save_result",
    "save_rom_family",
]

_FORMAT = "vane-estimator-result"
_ROM_FAMILY_FORMAT = "vane-rom-family"
_LPV_FAMILY_FORMAT = "vane-lpv-family"
#: Schema version of the LPV-family archive written by :func:`save_lpv_family`.
#: :func:`load_lpv_family` reads only this version and names both versions when an
#: archive records another one.
LPV_FAMILY_SCHEMA_VERSION = 1


def _npz_path(path: str | os.PathLike[str]) -> str:
    """Return ``path`` with a ``.npz`` suffix (added if absent).

    ``np.savez`` silently appends ``.npz`` to a suffixless path, so ``save_result`` and
    ``load_result`` must normalize identically -- otherwise ``save_result(r, "x")``
    writes ``x.npz`` but ``load_result("x")`` would look for ``x`` and miss it.
    """
    text = os.fspath(path)
    return text if text.endswith(".npz") else text + ".npz"


_Result = FilterResult | ScheduledFilterResult

# The top-level result types a file may hold (what ``save_result`` writes and
# ``load_result`` returns). Kept separate from the nested-dataclass registry so a
# crafted archive cannot name a nested-only type (e.g. ``ConsistencyReport``) at the top
# level and have ``load_result`` return something outside its documented return type.
_RESULT_REGISTRY: dict[str, type] = {
    cls.__name__: cls for cls in (FilterResult, ScheduledFilterResult)
}

# Dataclasses that may appear only *inside* a result.
_NESTED_REGISTRY: dict[str, type] = {ConsistencyReport.__name__: ConsistencyReport}


def _required_array_fields(cls: type) -> tuple[str, ...]:
    """Return the names of fields that must hold an array (a non-optional ``NDArray``).

    Read from the dataclass's own annotations (string forms under
    ``from __future__ import annotations``), independent of any file's self-description:
    a required array is ``npt.NDArray[...]`` without ``None``; an optional array
    (``... | None``) and the non-array fields are excluded. Used to reject a tampered
    archive that supplies a required array as a metadata ``null`` instead of an array.
    """
    annotations: dict[str, Any] = getattr(cls, "__annotations__", {})
    return tuple(
        name
        for name, ann in annotations.items()
        if isinstance(ann, str) and "NDArray" in ann and "None" not in ann
    )


def _array_field_names(cls: type) -> frozenset[str]:
    """Return every field whose annotation is an ``NDArray`` (required or optional).

    Such a field is stored as an array; in metadata it may only carry ``None`` (an
    absent optional array). A non-``None`` metadata value for one means a tampered file.
    """
    annotations: dict[str, Any] = getattr(cls, "__annotations__", {})
    return frozenset(
        name
        for name, ann in annotations.items()
        if isinstance(ann, str) and "NDArray" in ann
    )


def _encode_field(value: Any) -> Any:  # noqa: ANN401 - heterogeneous dataclass fields
    """Encode a non-array field into a JSON-serializable form."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        # A NumPy scalar (e.g. a np.float32 coverage_sigma passed through the public
        # API) is JSON-encoded as its Python equivalent so the result still round-trips.
        return value.item()
    if isinstance(value, list):
        return [_encode_field(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__dataclass__": type(value).__name__,
            "fields": {
                f.name: _encode_field(getattr(value, f.name)) for f in fields(value)
            },
        }
    msg = f"cannot serialize field of type {type(value).__name__}"
    raise TypeError(msg)


def _decode_field(value: Any) -> Any:  # noqa: ANN401 - heterogeneous dataclass fields
    """Decode a JSON value produced by :func:`_encode_field`."""
    if isinstance(value, dict) and "__dataclass__" in value:
        nested_name = value["__dataclass__"]
        if nested_name not in _NESTED_REGISTRY:
            msg = f"unknown nested dataclass {nested_name!r}"
            raise ValueError(msg)
        cls = _NESTED_REGISTRY[nested_name]
        return cls(**{k: _decode_field(v) for k, v in value["fields"].items()})
    if isinstance(value, list):
        return [_decode_field(item) for item in value]
    return value


def save_result(result: _Result, path: str | os.PathLike[str]) -> None:
    """Save an estimator result to a ``.npz`` file.

    Writes the result's NumPy arrays as plain arrays and the remaining fields (names,
    flags, the nested consistency report) as a JSON metadata blob, stamped with the
    result type and the producing VANE version. The file is self-describing and loads
    back with :func:`load_result` without executing any embedded code.

    Parameters
    ----------
    result : FilterResult or ScheduledFilterResult
        The estimator result to serialize.
    path : str or os.PathLike
        Destination file (a ``.npz`` is written; the suffix is added if absent).

    Raises
    ------
    TypeError
        If ``result`` is not a registered estimator-result type, or carries a field that
        cannot be serialized.
    """
    # Exact type, not ``isinstance``: a subclass would be written under its own name but
    # ``load_result`` only knows the three registered types, so the archive would be
    # unreloadable. Reject it at write time instead of producing a dead file.
    name = type(result).__name__
    if type(result) not in (FilterResult, ScheduledFilterResult):
        msg = f"save_result does not handle {name}"
        raise TypeError(msg)
    arrays: dict[str, np.ndarray] = {}
    meta_fields: dict[str, Any] = {}
    for field in fields(result):
        value = getattr(result, field.name)
        if isinstance(value, np.ndarray):
            arrays[field.name] = value
        else:
            meta_fields[field.name] = _encode_field(value)
    meta = {
        "__format__": _FORMAT,
        "__type__": name,
        "__vane_version__": vane_version(),
        "fields": meta_fields,
    }
    # Store the metadata as a 0-d string array so the archive is a single, pickle-free
    # npz (the arrays share the same file under their field names).
    payload: dict[str, np.ndarray] = {"__meta__": np.asarray(json.dumps(meta))}
    payload.update(arrays)
    # mypy reads a splatted dict as a possible ``allow_pickle=`` (bool) keyword; none of
    # our keys is that reserved name, so the arrays are all stored as named arrays. The
    # path is normalized so the written file matches what ``load_result`` will open.
    np.savez(_npz_path(path), **payload)  # type: ignore[arg-type]


def load_result(path: str | os.PathLike[str]) -> _Result:
    """Load an estimator result saved by :func:`save_result`.

    Reads the archive with ``allow_pickle=False`` (so a tampered file can carry no
    executable payload), reconstructs the recorded result type, and returns it exactly
    as saved -- arrays, names, flags, and the nested consistency report.

    Parameters
    ----------
    path : str or os.PathLike
        The ``.npz`` file written by :func:`save_result`.

    Returns
    -------
    FilterResult or ScheduledFilterResult
        The reconstructed result.

    Raises
    ------
    ValueError
        If the file is not a VANE estimator-result archive or records an unknown type.
    """
    with np.load(_npz_path(path), allow_pickle=False) as data:
        if "__meta__" not in data:
            msg = "file is not a VANE estimator-result archive (no metadata)"
            raise ValueError(msg)
        meta = json.loads(str(data["__meta__"]))
        if not isinstance(meta, dict) or meta.get("__format__") != _FORMAT:
            msg = "file is not a VANE estimator-result archive (wrong format tag)"
            raise ValueError(msg)
        type_name = meta.get("__type__")
        if type_name not in _RESULT_REGISTRY:
            msg = f"unknown estimator-result type {type_name!r}"
            raise ValueError(msg)
        cls = _RESULT_REGISTRY[type_name]
        meta_fields = meta.get("fields", {})
        array_names = _array_field_names(cls)
        kwargs: dict[str, Any] = {}
        for field in fields(cls):
            # A field is an array (in ``data.files``) or a non-array/None (in
            # ``meta_fields``). Present in neither means a corrupt or truncated archive:
            # reject it rather than default the field to ``None`` and build an invalid
            # result (the dataclass does not enforce its array annotations).
            if field.name in data.files:
                kwargs[field.name] = data[field.name]
            elif field.name in meta_fields:
                decoded = _decode_field(meta_fields[field.name])
                # An array-typed field may only be ``None`` in metadata (an absent
                # optional array); a non-``None`` value there is a tampered file that
                # would smuggle a list/scalar past the array contract.
                if field.name in array_names and decoded is not None:
                    msg = f"array field {field.name!r} must be an array, not metadata"
                    raise ValueError(msg)
                kwargs[field.name] = decoded
            else:
                msg = f"archive is missing required field {field.name!r}"
                raise ValueError(msg)
    result: _Result = cls(**kwargs)
    # Enforce the type's array contract regardless of the file's self-description: a
    # tampered archive that supplied a required array as a metadata ``null`` would have
    # built it as ``None`` here, which later breaks ``result.states.shape`` etc.
    for name in _required_array_fields(cls):
        if not isinstance(getattr(result, name), np.ndarray):
            # A corrupt-archive condition, not a caller type error: load_result's
            # contract is to raise ValueError for a bad file.
            msg = f"archive field {name!r} must be an array, not metadata"
            raise ValueError(msg)  # noqa: TRY004
    return result


def _validate_rom_family(family: RomFamily) -> None:
    """Validate the cross-system invariants needed for safe ROM-family reuse."""
    if type(family) is not RomFamily:
        msg = f"save_rom_family does not handle {type(family).__name__}"
        raise TypeError(msg)
    n_points = len(family.systems)
    if n_points == 0:
        msg = "ROM family must contain at least one system"
        raise ValueError(msg)
    arrays = {
        "operating_points": family.operating_points,
        "frequencies_hz": family.frequencies_hz,
        "damping_ratios": family.damping_ratios,
        "tracking_mac": family.tracking_mac,
    }
    for name, array in arrays.items():
        if not isinstance(array, np.ndarray):
            msg = f"ROM-family field {name!r} must be a NumPy array"
            raise TypeError(msg)
        if not np.all(np.isfinite(array)):
            msg = f"ROM-family field {name!r} must be finite"
            raise ValueError(msg)
    if family.operating_points.shape != (n_points,):
        msg = (
            "operating_points must have one value per system, got "
            f"{family.operating_points.shape} for {n_points} systems"
        )
        raise ValueError(msg)
    if n_points > 1:
        differences = np.diff(family.operating_points)
        if not (np.all(differences > 0.0) or np.all(differences < 0.0)):
            msg = (
                "operating_points must be strictly monotonic (ascending or descending)"
            )
            raise ValueError(msg)
    if family.frequencies_hz.ndim != 2:
        msg = "frequencies_hz must be a two-dimensional array"
        raise ValueError(msg)
    modal_shape = family.frequencies_hz.shape
    if modal_shape[0] != n_points:
        msg = "frequencies_hz must have one row per system"
        raise ValueError(msg)
    if family.damping_ratios.shape != modal_shape:
        msg = "damping_ratios shape must match frequencies_hz"
        raise ValueError(msg)
    if family.tracking_mac.shape != modal_shape:
        msg = "tracking_mac shape must match frequencies_hz"
        raise ValueError(msg)
    if np.any(family.tracking_mac < 0.0) or np.any(family.tracking_mac > 1.0):
        msg = "tracking_mac values must lie in the physical interval [0, 1]"
        raise ValueError(msg)
    if not isinstance(family.stabilized, np.ndarray):
        msg = "ROM-family field 'stabilized' must be a NumPy array"
        raise TypeError(msg)
    if family.stabilized.dtype != np.bool_:
        msg = "stabilized must be a boolean mask"
        raise ValueError(msg)
    if family.stabilized.shape != modal_shape:
        msg = "stabilized shape must match frequencies_hz"
        raise ValueError(msg)

    reference = family.systems[0]
    if type(reference) is not StateSpace:
        msg = "ROM-family systems must be exact StateSpace instances"
        raise TypeError(msg)
    expected_states = 2 * modal_shape[1]
    if reference.n_states != expected_states:
        msg = (
            f"ROM system has {reference.n_states} states but modal metadata implies "
            f"{expected_states}"
        )
        raise ValueError(msg)
    # D presence may differ across members: the family and its LPV blend treat a
    # missing feedthrough as zero, and the archive records presence per system. Only
    # the input/output dimensions a feedthrough must match are compared.
    signature = (
        reference.a.shape,
        None if reference.b is None else reference.b.shape,
        None if reference.c is None else reference.c.shape,
        (reference.n_outputs, reference.n_inputs),
        reference.state_names,
        reference.input_names,
        reference.output_names,
        reference.output_mbc_coordinates,
        reference.dt,
    )
    for index, system in enumerate(family.systems):
        if type(system) is not StateSpace:
            msg = f"ROM-family system {index} is not an exact StateSpace instance"
            raise TypeError(msg)
        candidate = (
            system.a.shape,
            None if system.b is None else system.b.shape,
            None if system.c is None else system.c.shape,
            (system.n_outputs, system.n_inputs),
            system.state_names,
            system.input_names,
            system.output_names,
            system.output_mbc_coordinates,
            system.dt,
        )
        if candidate != signature:
            msg = f"ROM-family system {index} does not share system 0's layout"
            raise ValueError(msg)


def save_rom_family(family: RomFamily, path: str | os.PathLike[str]) -> None:
    """Save a tracked :class:`RomFamily` as a transparent pickle-free archive.

    Each state-space matrix is stored under an explicit indexed key; names, sample
    times, and optional-matrix presence are JSON metadata.  No object arrays are used,
    so :func:`load_rom_family` always opens the archive with ``allow_pickle=False``.
    The per-mode reflection mask :attr:`RomFamily.stabilized` is stored alongside the
    modal metadata.
    """
    _validate_rom_family(family)
    systems_meta: list[dict[str, Any]] = []
    payload: dict[str, np.ndarray] = {
        "operating_points": family.operating_points,
        "frequencies_hz": family.frequencies_hz,
        "damping_ratios": family.damping_ratios,
        "tracking_mac": family.tracking_mac,
        "stabilized": family.stabilized,
    }
    for index, system in enumerate(family.systems):
        prefix = f"system_{index:04d}_"
        payload[prefix + "a"] = system.a
        presence: dict[str, bool] = {}
        for name in ("b", "c", "d"):
            matrix = getattr(system, name)
            presence[name] = matrix is not None
            if matrix is not None:
                payload[prefix + name] = matrix
        systems_meta.append(
            {
                "matrices_present": presence,
                "state_names": system.state_names,
                "input_names": system.input_names,
                "output_names": system.output_names,
                "output_mbc_coordinates": system.output_mbc_coordinates,
                "dt": system.dt,
            }
        )
    meta = {
        "__format__": _ROM_FAMILY_FORMAT,
        "__vane_version__": vane_version(),
        "n_systems": len(family.systems),
        "systems": systems_meta,
    }
    payload["__meta__"] = np.asarray(json.dumps(meta))
    np.savez(_npz_path(path), **payload)  # type: ignore[arg-type]


def load_rom_family(path: str | os.PathLike[str]) -> RomFamily:
    """Load a :class:`RomFamily` written by :func:`save_rom_family` safely.

    An archive without the :attr:`RomFamily.stabilized` mask loads with an
    all-``False`` one.
    """
    with np.load(_npz_path(path), allow_pickle=False) as data:
        if "__meta__" not in data:
            msg = "file is not a VANE ROM-family archive (no metadata)"
            raise ValueError(msg)
        meta = json.loads(str(data["__meta__"]))
        if not isinstance(meta, dict) or meta.get("__format__") != _ROM_FAMILY_FORMAT:
            msg = "file is not a VANE ROM-family archive (wrong format tag)"
            raise ValueError(msg)
        n_systems = meta.get("n_systems")
        systems_meta = meta.get("systems")
        if (
            not isinstance(n_systems, int)
            or n_systems <= 0
            or not isinstance(systems_meta, list)
            or len(systems_meta) != n_systems
        ):
            msg = "ROM-family archive has invalid system metadata"
            raise ValueError(msg)
        required = {
            "__meta__",
            "operating_points",
            "frequencies_hz",
            "damping_ratios",
            "tracking_mac",
        }
        systems: list[StateSpace] = []
        for index, system_meta in enumerate(systems_meta):
            if not isinstance(system_meta, dict):
                msg = f"ROM-family system metadata {index} is invalid"
                raise ValueError(msg)  # noqa: TRY004 - malformed archive contract
            presence = system_meta.get("matrices_present")
            if not isinstance(presence, dict) or set(presence) != {"b", "c", "d"}:
                msg = f"ROM-family system {index} has invalid matrix metadata"
                raise ValueError(msg)
            prefix = f"system_{index:04d}_"
            required.add(prefix + "a")
            matrices: dict[str, np.ndarray | None] = {}
            for name in ("b", "c", "d"):
                present = presence[name]
                if not isinstance(present, bool):
                    msg = f"ROM-family system {index} matrix flag {name!r} is invalid"
                    raise ValueError(msg)  # noqa: TRY004 - malformed archive contract
                key = prefix + name
                if present:
                    required.add(key)
                    if key not in data.files:
                        msg = f"ROM-family archive is missing matrix {key!r}"
                        raise ValueError(msg)
                    matrices[name] = np.asarray(data[key])
                else:
                    matrices[name] = None
            a_key = prefix + "a"
            if a_key not in data.files:
                msg = f"ROM-family archive is missing matrix {a_key!r}"
                raise ValueError(msg)
            try:
                systems.append(
                    StateSpace(
                        a=np.asarray(data[a_key]),
                        b=matrices["b"],
                        c=matrices["c"],
                        d=matrices["d"],
                        state_names=list(system_meta["state_names"]),
                        input_names=list(system_meta["input_names"]),
                        output_names=list(system_meta["output_names"]),
                        output_mbc_coordinates=list(
                            system_meta["output_mbc_coordinates"]
                        ),
                        dt=system_meta["dt"],
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                msg = f"ROM-family system {index} is invalid: {exc}"
                raise ValueError(msg) from exc
        missing = required.difference(data.files)
        if missing:
            msg = f"ROM-family archive is missing fields {sorted(missing)}"
            raise ValueError(msg)
        # ``stabilized`` is optional: an archive without the reflection mask loads
        # with an all-False mask (the RomFamily default).
        unexpected = set(data.files).difference(required | {"stabilized"})
        if unexpected:
            msg = f"ROM-family archive has unexpected fields {sorted(unexpected)}"
            raise ValueError(msg)
        frequencies = np.asarray(data["frequencies_hz"])
        stabilized = (
            np.asarray(data["stabilized"])
            if "stabilized" in data.files
            else np.zeros(frequencies.shape, dtype=np.bool_)
        )
        family = RomFamily(
            operating_points=np.asarray(data["operating_points"]),
            systems=systems,
            frequencies_hz=frequencies,
            damping_ratios=np.asarray(data["damping_ratios"]),
            tracking_mac=np.asarray(data["tracking_mac"]),
            stabilized=stabilized,
        )
    try:
        _validate_rom_family(family)
    except (TypeError, ValueError) as exc:
        msg = f"ROM-family archive is invalid: {exc}"
        raise ValueError(msg) from exc
    return family


def _system_layout(system: StateSpace) -> tuple[object, ...]:
    """Return the matrix shapes, names and sample time a family member must share."""
    return (
        system.a.shape,
        None if system.b is None else system.b.shape,
        None if system.c is None else system.c.shape,
        (system.n_outputs, system.n_inputs),
        system.state_names,
        system.input_names,
        system.output_names,
        system.output_mbc_coordinates,
        system.dt,
    )


def _require_names(value: object, label: str) -> list[str]:
    """Return ``value`` as a list of strings, or raise ``TypeError``."""
    if not isinstance(value, list) or not all(isinstance(n, str) for n in value):
        msg = f"LPV-family field {label!r} must be a list of strings"
        raise TypeError(msg)
    return value


def _validate_lpv_family(family: LPVFamily) -> None:
    """Validate the invariants an archived LPV family must satisfy."""
    if type(family) is not LPVFamily:
        msg = f"save_lpv_family does not handle {type(family).__name__}"
        raise TypeError(msg)
    if not isinstance(family.systems, list) or not family.systems:
        msg = "LPV family must contain at least one system"
        raise ValueError(msg)
    n_points = len(family.systems)
    for name in ("scheduling_points", "dropped_coupling_a", "dropped_coupling_c"):
        array = getattr(family, name)
        if not isinstance(array, np.ndarray):
            msg = f"LPV-family field {name!r} must be a NumPy array"
            raise TypeError(msg)
        if array.dtype.kind != "f":
            msg = f"LPV-family field {name!r} must be a floating-point array"
            raise TypeError(msg)
        if not np.all(np.isfinite(array)):
            msg = f"LPV-family field {name!r} must be finite"
            raise ValueError(msg)
    if family.scheduling_points.shape != (n_points,):
        msg = (
            "scheduling_points must have one value per system, got "
            f"{family.scheduling_points.shape} for {n_points} systems"
        )
        raise ValueError(msg)
    if n_points > 1 and not np.all(np.diff(family.scheduling_points) > 0.0):
        msg = "scheduling_points must be strictly ascending"
        raise ValueError(msg)
    for name in ("dropped_coupling_a", "dropped_coupling_c"):
        # The builder records one value per point; a family built by hand may leave
        # the norms at their empty default.
        if getattr(family, name).shape not in ((n_points,), (0,)):
            msg = f"{name} must have one value per system or be empty"
            raise ValueError(msg)
    for label in ("state_names", "input_names", "output_names", "auto_added_states"):
        _require_names(getattr(family, label), label)
    for label in ("scheduling_variable", "scheduling_unit"):
        if not isinstance(getattr(family, label), str):
            msg = f"LPV-family field {label!r} must be a string"
            raise TypeError(msg)
    reference: tuple[object, ...] | None = None
    for index, system in enumerate(family.systems):
        if type(system) is not StateSpace:
            msg = f"LPV-family system {index} is not an exact StateSpace instance"
            raise TypeError(msg)
        layout = _system_layout(system)
        if reference is None:
            reference = layout
        elif layout != reference:
            msg = f"LPV-family system {index} does not share system 0's layout"
            raise ValueError(msg)


def save_lpv_family(family: LPVFamily, path: str | os.PathLike[str]) -> None:
    """Save a physical-state :class:`~vane.sysid.physical_lpv.LPVFamily` to ``.npz``.

    The archive holds the scheduling points, the per-point dropped-coupling norms and
    each system's matrices as plain arrays under explicit indexed keys, and every
    other field (channel names, the scheduling variable and its unit, the auto-added
    states, each system's names, sample time and matrix presence) as JSON metadata.
    It is stamped with the format tag, :data:`LPV_FAMILY_SCHEMA_VERSION` and the
    producing VANE version. No object arrays are used, so :func:`load_lpv_family`
    opens it with ``allow_pickle=False``.

    Parameters
    ----------
    family : vane.sysid.physical_lpv.LPVFamily
        The family, e.g. from :func:`vane.export.build_physical_lpv_family`.
    path : str or os.PathLike
        Destination file (a ``.npz`` is written; the suffix is added if absent).

    Raises
    ------
    TypeError
        If ``family`` is not exactly an ``LPVFamily``, a system is not exactly a
        ``StateSpace``, or a field has the wrong type.
    ValueError
        If the family is empty, its arrays are non-finite or of the wrong length, the
        scheduling points are not strictly ascending, or the systems do not share one
        layout (matrix shapes, names and sample time).
    """
    _validate_lpv_family(family)
    payload: dict[str, np.ndarray] = {
        "scheduling_points": family.scheduling_points,
        "dropped_coupling_a": family.dropped_coupling_a,
        "dropped_coupling_c": family.dropped_coupling_c,
    }
    systems_meta: list[dict[str, Any]] = []
    for index, system in enumerate(family.systems):
        prefix = f"system_{index:04d}_"
        payload[prefix + "a"] = system.a
        presence: dict[str, bool] = {}
        for name in ("b", "c", "d"):
            matrix = getattr(system, name)
            presence[name] = matrix is not None
            if matrix is not None:
                payload[prefix + name] = matrix
        systems_meta.append(
            {
                "matrices_present": presence,
                "state_names": system.state_names,
                "input_names": system.input_names,
                "output_names": system.output_names,
                "output_mbc_coordinates": system.output_mbc_coordinates,
                "dt": system.dt,
            }
        )
    meta = {
        "__format__": _LPV_FAMILY_FORMAT,
        "__schema__": LPV_FAMILY_SCHEMA_VERSION,
        "__vane_version__": vane_version(),
        "n_systems": len(family.systems),
        "state_names": family.state_names,
        "input_names": family.input_names,
        "output_names": family.output_names,
        "scheduling_variable": family.scheduling_variable,
        "scheduling_unit": family.scheduling_unit,
        "auto_added_states": family.auto_added_states,
        "systems": systems_meta,
    }
    payload["__meta__"] = np.asarray(json.dumps(meta))
    np.savez(_npz_path(path), **payload)  # type: ignore[arg-type]


def _check_lpv_schema(meta: dict[str, Any]) -> None:
    """Raise ``ValueError`` unless the archive records the supported schema."""
    schema = meta.get("__schema__")
    written_by = meta.get("__vane_version__", "an unknown version")
    if schema is None:
        msg = (
            "LPV-family archive records no schema version; this VANE "
            f"{vane_version()} reads schema {LPV_FAMILY_SCHEMA_VERSION}"
        )
        raise ValueError(msg)
    if isinstance(schema, bool) or not isinstance(schema, int):
        msg = f"LPV-family archive has an invalid schema version {schema!r}"
        raise ValueError(msg)  # noqa: TRY004 - malformed archive contract
    if schema != LPV_FAMILY_SCHEMA_VERSION:
        remedy = (
            "upgrade VANE to read it"
            if schema > LPV_FAMILY_SCHEMA_VERSION
            else "re-create it from the .lin files with build_physical_lpv_family"
        )
        msg = (
            f"LPV-family archive uses schema {schema} (written by VANE {written_by}), "
            f"but this VANE {vane_version()} reads schema "
            f"{LPV_FAMILY_SCHEMA_VERSION} only; {remedy}"
        )
        raise ValueError(msg)


def _load_lpv_system(
    data: Any,  # noqa: ANN401 - an open NpzFile
    index: int,
    system_meta: object,
    required: set[str],
) -> StateSpace:
    """Rebuild member ``index`` of an LPV-family archive, recording its keys."""
    if not isinstance(system_meta, dict):
        msg = f"LPV-family system metadata {index} is invalid"
        raise ValueError(msg)  # noqa: TRY004 - malformed archive contract
    presence = system_meta.get("matrices_present")
    if not isinstance(presence, dict) or set(presence) != {"b", "c", "d"}:
        msg = f"LPV-family system {index} has invalid matrix metadata"
        raise ValueError(msg)
    prefix = f"system_{index:04d}_"
    matrices: dict[str, np.ndarray | None] = {}
    for name in ("a", "b", "c", "d"):
        present = True if name == "a" else presence[name]
        if not isinstance(present, bool):
            msg = f"LPV-family system {index} matrix flag {name!r} is invalid"
            raise ValueError(msg)  # noqa: TRY004 - malformed archive contract
        key = prefix + name
        if not present:
            matrices[name] = None
            continue
        if key not in data.files:
            msg = f"LPV-family archive is missing matrix {key!r}"
            raise ValueError(msg)
        required.add(key)
        matrices[name] = np.asarray(data[key])
    a_matrix = matrices["a"]
    if a_matrix is None:  # pragma: no cover - "a" is always present above
        msg = f"LPV-family archive is missing matrix {prefix + 'a'!r}"
        raise ValueError(msg)
    try:
        return StateSpace(
            a=a_matrix,
            b=matrices["b"],
            c=matrices["c"],
            d=matrices["d"],
            state_names=_require_names(system_meta["state_names"], "state_names"),
            input_names=_require_names(system_meta["input_names"], "input_names"),
            output_names=_require_names(system_meta["output_names"], "output_names"),
            output_mbc_coordinates=_require_names(
                system_meta["output_mbc_coordinates"], "output_mbc_coordinates"
            ),
            dt=system_meta["dt"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        msg = f"LPV-family system {index} is invalid: {exc}"
        raise ValueError(msg) from exc


def load_lpv_family(path: str | os.PathLike[str]) -> LPVFamily:
    """Load a physical-state LPV family saved by :func:`save_lpv_family`.

    The archive is opened with ``allow_pickle=False``. Its format tag and schema version
    are checked first, then every array and metadata field, and the rebuilt family is
    validated as on saving, so a file that loads is a family that
    :func:`save_lpv_family` could have written. The arrays, names, scheduling variable
    and unit, auto-added states and dropped-coupling norms come back exactly as saved.

    Parameters
    ----------
    path : str or os.PathLike
        The ``.npz`` file written by :func:`save_lpv_family` (the suffix is added if
        absent).

    Returns
    -------
    vane.sysid.physical_lpv.LPVFamily
        The reconstructed family.

    Raises
    ------
    ValueError
        If the file is not a VANE LPV-family archive, records no schema version or a
        schema other than :data:`LPV_FAMILY_SCHEMA_VERSION` (the message names both
        versions and the VANE version that wrote the file), or is incomplete, has
        unexpected entries or holds an invalid family.
    """
    with np.load(_npz_path(path), allow_pickle=False) as data:
        if "__meta__" not in data.files:
            msg = "file is not a VANE LPV-family archive (no metadata)"
            raise ValueError(msg)
        meta = json.loads(str(data["__meta__"]))
        if not isinstance(meta, dict) or meta.get("__format__") != _LPV_FAMILY_FORMAT:
            msg = "file is not a VANE LPV-family archive (wrong format tag)"
            raise ValueError(msg)
        _check_lpv_schema(meta)
        n_systems = meta.get("n_systems")
        systems_meta = meta.get("systems")
        if (
            isinstance(n_systems, bool)
            or not isinstance(n_systems, int)
            or n_systems <= 0
            or not isinstance(systems_meta, list)
            or len(systems_meta) != n_systems
        ):
            msg = "LPV-family archive has invalid system metadata"
            raise ValueError(msg)
        required = {
            "__meta__",
            "scheduling_points",
            "dropped_coupling_a",
            "dropped_coupling_c",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            msg = f"LPV-family archive is missing fields {missing}"
            raise ValueError(msg)
        systems = [
            _load_lpv_system(data, index, system_meta, required)
            for index, system_meta in enumerate(systems_meta)
        ]
        unexpected = sorted(set(data.files).difference(required))
        if unexpected:
            msg = f"LPV-family archive has unexpected fields {unexpected}"
            raise ValueError(msg)
        try:
            family = LPVFamily(
                scheduling_points=np.asarray(data["scheduling_points"]),
                systems=systems,
                state_names=_require_names(meta["state_names"], "state_names"),
                input_names=_require_names(meta["input_names"], "input_names"),
                output_names=_require_names(meta["output_names"], "output_names"),
                scheduling_variable=meta["scheduling_variable"],
                scheduling_unit=meta["scheduling_unit"],
                auto_added_states=_require_names(
                    meta["auto_added_states"], "auto_added_states"
                ),
                dropped_coupling_a=np.asarray(data["dropped_coupling_a"]),
                dropped_coupling_c=np.asarray(data["dropped_coupling_c"]),
            )
        except (KeyError, TypeError) as exc:
            msg = f"LPV-family archive metadata is invalid: {exc}"
            raise ValueError(msg) from exc
    try:
        _validate_lpv_family(family)
    except (TypeError, ValueError) as exc:
        msg = f"LPV-family archive is invalid: {exc}"
        raise ValueError(msg) from exc
    return family
