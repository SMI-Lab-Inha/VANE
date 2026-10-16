# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Internal strict, atomic JSON writing support."""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from pathlib import Path

from vane.utils.fs_retry import retry_transient, unlink_with_retry

logger = logging.getLogger(__name__)


def loads_json_strict(serialized: str) -> object:
    """Parse JSON while rejecting nonstandard numbers and duplicate object names.

    Besides the ``NaN``/``Infinity`` constants, a finite-looking literal that
    overflows to infinity when parsed (e.g. ``1e999``) is rejected too, so a
    strictly parsed document never contains a non-finite float.
    """

    def reject_constant(value: str) -> object:
        msg = f"nonstandard JSON numeric constant is not permitted: {value}"
        raise ValueError(msg)

    def finite_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            msg = f"JSON number overflows to a non-finite float: {value}"
            raise ValueError(msg)
        return number

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for name, value in pairs:
            if name in result:
                msg = f"duplicate JSON object name is not permitted: {name}"
                raise ValueError(msg)
            result[name] = value
        return result

    result: object = json.loads(
        serialized,
        parse_constant=reject_constant,
        parse_float=finite_float,
        object_pairs_hook=unique_object,
    )
    return result


def write_json_atomic(
    path: str | Path,
    payload: object,
    *,
    overwrite: bool,
    sort_keys: bool = True,
) -> Path:
    """Write strict JSON through an exclusive same-directory temporary file.

    When ``overwrite`` is false, installation uses an atomic hard-link operation so
    a destination created by another process after preflight is never replaced. A
    transient Windows access-denied or sharing-violation error retries the step that
    hit it (see :func:`vane.utils.fs_retry.retry_transient`): writing and installing
    together until the destination is installed, then removing the temporary file on
    its own, so a completed installation is never replayed (without ``overwrite`` a
    replay would report the destination as already present).
    """
    destination = Path(path)
    serialized = (
        json.dumps(payload, indent=2, sort_keys=sort_keys, allow_nan=False) + "\n"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporaries: list[Path] = []

    def write_once() -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f"{destination.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        temporaries.append(temporary)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            temporary.replace(destination)
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError as exc:
                msg = f"JSON destination already exists: {destination}"
                raise FileExistsError(msg) from exc

    try:
        retry_transient(write_once, description=f"write {destination.name}")
    finally:
        for temporary in temporaries:
            _remove_quietly(temporary)
    return destination


def _remove_quietly(path: Path) -> None:
    """Remove a temporary file (with retry), only warning when it cannot be removed."""
    try:
        unlink_with_retry(path, missing_ok=True)
    except OSError:
        logger.warning("could not remove temporary file %s", path)
