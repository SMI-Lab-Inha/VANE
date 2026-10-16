# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Keep or remove linearization states by description pattern or DOF class.

:func:`select_states` returns a copy of a linearized model with a subset of its
continuous states: the matching rows and columns of ``A``, the rows of ``B`` and the
columns of ``C`` follow the selection, and ``D`` is unchanged. It works on a raw
:class:`~vane.io.lin_reader.LinFile` (before the MBC3 transform) and on an
:class:`~vane.io.mbc_transform.MBCResult` (after it).

A second-order degree of freedom has two states, its displacement and its velocity
(``First time derivative of ...``). The two are always selected together: a pattern or
class that matches either one keeps, or removes, both, so the ``[q2, q2dot, q1]``
layout the MBC3 transform and the eigensolver rely on stays consistent.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, overload

import numpy as np

from vane.config.dof_map import DofCategory, classify_dof
from vane.io.lin_reader import LinFile, OperatingPointTable
from vane.io.mbc_transform import (
    _DERIV_MARKER_RE,
    MBCResult,
    _marker_partner_key,
    _pair_key,
)

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = ["select_states"]

# ElastoDyn names each DOF in its description, which pairs a velocity with its
# displacement even where the two descriptions differ otherwise.
_DOF_INDEX_RE = re.compile(r"internal DOF index\s*=\s*([A-Za-z0-9_(),]+)\)")
_Patterns = str | Sequence[str] | None
_Categories = Iterable[DofCategory | str] | None


def _as_patterns(patterns: _Patterns, name: str) -> list[re.Pattern[str]]:
    """Compile one pattern or a sequence of patterns (case-insensitive)."""
    if patterns is None:
        return []
    items = [patterns] if isinstance(patterns, str) else list(patterns)
    compiled: list[re.Pattern[str]] = []
    for item in items:
        if not isinstance(item, str):
            msg = f"{name} must be a regular expression or a sequence of them"
            raise TypeError(msg)
        try:
            compiled.append(re.compile(item, re.IGNORECASE))
        except re.error as exc:
            msg = f"{name}: invalid regular expression {item!r}: {exc}"
            raise ValueError(msg) from exc
    return compiled


def _as_categories(categories: _Categories, name: str) -> set[DofCategory]:
    """Convert DOF categories given as members or their string values."""
    if categories is None:
        return set()
    if isinstance(categories, (str, DofCategory)):
        categories = [categories]
    known = {member.value: member for member in DofCategory}
    out: set[DofCategory] = set()
    for item in categories:
        if isinstance(item, DofCategory):
            out.add(item)
        elif item in known:
            out.add(known[item])
        else:
            msg = f"{name}: unknown DOF category {item!r}"
            raise ValueError(msg)
    return out


def _lin_partners(descriptions: Sequence[str]) -> list[int]:
    """Return, for each state of a raw ``.lin`` file, the index of its pair partner.

    A state with no partner (a first-order state) is its own partner.
    """
    partners = list(range(len(descriptions)))
    by_key: dict[str, int] = {}
    by_dof: dict[str, int] = {}
    for index, description in enumerate(descriptions):
        if not _DERIV_MARKER_RE.search(description):
            by_key.setdefault(_pair_key(description).casefold(), index)
            dof = _DOF_INDEX_RE.search(description)
            if dof is not None:
                by_dof.setdefault(dof.group(1), index)
    for index, description in enumerate(descriptions):
        if _DERIV_MARKER_RE.search(description):
            dof = _DOF_INDEX_RE.search(description)
            partner = by_dof.get(dof.group(1)) if dof is not None else None
            if partner is None:
                partner = by_key.get(_marker_partner_key(description))
            if partner is not None:
                partners[index] = partner
                partners[partner] = index
    return partners


def _mbc_partners(n_states: int, ndof2: int) -> list[int]:
    """Return the pair partner of each state of the ``[q2, q2dot, q1]`` layout."""
    partners = list(range(n_states))
    for i in range(ndof2):
        partners[i] = i + ndof2
        partners[i + ndof2] = i
    return partners


def _matches(
    description: str,
    patterns: list[re.Pattern[str]],
    categories: set[DofCategory],
) -> bool:
    """Return whether a state description matches any pattern or category."""
    if any(p.search(description) for p in patterns):
        return True
    return bool(categories) and classify_dof(description).category in categories


def _selection(
    descriptions: Sequence[str],
    partners: Sequence[int],
    keep: _Patterns,
    drop: _Patterns,
    keep_categories: _Categories,
    drop_categories: _Categories,
) -> npt.NDArray[np.bool_]:
    """Return the boolean mask of retained states, with pairs selected together."""
    keep_p = _as_patterns(keep, "keep")
    drop_p = _as_patterns(drop, "drop")
    keep_c = _as_categories(keep_categories, "keep_categories")
    drop_c = _as_categories(drop_categories, "drop_categories")
    if not (keep_p or drop_p or keep_c or drop_c):
        msg = "give at least one of keep, drop, keep_categories or drop_categories"
        raise ValueError(msg)
    # Every selector must match at least one state: a pattern that matches nothing is
    # almost always a typing error, and silently ignoring it would hide that.
    for label, pats, cats in (("keep", keep_p, keep_c), ("drop", drop_p, drop_c)):
        for pattern in pats:
            if not any(pattern.search(d) for d in descriptions):
                msg = f"{label} pattern {pattern.pattern!r} matches no state"
                raise ValueError(msg)
        for category in cats:
            if not any(classify_dof(d).category is category for d in descriptions):
                msg = f"{label} category {category.value!r} matches no state"
                raise ValueError(msg)

    n = len(descriptions)
    direct_keep = np.array(
        [_matches(d, keep_p, keep_c) for d in descriptions], dtype=np.bool_
    )
    direct_drop = np.array(
        [_matches(d, drop_p, drop_c) for d in descriptions], dtype=np.bool_
    )
    partner = np.asarray(partners, dtype=np.intp)
    selected_keep: npt.NDArray[np.bool_] = direct_keep | direct_keep[partner]
    selected_drop: npt.NDArray[np.bool_] = direct_drop | direct_drop[partner]
    base = selected_keep if (keep_p or keep_c) else np.ones(n, dtype=np.bool_)
    mask: npt.NDArray[np.bool_] = np.logical_and(base, np.logical_not(selected_drop))
    if not mask.any():
        msg = "the selection removes every state"
        raise ValueError(msg)
    return mask


def _take_table(
    table: OperatingPointTable, mask: npt.NDArray[np.bool_]
) -> OperatingPointTable:
    """Return the rows of an operating-point table selected by ``mask``."""
    index = np.flatnonzero(mask)
    return OperatingPointTable(
        values=table.values[index],
        rotating_frame=table.rotating_frame[index],
        derivative_order=table.derivative_order[index],
        descriptions=[table.descriptions[i] for i in index],
    )


def _rows_cols(
    a: npt.NDArray[np.float64] | None,
    b: npt.NDArray[np.float64] | None,
    c: npt.NDArray[np.float64] | None,
    index: npt.NDArray[np.intp],
) -> tuple[
    npt.NDArray[np.float64] | None,
    npt.NDArray[np.float64] | None,
    npt.NDArray[np.float64] | None,
]:
    """Return ``A[index][:, index]``, ``B[index]`` and ``C[:, index]``."""
    return (
        None if a is None else a[np.ix_(index, index)],
        None if b is None else b[index, :],
        None if c is None else c[:, index],
    )


@overload
def select_states(
    model: LinFile,
    *,
    keep: _Patterns = ...,
    drop: _Patterns = ...,
    keep_categories: _Categories = ...,
    drop_categories: _Categories = ...,
) -> LinFile: ...


@overload
def select_states(
    model: MBCResult,
    *,
    keep: _Patterns = ...,
    drop: _Patterns = ...,
    keep_categories: _Categories = ...,
    drop_categories: _Categories = ...,
) -> MBCResult: ...


def select_states(
    model: LinFile | MBCResult,
    *,
    keep: _Patterns = None,
    drop: _Patterns = None,
    keep_categories: _Categories = None,
    drop_categories: _Categories = None,
) -> LinFile | MBCResult:
    r"""Return a copy of a linearized model with a subset of its states.

    Parameters
    ----------
    model : LinFile or MBCResult
        A parsed ``.lin`` file (before the MBC3 transform) or the result of
        :func:`~vane.io.mbc_transform.mbc3_transform` (after it).
    keep : str or Sequence[str] or None, optional
        Regular expressions (case-insensitive, matched anywhere in the state
        description). When ``keep`` or ``keep_categories`` is given, only the states
        that match one of them are retained.
    drop : str or Sequence[str] or None, optional
        Regular expressions of states to remove. Removal is applied after ``keep``.
    keep_categories, drop_categories : Iterable[DofCategory or str] or None, optional
        Degree-of-freedom classes (:class:`~vane.config.dof_map.DofCategory` members
        or their string values, as returned by
        :func:`~vane.config.dof_map.classify_dof`) to retain or remove.

    Returns
    -------
    LinFile or MBCResult
        A new object of the same type. ``A`` keeps the selected rows and columns,
        ``B`` the selected rows and ``C`` the selected columns; ``D``, the inputs and
        the outputs are unchanged. For an :class:`~vane.io.mbc_transform.MBCResult`
        the per-azimuth matrices, ``ndof2``, ``ndof1`` and the MBC coordinate labels
        follow the selection. For a :class:`~vane.io.lin_reader.LinFile` the module
        Jacobians, whose dimensions involve the states, are not carried over.

    Raises
    ------
    ValueError
        If no selector is given, a pattern or category matches no state, a pattern
        is not a valid regular expression, a category is unknown, or the selection
        removes every state.
    TypeError
        If ``model`` is neither a ``LinFile`` nor an ``MBCResult``, or a pattern is
        not a string.

    Notes
    -----
    The displacement and velocity of a second-order degree of freedom are selected
    together: if either description matches, both states are kept (or removed). Use
    the function before the MBC3 transform to remove a module's states from every
    azimuth, or after it to reduce the averaged model; in both cases the result can
    be passed on unchanged (for example to :func:`~vane.io.mbc_transform.mbc3_transform`
    or :func:`~vane.modal.eigensolver.modes_from_mbc`).

    Examples
    --------
    >>> reduced = select_states(lin, drop=r"\bHD\b")  # doctest: +SKIP
    >>> tower = select_states(
    ...     result, keep_categories=["tower_fore_aft_1", "tower_side_side_1"]
    ... )  # doctest: +SKIP
    """
    if isinstance(model, LinFile):
        descriptions = model.x.descriptions
        mask = _selection(
            descriptions,
            _lin_partners(descriptions),
            keep,
            drop,
            keep_categories,
            drop_categories,
        )
        index = np.flatnonzero(mask)
        a, b, c = _rows_cols(model.a, model.b, model.c, index)
        return replace(
            model,
            x=_take_table(model.x, mask),
            xdot=_take_table(model.xdot, mask),
            a=a,
            b=b,
            c=c,
            jacobians={},
            jacobians_included=False,
        )
    if isinstance(model, MBCResult):
        descriptions = model.state_descriptions
        n_states = len(descriptions)
        mask = _selection(
            descriptions,
            _mbc_partners(n_states, model.ndof2),
            keep,
            drop,
            keep_categories,
            drop_categories,
        )
        index = np.flatnonzero(mask)
        a, b, c = _rows_cols(model.avg_a, model.avg_b, model.avg_c, index)
        ndof2 = int(mask[: model.ndof2].sum())
        ndof1 = int(mask[2 * model.ndof2 :].sum())
        coordinates = (
            [model.mbc_coordinates[i] for i in index]
            if len(model.mbc_coordinates) == n_states
            else list(model.mbc_coordinates)
        )

        def _stack(
            matrices: list[npt.NDArray[np.float64]] | None, kind: str
        ) -> list[npt.NDArray[np.float64]] | None:
            if matrices is None:
                return None
            if kind == "a":
                return [m[np.ix_(index, index)] for m in matrices]
            if kind == "b":
                return [m[index, :] for m in matrices]
            return [m[:, index] for m in matrices]

        return replace(
            model,
            avg_a=a,
            avg_b=b,
            avg_c=c,
            state_descriptions=[descriptions[i] for i in index],
            ndof2=ndof2,
            ndof1=ndof1,
            mbc_coordinates=coordinates,
            per_azimuth_a=_stack(model.per_azimuth_a, "a"),
            per_azimuth_b=_stack(model.per_azimuth_b, "b"),
            per_azimuth_c=_stack(model.per_azimuth_c, "c"),
        )
    msg = f"model must be a LinFile or an MBCResult, got {type(model).__name__}"
    raise TypeError(msg)
