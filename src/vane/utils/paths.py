# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
r"""Path resolution that is safe against Windows extended-length (``\\?\``) paths.

On Windows, :meth:`pathlib.Path.resolve` can return a path with the extended-length
prefix ``\\?\`` (CPython keeps it when a transient error, such as a directory another
process is creating or deleting, stops it confirming the plain form). Under that prefix
Windows does not collapse ``..`` segments, so an OpenFAST deck that references a shared
sub-file as ``../Shared/File.dat`` no longer resolves. These helpers drop the prefix
when the plain path fits within ``MAX_PATH`` and otherwise collapse ``..`` explicitly.
"""

from __future__ import annotations

import ntpath
from pathlib import Path

__all__ = [
    "has_extended_prefix",
    "normalize_extended_path",
    "plain_path_text",
    "resolve_path",
]

_EXTENDED_PREFIX = "\\\\?\\"
_EXTENDED_UNC_PREFIX = "\\\\?\\UNC\\"
_MAX_PATH = 260  # Windows' classic path-length limit, including the terminator


def _normalize_extended_text(text: str) -> str:
    """Return ``text`` with an extended-length prefix normalized (see module doc)."""
    if text.startswith(_EXTENDED_UNC_PREFIX):
        prefix, body = _EXTENDED_UNC_PREFIX, "\\\\" + text[len(_EXTENDED_UNC_PREFIX) :]
    elif text.startswith(_EXTENDED_PREFIX):
        prefix, body = _EXTENDED_PREFIX, text[len(_EXTENDED_PREFIX) :]
    else:
        return text
    body = ntpath.normpath(body)
    if len(body) < _MAX_PATH:
        return body
    if prefix == _EXTENDED_UNC_PREFIX:
        return prefix + body[2:]
    return prefix + body


def normalize_extended_path(path: Path) -> Path:
    r"""Return ``path`` without a needless extended-length prefix.

    A path carrying the Windows ``\\?\`` prefix has its ``..`` segments collapsed
    and loses the prefix when the plain form fits within ``MAX_PATH``; a longer path
    keeps the prefix (it is needed there) but with ``..`` collapsed. Any other path
    is returned unchanged, so the function is a no-op off Windows.

    Parameters
    ----------
    path : pathlib.Path
        The path to normalize.

    Returns
    -------
    pathlib.Path
        The normalized path.
    """
    text = str(path)
    normalized = _normalize_extended_text(text)
    return path if normalized == text else Path(normalized)


def has_extended_prefix(path: str | Path) -> bool:
    r"""Return whether ``path`` carries the Windows extended-length ``\\?\`` prefix.

    Parameters
    ----------
    path : str or pathlib.Path
        The path to test.

    Returns
    -------
    bool
        ``True`` when the path text starts with ``\\?\``.
    """
    return str(path).startswith(_EXTENDED_PREFIX)


def plain_path_text(path: str | Path) -> str:
    r"""Return ``path`` as text without any extended-length ``\\?\`` prefix.

    Unlike :func:`normalize_extended_path`, the prefix is dropped even when the plain
    form exceeds ``MAX_PATH``: the result is the spelling a program that is not
    long-path aware (such as OpenFAST) would have to use, so its length can be
    checked against that program's limit.

    Parameters
    ----------
    path : str or pathlib.Path
        The path to convert.

    Returns
    -------
    str
        The path text, with ``..`` collapsed and the prefix removed when it had one;
        unchanged otherwise.
    """
    text = str(path)
    if text.startswith(_EXTENDED_UNC_PREFIX):
        return ntpath.normpath("\\\\" + text[len(_EXTENDED_UNC_PREFIX) :])
    if text.startswith(_EXTENDED_PREFIX):
        return ntpath.normpath(text[len(_EXTENDED_PREFIX) :])
    return text


def resolve_path(path: str | Path) -> Path:
    r"""Resolve ``path`` to an absolute path without a needless ``\\?\`` prefix.

    Parameters
    ----------
    path : str or pathlib.Path
        The path to resolve.

    Returns
    -------
    pathlib.Path
        ``Path(path).resolve()`` passed through :func:`normalize_extended_path`.
    """
    return normalize_extended_path(Path(path).resolve())
