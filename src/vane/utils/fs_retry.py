# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Bounded retry for transient Windows file-system errors.

On Windows, renaming, replacing, copying or deleting a file fails transiently with
``ERROR_ACCESS_DENIED`` (5), ``ERROR_SHARING_VIOLATION`` (32) or
``ERROR_LOCK_VIOLATION`` (33) while another process briefly holds a handle on the
file or its directory: an antivirus or search-indexer scan of a freshly written file,
or an OpenFAST process that has just exited and is still releasing its handles. Python
raises these as :class:`PermissionError` carrying the code in ``winerror``. A sweep that
writes hundreds of case files across many parallel workers meets them sooner or later,
and retrying the operation a moment later succeeds.

:func:`retry_transient` runs an operation and retries it, with a short exponential
backoff, only on such an error; any other exception -- including a POSIX
:class:`PermissionError`, which never carries ``winerror`` and is a real permission
problem -- propagates at once, so on POSIX the helpers behave exactly like the
operations they wrap. The last error is re-raised once the tries are exhausted.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "TRANSIENT_WINERRORS",
    "copy2_with_retry",
    "is_transient_file_error",
    "link_with_retry",
    "retry_transient",
    "rmtree_with_retry",
    "unlink_with_retry",
]

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

#: Windows error codes treated as transient: access denied, sharing violation and
#: lock violation.
TRANSIENT_WINERRORS: frozenset[int] = frozenset({5, 32, 33})

_DEFAULT_TRIES = 5
_DEFAULT_INITIAL_DELAY = 0.05  # s; doubled per retry: 0.05 + 0.1 + 0.2 + 0.4 s


def is_transient_file_error(error: BaseException) -> bool:
    """Return whether ``error`` is a transient Windows file-access error.

    Parameters
    ----------
    error : BaseException
        The exception raised by a file-system operation.

    Returns
    -------
    bool
        ``True`` for a :class:`PermissionError` whose ``winerror`` is one of
        :data:`TRANSIENT_WINERRORS`; ``False`` otherwise, and always on POSIX, where
        the attribute is absent.
    """
    if not isinstance(error, PermissionError):
        return False
    return getattr(error, "winerror", None) in TRANSIENT_WINERRORS


def _pause(seconds: float) -> None:
    """Sleep between tries (a seam the tests replace)."""
    time.sleep(seconds)


def retry_transient(
    operation: Callable[[], _T],
    *,
    description: str,
    tries: int = _DEFAULT_TRIES,
    initial_delay: float = _DEFAULT_INITIAL_DELAY,
) -> _T:
    """Run ``operation``, retrying it on a transient Windows file-access error.

    Parameters
    ----------
    operation : Callable[[], T]
        The operation to run. It is called again from the start on each retry, so it
        must be safe to repeat after a failed try (e.g. it cleans up its own temporary
        file before raising).
    description : str
        What the operation does, for the log message (e.g. ``"replace deck.fst"``).
    tries : int, optional
        The maximum number of calls (default 5).
    initial_delay : float, optional
        The pause before the first retry in seconds (default 0.05), doubled before
        each further retry -- about 0.75 s in all with the defaults.

    Returns
    -------
    T
        The operation's return value.

    Raises
    ------
    ValueError
        If ``tries`` is less than 1 or ``initial_delay`` is negative.
    PermissionError
        The last transient error, once ``tries`` calls have failed.
    Exception
        Any non-transient error raised by ``operation``, at once.
    """
    if tries < 1:
        msg = f"tries must be >= 1, got {tries}"
        raise ValueError(msg)
    if not initial_delay >= 0.0:
        msg = f"initial_delay must be non-negative, got {initial_delay}"
        raise ValueError(msg)
    delay = initial_delay
    attempt = 1
    while True:
        try:
            return operation()
        except PermissionError as error:
            if not is_transient_file_error(error) or attempt >= tries:
                if attempt > 1:
                    logger.warning(
                        "Could not %s after %d tries: %s", description, attempt, error
                    )
                raise
            logger.info(
                "Transient file error on %s (try %d/%d, retrying in %.2g s): %s",
                description,
                attempt,
                tries,
                delay,
                error,
            )
        _pause(delay)
        delay *= 2.0
        attempt += 1


def copy2_with_retry(source: str | Path, destination: str | Path) -> str:
    """Copy a file with :func:`shutil.copy2`, with retry.

    Parameters
    ----------
    source, destination : str or pathlib.Path
        The file to copy and its destination.

    Returns
    -------
    str
        The destination, as :func:`shutil.copy2` returns it (usable as a
        :func:`shutil.copytree` ``copy_function``).
    """
    copied = retry_transient(
        lambda: shutil.copy2(source, destination),
        description=f"copy {Path(source).name}",
    )
    return str(copied)


def link_with_retry(source: str | Path, destination: str | Path) -> None:
    """Hard-link ``destination`` to ``source`` with :func:`os.link`, with retry.

    Parameters
    ----------
    source, destination : str or pathlib.Path
        The existing file and the new link.
    """
    retry_transient(
        lambda: os.link(source, destination),
        description=f"hard-link {Path(source).name}",
    )


def unlink_with_retry(path: str | Path, *, missing_ok: bool = False) -> None:
    """Delete a file with :meth:`pathlib.Path.unlink`, with retry.

    Parameters
    ----------
    path : str or pathlib.Path
        The file to delete.
    missing_ok : bool, optional
        Whether a missing file is ignored (default ``False``).
    """
    target = Path(path)
    retry_transient(
        lambda: target.unlink(missing_ok=missing_ok),
        description=f"delete {target.name}",
    )


def rmtree_with_retry(path: str | Path) -> None:
    """Delete a directory tree with :func:`shutil.rmtree`, with retry.

    A retry resumes the deletion: whatever the failed try already removed stays
    removed, and the tree's root, which is removed last, still exists.

    Parameters
    ----------
    path : str or pathlib.Path
        The directory to delete.
    """
    retry_transient(
        lambda: shutil.rmtree(path),
        description=f"delete the directory {Path(path).name}",
    )
