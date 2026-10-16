# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.utils.fs_retry -- retrying transient Windows file errors."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from vane.io.fst_writer import FastInputFile
from vane.utils import fs_retry
from vane.utils.fs_retry import (
    copy2_with_retry,
    is_transient_file_error,
    link_with_retry,
    retry_transient,
    rmtree_with_retry,
    unlink_with_retry,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _windows_error(winerror: int) -> PermissionError:
    """Return a PermissionError carrying a Windows error code, on any platform."""
    error = PermissionError(13, "Access is denied")
    error.winerror = winerror  # type: ignore[misc]
    return error


@pytest.fixture
def pauses(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record the backoff pauses instead of sleeping."""
    recorded: list[float] = []
    monkeypatch.setattr(fs_retry, "_pause", recorded.append)
    return recorded


def _failing(
    real: Callable[..., object], failures: int, winerror: int = 5
) -> tuple[Callable[..., object], list[int]]:
    """Wrap ``real`` so its first ``failures`` calls raise a transient error."""
    calls: list[int] = []

    def wrapper(*args: object, **kwargs: object) -> object:
        calls.append(1)
        if len(calls) <= failures:
            raise _windows_error(winerror)
        return real(*args, **kwargs)

    return wrapper, calls


class TestIsTransientFileError:
    """Tests for classifying an error as transient."""

    @pytest.mark.parametrize("winerror", [5, 32, 33])
    def test_windows_access_errors_are_transient(self, winerror: int) -> None:
        """Access-denied, sharing and lock violations are transient."""
        assert is_transient_file_error(_windows_error(winerror))

    def test_other_errors_are_not(self) -> None:
        """A POSIX permission error, another code or another type is not."""
        assert not is_transient_file_error(PermissionError(13, "denied"))
        assert not is_transient_file_error(_windows_error(2))
        assert not is_transient_file_error(FileNotFoundError(2, "missing"))
        assert not is_transient_file_error(ValueError("x"))


class TestRetryTransient:
    """Tests for the bounded retry loop."""

    def test_succeeds_after_transient_failures(
        self, pauses: list[float], caplog: pytest.LogCaptureFixture
    ) -> None:
        """Transient failures are retried with a doubling backoff, then succeed."""
        operation, calls = _failing(lambda: "done", failures=3, winerror=32)
        with caplog.at_level(logging.INFO, logger="vane.utils.fs_retry"):
            assert retry_transient(operation, description="test op") == "done"
        assert len(calls) == 4
        assert pauses == pytest.approx([0.05, 0.1, 0.2])
        assert "Transient file error on test op (try 1/5" in caplog.text

    def test_reraises_after_the_last_try(
        self, pauses: list[float], caplog: pytest.LogCaptureFixture
    ) -> None:
        """A persistent transient error is re-raised after ``tries`` calls."""
        operation, calls = _failing(lambda: None, failures=100)
        with pytest.raises(PermissionError, match="Access is denied"):
            retry_transient(operation, description="test op")
        assert len(calls) == 5
        assert len(pauses) == 4
        assert sum(pauses) == pytest.approx(0.75)
        assert "Could not test op after 5 tries" in caplog.text

    def test_non_transient_error_is_raised_at_once(self, pauses: list[float]) -> None:
        """A POSIX-style permission error (no winerror) is never retried."""
        calls: list[int] = []

        def operation() -> None:
            calls.append(1)
            raise PermissionError(13, "Permission denied")

        with pytest.raises(PermissionError):
            retry_transient(operation, description="test op")
        assert calls == [1]
        assert pauses == []

    def test_other_exception_types_propagate(self, pauses: list[float]) -> None:
        """An error that is not a PermissionError propagates unchanged."""

        def operation() -> None:
            raise FileNotFoundError(2, "missing")

        with pytest.raises(FileNotFoundError):
            retry_transient(operation, description="test op")
        assert pauses == []

    def test_rejects_bad_arguments(self) -> None:
        """Fewer than one try and a negative delay are rejected."""
        with pytest.raises(ValueError, match="tries"):
            retry_transient(lambda: None, description="x", tries=0)
        with pytest.raises(ValueError, match="initial_delay"):
            retry_transient(lambda: None, description="x", initial_delay=-1.0)

    def test_default_pause_sleeps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The default pause delegates to time.sleep."""
        slept: list[float] = []
        monkeypatch.setattr(fs_retry.time, "sleep", slept.append)
        fs_retry._pause(0.25)
        assert slept == [0.25]


class TestFileHelpers:
    """Tests for the retrying wrappers of the file operations."""

    def test_copy2_retries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pauses: list[float]
    ) -> None:
        """copy2_with_retry copies after transient failures."""
        source = tmp_path / "a.txt"
        source.write_text("hello", encoding="utf-8")
        wrapped, calls = _failing(shutil.copy2, failures=2)
        monkeypatch.setattr(shutil, "copy2", wrapped)
        result = copy2_with_retry(source, tmp_path / "b.txt")
        assert result == str(tmp_path / "b.txt")
        assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "hello"
        assert len(calls) == 3

    def test_link_retries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pauses: list[float]
    ) -> None:
        """link_with_retry hard-links after a transient failure."""
        source = tmp_path / "a.txt"
        source.write_text("hello", encoding="utf-8")
        wrapped, calls = _failing(os.link, failures=1)
        monkeypatch.setattr(os, "link", wrapped)
        link_with_retry(source, tmp_path / "b.txt")
        assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "hello"
        assert len(calls) == 2

    def test_unlink_retries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pauses: list[float]
    ) -> None:
        """unlink_with_retry deletes after a transient failure; missing_ok works."""
        target = tmp_path / "a.txt"
        target.write_text("x", encoding="utf-8")
        wrapped, calls = _failing(Path.unlink, failures=1)
        monkeypatch.setattr(Path, "unlink", wrapped)
        unlink_with_retry(target)
        assert not target.exists()
        assert len(calls) == 2
        unlink_with_retry(target, missing_ok=True)
        with pytest.raises(FileNotFoundError):
            unlink_with_retry(target)

    def test_rmtree_retries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pauses: list[float]
    ) -> None:
        """rmtree_with_retry removes the tree after transient failures."""
        tree = tmp_path / "case"
        (tree / "sub").mkdir(parents=True)
        (tree / "sub" / "f.dat").write_text("x", encoding="utf-8")
        wrapped, calls = _failing(shutil.rmtree, failures=2)
        monkeypatch.setattr(shutil, "rmtree", wrapped)
        rmtree_with_retry(tree)
        assert not tree.exists()
        assert len(calls) == 3

    def test_rmtree_always_failing_reraises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pauses: list[float]
    ) -> None:
        """A tree that stays locked raises the last error after five tries."""
        tree = tmp_path / "case"
        tree.mkdir()
        wrapped, calls = _failing(shutil.rmtree, failures=100)
        monkeypatch.setattr(shutil, "rmtree", wrapped)
        with pytest.raises(PermissionError):
            rmtree_with_retry(tree)
        assert len(calls) == 5
        assert tree.exists()


class TestDeckWriteRetry:
    """FastInputFile.write survives a transiently locked replace."""

    def test_write_retries_a_denied_replace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pauses: list[float]
    ) -> None:
        """A replace denied twice succeeds on the third try, leaving no temp file."""
        deck_path = tmp_path / "ED.dat"
        deck_path.write_text("9.55 RotSpeed - rpm\n", encoding="utf-8")
        deck = FastInputFile.read(deck_path)
        deck.set("RotSpeed", 7.0)
        wrapped, calls = _failing(Path.replace, failures=2)
        monkeypatch.setattr(Path, "replace", wrapped)
        deck.write()
        assert len(calls) == 3
        assert FastInputFile.read(deck_path).get("RotSpeed") == "7.0"
        assert sorted(path.name for path in tmp_path.iterdir()) == ["ED.dat"]

    def test_write_always_denied_raises_and_cleans_up(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pauses: list[float]
    ) -> None:
        """A persistently denied replace raises, leaving the deck and no temps."""
        deck_path = tmp_path / "ED.dat"
        deck_path.write_text("9.55 RotSpeed - rpm\n", encoding="utf-8")
        deck = FastInputFile.read(deck_path)
        deck.set("RotSpeed", 7.0)
        wrapped, calls = _failing(Path.replace, failures=100)
        monkeypatch.setattr(Path, "replace", wrapped)
        with pytest.raises(PermissionError):
            deck.write()
        assert len(calls) == 5
        assert FastInputFile.read(deck_path).get("RotSpeed") == "9.55"
        assert sorted(path.name for path in tmp_path.iterdir()) == ["ED.dat"]

    def test_settle_copy_edit_retries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pauses: list[float]
    ) -> None:
        """The settle path's byte-level replace retries a denied replace too."""
        from vane.run.steady_state import _replace_file

        target = tmp_path / "HD.dat"
        target.write_bytes(b"old\n")
        wrapped, calls = _failing(Path.replace, failures=1)
        monkeypatch.setattr(Path, "replace", wrapped)
        _replace_file(target, b"new\n")
        assert target.read_bytes() == b"new\n"
        assert len(calls) == 2

    def test_json_write_retries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pauses: list[float]
    ) -> None:
        """The atomic JSON writer (provenance records) retries a denied replace."""
        from vane._json import write_json_atomic

        target = tmp_path / "provenance.json"
        target.write_text("{}", encoding="utf-8")
        wrapped, calls = _failing(Path.replace, failures=1)
        monkeypatch.setattr(Path, "replace", wrapped)
        write_json_atomic(target, {"a": 1}, overwrite=True)
        assert '"a": 1' in target.read_text(encoding="utf-8")
        assert len(calls) == 2

    def test_json_install_is_not_replayed_when_cleanup_is_locked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pauses: list[float]
    ) -> None:
        """A locked temporary file after a hard-link install is retried on its own.

        Replaying the whole write would make the no-overwrite install see its own
        destination and report it as already existing.
        """
        from vane._json import write_json_atomic

        target = tmp_path / "record.json"
        wrapped, calls = _failing(Path.unlink, failures=1)
        monkeypatch.setattr(Path, "unlink", wrapped)
        write_json_atomic(target, {"a": 1}, overwrite=False)
        assert '"a": 1' in target.read_text(encoding="utf-8")
        assert len(calls) == 2  # one denied cleanup, one successful retry
        assert not list(tmp_path.glob("record.json.*.tmp"))
