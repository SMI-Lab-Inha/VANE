# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.run.runner — running OpenFAST and collecting .lin output."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path, PureWindowsPath
from subprocess import CompletedProcess
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from vane.run import (
    DeckPathLengthError,
    OpenFastError,
    OpenFastFailureMode,
    OpenFastResult,
    OpenFastTimeoutError,
    classify_openfast_output,
    find_lin_files,
    find_openfast,
    run_openfast,
)
from vane.run import runner as runner_module

if TYPE_CHECKING:
    from collections.abc import Callable


def _patch_process(
    monkeypatch: pytest.MonkeyPatch,
    fake_run: Callable[..., CompletedProcess[str]],
) -> None:
    """Replace the process-tree runner with a ``subprocess.run``-style fake."""

    def fake_tree(
        command: list[str], cwd: Path, timeout: float | None, fst: Path
    ) -> tuple[int, str, str]:
        completed = fake_run(command, cwd=cwd, timeout=timeout)
        return completed.returncode, completed.stdout, completed.stderr

    monkeypatch.setattr(runner_module, "_run_process_tree", fake_tree)


def _pid_alive(pid: int) -> bool:
    """Return whether a process with ``pid`` is still running."""
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        synchronize_and_query = 0x00100000 | 0x1000
        handle = kernel32.OpenProcess(synchronize_and_query, False, pid)
        if not handle:
            return False
        try:
            return bool(kernel32.WaitForSingleObject(handle, 0) != 0)
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class TestFindLinFiles:
    """Tests for discovering full-system .lin files."""

    def test_filters_and_orders_by_index(self, tmp_path: Path) -> None:
        """Only ``<root>.<i>.lin`` are returned, ordered by integer index."""
        root = "model"
        for name in (
            f"{root}.1.lin",
            f"{root}.2.lin",
            f"{root}.10.lin",
            f"{root}.ED.1.lin",  # per-module file, excluded
            "other.1.lin",  # different root, excluded
        ):
            (tmp_path / name).write_text("", encoding="utf-8")
        files = find_lin_files(tmp_path / f"{root}.fst")
        assert [f.name for f in files] == [
            f"{root}.1.lin",
            f"{root}.2.lin",
            f"{root}.10.lin",
        ]

    def test_no_lin_files_returns_empty(self, tmp_path: Path) -> None:
        """A deck with no outputs yields an empty list."""
        assert find_lin_files(tmp_path / "model.fst") == []

    def test_stem_with_glob_metacharacters(self, tmp_path: Path) -> None:
        """A deck stem containing glob metacharacters matches literally."""
        for name in ("case[1].1.lin", "case[1].2.lin", "case1.1.lin"):
            (tmp_path / name).write_text("", encoding="utf-8")
        files = find_lin_files(tmp_path / "case[1].fst")
        # The bracketed stem is matched literally, not as a character class, so the
        # decoy ``case1.1.lin`` is excluded.
        assert [f.name for f in files] == ["case[1].1.lin", "case[1].2.lin"]


class TestFindOpenfast:
    """Tests for locating the executable."""

    def test_explicit_existing_path(self, tmp_path: Path) -> None:
        """An explicit path to an existing file resolves to that file."""
        exe = tmp_path / "openfast.exe"
        exe.write_text("", encoding="utf-8")
        assert find_openfast(exe) == exe.resolve()

    def test_explicit_path_is_resolved(self, tmp_path: Path) -> None:
        """An explicit path is resolved (so a later cwd change cannot break it).

        ``run_openfast`` invokes the executable with ``cwd`` set to the deck
        directory, so the returned path must be absolute and normalized.
        """
        exe = tmp_path / "openfast.exe"
        exe.write_text("", encoding="utf-8")
        (tmp_path / "sub").mkdir()
        # An absolute but un-normalized path (contains ``..``); resolve() collapses it.
        messy = tmp_path / "sub" / ".." / "openfast.exe"
        resolved = find_openfast(messy)
        assert resolved.is_absolute()
        assert ".." not in resolved.parts
        assert resolved == exe.resolve()

    def test_missing_executable_raises(self) -> None:
        """A bogus executable name not on PATH raises OpenFastError."""
        with pytest.raises(OpenFastError, match="not found on PATH"):
            find_openfast("definitely-not-a-real-openfast-binary-xyz")

    def test_path_hit_is_resolved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A relative ``which`` result is resolved to absolute (cwd-change safe)."""
        import shutil

        monkeypatch.setattr(shutil, "which", lambda _name: "bin/openfast")
        resolved = find_openfast("openfast")
        assert resolved.is_absolute()


class TestRunOpenfast:
    """Tests for the run wrapper (mocked executable) and validation."""

    def test_missing_deck_raises(self, tmp_path: Path) -> None:
        """Running a non-existent deck raises before invoking anything."""
        with pytest.raises(OpenFastError, match="deck not found"):
            run_openfast(tmp_path / "nope.fst")

    @pytest.mark.parametrize("timeout", [0.0, -1.0, np.nan, np.inf, -np.inf])
    def test_invalid_timeout_fails_before_stale_output_cleanup(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        timeout: float,
    ) -> None:
        """An invalid subprocess limit cannot mutate a deck's retained outputs."""
        deck = tmp_path / "model.fst"
        deck.write_text("x\n", encoding="utf-8")
        stale = tmp_path / "model.1.lin"
        stale.write_text("retained\n", encoding="utf-8")

        def fail_run(*_args: object, **_kwargs: object) -> None:
            pytest.fail("subprocess must not start for an invalid timeout")

        monkeypatch.setattr(subprocess, "Popen", fail_run)
        with pytest.raises(ValueError, match="timeout must be positive and finite"):
            run_openfast(deck, timeout=timeout)
        assert stale.read_text(encoding="utf-8") == "retained\n"

    def test_boolean_timeout_is_rejected(self, tmp_path: Path) -> None:
        """A Boolean cannot be interpreted silently as a 1-second process limit."""
        deck = tmp_path / "model.fst"
        deck.write_text("x\n", encoding="utf-8")
        boolean_timeout: Any = True
        with pytest.raises(TypeError, match="not a boolean"):
            run_openfast(deck, timeout=boolean_timeout)

    def test_invokes_and_collects_lin(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The wrapper invokes the exe in the deck dir and collects .lin output."""
        deck = tmp_path / "model.fst"
        deck.write_text("True Linearize - flag\n", encoding="utf-8")
        exe = tmp_path / "fake_openfast"
        exe.write_text("", encoding="utf-8")
        captured: dict[str, object] = {}

        def fake_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
            captured["command"] = command
            captured["cwd"] = kwargs.get("cwd")
            # Simulate OpenFAST writing one full-system .lin file.
            (tmp_path / "model.1.lin").write_text("", encoding="utf-8")
            return CompletedProcess(
                command, 0, stdout="OpenFAST terminated normally.\n", stderr=""
            )

        _patch_process(monkeypatch, fake_run)
        result = run_openfast(deck, executable=exe)

        assert result.succeeded
        assert result.n_lin == 1
        assert result.lin_files[0].name == "model.1.lin"
        assert captured["command"] == [str(exe), "model.fst"]
        assert captured["cwd"] == tmp_path

    def test_clears_stale_lin_before_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pre-existing .lin files are removed so the result reflects this run."""
        deck = tmp_path / "model.fst"
        deck.write_text("x\n", encoding="utf-8")
        (tmp_path / "model.1.lin").write_text("stale", encoding="utf-8")
        (tmp_path / "model.2.lin").write_text("stale", encoding="utf-8")
        for suffix in (".out", ".outb", ".sum"):
            deck.with_suffix(suffix).write_text("stale", encoding="utf-8")
        exe = tmp_path / "fake_openfast"
        exe.write_text("", encoding="utf-8")

        def fake_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
            # Write only one fresh file; the two stale ones must have been cleared.
            (tmp_path / "model.1.lin").write_text("fresh", encoding="utf-8")
            return CompletedProcess(command, 0, stdout="", stderr="")

        _patch_process(monkeypatch, fake_run)
        result = run_openfast(deck, executable=exe)
        assert result.n_lin == 1
        assert not (tmp_path / "model.2.lin").exists()
        assert result.output_file is None
        assert result.summary_file is None
        assert not deck.with_suffix(".out").exists()
        assert not deck.with_suffix(".outb").exists()
        assert not deck.with_suffix(".sum").exists()

    def test_clears_stale_module_lin_sidecars_before_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Per-module sidecars of this deck are cleared; other decks' files are kept."""
        deck = tmp_path / "model.fst"
        deck.write_text("x\n", encoding="utf-8")
        stale = ("model.1.ED.lin", "model.2.BD1.lin", "model.1.SrvD.lin")
        for name in stale:
            (tmp_path / name).write_text("stale", encoding="utf-8")
        other = tmp_path / "model_b.1.ED.lin"
        other.write_text("other deck", encoding="utf-8")
        exe = tmp_path / "fake_openfast"
        exe.write_text("", encoding="utf-8")

        def fake_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
            (tmp_path / "model.1.lin").write_text("fresh", encoding="utf-8")
            (tmp_path / "model.1.ED.lin").write_text("fresh", encoding="utf-8")
            return CompletedProcess(command, 0, stdout="", stderr="")

        _patch_process(monkeypatch, fake_run)
        run_openfast(deck, executable=exe)
        assert (tmp_path / "model.1.ED.lin").read_text(encoding="utf-8") == "fresh"
        assert not (tmp_path / "model.2.BD1.lin").exists()
        assert not (tmp_path / "model.1.SrvD.lin").exists()
        assert other.read_text(encoding="utf-8") == "other deck"

    def test_nonzero_return_code_is_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failing run returns succeeded=False rather than raising."""
        deck = tmp_path / "model.fst"
        deck.write_text("x\n", encoding="utf-8")
        exe = tmp_path / "fake_openfast"
        exe.write_text("", encoding="utf-8")

        def fake_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
            return CompletedProcess(command, 1, stdout="FATAL", stderr="")

        _patch_process(monkeypatch, fake_run)
        result = run_openfast(deck, executable=exe)
        assert not result.succeeded
        assert result.returncode == 1

    def test_captures_stderr_in_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Diagnostics on stderr are retained and exposed via ``output``."""
        deck = tmp_path / "model.fst"
        deck.write_text("x\n", encoding="utf-8")
        exe = tmp_path / "fake_openfast"
        exe.write_text("", encoding="utf-8")

        def fake_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
            return CompletedProcess(command, 1, stdout="out", stderr="FATAL error")

        _patch_process(monkeypatch, fake_run)
        result = run_openfast(deck, executable=exe)
        assert result.stderr == "FATAL error"
        assert "FATAL error" in result.output
        assert result.output == "outFATAL error"


class TestRunOpenfastIntegration:
    """Real OpenFAST run (skipped when the executable or r-test data is absent)."""

    def test_runs_standstill_linearization(
        self, openfast_exe: Path, rtest_dir: Path
    ) -> None:
        """A real standstill linearization produces a parseable .lin file."""
        from vane.io import read_lin_file

        deck = (
            rtest_dir
            / "glue-codes"
            / "openfast"
            / "WP_Stationary_Linear"
            / "WP_Stationary_Linear.fst"
        )
        if not deck.is_file():
            pytest.skip(f"r-test case not found: {deck}")

        result = run_openfast(deck, executable=openfast_exe, timeout=300)
        try:
            assert result.succeeded
            assert result.n_lin >= 1
            lin = read_lin_file(result.lin_files[0])
            assert lin.a is not None
            assert lin.a.shape[0] == lin.n_x
        finally:
            outputs = (*result.lin_files, result.output_file, result.summary_file)
            for produced in outputs:
                if produced is not None and produced.exists():
                    produced.unlink()


class TestRunOpenfastTimeoutKillsTree:
    """A timed-out wrapper executable must not leave an orphaned solver behind."""

    def test_timeout_kills_wrapper_grandchild(self, tmp_path: Path) -> None:
        """The whole tree dies and the call returns promptly after the timeout.

        The fake executable is a wrapper script (``.bat`` on Windows, ``sh`` on
        POSIX) that launches a long-sleeping child holding the output pipes, exactly
        as a site wrapper around OpenFAST does. Killing only the wrapper would leave
        that child alive and block on its pipes until it exits on its own.
        """
        deck = tmp_path / "model.fst"
        deck.write_text("x\n", encoding="utf-8")
        pid_file = tmp_path / "child.pid"
        child = tmp_path / "child.py"
        child.write_text(
            "import os, pathlib, sys, time\n"
            f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
            "print('solver started', flush=True)\n"
            "time.sleep(40)\n",
            encoding="utf-8",
        )
        if sys.platform == "win32":
            wrapper = tmp_path / "openfast.bat"
            wrapper.write_text(
                f'@echo off\r\n"{sys.executable}" "{child}"\r\n', encoding="utf-8"
            )
        else:
            wrapper = tmp_path / "openfast.sh"
            wrapper.write_text(
                f"#!/bin/sh\n'{sys.executable}' '{child}'\n", encoding="utf-8"
            )
            wrapper.chmod(0o755)
        timeout = 4.0

        start = time.monotonic()
        with pytest.raises(OpenFastTimeoutError, match="timed out"):
            run_openfast(deck, executable=wrapper, timeout=timeout)
        elapsed = time.monotonic() - start

        assert elapsed < timeout + 15.0
        assert pid_file.is_file(), "the wrapper never started its child"
        pid = int(pid_file.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 10.0
        while _pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not _pid_alive(pid)


class TestRunOpenfastOutputDecoding:
    """Console output that the locale encoding cannot decode is still captured."""

    def test_undecodable_bytes_do_not_abort_the_run(self, tmp_path: Path) -> None:
        """Bytes invalid in UTF-8, cp1252 and cp949 are replaced, not raised.

        OpenFAST echoes file paths and deck text in the console's code page, so its
        output can hold bytes the Python locale encoding rejects. The "deck" here is
        a Python script run by the Python executable, writing such bytes to both
        streams and exiting non-zero.
        """
        deck = tmp_path / "model.fst"
        deck.write_text(
            "import sys\n"
            "sys.stdout.buffer.write(b'path \\x81\\xff ok\\n')\n"
            "sys.stdout.flush()\n"
            "sys.stderr.buffer.write(b'Aborting \\x81\\xff\\n')\n"
            "sys.stderr.flush()\n"
            "sys.exit(3)\n",
            encoding="utf-8",
        )
        result = run_openfast(
            deck, executable=sys.executable, check_path_lengths=False, timeout=60.0
        )
        assert result.returncode == 3
        assert "path" in result.stdout
        assert "ok" in result.stdout
        assert "Aborting" in result.stderr


# OpenFAST console texts, abridged from real runs of the reference turbines.
_FATAL = " Simulation error level: FATAL ERROR"


def _console(*lines: str) -> str:
    """Join synthetic OpenFAST console lines."""
    return "\n".join(lines)


_MOTORING_OUTPUT = _console(
    " Time: 0 of 600 seconds.",
    "",
    "FAST_Solution:FAST_AdvanceStates:SrvD_UpdateStates:u%HSS_Spd is negative. "
    "Simple variable-speed control model is not valid for motoring situations.",
    "",
    " OpenFAST encountered an error at simulation time 6.0000E-02 of 600 seconds.",
    _FATAL,
    "",
    " Aborting OpenFAST.",
)

_OVERSPEED_OUTPUT = _console(
    "FAST_Solution:CalcSteady:The rotor is spinning too fast. The time step or "
    "NLinTimes is too large when CalcSteady=true.",
    "",
    " OpenFAST encountered an error at simulation time 48.25 of 600 seconds.",
    _FATAL,
)

_DIVERGENCE_OUTPUT = _console(
    "SmllRotTrans:Small angle assumption violated in SUBROUTINE SmllRotTrans() "
    "due to a large blade deflection.",
    "FAST_Solution:ED_ABM4:NaNs in the ElastoDyn state derivatives.",
    "",
    " OpenFAST encountered an error at simulation time 1.2345D+02 of 600 seconds.",
    _FATAL,
)

_INIT_OUTPUT = _console(
    "FAST_InitializeAll:SrvD_Init:VS_RtGnSp must be greater than zero.",
    "FAST_InitializeAll:SrvD_Init:HSS_Spd is negative (a stale message).",
    "",
    " OpenFAST encountered an error during module initialization.",
    _FATAL,
)


class TestClassifyOpenFastOutput:
    """Tests for classify_openfast_output (synthetic OpenFAST console text)."""

    def test_clean_success_is_not_a_failure(self) -> None:
        """A normal exit without the forced-linearization warning is no failure."""
        assert classify_openfast_output("OpenFAST terminated normally.", 0) is None

    def test_forced_linearization_is_trim_not_converged(self) -> None:
        """The forced-linearization warning marks a trim that ran out of time."""
        output = (
            "[WARNING] Steady state not found before end of simulation. "
            "Forcing linearization.\nOpenFAST terminated normally."
        )
        diagnosis = classify_openfast_output(output, 0)
        assert diagnosis is not None
        assert diagnosis.mode is OpenFastFailureMode.TRIM_NOT_CONVERGED
        assert diagnosis.retriable
        assert "Steady state not found" in diagnosis.openfast_message

    def test_motoring_with_failure_time(self) -> None:
        """A negative generator speed is motoring, with OpenFAST's failure time."""
        diagnosis = classify_openfast_output(_MOTORING_OUTPUT, 1)
        assert diagnosis is not None
        assert diagnosis.mode is OpenFastFailureMode.MOTORING
        assert diagnosis.failure_time == pytest.approx(0.06)
        assert "HSS_Spd is negative" in diagnosis.openfast_message
        assert diagnosis.summary.startswith("motoring: ")

    def test_rotor_overspeed(self) -> None:
        """The CalcSteady azimuth-step error is a rotor overspeed."""
        diagnosis = classify_openfast_output(_OVERSPEED_OUTPUT, 1)
        assert diagnosis is not None
        assert diagnosis.mode is OpenFastFailureMode.ROTOR_OVERSPEED
        assert diagnosis.failure_time == pytest.approx(48.25)

    def test_divergence_reads_fortran_exponent(self) -> None:
        """Small-angle/NaN signatures are a divergence; a D exponent is parsed."""
        diagnosis = classify_openfast_output(_DIVERGENCE_OUTPUT, 1)
        assert diagnosis is not None
        assert diagnosis.mode is OpenFastFailureMode.DIVERGENCE
        assert diagnosis.failure_time == pytest.approx(123.45)
        assert "Small angle" in diagnosis.openfast_message

    def test_module_initialization_error_is_not_retriable(self) -> None:
        """An initialisation failure wins over any other signature and is final."""
        diagnosis = classify_openfast_output(_INIT_OUTPUT, 1)
        assert diagnosis is not None
        assert diagnosis.mode is OpenFastFailureMode.INITIALIZATION_ERROR
        assert not diagnosis.retriable
        assert diagnosis.failure_time == 0.0
        assert "VS_RtGnSp must be greater than zero" in diagnosis.openfast_message

    def test_simulation_initialization_failure_time_is_zero(self) -> None:
        """A failure at t = 0 without a known signature is unknown at time zero."""
        output = (
            "FAST_Solution0:Some module error.\n"
            " OpenFAST encountered an error during simulation initialization.\n"
        )
        diagnosis = classify_openfast_output(output, 1)
        assert diagnosis is not None
        assert diagnosis.mode is OpenFastFailureMode.UNKNOWN
        assert diagnosis.failure_time == 0.0
        assert diagnosis.retriable

    def test_unrecognised_exit_keeps_the_output_tail(self) -> None:
        """An unrecognised non-zero exit is unknown and keeps OpenFAST's last lines."""
        output = "\n".join(f"line {index}" for index in range(20))
        diagnosis = classify_openfast_output(output, 3)
        assert diagnosis is not None
        assert diagnosis.mode is OpenFastFailureMode.UNKNOWN
        assert diagnosis.failure_time is None
        assert "code 3" in diagnosis.detail
        assert diagnosis.openfast_message.splitlines()[-1] == "line 19"
        assert "line 0" not in diagnosis.openfast_message

    def test_forced_then_crashed_is_trim_not_converged(self) -> None:
        """A forced linearization followed by an unexplained crash is a slow trim."""
        output = "Steady state not found before end of simulation.\nbye\n"
        diagnosis = classify_openfast_output(output, 2)
        assert diagnosis is not None
        assert diagnosis.mode is OpenFastFailureMode.TRIM_NOT_CONVERGED

    def test_result_diagnosis_property(self, tmp_path: Path) -> None:
        """OpenFastResult.diagnosis classifies the combined stdout and stderr."""
        result = OpenFastResult(
            fst_path=tmp_path / "Model.fst", returncode=1, stderr=_MOTORING_OUTPUT
        )
        assert result.diagnosis is not None
        assert result.diagnosis.mode is OpenFastFailureMode.MOTORING
        ok = OpenFastResult(fst_path=tmp_path / "Model.fst", returncode=0)
        assert ok.diagnosis is None


class TestRunOpenFastPaths:
    """Path handling before the OpenFAST process is launched."""

    @staticmethod
    def _deck_and_exe(tmp_path: Path) -> tuple[Path, Path]:
        deck = tmp_path / "model.fst"
        deck.write_text("True Linearize - flag\n", encoding="utf-8")
        exe = tmp_path / "fake_openfast"
        exe.write_text("", encoding="utf-8")
        return deck, exe

    @staticmethod
    def _recording_process(
        monkeypatch: pytest.MonkeyPatch, captured: dict[str, object]
    ) -> None:
        def fake_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
            captured["command"] = command
            captured["cwd"] = kwargs.get("cwd")
            return CompletedProcess(command, 0, stdout="", stderr="")

        _patch_process(monkeypatch, fake_run)

    def test_path_length_check_runs_before_launch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An over-long sub-file path stops the run before the process starts."""
        deck, exe = self._deck_and_exe(tmp_path)
        captured: dict[str, object] = {}
        self._recording_process(monkeypatch, captured)

        def too_long(path: Path) -> None:
            msg = f"path too long for {path}"
            raise DeckPathLengthError(msg)

        monkeypatch.setattr(runner_module, "check_deck_path_lengths", too_long)
        with pytest.raises(DeckPathLengthError, match="path too long"):
            run_openfast(deck, executable=exe, check_path_lengths=True)
        assert captured == {}

    def test_path_length_check_can_be_disabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``check_path_lengths=False`` launches without walking the deck."""
        deck, exe = self._deck_and_exe(tmp_path)
        captured: dict[str, object] = {}
        self._recording_process(monkeypatch, captured)
        calls: list[Path] = []
        monkeypatch.setattr(runner_module, "check_deck_path_lengths", calls.append)
        run_openfast(deck, executable=exe, check_path_lengths=False)
        assert calls == []
        assert "command" in captured

    @pytest.mark.parametrize(("platform", "expected"), [("win32", 1), ("linux", 0)])
    def test_path_length_check_defaults_to_windows_only(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        platform: str,
        expected: int,
    ) -> None:
        """By default the check runs only where the Windows path limit applies."""
        deck, exe = self._deck_and_exe(tmp_path)
        self._recording_process(monkeypatch, {})
        calls: list[Path] = []
        monkeypatch.setattr(runner_module, "check_deck_path_lengths", calls.append)
        monkeypatch.setattr(runner_module.sys, "platform", platform)
        run_openfast(deck, executable=exe)
        assert len(calls) == expected

    def test_real_deck_check_passes_for_a_short_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The real check accepts a deck in a short directory and the run proceeds."""
        deck, exe = self._deck_and_exe(tmp_path)
        captured: dict[str, object] = {}
        self._recording_process(monkeypatch, captured)
        run_openfast(deck, executable=exe, check_path_lengths=True)
        assert captured["command"] == [str(exe), "model.fst"]

    def test_long_extended_executable_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An executable nameable only with the extended prefix is rejected."""
        deck, _ = self._deck_and_exe(tmp_path)
        captured: dict[str, object] = {}
        self._recording_process(monkeypatch, captured)
        deep = "\\".join(["d" * 60] * 5)
        long_exe = PureWindowsPath(rf"\\?\C:\{deep}\openfast.exe")
        monkeypatch.setattr(runner_module, "find_openfast", lambda _exe: long_exe)
        with pytest.raises(OpenFastError, match="executable"):
            run_openfast(deck, check_path_lengths=False)
        assert captured == {}

    def test_short_extended_executable_is_passed_plain(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A short executable path loses its extended prefix before launch."""
        deck, _ = self._deck_and_exe(tmp_path)
        captured: dict[str, object] = {}
        self._recording_process(monkeypatch, captured)
        short_exe = PureWindowsPath(r"\\?\C:\OpenFAST\openfast.exe")
        monkeypatch.setattr(runner_module, "find_openfast", lambda _exe: short_exe)
        run_openfast(deck, check_path_lengths=False)
        command = captured["command"]
        assert isinstance(command, list)
        assert command[0] == r"C:\OpenFAST\openfast.exe"

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows path prefix")
    def test_extended_deck_directory_is_passed_plain(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A deck given with the extended prefix runs from the plain directory."""
        deck, exe = self._deck_and_exe(tmp_path)
        captured: dict[str, object] = {}
        self._recording_process(monkeypatch, captured)
        prefixed = Path("\\\\?\\" + str(deck))
        run_openfast(prefixed, executable=exe, check_path_lengths=False)
        assert str(captured["cwd"]) == str(tmp_path)
        assert not str(captured["cwd"]).startswith("\\\\?\\")

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows path prefix")
    def test_overlong_deck_directory_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A deck directory beyond the Windows limit fails before launch."""
        _, exe = self._deck_and_exe(tmp_path)
        captured: dict[str, object] = {}
        self._recording_process(monkeypatch, captured)
        top = Path("\\\\?\\" + str(tmp_path / "deep"))
        deep = top.joinpath(*(["d" * 60] * 5))
        try:
            deep.mkdir(parents=True)
            deck = deep / "model.fst"
            deck.write_text("True Linearize - flag\n", encoding="utf-8")
            with pytest.raises(OpenFastError, match="shorter output_dir"):
                run_openfast(deck, executable=exe, check_path_lengths=False)
        finally:
            shutil.rmtree(top, ignore_errors=True)
        assert captured == {}


def _wrap_inside(output: str, signature: str) -> str:
    """Break ``signature`` in ``output`` at its middle space, as OpenFAST wraps."""
    assert signature in output
    middle = len(signature) // 2
    spaces = [i for i, char in enumerate(signature) if char == " "]
    cut = min(spaces, key=lambda i: abs(i - middle))
    wrapped = signature[:cut] + "\n" + signature[cut + 1 :]
    return output.replace(signature, wrapped)


_SIM_INIT_OUTPUT = _console(
    "FAST_InitializeAll:something odd happened.",
    " OpenFAST encountered an error during simulation initialization.",
    _FATAL,
)


class TestWrappedConsoleOutput:
    """OpenFAST wraps console text at ~100 columns; signatures must still match."""

    @pytest.mark.parametrize(
        ("output", "returncode", "signature", "mode"),
        [
            (
                "[WARNING] Steady state not found before end of simulation. "
                "Forcing linearization.\nOpenFAST terminated normally.",
                0,
                "Steady state not found",
                OpenFastFailureMode.TRIM_NOT_CONVERGED,
            ),
            (
                "[WARNING] Steady state not found before end of simulation.\n" + _FATAL,
                1,
                "Steady state not found",
                OpenFastFailureMode.TRIM_NOT_CONVERGED,
            ),
            (_MOTORING_OUTPUT, 1, "HSS_Spd is negative", OpenFastFailureMode.MOTORING),
            (
                _OVERSPEED_OUTPUT,
                1,
                "rotor is spinning too fast",
                OpenFastFailureMode.ROTOR_OVERSPEED,
            ),
            (
                _INIT_OUTPUT,
                1,
                "error during module initialization",
                OpenFastFailureMode.INITIALIZATION_ERROR,
            ),
            (
                _console(
                    "SmllRotTrans:Small angle assumption violated in SmllRotTrans().",
                    _FATAL,
                ),
                1,
                "Small angle assumption violated",
                OpenFastFailureMode.DIVERGENCE,
            ),
            (
                _console("BD_Solve: the solution does not converge.", _FATAL),
                1,
                "does not converge",
                OpenFastFailureMode.DIVERGENCE,
            ),
            (
                _console("Solver failed to converge after 100 iterations.", _FATAL),
                1,
                "failed to converge",
                OpenFastFailureMode.DIVERGENCE,
            ),
            (
                _console("ElastoDyn: large deflection in blade 1.", _FATAL),
                1,
                "large deflection",
                OpenFastFailureMode.DIVERGENCE,
            ),
        ],
    )
    def test_wrap_inside_each_signature(
        self,
        output: str,
        returncode: int,
        signature: str,
        mode: OpenFastFailureMode,
    ) -> None:
        """A line break inside a signature does not change the classification."""
        diagnosis = classify_openfast_output(
            _wrap_inside(output, signature), returncode
        )
        assert diagnosis is not None
        assert diagnosis.mode is mode
        if mode is not OpenFastFailureMode.INITIALIZATION_ERROR:
            # The quoted message holds the whole signature (a tail is multi-line).
            message = " ".join(diagnosis.openfast_message.split()).lower()
            assert signature.lower() in message

    def test_wrapped_simulation_initialization_sets_time_zero(self) -> None:
        """A wrapped simulation-initialization error still means t = 0."""
        output = _wrap_inside(
            _SIM_INIT_OUTPUT, "error during simulation initialization"
        )
        diagnosis = classify_openfast_output(output, 1)
        assert diagnosis is not None
        assert diagnosis.failure_time == 0.0

    def test_wrapped_failure_time(self) -> None:
        """The failure time is read across a wrapped line."""
        output = _wrap_inside(
            _MOTORING_OUTPUT,
            "encountered an error at simulation time 6.0000E-02 of 600 seconds",
        )
        diagnosis = classify_openfast_output(output, 1)
        assert diagnosis is not None
        assert diagnosis.failure_time == pytest.approx(0.06)

    def test_motoring_wins_over_an_earlier_warning(self) -> None:
        """The wrapped motoring message beats an earlier divergence-like warning."""
        output = _console(
            "SmllRotTrans:Small angle assumption violated (a transient warning).",
            "FAST_Solution:FAST_UpdateStates:FAST_AdvanceStates:SrvD_UpdateStates:"
            "CalculateTorque:u%HSS_Spd is",
            "negative. Simple variable-speed control model is not valid for motoring "
            "situations.",
            "",
            " OpenFAST encountered an error at simulation time 41.135 of 600 seconds.",
            _FATAL,
        )
        diagnosis = classify_openfast_output(output, 1)
        assert diagnosis is not None
        assert diagnosis.mode is OpenFastFailureMode.MOTORING
        assert diagnosis.openfast_message.endswith(
            "u%HSS_Spd is negative. Simple variable-speed control model is not "
            "valid for motoring situations."
        )
        assert diagnosis.failure_time == pytest.approx(41.135)

    def test_real_wrapped_iea15_log_is_motoring(self) -> None:
        """The IEA 15-MW ws12 log wraps the motoring message (skipped if absent)."""
        runs = os.environ.get("VANE_REFERENCE_RUNS_DIR")
        if not runs:
            pytest.skip(
                "reference-turbine data unavailable; set VANE_REFERENCE_RUNS_DIR"
            )
        root = Path(runs)
        log = (
            root
            / "iea15_monopile"
            / "cases"
            / "ws12"
            / "IEA-15-240-RWT-Monopile"
            / "IEA-15-240-RWT-Monopile.attempt1.log"
        )
        if not log.is_file():
            pytest.skip(f"reference-turbine log not found: {log}")
        text = log.read_text(encoding="utf-8", errors="replace")
        assert "u%HSS_Spd is\n" in text.replace("\r\n", "\n")  # really wrapped
        diagnosis = classify_openfast_output(text, 1)
        assert diagnosis is not None
        assert diagnosis.mode is OpenFastFailureMode.MOTORING
        assert "HSS_Spd is negative" in diagnosis.openfast_message
        assert diagnosis.failure_time == pytest.approx(41.135)
