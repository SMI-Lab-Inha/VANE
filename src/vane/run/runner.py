# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Run the OpenFAST executable on an input deck and collect its ``.lin`` files.

OpenFAST is invoked as ``openfast <primary>.fst`` from the deck's directory (so the
relative sub-file references resolve), and on success writes one full-system
linearization file per linearization time, named ``<RootName>.<i>.lin`` for
``i = 1 .. NLinTimes``. This module wraps that invocation: it locates the
executable, runs it (capturing output and the return code), and discovers the
produced ``.lin`` files — which feed straight into :func:`vane.io.read_lin_file`
and the modal pipeline, closing the loop from model deck to identified modes.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import re
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from vane.run.preflight import check_deck_path_lengths
from vane.utils.fs_retry import unlink_with_retry
from vane.utils.paths import has_extended_prefix, normalize_extended_path

__all__ = [
    "OpenFastDiagnosis",
    "OpenFastError",
    "OpenFastFailureMode",
    "OpenFastResult",
    "OpenFastTimeoutError",
    "classify_openfast_output",
    "find_lin_files",
    "find_openfast",
    "run_openfast",
]

logger = logging.getLogger(__name__)

# Full-system linearization files are ``<root>.<index>.lin``; per-module files
# (LinOutMod) carry a module token, e.g. ``<root>.1.ED.lin``, and are excluded.
_LIN_INDEX_RE = re.compile(r"\.(\d+)\.lin$")
# Per-module sidecars: ``<root>.<index>.<Module>.lin`` (e.g. ``.1.ED.lin``).
_MODULE_LIN_SUFFIX_RE = re.compile(r"\.\d+\.[A-Za-z][A-Za-z0-9]*\.lin")
# After a timeout the process tree is killed; allow this long for the (now closed)
# output pipes to drain before abandoning them.
_KILL_DRAIN_SECONDS = 10.0


def _validated_timeout(timeout: float | None) -> float | None:
    """Return a finite positive timeout, or ``None`` for no time limit."""
    if timeout is None:
        return None
    if isinstance(timeout, bool):
        msg = "timeout must be a real number, not a boolean"
        raise TypeError(msg)
    try:
        value = float(timeout)
    except (TypeError, ValueError) as exc:
        msg = "timeout must be positive and finite when provided"
        raise ValueError(msg) from exc
    if not math.isfinite(value) or value <= 0.0:
        msg = "timeout must be positive and finite when provided"
        raise ValueError(msg)
    return value


class OpenFastError(RuntimeError):
    """Raised when the OpenFAST executable cannot be found or fails to run."""


class OpenFastTimeoutError(OpenFastError):
    """Raised when an OpenFAST run exceeds its timeout.

    A distinct subclass so callers can treat a timeout (a slow but retriable hard
    operating point) differently from a configuration failure (a missing deck or
    executable), which retrying would not fix.
    """


@dataclass
class OpenFastResult:
    """Outcome of an OpenFAST run.

    Parameters
    ----------
    fst_path : pathlib.Path
        The primary ``.fst`` deck that was run.
    returncode : int
        The executable's exit status (0 on success).
    lin_files : list[pathlib.Path]
        Full-system ``.lin`` files produced, ordered by linearization index.
    output_file : pathlib.Path or None
        The time-series output file (``.out``/``.outb``) if one was written.
    summary_file : pathlib.Path or None
        The ``.sum`` summary file if one was written.
    stdout : str
        Captured standard output of the run.
    stderr : str
        Captured standard error of the run (OpenFAST may write diagnostics here).
    """

    fst_path: Path
    returncode: int
    lin_files: list[Path] = field(default_factory=list)
    output_file: Path | None = None
    summary_file: Path | None = None
    stdout: str = ""
    stderr: str = ""

    @property
    def succeeded(self) -> bool:
        """Return whether OpenFAST exited successfully."""
        return self.returncode == 0

    @property
    def output(self) -> str:
        """Return the combined standard output and error of the run."""
        return self.stdout + self.stderr

    @property
    def n_lin(self) -> int:
        """Return the number of full-system ``.lin`` files produced."""
        return len(self.lin_files)

    @property
    def diagnosis(self) -> OpenFastDiagnosis | None:
        """Return the classified failure of this run, or ``None`` when none is seen.

        A shorthand for :func:`classify_openfast_output` on the captured output and
        return code. It does not check the ``.lin`` count against the requested
        ``NLinTimes``; the linearization driver does that.
        """
        return classify_openfast_output(self.output, self.returncode)


class OpenFastFailureMode(str, Enum):
    """Why an OpenFAST linearization run failed, as read from its output.

    The value of each member is the stable string recorded in provenance.

    Attributes
    ----------
    TRIM_NOT_CONVERGED
        The CalcSteady trim did not reach a periodic steady state before ``TMax``:
        OpenFAST logs ``Steady state not found before end of simulation`` and forces a
        single linearization, or fewer ``.lin`` files than ``NLinTimes`` are written.
    MOTORING
        The generator speed went negative (``u%HSS_Spd is negative``), which the simple
        variable-speed controller rejects. Usually an overshooting trim, or trim damping
        destabilising light blades when it happens near ``t = 0``.
    ROTOR_OVERSPEED
        CalcSteady saw the rotor advance more than one azimuth step in a time step
        (``The rotor is spinning too fast``); in practice a divergence.
    DIVERGENCE
        The run ended on a fatal error after signs of a diverging response (small-angle
        violations, NaNs, a structural solver that stopped converging).
    INITIALIZATION_ERROR
        OpenFAST failed while initialising its modules: an input problem that no trim
        setting fixes.
    TIMEOUT
        The run exceeded the caller's wall-clock timeout and was killed.
    INCOMPLETE_LINEARIZATION
        An untrimmed explicit-time sweep exited normally but wrote fewer ``.lin`` files
        than requested.
    NON_PERIODIC_SWEEP
        A CalcSteady run wrote every requested ``.lin`` file, but the rotor speed in
        their headers varies across the azimuth sweep by more than MBC3 accepts
        (``vane.io.mbc_transform.ROTOR_SPEED_DRIFT_RTOL``): the trim stopped on a
        response that is not periodic, typically because ``TrimTol`` was loosened. The
        linearization driver detects it from the files; :func:`classify_openfast_output`
        never returns it.
    NON_FINITE_LINEARIZATION
        The run exited normally and wrote every requested ``.lin`` file, but a file
        holds a NaN or infinite value (an operating-point value or a matrix entry).
        This is a modelling degeneracy, such as AeroDyn unsteady aerodynamics left on
        at the blade-tip node, that other trim settings cannot cure. The linearization
        driver detects it by scanning the files; :func:`classify_openfast_output`
        never returns it.
    CASE_ERROR
        Not an OpenFAST failure: the linearization driver raised an unexpected
        exception while it generated, ran or collected the case (a file-system error
        while the case was copied or configured, say). A sweep records it per case
        rather than aborting (see :meth:`vane.run.LinearizationDriver.run_sweep`); it
        is never retried, and :func:`classify_openfast_output` never returns it.
    UNKNOWN
        A non-zero exit with no recognised signature.
    """

    TRIM_NOT_CONVERGED = "trim_not_converged"
    MOTORING = "motoring"
    ROTOR_OVERSPEED = "rotor_overspeed"
    DIVERGENCE = "divergence"
    INITIALIZATION_ERROR = "initialization_error"
    TIMEOUT = "timeout"
    INCOMPLETE_LINEARIZATION = "incomplete_linearization"
    NON_PERIODIC_SWEEP = "non_periodic_sweep"
    NON_FINITE_LINEARIZATION = "non_finite_linearization"
    CASE_ERROR = "case_error"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class OpenFastDiagnosis:
    """A classified OpenFAST failure.

    Parameters
    ----------
    mode : OpenFastFailureMode
        The failure class.
    detail : str
        A one-line description of the failure.
    openfast_message : str
        The relevant excerpt of OpenFAST's own output (its fatal error message, or the
        last lines it printed), empty when none applies.
    failure_time : float or None
        The simulation time (s) at which OpenFAST stopped with a fatal error, parsed
        from ``encountered an error at simulation time <t> of <TMax> seconds``. ``0.0``
        for a failure during initialisation; ``None`` when OpenFAST did not report it.
    """

    mode: OpenFastFailureMode
    detail: str
    openfast_message: str = ""
    failure_time: float | None = None

    @property
    def retriable(self) -> bool:
        """Return whether other trim settings could plausibly fix the failure.

        An initialisation error is an input problem and a non-finite linearization a
        modelling degeneracy, so neither is retriable.
        """
        return self.mode not in _NON_RETRIABLE_MODES

    @property
    def summary(self) -> str:
        """Return the failure mode and detail as one line."""
        return f"{self.mode.value}: {self.detail}"


# Failures no trim setting can fix: an input problem, or a modelling degeneracy that
# makes the linearization non-finite (or an unexpected exception in the driver).
_NON_RETRIABLE_MODES = frozenset(
    {
        OpenFastFailureMode.INITIALIZATION_ERROR,
        OpenFastFailureMode.NON_FINITE_LINEARIZATION,
        OpenFastFailureMode.CASE_ERROR,
    }
)

# Output signatures, matched case-insensitively. The texts are OpenFAST's own messages
# (ServoDyn, the CalcSteady glue code and the OpenFAST driver program).
_FORCED_LINEARIZATION_SIGNATURE = "steady state not found"
_MOTORING_SIGNATURE = "hss_spd is negative"
_OVERSPEED_SIGNATURE = "rotor is spinning too fast"
_MODULE_INIT_SIGNATURE = "error during module initialization"
_SIMULATION_INIT_SIGNATURE = "error during simulation initialization"
_DIVERGENCE_RE = re.compile(
    r"small angle assumption violated|\bnans?\b|\binfinity\b|large deflection"
    r"|(?:does|did) not converge|failed to converge",
    re.IGNORECASE,
)
_FAILURE_TIME_RE = re.compile(
    r"encountered an error at simulation time\s+([-+0-9.EeDd]+)\s+of\s+"
    r"([-+0-9.EeDd]+)\s+seconds",
    re.IGNORECASE,
)
# How much of OpenFAST's output a diagnosis keeps when no specific line is matched.
_TAIL_LINES = 8
_MESSAGE_LIMIT = 1000


def classify_openfast_output(output: str, returncode: int) -> OpenFastDiagnosis | None:
    """Classify an OpenFAST run's failure from its console output.

    Parameters
    ----------
    output : str
        The run's combined standard output and error.
    returncode : int
        The executable's exit status.

    Returns
    -------
    OpenFastDiagnosis or None
        The classified failure, or ``None`` for a successful exit without the
        forced-linearization warning. A successful exit *with* that warning is
        :attr:`OpenFastFailureMode.TRIM_NOT_CONVERGED`.

    Notes
    -----
    A non-zero exit is classified in this order: an error during module
    initialisation, generator motoring, rotor overspeed, a divergence signature, the
    forced-linearization warning, and otherwise :attr:`OpenFastFailureMode.UNKNOWN`.
    Initialisation comes first because a deck that cannot be read fails there whatever
    else is printed.

    OpenFAST wraps its console messages at about 100 columns, so a signature can be
    split across two lines (``u%HSS_Spd is`` / ``negative.``). Matching is therefore
    done on the output with every run of whitespace, line breaks included, collapsed
    to a single space, and the quoted message joins the lines it spans.
    """
    text = _flatten(output).lower()
    if returncode == 0:
        if _FORCED_LINEARIZATION_SIGNATURE in text:
            return OpenFastDiagnosis(
                mode=OpenFastFailureMode.TRIM_NOT_CONVERGED,
                detail=(
                    "OpenFAST forced the linearization without finding the steady "
                    "state (CalcSteady did not converge before TMax)"
                ),
                openfast_message=_matching_line(
                    output, _FORCED_LINEARIZATION_SIGNATURE
                ),
            )
        return None
    failure_time = _failure_time(output)
    tail = _tail(output)
    if _MODULE_INIT_SIGNATURE in text:
        return OpenFastDiagnosis(
            mode=OpenFastFailureMode.INITIALIZATION_ERROR,
            detail=(
                f"OpenFAST exited with code {returncode} during module initialization "
                "(an input problem; retrying with other trim settings cannot fix it)"
            ),
            openfast_message=tail,
            failure_time=0.0,
        )
    if failure_time is None and _SIMULATION_INIT_SIGNATURE in text:
        failure_time = 0.0
    when = "" if failure_time is None else f" at t = {failure_time:g} s"
    if _MOTORING_SIGNATURE in text:
        return OpenFastDiagnosis(
            mode=OpenFastFailureMode.MOTORING,
            detail=f"the generator speed went negative (motoring){when}",
            openfast_message=_matching_line(output, _MOTORING_SIGNATURE),
            failure_time=failure_time,
        )
    if _OVERSPEED_SIGNATURE in text:
        return OpenFastDiagnosis(
            mode=OpenFastFailureMode.ROTOR_OVERSPEED,
            detail=f"the rotor spun too fast for the CalcSteady azimuth step{when}",
            openfast_message=_matching_line(output, _OVERSPEED_SIGNATURE),
            failure_time=failure_time,
        )
    match = _DIVERGENCE_RE.search(text)
    if match is not None:
        return OpenFastDiagnosis(
            mode=OpenFastFailureMode.DIVERGENCE,
            detail=f"the response diverged before a fatal error{when}",
            openfast_message=_matching_line(output, match.group(0).lower()) or tail,
            failure_time=failure_time,
        )
    if _FORCED_LINEARIZATION_SIGNATURE in text:
        return OpenFastDiagnosis(
            mode=OpenFastFailureMode.TRIM_NOT_CONVERGED,
            detail=(
                f"OpenFAST exited with code {returncode} after forcing the "
                "linearization without finding the steady state"
            ),
            openfast_message=tail,
            failure_time=failure_time,
        )
    return OpenFastDiagnosis(
        mode=OpenFastFailureMode.UNKNOWN,
        detail=f"OpenFAST exited with code {returncode}{when}",
        openfast_message=tail,
        failure_time=failure_time,
    )


def _failure_time(output: str) -> float | None:
    """Return the simulation time of OpenFAST's fatal error, if it reported one."""
    match = _FAILURE_TIME_RE.search(_flatten(output))
    if match is None:
        return None
    try:
        value = float(match.group(1).replace("D", "E").replace("d", "e"))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _flatten(output: str) -> str:
    """Return ``output`` with every whitespace run (line breaks too) as one space."""
    return " ".join(output.split())


def _matching_line(output: str, needle: str) -> str:
    """Return the output line(s) containing ``needle`` (case-insensitive), joined.

    The non-empty lines are joined with single spaces (undoing OpenFAST's console
    wrapping) and the lines that the first occurrence of ``needle`` spans are
    returned as one line.
    """
    lines = [" ".join(line.split()) for line in output.splitlines()]
    lines = [line for line in lines if line]
    starts: list[int] = []
    position = 0
    for line in lines:
        starts.append(position)
        position += len(line) + 1
    index = " ".join(lines).lower().find(_flatten(needle).lower())
    if index < 0:
        return ""
    end = index + len(_flatten(needle))
    first = max(i for i, start in enumerate(starts) if start <= index)
    last = max(i for i, start in enumerate(starts) if start < end)
    return " ".join(lines[first : last + 1])[:_MESSAGE_LIMIT]


def _tail(output: str) -> str:
    """Return the last few non-empty output lines, length-limited."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return "\n".join(lines[-_TAIL_LINES:])[-_MESSAGE_LIMIT:]


def find_openfast(executable: str | Path = "openfast") -> Path:
    """Locate the OpenFAST executable.

    Parameters
    ----------
    executable : str or pathlib.Path, optional
        Executable name to look up on ``PATH``, or an explicit path to it.

    Returns
    -------
    pathlib.Path
        The resolved executable path.

    Raises
    ------
    OpenFastError
        If the executable is neither an existing file nor found on ``PATH``.
    """
    candidate = Path(executable)
    if candidate.is_file():
        # Resolve to an absolute path: run_openfast invokes the executable with
        # cwd set to the deck directory, where a relative path would not resolve.
        return candidate.resolve()
    found = shutil.which(str(executable))
    if found is None:
        msg = (
            f"OpenFAST executable {executable!r} not found on PATH; pass an explicit "
            "path via the executable argument"
        )
        raise OpenFastError(msg)
    # ``which`` can return a relative path when PATH has a relative entry; resolve
    # it too, since run_openfast runs from the deck directory.
    return Path(found).resolve()


def find_lin_files(fst_path: str | Path) -> list[Path]:
    """Return the full-system ``.lin`` files for a deck, ordered by index.

    Parameters
    ----------
    fst_path : str or pathlib.Path
        The primary ``.fst`` deck (its stem and directory locate the outputs).

    Returns
    -------
    list[pathlib.Path]
        The ``<root>.<i>.lin`` files sorted by ``i``; per-module files
        (``<root>.<Module>.<i>.lin``) are excluded.
    """
    fst = Path(fst_path)
    root = fst.stem
    indexed: list[tuple[int, Path]] = []
    # Glob every ``*.lin`` and filter by literal name, rather than globbing on the
    # stem: a stem with glob metacharacters (e.g. ``case[1]``) would otherwise be
    # treated as a pattern and miss the real files.
    for candidate in fst.parent.glob("*.lin"):
        match = _LIN_INDEX_RE.search(candidate.name)
        # Keep only files whose segment between the root and ``.lin`` is a pure
        # integer index (the full-system files), so ``<root>.1.ED.lin`` is skipped.
        if match is not None and candidate.name == f"{root}.{int(match.group(1))}.lin":
            indexed.append((int(match.group(1)), candidate))
    return [path for _, path in sorted(indexed)]


def run_openfast(
    fst_path: str | Path,
    *,
    executable: str | Path = "openfast",
    timeout: float | None = None,
    extra_args: tuple[str, ...] = (),
    check_path_lengths: bool | None = None,
) -> OpenFastResult:
    r"""Run OpenFAST on a deck and collect the linearization output.

    The executable is invoked from the deck's directory so its relative sub-file
    references resolve. Pre-existing generated ``.lin``, time-series, and summary
    files for this deck are removed first so the returned paths reflect only this run.

    Parameters
    ----------
    fst_path : str or pathlib.Path
        The primary ``.fst`` deck to run. Should already be configured for
        linearization (``Linearize = True``).
    executable : str or pathlib.Path, optional
        OpenFAST executable name or path (default ``"openfast"`` on ``PATH``).
    timeout : float or None, optional
        Maximum seconds to allow the run; ``None`` waits indefinitely.
    extra_args : tuple[str, ...], optional
        Additional command-line arguments inserted before the deck path.
    check_path_lengths : bool or None, optional
        Run :func:`~vane.run.check_deck_path_lengths` before launching. ``None`` (the
        default) checks on Windows only, where the path limit applies.

    Returns
    -------
    OpenFastResult
        The return code, captured output, and the produced ``.lin`` files.

    Raises
    ------
    OpenFastError
        If the executable cannot be found, the deck does not exist, or the deck's
        directory or the executable can be named only with the Windows ``\\?\``
        extended-length prefix (longer than the Windows path limit). A shorter path
        given with the prefix is passed without it.
    vane.run.DeckPathLengthError
        If OpenFAST would open one of the deck's files by a path longer than the
        Windows limit (checked before the process is launched).
    OpenFastTimeoutError
        If the run exceeds ``timeout`` (a subclass of ``OpenFastError``).
    ValueError
        If ``timeout`` is not positive and finite.

    Notes
    -----
    The executable is started in its own process group (Windows) or session (POSIX).
    On a timeout the **whole process tree** is killed, not just the direct child, so a
    wrapper executable (a ``.bat`` or shell script that launches OpenFAST) cannot leave
    an orphaned solver running and holding the output pipes open.
    """
    validated_timeout = _validated_timeout(timeout)
    fst = Path(fst_path)
    if not fst.is_file():
        msg = f"OpenFAST deck not found: {fst}"
        raise OpenFastError(msg)
    # Pass both paths without the extended-length prefix: under a prefixed working
    # directory OpenFAST still does not collapse ``..`` and the prefix's characters
    # count against its path limit, so it only fails sooner.
    exe = normalize_extended_path(find_openfast(executable))
    working_dir = normalize_extended_path(fst.parent)
    if has_extended_prefix(working_dir):
        msg = (
            f"The deck directory {working_dir} is longer than the Windows path limit; "
            "OpenFAST cannot open its input files from there. Use a shorter output_dir"
        )
        raise OpenFastError(msg)
    if has_extended_prefix(exe):
        msg = (
            f"The OpenFAST executable path {exe} is longer than the Windows path "
            "limit; install or copy OpenFAST to a shorter path"
        )
        raise OpenFastError(msg)
    if check_path_lengths is None:
        check_path_lengths = sys.platform == "win32"
    if check_path_lengths:
        check_deck_path_lengths(fst)

    # Clear stale outputs so every returned path was produced by this run, including
    # per-module (LinOutMod) sidecars a previous run of this deck left behind.
    for stale in (*find_lin_files(fst), *_module_lin_files(fst)):
        unlink_with_retry(stale)
    for suffix in (".out", ".outb", ".sum"):
        unlink_with_retry(fst.with_suffix(suffix), missing_ok=True)

    command = [str(exe), *extra_args, fst.name]
    logger.info("Running OpenFAST: %s (cwd=%s)", " ".join(command), working_dir)
    returncode, stdout, stderr = _run_process_tree(
        command, working_dir, validated_timeout, fst
    )

    if returncode != 0:
        logger.warning("OpenFAST exited with code %d for %s", returncode, fst)
    return OpenFastResult(
        fst_path=fst,
        returncode=returncode,
        lin_files=find_lin_files(fst),
        output_file=_first_existing(fst, (".out", ".outb")),
        summary_file=_first_existing(fst, (".sum",)),
        stdout=stdout or "",
        stderr=stderr or "",
    )


def _run_process_tree(
    command: list[str], cwd: Path, timeout: float | None, fst: Path
) -> tuple[int, str, str]:
    """Run ``command`` in its own process group, killing the whole tree on timeout.

    Returns the exit code and the captured standard output and error. The output is
    decoded with the locale encoding, and an undecodable byte (OpenFAST echoes paths
    and deck text in the console code page) is replaced rather than raised: a decoding
    error in a pipe-reader thread would otherwise lose the whole output.

    Raises
    ------
    OpenFastTimeoutError
        If the command exceeds ``timeout``; the process tree has then been killed.
    """
    if sys.platform == "win32":
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    else:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
        )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _kill_process_tree(process)
        try:
            process.communicate(timeout=_KILL_DRAIN_SECONDS)
        except subprocess.TimeoutExpired:
            logger.warning(
                "OpenFAST output pipes for %s stayed open after the process tree was "
                "killed; abandoning them",
                fst,
            )
        msg = f"OpenFAST run timed out after {timeout} s for {fst}"
        raise OpenFastTimeoutError(msg) from exc
    except BaseException:
        # Never leave a solver tree running behind an interrupted caller.
        _kill_process_tree(process)
        raise
    return process.returncode, stdout, stderr


def _kill_process_tree(process: subprocess.Popen[str]) -> None:
    """Kill ``process`` and every descendant it started."""
    if sys.platform == "win32":
        # taskkill /T follows the parent-process links from the wrapper to OpenFAST.
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(process.pid)],
            capture_output=True,
            check=False,
        )
    else:
        # ProcessLookupError: the whole group has already exited.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    process.kill()  # no-op when already dead; ensures the direct child is reaped


def _module_lin_files(fst: Path) -> list[Path]:
    """Return the deck's per-module ``<root>.<index>.<Module>.lin`` sidecars."""
    root = fst.stem
    return [
        candidate
        for candidate in fst.parent.glob("*.lin")
        if candidate.name.startswith(f"{root}.")
        and _MODULE_LIN_SUFFIX_RE.fullmatch(candidate.name[len(root) :]) is not None
    ]


def _first_existing(fst: Path, suffixes: tuple[str, ...]) -> Path | None:
    """Return the first ``<root><suffix>`` file that exists, or ``None``."""
    for suffix in suffixes:
        candidate = fst.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None
