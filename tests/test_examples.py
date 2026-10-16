# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for the tutorial scripts in ``examples/``.

Each numbered script runs in a subprocess with ``--output`` pointing at a temporary
directory, so it never writes into the repository. The OpenFAST tutorial runs only
when ``VANE_OPENFAST`` and ``VANE_RTEST_DIR`` name an existing executable and a clone
of the OpenFAST ``r-test`` repository, and then only with ``--configure-only``, which
reads the deck without launching OpenFAST.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parent.parent
EXAMPLES = REPOSITORY / "examples"
SCRIPTS = sorted(EXAMPLES.glob("[0-9]*.py"))
OPENFAST_SCRIPT = "01_linearization_sweep.py"
DECK = Path("glue-codes", "openfast", "5MW_Land_DLL_WTurb", "5MW_Land_DLL_WTurb.fst")


def _environment(**overrides: str | None) -> dict[str, str]:
    """Return the environment of a script run.

    Parameters
    ----------
    **overrides : str or None
        Variables to set, or to remove when ``None``.

    Returns
    -------
    dict[str, str]
        The process environment with the Agg backend and ``src`` on the path.
    """
    environment = dict(os.environ)
    paths = [str(REPOSITORY / "src"), environment.get("PYTHONPATH", "")]
    environment["PYTHONPATH"] = os.pathsep.join(p for p in paths if p)
    environment["MPLBACKEND"] = "Agg"
    for name, value in overrides.items():
        if value is None:
            environment.pop(name, None)
        else:
            environment[name] = value
    return environment


def _run(
    script: Path, arguments: list[str], environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run one script and return the completed process.

    Parameters
    ----------
    script : pathlib.Path
        The script.
    arguments : list[str]
        Its command-line arguments.
    environment : dict[str, str]
        The process environment.

    Returns
    -------
    subprocess.CompletedProcess[str]
        The finished process with its captured output.
    """
    return subprocess.run(
        [sys.executable, str(script), *arguments],
        cwd=REPOSITORY,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def _openfast_available() -> bool:
    """Return whether the OpenFAST tutorial's inputs exist.

    Returns
    -------
    bool
        ``True`` when ``VANE_OPENFAST`` names a file and ``VANE_RTEST_DIR`` a clone
        that holds the NREL 5 MW land-based deck.
    """
    executable = os.environ.get("VANE_OPENFAST")
    clone = os.environ.get("VANE_RTEST_DIR")
    return bool(
        executable
        and clone
        and Path(executable).is_file()
        and (Path(clone) / DECK).is_file()
    )


class TestExamples:
    """Every tutorial script runs to completion."""

    def test_scripts_found(self) -> None:
        """The six numbered tutorial scripts are present."""
        assert len(SCRIPTS) == 6

    @pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.stem)
    def test_script_runs(self, script: Path, tmp_path: Path) -> None:
        """A script exits with status 0 and writes only into ``--output``."""
        arguments = ["--output", str(tmp_path)]
        if script.name == OPENFAST_SCRIPT:
            if not _openfast_available():
                pytest.skip("set VANE_OPENFAST and VANE_RTEST_DIR to run")
            arguments.append("--configure-only")
        completed = _run(script, arguments, _environment())
        assert completed.returncode == 0, completed.stderr
        if script.name != OPENFAST_SCRIPT:
            assert list(tmp_path.glob("*.png")), "no figure written"

    def test_openfast_tutorial_without_inputs(self, tmp_path: Path) -> None:
        """Without OpenFAST and r-test the sweep tutorial explains and exits 0."""
        environment = _environment(VANE_OPENFAST=None, VANE_RTEST_DIR=None)
        completed = _run(
            EXAMPLES / OPENFAST_SCRIPT, ["--output", str(tmp_path)], environment
        )
        assert completed.returncode == 0, completed.stderr
        assert "VANE_RTEST_DIR is not set" in completed.stdout
        assert "Nothing was run." in completed.stdout
