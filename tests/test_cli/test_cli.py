# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.cli — the ``vane`` command-line interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from typer.testing import CliRunner

from tests.conftest import build_lin_text
from vane import __version__
from vane.cli import app

if TYPE_CHECKING:
    from pathlib import Path

_STATES: list[tuple[str, bool, int]] = [
    ("ED 1st tower fore-aft (internal DOF index = DOF_TFA1), m", False, 2),
    (
        "ED First time derivative of 1st tower fore-aft "
        "(internal DOF index = DOF_TFA1), m/s",
        False,
        1,
    ),
]
_A = np.array([[0.0, 1.0], [-4.0, -0.2]], dtype=np.float64)

runner = CliRunner()


def _write_point(
    directory: Path, case: str, rotor_speed: float, *, wind_speed: float = 0.0
) -> list[Path]:
    """Write one operating point's two-azimuth .lin files and return their paths."""
    paths = []
    for index, azimuth in enumerate((0.0, 1.0), start=1):
        path = directory / f"{case}.{index}.lin"
        path.write_text(
            build_lin_text(
                states=_STATES,
                a=_A,
                rotor_speed=rotor_speed,
                azimuth=azimuth,
                wind_speed=wind_speed,
            ),
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def _sweep_dir(tmp_path: Path) -> Path:
    """Write a directory with two operating points at distinct rotor speeds."""
    directory = tmp_path / "sweep"
    directory.mkdir()
    _write_point(directory, "ws08", 1.0)
    _write_point(directory, "ws10", 1.5)
    return directory


class TestTopLevel:
    """Version and help."""

    def test_version(self) -> None:
        """``--version`` prints the version and exits cleanly."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.stdout

    def test_help_lists_commands(self) -> None:
        """The help text lists every subcommand."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for command in ("modes", "campbell", "study"):
            assert command in result.stdout


class TestModes:
    """The ``modes`` command."""

    def test_prints_table(self, tmp_path: Path) -> None:
        """It prints a non-empty modal table for one operating point."""
        files = _write_point(tmp_path, "pt", 1.0)
        result = runner.invoke(app, ["modes", *[str(f) for f in files]])
        assert result.exit_code == 0
        assert result.stdout.strip()

    def test_writes_csv(self, tmp_path: Path) -> None:
        """With -o it writes the table to disk."""
        files = _write_point(tmp_path, "pt", 1.0)
        out = tmp_path / "modes.csv"
        result = runner.invoke(app, ["modes", *[str(f) for f in files], "-o", str(out)])
        assert result.exit_code == 0
        assert out.is_file()
        assert out.read_text(encoding="utf-8").strip()

    def test_missing_file_is_rejected(self, tmp_path: Path) -> None:
        """A non-existent .lin path fails validation (exit code 2)."""
        result = runner.invoke(app, ["modes", str(tmp_path / "nope.lin")])
        assert result.exit_code == 2

    def test_malformed_lin_is_handled(self, tmp_path: Path) -> None:
        """A garbage .lin file exits 1 with a message, not a traceback."""
        bad = tmp_path / "bad.1.lin"
        bad.write_text("this is not a linearization file", encoding="utf-8")
        result = runner.invoke(app, ["modes", str(bad)])
        assert result.exit_code == 1
        # A clean typer.Exit surfaces as SystemExit, not the underlying parse error.
        assert isinstance(result.exception, SystemExit)


class TestCampbell:
    """The ``campbell`` command."""

    def test_prints_table(self, tmp_path: Path) -> None:
        """It prints the Campbell table for a directory of operating points."""
        result = runner.invoke(app, ["campbell", str(_sweep_dir(tmp_path))])
        assert result.exit_code == 0
        assert result.stdout.strip()

    def test_writes_csv(self, tmp_path: Path) -> None:
        """With -o it writes the Campbell table to disk."""
        out = tmp_path / "campbell.csv"
        result = runner.invoke(
            app, ["campbell", str(_sweep_dir(tmp_path)), "-o", str(out)]
        )
        assert result.exit_code == 0
        assert out.is_file()

    def test_writes_mat_without_blade_dofs(self, tmp_path: Path) -> None:
        """-o out.mat succeeds for a sweep whose labels carry no multi-blade coordinate.

        The all-None ``multiblade`` column used to crash ``savemat`` with a TypeError.
        """
        from scipy.io import loadmat

        out = tmp_path / "campbell.mat"
        result = runner.invoke(
            app, ["campbell", str(_sweep_dir(tmp_path)), "-o", str(out)]
        )
        assert result.exit_code == 0, result.output
        loaded = loadmat(out)
        assert "multiblade" in loaded

    def test_empty_directory_errors(self, tmp_path: Path) -> None:
        """A directory with no .lin files exits non-zero (handled, not a traceback)."""
        empty = tmp_path / "empty"
        empty.mkdir()
        result = runner.invoke(app, ["campbell", str(empty)])
        assert result.exit_code == 1

    def test_rotor_speed_plateau_points_at_wind_axis(self, tmp_path: Path) -> None:
        """An above-rated sweep fails on rpm with a hint, and runs on wind."""
        directory = tmp_path / "above_rated"
        directory.mkdir()
        for case, wind in (("ws14", 14.0), ("ws12", 12.0), ("ws16", 16.0)):
            _write_point(directory, case, 1.2671, wind_speed=wind)

        rpm = runner.invoke(app, ["campbell", str(directory)])
        assert rpm.exit_code == 1
        assert "--parameter wind" in rpm.output

        wind_run = runner.invoke(app, ["campbell", str(directory), "-p", "wind"])
        assert wind_run.exit_code == 0
        assert wind_run.stdout.strip()


class TestStudy:
    """The ``study`` command."""

    def test_writes_bundle(self, tmp_path: Path) -> None:
        """It writes a provenance bundle with the manifest and Campbell table."""
        bundle = tmp_path / "bundle"
        result = runner.invoke(
            app, ["study", str(_sweep_dir(tmp_path)), "-o", str(bundle)]
        )
        assert result.exit_code == 0
        assert (bundle / "provenance.json").is_file()
        assert (bundle / "campbell.csv").is_file()
        assert (bundle / "report.html").is_file()
        # The discovered <case> names reach the manifest (not stripped to empty).
        manifest = (bundle / "provenance.json").read_text(encoding="utf-8")
        assert "ws08" in manifest
        assert "ws10" in manifest

    def test_no_report_skips_html(self, tmp_path: Path) -> None:
        """``--no-report`` omits the HTML report."""
        bundle = tmp_path / "bundle"
        result = runner.invoke(
            app,
            ["study", str(_sweep_dir(tmp_path)), "-o", str(bundle), "--no-report"],
        )
        assert result.exit_code == 0
        assert (bundle / "provenance.json").is_file()
        assert not (bundle / "report.html").exists()

    def test_simultaneous_tracking(self, tmp_path: Path) -> None:
        """The tracking strategy option is accepted and runs."""
        bundle = tmp_path / "bundle"
        result = runner.invoke(
            app,
            [
                "study",
                str(_sweep_dir(tmp_path)),
                "-o",
                str(bundle),
                "-t",
                "simultaneous",
            ],
        )
        assert result.exit_code == 0
        assert (bundle / "provenance.json").is_file()
