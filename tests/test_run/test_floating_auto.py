# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.run.floating_auto -- automatic floating settle setup."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.conftest import build_lin_text
from vane.io.fst_writer import FastInputFile
from vane.run import floating_auto as floating_module
from vane.run import steady_state as steady_state_module
from vane.run.floating_auto import (
    CONTROLLER_LIBRARIES_ENV,
    ControllerLibraryResolution,
    PlatformProbe,
    PlatformProbeError,
    controller_library_search_paths,
    find_controller_library,
    is_floating_deck,
    plan_floating_settle,
    platform_probe_from_jacobians,
    probe_platform,
)
from vane.run.runner import OpenFastError, OpenFastResult, OpenFastTimeoutError
from vane.run.steady_state import SettlePlatformDamping, SteadyStateError

if TYPE_CHECKING:
    import numpy.typing as npt

_SERVO = """\
5 PCMode - pitch control mode
5 VSContrl - variable-speed control mode
0 HSSBrMode - brake
0 YCMode - yaw
"{library}" DLL_FileName - controller library
"DISCON.IN" DLL_InFile - controller input
"""

_ED = """\
0.0 RotSpeed - rpm
0.0 BlPitch(1) - deg
0.0 BlPitch(2) - deg
0.0 BlPitch(3) - deg
True GenDOF - generator DOF
True DrTrDOF - drivetrain DOF
True YawDOF - yaw DOF
{ptfm} PtfmSgDOF - platform surge DOF
{ptfm} PtfmPDOF - platform pitch DOF
0 PtfmSurge - m
0 PtfmPitch - deg
"""

_ROW_ZERO = "  0  0  0  0  0  0"
_HD = f"""\
------- HydroDyn Input File ----
             1   PotMod      - potential flow
             1   RdtnMod     - radiation
             1   ExctnMod    - excitation
             1   NBody       - Number of WAMIT bodies
             1   NBodyMod    - Body coupling model
           0.0   AddF0    - Additional preload
           0.0
           0.0
           0.0
           0.0
           0.0
{_ROW_ZERO}   AddBLin  - Additional linear damping
{_ROW_ZERO}
{_ROW_ZERO}
{_ROW_ZERO}
{_ROW_ZERO}
{_ROW_ZERO}
{_ROW_ZERO}   AddBQuad - Additional quadratic drag
{_ROW_ZERO}
{_ROW_ZERO}
{_ROW_ZERO}
{_ROW_ZERO}
{_ROW_ZERO}
"""


def _primary(*, sub: int = 0, mooring: int = 3, servo: bool = True) -> str:
    lines = [
        "0.0125 DT - time step",
        "60 TMax - run time",
        "False Linearize - flag",
        "False CalcSteady - flag",
        "1 NLinTimes - -",
        "0 LinTimes - s",
        "1 CompInflow - inflow",
        "2 CompAero - aero",
        f"{1 if servo else 0} CompServo - servo",
        "1 CompHydro - hydro",
        f"{sub} CompSub - substructure",
        f"{mooring} CompMooring - mooring",
        '"ED.dat" EDFile - ElastoDyn',
        '"HD.dat" HydroFile - HydroDyn',
        '"none" InflowFile - InflowWind',
    ]
    if servo:
        lines.append('"servo/SD.dat" ServoFile - ServoDyn')
    return "\n".join(lines) + "\n"


def _make_deck(
    tmp_path: Path,
    *,
    library: str = "/home/runner/lib/libdiscon.so",
    floating: bool = True,
    sub: int = 0,
    mooring: int = 3,
    servo: bool = True,
) -> Path:
    root = tmp_path / "template"
    (root / "servo").mkdir(parents=True)
    (root / "servo" / "SD.dat").write_text(
        _SERVO.format(library=library), encoding="utf-8"
    )
    (root / "ED.dat").write_text(
        _ED.format(ptfm="True" if floating else "False"), encoding="utf-8"
    )
    (root / "HD.dat").write_text(_HD, encoding="utf-8")
    fst = root / "Model.fst"
    fst.write_text(_primary(sub=sub, mooring=mooring, servo=servo), encoding="utf-8")
    return fst


@pytest.fixture(autouse=True)
def _host_dll(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the host a Windows-style ``.dll`` host whatever the test platform."""
    monkeypatch.setattr(floating_module, "_host_library_suffix", lambda: ".dll")


class TestFindControllerLibrary:
    """Locating a controller library this host can load."""

    def test_no_library_used(self, tmp_path: Path) -> None:
        """A deck that routes no control through a library needs none."""
        fst = _make_deck(tmp_path, servo=False)
        assert find_controller_library(fst, environ={}).status == "not_used"

    def test_deck_library_loadable(self, tmp_path: Path) -> None:
        """The deck's own host library is used unchanged."""
        fst = _make_deck(tmp_path, library="DISCON.dll")
        (fst.parent / "servo" / "DISCON.dll").write_bytes(b"MZ")
        resolution = find_controller_library(fst, environ={})
        assert resolution.status == "deck"
        assert resolution.library is None
        assert resolution.available

    def test_foreign_suffix_is_not_loadable(self, tmp_path: Path) -> None:
        """An existing library for another platform is not used as is."""
        fst = _make_deck(tmp_path, library="libdiscon.so")
        (fst.parent / "servo" / "libdiscon.so").write_bytes(b"ELF")
        (fst.parent / "servo" / "libdiscon.dll").write_bytes(b"MZ")
        resolution = find_controller_library(fst, environ={})
        assert resolution.status == "found"
        assert resolution.library is not None
        assert resolution.library.name == "libdiscon.dll"
        assert resolution.source == "next to the deck's library"

    def test_variant_next_to_primary_deck(self, tmp_path: Path) -> None:
        """A host build beside the primary deck matches by library name."""
        fst = _make_deck(tmp_path)
        (fst.parent / "DISCON_x64.dll").write_bytes(b"MZ")
        (fst.parent / "other.dll").write_bytes(b"MZ")
        resolution = find_controller_library(fst, environ={})
        assert resolution.status == "found"
        assert resolution.library is not None
        assert resolution.library.name == "DISCON_x64.dll"
        assert resolution.source == "next to the primary deck"

    def test_single_quoted_deck_library_matches_by_name(self, tmp_path: Path) -> None:
        """A Fortran single-quoted DLL_FileName is matched without its quotes."""
        fst = _make_deck(tmp_path)
        servo = fst.parent / "servo" / "SD.dat"
        servo.write_text(
            servo.read_text(encoding="utf-8").replace(
                '"/home/runner/lib/libdiscon.so"', "'/home/runner/lib/libdiscon.so'"
            ),
            encoding="utf-8",
        )
        assert "'/home/runner/lib/libdiscon.so'" in servo.read_text(encoding="utf-8")
        (fst.parent / "discon.dll").write_bytes(b"MZ")
        resolution = find_controller_library(fst, environ={})
        assert resolution.status == "found"
        assert resolution.library is not None
        assert resolution.library.name == "discon.dll"

    def test_prefers_64_bit_build(self, tmp_path: Path) -> None:
        """Within one directory a plain or 64-bit build beats a 32-bit one."""
        fst = _make_deck(tmp_path)
        library_dir = tmp_path / "libs"
        library_dir.mkdir()
        for name in ("libdiscon_win32.dll", "libdiscon_x64.dll"):
            (library_dir / name).write_bytes(b"MZ")
        resolution = find_controller_library(fst, search_paths=[library_dir])
        assert resolution.library is not None
        assert resolution.library.name == "libdiscon_x64.dll"
        assert resolution.source == "search_paths"

    def test_environment_directory(self, tmp_path: Path) -> None:
        """VANE_CONTROLLER_LIBRARIES directories are searched."""
        fst = _make_deck(tmp_path)
        library_dir = tmp_path / "controller"
        library_dir.mkdir()
        (library_dir / "libdiscon.dll").write_bytes(b"MZ")
        entries = (str(tmp_path / "missing"), f'"{library_dir}"', "")
        environ = {CONTROLLER_LIBRARIES_ENV: os.pathsep.join(entries)}
        resolution = find_controller_library(fst, environ=environ)
        assert resolution.status == "found"
        assert resolution.source == CONTROLLER_LIBRARIES_ENV
        assert resolution.to_dict()["library"] == str(resolution.library)

    def test_config_file_entries(self, tmp_path: Path) -> None:
        """The configuration file lists files or directories, comments ignored."""
        fst = _make_deck(tmp_path)
        config = tmp_path / "config"
        (config / "builds").mkdir(parents=True)
        (config / "builds" / "libDISCON.dll").write_bytes(b"MZ")
        (config / "controller_libraries.txt").write_text(
            "# host builds\n\nbuilds/libDISCON.dll  # controller\n", encoding="utf-8"
        )
        environ = {"VANE_CONFIG_DIR": str(config)}
        entries = controller_library_search_paths(environ)
        assert entries[0][0] == config / "builds" / "libDISCON.dll"
        resolution = find_controller_library(fst, environ=environ)
        assert resolution.status == "found"
        assert resolution.library is not None
        assert resolution.library.name == "libDISCON.dll"

    def test_file_entry_must_match_the_library(self, tmp_path: Path) -> None:
        """A listed library of another controller is never substituted."""
        fst = _make_deck(tmp_path)
        other = tmp_path / "turbsim.dll"
        other.write_bytes(b"MZ")
        resolution = find_controller_library(
            fst, environ={CONTROLLER_LIBRARIES_ENV: str(other)}
        )
        assert resolution.status == "missing"
        assert not resolution.available
        assert other in resolution.searched

    def test_missing(self, tmp_path: Path) -> None:
        """No host build anywhere is reported, not raised."""
        fst = _make_deck(tmp_path)
        resolution = find_controller_library(fst, environ={})
        assert resolution.status == "missing"
        assert resolution.deck_library == "/home/runner/lib/libdiscon.so"

    def test_config_dir_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The configuration directory follows the platform convention."""
        monkeypatch.setattr(floating_module.sys, "platform", "linux")
        assert floating_module._config_dir({"XDG_CONFIG_HOME": "/x"}).as_posix() == (
            "/x/vane"
        )
        assert floating_module._config_dir({}).name == "vane"
        monkeypatch.setattr(floating_module.sys, "platform", "win32")
        assert floating_module._config_dir({"APPDATA": "C:/a"}).as_posix() == (
            "C:/a/vane"
        )

    @pytest.mark.parametrize(
        ("name", "key"),
        [
            ("libdiscon.so", "discon"),
            ("DISCON_x64.dll", "discon"),
            ("libDISCON_win32.dylib", "discon"),
            ("lib", "lib"),
        ],
    )
    def test_library_key(self, name: str, key: str) -> None:
        """Library names reduce to the controller's identity."""
        assert floating_module._library_key(name) == key

    def test_host_suffix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each platform loads its own library suffix."""
        monkeypatch.undo()
        for platform, suffix in (("win32", ".dll"), ("darwin", ".dylib"), ("x", ".so")):
            monkeypatch.setattr(floating_module.sys, "platform", platform)
            assert floating_module._host_library_suffix() == suffix


class TestIsFloatingDeck:
    """Telling a floating platform from a fixed one."""

    def test_floating(self, tmp_path: Path) -> None:
        """Free platform DOFs with mooring and no substructure float."""
        assert is_floating_deck(_make_deck(tmp_path))

    def test_monopile_substructure(self, tmp_path: Path) -> None:
        """A SubDyn substructure carries the platform DOFs of a monopile."""
        assert not is_floating_deck(_make_deck(tmp_path, sub=1, mooring=0))

    def test_moored_subdyn_substructure_floats(self, tmp_path: Path) -> None:
        """A moored SubDyn substructure is a floating platform (e.g. a flexible semi).

        OpenFAST models a flexible floater with SubDyn and MoorDyn; a fixed-bottom
        substructure (monopile, jacket) is not moored.
        """
        assert is_floating_deck(_make_deck(tmp_path, sub=1, mooring=3))

    def test_fixed_platform(self, tmp_path: Path) -> None:
        """Locked platform DOFs are not floating."""
        assert not is_floating_deck(_make_deck(tmp_path, floating=False))

    @pytest.mark.parametrize("spelling", [".T.", ".true.", "T", "1"])
    def test_fortran_logical_platform_flags(
        self, tmp_path: Path, spelling: str
    ) -> None:
        """Every Fortran spelling of a free platform DOF counts (``.T.`` included)."""
        fst = _make_deck(tmp_path)
        ed = fst.parent / "ED.dat"
        ed.write_text(
            ed.read_text(encoding="utf-8").replace("True Ptfm", f"{spelling} Ptfm"),
            encoding="utf-8",
        )
        assert ".T. Ptfm" in ed.read_text(encoding="utf-8") or spelling != ".T."
        assert is_floating_deck(fst)

    def test_unreadable_deck(self, tmp_path: Path) -> None:
        """A deck without an ElastoDyn file is not floating."""
        fst = tmp_path / "Model.fst"
        fst.write_text("1 CompMooring - m\n", encoding="utf-8")
        assert not is_floating_deck(fst)


# Synthetic platform: six DOFs plus one tower DOF, lightly damped.
_DOF_TEXT = {
    "surge": "Platform horizontal surge translation DOF (internal DOF index = DOF_Sg)",
    "sway": "Platform horizontal sway translation DOF (internal DOF index = DOF_Sw)",
    "heave": "Platform vertical heave translation DOF (internal DOF index = DOF_Hv)",
    "roll": "Platform roll tilt rotation DOF (internal DOF index = DOF_R)",
    "pitch": "Platform pitch tilt rotation DOF (internal DOF index = DOF_P)",
    "yaw": "Platform yaw rotation DOF (internal DOF index = DOF_Y)",
    "tower": "1st tower fore-aft bending mode DOF (internal DOF index = DOF_TFA1)",
}
_MASS = np.array([3.0e7, 3.0e7, 4.5e7, 5.0e10, 5.0e10, 4.5e10, 4.0e5])
_PERIOD = np.array([125.0, 125.0, 20.0, 28.0, 28.0, 85.0, 2.0])
_PROBE = SettlePlatformDamping(
    surge=1e6, sway=1e6, heave=1e6, roll=1e9, pitch=1e9, yaw=1e9
)


def _descriptions() -> list[str]:
    names = list(_DOF_TEXT.values())
    return [f"ED {text}, m" for text in names] + [
        f"ED First time derivative of {text}, m/s" for text in names
    ]


def _state_matrix(added: SettlePlatformDamping | None) -> npt.NDArray[np.float64]:
    """Return ``A`` of a mass-spring-damper set with a weak surge-tower coupling."""
    n = _MASS.size
    stiffness = np.diag(_MASS * (2 * np.pi / _PERIOD) ** 2)
    stiffness[0, 6] = stiffness[6, 0] = -1e3
    damping = np.diag(0.02 * 2 * np.sqrt(np.diag(stiffness) * _MASS))
    if added is not None:
        damping[:6, :6] += np.diag(added.diagonal())
    minv = np.diag(1.0 / _MASS)
    a = np.zeros((2 * n, 2 * n))
    a[:n, n:] = np.eye(n)
    a[n:, :n] = -minv @ stiffness
    a[n:, n:] = -minv @ damping
    return a


class TestPlatformProbeFromJacobians:
    """The effective mass and period of each platform DOF from two linearizations."""

    def test_recovers_mass_and_period(self) -> None:
        """A single-DOF oscillator's mass and period are recovered exactly."""
        probe = platform_probe_from_jacobians(
            _state_matrix(None), _state_matrix(_PROBE), _descriptions(), _PROBE
        )
        for index, dof in enumerate(("surge", "sway", "heave", "roll", "pitch", "yaw")):
            # Roll/pitch and surge/sway are exactly degenerate: summed per cluster.
            assert probe.effective_mass[dof] == pytest.approx(_MASS[index], rel=1e-3)
            assert probe.natural_period[dof] == pytest.approx(_PERIOD[index], rel=1e-3)
            assert probe.damping_ratio[dof] == pytest.approx(0.02, rel=1e-2)
            assert probe.mode_frequency[dof] == pytest.approx(
                1 / _PERIOD[index], rel=1e-3
            )
        assert probe.n_states == 14
        damping = probe.settle_damping(0.5)
        assert damping.surge == pytest.approx(
            4 * math.pi * 0.5 * _MASS[0] / _PERIOD[0], rel=1e-3
        )
        record = probe.to_dict()
        assert record["probe_damping"]["surge"] == 1e6  # type: ignore[index]

    def test_unpaired_degenerate_eigenvectors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A repeated root's left/right eigenvectors need not pair up one to one.

        For a repeated eigenvalue any basis of its eigenspace is a valid output of the
        eigensolver, and the left and right bases are computed independently. Here
        the left eigenvectors of every degenerate pair (surge/sway, roll/pitch) are
        swapped, so each mode's own ``w^H u`` vanishes and its per-mode sensitivity is
        meaningless; the cluster sum is not, and must still identify the mode.
        """
        real_eig = floating_module.scipy.linalg.eig

        def unpaired_eig(
            a: npt.NDArray[np.float64], **kwargs: object
        ) -> tuple[
            npt.NDArray[np.complex128],
            npt.NDArray[np.complex128],
            npt.NDArray[np.complex128],
        ]:
            values, left, right = real_eig(a, **kwargs)
            left = left.copy()
            done: set[int] = set()
            for i in range(values.size):
                for j in range(i + 1, values.size):
                    if i in done or j in done:
                        continue
                    if abs(values[i] - values[j]) <= 1e-9 * abs(values[i]):
                        left[:, [i, j]] = left[:, [j, i]]
                        done.update((i, j))
            return values, left, right

        monkeypatch.setattr(floating_module.scipy.linalg, "eig", unpaired_eig)
        probe = platform_probe_from_jacobians(
            _state_matrix(None), _state_matrix(_PROBE), _descriptions(), _PROBE
        )
        for index, dof in enumerate(("surge", "sway", "heave", "roll", "pitch", "yaw")):
            assert probe.effective_mass[dof] == pytest.approx(_MASS[index], rel=1e-3)
            assert probe.natural_period[dof] == pytest.approx(_PERIOD[index], rel=1e-3)

    def test_undamped_dof_is_skipped(self) -> None:
        """A DOF without probe damping is not measured."""
        probe_damping = SettlePlatformDamping(surge=1e6)
        probe = platform_probe_from_jacobians(
            _state_matrix(None),
            _state_matrix(probe_damping),
            _descriptions(),
            probe_damping,
        )
        assert set(probe.effective_mass) == {"surge"}

    def test_shape_mismatch(self) -> None:
        """Matrices must match the state descriptions."""
        with pytest.raises(PlatformProbeError, match="do not match"):
            platform_probe_from_jacobians(
                np.zeros((2, 2)), np.zeros((2, 2)), _descriptions()
            )

    def test_no_platform_dof(self) -> None:
        """A model without free platform DOFs cannot be probed."""
        with pytest.raises(PlatformProbeError, match="no free ElastoDyn platform"):
            platform_probe_from_jacobians(
                np.zeros((2, 2)), np.zeros((2, 2)), ["ED tower, m", "ED tower, m/s"]
            )

    def test_damping_had_no_effect(self) -> None:
        """Unchanged matrices mean the damping was not linearized."""
        a = _state_matrix(None)
        with pytest.raises(PlatformProbeError, match="did not change"):
            platform_probe_from_jacobians(a, a.copy(), _descriptions())


def _probe_deck(tmp_path: Path) -> Path:
    """Return a floating deck the linearization configuration accepts."""
    return _make_deck(tmp_path)


def _fake_probe_run(calls: list[Path]) -> object:
    def fake(fst_path: Path, **_: object) -> OpenFastResult:
        calls.append(fst_path)
        lines = (
            (fst_path.parent / "HD.dat")
            .read_text(encoding="utf-8", errors="replace")
            .splitlines()
        )
        start = next(i for i, line in enumerate(lines) if "AddBLin" in line.split())
        damped = float(lines[start].split()[0]) > 0.0
        a = _state_matrix(_PROBE if damped else None)
        states = [(text, False, 2) for text in _descriptions()]
        lin = fst_path.with_suffix(".1.lin")
        lin.write_text(build_lin_text(states=states, a=a), encoding="utf-8")
        return OpenFastResult(fst_path=fst_path, returncode=0, lin_files=[lin])

    return fake


class TestProbePlatform:
    """The two standstill probe linearizations, with a fake OpenFAST."""

    def test_probe_runs_two_configured_cases(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both copies linearize at standstill without aero or controller."""
        fst = _probe_deck(tmp_path)
        template_hd = (fst.parent / "HD.dat").read_text(encoding="utf-8")
        calls: list[Path] = []
        monkeypatch.setattr(floating_module, "run_openfast", _fake_probe_run(calls))
        probe = probe_platform(fst, tmp_path / "work", probe_damping=_PROBE)
        assert probe.effective_mass["surge"] == pytest.approx(_MASS[0], rel=1e-3)
        assert sorted(path.parent.name for path in calls) == [
            "base",
            "damped",
        ]
        for path in calls:
            deck = FastInputFile.read(path)
            assert deck.get("CompAero") == "0"
            assert deck.get("CompServo") == "0"
            assert deck.get("Linearize") == "True"
            assert float(deck.get("TMax")) == pytest.approx(0.1)
            hydro = FastInputFile.read(path.parent / "HD.dat")
            assert hydro.get("RdtnMod") == "0"
            assert hydro.get("ExctnMod") == "0"
        assert (fst.parent / "HD.dat").read_text(encoding="utf-8") == template_hd

    def test_output_start_after_the_probe_run_is_reset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A deck TStart beyond the 0.1 s probe run is reset, or OpenFAST aborts."""
        fst = _probe_deck(tmp_path)
        text = fst.read_text(encoding="utf-8")
        fst.write_text(
            text.replace("60 TMax - run time", "60 TMax - run time\n120 TStart - s"),
            encoding="utf-8",
        )
        calls: list[Path] = []
        monkeypatch.setattr(floating_module, "run_openfast", _fake_probe_run(calls))
        probe_platform(fst, tmp_path / "work", probe_damping=_PROBE)
        assert len(calls) == 2
        for path in calls:
            assert float(FastInputFile.read(path).get("TStart")) == 0.0
        assert "120 TStart" in fst.read_text(encoding="utf-8")

    def test_non_utf8_hydrodyn_comment_does_not_disable_the_probe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A Windows-1252 byte in HydroDyn is carried through the damped probe copy.

        The damping was written through a strict UTF-8 decode, so the probe failed
        and the settle silently lost its automatic damping.
        """
        fst = _probe_deck(tmp_path)
        hydro = fst.parent / "HD.dat"
        hydro.write_bytes(b"! water at 15 \xb0C\n" + hydro.read_bytes())
        calls: list[Path] = []
        monkeypatch.setattr(floating_module, "run_openfast", _fake_probe_run(calls))
        probe = probe_platform(fst, tmp_path / "work", probe_damping=_PROBE)
        assert probe.effective_mass["surge"] == pytest.approx(_MASS[0], rel=1e-3)
        for path in calls:
            assert b"\xb0C" in (path.parent / "HD.dat").read_bytes()

    def test_transient_lock_on_a_previous_probe_is_retried(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A briefly locked file in a previous probe case does not lose the probe.

        Replacing an earlier probe case meets the same transient Windows sharing
        violations (an antivirus scan, a just-exited solver) as every other case
        directory, and is retried like them.
        """
        import shutil

        from vane.utils import fs_retry

        fst = _probe_deck(tmp_path)
        (tmp_path / "work" / "base").mkdir(parents=True)
        real_rmtree = shutil.rmtree
        failures = [1]

        def flaky_rmtree(path: object, *args: object, **kwargs: object) -> None:
            if failures:
                failures.pop()
                error = PermissionError(13, "sharing violation")
                error.winerror = 32  # type: ignore[attr-defined]
                raise error
            real_rmtree(path, *args, **kwargs)  # type: ignore[call-overload]

        monkeypatch.setattr(shutil, "rmtree", flaky_rmtree)
        monkeypatch.setattr(fs_retry, "_pause", lambda _seconds: None)
        monkeypatch.setattr(floating_module, "run_openfast", _fake_probe_run([]))
        probe = probe_platform(fst, tmp_path / "work", probe_damping=_PROBE)
        assert probe.effective_mass["surge"] == pytest.approx(_MASS[0], rel=1e-3)
        assert not failures

    def test_failed_run_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A probe run that fails is reported with its output."""
        fst = _probe_deck(tmp_path)

        def failing(fst_path: Path, **_: object) -> OpenFastResult:
            return OpenFastResult(fst_path=fst_path, returncode=1, stdout="FATAL")

        monkeypatch.setattr(floating_module, "run_openfast", failing)
        with pytest.raises(PlatformProbeError, match="FATAL"):
            probe_platform(fst, tmp_path / "work")

    def test_matrixless_lin_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A .lin without a state matrix cannot be used."""
        fst = _probe_deck(tmp_path)

        def no_matrix(fst_path: Path, **_: object) -> OpenFastResult:
            lin = fst_path.with_suffix(".1.lin")
            lin.write_text(build_lin_text(states=[]), encoding="utf-8")
            return OpenFastResult(fst_path=fst_path, returncode=0, lin_files=[lin])

        monkeypatch.setattr(floating_module, "run_openfast", no_matrix)
        with pytest.raises(PlatformProbeError, match="no state matrix"):
            probe_platform(fst, tmp_path / "work")

    def test_requires_damping_matrix(self, tmp_path: Path) -> None:
        """A deck without an AddBLin matrix is rejected before any run."""
        fst = _probe_deck(tmp_path)
        (fst.parent / "HD.dat").write_text("1 PotMod - p\n", encoding="utf-8")
        with pytest.raises(ValueError, match="AddBLin"):
            probe_platform(fst, tmp_path / "work")


class TestPlanFloatingSettle:
    """The automatic decisions and their record."""

    def test_missing_library_prescribes_the_rotor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No loadable library: prescribed rotor, damping from the probe."""
        monkeypatch.delenv(CONTROLLER_LIBRARIES_ENV, raising=False)
        monkeypatch.setenv("VANE_CONFIG_DIR", str(tmp_path / "noconfig"))
        fst = _make_deck(tmp_path)
        monkeypatch.setattr(floating_module, "run_openfast", _fake_probe_run([]))
        plan = plan_floating_settle(fst, tmp_path / "work")
        assert plan.prescribed_rotor
        assert plan.controller_library is None
        assert plan.floating
        assert plan.damping_source == "probe"
        assert plan.platform_damping is not None
        assert plan.platform_damping.surge == pytest.approx(
            4 * math.pi * 0.5 * _MASS[0] / _PERIOD[0], rel=1e-3
        )
        record = plan.to_dict()
        assert record["prescribed_rotor"] is True
        assert record["probe"] is not None
        assert any("prescribed" in note for note in plan.notes)

    def test_found_library_and_user_damping(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A found library is used; a given damping skips the probe."""
        fst = _make_deck(tmp_path)
        (fst.parent / "libdiscon.dll").write_bytes(b"MZ")
        calls: list[Path] = []
        monkeypatch.setattr(floating_module, "run_openfast", _fake_probe_run(calls))
        damping = SettlePlatformDamping(surge=1.0)
        plan = plan_floating_settle(fst, tmp_path / "work", platform_damping=damping)
        assert not plan.prescribed_rotor
        assert plan.controller.status == "found"
        assert plan.controller_library == (fst.parent / "libdiscon.dll").resolve()
        assert plan.platform_damping is damping
        assert plan.damping_source == "user"
        assert calls == []

    def test_user_library(self, tmp_path: Path) -> None:
        """A given library is checked and used."""
        fst = _make_deck(tmp_path)
        library = tmp_path / "mine.dll"
        library.write_bytes(b"MZ")
        plan = plan_floating_settle(
            fst, tmp_path / "work", controller_library=library, auto_damping=False
        )
        assert plan.controller.status == "user"
        assert plan.platform_damping is None
        assert plan.damping_source == "none"

    def test_failed_probe_settles_undamped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A probe failure is noted, never fatal."""
        fst = _make_deck(tmp_path, library="DISCON.dll")
        (fst.parent / "servo" / "DISCON.dll").write_bytes(b"MZ")

        def failing(fst_path: Path, **_: object) -> OpenFastResult:
            return OpenFastResult(fst_path=fst_path, returncode=1, stdout="boom")

        monkeypatch.setattr(floating_module, "run_openfast", failing)
        plan = plan_floating_settle(fst, tmp_path / "work")
        assert plan.controller.status == "deck"
        assert plan.platform_damping is None
        assert any("probe failed" in note for note in plan.notes)

    @pytest.mark.parametrize(
        "error",
        [
            OpenFastTimeoutError("OpenFAST run timed out after 60 s"),
            OpenFastError("OpenFAST executable 'openfast' not found on PATH"),
        ],
    )
    def test_probe_run_error_settles_undamped(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        error: OpenFastError,
    ) -> None:
        """A probe run that times out or cannot start is noted, never fatal."""
        fst = _make_deck(tmp_path, library="DISCON.dll")
        (fst.parent / "servo" / "DISCON.dll").write_bytes(b"MZ")

        def raising(fst_path: Path, **_: object) -> OpenFastResult:
            raise error

        monkeypatch.setattr(floating_module, "run_openfast", raising)
        plan = plan_floating_settle(fst, tmp_path / "work")
        assert plan.platform_damping is None
        assert plan.damping_source == "none"
        assert any("probe failed" in note for note in plan.notes)

    def test_fixed_deck_is_not_probed(self, tmp_path: Path) -> None:
        """Only a floating deck gets the damping accelerator."""
        fst = _make_deck(tmp_path, floating=False, servo=False)
        plan = plan_floating_settle(fst, tmp_path / "work")
        assert not plan.floating
        assert plan.controller.status == "not_used"
        assert plan.probe is None

    @pytest.mark.parametrize("fraction", [0.0, -1.0, float("nan")])
    def test_invalid_fraction(self, tmp_path: Path, fraction: float) -> None:
        """The damping fraction must be positive and finite."""
        with pytest.raises(ValueError, match="damping_fraction"):
            plan_floating_settle(
                _make_deck(tmp_path), tmp_path / "work", damping_fraction=fraction
            )

    def test_probe_is_a_dataclass_record(self) -> None:
        """The probe and resolution records serialize their content."""
        probe = PlatformProbe(
            effective_mass={"surge": 1.0},
            natural_period={"surge": 2.0},
            damping_ratio={"surge": 0.1},
            mode_frequency={"surge": 0.5},
            probe_damping=_PROBE,
            n_states=2,
        )
        assert probe.to_dict()["natural_period"] == {"surge": 2.0}
        resolution = ControllerLibraryResolution(status="missing")
        assert resolution.to_dict()["library"] is None


@pytest.mark.real_library_load
class TestLibraryLoadCheck:
    """A library file is loadable only if a child process can load it."""

    def test_corrupt_library_is_not_loadable(self, tmp_path: Path) -> None:
        """A file with the host suffix that is not a library is rejected."""
        bad = tmp_path / f"libdiscon{floating_module._host_library_suffix()}"
        bad.write_bytes(b"MZ but not a library")
        assert not floating_module._loadable(bad)

    def test_python_runtime_library_is_loadable(self) -> None:
        """A real shared library of this host loads (the interpreter runtime)."""
        import sysconfig

        candidates = [
            Path(sys.base_prefix)
            / f"python{sys.version_info[0]}{sys.version_info[1]}.dll",
            Path(sysconfig.get_config_var("LIBDIR") or "")
            / str(sysconfig.get_config_var("LDLIBRARY") or ""),
        ]
        real = next((p for p in candidates if p.is_file()), None)
        if real is None:
            pytest.skip("no interpreter shared library on this host")
        assert steady_state_module._loads_on_host(real)

    def test_outcome_is_cached(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A second check of an unchanged file does not start another process."""
        bad = tmp_path / "cached.dll"
        bad.write_bytes(b"not a library")
        assert not steady_state_module._loads_on_host(bad)
        monkeypatch.setattr(
            steady_state_module.subprocess,
            "run",
            lambda *a, **k: pytest.fail("load check was not cached"),
        )
        assert not steady_state_module._loads_on_host(bad)

    def test_unloadable_explicit_library_fails_fast(self, tmp_path: Path) -> None:
        """An explicit controller_library that cannot load is an error, not a pick.

        It used to be accepted on existence alone, which disabled discovery and
        the prescribed-rotor fallback, so every settle failed inside OpenFAST.
        """
        fst = _make_deck(tmp_path)
        bad = tmp_path / f"broken{floating_module._host_library_suffix()}"
        bad.write_bytes(b"MZ but not a library")
        with pytest.raises(SteadyStateError, match="cannot be loaded on this host"):
            plan_floating_settle(
                fst, tmp_path / "work", controller_library=bad, auto_damping=False
            )

    def test_unloadable_deck_library_fails_fast(self, tmp_path: Path) -> None:
        """The deck's own existing but unloadable library is reported before launch."""
        bad = tmp_path / "template" / "servo" / "broken.dll"
        fst = _make_deck(tmp_path, library=str(bad))
        bad.write_bytes(b"MZ but not a library")
        with pytest.raises(SteadyStateError, match="cannot be loaded on this host"):
            steady_state_module.check_controller_library(fst)
