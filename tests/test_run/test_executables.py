# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.run.executables -- deck format vs OpenFAST release."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from vane.run import OpenFastError
from vane.run import executables as module
from vane.run.executables import (
    OPENFAST_CONFIG_ENV,
    OPENFAST_EXECUTABLES_ENV,
    DeckGeneration,
    candidate_executables,
    detect_deck_generation,
    openfast_version,
    select_openfast_executable,
)

_V4_PRIMARY = """\
0.01 DT - step
1 CompElast - structure
0 CompSeaSt - sea state
{hydro_switch} CompHydro - hydro
"ED.dat" EDFile - ElastoDyn
"HD.dat" HydroFile - HydroDyn
"""


def _deck(
    tmp_path: Path,
    *,
    primary: str = _V4_PRIMARY,
    elastodyn: str = "63 TipRad - radius\n",
    hydrodyn: str | None = None,
) -> Path:
    root = tmp_path / "deck"
    root.mkdir(exist_ok=True)
    (root / "ED.dat").write_text(elastodyn, encoding="utf-8")
    if hydrodyn is not None:
        (root / "HD.dat").write_text(hydrodyn, encoding="utf-8")
    fst = root / "Model.fst"
    fst.write_text(
        primary.format(hydro_switch=1 if hydrodyn is not None else 0),
        encoding="utf-8",
    )
    return fst


class TestDetectDeckGeneration:
    """Keyword signatures of the OpenFAST input-format generations."""

    def test_v5_primary_keywords(self, tmp_path: Path) -> None:
        """v5 added the tight-coupling lines to the primary file."""
        primary = "3 ModCoupling - coupling\n1e-4 ConvTol - tol\n" + _V4_PRIMARY
        generation = detect_deck_generation(_deck(tmp_path, primary=primary))
        assert generation.label == "v5"
        assert generation.accepts((5, 0, 0))
        assert not generation.accepts((4, 2, 1))
        assert not generation.accepts((6, 0, 0))
        assert "ConvTol" in generation.evidence[0]

    def test_v5_elastodyn_keywords(self, tmp_path: Path) -> None:
        """ElastoDyn's PitchDOF alone marks a v5 deck."""
        generation = detect_deck_generation(
            _deck(tmp_path, elastodyn="False PitchDOF - pitch DOF\n")
        )
        assert generation.label == "v5"

    def test_v42_hub_teeter_inertia(self, tmp_path: Path) -> None:
        """v4.2 added ElastoDyn HubIner_Teeter."""
        generation = detect_deck_generation(
            _deck(tmp_path, elastodyn="0 HubIner_Teeter - teeter\n")
        )
        assert generation.label == "v4.2"
        assert generation.accepts((4, 2, 1))
        assert not generation.accepts((4, 0, 5))
        assert not generation.accepts((5, 0, 0))
        assert generation.describe() == ">=4.2.0, <5.0.0"

    def test_v41_hydrodyn_cylinder_tables(self, tmp_path: Path) -> None:
        """v4.1 split HydroDyn's member tables (NPropSetsCyl)."""
        generation = detect_deck_generation(
            _deck(tmp_path, hydrodyn="1 NPropSetsCyl - sets\n0 NPropSetsRec - s\n")
        )
        assert generation.label == "v4.1"
        assert generation.accepts((4, 1, 2))
        assert not generation.accepts((4, 2, 0))

    def test_v40_hydrodyn_single_table(self, tmp_path: Path) -> None:
        """v4.0 decks have HydroDyn NPropSets and no HubIner_Teeter."""
        generation = detect_deck_generation(
            _deck(tmp_path, hydrodyn="1 NPropSets - sets\n")
        )
        assert generation.label == "v4.0"
        assert generation.accepts((4, 0, 5))
        assert not generation.accepts((4, 1, 0))

    def test_land_v4_without_teeter(self, tmp_path: Path) -> None:
        """A land v4 deck without HubIner_Teeter reads on v4.0 and v4.1."""
        generation = detect_deck_generation(_deck(tmp_path))
        assert generation.label == "v4.0-v4.1"
        assert generation.accepts((4, 0, 5))
        assert generation.accepts((4, 1, 2))
        assert not generation.accepts((4, 2, 1))

    def test_disabled_hydrodyn_is_not_read(self, tmp_path: Path) -> None:
        """A switched-off HydroDyn file is inert, so it is not evidence."""
        fst = _deck(tmp_path, hydrodyn="1 NPropSets - sets\n")
        fst.write_text(
            fst.read_text(encoding="utf-8").replace("1 CompHydro", "0 CompHydro"),
            encoding="utf-8",
        )
        assert detect_deck_generation(fst).label == "v4.0-v4.1"

    def test_v3_has_no_seastate(self, tmp_path: Path) -> None:
        """Before v4.0 the primary file had no CompSeaSt line."""
        primary = _V4_PRIMARY.replace("0 CompSeaSt - sea state\n", "")
        generation = detect_deck_generation(_deck(tmp_path, primary=primary))
        assert generation.label == "v3"
        assert generation.accepts((3, 5, 5))
        assert not generation.accepts((4, 0, 0))

    def test_partial_deck_is_unknown(self, tmp_path: Path) -> None:
        """A deck without format signals accepts any release."""
        generation = detect_deck_generation(
            _deck(tmp_path, primary='"ED.dat" EDFile - ElastoDyn\n')
        )
        assert generation.label == "unknown"
        assert generation.max_version is None
        assert generation.accepts((1, 0, 0))
        assert generation.describe() == ">=0.0.0"
        assert generation.to_dict()["requires"] == ">=0.0.0"


def _fake_probe(
    monkeypatch: pytest.MonkeyPatch, versions: dict[str, str | None]
) -> list[str]:
    """Make ``<exe> -v`` print the banner of ``versions[exe name]``."""
    calls: list[str] = []

    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        name = Path(command[0]).name
        calls.append(name)
        version = versions.get(name)
        banner = "no banner" if version is None else f"OpenFAST-v{version}\n"
        return subprocess.CompletedProcess(command, 0, stdout=banner, stderr="")

    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module, "_version_cache", {})
    return calls


def _executables(tmp_path: Path, *names: str) -> list[Path]:
    paths = []
    for name in names:
        path = tmp_path / "bin" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"MZ")
        paths.append(path)
    return paths


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No PATH executables, no environment list, no configuration file."""
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.delenv(OPENFAST_EXECUTABLES_ENV, raising=False)
    monkeypatch.setenv(OPENFAST_CONFIG_ENV, str(tmp_path / "no-config.txt"))


class TestOpenFastVersion:
    """Reading and caching the version banner."""

    def test_banner_is_parsed_and_cached(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One probe per executable file; a dev suffix is ignored."""
        (exe,) = _executables(tmp_path, "openfast.exe")
        calls = _fake_probe(monkeypatch, {"openfast.exe": "4.2.1-12-gabcdef"})
        assert openfast_version(exe) == (4, 2, 1)
        assert openfast_version(exe) == (4, 2, 1)
        assert calls == ["openfast.exe"]

    def test_no_banner_or_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No banner or no executable: the version is unknown."""
        (exe,) = _executables(tmp_path, "wrapper.exe")
        _fake_probe(monkeypatch, {"wrapper.exe": None})
        assert openfast_version(exe) is None
        assert openfast_version(tmp_path / "absent.exe") is None

    def test_probe_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An executable that cannot be run has no version."""
        (exe,) = _executables(tmp_path, "broken.exe")

        def run(command: list[str], **_: object) -> None:
            msg = "not a valid application"
            raise OSError(msg)

        monkeypatch.setattr(module.subprocess, "run", run)
        monkeypatch.setattr(module, "_version_cache", {})
        assert openfast_version(exe) is None


@pytest.mark.usefixtures("isolated_env")
class TestSelectExecutable:
    """Choosing an executable that reads the deck's format."""

    def test_environment_list_picks_the_compatible_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A v4.2 deck gets the v4.2.1 binary, not the v5 one listed first."""
        v5, v42, v40 = _executables(tmp_path, "of5.exe", "of42.exe", "of40.exe")
        _fake_probe(
            monkeypatch, {"of5.exe": "5.0.0", "of42.exe": "4.2.1", "of40.exe": "4.0.5"}
        )
        monkeypatch.setenv(
            OPENFAST_EXECUTABLES_ENV, os.pathsep.join(str(p) for p in (v5, v42, v40))
        )
        fst = _deck(tmp_path, elastodyn="0 HubIner_Teeter - teeter\n")
        selection = select_openfast_executable(fst)
        assert selection.executable == v42.resolve()
        assert selection.version == (4, 2, 1)
        assert selection.source == OPENFAST_EXECUTABLES_ENV
        record = selection.to_dict()
        assert record["version"] == "4.2.1"
        assert len(record["considered"]) == 2  # stopped at the first match

    def test_configuration_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Executables listed in the configuration file are considered."""
        (v40,) = _executables(tmp_path, "of40.exe")
        _fake_probe(monkeypatch, {"of40.exe": "4.0.5"})
        config = tmp_path / "openfast.txt"
        config.write_text(f'# OpenFAST builds\n"{v40}"  # v4.0\n\n', encoding="utf-8")
        monkeypatch.setenv(OPENFAST_CONFIG_ENV, str(config))
        selection = select_openfast_executable(
            _deck(tmp_path, hydrodyn="1 NPropSets - s\n")
        )
        assert selection.executable == v40.resolve()
        assert selection.source == str(config)

    def test_explicit_compatible_or_unknown_is_used(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit executable is used when compatible or of unknown version."""
        v5, wrapper = _executables(tmp_path, "of5.exe", "wrap.exe")
        _fake_probe(monkeypatch, {"of5.exe": "5.0.0", "wrap.exe": None})
        primary = "3 ModCoupling - coupling\n" + _V4_PRIMARY
        fst = _deck(tmp_path, primary=primary)
        assert select_openfast_executable(fst, v5).source == "argument"
        selection = select_openfast_executable(fst, wrapper)
        assert selection.executable == wrapper.resolve()
        assert selection.version is None

    def test_explicit_incompatible_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit v5 executable is not silently swapped for a v4 deck."""
        v5, v42 = _executables(tmp_path, "of5.exe", "of42.exe")
        _fake_probe(monkeypatch, {"of5.exe": "5.0.0", "of42.exe": "4.2.1"})
        monkeypatch.setenv(OPENFAST_EXECUTABLES_ENV, str(v42))
        fst = _deck(tmp_path, elastodyn="0 HubIner_Teeter - teeter\n")
        with pytest.raises(OpenFastError, match=r">=4\.2\.0, <5\.0\.0") as info:
            select_openfast_executable(fst, v5)
        assert "HubIner_Teeter" in str(info.value)
        assert "5.0.0" in str(info.value)

    def test_explicit_missing_is_rejected(self, tmp_path: Path) -> None:
        """A named executable that does not exist fails at once."""
        with pytest.raises(OpenFastError, match="not found"):
            select_openfast_executable(_deck(tmp_path), tmp_path / "absent.exe")

    def test_none_compatible_names_the_needed_release(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only a v5 binary for a v4.0 deck: fail before launch with the range."""
        (v5,) = _executables(tmp_path, "of5.exe")
        _fake_probe(monkeypatch, {"of5.exe": "5.0.0"})
        monkeypatch.setenv(
            OPENFAST_EXECUTABLES_ENV, f"{v5}{os.pathsep}{tmp_path / 'gone.exe'}"
        )
        fst = _deck(tmp_path, hydrodyn="1 NPropSets - s\n")
        with pytest.raises(OpenFastError, match=r"needs OpenFAST >=4\.0\.0, <4\.1\.0"):
            select_openfast_executable(fst)

    def test_nothing_found(self, tmp_path: Path) -> None:
        """No executable anywhere: the error says where to list one."""
        with pytest.raises(OpenFastError, match=OPENFAST_EXECUTABLES_ENV):
            select_openfast_executable(_deck(tmp_path))

    def test_unknown_version_is_the_last_resort(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a known compatible one, an unreadable version is tried."""
        v5, wrapper = _executables(tmp_path, "of5.exe", "wrap.exe")
        _fake_probe(monkeypatch, {"of5.exe": "5.0.0", "wrap.exe": None})
        monkeypatch.setenv(OPENFAST_EXECUTABLES_ENV, f"{v5}{os.pathsep}{wrapper}")
        selection = select_openfast_executable(_deck(tmp_path))
        assert selection.executable == wrapper.resolve()

    def test_candidates_are_ordered_and_deduplicated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Argument first, then the list; the same file only once."""
        (exe,) = _executables(tmp_path, "of.exe")
        monkeypatch.setenv(OPENFAST_EXECUTABLES_ENV, f"{exe}{os.pathsep}{exe}")
        assert candidate_executables(exe) == [(str(exe), "argument")]
        assert candidate_executables() == [(str(exe), OPENFAST_EXECUTABLES_ENV)]

    def test_home_directory_is_expanded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``~`` in the configuration file, its variable and the list is the home."""
        (exe,) = _executables(tmp_path, "of.exe")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        (tmp_path / "of.txt").write_text("~/bin/of.exe\n", encoding="utf-8")
        monkeypatch.setenv(OPENFAST_CONFIG_ENV, "~/of.txt")
        expected = os.path.normcase(str(exe.resolve()))
        candidates = candidate_executables()
        assert len(candidates) == 1
        assert os.path.normcase(str(Path(candidates[0][0]).resolve())) == expected
        monkeypatch.setenv(OPENFAST_CONFIG_ENV, str(tmp_path / "no-config.txt"))
        monkeypatch.setenv(OPENFAST_EXECUTABLES_ENV, "~/bin/of.exe")
        candidates = candidate_executables()
        assert candidates[0][1] == OPENFAST_EXECUTABLES_ENV
        assert os.path.normcase(str(Path(candidates[0][0]).resolve())) == expected

    def test_path_lookup(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """An openfast executable on PATH is a candidate."""
        name = "openfast.exe" if os.name == "nt" else "openfast"
        (exe,) = _executables(tmp_path, name)
        exe.chmod(0o755)
        monkeypatch.setenv("PATH", str(exe.parent))
        assert ("openfast", "PATH") in candidate_executables()


def test_generation_record_round_trip() -> None:
    """The generation's record names its release range and evidence."""
    generation = DeckGeneration("v4.2", (4, 2, 0), (5, 0, 0), ("HubIner_Teeter",))
    assert generation.to_dict() == {
        "label": "v4.2",
        "requires": ">=4.2.0, <5.0.0",
        "evidence": ["HubIner_Teeter"],
    }


def _reference_root() -> Path:
    runs = os.environ.get("VANE_REFERENCE_RUNS_DIR")
    if not runs:
        pytest.skip("reference-turbine data unavailable; set VANE_REFERENCE_RUNS_DIR")
    root = Path(runs).parent
    if not (root / "IEA-15-240-RWT").is_dir():
        pytest.skip("reference-turbine decks unavailable; set VANE_REFERENCE_RUNS_DIR")
    return root


@pytest.mark.parametrize(
    ("deck", "label"),
    [
        ("IEA-3.4-130-RWT/openfast/IEA-3.4-130-RWT.fst", "v4.0-v4.1"),
        ("IEA-10.0-198-RWT/openfast/IEA-10.0-198-RWT.fst", "v4.0"),
        (
            "IEA-15-240-RWT/OpenFAST/IEA-15-240-RWT-Monopile/"
            "IEA-15-240-RWT-Monopile.fst",
            "v4.2",
        ),
    ],
)
def test_reference_decks(deck: str, label: str) -> None:
    """The IEA reference decks are recognised as the release they were written for."""
    assert detect_deck_generation(_reference_root() / deck).label == label


def test_rtest_deck_is_v5(rtest_dir: Path) -> None:
    """The r-test decks are in the v5 format."""
    deck = rtest_dir / "glue-codes/openfast/5MW_Land_DLL_WTurb/5MW_Land_DLL_WTurb.fst"
    assert detect_deck_generation(deck).label == "v5"
