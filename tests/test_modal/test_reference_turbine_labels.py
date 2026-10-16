# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Integration tests: mode labels of real full-system reference-turbine sweeps.

These run on locally produced CalcSteady linearizations (36 azimuths per operating
point) of the NREL 5-MW turbine, land-based and on the OC3 monopile. The data are
never committed: the location is taken from ``VANE_REFERENCE_RUNS_DIR``, and the
tests skip when it is not set.
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from vane.io import read_lin_file
from vane.io.mbc_transform import mbc3_transform
from vane.modal.eigensolver import ModalSolution, modes_from_mbc
from vane.modal.identifier import IdentificationResult, identify_modes
from vane.modal.labeler import label_solution

_CASES = {
    "nrel5mw_land": "5MW_Land_DLL_WTurb",
    "nrel5mw_oc3monopile": "5MW_OC3Mnpl_DLL_WTurb_WavesIrr",
}
# 6 m/s is the point where the rotor-azimuth pair comes back as a slow complex pair.
_WIND_SPEEDS = (4, 6, 16)


def _runs_dir() -> Path:
    """Return the reference-turbine linearization folder, or skip when unavailable."""
    candidates = []
    env = os.environ.get("VANE_REFERENCE_RUNS_DIR")
    if env:
        candidates.append(Path(env))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    pytest.skip("reference-turbine runs unavailable; set VANE_REFERENCE_RUNS_DIR")


def _point(case_dir: Path, stem: str) -> ModalSolution:
    """Return the MBC modal solution of one operating point's azimuth sweep."""
    pattern = re.compile(rf"{re.escape(stem)}\.(\d+)\.lin")
    files = sorted(
        (p for p in case_dir.glob(f"{stem}.*.lin") if pattern.fullmatch(p.name)),
        key=lambda p: int(pattern.fullmatch(p.name).group(1)),  # type: ignore[union-attr]
    )
    if len(files) < 3:
        pytest.skip(f"incomplete azimuth sweep in {case_dir}")
    return modes_from_mbc(mbc3_transform([read_lin_file(p) for p in files]))


@pytest.fixture(scope="module", params=sorted(_CASES))
def sweep(request: pytest.FixtureRequest) -> tuple[str, list[ModalSolution]]:
    """Return a case key and its modal solutions at the test wind speeds."""
    key: str = request.param
    root = _runs_dir() / key / "cases"
    solutions = []
    for wind in _WIND_SPEEDS:
        case_dir = root / f"ws{wind}" / _CASES[key]
        if not case_dir.is_dir():
            pytest.skip(f"reference-turbine case not found: {case_dir}")
        solutions.append(_point(case_dir, _CASES[key]))
    return key, solutions


def _tracks(solutions: list[ModalSolution]) -> IdentificationResult:
    return identify_modes(solutions)


def _labels_below(result: IdentificationResult, f_max: float) -> list[str]:
    return [
        track.label.label
        for track in result.tracks
        if float(np.mean(track.natural_frequencies_hz)) < f_max
    ]


class TestReferenceTurbineLabels:
    """Labels of the real NREL 5-MW land and OC3 monopile Campbell lines."""

    def test_rotor_tracks_carry_their_multiblade_type(
        self, sweep: tuple[str, list[ModalSolution]]
    ) -> None:
        """Each first-flap and first-edge whirl line is named with its type."""
        _, solutions = sweep
        labels = _labels_below(_tracks(solutions), 1.5)
        for name in (
            "1st Blade Flap (Regressive)",
            "1st Blade Flap (Collective)",
            "1st Blade Flap (Progressive)",
            "1st Blade Edge (Regressive)",
            "1st Blade Edge (Progressive)",
        ):
            assert labels.count(name) == 1, (name, labels)
        assert not any(
            label in ("1st Blade Flap", "1st Blade Edge") for label in labels
        )

    def test_whirl_order_follows_the_coleman_split(
        self, sweep: tuple[str, list[ModalSolution]]
    ) -> None:
        """Regressive < collective < progressive for the first flap at every point."""
        _, solutions = sweep
        lines = {
            track.label.multiblade: track.natural_frequencies_hz
            for track in _tracks(solutions).tracks
            if track.label.label.startswith("1st Blade Flap")
        }
        regressive = np.asarray(lines["regressive"])
        collective = np.asarray(lines["collective"])
        progressive = np.asarray(lines["progressive"])
        assert bool(np.all(regressive < collective))
        assert bool(np.all(collective < progressive))

    def test_second_tower_modes_are_tower_modes(
        self, sweep: tuple[str, list[ModalSolution]]
    ) -> None:
        """The 2nd tower bending lines are tower modes, never platform modes.

        Land: about 2.9 Hz (TP-500-38060 Table 9-1: 2.900 / 2.936 Hz). Monopile:
        about 2.4 / 2.5 Hz, where platform translation carries the monopile's
        flexibility and co-participates with the tower bending DOFs.
        """
        key, solutions = sweep
        result = _tracks(solutions)
        assert not any(
            label.startswith("Platform") for label in _labels_below(result, 3.5)
        )
        expected = (2.8, 3.0) if key == "nrel5mw_land" else (2.3, 2.6)
        for name in ("2nd Tower Fore-Aft", "2nd Tower Side-Side"):
            matches = [t for t in result.tracks if t.label.label == name]
            assert len(matches) == 1, name
            mean = float(np.mean(matches[0].natural_frequencies_hz))
            assert expected[0] < mean < expected[1], (name, mean)

    def test_no_rotor_azimuth_track(
        self, sweep: tuple[str, list[ModalSolution]]
    ) -> None:
        """The free rotor-azimuth (rigid-body) mode forms no Campbell line."""
        _, solutions = sweep
        result = _tracks(solutions)
        assert all(track.label.label != "Rotor/Generator" for track in result.tracks)
        assert min(t.natural_frequencies_hz[0] for t in result.tracks) > 0.2


def test_floating_rigid_body_modes_stay_platform(rtest_dir: Path) -> None:
    """A floating spar's six rigid-body modes keep their platform labels."""
    path = (
        rtest_dir
        / "glue-codes"
        / "openfast"
        / "5MW_OC3Spar_Linear"
        / "5MW_OC3Spar_Linear.1.lin"
    )
    if not path.is_file():
        pytest.skip(f"case not found: {path}")
    with warnings.catch_warnings():
        # A single azimuth of a rotating point warns; the platform modes are
        # unaffected by the rotor average.
        warnings.simplefilter("ignore")
        solution = modes_from_mbc(mbc3_transform([read_lin_file(path)]))
    labels = label_solution(solution)
    rigid = {
        label.label
        for label, frequency in zip(
            labels, solution.natural_frequencies_hz, strict=True
        )
        if frequency < 0.2
    }
    assert {
        "Platform Surge",
        "Platform Sway",
        "Platform Heave",
        "Platform Roll",
        "Platform Pitch",
        "Platform Yaw",
    } <= rigid
