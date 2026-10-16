# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for labelling BeamDyn-blade modes (vane.modal.labeler on BD nodal states).

The synthetic models prescribe BeamDyn-style nodal mode shapes (translation and
section rotation per finite-element node, in the blade reference frame) for three
identical blades, so the expected flap/edge/torsion family, mode order, and
multi-blade type of every mode is known exactly. The integration tests run on the
OpenFAST r-test BeamDyn linearizations and on a locally produced IEA 22-MW
rotor-only sweep; both skip when the data are absent.
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.linalg import null_space

from vane.config.dof_map import DofCategory
from vane.io import read_lin_file
from vane.io.fst_writer import FastInputFile
from vane.io.mbc_transform import mbc3_transform
from vane.modal.eigensolver import ModalSolution, compute_modes, modes_from_mbc
from vane.modal.labeler import (
    _is_rigid_profile,
    _shape_order,
    label_solution,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from tests.conftest import LinFileFactory

_NODES = tuple(range(2, 10))  # BeamDyn state nodes; node 1 (the root) is no state
_COMPONENTS = (
    ("translational", "X", "m"),
    ("translational", "Y", "m"),
    ("translational", "Z", "m"),
    ("rotational", "X", "rad"),
    ("rotational", "Y", "rad"),
    ("rotational", "Z", "rad"),
)
_SPAN = np.array([(node - 1) / (len(_NODES)) for node in _NODES], dtype=np.float64)


def _node_description(blade: int, node: int, kind: str, axis: str, unit: str) -> str:
    return (
        f"BD_{blade} finite element node {node} (number of elements = 1; element "
        f"order = {len(_NODES)}) {kind} displacement in {axis}, {unit}"
    )


def _blade_descriptions(blade: int) -> tuple[list[str], list[str]]:
    """Return the displacement and velocity state descriptions of one blade."""
    disp, vel = [], []
    for node in _NODES:
        for kind, axis, unit in _COMPONENTS:
            text = _node_description(blade, node, kind, axis, unit)
            disp.append(text)
            vel.append(
                text.replace(
                    f"BD_{blade} finite", f"BD_{blade} First time derivative of finite"
                )
                + "/s"
            )
    return disp, vel


def _rotation(order: int) -> npt.NDArray[np.float64]:
    """Return a section-rotation profile whose increments change sign order-1 times."""
    return np.sin((order - 0.5) * np.pi * _SPAN)


def _mode(family: str, order: int) -> npt.NDArray[np.float64]:
    """Return one blade's nodal shape for a bending/torsion mode or a rigid mode."""
    shape = np.zeros((len(_NODES), len(_COMPONENTS)), dtype=np.float64)
    if family == "rigid_edge":
        shape[:, 3] = 1.0  # rigid rotation about X ...
        shape[:, 1] = -_SPAN  # ... carries the nodes along -Y in proportion to span
        return shape.reshape(-1)
    theta = _rotation(order)
    deflection = np.cumsum(theta) / len(_NODES)
    if family == "flap":
        shape[:, 0], shape[:, 4] = deflection, theta
    elif family == "edge":
        shape[:, 1], shape[:, 3] = deflection, -theta
    else:  # torsion
        shape[:, 5] = theta
    return shape.reshape(-1)


def _blade_matrices(
    modes: list[tuple[str, int, float]],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return one blade's stiffness and damping for prescribed modes (rad/s)."""
    shapes = np.column_stack([_mode(family, order) for family, order, _ in modes])
    filler = null_space(shapes.T)
    basis = np.column_stack((shapes, filler))
    omegas = np.concatenate(
        ([w for *_, w in modes], 300.0 + 10.0 * np.arange(filler.shape[1]))
    )
    inverse = np.linalg.inv(basis)
    stiffness = basis @ np.diag(omegas**2) @ inverse
    damping = basis @ np.diag(2.0 * 0.01 * omegas) @ inverse
    return stiffness, damping


def _rotor_lin_files(
    make_lin_file: LinFileFactory,
    modes: list[tuple[str, int, float]],
    rotor_speed: float,
) -> list[Path]:
    """Write three azimuths of an isotropic three-blade BeamDyn rotor model."""
    stiffness, damping = _blade_matrices(modes)
    n = stiffness.shape[0]
    disp: list[str] = []
    vel: list[str] = []
    for blade in (1, 2, 3):
        d, v = _blade_descriptions(blade)
        disp += d
        vel += v
    k = np.kron(np.eye(3), stiffness)
    c = np.kron(np.eye(3), damping)
    a = np.zeros((6 * n, 6 * n), dtype=np.float64)
    a[: 3 * n, 3 * n :] = np.eye(3 * n)
    a[3 * n :, : 3 * n] = -k
    a[3 * n :, 3 * n :] = -c
    states = [(text, True, 2) for text in disp + vel]
    return [
        make_lin_file(states=states, a=a, rotor_speed=rotor_speed, azimuth=psi)
        for psi in (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0)
    ]


def _single_blade_solution(modes: list[tuple[str, int, float]]) -> ModalSolution:
    """Return the modal solution of one BeamDyn blade (no multi-blade transform)."""
    stiffness, damping = _blade_matrices(modes)
    n = stiffness.shape[0]
    a = np.zeros((2 * n, 2 * n), dtype=np.float64)
    a[:n, n:] = np.eye(n)
    a[n:, :n] = -stiffness
    a[n:, n:] = -damping
    disp, vel = _blade_descriptions(1)
    return compute_modes(a, ndof2=n, ndof1=0, descriptions=disp + vel)


_ROTOR_MODES = [
    ("rigid_edge", 1, 1.5),
    ("flap", 1, 4.0),
    ("edge", 1, 6.0),
    ("flap", 2, 11.0),
    ("edge", 2, 17.0),
    ("flap", 3, 23.0),
    ("torsion", 1, 29.0),
]


class TestShapeOrder:
    """Unit tests for the spanwise shape-order and rigidity helpers."""

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_order_from_rotation_increments(self, order: int) -> None:
        """The n-th bending profile has n-1 sign changes in its increments."""
        assert _shape_order(_rotation(order)) == order

    def test_order_ignores_a_rigid_offset(self) -> None:
        """A rigid root rotation added to every node does not change the order."""
        assert _shape_order(_rotation(2) + 5.0) == 2

    def test_constant_profile_is_first_order_and_rigid(self) -> None:
        """A constant profile has no increments: order 1, and it is rigid."""
        profile = np.full(6, 0.3)
        assert _shape_order(profile) == 1
        assert _is_rigid_profile(profile)

    def test_elastic_profile_is_not_rigid(self) -> None:
        """A cantilever-like rotation that grows from the root is elastic."""
        assert not _is_rigid_profile(_rotation(1))
        assert not _is_rigid_profile(np.zeros(4))


class TestBeamDynRotorLabels:
    """Labels of a synthetic, MBC-transformed three-blade BeamDyn rotor."""

    @pytest.fixture
    def labels(self, make_lin_file: LinFileFactory) -> list[tuple[float, str]]:
        """Return ``(frequency, label)`` of every mode of the synthetic rotor."""
        files = _rotor_lin_files(make_lin_file, _ROTOR_MODES, rotor_speed=0.3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            solution = modes_from_mbc(mbc3_transform([read_lin_file(f) for f in files]))
        result = label_solution(solution)
        return [
            (float(f), label.label)
            for f, label in zip(solution.natural_frequencies_hz, result, strict=True)
            if f < 10.0
        ]

    def test_every_bending_and_torsion_line_is_named(
        self, labels: list[tuple[float, str]]
    ) -> None:
        """Each family, order, and whirl type appears exactly once."""
        names = [name for _, name in labels]
        for family in (
            "1st Blade Flap",
            "2nd Blade Flap",
            "3rd Blade Flap",
            "1st Blade Edge",
            "2nd Blade Edge",
            "1st Blade Torsion",
        ):
            for whirl in ("Regressive", "Collective", "Progressive"):
                assert names.count(f"{family} ({whirl})") == 1, (family, whirl, names)
        assert not any(name.startswith("Unidentified") for name in names)

    def test_numbered_modes_sit_at_their_prescribed_frequency(
        self, labels: list[tuple[float, str]]
    ) -> None:
        """A collective line is at its blade frequency; whirls are split by Omega."""
        by_name = {name: f for f, name in labels}
        omega_hz = 0.3 / (2.0 * np.pi)
        for name, w in (
            ("1st Blade Flap", 4.0),
            ("2nd Blade Edge", 17.0),
            ("1st Blade Torsion", 29.0),
        ):
            f0 = w / (2.0 * np.pi)
            assert by_name[f"{name} (Collective)"] == pytest.approx(f0, rel=1e-3)
            assert by_name[f"{name} (Regressive)"] == pytest.approx(
                f0 - omega_hz, rel=1e-2
            )
            assert by_name[f"{name} (Progressive)"] == pytest.approx(
                f0 + omega_hz, rel=1e-2
            )

    def test_rigid_collective_edge_is_the_rotor_mode(
        self, labels: list[tuple[float, str]]
    ) -> None:
        """Rigid blade rotation is not numbered; the collective one is the rotor."""
        names = [name for _, name in labels]
        assert names.count("Rotor/Generator") == 1
        assert names.count("Blade Edge (Regressive)") == 1
        assert names.count("Blade Edge (Progressive)") == 1


class TestSingleBladeLabels:
    """Without MBC, BeamDyn modes are numbered from their spanwise shape."""

    def test_categories_follow_the_blade_frame_axes(self) -> None:
        """Flap, edge, and torsion modes get the BeamDyn family categories."""
        solution = _single_blade_solution(
            [("flap", 1, 4.0), ("edge", 1, 6.0), ("torsion", 1, 29.0)]
        )
        labels = label_solution(solution)[:3]
        assert [label.category for label in labels] == [
            DofCategory.BLADE_FLAP,
            DofCategory.BLADE_EDGE,
            DofCategory.BLADE_TORSION,
        ]
        assert [label.label for label in labels] == [
            "1st Blade Flap",
            "1st Blade Edge",
            "1st Blade Torsion",
        ]
        assert all(label.confidence > 0.99 for label in labels)

    def test_shape_sets_the_order_when_a_lower_mode_is_missing(self) -> None:
        """The lowest flap mode in the model is still '2nd' when its shape is."""
        solution = _single_blade_solution([("flap", 2, 11.0), ("flap", 3, 23.0)])
        labels = label_solution(solution)[:2]
        assert [label.label for label in labels] == ["2nd Blade Flap", "3rd Blade Flap"]


# --------------------------------------------------------------------------------------
# Integration tests on real BeamDyn linearizations.
# --------------------------------------------------------------------------------------


def _labelled(files: list[Path]) -> list[tuple[float, str]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solution = modes_from_mbc(mbc3_transform([read_lin_file(f) for f in files]))
    labels = label_solution(solution)
    return [
        (float(f), label.label)
        for f, label in zip(solution.natural_frequencies_hz, labels, strict=True)
    ]


def _nearest(labels: list[tuple[float, str]], name: str) -> float:
    matches = [f for f, label in labels if label == name]
    assert len(matches) == 1, (name, matches)
    return matches[0]


def test_rtest_beamdyn_land_labels(rtest_dir: Path) -> None:
    """The NREL 5-MW BeamDyn r-test linearization gets physical blade labels."""
    case = rtest_dir / "glue-codes" / "openfast" / "5MW_Land_BD_Linear"
    lin = case / "5MW_Land_BD_Linear.1.lin"
    if not lin.is_file():
        pytest.skip(f"r-test BeamDyn linearization not found: {lin}")
    labels = _labelled([lin])
    expected = {
        # Single azimuth at 12.1 rpm: the whirl lines sit at about f0 -/+ 0.2 Hz.
        "1st Blade Flap (Regressive)": 0.537,
        "1st Blade Flap (Collective)": 0.768,
        "1st Blade Flap (Progressive)": 0.936,
        "1st Blade Edge (Regressive)": 0.921,
        "1st Blade Edge (Progressive)": 1.318,
        "2nd Blade Flap (Regressive)": 1.827,
        "2nd Blade Flap (Collective)": 2.116,
        "2nd Blade Flap (Progressive)": 2.215,
        "2nd Blade Edge (Regressive)": 3.605,
        "2nd Blade Edge (Progressive)": 3.993,
        "1st Blade Torsion (Collective)": 5.657,
        "1st Tower Fore-Aft": 0.326,
        "1st Tower Side-Side": 0.309,
    }
    for name, frequency in expected.items():
        assert _nearest(labels, name) == pytest.approx(frequency, abs=0.01), name
    assert not any(
        label.startswith("Unidentified") for f, label in labels if 0.2 < f < 6.0
    )


def _reference_runs_dir() -> Path:
    candidates = []
    env = os.environ.get("VANE_REFERENCE_RUNS_DIR")
    if env:
        candidates.append(Path(env))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    pytest.skip("reference-turbine runs unavailable; set VANE_REFERENCE_RUNS_DIR")


def _uses_beamdyn(fst: Path) -> bool:
    """Return whether ``fst`` exists and models the blades with BeamDyn."""
    if not fst.is_file():
        return False
    deck = FastInputFile.read(fst)
    return "CompElast" in deck and deck.get("CompElast") == "2"


def test_iea22_rotor_only_matches_published_labels() -> None:
    """IEA 22-MW (BeamDyn) labels at 6 m/s match the published Campbell lines."""
    stem = "IEA-22-280-RWT-Monopile"
    runs = _reference_runs_dir()
    case = next(
        (
            candidate
            for name in ("iea22_rotoronly_beamdyn", "iea22_rotoronly")
            if _uses_beamdyn(
                candidate := runs / name / "cases" / "ws6" / stem / f"{stem}.fst"
            )
        ),
        None,
    )
    if case is None:
        pytest.skip("no BeamDyn IEA 22-MW rotor-only sweep in the reference runs")
    case = case.parent
    pattern = re.compile(rf"{re.escape(stem)}\.(\d+)\.lin")
    files = sorted(
        (p for p in case.glob(f"{stem}.*.lin") if pattern.fullmatch(p.name)),
        key=lambda p: int(pattern.fullmatch(p.name).group(1)),  # type: ignore[union-attr]
    )
    if len(files) != 36:
        pytest.skip(f"incomplete IEA 22-MW azimuth sweep in {case}")
    labels = _labelled(files)
    # Published rotor-only stability results at 6 m/s (OpenFAST, unsteady aero).
    published = {
        "1st Blade Edge (Regressive)": 0.483,
        "1st Blade Edge (Progressive)": 0.606,
        "1st Blade Edge (Collective)": 1.076,
        "2nd Blade Flap (Regressive)": 1.077,
        "2nd Blade Flap (Collective)": 1.136,
        "2nd Blade Flap (Progressive)": 1.198,
        "2nd Blade Edge (Regressive)": 1.514,
        "2nd Blade Edge (Progressive)": 1.637,
        "3rd Blade Flap (Regressive)": 2.255,
        "3rd Blade Flap (Collective)": 2.280,
        "3rd Blade Flap (Progressive)": 2.377,
    }
    for name, frequency in published.items():
        assert _nearest(labels, name) == pytest.approx(frequency, rel=0.02), name
    for whirl in ("Regressive", "Collective", "Progressive"):
        assert _nearest(labels, f"1st Blade Flap ({whirl})") < 0.75
