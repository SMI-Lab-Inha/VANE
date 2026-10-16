# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Shared pytest fixtures for the VANE test suite.

Synthetic linearization data and other fixtures shared across modules are
defined here. Real turbine ``.lin`` files must never be committed to the
repository; the :func:`rtest_dir` fixture instead points at a local clone of
the OpenFAST ``r-test`` repository named by ``VANE_RTEST_DIR`` for integration
tests, and skips when it is not set.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import numpy.typing as npt

# Channel specification: (description, rotating_frame, derivative_order).
ChannelSpec = tuple[str, bool, int]


def build_lin_text(
    *,
    states: Sequence[ChannelSpec],
    inputs: Sequence[ChannelSpec] = (),
    outputs: Sequence[ChannelSpec] = (),
    a: npt.NDArray[np.float64] | None = None,
    b: npt.NDArray[np.float64] | None = None,
    c: npt.NDArray[np.float64] | None = None,
    d: npt.NDArray[np.float64] | None = None,
    rotor_speed: float = 0.0,
    azimuth: float = 0.0,
    wind_speed: float = 0.0,
    sim_time: float = 0.0,
    state_values: Sequence[float] | None = None,
    input_values: Sequence[float] | None = None,
    output_values: Sequence[float] | None = None,
) -> str:
    """Build the text of a synthetic *modern* (ModGlue) OpenFAST ``.lin`` file.

    Parameters
    ----------
    states, inputs, outputs : Sequence[ChannelSpec]
        Channel specifications for the continuous states, inputs, and outputs.
        Each is a ``(description, rotating_frame, derivative_order)`` tuple. A
        matching continuous-state-derivative table is generated automatically
        from ``states``.
    a, b, c, d : npt.NDArray[np.float64] or None
        Optional state-space matrices. Omitted matrices are not written, mirroring
        OpenFAST's behaviour for empty blocks.
    rotor_speed : float, optional
        Rotor speed in rad/s written to the header.
    azimuth : float, optional
        Azimuth in rad written to the header.
    wind_speed : float, optional
        Wind speed in m/s written to the header.
    sim_time : float, optional
        Simulation time in seconds written to the header.
    state_values, input_values, output_values : Sequence[float] or None, optional
        Operating-point table values. ``None`` writes zero for every channel.

    Returns
    -------
    str
        The complete file contents, ready to be written to disk.
    """
    n_x, n_u, n_y = len(states), len(inputs), len(outputs)
    jac = "No"
    lines = [
        "",
        "Linearized model: synthetic test fixture written by the VANE tests",
        " linked with  NWTC Subroutine Library; ElastoDyn",
        "",
        "Description from the FAST input file: VANE synthetic linearization fixture",
        "",
        "Simulation information:",
        f"   Simulation time:                {sim_time: .4f} s",
        f"   Rotor Speed:                    {rotor_speed: .4f} rad/s",
        f"   Azimuth:                        {azimuth: .4f} rad",
        f"   Wind Speed:                     {wind_speed: .4f} m/s",
        f"   Number of continuous states:    {n_x:5d}",
        "   Number of discrete states:           0",
        "   Number of constraint states:         0",
        f"   Number of inputs:               {n_u:5d}",
        f"   Number of outputs:              {n_y:5d}",
        f"   Jacobians included in this file?    {jac}",
        "",
    ]

    deriv = [
        (f"First time derivative of {desc}", rot, max(order - 1, 1))
        for desc, rot, order in states
    ]
    lines += _table_lines("Order of continuous states:", states, state_values)
    lines += _table_lines("Order of continuous state derivatives:", deriv)
    lines += _table_lines("Order of inputs:", inputs, input_values)
    lines += _table_lines("Order of outputs:", outputs, output_values)

    lines.append("Linearized state matrices:")
    lines.append("")
    for name, matrix in (("A", a), ("B", b), ("C", c), ("D", d)):
        if matrix is not None:
            lines += _matrix_lines(name, matrix)
    return "\n".join(lines) + "\n"


def _table_lines(
    label: str,
    channels: Sequence[ChannelSpec],
    values: Sequence[float] | None = None,
) -> list[str]:
    """Render one operating-point table (header, separator, rows, blank line)."""
    out = [
        label,
        "   Row/Column Operating Point         "
        "Rotating Frame? Derivative Order Description",
        "   ---------- ---------------         "
        "--------------- ---------------- -----------",
    ]
    operating = [0.0] * len(channels) if values is None else list(values)
    if len(operating) != len(channels):
        msg = (
            f"operating-point values have length {len(operating)}, "
            f"expected {len(channels)}"
        )
        raise ValueError(msg)
    for i, ((desc, rotating, order), value) in enumerate(
        zip(channels, operating, strict=True), start=1
    ):
        flag = "T" if rotating else "F"
        out.append(
            f"   {i:8d}    {value: .12E}        {flag}        {order:8d}   {desc}"
        )
    out.append("")
    return out


def _matrix_lines(name: str, matrix: npt.NDArray[np.float64]) -> list[str]:
    """Render one matrix block (header line, value rows, blank line)."""
    n_rows, n_cols = matrix.shape
    out = [f"{name}: {n_rows} x {n_cols}"]
    out.extend("    " + "  ".join(f"{v: .12E}" for v in row) for row in matrix)
    out.append("")
    return out


class LinFileFactory(Protocol):
    """Callable that writes a synthetic ``.lin`` file and returns its path."""

    def __call__(self, **kwargs: object) -> Path:
        """Write a synthetic ``.lin`` file built from ``kwargs`` and return it."""
        ...


@pytest.fixture(autouse=True)
def _isolated_controller_libraries(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Keep the developer's controller-library configuration out of every test.

    Settle runs search ``VANE_CONTROLLER_LIBRARIES`` and VANE's configuration
    directory for a host controller library; a test sees neither unless it sets
    them itself.
    """
    monkeypatch.delenv("VANE_CONTROLLER_LIBRARIES", raising=False)
    monkeypatch.setenv(
        "VANE_CONFIG_DIR", str(tmp_path_factory.getbasetemp() / "vane_config")
    )


@pytest.fixture
def make_lin_file(tmp_path: Path) -> LinFileFactory:
    """Return a factory that writes synthetic ``.lin`` files to a temp dir.

    Returns
    -------
    LinFileFactory
        A callable accepting the same keyword arguments as
        :func:`build_lin_text` and returning the path to the written file.
    """
    counter = {"n": 0}

    def _factory(**kwargs: object) -> Path:
        counter["n"] += 1
        path = tmp_path / f"synthetic_{counter['n']}.1.lin"
        path.write_text(build_lin_text(**kwargs), encoding="utf-8")  # type: ignore[arg-type]
        return path

    return _factory


@pytest.fixture
def simple_lin_file(make_lin_file: LinFileFactory) -> Path:
    """Return a path to a minimal valid two-state synthetic ``.lin`` file.

    Returns
    -------
    pathlib.Path
        A ``.lin`` file with a single second-order DOF (one displacement state
        and its velocity), no inputs or outputs, and a 2x2 ``A`` matrix.
    """
    states = [
        (
            "ED 1st tower fore-aft bending mode DOF (internal DOF index = DOF_TFA1), m",
            False,
            2,
        ),
        (
            "ED First time derivative of 1st tower fore-aft bending mode DOF "
            "(internal DOF index = DOF_TFA1), m/s",
            False,
            1,
        ),
    ]
    a = np.array([[0.0, 1.0], [-4.0, -0.2]], dtype=np.float64)
    return make_lin_file(states=states, a=a)


@pytest.fixture(scope="session")
def rtest_dir() -> Path:
    """Return the local OpenFAST ``r-test`` directory, or skip if absent.

    The location is taken from the ``VANE_RTEST_DIR`` environment variable, which
    names a local clone of the OpenFAST ``r-test`` repository. These turbine files
    are never committed.

    Returns
    -------
    pathlib.Path
        Path to the ``r-test`` repository root.
    """
    from pathlib import Path

    candidates = []
    env = os.environ.get("VANE_RTEST_DIR")
    if env:
        candidates.append(Path(env))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    pytest.skip("r-test data unavailable; set VANE_RTEST_DIR to enable")


@pytest.fixture(scope="session")
def openfast_exe() -> Path:
    """Return the OpenFAST executable, or skip if it is not available.

    Honors the ``VANE_OPENFAST`` environment variable, otherwise looks for
    ``openfast`` on ``PATH``. Integration tests that actually run OpenFAST use
    this and skip cleanly when no executable is installed.

    Returns
    -------
    pathlib.Path
        Path to the OpenFAST executable.
    """
    import os
    import shutil
    from pathlib import Path

    override = os.environ.get("VANE_OPENFAST")
    found = override or shutil.which("openfast")
    if not found or not Path(found).exists():
        pytest.skip("OpenFAST executable unavailable; set VANE_OPENFAST to enable")
    return Path(found)


@pytest.fixture
def oc4semi_lin_file(rtest_dir: Path) -> Path:
    """Return the modern 5MW OC4 semi-sub glued ``.lin`` file from r-test.

    Returns
    -------
    pathlib.Path
        Path to ``5MW_OC4Semi_Linear.1.lin`` (138-state floating case).
    """
    path = (
        rtest_dir
        / "glue-codes"
        / "openfast"
        / "5MW_OC4Semi_Linear"
        / "5MW_OC4Semi_Linear.1.lin"
    )
    if not path.is_file():
        pytest.skip(f"r-test case not found: {path}")
    return path


@pytest.fixture
def calcsteady_lin_files(rtest_dir: Path) -> list[Path]:
    """Return the three modern CalcSteady azimuth ``.lin`` files from r-test.

    Returns
    -------
    list[pathlib.Path]
        The ``5MW_Land_Linear_Aero_CalcSteady.{1,2,3}.lin`` files (azimuths at
        roughly 0, 120, and 240 degrees) used to exercise the MBC3 transform.
    """
    case = rtest_dir / "glue-codes" / "openfast" / "5MW_Land_Linear_Aero_CalcSteady"
    paths = [case / f"5MW_Land_Linear_Aero_CalcSteady.{i}.lin" for i in (1, 2, 3)]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        pytest.skip(f"r-test CalcSteady files not found: {missing}")
    return paths
