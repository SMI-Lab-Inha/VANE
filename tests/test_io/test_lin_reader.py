# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.io.lin_reader — modern OpenFAST .lin parsing."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np
import pytest

from vane.io.lin_reader import (
    LinFileFormatError,
    _parse_float,
    _validate_lin_file,
    find_nonfinite_lin_value,
    read_lin_file,
)

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import LinFileFactory


def _edit_first_data_row(
    text: str, *, keep_order: bool, replace_order: str = ""
) -> str:
    """Return ``text`` with its first operating-point data row corrupted.

    The first table row (the one carrying a ``T``/``F`` rotating-frame flag) is either
    truncated just after the flag (``keep_order=False``) or has its derivative-order
    token replaced by ``replace_order`` (``keep_order=True``).
    """
    out: list[str] = []
    done = False
    for line in text.splitlines():
        tokens = line.split()
        flag_idx = next((i for i, t in enumerate(tokens) if t in ("T", "F")), -1)
        if not done and flag_idx >= 1 and flag_idx + 1 < len(tokens):
            head = tokens[: flag_idx + 1]
            if keep_order:
                out.append(
                    "   " + " ".join([*head, replace_order, *tokens[flag_idx + 2 :]])
                )
            else:
                out.append("   " + " ".join(head))
            done = True
            continue
        out.append(line)
    assert done, "no data row with a rotating-frame flag found"
    return "\n".join(out) + "\n"


# A minimal modern .lin whose single input is an orientation channel written
# with three comma-separated operating-point components.
_ORIENTATION_LIN = """
Linearized model: orientation fixture
 linked with  ElastoDyn; AeroDyn

Description from the FAST input file: orientation fixture

Simulation information:
   Simulation time:                     0.0000 s
   Rotor Speed:                         1.5000 rad/s
   Azimuth:                             0.5000 rad
   Wind Speed:                          8.0000 m/s
   Number of continuous states:         1
   Number of discrete states:           0
   Number of constraint states:         0
   Number of inputs:                    1
   Number of outputs:                   0
   Jacobians included in this file?    No

Order of continuous states:
   Row/Column Operating Point   Rotating Frame? Derivative Order Description
   ---------- ---------------   --------------- ---------------- -----------
          1    0.000000000000E+00        F               2         ED surge, m

Order of inputs:
   Column     Operating Point   Rotating Frame? Derivative Order Description
   ---------- ---------------   --------------- ---------------- -----------
          1    1.000000000000E+00,  2.000000000000E+00,  3.000000000000E+00\
        F               0   AD hub orientation, -

Linearized state matrices:

A: 1 x 1
    0.000000000000E+00

B: 1 x 1
    1.000000000000E+00
"""

# A minimal legacy .lin: note the state table header has no "Derivative Order".
_LEGACY_LIN = """
Linearized model: legacy fixture
 linked with  ElastoDyn

Description from the FAST input file: legacy fixture

Simulation information:
   Simulation time:                     0.0000 s
   Rotor Speed:                         0.0000 rad/s
   Azimuth:                             0.0000 rad
   Wind Speed:                          0.0000 m/s
   Number of continuous states:         2
   Number of inputs:                    0
   Number of outputs:                   0
   Jacobians included in this file?    No

Order of continuous states:
   Row/Column Operating Point     Rotating Frame? Description
   ---------- ---------------     --------------- -----------
          1    0.000E+00                  F        ED tower fore-aft
          2    0.000E+00                  F        ED tower fore-aft deriv

Linearized state matrices:

A: 2 x 2
    0.0  1.0
   -4.0 -0.2
"""

# A modern .lin whose header reports rotor speed in rpm and azimuth in deg, the
# units described (incorrectly) by the OpenFAST documentation.
_RPM_DEG_LIN = """
Linearized model: rpm/deg fixture
 linked with  ElastoDyn

Description from the FAST input file: rpm/deg fixture

Simulation information:
   Simulation time:                     0.0000 s
   Rotor Speed:                        12.0000 rpm
   Azimuth:                            90.0000 deg
   Wind Speed:                          8.0000 m/s
   Number of continuous states:         1
   Number of discrete states:           0
   Number of constraint states:         0
   Number of inputs:                    0
   Number of outputs:                   0
   Jacobians included in this file?    No

Order of continuous states:
   Row/Column Operating Point   Rotating Frame? Derivative Order Description
   ---------- ---------------   --------------- ---------------- -----------
          1    0.000000000000E+00        F               2         ED surge, m

Linearized state matrices:

A: 1 x 1
    0.000000000000E+00
"""


class TestReadLinFileSynthetic:
    """Unit tests using programmatically generated synthetic .lin files."""

    def test_parses_header_scalars(self, make_lin_file: LinFileFactory) -> None:
        """Header scalars are read with the correct units and values."""
        states = [("ED gen DOF, rad", False, 2)]
        path = make_lin_file(
            states=states,
            a=np.zeros((1, 1)),
            rotor_speed=1.25,
            azimuth=0.5,
            wind_speed=8.0,
            sim_time=10.0,
        )
        lin = read_lin_file(path)
        assert lin.rotor_speed == pytest.approx(1.25)
        assert lin.azimuth == pytest.approx(0.5)
        assert lin.wind_speed == pytest.approx(8.0)
        assert lin.sim_time == pytest.approx(10.0)
        assert lin.jacobians_included is False

    def test_rpm_and_degree_units_are_normalized(self, tmp_path: Path) -> None:
        """Rotor speed in rpm and azimuth in deg convert to rad/s and rad."""
        path = tmp_path / "rpmdeg.1.lin"
        path.write_text(_RPM_DEG_LIN, encoding="utf-8")
        lin = read_lin_file(path)
        assert lin.rotor_speed == pytest.approx(12.0 * np.pi / 30.0)
        assert lin.azimuth == pytest.approx(np.pi / 2.0)

    def test_underscore_rotor_speed_field_name(self, tmp_path: Path) -> None:
        """The 'Rotor_Speed' header label (per OpenFAST docs) is also accepted."""
        text = _RPM_DEG_LIN.replace("Rotor Speed:", "Rotor_Speed:")
        path = tmp_path / "underscore.1.lin"
        path.write_text(text, encoding="utf-8")
        lin = read_lin_file(path)
        assert lin.rotor_speed == pytest.approx(12.0 * np.pi / 30.0)

    def test_accepts_a_string_path(self, simple_lin_file: Path) -> None:
        """A plain ``str`` path reads the same file as a ``Path``."""
        from_str = read_lin_file(str(simple_lin_file))
        assert from_str.path == simple_lin_file
        np.testing.assert_array_equal(from_str.a, read_lin_file(simple_lin_file).a)

    def test_parses_state_table(self, simple_lin_file: Path) -> None:
        """The continuous-state table yields parallel descriptor arrays."""
        lin = read_lin_file(simple_lin_file)
        assert lin.n_x == 2
        assert len(lin.x.descriptions) == 2
        assert lin.x.derivative_order.tolist() == [2, 1]
        assert lin.x.rotating_frame.tolist() == [False, False]

    def test_parses_a_matrix(self, simple_lin_file: Path) -> None:
        """The A matrix is parsed with correct shape and values."""
        lin = read_lin_file(simple_lin_file)
        assert lin.a is not None
        assert lin.a.shape == (2, 2)
        np.testing.assert_allclose(lin.a, [[0.0, 1.0], [-4.0, -0.2]])

    def test_absent_matrices_are_none(self, simple_lin_file: Path) -> None:
        """Matrices not written by OpenFAST (no inputs/outputs) parse as None."""
        lin = read_lin_file(simple_lin_file)
        assert lin.b is None
        assert lin.c is None
        assert lin.d is None

    def test_missing_a_matrix_raises(
        self, simple_lin_file: Path, tmp_path: Path
    ) -> None:
        """A file that declares states but has no A matrix fails at parse time.

        A is load-bearing for modal analysis, so its absence is a truncated/malformed
        file, not a silently A-less LinFile that only breaks later in MBC.
        """
        out: list[str] = []
        skipping = False
        for line in simple_lin_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("A: "):  # drop the "A: r x c" block up to its blank line
                skipping = True
                continue
            if skipping:
                if line.strip() == "":
                    skipping = False
                continue
            out.append(line)
        path = tmp_path / "no_a.1.lin"
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
        with pytest.raises(LinFileFormatError, match=r"no A \(state\) matrix"):
            read_lin_file(path)

    def test_header_state_count_mismatch_raises(
        self, simple_lin_file: Path, tmp_path: Path
    ) -> None:
        """A header continuous-state count disagreeing with the table is rejected.

        Anchors the missing-A / truncation check to the header count: a more severely
        truncated file (op-tables also lost) would zero out n_x and otherwise slip past
        the A requirement.
        """
        out = [
            "   Number of continuous states:         5"  # table actually has 2
            if "Number of continuous states:" in line
            else line
            for line in simple_lin_file.read_text(encoding="utf-8").splitlines()
        ]
        path = tmp_path / "count_mismatch.1.lin"
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
        with pytest.raises(LinFileFormatError, match="header declares 5 continuous"):
            read_lin_file(path)

    def test_empty_state_count_value_raises(
        self, simple_lin_file: Path, tmp_path: Path
    ) -> None:
        """A present-but-empty continuous-state count is rejected.

        Leaving it ``None`` would silently disable the header-anchored missing-A check,
        so a truncated label with no value must fail at parse rather than slip through.
        """
        out = [
            "   Number of continuous states:"  # label present, value truncated away
            if "Number of continuous states:" in line
            else line
            for line in simple_lin_file.read_text(encoding="utf-8").splitlines()
        ]
        path = tmp_path / "empty_count.1.lin"
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
        with pytest.raises(LinFileFormatError, match="has no value"):
            read_lin_file(path)

    def test_parses_b_c_d_with_io(self, make_lin_file: LinFileFactory) -> None:
        """B, C, D are parsed when inputs and outputs are present."""
        path = make_lin_file(
            states=[("ED s, m", False, 2), ("ED ds, m/s", False, 1)],
            inputs=[("ED gen torque, Nm", False, 0)],
            outputs=[("ED rotor speed, rad/s", False, 0)],
            a=np.eye(2),
            b=np.array([[0.0], [1.0]]),
            c=np.array([[1.0, 0.0]]),
            d=np.array([[0.5]]),
        )
        lin = read_lin_file(path)
        assert lin.n_u == 1
        assert lin.n_y == 1
        assert lin.b is not None and lin.b.shape == (2, 1)
        assert lin.c is not None and lin.c.shape == (1, 2)
        assert lin.d is not None and lin.d.shape == (1, 1)
        np.testing.assert_allclose(lin.d, [[0.5]])

    def test_rotating_frame_flags(self, make_lin_file: LinFileFactory) -> None:
        """Blade (rotating-frame) states are flagged True; others False."""
        states = [
            ("ED tower FA, m", False, 2),
            ("ED flap blade 1 (DOF_BF(1,1)), m", True, 2),
            ("ED flap blade 2 (DOF_BF(2,1)), m", True, 2),
        ]
        path = make_lin_file(states=states, a=np.zeros((3, 3)))
        lin = read_lin_file(path)
        assert lin.x.rotating_frame.tolist() == [False, True, True]

    def test_modules_property(self, make_lin_file: LinFileFactory) -> None:
        """The module prefix is extracted from each channel description."""
        states = [("ED tower, m", False, 2), ("BD_1 node 2, m", True, 2)]
        path = make_lin_file(states=states, a=np.zeros((2, 2)))
        lin = read_lin_file(path)
        assert lin.x.modules == ["ED", "BD_1"]

    def test_overflow_marker_parsed_as_nan(
        self,
        simple_lin_file: Path,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A Fortran overflow marker in a matrix becomes NaN with a warning."""
        text = simple_lin_file.read_text(encoding="utf-8")
        corrupted = text.replace(" 1.000000000000E+00", " ******************", 1)
        assert corrupted != text
        path = tmp_path / "overflow.1.lin"
        path.write_text(corrupted, encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger="vane.io.lin_reader"):
            lin = read_lin_file(path)
        assert lin.a is not None
        assert np.isnan(lin.a).sum() == 1
        assert any("Overflow" in rec.message for rec in caplog.records)

    def test_msvc_nonfinite_spellings_parse(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Intel/MSVC non-finite spellings map to nan/inf (float() rejects them)."""
        with caplog.at_level(logging.WARNING, logger="vane.io.lin_reader"):
            assert np.isnan(_parse_float("1.#QNAN"))
            assert np.isnan(_parse_float("-1.#IND"))
            assert _parse_float("1.#INF") == np.inf
            assert _parse_float("-1.#INF") == -np.inf
        assert any("Non-finite marker" in rec.message for rec in caplog.records)

    def test_literal_non_finite_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """A literal nan/inf token is accepted but warned (not silently passed)."""
        with caplog.at_level(logging.WARNING, logger="vane.io.lin_reader"):
            assert np.isnan(_parse_float("nan"))
            assert _parse_float("inf") == np.inf
        assert any("Non-finite value" in rec.message for rec in caplog.records)

    def test_non_numeric_token_raises_lin_file_format_error(self) -> None:
        """A genuinely non-numeric token raises LinFileFormatError, not a ValueError."""
        with pytest.raises(LinFileFormatError, match="non-numeric value"):
            _parse_float("not_a_number")

    def test_non_finite_op_table_value_rejected(
        self, simple_lin_file: Path, tmp_path: Path
    ) -> None:
        """A non-finite operating-point baseline value is rejected, not passed on.

        Op-table values are written before the matrices, so corrupting the first one
        targets the ``x`` operating-point baseline.
        """
        text = simple_lin_file.read_text(encoding="utf-8")
        corrupted = text.replace("0.000000000000E+00", "1.#QNAN", 1)
        assert corrupted != text
        path = tmp_path / "op_nan.1.lin"
        path.write_text(corrupted, encoding="utf-8")
        with pytest.raises(LinFileFormatError, match="non-finite value in op table"):
            read_lin_file(path)

    def test_orientation_row_keeps_first_value(self, tmp_path: Path) -> None:
        """An orientation input row (3 comma values) keeps the first value."""
        path = tmp_path / "orientation.1.lin"
        path.write_text(_ORIENTATION_LIN, encoding="utf-8")
        lin = read_lin_file(path)
        assert lin.n_u == 1
        assert lin.u.values.tolist() == [pytest.approx(1.0)]
        assert lin.u.descriptions == ["AD hub orientation, -"]

    def test_legacy_format_raises(self, tmp_path: Path) -> None:
        """A legacy .lin (no Derivative Order column) raises a clear error."""
        path = tmp_path / "legacy.1.lin"
        path.write_text(_LEGACY_LIN, encoding="utf-8")
        with pytest.raises(LinFileFormatError, match="legacy"):
            read_lin_file(path)

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        """An empty file raises LinFileFormatError."""
        path = tmp_path / "empty.1.lin"
        path.write_text("", encoding="utf-8")
        with pytest.raises(LinFileFormatError, match="empty"):
            read_lin_file(path)

    def test_missing_header_raises(self, tmp_path: Path) -> None:
        """A file without the simulation-information header raises."""
        path = tmp_path / "nohdr.1.lin"
        path.write_text("just some text\nno header here\n", encoding="utf-8")
        with pytest.raises(LinFileFormatError, match="Simulation information"):
            read_lin_file(path)

    def test_truncated_matrix_raises(
        self, simple_lin_file: Path, tmp_path: Path
    ) -> None:
        """A matrix declaring more rows than present raises a clear error."""
        text = simple_lin_file.read_text(encoding="utf-8")
        truncated = text.replace("A: 2 x 2", "A: 3 x 2")
        path = tmp_path / "truncated.1.lin"
        path.write_text(truncated, encoding="utf-8")
        with pytest.raises(LinFileFormatError, match="truncated"):
            read_lin_file(path)

    def test_header_field_without_value_raises(
        self, simple_lin_file: Path, tmp_path: Path
    ) -> None:
        """A header scalar label with no value raises a format error, not IndexError."""
        text = simple_lin_file.read_text(encoding="utf-8")
        patched = "\n".join(
            "   Rotor Speed:" if "Rotor Speed:" in line else line
            for line in text.splitlines()
        )
        path = tmp_path / "noval.1.lin"
        path.write_text(patched + "\n", encoding="utf-8")
        with pytest.raises(LinFileFormatError, match="has no value"):
            read_lin_file(path)

    def test_truncated_table_row_raises(
        self, simple_lin_file: Path, tmp_path: Path
    ) -> None:
        """A row missing its derivative-order column raises a format error."""
        text = simple_lin_file.read_text(encoding="utf-8")
        path = tmp_path / "shortrow.1.lin"
        path.write_text(_edit_first_data_row(text, keep_order=False), encoding="utf-8")
        with pytest.raises(LinFileFormatError, match="Missing derivative-order column"):
            read_lin_file(path)

    def test_non_integer_derivative_order_raises(
        self, simple_lin_file: Path, tmp_path: Path
    ) -> None:
        """A non-integer derivative-order token raises a format error."""
        text = simple_lin_file.read_text(encoding="utf-8")
        corrupted = _edit_first_data_row(text, keep_order=True, replace_order="x")
        path = tmp_path / "badorder.1.lin"
        path.write_text(corrupted, encoding="utf-8")
        with pytest.raises(LinFileFormatError, match="not an integer"):
            read_lin_file(path)


class TestLinFileValidation:
    """Tests for post-parse validation of operating metadata and matrix shapes."""

    def test_non_finite_rotor_speed_raises(self, simple_lin_file: Path) -> None:
        """A non-finite operating scalar is rejected loudly."""
        lin = read_lin_file(simple_lin_file)
        lin.rotor_speed = float("nan")
        with pytest.raises(LinFileFormatError, match="non-finite rotor speed"):
            _validate_lin_file(lin)

    def test_non_finite_wind_speed_raises(self, simple_lin_file: Path) -> None:
        """A non-finite wind speed is rejected."""
        lin = read_lin_file(simple_lin_file)
        lin.wind_speed = float("inf")
        with pytest.raises(LinFileFormatError, match="non-finite wind speed"):
            _validate_lin_file(lin)

    def test_matrix_shape_mismatch_raises(self, simple_lin_file: Path) -> None:
        """An A matrix whose shape disagrees with the channel tables is rejected."""
        lin = read_lin_file(simple_lin_file)
        lin.a = np.zeros((2, 3), dtype=np.float64)  # not (n_x, n_x)
        with pytest.raises(LinFileFormatError, match="matrix A has shape"):
            _validate_lin_file(lin)

    def test_valid_file_passes_validation(self, simple_lin_file: Path) -> None:
        """A well-formed file validates without error."""
        _validate_lin_file(read_lin_file(simple_lin_file))


class TestReadLinFileRTest:
    """Integration tests against real modern r-test fixtures (skipped if absent)."""

    def test_oc4semi_dimensions(self, oc4semi_lin_file: Path) -> None:
        """The 5MW OC4 semi-sub case parses with documented dimensions."""
        lin = read_lin_file(oc4semi_lin_file)
        assert (lin.n_x, lin.n_u, lin.n_y) == (138, 13, 15)
        assert lin.a is not None and lin.a.shape == (138, 138)
        assert lin.b is not None and lin.b.shape == (138, 13)
        assert lin.c is not None and lin.c.shape == (15, 138)
        assert lin.d is not None and lin.d.shape == (15, 13)
        assert lin.jacobians_included is False

    def test_oc4semi_physics(self, oc4semi_lin_file: Path) -> None:
        """Header and descriptors match the known floating-platform case."""
        lin = read_lin_file(oc4semi_lin_file)
        assert lin.rotor_speed == pytest.approx(1.2566, abs=1e-3)
        # The six platform DOFs are present and tagged as second-order states.
        assert any("DOF_Sg" in desc for desc in lin.x.descriptions)
        assert any("DOF_P)" in desc for desc in lin.x.descriptions)
        assert 2 in lin.x.derivative_order.tolist()
        # Blade states are in the rotating frame.
        assert bool(lin.x.rotating_frame.any())
        assert "ED" in lin.x.modules


_TIP_STATE = "AD x3 blade 1, node 59, -"
_NONFINITE_STATES = [
    ("ED 1st flapwise bending-mode DOF of blade 1, m", True, 2),
    (_TIP_STATE, True, 1),
]


class TestFindNonfiniteLinValue:
    """The non-finite scan names the first NaN/infinite value and its channel."""

    def test_finite_file_returns_none(self, simple_lin_file: Path) -> None:
        """A clean file has nothing to report."""
        assert find_nonfinite_lin_value(simple_lin_file) is None

    def test_nan_operating_point_names_the_channel(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """A NaN state value is reported with its description and line."""
        path = make_lin_file(
            states=_NONFINITE_STATES,
            state_values=[0.5, float("nan")],
            a=np.zeros((2, 2)),
        )
        found = find_nonfinite_lin_value(path)
        assert found is not None
        assert found.section == "x"
        assert found.variable == _TIP_STATE
        assert found.token.lower() == "nan"
        lines = path.read_text(encoding="utf-8").splitlines()
        assert _TIP_STATE in lines[found.line_number - 1]
        assert _TIP_STATE in found.summary
        assert "table 'x'" in found.summary

    def test_infinite_matrix_entry_names_row_and_column(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """``read_lin_file`` accepts a non-finite matrix entry; the scan does not."""
        a = np.zeros((2, 2))
        a[1, 0] = np.inf
        path = make_lin_file(states=_NONFINITE_STATES, a=a)
        read_lin_file(path)  # tolerated by the reader as data
        found = find_nonfinite_lin_value(path)
        assert found is not None
        assert found.section == "A"
        assert found.variable.startswith("A[2, 1] (row: First time derivative of")
        assert "column: ED 1st flapwise" in found.variable
        assert "matrix 'A'" in found.summary

    @pytest.mark.parametrize("token", ["*****************", "1.#QNAN", "-Infinity"])
    def test_fortran_and_msvc_spellings(
        self, make_lin_file: LinFileFactory, token: str
    ) -> None:
        """Overflow runs and Intel/MSVC markers count as non-finite."""
        path = make_lin_file(states=_NONFINITE_STATES, a=np.full((2, 2), 7.0))
        text = path.read_text(encoding="utf-8")
        value = f"{7.0: .12E}"
        index = text.rindex(value)
        path.write_text(
            text[:index] + " " + token + text[index + len(value) :], encoding="utf-8"
        )
        found = find_nonfinite_lin_value(path)
        assert found is not None
        assert found.token == token
        assert found.variable.startswith("A[2, 2]")

    def test_words_in_descriptions_are_not_numbers(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """``NaN``/``Inf`` words and ``*`` in descriptions are not values."""
        states = [("ED inf NaN * InflowWind # channel, -", False, 1)]
        path = make_lin_file(states=states, a=np.zeros((1, 1)))
        assert find_nonfinite_lin_value(path) is None

    def test_header_scalar(self, make_lin_file: LinFileFactory) -> None:
        """A non-finite header scalar names its field."""
        path = make_lin_file(states=_NONFINITE_STATES, a=np.zeros((2, 2)))
        text = path.read_text(encoding="utf-8")
        start = text.index("Rotor Speed:")
        end = text.index("rad/s", start)
        path.write_text(
            text[:start] + "Rotor Speed:   NaN " + text[end:], encoding="utf-8"
        )
        found = find_nonfinite_lin_value(path)
        assert found is not None
        assert (found.section, found.variable) == ("header", "Rotor Speed")
        assert "header" in found.summary

    def test_later_orientation_component(self, tmp_path: Path) -> None:
        """Every component of an orientation channel is scanned."""
        path = tmp_path / "orientation.1.lin"
        path.write_text(
            _ORIENTATION_LIN.replace("2.000000000000E+00,", "NaN,"), encoding="utf-8"
        )
        found = find_nonfinite_lin_value(path)
        assert found is not None
        assert (found.section, found.variable) == ("u", "AD hub orientation, -")


def _iea22_nan_lin() -> Path:
    """Return a NaN ``.lin`` of the IEA 22-MW rotor-only sweep, or skip."""
    from pathlib import Path

    root = os.environ.get("VANE_REFERENCE_RUNS_DIR")
    if not root:
        pytest.skip("reference-turbine data unavailable; set VANE_REFERENCE_RUNS_DIR")
    if not Path(root).is_dir():
        pytest.skip("reference-turbine runs unavailable; set VANE_REFERENCE_RUNS_DIR")
    path = (
        Path(root)
        / "iea22_rotoronly_nan"
        / "cases"
        / "ws5"
        / "IEA-22-280-RWT-Monopile"
        / "IEA-22-280-RWT-Monopile.1.lin"
    )
    if not path.is_file():
        pytest.skip(f"IEA 22-MW NaN linearization not found: {path}")
    return path


class TestFindNonfiniteLinValueReal:
    """The IEA 22-MW case: UA on at the tip node under tip loss (skipped if absent)."""

    def test_names_the_tip_node_unsteady_aero_state(self) -> None:
        """The first non-finite value is the tip node's third UA state, blade 1."""
        path = _iea22_nan_lin()
        found = find_nonfinite_lin_value(path)
        assert found is not None
        assert found.section == "x"
        assert found.variable == _TIP_STATE
        with pytest.raises(LinFileFormatError, match="non-finite value in op table"):
            read_lin_file(path)
