# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.study — study orchestration and reproducibility provenance."""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from vane.io.lin_reader import LinFile, read_lin_file
from vane.pipeline import ModalPipeline
from vane.study import discover_operating_points, run_study

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import LinFileFactory

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
_TIMESTAMP = datetime(2026, 6, 9, 12, 0, 0, tzinfo=timezone.utc)


def _operating_point(
    make_lin_file: LinFileFactory, *, rotor_speed: float
) -> list[LinFile]:
    """Build a two-azimuth operating point at a given rotor speed."""
    return [
        read_lin_file(
            make_lin_file(states=_STATES, a=_A, rotor_speed=rotor_speed, azimuth=az)
        )
        for az in (0.0, 1.0)
    ]


# Above rated: rotor speeds equal to round-off (12.1 rpm), winds distinct.
_RATED_RAD_S = 12.1 * np.pi / 30.0
_PLATEAU_WINDS = (12.0, 14.0, 16.0, 18.0)


def _plateau_sweep(make_lin_file: LinFileFactory) -> list[list[LinFile]]:
    """Build an above-rated, out-of-wind-order sweep on a rotor-speed plateau."""
    points = []
    for wind, round_off in zip(_PLATEAU_WINDS, (1e-9, -2e-9, 3e-9, -1e-9), strict=True):
        points.append(
            [
                dataclasses.replace(
                    read_lin_file(
                        make_lin_file(states=_STATES, a=_A, wind_speed=wind, azimuth=az)
                    ),
                    rotor_speed=_RATED_RAD_S + round_off,
                )
                for az in (0.0, 1.0)
            ]
        )
    return [points[2], points[0], points[3], points[1]]


class TestRunStudy:
    """Tests for run_study and the recorded provenance."""

    def test_records_inputs_and_thresholds(self, make_lin_file: LinFileFactory) -> None:
        """Provenance records the source files, coverage, thresholds, and counts."""
        operating_points = [
            _operating_point(make_lin_file, rotor_speed=1.0),
            _operating_point(make_lin_file, rotor_speed=1.5),
        ]
        result = run_study(operating_points, harmonics=[1, 3], timestamp=_TIMESTAMP)
        provenance = result.provenance

        assert provenance.created_at == _TIMESTAMP.isoformat()
        assert provenance.vane_version
        assert provenance.parameter_name == "rotor_speed_rpm"
        assert provenance.harmonics == (1, 3)
        assert provenance.tracking_strategy == "single_path"
        assert provenance.branch_review is False
        assert provenance.resolve is False
        assert provenance.n_branch_groups == 0
        assert len(provenance.operating_points) == 2
        assert all(op.n_azimuths == 2 for op in provenance.operating_points)
        assert len(provenance.source_files) == 4

    def test_records_tracking_strategy(self, make_lin_file: LinFileFactory) -> None:
        """A non-default tracking strategy is captured in the provenance."""
        result = run_study(
            [_operating_point(make_lin_file, rotor_speed=1.0)],
            tracking_strategy="simultaneous",
            timestamp=_TIMESTAMP,
        )
        assert result.provenance.tracking_strategy == "simultaneous"

    def test_records_branch_review(self, make_lin_file: LinFileFactory) -> None:
        """Requesting branch review is captured with its group count in provenance."""
        result = run_study(
            [_operating_point(make_lin_file, rotor_speed=1.0)],
            branch_review=True,
            timestamp=_TIMESTAMP,
        )
        assert result.provenance.branch_review is True
        assert result.provenance.resolve is False
        assert result.provenance.n_branch_groups == len(
            result.pipeline.identification.branch_groups
        )

    def test_resolve_is_recorded_and_implies_review(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Requesting resolution records it and turns on review in the provenance."""
        result = run_study(
            [_operating_point(make_lin_file, rotor_speed=1.0)],
            resolve=True,
            timestamp=_TIMESTAMP,
        )
        assert result.provenance.resolve is True
        assert result.provenance.branch_review is True

    def test_records_audit_grade_per_operating_point_detail(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Each operating point records its full azimuths, parameter value, and env."""
        result = run_study(
            [_operating_point(make_lin_file, rotor_speed=1.0)], timestamp=_TIMESTAMP
        )
        op = result.provenance.operating_points[0]
        assert op.n_azimuths == 2
        assert len(op.azimuths_deg) == 2
        assert op.azimuth_min_deg <= op.azimuth_max_deg
        assert op.parameter_value == pytest.approx(op.rotor_speed_rpm)
        assert len(op.source_files) == 2
        environment = result.provenance.environment
        assert environment.python_version
        assert environment.platform
        assert environment.dependencies["numpy"]

    def test_unknown_parameter_name_raises(self, make_lin_file: LinFileFactory) -> None:
        """run_study validates the parameter name before doing any work."""
        with pytest.raises(ValueError, match="parameter_name"):
            run_study(
                [_operating_point(make_lin_file, rotor_speed=1.0)],
                parameter_name="bogus",
            )

    def test_rotor_speed_plateau_rejected_before_pipeline(
        self, make_lin_file: LinFileFactory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An above-rated rpm plateau is rejected up front, recommending wind."""

        def _unreachable(*args: object, **kwargs: object) -> None:
            msg = "the pipeline must not run on a plateaued axis"
            raise AssertionError(msg)

        monkeypatch.setattr(ModalPipeline, "run", _unreachable)
        with pytest.raises(ValueError, match="plateau") as excinfo:
            run_study(_plateau_sweep(make_lin_file), timestamp=_TIMESTAMP)
        assert "--parameter wind" in str(excinfo.value)

    def test_wind_axis_orders_a_plateau_sweep(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """The above-rated sweep runs on the wind axis, provenance in wind order."""
        result = run_study(
            _plateau_sweep(make_lin_file),
            parameter_name="wind_speed",
            timestamp=_TIMESTAMP,
        )
        winds = [op.wind_speed for op in result.provenance.operating_points]
        assert winds == sorted(_PLATEAU_WINDS)
        np.testing.assert_allclose(result.pipeline.campbell.parameter_values, winds)

    def test_parameter_rtol_is_forwarded(self, make_lin_file: LinFileFactory) -> None:
        """A zero tolerance restores exact-distinctness, both up front and in run."""
        result = run_study(
            _plateau_sweep(make_lin_file), parameter_rtol=0.0, timestamp=_TIMESTAMP
        )
        assert len(result.provenance.operating_points) == len(_PLATEAU_WINDS)
        # The tolerance that let the run through is part of its reproducibility record.
        assert result.provenance.parameter_rtol == 0.0
        assert result.provenance.to_dict()["parameter_rtol"] == 0.0

    def test_invalid_parameter_rtol_raises(self, make_lin_file: LinFileFactory) -> None:
        """run_study validates the plateau tolerance before doing any work."""
        with pytest.raises(ValueError, match="parameter_rtol"):
            run_study(
                [_operating_point(make_lin_file, rotor_speed=1.0)],
                parameter_rtol=-1.0,
            )

    def test_discovered_case_names_flow_into_provenance(
        self, make_lin_file: LinFileFactory, tmp_path: Path
    ) -> None:
        """Discovered case names are preserved in the provenance."""
        data_dir = tmp_path / "campaign"
        data_dir.mkdir()
        for index, azimuth in enumerate((0.0, 1.0), start=1):
            content = read_lin_file(
                make_lin_file(states=_STATES, a=_A, rotor_speed=1.0, azimuth=azimuth)
            ).path.read_text(encoding="utf-8")
            (data_dir / f"caseA.{index}.lin").write_text(content, encoding="utf-8")
        result = run_study(discover_operating_points(data_dir), timestamp=_TIMESTAMP)
        assert result.provenance.operating_points[0].name == "caseA"

    def test_provenance_order_matches_sorted_result(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Provenance operating points are sorted to match the Campbell result."""
        # Supplied out of ascending rotor-speed order.
        operating_points = [
            _operating_point(make_lin_file, rotor_speed=2.0),
            _operating_point(make_lin_file, rotor_speed=1.0),
        ]
        result = run_study(operating_points, timestamp=_TIMESTAMP)
        provenance_rpm = [
            op.rotor_speed_rpm for op in result.provenance.operating_points
        ]
        campbell_rpm = list(result.pipeline.campbell.parameter_values)
        assert provenance_rpm == sorted(provenance_rpm)
        np.testing.assert_allclose(provenance_rpm, campbell_rpm, rtol=1e-9)

    def test_harmonics_are_normalized_in_provenance(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Provenance records the pipeline-normalized (int) harmonics."""
        harmonics: Any = [3.0, 1.0]  # float input the pipeline normalizes to int
        result = run_study(
            [_operating_point(make_lin_file, rotor_speed=1.0)],
            harmonics=harmonics,
            timestamp=_TIMESTAMP,
        )
        assert result.provenance.harmonics == (3, 1)
        assert all(isinstance(n, int) for n in result.provenance.harmonics)

    def test_accepts_string_paths(
        self, make_lin_file: LinFileFactory, tmp_path: Path
    ) -> None:
        """discover_operating_points and write_bundle accept str paths."""
        result = run_study(
            [_operating_point(make_lin_file, rotor_speed=1.0)], timestamp=_TIMESTAMP
        )
        result.write_bundle(str(tmp_path / "bundle"))
        assert (tmp_path / "bundle" / "provenance.json").exists()

    def test_source_files_are_hashed(self, make_lin_file: LinFileFactory) -> None:
        """Each source file carries a 64-hex SHA-256 of its contents."""
        result = run_study(
            [_operating_point(make_lin_file, rotor_speed=1.0)], timestamp=_TIMESTAMP
        )
        for source in result.provenance.source_files:
            assert len(source.sha256) == 64
            assert int(source.sha256, 16) >= 0  # valid hex

    def test_write_bundle(self, make_lin_file: LinFileFactory, tmp_path: Path) -> None:
        """The bundle writes a provenance manifest and a Campbell table."""
        result = run_study(
            [
                _operating_point(make_lin_file, rotor_speed=1.0),
                _operating_point(make_lin_file, rotor_speed=1.5),
            ],
            timestamp=_TIMESTAMP,
        )
        bundle = tmp_path / "bundle"
        result.write_bundle(bundle)

        manifest = json.loads((bundle / "provenance.json").read_text(encoding="utf-8"))
        assert manifest["vane_version"]
        # Source files are grouped per operating point (audit-grade), with the env.
        assert len(manifest["operating_points"]) == 2
        assert sum(len(op["source_files"]) for op in manifest["operating_points"]) == 4
        assert manifest["environment"]["python_version"]
        assert "numpy" in manifest["environment"]["dependencies"]
        assert (bundle / "campbell.csv").exists()
        # The audit report is written by default.
        assert (bundle / "report.html").exists()

    def test_write_bundle_can_skip_report(
        self, make_lin_file: LinFileFactory, tmp_path: Path
    ) -> None:
        """write_bundle(report=False) omits the HTML report."""
        result = run_study(
            [_operating_point(make_lin_file, rotor_speed=1.0)], timestamp=_TIMESTAMP
        )
        bundle = tmp_path / "bundle"
        result.write_bundle(bundle, report=False)
        assert (bundle / "provenance.json").exists()
        assert not (bundle / "report.html").exists()


class TestDiscoverOperatingPoints:
    """Tests for discover_operating_points."""

    def test_groups_by_case_root(
        self, make_lin_file: LinFileFactory, tmp_path: Path
    ) -> None:
        """Files sharing a <case> root are grouped into one operating point."""
        content = read_lin_file(
            make_lin_file(states=_STATES, a=_A, rotor_speed=1.0, azimuth=0.0)
        ).path.read_text(encoding="utf-8")
        data_dir = tmp_path / "campaign"
        data_dir.mkdir()
        for name in ("caseA.1.lin", "caseA.2.lin", "caseB.1.lin"):
            (data_dir / name).write_text(content, encoding="utf-8")

        operating_points = discover_operating_points(data_dir)
        assert len(operating_points) == 2
        assert [len(op) for op in operating_points] == [2, 1]
        assert [op.name for op in operating_points] == ["caseA", "caseB"]
        # A discovered point behaves as a sequence of its files.
        assert operating_points[0][0] is operating_points[0].files[0]

    def test_recurses_without_merging_same_named_cases(
        self, make_lin_file: LinFileFactory, tmp_path: Path
    ) -> None:
        """Nested sweep directories are kept as distinct operating points."""
        content = read_lin_file(
            make_lin_file(states=_STATES, a=_A, rotor_speed=1.0, azimuth=0.0)
        ).path.read_text(encoding="utf-8")
        data_dir = tmp_path / "campaign"
        for folder in ("ws8", "ws12"):
            point_dir = data_dir / folder
            point_dir.mkdir(parents=True)
            for name in ("case.1.lin", "case.2.lin"):
                (point_dir / name).write_text(content, encoding="utf-8")

        operating_points = discover_operating_points(data_dir)

        assert [op.name for op in operating_points] == ["ws12/case", "ws8/case"]
        assert [len(op) for op in operating_points] == [2, 2]

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        """A directory with no .lin files raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match=r"No full-system \.lin files"):
            discover_operating_points(tmp_path)

    def test_per_module_lin_files_excluded(
        self, make_lin_file: LinFileFactory, tmp_path: Path
    ) -> None:
        """LinOutMod per-module files are not ingested as operating points."""
        content = read_lin_file(
            make_lin_file(states=_STATES, a=_A, rotor_speed=1.0, azimuth=0.0)
        ).path.read_text(encoding="utf-8")
        data_dir = tmp_path / "campaign"
        data_dir.mkdir()
        # Full-system sweep plus the ElastoDyn per-module file OpenFAST writes
        # alongside it when LinOutMod = True.
        for name in ("case.1.lin", "case.2.lin", "case.ED.1.lin"):
            (data_dir / name).write_text(content, encoding="utf-8")

        operating_points = discover_operating_points(data_dir)
        assert [op.name for op in operating_points] == ["case"]
        assert len(operating_points[0]) == 2

    def test_openfast_named_module_files_excluded(
        self, make_lin_file: LinFileFactory, tmp_path: Path
    ) -> None:
        """OpenFAST's ``<case>.<index>.<Module>.lin`` files are never operating points.

        OpenFAST writes the per-module files with the module token after the index,
        and appends the instance number for multi-instance modules (``BD1``).
        """
        content = read_lin_file(
            make_lin_file(states=_STATES, a=_A, rotor_speed=1.0, azimuth=0.0)
        ).path.read_text(encoding="utf-8")
        data_dir = tmp_path / "campaign"
        data_dir.mkdir()
        names = (
            "case.1.lin",
            "case.2.lin",
            "case.1.ED.lin",
            "case.1.BD1.lin",
            "case.2.SrvD.lin",
            "case.2.ExtPtfm.lin",
        )
        for name in names:
            (data_dir / name).write_text(content, encoding="utf-8")

        operating_points = discover_operating_points(data_dir)
        assert [op.name for op in operating_points] == ["case"]
        assert len(operating_points[0]) == 2

    def test_any_module_token_after_the_index_is_excluded(
        self, make_lin_file: LinFileFactory, tmp_path: Path
    ) -> None:
        """``<case>.<index>.<Token>.lin`` is excluded whatever the module token.

        Newer OpenFAST modules (``SEA``, ``SED``, ``ExtLd``, ``SlD``, ...) and any
        future one must not become a spurious ``case.1`` operating point.
        """
        content = read_lin_file(
            make_lin_file(states=_STATES, a=_A, rotor_speed=1.0, azimuth=0.0)
        ).path.read_text(encoding="utf-8")
        data_dir = tmp_path / "campaign"
        data_dir.mkdir()
        names = (
            "case.1.lin",
            "case.2.lin",
            "case.1.SEA.lin",
            "case.2.SED.lin",
            "case.1.ExtLd.lin",
            "case.2.SlD.lin",
            "case.1.NewMod2.lin",
        )
        for name in names:
            (data_dir / name).write_text(content, encoding="utf-8")

        operating_points = discover_operating_points(data_dir)
        assert [op.name for op in operating_points] == ["case"]
        assert len(operating_points[0]) == 2

    def test_module_like_root_without_twin_is_kept(
        self, make_lin_file: LinFileFactory, tmp_path: Path
    ) -> None:
        """A case root ending in a module token is kept when it has no full-system twin.

        ``case.ED.fst`` produces only ``case.ED.<i>.lin`` (no ``case.<i>.lin``), so the
        token is part of the case name, not a per-module marker, and must be discovered.
        """
        content = read_lin_file(
            make_lin_file(states=_STATES, a=_A, rotor_speed=1.0, azimuth=0.0)
        ).path.read_text(encoding="utf-8")
        data_dir = tmp_path / "campaign"
        data_dir.mkdir()
        for name in ("case.ED.1.lin", "case.ED.2.lin"):
            (data_dir / name).write_text(content, encoding="utf-8")

        operating_points = discover_operating_points(data_dir)
        assert [op.name for op in operating_points] == ["case.ED"]
        assert len(operating_points[0]) == 2

    def test_separate_module_like_run_with_partial_twin_is_kept(
        self, make_lin_file: LinFileFactory, tmp_path: Path
    ) -> None:
        """A separate ``case.ED`` run is kept whole when not every index is twinned.

        ``case.fst`` (one azimuth) and ``case.ED.fst`` (two azimuths) share index 1 but
        not index 2, so the ``case.ED`` group is a real operating point — it must be
        kept entirely, never split into a partial sweep.
        """
        content = read_lin_file(
            make_lin_file(states=_STATES, a=_A, rotor_speed=1.0, azimuth=0.0)
        ).path.read_text(encoding="utf-8")
        data_dir = tmp_path / "campaign"
        data_dir.mkdir()
        for name in ("case.1.lin", "case.ED.1.lin", "case.ED.2.lin"):
            (data_dir / name).write_text(content, encoding="utf-8")

        operating_points = discover_operating_points(data_dir)
        by_name = {op.name: len(op) for op in operating_points}
        assert by_name == {"case": 1, "case.ED": 2}

    def test_coincidental_full_twin_excluded_with_warning(
        self,
        make_lin_file: LinFileFactory,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The undecidable ``case.1`` + ``case.ED.1`` case excludes, but not silently.

        When every index of the module-like group has a twin, the common LinOutMod
        reading is taken (the file is excluded); a warning is logged so the choice is
        visible if the files were really separate full-system runs.
        """
        content = read_lin_file(
            make_lin_file(states=_STATES, a=_A, rotor_speed=1.0, azimuth=0.0)
        ).path.read_text(encoding="utf-8")
        data_dir = tmp_path / "campaign"
        data_dir.mkdir()
        for name in ("case.1.lin", "case.ED.1.lin"):
            (data_dir / name).write_text(content, encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="vane.study"):
            operating_points = discover_operating_points(data_dir)
        assert [op.name for op in operating_points] == ["case"]
        assert any("case.ED.1.lin" in rec.message for rec in caplog.records)
