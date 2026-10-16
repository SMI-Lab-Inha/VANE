# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Tests for vane.viz.report — the HTML study report."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from vane.io.lin_reader import read_lin_file
from vane.modal.branch import BranchGroup, BranchKind, BranchSubWindow
from vane.study import run_study
from vane.viz.report import study_report_html, write_report

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import LinFileFactory
    from vane.study import StudyResult

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


def _study(
    make_lin_file: LinFileFactory,
    *,
    branch_review: bool = False,
    resolve: bool = False,
) -> StudyResult:
    """Run a two-operating-point study against the synthetic system."""
    operating_points = [
        [
            read_lin_file(
                make_lin_file(states=_STATES, a=_A, rotor_speed=speed, azimuth=az)
            )
            for az in (0.0, 1.0)
        ]
        for speed in (1.0, 1.5)
    ]
    return run_study(operating_points, branch_review=branch_review, resolve=resolve)


def _wind_study(make_lin_file: LinFileFactory) -> StudyResult:
    """Run a two-point study parameterized by wind speed (no resonance search)."""
    operating_points = [
        [
            read_lin_file(
                make_lin_file(
                    states=_STATES,
                    a=_A,
                    rotor_speed=speed,
                    azimuth=az,
                    wind_speed=wind,
                )
            )
            for az in (0.0, 1.0, 2.0)
        ]
        for wind, speed in ((8.0, 1.8), (10.0, 2.2))
    ]
    return run_study(operating_points, parameter_name="wind_speed")


class TestResonanceNotEvaluated:
    """A wind-speed sweep never searches for resonances; the report must say so."""

    def test_wind_sweep_resonance_check_is_not_evaluated(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """The resonance-severity check is 'not evaluated', not a pass."""
        from vane.viz.report import _audit_findings

        result = _wind_study(make_lin_file)
        assert result.pipeline.resonances == []
        checks, _ = _audit_findings(result)
        severity = next(c for c in checks if c[0] == "Resonance severity")
        assert severity[1] is None
        assert "wind-speed" in severity[2]

    def test_wind_sweep_report_does_not_claim_no_resonances(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """The rendered report shows 'Not evaluated' and no clean-bill claims."""
        report = study_report_html(_wind_study(make_lin_file), include_plotlyjs=False)
        assert (
            '<tr><td>Resonance severity</td><td><span class="na">Not evaluated'
            in report
        )
        assert "Not evaluated (wind-speed sweep axis)" in report
        assert "No resonance crossings detected" not in report
        assert "no high-severity resonances" not in report
        assert "detected only for rotor-speed sweeps" in report

    def test_rotor_speed_sweep_still_passes(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """A rotor-speed sweep with no crossings keeps the pass and the clean text."""
        report = study_report_html(_study(make_lin_file), include_plotlyjs=False)
        assert "Not evaluated" not in report
        assert '<tr><td>Resonance severity</td><td><span class="pass">' in report


class TestStudyReport:
    """Tests for study_report_html and write_report."""

    def test_html_contains_each_section(self, make_lin_file: LinFileFactory) -> None:
        """The report is a full HTML document with every section."""
        result = _study(make_lin_file)
        report = study_report_html(result)
        assert report.startswith("<!DOCTYPE html>")
        for section in (
            "Provenance",
            "Checks",
            "Warnings",
            "Operating points",
            "Campbell diagram",
            "Tracked modes",
            "Resonance crossings",
            "Diagnostics",
            "Source files",
        ):
            assert section in report

    def test_low_confidence_track_is_flagged(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """A low-confidence track surfaces a warning and fails the confidence check."""
        result = _study(make_lin_file)
        result.pipeline.campbell.tracks[0].confidence = 0.1
        report = study_report_html(result)
        assert "low identification confidence" in report
        assert "&#9888; Review" in report  # the review badge appears

    def test_thin_coverage_not_flagged_without_mbc_transform(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """A rotating point where MBC was skipped (no triplets) is not flagged."""
        from vane.viz.report import _audit_findings

        result = _study(make_lin_file)  # rotating, 2 azimuths, but no blade triplets
        assert all(
            not mbc.performed_transformation for mbc in result.pipeline.mbc_results
        )
        checks, _ = _audit_findings(result)
        coverage = next(check for check in checks if check[0] == "Azimuth coverage")
        assert coverage[1] is True  # MBC did not run here, so no coverage complaint

    def test_thin_coverage_flagged_when_mbc_applied(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """A rotating, MBC-transformed point below 3 azimuths fails coverage."""
        from vane.viz.report import _audit_findings

        result = _study(make_lin_file)  # rotating points, two azimuths each
        points = result.provenance.operating_points
        assert all(p.n_azimuths < 3 and abs(p.rotor_speed_rpm) > 0 for p in points)
        # Simulate triplet-bearing points so the MBC transform is considered applied.
        for mbc in result.pipeline.mbc_results:
            mbc.performed_transformation = True

        checks, warnings = _audit_findings(result)
        coverage = next(check for check in checks if check[0] == "Azimuth coverage")
        assert coverage[1] is False  # now flagged (rotating + MBC + < 3 azimuths)
        assert any("unreliable" in warning.lower() for warning in warnings)

    def test_report_embeds_provenance_and_plot(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """The report embeds the Plotly figure and the hashed source files."""
        result = _study(make_lin_file)
        report = study_report_html(result)
        assert "plotly" in report.lower()
        # Every source file's hash is listed in the manifest.
        for source in result.provenance.source_files:
            assert source.sha256 in report

    def test_write_report_creates_file(
        self, make_lin_file: LinFileFactory, tmp_path: Path
    ) -> None:
        """write_report creates parent directories and writes the HTML."""
        result = _study(make_lin_file)
        output = tmp_path / "nested" / "report.html"
        write_report(result, output)
        assert output.exists()
        assert output.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")

    def test_write_report_pdf(
        self, make_lin_file: LinFileFactory, tmp_path: Path
    ) -> None:
        """write_report_pdf writes a PDF, or raises a clear error without WeasyPrint."""
        from vane.viz.report import write_report_pdf

        result = _study(make_lin_file)
        output = tmp_path / "report.pdf"
        try:
            from weasyprint import HTML  # noqa: F401
        except ImportError:
            with pytest.raises(ValueError, match="weasyprint"):
                write_report_pdf(result, output)
        else:
            write_report_pdf(result, output)
            assert output.is_file()
            assert output.read_bytes().startswith(b"%PDF")

    def test_write_report_accepts_str_path(
        self, make_lin_file: LinFileFactory, tmp_path: Path
    ) -> None:
        """write_report accepts a string path."""
        result = _study(make_lin_file)
        write_report(result, str(tmp_path / "report.html"))
        assert (tmp_path / "report.html").exists()

    def test_branch_review_section_present_when_enabled(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Enabling branch review adds the section and an on-state provenance row."""
        result = _study(make_lin_file, branch_review=True)
        report = study_report_html(result)
        assert "<h2>Branch review</h2>" in report
        assert "on, 0 contested group(s)" in report

    def test_branch_review_section_absent_when_disabled(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Without branch review the section is omitted but the row reads 'off'."""
        result = _study(make_lin_file)
        report = study_report_html(result)
        assert "<h2>Branch review</h2>" not in report
        assert "<td>off</td>" in report

    def test_resolution_is_noted_in_provenance(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """Requesting resolution adds the smooth-alternative note to the report."""
        result = _study(make_lin_file, resolve=True)
        report = study_report_html(result)
        assert "smooth alternative computed" in report

    def test_contested_group_renders_table_row(
        self, make_lin_file: LinFileFactory
    ) -> None:
        """A detected group renders a table row and the crossing badge."""
        result = _study(make_lin_file, branch_review=True)
        # Inject a hotspot referencing the study's first track so the populated
        # table and figure paths render (the synthetic system has no crossing).
        result.pipeline.identification.branch_groups = [
            BranchGroup(
                track_indices=(0,),
                op_lo=0,
                op_hi=1,
                frequency_band_hz=(1.0, 2.0),
                kind=BranchKind.CROSSING,
                min_separation_hz=0.05,
                min_separation_op=0,
                sub_windows=(BranchSubWindow(0, 1, BranchKind.CROSSING, False),),
                is_dense=False,
            )
        ]
        report = study_report_html(result)
        assert '<span class="flag">crossing</span>' in report
        assert "warrants manual review" in report
