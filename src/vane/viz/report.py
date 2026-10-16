# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Self-contained HTML audit report for a study result.

Turns a :class:`~vane.study.StudyResult` into a single, shareable HTML file that pairs
the analysis with its provenance: the environment and tuning the run used, the
operating points and their hashed source files, the interactive Campbell diagram, and
tables of the tracked modes (with identification confidence) and detected resonance
crossings (with damping-aware severity). It is the auditable artifact a digital-twin or
assessment workflow can attach to a result.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import TYPE_CHECKING

from vane.campbell.excitation import ResonanceSeverity
from vane.io.mbc_transform import _MIN_AZIMUTHS_FOR_MBC, _ROTATING_RPM_TOL
from vane.viz.campbell_plot import SEVERITY_COLORS, plot_campbell
from vane.viz.diagnostics import (
    plot_azimuth_coverage,
    plot_branch_review,
    plot_tracking_confidence,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vane.campbell.builder import CampbellDiagram
    from vane.campbell.excitation import ResonanceCrossing
    from vane.modal.branch import BranchGroup
    from vane.study import OperatingPointProvenance, Provenance, StudyResult

__all__ = ["study_report_html", "write_report", "write_report_pdf"]

_STYLE = """
body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; margin: 2rem;
       color: #222; max-width: 1100px; }
h1 { margin-bottom: 0.2rem; }
.meta { color: #666; margin-top: 0; }
h2 { border-bottom: 1px solid #ddd; padding-bottom: 0.2rem; margin-top: 2rem; }
table { border-collapse: collapse; width: 100%; margin: 0.5rem 0; font-size: 0.92rem; }
th, td { border: 1px solid #ddd; padding: 4px 8px; text-align: left; }
th { background: #f4f4f4; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
code { font-family: ui-monospace, "Cascadia Code", monospace; font-size: 0.85rem; }
.badge { color: #fff; border-radius: 4px; padding: 1px 7px; font-size: 0.82rem; }
.flag { color: #b8860b; font-weight: 600; }
.pass { color: #2e7d32; font-weight: 600; }
.warn { color: #b8860b; font-weight: 600; }
.na { color: #666; font-weight: 600; }
ul.warnings li { margin: 0.2rem 0; }
"""

# Resonance crossings are evaluated only when the study sweeps rotor speed.
_RPM_PARAMETER = "rotor_speed_rpm"

# Human-readable names of the sweep axes, for the not-evaluated notes.
_AXIS_LABELS = {"wind_speed": "wind-speed", _RPM_PARAMETER: "rotor-speed"}

# A track below this identification confidence is flagged for review in the report.
_LOW_CONFIDENCE = 0.5


def study_report_html(
    result: StudyResult,
    *,
    title: str = "VANE Study Report",
    include_plotlyjs: bool | str = True,
) -> str:
    """Render a study result as a self-contained HTML report.

    Parameters
    ----------
    result : StudyResult
        The analysis and its provenance.
    title : str, optional
        Report heading.
    include_plotlyjs : bool or str, optional
        How to embed Plotly.js, passed to ``Figure.to_html``. The default ``True``
        inlines the library so the report is fully self-contained and works offline
        (the intended audit artifact); pass ``"cdn"`` for a much smaller file that
        instead loads Plotly.js from a CDN at view time.

    Returns
    -------
    str
        A complete HTML document.
    """
    provenance = result.provenance
    # Plot with the study's own harmonics so the nP lines and resonance markers match
    # the provenance and the resonance table. The Campbell figure embeds Plotly.js; the
    # diagnostic figures reuse it (include_plotlyjs=False) to avoid re-embedding it.
    campbell = plot_campbell(result.pipeline.campbell, harmonics=provenance.harmonics)
    campbell_div = campbell.to_html(full_html=False, include_plotlyjs=include_plotlyjs)
    confidence_div = plot_tracking_confidence(
        result.pipeline.campbell, title="Mode-tracking confidence"
    ).to_html(full_html=False, include_plotlyjs=False)
    coverage_div = plot_azimuth_coverage(provenance, title="Azimuth coverage").to_html(
        full_html=False, include_plotlyjs=False
    )
    checks, warnings = _audit_findings(result)
    resonances_evaluated = _resonances_evaluated(result)
    sections = "\n".join(
        section
        for section in [
            _provenance_section(provenance),
            _checks_section(checks),
            _warnings_section(warnings, resonances_evaluated=resonances_evaluated),
            _operating_points_section(provenance),
            f"<h2>Campbell diagram</h2>\n{campbell_div}",
            _modes_section(result.pipeline.campbell),
            _resonances_section(
                result.pipeline.resonances,
                evaluated=resonances_evaluated,
                parameter_name=provenance.parameter_name,
            ),
            _branch_review_section(result),
            f"<h2>Diagnostics</h2>\n{confidence_div}\n{coverage_div}",
            _sources_section(provenance),
        ]
        if section
    )
    return (
        "<!DOCTYPE html>\n"
        f'<html lang="en"><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title><style>{_STYLE}</style></head>\n"
        f"<body>\n<h1>{html.escape(title)}</h1>\n"
        f'<p class="meta">Generated {html.escape(provenance.created_at)} '
        f"&middot; VANE {html.escape(provenance.vane_version)}</p>\n"
        f"{sections}\n</body></html>\n"
    )


def write_report(
    result: StudyResult,
    output_path: str | Path,
    *,
    title: str = "VANE Study Report",
    include_plotlyjs: bool | str = True,
) -> None:
    """Write a study report to ``output_path`` as HTML.

    Parameters
    ----------
    result : StudyResult
        The analysis and its provenance.
    output_path : str or pathlib.Path
        Destination ``.html`` file; parent directories are created.
    title : str, optional
        Report heading.
    include_plotlyjs : bool or str, optional
        Passed through to :func:`study_report_html`; defaults to ``True`` (inline, so
        the written file is fully self-contained).
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        study_report_html(result, title=title, include_plotlyjs=include_plotlyjs),
        encoding="utf-8",
    )


def write_report_pdf(
    result: StudyResult,
    output_path: str | Path,
    *,
    title: str = "VANE Study Report",
) -> Path:
    """Write the study report to ``output_path`` as PDF (requires WeasyPrint).

    Renders the report's auditable content — the heading and run metadata, the
    pass/review audit findings, the provenance (environment, operating points and
    their hashed source files), and the tracked-mode and resonance tables — to a
    paginated PDF via the optional `WeasyPrint <https://weasyprint.org>`_ engine. This
    is the print-ready, archivable companion to the interactive HTML report; because
    WeasyPrint renders HTML/CSS and does not execute JavaScript, the **interactive
    Campbell figure** is available in the HTML report (:func:`write_report`), not the
    PDF.

    Parameters
    ----------
    result : StudyResult
        The analysis and its provenance.
    output_path : str or pathlib.Path
        Destination ``.pdf`` file; parent directories are created.
    title : str, optional
        Report heading.

    Returns
    -------
    pathlib.Path
        The PDF file written.

    Raises
    ------
    ValueError
        If WeasyPrint is not installed (``pip install "vane[pdf]"``).
    """
    try:
        from weasyprint import HTML
    except ImportError as exc:
        msg = (
            "PDF report export requires the optional 'weasyprint' package; "
            'install it with: pip install "vane[pdf]"'
        )
        raise ValueError(msg) from exc

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # No inline plotly.js — the JS figures cannot render in WeasyPrint, and omitting the
    # library keeps the PDF source small; the auditable text/tables render fully.
    document = study_report_html(result, title=title, include_plotlyjs=False)
    HTML(string=document).write_pdf(str(destination))
    return destination


def _resonances_evaluated(result: StudyResult) -> bool:
    """Return whether the study's sweep axis supports resonance detection.

    Resonance crossings are located against the per-rev excitation lines, which
    need the rotor speed as the sweep parameter; a study parameterized by any
    other quantity (wind speed) never evaluates them, so an empty crossing list
    means "not evaluated", not "no resonance".
    """
    return result.provenance.parameter_name == _RPM_PARAMETER


def _sweep_axis_label(parameter_name: str) -> str:
    """Return a short, HTML-safe, human-readable name for a sweep parameter."""
    return html.escape(_AXIS_LABELS.get(parameter_name, parameter_name))


def _audit_findings(
    result: StudyResult,
) -> tuple[list[tuple[str, bool | None, str]], list[str]]:
    """Compute the pass/review checks and the specific warnings for a result.

    Returns
    -------
    tuple
        ``(checks, warnings)`` where ``checks`` is a list of
        ``(name, passed, detail)`` and ``warnings`` is a list of HTML-safe strings.
        ``passed`` is ``None`` for a check that was not evaluated (resonance
        severity on a sweep axis other than rotor speed); ``detail`` then says why.
    """
    tracks = result.pipeline.campbell.tracks
    resonances = result.pipeline.resonances
    points = result.provenance.operating_points
    low_confidence = [t for t in tracks if t.confidence < _LOW_CONFIDENCE]
    ambiguous = [t for t in tracks if t.is_ambiguous]
    high_severity = [r for r in resonances if r.severity is ResonanceSeverity.HIGH]
    # Flag exactly the condition the MBC layer warns about: a point where the MBC
    # transform was actually applied (blade triplets present), the rotor is spinning,
    # and the azimuth count is below _MIN_AZIMUTHS_FOR_MBC — so the report never looks
    # cleaner than the numerics warrant, and never flags a point MBC skipped (a
    # standstill point, or a fixed-frame study with no triplets).
    thin_coverage = [
        point
        for point, mbc in zip(points, result.pipeline.mbc_results, strict=True)
        if mbc.performed_transformation
        and abs(point.rotor_speed_rpm) >= _ROTATING_RPM_TOL
        and point.n_azimuths < _MIN_AZIMUTHS_FOR_MBC
    ]

    resonance_status: bool | None = not high_severity
    resonance_detail = f"{len(high_severity)} high-severity crossing(s)"
    if not _resonances_evaluated(result):
        resonance_status = None
        axis = _sweep_axis_label(result.provenance.parameter_name)
        resonance_detail = f"Not evaluated ({axis} sweep axis)"

    checks: list[tuple[str, bool | None, str]] = [
        (
            "Tracking confidence",
            not low_confidence,
            f"{len(low_confidence)} of {len(tracks)} tracks below "
            f"{_LOW_CONFIDENCE:.2f}",
        ),
        ("Track ambiguity", not ambiguous, f"{len(ambiguous)} ambiguous track(s)"),
        ("Resonance severity", resonance_status, resonance_detail),
        (
            "Azimuth coverage",
            not thin_coverage,
            f"{len(thin_coverage)} rotating operating point(s) with fewer than "
            f"{_MIN_AZIMUTHS_FOR_MBC} azimuths",
        ),
    ]

    warnings: list[str] = []
    warnings.extend(
        f"Mode &ldquo;{html.escape(track.label.label)}&rdquo; has low "
        f"identification confidence ({track.confidence:.2f})."
        for track in low_confidence
    )
    warnings.extend(
        f"Mode &ldquo;{html.escape(track.label.label)}&rdquo; is ambiguous "
        "(a possible crossing or veering)."
        for track in ambiguous
    )
    warnings.extend(
        f"High-severity {crossing.harmonic}P resonance at "
        f"{crossing.rotor_speed_rpm:.4g} rpm ({crossing.frequency_hz:.4g} Hz, "
        f"damping {crossing.damping_ratio:.3g})."
        for crossing in high_severity
    )
    warnings.extend(
        f"Rotating operating point "
        f"{html.escape(point.name) if point.name else '(unnamed)'} "
        f"({point.rotor_speed_rpm:.4g} rpm) has only {point.n_azimuths} azimuth(s) "
        f"(fewer than {_MIN_AZIMUTHS_FOR_MBC}); the MBC azimuth average may be "
        "unreliable."
        for point in thin_coverage
    )
    if result.provenance.branch_review:
        groups = result.pipeline.identification.branch_groups
        checks.append(
            (
                "Branch review",
                not groups,
                f"{len(groups)} contested branch group(s)",
            )
        )
        warnings.extend(_branch_warnings(result, groups))
    return checks, warnings


def _branch_warnings(result: StudyResult, groups: Sequence[BranchGroup]) -> list[str]:
    """Render one review warning per contested branch group."""
    tracks = result.pipeline.campbell.tracks
    parameter_values = result.pipeline.campbell.parameter_values
    parameter_name = html.escape(result.provenance.parameter_name)
    warnings: list[str] = []
    for group in groups:
        members = ", ".join(
            f"&ldquo;{html.escape(tracks[i].label.label)}&rdquo;"
            for i in group.track_indices
        )
        value = float(parameter_values[group.min_separation_op])
        warnings.append(
            f"A {group.kind.value} among modes {members} near {parameter_name}="
            f"{value:.4g} (closest approach {group.min_separation_hz:.3g} Hz) "
            "warrants manual review."
        )
    return warnings


def _branch_review_section(result: StudyResult) -> str:
    """Render the branch-review table and figure, or nothing if the pass was off."""
    if not result.provenance.branch_review:
        return ""
    diagram = result.pipeline.campbell
    groups = result.pipeline.identification.branch_groups
    figure_div = plot_branch_review(diagram, groups).to_html(
        full_html=False, include_plotlyjs=False
    )
    if not groups:
        return (
            "<h2>Branch review</h2>\n<p>None &mdash; no crossing or veering hotspots "
            f"were detected.</p>\n{figure_div}"
        )
    header = (
        f"<tr><th>Modes</th><th>Window ({html.escape(diagram.parameter_name)})</th>"
        "<th>Kind</th><th>Min &Delta;f (Hz)</th><th>Dense</th><th>Resolved</th></tr>"
    )
    rows = "\n".join(_branch_row(diagram, group) for group in groups)
    return f"<h2>Branch review</h2>\n<table>\n{header}\n{rows}\n</table>\n{figure_div}"


def _branch_row(diagram: CampbellDiagram, group: BranchGroup) -> str:
    """Render one branch-group table row."""
    members = ", ".join(
        html.escape(diagram.tracks[i].label.label) for i in group.track_indices
    )
    low = float(diagram.parameter_values[group.op_lo])
    high = float(diagram.parameter_values[group.op_hi])
    badge = (
        '<span class="flag">crossing</span>'
        if group.kind.value == "crossing"
        else "veering"
    )
    return (
        f"<tr><td>{members}</td><td>{low:.4g} &ndash; {high:.4g}</td>"
        f"<td>{badge}</td><td class='num'>{group.min_separation_hz:.3g}</td>"
        f"<td>{'yes' if group.is_dense else 'no'}</td>"
        f"<td>{'yes' if group.resolved else 'no'}</td></tr>"
    )


def _checks_section(checks: list[tuple[str, bool | None, str]]) -> str:
    """Render the validation-check table with pass/review/not-evaluated badges."""
    header = "<tr><th>Check</th><th>Status</th><th>Detail</th></tr>"
    rows = []
    for name, passed, detail in checks:
        if passed is None:
            badge = '<span class="na">Not evaluated</span>'
        elif passed:
            badge = '<span class="pass">&#10003; Pass</span>'
        else:
            badge = '<span class="warn">&#9888; Review</span>'
        note = "ok" if passed else detail
        rows.append(f"<tr><td>{name}</td><td>{badge}</td><td>{note}</td></tr>")
    return f"<h2>Checks</h2>\n<table>\n{header}\n" + "\n".join(rows) + "\n</table>"


def _warnings_section(warnings: list[str], *, resonances_evaluated: bool) -> str:
    """Render the specific warnings, or a clean-bill-of-health note."""
    if not warnings:
        resonance_note = (
            "there are no high-severity resonances"
            if resonances_evaluated
            else "resonances were not evaluated on this sweep axis"
        )
        return (
            "<h2>Warnings</h2>\n<p>None &mdash; all tracks are confident and "
            f"unambiguous, azimuth coverage is adequate, and {resonance_note}.</p>"
        )
    items = "\n".join(f"<li>{warning}</li>" for warning in warnings)
    return f'<h2>Warnings</h2>\n<ul class="warnings">\n{items}\n</ul>'


def _branch_review_summary(provenance: Provenance) -> str:
    """Summarize the branch-review (and resolution) provenance for one table row."""
    if not provenance.branch_review:
        return "off"
    summary = f"on, {provenance.n_branch_groups} contested group(s)"
    if provenance.resolve:
        summary += ", smooth alternative computed"
    return summary


def _provenance_section(provenance: Provenance) -> str:
    """Render the environment and tuning provenance."""
    environment = provenance.environment
    dependencies = ", ".join(
        f"{html.escape(name)} {html.escape(ver)}"
        for name, ver in environment.dependencies.items()
    )
    harmonics = ", ".join(f"{harmonic}P" for harmonic in provenance.harmonics) or "none"
    rows = [
        ("Operating parameter", html.escape(provenance.parameter_name)),
        ("Excitation harmonics", harmonics),
        (
            "Tuning",
            f"frequency_weight={provenance.frequency_weight:g}, "
            f"mac_threshold={provenance.mac_threshold:g}, "
            f"tracking_strategy={html.escape(provenance.tracking_strategy)}",
        ),
        (
            "Tracks / resonances",
            f"{provenance.n_tracks} / "
            + (
                str(provenance.n_resonances)
                if provenance.parameter_name == _RPM_PARAMETER
                else "not evaluated"
            ),
        ),
        ("Branch review", _branch_review_summary(provenance)),
        ("Python", html.escape(environment.python_version)),
        ("Platform", html.escape(environment.platform)),
        ("Dependencies", dependencies),
    ]
    body = "\n".join(
        f"<tr><th>{label}</th><td>{value}</td></tr>" for label, value in rows
    )
    return f"<h2>Provenance</h2>\n<table>\n{body}\n</table>"


def _operating_points_section(provenance: Provenance) -> str:
    """Render the per-operating-point coverage table."""
    header = (
        "<tr><th>Operating point</th><th>Parameter value</th><th>Azimuths</th>"
        "<th>Rotor speed (rpm)</th><th>Wind speed (m/s)</th><th>Files</th></tr>"
    )
    rows = "\n".join(
        _operating_point_row(point) for point in provenance.operating_points
    )
    return f"<h2>Operating points</h2>\n<table>\n{header}\n{rows}\n</table>"


def _operating_point_row(point: OperatingPointProvenance) -> str:
    """Render one operating-point row."""
    name = html.escape(point.name) if point.name else "&mdash;"
    return (
        f"<tr><td>{name}</td>"
        f'<td class="num">{point.parameter_value:.4g}</td>'
        f'<td class="num">{point.n_azimuths}</td>'
        f'<td class="num">{point.rotor_speed_rpm:.4g}</td>'
        f'<td class="num">{point.wind_speed:.4g}</td>'
        f'<td class="num">{len(point.source_files)}</td></tr>'
    )


def _modes_section(diagram: CampbellDiagram) -> str:
    """Render the tracked-mode table with confidence and ambiguity flags."""
    header = (
        "<tr><th>Mode</th><th>Frequency range (Hz)</th><th>Mean damping</th>"
        "<th>Confidence</th><th>Ambiguous</th></tr>"
    )
    rows = []
    for track in diagram.tracks:
        _, frequency, damping = diagram.track_curve(track)
        freq_range = (
            f"{float(frequency.min()):.4g} &ndash; {float(frequency.max()):.4g}"
            if frequency.size
            else "&mdash;"
        )
        mean_damping = f"{float(damping.mean()):.3g}" if damping.size else "&mdash;"
        ambiguous = '<span class="flag">yes</span>' if track.is_ambiguous else "no"
        rows.append(
            f"<tr><td>{html.escape(track.label.label)}</td>"
            f"<td>{freq_range}</td>"
            f'<td class="num">{mean_damping}</td>'
            f'<td class="num">{track.confidence:.2f}</td>'
            f"<td>{ambiguous}</td></tr>"
        )
    return (
        f"<h2>Tracked modes</h2>\n<table>\n{header}\n" + "\n".join(rows) + "\n</table>"
    )


def _resonances_section(
    resonances: list[ResonanceCrossing],
    *,
    evaluated: bool = True,
    parameter_name: str = _RPM_PARAMETER,
) -> str:
    """Render the resonance-crossing table, with a coloured severity badge.

    When ``evaluated`` is false (a sweep axis other than rotor speed) the section
    states that resonance detection was not run instead of reporting a clean
    result.
    """
    if not evaluated:
        axis = _sweep_axis_label(parameter_name)
        return (
            "<h2>Resonance crossings</h2>\n<p>Not evaluated &mdash; this study is "
            f"swept along the {axis} axis. Resonance crossings with the per-rev "
            "excitation lines are detected only for rotor-speed sweeps; re-run the "
            "study with <code>parameter_name=&quot;rotor_speed_rpm&quot;</code> to "
            "assess them.</p>"
        )
    if not resonances:
        return "<h2>Resonance crossings</h2>\n<p>No resonance crossings detected.</p>"
    header = (
        "<tr><th>Excitation</th><th>Rotor speed (rpm)</th><th>Frequency (Hz)</th>"
        "<th>Damping</th><th>Severity</th></tr>"
    )
    rows = []
    for crossing in resonances:
        color = SEVERITY_COLORS[crossing.severity]
        badge = (
            f'<span class="badge" style="background:{color}">'
            f"{html.escape(crossing.severity.value)}</span>"
        )
        rows.append(
            f"<tr><td>{crossing.harmonic}P</td>"
            f'<td class="num">{crossing.rotor_speed_rpm:.4g}</td>'
            f'<td class="num">{crossing.frequency_hz:.4g}</td>'
            f'<td class="num">{crossing.damping_ratio:.3g}</td>'
            f"<td>{badge}</td></tr>"
        )
    return (
        "<h2>Resonance crossings</h2>\n<table>\n"
        f"{header}\n" + "\n".join(rows) + "\n</table>"
    )


def _sources_section(provenance: Provenance) -> str:
    """Render the hashed source-file manifest."""
    header = "<tr><th>Source file</th><th>SHA-256</th></tr>"
    rows = "\n".join(
        f"<tr><td><code>{html.escape(source.path)}</code></td>"
        f"<td><code>{html.escape(source.sha256)}</code></td></tr>"
        for source in provenance.source_files
    )
    return f"<h2>Source files</h2>\n<table>\n{header}\n{rows}\n</table>"
