# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""High-level study orchestration with reproducibility provenance.

A study discovers the linearization files of a campaign, runs the modal pipeline, and
records the **provenance** of the result — the exact source files (with content
hashes) grouped by operating point, each point's case name, full azimuth list, and
operating-parameter value, the tuning thresholds, and the library and environment
versions — so a Campbell diagram can be tied back to precisely the inputs and
assumptions that produced it. ``write_bundle`` serializes a reproducibility package (a
JSON manifest plus the result tables).
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, overload

from vane._version import vane_version
from vane.campbell.excitation import DEFAULT_HARMONICS
from vane.export import campbell_table, write_table
from vane.io.lin_reader import read_lin_file
from vane.pipeline import (
    DEFAULT_PARAMETER_RTOL,
    ModalPipeline,
    _validate_parameter_name,
    _validate_parameter_rtol,
    _validate_sweep_axis,
)

if TYPE_CHECKING:
    from vane.io.lin_reader import LinFile
    from vane.modal.identifier import TrackingStrategy
    from vane.pipeline import PipelineResult

__all__ = [
    "DiscoveredOperatingPoint",
    "Environment",
    "OperatingPointProvenance",
    "Provenance",
    "SourceFile",
    "StudyResult",
    "discover_operating_points",
    "run_study",
]

logger = logging.getLogger(__name__)

_RPM_PARAMETER = "rotor_speed_rpm"
_WIND_PARAMETER = "wind_speed"
_DEG_PER_RAD = 180.0 / 3.141592653589793
_RPM_PER_RAD_S = 30.0 / 3.141592653589793
_HASH_CHUNK_BYTES = 65536
# Dependencies whose versions are recorded for reproducibility.
_RECORDED_DEPENDENCIES = (
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "plotly",
)


@dataclass(frozen=True)
class SourceFile:
    """A source linearization file and the hash of its contents.

    Parameters
    ----------
    path : str
        Path the file was read from.
    sha256 : str
        Hex-encoded SHA-256 digest of the file's bytes.
    """

    path: str
    sha256: str


@dataclass(frozen=True)
class DiscoveredOperatingPoint(Sequence["LinFile"]):
    """A discovered operating point: its case name and its azimuth files.

    Behaves as a sequence of its :class:`~vane.io.lin_reader.LinFile` objects, so it
    can be passed wherever a sequence of linearization files is expected, while also
    carrying the ``<case>`` grouping name that discovery would otherwise lose.

    Parameters
    ----------
    name : str
        The ``<case>`` root the files were grouped under.
    files : tuple[LinFile, ...]
        The parsed azimuth files of this operating point.
    """

    name: str
    files: tuple[LinFile, ...]

    @overload
    def __getitem__(self, index: int) -> LinFile: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[LinFile, ...]: ...

    def __getitem__(self, index: int | slice) -> LinFile | tuple[LinFile, ...]:
        """Return the file(s) at ``index``."""
        return self.files[index]

    def __len__(self) -> int:
        """Return the number of azimuth files."""
        return len(self.files)


@dataclass(frozen=True)
class Environment:
    """The software environment a study ran in.

    Parameters
    ----------
    python_version : str
        The CPython version (e.g. ``"3.11.9"``).
    platform : str
        The platform identification string.
    dependencies : dict[str, str]
        Version of each recorded scientific dependency.
    """

    python_version: str
    platform: str
    dependencies: dict[str, str]


@dataclass(frozen=True)
class OperatingPointProvenance:
    """Provenance of one operating point.

    Parameters
    ----------
    name : str
        The ``<case>`` grouping name (empty if the point was not discovered by name).
    n_azimuths : int
        Number of azimuth samples in the operating point.
    azimuths_deg : tuple[float, ...]
        Every sampled azimuth, in degrees.
    azimuth_min_deg, azimuth_max_deg : float
        Smallest and largest sampled azimuth, in degrees.
    rotor_speed_rpm : float
        Azimuth-averaged rotor speed, in rev/min.
    wind_speed : float
        Azimuth-averaged hub-height wind speed, in m/s.
    parameter_value : float
        The operating-parameter value used to order the sweep.
    source_files : tuple[SourceFile, ...]
        The files of this operating point, each with its content hash.
    """

    name: str
    n_azimuths: int
    azimuths_deg: tuple[float, ...]
    azimuth_min_deg: float
    azimuth_max_deg: float
    rotor_speed_rpm: float
    wind_speed: float
    parameter_value: float
    source_files: tuple[SourceFile, ...]


@dataclass(frozen=True)
class Provenance:
    """A reproducibility record for a study run.

    Parameters
    ----------
    vane_version : str
        Version of the library that produced the result.
    created_at : str
        ISO-8601 timestamp of when the provenance was recorded.
    parameter_name : str
        Operating parameter the sweep was run against.
    frequency_weight, mac_threshold : float
        Tracking tuning thresholds used.
    tracking_strategy : str
        Cross-operating-point tracking strategy used (``"single_path"`` or
        ``"simultaneous"``).
    branch_review : bool
        Whether the branch-review pass was run.
    resolve : bool
        Whether the smooth (minimum-curvature) alternative was computed for eligible
        branch groups.
    harmonics : tuple[int, ...]
        Excitation harmonics used for resonance detection.
    environment : Environment
        The Python, platform, and dependency versions.
    operating_points : tuple[OperatingPointProvenance, ...]
        Per-operating-point provenance, in the (sorted) order of the result.
    n_tracks, n_resonances, n_branch_groups : int
        Number of identified mode tracks, detected resonance crossings, and
        contested branch groups (the last is ``0`` unless ``branch_review`` was run).
    parameter_rtol : float, optional
        Relative plateau tolerance the sweep axis was validated with, so the run can be
        reproduced with the same acceptance (default
        :data:`vane.pipeline.DEFAULT_PARAMETER_RTOL`).
    """

    vane_version: str
    created_at: str
    parameter_name: str
    frequency_weight: float
    mac_threshold: float
    tracking_strategy: str
    branch_review: bool
    resolve: bool
    harmonics: tuple[int, ...]
    environment: Environment
    operating_points: tuple[OperatingPointProvenance, ...]
    n_tracks: int
    n_resonances: int
    n_branch_groups: int
    parameter_rtol: float = DEFAULT_PARAMETER_RTOL

    @property
    def source_files(self) -> tuple[SourceFile, ...]:
        """Return every source file across all operating points (flattened)."""
        return tuple(
            source for point in self.operating_points for source in point.source_files
        )

    def to_dict(self) -> dict[str, object]:
        """Return the provenance as a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class StudyResult:
    """The pipeline result of a study together with its provenance.

    Parameters
    ----------
    pipeline : PipelineResult
        Every intermediate product of the analysis.
    provenance : Provenance
        The reproducibility record.
    """

    pipeline: PipelineResult
    provenance: Provenance

    def write_bundle(self, output_dir: str | Path, *, report: bool = True) -> None:
        """Write a reproducibility bundle to ``output_dir``.

        The bundle contains ``provenance.json`` (the manifest of inputs and
        assumptions), ``campbell.csv`` (the tracked-mode result table), and, unless
        disabled, a self-contained ``report.html`` audit dashboard.

        Parameters
        ----------
        output_dir : str or pathlib.Path
            Destination directory; created if it does not exist.
        report : bool, optional
            Also write the HTML audit report (``report.html``). Default ``True``.
        """
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        manifest = json.dumps(self.provenance.to_dict(), indent=2)
        (destination / "provenance.json").write_text(manifest, encoding="utf-8")
        write_table(
            campbell_table(self.pipeline.campbell), destination / "campbell.csv"
        )
        if report:
            # Imported here to keep the visualization (Plotly) dependency out of the
            # import path of callers that only need the data products.
            from vane.viz.report import write_report

            write_report(self, destination / "report.html")


# OpenFAST glue-code module abbreviations. With ``LinOutMod = True`` each module
# also writes its own linearization as ``<case>.<index>.<Module>.lin`` (e.g.
# ``case.1.ED.lin``; multi-instance modules append the instance, ``case.1.BD1.lin``).
# Those describe a single module's sub-system, not the coupled turbine, and must never
# be ingested as operating points — the same exclusion the OpenFAST runner applies
# when it collects full-system files.
_OPENFAST_LIN_MODULES = frozenset(
    {
        "AD",
        "ADsk",
        "BD",
        "ED",
        "ExtInfw",
        "ExtLd",
        "ExtPtfm",
        "FEAM",
        "HD",
        "IceD",
        "IceF",
        "IfW",
        "MAP",
        "MD",
        "Orca",
        "SD",
        "SEA",
        "SED",
        "SeaSt",
        "SlD",
        "SrvD",
    }
)
# ``<case>.<index>.<Module>.lin`` — OpenFAST's per-module (LinOutMod) naming. A name
# whose final segment is an alphabetic token (plus an optional instance number) after
# an integer index can never be a full-system file, whatever the token: OpenFAST keeps
# adding modules (``SEA``, ``SED``, ``ExtLd``, ...), so this shape is excluded outright
# rather than checked against ``_OPENFAST_LIN_MODULES``.
_OPENFAST_MODULE_LIN_RE = re.compile(
    r"\.\d+\.(?P<module>[A-Za-z][A-Za-z]*?)(?P<instance>\d*)\.lin$"
)
# ``<case>.<Module>.<index>.lin`` — a module token between the case root and the
# integer azimuth index (a full-system output of a case whose root ends in a module
# token, or a module file under a non-OpenFAST naming).
_MODULE_LIN_RE = re.compile(r"\.(?P<module>[A-Za-z][A-Za-z0-9]*)\.(?P<index>\d+)\.lin$")
# ``<case>.<index>.lin`` — a full-system file (no module token before the index).
_FULL_SYSTEM_LIN_RE = re.compile(r"\.(?P<index>\d+)\.lin$")


def _module_sidecar_files(lin_files: list[Path], root_dir: Path) -> set[Path]:
    """Return the per-module (``LinOutMod``) sidecar files to exclude.

    OpenFAST's own per-module files, ``<root>.<index>.<Module>.lin``, are always
    excluded whatever the module token (``ED``, ``SEA``, ``ExtLd``, ``BD1``, ...):
    that name can never be a full-system output.

    A ``<root>.<Module>.<index>.lin`` group is treated as a sidecar of ``<root>`` only
    when **every** one of its azimuth indices has a full-system twin
    ``<root>.<index>.lin``. Deciding per group (rather than per file) keeps the result
    all-or-nothing — a real ``<root>.<Module>`` operating point is never split into a
    partial sweep — and a case whose root genuinely ends in a module token (e.g.
    ``case.ED.fst``) is preserved, because at least one of its indices has no twin.

    The residual case is genuinely undecidable from filenames alone: a directory holding
    exactly ``case.1.lin`` and ``case.ED.1.lin`` is either one ``LinOutMod`` run or two
    separate runs. The common ``LinOutMod`` reading is taken (the file is excluded), and
    :func:`discover_operating_points` logs a warning so the choice is never silent.
    """
    full_system_indices: dict[str, set[str]] = defaultdict(set)
    module_groups: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    always_excluded: set[Path] = set()
    for path in lin_files:
        rel_name = path.relative_to(root_dir).as_posix()
        if _OPENFAST_MODULE_LIN_RE.search(rel_name) is not None:
            always_excluded.add(path)
            continue
        match = _MODULE_LIN_RE.search(rel_name)
        if match is not None and match.group("module") in _OPENFAST_LIN_MODULES:
            root = rel_name[: match.start()]
            module_groups[(root, match.group("module"))][match.group("index")] = (
                rel_name
            )
            continue
        full = _FULL_SYSTEM_LIN_RE.search(rel_name)
        if full is not None:
            root = rel_name[: full.start()]
            full_system_indices[root].add(full.group("index"))

    exclude: set[Path] = set(always_excluded)
    for (root, _module), indexed_names in module_groups.items():
        if set(indexed_names) <= full_system_indices.get(root, set()):
            for path in lin_files:
                rel_name = path.relative_to(root_dir).as_posix()
                if rel_name in indexed_names.values():
                    exclude.add(path)
    return exclude


def discover_operating_points(directory: str | Path) -> list[DiscoveredOperatingPoint]:
    """Discover and group a directory's ``.lin`` files into operating points.

    Files are named ``<case>.<index>.lin`` by OpenFAST; those sharing a ``<case>``
    root are one operating point's azimuth sweep, and the case name is retained on the
    returned :class:`DiscoveredOperatingPoint`. Per-module files written under
    ``LinOutMod = True`` (OpenFAST names them ``<case>.<index>.<Module>.lin``, e.g.
    ``case.1.ED.lin``, ``case.1.SEA.lin``, ``case.1.BD1.lin``) are always
    **excluded**, whatever the alphabetic module token. Files named
    ``<case>.<Module>.<index>.lin`` (e.g. ``case.ED.1.lin``) are excluded when every
    index of the ``<case>.<Module>`` group has a full-system twin
    ``<case>.<index>.lin``: they describe a single module's sub-system, not the
    coupled turbine, so they must not be treated as operating points. A module-like
    group without a full set of twins is kept, since the case root then genuinely ends
    in that token (e.g. a separate ``case.ED.fst`` run). See
    :func:`_module_sidecar_files` for the residual ambiguity, excluded with a warning.

    Parameters
    ----------
    directory : str or pathlib.Path
        Directory containing ``.lin`` linearization files.

    Returns
    -------
    list[DiscoveredOperatingPoint]
        One named operating point per ``<case>`` root, ordered by case.

    Raises
    ------
    FileNotFoundError
        If the directory contains no full-system ``.lin`` files.
    """
    root_dir = Path(directory)
    lin_files = sorted(root_dir.rglob("*.lin"))
    excluded = _module_sidecar_files(lin_files, root_dir)
    if excluded:
        logger.warning(
            "Excluding %d per-module (LinOutMod) file(s) from discovery: %s. If these "
            "are instead full-system outputs of separate runs whose case name ends in "
            "a module token, keep them in their own directory.",
            len(excluded),
            ", ".join(
                sorted(path.relative_to(root_dir).as_posix() for path in excluded)
            ),
        )
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in lin_files:
        if path in excluded:
            continue
        root = path.relative_to(root_dir).as_posix().rsplit(".", 2)[0]
        groups[root].append(path)
    if not groups:
        msg = f"No full-system .lin files found in {directory}"
        raise FileNotFoundError(msg)
    return [
        DiscoveredOperatingPoint(
            name=root, files=tuple(read_lin_file(path) for path in groups[root])
        )
        for root in sorted(groups)
    ]


def run_study(
    operating_points: Sequence[Sequence[LinFile]],
    *,
    parameter_name: str = _RPM_PARAMETER,
    parameter_rtol: float = DEFAULT_PARAMETER_RTOL,
    frequency_weight: float = 0.5,
    mac_threshold: float = 0.5,
    harmonics: Sequence[int] = DEFAULT_HARMONICS,
    tracking_strategy: TrackingStrategy = "single_path",
    branch_review: bool = False,
    resolve: bool = False,
    timestamp: datetime | None = None,
) -> StudyResult:
    """Run the modal pipeline and record the provenance of the result.

    Parameters
    ----------
    operating_points : Sequence[Sequence[LinFile]]
        One azimuth sweep per operating point (e.g. from
        :func:`discover_operating_points`).
    parameter_name : str, optional
        Operating parameter to run against (``"rotor_speed_rpm"`` or
        ``"wind_speed"``). A sweep that extends past rated, where the trimmed rotor
        speed is constant, must use ``"wind_speed"``.
    parameter_rtol : float, optional
        Relative plateau tolerance of the sweep axis (see
        :meth:`vane.pipeline.ModalPipeline.run`). Default
        :data:`vane.pipeline.DEFAULT_PARAMETER_RTOL`.
    frequency_weight, mac_threshold : float, optional
        Tracking tuning thresholds.
    harmonics : Sequence[int], optional
        Excitation harmonics for resonance detection.
    tracking_strategy : {"single_path", "simultaneous"}, optional
        Cross-operating-point tracking strategy; recorded in the provenance. See
        :func:`vane.modal.identifier.identify_modes`.
    branch_review : bool, optional
        When ``True``, flag crossing/veering hotspots for audit; recorded in the
        provenance and rendered in the report. Default ``False``.
    resolve : bool, optional
        When ``True``, also compute the smooth (minimum-curvature) alternative for
        eligible hotspots (implies ``branch_review``). Default ``False``.
    timestamp : datetime or None, optional
        Timestamp to record (defaults to the current UTC time); injectable for
        reproducible tests.

    Returns
    -------
    StudyResult
        The pipeline result and its provenance.

    Raises
    ------
    ValueError
        If ``parameter_name`` is not a known operating parameter,
        ``parameter_rtol`` is invalid, or the operating-parameter values are
        non-finite or not distinct beyond ``parameter_rtol`` (a plateau, such as
        the constant rotor speed above rated; use ``"wind_speed"`` there).
    """
    _validate_parameter_name(parameter_name)
    _validate_parameter_rtol(parameter_rtol)
    if all(operating_points):
        # Reject a plateaued sweep axis before any MBC work (the pipeline repeats
        # the same check on its own averages); an empty point is left for the
        # pipeline to report.
        _validate_sweep_axis(
            [_operating_parameter(point, parameter_name) for point in operating_points],
            parameter_name,
            rtol=parameter_rtol,
            wind_speeds=[
                _operating_parameter(point, _WIND_PARAMETER)
                for point in operating_points
            ],
        )

    # The pipeline sorts operating points by the operating parameter; sort the same
    # way up front so the provenance order matches the sorted Campbell result.
    ordered = sorted(
        operating_points,
        key=lambda point: _operating_parameter(point, parameter_name),
    )
    pipeline = ModalPipeline(
        frequency_weight=frequency_weight,
        mac_threshold=mac_threshold,
        harmonics=harmonics,
        tracking_strategy=tracking_strategy,
        branch_review=branch_review,
        resolve=resolve,
    )
    result = pipeline.run(
        ordered, parameter_name=parameter_name, parameter_rtol=parameter_rtol
    )
    provenance = _build_provenance(
        ordered,
        result,
        parameter_name=parameter_name,
        frequency_weight=frequency_weight,
        mac_threshold=mac_threshold,
        tracking_strategy=pipeline.tracking_strategy,
        branch_review=pipeline.branch_review,
        resolve=pipeline.resolve,
        # The pipeline validates and normalizes the harmonics (e.g. float -> int);
        # record the normalized values, not the raw input.
        harmonics=pipeline.harmonics,
        parameter_rtol=float(parameter_rtol),
        timestamp=timestamp,
    )
    return StudyResult(pipeline=result, provenance=provenance)


def _operating_parameter(point: Sequence[LinFile], parameter_name: str) -> float:
    """Return an operating point's parameter value, matching the pipeline's average."""
    if not point:
        return 0.0
    if parameter_name == _RPM_PARAMETER:
        mean_rotor_speed = sum(lin.rotor_speed for lin in point) / len(point)
        return mean_rotor_speed * _RPM_PER_RAD_S
    return sum(lin.wind_speed for lin in point) / len(point)


def _build_provenance(
    operating_points: Sequence[Sequence[LinFile]],
    result: PipelineResult,
    *,
    parameter_name: str,
    frequency_weight: float,
    mac_threshold: float,
    tracking_strategy: str,
    branch_review: bool,
    resolve: bool,
    harmonics: Sequence[int],
    parameter_rtol: float,
    timestamp: datetime | None,
) -> Provenance:
    """Assemble a :class:`Provenance` from the inputs and the pipeline result."""
    moment = timestamp if timestamp is not None else datetime.now(timezone.utc)
    coverage: list[OperatingPointProvenance] = []
    for point in operating_points:
        files = list(point)
        azimuths_deg = tuple(lin.azimuth * _DEG_PER_RAD for lin in files)
        mean_rpm = (
            sum(lin.rotor_speed for lin in files) / len(files) * _RPM_PER_RAD_S
            if files
            else 0.0
        )
        mean_wind = sum(lin.wind_speed for lin in files) / len(files) if files else 0.0
        coverage.append(
            OperatingPointProvenance(
                name=getattr(point, "name", ""),
                n_azimuths=len(files),
                azimuths_deg=azimuths_deg,
                azimuth_min_deg=min(azimuths_deg) if azimuths_deg else 0.0,
                azimuth_max_deg=max(azimuths_deg) if azimuths_deg else 0.0,
                rotor_speed_rpm=float(mean_rpm),
                wind_speed=float(mean_wind),
                parameter_value=_operating_parameter(files, parameter_name),
                source_files=tuple(
                    SourceFile(path=str(lin.path), sha256=_file_sha256(lin.path))
                    for lin in files
                ),
            )
        )
    return Provenance(
        vane_version=vane_version(),
        created_at=moment.isoformat(),
        parameter_name=parameter_name,
        frequency_weight=frequency_weight,
        mac_threshold=mac_threshold,
        tracking_strategy=tracking_strategy,
        branch_review=branch_review,
        resolve=resolve,
        harmonics=tuple(harmonics),
        environment=_environment(),
        operating_points=tuple(coverage),
        n_tracks=len(result.tracks),
        n_resonances=len(result.resonances),
        n_branch_groups=len(result.identification.branch_groups),
        parameter_rtol=parameter_rtol,
    )


def _package_version(package: str) -> str:
    """Return the installed version of ``package``, or ``"unknown"`` if absent."""
    try:
        return version(package)
    except PackageNotFoundError:  # pragma: no cover - package always installed
        return "unknown"


def _environment() -> Environment:
    """Capture the Python, platform, and dependency versions."""
    return Environment(
        python_version=platform.python_version(),
        platform=platform.platform(),
        dependencies={
            package: _package_version(package) for package in _RECORDED_DEPENDENCIES
        },
    )


def _file_sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file's contents."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()
