# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Command-line interface for VANE.

Exposes the high-level analysis as a ``vane`` console command so a linearization
sweep can be post-processed without writing Python:

* ``vane modes`` — the modal solution for a single operating point's ``.lin`` files;
* ``vane campbell`` — the tracked Campbell diagram across a directory of operating
  points;
* ``vane study`` — a full provenance-tracked study written as a reproducibility bundle
  (``provenance.json`` + ``campbell.csv`` + an HTML audit report).

Each command is a thin wrapper over the same public API used from Python
(:class:`~vane.ModalPipeline`, :func:`~vane.run_study`,
:func:`~vane.discover_operating_points`, and :mod:`vane.export`), so the CLI adds
convenience without a second code path. Known input errors are reported as a concise
message on stderr with a non-zero exit code rather than a traceback.
"""

from __future__ import annotations

from contextlib import contextmanager
from enum import Enum
from pathlib import Path  # noqa: TC003  # Typer resolves command annotations at runtime
from typing import TYPE_CHECKING, Annotated

import typer

from vane import __version__
from vane.export import campbell_table, modes_table, write_table
from vane.io import mbc3_transform, read_lin_file
from vane.modal import modes_from_mbc
from vane.pipeline import ModalPipeline
from vane.study import discover_operating_points, run_study

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pandas as pd

app = typer.Typer(
    name="vane",
    help="Automated OpenFAST linearization post-processing and mode identification.",
    no_args_is_help=True,
    add_completion=False,
)


class Parameter(str, Enum):
    """Operating parameter to analyse a sweep against."""

    rpm = "rpm"
    wind = "wind"


class Tracking(str, Enum):
    """Cross-operating-point mode-tracking strategy."""

    single_path = "single_path"
    simultaneous = "simultaneous"


_PARAMETER_NAMES = {
    Parameter.rpm: "rotor_speed_rpm",
    Parameter.wind: "wind_speed",
}


@contextmanager
def _handle_errors() -> Iterator[None]:
    """Turn known input errors into a concise stderr message and exit code 1."""
    try:
        yield
    except (ValueError, FileNotFoundError, OSError) as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


def _emit_table(table: pd.DataFrame, output: Path | None) -> None:
    """Print the table, or write it to ``output`` (format chosen by suffix)."""
    if output is None:
        typer.echo(table.to_string(index=False))
    else:
        write_table(table, output)
        typer.echo(f"Wrote {len(table)} rows to {output}")


def _version_callback(value: bool) -> None:
    """Print the version and exit when ``--version`` is given."""
    if value:
        typer.echo(__version__)
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the VANE version and exit.",
        ),
    ] = False,
) -> None:
    """Automated OpenFAST linearization post-processing and mode identification."""


@app.command()
def modes(
    lin_files: Annotated[
        list[Path],
        typer.Argument(
            exists=True,
            dir_okay=False,
            help="The .lin files of one operating point's azimuth sweep.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Write the table here (.csv/.json/.xlsx/.mat by suffix) "
            "instead of stdout.",
        ),
    ] = None,
) -> None:
    """Compute the modal solution for one operating point's linearization files."""
    with _handle_errors():
        solution = modes_from_mbc(mbc3_transform([read_lin_file(p) for p in lin_files]))
        _emit_table(modes_table(solution), output)


@app.command()
def campbell(
    directory: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            help="Directory of .lin files grouped into operating points.",
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Write the Campbell table here (.csv/.json/.xlsx/.mat by suffix) "
            "instead of stdout.",
        ),
    ] = None,
    parameter: Annotated[
        Parameter,
        typer.Option(
            "--parameter",
            "-p",
            help="Operating parameter to track against; use 'wind' for a sweep "
            "that extends past rated (constant rotor speed).",
        ),
    ] = Parameter.rpm,
) -> None:
    """Build the tracked Campbell diagram across a directory of operating points."""
    with _handle_errors():
        points = discover_operating_points(directory)
        result = ModalPipeline().run(
            points,  # each point is a Sequence[LinFile] carrying its case name
            parameter_name=_PARAMETER_NAMES[parameter],
        )
        _emit_table(campbell_table(result.campbell), output)


@app.command()
def study(
    directory: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            help="Directory of .lin files grouped into operating points.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Directory to write the reproducibility bundle into.",
        ),
    ],
    parameter: Annotated[
        Parameter,
        typer.Option(
            "--parameter",
            "-p",
            help="Operating parameter to track against; use 'wind' for a sweep "
            "that extends past rated (constant rotor speed).",
        ),
    ] = Parameter.rpm,
    tracking: Annotated[
        Tracking,
        typer.Option(
            "--tracking", "-t", help="Cross-operating-point tracking strategy."
        ),
    ] = Tracking.single_path,
    branch_review: Annotated[
        bool,
        typer.Option(
            "--branch-review", help="Flag crossing/veering hotspots for audit."
        ),
    ] = False,
    resolve: Annotated[
        bool,
        typer.Option(
            "--resolve",
            help="Also compute the smooth alternative for hotspots (implies review).",
        ),
    ] = False,
    report: Annotated[
        bool,
        typer.Option(
            "--report/--no-report", help="Write the HTML audit report into the bundle."
        ),
    ] = True,
) -> None:
    """Run a provenance-tracked study and write a reproducibility bundle."""
    with _handle_errors():
        points = discover_operating_points(directory)
        result = run_study(
            points,  # pass the discovered points so their case names reach provenance
            parameter_name=_PARAMETER_NAMES[parameter],
            tracking_strategy=tracking.value,
            branch_review=branch_review,
            resolve=resolve,
        )
        result.write_bundle(output_dir, report=report)
        typer.echo(
            f"Wrote study bundle to {output_dir}: "
            f"{len(result.pipeline.tracks)} tracks, "
            f"{len(result.pipeline.resonances)} resonances."
        )


if __name__ == "__main__":  # pragma: no cover
    app()
