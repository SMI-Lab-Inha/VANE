<!-- SPDX-License-Identifier: Apache-2.0 -->

# VANE architecture

This document is for contributors and integrators who need to find their way around
the source. It outlines the data flow, the package layout and the public interfaces.
The equations are in the [theory pages](https://vane.readthedocs.io/en/latest/theory/index.html)
of the manual; the evidence behind each result is in [VALIDATION.md](VALIDATION.md).

## Data flow

VANE turns an OpenFAST model and its operating curve into linear models in five
stages. Each stage is usable on its own, and each passes plain, immutable data objects
to the next.

1. **Linearize** (`vane.run`). A deck is copied per operating point, configured for a
   `CalcSteady` linearization and run by the OpenFAST executable as a separate
   process. The result is a set of `.lin` files per operating point with a provenance
   record of the settings, attempts and checks.
2. **Read and transform** (`vane.io`). The `.lin` files of one operating point are
   parsed into `LinFile` objects and combined by the MBC3 transform into one
   azimuth-averaged state-space model in non-rotating coordinates, with the
   azimuth-resolved matrices kept on request.
3. **Identify** (`vane.modal`). The averaged state matrix is solved for its modes;
   participation factors attribute each mode to physical degrees of freedom; the
   labeller names it; the tracker links the modes of successive operating points.
   `vane.modal.floquet` and `vane.modal.averaging` analyse the periodic system and the
   error of the azimuth average.
4. **Assemble** (`vane.campbell`). Tracks become Campbell lines with `nP` excitation
   lines and resonance crossings.
5. **Export and report** (`vane.sysid`, `vane.export`, `vane.viz`, `vane.study`). The
   linear models are exported in full or reduced form, packaged for Kalman filtering
   and scheduling across operating points, plotted, and written with their provenance.

`vane.pipeline.ModalPipeline` chains stages 2 to 4, `vane.study.run_study` adds the
provenance record and the report, and `vane.cli` exposes both on the command line.

## Package layout

| Module | Responsibility |
|---|---|
| `vane.io` | `.lin` reader, state selection, `.fst` and module-file readers, a format-preserving deck editor, the MBC3 transform, WAMIT readers, compatibility corrections for known OpenFAST linearization defects |
| `vane.run` | The linearization driver and runner, deck configuration (`lin_config`), pre-flight checks, inference of turbine constants (`auto_config`), executable matching, the operating sweep (`auto_sweep`), settle runs (`steady_state`, `floating_auto`), trim tuning and verification, operating-point acceptance, and radiation fits |
| `vane.modal` | Eigensolution, MAC, participation, labelling, tracking, branch review, Floquet analysis, azimuth averaging and its error estimates, uncertainty and confidence calibration |
| `vane.campbell` | Campbell-diagram assembly, excitation lines and resonance detection |
| `vane.sysid` | State-space models, reduction, quantities of interest, noise models, Kalman models and the filter, estimator error bounds, LPV families, periodic models, serialization |
| `vane.config` | Classification of OpenFAST degree-of-freedom and output descriptions, reference geometry |
| `vane.viz` | Campbell, damping, mode-shape, diagnostic and quantity-of-interest plots, VTK export, the HTML report |
| `vane.export` | Tables (CSV, JSON, `.xlsx`, `.mat`) and the physical-state LPV export |
| `vane.study`, `vane.pipeline` | The high-level entry points |
| `vane.plugins` | The `Tracker` and `Labeler` protocols for alternative implementations |
| `vane.validation`, `vane.benchmark` | Plausibility checks against published frequencies and ground-truth benchmarking of the extraction |
| `vane.utils` | File-system helpers |

Names that start with an underscore are private and may change without notice.

## Design rules

- **The user supplies the operating point.** VANE computes no rotor performance. The
  steady operating curve is an input, or it comes from settle runs of the deck's own
  controller.
- **Fail before launch.** A deck that cannot be linearized, a missing operating
  schedule or an inconsistent argument stops with a specific error before any
  OpenFAST process starts. Unsupported inputs raise; they are never approximated
  silently.
- **Physics before heuristics.** Labels come from participation in named degrees of
  freedom, tracking from modal assurance and frequency continuity, and every
  confidence or ambiguity flag is reported with the label rather than hidden.
- **Provenance.** Every run, study and exported model records its inputs (with content
  hashes where they are files), its settings and the VANE version.
- **No hard-coded paths.** Executables, controller libraries and reference data are
  found through arguments, environment variables or documented search paths.
- **Typed, documented interfaces.** The package passes `mypy --strict`; every public
  function and class has a NumPy-style docstring; arrays are annotated with
  `numpy.typing`.

## Tests

`tests/` mirrors the package (`tests/test_io`, `tests/test_modal`, ...). Unit tests use
synthetic linearizations generated in `tests/conftest.py`; no OpenFAST output is
committed. Integration tests that need OpenFAST, the OpenFAST `r-test` decks or other
local reference data are enabled by environment variables and skip otherwise
(see [VALIDATION.md](VALIDATION.md)).
