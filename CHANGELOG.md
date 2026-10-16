<!-- SPDX-License-Identifier: Apache-2.0 -->

# Changelog

All notable changes to VANE are recorded in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-10-16

### Fixed

- `LICENSE` is now the unmodified text of the Apache License, Version 2.0. The file
  of 0.1.0 differed from it in the wording of sections 6 and 9 and of the appendix.
  The copyright statement is in `NOTICE`.
- `azimuth_tangent` sets the azimuth component of the tangent to exactly one, as
  documented; on some platforms the normalisation rounded it to 0.9999999999999999.
- The type annotations pass strict mypy with NumPy 2.2, the version installed on
  Python 3.10, as well as with later NumPy versions.
- A controller-library test no longer depends on the host platform.
- The README names the linked project documents by their purpose, and its
  documentation badge is served from shields.io.

## [0.1.0] - 2026-10-16

Initial public release.

### Added

- **Linearization set-up and sweep** (`vane.run.run_operating_sweep`). Each operating
  point's case is copied from the deck and every file it references and configured for
  a `CalcSteady` linearization. The copy root, the OpenFAST executable that reads the
  deck, the rated wind speed and generator torque, the rotor-size-scaled trim gain and
  tolerance, and the module settings are inferred from the deck and the user-supplied
  operating curve (a rotor-performance CSV or the curve arrays) and recorded with their
  sources. Decks that cannot be linearized, and sweeps without an operating schedule,
  are rejected before any OpenFAST process starts. The sweep runs in parallel; failed
  trims are classified and retried with a remedy for each failure class, and a point
  that never converges is recorded as such. `vane.run` re-exports every public name of
  its modules, including `curve_power_coefficient`.
- **Rotor-performance CSV units.** Units are read from the column headers. The power-
  and thrust-coefficient columns (`cp`, `ct` and their long names) are fractions (no
  unit, `[-]`, `[1]`, `[fraction]`) or declared percentages (`[%]`, `[percent]`,
  `[pct]`, converted); any other unit, a percentage written without its unit (a value
  above the Betz limit for `cp` or above 2 for `ct`), and fractions declared as percent
  are rejected with an error, consistently in `rotor_performance_from_csv` and
  `curve_power_coefficient`.
- **Floating turbines.** Settle-then-linearize operating points, with automatic search
  for a host controller library, a platform probe that sizes temporary settle damping,
  a two-stage settle with an undamped confirmation, settle criteria for sustained
  oscillations, state-space radiation fits generated from WAMIT data, and an optional
  check of the azimuth-mean platform rates of each linearized point.
- **Operating-point verification** (`vane.run.verify_sweep_operating_points`): each
  trimmed point is compared with the rotor-performance data.
- **Input and MBC3.** A reader for OpenFAST `.lin` files in the current format, readers
  and a format-preserving editor for `.fst` decks and their module files, the MBC3
  transform of three-bladed rotors with the azimuth-resolved matrices retained, and
  state selection by description pattern or DOF class (`vane.io.select_states`).
- **Mode identification and tracking.** Eigensolution with conjugate-pair handling and
  eigenvalue condition numbers, participation factors, physical labels with confidence,
  ambiguity flags and alternatives, collective, regressive and progressive blade whirl,
  heavily damped modes, MAC and MACXP, tracking across operating points (single-path
  and simultaneous), branch review of crossings and veerings, confidence calibration,
  and the `Tracker` and `Labeler` extension protocols.
- **Campbell diagrams.** Tracked frequency and damping lines on a rotor-speed or
  wind-speed axis, `nP` excitation lines and damping-weighted resonance crossings, with
  CSV, JSON, `.xlsx` and `.mat` tables and provenance-tracked studies (`vane.run_study`)
  with a self-contained HTML report.
- **Visualization.** Campbell diagrams with mode families, inline `nP` labels and
  line-end mode names, damping and mode-shape plots, 3-D and full-system mode-shape
  animation, VTK export and tracking diagnostics, in one consistent style with
  single- and double-column size presets and vector export, in matplotlib and Plotly.
- **Floquet analysis and azimuth averaging** (`vane.modal.floquet`,
  `vane.modal.averaging`, `vane.modal.averaging_error_estimate`): Floquet and Hill
  analysis of the linear time-periodic system, the interval-weighted mean of the
  delivered azimuths, harmonic fits, the azimuth-direction consistency test, tangent
  pinning of the rotor-azimuth state, per-branch sampling, quadrature and periodicity
  error estimates (`estimate_averaging_error`), the `lcm(N, B)` aliasing rule,
  azimuth-count design (`azimuth_count_table`, `alias_free_azimuth_count`) and a time
  step commensurate with the azimuth grid (`commensurate_time_step`). The Hill route
  (`floquet_modes`, `averaging_error`) needs near-uniform azimuths (relative 1e-3,
  `HILL_UNIFORM_RTOL`) and rejects an uneven grid with an error naming the worst
  interval; the monodromy route (`monodromy_exponents`) handles delivered, uneven
  grids.
- **State-space and LPV export.** Full-order state-space models with named states,
  inputs and outputs; modal and balanced reduction; quantities of interest;
  discretization; process and measurement noise; Kalman-filter models, a filter and
  steady-state estimator error bounds (`vane.sysid.assessment`); LPV model families
  across operating points (`vane.export.build_physical_lpv_family`); periodic
  reduced-order models; and pickle-free save and reload of estimator results, ROM
  families and physical LPV families (`save_lpv_family`, `load_lpv_family`, with a
  versioned archive schema and an exact round trip).
- **Command line.** `vane modes`, `vane campbell` and `vane study`.
- **Documentation.** A user guide, step-by-step tutorials with runnable scripts in
  `examples/` (synthetic inputs, except the OpenFAST sweep), the theory, the API
  reference and contract, and the validation record.

[Unreleased]: https://github.com/SMI-Lab-Inha/VANE/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/SMI-Lab-Inha/VANE/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/SMI-Lab-Inha/VANE/releases/tag/v0.1.0
