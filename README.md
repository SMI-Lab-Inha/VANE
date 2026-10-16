<!-- SPDX-License-Identifier: Apache-2.0 -->

# VANE

[![Release](https://img.shields.io/github/v/release/SMI-Lab-Inha/VANE?sort=semver)](https://github.com/SMI-Lab-Inha/VANE/releases/latest)
[![CI](https://github.com/SMI-Lab-Inha/VANE/actions/workflows/python.yml/badge.svg)](https://github.com/SMI-Lab-Inha/VANE/actions/workflows/python.yml)
[![Documentation](https://img.shields.io/readthedocs/vane/latest?label=docs)](https://vane.readthedocs.io/en/latest/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)

VANE (Visualization and Automated Natural-frequency Extraction) is an open-source
Python tool for the linearization analysis of wind turbines in OpenFAST. It takes an
OpenFAST model and the turbine's operating curve to labelled Campbell diagrams and
modal results:

1. it sets up the linearization case of every operating point from the deck and the
   curve, checks that the deck can be linearized, and runs the sweep in parallel,
   retrying trims that do not converge and settling floating turbines first;
2. it applies the multi-blade coordinate (MBC3) transform and averages over azimuth on
   the azimuths actually delivered;
3. it identifies, labels and tracks the modes across operating points from their
   physical participation;
4. it assembles Campbell diagrams with excitation lines and resonance crossings, with
   tabular and Excel summaries, and visualizes mode shapes (static, 3-D and VTK);
5. it checks the averaged model with Floquet analysis and per-branch averaging-error
   estimates; and
6. it exports state-space and linear parameter-varying (LPV) models for estimator and
   controller design.

VANE handles fixed-bottom and floating turbines. The current stable release is v0.1.0.
The public API is documented in the
[API reference](https://vane.readthedocs.io/en/latest/reference/api_reference.html) and
is stable. VANE follows semantic versioning: a breaking change to the documented public
API is made only with a new major version and is recorded in the
[changelog](CHANGELOG.md).

## Capabilities

| Step | Features |
|---|---|
| Linearization set-up | Copy of each case from the deck and every file it references; inference of the copy root, the OpenFAST executable that reads the deck, the rated wind speed and generator torque, and the rotor-size-scaled trim settings; module settings for a linearizable deck; pre-flight rejection of decks that cannot be linearized |
| Running the sweep | OpenFAST over a list of wind speeds from a user-supplied operating curve, in parallel; `CalcSteady` trims with classified, automatic retries; settle-then-linearize for floating turbines; state-space radiation fits from WAMIT data; checks of each trimmed point against the operating curve |
| MBC3 and azimuth averaging | `.lin` reader, MBC3 transform with the azimuth-resolved matrices retained, interval-weighted mean on the delivered azimuths, state selection by pattern or DOF class |
| Mode identification and tracking | Eigensolution with conjugate-pair handling and condition numbers, participation factors, physical labels with confidence and ambiguity flags, collective, regressive and progressive blade whirl, heavily damped modes, MAC and MACXP, tracking across operating points, branch review of crossings and veerings |
| Campbell diagrams | Tracked frequency and damping lines on a rotor-speed or wind-speed axis, `nP` excitation lines, damping-weighted resonance crossings, CSV, JSON, `.xlsx` and `.mat` tables, provenance-tracked studies with a self-contained HTML report, a command-line interface |
| Mode-shape visualization | Static participation plots, 3-D and full-system animations, VTK export; publication-quality figures with single- and double-column size presets and vector export |
| Floquet and averaging error | Floquet and Hill analysis of the periodic system, azimuth-direction consistency test and pinning, per-branch sampling, quadrature and periodicity error estimates, the `lcm(N, B)` aliasing rule, azimuth-count design and a commensurate time step |
| State-space and LPV export | Full and reduced (modal, balanced) state-space models, quantities of interest, discretization, process and measurement noise, Kalman-filter models and a filter, LPV model families across operating points, pickle-free save and reload |

## Installation

VANE needs Python 3.10 to 3.13 and is installed from GitHub. Either download the
wheel attached to the [v0.1.0 release](https://github.com/SMI-Lab-Inha/VANE/releases/tag/v0.1.0)
and install it:

```console
python -m pip install vane-0.1.0-py3-none-any.whl
```

or install the tagged release from the repository:

```console
python -m pip install "git+https://github.com/SMI-Lab-Inha/VANE.git@v0.1.0"
```

For development, clone the repository and install it in editable mode with
`python -m pip install -e ".[dev]"`.

Post-processing needs nothing beyond the Python dependencies. Running linearization
sweeps also needs an OpenFAST executable that reads your decks.

## Quick start

From a directory of `.lin` files, one operating point per azimuth sweep:

```python
from vane import ModalPipeline, discover_operating_points

points = discover_operating_points("path/to/lin_files")
result = ModalPipeline().run(points, parameter_name="wind_speed")

for track in result.campbell.tracks:
    print(track.label.label, track.natural_frequencies_hz)

system = result.state_space(0)  # A, B, C, D at the first operating point
```

To produce the `.lin` files, give VANE the deck, the wind speeds and the turbine's
steady operating curve (wind speed, rotor speed and blade pitch):

```python
from vane.run import run_operating_sweep

sweep = run_operating_sweep(
    "IEA-15-240-RWT-Monopile.fst",
    output_dir="sweep_out",
    wind_speeds=[4.0, 8.0, 12.0, 16.0, 20.0, 24.0],
    performance_csv="rotor_performance.csv",
)
```

The same analysis is available from the command line:

```console
vane campbell path/to/lin_files --parameter wind -o campbell.csv
vane study path/to/lin_files --parameter wind -o study_output
```

## Documentation

The manual at <https://vane.readthedocs.io> covers installation, the quick start, the
OpenFAST linearization sweep, mode identification, Campbell diagrams, state-space
export, azimuth averaging, step-by-step tutorials, the theory, the validation record,
and the API reference. The tutorial scripts are in the [examples folder](examples/README.md);
all but the OpenFAST sweep run on synthetic inputs without OpenFAST.

## Validation

VANE is verified against analytical solutions, exact invariants and prescribed-mode
models, and has been run end to end on the OpenFAST `r-test` reference systems (NREL
5 MW on land, OC3 monopile, OC3 Hywind spar, OC4 DeepCwind semi-submersible) and on
the NREL 5 MW and IEA 3.4, 10, 15 and 22 MW reference turbines. Cases, acceptance
values and the tests that reproduce them are in the
[validation record](VALIDATION.md).

## Repository layout

| Directory | Contents |
|---|---|
| `src/vane/` | The Python package |
| `tests/` | pytest unit, integration and regression tests |
| `docs/` | Sphinx manual, including the tutorials and the validation figures |
| `examples/` | Runnable tutorial scripts |
| `.github/` | Continuous integration, issue and pull-request templates |

The package structure is described in the [architecture overview](ARCHITECTURE.md).

## Citing

If you use VANE in academic work, please cite the software release you used
(VANE 0.1.0). The [citation file](CITATION.cff) holds the citation metadata (GitHub
**Cite this repository**); BibTeX is on the manual's
[How to cite](https://vane.readthedocs.io/en/latest/user_guide/citing.html) page.

## Contributing and support

Questions, bug reports and feature requests are welcome in
[GitHub Issues](https://github.com/SMI-Lab-Inha/VANE/issues); see
the [support guide](SUPPORT.md) for where each belongs. Read the
[contributing guide](CONTRIBUTING.md) before opening a pull request. Everyone taking part
is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md). Report security
issues privately as described in the [security policy](SECURITY.md).

VANE is research software. Users are responsible for model selection, convergence
checks and the engineering decisions they base on its results.

VANE is developed at the Marine Structural Mechanics and Integrity Lab (SMI Lab),
Inha University. Maintainer: Prof. Jae Hoon Seo, Department of Naval Architecture and
Ocean Engineering, Inha University, Republic of Korea, <jaehoon.seo@inha.ac.kr>

## Acknowledgements

VANE analyses linearizations produced by OpenFAST, maintained by NLR (National
Laboratory of the Rockies, formerly NREL), and its tests use the OpenFAST regression
test decks and the IEA Wind reference-turbine models. We are grateful to their
developers and maintainers.

## License and name

VANE is distributed under the Apache License 2.0. See [LICENSE](LICENSE) and
[NOTICE](NOTICE).

"VANE" identifies the official project in this repository. Modified versions are
welcome under the licence, but, as Section 6 of the Apache License 2.0 provides, they
may not use the name in a way that implies they are the official VANE or endorsed by
its maintainers; describing a work as "based on VANE" is fine.
