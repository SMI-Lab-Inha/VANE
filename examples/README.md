<!-- SPDX-License-Identifier: Apache-2.0 -->

# VANE tutorial scripts

This directory holds one runnable script per tutorial. The guided walkthroughs, with
the printed output and figures of each script, are in the documentation under
[docs/tutorials](../docs/tutorials/index.rst).

| Script | Tutorial | Needs |
| --- | --- | --- |
| `01_linearization_sweep.py` | Run an OpenFAST linearization sweep over an operating curve | OpenFAST and the OpenFAST r-test repository |
| `02_mbc_and_eigenanalysis.py` | Read `.lin` files, apply MBC3 and averaging, find the modes | nothing else |
| `03_mode_identification_campbell.py` | Label and track the modes, Campbell diagram, tables | nothing else (`openpyxl` for `.xlsx`) |
| `04_mode_shapes.py` | Participation charts, 3-D views and a VTK animation | nothing else |
| `05_azimuth_averaging.py` | Azimuth count, azimuth pinning, averaging-error estimate | nothing else |
| `06_state_space_export.py` | State-space, reduced, Kalman and LPV models | nothing else |

## Run a script

From the repository root, with VANE installed (see the installation page of the
documentation):

```console
python examples/02_mbc_and_eigenanalysis.py
python examples/03_mode_identification_campbell.py --output my_results
```

Without `--output`, the generated data (`.lin` files, tables, HTML and VTK files) go
to a temporary directory that is deleted when the script ends, and the figures go to
`docs/tutorials/figures/`. With `--output DIR`, every file goes to `DIR` (the data to
`DIR/data`). Each script runs in a few seconds and draws the same figure on every run.

`01_linearization_sweep.py` needs two environment variables:

```console
export VANE_RTEST_DIR=/path/to/r-test        # a clone of https://github.com/OpenFAST/r-test
export VANE_OPENFAST=/path/to/openfast       # or put openfast on PATH
python examples/01_linearization_sweep.py --configure-only
python examples/01_linearization_sweep.py --winds 6 10 14 18 --output sweep_results
```

Without them it prints what is missing and exits. `--configure-only` reads the deck
and the operating curve and prints the settings VANE would use, without launching
OpenFAST. A full run takes minutes per operating point; its results are kept in
`--output`, or in a new temporary directory whose path is printed.

## Synthetic linearization files

OpenFAST `.lin` files are not distributed with VANE. Tutorials 2 to 6 write small
synthetic files instead, with `_synthetic.py`: a three-bladed turbine with the first
tower fore-aft and side-side modes, the generator azimuth and a first flap mode on each
blade, written in OpenFAST's mixed frame at twelve azimuths per operating point. Its
numbers resemble a 5 MW-class turbine but describe no real machine. Tutorial 5 uses a
separate four-state periodic model from the same module. Use these files to learn the
workflow; use linearizations of your own turbine for results.

## Input data

| File | Contents |
| --- | --- |
| `data/nrel5mw_operating_curve.csv` | Wind speed, rotor speed, collective pitch and power coefficient of a peak-shaving schedule shaped like that of the NREL 5 MW turbine, with rated wind speed 12.12 m/s. The values are those of the operating curve in the VANE test suite (`tests/test_run/test_auto_config.py`). Replace it with your turbine's own steady operating schedule. |
