<!-- SPDX-License-Identifier: Apache-2.0 -->

# VANE release procedure

This procedure is for maintainers. It turns one reviewed commit into a VANE release.
Run it from a clean checkout of the release commit. Tags have the form
`vMAJOR.MINOR.PATCH`. VANE follows semantic versioning: a breaking change to the
documented public API is made only with a new major version and is recorded in the
changelog.

## 1. Set and check the release metadata

Update the version in `src/vane/__init__.py`, `pyproject.toml`, `docs/conf.py`,
`CITATION.cff` (`version` and `date-released`) and `CHANGELOG.md`. Move every release
note from `[Unreleased]` to the new dated heading and update the compare links at the
end of the changelog. `tests/test_version.py` checks that these files agree.

<!--
When the journal article that describes VANE is published, add a
`preferred-citation` entry (type: article, authors with ORCID and affiliation,
journal, year, volume, pages or article number, DOI) to CITATION.cff, and add the
article to README.md (Citing), docs/user_guide/citing.rst and a DOI badge in
README.md.
-->

## 2. Run the checks

```console
python -m pip install -e ".[dev,docs]"
python -m ruff check src tests examples docs
python -m ruff format --check src tests examples docs
python -m mypy src
python -m pytest
sphinx-build -W --keep-going -b html docs docs/_build/html
pre-commit run --all-files
git diff --check
```

Run the test suite on every supported Python version (3.10 to 3.13). Tests that need
OpenFAST or local reference data skip without it; before a release, run them with the
data available (see [VALIDATION.md](VALIDATION.md)) and confirm that VALIDATION.md
matches the results at the release commit.

## 3. Build and inspect the distributions

```console
python -m build --outdir dist
python -m venv wheel-test
wheel-test/bin/python -m pip install dist/vane-MAJOR.MINOR.PATCH-py3-none-any.whl
wheel-test/bin/vane --version
```

Both distributions must contain the Apache-2.0 licence and the NOTICE file; the wheel
must contain only the `vane` package and its metadata. Install the source distribution
in a second clean environment and import it as well. Then write the checksums of the
two files:

```console
cd dist
sha256sum vane-MAJOR.MINOR.PATCH-py3-none-any.whl vane-MAJOR.MINOR.PATCH.tar.gz > SHA256SUMS.txt
```

On Windows, `Get-FileHash <file> -Algorithm SHA256` gives the same digests.

## 4. Tag and publish

Tag only the commit that passed every check:

```console
git tag -s vMAJOR.MINOR.PATCH -m "VANE MAJOR.MINOR.PATCH"
git push origin vMAJOR.MINOR.PATCH
```

If no signing key is available, use an annotated tag and state this in the release
notes. Create a draft GitHub release from the tag with the changelog section as the
notes and attach exactly these assets:

- `vane-MAJOR.MINOR.PATCH-py3-none-any.whl`
- `vane-MAJOR.MINOR.PATCH.tar.gz`
- `SHA256SUMS.txt`

Publish after checking the notes, licence, citation, documentation links and asset
checksums on the draft page. VANE is distributed only through GitHub: users install the
attached wheel or `git+https://github.com/SMI-Lab-Inha/VANE.git@vMAJOR.MINOR.PATCH`.

## 5. Close the release

Confirm that the documentation builds from the tag on Read the Docs, that the GitHub
source archives contain no run output or local data, and that a fresh clone passes the
test suite. Then add an empty `[Unreleased]` section to the changelog.

## Publishing checklist for a new repository

Before the first release is published from a new GitHub repository:

- [ ] **About**: set the description and topics below, the website to
      <https://vane.readthedocs.io>, and enable Releases.
- [ ] **Features**: enable Issues (the templates in `.github/ISSUE_TEMPLATE` disable
      blank issues); disable the Wiki and Projects unless they will be maintained.
- [ ] **Security**: enable private vulnerability reporting (used by
      [SECURITY.md](SECURITY.md)) and secret scanning.
- [ ] **Actions**: allow GitHub-hosted runners; the workflow needs no secrets and only
      read access to the repository contents.
- [ ] **Branch protection** on `main`: require pull requests, the `python` workflow
      checks and linear history.
- [ ] **Read the Docs**: import the repository as project `vane`, build `latest` and
      the release tag, and check the badge in the README.
- [ ] **Citation**: confirm that **Cite this repository** renders from `CITATION.cff`.

Suggested description:

> OpenFAST linearization sweeps, physics-based mode identification, Campbell
> diagrams, azimuth-averaging error estimates and state-space export for fixed-bottom
> and floating wind turbines.

Suggested topics: `wind-energy`, `wind-turbine`, `floating-offshore-wind`,
`offshore-wind`, `openfast`, `linearization`, `modal-analysis`, `campbell-diagram`,
`multi-blade-coordinate-transform`, `floquet`, `state-space`, `kalman-filter`,
`digital-twin`, `python`.

## Release record: 0.1.0

The checks of sections 2 and 3 were run on the release tree on 2026-10-16, before the
release commit, on Windows 11 x64. OpenFAST was not run: the tests that need OpenFAST
or local reference data skipped, as they do on hosted CI runners. The tree includes the
saving and loading of physical LPV families with a versioned archive schema, the units
of the rotor-performance coefficient columns, the complete re-export of `vane.run`
(including `curve_power_coefficient`) and the near-uniform azimuth requirement of
Hill's method in `floquet_modes`.

| Item | Version |
| --- | --- |
| Python 3.10 | 3.10.21 (NumPy 2.2.6, SciPy 1.15.3, pandas 2.3.3, Matplotlib 3.10.9, Plotly 7.1.0) |
| Python 3.11 | 3.11.15 (NumPy 2.4.6, SciPy 1.17.1, pandas 3.0.3, Matplotlib 3.10.9, Plotly 6.8.0) |
| Python 3.12 | 3.12.14 (NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.6, Matplotlib 3.11.2, Plotly 7.1.0) |
| Python 3.13 | 3.13.15 (NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.6, Matplotlib 3.11.2, Plotly 7.1.0) |
| Tools | Ruff 0.15.16, mypy 2.1.0, pytest 9.0.3 (Python 3.11) and 9.1.1 (Python 3.10, 3.12, 3.13), pytest-cov 7.1.0, Sphinx 9.0.4, docutils 0.22.4, MyST-Parser 5.1.0, sphinx-rtd-theme 3.1.0, build 1.5.0, setuptools from the isolated build environment (>= 77) |

| Check | Result |
| --- | --- |
| `ruff check src tests examples docs` | passed, no findings |
| `ruff format --check src tests examples docs` | passed, 220 files formatted |
| `mypy --strict src` | passed, no issues in 92 source files |
| `pytest` on Python 3.10 | 3105 passed, 122 skipped; line coverage 95.15 % (minimum 85 %) |
| `pytest` on Python 3.11 | 3105 passed, 122 skipped; line coverage 95.14 % |
| `pytest` on Python 3.12 | 3105 passed, 122 skipped; line coverage 95.15 % |
| `pytest` on Python 3.13 | 3105 passed, 122 skipped; line coverage 95.14 % |
| Tutorial scripts (`tests/test_examples.py`) | every script without OpenFAST ran; the OpenFAST sweep skipped |
| `sphinx-build -W --keep-going -b html docs docs/_build/html` | built with no warnings (intersphinx online) |
| `sphinx-build -W --keep-going -b epub -d <doctrees> docs <out>` | built with no warnings (doctrees kept outside the EPUB output directory) |
| `python -m build` | wheel `vane-0.1.0-py3-none-any.whl` and sdist `vane-0.1.0.tar.gz` built |
| Clean-environment installs | the wheel and the sdist each installed into a new virtual environment; `import vane` and `vane --version` report 0.1.0, and an LPV family saved and loaded from each installation round-trips exactly |
| Wheel contents | only the `vane` package and its metadata, with LICENSE and NOTICE |
| Release-tree audit | no private paths, internal history or third-party tool names; every new source file carries the SPDX header; every repository URL uses `SMI-Lab-Inha/VANE` |

The release assets and `SHA256SUMS.txt` are produced from the tagged commit as
described in sections 3 and 4.
