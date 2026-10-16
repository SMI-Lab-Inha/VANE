<!-- SPDX-License-Identifier: Apache-2.0 -->

# Contributing to VANE

Contributions are welcome: bug reports, documentation fixes, validation cases and
code. This guide sets out how to contribute and the standards a change must meet. The
package structure is in
[ARCHITECTURE.md](https://github.com/SMI-Lab-Inha/VANE/blob/main/ARCHITECTURE.md), and
the evidence behind each result is in
[VALIDATION.md](https://github.com/SMI-Lab-Inha/VANE/blob/main/VALIDATION.md). Everyone
taking part follows the
[Code of Conduct](https://github.com/SMI-Lab-Inha/VANE/blob/main/CODE_OF_CONDUCT.md).

## Contributing upstream

If you have extended or corrected VANE in your own copy, please consider offering the
change here. A change merged upstream is tested in CI, carried into the validation
record and maintained with the rest of the code, and every user benefits from it. Open
an issue first for a larger change so that the design can be agreed before you invest
in it.

## Issues

- **Bugs**: use the bug-report template. Include the VANE version (`vane --version`),
  the Python version and platform, the OpenFAST version if a sweep is involved, a
  minimal script and the inputs that reproduce the problem, and the full error output.
- **Enhancements**: use the feature-request template and describe the engineering
  problem the feature solves.
- **Questions**: use the question template; see
  [SUPPORT.md](https://github.com/SMI-Lab-Inha/VANE/blob/main/SUPPORT.md).
- **Security issues**: follow
  [SECURITY.md](https://github.com/SMI-Lab-Inha/VANE/blob/main/SECURITY.md); do not open
  a public issue.

## Development setup

```console
git clone https://github.com/SMI-Lab-Inha/VANE.git
cd VANE
python -m pip install -e ".[dev,docs]"
pre-commit install
```

Run the checks that continuous integration runs:

```console
python -m ruff check src tests examples docs
python -m ruff format --check src tests examples docs
python -m mypy src
python -m pytest
sphinx-build -W --keep-going -b html docs docs/_build/html
```

`pytest` enforces at least 85 % line coverage of the package. Tests that need
OpenFAST, the OpenFAST `r-test` decks or other local reference data skip when the data
is not available; VALIDATION.md lists the environment variables that enable them.

## Pull requests

`main` holds the released code and the work for the next release.

1. Fork the repository and branch from `main` with a descriptive name
   (`fix/<topic>`, `feature/<topic>`, `docs/<topic>`).
2. Keep each pull request to one change. Add or update tests for it.
3. Run `pre-commit run --all-files` and the checks above.
4. Open the pull request against `main` and complete the template.
5. CI must pass and a maintainer must approve before merge. Pull requests are
   squash-merged.

Write commit messages as plain sentences in the imperative mood: a subject of at most
72 characters that says what the change does, and a body that explains why it is
needed.

```text
Report the alias order of a predicted azimuth grid
Reject a rotor-performance CSV with a duplicate wind speed
Describe the settle criteria in the sweep guide
```

Do not bypass hooks with `--no-verify`. Commits must be attributed to the people who
wrote them.

## Standards

- **Tests**: every behaviour change carries a pytest case. Numerical changes compare
  against a closed form, an exact invariant, a prescribed-mode model or an independent
  reference.
- **Validation**: if a change alters a validation result, tolerance or reference,
  update [VALIDATION.md](https://github.com/SMI-Lab-Inha/VANE/blob/main/VALIDATION.md)
  in the same pull request.
- **Documentation**: document only implemented and tested behaviour. Update the
  manual in `docs/` and add a
  [CHANGELOG.md](https://github.com/SMI-Lab-Inha/VANE/blob/main/CHANGELOG.md) entry
  under `[Unreleased]` for user-visible changes (add that heading above the latest
  release if it is not there yet).
- **Errors**: unsupported inputs stop with a specific error message; never return an
  approximate result silently.
- **Code**: `ruff` for linting and formatting, `mypy --strict` for types, NumPy-style
  docstrings for every public function and class, `numpy.typing` annotations for
  arrays, the `logging` module instead of `print`, specific exception types instead of
  bare `except:`, and no hard-coded paths.
- **Other codes**: describe differences from OpenFAST or any other code neutrally and
  with the evidence, and cite them as listed in the manual's references.

## Repository contents

- Manual pages go in `docs/`; the theory pages in `docs/theory/`.
- Synthetic test data is generated in the tests; do not commit OpenFAST output
  (`.lin`, `.outb`, `.out`), third-party decks, copyrighted papers or manuscript
  sources.

## Licence and sign-off

VANE does not use a contributor licence agreement. As Section 5 of the
[Apache License 2.0](https://github.com/SMI-Lab-Inha/VANE/blob/main/LICENSE) provides,
a contribution you submit is licensed under the same licence. Sign off each commit
(`git commit -s`) to confirm the
[Developer Certificate of Origin 1.1](https://developercertificate.org/): that you wrote
the change or otherwise have the right to submit it under this licence. New source
files start with the SPDX header used throughout the repository:

```python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
```
