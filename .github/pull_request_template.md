<!-- SPDX-License-Identifier: Apache-2.0 -->

## Summary

<!-- What this pull request changes and why. Link the issue it resolves (Fixes #123). -->

## Tests

<!-- Tests added or updated, and what they cover. -->

## Checklist

- [ ] `pre-commit run --all-files` passes
- [ ] `python -m ruff check src tests examples docs` and `python -m ruff format --check src tests examples docs` pass
- [ ] `python -m mypy src` passes
- [ ] `python -m pytest` passes, with coverage at or above 85 %
- [ ] `sphinx-build -W --keep-going -b html docs docs/_build/html` passes
- [ ] [VALIDATION.md](https://github.com/SMI-Lab-Inha/VANE/blob/main/VALIDATION.md) updated if a validation result changed
- [ ] [CHANGELOG.md](https://github.com/SMI-Lab-Inha/VANE/blob/main/CHANGELOG.md) updated under `[Unreleased]` (heading added if missing) for user-visible changes
- [ ] Documentation in `docs/` updated for changed APIs, options, or outputs
- [ ] Commits are signed off (`git commit -s`) under the
      [Developer Certificate of Origin](https://developercertificate.org/)
