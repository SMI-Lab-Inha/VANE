---
name: Bug report
about: Report incorrect results, a crash, or a rejected valid input
title: "Bug: <short description>"
labels: bug
---

<!-- SPDX-License-Identifier: Apache-2.0 -->

## What happened

<!-- The error message or the wrong result, and what you expected instead. -->

## How to reproduce

<!-- The smallest Python script or command that shows the problem, and the inputs it
     needs (.lin files, the .fst deck and its module files, an operating-curve CSV).
     Attach only files you may redistribute. -->

```python
from vane import ModalPipeline, discover_operating_points

result = ModalPipeline().run(discover_operating_points("lin_files"))
```

## Environment

- VANE version (`vane --version`):
- Python version and operating system:
- OpenFAST version, if a sweep or the `.lin` files are involved (`openfast -v`):
- How VANE was installed (release wheel, Git URL, or clone):

## Checks

- [ ] The problem occurs with the latest release.
- [ ] If results disagree with another code or a published value, I have described both
      models and the quantity compared (see
      [VALIDATION.md](https://github.com/SMI-Lab-Inha/VANE/blob/main/VALIDATION.md)).
