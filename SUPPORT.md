<!-- SPDX-License-Identifier: Apache-2.0 -->

# Getting help

VANE is maintained by the Marine Structural Mechanics and Integrity Lab (SMI Lab),
Inha University, on a best-effort basis. All support is through public GitHub Issues,
so that answers help the next user too.

1. **Read the manual first.** The [manual](https://vane.readthedocs.io) covers
   installation, the quick start, the OpenFAST linearization sweep, mode
   identification, Campbell diagrams, state-space export, azimuth averaging, the
   theory, the API reference, and the
   [known limitations](https://vane.readthedocs.io/en/latest/user_guide/limitations.html).
2. **Search existing issues**, open and closed, at
   <https://github.com/SMI-Lab-Inha/VANE/issues>.
3. **Open an issue** with the matching template:

   | You have | Template |
   |---|---|
   | A crash, a wrong result, or a valid input that is rejected | Bug report |
   | A capability you need | Feature request |
   | A question about inputs, methods or results | Question |

   Attach the smallest script and inputs that show the problem, and state the VANE
   version (`vane --version`), the Python version and, for a sweep, the OpenFAST
   version. Do not attach files you may not redistribute.

Security vulnerabilities are not reported in public; follow [SECURITY.md](SECURITY.md).
For questions about citing VANE or collaboration, contact the maintainer at
<jaehoon.seo@inha.ac.kr>.

The maintainers cannot review proprietary models or give design advice for a specific
project; VANE users remain responsible for their engineering decisions.
