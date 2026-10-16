<!-- SPDX-License-Identifier: Apache-2.0 -->

# Security policy

This policy tells you how to report a security vulnerability in VANE.

## Supported versions

Security fixes are released for the latest VANE version. Reproduce a report on that
version.

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a vulnerability

Do not disclose a vulnerability in a public issue. Use GitHub's private reporting form
under **Security > Advisories > Report a vulnerability** at
<https://github.com/SMI-Lab-Inha/VANE/security/advisories/new>. If the form is
unavailable, email `jaehoon.seo@inha.ac.kr` with the subject `VANE security report`.

Include the VANE version or commit, the operating system and Python version, the
smallest reproducer you can share safely, and the observed impact. We acknowledge
reports within 7 days, agree a disclosure schedule with the reporter, and publish a
fixed release and advisory when warranted.

## Scope

VANE reads OpenFAST input decks and linearization files, writes copies of decks, runs
the OpenFAST executable and a controller library named by the user or found on the
paths the documentation describes, and reads and writes JSON and NumPy archives. Treat
decks, controller libraries and archives from untrusted sources as you would any
untrusted program or data: a controller library is native code that OpenFAST loads and
runs. VANE does not load pickled data.

Convergence problems, modelling limitations and suspect results belong in the public
issue tracker unless they also present a security risk.
