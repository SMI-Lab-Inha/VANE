.. SPDX-License-Identifier: Apache-2.0

Installation
============

Requirements
------------

VANE supports Python 3.10 to 3.13 on Linux, macOS and Windows. Its runtime
dependencies are NumPy, SciPy, pandas, Matplotlib, Plotly and Typer; pip installs them
automatically.

Post-processing existing ``.lin`` files needs nothing else. Running linearization
sweeps (:doc:`linearization_sweep`) also needs an OpenFAST executable whose release
reads the input decks, and, for closed-loop settle runs of a floating turbine, a
controller library built for the host.

VANE is distributed through its GitHub repository,
<https://github.com/SMI-Lab-Inha/VANE>. A virtual environment is recommended.

Install the release wheel
-------------------------

Download ``vane-0.1.1-py3-none-any.whl`` from the
`v0.1.1 release <https://github.com/SMI-Lab-Inha/VANE/releases/tag/v0.1.1>`_, check it
against ``SHA256SUMS.txt`` from the same page, and install it:

.. code-block:: console

   $ python -m pip install vane-0.1.1-py3-none-any.whl

Install from the repository
---------------------------

Install the tagged release directly from GitHub:

.. code-block:: console

   $ python -m pip install "git+https://github.com/SMI-Lab-Inha/VANE.git@v0.1.1"

or clone the repository and install it, in editable mode with the development tools
for work on VANE itself:

.. code-block:: console

   $ git clone https://github.com/SMI-Lab-Inha/VANE.git
   $ cd VANE
   $ python -m pip install .
   $ python -m pip install -e ".[dev]"   # for development

Optional extras
---------------

``excel``
   ``openpyxl``, needed only to write ``.xlsx`` tables. CSV, JSON and ``.mat`` export
   need no extra.

``pdf``
   ``weasyprint``, needed only to render the HTML study report to PDF.

``docs``
   The Sphinx toolchain used to build this manual.

``dev``
   ``ruff``, ``mypy``, ``pytest`` and its plugins, ``pre-commit`` and ``build``.

.. code-block:: console

   $ python -m pip install ".[excel,pdf]"

Check the installation
----------------------

.. code-block:: console

   $ python -c "import vane; print(vane.__version__)"
   $ vane --version
   $ python -m pytest -q

Tests that need OpenFAST, the OpenFAST ``r-test`` decks or other local reference data
skip when that data is not available; the validation record (:doc:`validation`) lists
the environment variables that enable them.

.. note::

   On Windows, some OpenBLAS 0.3.34 builds have crashed with an access violation in
   ``numpy.linalg.eig`` and ``scipy.linalg.eig`` when eigenvectors of matrices of order
   200 or more were requested, which VANE's eigensolver does for full turbine models.
   If you meet this crash, use NumPy and SciPy builds linked against a different
   OpenBLAS release (0.3.33 is unaffected).
