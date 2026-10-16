# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Jae Hoon Seo, SMI Lab, Inha University
"""Sphinx configuration for the VANE documentation.

Build locally with ``sphinx-build -W --keep-going -b html docs docs/_build/html``.
Read the Docs uses ``../.readthedocs.yaml``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Document the source tree, not whichever VANE happens to be installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# -- Project information -----------------------------------------------------

project = "VANE"
author = "Jae Hoon Seo"
project_copyright = (
    "2026, Jae Hoon Seo, Marine Structural Mechanics and Integrity Lab (SMI Lab), "
    "Inha University"
)

release = "0.1.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# The included repository Markdown (CONTRIBUTING, CHANGELOG, VALIDATION) carries
# relative links to repository files that are not documents in this site.
suppress_warnings = ["myst.xref_missing"]

# NumPy dtype names rendered from docstrings (``np.bool_``, ``np.int_``) end in an
# underscore, which RST reads as a reference to a target ``np.bool`` / ``np.int``.
# Define those targets (pointing at the NumPy scalar-types docs) so the references
# resolve instead of broadly suppressing docutils warnings.
rst_epilog = """
.. _np.bool: https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool_
.. _np.int: https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.int_
"""

# Treat the package as importable for autodoc; fail the build on a broken reference.
nitpicky = False
add_module_names = False

# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_rtype = False

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}
# Resolving the mappings fetches each project's remote ``objects.inv`` inventory at
# build time; under ``sphinx-build -W`` an unreachable inventory becomes a hard error.
# A locked-down/offline build (no network) can set ``VANE_DOCS_OFFLINE=1`` to disable
# intersphinx instead of failing — cross-project references then render as plain text.
if os.environ.get("VANE_DOCS_OFFLINE"):
    intersphinx_mapping = {}

# -- MyST (Markdown for the changelog and validation pages) ------------------

myst_enable_extensions = ["dollarmath", "colon_fence"]
myst_heading_anchors = 3

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_title = f"VANE {release} documentation"
html_short_title = "VANE"
html_static_path = ["_static"]
html_last_updated_fmt = "%Y-%m-%d"
html_show_sourcelink = False
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "style_external_links": True,
    "prev_next_buttons_location": "both",
}
html_context = {
    "display_github": True,
    "github_user": "SMI-Lab-Inha",
    "github_repo": "VANE",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# -- LaTeX / PDF output ------------------------------------------------------

# XeLaTeX handles the Unicode used in the manual (Greek symbols, the multiplication
# sign, dashes) without per-character declarations.
latex_engine = "xelatex"
latex_elements = {"papersize": "a4paper", "pointsize": "10pt"}
latex_documents = [
    ("index", "VANE.tex", "VANE documentation", author, "manual"),
]

# -- EPUB output -------------------------------------------------------------

epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = project_copyright
epub_show_urls = "footnote"

# Copy button: strip shell and REPL prompts when copying code blocks.
copybutton_prompt_text = r">>> |\.\.\. |\$ |PS [^>]*> "
copybutton_prompt_is_regexp = True
