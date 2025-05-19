"""Sphinx configuration for EpitopeGraph documentation."""

import os
import sys
from datetime import datetime

# Add package to path
sys.path.insert(0, os.path.abspath(".."))

# Project information
project = "EpitopeGraph"
copyright = f"{datetime.now().year}, Raaga Karumanchi"
author = "Raaga Karumanchi"
version = "0.1.0"
release = "0.1.0"

# Extensions
extensions = [
    "sphinx.ext.autodoc",        # Automatic API documentation
    "sphinx.ext.napoleon",       # Support for NumPy/Google style docstrings
    "sphinx.ext.viewcode",       # Add links to source code
    "sphinx.ext.intersphinx",    # Link to other projects' documentation
    "sphinx.ext.mathjax",        # Math support
    "sphinx_copybutton",         # Copy button for code blocks
    "sphinx_rtd_theme",          # Read the Docs theme
    "myst_parser",               # Markdown support
    "nbsphinx",                  # Jupyter notebook support
    "sphinx.ext.autosummary",    # Generate autosummary pages
]

# Extension settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__"
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "biopython": ("https://biopython.org/docs/latest/api/", None),
}

# Theme settings
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "titles_only": False,
    "logo_only": True,
    "display_version": True,
    "prev_next_buttons_location": "both",
    "style_external_links": True,
    "style_nav_header_background": "#2980B9",
}

# Static files
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["custom.js"]

# Logo
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"

# Custom sidebar templates
html_sidebars = {
    "**": [
        "relations.html",
        "searchbox.html",
        "navigation.html",
    ]
}

# Output settings
html_title = f"{project} {version} documentation"
html_short_title = project
html_show_sphinx = False
html_show_copyright = True

# Notebook settings
nbsphinx_execute = "never"  # Don't execute notebooks during build
nbsphinx_allow_errors = True
nbsphinx_timeout = 60

# Markdown settings
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
] 