# SPDX-FileCopyrightText: Contributors to NPAP
# SPDX-License-Identifier: MIT

"""Sphinx configuration for NPAP documentation."""

import os
import sys

# Add package to path for autodoc
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "NPAP"
copyright = "2026, NPAP Contributors"
author = "Marco Anarmo"

# Version is read from the package
try:
    from importlib.metadata import version as get_version

    release = get_version("npap")
    version = ".".join(release.split(".")[:2])
except Exception:
    version = release = "dev"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser",
    "numpydoc",
    "sphinxcontrib.mermaid",
    "sphinx_reredirects",
]

# Redirects for old URLs
redirects = {
    "user-guide/getting-started": "installation.html",
    "user-guide/partitioning": "partitioning/index.html",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "diagram_prompts/**"]

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The master toctree document
master_doc = "index"

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "NPAP"

# Logo and favicon
html_logo = "assets/NPAP.svg"
html_favicon = "assets/NPAP.svg"

html_theme_options = {
    # Logo configuration
    "logo": {
        "image_light": "assets/NPAP.svg",
        "image_dark": "assets/NPAP.svg",
        "text": "NPAP",
    },
    # Note: CSS variables are set in _static/custom.css for full control
    # Navbar configuration - clean header with search on right
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["search-field", "theme-switcher", "navbar-icon-links"],
    "navbar_persistent": [],
    "navbar_align": "left",
    "search_bar_text": "Search",
    # Header links / icons
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/IEE-TUGraz/NPAP",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/npap",
            "icon": "fa-brands fa-python",
            "type": "fontawesome",
        },
    ],
    # Primary sidebar - disable on landing page
    "primary_sidebar_end": [],
    # Navigation and UI
    "show_nav_level": 2,
    "show_toc_level": 2,
    "navigation_with_keys": True,
    # Footer
    "footer_start": ["copyright"],
    "footer_end": [],
    # Secondary sidebar (right TOC) - will be hidden on index via html_sidebars
    "secondary_sidebar_items": ["page-toc"],
    # Misc
    "show_prev_next": True,
}

# GitHub repository configuration
html_context = {
    "github_user": "IEE-TUGraz",
    "github_repo": "NPAP",
    "github_version": "main",
    "doc_path": "docs",
}

# Hide all sidebars on index (landing) page for clean PyPSA-style look
html_sidebars = {
    "index": [],
}

# -- Extension configuration -------------------------------------------------

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"

# Autosummary settings
autosummary_generate = True

# Napoleon settings (NumPy style)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# Numpydoc settings
numpydoc_show_class_members = True
numpydoc_class_members_toctree = False

# MyST-Parser settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
    "attrs_inline",
    "attrs_block",
    "dollarmath",
    "amsmath",
]
myst_heading_anchors = 3
myst_dmath_double_inline = True  # Allow $$ for display math

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# Copybutton settings
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# Mermaid settings - NPAP branded theme
# Note: Theme is controlled via CSS for light/dark mode support
mermaid_version = "11"
mermaid_init_js = """
mermaid.initialize({
    startOnLoad: true,
    theme: 'base',
    themeVariables: {
        // Core brand colors
        primaryColor: '#2993B5',
        primaryTextColor: '#ffffff',
        primaryBorderColor: '#1d6f8a',

        // Secondary/line colors
        lineColor: '#64748b',
        secondaryColor: '#0fad6b',
        tertiaryColor: '#FFBF00',

        // Text colors
        textColor: '#1e293b',

        // Node styling
        nodeBorder: '#1d6f8a',
        nodeTextColor: '#ffffff',

        // Flowchart specific
        clusterBkg: 'rgba(41, 147, 181, 0.1)',
        clusterBorder: '#2993B5',

        // Edge/link colors
        edgeLabelBackground: '#ffffff',

        // Font
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
    },
    flowchart: {
        curve: 'basis',
        padding: 20,
        nodeSpacing: 50,
        rankSpacing: 50,
        htmlLabels: true,
        useMaxWidth: true
    }
});
"""
