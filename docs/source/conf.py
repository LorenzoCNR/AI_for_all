"""Sphinx configuration for the NeuroBridge documentation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

project = "NeuroBridge"
author = "Lorenzo Ognibeni"
copyright = "2026, Lorenzo Ognibeni"
release = "0.0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = []
html_static_path = ["_static"]
html_css_files = ["neurobridge.css"]

if importlib.util.find_spec("pydata_sphinx_theme") is not None:
    html_theme = "pydata_sphinx_theme"
    html_theme_options = {
        "show_toc_level": 2,
        "navigation_with_keys": True,
        "collapse_navigation": False,
        "navbar_align": "left",
        "secondary_sidebar_items": ["page-toc", "edit-this-page"],
        "icon_links": [
            {
                "name": "GitHub",
                "url": "https://github.com/LorenzoCNR",
                "icon": "fa-brands fa-github",
            },
        ],
    }
else:
    html_theme = "alabaster"

html_title = "NeuroBridge documentation"
html_short_title = "NeuroBridge"
html_show_sourcelink = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}
