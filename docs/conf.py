# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import wofry

# -- Project information -----------------------------------------------------

project = 'wofry'
copyright = '2024, M. Sanchez del Rio, L. Rebuffi'
author = 'M. Sanchez del Rio, L. Rebuffi'
release = '0.1'

# -- General configuration ---------------------------------------------------

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.viewcode',
              'sphinx.ext.napoleon',
              'sphinx.ext.autosummary',
              'nbsphinx'
]

autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_context = {
    'display_github': True,
    'github_user': 'oasys-kit',
    'github_repo': 'wofry',
    'github_version': 'master/docs/',
}
