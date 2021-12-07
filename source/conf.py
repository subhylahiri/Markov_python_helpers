"""Configuration file for the Sphinx documentation builder.
"""
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
# pylint: disable=invalid-name


# -- Project information -----------------------------------------------------

project = 'Markov Python Helpers'
copyright = '2021, Subhaneil Lahiri'
author = 'Subhaneil Lahiri'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    'sphinx.ext.intersphinx',  # Link to other project's documentation (see mapping below)
    "sphinx.ext.mathjax",
    "numpydoc",
    # 'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints', # Automatically document param types (less noise in class signature)
]
# My options for extensions:
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'member_order': "bysource",
}
autodoc_inherit_docstrings = True
autosummary_generate = True
autosummary_imported_members = True
autosummary_ignore_module_all = False
numpydoc_edit_link = False
add_module_names = False
# numpydoc_xref_param_type = True
# numpydoc_class_members_toctree = False
# Mappings for sphinx.ext.intersphinx. Projects have to have Sphinx-generated doc! (.inv file)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "autolink"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    'custom.css',
]
html_show_sourcelink = False
# html_sidebars = {
#    '**': ['localtoc.html', 'globaltoc.html', 'relations.html', 'searchbox.html'],
# }