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
sys.path.insert(0, os.path.abspath('../../isofit'))


# -- Project information -----------------------------------------------------

project = 'ISOFIT: Imaging Spectrometer Optimal FITting'
copyright = 'Copyright 2018 California Institute of Technology'
author = ''' D. R. Thompson, P. G. Brodrick, W. Olson Duvall, others'''


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.intersphinx',
    'autoapi.extension',
    'myst_parser'
]

autoapi_dirs = ['../../isofit']
autoapi_ignore = ['**/templates/*']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'sklearn': ('https://sklearn-features.readthedocs.io/en/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None)
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# make sure things will work with readthedocs
autodoc_mock_imports = ['ray']
master_doc = 'index'

autodoc_member_order = 'bysource'
todo_include_todos = True
add_module_names = False
modindex_common_prefix = ['isofit']
