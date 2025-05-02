# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
import glob

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))

project = 'Toponymy'
copyright = '2025, Leland McInnes and John Healy'
author = 'Leland McInnes and John Healy'
release = '0.3'

master_doc = "index"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "nbsphinx",
    "sphinx.ext.mathjax",
    #"sphinx_gallery.gen_gallery",
    # "sphinx_build_compatibility.extension",
]
# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_show_class_members = False

autodoc_default_flags = ['members', 'inherited-members']
autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_theme_options = {"navigation_depth": 3, "logo_only": False}

html_logo = "toponymy_logo_rtd.png"

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/{.major}".format(sys.version_info), None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scikit-learn": ("https://scikit-learn.org/", None),
}

nbsphinx_allow_errors = True