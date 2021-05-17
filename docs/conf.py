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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.pardir, 'src')))
print(repr(sys.path), file=sys.stderr)

# -- Project information -----------------------------------------------------

project = 'slp-coroutine'
copyright = '2021, Anselm Kruis'
author = 'Anselm Kruis'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'classic'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# This value contains a list of modules to be mocked up. This is useful when
# some external dependencies are not met at build time and break the building
# process. You may only specify the root package of the dependencies themselves
# and omit the sub-modules:
autodoc_mock_imports = ["stackless"]

# This value selects what content will be inserted into the main body of an
# autoclass directive. The possible values are:
#
# "class"
#     Only the class’ docstring is inserted. This is the default. You can still
#     document __init__ as a separate method using automethod or the members
#     option to autoclass.
#
# "both"
#     Both the class’ and the __init__ method’s docstring are concatenated and
#     inserted.
# "init"
#     Only the __init__ method’s docstring is inserted.
#
# If the class has no __init__ method or if the __init__ method’s docstring is
# empty, but the class has a __new__ method’s docstring, it is used instead.
autoclass_content = "both"
