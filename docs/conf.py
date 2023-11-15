"""Configuration file for the Sphinx documentation builder."""
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from importlib.metadata import version

project = "DIgital Signal ProcEssing Library"
author = (
    "Alf Scotland and Gautier Cosne and Adrien Juraver and Angelos Karatsidis "
    "and Joaquin Penalver de Andres"
)

# The full version, including alpha/beta/rc tags.
release = version("dispel")
# The short X.Y version.
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.mermaid",
    "sphinx_mdinclude",
    "pytest_doctestplus.sphinx.doctestplus",
    "dispel.sphinxext.dispel_features",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

nitpicky = True

# -- Autodoc configuration ---------------------------------------------------

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"

# Don't show class signature with the class' name.
autodoc_class_signature = "separated"

# -- Options for sphinx.ext.intersphinx --------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/dev", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "ground": ("https://ground.readthedocs.io/en/latest/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
