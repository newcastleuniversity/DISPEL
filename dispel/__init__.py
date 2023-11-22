"""Top-level package for Dispel."""

__author__ = (
    "Alf Scotland and Gautier Cosne and Adrien Juraver "
    "and Angelos Karatsidis and Joaquin Penalver de Andres"
)
__version__ = "0.0.0"

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dispel")
except PackageNotFoundError:
    # package is not installed
    pass
