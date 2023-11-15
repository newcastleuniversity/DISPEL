"""Functionality to read digital artefacts (DA) data.

DA data usually come into export files containing multiple patient. In order to adapt
the data to DISPEL, a preprocessing step has to be applied to the DA export to
explode the data into individual records per patient and per assessement. Once it's done
you can use the following example to get a DISPEL representation of the DA data.

Examples
--------
To read a DA json file and work with the contained data one can read the file:

.. testsetup:: da

    >>> import pkg_resources
    >>> path = pkg_resources.resource_filename(
    ...     'tests.providers.digital_artefacts', '_resources/ft/example-v2.json'
    ... )


.. doctest:: da

    >>> from dispel.providers.digital_artefacts.io import read_da
    >>> reading = read_da(path)

"""
from json import load

from dispel.data.core import Reading
from dispel.providers.digital_artefacts.io.ft import parse_ft_reading


def read_da(path: str) -> Reading:
    """Read a *Digital Artefact* data record.

    Parameters
    ----------
    path
        The path to the digital artefact record to be parsed

    Returns
    -------
    Reading
        The class representation of the record once parsed.
    """
    with open(path, encoding="utf-8") as fs:
        data = load(fs)

    return parse_ft_reading(data)
