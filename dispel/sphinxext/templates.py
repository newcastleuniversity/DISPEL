"""Templates for library specific Sphinx extensions."""
from os.path import dirname, join
from typing import Any

from jinja2 import Environment, FileSystemLoader

from dispel.data.validators import RangeValidator, SetValidator

_templates_path = join(dirname(__file__), "_templates")


def is_range_validator(validator: Any) -> bool:
    """Test if an object is a :class:`~dispel.data.validators.RangeValidator`.

    Parameters
    ----------
    validator
        The validator to be tested.

    Returns
    -------
    bool
        ``True`` if the passed argument is an instance of a ``RangeValidator``.

    """
    return isinstance(validator, RangeValidator)


def is_set_validator(validator: Any) -> bool:
    """Test if an object is a :class:`~dispel.data.validators.SetValidators`.

    Parameters
    ----------
    validator
        The validator to be tested.

    Returns
    -------
    bool
        ``True`` if the passed argument is an instance of a ``SetValidator``.

    """
    return isinstance(validator, SetValidator)


_env = Environment(loader=FileSystemLoader(_templates_path))
_env.tests["range_validator"] = is_range_validator
_env.tests["set_validator"] = is_set_validator

MEASURES_DETAIL = _env.get_template("measure_detail.rst")
