"""General utility functions."""
import re
from functools import update_wrapper
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from multimethod import multimethod


def camel_to_snake_case(value: str) -> str:
    """Transform the camel case string into a snake one.

    Parameters
    ----------
    value
        The string to process

    Returns
    -------
    str
        The passed string in lower case camel format.

    Raises
    ------
    TypeError
        When the given value is not a string.
    """
    if not isinstance(value, str):
        raise TypeError("value must be a string")

    return re.sub("([a-z])([A-Z])", r"\1_\2", value).lower()


def to_camel_case(value: str) -> str:
    """
    Transform a string into a camel case string.

    Parameters
    ----------
    value
        The string to process.

    Returns
    -------
    str
        The passed string in camel case format.
    """
    name = re.split("([^a-zA-Z0-9])", value)
    return name[0].lower() + "".join(a.capitalize() for a in name[1:] if a.isalnum())


def drop_none(data: List[Any]) -> List[Any]:
    """Drop ``None`` elements from a list."""
    return list(filter(lambda x: x is not None, data))


def convert_column_types(
    data: pd.DataFrame, column_type_fetcher: Callable[[Any], str]
) -> pd.DataFrame:
    """Convert columns of a pandas data frame to their indicated types.

    Parameters
    ----------
    data
        The data frame to be converted
    column_type_fetcher
        A function that retrieves the column type given its name.

    Returns
    -------
    pandas.DataFrame
        A pandas data frame containing the converted columns.
    """
    # change data types if needed
    for column, type_ in data.dtypes.iteritems():
        try:
            expected_type = np.dtype(column_type_fetcher(column))
        except Exception:  # pylint: disable=broad-except
            expected_type = np.dtype("U")
        if expected_type != type_:
            if np.issubdtype(expected_type, np.datetime64):
                try:
                    data[column] = pd.to_datetime(data[column], unit="ms")
                except ValueError:
                    data[column] = data[column].astype(expected_type)
            elif np.issubsctype(expected_type, bool):
                data[column] = data[column].replace(
                    {"true": True, "false": False, "True": True, "False": False}
                )
            elif np.issubsctype(expected_type, np.float32):
                data[column] = (
                    data[column]
                    .apply(lambda x: None if x == "null" else x)
                    .astype(expected_type)
                )
            else:
                data[column] = data[column].astype(expected_type)
    return data


def raise_multiple_errors(errors: List[Exception]):
    """Re-raise multiple exceptions one after the other.

    Parameters
    ----------
    errors
        List of exceptions to raise

    Raises
    ------
    Exception
        The exceptions given as input
    """
    if len(errors) == 0:
        return
    try:
        raise errors.pop(0)
    finally:
        raise_multiple_errors(errors)


def plural(single: str, count: int, multiple: Optional[str] = None) -> str:
    """Associate the word with its count.

    Parameters
    ----------
    single
        The single word.
    count
        The count of the given count.
    multiple
        The plural word of single if it shouldn't just end with an `s`.

    Returns
    -------
    str
        Its associated plural.
    """
    if multiple is None:
        multiple = single + ("s" if count != 1 else "")
    return f"{count} {single if count == 1 else multiple}"


class multidispatchmethod:  # pylint: disable=invalid-name
    """Multi-dispatch generic method descriptor."""

    def __init__(self, func: Callable[..., Any]):
        if not callable(func) and not hasattr(func, "__get__"):
            raise TypeError(f"{func} is not callable or a descriptor")

        self.dispatcher = multimethod(func)
        self.func = func

    def register(self, cls, method=None):
        """Register the method."""
        return self.dispatcher.register(cls, method)

    def __get__(self, obj, cls):
        def _method(*args, **kwargs):
            method = self.dispatcher[tuple(map(self.dispatcher.get_type, args))]
            return method(obj, *args, **kwargs)

        _method.__isabstractmethod__ = self.__isabstractmethod__
        _method.register = self.register
        update_wrapper(_method, self.func)
        return _method

    @property
    def __isabstractmethod__(self) -> bool:
        return getattr(self.func, "__isabstractmethod__", False)


def set_attributes_from_kwargs(
    obj: object, *attrs: str, pop: bool = True, **kwargs
) -> Dict[str, Any]:
    """Set attributes in object from kwargs.

    Parameters
    ----------
    obj
        The class object where the attributes are to be set.
    attrs
        The names of teh attributes that are to be set in the object.
    pop
        ``True`` if the attributes are to popped from the provided keyword arguments.
        ``False`` otherwise.
    kwargs
        The keyword argument from which the attributes are to be retrieved as well. If
        no values corresponding to the provided attributes are found a ``None`` value is
        set instead.

    Returns
    -------
    Dict[str, Any]
        A dictionary of the remaining key word arguments.
    """
    func = getattr(kwargs, "pop" if pop else "get")
    for attribute in attrs:
        kwargs_attribute = func(attribute, None)
        new_attribute = kwargs_attribute or getattr(obj, attribute, None)
        setattr(obj, attribute, new_attribute)
    return kwargs
