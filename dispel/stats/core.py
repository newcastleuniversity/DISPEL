"""Core functions to calculate statistics."""
import math
from functools import partial
from typing import Callable, Literal

import numpy as np
import pandas as pd


def mad(data: np.ndarray, axis=None):
    """Compute mean absolute deviation.

    Parameters
    ----------
    data
        The data from which to calculate the mean absolute deviation.
    axis
        The axis along which to calculate the mean absolute deviation.

    Returns
    -------
    numpy.ndarray
        The mean absolute deviation.
    """
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)


def variation(
    data: pd.Series, error: Literal["raise", "coerce", "omit"] = "coerce"
) -> float:
    """Compute the coefficient of variation of a series.

    Parameters
    ----------
    data
        A pandas series for which to compute the coefficient of variation.
    error
        Defines how to handle when the data mean is null is raised. The
        following options are available (default is ``coerce``)

            - ``raise``: ``ZeroDivisionError`` will be raised.
            - ``coerce``: variation will be set as 0.
            - ``omit``: variation will return nan.

    Returns
    -------
    float
        The coefficient of variation of the data using an unbiased standard
        deviation computation.

    Raises
    ------
    ZeroDivisionError
        If the data mean is null and the argument ``error`` is set to
        ``raised``.
    ValueError
        If the argument ``error`` is given an unsupported value.

    Examples
    --------
    Here are a few usage examples:

    .. testsetup:: variation

        import warnings
        import pandas as pd

        warnings.simplefilter('ignore', RuntimeWarning)

    .. doctest:: variation

        >>> import pandas as pd
        >>> from dispel.stats.core import variation
        >>> x = pd.Series([3.2, 4.1, 0., 1., -6.])
        >>> variation(x)
        8.626902135395195

    In case of ``ZeroDivisionError``, one can use the ``error`` argument to
    control the output.

    .. doctest:: variation

        >>> x = pd.Series([3., -4., 0., 2., -1.])
        >>> variation(x)
        0.0

        >>> x = pd.Series([3., -4., 0., 2., -1.])
        >>> variation(x, error='raise')
        Traceback (most recent call last):
        ...
        ZeroDivisionError: Cannot divide by null mean.

        >>> x = pd.Series([3., -4., 0., 2., -1.])
        >>> variation(x, error='omit')
        nan
    """
    res = data.std() / data.mean()
    if math.isinf(res):
        if error == "coerce":
            return 0.0
        if error == "omit":
            return np.nan
        if error == "raise":
            raise ZeroDivisionError("Cannot divide by null mean.")

        raise ValueError("Unsupported ``error`` value.")
    return res


# Setting variation name acronym
variation.__name__ = "cv"


def variation_increase(
    data: pd.Series, error: Literal["raise", "coerce", "omit"] = "coerce"
) -> float:
    """Compute the coefficient of variation increase for a series.

    The coefficient of variation increase corresponds to the the CV of the
    second half of the data minus that of the first half.

    Parameters
    ----------
    data
        A pandas series for which to compute the coefficient of variation
        increase.
    error
        Defines how to handle when the data mean is null is raised. The
        following options are available (default is ``coerce``)

            - ``raise``: ``ZeroDivisionError`` will be raised.
            - ``coerce``: variation will be set as 0.
            - ``omit``: variation will return nan.

    Returns
    -------
    float
        The coefficient of variation increase of the data using an unbiased
        standard deviation computation.

    Examples
    --------
    Here are a few usage examples:

    .. testsetup:: variation_increase

        import warnings
        import pandas as pd

        warnings.simplefilter('ignore', RuntimeWarning)

    .. doctest:: variation_increase

        >>> import pandas as pd
        >>> from dispel.stats.core import variation_increase
        >>> x = pd.Series([3.2, 4.1, 0., 1., -6.])
        >>> variation_increase(x)
        -2.4459184350510386

    In case of ``ZeroDivisionError``, one can use the ``error`` argument to
    control the output.

    .. doctest:: variation_increase

        >>> x = pd.Series([3., -4., 0., 1., -1.])
        >>> variation_increase(x, error='raise')
        Traceback (most recent call last):
        ...
        ZeroDivisionError: Cannot divide by null mean.

        >>> x = pd.Series([3., -3., 0., 2., -1.])
        >>> variation_increase(x, error='omit')
        nan
    """
    first_half = data[: (half_idx := len(data) // 2)]
    second_half = data[half_idx:]
    return variation(second_half, error) - variation(first_half, error)


# Setting variation increase name acronym
variation_increase.__name__ = "cvi"


def q_factory(percentile: float, name: str) -> Callable[[pd.Series], float]:
    """Create percentile aggregation method.

    Parameters
    ----------
    percentile
        The percentile used in the aggregation. This is passed to
        :meth:`pandas.Series.quantile`.
    name
        The name of the method.

    Returns
    -------
    Callable[[pandas.Series], float]
        Returns a callable aggregation method that returns the percentile
        specified in `percentile`.
    """
    func = partial(pd.Series.quantile, q=percentile)
    func.__name__ = name  # type: ignore
    return func


#: First quartile (Q1) aggregation
q1 = q_factory(0.25, "q1")

#: Third quartile (Q3) aggregation
q3 = q_factory(0.75, "q3")

#: Percentile 0.5 aggregation
percentile_05 = q_factory(0.05, "q05")

#: Percentile 0.95 aggregation
percentile_95 = q_factory(0.95, "q95")


def freq_nan(data: pd.Series) -> float:
    """Get the frequency of null values."""
    return data.isnull().sum() / len(data)


def iqr(data: pd.Series) -> float:
    """Compute the inter-quartile range."""
    return data.quantile(q=0.75) - data.quantile(q=0.25)


def npcv(data: pd.Series) -> float:
    """Compute the non-parametric coefficient of variation of a series."""
    return mad(data) / data.abs().mean()
