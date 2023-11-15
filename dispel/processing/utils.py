"""Utility functions around data processing."""
from abc import ABCMeta
from itertools import zip_longest
from typing import Union

import pandas as pd

from dispel.data.values import AbbreviatedValue as AV


class TaskMixin(metaclass=ABCMeta):
    """A mixin class for entities related to tasks."""

    #: The task name
    task_name: Union[AV, str]

    def get_task_name(self, **kwargs) -> Union[str, AV]:
        """Get the task name."""
        task_name = kwargs.get("task_name", None) or getattr(self, "task_name")
        if isinstance(task_name, (str, AV)):
            return task_name
        raise ValueError("Missing task name.")


def parallel_explode(data: pd.DataFrame, dtype="float64") -> pd.DataFrame:
    """Transform each element of a list-like to a row for all columns.

    Parameters
    ----------
    data
        The data pandas data frame to be exploded.
    dtype
        The type of the data frame values.

    Returns
    -------
    pandas.DataFrame
        Exploded lists to rows of all columns.

    Examples
    --------
    .. testsetup:: parallel_explode

        import pandas as pd
        from dispel.processing.utils import parallel_explode

    .. doctest:: parallel_explode

        >>> df = pd.DataFrame({
        ...        'a': [[2.], [3., 4.], [], [6., 7.]],
        ...        'b': [[8.], [9., 10.], [11.], [12., 13.]],
        ...    })
        >>> df
                    a             b
        0       [2.0]         [8.0]
        1  [3.0, 4.0]   [9.0, 10.0]
        2          []        [11.0]
        3  [6.0, 7.0]  [12.0, 13.0]
        >>> parallel_explode(df)
             a     b
        0  2.0   8.0
        1  3.0   9.0
        2  4.0  10.0
        3  6.0  11.0
        4  7.0  12.0
        5  NaN  13.0
    """
    return pd.DataFrame(
        (
            exploded_column
            for exploded_column in zip_longest(
                *[data[column].explode().dropna() for column in data]
            )
        ),
        columns=data.columns,
        dtype=dtype,
    )
