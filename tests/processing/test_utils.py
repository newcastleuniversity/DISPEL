"""Tests for :mod:`dispel.processing.utils`."""
import pandas as pd

from dispel.processing.utils import parallel_explode


def test_parallel_explode():
    """Testing :func:`dispel.processing.utils.flatten_data_frame`."""
    data = pd.DataFrame(
        {
            "a": [[2.0], [3.0, 4.0], [5.0], [6.0, 7.0]],
            "b": [[8.0], [9.0, 10.0], [11.0], [12.0, 13.0]],
        }
    )
    expected = pd.DataFrame(
        {"a": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0], "b": [8.0, 9.0, 10.0, 11.0, 12.0, 13.0]}
    )

    assert parallel_explode(data).equals(expected)
