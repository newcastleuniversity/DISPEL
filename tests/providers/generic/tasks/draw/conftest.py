"""All the fixtures needed to compute unit tests in the ``draw`` module."""

import pandas as pd
import pytest

from dispel.providers.generic.tasks.draw.intersections import (
    get_intersection_data,
    get_segment,
)


@pytest.fixture
def example_paths():
    """Create a data frame fixture with 2 points of user and model paths."""
    data = {"x": [0, 0, 0], "y": [0, 1, 2]}
    return pd.DataFrame(data)


@pytest.fixture
def example_segments(example_paths):
    """Create a fixture of segment objects based on `example_paths`."""
    return get_segment(example_paths)


@pytest.fixture
def data_to_flag_as_valid():
    """Create a fixture of example data to flags as valid or not."""
    dictionary = {
        "x": [60.0, 60.0, 0.0, 0.0],
        "y": [60.0, 60.0, 0.0, 0.0],
        "tsTouch": ["2021", "2021", "2021", "2021"],
        "touchAction": ["down", "move", "down", "move"],
    }
    return pd.DataFrame(dictionary)


@pytest.fixture
def intersection_data():
    """Create an example of data which intersect in two points."""
    data = pd.DataFrame(
        {
            "x": [0, 1, 2, 3, 4, 5, 6],
            "y": [0, 1, 2, 3, 2, 1, 0],
            "tsTouch": pd.date_range("now", periods=7, freq="1s"),
        }
    )
    model = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5, 6], "y": [3, 2, 1, 0, 1, 2, 3]})
    return data, model


@pytest.fixture
def intersection_data_formatted(intersection_data):
    """Create an example of formatted data which intersect in two points."""
    data, model = intersection_data
    user, ref = get_intersection_data(data, model)
    return user, ref
