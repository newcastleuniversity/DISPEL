"""Test for :mod:`dispel.providers.generic.flags.le_flags`."""

import numpy as np
import pandas as pd

from dispel.providers.generic.flags.le_flags import truncate_first_5_turns


def get_placement_example_data():
    """
    Get an example of placement and turn input and its truncated output.

    We define a dataframe where there are 4 bouts and 6 turns and after truncation of
    first 5 turns only 2 bouts remain.

    Returns
    -------
    (tuple): tuple containing:

        data (pandas.DataFrame) Dataframe containing placement input turns
        (pandas.DataFrame) Dataframe containing turns input expected (Dict): contains
        the expected values to be tested

    """
    # start timestamp common for all data
    time_0 = np.datetime64("2020-01-01T00:00:00.000")

    # Define the turn dataframe
    # define turn durations and deltas from start
    start_deltas = np.array([3, 6, 9, 12, 15, 18])
    duration = np.array([1, 1, 1, 1, 1, 1])
    end_deltas = start_deltas + duration
    # define actual datetimes
    start_time = [time_0 + np.timedelta64(delta, "s") for delta in start_deltas]
    end_time = [time_0 + np.timedelta64(delta, "s") for delta in end_deltas]

    turns = pd.DataFrame({"start": start_time, "end": end_time, "duration": duration})

    # Define the placement dataframe
    labels = ["belt", "pants", "else", "handheld"]
    # define bout durations and deltas from start
    duration = np.array([10, 10, 10, 10])
    start_deltas = np.cumsum(duration) - duration[0]
    end_deltas = start_deltas + duration
    # define actual datetimes
    start_time = [time_0 + np.timedelta64(delta, "s") for delta in start_deltas]
    end_time = [time_0 + np.timedelta64(delta, "s") for delta in end_deltas]

    data = pd.DataFrame(
        {
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "placement": labels,
        }
    )

    # Expected truncated data
    labels = ["belt", "pants"]
    # define bout durations and deltas from start
    duration = [10, 6]
    start_deltas = np.cumsum(duration) - duration[0]
    end_deltas = start_deltas + duration
    # define actual datetimes
    start_time = [time_0 + np.timedelta64(delta, "s") for delta in start_deltas]
    end_time = [time_0 + np.timedelta64(delta, "s") for delta in end_deltas]

    expected = pd.DataFrame(
        {
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "placement": labels,
        }
    )
    return data, turns, expected


def test_truncation():
    """Unit test to ensure placement produces expected result."""
    data, turns, expected = get_placement_example_data()
    result = truncate_first_5_turns(data, turns)

    assert isinstance(result, type(expected))
    assert result.shape == expected.shape
    assert all(result["placement"] == expected["placement"])
    assert all(result["duration"] == expected["duration"])
