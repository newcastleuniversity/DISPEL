"""All the fixtures needed to compute unit tests in the ``draw`` module."""

import pandas as pd
import pytest

from dispel.data.levels import Level
from dispel.providers.generic.tasks.cps.modalities import CPSLevel
from dispel.providers.generic.tasks.cps.utils import transform_user_input


@pytest.fixture
def cps_example():
    """Get a fixture of actual (ans) vs expected (displayed) digits/symbols."""
    data = pd.DataFrame([[1, 1], [2, 3]], columns=["actual", "expected"])
    return data


@pytest.fixture
def level_std():
    """Crate a fake instance of Level for symbol to digit modality."""
    start = "2020-01-01 15:20:30"
    end = "2020-01-01 15:20:35"
    level = Level(id_=CPSLevel.SYMBOL_TO_DIGIT, start=start, end=end)
    return level


@pytest.fixture
def level_dtd():
    """Crate a fake instance of Level for digit to digit modality."""
    start = "2020-01-01 15:20:30"
    end = "2020-01-01 15:20:35"
    level = Level(id_=CPSLevel.DIGIT_TO_DIGIT, start=start, end=end)
    return level


@pytest.fixture
def success_data():
    """Create a fixture of an example of CPS success data set."""
    data = pd.DataFrame({"success": [True, True, True, False, False, True]})
    return data


@pytest.fixture
def data_n_backs(level_dtd):
    """Create a dummy data frame to study nback."""
    data = pd.DataFrame(
        [
            [
                1,
                1,
                pd.Timestamp("2020-05-15 16:12:38.011"),
                pd.Timestamp("2020-05-15 16:12:39.071"),
            ],
            [
                2,
                2,
                pd.Timestamp("2020-05-15 16:12:40.011"),
                pd.Timestamp("2020-05-15 16:12:41.071"),
            ],
            [
                1,
                1,
                pd.Timestamp("2020-05-15 16:12:42.011"),
                pd.Timestamp("2020-05-15 16:12:43.071"),
            ],
            [
                1,
                1,
                pd.Timestamp("2020-05-15 16:12:44.011"),
                pd.Timestamp("2020-05-15 16:12:45.071"),
            ],
            [
                1,
                1,
                pd.Timestamp("2020-05-15 16:12:46.011"),
                pd.Timestamp("2020-05-15 16:12:48.071"),
            ],
        ],
        columns=["displayedValue", "userValue", "tsDisplay", "tsAnswer"],
    )
    new_data = transform_user_input(data, level_dtd)
    return new_data
