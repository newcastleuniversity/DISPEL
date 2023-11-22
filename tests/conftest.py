"""Fixture definitions for :mod:`tests` module."""
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pkg_resources
import pytest


def resource_path(path: str, module: str = "io") -> str:
    """Get the absolute path to a resource within testing."""
    return pkg_resources.resource_filename(f"tests.{module}", f"_resources/{path}")


pytest_plugins = [
    "tests.processing.test_core",
]


@contextmanager
def noop():
    """Define a no-op context manager."""
    yield


@pytest.fixture
def collection_data_frame():
    """Fixture representing the info obtained from a measures csv file."""
    data = pd.DataFrame(
        {
            "subject_id": {
                0: "user01",
                1: "user02",
                2: "user01",
                3: "user02",
                4: "user01",
                5: "user02",
                6: "user01",
                7: "user02",
                8: "user02",
                9: "user01",
            },
            "evaluation_code": {
                0: "evaluation",
                1: "evaluation",
                2: "evaluation",
                3: "evaluation",
                4: "evaluation",
                5: "evaluation",
                6: "evaluation",
                7: "evaluation",
                8: "evaluation",
                9: "evaluation",
            },
            "evaluation_uuid": {
                0: "evaluation_0101",
                1: "evaluation_0201",
                2: "evaluation_0102",
                3: "evaluation_0202",
                4: "evaluation_0101",
                5: "evaluation_0201",
                6: "evaluation_0102",
                7: "evaluation_0202",
                8: "evaluation_0203",
                9: "evaluation_0103",
            },
            "session_uuid": {
                0: "session_0101",
                1: "session_0102",
                2: "session_0103",
                3: "session_0104",
                4: "session_0105",
                5: "session_0106",
                6: "session_0107",
                7: "session_0108",
                8: "session_0109",
                9: "session_0110",
            },
            "session_code": {
                0: "daily",
                1: "daily",
                2: "daily",
                3: "daily",
                4: "daily",
                5: "daily",
                6: "daily",
                7: "daily",
                8: "daily",
                9: "daily",
            },
            "start_date": {
                0: "01/02/2020",
                1: "01/01/2020",
                2: "01/01/2020",
                3: "01/01/2019",
                4: "01/02/2020",
                5: "01/01/2020",
                6: "01/01/2020",
                7: "01/01/2019",
                8: "01/06/2020",
                9: "01/07/2020",
            },
            "end_date": {
                0: "01/02/2020",
                1: "01/01/2020",
                2: "01/01/2020",
                3: "01/01/2019",
                4: "01/02/2020",
                5: "01/01/2020",
                6: "01/01/2020",
                7: "01/01/2019",
                8: "01/06/2020",
                9: "01/07/2020",
            },
            "is_finished": {
                0: True,
                1: False,
                2: True,
                3: True,
                4: True,
                5: True,
                6: True,
                7: True,
                8: True,
                9: True,
            },
            "measure_id": {
                0: "feat_01",
                1: "feat_01",
                2: "feat_01",
                3: "feat_01",
                4: "feat_02",
                5: "feat_02",
                6: "feat_02",
                7: "feat_02",
                8: "feat_02",
                9: "feat_02",
            },
            "measure_name": {
                0: "measure_01",
                1: "measure_01",
                2: "measure_01",
                3: "measure_01",
                4: "measure_02",
                5: "measure_02",
                6: "measure_02",
                7: "measure_02",
                8: "measure_02",
                9: "measure_02",
            },
            "measure_value": {
                0: 10.0,
                1: 4.0,
                2: 8.0,
                3: 5.0,
                4: 4.0,
                5: 7.0,
                6: 3.0,
                7: 6.0,
                8: 6.0,
                9: np.nan,
            },
            "measure_unit": {
                0: "s",
                1: "s",
                2: "s",
                3: "s",
                4: "s",
                5: "s",
                6: "s",
                7: "s",
                8: "s",
                9: "s",
            },
            "measure_type": {
                0: "float64",
                1: "float64",
                2: "float64",
                3: "float64",
                4: "float64",
                5: "float64",
                6: "float64",
                7: "float64",
                8: "float64",
                9: "float64",
            },
            "trial": {0: 2, 1: 2, 2: 1, 3: 1, 4: 2, 5: 2, 6: 1, 7: 1, 8: 3, 9: 3},
        }
    )
    for column in ("start_date", "end_date"):
        data[column] = data[column].astype("datetime64[ms]")
    data["trial"] = data["trial"].astype("int16")
    data.sort_values("start_date", inplace=True, ignore_index=True)
    return data
