"""Test for :mod:`dispel.providers.generic.activity.placement`."""

from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from dispel.providers.generic.activity.placement import placement_classification_merged


def get_placement_example_data():
    """Get an example of placement input and its expected output.

    We define a dataframe where the gravity is linearly transitioning from "perfect"
    landscape to perfect "face up", then turns into a "perfect" portrait mode. The test
    data will have a period of transition and a period of static orientation. The
    acceleration norm is alternating from high dynamics (norm=1g) to no dynamics
    (norm=0g).

    Returns
    -------
    (tuple): tuple containing:

        data (pandas.DataFrame) Dataframe containing placement input expected (Dict):
        contains the expected values to be tested

    """
    # gravity
    gravity = [[]] * 10
    gravity[0] = [1, 0, 0]
    gravity[1] = [1, 1, 0]
    gravity[2] = [1, 1, 1]
    gravity[3] = [0, 1, 1]
    gravity[4] = [0, 0, 1]
    gravity[5] = [0, -1, 1]
    gravity[6] = [-1, -1, 1]
    gravity[7] = [-1, -1, 0]
    gravity[8] = [0, -1, 0]
    gravity[9] = [0, -1, 0]

    gravity = [g / np.linalg.norm(g) for g in gravity]
    gravity = np.array(gravity)

    # acceleration norm
    norm = np.array([1, 0] * 5)
    norm = norm[:, None]

    # concatenate gravity and accelerometer norm
    data = np.concatenate((gravity, norm), axis=1)
    data = np.repeat(data, 2, axis=0)

    # interpolate to 50 Hz
    x_interp = np.arange(0, 160, 1 / 50)
    x_init = np.arange(0, 160, 8)
    data_interp = np.array([np.interp(x_interp, x_init, i) for i in data.T]).T

    # create dataframe
    datetime_0 = np.datetime64("2020-01-01T00:00:00.000")
    datetime_index = [datetime_0 + np.timedelta64(int(x_ * 1000)) for x_ in x_interp]

    # expected
    expected: Dict[str, type, Union[Tuple[int, int], List[str], List[float]]] = {
        "shape_": (6, 4),
        "type_": pd.DataFrame,
        "labels": ["belt", "else", "table", "else", "belt", "pants"],
        "durations": [30.6, 34.0, 6.9, 34.0, 15.6, 38.90],
    }

    data = pd.DataFrame(
        data_interp,
        index=datetime_index,
        columns=["gravityX", "gravityY", "gravityZ", "norm"],
    )
    return data, expected


def test_placement():
    """Unit test to ensure placement produces expected result."""
    data, expected = get_placement_example_data()
    result = placement_classification_merged(
        data.loc[:, ["gravityX", "gravityY", "gravityZ"]], data.loc[:, ["norm"]]
    )

    assert isinstance(result, expected["type_"])
    assert result.shape == expected["shape_"]
    assert all(result.placement.values == expected["labels"])
    np.testing.assert_almost_equal(result.duration.values, expected["durations"])
