"""Tests for :mod:`dispel.providers.generic.tasks.gait.steps` for ads provider."""
from copy import deepcopy

import pytest

from dispel.data.collections import MeasureSet
from dispel.processing import process
from dispel.processing.assertions import (
    AssertEvaluationFinished,
    AssertRawDataSetPresent,
)
from dispel.processing.core import StopProcessingError
from dispel.processing.level import ProcessingStepGroup
from dispel.providers.ads.io import read_ads
from dispel.providers.generic.tasks.gait.steps import (
    TASK_NAME,
    GaitStepsInclLee,
    StepsGPS,
)
from tests.conftest import noop
from tests.processing.helper import assert_dict_values, assert_unique_measure_ids
from tests.providers.ads.conftest import (
    EXAMPLE_PATH_6MWT,
    EXAMPLE_PATH_6MWT_2455,
    EXAMPLE_PATH_6MWT_DUPLICATED_TS,
    EXAMPLE_PATH_6MWT_INF_SPEED,
    EXAMPLE_PATH_6MWT_NO_GPS,
    RESULTS_6MWT_EXP_HIGH,
    RESULTS_6MWT_EXP_LOW,
)

# paths containing example json files

GPS_STEPS = ProcessingStepGroup(
    steps=[
        AssertRawDataSetPresent("gps", "6mwt"),
        AssertEvaluationFinished(),
        StepsGPS(),
    ],
    task_name=TASK_NAME,
)


@pytest.fixture
def user_input(user_input_arg):
    """Create a fixture of an example of 6MW `Reading`."""
    return read_ads(user_input_arg)


@pytest.mark.parametrize(
    "user_input_arg, expected, exception, gps_only, high_rel_err",
    [
        (
            EXAMPLE_PATH_6MWT,
            RESULTS_6MWT_EXP_LOW,
            noop(),
            False,
            RESULTS_6MWT_EXP_HIGH,
        ),
        (
            EXAMPLE_PATH_6MWT_INF_SPEED,
            {
                "6mwt-distance_walked": 538.6518,
                "6mwt-walking_speed_non_stop-mean": 1.433052425249292,
            },
            noop(),
            True,
            None,
        ),
        (
            EXAMPLE_PATH_6MWT_DUPLICATED_TS,
            {
                "6mwt-distance_walked": 646.9174,
                "6mwt-walking_speed_non_stop-mean": 2.007307646286258,
            },
            noop(),
            True,
            None,
        ),
        (
            EXAMPLE_PATH_6MWT_NO_GPS,
            {},
            pytest.raises(StopProcessingError),
            True,
            None,
        ),
        (
            EXAMPLE_PATH_6MWT_2455,
            {
                "6mwt-distance_walked": 653.2846317924559,
                "6mwt-walking_speed_non_stop-mean": 1.9959956824814054,
            },
            noop(),
            True,
            None,
        ),
    ],
)
def test_6mwt_process(user_input, expected, exception, gps_only, high_rel_err):
    """Unit test to ensure the 6MWT measures are well computed."""
    reading = deepcopy(user_input)
    with exception:
        if gps_only:
            process(reading, GPS_STEPS)
        else:
            process(reading, GaitStepsInclLee())

        measure_set = reading.get_merged_measure_set()
        assert isinstance(measure_set, MeasureSet)

        assert_dict_values(measure_set, expected)
        if high_rel_err is not None:
            assert_dict_values(measure_set, high_rel_err, relative_error=1e-2)

        # Assert measures are unique
        assert_unique_measure_ids(reading)
