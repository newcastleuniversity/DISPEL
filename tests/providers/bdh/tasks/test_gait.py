"""Tests for :mod:`dispel.providers.bdh.tasks.gait`."""
from copy import deepcopy

from dispel.data.collections import FeatureSet
from dispel.processing import process
from dispel.providers.bdh.io import read_bdh
from dispel.providers.bdh.tasks.gait import BDHGaitStepsInclLee
from tests.processing.helper import assert_dict_values, assert_unique_feature_ids
from tests.providers.bdh.conftest import (
    EXAMPLE_PATH_6MWT,
    RESULTS_6MWT_EXP_HIGH,
    RESULTS_6MWT_EXP_LOW,
)


def test_6mwt_process():
    """Unit test to ensure the 6MWT features are well computed."""
    reading = deepcopy(read_bdh(EXAMPLE_PATH_6MWT))
    process(reading, BDHGaitStepsInclLee())

    feature_set = reading.get_level("6mwt").feature_set
    assert isinstance(feature_set, FeatureSet)

    assert_dict_values(feature_set, RESULTS_6MWT_EXP_LOW)
    assert_dict_values(feature_set, RESULTS_6MWT_EXP_HIGH, relative_error=1e-2)

    # Assert features are unique
    assert_unique_feature_ids(reading)
