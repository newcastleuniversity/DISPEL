"""Tests for the :mod:`dispel.providers.apdm.tasks.gait` module."""

from dispel.data.features import FeatureSet
from dispel.providers.apdm import PROVIDER_ID
from dispel.providers.apdm.io import read_apdm
from dispel.providers.apdm.tasks.gait import process_2mwt
from tests.processing.helper import (
    assert_dict_values,
    assert_unique_feature_ids,
    read_results,
)
from tests.providers import resource_path
from tests.providers.apdm.test_io import EXAMPLE_2MWT_PATH

EXPECTED_2MWT_PATH = resource_path(PROVIDER_ID, "2mwt/expected.json")


def test_2mwt_process():
    """Unit test to ensure the 2MWT features are well computed."""
    reading = read_apdm(EXAMPLE_2MWT_PATH)
    expected = read_results(EXPECTED_2MWT_PATH, True)

    process_2mwt(reading)
    feature_set = reading.get_level("apdm").feature_set

    assert isinstance(feature_set, FeatureSet)
    assert_dict_values(feature_set, expected, relative_error=1e-2)
    assert_unique_feature_ids(reading)
