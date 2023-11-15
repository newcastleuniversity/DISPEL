"""Test cases for :mod:`dispel.providers.ads.tasks.cps`."""

import pytest

from dispel.providers.generic.tasks.cps.utils import CPS_FLAG_NEDP
from tests.processing.helper import assert_level_values, read_results
from tests.providers.ads.conftest import RESULTS_PATH_CPS, RESULTS_PATH_CPS_NEW_FORMAT


@pytest.mark.parametrize("level,expected", read_results(RESULTS_PATH_CPS))
def test_cps_process(example_reading_processed_cps, level, expected):
    """Unit test to ensure the CPS features are well computed."""
    assert_level_values(example_reading_processed_cps, level, expected)


@pytest.mark.parametrize("level,expected", read_results(RESULTS_PATH_CPS_NEW_FORMAT))
def test_ads_cps_new_format(example_reading_processed_cps_new_format, level, expected):
    """Test processing ADS new format for CPS."""
    assert_level_values(example_reading_processed_cps_new_format, level, expected)


def tests_cps_flag(example_reading_processed_cps_new_format):
    """Test the cps flag."""
    lvl = example_reading_processed_cps_new_format.get_level("digit_to_digit")
    inv = lvl.feature_set.get("cps-dtd_rand_dig1-rt-mean").get_flags()
    assert len(inv) == 1
    assert inv[0] == CPS_FLAG_NEDP
