"""Test cases for :mod:`dispel.providers.bdh.tasks.cps`."""

import pytest

from tests.processing.helper import assert_level_values, read_results
from tests.providers.bdh.conftest import (
    RESULTS_PATH_CPS,
    RESULTS_PATH_CPS_BUG,
    RESULTS_PATH_CPS_TABLE_TYPE4,
)


@pytest.mark.parametrize("level,expected", read_results(RESULTS_PATH_CPS))
def test_cps_process_bdh(example_reading_processed_cps, level, expected):
    """Unit test to ensure the CPS features are well computed."""
    assert_level_values(example_reading_processed_cps, level, expected)


@pytest.mark.parametrize("level,expected", read_results(RESULTS_PATH_CPS_TABLE_TYPE4))
def test_cps_table_4_process_bdh(
    example_reading_processed_cps_table_type4, level, expected
):
    """Unit test to ensure the CPS features are well computed."""
    assert_level_values(example_reading_processed_cps_table_type4, level, expected)


@pytest.mark.parametrize("level,expected", read_results(RESULTS_PATH_CPS_BUG))
@pytest.mark.xfail
def test_cps_process_bdh_bug(example_reading_processed_cps_bug, level, expected):
    """Test CPS record with empty dtd level."""
    assert_level_values(example_reading_processed_cps_bug, level, expected)
