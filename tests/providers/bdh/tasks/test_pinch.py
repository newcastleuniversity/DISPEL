"""Test cases for :mod:`dispel.providers.generic.tasks.pinch`."""

import pytest

from dispel.providers.bdh.io import read_bdh
from dispel.providers.generic.tasks.pinch import process_pinch
from tests.processing.helper import assert_level_values, read_results
from tests.providers.bdh.conftest import EXAMPLE_PATH_PINCH_BUG as BUG_PATH_PINCH_BDH
from tests.providers.bdh.conftest import RESULTS_PATH_PINCH_UAT


def test_pinch_process_bdh_bug():
    """Pressure present in header but not in body."""
    with pytest.raises(AssertionError):
        process_pinch(read_bdh(BUG_PATH_PINCH_BDH))


@pytest.mark.parametrize("level_id, expected", read_results(RESULTS_PATH_PINCH_UAT))
def test_pinch_process_bdh_uat(example_reading_processed_pinch_uat, level_id, expected):
    """Test measures on a BDH formatted reading using the latest format."""
    assert_level_values(example_reading_processed_pinch_uat, level_id, expected)


def test_pinch_process_bdh_overwrite(example_reading_processed_pinch_overwrite):
    """Test measures on a BDH formatted reading using the latest format."""
    _ = example_reading_processed_pinch_overwrite
