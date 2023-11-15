"""Test cases for :mod:`dispel.providers`."""

import pytest

from dispel.data.features import FeatureSet
from dispel.providers import auto_process
from dispel.providers.ads.data import ADSReading
from dispel.providers.ads.io import read_ads
from tests.providers.ads.conftest import EXAMPLE_PATH_CPS


@pytest.fixture
def example_reading():
    """Fixture for an example reading."""
    return read_ads(EXAMPLE_PATH_CPS)


def test_auto_process(example_reading):
    """Test :func:`dispel.providers.auto_process`."""
    res = auto_process(example_reading).get_reading()
    assert isinstance(res, ADSReading)

    level_id = res.level_ids[0]
    fs = res.get_level(level_id).feature_set
    assert isinstance(fs, FeatureSet)
    assert len(fs) > 0
