"""Tests for :mod:`dispel.tasks.ft.py` with DA format."""
from copy import deepcopy

import pytest

from dispel.providers.digital_artefacts.data import DigitalArtefactsReading
from dispel.providers.digital_artefacts.io import read_da
from dispel.providers.digital_artefacts.tasks.ft import process_ft
from tests.processing.helper import assert_level_values
from tests.providers import resource_path

EXAMPLE_PATH_FT = resource_path("digital_artefacts", "ft/example-v2.json")


@pytest.fixture(scope="session")
def ft_reading_da():
    """Create a bdh finger tapping reading fixture."""
    return read_da(EXAMPLE_PATH_FT)


@pytest.fixture(scope="session")
def processed_ft(ft_reading_da):
    """Create a bdh finger tapping processed reading fixture."""
    da_reading = process_ft(deepcopy(ft_reading_da)).get_reading()
    assert isinstance(da_reading, DigitalArtefactsReading)
    return da_reading


@pytest.mark.parametrize(
    "level,expected",
    [
        (
            "domhand",
            {
                "ft-domhand_leftzone-valtap": 16,
                "ft-domhand_rightzone-valtap": 16,
                "ft-domhand-valtap": 32,
                "ft-domhand-tap_inter-mean": 0.5511714285714285,
                "ft-domhand-tap_inter-std": 0.1261846186474139,
                "ft-domhand-tap_inter-median": 0.55,
                "ft-domhand-tap_inter-min": 0.004,
                "ft-domhand-tap_inter-max": 0.875,
                "ft-domhand-tap_inter-iqr": 0.10750000000000004,
                "ft-domhand-valid_tap_inter-mean": 0.5843870967741935,
                "ft-domhand-valid_tap_inter-std": 0.2375750656696889,
                "ft-domhand-valid_tap_inter-median": 0.55,
                "ft-domhand-valid_tap_inter-min": 0.004,
                "ft-domhand-valid_tap_inter-max": 1.649,
                "ft-domhand-valid_tap_inter-iqr": 0.10750000000000004,
                "ft-domhand-total_tap": 36,
                "ft-domhand-double_tap_percentage": 2.62,
            },
        ),
        (
            "nondomhand",
            {
                "ft-nondomhand_leftzone-valtap": 10,
                "ft-nondomhand_rightzone-valtap": 11,
                "ft-nondomhand-valtap": 21,
                "ft-nondomhand-tap_inter-mean": 0.763125,
                "ft-nondomhand-tap_inter-std": 0.24105696828036677,
                "ft-nondomhand-tap_inter-median": 0.77,
                "ft-nondomhand-tap_inter-min": 0.07500000000000001,
                "ft-nondomhand-tap_inter-max": 1.258,
                "ft-nondomhand-tap_inter-iqr": 0.23475000000000001,
                "ft-nondomhand-valid_tap_inter-mean": 0.9157500000000001,
                "ft-nondomhand-valid_tap_inter-std": 0.5369731812375932,
                "ft-nondomhand-valid_tap_inter-median": 0.77,
                "ft-nondomhand-valid_tap_inter-min": 0.524,
                "ft-nondomhand-valid_tap_inter-max": 2.5050000000000003,
                "ft-nondomhand-valid_tap_inter-iqr": 0.22499999999999998,
                "ft-nondomhand-total_tap": 25,
                "ft-nondomhand-double_tap_percentage": 0.0,
            },
        ),
    ],
)
def test_da_ft_features(processed_ft, level, expected):
    """Test feature values for Digital Artefact - Finger Tapping Assessment."""
    assert_level_values(processed_ft, level, expected)
