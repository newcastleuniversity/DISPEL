"""Tests for the :mod:`dispel.providers.digital_artefacts.io` module."""
import typing
from json import load
from typing import Dict

import pytest

from dispel.data.core import Evaluation, Reading
from dispel.data.devices import IOSPlatform
from dispel.providers.digital_artefacts.io import read_da
from tests.providers import resource_path

EXAMPLE_PATH_DA_FT = resource_path("digital_artefacts", "ft/example-v2.json")


@pytest.fixture
def example_da_ft() -> Dict:
    """Create a fixture of a drawing example."""
    with open(EXAMPLE_PATH_DA_FT, encoding="utf-8") as fs:
        return load(fs)


@pytest.fixture()
def da_reading() -> Reading:
    """Create a fixture for digital artefact reading."""
    return read_da(EXAMPLE_PATH_DA_FT)


def test_parse_da(da_reading: Reading):
    """Testing :func:`dispel.io.digital_artefact.read_da`."""
    assert isinstance(da_reading, Reading)
    assert da_reading.level_ids[0].id == "domhand"
    dominant_hand_level = da_reading.get_level(da_reading.level_ids[0])
    assert len(dominant_hand_level.raw_data_sets) == 1


def test_evaluation_parsing(da_reading: Reading):
    """Test the parsing of the evaluation."""
    evaluation = da_reading.evaluation
    assert isinstance(evaluation, Evaluation)
    assert str(evaluation.uuid) == "1084896"
    assert evaluation.user_id == "00000000000000000000000"
    assert evaluation.id == "finger-tapping"


@typing.no_type_check
def test_device_parsing(da_reading: Reading):
    """Test the parsing of the device."""
    device = da_reading.device
    assert device.model == "iPhone11,8"
    assert isinstance(da_reading.device.platform, IOSPlatform)
    assert device.app_version_number == "com.brainbaseline.BrainBaseline-BiogenWatchALS"
    assert device.app_build_number == 14.2
    assert device.screen.width_dp_pt == 896
    assert device.screen.height_dp_pt == 414


def test_level_data(da_reading: Reading):
    """Test the parsing of the level data."""
    level_data = (
        da_reading.get_level("nondomhand")
        .get_raw_data_set("enriched_tap_events_ts")
        .data
    )
    assert level_data.shape == (25, 3)
    tap_events_type = level_data["location"].unique()
    assert "none" in tap_events_type
