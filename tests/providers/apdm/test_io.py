"""Tests for the :mod:`dispel.providers.apdm.io` module."""

import pandas as pd
import pytest

from dispel.data.core import Device, Evaluation, Reading
from dispel.data.raw import (
    RawDataSet,
    RawDataSetDefinition,
    RawDataSetSource,
    RawDataValueDefinition,
)
from dispel.io.raw import raw_data_set_definition_to_columns
from dispel.providers.apdm import PROVIDER_ID
from dispel.providers.apdm.data import APDMPlatform, ApdmSensorType
from dispel.providers.apdm.io import (
    DATA_SET_DEFINITIONS_DICT,
    extract_raw_data_set,
    get_apdm_reading,
    get_device,
    get_evaluation,
    read_apdm,
    read_apdm_as_data_frame,
)
from tests.providers import resource_path

EXAMPLE_2MWT_PATH = resource_path(PROVIDER_ID, "2mwt/example.h5")


def test_read_apdm_as_data_frame():
    """Test reading APDM h5 files."""
    res = read_apdm_as_data_frame(EXAMPLE_2MWT_PATH)
    assert isinstance(res, pd.DataFrame)
    assert res.shape == (15746, 14)
    assert "Lumbar_acc_x" in res.columns


def test_raw_data_set_definition_to_columns():
    """Test converting raw data set definition into columns."""
    definition = RawDataSetDefinition(
        "id",
        RawDataSetSource("manufacturer"),
        [
            RawDataValueDefinition("a", "Name of a"),
            RawDataValueDefinition("b", "Name of b"),
            RawDataValueDefinition("c", "Name of c"),
        ],
    )
    res = raw_data_set_definition_to_columns(definition)
    assert res == list("abc")


@pytest.fixture
def example():
    """Get a fixture to the example data set."""
    return read_apdm_as_data_frame(EXAMPLE_2MWT_PATH)


def test_extract_raw_data_set_empty_data_frame():
    """Test empty data frames raising error in extraction."""
    with pytest.raises(ValueError):
        extract_raw_data_set(
            ApdmSensorType.ACCELEROMETER, pd.DataFrame(), DATA_SET_DEFINITIONS_DICT
        )


@pytest.mark.parametrize("sensor", ApdmSensorType)
def test_extract_raw_data_set(example: pd.DataFrame, sensor: ApdmSensorType):
    """Test extracting raw data sets."""
    res = extract_raw_data_set(sensor, example, DATA_SET_DEFINITIONS_DICT)

    assert isinstance(res, RawDataSet)

    # test definition
    assert isinstance(res.definition, RawDataSetDefinition)
    assert res.id == sensor

    # test data set
    assert isinstance(res.data, pd.DataFrame)
    assert len(res.data) == 15746


def test_get_evaluation(example):
    """Test getting evaluation information."""
    res = get_evaluation(example)

    start = pd.Timestamp("2021-02-17 16:41:02.915052")
    end = pd.Timestamp("2021-02-17 16:43:05.914992")

    assert isinstance(res, Evaluation)

    # time information
    assert res.start == start
    assert res.end == end


def test_get_device(example):
    """Test getting device information."""
    res = get_device(example)

    assert isinstance(res, Device)
    assert isinstance(res.platform, APDMPlatform)


def test_get_apdm_reading(example):
    """Test extracting readings from APDM data frame."""
    res = get_apdm_reading(example)

    assert isinstance(res, Reading)
    assert isinstance(res.evaluation, Evaluation)
    assert isinstance(res.device, Device)

    # data sets
    assert len(res.get_level().raw_data_sets) == 6


def test_get_apdm_reading_incomplete(example):
    """Test reading data sets not extracting all available sensors."""
    sub_cols = [
        "Lumbar_acc_x",
        "Lumbar_acc_y",
        "Lumbar_acc_z",
        "ts",
    ]
    data = example[sub_cols]

    res = get_apdm_reading(data)

    assert len(res.get_level().raw_data_sets) == 2

    res_data_set = res.get_level().get_raw_data_set(ApdmSensorType.ACCELEROMETER)
    assert len(res_data_set.data) == 15746


def test_read_apdm():
    """Test reading APDM h5 files."""
    res = read_apdm(EXAMPLE_2MWT_PATH)
    assert isinstance(res, Reading)
