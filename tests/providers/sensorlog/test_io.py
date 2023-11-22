"""Tests for the :mod:`dispel.providers.sensorlog.io` module."""

import numpy as np
import pandas as pd
import pytest
from numpy.ma.testutils import assert_array_equal

from dispel.data.core import Device, Evaluation, Reading
from dispel.data.devices import IOSPlatform
from dispel.data.raw import (
    RawDataSet,
    RawDataSetDefinition,
    RawDataSetSource,
    RawDataValueDefinition,
)
from dispel.io.raw import raw_data_set_definition_to_columns
from dispel.providers.sensorlog import PROVIDER_ID
from dispel.providers.sensorlog.data import (
    DATA_SET_DEFINITIONS_DICT,
    SensorLogSensorType,
)
from dispel.providers.sensorlog.io import (
    extract_raw_data_set,
    get_device,
    get_evaluation,
    get_sensor_log_reading,
    read_sensor_log,
    read_sensor_log_as_data_frame,
)
from tests.providers import resource_path

EXAMPLE_PATH = resource_path(PROVIDER_ID, "example.json")


def test_read_sensor_log_as_data_frame():
    """Test reading SensorLog json files."""
    res = read_sensor_log_as_data_frame(EXAMPLE_PATH)
    assert isinstance(res, pd.DataFrame)
    assert res.shape == (3, 70)
    assert "accelerometerAccelerationX" in res.columns


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
    return read_sensor_log_as_data_frame(EXAMPLE_PATH)


def test_extract_raw_data_set_no_index():
    """Test if no index is specified in raw data set definition."""
    definition = RawDataSetDefinition(
        "example",
        RawDataSetSource("some vendor"),
        [
            RawDataValueDefinition("a", "Name of a"),
            RawDataValueDefinition("b", "Name of b"),
        ],
    )
    data = pd.DataFrame(dict(a=[0, 1, 2], b=["a", "b", "c"]))

    res = extract_raw_data_set(definition, data, DATA_SET_DEFINITIONS_DICT)

    assert res.data.index.name is None
    assert_array_equal(res.data.columns, ["a", "b"])


def test_extract_raw_data_set_index_conversion():
    """Test index conversion for raw data set extraction."""
    definition = RawDataSetDefinition(
        "example",
        RawDataSetSource("some vendor"),
        [
            RawDataValueDefinition("a", "Name of a", is_index=True),
            RawDataValueDefinition("b", "Name of b"),
        ],
    )
    data = pd.DataFrame(dict(a=[0, 1, 2], b=["a", "b", "c"]))

    res = extract_raw_data_set(definition, data, DATA_SET_DEFINITIONS_DICT)

    assert res.data.index.name == "a"
    assert res.data.columns == ["b"]


def test_extract_raw_data_set_multi_index_conversion():
    """Test multi index conversion for raw data set extraction."""
    definition = RawDataSetDefinition(
        "example",
        RawDataSetSource("some vendor"),
        [
            RawDataValueDefinition("a", "Name of a", is_index=True),
            RawDataValueDefinition("b", "Name of b", is_index=True),
            RawDataValueDefinition("c", "Name of c"),
        ],
    )
    data = pd.DataFrame(dict(a=[0, 1, 2], b=["a", "b", "c"], c=[0.1, 0.2, 0.3]))

    res = extract_raw_data_set(definition, data, DATA_SET_DEFINITIONS_DICT)

    assert res.data.index.names == ["a", "b"]
    assert res.data.columns == ["c"]


def test_extract_raw_data_set_type_conversion():
    """Test type conversion for raw data set extraction."""
    definition = RawDataSetDefinition(
        "example",
        RawDataSetSource("some vendor"),
        [
            RawDataValueDefinition("a", "a", data_type="datetime64"),
            RawDataValueDefinition("b", "b", data_type="datetime64[s]"),
            RawDataValueDefinition("c", "c", data_type="timedelta64[s]"),
            RawDataValueDefinition("d", "d", data_type="float"),
            RawDataValueDefinition("e", "e", data_type="int64"),
        ],
    )
    data = pd.DataFrame(
        dict(
            a=[
                "2020-05-05 09:11:29.596 +0200",
                "2020-05-05 09:11:29.598 +0200",
                "2020-05-05 09:11:29.604 +0200",
            ],
            b=[
                1588662689.000198,
                None,
                1588662689.599015,
            ],
            c=[29395.294288, 29395.314120, 29395.333951],
            d=[".1", ".2", ".3"],
            e=["1", "2", "3"],
        )
    )

    res = extract_raw_data_set(definition, data, DATA_SET_DEFINITIONS_DICT)

    expected_dtypes = [
        np.dtype("M8[ns]"),
        np.dtype("M8[ns]"),
        np.dtype("m8[ns]"),
        np.dtype("float64"),
        np.dtype("int64"),
    ]
    assert_array_equal(res.data.dtypes, expected_dtypes)


def test_extract_raw_data_set_value_conversion():
    """Test value conversion for raw data set extraction."""
    definition = RawDataSetDefinition(
        "example",
        RawDataSetSource("some vendor"),
        [
            RawDataValueDefinition("a", "a", data_type="datetime64"),
            RawDataValueDefinition("b", "b", data_type="datetime64[s]"),
            RawDataValueDefinition("c", "c", data_type="timedelta64[s]"),
            RawDataValueDefinition("d", "d", data_type="float"),
            RawDataValueDefinition("e", "e", data_type="int64"),
        ],
    )
    data = pd.DataFrame(
        dict(
            a=[
                "2020-06-23 11:12:01.123 +0200",
                "2020-06-23 03:13:02.456 +0200",
                "2020-06-23 07:11:29.789 +0200",
            ],
            b=[
                1588662689.000198,
                None,
                1588662689.599015,
            ],
            c=[29395.294288, 29395.314120, 29395.333951],
            d=[".1", ".2", ".3"],
            e=["1", "2", "3"],
        )
    )

    # result from the tested function
    res = extract_raw_data_set(definition, data, DATA_SET_DEFINITIONS_DICT)

    # define the expected value for a, special case because it is a
    # datetime format without unit: 'datetime64'
    expected_value_a = pd.DataFrame(
        dict(
            a=[
                "2020-06-23 09:12:01.123",
                "2020-06-23 01:13:02.456",
                "2020-06-23 05:11:29.789",
            ]
        ),
        dtype="datetime64[ns]",
    )

    # define the expected value for b, c, d, e
    expected_value = pd.DataFrame(
        dict(
            b=[
                1588662689.000198,
                None,
                1588662689.599015,
            ],
            c=np.array(
                [29395294288000, 29395314120000, 29395333951000],
                dtype="timedelta64[ns]",
            ),
            d=np.array([0.1, 0.2, 0.3], dtype="float64"),
            e=np.array([1, 2, 3], dtype="int64"),
        )
    )

    # convert b with pandas.to_datetime
    expected_value["b"] = pd.to_datetime(expected_value["b"], unit="s")

    # join 'a' and 'bcde'
    expected_value = expected_value_a.join(expected_value)

    # check that left and right DataFrame are equal.
    pd.testing.assert_frame_equal(res.data, expected_value)


def test_extract_raw_data_set_empty_data_frame():
    """Test empty data frames raising error in extraction."""
    with pytest.raises(ValueError):
        extract_raw_data_set(
            SensorLogSensorType.ACCELEROMETER, pd.DataFrame(), DATA_SET_DEFINITIONS_DICT
        )


@pytest.mark.parametrize("sensor", SensorLogSensorType)
def test_extract_raw_data_set(example: pd.DataFrame, sensor: SensorLogSensorType):
    """Test extracting raw data sets."""
    res = extract_raw_data_set(sensor, example, DATA_SET_DEFINITIONS_DICT)

    assert isinstance(res, RawDataSet)

    # test definition
    assert isinstance(res.definition, RawDataSetDefinition)
    assert res.id == sensor

    # test data set
    assert isinstance(res.data, pd.DataFrame)
    assert len(res.data) == 3


def test_get_evaluation(example):
    """Test getting evaluation information."""
    res = get_evaluation(example)

    start = pd.Timestamp("2020-05-05 09:11:29.596 +0200")
    end = pd.Timestamp("2020-05-05 09:11:29.604 +0200")

    assert isinstance(res, Evaluation)

    # time information
    assert res.start == start
    assert res.end == end


def test_get_device(example):
    """Test getting device information."""
    res = get_device(example)

    assert isinstance(res, Device)
    assert res.uuid == "E142433C-C949-4965-97E0-475F3131BA6B"
    assert isinstance(res.platform, IOSPlatform)


def test_get_sensor_log_reading(example):
    """Test extracting readings from SensorLog data frame."""
    res = get_sensor_log_reading(example)

    assert isinstance(res, Reading)
    assert isinstance(res.evaluation, Evaluation)
    assert isinstance(res.device, Device)

    # data sets
    assert len(res.get_level().raw_data_sets) == 16


def test_get_sensor_log_reading_incomplete(example):
    """Test reading data sets not extracting all available sensors."""
    sub_cols = [
        "accelerometerAccelerationX",
        "accelerometerAccelerationY",
        "accelerometerAccelerationZ",
        "accelerometerTimestamp_sinceReboot",
        "deviceID",
        "identifierForVendor",
        "loggingTime",
    ]
    data = example[sub_cols]

    res = get_sensor_log_reading(data)

    assert len(res.get_level().raw_data_sets) == 1

    res_data_set = res.get_level().get_raw_data_set(SensorLogSensorType.ACCELEROMETER)
    assert len(res_data_set.data) == 3


def test_read_sensor_log():
    """Test reading SensorLog json files."""
    res = read_sensor_log(EXAMPLE_PATH)
    assert isinstance(res, Reading)
