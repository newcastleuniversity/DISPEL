"""Tests for the :mod:`dispel.io.ads` module."""
import json

import pandas as pd
import pytest

from dispel.data.core import Evaluation, Session
from dispel.data.devices import Device, IOSPlatform, IOSScreen
from dispel.data.raw import (
    RawDataSetDefinition,
    RawDataSetSource,
    RawDataValueDefinition,
)
from dispel.io.core import convert_data_frame_type
from dispel.providers.ads import PROVIDER_ID
from dispel.providers.ads.data import ADSModalities, ADSReading
from dispel.providers.ads.io import (
    create_ads_raw_data_set_definition,
    create_ads_value_definitions,
    get_ads_raw_data_set,
    get_ads_raw_data_set_ids,
    parse_ads_raw_json,
    parse_device,
    parse_evaluation,
    parse_levels,
    parse_screen,
    parse_session,
    read_ads,
)
from dispel.providers.generic.tasks.cps.utils import (
    EXPECTED_DURATION_D2D,
    EXPECTED_DURATION_S2D,
    LEVEL_DURATION_DEF,
)
from tests.providers import resource_path
from tests.providers.ads.conftest import EXAMPLE_PATH_PINCH

CPS_EXAMPLE_PATH = resource_path(PROVIDER_ID, "CPS/example.json")


def test_get_ads_raw_data_set_ids(example_json_cps):
    """Testing getting the raw data set ids."""
    sensor_example = example_json_cps["mobileEvaluationTest"]["levels"][0]["sensors"]
    raw_data_set_ids = set(get_ads_raw_data_set_ids(sensor_example))
    actual_raw_data_set_ids = {"screen", "userInput"}

    assert raw_data_set_ids == actual_raw_data_set_ids


def test_parse_screen(example_json_cps):
    """Testing getting the screen information."""
    screen = parse_screen(
        example_json_cps["mobileDevice"]["platform"],
        example_json_cps["mobileDevice"]["screen"],
    )

    assert isinstance(screen, IOSScreen)
    assert screen.width_pixels == 1125
    assert screen.width_dp_pt == 375
    assert screen.density_dpi == 463
    assert screen.scale_factor == 3


def test_parse_device(example_json_cps):
    """Testing getting the device information."""
    device = parse_device(example_json_cps["mobileDevice"])

    assert isinstance(device, Device)
    assert isinstance(device.platform, IOSPlatform)
    assert device.model == "iPhone XS"
    assert device.model_code == "iPhone11,2"
    assert device.os_version == "13.4.1"
    assert device.app_version_number == "0.2.8"
    assert device.app_build_number == "27"


def test_parse_session(example_json_cps):
    """Testing getting the evaluation information."""
    session = parse_session(example_json_cps["mobileSession"])

    assert isinstance(session, Session)
    assert session.uuid == "de68dcf5-40ca-446c-b60b-41b71246910b"
    assert session.id == "konectomDaily"
    assert session.evaluation_codes == [
        "mood",
        "cps",
        "pinch",
        "drawing",
        "gripForce",
        "sbtUtt",
    ]
    assert not session.is_incomplete

    assert session.start == pd.Timestamp("2020-05-15T18:10:31.968+02:00").tz_convert(
        None
    )
    assert session.end == pd.Timestamp("2020-05-15T18:17:29.387+02:00").tz_convert(None)


def test_parse_evaluation(example_json_cps):
    """Testing getting the evaluation information."""
    evaluation = parse_evaluation(
        example_json_cps["uuid"],
        example_json_cps["mobileEvaluationTest"],
        example_json_cps["userId"],
    )

    assert isinstance(evaluation, Evaluation)
    assert evaluation.id == "cps"
    assert evaluation.finished
    assert evaluation.exit_reason == "nominal"

    assert evaluation.start == pd.Timestamp(1589559040890, unit="ms")
    assert evaluation.end == pd.Timestamp(1589559178007, unit="ms")


def test_create_ads_value_definitions(example_json_pinch):
    """Testing creation of ADS value definitions."""
    ads_value_definitions = {}
    levels = example_json_pinch["mobileEvaluationTest"]["levels"]
    raw_data_set_ids = ["accelerometer", "gyroscope", "screen"]
    for raw_data_set_id in raw_data_set_ids:
        value_definitions = []
        for sensor in levels[0]["sensors"]:
            if sensor["name"] == raw_data_set_id:
                value_definitions = create_ads_value_definitions(
                    sensor["measurements"][0]["values"], raw_data_set_id
                )
        ads_value_definitions[raw_data_set_id] = value_definitions

    for raw_data_set_id in raw_data_set_ids:
        for value_definition in ads_value_definitions[raw_data_set_id]:
            assert isinstance(value_definition, RawDataValueDefinition)


@pytest.fixture
def ads_value_definitions(example_json_pinch):
    """Get a fixture to the value definitions."""
    ads_value_definitions = {}
    levels = example_json_pinch["mobileEvaluationTest"]["levels"]
    raw_data_set_ids = ["accelerometer", "gyroscope", "screen"]
    for raw_data_set_id in raw_data_set_ids:
        value_definitions = []
        for sensor in levels[0]["sensors"]:
            if sensor["name"] == raw_data_set_id:
                value_definitions = create_ads_value_definitions(
                    sensor["measurements"][0]["values"], raw_data_set_id
                )
        ads_value_definitions[raw_data_set_id] = value_definitions
    return ads_value_definitions


def _test_create_ads_value_definitions_generic(actual_variables, value_definitions):
    variable_ids = {definition.id for definition in value_definitions}
    actual_variable_ids = set(actual_variables.keys())

    assert variable_ids == actual_variable_ids
    for definition in value_definitions:
        assert definition.unit == actual_variables[definition.id]["unit"]
        assert definition.data_type == actual_variables[definition.id]["dataType"]


def test_create_ads_value_definitions_screen(ads_value_definitions):
    """Testing creation of ADS screen value definitions."""
    value_definitions = ads_value_definitions["screen"]
    actual_variables = {
        "touchAction": {"unit": None, "dataType": "U"},
        "xPosition": {"unit": "point", "dataType": "float32"},
        "yPosition": {"unit": "point", "dataType": "float32"},
        "tsTouch": {"unit": "ms", "dataType": "datetime64[ms]"},
        "pressure": {"unit": None, "dataType": "float32"},
        "maxPressure": {"unit": None, "dataType": "float32"},
        "ledToSuccess": {"unit": None, "dataType": "bool"},
        "touchPathId": {"unit": None, "dataType": "int16"},
        "isValidPinch": {"unit": None, "dataType": "bool"},
        "majorRadius": {"unit": None, "dataType": "float32"},
        "majorRadiusTolerance": {"unit": None, "dataType": "float32"},
    }

    _test_create_ads_value_definitions_generic(actual_variables, value_definitions)


def test_create_ads_value_definitions_accelerometer(ads_value_definitions):
    """Testing creation of ADS accelerometer value definitions."""
    value_definitions = ads_value_definitions["accelerometer"]
    acc_characteristics = {"unit": "G", "dataType": "float32"}
    actual_variables = {
        "userAccelerationX": acc_characteristics,
        "userAccelerationY": acc_characteristics,
        "userAccelerationZ": acc_characteristics,
        "ts": {"unit": "ms", "dataType": "datetime64[ms]"},
        "gravityX": acc_characteristics,
        "gravityY": acc_characteristics,
        "gravityZ": acc_characteristics,
    }

    _test_create_ads_value_definitions_generic(actual_variables, value_definitions)


def test_create_ads_value_definitions_gyroscope(ads_value_definitions):
    """Testing creation of ADS gyroscope value definitions."""
    value_definitions = ads_value_definitions["gyroscope"]
    gyro_characteristics = {"unit": "rad/s", "dataType": "float32"}
    actual_variables = {
        "x": gyro_characteristics,
        "y": gyro_characteristics,
        "z": gyro_characteristics,
        "ts": {"unit": "ms", "dataType": "datetime64[ms]"},
    }

    _test_create_ads_value_definitions_generic(actual_variables, value_definitions)


def test_create_ads_value_definitions_contexts(example_json_pinch):
    """Testing creation of ADS context value definitions."""
    levels = example_json_pinch["mobileEvaluationTest"]["levels"]
    context_value_definitions = create_ads_value_definitions(
        levels[0]["contexts"], "contexts"
    )
    actual_variables = {
        "targetRadius": {"unit": "point", "dataType": "float32"},
        "xTargetBall": {"unit": "point", "dataType": "float32"},
        "yTargetBall": {"unit": "point", "dataType": "float32"},
        "usedHand": {"unit": None, "dataType": "U"},
    }

    _test_create_ads_value_definitions_generic(
        actual_variables, context_value_definitions
    )


def test_create_ads_raw_data_set_definition(ads_value_definitions):
    """Testing creation of ADS raw data set definitions."""
    is_computed_ids = {"screen": True, "accelerometer": False, "gyroscope": False}
    for raw_data_set_id, is_computed in is_computed_ids.items():
        definition = create_ads_raw_data_set_definition(
            ads_value_definitions[raw_data_set_id], raw_data_set_id
        )
        assert isinstance(definition, RawDataSetDefinition)
        assert definition.id == raw_data_set_id
        assert isinstance(definition.source, RawDataSetSource)
        assert definition.source.manufacturer == "ADS"
        assert definition.is_computed == is_computed


def test_get_ads_raw_data_set_screen(example_json_pinch):
    """Testing getting of ADS raw data set."""
    levels = example_json_pinch["mobileEvaluationTest"]["levels"][0]
    raw_data_set_id = "screen"
    data = convert_data_frame_type(get_ads_raw_data_set(levels, raw_data_set_id))
    assert isinstance(data, pd.DataFrame)
    assert len(data.index) == 26
    assert data.shape[1] == 11
    assert data["xPosition"].dtype == "float32"
    assert data["pressure"].dtype == "float32"
    assert data["majorRadius"].dtype == "float32"
    assert data["isValidPinch"].dtype == "bool"
    assert pd.api.types.is_datetime64_ns_dtype(data["tsTouch"])


def test_get_ads_raw_data_set_accelerometer(example_json_pinch):
    """Testing getting of ADS raw data set."""
    levels = example_json_pinch["mobileEvaluationTest"]["levels"][0]
    raw_data_set_id = "accelerometer"
    data = convert_data_frame_type(get_ads_raw_data_set(levels, raw_data_set_id))
    assert isinstance(data, pd.DataFrame)
    assert len(data.index) == 15
    assert data.shape[1] == 7
    assert data["userAccelerationX"].dtype == "float32"
    assert data["gravityY"].dtype == "float32"
    assert pd.api.types.is_datetime64_ns_dtype(data["ts"])


def test_get_ads_raw_data_set_gyroscope(example_json_pinch):
    """Testing getting of ADS raw data set."""
    levels = example_json_pinch["mobileEvaluationTest"]["levels"][0]
    raw_data_set_id = "gyroscope"
    data = convert_data_frame_type(get_ads_raw_data_set(levels, raw_data_set_id))
    assert isinstance(data, pd.DataFrame)
    assert len(data.index) == 15
    assert data.shape[1] == 4
    assert data["x"].dtype == "float32"
    assert data["y"].dtype == "float32"
    assert data["z"].dtype == "float32"
    assert pd.api.types.is_datetime64_ns_dtype(data["ts"])


def test_parse_ads_raw_json(example_json_cps):
    """Testing ADSReading functionality."""
    res = parse_ads_raw_json(example_json_cps)
    assert isinstance(res, ADSReading)
    assert pd.Timestamp(1589559181120, unit="ms") == res.date

    evaluation = parse_evaluation(
        example_json_cps["uuid"],
        example_json_cps["mobileEvaluationTest"],
        example_json_cps["userId"],
    )

    # define levels
    levels_data = example_json_cps["mobileEvaluationTest"]["levels"]

    # parse levels
    levels = parse_levels(levels_data, evaluation.id, ADSModalities(""))

    assert str(levels[0].id) == "symbol_to_digit"
    assert str(levels[1].id) == "digit_to_digit"

    assert levels[0].start != levels[1].start
    assert levels[0].end != levels[1].end


def test_read_ads():
    """Testing reading ADS JSON file."""
    ads_raw_json = read_ads(CPS_EXAMPLE_PATH)
    assert isinstance(ads_raw_json, ADSReading)


def test_read_another_ads():
    """Testing another ADS reading with empty measurements."""
    res = read_ads(EXAMPLE_PATH_PINCH)
    assert isinstance(res, ADSReading)


def test_read_ads_new_format(example_reading_pinch_new_format):
    """Testing ADS new format reading."""
    res = example_reading_pinch_new_format
    assert isinstance(res, ADSReading)


@pytest.mark.xfail
def test_export_features_from_data():
    """Testing JSON export of features from data."""
    with open(CPS_EXAMPLE_PATH, encoding="utf-8") as fs:
        data = json.load(fs)
    list_export = export_features(data)  # noqa: F821

    required_fields = {
        "feature_id",
        "feature_name",
        "feature_unit",
        "feature_type",
        "evaluation_uuid",
        "feature_value",
        "flag_ids",
        "flag_reasons",
    }

    assert isinstance(list_export, list)
    assert len(list_export) != 0
    assert required_fields <= set(list_export[0])


@pytest.mark.xfail
def test_export_features_from_path():
    """Testing JSON export of features from path."""
    # FIXME: fix test case
    list_export = export_features(CPS_EXAMPLE_PATH)  # noqa: F821

    assert isinstance(list_export, list)
    assert len(list_export) != 0


@pytest.mark.parametrize(
    "fixture_str",
    [
        "example_reading_cps",
        "example_reading_cps_new_format",
    ],
)
def test_expected_level_duration_from_context_cps(fixture_str, request):
    """Test correct setting of expected CPS level durations."""
    example_reading = request.getfixturevalue(fixture_str)
    duration_d2d = example_reading.get_level("digit_to_digit").context.get_raw_value(
        LEVEL_DURATION_DEF
    )
    duration_s2d = example_reading.get_level("symbol_to_digit").context.get_raw_value(
        LEVEL_DURATION_DEF
    )

    assert duration_d2d == EXPECTED_DURATION_D2D
    assert duration_s2d == EXPECTED_DURATION_S2D
