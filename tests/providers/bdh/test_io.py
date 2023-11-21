"""Tests for the :mod:`dispel.io.io` module."""
from datetime import datetime
from json import loads
from typing import List

from dispel.data.collections import MeasureSet
from dispel.data.core import Reading, ReadingSchema
from dispel.data.epochs import Epoch
from dispel.data.raw import (
    RawDataSetDefinition,
    RawDataSetSource,
    RawDataValueDefinition,
)
from dispel.data.values import ValueDefinition
from dispel.providers.bdh.io import (
    parse_bdh_reading,
    parse_epoch,
    parse_measure_definition,
    parse_measures,
    parse_raw_data_set_definition,
    parse_raw_data_source,
    parse_raw_data_value_definition,
    parse_schema,
    read_bdh,
)
from dispel.providers.generic.tasks.cps.utils import (
    EXPECTED_DURATION_D2D,
    EXPECTED_DURATION_S2D,
    LEVEL_DURATION_DEF,
)
from tests.providers.bdh.conftest import EXAMPLE_PATH_DRAW


def check_reading(example, level_ids: List[str], schema_name: str):
    """Test :func:`dispel.providers.bdh.io.parse_bdh_reading`."""
    res = parse_bdh_reading(example)
    assert isinstance(res, Reading)
    assert len(res.level_ids) == len(level_ids)
    for i, id_ in enumerate(level_ids):
        assert res.level_ids[i].id == id_
    assert res.schema is not None
    assert res.schema.name == schema_name


def test_parse_schema():
    """Testing :func:`dispel.providers.bdh.io.parse_schema`."""
    example_json = """{
        "namespace": "konectom",
        "name": "cps-test",
        "version": "1.0"
    }
    """
    example = loads(example_json)
    res = parse_schema(example)

    assert isinstance(res, ReadingSchema)
    assert res.namespace == "konectom"
    assert res.name == "cps-test"
    assert res.version == "1.0"


def test_parse_epoch_without_interruptions():
    """Testing :func:`dispel.providers.bdh.io.core.parse_epoch`."""
    example_json = """{
        "begin_timestamp": 0,
        "end_timestamp": 1,
        "numberOfInterrupts": 0,
        "interruptions": []
    }
    """
    example = loads(example_json)
    res = parse_epoch(example)

    assert isinstance(res, Epoch)
    assert isinstance(res.start, datetime)
    assert isinstance(res.end, datetime)


def test_parse_measure_value_definition():
    """Testing :func:`dispel.providers.bdh.io.parse_measure_definition`."""
    example_json = """{
        "unit": "count",
        "description": "The number of all responses"
    }
    """
    example = loads(example_json)
    res = parse_measure_definition("f1", example)

    assert isinstance(res, ValueDefinition)
    assert res.id == "f1"
    assert res.name == "f1"
    assert res.unit == "count"
    assert res.description == "The number of all responses"


def test_parse_raw_data_value_definition():
    """Testing :func:`dispel.providers.bdh.io.parse_raw_data_value_definition`."""
    example_json = """{
        "unit": "Date and time (ISO-8601)",
        "description": "The time the user responded (press in)"
    }
    """
    example = loads(example_json)
    res = parse_raw_data_value_definition("v1", example)

    assert isinstance(res, RawDataValueDefinition)
    assert res.id == "v1"
    assert res.name == "v1"
    assert res.unit == "Date and time (ISO-8601)"
    assert res.description == "The time the user responded (press in)"


def test_parse_raw_data_source_with_chipset():
    """Testing :func:`dispel.providers.bdh.io.parse_raw_data_source`."""
    example_json = """{
        "manufacturer": "BDH",
        "chipset": "Invesense",
        "reference": "Foo"
    }
    """
    example = loads(example_json)
    res = parse_raw_data_source(example)

    assert isinstance(res, RawDataSetSource)
    assert res.manufacturer == "BDH"
    assert res.chipset == "Invesense"
    assert res.reference == "Foo"


def test_parse_raw_data_source_without_chipset():
    """Testing :func:`dispel.providers.bdh.io.parse_raw_data_source`."""
    example_json = """{
        "manufacturer": "BDH"
    }
    """
    example = loads(example_json)
    res = parse_raw_data_source(example)

    assert isinstance(res, RawDataSetSource)
    assert res.manufacturer == "BDH"
    assert res.chipset is None
    assert res.reference is None


def test_parse_raw_data_set_definition():
    """Testing :func:`dispel.providers.bdh.io.parse_raw_data_set_definition`."""
    example_json = """{
        "computed": false,
        "source": {
            "manufacturer": "",
            "chipset": "",
            "reference": ""
        },
        "columns": {
            "timestamp" : {
                "unit": "ms",
                "description": "Timestamp of the measurement since 1 July 1970"
            },
            "xAxis" : {
                "unit": "m/s^2",
                "description": "The x-axis of the accelerometer"
            }
        }
    }
    """  # noqa: E501
    example = loads(example_json)
    res = parse_raw_data_set_definition("s1", example)

    assert isinstance(res, RawDataSetDefinition)
    assert res.id == "s1"
    assert isinstance(res.source, RawDataSetSource)
    assert len(res.value_definitions) == 2
    assert not res.is_computed

    for definition in res.value_definitions:
        assert isinstance(definition, RawDataValueDefinition)


def test_parse_measures():
    """Testing :func:`dispel.providers.bdh.io.parse_measures`."""
    example_json = """{
        "measureA": 55,
        "measureB": "text"
    }
    """
    example = loads(example_json)
    definitions = [
        ValueDefinition("measureA", "Measure A"),
        ValueDefinition("measureB", "Measure B"),
    ]
    res = parse_measures(example, definitions)

    assert isinstance(res, MeasureSet)
    assert len(res) == 2


def test_parse_bdh_reading_drawing(example_json_draw):
    """Testing :func:`dispel.providers.bdh.io.parse_bdh_reading`."""
    res = parse_bdh_reading(example_json_draw)
    assert isinstance(res, Reading)
    assert len(res.level_ids) == 16
    assert res.level_ids[0].id == "infinity-left"
    first_level = res.get_level(res.level_ids[0])
    assert len(first_level.raw_data_sets) == 5
    assert res.schema.name == "drawing-activity"


def test_parse_bdh_reading_cps(example_json_cps):
    """Testing :func:`dispel.providers.bdh.io.parse_bdh_reading`."""
    res = parse_bdh_reading(example_json_cps)
    assert isinstance(res, Reading)
    assert len(res.level_ids) == 2
    assert res.level_ids[0].id == "symbol_to_digit"
    first_level = res.get_level(res.level_ids[0])
    assert len(first_level.raw_data_sets) == 6
    assert res.schema.name == "cps-activity"


def test_parse_example_reading_mood(example_json_mood):
    """Testing :func:`dispel.providers.bdh.io.parse_bdh_reading`."""
    check_reading(example_json_mood, ["mood", "physical_state"], "mood-activity")


def test_parse_bdh_reading_sbt(example_json_sbt_utt):
    """Testing :func:`dispel.providers.bdh.io.parse_bdh_reading`."""
    check_reading(example_json_sbt_utt, ["sbt", "utt"], "sbut-activity")


def test_parse_bdh_reading_six_min_walk(example_json_6mwt_uat):
    """Testing :func:`dispel.providers.bdh.io.parse_bdh_reading`."""
    check_reading(example_json_6mwt_uat, ["6mwt"], "6mw-activity")


def test_parse_example_reading_msis29(example_json_msis29):
    """Testing :func:`dispel.providers.bdh.io.parse_bdh_reading`."""
    res = parse_bdh_reading(example_json_msis29)
    assert isinstance(res, Reading)
    assert len(res.level_ids) == 30
    assert res.level_ids[0].id == "msis29"
    first_level = res.get_level(res.level_ids[0])
    assert len(first_level.raw_data_sets) == 1
    assert res.schema.name == "msis29-activity"


def test_parse_bdh_reading_pinch(example_json_pinch):
    """Testing :func:`dispel.providers.bdh.io.parse_bdh_reading`."""
    res = parse_bdh_reading(example_json_pinch)
    assert isinstance(res, Reading)
    assert len(res.level_ids) == 59
    assert res.level_ids[0].id == "left-large"
    assert res.schema.name == "pinch-activity"


def test_read_bdh():
    """Testing :func:`dispel.providers.bdh.io.read_bdh`."""
    res = read_bdh(EXAMPLE_PATH_DRAW)
    assert isinstance(res, Reading)


def test_parse_table_type_4(example_json_cps_table_type4):
    """Testing that parsing does not raise an error for table type 4."""
    _ = parse_bdh_reading(example_json_cps_table_type4)


def test_parse_radius_level_index(example_json_pinch_radius_level_index):
    """Testing we can parse pinch records with a radius_level in {1,2,3,4}."""
    _ = parse_bdh_reading(example_json_pinch_radius_level_index)


def test_parse_bdh_reading_ps(example_json_ps):
    """Testing :func:`dispel.io.io.parse_reading`."""
    check_reading(example_json_ps, ["right", "left"], "sp-activity")


def test_parse_bdh_reading_fingertap(example_json_ft):
    """Testing :func:`dispel.io.io.parse_reading`."""
    check_reading(example_json_ft, ["right", "left"], "fingertap-activity")


def test_parse_bdh_reading_alsfrs(example_json_alsfrs):
    """Testing :func:`dispel.providers.bdh.io.parse_bdh_reading` for alsfrs."""
    check_reading(example_json_alsfrs, ["alsfrs"], "alsfrs-activity")


def test_parse_bdh_reading_pdq39(example_json_pdq39):
    """Testing :func:`dispel.providers.bdh.io.parse_bdh_reading` for pdq39."""
    check_reading(example_json_pdq39, ["pdq39"], "pdq39-activity")


def test_expected_level_duration_from_context_cps(example_reading_cps):
    """Test if the expected duration for CPS test is correctly set during reading."""
    duration_d2d = example_reading_cps.get_level(
        "digit_to_digit"
    ).context.get_raw_value(LEVEL_DURATION_DEF)
    duration_s2d = example_reading_cps.get_level(
        "symbol_to_digit"
    ).context.get_raw_value(LEVEL_DURATION_DEF)

    assert duration_d2d == EXPECTED_DURATION_D2D
    assert duration_s2d == EXPECTED_DURATION_S2D
