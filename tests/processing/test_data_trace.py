"""Test :mod:`dispel.data.data_trace`."""
from copy import deepcopy

import pytest
from multimethod import DispatchError

from dispel.data.flags import Flag
from dispel.data.levels import Level
from dispel.data.measures import MeasureValue
from dispel.data.values import ValueDefinition
from dispel.io.raw import generate_raw_data_set
from dispel.processing import ProcessingStep
from dispel.processing.data_trace import (
    DataTrace,
    GraphNoCycle,
    MissingSourceNode,
    NodeHasMultipleParents,
)
from tests.processing.test_extract import EXTRACT_STEP
from tests.processing.test_transform import CONCATENATE_STEP, TRANSFORM_STEP


def test_data_trace_creation(reading_example):
    """Test data trace graph creation from a reading."""
    dtg = DataTrace.from_reading(reading_example)
    assert isinstance(dtg, DataTrace)

    assert dtg.entity_count == 7
    assert dtg.leaf_count == 4
    assert len(dtg.children(dtg.root)) == 2
    assert repr(dtg) == (
        "<DataTrace of <Reading: 2 levels (0 flags)>: "
        "(7 entities, 0 processing steps)>"
    )


@pytest.fixture(scope="module")
def data_trace_example(reading_example):
    """Create a fixture for a data trace example."""
    reading = deepcopy(reading_example)

    level3 = Level(id_="level_3", start="now", end="now")
    reading.set(level3)

    measure2 = MeasureValue(ValueDefinition("feat2", "feat"), 5)
    reading.set(measure2)

    measure3 = MeasureValue(ValueDefinition("feat3", "feat"), 5)
    reading.set(measure3, level3)

    dtg = DataTrace.from_reading(reading)

    level1 = reading.get_level("level_1")
    raw_data_set1 = level1.get_raw_data_set("data-set-1")
    measure1 = MeasureValue(ValueDefinition("feat1", "feat"), 3)

    reading.set(measure1, level1)
    dtg.add_trace(raw_data_set1, measure1, EXTRACT_STEP)

    raw_data_set3 = generate_raw_data_set("data-set-3", ["a", "b"])

    reading.set(raw_data_set3, level1)
    dtg.add_trace(raw_data_set1, raw_data_set3, TRANSFORM_STEP)

    dtg.add_trace(raw_data_set3, level3, CONCATENATE_STEP)

    raw_data_set2 = level1.get_raw_data_set("data-set-2")
    dtg.add_trace(raw_data_set2, level3, CONCATENATE_STEP)
    return dtg


def assert_node_entity_presence(data_trace, entity):
    """Assert the presence of a node entity inside the data trace."""
    assert data_trace.has_node(entity)


def assert_node_entity_absence(data_trace, entity):
    """Assert the absence of a node entity inside the data trace."""
    assert not data_trace.has_node(entity)


def assert_node_entities(data_trace, *entities):
    """Assert the presence and absence of multiple node entities in DTG."""
    for entity in entities:
        assert_node_entity_presence(data_trace, entity)
        assert_node_entity_absence(data_trace, deepcopy(entity))


def test_data_trace_get_nodes(data_trace_example):
    """Test existent and non-existent node retrieval from data trace graph."""
    reading = data_trace_example.get_reading()
    level = reading.get_level("level_2")
    raw_data_set = level.get_raw_data_set("data-set-1")

    assert_node_entities(data_trace_example, reading, level, raw_data_set)


def test_data_trace_add_illegal_nodes_wrong_types(data_trace_example):
    """Test adding illegal nodes with wrong types to the data trace graph."""
    reading = data_trace_example.get_reading()
    raw_data_set = reading.get_level("level_1").get_raw_data_set("data-set-1")

    with pytest.raises(DispatchError):
        data_trace_example.add_trace("hello", 2)

    with pytest.raises(DispatchError):
        data_trace_example.add_trace(reading, 2)

    with pytest.raises(DispatchError):
        data_trace_example.add_trace(reading, raw_data_set)


def test_data_trace_add_illegal_source_node(data_trace_example):
    """Test adding illegal source node to the data trace graph."""
    dtg = deepcopy(data_trace_example)

    source = generate_raw_data_set("foo", ["a"])
    outcome = generate_raw_data_set("bar", ["b"])

    with pytest.raises(MissingSourceNode):
        dtg.add_trace(source, outcome, TRANSFORM_STEP)


def test_data_trace_add_illegal_outcome_node(data_trace_example):
    """Test adding illegal outcome node to the data trace graph."""
    dtg = deepcopy(data_trace_example)
    reading = dtg.get_reading()
    level1 = reading.get_level("level_1")
    level2 = reading.get_level("level_2")

    outcome = level1.get_raw_data_set("data-set-2")

    with pytest.raises(NodeHasMultipleParents):
        dtg.add_trace(level2, outcome)


def test_data_trace_cyclic_graph(data_trace_example):
    """Test attempting creation of a cyclic graph."""
    dtg = deepcopy(data_trace_example)
    reading = dtg.get_reading()
    level = reading.get_level("level_1")
    raw_data_set = level.get_raw_data_set("data-set-3")

    with pytest.raises(GraphNoCycle):
        dtg.add_trace(raw_data_set, level, CONCATENATE_STEP)


def test_data_trace_consistency_check(reading_example):
    """Test consistency check of the data trace graph."""
    # pylint: disable=protected-access
    dtg = DataTrace.from_reading(reading_example)
    warning_match = "No measure has been created from the following entity"

    with pytest.warns(UserWarning, match=warning_match) as record:
        dtg.check_data_set_usage()
    assert len(record) == 4

    # Fix consistency of data trace
    for id_ in ("level_1", "level_2"):
        level = dtg.get_reading().get_level(id_)
        raw_data_set1 = level.get_raw_data_set("data-set-1")
        raw_data_set2 = level.get_raw_data_set("data-set-2")
        measure1 = MeasureValue(ValueDefinition(f"{id_}_feat1", "feat"), 3)
        measure2 = MeasureValue(ValueDefinition(f"{id_}_feat2", "feat"), 3)
        dtg.add_trace(raw_data_set1, measure1, EXTRACT_STEP)
        dtg.add_trace(raw_data_set2, measure2, EXTRACT_STEP)

    with pytest.warns(None, match=warning_match) as record:
        dtg.check_data_set_usage()
    assert len(record) == 0

    dtg.add_trace(measure1, Level(id_="id", start=0, end=1), ProcessingStep())
    with pytest.raises(ValueError):
        dtg.check_data_set_usage()


def test_data_trace_get_measure_flags(reading_example):
    """Test data trace measure flag retrieval."""
    reading = deepcopy(reading_example)
    measure = MeasureValue(ValueDefinition("feat2", "feat"), 5)
    measure_set = reading.get_level("level_1").measure_set
    measure_set.set(measure)
    measure_set.get("feat2").add_flag(Flag("pinch-technical-deviation-ta", "reason 1"))
    dtg = DataTrace.from_reading(reading)
    assert dtg.get_measure_flags() == [Flag("pinch-technical-deviation-ta", "reason 1")]


def test_data_trace_add_measure_level_outcome_node(data_trace_example):
    """Test adding trace between a measure towards a level."""
    dtg = deepcopy(data_trace_example)
    reading = dtg.get_reading()
    level_1 = reading.get_level("level_1")
    level_2 = reading.get_level("level_2")
    feat_1 = level_1.measure_set.get("feat1")
    step = ProcessingStep()
    dtg.add_trace(feat_1, level_2, step)
