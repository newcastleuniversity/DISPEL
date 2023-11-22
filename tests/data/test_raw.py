"""Tests for :mod:`dispel.data.raw`."""
from copy import deepcopy

import pandas as pd
import pytest

from dispel.data.flags import Flag
from dispel.data.raw import (
    RawDataSet,
    RawDataSetDefinition,
    RawDataSetSource,
    RawDataValueDefinition,
)
from dispel.io.raw import generate_raw_data_set


def test_equality_raw_data_set_definitions():
    """Test equality operator for raw data set definitions."""
    source = RawDataSetSource("manufacturer")
    definition1 = RawDataValueDefinition("id1", "definition 1")
    definition2 = RawDataValueDefinition("id2", "definition 2", "s")

    rdd1 = RawDataSetDefinition("id", source, [definition1, definition2])
    rdd2 = RawDataSetDefinition("id", source, [definition2, definition1])
    rdd3 = RawDataSetDefinition("id1", source, [definition1, definition2])

    assert rdd1 == rdd2
    assert rdd1 != rdd3


def test_raw_data_set_definition_duplicate_definition():
    """Test that a data set definition can't have two identical definitions."""
    source = RawDataSetSource("manufacturer")
    definition1 = RawDataValueDefinition("id1", "definition 1")
    definition2 = RawDataValueDefinition("id1", "definition 2")

    with pytest.raises(ValueError):
        RawDataSetDefinition("ds1", source, [definition1, definition2])


def test_raw_data_set_mismatching_data_definition():
    """Test that data matches definition for raw data sets."""
    source = RawDataSetSource("manufacturer")
    value_definition1 = RawDataValueDefinition("col1", "Column 1")
    value_definition2 = RawDataValueDefinition("col2", "Column 2")
    rdd1 = RawDataSetDefinition("id", source, [value_definition1, value_definition2])

    data1 = pd.DataFrame({"col1": [0, 1, 2], "col3": [0, 1, 2]})

    # Has missing definition
    with pytest.raises(ValueError):
        RawDataSet(rdd1, data1)

    value_definition3 = RawDataValueDefinition("col3", "Column 3")
    value_definition4 = RawDataValueDefinition("col4", "Column 4")
    rdd2 = RawDataSetDefinition(
        "id",
        source,
        [value_definition1, value_definition2, value_definition3, value_definition4],
    )

    # Has too many definition
    with pytest.raises(ValueError):
        RawDataSet(rdd2, data1)


def test_raw_data_set_definition_value_definitions():
    """Test that value definitions match for raw data sets."""
    source = RawDataSetSource("manufacturer")
    definition1 = RawDataValueDefinition("col1", "Column 1")
    definition2 = RawDataValueDefinition("col2", "Column 2")
    rdd1 = RawDataSetDefinition("id", source, [definition1, definition2])

    assert set(rdd1.value_definitions) == {definition1, definition2}


@pytest.fixture(scope="module")
def raw_data_set():
    """Create a fixture for a raw data set."""
    return generate_raw_data_set("data-set-id", columns=["a"])


def test_raw_data_set_set_flag(raw_data_set):
    """Test set flag in raw data set."""
    data_set = deepcopy(raw_data_set)
    flag = Flag("cps-technical-deviation-ta", "reason")

    assert data_set.is_valid
    data_set.add_flag(flag)
    assert data_set.flag_count == 1


def test_raw_data_set_precision(raw_data_set):
    """Test raw data set precision."""
    data = pd.DataFrame({"col1": [1.11, 4.44, 5.55], "col2": [1.11, 4.44, 5.55]})

    source = RawDataSetSource("manufacturer")
    value_definition1 = RawDataValueDefinition("col1", "Column 1")
    value_definition2 = RawDataValueDefinition("col2", "Column 2")
    # raw data set definition
    rdd = RawDataSetDefinition("id", source, [value_definition1, value_definition2])
    # raw dataset
    rds = RawDataSet(rdd, data)

    assert not hasattr(rds, "raw_data")
    assert all(rds.data["col1"] == rds.data["col2"])

    # update first value definition with precision set to 1 digit
    value_definition1 = RawDataValueDefinition("col1", "Column 1", precision=1)
    rdd = RawDataSetDefinition("id", source, [value_definition1, value_definition2])
    rds = RawDataSet(rdd, data)

    assert hasattr(rds, "raw_data")
    assert not (any(rds.data["col1"] == rds.data["col2"]))
