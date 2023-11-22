"""Tests for :mod:`dispel.processing.transform`."""
from copy import deepcopy

import pandas as pd
import pytest

from dispel.data.core import Reading
from dispel.data.levels import Level
from dispel.data.raw import RawDataSet, RawDataValueDefinition
from dispel.processing import DataTrace, process
from dispel.processing.core import CoreProcessingStepGroup, InvalidDataError
from dispel.processing.level import LevelIdFilter
from dispel.processing.transform import (
    ConcatenateLevels,
    SuffixBasedNewDataSetIdMixin,
    TransformStep,
)


class _TransformStepPrototype(TransformStep):
    data_set_ids = "data-set-1"
    new_data_set_id = "sum-data-set-1"
    definitions = [RawDataValueDefinition("sum", "sum of values")]
    level_filter = LevelIdFilter("level_1")


TRANSFORM_STEP = _TransformStepPrototype(
    transform_function=lambda data: data.sum(axis=1)
)

reading_example_RAW_DATA_VALUE_DEFINITIONS = [
    RawDataValueDefinition("a", "A"),
    RawDataValueDefinition("b", "B"),
]


@pytest.mark.parametrize(
    "transform",
    [
        lambda data: data.sum(axis=1).to_numpy(),
        lambda data: data.sum(axis=1),
        lambda data: data.sum(axis=1).rename("sum").to_frame(),
    ],
    ids=["nparray", "series", "dataframe"],
)
def test_transform_step_single_data_set(reading_example, transform):
    """Test transforming single data sets with wrapping."""
    reading = deepcopy(reading_example)
    step = _TransformStepPrototype(transform_function=transform)
    output = next(step.process(reading))

    assert isinstance(output.level, Level)
    assert str(output.level.id) == "level_1"
    assert isinstance(output.result, RawDataSet)
    assert output.result.id == "sum-data-set-1"
    assert output.result.data.shape == (3, 1)
    assert output.result.data.iloc[0]["sum"] == 2


def test_transform_step_multiple_data_sets(reading_example):
    """Test transforming multiple data sets at once."""
    reading = deepcopy(reading_example)
    step = TransformStep(
        ["data-set-1", "data-set-2"],
        lambda a, b: a / b,
        "data-set-3",
        reading_example_RAW_DATA_VALUE_DEFINITIONS,
        level_filter="level_1",
    )

    output = next(step.process(reading))

    assert isinstance(output.level, Level)
    assert str(output.level.id) == "level_1"
    assert isinstance(output.result, RawDataSet)
    assert output.result.id == "data-set-3"
    assert output.result.data.shape == (3, 2)
    assert output.result.data.mean().mean() == 1 / 2

    # Tests level use in TransformStep
    def _division_with_level(a, b, level=None):
        res = a / b
        if level is not None:
            res += 10
        return res

    step_args_use = TransformStep(
        ["data-set-1", "data-set-2"],
        _division_with_level,
        "data-set-3",
        reading_example_RAW_DATA_VALUE_DEFINITIONS,
        level_filter="level_1",
    )

    output = next(step_args_use.process(reading))

    assert isinstance(output.level, Level)
    assert str(output.level.id) == "level_1"
    assert isinstance(output.result, RawDataSet)
    assert output.result.id == "data-set-3"
    assert output.result.data.shape == (3, 2)
    assert output.result.data.mean().mean() == 1 / 2 + 10


def test_transform_step_missing_data_set(reading_example):
    """Test that processing stops if data set is missing."""
    reading = deepcopy(reading_example)
    step = TransformStep(
        "missing-data-set",
        lambda x: x,
        "never-create",
        reading_example_RAW_DATA_VALUE_DEFINITIONS,
        level_filter="level_1",
    )

    result = next(step.process(reading, task_name="task"))
    assert isinstance(result.error, InvalidDataError)


def test_process_transform_step(reading_example):
    """Test processing a transform step that returns a RawDataSet."""
    reading = deepcopy(reading_example)
    res = process(reading, TRANSFORM_STEP).get_reading()
    assert isinstance(res, Reading)

    data_set = res.get_level("level_1").get_raw_data_set("sum-data-set-1")
    assert isinstance(data_set, RawDataSet)


def test_processing_result_integration_transformation(reading_example):
    """Test integration of transformation step results through processing."""
    reading = deepcopy(reading_example)
    data_trace = process(reading, TRANSFORM_STEP)

    assert isinstance(data_trace, DataTrace)
    assert repr(data_trace) == (
        "<DataTrace of <Reading: 2 levels (0 flags)>: "
        "(8 entities, 1 processing step)>"
    )

    level = reading.get_level("level_1")
    data_sets = level.raw_data_sets
    assert len(data_sets) == 3

    expected_data_set_ids = {"data-set-1", "data-set-2", "sum-data-set-1"}
    assert set(map(lambda d: d.id, data_sets)) == expected_data_set_ids

    data_set = level.get_raw_data_set("sum-data-set-1")
    parents = set(data_trace.parents(data_set))
    expected_parents = {level.get_raw_data_set("data-set-1")}

    assert parents == expected_parents
    assert data_trace.is_leaf(data_set)


CONCATENATE_STEP = ConcatenateLevels(
    new_level_id="level_1_and_2",
    data_set_id="data-set-1",
    level_filter=["level_1", "level_2"],
)


def test_concatenate_levels(reading_example):
    """Tests ConcatenateLevels transform step."""
    # concatenating level using Concatenate Levels
    reading = deepcopy(reading_example)
    process(reading, CONCATENATE_STEP)

    # get levels
    concatenated_level = reading.get_level("level_1_and_2")
    level_1 = reading.get_level("level_1")
    level_2 = reading.get_level("level_2")

    # get raw_data
    concatenated_data = concatenated_level.get_raw_data_set("data-set-1").data
    level_1_data = level_1.get_raw_data_set("data-set-1").data
    level_2_data = level_2.get_raw_data_set("data-set-1").data

    manually_cat_data = pd.concat([level_1_data, level_2_data])

    assert len(level_1_data) + len(level_2_data) == len(manually_cat_data)
    assert len(manually_cat_data) == len(concatenated_data)

    # compare the raw_data
    for column in level_1_data.columns:
        assert (manually_cat_data[column] == concatenated_data[column]).all()

    # compare context
    level_1_context = level_1.context
    level_2_context = level_2.context
    concatenated_level_context = concatenated_level.context

    for key in level_1_context:
        key_concat = key.id + "_0"
        assert (
            concatenated_level_context[key_concat].value == level_1_context[key].value
        )

    for key in level_2_context:
        key_concat = key.id + "_1"
        assert (
            concatenated_level_context[key_concat].value == level_2_context[key].value
        )

    # compare EffectiveTimeFrame
    res = concatenated_level_context["level_0"].value
    assert isinstance(res, Level)
    assert level_1.start == res.start
    assert level_1.end == res.end

    res = concatenated_level_context["level_1"].value
    assert isinstance(res, Level)
    assert level_2.start == res.start
    assert level_2.end == res.end


def test_concatenate_levels_multiple_data_sets(reading_example):
    """Tests ConcatenateLevels transform step with 2 data_set_id."""
    reading = deepcopy(reading_example)
    # concatenating level using Concatenate Levels
    process(
        reading,
        ConcatenateLevels(
            new_level_id="level_1_and_2",
            data_set_id=["data-set-1", "data-set-2"],
            level_filter=["level_1", "level_2"],
        ),
    )

    # get levels
    concatenated_level = reading.get_level("level_1_and_2")
    level_1 = reading.get_level("level_1")
    level_2 = reading.get_level("level_2")

    # get raw_data_sets
    concatenated_data_1 = concatenated_level.get_raw_data_set("data-set-1").data
    level_1_data_1 = level_1.get_raw_data_set("data-set-1").data
    level_2_data_1 = level_2.get_raw_data_set("data-set-1").data
    concatenated_data_2 = concatenated_level.get_raw_data_set("data-set-2").data
    level_1_data_2 = level_1.get_raw_data_set("data-set-2").data
    level_2_data_2 = level_2.get_raw_data_set("data-set-2").data

    manually_cat_data_1 = pd.concat([level_1_data_1, level_2_data_1])
    manually_cat_data_2 = pd.concat([level_1_data_2, level_2_data_2])

    assert len(level_1_data_1) + len(level_2_data_1) == len(manually_cat_data_1)
    assert len(manually_cat_data_1) == len(concatenated_data_1)
    assert len(level_1_data_2) + len(level_2_data_2) == len(manually_cat_data_2)
    assert len(manually_cat_data_2) == len(concatenated_data_2)


def test_transform_step_set_previous_used_in_data_set_ids():
    """Test chaining of transform steps."""
    step1 = TransformStep(new_data_set_id="example")
    step2 = TransformStep()

    chained_steps = step1 & step2

    assert isinstance(chained_steps, CoreProcessingStepGroup)
    assert step2.get_data_set_ids() == ["example"]


def test_transform_step_suffix_based_new_data_set_id():
    """Test that the new data set id for suffix mixin is correctly created."""

    class _SuffixTransformStep(SuffixBasedNewDataSetIdMixin, TransformStep):
        suffix = "suffix"
        data_set_ids = ["one", "two"]

    step = _SuffixTransformStep()

    assert step.get_new_data_set_id() == "one_two_suffix"


def test_transform_step_suffix_based_new_data_set_id_chained():
    """Test that chaining works as expected for suffix-based mixin."""
    step1 = TransformStep(new_data_set_id="example")

    class _SuffixTransformStep(SuffixBasedNewDataSetIdMixin, TransformStep):
        suffix = "suffix"

    step2 = _SuffixTransformStep()

    chained_steps = step1 & step2

    assert isinstance(chained_steps, CoreProcessingStepGroup)
    assert step2.get_new_data_set_id() == "example_suffix"
