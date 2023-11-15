"""Test :mod:`dispel.processing.extract`."""
from copy import deepcopy
from typing import Any

import numpy as np
import pytest

from dispel.data.collections import FeatureSet
from dispel.data.core import Evaluation, Reading
from dispel.data.epochs import EpochDefinition
from dispel.data.features import FeatureValue, FeatureValueDefinition
from dispel.data.flags import Flag, FlagSeverity, FlagType, WrappedResult
from dispel.data.levels import Level
from dispel.data.validators import RangeValidator
from dispel.data.values import ValueDefinition, ValueDefinitionPrototype
from dispel.processing import process
from dispel.processing.core import (
    CoreProcessingStepGroup,
    InvalidDataError,
    ProcessingResult,
)
from dispel.processing.data_set import transformation
from dispel.processing.data_trace import DataTrace
from dispel.processing.extract import (
    BASIC_AGGREGATIONS,
    AggregateFeatures,
    AggregateRawDataSetColumn,
    ExtractMultipleStep,
    ExtractStep,
    FeatureFlagStep,
)
from dispel.processing.flags import flag
from dispel.processing.transform import TransformStep

EXTRACT_STEP = ExtractStep(
    "data-set-1",
    len,
    ValueDefinition("len", "number of records"),
    level_filter="level_1",
)


def _assert_extract_results(step, reading):
    output = next(step.process(reading))

    assert isinstance(output.level, Level)
    assert str(output.level.id) == "level_1"

    value = output.result
    assert isinstance(value, FeatureValue)
    assert value.value == 3

    assert isinstance(value.definition, ValueDefinition)
    assert value.definition.name == "number of records"


def test_extract_step_single_data_set(reading_example):
    """Test extracting features from data sets."""
    reading = deepcopy(reading_example)
    _assert_extract_results(EXTRACT_STEP, reading)


@pytest.mark.parametrize(
    "func", [len, lambda x: len(x)]  # pylint: disable=unnecessary-lambda
)
def test_extract_step_class_defined(reading_example, func):
    """Test that construction with transform functions works correctly."""
    reading = deepcopy(reading_example)

    class _ExtractStep(ExtractStep):
        data_set_ids = ["data-set-1"]
        transform_function = func
        definition = ValueDefinition("len", "number of records")

    step = _ExtractStep(level_filter="level_1")

    _assert_extract_results(step, reading)


def test_extract_step_transform_function_decorator(reading_example):
    """Test transformation decorator."""
    reading = deepcopy(reading_example)

    class _ExtractStep(ExtractStep):
        data_set_ids = "data-set-1"
        definition = ValueDefinition("len", "number of records")

        @transformation
        def _len(self, data):
            return len(data)

    step = _ExtractStep(level_filter="level_1")

    _assert_extract_results(step, reading)


def test_extract_step_transform_function_decorator_w_args(reading_example):
    """Test arguments being passed for transformation decorator."""
    reading = deepcopy(reading_example)

    class _ExtractStep(ExtractStep):
        data_set_ids = "data-set-1"
        definition = ValueDefinitionPrototype()

        @transformation(id_="len", name="number of records")
        def _len(self, data):
            return len(data)

    step = _ExtractStep(level_filter="level_1")

    _assert_extract_results(step, reading)


def test_extract_step_feature_definition_prototype(reading_example):
    """Test extracting features using prototype definition."""
    reading = deepcopy(reading_example)
    step = ExtractStep(
        "data-set-1",
        len,
        ValueDefinitionPrototype(id_="{name}", unit="count"),
        level_filter="level_1",
    )

    # fail when not all arguments for definition creation are provided
    with pytest.raises(ValueError) as exception:
        next(step.process(reading))

    assert str(exception.value) == "Missing placeholder: 'name'"

    # test populating placeholders and arguments
    output = next(step.process(reading, name="example"))

    assert isinstance(output.level, Level)
    assert str(output.level.id) == "level_1"
    assert isinstance(output.result, FeatureValue)
    assert output.result.id == "example"
    assert output.result.definition.name == "example"
    assert output.result.definition.unit == "count"


def test_extract_multiple_step(reading_example):
    """Test extracting multiple features based on prototype."""
    reading = deepcopy(reading_example)
    step = ExtractMultipleStep(
        "data-set-1",
        [
            {"func": lambda x: x["a"].mean(), "method": "mean"},
            {"func": lambda x: x["a"].std(), "method": "stddev"},
        ],
        ValueDefinitionPrototype(id_="f-{method}", name="{method}"),
        level_filter="level_1",
    )

    res = list(step.process(reading))

    assert len(res) == 2
    mean_output_1, std_output_2 = res

    assert isinstance(mean_output_1, ProcessingResult)
    assert isinstance(std_output_2, ProcessingResult)

    assert isinstance(mean_output_1.level, Level)
    assert str(mean_output_1.level.id) == "level_1"
    assert isinstance(std_output_2.level, Level)
    assert str(std_output_2.level.id) == "level_1"

    assert isinstance(mean_output_1.result, FeatureValue)
    assert mean_output_1.result.value == 1
    assert mean_output_1.result.id == "f-mean"
    assert mean_output_1.result.definition.name == "mean"

    assert isinstance(std_output_2.result, FeatureValue)
    assert std_output_2.result.value == 0
    assert std_output_2.result.id == "f-stddev"
    assert std_output_2.result.definition.name == "stddev"

    # test level and reading use in ExtractMultipleStep
    def _mean_with_level(x, level):
        res = x["a"].mean()
        if level is not None:
            res += 10
        return res

    def _std_with_reading(x, reading):
        res = x["a"].std()
        if reading is not None:
            res += 5
        return res

    step_args_use = ExtractMultipleStep(
        "data-set-1",
        [
            {"func": _mean_with_level, "method": "mean"},
            {"func": _std_with_reading, "method": "stddev"},
        ],
        ValueDefinitionPrototype(id_="f-{method}", name="{method}"),
        level_filter="level_1",
    )

    res = list(step_args_use.process(reading))

    assert len(res) == 2
    mean_output_1, std_output_2 = res

    assert isinstance(mean_output_1, ProcessingResult)
    assert isinstance(std_output_2, ProcessingResult)

    assert isinstance(mean_output_1.level, Level)
    assert str(mean_output_1.level.id) == "level_1"
    assert isinstance(std_output_2.level, Level)
    assert str(std_output_2.level.id) == "level_1"

    assert isinstance(mean_output_1.result, FeatureValue)
    assert mean_output_1.result.value == 11
    assert mean_output_1.result.id == "f-mean"
    assert mean_output_1.result.definition.name == "mean"

    assert isinstance(std_output_2.result, FeatureValue)
    assert std_output_2.result.value == 5
    assert std_output_2.result.id == "f-stddev"
    assert std_output_2.result.definition.name == "stddev"


def test_process_extract_step(reading_example):
    """Test processing an extract step that returns a FeatureValue."""
    reading = deepcopy(reading_example)
    dtg = process(reading, EXTRACT_STEP)
    assert isinstance(dtg, DataTrace)

    res = dtg.get_reading()
    assert isinstance(res.get_level("level_1").feature_set, FeatureSet)

    feature = res.get_level("level_1").feature_set.get("len")
    assert feature.value == 3


def test_aggregate_raw_data_set_column(reading_example):
    """Test AggregateRawDataSetColumn step."""
    reading = deepcopy(reading_example)
    aggregate_step = AggregateRawDataSetColumn(
        "data-set-1",
        "a",
        BASIC_AGGREGATIONS,
        ValueDefinitionPrototype(
            id_="f-id", name="name", validator=RangeValidator(1, 2)
        ),
        level_filter="level_1",
    )

    res = list(aggregate_step.process(reading))

    assert len(res) == 2
    mean_output_1, std_output_2 = res

    assert isinstance(mean_output_1.level, Level)
    assert mean_output_1.level.id == "level_1"
    assert isinstance(std_output_2.level, Level)
    assert str(std_output_2.level.id) == "level_1"

    assert isinstance(mean_output_1.result, FeatureValue)
    assert mean_output_1.result.value == 1
    assert mean_output_1.result.id == "f-id"
    assert mean_output_1.result.definition.name == "name"

    assert isinstance(std_output_2.result, FeatureValue)
    assert std_output_2.result.value == 0
    assert std_output_2.result.id == "f-id"
    assert std_output_2.result.definition.name == "name"


def test_aggregate_raw_data_set_column_yield_if_nan(reading_example):
    """Test that yielding of NaN values is controlled correctly."""
    reading = deepcopy(reading_example)
    step = AggregateRawDataSetColumn(
        "data-set-1",
        "a",
        [("kurtosis", "kurtosis")],
        ValueDefinitionPrototype(id_="f-id", name="name"),
        level_filter="level_1",
    )

    process(reading, step)
    fs = reading.get_merged_feature_set()

    assert fs.empty


@pytest.fixture
def reading_example_agg():
    """Get fixture for a reading with features to aggregate."""
    reading_fs = FeatureSet()
    reading_fs.set(FeatureValue(ValueDefinition("id1", "name1"), 1))
    reading_fs.set(FeatureValue(ValueDefinition("id2", "name2"), 2))

    level_fs = FeatureSet()
    level_fs.set(FeatureValue(ValueDefinition("id3", "name3"), 3))
    level_fs.set(FeatureValue(ValueDefinition("id4", "name4"), 4))
    level_fs.set(FeatureValue(ValueDefinition("id6", "name6"), np.NaN))

    return Reading(
        Evaluation(
            uuid="test", start="now", end="now", definition=EpochDefinition(id_="test")
        ),
        feature_set=reading_fs,
        levels=[Level(id_="level", start="now", end="now", feature_set=level_fs)],
    )


def test_aggregate_features_constructor(reading_example_agg):
    """Test aggregating features via constructor."""
    reading = deepcopy(reading_example_agg)
    definition = FeatureValueDefinition("example", "name")
    step = AggregateFeatures(definition, ["id1", "id3", "id5"])
    process(reading, step)
    fs = reading.get_merged_feature_set()
    assert "example-name" in fs
    assert fs.get_raw_value("example-name") == 2


def test_aggregate_features_specify_agg_method(reading_example_agg):
    """Test aggregating features with a non-default agg-method."""
    reading = deepcopy(reading_example_agg)
    definition = FeatureValueDefinition("example", "name")
    step = AggregateFeatures(definition, ["id2", "id4"], sum)
    process(reading, step)
    fs = reading.get_merged_feature_set()
    assert "example-name" in fs
    assert fs.get_raw_value("example-name") == 6


def test_aggregate_features_class_variables(reading_example_agg):
    """Test aggregating features with a non-default agg-method."""
    reading = deepcopy(reading_example_agg)

    class _Test(AggregateFeatures):
        aggregation_method = max
        definition = FeatureValueDefinition("example", "another")
        feature_ids = ["id1", "id2", "id3"]

    step = _Test()
    process(reading, step)
    fs = reading.get_merged_feature_set()
    assert "example-another" in fs
    assert fs.get_raw_value("example-another") == 3


def test_aggregate_features_missing_feature(reading_example_agg):
    """Test aggregating features fail with missing feature."""
    with pytest.raises(InvalidDataError):
        step = AggregateFeatures(
            FeatureValueDefinition("example", "name"),
            ["id1", "id5"],
            fail_if_missing=True,
        )
        process(reading_example_agg, step)

    with pytest.raises(InvalidDataError):

        class _Test(AggregateFeatures):
            definition = FeatureValueDefinition("example", "name")
            feature_ids = ["id1", "id5"]
            fail_if_missing = True

        process(reading_example_agg, _Test())


@pytest.mark.parametrize("yield_if_nan", [True, False])
def test_aggregate_features_yield_if_nan(reading_example_agg, yield_if_nan):
    """Test that yielding of NaN values is controlled correctly."""
    reading = deepcopy(reading_example_agg)
    step = AggregateFeatures(
        definition=FeatureValueDefinition("example", "nan"),
        feature_ids=["id6"],
        yield_if_nan=yield_if_nan,
    )

    process(reading, step)
    fs = reading.get_merged_feature_set()

    assert ("example-nan" in fs) == yield_if_nan


def test_processing_result_integration_extraction(reading_example):
    """Test integration of extraction step results through processing."""
    reading = deepcopy(reading_example)
    data_trace = process(reading, EXTRACT_STEP)

    assert isinstance(data_trace, DataTrace)
    assert repr(data_trace) == (
        "<DataTrace of <Reading: 2 levels (0 flags)>: "
        "(8 entities, 1 processing step)>"
    )

    level = reading.get_level("level_1")

    feature_set = level.feature_set
    assert len(feature_set) == 1

    feature = feature_set.get("len")
    parents = set(data_trace.parents(feature))
    expected_parents = {level.get_raw_data_set("data-set-1")}

    assert parents == expected_parents
    assert data_trace.is_leaf(feature)


def test_extract_step_set_previous_used_in_data_set_ids():
    """Test chaining of transform and extract steps."""
    step1 = TransformStep(new_data_set_id="example")
    step2 = ExtractStep()

    chained_steps = step1 & step2

    assert isinstance(chained_steps, CoreProcessingStepGroup)
    assert step2.get_data_set_ids() == ["example"]


def test_feature_flag(reading_example):
    """Test feature flag."""
    reading = deepcopy(reading_example)
    reading.set(FeatureValue(ValueDefinition("feat", "feat"), 6))

    class _FeatureValueFlagger(FeatureFlagStep):
        task_name = "task"
        flag_name = "name"
        flag_type = FlagType.TECHNICAL
        flag_severity = FlagSeverity.INVALIDATION
        reason = "reason{number}"
        feature_ids = "feat"

        @flag(number=1)
        def _inv1(self, feature_value: Any) -> bool:
            return feature_value > 5

        @flag(number=2)
        def _inv2(self, feature_value: Any) -> bool:
            return feature_value <= 5

        @flag(number=3)
        def _inv3(self, feature_value: Any, reading: Reading) -> bool:
            assert isinstance(reading, Reading)
            return feature_value <= 5

    process(reading, _FeatureValueFlagger())

    feature_value = reading.get_merged_feature_set().get("feat")
    expected = [
        Flag("task-technical-invalidation-name", "reason2"),
        Flag("task-technical-invalidation-name", "reason3"),
    ]
    assert feature_value.get_flags() == expected


def test_wr_unfolding(reading_example):
    """Test transformation decorator."""
    reading = deepcopy(reading_example)

    class _ExtractStep(ExtractStep):
        data_set_ids = "data-set-1"
        definition = ValueDefinition("len", "number of records")

        @transformation
        def _test_transform(self, _data):
            res = WrappedResult(32.0)
            res.add_flag(Flag("cps-behavioral-deviation-test", reason="test"))
            return res

    step = _ExtractStep(level_filter="level_1")

    output = next(step.process(reading))
    feature_value = output.result
    assert isinstance(feature_value, FeatureValue)
    assert not feature_value.is_valid
    assert feature_value.value == 32.0
