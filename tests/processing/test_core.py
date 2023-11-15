"""Tests for :mod:`dispel.processing.core`."""
from copy import deepcopy
from functools import partial
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from dispel.data.core import Evaluation, Reading
from dispel.data.epochs import EpochDefinition
from dispel.data.flags import Flag, FlagSeverity, FlagType
from dispel.data.levels import Context, Level
from dispel.data.raw import RawDataSet
from dispel.data.validators import GREATER_THAN_ZERO, ValidationException
from dispel.data.values import Value, ValueDefinition
from dispel.io.raw import generate_raw_data_set_definition
from dispel.processing import process
from dispel.processing.core import (
    CoreProcessingStepGroup,
    FlagError,
    FlagReadingStep,
    InvalidDataError,
    Parameter,
    ProcessingControlResult,
    ProcessingResult,
    ProcessingStep,
    ProcessResultType,
)
from dispel.processing.data_trace import DataTrace
from dispel.processing.flags import flag
from tests.processing.test_transform import CONCATENATE_STEP


@pytest.fixture(scope="package")
def reading_example():
    """Provide an example reading for testing purposes."""
    columns = ["a", "b"]
    data1 = pd.DataFrame(np.ones((3, 2)), columns=columns)
    data_set1, data_set2 = map(
        partial(generate_raw_data_set_definition, columns=columns),
        ("data-set-1", "data-set-2"),
    )
    reading = Reading(
        # FIXME: correct constructor
        evaluation=Evaluation(
            uuid="id", start="now", end="now", definition=EpochDefinition(id_="example")
        ),
        levels=[
            Level(
                id_="level_1",
                start="now",
                end="now",
                context=Context(
                    [
                        Value(ValueDefinition("name_1", "name_1", "n/a"), 1),
                        Value(ValueDefinition("name_2", "name_2", "n/a"), 2),
                    ]
                ),
                raw_data_sets=[
                    RawDataSet(data_set1, data1),
                    RawDataSet(data_set2, data1 * 2),
                ],
            ),
            Level(
                id_="level_2",
                start="now",
                end="now",
                context=Context(
                    [
                        Value(ValueDefinition("name_3", "name_3", "n/a"), 3),
                        Value(ValueDefinition("name_4", "name_4", "n/a"), 4),
                    ]
                ),
                raw_data_sets=[
                    RawDataSet(data_set1, data1 * 3),
                    RawDataSet(data_set2, data1 * 4),
                ],
            ),
        ],
    )
    return reading


def test_process_process_step_no_effect(reading_example):
    """Test calling of process function."""
    reading = deepcopy(reading_example)
    step = ProcessingStep()
    step.process_reading = MagicMock()

    res = process(reading, step, foo="bar")

    assert isinstance(res, DataTrace)
    step.process_reading.assert_called_with(reading, foo="bar")


@pytest.mark.parametrize("value", [1, 0.5, "string"])
def test_process_process_step_invalid_value(reading_example, value):
    """Test failing on invalid values returned by processing step."""
    example = deepcopy(reading_example)

    class _FaultyStep(ProcessingStep):
        def process_reading(self, reading: Reading, **kwargs) -> ProcessResultType:
            yield ProcessingResult(value)  # type: ignore

    with pytest.raises(TypeError):
        process(example, _FaultyStep())


def test_processing_control_result_from_exception():
    """Test :class:`dispel.processing.core.LevelProcessingControlResult`."""
    try:
        assert 3 > 4, "message"
    except AssertionError as exception_message:
        exception = exception_message

    res = ProcessingControlResult.from_assertion_error(step=None, error=exception)
    assert isinstance(res.error, InvalidDataError)


def test_processing_result_integration_concatenation(reading_example):
    """Test integration of concatenation step results through processing."""
    reading = deepcopy(reading_example)
    data_trace = process(reading, CONCATENATE_STEP)

    assert isinstance(data_trace, DataTrace)
    assert repr(data_trace) == (
        "<DataTrace of <Reading: 3 levels (0 flags)>: "
        "(9 entities, 1 processing step)>"
    )

    level = reading.get_level("level_1_and_2")
    data_sets = level.raw_data_sets

    assert len(data_sets) == 1
    assert data_sets[0].id == "data-set-1"

    parents = set(data_trace.parents(level))
    expected_parents = {
        reading.get_level(id_).get_raw_data_set("data-set-1")
        for id_ in ("level_1", "level_2")
    }

    assert parents == expected_parents

    children = set(data_trace.children(level))
    expected_children = {data_sets[0]}

    assert children == expected_children


def test_process_step_chain_steps():
    """Test chaining of processing steps."""
    step1 = ProcessingStep()
    step1.process_reading = MagicMock()

    step2 = ProcessingStep()
    step2.process_reading = MagicMock()

    chained_step = step1 & step2

    assert isinstance(chained_step, CoreProcessingStepGroup)
    assert len(steps := chained_step.get_steps()) == 2
    assert steps[0] is step1
    assert steps[1] is step2
    assert steps[0].successor is step2
    assert steps[1].predecessor is step1

    step3 = ProcessingStep()
    step3.process_reading = MagicMock()
    added_step = chained_step & step3

    assert isinstance(added_step, CoreProcessingStepGroup)
    assert len(steps := added_step.get_steps()) == 3
    assert steps[2] is step3
    assert steps[2].predecessor is step2
    assert steps[1].successor is step3


def test_reading_flag(reading_example):
    """Test reading flag."""
    reading = deepcopy(reading_example)

    class _ReadingMoreTwoLevels(FlagReadingStep):
        task_name = "task"
        flag_name = "name"
        flag_type = FlagType.TECHNICAL
        flag_severity = FlagSeverity.DEVIATION
        reason = "reason{number}"

        @flag
        def _inv1(self, reading: Reading) -> bool:
            self.set_flag_kwargs(number=1)
            return len(reading) >= 3

        @flag(reason="foo")
        def _inv2(self, reading: Reading) -> bool:
            return len(reading) >= 3

        @flag(number=3)
        def _inv3(self, reading: Reading) -> bool:
            return len(reading) < 3

    process(reading, _ReadingMoreTwoLevels())

    expected = [
        Flag("task-technical-deviation-name", "reason1"),
        Flag("task-technical-deviation-name", "foo"),
    ]
    assert reading.get_flags() == expected


def test_reading_flag_stop_processing(reading_example):
    """Test stop_processing reading flag."""
    reading = deepcopy(reading_example)

    def _flag_reading(reading):
        return len(reading) >= 3

    class _CriticalReadingMoreTwoLevels(FlagReadingStep):
        stop_processing = True
        task_name = "task"
        flag_name = "name"
        flag_type = FlagType.TECHNICAL
        flag_severity = FlagSeverity.INVALIDATION
        reason = "reason"
        flagging_function = _flag_reading

    with pytest.raises(FlagError):
        process(reading, _CriticalReadingMoreTwoLevels())


class _TestParameter(Parameter):
    _registry = {}  # create private registry for tests


class _TestProcessingStepWithParameter(ProcessingStep):
    param_a = _TestParameter("param_a")
    param_b = _TestParameter("param_b", 5, GREATER_THAN_ZERO)

    def process_reading(self, reading: Reading, **kwargs) -> ProcessResultType:
        pass


def test_parameter():
    """Test the creation of parameters."""
    # test that the parameter is in the registry
    prefix = "tests.processing.test_core._TestProcessingStepWithParameter"
    param_a_id = f"{prefix}.param_a"
    param_b_id = f"{prefix}.param_b"

    assert _TestProcessingStepWithParameter.param_a.id == param_a_id
    assert _TestParameter.has_parameter(param_a_id)
    assert _TestParameter.has_parameter(param_b_id)

    # Test flag of parameters when setting (greater than zero validation)
    _TestParameter.set_value(param_b_id, 2)
    with pytest.raises(ValidationException):
        _TestParameter.set_value(param_b_id, -1)

    instance = _TestProcessingStepWithParameter()
    parameters = instance.get_parameters()

    assert len(parameters) == 2
    param_a, param_b = parameters
    assert param_a[0] == "param_a"
    assert param_b[0] == "param_b"
    assert param_a[1] == instance.param_a
    assert param_b[1] == instance.param_b
    assert param_b[1].value == 2
