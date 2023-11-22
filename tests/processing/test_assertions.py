"""Tests for :mod:`dispel.processing.core`."""
from unittest.mock import MagicMock

import pandas as pd
import pytest

from dispel.data.core import Evaluation, Reading
from dispel.data.epochs import EpochDefinition
from dispel.processing import process
from dispel.processing.assertions import (
    AssertEvaluationFinished,
    AssertRawDataSetPresent,
    NotEmptyDataSetAssertionMixin,
)
from dispel.processing.core import InvalidDataError, ProcessingStep, StopProcessingError
from dispel.processing.data_set import DataSetProcessingStep
from dispel.processing.level import LevelProcessingControlResult
from tests.data.helper import get_empty_reading_and_level, get_raw_data_set


class _MockProcessingStep(ProcessingStep):
    """A mock class for processing steps."""

    process_reading = MagicMock()


@pytest.mark.parametrize("finished", [True, False])
def test_assert_evaluation_finished(finished):
    """Test assertion step for finished evaluations."""
    evaluation = Evaluation(
        start="now",
        end="now",
        uuid="test",
        finished=finished,
        definition=EpochDefinition(id_="test"),
    )
    reading = Reading(evaluation=evaluation)

    check_step = _MockProcessingStep()

    steps = [AssertEvaluationFinished(), check_step]

    if not finished:
        with pytest.raises(StopProcessingError):
            process(reading, steps)
    else:
        process(reading, steps)

    assert check_step.process_reading.called


@pytest.mark.parametrize("add_data_set", [True, False])
def test_assert_raw_data_set_present(add_data_set):
    """Test assertion step for present data sets."""
    data_set = get_raw_data_set(pd.DataFrame(dict(test=range(10))))
    reading, level = get_empty_reading_and_level()

    check_step = _MockProcessingStep()

    steps = [AssertRawDataSetPresent("test"), check_step]

    if add_data_set:
        reading.set(data_set, level=level)
        process(reading, steps)
    else:
        with pytest.raises(StopProcessingError):
            process(reading, steps)

    assert check_step.process_reading.called


def test_assert_not_empty_data_set_assertion_mixin():
    """Test that the assertion mixin for empty data sets works."""

    class _Test(NotEmptyDataSetAssertionMixin, DataSetProcessingStep):
        data_set_ids = ["test"]
        assertion_message = "This {level.id} does not contain data on {data_set_id}"

    reading, level = get_empty_reading_and_level()
    level.set(get_raw_data_set(pd.DataFrame(columns=["test"])))

    step = _Test()
    res = next(step.process(reading))

    assert isinstance(res, LevelProcessingControlResult)
    assert res.level is level
    assert isinstance(res.error, InvalidDataError)
    assert res.error.step is step
    assert str(res.error) == "This test does not contain data on test at step: _Test."
    assert res.error_handling == _Test.empty_data_set_handling
