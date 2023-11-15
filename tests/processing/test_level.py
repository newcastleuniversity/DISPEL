"""Tests for :mod:`dispel.processing.filters`."""
from copy import deepcopy
from typing import Iterable, Set
from unittest.mock import MagicMock

import pandas as pd
import pytest

from dispel.data.core import Evaluation, Reading
from dispel.data.epochs import EpochDefinition
from dispel.data.features import FeatureSet, FeatureValue
from dispel.data.flags import Flag, FlagSeverity, FlagType
from dispel.data.levels import Level
from dispel.data.raw import (
    RawDataSet,
    RawDataSetDefinition,
    RawDataSetSource,
    RawDataValueDefinition,
)
from dispel.data.values import ValueDefinition
from dispel.processing import process
from dispel.processing.core import ProcessingStep, ProcessResultType
from dispel.processing.flags import flag
from dispel.processing.level import (
    FlagLevelStep,
    LevelFilter,
    LevelIdFilter,
    LevelProcessingStep,
    ProcessingStepGroup,
)


def test_level_filter():
    """Test boolean logic of filters."""

    class _TestFilter(LevelFilter):
        def __init__(self, res):
            self.res = res

        def repr(self) -> str:
            return str(self.res)

        def filter(self, levels: Iterable[Level]) -> Set[Level]:
            return set(levels) if self.res else set()

    level1 = Level(id_="test", start="now", end="now")
    filter1 = _TestFilter(False)
    filter2 = _TestFilter(True)

    # Test boolean operations
    assert level1 not in (filter1 & filter2).filter([level1])
    assert level1 in (filter1 | filter2).filter([level1])
    assert level1 not in (filter1 | filter1).filter([level1])
    assert level1 in (~filter1).filter([level1])
    assert level1 not in (~filter2).filter([level1])


def test_level_id_filter():
    """Test level id filters."""
    level1 = Level(id_="test", start="now", end="now")
    filter1 = LevelIdFilter("test")
    filter2 = LevelIdFilter("other")

    assert level1 in filter1.filter([level1])
    assert level1 not in filter2.filter([level1])


def test_level_processing_step_level_id():
    """Test processing with filters."""
    level = Level(id_="test", start="now", end="now")
    reading = Reading(
        Evaluation(
            uuid="test",
            start=level.start,
            end=level.end,
            definition=EpochDefinition(id_="test"),
        ),
        levels=[level],
    )

    # level is consumed
    class _TestLevelProcessingStep(LevelProcessingStep):
        process_level = MagicMock()

    step1 = _TestLevelProcessingStep()

    _ = list(step1.process(reading))
    step1.process_level.assert_called_once()
    step1.process_level.assert_called_with(level, reading)

    # level is not consumed
    class _TestNotConsumeLevelProcessingStep(LevelProcessingStep):
        process_level = MagicMock()

    step2 = _TestNotConsumeLevelProcessingStep(level_filter="other")

    _ = list(step2.process(reading))
    step2.process_level.assert_not_called()


def test_level_flag_step(reading_example):
    """Test level flag."""
    reading = deepcopy(reading_example)

    class _FlagLevelContext(FlagLevelStep):
        task_name = "task"
        flag_name = "name"
        flag_type = FlagType.TECHNICAL
        flag_severity = FlagSeverity.DEVIATION
        reason = "reason{number}"

        @flag(number=1)
        def _inv1(self, level: Level) -> bool:
            return level.context.has_value("name_1")

        @flag(number=2)
        def _inv2(self, level: Level) -> bool:
            return level.context.has_value("name_3")

        @flag(number=3)
        def _inv3(self, level, reading) -> bool:
            assert isinstance(level, Level)
            assert isinstance(reading, Reading)
            return False

    process(reading, _FlagLevelContext())

    expected1 = [
        Flag("task-technical-deviation-name", "reason2"),
        Flag("task-technical-deviation-name", "reason3"),
    ]
    assert reading.get_level("level_1").get_flags() == expected1

    expected2 = [
        Flag("task-technical-deviation-name", "reason1"),
        Flag("task-technical-deviation-name", "reason3"),
    ]
    assert reading.get_level("level_2").get_flags() == expected2


def test_processing_step_group_arguments():
    """Test successful passing of processing group arguments."""
    step1 = ProcessingStep()
    step1.process = MagicMock()

    step2 = ProcessingStep()
    step2.process = MagicMock()

    group = ProcessingStepGroup([step1, step2], foo=1, bar="baz")
    _ = list(group.process(None, foo=3, bam=2))

    step1.process.assert_called_with(None, foo=1, bar="baz", bam=2)
    step2.process.assert_called_with(None, foo=1, bar="baz", bam=2)


def test_processing_step_group_yield():
    """Test yielding all sub-results from processing step groups."""
    data_set = RawDataSet(
        RawDataSetDefinition(
            "some",
            RawDataSetSource("example"),
            [RawDataValueDefinition("bar", "baz")],
            True,
        ),
        pd.DataFrame(dict(bar=[0, 1, 2, 3])),
    )

    class _StepExtract(ProcessingStep):
        def process_reading(self, reading: Reading, **kwargs) -> ProcessResultType:
            yield data_set

    feature = FeatureValue(ValueDefinition("foo", "bar"), 5)

    class _StepFeatureValue(ProcessingStep):
        def process_reading(self, reading: Reading, **kwargs) -> ProcessResultType:
            yield feature

    feature_set = FeatureSet([feature])

    class _StepFeatureSet(ProcessingStep):
        def process_reading(self, reading: Reading, **kwargs) -> ProcessResultType:
            yield feature_set

    group = ProcessingStepGroup(
        [_StepExtract(), _StepFeatureValue(), _StepFeatureSet()]
    )
    res = list(group.process(None))

    assert res == [data_set, feature, feature_set]


@pytest.mark.filterwarnings("ignore:.*was not decorated.*")
def test_processing_step_group_filter():
    """Test injecting level filter into processing groups."""
    step1 = ProcessingStep()
    step1.process_reading = MagicMock()

    class _TestLevelProcessingStep(LevelProcessingStep):
        process_level = MagicMock()

    step2 = _TestLevelProcessingStep()

    class _CustomLevelFilter(LevelFilter):
        def __init__(self, val: str):
            self.val = val

        def filter(self, levels: Iterable[Level]) -> Set[Level]:
            return set(levels)

        def repr(self) -> str:
            return f"custom {self.val}"

    class _TestOtherLevelProcessingStep(LevelProcessingStep):
        process_level = MagicMock()

    step3 = _TestOtherLevelProcessingStep(level_filter=_CustomLevelFilter("a"))

    group = ProcessingStepGroup(
        [step1, step2, step3],
        level_filter=_CustomLevelFilter("b"),
    )

    decorated_steps = group.get_steps()

    assert not hasattr(decorated_steps[0], "level_filter")
    repr_step2 = repr(decorated_steps[1].get_level_filter())
    assert repr_step2 == "<_CustomLevelFilter: custom b>"
    repr_step3 = repr(decorated_steps[2].get_level_filter())
    assert repr_step3 == "<LevelFilter: (custom a and custom b)>"


def test_processing_step_group_cascaded_filter():
    """Test cascading level filters in multiple groups."""

    class _DummyFilter(LevelFilter):
        def __init__(self, name):
            self.name = name

        def filter(self, levels: Iterable[Level]) -> Set[Level]:
            return set(levels)

        def repr(self) -> str:
            return self.name

    class _TestLevelProcessingStep(LevelProcessingStep):
        process_level = MagicMock()

    lf1 = _DummyFilter("A")
    step1 = _TestLevelProcessingStep(level_filter=lf1)

    assert step1.get_level_filter() == lf1

    lf2 = _DummyFilter("B")
    step2 = ProcessingStepGroup([step1], level_filter=lf2)

    assert repr(step1.get_level_filter()) == repr(lf1 & lf2)

    lf3 = _DummyFilter("C")
    _ = ProcessingStepGroup([step2], level_filter=lf3)

    assert repr(step1.get_level_filter()) == repr(lf1 & (lf2 & lf3))


def test_processing_step_group_respect_overwritten_step_level_filter(reading_example):
    """Test that groups actually use get_level_filter() to filter."""

    class _NoLevels(LevelFilter):
        def repr(self) -> str:
            return "None"

        def filter(self, levels: Iterable[Level]) -> Set[Level]:
            return set()

    class _Step(LevelProcessingStep):
        process_level = MagicMock()

        def get_level_filter(self) -> LevelFilter:
            return LevelIdFilter("level_1")

    class _Group(ProcessingStepGroup):
        steps = [_Step()]
        level_filter = _NoLevels()

    reading = deepcopy(reading_example)
    process(reading, _Group())

    _Step.process_level.assert_not_called()


def _process_step_factory():
    """Get a mocked level processing step."""

    class _Step(LevelProcessingStep):
        process_level = MagicMock()

    return _Step()


def test_processing_step_group_allow_changing_level_filter(reading_example):
    """Test that level filters are chained correctly."""

    class _GroupStep(ProcessingStepGroup):
        steps = [_process_step_factory()]
        level_filter = LevelIdFilter("level_1")

    reading = deepcopy(reading_example)
    step = _GroupStep()
    process(reading, step)

    step.steps[0].process_level.assert_called_once()

    # Change filter
    step.level_filter = LevelIdFilter("non-existent")
    reading = deepcopy(reading_example)
    process(reading, step)

    # Since it is the same process function it was already called so we have
    # to use assert called once
    step.steps[0].process_level.assert_called_once()


def test_processing_step_group_allow_overwriting_get_filter(reading_example):
    """Test allowing to overwrite the group level filter."""

    class _OverwrittenGroup(ProcessingStepGroup):
        steps = [_process_step_factory()]

        def get_level_filter(self) -> LevelFilter:
            return LevelIdFilter("level_2") & self.level_filter

    reading = deepcopy(reading_example)
    step = _OverwrittenGroup()
    process(reading, step)

    step.steps[0].process_level.assert_called_once()


def test_processing_step_group_reassignment_warning():
    """Test that the user is warned if a step is re-assigned."""

    class _Step(LevelProcessingStep):
        process_level = MagicMock()

        def get_level_filter(self) -> LevelFilter:
            return LevelIdFilter("dummy")

    step = _Step()
    ProcessingStepGroup(steps=[step])

    with pytest.warns(UserWarning):
        ProcessingStepGroup(steps=[step])
