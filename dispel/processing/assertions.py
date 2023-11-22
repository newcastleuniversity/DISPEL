"""Assertions to be made on readings as part of processing steps."""
from abc import ABCMeta
from typing import Optional, Sequence

import pandas as pd
from deprecated.sphinx import deprecated

from dispel.data.core import Reading
from dispel.data.levels import Level
from dispel.processing.core import (
    ErrorHandling,
    ProcessingControlResult,
    ProcessingStep,
    ProcessResultType,
    StopProcessingError,
)
from dispel.processing.data_set import DataSetProcessingStepProtocol
from dispel.processing.level import (
    LevelFilterType,
    LevelProcessingControlResult,
    LevelProcessingStep,
)


class AssertEvaluationFinished(ProcessingStep):
    """Assertion to ensure evaluations are finished."""

    def process_reading(self, reading: Reading, **kwargs) -> ProcessResultType:
        """Ensure reading evaluation is finished."""
        if not reading.evaluation.finished:
            yield ProcessingControlResult(
                step=self,
                error=StopProcessingError("evaluation is not finished", self),
                error_handling=ErrorHandling.RAISE,
            )


@deprecated(version="0.0.51", reason="Use assert_level_valid")
class AssertRawDataSetPresent(LevelProcessingStep):
    """Assertion to ensure specific data sets are present."""

    def __init__(
        self, data_set_id: str, level_filter: Optional[LevelFilterType] = None
    ):
        super().__init__(level_filter=level_filter)
        self.data_set_id = data_set_id

    def process_level(
        self, level: Level, reading: Reading, **kwargs
    ) -> ProcessResultType:
        """Ensure level has data set id."""
        if not level.has_raw_data_set(self.data_set_id):
            yield LevelProcessingControlResult(
                step=self,
                error=StopProcessingError(
                    f"data set {self.data_set_id} is missing", self
                ),
                level=level,
                error_handling=ErrorHandling.RAISE,
            )


class NotEmptyDataSetAssertionMixin(DataSetProcessingStepProtocol, metaclass=ABCMeta):
    """A mixin to ensure that processed data sets are not empty."""

    #: The assertion message
    assertion_message = "Empty dataset {data_set_id} for level {level}"

    #: The handling if a data set is empty
    empty_data_set_handling = ErrorHandling.RAISE

    def assert_valid_data_sets(
        self,
        data_sets: Sequence[pd.DataFrame],
        level: Level,
        reading: Reading,
        **kwargs,
    ):
        """Assert that data sets are not empty."""
        super().assert_valid_data_sets(data_sets, level, reading, **kwargs)
        for data, data_set_id in zip(data_sets, self.get_data_set_ids()):
            assert not data.empty, (
                self.assertion_message.format(data_set_id=data_set_id, level=level),
                self.empty_data_set_handling,
            )
