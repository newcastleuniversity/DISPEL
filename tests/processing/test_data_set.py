"""Test :mod:`dispel.processing.data_set`."""
from copy import deepcopy

import pandas as pd

from dispel.data.core import Reading
from dispel.data.flags import Flag, FlagSeverity, FlagType
from dispel.data.levels import Level
from dispel.processing import process
from dispel.processing.data_set import FlagDataSetStep
from dispel.processing.flags import flag


def test_data_set_flag_unique_data_set(reading_example):
    """Test flag of one data set."""
    reading = deepcopy(reading_example)

    class _FlagDataSetMean(FlagDataSetStep):
        data_set_ids = "data-set-1"
        task_name = "task"
        flag_name = "name"
        flag_type = FlagType.TECHNICAL
        flag_severity = FlagSeverity.DEVIATION
        reason = "reason{number}"

        @flag(number=1)
        def _inv1(self, data: pd.DataFrame) -> bool:
            return data.mean().mean() > 1

        @flag(number=2)
        def _inv2(self, data, level: Level, reading: Reading) -> bool:
            assert isinstance(level, Level)
            assert isinstance(reading, Reading)
            return data.mean().mean() <= 1

    process(reading, _FlagDataSetMean())

    raw_data_set1 = reading.get_level("level_1").get_raw_data_set("data-set-1")
    expected1 = [Flag("task-technical-deviation-name", "reason1")]
    assert raw_data_set1.get_flags() == expected1

    raw_data_set2 = reading.get_level("level_2").get_raw_data_set("data-set-1")
    expected2 = [Flag("task-technical-deviation-name", "reason2")]
    assert raw_data_set2.get_flags() == expected2
