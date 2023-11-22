"""Helpers for tests on :mod:`dispel.data`."""

import pandas as pd

from dispel.data.core import Evaluation, Level, Reading
from dispel.data.epochs import EpochDefinition
from dispel.data.raw import (
    RawDataSet,
    RawDataSetDefinition,
    RawDataSetSource,
    RawDataValueDefinition,
)


def get_empty_reading_and_level(
    level_id: str = "test", evaluation_start="now", evaluation_end="now"
):
    """Get an empty reading with one level."""
    evaluation = Evaluation(
        uuid="test",
        start=evaluation_start,
        end=evaluation_end,
        definition=EpochDefinition(id_="test"),
    )

    level = Level(id_=level_id, start=evaluation.start, end=evaluation.end)
    reading = Reading(evaluation, levels=[level])

    return reading, level


def get_raw_data_set(data: pd.DataFrame, data_set_id: str = "test") -> RawDataSet:
    """Get a data set representation of a pandas data set."""
    return RawDataSet(
        RawDataSetDefinition(
            data_set_id,
            RawDataSetSource("test"),
            value_definitions_list=[
                RawDataValueDefinition(name, name) for name in data.columns
            ],
        ),
        data,
    )
