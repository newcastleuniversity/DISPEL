"""A module that contains functionality to filter SBT bouts."""

from typing import List, Union

import pandas as pd

from dispel.data.core import Reading
from dispel.data.levels import Level
from dispel.data.values import AbbreviatedValue as AV
from dispel.data.values import AVEnum
from dispel.processing.core import ProcessResultType
from dispel.processing.data_set import decorated_processing_function
from dispel.processing.extract import ExtractStep

DataType = Union[pd.DataFrame, List[pd.DataFrame]]


class SBTBoutStrategy:
    """A generic class to help filter datasets."""

    def get_view(self, data: DataType) -> List[pd.DataFrame]:
        """
        Create a view of the datasets.

        Parameters
        ----------
        data
            Any dataset or list of datasets.

        Returns
        -------
        List
            A list of the same dataset under a defined view.
        """
        # pylint: disable=unused-argument
        return list(data)


class FirstFiveSec(SBTBoutStrategy):
    """Create a view of the data, returning data within first five seconds."""

    def get_view(self, data: DataType) -> List[pd.DataFrame]:
        """Overwrite filter to return data within first five seconds."""
        start_time = min([df.index.min() for df in data])
        five_sec = start_time + pd.Timedelta(5, unit="s")
        return [
            df if df.empty else df[df.index < five_sec]  # type: ignore
            for df in list(data)
        ]


class AfterFiveSec(SBTBoutStrategy):
    """Create a view of the data, returning data after five seconds."""

    def get_view(self, data: DataType) -> List[pd.DataFrame]:
        """Overwrite filter to return data after five seconds."""
        start_time = min([df.index.min() for df in data])
        five_sec = start_time + pd.Timedelta(5, unit="s")
        return [
            df if df.empty else df[df.index >= five_sec]  # type: ignore
            for df in list(data)
        ]


class SBTBoutStrategyModality(AVEnum):
    """Enumerate bout strategy modalities."""

    FIRST_FIVE = AV("first five seconds", "first5s")
    AFTER_FIVE = AV("after five seconds", "after5s")
    BASIC = AV("complete signal", "full")

    @property
    def bout_cls(self):
        """Return BoutStrategy instance."""
        mapping = {
            self.FIRST_FIVE: FirstFiveSec(),
            self.AFTER_FIVE: AfterFiveSec(),
            self.BASIC: SBTBoutStrategy(),
        }
        return mapping[self]


class SBTBoutExtractStep(ExtractStep):
    """Base class for SBT bouts feature extraction."""

    bout_strategy: SBTBoutStrategy

    def __init__(self, *args, **kwargs):
        bout_strategy = kwargs.pop("bout_strategy", None)
        kwargs_copy = kwargs.copy()
        if "modalities" in kwargs:
            kwargs_copy.pop("modalities")

        super().__init__(*args, **kwargs_copy)
        if bout_strategy:
            self.bout_strategy = bout_strategy

    def process_level(
        self, level: Level, reading: Reading, **kwargs
    ) -> ProcessResultType:
        """Overwrite process_level."""
        # pylint: disable=unpacking-non-sequence
        data_sets = self.get_data_frames(level)
        filtered_data_sets = self.bout_strategy.get_view(data_sets)

        for function, func_kwargs in self.get_transform_functions():
            merged_kwargs = kwargs.copy()
            merged_kwargs.update(func_kwargs)
            yield from self.wrap_result(
                decorated_processing_function(
                    function, filtered_data_sets, reading, level
                ),
                level,
                reading,
                **merged_kwargs,
            )
