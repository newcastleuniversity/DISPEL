"""A module that contain functionality to filter walking bouts."""

from typing import List, Union

import pandas as pd

from dispel.data.values import AbbreviatedValue as AV
from dispel.data.values import AVEnum

DataType = Union[pd.DataFrame, List[pd.DataFrame]]


NO_BOUT_MODALITY = AV("no bout", "nb")


class BoutStrategy:
    """A generic class to help filter datasets."""

    def get_view(self, bouts: pd.DataFrame, *data: DataType) -> List[pd.DataFrame]:
        """
        Create a view of the datasets.

        A user can combined information from the walking bout detection and
        the datasets to generate a view of the data on which walking features
        will be computed.

        Parameters
        ----------
        bouts
            Output of the walking bout detection.
        data
            Any dataset or list of datasets.

        Returns
        -------
        List
            A list of the same dataset under a defined view.
        """
        # pylint: disable=unused-argument
        return list(data)


class LongestBoutStrategy(BoutStrategy):
    """Create a view of the data intersecting the longest walking bout."""

    def get_view(self, bouts: pd.DataFrame, *data: DataType) -> List[pd.DataFrame]:
        """Overwrite filter to return longest walking bout."""
        _walking = bouts[bouts.detected_walking]
        duration = _walking["end_time"] - _walking["start_time"]
        if len(duration) > 0:
            longest_bout = _walking.loc[duration.idxmax()]
            t_start = longest_bout.start_time
            t_end = longest_bout.end_time
            return [df[t_start:t_end] for df in list(data)]
        return list(data)


class FirstTwoMinBoutStrategy(BoutStrategy):
    """Create a view of the data returning the first two minutes."""

    def get_view(self, bouts: pd.DataFrame, *data: DataType) -> List[pd.DataFrame]:
        """Overwrite filter to return the first two minutes of the data."""
        two_min = bouts.start_time.min() + pd.Timedelta(120, unit="s")
        return [
            df if df.empty else df[df.index < two_min]  # type: ignore
            for df in list(data)
        ]


class BoutStrategyModality(AVEnum):
    """Enumerate bout strategy modalities."""

    FIRST_TWO = AV("first two minutes", "2min")
    LONGEST = AV("longest bout", "lb")
    BASIC = AV("all bouts", "ab")

    @property
    def bout_cls(self):
        """Return BoutStrategy instance."""
        mapping = {
            self.LONGEST: LongestBoutStrategy(),
            self.FIRST_TWO: FirstTwoMinBoutStrategy(),
            self.BASIC: BoutStrategy(),
        }
        return mapping[self]
