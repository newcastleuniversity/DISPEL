"""Filters used to process specific levels."""
from typing import Iterable, Set

from dispel.data.levels import Level
from dispel.processing.level import LevelFilter


class DurationFilter(LevelFilter):
    """A level filter to fetch level less than a given duration."""

    def __init__(self, max_duration: float):
        self.max_duration = max_duration

    def repr(self) -> str:
        """Get representation of the filter."""
        return f"Level with a duration < {self.max_duration} seconds."

    def filter(self, levels: Iterable[Level]) -> Set[Level]:
        """Keep all levels with duration less than max_duration."""
        out = set()
        for level in levels:
            if level.duration.total_seconds() < self.max_duration:
                out.add(level)
        return out


class AbsentDataSetFilter(LevelFilter):
    """Filter out levels with absent data set ids."""

    def __init__(self, data_set_id: str):
        self.data_set_id = data_set_id

    def repr(self):
        """Get representation of the filter."""
        return f"only levels with {self.data_set_id}>"

    def filter(self, levels: Iterable[Level]) -> Set[Level]:
        """Filter levels missing the specified data set id."""
        return set(filter(lambda x: x.has_raw_data_set(self.data_set_id), levels))


class NotEmptyDatasetFilter(LevelFilter):
    """Filter out levels without the dataset or with empty dataset."""

    def __init__(self, data_set_id: str):
        self.data_set_id = data_set_id

    def repr(self):
        """Get representation of the filter."""
        return f"only levels with dataset {self.data_set_id} and that is notempty."

    def filter(self, levels: Iterable[Level]) -> Set[Level]:
        """Filter levels missing a dataset or with empty dataset."""
        not_empty_levels = filter(
            lambda x: x.has_raw_data_set(self.data_set_id), levels
        )
        return set(
            filter(
                lambda x: len(x.get_raw_data_set(self.data_set_id).data) > 0,
                not_empty_levels,
            )
        )


class LastLevelFilter(LevelFilter):
    """A level filter to process only the last level."""

    def repr(self) -> str:
        """Get representation of the filter."""
        return "last level"

    def filter(self, levels: Iterable[Level]) -> Set[Level]:
        """Keep last level."""
        last_level = None
        out = set()
        for level in levels:
            last_level = level
        if last_level:
            out.add(last_level)
        return out


class NotEmptyDataSetFilter(LevelFilter):
    """Filter out levels with empty data set."""

    def __init__(self, data_set_id: str):
        self.data_set_id = data_set_id

    def repr(self):
        """Get representation of the filter."""
        return f"only level with not empty {self.data_set_id}>"

    def filter(self, levels: Iterable[Level]) -> Set[Level]:
        """Filter levels with empty dataset."""
        return set(
            filter(lambda x: len(x.get_raw_data_set(self.data_set_id).data) > 0, levels)
        )
