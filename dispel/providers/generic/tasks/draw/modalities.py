"""A module containing functionality to process/filter specific modalities."""
from typing import Iterable, Set

from dispel.data.levels import Level
from dispel.data.values import AVEnum
from dispel.processing.level import LevelFilter


class ShapeModality(AVEnum):
    """A modality for shapes."""

    SQUARE = ("square shape", "sc")
    SQUARE_COUNTER_CLOCK = ("square counter-clock shape", "scc")
    INFINITY = ("infinity shape", "inf")
    SPIRAL = ("spiral shape", "spi")


class AttemptModality(AVEnum):
    """A modality for drawing attempts."""

    FIRST = ("first attempt", "first")
    SECOND = ("second attempt", "sec")


_shape_mapping = {
    ShapeModality.SQUARE: "squareClock",
    ShapeModality.SQUARE_COUNTER_CLOCK: "squareCounterClock",
    ShapeModality.SPIRAL: "spiral",
    ShapeModality.INFINITY: "infinity",
}

_attempt_mapping = {AttemptModality.FIRST: 1, AttemptModality.SECOND: 2}


class AttemptModalityFilter(LevelFilter):
    """Filter for same attempt modality."""

    def __init__(self, attempt: AttemptModality):
        self.attempt = attempt

    def repr(self):
        """Get representation of the filter."""
        return f"only {self.attempt.av}>"

    def filter(self, levels: Iterable[Level]) -> Set[Level]:
        """Filter levels performed with a specific attempt."""
        return set(
            filter(
                lambda x: "attempt" in x.context
                and x.context.get_raw_value("attempt")
                == _attempt_mapping[self.attempt],
                levels,
            )
        )


class ShapeModalityFilter(LevelFilter):
    """Filter for same shape modality."""

    def __init__(self, shape: ShapeModality):
        self.shape = shape

    def repr(self):
        """Get representation of the filter."""
        return f"only {self.shape.av}>"

    def filter(self, levels: Iterable[Level]) -> Set[Level]:
        """Filter levels performed with a specific attempt."""
        return set(
            filter(
                lambda x: "levelType" in x.context
                and x.context.get_raw_value("levelType") == _shape_mapping[self.shape],
                levels,
            )
        )


class CompletedDrawFilter(LevelFilter):
    """Filter for completed drawing shape."""

    def repr(self):
        """Get representation of the filter."""
        return "only completed draw shapes>"

    def filter(self, levels: Iterable[Level]) -> Set[Level]:
        """Filter levels with incomplete drawn shapes."""

        def _filter(level: Level):
            if level.has_raw_data_set("screen"):
                # ADS Format
                screen = level.get_raw_data_set("screen").data
                if "inEndZone" in screen:
                    return True  # screen.inEndZone.any()
            # BDH Format
            return True

        return set(filter(_filter, levels))
