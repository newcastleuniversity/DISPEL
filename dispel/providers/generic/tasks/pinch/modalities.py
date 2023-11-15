"""A module containing functionality to process/filter specific modalities."""
from typing import Iterable, Set

from dispel.data.levels import Level
from dispel.data.values import AVEnum
from dispel.processing.level import LevelFilter


class BubbleSizeModality(AVEnum):
    """Bubble size modality."""

    SMALL = ("small bubbles", "s")
    MEDIUM = ("medium bubbles", "m")
    LARGE = ("large bubbles", "l")
    EXTRA_LARGE = ("extra large bubbles", "xl")


class FingerModality(AVEnum):
    """Pinching fingers modality."""

    TOP_FINGER = ("top finger", "tf")
    BOTTOM_FINGER = ("bottom finger", "bf")


class AttemptOutcomeModality(AVEnum):
    """Pinching attempt success modality."""

    @property
    def is_success(self) -> bool:
        """Check if it is success modality."""
        return self == self.SUCCESS

    SUCCESS = ("successful pinches", "sp")
    FAILURE = ("failed pinches", "fp")
    ALL = ("all pinches", "ap")


class AttemptSelectionModality(AVEnum):
    """Pinching attempt selection modality."""

    @property
    def is_first(self) -> bool:
        """Check if it is the first attempt."""
        return self == self.FIRST

    FIRST = ("first attempt", "fa")
    ALL = ("all attempts", "aa")


class BubbleSizeModalityFilter(LevelFilter):
    """Filter for same bubble size modality."""

    def __init__(self, size: BubbleSizeModality):
        self.size = size

    def repr(self):
        """Get representation of the filter."""
        return f"only {self.size.av}"

    def filter(self, levels: Iterable[Level]) -> Set[Level]:
        """Filter levels performed with a specific bubble size."""
        return set(
            filter(
                lambda level: self.size.variable
                == level.context.get_raw_value("bubbleSize"),
                filter(lambda level: level.context.has_value("bubbleSize"), levels),
            )
        )
