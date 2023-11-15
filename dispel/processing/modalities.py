"""A module containing functionality to process specific modalities."""

from typing import Iterable, Set

from dispel.data.levels import Level
from dispel.data.values import AVEnum
from dispel.processing.level import LevelFilter


class HandModality(AVEnum):
    """Handedness for tasks that are applied to different hands."""

    LEFT = ("left hand", "left")
    RIGHT = ("right hand", "right")


class HandModalityFilter(LevelFilter):
    """Filter for same hand modality."""

    def __init__(self, hand: HandModality):
        self.hand = hand

    def repr(self):
        """Get representation of the filter."""
        return f"only {self.hand.av}>"

    def filter(self, levels: Iterable[Level]) -> Set[Level]:
        """Filter levels performed with a specific hand."""
        return set(
            filter(
                lambda x: "usedHand" in x.context
                and x.context.get_raw_value("usedHand") == self.hand.abbr,
                levels,
            )
        )


class SensorModality(AVEnum):
    # FIXME remove class
    """Sensor types enumerator."""

    def unit(self, order: int = 1) -> str:
        """Get the unit of the sensor signal.

        Parameters
        ----------
        order
            The unit order.

        Returns
        -------
        str
            The unit of the sensor.
        """
        basis = {"acc": "G", "gyr": "rad/s", "itrem": "pixel"}[self.abbr]
        if order == 1:
            return basis
        return "/".join([x + f"^{order}" for x in basis.split("/")])

    ACCELEROMETER = ("accelerometer", "acc")
    GYROSCOPE = ("gyroscope", "gyr")
    INTENTIONAL = ("intentional tremors", "itrem")


class LimbModality(AVEnum):
    """Type of limb exercises enumerator."""

    UPPER_LIMB = ("upper limb", "upper")
    LOWER_LIMB = ("lower limb", "lower")
