"""A module to detect and process phone orientation during tests."""
import math
from collections import namedtuple
from operator import ge, le
from typing import Callable, Iterable, Union

import pandas as pd

from dispel.data.values import AVEnum

#: tuple of phone range orientation in degree
OrientationRange = namedtuple("OrientationRange", ["lower", "upper"])


class PhoneOrientation(AVEnum):
    """A class enumerator regrouping supported phone orientations."""

    LANDSCAPE_RIGHT = ("landscape right mode", "lrm")
    LANDSCAPE_LEFT = ("landscape left mode", "llm")
    PORTRAIT_UPRIGHT = ("portrait upright mode", "upm")
    PORTRAIT_UPSIDE_DOWN = ("portrait upside down mode", "udpm")
    FACE_UP = ("face up mode", "fum")
    FACE_DOWN = ("face down mode", "fdm")

    @property
    def is_portrait(self) -> bool:
        """Return whether the orientation is in portrait mode."""
        return self in (self.PORTRAIT_UPRIGHT, self.PORTRAIT_UPSIDE_DOWN)

    @property
    def is_landscape(self) -> bool:
        """Return whether the orientation is in landscape mode."""
        return self in (self.LANDSCAPE_RIGHT, self.LANDSCAPE_LEFT)

    @property
    def is_flat(self) -> bool:
        """Return whether the orientation is in a flat mode."""
        return self in (self.FACE_UP, self.FACE_DOWN)

    @property
    def is_up(self) -> bool:
        """Return whether the orientation is upright."""
        return self in (self.LANDSCAPE_RIGHT, self.PORTRAIT_UPRIGHT, self.FACE_UP)

    @property
    def is_down(self) -> bool:
        """Return whether the orientation is upside down."""
        return self in (self.LANDSCAPE_LEFT, self.PORTRAIT_UPSIDE_DOWN, self.FACE_DOWN)

    @property
    def frontal_axis(self) -> str:
        """Get the name of the axis facing the observer."""
        if self.is_landscape or self.is_portrait:
            return "gravityZ"
        if self.is_flat:
            return "gravityX"
        raise ValueError("Unsupported phone orientation.")

    @property
    def top_axis(self) -> str:
        """Get the name of the axis perpendicular to the ground."""
        if self.is_landscape:
            return "gravityX"
        if self.is_portrait:
            return "gravityY"
        if self.is_flat:
            return "gravityZ"
        raise ValueError("Unsupported phone orientation.")

    @property
    def side_axis(self) -> str:
        """Get the name of the axis on the side of the phone."""
        if self.is_landscape or self.is_flat:
            return "gravityY"
        if self.is_portrait:
            return "gravityX"
        raise ValueError("Unsupported phone orientation.")

    def _get_top_axis_threshold(self, pitch_angle: float, yaw_angle: float) -> float:
        """Get the threshold for the top axis."""
        angle = min(pitch_angle, yaw_angle) if self.is_flat else pitch_angle
        return math.cos(angle)

    def _get_top_axis_sign(self) -> int:
        """Get the sign of the top axis."""
        if self.is_up:
            return -1
        if self.is_down:
            return 1
        raise ValueError("Phone orientation can only be facing up or down.")

    def is_top_axis_valid(
        self, top_axis_gravity: float, pitch_angle: float, yaw_angle: float
    ) -> bool:
        """Check whether the condition on the top axis is verified."""
        threshold = self._get_top_axis_threshold(pitch_angle, yaw_angle)
        operator = ge if (sign := self._get_top_axis_sign()) > 0 else le
        return operator(top_axis_gravity, sign * threshold)

    def get_classifier(
        self,
        pitch_freedom: Union[float, OrientationRange],
        yaw_freedom: Union[float, OrientationRange],
    ) -> Callable[[pd.DataFrame], pd.Series]:
        """Get the phone orientation classifier.

        Parameters
        ----------
        pitch_freedom
            The degree of freedom of the pitch angle in degrees.
        yaw_freedom
            The degree of freedom of the yaw angle in degrees.

        Returns
        -------
        Callable[[pandas.DataFrame], pandas.Series]
            A function that takes as input the gravity data frame and output a pandas
            series of boolean corresponding to the phone orientation classification.
        """

        def _unitary_classifier(row: pd.Series) -> bool:
            if not isinstance(pitch_freedom, OrientationRange):
                pitch_range = OrientationRange(
                    lower=-pitch_freedom / 2, upper=pitch_freedom / 2
                )
            else:
                pitch_range = pitch_freedom

            if not isinstance(yaw_freedom, OrientationRange):
                yaw_range = OrientationRange(
                    lower=-yaw_freedom / 2, upper=yaw_freedom / 2
                )
            else:
                yaw_range = yaw_freedom

            pitch_angle_low = math.radians(pitch_range.lower)
            pitch_angle_high = math.radians(pitch_range.upper)
            yaw_angle_low = math.radians(yaw_range.lower)
            yaw_angle_high = math.radians(yaw_range.upper)

            return (
                self.is_top_axis_valid(
                    row[self.top_axis], pitch_angle_low, yaw_angle_low
                )
                and math.sin(pitch_angle_low)
                <= row[self.frontal_axis]
                <= math.sin(pitch_angle_high)
                and math.sin(yaw_angle_low)
                <= row[self.side_axis]
                <= math.sin(yaw_angle_high)
            )

        def _classifier(data: pd.DataFrame) -> pd.Series:
            expected_columns = {f"gravity{axis}" for axis in "XYZ"}
            assert expected_columns <= (
                columns := set(data.columns)
            ), f"Missing gravity columns {expected_columns - columns}"

            return data.apply(_unitary_classifier, axis=1)

        return _classifier


PhoneOrientationType = Union[PhoneOrientation, Iterable[PhoneOrientation]]
