"""A module dedicated to format user pinch into attempts."""
import datetime
import re
from dataclasses import dataclass, field
from functools import cached_property, partial
from typing import ClassVar, List, Optional, Sequence, Tuple, Type, Union, cast

import numpy as np
import pandas as pd

from dispel.data.levels import Context, Level
from dispel.processing.modalities import HandModality
from dispel.providers.generic.tasks.pinch.modalities import (
    AttemptOutcomeModality,
    AttemptSelectionModality,
    BubbleSizeModality,
    FingerModality,
)
from dispel.providers.generic.touch import Gesture, Touch
from dispel.signal.core import euclidean_distance
from dispel.utils import plural


def _remove_attempt(level_id: str):
    """Remove attempt enumerator from level id."""
    return re.sub(r"-\d+$", "", level_id)


@dataclass
class PinchTouch(Touch):
    """A pinch touch interaction."""

    #: Is successful if the gesture of the touch led to the burst of the target
    is_successful: bool = field(init=False)

    def __post_init__(self, data: pd.DataFrame):
        super().__post_init__(data)

        # Check if the pinching was successful
        assert "ledToSuccess" in data.columns, "Missing ledToSuccess column"
        self.is_successful = data["ledToSuccess"].any()
        self._data.drop(columns="ledToSuccess", inplace=True)

    @cached_property
    def valid_pinch_timestamps(self) -> pd.Series:
        """Get the valid pinching time stamps.

        Returns
        -------
        pandas.Series
            The timestamps of the valid pinching moments.

        Raises
        ------
        KeyError
            If `isValid Pinch` is missing from the touch data frame.
        """
        if not self.is_valid_pinch_exists:
            raise KeyError(
                "`isValidPinch` variable is missing from the touch data frame."
            )
        return self._data[self._data["isValidPinch"]].index.to_series()

    @property
    def pinch_begin(self) -> datetime.datetime:
        """Get the start of the pinching.

        Returns
        -------
        datetime.datetime
            The time stamp when the pinching started.
        """
        return self.valid_pinch_timestamps.min()

    @property
    def pinch_end(self) -> datetime.datetime:
        """Get the start of the pinching.

        Returns
        -------
        datetime.datetime
            The time stamp when the pinching started.
        """
        return self.valid_pinch_timestamps.max()

    def set_is_valid_pinch(self, is_valid_pinch: pd.Series):
        """Set is valid pinch variable inside touch data frame.

        Parameters
        ----------
        is_valid_pinch
            A pandas series containing isValidPinch boolean values and time
            stamps as indexes.
        """
        intersected_idx = self._data.index.intersection(is_valid_pinch.index)
        self._data["isValidPinch"] = is_valid_pinch[intersected_idx]

    @property
    def is_valid_pinch_exists(self) -> bool:
        """Check whether is valid pinch variable exists in touch data frame."""
        return "isValidPinch" in self._data.columns


@dataclass
class PinchAttempt(Gesture):
    """A pinching gesture."""

    TOUCH_CLS: ClassVar[Type[Touch]] = PinchTouch

    #: Is valid if it is comprised of two touch interactions
    is_valid: bool

    #: Is successful if the gesture led to the burst of the target
    is_successful: bool

    #: The top finger touch interaction
    top_finger: PinchTouch

    #: The bottom finger touch interaction
    bottom_finger: PinchTouch

    #: The time stamp when the pinching started
    pinch_begin: datetime.datetime

    #: The time stamp when the pinching ended
    pinch_end: datetime.datetime

    @staticmethod
    def add_is_valid_pinch(
        first: PinchTouch,
        second: PinchTouch,
        target_radius: float,
        target_coords: Tuple[float, float],
    ) -> Tuple[PinchTouch, PinchTouch]:
        """Add is valid pinch variable to the given touches.

        Parameters
        ----------
        first
            The first pinch touch.
        second
            The second pinch touch.
        target_radius
            The radius of the pinch target.
        target_coords
            The coordinates of the pinch target.

        Returns
        -------
        Tuple[PinchTouch, PinchTouch]
            The modified given pinch touches.
        """
        first_data = first.get_data()
        second_data = second.get_data()

        # Initialize the isValidPinch output to false
        is_valid_pinch = pd.Series(
            [False] * len(index_ := first_data.index.union(second_data.index)),
            index=index_,
            name="isValidPinch",
        )

        def _inside_target(pos1, pos2) -> bool:
            """Check if two positions are inside the pinch target."""
            return (
                euclidean_distance(pos1, target_coords) < target_radius
                and euclidean_distance(pos2, target_coords) < target_radius
            )

        def _first_positions_valid(
            pos1: Tuple[float, float], pos2: Tuple[float, float]
        ) -> bool:
            """Check if the first contact points with the screen are valid.

            Parameters
            ----------
            pos1
                The first touch position.
            pos2
                The second touch position.

            Returns
            -------
            bool
                ``True`` if the first points of contact are valid.
                ``False`` otherwise.

            Notes
            -----
            A valid initial points of contact varify the following criteria:

                - Distance between the two points is greater than the diameter.
                - Distance between each point and the radius is greater than
                  the radius.
            """
            return (
                euclidean_distance(pos1, pos2) > 2 * target_radius
                and euclidean_distance(pos1, target_coords) > target_radius
                and euclidean_distance(pos2, target_coords) > target_radius
            )

        if _first_positions_valid(first.first_position, second.first_position):
            first_position, second_position = None, None
            # Iterate over the sorted merged indexes between the two fingers.
            for index in is_valid_pinch.index:
                # Obtain coordinates for the timestamp index if existent in
                # first touch
                if index in first_data.index:
                    first_position = tuple(first_data.loc[index][["x", "y"]])
                # Obtain coordinates for the timestamp index if existent in
                # first touch
                if index in second_data.index:
                    second_position = tuple(second_data.loc[index][["x", "y"]])
                # If two positions are existent evaluate whether they are
                # inside pinch target
                if first_position and second_position:
                    is_valid_pinch[index] = _inside_target(
                        first_position, second_position
                    )
        # Set isValidPinch variable in the given touches
        first.set_is_valid_pinch(is_valid_pinch)
        second.set_is_valid_pinch(is_valid_pinch)
        return first, second

    @classmethod
    def gesture_factory(cls, touches: Sequence[Touch], **kwargs) -> "Gesture":
        """Get a pinch attempt gesture from touches."""
        # Check if we have just two touches!
        is_valid = len(touches) == 2
        assert is_valid, "A Pinch attempt should only have two touches"

        # Check if the pinching was successful
        is_successful = any(cast(PinchTouch, t).is_successful for t in touches)

        if not isinstance((context := kwargs.get("context")), Context):
            raise ValueError("Missing context.")

        target_radius = context.get_raw_value("targetRadius")
        target_coords = (
            context.get_raw_value("xTargetBall"),
            context.get_raw_value("yTargetBall"),
        )

        # assign top and bottom filters
        def _first_y(touch: Touch) -> float:
            return touch.positions["y"].iloc[0]

        first, second = cast(Sequence[PinchTouch], touches)

        if _first_y(first) < _first_y(second):
            top_finger = first
            bottom_finger = second
        else:
            top_finger = second
            bottom_finger = first

        # Compute is valid pinch variable if not found
        if not (first.is_valid_pinch_exists and second.is_valid_pinch_exists):
            first, second = cls.add_is_valid_pinch(
                first, second, target_radius, target_coords
            )

        # determine pinch begin and end
        pinch_begin = min(first.pinch_begin, second.pinch_begin)
        pinch_end = max(first.pinch_end, second.pinch_end)

        return cls(
            touches,
            is_valid=is_valid,
            is_successful=is_successful,
            top_finger=top_finger,
            bottom_finger=bottom_finger,
            pinch_begin=pinch_begin,
            pinch_end=pinch_end,
        )

    @classmethod
    def from_data_frame(cls, data: pd.DataFrame, **kwargs) -> List[Gesture]:
        """Create PinchAttempt from a data frame.

        Parameters
        ----------
        data
            A data frame containing touch events.
        kwargs
            Additional key word arguments passed to
            :meth:`~dispel.providers.generic.touch.Gesture.gesture_factory`.

        Returns
        -------
        List[PinchAttempt]
            A sequence of PinchAttempt based on the provided ``data``. The data
            frame is split according to the ``touchPathId`` into separate touch
            interactions. Consecutively overlapping touches are combined as
            PinchAttempt.
        """
        assert (
            "touchPathId" in data.columns
        ), "A pinch attempt need touchPathId information"

        touches = cls._expand_touches(data)

        assert len(touches) > 0, "No touch interaction contained in data"

        gestures = []
        gesture_touches: List[Touch] = []
        for result in sorted(touches, key=lambda x: x.begin):
            if not gesture_touches or gesture_touches[-1].overlaps(result):
                gesture_touches.append(result)
            else:
                if len(gesture_touches) == 2:
                    gestures.append(gesture_touches)
                gesture_touches = [result]

        if len(gesture_touches) == 2:
            gestures.append(gesture_touches)

        return list(map(partial(cls.gesture_factory, **kwargs), gestures))

    @property
    def pinching_duration(self) -> datetime.timedelta:
        """Get the duration of the pinching.

        Returns
        -------
        datetime.timedelta
            Returns the duration of the pinching if the gesture is considered
            a valid pinch based on ``pinch_begin`` and ``pinch_end``. If the
            pinch is not valid according to the app (``isValidPinch`` in the
            ``screen`` data set is ``false``) it returns ``None``.
        """
        return self.pinch_end - self.pinch_begin

    @property
    def first_push_top_fingers(self) -> float:
        """Get the first pressure measurement of the top finger.

        Returns
        -------
        float
            The first pressure reading from the top finger that was non-zero.
        """
        return self.top_finger.initial_pressure

    @property
    def first_push_bottom_fingers(self) -> float:
        """Get the first pressure measurement of the bottom finger.

        Returns
        -------
        float
            The first pressure reading from the bottom finger that was
            non-zero.
        """
        return self.bottom_finger.initial_pressure

    @property
    def double_touch_asynchrony(self) -> datetime.timedelta:
        """Get the double touch asynchrony of the pinch attempt.

        Returns
        -------
        datetime.timedelta
            The time difference between the first and second finger touching
            the screen for a pinch attempt.
        """
        return abs(self.top_finger.begin - self.bottom_finger.begin)


class PinchTarget:
    """A pinch target class.

    This encapsulates pinch targets for each level e.g. `'right-small'`,
    `'right-small-01'` etc.

    Attributes
    ----------
    id: str
        The pinch target identifier e.g. `'right-small-01'`.
    parent_id: str
        The pinch target parent identifier e.g. `'right-small'`.
    hand: HandModality
        The hand used for the pinch.
    size: BubbleSizeModality
        The target bubble size.
    radius: float
        The radius of the pinch target.
    coordinates: Tuple[float, float]
        The coordinates of the pinch target.
    appearance: pandas.Timestamp
        The timestamp at which the target initially appeared.
    attempts: List[PinchAttempt]
        A list of the pinch attempts for the current target.
    """

    def __init__(self):
        self.id: str = field(init=False)
        self.parent_id: str = field(init=False)
        self.hand: HandModality = field(init=False)
        self.size: BubbleSizeModality = field(init=False)
        self.radius: float = field(init=False)
        self.coordinates: Tuple[float, float] = field(init=False)
        self.appearance: datetime = field(init=False)
        self.attempts: List[PinchAttempt] = []

    def __repr__(self):
        return (
            f"<Pinch Target: {self.id}, " f'{plural("attempt", len(self.attempts))}.>'
        )

    @classmethod
    def from_level(cls, level: Level) -> Optional["PinchTarget"]:
        """Initialize pinch target from level.

        Parameters
        ----------
        level
            The level from which the pinch target is to be initialized.

        Returns
        -------
        PinchTarget
            The pinch target.

        Raises
        ------
        ValueError
            If the PinchTarget id doesn't start with a `hand-size` format
        """
        target = cls()
        parent_id = _remove_attempt(id_ := str(level.id))
        try:
            components = parent_id.split("-")
            hand = HandModality.from_variable(components[0])
            size = BubbleSizeModality.from_variable(components[1])
        except Exception as error:
            raise ValueError(
                "PinchTarget id must start with a `hand-size` format."
            ) from error
        target.id = id_
        target.parent_id = parent_id
        target.hand = hand
        target.size = size

        context = level.context
        target.radius = context.get_raw_value("targetRadius")
        target.coordinates = (
            context.get_raw_value("xTargetBall"),
            context.get_raw_value("yTargetBall"),
        )
        target.appearance = level.start

        data = level.get_raw_data_set("screen").data
        if not data.empty:
            attempts: List[PinchAttempt] = []
            try:
                attempts = cast(
                    List[PinchAttempt],
                    PinchAttempt.from_data_frame(data, context=context),
                )
            except AssertionError:
                pass
            for attempt in filter(lambda a: a.is_valid, attempts):
                target.attempts.append(attempt)
        return target

    @property
    def has_attempts(self) -> bool:
        """Get whether the pinch target has attempts."""
        return len(self.attempts) > 0

    @property
    def first_attempt(self) -> PinchAttempt:
        """Get the first pinch attempt."""
        return min(self.attempts, key=lambda a: a.begin)

    def get_attempts_from(self, modality: AttemptOutcomeModality) -> List[PinchAttempt]:
        """Get the list of attempts from success pinche modality."""
        if modality == AttemptOutcomeModality.ALL:
            return self.attempts
        return [a for a in self.attempts if a.is_successful == modality.is_success]

    def contact_coordinates_top_fingers(
        self, outcome: AttemptOutcomeModality = AttemptOutcomeModality.ALL
    ) -> List[Tuple[float, float]]:
        """Get the attempts contact coordinates for top fingers."""
        return list(
            map(lambda a: a.top_finger.first_position, self.get_attempts_from(outcome))
        )

    def contact_coordinates_bottom_fingers(
        self, outcome: AttemptOutcomeModality = AttemptOutcomeModality.ALL
    ) -> List[Tuple[float, float]]:
        """Get the attempts contact coordinates for bottom fingers."""
        return list(
            map(
                lambda a: a.bottom_finger.first_position,
                self.get_attempts_from(outcome),
            )
        )

    def get_contact_distance(self, finger_coordinates: Tuple[float, float]) -> float:
        """Get contact distance given contact coordinates.

        The contact distance is the euclidean distance between the finger
        coordinates and the surface of the pinch target i.e. the distance
        between the finger coordinates and the target center coordinates
        minus the radius of the target.

        Parameters
        ----------
        finger_coordinates
            The coordinates of the finger.

        Returns
        -------
        float
            The contact distance for the given finger coordinates.
        """
        return euclidean_distance(self.coordinates, finger_coordinates) - self.radius

    def contact_distances(
        self,
        finger: FingerModality,
        outcome: AttemptOutcomeModality = AttemptOutcomeModality.ALL,
    ) -> List[float]:
        """Get contact distances for fingers."""
        if finger == FingerModality.TOP_FINGER:
            top_finger_coordinates = self.contact_coordinates_top_fingers(outcome)
            return list(map(self.get_contact_distance, top_finger_coordinates))

        bottom_finger_coordinates = self.contact_coordinates_bottom_fingers(outcome)
        return list(map(self.get_contact_distance, bottom_finger_coordinates))

    @property
    def total_duration(self) -> datetime.timedelta:
        """Get the total duration of the pinch attempts."""
        try:
            return self.attempts[-1].end - self.first_attempt.begin
        except IndexError:
            return datetime.timedelta(0)

    @property
    def reaction_time(self) -> pd.Timedelta:
        """Get the user reaction time."""
        try:
            return self.attempts[0].begin - self.appearance
        except IndexError:
            return pd.Timedelta("nan")

    def first_pushes(
        self,
        finger: FingerModality,
        outcome: AttemptOutcomeModality = AttemptOutcomeModality.ALL,
    ) -> List[float]:
        """Get the first pushes of a pinch target's fingers attempts."""
        if finger == FingerModality.TOP_FINGER:
            return list(
                map(lambda a: a.first_push_top_fingers, self.get_attempts_from(outcome))
            )

        return list(
            map(lambda a: a.first_push_bottom_fingers, self.get_attempts_from(outcome))
        )


def total_number_pinches(target: PinchTarget) -> int:
    """Return total number of pinches.

    A pinch is defined as a potential pinch with at least one touch event
    marked with isValidPinch.

    Parameters
    ----------
    target
        A pinch target object.

    Returns
    -------
    int
        The total number of pinches.
    """
    return len(target.attempts)


def number_successful_pinches(target: PinchTarget) -> int:
    """Return number of successful pinches.

    A pinch is defined as a potential pinch with at least one touch event
    marked with ledToSuccess.

    Parameters
    ----------
    target
        A pinch target object.

    Returns
    -------
    int
        Number of successful pinches
    """
    return sum(attempt.is_successful for attempt in target.attempts)


def success_duration(
    target: PinchTarget, modality: AttemptOutcomeModality = AttemptOutcomeModality.ALL
) -> List[float]:
    """Compute the success duration.

    The success duration i.e. the duration spent before succeeding at
    pinching a bubble.

    Parameters
    ----------
    target
        A pinch target object.
    modality
        Pinching attempt success modality.

    Returns
    -------
    List[float]
        A list regrouping the computed success durations in s.

    """
    return [
        (attempt.pinch_begin - attempt.begin).total_seconds()
        for attempt in filter(
            lambda x: x.pinch_begin is not None, target.get_attempts_from(modality)
        )
    ]


def pinching_duration(
    target: PinchTarget, modality: AttemptOutcomeModality = AttemptOutcomeModality.ALL
) -> List[float]:
    """Compute the pinching duration.

    The pinching duration i.e. the duration spent actually deforming the
    bubble.

    Parameters
    ----------
    target
        A pinch target object.
    modality
        Pinching attempt success modality.

    Returns
    -------
    List[float]
        A list regrouping the computed pinching durations in s.

    """
    return [
        attempt.pinching_duration.total_seconds()
        for attempt in filter(
            lambda x: x.pinching_duration is not None,
            target.get_attempts_from(modality),
        )
    ]


def total_duration(target: PinchTarget) -> float:
    """Compute the total duration of the pinch attempts.

    Parameters
    ----------
    target
        A pinch target object.

    Returns
    -------
    float
        The computed total duration in s.

    """
    return target.total_duration.total_seconds()


def reaction_time(target: PinchTarget) -> float:
    """Compute the reaction time.

    The reaction time of the user between the bubble appearance and the first
    touch event.

    Parameters
    ----------
    target
        A pinch target object.

    Returns
    -------
    float
        The user reaction time in ms.
    """
    return target.reaction_time.total_seconds() * 1e3


def dwell_time(
    target: PinchTarget, attempt: AttemptSelectionModality
) -> Union[float, List[float]]:
    """Compute the pinching attempts' dwell times.

    The dwell time i.e. time spent between the first screen touching and the
    initiation of the movement.

    Parameters
    ----------
    target
        A pinch target object.
    attempt
        Pinching attempt selection modality.

    Returns
    -------
    List[float]
        A list regrouping the computed dwell times in ms.
    """
    if attempt.is_first:
        if target.has_attempts:
            return target.first_attempt.dwell_time.total_seconds() * 1e3
        return np.nan

    return [attempt.dwell_time.total_seconds() * 1e3 for attempt in target.attempts]


def double_touch_asynchrony(
    target: PinchTarget, attempt: AttemptSelectionModality
) -> Union[float, List[float]]:
    """Compute the pinching attempts' double touch asynchrony.

    The double touch asynchrony i.e. time difference between the first and
    second finger touching the screen for all pinch attempts.

    Parameters
    ----------
    target
        A pinch target object.
    attempt
        Pinching attempt selection modality.

    Returns
    -------
    List[float]
        A list regrouping the computed double touch asynchrony in ms.
    """
    if attempt.is_first:
        if target.has_attempts:
            dat = target.first_attempt.double_touch_asynchrony
            return dat.total_seconds() * 1e3
        return np.nan

    return [
        attempt.double_touch_asynchrony.total_seconds() * 1e3
        for attempt in target.attempts
    ]
