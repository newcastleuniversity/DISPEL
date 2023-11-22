r"""
The touch module provides general abstraction classes to handle touch events.

Touch events are represented in :class:`Touch`\ s and :class:`Gesture`\ s that
represent one consecutive interaction (touch down to touch up) and overlapping
interactions, respectively.
"""
import datetime
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import InitVar, dataclass, field
from functools import cached_property, partial
from itertools import combinations
from types import SimpleNamespace
from typing import ClassVar, Dict, List, Optional, Sequence, Tuple, Type, Union, cast

import numpy as np
import pandas as pd
from ground.core.geometries import Multisegment, Point, Segment

from dispel.signal.core import euclidean_norm, index_time_diff

#: List of possible touch actions.
TOUCH_ACTIONS = SimpleNamespace(up="up", down="down", move="move")

#: The valid column names to use the `touch` module.'
TOUCH_COLUMNS = [
    "xPosition",
    "yPosition",
    "touchAction",
    "tsTouch",
    "touchPathId",
    "pressure",
]


@dataclass
class Pointer:
    """Touch pointer reference."""

    #: The index of the pointer
    index: int = 0


class UnknownPointer(Pointer):
    """Unknown pointer reference.

    This pointer can be used if the pointer is unknown.
    """

    def __eq__(self, other):
        return False


@dataclass
class Touch:
    """A consecutive interaction with the screen.

    The :class:`Touch` represents the touch events of one consecutive
    interaction with the screen from the first ``down`` action until the ``up``
    action.
    """

    data: InitVar[pd.DataFrame]

    _data: pd.DataFrame = field(init=False, repr=False, default=None)

    #: The pointer information of the touch interaction
    pointer: Pointer = UnknownPointer()

    def _count_action(self, action: str) -> int:
        return len(self._data[self._data["touchAction"] == action])

    def __post_init__(self, data: pd.DataFrame):
        # check min length of data frame
        assert len(data) > 1, "At least two touch events are expected"

        # rename x and y positions for uniform data format.
        data = data.rename(columns={"xPosition": "x", "yPosition": "y"})

        # check required columns are present
        required_columns = {"tsTouch", "touchAction", "x", "y"}
        assert required_columns.issubset(
            data.columns
        ), "Not all required columns in data set"

        # check that timestamps are timestamps
        assert pd.api.types.is_datetime64_dtype(
            data["tsTouch"]
        ), "tsTouch needs to be date time data type"

        # check that we have only positive coordinates
        assert (data["x"] >= 0).all(), "x needs to be positive"
        assert (data["y"] >= 0).all(), "y needs to be positive"

        # transform and sort data
        self._data = data.set_index("tsTouch").sort_index()

        if not self._data.index.is_unique:
            warnings.warn("Touch event index is not unique", UserWarning)
            self._data = self._data[~self._data.index.duplicated(keep="last")]

        # ensure to have only one down and up action at most at the right
        # position, i.e. down, [move], up
        down_actions = self._count_action("down")
        assert down_actions < 2, 'Only one or no "down" action expected'
        if down_actions:
            assert (
                self._data.iloc[0]["touchAction"] == "down"
            ), 'Expect "down" action to be first event'

        up_actions = self._count_action("up")
        assert up_actions < 2, 'Only one or no "up" action expected'
        if up_actions:
            assert (
                self._data.iloc[-1]["touchAction"] == "up"
            ), 'Expect "up" action to be last event'

    def __len__(self):
        return len(self._data)

    def get_data(self) -> pd.DataFrame:
        """Get a copy of the touch data frame."""
        return self._data.copy()

    @cached_property
    def begin(self) -> datetime.datetime:
        """Get the start of the touch.

        Returns
        -------
        datetime.datetime
            The date time of the first touch event
        """
        return self._data.index[0]

    @cached_property
    def end(self) -> datetime.datetime:
        """Get the end of the touch.

        Returns
        -------
        datetime.datetime
            The date time of the last touch event
        """
        return self._data.index[-1]

    @cached_property
    def duration(self) -> datetime.timedelta:
        """Get the duration of the touch.

        Returns
        -------
        datetime.timedelta
            The duration of the touch interaction from its start to its end.
        """
        return self.end - self.begin

    @cached_property
    def time_deltas(self) -> pd.Series:
        """Get the time differences between consecutive touch events.

        Returns
        -------
        pandas.Series
            Returns a series of time differences in milliseconds between
            consecutive touch events with the index based on the touch event
            time stamp.
        """
        return index_time_diff(self._data).dropna() * 1e3

    @property
    def is_incomplete(self) -> bool:
        """Is the interaction incomplete.

        Returns
        -------
        bool
            Returns ``True`` if either the ``down`` or ``up`` action is not
            observed. Otherwise, ``False``.
        """
        return self._count_action("down") != 1 or self._count_action("up") != 1

    def overlaps(self, other: "Touch") -> bool:
        """Check if the touch has overlapping time points.

        Parameters
        ----------
        other
            The other touch interaction to test for overlapping.

        Returns
        -------
        bool
            Returns ``True`` if the ``start`` or ``end`` of `other` fall
            within the start or end of this touch interaction (boundaries
            included). Otherwise, ``False``.
        """

        def _contained(timestamp: datetime.datetime) -> bool:
            return self.begin <= timestamp <= self.end

        return _contained(other.begin) or _contained(other.end)

    @cached_property
    def positions(self) -> pd.DataFrame:
        """Get the x and y coordinates of the touch interactions.

        Returns
        -------
        pandas.DataFrame
            A data frame with two columns ``XPosition`` and ``yPosition`` and
            the time points as index.
        """
        return self._data[["x", "y"]].copy()

    @cached_property
    def first_position(self) -> Tuple[float, float]:
        """Get the first position of the touch interaction.

        Returns
        -------
        Tuple[float, float]
            The x and y coordinate of the first touch interaction with the
            screen.
        """
        first = self.positions.iloc[0]
        return first["x"], first["y"]

    def to_segments(self) -> Multisegment:
        """Get multi-segment representation of the touch interaction."""

        def _to_point(row: pd.Series) -> Point:
            return Point(row["x"], row["y"])

        points = self.positions.apply(_to_point, axis=1).tolist()
        segments = [Segment(start, end) for start, end in zip(points[:-1], points[1:])]

        return Multisegment(segments)

    @cached_property
    def displacements(self) -> pd.DataFrame:
        """Get the displacements per touch event.

        Returns
        -------
        pandas.DataFrame
            Returns the changes in position in x and y direction at each touch
            event.
        """
        return self.positions.diff().dropna()

    @cached_property
    def movement_begin(self) -> datetime.datetime:
        """Get the start of the movement for the touch.

        Returns
        -------
        datetime.datetime
            The date time of the first recorded movement of the touch.
        """
        return self.displacements.ne(0).idxmax().min()

    @cached_property
    def length(self) -> float:
        """Get the length of the interaction.

        Returns
        -------
        float
            The length of the interaction based on the sum of euclidean norm
            vector of displacements.
        """
        return euclidean_norm(self.displacements).sum()

    @cached_property
    def velocity(self) -> pd.DataFrame:
        """Get the interaction velocity.

        Returns
        -------
        pandas.DataFrame
            The change in position over time for the respective axis.
        """
        return (self.displacements.T / self.time_deltas).T

    @cached_property
    def speed(self) -> pd.Series:
        """Get the pointer speed of the interaction.

        Returns
        -------
        pandas.Series
            The change in position over time.
        """
        return euclidean_norm(self.displacements) / self.time_deltas

    @cached_property
    def acceleration(self) -> pd.Series:
        """Get the pointer acceleration during the interaction.

        Returns
        -------
        pandas.Series
            The change of velocity over time.
        """
        return (self.speed.diff() / self.time_deltas).dropna()

    @cached_property
    def jerk(self) -> pd.Series:
        """Get the pointer jerk during the interaction.

        Returns
        -------
        pandas.Series
            The change of acceleration over time.
        """
        return (self.acceleration.diff() / self.time_deltas).dropna()

    @cached_property
    def mean_squared_jerk(self):
        """Get the average squared jerk.

        Returns
        -------
        float
            The average squared jerk.
        """
        return np.mean(self.jerk**2)

    @property
    def has_pressure(self) -> bool:
        """Has the interaction pressure information.

        Returns
        -------
        bool
            ``True`` if pressure information is available. Otherwise, ``False``
        """
        return "pressure" in self._data

    @cached_property
    def pressure(self) -> pd.Series:
        """Get the pressure information associated with the touch.

        Returns
        -------
        pandas.Series
            The pressure values excluding ``0``-values at the very first
            reading.
        """
        assert self.has_pressure, "Touch has no pressure information"

        # ignore pressure that is zero and at the first position
        if self._data["pressure"].iloc[0] == 0:
            return self._data["pressure"].iloc[1:].copy()

        return self._data["pressure"].copy()

    @property
    def initial_pressure(self) -> float:
        """Get the initially exerted pressure of the pointer.

        Returns
        -------
        float
            The pressure exerted on the screen upon initiation of the
            interaction. If the first reading was ``0`` the subsequent one will
            be returned.
        """
        pressure = self.pressure

        assert len(pressure) > 0, "Not enough pressure readings"
        return pressure.iloc[0]

    @cached_property
    def pressure_change(self) -> pd.Series:
        """Get the pressure change.

        Returns
        -------
        pandas.Series
            The first derivative of the pressure information.
        """
        return (self.pressure.diff() / self.time_deltas).dropna()

    @cached_property
    def pressure_acceleration(self) -> pd.Series:
        """Get the pressure acceleration.

        Returns
        -------
        pandas.Series
            The second derivative of the pressure information.
        """
        return (self.pressure_change.diff() / self.time_deltas).dropna()

    @cached_property
    def pressure_jerk(self) -> pd.Series:
        """Get the pressure jerk.

        Returns
        -------
        pandas.Series
            The change of pressure acceleration over time.
        """
        return (self.pressure_acceleration.diff() / self.time_deltas).dropna()

    @cached_property
    def mean_squared_pressure_jerk(self):
        """Get the average squared jerk of the applied pressure.

        Returns
        -------
        float
            The average squared jerk of the applied pressure.
        """
        return (self.pressure_jerk**2).mean()

    @cached_property
    def has_major_radius(self):
        """Has major radius measurements."""
        return "majorRadius" in self._data

    @cached_property
    def major_radius(self) -> pd.Series:
        """Get the major radius of the touch interaction."""
        assert self.has_major_radius, "Touch has no major radius"
        return self._data["majorRadius"]

    @cached_property
    def major_radius_tolerance(self) -> float:
        """Get the major radius tolerance property."""
        return self._data["majorRadiusTolerance"].iloc[0]


@dataclass
class Gesture:
    """A collection of overlapping :class:`Touch` interactions."""

    #: The touch interactions comprising the gesture
    touches: Sequence[Touch]

    #: The class used to create touches in :func:`touch_factory`.
    TOUCH_CLS: ClassVar[Type[Touch]] = Touch

    def __post_init__(self):
        assert len(self.touches) > 0, "Gesture has to have one or more touches"

        assert all(
            isinstance(t, self.TOUCH_CLS) for t in self.touches
        ), f"All touches need to be an instance of {self.TOUCH_CLS.__name__}"

        if len(self.touches) > 1:
            # ensure pointer are known
            assert all(
                not isinstance(t, UnknownPointer) for t in self.touches
            ), "Multi-touch gestures cannot have unknown pointers"

            ids = [t.pointer.index for t in self.touches]
            assert len(ids) == len(set(ids)), "Pointer indices need to be unique"

            overlapped = defaultdict(bool)
            for a, b in combinations(self.touches, 2):
                overlaps = a.overlaps(b)
                overlapped[a.pointer.index] |= overlaps
                overlapped[b.pointer.index] |= overlaps

            assert all(
                overlapped.values()
            ), "Each touch needs to overlap with at least one other one"

    @cached_property
    def first_touch(self) -> Touch:
        """Get the first touch interaction."""
        return min(self.touches, key=lambda t: t.begin)

    @cached_property
    def last_touch(self) -> Touch:
        """Get the last touch interaction."""
        return max(self.touches, key=lambda t: t.end)

    @cached_property
    def begin(self) -> datetime.datetime:
        """Get the start date time of the gesture."""
        return self.first_touch.begin

    @cached_property
    def end(self) -> datetime.datetime:
        """Get the end date time of the gesture."""
        return self.last_touch.end

    @cached_property
    def duration(self) -> datetime.timedelta:
        """Get the duration of the gesture."""
        return self.end - self.begin

    @cached_property
    def first_movement(self) -> Touch:
        """Get the first moved touch interaction."""
        return min(self.touches, key=lambda t: t.movement_begin)

    @cached_property
    def movement_begin(self) -> datetime.datetime:
        """Get the start of the movement for the gesture.

        Returns
        -------
        datetime.datetime
            The date time of the first recorded movement of the gesture.
        """
        return self.first_movement.movement_begin

    @cached_property
    def dwell_time(self) -> datetime.timedelta:
        """Get the dwell time of the gesture.

        Returns
        -------
        datetime.timedelta
            The time spent between the first touch on the screen and the
            first movement made after that.
        """
        return self.movement_begin - self.begin

    @classmethod
    def touch_factory(cls, touch_data: pd.DataFrame, pointer: Pointer) -> Touch:
        """Get a touch instance for touch data frame.

        This method can be overwritten by inheriting classes to customize how
        touches are constructed.

        Parameters
        ----------
        touch_data
            The touch events for one consecutive touch interaction from one
            pointer.
        pointer
            The pointer of the touch events

        Returns
        -------
        Touch
            A touch interaction of a single pointer
        """
        return cls.TOUCH_CLS(touch_data, pointer)

    @classmethod
    def gesture_factory(cls, touches: Sequence[Touch], **kwargs) -> "Gesture":
        """Get a gesture instance for a list of touches.

        This method can be overwritten by inheriting classes to customize how
        gestures are constructed.

        Parameters
        ----------
        touches
            The results from the touch factory. See :meth:`touch_factory` for
            details.
        kwargs
            Additional key word arguments for overridden methods to eventually
            use.

        Returns
        -------
        Gesture
            The gesture representing the passed :class:`Touch` objects in
            ``touches``.
        """
        # pylint: disable=unused-argument
        return cls(touches)

    @classmethod
    def _expand_touches(cls, data: pd.DataFrame) -> List[Touch]:
        touches = []
        for pointer_id, pointer_data in data.groupby("touchPathId"):
            touches.append(
                cls.touch_factory(pointer_data, Pointer(cast(int, pointer_id)))
            )

        return touches

    @classmethod
    def from_data_frame(cls, data: pd.DataFrame, **kwargs) -> List["Gesture"]:
        """Create gestures from a data frame.

        Parameters
        ----------
        data
            A data frame containing touch events.
        kwargs
            Additional key word arguments passed to
            :meth:`Gesture.gesture_factory`.

        Returns
        -------
        List[Gesture]
            A sequence of gestures based on the provided ``data``. The data
            frame is split according to the ``touchPathId`` into separate touch
            interactions. Consecutively overlapping touches are combined as
            gestures.
        """
        if "touchPathId" in data.columns:
            touches = cls._expand_touches(data)
        else:
            touches = [cls.touch_factory(data, UnknownPointer())]

        assert len(touches) > 0, "No touch interaction contained in data"

        gestures = []
        gesture_touches: List[Touch] = []
        for result in sorted(touches, key=lambda x: x.begin):
            if not gesture_touches or gesture_touches[-1].overlaps(result):
                gesture_touches.append(result)
            else:
                gestures.append(gesture_touches)
                gesture_touches = [result]

        gestures.append(gesture_touches)

        return list(map(partial(cls.gesture_factory, **kwargs), gestures))


def split_touches(
    raw_data: pd.DataFrame, begin: int, end: Optional[int]
) -> pd.DataFrame:
    """Split touch events.

    This function reassign the touch path ids of raw touch events in a data
    frame by making sure that the path ids are unique and given in ascending
    order. One can also filter the input sequence by specifying the start
    and end timestamp.


    Parameters
    ----------
    raw_data
        Dataframe of the raw touch events
    begin
        Raw timestamp of the beginning of the sequence
    end
        An optional raw timestamp of the end of the sequence

    Returns
    -------
    Dict[str, List]
        Touch data for the level
    """
    new_data: Dict[str, List] = {}
    timestamp = raw_data["tsTouch"]
    for key in raw_data:
        assert key in TOUCH_COLUMNS, f"The column {key} is not valid"
        new_data[key] = []
        touch_path_ids = set()
        for i, time in enumerate(timestamp):
            if time >= begin:
                if end is not None and time <= end:
                    if raw_data["touchAction"][i] == "down":
                        touch_path_ids.add(raw_data["touchPathId"][i])
                    if raw_data["touchPathId"][i] in touch_path_ids:
                        new_data[key].append(raw_data[key][i])
                elif raw_data["touchPathId"][i] in touch_path_ids:
                    if raw_data["touchAction"][i] == "down":
                        touch_path_ids.remove(raw_data["touchPathId"][i])
                    else:
                        new_data[key].append(raw_data[key][i])
                else:
                    break
    new_data = reassign_touch_path_id(new_data)

    return pd.DataFrame(new_data)


def reassign_touch_path_id(
    data: Union[pd.DataFrame, Dict[str, List]]
) -> Dict[str, List]:
    """Reassign touch path ids.

    Touch path ids are sometimes mixed when there are no fingers on the screen.
    This function makes sure consecutive touches are assigned ascending new
    ids.

    Parameters
    ----------
    data
        Dictionary of raw touch data

    Returns
    -------
    Dict[str, List]
        Touch data for the level
    """
    alt_ids: Dict[int, int] = {}
    current_max = 0
    new_data = deepcopy(data)
    timestamp = new_data["tsTouch"]
    for i in range(len(timestamp)):
        touch_path_id = new_data["touchPathId"][i]
        if new_data["touchAction"][i] == "down":
            if touch_path_id in alt_ids:
                current_max = max(alt_ids) + 1
                alt_ids[touch_path_id] = current_max
                alt_ids[current_max] = current_max
            else:
                alt_ids[touch_path_id] = touch_path_id
                current_max = max(current_max, touch_path_id)
        new_data["touchPathId"][i] = alt_ids[touch_path_id]
    return new_data
