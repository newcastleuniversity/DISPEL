"""Drawing test related functionality.

This module contains functionality to wrap all touches from a *Drawing* test
(DRAW) into a class `DrawShape`.
"""
from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Optional, Tuple, cast

import numpy as np
import pandas as pd

from dispel.data.core import Reading
from dispel.data.levels import Level
from dispel.data.validators import RangeValidator, ValidationException
from dispel.providers.generic.tasks.draw.intersections import (
    get_intersection_data,
    get_intersection_measures,
)
from dispel.providers.generic.tasks.draw.shapes import (
    extract_distance_axis_sc,
    extract_distance_axis_scc,
    get_reference_path,
    get_user_path,
    get_valid_up_sampled_path,
    remove_overshoots,
    remove_reference_head,
    sc_corners_max_dist,
    scc_corners_max_dist,
)
from dispel.providers.generic.tasks.draw.tremor import get_deceleration_profile_data
from dispel.providers.generic.touch import Pointer, Touch, UnknownPointer
from dispel.signal.dtw import get_minimal_matches


@dataclass
class DrawTouch(Touch):
    """A draw touch interaction."""

    #: Is valid if the touch stated in the valid area.
    is_valid: bool = field(init=False)

    def __post_init__(self, data: pd.DataFrame):
        super().__post_init__(data)

        # Get whether the trajectory is valid or not.
        assert "isValidArea" in data.columns, "Missing isValidArea column"
        self.is_valid = data["isValidArea"].all()

    @property
    def valid_up_sampled_path(self) -> pd.DataFrame:
        """Provide an up sampled trajectory from a DrawTouch."""
        return get_valid_up_sampled_path(self.get_data())


class DrawShape:
    """To encapsulate the user's interaction during a specific attempt.

    The user's data are segmented into `DrawTouch`.

    Attributes
    ----------
    id: str
        The level id corresponding to the drawn shape.
    touches: List[DrawTouch]
        A list of `DrawTouch`
    """

    def __init__(self):
        self.id: str = field(init=False)
        self.reference: pd.DataFrame = field(init=False)
        self.touches: List[DrawTouch] = []

    @classmethod
    def from_level(cls, level: Level, reading: Reading) -> Optional["DrawShape"]:
        """Initialize a DrawShape from level.

        Parameters
        ----------
        level
            The level from which the drawn shape is to be initialized.
        reading
            The reading corresponding to the evaluation.

        Returns
        -------
        DrawShape
            The drawn shape.
        """

        def _expand_touches(data: pd.DataFrame):
            touches = []
            for pointer_id, pointer_data in data.groupby("touchPathId"):
                if len(pointer_data) > 1:
                    touches.append(
                        DrawTouch(pointer_data, Pointer(cast(int, pointer_id)))
                    )
            return touches

        shape = cls()
        shape.id = str(level.id)
        assert reading.device is not None, "Device cannot be null."
        assert reading.device.screen is not None, "Screen cannot be null."
        height = reading.device.screen.height_dp_pt
        assert height is not None, "Height cannot be null."
        shape.reference = get_reference_path(level, height)
        screen = level.get_raw_data_set("screen").data.copy()

        formatted_data = get_user_path(screen, shape.get_reference, height)

        # Segment the trajectory in touchPathIds
        formatted_data["touchPathId"] = np.nan
        down_id = formatted_data[formatted_data["touchAction"] == "down"].index

        # If the first touch action isn't a down, correct it
        if len(down_id) > 0 and down_id[0] != 0:
            down_id = down_id.insert(0, 0)

        for counter, idx in enumerate(down_id):
            formatted_data.loc[idx, "touchPathId"] = counter
        formatted_data["touchPathId"] = (
            formatted_data["touchPathId"].ffill().astype(int)
        )

        shape.touches = _expand_touches(formatted_data)
        return shape

    @property
    def get_reference(self) -> pd.DataFrame:
        """Get the reference trajectory."""
        return self.reference.copy()

    @property
    def touches_count(self) -> int:
        """Get whether the drawn shape has attempts."""
        return len(self.touches)

    @property
    def has_touch(self) -> bool:
        """Get whether the drawn shape has attempts."""
        return self.touches_count > 0

    @cached_property
    def all_data(self) -> pd.DataFrame:
        """Get all data corresponding to the drawn shape."""
        touches = map(lambda t: t.get_data(), self.touches)
        return pd.concat(touches).sort_index()

    @property
    def valid_data(self) -> pd.DataFrame:
        """Get only valid data corresponding to the drawn shape."""
        data = self.all_data
        return data[data["isValidArea"]]

    @property
    def has_invalid_data(self) -> bool:
        """Get whether the drawn shape has invalid data."""
        valid_gestures = sum(gesture.is_valid for gesture in self.touches)
        return valid_gestures != self.touches_count

    @cached_property
    def aggregate_valid_touches(self) -> DrawTouch:
        r"""Aggregate only valid :class:`DrawTouch`\s."""
        data = self.valid_data.reset_index()
        data["touchAction"] = "move"
        data.loc[0, "touchAction"] = "down"
        return DrawTouch(data, UnknownPointer())

    @cached_property
    def aggregate_all_touches(self) -> DrawTouch:
        r"""Aggregate all :class:`DrawTouch`\s."""
        data = self.all_data.reset_index()
        data["touchAction"] = "move"
        data.loc[0, "touchAction"] = "down"
        return DrawTouch(data, UnknownPointer())

    @cached_property
    def get_valid_attributions(self) -> Tuple[float, pd.DataFrame]:
        """Get DTW attributions on raw and valid user trajectory."""
        touch = self.aggregate_valid_touches
        return get_minimal_matches(touch.positions, self.get_reference)

    @property
    def valid_coupling_measure(self) -> float:
        """Get the coupling measure from valid attributions."""
        return self.get_valid_attributions[0]

    @property
    def valid_matches(self) -> pd.DataFrame:
        """Get the minimum matches from valid attributions."""
        return self.get_valid_attributions[1]

    @cached_property
    def up_sampled_valid_data(self) -> pd.DataFrame:
        """Get DTW attributions on up sampled and valid user trajectory."""
        return self.aggregate_valid_touches.valid_up_sampled_path

    @cached_property
    def get_up_sampled_valid_attributions(self) -> Tuple[float, pd.DataFrame]:
        """Get DTW attributions on up sampled and valid user trajectory."""
        up_sampled_data = self.aggregate_valid_touches.valid_up_sampled_path
        return get_minimal_matches(up_sampled_data[["x", "y"]], self.get_reference)

    @property
    def up_sampled_valid_coupling_measure(self) -> float:
        """Get the coupling measure from up sampled and valid attributions.

        Returns
        -------
        float
            The coupling measure specific to the valid, up sampled user
            trajectory without overshoot.
        """
        return self.get_up_sampled_valid_attributions[0]

    @property
    def up_sampled_valid_matches(self) -> pd.DataFrame:
        """Get the minimum matches from up sampled and valid attributions.

        Returns
        -------
        pandas.DataFrame
            The The different point attributed to a valid up sampled user
            specific to the valid, up sampled user trajectory without
            overshoot.
        """
        return self.get_up_sampled_valid_attributions[1]

    @cached_property
    def up_sampled_data_without_overshoot(self) -> pd.DataFrame:
        """Get valid user data without overshoot."""
        return remove_overshoots(
            self.aggregate_valid_touches.valid_up_sampled_path, self.get_reference
        ).reset_index(drop=True)

    @cached_property
    def reference_without_head(self) -> pd.DataFrame:
        """Get the reference trajectory without the head.

        The new reference has the closest point of the first user point without
        early starting points as its head.

        Returns
        -------
        pandas.DataFrame
            The new reference trajectory fitting perfectly with the user
            trajectory without early starting points and final overshoot.
        """
        return remove_reference_head(
            self.up_sampled_data_without_overshoot, self.get_reference
        )

    @cached_property
    def get_up_sampled_valid_no_overshoot_attributions(
        self,
    ) -> Tuple[float, pd.DataFrame]:
        """Get DTW attributions on up sampled and valid user trajectory.

        The DTW attribution is computed between the user trajectory without
        both early starting points and final overshoot, and second, the
        reference trajectory without an irrelevant head. As a reminder, an
        irrelevant head is composed of points anterior to the closest point of
        the user head without early starting points.

        Returns
        -------
        Tuple[float, pandas.DataFrame]
            A tuple containing the coupling measure and the pandas data frame
            where lie the user/reference attributions.
        """
        up_sampled_data = self.up_sampled_data_without_overshoot
        return get_minimal_matches(
            up_sampled_data[["x", "y"]], self.reference_without_head
        )

    @property
    def up_sampled_valid_no_overshoot_coupling_measure(self) -> float:
        """Get the coupling measure from up sampled and valid attributions.

        Specific for data without overshoot.

        Returns
        -------
        float
            The coupling measure specific to the valid, up sampled user
            trajectory without overshoot.
        """
        return self.get_up_sampled_valid_no_overshoot_attributions[0]

    @property
    def up_sampled_valid_no_overshoot_matches(self) -> pd.DataFrame:
        """Get the minimum matches from up sampled and valid attributions.

        Specific for data without overshoot.

        Returns
        -------
        pandas.DataFrame
            The matches specific to the valid, up sampled user
            trajectory without overshoot.
        """
        return self.get_up_sampled_valid_no_overshoot_attributions[1]

    @cached_property
    def intersection_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get formatted data for intersection detection."""
        data = self.valid_data
        return get_intersection_data(data.reset_index(), self.get_reference)

    @cached_property
    def intersection_measures(self) -> pd.DataFrame:
        """Get intersection detection measures."""
        user, ref = self.intersection_data
        return get_intersection_measures(user, ref)

    @cached_property
    def deceleration_data(self) -> pd.DataFrame:
        """Get data to compute Intentional Tremors on specific segment."""
        return get_deceleration_profile_data(
            self.valid_data.reset_index(), self.get_reference, level_id=str(self.id)
        )

    @cached_property
    def is_square_like_shape(self) -> bool:
        """Whether the shape is a square-like shape or not."""
        return self.id.startswith("square")

    @cached_property
    def corners_max_dist(self) -> Tuple[float, float, float]:
        """Get the 3 maximum distances from corner zones."""
        user = self.up_sampled_data_without_overshoot
        ref = self.reference_without_head
        attrib = self.up_sampled_valid_no_overshoot_matches
        if self.id.startswith("square_counter"):
            return scc_corners_max_dist(user, ref, attrib)
        if self.id.startswith("square_clock"):
            return sc_corners_max_dist(user, ref, attrib)
        return np.nan, np.nan, np.nan

    @cached_property
    def axis_overshoots(self) -> Tuple[float, float, float]:
        """Get the 3 distances from perpendicular axes."""
        ref = self.reference_without_head
        data = self.up_sampled_data_without_overshoot
        if self.id.startswith("square_counter"):
            return extract_distance_axis_scc(data, ref)
        if self.id.startswith("square_clock"):
            return extract_distance_axis_sc(data, ref)
        return np.nan, np.nan, np.nan

    @cached_property
    def distance_ratio(self) -> float:
        """Compute distance ratio between user and expected trajectories.

        The distance ratio is defined as the quotient of the valid up sampled
        user path length divided by the expected path length.

        Returns
        -------
        float
            The distance ratio between the valid up sampled user path length
            and the expected path length.
        """
        data = self.up_sampled_valid_data
        ref = self.get_reference
        return (
            data.diff()
            .apply(lambda s: np.sqrt(pow(s.x, 2) + pow(s.y, 2)), axis=1)
            .sum()
            / ref.diff()
            .apply(lambda s: np.sqrt(pow(s.x, 2) + pow(s.y, 2)), axis=1)
            .sum()
        )

    def check_dist_thresh(self, validator: RangeValidator) -> bool:
        """Check if distance ratio lies within a RangeValidator threshold."""
        try:
            validator(self.distance_ratio)
        except ValidationException:
            return False
        return True
