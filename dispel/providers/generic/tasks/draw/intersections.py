"""A module dedicated to drawing intersection detections."""
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from bentley_ottmann.planar import segments_intersect
from ground.core import geometries

from dispel.providers.generic.tasks.draw.shapes import get_valid_path
from dispel.signal.core import euclidean_distance, euclidean_norm


class Point(geometries.Point):
    """Wrap Point object in a custom class."""

    @property
    def coordinates(self) -> Tuple[float, float]:
        """Return the coordinates of the ``Point`` object."""
        return self.x, self.y

    def to_numpy(self) -> np.ndarray:
        """Transform point to numpy array."""
        return np.array(self.coordinates)

    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        return Point(x, y)


class Segment(geometries.Segment):
    """Wrap Segment object in a custom class."""

    def __init__(self, start: Point, end: Point):  # pylint: disable=W0231
        self._start, self._end = start, end

    @property
    def segment(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Return the coordinates of the two points forming the segment."""
        return self._start.coordinates, self._end.coordinates

    @property
    def distance(self) -> float:
        """Return the segment length."""
        return euclidean_distance(*self.segment)

    def diff(self) -> Point:
        """Compute the difference between segment points."""
        return self._end - self._start


def _create_segment(row: pd.Series) -> Segment:
    """Create a Segment object for model path."""
    return Segment(Point(row["x"], row["y"]), Point(row["x-1"], row["y-1"]))


def _create_dist(row: pd.Series) -> float:
    """Extract distances from a specific model segment object."""
    return row["seg"].distance


def get_segment(data: pd.DataFrame) -> pd.DataFrame:
    """Get all `Segment` of all data points."""
    new_data = data.copy()
    new_data[["x-1", "y-1"]] = new_data[["x", "y"]].shift(1)
    series = {"seg": _create_segment, "dist": _create_dist}
    for ser, func in series.items():
        new_data[ser] = new_data.apply(func, axis=1)
    new_data["tot_length"] = new_data["dist"].cumsum()
    return new_data[["seg", "dist", "tot_length"]]


def get_ratio(data: pd.DataFrame) -> pd.Series:
    """Add the traveled distance ratio of a segment.

    This get the traveled distance ratio of all user and model segments. This
    ratio corresponds to the 'traveled' distance from zero to the segment over
    the total distance of the drawn/ground truth shape.

    Parameters
    ----------
    data
        A pandas data frame containing at least `dist` and `tot_length` pandas
        Series.

    Returns
    -------
    pandas.Series
        The pandas series corresponding to the `ratio` defined as the total
        length of a specific point from the origin of the draw over the total
        length of the draw.
    """
    return pd.Series(data["tot_length"] / data["dist"].sum(), name="ratio")


def get_intersection_data(
    user: pd.DataFrame, ref: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Transform `{level_id}_user_paths` data to study intersections.

    This transform the `{level_id}_user_paths` data in order to get the
    proper data to capture the number of intersections between the user path
    and the model path.

    Parameters
    ----------
    user
        The `{level_id}_user_paths` data frame.
    ref
        The reference trajectory corresponding to the current shape.

    Returns
    -------
    Tuple[pandas.DataFrame, pandas.DataFrame]
        The proper user data frame to compute tremor-related measures and the
        proper reference data to compute tremor-related measures.
    """
    new_user = user[["x", "y", "tsTouch"]].copy()
    new_ref = ref.copy()
    new_user = new_user.drop_duplicates(
        subset=["x", "y"], keep="last", ignore_index=True
    )

    #: Get the euclidean relative norm of position-related model and user data.
    new_user = new_user.join(euclidean_norm(new_user[["x", "y"]]).rename("norm"))
    new_ref = new_ref.join(euclidean_norm(new_ref[["x", "y"]]).rename("norm"))

    #: Get Segments, segment lengths and traveled distance for each pairs of
    # consecutive points.
    new_user = new_user.join(get_segment(new_user))
    new_ref = new_ref.join(get_segment(new_ref))

    #: Get the traveled distance over total length of a shape ratios for each
    # point of the user and model paths.
    new_ref = new_ref.join(get_ratio(new_ref))
    new_user = new_user.join(get_ratio(new_user))
    columns = ["seg", "norm", "ratio"]
    columns_user = ["tsTouch", "seg", "norm", "ratio"]
    return new_user[columns_user], new_ref[columns]


def _get_ratio_flag(
    user: pd.DataFrame,
    ref: pd.DataFrame,
    index_user,
    index_model,
    threshold: Optional[float] = 0.30,
) -> float:
    """Check whether the intersection occurs in a meaningful area of shapes."""
    return abs(ref["ratio"][index_model] - user["ratio"][index_user]) < threshold


def _get_angle_flag(
    seg_user: Segment, seg_model: Segment, threshold: Optional[float] = 0.1
) -> bool:
    """Check if the segments involved in the intersection are collinear."""
    v = seg_model.diff().to_numpy()
    u = seg_user.diff().to_numpy()
    c = np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)
    angle = np.degrees(np.arccos(np.clip(c, -1, 1)))
    return angle > threshold


def _data_filter_selection(
    user: pd.DataFrame,
    ref: pd.DataFrame,
    user_index: int,
    ratio_filter: Optional[float] = 0.1,
    norm_filter: Optional[float] = 20.0,
) -> pd.DataFrame:
    """Apply filter on the norm and the distance ratio to get selected data."""
    #: Compute the norm of the barycenter of the segment
    norm_bary = (user["norm"][user_index] + user["norm"][user_index - 1]) / 2
    ratio_user = user["ratio"][user_index]

    #: Use the norm of the barycenter of the segment and the traveled
    # distance ratio as filters to select only relevant segments to
    # compare to.
    model_data = ref.loc[
        (ref["norm"] > norm_bary - norm_filter)
        & (ref["norm"] < norm_bary + norm_filter)
        & (
            (ref["ratio"] < ratio_user + ratio_filter)
            & (ref["ratio"] > ratio_user - ratio_filter)
        ),
        "seg",
    ]
    return model_data


def get_intersection_measures(
    user: pd.DataFrame,
    ref: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute the number of intersections between user and model paths.

    Parameters
    ----------
    user
        The `{level_id}_intersection_detection` data frame.
    ref
        The reference trajectory corresponding to the current shape.

    Returns
    -------
    pandas.DataFrame
        A pandas data frame containing tremor measures related to path
        intersection.
    """

    def _check_intersections(model_seg: Segment, user_seg: Segment) -> bool:
        """Check if two given segments intersects."""
        return segments_intersect([user_seg, model_seg])

    intersections: List[datetime] = []

    #: Loop over all segment (index == 0 belongs only to the first segment)
    for id_user in range(1, user["norm"].size - 1):
        seg_user = user["seg"][id_user]

        model_data = _data_filter_selection(user, ref, user_index=id_user)

        #: loop over all the selected model segments to compare to the
        # current user segment.
        checked = model_data.apply(lambda x, y=seg_user: _check_intersections(x, y))
        if checked.any():
            #: Get the index of the intersected model segment
            id_model_seg = checked.loc[checked].index.values
        else:
            continue

        for intersection in id_model_seg:
            #: Check if the intersection is time consistent
            ratio = _get_ratio_flag(user, ref, id_user, intersection)
            #: Check if the user and model segments are
            # non-collinear (approximated to angle < 0.1 degree)
            angle = _get_angle_flag(seg_user, ref["seg"][intersection])
            #: Check if the two previous conditions are respected.
            if angle and ratio:
                #: Store the first intersection.
                if len(intersections) == 0:
                    intersections.append(user["tsTouch"][id_user])
                #: Store the current intersection only if it is
                # consistent with the sampling rate (50Hz ?).
                # (Avoid non consistent intersections to be
                # considered)
                elif len(intersections) != 0 and (
                    (user["tsTouch"][id_user] - intersections[-1]).total_seconds()
                    > 0.02
                ):
                    intersections.append(user["tsTouch"][id_user])

    #: Store the results into a pandas data frame.
    res = pd.DataFrame(columns=["tsRaw", "tsDiff", "cross_per_sec", "freqDiff"])
    res.tsRaw = pd.Series(intersections)
    #: If no crossing, returns res with only 0 as number of intersection per
    # second, np.nan for the rest
    if len(res) == 0:
        res.loc[0, "cross_per_sec"] = 0
        return res
    res["tsDiff"] = (res["tsRaw"].diff()).dt.total_seconds()
    res["freqDiff"] = 1 / res["tsDiff"]
    res["cross_per_sec"] = (
        len(intersections) / (user.tsTouch.max() - user.tsTouch.min()).total_seconds()
    )
    return res


def compute_intersection_analysis(
    user: pd.DataFrame, reference: pd.DataFrame
) -> pd.DataFrame:
    """Compute the tremor-related measures according to intersections.

    First get only the valid user trajectory, then format the data needed, and
    then extract the measures.

    Parameters
    ----------
    user
        A pandas data frame obtained via a
        :class:`dispel.providers.generic.tasks.draw.touch.DrawShape`
    reference
        The reference trajectory corresponding to the current shape.

    Returns
    -------
    pandas.DataFrame
        The proper pandas data frame to compute tremor measures.
    """
    new_data = get_valid_path(user)
    new_user, new_ref = get_intersection_data(new_data, reference)
    return get_intersection_measures(new_user, new_ref)
