"""All generic transformation steps related to the placement."""
from typing import List

import numpy as np
import pandas as pd

from dispel.data.raw import RawDataValueDefinition
from dispel.processing.transform import TransformStep

THRES_BELT_MIN_ABS_MEAN_GRAVITY_X_1 = np.cos(np.radians(53))
"""belt threshold 1: [0, 53] deg X axes max."""
THRES_PANT_MIN_ABS_MEAN_GRAVITY_Y_1 = np.cos(np.radians(45))
"""pant threshold 1: [0, 45] deg Y axes max."""
THRES_TABL_MIN_ABS_MEAN_GRAVITY_Z_1 = np.cos(np.radians(18))
"""table threshold 1: [0, 18] deg Z axis max (to be combined with low std)."""
THRES_TABL_MIN_ABS_MEAN_GRAVITY_Z_2 = np.cos(np.radians(8))
"""table threshold 2: [0, 8] deg Z axes max."""
THRES_HAND_MAX_MEAN_GRAVITY_Z_1 = -np.cos(np.radians(50))
"""handheld threshold 1: [8, 50] (or [18, 50]) deg Z axis max."""
THRES_TABL_MAX_STD_ACC_NORM_1 = 0.01
"""table threshold 1: standard deviation of acc norm."""

ROLLING_WINDOW_SIZE = "4s"
"""This is the size of the sliding window, denoting the duration between begin
 and end of sliding window."""

ROLLING_STEP_SIZE = "100ms"
"""This is the step of the sliding window, denoting the size of the "sliding"
 action, which is the length of sequence you move between each window."""


PLACEMENT_DEFINITIONS: List[RawDataValueDefinition] = [
    RawDataValueDefinition(
        id_="start_time",
        name="start time",
        description="A series of the start time for each merged time window.",
        data_type="datetime64",
    ),
    RawDataValueDefinition(
        id_="end_time",
        name="end time",
        description="A series of the end time for each merged time window.",
        data_type="datetime64",
    ),
    RawDataValueDefinition(
        id_="duration",
        name="duration",
        description="A series of the duration for each merged time window.",
        data_type="float64",
    ),
    RawDataValueDefinition(
        id_="placement",
        name="detected placement",
        description="A string time series indicating the predicted"
        "placement for each merged time window.",
        data_type="str",
    ),
]


def placement_classification_one_window(window_measures: pd.Series) -> str:
    """Classify placement for a window of measure data.

    Parameters
    ----------
    window_measures
        A pd.Series containing the measure for a specific time

    Returns
    -------
    str
        The label of the predicted class
    """
    if abs(window_measures.loc["mean_gravityX"]) > THRES_BELT_MIN_ABS_MEAN_GRAVITY_X_1:
        # if phone placed roughly landscape -> belt
        res = "belt"
    elif (
        abs(window_measures.loc["mean_gravityY"]) > THRES_PANT_MIN_ABS_MEAN_GRAVITY_Y_1
    ):
        # if phone placed landscape -> pants
        res = "pants"
    elif (
        abs(window_measures.loc["mean_gravityZ"]) > THRES_TABL_MIN_ABS_MEAN_GRAVITY_Z_1
    ) & (window_measures.loc["std_norm"] < THRES_TABL_MAX_STD_ACC_NORM_1):
        # if phone placed face up/down and not moving -> table
        res = "table"
    elif (
        abs(window_measures.loc["mean_gravityZ"]) > THRES_TABL_MIN_ABS_MEAN_GRAVITY_Z_2
    ):
        # if phone placed precisely face up/down -> table
        res = "table"
    elif window_measures.loc["mean_gravityZ"] < THRES_HAND_MAX_MEAN_GRAVITY_Z_1:
        # if phone placed face up with a tilt -> handheld
        res = "handheld"
    else:
        # in all other cases -> else
        res = "else"
    return res


def placement_classification(
    data: pd.DataFrame,
    rolling_window_size: str = ROLLING_WINDOW_SIZE,
    rolling_step_size: str = ROLLING_STEP_SIZE,
) -> pd.DataFrame:
    """Extract measures and predict labels given the gravity and norm data.

    Parameters
    ----------
    data
        A pd.DataFrame containing the gravity and accelerometer norm data
    rolling_window_size
        A string containing the size of the sliding window
        (duration between begin and end of sliding window)
    rolling_step_size
        A string containing the step of the sliding window
        (duration between centers of consecutive windows)

    Returns
    -------
    pandas.DataFrame
        A dataframe containing the extracted measures and label per window

    """
    if "ts" in data.columns:
        data = data.set_index("ts")

    # extract placement measures
    mean = (
        data.loc[:, ["gravityX", "gravityY", "gravityZ"]]
        .rolling(window=rolling_window_size, center=True)
        .mean()
    )
    std = data.loc[:, ["norm"]].rolling(window=rolling_window_size, center=True).std()
    # rename columns
    mean.columns = "mean_" + mean.columns
    std.columns = "std_" + std.columns

    measures = pd.concat([mean, std], axis=1)
    # downsampling here to introduce step size in the sliding window
    # The sliding window computes a moving mean and moving standard deviation
    # for each sample. We need to have a single measure value for each step
    # in the overlapping sliding window process so we take the first.
    measures = measures.resample(rolling_step_size).first()
    predictions = measures.apply(placement_classification_one_window, axis=1)

    measures_predictions = measures.copy()
    measures_predictions["placement"] = predictions

    return measures_predictions


def merge_adjacent_annotations_rolling(
    measures_predictions: pd.DataFrame, rolling_step_size: str = ROLLING_STEP_SIZE
) -> pd.DataFrame:
    """Merge adjacent annotations into segments of continuous labels.

    Parameters
    ----------
    measures_predictions
        A pd.DataFrame containing the measures and predicted class
    rolling_step_size
        A str indicating the size of the sliding window

    Returns
    -------
    pandas.DataFrame
        contains start time, end time and label of each continuous segment
    """
    annotations = measures_predictions[["placement"]].copy()
    annotations["start_time"] = annotations.index
    # add the end times as the start times plus rolling step size
    annotations["end_time"] = annotations.index + pd.Timedelta(rolling_step_size)
    annotations = annotations.astype({"placement": "category"})
    # in the fixed window dataframe we look if there are differences between
    # consecutive windows and if so we derive a new adjacent window index
    annotations["window_index_adjacent"] = (
        annotations.placement.cat.codes.diff().abs() > 0
    ).cumsum()
    start = annotations.groupby("window_index_adjacent").start_time.first()
    end = annotations.groupby("window_index_adjacent").end_time.last()
    duration = (end - start).dt.total_seconds()
    duration = duration.rename("duration")
    placement = annotations.groupby("window_index_adjacent").placement.first()
    # create the final dataframe
    annotations_adjacent = pd.DataFrame(
        data={
            "start_time": start,
            "end_time": end,
            "duration": duration,
            "placement": placement,
        }
    )

    return annotations_adjacent


def placement_classification_merged(
    acc_ts: pd.DataFrame, acc_ts_euclidean_norm: pd.DataFrame
):
    """Concatenate accelerometer and norm data and call placement function.

    Parameters
    ----------
    acc_ts
        A pd.DataFrame containing the acceleration and gravity axis
    acc_ts_euclidean_norm
        A pd.DataFrame containing the norm of the acceleration

    Returns
    -------
    pandas.DataFrame
        contains start time, end time and label of each continuous segment
    """
    data = pd.concat([acc_ts, acc_ts_euclidean_norm], axis=1)
    return merge_adjacent_annotations_rolling(placement_classification(data))


class ClassifyPlacement(TransformStep):
    """Classify placement in 100 ms windows."""

    data_set_ids = ["acc_ts_resampled", "acc_ts_resampled_euclidean_norm"]
    transform_function = placement_classification_merged
    definitions = PLACEMENT_DEFINITIONS
    new_data_set_id = "placement_bouts"
