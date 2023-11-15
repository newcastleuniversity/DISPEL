"""A module dedicated to the Dynamic Time Warping."""
from typing import Tuple

import numpy as np
import pandas as pd
from fastdtw import fastdtw

from dispel.signal.core import euclidean_distance


def get_minimal_matches(
    actual: pd.DataFrame, expected: pd.DataFrame, distance: int = 2
) -> Tuple[float, pd.DataFrame]:
    """Compute minimal euclidean distance between actual and expected paths.

    Based on the Dynamic Time Warping algorithm (`fastdtw` library), compute all the
    detected attributions between an actual and an expected trajectory and then extract
    the minimum ones.

    Parameters
    ----------
    actual
        The actual trajectory (x and y coordinates).
    expected
        The expected trajectory (x and y coordinates).
    distance
        The chosen metric to compute DTW, 2 being by default and means euclidean
        distance.

    Returns
    -------
    Tuple[float, pandas.DataFrame]
        A coupling measure and a pandas data frame with minimal attributions indexes for
        actual and expected trajectories and the euclidean distance between each
        attribution.
    """
    path_model = expected[["x", "y"]].to_numpy()
    # Detect if up-sampling has been previously done on data.
    path_user = actual[["x", "y"]].to_numpy()

    # Compute the fast Dynamic Time Warping similarity measures.
    coupling_measure, matches = fastdtw(path_user, path_model, dist=distance)

    # Store matched attributions into a pandas data frame.
    matches = pd.DataFrame(matches).rename(columns={0: "actual", 1: "expect"})

    # Compute the distance between each attributed points.
    matches["min_distance"] = matches.apply(
        lambda row: euclidean_distance(
            (actual["x"][row["actual"]], actual["y"][row["actual"]]),
            (expected["x"][row["expect"]], expected["y"][row["expect"]]),
        ),
        axis=1,
    )

    # Keep only one attribution between a user point and a model point based on keeping
    # the minimum distance.
    min_matches = matches.groupby(by="actual").min().reset_index()
    return coupling_measure, min_matches


def get_dtw_distance(user: pd.DataFrame, reference: pd.DataFrame) -> pd.Series:
    """Extract information about DTW metrics.

    This implementation is using Dynamic Time Warping (`fastdtw` library) to compute the
    similarity measures between the user and the model paths. This algorithm returns the
    coupling measure value (which is the value of the similarity metric) and the
    attributions between a model point and the user points close enough to be considered
    as a potential similar point (i.e. ``attributions == [(m0, u0),(m1, u1),(m1, u2),
    (m1, u3), (m2, u1), (m2, u2), (m2, u2), (m2, u3), (m2,u4),...]`` with mi the ith
    model point index and uj the jth user point index). Then, we isolate the minimum
    distance between each model point and attributed user points. Those features are in
    fact equivalent to those obtained with the variant Fréchet similarity measure
    algorithm with a back propagation. This implementation is around 30 times faster
    than the mentioned Fréchet algorithm.

    Parameters
    ----------
    user
        A pandas data frame composed of the user paths or his up sampled ones.
    reference
        The reference trajectory corresponding to the current level.

    Returns
    -------
    pandas.Series
        A pandas data frame which contains the DTW coupling measure, the mean and median
        minimum euclidean distance between the closest attributions, the standard
        deviation of minimum euclidean distance between the closest attributions and the
        sum of all the minimum euclidean distance between the closest attributions.
    """
    coupling_measure, min_matches = get_minimal_matches(user, reference)

    # Create the pandas data frame with all features.
    dtw_dict = {
        "dtw_coupling_measure": coupling_measure,
        "dtw_mean_distance": np.mean(min_matches["min_distance"]),
        "dtw_median_distance": np.median(min_matches["min_distance"]),
        "dtw_std_distance": np.std(min_matches["min_distance"]),
        "dtw_total_distance": np.sum(min_matches["min_distance"]),
    }
    return pd.Series(dtw_dict, name="dtw_data")
