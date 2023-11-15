"""A module dedicated to intentional tremors computation."""
import pandas as pd
from fastdtw import fastdtw

from dispel.providers.generic.tasks.draw.intersections import get_segment
from dispel.providers.generic.tasks.draw.shapes import get_segment_deceleration
from dispel.signal.core import euclidean_distance

#: The number of data point of the square-like model shapes.
_SQUARE_SHAPES_SIZE = 1038


def _get_segment_indexes(data: pd.DataFrame, segment: range, model_len: int):
    """Get the reference segment on which to compute intentional tremors.

    Only for square-like shapes. Support also BDH format.

    Parameters
    ----------
    data
        The pandas data frame containing all the minimum distance attributions
        from Dynamic Time Warping between the user trajectory and the
        reference.
    segment
        The range of indexes defining the segment on which to compute the
        intentional tremors.
    model_len
        The length of the current reference trajectory.

    Returns
    -------
    pandas.DataFrame
        The minimum distance attributions from Dynamic Time Warping between the
        user trajectory and the reference corresponding to the segment of
        interest.
    """
    if model_len == _SQUARE_SHAPES_SIZE:
        segment_data = data.loc[data["ref"].isin(segment)]
    else:
        bdh_first_idx = int((segment[0] / _SQUARE_SHAPES_SIZE) * model_len)
        bdh_last_idx = int((segment[-1] / _SQUARE_SHAPES_SIZE) * model_len)
        segment_data = data[bdh_first_idx:bdh_last_idx]
    return segment_data


def get_deceleration_profile_data(
    user: pd.DataFrame, ref: pd.DataFrame, level_id: str
) -> pd.DataFrame:
    """Transform `{level}_path` data frame to `{level}_deceleration` one.

    It extracts the proper data to compute intentional tremors (`tsTouch` and
    `minimal distance` pd.Series.).

    Parameters
    ----------
    user
        The `{level}_user_paths` data frame.
    ref
        The `{level}_reference` data frame.
    level_id
        The given level on which to compute the data transformation.

    Returns
    -------
    pandas.DataFrame
        The `{level}_deceleration` data frame containing `tsTouch` and
        `min_distance` pd.Series of the user points attributed to the specific
        segment where the deceleration profile is relevant to compute
        intentional tremors.
    """
    new_user = user.join(get_segment(user))
    new_ref = ref.join(get_segment(ref))

    path_model = new_ref[["x", "y"]].to_numpy()
    path_user = new_user[["x", "y"]].to_numpy()

    # Apply DTW on the two paths.
    _, matches = fastdtw(path_user, path_model, dist=2)

    # Store matched attributions into a pandas data frame.
    matches = pd.DataFrame(matches).rename(columns={0: "user", 1: "ref"})

    # Compute the distance between each attributed points.
    matches["min_distance"] = matches.apply(
        lambda row: euclidean_distance(
            new_user["seg"][row["user"]].segment[0],
            new_ref["seg"][row["ref"]].segment[0],
        ),
        axis=1,
    )

    # Keep only one attribution between a user point and a model point based on
    # keeping the minimum distance.
    min_matches = matches.groupby(by="user").min().reset_index()

    # Get the specific segment on which to compute the intentional tremors.
    deceleration_indexes = get_segment_deceleration(level_id)
    reference_len = len(path_model)
    segment_data = _get_segment_indexes(
        min_matches, deceleration_indexes, reference_len
    )
    # Format the new data frame based on only the segment on which to compute
    # intentional tremors.
    segment_data = segment_data.join(
        new_user.loc[segment_data["user"].values, ["tsTouch", "x", "y"]]
    )
    return segment_data.drop(["user", "ref"], axis=1).astype(
        {"min_distance": float, "x": float, "y": float}
    )
