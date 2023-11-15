"""A module dedicated to format user and reference trajectories."""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from dispel.data.levels import Level
from dispel.signal.core import compute_rotation_matrix_2d, euclidean_distance
from dispel.signal.interpolators import cubic_splines, custom_interpolator_1d

LEVEL_TO_MODEL = {
    "square_counter_clock": "squareCounterClock",
    "square_clock": "squareClock",
    "infinity": "infinity",
    "spiral": "spiral",
}

AREA_RADIUS = 40
r"""The radius of a circle centered in the first reference point of a drawing
shape that  define the valid starting area."""

CTL = (56, 639)
CBL = (56, 123)
CBR = (319, 123)
CTR = (319, 639)
r"""Corner definitions, i.e. CTL for Corner Top Left, ..."""

HALF_SCREEN_X = 187.5
r"""Define the half of the x axis of the screen."""

HALF_SCREEN_Y = 406
r"""Define the half of the y axis of the screen."""

SHAPE_BOUNDARIES = ((56, 319), (123, 639))
r"""Define boundaries of a square-like shape in order to detect if a user point
is inside or outside the shape in the screen x and y axes coordinates."""

SHAPE_SIZE = {
    "ADS": {
        "infinity": 1000,
        "spiral": 779,
        "square_clock": 1038,
        "square_counter_clock": 1038,
    },
    "BDH": {
        "infinity": 641,
        "spiral": 607,
        "square_clock": 124,
        "square_counter_clock": 124,
    },
}
r"""Define the number of points of the reference shape."""


def generate_model_coordinates(level_name) -> Dict[str, np.ndarray]:
    """Generate the model coordinates for a given level name.

    Parameters
    ----------
    level_name
        The name of the desired level.

    Returns
    -------
    Dict[str, numpy.ndarray]
        The 2 dimension numpy array corresponding to the desired level (shape).

    Raises
    ------
    ValueError
        If the given level name is unrecognized.
    """
    if level_name == "squareCounterClock":
        return generate_square_counter_clock()
    if level_name == "squareClock":
        return generate_square_clock()
    if level_name == "infinity":
        return generate_infinity()
    if level_name == "spiral":
        return generate_spiral()

    raise ValueError(f"Unknown level name: {level_name}")


def generate_infinity() -> Dict[str, np.ndarray]:
    """Generate the 'infinity' level (shape)."""
    alpha = 280
    angle = 90

    time = np.linspace(0, 2 * np.pi, num=1000)

    x = alpha * np.cos(time) / (np.sin(time) ** 2 + 1)
    y = alpha * np.cos(time) * np.sin(time) / (np.sin(time) ** 2 + 1)

    path = [
        compute_rotation_matrix_2d([0, 0], [x, y], angle * np.pi / 180)
        for x, y in zip(x, y)
    ]

    x_rot = np.asarray([path[i][0] for i in range(len(path))])
    y_rot = np.asarray([path[i][1] for i in range(len(path))])

    width = 375
    height = 812

    x_rot = x_rot + width / 2
    y_rot = y_rot - height / 2

    return dict(xr=x_rot, yr=y_rot)


def generate_square_clock() -> Dict[str, np.ndarray]:
    """Generate the 'squareClock' level (shape)."""
    d_inter_point = 1.5
    step = 50

    x_start = 56
    y_start = -123
    y_end = -566 - 123

    n_points = int(np.abs(y_end - y_start) / d_inter_point)
    y_seg_1 = np.linspace(y_start, y_end, n_points)
    x_seg_1 = np.full(len(y_seg_1), x_start)

    y_start = -566 - 123
    x_start = 56
    x_end = 56 + 263

    n_points = int((x_end - x_start) / d_inter_point)
    x_seg_2 = np.linspace(x_start, x_end, n_points)
    y_seg_2 = np.full(len(x_seg_2), y_start)

    x_start = 56 + 263
    y_start = -566 - 123
    y_end = -123 - step

    n_points = int(np.abs(y_end - y_start) / d_inter_point)
    y_seg_3 = np.linspace(y_start, y_end, n_points)
    x_seg_3 = np.full(len(y_seg_3), x_start)

    y_start = -123 - step
    x_start = 56 + 263
    x_end = 56 + step

    n_points = int(np.abs(x_end - x_start) / d_inter_point)
    x_seg_4 = np.linspace(x_start, x_end, n_points)
    y_seg_4 = np.full(len(x_seg_4), y_start)

    x_rot = np.concatenate((x_seg_1, x_seg_2, x_seg_3, x_seg_4))
    y_rot = np.concatenate((y_seg_1, y_seg_2, y_seg_3, y_seg_4))

    x_rot = np.asarray(x_rot)
    y_rot = np.asarray(y_rot)

    x_rot = np.flip(x_rot)
    y_rot = np.flip(y_rot)

    return dict(xr=x_rot, yr=y_rot)


def generate_square_counter_clock() -> Dict[str, np.ndarray]:
    """Generate the 'squareCounterClock' level (shape)."""
    rectangle_coordinates = generate_square_clock()

    x_rot = rectangle_coordinates["xr"]
    y_rot = rectangle_coordinates["yr"]

    return dict(xr=375 - x_rot, yr=y_rot)


def generate_spiral() -> Dict[str, np.ndarray]:
    """Generate the 'spiral' level (shape)."""
    inter_point_distance = 0.2359

    # Initialization
    pi_number = 8
    path_theta = ((inter_point_distance * 1.7 * (pi_number * np.pi) ** 2 / 200) / 4) * 2
    path_cumulative_theta = 0.0
    path_lv: List[List[float]] = [[], []]
    index = 0
    path_cumulative_distance = 0

    # Generation of the points
    while path_cumulative_distance < np.pi * pi_number:
        index = index + 1
        path_cumulative_distance = np.sqrt(path_cumulative_theta)
        radius = path_cumulative_distance
        path_lv[0].append(radius * np.cos(path_cumulative_distance))
        path_lv[1].append(radius * np.sin(path_cumulative_distance))
        path_cumulative_theta = path_cumulative_theta + path_theta

    x = -np.asarray(path_lv[0]) * 8
    y = np.asarray(path_lv[1]) * 8

    angle = 90
    path = [
        compute_rotation_matrix_2d([0, 0], [x, y], angle * np.pi / 180)
        for x, y in zip(x, y)
    ]

    x_rot = np.asarray([path[i][0] for i in range(len(path))])
    y_rot = np.asarray([path[i][1] for i in range(len(path))])

    x_rot = x_rot + 375 / 2
    y_rot = y_rot - 812 / 2

    x_rot = 375 - x_rot

    x_rot = x_rot[30:-190] - 10
    y_rot = y_rot[30:-190]

    return dict(xr=x_rot, yr=y_rot)


def get_proper_level_to_model(level_id: str) -> str:
    """Extract the correct camel type key to explore the 'screen' data set.

    Parameters
    ----------
    level_id
        The desired level id to consider.

    Returns
    -------
    str
        The corresponding level id in Camel type.

    Raises
    ------
    KeyError
        If the given identifier doesn't match any level.
    """
    for key in filter(level_id.startswith, LEVEL_TO_MODEL):
        return LEVEL_TO_MODEL[key]

    raise KeyError(f"{level_id} does not match any level_id.")


def _change_references(data: pd.DataFrame, height: float):
    """Set up the user and reference shapes into the device reference in pt.

    It also corrects the old data format. Indeed, the expected trajectory has
    been made based on a negative y axis. New formats don't provide yPosition
    in a negative referential, but in a positive one. Then, that is why we
    check the sign of the yPosition of the user's trajectory: if it is
    negative, the trajectory is in the same referential than the expected one.
    If not, a rotation is applied to the user's trajectory by changing the
    positive sign of the user's yPosition into a negative one.
    Then, the data format is projected to the screen references in ``points``
    of the device.

    Parameters
    ----------
    data
        A pandas data frame corresponding to the user or reference trajectory.
    height
        The height in `pts` corresponding to the screen of the user's
        smartphone.

    Returns
    -------
    pandas.DataFrame
        The updated (if necessary) pandas data frame corresponding to the user
        trajectory.
    """
    if ((ordinate := data["y"]) < 0).any():
        data["y"] = ordinate + height
    else:
        data["y"] = height - ordinate
    return data


def check_is_in_area(ser: pd.Series, x_ref: float, y_ref: float) -> bool:
    """Return if the user points are within a specific area.

    The area is considered as the circle of 40 pts radius with x_ref and y_ref
    as its center.

    Parameters
    ----------
    ser
        A pandas series corresponding to a user point.
    x_ref
        The x coordinate of the reference trajectory to be considered as the
        center of the area.
    y_ref
        The y coordinate of the reference trajectory to be considered as the
        center of the area.

    Returns
    -------
    bool
        Whether the point is in the starting area or not.
    """
    # Acceptance threshold is the area radius +- 10% tolerance
    return (
        np.sqrt((ser["x"] - x_ref) ** 2 + (ser["y"] - y_ref) ** 2) < AREA_RADIUS * 1.1
    )


def flag_valid_area(user: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Add a series named `isValidArea` to determine if a point is valid.

    e.g. if a point or a Series of point belongs to a draw which has triggered
    the starting area condition.

    Parameters
    ----------
    user
        A pandas data frame with user trajectory, touch actions
        and timestamps of touch events.
    reference
        The reference trajectory corresponding to the current shape.

    Returns
    -------
    pandas.DataFrame
        The same pandas data frame given as input plus a pandas Series
        `isValidArea`.

    Raises
    ------
    ValueError
        If no `down` touch action is detected.
    """
    actions = user[["touchAction", "x", "y"]].copy()
    down = actions.loc[actions["touchAction"] == "down", ["x", "y"]]

    if len(down) < 1:
        raise ValueError("No down touchAction")

    ref_x = reference.iloc[0]["x"]
    ref_y = reference.iloc[0]["y"]
    down = down.apply(lambda s: check_is_in_area(s, ref_x, ref_y), axis=1)
    # Flag every action posterior to the first valid event as valid
    actions["isValidArea"] = False
    # Find the closest down point to the center of the valid area
    index_min = down.loc[down].index
    if len(index_min) > 0:
        first_valid = min(down.loc[down].index)
        actions.loc[first_valid:, "isValidArea"] = True
    return actions["isValidArea"]


def get_user_path(
    data: pd.DataFrame, reference: pd.DataFrame, height: float
) -> pd.DataFrame:
    """Extract user path coordinates, timestamps and touch actions.

    Parameters
    ----------
    data
        A pandas data frame corresponding to the 'screen' raw data set.
    reference
        The reference trajectory corresponding to the current shape.
    height
        The height in `pts` corresponding to the screen of the user's
        smartphone.

    Returns
    -------
    pandas.DataFrame
        A pandas data frame composed of user path coordinates, timestamps,
        touch actions and the valid or non valid flag.
    """
    cols = ["xPosition", "yPosition", "touchAction", "tsTouch", "pressure"]
    renamed_cols = {"xPosition": "x", "yPosition": "y"}
    checked_data = _change_references(data[cols].rename(columns=renamed_cols), height)
    flags = flag_valid_area(checked_data, reference)
    return pd.concat([checked_data, flags], axis=1)


def get_reference_path(level: Level, height: float) -> pd.DataFrame:
    """Extract the reference trajectory x and y coordinates.

    Parameters
    ----------
    level
        The desired level on which you want to compute model path.
    height
        The height in `pts` corresponding to the screen of the user's
        smartphone.

    Returns
    -------
    pandas.DataFrame
        A pandas data frame composed of the reference trajectory x and y
        coordinates.
    """
    # Check if BDH FORMAT
    # When it is in BDH format we will correct the orientation first by
    # Rotating the model path by 90 degree vertical
    if "target_figure_position_xy" in level.context:
        raw = level.context.get_raw_value("target_figure_position_xy")
        model_path = pd.DataFrame(raw, columns=["x", "y"])
        model_path["y"] = -1 * model_path["y"]

        # Interpolating to have same resolution
        shape_name = level.id.id.split("-")[0]
        up_sampling_factor = (
            SHAPE_SIZE["ADS"][shape_name] / SHAPE_SIZE["BDH"][shape_name]
        )
        _ref = model_path[["x", "y"]].to_numpy()
        if shape_name in {"spiral", "infinity"}:
            kind = "cubic"
        else:
            kind = "linear"

        model_path = pd.DataFrame(
            custom_interpolator_1d(_ref, up_sampling_factor, kind=kind),
            columns=["x", "y"],
        )
        # For the specific case of infinity shape
        # We also correct the starting point of the model path to match the
        # Small flag indicating the center of the start area.
        if shape_name == "infinity":
            roll_ = model_path["y"].argmax()
            model_path["x"] = np.roll(model_path["x"], -roll_)
            model_path["y"] = np.roll(model_path["y"], -roll_)
        return _change_references(model_path, height)
    raw = generate_model_coordinates(get_proper_level_to_model(level_id=str(level.id)))
    return _change_references(pd.DataFrame({"x": raw["xr"], "y": raw["yr"]}), height)


def get_valid_path(data: pd.DataFrame) -> pd.DataFrame:
    """Keep only the valid trajectory of the user.

    Parameters
    ----------
    data
        The pandas data frame obtained via a
        :class:`~dispel.providers.generic.tasks.draw.touch.DrawShape`.

    Returns
    -------
    pandas.DataFrame
        The pandas data frame given in input but composed of only the valid
        user trajectory.
    """
    return data.loc[data["isValidArea"]].reset_index(drop=True)


def up_sample_user_path(
    data: pd.DataFrame, up_sampling_factor: float = 5
) -> pd.DataFrame:
    """Compute an up-sampled Bezier curve for a given user trajectory.

    Parameters
    ----------
    data
        A pandas data frame composed of the valid user trajectory and no
        duplicates.
    up_sampling_factor
        The up sampling factor you want to apply.

    Returns
    -------
    pandas.DataFrame
        A pandas data frame composed of x and y up sampled user coordinates,
        namely ``x`` and ``y``.
    """
    path_user = data[["x", "y"]].to_numpy()
    return pd.DataFrame(
        cubic_splines(path_user, up_sampling_factor), columns=["x", "y"]
    )


def get_valid_up_sampled_path(data: pd.DataFrame) -> pd.DataFrame:
    """Extract an up sampled user trajectory.

    It starts by filtering and keeping only the valid trajectory, then remove
    x and y potential duplicates (needed for the interpolation), and then apply
    the interpolation. It only returns a pandas data frame composed of x and y
    up sampled user coordinates, namely ``x`` and ``y``.

    Parameters
    ----------
    data
        The pandas data frame obtained via a
        :class:`~dispel.providers.generic.tasks.draw.touch.DrawShape`.

    Returns
    -------
    pandas.DataFrame
        A pandas data frame composed of valid up sampled user coordinates
        ``x`` and ``y``.
    """
    valid_paths = get_valid_path(data)
    valid_paths = valid_paths.drop_duplicates(subset=["x", "y"])
    valid_paths_up = up_sample_user_path(valid_paths)
    return valid_paths_up


def get_segment_deceleration(level_id: str) -> range:
    """Retrieve the segment of the deceleration profile of a given shape.

    The aim of that is to compute intentional tremors for square-like shapes.
    Segments on which to compute intentional tremors have been determined by
    capturing the velocity profile of each shape against 50 Healthy Control
    Participants evaluations. For each shape, the segment
    is defined as model points where the mean velocity is comprised between 50%
    and 20% of the maximum of the velocity profile.

    Parameters
    ----------
    level_id
        The level id on which to retrieve proper the deceleration profile of
        the shape.

    Returns
    -------
    range
        The list of model path indexes to consider to build the signal to study
        intentional tremors.

    Raises
    ------
    KeyError
        If passed level_id is not in segments.
    """
    #: Those ranges have been extracted from the exploratory work on
    # intentional tremors (see `Draw Tremor.ipynb` in `Exploratory Data
    # Analysis`). They corresponds to segments approaching the second corner of
    # the shape, and on which we expect to observe intentonal tremors from
    # patients. Those segments may need to be tweaked with patient data.
    segments = {
        "square_clock-right": range(415, 472),
        "square_clock-left": range(415, 479),
        "square_counter_clock-right": range(422, 476),
        "square_counter_clock-left": range(386, 490),
    }
    for key in filter(level_id.startswith, segments):
        return segments[key]
    raise KeyError(
        "The intentional tremors for the desired level is not"
        " implemented, please provide a level in"
        " [square_clock-right, square_clock-right-02,"
        " square_clock-left,square_clock-left-02,"
        " square_counter_clock-right,"
        " square_counter_clock-right-02, square_counter_clock-left,"
        " square_counter_clock-left-02]"
    )


def remove_overshoots(data: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Format data to remove overshoots from user trajectory.

    Set the first user point as the closest point from the head of the
    reference trajectory and set the last user point as the closest point from
    the tail of the reference trajectory.

    Parameters
    ----------
    data
        A pandas data frame corresponding to the user data coming from an
        aggregated valid touch of a
        :class:`~dispel.providers.generic.tasks.draw.touch.DrawShape`.
    reference
        A pandas data frame corresponding to the reference trajectory coming
        from a :class:`~dispel.providers.generic.tasks.draw.touch.DrawShape`.

    Returns
    -------
    pandas.DataFrame
        The user data without overshoots (head and tail).
    """
    # Set x and y coordinates from the first and the last reference trajectory.
    x_ref_start = reference.iloc[0]["x"]
    y_ref_start = reference.iloc[0]["y"]
    x_ref_end = reference.iloc[-1]["x"]
    y_ref_end = reference.iloc[-1]["y"]
    # Compute the distance of user points from the head of the reference.
    data["distance_first"] = data.apply(
        lambda s: euclidean_distance([s["x"], s["y"]], [x_ref_start, y_ref_start]),
        axis=1,
    )
    # Compute the distance of user points from the tail of the reference.
    data["distance_last"] = data.apply(
        lambda s: euclidean_distance([s["x"], s["y"]], [x_ref_end, y_ref_end]), axis=1
    )
    # Flag which points are in the valid starting area.
    data = data.join(
        data.apply(
            lambda s: check_is_in_area(s, x_ref_start, y_ref_start), axis=1
        ).rename("start_area")
    )
    # Flag which points are in the end area defined by the 40 pts radius circle
    # with (x_ref_end, y_ref_end) point as its center.
    data = data.join(
        data.apply(lambda s: check_is_in_area(s, x_ref_end, y_ref_end), axis=1).rename(
            "end_area"
        )
    )
    # Determine the first user point going out from the valid starting area.
    min_data = data[~data["start_area"]].index
    if len(min_data) == 0:
        return pd.DataFrame(columns=data.columns)
    first_out_start_area = min(data[~data["start_area"]].index)
    # Determine the last user point before going within the end area.
    max_end_data = data[data["end_area"]].index
    if len(max_end_data) == 0:
        return pd.DataFrame(columns=data.columns)
    max_out_end = data[(~data["end_area"]) & (data.index < max(max_end_data))].index
    if len(max_out_end) == 0:
        return pd.DataFrame(columns=data.columns)
    last_out_end_area = max(max_out_end)
    # Remove early starting points from user trajectory.
    mask_start = data.loc[
        (data["start_area"]) & (data.index < first_out_start_area), "distance_first"
    ]
    idx_start = mask_start.loc[mask_start == mask_start.min()].index.item()
    # Remove overshoot from user trajectory.
    mask_end = data.loc[
        (data["end_area"]) & (data.index > last_out_end_area), "distance_last"
    ]
    idx_end = mask_end.loc[mask_end == mask_end.min()].index.item()
    return data.iloc[idx_start : idx_end + 1]


def remove_reference_head(data: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Remove the reference head if the first user point doesn't match.

    The function removes the reference head when the reference point that is
    the closest to the first user point is different from the first reference
    point.

    Parameters
    ----------
    data
        A pandas data frame containing the user trajectory
        (with or without overshoot).
    reference
        A pandas data frame containing the reference trajectory.

    Returns
    -------
    pandas.DataFrame
        The reference trajectory without the head if needed.
    """
    if len(data) == 0:
        return reference
    new_ref = reference.copy()
    # Flag which points are in the valid starting area defined by the 40 pts
    # radius circle.
    new_ref = new_ref.join(
        new_ref.apply(
            lambda s: check_is_in_area(s, new_ref.iloc[0]["x"], new_ref.iloc[0]["y"]),
            axis=1,
        ).rename("start_area")
    )
    # Compute the distance of reference points from the head of the user
    # trajectory.
    new_ref["distance_first"] = new_ref.apply(
        lambda s: euclidean_distance(
            [s["x"], s["y"]], [data.iloc[0]["x"], data.iloc[0]["y"]]
        ),
        axis=1,
    )
    # Determine the first reference point going out from the valid starting
    # area.
    first_out_start_area = min(new_ref[~new_ref["start_area"]].index)
    # Remove reference points anterior to the closest one to the first point of
    # the user trajectory.
    mask_start = new_ref.loc[
        (new_ref["start_area"]) & (new_ref.index < first_out_start_area),
        "distance_first",
    ]
    ref_idx_first = mask_start.loc[mask_start == mask_start.min()].index.item()
    return new_ref[ref_idx_first:].reset_index().rename(columns={"index": "old_index"})


# The next functions define 4 parts of the square-like shapes, splitting them
# by drawing a line at the center of each axis of shapes.


def top_left(data: pd.DataFrame) -> pd.DataFrame:
    """Return the top left part of a square-like shape (user/reference).

    Parameters
    ----------
    data
        A pandas data frame containing at least a trajectory
        (x and y coordinates) of a square-like shape.

    Returns
    -------
    pandas.DataFrame
        The top left part of the square-like shape given as input.
    """
    return data.loc[(data["x"] < HALF_SCREEN_X) & (data["y"] > HALF_SCREEN_Y)]


def bottom_left(data: pd.DataFrame) -> pd.DataFrame:
    """Return the bottom left part of a square-like shape (user/reference).

    Parameters
    ----------
    data
        A pandas data frame containing at least a trajectory
        (x and y coordinates) of a square-like shape.

    Returns
    -------
    pandas.DataFrame
        The bottom left part of the square-like shape given as input.
    """
    return data.loc[(data["x"] < HALF_SCREEN_X) & (data["y"] < HALF_SCREEN_Y)]


def top_right(data: pd.DataFrame) -> pd.DataFrame:
    """Return the top right part of a square-like shape (user/reference).

    Parameters
    ----------
    data
        A pandas data frame containing at least a trajectory
        (x and y coordinates) of a square-like shape.

    Returns
    -------
    pandas.DataFrame
        The top right part of the square-like shape given as input.
    """
    return data.loc[(data["x"] > HALF_SCREEN_X) & (data["y"] > HALF_SCREEN_Y)]


def bottom_right(data: pd.DataFrame) -> pd.DataFrame:
    """Return the bottom right part of a square-like shape (user/reference).

    Parameters
    ----------
    data
        A pandas data frame containing at least a trajectory
        (x and y coordinates) of a square-like shape.

    Returns
    -------
    pandas.DataFrame
        The bottom right part of the square-like shape given as input.
    """
    return data.loc[(data["x"] > HALF_SCREEN_X) & (data["y"] < HALF_SCREEN_Y)]


def extract_distance_axis_sc(
    data: pd.DataFrame, ref: pd.DataFrame
) -> Tuple[float, float, float]:
    """Extract the distance between draws and reference perpendicular axes.

    This function is specific to the `square_clock` shapes.

    This distance is defined as follow: the distance between a user trajectory
    and an encountered perpendicular axis corresponding to a 90° trajectory
    change of the square-like shape. If the user overshoots
    (goes outside the square) the 90° trajectory change, the distance will be
    the perpendicular projection of the most distant user point to the axis and
    the axis itself. If the user never overshoot (stay within the square) the
    axis, the considered distance will be the perpendicular projection of the
    closest user point of the axis and the axis itself.

    Parameters
    ----------
    data
        A pandas data frame composed of at least the user trajectory to
        consider.

    ref
        A pandas data frame composed of at least the reference trajectory to
        consider.

    Returns
    -------
    Tuple[float, float, float]
        A tuple composed of the extracted aforementioned distances (for the 3
        axes).
    """
    top_r = top_right(data)
    top_r_ref = top_right(ref)
    top_r_dist = top_r.x.max() - top_r_ref.x.max()

    bott_r = bottom_right(data)
    bott_r_ref = bottom_right(ref)
    bott_r_dist = bott_r_ref.y.min() - bott_r.y.min()

    bott_l = bottom_left(data)
    bott_l_ref = bottom_left(ref)
    bott_l_dist = bott_l_ref.x.min() - bott_l.x.min()
    return top_r_dist, bott_r_dist, bott_l_dist


def extract_distance_axis_scc(
    data: pd.DataFrame, ref: pd.DataFrame
) -> Tuple[float, float, float]:
    """Extract the distance between draws and reference perpendicular axes.

    This function is specific to the `square_counter_clock` shapes.

    This distance is defined as follow: the distance between a user trajectory
    and an encountered perpendicular axis corresponding to a 90° trajectory
    change of the square-like shape. If the user overshoots
    (goes outside the square) the 90° trajectory change, the distance will be
    the perpendicular projection of the most distant user point to the axis and
    the axis itself. If the user never overshoot (stay within the square) the
    axis, the considered distance will be the perpendicular projection of the
    closest user point of the axis and the axis itself.

    Parameters
    ----------
    data
        A pandas data frame composed of at least the user trajectory to
        consider.

    ref
        A pandas data frame composed of at least the reference trajectory to
        consider.

    Returns
    -------
    Tuple[float, float, float]
        A tuple composed of the extracted aforementioned distances (for the 3
        axes).
    """
    top_l = top_left(data)
    top_l_ref = top_left(ref)
    top_l_dist = top_l_ref.x.min() - top_l.x.min()

    bott_l = bottom_left(data)
    bott_l_ref = bottom_left(ref)
    bott_l_dist = bott_l_ref.y.min() - bott_l.y.min()

    bott_r = bottom_right(data)
    bott_r_ref = bottom_right(ref)
    bott_r_dist = bott_r.x.max() - bott_r_ref.x.max()
    return top_l_dist, bott_l_dist, bott_r_dist


def get_max_dist_corner(
    user: pd.DataFrame, attrib: pd.DataFrame, ref: pd.DataFrame, corner: Tuple[int, int]
) -> float:
    """Retrieve the Fréchet distance within a corner zone.

    The Fréchet distance is the maximum of the minimum distance between the
    user and the reference trajectories based on the dynamic time warping
    attributions.

    Parameters
    ----------
    user
        The user trajectory user for computing the DTW attributions.
    attrib
        The related dynamic time warping attributions for the considered
        reference trajectory with the user trajectory of interest.
    ref
        The reference trajectory
    corner
        The corner zone of the reference trajectory to consider.

    Returns
    -------
    float
        The maximum of the minimum distances from DTW attributions within the
        considered corner zone.
    """
    is_in = ref.apply(lambda s: check_is_in_area(s, corner[0], corner[1]), axis=1)
    idx = ref[is_in].index
    if idx.size == 0:
        return np.nan
    narrow_attrib = attrib.loc[attrib.expect.isin(idx)]
    max_dist = narrow_attrib.min_distance.max()
    user_idx = narrow_attrib.loc[
        narrow_attrib.min_distance == max_dist, "actual"
    ].item()
    user_point = user.loc[user_idx, ["x", "y"]]
    if (
        (user_point["x"] > SHAPE_BOUNDARIES[0][0])
        & (user_point["x"] < SHAPE_BOUNDARIES[0][1])
        & (user_point["y"] > SHAPE_BOUNDARIES[1][0])
        & (user_point["y"] < SHAPE_BOUNDARIES[1][1])
    ):
        return -max_dist
    return max_dist


def scc_corners_max_dist(
    user: pd.DataFrame, ref: pd.DataFrame, attrib: pd.DataFrame
) -> Tuple[float, float, float]:
    """Compute maximum distance per corner zones.

    For square counter clock shapes.

    Parameters
    ----------
    user
        A pandas data frame composed of at least the user trajectory used for
        computing DTW attributions.
    ref
        A pandas data frame composed of at least the reference trajectory.
    attrib
        The related dynamic time warping attributions for the considered
        reference trajectory with the user trajectory of interest.

    Returns
    -------
    Tuple[float, float, float]
        A tuple composed of the extracted aforementioned distances (for the 3
        related corners).
    """
    dist_ctl = get_max_dist_corner(user, attrib, ref, CTL)
    dist_cbl = get_max_dist_corner(user, attrib, ref, CBL)
    dist_cbr = get_max_dist_corner(user, attrib, ref, CBR)
    return dist_ctl, dist_cbl, dist_cbr


def sc_corners_max_dist(
    user: pd.DataFrame, ref: pd.DataFrame, attrib: pd.DataFrame
) -> Tuple[float, float, float]:
    """Compute maximum distance per corner zones.

    For square clock shapes.

    Parameters
    ----------
     user
        A pandas data frame composed of at least the user trajectory used for
        computing DTW attributions.
    ref
        A pandas data frame composed of at least the reference trajectory.
    attrib
        The related dynamic time warping attributions for the considered
        reference trajectory with the user trajectory of interest.

    Returns
    -------
    Tuple[float, float, float]
        A tuple composed of the extracted aforementioned distances (for the 3
        related corners).
    """
    dist_ctr = get_max_dist_corner(user, attrib, ref, CTR)
    dist_cbr = get_max_dist_corner(user, attrib, ref, CBR)
    dist_cbl = get_max_dist_corner(user, attrib, ref, CBL)
    return dist_ctr, dist_cbr, dist_cbl
