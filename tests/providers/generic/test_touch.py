"""Tests for :mod:`dispel.processing.touch`."""

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import pytest
from ground.core.geometries import Multisegment, Point

from dispel.providers.bdh.io import read_bdh
from dispel.providers.generic.touch import Gesture, Touch, split_touches
from dispel.signal.core import euclidean_norm
from tests.providers.bdh.tasks.test_ft import PATH_FT

_POSITION_COLUMNS = ["x", "y"]

_TOUCH_BEGIN = pd.Timestamp("2021")

_TOUCH_POINTS = 20

_TOUCH_DT = "50ms"

_TOUCH_DX = 1

_TOUCH_DY = 2


def generate_actions(n_events: int) -> Sequence[str]:
    """Generate action sequence for touch interactions."""
    assert n_events > 2
    return ["down"] + ["move"] * (n_events - 2) + ["up"]


@pytest.fixture
def example_touch_time_range() -> pd.DatetimeIndex:
    """Create example fixture for time stamps."""
    return pd.date_range(
        _TOUCH_BEGIN, periods=_TOUCH_POINTS, freq=_TOUCH_DT, name="tsTouch"
    )


@pytest.fixture
def example_touch_positions() -> pd.DataFrame:
    """Create example fixture for touch coordinates."""
    return pd.DataFrame(
        {
            "x": np.arange(_TOUCH_POINTS) ** _TOUCH_DX,
            "y": np.arange(_TOUCH_POINTS) ** _TOUCH_DY,
        }
    )


@pytest.fixture
def example_touch_data(
    example_touch_time_range: pd.DatetimeIndex, example_touch_positions: pd.DataFrame
) -> pd.DataFrame:
    """Create example data for touch interactions."""
    data = {
        "tsTouch": example_touch_time_range,
        "touchAction": generate_actions(_TOUCH_POINTS),
        "x": example_touch_positions["x"],
        "y": example_touch_positions["y"],
    }

    return pd.DataFrame(data)


@pytest.mark.parametrize("drop", (None, "tsTouch", "touchAction", "x", "y"))
def test_touch_post_init_correct_columns(
    example_touch_data: pd.DataFrame, drop: Optional[str]
):
    """Test that the creation of Touch asserts all required columns."""
    if not drop:
        touch = Touch(example_touch_data)
        assert isinstance(touch, Touch)
    else:
        with pytest.raises(AssertionError):
            Touch(example_touch_data.drop(drop, axis="columns"))


def test_touch_post_init_min_events(example_touch_data: pd.DataFrame):
    """Test that the creation of Touch checks for minimum number of events."""
    with pytest.raises(AssertionError):
        Touch(pd.DataFrame(columns=example_touch_data.columns))

    with pytest.raises(AssertionError):
        Touch(example_touch_data.iloc[0:1])


@pytest.fixture
def example_touch(example_touch_data: pd.DataFrame) -> Touch:
    """Create example touch object."""
    return Touch(example_touch_data)


@pytest.fixture
def example_touch_w_pressure(example_touch_data: pd.DataFrame) -> Touch:
    """Create example touch with pressure."""
    data = example_touch_data.copy()
    data["pressure"] = np.arange(len(data)) ** 2

    return Touch(data)


def test_touch_begin(example_touch: Touch):
    """Test that the start is correctly identified from the data frame."""
    expected_begin = _TOUCH_BEGIN
    assert example_touch.begin == expected_begin


def test_touch_end(example_touch: Touch):
    """Test that the end is correctly identified from the data frame."""
    expected_end = _TOUCH_BEGIN + pd.Timedelta(_TOUCH_DT) * (_TOUCH_POINTS - 1)
    assert example_touch.end == expected_end


def test_touch_duration(example_touch: Touch):
    """Test that the duration is correctly computed from the data frame."""
    expected_duration = pd.Timedelta(_TOUCH_DT) * (_TOUCH_POINTS - 1)
    assert example_touch.duration == expected_duration


def test_touch_length():
    """Test that the length of the interaction is calculated correctly."""
    data = pd.DataFrame(
        {
            "tsTouch": pd.date_range("2021", periods=3, freq=_TOUCH_DT),
            "touchAction": ["down", "move", "up"],
            "xPosition": [0, 3, 6],
            "yPosition": [0, 4, 8],
        }
    )
    touch = Touch(data)

    assert touch.length == 10


def test_touch_not_is_incomplete(example_touch: Touch):
    """Test that a complete interaction is complete."""
    assert not example_touch.is_incomplete


def test_touch_is_incomplete(example_touch_data: pd.DataFrame):
    """Test that missing down or up lead to incomplete touch interactions."""
    touch_a = Touch(example_touch_data[:-1])
    assert touch_a.is_incomplete

    touch_b = Touch(example_touch_data[1:])
    assert touch_b.is_incomplete


def test_touch_overlaps(
    example_touch_data: pd.DataFrame, example_touch_time_range: pd.DatetimeIndex
):
    """Test overlapping of touch interactions."""
    touch_a = Touch(example_touch_data)

    # identity
    assert touch_a.overlaps(touch_a)

    shift = _TOUCH_POINTS / 2
    shift_no_overlap = _TOUCH_POINTS

    # forward overlap
    forward_overlap = example_touch_data.copy()
    forward_overlap["tsTouch"] = example_touch_time_range.shift(shift)
    touch_b = Touch(forward_overlap)

    assert touch_a.overlaps(touch_b)
    assert touch_b.overlaps(touch_a)

    # forward no overlap
    forward_no_overlap = example_touch_data.copy()
    forward_no_overlap["tsTouch"] = example_touch_time_range.shift(shift_no_overlap)
    touch_c = Touch(forward_no_overlap)

    assert not touch_a.overlaps(touch_c)
    assert not touch_c.overlaps(touch_a)

    # backward overlap
    backward_overlap = example_touch_data.copy()
    backward_overlap["tsTouch"] = example_touch_time_range.shift(-shift)
    touch_d = Touch(backward_overlap)

    assert touch_a.overlaps(touch_d)
    assert touch_d.overlaps(touch_a)

    # backward no overlap
    backward_no_overlap = example_touch_data.copy()
    backward_no_overlap["tsTouch"] = example_touch_time_range.shift(-shift_no_overlap)
    touch_e = Touch(backward_no_overlap)

    assert not touch_a.overlaps(touch_e)
    assert not touch_e.overlaps(touch_a)


def test_touch_to_segments(example_touch: Touch, example_touch_data: pd.DataFrame):
    """Test conversion of touches into Multisegments."""
    res = example_touch.to_segments()

    assert isinstance(res, Multisegment)

    segments = res.segments
    assert len(segments) == _TOUCH_POINTS - 1

    def _convert_points(row: pd.Series) -> Point:
        return Point(row["x"], row["y"])

    data = example_touch_data
    expected_begins = data.iloc[:-1].apply(_convert_points, axis=1).tolist()
    expected_ends = data.iloc[1:].apply(_convert_points, axis=1).tolist()

    assert all(a.start == e for a, e in zip(segments, expected_begins))
    assert all(a.end == e for a, e in zip(segments, expected_ends))


def test_touch_positions(example_touch: Touch, example_touch_data: pd.DataFrame):
    """Test retrieving touch positions."""
    pd.testing.assert_frame_equal(
        example_touch.positions,
        example_touch_data.set_index("tsTouch")[_POSITION_COLUMNS],
    )


def test_touch_displacements(example_touch: Touch, example_touch_data: pd.DataFrame):
    """Test retrieving displacements."""
    expected = (
        example_touch_data.set_index("tsTouch")[_POSITION_COLUMNS].diff().dropna()
    )
    pd.testing.assert_frame_equal(example_touch.displacements, expected)


def test_touch_movement_begin(example_touch: Touch, example_touch_data: pd.DataFrame):
    """Test retrieving movement begin."""
    expected = pd.to_datetime(example_touch_data.iloc[1]["tsTouch"])
    assert (expected - example_touch.movement_begin).total_seconds() == 0


def test_touch_velocity(example_touch: Touch):
    """Test velocity computation."""
    px_dt = pd.Timedelta("1ms") / pd.Timedelta(_TOUCH_DT)

    expected = example_touch.displacements * px_dt
    pd.testing.assert_frame_equal(example_touch.velocity, expected)


def test_touch_velocity_constant(example_touch_time_range: pd.DatetimeIndex):
    """Test velocity computation for constant drawing interaction."""
    px_dt = pd.Timedelta("1ms") / pd.Timedelta(_TOUCH_DT)

    data = pd.DataFrame(
        {
            "tsTouch": example_touch_time_range,
            "touchAction": generate_actions(_TOUCH_POINTS),
            "x": np.arange(_TOUCH_POINTS),
            "y": np.arange(_TOUCH_POINTS),
        }
    )

    touch = Touch(data)

    assert (touch.velocity == px_dt).all(axis=None)


def test_touch_speed(example_touch: Touch):
    """Test speed computation."""
    px_dt = pd.Timedelta("1ms") / pd.Timedelta(_TOUCH_DT)
    norm = euclidean_norm(example_touch.displacements)
    expected = norm * px_dt

    pd.testing.assert_series_equal(example_touch.speed, expected)


def test_touch_acceleration(example_touch: Touch):
    """Test acceleration computation."""
    delta = example_touch.speed.diff()
    expected = (delta / example_touch.time_deltas).dropna()

    pd.testing.assert_series_equal(example_touch.acceleration, expected)


@pytest.mark.xfail
def test_touch_jerk(example_touch: Touch):
    """Test jerk computation."""
    res = example_touch.jerk

    assert len(res.unique()) == 1


def test_touch_has_pressure(example_touch: Touch, example_touch_w_pressure: Touch):
    """Test correct testing for present pressure information."""
    assert not example_touch.has_pressure
    assert example_touch_w_pressure.has_pressure


def test_touch_pressure(example_touch_w_pressure: Touch):
    """Test accessing pressure information."""
    assert isinstance(example_touch_w_pressure.pressure, pd.Series)


def test_touch_pressure_acceleration(example_touch_w_pressure: Touch):
    """Test pressure acceleration computation of pressure."""
    res = example_touch_w_pressure.pressure_acceleration

    # as values were x^2 jerk has to be constant
    assert len(res.unique()) == 4


def generate_touch_data_set(
    timestamps: pd.DatetimeIndex, offset: int, path_id: int
) -> pd.DataFrame:
    """Generate a test data set for touch interactions."""
    return pd.DataFrame(
        {
            "tsTouch": timestamps,
            "touchAction": generate_actions(len(timestamps)),
            "xPosition": np.arange(offset, offset + len(timestamps)),
            "yPosition": np.arange(offset, offset + len(timestamps)),
            "touchPathId": path_id,
        }
    )


@pytest.fixture
def example_gesture_data() -> pd.DataFrame:
    """Create a fixture for touch data interaction."""
    # generate data for two gestures, one with two touches and one with a
    # single one
    index_1 = pd.date_range("2021", periods=10, freq=_TOUCH_DT)
    index_2 = index_1 + pd.Timedelta(_TOUCH_DT)
    index_3 = index_2 + pd.Timedelta(_TOUCH_DT) * 12

    return pd.concat(
        [
            generate_touch_data_set(index_1, 0, 0),
            generate_touch_data_set(index_2, 10, 1),
            generate_touch_data_set(index_3, 20, 2),
        ],
        ignore_index=True,
    )


def test_gesture_from_data_frame(example_gesture_data: pd.DataFrame):
    """Test converting touch event data frame into gestures."""
    gestures = Gesture.from_data_frame(example_gesture_data)

    assert isinstance(gestures, list)
    assert len(gestures) == 2
    assert all(isinstance(g, Gesture) for g in gestures)

    first_gesture = gestures[0]

    assert isinstance(first_gesture, Gesture)
    assert len(first_gesture.touches) == 2
    assert all(isinstance(t, Touch) for t in first_gesture.touches)

    second_gesture = gestures[1]

    assert isinstance(second_gesture, Gesture)
    assert len(second_gesture.touches) == 1
    assert isinstance(second_gesture.touches[0], Touch)


@pytest.fixture
def example_gesture(example_gesture_data: pd.DataFrame) -> Gesture:
    """Create example gesture fixture."""
    return Gesture.from_data_frame(example_gesture_data)[0]


def test_gesture_first_touch(example_gesture: Gesture):
    """Test getting the first touch interaction of a gesture."""
    assert example_gesture.first_touch == example_gesture.touches[0]


def test_gesture_last_touch(example_gesture: Gesture):
    """Test getting the last touch interaction of a gesture."""
    assert example_gesture.last_touch == example_gesture.touches[-1]


def test_gesture_begin(example_gesture: Gesture):
    """Test getting the start date time of a gesture."""
    assert example_gesture.begin == min(t.begin for t in example_gesture.touches)


def test_gesture_end(example_gesture: Gesture):
    """Test getting the end date time of a gesture."""
    assert example_gesture.end == max(t.end for t in example_gesture.touches)


def test_gesture_duration(example_gesture: Gesture):
    """Test getting the duration of a gesture."""
    expected = example_gesture.end - example_gesture.begin
    assert example_gesture.duration == expected


def test_gesture_first_movement(example_gesture: Gesture):
    """Test getting the first moved touch of a gesture."""
    assert example_gesture.first_movement == example_gesture.touches[0]


def test_gesture_movement_begin(example_gesture: Gesture):
    """Test getting the movement begin of a gesture."""
    expected = example_gesture.touches[0].movement_begin
    assert example_gesture.movement_begin == expected


def test_gesture_dwell_time(example_gesture: Gesture):
    """Test getting the movement begin of a gesture."""
    assert example_gesture.dwell_time == pd.Timedelta(50, unit="ms")


@pytest.fixture(scope="session")
def ft_reading():
    """Create a bdh finger tapping reading fixture."""
    return read_bdh(PATH_FT)


def test_path_id_preprocessing(ft_reading):
    """Test the raw touch events preprocessing."""
    max_touch_path_id = 169
    touch_events = ft_reading.get_level("left").get_raw_data_set("screen").data
    processed_touch_events = split_touches(
        touch_events, touch_events["tsTouch"].min(), touch_events["tsTouch"].max()
    )
    # Make sure that the generated touch path ids are here
    assert list(processed_touch_events["touchPathId"].unique()) == list(
        range(0, max_touch_path_id)
    )
