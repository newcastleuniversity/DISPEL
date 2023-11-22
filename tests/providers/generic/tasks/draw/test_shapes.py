"""Tests for :mod:`dispel.providers.generic.tasks.draw.shapes`."""
import numpy as np
import pandas as pd
import pytest

from dispel.providers.generic.tasks.draw.shapes import (
    flag_valid_area,
    generate_infinity,
    generate_spiral,
    generate_square_clock,
    generate_square_counter_clock,
    get_proper_level_to_model,
    get_segment_deceleration,
    get_valid_path,
    remove_overshoots,
    remove_reference_head,
)


def test_infinity():
    """Test the proper computation of the `infinity` shape."""
    res = generate_infinity()

    assert len(res["xr"]) == 1000
    assert len(res["yr"]) == 1000
    assert res["xr"][0] == 187.50000000000003
    assert res["yr"][0] == -126.0
    assert res["xr"][len(res)] == 183.97882277187645
    assert res["xr"][len(res)] == 183.97882277187645


def test_spiral():
    """Test the proper computation of the `spiral` shape."""
    res = generate_spiral()

    assert len(res["xr"]) == 779
    assert len(res["yr"]) == 779
    assert res["xr"][0] == 144.78839803480594
    assert res["yr"][0] == -393.9231218262853
    assert res["xr"][len(res)] == 142.28328637460368
    assert res["yr"][len(res)] == -398.467207449049


def test_square_clock():
    """Test the proper computation of the `square_clock` shape."""
    res = generate_square_clock()

    assert len(res["xr"]) == 1038
    assert len(res["yr"]) == 1038
    assert res["xr"][0] == 106.0
    assert res["yr"][0] == -173.0
    assert res["xr"][len(res)] == 109.02127659574467
    assert res["yr"][len(res)] == -173.0


def test_square_counter_clock():
    """Test the proper computation of the `square_counter_clock` shape."""
    res = generate_square_counter_clock()

    assert len(res["xr"]) == 1038
    assert len(res["yr"]) == 1038
    assert res["xr"][0] == 269.0
    assert res["yr"][0] == -173.0
    assert res["xr"][len(res)] == 265.97872340425533
    assert res["yr"][len(res)] == -173.0


def test_deceleration_indexes():
    """Test :func:`dispel.signal.drawing_lib.get_segment_deceleration`."""
    assert get_segment_deceleration("square_clock-right") == range(415, 472)

    with pytest.raises(KeyError):
        get_segment_deceleration("non_valid_level_id")


@pytest.mark.parametrize(
    "level,model",
    [
        ("square_counter_clock-right", "squareCounterClock"),
        ("square_clock-right_2", "squareClock"),
        ("infinity-left_2", "infinity"),
        ("spiral-left", "spiral"),
    ],
)
def test_get_proper_level_to_model(level, model):
    """Test if level_ids match with level names."""
    res = get_proper_level_to_model(level)
    assert res == model


def test_flag_valid_area(data_to_flag_as_valid):
    """Test the consistency of the flagging method of a gesture."""
    ref = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 1.0, 2.0, 3.0]})
    flagged = flag_valid_area(data_to_flag_as_valid, ref)
    assert flagged.to_list() == [False, False, True, True]


def test_get_valid_path(data_to_flag_as_valid):
    """Test the consistency of the only valid data selection method."""
    ref = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 1.0, 2.0, 3.0]})
    flagged = flag_valid_area(data_to_flag_as_valid, ref)
    valid = get_valid_path(pd.concat([data_to_flag_as_valid, flagged], axis=1))
    dictionary = {
        "x": [0.0, 0.0],
        "y": [0.0, 0.0],
        "tsTouch": ["2021", "2021"],
        "touchAction": ["down", "move"],
        "isValidArea": [True, True],
    }
    pd.testing.assert_frame_equal(valid, pd.DataFrame(dictionary))


def test_remove_overshoot():
    """Test the good behavior of removing overshoot."""
    l_data = np.linspace(1, 100, 100)
    l_ref = np.linspace(6, 95, 90)
    data = pd.DataFrame({"x": l_data, "y": l_data})
    ref = pd.DataFrame({"x": l_ref, "y": l_ref})
    res = remove_overshoots(data, ref)
    assert len(res[["x", "y"]]) == 90


def test_remove_head():
    """Test the good behavior of removing reference head."""
    l_ref = np.linspace(1, 100, 100)
    l_data = np.linspace(6, 95, 90)
    ref = pd.DataFrame({"x": l_ref, "y": l_ref})
    data = pd.DataFrame({"x": l_data, "y": l_data})
    res = remove_reference_head(data, ref)
    assert len(res) == 95
