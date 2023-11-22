"""Tests for :mod:`dispel.providers.generic.activity.turning`."""

import pandas as pd
import pytest

from dispel.providers.generic.activity.turning import Turn, el_gohary_detect_turns


@pytest.fixture
def example_turn_data():
    """Get example turn data."""
    index = pd.date_range("now", periods=61, freq="20ms")
    values = [0] * 10 + list(range(20)) + [20] + list(reversed(range(20))) + [0] * 10
    return pd.Series(values, index=index)


def test_turn_expand(example_turn_data):
    """Test :meth:`dispel.providers.generic.activity.turning.Turn.expand`."""
    index = example_turn_data.index
    turn = Turn(index[30], index[30], example_turn_data)

    # test simple expansion
    expanded_turn = turn.expand(3)
    assert expanded_turn.start == index[13]
    assert expanded_turn.end == index[-14]

    # test boundaries of underlying data series
    expanded_turn = turn.expand(0)
    assert expanded_turn.start == index[0]
    assert expanded_turn.end == index[-1]

    # test negative values
    negative_turn = Turn(index[30], index[30], example_turn_data * -1)
    expanded_turn = negative_turn.expand(3)
    assert expanded_turn.start == index[13]
    assert expanded_turn.end == index[-14]


def test_turn_data(example_turn_data):
    """Test :meth:`dispel.providers.generic.activity.turning.Turn.data`."""
    start = example_turn_data.index[2]
    end = example_turn_data.index[5]

    turn = Turn(start, end, example_turn_data)
    res = turn.data

    assert isinstance(res, pd.Series)
    assert res.index[0] == start
    assert res.index[-1] == end


def test_turn_duration(example_turn_data):
    """Test :meth:`dispel.providers.generic.activity.turning.Turn.duration`."""
    start = example_turn_data.index[3]
    end = example_turn_data.index[5]
    expected_duration = (end - start).total_seconds()

    turn = Turn(start, end, example_turn_data)
    assert turn.duration == expected_duration


def test_turn_direction(example_turn_data):
    """Test :meth:`dispel.providers.generic.activity.turning.Turn.direction`."""
    index = example_turn_data.index
    positive_turn = Turn(index[0], index[-1], example_turn_data)
    negative_turn = Turn(index[0], index[-1], example_turn_data * -1)

    assert positive_turn.direction == 1
    assert negative_turn.direction == -1


def test_turn_angle(example_turn_data):
    """Test :meth:`dispel.providers.generic.activity.turning.Turn.angle`."""
    index = example_turn_data.index

    # no turning
    no_turn = Turn(index[0], index[0], example_turn_data)
    assert no_turn.angle == 0

    # full turn
    expected_angle = example_turn_data.sum() * index.freq.nanos / 1e9
    turn = Turn(index[0], index[-1], example_turn_data)
    assert turn.angle == expected_angle


def test_turn_merge(example_turn_data):
    """Test :meth:`dispel.providers.generic.activity.turning.Turn.merge`."""
    index = example_turn_data.index
    turn = Turn(index[0], index[4], example_turn_data)

    # self merge - self :)
    merged = turn.merge(turn)
    assert merged.start == turn.start
    assert merged.end == turn.end
    pd.testing.assert_series_equal(merged.data, turn.data)

    # other merge
    other_turn = Turn(index[3], index[5], example_turn_data)
    other_merged = turn.merge(other_turn)
    assert other_merged.start == index[0]
    assert other_merged.end == index[5]


def test_el_gohary_detect_turns(example_turn_data):
    """Test El Gohary turn detection.

    Testing :meth:`dispel.providers.generic.activity.turning.el_gohary_detect_turns`.
    """  # noqa: DAR101
    index = example_turn_data.index

    # positive turn
    res = el_gohary_detect_turns(example_turn_data * 10)
    assert len(res) == 1
    turn, *_ = res

    assert isinstance(turn, Turn)
    assert turn.start == index[11]
    assert turn.end == index[-12]

    # negative turn
    res = el_gohary_detect_turns(example_turn_data * -10)
    assert len(res) == 1
    turn, *_ = res

    assert isinstance(turn, Turn)
    assert turn.start == index[11]
    assert turn.end == index[-12]

    # two turns, opposite direction
    data = pd.concat(
        [example_turn_data * 10, example_turn_data * -10], ignore_index=True
    )
    data.index = pd.date_range("now", periods=len(data), freq="20ms")
    res = el_gohary_detect_turns(data)

    assert len(res) == 2

    # two turns but merged
    data = pd.concat(
        [example_turn_data[:-12] * 10, example_turn_data[12:] * 10], ignore_index=True
    )
    data.index = pd.date_range("now", periods=len(data), freq="20ms")
    res = el_gohary_detect_turns(data)

    assert len(res) == 1
