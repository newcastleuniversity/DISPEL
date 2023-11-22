"""Tests for :mod:`dispel.data.epochs`."""

import pandas as pd
import pytest

from dispel.data.epochs import Epoch


@pytest.mark.parametrize(
    "start,end",
    [(0, 1), ("2019", "2020"), ("2020-01-01 15:20:30", "2020-01-01 15:20:35")],
)
def test_epoch_converts_type(start, end):
    """Test that effective time frames convert types."""
    epoch = Epoch(start, end)

    assert isinstance(epoch.start, pd.Timestamp)
    assert isinstance(epoch.end, pd.Timestamp)
    assert epoch.start <= epoch.end


def test_epoch_not_negative():
    """Test that time frames have not negative time spans."""
    with pytest.raises(ValueError):
        Epoch(1, 0)


@pytest.mark.parametrize(
    "value,res",
    [
        (pd.Timestamp(1.5), True),
        (pd.Timestamp(0), False),
        (pd.Timestamp(3), False),
        (Epoch(1.1, 1.2), True),
        (Epoch(1.1, 3), True),
        (Epoch(0, 0.9), False),
        (Epoch(0, 1), True),
    ],
)
def test_epoch_overlaps(value, res):
    """Test overlapping functionality."""
    epoch = Epoch(1, 2)
    assert epoch.overlaps(value) == res


def test_epoch_overlaps_type():
    """Test unsupported types for overlap testing."""
    with pytest.raises(ValueError):
        Epoch(1, 2).overlaps(2)


def test_epoch_start_not_none():
    """Test that a time frame always has to have a start date."""
    with pytest.raises(ValueError):
        Epoch(None, 1)


def test_epoch_incomplete():
    """Test if a time frame is incomplete."""
    epoch = Epoch(1, None)
    assert epoch.is_incomplete
