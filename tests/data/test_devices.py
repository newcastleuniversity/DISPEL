"""Tests for :mod:`dispel.data.devices`."""

import pytest

from dispel.data.devices import Screen


@pytest.mark.parametrize(
    "width,height", [(-50, 50), (50, -50), (-50, -50), (0, 1), (1, 0), (0, 0)]
)
def test_screen_values_not_negative_or_zero(width, height):
    """Test that screen dimensions cannot be negative."""
    with pytest.raises(ValueError):
        Screen(width, height)
