"""Test cases for :mod:`dispel.stats.core`."""
import math

import numpy as np
import pandas as pd
import pytest

from dispel.stats.core import freq_nan, mad, variation, variation_increase

DATA_EXAMPLE = pd.Series([2, 2, 3, 4, 14])


def test_mad():
    """Test :func:`dispel.stats.core.mad`."""
    res = mad(DATA_EXAMPLE)

    assert res == 3.6


def test_variation():
    """Test :func:`dispel.stats.core.variation`."""
    assert variation(DATA_EXAMPLE) == 1.0198039027185568

    null_mean_data = pd.Series([3.0, -4.0, 0.0, 2.0, -1.0])

    with pytest.raises(ZeroDivisionError):
        _ = variation(null_mean_data, error="raise")

    assert variation(null_mean_data) == 0.0
    assert math.isnan(variation(null_mean_data, error="omit"))


def test_variation_increase():
    """Test :func:`dispel.stats.core.variation_increase`."""
    res = variation_increase(DATA_EXAMPLE)

    assert res == 0.8689660757568884


def test_freq_nan():
    """Test :func:`dispel.stats.core.freq_nan`."""
    data = pd.Series([1, np.NaN, 2, pd.NaT])
    res = freq_nan(data)
    assert res == 0.5
