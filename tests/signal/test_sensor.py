"""Test cases for :mod:`dispel.signal.sensor`."""

import numpy as np
import pandas as pd

from dispel.signal.sensor import check_amplitude, detrend_signal


def test_detrend_data():
    """Test :func:`dispel.signal.sensor.detrend_data`."""
    x = np.arange(1, 101, 1)
    y1_true = 10 * np.sin(x)
    y2_true = 5 * np.sin(x)
    y_trend = x * 0.5 + 3
    y1_raw = y1_true + y_trend
    y2_raw = y2_true + y_trend
    time_index = pd.date_range(pd.to_datetime("2020-01-01"), periods=100).tolist()
    y_trend = pd.DataFrame({"y1": y1_raw, "y2": y2_raw}, index=time_index)

    y_detrend = y_trend.apply(detrend_signal)
    np.testing.assert_allclose(y1_true, y_detrend.y1.values, atol=0.6)
    np.testing.assert_allclose(y2_true, y_detrend.y2.values, atol=0.6)


def test_check_amplitude():
    """Test :func:`dispel.signal.sensor.resample_data`."""
    x = np.arange(1, 101, 1)

    data = pd.DataFrame({"y1": np.sin(x), "y2": 10 * np.cos(x)})

    assert check_amplitude(data, 0, 30)
    assert not check_amplitude(data, 0, 1)
