"""Test cases for :mod:`dispel.signal.filter`."""
import numpy as np
import pandas as pd
import pytest

from dispel.signal.filter import (
    butterworth_band_pass_filter,
    butterworth_high_pass_filter,
    butterworth_low_pass_filter,
)


def _test_butterworth_filter(method, zero_phase):
    """Test :func:`dispel.signal.filter.butterworth_low_pass_filter`."""
    # test without time series
    n_samples = 10
    data = pd.Series(np.ones(n_samples))
    order = 2
    cutoff = 1

    res = method(data, cutoff, order, 20, zero_phase)
    assert isinstance(res, pd.Series)

    # test with time series index
    data = pd.Series(
        np.ones(n_samples), index=pd.date_range("now", periods=n_samples, freq="50ms")
    )
    res_ts = method(data, cutoff, order, zero_phase=zero_phase)

    assert isinstance(res_ts, pd.Series)
    np.testing.assert_almost_equal(res.values, res_ts.values)

    # shifting of values to ensure original alignment based on order
    if not zero_phase:
        assert res[-order:].isna().all()


@pytest.mark.parametrize("zero_phase", [True, False])
def test_butterworth_low_pass_filter(zero_phase: bool):
    """Test :func:`dispel.signal.filter.butterworth_low_pass_filter`."""
    _test_butterworth_filter(butterworth_low_pass_filter, zero_phase)


@pytest.mark.parametrize("zero_phase", [True, False])
def test_butterworth_high_pass_filter(zero_phase: bool):
    """Test :func:`dispel.signal.filter.butterworth_high_pass_filter`."""
    _test_butterworth_filter(butterworth_high_pass_filter, zero_phase)


@pytest.mark.parametrize("zero_phase", [True])
def test_butterworth_band_pass_filter(zero_phase: bool):
    """Test :func:`dispel.signal.filter.butterworth_band_pass_filter`."""
    # test without time series
    n_samples = 50
    data = pd.Series(np.ones(n_samples))
    order = 2
    lowcut = 1
    highcut = 2

    res = butterworth_band_pass_filter(data, lowcut, highcut, order, 20, zero_phase)
    assert isinstance(res, pd.Series)

    # test with time series index
    data = pd.Series(
        np.ones(n_samples), index=pd.date_range("now", periods=n_samples, freq="50ms")
    )
    res_ts = butterworth_band_pass_filter(data, lowcut, highcut, order, zero_phase)

    assert isinstance(res, pd.Series)
    np.testing.assert_almost_equal(res.values, res_ts.values)

    # shifting of values to ensure original alignment based on order
    if not zero_phase:
        assert res[-order:].isna().all()
