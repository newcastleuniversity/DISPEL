"""Test cases for :mod:`dispel.signal.core`."""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from dispel.signal.core import (
    compute_rotation_matrix_2d,
    compute_rotation_matrix_3d,
    discretize_sampling_frequency,
    euclidean_distance,
    euclidean_norm,
    extract_sampling_frequency,
    get_cartesian,
    non_uniform_power_spectrum,
)


def _get_non_uniform_data(frac_points: float) -> pd.Series:
    """Provide an example of non uniformly sampled data.

    Parameters
    ----------
    frac_points
        Fraction of points to select

    Returns
    -------
    pandas.Series
        Non uniformly sampled data
    """
    a = 2.0
    w = 1.0
    phi = 0.5 * np.pi
    n_in = 1000
    np.random.seed(42)
    r = np.random.rand(n_in)
    x = np.linspace(0.01, 1e9 * np.pi, n_in)
    x = x[r >= frac_points]
    y = a * np.sin(w * x + phi)

    return pd.Series(index=pd.to_datetime(x, unit="ns"), data=y)


@pytest.fixture(scope="module")
def non_uniform_data():
    """Provide an example of non uniformly sampled data."""
    return _get_non_uniform_data(0.9)


@pytest.fixture(scope="module")
def non_uniform_data_bis():
    """Provide an example of non uniformly sampled data."""
    return _get_non_uniform_data(0.2)


@pytest.mark.parametrize(
    "data,result",
    [
        ([[2, 2, 2]], [3.464]),
        ([[0, 0, 0]], [0.0]),
        ([[2, 3, 4], [1, 2, 3], [4, 5, 6]], [5.385, 3.742, 8.775]),
    ],
)
def test_euclidean_norm(data, result):
    """Testing :func:`dispel.signal.core.euclidean_norm`."""
    res = euclidean_norm(pd.DataFrame(data))
    assert_series_equal(pd.Series(result), res, rtol=0.5e-3)


@pytest.mark.parametrize(
    "point1,point2,result",
    [([2, 2], [2, 2], 0.0), ((0, 0), (4, 2), 4.472), ((223, 346), (125, 543), 220.029)],
)
def test_euclidean_distance(point1, point2, result):
    """Testing :func:`dispel.signal.core.euclidean_distance`."""
    res = euclidean_distance(point1, point2)
    assert res == pytest.approx(result, rel=1.0e-2)


def test_compute_rotation_matrix_3d():
    """Testing :func:`dispel.signal.core.compute_rotation_matrix_3d`."""
    a = np.array([1, 0, 0])
    b = np.array([0, 1, 0])
    res = compute_rotation_matrix_3d(a, b)

    assert res.shape == (3, 3)
    np.testing.assert_equal(res @ a, b)

    # test opposing vectors
    c = np.array([-1, 0, 0])
    res = compute_rotation_matrix_3d(a, c)
    assert res.shape == (3, 3)
    np.testing.assert_equal(res @ a, c)


def test_compute_rotation_matrix_2d():
    """Test the rotation operation of a 2 dimensional matrix."""
    x, y = compute_rotation_matrix_2d([0, 0], [0, 1], 90 * np.pi / 180)

    assert x == -1.0
    assert round(y, 2) == 0.00


def test_get_cartesian():
    """Test the conversion from latitude and longitude to meters."""
    lat = [48.837787, 48.836857]
    lon = [2.419519, 2.420356]
    x, y, _ = get_cartesian(lat, lon)
    assert x[0] == 0
    assert y[0] == 0
    assert x[1] == pytest.approx(75.1965, 1e-4)
    assert y[1] == pytest.approx(64.4912, 1e-4)


def test_non_uniform_power_spectrum(non_uniform_data):
    """Test the power spectrum for non uniformly sampled data."""
    result = non_uniform_power_spectrum(non_uniform_data)
    assert result.size == 750


def test_extract_sampling_frequency(non_uniform_data_bis):
    """Test the extract_sampling_frequency function."""
    fs = extract_sampling_frequency(non_uniform_data_bis)
    assert round(fs) == 318


def test_discretize_frequency():
    """Test the consistency of the automatic sampling frequency retrivial."""
    indexes = pd.date_range(start=datetime(2000, 1, 1), periods=50, freq="22ms")
    data = pd.DataFrame({"dt": indexes}).set_index("dt")
    new_freq = discretize_sampling_frequency(data, [20, 50])

    assert new_freq == 50

    example_40_hz = pd.date_range(start=datetime(2000, 1, 1), periods=50, freq="25ms")
    example_df_40_hz = pd.DataFrame({"dt": example_40_hz}).set_index("dt")
    with pytest.raises(ValueError):
        discretize_sampling_frequency(example_df_40_hz, [20, 50])

    discrete_freq_example_df_40hz = discretize_sampling_frequency(
        example_df_40_hz, [20, 50], max_frequency_distance=15
    )
    assert discrete_freq_example_df_40hz == 50


@pytest.mark.xfail
def test_cross_corr():
    """Test :func:`dispel.signal.core.cross_corr`."""
    raise NotImplementedError


@pytest.mark.xfail
def test_autocorr():
    """Test :func:`dispel.signal.core.autocorr`."""
    raise NotImplementedError


@pytest.mark.xfail
def test_autocov():
    """Test :func:`dispel.signal.core.autocov`."""
    raise NotImplementedError


@pytest.mark.xfail
def test_scale_corr():
    """Test :func:`dispel.signal.core.scale_corr`."""
    raise NotImplementedError
