"""Core functionality for signal processing tasks."""
import math
import warnings
from typing import Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from gatspy.periodic import LombScargle
from scipy import signal


def euclidean_norm(data: pd.DataFrame) -> pd.Series:
    """Calculate the euclidean norm of a pandas Data Frame.

    Parameters
    ----------
    data
        A pandas data frame for which to compute the euclidean norm

    Returns
    -------
    pandas.Series
        The euclidean norm of ``data``
    """
    return data.pow(2).sum(axis=1).apply(np.sqrt)


def euclidean_distance(point1: Iterable[float], point2: Iterable[float]) -> float:
    """Calculate the euclidean distance between two points.

    This particular algorithm is chosen based on question 37794849 on StackOverflow.

    Parameters
    ----------
    point1
        The coordinates of `point1`.
    point2
        The coordinates of `point2`.

    Returns
    -------
    float
        The euclidean distance between ``point1`` and ``point2``.
    """
    dists = [(a - b) ** 2 for a, b in zip(point1, point2)]
    dist = math.sqrt(sum(dists))
    return dist


def compute_rotation_matrix_3d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute a rotation matrix from unit vector ``a`` onto unit vector ``b``.

    Parameters
    ----------
    a
        The first unit vector
    b
        The second unit vector

    Returns
    -------
    numpy.ndarray
        The rotation matrix from unit vector ``a`` onto unit vector ``b``.

    See Also
    --------
        Implementation inspired by https://math.stackexchange.com/a/476311 and adapted
        to handle a series of rotations
    """
    assert a.shape == (3,)
    assert b.shape == (3,)

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    # make sure we have norm vectors!
    if a_norm != 1:
        a = a / a_norm
    if b_norm != 1:
        b = b / b_norm

    v = np.cross(a, b)
    c = np.dot(a, b)
    i = np.identity(3)

    # the vectors are equal - no rotation needed
    if (a == b).all():
        return i

    # the vectors are in opposite direction
    if c == -a_norm * b_norm:
        return i * -1

    v_x = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    r = i + v_x + np.matmul(v_x, v_x) * 1 / (1 + c)

    return r


def compute_rotation_matrix_2d(
    origin: list, point: list, angle: float
) -> Tuple[float, float]:
    """Compute a 2 dimension matrix rotation.

    Parameters
    ----------
    origin
        The origin of the referential.
    point
        The point to rotate.
    angle
        The angle rotation.

    Returns
    -------
    Tuple[float, float]
        The x and y coordinates after rotation.
    """
    origin_x, origin_y = origin
    point_x, point_y = point

    rotated_x = (
        origin_x
        + math.cos(angle) * (point_x - origin_x)
        - math.sin(angle) * (point_y - origin_y)
    )
    rotated_y = (
        origin_y
        + math.sin(angle) * (point_x - origin_x)
        + math.cos(angle) * (point_y - origin_y)
    )

    return rotated_x, rotated_y


def sparc(
    movement: np.ndarray,
    sample_freq: float = 60.0,
    pad_level: int = 4,
    cut_off_freq: float = 10.0,
    amp_th: float = 0.05,
) -> Tuple[float, tuple, tuple]:
    """Compute the spectral arc length of a signal.

    Parameters
    ----------
    movement
        The given 1 dimensional signal (x, y or z axis).
    sample_freq
        The sampling rate.
    pad_level
        The padding level.
    cut_off_freq
        The frequency cut off.
    amp_th
        The amplitude threshold.

    Returns
    -------
    new_sal: float
        The spectral arc length value.
    (freq, mag_spec): tuple
        A tuple containing both frequencies and the normalized magnitude spectrum of the
        movement.
    (freq_sel, mag_spec_sel): tuple
        A tuple containing both frequencies (after applying a cutoff) and the normalized
        magnitude spectrum (after applying a cutoff) of the movement.
    """
    # Number of zeros to be padded.
    n_fft = int(pow(2, np.ceil(np.log2(len(movement))) + pad_level))

    # Frequency
    freq = np.arange(0, sample_freq, sample_freq / n_fft)
    # Normalized magnitude spectrum
    mag_spec = abs(np.fft.fft(movement, n_fft))
    mag_spec = mag_spec / max(mag_spec)
    # Indices to choose only the spectrum within the given cut-off frequency
    # Fc.
    # NOTE: This is a low pass filtering operation to get rid of high frequency
    # noise from affecting the next step (amplitude threshold based cut off for
    # arc length calculation).
    cut_off_freq_inx = ((freq <= cut_off_freq) * 1).nonzero()
    freq_sel = freq[cut_off_freq_inx]
    mag_spec_sel = mag_spec[cut_off_freq_inx]
    # Choose the amplitude threshold based cut-off frequency.
    # Index of the last point on the magnitude spectrum that is greater than
    # or equal to the amplitude threshold.
    inx = ((mag_spec_sel >= amp_th) * 1).nonzero()[0]
    cut_off_freq_inx_range = range(inx[0], inx[-1] + 1)
    freq_sel = freq_sel[cut_off_freq_inx_range]
    mag_spec_sel = mag_spec_sel[cut_off_freq_inx_range]
    # Calculate arc length
    new_sal = -sum(
        np.sqrt(
            pow(np.diff(freq_sel) / (freq_sel[-1] - freq_sel[0]), 2)
            + pow(np.diff(mag_spec_sel), 2)
        )
    )
    return new_sal, (freq, mag_spec), (freq_sel, mag_spec_sel)


def extract_sampling_frequency(data: pd.Series) -> float:
    """Extract the median sampling frequency in Hz of a time series.

    Parameters
    ----------
    data
        Any pandas series with a time series as index.

    Returns
    -------
    float
        The sampling frequency

    Raises
    ------
    ValueError
        if any difference in consecutive timestamps is null.
    """
    try:
        return 1 / index_time_diff(data).median()
    except ZeroDivisionError as error:
        raise ValueError(
            "Difference in consecutive timestamps cannot be null."
        ) from error


def assert_time_series_has_frequency(data: pd.Series) -> None:
    """Check whether a time series contains frequency as index."""
    if data.index.name != "freq":
        raise ValueError("Missing frequencies in data index.")


def get_sampling_rate_idx(data: pd.Series) -> float:
    """Get sampling rate from time series index.

    Parameters
    ----------
    data
        A pandas series containing the signal data for which sampling frequency is to be
        extracted.

    Returns
    -------
    float
        The sampling frequency.

    Raises
    ------
    ValueError
        Raises a value error if data has not been resampled to a constant sampling rate.
    """
    if data.index.freq is None:
        raise ValueError(
            "One is trying to extract the sampling frequency on a time series that has "
            "not been resampled to a constant sampling rate."
        )
    return 1 / data.index.freq.delta.total_seconds()


def uniform_power_spectrum(data: pd.Series) -> pd.Series:
    """Compute the power spectrum of a signal.

    Parameters
    ----------
    data
        An pandas series containing the signal data for which the power spectrum is to
        be computed.

    Returns
    -------
    pandas.Series
        Two arrays, one containing the signal's frequency and the other the power
        spectrum.
    """
    freq = get_sampling_rate_idx(data)
    freqs, ps_data = signal.welch(
        data, nperseg=min(256, len(data)), fs=freq, scaling="spectrum"
    )
    ps_data = pd.Series(ps_data, index=freqs).apply(np.real)
    ps_data.index = ps_data.index.set_names("freq")
    return ps_data


def non_uniform_power_spectrum(
    data: pd.Series,
):
    """Compute the power spectrum for non uniformly sampled data.

    The algorithm for default frequencies and converting them to angular frequencies is
    taken from
    https://jakevdp.github.io/blog/2015/06/13/lomb-scargle-in-python/. Notably, we have
    decided to use `gatspy` rather than `scipy`.

    Parameters
    ----------
    data
        An pandas series containing the signal data for which the power spectrum is to
        be computed, indexed by time

    Returns
    -------
    pandas.Series
        One Dataframe containing the signal's periodogram indexed by frequencies.
    """
    indexes = (data.index - data.index[0]).to_series().dt.total_seconds()
    model = LombScargle().fit(indexes, data.values.squeeze().astype(float))
    period, power = model.periodogram_auto()

    pd_data = pd.Series(data=power, index=1.0 / period)
    pd_data.index = pd_data.index.set_names("freq")

    return pd_data


def entropy(power_spectrum_: np.ndarray) -> float:
    """Compute the entropy of a signal.

    Parameters
    ----------
    power_spectrum_
        An array containing the power spectrum of the signal in question.

    Returns
    -------
    float
        The signal's entropy.
    """
    data = power_spectrum_ / np.sum(power_spectrum_)
    return -np.sum(data * np.log2(data))


def peak(
    power_spectrum_: pd.Series,
) -> float:
    """Compute the peak frequency of a signal.

    Parameters
    ----------
    power_spectrum_
        An array containing the power spectrum of the signal in question.

    Returns
    -------
    float
        The signal's peak frequency.
    """
    assert_time_series_has_frequency(power_spectrum_)
    return power_spectrum_.idxmax()


def energy(
    power_spectrum_: pd.Series,
    lowcut: Optional[float] = None,
    highcut: Optional[float] = None,
) -> float:
    """Compute the energy of a signal.

    Parameters
    ----------
    power_spectrum_
        A pandas series containing the power spectrum of the signal in question and the
        frequencies in index.
    lowcut
        The lower bound of frequencies to filter.
    highcut
        The higher bound of frequencies to filter.

    Returns
    -------
    float
        The signal's energy.
    """
    assert_time_series_has_frequency(power_spectrum_)
    if lowcut is not None:
        mask = power_spectrum_.index.to_series().between(lowcut, highcut)
        windowed_data = power_spectrum_[mask]
    else:
        windowed_data = power_spectrum_
    return 0.5 * (windowed_data * windowed_data.index).sum()


def amplitude(power_spectrum_: np.ndarray) -> float:
    """Compute the amplitude of a signal.

    Parameters
    ----------
    power_spectrum_
        An array containing the power spectrum of the signal in question.

    Returns
    -------
    float
        The signal's amplitude.
    """
    return np.max(power_spectrum_)


def get_cartesian(
    lat: Sequence[float], lon: Sequence[float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform latitude longitude to cartesian coordinates in meters."""
    latitude, longitude = np.deg2rad(np.array(lat)), np.deg2rad(np.array(lon))
    earth_r = 6371000  # radius of the earth in meters
    x = earth_r * np.cos(latitude) * np.cos(longitude)
    y = earth_r * np.cos(latitude) * np.sin(longitude)
    z = earth_r * np.sin(latitude)
    return x - x[0], y - y[0], z - z[0]


def integrate_time_series(data: pd.Series) -> np.ndarray:
    """Compute the integral of a time series.

    Parameters
    ----------
    data
        Input series to integrate. Must have a DatetimeIndex.

    Returns
    -------
    numpy.ndarray
        Definite integral as approximated by trapezoidal rule.

    Raises
    ------
    TypeError
        Raises a type error if the index of the series to integrate is not a
        DateTimeIndex.
    """
    index = data.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(
            "The time series to integrate must have a "
            f"DateTimeIndex. But index is of type {type(index)}."
        )
    return np.trapz(data.to_numpy().squeeze(), pd.to_numeric(index) / 10**9)


def index_time_diff(data: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    """Get the time difference from the index in seconds.

    Parameters
    ----------
    data
        The series or data frame with a time-based index

    Returns
    -------
    pandas.Series
        A series containing the time difference in seconds between each row based on the
        index.
    """
    assert isinstance(
        data.index, (pd.DatetimeIndex, pd.TimedeltaIndex)
    ), "Index must be a pandas DatetimeIndex or TimedeltaIndex"

    return data.index.to_series().diff().dt.total_seconds()


def derive_time_series(data: pd.Series) -> pd.Series:
    """Derive a series based on its time-based index.

    Parameters
    ----------
    data
        The series for which to derive the values based on time. The series must have a
        time-based index. See :func:`index_time_diff`.

    Returns
    -------
    pandas.Series
        The time derivative of the values of ``data``.
    """
    return data.diff() / index_time_diff(data)


def derive_time_data_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Derive a data frame based on its time-based index.

    This method is preferably used for data frames for which one wants to derive for
    each column (instead of using ``data.apply(derive_time_series)``).

    Parameters
    ----------
    data
        The pandas data frame for which to derive the values based on time. The data
        frame must have a time-based index. See :func:`index_time_diff`.

    Returns
    -------
    pandas.DataFrame
        The time derivative of the values of ``data``.

    """
    delta_t = index_time_diff(data)
    return (data.diff().T / delta_t).T


def discretize_sampling_frequency(
    data: pd.Series, fs_expected: List[int], max_frequency_distance: int = 5
) -> int:
    """Discretize the sampling frequency from a time series.

    First we extract the median sampling frequency of data, then return the closest
    expected frequency if the estimated sampling frequency is close enough
    (``np.abs(fs_expected, fs_estimate) < 5``) to one of the expected sampling
    frequencies.

    Parameters
    ----------
    data
        Any pandas series with a time series as index.
    fs_expected
        An iterable of expected sampling frequency in Hz.
    max_frequency_distance
        An optional integer specifying the maximum accepted distance between the
        expected frequency and the estimated frequency above which we raise an error.

    Returns
    -------
    int
        Discretized sampling frequency.

    Raises
    ------
    ValueError
        If estimated sampling frequency is too far
        (abs distance > max_frequency_distance) from all the expected sampling frequency
        in ``fs_expected``.
    """
    # Estimate sampling_frequency
    fs_estimate = extract_sampling_frequency(data)

    # Compute the distance to expected frequencies
    frequency_distance = np.abs(np.array(fs_expected) - fs_estimate)

    # Check if we are close enough otherwise raise a warning
    if min(frequency_distance) > max_frequency_distance:
        raise ValueError(
            f"Estimated sampling frequency {fs_estimate} is further than 5 Hz from "
            f"expected frequencies: {fs_expected}."
        )
    if min(frequency_distance) > 1:
        warnings.warn(
            f"Estimated sampling frequency {fs_estimate} is further than 1 Hz from "
            f"expected frequencies: {fs_expected}."
        )
    return fs_expected[int(np.argmin(frequency_distance))]


def cross_corr(
    data_1: npt.NDArray[np.float64], data_2: npt.NDArray[np.float64]
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute cross-correlation between two signals.

    Uses cross-correlation with the same signal as input twice.

    Parameters
    ----------
    data_1
        A first signal passed as an iterable of floats.
    data_2
        A second signal passed as an iterable of floats.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        A Tuple containing temporal delays and auto-correlation array.
    """
    n_samples = len(list(data_1))
    corr = np.correlate(data_1, data_2, mode="full")
    lags = np.arange(-(n_samples - 1), n_samples)
    return lags, corr


def autocov(data: npt.NDArray[np.float64]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute auto-covariance of a signal.

    Uses autocovariance from as autocorrelation of demeaned signal.

    Parameters
    ----------
    data
        A signal passed as an iterable of floats.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        A Tuple containing temporal delays  and auto-covariance array.
    """
    # compute autocovariance
    return autocorr(data - np.mean(data))


def autocorr(data: npt.NDArray[np.float64]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute auto-correlation of a signal.

    Uses cross-correlation with the same signal as input twice.

    Parameters
    ----------
    data
        A signal passed as an iterable of floats.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        A Tuple containing temporal delays  and auto-correlation array.
    """
    return cross_corr(data, data)


def scale_corr(
    corr: np.ndarray,
    n_samples: int,
    phase_shift: np.ndarray,
    method: Literal["biased", "unbiased"],
) -> np.ndarray:
    """Scale auto-correlation.

    Parameters
    ----------
    corr
        An iterable of floats containing the correlations for each delay.
    n_samples
        The number of samples of the input signal (denoted N in equation).
    phase_shift
        The phase shift in number of samples (denoted m in equation).
    method
        the method to be used (biased or unbiased)

    Returns
    -------
    numpy.ndarray
        An array containing the scaled auto-correlation values.
    """
    if method == "biased":
        corr = corr / n_samples
    elif method == "unbiased":
        corr /= n_samples - abs(phase_shift)

    return corr


def scaled_autocorr(
    data: npt.NDArray[np.float64],
    method: Literal["unbiased"] = "unbiased",
    do_autocov: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute scaled auto-correlation function of a signal.

    Parameters
    ----------
    data
        An iterable signal of floats.
    method
        A string defining the scaling method to be used.
    do_autocov
        A boolean denoting whether autocovariance should be used for calculation of the
        autocorrelation function.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        A Tuple containing lags and auto-correlation output..
    """
    if do_autocov:
        lags, corr = autocov(data)
    else:
        lags, corr = autocorr(data)

    corr = scale_corr(corr, len(list(data)), lags, method)
    return lags, corr


def signal_duration(data: Union[pd.Series, pd.DataFrame]) -> float:
    """Get signal duration from time-based indices.

    Parameters
    ----------
    data
        The signal of which we want to compute the duration based on its index. The
        index has to be either a TimedeltaIndex or DatetimeIndex.

    Returns
    -------
    float
        The duration of the signal (in seconds) from the index.
    """
    assert isinstance(data.index, (pd.TimedeltaIndex, pd.DatetimeIndex))
    return (data.index.max() - data.index.min()).total_seconds()
