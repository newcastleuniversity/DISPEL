"""Signal filtering functions."""
from typing import List, Optional, Union

import pandas as pd
from scipy import signal


def _butterworth_filter(
    data: pd.Series,
    filter_type: str,
    cutoff: Union[float, List[float]],
    order: int = 2,
    freq: Optional[float] = None,
    zero_phase: Optional[bool] = False,
) -> pd.Series:
    # determine frequency from series if available
    if hasattr(data.index, "freq") and data.index.freq:
        freq = 1e9 / data.index.freq.nanos
    if freq is None:
        raise ValueError(
            "Sampling rate can only be determined from fixed 'pandas time series "
            'indices. Please specify "fs".'
        )

    # create filter
    nyq = 0.5 * freq

    if isinstance(cutoff, (int, float)):
        normal_cutoff: Union[float, List[float]] = cutoff / nyq
    elif isinstance(cutoff, list):
        normal_cutoff = [val / nyq for val in cutoff]
    else:
        raise TypeError(f"Unsupported cutoff type: {type(cutoff)}")

    b, a = signal.butter(order, normal_cutoff, btype=filter_type, analog=False)

    # apply filter depending on phase profile
    if zero_phase:
        data_filtered = signal.filtfilt(b, a, data.values)
    else:
        data_filtered = signal.lfilter(b, a, data.values)

    res = pd.Series(data_filtered, index=data.index, name=data.name)

    if not zero_phase:
        res = res.shift(-order)

    return res


def butterworth_low_pass_filter(
    data: pd.Series,
    cutoff: float,
    order: int = 2,
    freq: Optional[float] = None,
    zero_phase: Optional[bool] = True,
) -> pd.Series:
    """Filter a series with a butterworth low-pass filter.

    Parameters
    ----------
    data
        The time series to be filtered
    cutoff
        The lower bound of frequencies to filter
    order
        The order of the filter
    freq
        The sampling frequency of the time series in Hz. If the passed ``data`` has
        an evenly spaced time series index it will be determined automatically.
    zero_phase
        Boolean indicating whether zero phase filter (filtfilt) to be used

    Returns
    -------
    pandas.Series
        The filtered ``data``.
    """
    return _butterworth_filter(data, "low", cutoff, order, freq, zero_phase)


def butterworth_high_pass_filter(
    data: pd.Series,
    cutoff: float,
    order: int = 2,
    freq: Optional[float] = None,
    zero_phase: Optional[bool] = True,
) -> pd.Series:
    """Filter a series with a butterworth high-pass filter.

    Parameters
    ----------
    data
        The time series to be filtered
    cutoff
        The upper bound of frequencies to filter
    freq
        The sampling frequency of the time series in Hz. If the passed ``data`` has
        an evenly spaced time series index it will be determined automatically.
    order
        The order of the filter
    zero_phase
        Boolean indicating whether zero phase filter (filtfilt) to be used

    Returns
    -------
    pandas.Series
        The filtered ``data``.
    """
    return _butterworth_filter(data, "high", cutoff, order, freq, zero_phase)


def butterworth_band_pass_filter(
    data: pd.Series,
    lower_bound: float,
    upper_bound: float,
    order: int = 2,
    freq: Optional[float] = None,
    zero_phase: Optional[bool] = True,
) -> pd.Series:
    """Filter a series with a butterworth band-pass filter.

    Parameters
    ----------
    data
        The time series to be filtered
    lower_bound
        The lower bound of frequencies to filter
    upper_bound
        The upper bound of frequencies to filter
    freq
        The sampling frequency of the time series in Hz. If the passed ``data`` has
        an evenly spaced time series index it will be determined automatically.
    order
        The order of the filter
    zero_phase
        Boolean indicating whether zero phase filter (filtfilt) to be used

    Returns
    -------
    pandas.Series
        The filtered ``data``.
    """
    return _butterworth_filter(
        data, "band", [lower_bound, upper_bound], order, freq, zero_phase
    )


def savgol_filter(data: pd.Series, window: int = 41, order: int = 3) -> pd.Series:
    """Apply the Savitzky-Golay filter on a class:`~pandas.Series`.

    Parameters
    ----------
    data
        Input data
    window
        the length of the filter window
    order
        The order of the polynomial used to fit the samples

    Returns
    -------
    pandas.Series
        Filtered data
    """
    # apply filter.
    res = pd.Series(
        signal.savgol_filter(data, window, order), index=data.index, name=data.name
    )
    return res
