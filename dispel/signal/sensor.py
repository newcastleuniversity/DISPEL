"""Sensor functionality for signal processing tasks."""

import pandas as pd
from scipy import stats

#: A dictionary for sensor units.
SENSOR_UNIT = {"acc": "G", "gyr": "rad/s", "diss": "pixel"}


def detrend_signal(signal: pd.Series) -> pd.Series:
    """Detrend signal and remove offset component.

    The final signal will end up centered on zero and stationary. This function is based
    on :func:`scipy.stats.linregress`.

    Parameters
    ----------
    signal: pandas.Series
        The raw signal.

    Returns
    -------
    pandas.Series
        The detrended signal.
    """
    original_x = signal.index.to_numpy(float)
    signal_without_na = signal.dropna()
    y = signal_without_na.to_numpy(float)
    x = signal_without_na.index.to_numpy(float)
    (
        slope,
        intercept,
        *_,
    ) = stats.linregress(x, y)
    y_estimate = slope * original_x + intercept
    return signal - y_estimate


def check_amplitude(
    data: pd.DataFrame, min_amplitude: float, max_amplitude: float
) -> bool:
    """Check if the signal amplitudes belong to a reasonable range.

    The function will return true only if all the values of each column are between the
    min and max amplitude bounds.

    Parameters
    ----------
    data
        A data frame containing one column or more. The data contains in columns must
        all have the same nature as the bounds are applied on the entire data frame.
    min_amplitude
        The expected min amplitude.
    max_amplitude
        The expected max amplitude.

    Returns
    -------
    bool
        ``True`` if all the values are in the range. ``False`` otherwise.
    """
    amplitude = data.max() - data.min()
    return amplitude.between(left=min_amplitude, right=max_amplitude).all()


def find_zero_crossings(data: pd.DataFrame, col: str) -> pd.DataFrame:
    """Find zero crossing in the signal."""
    zero_crossings = data.index[(data[col] > 0).diff().fillna(False)]
    return data.loc[zero_crossings, col]
