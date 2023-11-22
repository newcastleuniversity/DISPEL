"""signal.vectorial module.

A module containing common vector operations
"""
import numpy as np
import pandas as pd


def resultant_norm_planar(comp1: pd.Series, comp2: pd.Series) -> pd.Series:
    """Compute the norm of the resultant of a 2-dimensional vector on a plane.

    The norm of the resultant of 2-components represents the magnitude of a
    2-dimensional vector.

    Parameters
    ----------
    comp1
        The first component of the signal
    comp2
        The second component of the signal

    Returns
    -------
    pd.Series
        A series comprising the norm values of the resultant of 2-dimensional
        vectorial timeseries
    """
    return np.sqrt(comp1**2 + comp2**2)


def mean_norm_planar(comp1: pd.Series, comp2: pd.Series) -> float:
    """Compute the mean norm of a 2-dimensional timeseries.

    The mean norm of a 2-dimensional timeseries is referred to as the Average
    Acceleration Amplitude eq. A2 of Martinez(2012)
    https://doi.org/10.1080/10255842.2011.565753

    Parameters
    ----------
    comp1
        The first component of the signal
    comp2
        The second component of the signal

    Returns
    -------
    float
        The average value of the norm of a 2 dimensional timeseries
    """
    return resultant_norm_planar(comp1, comp2).mean()


def rms_planar(comp1: pd.Series, comp2: pd.Series) -> float:
    """Compute the RMS of a 2-dimensional timeseries.

    The Root-Mean-Square of a 2-dimensional timeseries as presented in eq. A4 of
    Martinez(2012) https://doi.org/10.1080/10255842.2011.565753

    Parameters
    ----------
    comp1
        The first component of the signal
    comp2
        The second component of the signal

    Returns
    -------
    float
        The RMS value of a 2-dimensional timeseries
    """
    return np.sqrt(np.mean(comp1**2 + comp2**2))
