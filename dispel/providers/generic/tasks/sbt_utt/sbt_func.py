"""Functionality implemented in SBT.steps module."""
import numpy as np
import pandas as pd

from dispel.providers.generic.tasks.sbt_utt.const import MIN_MOTION_DUR
from dispel.signal.core import signal_duration
from dispel.signal.geometric import extract_ellipse_axes
from dispel.signal.vectorial import mean_norm_planar, resultant_norm_planar, rms_planar


def label_bouts(data: pd.Series) -> pd.Series:
    """Label each valid and invalid chunk as a bout.

    Parameters
    ----------
    data
        A Series that contains one column including the flag continuous
        signal

    Returns
    -------
    Series
        A labelled pd.Series where each valid/invalid bout is assigned an
        increasing integer number
    """
    # We increase a counter number everytime the flag changes (solution
    # inspired in StakOverflow community
    return data.astype(bool).diff().fillna(method="bfill").cumsum()


def reject_short_bouts(bout_mask: pd.Series, flag: pd.Series) -> pd.Series:
    """Reject bouts whose duration is less than MIN_MOTION_DUR seconds.

    Parameters
    ----------
    bout_mask
        A Series containing a flag_signal and a bout_number.
    flag
        A Series containing a flag_signal and a bout_number.

    Returns
    -------
    Series
        A Series with a flag_signal where the valence has been inverted
        in case its duration is below MIN_MOTION_DUR seconds.

    """
    flag = flag.astype(bool)
    for _, bout in bout_mask.groupby(bout_mask):
        if signal_duration(bout) < MIN_MOTION_DUR:
            flag.loc[bout.index] = ~flag.loc[bout.index]
    return flag


def data_coverage_fraction(data: pd.Series) -> float:
    """Compute the portion of data covered by a binary flag over signal length.

    Parameters
    ----------
    data
        A binary Series flag (0/1) that flags some behaviour of interest.

    Returns
    -------
    float
        A value between 0 and 1 that represents how much of the data
        flag signal covers the totality of the recording.
    """
    return round(float(1 - len(data[data.values]) / len(data)), 2)


def sway_total_excursion(comp1: pd.Series, comp2: pd.Series) -> float:
    """Compute the total amount of acceleration increments on a plane.

    The Sway Total Excursion (a.k.a. TOTEX) provides a means of quantifying
    how much sway occurred on a plane over the total of an assessment.
    It is a complementary measure to the sway areas, as it covers also the
    amount of sway occurred within a given geometric area.

    Parameters
    ----------
    comp1
        The first component of the signal
    comp2
        The second component of the signal

    Returns
    -------
    float
        The sway total excursion value
    """
    # We take the sum of the norm of all acceleration increments
    return resultant_norm_planar(comp1.diff(), comp2.diff()).sum()


def sway_jerk(comp1: pd.Series, comp2: pd.Series) -> float:
    """Compute the jerk of the sway.

    The Sway Jerk provides a means of quantifying the average sway occurred
    on a plane per time unit over the total of an assessment .
    It is a complementary measure to the sway areas, as it covers also the
    amount of sway occurred within a given geometric area. It takes special
    relevance when algorithms to remove outliers are applied and the
    timeseries span used for different features is different. In other
    words, a normalised version of the sway total excursion. See an example
    of concomitant use with sway total excursion in Mancini(2012),
    https://doi.org/10.1186/1743-0003-9-59

    Parameters
    ----------
    comp1
        The first component of the signal
    comp2
        The second component of the signal

    Returns
    -------
    float
        The sway total excursion value
    """
    total_excursion = sway_total_excursion(comp1, comp2)

    trial_time = (comp1.last_valid_index() - comp1.first_valid_index()).seconds

    return total_excursion / trial_time


def circle_area(comp1: pd.Series, comp2: pd.Series) -> float:
    """Compute the area of a circle comprising data within the 95-percentile.

    The area of the circle comprising data points of a timeseries within the
    95-percentile is computed following Original paper (eq.12) by
    Prieto(1996) https://doi.org/10.1109/10.532130

    Parameters
    ----------
    comp1
        The first component of the signal
    comp2
        The second component of the signal

    Returns
    -------
    float
        The value of the estimated area covering the 95-percentile of the data
        of a 2-dimensional timeseries.
    """
    # Being z095 the z-statistic at the 95% confidence level
    z095 = 1.645

    # Based on the average acceleration magnitude and the root-mean-square
    # acceleration
    rmsa_r = rms_planar(comp1, comp2)
    aam_r = mean_norm_planar(comp1, comp2)

    # Computing the Standard Deviation of the 2-dimensional timeseries as in
    # (eq.13) in Prieto(1996)
    std_resultant = np.sqrt(rmsa_r**2 - aam_r**2)

    # To obtain the area of a circle that includes 95% of the resultant
    # norms of the acceleration (see (eq.12) in Prieto(1996))
    return np.pi * (aam_r + z095 * std_resultant) ** 2


def ellipse_area(comp1: pd.Series, comp2: pd.Series) -> float:
    """Compute Ellipse Area as the 95% percentile of the PCA-fitted axes.

    The Ellipse Area is computed from the values of the estimated minor and
    major (a,b) axes of an ellipse. The axes are estimated using a 2-PCA
    component analyses on the 2-dimensional timeseries of data. The Ellipse
    area is computed as the multiplication of its semi-axes and the number pi.

    Parameters
    ----------
    comp1
        The first component of the signal
    comp2
        The second component of the signal

    Returns
    -------
    float
        The value of the estimated ellipse area covering the 95-percentile of
        the data of a 2-dimensional timeseries.
    """
    df = pd.concat([comp1, comp2], axis=1)
    # The axes are estimated using a 2-PCA component analyses on
    # the 2-dimensional timeseries of data.
    a, b = extract_ellipse_axes(df)
    # The Ellipse area is computed as the multiplication of its semi-axes
    # and the number pi.
    return np.pi * (a / 2) * (b / 2)
