r"""Reliability analyses module.

Intraclass correlation coefficients
===================================

This module contains functions to compute intraclass correlation coefficient (ICC)
models\ [1]_\ [2]_\ .

Those functions allow to compute single score or average score ICCs as an index of
inter-rater reliability of quantitative data. Additionally, F-test and confidence
interval are computed.

.. list-table:: Implemented ICCs
    :header-rows: 1
    :stub-columns: 1

    * - ICC
      - :class:`Model <ICCModel>`
      - :class:`Description <ICCDesc>`
      - :class:`Unit <ICCUnit>`
      - Function
    * - ICC(1, 1)
      - :data:`~ICCModel.ONE_WAY`
      - :data:`~ICCDesc.AGREEMENT`
      - :data:`~ICCUnit.SINGLE`
      - :func:`icc_oneway_random_absolute_single`
    * - ICC(2, 1)
      - :data:`~ICCModel.TWO_WAY`
      - :data:`~ICCDesc.AGREEMENT`
      - :data:`~ICCUnit.SINGLE`
      - :func:`icc_two_way_random_absolute_single`
    * - ICC(3,1)
      - :data:`~ICCModel.TWO_WAY`
      - :data:`~ICCDesc.CONSISTENCY`
      - :data:`~ICCUnit.SINGLE`
      - :func:`icc_two_way_mixed_consistency_single`
    * - ICC(1,k)
      - :data:`~ICCModel.ONE_WAY`
      - :data:`~ICCDesc.AGREEMENT`
      - :data:`~ICCUnit.AVERAGE`
      - :func:`icc_oneway_random_absolute_average`
    * - ICC(2,k)
      - :data:`~ICCModel.TWO_WAY`
      - :data:`~ICCDesc.AGREEMENT`
      - :data:`~ICCUnit.AVERAGE`
      - :func:`icc_two_way_random_absolute_average`
    * - ICC(3,k)
      - :data:`~ICCModel.TWO_WAY`
      - :data:`~ICCDesc.CONSISTENCY`
      - :data:`~ICCUnit.AVERAGE`
      - :func:`icc_two_way_mixed_consistency_average`

When considering which form of ICC is appropriate for an actual set of data one has take
several decisions (Shrout & Fleiss, 1979)\ [3]_\ :

 - 1. Should only the subjects be considered as random effects
   (:data:`ICCModel.ONE_WAY`) or are subjects and raters randomly chosen from a bigger
   pool of persons (:data:`ICCModel.TWO_WAY`).
 - 2. If differences in judges' mean ratings are of interest, inter-rater
   :data:`ICCDesc.AGREEMENT` instead of :data:`ICCDesc.CONSISTENCY` should be computed.
 - 3. If the unit of analysis is a mean of several ratings, unit should be changed to
   :data:`ICCUnit.AVERAGE`. In most cases, however, single values
   (:data:`ICCUnit.SINGLE`) are regarded.

The implementations of the ICCs are based on:

- https://www.rdocumentation.org/packages/irr/versions/0.84.1/topics/icc
- https://www.rdocumentation.org/packages/psych/versions/1.9.12.31/topics/ICC

References
----------
.. [1] Bartko, J.J. (1966). The intraclass correlation coefficient as a measure of
   reliability. Psychological Reports, 19, 3-11.
.. [2] McGraw, K.O., & Wong, S.P. (1996), Forming inferences about some intraclass
   correlation coefficients. Psychological Methods, 1, 30-46.
.. [3] Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: uses in assessing
   rater reliability. Psychological bulletin, 86(2), 420.

"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Literal, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f as density

from dispel.data.collections import FeatureCollection


class StringEnum(Enum):
    """String enumerator."""

    def __str__(self) -> str:
        return str(self.value)


class ICCKind(StringEnum):
    """The kind of ICC analysis."""

    TEST_RETEST = "test retest"
    PARALLEL_FORM = "parallel form"
    OTHER = "other"


class ICCModel(StringEnum):
    """The model type used to perform the ICC analysis."""

    #: Random subject effects
    ONE_WAY = "oneway"

    #: Random subject and repetition effects
    TWO_WAY = "two-way"


class ICCDesc(StringEnum):
    """The ICC description."""

    #: Agreement between raters
    AGREEMENT = "agreement"

    #: Consistency between raters
    CONSISTENCY = "consistency"


class ICCUnit(StringEnum):
    """The ICC unit."""

    #: Values analyzed are single values
    SINGLE = "single"

    #: Values analyzed are aggregates of multiple values
    AVERAGE = "average"


@dataclass
class ICCResult:
    """ICC reliability analysis results."""

    #: The model of ICC
    model: ICCModel
    #: The description of ICC
    desc: ICCDesc
    #: The unit of ICC
    unit: ICCUnit
    #: The kind of the ICC
    kind: ICCKind
    #: The ICC value
    value: float
    #: The lower bound of the 95% confidence interval
    l_bound: float
    #: The upper bound of the 95% confidence interval
    u_bound: float
    #: The p-value of the test must be under 0.05
    p_value: float
    #: The subject sample size associated to the current study and ICC
    sample_size: int
    #: The number of sessions considered in the ICC analysis
    sessions: int
    #: The power of the study regarding the sample size and ICC
    power: float = 0.8


class ICCResultSetStudy(str, Enum):
    """The type of the study."""

    STUDY_CONTROL = "control"
    STUDY_CLINICAL = "clinical"


@dataclass
class ICCResultSet:
    """Class ensemble of ICC scores for multiple features."""

    #: Sort of study concerned either control or patient
    study: ICCResultSetStudy
    #: The null hypothesis reference for feature ICC scores
    p0_icc: float
    #: The ICC scores for each feature associated by their feature_id
    iccs: Dict[str, ICCResult] = field(default_factory=dict)

    def to_data_frame(self) -> pd.DataFrame:
        """Export ICC result set to a pandas data frame format."""
        return pd.DataFrame(
            dict(
                study=str(self.study),
                p0_icc=self.p0_icc,
                feature_id=key,
                icc_model=str(value.model),
                icc_description=str(value.desc),
                icc_unit=str(value.unit),
                icc_kind=str(value.kind),
                icc_value=value.value,
                lower_bound=value.l_bound,
                upper_bound=value.u_bound,
                p_value=value.p_value,
                sample_size=value.sample_size,
                sessions=value.sessions,
                power=value.power,
            )
            for key, value in self.iccs.items()
        )


def _get_z_alpha(p_value: float, tails: Literal[1, 2]) -> float:
    if tails == 2:
        return stats.norm.ppf(1 - p_value / 2)
    if tails == 1:
        return stats.norm.ppf(1 - p_value)

    raise ValueError("tails can only be 1 or 2")


def _get_f_p(k: int, icc: float, p0_icc: float) -> Tuple[float, float]:
    # Calculate Fp and Fp0
    f_p = (1 + (k - 1) * icc) / (1 - icc)
    f_p0 = (1 + (k - 1) * p0_icc) / (1 - p0_icc)

    return f_p, f_p0


def icc_sample_size(
    icc: float,
    p0_icc: float,
    n_ratings: int,
    p_value: float = 0.05,
    tails: Literal[1, 2] = 2,
    power: float = 0.8,
) -> int:
    """Compute the sample size for an ICC reliability scoring. See [4]_.

    Parameters
    ----------
    icc
        The ICC score expected during the study
    p0_icc
        The null hypothesis value of the expected ICC
    n_ratings
        The number of ratings for each subject
    p_value
        The desired ``p_value``. Set at ``0.05`` for statistical tests
    tails
        Unilateral (1) or Bilateral (2) test
    power
        The statistical power of the test. Always sets at 0.8 for clinical study.

    Returns
    -------
    int
        The sample size of the study

    References
    ----------
    .. [4] https://www.rdocumentation.org/packages/ICC.Sample.Size/versions/1.0
    """
    z_alpha = _get_z_alpha(p_value, tails)
    f_p, f_p0 = _get_f_p(n_ratings, icc, p0_icc)

    # Calculate N, rounded up to nearest integer
    n_raw = 1 + (2 * (z_alpha + stats.norm.ppf(power)) ** 2 * n_ratings) / (
        (np.log(f_p / f_p0)) ** 2 * (n_ratings - 1)
    )

    return np.ceil(n_raw)


def icc_power(
    icc: float,
    p0_icc: float,
    n_ratings: int,
    n_subjects: int,
    p_value: float = 0.05,
    tails: Literal[1, 2] = 2,
) -> float:
    """Compute the power of an ICC score obtained during a study.

    Parameters
    ----------
    icc
        The ICC obtained during the study
    p0_icc
        The null hypothesis value of the obtained ICC
    n_ratings
        The number of ratings for each subject
    n_subjects
        Number of subjects during the study
    p_value
        The ``p_value`` of the study. Should have been set at ``0.05``.
    tails
        Unilateral (1) or Bilateral (2) test.

    Returns
    -------
    float
        The power associates with the ICC study
    """
    z_alpha = _get_z_alpha(p_value, tails)
    f_p, f_p0 = _get_f_p(n_ratings, icc, p0_icc)

    # Calculate z_b
    z_b = (
        np.sqrt(((n_ratings - 1) * (n_subjects - 1)) / (2 * n_ratings))
        * (np.log(f_p / f_p0))
        - z_alpha
    )

    # Calculate power
    return np.round(stats.norm.cdf(z_b), 2)


@dataclass(frozen=True)
class _ICCParameters:
    """Ensemble of ICC parameter."""

    #: The number of subjects
    n_subjects: int
    #: The number of raters
    n_raters: int
    #: The error risk
    alpha: float
    #: The mean square for rows
    ms_r: float
    #: The mean square for residual sources of variance
    ms_w: float
    #: The mean square for columns
    ms_c: float
    #: The mean square for error
    ms_e: float


@dataclass(frozen=True)
class _ICCBounds:
    l_bound: float
    u_bound: float


def _icc_parameters(ratings: pd.DataFrame, confidence_level: float) -> _ICCParameters:
    """
    Compute the parameters of the ICC computations.

    Parameters
    ----------
    ratings
        Matrix with n subjects m raters
    confidence_level
        Confidence level of the interval.

    Raises
    ------
    ValueError
        If only one subject is given.

    Returns
    -------
    _ICCParameters
        The ensemble of ICC parameters

    """
    ratings = ratings.values
    n_subjects, n_raters = ratings.shape
    if n_subjects < 2:
        raise ValueError("One subject only. Add more subjects for ICC.")

    ss_total = ratings.var(ddof=1) * (n_subjects * n_raters - 1)
    alpha = 1 - confidence_level

    ms_r = ratings.mean(axis=1).var(ddof=1) * n_raters
    ms_w = (ratings.var(axis=1, ddof=1) / n_subjects).sum()
    ms_c = ratings.mean(axis=0).var(ddof=1) * n_subjects
    ms_e = (ss_total - ms_r * (n_subjects - 1) - ms_c * (n_raters - 1)) / (
        (n_subjects - 1) * (n_raters - 1)
    )

    return _ICCParameters(n_subjects, n_raters, alpha, ms_r, ms_w, ms_c, ms_e)


def _icc_single_confidence_interval(
    n_raters: int, alpha: float, f_value: float, df1: int, df2: int
) -> _ICCBounds:
    f_l = f_value / density.ppf(1 - alpha, df1, df2)
    f_u = f_value * density.ppf(1 - alpha, df2, df1)
    l_bound = (f_l - 1) / (f_l + (n_raters - 1))
    u_bound = (f_u - 1) / (f_u + (n_raters - 1))

    return _ICCBounds(l_bound, u_bound)


def icc_oneway_random_absolute_single(
    ratings: pd.DataFrame, confidence_level: float = 0.95
) -> ICCResult:
    """
    Compute the ICC(1,1) score.

    ICC(1,1) : One-way random effects, absolute agreement, single
    rater/measurement.

    Parameters
    ----------
    ratings
        Matrix with n subjects m raters, i.e. array-like, shape (n_subjects, n_raters)
    confidence_level
        Confidence level of the interval.

    Returns
    -------
    ICCResult
        ICC with its relative test information
    """
    params = _icc_parameters(ratings, confidence_level)

    icc_value = (params.ms_r - params.ms_w) / (
        params.ms_r + (params.n_raters - 1) * params.ms_w
    )

    df1 = params.n_subjects - 1
    df2 = params.n_subjects * (params.n_raters - 1)

    f_value = params.ms_r / params.ms_w
    p_value = 1 - density.cdf(f_value, df1, df2)

    bounds = _icc_single_confidence_interval(
        params.n_raters, params.alpha, f_value, df1, df2
    )

    return ICCResult(
        ICCModel.ONE_WAY,
        ICCDesc.AGREEMENT,
        ICCUnit.SINGLE,
        ICCKind.OTHER,
        icc_value,
        bounds.l_bound,
        bounds.u_bound,
        p_value,
        params.n_subjects,
        params.n_raters,
    )


def _icc_two_way_random_confidence_interval(
    alpha: float,
    ms_c: float,
    ms_e: float,
    ms_r: float,
    n_raters: int,
    n_subjects: int,
    v_d: float,
    v_n: float,
) -> Tuple[float, float]:
    v = v_n / v_d
    f_l = density.ppf(1 - alpha, n_subjects - 1, v)
    f_u = density.ppf(1 - alpha, v, n_subjects - 1)

    tmp = n_raters * ms_c + (n_raters * n_subjects - n_raters - n_subjects) * ms_e
    l_bound = (n_subjects * (ms_r - f_l * ms_e)) / (f_l * tmp + n_subjects * ms_r)
    u_bound = (n_subjects * (f_u * ms_r - ms_e)) / (tmp + n_subjects * f_u * ms_r)

    return l_bound, u_bound


def _icc_two_way_random_absolute_single_confidence_interval(
    n_subjects: int,
    n_raters: int,
    alpha: float,
    ms_r: float,
    ms_e: float,
    ms_c: float,
    icc_value: float,
) -> _ICCBounds:
    f_j = ms_c / ms_e
    v_n = (
        (n_raters - 1)
        * (n_subjects - 1)
        * (
            (
                n_raters * icc_value * f_j
                + n_subjects * (1 + (n_raters - 1) * icc_value)
                - n_raters * icc_value
            )
        )
        ** 2
    )
    v_d = (n_subjects - 1) * n_raters**2 * (icc_value**2) * f_j**2 + (
        n_subjects * (1 + (n_raters - 1) * icc_value) - n_raters * icc_value
    ) ** 2

    bounds = _icc_two_way_random_confidence_interval(
        alpha, ms_c, ms_e, ms_r, n_raters, n_subjects, v_d, v_n
    )
    return _ICCBounds(*bounds)


def icc_two_way_random_absolute_single(
    ratings: pd.DataFrame, confidence_level: float = 0.95
) -> ICCResult:
    """
    Compute the ICC(2,1) score.

    ICC(2,1): Two-way random effects, absolute agreement, single rater/measurement.

    Parameters
    ----------
    ratings
        Matrix with n subjects m raters, i.e. array-like, shape (n_subjects, n_raters)
    confidence_level
        Confidence level of the interval.

    Returns
    -------
    ICCResult
        ICC with its relative test information
    """
    params = _icc_parameters(ratings, confidence_level)

    icc_value = (params.ms_r - params.ms_e) / (
        params.ms_r
        + (params.n_raters - 1) * params.ms_e
        + (params.n_raters / params.n_subjects) * (params.ms_c - params.ms_e)
    )

    df1 = params.n_subjects - 1
    df2 = (params.n_subjects - 1) * (params.n_raters - 1)

    f_value = params.ms_r / params.ms_e
    p_value = 1 - density.cdf(f_value, df1, df2)

    bounds = _icc_two_way_random_absolute_single_confidence_interval(
        params.n_subjects,
        params.n_raters,
        params.alpha,
        params.ms_r,
        params.ms_e,
        params.ms_c,
        icc_value,
    )

    return ICCResult(
        ICCModel.TWO_WAY,
        ICCDesc.AGREEMENT,
        ICCUnit.SINGLE,
        ICCKind.OTHER,
        icc_value,
        bounds.l_bound,
        bounds.u_bound,
        p_value,
        params.n_subjects,
        params.n_raters,
    )


def icc_two_way_mixed_consistency_single(
    ratings: pd.DataFrame, confidence_level: float = 0.95
) -> ICCResult:
    """
    Compute the ICC(3,1) score.

    ICC(3,1): Two-way mixed effects, consistency, single rater/measurement.

    Parameters
    ----------
    ratings
        Matrix with n subjects m raters, i.e. array-like, shape (n_subjects, n_raters)
    confidence_level
        Confidence level of the interval.

    Returns
    -------
    ICCResult
        ICC with its relative test information
    """
    params = _icc_parameters(ratings, confidence_level)
    icc_value = (params.ms_r - params.ms_e) / (
        params.ms_r + (params.n_raters - 1) * params.ms_e
    )

    return _icc_two_way_mixed_consistency(ICCUnit.SINGLE, icc_value, params)


def _icc_oneway_random_absolute_average_confidence_interval(
    alpha: float, ms_r: float, ms_w: float, df1: int, df2: int
) -> _ICCBounds:
    f_l = (ms_r / ms_w) / density.ppf(1 - alpha, df1, df2)
    f_u = (ms_r / ms_w) * density.ppf(1 - alpha, df2, df1)
    l_bound = 1 - 1 / f_l
    u_bound = 1 - 1 / f_u
    return _ICCBounds(l_bound, u_bound)


def icc_oneway_random_absolute_average(
    ratings: pd.DataFrame, confidence_level: float = 0.95
) -> ICCResult:
    """
    Compute the ICC(1,k) score.

    ICC(1,k): One-way random effects, absolute agreement, multiple raters/measurements.

    Parameters
    ----------
    ratings
        Matrix with n subjects m raters, i.e. array-like, shape (n_subjects, n_raters)
    confidence_level
        Confidence level of the interval.

    Returns
    -------
    ICCResult
        ICC with its relative test information.
    """
    params = _icc_parameters(ratings, confidence_level)

    icc_value = (params.ms_r - params.ms_w) / params.ms_r

    df1 = params.n_subjects - 1
    df2 = params.n_subjects * (params.n_raters - 1)

    f_value = params.ms_r / params.ms_w
    p_value = 1 - density.cdf(f_value, df1, df2)

    bounds = _icc_oneway_random_absolute_average_confidence_interval(
        params.n_raters, params.alpha, f_value, df1, df2
    )

    return ICCResult(
        ICCModel.ONE_WAY,
        ICCDesc.AGREEMENT,
        ICCUnit.AVERAGE,
        ICCKind.OTHER,
        icc_value,
        bounds.l_bound,
        bounds.u_bound,
        p_value,
        params.n_subjects,
        params.n_raters,
    )


def _icc_two_way_random_absolute_average_confidence_interval(
    n_subjects: int, n_raters: int, alpha: float, ms_r: float, ms_e: float, ms_c: float
) -> _ICCBounds:
    icc2 = (ms_r - ms_e) / (
        ms_r + (n_raters - 1) * ms_e + (n_raters / n_subjects) * (ms_c - ms_e)
    )
    f_j = ms_c / ms_e
    v_n = (
        (n_raters - 1)
        * (n_subjects - 1)
        * (
            (
                n_raters * icc2 * f_j
                + n_subjects * (1 + (n_raters - 1) * icc2)
                - n_raters * icc2
            )
        )
        ** 2
    )
    tmp = (n_subjects - 1) * n_raters**2 * icc2**2 * f_j**2
    v_d = tmp + (n_subjects * (1 + (n_raters - 1) * icc2) - n_raters * icc2) ** 2

    lb2, ub2 = _icc_two_way_random_confidence_interval(
        alpha, ms_c, ms_e, ms_r, n_raters, n_subjects, v_d, v_n
    )

    l_bound = lb2 * n_raters / (1 + lb2 * (n_raters - 1))
    u_bound = ub2 * n_raters / (1 + ub2 * (n_raters - 1))
    return _ICCBounds(l_bound, u_bound)


def icc_two_way_random_absolute_average(
    ratings: pd.DataFrame, confidence_level: float = 0.95
) -> ICCResult:
    """
    Compute the ICC(2,k) score.

    ICC(2,k): Two-way random effects, absolute agreement, average raters/measurements.

    Parameters
    ----------
    ratings
        Matrix with n subjects m raters, i.e. array-like, shape (n_subjects, n_raters)
    confidence_level
        Confidence level of the interval.

    Returns
    -------
    ICCResult
        ICC with its relative test information.
    """
    params = _icc_parameters(ratings, confidence_level)

    icc_value = (params.ms_r - params.ms_e) / (
        params.ms_r + (params.ms_c - params.ms_e) / params.n_subjects
    )

    df1 = params.n_subjects - 1
    df2 = (params.n_subjects - 1) * (params.n_raters - 1)

    f_value = params.ms_r / params.ms_e
    p_value = 1 - density.cdf(f_value, df1, df2)

    bounds = _icc_two_way_random_absolute_average_confidence_interval(
        params.n_subjects,
        params.n_raters,
        params.alpha,
        params.ms_r,
        params.ms_e,
        params.ms_c,
    )

    return ICCResult(
        ICCModel.TWO_WAY,
        ICCDesc.AGREEMENT,
        ICCUnit.AVERAGE,
        ICCKind.TEST_RETEST,
        icc_value,
        bounds.l_bound,
        bounds.u_bound,
        p_value,
        params.n_subjects,
        params.n_raters,
    )


def _icc_two_way_mixed_consistency_average_confidence_interval(
    alpha: float, f_value: float, df1: int, df2: int
) -> _ICCBounds:
    f_l = f_value / density.ppf(1 - alpha, df1, df2)
    f_u = f_value * density.ppf(1 - alpha, df2, df1)
    l_bound = 1 - 1 / f_l
    u_bound = 1 - 1 / f_u
    return _ICCBounds(l_bound, u_bound)


def _icc_two_way_mixed_consistency(
    icc_unit: ICCUnit, icc_value: float, params: _ICCParameters
) -> ICCResult:
    df1 = params.n_subjects - 1
    df2 = (params.n_subjects - 1) * (params.n_raters - 1)

    f_value = params.ms_r / params.ms_e
    p_value = 1 - density.cdf(f_value, df1, df2)

    bounds = _icc_single_confidence_interval(
        params.n_raters, params.alpha, f_value, df1, df2
    )

    return ICCResult(
        ICCModel.TWO_WAY,
        ICCDesc.CONSISTENCY,
        icc_unit,
        ICCKind.OTHER,
        icc_value,
        bounds.l_bound,
        bounds.u_bound,
        p_value,
        params.n_subjects,
        params.n_raters,
    )


def icc_two_way_mixed_consistency_average(
    ratings: pd.DataFrame, confidence_level: float = 0.95
) -> ICCResult:
    """
    Compute the ICC(3,k) score.

    ICC(3,k): Two-way mixed effects, consistency, average raters/measurements.

    Parameters
    ----------
    ratings
        Matrix with n subjects m raters, i.e. array-like, shape (n_subjects, n_raters)
    confidence_level
        Confidence level of the interval.

    Returns
    -------
    ICCResult
        ICC with its relative test information
    """
    params = _icc_parameters(ratings, confidence_level)
    icc_value = (params.ms_r - params.ms_e) / params.ms_r

    return _icc_two_way_mixed_consistency(ICCUnit.AVERAGE, icc_value, params)


def icc_test_retest(
    data: pd.DataFrame, study: ICCResultSetStudy = ICCResultSetStudy.STUDY_CONTROL
) -> ICCResult:
    """Compute the test-retest ICC of a data frame.

    Implemented by: Mind-the-Pineapple/ICC is licensed under the MIT License. See [5]_
    and [6]_.

    Parameters
    ----------
    data
        A N*M pandas DataFrame containing N subjects and M ratings
    study
        Status of the study

    Returns
    -------
    ICCResult
        The test retest ICC score containing the value with its definition and its 95%
        confidence interval

    References
    ----------
    .. [5] https://www.rdocumentation.org/packages/irr/versions/0.84.1
    .. [6] https://pypi.org/project/pyirr/
    """
    if study == ICCResultSetStudy.STUDY_CLINICAL:
        data = data.iloc[:, [0, -1]]

    result = icc_two_way_random_absolute_average(data)
    result.sample_size = data.shape[0]
    result.kind = ICCKind.TEST_RETEST

    return result


def icc_parallel_form(form1: pd.DataFrame, form2: pd.DataFrame) -> ICCResult:
    """Compute the icc score from parallel form features.

    This score allows the comparison of to features of same nature and definition but
    obtained in different condition (example : CPS mean RT on predefinedKey1 compared to
    predefinedKey2)

    Parameters
    ----------
    form1: pandas.DataFrame
        A data frame containing the M feature form 1 values for all users
    form2: pandas.DataFrame
        A data frame containing the M' feature form 2 values for all users

    Returns
    -------
    ICCResult
        The parallel form ICC score containing the value with its definition and its 95%
        confidence interval
    """
    data = pd.concat([form1, form2], axis=1)
    data.dropna(axis=0, inplace=True)
    result = icc_two_way_random_absolute_average(data)
    result.kind = ICCKind.PARALLEL_FORM

    return result


def ensure_session_standards(
    data: pd.DataFrame, session_min: int = 8, null_ratio: float = 0.1
) -> pd.DataFrame:
    """Ensure sessions have sufficient support.

    This transformation ensures that sessions have sufficient support, i.e. a subject is
    only considered if it has contributed more than ``session_min`` sessions. A session
    is dropped if it has a higher null ratio than ``null_ratio``.

    Parameters
    ----------
    data
        A data frame with subjects as rows, sessions as columns, and cells containing
        the feature values.
    session_min
        The minimum number of required sessions for each subject to be considered in the
        analysis.
    null_ratio
        The ratio of null values across subjects for a particular session below which it
        is taken into account.

    Returns
    -------
    pandas.DataFrame
        The filtered ``data`` containing only subjects that have contributed at least
        ``session_min`` sessions, sessions that have a lower ratio of null values than
        ``null_ratio``, and subjects that have no null value for the latter sessions.
    """
    subject_mask = data.shape[1] - data.isnull().sum(axis=1) >= session_min
    data = data.loc[subject_mask, :]
    data = data.loc[:, data.isnull().mean() < null_ratio].dropna()

    return data


def icc_test_retest_session_safe(
    feature_collection: FeatureCollection,
    feature_id: str,
    study: ICCResultSetStudy = ICCResultSetStudy.STUDY_CONTROL,
    session_min: int = 8,
    null_ratio: float = 0.1,
) -> ICCResult:
    """Compute the ICC test retest score for one feature.

    Parameters
    ----------
    feature_collection
        A collection of features.
    feature_id
        The feature id to be used for the computation of the ICC scores
    study
        Type of the study used to determine the :math:`p_0^{icc}`. `0` is used for
        control studies and `0.6` for clinical ones.
    session_min
        See :func:`ensure_session_standards`.
    null_ratio
        See :func:`ensure_session_standards`.

    Returns
    -------
    ICCResult
        The ICC score results for the provided feature.

    """
    p0_icc = 0.0 if study == ICCResultSetStudy.STUDY_CONTROL else 0.6

    data = feature_collection.get_feature_values_by_trials(feature_id=feature_id)
    data = ensure_session_standards(data, session_min, null_ratio)

    res = icc_test_retest(data, study)
    res.power = icc_power(res.value, p0_icc, data.shape[1], res.sample_size)

    return res


def icc_set_test_retest(
    feature_collection: FeatureCollection,
    study: ICCResultSetStudy = ICCResultSetStudy.STUDY_CONTROL,
    session_min: int = 8,
    null_ratio: float = 0.1,
    errors: Literal["raise", "ignore"] = "raise",
) -> ICCResultSet:
    """Compute the ICC test retest score for all features.

    It takes into consideration the study type, which could be either "control" or
    "clinical". It also ensures sessions have sufficient support. See
    :func:`icc_test_retest_session_safe` for details.

    Parameters
    ----------
    feature_collection
        A collection of features
    study
        Type of the study used to determine the :math:`p_0^{icc}`. `0` is used for
        control studies and `0.6` for clinical ones.
    session_min
        See :func:`ensure_session_standards`.
    null_ratio
        See :func:`ensure_session_standards`.
    errors
        How to handle errors occurring during the computation of ICC scores.
        - If 'raise', then errors will be risen.
        - If 'ignore', then the feature will be skipped.

    Returns
    -------
    ICCResultSet
        A set of feature ICC

    Raises
    ------
    ValueError
        If ``errors`` is set to 'raise', :class:`ValueError` will be risen again if it
        occurred in :func:`icc_test_retest_session_safe`.
    """
    p0_icc = 0.0 if study == ICCResultSetStudy.STUDY_CONTROL else 0.6
    features_icc = ICCResultSet(study, p0_icc)

    for feature_id in feature_collection.feature_ids:
        try:
            features_icc.iccs[feature_id] = icc_test_retest_session_safe(
                feature_collection, feature_id, study, session_min, null_ratio
            )
        except ValueError:
            if errors == "raise":
                raise

    return features_icc
