"""A module containing functionality to process cps reaction time."""
from collections import defaultdict
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from dispel.data.flags import Flag, WrappedResult
from dispel.data.levels import Level
from dispel.data.values import AbbreviatedValue as AV
from dispel.data.values import ValueDefinition
from dispel.processing.extract import DEFAULT_AGGREGATIONS
from dispel.providers.generic.tasks.cps.modalities import (
    _DIGIT_RANGE,
    NBACKS,
    CPSLevel,
    ThirdsModality,
    ThirdsPairModality,
)
from dispel.stats.core import iqr, percentile_05, percentile_95, variation

TASK_NAME = AV("Cognitive Processing Speed test", "CPS")

EXPECTED_DURATION_S2D = 90
"""The expected duration of the symbol-to-digit part."""

EXPECTED_DURATION_D2D = 20
"""The expected duration of the digit-to-digit part."""

MINIMAL_N_DATA_POINTS = 3
r"""The # of data points for a feature below which we create flag."""

CPS_BASIC_AGGREGATION: List[Tuple[str, str]] = [
    *DEFAULT_AGGREGATIONS,
    ("q95", "95th percentile"),
    ("q05", "5th percentile"),
    ("iqr", "iqr"),
]
r"""The basic aggregations that are used in the test."""

CPS_EXTENDED_AGGREGATION: List[Tuple[str, str]] = [
    *CPS_BASIC_AGGREGATION,
    ("skew", "skewness"),
    ("kurtosis", "kurtosis"),
]
r"""The extended aggregations that are used in the test."""

CPS_SYMBOL_SPECIFIC_AGGREGATION: List[Tuple[str, str]] = DEFAULT_AGGREGATIONS
r"""Symbol specific aggregation."""

CPS_AGGREGATION_LIST: List[str] = [agg[0] for agg in CPS_EXTENDED_AGGREGATION]
r"""The CPS aggregation in a single list format."""

AV_REACTION_TIME = AV("reaction time", "rt")
r"""Abbreviation for the reaction time."""

CPS_FLAG_NEDP = Flag(
    "cps-behavioral-deviation-nedp",
    reason="Not enough data points used to compute the feature",
)


STD_KEY_RANDOM_RT_MEAN = "cps-std_rand_keyr-rt-mean"
STD_KEY_FIXED1_RT_MEAN = "cps-std_rand_key1-rt-mean"
STD_KEY_FIXED2_RT_MEAN = "cps-std_rand_key2-rt-mean"
DTD_RT_MEAN = "cps-dtd_rand-rt-mean"

EXTRA_MODALITY_LIST = ["unique", "pair", "l5", "f5", "5lvs5f", "third3third1"]
r"""The modality to aggregate when merging key1 and key2."""

LEVEL_DURATION_DEF = ValueDefinition("levelDuration", "expected level duration", "s")
r"""Expected level duration context value."""


def reaction_time(data: pd.DataFrame) -> pd.Series:
    """
    Compute the reaction time.

    Parameters
    ----------
    data
        The ``userInput`` data frame from the CPS task.

    Returns
    -------
    pandas.Series
        A pandas Series containing the computed reaction time based on the
        difference between display time (``tsDisplay``) and response time
        (``tsAnswer``).

    """
    delta = data["tsAnswer"] - data["tsDisplay"]
    return delta.dt.total_seconds().rename("reactionTime")


def agg_reaction_time(
    data: pd.DataFrame,
    agg: Union[str, AV],
    key: Optional[Union[List[Any], AV]],
    lower: Optional[int] = 0,
    upper: Optional[int] = None,
) -> WrappedResult[float]:
    """Aggregate ``reactionTime`` returned by `correct_data_selection`.

    Parameters
    ----------
    data
        A pandas.DataFrame obtained from a reading raw data set
        ``keys-analysis``.
    agg
        reaction time Aggregation
    key
        key selection
    lower
        The lower index to select of the data frame.
    upper
        The upper index to select of the data frame.

    Returns
    -------
    WrappedResult
        The aggregate reaction time wrapped into a `WrappedResult` class
        that contains flags generated during the computation.

    """
    agg = agg.abbr if isinstance(agg, AV) else agg
    key = [key] if key and not isinstance(key, list) else key
    data = (
        data[data["expected"].isin([k.value for k in key])]  # type: ignore
        if key
        else data
    )

    corr_data = correct_data_selection(data, lower, upper)

    if agg == "cv":
        result = variation(corr_data["reactionTime"])
    elif agg == "q05":
        result = percentile_05(corr_data["reactionTime"])
    elif agg == "q95":
        result = percentile_95(corr_data["reactionTime"])
    elif agg == "iqr":
        result = iqr(corr_data["reactionTime"])
    else:
        result = corr_data["reactionTime"].agg(agg)

    if len(corr_data) < MINIMAL_N_DATA_POINTS:
        wrapped_result: WrappedResult[float] = WrappedResult(result)
        wrapped_result.add_flag(CPS_FLAG_NEDP)
        return wrapped_result

    return WrappedResult(result)


def transform_user_input(data: pd.DataFrame, level: Level) -> pd.DataFrame:
    """
    Create a uniform data frame from user responses to perform analyses.

    Parameters
    ----------
    data
        A pandas data frame obtained from a reading raw data set ``userInput``
    level
        The level to be processed

    Raises
    ------
    ValueError
        Make sure length between input data and transformed dataset are
        consistent

    Returns
    -------
    pandas.DataFrame
        The proper pandas data frame containing ``expect``, ``actual``
        ,``reactionTime`` and ``tsAnswer`` pandas.Series to perform the digits
        or symbols analyses.
    """
    col_suffix = "Symbol" if level.id == CPSLevel.SYMBOL_TO_DIGIT else "Value"
    new_data = data.copy()
    new_data.sort_values(by=["tsAnswer"], inplace=True)

    if level.id == CPSLevel.SYMBOL_TO_DIGIT:
        exp = f"displayed{col_suffix}"
        act = f"user{col_suffix}"
        new_data[exp] = new_data[exp].str.extract(r"(\d+)").astype("int16")
        new_data[act] = new_data[act].str.extract(r"(\d+)").astype("int16")

    expected = new_data[f"displayed{col_suffix}"].rename("expected")
    actual = new_data[f"user{col_suffix}"].rename("actual")
    mismatch = expected != actual
    len_transformed = len(new_data["tsAnswer"])
    len_original = len(mismatch)
    if len_transformed != len_original:
        raise ValueError(
            f"Inconsistent length between input data and "
            f"transformed dataset : {len_transformed} vs "
            f"{len_original}"
        )
    return pd.concat(
        [
            expected,
            actual,
            mismatch.rename("mismatching"),
            reaction_time(new_data),
            new_data["tsAnswer"],
        ],
        axis=1,
    )


def correct_data_selection(
    data: pd.DataFrame, lower: Optional[int] = 0, upper: Optional[int] = None
) -> pd.DataFrame:
    """Select correct responses between two indexes of data.

    Parameters
    ----------
    data
        A pandas.DataFrame obtained from a reading raw data set
        ``keys-analysis``.
    lower
        The lower index to select of the data frame.
    upper
        The upper index to select of the data frame.

    Returns
    -------
    pandas.DataFrame
        A pandas data frame containing only correct responses.
    """
    if isinstance(upper, pd.Timestamp) or isinstance(lower, pd.Timestamp):
        sub_data = get_subset_from_ts(data, lower, upper)
    else:
        sub_data = data.iloc[lower:upper]
    corr_data = sub_data.loc[sub_data.expected == sub_data.actual]
    return corr_data


def compute_confusion_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the confusion matrix for each symbols/digits.

    Parameters
    ----------
    data
        A pandas data frame coming from
        :class:`dispel.providers.generic.tasks.cps.steps.TransformKeysAnalysisData`.

    Returns
    -------
    pandas.DataFrame
        The confusion matrix for the current level.
    """
    matrix = pd.DataFrame(np.nan, index=_DIGIT_RANGE, columns=_DIGIT_RANGE)
    conf = data.groupby(["actual", "expected"]).size().unstack()
    return matrix.combine_first(conf).fillna(0)


def compute_confusion_error_rate(
    data: pd.DataFrame, val1: int, val2: int
) -> np.float64:
    """Compute the confusion rate.

    The confusion rate is computed with respect to the two symbols or digits
    from the confusion matrix.

    Parameters
    ----------
    data
        A pandas data frame extracted by
        :class:`dispel.providers.generic.tasks.cps.steps.TransformKeysAnalysisData`.
    val1
        The first symbol/digit to compare.
    val2
        The second symbol/digit to compare.

    Returns
    -------
    numpy.float64
        Confusion error rate between the first and the second symbol/digit.
    """
    err_series1 = data[val1].sum() - data[val1][val1]
    err_series2 = data[val2].sum() - data[val2][val2]
    err_conf_series1 = data[val1][val2]
    err_conf_series2 = data[val2][val1]

    conf_series1 = err_conf_series1 / err_series1 if err_series1 != 0 else 0
    conf_series2 = err_conf_series2 / err_series2 if err_series2 != 0 else 0
    return np.float64((conf_series1 + conf_series2) / 2)


def compute_streak(frame: pd.DataFrame) -> Tuple[np.int64, np.int64]:
    """
    Compute the longest streak of incorrect and correct responses.

    Parameters
    ----------
    frame
        A pandas.DataFrame obtained from a reading raw data set ``userInput``.

    Returns
    -------
    Tuple[numpy.int64, numpy.int64]
        The longest streak of correct responses for a given level. And the
        longest streak of incorrect responses for a given level.
    """
    data = frame["success"].to_frame()
    data["streak"] = (data["success"].diff() != 0).cumsum()
    count = data.groupby("streak").count()
    val = data.groupby("streak").mean()
    count.rename(columns={"success": "count"}, inplace=True)
    val.rename(columns={"success": "val"}, inplace=True)
    streak = pd.concat([count, val], axis=1)

    max_correct_streak = streak.loc[streak.val == 1, "count"].max()
    max_incorrect_streak = streak.loc[streak.val == 0, "count"].max()

    if pd.isna(max_correct_streak):
        max_correct_streak = 0
    if pd.isna(max_incorrect_streak):
        max_incorrect_streak = 0

    return max_correct_streak, max_incorrect_streak


def linear_regression(data: pd.Series) -> LinearRegression:
    """Compute a linear regression on a pandas.Series based on its index.

    Parameters
    ----------
    data
        The pandas Series on which we desire to compute a linear regression.

    Returns
    -------
    LinearRegression
        The model object resulting of the sklearn API.
    """
    x = data.index.values.reshape(-1, 1)
    return LinearRegression().fit(x, data)


def compute_response_time_linear_regression(
    data: pd.DataFrame, to_drop: int
) -> Tuple[float, float]:
    """Compute a linear regression and extract slope coefficient and r2 score.

    The linear regression is made on the ``reactionTime`` pandas.Series.

    Parameters
    ----------
    data
        The ``keys-analysis`` raw data frame.
    to_drop
        The number of responses to drop at the beginning of the test.

    Returns
    -------
    Tuple[float, float]
        The slope coefficient and the r2 score of the model.
    """
    corr_data = correct_data_selection(data, 0, len(data))
    try:
        response_time = corr_data["reactionTime"].shift(-1 * to_drop).dropna()
        model = linear_regression(response_time)
        x = response_time.index.values.reshape(-1, 1)
        pred = pd.Series(model.predict(x), index=response_time.index)
    except (KeyError, ValueError):
        return np.nan, np.nan
    return model.coef_.item(), r2_score(response_time, pred)


def study2and3back(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract 1Back, 2Back and 3Back reaction time for correct responses only.

    Parameters
    ----------
    data
        A pandas data frame obtained from
        :class:`dispel.providers.generic.tasks.cps.steps.TransformKeysAnalysisData`.

    Returns
    -------
    pandas.DataFrame
        a pandas data frame containing 1,2 and 3 back and current reaction time
        when each 1,2 or 3 back is displayed for a given level.
    """
    corr_data = data.loc[data.expected == data.actual]
    corr_data.reset_index(drop=True, inplace=True)
    # init list
    feature_dict = defaultdict(list)

    for index, item in enumerate(corr_data["expected"]):
        # enumerate through the different lags
        for lag in NBACKS:
            if item == corr_data["expected"].shift(lag)[index]:
                feature_dict[f"back{lag}"].append(
                    corr_data["reactionTime"].shift(lag)[index]
                )
                feature_dict[f"current{lag}"].append(corr_data["reactionTime"][index])

    # add rtBack features
    series_list = [
        pd.Series(feature_dict[f"back{it}"], name=f"rtBack{it}", dtype="float64")
        for it in NBACKS
    ]

    # add rtCurrent features
    series_list += [
        pd.Series(feature_dict[f"current{it}"], name=f"rtCurrent{it}", dtype="float64")
        for it in NBACKS
    ]
    return pd.concat(series_list, axis=1)


def get_subset_from_ts(
    data: pd.DataFrame, lower: pd.Timestamp, upper: pd.Timestamp
) -> pd.DataFrame:
    """
    Select a subset of keys-analysis dataset based on timestamps.

    Parameters
    ----------
    data
        Input keys-analysis dataframe
    lower
        Lower bound
    upper
        Upper bound

    Returns
    -------
    pd.DataFrame
        The filtered version of the input
    """
    return data.loc[(data["tsAnswer"] >= lower) & (data["tsAnswer"] <= upper)]


def get_third_data(
    data: pd.DataFrame, subset: ThirdsModality, level: Level
) -> pd.DataFrame:
    """
    Get the data for a particular third.

    Parameters
    ----------
    data
        The input key analysis dataset
    subset
        The third modality
    level
        The current level to get the duration

    Returns
    -------
    pd.DataFrame
        The filtered version of the input
    """
    duration = level.context.get("levelDuration").value
    lower, upper = subset.get_lower_upper(data, duration)
    return get_subset_from_ts(data, lower, upper)


def compute_correct_third_from_paired(
    data: pd.DataFrame, subset: ThirdsPairModality, level: Level, is_left: bool
) -> int:
    """Compute the number of correct responses for a specific third."""
    duration = level.context.get("levelDuration").value
    if is_left:
        low, up = subset.left.get_lower_upper(data, duration)  # type: ignore
    elif not is_left:
        low, up = subset.right.get_lower_upper(data, duration)  # type: ignore
    else:
        ValueError(f"is_left should be boolean but is {type(is_left)}")

    filtered_data_right = get_subset_from_ts(data, low, up)

    return (~filtered_data_right["mismatching"]).sum()


@staticmethod  # type: ignore
def _compute_substitution_time(values: List[float]) -> Union[None, float]:
    """Compute the substitution time.

    The substitution time is defined as the difference between the
    symbol to digit reaction time (the time required to associate a symbol
    with a number) and the digit to digit reaction time (the time required to
    associate a number with a number).

    Parameters
    ----------
    values
        A list of expected size 2 containing in first position, the symbol to
        digit reaction time and in second position the digit to digit reaction
        time.

    Returns
    -------
    float
        The substitution time.
    """
    # Works for random keys and fixed keys.
    if len(values) < 2:
        return None
    # Difference std_rt - dtd_rt
    return values[0] - values[1]
