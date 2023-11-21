"""Grip Force test related functionality.

This module contains functionality to extract measures for the *Grip Force*
test (GRIP).
"""
import functools
from collections import namedtuple
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from dispel.data.levels import Context, Level
from dispel.data.measures import MeasureValueDefinitionPrototype
from dispel.data.raw import DEFAULT_COLUMNS, USER_ACC_MAP, RawDataValueDefinition
from dispel.data.validators import GREATER_THAN_ZERO, RangeValidator
from dispel.data.values import AbbreviatedValue as AV
from dispel.data.values import AVEnum
from dispel.processing import ProcessingStep
from dispel.processing.extract import (
    DEFAULT_AGGREGATIONS,
    AggregateRawDataSetColumn,
    ExtractMultipleStep,
    ExtractStep,
)
from dispel.processing.level import LevelFilter, LevelIdFilter, ProcessingStepGroup
from dispel.processing.modalities import HandModality, SensorModality
from dispel.processing.transform import ConcatenateLevels, TransformStep
from dispel.providers.ads.data import ADSReading
from dispel.providers.generic.sensor import (
    FREQ_100HZ,
    RenameColumns,
    Resample,
    SetTimestampIndex,
)
from dispel.providers.generic.tremor import TremorMeasures
from dispel.providers.registry import process_factory
from dispel.stats.core import variation
from dispel.utils import to_camel_case

TASK_NAME = AV("Grip Force test", "GRIP")

TEST_DURATION_VALIDATOR = RangeValidator(lower_bound=0, upper_bound=48)

CategoryToDiff = namedtuple("CategoryToDiff", ["category_1", "category_2"])
r"""Tuple-like object with the two pressure categories used to compute mean
applied force difference."""

# Plateau detection algorithm constants
N_SEQUENCE = 8
SMALL_GAP_SIZE = 15
MOVING_AVG_SIZE = 10
LOOK_BEHIND = 100
LOOK_AHEAD = 40


class PlateauModality(AVEnum):
    """Enumerated constant representing the plateau modality.

    This modality defines the subpart of the plateau that is kept
    for measure computation.
    """

    def __init__(
        self,
        name: str,
        abbr: str,
        start_offset: Optional[int] = None,
        end_offset: Optional[int] = None,
    ):
        self.start_offset = pd.Timedelta(start_offset, unit="s")
        self.end_offset = pd.Timedelta(end_offset, unit="s")
        super().__init__(name, abbr)

    MIDDLE_FOUR = ("four seconds in the middle", "s4m", 1, 1)
    MIDDLE_TWO = ("two seconds in the middle", "s2m", 2, 2)
    LAST_FIVE = ("last five seconds", "s5l", 1, 0)


class TargetPressureModality(AVEnum):
    """Enumerated constant representing the pressure modalities."""

    LOW = ("low target pressure", "low")
    MEDIUM = ("medium target pressure", "medium")
    HIGH = ("high target pressure", "high")
    VERY_HIGH = ("very high target pressure", "very_high")


class PositionModality(AVEnum):
    """Screen thumb position for grip force task."""

    @property
    def column(self) -> str:
        """Create a camel case string of position modalities."""
        return to_camel_case(str(self))

    X = ("x thumb position", "xpos")
    Y = ("y thumb position", "ypos")


def smooth_discrete_pdf(data: pd.Series) -> pd.Series:
    """Compute the discrete probability density function (PDF).

    Then smooth the values by filtering out small spikes.

    Parameters
    ----------
    data
        A series of a signal. e.g. 'pressure'

    Returns
    -------
    pandas.Series
        A series of the discrete derivative of the signal where small
        spikes have been smoothed.

    Raises
    ------
    ValueError
        Raises a value error if the pandas.Series is too short.
    """
    if len(data) < 2:
        raise ValueError("The pd.series is too short to be smoothed." f"data={data}")
    res = data.diff().abs()
    res.iloc[0] = 0

    mask = (res > 0) & (res.shift(-2) == 0)
    res[mask] = 0

    return res


def avg_diff_discrete_pdf(data: pd.DataFrame, context: Context) -> pd.DataFrame:
    """Compute the average of the differentiated discrete PDF.

    This corresponds to the average of the second discrete
    derivative of the pressure signal. In the process of computing the
    metric, we add several columns, namely 'discrete-pdf', 'diff-discrete-pdf'
    and `level-sequence` where we set the contextual level information
    under a series format with integers going from 0 to 7 indicating
    which level we are in.

    Parameters
    ----------
    data
        The two columns 'discrete-pdf' and 'pressure' of the data frame
        obtained by concatenating the levels.
    context
        The merged context obtained from the concatenation of the
        levels.

    Returns
    -------
    pandas.DataFrame
        A new data frame with `discrete-pdf`, `diff-discrete-pdf`,
        `avg-diff-discrete-pdf` and `level-sequence` columns
    """
    # smoothing the discrete pdf
    output = pd.DataFrame()
    output["discrete-pdf"] = smooth_discrete_pdf(data["pressure"])

    # differentiate the discrete_pdf
    diff_discrete_pdf = output["discrete-pdf"].diff().abs()
    diff_discrete_pdf.iloc[0] = 0
    output["diff-discrete-pdf"] = diff_discrete_pdf

    # init avg_diff_discrete_pdf
    output["avg-diff-discrete-pdf"] = 0
    output["level-sequence"] = -1

    # goes through the eighth different levels
    for sequence in range(N_SEQUENCE):
        level = context.get_raw_value(f"level_{sequence}")
        mask = (diff_discrete_pdf.index >= level.start) & (
            diff_discrete_pdf.index <= level.end
        )
        # copy the label
        output.loc[mask, "level-sequence"] = sequence
        # compute the mean for each level
        pressure_level = output.loc[mask, "diff-discrete-pdf"]
        output.loc[mask, "avg-diff-discrete-pdf"] = np.float64(np.mean(pressure_level))

    output["level-sequence"] = output["level-sequence"].astype(int)
    return output


def get_target_pressure(context: Context, seq: int) -> float:
    """Get target pressure from context and level sequence.

    Parameters
    ----------
    context
        The merged context obtained from the concatenation of the
        levels.
    seq
        The sequence of interest: e.g. a value in {0,1,..,7}

    Returns
    -------
    float
        Target pressure corresponding to the sequence seq.
    """
    return context.get_raw_value(f"targetPressure_{seq}")


def compute_second_derivative_spikes(data: pd.DataFrame) -> pd.Series:
    """Find spikes in diff discrete pdf.

    The method is simple, we compare the moving average of the second
    derivative of the pressure signal (diff discrete pdf) to the
    average of the latter on the entire level-sequence. A spike or a
    plateau is detected if the moving average is higher than the
    level mean.

    Parameters
    ----------
    data
        A data frame with the two columns 'diff-discrete-pdf' and
        'avg-diff-discrete-pdf' (e.g.: obtained with avg_diff_discrete_pdf).

    Returns
    -------
    pandas.Series
        A series with values in {0, 1} indicating if a plateau has been
        detected.

    Raises
    ------
    ValueError
        A value error is raised if the length of the data is too short.
    """
    if len(data) < 20:
        raise ValueError(
            f"Length of the data is too short. len(data) = {len(data)} "
            "is less than 20"
        )
    # initialize the detected-plateau
    detected_plateau = data["avg-diff-discrete-pdf"].copy() * 0
    for i in range(len(data) - 20):
        if (
            np.mean(data["diff-discrete-pdf"].iloc[i : i + MOVING_AVG_SIZE])
            < data["avg-diff-discrete-pdf"].iloc[i]
        ):
            detected_plateau.iloc[i] = 1
    return detected_plateau


def fill_plateau(data: pd.Series) -> pd.Series:
    """Fill small gaps in the plateau.

    Parameters
    ----------
    data
        A series of values in {0, 1} typically the one obtained after applying
        the `compute_second_derivative_spikes` function.

    Returns
    -------
    pandas.Series
        A modified version of the series where small gaps in the plateau have
        been filled.

    Raises
    ------
    ValueError
        A value error is raised if the length of the data is too short.
    """
    if len(data) < 20:
        raise ValueError(
            "Length of the data is too short."
            f"len(data) = {len(data)} is less than 20"
        )
    data = data.copy()
    for i in range(len(data) - 20):
        if (data.iloc[i] == 1) and (data.iloc[i + SMALL_GAP_SIZE] == 1):
            data.iloc[i : i + SMALL_GAP_SIZE] = 1
    return data


def remove_short_plateau(data: pd.Series, plateau_size: int = 30) -> pd.Series:
    """Remove plateau shorter than plateau_size.

    Parameters
    ----------
    data
        A series of values in {0, 1}, typically the one obtained after applying
        the `fill_plateau` function.
    plateau_size
        The minimum size of a plateau, by default 30

    Returns
    -------
    pandas.Series
        A modified version of the series where plateau shorter than
        the threshold plateau_size have been removed.
    """
    i = 0
    data = data.copy()
    while i < len(data):
        if data.iloc[i] == 1:
            count = 0
            while i < len(data) and data.iloc[i] == 1:
                count += 1
                i += 1
            if count < plateau_size:
                data.iloc[i - count : i] = 0
        i += 1

    return data


def extend_and_convert_plateau(
    pressure: pd.Series, detected_plateau: pd.Series
) -> pd.Series:
    """Extend the plateau under certain conditions.

    The main idea is to fill large gaps between a current index and
    a lookahead index where a plateau is detected. Gaps are filled whenever the
    lookahead mean (defined as the mean between the current index and the
    lookahead) does not differ too much (0.5 pressure unit)
    from a `lookbehind` mean (defined as the mean from a lookbehind index until
    the current index
    ).

    Parameters
    ----------
    pressure
        A Series of 'pressure'
    detected_plateau
        A series of detected plateau

    Returns
    -------
    pandas.Series
        The updated series 'detected-plateau' where large gaps have been filled
        and then where '1' have been replaced with values from the 'pressure'
        column.

    Raises
    ------
    ValueError
        A value error is raised if the length of the data is too short.
    """
    if len(pressure) < LOOK_AHEAD:
        raise ValueError(
            "Length of the data is too short."
            f"len(pressure) = {len(pressure)} is less than 20"
        )

    detected_plateau = detected_plateau.copy()
    for i in range(LOOK_AHEAD, len(pressure) - LOOK_AHEAD):
        # compute the look behind mean
        rolling_mean = np.mean(pressure.iloc[np.max((0, i - LOOK_BEHIND)) : i])
        # compute the look ahead mean
        mean_forward = np.mean(pressure.iloc[i : i + LOOK_AHEAD])
        # check the three conditions
        if (
            detected_plateau.iloc[i] == 1 and detected_plateau.iloc[i + LOOK_AHEAD] == 1
        ) and np.abs(mean_forward - rolling_mean) < 0.3:
            detected_plateau.iloc[i : i + LOOK_AHEAD] = pressure.iloc[
                i : i + LOOK_AHEAD
            ]

        # finish setting the pressure value
        if detected_plateau.iloc[i] == 1:
            detected_plateau.iloc[i] = pressure.iloc[i]

    for i in range(LOOK_AHEAD):
        # finish setting the pressure value
        if detected_plateau.iloc[i] == 1:
            detected_plateau.iloc[i] = pressure.iloc[i]

    return detected_plateau


def refined_target(data: pd.DataFrame, contexts: Context) -> pd.Series:
    """Refine the target pressure for the detected plateau.

    The method is simple, on each detected plateau, we take
    the maximum occurring target sequence as `the label` and add
    the refined target as the value of the target pressure for this
    label (or level sequence).

    Parameters
    ----------
    data
        A dataframe with the columns {'detected-plateau', 'level-sequence'}
    contexts
        The merged context obtained from the concatenation of the
        levels.

    Returns
    -------
    pandas.Series
        The series 'refined-target' corresponding to the target pressure
        extended to the size of the 'detected-plateau'.

    """
    target = data["detected-plateau"] * 0
    index = 0
    while index < len(data):
        if data["detected-plateau"].iloc[index] != 0:
            offset_plateau = index
            while (
                offset_plateau < len(data)
                and data["detected-plateau"].iloc[offset_plateau] != 0
            ):
                offset_plateau = offset_plateau + 1

            maximum_occurring_sequence = np.bincount(
                data["level-sequence"].iloc[index:offset_plateau]
            ).argmax()

            target.iloc[index:offset_plateau] = get_target_pressure(
                contexts, maximum_occurring_sequence  # type: ignore
            )

            index = offset_plateau
        else:
            index += 1
    return target


class FilterIncompleteSequences(LevelFilter):
    """A filter to skip levels that do not contain eight sequences.

    Notes
    -----
    The filter expects concatenated levels from
    :class:`~dispel.processing.transform.ConcatenateLevels` and should contain the
    target pressure values in the context (i.e. ``targetPressure_{index}``.)
    """

    def filter(self, levels: Iterable[Level]) -> Set[Level]:
        """Check if there is contextual info for the eight sequences."""
        out = set()
        for level in levels:
            good = True
            for index in range(N_SEQUENCE):
                if f"targetPressure_{index}" not in level.context:
                    good = False
                    break
            if good:
                out.add(level)
        return out

    def repr(self) -> str:
        """Get representation of the filter."""
        return "filter incomplete levels"


FILTER_INCOMPLETE_SEQUENCES = FilterIncompleteSequences()


def compute_plateau(data: pd.DataFrame, level: Level) -> pd.DataFrame:
    """Compute the plateau.

    To compute the plateau we follow several steps:
    The first step is to compute the first and second discrete
    derivative of the resampled and interpolated pressure signal.
    (see :func:`avg_diff_discrete_pdf`).

    Then we initialize our plateau detection algorithm by finding the
    spikes in the second discrete derivative of the signal (see
    :func:`compute_second_derivative_spikes`).

    A third step consists in filling all small gaps in the detected
    plateau. (see :func:`fill_plateau`).

    Once plateau are filled we filter the result by removing the very
    short detected plateau (see :func:`remove_short_plateau`).

    The next step consists in extending the plateau filling large
    gap under certain conditions. (see
    :func:`extend_and_convert_plateau`).

    Finally we create a target pressure signal based on the identified
    plateau. The idea is that a plateau often goes beyond the level
    border and hence should not be compared to the target pressure
    of the next level but on the main one on which the plateau resides.
    (see :func:`refined_target`).

    Parameters
    ----------
    data
        The data on which the plateau computation is to be performed.
    level
        The level corresponding to the given data.

    Returns
    -------
    pandas.DataFrame
        The pandas data frame containing the different plateaus.
    """
    # compute average diff discrete pdf
    plateau_df = avg_diff_discrete_pdf(data, level.context)

    # first step of plateau detection
    plateau_df["detected-plateau"] = compute_second_derivative_spikes(
        plateau_df[["diff-discrete-pdf", "avg-diff-discrete-pdf"]]
    )

    # Fill small gaps in the plateau
    plateau_df["detected-plateau"] = fill_plateau(plateau_df["detected-plateau"])

    # remove all the short plateau
    plateau_df["detected-plateau"] = remove_short_plateau(
        plateau_df["detected-plateau"]
    )

    # extend and convert the plateau to the pressure value
    plateau_df["detected-plateau"] = extend_and_convert_plateau(
        data["pressure"], plateau_df["detected-plateau"]
    )

    # refine the plateau and define the target accordingly
    plateau_df["refined-target"] = refined_target(
        plateau_df[["detected-plateau", "level-sequence"]], level.context
    )

    return plateau_df


class TransformPlateau(TransformStep):
    """An transform processing step for plateau detection."""

    def __init__(self, level_id):
        super().__init__(
            "screen_ts_resampled",
            transform_function=compute_plateau,
            new_data_set_id="plateau",
            definitions=[
                RawDataValueDefinition("discrete-pdf", "Discrete PDF"),
                RawDataValueDefinition(
                    "diff-discrete-pdf", "Differentiated discrete PDF"
                ),
                RawDataValueDefinition(
                    "avg-diff-discrete-pdf", "Average differentiated discrete PDF"
                ),
                RawDataValueDefinition("detected-plateau", "Detected plateau"),
                RawDataValueDefinition("level-sequence", "Level Sequence"),
                RawDataValueDefinition("refined-target", "Refined Target"),
            ],
            level_filter=LevelIdFilter(level_id) & FILTER_INCOMPLETE_SEQUENCES,
        )


def time_defined_plateau(
    data: pd.DataFrame, level: Level, plateau: PlateauModality
) -> pd.DataFrame:
    """Extract time-defined plateau.

    This function extracts the plateaus as defined in the plateau modality.
    More formally, given a level that is defined by the following time
    windows: [start: end]. The plateau offset refines this time windows
    only keeping the section
    ``[start + plateau.start_offset: end - plateau.end_offset]``.

    Parameters
    ----------
    data
        A data frame such as the one obtained after resampling screen.
        e.g.: `screen_ts_resampled`
    level
        Any Level
    plateau
        A plateau modality defining the time windows on which measures
        will be computed.

    Returns
    -------
    pandas.DataFrame
        A data frame with three columns: ``detected-plateau`` is equals to
        the pressure on the plateau zero otherwise.
        ``level-sequence`` are integers indicating the level.
        ``refined-target`` is the target value.
    """
    contexts = level.context

    res = data.copy()
    res["detected-plateau"] = 0
    res["level-sequence"] = -1

    # goes through the eighth different levels
    for sequence in range(N_SEQUENCE):
        level = contexts.get_raw_value(f"level_{sequence}")
        mask = (res.index >= level.start + plateau.start_offset) & (
            res.index <= level.end - plateau.end_offset
        )

        # copy the label
        res.loc[mask, "level-sequence"] = sequence
        res.loc[mask, "detected-plateau"] = res.loc[mask, "pressure"]

    res["level-sequence"] = res["level-sequence"].astype(int)

    # define the target
    res["refined-target"] = res["level-sequence"].apply(
        lambda x: get_target_pressure(contexts, x) if x >= 0 else 0
    )

    return res[["detected-plateau", "level-sequence", "refined-target"]]


class TransformTimeDefinedPlateau(TransformStep):
    """An transform processing step for time-defined plateau."""

    def __init__(self, level_id, plateau: PlateauModality):
        super().__init__(
            "screen_ts_resampled",
            transform_function=functools.partial(time_defined_plateau, plateau=plateau),
            new_data_set_id=f"plateau_{plateau.av}",
            definitions=[
                RawDataValueDefinition("detected-plateau", "detected-plateau"),
                RawDataValueDefinition("level-sequence", "Level Sequence"),
                RawDataValueDefinition("refined-target", "Refined Target"),
            ],
            level_filter=LevelIdFilter(level_id) & FILTER_INCOMPLETE_SEQUENCES,
        )


class TransformPressureError(TransformStep):
    """A TransformStep for pressure error related measures."""

    def __init__(self, level_id, plateau: Optional[PlateauModality] = None):
        def _pressure_error(data):
            mask = data["refined-target"] != 0
            return data.loc[mask, "detected-plateau"] - data.loc[mask, "refined-target"]

        data_set_id = "plateau"
        new_data_set_id = "pressure-error"

        if plateau:
            data_set_id = f"{data_set_id}_{plateau.av}"
            new_data_set_id = f"{new_data_set_id}_{plateau.av}"

        super().__init__(
            data_set_id,
            transform_function=_pressure_error,
            new_data_set_id=new_data_set_id,
            definitions=[RawDataValueDefinition("pressure-error", "Pressure Error")],
            level_filter=LevelIdFilter(level_id) & FILTER_INCOMPLETE_SEQUENCES,
        )


def target_pressure_to_category(
    contexts: Context,
) -> Tuple[Dict[Any, TargetPressureModality], List]:
    """Create a mapping between target pressure and their level.

    Parameters
    ----------
    contexts
        The merged context obtained from the concatenation of the
        levels.

    Returns
    -------
    Tuple[Dict[Any, str], List]
        The first element is a dictionary with as keys, the values of
        the four different target pressure and as values, the pressure category
        i.e values are in `{'low', 'medium', 'high', 'very_high'}`.
        The Second element is the sequence of pressure category e.g.:
        `['low', 'high', 'very_high', 'medium', 'high', 'medium',
        'very_high','low']`.
    """
    target_pressure = [get_target_pressure(contexts, seq) for seq in range(N_SEQUENCE)]
    four_target = sorted(list(dict.fromkeys(target_pressure)))
    target_to_cat = {
        four_target[0]: TargetPressureModality.LOW,
        four_target[1]: TargetPressureModality.MEDIUM,
        four_target[2]: TargetPressureModality.HIGH,
        four_target[3]: TargetPressureModality.VERY_HIGH,
    }
    return (target_to_cat, [target_to_cat[pressure] for pressure in target_pressure])


class TransformRefinedTargetCategory(TransformStep):
    """A TransformStep for to identify pressure categories."""

    def __init__(self, level_id, plateau: Optional[PlateauModality] = None):
        def target_to_category(data: pd.DataFrame, level: Level) -> pd.DataFrame:
            """Map refined-target column to a target pressure category."""
            mapping, _ = target_pressure_to_category(level.context)  # type: ignore
            mask = data["refined-target"] != 0
            return data["refined-target"][mask].replace(mapping)

        data_set_id = "plateau"
        new_data_set_id = "pressure-category"

        if plateau:
            data_set_id = f"{data_set_id}_{plateau.av}"
            new_data_set_id = f"{new_data_set_id}_{plateau.av}"

        super().__init__(
            data_set_id,
            transform_function=target_to_category,
            new_data_set_id=new_data_set_id,
            definitions=[
                RawDataValueDefinition("pressure-category", "Pressure category")
            ],
            level_filter=LevelIdFilter(level_id) & FILTER_INCOMPLETE_SEQUENCES,
        )


def filter_category(
    data: pd.DataFrame,
    category: pd.DataFrame,
    cat_str: Optional[TargetPressureModality] = None,
) -> pd.DataFrame:
    """Filter data based on category.

    The data frame data is filtered to only keep values for which the
    `'pressure-category'` is equal to the `cat_str`. The exception is
    when `cat_str == 'all'` where data is returned without filtering.

    Parameters
    ----------
    data
        A dataframe typically obtained after applying the TransformStep :
        :class:`TransformPlateau`.
    category
        A dataframe with column ``'pressure-category'`` typically obtained
        after applying the TransformStep :
        :class: `TransformRefinedTargetCategory.`
    cat_str
        The pressure category indicating on which
        category the applied force is selected.

    Returns
    -------
    pandas.DataFrame
        The filtered version of the data frame data.
    """
    if not cat_str:
        return data

    mask = category["pressure-category"] == cat_str
    return data[mask]


def rms_pressure_error(
    pressure_error: pd.DataFrame,
    category: pd.DataFrame,
    only_category: Optional[TargetPressureModality] = None,
) -> float:
    """Compute the Root Mean Square of `'pressure-error'` column.

    Parameters
    ----------
    pressure_error
        A dataframe with column `'pressure-error'` typically obtained after
        applying the TransformStep : :class:`TransformPressureError`.
    category
        A dataframe with column `'pressure-category'` typically obtained after
        applying the TransformStep : :class:`TransformRefinedTargetCategory`.
    only_category
        The pressure category indicating on which
        category the RMSE is computed.

    Returns
    -------
    float
        The Root Mean Square of the `'pressure-error'` column on the defined
        pressure category cat str
    """
    data = filter_category(pressure_error["pressure-error"], category, only_category)
    return (data**2).mean() ** 0.5


def _hand_to_level_id(hand: HandModality) -> str:
    return f"{hand.name.lower()}-all"


class ExtractRMSPressure(ExtractStep):
    """An extraction processing step for pressure accuracy measures.

    Parameters
    ----------
    hand
        The hand used to perform the test
    category
        The pressure category indicating on which category the RMSE is
        computed.
    plateau
        The plateau on which the extraction is to be performed
    """

    def __init__(
        self,
        hand: HandModality,
        category: Optional[TargetPressureModality] = None,
        plateau: Optional[PlateauModality] = None,
    ):
        modalities: List[AV] = [hand.av]
        description = (
            f"The measure of the distance between the ideal "
            f"target force and the actual force applied by the "
            f"subject for the {hand}"
        )
        data_set_ids = ["pressure-error", "pressure-category"]
        if category:
            modalities.append(category.av)
            description += f" and {category} target force category"
        if plateau:
            modalities.append(plateau.av)
            description += f" and the plateau modality {plateau}"
            data_set_ids = [id_ + f"_{plateau.av}" for id_ in data_set_ids]
        description += "."

        definition = MeasureValueDefinitionPrototype(
            measure_name=AV("pressure", "pressure"),
            data_type="float64",
            modalities=modalities,
            aggregation=AV("root mean square error", "RMSE"),
            description=description,
        )

        super().__init__(
            data_set_ids,
            transform_function=functools.partial(
                rms_pressure_error, only_category=category
            ),
            definition=definition,
            level_filter=LevelIdFilter(_hand_to_level_id(hand))
            & FILTER_INCOMPLETE_SEQUENCES,
        )


def applied_force(
    detected_plateau: pd.DataFrame,
    category: pd.DataFrame,
    cat_str: Optional[TargetPressureModality] = None,
) -> pd.Series:
    """Select the applied force or partial 'detected-plateau' column.

    Parameters
    ----------
    detected_plateau
        A dataframe with column 'detected-plateau' typically obtained after
        applying the TransformStep : :class:`TransformPlateau`.
    category
        A dataframe with column ``'pressure-category'`` typically obtained
        after applying the TransformStep :
        :class: `TransformRefinedTargetCategory.`
    cat_str
        The pressure category indicating on which
        category the applied force is selected.

    Returns
    -------
    pandas.Series
        The applied force or the 'detected-plateau' column on the defined
        pressure category cat_str.
    """
    non_null = detected_plateau["detected-plateau"] != 0
    data = filter_category(detected_plateau[non_null], category, cat_str)
    return data["detected-plateau"] / data["refined-target"]


class ExtractAppliedForceStats(ExtractMultipleStep):
    """An extraction processing step for applied force measures.

    Parameters
    ----------
    hand
        The hand used to perform the test.
    category
        The pressure category indicating on which category the applied force
        stats are computed.
    plateau
        The plateau on which the extraction is to be performed.
    percentile
        Compute the q-th percentile of the applied forces in the
        aggregation methods.
    """

    def __init__(
        self,
        hand: HandModality,
        category: Optional[TargetPressureModality] = None,
        plateau: Optional[PlateauModality] = None,
        percentile: int = 90,
    ):
        def _function_factory(agg, agg_label):
            return dict(
                func=lambda x, y: applied_force(x, y, category).agg(agg),
                aggregation=AV(agg_label, agg),
            )

        functions = [
            _function_factory(agg, agg_label) for agg, agg_label in DEFAULT_AGGREGATIONS
        ]
        functions += [
            dict(
                func=lambda x, y: variation(applied_force(x, y, category)),
                aggregation=AV("coefficient of variation", "CV"),
            )
        ]
        functions += [
            dict(
                func=lambda x, y: np.percentile(
                    applied_force(x, y, category), percentile
                ),
                aggregation=AV(f"{percentile}-th percentile", f"{percentile}th"),
            )
        ]

        modalities = [hand.av]
        description = (
            f"The applied force is a normalized version of the "
            f"pressure, measured on plateaus for the {hand}, by "
            f"the target pressure on the same plateau."
        )
        data_set_ids = ["plateau", "pressure-category"]
        if category:
            modalities.append(category.av)
            description += f" and {category} target pressure category."
        if plateau:
            modalities.append(plateau.av)
            description += f" the plateau modality {plateau}"
            data_set_ids = [
                "plateau" + f"_{plateau}",
                "pressure-category" + f"_{plateau}",
            ]
        description += "."

        definition = MeasureValueDefinitionPrototype(
            measure_name=AV("applied force", "applied_force"),
            data_type="float64",
            modalities=modalities,
            description=description,
        )
        super().__init__(
            data_set_ids,
            functions,
            definition=definition,
            level_filter=LevelIdFilter(_hand_to_level_id(hand))
            & FILTER_INCOMPLETE_SEQUENCES,
        )


class ExtractMeanAppliedForceDiff(ExtractMultipleStep):
    """An extraction processing step for mean applied force difference (MAF).

    Parameters
    ----------
    hand
        The hand used to perform the test.
    categories_to_diff
        The :class:`NamedTuple` CategoryToDiff containing the two pressure
        categories on which the MAF difference is compute.
    plateau
        The plateau on which the extraction is to be performed.
    """

    def __init__(
        self,
        hand: HandModality,
        categories_to_diff: CategoryToDiff,
        plateau: Optional[PlateauModality] = None,
    ):
        def _mean_maf_diff(
            detected_plateau: pd.DataFrame,
            category: pd.DataFrame,
            categories_to_diff: CategoryToDiff,
        ):
            mean_maf_1 = np.mean(
                applied_force(detected_plateau, category, categories_to_diff.category_1)
            )
            mean_maf_2 = np.mean(
                applied_force(detected_plateau, category, categories_to_diff.category_2)
            )
            return np.abs(mean_maf_1 - mean_maf_2)

        functions = [dict(func=lambda x, y: _mean_maf_diff(x, y, categories_to_diff))]

        modalities = [
            hand.av,
            categories_to_diff.category_1.av,
            categories_to_diff.category_2.av,
        ]
        description = (
            f"Absolute difference between the mean applied force "
            f"(pressure normalized by the Target Force) on the"
            f" {categories_to_diff.category_1} and the mean "
            f"applied force on the {categories_to_diff.category_2}"
            f" for the {hand}"
        )
        data_set_ids = ["plateau", "pressure-category"]

        if plateau:
            modalities.append(plateau.av)
            description += f" the plateau modality {plateau}"
            data_set_ids = [
                "plateau" + f"_{plateau}",
                "pressure-category" + f"_{plateau}",
            ]
        description += "."

        definition = MeasureValueDefinitionPrototype(
            measure_name=AV("MAF difference", "maf_diff"),
            data_type="float64",
            modalities=modalities,
            description=description,
        )
        super().__init__(
            data_set_ids,
            functions,
            definition=definition,
            level_filter=LevelIdFilter(_hand_to_level_id(hand))
            & FILTER_INCOMPLETE_SEQUENCES,
        )


class ExtractThumbPositionVariation(ExtractMultipleStep):
    """An extraction processing step for thumb position on screen.

    Parameters
    ----------
    hand
        The hand used to perform the test
    """

    def __init__(self, hand: HandModality):
        def _thumb_variation(position: PositionModality):
            return dict(
                func=lambda data: variation(data[position.column]),
                modalities=[hand.av, position.av],
                position=position,
            )

        definition = MeasureValueDefinitionPrototype(
            measure_name=AV("coefficient of variation", "cv"),
            data_type="float64",
            description="The coefficient of variation of the {position} for "
            f"the {hand}.",
            validator=GREATER_THAN_ZERO,
        )
        super().__init__(
            "screen",
            transform_functions=list(map(_thumb_variation, PositionModality)),
            definition=definition,
            level_filter=LevelIdFilter(_hand_to_level_id(hand))
            & FILTER_INCOMPLETE_SEQUENCES,
        )


def reaction_time_between_level(plateau: pd.DataFrame, level: Level) -> pd.DataFrame:
    """Compute the reaction time between the 8 different levels."""
    contexts = level.context
    reaction_time = [None] * (N_SEQUENCE - 1)
    transition_time = [None] * (N_SEQUENCE - 1)
    transition_start = [None] * (N_SEQUENCE - 1)
    transition_end = [None] * (N_SEQUENCE - 1)
    _, pressure_category = target_pressure_to_category(contexts)  # type: ignore
    transition_cat = [
        tuple(pressure_category[it : it + 2]) for it in range(N_SEQUENCE - 1)
    ]

    for seq in range(N_SEQUENCE - 1):
        # find the current level
        mask = plateau["level-sequence"] == seq

        # find the next level
        mask_next = plateau["level-sequence"] == seq + 1

        # identify the end of the level
        end_level = plateau[mask].index[-1]

        # identify the current level target
        curr_level = plateau.loc[mask, "refined-target"][-1]

        # identify the end of the next level
        end_next_level = plateau[mask_next].index[-1]

        # search for a change in refined-target
        mask_plateau = (
            plateau.loc[end_level:end_next_level, "refined-target"] != curr_level
        )

        # find the end of the plateau
        in_between = plateau.loc[end_level:end_next_level, "refined-target"][
            mask_plateau
        ]
        end_plateau = in_between.index[0] if len(in_between) > 0 else end_level

        # add the reaction time
        reaction_time[seq] = (end_plateau - end_level).total_seconds()

        # transition time
        # find the start of the next plateau
        till_next_level_end = plateau.loc[end_plateau:end_next_level, "refined-target"]
        next_plateau = till_next_level_end != 0

        end_transition = till_next_level_end[next_plateau].index[0]
        transition_time[seq] = (end_transition - end_plateau).total_seconds()
        transition_end[seq] = end_transition
        transition_start[seq] = end_plateau

    return pd.DataFrame(
        {
            "reaction-time": reaction_time,
            "transition-time": transition_time,
            "transition-start": transition_start,
            "transition-end": transition_end,
            "transition-category": transition_cat,
        }
    )


class TransformTimeRelatedInfo(TransformStep):
    """An extraction processing step for Time related measures."""

    def __init__(self, level_id: str):
        super().__init__(
            "plateau",
            transform_function=reaction_time_between_level,
            new_data_set_id="time-info",
            definitions=[
                RawDataValueDefinition("reaction-time", "Reaction time"),
                RawDataValueDefinition(
                    "transition-time", "Transition Time between two plateaus."
                ),
                RawDataValueDefinition("transition-category", "Transition category"),
                RawDataValueDefinition(
                    "transition-start", "Start time of each transition"
                ),
                RawDataValueDefinition("transition-end", "End time of each transition"),
            ],
            level_filter=LevelIdFilter(level_id) & FILTER_INCOMPLETE_SEQUENCES,
        )


class ExtractReactionTimeStats(AggregateRawDataSetColumn):
    """An extraction processing step for Reaction Time measures.

    Parameters
    ----------
    hand
        The hand used to perform the test
    """

    def __init__(self, hand: HandModality):
        definition = MeasureValueDefinitionPrototype(
            measure_name=AV("reaction time", "rt"),
            data_type="float64",
            modalities=[hand.av],
            description=f"The {{aggregation}} time taken to initiate force "
            f"adaptation when the new force target level is "
            f"displayed for the {hand} hand.",
            validator=TEST_DURATION_VALIDATOR,
        )

        super().__init__(
            "time-info",
            "reaction-time",
            aggregations=DEFAULT_AGGREGATIONS,
            definition=definition,
            level_filter=LevelIdFilter(_hand_to_level_id(hand))
            & FILTER_INCOMPLETE_SEQUENCES,
        )


class FilterTransition(FilterIncompleteSequences):
    """Filter incomplete sequences or level without the selected transition."""

    def __init__(
        self, from_pressure: TargetPressureModality, to_pressure: TargetPressureModality
    ):
        self.from_pressure = from_pressure
        self.to_pressure = to_pressure

    def repr(self) -> str:
        """Get representation of the filter."""
        return (
            f"filter complete sequences from {self.from_pressure} to "
            f"{self.to_pressure}"
        )

    def filter(self, levels: Iterable[Level]) -> Set[Level]:
        """Filter for transitions."""
        out = set()
        for level in super().filter(levels):
            transition = [self.from_pressure, self.to_pressure]
            if level.context is not None:
                _, target_state = target_pressure_to_category(level.context)
                for index in range(N_SEQUENCE - 1):
                    if transition == target_state[index : index + 2]:
                        out.add(level)

        return out


def compute_transition(
    data: pd.DataFrame,
    transition: Tuple[TargetPressureModality, TargetPressureModality],
) -> float:
    """Find all the transitions time of the selected transition."""
    mask = data["transition-category"] == transition
    return data.loc[mask, "transition-time"].mean()


class ExtractTransitionTime(ExtractStep):
    """An extraction processing step for Transition Time measures.

    Parameters
    ----------
    hand
        The hand used to perform the test
    transition
        A tuple indicating the two plateaus defining a transition. e.g.:
        {'low','high'} indicates a transition from the ``'low'`` plateau to the
        ``'high'`` plateau.
    """

    def __init__(
        self,
        hand: HandModality,
        transition: Tuple[TargetPressureModality, TargetPressureModality],
    ):
        cat_from, cat_to = transition

        definition = MeasureValueDefinitionPrototype(
            measure_name=AV("transition time", "tt"),
            data_type="float64",
            modalities=[
                hand.av,
                AV(f"from {cat_from} to {cat_to}", f"{cat_from.abbr}_to_{cat_to.abbr}"),
            ],
            aggregation="mean",
            description=f"The mean time taken to adjust the exerted force to "
            f"the next force target level from {cat_from} to "
            f"{cat_to} for the {hand}. It is calculated as the "
            f"time elapsed between the end of one plateau to the "
            f"beginning of the next one.",
            validator=TEST_DURATION_VALIDATOR,
        )
        super().__init__(
            "time-info",
            transform_function=functools.partial(
                compute_transition, transition=transition
            ),
            definition=definition,
            level_filter=LevelIdFilter(_hand_to_level_id(hand))
            & FilterTransition(*transition),
        )


def compute_overshoot(
    pressure: pd.DataFrame,
    _plateau: pd.DataFrame,
    time_info: pd.DataFrame,
    level: Level,
) -> pd.DataFrame:
    """Compute pressure overshoot when changing pressure level."""
    contexts = level.context
    pressure_overshoot = [None] * (N_SEQUENCE - 1)
    for seq in range(N_SEQUENCE - 1):
        t_start = time_info.loc[seq, "transition-start"]
        t_end = time_info.loc[seq, "transition-end"]
        t_cat = time_info.loc[seq, "transition-category"]
        next_plateau = get_target_pressure(contexts, seq + 1)
        if t_cat[0] <= t_cat[1]:
            pressure_overshoot[seq] = max(
                pressure.loc[t_start:t_end, "pressure"] - next_plateau
            )
        else:
            pressure_overshoot[seq] = min(
                pressure.loc[t_start:t_end, "pressure"] - next_plateau
            )
    return pd.DataFrame({"pressure-overshoot": pressure_overshoot})


class TransformOvershoot(TransformStep):
    """An extraction step to measure the overshoot/undershoot.

    Parameters
    ----------
    hand
        The hand on which the overshoot is to be computed.
    """

    def __init__(self, hand: HandModality):
        level_id = f"{hand.abbr}-all"
        new_data_set_id = f"{hand.abbr}-overshoot"
        super().__init__(
            ["screen_ts_resampled", "plateau", "time-info"],
            transform_function=compute_overshoot,
            new_data_set_id=new_data_set_id,
            definitions=[
                RawDataValueDefinition("pressure-overshoot", "Pressure overshoot.")
            ],
            level_filter=LevelIdFilter(level_id) & FILTER_INCOMPLETE_SEQUENCES,
        )


class ExtractOvershootStats(AggregateRawDataSetColumn):
    """An extraction processing step for overshoot measures.

    Parameters
    ----------
    hand
        The hand on which the overshoot measures are to be computed.
    """

    def __init__(self, hand: HandModality):
        level_id = f"{hand.abbr}-all"
        definition = MeasureValueDefinitionPrototype(
            measure_name=AV(
                f"{hand.abbr} pressure overshoot", f"{hand.abbr}-pressure_overshoot"
            ),
            data_type="float64",
            description="The {{aggregation}} overshoot. The overshoot is "
            "defined as the extra force being used when going"
            "from one target to the next one.",
        )

        super().__init__(
            f"{hand.abbr}-overshoot",
            "pressure-overshoot",
            aggregations=DEFAULT_AGGREGATIONS,
            definition=definition,
            level_filter=LevelIdFilter(level_id) & FILTER_INCOMPLETE_SEQUENCES,
        )


class GripTremorMeasures(ProcessingStepGroup):
    """A group of grip processing steps for tremor measures.

    Parameters
    ----------
    hand
        The hand on which the tremor measures are to be computed.
    sensor
        The sensor on which the tremor measures are to be computed.
    """

    def __init__(self, hand: HandModality, sensor: SensorModality):
        data_set_id = str(sensor)
        level_id = f"{hand.abbr}-all"
        steps = [
            RenameColumns(data_set_id, level_id, **USER_ACC_MAP),
            SetTimestampIndex(
                f"{data_set_id}_renamed", DEFAULT_COLUMNS, "ts", duplicates="last"
            ),
            Resample(
                f"{data_set_id}_renamed_ts",
                aggregations=["mean", "ffill"],
                columns=DEFAULT_COLUMNS,
                freq=20,
            ),
            TremorMeasures(
                sensor=sensor, data_set_id=f"{data_set_id}_renamed_ts_resampled"
            ),
        ]
        super().__init__(
            steps,
            task_name=TASK_NAME,
            modalities=[hand.av],
            hand=hand,
            level_filter=level_id,
        )


STEPS: List[ProcessingStep] = []

for _hand in HandModality:
    _hand_abbr = _hand.abbr
    _level_ids = [_hand_abbr] + [f"{_hand_abbr}-0{i}" for i in range(2, 9)]
    _level_id = f"{_hand_abbr}-all"

    # generic steps
    STEPS += [
        # transformation
        ConcatenateLevels(
            _level_id, ["screen", "accelerometer", "gyroscope"], level_filter=_level_ids
        ),
        SetTimestampIndex(
            "screen", ["pressure"], "tsTouch", level_filter=_level_id, duplicates="last"
        ),
        Resample(
            data_set_id="screen_ts",
            aggregations=["pad"],
            columns=["pressure"],
            freq=FREQ_100HZ,
            level_filter=_level_id,
        ),
        TransformPlateau(_level_id),
        TransformRefinedTargetCategory(_level_id),
        TransformPressureError(_level_id),
        TransformTimeRelatedInfo(_level_id),
        TransformOvershoot(hand=_hand),
        # extraction
        ExtractRMSPressure(hand=_hand),
        ExtractAppliedForceStats(hand=_hand),
        ExtractMeanAppliedForceDiff(
            hand=_hand,
            categories_to_diff=CategoryToDiff(
                TargetPressureModality.HIGH, TargetPressureModality.VERY_HIGH
            ),
        ),
        ExtractReactionTimeStats(hand=_hand),
        ExtractThumbPositionVariation(hand=_hand),
        ExtractOvershootStats(hand=_hand),
    ]
    for _plateau in PlateauModality:
        STEPS += [
            # Transformation
            # Time defined plateau definition scenarios
            TransformTimeDefinedPlateau(_level_id, plateau=_plateau),
            TransformRefinedTargetCategory(_level_id, plateau=_plateau),
            TransformPressureError(_level_id, plateau=_plateau),
            # Extraction
            ExtractRMSPressure(hand=_hand, plateau=_plateau),
            ExtractAppliedForceStats(hand=_hand, plateau=_plateau),
            ExtractMeanAppliedForceDiff(
                hand=_hand,
                categories_to_diff=CategoryToDiff(
                    TargetPressureModality.HIGH, TargetPressureModality.VERY_HIGH
                ),
                plateau=_plateau,
            ),
        ]

    # add pressure category specific extractions
    for _cat in TargetPressureModality:
        STEPS += [
            ExtractRMSPressure(hand=_hand, category=_cat),
            ExtractAppliedForceStats(hand=_hand, category=_cat),
        ]
        for _plateau in PlateauModality:
            STEPS += [
                ExtractRMSPressure(hand=_hand, category=_cat, plateau=_plateau),
                ExtractAppliedForceStats(hand=_hand, category=_cat, plateau=_plateau),
            ]

        # transition specific extractions
        for _cat_to in TargetPressureModality:
            if _cat != _cat_to:
                STEPS += [ExtractTransitionTime(hand=_hand, transition=(_cat, _cat_to))]
    STEPS += [
        GripTremorMeasures(_hand, sensor)
        for sensor in [SensorModality.ACCELEROMETER, SensorModality.GYROSCOPE]
    ]

STEPS = [ProcessingStepGroup(steps=STEPS, task_name=TASK_NAME)]

process_grip = process_factory(
    task_name=TASK_NAME,
    steps=STEPS,
    codes=("gripForce", "gripForce-test"),
    supported_type=ADSReading,
)
