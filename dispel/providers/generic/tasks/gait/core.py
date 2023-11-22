"""All generic transformation steps related to the gait."""
from abc import ABCMeta, abstractmethod
from enum import IntEnum
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from dispel.data.core import Reading
from dispel.data.levels import Level
from dispel.data.measures import MeasureValueDefinitionPrototype
from dispel.data.raw import RawDataValueDefinition
from dispel.data.validators import BETWEEN_MINUS_ONE_AND_ONE, GREATER_THAN_ZERO
from dispel.data.values import AbbreviatedValue as AV
from dispel.data.values import ValueDefinition
from dispel.processing.core import Parameter as P
from dispel.processing.core import ProcessResultType
from dispel.processing.data_set import decorated_processing_function, transformation
from dispel.processing.extract import (
    DEFAULT_AGGREGATIONS,
    DEFAULT_AGGREGATIONS_Q95,
    AggregateRawDataSetColumn,
    ExtractStep,
)
from dispel.processing.transform import TransformStep
from dispel.providers.generic.tasks.gait.bout_strategy import BoutStrategy
from dispel.signal.core import (
    integrate_time_series,
    scaled_autocorr,
    uniform_power_spectrum,
)
from dispel.signal.filter import butterworth_low_pass_filter

WALKING_BOUT_DYNAMICS_ROLLING_WINDOW_SIZE = P(
    "WALKING_BOUT_DYNAMICS_ROLLING_WINDOW_SIZE",
    "1000ms",
    description="Walking bout dynamics rolling window size in ms.",
)
WALKING_BOUT_DYNAMICS_ROLLING_STEP_SIZE = P(
    "WALKING_BOUT_DYNAMICS_ROLLING_STEP_SIZE",
    "100ms",
    description="Walking bout dynamics rolling step size in ms.",
)
WALKING_BOUT_DYNAMICS_THRES_UPRIGHT = P(
    "WALKING_BOUT_DYNAMICS_THRES_UPRIGHT",
    0.03,
    description="Walking bout dynamics upright motion threshold in m/s^2.",
)
WALKING_BOUT_DYNAMICS_THRES_MOVING = P(
    "WALKING_BOUT_DYNAMICS_THRES_MOVING",
    0.3,
    description="Walking bout dynamics moving threshold in m/s^2.",
)
WALKING_BOUT_DYNAMICS_THRES_MIN_BOUT = P(
    "WALKING_BOUT_DYNAMICS_THRES_MIN_BOUT",
    1,
    description="Walking bout dynamics minimum bout duration in seconds.",
)
WALKING_BOUT_DYNAMICS_CUTOFF = P(
    "WALKING_BOUT_DYNAMICS_CUTOFF",
    17.0,
    description="Walking bout dynamics acceleration butterworth filter cutoff "
    "freq in Hz.",
)
WALKING_BOUT_TURN_MAX_ANGLE = P(
    "WALKING_BOUT_TURN_MAX_ANGLE",
    60,
    description="The maximum turn angle in degrees to consider straight walking.",
)
WALKING_BOUT_TRUE_FINAL_THRES_MIN_BOUT = P(
    "WALKING_BOUT_TRUE_FINAL_THRES_MIN_BOUT",
    3,
    description="Walking bout final minimum size of walking segments in seconds.",
)
WALKING_BOUT_FALSE_FINAL_THRES_MIN_BOUT = P(
    "WALKING_BOUT_FALSE_FINAL_THRES_MIN_BOUT",
    0.3,
    description="Walking bout final minimum size of non-walking segments in "
    "seconds.",
)
WALKING_BOUT_SKIP_FIRST_SECONDS = P(
    "WALKING_BOUT_SKIP_FIRST_SECONDS",
    10,
    description="The number of seconds to skip from the beginning of the 6mwt data.",
)
REGULARITY_WINDOW_SIZE_IN_STEPS = P[int](
    "REGULARITY_WINDOW_SIZE_IN_STEPS",
    14,
    description="The sliding window size of the signal input to the "
    "auto-correlation.",
)
REGULARITY_WINDOW_STEP_IN_STEPS = P[int](
    "REGULARITY_WINDOW_STEP_IN_STEPS",
    7,
    description="The sliding window step of the signal input to the "
    "auto-correlation.",
)
REGULARITY_FIND_PEAK_DISTANCE = P(
    "REGULARITY_FIND_PEAK_DISTANCE",
    0.3,
    description="Required minimal horiz dist (> 0) in seconds between "
    "neighbouring peaks.",
)
REGULARITY_FIND_PEAK_HEIGHT_CENTILE = P(
    "REGULARITY_FIND_PEAK_HEIGHT_CENTILE",
    80,
    description="Required height threshold as a percentile of all data.",
)
REGULARITY_PEAK_SEARCH_INTERVAL_IN_SEC = P(
    "REGULARITY_PEAK_SEARCH_INTERVAL_IN_SEC",
    4,
    description="The maximum time from the zero lag to search for peaks.",
)
REGULARITY_SIGNAL_FILTER_CUTOFF_FREQ = P(
    "REGULARITY_SIGNAL_FILTER_CUTOFF_FREQ",
    3,
    description="The frequency to use to filter the signal to get rid of peaks.",
)
REGULARITY_SIGNAL_FILTER_ORDER = P(
    "REGULARITY_SIGNAL_FILTER_ORDER",
    2,
    description="The order of the low pass filter of the signal used to "
    "detect peaks/lags.",
)
REGULARITY_LOCAL_SEARCH_SLICE_SIZE = P(
    "REGULARITY_LOCAL_SEARCH_SLICE_SIZE",
    5,
    description="The size in number of samples of the window to search for "
    "refined peaks.",
)


class StepEvent(IntEnum):
    """Generic events for step annotation."""

    UNKNOWN = 0
    INITIAL_CONTACT = 1
    MID_SWING = 2
    FOOT_OFF = 3


class FootUsed(IntEnum):
    """Information on the foot being used for step annotation."""

    LEFT = 1
    RIGHT = 2
    UNKNOWN = 0


DEF_FOOT = RawDataValueDefinition(
    id_="foot", name="foot", data_type="int", description="The foot being used."
)

DEF_EVENT = RawDataValueDefinition(
    id_="event",
    name="event",
    data_type="int",
    description="The event detected at the specific time in the gait cycle.",
)

DEF_BOUT_ID = RawDataValueDefinition(
    id_="bout_id",
    name="bout id",
    description="A series of float starting at 0 incrementing "
    "by one each new walking bout, set to NaN when"
    "the bout is not a walking bout.",
    data_type="float64",
)

DEF_DURATION = RawDataValueDefinition(
    id_="duration",
    name="duration",
    description="A series of float64 containing the duration of the walking"
    "and non-walking bouts.",
    data_type="float64",
)

DEF_BOUT_START = RawDataValueDefinition(
    id_="start_time",
    name="start time",
    description="A series of the start time for each bouts.",
    data_type="datetime64",
)

DEF_BOUT_END = RawDataValueDefinition(
    id_="end_time",
    name="end time",
    description="A series of the end time for each bouts.",
    data_type="datetime64",
)

DEF_DETECTED = RawDataValueDefinition(
    id_="detected_walking",
    name="detected walking",
    description="A boolean time series flag to True when thebout is a walking bout.",
    data_type="bool",
)

DEF_STEP_DURATION = RawDataValueDefinition(
    id_="step_duration",
    name="step duration",
    description="Step duration is the amount of time to perform one step.",
    unit="s",
    data_type="float64",
)

DEF_STEP_POWER = RawDataValueDefinition(
    id_="step_power",
    name="step power",
    description="Step power is defined as the integral of centered magnitude "
    "acceleration for the period of a single step.",
    unit="m^2/s^3",
    data_type="float64",
)

DEF_STEP_INTENSITY = RawDataValueDefinition(
    id_="step_intensity",
    name="step intensity",
    description="Step intensity is defined as the RMS of the magnitude of the "
    "acceleration for the period of a single step.",
    unit="m/s^2",
    data_type="float64",
)

DEF_STEP_REGULARITY = RawDataValueDefinition(
    id_="step_regularity",
    name="step regularity",
    description="Step regularity is defined as the first maximum of the "
    "autocorrelation function of a signal.",
    data_type="float64",
)

DEF_STRIDE_REGULARITY = RawDataValueDefinition(
    id_="stride_regularity",
    name="stride regularity",
    description="Stride regularity is defined as the second maximum of the "
    "autocorrelation function of a signal.",
    data_type="float64",
)

DEF_LAG_STEP_REGULARITY = RawDataValueDefinition(
    id_="lag_step_regularity",
    name="lag step regularity",
    description="Lag of step regularity is defined as delay in seconds at"
    "the first maximum of the autocorreleration of a signal.",
    data_type="float64",
)

DEF_LAG_STRIDE_REGULARITY = RawDataValueDefinition(
    id_="lag_stride_regularity",
    name="lag stride regularity",
    description="Lag of stride regularity is defined as delay in seconds "
    "at the second maximum of the autocorreleration of a signal.",
    data_type="float64",
)


class DetectStepsWithoutBoutsBase(TransformStep, metaclass=ABCMeta):
    """Generic detect steps transform without walking bouts."""

    definitions = [DEF_FOOT, DEF_EVENT]


class DetectStepsProcessingBase(TransformStep, metaclass=ABCMeta):
    """An abstract base class for step detection.

    Given the step detection algorithm specified through the method argument,
    e.g.: :func:`dispel.providers.generic.tasks.gait.lee.detect_steps`, the transform
    step run the step detection on each of the walking bouts and create a
    generic pandas.DataFrame with annotated events as in Bourke et. al.
    """

    definitions = [DEF_FOOT, DEF_EVENT, DEF_BOUT_ID]

    @transformation
    def _detect_steps(self, bouts, *data_sets):
        """For each walking bout, run the step_detection."""
        # pylint: disable=no-member
        if not (
            hasattr(self, "step_detection_method")
            and callable(self.step_detection_method)
        ):
            raise NotImplementedError(
                "The step detection method has not been defined correctly as "
                "a callable."
            )
        steps = []
        walk_bouts = bouts[bouts["detected_walking"]]
        for _, bout in walk_bouts.iterrows():
            bout_data_sets = [
                data_set[bout["start_time"] : bout["end_time"]]
                for data_set in data_sets
            ]
            bout_steps = self.step_detection_method(*bout_data_sets)
            bout_steps["bout_id"] = bout["bout_id"]
            steps.append(bout_steps)
        if len(steps) > 0:
            return pd.concat(steps).sort_index()
        return pd.DataFrame(columns=["foot", "event", "bout_id"])


class GaitBoutExtractStep(ExtractStep):
    """Base class for gait bouts measure extraction."""

    bout_strategy: BoutStrategy

    def __init__(self, *args, **kwargs):
        bout_strategy = kwargs.pop("bout_strategy", None)
        super().__init__(*args, **kwargs)
        if bout_strategy:
            self.bout_strategy = bout_strategy

    def process_level(
        self, level: Level, reading: Reading, **kwargs
    ) -> ProcessResultType:
        """Overwrite process level."""
        # pylint: disable=unpacking-non-sequence
        bouts, *data_sets = self.get_data_frames(level)

        filtered_data_sets = self.bout_strategy.get_view(bouts, *data_sets)

        for function, func_kwargs in self.get_transform_functions():
            merged_kwargs = kwargs.copy()
            merged_kwargs.update(func_kwargs)

            yield from self.wrap_result(
                decorated_processing_function(
                    function, filtered_data_sets, reading, level
                ),
                level,
                reading,
                **merged_kwargs,
            )


def get_step_detector_from_data_set_ids(
    data_set_ids: Union[str, List[str]], position: int
) -> str:
    """Get step detector from data_set_ids."""
    return data_set_ids[position].split("_")[0]


class TransformStepDetection(TransformStep):
    """
    A transform step that creates a generic data set for step analysis.

    Given the step detection algorithm specified through the method argument,
    e.g.: :func:`dispel.providers.generic.tasks.gait.lee.detect_steps`, the transform
    step create a generic data frame with annotated events as in Bourke et. al.
    """

    definitions = [DEF_FOOT, DEF_EVENT]


def step_count(data: pd.DataFrame) -> int:
    """Count the number of steps.

    Parameters
    ----------
    data
        A pandas data frame containing the detected steps.

    Returns
    -------
    int
        The number of steps.
    """
    return (data["event"] == StepEvent.INITIAL_CONTACT).sum()


class ExtractStepCount(GaitBoutExtractStep):
    """Extract step count.

    Parameters
    ----------
    data_set_ids
        A list of two elements with the id of the walking bout data set and
        the id of the step_dataset containing the detected steps formatted as
        the output of TransformStepDetection.
    level_filter
        An optional :class:`~dispel.processing.level.LevelFilter` to determine
        the levels for extraction.
    """

    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("step count", "sc"),
        data_type="uint16",
        validator=GREATER_THAN_ZERO,
        description="The number of steps detected with {step_detector} "
        "algorithm with the bout strategy {bout_strategy_repr}.",
    )

    transform_function = step_count

    def get_definition(self, **kwargs) -> ValueDefinition:
        """Get the measure definition."""
        step_detector = get_step_detector_from_data_set_ids(
            next(iter(self.get_data_set_ids())), 1
        )
        kwargs["step_detector"] = step_detector
        return super().get_definition(**kwargs)


def power_bout_div_steps(
    acc_magnitude: pd.Series,
    step_dataset: pd.DataFrame,
    walking_bouts: Optional[pd.DataFrame] = None,
) -> float:
    """Compute the Step Power based on bout power divided by number of steps.

    The step power is defined as the integral of centered magnitude
    acceleration divided by the number of steps. For more information
    please see the section 2.3 of [1]_.

    Parameters
    ----------
    acc_magnitude
        A pandas series containing the magnitude of the acceleration.
    step_dataset
        A pandas data frame containing the detected steps formatted as the
        output of TransformStepDetection.
    walking_bouts
        A dataframe with the concatenated walking bouts.

    Returns
    -------
    float
        The step power.

    References
    ----------
    .. [1] Cheng WY. et al. (2018) Large-Scale Continuous Mobility Monitoring
       of Parkinson’s Disease Patients Using Smartphones.
    """

    def _bout_power(start: int, end: int, magnitude: pd.Series):
        """Compute the power of a walking bout."""
        abs_acc = magnitude[start:end].dropna().abs()
        return compute_step_power_single(abs_acc)

    step_dataset = step_dataset[step_dataset.event == StepEvent.INITIAL_CONTACT]
    if walking_bouts is None:
        t_start = step_dataset.index[0]
        t_end = step_dataset.index[-1]
        return _bout_power(t_start, t_end, acc_magnitude) / step_count(step_dataset)
    # keep true walking bout only
    walk_bouts = walking_bouts[walking_bouts["detected_walking"]]

    # Initialize the walking power to zero
    # The step power over several walking bout is defined as the sum
    # of the power of each walking bout divided by the total number of
    # steps, it is different from the sum of the step power for each bouts.
    walking_power = 0
    step_counts = 0
    for bout_it in range(len(walk_bouts)):
        t_start = walk_bouts.iloc[bout_it]["start_time"]
        t_end = walk_bouts.iloc[bout_it]["end_time"]
        walking_power += _bout_power(t_start, t_end, acc_magnitude)
        step_counts += step_count(step_dataset[t_start:t_end])
    if step_counts == 0:
        return 0
    return walking_power / step_counts


class ExtractPowerBoutDivSteps(GaitBoutExtractStep):
    """Extract step to compute the Step Power.

    Parameters
    ----------
    data_set_ids
        The data set ids corresponding to first the walking bouts
        then the signal to take as the magnitude of acceleration and
        the step dataset.
    level_filter
        An optional :class:`~dispel.processing.level.LevelFilter` to determine
        the levels for extraction.
    """

    def __init__(self, data_set_ids: str, **kwargs):
        step_detector = get_step_detector_from_data_set_ids(data_set_ids, 2)

        def _function(acc_magnitude, step_dataset, level):
            return power_bout_div_steps(
                acc_magnitude["vertical_acc"],
                step_dataset,
                level.get_raw_data_set("movement_bouts").data,
            )

        description = (
            "The integral of the centered acceleration magnitude "
            "between the first and last step divided by the "
            f"number of steps computed with {step_detector} "
            "algorithm and the bout strategy {bout_strategy_repr}."
        )

        definition = MeasureValueDefinitionPrototype(
            measure_name=AV("step power", "sp"),
            data_type="int16",
            validator=GREATER_THAN_ZERO,
            description=description,
        )
        super().__init__(data_set_ids, _function, definition, **kwargs)


def compute_step_power_single(acc_norm: pd.Series) -> Union[float, np.ndarray]:
    """Compute the Step Power for a single step.

    The step power is defined as the integral of centered magnitude
    acceleration for the period of a single step.

    Parameters
    ----------
    acc_norm
        A pandas series containing the magnitude of the acceleration.

    Returns
    -------
    float
        The step power.

    References
    ----------
    .. [1] Cheng WY. et al. (2018) Large-Scale Continuous Mobility Monitoring
       of Parkinson’s Disease Patients Using Smartphones.
    """
    return integrate_time_series((acc_norm - acc_norm.mean()) ** 2)


def compute_step_intensity_single(acc_norm: pd.Series) -> Union[float, np.ndarray]:
    """Compute the Step Intensity for a single step.

    The step intensity is defined as the RMS of the magnitude of
    acceleration for the period of a single step.

    Parameters
    ----------
    acc_norm
        A pandas series containing the magnitude of the acceleration.

    Returns
    -------
    float
        The step intensity.

    References
    ----------
    .. [1] Angelini L et al. (2020) Is a Wearable Sensor-Based Characterisation
       of Gait Robust Enough to Overcome Differences Between Measurement
       Protocols? A Multi-Centric Pragmatic Study in Patients with MS.
       https://doi.org/10.3390/s20010079
    """
    return np.sqrt(np.mean(acc_norm))


def compute_step_vigor_multiple(
    acc: pd.DataFrame, step_dataset: pd.DataFrame, component: str, bout_filter: bool
) -> pd.DataFrame:
    """Compute the step vigor for multiple steps.

    The step vigor is defined as the step power and step intensity properties.

    Parameters
    ----------
    acc
        A pandas DataFrame containing the at least one column with the intended
         acceleration signal component to be used for step power.
    step_dataset
        A pandas series containing at the minimum the the times of the initial
        contacts.
    component
        A str indicating the column name of acc to be used.
    bout_filter
        A boolean indicating whether to take into account bouts in the
        transformation of the dataset.

    Returns
    -------
    pandas.DataFrame
        The step power and intensity values.

    """

    def _transform_step_vigor(
        signal: pd.Series, start: pd.Timestamp, end: pd.Timestamp, bout_id: int
    ) -> pd.Series:
        """Wrap compute step vigor single."""
        step_power = compute_step_power_single(signal.loc[start:end])
        step_intensity = compute_step_intensity_single(signal.loc[start:end])
        series = pd.Series(
            {"index": start, "step_power": step_power, "step_intensity": step_intensity}
        )
        if bout_id is not None:
            series["bout_id"] = bout_id
        return series

    # get a copy to be able to manipulate as we wish
    step_ds = step_dataset.copy()
    # add next initial contact time
    step_ds["IC_next"] = step_ds["IC"].shift(-1)
    if bout_filter:
        # find bout transitions and remove last step in each bout
        idx = (step_ds.bout_id.diff() == 0).shift(-1).fillna(False)
        # filter only steps that have the next IC in the same bout
        step_ds = step_ds[idx].reset_index(drop=True)

    if len(step_ds) == 0:
        return pd.DataFrame(columns=["step_power", "step_intensity", "bout_id"])
    return step_ds.apply(
        lambda x: _transform_step_vigor(
            signal=acc[component],
            start=x["IC"],
            end=x["IC_next"],
            bout_id=x.bout_id if bout_filter else None,
        ),
        axis=1,
    ).set_index("index")


class TransformStepVigor(TransformStep):
    """Transform step vigor.

    Create a data set with columns ``step_power``, ``step_intensity``,  and
    ``bout_id`` indicating the power, intensity and respective bout identifier
    of each step.
    """

    def __init__(self, component: str, *args, **kwargs):
        self.component = component
        super().__init__(*args, **kwargs)

    @transformation
    def _step_vigor(
        self, acc_magnitude: pd.DataFrame, step_dataset: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute the step vigor properties (power and intensity)."""
        return compute_step_vigor_multiple(
            acc=acc_magnitude,
            step_dataset=step_dataset,
            component=self.component,
            bout_filter=True,
        )

    definitions = [DEF_STEP_POWER, DEF_STEP_INTENSITY, DEF_BOUT_ID]

    def get_new_data_set_id(self) -> str:
        """Overwrite new data set id."""
        return f"{self.data_set_ids[1]}_step_vigor"  # type: ignore


class TransformStepVigorWithoutBout(TransformStep):
    """Transform step vigor without bout.

    Create a data set with two columns ``step_power`` and ``step_intensity``
    indicating the step power and intensity, respectively for each step.
    """

    def __init__(self, component: str, *args, **kwargs):
        self.component = component
        super().__init__(*args, **kwargs)

    @transformation
    def _step_vigor(
        self, acc_magnitude: pd.DataFrame, step_dataset: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute the step vigor properties (power and intensity)."""
        return compute_step_vigor_multiple(
            acc=acc_magnitude,
            step_dataset=step_dataset,
            component=self.component,
            bout_filter=False,
        )

    definitions = [DEF_STEP_POWER, DEF_STEP_INTENSITY]

    def get_new_data_set_id(self) -> str:
        """Overwrite new data set id."""
        return f"{self.data_set_ids[1]}_step_vigor"  # type: ignore


class TransformStepDuration(TransformStep):
    """Transform step duration.

    Create a data set with one column ``step_duration`` indicating the amount
    of time to perform each steps.
    """

    @transformation
    def _step_duration(self, data: pd.DataFrame) -> pd.DataFrame:
        """Measure the step duration."""
        list_df = []
        for bout_id in data.bout_id.unique():
            res = data[data.bout_id == bout_id]
            ic_mask = res["event"] == StepEvent.INITIAL_CONTACT
            list_df.append(
                pd.DataFrame(
                    {
                        "step_duration": res.index[ic_mask]
                        .to_series()
                        .diff()
                        .dt.total_seconds(),
                        "bout_id": bout_id,
                    }
                ).set_index(res.index[ic_mask])
            )
        if len(list_df) > 0:
            return pd.concat(list_df)
        return pd.DataFrame(columns=["step_duration", "bout_id"])

    definitions = [DEF_STEP_DURATION, DEF_BOUT_ID]

    def get_new_data_set_id(self) -> str:
        """Overwrite new data set id."""
        return f"{self.data_set_ids}_step_duration"


class TransformStepDurationWithoutBout(TransformStep):
    """Transform step duration without using walking bout.

    Create a data set with one column ``step_duration`` indicating the amount
    of time to perform each steps.
    """

    @transformation
    def _step_duration(self, data: pd.DataFrame) -> pd.DataFrame:
        """Measure the step duration."""
        ic_mask = data["event"] == StepEvent.INITIAL_CONTACT
        return data.index[ic_mask].to_series().diff().dt.total_seconds()

    definitions = [DEF_STEP_DURATION]

    def get_new_data_set_id(self) -> str:
        """Overwrite new data set id."""
        return f"{self.data_set_ids}_step_duration"


class GaitBoutAggregateStep(AggregateRawDataSetColumn):
    """Base class for gait bouts agg raw data set measure extraction."""

    bout_strategy: BoutStrategy

    def __init__(self, *args, **kwargs):
        bout_strategy = kwargs.pop("bout_strategy", None)
        kwargs_copy = kwargs.copy()
        if "modalities" in kwargs:
            kwargs_copy.pop("modalities")

        super().__init__(*args, **kwargs_copy)
        if bout_strategy:
            self.bout_strategy = bout_strategy

    def process_level(
        self, level: Level, reading: Reading, **kwargs
    ) -> ProcessResultType:
        """Overwrite process_level."""
        # pylint: disable=unpacking-non-sequence
        bouts, *data_sets = self.get_data_frames(level)
        filtered_data_sets = self.bout_strategy.get_view(bouts, *data_sets)

        for function, func_kwargs in self.get_transform_functions():
            merged_kwargs = kwargs.copy()
            merged_kwargs.update(func_kwargs)
            yield from self.wrap_result(
                decorated_processing_function(
                    function, filtered_data_sets, reading, level
                ),
                level,
                reading,
                **merged_kwargs,
            )


class ExtractStepPowerAll(GaitBoutAggregateStep):
    """Extract step power related measures.

    Parameters
    ----------
    data_set_ids
        The data set ids that will be considered to extract step power
        measures.
    """

    def __init__(self, data_set_ids: List[str], **kwargs):
        step_detector = get_step_detector_from_data_set_ids(data_set_ids, 1)
        description = (
            "The {aggregation} step power computed with the  "
            f"{step_detector} algorithm. It is computed with the bout "
            "strategy {bout_strategy_repr}."
        )

        definition = MeasureValueDefinitionPrototype(
            measure_name=AV("step power", "sp"),
            data_type="float64",
            unit="m^2/s^3",
            validator=GREATER_THAN_ZERO,
            description=description,
        )

        super().__init__(
            data_set_ids, "step_power", DEFAULT_AGGREGATIONS_Q95, definition, **kwargs
        )


class ExtractStepIntensityAll(GaitBoutAggregateStep):
    """Extract step intensity related measures.

    Parameters
    ----------
    data_set_ids
        The data set ids that will be considered to extract step intensity
        measures.
    """

    def __init__(self, data_set_ids: List[str], **kwargs):
        step_detector = get_step_detector_from_data_set_ids(data_set_ids, 1)
        description = (
            "The {aggregation} step intensity computed with the  "
            f"{step_detector} algorithm. It is computed with the bout "
            "strategy {bout_strategy_repr}."
        )

        definition = MeasureValueDefinitionPrototype(
            measure_name=AV("step intensity", "si"),
            data_type="float64",
            unit="m/s^2",
            validator=GREATER_THAN_ZERO,
            description=description,
        )

        super().__init__(
            data_set_ids, "step_intensity", DEFAULT_AGGREGATIONS, definition, **kwargs
        )


class ExtractStepDurationAll(GaitBoutAggregateStep):
    """Extract step duration related measures.

    Parameters
    ----------
    data_set_ids
        The data set ids that will be considered to extract step duration
        measures.
    """

    def __init__(self, data_set_ids: List[str], **kwargs):
        step_detector = get_step_detector_from_data_set_ids(data_set_ids, 1)
        description = (
            "The {aggregation} time of a step detected with the "
            f"{step_detector} algorithm. It is computed with the bout "
            "strategy {bout_strategy_repr}."
        )

        definition = MeasureValueDefinitionPrototype(
            measure_name=AV("step duration", "step_dur"),
            data_type="float64",
            unit="s",
            validator=GREATER_THAN_ZERO,
            description=description,
        )

        super().__init__(
            data_set_ids, "step_duration", DEFAULT_AGGREGATIONS, definition, **kwargs
        )


class AverageRollingVerticalAcceleration(TransformStep):
    """
    Extract and smooth vertical acceleration (in unit `G`).

    The vertical acceleration is smoothed by using a centered moving average
    method.

    Parameters
    ----------
    data_set_id
        The id corresponding to the preprocessed accelerometer signal.
    axis
        The axis on which the centered moving average is computed.
    rolling_window
        The window size used in the centered moving average operation. Default
        value 5 has been determined empirically in order to provide the best
        results.
    """

    def __init__(
        self,
        data_set_id: str,
        axis: str = "userAccelerationX",
        rolling_window: int = 5,
        change_sign: bool = False,
    ):
        def _transform_vertical_acc(data: pd.DataFrame) -> pd.DataFrame:
            vert_acc = data[axis].rename("vertical_acc").to_frame()
            sign = -1.0 if change_sign else 1.0
            return vert_acc.rolling(window=rolling_window, center=True).mean() * sign

        super().__init__(
            data_set_ids=data_set_id,
            transform_function=_transform_vertical_acc,
            new_data_set_id="vertical_acceleration",
            definitions=[
                RawDataValueDefinition(
                    id_="vertical_acc",
                    name="Vertical Acceleration",
                    unit="-G",
                    description="The vertical acceleration is computed by "
                    "extracting first the preprocessed "
                    "accelerometer signal then applying a "
                    "centered moving average with a rolling "
                    "windows of size {window_size} and finally "
                    "multiplying by minus one to have a "
                    "positively oriented vertical acceleration",
                    data_type="float64",
                )
            ],
        )


def walking_detection_harmonic(
    vertical_acc: pd.DataFrame, abs_thresh: float = 0.01
) -> pd.DataFrame:
    """Detect walking bouts.

    A sliding window of three seconds is moved through the time series of
    vertical acceleration to detect walking bouts. Walking is identified based
    on spectral measures computed on the temporal window. More specifically,
    walking is seen if the maximum amplitude of the power spectral density
    in the walking frequency band [0.6 Hz - 2.0 Hz] is more significant (and
    more distant than a given threshold) than the maximum amplitude of the
    power spectral density of the non-walking frequency band [0 - 0.6 Hz].

    Parameters
    ----------
    vertical_acc
        A smoothed version of the vertical acceleration typically obtained
        using the transform step `~AverageRollingVerticalAcceleration`.
    abs_thresh
        A threshold defining the minimum distance that should separate the
        amplitude of the walking harmonic from the amplitude of the
        non-walking harmonic.

    Returns
    -------
    pandas.DataFrame
        A data frame with the maximum walking and non-walking harmonic as well
        as the detected walking.

    Raises
    ------
    ValueError
        Raises a value error if vertical_acc is not sampled at a constant
        sampling rate.
    """
    # Define the size of the temporal window
    # 3 seconds as in CWT
    if vertical_acc.index.freq is None:
        raise ValueError(
            "One is trying to detect walking bout"
            "on a time series that has not been resampled to "
            "a constant sampling rate."
        )
    fs = 1 / vertical_acc.index.freq.delta.total_seconds()
    slide_duration = round(3 * fs)

    # Storing results
    start_time = []
    end_time = []
    max_walk_harmonic = []
    max_non_walk_harmonic = []

    # Initialize detected walking to false
    detected_walking = []

    # Initialize a counter
    counter = 0

    while counter < len(vertical_acc) - 1:
        # Select a sub part of the signal, a temporal window of three seconds
        end_it = min(counter + slide_duration, len(vertical_acc) - 1)
        start_it = counter
        if (end_it - counter) < slide_duration:
            start_it = (end_it - slide_duration) + 1

        sub_data = vertical_acc[start_it:end_it]
        start_time.append(vertical_acc.index[counter])
        end_time.append(vertical_acc.index[end_it])

        # Compute the Power Spectrum
        ps_data = uniform_power_spectrum(sub_data["vertical_acc"].dropna())

        # Compute walking harmonic between 0.6 and 2Hz
        walking_harmonic = ps_data[(ps_data.index > 0.6) & (ps_data.index < 2)]

        # Compute non-walking harmonic between 0 and 0.6 Hz
        non_walking_harmonic = ps_data[ps_data.index <= 0.6]

        # Keep the maximum
        max_walk_harmonic.append(walking_harmonic.max())
        max_non_walk_harmonic.append(non_walking_harmonic.max())

        # Condition to detect walking
        # Walking is detected if the maximum walking harmonic amplitude is
        # greater than the non-walking harmonic threshold by absolute_threshold
        if (max_walk_harmonic[-1] - max_non_walk_harmonic[-1]) > abs_thresh:
            # Update walking
            detected_walking.append(True)
        else:
            detected_walking.append(False)
        counter += slide_duration

    return pd.DataFrame(
        {
            "start_time": start_time,
            "end_time": end_time,
            "walking_harmonic_max": max_walk_harmonic,
            "non_walking_harmonic_max": max_non_walk_harmonic,
            "detected_walking": detected_walking,
        }
    )


def movement_detection(
    acc_ts_rotated_resampled_detrend: pd.DataFrame,
) -> pd.DataFrame:
    """Detect movement bouts.

    Parameters
    ----------
    acc_ts_rotated_resampled_detrend
        A pd.DataFrame containing the accelerometer signal rotated in a gravity
        based coordinate frame with x pointing towards gravity

    Returns
    -------
    pandas.DataFrame
        A pd.DataFrame containing the start/end timestamp, duration of each
        detected movement or non-movement bout
    """
    # 1. low pass filter
    acc_filtered = acc_ts_rotated_resampled_detrend.apply(
        butterworth_low_pass_filter,
        cutoff=WALKING_BOUT_DYNAMICS_CUTOFF.value,
        zero_phase=True,
        axis=0,
    )

    # 2. Extract measures from a moving sliding window (mean_x and sum_std)
    mean_x = (
        acc_filtered.loc[:, "userAccelerationX"]
        .abs()
        .rolling(window=WALKING_BOUT_DYNAMICS_ROLLING_WINDOW_SIZE.value, center=True)
        .mean()
    )
    std = acc_filtered.rolling(
        window=WALKING_BOUT_DYNAMICS_ROLLING_WINDOW_SIZE.value, center=True
    ).std()
    sum_std = std.sum(axis=1)

    mean_x = mean_x.resample(WALKING_BOUT_DYNAMICS_ROLLING_STEP_SIZE.value).first()
    sum_std = sum_std.resample(WALKING_BOUT_DYNAMICS_ROLLING_STEP_SIZE.value).first()

    # 3. apply thresholds to determine:
    #  - upright orientation (mean(a_v))
    #  - moving or not (ssd)
    condition_upright = mean_x > WALKING_BOUT_DYNAMICS_THRES_UPRIGHT.value
    condition_moving = sum_std > WALKING_BOUT_DYNAMICS_THRES_MOVING.value

    # 4. determine start/stop times upright and moving (a_v and ssd),
    # analyse all bouts
    walking = condition_upright & condition_moving
    movement_bouts = pd.DataFrame(walking.rename("detected_walking"))
    # 5. combine consecutive bouts of same label into one single bout
    movement_bouts = format_walking_bouts(movement_bouts)
    # 6. get last time available
    t_end = movement_bouts["end_time"].iloc[-1]

    # 7. get only bouts above 1 second
    movement_bouts = movement_bouts[
        movement_bouts["duration"] > WALKING_BOUT_DYNAMICS_THRES_MIN_BOUT.value
    ]

    # 8. set end time columns as the start time of the next segment
    movement_bouts.reset_index(inplace=True)
    end_time = movement_bouts.loc[:, "start_time"].shift(-1)
    movement_bouts.loc[:, "end_time"] = end_time
    # 9. set the last end time that has no next time as the last time in signal
    movement_bouts.iloc[-1, movement_bouts.columns.get_indexer(["end_time"])] = t_end
    # 10. combine consecutive bouts of same label into one single bout
    movement_bouts = format_walking_bouts(movement_bouts, "start_time", "end_time")

    return movement_bouts


class DetectWalking(TransformStep):
    """Identify walking in three seconds fixed temporal windows."""

    def __init__(self):
        super().__init__(
            data_set_ids="vertical_acceleration",
            transform_function=walking_detection_harmonic,
            new_data_set_id="walking_fixed_windows",
            definitions=[
                RawDataValueDefinition(
                    id_="start_time",
                    name="start time",
                    description="A series of the start time for each time window.",
                    data_type="datetime64",
                ),
                RawDataValueDefinition(
                    id_="end_time",
                    name="end time",
                    description="A series of the end time for each time window.",
                    data_type="datetime64",
                ),
                RawDataValueDefinition(
                    id_="detected_walking",
                    name="detected walking",
                    description="A boolean time series flag to True when the"
                    "temporal window is a walking bout.",
                    data_type="bool",
                ),
                RawDataValueDefinition(
                    id_="walking_harmonic_max",
                    name="walking harmonic max",
                    description="A series of maximum of the walking harmonic "
                    "for each time window.",
                    data_type="float64",
                ),
                RawDataValueDefinition(
                    id_="non_walking_harmonic_max",
                    name="non walking harmonic max",
                    description="A series of maximum of the non-walking "
                    "harmonic for each time window.",
                    data_type="float64",
                ),
            ],
        )


def format_walking_bouts(
    walking_segments: pd.DataFrame,
    start_time_col: str = "ts",
    end_time_col: str = "ts",
    bool_col: str = "detected_walking",
) -> pd.DataFrame:
    """Group consecutive flags of same type in a single bin.

    Parameters
    ----------
    walking_segments
        a pd.DataFrame containing the flags
    start_time_col
        the start time column name
    end_time_col
        the end time column name
    bool_col
        the name of the column to be looked for activity

    Returns
    -------
    pandas.DataFrame
        The dataframe with grouped consecutive flags of same type
    """
    walking_segments.reset_index(inplace=True)
    group_id = (
        walking_segments[bool_col].diff().abs().cumsum().fillna(0).astype(int) + 1
    )
    walking_segments["group_id"] = group_id

    start_time = (
        walking_segments.groupby("group_id")
        .first()[start_time_col]
        .rename("start_time")
    )
    end_time = (
        walking_segments.groupby("group_id").last()[end_time_col].rename("end_time")
    )
    duration = (end_time - start_time).dt.total_seconds().rename("duration")
    detected_walking = walking_segments.groupby("group_id").first()[bool_col]

    windows_formatted = pd.concat(
        [start_time, end_time, detected_walking, duration], axis=1
    )
    windows_formatted["bout_id"] = None
    windows_formatted.loc[windows_formatted[bool_col], "bout_id"] = np.arange(
        windows_formatted[bool_col].sum()
    )

    return windows_formatted


def get_mask_from_intervals(
    intervals: pd.DataFrame,
    ts_index: pd.DatetimeIndex,
    bool_column: str = "detected_walking",
    start_column: str = "start_time",
    end_column: str = "end_time",
) -> pd.Series:
    """Get a boolean mask indicating walking or not walking in a signal.

    Parameters
    ----------
    intervals
         pd.DataFrame: contains start_time and end_time for each interval/bout
    ts_index
        pd.DatetimeIndex: a datetime index of the signal to be masked
    bool_column
        str indicating the column of boolean type to be used for the mask
    start_column
        str indicating the column of the start time of the interval
    end_column
        str indicating the column of the end time of the interval

    Returns
    -------
    pandas.Series
        contains the boolean mask of len(timestamp_index)
    """
    # a list of len n_walking_bouts containing all indexes in each bout
    if bool_column == "":
        index_true = [
            np.where((ts_index >= row[start_column]) & (ts_index < row[end_column]))[
                0
            ].tolist()
            for i, row in intervals.iterrows()
        ]

    else:
        index_true = [
            np.where((ts_index >= row[start_column]) & (ts_index < row[end_column]))[
                0
            ].tolist()
            for i, row in intervals.iterrows()
            if row[bool_column]
        ]

    # flatten list of lists to a single list
    index_true_flat = [l1 for l2 in index_true for l1 in l2]
    # convert to timestamp index
    ts_index_true = ts_index[index_true_flat]
    # create a pd.Series of boolean type and initialize all to False
    mask = pd.Series(False, index=ts_index, dtype="bool")
    # update to True only indexes found above
    mask.loc[ts_index_true] = True

    return mask


def walking_placement_detection(
    movement_bouts: pd.DataFrame, placement_bouts: pd.DataFrame, placement: str = "belt"
) -> pd.DataFrame:
    """Detect intersection of walking and intended placement.

    Parameters
    ----------
    movement_bouts
        A pd.DataFrame containing the intervals of the walking bouts
    placement_bouts
        A pd.DataFrame containing the intervals of the placement bouts
    placement
        A 'str' containing the label of the intended placement

    Returns
    -------
    pandas.DataFrame
        A pd.DataFrame containing the start/end timestamp, duration of each
        detected walking or no walking bout considering intended placement
    """
    # construct a time-series from the intervals
    start_time = movement_bouts.iloc[0, :].loc["start_time"]
    end_time = movement_bouts.iloc[-1, :].loc["end_time"]
    ts_index = pd.date_range(start_time, end_time, freq="20ms")

    placement_bouts["detected_target_placement"] = False
    placement_bouts.loc[
        placement_bouts.loc[:, "placement"] == placement, "detected_target_placement"
    ] = True

    # convert intervals into time-series
    walking_mask = get_mask_from_intervals(movement_bouts, ts_index, "detected_walking")
    placement_mask = get_mask_from_intervals(
        placement_bouts, ts_index, "detected_target_placement"
    )

    # apply a logical AND to find intersection of walking and target placement
    walking_placement_mask = walking_mask & placement_mask
    walking_placement_mask = walking_placement_mask.reset_index()

    walking_placement_mask = walking_placement_mask.rename(
        columns={"index": "ts", 0: "detected_walking"}
    )

    # format gaps to convert from a mask time-series to an interval format
    walking_placement_bouts = format_walking_bouts(walking_placement_mask)

    # store end time
    t_end = movement_bouts["end_time"].iloc[-1]

    # filter only bouts of duration more than a threshold
    walking_placement_bouts = walking_placement_bouts[
        walking_placement_bouts["duration"] > WALKING_BOUT_DYNAMICS_THRES_MIN_BOUT.value
    ]

    # apply end time
    movement_bouts.iloc[-1, movement_bouts.columns.get_indexer(["end_time"])] = t_end

    # reformat bout to fill the gaps
    walking_placement_bouts = format_walking_bouts(
        walking_placement_bouts, "start_time", "end_time"
    )

    return walking_placement_bouts


def remove_short_bouts(
    data: pd.DataFrame,
    true_thres: float = WALKING_BOUT_TRUE_FINAL_THRES_MIN_BOUT.value,
    false_thres: float = WALKING_BOUT_FALSE_FINAL_THRES_MIN_BOUT.value,
    bool_col: str = "detected_walking",
) -> pd.DataFrame:
    """Remove short walking and non-walking bouts.

    Parameters
    ----------
    data
        A pd.DataFrame containing the intervals of the bouts with including
        at least `bool_col` and `duration` as columns
    true_thres
        A float indicating the minimum duration of a true bout
    false_thres
        A float indicating the minimum duration of a false bout
    bool_col
        A 'str' containing the column name of the boolean

    Returns
    -------
    pandas.DataFrame
        A pd.DataFrame containing the filtered bout data satisfying the
        bout type and duration criteria.
    """
    # if duration of a bout greater than a threshold
    is_true_min_duration = data["duration"] > true_thres
    # ... AND if that bout is walking
    is_true = data[bool_col]
    # OR
    # if duration of a bout greater than another threshold
    is_false_min_duration = data["duration"] > false_thres
    # ... AND that bout is not walking
    is_false = ~data[bool_col]
    # ... then keep only these data
    is_min_duration = (is_true & is_true_min_duration) | (
        is_false & is_false_min_duration
    )

    return data[is_min_duration]


def walking_placement_no_turn_detection(
    movement_bouts: pd.DataFrame,
    placement_bouts: pd.DataFrame,
    turn_bouts: pd.DataFrame,
    placement_label: str = "belt",
) -> pd.DataFrame:
    """Intersect walking bouts, placement and no turn constraints.

    This function finds all the walking bout and apply a logical and with the
    mask of the walking bout, the placement detection that respects the label
    e.g.: 'belt' and the non-turn detection mask.
    A postprocessing step to remove short bouts and fill in gaps at the
    beginning and end of the signal is applied.

    Parameters
    ----------
    movement_bouts
        A pd.DataFrame containing the intervals of the walking bouts
    placement_bouts
        A pd.DataFrame containing the intervals of the placement bouts
    turn_bouts
        A pd.DataFrame containing the intervals of the turns
    placement_label
        A 'str' containing the label of the intended placement

    Returns
    -------
    pandas.DataFrame
        A pd.DataFrame containing the start/end timestamp, duration of each
        detected walking or no walking bout considering intended placement
    """
    # TODO: simplify and break down the logic of this function

    # construct a time-series from the intervals
    t_start = movement_bouts.iloc[0, :].loc["start_time"]
    t_end = movement_bouts.iloc[-1, :].loc["end_time"]
    ts_index = pd.date_range(t_start, t_end, freq="20ms")

    # add detected placement as a column
    placement_bouts["detected_target_placement"] = False
    placement_bouts.loc[
        placement_bouts.loc[:, "placement"] == placement_label,
        "detected_target_placement",
    ] = True

    # remove turns smaller than a threshold to be considered turns
    turn_bouts = turn_bouts[
        turn_bouts["angle"].abs() > np.radians(WALKING_BOUT_TURN_MAX_ANGLE.value)
    ]
    # convert intervals into time-series for easier overlay
    walking_mask = get_mask_from_intervals(movement_bouts, ts_index, "detected_walking")
    placement_mask = get_mask_from_intervals(
        placement_bouts, ts_index, "detected_target_placement"
    )
    turn_mask = get_mask_from_intervals(turn_bouts, ts_index, "", "start", "end")
    # create no turn mask
    no_turn_mask = ~turn_mask

    # apply a logical AND to intersect walking, placement and no turns
    walking_placement_no_turn_mask = walking_mask & placement_mask & no_turn_mask
    # reset index
    walking_placement_no_turn_mask = walking_placement_no_turn_mask.reset_index()
    # rename columns
    walking_placement_no_turn_mask = walking_placement_no_turn_mask.rename(
        columns={"index": "ts", 0: "detected_walking"}
    )
    # convert timestamp to datetime format
    walking_placement_no_turn_mask["ts"] = pd.to_datetime(
        walking_placement_no_turn_mask["ts"]
    )
    # set everything below WALKING_BOUT_SKIP_FIRST_SECONDS seconds to False
    thres_first = walking_placement_no_turn_mask.loc[0, "ts"] + pd.Timedelta(
        WALKING_BOUT_SKIP_FIRST_SECONDS.value, "s"
    )
    walking_placement_no_turn_mask.loc[
        walking_placement_no_turn_mask.ts < thres_first, "detected_walking"
    ] = False

    # remove the fist 10 seconds
    # convert back to segments-based dataframe
    walking_placement_no_turn_bouts = format_walking_bouts(
        walking_placement_no_turn_mask
    )

    # filter only bouts of duration more than a threshold
    walking_placement_no_turn_bouts = remove_short_bouts(
        walking_placement_no_turn_bouts
    )

    # apply start and end time in case we "lost" the first or last bouts
    walking_placement_no_turn_bouts.iloc[
        0, walking_placement_no_turn_bouts.columns.get_indexer(["start_time"])
    ] = t_start
    walking_placement_no_turn_bouts.iloc[
        -1, walking_placement_no_turn_bouts.columns.get_indexer(["end_time"])
    ] = t_end

    # reformat bout to fill any gaps
    walking_placement_no_turn_bouts = format_walking_bouts(
        walking_placement_no_turn_bouts, "start_time", "end_time"
    )

    return walking_placement_no_turn_bouts


class FormatWalkingBouts(TransformStep):
    """Format fixed three seconds walking bouts into bouts of any size."""

    def __init__(self):
        super().__init__(
            data_set_ids="walking_fixed_windows",
            transform_function=format_walking_bouts,
            new_data_set_id="movement_bouts",
            definitions=[DEF_BOUT_START, DEF_BOUT_END, DEF_DETECTED, DEF_BOUT_ID],
        )


class DetectBouts(TransformStep):
    """Detect walking bouts."""

    definitions = [
        DEF_BOUT_START,
        DEF_BOUT_END,
        DEF_DETECTED,
        DEF_DURATION,
        DEF_BOUT_ID,
    ]


class DetectBoutsHarmonic(DetectBouts):
    """Detect walking bouts with CWT."""

    data_set_ids = "vertical_acceleration"
    new_data_set_id = "movement_bouts"

    @transformation
    def _detect_bouts(self, data):
        return format_walking_bouts(walking_detection_harmonic(data))


class DetectMovementBouts(DetectBouts):
    """Detect walking bouts based on dynamics algorithm."""

    data_set_ids = "acc_ts_rotated_resampled_detrend"
    new_data_set_id = "movement_bouts"

    # parameters
    rolling_window_size = WALKING_BOUT_DYNAMICS_ROLLING_WINDOW_SIZE
    rolling_step_size = WALKING_BOUT_DYNAMICS_ROLLING_STEP_SIZE
    thres_upright = WALKING_BOUT_DYNAMICS_THRES_UPRIGHT
    thres_moving = WALKING_BOUT_DYNAMICS_THRES_MOVING
    thres_min_bout = WALKING_BOUT_DYNAMICS_THRES_MIN_BOUT
    cutoff = WALKING_BOUT_DYNAMICS_CUTOFF

    @transformation
    def _detect_bouts(self, data):
        return movement_detection(data)


class DetectBoutsBeltPlacement(DetectBouts):
    """Detect walking bouts filtered for placement."""

    data_set_ids = ["movement_bouts", "placement_bouts"]
    new_data_set_id = "walking_placement_bouts"

    thres_min_bout = WALKING_BOUT_DYNAMICS_THRES_MIN_BOUT

    @transformation
    def _detect_bouts(self, walking_bouts, placement_bouts):
        return walking_placement_detection(walking_bouts, placement_bouts)


class DetectBoutsBeltPlacementNoTurns(DetectBouts):
    """Detect walking bouts filtered for placement and no turns."""

    data_set_ids = [
        "movement_bouts",
        "placement_bouts",
        "gyroscope_ts_rotated_resampled_butterworth_low_pass_filter_x_turns",
    ]
    new_data_set_id = "walking_placement_no_turn_bouts"

    # parameters
    turn_max_angle = WALKING_BOUT_TURN_MAX_ANGLE
    true_final_thres_min_bout = WALKING_BOUT_TRUE_FINAL_THRES_MIN_BOUT
    false_final_thres_min_bout = WALKING_BOUT_FALSE_FINAL_THRES_MIN_BOUT
    skip_first_seconds = WALKING_BOUT_SKIP_FIRST_SECONDS

    @transformation
    def _detect_bouts(
        self,
        walking_bouts,
        placement_bouts,
        gyroscope_ts_rotated_resampled_butterworth_low_pass_filter_x_turns,
    ):
        return walking_placement_no_turn_detection(
            walking_bouts,
            placement_bouts,
            gyroscope_ts_rotated_resampled_butterworth_low_pass_filter_x_turns,
        )


def find_peaks_in_local_slice(
    initial_peak_idx: np.ndarray,
    autocorr_values: np.ndarray,
    lag_values: np.ndarray,
    slice_size: int = REGULARITY_LOCAL_SEARCH_SLICE_SIZE.value,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find peaks in local slice.

    Parameters
    ----------
    initial_peak_idx
        A numpy.ndarray containing the initial guesses for the peak indices.
    autocorr_values
        The autocorrelation values.
    lag_values
        The value of the lags.
    slice_size
        The size of the slice to look for peaks forward and backward in number
        of samples.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
    A Tuple containing the refined peaks and lags.
    """
    autocorr_peaks = []
    lags_peaks = []
    for idx in initial_peak_idx:
        slice_start = max(0, idx - slice_size)
        slice_end = min(len(autocorr_values), idx + slice_size)
        slice_autocorr = autocorr_values[slice_start:slice_end]
        slice_lags = lag_values[slice_start:slice_end]

        # find the maximum value in the slice, its index and lag
        slice_peak = np.max(slice_autocorr)
        slice_idx = np.argmax(slice_autocorr)
        slice_lag = slice_lags[slice_idx]
        # append to lists
        autocorr_peaks.append(slice_peak)
        lags_peaks.append(slice_lag)

    # convert lists to nd array and return
    return np.array(autocorr_peaks), np.array(lags_peaks)


def compute_gait_regularity_single(
    data: pd.Series,
    find_peak_height_centile: float = REGULARITY_FIND_PEAK_HEIGHT_CENTILE.value,
    find_peak_distance: float = REGULARITY_FIND_PEAK_DISTANCE.value,
) -> Tuple[float, float, float, float]:
    """Compute gait regularity based on filtered signal.

    The step and stride regularity is defined as the first and second peak
    of the unbiased normalized autocorrelation. In order to minimize the
    negative effect of local maxima we filter the accelerometer signal and
    detect the lags based on that, which we then apply in the autocorrelation
    of the non-filtered signal.

    Parameters
    ----------
    data
        A pandas Series containing the accelerometer signal.
    find_peak_height_centile
        Required height of peak as a percentile of the autocorrelation signal.
    find_peak_distance
        Required minimal horizontal distance (>0) in seconds between
        neighbouring peaks.

    Returns
    -------
    Tuple[float, float, float, float]
        A Tuple containing the step and stride regularity and respective lags.

    References
    ----------
    .. [1] Angelini L et al. (2020) Is a Wearable Sensor-Based Characterisation
       of Gait Robust Enough to Overcome Differences Between Measurement
       Protocols? A Multi-Centric Pragmatic Study in Patients with MS.
       https://doi.org/10.3390/s20010079
       [2] Moe-Nilssen R et al. (2004) Estimation of gait cycle characteristics
       by trunk accelerometry
       https://doi.org/10.1016/s0021-9290(03)00233-1
    """
    # compute auto-correlation
    sampling_freq = 1e9 / data.index.freq.nanos
    data_filtered = butterworth_low_pass_filter(
        data,
        cutoff=REGULARITY_SIGNAL_FILTER_CUTOFF_FREQ.value,
        order=REGULARITY_SIGNAL_FILTER_ORDER.value,
        zero_phase=True,
    )
    # get original autocorrelation to apply detected lags to compute regularity
    lags_orig, autocorr_values_orig = scaled_autocorr(data.values)

    # get filtered autocorrelation to find peaks and lags
    _, autocorr_values_filt = scaled_autocorr(data_filtered.values)

    # get the middle of the autocorrelation values as the zero lag location
    pos_lag_idx = len(autocorr_values_orig) // 2

    # trim corr signal and its lags to search for peaks only in the first
    # `peak_interval` seconds (back and forth)
    peak_interval = REGULARITY_PEAK_SEARCH_INTERVAL_IN_SEC.value
    start = int(pos_lag_idx - peak_interval * sampling_freq + 1)
    end = int(pos_lag_idx + peak_interval * sampling_freq)
    autocorr_values_filt_trimmed = autocorr_values_filt[start:end]
    autocorr_values_orig_trimmed = autocorr_values_orig[start:end]
    lags_trimmed = lags_orig[start:end]

    # find peaks of the trimmed correlation signal that are at least 80%
    autocorr_peak_idx, _ = find_peaks(
        x=autocorr_values_filt_trimmed,
        height=np.percentile(autocorr_values_filt_trimmed, find_peak_height_centile),
        distance=find_peak_distance * sampling_freq,
    )

    # early return to prevent breaking later when fewer than 5 peaks detected
    # assuming a symmetric [-2, -1, 0, 1, 2] minimum formation
    if len(autocorr_peak_idx) < 5:
        return np.nan, np.nan, np.nan, np.nan

    # local slice search for refined peaks
    autocorr_peaks, lags_peaks = find_peaks_in_local_slice(
        autocorr_peak_idx, autocorr_values_orig_trimmed, lags_trimmed
    )

    # mirrored positive peaks only
    pos_lag_peak_idx = len(autocorr_peaks) // 2
    peaks_pos_lag = autocorr_peaks[pos_lag_peak_idx:]
    lags_pos_lag = lags_peaks[pos_lag_peak_idx:] / sampling_freq

    # compute zero phase, first phase (step) and second phase (stride)
    pos_lag_idx = len(autocorr_values_orig_trimmed) // 2
    a_no_phase = autocorr_values_orig_trimmed[pos_lag_idx]
    a_d1 = peaks_pos_lag[1]
    a_d2 = peaks_pos_lag[2]

    # compute step and sr regularity properties
    step_regularity = a_d1 / a_no_phase
    stride_regularity = a_d2 / a_no_phase

    return step_regularity, stride_regularity, lags_pos_lag[1], lags_pos_lag[2]


def compute_gait_regularity_multiple(
    acc: pd.DataFrame,
    step_dataset: pd.DataFrame,
    window_size_in_steps: int = REGULARITY_WINDOW_SIZE_IN_STEPS.value,
    window_step_in_steps: int = REGULARITY_WINDOW_STEP_IN_STEPS.value,
    component: str = "norm",
    bout_filter: bool = False,
) -> pd.DataFrame:
    """Compute the step and stride regularity.

    The step and stride regularity is defined as the first and second peak
    of the unbiased normalized autocorrelation.

    Parameters
    ----------
    acc
        A pandas DataFrame containing the at least one column with the intended
         acceleration signal component to be used for step & stride regularity.
    step_dataset
        A pandas series containing at the minimum the the times of the initial
        contacts ('IC' column).
    window_size_in_steps
        The interval expressed in number of succeeding steps to be used to
        trim the signal to compute the regularity.
    window_step_in_steps
        The step of the sliding window expressed in number of steps.
    component
        A str indicating the column name of acc to be used.
    bout_filter
        A boolean indicating whether to take into account bouts in the
        transformation of the dataset.

    Returns
    -------
    pandas.DataFrame
        The step and stride regularity values.

    """

    def _transform_gait_regularity(
        signal: pd.Series, start: pd.Timestamp, end: pd.Timestamp, bout_id: int
    ) -> pd.Series:
        """Wrap compute step and stride regularity single."""
        step_reg, stride_reg, lag_1, lag_2 = compute_gait_regularity_single(
            signal.loc[start:end]
        )

        series = pd.Series(
            {
                "index": start,
                "step_regularity": step_reg,
                "stride_regularity": stride_reg,
                "lag_step_regularity": lag_1,
                "lag_stride_regularity": lag_2,
            }
        )
        if bout_id is not None:
            series["bout_id"] = bout_id
        return series

    # get a copy to be able to manipulate as we wish
    step_ds = step_dataset.copy()
    # add next initial contact time based on number of steps
    step_ds["IC_next"] = step_ds["IC"].shift(-window_size_in_steps)
    # remove nat from start and end to prevent slicing with NaT
    step_ds = step_ds.dropna(subset=["IC", "IC_next"]).reset_index(drop=True)
    if bout_filter:
        # find bout transitions and remove steps that don't have enough
        # succeeding steps in each bout
        idx = step_ds.bout_id == step_ds.shift(-window_size_in_steps).bout_id
        step_ds = step_ds[idx].reset_index(drop=True)
        # use only every x steps based on sliding window step size
        step_ds["step_index"] = step_ds.groupby("bout_id").cumcount()
        idx = np.mod(step_ds["step_index"], window_step_in_steps) == 0
        step_ds = step_ds[idx].reset_index(drop=True)
    else:
        # use only every x steps based on sliding window step size
        step_ds["ones"] = 1
        step_ds["step_index"] = step_ds["ones"].cumsum() - 1
        idx = np.mod(step_ds["step_index"], window_step_in_steps) == 0
        step_ds = step_ds[idx].reset_index(drop=True)

    if len(step_ds) == 0:
        return pd.DataFrame(
            columns=[
                "step_regularity",
                "stride_regularity",
                "lag_step_regularity",
                "lag_stride_regularity",
                "bout_id",
            ]
        )
    return step_ds.apply(
        lambda x: _transform_gait_regularity(
            signal=acc[component],
            start=x["IC"],
            end=x["IC_next"],
            bout_id=x.bout_id if bout_filter else None,
        ),
        axis=1,
    ).set_index("index")


class _TransformGaitRegularityBase(TransformStep, metaclass=ABCMeta):
    # parameters
    window_size_in_steps = REGULARITY_WINDOW_SIZE_IN_STEPS
    window_step_in_steps = REGULARITY_WINDOW_STEP_IN_STEPS
    find_peak_distance = REGULARITY_FIND_PEAK_DISTANCE
    find_peak_height_centile = REGULARITY_FIND_PEAK_HEIGHT_CENTILE
    peak_search_interval_in_sec = REGULARITY_PEAK_SEARCH_INTERVAL_IN_SEC
    signal_filter_cutoff_freq = REGULARITY_SIGNAL_FILTER_CUTOFF_FREQ
    signal_filter_order = REGULARITY_SIGNAL_FILTER_ORDER
    local_search_slice_size = REGULARITY_LOCAL_SEARCH_SLICE_SIZE

    def __init__(self, component: str, *args, **kwargs):
        self.component = component
        super().__init__(*args, **kwargs)

    def get_new_data_set_id(self) -> str:
        """Overwrite new data set id."""
        return f"{self.data_set_ids[1]}_gait_regularity"  # type: ignore

    @abstractmethod
    def _get_bout_filter_flag(self) -> bool:
        raise NotImplementedError()

    @transformation
    def _gait_regularity(
        self, acc_signal: pd.DataFrame, step_dataset: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute the gait regularity."""
        return compute_gait_regularity_multiple(
            acc=acc_signal,
            step_dataset=step_dataset,
            component=self.component,
            bout_filter=self._get_bout_filter_flag(),
        )


class TransformGaitRegularity(_TransformGaitRegularityBase):
    """Transform gait regularity.

    Create a data set with columns ``step_regularity``, ``stride_regularity``,
    and ``bout_id`` indicating the step and stride regularity per step and
    corresponding bout identifier.
    """

    def _get_bout_filter_flag(self) -> bool:
        return True

    definitions = [
        DEF_STEP_REGULARITY,
        DEF_STRIDE_REGULARITY,
        DEF_LAG_STEP_REGULARITY,
        DEF_LAG_STRIDE_REGULARITY,
        DEF_BOUT_ID,
    ]


class TransformGaitRegularityWithoutBout(_TransformGaitRegularityBase):
    """Transform gait regularity without bout.

    Create a data set with two columns ``step_regularity`` and
    ``stride_regularity`` per step.
    """

    def _get_bout_filter_flag(self) -> bool:
        return False

    definitions = [
        DEF_STEP_REGULARITY,
        DEF_STRIDE_REGULARITY,
        DEF_LAG_STEP_REGULARITY,
        DEF_LAG_STRIDE_REGULARITY,
    ]


class ExtractStepRegularity(GaitBoutAggregateStep):
    """Extract step regularity aggregate measures.

    Parameters
    ----------
    data_set_ids
        The data set ids that will be considered to extract step regularity.
    """

    def __init__(self, data_set_ids: List[str], **kwargs):
        step_detector = get_step_detector_from_data_set_ids(data_set_ids, 1)
        description = (
            "The {aggregation} step regularity computed with the  "
            f"{step_detector} algorithm. It is computed with the bout "
            "strategy {bout_strategy_repr}."
        )

        definition = MeasureValueDefinitionPrototype(
            measure_name=AV("step regularity", "step_regularity"),
            data_type="float64",
            validator=BETWEEN_MINUS_ONE_AND_ONE,
            description=description,
        )

        super().__init__(
            data_set_ids, "step_regularity", DEFAULT_AGGREGATIONS, definition, **kwargs
        )


class ExtractStrideRegularity(GaitBoutAggregateStep):
    """Extract stride regularity aggregate measures.

    Parameters
    ----------
    data_set_ids
        The data set ids that will be considered to extract stride regularity.
    """

    def __init__(self, data_set_ids: List[str], **kwargs):
        step_detector = get_step_detector_from_data_set_ids(data_set_ids, 1)
        description = (
            "The {aggregation} stride regularity computed with the  "
            f"{step_detector} algorithm. It is computed with the bout "
            "strategy {bout_strategy_repr}."
        )

        definition = MeasureValueDefinitionPrototype(
            measure_name=AV("stride regularity", "stride_regularity"),
            data_type="float64",
            validator=BETWEEN_MINUS_ONE_AND_ONE,
            description=description,
        )

        super().__init__(
            data_set_ids,
            "stride_regularity",
            DEFAULT_AGGREGATIONS,
            definition,
            **kwargs,
        )
