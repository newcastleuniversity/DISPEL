# pylint: disable=duplicate-code
# pylint: disable=too-many-lines
"""Pinch test related functionality.

This module contains functionality to extract measures for the *Pinch* test
(PINCH).
"""
from abc import ABCMeta, abstractmethod
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from dispel.data.core import Reading
from dispel.data.flags import Flag
from dispel.data.levels import Level
from dispel.data.measures import MeasureValueDefinitionPrototype
from dispel.data.raw import DEFAULT_COLUMNS, USER_ACC_MAP, RawDataValueDefinition
from dispel.data.validators import GREATER_THAN_ZERO, RangeValidator
from dispel.data.values import AbbreviatedValue as AV
from dispel.data.values import ValueDefinition
from dispel.processing import ProcessingStep
from dispel.processing.data_set import transformation
from dispel.processing.extract import (
    BASIC_AGGREGATIONS,
    DEFAULT_AGGREGATIONS_CV,
    DEFAULT_AGGREGATIONS_Q95_CV,
    AggregateModalities,
    AggregateRawDataSetColumn,
    ExtractStep,
    agg_column,
)
from dispel.processing.level import LevelFilter, LevelIdFilter, ProcessingStepGroup
from dispel.processing.level_filters import AbsentDataSetFilter, NotEmptyDatasetFilter
from dispel.processing.modalities import (
    HandModality,
    HandModalityFilter,
    SensorModality,
)
from dispel.processing.transform import ConcatenateLevels, TransformStep
from dispel.processing.utils import parallel_explode
from dispel.providers.generic.activity.orientation import UpperLimbOrientationFlagger
from dispel.providers.generic.flags.ue_flags import OnlyOneHandPerformed
from dispel.providers.generic.sensor import (
    FREQ_20HZ,
    RenameColumns,
    Resample,
    SetTimestampIndex,
    TransformUserAcceleration,
)
from dispel.providers.generic.tasks.pinch.attempts import (
    PinchAttempt,
    PinchTarget,
    PinchTouch,
    double_touch_asynchrony,
    dwell_time,
    number_successful_pinches,
    pinching_duration,
    success_duration,
    total_duration,
    total_number_pinches,
)
from dispel.providers.generic.tasks.pinch.modalities import (
    AttemptOutcomeModality,
    AttemptSelectionModality,
    BubbleSizeModality,
    BubbleSizeModalityFilter,
    FingerModality,
)
from dispel.providers.generic.touch import Touch
from dispel.providers.generic.tremor import TremorMeasures
from dispel.signal.filter import butterworth_low_pass_filter
from dispel.stats.core import percentile_95, variation, variation_increase

TASK_NAME = AV("Pinch test", "PINCH")

#: A validator that ensures values are comprise between zero and forty seconds
ATTEMPT_DURATION_VALIDATOR = RangeValidator(lower_bound=0, upper_bound=40)

#: A validator that ensures values are comprise between zero and forty seconds
#: in milliseconds
ATTEMPT_DURATION_VALIDATOR_MS = RangeValidator(lower_bound=0, upper_bound=4e4)

_LEVELS = [
    f"{hand.abbr}-{size.variable}"
    for hand in HandModality
    for size in BubbleSizeModality
]

PINCH_BASIC_MODALITY_AGGREGATIONS: List[
    Tuple[Union[Callable[[Any], float], str], str]
] = [*BASIC_AGGREGATIONS, (variation, "coefficient of variation")]

PINCH_EXTENDED_MODALITY_AGGREGATIONS: List[
    Tuple[Union[Callable[[Any], float], str], str]
] = [*PINCH_BASIC_MODALITY_AGGREGATIONS, ("median", "median")]


class BubbleSizeTransformMixin(metaclass=ABCMeta):
    """A raw data set transformation processing step for Bubble.

    Parameters
    ----------
    size
        Bubble size modality.
    """

    def __init__(self, *args, **kwargs):
        self.size: BubbleSizeModality = kwargs.pop("size")
        super().__init__(*args, **kwargs)  # type: ignore

    def get_definition(self, **kwargs) -> ValueDefinition:
        """Get the definition."""
        kwargs["size"] = self.size.av
        return super().get_definition(**kwargs)  # type: ignore

    def get_level_filter(self) -> LevelFilter:
        """Get the level filter based on the bubble size."""
        return (
            LevelIdFilter(self.size.variable)
            & super().get_level_filter()  # type: ignore
        )


class HandModalityFilterMixin(metaclass=ABCMeta):
    """A raw data set transformation processing step for Bubble.

    Parameters
    ----------
    hand
        Handedness for tasks that are applied to different hands.
    """

    def __init__(self, *args, **kwargs):
        self.hand: HandModality = kwargs.pop("hand")
        super().__init__(*args, **kwargs)  # type: ignore

    def get_definition(self, **kwargs) -> ValueDefinition:
        """Get the definition."""
        kwargs["hand"] = self.hand.av
        return super().get_definition(**kwargs)  # type: ignore

    def get_level_filter(self) -> LevelFilter:
        """Get level filter."""
        return (
            HandModalityFilter(self.hand) & super().get_level_filter()  # type: ignore
        )


class PinchConcatenateTargets(TransformStep):
    """A pinch target level concatenation step."""

    data_set_ids = "screen"

    new_data_set_id = "melted_targets"

    definitions = [
        RawDataValueDefinition(
            "targets",
            "Concatenated pinch level data",
            description="Array of PinchTarget objects for level {level_id}",
        )
    ]

    @staticmethod
    def _segment_pinches(levels: Iterable[Level]) -> Dict[str, List[PinchTarget]]:
        """Segment data from multiple pinch level data."""
        targets: Dict[str, List[PinchTarget]] = {}
        for level in filter(lambda x: x.has_raw_data_set("screen"), levels):
            target = PinchTarget.from_level(level)
            if target is not None:
                targets.setdefault(target.parent_id, []).append(target)
        return targets

    @transformation
    def _segment_levels(self, _, level: Level, reading: Reading) -> pd.DataFrame:
        pinches = self._segment_pinches(reading.levels)
        return pd.DataFrame({"targets": pinches[str(level.id)]})


class PinchConcatenateHands(HandModalityFilterMixin, ConcatenateLevels):
    """A pinch hand level concatenation step.

    Parameters
    ----------
    hand
        The hand to which the levels are to be concatenated.
    data_set_id
        The data set id(s) that will be merged.
    """

    def __init__(self, hand: HandModality, data_set_id: Union[str, List[str]]):
        super().__init__(hand=hand, new_level_id=hand.abbr, data_set_id=data_set_id)


class PinchConcatenateBubbles(ConcatenateLevels):
    """A pinch bubble size level concatenation step for target data sets.

    Parameters
    ----------
    size
        The hand to which the levels are to be concatenated.
    """

    def __init__(self, size: BubbleSizeModality):
        level_filter = AbsentDataSetFilter("melted_targets") & BubbleSizeModalityFilter(
            size
        )

        super().__init__(
            new_level_id=size.variable,
            data_set_id="melted_targets",
            level_filter=level_filter,
        )


class PinchConcatenateTargetsLevels(ConcatenateLevels):
    """A level concatenation step."""

    def __init__(self):
        super().__init__(
            new_level_id="melted_levels",
            data_set_id="melted_targets",
            level_filter=AbsentDataSetFilter("melted_targets"),
        )


class TransformMeltedLevels(TransformStep):
    """Transformation step based on melted levels."""

    data_set_ids = "melted_targets"


class TransformApplyMeltedLevels(TransformMeltedLevels, metaclass=ABCMeta):
    """A Transformation step that applies a function on targets."""

    apply: Callable[..., Any]
    target_dtype = "float64"

    def get_apply_function(self) -> Any:
        """Get the function to be applied to the data set."""
        func = self.apply
        if func is not None and hasattr(func, "__func__"):
            return func.__func__  # type: ignore
        return func

    def post_process_apply(self, data: Any) -> Any:
        """Post process the data returned from the applied function."""
        return data

    @transformation
    def _apply(self, data: pd.DataFrame) -> Any:
        return self.post_process_apply(
            data["targets"].apply(self.get_apply_function())
        ).astype(self.target_dtype)


def reaction_time_extended(target: PinchTarget) -> Tuple:
    """Extract reaction time, hand, size and appearance of a target.

    The reaction time of the user between the bubble appearance and the first
    touch event.

    Parameters
    ----------
    target
        A pinch target object.

    Returns
    -------
    Tuple
        The user reaction time in ms, the hand, the size, and the appearance
        timestamp.
    """
    return (
        target.reaction_time.total_seconds() * 1e3,
        target.hand.name.lower(),
        target.size.name.lower(),
        target.appearance,
    )


def compute_reaction_time(data: pd.Series, reading: Reading) -> pd.Series:
    """Compute the reaction time for each target extended with hand and size.

    Parameters
    ----------
    data
        The series of targets.
    reading
        The associated reading

    Returns
    -------
    pd.Series
        A Series of the reaction time, for each targets.
    """
    # find the timestamp when the first bubble appeared for left hand
    if "left" not in set(reading.level_ids):
        ts_min_left = None
    else:
        ts_min_left = (
            reading.get_level("left")
            .get_raw_data_set("melted_targets")
            .data["targets"]
            .apply(lambda x: x.appearance)
            .min()
        )

    # find the timestamp when the first bubble appeared for right hand
    if "right" not in set(reading.level_ids):
        ts_min_right = None
    else:
        ts_min_right = (
            reading.get_level("right")
            .get_raw_data_set("melted_targets")
            .data["targets"]
            .apply(lambda x: x.appearance)
            .min()
        )

    # Create a dataframe with targets, hand, size, appearance
    df = pd.DataFrame({"targets": data["targets"]})
    df["reaction_time"], df["hand"], df["size"], df["appearance"] = zip(
        *df.targets.map(reaction_time_extended)
    )

    # Create a mask to remove the first reaction_time of each hand
    l_mask = df.appearance == ts_min_left
    r_mask = df.appearance == ts_min_right
    mask = ~(l_mask | r_mask)
    return df.loc[mask, "reaction_time"]


class TransformReactionTime(TransformMeltedLevels):
    """A raw data set transformation step to get user's reaction time."""

    new_data_set_id = "reaction-time"
    definitions = [
        RawDataValueDefinition(
            "reaction_time",
            "Reaction time data",
            "float64",
            description="The time spent between the appearance of the "
            "pinch target and actually touching the screen.",
        )
    ]

    @transformation
    def compute_reaction_time(self, data: pd.Series, reading: Reading) -> pd.Series:
        """Overwrite transformation."""
        return compute_reaction_time(data, reading)


class TransformTargetProperties(TransformApplyMeltedLevels):
    """A raw data set transformation processing step for target properties."""

    new_data_set_id = "target-properties"
    definitions = [
        RawDataValueDefinition(
            "total_pinches",
            "total pinches data",
            "int32",
            description="The total number of pinches, where a pinch "
            "attempt is any screen interaction with at least "
            "two fingers down.",
        ),
        RawDataValueDefinition(
            "successful_pinches",
            "successful pinches data",
            "int32",
            description="The number of successful pinches, where a "
            "successful pinch attempt is any screen "
            "interaction with at least two fingers down that"
            "eventually leads to the target bubble bursting.",
        ),
    ]
    target_dtype = "int32"

    def get_apply_function(self) -> Any:
        """Get the number of total and successful pinch attempts."""
        return lambda t: pd.Series(
            {
                "total_pinches": total_number_pinches(t),
                "successful_pinches": number_successful_pinches(t),
            }
        )


class TransformTotalDuration(TransformApplyMeltedLevels):
    """A raw data set transformation step to get pinch total duration."""

    new_data_set_id = "total-duration"
    definitions = [
        RawDataValueDefinition(
            "total_duration",
            "Total duration data",
            "float64",
            description="The total time spent during the pinching "
            "events of one pinch target.",
        )
    ]
    apply = total_duration


class TransformApplyMeasureMixin(metaclass=ABCMeta):
    """A raw data set transformation processing step for measures."""

    description: str = ""
    measure: str = ""

    @property
    def data_id(self):
        """Get data set id from measure name."""
        return self.measure.replace(" ", "_")

    @property
    def measure_id(self):
        """Get data measure id from measure name."""
        return self.measure.replace(" ", "-")

    def get_data_set_id(self) -> str:
        """Get data set id."""
        return self.data_id

    def get_column_id(self) -> str:
        """Get column id."""
        return self.measure_id

    def get_column_name(self) -> str:
        """Get column name."""
        return f"{self.measure} data"

    def get_new_data_set_id(self) -> str:
        """Get new data set id."""
        return self.data_id

    def get_definitions(self) -> List[RawDataValueDefinition]:
        """Get definition."""
        return [
            RawDataValueDefinition(
                self.get_column_id(),
                self.get_column_name(),
                "float64",
                description=self.description.format(**self.__dict__),
            )
        ]


class TransformReactionTimeByBubbleSize(
    BubbleSizeTransformMixin, TransformApplyMeasureMixin, TransformMeltedLevels
):
    """A transformation step to get user's reaction time by bubble size."""

    measure = "bubble reaction time"
    description = (
        "The time spent between the appearance of the "
        "pinch target and actually touching the screen"
        " for bubble size {size}."
    )

    @transformation
    def compute_reaction_time(self, data: pd.Series, reading: Reading) -> pd.Series:
        """Overwrite transformation."""
        return compute_reaction_time(data, reading)


class TransformParallelExplodeMeltedLevels(
    TransformApplyMeasureMixin, TransformApplyMeltedLevels, metaclass=ABCMeta
):
    """A Transformation step that applies function with a post process."""

    def __init__(
        self,
        finger: FingerModality,
        outcome: AttemptOutcomeModality = AttemptOutcomeModality.ALL,
    ):
        self.finger = finger
        self.outcome = outcome
        super().__init__()

    def get_new_data_set_id(self) -> str:
        """Get new data set id."""
        return f"{self.finger.abbr}_{self.outcome.abbr}_" f"{self.data_id}"

    def get_column_id(self) -> str:
        """Get column id."""
        return f"{self.finger.abbr}-{self.outcome.abbr}-{self.measure_id}"

    def get_column_name(self) -> str:
        """Get column name."""
        return f"{self.finger} {self.measure} data for {self.outcome} attempt"

    @abstractmethod
    def get_property(self, target: PinchTarget) -> Any:
        """Get property from an attempt."""
        raise NotImplementedError

    def get_apply_function(self) -> Any:
        """Get the pinching attempts' property."""
        return lambda target: pd.Series(
            {self.get_column_id(): self.get_property(target)}
        )

    def post_process_apply(self, data: Any) -> Any:
        """Parallel explode passed data."""
        return (
            parallel_explode(data, self.target_dtype)
            if not parallel_explode(data, self.target_dtype).empty
            else pd.Series([np.nan])
        )


class TransformContactDistance(TransformParallelExplodeMeltedLevels):
    """A raw data set transformation processing step for contact distance."""

    measure = "contact distance"
    description = (
        "The distance between the initial point of "
        "contact with the screen and the surface of "
        "the target bubble for the {finger} for "
        "{outcome} attempt."
    )

    def get_property(self, target: PinchTarget) -> List[float]:
        """Get property from a target."""
        return target.contact_distances(self.finger, self.outcome)


class TransformFirstPushes(TransformParallelExplodeMeltedLevels):
    """A raw data set transformation processing step first finger pushes."""

    measure = "first push"
    description = (
        "The value of the first push of "
        "pressure applied on the screen by the "
        "{finger} for {outcome} attempt."
    )

    def get_property(self, target: PinchTarget) -> List[float]:
        """Get property from a target."""
        return target.first_pushes(self.finger, self.outcome)


class TransformWrapExplodeMeltedLevels(TransformApplyMeltedLevels):
    """A Transformation step that applies a function on targets."""

    def post_process_apply(self, data: Any) -> Any:
        """Apply post-processing."""
        return (
            data.explode().dropna()
            if not data.explode().dropna().empty
            else pd.Series([np.nan])
        )


class TransformExplodeMeltedLevelsSuccess(
    TransformApplyMeasureMixin, TransformWrapExplodeMeltedLevels
):
    """A Transformation step that applies a function on targets."""

    def __init__(self, outcome: AttemptOutcomeModality = AttemptOutcomeModality.ALL):
        self.outcome = outcome
        super().__init__()

    def get_new_data_set_id(self) -> str:
        """Get new data set id."""
        return f"{self.outcome.abbr}_{self.data_id}"

    def get_column_id(self) -> str:
        """Get column id."""
        return f"{self.outcome.abbr}-{self.measure_id}"

    def get_column_name(self) -> str:
        """Get column name."""
        return f"{self.outcome} {self.measure}"


class TransformSuccessDeformingDuration(TransformExplodeMeltedLevelsSuccess):
    """A raw data set transformation step to get user's success duration."""

    measure = "success duration"
    description = (
        "Time spent before succeeding at deforming the"
        " target bubble during {outcome} pinch attempt."
    )

    def get_apply_function(self) -> Any:
        """Get the success duration from attempt."""
        return lambda t: success_duration(t, self.outcome)


class TransformPinchingDuration(TransformExplodeMeltedLevelsSuccess):
    """
    A raw data set transformation step to get user's pinching duration.

    To get the pinching duration of the user.
    """

    measure = "pinching duration"
    description = (
        "Time spent actually deforming the target "
        "bubble during {outcome} pinch attempt."
    )

    def get_apply_function(self) -> Any:
        """Get the pinching duration from attempt."""
        return lambda t: pinching_duration(t, self.outcome)


class TransformExplodeMeltedLevelsAttempt(
    TransformApplyMeasureMixin, TransformWrapExplodeMeltedLevels
):
    """A Transformation step that applies a function on targets."""

    def __init__(self, attempt: AttemptSelectionModality):
        self.attempt = attempt
        super().__init__()

    def get_new_data_set_id(self) -> str:
        """Get new data set id."""
        return f"{self.attempt.abbr}_{self.data_id}"

    def get_column_id(self) -> str:
        """Get column id."""
        return f"{self.attempt.abbr}-{self.measure_id}"

    def get_column_name(self) -> str:
        """Get column name."""
        return f"{self.attempt} {self.measure}"


class TransformDwellTime(TransformExplodeMeltedLevelsAttempt):
    """A raw data set transformation step to get user's dwell time."""

    measure = "dwell time"
    description = (
        "Time spent between the first screen touching "
        "and the initiation of the movement for "
        "{attempt} pinch attempts."
    )

    def get_apply_function(self) -> Any:
        """Get the pinching attempts' dwell times."""
        return partial(dwell_time, attempt=self.attempt)


class TransformDoubleTouchAsynchrony(TransformExplodeMeltedLevelsAttempt):
    """A transformation step to get user's double touch asynchrony."""

    measure = "double touch asynchrony"
    description = (
        "Time difference between the first and second"
        " finger touching the screen for {attempt} "
        "pinch attempts."
    )

    def get_apply_function(self) -> Any:
        """Get the pinching attempts' double touch asynchrony."""
        return partial(double_touch_asynchrony, attempt=self.attempt)


class TransformAttempt(
    TransformApplyMeasureMixin, TransformWrapExplodeMeltedLevels, metaclass=ABCMeta
):
    """A raw data set transformation processing step for finger.

    Parameters
    ----------
    finger
        The desired finger to compute ('top' or 'bottom').
    outcome
        Pinching attempt success modality.
    """

    def __init__(
        self,
        finger: FingerModality,
        outcome: AttemptOutcomeModality = AttemptOutcomeModality.ALL,
    ):
        self.finger = finger
        self.outcome = outcome
        super().__init__()

    def get_new_data_set_id(self) -> str:
        """Get new data set id."""
        return f"{self.finger.abbr}_{self.outcome.abbr}_" f"{self.data_id}"

    def get_column_id(self) -> str:
        """Get column id."""
        return f"{self.finger.abbr}-{self.outcome.abbr}-{self.measure_id}"

    def get_column_name(self) -> str:
        """Get column name."""
        return f"{self.finger} {self.measure} data for {self.outcome} attempt"

    def get_finger(self, attempt: PinchAttempt) -> PinchTouch:
        """Get finger from an attempt."""
        if self.finger == FingerModality.TOP_FINGER:
            return attempt.top_finger
        return attempt.bottom_finger

    @abstractmethod
    def get_property(self, attempt: Touch) -> Any:
        """Get property from an attempt."""
        raise NotImplementedError

    def _touch_function(self, target: PinchTarget) -> List[float]:
        """Get the measure for a specific finger for all attempt."""
        touch_method = [
            self.get_property(self.get_finger(attempt))
            for attempt in target.get_attempts_from(self.outcome)
            if self.get_finger(attempt) is not None
        ]
        if not touch_method or isinstance(touch_method[0], float):
            return touch_method

        return sum(touch_method, [])

    def get_apply_function(self) -> Any:
        """Get the pinching attempts' for the specific measure."""
        return self._touch_function


class TransformPressures(TransformAttempt):
    """A raw data set transformation processing step for finger pressures."""

    measure = "pressure"
    description = (
        "The pressure applied on the screen by the {finger} for {outcome} attempt."
    )

    def get_property(self, attempt: Touch) -> List[float]:
        """Get property from an attempt."""
        return attempt.pressure.tolist()


class TransformSpeed(TransformAttempt):
    """A raw data set transformation processing step to get finger speed."""

    measure = "speed"
    description = "The speed of the {finger} during {outcome} pinch attempts."

    def get_property(self, attempt: Touch) -> List[float]:
        """Get property from an attempt."""
        return attempt.speed.tolist()


class TransformJerk(TransformAttempt):
    """A raw data set transformation processing step to get jerk movements."""

    measure = "jerk"
    description = "The jerk of the {finger} during {outcome} pinch attempts."

    def get_property(self, attempt: Touch) -> List[float]:
        """Get property from an attempt."""
        return attempt.jerk.tolist()


class TransformMeanSquaredJerk(TransformAttempt):
    """A raw data set transformation processing step for mean squared jerk."""

    measure = "ms jerk"
    description = (
        "The mean squared jerk of the {finger} during {outcome} pinch attempts."
    )

    def get_property(self, attempt: Touch) -> List[float]:
        """Get property from an attempt."""
        return [attempt.mean_squared_jerk]


class TransformPressureJerk(TransformAttempt):
    """A raw data set transformation processing step to get jerk pressure."""

    measure = "pressure jerk"
    description = "The jerk pressure of the {finger} during {outcome} pinch attempts."

    def get_property(self, attempt: Touch) -> List[float]:
        """Get property from an attempt."""
        return attempt.pressure_jerk.tolist()


class TransformMeanSquaredPressureJerk(TransformAttempt):
    """A processing step to get mean squared jerk pressure."""

    measure = "ms pressure jerk"
    description = (
        "The mean squared jerk pressure of the {finger} "
        "during {outcome} pinch attempts."
    )

    def get_property(self, attempt: Touch) -> List[float]:
        """Get property from an attempt."""
        return [attempt.mean_squared_pressure_jerk]


class TargetPropertiesExtractStep(ExtractStep):
    """A base class for all extraction steps from target properties."""

    data_set_ids = "target-properties"


class ExtractTotalPinchAttempts(TargetPropertiesExtractStep):
    """Extract the total pinch attempts."""

    transform_function = agg_column("total_pinches", "sum")
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("total pinch attempts", "att"),
        description="The total number of pinch attempts on {size} size for "
        "the {hand} hand.",
        data_type="int32",
        validator=GREATER_THAN_ZERO,
    )


class ExtractSuccessfulPinchAttempts(TargetPropertiesExtractStep):
    """Extract successful pinch attempts."""

    transform_function = agg_column("successful_pinches", "sum")
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("successful pinch attempts", "succ"),
        description="The number successful pinch attempts on {size} size for "
        "the {hand} hand.",
        data_type="int32",
        validator=GREATER_THAN_ZERO,
    )


class ExtractPinchAccuracy(TargetPropertiesExtractStep):
    """A pinch accuracy extraction step."""

    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("pinching accuracy", "acc"),
        data_type="float64",
        validator=RangeValidator(0, 1),
        description="The ratio of successful pinches on {size} size "
        "amongst all pinch attempts for the {hand} hand.",
    )

    def flag_data_sets(
        self,
        data_sets: Sequence[pd.DataFrame],
        level: Level,
        reading: Reading,
        **kwargs,
    ) -> Generator[Flag, None, None]:
        """Flag that there is at least one pinch attempt."""
        super().flag_data_sets(data_sets, level, reading, **kwargs)
        if data_sets[0]["total_pinches"].sum() <= 0:
            yield Flag(
                id_="pinch-technical-deviation-ma",
                reason=f"Missing pinch attempts for {level}.",
            )

    @transformation
    def accuracy(self, data: pd.DataFrame) -> Union[float, np.float64]:
        """Extract pinch accuracy."""
        if (total_pinches := data["total_pinches"].sum()) == 0:
            return np.nan

        return np.float64(data["successful_pinches"].sum() / total_pinches)


class ExtractSuccessBase(TransformApplyMeasureMixin, AggregateRawDataSetColumn):
    """A measure extraction processing step for success modality.

    Parameters
    ----------
    outcome
        Pinching attempt success modality
    """

    def __init__(self, outcome: AttemptOutcomeModality = AttemptOutcomeModality.ALL):
        self.outcome = outcome
        super().__init__(
            data_set_id=self.get_data_set_id(), column_id=self.get_column_id()
        )

    def get_data_set_id(self) -> str:
        """Get data set id."""
        return f"{self.outcome.av.abbr}_{self.data_id}"

    def get_column_id(self) -> str:
        """Get column id."""
        return f"{self.outcome.av.abbr}-{self.measure_id}"

    def get_definition(self, **kwargs) -> ValueDefinition:
        """Get value definition."""
        kwargs["outcome"] = self.outcome.av
        return super().get_definition(**kwargs)


class ExtractSuccessDeformingDuration(ExtractSuccessBase):
    """A measure extraction processing step for multiple success durations."""

    measure = "success duration"
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("success duration", "sd"),
        data_type="float64",
        unit="s",
        validator=ATTEMPT_DURATION_VALIDATOR,
        description="Time {aggregation} spent before succeeding at "
        "deforming {size} size during "
        "{outcome} pinch attempt with the "
        "{hand} hand.",
    )
    aggregations = PINCH_BASIC_MODALITY_AGGREGATIONS


class ExtractPinchingDuration(ExtractSuccessBase):
    """A measure extraction processing step for multiple pinching durations."""

    measure = "pinching duration"
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("pinching duration", "pd"),
        data_type="float64",
        unit="s",
        validator=ATTEMPT_DURATION_VALIDATOR,
        description="Time {aggregation} spent actually deforming {size} size "
        "during {outcome} pinch attempt with the {hand} hand.",
    )
    aggregations = PINCH_BASIC_MODALITY_AGGREGATIONS


class ExtractDwellTime(AggregateRawDataSetColumn):
    """A measure extraction processing step for multiple dwell times."""

    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("dwell time", "dt"),
        data_type="float64",
        unit="ms",
        validator=GREATER_THAN_ZERO,
        description="The {aggregation} time spent between the first screen"
        " touching and the initiation of the movement "
        "for {attempt} attempts made of the bubble of size "
        "{size} with the {hand} hand.",
    )
    aggregations = DEFAULT_AGGREGATIONS_CV

    def __init__(self, attempt: AttemptSelectionModality):
        self.attempt = attempt
        super().__init__(
            data_set_id=f"{self.attempt.abbr}_dwell_time",
            column_id=f"{self.attempt.abbr}-dwell-time",
        )


class ExtractDoubleTouchAsynchrony(AggregateRawDataSetColumn):
    """An extraction step for multiple double touch asynchrony values."""

    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("double touch asynchrony", "dta"),
        data_type="float64",
        unit="ms",
        validator=GREATER_THAN_ZERO,
        description="The {aggregation} time difference between the first "
        "and second finger touching the screen for "
        " {attempt} attempts made of the bubble of size "
        "{size} with the {hand} hand.",
    )
    aggregations = DEFAULT_AGGREGATIONS_CV

    def __init__(self, attempt: AttemptSelectionModality):
        self.attempt = attempt
        super().__init__(
            data_set_id=f"{self.attempt.abbr}_double_touch_asynchrony",
            column_id=f"{self.attempt.abbr}-double-touch-asynchrony",
        )


class ExtractFingerSuccessBase(TransformApplyMeasureMixin, AggregateRawDataSetColumn):
    """A measure extraction processing step for finger and success modalities.

    Parameters
    ----------
    finger
        The desired finger to compute ('top' or 'bottom').
    outcome
        Pinching attempt success modality
    """

    def __init__(
        self,
        finger: FingerModality,
        outcome: AttemptOutcomeModality = AttemptOutcomeModality.ALL,
    ):
        self.outcome = outcome
        self.finger = finger
        super().__init__(
            data_set_id=self.get_data_set_id(), column_id=self.get_column_id()
        )

    def get_data_set_id(self) -> str:
        """Get data set id."""
        return f"{self.finger.av.abbr}_{self.outcome.av.abbr}_{self.data_id}"

    def get_column_id(self) -> str:
        """Get column id."""
        return f"{self.finger.av.abbr}-{self.outcome.av.abbr}" f"-{self.measure_id}"

    def get_definition(self, **kwargs) -> ValueDefinition:
        """Get value definition."""
        kwargs["finger"] = self.finger.av
        kwargs["outcome"] = self.outcome.av
        return super().get_definition(**kwargs)


class ExtractContactDistance(ExtractFingerSuccessBase):
    """A measure extraction processing step for multiple contact distances."""

    measure = "contact distance"
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("contact distance", "cd"),
        data_type="float64",
        unit="point",
        description="The {aggregation} distance between the initial "
        "point of contact on the screen with the {finger} "
        "of the {hand} hand and the surface of the target "
        "bubble for {size} size for the {outcome} attempt.",
    )
    aggregations = PINCH_EXTENDED_MODALITY_AGGREGATIONS


class ExtractFirstPushes(ExtractFingerSuccessBase):
    """A measure extraction processing step for first finger pushes."""

    measure = "first push"
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("first push", "fp"),
        data_type="float64",
        validator=GREATER_THAN_ZERO,
        description="The {aggregation} first pressure value applied on"
        " the screen for {outcome} attempt by the {finger} of "
        "the {hand} hand for {size} size.",
    )
    aggregations = [
        ("mean", "mean"),
        ("median", "median"),
        (variation, "coefficient of variation"),
    ]


class ExtractPressures(ExtractFingerSuccessBase):
    """A measure extraction processing step for multiple finger pressures."""

    measure = "pressure"
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("pressure", "press"),
        data_type="float64",
        validator=GREATER_THAN_ZERO,
        description="The {aggregation} pressure applied on the screen "
        "by the {finger} of the {hand} hand for {size} size for "
        "{outcome} attempt.",
    )
    aggregations = [
        ("mean", "mean"),
        ("median", "median"),
        ("std", "standard deviation"),
        ("skew", "skewness"),
        ("kurtosis", "kurtosis"),
        (variation, "coefficient of variation"),
    ]


class ExtractSpeed(ExtractFingerSuccessBase):
    """A measure extraction processing step for multiple finger speed."""

    measure = "speed"
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("speed", "speed"),
        data_type="float64",
        validator=GREATER_THAN_ZERO,
        description="The {aggregation} speed "
        "by the {finger} of the {hand} hand for {size} size for"
        " {outcome} attempts.",
    )
    aggregations = [
        ("mean", "mean"),
        ("median", "median"),
        ("std", "standard deviation"),
        (percentile_95, "95th percentile"),
    ]


class ExtractJerk(ExtractFingerSuccessBase):
    """A measure extraction processing step for multiple jerk finger."""

    measure = "jerk"
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("movement jerk", "jerk"),
        data_type="float64",
        description="The {aggregation} jerk "
        "by the {finger} of the {hand} hand for {size} size for "
        "{outcome}  attempts.",
    )
    aggregations = [
        ("mean", "mean"),
        ("median", "median"),
        ("std", "standard deviation"),
    ]


class ExtractMeanSquaredJerk(ExtractFingerSuccessBase):
    """A measure extraction for multiple mean squared jerk."""

    measure = "ms jerk"
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("mean squared jerk", "msj"),
        data_type="float64",
        unit="px^2/ms^6",
        validator=GREATER_THAN_ZERO,
        description="The {aggregation} mean squared jerk"
        " applied on the screen for "
        "{outcome} attempts by the"
        " {finger} of the {hand} hand for {size} size.",
    )
    aggregations = PINCH_EXTENDED_MODALITY_AGGREGATIONS


class ExtractPressureJerk(ExtractFingerSuccessBase):
    """A measure extraction processing step for multiple jerk pressures."""

    measure = "pressure jerk"
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("pressure jerk", "press_jerk"),
        data_type="float64",
        description="The {aggregation} jerk pressure applied on "
        "the screen for the {outcome} "
        "attempts by the {finger} "
        "of the {hand} hand for {size} size.",
    )
    aggregations = PINCH_EXTENDED_MODALITY_AGGREGATIONS


class ExtractMeanSquaredPressureJerk(ExtractFingerSuccessBase):
    """A measure extraction for multiple mean squared jerk of pressures."""

    measure = "ms pressure jerk"
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("mean squared pressure jerk", "mspj"),
        data_type="float64",
        unit="pa^2/ms^6",
        validator=GREATER_THAN_ZERO,
        description="The {aggregation} mean squared jerk pressure"
        " applied on the screen for the "
        "{outcome} attempt by the "
        "{finger} of the {hand} hand for {size} size.",
    )
    aggregations = [
        ("mean", "mean"),
        ("median", "median"),
        ("std", "standard deviation"),
    ]


class ExtractReactionTime(AggregateRawDataSetColumn):
    """A measure reaction time extraction processing step."""

    level_filter = LevelIdFilter("melted_levels")
    data_set_ids = "reaction-time"
    column_id = "reaction_time"
    definition = MeasureValueDefinitionPrototype(
        task_name=TASK_NAME,
        measure_name=AV("reaction time", "rt"),
        data_type="float64",
        unit="ms",
        validator=ATTEMPT_DURATION_VALIDATOR_MS,
        description="The {aggregation} time spent between the appearance "
        "of target bubbles and actually touching the screen.",
    )
    aggregations = [
        ("mean", "mean"),
        (variation_increase, "coefficient of variation increase"),
        ("median", "median"),
        ("std", "standard deviation"),
        ("min", "minimum"),
        ("max", "maximum"),
        (variation, "coefficient of variation"),
        (percentile_95, "95th percentile"),
    ]


class ExtractReactionTimeByBubbleSize(
    BubbleSizeTransformMixin, AggregateRawDataSetColumn
):
    """A measure reaction time by bubble size extraction processing step."""

    data_set_ids = "bubble_reaction_time"
    column_id = "bubble-reaction-time"
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("reaction time", "rt"),
        data_type="float64",
        unit="ms",
        validator=ATTEMPT_DURATION_VALIDATOR_MS,
        description="The {aggregation} time spent between the appearance "
        "of target bubble {size} size and actually touching the "
        "screen.",
    )
    aggregations = PINCH_BASIC_MODALITY_AGGREGATIONS


class ExtractReactionTimeByHand(HandModalityFilterMixin, AggregateRawDataSetColumn):
    """A pinch reaction time by hand extraction step."""

    data_set_ids = "reaction-time"
    column_id = "reaction_time"
    definition = MeasureValueDefinitionPrototype(
        task_name=TASK_NAME,
        measure_name=AV("reaction time", "rt"),
        data_type="float64",
        unit="ms",
        validator=ATTEMPT_DURATION_VALIDATOR_MS,
        description="The {aggregation} time spent between the appearance "
        "of target bubbles and actually touching the screen.",
    )
    aggregations = [
        ("mean", "mean"),
        (variation_increase, "coefficient of variation increase"),
        ("median", "median"),
        ("std", "standard deviation"),
        ("min", "minimum"),
        ("max", "maximum"),
        (variation, "coefficient of variation"),
        (percentile_95, "95th percentile"),
    ]

    def get_level_filter(self) -> LevelFilter:
        """Get level filter."""
        return LevelIdFilter(self.hand.variable)


class ExtractFirstReactionTimeByHand(HandModalityFilterMixin, ExtractStep):
    """A first pinch reaction time by hand extraction step."""

    data_set_ids = "melted_targets"
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("first pinch reaction time", "fprt"),
        data_type="float64",
        unit="ms",
        validator=ATTEMPT_DURATION_VALIDATOR_MS,
        description="The time spent between the appearance "
        "of the target and actually touching the "
        "screen for the {hand} hand.",
    )

    def get_level_filter(self) -> LevelFilter:
        """Get level filter."""
        return LevelIdFilter(self.hand.variable)

    @transformation
    def _first_pinch_reaction_time(self, data: pd.DataFrame) -> float:
        first_target: PinchTarget = min(data["targets"], key=lambda t: t.appearance)
        return first_target.reaction_time.total_seconds() * 1e3


class ExtractTotalDuration(AggregateRawDataSetColumn):
    """A measure total duration extraction processing step."""

    data_set_ids = "total-duration"
    column_id = "total_duration"
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("duration", "dur"),
        data_type="float64",
        unit="s",
        validator=ATTEMPT_DURATION_VALIDATOR,
        description="The {aggregation} time spent during pinching events "
        "of {size} size using the {hand} hand.",
    )
    aggregations = PINCH_BASIC_MODALITY_AGGREGATIONS


class PinchAggregateModalitiesByHand(AggregateModalities):
    """Base step to aggregate measures by hand for PINCH task.

    Parameters
    ----------
    hand
        The hand modality for which to aggregate measures.
    """

    def __init__(self, hand: HandModality):
        super().__init__()
        self.hand = hand

    def get_definition(self, **kwargs) -> ValueDefinition:
        """Get the definition."""
        new_kwargs = kwargs.copy()
        new_kwargs["modalities"] = [self.hand.av]
        new_kwargs["hand"] = self.hand.av
        return cast(MeasureValueDefinitionPrototype, self.definition).create_definition(
            **new_kwargs
        )

    def get_modalities(self) -> List[List[Union[str, AV]]]:
        """Get the modalities."""
        ids = []
        for size in BubbleSizeModality:
            ids.append([self.hand.av, size.av])

        return ids


class AggregateSuccessfulPinchesByHand(PinchAggregateModalitiesByHand):
    """Aggregate successful pinch attempts by hand."""

    definition = ExtractSuccessfulPinchAttempts.definition.derive(
        description="The number of successful pinch attempts for the {hand} hand."
    )

    @staticmethod
    def _agg_method(data):
        return None if len(data) == 0 else sum(data)

    aggregation_method = _agg_method


class AggregateDoubleTouchAsynchronyByHand(AggregateModalities):
    """Aggregate double touch asynchrony by hand."""

    def __init__(self, hand: HandModality, attempt: AttemptSelectionModality):
        super().__init__()
        self.hand = hand
        self.attempt = attempt

    def get_definition(self, **kwargs) -> ValueDefinition:
        """Get the definition."""
        new_kwargs = kwargs.copy()
        new_kwargs["modalities"] = [self.hand.av, self.attempt.av]
        new_kwargs["hand"] = self.hand.av
        new_kwargs["attempt"] = self.attempt.av
        return cast(MeasureValueDefinitionPrototype, self.definition).create_definition(
            **new_kwargs
        )

    def get_modalities(self) -> List[List[Union[str, AV]]]:
        """Get the modalities."""
        ids = []
        for size in BubbleSizeModality:
            ids.append([self.hand.av, size.av, self.attempt.av])

        return ids

    definition = ExtractDoubleTouchAsynchrony.definition.derive(
        aggregation="mean",
        description="The mean time difference between the first "
        "and second finger touching the screen for {attempt} "
        "pinch attempt made with the {hand} hand.",
    )


class AggregatePinchAccuracyByHand(PinchAggregateModalitiesByHand):
    """Aggregate pinch accuracy by hand."""

    definition = ExtractPinchAccuracy.definition.derive(
        description="The ratio of successful pinches amongst all pinch "
        "attempts for the {hand} hand."
    )


class AggregateTotalPinchAttemptsByHand(PinchAggregateModalitiesByHand):
    """Extract the total pinch attempts."""

    definition = ExtractTotalPinchAttempts.definition.derive(
        description="The total number of pinch attempts for the {hand} hand."
    )

    @staticmethod
    def _agg_method(data):
        return None if len(data) == 0 else sum(data)

    aggregation_method = _agg_method


class PinchProcessingLevel(ProcessingStepGroup):
    """A group of pinch processing steps for measures by level id.

    Parameters
    ----------
    level_id
        Level id.
    """

    def __init__(self, level_id: str):
        steps = [
            PinchConcatenateTargets(level_filter=LevelIdFilter(level_id)),
        ]

        super().__init__(steps)


class PinchProcessingLevelTarget(ProcessingStepGroup):
    """A group of pinch processing steps for measures by targets."""

    def __init__(self):
        steps = [PinchConcatenateTargetsLevels(), TransformReactionTime()]

        super().__init__(steps, task_name=TASK_NAME)


class PinchProcessingSize(ProcessingStepGroup):
    """A group of pinch processing steps for measures by bubble size."""

    def __init__(self, size: BubbleSizeModality):
        steps = [
            PinchConcatenateBubbles(size=size),
            TransformReactionTimeByBubbleSize(size=size),
            ExtractReactionTimeByBubbleSize(size=size),
        ]
        super().__init__(
            steps, task_name=TASK_NAME, modalities=[size.av], size=size.variable
        )


class PinchConcatenateHandsGroup(ProcessingStepGroup):
    """A group of pinch processing steps for measures by hands.

    Parameters
    ----------
    hand
        Handedness for tasks that are applied to different hands.
    """

    def __init__(self, hand: HandModality):
        steps: List[ProcessingStep] = [
            PinchConcatenateHands(
                hand,
                list(
                    map(
                        str,
                        [
                            SensorModality.ACCELEROMETER,
                            SensorModality.GYROSCOPE,
                            "melted_targets",
                        ],
                    )
                ),
            )
        ]

        super().__init__(steps, modalities=[hand.av], hand=hand, task_name=TASK_NAME)


class PinchReactionTimeByHand(ProcessingStepGroup):
    """A group of pinch processing steps for reaction time measures by hands.

    Parameters
    ----------
    hand
        Handedness for tasks that are applied to different hands.
    """

    def __init__(self, hand: HandModality):
        steps: List[ProcessingStep] = [
            ExtractFirstReactionTimeByHand(hand=hand),
            TransformReactionTime(level_filter=LevelIdFilter(hand.variable)),
            ExtractReactionTimeByHand(hand=hand),
        ]

        super().__init__(steps, modalities=[hand.av], hand=hand, task_name=TASK_NAME)


class PinchProcessingHandSensor(ProcessingStepGroup):
    """A group of pinch processing steps for tremor measures.

    Parameters
    ----------
    hand
        The hand on which the tremor measures are to be computed.
    sensor
        The sensor on which the tremor measures are to be computed.
    """

    def __init__(self, hand: HandModality, sensor: SensorModality):
        data_set_id = str(sensor)

        steps = [
            RenameColumns(data_set_id, hand.abbr, **USER_ACC_MAP),
            SetTimestampIndex(
                f"{data_set_id}_renamed", DEFAULT_COLUMNS, "ts", duplicates="last"
            ),
            Resample(
                f"{data_set_id}_renamed_ts",
                aggregations=["mean", "ffill"],
                columns=DEFAULT_COLUMNS,
                freq=FREQ_20HZ,
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
            level_filter=LevelIdFilter(hand.abbr) & NotEmptyDatasetFilter(data_set_id),
        )


class PinchProcessingHandSize(ProcessingStepGroup):
    """A group of pinch processing steps for measures by bubbles and hands.

    Parameters
    ----------
    hand
        Handedness for tasks that are applied to different hands.
    size
        Bubble size modality.
    """

    def __init__(self, hand: HandModality, size: BubbleSizeModality):
        steps = [
            TransformTargetProperties(),
            TransformTotalDuration(),
            TransformSuccessDeformingDuration(),
            TransformPinchingDuration(),
            ExtractSuccessDeformingDuration(),
            ExtractPinchingDuration(),
            ExtractTotalPinchAttempts(),
            ExtractSuccessfulPinchAttempts(),
            ExtractPinchAccuracy(),
            ExtractTotalDuration(),
        ]

        super().__init__(
            steps,
            task_name=TASK_NAME,
            modalities=[hand.av, size.av],
            hand=hand,
            size=size,
            level_filter=HandModalityFilter(hand) & BubbleSizeModalityFilter(size),
        )


class PinchProcessingHandSizeAttempt(ProcessingStepGroup):
    """A group of pinch processing steps by bubbles, hands and attempts.

    Parameters
    ----------
    hand
        Handedness for tasks that are applied to different hands.
    size
        Bubble size modality.
    attempt
        Pinching attempt selection modality.
    """

    def __init__(
        self,
        hand: HandModality,
        size: BubbleSizeModality,
        attempt: AttemptSelectionModality,
    ):
        steps = [
            TransformDwellTime(attempt),
            TransformDoubleTouchAsynchrony(attempt),
            ExtractDwellTime(attempt),
            ExtractDoubleTouchAsynchrony(attempt),
        ]

        super().__init__(
            steps,
            task_name=TASK_NAME,
            modalities=[hand.av, size.av, attempt.av],
            hand=hand,
            size=size,
            attempt=attempt,
            level_filter=HandModalityFilter(hand) & BubbleSizeModalityFilter(size),
        )


class PinchProcessingHandSizeFinger(ProcessingStepGroup):
    """A group of pinch processing steps by bubbles, hands and finger.

    Parameters
    ----------
    hand
        Handedness for tasks that are applied to different hands.
    size
        Bubble size modality.
    finger
        Pinching fingers modality.
    """

    def __init__(
        self, hand: HandModality, size: BubbleSizeModality, finger: FingerModality
    ):
        steps = [
            TransformFirstPushes(finger),
            TransformContactDistance(finger),
            TransformPressures(finger),
            TransformSpeed(finger),
            TransformJerk(finger),
            TransformPressureJerk(finger),
            TransformMeanSquaredJerk(finger),
            TransformMeanSquaredPressureJerk(finger),
            ExtractContactDistance(finger),
            ExtractFirstPushes(finger),
            ExtractPressures(finger),
            ExtractSpeed(finger),
            ExtractJerk(finger),
            ExtractPressureJerk(finger),
            ExtractMeanSquaredJerk(finger),
            ExtractMeanSquaredPressureJerk(finger),
        ]

        super().__init__(
            steps,
            task_name=TASK_NAME,
            modalities=[hand.av, size.av, finger.av],
            hand=hand,
            size=size,
            finger=finger,
            level_filter=HandModalityFilter(hand) & BubbleSizeModalityFilter(size),
        )


def duration_extended(target: PinchTarget) -> Tuple:
    """Extract several parameters from a pinch target.

    The parameters extracted are: duration, hand, size, appearance,  number of
    attempts and whether the first attempt was successful.

    Parameters
    ----------
    target
        A pinch target object.

    Returns
    -------
    Tuple
        The user duration of the first attempt time in seconds, the hand,
        the size, the appearance timestamp, the number of attempts for the
        target and if the first attempt was successful.
    """
    duration = None
    is_success = False
    if len(target.attempts) != 0:
        duration = target.attempts[0].duration.total_seconds()
        is_success = target.attempts[0].is_successful
    return (
        duration,
        target.hand.name.lower(),
        target.size.name.lower(),
        target.appearance,
        len(target.attempts),
        is_success,
    )


class TransformSingleShotDuration(TransformMeltedLevels):
    """A raw data set transformation step to get single shot duration."""

    new_data_set_id = "single-shot-duration"
    definitions = [
        RawDataValueDefinition(
            "targets", "targets", description="The targets objects."
        ),
        RawDataValueDefinition(
            "duration",
            "duration",
            data_type="float64",
            unit="ms",
            description="The {aggregation} time spent during pinching a bubble"
            "successfully at first attempt using the {hand} hand.",
        ),
        RawDataValueDefinition(
            "hand",
            "hand",
            data_type="str",
            description="The hand being used to pinch the bubble.",
        ),
        RawDataValueDefinition(
            "size",
            "size",
            data_type="str",
            description="The size of the bubble being pinched.",
        ),
        RawDataValueDefinition(
            "appearance", "appearance", description="The timestamp the bubble appeared."
        ),
        RawDataValueDefinition(
            "n_attempts",
            "number of attempts",
            description="The number of attempt to pinch the bubble.",
        ),
        RawDataValueDefinition(
            "is_successful",
            "is successful",
            description="A boolean indicating if the target is successful.",
        ),
    ]

    @transformation
    def single_shot(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a data set of single shot duration."""
        df = pd.DataFrame({"targets": data["targets"]})
        (
            df["duration"],
            df["hand"],
            df["size"],
            df["appearance"],
            df["n_attempts"],
            df["is_successful"],
        ) = zip(*df.targets.map(duration_extended))
        return df.loc[(df["n_attempts"] == 1) & df.is_successful]


class ExtractSingleShotDuration(HandModalityFilterMixin, AggregateRawDataSetColumn):
    """Aggregate single shot duration processing step."""

    data_set_ids = "single-shot-duration"
    column_id = "duration"
    aggregations = PINCH_BASIC_MODALITY_AGGREGATIONS
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("single_shot_dur", "dur1shot"),
        data_type="float64",
        unit="s",
        validator=ATTEMPT_DURATION_VALIDATOR,
        description="The {aggregation} time spent during pinching a bubble"
        "successfully at first attempt using the {hand} hand.",
    )

    def get_level_filter(self) -> LevelFilter:
        """Get level filter."""
        return LevelIdFilter(self.hand.variable)


class Pinch1ShotDurationByHand(ProcessingStepGroup):
    """A group of pinch processing steps for single shot duration by hands.

    Parameters
    ----------
    hand
        Handedness for tasks that are applied to different hands.
    """

    def __init__(self, hand: HandModality):
        steps = [
            TransformSingleShotDuration(level_filter=LevelIdFilter(hand.variable)),
            ExtractSingleShotDuration(hand=hand),
        ]

        super().__init__(steps, modalities=[hand.av], task_name=TASK_NAME, hand=hand)


def attempts_from_targets(target: PinchTarget) -> Tuple:
    """Extract several parameters from a pinch target.

    The parameters extracted are: hand, size, appearance and attempts.

    Parameters
    ----------
    target
        A pinch target object.

    Returns
    -------
    Tuple
        The hand, the size, the appearance timestamp, the attempts for the
        target.
    """
    return (
        target.hand.name.lower(),
        target.size.name.lower(),
        target.appearance,
        target.attempts,
    )


def peak_properties(attempt: PinchAttempt) -> Tuple:
    """Extract several peak properties from a pinch target."""
    # Top finger computation
    tf_data = attempt.top_finger.speed
    tf_data = tf_data.loc[tf_data.index.drop_duplicates(keep="last")]
    tf_data = tf_data.resample(pd.Timedelta(1 / 200, unit="s")).agg("ffill")
    # return NaN if we don't have enough non NaN values
    if len(tf_data.dropna()) <= 12:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    tf_data_f = butterworth_low_pass_filter(
        tf_data.dropna(), order=3, cutoff=10, zero_phase=True
    )

    # identify the peaks on the top finger trace
    tf_peaks_index, _ = find_peaks(tf_data_f)
    tf_peaks = tf_data_f.iloc[tf_peaks_index]

    # Bottom Finger Computation
    bf_data = attempt.bottom_finger.speed
    bf_data = bf_data.loc[bf_data.index.drop_duplicates(keep="last")]
    bf_data = bf_data.resample(pd.Timedelta(1 / 200, unit="s")).agg("ffill")
    # return NaN if we don't have enough non NaN values
    if len(bf_data.dropna()) <= 12:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    bf_data_f = butterworth_low_pass_filter(
        bf_data.dropna(), order=3, cutoff=10, zero_phase=True
    )
    # identify the peaks on the bottom finger trace
    bf_peaks_index, _ = find_peaks(bf_data_f)
    bf_peaks = bf_data_f.iloc[bf_peaks_index]

    return (
        tf_data.index[0],
        tf_peaks.index,
        tf_peaks.values,
        bf_data.index[0],
        bf_peaks.index,
        bf_peaks.values,
    )


class TransformAttemptPeakSpeed(TransformMeltedLevels):
    """Compute attempts peak speed properties for successful pinches."""

    new_data_set_id = "peak-speed-properties"
    definitions = [
        RawDataValueDefinition(
            "targets",
            "targets",
        ),
        RawDataValueDefinition(
            "attempts",
            "attempts",
        ),
        RawDataValueDefinition(
            "hand",
            "hand",
            data_type="str",
            description="The hand used to pinch the bubble.",
        ),
        RawDataValueDefinition(
            "size",
            "size",
            data_type="str",
            description="The size of the bubble pinched.",
        ),
        RawDataValueDefinition(
            "appearance",
            "appearance",
            description="The timestamp when the bubble appeared.",
        ),
        RawDataValueDefinition(
            "attempt_start",
            "attempt_start",
            description="The timestamp when the attempt starts.",
        ),
        RawDataValueDefinition(
            "tf_start",
            "tf_start",
            description="The timestamp when the top finger touch starts.",
        ),
        RawDataValueDefinition(
            "tf_peaks_index",
            "tf_peaks_index",
            description="The timestamps at which peaks in the top finger "
            "speed are detected.",
        ),
        RawDataValueDefinition(
            "tf_peaks_amp",
            "tf_peaks_amp",
            description="The speed amplitude of the top finger speed peaks.",
        ),
        RawDataValueDefinition(
            "bf_start",
            "bf_start",
            description="The timestamp when the bottom finger touch starts.",
        ),
        RawDataValueDefinition(
            "bf_peaks_index",
            "bf_peaks_index",
            description="The timestamps at which peaks in the bottom finger "
            "speed are detected.",
        ),
        RawDataValueDefinition(
            "bf_peaks_amp",
            "bf_peaks_amp",
            description="The speed amplitude of the bottom finger speed peaks.",
        ),
    ]

    @transformation
    def get_peak_speed_properties(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a data set of single shot duration."""
        df = pd.DataFrame({"targets": data["targets"]})
        df["hand"], df["size"], df["appearance"], df["attempts"] = zip(
            *df.targets.map(attempts_from_targets)
        )
        # Explode all the attempts
        df = df.explode("attempts").dropna()

        # Only keep successful attempts
        mask = df.attempts.apply(lambda x: x.is_successful)
        df = df.loc[mask]

        # add attempt start
        df["attempt_start"] = df.attempts.apply(lambda x: x.first_touch.begin)

        if len(df) == 0:
            df["tf_start"] = None
            df["tf_peaks_index"] = None
            df["tf_peaks_amp"] = None
            df["bf_start"] = None
            df["bf_peaks_index"] = None
            df["bf_peaks_amp"] = None
            return df

        # Compute peak properties
        (
            df["tf_start"],
            df["tf_peaks_index"],
            df["tf_peaks_amp"],
            df["bf_start"],
            df["bf_peaks_index"],
            df["bf_peaks_amp"],
        ) = zip(*df.attempts.map(peak_properties))
        return df


class TransformTimeToPeak(TransformStep):
    """Compute attempts time to peak speed properties."""

    data_set_ids = "peak-speed-properties"
    new_data_set_id = "time-to-peak-speed"
    definitions = [
        RawDataValueDefinition(
            "tf_peak_time",
            "tf_peak_time",
            description="The time to the first peak of speed amplitude of the "
            "top finger.",
        ),
        RawDataValueDefinition(
            "bf_peak_time",
            "bf_peak_time",
            description="The time to the first peak of speed amplitude of the "
            "bottom finger.",
        ),
        RawDataValueDefinition(
            "tf_n_peaks",
            "tf_n_peaks",
            description="The number of peak speed for the top finger.",
        ),
        RawDataValueDefinition(
            "bf_n_peaks",
            "bf_n_peaks",
            description="The number of peak speed for the bottom finger.",
        ),
        RawDataValueDefinition(
            "peak_asynchro",
            "peak_asynchro",
            description="The difference between timestamp the top finger "
            "reach its first peak speed and the timestamp the "
            "bottom finger reach its first peak speed.",
            unit="ms",
        ),
    ]

    @transformation
    def get_time_to_peak(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a data set of time to peak."""
        # Add number of peaks
        res = data.copy().dropna()
        if len(res) == 0:
            return pd.DataFrame(
                columns=[
                    "bf_n_peaks",
                    "bf_peak_time",
                    "tf_n_peaks",
                    "tf_peak_time",
                    "peak_asynchro",
                ]
            )
        res["bf_n_peaks"] = res.bf_peaks_index.map(len)
        res["tf_n_peaks"] = res.tf_peaks_index.map(len)
        mask = (res.bf_n_peaks > 0) & (res.tf_n_peaks > 0)

        # Select timestamp of the first peak speed for the bottom finger
        res["bf_peak_time"] = None
        res["bf_1_peak_ts"] = None
        res.loc[mask, "bf_1_peak_ts"] = res.loc[mask, "bf_peaks_index"].apply(
            lambda x: x[0]
        )

        # Compute time to first peak for the bottom finger
        res.loc[mask, "bf_peak_time"] = (
            pd.to_datetime(res.loc[mask, "bf_1_peak_ts"]) - res.loc[mask, "bf_start"]
        ).dt.total_seconds()

        # Select timestamp of the first peak speed for the top finger
        res["tf_peak_time"] = None
        res["tf_1_peak_ts"] = None
        res.loc[mask, "tf_1_peak_ts"] = res.loc[mask, "tf_peaks_index"].apply(
            lambda x: x[0]
        )

        # Compute time to first peak speed for the top finger
        res.loc[mask, "tf_peak_time"] = (
            pd.to_datetime(res.loc[mask, "tf_1_peak_ts"]) - res.loc[mask, "tf_start"]
        ).dt.total_seconds()

        # Compute asynchronicity
        res["peak_asynchro"] = (
            res["tf_1_peak_ts"] - res["bf_1_peak_ts"]
        ).dt.total_seconds() * 1000

        return res[
            [
                "bf_n_peaks",
                "bf_peak_time",
                "tf_n_peaks",
                "tf_peak_time",
                "peak_asynchro",
            ]
        ]


class ExtractTimeToPeakSpeedFinger(HandModalityFilterMixin, AggregateRawDataSetColumn):
    """Aggregate single time to peak processing step."""

    def __init__(self, hand: HandModality, finger: FingerModality):
        self.finger = finger
        self.column_id = f"{self.finger.abbr}_peak_time"
        super().__init__(hand=hand)

    data_set_ids = "time-to-peak-speed"
    aggregations = DEFAULT_AGGREGATIONS_Q95_CV
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("peak_time", "peaktime"),
        data_type="float64",
        unit="s",
        description="The {aggregation} time to the first peak of speed "
        "of the {hand} hand for the {finger} finger for "
        "successful attempts.",
    )

    def get_level_filter(self) -> LevelFilter:
        """Get level filter."""
        return LevelIdFilter(self.hand.variable)


class ExtractTimeToPeakAsynchronicity(
    HandModalityFilterMixin, AggregateRawDataSetColumn
):
    """Aggregate time to peak asynchronicity."""

    column_id = "peak_asynchro"
    data_set_ids = "time-to-peak-speed"
    aggregations = DEFAULT_AGGREGATIONS_Q95_CV
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("peak speed asynchronicity", "peak_speed_async"),
        data_type="float64",
        unit="ms",
        description="The {aggregation} duration between the bottom finger "
        "first peak speed and the top finger first peak speed for "
        "successful pinches attempts of the {hand} hand.",
    )

    def get_level_filter(self) -> LevelFilter:
        """Get level filter."""
        return LevelIdFilter(self.hand.variable)


class PinchTimeToPeakByHand(ProcessingStepGroup):
    """A group of pinch steps for time to peak speed computation.

    Parameters
    ----------
    hand
        Handedness for tasks that are applied to different hands.
    """

    def __init__(self, hand: HandModality):
        steps = [
            TransformAttemptPeakSpeed(level_filter=LevelIdFilter(hand.variable)),
            TransformTimeToPeak(
                level_filter=LevelIdFilter(hand.variable),
            ),
            ExtractTimeToPeakAsynchronicity(hand=hand),
        ]

        # pylint: disable=no-member
        super().__init__(
            steps,
            modalities=[hand.av, AttemptOutcomeModality.SUCCESS.av],
            task_name=TASK_NAME,
            hand=hand,
        )
        # pylint: enable=no-member


class ExtractPinchTimeToPeakByHandFinger(ProcessingStepGroup):
    """A group of pinch extract steps for time to peak speed computation.

    Parameters
    ----------
    hand
        Handedness for tasks that are applied to different hands.
    finger
        Pinching fingers modality.
    """

    def __init__(self, hand: HandModality, finger: FingerModality):
        steps = [ExtractTimeToPeakSpeedFinger(hand, finger)]
        outcome = AttemptOutcomeModality.SUCCESS

        # pylint: disable=no-member
        super().__init__(
            steps,
            modalities=[hand.av, finger.av, outcome.av],
            task_name=TASK_NAME,
            hand=hand,
            finger=finger,
        )
        # pylint: enable=no-member


class PinchProcessingHandSizeSuccess(ProcessingStepGroup):
    """A group of pinch processing steps by bubbles, hands and success.

    Parameters
    ----------
    hand
        Handedness for tasks that are applied to different hands.
    size
        Bubble size modality.
    outcome
        Pinching attempt success modality.
    """

    def __init__(
        self,
        hand: HandModality,
        size: BubbleSizeModality,
        outcome: AttemptOutcomeModality,
    ):
        steps = [
            TransformSuccessDeformingDuration(outcome),
            TransformPinchingDuration(outcome),
            ExtractSuccessDeformingDuration(outcome),
            ExtractPinchingDuration(outcome),
        ]

        super().__init__(
            steps,
            task_name=TASK_NAME,
            modalities=[hand.av, size.av, outcome.av],
            hand=hand.av,
            size=size.av,
            outcome=outcome.av,
            level_filter=HandModalityFilter(hand) & BubbleSizeModalityFilter(size),
        )


class PinchProcessingHandSizeFingerSuccess(ProcessingStepGroup):
    """A group of pinch processing steps by bubbles, hands, finger and success.

    Parameters
    ----------
    hand
        Handedness for tasks that are applied to different hands.
    size
        Bubble size modality.
    outcome
        Pinching attempt success modality.
    finger
        Pinching fingers modality.
    """

    def __init__(
        self,
        hand: HandModality,
        size: BubbleSizeModality,
        outcome: AttemptOutcomeModality,
        finger: FingerModality,
    ):
        steps = [
            TransformFirstPushes(finger, outcome),
            TransformContactDistance(finger, outcome),
            TransformPressures(finger, outcome),
            TransformSpeed(finger, outcome),
            TransformJerk(finger, outcome),
            TransformPressureJerk(finger, outcome),
            TransformMeanSquaredJerk(finger, outcome),
            TransformMeanSquaredPressureJerk(finger, outcome),
            ExtractContactDistance(finger, outcome),
            ExtractFirstPushes(finger, outcome),
            ExtractPressures(finger, outcome),
            ExtractSpeed(finger, outcome),
            ExtractJerk(finger, outcome),
            ExtractPressureJerk(finger, outcome),
            ExtractMeanSquaredJerk(finger, outcome),
            ExtractMeanSquaredPressureJerk(finger, outcome),
        ]

        super().__init__(
            steps,
            task_name=TASK_NAME,
            modalities=[hand.av, size.av, finger.av, outcome.av],
            hand=hand,
            size=size,
            outcome=outcome,
            finger=finger,
            level_filter=HandModalityFilter(hand) & BubbleSizeModalityFilter(size),
        )


class PinchProcessingAggregate(ProcessingStepGroup):
    """A group of pinch aggregating steps on hands.

    Parameters
    ----------
    hand
        Handedness for tasks that are applied to different hands.
    """

    def __init__(self, hand: HandModality):
        steps = [
            AggregateSuccessfulPinchesByHand(hand),
            AggregatePinchAccuracyByHand(hand),
            AggregateTotalPinchAttemptsByHand(hand),
            *(
                AggregateDoubleTouchAsynchronyByHand(hand, attempt)
                for attempt in AttemptSelectionModality
            ),
        ]

        super().__init__(steps, task_name=TASK_NAME)


class PinchFlag(ProcessingStepGroup):
    """Processing group for all drawing flag."""

    def __init__(self):
        steps = [
            TransformUserAcceleration(),
            UpperLimbOrientationFlagger(),
            OnlyOneHandPerformed(task_name=TASK_NAME),
        ]

        super().__init__(steps, task_name=TASK_NAME)


class PinchProcessingStepGroup(ProcessingStepGroup):
    """A group of all pinch processing steps for measures extraction."""

    def __init__(self):
        steps = [
            PinchFlag(),
            *(PinchProcessingLevel(level_id) for level_id in _LEVELS),
            PinchConcatenateTargetsLevels(),
            *(PinchConcatenateHandsGroup(hand) for hand in HandModality),
            *(PinchReactionTimeByHand(hand) for hand in HandModality),
            *(Pinch1ShotDurationByHand(hand) for hand in HandModality),
            *(PinchTimeToPeakByHand(hand) for hand in HandModality),
            *(
                ExtractPinchTimeToPeakByHandFinger(hand, finger)
                for hand in HandModality
                for finger in FingerModality
            ),
            *(PinchProcessingSize(size) for size in BubbleSizeModality),
            TransformReactionTime(level_filter=LevelIdFilter("melted_levels")),
            ExtractReactionTime(),
            *(
                PinchProcessingHandSensor(hand, sensor)
                for hand in HandModality
                for sensor in [SensorModality.ACCELEROMETER, SensorModality.GYROSCOPE]
            ),
            *(
                PinchProcessingHandSize(hand, size)
                for hand in HandModality
                for size in BubbleSizeModality
            ),
            *(
                PinchProcessingHandSizeAttempt(hand, size, attempt)
                for hand in HandModality
                for size in BubbleSizeModality
                for attempt in AttemptSelectionModality
            ),
            *(
                PinchProcessingHandSizeFinger(hand, size, finger)
                for hand in HandModality
                for size in BubbleSizeModality
                for finger in FingerModality
            ),
            *(
                PinchProcessingHandSizeSuccess(hand, size, outcome)
                for hand in HandModality
                for size in BubbleSizeModality
                for outcome in [
                    AttemptOutcomeModality.SUCCESS,
                    AttemptOutcomeModality.FAILURE,
                ]
            ),
            *(
                PinchProcessingHandSizeFingerSuccess(hand, size, outcome, finger)
                for hand in HandModality
                for size in BubbleSizeModality
                for outcome in [
                    AttemptOutcomeModality.SUCCESS,
                    AttemptOutcomeModality.FAILURE,
                ]
                for finger in FingerModality
            ),
            *(PinchProcessingAggregate(hand) for hand in HandModality),
        ]

        super().__init__(steps, task_name=TASK_NAME)


STEPS = [PinchProcessingStepGroup()]
