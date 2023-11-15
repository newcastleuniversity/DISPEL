# pylint: disable=duplicate-code
"""Drawing test related functionality.

This module contains functionality to extract features for the *Drawing* test
(DRAW).
"""
import warnings
from abc import ABCMeta
from functools import partial
from itertools import product
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from dispel.data.core import Reading
from dispel.data.features import FeatureValue, FeatureValueDefinitionPrototype
from dispel.data.flags import FlagSeverity, FlagType
from dispel.data.levels import Level
from dispel.data.raw import (
    DEFAULT_COLUMNS,
    PRESSURE_VALIDATOR,
    USER_ACC_MAP,
    RawDataSet,
    RawDataSetDefinition,
    RawDataSetSource,
    RawDataValueDefinition,
)
from dispel.data.validators import GREATER_THAN_ZERO, RangeValidator
from dispel.data.values import AbbreviatedValue as AV
from dispel.data.values import ValueDefinition
from dispel.io.raw import generate_raw_data_value_definition
from dispel.processing.core import ErrorHandling, ProcessingStep, ProcessResultType
from dispel.processing.data_set import RawDataSetProcessingResult, transformation
from dispel.processing.extract import (
    BASIC_AGGREGATIONS,
    DEFAULT_AGGREGATIONS,
    DEFAULT_AGGREGATIONS_Q95_CV,
    AggregateFeatures,
    AggregateModalities,
    AggregateRawDataSetColumn,
    ExtractMultipleStep,
    ExtractStep,
    FeatureDefinitionMixin,
)
from dispel.processing.flags import flag
from dispel.processing.level import (
    FlagLevelStep,
    LevelFilter,
    LevelIdFilter,
    LevelProcessingResult,
    LevelProcessingStep,
    LevelProcessingStepProtocol,
    ProcessingStepGroup,
)
from dispel.processing.level_filters import NotEmptyDatasetFilter
from dispel.processing.modalities import (
    HandModality,
    HandModalityFilter,
    SensorModality,
)
from dispel.processing.transform import ConcatenateLevels, TransformStep
from dispel.providers.generic.activity.orientation import UpperLimbOrientationFlagger
from dispel.providers.generic.flags.ue_flags import OnlyOneHandPerformed
from dispel.providers.generic.sensor import (
    FREQ_20HZ,
    FREQ_60HZ,
    RenameColumns,
    Resample,
    SetTimestampIndex,
    TransformUserAcceleration,
)
from dispel.providers.generic.tasks.draw.modalities import (
    AttemptModality,
    AttemptModalityFilter,
    ShapeModality,
    ShapeModalityFilter,
)
from dispel.providers.generic.tasks.draw.shapes import get_user_path, get_valid_path
from dispel.providers.generic.tasks.draw.touch import DrawShape, DrawTouch
from dispel.providers.generic.tremor import TremorFeatures
from dispel.signal.core import euclidean_norm, sparc
from dispel.stats.core import variation

TASK_NAME = AV("Drawing test", "DRAW")

# The definition of the DrawShape raw data set.
_SHAPE_DEFINITION = [generate_raw_data_value_definition("shape")]

#: The new names for reference and user trajectories (x and y coordinates).
_TRAJECTORY = ["x", "y"]

_UNITS = ["float64", "float64"]

_TRAJECTORY_DEFINITIONS = list(
    map(generate_raw_data_value_definition, _TRAJECTORY, _UNITS)
)

RADIUS = 8
r"""The radius of the circle used to consider if a point from the reference
path has been covered by the user trajectory. The value of 8 corresponds to
the size of a fingertip."""

SHAPE_TO_ABBR = {
    "square_clock": "sc",
    "square_counter_clock": "scc",
    "infinity": "inf",
    "spiral": "spi",
}

DRAWING_SIM_MEDIAN_Q95_MINUS_Q05 = {
    "inf": 8.096,
    "sc": 8.641,
    "scc": 8.952,
    "spi": 6.674,
}
r"""The distance between quantile 95 and quantile 5 of similarity median on
healthy population for the different shape."""

DRAWING_SIM_MEDIAN_MEAN = {"inf": 6.353, "sc": 5.753, "scc": 5.989, "spi": 5.821}
r"""Mean of similarity median on healthy population for the different shape."""

DRAWING_USER_DURATION_Q95_MINUS_Q05 = {
    "inf": 6012,
    "sc": 5699,
    "scc": 5738,
    "spi": 8671,
}
r"""The distance between quantile 95 and quantile 5 of user duration on
healthy population for the different shape."""

DRAWING_USER_DURATION_MEAN = {
    "inf": 4004,
    "sc": 3877,
    "scc": 3985,
    "spi": 5571,
}
r"""Mean of user duration on healthy population for the different shape."""


def get_user_duration(data: pd.DataFrame) -> float:
    """Compute the duration of the total interaction of the user with the test.

    Also compute the reaction time of the user between the beginning of the
    test and his first interaction.

    Parameters
    ----------
    data
        A pandas data frame composed of at least the user path and associated
        timestamps as index.

    Returns
    -------
    float
        The total duration (in milliseconds) of the user drawing action.
    """
    timestamps = data.index
    return (timestamps.max() - timestamps.min()).total_seconds() * 1e3


def get_instant_speed(data: pd.DataFrame) -> pd.DataFrame:
    """Compute the instantaneous speed of the drawing.

    Parameters
    ----------
    data
        A pandas data frame composed of at least the user path and associated
        timestamps as index.

    Returns
    -------
    numpy.float64
        The instantaneous speed of the drawing.
    """
    # Get rid of duplicated timestamps
    shape_data = data["shape"][0]
    data = getattr(shape_data, "valid_data")
    data = data[~data.index.duplicated(keep="last")]
    dist = euclidean_norm(data[["x", "y"]].diff()).astype(float)
    speed = dist / data.index.to_series().diff().dt.total_seconds()
    return pd.DataFrame(dict(distance=dist, speed=speed))


def get_speed_accuracy(data: pd.DataFrame, mean_dist: float) -> float:
    """Compute the speed accuracy of the user for a given level.

    Parameters
    ----------
    data
        A pandas data frame corresponding to the
        :class:`~dispel.providers.generic.tasks.draw.touch.DrawShape` data of the given
        level.
    mean_dist
        The mean dtw minimum distance for the given level.

    Returns
    -------
    float
        The speed accuracy for the given level (unit: point-1.ms-1).

    Raises
    ------
    AssertionError
        If ``speed * accuracy`` is equal to zero and ends up with a
        ZeroDivisionError for the ratio: ``1 / (speed * accuracy)``.
    """
    duration_params = get_user_duration(data)
    # explicit time depending on accuracy
    try:
        speed_accuracy = 1 / (duration_params * mean_dist)
    except ZeroDivisionError as exception:
        raise AssertionError(
            "``speed * accuracy`` cannot be equal to zero."
        ) from exception
    return speed_accuracy


def reaction_time(data: pd.DataFrame, level: Level) -> float:
    """Compute the reaction time.

    The reaction time of the user between the shape appearance and the first
    touch event.

    Parameters
    ----------
    data
        pandas data frame containing at least 'tsTouch' pd Series.
    level
        The level to be processed.

    Returns
    -------
    float
        the user's reaction time for the given level (in milliseconds).
    """
    first_touch = data.tsTouch.min()
    level_start = level.start
    return (first_touch - level_start).total_seconds() * 1e3


def wrap_reaction_time(data: pd.DataFrame, level: Level) -> pd.Series:
    """Wrap reaction time in a Series for a better aggregation.

    Parameters
    ----------
    data
        pandas data frame containing at least 'tsTouch' pd Series.
    level
        The level to be processed.

    Returns
    -------
    pandas.Series
        A pandas Series of the user's reaction time for the given level
        (in milliseconds).
    """
    return pd.Series({"reaction_time": reaction_time(data, level)})


class CreateShapes(LevelProcessingStep):
    """A LevelProcessingStep to create a ``DrawShape`` per level."""

    def process_level(
        self, level: Level, reading: Reading, **kwargs
    ) -> ProcessResultType:
        """Process the provided Level.

        Parameters
        ----------
        level
            The level to be processed
        reading
            The reading to be processed
        kwargs
            Additional arguments passed by :meth:`process_level`.

        Yields
        ------
        ProcessResultType
            Passes through anything that is yielded from the
            :meth:`process_level` function.
        """

        def _create_draw_shapes(
            level: Level, reading: Reading
        ) -> Tuple[pd.DataFrame, bool]:
            try:
                shape_data = DrawShape.from_level(level, reading)
            except AssertionError as assertion_error:
                warnings.warn(
                    f"When creating shape for level {str(level.id)} the"
                    f"post init raised an error: {assertion_error}",
                    Warning,
                )
                return pd.DataFrame(columns=["shape"]), False
            return pd.DataFrame({"shape": [shape_data]}), True

        data, error = _create_draw_shapes(level, reading)

        # second condition
        level.context.set(
            error,
            ValueDefinition(
                "is_creatable_shape",
                "If a shape does not raise an error when creating from level.",
                description="True if the shape is creatable False otherwise.",
            ),
        )
        raw_data_set = RawDataSet(
            definition=RawDataSetDefinition(
                id="shape",
                source=RawDataSetSource("ads"),
                value_definitions_list=_SHAPE_DEFINITION,
            ),
            data=data,
        )

        yield RawDataSetProcessingResult(
            step=self,
            sources=level.get_raw_data_set("screen"),
            level=level,
            result=raw_data_set,
        )


class ValidPathAssertionMixin(LevelProcessingStepProtocol, metaclass=ABCMeta):
    """Assertion mixin to ensure a valid path is present."""

    #: The error handling should no valid path be obtained
    missing_path_error_handling = ErrorHandling.IGNORE

    def assert_valid_level(self, level: Level, reading: Reading, **kwargs):
        """Assert that there are valid paths."""
        if not level.context.get_raw_value("is_valid_path"):
            raise AssertionError("Invalid user path", self.missing_path_error_handling)


class CreatableShape(LevelFilter):
    """A level filter to fetch level with creatable shapes only."""

    def repr(self) -> str:
        """Get representation of the filter."""
        return "Creatable shapes"

    def filter(self, levels: Iterable[Level]) -> Union[Set, Set[Level]]:
        """Keep level with a creatable shape from level."""
        out = set()
        for level in levels:
            if (
                "is_creatable_shape" in level.context
                and level.context.get_raw_value("is_creatable_shape") is True
            ):
                out.add(level)
        return out


def _flag_level_is_continuous(level: Level):
    """Return False if the level include a non-continuous shape."""
    if not level.has_raw_data_set("screen"):
        return True
    screen = level.get_raw_data_set("screen").data
    # Flag there is not several touchPathId
    if "inEndZone" not in screen.columns:
        return len(screen["touchPathId"].unique()) == 1
    # Flag there is not several down touchAction
    condition_down = (screen["touchAction"] == "down").sum() == 1
    # Flag there is not several up touchAction
    condition_up = (screen["touchAction"] == "up").sum() <= 1
    return condition_down & condition_up


class ContinuousLevel(LevelFilter):
    """Filter for continuous drawing shape."""

    def repr(self):
        """Get representation of the filter."""
        return "only continuously drawn shapes"

    def filter(self, levels: Iterable[Level]) -> Set[Level]:
        """Filter levels with continuous drawn shapes."""
        return set(filter(_flag_level_is_continuous, levels))


class TransformValidUserPath(FlagLevelStep):
    """A Transform step to determine if the user path is valid."""

    level_filter = CreatableShape() & ContinuousLevel()

    @staticmethod
    def is_valid_level(level: Level, reading: Reading, **_kwargs):
        """Assert that there are valid paths."""
        data = level.get_raw_data_set("screen").data
        if not level.has_raw_data_set("shape"):
            return False
        shape = level.get_raw_data_set("shape").data["shape"]
        if len(shape) == 0:
            return False
        shape = shape[0]
        ref = shape.get_reference
        if reading.device is None:
            return False
        if reading.device.screen is None:
            return False
        height = reading.device.screen.height_dp_pt
        if height is None:
            return False
        paths = get_user_path(data, ref, height)
        valid_paths = get_valid_path(paths)
        # first condition to be valid
        condition_1 = (
            valid_paths[["x", "y", "touchAction", "isValidArea"]].dropna().size > 0
        )
        if not condition_1:
            result = False
        else:
            result = len(shape.up_sampled_data_without_overshoot) > 0
        # second condition
        level.context.set(
            result,
            ValueDefinition(
                "is_valid_shape",
                "Contains a valid shape",
                description="True if the shape is not empty after "
                "applying pre-processing removing overshoots.",
            ),
        )
        return result

    task_name = TASK_NAME
    flag_name = AV("valid user path", "val_user_path")
    flag_type = FlagType.TECHNICAL
    flag_severity = FlagSeverity.DEVIATION
    reason = "The user path after pre-processing is empty. "
    flagging_function = is_valid_level


class ValidUserPath(LevelFilter):
    """A level filter to fetch level with valid user path only."""

    def repr(self) -> str:
        """Get representation of the filter."""
        return "valid user path"

    def filter(self, levels: Iterable[Level]) -> Union[Set, Set[Level]]:
        """Keep level with a valid user path."""
        out = set()
        for level in levels:
            if "is_valid_shape" in level.context and level.context.get_raw_value(
                "is_valid_shape"
            ):
                out.add(level)
        return out


class FlagContinuousDrawing(FlagLevelStep):
    """Flag the user do not lift the finger while drawing."""

    task_name = TASK_NAME
    flag_name = AV("continous drawing", "continuous_drawing")
    flag_type = FlagType.BEHAVIORAL
    flag_severity = FlagSeverity.DEVIATION
    reason = (
        "The drawing is not continuous, the user has lifted the finger"
        "during level {level_id}."
    )

    @flag
    def _check_continuous_drawing(self, level: Level, **kwargs) -> bool:
        self.set_flag_kwargs(level_id=level.id, **kwargs)
        return _flag_level_is_continuous(level)


class InfinityShapes(LevelFilter):
    """A level filter to fetch level from infinity shapes."""

    def repr(self) -> str:
        """Get representation of the filter."""
        return "infinity shapes"

    def filter(self, levels: Iterable[Level]) -> Union[Set, Set[Level]]:
        """Get infinity shapes."""
        out = set()
        for level in levels:
            if "inf" in str(level.id):
                out.add(level)
        return out


class CreatableShapeFlag(FlagLevelStep):
    """A Transform step to determine if the shape is creatable in a level."""

    @staticmethod
    def shape_is_creatable(level: Level, **_kwargs):
        """Assert that the shape can be created from the level."""
        if "is_creatable_shape" in level.context:
            return level.context.get_raw_value("is_creatable_shape")
        return False

    task_name = TASK_NAME
    flag_name = AV("draw creatable shape", "draw_creatable_shape")
    flag_type = FlagType.TECHNICAL
    flag_severity = FlagSeverity.DEVIATION
    reason = "The shape was impossible to create from the level. "
    flagging_function = shape_is_creatable


class TransformDecelerationProfile(TransformStep):
    """A raw data transformation step to get the user deceleration profile."""

    data_set_ids = "shape"
    new_data_set_id = "deceleration"
    definitions = [
        RawDataValueDefinition(
            "tsTouch", "timestamp of interactions data", "datetime64"
        ),
        RawDataValueDefinition("min_distance", "DTW distance data", "float64"),
        RawDataValueDefinition("x", "x user trajectory", "float64"),
        RawDataValueDefinition("y", "y user trajectory", "float64"),
    ]

    @transformation
    def retrieve_deceleration_from_shape(self, data: pd.DataFrame):
        """Get the deceleration data."""
        return data["shape"][0].deceleration_data

    def assert_valid_level(
        self, level: Level, reading: Reading, **kwargs
    ):  # type: ignore
        """Flag the presence of a valid user path."""
        super().assert_valid_level(level, reading, **kwargs)  # type: ignore
        draw = level.get_raw_data_set("shape").data["shape"][0]
        data = draw.aggregate_valid_touches.positions
        if data.dropna().empty:
            raise AssertionError(
                f"No user path found for level {level.id}", ErrorHandling.RAISE
            )


class TransformReactionTime(TransformStep, ValidPathAssertionMixin):
    """A raw data set transformation step to get user's reaction time."""

    data_set_ids = "screen"
    new_data_set_id = "reaction-time"
    transform_function = wrap_reaction_time
    definitions = [
        RawDataValueDefinition("reaction_time", "Reaction time data", "float64")
    ]


class TransformInstantSpeed(TransformStep, ValidPathAssertionMixin):
    """A raw data set transformation step to get user's instantaneous speed."""

    data_set_ids = "shape"
    new_data_set_id = "instantaneous_speed"
    transform_function = get_instant_speed
    definitions = [
        RawDataValueDefinition("speed", "speed", "float64"),
        RawDataValueDefinition("distance", "distance", "float64"),
    ]


class AggregateInstantSpeed(AggregateRawDataSetColumn):
    """Extract instant speed features."""

    data_set_ids = "instantaneous_speed"

    def __init__(self) -> None:
        definition = FeatureValueDefinitionPrototype(
            feature_name=AV("instant speed", "speed"),
            data_type="float64",
            unit="point.s-1",
            description="The {aggregation} of the instantaneous speed while "
            "drawing.",
        )

        super().__init__(
            data_set_id="instantaneous_speed",
            column_id="speed",
            aggregations=DEFAULT_AGGREGATIONS_Q95_CV,
            definition=definition,
        )


class ExtractShapeMixIn(LevelProcessingStep, FeatureDefinitionMixin, metaclass=ABCMeta):
    """A Transformation step that applies a function on targets."""

    data_set_ids = "shape"
    properties: Union[str, Sequence[str]]
    extract: Callable[..., Any]
    target_dtype = "float64"

    def get_properties(self, shape: DrawShape) -> Tuple[Any, ...]:
        """Get property from an attempt."""
        properties = self.properties
        assert properties is not None, "No properties are given."
        if isinstance(properties, str):
            properties = [properties]
        return tuple(map(partial(getattr, shape), properties))

    def get_extract_function(self):
        """Get the function to be applied to the data set."""
        func = self.extract
        if func is not None and hasattr(func, "__func__"):
            return func.__func__  # type: ignore
        return func

    def _extract(self, data: pd.DataFrame) -> Any:
        properties = self.get_properties(data.iloc[0, 0])

        return np.array([self.get_extract_function()(*properties)]).astype(
            self.target_dtype
        )[0]

    def process_level(self, level, reading, **kwargs):
        """Overwrite process level."""
        kwargs_extended = kwargs.copy()
        kwargs_extended["shape"] = str(level.id)
        raw_data_set = level.get_raw_data_set("shape")
        res = self._extract(raw_data_set.data)
        yield LevelProcessingResult(
            step=self,
            sources=raw_data_set,
            level=level,
            result=FeatureValue(self.get_definition(**kwargs_extended), res),
        )


class ExtractDuration(ExtractShapeMixIn, ValidPathAssertionMixin):
    """Extract total duration of drawing."""

    properties = "all_data"
    extract = get_user_duration
    definition = FeatureValueDefinitionPrototype(
        feature_name=AV("drawing duration", "user_dur"),
        data_type="float64",
        unit="ms",
        validator=GREATER_THAN_ZERO,
        description="The time spend between the first and last interaction"
        "of the subject with the screen while drawing {shape}"
        "with their {hand} hand for the {attempt} attempt.",
    )


class ExtractIntersections(ExtractShapeMixIn, ValidPathAssertionMixin):
    """Extract total number of intersections between the user and the model."""

    properties = "intersection_features"
    extract = len
    definition = FeatureValueDefinitionPrototype(
        feature_name=AV("intersections", "cross"),
        data_type="int32",
        validator=GREATER_THAN_ZERO,
        description="The number of times the user cross the shape line "
        "with his finger while drawing {shape} shape"
        "with their {hand} hand for the {attempt} attempt.",
    )


class ExtractIntersectionsPerSeconds(ExtractShapeMixIn, ValidPathAssertionMixin):
    """Extract the mean number of intersections per second."""

    @staticmethod
    def get_cross_per_sec(data) -> np.float64:
        """Get the number of crossings per second."""
        return data["cross_per_sec"][0]

    properties = "intersection_features"
    extract = get_cross_per_sec
    definition = FeatureValueDefinitionPrototype(
        feature_name=AV("intersections per second", "cross_per_sec"),
        data_type="float64",
        validator=GREATER_THAN_ZERO,
        description="The mean number of intersection per second the user"
        " performs with his finger while drawing {shape} shape "
        "with their {hand} hand for the {attempt} attempt.",
    )


class ExtractIntersectionsFrequency(ExtractMultipleStep, ValidPathAssertionMixin):
    """Extract total number of intersections between the user and the model."""

    def __init__(self) -> None:
        def _intersection_frequency_factory(agg: str, agg_label: str) -> Dict[str, Any]:
            return dict(
                func=lambda data: (
                    1 / data["shape"][0].intersection_features["tsDiff"]
                ).agg(agg),
                aggregation=AV(agg_label, agg),
                unit="Hz",
            )

        def _cv_intersection_frequency_factory() -> Dict[str, Any]:
            return dict(
                func=lambda data: variation(
                    data["shape"][0].intersection_features["freqDiff"]
                ),
                aggregation=AV("coefficient of variation", "cv"),
                unit="Hz",
            )

        data_set = "shape"
        function = [
            _intersection_frequency_factory(agg, agg_label)
            for agg, agg_label in BASIC_AGGREGATIONS
        ]
        function += [_cv_intersection_frequency_factory()]
        definition = FeatureValueDefinitionPrototype(
            feature_name=AV("intersections frequency", "cross_freq"),
            data_type="float64",
            validator=GREATER_THAN_ZERO,
            description="The {aggregation} intersection frequency of the user"
            " while drawing {shape} shape with their {hand} hand "
            "for the {attempt} attempt.",
        )
        super().__init__(data_set, function, definition)


class ExtractSparc(ExtractShapeMixIn, ValidPathAssertionMixin):
    """Extract spectral arc length of the up sampled user draw.

    The spectral arc length is a smoothness measure method. For more
    information about the ``sparc`` function, see
    :func:`~dispel.signal.core.sparc`.
    """

    @staticmethod
    def sparc_call(touch: DrawTouch) -> float:
        """Extract sparc."""
        data = touch.valid_up_sampled_path
        sal, *_ = sparc(data["x"].to_numpy())
        return sal

    properties = "aggregate_valid_touches"
    extract = sparc_call
    definition = FeatureValueDefinitionPrototype(
        feature_name=AV("smoothness score", "smooth"),
        data_type="float64",
        validator=RangeValidator(upper_bound=0),
        description="A digital score of tremor using spectral arc length "
        "(SPARC) measurement algorithm for {shape} shape drawn "
        "with their {hand} hand for the {attempt} attempt.",
    )


def _extract_speed_acc(
    path: pd.DataFrame,
    matches: pd.DataFrame,
    agg: str,
) -> np.float64:
    """Extract speed accuracy."""
    return get_speed_accuracy(path, float(_dtw_agg_dist(agg, matches)))  # type: ignore


class ExtractSpeedAccuracy(ExtractShapeMixIn, ValidPathAssertionMixin):
    """Extract speed accuracy."""

    properties = ["valid_data", "up_sampled_valid_no_overshoot_matches"]

    def __init__(self, aggregation: str, **kwargs):
        self.extract = lambda x, y: _extract_speed_acc(  # type: ignore
            x, y, aggregation
        )
        self.definition = FeatureValueDefinitionPrototype(
            feature_name=AV("accuracy-normalized duration", "dur_acc"),
            data_type="float64",
            unit="point-1.ms-1",
            aggregation=aggregation,
            validator=GREATER_THAN_ZERO,
            description="The accuracy of the subject while drawing {shape} "
            "with their {hand} hand for the {attempt} attempt "
            "normalized by the time spend between the first and "
            "last interaction of the subject with the screen. "
            f"Accuracy is one over the {aggregation} similarity "
            "between reference shape and drawn shape measured "
            "using dynamic time warping. See "
            ":func:`~dispel.signal.dtw.get_dtw_distance`.",
        )
        super().__init__(**kwargs)


def _extract_dur_acc_normed_combined(
    data: pd.DataFrame, matches: pd.DataFrame, shape: str
) -> float:
    """Extract duration accuracy normed and combined."""
    shape_abbr = SHAPE_TO_ABBR[shape.split("-")[0]]
    duration_params = get_user_duration(data)
    normed_score_duration = (
        duration_params / DRAWING_USER_DURATION_Q95_MINUS_Q05[shape_abbr]
    )
    similarity_median = float(_dtw_agg_dist("median", matches))
    normed_sim_median = similarity_median / DRAWING_SIM_MEDIAN_Q95_MINUS_Q05[shape_abbr]
    return normed_score_duration + normed_sim_median


class ExtractDurationAccuracyNormedCombined(ExtractShapeMixIn, ValidPathAssertionMixin):
    """Extract duration accuracy normalized and combined score."""

    properties = ["valid_data", "up_sampled_valid_no_overshoot_matches"]

    @transformation
    def _extract(self, data: pd.DataFrame) -> Any:
        def _extract_func(x, y):
            return _extract_dur_acc_normed_combined(x, y, data.iloc[0, 0].id)

        properties = self.get_properties(data.iloc[0, 0])
        return np.array([_extract_func(*properties)]).astype(self.target_dtype)[0]

    def __init__(self, **kwargs):
        self.definition = FeatureValueDefinitionPrototype(
            feature_name=AV(
                "accuracy and duration normalized then combined",
                "dur_acc_normed_combined",
            ),
            data_type="float64",
            validator=GREATER_THAN_ZERO,
            description="This feature is a combination of several measurements"
            "of the subject while drawing {shape} with their "
            "{hand} hand for the {attempt} attempt. It is "
            "computed by combining the duration and accuracy. The "
            "duration is the time spend between the first and "
            "last interaction of the subject with the screen. "
            "Accuracy is the median similarity between reference "
            "shape and drawn shape measured using dynamic time "
            "warping. "
            "See :func:`~dispel.signal.dtw.get_dtw_distance`. The "
            "combination is done following the formula: "
            "score = (normed_duration + normed_accuracy). The "
            "normalised version of duration and accuracy are "
            "computed by dividing the original property by the "
            "inter-quartile (Q95-Q05) of the resp. property "
            "computed on healthy population.",
        )
        super().__init__(**kwargs)


class ExtractDrawingCompletionRatio(ExtractShapeMixIn, ValidPathAssertionMixin):
    """Extract the completion ratio."""

    @staticmethod
    def identity(x):
        """Identity function."""
        return x

    properties = "distance_ratio"
    extract = identity
    definition = FeatureValueDefinitionPrototype(
        feature_name=AV("completion ratio", "c_ratio"),
        data_type="float",
        description="Percentage of completion of the shape {shape} shape with"
        " their {hand} hand for the {attempt} attempt.",
    )


class ExtractPressure(ExtractMultipleStep, ValidPathAssertionMixin):
    """Extract pressure-related features."""

    def __init__(self) -> None:
        function = {
            "func": lambda data: data["pressure"].mean(),
            "aggregation": AV("mean", "mean"),
            "validator": PRESSURE_VALIDATOR,
        }
        description = (
            "The {aggregation} pressure applied on the screen"
            "while drawing {shape} shape with the {hand} hand for "
            "the {attempt} attempt."
        )

        functions = [
            function,
            {
                "func": lambda data: variation(data["pressure"]),
                "aggregation": AV("coefficient of variation", "cv"),
            },
        ]

        definition = FeatureValueDefinitionPrototype(
            feature_name=AV("pressure", "press"),
            data_type="float64",
            description=description,
        )

        super().__init__("screen", functions, definition)


class ExtractReactionTime(ExtractStep, ValidPathAssertionMixin):
    """Extract reaction time features."""

    data_set_ids = "screen"
    transform_function = reaction_time
    definition = FeatureValueDefinitionPrototype(
        feature_name=AV("reaction time", "rt"),
        data_type="float64",
        unit="ms",
        validator=GREATER_THAN_ZERO,
        description="Time taken to initiate a goal directed movement "
        "after the {shape} shape is displayed for the {hand} hand "
        "at the {attempt} attempt.",
    )


class ExtractReactionTimeAll(AggregateRawDataSetColumn, ValidPathAssertionMixin):
    """Extract reaction time related features for all levels."""

    def __init__(self) -> None:
        definition = FeatureValueDefinitionPrototype(
            feature_name=AV("reaction time", "rt"),
            data_type="float64",
            unit="ms",
            validator=GREATER_THAN_ZERO,
            description="The {aggregation} time taken to initiate a goal "
            "directed movement after a shape is displayed.",
        )

        super().__init__(
            data_set_id="reaction-time",
            column_id="reaction_time",
            aggregations=DEFAULT_AGGREGATIONS,
            definition=definition,
        )


class ExtractPressureAll(AggregateRawDataSetColumn, ValidPathAssertionMixin):
    """Extract pressure related features for all levels."""

    def __init__(self) -> None:
        definition = FeatureValueDefinitionPrototype(
            feature_name=AV("pressure", "press"),
            data_type="float64",
            validator=PRESSURE_VALIDATOR,
            description="The {aggregation} pressure applied on the screen.",
        )

        super().__init__(
            data_set_id="screen",
            column_id="pressure",
            aggregations=DEFAULT_AGGREGATIONS,
            definition=definition,
        )


class ExtractPressureAllCV(AggregateRawDataSetColumn, ValidPathAssertionMixin):
    """Extract pressure related features for all levels."""

    def __init__(self):
        definition = FeatureValueDefinitionPrototype(
            feature_name=AV("pressure", "press"),
            data_type="float64",
            description="The {aggregation} pressure applied on the screen.",
        )

        super().__init__(
            data_set_id="screen",
            column_id="pressure",
            aggregations=[(variation, "coefficient of variation")],
            definition=definition,
        )


def _dtw_agg_dist(agg: str, matches: pd.DataFrame):
    """Extract aggregated minimum distance."""
    if agg == "mean":
        return np.mean(matches["min_distance"])
    if agg == "median":
        return np.median(matches["min_distance"])
    raise ValueError(f"Aggregation -{agg} is not defined for similarity")


class ExtractDTW(ExtractShapeMixIn, ValidPathAssertionMixin):
    """Abstract class to aggregate similarity using dynamic time warping."""

    properties = "up_sampled_valid_no_overshoot_matches"

    def __init__(self, aggregation: str, **kwargs):
        self.extract = lambda x: _dtw_agg_dist(aggregation, x)  # type: ignore
        self.definition = FeatureValueDefinitionPrototype(
            feature_name=AV("similarity", "sim"),
            data_type="float64",
            unit="point",
            aggregation=aggregation,
            validator=GREATER_THAN_ZERO,
            description=f"The {aggregation} coupling distance between the "
            "ideal {shape} shape target and the trajectory drawn "
            "with the {hand} hand at the {attempt} attempt. "
            "Coupling distance is measured using dynamic time "
            "warping.",
        )

        super().__init__(**kwargs)


class DrawTremorFeatures(ProcessingStepGroup):
    """A group of drawing processing steps for tremor features.

    Parameters
    ----------
    hand
        The hand on which the tremor features are to be computed.
    sensor
        The sensor on which the tremor features are to be computed.
    """

    def __init__(self, hand: HandModality, sensor: SensorModality):
        data_set_id = str(sensor)
        steps = [
            RenameColumns(data_set_id, **USER_ACC_MAP),
            SetTimestampIndex(
                f"{data_set_id}_renamed", DEFAULT_COLUMNS, duplicates="last"
            ),
            Resample(
                data_set_id=f"{data_set_id}_renamed_ts",
                aggregations=["mean", "ffill"],
                columns=DEFAULT_COLUMNS,
                freq=FREQ_20HZ,
            ),
            TremorFeatures(
                sensor=sensor, data_set_id=f"{data_set_id}_renamed_ts_resampled"
            ),
        ]
        super().__init__(
            steps,
            task_name=TASK_NAME,
            modalities=[hand.av],
            hand=hand,
            level_filter=LevelIdFilter(f"{hand.abbr}-all")
            & NotEmptyDatasetFilter(data_set_id),
        )


class DrawIntentionalTremorFeatures(ProcessingStepGroup):
    """A group of drawing processing steps for tremor features."""

    def __init__(self) -> None:
        data_set_id = "deceleration"
        new_column_names = {
            "x": "x_traj",
            "y": "y_traj",
            "min_distance": "diss",
            "tsTouch": "ts",
        }
        steps = [
            RenameColumns(data_set_id, **new_column_names),
            SetTimestampIndex(
                f"{data_set_id}_renamed",
                ["x_traj", "y_traj", "diss"],
                duplicates="last",
            ),
            Resample(
                f"{data_set_id}_renamed_ts",
                aggregations=["mean", "ffill"],
                columns=["x_traj", "y_traj", "diss"],
                freq=FREQ_60HZ,
            ),
            TremorFeatures(
                sensor=SensorModality.INTENTIONAL,
                data_set_id=f"{data_set_id}_renamed_ts_resampled",
                add_norm=False,
                add_average_signal=False,
                columns=["diss", "x_traj", "y_traj"],
            ),
        ]
        super().__init__(steps, level_filter=NotEmptyDatasetFilter(data_set_id))


class DRAWProcessingStepsGroupAll(ProcessingStepGroup):
    """Processing group for all aggregated levels."""

    def __init__(self) -> None:
        steps = [ExtractPressureAll(), ExtractPressureAllCV(), ExtractReactionTimeAll()]

        super().__init__(
            steps, task_name=TASK_NAME, level_filter=LevelIdFilter("all_levels")
        )


class ExtractCornerMeanDistance(ExtractShapeMixIn, ValidPathAssertionMixin):
    """Extract c accuracy."""

    @staticmethod
    def extract_mean_corner_max_dist(
        distances: Tuple[float, float, float],
    ) -> np.float64:
        """Extract mean maximum corner Frechet distance."""
        return np.mean(distances)  # type: ignore

    properties = "corners_max_dist"
    extract = extract_mean_corner_max_dist
    definition = FeatureValueDefinitionPrototype(
        feature_name=AV("mean corner max distance", "corner"),
        data_type="float64",
        unit="point",
        validator=RangeValidator(
            lower_bound=-812,
            upper_bound=812,
        ),
        description="The mean maximum distances from corners of the subject"
        "while drawing {shape} shape with their {hand} hand for "
        "the {attempt} attempt.",
    )


class AggregateCornerMeanDistance(AggregateFeatures):
    """Aggregate mean corner max distance feature over all attempts."""

    feature_ids = []
    for shape in ShapeModality:
        for attempt in AttemptModality:
            for hand in HandModality:
                feature_ids += [f"draw-{hand.abbr}_{shape.abbr}_{attempt.abbr}-corner"]
    definition = FeatureValueDefinitionPrototype(
        task_name=TASK_NAME,
        feature_name=AV("mean corner max distance", "corner"),
        data_type="float64",
        unit="point",
        aggregation="mean",
        validator=RangeValidator(
            lower_bound=-812,
            upper_bound=812,
        ),
        description="The mean maximum distances from corners of the subject"
        "while drawing over all attempts.",
    )


class ExtractAxesMeanDistance(ExtractShapeMixIn, ValidPathAssertionMixin):
    """Extract mean overshoot distance."""

    @staticmethod
    def extract_mean_axes_dist(
        distances: Tuple[float, float, float],
    ) -> np.float64:
        """Extract mean overshoot distances."""
        return np.mean(distances)  # type: ignore

    properties = "axis_overshoots"
    extract = extract_mean_axes_dist
    definition = FeatureValueDefinitionPrototype(
        feature_name=AV("mean axes overshoots", "axes_over"),
        data_type="float64",
        unit="point",
        validator=RangeValidator(
            lower_bound=-812,
            upper_bound=812,
        ),
        description="The mean overshoot distance from axes of the subject "
        "while drawing {shape} shape with their {hand} hand for "
        "the {attempt}. To ensure a unbiased value distribution, "
        "if the user does not go beyond an axis (no overshoot), "
        "the value will be negative. A user performing the "
        "drawing close to perfectly will thus present a average "
        "score close to zero.",
    )


class AggregateAxesMeanDistance(AggregateFeatures):
    """Aggregate mean corner max distance feature over all attempts."""

    feature_ids = []
    for shape in ShapeModality:
        for attempt in AttemptModality:
            for hand in HandModality:
                feature_ids += [
                    f"draw-{hand.abbr}_{shape.abbr}_{attempt.abbr}-axes_over"
                ]
    definition = FeatureValueDefinitionPrototype(
        task_name=TASK_NAME,
        feature_name=AV("mean axes overshoots", "axes_over"),
        data_type="float64",
        unit="point",
        aggregation="mean",
        validator=RangeValidator(lower_bound=-812, upper_bound=812),
        description="The mean overshoot distance from axes of the subject "
        "while drawing over all attempts. To ensure a unbiased "
        "value distribution, if the user does not go beyond an "
        "axis (no overshoot), the value will be negative. A user "
        "performing the drawing close to perfectly will thus "
        "present a average score close to zero.",
    )


class DrawAggregateModalitiesByHand(AggregateModalities):
    """Base step to aggregate features by hand for DRAW task.

    From the definition of the feature, all the features for the different
    shapes and attempts are retrieved (see get_modalities).
    """

    def __init__(self, hand: HandModality):
        self.hand = hand
        super().__init__()

    def get_definition(self, **kwargs) -> ValueDefinition:
        """Get the definition."""
        return cast(FeatureValueDefinitionPrototype, self.definition).create_definition(
            modalities=[self.hand.av], hand=self.hand.av
        )

    def get_modalities(self) -> List[List[Union[str, AV]]]:
        """Retrieve all modalities combinations for a given hand."""
        ids = []
        for shape in ShapeModality:
            for attempt in AttemptModality:
                ids.append([self.hand.av, shape.av, attempt.av])

        return ids


class DrawAggregateModalitiesByHandAndShape(AggregateModalities):
    """Base step to aggregate features by hand and shape for DRAW task.

    From the definition of the feature, all the features for the different
    attempts are retrieved (see get_modalities).
    """

    def __init__(self, hand: HandModality, shape: ShapeModality):
        self.hand = hand
        self.shape = shape
        super().__init__()

    def get_definition(self, **kwargs) -> ValueDefinition:
        """Get the definition."""
        return cast(FeatureValueDefinitionPrototype, self.definition).create_definition(
            modalities=[self.hand.av, self.shape.av],
            hand=self.hand.av,
            shape=self.shape,
        )

    def get_modalities(self) -> List[List[Union[str, AV]]]:
        """Retrieve all modalities combinations for a given hand and shape."""
        ids = []
        for attempt in AttemptModality:
            ids.append([self.hand.av, self.shape.av, attempt.av])

        return ids


class AggregateSimilarityByHand(DrawAggregateModalitiesByHand):
    """Average similarity values by hand."""

    definition = FeatureValueDefinitionPrototype(
        task_name=TASK_NAME,
        feature_name=AV("similarity", "sim"),
        data_type="float64",
        unit="point",
        aggregation="mean",
        validator=GREATER_THAN_ZERO,
        description="The mean coupling distance between the ideal target "
        "and the trajectory drawn with the {hand} hand for all "
        "shapes and attempts. Coupling distance is measured "
        "using dynamic time warping.",
    )


class AggregateSpeedSimilarityByHand(DrawAggregateModalitiesByHand):
    """Average speed/similarity values by hand."""

    definition = FeatureValueDefinitionPrototype(
        task_name=TASK_NAME,
        feature_name=AV("accuracy-normalized duration", "dur_acc"),
        data_type="float64",
        unit="point-1.ms-1",
        aggregation="mean",
        validator=GREATER_THAN_ZERO,
        description="The accuracy of the subject while drawing with their "
        "{hand} hand for all attempts normalized by the time "
        "spend between the first and last interaction of the "
        "subject with the screen. Accuracy is one over the "
        "dissimilarity between reference shape and drawn shape "
        "measured using dynamic time warping. See "
        ":func:`~dispel.signal.dtw.get_dtw_distance`.",
    )


class AggregateSparcByHand(DrawAggregateModalitiesByHand):
    """Average sparc values by hand."""

    definition = FeatureValueDefinitionPrototype(
        task_name=TASK_NAME,
        feature_name=AV("smoothness score", "smooth"),
        data_type="float64",
        validator=RangeValidator(upper_bound=0),
        aggregation="mean",
        description="A digital score of tremor using spectral arc length "
        "(SPARC) measurement algorithm for all shapes and "
        "attempts drawn with their {hand} hand.",
    )


class AggregateDurationByHand(DrawAggregateModalitiesByHand):
    """Average duration of drawing by hand."""

    definition = FeatureValueDefinitionPrototype(
        task_name=TASK_NAME,
        feature_name=AV("drawing duration", "user_dur"),
        data_type="float64",
        unit="ms",
        validator=GREATER_THAN_ZERO,
        aggregation="mean",
        description="The average time spend between the first and last "
        "interaction of the subject with the screen while drawing "
        "with their {hand} hand for all shapes and attempts.",
    )


class AggregateIntersectionsByHand(DrawAggregateModalitiesByHand):
    """Average total number of intersections by hand."""

    definition = FeatureValueDefinitionPrototype(
        task_name=TASK_NAME,
        feature_name=AV("intersections", "cross"),
        data_type="int32",
        validator=GREATER_THAN_ZERO,
        aggregation="mean",
        description="The average number of times the user cross the shape "
        "line with his finger while drawing with their {hand} "
        "hand for all the shapes and attempts.",
    )


class AggregateIntersectionsPerSecondsByHand(DrawAggregateModalitiesByHand):
    """Average the mean number of intersections per second by hand."""

    definition = FeatureValueDefinitionPrototype(
        task_name=TASK_NAME,
        feature_name=AV("intersections per second", "cross_per_sec"),
        data_type="float64",
        validator=GREATER_THAN_ZERO,
        aggregation="mean",
        description="The average number of intersection per second the "
        "user performs with his finger while drawing all shapes "
        "with their {hand} hand for all shapes and attempts.",
    )


class AggregateSimilarity(AggregateFeatures):
    """Average similarity values by hand."""

    feature_ids = []
    for shape in ShapeModality:
        for attempt in AttemptModality:
            for hand in HandModality:
                feature_ids += [
                    f"draw-{hand.abbr}_{shape.abbr}_{attempt.abbr}-sim-mean"
                ]

    definition = FeatureValueDefinitionPrototype(
        task_name=TASK_NAME,
        feature_name=AV("similarity", "sim"),
        data_type="float64",
        unit="point",
        aggregation="mean",
        validator=GREATER_THAN_ZERO,
        description="The mean coupling distance between the ideal target "
        "and the trajectory drawn with both hands for all "
        "shapes and attempts. Coupling distance is measured "
        "using dynamic time warping.",
    )


class AggregateSpeedSimilarity(AggregateFeatures):
    """Average speed/similarity globally."""

    feature_ids = []
    for shape in ShapeModality:
        for attempt in AttemptModality:
            for hand in HandModality:
                feature_ids += [
                    f"draw-{hand.abbr}_{shape.abbr}_{attempt.abbr}-dur_acc-mean"
                ]

    definition = FeatureValueDefinitionPrototype(
        task_name=TASK_NAME,
        feature_name=AV("accuracy-normalized duration", "dur_acc"),
        data_type="float64",
        unit="point-1.ms-1",
        validator=GREATER_THAN_ZERO,
        aggregation="mean",
        description="The mean of the accuracy of the subject while drawing "
        "with both hands for all shapes and attempts normalized "
        "by the time spend between the first and last interaction "
        "of the subject with the screen. Accuracy is one over the "
        "dissimilarity between reference shape and drawn shape "
        "measured using dynamic time warping. See "
        ":func:`~dispel.signal.dtw.get_dtw_distance`.",
    )


class AggregateSparc(AggregateFeatures):
    """Average smoothness scores globally."""

    feature_ids = []
    for shape in ShapeModality:
        for attempt in AttemptModality:
            for hand in HandModality:
                feature_ids += [f"draw-{hand.abbr}_{shape.abbr}_{attempt.abbr}-smooth"]

    definition = FeatureValueDefinitionPrototype(
        task_name=TASK_NAME,
        feature_name=AV("smoothness score", "smooth"),
        data_type="float64",
        validator=RangeValidator(upper_bound=0),
        aggregation="mean",
        description="A digital score of tremor using spectral arc length "
        "(SPARC) measurement algorithm for all shapes and "
        "attempts drawn with both hands.",
    )


class AggregateDuration(AggregateFeatures):
    """Average duration of drawing on all shapes, attempts and hands."""

    feature_ids = []
    for shape in ShapeModality:
        for attempt in AttemptModality:
            for hand in HandModality:
                feature_ids += [
                    f"draw-{hand.abbr}_{shape.abbr}_{attempt.abbr}-user_dur"
                ]

    definition = FeatureValueDefinitionPrototype(
        task_name=TASK_NAME,
        feature_name=AV("drawing duration", "user_dur"),
        data_type="float64",
        unit="ms",
        validator=GREATER_THAN_ZERO,
        aggregation="mean",
        description="The mean time spend between the first and last "
        "interaction of the subject with the screen while drawing "
        "all shapes and attempts with both hands.",
    )


class AggregateIntersectionsPerSeconds(AggregateFeatures):
    """Average the number of intersections per second globally."""

    feature_ids = []
    for shape in ShapeModality:
        for attempt in AttemptModality:
            for hand in HandModality:
                feature_ids += [
                    f"draw-{hand.abbr}_{shape.abbr}_{attempt.abbr}-cross_per_sec"
                ]

    definition = FeatureValueDefinitionPrototype(
        task_name=TASK_NAME,
        feature_name=AV("intersections per second", "cross_per_sec"),
        data_type="float64",
        validator=GREATER_THAN_ZERO,
        aggregation="mean",
        description="The average number of intersection per second the "
        "user performs with his finger while drawing with both "
        "hands for all shapes and attempts.",
    )


class AggregateIntersections(AggregateFeatures):
    """Average the number of intersections globally."""

    feature_ids = []
    for shape in ShapeModality:
        for attempt in AttemptModality:
            for hand in HandModality:
                feature_ids += [f"draw-{hand.abbr}_{shape.abbr}_{attempt.abbr}-cross"]

    definition = FeatureValueDefinitionPrototype(
        task_name=TASK_NAME,
        feature_name=AV("intersections", "cross"),
        data_type="int32",
        validator=GREATER_THAN_ZERO,
        aggregation="mean",
        description="The number of times the user cross the shape line "
        "with his finger while drawing with both hand for all the "
        "attempts.",
    )


class AggregateDistanceThresholdByHandAndShape(DrawAggregateModalitiesByHandAndShape):
    """Aggregate Distance threshold flag values by hand and shape."""

    definition = FeatureValueDefinitionPrototype(
        task_name=TASK_NAME,
        feature_name=AV("distance threshold", "dist_thresh"),
        data_type="bool",
        description="Whether the user completes at least 80% of the expected "
        "shape {shape} with their {hand} hand for both attempts.",
    )


class DrawUserDistanceThreshold(FlagLevelStep):
    """Flag if a drawing distance ratio is within expected range."""

    task_name = TASK_NAME
    flag_name = AV("distance threshold", "dist_thresh")
    flag_type = FlagType.BEHAVIORAL
    flag_severity = FlagSeverity.DEVIATION
    reason = (
        "The user drawing distance ratio is outside the expected "
        "range of 0.8 - 1.20 for the level {level_id}."
    )
    level_filter = ValidUserPath() & CreatableShape()

    @flag
    def _check_distance_threshold(self, level: Level, **_kwargs) -> bool:
        self.set_flag_kwargs(level_id=level.id)
        return (
            level.get_raw_data_set("shape")
            .data["shape"][0]
            .check_dist_thresh(RangeValidator(lower_bound=0.8, upper_bound=1.20))
        )


class FlagCompleteDrawing(FlagLevelStep):
    """Flag the drawing is complete."""

    task_name = TASK_NAME
    flag_name = AV("complete drawing", "complete_drawing")
    flag_type = FlagType.TECHNICAL
    flag_severity = FlagSeverity.DEVIATION
    reason = (
        "The drawing is not complete, the user has not reached the"
        "endZone or is not considered completed in the context during "
        "level {level_id}."
    )

    @flag
    def _check_complete_drawing(self, level: Level, **kwargs) -> bool:
        self.set_flag_kwargs(level_id=level.id, **kwargs)
        screen = level.get_raw_data_set("screen").data
        if "inEndZone" in screen:
            return screen.inEndZone.any()
        return level.context.get_raw_value("drawing_figure_completed")


class DrawTransformAndExtract(ProcessingStepGroup):
    """Processing group for all individual levels (shapes).

    Parameters
    ----------
    hand
        The hand on which the tremor features are to be computed.
    shape
        The shape on which the tremor features are to be computed.
    attempt
        The attempt on which the tremor features are to be computed.
    """

    def __init__(
        self, hand: HandModality, shape: ShapeModality, attempt: AttemptModality
    ) -> None:
        steps = [
            TransformReactionTime(),
            TransformInstantSpeed(),
            ExtractSpeedAccuracy("mean"),
            ExtractSpeedAccuracy("median"),
            ExtractDurationAccuracyNormedCombined(),
            ExtractDTW("mean"),
            ExtractDTW("median"),
            ExtractPressure(),
            ExtractDuration(),
            ExtractSparc(),
            ExtractReactionTime(),
            AggregateInstantSpeed(),
            ExtractDrawingCompletionRatio(),
        ]
        if shape == ShapeModality.SPIRAL:
            steps += [
                ExtractIntersections(),
                ExtractIntersectionsPerSeconds(),
                ExtractIntersectionsFrequency(),
            ]
        if shape in [ShapeModality.SQUARE, ShapeModality.SQUARE_COUNTER_CLOCK]:
            steps += [
                TransformDecelerationProfile(),
                DrawIntentionalTremorFeatures(),  # type: ignore
                ExtractCornerMeanDistance(),
                ExtractAxesMeanDistance(),
            ]
        super().__init__(
            steps,
            task_name=TASK_NAME,
            modalities=[hand.av, shape.av, attempt.av],
            hand=hand.av,
            shape=shape.av,
            attempt=attempt.av,
            level_filter=HandModalityFilter(hand)
            & ShapeModalityFilter(shape)
            & AttemptModalityFilter(attempt)
            & ValidUserPath()
            & CreatableShape()
            & ContinuousLevel(),
        )


class DrawingFlag(ProcessingStepGroup):
    """Processing group for all drawing flag."""

    def __init__(self) -> None:
        steps = [
            TransformUserAcceleration(),
            UpperLimbOrientationFlagger(),
            OnlyOneHandPerformed(task_name=TASK_NAME),
        ]

        super().__init__(steps, task_name=TASK_NAME)


def compute_pacman_score(
    shape_dataset: pd.DataFrame,
) -> float:
    """Compute the pacman score for a level.

    Parameters
    ----------
    shape_dataset: pd.DataFrame
        A data frame containing the DrawShape object

    Returns
    -------
    float
        The pacman score.
    """
    # Get the up-sampled path and the reference from the shape dataset
    user_path = shape_dataset["shape"][0].aggregate_valid_touches.valid_up_sampled_path
    reference = shape_dataset["shape"][0].reference
    user_x = user_path["x"].dropna().to_numpy()
    user_y = user_path["y"].dropna().to_numpy()
    ref_x = reference["x"].to_numpy()
    ref_y = reference["y"].to_numpy()

    path = np.column_stack((user_x, user_y))
    ref = np.column_stack((ref_x, ref_y))
    # Is the distance between target and ref points within the radius range
    eaten_by_user = (cdist(path, ref) < RADIUS).any(axis=0)

    return np.sum(eaten_by_user) / len(eaten_by_user)


class AddRawPacmanScore(ExtractStep, ValidPathAssertionMixin):
    """
    Add the raw pacman score.

    A target's point is considered as 'eaten' if that point is within the
    radius range of any other user path's point. The pacman score is the ratio
    between the number of 'eaten' points and the total number of target's
    point.
    """

    def __init__(self, level_filter: LevelFilter, **kwargs: object) -> None:
        data_set_ids = "shape"
        transform_function = compute_pacman_score
        definition = FeatureValueDefinitionPrototype(
            task_name=TASK_NAME,
            feature_name=AV("raw pacman score", "raw_pacman_score"),
            description="The raw pacman score is the ratio between the number "
            "of eaten points and the total number of target points"
            ". A target point is considered eaten if that point is"
            "within a radius range of any other point of the user "
            "path. See Radius definition for more information.",
            data_type="float",
            **kwargs,
        )
        super().__init__(
            data_set_ids=data_set_ids,
            transform_function=transform_function,
            definition=definition,
            level_filter=level_filter,
        )


class DrawOppositeDirection(FlagLevelStep):
    """Flag infinity shape drawn clockwise."""

    task_name = TASK_NAME
    flag_name = AV("opposite direction", "opp_direction")
    flag_type = FlagType.BEHAVIORAL
    flag_severity = FlagSeverity.DEVIATION
    reason = "The user is drawing in the opposite direction for the level {level_id}."
    level_filter = InfinityShapes() & CreatableShape() & ContinuousLevel()

    @flag
    def _assess_valid_direction(self, level: Level, **_kwargs) -> bool:
        self.set_flag_kwargs(level_id=level.id)
        # This flag only applies to infinity shapes
        if "inf" not in str(level.id):
            raise ValueError("This flag should run for infinity shape only.")
        # Get the shape
        _shape = level.get_raw_data_set("shape").data["shape"][0]

        # Compute the x coordinate of the user_path of the first quarter
        all_data = _shape.all_data
        x_quarter = all_data[: len(all_data) // 4]["x"].median()

        # center_x
        x_min = _shape.reference.x.min()
        x_max = _shape.reference.x.max()
        center_x = x_min + (x_max - x_min) / 2

        return x_quarter < center_x


class DrawOvershootRemoval(FlagLevelStep):
    """Flag drawing for which we removed more than 10% of user path."""

    level_filter = CreatableShape() & ValidUserPath() & ContinuousLevel()
    task_name = TASK_NAME
    flag_name = AV("overshoot removal", "excessive_overshoot_removal")
    flag_type = FlagType.TECHNICAL
    flag_severity = FlagSeverity.DEVIATION
    reason = (
        "The algorithm detected an overshoot of more than 10% for"
        " the level {level_id}."
    )

    @flag
    def _assess_overshoot_size(self, level: Level, **_kwargs) -> bool:
        self.set_flag_kwargs(level_id=level.id)
        # Get the shape
        _shape = level.get_raw_data_set("shape").data["shape"]
        if len(_shape) == 0:
            return False
        _shape = _shape[0]
        len_up_wo_overshoot = len(_shape.up_sampled_data_without_overshoot)
        len_up = len(_shape.up_sampled_valid_data)
        return (abs(len_up - len_up_wo_overshoot) / len_up) < 0.10


STEPS: List[ProcessingStep] = []

STEPS += [
    FlagCompleteDrawing(),
    CreateShapes(),
    CreatableShapeFlag(),
    TransformValidUserPath(),
    FlagContinuousDrawing(),
    DrawOvershootRemoval(),
    DrawOppositeDirection(),
    DrawUserDistanceThreshold(),
    DrawingFlag(),
]

for _hand in HandModality:
    for _shape, _attempt in product(ShapeModality, AttemptModality):
        STEPS += [DrawTransformAndExtract(_hand, _shape, _attempt)]

    STEPS += [
        ConcatenateLevels(
            new_level_id=f"{_hand.abbr}-all",
            data_set_id=["accelerometer", "gyroscope"],
            level_filter=HandModalityFilter(_hand)
            & ValidUserPath()
            & CreatableShape()
            & ContinuousLevel(),
        )
    ]

    STEPS += [
        DrawTremorFeatures(_hand, sensor)
        for sensor in [SensorModality.ACCELEROMETER, SensorModality.GYROSCOPE]
    ]

    STEPS += [
        AggregateSimilarityByHand(_hand),
        AggregateSpeedSimilarityByHand(_hand),
        AggregateSparcByHand(_hand),
        AggregateDurationByHand(_hand),
        AggregateIntersectionsPerSecondsByHand(_hand),
        AggregateIntersectionsByHand(_hand),
    ]
STEPS += [
    AggregateSimilarity(),
    AggregateSpeedSimilarity(),
    AggregateSparc(),
    AggregateDuration(),
    AggregateIntersectionsPerSeconds(),
    AggregateIntersections(),
]

for _hand in HandModality:
    for _shape in ShapeModality:
        STEPS += [AggregateDistanceThresholdByHandAndShape(_hand, _shape)]
STEPS += [
    ConcatenateLevels(
        new_level_id="all_levels",
        data_set_id=["screen", "reaction-time"],
        level_filter=ValidUserPath()
        & CreatableShape()
        & ContinuousLevel()
        & (
            ShapeModalityFilter(ShapeModality.SPIRAL)
            | ShapeModalityFilter(ShapeModality.INFINITY)
            | ShapeModalityFilter(ShapeModality.SQUARE)
            | ShapeModalityFilter(ShapeModality.SQUARE_COUNTER_CLOCK)
        ),
    ),
    DRAWProcessingStepsGroupAll(),
    AggregateCornerMeanDistance(),
    AggregateAxesMeanDistance(),
]

# Extra pacman score features
for hand in HandModality:
    for shape, attempt in product(ShapeModality, AttemptModality):
        modalities = [hand.av, shape.av, attempt.av]
        STEPS.append(
            ProcessingStepGroup(
                [
                    AddRawPacmanScore(
                        level_filter=HandModalityFilter(hand)
                        & ShapeModalityFilter(shape)
                        & AttemptModalityFilter(attempt)
                        & ValidUserPath()
                        & CreatableShape()
                        & ContinuousLevel()
                    )
                ],
                modalities=[hand.av, shape.av, attempt.av],
                hand=hand.av,
                shape=shape.av,
                attempt=attempt.av,
            )
        )
