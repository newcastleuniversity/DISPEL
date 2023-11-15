"""Six-Minute Walk test related functionality.

This module contains functionality to extract features for the *Six-minute
Walk* test (6MWT).
"""
from typing import Any, Callable, List, cast

import numpy as np
import pandas as pd

from dispel.data.core import Reading
from dispel.data.features import FeatureValue, FeatureValueDefinitionPrototype
from dispel.data.levels import Level
from dispel.data.raw import (
    ACCELEROMETER_COLUMNS,
    DEFAULT_COLUMNS,
    GRAVITY_COLUMNS,
    RawDataValueDefinition,
)
from dispel.data.validators import GREATER_THAN_ZERO
from dispel.data.values import AbbreviatedValue as AV
from dispel.processing.assertions import AssertEvaluationFinished
from dispel.processing.core import ProcessingStep, ProcessResultType
from dispel.processing.data_set import transformation
from dispel.processing.extract import (
    ExtractMultipleStep,
    FeatureDefinitionMixin,
    agg_column,
)
from dispel.processing.level import (
    LevelProcessingResult,
    LevelProcessingStep,
    ProcessingStepGroup,
)
from dispel.processing.level_filters import LastLevelFilter
from dispel.processing.modalities import LimbModality, SensorModality
from dispel.processing.transform import Apply, TransformStep
from dispel.providers.generic.activity.placement import ClassifyPlacement
from dispel.providers.generic.activity.turning import ElGoharyTurnDetection
from dispel.providers.generic.flags.generic import (
    FrequencyLowerThanGaitThres,
    MaxTimeIncrement,
)
from dispel.providers.generic.flags.le_flags import (
    ExcessiveTurns,
    LargeTurnsPerMinute,
    NoMovementDetected,
    NonBeltDetected6MWT,
    NotEnoughRectilinearWalkingOnBeltDetected,
    PercentageNotMoving,
    PercentageNotWalking,
)
from dispel.providers.generic.preprocessing import Detrend, PreprocessingSteps
from dispel.providers.generic.sensor import (
    FREQ_50HZ,
    EuclideanNorm,
    RenameColumns,
    Resample,
)
from dispel.providers.generic.tasks.gait.bout_strategy import (
    NO_BOUT_MODALITY,
    BoutStrategyModality,
)
from dispel.providers.generic.tasks.gait.core import (
    AverageRollingVerticalAcceleration,
    DetectBoutsBeltPlacementNoTurns,
    DetectBoutsHarmonic,
    DetectMovementBouts,
    ExtractStepCount,
    TransformGaitRegularity,
    TransformGaitRegularityWithoutBout,
    TransformStepDuration,
    TransformStepDurationWithoutBout,
    TransformStepVigor,
    TransformStepVigorWithoutBout,
)
from dispel.providers.generic.tasks.gait.del_din import (
    STEP_LENGTH_HEIGHT_RATIO,
    CWTFeatureTransformation,
    CWTFeatureWithoutBoutTransformation,
    FormatAccelerationGaitPy,
    GaitPyDetectSteps,
    GaitPyDetectStepsWithoutBout,
    GaitPyFeatures,
    GaitPyFeaturesWithoutBout,
    HeightChangeCOM,
    OptimizeGaitpyStepDataset,
    OptimizeGaitpyStepDatasetWithoutWalkingBout,
    get_subject_height,
)
from dispel.providers.generic.tasks.gait.lee import (
    LEE_MOD,
    LeeDetectSteps,
    LeeDetectStepsWithoutBout,
    LeeFeaturesGroup,
    LeeFeaturesWithoutBoutGroup,
    LeeTransformHipRotation,
)
from dispel.providers.generic.tremor import TremorFeatures
from dispel.signal.core import euclidean_norm
from dispel.signal.filter import butterworth_low_pass_filter

TASK_NAME = AV("Six-minute walk test", "6MWT")


class ComputeDistanceAndSpeed(TransformStep):
    """
    A raw data set transformation step to get user's distance and speed.

    When calculating the speed, we have chosen to ignore stoppage time.For
    more information, see the :class:`~dispel.processing.transform.TransformStep`.
    """

    data_set_ids = "gps"
    new_data_set_id = "distance_and_speed"
    definitions = [
        RawDataValueDefinition("distance", "calculated distance", "float64"),
        RawDataValueDefinition("speed", "calculated speed", "float64"),
    ]

    @staticmethod
    @transformation
    def _distance_speed(data: pd.DataFrame) -> pd.DataFrame:
        # Get rid of duplicated timestamps
        data = data[~data.ts.duplicated(keep="last")]
        displacement = data[["x", "y"]].diff().fillna(0)
        distance = euclidean_norm(displacement).astype(float)
        speed = distance / data.ts.diff().dt.total_seconds()

        return pd.DataFrame(dict(distance=distance, speed=speed))


class ComputeDistanceInertial(LevelProcessingStep, FeatureDefinitionMixin):
    """Compute distance from step count and step length."""

    definition = FeatureValueDefinitionPrototype(
        feature_name=AV(
            "distance walked from step count and height-based step length",
            "distance_walked_sc_hbsl",
        ),
        description="The distance walked from step count and height-based "
        "step len in meters.",
        unit="m",
        data_type="float",
    )

    def process_level(
        self, level: Level, reading: Reading, **kwargs
    ) -> ProcessResultType:
        """Process level to extract the distance walk."""
        feature = cast(
            FeatureValue,
            level.feature_set.get(
                ExtractStepCount.definition.create_definition(
                    modalities=[AV("gaitpy", "gp"), NO_BOUT_MODALITY],
                    bout_strategy_repr=NO_BOUT_MODALITY.abbr,
                    step_detector="unknown",
                    **kwargs,
                ).id
            ),
        )
        height = get_subject_height(level.context)
        result = feature.value * height * STEP_LENGTH_HEIGHT_RATIO

        yield LevelProcessingResult(
            step=self,
            sources=[level, feature],
            level=level,
            result=self.get_value(result, **kwargs),
        )


class SixMinuteWalkTestFeatures(ExtractMultipleStep):
    """A group of Gait extraction steps."""

    data_set_ids = "distance_and_speed"
    definition = FeatureValueDefinitionPrototype(
        data_type="float64", validator=GREATER_THAN_ZERO
    )
    transform_functions = [
        {
            "func": agg_column("distance", np.sum),
            "feature_name": AV("total distance walked", "distance_walked"),
            "unit": "m",
            "description": "The total distance walked during six minutes. "
            "The distance is measured based on GPS "
            "positions taken each second and derived to "
            "result in total distance.",
        },
        {
            "func": agg_column("speed", cast(Callable[[Any], float], np.mean)),
            "feature_name": AV(
                "walking speed (excluding stops)", "walking_speed_non_stop"
            ),
            "unit": "m/s",
            "description": "Mean walking speed measured using GPS "
            "positions measured each second excluding "
            "periods where between two measurements the "
            "displacement was smaller than one meter.",
            "aggregation": "mean",
        },
    ]


class GaitTremorFeatures(ProcessingStepGroup):
    """A group of gait steps for tremor features."""

    new_column_names = {
        "userAccelerationX": "x",
        "userAccelerationY": "y",
        "userAccelerationZ": "z",
    }
    steps = [
        RenameColumns("acc_ts_rotated_resampled_detrend", **new_column_names),
        TremorFeatures(
            sensor=SensorModality.ACCELEROMETER,
            data_set_id="acc_ts_rotated_resampled_detrend_renamed",
            add_norm=False,
        ),
        TremorFeatures(
            sensor=SensorModality.GYROSCOPE,
            data_set_id="gyroscope_ts_rotated_resampled",
            add_norm=False,
        ),
    ]


class TechnicalFlags(ProcessingStepGroup):
    """Technical flag steps."""

    steps = [FrequencyLowerThanGaitThres(), MaxTimeIncrement()]


class FlagPreprocessing(ProcessingStepGroup):
    """Preprocessing required for flags."""

    steps = [
        PercentageNotWalking(),
        PercentageNotMoving(),
        LargeTurnsPerMinute(),
    ]


class BehavioralDeviations(ProcessingStepGroup):
    """Behavioral deviation steps."""

    steps = [
        NonBeltDetected6MWT(),
        NoMovementDetected(),
        ExcessiveTurns(),
        NotEnoughRectilinearWalkingOnBeltDetected(),
    ]


class BehavioralInvalidations(ProcessingStepGroup):
    """Behavioral invalidation steps."""

    steps = [
        NonBeltDetected6MWT(acceptance_threshold=50),
        NoMovementDetected(acceptance_threshold=50),
        ExcessiveTurns(acceptance_threshold=5),
        NotEnoughRectilinearWalkingOnBeltDetected(acceptance_threshold=50),
    ]


class StepsGPS(ProcessingStepGroup):
    """Steps to process features from GPS signals."""

    steps = [
        ComputeDistanceAndSpeed(),
        SixMinuteWalkTestFeatures(),
    ]

    level_filter = LastLevelFilter()


class GaitPreprocessingSteps(ProcessingStepGroup):
    """Steps to pre-process gait activities."""

    steps = [
        PreprocessingSteps(
            data_set_id="accelerometer",
            limb=LimbModality.LOWER_LIMB,
            sensor=SensorModality.ACCELEROMETER,
            columns=ACCELEROMETER_COLUMNS,
            resample_freq=FREQ_50HZ,
        ),
        PreprocessingSteps(
            data_set_id="gyroscope",
            limb=LimbModality.LOWER_LIMB,
            sensor=SensorModality.GYROSCOPE,
            columns=DEFAULT_COLUMNS,
            resample_freq=FREQ_50HZ,
        ),
    ]


class TremorSteps(ProcessingStepGroup):
    """Steps to extract tremor-based features."""

    steps = [
        # Low Pass Gyroscope and Extracting Tremor Features
        Apply(
            "gyroscope_ts_rotated_resampled",
            butterworth_low_pass_filter,
            dict(order=5, cutoff=1.5, zero_phase=True),
            ["x"],
        ),
        # Extract Tremor Features
        GaitTremorFeatures(),
    ]


class WalkingBoutDetectionSteps(ProcessingStepGroup):
    """Steps to detect walking bouts."""

    steps = [
        # Format Accelerometer and detect walking bouts
        AverageRollingVerticalAcceleration(
            data_set_id="acc_ts_rotated_resampled_detrend", change_sign=True
        ),
        # Detect Walking Bouts
        DetectBoutsHarmonic(),
    ]


class WalkingBoutDynamicsDetectionSteps(ProcessingStepGroup):
    """Steps to detect dynamic walking bouts."""

    steps = [
        # Format Accelerometer and detect walking bouts
        AverageRollingVerticalAcceleration(
            data_set_id="acc_ts_rotated_resampled_detrend", change_sign=True
        ),
        # Detect Walking Bouts
        DetectMovementBouts(),
    ]


class PlacementClassificationSteps(ProcessingStepGroup):
    """Steps to classify device placement."""

    steps = [
        Resample(
            data_set_id="acc_ts",
            freq=FREQ_50HZ,
            aggregations=["mean", "ffill"],
            columns=ACCELEROMETER_COLUMNS + GRAVITY_COLUMNS,
        ),
        EuclideanNorm(data_set_id="acc_ts_resampled", columns=ACCELEROMETER_COLUMNS),
        # Format Accelerometer and detect walking bouts
        ClassifyPlacement(),
    ]


class TurnDetectionSteps(ProcessingStepGroup):
    """Steps to detect turns."""

    steps = [
        ElGoharyTurnDetection(
            "x", "gyroscope_ts_rotated_resampled_butterworth_low_pass_filter"
        ),
    ]


class MergeDynamicsPlacementTurn(ProcessingStepGroup):
    """Detect walking bouts filtered for placement and no turns."""

    steps = [DetectBoutsBeltPlacementNoTurns()]


class LeeDetectionAndTransformSteps(ProcessingStepGroup):
    """Lee et al. step detection processing steps."""

    steps = [
        # Lee Step detection
        LeeDetectSteps(data_set_ids=["movement_bouts", "vertical_acceleration"]),
        LeeDetectStepsWithoutBout(),
        # Transform Step TimeF
        TransformStepDuration(data_set_ids="lee_with_walking_bouts"),
        TransformStepDurationWithoutBout(data_set_ids="lee"),
        # Transform Hip Rotation
        LeeTransformHipRotation(on_walking_bouts=True),
        LeeTransformHipRotation(on_walking_bouts=False),
    ]


class GaitPySteps(ProcessingStepGroup):
    """GaitPy transformation steps."""

    steps = [
        # Format Acceleration to run GaitPy Step Detection
        FormatAccelerationGaitPy(),
        # Detect Steps using GaitPy Wavelet Transform
        GaitPyDetectStepsWithoutBout(data_set_ids="acc_ts_g_rotated_resampled_detrend"),
        GaitPyDetectSteps(
            data_set_ids=[
                "walking_placement_no_turn_bouts",
                "acc_ts_g_rotated_resampled_detrend",
            ]
        ),
        # Apply Physiological constraints
        OptimizeGaitpyStepDatasetWithoutWalkingBout("gaitpy"),
        OptimizeGaitpyStepDataset("gaitpy_with_walking_bouts"),
        HeightChangeCOM(
            data_set_ids=[
                "gaitpy_with_walking_bouts_optimized",
                "acc_ts_g_rotated_resampled_detrend",
            ],
            new_data_set_id="height_change_com_with_walking_bouts",
        ),
        HeightChangeCOM(
            data_set_ids=["gaitpy_optimized", "acc_ts_g_rotated_resampled_detrend"],
            new_data_set_id="height_change_com",
        ),
        # add euclidean norm of accelerometer combined with gyro
        EuclideanNorm(
            data_set_id="acc_ts_g_rotated_resampled",
            columns=[f"acceleration_{ax}" for ax in DEFAULT_COLUMNS],
        ),
        Detrend(
            data_set_id="acc_ts_g_rotated_resampled_euclidean_norm", columns=["norm"]
        ),
        # compute step power without bout
        TransformStepVigorWithoutBout(
            "norm",
            data_set_ids=[
                "acc_ts_g_rotated_resampled_euclidean_norm",
                "gaitpy_optimized",
            ],
        ),
        # compute step power with bout information
        TransformStepVigor(
            "norm",
            data_set_ids=[
                "acc_ts_g_rotated_resampled_euclidean_norm",
                "gaitpy_with_walking_bouts_optimized",
            ],
        ),
        TransformGaitRegularityWithoutBout(
            "norm",
            data_set_ids=[
                "acc_ts_g_rotated_resampled_euclidean_norm_detrend",
                "gaitpy_optimized",
            ],
        ),
        TransformGaitRegularity(
            "norm",
            data_set_ids=[
                "acc_ts_g_rotated_resampled_euclidean_norm_detrend",
                "gaitpy_with_walking_bouts_optimized",
            ],
        ),
        # Compute the GaitPy features for each gait cycle
        CWTFeatureWithoutBoutTransformation(
            data_set_ids=["gaitpy_optimized", "height_change_com"],
            new_data_set_id="cwt_features",
        ),
        # Compute the GaitPy features for each gait cycle
        CWTFeatureTransformation(
            data_set_ids=[
                "gaitpy_with_walking_bouts_optimized",
                "height_change_com_with_walking_bouts",
            ],
            new_data_set_id="cwt_features_with_walking_bouts",
        ),
    ]


class LeeFeatureSteps(ProcessingStepGroup):
    """Lee et al. feature extraction steps."""

    steps: List[ProcessingStep] = [
        LeeFeaturesGroup(
            bout_strategy=bout_strategy,
            modalities=[LEE_MOD, bout_strategy.av],
            bout_strategy_repr=bout_strategy.av,
        )
        for bout_strategy in BoutStrategyModality
    ]
    steps.append(LeeFeaturesWithoutBoutGroup(modalities=[LEE_MOD, NO_BOUT_MODALITY]))


class GaitPyFeatureSteps(ProcessingStepGroup):
    """GaitPy feature extraction steps."""

    steps: List[ProcessingStep] = [
        GaitPyFeatures(
            bout_strategy=bout_strategy,
            modalities=[AV("gaitpy", "gp"), bout_strategy.av],
            bout_strategy_repr=bout_strategy.av,
        )
        for bout_strategy in BoutStrategyModality
    ]
    steps.append(
        GaitPyFeaturesWithoutBout(
            modalities=[AV("gaitpy", "gp"), NO_BOUT_MODALITY],
        )
    )


class StepsDistanceInertial(ProcessingStepGroup):
    """Distance inertial processing steps."""

    steps = [ComputeDistanceInertial()]


class GaitCoreSteps(ProcessingStepGroup):
    """Core steps to process gait features."""

    steps = [
        # Preprocessing Steps and Transforms
        GaitPreprocessingSteps(),
        # Compute Tremor Features
        TremorSteps(),
        # Detect Bout Steps
        WalkingBoutDynamicsDetectionSteps(),
        # Detect Placement Steps
        PlacementClassificationSteps(),
        # Detect turns
        TurnDetectionSteps(),
        # Merge walking dynamics bouts, placement bout and turn bouts.
        MergeDynamicsPlacementTurn(),
        # GaitPy transformation
        GaitPySteps(),
        # GaitPy Features computation
        GaitPyFeatureSteps(),
        # Distance steps inertial
        StepsDistanceInertial(),
    ]

    level_filter = LastLevelFilter()


class GaitSteps(ProcessingStepGroup):
    """Gait processing steps."""

    steps = [
        # Sanity Check
        AssertEvaluationFinished(),
        # Flags
        TechnicalFlags(),
        # GPS features
        StepsGPS(),
        # core steps
        GaitCoreSteps(),
        # preprocess flags
        FlagPreprocessing(),
        # behavioral deviations (strict thresholds)
        BehavioralDeviations(),
        # behavioral invalidations (tolerant thresholds)
        BehavioralInvalidations(),
    ]
    kwargs = {"task_name": TASK_NAME}


class GaitStepsInclLee(ProcessingStepGroup):
    """Gait processing steps including Lee features."""

    steps = [
        GaitSteps(),
        LeeDetectionAndTransformSteps(),
        LeeFeatureSteps(),
    ]
    level_filter = LastLevelFilter()
    kwargs = {"task_name": TASK_NAME}
