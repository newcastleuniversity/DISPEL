"""U-turn Test module.

A module containing the functionality to process the *U-turn* test (UTT).
"""

import pandas as pd

from dispel.data.core import Reading
from dispel.data.measures import MeasureValueDefinitionPrototype
from dispel.data.raw import ACCELEROMETER_COLUMNS, DEFAULT_COLUMNS, GRAVITY_COLUMNS
from dispel.data.validators import GREATER_THAN_ZERO
from dispel.data.values import AbbreviatedValue as AV
from dispel.processing.core import (
    ErrorHandling,
    ProcessingControlResult,
    ProcessingStep,
    ProcessResultType,
    StopProcessingError,
)
from dispel.processing.extract import ExtractStep
from dispel.processing.level import LevelIdFilter, ProcessingStepGroup
from dispel.processing.level_filters import DurationFilter, LastLevelFilter
from dispel.processing.modalities import LimbModality, SensorModality
from dispel.processing.transform import Apply
from dispel.providers.generic.activity.placement import ClassifyPlacement
from dispel.providers.generic.activity.turning import (
    ElGoharyTurnDetection,
    ExtractTurnMeasures,
    RefineUTurns,
    TransformAbsTurnSpeed,
    TurnModality,
    WithinTurnSpeed,
)
from dispel.providers.generic.flags.generic import (
    FrequencyLowerThanGaitThres,
    MaxTimeIncrement,
)
from dispel.providers.generic.flags.le_flags import (
    NonBeltDetectedUTT,
    NoTurnsDetected,
    TruncateFirst5Turns,
)
from dispel.providers.generic.preprocessing import PreprocessingSteps
from dispel.providers.generic.sensor import FREQ_50HZ, EuclideanNorm, Resample
from dispel.providers.generic.tasks.gait.bout_strategy import NO_BOUT_MODALITY
from dispel.providers.generic.tasks.gait.core import (
    AverageRollingVerticalAcceleration,
    TransformStepDurationWithoutBout,
    step_count,
)
from dispel.providers.generic.tasks.gait.lee import (
    LEE_MOD,
    LeeDetectStepsWithoutBout,
    LeeMeasuresWithoutBoutGroup,
)
from dispel.signal.core import signal_duration
from dispel.signal.filter import butterworth_low_pass_filter

TASK_NAME = AV("U-turn test", "UTT")

MAX_UTT_DURATION_SECONDS = 300
r"""The maximum duration of a computable U-Turn Test."""

LEVEL_FILTER = (
    LevelIdFilter("utt") & LastLevelFilter() & DurationFilter(MAX_UTT_DURATION_SECONDS)
)
r"""Define the filter to use to compute UTT measures."""


class ExtractWalkingSpeed(ExtractStep):
    """Extract the UTT walking speed in step per seconds.

    Parameters
    ----------
    step_detection_data_set
        The identifier of the step detection output data set.
    accelerometer_data_set
        The identifier of the accelerometer norm input data set.
    """

    def __init__(self, step_detection_data_set: str, accelerometer_data_set: str):
        def _walking_speed(steps: pd.DataFrame, data: pd.DataFrame) -> float:
            return step_count(steps) / signal_duration(data)

        definition = MeasureValueDefinitionPrototype(
            measure_name=AV("walking speed", "ws"),
            data_type="float64",
            unit="step/s",
            validator=GREATER_THAN_ZERO,
            modalities=[LEE_MOD],
            description=f"Mean walking speed measured using the "
            f"{step_detection_data_set} step detection algorithm in terms of number "
            f"of steps per second.",
        )
        super().__init__(
            [step_detection_data_set, accelerometer_data_set],
            _walking_speed,
            definition,
        )


class AssertUTTDurationLess5Min(ProcessingStep):
    """Assertion UTT duration is less than five minutes."""

    def process_reading(self, reading: Reading, **kwargs) -> ProcessResultType:
        """Ensure UTT duration is less than five minutes."""
        if "utt" in reading.level_ids and not (
            reading.get_level("utt").duration.total_seconds() < MAX_UTT_DURATION_SECONDS
        ):
            yield ProcessingControlResult(
                step=self,
                error=StopProcessingError(
                    "The U-Turn test duration exceeds "
                    f"{MAX_UTT_DURATION_SECONDS} seconds.",
                    self,
                ),
                error_handling=ErrorHandling.RAISE,
            )


# turn speed measures
class ExtractTurnForModalitiesMeasures(ProcessingStepGroup):
    """Processing steps to extract turns for modalities."""

    steps = [
        ExtractTurnMeasures(
            turns_data_set_id="gyroscope_ts_rotated_resampled_butterworth_low_"
            "pass_filter_x_turns_u_refined",
            turn_modality=turn_modality,
        )
        for turn_modality in TurnModality
    ]


class TechnicalFlags(ProcessingStepGroup):
    """Processing steps for technical flags."""

    steps = [
        FrequencyLowerThanGaitThres(
            task_name=TASK_NAME,
            level_filter=TASK_NAME.abbr,
        ),
        MaxTimeIncrement(
            task_name=TASK_NAME,
            level_filter=TASK_NAME.abbr,
        ),
    ]


class PlacementClassification(ProcessingStepGroup):
    """Placement classification processing steps."""

    steps = [
        Resample(
            data_set_id="acc_ts",
            freq=FREQ_50HZ,
            aggregations=["mean", "ffill"],
            columns=ACCELEROMETER_COLUMNS + GRAVITY_COLUMNS,
        ),
        EuclideanNorm(
            data_set_id="acc_ts_resampled",
            columns=ACCELEROMETER_COLUMNS,
            level_filter="utt",
        ),
        # Format Accelerometer and detect walking bouts
        ClassifyPlacement(),
    ]


class BehavioralDeviations(ProcessingStepGroup):
    """Behavioral flags processing steps."""

    steps = [
        PlacementClassification(),
        TruncateFirst5Turns(
            data_set_ids="placement_bouts",
        ),
        NonBeltDetectedUTT(
            task_name=TASK_NAME,
        ),
        NoTurnsDetected(
            task_name=TASK_NAME,
        ),
    ]


class BehavioralInvalidations(ProcessingStepGroup):
    """Behavioral invalidation processing steps."""

    steps = [
        NonBeltDetectedUTT(
            acceptance_threshold=30,
            task_name=TASK_NAME,
        ),
    ]


class TurnPreprocessing(ProcessingStepGroup):
    """Turn pre-processing steps."""

    steps = [
        PreprocessingSteps(
            "accelerometer",
            LimbModality.LOWER_LIMB,
            SensorModality.ACCELEROMETER,
            columns=ACCELEROMETER_COLUMNS,
        ),
        PreprocessingSteps(
            "gyroscope",
            LimbModality.LOWER_LIMB,
            SensorModality.GYROSCOPE,
            columns=DEFAULT_COLUMNS,
        ),
        Apply(
            "gyroscope_ts_rotated_resampled",
            butterworth_low_pass_filter,
            dict(order=5, cutoff=1.5, zero_phase=True),
            ["x"],
        ),
    ]


class TurnProcessing(ProcessingStepGroup):
    """Turn processing steps."""

    steps = [
        ElGoharyTurnDetection(
            "x", "gyroscope_ts_rotated_resampled_butterworth_low_pass_filter"
        ),
        RefineUTurns(
            "x",
            [
                "gyroscope_ts_rotated_resampled_butterworth_low_pass_filter",
                "gyroscope_ts_rotated_resampled_butterworth_low_pass_filter_x_turns",
            ],
        ),
        TransformAbsTurnSpeed(
            "gyroscope_ts_rotated_resampled_butterworth_low_pass_filter_x_turns"
            "_u_refined"
        ),
        Apply(
            "gyroscope_ts_rotated_resampled",
            butterworth_low_pass_filter,
            dict(order=2, cutoff=5, zero_phase=True),
            list("xyz"),
            new_data_set_id="gyroscope_ts_rotated_resampled_butterworth_low_pass"
            "_filter_5hz",
        ),
        EuclideanNorm(
            data_set_id="gyroscope_ts_rotated_resampled_butterworth_low"
            "_pass_filter_5hz",
            columns=DEFAULT_COLUMNS,
            level_filter="utt",
        ),
        WithinTurnSpeed(
            "x",
            [
                "gyroscope_ts_rotated_resampled_butterworth_low_pass_filter_5hz",
                "gyroscope_ts_rotated_resampled_butterworth_low_pass_filter"
                "_x_turns_u_refined",
            ],
        ),
        WithinTurnSpeed(
            "norm",
            [
                "gyroscope_ts_rotated_resampled_butterworth_low_pass_filter_5hz_"
                "euclidean_norm",
                "gyroscope_ts_rotated_resampled_butterworth_low_pass_filter"
                "_x_turns_u_refined",
            ],
        ),
    ]


class LeeStepProcessing(ProcessingStepGroup):
    """Lee-specific processing steps."""

    steps = [
        AverageRollingVerticalAcceleration(
            data_set_id="acc_ts_rotated_resampled_detrend", change_sign=True
        ),
        # Lee Step Detection Measures
        # Without walking bouts Detection and Step Count
        LeeDetectStepsWithoutBout(),
        TransformStepDurationWithoutBout(data_set_ids="lee"),
        LeeMeasuresWithoutBoutGroup(modalities=[LEE_MOD, NO_BOUT_MODALITY]),
    ]


#: UTT processing steps
class UTTProcessingSteps(ProcessingStepGroup):
    """UTT processing steps."""

    steps = [
        TurnPreprocessing(),
        TurnProcessing(),
        ExtractTurnForModalitiesMeasures(),
        LeeStepProcessing(),
        ExtractWalkingSpeed(
            step_detection_data_set="lee",
            accelerometer_data_set="vertical_acceleration",
        ),
        BehavioralDeviations(),
        BehavioralInvalidations(),
    ]

    level_filter = LEVEL_FILTER
    kwargs = {"task_name": TASK_NAME}
