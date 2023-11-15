"""A module to store the lower extremity (le) related flags."""
from typing import Tuple

import pandas as pd
from numpy import deg2rad

from dispel.data.features import FeatureValueDefinitionPrototype
from dispel.data.flags import FlagSeverity, FlagType
from dispel.data.levels import Level
from dispel.data.raw import ACCELEROMETER_COLUMNS, GRAVITY_COLUMNS
from dispel.data.validators import GREATER_THAN_ZERO
from dispel.data.values import AbbreviatedValue as AV
from dispel.processing.data_set import transformation
from dispel.processing.extract import ExtractStep
from dispel.processing.flags import flag
from dispel.processing.level import FlagLevelStep, ProcessingStepGroup
from dispel.processing.transform import TransformStep
from dispel.providers.generic.activity.placement import (
    PLACEMENT_DEFINITIONS,
    ClassifyPlacement,
)
from dispel.providers.generic.sensor import FREQ_50HZ, EuclideanNorm, Resample
from dispel.providers.generic.tasks.gait.core import WALKING_BOUT_TURN_MAX_ANGLE

MAX_GAIT_PERC_NOT_IN_BELT = 10
"""Maximum percentage allowed not in belt for Gait."""

MAX_SBT_PERC_NOT_IN_BELT = 0
"""Maximum percentage allowed not in belt for SBT assessment."""

MAX_PERC_NOT_MOVING = 10
"""Maximum percentage of level duration allowed not moving."""

MAX_PERC_NOT_RECTILINEAR_BELT_WALK = 10
"""Maximum percentage of level duration allowed not rectilinear on-belt walking."""

MAX_TURNS_PER_MIN = 3
"""Maximum number of turns per minute allowed."""

START_TIME_ASSESSMENT = 0
"""Time of start of the assessment in seconds."""

END_POSTURAL_ADJUSTMENT = 5
"""Time of end of postural adjustment in seconds."""

MAX_FRACTION_EXCESSIVE_MOTION = 0.0
"""The maximum fraction of excessive motion over the total duration of
the assessment."""
TASK_NAME_SBT = AV("Static Balance Test", "sbt")


def truncate_placement(
    df_placement: pd.DataFrame, threshold_start: float
) -> pd.DataFrame:
    """Truncate `placement_bouts` before threshold_start."""
    # Detect start time
    time_zero = df_placement.iloc[0]["start_time"]
    # Compute time of postural adjustment end from start
    end_posture_adjust = time_zero + pd.Timedelta(threshold_start, unit="s")

    # Find bout containing the threshold_start and posterior
    idx = df_placement["end_time"] >= end_posture_adjust
    # Filter those bouts
    df_placement = df_placement.loc[idx, :].reset_index(drop=True)
    # Adjust first start_time and duration
    df_placement.loc[0, "start_time"] = end_posture_adjust
    df_placement.loc[:, "duration"] = (
        df_placement["end_time"] - df_placement["start_time"]
    ).dt.total_seconds()
    return df_placement


class PlacementClassificationGroup(ProcessingStepGroup):
    """A ProcessingStepGroup concatenating PlacementClassificationGroup steps."""

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
        ),
        # Format Accelerometer and detect walking bouts
        ClassifyPlacement(),
    ]


def perc_non_belt(
    level: Level, data_set_id: str = "placement_bouts", start_time: float = 0
) -> Tuple[float, list]:
    """Return non belt percentage and placements."""
    df_placement = level.get_raw_data_set(data_set_id).data

    df_placement = truncate_placement(df_placement, start_time)

    belt_duration = df_placement[df_placement["placement"] == "belt"].duration.sum()
    total_duration = df_placement.duration.sum()
    perc_not_belt = 100 * (1 - (belt_duration / total_duration))
    return perc_not_belt, df_placement.placement.to_list()


class NonBeltDetected(FlagLevelStep):
    """Flag record with non-belt period greater than threshold."""

    flag_name = AV(
        "{non_belt_portion}% was detected at {placements} (more than the"
        "{acceptance_threshold}% accepted)",
        "non_belt_greater_than_{acceptance_threshold}_perc",
    )
    flag_type = FlagType.BEHAVIORAL
    flag_severity = FlagSeverity.DEVIATION
    reason = (
        "{non_belt_portion}% non-belt portion detected at {placements} "
        "(exceeds {acceptance_threshold}% accepted)"
    )


class NonBeltPostAdjustmentDetected(FlagLevelStep):
    """Flag record with non-belt period greater than threshold."""

    flag_name = AV(
        "{non_belt_portion}% (more than the {acceptance_threshold}% "
        "accepted) was detected post-adjustment at {placements}",
        "non_belt_post_adjustment_greater_than_{acceptance_threshold}_perc",
    )
    flag_type = FlagType.BEHAVIORAL
    flag_severity = FlagSeverity.INVALIDATION
    reason = (
        "{non_belt_portion}% non-belt post-adjustment portion detected"
        " at {placements} (exceeds {acceptance_threshold}% accepted)"
    )


class NonBeltDetected6MWT(NonBeltDetected):
    """Flag gait record with non-belt period greater than threshold."""

    def __init__(
        self, acceptance_threshold: float = MAX_GAIT_PERC_NOT_IN_BELT, *args, **_kwargs
    ):
        self.acceptance_threshold = acceptance_threshold

        super().__init__(*args, **_kwargs)

    @flag
    def _off_belt_period(
        self,
        level: Level,
        start_time: float = START_TIME_ASSESSMENT,
        **_kwargs,
    ) -> bool:
        non_belt_portion, placements = perc_non_belt(
            level, "placement_bouts", start_time
        )

        self.set_flag_kwargs(
            acceptance_threshold=self.acceptance_threshold,
            non_belt_portion=non_belt_portion,
            placements=placements,
        )

        return non_belt_portion <= self.acceptance_threshold


class NonBeltDetectedUTT(NonBeltDetected):
    """Flag gait record with non-belt period greater than threshold."""

    def __init__(
        self, acceptance_threshold: float = MAX_GAIT_PERC_NOT_IN_BELT, *args, **kwargs
    ):
        self.acceptance_threshold = acceptance_threshold

        super().__init__(*args, **kwargs)

    @flag
    def _off_belt_period(
        self,
        level: Level,
        start_time: float = START_TIME_ASSESSMENT,
        **_kwargs,
    ) -> bool:
        non_belt_portion, placements = perc_non_belt(
            level, "placement_bouts_first_5_turns", start_time
        )

        self.set_flag_kwargs(
            acceptance_threshold=self.acceptance_threshold,
            non_belt_portion=non_belt_portion,
            placements=placements,
        )
        return non_belt_portion <= self.acceptance_threshold


class FlagNonBeltSBT(NonBeltDetected):
    """Flag SBT records with non-belt period greater than threshold."""

    acceptance_threshold: float = MAX_SBT_PERC_NOT_IN_BELT
    placements: list = []

    @flag
    def _off_belt_period(
        self,
        level: Level,
        acceptance_threshold: float = acceptance_threshold,
        start_time: float = START_TIME_ASSESSMENT,
        **_kwargs,
    ) -> bool:
        non_belt_portion, placements = perc_non_belt(
            level, "placement_bouts", start_time
        )

        self.set_flag_kwargs(
            acceptance_threshold=acceptance_threshold,
            non_belt_portion=non_belt_portion,
            placements=placements,
        )

        return non_belt_portion <= acceptance_threshold


class FlagAdjustmentNonBeltSBT(NonBeltPostAdjustmentDetected):
    """Flag SBT records with non-belt post-adjustment period."""

    acceptance_threshold: float = MAX_SBT_PERC_NOT_IN_BELT
    placements: list = []

    @flag
    def _off_belt_period(
        self,
        level: Level,
        acceptance_threshold: float = acceptance_threshold,
        start_time: float = END_POSTURAL_ADJUSTMENT,
        **_kwargs,
    ) -> bool:
        non_belt_portion, placements = perc_non_belt(
            level, "placement_bouts", start_time
        )

        self.set_flag_kwargs(
            acceptance_threshold=acceptance_threshold,
            non_belt_portion=non_belt_portion,
            placements=placements,
        )

        return non_belt_portion <= acceptance_threshold


class LargeTurnsPerMinute(ExtractStep):
    """Extract rate of large turns used in rectilinear walking bout detection."""

    data_set_ids = [
        "gyroscope_ts_rotated_resampled_butterworth_low_pass_filter_x_turns",
        "accelerometer",
    ]
    definition = FeatureValueDefinitionPrototype(
        feature_name=AV("large_turns_per_min", "large_turns_per_min"),
        data_type="float32",
        validator=GREATER_THAN_ZERO,
        description="The number of large turns per minute.",
    )

    @staticmethod
    @transformation
    def large_turns_per_min(turns: pd.DataFrame, acc: pd.DataFrame) -> float:
        """Compute number of large turns per minute."""
        # number of large turns that affect rectilinear walking detection
        n_turns = (
            turns["angle"].abs() > deg2rad(WALKING_BOUT_TURN_MAX_ANGLE.value)
        ).sum()
        # compute level duration from accelerometer data
        duration_seconds = (acc["ts"].max() - acc["ts"].min()).total_seconds()
        duration_minutes = duration_seconds / 60
        # return turns per minute
        return n_turns / duration_minutes


class ExcessiveTurns(FlagLevelStep):
    """Flag record with at least a pre-specified number of turns per minute."""

    flag_type = FlagType.BEHAVIORAL
    flag_severity = FlagSeverity.DEVIATION

    def __init__(
        self, acceptance_threshold: float = MAX_TURNS_PER_MIN, *args, **kwargs
    ):
        self.flag_name = AV(
            f"At least {acceptance_threshold} turns per minute detected",
            f"number_of_turns_per_min_greater_than_{acceptance_threshold}",
        )
        self.reason = f"At least {acceptance_threshold} turns per minute detected."
        self.acceptance_threshold = acceptance_threshold

        super().__init__(*args, **kwargs)

    @flag
    def _turns_per_minute(self, level: Level, **_kwargs) -> bool:
        task_name = _kwargs["task_name"].abbr.lower()
        feature_id = f"{task_name}-large_turns_per_min"
        turns_per_minute = level.feature_set.get(feature_id).value
        return turns_per_minute < self.acceptance_threshold


def percentage_off_bouts(data: pd.DataFrame, **_kwargs) -> bool:
    """Compute percentage of non-moving portion of the signal."""
    movement_duration = data[data["detected_walking"]].duration.sum()
    total_duration = data.duration.sum()
    perc_off_bouts = 100 * (1 - (movement_duration / total_duration))
    return perc_off_bouts


class PercentageNotMoving(ExtractStep):
    """Extract percentage of recording where no walking is detected.."""

    data_set_ids = ["movement_bouts"]
    definition = FeatureValueDefinitionPrototype(
        feature_name=AV("perc_not_moving", "perc_not_moving"),
        data_type="float32",
        validator=GREATER_THAN_ZERO,
        description="The percentage of the signal with no movement detected.",
    )

    @staticmethod
    @transformation
    def no_movement_period(df_walking: pd.DataFrame, **_kwargs) -> bool:
        """Compute percentage of non moving portion of the signal."""
        return percentage_off_bouts(df_walking)


class PercentageNotWalking(ExtractStep):
    """Extract percentage of recording where no walking is detected.."""

    data_set_ids = ["walking_placement_no_turn_bouts"]
    definition = FeatureValueDefinitionPrototype(
        feature_name=AV("perc_not_walking", "perc_not_walking"),
        data_type="float32",
        validator=GREATER_THAN_ZERO,
        description="The percentage of the signal with no walking detected.",
    )

    @staticmethod
    @transformation
    def no_walking_period(df_walking: pd.DataFrame, **_kwargs) -> bool:
        """Compute percentage of non walking portion of the signal."""
        return percentage_off_bouts(df_walking)


class NoMovementDetected(FlagLevelStep):
    """Flag record with at least a pre-specified percentage of non-movement."""

    flag_type = FlagType.BEHAVIORAL
    flag_severity = FlagSeverity.DEVIATION

    def __init__(
        self, acceptance_threshold: float = MAX_PERC_NOT_MOVING, *args, **kwargs
    ):
        self.flag_name = AV(
            f"At least {acceptance_threshold}% non-movement period detected",
            f"non_movement_greater_than_{acceptance_threshold}_perc",
        )
        self.reason = f"At least {acceptance_threshold}% non-movement period detected."
        self.acceptance_threshold = acceptance_threshold

        super().__init__(*args, **kwargs)

    @flag
    def _no_movement_period(self, level: Level, **_kwargs) -> bool:
        task_name = _kwargs["task_name"].abbr.lower()
        feature_id = f"{task_name}-perc_not_moving"
        perc_not_walking = level.feature_set.get(feature_id).value
        return perc_not_walking < self.acceptance_threshold


class NotEnoughRectilinearWalkingOnBeltDetected(FlagLevelStep):
    """Flag record containing less than a % of rectilinear walking on belt."""

    flag_type = FlagType.BEHAVIORAL
    flag_severity = FlagSeverity.DEVIATION

    def __init__(
        self,
        acceptance_threshold: float = MAX_PERC_NOT_RECTILINEAR_BELT_WALK,
        *args,
        **_kwargs,
    ):
        self.flag_name = AV(
            f"At least {acceptance_threshold}% non-rectilinear belt walking detected",
            f"non_rectilinear_on_belt_walking_greater_than_{acceptance_threshold}_perc",
        )
        self.reason = (
            f"At least {acceptance_threshold}% non-rectilinear on-belt period."
        )
        self.acceptance_threshold = acceptance_threshold

        super().__init__(*args, **_kwargs)

    @flag
    def _no_walking_period(self, level: Level, **_kwargs) -> bool:
        task_name = _kwargs["task_name"].abbr.lower()
        feature_id = f"{task_name}-perc_not_walking"
        perc_not_walking = level.feature_set.get(feature_id).value
        return perc_not_walking < self.acceptance_threshold


class NoTurnsDetected(FlagLevelStep):
    """Flag record with no turns present."""

    flag_name = AV("No turns detected.", "no_turns_detected")
    flag_type = FlagType.BEHAVIORAL
    flag_severity = FlagSeverity.DEVIATION
    reason = "At least 1 turn should be detected."

    @flag
    def _check_n_turns(self, level: Level, **_kwargs) -> bool:
        turns = level.get_raw_data_set(
            "gyroscope_ts_rotated_resampled_butterworth_low_pass_filter"
            "_x_turns_u_refined"
        ).data

        return turns.shape[0] > 0


def truncate_first_5_turns(data: pd.DataFrame, turns: pd.DataFrame) -> pd.DataFrame:
    """Truncate the data frame to keep first 5 turns."""
    # handle corner case with no turns - return all data
    if turns.shape[0] == 0:
        return data

    # ensure there is a fourth turn otherwise take the last detected turn
    last_index = min(4, turns.shape[0] - 1)
    # find end time of fourth (or last) turn
    ts_ends = turns.iloc[last_index]["end"]
    # find indexes where start time is before end time of 4th (last) turn
    idx = data["start_time"] < ts_ends
    idx = idx[idx].index
    # filter respective bouts
    data = data.loc[idx, :]
    # replace end time of bout with end time of 4th (last) turn
    data.loc[data.shape[0] - 1, "end_time"] = ts_ends
    # update duration
    data.loc[:, "duration"] = (data["end_time"] - data["start_time"]).dt.total_seconds()
    return data


class TruncateFirst5Turns(TransformStep):
    """Truncate UTT placement_bouts to keep only the first 5 turns."""

    def get_new_data_set_id(self) -> str:
        """Suffix new_data_set_id with current data_set_id & first_5_turns."""
        assert len(data_set_ids := list(self.get_data_set_ids())) == 1
        return data_set_ids[0] + "_first_5_turns"

    @transformation
    def _truncate_first_5_turns(self, data: pd.DataFrame, level: Level):
        """Truncate the data frame to keep first 5 turns."""
        turns = level.get_raw_data_set(
            "gyroscope_ts_rotated_resampled_butterworth_low_pass_filter"
            "_x_turns_u_refined"
        ).data

        return truncate_first_5_turns(data, turns)

    definitions = PLACEMENT_DEFINITIONS


def truncate_signal(signal: pd.DataFrame, thrs_start: float) -> pd.DataFrame:
    """Truncate `signal` before threshold_start."""
    # Detect start time
    time_zero = signal.index[0]
    # Compute time of postural adjustment end from start
    end_posture_adjust = time_zero + pd.Timedelta(thrs_start, unit="s")

    # Find posterior to start_time
    idx = signal.index >= end_posture_adjust
    # Preserve signal from start_time
    return signal.loc[idx, :]


def perc_excessive_motion(
    level: Level, data_set_id: str = "flagged_motion", thrs_start: float = 0
) -> float:
    """Return excessive motion percentage."""
    # Select the target dataset to probe
    flag = level.get_raw_data_set(data_set_id).data
    # Truncate the signal to the corresponding start time (adjust/stabilise)
    flag = truncate_signal(flag, thrs_start)
    # Output the percentage of excessive motion period over total duration
    return (len(flag[flag["flag"]]) / len(flag)) * 100


class ExcessiveMotionFlagger(FlagLevelStep):
    """Flag minimum length of valid recording without excessive motion."""

    flag_type = FlagType.BEHAVIORAL
    flag_severity = FlagSeverity.DEVIATION
    acceptance_threshold: float = MAX_FRACTION_EXCESSIVE_MOTION
    start_time: float = START_TIME_ASSESSMENT

    @flag
    def _excessive_motion_detected(
        self,
        level: Level,
        acceptance_threshold: float = acceptance_threshold,
        start_time: float = start_time,
        **kwargs,
    ) -> bool:
        excessive_motion_portion = perc_excessive_motion(
            level, "flagged_motion", start_time
        )

        self.set_flag_kwargs(
            acceptance_threshold=acceptance_threshold,
            excessive_motion_portion=excessive_motion_portion,
        )
        super().__init__(**kwargs)

        return excessive_motion_portion <= acceptance_threshold


class FlagStabilisationExcessiveMotion(ExcessiveMotionFlagger):
    """Flag minimum length of valid recording without excessive motion."""

    flag_name = AV(
        "Excessive Motion Detected Post-adjustment", "excessive_post_adjustment_motion"
    )
    reason = (
        "{excessive_motion_portion:.1f}% excessive motion post-adjustment portion "
        "detected (exceeds {acceptance_threshold}% accepted)"
    )

    def __init__(self, **kwargs):
        self.start_time = (START_TIME_ASSESSMENT,)

        super().__init__(**kwargs)


class FlagPostAdjustExcessiveMotion(ExcessiveMotionFlagger):
    """Flag minimum length of valid recording without excessive motion."""

    flag_name = AV("Excessive Motion Detected during Stabilisation", "excessive_motion")
    reason = (
        "{excessive_motion_portion}% excessive motion portion detected "
        "(exceeds {acceptance_threshold}% accepted)"
    )

    def __init__(self, **kwargs):
        self.start_time = (END_POSTURAL_ADJUSTMENT,)

        super().__init__(**kwargs)
