"""A Module containing the Classes necessary to create SBT AP-ML gravity trajectories.

Among others, functionality is implemented to create diverse shapes(e.g., Circle,
Ellipse, Upsampled, Downsampled, rotations, etc).
"""

from typing import List

import numpy as np
import pandas as pd

from dispel.processing import ProcessingStep
from dispel.processing.data_set import StorageError, transformation
from dispel.processing.level import LevelIdFilter, ProcessingStepGroup
from dispel.processing.transform import TransformStep
from dispel.providers.generic.activity.placement import PLACEMENT_DEFINITIONS
from dispel.providers.generic.flags.le_flags import (
    TASK_NAME_SBT,
    FlagAdjustmentNonBeltSBT,
    FlagNonBeltSBT,
    PlacementClassificationGroup,
)
from dispel.providers.generic.tasks.sbt_utt.sbt import (
    BehaviouralFlagsGroup,
    DetectExcessiveMotionGroup,
    ExtractSpatioTemporalMeasuresGroup,
    FormattingGroup,
    PreProcessingStepsGroup,
    SBTTremorMeasuresGroup,
    TechnicalFlagsGroup,
    TransformAxisNames,
    TransformJerkNorm,
    TransformMergeValidSegments,
    TremorAndAxesGroup,
)
from dispel.signal.geometric import (
    downsample_dataset,
    draw_circle,
    draw_ellipse,
    rotate_points,
    synthetic_outliers,
    upsample_dataset,
)


class PreMartinezSteps(ProcessingStepGroup):
    """ProcessingStepGroup to do PreMartinez steps."""

    steps = [
        TechnicalFlagsGroup(),
        FormattingGroup(),
        PreProcessingStepsGroup(),
        SBTTremorMeasuresGroup("acc_ts_rotated_resampled_detrend"),
        TransformAxisNames("acc_ts_rotated_resampled_detrend_svgf_bhpf"),
    ]


class PostMartinezSteps(ProcessingStepGroup):
    """ProcessingStepGroup to do PostMartinez steps."""

    steps = [
        TransformJerkNorm(),
        DetectExcessiveMotionGroup(),
        ExtractSpatioTemporalMeasuresGroup(),
        BehaviouralFlagsGroup(),
    ]


class CreateSyntheticMartinezAcc(TransformStep):
    """Create data attributes for a synthetic Martinez Acceleration dataset."""

    data_set_ids = "martinez_accelerations"
    new_data_set_id = "martinez_accelerations"
    storage_error = StorageError.OVERWRITE
    level_filter = LevelIdFilter("sbt")
    definitions = TransformMergeValidSegments.definitions


class CreateSyntheticCircle(CreateSyntheticMartinezAcc):
    """Create data corresponding to a radius 1 circle for unit-testing."""

    @transformation
    def _create_circle(self, data: pd.DataFrame) -> pd.DataFrame:
        """Assign polar coordinate circle to AP and ML components."""
        x, y = draw_circle(length=len(data), radius=1)

        data.ap = x
        data.ml = y

        return data


class CreateSyntheticEllipseRotated(CreateSyntheticMartinezAcc):
    """Create data corresponding to 90deg rotated ellipse for unit-testing."""

    @transformation
    def _create_ellipse(self, data: pd.DataFrame) -> pd.DataFrame:
        """Assign polar coordinate circle to AP and ML components."""
        x, y = draw_ellipse(length=len(data), a=1, b=2)

        x_r, y_r = rotate_points(x, y, np.pi / 2)

        data.ap = x_r
        data.ml = y_r

        return data


class CreateSyntheticEllipse(CreateSyntheticMartinezAcc):
    """Create data corresponding to A=1 B=2 axes ellipse for unit-testing."""

    @transformation
    def _create_ellipse(self, data: pd.DataFrame) -> pd.DataFrame:
        """Assign polar coordinate circle to AP and ML components."""
        x, y = draw_ellipse(length=len(data), a=1, b=2)

        data.ap = x
        data.ml = y

        return data


class CreateSyntheticEllipseDoubled(CreateSyntheticMartinezAcc):
    """Create data corresponding to A=2 B=4 axes ellipse for unit-testing."""

    @transformation
    def _create_ellipse(self, data: pd.DataFrame) -> pd.DataFrame:
        """Assign polar coordinate circle to AP and ML components."""
        x, y = draw_ellipse(length=len(data), a=1 * 2, b=2 * 2)

        data.ap = x
        data.ml = y

        return data


class CreateSyntheticEllipseDoubledRotated(CreateSyntheticMartinezAcc):
    """Create data corresponding to a rotated 2*AB ellipse for unit-testing."""

    @transformation
    def _create_ellipse(self, data: pd.DataFrame) -> pd.DataFrame:
        """Assign polar coordinate circle to AP and ML components."""
        # We generate 2*AB ellipse points
        x, y = draw_ellipse(length=len(data), a=1 * 2, b=2 * 2)
        # We rotate points by 90 deg
        x_r, y_r = rotate_points(x, y, np.pi / 2)

        data.ap = x_r
        data.ml = y_r

        return data


class CreateSyntheticEllipseSameAxes(CreateSyntheticMartinezAcc):
    """Create data corresponding to a A=1 B=1 ellipse for unit-testing."""

    @transformation
    def _create_ellipse(self, data: pd.DataFrame) -> pd.DataFrame:
        """Assign polar coordinate circle to AP and ML components."""
        # We generate 2*AB ellipse points
        x, y = draw_ellipse(length=len(data), a=1, b=1)

        data.ap = x
        data.ml = y

        return data


class CreateSyntheticEllipseStretched(CreateSyntheticMartinezAcc):
    """Create data corresponding to a A=1 B=10 ellipse for unit-testing."""

    @transformation
    def _create_ellipse(self, data: pd.DataFrame) -> pd.DataFrame:
        """Assign polar coordinate circle to AP and ML components."""
        # We generate 2*AB ellipse points
        x, y = draw_ellipse(length=len(data), a=1, b=10)

        data.ap = x
        data.ml = y

        return data


class CreateSyntheticEllipseOutliers(CreateSyntheticMartinezAcc):
    """Create data corresponding to AB ellipse w/ outliers for unit-testing."""

    @transformation
    def _create_ellipse(self, data: pd.DataFrame) -> pd.DataFrame:
        """Assign polar coordinate circle to AP and ML components."""
        # We generate 2*AB ellipse points
        x, y = draw_ellipse(length=len(data), a=1, b=2)
        # We add outliers for 4% of data points (< 5%)
        x_r, y_r = synthetic_outliers(x, y, ratio_outlier=0.04)

        data.ap = x_r
        data.ml = y_r

        return data


class CreateSyntheticEllipseOutliersDoubled(CreateSyntheticMartinezAcc):
    """Create data of 2*AB ellipse w/ outliers for unit-testing."""

    @transformation
    def _create_ellipse(self, data: pd.DataFrame) -> pd.DataFrame:
        """Assign polar coordinate circle to AP and ML components."""
        # We generate 2*AB ellipse points
        x, y = draw_ellipse(length=len(data), a=1 * 2, b=2 * 2)
        # We add outliers
        x_r, y_r = synthetic_outliers(x, y, ratio_outlier=0.04)

        data.ap = x_r
        data.ml = y_r

        return data


class CreateSyntheticEllipseUpsampled(CreateSyntheticMartinezAcc):
    """Create data corresponding to upsampled ellipse for unit-testing."""

    @transformation
    def _create_ellipse(self, data: pd.DataFrame) -> pd.DataFrame:
        """Assign polar coordinate circle to AP and ML components."""
        # We generate 2*AB ellipse points
        x, y = draw_ellipse(length=len(data), a=1, b=2)

        data.ap = x
        data.ml = y

        # We add outliers for 4% of data points (< 5%)
        data = upsample_dataset(data, factor_freq=2)

        return data


class CreateSyntheticEllipseDownsampled(CreateSyntheticMartinezAcc):
    """Create data corresponding to a downsampled ellipse for unit-testing."""

    @transformation
    def _create_ellipse(self, data: pd.DataFrame) -> pd.DataFrame:
        """Assign polar coordinate circle to AP and ML components."""
        # We generate 2*AB ellipse points
        x, y = draw_ellipse(length=len(data), a=1, b=2)

        data.ap = x
        data.ml = y

        # We downsample
        data = downsample_dataset(data, ratio_freq=0.3)

        return data


class CreateSyntheticPlacement(TransformStep):
    """Create data corresponding to complex placement_bouts detected."""

    data_set_ids = "placement_bouts"
    new_data_set_id = "placement_bouts"
    storage_error = StorageError.OVERWRITE
    level_filter = LevelIdFilter("sbt")
    definitions = PLACEMENT_DEFINITIONS


class CreateSyntheticPlacementBoutsAdjust(CreateSyntheticPlacement):
    """Create data corresponding to complex placement_bouts detected."""

    @transformation
    def _create_new_bouts(self, data: pd.DataFrame) -> pd.DataFrame:
        # Make a view of the original_bout
        original_bout = data

        # Create 4 copies of the original bout to modify
        duplicate = data.append(original_bout).reset_index()
        duplicate = duplicate.append(duplicate).reset_index(drop=True)

        # Mark time of first bout and define bout boundaries at 3, 10, 20 sec
        time_zero = min([duplicate.start_time.min()])
        five_sec = time_zero + pd.Timedelta(5, unit="s")
        ten_sec = time_zero + pd.Timedelta(10, unit="s")
        twenty_sec = time_zero + pd.Timedelta(20, unit="s")

        # Define new end, start and duration for first bout
        duplicate.loc[0, "end_time"] = five_sec
        duplicate.loc[0, "duration"] = (five_sec - time_zero).total_seconds()

        # Define new end, start and duration for second bout
        duplicate.loc[1, "start_time"] = five_sec + pd.Timedelta(10, unit="ms")
        duplicate.loc[1, "end_time"] = ten_sec
        duplicate.loc[1, "duration"] = (ten_sec - five_sec).total_seconds()

        # Define new end, start and duration for third bout
        duplicate.loc[2, "start_time"] = ten_sec + pd.Timedelta(10, unit="ms")
        duplicate.loc[2, "end_time"] = twenty_sec
        duplicate.loc[2, "duration"] = (twenty_sec - ten_sec).total_seconds()

        # Define new end, start and duration for fourth bout
        duplicate.loc[3, "start_time"] = twenty_sec + pd.Timedelta(10, unit="ms")
        duplicate.loc[3, "duration"] = duplicate.loc[3, "duration"] - sum(
            duplicate.loc[0:2, "duration"]
        )

        # Define categories
        duplicate["placement"] = pd.Categorical(["pants", "belt", "belt", "belt"])

        duplicate = duplicate.drop(columns=["window_index_adjacent"])
        return duplicate


class CreateSyntheticPlacementBouts(CreateSyntheticPlacement):
    """Create data corresponding to complex placement_bouts detected."""

    @transformation
    def _create_new_bouts(self, data: pd.DataFrame) -> pd.DataFrame:
        # Make a view of the original_bout
        original_bout = data
        # Create 4 copies of the original bout to modify
        duplicate = data.append(original_bout).reset_index()
        duplicate = duplicate.append(duplicate).reset_index(drop=True)

        # Mark time of first bout and define bout boundaries at 3, 10, 20 sec
        time_zero = min([duplicate.start_time.min()])
        three_sec = time_zero + pd.Timedelta(3, unit="s")
        ten_sec = time_zero + pd.Timedelta(10, unit="s")
        twenty_sec = time_zero + pd.Timedelta(20, unit="s")

        # Define new end, start and duration for first bout
        duplicate.loc[0, "end_time"] = three_sec
        duplicate.loc[0, "duration"] = (three_sec - time_zero).total_seconds()

        # Define new end, start and duration for second bout
        duplicate.loc[1, "start_time"] = three_sec + pd.Timedelta(10, unit="ms")
        duplicate.loc[1, "end_time"] = ten_sec
        duplicate.loc[1, "duration"] = (ten_sec - three_sec).total_seconds()

        # Define new end, start and duration for third bout
        duplicate.loc[2, "start_time"] = ten_sec + pd.Timedelta(10, unit="ms")
        duplicate.loc[2, "end_time"] = twenty_sec
        duplicate.loc[2, "duration"] = (twenty_sec - ten_sec).total_seconds()

        # Define new end, start and duration for fourth bout
        duplicate.loc[3, "start_time"] = twenty_sec + pd.Timedelta(10, unit="ms")
        duplicate.loc[3, "duration"] = duplicate.loc[3, "duration"] - sum(
            duplicate.loc[0:2, "duration"]
        )

        # Define categories
        duplicate["placement"] = pd.Categorical(["pants", "handheld", "belt", "pants"])

        duplicate = duplicate.drop(columns=["window_index_adjacent"])
        return duplicate


class SBTSyntheticComplexProcessing(ProcessingStepGroup):
    """All processing steps to extract SBT measures."""

    pre_steps: List[ProcessingStep] = [
        TechnicalFlagsGroup(),
        FormattingGroup(),
        PreProcessingStepsGroup(),
        TremorAndAxesGroup(),
        DetectExcessiveMotionGroup(),
        ExtractSpatioTemporalMeasuresGroup(),
        PlacementClassificationGroup(),
    ]

    post_steps: List[ProcessingStep] = [
        FlagNonBeltSBT(),
        FlagAdjustmentNonBeltSBT(),
    ]

    level_filter = LevelIdFilter("sbt")
    kwargs = {"task_name": TASK_NAME_SBT}

    def set_steps(self, steps: List[ProcessingStep]):
        """Set processing steps part of the group."""
        super().set_steps(self.pre_steps + steps + self.post_steps)


class BalanceSyntheticProcessingSteps(ProcessingStepGroup):
    """A mixing class injects synthetic trajectories in SBT pipeline."""

    level_filter = LevelIdFilter("sbt")
    kwargs = {"task_name": TASK_NAME_SBT}

    def __init__(self, synthetic_step: ProcessingStep, *args, **kwargs):
        super().__init__(
            *args,
            steps=[
                PreMartinezSteps(),
                synthetic_step,
                PostMartinezSteps(),
            ],
            **kwargs,
        )
