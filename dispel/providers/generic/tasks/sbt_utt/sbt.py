"""Static Balance Test module.

A module containing the functionality to process the *Static Balance* test
(SBT).
"""
from typing import Iterable, List, Optional

import pandas as pd
from scipy.signal import medfilt
from scipy.stats import iqr as scipy_iqr

from dispel.data.core import EntityType, Reading
from dispel.data.features import FeatureValueDefinitionPrototype
from dispel.data.flags import FlagSeverity, FlagType
from dispel.data.levels import Level
from dispel.data.raw import ACCELEROMETER_COLUMNS, RawDataValueDefinition
from dispel.data.validators import GREATER_THAN_ZERO
from dispel.data.values import AbbreviatedValue as AV
from dispel.processing import ProcessingStep
from dispel.processing.data_set import FlagDataSetStep, transformation
from dispel.processing.extract import ExtractStep
from dispel.processing.flags import flag
from dispel.processing.level import LevelIdFilter, ProcessingStepGroup
from dispel.processing.modalities import LimbModality, SensorModality
from dispel.processing.transform import TransformStep
from dispel.providers.generic.flags.generic import (
    FrequencyLowerThanSBTThres,
    MaxTimeIncrement,
)
from dispel.providers.generic.flags.le_flags import (
    FlagAdjustmentNonBeltSBT,
    FlagNonBeltSBT,
    FlagPostAdjustExcessiveMotion,
    FlagStabilisationExcessiveMotion,
    PlacementClassificationGroup,
)
from dispel.providers.generic.preprocessing import (
    FilterPhysiologicalNoise,
    FilterSensorNoise,
    PreprocessingSteps,
)
from dispel.providers.generic.sensor import RenameColumns, TransformUserAcceleration
from dispel.providers.generic.tasks.sbt_utt.const import (
    FIXED_SIGN_AMP_THRESHOLD,
    KERNEL_WINDOW_SEGMENT,
    MIN_COVERAGE_THRESHOLD,
    TASK_NAME_SBT,
)
from dispel.providers.generic.tasks.sbt_utt.sbt_bouts import (
    SBTBoutExtractStep,
    SBTBoutStrategyModality,
)
from dispel.providers.generic.tasks.sbt_utt.sbt_func import (
    circle_area,
    data_coverage_fraction,
    ellipse_area,
    label_bouts,
    reject_short_bouts,
    sway_jerk,
    sway_total_excursion,
)
from dispel.providers.generic.tremor import TremorFeatures
from dispel.signal.accelerometer import (
    AP_ML_COLUMN_MAPPINGS,
    transform_ap_ml_acceleration,
)
from dispel.signal.core import derive_time_data_frame, euclidean_norm


class SBTTremorFeaturesGroup(ProcessingStepGroup):
    """Tremor feature for the SBT from accelerometer."""

    def __init__(self, data_set_id, **kwargs):
        new_column_names = {
            "userAccelerationX": "x",
            "userAccelerationY": "y",
            "userAccelerationZ": "z",
        }
        steps = [
            RenameColumns(data_set_id, **new_column_names),
            TremorFeatures(
                sensor=SensorModality.ACCELEROMETER,
                data_set_id=f"{data_set_id}_renamed",
                add_norm=False,
            ),
        ]

        super().__init__(steps, **kwargs)


class TransformAxisNames(TransformStep):
    """A raw data set transformation processing step to change axis names.

    This step is done to match Martinez et al 2012.
    """

    new_data_set_id = "martinez_accelerations"
    transform_function = transform_ap_ml_acceleration

    def get_definitions(self) -> List[RawDataValueDefinition]:
        """Get the data set definitions."""
        return [
            RawDataValueDefinition(
                column, f"{column} data", data_type="float64", unit="G"
            )
            for column in AP_ML_COLUMN_MAPPINGS.values()
        ]


class TransformJerkNorm(TransformStep):
    """Transform AP/ML acceleration into jerk norm."""

    data_set_ids = TransformAxisNames.new_data_set_id  # pylint: disable=E1101
    new_data_set_id = "jerk_norm"
    definitions = [
        RawDataValueDefinition("jerk_norm", "Jerk norm (AP/ML)", data_type="float64")
    ]

    @transformation
    def jerk_norm(self, data: pd.DataFrame) -> pd.Series:
        """Compute the jerk norm."""
        return euclidean_norm(derive_time_data_frame(data[["ap", "ml"]]))


class FlagExcessiveMotion(TransformStep):
    """TransformStep that creates a flag signal to mark excessive motion.

    The flagging segments the accelerometer norm which
    can be considered an excessive motion (w.r.t, normal sway during balance,
    e.g., belt placing or re-gaining balance after tripping,
    based on the amplitude.
    Developed using the SBT_flag_scenarios.
    """

    data_set_ids = "jerk_norm"
    new_data_set_id = "flagged_motion"
    definitions = [RawDataValueDefinition("flag", "Excessive Motion Segments")]

    @staticmethod
    @transformation
    def detect_excessive_motion(
        data: pd.DataFrame,
        fixed_threshold: float = FIXED_SIGN_AMP_THRESHOLD,
        kernel_width: int = KERNEL_WINDOW_SEGMENT,
    ) -> pd.Series:
        """Flag segments of excessive motion in accelerometer jerk during sway.

        Output is a raw_dataset with 0 (not excessive motion) or 1
        (excessive motion)

        Parameters
        ----------
        data
            A data frame with the jerk_norm.
        fixed_threshold
            A threshold to cap the amplitude of the jerk that we consider
            normal to look only in that kind of data for the statistical
            threshold definition.
        kernel_width
            The width of the running window used to median filter the
            excessive_motion mask. This is to prevent spikes from appearing
            when the signal surpasses the statistical_threshold for few
            instances.

        Returns
        -------
        Series
            A Series with flagged_motion which becomes a new
            raw_dataset

        """
        # Identify which part of the magnitude of the accelerometer signal
        # is greater than a fixed threshold
        mask = data["jerk_norm"] > fixed_threshold

        # Define an adaptive threshold based on 2.5 IQR on the quiet
        # activity periods
        statistical_threshold = 2.5 * scipy_iqr(data.loc[~mask, "jerk_norm"])

        # Detect excessive motion
        mask_excessive_motion = data.jerk_norm > statistical_threshold

        # Smoothing the values removing small gap (2+ secs , 50Hz+1 * 2)
        # TODO think if I can play with decorators here (perfect composition)
        flag_smooth = pd.Series(
            medfilt(mask_excessive_motion * 1.0, kernel_size=kernel_width), data.index
        )
        return reject_short_bouts(label_bouts(flag_smooth), flag_smooth)


class TransformMergeValidSegments(TransformStep):
    """Mask the AP-ML acceleration, under the flagged_motion.

    The output contains only segments of AP-ML acceleration which were not
    identified as an excessive motion.
    Outputs the merged valid AP/ML/Vertical acceleration data.
    """

    data_set_ids = ["martinez_accelerations", "flagged_motion"]
    new_data_set_id = "concat_valid_martinez_acc"
    definitions = [
        RawDataValueDefinition("v", "Continuous valid acceleration V data merged"),
        RawDataValueDefinition("ml", "Continuous valid acceleration ML data merged"),
        RawDataValueDefinition("ap", "Continuous valid acceleration AP data merged"),
    ]

    @transformation
    def _valid_concat_data(self, data: pd.DataFrame, flag: pd.Series) -> pd.DataFrame:
        """Take only valid_continuous_data segments and concatenate them.

        Parameters
        ----------
        data
            Martinez Accelerations (AP,ML, V)
        flag
            Masking flag (1: excessive motion data, 0: valid data)

        Returns
        -------
        DataFrame
            Outputs the masked Martinez Accelerations (AP,ML, V)
        """
        # Filtering data masked by excessive motion flag
        return data.loc[~flag["flag"], ["v", "ml", "ap"]]


class ExtractValidSegmentCoverage(ExtractStep):
    """Extract segments of data without excessive motion.

    The coverage is defined as a ratio, with numerator the length of the valid
    data (not flagged as invalid) and, denominator being the length of
    the data contained originally in the recording. This value is
    standardized to a percentage (0-100) %.
    It has value between 0 and 100 %.
    """

    data_set_ids = "flagged_motion"
    definition = FeatureValueDefinitionPrototype(
        feature_name=AV("coverage valid data", "coverage_valid"),
        data_type="float64",
        unit="%",
        validator=GREATER_THAN_ZERO,
        description="The portion of data (being 100 % the complete recording "
        "length) which is valid, i.e., that does not contain "
        "a signal with excessive motion.",
    )

    transform_function = data_coverage_fraction


class FlagMinSegmentCoverage(FlagDataSetStep):
    """Flag minimum length of valid recording without excessive motion.

    Parameters
    ----------
    acceptance_threshold
        The threshold below which the data set is to be flagged. If the
        fed signal does not last for more than ``acceptance_threshold`` of the
        total recorded test length, then the associated level is flagged.
        Should be within ``[0, 100] %``.
    """

    data_set_ids = ["flagged_motion"]
    flag_name = AV("Too short sway", "sway_too_short")
    flag_type = FlagType.BEHAVIORAL
    flag_severity = FlagSeverity.DEVIATION
    reason = (
        "After removing segments with excessive motion"
        "(i.e., behavioural flag bouts) the remaining data is "
        "too short {percentage_valid} % of the total, i.e., "
        "less than {threshold} %"
    )

    acceptance_threshold: float = MIN_COVERAGE_THRESHOLD

    @flag
    def _min_valid_length_flag(
        self, data: pd.Series, acceptance_threshold: float = acceptance_threshold
    ) -> bool:
        percentage_valid = data_coverage_fraction(data) * 100

        self.set_flag_kwargs(
            threshold=acceptance_threshold,
            percentage_valid=percentage_valid,
        )
        return percentage_valid >= acceptance_threshold

    def get_flag_targets(
        self, reading: Reading, level: Optional[Level] = None, **kwargs
    ) -> Iterable[EntityType]:
        """Get flag targets."""
        assert level is not None, "Level cannot be null"
        return [level]


class ExtractSwayTotalExcursion(SBTBoutExtractStep):
    """Extract Sway Total Excursion.

    For details about the algorithm, check equation (eq.8) by Prieto(1996)
    https://doi.org/10.1109/10.532130

    Parameters
    ----------
    data_set_ids
        The data set ids that will be used to extract Sway Total Excursion
        features.
    """

    def __init__(self, data_set_ids: List[str], **kwargs):
        description = (
            "The Sway Total Excursion (a.k.a. TOTEX) computed with the eq.8 "
            "of Prieto (1996) algorithm. This feature quantifies the total "
            "amount of acceleration increments in the "
            "Anterior/Posterior-Medio/Lateral plane of the movement."
            "It is computed with the bout strategy {bout_strategy_repr}. See"
            "also https://doi.org/10.1109/10.532130."
        )

        definition = FeatureValueDefinitionPrototype(
            feature_name=AV("sway total excursion", "totex"),
            data_type="float64",
            unit="mG",
            validator=GREATER_THAN_ZERO,
            description=description,
        )

        def _compute_totex(data):
            return sway_total_excursion(data.ap, data.ml) * 1e3

        super().__init__(data_set_ids, _compute_totex, definition, **kwargs)


class ExtractSwayJerk(SBTBoutExtractStep):
    """Extract Sway Jerk.

    For details about the algorithm, check Table 2 by Mancini(2012),
    https://doi.org/10.1186/1743-0003-9-59
    It is a complementary measure to the sway areas, as it covers also the
    amount of sway occurred within a given geometric area. It takes special
    relevance when algorithms to remove outliers are applied and the timeseries
    span used for different features is different. In other words, a normalised
    version of the sway total excursion. See an example of
    concomitant use with sway total excursion in Mancini(2012)

    Parameters
    ----------
    data_set_ids
        The data set ids that will be used to extract Sway Jerk
        features.
    """

    def __init__(self, data_set_ids: List[str], **kwargs):
        description = (
            "The Sway Jerk provides a means of quantifying the average "
            "sway occurred on a plane per time unit over the total of "
            "an assessment."
            "It is computed with the bout strategy {bout_strategy_repr}. See"
            " also https://doi.org/10.1186/1743-0003-9-59."
        )

        definition = FeatureValueDefinitionPrototype(
            feature_name=AV("sway jerk", "jerk"),
            data_type="float64",
            unit="mG/s",
            validator=GREATER_THAN_ZERO,
            description=description,
        )

        def _compute_jerk(data):
            return sway_jerk(data.ap, data.ml) * 1e3

        super().__init__(data_set_ids, _compute_jerk, definition, **kwargs)


class ExtractCircleArea(SBTBoutExtractStep):
    """Extract circle Area.

    For details about the algorithm, check equation (eq.12) by Prieto(1996)
    https://doi.org/10.1109/10.532130

    Parameters
    ----------
    data_set_ids
        The data set ids that will be considered to extract Circle Area
        features.
    """

    def __init__(self, data_set_ids: List[str], **kwargs):
        description = (
            "The ellipse area computed with the eq.12 of Prieto (1996) "
            "algorithm. It is computed with the bout strategy "
            "{bout_strategy_repr}. See also "
            "https://doi.org/10.1109/10.532130."
        )

        definition = FeatureValueDefinitionPrototype(
            feature_name=AV("circle area", "ca"),
            data_type="float64",
            unit="microG^2",
            validator=GREATER_THAN_ZERO,
            description=description,
        )

        def _compute_circle_area(data):
            return circle_area(data.ap, data.ml) * 1e6

        super().__init__(data_set_ids, _compute_circle_area, definition, **kwargs)


class ExtractEllipseArea(SBTBoutExtractStep):
    """Extract Ellipse Area.

    For details about the algorithm, check equation eq.18 of Schubert(2014)
    https://doi.org/10.1016/j.gaitpost.2013.09.001)

    Parameters
    ----------
    data_set_ids
        The data set ids that will be considered to extract Ellipse Area
        features.
    """

    def __init__(self, data_set_ids: List[str], **kwargs):
        description = (
            "The ellipse area computed with the eq.18 of Schubert (2014) "
            "algorithm. It is computed with the bout strategy "
            "{bout_strategy_repr}. This is a PCA-based variation of "
            "https://doi.org/10.1016/j.gaitpost.2013.09.001."
        )

        definition = FeatureValueDefinitionPrototype(
            feature_name=AV("ellipse area", "ea"),
            data_type="float64",
            unit="microG^2",
            validator=GREATER_THAN_ZERO,
            description=description,
        )

        def _compute_ellipse_area(data):
            return ellipse_area(data.ap, data.ml) * 1e6

        super().__init__(data_set_ids, _compute_ellipse_area, definition, **kwargs)


class ExtractSpatioTemporalFeaturesBatch(ProcessingStepGroup):
    """Extract postural adjustment features given a bout strategy."""

    def __init__(self, bout_strategy: SBTBoutStrategyModality, **kwargs):
        data_set_id = ["martinez_accelerations"]
        steps: List[ProcessingStep] = [
            ExtractSwayTotalExcursion(
                data_set_id, bout_strategy=bout_strategy.bout_cls
            ),
            ExtractSwayJerk(data_set_id, bout_strategy=bout_strategy.bout_cls),
            ExtractCircleArea(data_set_id, bout_strategy=bout_strategy.bout_cls),
            ExtractEllipseArea(data_set_id, bout_strategy=bout_strategy.bout_cls),
        ]
        super().__init__(steps, **kwargs)


class TechnicalFlagsGroup(ProcessingStepGroup):
    """A ProcessingStepGroup concatenating TechnicalFlagsGroup steps."""

    steps = [
        FrequencyLowerThanSBTThres(),
        MaxTimeIncrement(),
    ]


class FormattingGroup(ProcessingStepGroup):
    """A ProcessingStepGroup concatenating Formatting steps."""

    steps = [TransformUserAcceleration()]


class PreProcessingStepsGroup(ProcessingStepGroup):
    """A ProcessingStepGroup concatenating PreProcessingStepsGroup steps."""

    steps = [
        PreprocessingSteps(
            "accelerometer",
            LimbModality.LOWER_LIMB,
            SensorModality.ACCELEROMETER,
            columns=ACCELEROMETER_COLUMNS,
        ),
        FilterSensorNoise(
            "acc_ts_rotated_resampled_detrend", columns=ACCELEROMETER_COLUMNS
        ),
        FilterPhysiologicalNoise(
            "acc_ts_rotated_resampled_detrend_svgf", columns=ACCELEROMETER_COLUMNS
        ),
    ]


class TremorAndAxesGroup(ProcessingStepGroup):
    """A ProcessingStepGroup concatenating TremorAndAxesGroup steps."""

    steps = [
        SBTTremorFeaturesGroup("acc_ts_rotated_resampled_detrend"),
        TransformAxisNames("acc_ts_rotated_resampled_detrend_svgf_bhpf"),
        TransformJerkNorm(),
    ]


class DetectExcessiveMotionGroup(ProcessingStepGroup):
    """A ProcessingStepGroup concatenating DetectExcessiveMotionGroup steps."""

    steps = [
        # Steps Group to obtain Sway Path
        FlagExcessiveMotion(),
        TransformMergeValidSegments(),
        ExtractValidSegmentCoverage(),
    ]


class ExtractSpatioTemporalFeaturesGroup(ProcessingStepGroup):
    """A ProcessingStepGroup concatenating ExtractSpatioTemporalFeaturesBatch steps."""

    steps = [
        ExtractSpatioTemporalFeaturesBatch(
            bout_strategy=bout_strategy,
            modalities=[bout_strategy.av],
            bout_strategy_repr=bout_strategy.av,
        )
        for bout_strategy in SBTBoutStrategyModality
    ]


class BehaviouralFlagsGroup(ProcessingStepGroup):
    """A ProcessingStepGroup concatenating BehaviouralFlagsGroup steps."""

    steps = [
        # Detect Placement Steps
        PlacementClassificationGroup(),
        FlagNonBeltSBT(),
        FlagAdjustmentNonBeltSBT(),
        FlagStabilisationExcessiveMotion(),
        FlagPostAdjustExcessiveMotion(),
    ]

    kwargs = {"task_name": TASK_NAME_SBT}


class SBTProcessingSteps(ProcessingStepGroup):
    """All processing steps to extract SBT features."""

    steps = [
        TechnicalFlagsGroup(),
        FormattingGroup(),
        PreProcessingStepsGroup(),
        TremorAndAxesGroup(),
        DetectExcessiveMotionGroup(),
        ExtractSpatioTemporalFeaturesGroup(),
        BehaviouralFlagsGroup(),
    ]

    level_filter = LevelIdFilter("sbt")
    kwargs = {"task_name": TASK_NAME_SBT}
