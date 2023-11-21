"""A module for phone orientation processing steps."""
from abc import ABCMeta
from typing import Iterable, List, Optional, Union

import pandas as pd

from dispel.data.core import EntityType, Reading
from dispel.data.flags import FlagSeverity, FlagType
from dispel.data.levels import Level
from dispel.data.raw import RawDataValueDefinition
from dispel.data.values import AbbreviatedValue as AV
from dispel.processing import ProcessingStep
from dispel.processing.data_set import FlagDataSetStep, transformation
from dispel.processing.flags import flag
from dispel.processing.level import ProcessingStepGroup
from dispel.processing.transform import TransformStep
from dispel.signal.orientation import (
    OrientationRange,
    PhoneOrientation,
    PhoneOrientationType,
)


class PhoneOrientationMixin(metaclass=ABCMeta):
    """Phone orientation processing step Mixin class."""

    orientation_mode: PhoneOrientationType

    def __init__(
        self,
        pitch_freedom: Union[float, OrientationRange] = 70,
        yaw_freedom: Union[float, OrientationRange] = 15,
        orientation_mode: Optional[PhoneOrientationType] = None,
        **kwargs,
    ):
        self.pitch_freedom = pitch_freedom
        self.yaw_freedom = yaw_freedom

        if orientation_mode:
            self.orientation_mode = orientation_mode

        super().__init__(**kwargs)  # type: ignore

    def get_orientation_modes(self) -> List[PhoneOrientation]:
        """Get an iterable of orientation modes."""
        if isinstance(self.orientation_mode, PhoneOrientation):
            return [self.orientation_mode]
        return list(self.orientation_mode)

    def get_classification(
        self, data: pd.DataFrame, orientation_mode: PhoneOrientation
    ) -> pd.Series:
        """Get the classification related to the provided orientation."""
        classifier = orientation_mode.get_classifier(
            pitch_freedom=self.pitch_freedom, yaw_freedom=self.yaw_freedom
        )
        return classifier(data)


class PhoneOrientationTransform(PhoneOrientationMixin, TransformStep):
    """A transform step for phone orientation classification.

    The transformation step produces a pandas data frame containing one boolean
    pandas series per given orientation mode. Tha resulted pandas series are
    point-wise classifications of the input gravity signal. The gravity data
    frame index is passed to the transformation data frame.

    Parameters
    ----------
    gravity_data_set_id
        The identifier of the data set containing the gravity signal.
        The data frame contained in the gravity data set must contain three
        main columns ``gravityX``, ``gravityY`` and ``gravityZ``.
    pitch_freedom
        The degree of freedom of the pitch angle in degrees.
    yaw_freedom
        The degree of freedom of the yaw angle in degrees.
    orientation_mode
        The phone orientation mode(s) that are to be flagged. See
        :class:`~dispel.signal.orientation.PhoneOrientation`.
    """

    new_data_set_id = "phone-orientation"

    def __init__(
        self,
        gravity_data_set_id: str,
        pitch_freedom: Union[float, OrientationRange] = 70,
        yaw_freedom: Union[float, OrientationRange] = 15,
        orientation_mode: Optional[PhoneOrientationType] = None,
        **kwargs,
    ):
        super().__init__(
            data_set_ids=gravity_data_set_id,
            pitch_freedom=pitch_freedom,
            yaw_freedom=yaw_freedom,
            orientation_mode=orientation_mode,
            **kwargs,
        )

    def get_definitions(self) -> List[RawDataValueDefinition]:
        """Get the definitions of the raw data set values."""
        return [
            RawDataValueDefinition(
                mode.variable, f"Binary classification of {mode.av} orientation."
            )
            for mode in self.get_orientation_modes()
        ]

    @transformation
    def _orientation_classifications(self, data: pd.DataFrame):
        return pd.DataFrame(
            {
                mode.variable: self.get_classification(data, mode)
                for mode in self.get_orientation_modes()
            },
            index=data.index,
        )


class PhoneOrientationFlagger(PhoneOrientationMixin, FlagDataSetStep):
    """A data set flagger for phone orientation.

    Parameters
    ----------
    gravity_data_set_id
        The identifier of the data set containing the gravity signal.
    pitch_freedom
        The degree of freedom of the pitch angle in degrees, if it is a float
        then the pitch has to be within [-pitch_freedom, pitch_freedom], else
        it should be a tuple defining the range of the pitch.
    yaw_freedom
        The degree of freedom of the yaw angle in degrees, if it is a float
        then the yaw has to be within [-yaw_freedom, yaw_freedom], else it
        should be a tuple defining the range of the yaw.
    acceptance_threshold
        The threshold below which the data set is to be flagged. If the
        fed signal does not match more than ``acceptance_threshold`` % of the
        phone orientation mode, then the associated level is flagged.
        Should be within ``[0, 1]``.
    orientation_mode
        The phone orientation mode(s) that are to be flagged. See
        :class:`~dispel.signal.orientation.PhoneOrientation`.
    """

    flag_type: Union[FlagType, str] = FlagType.BEHAVIORAL
    flag_severity: Union[FlagSeverity, str] = FlagSeverity.DEVIATION
    flag_name = AV("{orientation_mode} orientation", "{orientation_mode.abbr}o")
    reason = (
        "The phone has not been kept at a {orientation_mode} for more "
        "than {threshold}% of the test."
    )

    def __init__(
        self,
        gravity_data_set_id: str,
        pitch_freedom: Union[float, OrientationRange] = 70,
        yaw_freedom: Union[float, OrientationRange] = 15,
        acceptance_threshold: float = 0.9,
        orientation_mode: Optional[PhoneOrientationType] = None,
        **kwargs,
    ):
        assert (
            0 <= acceptance_threshold <= 1
        ), f"{acceptance_threshold=} has to be within [0, 1]."
        self.threshold = acceptance_threshold

        super().__init__(
            data_set_ids=gravity_data_set_id,
            pitch_freedom=pitch_freedom,
            yaw_freedom=yaw_freedom,
            orientation_mode=orientation_mode,
            **kwargs,
        )

    def get_merged_orientation_mode_av(self, percentages: List[float]) -> AV:
        """Get mode orientation abbreviated value."""
        assert len(percentages) == len(modes := self.get_orientation_modes())
        modes_av = [mode.av for mode in modes]
        modes_av_repr = [
            f"{mode.av} ({round(percentage * 100, 1)}%)"
            for mode, percentage in zip(modes, percentages)
        ]
        return AV(" or ".join(modes_av_repr), "".join(word.abbr for word in modes_av))

    @flag
    def _orientation_mode_flag(self, data: pd.DataFrame, level: Level) -> bool:
        percentages = []
        for orientation_mode in self.get_orientation_modes():
            classifications = self.get_classification(data, orientation_mode)
            counts = classifications.value_counts(normalize=True)
            percentages.append(0.0 if True not in counts else counts[True])

        self.set_flag_kwargs(
            threshold=round(self.threshold * 100, 1),
            orientation_mode=self.get_merged_orientation_mode_av(percentages),
            level_id=level.id,
        )
        return any(percentage >= self.threshold for percentage in percentages)

    def get_flag_targets(
        self, reading: Reading, level: Optional[Level] = None, **kwargs
    ) -> Iterable[EntityType]:
        """Get flag targets."""
        assert level is not None, "Level cannot be null"
        return [level]


class LandscapeModeFlagger(PhoneOrientationFlagger):
    """A data set flagger for landscape mode."""

    orientation_mode = (
        PhoneOrientation.LANDSCAPE_RIGHT,
        PhoneOrientation.LANDSCAPE_LEFT,
    )


class UprightPortraitModeFlagger(PhoneOrientationFlagger):
    """A data set flagger for upright portrait mode."""

    orientation_mode = PhoneOrientation.PORTRAIT_UPRIGHT


class UpperLimbOrientationFlagger(ProcessingStepGroup):
    """A group of pinch processing steps for measures by level id.

    The pitch_freedom and yaw_freedom range were defined using Gravity values
    and limited range when positioning the phone in portrait mode. The
    threshold of 0.7 take into account the fact that the user can move the
    phone between levels and then flag the orientations at the
    beginning of the level.
    """

    def __init__(self, **kwargs):
        steps: List[ProcessingStep] = [
            UprightPortraitModeFlagger(
                gravity_data_set_id="acc",
                pitch_freedom=OrientationRange(lower=-90, upper=10),
                yaw_freedom=OrientationRange(lower=-20, upper=20),
                acceptance_threshold=0.7,
                reason="The phone has not been kept at a {orientation_mode} "
                "for more than {threshold}% of the test at the level "
                "{level_id}.",
            )
        ]
        super().__init__(steps, **kwargs)


class UpsideDownPortraitModeFlagger(PhoneOrientationFlagger):
    """A data set flagger for upside down portrait mode."""

    orientation_mode = PhoneOrientation.PORTRAIT_UPSIDE_DOWN
