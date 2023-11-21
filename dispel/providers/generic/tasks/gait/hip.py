"""A module specific to measures related to hip rotation."""
from typing import List, Optional

import numpy as np
import pandas as pd

from dispel.data.measures import MeasureValueDefinitionPrototype
from dispel.data.raw import RawDataValueDefinition
from dispel.data.validators import GREATER_THAN_ZERO
from dispel.data.values import AbbreviatedValue as AV
from dispel.processing.core import CoreProcessingStepGroup, ProcessingStep
from dispel.processing.extract import DEFAULT_AGGREGATIONS, AggregateRawDataSetColumn
from dispel.processing.transform import TransformStep
from dispel.providers.generic.tasks.gait.core import GaitBoutAggregateStep, StepEvent
from dispel.signal.core import integrate_time_series

HIP_SIGNS = ["positive", "negative"]
r"""Define the possible hip rotation sign."""


def compute_hip_rotation(
    rotation_speed: pd.DataFrame, step_detection: pd.DataFrame, on_walking_bouts: bool
) -> pd.DataFrame:
    """Compute hip rotation between consecutive steps.

    Parameters
    ----------
    rotation_speed
        A series of angular velocity, typically the gravity-rotated,
        vertical x-axis.
    step_detection
        The data frame that contains the step detection generic data set.
    on_walking_bouts
        A boolean indicating if the computation should be made on walking
        bouts only.

    Returns
    -------
    pandas.DataFrame
        A data frame with a column ``hip_rotation`` indicating the hip rotation
        computed between consecutive steps.
    """
    # find the time of initial contacts
    ic_mask = step_detection["event"] == StepEvent.INITIAL_CONTACT

    time_contact = step_detection.index[ic_mask]
    if len(time_contact) < 1:
        return pd.DataFrame(columns=["hip_rotation"])

    # Initialize hip rotation
    hip_rotation: List[Optional[np.ndarray]] = [None] * len(time_contact)
    t_2 = time_contact[0]

    # Compute hip rotation between steps
    for k in range(len(time_contact) - 1):
        t_1 = time_contact[k]
        t_2 = time_contact[k + 1]
        hip_rotation[k] = integrate_time_series(rotation_speed[t_1:t_2])

    # Check rotation speed has more than one element
    # before trying to integrate.
    if len(rotation_speed[t_2:]) > 1:
        hip_rotation[-1] = integrate_time_series(rotation_speed[t_2:])

    # Convert results to a data frame
    res = pd.DataFrame({"hip_rotation": hip_rotation}).set_index(time_contact)

    if on_walking_bouts:
        # TODO Change this part there is no more detected walking
        diff_mask = step_detection.loc[ic_mask, "bout_id"].diff()
        diff_mask.iloc[0] = False
        walking_mask = ~diff_mask.astype(bool)
        return res[walking_mask]
    return res


class TransformHipRotation(TransformStep):
    """A transform step to compute hip rotation.

    The Hip rotation is computed as the integration of the angular velocity
    along the rotated, vertical x-axis.

    Parameters
    ----------
    rotation_speed_id
        The raw data set that contains the angular velocity series.
    step_detection_id
        The raw data set that contains the step detection generic data set.
        e.g.: ``lee``.
    component
        The component defining the rotation.
    on_walking_bouts
        A boolean indicating if the computation should be made on walking
        bouts only.
    """

    def __init__(
        self,
        rotation_speed_id: str,
        step_detection_id: str,
        component: str,
        on_walking_bouts: bool,
    ):
        if on_walking_bouts:
            step_detector = step_detection_id.split("_with_walking_bouts")[0]
        else:
            step_detector = step_detection_id

        def _compute_hip_rotation(rotation_speed, step_detection):
            # select specific component
            rot = rotation_speed[component].dropna()
            return compute_hip_rotation(rot, step_detection, on_walking_bouts)

        new_data_set_id = f"hip_rotation_{step_detection_id}"

        super().__init__(
            data_set_ids=[rotation_speed_id, step_detection_id],
            transform_function=_compute_hip_rotation,
            new_data_set_id=new_data_set_id,
            definitions=[
                RawDataValueDefinition(
                    id_="hip_rotation",
                    name="Hip Rotation",
                    unit="rad",
                    description="The hip rotation between consecutive "
                    f"steps detected with {step_detector}. "
                    "The hip rotation is computed by integrating "
                    "the rotation speed along the vertical axis "
                    "between consecutive steps. "
                    "The integration is done using the composite "
                    "trapezoidal rule.",
                    data_type="float64",
                )
            ],
        )


class SignHipRotation(TransformStep):
    """
    Separate positive and negative Hip Rotation, keep absolute value.

    Parameters
    ----------
    step_detection_id
        The raw data set that contains the step detection generic data set.
        e.g.: ``lee``.
    hip_rotation_sign
        A string indicating the sign of the hip rotation we want to keep either
        positive or negative.
    on_walking_bouts
        A boolean indicating if the computation should be made on walking
        bouts only.
    """

    def __init__(
        self, step_detection_id: str, hip_rotation_sign: str, on_walking_bouts: bool
    ):
        if on_walking_bouts:
            step_detector = step_detection_id.split("_with_walking_bouts")[0]
        else:
            step_detector = step_detection_id

        def _transform_function(data: pd.DataFrame) -> pd.DataFrame:
            if hip_rotation_sign == "positive":
                return data[data["hip_rotation"] > 0]
            return data[data["hip_rotation"] <= 0].abs()

        new_data_set_id = f"hip_rotation_{hip_rotation_sign}_{step_detection_id}"

        super().__init__(
            data_set_ids=f"hip_rotation_{step_detection_id}",
            transform_function=_transform_function,
            new_data_set_id=new_data_set_id,
            definitions=[
                RawDataValueDefinition(
                    id_="hip_rotation",
                    name="Hip Rotation",
                    unit="rad",
                    description=f"The absolute {hip_rotation_sign} hip "
                    "rotation between two steps detected with "
                    f"{step_detector} algorithm. "
                    "See :class:`~TransformHipRotation`",
                    data_type="float64",
                )
            ],
        )


class AggHipRotation(GaitBoutAggregateStep):
    """Extract Hip Rotation related measures.

    Parameters
    ----------
    step_detection_id
        The raw data set that contains the step detection generic data set.
        e.g.: ``lee``.
    hip_rotation_sign
        A string indicating the sign of the hip rotation we want to keep either
        positive or negative.
    """

    def __init__(self, step_detection_id: str, hip_rotation_sign: str, **kwargs):
        step_detector = step_detection_id.split("_")[0]
        data_set_id = f"hip_rotation_{hip_rotation_sign}_{step_detection_id}"
        _name = f"{hip_rotation_sign} hip rotation"
        _id = f"hr_{hip_rotation_sign[:3]}"
        description = (
            "The {aggregation} hip rotation between two steps "
            f"for {hip_rotation_sign} hip rotation detected "
            f"with {step_detector}. It is computed with the bout "
            "bout strategy {bout_strategy_repr}."
        )

        definition = MeasureValueDefinitionPrototype(
            measure_name=AV(_name, _id),
            data_type="float64",
            unit="rad",
            description=description,
            validator=GREATER_THAN_ZERO,
        )
        super().__init__(
            ["movement_bouts", data_set_id],
            "hip_rotation",
            DEFAULT_AGGREGATIONS,
            definition,
            **kwargs,
        )


class AggHipRotationWithoutBout(AggregateRawDataSetColumn):
    """Extract Hip Rotation related measures.

    Parameters
    ----------
    step_detection_id
        The raw data set that contains the step detection generic data set.
        e.g.: ``lee``.
    hip_rotation_sign
        A string indicating the sign of the hip rotation we want to keep either
        positive or negative.
    """

    def __init__(self, step_detection_id: str, hip_rotation_sign: str, **kwargs):
        step_detector = step_detection_id.split("_")[0]
        data_set_id = f"hip_rotation_{hip_rotation_sign}_{step_detection_id}"
        _name = f"{hip_rotation_sign} hip rotation"
        _id = f"hr_{hip_rotation_sign[:3]}"
        description = (
            "The {aggregation} hip rotation between two steps "
            f"for {hip_rotation_sign} hip rotation detected "
            f"with {step_detector}. It is computed without the "
            "walking bouts."
        )

        definition = MeasureValueDefinitionPrototype(
            measure_name=AV(_name, _id),
            data_type="float64",
            unit="rad",
            description=description,
            validator=GREATER_THAN_ZERO,
        )
        super().__init__(
            data_set_id, "hip_rotation", DEFAULT_AGGREGATIONS, definition, **kwargs
        )


class HipRotationGroup(CoreProcessingStepGroup):
    """Group processing steps for Hip Rotation transforms.

    Parameters
    ----------
    step_detection_id
        The raw data set that contains the step detection generic data set.
        e.g.: ``lee``.
    on_walking_bouts
        A boolean indicating if the computation should be made on walking
        bouts only.
    kwargs
        Additional named arguments are passed to the
        :meth:`~dispel.processing.core.ProcessingStep.process` function of each
        step.
    """

    def __init__(self, step_detection_id: str, on_walking_bouts: bool, **kwargs):
        steps: List[ProcessingStep] = [
            # Transform Hip Rotation with lee step detection
            TransformHipRotation(
                rotation_speed_id="gyroscope_ts_rotated_resampled_"
                "butterworth_low_pass_filter",
                step_detection_id=step_detection_id,
                component="x",
                on_walking_bouts=on_walking_bouts,
            )
        ]
        for hip_sign in HIP_SIGNS:
            steps.extend(
                [
                    SignHipRotation(
                        step_detection_id=step_detection_id,
                        hip_rotation_sign=hip_sign,
                        on_walking_bouts=on_walking_bouts,
                    )
                ]
            )

        super().__init__(steps, **kwargs)


class ExtractHipRotation(CoreProcessingStepGroup):
    """Group processing steps for Hip Rotation measures.

    Parameters
    ----------
    step_detection_id
        The raw data set that contains the step detection generic data set.
        e.g.: ``lee``.
    on_walking_bouts
        A boolean indicating if the computation should be made on walking
        bouts only.
    kwargs
        Additional named arguments are passed to the
        :meth:`~dispel.processing.core.ProcessingStep.process` function of each
        step.
    """

    def __init__(self, data_set_id: str, **kwargs):
        steps: List[ProcessingStep] = []
        for hip_sign in HIP_SIGNS:
            steps.extend([AggHipRotation(data_set_id, hip_sign, **kwargs)])
        super().__init__(steps, **kwargs)


class ExtractHipRotationWithoutBouts(CoreProcessingStepGroup):
    """Group processing steps for Hip Rotation measures without walking bouts.

    Parameters
    ----------
    step_detection_id
        The raw data set that contains the step detection generic data set.
        e.g.: ``lee``.
    on_walking_bouts
        A boolean indicating if the computation should be made on walking
        bouts only.
    kwargs
        Additional named arguments are passed to the
        :meth:`~dispel.processing.core.ProcessingStep.process` function of each
        step.
    """

    def __init__(self, data_set_id: str, **kwargs):
        steps: List[ProcessingStep] = []
        for hip_sign in HIP_SIGNS:
            steps.extend([AggHipRotationWithoutBout(data_set_id, hip_sign, **kwargs)])
        super().__init__(steps, **kwargs)
