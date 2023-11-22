"""Pronation Supination assessment related functionality.

This module contains functionality to extract measures for the
*Pronation Supination* assessment (PS).
"""
from typing import Any, Callable, List, Tuple, Union

import numpy as np
import pandas as pd

from dispel.data.levels import Level
from dispel.data.measures import MeasureValueDefinitionPrototype
from dispel.data.raw import DEFAULT_COLUMNS, RawDataValueDefinition
from dispel.data.validators import GREATER_THAN_ZERO
from dispel.data.values import AbbreviatedValue as AV
from dispel.data.values import AVEnum
from dispel.processing.data_set import transformation
from dispel.processing.extract import AggregateRawDataSetColumn, ExtractStep
from dispel.processing.level import ProcessingStepGroup
from dispel.processing.modalities import HandModality
from dispel.processing.transform import Apply, TransformStep
from dispel.providers.generic.sensor import (
    Resample,
    SetTimestampIndex,
    TransformFindZeroCrossings,
    Upsample,
)
from dispel.providers.registry import process_factory
from dispel.signal.filter import butterworth_low_pass_filter
from dispel.stats.core import npcv

TASK_NAME = AV("Pronation-Supination Assessment", "PS")

MIN_EVENT_ROTATION = 30
r"""Define the minimal rotation of a valid event such as pronation and supination."""

ACTIVE_PHASE_DUR = pd.Timedelta("7s")
r"""Define the duration of the active test phase of the pronation supination test."""


PS_AGGREGATION: List[Tuple[Union[Callable[[Any], float], str], str]] = [
    ("mean", "mean"),
    ("median", "median"),
    (npcv, "non parametric coefficient of variation"),
]


class Movement(AVEnum):
    """Enumerated constant representing the event modalities."""

    PRONATION = ("pronation", "pro")
    SUPINATION = ("supination", "sup")
    PROSUP = ("pronation_supination", "prosup")


class TransformIntegrateBetweenZeros(TransformStep):
    """Integrate gyroscope signal between zero crossings."""

    new_data_set_id = "ps_event"
    definitions: List[RawDataValueDefinition] = [
        RawDataValueDefinition(
            id_="zero_crossings",
            name="zero crossings",
            description="Zero-crossings of the gyroscope signal.",
            data_type="float",
        ),
        RawDataValueDefinition(
            id_="rotation",
            name="rotation",
            description="Rotation in degrees estimated by integrating the gyroscope "
            "signal between two consecutive zero-crossings.",
            data_type="float",
            unit="deg",
        ),
        RawDataValueDefinition(
            id_="abs_rotation",
            name="absolute rotation",
            description="Absolute rotation of the gyroscope between two zero crossings "
            "in degrees.",
            data_type="float",
            unit="deg",
        ),
    ]

    @staticmethod
    @transformation
    def integrate_between_zeros(
        data: pd.DataFrame, zeros: pd.DataFrame
    ) -> pd.DataFrame:
        """Integrate gyroscope between zero-crossings."""
        integrals_ = []
        ps_event = zeros.copy()
        for i in range(len(zeros) - 1):
            i_0 = zeros.index[i]
            i_1 = zeros.index[i + 1]
            integrals_.append(
                np.trapz(
                    data[i_0:i_1].z.values,
                    (data[i_0:i_1].index - data.index[0]).total_seconds(),
                )
            )
        ps_event["rotation"] = integrals_ + [None]
        ps_event["rotation"] = np.rad2deg(ps_event["rotation"])
        ps_event["abs_rotation"] = ps_event.rotation.abs()
        return ps_event


class TransformIdentifyEvent(TransformStep):
    """Identify pronation supination given the rotation and the hand used.

    Only pronation event and supination event for which the associated rotation is
    greater than MIN_EVENT_ROTATION in absolute value are considered.
    """

    new_data_set_id = "ps_event"
    definitions = TransformIntegrateBetweenZeros.definitions + [
        RawDataValueDefinition(
            "event",
            "Event indicating if the motion was a pronation or supination.",
            data_type="str",
        )
    ]

    @staticmethod
    @transformation
    def identify_event(ps_event: pd.DataFrame, level: Level) -> pd.DataFrame:
        """Identify whether the event is a supination or a pronation."""
        level_id = str(level.id)
        ps_event["event"] = None
        neg_mask = ps_event.rotation < -MIN_EVENT_ROTATION
        pos_mask = ps_event.rotation > MIN_EVENT_ROTATION
        if level_id == "right":
            ps_event.loc[neg_mask, "event"] = "supination"
            ps_event.loc[pos_mask, "event"] = "pronation"
        elif level_id == "left":
            ps_event.loc[neg_mask, "event"] = "pronation"
            ps_event.loc[pos_mask, "event"] = "supination"
        else:
            raise ValueError(f"level_id: {level_id} is not in ['left', 'right']")
        return ps_event


class TransformFilterEvent(TransformStep):
    """Enrich event with time boundaries and only keep defined event."""

    new_data_set_id = "ps_event"
    definitions = TransformIdentifyEvent.definitions + [
        RawDataValueDefinition(
            "start",
            "Start timestamp indicating the beginning of the event.",
            data_type="datetime64[ns]",
        ),
        RawDataValueDefinition(
            "end",
            "end timestamp indicating the end of the event.",
            data_type="datetime64[ns]",
        ),
    ]

    @staticmethod
    @transformation
    def filter_event(ps_event: pd.DataFrame) -> pd.DataFrame:
        """Add start and end of event and keep only defined events."""
        event_start = ps_event.index[:-1]
        event_end = ps_event.index[1:]
        # Remove last un-needed zero
        ps_event = ps_event[:-1]
        ps_event["start"] = event_start
        ps_event["end"] = event_end
        return ps_event.dropna()


class TransformRemovePriorEvent(TransformStep):
    """Remove event ending before active test phase."""

    new_data_set_id = "ps_event"
    definitions = TransformFilterEvent.definitions

    @staticmethod
    @transformation
    def truncate_active_event(ps_event: pd.DataFrame, level: Level) -> pd.DataFrame:
        """Remove potential event ending prior the active phase of 7 s."""
        return ps_event.loc[ps_event.end > level.end - ACTIVE_PHASE_DUR]


class TransformRotationSpeed(TransformStep):
    """Compute rotation speed per motion."""

    new_data_set_id = "ps_event"
    definitions = TransformFilterEvent.definitions + [
        RawDataValueDefinition(
            "delta_t", "Duration of the motion.", data_type="str", unit="s"
        ),
        RawDataValueDefinition(
            "rotation_speed",
            "Average rotation speed in degrees during the motion.",
            data_type="str",
            unit="deg",
        ),
        RawDataValueDefinition(
            "abs_rotation_speed",
            "Absolute value of the average rotation speed in degrees.",
            data_type="str",
            unit="deg",
        ),
    ]

    @staticmethod
    @transformation
    def compute_rot_speed(data: pd.DataFrame) -> pd.DataFrame:
        """Compute average rotation speed during a pronation or supination."""
        # Time between two zero crossing is obtained by differentiating the
        # timestamp of the index.
        data["delta_t"] = (data.end - data.start).dt.total_seconds()
        data["rotation_speed"] = data.rotation / (data["delta_t"])
        data["abs_rotation_speed"] = data.rotation_speed.abs()
        return data


class TransformAmplitudePeakToPeak(TransformStep):
    """Compute amplitude of rotation peak to peak."""

    new_data_set_id = "ps_event_peak_peak"
    definitions: List[RawDataValueDefinition] = [
        RawDataValueDefinition(
            "start",
            "Start timestamp indicating the beginning of the event.",
            data_type="datetime64[ns]",
        ),
        RawDataValueDefinition(
            "end",
            "end timestamp indicating the end of the event.",
            data_type="datetime64[ns]",
        ),
        RawDataValueDefinition(
            "delta_t", "Duration of the motion.", data_type="str", unit="s"
        ),
        RawDataValueDefinition(
            "abs_rotation",
            "Absolute rotation for a pronation supination in degrees.",
            data_type="float",
            unit="deg",
        ),
    ]

    @staticmethod
    @transformation
    def compute_abs_rot_peak_peak(data: pd.DataFrame) -> pd.DataFrame:
        """Compute absolute rotation peak to peak of pronation - supination."""
        amplitude_gp = data.iloc[1:-1].groupby(data.iloc[1:-1].reset_index().index // 2)
        amplitude = amplitude_gp[["abs_rotation", "event", "delta_t"]].sum()
        amplitude["start"] = amplitude_gp["start"].min()
        amplitude["end"] = amplitude_gp["end"].max()
        amplitude["count"] = amplitude_gp["start"].count()
        amplitude = amplitude.loc[amplitude["count"] == 2]
        return amplitude.drop(columns="count")


class AggregateMotion(AggregateRawDataSetColumn):
    """An Aggregation of motion measures."""

    def get_data_frames(self, level: Level) -> List[pd.DataFrame]:
        """Get the raw data from all data sets in question.

        Parameters
        ----------
        level
            The level from which to get the data sets.

        Returns
        -------
        List[pandas.DataFrame]
            A list of all raw data frames with the specified ids masked with
            event being equal to the movement specified.
        """
        if self.movement.variable == "prosup":
            return list(
                map(
                    lambda r: r.data,
                    self.get_raw_data_sets(level),
                )
            )
        return list(
            map(
                lambda r: r.data[r.data["event"] == self.movement.variable],
                self.get_raw_data_sets(level),
            )
        )

    def __init__(self, data_set_id: str, column_id: str, movement: Movement, **kwargs):
        self.movement = movement
        description = f"The {{aggregation}} of {column_id} for {movement}."
        definition = MeasureValueDefinitionPrototype(
            measure_name=AV(
                f"{movement.variable} {column_id.replace('_', ' ')}", f"{column_id}"
            ),
            data_type="float64",
            description=description,
        )
        super().__init__(data_set_id, column_id, PS_AGGREGATION, definition, **kwargs)


class AggregateAmplitude(AggregateRawDataSetColumn):
    """An aggregation processing step for the amplitude peak to peak."""

    data_set_ids = "ps_event_peak_peak"
    column_id = "abs_rotation"
    aggregations = PS_AGGREGATION
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("amplitude", "amp"),
        description="The {aggregation} amplitude of rotation of an"
        "entire pronation supination cycle.",
        unit="deg",
        data_type="float",
    )


class AggregateDuration(AggregateRawDataSetColumn):
    """An aggregation processing step for the duration peak to peak."""

    data_set_ids = "ps_event_peak_peak"
    column_id = "delta_t"
    aggregations = [("mean", "mean")]
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("duration", "dur"),
        description="The {aggregation} duration of an"
        "entire pronation supination cycle.",
        unit="s",
        data_type="float",
    )


class ExtractNEvent(ExtractStep):
    """Extract number of events that are either pronation or supination."""

    data_set_ids = "ps_event"
    description = "The number of events that are a pronation or a supination."
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("event count", "ec"),
        data_type="int16",
        validator=GREATER_THAN_ZERO,
        description=description,
        task_name=TASK_NAME,
    )

    @staticmethod
    @transformation
    def count_events(data: pd.DataFrame) -> int:
        """Count the number of pronation or supination."""
        return data.event.isin({"pronation", "supination"}).sum()


class ExtractAvgMovementPowerFromTimeSeries(ExtractStep):
    """Extract average movement power from a time series."""

    description = (
        "The average power of the signal in the defined 0 - 4 Hz frequency band."
    )
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("movement power mean", "ts_mvmt_power-mean"),
        data_type="float",
        description=description,
        task_name=TASK_NAME,
    )

    @staticmethod
    @transformation
    def average_power(data: pd.DataFrame) -> int:
        """Compute the average power of the time series."""
        # integrate the power in the appropriate frequency
        return (data**2).mean().squeeze()


def decrement_simple(data: pd.Series) -> float:
    """Compute simplified version of decrement of a series."""
    first_quarter = data[: (quarter_idx := len(data) // 4)]
    second_half = data[-quarter_idx:]
    return np.median(first_quarter) - np.median(second_half)


class ExtractAmplitudeDecrementSimple(ExtractStep):
    """Extract the simple version of decrement in amplitude."""

    data_set_ids = "ps_event_peak_peak"
    description = (
        "The simple decrement of amplitude defined as the difference in "
        "median amplitude between the first and last 25% of the task."
    )
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("amplitude simple decrement", "amp_simple_dec"),
        data_type="float",
        description=description,
        task_name=TASK_NAME,
    )

    @staticmethod
    @transformation
    def decrement_amplitude_simple(data: pd.DataFrame) -> float:
        """Compute the simple decrement of amplitude."""
        return decrement_simple(data["abs_rotation"])


class ExtractSpeedDecrementSimple(ExtractStep):
    """Extract the simple version of decrement in speed."""

    data_set_ids = "ps_event"
    description = (
        "The simple decrement of speed defined as the difference in "
        "median speed between the first and last 25% of the task."
    )
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("speed simple decrement", "speed_simple_dec"),
        data_type="float",
        description=description,
        task_name=TASK_NAME,
    )

    @staticmethod
    @transformation
    def decrement_speed_simple(data: pd.DataFrame) -> float:
        """Compute the simple decrement of speed."""
        return decrement_simple(data["abs_rotation"])


COL_TO_AGG: List[str] = ["abs_rotation", "abs_rotation_speed"]
r"""Columns of the event data set that we want to aggregate as measures over
the different events."""


class PreprocessingPSGroup(ProcessingStepGroup):
    """Pronation Supination preprocessing steps."""

    steps = [
        SetTimestampIndex("gyroscope", DEFAULT_COLUMNS, duplicates="first"),
        # Resample with mean then ffill to remove eventual remaining nans
        Resample(
            data_set_id="gyroscope_ts",
            freq=50,
            aggregations=["mean", "ffill"],
            columns=DEFAULT_COLUMNS,
        ),
        Apply(
            "gyroscope_ts_resampled",
            butterworth_low_pass_filter,
            dict(order=3, cutoff=10, zero_phase=True),
            ["z"],
        ),
    ]


class MovementPowerGroup(ProcessingStepGroup):
    """Pronation Supination movement power processing steps group."""

    steps = [
        Apply(
            data_set_id="gyroscope_ts_resampled",
            new_data_set_id="gyroscope_ts_resampled_low_pass_4hz",
            method=butterworth_low_pass_filter,
            method_kwargs=dict(order=3, cutoff=4, zero_phase=True),
            columns=["z"],
        ),
        ExtractAvgMovementPowerFromTimeSeries(
            data_set_ids="gyroscope_ts_resampled_low_pass_4hz"
        ),
    ]


class EventDetectionAndFiltering(ProcessingStepGroup):
    """Pronation Supination event detection and filtering processing steps group."""

    steps = [
        Upsample(
            interpolation_method="linear",
            freq=1000,
            data_set_id="gyroscope_ts_resampled_butterworth_low_pass_filter",
            columns=["z"],
        ),
        TransformFindZeroCrossings(
            data_set_ids="gyroscope_ts_resampled_butterworth_low_pass_filter_upsampled",
            column="z",
            new_data_set_id="zero_crossings",
        ),
        TransformIntegrateBetweenZeros(
            data_set_ids=[
                "gyroscope_ts_resampled_butterworth_low_pass_filter_upsampled",
                "zero_crossings",
            ]
        ),
        TransformIdentifyEvent("ps_event", storage_error="overwrite"),
        TransformFilterEvent("ps_event", storage_error="overwrite"),
        TransformRemovePriorEvent("ps_event", storage_error="overwrite"),
    ]


class AggregateAmplitudeAndDurationGroup(ProcessingStepGroup):
    """Aggregate amplitude and duration group."""

    steps = [
        ProcessingStepGroup(
            steps=[AggregateAmplitude(), AggregateDuration()],
            modalities=[hand.av],
            level_filter=hand.abbr,
        )
        for hand in HandModality
    ]


class AggregatePerHandAndMotionGroup(ProcessingStepGroup):
    """Aggregate per hand and motion group."""

    steps = [
        ProcessingStepGroup(
            steps=[
                AggregateMotion("ps_event", column, movement) for column in COL_TO_AGG
            ],
            modalities=[hand.av, movement.av],
            level_filter=hand.abbr,
        )
        for movement in Movement
        for hand in HandModality
    ]


class PerHandEventExtractionGroup(ProcessingStepGroup):
    """Pronation Supination per hand measure extraction group."""

    steps = [
        ProcessingStepGroup(
            [
                ExtractNEvent(),
                ExtractAmplitudeDecrementSimple(),
                ExtractSpeedDecrementSimple(),
            ],
            task_name=TASK_NAME,
            modalities=[hand.av],
            level_filter=hand.abbr,
        )
        for hand in HandModality
    ]


class EventMeasureExtractionGroup(ProcessingStepGroup):
    """Pronation Supination event-based measure extraction group."""

    steps = [
        TransformRotationSpeed("ps_event", storage_error="overwrite"),
        # Create a dataset for peak to peak
        TransformAmplitudePeakToPeak("ps_event"),
        # Aggregate amplitude and duration per hand
        AggregateAmplitudeAndDurationGroup(),
        # Aggregate per hand and Motion
        AggregatePerHandAndMotionGroup(),
        # Aggregate per hand
        PerHandEventExtractionGroup(),
    ]


class BDHPronationSupinationSteps(ProcessingStepGroup):
    """Generic Pronation Supination processing steps."""

    steps = [
        PreprocessingPSGroup(),
        MovementPowerGroup(),
        EventDetectionAndFiltering(),
        EventMeasureExtractionGroup(),
    ]
    kwargs = {"task_name": TASK_NAME}


process_ps = process_factory(
    task_name=TASK_NAME,
    steps=BDHPronationSupinationSteps(),
    codes="sp-activity",
)
