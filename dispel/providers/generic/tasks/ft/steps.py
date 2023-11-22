"""Finger tapping assessment related functionality.

This module contains functionality to extract measures for the
*Finger tapping* assessment.
"""
# pylint: disable=cell-var-from-loop
from typing import List, Optional

import pandas as pd

from dispel.data.measures import MeasureValueDefinition, MeasureValueDefinitionPrototype
from dispel.data.raw import RawDataValueDefinition
from dispel.data.validators import GREATER_THAN_ZERO
from dispel.data.values import AbbreviatedValue
from dispel.data.values import AbbreviatedValue as AV
from dispel.processing import ProcessingStep
from dispel.processing.data_set import transformation
from dispel.processing.extract import (
    DEFAULT_AGGREGATIONS_IQR,
    AggregateModalities,
    AggregateRawDataSetColumn,
    ExtractStep,
)
from dispel.processing.level import ProcessingStepGroup
from dispel.processing.transform import TransformStep
from dispel.providers.generic.tasks.ft.const import (
    ENRICHED_TOUCH_ATTRIBUTES,
    TASK_NAME,
    AllHandsModalities,
    TappingTarget,
)
from dispel.providers.generic.tasks.ft.flags import GenericFlagsStepGroup


class TransformInvalidTaps(TransformStep):
    """Remove the invalid taps.

    The first tap is always valid if it is inside one of the two zones.
    A left tap is valid if it follows a right tap and vice versa.
    If a tap event occurs twice or more, then we keep the first event.
    Generally speaking, this class returns the in-target consecutive tap events
    being different.
    """

    data_set_ids = "enriched_tap_events_ts"
    new_data_set_id = "valid_enriched_tap_events_ts"
    definitions = [
        RawDataValueDefinition(column, column)
        for column in ["end", "first_position"] + ENRICHED_TOUCH_ATTRIBUTES
    ]

    @staticmethod
    @transformation
    def filter_invalid_taps(data: pd.DataFrame) -> pd.DataFrame:
        """Return a filtered version of the tap dataset."""
        # Remove the 'none' (i.e the taps outside the two zones)
        # pylint: disable=no-member
        in_target_events = data.mask(
            data["location"] == TappingTarget.OUTSIDE.abbr
        ).dropna()

        # Filter the consecutive taps on the same zone, and keep the first one
        valid_events = in_target_events.loc[
            in_target_events.location.shift() != in_target_events.location
        ]

        valid_events["end"] = valid_events.index + pd.to_timedelta(
            valid_events["tap_duration"], unit="ms"
        )
        return valid_events


class ExtractValidTaps(ExtractStep):
    """Count the number of valid taps."""

    data_set_ids = "valid_enriched_tap_events_ts"

    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("valid tap events", "valtap"),
        data_type="int16",
        validator=GREATER_THAN_ZERO,
        description="The number of valid tap events on {target} target.",
        task_name=TASK_NAME,
    )

    def __init__(self, target: Optional[TappingTarget], *args, **kwargs):
        self.target = target
        super().__init__(*args, **kwargs)

    @transformation
    def count_valid_taps(self, data):
        """Count the number of valid taps."""
        if self.target is not None:
            return data.location.eq(self.target.abbr).agg("sum")
        return data.location.agg("count")


class ExtractTotalTaps(ExtractStep):
    """Count the number of valid taps."""

    data_set_ids = "enriched_tap_events_ts"

    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("total tap events", "total_tap"),
        data_type="int16",
        validator=GREATER_THAN_ZERO,
        description="The total number of tap events.",
        task_name=TASK_NAME,
    )

    @transformation
    def count_valid_taps(self, data):
        """Count the number of valid taps."""
        return data.shape[0]


class AggregateDoubleTaps(ExtractStep):
    """Count the number of valid taps."""

    data_set_ids = "valid_enriched_tap_events_ts"

    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("double tap percentage", "double_tap_percentage"),
        data_type="int16",
        description="The percentage of time spent double tapping",
        task_name=TASK_NAME,
    )

    @transformation
    def quantify_double_taps(self, df):
        """Detect double taps."""
        data = df.copy()
        data["delta_overshoot"] = (
            data["end"].shift() - data.index.to_series()
        ).dt.total_seconds()
        double_tap_total_time = (
            data[data["delta_overshoot"] > 0]["delta_overshoot"].sum() * 1e3
        )
        total_time = data["tap_duration"].sum()
        return round((double_tap_total_time * 100) / total_time, 2)


class TransformTapIntervalDistribution(TransformStep):
    """Extract the valid or all tap interval distribution."""

    def __init__(self, valid_taps=True, **kwargs):
        self.valid_taps = valid_taps
        is_valid_prefix = "valid_" if self.valid_taps else ""
        super().__init__(
            data_set_ids=f"{is_valid_prefix}enriched_tap_events_ts",
            new_data_set_id=f"{is_valid_prefix}tap_interval_distribution",
            definitions=[
                RawDataValueDefinition(
                    "interval_with_previous_tap",
                    "Interval with previous tap",
                    data_type="float",
                )
            ],
            **kwargs,
        )

    @staticmethod
    @transformation
    def extract_tap_distribution(data: pd.DataFrame) -> pd.DataFrame:
        """Return a filtered version of the tap dataset."""
        # Remove the 'none' (i.e the taps outside the two zones)

        return (data.index - data.index.to_series().shift()).dt.total_seconds()


class AggregateTapInterval(AggregateRawDataSetColumn):
    """An extraction processing step to extract valid tap interval measures."""

    def __init__(self, valid_taps=True, **kwargs):
        self.valid_taps = valid_taps
        is_valid_prefix = "valid_" if self.valid_taps else ""
        super().__init__(
            f"{is_valid_prefix}tap_interval_distribution",
            "interval_with_previous_tap",
            aggregations=DEFAULT_AGGREGATIONS_IQR,
            definition=MeasureValueDefinitionPrototype(
                measure_name=AbbreviatedValue(
                    "tap interval", f"{is_valid_prefix}tap_inter"
                ),
                description="The {aggregation}"
                + f"of the {is_valid_prefix.replace('_', '')} tap interval distribution for the hand",
                unit="s",
                data_type="float",
            ),
            **kwargs,
        )


class AggregateTaps(AggregateModalities):
    """Compute the patient score for the FT test."""

    definition = MeasureValueDefinition(
        task_name=TASK_NAME,
        measure_name=AV("valid tap events", "valtap"),
        description="The total number of valid tap events during the test",
    )
    modalities = [[hand.av] for hand in AllHandsModalities]
    aggregation_method = sum


class PreprocessingStepGroup(ProcessingStepGroup):
    """Generic preprocessing steps for finger tapping."""

    steps = [
        TransformInvalidTaps(),
        TransformTapIntervalDistribution(valid_taps=False),
        TransformTapIntervalDistribution(),
    ]


# pylint: disable=no-member
class MeasureExtractionStepGroup(ProcessingStepGroup):
    """Generic measure extraction steps for finger tapping."""

    steps = [
        *[
            ProcessingStepGroup(
                [ExtractValidTaps(target)],
                task_name=TASK_NAME,
                modalities=[hand.av, str(target.av)],
                target=target.av,
                level_filter=hand.abbr,
            )
            for target in (TappingTarget.LEFT, TappingTarget.RIGHT)
            for hand in AllHandsModalities
        ],
        *[
            ProcessingStepGroup(
                [
                    ExtractValidTaps(target=None),
                    AggregateTapInterval(valid_taps=False),
                    AggregateTapInterval(),
                    ExtractTotalTaps(),
                    AggregateDoubleTaps(),
                ],
                task_name=TASK_NAME,
                modalities=[hand.av],
                level_filter=hand.abbr,
                target=None,
            )
            for hand in AllHandsModalities
        ],
    ]


class MeasureAggregationStepGroup(ProcessingStepGroup):
    """Generic measure aggregation step group for finger tapping."""

    steps = [
        # Compute finger tapping patient score from the two levels
        AggregateTaps()
    ]


class GenericFingerTappingSteps(ProcessingStepGroup):
    """Generic measure aggregation step group for finger tapping."""

    steps: List[ProcessingStep] = [
        PreprocessingStepGroup(),
        MeasureExtractionStepGroup(),
        MeasureAggregationStepGroup(),
        GenericFlagsStepGroup(),
    ]
    kwargs = {"task_name": TASK_NAME}
