"""Passive testing related functionality.

This module contains functionality to extract features for *Passive Testing*.
"""
from typing import List

import numpy as np
import pandas as pd

from dispel.data.features import FeatureValueDefinition, FeatureValueDefinitionPrototype
from dispel.data.validators import GREATER_THAN_ZERO
from dispel.data.values import AbbreviatedValue as AV
from dispel.processing import ProcessingStep
from dispel.processing.extract import (
    DEFAULT_AGGREGATIONS_Q95,
    AggregateRawDataSetColumn,
    ExtractStep,
    agg_column,
)
from dispel.processing.level import ProcessingStepGroup
from dispel.providers.ads.data import ADSReading
from dispel.providers.registry import process_factory

TASK_NAME = AV("Passive test", "PASSIVE")


class ExtractNumberOfSteps(ExtractStep):
    """Extract the total number of steps."""

    def __init__(self):
        definition = FeatureValueDefinition(
            task_name=TASK_NAME,
            feature_name=AV("total steps", "steps"),
            data_type="int",
            validator=GREATER_THAN_ZERO,
        )
        super().__init__(
            "pedometer",
            transform_function=agg_column("numberOfSteps", np.sum),
            definition=definition,
            level_filter="passive",
        )


class ExtractNumberOfActiveMinutes(ExtractStep):
    """Extract the total number of active minutes."""

    @staticmethod
    def count_bouts(data: pd.DataFrame) -> int:
        """Count the number of active bouts."""
        return data.averageActivePace.count()

    def __init__(self):
        definition = FeatureValueDefinition(
            task_name=TASK_NAME,
            feature_name=AV("active duration", "active_duration"),
            description="Number of active minutes during the day.",
            data_type="int",
            validator=GREATER_THAN_ZERO,
        )
        super().__init__(
            "pedometer",
            transform_function=self.count_bouts,
            definition=definition,
            level_filter="passive",
        )


class AggregateAverageActivePace(AggregateRawDataSetColumn):
    """Aggregate average active pace steps."""

    def __init__(self, **kwargs):
        data_set_id = "pedometer"
        description = (
            "The {aggregation} of the average active pace during "
            "each bout of one minute."
        )

        definition = FeatureValueDefinitionPrototype(
            feature_name=AV("average active pace", "avg_act_pace"),
            data_type="float64",
            unit="s.steps^-1",
            description=description,
            validator=GREATER_THAN_ZERO,
        )
        super().__init__(
            data_set_id,
            "averageActivePace",
            DEFAULT_AGGREGATIONS_Q95,
            definition,
            **kwargs,
        )


STEPS: List[ProcessingStep] = [
    ProcessingStepGroup(
        steps=[
            ExtractNumberOfSteps(),
            ExtractNumberOfActiveMinutes(),
            AggregateAverageActivePace(),
        ],
        task_name=TASK_NAME,
    )
]

process_passive = process_factory(
    task_name=TASK_NAME,
    steps=STEPS,
    codes=("passive", "passive-activity"),
    supported_type=ADSReading,
)
