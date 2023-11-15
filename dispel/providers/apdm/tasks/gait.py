"""Transform module for transform step specific to APDM format.

The transform step below are steps only modifying a reading coming from a APDM input.
"""

import pandas as pd

from dispel.data.raw import DEFAULT_COLUMNS, RawDataValueDefinition
from dispel.data.values import AbbreviatedValue as AV
from dispel.processing.data_set import StorageError, transformation
from dispel.processing.level import ProcessingStepGroup
from dispel.processing.transform import TransformStep
from dispel.providers.apdm.data import APDMReading
from dispel.providers.generic.flags.le_flags import NoMovementDetected
from dispel.providers.generic.tasks.gait.steps import (
    FlagPreprocessing,
    GaitCoreSteps,
    TechnicalFlags,
)
from dispel.providers.registry import process_factory
from dispel.signal.accelerometer import GRAVITY_CONSTANT as G_CONST

TASK_NAME = AV("Two-minute walk test", "2MWT")


class ConvertToG(TransformStep):
    """Scale from one m/s^2 to g.

    The step expects a unique data set id for `data_set_ids` pointing to a data frame
    containing both acceleration and gravity with a :class:`pandas.DatetimeIndex` index.
    """

    definitions = [RawDataValueDefinition("ts", "ts", unit="ms")] + [
        RawDataValueDefinition(ax, ax, unit="g") for ax in DEFAULT_COLUMNS
    ]

    @transformation
    def _transform(self, data) -> pd.DataFrame:
        data.loc[:, DEFAULT_COLUMNS] = data.loc[:, DEFAULT_COLUMNS] / G_CONST

        return data

    def get_new_data_set_id(self):
        """Overwrite new_data_set_id."""
        return f"{self.data_set_ids[0]}"


class ConvertAccelerometersUnits(ProcessingStepGroup):
    """Convert the unit of accelerometers to G."""

    steps = [
        ConvertToG(data_set_ids=id_, storage_error=StorageError.OVERWRITE)
        for id_ in ["raw_accelerometer", "accelerometer"]
    ]


class TwoMinuteWalkTestProcessingSteps(ProcessingStepGroup):
    """Steps to extract 2MWT features from a reading."""

    steps = [
        # first check technical flags
        TechnicalFlags(),
        # convert to right units
        ConvertAccelerometersUnits(),
        # run the core steps to extract intermediate raw datasets and features
        GaitCoreSteps(),
        # preprocess flags
        FlagPreprocessing(),
        # run the movement invalidations
        NoMovementDetected(),
    ]

    kwargs = {"task_name": TASK_NAME}


STEPS = [TwoMinuteWalkTestProcessingSteps()]

process_2mwt = process_factory(
    task_name=TASK_NAME,
    steps=STEPS,
    codes="2mwt",
    supported_type=APDMReading,
)
