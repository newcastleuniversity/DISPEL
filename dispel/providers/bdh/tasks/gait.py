"""Processing functionality for BDH Gait task."""

import pandas as pd

from dispel.data.levels import Level
from dispel.processing.data_set import StorageError, transformation
from dispel.processing.level import ProcessingStepGroup
from dispel.providers.bdh.data import BDHReading
from dispel.providers.bdh.transform import DEFINITIONS, ProcessBDHOnlyMixIn
from dispel.providers.generic.tasks.gait.steps import (
    TASK_NAME,
    GaitSteps,
    GaitStepsInclLee,
)
from dispel.providers.registry import process_factory


class Truncate6MWTBDH(ProcessBDHOnlyMixIn):
    """
    Truncate BDH 6MWT sensor data to keep the relevant six minutes.

    For BDH Format, IMUs are recorded during the entire session of the six
    minutes walk test, even during the voice instructions. To match ADS Format
    of the 6MWT and only get IMUs during the relevant period, the data has to
    be truncated to between the bip of the beginning and 6minutes later.
    """

    def get_new_data_set_id(self) -> str:
        """Overwrite new_data_set_id with current data_set_id."""
        assert len(data_set_ids := list(self.get_data_set_ids())) == 1
        return data_set_ids[0]

    @staticmethod
    @transformation
    def truncate_6min_after_bip(data: pd.DataFrame, level: Level):
        """Truncate the data frame to keep six minutes after the bip."""
        voice_instructions = level.get_raw_data_set("voice_instructions").data
        ts_start = pd.to_datetime(
            voice_instructions["begin_timestamp"], unit="ms"
        ).iloc[2]
        ts_ends = ts_start + pd.Timedelta(360, unit="s")
        return data[(ts_start <= data.ts) & (data.ts <= ts_ends)].copy()


class TruncateSensors6MWT(ProcessingStepGroup):
    """Truncate 6MWT sensors."""

    steps = [
        Truncate6MWTBDH(
            data_set_ids=id_,
            definitions=DEFINITIONS,
            storage_error=StorageError.OVERWRITE,
        )
        for id_ in ["accelerometer", "gyroscope", "gravity"]
    ]


class BDHGaitSteps(ProcessingStepGroup):
    """BDH-specific processing steps for gait."""

    steps = [
        TruncateSensors6MWT(),
        GaitSteps(),
    ]


class BDHGaitStepsInclLee(ProcessingStepGroup):
    """BDH-specific processing steps for gait including Lee algorithm."""

    steps = [TruncateSensors6MWT(), GaitStepsInclLee()]


process_6mwt = process_factory(
    task_name=TASK_NAME,
    steps=BDHGaitSteps(),
    codes=("6mwt-activity", "6mw-activity"),
    supported_type=BDHReading,
)
