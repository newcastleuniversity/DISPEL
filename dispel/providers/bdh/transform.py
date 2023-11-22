"""Transform module for transform step specific to io format.

The transform step below are steps only modifying a reading coming from a
BDH input.
"""
from abc import ABC
from typing import Any, Dict, List

import pandas as pd

from dispel.data.core import Reading
from dispel.data.levels import Level
from dispel.data.raw import DEFAULT_COLUMNS, RawDataValueDefinition
from dispel.processing.core import ProcessingStep, ProcessResultType
from dispel.processing.data_set import transformation
from dispel.processing.level import LevelFilter, LevelIdFilter, ProcessingStepGroup
from dispel.processing.transform import TransformStep
from dispel.providers.bdh.data import BDHReading


class ProcessBDHOnlyMixIn(ABC, TransformStep):
    """An abstract class to process only if an instance is a BDH Reading."""

    def process_reading(self, reading: Reading, **kwargs) -> ProcessResultType:
        """Overwrite process reading."""
        if isinstance(reading, BDHReading):
            yield from super().process_reading(reading, **kwargs)


class ProcessBDHProcessingStepGroupOnly(ABC):
    """An abstract class to process only if an instance is a BDH Reading."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs or self.kwargs
        super().__init__(kwargs)

    def get_kwargs(self) -> Dict[str, Any]:
        """Get keyword arguments to be added to the processing."""
        return self.kwargs

    # pylint: disable=no-member
    def process_reading(self, reading: Reading, **kwargs) -> ProcessResultType:
        """Overwrite process_reading."""
        (updated_kwargs := kwargs.copy()).update(self.get_kwargs())

        if isinstance(reading, BDHReading):
            # type: ignore
            yield from super().process_reading(  # type: ignore
                reading, **updated_kwargs
            )


class TruncateBDHVoiceInstructions(ProcessBDHOnlyMixIn):
    """
    Truncate voice instructions in UTT test for BDH format.

    For BDH Format, IMUs are recorded during the entire session of SBT-UTT
    test, even during the voice instructions. To match ADS Format of the UTT
    and only get IMUs during the actual UTT, the data has to be truncated after
    the third voice instruction.
    """

    level_filter: LevelFilter = LevelIdFilter("utt")

    def get_new_data_set_id(self) -> str:
        """Overwrite new_data_set_id with current data_set_id."""
        assert len(data_set_ids := list(self.get_data_set_ids())) == 1
        return data_set_ids[0]

    @staticmethod
    def get_voice_instruction_end(level: Level) -> pd.Timestamp:
        """Get the timestamp at which the voice instructions end."""
        data = level.get_raw_data_set("voice_instructions").data
        return pd.to_datetime(data["start_timestamp"].iloc[2], unit="ms")

    @transformation
    def truncate(self, data: pd.DataFrame, level: Level):
        """Truncate the data frame removing voice instructions."""
        return data[data.ts >= self.get_voice_instruction_end(level)].copy()


class TruncateLastThirtySecondsBDH(ProcessBDHOnlyMixIn):
    """
    Truncate BDH sensor data to keep the last 30 seconds.

    For BDH Format, IMUs are recorded during the entire session of SBT-UTT
    test, even during the voice instructions. To match ADS Format of the SBT
    and only get IMUs during the actual SBT, the data has to be truncated to
    keep the last 30 seconds.
    """

    level_filter: LevelFilter = LevelIdFilter("sbt")

    def get_new_data_set_id(self) -> str:
        """Overwrite new_data_set_id with current data_set_id."""
        assert len(data_set_ids := list(self.get_data_set_ids())) == 1
        return data_set_ids[0]

    @transformation
    def truncate_last_30_seconds(self, data: pd.DataFrame):
        """Truncate the data frame to keep 30 last Seconds."""
        return data[data.ts >= data.iloc[-1]["ts"] - pd.Timedelta(30, unit="s")].copy()


class TruncateFirstTwoMinBDH(ProcessBDHOnlyMixIn):
    """
    Truncate BDH sensor data to keep only the first two minutes.

    Due to the fact the SBT-UTT test does not have a maximum time allowed
    by Konectom during recording this is to prevent that extremely long
    recordings are being processed and takes into account only the first
    two minutes. In practice, this also addresses a Konectom isssue where
    there is a gap in timestamps in UTT that can span to hours.
    """

    def get_new_data_set_id(self) -> str:
        """Overwrite new_data_set_id with current data_set_id."""
        assert len(data_set_ids := list(self.get_data_set_ids())) == 1
        return data_set_ids[0]

    @transformation
    def truncate_first_two_minutes(self, data: pd.DataFrame):
        """Truncate the data frame to keep first two minutes."""
        return data[data.ts <= data.iloc[0]["ts"] + pd.Timedelta(120, unit="s")].copy()


DEFINITIONS = [
    RawDataValueDefinition(column, column) for column in DEFAULT_COLUMNS + ["ts"]
]


TRUNCATE_SENSORS: List[ProcessingStep] = [
    TruncateBDHVoiceInstructions(
        data_set_ids="accelerometer", definitions=DEFINITIONS, storage_error="overwrite"
    ),
    TruncateBDHVoiceInstructions(
        data_set_ids="gyroscope", definitions=DEFINITIONS, storage_error="overwrite"
    ),
    TruncateBDHVoiceInstructions(
        data_set_ids="gravity", definitions=DEFINITIONS, storage_error="overwrite"
    ),
    TruncateFirstTwoMinBDH(
        data_set_ids="accelerometer", definitions=DEFINITIONS, storage_error="overwrite"
    ),
    TruncateFirstTwoMinBDH(
        data_set_ids="gyroscope", definitions=DEFINITIONS, storage_error="overwrite"
    ),
    TruncateFirstTwoMinBDH(
        data_set_ids="gravity", definitions=DEFINITIONS, storage_error="overwrite"
    ),
]


class TruncateSensorsSBT(ProcessingStepGroup):
    """A ProcessingStepGroup concatenating Sensor Truncation steps."""

    steps = [
        TruncateLastThirtySecondsBDH(
            data_set_ids="accelerometer",
            definitions=DEFINITIONS,
            storage_error="overwrite",
        ),
        TruncateLastThirtySecondsBDH(
            data_set_ids="gyroscope", definitions=DEFINITIONS, storage_error="overwrite"
        ),
        TruncateLastThirtySecondsBDH(
            data_set_ids="gravity", definitions=DEFINITIONS, storage_error="overwrite"
        ),
    ]
