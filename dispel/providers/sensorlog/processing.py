"""Processing steps to transform SensorLog readings into internal structure."""
import pandas as pd

from dispel.data.levels import Level
from dispel.data.raw import (
    ACCELEROMETER_COLUMNS,
    DEFAULT_COLUMNS,
    GRAVITY_COLUMNS,
    RawDataValueDefinition,
)
from dispel.processing.data_set import StorageError, transformation
from dispel.processing.level import ProcessingStepGroup
from dispel.processing.transform import TransformStep

_TS_COLUMN = RawDataValueDefinition("ts", "ts")
_ACC_COLUMNS = [
    RawDataValueDefinition(f"userAcceleration{ax}", ax) for ax in list("XYZ")
]
_GRAVITY_COLUMNS = [RawDataValueDefinition(f"gravity{ax}", ax) for ax in list("XYZ")]
_GYR_COLUMNS = [RawDataValueDefinition(axis, axis) for axis in DEFAULT_COLUMNS]


class MeltUserMotionAndGravity(TransformStep):
    """A step to melt the motion and gravity measurements from sensorlog."""

    data_set_ids = ["motion-user-acceleration", "motion-gravity"]
    storage_error = StorageError.OVERWRITE
    definitions = [_TS_COLUMN] + _ACC_COLUMNS + _GRAVITY_COLUMNS

    @transformation
    def _melt(
        self,
        user: pd.DataFrame,
        gravity: pd.DataFrame,
    ) -> pd.DataFrame:
        data = pd.concat([user, gravity], axis=1)
        data.columns = ACCELEROMETER_COLUMNS + GRAVITY_COLUMNS
        data.index.name = "ts"
        data.sort_index(inplace=True)
        data.reset_index(inplace=True)

        return data


class AlignTimestamp(TransformStep):
    """Align the timestamp of a data set to refer to logging time."""

    ts_column = "ts"
    storage_error = StorageError.OVERWRITE

    def get_new_data_set_id(self) -> str:
        """Return the same data set id as provided for the input."""
        assert (
            len(data_set_ids := list(self.get_data_set_ids())) == 1
        ), "Only one data set id is allowed to be processed."
        return data_set_ids[0]

    @transformation
    def _align(self, data: pd.DataFrame, level: Level) -> pd.DataFrame:
        # Logging based offset to align time stamps
        start = data[self.ts_column].min()
        offset = level.start - start
        data[self.ts_column] += offset

        return data


class PreprocessAcceleration(ProcessingStepGroup):
    """Preprocessing steps to transform the acceleration."""

    # pylint: disable=no-member
    steps = [
        MeltUserMotionAndGravity(new_data_set_id="acc"),
        AlignTimestamp(
            data_set_ids="acc",
            definitions=MeltUserMotionAndGravity.definitions,
        ),
    ]


class RenameAndUnsetTimestampIndexForGyroscope(TransformStep):
    """A step to rename columns and reset index for gyro."""

    data_set_ids = "gyro"
    new_data_set_id = "gyroscope"
    definitions = [_TS_COLUMN] + _GYR_COLUMNS

    @transformation
    def _transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data.index.name = "ts"
        data.reset_index(inplace=True)
        data.columns = [
            definition.id
            for definition in self.get_definitions()  # pylint: disable=E1133
        ]

        return data


class PreprocessGyroscope(ProcessingStepGroup):
    """Preprocessing steps to transform the gyroscope."""

    # pylint: disable=no-member
    steps = [
        RenameAndUnsetTimestampIndexForGyroscope(),
        AlignTimestamp(
            data_set_ids="gyroscope",
            definitions=RenameAndUnsetTimestampIndexForGyroscope.definitions,
        ),
    ]
