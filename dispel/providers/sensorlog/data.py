"""Data models used for the SensorLog data."""
from typing import List, Optional, Union

from dispel.data.raw import (
    RawDataSetDefinition,
    RawDataSetSource,
    RawDataValueDefinition,
    SensorType,
)


class SensorLogSensorType(SensorType):
    """Types of sensors available from SensorLog json files."""

    ACCELEROMETER = "accelerometer"
    ACTIVITY = "activity"
    ALTIMETER = "altimeter"
    BATTERY = "battery"
    CORE_LOCATION = "location"
    DECIBELS = "decibels"
    DEVICE_MOTION = "motion"
    DEVICE_MOTION_MAGNETIC_FIELD = "motion-magnetic-field"
    DEVICE_MOTION_QUATERNION = "motion-quaternion"
    DEVICE_MOTION_ROTATION_RATE = "motion-rotation-rate"
    DEVICE_MOTION_USER_ACCELERATION = "motion-user-acceleration"
    DEVICE_MOTION_GRAVITY = "motion-gravity"
    GYRO = "gyro"
    HEADING = "heading"
    MAGNETOMETER = "magnetometer"
    ORIENTATION = "orientation"


DEFAULT_RAW_DATA_SET_SOURCE = RawDataSetSource("SensorLog")

LOGGING_TIME_VALUE_DEFINITION = RawDataValueDefinition(
    "loggingTime", "Timestamp", data_type="datetime64", is_index=True
)


def create_sensor_raw_data_value_definitions(
    sensor: str,
    measure: str,
    axes: Union[str, List[str]],
    units: Union[str, List[Optional[str]], None] = None,
    timestamp_reference: str = "sinceReboot",
    timestamp_unit: Optional[str] = "seconds",
    timestamp_data_type: str = "timedelta64[s]",
) -> List[RawDataValueDefinition]:
    """Create the definitions for raw data values."""
    definitions = [
        RawDataValueDefinition(
            f"{sensor}Timestamp_{timestamp_reference}",
            "Timestamp",
            timestamp_unit,
            is_index=True,
            data_type=timestamp_data_type,
        )
    ]

    if units is None or isinstance(units, str):
        units = [units] * len(axes)

    for axis, unit in zip(axes, units):
        definitions.append(
            RawDataValueDefinition(
                f"{sensor}{measure}{axis}", f"{measure or sensor} {axis}", unit
            )
        )

    return definitions


DATA_SET_DEFINITIONS = [
    RawDataSetDefinition(
        SensorLogSensorType.ACCELEROMETER,
        DEFAULT_RAW_DATA_SET_SOURCE,
        create_sensor_raw_data_value_definitions(
            "accelerometer", "Acceleration", "XYZ", "G"
        ),
    ),
    RawDataSetDefinition(
        SensorLogSensorType.ACTIVITY,
        DEFAULT_RAW_DATA_SET_SOURCE,
        create_sensor_raw_data_value_definitions(
            "activity", "Activity", ["Confidence", "StartDate"]
        )
        + [RawDataValueDefinition("activity", "Activity")],
        is_computed=True,
    ),
    RawDataSetDefinition(
        SensorLogSensorType.ALTIMETER,
        DEFAULT_RAW_DATA_SET_SOURCE,
        create_sensor_raw_data_value_definitions(
            "altimeter",
            "",
            ["Pressure", "RelativeAltitude", "Reset"],
            ["kPA", "m", None],
        ),
    ),
    RawDataSetDefinition(
        SensorLogSensorType.BATTERY,
        DEFAULT_RAW_DATA_SET_SOURCE,
        [
            LOGGING_TIME_VALUE_DEFINITION,
            RawDataValueDefinition("batteryLevel", "Battery Level"),
            RawDataValueDefinition("batteryState", "Battery State"),
        ],
    ),
    RawDataSetDefinition(
        SensorLogSensorType.CORE_LOCATION,
        DEFAULT_RAW_DATA_SET_SOURCE,
        create_sensor_raw_data_value_definitions(
            "location",
            "",
            [
                "Latitude",
                "Longitude",
                "Altitude",
                "Speed",
                "Course",
                "Floor",
                "HorizontalAccuracy",
                "VerticalAccuracy",
                "MagneticHeading",
                "TrueHeading",
            ],
            timestamp_reference="since1970",
            timestamp_unit=None,
            timestamp_data_type="datetime64[s]",
        ),
    ),
    RawDataSetDefinition(
        SensorLogSensorType.DECIBELS,
        DEFAULT_RAW_DATA_SET_SOURCE,
        [
            LOGGING_TIME_VALUE_DEFINITION,
            RawDataValueDefinition(
                "avAudioRecorderAveragePower", "Average Decibels", "dB"
            ),
            RawDataValueDefinition("avAudioRecorderPeakPower", "Peak Decibels", "dB"),
        ],
    ),
    RawDataSetDefinition(
        SensorLogSensorType.DEVICE_MOTION,
        DEFAULT_RAW_DATA_SET_SOURCE,
        create_sensor_raw_data_value_definitions(
            "motion",
            "",
            [
                "Pitch",
                "Roll",
                "Yaw",
            ],
            "rad",
        ),
        is_computed=True,
    ),
    RawDataSetDefinition(
        SensorLogSensorType.DEVICE_MOTION_MAGNETIC_FIELD,
        DEFAULT_RAW_DATA_SET_SOURCE,
        create_sensor_raw_data_value_definitions(
            "motion", "MagneticField", ["X", "Y", "Z", "CalibrationAccuracy"]
        ),
        is_computed=True,
    ),
    RawDataSetDefinition(
        SensorLogSensorType.DEVICE_MOTION_QUATERNION,
        DEFAULT_RAW_DATA_SET_SOURCE,
        create_sensor_raw_data_value_definitions("motion", "Quaternion", "WXYZ"),
        is_computed=True,
    ),
    RawDataSetDefinition(
        SensorLogSensorType.DEVICE_MOTION_ROTATION_RATE,
        DEFAULT_RAW_DATA_SET_SOURCE,
        create_sensor_raw_data_value_definitions("motion", "RotationRate", "XYZ"),
        is_computed=True,
    ),
    RawDataSetDefinition(
        SensorLogSensorType.DEVICE_MOTION_USER_ACCELERATION,
        DEFAULT_RAW_DATA_SET_SOURCE,
        create_sensor_raw_data_value_definitions("motion", "UserAcceleration", "XYZ"),
        is_computed=True,
    ),
    RawDataSetDefinition(
        SensorLogSensorType.DEVICE_MOTION_GRAVITY,
        DEFAULT_RAW_DATA_SET_SOURCE,
        create_sensor_raw_data_value_definitions("motion", "Gravity", "XYZ"),
        is_computed=True,
    ),
    RawDataSetDefinition(
        SensorLogSensorType.GYRO,
        DEFAULT_RAW_DATA_SET_SOURCE,
        create_sensor_raw_data_value_definitions("gyro", "Rotation", "XYZ", "rad/sec"),
    ),
    RawDataSetDefinition(
        SensorLogSensorType.HEADING,
        DEFAULT_RAW_DATA_SET_SOURCE,
        create_sensor_raw_data_value_definitions(
            "location",
            "Heading",
            ["X", "Y", "Z", "Accuracy"],
            timestamp_reference="since1970",
            timestamp_unit=None,
            timestamp_data_type="datetime64[s]",
        ),
        is_computed=True,
    ),
    RawDataSetDefinition(
        SensorLogSensorType.MAGNETOMETER,
        DEFAULT_RAW_DATA_SET_SOURCE,
        create_sensor_raw_data_value_definitions("magnetometer", "", "XYZ", "µT"),
    ),
    RawDataSetDefinition(
        SensorLogSensorType.ORIENTATION,
        DEFAULT_RAW_DATA_SET_SOURCE,
        [
            LOGGING_TIME_VALUE_DEFINITION,
            RawDataValueDefinition("deviceOrientation", "Orientation"),
        ],
    ),
]

DATA_SET_DEFINITIONS_DICT = {d.id: d for d in DATA_SET_DEFINITIONS}
