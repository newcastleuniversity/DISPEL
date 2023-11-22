"""Additional models for APDM data sets."""
from dispel.data.core import Reading
from dispel.data.devices import PlatformType
from dispel.data.raw import SensorType


class APDMPlatform(PlatformType):
    """APDM sensors used to record a reading."""

    def repr(self) -> str:
        """Get a string representation of the platform."""
        return "apdm"


class APDMReading(Reading):
    """APDM reading."""


class ApdmSensorType(SensorType):
    """Types of sensors available from APDM h5 files."""

    RAW_ACCELEROMETER = "raw_accelerometer"
    ACCELEROMETER = "accelerometer"
    RAW_GYROSCOPE = "raw_gyroscope"
    GYROSCOPE = "gyroscope"
    RAW_MAGNETIC_FIELD = "raw_magnetic_field"
    ATTITUDE = "attitude"
