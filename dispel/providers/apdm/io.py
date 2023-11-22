"""Functionality to read files from APDM."""

import datetime
import hashlib
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd

from dispel.data.core import Device, Evaluation, Reading
from dispel.data.levels import Level
from dispel.data.raw import (
    RawDataSet,
    RawDataSetDefinition,
    RawDataSetSource,
    RawDataValueDefinition,
)
from dispel.io.raw import (
    raw_data_set_definition_to_columns,
    raw_data_set_definition_to_index,
)
from dispel.providers.apdm.data import APDMPlatform, APDMReading, ApdmSensorType

# FIXME: following placeholders need to be fixed
FILENAME_PLACEHOLDER = "filename_placeholder"
DEVICE_PLACEHOLDER = "device id apdm"

_DEFAULT_RAW_DATA_SET_SOURCE = RawDataSetSource("APDM")


class Apdm:
    """Encapsulate APDM wearable sensor data."""

    def __init__(self, file):
        """
        Initialize APDM object.

        The class initiation reads the initial hdf file and setups the user class.

        Parameters
        ----------
        file
            The file location for ADPM sensor data

        """
        self.hfile = file
        self.sensors = self.hfile.get("Sensors")
        self.processed = self.hfile.get("Processed")
        self.sensor_dict = self.set_dictionary()

    def set_dictionary(self):
        """Get a dictionary mapping sensor locations to numbers."""
        label = "Label 0"
        sensors_dict = {}
        for sensor in list(self.sensors.items()):
            sensor_i = self.sensors.get(sensor[0])
            config = sensor_i.get("Configuration")
            sensors_dict[config.attrs[label].decode("UTF-8")] = int(sensor[0])
        return sensors_dict

    def get_acceleration(self, loc):
        """Get acceleration of sensor in location loc."""
        sensors_i = self.sensors.get(str(self.sensor_dict[loc]))
        return np.array(sensors_i.get("Accelerometer"))

    def get_gyroscope(self, loc):
        """Get angular velocity of sensor in location loc."""
        sensor_i = self.sensors.get(str(self.sensor_dict[loc]))
        return np.array(sensor_i.get("Gyroscope"))

    def get_time(self, loc):
        """Get time of sensor in location loc."""
        sensor_i = self.sensors.get(str(self.sensor_dict[loc]))
        timestamp = np.array(sensor_i.get("Time"))
        return (timestamp - timestamp[0]) * 1.0e-6

    def get_unix_timestamp(self, loc):
        """Get unix timestamp for the beginning of the file."""
        sensor_i = self.sensors.get(str(self.sensor_dict[loc]))
        timestamp = np.array(sensor_i.get("Time"))
        return timestamp

    def get_temperature(self, loc):
        """Get temperature of sensor in location loc."""
        sensor_i = self.sensors.get(str(self.sensor_dict[loc]))
        return sensor_i.get("Temperature")

    def get_sampling_rate(self, loc):
        """Get sample rate of sensor in location loc."""
        sensor_i = self.sensors.get(str(self.sensor_dict[loc]))
        config = sensor_i.get("Configuration")
        return int(config.attrs["Sample Rate"])

    def get_magnetometer(self, loc):
        """Get acceleration of sensor in location loc."""
        sensors_i = self.sensors.get(str(self.sensor_dict[loc]))
        return np.array(sensors_i.get("Magnetometer"))

    def get_orientation(self, loc):
        """Get orientation in a quaternion form for the sensor."""
        sensors_i = self.processed.get(str(self.sensor_dict[loc]))
        return np.array(sensors_i.get("Orientation"))

    @staticmethod
    def convert_unix_microseconds(unix_time):
        """Convert unix timestamp to micro seconds."""
        ans = datetime.datetime.fromtimestamp(unix_time / 1e6, tz=datetime.timezone.utc)

        return ans

    def create_dataframe(self, location=None):
        """Create a data frame with all sensor information."""
        data = pd.DataFrame()
        data["ts"] = pd.to_datetime(
            [
                self.convert_unix_microseconds(x)
                for x in self.get_unix_timestamp("Lumbar")
            ]
        ).tz_localize(None)
        if location:
            locations = [location]
        else:
            locations = self.sensor_dict.keys()

        for loc in locations:
            data[[loc + "_acc_x", loc + "_acc_y", loc + "_acc_z"]] = pd.DataFrame(
                self.get_acceleration(loc), index=data.index
            )
            data[[loc + "_gyro_x", loc + "_gyro_y", loc + "_gyro_z"]] = pd.DataFrame(
                self.get_gyroscope(loc), index=data.index
            )
            data[[loc + "_mag_x", loc + "_mag_y", loc + "_mag_z"]] = pd.DataFrame(
                self.get_magnetometer(loc), index=data.index
            )
            data[[loc + "_q1", loc + "_q2", loc + "_q3", loc + "_q4"]] = pd.DataFrame(
                self.get_orientation(loc), index=data.index
            )
            return data


def _create_sensor_raw_data_value_definition(
    sensor: str,
    measure: str,
    sources: Union[str, List[str]],
    targets: Union[str, List[str]],
    units: Union[str, List[Optional[str]], None] = None,
) -> List[RawDataValueDefinition]:
    definitions = [
        RawDataValueDefinition(
            "ts",
            "ts",
            "ms",
            description="ts",
            is_index=False,
            data_type="timedelta64[ms]",
        )
    ]

    if units is None or isinstance(units, str):
        units = [units] * len(sources)

    for source, target, unit in zip(sources, targets, units):
        definitions.append(
            RawDataValueDefinition(
                f"{target}", f"{measure or sensor} {target}", unit, f"{source}"
            )
        )

    return definitions


DATA_SET_DEFINITIONS = [
    RawDataSetDefinition(
        ApdmSensorType.ACCELEROMETER,
        _DEFAULT_RAW_DATA_SET_SOURCE,
        _create_sensor_raw_data_value_definition(
            "accelerometer",
            "acceleration",
            sources=["Lumbar_acc_" + suffix for suffix in "xyz"],
            targets=list("xyz"),
            units="g",
        ),
    ),
    RawDataSetDefinition(
        ApdmSensorType.RAW_ACCELEROMETER,
        _DEFAULT_RAW_DATA_SET_SOURCE,
        _create_sensor_raw_data_value_definition(
            "raw_accelerometer",
            "raw acceleration",
            sources=["Lumbar_acc_" + suffix for suffix in "xyz"],
            targets=list("xyz"),
            units="g",
        ),
    ),
    RawDataSetDefinition(
        ApdmSensorType.GYROSCOPE,
        _DEFAULT_RAW_DATA_SET_SOURCE,
        _create_sensor_raw_data_value_definition(
            "gyroscope",
            "angular velocity",
            sources=["Lumbar_gyro_" + suffix for suffix in "xyz"],
            targets=list("xyz"),
            units="rad/s",
        ),
    ),
    RawDataSetDefinition(
        ApdmSensorType.RAW_GYROSCOPE,
        _DEFAULT_RAW_DATA_SET_SOURCE,
        _create_sensor_raw_data_value_definition(
            "raw_gyroscope",
            "raw angular velocity",
            sources=["Lumbar_gyro_" + suffix for suffix in "xyz"],
            targets=list("xyz"),
            units="rad/s",
        ),
    ),
    RawDataSetDefinition(
        ApdmSensorType.RAW_MAGNETIC_FIELD,
        _DEFAULT_RAW_DATA_SET_SOURCE,
        _create_sensor_raw_data_value_definition(
            "raw_magnetic_field",
            "raw magnetic field",
            sources=["Lumbar_mag_" + suffix for suffix in "xyz"],
            targets=list("xyz"),
            units="mT",
        ),
    ),
    RawDataSetDefinition(
        ApdmSensorType.ATTITUDE,
        _DEFAULT_RAW_DATA_SET_SOURCE,
        _create_sensor_raw_data_value_definition(
            "attitude",
            "orientation",
            sources=["Lumbar_q" + suffix for suffix in "1234"],
            targets=list("wxyz"),
            units="",
        ),
    ),
]

DATA_SET_DEFINITIONS_DICT = {d.id: d for d in DATA_SET_DEFINITIONS}


def extract_raw_data_set(
    sensor_or_definition: Union[str, ApdmSensorType, RawDataSetDefinition],
    data: pd.DataFrame,
    data_set_definitions_dict: Dict[Union[str, ApdmSensorType], RawDataSetDefinition],
) -> RawDataSet:
    """Extract raw data set based on sensor type from data frame.

    Parameters
    ----------
    sensor_or_definition
        The sensor to be extracted or the definition of the data set
    data
        The data frame obtained with :func:`read_sensor_log_as_data_frame`
    data_set_definitions_dict
        A dictionary mapping sensor type to raw data set definitions

    Returns
    -------
    RawDataSet
        The raw data set for the specified sensor.

    Raises
    ------
    ValueError
        If the ``data`` is not a filled pandas data frame.
    ValueError
        If the ``sensor_or_definition``'s definition is not found.
    """
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("Data must be a pandas data frame with values")

    if not isinstance(sensor_or_definition, RawDataSetDefinition):
        if sensor_or_definition not in data_set_definitions_dict:
            raise ValueError(f"Missing definition for {sensor_or_definition}")

        definition = data_set_definitions_dict[sensor_or_definition]
    else:
        definition = sensor_or_definition

    columns = raw_data_set_definition_to_columns(definition)
    sensor_data = data.rename(
        columns={d.description: d.id for d in definition.value_definitions}
    )
    sensor_data = sensor_data[columns].copy()

    # set indices if specified
    index_cols = raw_data_set_definition_to_index(definition)
    if index_cols:
        sensor_data.set_index(index_cols, drop=True, inplace=True)
    return RawDataSet(definition, sensor_data)


def get_effective_time_frame(data: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Get the effective time frame for the APDM reading.

    Parameters
    ----------
    data
        The data frame obtained with :func:`read_sensor_log_as_data_frame`

    Returns
    -------
    Tuple[pd.Timestamp, pd.Timestamp]
        The start and end dates for the time frame spanned by `data`.

    """
    times = data["ts"]
    start, end = times.iloc[[0, -1]]
    return start, end


def get_evaluation(data: pd.DataFrame) -> Evaluation:
    """Get the evaluation for the APDM reading.

    Parameters
    ----------
    data
        The data frame obtained with :func:`read_apdm_as_data_frame`

    Returns
    -------
    Evaluation
        The evaluation information
    """
    start, end = get_effective_time_frame(data)
    if "filename" in data.columns:
        filename = data["filename"].iloc[0]
    else:
        filename = FILENAME_PLACEHOLDER

    file_id = filename.encode()
    evaluation_uuid = hashlib.md5(file_id).hexdigest()
    return Evaluation(uuid=evaluation_uuid, start=start, end=end, finished=True)


def get_device(data: pd.DataFrame) -> Device:
    """Get device information from the APDM reading.

    Parameters
    ----------
    data
        The data frame obtained with :func:`read_apdm_as_data_frame`

    Returns
    -------
    Device
        The device information present in the APDM reading
    """
    if "device" in data.columns:
        device = data.device.iloc[0]
    else:
        device = DEVICE_PLACEHOLDER
    return Device(model=device, platform=APDMPlatform())


def read_apdm_as_data_frame(path: str) -> pd.DataFrame:
    """Read an APDM h5 file into a pandas data frame.

    Parameters
    ----------
    path
        The path to the APDM h5 file

    Returns
    -------
    pandas.DataFrame
        The data frame representation of the APDM h5 file.
    """
    with open(path, "rb") as file:
        bytes_data = file.read()

    bytes_io_data = BytesIO(bytes_data)
    hfile = h5py.File(bytes_io_data, "r")
    apdm_sensors = Apdm(hfile)

    # get only lumbar sensor in dataframe
    data = apdm_sensors.create_dataframe("Lumbar")
    return data


def get_apdm_reading(data: pd.DataFrame) -> APDMReading:
    """Get the reading representation of an APDM data frame.

    Parameters
    ----------
    data
        The data frame obtained with :func:`read_apdm_as_data_frame`

    Returns
    -------
    APDMReading
        The :class:`~dispel.data.APDMReading` representation of the APDM h5
        file.
    """
    data_sets = []

    # try to extract all present data sets
    for raw_data_set in ApdmSensorType:
        try:
            data_sets.append(
                extract_raw_data_set(raw_data_set, data, DATA_SET_DEFINITIONS_DICT)
            )
        except KeyError as exc:
            print(exc)

    evaluation = get_evaluation(data)
    device = get_device(data)
    start, end = get_effective_time_frame(data)
    return APDMReading(
        evaluation=evaluation,
        levels=[Level(id_="apdm", start=start, end=end, raw_data_sets=data_sets)],
        device=device,
    )


def read_apdm(path: str) -> Reading:
    """Read data from SensorLog JSON file.

    Parameters
    ----------
    path
        The path to the JSON file containing the data to be read.

    Returns
    -------
    Reading
        The :class:`~dispel.data.core.Reading` representation of the SensorLog
        json file.
    """
    data = read_apdm_as_data_frame(path)
    return get_apdm_reading(data)
