"""Functionality to read files from SensorLog."""
import hashlib
from typing import Union

import pandas as pd

from dispel.data.core import Device, Evaluation, Reading
from dispel.data.devices import IOSPlatform
from dispel.data.levels import Level
from dispel.io.raw import extract_raw_data_set
from dispel.providers.sensorlog import PROVIDER_ID
from dispel.providers.sensorlog.data import (
    DATA_SET_DEFINITIONS_DICT,
    SensorLogSensorType,
)


def get_evaluation(data: Union[pd.DataFrame, dict]) -> Evaluation:
    """Get the evaluation for the SensorLog reading.

    Parameters
    ----------
    data
        The data frame obtained with :func:`read_sensor_log_as_data_frame`

    Returns
    -------
    Evaluation
        The evaluation information
    """
    start, end = pd.to_datetime(data["loggingTime"]).agg(["min", "max"])

    evaluation_uuid = hashlib.md5(
        (data["identifierForVendor"][0] + data["loggingTime"][0]).encode()
    ).hexdigest()

    return Evaluation(start=start, end=end, uuid=evaluation_uuid)


def get_device(data: Union[pd.DataFrame, dict]) -> Device:
    """Get device information from the SensorLog reading.

    Parameters
    ----------
    data
        The data frame obtained with :func:`read_sensor_log_as_data_frame`

    Returns
    -------
    Device
        The device information present in the SensorLog reading

    Raises
    ------
    ValueError
        If a wrong identifier definition in data set is given.
    """
    device_uuid = data["identifierForVendor"].unique()

    if len(device_uuid) != 1:
        raise ValueError("Wrong identifier definition in data set")

    return Device(uuid=device_uuid[0], platform=IOSPlatform())


def read_sensor_log_as_data_frame(path: str) -> pd.DataFrame:
    """Read a SensorLog json file into a pandas data frame.

    Parameters
    ----------
    path
        The path to the SensorLog json file

    Returns
    -------
    pandas.DataFrame
        The data frame representation of the SensorLog json file.
    """
    return pd.read_json(path)


def get_sensor_log_reading(data: Union[pd.DataFrame, dict]) -> Reading:
    """Get the reading representation of a SensorLog data frame.

    Parameters
    ----------
    data
        The data frame obtained with :func:`read_sensor_log_as_data_frame`

    Returns
    -------
    Reading
        The :class:`~dispel.data.core.Reading` representation of the SensorLog json file.
    """
    data_sets = []

    # try to extract all present data sets
    for sensor_type in SensorLogSensorType:
        try:
            data_sets.append(
                extract_raw_data_set(sensor_type, data, DATA_SET_DEFINITIONS_DICT)
            )
        except KeyError:
            pass  # skipping columns not being present

    evaluation = get_evaluation(data)
    device = get_device(data)

    return Reading(
        evaluation=evaluation,
        levels=[
            Level(
                id_=PROVIDER_ID,
                start=evaluation.start,
                end=evaluation.end,
                raw_data_sets=data_sets,
            )
        ],
        device=device,
    )


def read_sensor_log(path: str) -> Reading:
    """Read data from SensorLog JSON file.

    Parameters
    ----------
    path
        The path to the JSON file containing the data to be read.

    Returns
    -------
    Reading
        The :class:`~dispel.data.core.Reading` representation of the SensorLog json file.
    """
    data = read_sensor_log_as_data_frame(path)
    return get_sensor_log_reading(data)
