"""Common functionality to read files."""
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from dispel.data.raw import (
    RawDataSet,
    RawDataSetDefinition,
    RawDataSetSource,
    RawDataValueDefinition,
    SensorType,
)
from dispel.data.values import ValueDefinition


def raw_data_set_definition_to_columns(definition: RawDataSetDefinition) -> List[str]:
    """Get column names for data set definition.

    Parameters
    ----------
    definition
        The raw data set definition.

    Returns
    -------
    List[str]
        The column names of the value definitions of the data set.

    """
    return [d.id for d in definition.value_definitions]


def raw_data_set_definition_to_index(definition: RawDataSetDefinition) -> List[str]:
    """Get index names for data set definition.

    Parameters
    ----------
    definition
        The raw data set definition

    Returns
    -------
    List[str]
        The list of indices, if any value definition is part of the index

    """
    return [d.id for d in definition.value_definitions if d.is_index]


def extract_raw_data_set(
    sensor_or_definition: Union[str, RawDataSetDefinition],
    data: pd.DataFrame,
    data_set_definitions_dict: Dict[Union[str, SensorType], RawDataSetDefinition],
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
    sensor_data = data[columns].copy()

    # convert types if specified
    for value_definition in definition.value_definitions:
        sensor_data = convert_types(sensor_data, value_definition)

    # set indices if specified
    index_cols = raw_data_set_definition_to_index(definition)
    if index_cols:
        sensor_data.set_index(index_cols, drop=True, inplace=True)

    return RawDataSet(definition, sensor_data)


def convert_types(
    sensor_data: pd.DataFrame, value_definition: ValueDefinition
) -> pd.DataFrame:
    """Convert sensor data types.

    The conversion happens only if specified in
    :class:`dispel.data.raw.RawDataSetDefinition`.

    Parameters
    ----------
    sensor_data
        Copy of the data frame output of :func:`read_sensor_log_as_data_frame`
    value_definition
        The definition of a value.

    Returns
    -------
    pandas.DataFrame
        The updated data frame with formatted types.
    """
    # extract the data_type
    data_type = value_definition.data_type

    if data_type is not None:
        # get value id
        id_ = value_definition.id

        # check for special case with timedelta64
        if np.issubdtype(data_type, np.timedelta64):
            unit = get_unit_from_datatype(data_type)
            sensor_data[id_] = pd.to_timedelta(sensor_data[id_], unit=unit)
        elif np.issubdtype(data_type, np.datetime64):
            unit = get_unit_from_datatype(data_type)
            sensor_data[id_] = pd.to_datetime(sensor_data[id_], unit=unit).astype(
                "datetime64[ns]"
            )
        else:
            # cast sensor data in new data_type
            sensor_data[id_] = sensor_data[id_].astype(data_type, errors="ignore")
    return sensor_data


def get_unit_from_datatype(data_type: str) -> Union[str, None]:
    """Parse data_type string to extract the unit.

    Parameters
    ----------
    data_type
        The data type, e.g., ``'timedelta64[us]'``.

    Returns
    -------
    str
        The unit, e.g. the unit of ``'timedelta64[us]'`` is ``us``
    """
    split = data_type.split("[")

    # check if there is a unit
    if len(split) <= 1:
        return None

    return split[1][:-1]


def generate_raw_data_value_definition(column: str, unit: Optional[str] = None):
    """Create a basic RawDataValueDefinition for data (unit is optional)."""
    return RawDataValueDefinition(column, f"{column} data", unit)


def generate_raw_data_set_definition(
    data_set_id: str,
    columns: Sequence[str],
):
    """Create a basic RawDataSetDefinition for data."""
    definitions = [RawDataValueDefinition(column, column.upper()) for column in columns]
    return RawDataSetDefinition(data_set_id, RawDataSetSource("example"), definitions)


def generate_raw_data_set(data_set_id: str, columns: Sequence[str]) -> RawDataSet:
    """Generate a random raw data set."""
    return RawDataSet(
        generate_raw_data_set_definition(data_set_id, columns),
        pd.DataFrame(0, index=range(2), columns=columns),
    )
