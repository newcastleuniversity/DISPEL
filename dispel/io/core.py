"""Core functionality to read files.

TODO: investigate to merge with dispel.io.raw
"""
from typing import Any, Hashable, Optional

import numpy as np
import pandas as pd

from dispel.utils import convert_column_types

TYPE_MAPPINGS = {
    "int16": [
        "touchPathId",
        "numberOfSteps",
        "distance",
        "floorsAscended",
        "floorsDescended",
        "answer",
        "touch_path_id",
        "displayedValue",
        "userValue",
        "height",
        "width",
        "id",
    ],
    "float32": [
        "x",
        "y",
        "z",
        "w",
        "userAccelerationX",
        "userAccelerationY",
        "userAccelerationZ",
        "gravityX",
        "gravityY",
        "gravityZ",
        "xPosition",
        "yPosition",
        "pressure",
        "targetRadius",
        "xTargetBall",
        "yTargetBall",
        "maxPressure",
        "majorRadiusTolerance",
        "majorRadius",
        "averageActivePace",
        "xBallPosition",
        "yBallPosition",
        "xThumbPosition",
        "yThumbPosition",
        "targetPressure",
        "word_id",
        "subject_height",
        "level",
    ],
    "datetime64[ms]": [
        "ts",
        "tsTouch",
        "beginTimestamp",
        "endTimestamp",
        "tsAnswer",
        "tsDisplay",
        "appearance_timestamp",
        "disappearance_timestamp",
        "timestamp_out",
        "timestamp",
    ],
    "bool": [
        "ledToSuccess",
        "isValidPinch",
        "inEndZone",
        "success",
        "isInTouchingArea",
        "predefinedKey1",
        "predefinedKey2",
        "randomKey",
        "predefinedSequence",
        "randomSequence",
        "success",
    ],
}


def get_data_type_mapping(variable_name: Optional[Hashable]) -> str:
    """Get the data type for a variable name.

    Parameters
    ----------
    variable_name
        The name of the variable.

    Returns
    -------
    str
        The variable data type.
    """
    for type_, variables in TYPE_MAPPINGS.items():
        if variable_name in variables:
            return type_

    return "U"


def convert_literal_type(name: str, value: Any) -> Any:
    """Convert a literal based on the core type mapping.

    Parameters
    ----------
    name
        The name of the variable to be used in the type mapping. This is passed to
        :func:`get_data_type_mapping`.
    value
        The value to be converted

    Returns
    -------
    Any
        The converted value

    """
    type_str = get_data_type_mapping(name)
    expected_type = np.dtype(type_str)

    if np.issubdtype(expected_type, np.datetime64):
        return pd.to_datetime(value, unit="ms")
    if np.issubsctype(expected_type, np.bool_):
        return {"true": True, "false": False}[value]
    if np.issubsctype(expected_type, np.float32):
        value = None if value == "null" else value

    return np.array(value).astype(expected_type).item()


def convert_data_frame_type(data: pd.DataFrame) -> pd.DataFrame:
    """Convert a data frame based on the core type mapping.

    Parameters
    ----------
    data
        The data frame to be converted

    Returns
    -------
    pandas.DataFrame
        The data frame with converted columns according to
        :func:`get_data_type_mapping`.
    """
    return convert_column_types(data, get_data_type_mapping)
