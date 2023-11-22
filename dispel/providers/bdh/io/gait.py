"""Function for converting 6mwt BDH JSON files into a reading."""
from typing import Any, Dict

import pandas as pd

from dispel.data.levels import LevelId
from dispel.data.raw import RawDataSet, RawDataSetDefinition
from dispel.data.values import DefinitionId
from dispel.providers.bdh.io.core import convert_dataset, parse_raw_data_set
from dispel.signal.core import get_cartesian


def get_level_id(_: dict) -> LevelId:
    """Parse level id from level type and configuration.

    Returns
    -------
    LevelId
        Level id for the level
    """
    return LevelId("6mwt")


def convert_gps(data: Dict[str, Any], definition: RawDataSetDefinition) -> RawDataSet:
    """Convert gps dataset in ADS format."""
    ref = {
        "timestamp": "ts",
        "horizontal_accuracy": "HorizontalAccuracy",
        "vertical_accuracy": "VerticalAccuracy",
    }
    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        unit=definition.get_value_definition(DefinitionId("timestamp")).unit,
    )

    # This is only here for backwards compatibility. BDH used to store
    # relative latitude and longitude, which are insufficient for calculating
    # distances. The new format stores distance_x and distance_y.
    if "longitude" in data.keys():
        assert "distance_x" not in data.keys()
        data["longitude"], data["latitude"], _ = get_cartesian(
            data["latitude"], data["longitude"]
        )
        ref["latitude"], ref["longitude"] = "y", "x"
    else:
        ref["distance_y"], ref["distance_x"] = "y", "x"

    data, new_definitions = convert_dataset(data, definition, ref)
    definition_acc = RawDataSetDefinition(
        "gps", definition.source, new_definitions, is_computed=False
    )
    return parse_raw_data_set(data, definition_acc)
