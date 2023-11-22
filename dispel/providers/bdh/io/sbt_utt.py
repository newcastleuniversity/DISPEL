"""Function for converting sbt_utt BDH JSON files into a reading."""
from typing import Any, Dict

import pandas as pd

from dispel.data.levels import LevelId
from dispel.data.raw import RawDataSet, RawDataSetDefinition
from dispel.data.values import DefinitionId
from dispel.providers.bdh.io.core import convert_dataset, parse_raw_data_set


def get_level_id(config: dict) -> LevelId:
    """Parse level id from level type and configuration.

    Parameters
    ----------
    config
        The level configuration

    Returns
    -------
    LevelId
        Level id for the level.

    Raises
    ------
    NotImplementedError
        If the given mode parsing has not been implemented.
    """
    if config["mode"] == "static-balance":
        return LevelId("sbt")
    if config["mode"] == "u-turn":
        return LevelId("utt")

    raise NotImplementedError(f"Level Id is not implemented for mode: {config['mode']}")


def convert_timestamp(
    id_: str, data: Dict[str, Any], definition: RawDataSetDefinition
) -> RawDataSet:
    """Convert timestamp in ADS format."""
    ref = {"timestamp": "ts"}
    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        unit=definition.get_value_definition(DefinitionId("timestamp")).unit,
    )
    data, new_definitions = convert_dataset(data, definition, ref)
    definition_acc = RawDataSetDefinition(
        id_, definition.source, new_definitions, is_computed=True
    )
    return parse_raw_data_set(data, definition_acc)
