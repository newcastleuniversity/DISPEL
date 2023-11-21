"""Function for converting BDH JSON files into a reading."""
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from dispel.data.epochs import Epoch
from dispel.data.levels import LevelId
from dispel.data.raw import RawDataSet, RawDataSetDefinition, RawDataValueDefinition
from dispel.io.core import convert_data_frame_type

OPTIONAL_FIELDS = {"reference_table"}

KEYS = SimpleNamespace(
    acquisition_provenance="acquisition_provenance",
    body="body",
    chipset="chipset",
    completion="completion",
    computed="computed",
    configuration="configuration",
    creation_date_time="creation_date_time",
    description="description",
    drawing_figure_name="drawing_figure_name",
    drawing_hand="drawing_hand",
    effective_time_frame="effective_time_frame",
    end_date_time="end_timestamp",
    measures="measures",
    mobile_computed_measures="mobile_computed_measures",
    header="header",
    id="id",
    interruption_reason="interruption_reason",
    kernel_version="kernel_version",
    levels="levels",
    manufacturer="manufacturer",
    model_name="model_name",
    model_number="model_number",
    name="name",
    os_version="os_version",
    raw_data="raw_data",
    values="columns",
    reference="reference",
    repetition_number="repetition_number",
    schema_id="schema_id",
    screen_resolution_pixel="screen_resolution_pixel",
    source="source",
    source_device="source_device",
    start_date_time="begin_timestamp",
    unit="unit",
    user_id="user_id",
)


def parse_raw_data_set(data: Dict, definition: RawDataSetDefinition) -> RawDataSet:
    """Parse raw data set for a reading.

    Parameters
    ----------
    data
        The raw data set data in BDH json format.
    definition
        The definition of the raw data set.

    Returns
    -------
    RawDataSet
        The created raw data set object.
    """
    try:
        data_frame = pd.DataFrame(data)
    except ValueError:
        data_frame = pd.DataFrame(data, index=[0])

    column_names = [d.id for d in definition.value_definitions]
    if data_frame.empty:
        return RawDataSet(
            definition=definition, data=pd.DataFrame(columns=column_names)
        )

    if len(data_frame.columns) != len(definition.value_definitions):
        for definition_value in definition.value_definitions:
            if definition_value.name not in data_frame.columns:
                assert (
                    definition_value.name in OPTIONAL_FIELDS
                ), "Inconsistent number of values with value definition"
                data_frame[definition_value.name] = np.nan

    data_frame = convert_data_frame_type(data_frame)

    return RawDataSet(definition=definition, data=data_frame)


def parse_raw_data_sets(
    data: Dict, definitions: Dict[str, RawDataSetDefinition]
) -> List[RawDataSet]:
    """Parse raw data sets for a reading.

    Parameters
    ----------
    data
        The data for all raw data sets.
    definitions
        A dictionary with the definitions of the data sets passed in `data`.

    Returns
    -------
    List[RawDataSet]
        A list of created raw data sets.

    Raises
    ------
    ValueError
        If a definition is not found for a data set.
    """
    data_sets = []
    for id_, data_raw in data.items():
        if id_ not in definitions:
            raise ValueError(f"No definition found for data set {id_}")

        data_sets.append(parse_raw_data_set(data_raw, definitions[id_]))

    return data_sets


def parse_epoch(data: Dict) -> Epoch:
    """Parse the effective time frame of a reading.

    Parameters
    ----------
    data
        A dictionary containing the effective time frame information in BDH json format.

    Returns
    -------
    Epoch
        Returns an epoch representation of the effective time frame of a reading
    """
    return Epoch(
        pd.to_datetime(data[KEYS.start_date_time], unit="ms"),
        pd.to_datetime(data[KEYS.end_date_time], unit="ms"),
    )


def convert_dataset(
    data: Dict[str, Any], definition: RawDataSetDefinition, ref: Dict[str, str]
) -> Tuple[Dict[str, Any], List[RawDataValueDefinition]]:
    """Convert dataset column names."""
    for old_key, new_key in ref.items():
        data[new_key] = data[old_key]
        del data[old_key]
    _definitions = deepcopy(definition).value_definitions
    new_definitions: List[RawDataValueDefinition] = []
    for value_definition in _definitions:
        try:
            value_definition.id = ref[value_definition.id]
            value_definition.name = ref[value_definition.name]
        except KeyError:
            pass
        new_definitions.append(value_definition)
    return data, new_definitions


def get_level_id_two_hands(config: dict) -> LevelId:
    """
    Parse level id from level type and configuration for two hands level test.

    Parameters
    ----------
    config
        The level configuration

    Returns
    -------
    LevelId
        Level id for the level
    """
    context = config["hand"]
    return LevelId(context)
