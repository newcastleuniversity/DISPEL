"""Function for converting CPS test BDH JSON files into a reading."""
from copy import deepcopy
from typing import Any, Dict, List

from dispel.data.levels import LevelId
from dispel.data.raw import RawDataSet, RawDataSetDefinition, RawDataValueDefinition
from dispel.data.values import Value, ValueDefinition
from dispel.providers.bdh.io.core import parse_raw_data_set


def translate_reference_table_type(table_type: int) -> List[Value]:
    """Convert the reference table for CPS test into the expected format.

    Parameters
    ----------
    table_type
        Either 1, 2, or 3

    Returns
    -------
    List[Value]
        List of value definitions

    Raises
    ------
    ValueError
        Raises a value error if the table_type is not in {1,2,3}.
    """
    if table_type not in {1, 2, 3, 4}:
        raise ValueError(
            "table_type should be 1, 2, 3 or 4 but is equal to " f"{table_type}"
        )
    _context = []
    keys = {
        1: "predefinedKey1",
        2: "predefinedKey2",
        3: "randomKey",
        4: "predefinedKey3",
    }
    for key, table in keys.items():
        _context.append(Value(ValueDefinition(table, table), key == table_type))
    return _context


def translate_sequence_type(sequence_type: str) -> List[Value]:
    """Convert the sequence type for the CPS test into the expected format."""
    _context = []
    keys = {"predefined": "predefinedSequence", "random": "randomSequence"}
    for key, table in keys.items():
        _context.append(Value(ValueDefinition(table, table), key == sequence_type))

    return _context


def _get_display_value(data: Dict[str, Any]) -> Dict:
    """Use reference_table to convert symbols into values."""
    data["displayedValue"] = [
        data["reference_table"][i].index(data["displayedSymbol"][i]) + 1
        for i in range(len(data["displayedSymbol"]))
    ]
    data["userValue"] = []
    for i, s in enumerate(data["userSymbol"]):
        if s is None:
            data["userValue"].append(None)
        else:
            data["userValue"].append(
                data["reference_table"][i].index(data["userSymbol"][i]) + 1
            )
    data["displayedSymbol"] = [f"Symbol{i}" for i in data["displayedSymbol"]]
    data["userSymbol"] = [f"Symbol{i}" if i else "None" for i in data["userSymbol"]]
    del data["reference_table"]
    return data


def convert_activity_sequence(
    data: Dict[str, Any], definition: RawDataSetDefinition
) -> RawDataSet:
    """Convert activity sequence dataset to userInput."""
    _definitions = deepcopy(definition).value_definitions
    new_definitions = []
    is_std = "reference_table" in data
    if is_std:
        # This is digit_to_symbol
        ref = {
            "presented_symbol_timestamp": "tsDisplay",
            "response_timestamp": "tsAnswer",
            "presented_symbol": "displayedSymbol",
            "response": "userSymbol",
        }
    else:
        ref = {
            "presented_symbol_timestamp": "tsDisplay",
            "response_timestamp": "tsAnswer",
            "presented_symbol": "displayedValue",
            "response": "userValue",
        }

    for value_definition in _definitions:
        try:
            value_definition.id = ref[value_definition.id]
            value_definition.name = ref[value_definition.name]
        except KeyError:
            pass
        if value_definition.id != "reference_table":
            new_definitions.append(value_definition)
    new_definitions.append(RawDataValueDefinition("success", "success"))

    if is_std:
        new_definitions.append(RawDataValueDefinition("userValue", "userValue"))
        new_definitions.append(
            RawDataValueDefinition("displayedValue", "displayedValue")
        )

    definition_user_input = RawDataSetDefinition(
        "userInput", definition.source, new_definitions, is_computed=True
    )

    for old_key, new_key in ref.items():
        data[new_key] = data[old_key]
        del data[old_key]

    if is_std:
        data = _get_display_value(data)

    if len(data["userValue"]) > 0 and data["userValue"][-1] is None:
        for _, value in data.items():
            value.pop()

    data["success"] = [
        data["userValue"][i] == data["displayedValue"][i]
        for i in range(len(data["userValue"]))
    ]

    return parse_raw_data_set(data, definition_user_input)


def get_level_id(config: dict) -> LevelId:
    """Parse level id from level type and configuration.

    Parameters
    ----------
    config
        The level configuration

    Returns
    -------
    LevelId
        Level id for the level
    """
    if config["mode"] == "symbol-to-digit":
        context = ["symbol_to_digit"]
    elif config["mode"] == "digit-to-digit":
        context = ["digit_to_digit"]
    return LevelId(context)
