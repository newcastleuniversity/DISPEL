"""Function for converting Drawing test BDH JSON files into a reading."""
from copy import deepcopy
from typing import Any, Dict

import pandas as pd

from dispel.data.levels import LevelId
from dispel.data.raw import RawDataSet, RawDataSetDefinition, RawDataValueDefinition
from dispel.data.values import DefinitionId
from dispel.providers.bdh.io.core import KEYS, parse_raw_data_set


def _translate_figure(ctx: str) -> str:
    ref = {"rectangle": "square", "clockwise": "clock", "infinity_loop": "infinity"}
    for old, new in ref.items():
        ctx = ctx.replace(old, new)
    return ctx


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
    figure = _translate_figure(config[KEYS.drawing_figure_name])
    # For the drawing activity, the level_id id hand_shape_attempt
    context = [figure, config[KEYS.drawing_hand]]
    return LevelId(context)


def convert_touch_events(
    data: Dict[str, Any], definition: RawDataSetDefinition
) -> RawDataSet:
    """Convert touch events dataset to screen format."""
    actions = {"start": "down", "end": "up", "move": "move"}
    ref = {
        "x": "xPosition",
        "y": "yPosition",
        "timestamp": "tsTouch",
        "action": "touchAction",
        "touch_path_id": "touchPathId",
    }
    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        unit=definition.get_value_definition(DefinitionId("timestamp")).unit,
    )
    data["action"] = [actions[action] for action in data["action"]]
    for old_key, new_key in ref.items():
        if old_key in data:
            data[new_key] = data[old_key]
            del data[old_key]
    _definitions = deepcopy(definition).value_definitions
    new_definitions = []
    for value_definition in _definitions:
        try:
            value_definition.id = ref[value_definition.id]
        except KeyError:
            pass
        new_definitions.append(value_definition)
    if "pressure" not in data:
        data["pressure"] = 0
        new_definitions.append(RawDataValueDefinition("pressure", "pressure"))
    definition_screen = RawDataSetDefinition(
        "screen", definition.source, new_definitions, is_computed=True
    )
    return parse_raw_data_set(data, definition_screen)
