"""Functions for reading BDH formatted mood data."""
from copy import deepcopy
from dataclasses import replace
from typing import Any, Dict, List

from dispel.data.levels import LevelId
from dispel.data.raw import RawDataValueDefinition
from dispel.providers.bdh.io.core import KEYS

SIZE_NAMES = ["small", "medium", "large", "extra_large"]


def _split_dataset(raw_data: Dict[str, List], begin: int, end: int) -> Dict[str, List]:
    """Split raw data into levels.

    Parameters
    ----------
    raw_data
        Dictionary of raw data
    begin
        Raw timestamp of the appearance of the balloon
    end
        Raw timestamp of the bursting of the balloon

    Returns
    -------
    Dict[str, List]
        Raw data for the level
    """
    assert (
        "touch_path_id" not in raw_data.keys()
    ), "Please use _split_touches for touch events."
    new_data: Dict[str, List] = {}
    for key in raw_data:
        new_data[key] = []
        try:
            timestamp = raw_data["timestamp"]
        except KeyError:
            try:
                timestamp = raw_data["first_finger_press_in_timestamp"]
            except KeyError:
                timestamp = raw_data["appearance_timestamp"]
        for i, time in enumerate(timestamp):
            if time >= begin:
                if end is not None and time <= end:
                    new_data[key].append(raw_data[key][i])
                else:
                    break
    return new_data


def _split_touches(raw_data: Dict[str, List], begin: int, end: int) -> Dict[str, List]:
    """Split touch events into levels.

    We need to treat touch event differently because a touch action started on one level
    might "leak" into the next one. E.g. one finger pinching might be lifted after the
    appearance of the next balloon.

    Parameters
    ----------
    raw_data
        Dictionary of raw touch data
    begin
        Raw timestamp of the appearance of the balloon
    end
        Raw timestamp of the bursting of the balloon

    Returns
    -------
    Dict[str, List]
        Touch data for the level
    """
    new_data: Dict[str, List] = {}
    timestamp = raw_data["timestamp"]
    for key in raw_data:
        new_data[key] = []
        touch_path_ids = set()
        for i, time in enumerate(timestamp):
            if time >= begin:
                if end is not None and time <= end:
                    if raw_data["action"][i] == "start":
                        touch_path_ids.add(raw_data["touch_path_id"][i])
                    if raw_data["touch_path_id"][i] in touch_path_ids:
                        new_data[key].append(raw_data[key][i])
                elif raw_data["touch_path_id"][i] in touch_path_ids:
                    if raw_data["action"][i] == "start":
                        touch_path_ids.remove(raw_data["touch_path_id"][i])
                    else:
                        new_data[key].append(raw_data[key][i])
                else:
                    break
    new_data = _reassign_touch_path_id(new_data)
    return new_data


def _reassign_touch_path_id(data: Dict[str, List]) -> Dict[str, List]:
    """Reassign touch path ids.

    Touch path ids are sometimes when there are no fingers on the screen. This function
    makes sure consecutive touches are assigned new ids.

    Parameters
    ----------
    data
        Dictionary of raw touch data

    Returns
    -------
    Dict[str, List]
        Touch data for the level
    """
    alt_ids: Dict[int, int] = {}
    current_max = 0
    new_data = deepcopy(data)
    timestamp = new_data["timestamp"]
    for i in range(len(timestamp)):
        touch_path_id = new_data["touch_path_id"][i]
        if new_data["action"][i] == "start":
            if touch_path_id in alt_ids:
                while current_max in alt_ids:
                    current_max += 1
                alt_ids[touch_path_id] = current_max
                alt_ids[current_max] = current_max
            else:
                alt_ids[touch_path_id] = touch_path_id
                current_max = max(current_max, touch_path_id)
        new_data["touch_path_id"][i] = alt_ids[touch_path_id]
    return new_data


def _add_success_and_validity(
    touch: Dict[str, List], pinches: Dict[str, List]
) -> Dict[str, List]:
    success = [True for _ in touch["x"]]
    valid = [False for _ in touch["x"]]
    for i, pinch in enumerate(pinches["first_finger_press_in_timestamp"]):
        begin = pinch
        for j, timestamp in enumerate(touch["timestamp"]):
            if timestamp >= begin:
                success[j] = pinches["successful"][i]
        if (
            pinches["second_finger_press_out_timestamp"][i] is not None
            or pinches["successful"][i]
        ):
            end = pinches["second_finger_press_out_timestamp"][i]
            if end is None:
                end = len(success)
            for j, timestamp in enumerate(touch["timestamp"]):
                if begin <= timestamp <= end:
                    valid[j] = True
    touch["ledToSuccess"] = success
    touch["isValidPinch"] = valid
    return touch


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
    context = [config["hand"], config["bubbleSize"]]
    return LevelId(context)


def create_levels(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create levels from uni-level activity data.

    Parameters
    ----------
    data
        Level data

    Returns
    -------
    Dict[str, Any]
        Level data
    """
    new_levels = []
    for level in data[KEYS.levels]:
        hand = level["name"]
        balls = level["raw_data"]["balls"]
        # Support for indexing starting at 1
        radius_level_set = set(balls["radius_level"])
        if radius_level_set != {0, 1, 2, 3}:
            min_radius_level = min(balls["radius_level"])
            balls["radius_level"] = [
                radius - min_radius_level for radius in balls["radius_level"]
            ]
        for i, size in enumerate(balls["radius_level"]):
            begin = balls["appearance_timestamp"][i]
            end = balls["burst_timestamp"][i]
            new_level = deepcopy(level)
            new_level["configuration"]["bubbleSize"] = SIZE_NAMES[size]
            new_level["name"] = f"{hand}-{SIZE_NAMES[size]}"
            new_level["raw_data"] = {}
            for key, value in level["raw_data"].items():
                if key == "touch_events":
                    new_level["raw_data"][key] = _split_touches(value, begin, end)
                else:
                    new_level["raw_data"][key] = _split_dataset(value, begin, end)
            new_level["configuration"]["targetRadius"] = balls["radius"][i]
            new_level["configuration"]["xTargetBall"] = balls["center_x"][i]
            new_level["configuration"]["yTargetBall"] = balls["center_y"][i]
            new_level["configuration"]["usedHand"] = new_level["configuration"]["hand"]
            new_level["raw_data"]["touch_events"] = _add_success_and_validity(
                new_level["raw_data"]["touch_events"], new_level["raw_data"]["pinches"]
            )

            new_level["effective_time_frame"]["begin_timestamp"] = begin
            new_level["effective_time_frame"]["end_timestamp"] = end

            new_levels.append(new_level)

    data[KEYS.levels] = new_levels
    return data


def update_raw_data_definition(definitions: Dict[str, Any]) -> Dict[str, Any]:
    """Update raw data definitions.

    Parameters
    ----------
    definitions
        Raw data definitions

    Returns
    -------
    Dict[str, Any]
        Level data

    """
    old_definition = definitions["touch_events"]
    old_value_definitions = list(old_definition.value_definitions)

    old_value_definitions.append(RawDataValueDefinition("ledToSuccess", "ledToSuccess"))
    old_value_definitions.append(RawDataValueDefinition("isValidPinch", "isValidPinch"))

    new_definition = replace(
        old_definition, value_definitions_list=old_value_definitions
    )
    definitions["touch_events"] = new_definition

    return definitions
