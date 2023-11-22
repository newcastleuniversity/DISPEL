"""Functions for reading BDH formatted mood data."""
from copy import deepcopy
from typing import Any, Dict, List

from dispel.data.levels import LevelId
from dispel.data.raw import RawDataSet, RawDataSetDefinition
from dispel.providers.bdh.io.core import KEYS, convert_dataset, parse_raw_data_set


def _split_answers(answers: Dict[str, List[Any]]):
    mood_answers = {}
    physical_answers = {}
    for key, value in answers.items():
        mood_value = []
        physical_value = []
        for i, v in enumerate(value):
            if answers["displayed_question_id"][i] == 1:
                mood_value.append(v)
            else:
                physical_value.append(v)
        mood_answers[key] = mood_value
        physical_answers[key] = physical_value
    return mood_answers, physical_answers


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

    Raises
    ------
    NotImplementedError
        If the given mode parsing has not been implemented.
    """
    if config["idMoodscale"] == "mood":
        return LevelId("mood")
    if config["idMoodscale"] == "physicalState":
        return LevelId("physical_state")

    raise NotImplementedError(f"Level Id is not implemented for mode: {config['mode']}")


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
    data[KEYS.levels].append(deepcopy(data[KEYS.levels][0]))
    answers = data[KEYS.levels][0]["raw_data"]["answers"]
    mood_answers, physical_answers = _split_answers(answers)
    flagged_answers = data[KEYS.levels][0]["raw_data"]["validated_answers"]
    mood_flagged_answers, physical_flagged_answers = _split_answers(flagged_answers)

    data[KEYS.levels][0]["name"] = "mood"
    data[KEYS.levels][0]["config"] = "mood"
    data[KEYS.levels][0]["configuration"] = {"idMoodscale": "mood"}
    data[KEYS.levels][0]["raw_data"]["answers"] = mood_answers
    data[KEYS.levels][0]["raw_data"]["validated_answers"] = mood_flagged_answers

    data[KEYS.levels][1]["raw_data"]["answers"] = physical_answers
    data[KEYS.levels][1]["raw_data"]["validated_answers"] = physical_flagged_answers
    data[KEYS.levels][1]["configuration"] = {"idMoodscale": "physicalState"}
    data[KEYS.levels][1]["name"] = "physical_state"
    data[KEYS.levels][1]["config"] = "physical_state"

    return data


def convert_flagged_answers(
    data: Dict[str, Any], definition: RawDataSetDefinition
) -> RawDataSet:
    """Convert flagged_answers dataset to userInput format."""
    ref = {
        "response_id": "answer",
        "presentation_timestamp": "tsDisplay",
        "response_timestamp": "tsAnswer",
    }
    data, new_definitions = convert_dataset(data, definition, ref)
    definition_user_input = RawDataSetDefinition(
        "userInput", definition.source, new_definitions, is_computed=True
    )

    return parse_raw_data_set(data, definition_user_input)
