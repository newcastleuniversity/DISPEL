"""Functions for reading BDH formatted msis29 data."""
from copy import deepcopy
from typing import Any, Dict

from dispel.data.levels import LevelId
from dispel.data.raw import RawDataSet, RawDataSetDefinition
from dispel.providers.bdh.io.core import KEYS, parse_raw_data_set


def get_level_id(_: dict) -> LevelId:
    """Parse level id from level type and configuration."""
    return LevelId("msis29")


def create_levels(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create levels from uni-level activity data for MSIS29.

    Parameters
    ----------
    data
        Level data

    Returns
    -------
    Dict[str, Any]
        Dictionary containing newly created levels.

    """
    answers = data[KEYS.levels][0]["raw_data"]["activity_sequence"]["response_id"]
    questions = data[KEYS.levels][0]["raw_data"]["activity_sequence"][
        "displayed_question_id"
    ]
    question_timestamp = data[KEYS.levels][0]["raw_data"]["activity_sequence"][
        "presentation_timestamp"
    ]

    for _ in range(len(answers) - 1):
        data[KEYS.levels].append(deepcopy(data[KEYS.levels][0]))

    for i, level in enumerate(data[KEYS.levels]):
        level["name"] = "msis29"
        level["configuration"] = {"idMsis29": f"{i}"}
        level["raw_data"]["activity_sequence"]["response_id"] = answers[i]
        level["raw_data"]["activity_sequence"]["displayed_question_id"] = questions[i]
        level["raw_data"]["activity_sequence"][
            "presentation_timestamp"
        ] = question_timestamp[i]
        for cat in ["response_timestamp", "response_text", "displayed_question_text"]:
            level["raw_data"]["activity_sequence"][cat] = level["raw_data"][
                "activity_sequence"
            ][cat][i]
        level["effective_time_frame"] = {
            "begin_timestamp": question_timestamp[i],
            "end_timestamp": level["raw_data"]["activity_sequence"][
                "response_timestamp"
            ],
        }

    return data


def convert_activity_sequence(
    data: Dict[str, Any], definition: RawDataSetDefinition
) -> RawDataSet:
    """Convert activity sequence dataset to userInput."""
    _definitions = deepcopy(definition).value_definitions
    new_definitions = []

    ref = {
        "presentation_timestamp": "tsDisplay",
        "response_timestamp": "tsAnswer",
        "displayed_question_id": "displayedValue",
        "response_id": "answer",
    }

    for value_definition in _definitions:
        try:
            value_definition.id = ref[value_definition.id]
            value_definition.name = ref[value_definition.name]
        except KeyError:
            pass
        new_definitions.append(value_definition)

    for old_key, new_key in ref.items():
        data[new_key] = data[old_key]
        del data[old_key]

    definition_user_input = RawDataSetDefinition(
        "userInput", definition.source, new_definitions, is_computed=True
    )

    return parse_raw_data_set(data, definition_user_input)
