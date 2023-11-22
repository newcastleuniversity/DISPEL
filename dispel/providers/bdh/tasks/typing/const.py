"""Modalities and constants used in typing module."""
from enum import Enum

import pandas as pd

from dispel.data.raw import RawDataValueDefinition

STR_TO_CALLABLE = {
    "mean": pd.DataFrame.mean,
    "median": pd.DataFrame.median,
    "std": pd.DataFrame.std,
}

N_LEVELS = 20
r"""A fixed unreachable number of difficulty level in typing."""

DEF_AUTOCOMPLETE_KEY = RawDataValueDefinition(
    id_="autocomplete_per_key",
    name="autocomplete per key",
    description="A boolean indicating if the key typed is an autocompletion",
    data_type="bool",
)

DEF_AUTOCOMPLETE_WORD = RawDataValueDefinition(
    id_="autocomplete_per_word",
    name="autocomplete per word",
    description="A bool indicating if a word was typed using autocompletion",
    data_type="bool",
)

DEF_SUBMISSION_STATE = RawDataValueDefinition(
    id_="submission_state",
    name="submission state",
    description="The submission state at the time of the key pressed. Available states "
    "are the following: ``Correct``, ``Incorrect``, and ``Correcting``. The state is "
    "``Correct`` if the textbox is empty or if it forms a subpart of the word displayed"
    " on the screen. The state is incorrect whenever the user types a key (that is not "
    "backspace), leading to a textbox that does not form a subpart of the word "
    "displayed on the screen. Finally, the state is identified as correcting whenever "
    "the user corrects its mistake by typing backspace; it can be seen as an "
    "intermediate state between an incorrect and a correct state.",
    data_type="bool",
)

DEF_SUCCESS = RawDataValueDefinition(
    id_="success",
    name="success",
    description="A boolean indicating if the word was writtensuccessfully.",
    data_type="bool",
)

DEF_DISPLAYED_WORD = RawDataValueDefinition(
    id_="displayed_word",
    name="displayed word",
    description="The word displayed on the screen. This is the word the user has to "
    "type.",
    data_type="str",
)


DEF_INCORRECT_STATE_DURATION = RawDataValueDefinition(
    id_="duration",
    name="incorrect state duration",
    description="The time spent in incorrect states.",
    data_type="float",
    unit="s",
)


DEF_DISPLAYED_INPUT = RawDataValueDefinition(
    id_="displayed_input",
    name="displayed input",
    description="The textbox input, this corresponds to the letters that have been "
    "typed and appear in the small textbox.",
    data_type="str",
)

DEF_STATE_DUR = RawDataValueDefinition(
    id_="duration",
    name="Duration of a given state.",
    description="",
    unit="s",
    data_type="float",
)

DEF_KEY = RawDataValueDefinition(
    id_="key", name="key", description="The key that has been typed.", data_type="str"
)

DEF_T_BETWEEN_STATE = RawDataValueDefinition(
    id_="t_between_states",
    name="Time between states.",
    description="Time elapsed between two states.",
    unit="s",
    data_type="float",
)

DEF_REACTION_TIME = RawDataValueDefinition(
    id_="reaction_time",
    name="reaction time",
    unit="s",
    description="The interval between the appearance of a word and the moment the user "
    "releases the first key.",
    data_type="float",
)

DEF_REACTION_TIME_FC = RawDataValueDefinition(
    id_="reaction_time_first_correct",
    name="reaction time first correct",
    unit="s",
    description="The interval between the appearance of a word and the moment the user "
    "releases the first correct letter key.",
    data_type="float",
)

DEF_KEY_INTERVALS = RawDataValueDefinition(
    id_="key_intervals",
    name="key intervals",
    unit="s",
    description="The time intervals between key pressed.",
    data_type="float",
)

DEF_WORD_DURATION = RawDataValueDefinition(
    id_="word_duration",
    name="word duration",
    unit="s",
    description="The time elapsed between the appearance and disappearance of a word.",
    data_type="float",
)

DEF_DIFFICULTY_LEVEL = RawDataValueDefinition(
    id_="difficulty_level",
    name="difficulty level",
    description="The difficulty level is an integer starting at zero that increments "
    "by one every time a user successfully writes three words. These three words does "
    "not have to be in a row.",
    data_type="int",
)

DEF_MEAN_WORD_DURATION = RawDataValueDefinition(
    id_="mean", name="mean word duration", unit="s", data_type="float"
)

DEF_MEAN_MEDIAN_WORD_DURATION = RawDataValueDefinition(
    id_="median",
    name="median word duration",
    unit="s",
    data_type="float",
)

DEF_STD_WORD_DURATION = RawDataValueDefinition(
    id_="std",
    name="standard deviation of word duration",
    unit="s",
    data_type="float",
)


DEF_WORD_DUR_DIFF = RawDataValueDefinition(
    id_="difference",
    name="difference",
    description="The difference of the word duration per difficulty level. It is the "
    "derivative of the mean word duration w.r.t the difficulty level.",
    data_type="float",
)

DEF_REACTION_DURATION = RawDataValueDefinition(
    id_="duration",
    name="reaction duration for correct words",
    description="The reaction duration for a correct word.",
    unit="s",
    data_type="float",
)


DEF_CORRECTING_DURATION = RawDataValueDefinition(
    id_="duration",
    name="correcting state duration",
    description="The time spent in correcting states.",
    data_type="float",
    unit="s",
)

DEF_CORRECTING_CORRECT_DURATION = RawDataValueDefinition(
    id_="duration",
    name="correcting state duration for correct words.",
    description="The time spent in correcting states for acorrect word.",
    data_type="float",
    unit="s",
)

DEF_COUNT_CONSEC_STATES = RawDataValueDefinition(
    id_="count",
    name="count consecutive equal states",
    description="number of consecutive submission with the same correct_submission "
    "state.",
    data_type="int",
)

DEF_IS_ERROR_FREE = RawDataValueDefinition(
    id_="is_error_free",
    name="is error free",
    description="A boolean indicating if the word displayed on screen has been typed "
    "without error.",
    data_type="bool",
)

DEF_REACTING_TIME = RawDataValueDefinition(
    id_="t_between_states",
    name="reacting time",
    description="The reacting time is the time elapsed between an incorrect state and "
    "a correcting state in other words, the time to react to a mistake.",
    data_type="float",
    unit="s",
)

DEF_REACTING_TIME_CORRECT = RawDataValueDefinition(
    id_="reacting_time",
    name="reacting time correct",
    description="The reacting time for correct words.",
    unit="s",
    data_type="float",
)


DEF_IS_ERROR_FREE = RawDataValueDefinition(
    id_="is_error_free",
    name="is error free",
    description="A boolean indicating if the word displayed on screen has been typed "
    "without error.",
    data_type="bool",
)

DEF_APPEARANCE_TS = RawDataValueDefinition(
    id_="appearance_timestamp",
    name="appearance timestamp",
    description="Timestamp corresponding to the appearance of the word.",
)

DEF_SIMILARITY_RATIO = RawDataValueDefinition(
    id_="similarity_ratio",
    name="similarity ratio",
    description="Similarity ratio between the word being typed and the word on screen. "
    "It is computed with the sequence matcher from difflib.",
    data_type="str",
)

DEF_LETTER_TYPED_OVER_LEN = RawDataValueDefinition(
    id_="letters_typed_over_length",
    name="letters typed over length",
    description="The number of letters typed divided by the length of the word for "
    "each completed word.",
    data_type="str",
)

DEF_IS_LETTER = RawDataValueDefinition(
    id_="key_is_letter",
    name="key pressed is letter",
    description="Boolean indicating if a key pressed is a letter.",
    data_type="bool",
)

DEF_MAX_DEVIATION_LETTER_INTERVAL = RawDataValueDefinition(
    id_="max_deviation_letter_interval",
    name="maximum deviation letter interval",
    description="The maximum deviation of the letter interval for each word. The "
    "deviation is computed as the squared error between the current interval and the "
    "mean of these intervals within a word.",
    data_type="float",
    unit="s",
)

DEF_WORD_ID = RawDataValueDefinition(
    id_="word_id",
    name="word id",
    description="The id of the word that is being typed.",
    data_type="float",
)

DEF_TS_OUT = RawDataValueDefinition(
    id_="timestamp_out",
    name="timestamp out",
    description="The timestamp when a key has been released.",
    data_type="datetime64[ns]",
)


class WordState(Enum):
    """Generic state for a word."""

    INIT = 0
    CORRECT = 1
    INCORRECT = 2


class KeyState(Enum):
    """Generic state for a key."""

    INIT = 0
    CORRECT = 1
    INCORRECT = 2
    CORRECTING = 3
