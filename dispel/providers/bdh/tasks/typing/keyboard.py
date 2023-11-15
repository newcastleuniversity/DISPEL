"""Keyboard functions to extract relevant properties for Typing."""
from difflib import SequenceMatcher
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from unidecode import unidecode

from dispel.providers.bdh.tasks.typing.const import KeyState


def transform_reaction_time(
    word: pd.DataFrame, key_typed: pd.DataFrame
) -> pd.DataFrame:
    """Compute the reaction time and the first correct letter reaction time.

    The reaction time is the time elapsed between the appearance of a word
    and the time the user typed a letter. The first correct letter reaction
    time follows the same definition but measures the time elapsed until a
    correct letter is typed.

    Parameters
    ----------
    word : pd.DataFrame
        A data frame listing all the words displayed on screen during the
        test with the following columns: ``words``, ``appearance_timestamp``
        and ``disappearance_timestamp``.

    key_typed : pd.DataFrame
        The dataframe with the list of key pressed by the user,
        it should contain the columns: ``timestamp_out`` and ``key``.

    Returns
    -------
    pd.DataFrame
        A DataFrame with two columns: the reaction time and the
        reaction time to the first correct letter.
    """

    def _compute_reaction_time(
        row: pd.Series, key_typed: pd.DataFrame, first_correct: bool = False
    ) -> pd.DataFrame:
        # Mask df_key between start and end
        mask = (key_typed["timestamp_out"] > row["appearance_timestamp"]) & (
            key_typed["timestamp_out"] <= row["disappearance_timestamp"]
        )
        if not first_correct:
            return (
                key_typed.loc[mask, "timestamp_out"].min() - row["appearance_timestamp"]
            ).total_seconds()

        # Find the first correct letter in the masked keys
        condition = key_typed.loc[mask, "key"] == row["word"][0]
        # Take the first timestamp flagging this condition
        return (
            key_typed.loc[mask][condition]["timestamp_out"].min()
            - row["appearance_timestamp"]
        ).total_seconds()

    return pd.DataFrame(
        {
            "reaction_time": word.apply(
                lambda x: _compute_reaction_time(x, key_typed), axis=1
            ),
            "reaction_time_first_correct": word.apply(
                lambda x: _compute_reaction_time(x, key_typed, True), axis=1
            ),
        }
    ).set_index(word["appearance_timestamp"])


def compute_rt_slope(df_reaction_time: pd.DataFrame, col: str) -> float:
    """Compute the mean of the reaction time slope.

    Parameters
    ----------
    df_reaction_time : pd.DataFrame
        The reaction time dataframe (first letter and first correct letter)

    col : String
        The column to be used : reaction_time or reaction_time_first_correct

    Returns
    -------
    float
        A float which is the mean slope of the reaction time (first letter
         or first correct letter)
    """
    return (
        df_reaction_time[col].diff()
        / df_reaction_time.index.to_series().diff().dt.total_seconds()
    ).mean()


def total_words_typed(word: pd.DataFrame) -> int:
    """Find total number of words typed.

    The total number of words typed successfully is the number of True values
    in the columns ``success``.

    Parameters
    ----------
    word: pd.DataFrame
        A data frame containing the word information with the column
        ``success``.

    Returns
    -------
    int
        Number of words successfully typed.
    """
    return word["success"].sum()


def time_intervals(data: pd.DataFrame, interval_column: Optional[str]) -> pd.DataFrame:
    """Find time intervals between consecutive element of a data frame.

    Parameters
    ----------
    data: pd.DataFrame
        Any data frame with a timestamp index.
    interval_column: Optional[str]
        When specified, indicates that the interval should be computed with
        values from a column and not the index.

    Returns
    -------
    pd.Series
        Time intervals between samples.
    """
    if not interval_column:
        return data.index.to_series().diff().dt.total_seconds()
    return data[interval_column].diff().dt.total_seconds()


def word_duration(word: pd.DataFrame) -> pd.DataFrame:
    """Compute the time spent per word.

    Parameters
    ----------
    word: pd.DataFrame
        A data frame containing the word information with the column
        ``success``.

    Returns
    -------
    pd.Series
        A data frame containing the time spent per word.
    """
    return (
        word["disappearance_timestamp"] - word["appearance_timestamp"]
    ).dt.total_seconds()


def get_submission_state(word: pd.DataFrame, key_typed: pd.DataFrame) -> pd.DataFrame:
    """Compute the submission state.

    Create a data frame that contains the state at each time a user types a
    key. Available states are the following: ``Correct``, ``Incorrect``, and
    ``Correcting``. The state is ``Correct`` if the textbox is empty or if
    it forms a subpart of the word displayed on the screen. The state is
    incorrect whenever the user types a key (that is not backspace), leading
    to a textbox that does not form a subpart of the word displayed on the
    screen. Finally, the state is identified as correcting whenever the user
    corrects its mistake by typing backspace; it can be seen as an
    intermediate state between an incorrect and a correct state.

    Parameters
    ----------
    word : pd.DataFrame
        A data frame listing all the words displayed on screen during the
        test with the following columns: ``words``, ``appearance_timestamp``
        and ``disappearance_timestamp``.

    key_typed : pd.DataFrame
        The dataframe with the list of key pressed by the user,
        it should contain the columns: ``timestamp_out`` and ``key``.

    Returns
    -------
    pd.DataFrame
        A data frame that contains the user state after each keystroke.
    """

    def _displayed_word(timestamp):
        """Find the word displayed on screen."""
        return word["word"].values[
            (timestamp < word["disappearance_timestamp"])
            & (timestamp > word["appearance_timestamp"])
        ][0]

    input_and_screen = pd.DataFrame(
        {
            "displayed_input": key_typed["displayed_input"],
            "displayed_word": key_typed["timestamp_out"].apply(_displayed_word),
        }
    )

    is_correct = input_and_screen.apply(
        lambda x: unidecode(x["displayed_word"]).startswith(x["displayed_input"]),
        axis=1,
    )

    backspace_pressed = key_typed["key"] == "Backspace"
    word_changed = input_and_screen["displayed_word"] != input_and_screen[
        "displayed_word"
    ].shift(1)
    word_changed[0] = False

    # update the states
    submission_state: List[KeyState] = [KeyState.INIT] * len(is_correct)

    for i in range(len(is_correct)):
        if word_changed.iloc[i] or is_correct.iloc[i]:
            submission_state[i] = KeyState.CORRECT
            continue
        if not is_correct.iloc[i - 1] and backspace_pressed.iloc[i]:
            submission_state[i] = KeyState.CORRECTING
            continue
        if not is_correct.iloc[i]:
            submission_state[i] = KeyState.INCORRECT

    res = pd.DataFrame(
        {
            "submission_state": submission_state,
            "displayed_word": input_and_screen["displayed_word"],
            "displayed_input": input_and_screen["displayed_input"],
            "key": key_typed["key"],
        }
    )

    res["success"] = res["displayed_word"].apply(
        lambda x: word.loc[word["word"] == x, "success"].values[0]
    )

    return res.set_index(key_typed["timestamp_out"])


def detect_key_autocompletion(submission_state: pd.DataFrame) -> pd.DataFrame:
    """Detect the autocompletion at the key level.

    Create a data frame that contains all the typed keys
    the autocompletion status.

    Parameters
    ----------
    submission_state : pd.DataFrame
        A data frame that contains the user state after each keystroke.

    Returns
    -------
    pd.DataFrame
        An new dataset with the displayed word, all the key typed and
        autocompletion information.
    """
    submission_state_copy = submission_state[["displayed_word", "key"]].copy()

    # Detect the non autocompleted keys
    to_keep = (submission_state_copy["key"].str.len() == 1) | (
        submission_state_copy["key"].str.match("Backspace")
    )

    submission_state_copy["autocomplete_per_key"] = ~to_keep

    return submission_state_copy


def detect_word_autocompletion(keys_with_autocompletion: pd.DataFrame) -> pd.DataFrame:
    """Detect the autocompletion for all the words.

    Parameters
    ----------
    keys_with_autocompletion : pd.DataFrame
        A data frame that .

    Returns
    -------
    pd.DataFrame
        The word dataset without the autocompleted words
    """
    # Detect the words with autocompletion from the keys
    per_word_autocompletion = keys_with_autocompletion.groupby(
        "displayed_word", sort=False
    ).apply(lambda x: x["autocomplete_per_key"].any())

    return pd.DataFrame(per_word_autocompletion, columns=["autocomplete_per_word"])


def total_autocomplete(keys_with_autocompletion: pd.DataFrame) -> int:
    """Get the total number of autocompletions.

    Parameters
    ----------
    keys_with_autocompletion: pd.DataFrame
        A dataframe indicating whether a key was autocompleted or not.

    Returns
    -------
    int
        Number of autocompletions performed by the user.
    """
    return keys_with_autocompletion["autocomplete_per_key"].sum()


def get_state_durations(
    submission_state: pd.DataFrame,
    key_intervals: pd.DataFrame,
    word: pd.DataFrame,
) -> pd.DataFrame:
    """Return submission state duration.

    To compute the state duration, we group by displayed word and
    by submission state. Indeed, we want to compute statistics
    per word on the time spent in a given state or between states.

    Parameters
    ----------
    submission_state : pd.DataFrame
        Data frame containing the submission state each time a user
        types a key.
    key_intervals : pd.DataFrame
        Time intervals between samples
    word : pd.DataFrame
        A data frame listing all the words displayed on screen during
        the test with the following columns: ``words``,
        ``appearance_timestamp`` and ``disappearance_timestamp``.

    Returns
    -------
    pd.DataFrame
        A data frame that contains duration of each correcting state.
    """
    # We group by displayed word and submission_state to keep track of
    # whether the state corresponds to a successfully typed word or
    # not. Otherwise, two correctly typed words would just regroup into
    # one `Correct` state.
    temp_df = submission_state.copy()
    intervals = key_intervals["key_intervals"].values
    temp_df["key_intervals"] = intervals
    temp_df["success"] = submission_state["displayed_word"].apply(
        lambda x: word.loc[word["word"] == x, "success"].values[0]
    )
    t_between_states = submission_state.index.to_series().diff().dt.total_seconds()
    temp_df["t_between_states"] = t_between_states
    temp_df["index"] = submission_state.index
    groups = temp_df.groupby(
        [
            "displayed_word",
            (
                submission_state["submission_state"].shift()
                != submission_state["submission_state"]
            ).cumsum(),
        ]
    )
    res = pd.DataFrame(groups)
    res["submission_state"] = groups.submission_state.apply(lambda x: x.iloc[0]).values
    res["duration"] = groups.sum()["key_intervals"].values
    res["success"] = groups.success.apply(lambda x: x.all()).values
    res["t_between_states"] = groups.t_between_states.apply(lambda x: x.iloc[0]).values
    res["displayed_word"] = groups.displayed_word.apply(lambda x: x.iloc[0]).values
    res["index"] = groups.index.apply(lambda x: x.iloc[0]).values
    # These columns were created by the groupby object being converted
    # into a dataframe.
    res.drop([0, 1], axis=1, inplace=True)
    return res.set_index("index").sort_index()


def get_reaction_duration(state_durations: pd.DataFrame) -> pd.DataFrame:
    """Compute the reaction duration.

    The reaction duration is the time elapsed in an incorrect state
    for a correct submission.

    Parameters
    ----------
    state_durations : pd.DataFrame
        Data frame containing submission states and duration.

    Returns
    -------
    pd.DataFrame
        A data frame that contains reaction duration.
    """
    return state_durations.loc[
        state_durations["submission_state"] == KeyState.INCORRECT,
        ["duration", "success"],
    ]


def get_correct_reaction_duration(
    reaction_duration: pd.DataFrame,
) -> pd.DataFrame:
    """Return reaction durations for correct words.

    Parameters
    ----------
    reaction_duration : pd.DataFrame
        Data frame containing reacting durations.

    Returns
    -------
    pd.DataFrame
        A data frame that contains reaction duration for correct
        words.
    """
    return reaction_duration.loc[reaction_duration["success"], "duration"]


def get_correcting_duration(state_durations: pd.DataFrame) -> pd.DataFrame:
    """Compute correcting duration.

    The Correcting Duration is the time elapsed in a correcting state
    for a correct submission, in other words, the time spent correcting
    a mistake.

    Parameters
    ----------
    state_durations : pd.DataFrame
        Data frame containing submission states and duration.

    Returns
    -------
    pd.DataFrame
        A data frame that contains correcting duration.
    """
    return state_durations.loc[
        state_durations["submission_state"] == KeyState.CORRECTING,
        ["duration", "success"],
    ]


def get_correct_correcting_duration(
    correcting_duration: pd.DataFrame,
) -> pd.DataFrame:
    """Return correcting duration for correct words.

    Parameters
    ----------
    correcting_duration : pd.DataFrame
        Data frame containing correcting durations.

    Returns
    -------
    pd.DataFrame
        A data frame that contains correcting duration for correct
        words.
    """
    return correcting_duration.loc[correcting_duration["success"], "duration"]


def get_reacting_times(state_durations: pd.DataFrame) -> pd.DataFrame:
    """Return reacting times.

    The reacting time is the time elapsed between an incorrect state
    and a correcting state for a correct submission, in other words,
    the time to react to a mistake.

    Parameters
    ----------
    state_durations : pd.DataFrame
        Data frame containing submission states and duration.

    Returns
    -------
    pd.DataFrame
        A data frame containing reacting times.
    """
    return state_durations.loc[
        state_durations["submission_state"] == KeyState.CORRECTING,
        ["t_between_states", "success"],
    ]


def get_correct_reacting_time(reacting_times: pd.DataFrame) -> pd.DataFrame:
    """Return reacting times for correct words.

    Parameters
    ----------
    reacting_times : pd.DataFrame
        Data frame containing correction differences.

    Returns
    -------
    pd.DataFrame
        A data frame that contains reacting duration for correct words.
    """
    return reacting_times.loc[reacting_times["success"], "t_between_states"]


def find_consec_element_and_count(data: pd.DataFrame) -> pd.DataFrame:
    """Find consecutive elements and count them.

    Parameters
    ----------
    data : pd.DataFrame
        Data frame to compute feature related to successfully written words
        with(out) error, and also a streak of words with(out) error(s).

    Returns
    -------
    pd.DataFrame
        A data frame with three columns: the count, the submission state and if
        the sequence was free of error
    """
    res = data.copy()
    # Assign a group to each consecutive elements
    res["subgroup"] = (res["is_error_free"] != res["is_error_free"].shift(1)).cumsum()
    group = res.groupby("subgroup")
    return pd.DataFrame(
        {
            "count": group["is_error_free"].count(),
            "success": group["success"].any(),
            "is_error_free": group["is_error_free"].any(),
        }
    )


def count_words_typed_in_row(data: pd.DataFrame, error_free: bool = True) -> int:
    """Count the successfully typed words with(out) error in a row.

    The parameter ``error_free`` indicates if we look at consecutive words that has
    been successfully typed and: without error (True) or with error (False).

    Parameters
    ----------
    data
        Dataframe containing typing data
    error_free
        If True, only count words typed with no error

    Returns
    -------
        Number of words typed in a row
    """
    mask = data.is_error_free
    if not error_free:
        mask = ~mask
    res = data.loc[data.success & mask, "count"]
    if len(res) > 0:
        return res.max()
    return 0


def similarity_ratio(x: Union[pd.Series, pd.DataFrame]) -> float:
    """Compute the similarity ratio."""
    a, b = x.values
    return SequenceMatcher(None, a, b).ratio()


def apply_similarity_ratio(submission_state: pd.DataFrame) -> pd.Series:
    """Apply similarity ratio.

    The similarity ratio is computed between the word being typed
    ``displayed_input`` and the word on screen ``displayed_word``.

    Parameters
    ----------
    submission_state
        A data frame that contains the user state after each keystroke.

    Returns
    -------
        Similarity ratio between the displayed input and
        displayed_word.
    """
    res = submission_state[["displayed_input", "displayed_word"]].copy()
    res["similarity_ratio"] = res.apply(similarity_ratio, axis=1)
    return res


def count_incorrect_words(word: pd.DataFrame) -> float:
    """Count the number of incorrect words.

    Parameters
    ----------
    word : pd.DataFrame
        A data frame of the words.

    Returns
    -------
    int
        The number of incorrect words.
    """
    return (~word["success"]).sum()


def count_key_pressed(key_pressed: pd.Series, alphabet: bool = False) -> int:
    """Count the number of keys pressed.

    Parameters
    ----------
    key_pressed
        A pd.Series indicating the key pressed during the test.
    alphabet
        An optional argument indicating if the count should be done on alphabet
        letters only.

    Returns
    -------
    int
        The number of keys pressed.
    """
    if alphabet:
        return key_pressed.apply(lambda x: x.isalpha()).sum()
    return key_pressed.sum()


def ratio_key_pressed(
    submission_state: pd.DataFrame, word: pd.DataFrame
) -> pd.DataFrame:
    """Group by word and count the letters typed for completed words.

    Parameters
    ----------
    submission_state
        A data frame that contains the user state after each keystroke.
    word
        A data frame listing all the words displayed on screen during the
        test with the following columns: ``words``, ``appearance_timestamp``
        and ``disappearance_timestamp``.

    Returns
    -------
    pd.DataFrame
        A data frame with a column ``letters_typed_over_length``
        containing the following ratio: number of letters typed divided
        by the length of the word (for completed words).
    """
    res = (
        submission_state[["displayed_word", "key"]]
        .groupby(by="displayed_word")
        .apply(lambda x: count_key_pressed(x["key"], True))
    )

    # keep the completed words only
    completed_word = word.loc[word["success"], "word"]
    res = res[completed_word]

    # compute word length
    word_length = res.index.to_series().apply(len)

    # return the ratio letters typed over word length
    return pd.DataFrame({"letters_typed_over_length": res / word_length}).set_index(
        res.index
    )


def letter_interval(
    key_is_letter: pd.Series,
    key_intervals: pd.Series,
    extra_mask: Optional[pd.Series] = None,
) -> pd.Series:
    """Compute the interval between two consecutive letters.

    Parameters
    ----------
    key_is_letter
        A boolean series indicating if a key is a letter.
    key_intervals
        A series of float indicating the time between two consecutive keys.
    extra_mask
        An optional series of boolean that will be used as an extra mask
        to filter intervals.

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        The first element of the tuple is the time between each consecutive
        letter keys. The second element is the mask that has been used to
        filter the `key_intervals` to find the letters of interests.
    """
    mask = key_is_letter & key_is_letter.shift(1)
    if extra_mask is not None:
        mask = mask & extra_mask
    return key_intervals[mask], mask


def keep_top_ten(data: pd.DataFrame) -> pd.Series:
    """Keep top 10% intervals.

    Parameters
    ----------
    data
        A dataframe of letter intervals with the column `letter_intervals`.

    Returns
    -------
        The top 10 percent letter intervals.
    """
    res = data["letter_intervals"]
    if res.empty:
        return res
    return res[res >= np.percentile(res, q=90)]


def max_letter_interval_dev(data: pd.DataFrame) -> pd.Series:
    """Compute the maximum letter interval deviation from the mean."""
    return ((data["letter_intervals"] - data["letter_intervals"].mean()) ** 2).max()


def filter_intervals(data: pd.DataFrame, differentiate: bool) -> pd.Series:
    """Compute letter intervals prior a mistake within a word."""
    mask = (data["submission_state"] == KeyState.CORRECT).to_numpy()
    # Find first non-correct element if there is one
    not_correct = np.nonzero(~mask)[0]
    if len(not_correct) > 0:
        # turn to False every Correct letters typed after a mistake
        mask[not_correct[0] :] = False
    mask = mask & data["is_letter"].to_numpy()
    res = data.index.to_series().diff()
    if differentiate:
        res = res.diff()
    return res[mask].dt.total_seconds()


def interval_until_mistake(
    submission_state: pd.DataFrame,
    key_is_letter: pd.DataFrame,
    differentiate: bool,
) -> pd.DataFrame:
    """Compute letter intervals prior a mistake for each word."""
    col = "interval_until_mistake"
    if differentiate:
        col = f"derived_{col}"
    res = submission_state[["displayed_word", "submission_state"]].copy()
    res["is_letter"] = key_is_letter["key_is_letter"].values
    multi_index_res = (
        res.groupby(by="displayed_word")
        .apply(lambda x: filter_intervals(x, differentiate))
        .dropna()
    )
    if multi_index_res.empty:
        return pd.DataFrame({col: []})
    return pd.DataFrame({col: multi_index_res.values})


def compute_typing_speed_slope(
    word: pd.DataFrame, submission_state: pd.DataFrame
) -> float:
    """Compute the slope of the typing speed."""

    def _compute_char_speed(group: pd.DataFrame):
        # Return the mean interval between the characters of a word
        return group.index.to_series().diff().dt.total_seconds().mean()

    # Get the right columns and change index for the merge
    df_word_filtered = word[["appearance_timestamp", "word"]].copy().set_index("word")
    # Compute the typing speed for each word
    char_speed = pd.DataFrame(
        submission_state.groupby("displayed_word").apply(_compute_char_speed),
        columns=["mean_char_speed"],
    )
    # Index based inner join to discard uncompleted words
    res = pd.merge(df_word_filtered, char_speed, left_index=True, right_index=True)
    # Return the mean slope
    return (
        res["mean_char_speed"].diff()
        / res["appearance_timestamp"].diff().dt.total_seconds()
    ).mean()
