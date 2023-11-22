# pylint: disable=too-many-lines
"""Typing module."""
import math
from functools import partial
from typing import List, Optional

import pandas as pd

from dispel.data.core import Reading
from dispel.data.flags import FlagSeverity, FlagType
from dispel.data.measures import MeasureValueDefinitionPrototype
from dispel.data.raw import DEFAULT_COLUMNS, RawDataValueDefinition
from dispel.data.values import AbbreviatedValue as AV
from dispel.processing.assertions import AssertEvaluationFinished
from dispel.processing.core import FlagReadingStep, ProcessingStep
from dispel.processing.data_set import StorageError, transformation
from dispel.processing.extract import (
    DEFAULT_AGGREGATIONS,
    DEFAULT_AGGREGATIONS_IQR,
    AggregateRawDataSetColumn,
    ExtractMultipleStep,
    ExtractStep,
)
from dispel.processing.flags import flag
from dispel.processing.level import ProcessingStepGroup
from dispel.processing.level_filters import LastLevelFilter, NotEmptyDataSetFilter
from dispel.processing.modalities import SensorModality
from dispel.processing.transform import TransformStep
from dispel.providers.bdh.data import BDHReading
from dispel.providers.bdh.tasks.typing.const import *
from dispel.providers.bdh.tasks.typing.keyboard import *
from dispel.providers.generic.preprocessing import Detrend
from dispel.providers.generic.sensor import Resample, SetTimestampIndex
from dispel.providers.generic.tremor import TremorMeasures
from dispel.providers.registry import process_factory

# Define constants
TASK_NAME = AV("Typing test", "TT")


# Define transform steps
class TransformReactionTime(TransformStep):
    """A transform step to find the reaction time for each word."""

    data_set_ids = ["word", "key_typed"]
    transform_function = transform_reaction_time
    new_data_set_id = "reaction_time_per_word"
    definitions = [DEF_REACTION_TIME, DEF_REACTION_TIME_FC]


class AggregateReactionTime(AggregateRawDataSetColumn):
    """An aggregation step for the reaction time measures."""

    data_set_ids = TransformReactionTime.new_data_set_id
    column_id = DEF_REACTION_TIME.id.id
    aggregations = DEFAULT_AGGREGATIONS
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("reaction time", "rt"),
        description="The {aggregation} time spent between the appearance of a word and "
        "the moment the user releases the first key for each word.",
        unit="s",
        data_type="float",
    )


class AggregateReactionTimeCorrectLetter(AggregateRawDataSetColumn):
    """An aggregation step for reaction time first correct letter measures."""

    data_set_ids = TransformReactionTime.new_data_set_id
    column_id = DEF_REACTION_TIME_FC.id.id
    aggregations = DEFAULT_AGGREGATIONS
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("reaction time first correct", "rt_fc"),
        description="The {aggregation} time spent between the appearance of a word and "
        "the moment the user releases the first correct letter key for "
        "each word.",
        unit="s",
        data_type="float",
    )


class ExtractReactionTimeSlope(ExtractStep):
    """Extract reaction time slope.

    Parameters
    ----------
    is_correct
        A boolean value indicating if the ExtractStep should compute
        the reaction time slope on any letter or with first correct
        letter
    """

    def __init__(self, is_correct: bool, **kwargs):
        col = "reaction_time"
        name = "reaction time"
        abv = "rt"
        description = "The mean slope of the reaction time to write the first"
        if is_correct:
            col += "_first_correct"
            name += "correct slope"
            abv += "_fc"
            description = "correct"
        description += "letter."
        name += "slope"
        abv += "_slope"

        super().__init__(
            ["reaction_time_per_word"],
            transform_function=lambda x: compute_rt_slope(x, col),
            definition=MeasureValueDefinitionPrototype(
                measure_name=AV(name, abv),
                description=f"{description}. The slope is computed as the discrete "
                "derivatives of the reaction time with respect to x: the "
                "appearance of the word.",
                data_type="float",
            ),
            **kwargs,
        )


class ExtractPatientScore(ExtractStep):
    """Extract user's typing score.

    The typing score is the total number of words correctly typed.
    """

    data_set_ids = "word"
    transform_function = total_words_typed
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("number of correct words", "n_correct_words"),
        description="The total number of words successfully typed.",
        data_type="int",
    )


class TransformKeyInterval(TransformStep):
    """A transform step to find the time intervals between two key pressed."""

    data_set_ids = "key_typed"
    transform_function = lambda x: time_intervals(x, "timestamp_out")
    new_data_set_id = "key_intervals"
    definitions = [DEF_KEY_INTERVALS]


class TransformWordDuration(TransformStep):
    """A transform step to compute the duration of a word."""

    data_set_ids = "word"
    transform_function = word_duration
    new_data_set_id = "word_duration"
    definitions = [DEF_WORD_DURATION]


class AggregateWordDuration(AggregateRawDataSetColumn):
    """An aggregation step for Word duration measures."""

    data_set_ids = TransformWordDuration.new_data_set_id
    column_id = DEF_WORD_DURATION.id.id
    aggregations = DEFAULT_AGGREGATIONS
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("word duration", "word_duration"),
        description="The {aggregation} time spent to write a word.",
        unit="s",
        data_type="float",
    )


class TransformDifficultyLevel(TransformStep):
    """A transform step to compute the difficulty level of each word."""

    data_set_ids = "word"

    @staticmethod
    @transformation
    def get_difficulty(word: pd.DataFrame) -> pd.Series:
        """Get the level of difficulty."""
        difficulty_level = word.groupby("level").grouper.group_info[0]
        return pd.Series(max(difficulty_level) - difficulty_level)

    new_data_set_id = "difficulty_level"
    definitions = [DEF_DIFFICULTY_LEVEL]


class TransformWordDurationPerDifficulty(TransformStep):
    """Compute aggregates related to word duration per level difficulty."""

    data_set_ids = ["word", "word_duration", "difficulty_level"]

    @staticmethod
    @transformation
    def transform_duration_per_difficulty(
        word: pd.DataFrame,
        word_duration: pd.DataFrame,
        difficulty_level: pd.DataFrame,
    ) -> pd.DataFrame:
        """Aggregate word duration per level of difficulty."""
        res = pd.DataFrame(
            {
                "success": word["success"],
                "difficulty_level": difficulty_level["difficulty_level"],
                "word_duration": word_duration["word_duration"],
                "word": word["word"],
            }
        )
        groups = res[res["success"]].groupby(by=["difficulty_level"])
        dict_measures = {}
        for agg, agg_func in STR_TO_CALLABLE.items():
            dict_measures[agg] = groups.agg(agg_func)["word_duration"]
        return pd.DataFrame(dict_measures)

    new_data_set_id = "word_duration_per_difficulty"
    definitions = [
        DEF_MEAN_WORD_DURATION,
        DEF_MEAN_MEDIAN_WORD_DURATION,
        DEF_STD_WORD_DURATION,
    ]


class ExtractWordDurationPerDifficulty(ExtractMultipleStep):
    """Extract aggregated representations of word duration per difficulty."""

    def __init__(self, **kwargs):
        def extract_word_duration(
            data: pd.DataFrame, difficulty: int, agg: str
        ) -> float:
            """Extract word duration aggregated representation."""
            try:
                return data.loc[difficulty, agg]
            except KeyError:
                # There is no data for the given level
                return math.nan

        functions = []
        for difficulty in range(N_LEVELS):
            for agg, _ in STR_TO_CALLABLE.items():
                functions += [
                    dict(
                        func=partial(
                            extract_word_duration, difficulty=difficulty, agg=agg
                        ),
                        agg=agg,
                        difficulty=difficulty,
                    )
                ]

        super().__init__(
            TransformWordDurationPerDifficulty.new_data_set_id,
            transform_functions=functions,
            definition=MeasureValueDefinitionPrototype(
                measure_name=AV(
                    "word duration {difficulty} {agg}",
                    "word_duration-{difficulty}-{agg}",
                ),
                unit="s",
                description="The {agg} word duration for the {difficulty} "
                "difficulty level.",
                data_type="float",
            ),
            **kwargs,
        )


class TransformWordDurationLevelDifference(TransformStep):
    """A transform step to find the slope of word duration.

    The slope is computed as the following: Let us consider the x-axis: level,
    and y-axis: word_duration-mean, the slope is given by the differentiation
    of y-axis (x-axis being incremented by one every-time).
    """

    @staticmethod
    @transformation
    def differentiate(data: pd.DataFrame) -> pd.Series:
        """Differentiate average word duration."""
        return data["mean"].diff()[1:]

    data_set_ids = TransformWordDurationPerDifficulty.new_data_set_id
    new_data_set_id = "word_duration_per_difficulty_mean_difference"
    definitions = [DEF_WORD_DUR_DIFF]


class AggregateWordDurationLevelSlope(AggregateRawDataSetColumn):
    """An aggregation step for statistics about the slope of word duration."""

    data_set_ids = TransformWordDurationLevelDifference.new_data_set_id
    column_id = DEF_WORD_DUR_DIFF.id.id
    aggregations = DEFAULT_AGGREGATIONS
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("word duration slope", "word_duration_slope"),
        description="The {aggregation} of the word duration slope. The slope is "
        "computed by differencing the word duration mean per difficulty "
        "level w.r.t difficulty level.",
        data_type="float",
    )


class ExtractTimeToFinishLastThreeWords(ExtractStep):
    """Extract aggregated representation of the last three words duration."""

    def __init__(self, agg: str, **kwargs):
        def extract_last_three(
            word_duration: pd.DataFrame, word: pd.DataFrame, agg: str
        ) -> float:
            """Extract last three typed word duration."""
            agg_func = STR_TO_CALLABLE[agg]
            res = word_duration.loc[word["success"], "word_duration"]
            if len(res) >= 3:
                res = res.tail(3)
            return res.agg(agg_func)

        super().__init__(
            ["word_duration", "word"],
            transform_function=partial(extract_last_three, agg=agg),
            definition=MeasureValueDefinitionPrototype(
                measure_name=AV(
                    f"last three word duration {agg}", f"word_duration-last_three-{agg}"
                ),
                unit="s",
                description=f"The {agg} duration of the last three words.",
                data_type="float",
            ),
            **kwargs,
        )


class TransformSubmissionState(TransformStep):
    """Create a dataframe that contains the submission state.

    This transform step translate the state at each time a user types a key. Available
    states are the following: ``Correct``, ``Incorrect``, and ``Correcting``. The state
    is ``Correct`` if the textbox is empty or if it forms a subpart of the word
    displayed on the screen. The state is incorrect whenever the user types a key (that
    is not backspace), leading to a textbox that does not form a subpart of the word
    displayed on the screen. Finally, the state is identified as correcting whenever
    the user corrects its mistake by typing backspace; it can be seen as an
    intermediate state between an incorrect and a correct state.
    """

    data_set_ids = ["word", "key_typed"]
    transform_function = get_submission_state
    new_data_set_id = "submission_state"
    definitions = [
        DEF_SUBMISSION_STATE,
        DEF_DISPLAYED_WORD,
        DEF_DISPLAYED_INPUT,
        DEF_KEY,
        DEF_SUCCESS,
    ]


class TransformDetectKeyAutocompletion(TransformStep):
    """Detect the autocompletion at a key level."""

    data_set_ids = TransformSubmissionState.new_data_set_id
    transform_function = detect_key_autocompletion
    new_data_set_id = "keys_with_autocompletion"
    definitions = [DEF_DISPLAYED_WORD, DEF_KEY, DEF_AUTOCOMPLETE_KEY]


class TransformDetectWordAutocompletion(TransformStep):
    """A new dataset indicating if a word has been completed.

    If autocompletion was used the associated word will be mark as True.
    """

    data_set_ids = TransformDetectKeyAutocompletion.new_data_set_id
    transform_function = detect_word_autocompletion
    new_data_set_id = "autocompletion_per_word"
    definitions = [DEF_AUTOCOMPLETE_WORD]


class ExtractAutocomplete(ExtractStep):
    """Extract the number of autocompletions."""

    data_set_ids = TransformDetectKeyAutocompletion.new_data_set_id
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("number of autocompletions", "n_autocompletion"),
        description="The total number of autocompletions.",
        data_type="int",
    )
    transform_function = total_autocomplete


class FlagAutoComplete(FlagReadingStep):
    """Flag the reading if any autocomplete key is detected."""

    # pylint: disable=unused-argument
    task_name = TASK_NAME
    flag_name = AV("autocomplete", "auto_complete")
    flag_type = FlagType.BEHAVIORAL
    flag_severity = FlagSeverity.DEVIATION
    reason = "The user has autocompleted at least one word."

    @flag
    def flag_autocomplete(self, reading: Reading, **kwargs) -> bool:
        """Indicate if there was no autocompletion during the evaluation."""
        return (
            reading.get_merged_measure_set().get_raw_value("tt-n_autocompletion") == 0
        )


class TransformStateDurations(TransformStep):
    """Create a data frame that contains the duration of each state."""

    data_set_ids = ["submission_state", "key_intervals", "word"]
    new_data_set_id = "state_durations"
    definitions = [
        DEF_SUBMISSION_STATE,
        DEF_STATE_DUR,
        DEF_T_BETWEEN_STATE,
        DEF_SUCCESS,
        DEF_DISPLAYED_WORD,
    ]
    transform_function = get_state_durations


class TransformReactionDuration(TransformStep):
    """Create a data frame that contains the reaction duration."""

    data_set_ids = TransformStateDurations.new_data_set_id
    new_data_set_id = "reaction_duration"
    definitions = [DEF_INCORRECT_STATE_DURATION, DEF_SUCCESS]
    transform_function = get_reaction_duration


class TransformReactionDurationCorrectSubmissions(TransformStep):
    """Filter reaction duration values for correct words."""

    data_set_ids = TransformReactionDuration.new_data_set_id
    transform_function = get_correct_reaction_duration
    new_data_set_id = "reaction_durations_correct_submissions"
    definitions = [DEF_REACTION_DURATION]


class AggregateReactionDuration(AggregateRawDataSetColumn):
    """An aggregation step for reaction duration measure."""

    data_set_ids = TransformReactionDurationCorrectSubmissions.new_data_set_id
    column_id = DEF_REACTION_DURATION.id.id
    aggregations = DEFAULT_AGGREGATIONS
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("reaction duration", "reaction_duration"),
        description="The {aggregation} reaction duration for a correct submission.",
        unit="s",
        data_type="float",
    )


class TransformCorrectingDuration(TransformStep):
    """Create a data frame that contains correcting durations."""

    data_set_ids = TransformStateDurations.new_data_set_id
    transform_function = get_correcting_duration
    new_data_set_id = "correcting_duration"
    definitions = [DEF_CORRECTING_DURATION, DEF_SUCCESS]


class TransformCorrectingDurationCorrectSubmissions(TransformStep):
    """Filter correcting duration values for correct words."""

    data_set_ids = TransformCorrectingDuration.new_data_set_id
    transform_function = get_correct_correcting_duration
    new_data_set_id = "correcting_duration_correct_submissions"
    definitions = [DEF_CORRECTING_CORRECT_DURATION]


class AggregateCorrectingDuration(AggregateRawDataSetColumn):
    """An extraction processing step for correcting duration measures."""

    data_set_ids = TransformCorrectingDurationCorrectSubmissions.new_data_set_id
    column_id = DEF_CORRECTING_CORRECT_DURATION.id.id
    aggregations = DEFAULT_AGGREGATIONS
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("correcting duration", "correcting_duration"),
        description="The {aggregation} correcting duration for a correct submission.",
        unit="s",
        data_type="float",
    )


class TransformReactingTime(TransformStep):
    """Create a data frame that contains reacting time."""

    data_set_ids = TransformStateDurations.new_data_set_id
    transform_function = get_reacting_times
    new_data_set_id = "reacting_times"
    definitions = [DEF_REACTING_TIME, DEF_SUCCESS]


class TransformReactingTimeCorrectSubmissions(TransformStep):
    """Filter reacting times values for correct words."""

    data_set_ids = TransformReactingTime.new_data_set_id
    transform_function = get_correct_reacting_time
    new_data_set_id = "reacting_times_correct_submissions"
    definitions = [DEF_REACTING_TIME_CORRECT]


class AggregateReactingTime(AggregateRawDataSetColumn):
    """An aggregation processing step for reacting time measures."""

    data_set_ids = TransformReactingTimeCorrectSubmissions.new_data_set_id
    column_id = DEF_REACTING_TIME_CORRECT.id.id
    aggregations = DEFAULT_AGGREGATIONS
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("reacting time", "reacting_time"),
        description="The {aggregation} reacting time for a correct word.",
        unit="s",
        data_type="float",
    )


class TransformCorrectSubmissionAndTime(TransformStep):
    """A data set to deal with correct (consecutive) words with(out) errors.

    This transform step aims to create a data set appropriate to compute measure
    related to successfully written words with(out) error, and also a streak of words
    with(out) error(s).

    The transformation, based on the ``submission_state`` data set, applies the
    following modifications:

    - Add the appearance timestamp as a column
    - Create a boolean translating if the state is ``Correct`` or not
    - Group the data frame by displayed word and aggregate with the min.

    This results in a data frame indexed with ``displayed_word`` and has three columns:
    ``appearance_timestamp``, ``is_error_free``, ``success``. Here
    ``appearance_timestamp`` is the timestamp corresponding to the appearance of the
    word, ``is_error_free`` is a boolean set to ``True`` if all correction_state were
    ``Correct``. Finally, ``success`` indicates if the word was a correct submission.
    """

    data_set_ids = ["word", "submission_state"]

    @staticmethod
    @transformation
    def correct_and_ts(
        word: pd.DataFrame, submission_state: pd.DataFrame
    ) -> pd.DataFrame:
        """Group by word on screen and aggregate with the min."""
        res = word[["word", "appearance_timestamp", "success"]].copy()
        sub_state_copy = submission_state[["displayed_word", "submission_state"]].copy()
        sub_state_copy["is_error_free"] = (
            sub_state_copy["submission_state"] == KeyState.CORRECT
        )
        group = sub_state_copy.groupby(by="displayed_word")
        res.set_index("word", inplace=True)
        is_error_free = group["is_error_free"].all()
        is_error_free.index.name = "word"
        res["is_error_free"] = is_error_free
        return res

    new_data_set_id = "correct_sub_and_time"
    definitions = [DEF_SUCCESS, DEF_IS_ERROR_FREE, DEF_APPEARANCE_TS]


class ExtractWordTypedWithOrWoError(ExtractStep):
    """Count the words successfully typed and written with(out) error."""

    def __init__(self, error_free: bool = True, **kwargs):
        if error_free:
            with_or_wo = "without"
        else:
            with_or_wo = "with"

        def count_words_typed(data: pd.DataFrame, error_free: bool = True) -> int:
            """Count the number of words typed with(out) any error."""
            mask = data.is_error_free
            if not error_free:
                mask = ~mask.fillna(False)
            return len(data.loc[data.success & mask])

        super().__init__(
            data_set_ids=TransformCorrectSubmissionAndTime.new_data_set_id,
            transform_function=lambda x: count_words_typed(x, error_free),
            definition=MeasureValueDefinitionPrototype(
                measure_name=AV(
                    f"words typed {with_or_wo} error", f"words_typed_{with_or_wo}_error"
                ),
                description=f"Number of correct words typed {with_or_wo} error.",
                data_type="int",
            ),
            **kwargs,
        )


class TransformCorrectSubmissionAndTimeInRow(TransformStep):
    """A transform step to count consecutive states.

    The transformation, based on the ``correct_sub_and_time`` data set, works as the
    following : Assign a subgroup number to each group of consecutive ``is_error_free``
    values. Then group by subgroup number and aggregate with count. Finally, it returns
    a data frame with three columns: the count, the submission state and if the
    sequence was free of error.
    """

    data_set_ids = TransformCorrectSubmissionAndTime.new_data_set_id
    transform_function = find_consec_element_and_count
    new_data_set_id = "correct_sub_in_row"
    definitions = [DEF_SUCCESS, DEF_COUNT_CONSEC_STATES, DEF_IS_ERROR_FREE]


class ExtractWordTypedWithOrWoErrorInRow(ExtractStep):
    """Count the successfully typed words written with(out) error in a row."""

    def __init__(self, error_free: bool = True, **kwargs):
        if error_free:
            with_or_wo = "without"
        else:
            with_or_wo = "with"
        super().__init__(
            data_set_ids=TransformCorrectSubmissionAndTimeInRow.new_data_set_id,
            transform_function=lambda x: count_words_typed_in_row(x, error_free),
            definition=MeasureValueDefinitionPrototype(
                measure_name=AV(
                    f"consecutive words typed {with_or_wo} error",
                    f"consecutive_words_typed_{with_or_wo}_error",
                ),
                data_type="int",
            ),
            **kwargs,
        )


class ExtractIncorrectWords(ExtractStep):
    """Extract the number of incorrect words."""

    data_set_ids = "word"
    transform_function = count_incorrect_words
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("number of Incorrect words", "n_incorrect_words"),
        description="Number of Incorrect words.",
        data_type="int",
    )


class TransformSimilarityRatio(TransformStep):
    """A transform step to compute similarity metric between answer and target.

    The similarity metric that has been chosen is the ratio returned by the
    sequence matcher from difflib. It is a float in the range [0, 1]. Where T
    is the total number of elements in both sequences, and M is the number of
    matches, this is 2.0*M / T. Note that this is 1.0 if the sequences are
    identical, and 0.0 if they have nothing in common.
    """

    data_set_ids = TransformSubmissionState.new_data_set_id
    transform_function = apply_similarity_ratio
    new_data_set_id = "similarity_ratio"
    definitions = [DEF_DISPLAYED_INPUT, DEF_DISPLAYED_WORD, DEF_SIMILARITY_RATIO]


class TransformSimilarityRatioGroup(TransformStep):
    """Group similarity ratio by displayed_word and keep the max.

    In order to only keep incorrect words we remove similarity ratio equal to one.
    """

    data_set_ids = TransformSimilarityRatio.new_data_set_id

    @staticmethod
    @transformation
    def group_and_max_sim_ratio(data: pd.DataFrame) -> pd.DataFrame:
        """Group by displayed word and get the similarity ratio max."""
        res = data.groupby(by="displayed_word")["similarity_ratio"].max()
        return res[res != 1]

    new_data_set_id = "similarity_ratio_grouped"
    definitions = [DEF_SIMILARITY_RATIO]


class AggregateSimilarityRatioMeasures(AggregateRawDataSetColumn):
    """Aggregate similarity ratio measures."""

    data_set_ids = TransformSimilarityRatioGroup.new_data_set_id
    column_id = DEF_SIMILARITY_RATIO.id.id
    aggregations = DEFAULT_AGGREGATIONS
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("similarity ratio", "sim_ratio"),
        description="The {aggregation} similarity ratio between word being typed and "
        "word on screen.",
        data_type="float",
    )


class ExtractCountKeyPressed(ExtractStep):
    """Count the number of keys pressed."""

    data_set_ids = "key_typed"

    @staticmethod
    @transformation
    def count_key_pressed(key_typed: pd.DataFrame) -> int:
        """Count the number of keys pressed."""
        return count_key_pressed(key_typed["key"])

    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("keys pressed", "keys_pressed"),
        description="Number of keys pressed.",
        data_type="int",
    )


class TransformLettersTypedPerWordRatio(TransformStep):
    """
    Compute the ratio of the letters typed per word divided by its length.

    This transform steps computes the ratio for completed words only.
    """

    data_set_ids = ["submission_state", "word"]
    transform_function = ratio_key_pressed
    new_data_set_id = "letters_typed_over_length"
    definitions = [DEF_LETTER_TYPED_OVER_LEN]


class AggregateLettersTypedPerWordRatio(AggregateRawDataSetColumn):
    """Aggregate measures related to the ratio of letters typed."""

    data_set_ids = TransformLettersTypedPerWordRatio.new_data_set_id
    column_id = DEF_LETTER_TYPED_OVER_LEN.id.id
    aggregations = DEFAULT_AGGREGATIONS
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("letters typed over length", "ratio_letters_typed_len"),
        description="The {aggregation} of the ratio of letters typed over the "
        "respective length of the word for completed word only.",
        data_type="float",
    )


class ExtractCountCorrectLetters(ExtractStep):
    """Extract the number of correct letters."""

    @staticmethod
    @transformation
    def count_correct_letters(submission_state: pd.DataFrame) -> int:
        """Count the number of correct letters."""
        is_alphabet_letter = submission_state["key"].apply(lambda x: x.isalpha())
        is_correct = submission_state["submission_state"] == KeyState.CORRECT
        return (is_alphabet_letter & is_correct).sum()

    data_set_ids = TransformSubmissionState.new_data_set_id
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("correct letters", "correct_letters"),
        description="Number of correct letters.",
        data_type="int",
    )


class TransformIsLetter(TransformStep):
    """A transform step to find if a key is a letter."""

    data_set_ids = "key_typed"

    @staticmethod
    @transformation
    def is_letter(key_typed: pd.DataFrame) -> pd.Series:
        """Identify if a key typed is a letter."""
        return key_typed["key"].apply(lambda x: x.isalpha())

    new_data_set_id = "key_is_letter"
    definitions = [DEF_IS_LETTER]


class TransformLetterInterval(TransformStep):
    """A transform step to compute the time between letters.

    The time between letters is computed as the time separating two keys that
    are letters (see isalpha() method), for example, interval between words
    isn't considered.

    Parameters
    ----------
    category
        The category on which one wants to filter the data set. If not provided
        the data set will return interval between letters for every words.
        When provided category should be either ``Correct`` or ``Incorrect``.

    Raises
    ------
    ValueError
        If the category provided is not allowed. Or if a category is provided
        but the optional dataset word_is_typed is not provided.
    """

    def __init__(self, category: Optional[WordState] = None):
        def transform_letter_interval(
            key_is_letter: pd.DataFrame,
            key_intervals: pd.DataFrame,
            submission_state: pd.DataFrame,
        ):
            """Compute letter intervals for a given submission state."""
            if not category:
                letter_intervals, mask = letter_interval(
                    key_is_letter["key_is_letter"], key_intervals["key_intervals"]
                )
                return pd.DataFrame(
                    {
                        "letter_intervals": letter_intervals,
                        "displayed_word": submission_state.reset_index().loc[
                            mask, "displayed_word"
                        ],
                    }
                )

            extra_mask = submission_state["success"]
            if category == WordState.INCORRECT:
                extra_mask = ~extra_mask

            letter_intervals, mask = letter_interval(
                key_is_letter["key_is_letter"],
                key_intervals["key_intervals"],
                extra_mask=extra_mask.values,
            )
            return pd.DataFrame(
                {
                    "letter_intervals": letter_intervals,
                    "displayed_word": submission_state.reset_index().loc[
                        mask, "displayed_word"
                    ],
                }
            )

        data_set_ids = ["key_is_letter", "key_intervals", "submission_state"]
        new_data_set_id = "letter_intervals"
        description = "The time between consecutive letters"
        if category:
            category_name = category.name.lower()
            new_data_set_id = f"{new_data_set_id}_{category_name}"
            description += f"for the {category_name} words"
        description += "."

        super().__init__(
            data_set_ids=data_set_ids,
            transform_function=transform_letter_interval,
            new_data_set_id=new_data_set_id,
            definitions=[
                RawDataValueDefinition(
                    id_="letter_intervals",
                    name="letter intervals",
                    description=description,
                    data_type="float",
                    unit="s",
                ),
                DEF_DISPLAYED_WORD,
            ],
        )


class AggregateLettersIntervals(AggregateRawDataSetColumn):
    """Extract letter intervals related measures.

    Parameters
    ----------
    category
        The category on which one wants to extract measures. If not provided the
        measures will be extracted based on the data set computed on every word.
    """

    def __init__(self, category: Optional[WordState] = None, **kwargs):
        data_set_id = "letter_intervals"
        measure_name = "letter intervals"
        measure_abbr = "letter_intervals"
        description = "The {aggregation} time interval between two letters"
        if category:
            category_name = category.name.lower()
            data_set_id = f"{data_set_id}_{category_name}"
            measure_name = f"{measure_name} {category_name}"
            measure_abbr = f"{measure_abbr}_{category_name}"
            description += f"for {category_name} words only"

        description += "."
        super().__init__(
            data_set_id,
            "letter_intervals",
            aggregations=DEFAULT_AGGREGATIONS,
            definition=MeasureValueDefinitionPrototype(
                measure_name=AV(measure_name, measure_abbr),
                unit="s",
                description=description,
                data_type="float",
            ),
            **kwargs,
        )


class TransformTop10Interval(TransformStep):
    """A transform step to find the top ten percent letters intervals.

    Parameters
    ----------
    category
        The category on which one wants to extract measures. If not provided
        the measures will be extracted based on the data set computed on every
        letter intervals.
    """

    def __init__(self, category: Optional[KeyState] = None):
        data_set_id = "letter_intervals"
        new_data_set_id = "top_10_letter_intervals"
        description = "The {aggregation} time interval between two letters"
        if category:
            category_name = category.name.lower()
            new_data_set_id = f"{new_data_set_id}_{category_name}"
            data_set_id = f"{data_set_id}_{category_name}"
            description += f"for {category_name} letters only"
        description += "."

        super().__init__(
            data_set_id,
            transform_function=keep_top_ten,
            new_data_set_id=new_data_set_id,
            definitions=[
                RawDataValueDefinition(
                    id_="letter_intervals",
                    name="letter intervals",
                    description=description,
                    data_type="float",
                    unit="s",
                )
            ],
        )


class AggregateTop10IntervalDefaultMeasures(AggregateRawDataSetColumn):
    """Extract measures related to the top ten percent letters intervals.

    Parameters
    ----------
    category
        The category on which one wants to extract measures. If not provided
        the measures will be extracted based on the data set computed on every
        letter intervals.

    """

    def __init__(self, category: Optional[KeyState] = None, **kwargs):
        data_set_id = "top_10_letter_intervals"
        measure_name_str = "top 10 letter intervals"
        measure_abbr = "top_10_letter_intervals"
        description = (
            "The {aggregation} of the top 10 percent time "
            "interval between two letters"
        )
        if category:
            category_name = category.name.lower()
            data_set_id = f"{data_set_id}_{category_name}"
            measure_name_str = f"{measure_name_str} {category_name}"
            measure_abbr = f"{measure_abbr}_{category_name}"
            description += f"for {category_name} letters only"
        description += "."
        measure_name = AV(measure_name_str, measure_abbr)
        transform_functions = DEFAULT_AGGREGATIONS_IQR

        super().__init__(
            data_set_id,
            column_id="letter_intervals",
            aggregations=transform_functions,
            definition=MeasureValueDefinitionPrototype(
                measure_name=measure_name, description=description, data_type="float"
            ),
            **kwargs,
        )


class TransformMaxDeviation(TransformStep):
    """A transform step to evaluate the maximum deviation per word.

    The deviation is computed as the squared error between the current interval
    and the mean of these intervals within a word.
    """

    data_set_ids = "letter_intervals"

    @staticmethod
    @transformation
    def compute_max_letter_interval_deviation(data: pd.DataFrame) -> pd.DataFrame:
        """Compute the maximum letter interval deviation."""
        res = data.groupby(by="displayed_word").apply(max_letter_interval_dev)
        if len(res) == 0:
            return pd.DataFrame({"max_deviation_letter_interval": []})
        return res

    new_data_set_id = "max_deviation_letter_interval"
    definitions = [DEF_MAX_DEVIATION_LETTER_INTERVAL]


class AggregateMaxDeviation(AggregateRawDataSetColumn):
    """Aggregate step related to the maximum deviation of letter intervals."""

    data_set_ids = ["max_deviation_letter_interval"]
    column_id = "max_deviation_letter_interval"
    aggregations = DEFAULT_AGGREGATIONS_IQR
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV(
            "letter intervals maximum deviation ", "letter_interval_max_dev"
        ),
        description="The {aggregation} of the maximum deviation of "
        "the letter interval for each word.",
        unit="s",
        data_type="float",
    )


class TransformIntervalCorrectLettersUntilMistake(TransformStep):
    """Compute the interval between correct letters until a mistake.

    Parameters
    ----------
    differentiate
        An boolean indicating if we want to compute the interval or the
        derivative of the intervals.
    """

    def __init__(self, differentiate: bool = False):
        def _transform_function(x: pd.DataFrame, y: pd.DataFrame):
            """Format _interval_until_mistake as a transform function."""
            return interval_until_mistake(x, y, differentiate)

        new_data_set_id = "interval_until_mistake"
        description = "The time interval between two consecutive correct letters"
        raw_data_id_ = "interval_until_mistake"
        raw_data_name = "interval_until_mistake"
        if differentiate:
            description += "after differentiation"
            new_data_set_id = f"derived_{new_data_set_id}"
            raw_data_id_ = f"derived_{raw_data_id_}"
            raw_data_name = f"derived {raw_data_name}"
        description += ". Correct letters are considered until the user make a mistake."
        super().__init__(
            data_set_ids=["submission_state", "key_is_letter"],
            transform_function=_transform_function,
            new_data_set_id=new_data_set_id,
            definitions=[
                RawDataValueDefinition(
                    id_=raw_data_id_,
                    name=raw_data_name,
                    description=description,
                    data_type="float",
                )
            ],
        )


class AggregateIntervalCorrectLetters(AggregateRawDataSetColumn):
    """Interval between correct letters until a mistake - measures.

    Parameters
    ----------
    differentiate
        An boolean indicating if we want to compute the interval or the
        derivative of the intervals.
    """

    def __init__(self, differentiate: bool = False, **kwargs):
        name = "interval until mistake"
        abbr = "interval_until_mistake"
        data_set_id = "interval_until_mistake"
        description = (
            "The {aggregation} of the time interval between two "
            "consecutive correct letters"
        )
        if differentiate:
            name = f"derived {name}"
            abbr = f"derived_{abbr}"
            data_set_id = f"derived_{data_set_id}"
            description += "after differentiation"
        description += ". Correct letters are considered until the user make a mistake."
        measure_name = AV(name, abbr)
        super().__init__(
            data_set_id,
            data_set_id,
            aggregations=DEFAULT_AGGREGATIONS,
            definition=MeasureValueDefinitionPrototype(
                measure_name=measure_name,
                description=description,
                data_type="float",
                unit="s",
            ),
            **kwargs,
        )


class ExtractRatioWordsLetters(ExtractStep):
    """Compute the ratio of correct words divided by the number of letters."""

    data_set_ids = ["word", "key_is_letter"]

    @staticmethod
    @transformation
    def compute_ratio(word: pd.DataFrame, key_is_letter: pd.DataFrame) -> float:
        """Compute ratio of correct words by number of letters typed."""
        return word["success"].sum() / key_is_letter["key_is_letter"].sum()

    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("ratio correct words letters", "ratio_correct_words_letters"),
        description="The number of correctly typed words divided by the total number "
        "of letters.",
        data_type="float",
    )


class ExtractTypingSpeedSlope(ExtractStep):
    """Extract typing speed slope to quantify the fatigability.

    The typing speed slope is computed with discrete differentiation of the character
    speed (or typing speed) per word and the differentiation of the appearance
    timestamp of the words
    """

    data_set_ids = ["word", "submission_state"]
    transform_function = compute_typing_speed_slope
    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("typing speed slope", "typing_speed_slope"),
        description="Typing speed slope.",
        data_type="float",
    )


class TypingPreprocessingIMUGroup(ProcessingStepGroup):
    r"""A Preprocessing step to preprocess typing IMU Signal.

    Parameters
    ----------
    data_set_id
        The data set id on which the transformation is to be performed.
    resample_freq
        Optionally, the frequency to which resample the data during the
        resample step.
    columns
        Optional argument to specify the columns on which the preprocessing
        steps should be applied.
    max_frequency_distance
        An optional integer specifying the maximum accepted
        distance between the expected frequency and the estimated frequency
        above which we raise an error.
    """

    def __init__(
        self,
        data_set_id: str,
        resample_freq: Optional[float] = None,
        columns: Optional[List[str]] = None,
        max_frequency_distance: Optional[int] = None,
        **kwargs,
    ):
        columns = columns or DEFAULT_COLUMNS
        max_frequency_distance = max_frequency_distance or 10

        steps: List[ProcessingStep] = [
            SetTimestampIndex(data_set_id, columns, duplicates="first"),
            Resample(
                data_set_id=f"{data_set_id}_ts",
                freq=resample_freq,
                aggregations=["mean", "ffill"],
                columns=columns,
                max_frequency_distance=max_frequency_distance,
            ),
            Detrend(data_set_id=f"{data_set_id}_ts_resampled", columns=columns),
        ]
        level_filter = NotEmptyDataSetFilter(data_set_id) & LastLevelFilter()
        super().__init__(steps, level_filter=level_filter, **kwargs)


class TypingTremorMeasuresGroup(ProcessingStepGroup):
    """A group of typing processing steps for tremor measures."""

    steps = [
        TremorMeasures(
            sensor=SensorModality.ACCELEROMETER,
            data_set_id="accelerometer_ts_resampled_detrend",
            add_norm=False,
            level_filter=NotEmptyDataSetFilter("accelerometer"),
        ),
        TremorMeasures(
            sensor=SensorModality.GYROSCOPE,
            data_set_id="gyroscope_ts_resampled_detrend",
            add_norm=False,
            level_filter=NotEmptyDataSetFilter("gyroscope"),
        ),
    ]


class TransformKeyTyped(TransformStep):
    """Remove all key pressed after the last word disappeared."""

    new_data_set_id = "key_typed"
    data_set_ids = ["key_typed", "word"]
    definitions = [DEF_DISPLAYED_INPUT, DEF_TS_OUT, DEF_WORD_ID, DEF_KEY]

    storage_error = StorageError.OVERWRITE

    @staticmethod
    @transformation
    def remove_outdated_keys(key_typed: pd.DataFrame, word: pd.DataFrame):
        """Remove keys released after last word disappearance timestamp."""
        return key_typed.loc[
            key_typed["timestamp_out"] <= word.iloc[-1]["disappearance_timestamp"]
        ]


class PreprocessingTypingGroup(ProcessingStepGroup):
    """BDH typing preprocessing steps."""

    steps = [
        # Assert test has been completed
        AssertEvaluationFinished(),
        # Remove keys that appear after the last word has disappeared
        TransformKeyTyped(),
        # Define key interval
        TransformKeyInterval(),
        # Define each state
        TransformSubmissionState(),
        TransformStateDurations(),
        # Detect autocompletion
        TransformDetectKeyAutocompletion(),
        TransformDetectWordAutocompletion(),
        # Preprocessing IMU
        TypingPreprocessingIMUGroup(data_set_id="accelerometer"),
        TypingPreprocessingIMUGroup(data_set_id="gyroscope"),
    ]


class ReactionTimeGroup(ProcessingStepGroup):
    """BDH Typing Reaction time processing steps."""

    steps = [
        TransformReactionTime(),
        AggregateReactionTime(),
        AggregateReactionTimeCorrectLetter(),
        ExtractReactionTimeSlope(is_correct=True),
        ExtractReactionTimeSlope(is_correct=False),
    ]


class WordDurationGroup(ProcessingStepGroup):
    """BDH Typing word duration processing steps."""

    steps = [
        # Word Duration
        TransformWordDuration(),
        AggregateWordDuration(),
        # Word Duration per Difficulty level
        TransformDifficultyLevel(),
        TransformWordDurationPerDifficulty(),
        ExtractWordDurationPerDifficulty(),
        TransformWordDurationLevelDifference(),
        AggregateWordDurationLevelSlope(),
    ]


class TimeToFinishGroup(ProcessingStepGroup):
    """BDH Typing time to finish processing steps."""

    steps = [
        # Time to finish
        ExtractTimeToFinishLastThreeWords("mean"),
        ExtractTimeToFinishLastThreeWords("median"),
        ExtractTimeToFinishLastThreeWords("std"),
    ]


class ReactionCorrectingReactingDurationGroup(ProcessingStepGroup):
    """BDH Typing reaction, correcting and reacting duration processing steps."""

    steps = [
        # Reaction duration
        TransformReactionDuration(),
        TransformReactionDurationCorrectSubmissions(),
        AggregateReactionDuration(),
        # Correcting duration
        TransformCorrectingDuration(),
        TransformCorrectingDurationCorrectSubmissions(),
        AggregateCorrectingDuration(),
        # Reacting Time
        TransformReactingTime(),
        TransformReactingTimeCorrectSubmissions(),
        AggregateReactingTime(),
    ]


class CountWordsGroup(ProcessingStepGroup):
    """BDH Typing step counting words typed successfully with(out) errors."""

    steps = [
        # Counting words typed successfully with(out) errors
        TransformCorrectSubmissionAndTime(),
        ExtractWordTypedWithOrWoError(),
        ExtractWordTypedWithOrWoError(error_free=False),
        TransformCorrectSubmissionAndTimeInRow(),
        ExtractWordTypedWithOrWoErrorInRow(),
        ExtractWordTypedWithOrWoErrorInRow(error_free=False),
        ExtractIncorrectWords(),
    ]


class CountLettersGroup(ProcessingStepGroup):
    """BDH Typing steps counting number of letters and similarity ratio."""

    steps = [
        # Total number of letters
        TransformSimilarityRatio(),
        TransformSimilarityRatioGroup(),
        AggregateSimilarityRatioMeasures(),
        TransformLettersTypedPerWordRatio(),
        AggregateLettersTypedPerWordRatio(),
        ExtractCountCorrectLetters(),
    ]


class TimeBetweenLettersGroup(ProcessingStepGroup):
    """BDH Typing steps measuring intervals between letters."""

    steps = [
        # Time between letters
        TransformIsLetter(),
        TransformLetterInterval(),
        TransformLetterInterval(WordState.CORRECT),
        TransformLetterInterval(WordState.INCORRECT),
        AggregateLettersIntervals(),
        AggregateLettersIntervals(WordState.CORRECT),
        AggregateLettersIntervals(WordState.INCORRECT),
        # Keep top 10 intervals
        TransformTop10Interval(),
        TransformTop10Interval(KeyState.CORRECT),
        AggregateTop10IntervalDefaultMeasures(),
        AggregateTop10IntervalDefaultMeasures(KeyState.CORRECT),
        TransformMaxDeviation(),
        AggregateMaxDeviation(),
        # Speed for correct letters until mistake
        TransformIntervalCorrectLettersUntilMistake(),
        AggregateIntervalCorrectLetters(),
        TransformIntervalCorrectLettersUntilMistake(True),
        AggregateIntervalCorrectLetters(True),
    ]


class FlagAutoCompleteGroup(ProcessingStepGroup):
    """BDH Typing steps to flag autocomplete behavior."""

    steps = [ExtractAutocomplete(), FlagAutoComplete()]


class RatioAndSlopeGroup(ProcessingStepGroup):
    """BDH Typing steps measuring ratio of words letters and speed slope."""

    steps = [
        # Ratio correct words and total letters
        ExtractRatioWordsLetters(),
        # Typing frequency slope (fatigability)
        ExtractTypingSpeedSlope(),
    ]


class BDHTypingSteps(ProcessingStepGroup):
    """BDH-specific processing steps for typing."""

    steps = [
        PreprocessingTypingGroup(),
        ReactionTimeGroup(),
        ExtractPatientScore(),
        WordDurationGroup(),
        TimeToFinishGroup(),
        ReactionCorrectingReactingDurationGroup(),
        CountWordsGroup(),
        CountLettersGroup(),
        TimeBetweenLettersGroup(),
        RatioAndSlopeGroup(),
        FlagAutoCompleteGroup(),
        TypingTremorMeasuresGroup(),
    ]
    kwargs = {"task_name": TASK_NAME}


process_typing = process_factory(
    task_name=TASK_NAME,
    steps=BDHTypingSteps(),
    codes=("typing", "typing-activity"),
    supported_type=BDHReading,
)
