"""Test cases for :mod:`dispel.providers.ads.tasks.msis29`."""
import pandas as pd

from dispel.data.core import Reading
from dispel.data.levels import Level
from dispel.processing import process
from dispel.providers.generic.surveys import (
    SURVEY_RESPONSES_LEVEL_ID,
    ConcatenateSurveyLevels,
)
from dispel.providers.generic.tasks.msis29 import process_msis29
from tests.processing.helper import assert_measure_from_reading_value


def test_concatenate_survey_levels(example_reading_msis29_multiple_answers):
    """Test concatenation of individual questions in levels."""
    level_count = len(example_reading_msis29_multiple_answers.levels)

    step = ConcatenateSurveyLevels("idMsis29")
    res = process(example_reading_msis29_multiple_answers, step).get_reading()

    assert isinstance(res, Reading)

    level = res.get_level("survey_responses")
    assert isinstance(level, Level)
    assert level.has_raw_data_set("responses")

    data_set = level.get_raw_data_set("responses")
    data = data_set.data

    assert isinstance(data, pd.DataFrame)

    expected_columns = {
        "question_id",
        "ts_question_displayed",
        "ts_question_hidden",
        "ts_response",
        "response",
        "response_time",
    }
    assert set(data.columns) == expected_columns
    assert len(data) == level_count


def _assert_msis_29_measures(
    reading, expected_all, expected_physical, expected_psychological
):
    res = process_msis29(reading).get_reading()

    assert_measure_from_reading_value(
        res, "msis29-ans_all", expected_all, SURVEY_RESPONSES_LEVEL_ID
    )
    assert_measure_from_reading_value(
        res, "msis29-ans_phys", expected_physical, SURVEY_RESPONSES_LEVEL_ID
    )
    assert_measure_from_reading_value(
        res, "msis29-ans_psy", expected_psychological, SURVEY_RESPONSES_LEVEL_ID
    )


def test_process_msis29_ads(example_reading_msis29):
    """Unit test to ensure the MSIS29 scale measures are well computed."""
    _assert_msis_29_measures(example_reading_msis29, 73, 50, 23)


def test_process_msis29_ads_multiple_answers(example_reading_msis29_multiple_answers):
    """Test multiple answers."""
    _assert_msis_29_measures(example_reading_msis29_multiple_answers, 75, 55, 20)


def test_process_msis29_ads_answer(example_reading_msis29):
    """Unit test to ensure the MSIS29 answer measures are well computed."""
    res = process_msis29(example_reading_msis29).get_reading()

    assert_measure_from_reading_value(
        res, "msis29-q_1-res", 3, SURVEY_RESPONSES_LEVEL_ID
    )
    assert_measure_from_reading_value(
        res, "msis29-q_29-res", 1, SURVEY_RESPONSES_LEVEL_ID
    )
