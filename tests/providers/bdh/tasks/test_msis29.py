"""Test cases for :mod:`dispel.providers.bdh.tasks.msis29`."""

from dispel.providers.generic.surveys import SURVEY_RESPONSES_LEVEL_ID
from dispel.providers.generic.tasks.msis29 import process_msis29
from tests.processing.helper import assert_feature_from_reading_value


def test_process_msis29_bdh_answer(example_reading_msis29):
    """Unit test to ensure the MSIS29 answer features are well computed."""
    res = process_msis29(example_reading_msis29).get_reading()

    assert_feature_from_reading_value(
        res, "msis29-q_1-res", 1, SURVEY_RESPONSES_LEVEL_ID
    )
    assert_feature_from_reading_value(
        res, "msis29-q_29-res", 1, SURVEY_RESPONSES_LEVEL_ID
    )
