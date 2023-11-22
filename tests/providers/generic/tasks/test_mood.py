"""Tests for :mod:`dispel.providers.generic.tasks.mood`."""

from dispel.providers.generic.surveys import SURVEY_RESPONSES_LEVEL_ID
from dispel.providers.generic.tasks.mood import process_mood
from tests.processing.helper import assert_measure_from_reading_value


def assert_mood_measures(reading, expected_mood, expected_physical):
    """Test :func:`dispel.providers.generic.tasks.mood.process_mood`."""
    res = process_mood(reading).get_reading()

    assert_measure_from_reading_value(
        res, "mood-q_psy-res", expected_mood, SURVEY_RESPONSES_LEVEL_ID
    )
    assert_measure_from_reading_value(
        res, "mood-q_phys-res", expected_physical, SURVEY_RESPONSES_LEVEL_ID
    )
