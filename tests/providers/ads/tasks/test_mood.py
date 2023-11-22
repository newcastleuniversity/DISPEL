"""Test cases for :mod:`dispel.providers.ads.tasks.mood`."""
from tests.providers.generic.tasks.test_mood import assert_mood_measures


def test_process_mood_ads(example_reading_mood):
    """Unit test to ensure the Mood scale measures are well computed."""
    assert_mood_measures(example_reading_mood, 1, 1)


def test_process_mood_ads_multiple_answers(example_reading_mood_multiple_answers):
    """Test processing of multiple mood answers."""
    assert_mood_measures(example_reading_mood_multiple_answers, 1, 5)


def test_multiple_answers_same_question_synthetic(
    example_reading_mood_multiple_answers_same_q_synth,
):
    """Test processing of multiple mood answers."""
    assert_mood_measures(example_reading_mood_multiple_answers_same_q_synth, 1, 1)


def test_multiple_answers_same_question_real(
    example_reading_mood_multiple_answers_same_q_real,
):
    """Test processing of multiple mood answers."""
    assert_mood_measures(example_reading_mood_multiple_answers_same_q_real, 2, 3)
