"""Test cases for :mod:`dispel.providers.bdh.tasks.mood`."""

from tests.providers.generic.tasks.test_mood import assert_mood_measures


def test_process_mood_bdh(example_reading_mood):
    """Unit test to ensure the Mood scale measures are well computed."""
    assert_mood_measures(example_reading_mood, 1, 1)
