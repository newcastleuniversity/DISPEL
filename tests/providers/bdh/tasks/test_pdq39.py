"""Tests for :mod:`dispel.providers.bdh.tasks.pdq39`."""

from dispel.providers.bdh.tasks.pdq39 import process_pdq_39
from tests.processing.helper import assert_level_values


def test_process_pdq_39(example_reading_pdq39):
    """Test we can process pdq 39 reading."""
    process_pdq_39(example_reading_pdq39)
    pdq_39_scores = {
        "pdq39-mobility_score": 0.0,
        "pdq39-activities_of_daily_living _score": 0.0,
        "pdq39-emotional_well_being_score": 0.0,
        "pdq39-stigma_score": 0.0,
        "pdq39-social_support_score": 0.0,
        "pdq39-cognition_score": 0.0,
        "pdq39-communication_score": 0.0,
        "pdq39-bodily_discomfort_score": 0.0,
        "pdq39-total_score": 0.0,
    }
    assert_level_values(example_reading_pdq39, "pdq39", pdq_39_scores)
