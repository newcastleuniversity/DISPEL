"""Test cases for :mod:`dispel.providers.bdh.tasks.alsfrs`."""

from dispel.providers.bdh.tasks.alsfrs import process_als_frs
from tests.processing.helper import assert_level_values


def test_process_als_frs(example_reading_alsfrs):
    """Test we can process als-frs reading."""
    process_als_frs(example_reading_alsfrs)
    als_frs_scores = {
        "alsfrs-total_score": 48,
        "alsfrs-speech_score": 4,
        "alsfrs-salivation_score": 4,
        "alsfrs-swallowing_score": 4,
        "alsfrs-handwritting_score": 4,
        "alsfrs-cutting_food_score": 4,
        "alsfrs-dressing_and_hygiene_score": 4,
        "alsfrs-turning_in_bed_score": 4,
        "alsfrs-walking_score": 4,
        "alsfrs-climbing_stairs_score": 4,
        "alsfrs-dyspnea_score": 4,
        "alsfrs-orthopnea_score": 4,
        "alsfrs-respiratory_insufficiency_score": 4,
    }
    assert_level_values(example_reading_alsfrs, "alsfrs", als_frs_scores)
