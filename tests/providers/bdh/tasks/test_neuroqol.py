"""Tests for :mod:`dispel.providers.bdh.tasks.neuroqol.py`."""

from dispel.providers.bdh.tasks.neuroqol import process_neuroqol
from tests.processing.helper import assert_level_values


def test_process_neuroqol(example_reading_neuroqol):
    """Test we can process neuroqol reading."""
    process_neuroqol(example_reading_neuroqol)

    neuroqol_t_scores = {
        "neuroqol-upper_extremity-t_score": 41.86741276648121,
        "neuroqol-upper_extremity-standard_error": 0.2676179022904041,
        "neuroqol-lower_extremity-t_score": 49.156606300560156,
        "neuroqol-lower_extremity-standard_error": 0.2718744367287741,
        "neuroqol-sleep-t_score": 54.60882639587884,
        "neuroqol-sleep-standard_error": 0.3048417246230962,
        "neuroqol-fatigue-t_score": 41.98785142936781,
        "neuroqol-fatigue-standard_error": 0.25522795310636054,
        "neuroqol-anxiety-t_score": 48.717436044154695,
        "neuroqol-anxiety-standard_error": 0.2683579119863332,
        "neuroqol-depression-t_score": 53.44896347710136,
        "neuroqol-depression-standard_error": 0.2348916415677409,
        "neuroqol-stigma-t_score": 60.057676718634646,
        "neuroqol-stigma-standard_error": 0.265701027845108,
        "neuroqol-cognitive_function-t_score": 48.32257844440012,
        "neuroqol-cognitive_function-standard_error": 0.2953986350264367,
        "neuroqol-ability_participate_social_roles-t_score": 36.35947313540824,
        "neuroqol-ability_participate_social_roles-standard_error": 0.1739517884166932,
        "neuroqol-satisfaction_social_roles-t_score": 39.663950345960565,
        "neuroqol-satisfaction_social_roles-standard_error": 0.13016376832183252,
    }

    assert_level_values(example_reading_neuroqol, "all_levels", neuroqol_t_scores)
