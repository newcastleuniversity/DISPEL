"""Tests for :mod:`dispel.providers.ads.tasks.passive`."""
from copy import deepcopy

from dispel.data.collections import MeasureSet
from dispel.data.core import Reading
from dispel.providers.ads.tasks.passive import process_passive
from tests.processing.helper import assert_dict_values


def test_process_passive(example_reading_passive):
    """Unit test to ensure the Passive measures are well computed."""
    reading = deepcopy(example_reading_passive)
    process_passive(reading)
    assert isinstance(reading, Reading)

    ms = reading.get_level("passive").measure_set
    assert isinstance(ms, MeasureSet)

    expected_pt_measures = {
        "passive-steps": 2308,
        "passive-active_duration": 48,
        "passive-avg_act_pace-mean": 1.7305275,
        "passive-avg_act_pace-std": 2.2213461,
        "passive-avg_act_pace-median": 0.91000426,
        "passive-avg_act_pace-min": 0.54469085,
        "passive-avg_act_pace-max": 12.202964,
        "passive-avg_act_pace-q95": 5.494567,
    }
    assert_dict_values(ms, expected_pt_measures)
