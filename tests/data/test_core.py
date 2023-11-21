"""Tests for :mod:`dispel.data.core`."""

from copy import deepcopy

import pytest

from dispel.data.core import Evaluation, Reading, Session
from dispel.data.epochs import EpochDefinition
from dispel.data.flags import Flag
from dispel.data.measures import MeasureSet, MeasureValue, MeasureValueDefinition
from dispel.data.raw import RawDataValueDefinition
from tests.conftest import resource_path
from tests.data.test_levels import EXAMPLE_LEVELS

EXAMPLE_PATH = resource_path("example_csv.csv", "stats")

EXAMPLE_PATH_2 = resource_path("learning_framework_example.csv", "stats")

EXAMPLE_PATH_3 = resource_path("learning_one_measure_id_multiple_user_ids.csv", "stats")


@pytest.mark.parametrize("start,end", [(4, 6), (4, 11), (7, 12)])
def test_evaluation_info_times_consistent(start, end):
    """Test if evaluation info can be created with inconsistent time frames."""
    session = Session(start=5, end=10)
    evaluation = Evaluation(uuid="id", start=start, end=end)

    with pytest.raises(ValueError):
        Reading(evaluation, session)


EXAMPLE_READING_RAW_DATA_VALUE_DEFINITIONS = [
    RawDataValueDefinition("a", "A"),
    RawDataValueDefinition("b", "B"),
]


@pytest.fixture(scope="module")
def example_reading():
    """Provide an example reading for testing purposes."""
    return Reading(
        evaluation=Evaluation(
            uuid="eval_uuid",
            start=0,
            end=1,
            definition=EpochDefinition(id_="example"),
        ),
        levels=EXAMPLE_LEVELS,
    )


def test_get_merged_measure_set(example_reading):
    """Test :meth:`dispel.data.core.Reading.get_merged_measure_set`."""
    reading = deepcopy(example_reading)
    all_measure_values = [
        MeasureValue(MeasureValueDefinition(f"id-{i}", f"name {i}"), i)
        for i in range(10)
    ]
    reading.measure_set = MeasureSet(all_measure_values[:3])
    reading.set(MeasureSet(all_measure_values[3:7]), "level_1")
    reading.set(MeasureSet(all_measure_values[7:]), "level_2")

    merged_measure_set = MeasureSet(all_measure_values)
    assert reading.get_merged_measure_set() == merged_measure_set


def test_set_flag(example_reading):
    """Test flag setting."""
    reading = deepcopy(example_reading)
    assert reading.is_valid

    flag = Flag("cps-technical-deviation-ta", "reason")
    reading.set(flag)
    assert reading.flag_count == 1

    level = reading.get_level("level_1")
    assert level.is_valid
    level.set(flag)
    assert level.flag_count == 1
