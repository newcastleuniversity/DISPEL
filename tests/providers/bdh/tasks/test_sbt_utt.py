"""Test cases for :mod:`dispel.providers.bdh.tasks.sbt_utt`."""

import pytest

from dispel.data.core import Reading
from dispel.data.measures import MeasureSet
from dispel.providers.bdh.tasks.sbt_utt import process_sbt_utt
from tests.conftest import resource_path
from tests.processing.helper import (
    assert_dict_values,
    assert_unique_measure_ids,
    read_results,
)

EXAMPLE_SBTUTT_PATH = resource_path(
    "SBT-UTT/expected/expected_values.json", "providers.bdh"
)


def test_process_utt_wo_turns(example_reading_sbt_utt_no_turns):
    """Test if we can process a u-turn when we detect no turns."""
    _ = process_sbt_utt(example_reading_sbt_utt_no_turns)


def test_sbtutt_process(example_reading_processed_sbt_utt):
    """Unit test to ensure the SBT-UTT measures are well computed."""
    expected = read_results(EXAMPLE_SBTUTT_PATH, False)

    measure_set_sbt = example_reading_processed_sbt_utt.get_level("sbt").measure_set
    measure_set_utt = example_reading_processed_sbt_utt.get_level("utt").measure_set

    assert isinstance(measure_set_sbt, MeasureSet)
    assert isinstance(measure_set_utt, MeasureSet)

    assert_dict_values(measure_set_sbt, expected[0][1], relative_error=1e-2)
    assert_dict_values(measure_set_utt, expected[1][1], relative_error=1e-2)
    assert_unique_measure_ids(example_reading_processed_sbt_utt)


@pytest.mark.xfail(raises=ValueError)
def test_sbtutt_process_bdh_input_bug(example_reading_sbt_utt_bug):
    """Ensure SBT-UTT measures are well computed with BDH format."""
    res = process_sbt_utt(example_reading_sbt_utt_bug).get_reading()
    assert isinstance(res, Reading)
