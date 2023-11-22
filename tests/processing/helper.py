"""Helper library to refactor tests.processing."""
import json
import math
import operator
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest

from dispel.data.collections import MeasureSet
from dispel.data.core import Reading
from dispel.data.levels import LevelId, LevelIdType
from dispel.data.values import DefinitionIdType


def assert_measure_from_reading_value(
    reading: Reading,
    measure_id: DefinitionIdType,
    expected_value: Any,
    level_id: Optional[Union[str, LevelId]] = None,
    comparator=None,
    assertion_message: str = None,
) -> None:
    """Assert equivalence of a measure value computed from a reading."""
    source = reading.get_measure_set(level_id)
    actual_value = source.get_raw_value(measure_id)
    comparator = comparator or operator.eq
    mess = (
        f"Actual value ({actual_value}) did not match expected value "
        f"({expected_value}) when compared with {comparator.__name__}"
    )

    assertion_message = assertion_message or mess
    assert comparator(actual_value, expected_value), assertion_message


def assert_measure_value_almost_equal(
    reading: Reading,
    measure_id: DefinitionIdType,
    expected_value: Any,
    level_id: Optional[Union[str, LevelId]] = None,
) -> None:
    """Assert almost equal for measure value computed from reading."""
    assert_measure_from_reading_value(
        reading,
        measure_id,
        expected_value,
        level_id,
        np.testing.assert_almost_equal,
    )


def assert_measure_value(
    measure_set: MeasureSet,
    measure_id: str,
    expected_value: float,
    relative_error: float = 1e-6,
    absolute_error: float = 1e-6,
):
    """Assert a numerical measure value is equal to an expected value."""
    value = measure_set.get_raw_value(measure_id)
    if value == pytest.approx(expected_value, abs=absolute_error):
        return
    if not math.isnan(expected_value):
        msg = (
            f"Actual value ({value}) of {measure_id} did not match "
            f"expected value ({expected_value}) while allowing "
            f"{relative_error} error."
        )
        assert value == pytest.approx(expected_value, rel=relative_error), msg
    else:
        msg = f"Actual value ({value}) of {measure_id} was not NaN"
        assert math.isnan(value), msg


def assert_dict_values(
    ms: MeasureSet,
    expected_values: Dict[str, Any],
    relative_error: float = 1e-6,
    absolute_error: float = 1e-6,
):
    """Assert a dictionary of numerical measure match expected values.

    The keys of expected_values are the measure_id and the item the expected_values.
    """
    for key in expected_values:
        assert_measure_value(
            ms, key, expected_values[key], relative_error, absolute_error
        )


def assert_level_values(
    reading: Reading,
    level_id: Optional[LevelIdType],
    expected: Dict[str, Any],
    **kwargs,
):
    """Assert a level contains the expected values."""
    ms = reading.get_measure_set(level_id)

    assert isinstance(ms, MeasureSet)
    n_actual = len(ms)
    n_expected = len(expected)
    assert n_actual >= n_expected, f"Expected {n_expected} measures but got {n_actual}."

    if n_actual > n_expected:
        untested = {f.id for f in ms.ids() if f.id not in expected}
        warnings.warn(
            f"Some measures are not under regression testing "
            f"({n_actual - n_expected}): {untested}",
            UserWarning,
        )

    assert_dict_values(ms, expected, **kwargs)


def assert_unique_measure_ids(reading: Reading):
    """Assert that measure ids are unique."""
    merged = reading.get_merged_measure_set()
    duplicated = [i for i, count in Counter(merged.ids()).items() if count > 1]
    assert (
        not duplicated
    ), f"Reading cannot contain duplicated measure ids: {duplicated}"


def assert_list_measure_ids_not_isinstance(ms: MeasureSet, measure_ids: List[str]):
    """Assert a dictionary of numerical measure matches expected values.

    The keys of expected_values are the measure_id and the item the expected_values.
    """
    for key in measure_ids:
        assert key not in ms


def read_results(
    file_name: str, ignore_levels: bool = False
) -> Union[List[Tuple[str, Dict[str, Any]]], Dict[str, Any]]:
    """Read expected test results from a json file.

    Parameters
    ----------
    file_name
        The path to the JSON file containing the expected results. The expected format
        of the JSON file is as follows:

        .. code-block:: json

            {
                "[level-id-1]": {
                    "[measure-1]": 1,
                    "[measure-2]": 0,
                    ...
                },
                "[level-id-2]": { ... }
            }

    ignore_levels
        By default, the result is a list of tuples with level ID and a dictionary of
        expected results with measure ID as key and values the expected measure values.
        If ``True``, the dictionary of the first entry of the file will be returned
        without the respective level.

    Returns
    -------
    Union[List[Tuple[str, Dict[str, Any]]], Dict[str, Any]]
        Depending on ``ignore_levels``, the result is either a list of tuples consisting
        of the level ID for the expected values and a dictionary containing the measure
        IDs as key and expected results as values or the expected values directly.
    """
    with open(file_name, "r", encoding="utf-8") as file:
        data = json.load(file)
    if ignore_levels:
        return list(data.values())[0]

    results = []
    for level, measures in data.items():
        if level == "None":
            level = None
        results.append((level, measures))

    return results
