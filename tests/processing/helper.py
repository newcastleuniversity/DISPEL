"""Helper library to refactor tests.processing."""
import json
import math
import operator
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytest

from dispel.data.collections import FeatureSet
from dispel.data.core import Reading
from dispel.data.levels import LevelId, LevelIdType
from dispel.data.values import DefinitionIdType


def assert_feature_from_reading_value(
    reading: Reading,
    feature_id: DefinitionIdType,
    expected_value: Any,
    level_id: Optional[Union[str, LevelId]] = None,
    comparator=None,
    assertion_message: str = None,
) -> None:
    """Assert equivalence of a feature value computed from a reading."""
    source = reading.get_feature_set(level_id)
    actual_value = source.get_raw_value(feature_id)
    comparator = comparator or operator.eq
    mess = (
        f"Actual value ({actual_value}) did not match expected value "
        f"({expected_value}) when compared with {comparator.__name__}"
    )

    assertion_message = assertion_message or mess
    assert comparator(actual_value, expected_value), assertion_message


def assert_feature_value_almost_equal(
    reading: Reading,
    feature_id: DefinitionIdType,
    expected_value: Any,
    level_id: Optional[Union[str, LevelId]] = None,
) -> None:
    """Assert almost equal for feature value computed from reading."""
    assert_feature_from_reading_value(
        reading,
        feature_id,
        expected_value,
        level_id,
        np.testing.assert_almost_equal,
    )


def assert_feature_value(
    feature_set: FeatureSet,
    feature_id: str,
    expected_value: float,
    relative_error: float = 1e-6,
    absolute_error: float = 1e-6,
):
    """Assert a numerical feature value is equal to an expected value."""
    value = feature_set.get_raw_value(feature_id)
    if value == pytest.approx(expected_value, abs=absolute_error):
        return
    if not math.isnan(expected_value):
        msg = (
            f"Actual value ({value}) of {feature_id} did not match "
            f"expected value ({expected_value}) while allowing "
            f"{relative_error} error."
        )
        assert value == pytest.approx(expected_value, rel=relative_error), msg
    else:
        msg = f"Actual value ({value}) of {feature_id} was not NaN"
        assert math.isnan(value), msg


def assert_dict_values(
    fs: FeatureSet,
    expected_values: Dict[str, Any],
    relative_error: float = 1e-6,
    absolute_error: float = 1e-6,
):
    """Assert a dictionary of numerical feature match expected values.

    The keys of expected_values are the feature_id and the item the expected_values.
    """
    for key in expected_values:
        assert_feature_value(
            fs, key, expected_values[key], relative_error, absolute_error
        )


def assert_level_values(
    reading: Reading,
    level_id: Optional[LevelIdType],
    expected: Dict[str, Any],
    **kwargs,
):
    """Assert a level contains the expected values."""
    fs = reading.get_feature_set(level_id)

    assert isinstance(fs, FeatureSet)
    n_actual = len(fs)
    n_expected = len(expected)
    assert n_actual >= n_expected, f"Expected {n_expected} features but got {n_actual}."

    if n_actual > n_expected:
        untested = {f.id for f in fs.ids() if f.id not in expected}
        warnings.warn(
            f"Some features are not under regression testing "
            f"({n_actual - n_expected}): {untested}",
            UserWarning,
        )

    assert_dict_values(fs, expected, **kwargs)


def assert_unique_feature_ids(reading: Reading):
    """Assert that feature ids are unique."""
    merged = reading.get_merged_feature_set()
    duplicated = [i for i, count in Counter(merged.ids()).items() if count > 1]
    assert (
        not duplicated
    ), f"Reading cannot contain duplicated feature ids: {duplicated}"


def assert_list_feature_ids_not_isinstance(fs: FeatureSet, feature_ids: List[str]):
    """Assert a dictionary of numerical feature matches expected values.

    The keys of expected_values are the feature_id and the item the expected_values.
    """
    for key in feature_ids:
        assert key not in fs


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
                    "[feature-1]": 1,
                    "[feature-2]": 0,
                    ...
                },
                "[level-id-2]": { ... }
            }

    ignore_levels
        By default, the result is a list of tuples with level ID and a dictionary of
        expected results with feature ID as key and values the expected feature values.
        If ``True``, the dictionary of the first entry of the file will be returned
        without the respective level.

    Returns
    -------
    Union[List[Tuple[str, Dict[str, Any]]], Dict[str, Any]]
        Depending on ``ignore_levels``, the result is either a list of tuples consisting
        of the level ID for the expected values and a dictionary containing the feature
        IDs as key and expected results as values or the expected values directly.
    """
    with open(file_name, "r", encoding="utf-8") as file:
        data = json.load(file)
    if ignore_levels:
        return list(data.values())[0]

    results = []
    for level, features in data.items():
        if level == "None":
            level = None
        results.append((level, features))

    return results
