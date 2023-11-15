"""Utility functions for testing."""
import json
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from dispel.data.core import Reading
from dispel.data.features import FeatureSet

DEFAULT_PRECISION = 10


def generate_regression_value(value: Any, precision: int = DEFAULT_PRECISION) -> str:
    """Generate a string representation for the regression value."""
    if pd.isnull(value):
        return "np.NaN"
    if isinstance(value, float) or np.issubdtype(np.dtype(type(value)), np.inexact):
        return str(round(value, precision))
    if isinstance(value, int) or np.issubdtype(np.dtype(type(value)), np.integer):
        return str(value)
    return f"'{value}'"


def generate_regression_dict_from_feature_set(
    feature_set: FeatureSet, precision: int = DEFAULT_PRECISION
) -> str:
    """Generate the regression dictionary for a feature set."""
    values = (
        f"'{f.id}': {generate_regression_value(f.value, precision)}"
        for f in feature_set.values()
    )
    return "{" + ",\n".join(values) + "}"


def generate_regression_dict_from_reading(
    reading: Reading, precision: int = DEFAULT_PRECISION
) -> str:
    """Generate the regression dictionary for a reading."""
    feature_sets = {
        level.id: level.feature_set
        for level in reading.levels
        if not level.feature_set.empty
    }
    if not reading.feature_set.empty:
        feature_sets[None] = reading.feature_set

    def _format_id(id_: Optional[str]) -> str:
        if id_ is None:
            return "None"
        return f"'{id_}'"

    def _format_feature_set(id_: str, fs: FeatureSet) -> str:
        fs_format = generate_regression_dict_from_feature_set(fs, precision)
        return f"({_format_id(id_)},\n{fs_format})"

    res = "@pytest.mark.parametrize('level_id,expected', [\n"
    res += ",\n".join(_format_feature_set(k, v) for k, v in feature_sets.items())
    res += "\n])"

    return res


def get_features(feature_set: FeatureSet) -> Dict[str, Any]:
    """Generate feature results from feature set."""
    res = {}
    for feature in feature_set.values():
        if isinstance(feature.value, np.integer):
            value = int(feature.value)
        else:
            value = float(feature.value)
        res[str(feature.id)] = value
    return res


def generate_results(reading: Reading) -> Dict[str, Dict[str, Any]]:
    """Generate feature results from reading."""
    res = {"None": get_features(reading.feature_set)}
    for level in reading.levels:
        if not level.feature_set.empty:
            res[str(level.id)] = get_features(level.feature_set)
    return res


def write_results(reading: Reading, name: str) -> None:
    """Write feature results in a JSON file."""
    with open(f"{name}.json", "w", encoding="utf-8") as outfile:
        json.dump(generate_results(reading), outfile)
