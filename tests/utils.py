"""Utility functions for testing."""
import json
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from dispel.data.core import Reading
from dispel.data.measures import MeasureSet

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


def generate_regression_dict_from_measure_set(
    measure_set: MeasureSet, precision: int = DEFAULT_PRECISION
) -> str:
    """Generate the regression dictionary for a measure set."""
    values = (
        f"'{f.id}': {generate_regression_value(f.value, precision)}"
        for f in measure_set.values()
    )
    return "{" + ",\n".join(values) + "}"


def generate_regression_dict_from_reading(
    reading: Reading, precision: int = DEFAULT_PRECISION
) -> str:
    """Generate the regression dictionary for a reading."""
    measure_sets = {
        level.id: level.measure_set
        for level in reading.levels
        if not level.measure_set.empty
    }
    if not reading.measure_set.empty:
        measure_sets[None] = reading.measure_set

    def _format_id(id_: Optional[str]) -> str:
        if id_ is None:
            return "None"
        return f"'{id_}'"

    def _format_measure_set(id_: str, ms: MeasureSet) -> str:
        fs_format = generate_regression_dict_from_measure_set(ms, precision)
        return f"({_format_id(id_)},\n{fs_format})"

    res = "@pytest.mark.parametrize('level_id,expected', [\n"
    res += ",\n".join(_format_measure_set(k, v) for k, v in measure_sets.items())
    res += "\n])"

    return res


def get_measures(measure_set: MeasureSet) -> Dict[str, Any]:
    """Generate measure results from measure set."""
    res = {}
    for measure in measure_set.values():
        if isinstance(measure.value, np.integer):
            value = int(measure.value)
        else:
            value = float(measure.value)
        res[str(measure.id)] = value
    return res


def generate_results(reading: Reading) -> Dict[str, Dict[str, Any]]:
    """Generate measure results from reading."""
    res = {"None": get_measures(reading.measure_set)}
    for level in reading.levels:
        if not level.measure_set.empty:
            res[str(level.id)] = get_measures(level.measure_set)
    return res


def write_results(reading: Reading, name: str) -> None:
    """Write measure results in a JSON file."""
    with open(f"{name}.json", "w", encoding="utf-8") as outfile:
        json.dump(generate_results(reading), outfile)
