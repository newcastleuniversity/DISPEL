"""Unit tests for :mod:`dispel.docutils`."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from dispel.data.validators import RangeValidator, SetValidator
from dispel.data.values import ValueDefinition
from dispel.docutils import (
    convert_measure_value_definition_to_dict,
    get_measure_value_definitions_data_frame,
    measure_value_definitions_to_data_frame,
)
from dispel.processing import ProcessingStep
from dispel.processing.extract import ExtractStep
from dispel.processing.trace import collect_measure_value_definitions
from dispel.providers.registry import PROCESSING_STEPS


def test_convert_measure_value_definition_to_dict():
    """Test convert_measure_value_definition_to_dict."""
    definition = ValueDefinition(
        id_="test",
        name="Test",
        unit="n/a",
        description="Test description",
        data_type="int64",
    )
    step = ExtractStep("dummy", lambda *_: None, definition)

    assert convert_measure_value_definition_to_dict(step, definition) == {
        "id": "test",
        "name": "Test",
        "description": "Test description",
        "unit": "n/a",
        "data_type": "int64",
        "values_min": None,
        "values_max": None,
        "values_in": None,
        "produced_by": repr(step),
    }


@pytest.mark.parametrize(
    "validator,values_min,values_max",
    [
        (RangeValidator(lower_bound=0), 0, None),
        (RangeValidator(upper_bound=0), None, 0),
        (RangeValidator(lower_bound=0, upper_bound=1), 0, 1),
    ],
)
def test_convert_measure_value_definition_to_dict_range_validator(
    validator, values_min, values_max
):
    """Test convert_measure_value_definition_to_dict with range validator."""
    definition = ValueDefinition(id_="test", name="Test", validator=validator)
    step = ExtractStep("dummy", lambda *_: None, definition)

    assert convert_measure_value_definition_to_dict(step, definition) == {
        "id": "test",
        "name": "Test",
        "description": None,
        "unit": None,
        "data_type": None,
        "values_min": values_min,
        "values_max": values_max,
        "values_in": None,
        "produced_by": repr(step),
    }


@pytest.mark.parametrize(
    "values,expected",
    [({1: "one", 2: "two"}, {1: "one", 2: "two"}), ([1, 2, 3], {1, 2, 3})],
)
def test_convert_measure_value_definition_to_dict_set_validator(values, expected):
    """Test convert_measure_value_definition_to_dict with set validator."""
    definition = ValueDefinition(
        id_="test", name="Test", validator=SetValidator(values)
    )
    step = ExtractStep("dummy", lambda *_: None, definition)

    assert convert_measure_value_definition_to_dict(step, definition) == {
        "id": "test",
        "name": "Test",
        "description": None,
        "unit": None,
        "data_type": None,
        "values_min": None,
        "values_max": None,
        "values_in": expected,
        "produced_by": repr(step),
    }


def test_measure_value_definitions_to_data_frame():
    """Test measure value definitions to data frame."""
    step = ProcessingStep()
    definition = ValueDefinition("a", "A")
    data = measure_value_definitions_to_data_frame({(step, definition)})
    expected = pd.DataFrame(
        [
            {
                "id": "a",
                "name": "A",
                "description": None,
                "unit": None,
                "data_type": None,
                "values_min": None,
                "values_max": None,
                "values_in": None,
                "produced_by": repr(step),
            }
        ]
    )
    assert_frame_equal(data, expected)


def test_get_measure_value_definitions_data_frame():
    """Test get measure value definitions data frame."""
    data = get_measure_value_definitions_data_frame()
    assert isinstance(data, pd.DataFrame)
    expected = {
        "id",
        "name",
        "description",
        "unit",
        "data_type",
        "values_min",
        "values_max",
        "values_in",
        "produced_by",
        "measure_name",
        "aggregation",
        "task_name",
        *[f"modality_{i}" for i in range(4)],
    }
    assert set(data.columns) == expected
    assert not data.empty


@pytest.mark.parametrize(
    "steps",
    (
        pytest.param(v, id=f"{codes}-{rtype}")
        for (codes, rtype), v in PROCESSING_STEPS.items()
        if codes != ("voice-activity",)
        # FIXME: remove once voice measures are implemented
    ),
)
def test_measure_definition_generation(steps):
    """A simple test to ensure measure value definitions can be collected."""
    res = list(collect_measure_value_definitions(steps))
    assert len(res) > 0
