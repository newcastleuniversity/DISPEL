"""Tests for :mod:`dispel.data.measures`."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from dispel.data.flags import Flag
from dispel.data.measures import (
    MeasureId,
    MeasureSet,
    MeasureValue,
    MeasureValueDefinition,
    MeasureValueDefinitionPrototype,
    row_to_definition,
    row_to_value,
)
from dispel.data.raw import MissingColumnError
from dispel.data.validators import GREATER_THAN_ZERO
from dispel.data.values import AbbreviatedValue as AV
from dispel.data.values import ValueDefinition


def test_task_measure_id():
    """Test the task measure id representation."""
    measure1 = MeasureId(task_name="x", measure_name="y")
    assert measure1.id == "x-y"

    # optional modality
    measure2 = MeasureId(task_name="x", measure_name="y", modalities=["z"])
    assert measure2.id == "x-z-y"

    # optional aggregation
    measure3 = MeasureId(task_name="x", measure_name="y", aggregation="z")
    assert measure3.id == "x-y-z"

    # optional sub-modality
    measure5 = MeasureId(task_name="x", measure_name="y", modalities=["z", "submod"])
    assert measure5.id == "x-z_submod-y"

    # full options
    measure6 = MeasureId(
        task_name=AV("Cognitive Processing Speed", "CPS"),
        measure_name=AV("reaction time", "rt"),
        modalities=[AV("symbol-to-digit", "std"), AV("second key-set", "key2")],
        aggregation=AV("standard deviation", "std"),
    )
    assert measure6.id == "cps-std_key2-rt-std"


def test_task_measure_definition():
    """Test the task measure definition representation."""
    task_name = AV("Cognitive Processing Speed", "CPS")
    measure_name = AV("reaction time", "rt")
    definition1 = MeasureValueDefinition(
        task_name=task_name,
        measure_name=measure_name,
    )

    assert definition1.name == "CPS reaction time"

    # optional modalities
    definition2 = MeasureValueDefinition(
        task_name=task_name,
        measure_name=measure_name,
        modalities=[AV("symbol-to-digit", "std"), AV("key set 1", "key1")],
    )

    assert definition2.name == "CPS symbol-to-digit key set 1 reaction time"

    # optional aggregation
    definition3 = MeasureValueDefinition(
        task_name=task_name,
        measure_name=measure_name,
        aggregation=AV("standard deviation", "std"),
    )

    assert definition3.name == "CPS standard deviation reaction time"

    # full options
    definition4 = MeasureValueDefinition(
        task_name=task_name,
        measure_name=measure_name,
        unit="s",
        description="The standard deviation of reaction time from stimuli to "
        "button press in.",
        data_type="float64",
        validator=GREATER_THAN_ZERO,
        modalities=[AV("symbol-to-digit", "std"), AV("key set 1", "key1")],
        aggregation=AV("standard deviation", "std"),
    )

    assert (
        definition4.name == "CPS symbol-to-digit key set 1 standard "
        "deviation reaction time"
    )
    assert definition4.id == "cps-std_key1-rt-std"

    # test hashing
    assert hash(definition4) == hash(definition4)


def test_task_measure_definition_prototype():
    """Test formatting is passed onto types having an abbreviated value."""
    prototype = MeasureValueDefinitionPrototype(
        task_name="task", measure_name=AV("Measure {ph}", "measure_{ph}")
    )
    definition = prototype.create_definition(ph=1)
    assert definition.id == "task-measure_1"


def test_measure_set_from_csv(collection_data_frame):
    """Test if the initialization from a csv is corrected for a MeasureSet."""
    data = collection_data_frame.drop_duplicates("measure_id")
    measure_set = MeasureSet.from_data_frame(data)
    feat_02_def = measure_set.get_definition("feat_02")
    feat_02_value = measure_set.get_raw_value("feat_02")

    assert feat_02_def.name == "measure_02"
    assert feat_02_value == 6.0
    assert isinstance(feat_02_value, np.float64)


def test_measure_set_from_csv_missing_input(collection_data_frame):
    """Test the initialization from a csv in case of missing input."""
    with pytest.raises(MissingColumnError):
        MeasureSet.from_data_frame(collection_data_frame.drop("measure_id", 1))


def test_row_to_definition():
    """Test :func:`dispel.data.measures.row_to_definition`."""
    row = pd.Series(
        {
            "measure_name": "feat",
            "measure_id": "ft",
            "measure_unit": "s",
            "measure_type": "int16",
        }
    )
    definition = ValueDefinition(id_="ft", name="feat", unit="s", data_type="int16")
    assert row_to_definition(row) == definition


def test_row_to_definition_missing_input():
    """Test missing input for :func:`dispel.data.measures.row_to_definition`."""
    row = pd.Series({"measure_name": "feat", "measure_id": "ft", "measure_unit": "s"})
    with pytest.raises(MissingColumnError):
        _ = row_to_definition(row)


@pytest.fixture(scope="module")
def measure_value():
    """Create a fixture for measure value."""
    definition = ValueDefinition(id_="ft", name="feat", unit="s", data_type="float64")
    return MeasureValue(definition, 3.2)


def test_row_to_value(measure_value):
    """Test :func:`dispel.data.measures.row_to_definition`."""
    row = pd.Series(
        {
            "measure_name": "feat",
            "measure_id": "ft",
            "measure_unit": "s",
            "measure_type": "float64",
            "measure_value": "3.2",
        }
    )
    assert row_to_value(row) == measure_value


def test_measure_value_set_flag(measure_value):
    """Test setting flag in measure value."""
    feat = deepcopy(measure_value)
    flag = Flag("cps-technical-deviation-ta", "reason")

    assert feat.is_valid
    feat.add_flag(flag)
    assert feat.flag_count == 1
