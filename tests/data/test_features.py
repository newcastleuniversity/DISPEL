"""Tests for :mod:`dispel.data.features`."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from dispel.data.features import (
    FeatureId,
    FeatureSet,
    FeatureValue,
    FeatureValueDefinition,
    FeatureValueDefinitionPrototype,
    row_to_definition,
    row_to_value,
)
from dispel.data.flags import Flag
from dispel.data.raw import MissingColumnError
from dispel.data.validators import GREATER_THAN_ZERO
from dispel.data.values import AbbreviatedValue as AV
from dispel.data.values import ValueDefinition


def test_task_feature_id():
    """Test the task feature id representation."""
    feature1 = FeatureId(task_name="x", feature_name="y")
    assert feature1.id == "x-y"

    # optional modality
    feature2 = FeatureId(task_name="x", feature_name="y", modalities=["z"])
    assert feature2.id == "x-z-y"

    # optional aggregation
    feature3 = FeatureId(task_name="x", feature_name="y", aggregation="z")
    assert feature3.id == "x-y-z"

    # optional sub-modality
    feature5 = FeatureId(task_name="x", feature_name="y", modalities=["z", "submod"])
    assert feature5.id == "x-z_submod-y"

    # full options
    feature6 = FeatureId(
        task_name=AV("Cognitive Processing Speed", "CPS"),
        feature_name=AV("reaction time", "rt"),
        modalities=[AV("symbol-to-digit", "std"), AV("second key-set", "key2")],
        aggregation=AV("standard deviation", "std"),
    )
    assert feature6.id == "cps-std_key2-rt-std"


def test_task_feature_definition():
    """Test the task feature definition representation."""
    task_name = AV("Cognitive Processing Speed", "CPS")
    feature_name = AV("reaction time", "rt")
    definition1 = FeatureValueDefinition(
        task_name=task_name,
        feature_name=feature_name,
    )

    assert definition1.name == "CPS reaction time"

    # optional modalities
    definition2 = FeatureValueDefinition(
        task_name=task_name,
        feature_name=feature_name,
        modalities=[AV("symbol-to-digit", "std"), AV("key set 1", "key1")],
    )

    assert definition2.name == "CPS symbol-to-digit key set 1 reaction time"

    # optional aggregation
    definition3 = FeatureValueDefinition(
        task_name=task_name,
        feature_name=feature_name,
        aggregation=AV("standard deviation", "std"),
    )

    assert definition3.name == "CPS standard deviation reaction time"

    # full options
    definition4 = FeatureValueDefinition(
        task_name=task_name,
        feature_name=feature_name,
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


def test_task_feature_definition_prototype():
    """Test formatting is passed onto types having an abbreviated value."""
    prototype = FeatureValueDefinitionPrototype(
        task_name="task", feature_name=AV("Feature {ph}", "feature_{ph}")
    )
    definition = prototype.create_definition(ph=1)
    assert definition.id == "task-feature_1"


def test_feature_set_from_csv(collection_data_frame):
    """Test if the initialization from a csv is corrected for a FeatureSet."""
    data = collection_data_frame.drop_duplicates("feature_id")
    feature_set = FeatureSet.from_data_frame(data)
    feat_02_def = feature_set.get_definition("feat_02")
    feat_02_value = feature_set.get_raw_value("feat_02")

    assert feat_02_def.name == "feature_02"
    assert feat_02_value == 6.0
    assert isinstance(feat_02_value, np.float64)


def test_feature_set_from_csv_missing_input(collection_data_frame):
    """Test the initialization from a csv in case of missing input."""
    with pytest.raises(MissingColumnError):
        FeatureSet.from_data_frame(collection_data_frame.drop("feature_id", 1))


def test_row_to_definition():
    """Test :func:`dispel.data.features.row_to_definition`."""
    row = pd.Series(
        {
            "feature_name": "feat",
            "feature_id": "ft",
            "feature_unit": "s",
            "feature_type": "int16",
        }
    )
    definition = ValueDefinition(id_="ft", name="feat", unit="s", data_type="int16")
    assert row_to_definition(row) == definition


def test_row_to_definition_missing_input():
    """Test missing input for :func:`dispel.data.features.row_to_definition`."""
    row = pd.Series({"feature_name": "feat", "feature_id": "ft", "feature_unit": "s"})
    with pytest.raises(MissingColumnError):
        _ = row_to_definition(row)


@pytest.fixture(scope="module")
def feature_value():
    """Create a fixture for feature value."""
    definition = ValueDefinition(id_="ft", name="feat", unit="s", data_type="float64")
    return FeatureValue(definition, 3.2)


def test_row_to_value(feature_value):
    """Test :func:`dispel.data.features.row_to_definition`."""
    row = pd.Series(
        {
            "feature_name": "feat",
            "feature_id": "ft",
            "feature_unit": "s",
            "feature_type": "float64",
            "feature_value": "3.2",
        }
    )
    assert row_to_value(row) == feature_value


def test_feature_value_set_flag(feature_value):
    """Test setting flag in feature value."""
    feat = deepcopy(feature_value)
    flag = Flag("cps-technical-deviation-ta", "reason")

    assert feat.is_valid
    feat.add_flag(flag)
    assert feat.flag_count == 1
