"""Tests for :mod:`dispel.data.collections`."""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal

from dispel.data.collections import FeatureCollection, FeatureNotFound, SubjectNotFound
from dispel.data.core import Evaluation, Reading, Session
from dispel.data.epochs import EpochDefinition
from dispel.data.features import FeatureSet
from dispel.data.values import ValueDefinition
from tests.conftest import resource_path

COLLECTION_CSV_PATH = resource_path("collection-example.csv", "data")


@pytest.fixture(scope="module")
def readings():
    """Fixture for multiple readings."""
    session = Session(start=5, end=10, definition=EpochDefinition(id_="s1"))
    evaluation1 = Evaluation(
        uuid="id1",
        start=6,
        end=9,
        definition=EpochDefinition(id_="code1"),
        user_id="user1",
    )
    evaluation2 = Evaluation(
        uuid="id2",
        start=5.5,
        end=8.5,
        definition=EpochDefinition(id_="code2"),
        user_id="user2",
    )

    definition1 = ValueDefinition("f1", "feature 1")
    definition2 = ValueDefinition("f2", "feature 2")
    definition3 = ValueDefinition("f3", "feature 3")
    definition4 = ValueDefinition("f4", "feature 4")

    feature_set1 = FeatureSet([7, 3], [definition1, definition2])
    feature_set2 = FeatureSet([-1, 0], [definition3, definition4])

    reading1 = Reading(evaluation1, session, feature_set=feature_set1)
    reading2 = Reading(evaluation2, session, feature_set=feature_set2)
    readings = [reading1, reading2]
    return readings


def test_feature_collection_from_feature_set(readings):
    """Test :meth:`dispel.data.collections.FeatureCollection.from_feature_set`."""
    evaluation = Evaluation(
        uuid="id3",
        start=4,
        end=10,
        definition=EpochDefinition(id_="code1"),
        user_id="user1",
    )
    fs = readings[0].feature_set + readings[1].feature_set
    fc1 = FeatureCollection.from_feature_set(
        fs, readings[0].evaluation, readings[0].session
    )
    fc2 = FeatureCollection.from_feature_set(fs, evaluation, readings[0].session)

    fc = fc1 + fc2

    assert len(fc) == 8
    assert set(fc.data["evaluation_uuid"]) == {"id1", "id3"}

    eval_id1_counts = set(fc.data[fc.data["evaluation_uuid"] == "id1"]["trial"])
    assert eval_id1_counts == {2}
    eval_id3_counts = set(fc.data[fc.data["evaluation_uuid"] == "id3"]["trial"])
    assert eval_id3_counts == {1}


def test_feature_collection_from_data_frame(collection_data_frame):
    """Test creating a collection from a CSV and a data frame."""
    fc = FeatureCollection.from_csv(COLLECTION_CSV_PATH)
    expected = collection_data_frame.dropna(subset=["feature_value"])

    for column in collection_data_frame:
        assert_series_equal(expected[column], fc.data[column])

    assert len(fc.feature_definitions) == 2
    feat_def = fc.get_feature_definition("feat_01")
    assert feat_def.name == "feature_01"
    assert feat_def.unit == "s"


def test_feature_collection_from_reading(readings):
    """Test the feature collection from one reading."""
    fc = FeatureCollection.from_reading(readings[0])

    assert isinstance(fc, FeatureCollection)
    assert len(fc) == 2
    assert fc.size == 36

    expected_feature_ids = pd.Series({0: "f1", 1: "f2"}, name="feature_id")
    assert_series_equal(fc.data.feature_id, expected_feature_ids)

    expected_feature_definitions = set(readings[0].feature_set.definitions())
    assert set(fc.feature_definitions) == expected_feature_definitions


def test_feature_collection_from_readings(readings):
    """Test the feature collection from multiple readings."""
    fc = FeatureCollection.from_readings(readings)

    assert isinstance(fc, FeatureCollection)
    assert len(fc) == 4
    assert fc.size == 72

    expected_feature_ids = pd.Series(
        {0: "f3", 1: "f4", 2: "f1", 3: "f2"}, name="feature_id"
    )
    assert_series_equal(fc.data.feature_id, expected_feature_ids)

    expected_subject_ids = pd.Series(
        {0: "user2", 1: "user2", 2: "user1", 3: "user1"}, name="subject_id"
    )
    assert_series_equal(fc.data.subject_id, expected_subject_ids)


def test_feature_collection_from_data_missing_input(collection_data_frame):
    """Test the initialization from a csv in case of missing input."""
    with pytest.raises(KeyError):
        FeatureCollection.from_data_frame(collection_data_frame.drop("feature_id", 1))


@pytest.fixture
def feature_collection_example(collection_data_frame):
    """Get a fixture of a feature collection example."""
    return FeatureCollection.from_data_frame(collection_data_frame)


def test_extend_feature_collection(feature_collection_example, readings):
    """Test :meth:`dispel.data.collections.FeatureCollection.extend`."""
    fc = feature_collection_example
    fc1 = FeatureCollection.from_reading(readings[0])
    fc.extend(fc1)

    assert set(fc.subject_ids) == {"user1", "user01", "user02"}
    assert len(fc) == 11

    fc2 = FeatureCollection.from_readings(readings)
    fc.extend(fc2)
    assert set(fc.subject_ids) == {"user1", "user2", "user01", "user02"}
    assert len(fc) == 13


def test_feature_collection_get_data(feature_collection_example):
    """Test :meth:`dispel.data.collections.FeatureCollection.get_data`."""
    fc = feature_collection_example

    data1 = fc.get_data()
    assert data1.equals(fc.data)

    data2 = fc.get_data(subject_id="user01")
    assert data2.equals(fc.data[fc.data["subject_id"] == "user01"])

    data3 = fc.get_data(feature_id="feat_02")
    assert data3.equals(fc.data[fc.data["feature_id"] == "feat_02"])

    data3 = fc.get_data(subject_id="user02", feature_id="feat_02")
    expected_data_frame = fc.data[
        (fc.data["subject_id"] == "user02") & (fc.data["feature_id"] == "feat_02")
    ]
    assert_frame_equal(data3, expected_data_frame)


@pytest.mark.parametrize(
    "subject_id, feature_id, error",
    [
        ("user", None, SubjectNotFound),
        (None, "feat", FeatureNotFound),
        ("user01", "feat", FeatureNotFound),
        ("user", "feat_02", SubjectNotFound),
    ],
)
def test_feature_collection_get_data_error(
    feature_collection_example, subject_id, feature_id, error
):
    """Test :meth:`dispel.data.collections.FeatureCollection.get_data`."""
    fc = feature_collection_example

    with pytest.raises(error):
        _ = fc.get_data(subject_id=subject_id, feature_id=feature_id)


def test_feature_collection_get_features_over_time(feature_collection_example):
    """Test feature value retrieval over time."""
    fc = feature_collection_example
    subject_id = "user01"
    feature_id = "feat_02"

    data = fc.get_feature_values_over_time(subject_id=subject_id, feature_id=feature_id)
    dates = ["01/01/2020", "01/02/2020"]
    expected = pd.Series(
        [3.0, 4.0],
        index=pd.Series(map(pd.Timestamp, dates), name="start_date"),
        name=feature_id,
    )
    assert_series_equal(data, expected)


def test_feature_collection_get_features_by_trials(feature_collection_example):
    """Test feature value retrieval over sessions."""
    fc = feature_collection_example
    feature_id = "feat_02"

    data = fc.get_feature_values_by_trials(feature_id=feature_id)
    expected = pd.DataFrame(
        {
            1: {"user01": 3.0, "user02": 6.0},
            2: {"user01": 4.0, "user02": 7.0},
            3: {"user01": np.nan, "user02": 6.0},
        }
    )
    expected.index.name = "subject_id"
    expected.columns.name = "trial"
    assert_frame_equal(data, expected)


def test_feature_collection_get_features_by_trials_empty_evaluation():
    """Test that evaluations with NaN feature values do not lead to trials."""
    time = pd.date_range("now", periods=7, freq="1d")
    data = pd.DataFrame(
        {
            "subject_id": ["user1"] * 3 + ["user2"] * 4,
            "session_uuid": range(7),
            "evaluation_uuid": range(7),
            "feature_value": [1.0, np.NaN, 2.0, 2.0, 4.0, None, 8.0],
            "start_date": time,
            "end_date": time + pd.Timedelta("30s"),
        }
    )
    data["session_code"] = "nd"
    data["evaluation_code"] = "eval"
    data["feature_name"] = "feature 01"
    data["feature_id"] = "feat_01"
    data["feature_type"] = "float64"
    data["feature_unit"] = "s"
    data["is_finished"] = True

    fc = FeatureCollection.from_data_frame(data)
    res = fc.get_feature_values_by_trials("feat_01")
    expected = pd.DataFrame(
        {
            1: {"user1": 1.0, "user2": 2.0},
            2: {"user1": 2.0, "user2": 4.0},
            3: {"user1": np.NaN, "user2": 8.0},
        }
    )
    expected.index.name = "subject_id"
    expected.columns.name = "trial"
    assert_frame_equal(res, expected)


def test_feature_collection_get_aggregated_features_over_period(
    feature_collection_example,
):
    """Test aggregated feature value retrieval over period."""
    fc = feature_collection_example
    feature_id = "feat_02"

    data = fc.get_aggregated_features_over_period(
        feature_id=feature_id, period="14d", aggregation="mean"
    )
    user_01_values = data.loc["user01"].values
    assert user_01_values[0] == 3.5
    assert np.isnan(user_01_values[1])
    assert_array_equal(data.loc["user02"].values, np.array([6.5, 6.0]))
