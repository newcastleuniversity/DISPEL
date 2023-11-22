"""Test cases for :mod:`dispel.stats.learning`."""
# pylint: disable=duplicate-code

import numpy as np
import pandas as pd
import pytest

from dispel.data.collections import MeasureCollection
from dispel.stats.learning import (
    DelayParameters,
    LearningCurve,
    LearningModel,
    LearningResult,
    compute_delay,
    compute_learning_model,
    extract_learning_for_all_subjects,
    extract_learning_for_one_subject,
)
from tests.conftest import resource_path

EXAMPLE_PATH = resource_path("single-user-learning-example.csv", "stats")

EXAMPLE_PATH_2 = resource_path("multiple-user-learning-example.csv", "stats")

# Fix random seed
np.random.seed(0)


@pytest.fixture
def single_user_learning_example():
    """Create a fixture for single user data for the learning analysis."""
    return MeasureCollection.from_csv(EXAMPLE_PATH)


@pytest.fixture
def multiple_user_learning_example():
    """Create a fixture for multiple user data for the learning analysis."""
    return MeasureCollection.from_csv(EXAMPLE_PATH_2)


@pytest.mark.parametrize("asymptote, slope", [(2.5, 1.4), (2.0, 1.0), (3.0, 2.0)])
def test_fit_learning_model(asymptote, slope):
    """Test :meth:`dispel.stats.learning.LearningCurve.fit`."""
    x = np.linspace(1, 40)
    y_clean = LearningCurve.compute_learning(x, asymptote, slope)
    y_noise = 0.2 * np.random.normal(size=x.size)
    y = y_clean + y_noise
    curve = LearningCurve.fit(x, y)
    assert curve.asymptote == pytest.approx(asymptote, 5e-2)
    assert curve.slope == pytest.approx(slope, 5e-1)


def test_compute_warm_up():
    """Test :meth:`dispel.stats.learning.LearningCurve.get_warm_up`."""
    data = np.arange(0, 6)
    curve = LearningCurve(asymptote=4, slope=1)
    warm_up = curve.get_warm_up(data)
    assert warm_up == 4


def test_compute_delay(single_user_learning_example):
    """Test the good computation of delay-related measures."""
    delay = compute_delay(single_user_learning_example.data.start_date)
    assert delay.mean == pytest.approx(1.0069029706790125)
    assert delay.median == pytest.approx(0.9955124652777778)
    assert delay.max == pytest.approx(1.3589436574074074)


def get_single_measure_values(n, asymptote, slope):
    """Create a data frame containing measure values."""
    curve = LearningCurve(asymptote, slope)

    time_idx = pd.date_range("now", periods=n, freq="1d")
    index = pd.MultiIndex.from_tuples(
        zip(time_idx, range(1, n + 1)), names=["start_date", "trial"]
    )
    y = curve(np.arange(1, n + 1))

    return pd.Series(y, index=index)


def test_compute_learning_model():
    """Test computation of learning and delay parameters."""
    data = get_single_measure_values(10, 20, 2)
    model, delay = compute_learning_model(data)

    assert model.curve.asymptote == pytest.approx(20)
    assert model.curve.slope == pytest.approx(2)
    assert delay.mean == 1


def test_compute_learning_model_insufficient_data():
    """Test that learning also works for single measure values."""
    data = get_single_measure_values(1, 1, 1)
    model, delay = compute_learning_model(data)

    assert isinstance(model, LearningModel)
    assert isinstance(delay, DelayParameters)


def test_extract_learning_for_one_subject(single_user_learning_example):
    """Test :func:`dispel.stats.learning.extract_learning_for_one_subject`."""
    measure_id = "CPS-dtd-rt-mean-01"
    subject_id = "4f96b3927ec780c373277094a01bb664a6bb67ca71b77296474a10f3a7ad36df"
    fc = single_user_learning_example
    learning_result = extract_learning_for_one_subject(
        fc, measure_id=measure_id, subject_id=subject_id, reset_trials=False
    )

    assert isinstance(learning_result, LearningResult)
    params = learning_result.get_parameters()
    assert params.nb_outliers.item() == 3
    assert params.r2_score.item() == pytest.approx(0.18820266198260116)
    assert params.learning_rate.item() == pytest.approx(-0.07488577904149303)
    assert params.optimal_performance.item() == pytest.approx(0.53149624120954)
    assert params.slope_coefficient.item() == pytest.approx(-0.039801510080601)
    assert params.warm_up.item() == 5

    new_data = learning_result.get_new_data(subject_id)
    assert list(new_data.index) == list(set(range(1, 23)) - {2, 4, 17})


def test_extract_learning_for_all_subjects(multiple_user_learning_example):
    """Test :func:`dispel.stats.learning.extract_learning_for_all_subjects`."""
    fc = multiple_user_learning_example
    learning_result = extract_learning_for_all_subjects(fc, "measure_id_1")
    assert isinstance(learning_result, LearningResult)
    expected_columns = {
        "subject_id",
        "measure_id",
        "optimal_performance",
        "slope_coefficient",
        "learning_rate",
        "warm_up",
        "r2_score",
        "nb_outliers",
        "delay_mean",
        "delay_median",
        "delay_max",
    }
    learning_parameters = learning_result.get_parameters()
    assert set(learning_parameters.columns) == expected_columns
    assert len(learning_parameters) == 2
    assert set(learning_parameters["subject_id"]) == {"user_id_1", "user_id_2"}

    user_id_1 = learning_parameters.loc[
        learning_parameters["subject_id"] == "user_id_1"
    ]
    assert user_id_1.optimal_performance.iloc[0] == pytest.approx(0.57088461)

    user_id_2 = learning_parameters.loc[
        learning_parameters["subject_id"] == "user_id_2"
    ]
    assert user_id_2.optimal_performance.iloc[0] == pytest.approx(0.404033444)

    new_data = learning_result.get_new_data("user_id_1")
    assert isinstance(new_data, pd.Series)
    assert list(new_data) == pytest.approx([0.52673076923076, 0.5488076923076])
