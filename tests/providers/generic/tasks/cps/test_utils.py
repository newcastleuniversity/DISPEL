"""Tests for :mod:`dispel.providers.generic.tasks.cps.time`."""
import pandas as pd
import pytest
from pytest import approx as apx

from dispel.providers.generic.tasks.cps.steps import (
    compute_confusion_error_rate,
    compute_confusion_matrix,
    compute_response_time_linear_regression,
    compute_streak,
    study2and3back,
    transform_user_input,
)
from dispel.providers.generic.tasks.cps.utils import (
    correct_data_selection,
    reaction_time,
)


def test_correct_data_selection(cps_example):
    """Tests :func:`dispel.providers.generic.tasks.cps.correct_data_selection`."""
    data = correct_data_selection(cps_example, 0, 1)
    assert len(data) == 1
    assert data.expected[0] == 1 and data.actual[0] == 1


def test_compute_confusion_matrix(cps_example):
    """Ensure good computation of digit_to_symbol confusion matrix."""
    confusion_matrix = compute_confusion_matrix(cps_example)
    assert confusion_matrix[3][2] == 1.0
    assert confusion_matrix[1][1] == 1.0


def test_compute_confusion_error_rate(cps_example):
    """Test :func:`dispel.providers.generic.tasks.cps.compute_confusion_error_rate`."""
    confusion_matrix = compute_confusion_matrix(cps_example)
    err_rate = compute_confusion_error_rate(confusion_matrix, 3, 2)
    assert err_rate == 0.5


def test_compute_streak(success_data):
    """Test :func:`dispel.providers.generic.tasks.cps.compute_streak`."""
    max_good, max_wrong = compute_streak(success_data)
    assert max_good == 3
    assert max_wrong == 2


def test_reaction_time():
    """Ensure the good processing of specific reaction time."""
    data = pd.DataFrame(
        [
            [
                pd.Timestamp("2020-05-15 16:12:38.011"),
                pd.Timestamp("2020-05-15 16:12:39.071"),
            ]
        ],
        columns=["tsDisplay", "tsAnswer"],
    )
    reac_time = reaction_time(data)
    assert reac_time[0] == 1.06


def test_transform_data_std(level_std):
    """Ensure the good processing of symbol pairs analysis data frame."""
    data = pd.DataFrame(
        [
            [
                "symbol1",
                "symbol1",
                pd.Timestamp("2020-05-15 16:12:38.011"),
                pd.Timestamp("2020-05-15 16:12:39.071"),
            ]
        ],
        columns=["displayedSymbol", "userSymbol", "tsDisplay", "tsAnswer"],
    )
    new_data = transform_user_input(data, level_std)
    assert new_data["reactionTime"][0] == 1.06
    assert "actual" in new_data.columns
    assert "expected" in new_data.columns
    assert "reactionTime" in new_data.columns
    assert new_data["expected"][0] == 1


def test_transform_data_dtd(level_dtd):
    """Ensure the good processing of digit pairs analysis data frame."""
    data = pd.DataFrame(
        [
            [
                1,
                1,
                pd.Timestamp("2020-05-15 16:12:38.011"),
                pd.Timestamp("2020-05-15 16:12:39.071"),
            ]
        ],
        columns=["displayedValue", "userValue", "tsDisplay", "tsAnswer"],
    )
    new_data = transform_user_input(data, level_dtd)
    assert new_data["reactionTime"][0] == 1.06
    assert "actual" in new_data.columns
    assert "expected" in new_data.columns
    assert "reactionTime" in new_data.columns
    assert new_data["expected"][0] == 1


def test_study_2_and_3_back(data_n_backs):
    """Unit test to ensure the good computation of the n-back data frame."""
    backs = study2and3back(data_n_backs)
    assert backs["rtCurrent1"].count() == 2
    assert backs["rtCurrent1"].mean() == 1.56
    assert backs["rtCurrent2"].count() == 2
    assert backs["rtCurrent2"].mean() == 1.56
    assert backs["rtCurrent3"].count() == 1
    assert backs["rtCurrent3"].mean() == 1.06
    assert len(backs) == 2


@pytest.fixture
def linear_reg_data():
    """Create a fixture of an example of CPS ``keys-analysis`` set."""
    frame = pd.DataFrame(
        [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 5], [5, 5, 6]],
        columns=["expected", "actual", "reactionTime"],
    )
    return frame


def test_compute_linear_regression(linear_reg_data):
    """Test the computation of linear regression and output data format."""
    s_1, r_2_1 = compute_response_time_linear_regression(linear_reg_data, 0)
    s_2, r_2_2 = compute_response_time_linear_regression(linear_reg_data, 1)
    s_3, r_2_3 = compute_response_time_linear_regression(linear_reg_data, 2)
    s_4, r_2_4 = compute_response_time_linear_regression(linear_reg_data, 3)
    assert s_1 == apx(1.3, 1e-6) and r_2_1 == apx(0.982558, 1e-6)
    assert s_2 == apx(1.4, 1e-6) and r_2_2 == apx(0.980000, 1e-6)
    assert s_3 == apx(1.5, 1e-6) and r_2_3 == apx(0.964286, 1e-6)
    assert s_4 == apx(1.0, 1e-6) and r_2_4 == apx(1.000000, 1e-6)


@pytest.fixture
def linear_reg_nan():
    """Create a fixture of an example of CPS ``keys-analysis`` set."""
    frame = pd.DataFrame(
        [[1, 1, 1], [2, 2, 2]], columns=["expected", "actual", "reactionTime"]
    )
    return frame


def test_compute_linear_regression_nan(linear_reg_nan):
    """Test the computation of linear regression and output data format."""
    s_1, r_2_1 = compute_response_time_linear_regression(linear_reg_nan, 0)
    s_2, r_2_2 = compute_response_time_linear_regression(linear_reg_nan, 1)
    s_3, r_2_3 = compute_response_time_linear_regression(linear_reg_nan, 2)
    s_4, r_2_4 = compute_response_time_linear_regression(linear_reg_nan, 3)
    assert s_1 == apx(1.0, 1e-6) and r_2_1 == apx(1.000000, 1e-6)
    assert s_2 == apx(0.0, 1e-6) and pd.isnull(r_2_2)
    assert pd.isnull(s_3) and pd.isnull(r_2_3)
    assert pd.isnull(s_4) and pd.isnull(r_2_4)
