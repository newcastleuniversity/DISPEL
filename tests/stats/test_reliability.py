"""Test cases for :mod:`dispel.stats.reliability`."""

import pandas as pd
import pytest

from dispel.data.collections import FeatureCollection
from dispel.stats.reliability import (
    icc_parallel_form,
    icc_power,
    icc_sample_size,
    icc_set_test_retest,
    icc_test_retest,
)
from tests.conftest import resource_path

EXAMPLE_PATH = resource_path("reliability-example.csv", "stats")


@pytest.fixture
def data():
    """Get a fixture to the example data set."""
    return FeatureCollection.from_csv(EXAMPLE_PATH)


@pytest.fixture
def dataframe():
    """Fixture of dataframe example."""
    value = [
        [9, 2, 5, 8],
        [6, 1, 3, 2],
        [8, 4, 6, 8],
        [7, 1, 2, 6],
        [10, 5, 6, 9],
        [6, 2, 4, 7],
    ]
    return pd.DataFrame(value)


def test_from_test_retest(dataframe):
    """Test of the icc_test_retest."""
    icc = icc_test_retest(dataframe)

    assert icc.sample_size == 6
    assert icc.sessions == 4

    assert icc.value == pytest.approx(0.6200505475989893)
    assert icc.l_bound == pytest.approx(0.1520370538559581)
    assert icc.u_bound == pytest.approx(0.8994767001136251)
    assert icc.p_value == pytest.approx(0.0001345665164843579)


def test_icc_sample_size():
    """Test of the icc_sample_size."""
    result = icc_sample_size(icc=0.65, p0_icc=0, n_ratings=2)
    assert result == 15


def test_icc_power():
    """Test of the icc_power."""
    result = icc_power(icc=0.8, p0_icc=0, n_ratings=2, n_subjects=15)
    assert result == 0.98


def test_icc_features_set_test_retest(data):
    """Test of the function icc_features_set."""
    feature_iccs = icc_set_test_retest(data, session_min=2)
    assert len(feature_iccs.iccs.keys()) == 2
    assert feature_iccs.iccs["feat_02"].value == pytest.approx(0.979591836)
    assert feature_iccs.iccs["feat_02"].sample_size == 2
    assert feature_iccs.iccs["feat_02"].power == pytest.approx(0.63)


def test_icc_parallel_form():
    """Test the parallel form reliability score."""
    form1 = [7, 6, 5, 4, 5, 6, 5, 7]
    user = ["userA", "userB", "userC", "userD", "userE", "userF", "userG", "userH"]
    form2 = [6, 5, 5, 4, 5, 6, 6]
    df_form1 = pd.DataFrame(form1, index=user)
    df_form2 = pd.DataFrame(form2, index=user[0:7])
    result = icc_parallel_form(df_form1, df_form2)
    assert result.value == pytest.approx(0.8301886792452833)
