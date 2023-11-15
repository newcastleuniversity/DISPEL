"""Tests for :mod:`dispel.providers.bdh.tasks.ft`."""
import json
from copy import deepcopy

import pytest

from dispel.data.core import Reading
from dispel.io.export import export_features
from dispel.providers.bdh import PROVIDER_ID
from dispel.providers.bdh.io import read_bdh
from dispel.providers.bdh.tasks.ft import process_ft
from tests.processing.helper import assert_level_values
from tests.providers import resource_path

PATH_FT = resource_path(PROVIDER_ID, "FINGERTAP/fingertap_uat_3.json")
PATH_DEVIATED_FT = resource_path(
    PROVIDER_ID, "FINGERTAP/deviation_one_side_tapping.json"
)
FEATURE = "mobile_computed_features"
TAPS = "number_of_successful_taps"


@pytest.fixture(scope="session")
def ft_record():
    """Create a bdh finger tapping record fixture."""
    with open(PATH_FT, encoding="utf8") as file:
        raw_record = json.load(file)
    return raw_record


@pytest.fixture(scope="session")
def ft_reading():
    """Create a bdh finger tapping reading fixture."""
    return read_bdh(PATH_FT)


@pytest.fixture(scope="session")
def processed_ft(ft_reading):
    """Create a bdh finger tapping processed reading fixture."""
    return process_ft(deepcopy(ft_reading)).get_reading()


@pytest.fixture(scope="session")
def deviated_record():
    """Create a bdh finger tapping processed reading fixture."""
    return process_ft(deepcopy(read_bdh(PATH_DEVIATED_FT)))


def test_agreement_mobile_dal(processed_ft, ft_record):
    """Test the agreement between mobile computed features and DISPEL features."""
    assert isinstance(processed_ft, Reading)
    right_hand_score = (
        processed_ft.get_level("right").feature_set.get("ft-right-valtap").value
    )
    left_hand_score = (
        processed_ft.get_level("left").feature_set.get("ft-left-valtap").value
    )
    dispel_total_taps = right_hand_score + left_hand_score
    mobile_computed_total_taps = ft_record["body"][FEATURE][TAPS]
    assert mobile_computed_total_taps == dispel_total_taps
    assert mobile_computed_total_taps == 276
    assert mobile_computed_total_taps == processed_ft.feature_set.get("ft-valtap").value


def test_all_taps_timestamp_consistency(processed_ft):
    """Check the consistency between app timestamps and raw data timestamps."""
    level_right = processed_ft.get_level("right")
    level_left = processed_ft.get_level("left")

    left_tap_from_raw = level_left.get_raw_data_set("taps_from_raw").data
    right_tap_from_raw = level_right.get_raw_data_set("taps_from_raw").data

    left_tap_from_mobile = level_left.get_raw_data_set("tap_events").data
    right_tap_from_mobile = level_right.get_raw_data_set("tap_events").data

    # Assert that we have the same number of taps from mobile and from raw
    assert left_tap_from_mobile.shape[0] == pytest.approx(left_tap_from_raw.shape[0], 1)
    assert right_tap_from_mobile.shape[0] == pytest.approx(
        right_tap_from_raw.shape[0], 1
    )


@pytest.mark.parametrize(
    "level, expected",
    [
        (
            "right",
            {
                "ft-right_leftzone-valtap": 66,
                "ft-right_rightzone-valtap": 66,
                "ft-right-valtap": 132,
                "ft-right-tap_inter-mean": 0.10375401069518717,
                "ft-right-tap_inter-std": 0.03635848132053055,
                "ft-right-tap_inter-median": 0.099,
                "ft-right-tap_inter-min": 0.034,
                "ft-right-tap_inter-max": 0.197,
                "ft-right-tap_inter-iqr": 0.037500000000000006,
                "ft-right-valid_tap_inter-mean": 0.13437404580152673,
                "ft-right-valid_tap_inter-std": 0.1121342146288485,
                "ft-right-valid_tap_inter-median": 0.1,
                "ft-right-valid_tap_inter-min": 0.048,
                "ft-right-valid_tap_inter-max": 0.8670000000000001,
                "ft-right-valid_tap_inter-iqr": 0.041999999999999996,
                "ft-right-total_tap": 188,
                "ft-right-double_tap_percentage": 0.0,
            },
        ),
        (
            "left",
            {
                "ft-left_leftzone-valtap": 72,
                "ft-left_rightzone-valtap": 72,
                "ft-left-valtap": 144,
                "ft-left-tap_inter-mean": 0.11518452380952382,
                "ft-left-tap_inter-std": 0.01926482316939174,
                "ft-left-tap_inter-median": 0.115,
                "ft-left-tap_inter-min": 0.063,
                "ft-left-tap_inter-max": 0.17500000000000002,
                "ft-left-tap_inter-iqr": 0.027499999999999997,
                "ft-left-valid_tap_inter-mean": 0.13463636363636364,
                "ft-left-valid_tap_inter-std": 0.08750553668233811,
                "ft-left-valid_tap_inter-median": 0.116,
                "ft-left-valid_tap_inter-min": 0.08,
                "ft-left-valid_tap_inter-max": 0.8500000000000001,
                "ft-left-valid_tap_inter-iqr": 0.0315,
                "ft-left-total_tap": 169,
                "ft-left-double_tap_percentage": 0.0,
            },
        ),
    ],
)
def test_bdh_ft_features(processed_ft, level, expected):
    """Test feature values for Konectom - Finger Tapping Assessment."""
    assert_level_values(processed_ft, level, expected)


def test_bdh_flagging(deviated_record):
    """Test one side tapping and invalid min number of taps deviations."""
    expected = {
        "ft-behavioral-deviation-one_sided_tapping",
        "ft-behavioral-deviation-invalid_min_n_taps",
    }

    df = export_features(deviated_record)
    res = {inv for i in df.flag_ids.unique() for inv in i.split(";")}
    assert expected == res
