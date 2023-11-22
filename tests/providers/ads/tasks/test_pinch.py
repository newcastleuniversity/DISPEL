"""Test cases for :mod:`dispel.providers.ads.tasks.pinch`."""

from copy import deepcopy

import pandas as pd
import pytest

from dispel.data.levels import Level
from dispel.data.raw import (
    RawDataSet,
    RawDataSetDefinition,
    RawDataSetSource,
    RawDataValueDefinition,
)
from dispel.processing import process
from dispel.processing.level import ProcessingStepGroup
from dispel.processing.modalities import HandModality, SensorModality
from dispel.providers.generic.sensor import Resample, SetTimestampIndex
from dispel.providers.generic.tasks.pinch.attempts import (
    PinchTarget,
    dwell_time,
    number_successful_pinches,
    pinching_duration,
    reaction_time,
    success_duration,
    total_duration,
    total_number_pinches,
)
from dispel.providers.generic.tasks.pinch.modalities import (
    AttemptSelectionModality,
    BubbleSizeModality,
    FingerModality,
)
from dispel.providers.generic.tremor import TremorMeasures
from tests.processing.helper import (
    assert_dict_values,
    assert_level_values,
    assert_unique_measure_ids,
    read_results,
)
from tests.providers.ads.conftest import (
    RESULTS_PATH_PINCH2,
    RESULTS_PATH_PINCH_NEW_FORMAT,
)


def test_pinch_target(example_reading_pinch2):
    """Test :class:`dispel.providers.generic.tasks.pinch.attempts.PinchTarget`."""
    level_id = "right-extra_large"
    level = example_reading_pinch2.get_level(level_id)
    target = PinchTarget.from_level(level)

    assert target.radius == pytest.approx(102, 1.0e-3)
    assert target.coordinates == pytest.approx((187.5, 427.97702), 1.0e-3)
    assert target.hand == HandModality.RIGHT
    assert target.size == BubbleSizeModality.EXTRA_LARGE
    assert target.appearance == pd.Timestamp("2020-07-15 12:39:17.754000")
    assert target.first_pushes(FingerModality.TOP_FINGER) == pytest.approx(
        [1.96667, 0.63333], 1.0e-5
    )
    assert target.first_pushes(FingerModality.BOTTOM_FINGER) == pytest.approx(
        [0.36667, 1.71667], 1.0e-5
    )

    attempt = target.attempts[0]
    assert attempt.begin == pd.Timestamp("2020-07-15 12:39:18.802000")
    assert attempt.end == pd.Timestamp("2020-07-15 12:39:19.027000")
    assert attempt.pinch_begin == pd.Timestamp("2020-07-15 12:39:18.879000")
    assert attempt.first_push_top_fingers == pytest.approx(1.96667, 1.0e-5)
    assert attempt.first_push_bottom_fingers == pytest.approx(0.36667, 1.0e-5)

    expected_dta = pd.Timestamp("2020-07-15 12:39:18.806000") - pd.Timestamp(
        "2020-07-15 12:39:18.802000"
    )
    assert attempt.double_touch_asynchrony == expected_dta

    top_finger_coords = attempt.top_finger.first_position
    assert top_finger_coords == (pytest.approx((244.99998, 292.66663), 1.0e-3))

    tf_contact_distance = target.contact_distances(FingerModality.TOP_FINGER)
    bf_contact_distance = target.contact_distances(FingerModality.BOTTOM_FINGER)
    assert tf_contact_distance[0] == pytest.approx(45.016, 1.0e-2)
    assert bf_contact_distance[0] == pytest.approx(43.705, 1.0e-2)


def test_total_number_pinches(example_pinch_target):
    """Testing :func:`dispel.providers.generic.tasks.pinch.attempts.total_number_pinches`."""
    assert total_number_pinches(example_pinch_target) == 2


def test_number_successful_pinches(example_pinch_target):
    """Testing :func:`dispel.providers.generic.tasks.pinch.attempts.number_successful_pinches`."""
    assert number_successful_pinches(example_pinch_target) == 1


def test_success_duration(example_pinch_target):
    """Testing :func:`dispel.providers.generic.tasks.pinch.attempts.success_duration`."""
    success_durations = success_duration(example_pinch_target)
    assert success_durations[0] == pytest.approx(0.077, 1.0e-5)
    assert success_durations[1] == pytest.approx(0.096, 1.0e-5)


def test_pinching_duration(example_pinch_target):
    """Testing :func:`dispel.providers.generic.tasks.pinch.attempts.pinching_duration`."""
    pinching_durations = pinching_duration(example_pinch_target)
    assert pinching_durations[0] == pytest.approx(0.135, 1.0e-5)
    assert pinching_durations[1] == pytest.approx(0.112, 1.0e-5)


def test_total_duration(example_pinch_target):
    """Testing :func:`dispel.providers.generic.tasks.pinch.attempts.total_duration`."""
    total_duration_ = total_duration(example_pinch_target)
    expected = (
        pd.Timestamp("2020-07-15 12:39:19.632000")
        - pd.Timestamp("2020-07-15 12:39:18.802000")
    ).total_seconds()
    assert total_duration_ == pytest.approx(expected, 1.0e-5)


def test_reaction_time(example_pinch_target):
    """Testing :func:`dispel.providers.generic.tasks.pinch.attempts.reaction_time`."""
    assert reaction_time(example_pinch_target) == pytest.approx(1048.0, 1.0e-3)


def test_dwell_time(example_pinch_target):
    """Testing :func:`dispel.providers.generic.tasks.pinch.attempts.dwell_time`."""
    assert dwell_time(
        example_pinch_target, AttemptSelectionModality.ALL
    ) == pytest.approx([4.0, 11.0])


def test_first_dwell_time(example_pinch_target):
    """Testing :func:`dispel.providers.generic.tasks.pinch.attempts.dwell_time`."""
    assert dwell_time(
        example_pinch_target, AttemptSelectionModality.FIRST
    ) == pytest.approx(4.0)


@pytest.mark.parametrize(
    "level, flags",
    [
        ("right-extra_large", []),
        ("right-medium", []),
        ("right-small", []),
        ("right-large", []),
        ("left-medium", []),
        ("left-extra_large", []),
        ("left-small", []),
        ("left-large", []),
        ("melted_levels", []),
        ("small", []),
        ("medium", []),
        ("large", []),
        ("extra_large", []),
        ("left", []),
        ("right", []),
    ],
)
def test_pinch_phone_orientation_transformation(
    example_reading_processed_pinch2, level, flags
):
    """Test phone orientation transformation for Pinch task."""
    invs = example_reading_processed_pinch2.get_level(level).get_flags()

    assert invs == flags


@pytest.mark.parametrize("level_id,expected", read_results(RESULTS_PATH_PINCH2))
def test_pinch_process(example_reading_processed_pinch2, level_id, expected):
    """Unit test to ensure the pinch measures are well computed."""
    assert_level_values(example_reading_processed_pinch2, level_id, expected)


@pytest.mark.parametrize(
    "level_id,expected", read_results(RESULTS_PATH_PINCH_NEW_FORMAT)
)
def test_pinch_process_new_format(
    example_reading_processed_pinch_new_format, level_id, expected
):
    """Unit test to ensure the pinch new format measures are well computed."""
    assert_level_values(example_reading_processed_pinch_new_format, level_id, expected)


def test_pinch_process_ads_unique_measure_ids(example_reading_processed_pinch2):
    """Test that measure ids are unique."""
    assert_unique_measure_ids(example_reading_processed_pinch2)


@pytest.mark.parametrize(
    "sampling_frequency,expected_values",
    [
        (
            20.0,
            {
                "pinch-left-acc_sig_ene": 0.9120702743530273,
                "pinch-left-acc_ps_ene_x": 0.008427435924481277,
                "pinch-left-acc_ps_ene_y": 0.0035950599324322584,
                "pinch-left-acc_ps_ene_z": 0.065499786692469,
                "pinch-left-acc_ps_ene_xyz": 0.007302683940076308,
                "pinch-left-acc_ps_peak_x": 4.3478260869565215,
                "pinch-left-acc_ps_peak_y": 4.3478260869565215,
                "pinch-left-acc_ps_peak_z": 5.217391304347825,
                "pinch-left-acc_ps_peak_xyz": 1.7391304347826084,
                "pinch-left-acc_ps_ent_x": 2.9046479364851905,
                "pinch-left-acc_ps_ent_y": 3.0672721030596506,
                "pinch-left-acc_ps_ent_z": 2.8777887437777494,
                "pinch-left-acc_ps_ent_xyz": 3.183497912517169,
                "pinch-left-acc_ps_amp_x": 0.0021483702585101128,
                "pinch-left-acc_ps_amp_y": 0.000891252770088613,
                "pinch-left-acc_ps_amp_z": 0.023922281339764595,
                "pinch-left-acc_ps_amp_xyz": 0.011607293970882893,
            },
        )
    ],
)
def test_compute_tremor_measures(
    example_reading_pinch2, sampling_frequency, expected_values
):
    """Test :class:`dispel.processing.tremor_lib.TremorMeasures`."""
    reading = deepcopy(example_reading_pinch2)
    sensor = SensorModality.ACCELEROMETER
    columns = list("xyz")
    hand = "left"

    class _PinchTremorMeasures(ProcessingStepGroup):
        def __init__(self, hand_: str):
            steps = [
                SetTimestampIndex(str(sensor), columns, "ts", duplicates="last"),
                Resample(
                    f"{str(sensor)}_ts",
                    aggregations=["mean", "ffill"],
                    columns=columns,
                    freq=sampling_frequency,
                ),
                TremorMeasures(
                    sensor=sensor,
                    data_set_id=f"{str(sensor)}_ts_resampled",
                    columns=columns,
                ),
            ]
            super().__init__(
                steps, task_name="pinch", modalities=[hand_], level_filter=hand_
            )

    step = _PinchTremorMeasures(hand)
    data = reading.get_level("left-medium").get_raw_data_set(str(sensor)).data.copy()
    data.rename(
        columns={
            "userAccelerationX": "x",
            "userAccelerationY": "y",
            "userAccelerationZ": "z",
        },
        inplace=True,
    )
    reading.set(
        Level(
            id_=hand,
            start=2,
            end=3,
            raw_data_sets=[
                RawDataSet(
                    RawDataSetDefinition(
                        str(sensor),
                        RawDataSetSource("manufacturer"),
                        [
                            RawDataValueDefinition("x", "x"),
                            RawDataValueDefinition("y", "y"),
                            RawDataValueDefinition("z", "z"),
                            RawDataValueDefinition("ts", "ts"),
                        ],
                    ),
                    data[[*columns, "ts"]],
                )
            ],
        )
    )
    process(reading, step)
    measure_set = reading.get_level(hand).measure_set
    assert_dict_values(measure_set, expected_values)
