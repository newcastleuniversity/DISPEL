"""Tests for :mod:`dispel.providers.generic.tasks.ps`."""

from dispel.data.core import Reading
from tests.processing.helper import assert_level_values


def test_process_ps(example_reading_processed_ps):
    """Test end-to-end processing of Pronation Supination Assessment."""
    assert isinstance(example_reading_processed_ps, Reading)
    right_hand = example_reading_processed_ps.get_level("right")
    left_hand = example_reading_processed_ps.get_level("left")
    assert len(right_hand.get_raw_data_set("ps_event").data) == 25
    assert len(left_hand.get_raw_data_set("ps_event").data) == 27
    expected_right = {
        "ps-right-speed_simple_dec": -11.6735127767,
        "ps-ts_mvmt_power-mean": 219.736355748,
        "ps-right-amp-mean": 413.5901673128,
        "ps-right-amp-median": 412.3253254305,
        "ps-right-amp-npcv": 0.0210450659,
        "ps-right-dur-mean": 0.5325454545,
        "ps-right_pro-abs_rotation-mean": 207.1005583148,
        "ps-right_pro-abs_rotation-median": 207.7908773022,
        "ps-right_pro-abs_rotation-npcv": 0.0267162431,
        "ps-right_pro-abs_rotation_speed-mean": 773.7176653152,
        "ps-right_pro-abs_rotation_speed-median": 782.3255061678,
        "ps-right_pro-abs_rotation_speed-npcv": 0.0423342627,
        "ps-right_sup-abs_rotation-mean": 206.4869149562,
        "ps-right_sup-abs_rotation-median": 207.3608103789,
        "ps-right_sup-abs_rotation-npcv": 0.0235274649,
        "ps-right_sup-abs_rotation_speed-mean": 781.4904704519,
        "ps-right_sup-abs_rotation_speed-median": 791.4534747285,
        "ps-right_sup-abs_rotation_speed-npcv": 0.0762245823,
        "ps-right_prosup-abs_rotation-mean": 206.7814637683,
        "ps-right_prosup-abs_rotation-median": 207.5651422373,
        "ps-right_prosup-abs_rotation-npcv": 0.0252503664,
        "ps-right_prosup-abs_rotation_speed-mean": 777.7595239863,
        "ps-right_prosup-abs_rotation_speed-median": 791.4534747285,
        "ps-right_prosup-abs_rotation_speed-npcv": 0.0602336488,
        "ps-right-ec": 25,
        "ps-right-amp_simple_dec": -19.0030463227,
    }
    assert_level_values(
        example_reading_processed_ps, "right", expected_right, relative_error=1e-2
    )


def test_process_static_ps(example_reading_processed_ps_static):
    """Test end-to-end processing of Pronation Supination Assessment."""
    expected_right = {"ps-right-ec": 0}
    expected_left = {"ps-left-ec": 0}
    assert_level_values(example_reading_processed_ps_static, "right", expected_right)
    assert_level_values(example_reading_processed_ps_static, "left", expected_left)


def test_process_slow_shaky_ps(example_reading_processed_ps_slow_shaky):
    """Test end-to-end processing of Pronation Supination Assessment."""
    expected_left = {
        "ps-ts_mvmt_power-mean": 1.5562860898,
        "ps-left_pro-abs_rotation-mean": 197.2885541433,
        "ps-left_pro-abs_rotation-median": 197.2885541433,
        "ps-left_pro-abs_rotation-npcv": 0.0,
        "ps-left_pro-abs_rotation_speed-mean": 123.6918834754,
        "ps-left_pro-abs_rotation_speed-median": 123.6918834754,
        "ps-left_pro-abs_rotation_speed-npcv": 0.0,
        "ps-left_sup-abs_rotation-mean": 214.4202674542,
        "ps-left_sup-abs_rotation-median": 214.4202674542,
        "ps-left_sup-abs_rotation-npcv": 0.0382332097,
        "ps-left_sup-abs_rotation_speed-mean": 139.6801331916,
        "ps-left_sup-abs_rotation_speed-median": 139.6801331916,
        "ps-left_sup-abs_rotation_speed-npcv": 0.0167972276,
        "ps-left_prosup-abs_rotation-mean": 208.7096963505,
        "ps-left_prosup-abs_rotation-median": 206.2222924085,
        "ps-left_prosup-abs_rotation-npcv": 0.0444270883,
        "ps-left_prosup-abs_rotation_speed-mean": 134.3507166195,
        "ps-left_prosup-abs_rotation_speed-median": 137.333894201,
        "ps-left_prosup-abs_rotation_speed-npcv": 0.0528905907,
        "ps-left-ec": 3,
    }
    assert_level_values(
        example_reading_processed_ps_slow_shaky,
        "left",
        expected_left,
        relative_error=1e-2,
    )
