"""Pytest configuration for AdS provider."""
from copy import deepcopy

import pytest

from dispel.data.levels import Level
from dispel.io.utils import load_json
from dispel.processing import process
from dispel.providers.ads import PROVIDER_ID
from dispel.providers.ads.io import parse_ads_raw_json, read_ads
from dispel.providers.ads.tasks.sbt_utt import process_sbt_utt
from dispel.providers.generic.tasks.cps import process_cps
from dispel.providers.generic.tasks.draw import process_draw
from dispel.providers.generic.tasks.draw.shapes import (
    get_reference_path,
    get_user_path,
    get_valid_up_sampled_path,
)
from dispel.providers.generic.tasks.draw.steps import FlagCompleteDrawing
from dispel.providers.generic.tasks.pinch import process_pinch
from dispel.providers.generic.tasks.pinch.attempts import PinchTarget
from dispel.signal.dtw import get_dtw_distance
from tests.processing.helper import read_results
from tests.providers import resource_path

EXAMPLE_PATH_MOOD = resource_path(PROVIDER_ID, "MOOD/example.json")
EXAMPLE_PATH_MOOD_NEW_FORMAT = resource_path(PROVIDER_ID, "MOOD/new-format.json")
EXAMPLE_PATH_MOOD_MULTIPLE_ANSWERS = resource_path(
    PROVIDER_ID, "MOOD/multiple_answers.json"
)
EXAMPLE_PATH_MOOD_MULTIPLE_ANSWER_SAME_Q_SYNTH = resource_path(
    PROVIDER_ID, "MOOD/multiple_answers_same_question_synthetic.json"
)
EXAMPLE_PATH_MOOD_MULTIPLE_ANSWER_SAME_Q_REAL = resource_path(
    PROVIDER_ID, "MOOD/multiple_answers_same_question_real.json"
)


@pytest.fixture(scope="module")
def example_reading_mood():
    """Fixture for a mood example."""
    return read_ads(EXAMPLE_PATH_MOOD)


@pytest.fixture(scope="module")
def example_mood_new_format_reading_example():
    """Fixture for a mood example with the new format."""
    # FIXME: add it under regression as it is not used
    return read_ads(EXAMPLE_PATH_MOOD_NEW_FORMAT)


@pytest.fixture(scope="module")
def example_reading_mood_multiple_answers():
    """Fixture for a mood example with multiple answers."""
    return read_ads(EXAMPLE_PATH_MOOD_MULTIPLE_ANSWERS)


@pytest.fixture(scope="module")
def example_reading_mood_multiple_answers_same_q_synth():
    """Fixture for a mood example with multiple answers and same Q-synth."""
    return read_ads(EXAMPLE_PATH_MOOD_MULTIPLE_ANSWER_SAME_Q_SYNTH)


@pytest.fixture(scope="module")
def example_reading_mood_multiple_answers_same_q_real():
    """Fixture for a mood example with multiple answers and same Q-real."""
    return read_ads(EXAMPLE_PATH_MOOD_MULTIPLE_ANSWER_SAME_Q_REAL)


EXAMPLE_PATH_MSIS29 = resource_path(PROVIDER_ID, "MSIS29/example.json")
EXAMPLE_PATH_MSIS29_MULTIPLE_ANSWERS = resource_path(
    PROVIDER_ID, "MSIS29/multiple_answers.json"
)


@pytest.fixture
def example_reading_msis29():
    """Fixture for a MSIS-29 example."""
    return read_ads(EXAMPLE_PATH_MSIS29)


@pytest.fixture
def example_reading_msis29_multiple_answers():
    """Fixture for a MSIS-29 example with multiple answers."""
    return read_ads(EXAMPLE_PATH_MSIS29_MULTIPLE_ANSWERS)


EXAMPLE_PATH_DRAW = resource_path(PROVIDER_ID, "DRAW/example.json")
EXAMPLE_PATH_DRAW_NEW_FORMAT = resource_path(PROVIDER_ID, "DRAW/new-format.json")
EXAMPLE_PATH_DRAW_NF_NO_END_ZONE = resource_path(
    PROVIDER_ID, "DRAW/new-format_no_end_zone.json"
)
EXAMPLE_PATH_DRAW_ORIENTATION = resource_path(
    PROVIDER_ID, "DRAW/drawing_orientation_invalid.json"
)
EXAMPLE_PATH_DRAW_DISTANCE_RATIO = resource_path(PROVIDER_ID, "DRAW/invalid-draws.json")
EXAMPLE_PATH_DRAW_NOT_CONTINUOUS = resource_path(
    PROVIDER_ID, "DRAW/example_not_continuous.json"
)


@pytest.fixture(scope="module")
def example_reading_draw():
    """Fixture for Drawing example."""
    return read_ads(EXAMPLE_PATH_DRAW)


@pytest.fixture
def example_level_draw_sccr(example_reading_draw):
    """Fixture for square counter-clock right hand level example for Drawing."""
    return example_reading_draw.get_level("square_counter_clock-right")


@pytest.fixture(scope="module")
def example_reading_draw_orientation():
    """Fixture for Drawing example."""
    return read_ads(EXAMPLE_PATH_DRAW_ORIENTATION)


@pytest.fixture(scope="module")
def example_reading_draw_new_format():
    """Fixture for Drawing example using the new format."""
    return read_ads(EXAMPLE_PATH_DRAW_NEW_FORMAT)


@pytest.fixture(scope="module")
def example_reading_draw_nf_no_end_zone():
    """Fixture for Drawing example."""
    return read_ads(EXAMPLE_PATH_DRAW_NF_NO_END_ZONE)


@pytest.fixture(scope="module")
def example_reading_draw_distance_ratio():
    """Fixture for Drawing example."""
    return read_ads(EXAMPLE_PATH_DRAW_DISTANCE_RATIO)


@pytest.fixture
def height_screen(example_reading_draw):
    """Get the height in `pts` of the user's smartphone."""
    # TODO: fix naming or actual usage of the fixture
    return example_reading_draw.device.screen.height_dp_pt


@pytest.fixture
def example_data_draw_sccr_screen(example_level_draw_sccr):
    """Fixture for data for screen from square counter-clock right-handed Drawing."""
    return example_level_draw_sccr.get_raw_data_set("screen").data


@pytest.fixture
def example_data_draw_sccr_screen_path(
    example_level_draw_sccr, example_data_draw_sccr_screen, height_screen
):
    """Fixture for path data for square counter-clock right-handed Drawing."""
    reference = get_reference_path(example_level_draw_sccr, height_screen)
    return get_user_path(example_data_draw_sccr_screen, reference, height_screen)


@pytest.fixture
def example_data_draw_sccr_screen_path_up_sampled(
    example_level_draw_sccr, example_data_draw_sccr_screen, height_screen
):
    """Up-sampled example path data for square counter-clock right-handed Drawing."""
    reference = get_reference_path(example_level_draw_sccr, height_screen)
    path = get_user_path(example_data_draw_sccr_screen, reference, height_screen)
    return get_valid_up_sampled_path(path)


@pytest.fixture
def paths_drawing_sccr_data(example_data_draw_sccr_screen, height_screen):
    """Fixture for data for paths."""
    # TODO fix naming
    level = Level(id_="square_counter_clock-right", start=0, end=1)
    ref = get_reference_path(level, height_screen)
    return get_user_path(example_data_draw_sccr_screen, ref, height_screen)


@pytest.fixture
def dtw_raw_data(example_data_draw_sccr_screen_path_up_sampled, height_screen):
    """Fixture for DTW raw data."""
    # TODO: fix naming
    level = Level(id_="square_counter_clock-right", start=0, end=1)
    ref = get_reference_path(level, height_screen)
    dtw_data = get_dtw_distance(example_data_draw_sccr_screen_path_up_sampled, ref)
    dtw_data.name = "dtw_data"
    return dtw_data.to_frame()


@pytest.fixture(scope="module")
def example_reading_processed_draw(example_reading_draw):
    """Fixture for processed Drawing reading."""
    return process_draw(deepcopy(example_reading_draw)).get_reading()


@pytest.fixture(scope="module")
def example_reading_processed_draw_orientation(example_reading_draw_orientation):
    """Fixture for processed Drawing reading with different orientation."""
    return process_draw(deepcopy(example_reading_draw_orientation)).get_reading()


@pytest.fixture(scope="module")
def example_reading_processed_draw_new_format(example_reading_draw_new_format):
    """Fixture for processed Drawing reading in new format."""
    return process_draw(deepcopy(example_reading_draw_new_format)).get_reading()


@pytest.fixture(scope="module")
def example_reading_processed_draw_nf_no_end_zone(
    example_reading_draw_nf_no_end_zone,
):
    """Fixture for a new format drawing without end zone."""
    # FIXME: consider inlining in test case as fixture is not general purpose.
    return process(
        example_reading_draw_nf_no_end_zone, [FlagCompleteDrawing()]
    ).get_reading()


@pytest.fixture(scope="module")
def example_reading_processed_draw_distance_ratio(example_reading_draw_distance_ratio):
    """Fixture for a processed drawing reading for flag."""
    return process_draw(deepcopy(example_reading_draw_distance_ratio)).get_reading()


@pytest.fixture(scope="module")
def example_reading_processed_draw_not_continuous():
    """Fixture for a processed drawing reading from ADS data."""
    return process_draw(read_ads(EXAMPLE_PATH_DRAW_NOT_CONTINUOUS)).get_reading()


@pytest.fixture
def example_data_draw_sccl_shape(example_reading_processed_draw):
    """Fixture for Drawing example for square_counter_clock shape."""
    return (
        example_reading_processed_draw.get_level("square_counter_clock-left")
        .get_raw_data_set("shape")
        .data["shape"][0]
    )


@pytest.fixture
def example_data_draw_scl_shape(example_reading_processed_draw):
    """Create a DrawShape example for square_clock shape."""
    return (
        example_reading_processed_draw.get_level("square_clock-left")
        .get_raw_data_set("shape")
        .data["shape"][0]
    )


EXAMPLE_PATH_GRIP = resource_path(PROVIDER_ID, "GRIP/example.json")
EXAMPLE_PATH_GRIP_NEW_FORMAT = resource_path(PROVIDER_ID, "GRIP/new-format.json")


@pytest.fixture(scope="module")
def example_reading_grip():
    """Fixture for Grip example."""
    return read_ads(EXAMPLE_PATH_GRIP)


@pytest.fixture(scope="module")
def example_reading_grip_new_format():
    """Get a fixture to the example of ADS Grip reading in the new format."""
    return read_ads(EXAMPLE_PATH_GRIP_NEW_FORMAT)


EXAMPLE_PATH_SBT_UTT = resource_path(PROVIDER_ID, "SBT-UTT/example.json")
EXAMPLE_PATH_SBT_UTT_NEW_FORMAT = resource_path(PROVIDER_ID, "SBT-UTT/new-format.json")


EXAMPLE_PATH_SBT_UTT_UP_SAMPLE = resource_path(
    PROVIDER_ID, "SBT-UTT/6b69efb2-587b-4da8-b21f-5d40279d6369.json"
)
EXAMPLE_PATH_SBT_UTT_UP_SAMPLE2 = resource_path(
    PROVIDER_ID, "SBT-UTT/f07370eb-ce21-4096-8a26-6e325088ceee.json"
)
EXAMPLE_PATH_SBT_UTT_SEGMENTATION_NO_MOTION = resource_path(
    PROVIDER_ID, "SBT-UTT/78bf620a-e7c9-41a1-94e1-91686afd97e2.json"
)
EXAMPLE_PATH_SBT_UTT_SEGMENTATION_FLAGGED = resource_path(
    PROVIDER_ID, "SBT-UTT/92cab597-1c55-4f1b-b58e-fe13beec1641.json"
)
EXAMPLE_PATH_SBT_UTT_SEGMENTATION_FIXED = resource_path(
    PROVIDER_ID, "SBT-UTT/5924fb84-3428-4f9a-a542-a82658851ca5.json"
)
EXAMPLE_PATH_SBT_UTT_PORTRAIT = resource_path(
    PROVIDER_ID, "SBT-UTT/sbt-portrait-example.json"
)


@pytest.fixture(scope="module")
def example_reading_sbt_utt():
    """Fixture for SBT/UTT example."""
    return read_ads(EXAMPLE_PATH_SBT_UTT)


@pytest.fixture(scope="module")
def example_reading_sbt_utt_new_format():
    """Fixture for SBT/UTT example using the new format."""
    return read_ads(EXAMPLE_PATH_SBT_UTT_NEW_FORMAT)


@pytest.fixture
def example_reading_sbt_utt_up_sample():
    """Fixture for SBT/UTT example up-sampled."""
    return read_ads(EXAMPLE_PATH_SBT_UTT_UP_SAMPLE)


@pytest.fixture
def example_reading_sbt_utt_up_sample2():
    """Fixture for SBT/UTT example up-sampled."""
    return read_ads(EXAMPLE_PATH_SBT_UTT_UP_SAMPLE2)


@pytest.fixture
def example_reading_sbt_utt_segmentation_no_motion():
    """Fixture for SBT/UTT example without motion."""
    return read_ads(EXAMPLE_PATH_SBT_UTT_SEGMENTATION_NO_MOTION)


@pytest.fixture
def example_reading_sbt_utt_segmentation_flagged():
    """Fixture for SBT/UTT example with flagged segmentation."""
    return read_ads(EXAMPLE_PATH_SBT_UTT_SEGMENTATION_FLAGGED)


@pytest.fixture(scope="module")
def example_reading_processed_sbt_utt(example_reading_sbt_utt):
    """Fixture for processed SBT/UTT reading."""
    return process_sbt_utt(deepcopy(example_reading_sbt_utt)).get_reading()


@pytest.fixture
def example_reading_sbt_utt_motion_fixed():
    """Fixture for SBT/UTT reading example with fixed motion."""
    return read_ads(EXAMPLE_PATH_SBT_UTT_SEGMENTATION_FIXED)


@pytest.fixture(scope="module")
def example_reading_processed_sbt_utt_new_format(example_reading_sbt_utt_new_format):
    """Fixture for processed SBT/UTT reading new format."""
    return process_sbt_utt(deepcopy(example_reading_sbt_utt_new_format)).get_reading()


@pytest.fixture(scope="module")
def example_reading_sbt_utt_portrait():
    """Fixture for SBT/UTT reading in portrait orientation."""
    return read_ads(EXAMPLE_PATH_SBT_UTT_PORTRAIT)


@pytest.fixture(scope="module")
def example_reading_processed_sbt_utt_portrait(example_reading_sbt_utt_portrait):
    """Fixture for processed SBT/UTT reading in portrait orientation."""
    return process_sbt_utt(deepcopy(example_reading_sbt_utt_portrait)).get_reading()


@pytest.fixture(scope="module")
def example_accelerometer(example_reading_sbt_utt):
    """Get example data set for the accelerometer."""
    # FIXME: replace with synthetic data for module
    return (
        example_reading_sbt_utt.get_level("utt").get_raw_data_set("accelerometer").data
    )


EXAMPLE_PATH_CPS = resource_path(PROVIDER_ID, "CPS/example.json")
EXAMPLE_PATH_CPS_NEW_FORMAT = resource_path(PROVIDER_ID, "CPS/new-format.json")
RESULTS_PATH_CPS = resource_path(PROVIDER_ID, "CPS/results-ads-example.json")
RESULTS_PATH_CPS_NEW_FORMAT = resource_path(
    PROVIDER_ID, "CPS/results-ads-example-new-format.json"
)


@pytest.fixture
def example_json_cps():
    """Fixture for CPS JSON example."""
    return load_json(EXAMPLE_PATH_CPS, encoding="utf-8")


@pytest.fixture
def example_reading_cps(example_json_cps):
    """Fixture for CPS reading example."""
    return parse_ads_raw_json(example_json_cps)


@pytest.fixture
def example_reading_cps_new_format():
    """Fixture for CPS reading example in new format."""
    return read_ads(EXAMPLE_PATH_CPS_NEW_FORMAT)


@pytest.fixture
def example_reading_processed_cps(example_reading_cps):
    """Fixture for processed CPS reading example."""
    return process_cps(deepcopy(example_reading_cps)).get_reading()


@pytest.fixture
def example_reading_processed_cps_new_format(example_reading_cps_new_format):
    """Fixture for processed CPS reading example in new format."""
    return process_cps(deepcopy(example_reading_cps_new_format)).get_reading()


EXAMPLE_PATH_PINCH = resource_path(PROVIDER_ID, "PINCH/example.json")
EXAMPLE_PATH_PINCH2 = resource_path(PROVIDER_ID, "PINCH/example-2.json")
RESULTS_PATH_PINCH2 = resource_path(PROVIDER_ID, "PINCH/example-2-results.json")
EXAMPLE_PATH_PINCH_NEW_FORMAT = resource_path(PROVIDER_ID, "PINCH/new-format.json")
RESULTS_PATH_PINCH_NEW_FORMAT = resource_path(
    PROVIDER_ID, "PINCH/new-format-results.json"
)


@pytest.fixture(scope="module")
def example_json_pinch():
    """Fixture for Pinch JSON example."""
    return load_json(EXAMPLE_PATH_PINCH, encoding="utf-8")


@pytest.fixture(scope="module")
def example_reading_pinch2():
    """Fixture for Pinch reading example."""
    return read_ads(EXAMPLE_PATH_PINCH2)


@pytest.fixture(scope="module")
def example_reading_pinch_new_format():
    """Fixture for Pinch reading example in new format."""
    return read_ads(EXAMPLE_PATH_PINCH_NEW_FORMAT)


@pytest.fixture(scope="module")
def example_level_pinch2_rel(example_reading_pinch2):
    """Fixture for Pinch level."""
    return example_reading_pinch2.get_level("right-extra_large")


@pytest.fixture(scope="module")
def example_pinch_target(example_level_pinch2_rel):
    """Fixture for Pinch target example."""
    return PinchTarget.from_level(example_level_pinch2_rel)


@pytest.fixture(scope="module")
def example_reading_processed_pinch2(example_reading_pinch2):
    """Fixture for processed Pinch example."""
    return process_pinch(deepcopy(example_reading_pinch2)).get_reading()


@pytest.fixture(scope="module")
def example_reading_processed_pinch_new_format(example_reading_pinch_new_format):
    """Fixture for processed Pinch example with new format."""
    return process_pinch(deepcopy(example_reading_pinch_new_format)).get_reading()


EXAMPLE_PATH_6MWT = resource_path(PROVIDER_ID, "6MWT/basic.json")
EXAMPLE_PATH_6MWT_INF_SPEED = resource_path(PROVIDER_ID, "6MWT/inf_speed.json")
EXAMPLE_PATH_6MWT_DUPLICATED_TS = resource_path(PROVIDER_ID, "6MWT/duplicated_ts.json")
EXAMPLE_PATH_6MWT_NO_GPS = resource_path(PROVIDER_ID, "6MWT/no_gps.json")
EXAMPLE_PATH_6MWT_2455 = resource_path(PROVIDER_ID, "6MWT/2455.json")
RESULTS_PATH_6MWT_EXP_LOW = resource_path(
    PROVIDER_ID, "6MWT/expected/test_gait_ads_low.json"
)
RESULTS_PATH_6MWT_EXP_HIGH = resource_path(
    PROVIDER_ID, "6MWT/expected/test_gait_ads_high.json"
)
RESULTS_6MWT_EXP_LOW = read_results(RESULTS_PATH_6MWT_EXP_LOW, True)
RESULTS_6MWT_EXP_HIGH = read_results(RESULTS_PATH_6MWT_EXP_HIGH, True)


EXAMPLE_PATH_PASSIVE = resource_path(PROVIDER_ID, "PASSIVE/example.json")


@pytest.fixture(scope="module")
def example_reading_passive():
    """Fixture for Passive reading example."""
    return read_ads(EXAMPLE_PATH_PASSIVE)
