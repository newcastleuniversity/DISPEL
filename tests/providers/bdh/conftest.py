"""Pytest configuration for BDH provider test cases."""

from copy import deepcopy
from typing import Dict

import pytest

from dispel.io.utils import load_json
from dispel.providers.bdh import PROVIDER_ID
from dispel.providers.bdh import PROVIDER_ID as BDH_PROVIDER
from dispel.providers.bdh.io import parse_bdh_reading, read_bdh
from dispel.providers.bdh.tasks.sbt_utt import process_sbt_utt
from dispel.providers.bdh.tasks.typing.steps import process_typing
from dispel.providers.bdh.tasks.voice import process_voice
from dispel.providers.generic.tasks.cps import process_cps
from dispel.providers.generic.tasks.draw import process_draw
from dispel.providers.generic.tasks.pinch import process_pinch
from dispel.providers.generic.tasks.ps import process_ps
from tests.processing.helper import read_results
from tests.providers import resource_path

EXAMPLE_PATH_MOOD = resource_path(PROVIDER_ID, "MOOD/example-mood.json")


@pytest.fixture
def example_json_mood() -> Dict:
    """Fixture for Mood JSON example."""
    return load_json(EXAMPLE_PATH_MOOD, encoding="utf-8")


@pytest.fixture
def example_reading_mood():
    """Fixture for Mood reading example."""
    return read_bdh(EXAMPLE_PATH_MOOD)


EXAMPLE_PATH_MSIS29 = resource_path(PROVIDER_ID, "MSIS29/example-msis29.json")


@pytest.fixture
def example_json_msis29() -> Dict:
    """Fixture for MSIS-29 JSON example."""
    return load_json(EXAMPLE_PATH_MSIS29, encoding="utf-8")


@pytest.fixture
def example_reading_msis29():
    """Fixture for MSIS-29 reading example."""
    return read_bdh(EXAMPLE_PATH_MSIS29)


EXAMPLE_PATH_ALSFRS = resource_path(PROVIDER_ID, "ALSFRS/alsfrs-example.json")


@pytest.fixture(scope="module")
def example_json_alsfrs():
    """Fixture for ALS-FRS JSON example."""
    return load_json(EXAMPLE_PATH_ALSFRS, encoding="utf-8")


@pytest.fixture(scope="module")
def example_reading_alsfrs(example_json_alsfrs):
    """Fixture for ALS-FRS reading example."""
    return parse_bdh_reading(example_json_alsfrs)


EXAMPLE_PATH_NEUROQOL = resource_path(PROVIDER_ID, "NEUROQOL/neuroqol_example.json")


@pytest.fixture(scope="module")
def example_json_neuroqol():
    """Fixture for NEUROQOL JSON example."""
    return load_json(EXAMPLE_PATH_NEUROQOL, encoding="utf-8")


@pytest.fixture(scope="module")
def example_reading_neuroqol(example_json_neuroqol):
    """Fixture for NEUROQOL reading example."""
    return parse_bdh_reading(example_json_neuroqol)


EXAMPLE_PATH_PDQ39 = resource_path(PROVIDER_ID, "PDQ39/pdq39-example.json")


@pytest.fixture(scope="module")
def example_json_pdq39():
    """Fixture for PDQ-39 JSON example."""
    return load_json(EXAMPLE_PATH_PDQ39, encoding="utf8")


@pytest.fixture(scope="module")
def example_reading_pdq39(example_json_pdq39):
    """Fixture for PDQ-39 reading example."""
    return parse_bdh_reading(example_json_pdq39)


EXAMPLE_PATH_CPS = resource_path(PROVIDER_ID, "CPS/uat_cps.json")
EXAMPLE_PATH_CPS_BUG = resource_path(PROVIDER_ID, "CPS/bug-cps.json")
EXAMPLE_PATH_CPS_TABLE_TYPE4 = resource_path(PROVIDER_ID, "CPS/cps_table_type_4.json")
RESULTS_PATH_CPS = resource_path(PROVIDER_ID, "CPS/results-uat-cps.json")
RESULTS_PATH_CPS_BUG = resource_path(PROVIDER_ID, "CPS/results-cps-reading-bug.json")
RESULTS_PATH_CPS_TABLE_TYPE4 = resource_path(
    PROVIDER_ID, "CPS/results-cps-table-type-4.json"
)


@pytest.fixture
def example_json_cps() -> Dict:
    """Fixture for CPS JSON example."""
    return load_json(EXAMPLE_PATH_CPS, encoding="utf-8")


@pytest.fixture
def example_json_cps_table_type4():
    """Fixture for CPS JSON example."""
    return load_json(EXAMPLE_PATH_CPS_TABLE_TYPE4, encoding="utf-8")


@pytest.fixture
def example_reading_cps(example_json_cps):
    """Fixture for CPS reading example."""
    return parse_bdh_reading(example_json_cps)


@pytest.fixture
def example_reading_cps_bug():
    """Fixture for CPS reading example with a bug."""
    return read_bdh(EXAMPLE_PATH_CPS_BUG)


@pytest.fixture
def example_reading_processed_cps(example_reading_cps):
    """Fixture for processed CPS reading example."""
    return process_cps(example_reading_cps).get_reading()


@pytest.fixture
def example_reading_processed_cps_bug(example_reading_cps_bug):
    """Fixture for processed CPS reading example with bug."""
    return process_cps(example_reading_cps_bug).get_reading()


@pytest.fixture
def example_reading_cps_table_type4(example_json_cps_table_type4):
    """Fixture for CPS reading example."""
    return parse_bdh_reading(example_json_cps_table_type4)


@pytest.fixture
def example_reading_processed_cps_table_type4(example_reading_cps_table_type4):
    """Fixture for processed CPS reading example."""
    return process_cps(example_reading_cps_table_type4).get_reading()


EXAMPLE_PATH_PINCH_BUG = resource_path(PROVIDER_ID, "PINCH/bug-pinch.json")
EXAMPLE_PATH_PINCH_UAT = resource_path(PROVIDER_ID, "PINCH/uat_pinch.json")
EXAMPLE_PATH_PINCH_OVERWRITE = resource_path(
    PROVIDER_ID, "PINCH/uat_pinch_overwrite.json"
)
EXAMPLE_PATH_PINCH_RADIUS_LEVEL_INDEX = resource_path(
    PROVIDER_ID, "PINCH/pinch_radius_lvl_index.json"
)
RESULTS_PATH_PINCH_UAT = resource_path(PROVIDER_ID, "PINCH/uat_pinch_results.json")


@pytest.fixture
def example_json_pinch() -> Dict:
    """Fixture for Pinch JSON example."""
    return load_json(EXAMPLE_PATH_PINCH_UAT, encoding="utf-8")


@pytest.fixture
def example_json_pinch_radius_level_index() -> Dict:
    """Fixture for a pinch example with different radius level.

    Radius level is expected to have values within {0, 1, 2, 3} with an index
    starting at 0. However, for some devices, BDH (former BDH) mobile team has
    implemented it beginning at 1, with values within {1, 2, 3, 4}.

    Returns
    -------
    Dict
        A dictionary of the fixture.
    """
    return load_json(EXAMPLE_PATH_PINCH_RADIUS_LEVEL_INDEX, encoding="utf-8")


@pytest.fixture(scope="module")
def example_reading_processed_pinch_uat():
    """Fixture for processed Pinch reading example from UAT."""
    return process_pinch(read_bdh(EXAMPLE_PATH_PINCH_UAT)).get_reading()


@pytest.fixture(scope="module")
def example_reading_processed_pinch_overwrite():
    """Fixture for processed Pinch reading example (overwrite)."""
    return process_pinch(read_bdh(EXAMPLE_PATH_PINCH_OVERWRITE)).get_reading()


EXAMPLE_PATH_DRAW = resource_path(PROVIDER_ID, "DRAW/uat_drawing.json")
EXAMPLE_PATH_DRAW_ONE_HAND = resource_path(
    PROVIDER_ID, "DRAW/uat_drawing_one_hand.json"
)
EXAMPLE_PATH_DRAW_BUG = resource_path(PROVIDER_ID, "DRAW/bug-drawing.json")
EXAMPLE_PATH_DRAW_BUG_NO_CORNER = resource_path(PROVIDER_ID, "DRAW/drawing_bug.json")
EXAMPLE_PATH_DRAW_BUG_PARSING = resource_path(
    PROVIDER_ID, "DRAW/bug_parsing_drawing.json"
)
EXAMPLE_PATH_DRAW_ORIENTATION_INVALID = resource_path(
    PROVIDER_ID, "DRAW/drawing_orientation_invalid.json"
)
EXAMPLE_PATH_DRAW_NON_CONTINUOUS = resource_path(
    PROVIDER_ID, "DRAW/non_continuous_iphone_X.json"
)
EXAMPLE_PATH_DRAW_OPPOSITE_DIRECTION = resource_path(
    PROVIDER_ID, "DRAW/opp_direction.json"
)


@pytest.fixture(scope="module")
def example_json_draw() -> Dict:
    """Fixture for Drawing JSON example."""
    return load_json(EXAMPLE_PATH_DRAW, encoding="utf-8")


@pytest.fixture(scope="module")
def example_reading_processed_draw(example_json_draw):
    """Fixture for processed Drawing reading example."""
    return process_draw(parse_bdh_reading(example_json_draw)).get_reading()


@pytest.fixture(scope="module")
def example_reading_draw_one_hand():
    """Fixture for a Drawing reading with one hand only."""
    return read_bdh(EXAMPLE_PATH_DRAW_ONE_HAND)


@pytest.fixture(scope="module")
def example_json_draw_orientation_invalid():
    """Fixture for a Drawing reading that should be flagged."""
    return load_json(EXAMPLE_PATH_DRAW_ORIENTATION_INVALID)


@pytest.fixture(scope="module")
def example_reading_processed_draw_orientation_invalid(
    example_json_draw_orientation_invalid,
):
    """Fixture for a processed drawing reading."""
    return process_draw(
        parse_bdh_reading(example_json_draw_orientation_invalid)
    ).get_reading()


@pytest.fixture(scope="module")
def example_reading_processed_draw_non_continuous():
    """Fixture for a processed drawing reading."""
    return process_draw(read_bdh(EXAMPLE_PATH_DRAW_NON_CONTINUOUS)).get_reading()


@pytest.fixture(scope="module")
def example_reading_processed_draw_bug():
    """Fixture for processed Drawing reading with bug."""
    return process_draw(read_bdh(EXAMPLE_PATH_DRAW_BUG)).get_reading()


@pytest.fixture(scope="module")
def example_reading_processed_draw_bug_no_corner():
    """Fixture for processed Drawing reading with no corner bug."""
    return process_draw(read_bdh(EXAMPLE_PATH_DRAW_BUG_NO_CORNER)).get_reading()


@pytest.fixture(scope="module")
def example_reading_processed_draw_bug_parsing():
    """Fixture for processed Drawing with parsing bug."""
    return read_bdh(EXAMPLE_PATH_DRAW_BUG_PARSING)


@pytest.fixture(scope="module")
def example_reading_processed_draw_opposite_direction():
    """Fixture for a drawing in opposite direction from BDH data."""
    return read_bdh(EXAMPLE_PATH_DRAW_OPPOSITE_DIRECTION)


EXAMPLE_PATH_FT = resource_path(PROVIDER_ID, "FINGERTAP/fingertap_1.json")
EXAMPLE_PATH_FT_UNFINISHED = resource_path(BDH_PROVIDER, "FINGERTAP/unfinished_ft.json")


@pytest.fixture
def example_json_ft() -> Dict:
    """Fixture for Finger Tapping JSON example."""
    return load_json(EXAMPLE_PATH_FT, encoding="utf-8")


EXAMPLE_PATH_TYPING = resource_path(PROVIDER_ID, "TYPING/iphone/typing-example.json")
EXAMPLE_PATH_TYPING_AUTOCOMPLETE = resource_path(
    PROVIDER_ID, "TYPING/iphone/uat_typing_autocomplete.json"
)
EXAMPLE_PATH_TYPING_BUG_TIMESTAMP = resource_path(
    PROVIDER_ID, "TYPING/android/uat_typing_bug_ts.json"
)
EXAMPLE_PATH_TYPING_BUG_MISSING_IMU = resource_path(
    PROVIDER_ID, "TYPING/iphone/typing-example_wo_imus.json"
)
EXAMPLE_PATH_TYPING_CANCELLED = resource_path(
    PROVIDER_ID, "TYPING/iphone/typing-example-cancelled.json"
)


@pytest.fixture(scope="module")
def example_reading_typing():
    """Fixture for Typing reading example."""
    return read_bdh(EXAMPLE_PATH_TYPING)


@pytest.fixture(scope="module")
def example_reading_typing_autocomplete():
    """Fixture for Typing reading example with autocomplete."""
    return read_bdh(EXAMPLE_PATH_TYPING_AUTOCOMPLETE)


@pytest.fixture(scope="module")
def example_reading_typing_bug_timestamp():
    """Fixture for Typing reading example with timestamp bug."""
    # In this example, some keys are released after the last word disappearance
    # timestamp.
    return read_bdh(EXAMPLE_PATH_TYPING_BUG_TIMESTAMP)


@pytest.fixture(scope="module")
def example_reading_typing_bug_missing_imu():
    """Fixture for Typing reading with missing IMU data."""
    return read_bdh(EXAMPLE_PATH_TYPING_BUG_MISSING_IMU)


@pytest.fixture(scope="module")
def example_reading_typing_cancelled():
    """Fixture for Typing reading example that was cancelled."""
    return read_bdh(EXAMPLE_PATH_TYPING_CANCELLED)


@pytest.fixture(scope="module")
def example_reading_processed_typing(example_reading_typing):
    """Fixture for processed Typing reading."""
    return process_typing(example_reading_typing).get_reading()


@pytest.fixture(scope="module")
def example_reading_processed_typing_bug_timestamp(
    example_reading_typing_bug_timestamp,
):
    """Fixture for processed Typing reading with timestamp bug."""
    return process_typing(example_reading_typing_bug_timestamp).get_reading()


@pytest.fixture(scope="module")
def example_reading_processed_typing_bug_missing_imu(
    example_reading_typing_bug_missing_imu,
):
    """Fixture for processed Typing reading with missing IMU."""
    return process_typing(example_reading_typing_bug_missing_imu).get_reading()


@pytest.fixture(scope="module")
def example_reading_processed_typing_autocomplete(example_reading_typing_autocomplete):
    """Fixture for processed Typing reading with autocomplete."""
    return process_typing(example_reading_typing_autocomplete).get_reading()


EXAMPLE_PATH_PS = resource_path(PROVIDER_ID, "PS/ps_staging_v4.json")
EXAMPLE_PATH_PS_STATIC = resource_path(PROVIDER_ID, "PS/static_ps.json")
EXAMPLE_PATH_PS_SLOW_SHAKY = resource_path(PROVIDER_ID, "PS/slow_shaky_ps.json")


@pytest.fixture(scope="module")
def example_json_ps() -> Dict:
    """Fixture for PS JSON example."""
    return load_json(EXAMPLE_PATH_PS, encoding="utf-8")


@pytest.fixture(scope="module")
def example_reading_ps(example_json_ps):
    """Fixture for PS reading example."""
    return parse_bdh_reading(example_json_ps)


@pytest.fixture(scope="module")
def example_reading_ps_static():
    """Fixture for PS reading with static motion."""
    return read_bdh(EXAMPLE_PATH_PS_STATIC)


@pytest.fixture(scope="module")
def example_reading_ps_slow_shaky():
    """Fixture for PS with slow shaky motion."""
    return read_bdh(EXAMPLE_PATH_PS_SLOW_SHAKY)


@pytest.fixture(scope="module")
def example_reading_processed_ps(example_reading_ps):
    """Fixture for processed PS example."""
    return process_ps(deepcopy(example_reading_ps)).get_reading()


@pytest.fixture(scope="module")
def example_reading_processed_ps_static(example_reading_ps_static):
    """Fixture for processed PS reading with static motion."""
    return process_ps(deepcopy(example_reading_ps_static)).get_reading()


@pytest.fixture(scope="module")
def example_reading_processed_ps_slow_shaky(example_reading_ps_slow_shaky):
    """Fixture for processed PS reading with slow shaky motion."""
    return process_ps(deepcopy(example_reading_ps_slow_shaky)).get_reading()


EXAMPLE_PATH_SBT_UTT = resource_path(PROVIDER_ID, "SBT-UTT/igni_sbt_utt.json")
EXAMPLE_PATH_SBT_UTT_BUG = resource_path(BDH_PROVIDER, "SBT-UTT/bug-sbt_utt.json")
EXAMPLE_PATH_SBT_UTT_NO_TURNS = resource_path(BDH_PROVIDER, "SBT-UTT/uat_sbt_utt.json")


@pytest.fixture(scope="module")
def example_json_sbt_utt() -> Dict:
    """Fixture for SBT/UTT JSON."""
    return load_json(EXAMPLE_PATH_SBT_UTT, encoding="utf-8")


@pytest.fixture(scope="module")
def example_reading_sbt_utt(example_json_sbt_utt):
    """Fixture for SBT/UTT example reading."""
    return parse_bdh_reading(example_json_sbt_utt)


@pytest.fixture
def example_reading_sbt_utt_bug():
    """Fixture for SBT/UTT example with bug."""
    return read_bdh(EXAMPLE_PATH_SBT_UTT_BUG)


@pytest.fixture
def example_reading_sbt_utt_no_turns():
    """Fixture for SBT/UTT example reading with no turns."""
    return read_bdh(EXAMPLE_PATH_SBT_UTT_NO_TURNS)


@pytest.fixture(scope="module")
def example_reading_processed_sbt_utt(example_reading_sbt_utt):
    """Fixture for a processed SBT-UTT reading from ADS data."""
    return process_sbt_utt(deepcopy(example_reading_sbt_utt)).get_reading()


EXAMPLE_PATH_6MWT = resource_path(BDH_PROVIDER, "6MWT/igni_6mwt.json")
EXAMPLE_PATH_6MWT_UAT = resource_path(PROVIDER_ID, "6MWT/uat_6mwt.json")
EXAMPLE_PATH_6MWT_BUG_INDEX = resource_path(BDH_PROVIDER, "6MWT/bug-6mwt-index.json")
RESULTS_PATH_6MWT_EXP_LOW = resource_path(
    BDH_PROVIDER, "6MWT/expected/test_gait_bdh_low.json"
)
RESULTS_PATH_6MWT_EXP_HIGH = resource_path(
    BDH_PROVIDER, "6MWT/expected/test_gait_bdh_high.json"
)
RESULTS_6MWT_EXP_LOW = read_results(RESULTS_PATH_6MWT_EXP_LOW, True)
RESULTS_6MWT_EXP_HIGH = read_results(RESULTS_PATH_6MWT_EXP_HIGH, True)


@pytest.fixture
def example_json_6mwt_uat() -> Dict:
    """Fixture for 6MWT example json."""
    return load_json(EXAMPLE_PATH_6MWT_UAT, encoding="utf-8")


EXAMPLE_PATH_VOICE = resource_path(PROVIDER_ID, "VOICE/android/voice_redmi8_1.json")
EXAMPLE_PATH_VOICE_INVALID = resource_path(
    PROVIDER_ID, "VOICE/android/invalid_voice.json"
)


@pytest.fixture(scope="module")
def example_reading_voice():
    """Fixture for voice reading."""
    return read_bdh(EXAMPLE_PATH_VOICE)


@pytest.fixture(scope="module")
def example_reading_voice_invalid():
    """Fixture for invalid voice reading."""
    return read_bdh(EXAMPLE_PATH_VOICE_INVALID)


@pytest.fixture(scope="module")
def example_reading_processed_voice(example_reading_voice):
    """Fixture for processed voice reading."""
    return process_voice(deepcopy(example_reading_voice)).get_reading()
