"""Test cases for :mod:`dispel.providers.ads.tasks.grip`."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from dispel.data.levels import Context, Level
from dispel.data.values import Value, ValueDefinition
from dispel.processing import process
from dispel.processing.level import LevelIdFilter
from dispel.processing.transform import ConcatenateLevels
from dispel.providers.ads.tasks.grip import (
    N_SEQUENCE,
    TargetPressureModality,
    TransformPlateau,
    TransformPressureError,
    avg_diff_discrete_pdf,
    compute_second_derivative_spikes,
    extend_and_convert_plateau,
    fill_plateau,
    get_target_pressure,
    process_grip,
    refined_target,
    remove_short_plateau,
    rms_pressure_error,
    smooth_discrete_pdf,
)
from dispel.providers.generic.sensor import FREQ_100HZ, Resample, SetTimestampIndex
from tests.processing.helper import assert_level_values


@pytest.fixture
def pressure_data():
    """Create a fixture of an interpolated pressure dataframe."""
    return pd.DataFrame(
        {"pressure": [0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1]},
        index=pd.date_range(0, periods=11, freq="3ns"),
    )


@pytest.fixture
def timeframe_contexts():
    """Create a fixture of contexts with effective_time_frame."""

    def _value_factory(key, value):
        return Value(ValueDefinition(key, key), value)

    context = Context()
    context.set(_value_factory("level_0", Level(id_="1", start=0, end=3 * 3)))

    for i in range(1, N_SEQUENCE):
        context.set(
            _value_factory(
                f"level_{i}", Level(id_=str(i), start=i * 3 + 3 * 3, end=i * 3 + 3 * 3)
            )
        )

    return context


@pytest.fixture
def target_pressure_context():
    """Create a fixture of contexts with targetPressure."""
    return Context(
        [
            Value(
                ValueDefinition(f"targetPressure_{i}", f"targetPressure Level {i}"),
                i + 0.5,
            )
            for i in range(N_SEQUENCE)
        ]
    )


def test_smooth_discrete_pdf(pressure_data):
    """Test function smooth_discrete_pdf."""
    assert (
        smooth_discrete_pdf(pressure_data["pressure"])
        == [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    ).all()


def test_avg_diff_discrete_pdf(pressure_data, timeframe_contexts):
    """Test avg_diff_discrete_pdf."""
    # run on the dummy example
    data = avg_diff_discrete_pdf(pressure_data, timeframe_contexts)

    assert (data["discrete-pdf"] == [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]).all()
    assert (data["diff-discrete-pdf"] == [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]).all()
    assert (
        data["avg-diff-discrete-pdf"] == [0.25, 0.25, 0.25, 0.25, 0, 0, 1, 0, 0, 0, 0]
    ).all()
    assert (data["level-sequence"] == [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7]).all()


def test_get_target_pressure(target_pressure_context):
    """Test function get_target_pressure."""
    for seq in range(8):
        assert get_target_pressure(target_pressure_context, seq) == seq + 0.5


@pytest.fixture
def diff_discrete_pdf_data():
    """Create a fixture with predictable second discrete derivative."""
    return pd.DataFrame(
        {
            "diff-discrete-pdf": [0] * 5 + [1] * 25 + [0] * 15,
            "avg-diff-discrete-pdf": [5] * 10 + [0] * 35,
        }
    )


def test_compute_second_derivative_spikes(diff_discrete_pdf_data):
    """Test function compute_second_derivative_spikes."""
    # compute average diff discrete pdf
    detected_plateau = compute_second_derivative_spikes(
        diff_discrete_pdf_data[["diff-discrete-pdf", "avg-diff-discrete-pdf"]]
    )
    assert (detected_plateau == [1] * 10 + [0] * 35).all()


@pytest.fixture
def detected_plateau_data():
    """Create a fixture with predictable filled plateau."""
    return pd.DataFrame(
        {"detected-plateau": [1] * 10 + [0] * 8 + [1] + [0] * 20 + [1] * 5}
    )


def test_fill_plateau(detected_plateau_data):
    """Test function fill_plateau."""
    filled_plateau = fill_plateau(detected_plateau_data["detected-plateau"])
    assert (filled_plateau == [1] * 19 + [0] * 20 + [1] * 5).all()


@pytest.fixture
def detected_plateau_w_short_data():
    """Create a fixture with removable short plateaus."""
    return pd.DataFrame({"detected-plateau": [0] * 8 + [1] * 40 + [0] * 20 + [1] * 5})


def test_remove_short_plateau(detected_plateau_w_short_data):
    """Test remove_short_plateau."""
    long_plateau = remove_short_plateau(
        detected_plateau_w_short_data["detected-plateau"]
    )
    assert (long_plateau == [0] * 8 + [1] * 40 + [0] * 25).all()


@pytest.fixture
def extendable_plateau():
    """Create a fixture with extendable plateaus."""
    return pd.DataFrame(
        {
            "detected-plateau": [1] * 200 + [0] * 200,
            "pressure": [5] * 180 + [4.5] * 20 + [1.1] * 200,
        }
    )


def test_extend_and_convert_plateau(extendable_plateau):
    """Test extend and convert plateau function."""
    extended_plateau = extend_and_convert_plateau(
        extendable_plateau["pressure"], extendable_plateau["detected-plateau"]
    )
    assert (extended_plateau == [5] * 180 + [4.5] * 20 + [0] * 200).all()


@pytest.fixture
def plateau_and_level():
    """Create a fixture with predictable refined target."""
    return pd.DataFrame(
        {
            "detected-plateau": [5] * 180 + [1.5] * 20 + [0] * 200,
            "level-sequence": [0] * 180 + [1] * 220,
        }
    )


def test_refined_target(target_pressure_context, plateau_and_level):
    """Test refined target function."""
    refined_target_df = refined_target(
        plateau_and_level[["detected-plateau", "level-sequence"]],
        target_pressure_context,
    )
    assert (refined_target_df.iloc[0:200] == 0.5).all()
    assert (refined_target_df.iloc[200:] == 0).all()


LEVEL_ID_RIGTHS = ["right"] + [f"right-0{attempt}" for attempt in range(2, 9)]


@pytest.fixture
def dummy_pressure_error():
    """Create a fixture with dummy pressure error."""
    return pd.DataFrame({"pressure-error": [1, 2, 3, 4, 5]})


@pytest.fixture
def dummy_pressure_cat():
    """Create a fixture with dummy pressure categories."""
    return pd.DataFrame(
        {
            "pressure-category": [
                TargetPressureModality.LOW,
                TargetPressureModality.LOW,
                TargetPressureModality.MEDIUM,
                TargetPressureModality.MEDIUM,
                TargetPressureModality.MEDIUM,
            ]
        }
    )


def test_rms_pressure_error(dummy_pressure_error, dummy_pressure_cat):
    """Tests ExtractRMSPressure(ExtractStep)."""
    assert rms_pressure_error(dummy_pressure_error, dummy_pressure_cat) == np.sqrt(
        5 * (5 + 1) * (2 * 5 + 1) / 6 / 5
    )
    assert rms_pressure_error(
        dummy_pressure_error, dummy_pressure_cat, TargetPressureModality.LOW
    ) == np.sqrt((1 + 4) / 2)
    assert rms_pressure_error(
        dummy_pressure_error, dummy_pressure_cat, TargetPressureModality.MEDIUM
    ) == np.sqrt((9 + 16 + 25) / 3)


def test_transform_plateau(example_reading_grip):
    """Tests TransformPlateau(TransformStep)."""
    reading = deepcopy(example_reading_grip)
    process(
        reading,
        [
            ConcatenateLevels(
                new_level_id="right-all",
                data_set_id="screen",
                level_filter=LevelIdFilter(LEVEL_ID_RIGTHS),
            ),
            SetTimestampIndex(
                "screen",
                columns=["pressure"],
                time_stamp_column="tsTouch",
                level_filter=["right-all", "left-all"],
            ),
            Resample(
                "screen_ts",
                aggregations=["pad"],
                columns=["pressure"],
                freq=FREQ_100HZ,
                level_filter=["right-all", "left-all"],
            ),
            TransformPlateau("right-all"),
        ],
    )

    plateau_data = reading.get_level("right-all").get_raw_data_set("plateau").data

    expected_columns = [
        "discrete-pdf",
        "diff-discrete-pdf",
        "avg-diff-discrete-pdf",
        "detected-plateau",
        "level-sequence",
        "refined-target",
    ]
    for col in expected_columns:
        assert col in plateau_data.columns


def test_transform_pressure_error(example_reading_grip):
    """Tests TransformPressureError(TransformStep)."""
    reading = deepcopy(example_reading_grip)
    process(
        reading,
        [
            ConcatenateLevels(
                new_level_id="right-all",
                data_set_id="screen",
                level_filter=LevelIdFilter(LEVEL_ID_RIGTHS),
            ),
            SetTimestampIndex(
                "screen",
                columns=["pressure"],
                time_stamp_column="tsTouch",
                level_filter=["right-all", "left-all"],
            ),
            Resample(
                "screen_ts",
                aggregations=["pad"],
                columns=["pressure"],
                freq="10ms",
                level_filter=["right-all", "left-all"],
            ),
            TransformPlateau("right-all"),
            TransformPressureError("right-all"),
        ],
    )
    level = reading.get_level("right-all")
    pressure_error_data = level.get_raw_data_set("pressure-error").data

    assert "pressure-error" in pressure_error_data.columns

    plateau_data = level.get_raw_data_set("plateau").data
    mask = pressure_error_data.index
    assert (
        (
            plateau_data.loc[mask, "detected-plateau"]
            - plateau_data.loc[mask, "refined-target"]
        ).values
        == pressure_error_data["pressure-error"].values
    ).all()


@pytest.fixture(scope="module")
def processed_grip_reading_ads(example_reading_grip):
    """Create a fixture for a processed grip reading from ADS data."""
    return process_grip(deepcopy(example_reading_grip)).get_reading()


@pytest.fixture(scope="module")
def example_reading_processed_draw_new_format(example_reading_grip_new_format):
    """Create a fixture for a processed grip reading from ADS data."""
    return process_grip(example_reading_grip_new_format).get_reading()


@pytest.mark.parametrize(
    "level_id,expected",
    [
        (
            "right-all",
            {
                "grip-right-pressure-rmse": 0.12393454980582448,
                "grip-right_low-pressure-rmse": 0.08902126389711651,
                "grip-right_medium-pressure-rmse": 0.12676716986055306,
                "grip-right_high-pressure-rmse": 0.15026533053861998,
                "grip-right_very_high-pressure-rmse": 0.11925875216644657,
                "grip-right_s4m-pressure-rmse": 0.14065566399912732,
                "grip-right_very_high_s4m-pressure-rmse": 0.13460700635294073,
                "grip-right_low_s4m-pressure-rmse": 0.09518487633865479,
                "grip-right_medium_s4m-pressure-rmse": 0.14098054569652418,
                "grip-right_high_s4m-pressure-rmse": 0.1780673301359728,
                "grip-right_s2m-pressure-rmse": 0.11160083017618959,
                "grip-right_high_s2m-pressure-rmse": 0.13316125115510993,
                "grip-right_very_high_s2m-pressure-rmse": 0.14039149993957986,
                "grip-right_low_s2m-pressure-rmse": 0.05006247331894828,
                "grip-right_medium_s2m-pressure-rmse": 0.09929445455988514,
                "grip-right_s5l-pressure-rmse": 0.13271698317026775,
                "grip-right_high_s5l-pressure-rmse": 0.1702326373753715,
                "grip-right_very_high_s5l-pressure-rmse": 0.12304874286932548,
                "grip-right_low_s5l-pressure-rmse": 0.0911474015828885,
                "grip-right_medium_s5l-pressure-rmse": 0.12976626523775592,
                "grip-right-acc_sig_ene": 0.5042184591293335,
                "grip-right-acc_ps_ene_x": 4.491406756124139e-05,
                "grip-right-acc_ps_ene_y": 1.1289137070019661e-05,
                "grip-right-acc_ps_ene_z": 0.00013674604605107277,
                "grip-right-acc_ps_ene_xyz": 5.914258805139738e-05,
                "grip-right-acc_ps_peak_x": 0.390625,
                "grip-right-acc_ps_peak_y": 0.3125,
                "grip-right-acc_ps_peak_z": 9.453125,
                "grip-right-acc_ps_peak_xyz": 0.15625,
                "grip-right-acc_ps_ent_x": 6.203073933117769,
                "grip-right-acc_ps_ent_y": 6.753858236197177,
                "grip-right-acc_ps_ent_z": 6.590158621349445,
                "grip-right-acc_ps_ent_xyz": 6.422915235082354,
                "grip-right-acc_ps_amp_x": 7.290827397810062e-06,
                "grip-right-acc_ps_amp_y": 6.990337624301901e-07,
                "grip-right-acc_ps_amp_z": 9.0196999735781e-06,
                "grip-right-acc_ps_amp_xyz": 8.589388016844168e-06,
                "grip-right-gyr_sig_ene": 3.8228230476379395,
                "grip-right-gyr_ps_ene_x": 0.002480414521457419,
                "grip-right-gyr_ps_ene_y": 0.0005994565352107628,
                "grip-right-gyr_ps_ene_z": 0.0002378240001110754,
                "grip-right-gyr_ps_ene_xyz": 0.0016757538296019447,
                "grip-right-gyr_ps_peak_x": 8.75,
                "grip-right-gyr_ps_peak_y": 0.234375,
                "grip-right-gyr_ps_peak_z": 0.234375,
                "grip-right-gyr_ps_peak_xyz": 0.078125,
                "grip-right-gyr_ps_ent_x": 6.896227470619108,
                "grip-right-gyr_ps_ent_y": 5.7491422258265406,
                "grip-right-gyr_ps_ent_z": 3.4844201643892974,
                "grip-right-gyr_ps_ent_xyz": 4.662722968470728,
                "grip-right-gyr_ps_amp_x": 7.896569150034338e-05,
                "grip-right-gyr_ps_amp_y": 0.0002539158158469945,
                "grip-right-gyr_ps_amp_z": 0.0020908652804791927,
                "grip-right-gyr_ps_amp_xyz": 0.0018887771293520927,
                "grip-right-applied_force-min": 0.8666666746139526,
                "grip-right-applied_force-max": 1.2833333015441895,
                "grip-right-applied_force-mean": 1.0023576540453083,
                "grip-right-applied_force-median": 1,
                "grip-right-applied_force-std": 0.05372798101635858,
                "grip-right-applied_force-cv": 0.05360160697084874,
                "grip-right_low-applied_force-min": 0.8666666746139526,
                "grip-right_low-applied_force-max": 1.2833333015441895,
                "grip-right_low-applied_force-mean": 1.011221292185634,
                "grip-right_low-applied_force-median": 0.9833333492279053,
                "grip-right_low-applied_force-std": 0.08835732745454544,
                "grip-right_low-applied_force-cv": 0.08737684633159933,
                "grip-right_medium-applied_force-min": 0.8910256165724534,
                "grip-right_medium-applied_force-max": 1.1153846520643969,
                "grip-right_medium-applied_force-mean": 1.0147639259644683,
                "grip-right_medium-applied_force-median": 1.0192308432957151,
                "grip-right_medium-applied_force-std": 0.0464892352483451,
                "grip-right_medium-applied_force-cv": 0.045812857610364945,
                "grip-right_high-applied_force-min": 0.9206348941439674,
                "grip-right_high-applied_force-max": 1.067460264478411,
                "grip-right_high-applied_force-mean": 0.9923518154992981,
                "grip-right_high-applied_force-median": 0.9960318064022885,
                "grip-right_high-applied_force-std": 0.03496713918865262,
                "grip-right_high-applied_force-cv": 0.03523663547797213,
                "grip-right_very_high-applied_force-min": 0.9396551395284719,
                "grip-right_very_high-applied_force-max": 1.028735637664795,
                "grip-right_very_high-applied_force-mean": 0.9915412436187123,
                "grip-right_very_high-applied_force-median": 0.9971263985099004,
                "grip-right_very_high-applied_force-std": 0.018750335087011863,
                "grip-right_very_high-applied_force-cv": 0.018910292645599847,
                "grip-right_s4m-applied_force-std": 0.05928894363588901,
                "grip-right_s4m-applied_force-mean": 1.0045354019514994,
                "grip-right_s4m-applied_force-min": 0.8373016215415965,
                "grip-right_s4m-applied_force-median": 1.0,
                "grip-right_s4m-applied_force-cv": 0.059021258504886,
                "grip-right_s4m-applied_force-max": 1.2833333015441895,
                "grip-right_very_high_s4m-applied_force-std": 0.02006262970462202,
                "grip-right_very_high_s4m-applied_force-mean": 0.9883119942389533,
                "grip-right_very_high_s4m-applied_force-min": 0.9396551086276117,
                "grip-right_very_high_s4m-applied_force-median": 0.9913792777430015,
                "grip-right_very_high_s4m-applied_force-cv": 0.020299894994263615,
                "grip-right_very_high_s4m-applied_force-max": 1.0431034468583913,
                "grip-right_low_s4m-applied_force-std": 0.09181981235169866,
                "grip-right_low_s4m-applied_force-mean": 1.025302244359965,
                "grip-right_low_s4m-applied_force-min": 0.8666666746139526,
                "grip-right_low_s4m-applied_force-median": 1.0166666507720947,
                "grip-right_low_s4m-applied_force-cv": 0.08955389774750398,
                "grip-right_low_s4m-applied_force-max": 1.2833333015441895,
                "grip-right_medium_s4m-applied_force-std": 0.05179229378792016,
                "grip-right_medium_s4m-applied_force-mean": 1.016157729357694,
                "grip-right_medium_s4m-applied_force-min": 0.8910256492550794,
                "grip-right_medium_s4m-applied_force-median": 1.0192308432957151,
                "grip-right_medium_s4m-applied_force-cv": 0.05096875444785299,
                "grip-right_medium_s4m-applied_force-max": 1.1153846929764635,
                "grip-right_high_s4m-applied_force-std": 0.04099809801393287,
                "grip-right_high_s4m-applied_force-mean": 0.989102230684064,
                "grip-right_high_s4m-applied_force-min": 0.8373016215415965,
                "grip-right_high_s4m-applied_force-median": 0.9920634992719151,
                "grip-right_high_s4m-applied_force-cv": 0.041449808464771684,
                "grip-right_high_s4m-applied_force-max": 1.0714286200854266,
                "grip-right-rt-min": 0.31,
                "grip-right-rt-max": 0.43,
                "grip-right-rt-mean": 0.35857142857142854,
                "grip-right-rt-median": 0.32,
                "grip-right-rt-std": 0.057858612562167844,
                "grip-right_low_to_high-tt-mean": 1.1,
                "grip-right_medium_to_high-tt-mean": 0.31,
                "grip-right_medium_to_very_high-tt-mean": 0.72,
                "grip-right_high_to_medium-tt-mean": 0.69,
                "grip-right_high_to_very_high-tt-mean": 0.72,
                "grip-right_very_high_to_low-tt-mean": 0.23,
                "grip-right_very_high_to_medium-tt-mean": 0.71,
                "grip-right_xpos-cv": 0.008721775,
                "grip-right_ypos-cv": 0.0049452735,
                "grip-right-pressure_overshoot-mean": -0.11666662352425712,
                "grip-right-pressure_overshoot-std": 0.5846017624595308,
                "grip-right-pressure_overshoot-median": 0.23333311080932617,
                "grip-right-pressure_overshoot-min": -0.9833332300186157,
                "grip-right-pressure_overshoot-max": 0.3000001907348633,
                "grip-right-applied_force-90th": 1.064102597020318,
                "grip-right_high_very_high-maf_diff": 0.0008106495534743186,
                "grip-right_s4m-applied_force-90th": 1.0666667222976685,
                "grip-right_high_very_high_s4m-maf_diff": 0.0007902364451106703,
                "grip-right_s2m-applied_force-mean": 1.0015939831438143,
                "grip-right_s2m-applied_force-std": 0.037240175306739946,
                "grip-right_s2m-applied_force-median": 1.0,
                "grip-right_s2m-applied_force-min": 0.8910256492550794,
                "grip-right_s2m-applied_force-max": 1.1166666746139526,
                "grip-right_s2m-applied_force-cv": 0.03718090956362385,
                "grip-right_s2m-applied_force-90th": 1.0384615948919735,
                "grip-right_high_very_high_s2m-maf_diff": 0.004465414556420577,
                "grip-right_s5l-applied_force-mean": 1.0024679966892813,
                "grip-right_s5l-applied_force-std": 0.055215245023469174,
                "grip-right_s5l-applied_force-median": 1.0,
                "grip-right_s5l-applied_force-min": 0.8373016215415965,
                "grip-right_s5l-applied_force-max": 1.2833333015441895,
                "grip-right_s5l-applied_force-cv": 0.055079309470049194,
                "grip-right_s5l-applied_force-90th": 1.064102597020318,
                "grip-right_high_very_high_s5l-maf_diff": 0.0022415273969320637,
                "grip-right_low-applied_force-90th": 1.1166666746139526,
                "grip-right_low_s4m-applied_force-90th": 1.1833332777023315,
                "grip-right_low_s2m-applied_force-mean": 1.0261250083148479,
                "grip-right_low_s2m-applied_force-std": 0.04275869432302697,
                "grip-right_low_s2m-applied_force-median": 1.0166666507720947,
                "grip-right_low_s2m-applied_force-min": 0.9333333373069763,
                "grip-right_low_s2m-applied_force-max": 1.1166666746139526,
                "grip-right_low_s2m-applied_force-cv": 0.04167006356588791,
                "grip-right_low_s2m-applied_force-90th": 1.0833333730697632,
                "grip-right_low_s5l-applied_force-mean": 1.0166666645112388,
                "grip-right_low_s5l-applied_force-std": 0.08966209053550867,
                "grip-right_low_s5l-applied_force-median": 1.0,
                "grip-right_low_s5l-applied_force-min": 0.8666666746139526,
                "grip-right_low_s5l-applied_force-max": 1.2833333015441895,
                "grip-right_low_s5l-applied_force-cv": 0.0881922203858367,
                "grip-right_low_s5l-applied_force-90th": 1.149999976158142,
                "grip-right_medium-applied_force-90th": 1.064102597020318,
                "grip-right_medium_s4m-applied_force-90th": 1.0833333486165762,
                "grip-right_medium_s2m-applied_force-mean": 1.0051314478670328,
                "grip-right_medium_s2m-applied_force-std": 0.037891136138837093,
                "grip-right_medium_s2m-applied_force-median": 1.0192308432957151,
                "grip-right_medium_s2m-applied_force-min": 0.8910256492550794,
                "grip-right_medium_s2m-applied_force-max": 1.064102597020318,
                "grip-right_medium_s2m-applied_force-cv": 0.037697692395601626,
                "grip-right_medium_s2m-applied_force-90th": 1.0384615948919735,
                "grip-right_medium_s5l-applied_force-mean": 1.0149466343193367,
                "grip-right_medium_s5l-applied_force-std": 0.04764330178474576,
                "grip-right_medium_s5l-applied_force-median": 1.0192308432957151,
                "grip-right_medium_s5l-applied_force-min": 0.8910256492550794,
                "grip-right_medium_s5l-applied_force-max": 1.1153846929764635,
                "grip-right_medium_s5l-applied_force-cv": 0.04694168163501251,
                "grip-right_medium_s5l-applied_force-90th": 1.070512847552404,
                "grip-right_high-applied_force-90th": 1.0396826171730866,
                "grip-right_high_s4m-applied_force-90th": 1.0396826171730866,
                "grip-right_high_s2m-applied_force-mean": 0.9853373481131666,
                "grip-right_high_s2m-applied_force-std": 0.02814601291302105,
                "grip-right_high_s2m-applied_force-median": 0.9841269985438301,
                "grip-right_high_s2m-applied_force-min": 0.9206349359528194,
                "grip-right_high_s2m-applied_force-max": 1.0198413085865434,
                "grip-right_high_s2m-applied_force-cv": 0.02856484935531791,
                "grip-right_high_s2m-applied_force-90th": 1.0119048078584583,
                "grip-right_high_s5l-applied_force-mean": 0.9889167105659626,
                "grip-right_high_s5l-applied_force-std": 0.03900629871845262,
                "grip-right_high_s5l-applied_force-median": 0.9960318064022885,
                "grip-right_high_s5l-applied_force-min": 0.8373016215415965,
                "grip-right_high_s5l-applied_force-max": 1.0714286200854266,
                "grip-right_high_s5l-applied_force-cv": 0.03944346202434894,
                "grip-right_high_s5l-applied_force-90th": 1.0396826171730866,
                "grip-right_very_high-applied_force-90th": 1.0114942415337977,
                "grip-right_very_high_s4m-applied_force-90th": 1.0143677608105968,
                "grip-right_very_high_s2m-applied_force-mean": 0.9898027626695872,
                "grip-right_very_high_s2m-applied_force-std": 0.021980081923441537,
                "grip-right_very_high_s2m-applied_force-median": 0.9971263985099004,
                "grip-right_very_high_s2m-applied_force-min": 0.9396551086276117,
                "grip-right_very_high_s2m-applied_force-max": 1.0172413623006964,
                "grip-right_very_high_s2m-applied_force-cv": 0.022206527151085412,
                "grip-right_very_high_s2m-applied_force-90th": 1.0143677608105968,
                "grip-right_very_high_s5l-applied_force-mean": 0.9911582379628947,
                "grip-right_very_high_s5l-applied_force-std": 0.019294665111035714,
                "grip-right_very_high_s5l-applied_force-median": 0.9971263985099004,
                "grip-right_very_high_s5l-applied_force-min": 0.9396551086276117,
                "grip-right_very_high_s5l-applied_force-max": 1.0431034468583913,
                "grip-right_very_high_s5l-applied_force-cv": 0.019466785798695076,
                "grip-right_very_high_s5l-applied_force-90th": 1.0143677608105968,
            },
        ),
        (
            "left-all",
            {
                "grip-left-pressure-rmse": 0.25851664964044324,
                "grip-left_low-pressure-rmse": 0.09439748293253869,
                "grip-left_medium-pressure-rmse": 0.1273339792321846,
                "grip-left_high-pressure-rmse": 0.39245827007783385,
                "grip-left_very_high-pressure-rmse": 0.3247256439856343,
                "grip-left_s4m-pressure-rmse": 0.20857976383740842,
                "grip-left_very_high_s4m-pressure-rmse": 0.35401374761454896,
                "grip-left_low_s4m-pressure-rmse": 0.10081955596727245,
                "grip-left_medium_s4m-pressure-rmse": 0.16442381028832434,
                "grip-left_high_s4m-pressure-rmse": 0.09555922187887236,
                "grip-left_s2m-pressure-rmse": 0.09715931915272603,
                "grip-left_high_s2m-pressure-rmse": 0.09548544831204411,
                "grip-left_very_high_s2m-pressure-rmse": 0.1305490443092726,
                "grip-left_low_s2m-pressure-rmse": 0.04832614617789052,
                "grip-left_medium_s2m-pressure-rmse": 0.09625731891382429,
                "grip-left_s5l-pressure-rmse": 0.19185962411072385,
                "grip-left_high_s5l-pressure-rmse": 0.09000537600524,
                "grip-left_very_high_s5l-pressure-rmse": 0.32054906999211064,
                "grip-left_low_s5l-pressure-rmse": 0.09774257918483882,
                "grip-left_medium_s5l-pressure-rmse": 0.14961417073193,
                "grip-left-acc_sig_ene": 0.3951053321361542,
                "grip-left-acc_ps_ene_x": 2.873769954642169e-05,
                "grip-left-acc_ps_ene_y": 2.302886552230099e-05,
                "grip-left-acc_ps_ene_z": 6.713242138245423e-05,
                "grip-left-acc_ps_ene_xyz": 4.6765561885209195e-05,
                "grip-left-acc_ps_peak_x": 3.046875,
                "grip-left-acc_ps_peak_y": 9.296875,
                "grip-left-acc_ps_peak_z": 9.0625,
                "grip-left-acc_ps_peak_xyz": 0.46875,
                "grip-left-acc_ps_ent_x": 6.570514768750408,
                "grip-left-acc_ps_ent_y": 6.658750126202161,
                "grip-left-acc_ps_ent_z": 6.456559137160855,
                "grip-left-acc_ps_ent_xyz": 6.835545948852005,
                "grip-left-acc_ps_amp_x": 1.5259507790688076e-06,
                "grip-left-acc_ps_amp_y": 1.5357704796770122e-06,
                "grip-left-acc_ps_amp_z": 7.459480912075378e-06,
                "grip-left-acc_ps_amp_xyz": 1.6923471548579982e-06,
                "grip-left-gyr_sig_ene": 1.7657331228256226,
                "grip-left-gyr_ps_ene_x": 0.000823229146060811,
                "grip-left-gyr_ps_ene_y": 0.00035483072503783575,
                "grip-left-gyr_ps_ene_z": 0.00018793090034829874,
                "grip-left-gyr_ps_ene_xyz": 0.0006117804478655842,
                "grip-left-gyr_ps_peak_x": 0.78125,
                "grip-left-gyr_ps_peak_y": 2.96875,
                "grip-left-gyr_ps_peak_z": 0.234375,
                "grip-left-gyr_ps_peak_xyz": 0.15625,
                "grip-left-gyr_ps_ent_x": 6.7062211055024,
                "grip-left-gyr_ps_ent_y": 6.839050751336785,
                "grip-left-gyr_ps_ent_z": 6.5426977568292015,
                "grip-left-gyr_ps_ent_xyz": 6.24798787323569,
                "grip-left-gyr_ps_amp_x": 7.247346366057172e-05,
                "grip-left-gyr_ps_amp_y": 1.1988299775111955e-05,
                "grip-left-gyr_ps_amp_z": 3.584683145163581e-05,
                "grip-left-gyr_ps_amp_xyz": 0.00011552128853509203,
                "grip-left-applied_force-min": 0.23809523809523808,
                "grip-left-applied_force-max": 1.2833333015441895,
                "grip-left-applied_force-mean": 1.0054708781031942,
                "grip-left-applied_force-median": 0.9960318064022885,
                "grip-left-applied_force-std": 0.07527772583784806,
                "grip-left-applied_force-cv": 0.07486813141705144,
                "grip-left_low-applied_force-min": 0.8833333253860474,
                "grip-left_low-applied_force-max": 1.2833333015441895,
                "grip-left_low-applied_force-mean": 1.0313315693275784,
                "grip-left_low-applied_force-median": 1.0333333015441895,
                "grip-left_low-applied_force-std": 0.08908579175874573,
                "grip-left_low-applied_force-cv": 0.08637938991514542,
                "grip-left_medium-applied_force-min": 0.9230769597567045,
                "grip-left_medium-applied_force-max": 1.1602564041431134,
                "grip-left_medium-applied_force-mean": 1.0090680080372976,
                "grip-left_medium-applied_force-median": 0.9935897494679139,
                "grip-left_medium-applied_force-std": 0.04815155467390513,
                "grip-left_medium-applied_force-cv": 0.04771883985060929,
                "grip-left_high-applied_force-min": 0.23809523809523808,
                "grip-left_high-applied_force-max": 1.0793650717962355,
                "grip-left_high-applied_force-mean": 0.9898771865188714,
                "grip-left_high-applied_force-median": 0.9960318064022885,
                "grip-left_high-applied_force-std": 0.092947594413472,
                "grip-left_high-applied_force-cv": 0.09389810744133158,
                "grip-left_very_high-applied_force-min": 0.7212643787778658,
                "grip-left_very_high-applied_force-max": 1.1494252599518875,
                "grip-left_very_high-applied_force-mean": 0.9878262627040199,
                "grip-left_very_high-applied_force-median": 0.9942528792331011,
                "grip-left_very_high-applied_force-std": 0.05467231615588143,
                "grip-left_very_high-applied_force-cv": 0.055346084853245874,
                "grip-left_s4m-applied_force-std": 0.06808034395318462,
                "grip-left_s4m-applied_force-mean": 1.0063728554127758,
                "grip-left_s4m-applied_force-min": 0.7212643550588557,
                "grip-left_s4m-applied_force-median": 0.9960318064022885,
                "grip-left_s4m-applied_force-cv": 0.06764922522205813,
                "grip-left_s4m-applied_force-max": 1.2833333015441895,
                "grip-left_very_high_s4m-applied_force-std": 0.058286443615042344,
                "grip-left_very_high_s4m-applied_force-mean": 0.9817672187741965,
                "grip-left_very_high_s4m-applied_force-min": 0.7212643550588557,
                "grip-left_very_high_s4m-applied_force-median": 0.9942528792331011,
                "grip-left_very_high_s4m-applied_force-cv": 0.059368903850565466,
                "grip-left_very_high_s4m-applied_force-max": 1.1408045821089718,
                "grip-left_low_s4m-applied_force-std": 0.09271064398315053,
                "grip-left_low_s4m-applied_force-mean": 1.0397500117868186,
                "grip-left_low_s4m-applied_force-min": 0.8999999761581421,
                "grip-left_low_s4m-applied_force-median": 1.0166666507720947,
                "grip-left_low_s4m-applied_force-cv": 0.08916628317592087,
                "grip-left_low_s4m-applied_force-max": 1.2833333015441895,
                "grip-left_medium_s4m-applied_force-std": 0.06324415608868472,
                "grip-left_medium_s4m-applied_force-mean": 1.0021101412512592,
                "grip-left_medium_s4m-applied_force-min": 0.7692307974459868,
                "grip-left_medium_s4m-applied_force-median": 0.9871794989358278,
                "grip-left_medium_s4m-applied_force-cv": 0.0631109830000488,
                "grip-left_medium_s4m-applied_force-max": 1.160256446701066,
                "grip-left_high_s4m-applied_force-std": 0.022715551594267518,
                "grip-left_high_s4m-applied_force-mean": 1.0015361362853106,
                "grip-left_high_s4m-applied_force-min": 0.9523809956314903,
                "grip-left_high_s4m-applied_force-median": 0.9960318064022885,
                "grip-left_high_s4m-applied_force-cv": 0.022680710931229416,
                "grip-left_high_s4m-applied_force-max": 1.0793651208135115,
                "grip-left-rt-min": 0.28,
                "grip-left-rt-max": 0.38,
                "grip-left-rt-mean": 0.34428571428571425,
                "grip-left-rt-median": 0.34,
                "grip-left-rt-std": 0.035989416433697484,
                "grip-left_low_to_high-tt-mean": 0.52,
                "grip-left_low_to_very_high-tt-mean": 0.25,
                "grip-left_medium_to_high-tt-mean": 0.86,
                "grip-left_medium_to_very_high-tt-mean": 0.4,
                "grip-left_high_to_low-tt-mean": 0.41,
                "grip-left_very_high_to_low-tt-mean": 0.37,
                "grip-left_very_high_to_medium-tt-mean": 1.0,
                "grip-left_xpos-cv": 0.010241594,
                "grip-left_ypos-cv": 0.0018111612,
                "grip-left-pressure_overshoot-mean": -0.05000002043587821,
                "grip-left-pressure_overshoot-std": 0.6327482206388463,
                "grip-left-pressure_overshoot-median": 0.016666889190673828,
                "grip-left-pressure_overshoot-min": -1.1833332777023315,
                "grip-left-pressure_overshoot-max": 0.866666316986084,
                "grip-left-applied_force-90th": 1.0666667222976685,
                "grip-left_high_very_high-maf_diff": 0.0020510012531911315,
                "grip-left_s4m-applied_force-90th": 1.070512847552404,
                "grip-left_high_very_high_s4m-maf_diff": 0.019768917511114092,
                "grip-left_s2m-applied_force-mean": 0.9923923490373867,
                "grip-left_s2m-applied_force-std": 0.03354147836033341,
                "grip-left_s2m-applied_force-median": 0.9960318064022885,
                "grip-left_s2m-applied_force-min": 0.9166666865348816,
                "grip-left_s2m-applied_force-max": 1.0666667222976685,
                "grip-left_s2m-applied_force-cv": 0.03379860636054721,
                "grip-left_s2m-applied_force-90th": 1.0384615948919735,
                "grip-left_high_very_high_s2m-maf_diff": 0.006941095038262901,
                "grip-left_s5l-applied_force-mean": 1.0046602936362805,
                "grip-left_s5l-applied_force-std": 0.064969844157117,
                "grip-left_s5l-applied_force-median": 0.9960318064022885,
                "grip-left_s5l-applied_force-min": 0.7212643550588557,
                "grip-left_s5l-applied_force-max": 1.2833333015441895,
                "grip-left_s5l-applied_force-cv": 0.06466847009745384,
                "grip-left_s5l-applied_force-90th": 1.0666667222976685,
                "grip-left_high_very_high_s5l-maf_diff": 0.018499983621374727,
                "grip-left_low-applied_force-90th": 1.1333333253860474,
                "grip-left_low_s4m-applied_force-90th": 1.1833332777023315,
                "grip-left_low_s2m-applied_force-mean": 0.981375016272068,
                "grip-left_low_s2m-applied_force-std": 0.04464874198612398,
                "grip-left_low_s2m-applied_force-median": 0.9833333492279053,
                "grip-left_low_s2m-applied_force-min": 0.9166666865348816,
                "grip-left_low_s2m-applied_force-max": 1.0666667222976685,
                "grip-left_low_s2m-applied_force-cv": 0.04549610622423461,
                "grip-left_low_s2m-applied_force-90th": 1.0666667222976685,
                "grip-left_low_s5l-applied_force-mean": 1.0307500141263009,
                "grip-left_low_s5l-applied_force-std": 0.09282599356871332,
                "grip-left_low_s5l-applied_force-median": 1.0166666507720947,
                "grip-left_low_s5l-applied_force-min": 0.8833333253860474,
                "grip-left_low_s5l-applied_force-max": 1.2833333015441895,
                "grip-left_low_s5l-applied_force-cv": 0.09005674731656038,
                "grip-left_low_s5l-applied_force-90th": 1.151666641235352,
                "grip-left_medium-applied_force-90th": 1.07692309808449,
                "grip-left_medium_s4m-applied_force-90th": 1.0961538496807484,
                "grip-left_medium_s2m-applied_force-mean": 0.988821950490423,
                "grip-left_medium_s2m-applied_force-std": 0.03533821758438227,
                "grip-left_medium_s2m-applied_force-median": 0.9871794989358278,
                "grip-left_medium_s2m-applied_force-min": 0.9230769936149669,
                "grip-left_medium_s2m-applied_force-max": 1.0512820959561457,
                "grip-left_medium_s2m-applied_force-cv": 0.03573769531193728,
                "grip-left_medium_s2m-applied_force-90th": 1.0384615948919735,
                "grip-left_medium_s5l-applied_force-mean": 1.00413919002719,
                "grip-left_medium_s5l-applied_force-std": 0.057423514474798834,
                "grip-left_medium_s5l-applied_force-median": 0.9935897494679139,
                "grip-left_medium_s5l-applied_force-min": 0.7692307974459868,
                "grip-left_medium_s5l-applied_force-max": 1.160256446701066,
                "grip-left_medium_s5l-applied_force-cv": 0.05718680741187277,
                "grip-left_medium_s5l-applied_force-90th": 1.07692309808449,
                "grip-left_high-applied_force-90th": 1.0119048078584583,
                "grip-left_high_s4m-applied_force-90th": 1.0277778093146284,
                "grip-left_high_s2m-applied_force-mean": 0.996219808846003,
                "grip-left_high_s2m-applied_force-std": 0.02244615918417219,
                "grip-left_high_s2m-applied_force-median": 0.9960318064022885,
                "grip-left_high_s2m-applied_force-min": 0.9523809956314903,
                "grip-left_high_s2m-applied_force-max": 1.0595238122269681,
                "grip-left_high_s2m-applied_force-cv": 0.022531331925805892,
                "grip-left_high_s2m-applied_force-90th": 1.0119048078584583,
                "grip-left_high_s5l-applied_force-mean": 1.0008275436078984,
                "grip-left_high_s5l-applied_force-std": 0.021426565292017967,
                "grip-left_high_s5l-applied_force-median": 0.9960318064022885,
                "grip-left_high_s5l-applied_force-min": 0.9523809956314903,
                "grip-left_high_s5l-applied_force-max": 1.0793651208135115,
                "grip-left_high_s5l-applied_force-cv": 0.02140884853625932,
                "grip-left_high_s5l-applied_force-90th": 1.0158731149888318,
                "grip-left_very_high-applied_force-90th": 1.0229884830675953,
                "grip-left_very_high_s4m-applied_force-90th": 1.0229884830675953,
                "grip-left_very_high_s2m-applied_force-mean": 1.003160903884266,
                "grip-left_very_high_s2m-applied_force-std": 0.022313313310339318,
                "grip-left_very_high_s2m-applied_force-median": 1.0,
                "grip-left_very_high_s2m-applied_force-min": 0.9626436739085075,
                "grip-left_very_high_s2m-applied_force-max": 1.054597688392189,
                "grip-left_very_high_s2m-applied_force-cv": 0.022243005308462054,
                "grip-left_very_high_s2m-applied_force-90th": 1.054597688392189,
                "grip-left_very_high_s5l-applied_force-mean": 0.9823275599865237,
                "grip-left_very_high_s5l-applied_force-std": 0.05239160036482425,
                "grip-left_very_high_s5l-applied_force-median": 0.9942528792331011,
                "grip-left_very_high_s5l-applied_force-min": 0.7212643550588557,
                "grip-left_very_high_s5l-applied_force-max": 1.1408045821089718,
                "grip-left_very_high_s5l-applied_force-cv": 0.05333414484018243,
                "grip-left_very_high_s5l-applied_force-90th": 1.0229884830675953,
            },
        ),
    ],
)
def test_grip_process(processed_grip_reading_ads, level_id, expected):
    """Unit test to ensure the GRIP measures are well computed."""
    assert_level_values(processed_grip_reading_ads, level_id, expected)


@pytest.mark.parametrize(
    "level_id,expected",
    [
        (
            "right-all",
            {
                # FIXME provide regression values
            },
        ),
        (
            "left-all",
            {
                # FIXME provide regression values
            },
        ),
    ],
)
@pytest.mark.skip
def test_grip_process_new_format(
    example_reading_processed_sbt_utt_new_format, level_id, expected
):
    """Unit test to ensure the GRIP measures are well computed."""
    # FIXME: replace with actual GRIP and not SBT-UTT test case...
    assert_level_values(
        example_reading_processed_sbt_utt_new_format, level_id, expected
    )


def test_grip_overshoot_process(processed_grip_reading_ads):
    """Unit test to ensure GRIP overshoot measures are well computed."""
    res = processed_grip_reading_ads
    expected_values = [
        0.3000001907348633,
        0.25,
        -0.9499999284744263,
        0.08333349227905273,
        -0.9833332300186157,
        0.23333311080932617,
        0.25,
    ]
    assert (
        res.get_level("right-all")
        .get_raw_data_set("right-overshoot")
        .data["pressure-overshoot"]
        .values
        == expected_values
    ).all()
