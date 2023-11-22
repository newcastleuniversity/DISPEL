"""Test cases for :mod:`dispel.providers.ads.tasks.sbt_utt`."""
from copy import deepcopy

import pytest

from dispel.data.core import Reading
from dispel.data.flags import Flag
from dispel.data.measures import MeasureSet
from dispel.processing import process
from dispel.providers.ads.tasks.sbt_utt import process_sbt_utt
from dispel.providers.generic.activity.orientation import (
    PhoneOrientationTransform,
    UprightPortraitModeFlagger,
    UpsideDownPortraitModeFlagger,
)
from dispel.providers.generic.tasks.sbt_utt.sbt_synth import *
from dispel.signal.orientation import PhoneOrientation
from tests.conftest import resource_path
from tests.processing.helper import (
    assert_dict_values,
    assert_level_values,
    read_results,
)

EXAMPLE_SBTUTT_PATH = resource_path(
    "SBT-UTT/expected/expected_sbtutt_process.json", "providers.ads"
)
EXAMPLE_SBTUTT_UPSAMPLE_PATH = resource_path(
    "SBT-UTT/expected/expected_sbt_upsampling.json", "providers.ads"
)
EXAMPLE_SBTUTT_NOMOTION_PATH = resource_path(
    "SBT-UTT/expected/expected_sbt_nomotion.json", "providers.ads"
)
EXAMPLE_SBTUTT_FLAGGED_PATH = resource_path(
    "SBT-UTT/expected/expected_sbt_segment_flag.json", "providers.ads"
)
EXAMPLE_SBTUTT_FIXED_PATH = resource_path(
    "SBT-UTT/expected/expected_sbt_motion_fixed.json", "providers.ads"
)
EXAMPLE_UTT_UPSAMPLED_PATH = resource_path(
    "SBT-UTT/expected/expected_utt_upsample.json", "providers.ads"
)
EXAMPLE_UTT_PROCESS_PATH = resource_path(
    "SBT-UTT/expected/expected_utt_process.json", "providers.ads"
)


def test_sbt_process_synth_ellipse_area_excursion(example_reading_sbt_utt):
    """Ensure that the area and perimeter of a ellipse with axes AB is correct.

    To test the validity of the ellipse area and total excursion, we compute the area
    and arc length of a AB axes ellipse. As an input, we create a synthetic ellipse
    in `CreateSyntheticEllipse`.
    """
    reading = deepcopy(example_reading_sbt_utt)

    steps_sbt_synthetic_ellipse = BalanceSyntheticProcessingSteps(
        CreateSyntheticEllipse()
    )
    process(reading, steps_sbt_synthetic_ellipse)

    expected_totex = 9.68845 * 1e3
    expected_ea = 1 * 2 * np.pi * 1e6

    actual_totex = reading.get_level("sbt").measure_set.get("sbt-full-totex").value
    actual_ea = reading.get_level("sbt").measure_set.get("sbt-full-ea").value

    assert pytest.approx(expected_totex, 0.1) == actual_totex
    assert pytest.approx(expected_ea, 0.1) == actual_ea


def test_sbt_process_synth_ellipse_rotated(example_reading_sbt_utt):
    """Test the area and perimeter of a rotated ellipse.

    To test the validity of the ellipse area and total excursion, we compute the area
    and arc length of a AB axa synthetic ellipse in `CreateSyntheticEllipseRotated`.
    The value of area and arc length is preserved because data point cloud is just
    rotated.
    """
    reading = deepcopy(example_reading_sbt_utt)

    steps_sbt_synthetic_ellipse_rotated = BalanceSyntheticProcessingSteps(
        CreateSyntheticEllipseRotated()
    )
    process(reading, steps_sbt_synthetic_ellipse_rotated)

    expected_totex = 9.68845 * 1e3
    expected_ea = 1 * 2 * np.pi * 1e6

    actual_totex = reading.get_level("sbt").measure_set.get("sbt-full-totex").value
    actual_ea = reading.get_level("sbt").measure_set.get("sbt-full-ea").value

    assert pytest.approx(expected_totex, 0.1) == actual_totex
    assert pytest.approx(expected_ea, 0.1) == actual_ea


def test_sbt_process_synth_ellipse_doubled(example_reading_sbt_utt):
    """Test the area and perimeter of a doubled ellipse.

    To test the validity of the ellipse area and total excursion, we compute the area
    and arc length of a AB axes ellipse. As an input, we create a synthetic ellipse in
    `CreateSyntheticEllipse`. The value of area and arc length is doubled because data
    point cloud is just expanded to double.
    """
    reading = deepcopy(example_reading_sbt_utt)

    steps_sbt_synthetic_ellipse_doubled_rotated = BalanceSyntheticProcessingSteps(
        CreateSyntheticEllipseDoubledRotated()
    )
    process(reading, steps_sbt_synthetic_ellipse_doubled_rotated)

    expected_totex = 9.68845 * 2 * 1e3
    expected_ea = (1 * 2) * (2 * 2) * np.pi * 1e6

    actual_totex = reading.get_level("sbt").measure_set.get("sbt-full-totex").value
    actual_ea = reading.get_level("sbt").measure_set.get("sbt-full-ea").value

    assert pytest.approx(expected_totex, 0.1) == actual_totex
    assert pytest.approx(expected_ea, 0.1) == actual_ea


def test_sbt_process_synth_ellipse_doubled_rot(example_reading_sbt_utt):
    """Test the area and perimeter of a rotated and doubled ellipse.

    To test the validity of the ellipse area and total excursion, we compute the area
    and arc length of a AB axes ellipse. As an input, we create a synthetic ellipse in
    `CreateSyntheticEllipse`. The value of area and arc length is doubled because data
    point cloud is just expanded to double.
    """
    reading = deepcopy(example_reading_sbt_utt)

    steps_sbt_synthetic_ellipse_doubled = BalanceSyntheticProcessingSteps(
        CreateSyntheticEllipseDoubled()
    )
    process(reading, steps_sbt_synthetic_ellipse_doubled)

    expected_totex = 9.68845 * 2 * 1e3
    expected_ea = (1 * 2) * (2 * 2) * np.pi * 1e6

    actual_totex = reading.get_level("sbt").measure_set.get("sbt-full-totex").value
    actual_ea = reading.get_level("sbt").measure_set.get("sbt-full-ea").value

    assert pytest.approx(expected_totex, 0.1) == actual_totex
    assert pytest.approx(expected_ea, 0.1) == actual_ea


def test_sbt_process_synth_ellipse_same_axes(example_reading_sbt_utt):
    """Ensure that the area and perimeter of a ellipse with same axes length 1.

    To test the validity of the ellipse area and total excursion, we compute the area
    and arc length of a AB = 1 axes ellipse. As an input, we create a synthetic ellipse
    in `CreateSyntheticEllipse`. The value of area and arc length is same as for circle
    because data point cloud resembles a circle.
    """
    reading = deepcopy(example_reading_sbt_utt)

    steps_sbt_synthetic_ellipse_same_axes = BalanceSyntheticProcessingSteps(
        CreateSyntheticEllipseSameAxes()
    )
    process(reading, steps_sbt_synthetic_ellipse_same_axes)

    expected_ca = 1**2 * np.pi * 1e6
    expected_ea = 1 * 1 * np.pi * 1e6
    expected_totex = 2 * np.pi * 1 * 1e3
    expected_jerk = expected_totex / 30

    actual_ca = reading.get_level("sbt").measure_set.get("sbt-full-ca").value
    actual_ea = reading.get_level("sbt").measure_set.get("sbt-full-ea").value
    actual_totex = reading.get_level("sbt").measure_set.get("sbt-full-totex").value
    actual_jerk = reading.get_level("sbt").measure_set.get("sbt-full-jerk").value

    assert pytest.approx(expected_ca, 0.1) == actual_ca
    assert pytest.approx(expected_ea, 0.1) == actual_ea
    assert pytest.approx(expected_totex, 0.1) == actual_totex
    assert pytest.approx(expected_jerk, 0.1) == actual_jerk


def test_sbt_process_synth_ellipse_stretched(example_reading_sbt_utt):
    """Test the area and perimeter of a stretched ellipse.

    To test the validity of the ellipse area and total excursion, we compute the area
    and arc length of an AB = 1 axes ellipse. As an input, we create a synthetic ellipse
    in `CreateSyntheticEllipse`. The value of area and arc length is still correct as
    it's just a stretched ellipse.
    """
    reading = deepcopy(example_reading_sbt_utt)

    steps_sbt_synthetic_ellipse_stretched = BalanceSyntheticProcessingSteps(
        CreateSyntheticEllipseStretched()
    )
    process(reading, steps_sbt_synthetic_ellipse_stretched)

    expected_totex = 40.63927 * 1e3
    expected_ea = 1 * 10 * np.pi * 1e6

    actual_totex = reading.get_level("sbt").measure_set.get("sbt-full-totex").value
    actual_ea = reading.get_level("sbt").measure_set.get("sbt-full-ea").value

    assert pytest.approx(expected_totex, 0.1) == actual_totex
    assert pytest.approx(expected_ea, 0.1) == actual_ea


def test_sbt_process_synth_ellipse_outliers(example_reading_sbt_utt):
    """Test the area and perimeter of an ellipse with outliers.

    To test the validity of the ellipse area and total excursion, we compute the area
    and arc length of an AB = 1 axes ellipse. As an input, we create a synthetic ellipse
    in `CreateSyntheticEllipse`. The value of area and arc length is preserved because
    data point cloud considered matches the 95-percentile of data.
    """
    reading = deepcopy(example_reading_sbt_utt)

    steps_sbt_synthetic_ellipse_outliers = BalanceSyntheticProcessingSteps(
        CreateSyntheticEllipseOutliers()
    )
    process(reading, steps_sbt_synthetic_ellipse_outliers)

    expected_ea = 1 * 2 * np.pi * 1e6

    actual_ea = reading.get_level("sbt").measure_set.get("sbt-full-ea").value

    assert pytest.approx(expected_ea, 0.1) == actual_ea


def test_sbt_process_synth_ellipse_outliers_doubled(example_reading_sbt_utt):
    """Test the area and perimeter of a doubled ellipse with outliers.

    To test the validity of the ellipse area and total excursion, we compute the area
    and arc length of an AB = 1 axes ellipse. As an input, we create a synthetic ellipse
    in `CreateSyntheticEllipse`. The value of area and arc length is just doubled
    because data point cloud considered matches the 95-percentile of data
    """
    reading = deepcopy(example_reading_sbt_utt)

    steps_sbt_synthetic_ellipse_outliers_doubled = BalanceSyntheticProcessingSteps(
        CreateSyntheticEllipseOutliersDoubled()
    )
    process(reading, steps_sbt_synthetic_ellipse_outliers_doubled)

    expected_ea = (1 * 2) * (2 * 2) * np.pi * 1e6

    actual_ea = reading.get_level("sbt").measure_set.get("sbt-full-ea").value

    assert pytest.approx(expected_ea, 0.1) == actual_ea


def test_sbt_process_synth_ellipse_downsampled(example_reading_sbt_utt):
    """Test the area and perimeter of a downsampled ellipse.

    To test the validity of the ellipse area and total excursion, we compute the area
    and arc length of a AB = 1 axes ellipse. As an input, we create a synthetic ellipse
    in `CreateSyntheticEllipse`. The value of area and arc length is preserved because
    data point cloud considered matches the 95-percentile of data, regardless of the
    data sample size.
    """
    reading = deepcopy(example_reading_sbt_utt)

    steps_sbt_synthetic_ellipse_downsampled = BalanceSyntheticProcessingSteps(
        CreateSyntheticEllipseDownsampled()
    )
    process(reading, steps_sbt_synthetic_ellipse_downsampled)

    expected_ea = 1 * 2 * np.pi * 1e6

    actual_ea = reading.get_level("sbt").measure_set.get("sbt-full-ea").value

    assert pytest.approx(expected_ea, 0.1) == actual_ea


def test_sbt_process_synth_ellipse_upsampled(example_reading_sbt_utt):
    """Test the area and perimeter of an upsampled ellipse.

    To test the validity of the ellipse area and total excursion, we compute the area
    and arc length of a AB = 1 axes ellipse. As an input, we create a synthetic ellipse
    in `CreateSyntheticEllipse`. The value of area and arc length is preserved because
    data point cloud considered matches the 95-percentile of data, regardless of the
    data sample size.
    """
    reading = deepcopy(example_reading_sbt_utt)

    steps_sbt_synthetic_ellipse_upsampled = BalanceSyntheticProcessingSteps(
        CreateSyntheticEllipseUpsampled()
    )
    process(reading, steps_sbt_synthetic_ellipse_upsampled)

    expected_ea = 1 * 2 * np.pi * 1e6

    actual_ea = reading.get_level("sbt").measure_set.get("sbt-full-ea").value

    assert pytest.approx(expected_ea, 0.1) == actual_ea


def test_sbt_process_first_less_after5s_measures(example_reading_sbt_utt):
    """Ensure that the measures computed are always smaller in first 5 seconds.

    To make sure that the SBT BoutModalities are correctly computed, we want to ensure
    that the after5s value is always bigger than the value of the first5s. This is
    applicable to the spatio-temporal measures, as the amplitude on the synthetic
    ellipse is always the same (inc. same axes, for this synthetic dataset) and
    generated chronologically (i.e., temporally, there will be less path and area in the
    first 5 seconds). The jerk is difficult to asser (i.e., there is less acceleration,
    e.g., less area, but also less time, which increases the value of jerk) -> leave
    out. The circle area is also difficult to assert: it approximates the value based on
    statistical estimates of the radius (i.e, as soon as there is points, there is a
    radius, which yields equal values of circle area).
    """
    reading = deepcopy(example_reading_sbt_utt)

    steps_sbt_synthetic_ellipse_same_axes = BalanceSyntheticProcessingSteps(
        CreateSyntheticEllipseSameAxes()
    )
    process(reading, steps_sbt_synthetic_ellipse_same_axes)

    actual_ea_full = reading.get_level("sbt").measure_set.get("sbt-full-ea").value
    actual_totex_full = reading.get_level("sbt").measure_set.get("sbt-full-totex").value

    actual_ea_first5s = reading.get_level("sbt").measure_set.get("sbt-first5s-ea").value
    actual_totex_first5s = (
        reading.get_level("sbt").measure_set.get("sbt-first5s-totex").value
    )

    actual_ea_after5s = reading.get_level("sbt").measure_set.get("sbt-after5s-ea").value
    actual_totex_after5s = (
        reading.get_level("sbt").measure_set.get("sbt-after5s-totex").value
    )

    assert actual_ea_full > actual_ea_after5s > actual_ea_first5s
    assert actual_totex_full > actual_totex_after5s > actual_totex_first5s


@pytest.mark.skip(
    reason="We have deprecated `sbt-sway` and  `sbt-sway_fixed_normalised`"
)
def test_sbtutt_process_segment_sway_fix(example_reading_sbt_utt_motion_fixed):
    """Unit test to ensure the SBT-UTT sway path is fixed after segmentation.

    This test ensures that the sway value fixed is smaller than original sway value.
    """
    reading = deepcopy(example_reading_sbt_utt_motion_fixed)
    process_sbt_utt(reading)

    sway_original = reading.get_level("sbt").measure_set.get_raw_value("sbt-sway")

    sway_fixed = reading.get_level("sbt").measure_set.get_raw_value(
        "sbt-sway_fixed_normalised"
    )

    assert sway_original > sway_fixed


def test_sbt_post_adjustment_excessive_motion(
    example_reading_sbt_utt_segmentation_flagged,
):
    """Test too short sway flag for SBT task."""
    reading = deepcopy(example_reading_sbt_utt_segmentation_flagged)
    process_sbt_utt(reading)

    sbt_invs = reading.get_level("sbt").get_flags()

    expected = [
        Flag(
            "sbt-behavioral-deviation-excessive_motion",
            "100.0% excessive motion portion detected (exceeds 0.0% accepted)",
        ),
        Flag(
            "sbt-behavioral-deviation-excessive_post_adjustment_motion",
            "100.0% excessive motion post-adjustment portion detected "
            "(exceeds 0.0% accepted)",
        ),
        Flag(
            "sbt-technical-invalidation-freq_low_40hz",
            "The median frequency is lower than 40Hz.",
        ),
    ]

    assert set(sbt_invs) == set(expected)


@pytest.mark.skip(reason="We have replaced `LandscapeModeFlagger` by `NonBeltDetected`")
def test_sbt_phone_orientation_flag(example_reading_processed_sbt_utt_portrait):
    """Test phone orientation flag for SBT task."""
    reading = deepcopy(example_reading_processed_sbt_utt_portrait)
    sbt_invs = reading.get_level("sbt").get_flags()
    expected = Flag(
        "sbt-behavioral-deviation-lrmllmo",
        "The phone has not been kept at a landscape right mode (0.0%) or "
        "landscape left mode (0.0%) for more than 90.0% of the test.",
    )
    assert sbt_invs == [expected]

    flag_steps = ProcessingStepGroup(
        [
            UpsideDownPortraitModeFlagger(
                gravity_data_set_id="acc",
                pitch_freedom=70,
                yaw_freedom=15,
                acceptance_threshold=0.8,
            ),
            UprightPortraitModeFlagger(
                gravity_data_set_id="acc",
                pitch_freedom=70,
                yaw_freedom=15,
                acceptance_threshold=0.8,
            ),
        ],
        task_name="utt",
        level_filter="utt",
    )

    process(reading, flag_steps)

    sbt_invs = reading.get_level("utt").get_flags()
    expected = [
        Flag(
            "utt-behavioral-deviation-udpmo",
            "The phone has not been kept at a portrait upside down mode "
            "(63.5%) for more than 80.0% of the test.",
        ),
        Flag(
            "utt-behavioral-deviation-upmo",
            "The phone has not been kept at a portrait upright mode (0.5%) "
            "for more than 80.0% of the test.",
        ),
    ]
    assert sbt_invs == expected


def test_sbt_phone_placement_flag(example_reading_processed_sbt_utt_portrait):
    """Test phone placement flag for SBT task."""
    reading = deepcopy(example_reading_processed_sbt_utt_portrait)
    sbt_invs = reading.get_level("sbt").get_flags()
    expected = [
        Flag(
            "sbt-behavioral-deviation-non_belt_greater_than_0_perc",
            "100.0% non-belt portion detected at ['pants'] (exceeds 0% accepted)",
        ),
        Flag(
            "sbt-behavioral-invalidation-non_belt_post_adjustment_greater_than_0_perc",
            "100.0% non-belt post-adjustment portion detected at ['pants'] "
            "(exceeds 0% accepted)",
        ),
    ]
    assert sbt_invs == expected


def test_sbt_flag_synth_placement_complex(example_reading_sbt_utt_portrait):
    """Ensure that placement flag all if beyond adjustment.

    Test that it flags a full reading if the placement flag
    crosses the adjustment end threshold.
    """
    reading = deepcopy(example_reading_sbt_utt_portrait)
    process(reading, SBTSyntheticComplexProcessing([CreateSyntheticPlacementBouts()]))

    sbt_invs = reading.get_level("sbt").get_flags()
    expected = [
        Flag(
            "sbt-behavioral-deviation-non_belt_greater_than_0_perc",
            "66.77751912204855% non-belt portion detected "
            "at ['pants', 'handheld', 'belt', 'pants'] (exceeds 0% accepted)",
        ),
        Flag(
            "sbt-behavioral-invalidation-non_belt_post_adjustment_greater_than_0_perc",
            "60.16746411483253% non-belt post-adjustment portion detected "
            "at ['handheld', 'belt', 'pants'] (exceeds 0% accepted)",
        ),
    ]
    assert sbt_invs == expected


def test_sbt_flag_synth_placement_adjust(example_reading_sbt_utt_portrait):
    """Ensure that it does not flag all if adjustment.

    Test that it does not flag a full reading if the placement
    flag does not cross the adjustment end threshold.
    """
    reading = deepcopy(example_reading_sbt_utt_portrait)
    process(
        reading, SBTSyntheticComplexProcessing([CreateSyntheticPlacementBoutsAdjust()])
    )

    sbt_invs = reading.get_level("sbt").get_flags()
    expected = [
        Flag(
            "sbt-behavioral-deviation-non_belt_greater_than_0_perc",
            "16.627868307283013% non-belt portion detected "
            "at ['pants', 'belt', 'belt', 'belt'] (exceeds 0% accepted)",
        ),
    ]
    assert sbt_invs == expected


def test_utt_phone_orientation_transformation(
    example_reading_processed_sbt_utt_portrait,
):
    """Test phone orientation transformation for UTT task."""
    reading = deepcopy(example_reading_processed_sbt_utt_portrait)
    steps = ProcessingStepGroup(
        [
            PhoneOrientationTransform(
                gravity_data_set_id="acc",
                pitch_freedom=70,
                yaw_freedom=15,
                orientation_mode=(
                    PhoneOrientation.PORTRAIT_UPSIDE_DOWN,
                    PhoneOrientation.LANDSCAPE_LEFT,
                    PhoneOrientation.PORTRAIT_UPRIGHT,
                    PhoneOrientation.FACE_UP,
                ),
            ),
        ],
        task_name="utt",
        level_filter="utt",
    )

    process(reading, steps)

    level = reading.get_level("utt")
    assert level.has_raw_data_set("phone-orientation")

    raw_data_set = level.get_raw_data_set("phone-orientation")
    expected = ["portrait_upside_down", "landscape_left", "portrait_upright", "face_up"]
    assert list(raw_data_set.data.columns) == expected


def test_process_utt(example_reading_processed_sbt_utt):
    """Test end-to-end processing of U-turn test."""
    assert isinstance(example_reading_processed_sbt_utt, Reading)

    measure_set = example_reading_processed_sbt_utt.get_level("utt").measure_set
    assert isinstance(measure_set, MeasureSet)

    expected = read_results(EXAMPLE_UTT_PROCESS_PATH, True)

    assert_dict_values(measure_set, expected, 1e-5)

    for measure_id in expected:
        definition = measure_set.get_definition(measure_id)

        if ("_ts_" in measure_id) or ("_turn_speed_" in measure_id):
            assert definition.unit == "rad/s"
        elif "dur" in measure_id:
            assert definition.unit == "s"
        elif measure_id == "utt-walking_speed":
            assert definition.unit == "step/s"


def test_sbtutt_process(example_reading_processed_sbt_utt):
    """Unit test to ensure the SBT-UTT measures are well computed."""
    expected = read_results(EXAMPLE_SBTUTT_PATH, True)
    assert_level_values(example_reading_processed_sbt_utt, "sbt", expected)


def test_sbtutt_process_upsampling(example_reading_sbt_utt_up_sample2):
    """Unit test to ensure the SBT-UTT measures are well computed.

    This test ensures treatments can run even when the user has not turned.
    """
    reading = deepcopy(example_reading_sbt_utt_up_sample2)
    process_sbt_utt(reading)

    expected = read_results(EXAMPLE_SBTUTT_UPSAMPLE_PATH, True)

    assert_level_values(reading, "sbt", expected, relative_error=1e-4)


def test_sbtutt_process_segment_feat_no_motion(
    example_reading_sbt_utt_segmentation_no_motion,
):
    """Ensure measures are ok after segmentation.

    This test ensures measure values are the same even in a sample where there is no
    excessive motion to segment. The dummy sample added contains no invalid segment.
    """
    reading = deepcopy(example_reading_sbt_utt_segmentation_no_motion)
    process_sbt_utt(reading)

    expected = read_results(EXAMPLE_SBTUTT_NOMOTION_PATH, True)

    assert_level_values(reading, "sbt", expected)


def test_sbtutt_process_segment_feat_invalid(
    example_reading_sbt_utt_segmentation_flagged,
):
    """Ensure measures are ok when an flag is present.

    This test ensures measure values are the same even in a sample where there is an
    excessive motion to segment which has a too wide coverage. The dummy sample added
    contains too long invalid segment.
    """
    reading = deepcopy(example_reading_sbt_utt_segmentation_flagged)
    process_sbt_utt(reading)

    expected = read_results(EXAMPLE_SBTUTT_FLAGGED_PATH, True)

    assert_level_values(reading, "sbt", expected)


def test_sbtutt_process_segment_feat_fix(example_reading_sbt_utt_motion_fixed):
    """Ensure measures are ok after segmentation when excessive motion.

    This test ensures measure values are the same even in a sample where there are two
    excessive motions to segment. The dummy sample added contains two invalid segment.
    """
    reading = deepcopy(example_reading_sbt_utt_motion_fixed)
    process_sbt_utt(reading)

    expected = read_results(EXAMPLE_SBTUTT_FIXED_PATH, True)

    assert_level_values(reading, "sbt", expected)


def test_sbt_synth_circle_spatiotemporalmeasures(example_reading_sbt_utt):
    """Ensure that the area and perimeter of a radius 1 circle is correct.

    To test the validity of the circle area and total excursion, we compute the area and
    arc length of a unit radius circle. As an input, we create a synthetic circle in
    `CreateSyntheticCircle`. Also the area of the ellipse shall match the area of the
    circle or radius 1.
    """
    reading = deepcopy(example_reading_sbt_utt)

    steps_sbt_synthetic_circle = BalanceSyntheticProcessingSteps(
        CreateSyntheticCircle()
    )
    process(reading, steps_sbt_synthetic_circle)

    expected_ca = 1**2 * np.pi * 1e6
    expected_ea = 1 * 1 * np.pi * 1e6
    expected_totex = 2 * np.pi * 1 * 1e3
    expected_jerk = expected_totex / 30

    actual_ca = reading.get_level("sbt").measure_set.get("sbt-full-ca").value
    actual_ea = reading.get_level("sbt").measure_set.get("sbt-full-ea").value
    actual_totex = reading.get_level("sbt").measure_set.get("sbt-full-totex").value
    actual_jerk = reading.get_level("sbt").measure_set.get("sbt-full-jerk").value

    assert pytest.approx(expected_ca, 0.1) == actual_ca
    assert pytest.approx(expected_ea, 0.1) == actual_ea
    assert pytest.approx(expected_totex, 0.1) == actual_totex
    assert pytest.approx(expected_jerk, 0.1) == actual_jerk


def test_sbtutt_process_utt_upsample(example_reading_sbt_utt_up_sample):
    """Unit test to ensure the SBT-UTT measures are well computed.

    This test ensures treatments can run even when the user has not turned.
    """  # noqa: DAR101
    reading = deepcopy(example_reading_sbt_utt_up_sample)
    process_sbt_utt(reading)

    expected = read_results(EXAMPLE_UTT_UPSAMPLED_PATH, True)

    assert_level_values(reading, "utt", expected)


def test_sbt_no_belt_low_frequency_flags(example_reading_sbt_utt):
    """Test phone orientation flag for SBT task."""
    reading = deepcopy(example_reading_sbt_utt)
    process_sbt_utt(reading)

    sbt_invs = reading.get_level("sbt").get_flags()
    expected = [
        Flag(
            "sbt-technical-invalidation-freq_low_40hz",
            "The median frequency is lower than 40Hz.",
        ),
    ]
    assert set(sbt_invs) == set(expected)
