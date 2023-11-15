"""Tests for :mod:`dispel.processing.preprocessing.core`."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from dispel.data.raw import RawDataSet
from dispel.processing import process
from dispel.processing.level import LevelIdFilter
from dispel.processing.modalities import LimbModality, SensorModality
from dispel.providers.ads import PROVIDER_ID
from dispel.providers.ads.io import read_ads
from dispel.providers.generic.preprocessing import (
    AmplitudeRangeFilter,
    Detrend,
    FilterPhysiologicalNoise,
    FilterSensorNoise,
    PreprocessingSteps,
    Resample,
    RotateFrame,
)
from dispel.providers.generic.sensor import FREQ_20HZ, SetTimestampIndex
from tests.providers import resource_path

# Flat signal execution lying on table
EXAMPLE_PATH_FLAT = resource_path(
    PROVIDER_ID, "SBT-UTT/5ed5-40b8-bb35-ce5975677185_2020-12-1_11-17-39.json"
)

EXAMPLE_PATH = resource_path(
    PROVIDER_ID, "SBT-UTT/6b69efb2-587b-4da8-b21f-5d40279d6369.json"
)


@pytest.fixture
def resample_sbt():
    """Get a fixture of the resample sbt level."""
    reading = read_ads(EXAMPLE_PATH_FLAT)
    columns = [
        "userAccelerationX",
        "userAccelerationY",
        "userAccelerationZ",
        "gravityX",
        "gravityY",
        "gravityZ",
    ]

    resample_steps = [
        SetTimestampIndex(
            "accelerometer",
            columns,
        ),
        Resample(
            data_set_id="accelerometer_ts",
            aggregations=["mean", "ffill"],
            columns=columns,
            freq=FREQ_20HZ,
            level_filter="sbt",
        ),
    ]

    return process(reading, resample_steps).get_reading()


def test_resample_preprocessing_step(resample_sbt):
    """Test :class:`dispel.processing.preprocessing.ResamplePreprocessingStep`."""
    acc_res = (
        resample_sbt.get_level("sbt")
        .get_raw_data_set("accelerometer_ts_resampled")
        .data
    )
    time_index_diff = np.diff(acc_res.index.values)
    resample_rate = int(1e9 / time_index_diff.mean().astype(int))

    assert resample_rate == 20.0


def test_filter_svgf_preprocessing_step(resample_sbt):
    """:class:`dispel.processing.preprocessing.FilterSVGPreprocessingStep`."""
    reading = deepcopy(resample_sbt)
    filter_svg_steps = FilterSensorNoise(
        level_filter="sbt", data_set_id="accelerometer", columns=["userAccelerationX"]
    )

    process(reading, filter_svg_steps)

    acc_filter = reading.get_level("sbt").get_raw_data_set("accelerometer_svgf").data

    assert isinstance(acc_filter, pd.DataFrame)


def test_filter_bhpf_preprocessing_step(resample_sbt):
    """:class:`dispel.processing.preprocessing.FilterBHPPreprocessingStep`."""
    reading = deepcopy(resample_sbt)
    filter_svg_steps = FilterPhysiologicalNoise(
        level_filter="sbt",
        data_set_id="accelerometer_ts_resampled",
        columns=["userAccelerationX"],
    )

    process(reading, filter_svg_steps)

    assert isinstance(
        reading.get_level("sbt").get_raw_data_set("accelerometer_ts_resampled_bhpf"),
        RawDataSet,
    )

    with pytest.raises(ValueError):
        reading.get_level("utt").get_raw_data_set("accelerometer_ts_resampled_bhpf")


def test_detrend_preprocessing_step(resample_sbt):
    """:class:`dispel.processing.preprocessing.DetrendPreprocessingStep`."""
    reading = deepcopy(resample_sbt)
    detrend_step = Detrend(
        level_filter="sbt",
        data_set_id="accelerometer_ts_resampled",
        columns=["userAccelerationX"],
    )

    process(reading, detrend_step)

    acc_detrend = (
        reading.get_level("sbt")
        .get_raw_data_set("accelerometer_ts_resampled_detrend")
        .data
    )

    assert isinstance(acc_detrend, pd.DataFrame)


def test_amplitude_filter(resample_sbt):
    """:class:`dispel.processing.preprocessing.AmplitudeRangeFilter`."""
    reading = deepcopy(resample_sbt)
    level_filter = LevelIdFilter("sbt")
    resample_sbt.get_level("sbt").get_raw_data_set(
        "accelerometer_ts_resampled"
    ).data = (
        resample_sbt.get_level("sbt")
        .get_raw_data_set("accelerometer_ts_resampled")
        .data
        * 100
    )
    level_filter &= AmplitudeRangeFilter(
        "accelerometer_ts_resampled", min_amplitude=0, max_amplitude=0
    )

    filt_svg_steps = FilterSensorNoise(
        level_filter=level_filter,
        data_set_id="accelerometer_ts_resampled",
        columns=["userAccelerationX"],
    )

    process(reading, filt_svg_steps)

    # Reading of phone put on table
    with pytest.raises(ValueError):
        _ = (
            reading.get_level("sbt")
            .get_raw_data_set("accelerometer_ts_resampled_svgf")
            .data
        )


def test_rotate_frame_preprocessing_step(resample_sbt):
    """:class:`dispel.processing.preprocessing.RotateFramePreprocessingStep`."""
    reading = deepcopy(resample_sbt)
    rotation_step = RotateFrame(
        level_filter="sbt",
        frame=(-1, 0, 0),
        data_set_id="accelerometer_ts_resampled",
        gravity_data_set_id="accelerometer_ts_resampled",
        columns=["gravityX", "gravityY", "gravityZ"],
    )

    process(reading, rotation_step)

    acc_rotated = (
        reading.get_level("sbt")
        .get_raw_data_set("accelerometer_ts_resampled_rotated")
        .data
    )

    assert np.round(acc_rotated.gravityX.mean(), 0) == -1.0
    assert np.round(acc_rotated.gravityY.mean(), 0) == 0.0
    assert np.round(acc_rotated.gravityZ.mean(), 0) == 0.0


def test_preprocessing_steps_lower_limb():
    """Test :class:`dispel.processing.preprocessing.PreprocessingSteps`."""
    reading = read_ads(EXAMPLE_PATH)

    level = "utt"
    level_filt = LevelIdFilter(level)

    dataset_id = "accelerometer"
    columns = ["userAccelerationX", "userAccelerationY", "userAccelerationZ"]

    preprocessing_steps = PreprocessingSteps(
        data_set_id=dataset_id,
        limb=LimbModality.LOWER_LIMB,
        sensor=SensorModality.ACCELEROMETER,
        columns=columns,
        level_filter=level_filt,
    )

    process(reading, preprocessing_steps)

    assert isinstance(
        reading.get_level(level).get_raw_data_set("acc_ts_rotated_resampled_detrend"),
        RawDataSet,
    )


def test_preprocessing_steps_upper_limb():
    """Test :class:`dispel.processing.preprocessing.PreprocessingSteps`."""
    reading = read_ads(EXAMPLE_PATH)

    level = "utt"
    level_filt = LevelIdFilter(level)

    dataset_id = "accelerometer"
    columns = [
        "userAccelerationX",
        "userAccelerationY",
        "userAccelerationZ",
        "gravityX",
        "gravityY",
        "gravityZ",
    ]

    steps = PreprocessingSteps(
        data_set_id=dataset_id,
        limb=LimbModality.UPPER_LIMB,
        sensor=SensorModality.ACCELEROMETER,
        columns=columns,
        level_filter=level_filt,
    )

    process(reading, steps)

    with pytest.raises(ValueError):
        reading.get_level(level).get_raw_data_set(
            f"{dataset_id}_ts_resampled_svgf_bhpf_detrend_rotated"
        )
