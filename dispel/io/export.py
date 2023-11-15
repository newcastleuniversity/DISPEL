"""A module to automatically process readings and export features."""

from typing import cast

import pandas as pd

from dispel.data.devices import Device, PlatformType
from dispel.data.features import FeatureValue
from dispel.io import read
from dispel.io.flags import merge_flags
from dispel.processing.data_trace import DataTrace
from dispel.providers import auto_process


def export_features(trace: DataTrace) -> pd.DataFrame:
    """Export features with additional meta information from processing trace.

    FIXME: documentation
    """
    reading = trace.get_reading()
    evaluation = reading.evaluation

    rows = []
    feature_set = reading.get_merged_feature_set()
    for feature in feature_set.values():
        feature_value = cast(FeatureValue, feature)
        flags = trace.get_flags(feature_value)
        rows.append(
            dict(
                evaluation_uuid=evaluation.uuid,
                **feature_value.to_dict(stringify=True),
                **merge_flags(flags),
            )
        )

    # enrich with additional reading and evaluation information
    df = pd.DataFrame(rows)

    evaluation_dict = evaluation.to_dict()
    for key, value in evaluation_dict.items():
        df[key] = value

    assert isinstance(device := reading.device, Device)
    df["version_number"] = device.app_version_number
    assert isinstance(platform := device.platform, PlatformType)
    df["platform_value"] = platform.repr()

    return df


def process_and_export(data: dict) -> dict:
    """FIXME: write documentation."""
    reading = read(data)

    # only process completed records
    if not reading.evaluation.finished:
        raise ValueError("Can only process finished evaluations.")

    data_trace = auto_process(reading)
    data_trace.check_data_set_usage()
    features = export_features(data_trace)

    return features.to_dict()
