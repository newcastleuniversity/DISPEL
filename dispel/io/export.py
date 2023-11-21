"""A module to automatically process readings and export measures."""

from typing import cast

import pandas as pd

from dispel.data.devices import Device, PlatformType
from dispel.data.measures import MeasureValue
from dispel.io import read
from dispel.io.flags import merge_flags
from dispel.processing.data_trace import DataTrace
from dispel.providers import auto_process


def export_measures(trace: DataTrace) -> pd.DataFrame:
    """Export measures with additional meta information from processing trace.

    FIXME: documentation
    """
    reading = trace.get_reading()
    evaluation = reading.evaluation

    rows = []
    measure_set = reading.get_merged_measure_set()
    for measure in measure_set.values():
        measure_value = cast(MeasureValue, measure)
        flags = trace.get_flags(measure_value)
        rows.append(
            dict(
                evaluation_uuid=evaluation.uuid,
                **measure_value.to_dict(stringify=True),
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
    measures = export_measures(data_trace)

    return measures.to_dict()
