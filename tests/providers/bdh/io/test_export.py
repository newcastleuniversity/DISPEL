"""Test cases for :mod:`dispel.providers.bdh.io.export`."""

import itertools

import pandas as pd
import pytest

from dispel.io.export import process_and_export
from dispel.io.utils import load_json
from dispel.providers import auto_process
from dispel.providers.bdh.io import parse_bdh_reading
from tests.providers.bdh.conftest import EXAMPLE_PATH_CPS, EXAMPLE_PATH_FT_UNFINISHED


def test_incomplete_record_processing():
    """Test whether an error is raised when the evaluation is incomplete."""
    data = load_json(EXAMPLE_PATH_FT_UNFINISHED)
    with pytest.raises(ValueError):
        process_and_export(data)


def test_features(example_json_cps):
    """Test exporting BDH reading features."""
    output = process_and_export(load_json(EXAMPLE_PATH_CPS))
    df = pd.DataFrame(output)
    assert df.columns.tolist() == [
        "evaluation_uuid",
        "feature_id",
        "feature_name",
        "feature_value",
        "feature_unit",
        "feature_type",
        "feature_min",
        "feature_max",
        "flag_ids",
        "flag_reasons",
        "evaluation_code",
        "start_date",
        "end_date",
        "uuid",
        "user_id",
        "is_finished",
        "exit_reason",
        "acquisition_provenance_source_name",
        "acquisition_provenance_source_version",
        "acquisition_provenance_source_build_number",
        "acquisition_provenance_source_device_platform",
        "acquisition_provenance_source_device_os_version",
        "acquisition_provenance_source_device_screen_width_pixels",
        "acquisition_provenance_source_device_screen_height_pixels",
        "acquisition_provenance_source_device_battery_percentage",
        "acquisition_provenance_source_device_time_zone",
        "acquisition_provenance_source_device_time_since_boot",
        "acquisition_provenance_source_device_cpu_load",
        "acquisition_provenance_source_device_model_name",
        "acquisition_provenance_source_device_model_number",
        "acquisition_provenance_source_device_kernel_version",
        "completion",
        "interruption_reason",
        "effective_time_frame_begin_timestamp",
        "effective_time_frame_end_timestamp",
        "inclinic_mode",
        "session_cluster_id",
        "cluster_activity_codes",
        "attempt_number",
        "id",
        "upload_datetime",
        "study_id",
        "schema_id_namespace",
        "schema_id_name",
        "schema_id_version",
        "version_number",
        "platform_value",
    ]

    reading = parse_bdh_reading(load_json(EXAMPLE_PATH_CPS))
    auto_process(reading)
    assert len(df) == len(reading.get_merged_feature_set())


def test_flags_export(example_json_draw_orientation_invalid):
    """Test the exporting of the flags."""
    res = process_and_export(example_json_draw_orientation_invalid)
    df = pd.DataFrame(res)
    flags = [i.split(";") for i in df["flag_ids"].unique()]
    assert "draw-behavioral-deviation-upmo" in set(list(itertools.chain(*flags)))
