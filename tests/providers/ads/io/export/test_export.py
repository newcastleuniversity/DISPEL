"""Test cases for :mod:`dispel.providers.ads.io.export`."""

import pandas as pd

from dispel.io.export import process_and_export
from dispel.io.utils import load_json
from dispel.providers import auto_process
from dispel.providers.ads.io import parse_ads_raw_json
from tests.providers.ads.conftest import EXAMPLE_PATH_6MWT


def test_export_ads_features():
    """Test whether an error is raised when the record is incomplete."""
    json_data = load_json(EXAMPLE_PATH_6MWT)

    # process and export features automatically using registry and providers
    output = process_and_export(json_data)
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
        "version_number",
        "platform_value",
    ]

    # process reading manually
    reading = parse_ads_raw_json(json_data)
    auto_process(reading)
    assert len(df) == len(reading.get_merged_feature_set())
