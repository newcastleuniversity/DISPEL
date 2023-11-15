"""Functions for reading BDH NEUROQOL data."""

from typing import Any, Dict

from dispel.data.raw import RawDataSet, RawDataSetDefinition
from dispel.providers.bdh.io.core import parse_raw_data_set


def convert_activity_sequence(
    data: Dict[str, Any], definition: RawDataSetDefinition
) -> RawDataSet:
    """Convert activity sequence dataset to userInput."""
    return parse_raw_data_set(data, definition)
