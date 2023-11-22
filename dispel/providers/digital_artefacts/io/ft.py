"""Functionality to read digital artefacts (DA) finger tapping records."""

from typing import Any, Dict

import pandas as pd

from dispel.data.epochs import Epoch
from dispel.data.levels import Level
from dispel.data.raw import (
    RawDataSet,
    RawDataSetDefinition,
    RawDataSetSource,
    RawDataValueDefinition,
)
from dispel.providers.digital_artefacts.data import DigitalArtefactsReading
from dispel.providers.digital_artefacts.io.generic import parse_device, parse_evaluation
from dispel.providers.registry import register_reader

#: The list of level in the finger tap Digital Artefact records
FT_DA_LEVEL = ["domhand", "nondomhand"]

#: Duration of the finger tap Digital Artefact test
TEST_DURATION = 20


def format_da_ft_to_bdh(level_data: Dict) -> pd.DataFrame:
    """Transform the digital artefact ft data into BDH ft format.

    Parameters
    ----------
    level_data
        Dict that contains the level data from DA

    Returns
    -------
    pandas.DataFrame
        A pandas dataframe in the BDH format

    Raises
    ------
    ValueError
        If block number is not unique


    """
    if len(pd.Series(level_data["block"]).unique()) > 1:
        raise ValueError("Mixed block inside one level")

    tapping_data = pd.DataFrame(level_data)[
        [
            "tap type",
            "tap start (ms)",
            "tap duration (ms)",
            "x (pts)",
            "y (pts)",
            "distance to target center (pts)",
            "start time (utc)",
        ]
    ]
    # Rename columns to match BDH format
    tapping_data.rename(
        columns={
            "tap type": "location",
            "tap duration (ms)": "tap_duration",
            "x (pts)": "x_coordinate",
            "y (pts)": "y_coordinate",
            "distance to target center (pts)": "distance_to_target_center",
        },
        inplace=True,
    )

    # Replace miss by none to match BDH format
    tapping_data["location"] = (
        tapping_data["location"].map({"miss": "none"}).fillna(tapping_data["location"])
    )
    # Convert to timestamp series
    tapping_data["start time (utc)"] = pd.to_datetime(tapping_data["start time (utc)"])
    # Generate new timestamps by adding tap start and start time
    tapping_data["timestamp"] = tapping_data["start time (utc)"] + pd.to_timedelta(
        tapping_data["tap start (ms)"], unit="ms"
    )
    tapping_data["first_position"] = list(
        zip(tapping_data.x_coordinate, tapping_data.y_coordinate)
    )
    # Set the press_in_ts as the new index
    tapping_data.set_index("timestamp", inplace=True)
    # Filter the data to match the BDH format
    bdh_format_data = tapping_data[
        [
            "location",
            "tap_duration",
            "first_position",
        ]
    ]
    return bdh_format_data


def generate_raw_dataset(bdh_format_data: pd.DataFrame) -> RawDataSet:
    """Transform the BDH format data into a RawDataSet.

    Parameters
    ----------
    bdh_format_data
        The finger tapping data in the io format

    Returns
    -------
    RawDataSet
        The RawDataSet representation of the finger tapping data

    """
    definitions = [
        RawDataValueDefinition(column, column.upper())
        for column in bdh_format_data.columns
    ]
    dataset_source = RawDataSetSource("da")
    return RawDataSet(
        RawDataSetDefinition("enriched_tap_events_ts", dataset_source, definitions),
        bdh_format_data,
    )


def get_level_epoch(level_data: Dict) -> Epoch:
    """Extract level epoch from the digital artefact record.

    Parameters
    ----------
    level_data
        The dictionary containing the level data

    Returns
    -------
    Epoch
        The epoch object of the current level
    """
    start = pd.Timestamp(pd.Series(level_data["start time (utc)"]).iloc[0])
    return Epoch(start=start, end=start + pd.Timedelta(TEST_DURATION, unit="s"))


def create_bdh_level(level_data: Dict, level_id: str) -> Level:
    """Create a :class:`~dispel.data.core.Level` from the DA level data.

    Parameters
    ----------
    level_data
        A dict containing the DA level data
    level_id
        The id of the level

    Returns
    -------
    Level
        The level representation of the input data
    """
    epoch = get_level_epoch(level_data)
    bdh_level_data = format_da_ft_to_bdh(level_data)
    datasets = generate_raw_dataset(bdh_level_data)
    return Level(
        id_=level_id, start=epoch.start, end=epoch.end, raw_data_sets=[datasets]
    )


def parsable_da_json(value: Any) -> bool:
    """Test if a value is a dictionary and can be parsed by the DA parser."""
    if not isinstance(value, dict):
        return False

    return "app" in value.keys()


@register_reader(parsable_da_json, DigitalArtefactsReading)
def parse_ft_reading(record: Dict) -> DigitalArtefactsReading:
    """Get the reading representation of a da data frame.

    Parameters
    ----------
    record
        The dictionary representing the digital artefact record

    Returns
    -------
    DigitalArtefactsReading
        The reading representation of the DA file
    """
    evaluation = parse_evaluation(record)
    device = parse_device(record)
    test_data = [(record[hand], hand) for hand in FT_DA_LEVEL]
    levels = [create_bdh_level(level_data, hand) for level_data, hand in test_data]

    return DigitalArtefactsReading(
        evaluation=evaluation,
        levels=levels,
        device=device,
    )
