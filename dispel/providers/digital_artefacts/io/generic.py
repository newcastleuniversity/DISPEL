"""Generic parsing functionality for digital artefacts (DA) records."""

from typing import Dict

import pandas as pd

from dispel.data.core import Evaluation
from dispel.data.devices import Device, IOSPlatform, Screen
from dispel.data.epochs import Epoch, EpochDefinition


def parse_device(record: Dict) -> Device:
    """Parse the device information.

    Parameters
    ----------
    record
        The dictionary containing the metadata about the Device

    Returns
    -------
    Device
        The representation of the device
    """
    screen = Screen(
        width_dp_pt=record["device"]["screen_width_pt"],
        height_dp_pt=record["device"]["screen_height_pt"],
        width_pixels=1,
        height_pixels=1,
    )

    # Derive correct platform if there are more possibilities than iOS.
    assert record["device"]["os"] == "iOS"

    return Device(
        platform=IOSPlatform(),
        model=record["device"]["platform"],
        screen=screen,
        app_build_number=record["app"]["app_build_number"],
        app_version_number=record["app"]["app_version_number"],
    )


def parse_evaluation_epoch(record: Dict) -> Epoch:
    """Extract the global epoch of the assessment.

    Parameters
    ----------
    record
        The DA record containing the session information.

    Returns
    -------
    Epoch
        The epoch object of for the evaluation of the reading.

    """
    start = pd.Timestamp(record["session"]["start_utc"])
    end = start + pd.Timedelta(record["session"]["duration"], unit="ms")
    return Epoch(start, end)


def parse_evaluation(record: Dict) -> Evaluation:
    """Parse the evaluation information of the session.

    Parameters
    ----------
    record
        The DA record containing the session information.

    Returns
    -------
    Evaluation
        The evaluation representation of the session

    """
    id_ = record["session"]["test_id"]
    epoch = parse_evaluation_epoch(record)

    return Evaluation(
        start=epoch.start,
        end=epoch.end,
        definition=EpochDefinition(id_=id_),
        uuid=record["session"]["session_id"],
        user_id=record["session"]["subject_id"],
        finished=True,
    )
