"""Test cases for :mod:`dispel.providers.generic.tasks.gait`."""

import pandas as pd

from dispel.providers.generic.tasks.gait.core import StepEvent
from dispel.providers.generic.tasks.gait.hip import compute_hip_rotation


def test_hip_rotation():
    """Tests compute_hip_rotation with synthetic data."""
    _step_detection = pd.DataFrame(
        {"event": [StepEvent.INITIAL_CONTACT] * 11 + [StepEvent.UNKNOWN] * 2},
        index=pd.date_range(0, periods=13, freq="2s"),
    )
    _rot_speed = pd.DataFrame(
        {"rotation_speed": [0, 1, 1, 4, 4, 4, 2, 2, 2, 1, 1, 0, 1]},
        index=pd.date_range(0, periods=13, freq="2s"),
    )["rotation_speed"]

    expected = pd.DataFrame(
        {"hip_rotation": [1.0, 2.0, 5.0, 8.0, 8.0, 6.0, 4.0, 4.0, 3.0, 2.0, 2.0]},
        index=pd.date_range(0, periods=11, freq="2s"),
    )
    res = compute_hip_rotation(_rot_speed, _step_detection, on_walking_bouts=False)
    assert res.equals(expected)
