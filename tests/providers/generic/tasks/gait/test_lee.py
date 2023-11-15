"""Tests for :mod:`dispel.providers.generic.tasks.gait.lee`."""
import pandas as pd
import pytest

from dispel.providers.generic.tasks.gait.lee import StepState
from dispel.providers.generic.tasks.gait.lee import _check_state as check_state


@pytest.fixture
def dummy_norm():
    """Create a fixture of acceleration norm."""
    return pd.DataFrame({"norm": [1, 0, 2, 3, 11, 4, 3, 0, 4, 5, 10, 2, 3]}).set_index(
        pd.to_datetime(list(range(13)), unit="s")
    )


@pytest.mark.parametrize(
    "index_s, index_c, acc_threshold, greater, further, expected",
    [
        # scenario when candidate is a peak but too close to the previous
        # valley
        (1, 3, None, True, True, False),
        # scenario when candidate is a peak and a new peak is detected. There
        # is enough distance separating it from the previous state which was a
        # valley.
        (1, 4, None, True, True, True),
        # scenario when the candidate is a valley and the previous state was a
        # valley. A new valley is detected when the distance with the previous
        # one is less than the time threshold and when the magnitude is lower.
        (5, 6, 3.5, False, False, True),
    ],
)
def test_check_state(
    dummy_norm, index_s, index_c, acc_threshold, greater, further, expected
):
    """Test some of the logic of the check_state function."""
    check = check_state(
        data=dummy_norm["norm"],
        last_state=StepState.VALLEY,
        expected_state=StepState.VALLEY,
        index_s=index_s,
        index_c=index_c,
        t_thresh=2.5,
        acc_threshold=acc_threshold,
        greater=greater,
        further=further,
    )
    assert check == expected
