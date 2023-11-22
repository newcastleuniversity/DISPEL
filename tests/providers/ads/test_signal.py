"""Test cases for :mod:`dispel.signal.dtw`.

FIXME: move test cases out of dispel.providers as soon as synthetic test case exists.
"""

import numpy as np
import pandas as pd
import pytest

from dispel.providers.generic.tasks.draw.shapes import get_reference_path
from dispel.signal.accelerometer import (
    apply_rotation_matrices,
    compute_rotation_matrices,
)
from dispel.signal.dtw import get_dtw_distance


def test_get_dtw_distance(
    example_data_draw_sccr_screen_path_up_sampled,
    example_reading_draw,
    height_screen,
):
    """Test the good calculation of the sim related measures."""
    # FIXME replace fixture with synthetic data and make test case test-module
    #  independent
    ref = get_reference_path(
        example_reading_draw.get_level("square_counter_clock-right"),
        height_screen,
    )
    dtw_data = get_dtw_distance(example_data_draw_sccr_screen_path_up_sampled, ref)
    assert dtw_data["dtw_coupling_measure"] == pytest.approx(4075.932083378255)
    assert dtw_data["dtw_mean_distance"] == pytest.approx(3.8412161373461386)
    assert dtw_data["dtw_median_distance"] == pytest.approx(3.39585264856729)
    assert dtw_data["dtw_std_distance"] == pytest.approx(2.790462199636898)
    assert dtw_data["dtw_total_distance"] == pytest.approx(3495.5066849849864)


def test_compute_rotation_matrices(example_accelerometer):
    """Test :func:`dispel.signal.accelerometer.compute_rotation_matrices`."""
    # FIXME replace fixture with synthetic data
    gravity = example_accelerometer[[f"gravity{a}" for a in list("XYZ")]]
    res = compute_rotation_matrices(gravity, (-1, 0, 0))
    assert isinstance(res, pd.Series)
    assert len(gravity) == len(res)


def test_apply_rotation_matrices(example_accelerometer):
    """Test :func:`dispel.signal.accelerometer.apply_rotation_matrices`."""
    # FIXME replace fixture with synthetic data
    gravity = example_accelerometer[[f"gravity{a}" for a in list("XYZ")]]
    rotation = compute_rotation_matrices(gravity, (-1, 0, 0))

    sensor_data = pd.DataFrame(np.ones((len(rotation), 3)))
    res = apply_rotation_matrices(rotation, sensor_data)

    assert sensor_data.shape == res.shape
