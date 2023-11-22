"""Test cases for :mod:`dispel.signal.drawingLib`."""
import numpy as np

from dispel.signal.interpolators import cubic_splines


def test_up_sampling():
    """Test the proper computation of 2D splines."""
    path = np.stack([np.arange(1, 7)] * 2, axis=1)
    up_sampling_factor = 5
    out = cubic_splines(path, up_sampling_factor)
    # Should retrieve len(path) * up_sampling_factor : 6 * 5 = 30
    assert out.shape[0] == 30
