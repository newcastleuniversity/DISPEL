"""Specific functionality for signal processing of interpolations."""
import numpy as np
from scipy.interpolate import interp1d


def custom_interpolator_1d(
    path: np.ndarray, up_sampling_factor: float, kind: str
) -> np.ndarray:
    """Interpolate x and y coordinates of a trajectory.

    Parameters
    ----------
    path
        The given 2 dimensional path to interpolate.
    up_sampling_factor
        The up-sampling factor.
    kind
        Specifies the kind of interpolation as a string or as an integer specifying the
        order of the spline interpolator to use. The string has to be one of 'linear',
        'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', or
        'next'. 'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline
        interpolation of zeroth, first, second or third order; 'previous' and 'next'
        simply return the previous or next value of the point; 'nearest-up' and
        'nearest' differ when interpolating half-integers (e.g. 0.5, 1.5) in that
        'nearest-up' rounds up and 'nearest' rounds down. Default is 'linear'.

    Returns
    -------
    numpy.ndarray
        The interpolated trajectory.
    """
    if path.size == 0:
        return path
    distance = np.cumsum(np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0)
    alpha = np.linspace(0, distance.max(), round(path.shape[0] * up_sampling_factor))
    interpolator = interp1d(distance, path, kind, axis=0)
    return interpolator(alpha)


def cubic_splines(path: np.ndarray, up_sampling_factor: float) -> np.ndarray:
    """Interpolate x and y coordinates of a trajectory using cubic splines.

    According to `Scipy` documentation, ``Interpolate data with a piecewise cubic
    polynomial which is twice continuously differentiable. Here spline interpolation is
    preferred to polynomial interpolation because it yields similar results, even when
    using low degree polynomials, while avoiding Runge's phenomenon for higher
    degrees.``

    Parameters
    ----------
    path
        The given 2 dimensional path to interpolate.
    up_sampling_factor
        The up-sampling factor.

    Returns
    -------
    numpy.ndarray
        The interpolated trajectory.
    """
    return custom_interpolator_1d(path, up_sampling_factor, kind="cubic")
