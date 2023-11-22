"""signal.geometric module.

A module containing common operations on different geometries.
"""
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def extract_ellipse_dir_vectors(comps: pd.DataFrame) -> Tuple[float, float]:
    """Extract eigen-director vectors of ellipse representing the data.

    Parameters
    ----------
    comps
        A pd.DataFrame with a 2-dimensional timeseries

    Returns
    -------
    Union[array, array]
        Tuple containing:
            v1_v_hat : np.array
                An array with the first PCA eigenvector
            v2_v_hat : np.array
                An array with the second PCA eigenvector

    """
    # Extract PCA components of the 2-dimensional planar timeseries
    pca = PCA(n_components=2)
    pca = pca.fit(comps)

    # Take components of 2-PCA analyses and turn into array
    v1_v = pca.components_[0]
    v2_v = pca.components_[1]

    # Select longest component as major, remaining as minor
    norm_v1 = np.linalg.norm(v1_v)
    norm_v2 = np.linalg.norm(v2_v)

    # Take the unitary values of the director vectors
    v1_v_hat = v1_v / norm_v1
    v2_v_hat = v2_v / norm_v2

    return v1_v_hat, v2_v_hat


def extract_ellipse_axes(comps: pd.DataFrame) -> Tuple[float, float]:
    """Extract length of the axes of an ellipse covering 95-percentile of data.

    Parameters
    ----------
    comps
        A pd.DataFrame with a 2-dimensional timeseries

    Returns
    -------
    Union[int, int]
        Tuple containing:
            major_axis : float
                The length of the major axis of an ellipse
            minor_axis : float
                The length of the minor axis of an ellipse
    """
    # Extract PCA components of the 2-dimensional planar timeseries
    pca = PCA(n_components=2)
    pca = pca.fit(comps)

    # Transform distribution to canonical cartesian axes
    data_transformed = pca.transform(comps)
    data_transformed_df = pd.DataFrame(data_transformed, columns=["ap", "ml"])

    # Compute the min and max boundaries of 95% of data covered by the ellipse
    ml_min = np.quantile(data_transformed_df.ml, 0.05)
    ml_max = np.quantile(data_transformed_df.ml, 0.95)
    ap_min = np.quantile(data_transformed_df.ap, 0.05)
    ap_max = np.quantile(data_transformed_df.ap, 0.95)

    # Compute the range of each axes (i.e., ml and ap)
    rang_ml = abs(ml_max - ml_min)
    rang_ap = abs(ap_max - ap_min)

    # Select the minor and major axes
    major_axis = max([rang_ml, rang_ap])
    minor_axis = min([rang_ml, rang_ap])

    return major_axis, minor_axis


def downsample_dataset(data: pd.DataFrame, ratio_freq: float = 0.5) -> pd.DataFrame:
    """Downsample the dataset to a fraction of original frequency.

    Parameters
    ----------
    data
        A pd.Dataframe with the original dataset
    ratio_freq
        A float with the ratio of downsampling from original sampling frequency

    Returns
    -------
    pd.Dataframe
        The resulting downsampled dataset
    """
    assert ratio_freq <= 1
    assert ratio_freq > 0
    # Measure sampling time
    t_sample = data.reset_index().ts.diff().min().microseconds / 1e6
    # Parse sampling time to rule for resample
    rule = f"{t_sample / ratio_freq:.3f}S"
    # Resample data to downsample by ratio_freq
    return data.resample(rule).bfill()


def upsample_dataset(data: pd.DataFrame, factor_freq: float = 2) -> pd.DataFrame:
    """Upsample the datasat to a factor of original frequency.

    Parameters
    ----------
    data
        A pd.Dataframe with the original dataset
    factor_freq
        A float with the factor of upsampling from original sampling rate

    Returns
    -------
    pd.Dataframe
        The resulting upsampled dataset
    """
    assert factor_freq >= 1
    # Measure sampling time
    t_sample = data.reset_index().ts.diff().min().microseconds / 1e6
    # Parse sampling time to rule for resample
    rule = f"{t_sample / factor_freq:.3f}S"
    # Resample data to upsample by factor factor_freq
    return data.resample(rule).bfill()


def draw_circle(length: int = 100, radius: int = 1):
    """Draw a circle from polar to cartesian coordinates."""
    angles = np.linspace(0, 2 * np.pi, length)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return x, y


def draw_ellipse(length: int = 100, a: int = 1, b: int = 2):
    """Draw a circle from polar to cartesian coordinates."""
    angles = np.linspace(0, 2 * np.pi, length)
    x = a * np.cos(angles)
    y = b * np.sin(angles)
    return x, y


def synthetic_outliers(x: pd.Series, y: pd.Series, ratio_outlier: float):
    """Generate outliers for a point cloud.

    Parameters
    ----------
    x
        pd.Series of the first coordinate of the point cloud
    y
        pd.Series of the second coordinate of the point cloud
    ratio_outlier
        float indicating amount of points to be turned into outlier

    Returns
    -------
        A tuple with two coordinates of the point cloud with outliers

    """
    series = {"ap": x, "ml": y}
    df = pd.DataFrame(series)
    # Sample random_points corresponding to ratio_outlier % points of the total
    random_points = df.sample(frac=ratio_outlier, random_state=1)
    # The random points are scaled up by, e.g., a factor of 2
    df.iloc[random_points.index] = random_points * 2
    return np.array(df.ap), np.array(df.ml)


def rotate_points(x: pd.Series, y: pd.Series, angle: float):
    """Rotate a point cloud.

    Parameters
    ----------
    x
        pd.Series of the first coordinate of the point cloud
    y
        pd.Series of the second coordinate of the point cloud
    angle
        float of the angle to rotate in radians

    Returns
    -------
        A tuple with two coordinates of the rotated point cloud

    """
    x_rot = x * np.cos(angle) - y * np.sin(angle)
    y_rot = y * np.cos(angle) + x * np.sin(angle)
    return x_rot, y_rot
