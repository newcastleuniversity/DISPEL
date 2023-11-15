"""Accelerometer functionality for signal processing tasks."""
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import quaternion
from scipy.spatial.transform import Rotation

from dispel.signal.core import compute_rotation_matrix_3d

GRAVITY_CONSTANT = 9.80665
r"""The gravitational acceleration near Earth's surface."""


def quaternion_arr_normalize(q_arr: np.ndarray) -> np.ndarray:
    """Normalize an array of quaternions.

    Parameters
    ----------
    q_arr
        A numpy.ndarray of shape (n_samples, 4) containing the quaternions.

    Returns
    -------
    numpy.ndarray
        The unit quaternion in the same format as the input.
    """
    is_quat_array = False
    if isinstance(q_arr[0], quaternion.quaternion):
        is_quat_array = True

    # quat array to float array
    if is_quat_array:
        q_arr = quaternion.as_float_array(q_arr)
    q_arr = q_arr / np.linalg.norm(q_arr, axis=1)[:, None]

    q_arr_out = quaternion.as_quat_array(q_arr) if is_quat_array else q_arr
    return q_arr_out


def quaternion_rotate_vector(q_ba_np: np.ndarray, v_a_np: np.ndarray) -> np.ndarray:
    """Rotate a time series of a 3d vector by a quaternion.

    Parameters
    ----------
    q_ba_np
        A numpy.ndarray of shape (n_samples, 4) containing the quaternions, expressing
        the rotation from coordinate frame a to b.
    v_a_np
        A numpy.ndarray of shape (n_samples, 3) containing the vector, expressed in
        coordinate frame a.

    Returns
    -------
    numpy.ndarray
        A numpy.ndarray of shape (n_samples, 3) containing the vector, expressed in
        coordinate frame b.
    """
    # assert this is actually an array of expected dimensions
    assert q_ba_np.shape[1] == 4
    assert v_a_np.shape[1] == 3
    assert q_ba_np.shape[0] == v_a_np.shape[0]

    # number of samples
    n_samples = q_ba_np.shape[0]

    # convert float array to quat array
    q_ba = quaternion.as_quat_array(q_ba_np)

    # ensure the quaternion is unit
    q_ba = quaternion_arr_normalize(q_ba)

    # add zeros as the real quat part
    v_a = np.c_[np.zeros(n_samples), v_a_np]

    # convert float array to quat array
    v_a = quaternion.as_quat_array(v_a)

    # take the conjugate to express opposite rotation
    q_ab = np.conjugate(q_ba)

    # multiply quaternions to rotate vector to frame b:
    # v_b = q_ba * v_a * q_ab
    v_b = np.multiply(np.multiply(q_ba, v_a), q_ab)

    # convert quat array to float array and take only the imaginary component
    v_b_np = quaternion.as_float_array(v_b)[:, 1:]

    return v_b_np


def remove_gravity_component(data: pd.DataFrame):
    """Remove the gravity component of acceleration.

    Based on paper :
    Two-stage Recognition of Raw Acceleration Signals for 3-D Gesture-Understanding Cell
    Phones, Cho et al., 2006

    Get the linear accelerations - without gravity component - by subtracting the mean
    acceleration signals, such that
    :math:`A_1(t) = A(t) - A_mean where A(t) = [a_x(t),a_y(t),a_z(t)]`

    Parameters
    ----------
    data
        Input acceleration data

    Returns
    -------
    Tuple[pandas.DataFrame, float]
        A tuple with first, the acceleration data with mean removed and second the mean.
    """
    mean_acc = data.mean()
    lin_acc = data - mean_acc

    return lin_acc, mean_acc


def remove_gravity_component_ori(
    acc_sensor, q_global_sensor, unit: str = "g"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove the gravity component of acceleration based on orientation.

    Get the linear accelerations - without gravity component - by converting to the
    global coordinate frame, subtracting the constant gravity and converting back to the
    initial sensor frame.

    Parameters
    ----------
    acc_sensor
        Input acceleration data expressed on the sensor frame.
    q_global_sensor
        Input orientation from sensor to global coordinate frame.
    unit
        The unit in which the accelerometer data is expressed.

    Returns
    -------
    Tuple[pandas.DataFrame, pandas.DataFrame]
        A tuple with first, the acceleration data with mean removed and second the
        gravity.
    """
    if unit == "g":
        gravity_constant = 1.0
    else:
        gravity_constant = GRAVITY_CONSTANT

    # rotate acceleration from sensor to global frame
    acc_global = quaternion_rotate_vector(q_global_sensor, acc_sensor)

    # derive the user acceleration by subtracting gravity
    user_acc_global = acc_global - [0, 0, gravity_constant]

    # derive gravity in global as array (should be [0, 0, GRAVITY_CONSTANT])
    gravity_global = acc_global - user_acc_global

    # get the conjugate quaternion to express the opposite rotation
    q_sensor_global = np.conjugate(quaternion.as_quat_array(q_global_sensor))
    q_sensor_global = quaternion.as_float_array(q_sensor_global)

    # rotate user and gravity back to the sensor frame
    user_acc_sensor = quaternion_rotate_vector(q_sensor_global, user_acc_global)
    gravity_sensor = quaternion_rotate_vector(q_sensor_global, gravity_global)

    return user_acc_sensor, gravity_sensor


def dot_diag_einsum(v_1: np.ndarray, v_2: np.ndarray) -> np.ndarray:
    """Get the diagonal of the dot product of two 2D arrays (fast).

    This function is based on np.einsum configured in such a way to provide the dot
    product diagonal of two 2D arrays. For benchmark example (n_samples=18000), runtime
    is 2 ms.

    Parameters
    ----------
    v_1
        An np.ndarray of shape (n_samples, 3)
    v_2
        An np.ndarray of shape (n_samples, 3)

    Returns
    -------
    numpy.ndarray
        An array of shape (n_samples, 1) containing the dot product diagonal

    """
    return np.einsum("ij,ij->i", v_1, v_2)


def dot_diag_list(v_1: np.ndarray, v_2: np.ndarray) -> np.ndarray:
    """Get the diagonal of the dot product of two 2D arrays (slow).

    This approach computes the dot product for each time sample of the arrays of shape
    (n_samples, 3). For benchmark example (n_samples=18000), runtime is 60 ms.

    Parameters
    ----------
    v_1
        An np.ndarray of shape (n_samples, 3)
    v_2
        An np.ndarray of shape (n_samples, 3)

    Returns
    -------
    numpy.ndarray
        An array of shape (n_samples, 1) containing the dot product diagonal

    """
    return np.array([np.dot(v_1_, v_2_) for v_1_, v_2_ in zip(v_1, v_2)])


def dot_diag_matrix(v_1: np.ndarray, v_2: np.ndarray) -> np.ndarray:
    """Get the diagonal of the dot product of two 2D arrays (very slow).

    This approach computes the dot product array of two arrays of shape (n_samples, 3)
    and then takes the diagonal. For benchmark example (n_samples=18000), runtime is 2
    seconds.

    Parameters
    ----------
    v_1
        An np.ndarray of shape (n_samples, 3)
    v_2
        An np.ndarray of shape (n_samples, 3)

    Returns
    -------
    numpy.ndarray
        An array of shape (n_samples, 1) containing the dot product diagonal

    """
    return np.diag(np.dot(v_1, v_2.T))


def orthogonal(v_1):
    """Find orthogonal quaternion to a vector provided.

    Based on C++ implementation here:
    https://github.com/PX4/PX4-Matrix/blob/master/matrix/Quaternion.hpp
    (lines 192-224)

    Parameters
    ----------
    v_1
        An np.ndarray of shape (3, )

    Returns
    -------
    numpy.ndarray
        An array of shape (4,) containing the 180 deg quaternion rotation

    """
    cross_product = np.abs(v_1)

    if cross_product[0] < cross_product[1]:
        if cross_product[0] < cross_product[2]:
            cross_product = [1, 0, 0]
        else:
            cross_product = [0, 0, 1]
    else:
        if cross_product[1] < cross_product[2]:
            cross_product = [0, 1, 0]
        else:
            cross_product = [0, 0, 1]

    # compute cross-product (vector part)
    cross_product = np.cross(v_1, cross_product)

    # add scalar part to construct the quaternion
    quaternion = np.insert(cross_product, 0, 0.0)

    return quaternion


def compute_quaternion_between_vectors(v_1: np.ndarray, v_2: np.ndarray) -> np.ndarray:
    """Compute quaternion given two vectors.

    Parameters
    ----------
    v_1
        An np.ndarray of shape (n_samples, 3) denoting the first vector
    v_2
        An np.ndarray of shape (n_samples, 3) denoting the second vector

    Returns
    -------
    numpy.ndarray
        An array containing the quaternion of shape (n_samples, 4).

    See proposed solution with pseudocode here:
    https://stackoverflow.com/questions/1171849/finding-quaternion-
    representing-the-rotation-from-one-vector-to-another
    """
    # normalize input vectors
    v_1 = v_1 / np.linalg.norm(v_1)
    v_2 = v_2 / np.linalg.norm(v_2)

    # compute the half vector
    half = v_1 + v_2
    half = half / np.linalg.norm(half)

    # compute the cross and dot products
    q_v = np.cross(v_1, half)
    q_w = dot_diag_einsum(v_1, half)

    # merge the scalar with vector part
    quaternion = np.insert(q_v, 0, q_w, axis=1)

    # address corner case of 180 degrees which results in div by zero
    cross_product = np.cross(v_1, v_2)
    dot_product = dot_diag_einsum(v_1, v_2)
    # two vectors are parallel when their cross product is 0 and of opposite
    # direction if their dot product is negative
    case_180_deg = (np.isclose(np.linalg.norm(cross_product, axis=1), 0)) & (
        dot_product < 0
    )

    # if at least one angle is 180 deg then replace the computed quaternion
    # with an orthogonal of 180 deg
    if case_180_deg.any():
        # apply the orthogonal function only in the ids where 180 deg
        # angle between vectors was detected
        quaternion[case_180_deg] = np.apply_along_axis(orthogonal, 1, v_1[case_180_deg])

    # normalize output quaternion
    quaternion = quaternion / np.linalg.norm(quaternion, axis=1)[:, None]

    return quaternion


def compute_rotation_matrices_quaternion(
    gravity: pd.DataFrame, target_gravity: Tuple[float, float, float]
) -> pd.Series:
    """Compute rotation matrices from gravity time series (quaternion-based).

    Parameters
    ----------
    gravity
        The gravity time series obtained from the accelerometer sensor.
    target_gravity
        The unit vector onto which to rotate to

    Returns
    -------
    pandas.Series
        A series of rotation matrix objects for the provided ``gravity`` entries.

    """
    # convert target into an array of same shape as gravity
    frame = np.tile(target_gravity, (gravity.shape[0], 1))

    # get quaternion from directional vectors
    quaternion = compute_quaternion_between_vectors(gravity.values, frame)

    # move scalar part to the end to prepare for scipy rotation format
    quaternion = np.roll(quaternion, shift=-1, axis=1)

    # convert quaternion to matrices
    matrices_list = Rotation.from_quat(quaternion).as_matrix()

    return pd.Series([m for m in matrices_list], index=gravity.index)


def compute_rotation_matrices(
    gravity: pd.DataFrame, target_gravity: Tuple[float, float, float]
) -> pd.Series:
    """Compute rotation matrices based on gravity time series.

    Parameters
    ----------
    gravity
        The gravity time series obtained from the accelerometer sensor.
    target_gravity
        The unit vector onto which to rotate to

    Returns
    -------
    pandas.Series
        A series of rotation matrix objects for the provided ``gravity`` entries.

    """
    return gravity.apply(compute_rotation_matrix_3d, b=np.array(target_gravity), axis=1)


def apply_rotation_matrices(
    rotation_matrices: pd.Series, sensor: pd.DataFrame
) -> pd.DataFrame:
    """Apply rotation matrices on a sensor time series.

    Parameters
    ----------
    rotation_matrices
        The rotation matrices obtained with :func:`compute_rotation_matrices`
    sensor
        The sensor time series to be rotated

    Returns
    -------
    pandas.DataFrame
        The rotated sensor values based on ``rotation_matrices``.

    """
    # TODO: fix for BDH format
    common_timestamps = sensor.index.intersection(rotation_matrices.index)
    if len(common_timestamps) < 0.7 * len(sensor.index):
        warnings.warn(
            "More than 30% of the sensor signal has been ignored.", UserWarning
        )

    return pd.DataFrame(
        (
            ri @ vi
            for ri, vi in zip(
                rotation_matrices.loc[common_timestamps],
                sensor.loc[common_timestamps].values,
            )
        ),
        index=common_timestamps,
        columns=sensor.columns,
    )


AP_ML_COLUMN_MAPPINGS = {
    "userAccelerationZ": "ap",
    "userAccelerationY": "ml",
    "userAccelerationX": "v",
}


def transform_ap_ml_acceleration(data: pd.DataFrame) -> pd.DataFrame:
    """Transform accelerometer axis from X,Y,Z to AP,ML, and v."""
    return data.rename(columns=AP_ML_COLUMN_MAPPINGS) * -1
