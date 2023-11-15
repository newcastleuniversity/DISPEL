"""Generic functionality for signal processing steps."""
from functools import partial
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from dispel.data.core import Reading
from dispel.data.features import FeatureValueDefinitionPrototype
from dispel.data.levels import Level
from dispel.data.raw import (
    ACCELEROMETER_COLUMNS,
    DEFAULT_COLUMNS,
    GRAVITY_COLUMNS,
    RawDataValueDefinition,
)
from dispel.data.values import AbbreviatedValue as AV
from dispel.processing.assertions import NotEmptyDataSetAssertionMixin
from dispel.processing.data_set import transformation
from dispel.processing.extract import ExtractMultipleStep, ExtractStep
from dispel.processing.level import LevelFilterType
from dispel.processing.modalities import SensorModality
from dispel.processing.transform import Apply, TransformStep
from dispel.providers.bdh.data import BDHReading
from dispel.signal.accelerometer import (
    GRAVITY_CONSTANT,
    apply_rotation_matrices,
    compute_rotation_matrices_quaternion,
    remove_gravity_component,
    remove_gravity_component_ori,
)
from dispel.signal.core import (
    amplitude,
    discretize_sampling_frequency,
    energy,
    entropy,
    euclidean_norm,
    peak,
)
from dispel.signal.sensor import SENSOR_UNIT, find_zero_crossings

# Define expected sampling frequencies
FREQ_20HZ = 20
FREQ_50HZ = 50
FREQ_60HZ = 60
FREQ_100HZ = 100  # SensorLog can sample at 100Hz
FREQ_128HZ = 128  # APDM files are sampled at 128Hz

VALID_FREQ_LIST = [FREQ_20HZ, FREQ_50HZ, FREQ_100HZ, FREQ_128HZ]


class RenameColumns(TransformStep):
    r"""Rename and select columns of a raw data set.

    Parameters
    ----------
    data_set_id
        The data set id of the time series to be renamed.
    level_filter
        An optional :class:`~dispel.processing.level.LevelFilter` to determine
        the levels to be transformed. If no filter is provided, all levels
        will be transformed. The ``level_filter`` also accepts :class:`str`,
        :class:`~dispel.data.core.LevelId`\ s and lists of either and passes them
        to a :class:`~dispel.processing.level.LevelFilter` for convenience.
    kwargs
        All arguments passed into this class will serve as a renaming mapping
        for the raw data set.
    """

    def __init__(
        self, data_set_id: str, level_filter: Optional[LevelFilterType] = None, **kwargs
    ):
        def _transform_function(data: pd.DataFrame) -> pd.DataFrame:
            data_ = data.rename(columns=kwargs)
            return data_[kwargs.values()]

        super().__init__(
            data_set_id,
            _transform_function,
            f"{data_set_id}_renamed",
            [RawDataValueDefinition(column, column) for column in kwargs.values()],
            level_filter=level_filter,
        )


class SetTimestampIndex(TransformStep):
    r"""Create a new time series based on a date time or time delta column.

    Parameters
    ----------
    data_set_id
        The data set id of the time series to be transformed.
    columns
        The columns to consider in the new raw data set.
    time_stamp_column
        The time series column name to use as index.
    level_filter
        An optional :class:`dispel.processing.level.LevelFilter` to determine the
        levels to be transformed. If no filter is provided, all levels will be
        transformed. The ``level_filter`` also accepts :class:`str`,
        :class:`~dispel.data.core.LevelId`\ s and lists of either and passes them
        to a :class:`~dispel.processing.level.LevelFilter` for convenience.
    duplicates
        The strategy used to handle duplicates.
        Has to be one of ``ignore``, ``raise``, ``first``, ``last``.
    """

    def __init__(
        self,
        data_set_id: str,
        columns: List[str],
        time_stamp_column: str = "ts",
        level_filter: Optional[LevelFilterType] = None,
        duplicates: Optional[str] = None,
    ):
        def _transform_function(
            data: pd.DataFrame, rm_duplicate: Optional[str]
        ) -> pd.DataFrame:
            if rm_duplicate is None:
                return data.set_index(time_stamp_column)[columns].copy()
            res = data.set_index(time_stamp_column)[columns].copy()
            return res[~res.index.duplicated(keep=duplicates)]

        super().__init__(
            data_set_id,
            lambda x: _transform_function(x, duplicates),
            f"{data_set_id}_ts",
            [RawDataValueDefinition(column, column) for column in columns],
            level_filter=level_filter,
        )


class Trim(TransformStep):
    """Trim a sensor signal at the beginning and/or end.

    Parameters
    ----------
    trim_left
        The amount of data to trim from the left side of the sensor readings.
    trim_right
        The amount of data to trim from the right side of the sensor readings.
    ts_column
        The column id to be used in the provided raw data set through
        ``data_set_ids``. If no column is provided, the data set is expected
        to have a time-based index that is used to trim the data set.
    """

    trim_left = pd.Timedelta(0)
    trim_right = pd.Timedelta(0)
    ts_column: Optional[str] = None

    def __init__(self, *args, **kwargs):
        if (left := kwargs.pop("trim_left", None)) is not None:
            self.trim_left = left
        if (right := kwargs.pop("trim_right", None)) is not None:
            self.trim_right = right
        if (column := kwargs.pop("ts_column", None)) is not None:
            self.ts_column = column

        super().__init__(*args, **kwargs)

    @transformation
    def _trim(self, data: pd.DataFrame) -> pd.DataFrame:
        ts_col = data.index if self.ts_column is None else data[self.ts_column]

        if self.trim_left > pd.Timedelta(0):
            data = data[ts_col > ts_col.min() + self.trim_left]
        if self.trim_right > pd.Timedelta(0):
            data = data[ts_col < ts_col.max() - self.trim_right]

        return data.copy()


class Resample(NotEmptyDataSetAssertionMixin, TransformStep):
    r"""Resample a time-based raw data set to a specific sampling frequency.

    The resampling creates a new raw data set which is accessible via the
    data set comprised of the original one concatenated with ``_resampled``.

    Parameters
    ----------
    data_set_id
        The data set to be resampled. This has to be a data set that uses a
        time-based index. You might first have to apply the
        :class:`SetTimestampIndex` processing step before you can apply
        this step.
    aggregations
        A list of resampling methods to be applied in order. Each can be any
         method that is also accepted by :meth:`pandas.DataFrame.agg`.
    columns
        The columns to be considered during the resampling.
    freq
        The frequency to resample to. See also
        :meth:`pandas.DataFrame.resample` for details. If freq is not provided
        the frequency is estimated automatically taking the median frequency.
    max_frequency_distance
        An optional integer specifying the maximum accepted
        distance between the expected frequency and the estimated frequency
        above which we raise an error.
    level_filter
        An optional :class:`dispel.processing.level.LevelFilter` to determine the
        levels to be transformed. If no filter is provided, all levels will be
        transformed. The ``level_filter`` also accepts :class:`str`,
        :class:`~dispel.data.core.LevelId`\ s and lists of either and passes them
        to a :class:`~dispel.processing.level.LevelFilter` for convenience.
    """

    def __init__(
        self,
        data_set_id: str,
        aggregations: Iterable[str],
        columns: Iterable[str],
        freq: Optional[Union[float, str]] = None,
        max_frequency_distance: Optional[int] = None,
        level_filter: Optional[LevelFilterType] = None,
    ):
        def _resample(
            data: pd.DataFrame, sampling_frequency: Optional[Union[float, str]] = None
        ) -> pd.DataFrame:
            # Check if a sampling frequency is provided
            # If not, we discretized the sampling frequency
            if sampling_frequency is None:
                discretize_args = [data, VALID_FREQ_LIST]
                if max_frequency_distance:
                    discretize_args.append(max_frequency_distance)
                sampling_frequency = discretize_sampling_frequency(*discretize_args)
            # Convert the float sampling frequency to a Timedelta format
            if not isinstance(sampling_frequency, str):
                sampling_frequency = pd.Timedelta(1 / sampling_frequency, unit="s")
            resample_obj = data[columns].resample(sampling_frequency)
            for method in aggregations:
                resample_obj = resample_obj.agg(method)
            return resample_obj

        def _definition_factory(column: str) -> RawDataValueDefinition:
            return RawDataValueDefinition(
                column, f"{column} resampled with {aggregations}"
            )

        super().__init__(
            data_set_id,
            partial(_resample, sampling_frequency=freq),
            f"{data_set_id}_resampled",
            [_definition_factory(column) for column in columns],
            level_filter=level_filter,
        )


class Upsample(Apply):
    r"""Upsample a time-based raw data set to a specific sampling frequency.

    The upsampling creates a new raw data set which is an upsampled version
    of the original data set identified by data_set_id. The upsampled data
    set is accessible via the new_data_set_id which is a concatenation of the
    original data_set_id and a suffix ``_upsampled``.

    Parameters
    ----------
    interpolation_method
        Interpolation technique to use to fill NaN values. It should be a
         method that is also accepted by :meth:`pandas.DataFrame.interpolate`.
    freq
        The frequency to upsample to. See also
        :meth:`pandas.DataFrame.resample` for details.
    """

    def get_new_data_set_id(self) -> str:
        """Overwrite new_data_set_id."""
        return f"{self.get_data_set_ids()[0]}_upsampled"  # type: ignore

    def __init__(self, interpolation_method: str, freq: Union[float, str], **kwargs):
        def _upsample(
            data: pd.DataFrame, sampling_frequency: Union[float, str]
        ) -> pd.DataFrame:
            """Upsample a dataframe to a given sampling frequency."""
            # Convert the float sampling frequency to a Timedelta format
            if not isinstance(sampling_frequency, str):
                sampling_frequency = pd.Timedelta(1 / sampling_frequency, unit="s")
            resample_obj = data.resample(sampling_frequency)
            return resample_obj.interpolate(interpolation_method)

        super().__init__(
            method=_upsample, method_kwargs={"sampling_frequency": freq}, **kwargs
        )


class ExtractAverageSignalEnergy(NotEmptyDataSetAssertionMixin, ExtractStep):
    r"""An average signal energy extraction step.

    Parameters
    ----------
    sensor
        The type of sensor on which the extraction is to be performed.
    data_set_id
        The data set id on which the extraction is to be performed.
    columns
        The columns onto which the signal energy is to be computed.
    level_filter
        An optional :class:`~dispel.processing.level.LevelFilter` to determine the
        levels to be transformed. If no filter is provided, all levels will be
        transformed. The ``level_filter`` also accepts :class:`str`,
        :class:`~dispel.data.core.LevelId`\ s and lists of either and passes them
        to a :class:`~dispel.processing.level.LevelFilter` for convenience.
    """

    def __init__(
        self,
        sensor: SensorModality,
        data_set_id: str,
        columns: List[str],
        level_filter: Optional[LevelFilterType] = None,
    ):
        def _average_signal(data: pd.DataFrame):
            return np.linalg.norm(data[columns], ord=2)

        super().__init__(
            data_set_id,
            _average_signal,
            definition=FeatureValueDefinitionPrototype(
                feature_name=AV(f"average {sensor} energy", f"{sensor.abbr}_sig_ene"),
                data_type="float64",
                description=f"The average {sensor} energy of the "
                f'{"".join(columns)} columns of the signal.',
                unit=SENSOR_UNIT[sensor.abbr],
            ),
            level_filter=level_filter,
        )


class ExtractPowerSpectrumFeatures(NotEmptyDataSetAssertionMixin, ExtractMultipleStep):
    r"""A feature extraction processing step for power spectrum features.

    Parameters
    ----------
    sensor
        The type of sensor on which the extraction is to be performed.
    data_set_id
        The data set id on which the extraction is to be performed.
    columns
        The columns onto which the power spectrum features are to be extracted.
    lower_bound
        The lower bound of frequencies below which the signal is filtered.
    upper_bound
        The higher bound of frequencies above which the signal is filtered.
    level_filter
        An optional :class:`~dispel.processing.level.LevelFilter` to determine the
        levels to be transformed. If no filter is provided, all levels will be
        transformed. The ``level_filter`` also accepts :class:`str`,
        :class:`~dispel.data.core.LevelId`\ s and lists of either and passes them
        to a :class:`~dispel.processing.level.LevelFilter` for convenience.
    """

    def __init__(
        self,
        sensor: SensorModality,
        data_set_id: str,
        columns: List[str],
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        level_filter: Optional[LevelFilterType] = None,
    ):
        unit = sensor.unit(order=2)

        atomic_functions = [
            {
                "func": partial(energy, lowcut=lower_bound, highcut=upper_bound),
                "name": AV("energy", "ene"),
                "description": "The power spectrum energy summed between the "
                f"frequencies ({lower_bound}, {upper_bound}) "
                f"of the {{axis}} axis for the {sensor} "
                f"signal.",
                "unit": unit,
                "outcome_uuid": "99ef9a8d-a925-4eb0-9e80-be58cd4a9ac9",
            },
            {
                "func": peak,
                "name": AV("peak", "peak"),
                "description": f"The frequency at which the power spectrum of "
                "the {axis} axis reaches its maximum value for "
                f"the {sensor} signal.",
                "unit": "Hz",
                "outcome_uuid": "87512c93-3a5b-4c9e-9575-fd9ed19649ca",
            },
            {
                "func": entropy,
                "name": AV("entropy", "ent"),
                "description": "The power spectrum entropy of the {axis} axis "
                f"for the {sensor} signal.",
                "unit": unit,
                "outcome_uuid": "6726bb5a-8084-49f5-a53e-6a28a8f27695",
            },
            {
                "func": amplitude,
                "name": AV("amplitude", "amp"),
                "description": "The power spectrum amplitude (i.e. the maximum"
                " value) of the {axis} axis for the "
                f"{sensor} signal.",
                "unit": unit,
                "outcome_uuid": "bde2c1f9-abf7-41e7-91f8-e0ddddf34a5c",
            },
        ]

        def _function_factory(atomic_function, axis):
            return dict(
                func=lambda x: atomic_function["func"](x[axis]),
                description=atomic_function["description"].format(axis=axis),
                unit=atomic_function["unit"],
                feature_name=AV(
                    f'{sensor} power spectrum {atomic_function["name"]} {axis}'
                    f" axis",
                    f'{sensor.abbr}_ps_{atomic_function["name"].abbr}_{axis}',
                ),
            )

        functions = [
            _function_factory(atomic_function, axis)
            for atomic_function in atomic_functions
            for axis in columns
        ]

        super().__init__(
            data_set_id,
            functions,
            definition=FeatureValueDefinitionPrototype(data_type="float64"),
            level_filter=level_filter,
        )


class ComputeGravityRotationMatrices(TransformStep):
    r"""Compute a series of rotation matrices to align sensors to gravity.

    This transformation step creates a series of rotation matrices based on the
    gravity information contained in the accelerometer sensor. This allows to
    rotate other sensors on a desired orientation related to gravity. This is
    in particular of interest if we want to measure physical interactions with
    devices around the plane perpendicular to gravity.

    Parameters
    ----------
    target_gravity
        The target gravity vector, e.g. ``(-1, 0, 0)`` to create rotation
        matrices that rotate the x-axis of a device onto gravity.
    level_filter
        An optional :class:`~dispel.processing.level.LevelFilter` to determine the
        levels to be transformed. If no filter is provided, all levels will be
        transformed. The ``level_filter`` also accepts :class:`str`,
        :class:`~dispel.data.core.LevelId`\ s and lists of either and passes them
        to a :class:`~dispel.processing.level.LevelFilter` for convenience.
    """

    def __init__(
        self, data_set_id: str, target_gravity: Tuple[float, float, float], **kwargs
    ):
        def _transform_function(data: pd.DataFrame) -> pd.Series:
            return compute_rotation_matrices_quaternion(
                data[GRAVITY_COLUMNS], target_gravity
            )

        super().__init__(
            data_set_id,
            _transform_function,
            "gravity_rotation_matrices",
            [RawDataValueDefinition("rotation_matrix", "Rotation Matrix")],
            **kwargs,
        )


class RotateSensorWithGravityRotationMatrices(TransformStep):
    r"""Apply a series of rotation matrices to a sensor.

    This is a complementary step to :class:`ComputeGravityRotationMatrices` and
    applies the rotation matrices to the specified sensor.

    Parameters
    ----------
    data_set_id
        The id of the sensor data set to be rotated.
    columns
        The columns of the sensor data set to be considered in the rotation.
    level_filter
        An optional :class:`~dispel.processing.level.LevelFilter` to determine the
        levels to be transformed. If no filter is provided, all levels will be
        transformed. The ``level_filter`` also accepts :class:`str`,
        :class:`~dispel.data.core.LevelId`\ s and lists of either and passes them
        to a :class:`~dispel.processing.level.LevelFilter` for convenience.

    Examples
    --------
    Assuming you want to rotate the gyroscope vector onto gravity you can
    achieve this by chaining the following steps:

    .. doctest:: processing

        >>> from dispel.data.raw import DEFAULT_COLUMNS
        >>> from dispel.processing import process
        >>> from dispel.providers.generic.sensor import (
        ...     ComputeGravityRotationMatrices,
        ...     RotateSensorWithGravityRotationMatrices
        ... )
        >>> cols = DEFAULT_COLUMNS
        >>> steps = [
        ...     ComputeGravityRotationMatrices('accelerometer', (-1, 0, 0)),
        ...     RotateSensorWithGravityRotationMatrices('gyroscope', cols)
        ... ]
        >>> _ = process(reading, steps)  # doctest: +SKIP

    The results of the roation are available in the raw data set with the id
    ``<data_set_id>_rotated``:

    .. doctest:: processing
        :options: +NORMALIZE_WHITESPACE

        >>> level = reading.get_level(level_id)  # doctest: +SKIP
        >>> level.get_raw_data_set('gyroscope').data.head()  # doctest: +SKIP
                  x         y         z                      ts
        0  0.035728 -0.021515  0.014879 2020-05-04 17:31:38.574
        1 -0.012046  0.005010 -0.009029 2020-05-04 17:31:38.625
        2  0.006779  0.000761 -0.003253 2020-05-04 17:31:38.680
        3  0.032636 -0.020272 -0.021915 2020-05-04 17:31:38.729
        4  0.007495 -0.014061  0.012886 2020-05-04 17:31:38.779
        >>> level.get_raw_data_set(
        ...     'gyroscope_rotated'
        ... ).data.head()  # doctest: +SKIP
                  x         y         z
        0 -0.002309 -0.042509 -0.012182
        1 -0.003754  0.014983  0.003624
        2 -0.002237 -0.002116 -0.006901
        3 -0.030461 -0.021654 -0.023656
        4  0.001203 -0.019580  0.005924
    """

    def __init__(
        self,
        data_set_id: str,
        columns: Iterable[str],
        level_filter: Optional[LevelFilterType] = None,
    ):
        def _transform_function(
            sensor_df: pd.DataFrame, matrices: pd.DataFrame
        ) -> pd.DataFrame:
            return apply_rotation_matrices(
                matrices["rotation_matrix"], sensor_df[columns]
            )

        def _definition_factory(column: str) -> RawDataValueDefinition:
            return RawDataValueDefinition(column, f"{column} rotated")

        super().__init__(
            [data_set_id, "gravity_rotation_matrices"],
            _transform_function,
            f"{data_set_id}_rotated",
            [_definition_factory(column) for column in columns],
            level_filter=level_filter,
        )


class TransformUserAcceleration(TransformStep):
    r"""Format accelerometer data to ADS format if not already the case.

    Prior to formatting, linear acceleration and gravity are decoupled
    from acceleration.

    Parameters
    ----------
    level_filter
        An optional :class:`dispel.processing.level.LevelFilter` to determine the
        levels to be transformed. If no filter is provided, all levels will be
        transformed. The ``level_filter`` also accepts :class:`str`,
        :class:`~dispel.data.core.LevelId`\ s and lists of either and passes them
        to a :class:`~dispel.processing.level.LevelFilter` for convenience.
    """

    data_set_ids = "accelerometer"
    new_data_set_id = "acc"

    definitions = (
        [
            RawDataValueDefinition(
                f"userAcceleration{axis}",
                f"Linear Acceleration along the {axis} axis.",
                data_type="float",
            )
            for axis in "XYZ"
        ]
        + [
            RawDataValueDefinition(
                f"gravity{axis}",
                f"gravity component along the {axis} axis.",
                data_type="float",
            )
            for axis in "XYZ"
        ]
        + [RawDataValueDefinition("ts", "time index")]
    )

    @staticmethod
    def add_gravity(
        accelerometer: pd.DataFrame,
        level: Level,
        gravity: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Format gravity data to ADS format."""
        if gravity is None:
            cols = ["x", "y", "z"]
            raw_acc = level.get_raw_data_set("raw_accelerometer").data
            accelerometer = raw_acc
            if level.has_raw_data_set("attitude"):
                ori = level.get_raw_data_set("attitude").data
                ori_cols = ["w", "x", "y", "z"]
                lin_accelerometer, gravity = remove_gravity_component_ori(
                    accelerometer[cols].values, ori[ori_cols].values
                )
                lin_accelerometer = pd.DataFrame(lin_accelerometer, columns=cols)
                gravity = pd.DataFrame(gravity, columns=cols)
            else:
                lin_accelerometer, gravity = remove_gravity_component(
                    accelerometer[cols]
                )

            res = pd.DataFrame(
                {
                    "userAccelerationX": lin_accelerometer["x"],
                    "userAccelerationY": lin_accelerometer["y"],
                    "userAccelerationZ": lin_accelerometer["z"],
                }
            )
            res["gravityX"] = gravity["x"]
            res["gravityY"] = gravity["y"]
            res["gravityZ"] = gravity["z"]
            res["ts"] = accelerometer["ts"]
        else:
            # Merging on the timestamps vs. on the indexes
            acc_renamed = accelerometer.rename(
                mapper={
                    "x": "userAccelerationX",
                    "y": "userAccelerationY",
                    "z": "userAccelerationZ",
                },
                axis=1,
            )
            gravity_renamed = gravity.rename(
                mapper={"x": "gravityX", "y": "gravityY", "z": "gravityZ"}, axis=1
            )
            merged = acc_renamed.merge(gravity_renamed, how="outer")
            merged = merged.set_index("ts")
            merged_sorted = merged.sort_index()
            merged_sorted_interpolated = merged_sorted.interpolate(
                method="nearest", limit_direction="both"
            )
            res = merged_sorted_interpolated.loc[acc_renamed.ts].reset_index()
        return res.dropna()

    @staticmethod
    @transformation
    def _reformat(accelerometer: pd.DataFrame, level: Level) -> pd.DataFrame:
        target_cols = {
            f"{sensor}{axis}"
            for sensor in ("userAcceleration", "gravity")
            for axis in "XYZ"
        }
        if not target_cols.issubset(accelerometer.columns):
            try:
                return TransformUserAcceleration.add_gravity(
                    accelerometer, level, level.get_raw_data_set("gravity").data
                )
            except ValueError:
                # Happens in BDH pinch
                return TransformUserAcceleration.add_gravity(accelerometer, level)
        return accelerometer


class TransformGyroscope(TransformStep):
    r"""Format gyroscope data to ADS format if not already the case.

    On ADS format, the gyroscope is synchronized with the accelerometer. Here
    we make sure gyroscope is synchronized with the acc data set.

    Parameters
    ----------
    level_filter
        An optional :class:`dispel.processing.level.LevelFilter` to determine the
        levels to be transformed. If no filter is provided, all levels will be
        transformed. The ``level_filter`` also accepts :class:`str`,
        :class:`~dispel.data.core.LevelId`\ s and lists of either and passes them
        to a :class:`~dispel.processing.level.LevelFilter` for convenience.
    """

    data_set_ids = ["acc", "gyroscope"]
    new_data_set_id = "gyroscope"

    definitions = [
        RawDataValueDefinition(
            axis, f"Rotation speed along the {axis} axis.", data_type="float"
        )
        for axis in "xyz"
    ] + [RawDataValueDefinition("ts", "time index")]

    @staticmethod
    @transformation
    def _synchronize_gyroscope(
        accelerometer: pd.DataFrame, gyroscope: pd.DataFrame, reading: Reading
    ) -> pd.DataFrame:
        if isinstance(reading, BDHReading):
            # Merging on the timestamps vs. on the indexes
            acc_renamed = accelerometer.rename(
                mapper={
                    "x": "userAccelerationX",
                    "y": "userAccelerationY",
                    "z": "userAccelerationZ",
                },
                axis=1,
            )
            return pd.merge_asof(acc_renamed, gyroscope, on="ts", direction="nearest")[
                ["ts", "x", "y", "z"]
            ]
        return gyroscope


class EuclideanNorm(TransformStep):
    r"""Compute euclidean norm of the specified columns of a raw data set.

    Parameters
    ----------
    data_set_id
        The data set id of the data set on which the method is to be applied
    columns
        The columns to be considered during the method application.
    drop_nan
        ```True`` if NaN values are to be dropped after transformation.
    level_filter
        An optional :class:`dispel.processing.level.LevelFilter` to determine the
        levels to be transformed. If no filter is provided, all levels will be
        transformed. The ``level_filter`` also accepts :class:`str`,
        :class:`~dispel.data.core.LevelId`\ s and lists of either and passes them
        to a :class:`dispel.processing.level.LevelIdFilter` for convenience.
    """

    def __init__(
        self,
        data_set_id: str,
        columns: Optional[List[str]] = None,
        drop_nan: bool = False,
        level_filter: Optional[LevelFilterType] = None,
    ):
        columns = columns or DEFAULT_COLUMNS

        def _transform_function(data: pd.DataFrame) -> pd.Series:
            res = euclidean_norm(data[columns])
            if drop_nan:
                return res.dropna()
            return res

        definition = RawDataValueDefinition(
            "norm", f"euclidean_norm computed on {columns}"
        )

        super().__init__(
            data_set_id,
            _transform_function,
            f"{data_set_id}_euclidean_norm",
            [definition],
            level_filter=level_filter,
        )


class AddGravityAndScale(TransformStep):
    """Add gravity to userAcceleration and scale to m/s^2.

    The step expects a unique data set id for `data_set_ids` pointing to a
    data frame containing both acceleration and gravity with a
    :class:`pandas.DatetimeIndex` index.
    """

    definitions = [
        RawDataValueDefinition(f"acceleration_{ax}", f"acceleration_{ax}")
        for ax in DEFAULT_COLUMNS
    ]

    @transformation
    def _transform(self, data) -> pd.DataFrame:
        acc = {}
        for i, ax in enumerate(DEFAULT_COLUMNS):
            acc[f"acceleration_{ax}"] = (
                data[ACCELEROMETER_COLUMNS[i]] + data[GRAVITY_COLUMNS[i]]
            )
        return pd.DataFrame(acc, index=data.index) * GRAVITY_CONSTANT

    def get_new_data_set_id(self):
        """Overwrite new_data_set_id."""
        return f"{self.data_set_ids}_g"


class TransformFindZeroCrossings(TransformStep):
    """Find zero crossings in the signal.

    To find the zeros, the function identifies the sign change in the signal by
    differentiating `data > 0`.

    Attributes
    ----------
    column
        The column to be used to find the zero crossings in the data set

    Parameters
    ----------
    column
        See :data:FindZeroCrossings.column`.
    """

    column: str

    definitions: List[RawDataValueDefinition] = [
        RawDataValueDefinition(
            "zero_crossings", "Zero-crossings of the signal.", data_type="float"
        )
    ]

    def __init__(self, *args, column: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.column = column or self.column

    @transformation
    def _find_zero_crossings(self, data: pd.DataFrame) -> pd.DataFrame:
        return find_zero_crossings(data, self.column)
