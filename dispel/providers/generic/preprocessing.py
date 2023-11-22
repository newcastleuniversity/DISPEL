"""Core functionalities to preprocess signal data."""
from typing import Iterable, List, Optional, Set, Tuple

from dispel.data.levels import Level
from dispel.data.raw import DEFAULT_COLUMNS, GRAVITY_COLUMNS
from dispel.processing import ProcessingStep
from dispel.processing.level import (
    DefaultLevelFilter,
    LevelFilter,
    LevelFilterType,
    LevelIdFilter,
    ProcessingStepGroup,
)
from dispel.processing.modalities import LimbModality, SensorModality
from dispel.processing.transform import Apply
from dispel.providers.generic.sensor import (
    ComputeGravityRotationMatrices,
    Resample,
    RotateSensorWithGravityRotationMatrices,
    SetTimestampIndex,
    TransformGyroscope,
    TransformUserAcceleration,
)
from dispel.signal.filter import butterworth_high_pass_filter, savgol_filter
from dispel.signal.sensor import check_amplitude, detrend_signal


class FilterSensorNoise(Apply):
    r"""Apply a filter that will remove any sensor noise into a given dataset.

    This filter is a Savitzky-Golay one.

    Parameters
    ----------
    data_set_id
        The data set id on which the transformation is to be performed ('accelerometer',
        'gyroscope').
    columns
        The columns onto which the filtering step has to be applied.
    kwargs
        Additional arguments that are passed to the
        :meth:`~dispel.processing.core.ProcessingStep.process` function of each step. This
        allows to provide additional values, such as placeholder values in value
        definitions to the actual processing function.

    Notes
    -----
    The Savitzky-Golay is tuned as in [Martinez et. al. 2012]_ to remove sensor noise
    and to smooth the signal. The windows size is thus set up to 41 points and the
    filter is of order-3.

    """

    def __init__(self, data_set_id: str, columns: Optional[List[str]] = None, **kwargs):
        columns = columns or DEFAULT_COLUMNS

        super().__init__(
            data_set_id=data_set_id,
            method=savgol_filter,
            method_kwargs=dict(window=41, order=3),
            columns=columns,
            new_data_set_id=f"{data_set_id}_svgf",
            drop_nan=True,
            **kwargs,
        )


class FilterPhysiologicalNoise(Apply):
    r"""Apply a filter that will remove any physiological noise into a dataset.

    This filter is a butterworth high-pass one.

    Parameters
    ----------
    data_set_id
        The data set id on which the transformation is to be performed ('accelerometer',
        'gyroscope').
    columns
        The columns onto which the filtering step has to be applied.
    sampling_frequency
        Optional the initial sampling frequency.
    kwargs
        Additional arguments that are passed to the
        :meth:`~dispel.processing.core.ProcessingStep.process` function of each step. This
        allows to provide additional values, such as placeholder values in value
        definitions to the actual processing function.

    Notes
    -----
    The Butterwoth highpass filter is tuned as in [Martinez et. al. 2012]_ to remove
    physiological noise. The cut-off of is of 0.2HZ which is the standard breath
    frequency.

    .. [Martinez et. al. 2012] MARTINEZ-MENDEZ, Rigoberto, SEKINE,
           Masaki, et TAMURA, Toshiyo.
           Postural sway parameters using a triaxial accelerometer: comparing elderly
           and young healthy adults. Computer methods in biomechanics and biomedical
           engineering, 2012, vol. 15, no 9, p. 899-910.

    """

    def __init__(
        self,
        data_set_id: str,
        columns: Optional[List[str]] = None,
        sampling_frequency: Optional[float] = None,
        **kwargs,
    ):
        columns = columns or DEFAULT_COLUMNS

        super().__init__(
            data_set_id=data_set_id,
            method=butterworth_high_pass_filter,
            method_kwargs=dict(
                order=2, cutoff=0.3, freq=sampling_frequency, zero_phase=True
            ),
            columns=columns,
            new_data_set_id=f"{data_set_id}_bhpf",
            drop_nan=True,
            **kwargs,
        )


class Detrend(Apply):
    r"""A detrending preprocessing step according a given data set.

    Parameters
    ----------
    data_set_id
        The data set id on which the transformation is to be performed ('accelerometer',
        'gyroscope').
    columns
        The columns onto which the detrending steps have to be applied.
    kwargs
        Additional arguments that are passed to the
        :meth:`~dispel.processing.core.ProcessingStep.process` function of each step. This
        allows to provide additional values, such as placeholder values in value
        definitions to the actual processing function.

    """

    def __init__(self, data_set_id: str, columns: Optional[List[str]] = None, **kwargs):
        columns = columns or DEFAULT_COLUMNS

        super().__init__(
            data_set_id=data_set_id,
            method=detrend_signal,
            columns=columns,
            new_data_set_id=f"{data_set_id}_detrend",
            drop_nan=True,
            **kwargs,
        )


class AmplitudeRangeFilter(LevelFilter):
    r"""Filter aberrant signal amplitude.

    Parameters
    ----------
    data_set_id
        The data set id on which the transformation is to be performed ('accelerometer',
        'gyroscope').
    max_amplitude
        A float which is the maximum expected amplitude values.
    min_amplitude
        A float which is the minimum expected amplitude values.
    columns
        The columns onto which the detrending steps have to be applied.
    """

    def __init__(
        self,
        data_set_id: str,
        max_amplitude: float,
        min_amplitude: float,
        columns: Optional[List[str]] = None,
    ):
        self.data_set_id = data_set_id
        self.columns = columns
        self.max_amplitude = max_amplitude
        self.min_amplitude = min_amplitude

    def repr(self):
        """Get representation of the filter."""
        return f"only {self.data_set_id} signal with acceptable amplitude>"

    def filter(self, levels: Iterable[Level]) -> Set[Level]:
        """Filter levels with acceptable signal amplitude."""

        def _amplitude_filter(level: Level):
            if level.has_raw_data_set(self.data_set_id):
                data = level.get_raw_data_set(self.data_set_id).data
                if self.columns:
                    data = data[self.columns]
                return check_amplitude(data, self.min_amplitude, self.max_amplitude)
            return True

        return set(filter(_amplitude_filter, levels))


class RotateFrame(ProcessingStepGroup):
    r"""A changing referential preprocessing step according a given data set.

    Parameters
    ----------
    data_set_id
        The data set id on which the transformation is to be performed.
    gravity_data_set_id
        The dataset id containing the gravity components.
    frame
        The new desired frame.
    columns
        The columns onto which the resampling steps have to be applied.
    kwargs
        Additional arguments that are passed to the
        :meth:`~dispel.processing.core.ProcessingStep.process` function of each step.
        This allows to provide additional values, such as placeholder values in value
        definitions to the actual processing function.
    """

    def __init__(
        self,
        data_set_id: str,
        gravity_data_set_id: str,
        frame: Tuple[int, int, int],
        columns: Optional[List[str]] = None,
        **kwargs,
    ):
        columns = columns or DEFAULT_COLUMNS

        steps: List[ProcessingStep] = [
            ComputeGravityRotationMatrices(
                gravity_data_set_id, frame, storage_error="ignore"
            ),
            RotateSensorWithGravityRotationMatrices(
                data_set_id,
                columns,
            ),
        ]

        super().__init__(
            steps,
            **kwargs,
        )


class PreprocessingSteps(ProcessingStepGroup):
    r"""A changing referential preprocessing step according a given data set.

    Parameters
    ----------
    data_set_id
        The data set id on which the transformation is to be performed.
    limb
        The modality regarding if the exercise is upper or lower limb.
    sensor
        The modality regarding the type of sensor either accelerometer or gyroscope.
    resample_freq
        Optionally, the frequency to which resample the data during the resample step.
    columns
        Optionally, the columns on which the preprocessing steps need to be applied.
    level_filter
        An optional :class:`~dispel.processing.level.LevelFilter` to determine the levels
        to be transformed. If no filter is provided, all levels will be transformed. The
        ``level_filter`` also accepts :class:`str`, :class:`~dispel.data.core.LevelId`\ s
        and lists of either and passes them to a
        :class:`~dispel.processing.level.LevelIdFilter` for convenience.
    """

    def __init__(
        self,
        data_set_id: str,
        limb: LimbModality,
        sensor: SensorModality,
        resample_freq: Optional[float] = None,
        columns: Optional[List[str]] = None,
        level_filter: LevelFilterType = DefaultLevelFilter(),
    ):
        columns = columns or DEFAULT_COLUMNS
        extra_columns = []

        if not isinstance(level_filter, LevelFilter):
            level_filter = LevelIdFilter(level_filter)

        # Need to be computed even if only gyroscope signals are preprocessed to make
        # sure `acc` data set is available to compute gravity rotation matrices
        steps: List[ProcessingStep] = [
            TransformUserAcceleration(storage_error="ignore"),
            TransformGyroscope(storage_error="overwrite"),
        ]

        if sensor == SensorModality.ACCELEROMETER:
            data_set_id = "acc"
            extra_columns = GRAVITY_COLUMNS

        steps += [
            SetTimestampIndex(
                data_set_id, list(set(columns).union(extra_columns)), duplicates="first"
            )
        ]

        if limb == LimbModality.LOWER_LIMB:
            steps += [
                RotateFrame(
                    data_set_id=f"{data_set_id}_ts",
                    gravity_data_set_id="acc_ts",
                    frame=(-1, 0, 0),
                    columns=columns,
                ),
                Resample(
                    data_set_id=f"{data_set_id}_ts_rotated",
                    freq=resample_freq,
                    aggregations=["mean", "ffill"],
                    columns=columns,
                ),
                Detrend(
                    data_set_id=f"{data_set_id}_ts_rotated_resampled", columns=columns
                ),
            ]
        else:
            steps += [
                Resample(
                    data_set_id=f"{data_set_id}_ts",
                    freq=resample_freq,
                    aggregations=["mean", "ffill"],
                    columns=columns,
                ),
                Detrend(data_set_id=f"{data_set_id}_ts_resampled", columns=columns),
            ]

        super().__init__(steps, level_filter=level_filter)
