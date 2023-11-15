"""Core utility function used in del din.

The code implementation is inspired by GaitPy.
"""

import warnings
from enum import IntEnum
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import pywt
from scipy import integrate
from scipy.signal import find_peaks

from dispel.data.features import FeatureValueDefinitionPrototype
from dispel.data.levels import Context, Level
from dispel.data.raw import DEFAULT_COLUMNS, RawDataValueDefinition
from dispel.data.validators import BETWEEN_MINUS_ONE_AND_ONE, GREATER_THAN_ZERO
from dispel.data.values import AbbreviatedValue as AV
from dispel.processing import ProcessingStep
from dispel.processing.data_set import transformation
from dispel.processing.extract import (
    DEFAULT_AGGREGATIONS,
    DEFAULT_AGGREGATIONS_Q95,
    DEFAULT_AGGREGATIONS_Q95_CV,
    AggregateRawDataSetColumn,
    ExtractStep,
)
from dispel.processing.level import ProcessingStepGroup
from dispel.processing.transform import TransformStep
from dispel.providers.generic.preprocessing import Detrend, Resample, RotateFrame
from dispel.providers.generic.sensor import AddGravityAndScale
from dispel.providers.generic.tasks.gait.bout_strategy import BoutStrategyModality
from dispel.providers.generic.tasks.gait.core import (
    DEF_BOUT_ID,
    DetectStepsProcessingBase,
    DetectStepsWithoutBoutsBase,
    ExtractPowerBoutDivSteps,
    ExtractStepIntensityAll,
    ExtractStepPowerAll,
    ExtractStepRegularity,
    ExtractStrideRegularity,
    FootUsed,
    GaitBoutAggregateStep,
    GaitBoutExtractStep,
    StepEvent,
    power_bout_div_steps,
)
from dispel.signal.core import get_sampling_rate_idx
from dispel.signal.filter import butterworth_low_pass_filter
from dispel.signal.sensor import detrend_signal


class FootContact(IntEnum):
    """Generic foot contact for peak detection."""

    INITIAL_CONTACT = 1
    FINAL_CONTACT = 2


INITIAL_CONTACT_PROMINENCE = 5
r"""Prominence of a vertical acceleration peak indicating a heel strike or
foot initial contact."""

FINAL_CONTACT_PROMINENCE = 10
r"""Prominence of a vertical acceleration peak indicating a toe off or
foot final contact."""

STEP_LENGTH_HEIGHT_RATIO = 0.41
r"""Average step length relative to subject height."""

SENSOR_HEIGHT_RATIO = 0.53
r"""Height of the sensor relative to subject height."""

DEFAULT_SUBJECT_HEIGHT = 1.7
r"""A default height that we will use to compute spatial features until we
have access to the correct height."""

DEFAULT_SENSOR_HEIGHT = SENSOR_HEIGHT_RATIO * DEFAULT_SUBJECT_HEIGHT
r"""A default sensor height given sensor_height ratio and subject height."""

GAIT_CYCLE_FORWARD_IC = 2.25
r"""Maximum allowable time (seconds) between initial contact of same foot"""

LOADING_FORWARD_IC = GAIT_CYCLE_FORWARD_IC * 0.2
r"""Maximum time (seconds) for loading phase."""

STANCE_FORWARD_IC = (GAIT_CYCLE_FORWARD_IC / 2) + LOADING_FORWARD_IC
r"""Maximum time (seconds) for stance phase."""

MIN_SAMPLES_HEIGHT_CHANGES_COM = 15
r"""The minimum number of samples to compute changes of height of the center
of mass."""

MIN_FREQ_TO_FILTER = 40
r"""The minimum sampling frequency at which we start low pass filter before
computing the wavelets."""

IC_DEF = RawDataValueDefinition(
    id_="IC",
    name="Initial Contact",
    data_type="datetime64[ms]",
    description="Initial contact timestamp.",
)

FC_DEF = RawDataValueDefinition(
    id_="FC",
    name="Final Contact",
    data_type="datetime64[ms]",
    description="Final contact timestamp.",
)

FC_OPP_FOOT_DEF = RawDataValueDefinition(
    id_="FC_opp_foot",
    name="Final Contact of the opposite foot",
    data_type="datetime64[ms]",
    description="Final contact of the opposite foot to the one "
    "performing the initial contact timestamp.",
)

GAIT_CYCLE_DEF = RawDataValueDefinition(
    id_="Gait_Cycle",
    name="Gait Cycle",
    data_type="bool",
    description="Gait Cycle",
)

STEPS_DEF = RawDataValueDefinition(
    id_="steps",
    name="steps",
    description="Number of steps",
    data_type="int",
)

STRIDE_DUR_DEF = RawDataValueDefinition(
    id_="stride_dur",
    name="stride duration",
    description="Stride duration is defined as the time elapsed "
    "between the first contact of two consecutive "
    "footsteps of the same foot.",
    data_type="float64",
    unit="s",
    precision=3,
)

STRIDE_DUR_ASYM_DEF = RawDataValueDefinition(
    id_="stride_dur_asym",
    name="stride duration asymmetry",
    data_type="float64",
    description="Stride duration asymmetry is defined as the absolute "
    "difference of the stride duration between the right "
    "and left foot.",
    unit="s",
    precision=3,
)

STRIDE_DUR_ASYM_LN_DEF = RawDataValueDefinition(
    id_="stride_dur_asym_ln",
    name="stride duration asymmetry (ln)",
    data_type="float64",
    description="Stride duration asymmetry ln is defined as the natural "
    "logarithm of the ratio of the minimum duration to the "
    "maximum duration of two consecutive strides.",
    precision=3,
)

STEP_DUR_DEF = RawDataValueDefinition(
    id_="step_dur",
    name="step dur",
    data_type="float64",
    description="Step duration is the time elapsed between two "
    "consecutive footsteps. More specifically between two "
    "consecutive initial contact events.",
    unit="s",
    precision=3,
)

STEP_DUR_ASYM_DEF = RawDataValueDefinition(
    id_="step_dur_asym",
    name="step duration asymmetry",
    data_type="float64",
    description="Step duration asymmetry is defined as the absolute "
    "difference of the step duration between the right "
    "and left foot.",
    unit="s",
    precision=3,
)

STEP_DUR_ASYM_LN_DEF = RawDataValueDefinition(
    id_="step_dur_asym_ln",
    name="step duration asymmetry (ln)",
    data_type="float64",
    description="Step duration asymmetry ln is defined as the natural "
    "logarithm of the ratio of the minimum duration to the "
    "maximum duration of two consecutive steps.",
    precision=3,
)

CADENCE_DEF = RawDataValueDefinition(
    id_="cadence",
    name="cadence",
    data_type="float64",
    description="Cadence is defined as the instantaneous number of "
    "steps per minutes.",
    precision=3,
)

INIT_DOUBLE_SUPP_DEF = RawDataValueDefinition(
    id_="init_double_supp",
    name="initial double support",
    data_type="float64",
    description="Initial double support phase is the sub-phase "
    "between heel contact of the phase to contralateral "
    "foot-off.",
    unit="s",
    precision=3,
)

INIT_DOUBLE_SUPP_ASYM_DEF = RawDataValueDefinition(
    id_="init_double_supp_asym",
    name="initial double support asymmetry",
    data_type="float64",
    description="Initial double support asymmetry is defined as the "
    "absolute difference of the initial double support "
    "duration between the right and left foot.",
    unit="s",
    precision=3,
)

TERM_DOUBLE_SUPP_DEF = RawDataValueDefinition(
    id_="term_double_supp",
    name="terminal double support",
    data_type="float64",
    description="Terminal double support phase is the  sub-phase "
    "from contralateral foot-on to the toe-off.",
    unit="s",
    precision=3,
)

TERM_DOUBLE_SUPP_ASYM_DEF = RawDataValueDefinition(
    id_="term_double_supp_asym",
    name="terminal double support asymmetry",
    data_type="float64",
    description="Terminal double support asymmetry is defined "
    "as the absolute difference of the terminal"
    "double support duration between the right and"
    "left foot.",
    unit="s",
    precision=3,
)

DOUBLE_SUPP_DEF = RawDataValueDefinition(
    id_="double_supp",
    name="double support",
    data_type="float64",
    description="Double support phase is defined as the sum of"
    "the initial and terminal double support "
    "phase.",
    unit="s",
    precision=3,
)

DOUBLE_SUPP_ASYM_DEF = RawDataValueDefinition(
    id_="double_supp_asym",
    name="double support asymmetry",
    data_type="float64",
    description="Double support asymmetry is defined "
    "as the absolute difference of the double "
    "support duration between the right and left "
    "foot.",
    unit="s",
    precision=3,
)

SINGLE_LIMB_SUPP_DEF = RawDataValueDefinition(
    id_="single_limb_supp",
    name="single limb support",
    data_type="float64",
    description="Single limb support represents a phase in "
    "the gait cycle when the body weight is "
    "entirely supported by one limb, while the "
    "contra-lateral limb swings forward.",
    unit="s",
    precision=3,
)

SINGLE_LIMB_SUPP_ASYM_DEF = RawDataValueDefinition(
    id_="single_limb_supp_asym",
    name="single limb support asymmetry",
    data_type="float64",
    description="Single limb support asymmetry is defined "
    "as the absolute difference of the single "
    "limb support duration between the right "
    "and left foot.",
    unit="s",
    precision=3,
)

STANCE_DEF = RawDataValueDefinition(
    id_="stance",
    name="stance",
    data_type="float64",
    description="The stance phase of gait begins when the "
    "foot first touches the ground and ends when "
    "the same foot leaves the ground.",
    unit="s",
    precision=3,
)

STANCE_ASYM_DEF = RawDataValueDefinition(
    id_="stance_asym",
    name="stance asymmetry",
    data_type="float64",
    description="Stance asymmetry is defined as the absolute "
    "difference of the stance duration between "
    "the right and left foot.",
    unit="s",
    precision=3,
)

SWING_DEF = RawDataValueDefinition(
    id_="swing",
    name="swing",
    data_type="float64",
    description="The swing phase of gait begins when the foot "
    "first leaves the ground and ends when the "
    "same foot touches the ground again.",
    unit="s",
    precision=3,
)

SWING_ASYM_DEF = RawDataValueDefinition(
    id_="swing_asym",
    name="swing asymmetry",
    data_type="float64",
    description="Swing asymmetry is defined as the absolute "
    "difference of the swing duration between "
    "the right and left foot.",
    unit="s",
    precision=3,
)

STEP_LEN_DEF = RawDataValueDefinition(
    id_="step_len",
    name="step length",
    data_type="float64",
    description="Step length is the distance between the "
    "point of initial contact of one foot and the "
    "point of initial contact of the opposite "
    "foot.",
    unit="m",
    precision=3,
)

STEP_LEN_ASYM_DEF = RawDataValueDefinition(
    id_="step_len_asym",
    name="step length asymmetry",
    data_type="float64",
    description="Step length asymmetry is defined as the "
    "absolute difference of the step length "
    "between the right and left foot.",
    unit="m",
    precision=3,
)

STEP_LEN_ASYM_LN_DEF = RawDataValueDefinition(
    id_="step_len_asym_ln",
    name="step length asymmetry ln",
    data_type="float64",
    description="Step length asymmetry ln is defined as the natural logarithm"
    "of the ratio in the length between consecutive steps.",
    precision=3,
)

STRIDE_LEN_DEF = RawDataValueDefinition(
    id_="stride_len",
    name="stride length",
    data_type="float64",
    description="Stride length is the distance covered when "
    "you take two steps, one with each foot.",
    unit="m",
    precision=3,
)

STRIDE_LEN_ASYM_DEF = RawDataValueDefinition(
    id_="stride_len_asym",
    name="stride length asymmetry",
    data_type="float64",
    description="Stride length asymmetry is defined as the "
    "absolute difference of the stride length "
    "between the right and left foot.",
    unit="m",
    precision=3,
)

STRIDE_LEN_ASYM_LN_DEF = RawDataValueDefinition(
    id_="stride_len_asym_ln",
    name="stride length asymmetry ln",
    data_type="float64",
    description="Stride length asymmetry ln is defined as the natural"
    "logarithm of the ratio in the length between consecutive "
    "strides.",
    precision=3,
)

GAIT_SPEED_DEF = RawDataValueDefinition(
    id_="gait_speed",
    name="gait speed",
    data_type="float64",
    description="Gait speed is defined as the instantaneous "
    "walking speed in m/s. It is computed as the "
    "length of a stride divided by the duration.",
    unit="m/s",
    precision=3,
)

CWT_FEATURES_DESC_MAPPING = {
    str(STEPS_DEF.id): STEPS_DEF.description,
    str(STRIDE_DUR_DEF.id): STRIDE_DUR_DEF.description,
    str(STRIDE_DUR_ASYM_DEF.id): STRIDE_DUR_ASYM_DEF.description,
    str(STRIDE_DUR_ASYM_LN_DEF.id): STRIDE_DUR_ASYM_LN_DEF.description,
    str(STEP_DUR_DEF.id): STEP_DUR_DEF.description,
    str(STEP_DUR_ASYM_DEF.id): STEP_DUR_ASYM_DEF.description,
    str(STEP_DUR_ASYM_LN_DEF.id): STEP_DUR_ASYM_LN_DEF.description,
    str(CADENCE_DEF.id): CADENCE_DEF.description,
    str(INIT_DOUBLE_SUPP_DEF.id): INIT_DOUBLE_SUPP_DEF.description,
    str(INIT_DOUBLE_SUPP_ASYM_DEF.id): INIT_DOUBLE_SUPP_ASYM_DEF.description,
    str(TERM_DOUBLE_SUPP_DEF.id): TERM_DOUBLE_SUPP_DEF.description,
    str(TERM_DOUBLE_SUPP_ASYM_DEF.id): TERM_DOUBLE_SUPP_ASYM_DEF.description,
    str(DOUBLE_SUPP_DEF.id): DOUBLE_SUPP_DEF.description,
    str(DOUBLE_SUPP_ASYM_DEF.id): DOUBLE_SUPP_ASYM_DEF.description,
    str(SINGLE_LIMB_SUPP_DEF.id): SINGLE_LIMB_SUPP_DEF.description,
    str(SINGLE_LIMB_SUPP_ASYM_DEF.id): SINGLE_LIMB_SUPP_ASYM_DEF.description,
    str(STANCE_DEF.id): STANCE_DEF.description,
    str(STANCE_ASYM_DEF.id): STANCE_ASYM_DEF.description,
    str(SWING_DEF.id): SWING_DEF.description,
    str(SWING_ASYM_DEF.id): SWING_ASYM_DEF.description,
    str(STEP_LEN_DEF.id): STEP_LEN_DEF.description,
    str(STEP_LEN_ASYM_DEF.id): STEP_LEN_ASYM_DEF.description,
    str(STEP_LEN_ASYM_LN_DEF.id): STEP_LEN_ASYM_LN_DEF.description,
    str(STRIDE_LEN_DEF.id): STRIDE_LEN_DEF.description,
    str(STRIDE_LEN_ASYM_DEF.id): STRIDE_LEN_ASYM_DEF.description,
    str(STRIDE_LEN_ASYM_LN_DEF.id): STRIDE_LEN_ASYM_LN_DEF.description,
    str(GAIT_SPEED_DEF.id): GAIT_SPEED_DEF.description,
}
r"""Mapping features to description to help fill the description of cwt
features when aggregating columns."""

CWT_FEATURES_PRECISION_MAPPING = {
    str(STEPS_DEF.id): STEPS_DEF.precision,
    str(STRIDE_DUR_DEF.id): STRIDE_DUR_DEF.precision,
    str(STRIDE_DUR_ASYM_DEF.id): STRIDE_DUR_ASYM_DEF.precision,
    str(STRIDE_DUR_ASYM_LN_DEF.id): STRIDE_DUR_ASYM_LN_DEF.precision,
    str(STEP_DUR_DEF.id): STEP_DUR_DEF.precision,
    str(STEP_DUR_ASYM_DEF.id): STEP_DUR_ASYM_DEF.precision,
    str(STEP_DUR_ASYM_LN_DEF.id): STEP_DUR_ASYM_LN_DEF.precision,
    str(CADENCE_DEF.id): CADENCE_DEF.precision,
    str(INIT_DOUBLE_SUPP_DEF.id): INIT_DOUBLE_SUPP_DEF.precision,
    str(INIT_DOUBLE_SUPP_ASYM_DEF.id): INIT_DOUBLE_SUPP_ASYM_DEF.precision,
    str(TERM_DOUBLE_SUPP_DEF.id): TERM_DOUBLE_SUPP_DEF.precision,
    str(TERM_DOUBLE_SUPP_ASYM_DEF.id): TERM_DOUBLE_SUPP_ASYM_DEF.precision,
    str(DOUBLE_SUPP_DEF.id): DOUBLE_SUPP_DEF.precision,
    str(DOUBLE_SUPP_ASYM_DEF.id): DOUBLE_SUPP_ASYM_DEF.precision,
    str(SINGLE_LIMB_SUPP_DEF.id): SINGLE_LIMB_SUPP_DEF.precision,
    str(SINGLE_LIMB_SUPP_ASYM_DEF.id): SINGLE_LIMB_SUPP_ASYM_DEF.precision,
    str(STANCE_DEF.id): STANCE_DEF.precision,
    str(STANCE_ASYM_DEF.id): STANCE_ASYM_DEF.precision,
    str(SWING_DEF.id): SWING_DEF.precision,
    str(SWING_ASYM_DEF.id): SWING_ASYM_DEF.precision,
    str(STEP_LEN_DEF.id): STEP_LEN_DEF.precision,
    str(STEP_LEN_ASYM_DEF.id): STEP_LEN_ASYM_DEF.precision,
    str(STEP_LEN_ASYM_LN_DEF.id): STEP_LEN_ASYM_LN_DEF.precision,
    str(STRIDE_LEN_DEF.id): STRIDE_LEN_DEF.precision,
    str(STRIDE_LEN_ASYM_DEF.id): STRIDE_LEN_ASYM_DEF.precision,
    str(STRIDE_LEN_ASYM_LN_DEF.id): STRIDE_LEN_ASYM_LN_DEF.precision,
    str(GAIT_SPEED_DEF.id): GAIT_SPEED_DEF.precision,
}
r"""Mapping features to precision to map the precision of cwt
features when aggregating columns."""

CWT_FEATURES_DEF = [
    STEPS_DEF,
    STRIDE_DUR_DEF,
    STRIDE_DUR_ASYM_DEF,
    STRIDE_DUR_ASYM_LN_DEF,
    STEP_DUR_DEF,
    STEP_DUR_ASYM_DEF,
    STEP_DUR_ASYM_LN_DEF,
    CADENCE_DEF,
    INIT_DOUBLE_SUPP_DEF,
    INIT_DOUBLE_SUPP_ASYM_DEF,
    TERM_DOUBLE_SUPP_DEF,
    TERM_DOUBLE_SUPP_ASYM_DEF,
    DOUBLE_SUPP_DEF,
    DOUBLE_SUPP_ASYM_DEF,
    SINGLE_LIMB_SUPP_DEF,
    SINGLE_LIMB_SUPP_ASYM_DEF,
    STANCE_DEF,
    STANCE_ASYM_DEF,
    SWING_DEF,
    SWING_ASYM_DEF,
    STEP_LEN_DEF,
    STEP_LEN_ASYM_DEF,
    STEP_LEN_ASYM_LN_DEF,
    STRIDE_LEN_DEF,
    STRIDE_LEN_ASYM_DEF,
    STRIDE_LEN_ASYM_LN_DEF,
    GAIT_SPEED_DEF,
]


def _detect_peaks(y: Iterable, prominence: float) -> Iterable[int]:
    """
    Find peaks inside a signal based on peak properties.

    This function takes a 1-D array and finds all local maxima by
    simple comparison of neighboring values. Then a subset of these
    peaks are selected by specifying conditions on the prominence of the peak.

    Parameters
    ----------
    y
        A signal with peaks.
    prominence
        The prominence of a peak measures how much a peak stands out from the
        surrounding baseline of the signal and is defined as the vertical
        distance between the peak and its lowest contour line.

    Returns
    -------
    Iterable[int]
        Indices of peaks in `y` that satisfy the prominence conditions.
    """
    peaks, _ = find_peaks(y, prominence=prominence)
    return peaks


def _detrend_and_low_pass(data: pd.Series, sampling_rate: float) -> pd.Series:
    """
    Detrend and eventually filter the data.

    Before computing the wavelet, in gaitpy library the authors decide
    to always detrend the vertical acceleration signal and eventually
    low-pass filter if the sampling frequency is greater than 40 Hz.

    Parameters
    ----------
    data
        Any pandas series regularly sampled.
    sampling_rate
        The sampling rate of the time series.

    Returns
    -------
    pandas.Series
        The same series detrended and eventually low-pass filtered if its
        sampling frequency is greater than 40 Hz.
    """
    # Detrend data
    detrended_data = detrend_signal(data)

    # Low pass filter if greater than 40 hz
    if sampling_rate >= MIN_FREQ_TO_FILTER:
        return butterworth_low_pass_filter(
            data=detrended_data, cutoff=20, order=4, freq=sampling_rate, zero_phase=True
        )
    return detrended_data


def _continuous_wavelet_transform(
    data: pd.Series, ic_prom: float, fc_prom: float
) -> Tuple[Iterable, Iterable, Iterable, Iterable]:
    """
    Find the indices of the Initial contact or Final contact of the foot.

    Parameters
    ----------
    data
        The vertical acceleration
    ic_prom
        Initial contact prominence. For more info, see _detect_peaks prominence
        definition.
    fc_prom
        Final contact prominence. For more info, see _detect_peaks prominence
        definition.

    Returns
    -------
    Tuple[Iterable, Iterable, Iterable, Iterable]:
        The indices of the initial contacts and final contacts in `y_accel` and
        the wavelet transforms.
    """
    # Get sampling rate
    sampling_rate = get_sampling_rate_idx(data)

    # CWT wavelet parameters from GaitPy
    # The smaller the scale factor, the more “compressed” the wavelet.
    # Conversely, the larger the scale, the more stretched the wavelet.
    scale_cwt_ic = float(sampling_rate) / 5.0
    scale_cwt_fc = float(sampling_rate) / 6.0
    wavelet_name = "gaus1"

    # Detrend and low pass
    filtered_data = _detrend_and_low_pass(data, sampling_rate)

    # Numerical integration of vertical acceleration
    integrated_data = integrate.cumtrapz(-filtered_data)

    # Gaussian continuous wavelet transform
    cwt_ic, _ = pywt.cwt(
        data=integrated_data, scales=scale_cwt_ic, wavelet=wavelet_name
    )
    differentiated_data = cwt_ic[0]

    # initial contact (heel strike) peak detection
    ic_peaks = _detect_peaks(pd.Series(-differentiated_data), ic_prom)

    # Gaussian continuous wavelet transform
    cwt_fc, _ = pywt.cwt(
        data=-differentiated_data, scales=scale_cwt_fc, wavelet=wavelet_name
    )
    re_differentiated_data = cwt_fc[0]

    # final contact (toe off) peak detection
    fc_peaks = _detect_peaks(pd.Series(re_differentiated_data), fc_prom)
    return ic_peaks, fc_peaks, differentiated_data, re_differentiated_data


def compute_asymmetry(data: pd.Series) -> pd.Series:
    """Compute absolute asymmetry.

    Parameters
    ----------
    data
        The cyclic property dataset to be analyzed.

    Returns
    -------
    pandas.Series
        A pandas Series containing the absolute difference between consecutive
        items of the dataset.
    """
    return abs(data.shift(-1) - data)


def compute_ln_asymmetry(data: pd.Series) -> pd.Series:
    """Compute natural logarithm of asymmetry ratio.

    Parameters
    ----------
    data
        The cyclic property dataset to be analyzed.

    Returns
    -------
    pandas.Series
        A pandas Series containing the natural logarithm of the ratio of the
        minimum and maximum cyclic property, denoting normalized asymmetry.
    """
    shifted = pd.concat([data.shift(-1), data], axis=1)
    min_ = shifted.min(axis=1)
    max_ = shifted.max(axis=1)

    return pd.Series(np.log(min_ / max_), index=data.index)


def gaitpy_step_detection(data: pd.DataFrame) -> pd.DataFrame:
    """
    Detect Initial and Final contact of the foot in vertical acceleration.

    This method is based on GaitPy algorithm which consists in using wavelets
    transforms to decompose the vertical acceleration signal and the velocity
    to detect initial and final contacts. It is then formatted to the step
    data set format.

    Parameters
    ----------
    data
        The vertical acceleration.

    Returns
    -------
    pandas.DataFrame
        A data frame of the different gait event annotated under the step
        data set format with three columns:`event`, `foot` and `timestamp`.
    """
    ic_peaks_grav, fc_peaks_grav, *_ = _continuous_wavelet_transform(
        data["acceleration_x"],
        ic_prom=INITIAL_CONTACT_PROMINENCE,
        fc_prom=FINAL_CONTACT_PROMINENCE,
    )
    df_ic = pd.DataFrame(
        {
            "event": StepEvent.INITIAL_CONTACT,
            "foot": FootUsed.UNKNOWN,
            "timestamp": data.iloc[ic_peaks_grav].index,
        }
    ).set_index(keys="timestamp")

    df_fc = pd.DataFrame(
        {
            "event": StepEvent.FOOT_OFF,
            "foot": FootUsed.UNKNOWN,
            "timestamp": data.iloc[fc_peaks_grav].index,
        }
    ).set_index(keys="timestamp")
    return pd.concat([df_ic, df_fc]).sort_index()


class GaitPyDetectStepsWithoutBout(DetectStepsWithoutBoutsBase):
    """
    Detect Initial and Final contact of the foot without walking bouts.

    For more information about the step detection methodology, see:
    :class:`GaitPyDetectSteps`.
    """

    new_data_set_id = "gaitpy"
    transform_function = gaitpy_step_detection


class GaitPyDetectSteps(DetectStepsProcessingBase):
    """
    Detect Initial and Final contact of the foot in vertical acceleration.

    This method is based on GaitPy algorithm which consists in using wavelets
    transforms to decompose the vertical acceleration signal and the velocity
    to detect initial and final contacts. It is ran on each of the detected
    walking bout and then aggregated in a common data frame with an extra
    columns named `bout_id` indicating the id of the walking bout.
    It expects two data set id, the first one linking to the acceleration
    dataset containing the acceleration with gravity rotated resampled and
    detrended. The second one linking to the walking bout data set.
    """

    new_data_set_id = "gaitpy_with_walking_bouts"

    @staticmethod
    def step_detection_method(data: pd.DataFrame) -> pd.DataFrame:
        """Define and declare the step detection as a static method."""
        return gaitpy_step_detection(data)


class FormatAccelerationGaitPy(ProcessingStepGroup):
    """Format acceleration to vertical acceleration."""

    data_set_id = "acc"
    columns_with_g = [f"acceleration_{ax}" for ax in DEFAULT_COLUMNS]
    steps: List[ProcessingStep] = [
        AddGravityAndScale(f"{data_set_id}_ts"),
        RotateFrame(
            data_set_id=f"{data_set_id}_ts_g",
            gravity_data_set_id=f"{data_set_id}_ts",
            frame=(1, 0, 0),
            columns=columns_with_g,
        ),
        Resample(
            data_set_id=f"{data_set_id}_ts_g_rotated",
            aggregations=["mean", "ffill"],
            columns=columns_with_g,
        ),
        Detrend(
            data_set_id=f"{data_set_id}_ts_g_rotated_resampled", columns=columns_with_g
        ),
    ]


def _optimize_gaitpy_step_dataset(gaitpy_step: pd.DataFrame) -> pd.DataFrame:
    """
    Apply physiological constraints to filter impossible gait events.

    Several optimization steps are applied to the gaitpy step dataset. The
    first optimization implies constraint on times. Maximum allowable time are
    defined for: loading phases (each initial contact requires 1 forward final
    contact within 0.225 seconds) and stance phases (each initial contact
    requires at least 2 forward final contact's within 1.35 second) and entire
    gait cycles (each initial contact requires at least 2 initial contacts
    within 2.25 seconds after).

    Parameters
    ----------
    gaitpy_step
        The step dataset computed applying GaitPy wavelet transforms and
        peak detection.

    Returns
    -------
    pandas.DataFrame
        A data frame with optimized gait cycle.
    """
    ic_times = gaitpy_step.loc[gaitpy_step["event"] == StepEvent.INITIAL_CONTACT].index
    fc_times = gaitpy_step.loc[gaitpy_step["event"] == StepEvent.FOOT_OFF].index

    initial_contacts = []
    final_contacts = []
    final_contacts_opp_foot = []

    loading_forward_max_vect = ic_times + pd.Timedelta(
        LOADING_FORWARD_IC * 1000.0, unit="ms"
    )

    stance_forward_max_vect = ic_times + pd.Timedelta(
        STANCE_FORWARD_IC * 1000.0, unit="ms"
    )

    for i, current_ic in enumerate(ic_times):
        loading_forward_max = loading_forward_max_vect[i]
        stance_forward_max = stance_forward_max_vect[i]
        # Loading Response Constraint
        # Each IC requires 1 forward FC within 0.225 seconds
        # (opposite foot toe off)
        loading_forward_fcs = fc_times[
            (fc_times > current_ic) & (fc_times < loading_forward_max)
        ]
        # Stance Phase Constraint
        # Each IC requires at least 2 forward FC's within 1.35 second
        # (2nd FC is current IC's matching FC)
        stance_forward_fcs = fc_times[
            (fc_times > current_ic) & (fc_times < stance_forward_max)
        ]

        if len(loading_forward_fcs) == 1 and len(stance_forward_fcs) >= 2:
            initial_contacts.append(current_ic)
            final_contacts.append(stance_forward_fcs[1])
            final_contacts_opp_foot.append(stance_forward_fcs[0])

    res = pd.DataFrame(
        {
            "IC": initial_contacts,
            "FC": final_contacts,
            "FC_opp_foot": final_contacts_opp_foot,
        }
    ).reset_index(drop=True)

    if len(res) == 0:
        res["Gait_Cycle"] = False  # pylint: disable=E1137
        return res

    # Gait Cycles Constraint
    # Each ic requires at least 2 ics within 2.25 seconds after
    interval_1 = res.IC.diff(-1).dt.total_seconds().abs()
    interval_2 = res.IC.diff(-2).dt.total_seconds().abs()
    full_gait_cycle = (interval_1 <= GAIT_CYCLE_FORWARD_IC) & (
        interval_2 <= GAIT_CYCLE_FORWARD_IC
    )
    res["Gait_Cycle"] = full_gait_cycle  # pylint: disable=E1137
    return res


class OptimizeGaitpyStepDataset(TransformStep):
    """
    Apply physiological constraints to filter impossible gait events.

    It expects a unique data set id, linking to the gaitpy step data set.
    """

    definitions = [IC_DEF, FC_DEF, FC_OPP_FOOT_DEF, GAIT_CYCLE_DEF, DEF_BOUT_ID]

    def get_new_data_set_id(self) -> str:
        """Overwrite new_data_set_id."""
        return f"{self.data_set_ids}_optimized"

    @transformation
    def _transform_function(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize gaitpy dataset for each walking bout."""
        bout_ids = data["bout_id"].unique()
        list_df = []
        for bout_id in bout_ids:
            res = _optimize_gaitpy_step_dataset(data[data.bout_id == bout_id])
            res["bout_id"] = bout_id  # pylint: disable=E1137
            if len(res) > 0:
                list_df.append(res)
        if len(list_df) > 0:
            return pd.concat(list_df).reset_index(drop=True).sort_values("IC")
        return pd.DataFrame(
            columns=["IC", "FC", "FC_opp_foot", "Gait_Cycle", "bout_id"]
        )


class OptimizeGaitpyStepDatasetWithoutWalkingBout(TransformStep):
    """
    Apply physiological constraints to filter impossible gait events.

    It expects a unique data set id, linking to the gaitpy step data set
    without walking bouts.
    """

    definitions = [IC_DEF, FC_DEF, FC_OPP_FOOT_DEF, GAIT_CYCLE_DEF]

    def get_new_data_set_id(self) -> str:
        """Overwrite new_data_set_id."""
        return f"{self.data_set_ids}_optimized"

    transform_function = _optimize_gaitpy_step_dataset


def _height_change_com(
    optimized_gait_ic: pd.Series, vertical_acc: pd.Series
) -> pd.Series:
    """Compute height change of the center of mass when walking.

    Parameters
    ----------
    optimized_gait_ic
        A series of the data frame with optimized gait initial contacts.
    vertical_acc
        A series of the vertical acceleration.

    Returns
    -------
    pandas.Series
        A pandas series of the center of mass height changes.
    """
    # Get the sampling rate
    sampling_rate = get_sampling_rate_idx(vertical_acc)

    # Initialize the height changes of the center of mass
    height_change_center_of_mass = [None] * len(optimized_gait_ic)

    # Scale the minimum number of samples to compute the height change of the
    # Center of Mass with the sampling frequency
    min_samples_hccom = MIN_SAMPLES_HEIGHT_CHANGES_COM * sampling_rate / 50

    # Changes in Height of the Center of Mass
    for i in range(0, len(optimized_gait_ic) - 1):
        ic_index = optimized_gait_ic.iloc[i]
        post_ic_index = optimized_gait_ic.iloc[i + 1]
        step_raw = vertical_acc[ic_index:post_ic_index]
        if len(step_raw) <= min_samples_hccom:
            continue

        # Filter and low pass
        step_filtered = _detrend_and_low_pass(step_raw, sampling_rate)

        # Compute vertical displacement
        step_integrate_1 = integrate.cumtrapz(step_filtered) / sampling_rate
        step_integrate_2 = integrate.cumtrapz(step_integrate_1) / sampling_rate

        # Keep the max
        height_change_center_of_mass[i] = max(step_integrate_2) - min(step_integrate_2)
    return pd.Series(height_change_center_of_mass)


class HeightChangeCOM(TransformStep):
    """Compute the height changes of the center of mass.

    It expects two data set ids, first: the data set id mapping the output of
    FormatAccelerationGaitPy and second an id linking to the acceleration
    dataset containing the acceleration with gravity rotated resampled and
    detrended.
    """

    definitions = [
        RawDataValueDefinition(
            id_="height_com",
            name="height change center of mass",
            unit="m",
            data_type="float64",
        )
    ]

    @transformation
    def _transform_function(self, optimized_gait, vertical_acc):
        return _height_change_com(optimized_gait["IC"], vertical_acc["acceleration_x"])


def compute_stride_duration(data: pd.DataFrame) -> pd.DataFrame:
    """Compute stride duration and stride duration asymmetry.

    Stride duration is defined as the time elapsed between the first contact
    of two consecutive footsteps of the same foot.
    The asymmetry is defined as the difference of the feature between the right
    and left foot. It is given as an absolute difference as we don't know
    which foot is which.

    Parameters
    ----------
    data
        The gaitpy_optimized with step events output of the TransformStep
        OptimizeGaitpyStepDataset.

    Returns
    -------
    pandas.DataFrame
        A data frame with stride duration and stride duration asymmetry for
        each valid gait cycle.
    """
    # gait cycle duration
    data.loc[data.Gait_Cycle, "stride_dur"] = (
        data.IC.shift(-2) - data.IC
    ).dt.total_seconds()

    # asymmetry
    data.loc[data.Gait_Cycle, "stride_dur_asym"] = compute_asymmetry(data["stride_dur"])

    # asymmetry ln
    data.loc[data.Gait_Cycle, "stride_dur_asym_ln"] = compute_ln_asymmetry(
        data["stride_dur"]
    )
    return data


def compute_step_duration_and_cadence(data: pd.DataFrame) -> pd.DataFrame:
    """Compute step duration, step duration asymmetry and cadence.

    Step duration is the time elapsed between two consecutive footsteps. More
    specifically between two consecutive initial contact events.
    The asymmetry is defined as the difference of the feature
    between the right and left foot. It is given as an absolute difference as
    we don't know which foot is which.
    Cadence is defined as the instantaneous number of steps per minutes.

    Parameters
    ----------
    data
        The gaitpy_optimized with step events output of the TransformStep
        OptimizeGaitpyStepDataset.

    Returns
    -------
    pandas.DataFrame
        A data frame with step duration, step duration asymmetry and cadence
        for each valid gait cycle.
    """
    # step duration
    data.loc[data.Gait_Cycle, "step_dur"] = (
        data.IC.shift(-1) - data.IC
    ).dt.total_seconds()

    # asymmetry
    data.loc[data.Gait_Cycle, "step_dur_asym"] = compute_asymmetry(data["step_dur"])

    # asymmetry ln
    data.loc[data.Gait_Cycle, "step_dur_asym_ln"] = compute_ln_asymmetry(
        data["step_dur"]
    )

    # cadence
    data.loc[data.Gait_Cycle, "cadence"] = 60 / data.step_dur
    return data


def compute_stance(data: pd.DataFrame) -> pd.DataFrame:
    """Compute stance and stance asymmetry.

    The stance phase of gait begins when the foot first touches the ground and
    ends when the same foot leaves the ground. The stance phase makes up
    approximately 60% of the gait cycle. It can also be seen as the time
    elapsed between the final contact and initial contact.
    The asymmetry is defined as the difference of the feature
    between the right and left foot. It is given as an absolute difference as
    we don't know which foot is which.


    Parameters
    ----------
    data
        The gaitpy_optimized with step events output of the TransformStep
        OptimizeGaitpyStepDataset.

    Returns
    -------
    pandas.DataFrame
        Input data frame with stride duration and stride duration asymmetry
        added for each valid gait cycle.
    """
    # stance
    data.loc[data.Gait_Cycle, "stance"] = (data.FC - data.IC).dt.total_seconds()

    # asymmetry
    data.loc[data.Gait_Cycle, "stance_asym"] = compute_asymmetry(data["stance"])

    return data


def compute_initial_double_support(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute initial double support and initial double support asymmetry.

    Initial double support phase is the sub-phase between heel contact of the
    phase to contralateral foot-off. This phase makes up approximately 14-20%
    of the stance phase.

    Parameters
    ----------
    data
        The gaitpy_optimized with step events output of the TransformStep
        OptimizeGaitpyStepDataset.

    Returns
    -------
    pandas.DataFrame
        Input data frame with initial double support and initial double support
        asymmetry added for each valid gait cycle.
    """
    # Initial double support
    data.loc[data.Gait_Cycle, "init_double_supp"] = (
        data.FC_opp_foot - data.IC
    ).dt.total_seconds()

    # Asymmetry
    data.loc[data.Gait_Cycle, "init_double_supp_asym"] = compute_asymmetry(
        data["init_double_supp"]
    )
    return data


def compute_terminal_double_support(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute terminal double support and terminal double support asymmetry.

    Terminal double support phase is the sub-phase from contralateral foot-on
    to the toe-off. This phase makes up approximately 14-20% of the stance
    phase.

    Parameters
    ----------
    data
        The gaitpy_optimized with step events output of the TransformStep
        OptimizeGaitpyStepDataset.

    Returns
    -------
    pandas.DataFrame
        Input data frame with terminal double support and terminal double
        support asymmetry added for each valid gait cycle.
    """
    # terminal double support
    data.loc[data.Gait_Cycle, "term_double_supp"] = (
        data.FC - data.IC.shift(-1)
    ).dt.total_seconds()

    # asymmetry
    data.loc[data.Gait_Cycle, "term_double_supp_asym"] = compute_asymmetry(
        data["term_double_supp"]
    )
    return data


def compute_double_support(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute double support and double support asymmetry.

    Double support phase is defined as the sum of the initial and terminal
    double support phase. This makes up approximately 28-40% of the stance
    phase.

    Parameters
    ----------
    data
        The gaitpy_optimized with step events output of the TransformStep
        OptimizeGaitpyStepDataset.

    Returns
    -------
    pandas.DataFrame
        Input data frame with double support and double support asymmetry added
        for each valid gait cycle.
    """
    # double support
    data.loc[data.Gait_Cycle, "double_supp"] = (
        data.init_double_supp + data.term_double_supp
    )

    # asymmetry
    data.loc[data.Gait_Cycle, "double_supp_asym"] = compute_asymmetry(
        data["double_supp"]
    )
    return data


def compute_single_limb_support(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute single limb support and single limb support asymmetry.

    Single limb support represents a phase in the gait cycle when the body
    weight is entirely supported by one limb, while the contra-lateral limb
    swings forward.

    Parameters
    ----------
    data
        The gaitpy_optimized with step events output of the TransformStep
        OptimizeGaitpyStepDataset.

    Returns
    -------
    pandas.DataFrame
        Input data frame with single limb support and single limb support
        asymmetry added for each valid gait cycle.
    """
    # single limb support
    data.loc[data.Gait_Cycle, "single_limb_supp"] = (
        data.IC.shift(-1) - data.FC.shift(1)
    ).dt.total_seconds()

    # asymmetry
    data.loc[data.Gait_Cycle, "single_limb_supp_asym"] = compute_asymmetry(
        data["single_limb_supp"]
    )
    return data


def compute_swing(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute swing and swing asymmetry.

    The swing phase of gait begins when the foot first leaves the ground and
    ends when the same foot touches the ground again. The swing phase makes up
    40% of the gait cycle.

    Parameters
    ----------
    data
        The gaitpy_optimized with step events output of the TransformStep
        OptimizeGaitpyStepDataset.

    Returns
    -------
    pandas.DataFrame
        Input data frame with swing and swing asymmetry added for each valid
        gait cycle.
    """
    data.loc[data.Gait_Cycle, "swing"] = (
        data.IC.shift(-2) - data.FC
    ).dt.total_seconds()

    # asymmetry
    data.loc[data.Gait_Cycle, "swing_asym"] = compute_asymmetry(data["swing"])
    return data


def compute_step_length(
    data: pd.DataFrame, height_changes_com: pd.Series, sensor_height: float
) -> pd.DataFrame:
    """Compute step length and step length asymmetry.

    Step length is the distance between the point of initial contact of one
    foot and the point of initial contact of the opposite foot.

    Parameters
    ----------
    data
        The gaitpy_optimized with step events output of the TransformStep
        OptimizeGaitpyStepDataset.
    height_changes_com
        A panda series with the changes in height of the center of mass.
    sensor_height
        The height of the smartphone sensor in meters when located at the
        lumbar position.

    Returns
    -------
    pandas.DataFrame
        Input data frame with step length and step length asymmetry added for
        each valid gait cycle.
    """
    try:
        res = 2 * sensor_height * height_changes_com - height_changes_com**2
        # if negative set to nan to prevent errors in sqrt computation
        # this typically occurs due to drift due to long intervals introduced
        # by steps belonging to different walking bouts
        res[res < 0] = np.nan
        step_length = 2 * (np.sqrt(res))
    except TypeError:
        step_length = np.nan

    data.loc[data.Gait_Cycle, "step_len"] = step_length

    # asymmetry
    data.loc[data.Gait_Cycle, "step_len_asym"] = compute_asymmetry(data["step_len"])

    # asymmetry ln
    data.loc[data.Gait_Cycle, "step_len_asym_ln"] = compute_ln_asymmetry(
        data["step_len"]
    )
    return data


def compute_stride_length(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute stride length and stride length asymmetry.

    Stride length is the distance covered when you take two steps,
    one with each foot.

    Parameters
    ----------
    data
        The gaitpy_optimized with step events output of the TransformStep
        OptimizeGaitpyStepDataset.

    Returns
    -------
    pandas.DataFrame
        Input data frame with stride length and stride length asymmetry added
        for each valid gait cycle.
    """
    # stride length
    data.loc[data.Gait_Cycle, "stride_len"] = data.step_len.shift(-1) + data.step_len

    # asymmetry
    data.loc[data.Gait_Cycle, "stride_len_asym"] = compute_asymmetry(data["stride_len"])

    # asymmetry ln
    data.loc[data.Gait_Cycle, "stride_len_asym_ln"] = compute_ln_asymmetry(
        data["stride_len"]
    )
    return data


def compute_gait_speed(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute gait speed.

    Gait speed is defined as the instantaneous walking speed in m/s. It is
    computed as the length of a stride divided by the duration.

    Parameters
    ----------
    data
        The gaitpy_optimized with step events output of the TransformStep
        OptimizeGaitpyStepDataset.

    Returns
    -------
    pandas.DataFrame
        Input data frame with step velocity added for each valid gait cycle.
    """
    data.loc[data.Gait_Cycle, "gait_speed"] = data.stride_len / data.stride_dur
    return data


def _cwt_feature_extraction(
    data: pd.DataFrame, height_changes_com: pd.DataFrame, sensor_height: float
) -> pd.DataFrame:
    """
    Compute several temporal and spatial features for each gait cycle.

    Parameters
    ----------
    data
        The data frame of the gaitpy step dataset after optimization, with
        step annotated with IC, FC, FC_opp_foot and GaitCycle.
    height_changes_com
        A data frame with the changes in height of the center of mass.
    sensor_height
        The height of the smartphone sensor in meters when located at the
        lumbar position.

    Returns
    -------
    pandas.DataFrame
        The optimized gait steps data frame with temporal and spatial features
        computed for each gait cycle.
    """

    def _compute_cwt_features(
        data: pd.DataFrame, height_changes_com: pd.Series
    ) -> pd.DataFrame:
        """Compute all the cwt features."""
        # work on a copy of the dataframe
        res = data.copy()

        # steps
        res["steps"] = len(res)

        # gait cycle duration features
        res = compute_stride_duration(res)

        # step time and cadence
        res = compute_step_duration_and_cadence(res)

        # initial double support
        res = compute_initial_double_support(res)

        # terminal double support
        res = compute_terminal_double_support(res)

        # double support
        res = compute_double_support(res)

        # single limb support
        res = compute_single_limb_support(res)

        # stance
        res = compute_stance(res)

        # swing
        res = compute_swing(res)

        # step length
        res = compute_step_length(res, height_changes_com, sensor_height)

        # stride length
        res = compute_stride_length(res)

        # step velocity
        res = compute_gait_speed(res)

        return res

    if "bout_id" not in data.columns:
        features = _compute_cwt_features(data, height_changes_com["height_com"])
        return features.set_index("FC", drop=False)

    list_df_features = []

    for bout_id in data.bout_id.unique():
        # Filter on bout
        bout_data = data[data.bout_id == bout_id]

        bout_data = _compute_cwt_features(
            bout_data, height_changes_com.loc[data.bout_id == bout_id, "height_com"]
        )

        # append to the list of df_features
        list_df_features.append(bout_data)
    if len(list_df_features) > 0:
        return pd.concat(list_df_features).set_index("FC", drop=False)
    column_names = ["IC", "FC", "FC_opp_foot", "Gait_Cycle", "bout_id"]
    column_names += list(CWT_FEATURES_DESC_MAPPING)
    return pd.DataFrame(columns=column_names)


def get_subject_height(context: Context) -> float:
    """Get subject height from context if available else return default."""
    if "subject_height" in context:
        return context["subject_height"].value
    warnings.warn(
        "Subject height is not available in context "
        "using DEFAULT_SENSOR_HEIGHT: "
        f"{DEFAULT_SUBJECT_HEIGHT}."
    )
    return DEFAULT_SUBJECT_HEIGHT


def get_sensor_height(context: Context) -> float:
    """Get sensor height from context if available else return default."""
    if "subject_height" in context:
        return SENSOR_HEIGHT_RATIO * context["subject_height"].value
    warnings.warn(
        "Subject height is not available in context to compute "
        "sensor_height: using DEFAULT_SENSOR_HEIGHT: "
        f"{DEFAULT_SENSOR_HEIGHT}."
    )
    return DEFAULT_SENSOR_HEIGHT


class CWTFeatureTransformation(TransformStep):
    """Compute temporal and spatial features for each gait cycle.

    It expects two data set ids, first: The data set id of the gaitpy step
    dataset after optimization, with step annotated with IC, FC, FC_opp_foot
    and GaitCycle and second, the data set id of the height changes of the
    center of mass.
    """

    definitions = [
        IC_DEF,
        FC_DEF,
        FC_OPP_FOOT_DEF,
        GAIT_CYCLE_DEF,
        DEF_BOUT_ID,
    ] + CWT_FEATURES_DEF

    @transformation
    def extract_features(
        self, data: pd.DataFrame, height_changes_com: pd.DataFrame, level: Level
    ):
        """Extract cwt features."""
        sensor_height = get_sensor_height(level.context)
        return _cwt_feature_extraction(
            data, height_changes_com, sensor_height=sensor_height
        )


class CWTFeatureWithoutBoutTransformation(TransformStep):
    """Compute temporal and spatial features for each gait cycle.

    It expects two data set ids, first: The data set id of the gaitpy step
    dataset after optimization, with step annotated with IC, FC, FC_opp_foot
    and GaitCycle and second, the data set id of the height changes of the
    center of mass.
    """

    definitions = [IC_DEF, FC_DEF, FC_OPP_FOOT_DEF, GAIT_CYCLE_DEF] + CWT_FEATURES_DEF

    @transformation
    def extract_features(
        self, data: pd.DataFrame, height_changes_com: pd.DataFrame, level: Level
    ):
        """Extract cwt features."""
        sensor_height = get_sensor_height(level.context)
        return _cwt_feature_extraction(
            data, height_changes_com, sensor_height=sensor_height
        )


class AggregateCWTFeature(GaitBoutAggregateStep):
    """An Aggregation of cwt features on walking bouts."""

    def __init__(self, data_set_id: str, column_id: str, **kwargs):
        # type: ignore
        description = (
            f"The {{aggregation}} of {column_id} "
            "with the bout strategy {bout_strategy_repr}. "
            f"{CWT_FEATURES_DESC_MAPPING[column_id]} ."
        )
        definition = FeatureValueDefinitionPrototype(
            feature_name=AV(column_id.replace("_", " "), column_id),
            data_type="float64",
            description=description,
            precision=CWT_FEATURES_PRECISION_MAPPING[column_id],
        )

        super().__init__(
            ["walking_placement_no_turn_bouts", data_set_id],
            column_id,
            DEFAULT_AGGREGATIONS_Q95_CV,
            definition,
            **kwargs,
        )


class AggregateCWTFeatureWithoutBout(AggregateRawDataSetColumn):
    """An Aggregation of cwt features not on walking bouts."""

    def __init__(self, data_set_id: str, column_id: str, **kwargs):
        description = (
            f"The {{aggregation}} of {column_id}."
            f"{CWT_FEATURES_DESC_MAPPING[column_id]} ."
        )
        definition = FeatureValueDefinitionPrototype(
            feature_name=AV(column_id.replace("_", " "), column_id),
            data_type="float64",
            description=description,
            precision=CWT_FEATURES_PRECISION_MAPPING[column_id],
        )

        super().__init__(
            data_set_id, column_id, DEFAULT_AGGREGATIONS_Q95_CV, definition, **kwargs
        )


class GaitPyCountSteps(GaitBoutExtractStep):
    """Extract the total number of steps counted with gaitpy."""

    def __init__(self, data_set_id, **kwargs):
        definition = FeatureValueDefinitionPrototype(
            feature_name=AV("step count", "sc"),
            data_type="int16",
            validator=GREATER_THAN_ZERO,
            description="The number of steps detected with gaitpy algorithm "
            "with the bout strategy {bout_strategy_repr}.",
        )
        super().__init__(
            ["walking_placement_no_turn_bouts", data_set_id], len, definition, **kwargs
        )


class GaitPyPowerBoutDivSteps(ExtractPowerBoutDivSteps):
    """Extract step power from GaitPy with walking bouts."""

    def __init__(self, **kwargs):
        data_set_ids = [
            "walking_placement_no_turn_bouts",
            "vertical_acceleration",
            "gaitpy_with_walking_bouts",
        ]
        super().__init__(data_set_ids=data_set_ids, **kwargs)


class GaitPyPowerBoutDivStepsWithoutBout(ExtractStep):
    """Extract step power with GaitPy dataset without walking bouts."""

    def __init__(self, **kwargs):
        data_set_ids = ["vertical_acceleration", "gaitpy"]
        definition = FeatureValueDefinitionPrototype(
            feature_name=AV("step power", "sp"),
            data_type="int16",
            validator=GREATER_THAN_ZERO,
            description="The integral of the centered acceleration magnitude "
            "between the first and last step divided by the "
            "number of steps computed with the GaitPy algorithm.",
        )
        super().__init__(
            data_set_ids=data_set_ids,
            transform_function=power_bout_div_steps,
            definition=definition,
            **kwargs,
        )


class GaitPyStepCountWithoutBout(ExtractStep):
    """Extract step count with GaitPy dataset without walking bouts."""

    def __init__(self, **kwargs):
        data_set_ids = "cwt_features"
        description = (
            "The number of steps detected with gaitpy"
            " algorithm not using walking bouts."
        )
        definition = FeatureValueDefinitionPrototype(
            feature_name=AV("step count", "sc"),
            data_type="int16",
            validator=GREATER_THAN_ZERO,
            description=description,
        )
        super().__init__(
            data_set_ids=data_set_ids,
            transform_function=len,
            definition=definition,
            **kwargs,
        )


class AggregateAllCWTFeatures(ProcessingStepGroup):
    """A group of gait processing steps for cwt features on walking bouts."""

    def __init__(self, data_set_id: str, **kwargs):
        feature_ids = [
            "stride_dur",
            "stride_dur_asym",
            "step_dur",
            "step_dur_asym",
            "stride_dur_asym_ln",
            "step_dur_asym_ln",
            "cadence",
            "init_double_supp",
            "init_double_supp_asym",
            "term_double_supp",
            "term_double_supp_asym",
            "double_supp",
            "double_supp_asym",
            "single_limb_supp",
            "single_limb_supp_asym",
            "stance",
            "stance_asym",
            "swing",
            "swing_asym",
            "step_len",
            "step_len_asym",
            "stride_len",
            "stride_len_asym",
            "gait_speed",
            "stride_len_asym_ln",
            "step_len_asym_ln",
        ]
        steps = [
            AggregateCWTFeature(data_set_id, feature_id, **kwargs)
            for feature_id in feature_ids
        ]
        super().__init__(steps=steps, **kwargs)


class AggregateAllCWTFeaturesWithoutBout(ProcessingStepGroup):
    """Group of gait processing steps for cwt features not on walking bout."""

    def __init__(self, data_set_id: str, **kwargs):
        feature_ids = [
            "stride_dur",
            "stride_dur_asym",
            "step_dur",
            "step_dur_asym",
            "stride_dur_asym_ln",
            "step_dur_asym_ln",
            "cadence",
            "init_double_supp",
            "init_double_supp_asym",
            "term_double_supp",
            "term_double_supp_asym",
            "double_supp",
            "double_supp_asym",
            "single_limb_supp",
            "single_limb_supp_asym",
            "stance",
            "stance_asym",
            "swing",
            "swing_asym",
            "step_len",
            "step_len_asym",
            "stride_len",
            "stride_len_asym",
            "gait_speed",
            "stride_len_asym_ln",
            "step_len_asym_ln",
        ]
        steps = [
            AggregateCWTFeatureWithoutBout(data_set_id, feature_id, **kwargs)
            for feature_id in feature_ids
        ]
        super().__init__(steps=steps, **kwargs)


class GaitPyFeatures(ProcessingStepGroup):
    """Extract Del Din features based on GaitPy Steps and a bout strategy."""

    def __init__(self, bout_strategy: BoutStrategyModality, **kwargs):
        data_set_id = "cwt_features_with_walking_bouts"
        steps: List[ProcessingStep] = [
            AggregateAllCWTFeatures(
                data_set_id=data_set_id, bout_strategy=bout_strategy.bout_cls
            ),
            GaitPyCountSteps(
                data_set_id=data_set_id, bout_strategy=bout_strategy.bout_cls
            ),
            GaitPyStepPower(bout_strategy=bout_strategy.bout_cls),
            GaitPyStepIntensity(bout_strategy=bout_strategy.bout_cls),
            GaitPyStepRegularity(bout_strategy=bout_strategy.bout_cls),
            GaitPyStrideRegularity(bout_strategy=bout_strategy.bout_cls),
        ]
        super().__init__(steps, **kwargs)


class GaitPyFeaturesWithoutBout(ProcessingStepGroup):
    """Extract Del Din features based on GaitPy Steps and a bout strategy."""

    def __init__(self, **kwargs):
        data_set_id = "cwt_features"
        steps: List[ProcessingStep] = [
            AggregateAllCWTFeaturesWithoutBout(data_set_id=data_set_id),
            GaitPyStepCountWithoutBout(),
            GaitPyStepPowerWithoutBout(),
            GaitPyStepIntensityWithoutBout(),
            GaitPyStepRegularityWithoutBout(),
            GaitPyStrideRegularityWithoutBout(),
        ]
        super().__init__(steps, **kwargs)


class GaitPyStepPowerWithoutBout(AggregateRawDataSetColumn):
    """Extract step power without walking bout."""

    data_set_ids = "gaitpy_optimized_step_vigor"
    column_id = "step_power"
    aggregations = DEFAULT_AGGREGATIONS_Q95
    definition = FeatureValueDefinitionPrototype(
        feature_name=AV("step power", "sp"),
        data_type="float64",
        unit="m^2/s^3",
        validator=GREATER_THAN_ZERO,
        description="The {aggregation} step power across detected steps.",
    )


class GaitPyStepPower(ExtractStepPowerAll):
    """Extract step power for gaitpy."""

    def __init__(self, **kwargs):
        super().__init__(
            data_set_ids=[
                "walking_placement_no_turn_bouts",
                "gaitpy_with_walking_bouts_optimized_step_vigor",
            ],
            **kwargs,
        )


class GaitPyStepIntensityWithoutBout(AggregateRawDataSetColumn):
    """Extract step intensity without walking bout."""

    data_set_ids = "gaitpy_optimized_step_vigor"
    column_id = "step_intensity"
    aggregations = DEFAULT_AGGREGATIONS
    definition = FeatureValueDefinitionPrototype(
        feature_name=AV("step intensity", "si"),
        data_type="float64",
        unit="m/s^2",
        validator=GREATER_THAN_ZERO,
        description="The {aggregation} step intensity across detected steps.",
    )


class GaitPyStepIntensity(ExtractStepIntensityAll):
    """Extract step intensity for gaitpy."""

    def __init__(self, **kwargs):
        super().__init__(
            data_set_ids=[
                "walking_placement_no_turn_bouts",
                "gaitpy_with_walking_bouts_optimized_step_vigor",
            ],
            **kwargs,
        )


class GaitPyStepRegularityWithoutBout(AggregateRawDataSetColumn):
    """Extract step regularity without walking bout."""

    data_set_ids = "gaitpy_optimized_gait_regularity"
    column_id = "step_regularity"
    aggregations = DEFAULT_AGGREGATIONS
    definition = FeatureValueDefinitionPrototype(
        feature_name=AV("step regularity", "step_regularity"),
        data_type="float64",
        unit="s",
        validator=BETWEEN_MINUS_ONE_AND_ONE,
        description="The {aggregation} step regularity across all steps.",
    )


class GaitPyStrideRegularityWithoutBout(AggregateRawDataSetColumn):
    """Extract stride regularity without walking bout."""

    data_set_ids = "gaitpy_optimized_gait_regularity"
    column_id = "stride_regularity"
    aggregations = DEFAULT_AGGREGATIONS
    definition = FeatureValueDefinitionPrototype(
        feature_name=AV("stride regularity", "stride_regularity"),
        data_type="float64",
        unit="s",
        validator=BETWEEN_MINUS_ONE_AND_ONE,
        description="The {aggregation} stride regularity across all steps.",
    )


class GaitPyStepRegularity(ExtractStepRegularity):
    """Extract step regularity for gaitpy."""

    def __init__(self, **kwargs):
        super().__init__(
            data_set_ids=[
                "walking_placement_no_turn_bouts",
                "gaitpy_with_walking_bouts_optimized_gait_regularity",
            ],
            **kwargs,
        )


class GaitPyStrideRegularity(ExtractStrideRegularity):
    """Extract stride regularity for gaitpy."""

    def __init__(self, **kwargs):
        super().__init__(
            data_set_ids=[
                "walking_placement_no_turn_bouts",
                "gaitpy_with_walking_bouts_optimized_gait_regularity",
            ],
            **kwargs,
        )
