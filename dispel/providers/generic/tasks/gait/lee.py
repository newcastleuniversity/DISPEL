"""Step detection module specific to Lee et al. algorithm.

This module contains functionality to perform step detection with a revisited
version of the Lee et al. algorithm.
"""
import enum
from typing import List, Optional, Tuple

import pandas as pd

from dispel.data.measures import MeasureValueDefinitionPrototype
from dispel.data.validators import GREATER_THAN_ZERO
from dispel.data.values import AbbreviatedValue as AV
from dispel.processing.core import ProcessingStep
from dispel.processing.extract import (
    DEFAULT_AGGREGATIONS,
    AggregateRawDataSetColumn,
    ExtractStep,
)
from dispel.processing.level import ProcessingStepGroup
from dispel.providers.generic.tasks.gait.bout_strategy import BoutStrategyModality
from dispel.providers.generic.tasks.gait.core import (
    DetectStepsProcessingBase,
    DetectStepsWithoutBoutsBase,
    ExtractPowerBoutDivSteps,
    ExtractStepCount,
    ExtractStepDurationAll,
    FootUsed,
    StepEvent,
    power_bout_div_steps,
    step_count,
)
from dispel.providers.generic.tasks.gait.hip import (
    ExtractHipRotation,
    ExtractHipRotationWithoutBouts,
    HipRotationGroup,
)
from dispel.signal.core import index_time_diff


class StepState(enum.IntEnum):
    """Step detection states for Lee et al. algorithm."""

    PEAK = 2
    VALLEY = 1
    INTMD = 0
    INITIAL = -1


LEE_MOD = AV("Lee algorithm", "lee")
r"""A modality indicating something has been computed with Lee algorithm."""

# MODEL CONSTANTS
K_SIGMA = 25
r"""Parameter K_SIGMA should be selected such that the step deviation can
reflect the long-term variation in the statistics of the vertical acceleration.
The value of 25 is assigned to K to cover one step cycle in normal walking
speed with the sampling rate of 50 Hz."""
M_SIGMA = 10
r"""Parameter M should be selected with :math:`\beta` such that the statistics
of peak or valley intervals can reflect the time-varying speed of walking or
running and the noisy peaks or valleys can be delineated from real peaks or
valleys."""
ALPHA = 4
r"""Parameter :math:`\alpha` is a magnitude constant that should be assigned
so as not to disturb the peak or valley detection due to large step deviation
during step mode change, especially from running to walking."""
BETA = 1 / 3
r"""Parameter :math:`\beta` is a time scale constant that should be assigned
with M. It is used to rescale (as a denominator) the standard deviation of
the last M peak or valley time-intervals when computing the time threshold
used to accept or reject any peak or valley candidate."""

# Parameters Initialization (not defined in the paper).
DEFAULT_MU = 0.25
r"""Default parameter :math:`\mu` is used to initiate the average time between
two consecutive peaks (valleys) for the last M peaks (valleys)."""
DEFAULT_SIGMA = 0.0
r"""Default parameter :math:`\sigma` is used to initiate the standard deviation
of the time between two consecutive peaks (valleys) for the last M peaks
(valleys)."""
DEFAULT_SIGMA_A = 0.0
r"""Default parameter :math:`\sigma` is used to initiate the standard deviation
of the vertical acceleration for recent K_SIGMA acceleration samples."""
DEFAULT_PEAK_THRESHOLD = 0.025  # the adaptive time threshold for peaks
r"""Default parameter peak threshold is used to initialize the adaptive time
threshold for peaks. This threshold will be used to accept or reject a peak
candidate based on the time-interval separating it from the previous peak in
addition to other conditions."""
DEFAULT_VALLEY_THRESHOLD = 0.025  # the adaptive time threshold for valley
r"""Default parameter valley threshold is used to initialize the valley
threshold. This threshold will be used to accept or reject any valley
candidates based on the time interval separating it from the previous valley
candidate and other conditions."""
DEFAULT_VALLEY_ACC = 0.0
r"""Default parameter used to initialize the vertical acceleration of a
valley."""
DEFAULT_PEAK_ACC = 1.0
r"""Default parameter used to initialize the vertical acceleration of a
peak."""


def _detect_candidate(
    data: pd.Series, index_sample: int, mu_a: float, sigma_a: float, alpha: float
) -> StepState:
    """Detect peak and valley candidates in the signal.

    This function labels each sample as valley, peak, or intermediate.
    The sample is considered a peak if: case 1 it is the first sample or case 2
    , if the sample vertical acceleration is greater than the previous and next
    vertical acceleration and more significant than the average detection step
    (plus a modulation).

    Parameters
    ----------
    data
        A series of the vertical acceleration.
    index_sample
        An integer indicating which sample is under examination.
    mu_a
        The average of the vertical acceleration of a step. Defined as
        the mean of the magnitude of the recent peak and recent valley.
    sigma_a
        The standard deviation of the vertical acceleration.
    alpha
        A constant to modulate the threshold on vertical acceleration used to
        label a sample.

    Returns
    -------
    StepState
        A label indicating if the sample is a good candidate for a peak,
        a valley or an intermediate sample.
    """
    if index_sample == 1:
        return StepState.PEAK
    acc_minus, acc, acc_plus = data.iloc[index_sample - 1 : index_sample + 2]
    if acc > max(max(acc_minus, acc_plus), mu_a + sigma_a / alpha):
        return StepState.PEAK
    if acc < min(min(acc_minus, acc_plus), mu_a - sigma_a / alpha):
        return StepState.VALLEY
    return StepState.INTMD


def _update_peak_valley(
    data: pd.DataFrame,
    new_state: StepState,
    index_sample: int,
    beta: float,
    m_sigma: int,
) -> Tuple[float, float]:
    """Update a peak or a valley.

    Parameters
    ----------
    data
        A data frame of the vertical acceleration and states.
    new_state
        Either peak or valley, it indicates if a peak or a valley is to be
        updated.
    index_sample
        An integer indicating which sample is under examination.
    beta
        A time scale constant.
    m_sigma
        A parameter used to delineate noisy peaks or valley from real peaks or
        valleys.

    Returns
    -------
    Tuple[float, float]
        The vertical acceleration of current sample. And the minimum time
        distance to the recent peak (or valley).
    """
    peaks_or_valley = data.loc[data["state"] == new_state]
    t_between = index_time_diff(peaks_or_valley)[1:]

    if len(t_between) > 1:
        # enough data
        sigma = t_between.tail(m_sigma).std()
        mu_x = t_between.tail(m_sigma).mean()
    elif len(t_between) == 1:
        # just enough for the mean
        mu_x = t_between.tail(m_sigma).mean()
        sigma = DEFAULT_SIGMA
    else:
        # initialization
        sigma = DEFAULT_SIGMA
        mu_x = DEFAULT_MU
    threshold = mu_x - sigma / beta
    return data.iloc[index_sample]["vertical_acc"], threshold


def _update_sigma(data: pd.Series) -> float:
    """Update sigma.

    ``sigma_a`` is defined as the standard deviation of the vertical
    acceleration for recent ``k_sigma`` acceleration samples.

    Parameters
    ----------
    data
        A series of the last k_sigma vertical acceleration.

    Returns
    -------
    float
        The standard deviation of the vertical acceleration over the last
        k_sigma samples.
    """
    if len(data) > 1:
        return data.std()
    return DEFAULT_SIGMA_A


def _check_state(
    data: pd.Series,
    last_state: StepState,
    expected_state: StepState,
    index_s: int,
    index_c: int,
    t_thresh: float,
    acc_threshold: Optional[float] = None,
    greater: bool = True,
    further: bool = True,
) -> bool:
    """Check conditions on the last_state, time, and vertical acceleration.

    Parameters
    ----------
    data
        A series of the vertical acceleration.
    last_state
        Either peak or valley, it indicates if a peak or a valley was the last
        specific state.
    expected_state
        Expected value for the last_state.
    index_s
        Index of the last state.
    index_c
        Index of the current state.
    t_thresh
        A threshold on time to remove close peaks/valleys
    acc_threshold
        A threshold on the vertical acceleration to potentially replace
        peaks/valleys.
    greater
        A boolean deciding if acc_threshold should be compared with a greater
        or less than comparator.
    further
        A boolean deciding if the time threshold should be compared with a
        a greater or less than comparator.

    Returns
    -------
    bool
        A boolean indicating if the conditions are respected
    """
    # Does the last_state matches the expected state
    c_1 = last_state == expected_state
    # Is the current sample far enough from the previous specific state
    c_2 = (data.index[index_c] - data.index[index_s]).total_seconds()
    if further:
        c_2 = c_2 > t_thresh
    else:
        c_2 = c_2 <= t_thresh
    # If the acceleration threshold is provided compare it to the current
    # sample
    if acc_threshold:
        c_3 = data[data.index[index_c]]
        if greater:
            c_3 = c_3 > acc_threshold
        else:
            c_3 = c_3 <= acc_threshold
        return c_1 and c_2 and c_3
    return c_1 and c_2


def _detect_steps(data: pd.Series) -> pd.DataFrame:
    """Step Detection Algorithm from Lee et al. 2015.

    Full reference: Lee, H.-H.; Choi, S.; Lee, M.-J.
    Step Detection Robust against the Dynamics of Smartphones.
    Sensors 2015, 15, 27230-27250.

    Parameters
    ----------
    data
        A series of the vertical acceleration.

    Returns
    -------
    pandas.DataFrame
        A data frame containing the candidate and final state detected, as
        well as the step_index, a variable keeping track of when a step is
        detected.
    """
    res = pd.DataFrame(
        {
            "vertical_acc": data.copy(),
            "state": StepState.INTMD,
            "candidate_state": None,
            "step_index": 0,
        }
    )

    # the adaptive time threshold for peaks
    peak_threshold = DEFAULT_PEAK_THRESHOLD
    # the adaptive time threshold for valley
    valley_threshold = DEFAULT_VALLEY_THRESHOLD
    acc_valley = DEFAULT_VALLEY_ACC
    acc_peak = DEFAULT_PEAK_ACC
    # the step average
    mu_a = res["vertical_acc"].mean()
    # the step deviation of the vertical acceleration
    sigma_a = DEFAULT_SIGMA_A

    # ``last_state`` tracks the last particular state, either peak or valley
    # and will replace Sn-1 in the algorithm description page 27240.
    last_state = StepState.INITIAL
    step_index = 0
    it_peak = 0
    it_valley = 0

    def _set_state(i: int, state: StepState, state_column: str = "state"):
        """Set the state at a given index."""
        res.loc[res.index[i], state_column] = state

    for it_n in range(1, len(res) - 1):
        # Determine if the sample is a potential peak or valley candidate
        candidate_state = _detect_candidate(
            res["vertical_acc"], it_n, mu_a, sigma_a, ALPHA
        )
        # Save the candidate state
        _set_state(it_n, candidate_state, "candidate_state")
        # Initialize the ``state`` to 'intermediate'
        _set_state(it_n, StepState.INTMD)

        if candidate_state == StepState.PEAK:
            if _check_state(
                data=res["vertical_acc"],
                last_state=last_state,
                expected_state=StepState.VALLEY,
                index_s=it_peak,
                index_c=it_n,
                t_thresh=peak_threshold,
            ):
                # (2)
                _set_state(it_n, StepState.PEAK)
                last_state = StepState.PEAK
                it_peak = it_n
                acc_peak, peak_threshold = _update_peak_valley(
                    res, StepState.PEAK, it_n, BETA, M_SIGMA
                )
                mu_a = (acc_peak + acc_valley) / 2
            elif _check_state(
                data=res["vertical_acc"],
                last_state=last_state,
                expected_state=StepState.PEAK,
                index_s=it_peak,
                index_c=it_n,
                t_thresh=peak_threshold,
                acc_threshold=acc_peak,
                further=False,
            ):
                # (3)
                _set_state(it_peak, StepState.INTMD)
                _set_state(it_n, StepState.PEAK)
                last_state = StepState.PEAK
                it_peak = it_n
                acc_peak, peak_threshold = _update_peak_valley(
                    res, StepState.PEAK, it_n, BETA, M_SIGMA
                )
            # This should only be triggered once at the initialization of the
            # algorithm, when it is 1.
            elif last_state == StepState.INITIAL:
                # (1)
                _set_state(it_n, StepState.PEAK)
                last_state = StepState.PEAK
                it_peak = it_n
                acc_peak, peak_threshold = _update_peak_valley(
                    res, StepState.PEAK, it_n, BETA, M_SIGMA
                )
        elif candidate_state == StepState.VALLEY:
            if _check_state(
                data=res["vertical_acc"],
                last_state=last_state,
                expected_state=StepState.PEAK,
                index_s=it_valley,
                index_c=it_n,
                t_thresh=valley_threshold,
            ):
                # (4)
                _set_state(it_n, StepState.VALLEY)
                last_state = StepState.VALLEY
                it_valley = it_n
                acc_valley, valley_threshold = _update_peak_valley(
                    res, StepState.VALLEY, it_n, BETA, M_SIGMA
                )
                step_index += 1
                mu_a = (acc_peak + acc_valley) / 2
            elif _check_state(
                data=res["vertical_acc"],
                last_state=last_state,
                expected_state=StepState.VALLEY,
                index_s=it_valley,
                index_c=it_n,
                t_thresh=valley_threshold,
                acc_threshold=acc_valley,
                greater=False,
                further=False,
            ):
                # (5)
                _set_state(it_valley, StepState.INTMD)
                _set_state(it_n, StepState.VALLEY)
                last_state = StepState.VALLEY
                it_valley = it_n
                acc_valley, valley_threshold = _update_peak_valley(
                    res, StepState.VALLEY, it_n, BETA, M_SIGMA
                )
        # Update sigma
        sigma_a = _update_sigma(res.iloc[max(0, it_n - K_SIGMA) : it_n]["vertical_acc"])
        res.loc[res.index[it_n], "step_index"] = step_index
    return res


def detect_steps(data: pd.DataFrame) -> pd.DataFrame:
    """Run step Detection Algorithm from Lee et al. and format the results.

    We use a revisited Lee et al. algorithm since we don't perform step
    detection on the acceleration norm but on the vertical acceleration.
    The results are formatted to return a generic data frame with the following
    columns: ``timestamp``, ``event``, ``foot``. Where ``event`` annotate what
    is happening as in Bourke et al. doi:10.3390/s20205906.

    Parameters
    ----------
    data
        A data frame containing a column 'vertical_acc' referring to the
        vertical acceleration.

    Returns
    -------
    pandas.DataFrame
        A pandas data frame with columns ``event``, ``foot`` and ``timestamp``.
    """
    detected_steps = _detect_steps(data["vertical_acc"])
    timestamp = detected_steps.index[detected_steps["state"] == StepState.PEAK]
    return pd.DataFrame(
        {
            "event": StepEvent.INITIAL_CONTACT,
            "foot": FootUsed.UNKNOWN,
            "timestamp": timestamp,
        }
    ).set_index(keys="timestamp")


class LeeDetectSteps(DetectStepsProcessingBase):
    """Detect steps using Lee et al. algorithm on vertical acceleration."""

    new_data_set_id = "lee_with_walking_bouts"

    @staticmethod
    def step_detection_method(data: pd.DataFrame) -> pd.DataFrame:
        """Define and declare the step detection as a static method."""
        return detect_steps(data)


class LeeDetectStepsWithoutBout(DetectStepsWithoutBoutsBase):
    """Detect steps using Lee et al. algorithm on vertical acceleration."""

    data_set_ids = "vertical_acceleration"
    new_data_set_id = "lee"
    transform_function = detect_steps


class LeeStepCountWithoutBout(ExtractStep):
    """Extract step count with lee dataset without walking bouts."""

    def __init__(self, **kwargs):
        data_set_ids = "lee"
        definition = MeasureValueDefinitionPrototype(
            measure_name=AV("step count", "sc"),
            data_type="int16",
            validator=GREATER_THAN_ZERO,
            description="The number of steps detected with lee algorithm.",
        )
        super().__init__(
            data_set_ids=data_set_ids,
            transform_function=step_count,
            definition=definition,
            **kwargs,
        )


class LeeStepPowerWithoutBout(ExtractStep):
    """Extract step count with lee dataset without walking bouts."""

    def __init__(self, **kwargs):
        data_set_ids = ["vertical_acceleration", "lee"]
        definition = MeasureValueDefinitionPrototype(
            measure_name=AV("step power", "sp"),
            data_type="int16",
            validator=GREATER_THAN_ZERO,
            description="The integral of the centered acceleration magnitude "
            "between the first and last step divided by the "
            "number of steps computed with lee algorithm.",
        )
        super().__init__(
            data_set_ids=data_set_ids,
            transform_function=power_bout_div_steps,
            definition=definition,
            **kwargs,
        )


class LeeStepDurWithoutBout(AggregateRawDataSetColumn):
    """Extract step duration without walking bout."""

    def __init__(self, **kwargs):
        definition = MeasureValueDefinitionPrototype(
            measure_name=AV("step duration", "step_dur"),
            data_type="float64",
            unit="s",
            validator=GREATER_THAN_ZERO,
            description="The {aggregation} time of a detected step.",
        )
        super().__init__(
            "lee_step_duration",
            "step_duration",
            DEFAULT_AGGREGATIONS,
            definition,
            **kwargs,
        )


class LeeStepCount(ExtractStepCount):
    """Extract step count."""

    def __init__(self, **kwargs):
        data_set_ids = ["movement_bouts", "lee_with_walking_bouts"]
        super().__init__(data_set_ids=data_set_ids, **kwargs)


class LeeStepPower(ExtractPowerBoutDivSteps):
    """Extract step power."""

    def __init__(self, **kwargs):
        data_set_ids = [
            "movement_bouts",
            "vertical_acceleration",
            "lee_with_walking_bouts",
        ]
        super().__init__(data_set_ids=data_set_ids, **kwargs)


class LeeStepDur(ExtractStepDurationAll):
    """Extract step power."""

    def __init__(self, **kwargs):
        super().__init__(
            data_set_ids=["movement_bouts", "lee_with_walking_bouts_step_duration"],
            **kwargs,
        )


class LeeTransformHipRotation(HipRotationGroup):
    """Transform for positive and negative hip rotation."""

    def __init__(self, on_walking_bouts: bool, **kwargs):
        step_detection_id = "lee"
        if on_walking_bouts:
            step_detection_id += "_with_walking_bouts"
        super().__init__(
            step_detection_id=step_detection_id,
            on_walking_bouts=on_walking_bouts,
            **kwargs,
        )


class LeeHipRotation(ExtractHipRotation):
    """Extract Hip Rotation."""

    def __init__(self, **kwargs):
        data_set_id = "lee_with_walking_bouts"
        super().__init__(data_set_id=data_set_id, **kwargs)


class LeeHipRotationWithoutBout(ExtractHipRotationWithoutBouts):
    """Extract Hip Rotation."""

    def __init__(self, **kwargs):
        data_set_id = "lee"
        super().__init__(data_set_id=data_set_id, **kwargs)


class LeeMeasuresGroup(ProcessingStepGroup):
    """Extract Lee measures based on Lee Steps and a bout strategy."""

    def __init__(self, bout_strategy: BoutStrategyModality, **kwargs):
        steps: List[ProcessingStep] = [
            LeeStepCount(bout_strategy=bout_strategy.bout_cls),
            LeeStepPower(bout_strategy=bout_strategy.bout_cls),
            LeeStepDur(bout_strategy=bout_strategy.bout_cls),
            LeeHipRotation(bout_strategy=bout_strategy.bout_cls),
        ]
        super().__init__(steps, **kwargs)


class LeeMeasuresWithoutBoutGroup(ProcessingStepGroup):
    """Extract Lee measures based on Lee Steps and a bout strategy."""

    def __init__(self, **kwargs):
        steps: List[ProcessingStep] = [
            LeeStepCountWithoutBout(),
            LeeStepPowerWithoutBout(),
            LeeStepDurWithoutBout(),
            LeeHipRotationWithoutBout(),
        ]
        super().__init__(steps, **kwargs)
