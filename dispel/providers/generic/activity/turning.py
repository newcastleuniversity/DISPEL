"""A module to detect and process turns during tests."""
from abc import ABCMeta
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from scipy import signal

from dispel.data.levels import Level
from dispel.data.measures import MeasureValueDefinitionPrototype
from dispel.data.raw import RawDataValueDefinition
from dispel.data.validators import GREATER_THAN_ZERO
from dispel.data.values import AbbreviatedValue as AV
from dispel.data.values import AVEnum
from dispel.processing import ProcessingStep
from dispel.processing.data_set import DataSetProcessingStepMixin, transformation
from dispel.processing.extract import (
    DEFAULT_AGGREGATIONS,
    AggregateRawDataSetColumn,
    ExtractStep,
    MeasureDefinitionMixin,
)
from dispel.processing.level import ProcessingStepGroup
from dispel.processing.transform import TransformStep

EL_GOHARY_TH_MAX = np.radians(15)
EL_GOHARY_TH_MIN = np.radians(5)
EL_GOHARY_TH_MERGE = pd.Timedelta(seconds=0.05)
EL_GOHARY_FINAL_TH_MERGE = pd.Timedelta(seconds=0.3)
TH_MIN_TURN_DURATION = 0.5
TH_MAX_TURN_DURATION = 10
TH_MIN_TURN_ANGLE_DEGREE = 45
U_REFINE_TH_MAX_TURN_DURATION = pd.Timedelta(3, "s")
U_REFINE_TH_CURR = np.radians(170)
U_REFINE_TH_LAST = np.radians(170)
U_REFINE_TH_SUM = np.radians(270)
U_REFINE_TH_MIN = np.radians(90)

TURN_DATA_SET_DEFINITION = [
    RawDataValueDefinition("start", "The start of the turn"),
    RawDataValueDefinition("end", "The end of the turn"),
    RawDataValueDefinition("duration", "The duration of the turn", "s"),
    RawDataValueDefinition("angle", "The angle of the turn", "rad"),
    RawDataValueDefinition("turn_cls", "The Turn class"),
]

TURN_SPEED_PROPERTIES_SET_DEFINITION = [
    RawDataValueDefinition("mean", "mean of turn speed across a turn", "rad/s"),
    RawDataValueDefinition("q95", "95th centile of turn speed across a turn", "rad/s"),
]


class Turn:
    """Class to encapsulate turns and turn related gyroscope data.

    Parameters
    ----------
    start
        The start date time of the turn.
    end
        The end date time of the turn.
    data
        The angular velocity time series of the turn. The data set should
        ensure to be at least of the duration of time provided with
        ``start`` and ``end``. It should be in rad/s and sampling has to be
        at a constant frequency.

    Attributes
    ----------
    start
        The start date time of the turn.
    end
        The end date time of the turn.
    """

    def __init__(self, start: datetime, end: datetime, data: pd.Series):
        self.start = start
        self.end = end
        self._data = data

    def expand(self, threshold: float) -> "Turn":
        """Expand the ``start`` and ``end`` of the turn to the given threshold.

        This expands the turn until start and end are below the provided
        threshold of turn speed. The expansion relies on the data provided
        during the construction of the turn and should not be confused with
        what is available via :data:`data`, which is always limited to the
        boundaries specified with :data:`start` and :data:`end`.

        Parameters
        ----------
        threshold
            The threshold until which to expand the turn.

        Returns
        -------
        Turn
            A new turn with expanded start and end time stamps based on
            associated data.
        """
        below = self._data * self.direction < threshold

        before = below[: self.start]  # type: ignore
        after = below[self.end :]  # type: ignore

        start, end = self.start, self.end
        freq = self._data.index.freq.delta

        # adjust start
        if not before[before].empty:
            start = before[before].index[-1] + freq
        elif not before.any():
            start = before.index[0]

        # adjust end
        if not after[after].empty:
            end = after[after].index[0] - freq
        elif not after.any():
            end = after.index[-1]

        return Turn(start, end, self._data)

    @property
    def data(self) -> pd.Series:
        """Get the angular velocity data associated with the turn.

        Returns
        -------
        pandas.Series
            Angular velocity series between :data:`start` and :data:`end` of
            the turn.
        """
        return self._data[self.start : self.end]  # type: ignore

    @property
    def duration(self) -> float:
        """Get the duration of the turn.

        Returns
        -------
        float
            The duration of the turn in seconds.
        """
        return (self.end - self.start).total_seconds()

    @property
    def direction(self) -> int:
        """Get the direction of the turn.

        Returns
        -------
        int
            The direction of the turn. If the turning direction is positive
            ``1`` is returned. Otherwise ``-1``.
        """
        return 1 if self.angle > 0 else -1

    @property
    def angle(self) -> float:
        """Get the angle of the turn.

        Returns
        -------
        float
            The angle of the turn in rad/s.
        """
        delta = self._data.index.freq.delta.total_seconds()
        return self.data.sum() * delta

    def merge(self, other) -> "Turn":
        """Merge this turn with another one.

        Parameters
        ----------
        other
            The other turn to merge this one with.

        Returns
        -------
        Turn
            The new merged turn. The new turn uses the earlier start and later
            end of both, respectively. The data will be based on this turn.
        """
        return Turn(min(self.start, other.start), max(self.end, other.end), self._data)

    def __repr__(self):
        return (
            f"<Turn: {self.start} - {self.end} ({self.duration} s, "
            f"{self.angle.round(3)} rad, {self.direction})>"
        )


class TransformAbsTurnSpeed(TransformStep):
    """Get absolute turns speed."""

    new_data_set_id = "abs_turn_speed"
    definitions = [
        RawDataValueDefinition(
            "abs_turn_speed", "The absolute value of the average turn speed"
        )
    ]

    @staticmethod
    @transformation
    def _abs_turn_speed(data: pd.DataFrame) -> pd.Series:
        return (data["angle"] / data["duration"]).abs()


def merge_turns(
    turns: List[Turn],
    th_time: pd.Timedelta = pd.Timedelta(0, "s"),
    th_curr: float = np.inf,
    th_last: float = np.inf,
    th_sum: float = np.inf,
) -> List[Turn]:
    """Merge turns that add up to approximately u-turns.

    This function takes a list of turns and merges each of them with its
    preceding turn if that ended less than a time threshold ago and the
    consecutive turns are below a set of thresholds in size.

    Parameters
    ----------
    turns
        A list of Turn objects to be potentially merged.
    th_time
        A pandas.Timedelta containing the maximum time distance between turns
        that should be used for merging.
    th_curr
        A float indicating the maximum value of the current turn to be
        considered for merging
    th_last
        A float indicating the maximum value of the last turn to be
        considered for merging
    th_sum
        A float indicating the maximum value of the sum of current and
        previous turn to be considered for merging

    Returns
    -------
    List[Turn]
        A list of merged turns.
    """
    merged_turns: List[Turn] = []
    for turn in turns:
        # initial turn
        if not merged_turns:
            merged_turns.append(turn)
        else:
            last_turn = merged_turns[-1]

            condition = {
                "dir": turn.direction == last_turn.direction,
                "time": (turn.start - last_turn.end) < th_time,
                "curr": abs(turn.angle) < th_curr,
                "last": abs(last_turn.angle) < th_last,
                "sum": abs(turn.angle) + abs(last_turn.angle) < th_sum,
            }

            if all(condition.values()):
                merged_turns[-1] = last_turn.merge(turn)
            else:
                merged_turns.append(turn)

    return merged_turns


def el_gohary_detect_turns(data: pd.Series) -> List[Turn]:
    """Detect turns based on the El Gohary et al. algorithm [1]_ .

    This method performs the detection after aligning and filtering the
    gyroscope time series (see El Gohary et al. algorithm 1, row 4).

    Parameters
    ----------
    data
        A pandas series of angular velocity used to search for turns.

    Returns
    -------
    List[Turn]
        A list of detected turns.

    References
    ----------
    .. [1] El-Gohary, Mahmoud, et al. "Continuous monitoring of turning in
       patients with movement disability." Sensors 14.1 (2014): 356-369.
       https://doi.org/10.3390/s140100356
    """
    # detect peaks
    peak_index, _ = signal.find_peaks(data.abs(), EL_GOHARY_TH_MAX)
    peaks = pd.Series(data[peak_index], index=data.index[peak_index])

    # initialize turns with peaks found
    turns = [Turn(x, x, data) for x in peaks.index.tolist()]

    # expand turns
    expanded_turns = [t.expand(EL_GOHARY_TH_MIN) for t in turns]

    # merge turns
    merged_turns = merge_turns(expanded_turns, EL_GOHARY_TH_MERGE)

    # filter turns
    filtered_turns = []
    for turn in merged_turns:
        if TH_MIN_TURN_DURATION < turn.duration < TH_MAX_TURN_DURATION and abs(
            turn.angle
        ) >= np.radians(TH_MIN_TURN_ANGLE_DEGREE):
            filtered_turns.append(turn)

    return filtered_turns


def remove_turns_below_thres(turns: List[Turn], th_min: float = 0.0) -> List[Turn]:
    """Remove turns that are below a given threshold.

    This function receives a list of turns and removes the ones that are lower
    than a given threshold.

    Parameters
    ----------
    turns
        A list of Turn objects to be potentially merged.
    th_min
        Minimum angle in radians

    Returns
    -------
    List[Turn]
        A list of merged turns.
    """
    filtered_turns = []
    for turn in turns:
        if np.abs(turn.angle) > th_min:
            filtered_turns.append(turn)

    return filtered_turns


class ElGoharyTurnDetection(TransformStep):
    """El Gohary et al. turn detection processing step.

    This processing step is to apply the :func:`el_gohary_detect_turns` to
    an angular velocity time series that has undergone the described filtering.

    Parameters
    ----------
    component
        The column name of the series contained in raw data set.
    """

    definitions = TURN_DATA_SET_DEFINITION

    def __init__(self, component: str, *args, **kwargs):
        self.component = component
        super().__init__(*args, **kwargs)

    def get_new_data_set_id(self) -> str:
        """Get the id of the new data set to be created."""
        return f"{self.data_set_ids}_{self.component}_turns"

    @transformation
    def _detect_turns(self, data: pd.DataFrame) -> pd.DataFrame:
        turns = el_gohary_detect_turns(data[self.component])
        # Check if turns is empty
        if len(turns) == 0:
            return pd.DataFrame(columns=[_def.id for _def in TURN_DATA_SET_DEFINITION])
        out = [
            {
                "start": turn.start,
                "end": turn.end,
                "duration": turn.duration,
                "angle": turn.angle,
                "turn_cls": turn,
            }
            for turn in turns
        ]

        return pd.DataFrame(out)


def refine_turns(
    data: pd.Series,
    turns_df: pd.DataFrame,
    th_time: pd.Timedelta = pd.Timedelta(0, "s"),
    th_curr: float = np.inf,
    th_last: float = np.inf,
    th_sum: float = np.inf,
    th_min: float = 0.0,
) -> List[Turn]:
    """Refine turns from El-Gohary output.

    This function refines turns by merging consecutive turns with time distance
    (defined as the period between the end of a turn and the start of the next)
    less than a specified threshold as well as a set of thresholds constraining
    the size of refined merged angles.

    Parameters
    ----------
    data
        A pandas series of angular velocity used to search for turns.
    turns_df
        A pd.DataFrame containing the turns to be refined.
    th_time
        A pandas.Timedelta containing the maximum time distance between turns
        that should be used for merging.
    th_curr
        A float indicating the maximum value of the current turn to be
        considered for merging (in radians)
    th_last
        A float indicating the maximum value of the last turn to be
        considered for merging (in radians)
    th_sum
        A float indicating the maximum value of the sum of current and
        previous turn to be considered for merging (in radians)
    th_min
        A float indicating the minimum value of the angles to be kept in the
        data (in radians)

    Returns
    -------
    List[Turn]
        A list of refined turns.
    """
    # convert dataframe to list of Turn objects
    turns = [Turn(x, y, data) for x, y in zip(turns_df["start"], turns_df["end"])]

    # merge each turn with its previous if it is of same direction and the
    # previous ended EL_GOHARY_FINAL_TH_MERGE seconds ago or less
    merged_turns = merge_turns(turns, th_time, th_curr, th_last, th_sum)

    # remove turns smaller than threshold
    turns = remove_turns_below_thres(merged_turns, th_min)

    return turns


class RefineTurns(TransformStep):
    """El Gohary et al. turn detection processing step.

    This processing step is to apply the :func:`el_gohary_detect_turns` to
    a angular velocity time series that has undergone the described filtering.

    Parameters
    ----------
    component
        The column name of the series contained in raw data set.
    """

    definitions = TURN_DATA_SET_DEFINITION
    th_time = EL_GOHARY_FINAL_TH_MERGE

    def __init__(self, component: str, *args, **kwargs):
        self.component = component
        super().__init__(*args, **kwargs)

    def get_new_data_set_id(self) -> str:
        """Get new data set id."""
        return f"{self.data_set_ids[1]}_refined"  # type: ignore

    @transformation
    def _refine_turns(
        self, angular_velocity: pd.DataFrame, turns: pd.DataFrame
    ) -> pd.DataFrame:
        turns = refine_turns(angular_velocity[self.component], turns, self.th_time)
        # Check if turns is empty
        if len(turns) == 0:
            return pd.DataFrame(columns=[_def.id for _def in TURN_DATA_SET_DEFINITION])
        return pd.DataFrame(
            [
                {
                    "start": turn.start,
                    "end": turn.end,
                    "duration": turn.duration,
                    "angle": turn.angle,
                    "turn_cls": turn,
                }
                for turn in turns
            ]
        )


class RefineUTurns(TransformStep):
    """Refine Turns based on assumption of 180 degrees.

    This processing step is to consecutive turns occuring soon after another
    and merge them if they are smaller than a threshold.

    Parameters
    ----------
    component
        The column name of the series contained in raw data set.
    """

    definitions = TURN_DATA_SET_DEFINITION
    th_time = U_REFINE_TH_MAX_TURN_DURATION
    th_curr = U_REFINE_TH_CURR
    th_last = U_REFINE_TH_LAST
    th_sum = U_REFINE_TH_SUM
    th_min = U_REFINE_TH_MIN

    def __init__(self, component: str, *args, **kwargs):
        self.component = component
        super().__init__(*args, **kwargs)

    def get_new_data_set_id(self) -> str:
        """Get new data set id."""
        return f"{self.data_set_ids[1]}_u_refined"  # type: ignore

    @transformation
    def _refine_turns(
        self,
        angular_velocity: pd.DataFrame,
        turns: pd.DataFrame,
    ) -> pd.DataFrame:
        turns = refine_turns(
            angular_velocity[self.component],
            turns,
            self.th_time,
            self.th_curr,
            self.th_last,
            self.th_sum,
            self.th_min,
        )
        # Check if turns is empty
        if len(turns) == 0:
            return pd.DataFrame(columns=[_def.id for _def in TURN_DATA_SET_DEFINITION])
        return pd.DataFrame(
            [
                {
                    "start": turn.start,
                    "end": turn.end,
                    "duration": turn.duration,
                    "angle": turn.angle,
                    "turn_cls": turn,
                }
                for turn in turns
            ]
        )


class WithinTurnSpeed(TransformStep):
    """Compute within-turn properties of the turn speed.

    This processing step is to compute sensor-derived aggregated properties
    of the angular velocity signal within each turn.

    Parameters
    ----------
    component
        The column name of the series contained in raw data set.
    """

    definitions = TURN_SPEED_PROPERTIES_SET_DEFINITION

    def __init__(self, component: str, *args, **kwargs):
        self.component = component
        super().__init__(*args, **kwargs)

    def get_new_data_set_id(self) -> str:
        """Get new data set id."""
        return f"abs_turn_speed_{self.component}"  # type: ignore

    @transformation
    def _compute_within_turn_aggregates(
        self, angular_velocity: pd.DataFrame, turns: pd.DataFrame
    ) -> pd.DataFrame:
        if turns.shape[0] == 0:
            return pd.DataFrame(columns=["mean", "q95"])

        aggregates_list = []
        for _, turn in turns.iterrows():
            turn_properties = {}
            start = turn.start
            end = turn.end
            angular_velocity_turn = abs(angular_velocity.loc[start:end, self.component])
            turn_properties["mean"] = np.mean(angular_velocity_turn)
            turn_properties["q95"] = np.percentile(angular_velocity_turn, 95)
            aggregates_list.append(turn_properties)

        return pd.DataFrame(aggregates_list)


class TurnModality(AVEnum):
    """An enumeration for turn modalities."""

    ALL = ("all turns", "all")
    FIRST_FIVE = ("first five turns", "first5")
    FIRST_FOUR = ("first four turns", "first4")
    TWO_THREE_FOUR = ("second, third, and forth turn", "2to4")


TURN_ID_MAPPING = {
    TurnModality.FIRST_FIVE: list(range(5)),
    TurnModality.FIRST_FOUR: list(range(4)),
    TurnModality.TWO_THREE_FOUR: list(range(1, 4)),
}


class TurnModalityMixIn(
    MeasureDefinitionMixin, DataSetProcessingStepMixin, metaclass=ABCMeta
):
    """A modality to filter on subsets of turns.

    Parameters
    ----------
    turn_modality
        A TurnModality object denoting the turns to be used.
    """

    def __init__(self, turn_modality: TurnModality, *args, **kwargs):
        self.turn_modality = turn_modality
        super().__init__(*args, **kwargs)

    def get_definition(self, **kwargs):
        """Get definition."""
        modalities = kwargs.pop("modalities", [])
        modalities.append(self.turn_modality.av)
        return super().get_definition(modalities=modalities, **kwargs)

    def get_data_frames(self, level: Level) -> List[pd.DataFrame]:
        """Get data frames."""
        dfs = super().get_data_frames(level)
        if self.turn_modality is TurnModality.ALL:
            return dfs

        return [dfs[0][dfs[0].index.isin(TURN_ID_MAPPING[self.turn_modality])]]


class ExtractNumberOfTurns(TurnModalityMixIn, ExtractStep):
    """Extract the number of turns during the utt test."""

    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("number of turns", "turns"),
        data_type="int",
        validator=GREATER_THAN_ZERO,
        description=f"The number of turns detected during the "
        f"{{task_name}} using the El Gohary et al. 2014 "
        f"(https://dx.doi.org/10.3390/s140100356) "
        f"algorithm. Only turns with durations between "
        f"{TH_MIN_TURN_DURATION} and "
        f"{TH_MAX_TURN_DURATION} s and turn angles over "
        f"{TH_MIN_TURN_ANGLE_DEGREE}° are considered.",
    )

    transform_function = len


class AggregateAbsTurnSpeed(TurnModalityMixIn, AggregateRawDataSetColumn):
    """Aggregate absolute turn speed based on modality."""

    aggregations = DEFAULT_AGGREGATIONS

    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("absolute turn speed", "ts"),
        unit="rad/s",
        data_type="float64",
        validator=GREATER_THAN_ZERO,
        description=f"The {{aggregation}} absolute turn speed detected "
        f"in {{task_name}} using the El Gohary et al. 2014 "
        f"(https://dx.doi.org/10.3390/s140100356) "
        f"algorithm. Only turns with durations between "
        f"{TH_MIN_TURN_DURATION} and "
        f"{TH_MAX_TURN_DURATION} s and turn angles over "
        f"{TH_MIN_TURN_ANGLE_DEGREE}° are considered.",
    )


class AggregateTurnDuration(TurnModalityMixIn, AggregateRawDataSetColumn):
    """Aggregate turn duration based on modality."""

    aggregations = DEFAULT_AGGREGATIONS

    definition = MeasureValueDefinitionPrototype(
        measure_name=AV("turn duration", "dur"),
        unit="s",
        data_type="float64",
        validator=GREATER_THAN_ZERO,
        description=f"The {{aggregation}} turn duration "
        f"in {{task_name}} using the El Gohary et al. 2014 "
        f"(https://dx.doi.org/10.3390/s140100356) "
        f"algorithm. Only turns with durations between "
        f"{TH_MIN_TURN_DURATION} and "
        f"{TH_MAX_TURN_DURATION} s and turn angles over "
        f"{TH_MIN_TURN_ANGLE_DEGREE}° are considered.",
    )


class AggregateTurnSpeedProperties(TurnModalityMixIn, AggregateRawDataSetColumn):
    """Aggregate properties based on modality."""

    def __init__(self, turn_modality, data_set_id, column_id):
        self.turn_modality = turn_modality
        self.data_set_id = data_set_id
        self.column_id = column_id

        super().__init__(
            self.turn_modality,
            self.data_set_id,
            self.column_id,
            aggregations=DEFAULT_AGGREGATIONS,
            definition=MeasureValueDefinitionPrototype(
                measure_name=AV(
                    f"{self.column_id}_{self.data_set_id}",
                    f"{self.column_id}_{self.data_set_id}",
                ),
                unit="rad/s",
                data_type="float64",
                validator=GREATER_THAN_ZERO,
                description=f"The {{aggregation}} turn properties detected "
                f"in {{task_name}} using the El Gohary et al. 2014"
                f"(https://dx.doi.org/10.3390/s140100356) "
                f"algorithm. Only turns with durations between "
                f"{TH_MIN_TURN_DURATION} and "
                f"{TH_MAX_TURN_DURATION} s and turn angles over "
                f"{TH_MIN_TURN_ANGLE_DEGREE}° are considered.",
            ),
        )


class ExtractTurnMeasures(ProcessingStepGroup):
    """Extract Turn Measures based on Turn Selection Modality."""

    def __init__(self, turns_data_set_id, turn_modality, **kwargs):
        steps: List[ProcessingStep] = [
            ExtractNumberOfTurns(
                turn_modality=turn_modality, data_set_ids=turns_data_set_id
            ),
            AggregateTurnDuration(
                turn_modality=turn_modality,
                data_set_id=turns_data_set_id,
                column_id="duration",
            ),
            AggregateAbsTurnSpeed(
                turn_modality=turn_modality,
                data_set_id="abs_turn_speed",
                column_id="abs_turn_speed",
            ),
            AggregateTurnSpeedProperties(
                turn_modality=turn_modality,
                data_set_id="abs_turn_speed_x",
                column_id="mean",
            ),
            AggregateTurnSpeedProperties(
                turn_modality=turn_modality,
                data_set_id="abs_turn_speed_x",
                column_id="q95",
            ),
            AggregateTurnSpeedProperties(
                turn_modality=turn_modality,
                data_set_id="abs_turn_speed_norm",
                column_id="mean",
            ),
            AggregateTurnSpeedProperties(
                turn_modality=turn_modality,
                data_set_id="abs_turn_speed_norm",
                column_id="q95",
            ),
        ]
        super().__init__(steps, **kwargs)
