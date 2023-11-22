"""Core data model for the analysis library."""
from collections import defaultdict
from dataclasses import dataclass
from functools import singledispatchmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, ValuesView, cast

import pandas as pd

from dispel.data.devices import Device
from dispel.data.epochs import Epoch
from dispel.data.flags import Flag, FlagMixIn
from dispel.data.levels import Level, LevelEpoch, LevelId, LevelIdType
from dispel.data.measures import MeasureSet, MeasureValue
from dispel.data.raw import RawDataSet
from dispel.data.values import ValueDefinition
from dispel.utils import plural


@dataclass(frozen=True)
class ReadingSchema:
    """Schema definition for reading."""

    #: The namespace of the schema
    namespace: str
    #: The name of the schema
    name: str
    #: The version of the schema
    version: str


class Evaluation(Epoch):
    """Evaluation information for a :class:`Reading`.

    The evaluation corresponds to the json related task, whereas the session corresponds
    to the group of tasks that the evaluation finds itself in.

    FIXME: DOC

    Attributes
    ----------
    uuid
        The unique unified identifier of the evaluation
    finished
        ``True`` if the concerned task has been finished normally. ``False`` otherwise.
    exit_reason
        The exit condition. It determines the type of interruption if the test was
        interrupted, as well as the reason for the end of the test if the test has
        been completed.
    user_id
        The identifier of the user
    """

    def __init__(
        self,
        *args,
        uuid: str,
        finished: Optional[bool] = None,
        exit_reason: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if self.is_incomplete:
            raise ValueError("Evaluation epoch must always be complete")

        self.uuid = uuid
        self.finished = finished
        self.exit_reason = exit_reason
        self.user_id = user_id

    def to_dict(self):
        """Retrieve values of evaluation as dictionary."""
        return {
            "evaluation_code": str(self.id),
            "start_date": str(self.start),
            "end_date": str(self.end),
            "uuid": self.uuid,
            "user_id": self.user_id if self.user_id else "",
            "is_finished": self.finished if self.finished else "",
            "exit_reason": self.exit_reason if self.exit_reason else "",
        }


class Session(Epoch):
    """Session information for a :class:`Reading`.

    The session corresponds to the group of tasks that the evaluation finds itself in.

    FIXME: DOC

    Attributes
    ----------
    uuid
        The unique unified identifier of the session
    evaluation_codes
        An iterable of task types available in the session. Ordered by display order.
    """

    def __init__(
        self,
        *args,
        uuid: Optional[str] = None,
        evaluation_codes: Optional[Iterable[str]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.uuid = uuid
        self.evaluation_codes = evaluation_codes


class Reading(FlagMixIn):
    """A data capture from an experiment.

    Attributes
    ----------
    evaluation
        The evaluation information for this reading
    session
        The session information for this reading
    measure_set
        A list of measures already processed on the device
    schema
        The schema of the reading
    date
        The time the reading was recorded
    device
        The device that captured the reading

    Parameters
    ----------
    evaluation
        The evaluation information for this reading
    session
        The session information for this reading
    levels
        An iterable of Level
    measure_set
        A list of measures already processed on the device
    schema
        The schema of the reading
    date
        The time the reading was recorded
    device
        The device that captured the reading
    """

    def __init__(
        self,
        evaluation: Evaluation,
        session: Optional[Session] = None,
        levels: Optional[Iterable[Level]] = None,
        measure_set: Optional[MeasureSet] = None,
        schema: Optional[ReadingSchema] = None,
        date: Any = None,
        device: Optional[Device] = None,
    ):
        super().__init__()
        self.evaluation = evaluation
        self.session = session
        self.measure_set: MeasureSet = measure_set or MeasureSet()
        self.schema = schema
        self.date = pd.Timestamp(date) if date else None
        self.device = device
        self._attempt: Dict[str, int] = defaultdict(int)

        # verify time frame compatibility
        if (
            self.session
            and not self.session.is_incomplete
            and not self.session.contains(self.evaluation)
        ):
            raise ValueError("Evaluation start and end must be within session")

        # create dictionary of levels
        self._levels: Dict[LevelId, Level] = {}

        # set level if arg is provided
        if levels:
            for level in levels:
                self.set(level)

    def get_level(self, level_id: Optional[LevelIdType] = None) -> Level:
        """Get level for a given level_id.

        Parameters
        ----------
        level_id
            The id identifying the level.

        Returns
        -------
        Level
            The level identified by ``level_id``. If no level id is provided and the
            reading contains only one level it will be returned. Otherwise, the function
            will raise a :class:`ValueError`.

        Raises
        ------
        ValueError
            If the given id does not match any existing level within the reading.
        ValueError
            If no id has been provided, and there are multiple levels withing the
            reading.
        """
        # check if an arg is provided
        if level_id:
            if isinstance(level_id, str):
                level_id = LevelId.from_str(level_id)  # type: ignore
            # check that this is a correct id
            if level_id not in self._levels:
                raise ValueError(
                    f"{level_id=} does not match any Level in {self._levels.keys()}"
                )
            return self._levels[level_id]  # type: ignore

        # if no level_id provided, check if there is only one level
        if len(self._levels) == 1:
            return next(iter(self._levels.values()))

        # if not, ask user for a level_id
        raise ValueError(
            f"There are {len(self._levels)} levels, please provide a level_id in"
            f" {self._levels.keys()}"
        )

    def __repr__(self) -> str:
        return f'<Reading: {plural("level", len(self))} ({self.flag_count_repr})>'

    def __iter__(self) -> Iterable[Tuple[LevelIdType, Level]]:
        yield from self._levels.items()

    def __len__(self) -> int:
        return len(self._levels)

    @property
    def empty(self) -> bool:
        """Check whether the reading is empty."""
        return len(self) == 0

    @property
    def levels(self) -> ValuesView[Level]:
        """Get a list of all Level in the reading."""
        return self._levels.values()

    @property
    def level_ids(self) -> List[LevelId]:
        """Get the list of level_id keys."""
        return [level.id for level in self._levels.values()]

    def has_raw_data_set(
        self,
        data_set_id: str,
        level_id: LevelIdType,
    ) -> bool:
        """Check whether the reading contains the desired raw data set.

        Parameters
        ----------
        data_set_id
            The id of the raw data set that will be searched for.
        level_id
            The level id in which the raw data set is to searched for.

        Returns
        -------
        bool
            ``True`` if the raw data set exists inside the given level. ``False``
            otherwise.
        """
        return self.get_level(level_id).has_raw_data_set(data_set_id)

    def get_raw_data_set(
        self,
        data_set_id: str,
        level_id: LevelIdType,
    ) -> RawDataSet:
        """Get the raw data set for a given data set id and a level.

        Parameters
        ----------
        data_set_id
            The id of the raw data set that will be retrieved.
        level_id
            The level id from which the raw data set is to retrieved.

        Returns
        -------
        RawDataSet
            The raw data set with the matching id.
        """
        return self.get_level(level_id).get_raw_data_set(data_set_id)

    def get_measure_set(self, level_id: Optional[LevelIdType] = None) -> MeasureSet:
        """Get measure_set from level identified with level_id."""
        if not level_id:
            return self.measure_set
        return self.get_level(level_id).measure_set

    def get_merged_measure_set(self) -> MeasureSet:
        """Get a measure set containing all the reading's measure values."""
        return sum(
            (self.measure_set, *(level.measure_set for level in self.levels)),
            MeasureSet(),
        )

    @singledispatchmethod
    def set(self, value, **kwargs):
        """Set a value inside a reading."""
        raise TypeError(f"Unsupported set type: {type(value)}")

    def _get_level(self, level: Optional[Union[LevelIdType, Level]] = None) -> Level:
        """Get level from id or level itself."""
        if isinstance(level, Level):
            return level
        return self.get_level(level)

    @set.register(MeasureSet)
    def _measure_set(
        self,
        value: MeasureSet,
        level: Optional[Union[LevelIdType, Level]] = None,
    ):
        if level is None:
            self.measure_set += value
        else:
            self._get_level(level).set(value)

    @set.register(MeasureValue)
    def _measure_value(
        self,
        value: MeasureValue,
        level: Optional[Union[LevelIdType, Level]] = None,
        epoch: Optional[LevelEpoch] = None,
    ):
        if epoch is not None:
            epoch.set(value)
        else:
            if level is None:
                measure_set = self.measure_set
            else:
                measure_set = self._get_level(level).measure_set

            measure_set.set(value)

    @set.register(RawDataSet)
    def _raw_data_set(
        self,
        value: RawDataSet,
        level: Union[LevelIdType, Level],
        concatenate: bool = False,
        overwrite: bool = False,
    ):
        self._get_level(level).set(value, concatenate=concatenate, overwrite=overwrite)

    @set.register(LevelEpoch)
    def _epoch_measure_set(self, value: LevelEpoch, level: Union[LevelIdType, Level]):
        self._get_level(level).set(value)

    @set.register(Level)
    def _level(self, value: Level):
        """Set a level."""
        level_id_str = str(value.id)
        for lev in self._levels:
            if str(lev).startswith(level_id_str) and level_id_str in self._attempt:
                self._attempt[level_id_str] += 1
                break
        if level_id_str not in self._attempt:
            new_level = LevelId.from_str(level_id_str)
            self._levels[new_level] = value  # type: ignore
            self._attempt[str(new_level.id)] = 1
        else:
            new_level_id_str = "-".join(
                [level_id_str, str(self._attempt[level_id_str]).zfill(2)]
            )
            value.id = cast(LevelId, LevelId.from_str(new_level_id_str))
            self._levels[value.id] = value
        # TODO: use sorting by effective time frame to ensure orders to
        #  attempts :
        #  level_ids = sorted(level_ids, key=lambda x:
        #  reading.get_level(x).effective_time_frame.start )
        self._levels[value.id].context.set(
            value=self._attempt[level_id_str],
            definition=ValueDefinition(
                id_="attempt", name=f"The attempt number: {self._attempt[level_id_str]}"
            ),
        )

    @set.register(Flag)
    def _set_flag(self, value: Flag):
        self.add_flag(value)


EntityType = Union[Reading, Level, RawDataSet, MeasureValue, LevelEpoch]
