"""A module containing models for levels."""

from functools import singledispatchmethod
from typing import Any, Dict, Iterable, List, Optional, Union, cast

import pandas as pd
from packaging import version

from dispel.data.epochs import Epoch, EpochDefinition
from dispel.data.flags import Flag
from dispel.data.measures import MeasureSet, MeasureValue
from dispel.data.raw import RawDataSet
from dispel.data.values import DefinitionId, ValueSet


def _modalities_to_level_id(modalities) -> str:
    if not isinstance(modalities, (str, list)):
        raise TypeError(
            f"Modalities should be of type str or list but has type {type(modalities)}"
        )
    if isinstance(modalities, str):
        modalities = [modalities]
    return "-".join(modalities)


class LevelId(DefinitionId):
    """Match evaluation code with level id.

    Parameters
    ----------
    modalities
        One or several context information on which the ``LevelId`` is built.
    """

    def __init__(self, modalities: Union[str, List[str]]):
        self.modalities = modalities
        id_ = _modalities_to_level_id(modalities)
        super().__init__(id_)

    @staticmethod
    def _modalities_to_level_id(modalities) -> str:
        if not isinstance(modalities, (str, list)):
            raise TypeError(
                "Modalities should be of type str or list but has type"
                f" {type(modalities)}"
            )
        if isinstance(modalities, str):
            modalities = [modalities]
        return "-".join(modalities)


LevelIdType = Union[str, LevelId]


class Context(ValueSet):
    """Contextual information for a level."""


class RawDataSetAlreadyExists(Exception):
    """Exception risen when a raw data set with an existent id is set in level.

    Parameters
    ----------
    raw_data_set_id
        The identifier of the raw data set in question.
    message
        An optional message.
    """

    def __init__(self, raw_data_set_id: str, level_id: LevelIdType, message: str = ""):
        self.raw_data_set_id = raw_data_set_id
        self.level_id = level_id
        self.message = f" {message}" if message else message
        super().__init__(
            f"Raw data set {raw_data_set_id} already exists in level "
            f"{self.level_id}.{self.message}"
        )


class LevelEpochMeasureValue(MeasureValue):
    """A measure value for a specific epoch.

    Parameters
    ----------
    epoch
        The epoch for which the measure value was extracted.

    Attributes
    ----------
    epoch
        The epoch for which the measure value was extracted.
    """

    def __init__(self, epoch: "LevelEpoch", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch = epoch

    def __hash__(self):
        return hash((self.definition, self.value, self.epoch))

    def to_dict(self, stringify: bool = False) -> Dict[str, Optional[Any]]:
        """Get a dictionary representation of measure information.

        Parameters
        ----------
        stringify
            ``True`` if all dictionary values are converted to strings. ``False``
            otherwise.

        Returns
        -------
        Dict[str, Optional[Any]]
            A dictionary summarizing measure value and epoch information.
        """
        res = super().to_dict(stringify=stringify)

        if self.epoch.definition is not None:
            res["epoch_id"] = self.epoch.definition.id
            res["epoch_name"] = self.epoch.definition.name

        if stringify:
            res["epoch_start"] = self._to_string(self.epoch.start)
            res["epoch_end"] = self._to_string(self.epoch.end)
        else:
            res["epoch_start"] = self.epoch.start
            res["epoch_end"] = self.epoch.end

        return res


class LevelEpoch(Epoch, MeasureSet):
    """Level specific epoch with measures.

    This data model allows to store measures that are specific to a given time point
    during an evaluation.
    """

    VALUE_CLS = LevelEpochMeasureValue


class Level(Epoch):
    """An entity to separate sub-task inside each test (Levels).

    FIXME: DOC

    Attributes
    ----------
    context
        Contextual information about the level
    measure_set
        A :class:'~dispel.data.measures.MeasureSet' of a given Level

    Parameters
    ----------
    id_
        The identifier of a given Level.
    start
        The timestamp of the beginning of the level
    end
        The timestamp of the end of the level
    context
        Contextual information about the level
    raw_data_sets
        An iterable of :class:'~dispel.data.raw.RawDataSet' of a given Level
    measure_set
        A :class:'~dispel.data.measures.MeasureSet' of a given Level
    epochs
        An iterable of :class:`~dispel.data.measures.EpochMeasureSet` to be added to the
        level.
    """

    def __init__(
        self,
        id_: Union[str, List[str], LevelId],
        start: Any,
        end: Any,
        context: Optional[Context] = None,
        raw_data_sets: Optional[Iterable[RawDataSet]] = None,
        measure_set: Optional[MeasureSet] = None,
        epochs: Optional[Iterable[LevelEpoch]] = None,
    ):
        if not isinstance(id_, LevelId):
            id_ = LevelId(id_)

        definition = EpochDefinition(id_=id_)
        super().__init__(start=start, end=end, definition=definition)

        self.context = context or Context()
        self.measure_set = measure_set or MeasureSet()

        # create dictionary of raw data sets
        self._raw_data_sets: Dict[str, RawDataSet] = {}

        # set raw data sets if arg is provided
        if raw_data_sets:
            for raw_data_set in raw_data_sets:
                self.set(raw_data_set)

        # create data frame for each epoch
        self._epochs = pd.DataFrame(columns=["definition_id", "start", "end", "epoch"])
        if epochs:
            for epoch in epochs:
                self.set(epoch)

    @property
    def id(self) -> LevelId:
        """Get the ID of the level from its definition.

        Returns
        -------
        LevelId
            The ID of the definition provided via `definition`.
        """
        assert self.definition is not None, "Require definition to access id"
        return cast(LevelId, self.definition.id)

    @id.setter
    def id(self, value: Union[str, DefinitionId]):
        """Set the ID of the level's definition.

        Parameters
        ----------
        value
            The ID to be set.
        """
        assert self.definition is not None, "Require definition to set id"
        self.definition.id = value  # type: ignore

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return f"<Level: {self.id} ({self.flag_count_repr})>"

    @property
    def raw_data_sets(self) -> List[RawDataSet]:
        """Get all raw data sets."""
        return list(self._raw_data_sets.values())

    def has_raw_data_set(self, id_: str) -> bool:
        """Return ``True`` if the level contains the desired raw data set."""
        return id_ in self._raw_data_sets

    def get_raw_data_set(self, id_: str) -> RawDataSet:
        """Get the raw data set for a given data set id.

        Parameters
        ----------
        id_
            The id of the raw data set to be returned

        Returns
        -------
        RawDataSet
            The raw data set with the matching id

        Raises
        ------
        ValueError
            If the given id does not correspond to any existing raw data set within the
            level.
        """
        if id_ not in self._raw_data_sets:
            raise ValueError(
                f'Unknown data set with id: "{id_}" for level_id == "{self.id}" '
                f"please provide an id within {list(self._raw_data_sets.keys())}"
            )

        return self._raw_data_sets[id_]

    @property
    def epochs(self) -> List[LevelEpoch]:
        """Get all epoch measure sets."""
        return self._epochs["epoch"].tolist()

    @singledispatchmethod
    def set(self, value, **kwargs):
        """Set a value inside a level."""
        raise TypeError(f"Unsupported set type: {type(value)}")

    @set.register(MeasureSet)
    def _set_measure_set(self, value: MeasureSet):
        self.measure_set += value

    @set.register(MeasureValue)
    def _set_measure_value(self, value: MeasureValue):
        self.measure_set.set(value)

    @set.register(RawDataSet)
    def _set_raw_data_set(
        self, value: RawDataSet, concatenate: bool = False, overwrite: bool = False
    ):
        if overwrite and concatenate:
            raise ValueError(
                "You cannot both concatenate and overwrite an existing raw data set. "
                "Only one of these arguments must be set to ``True``."
            )

        if (id_ := value.id) in self._raw_data_sets:  # pylint: disable=all
            if concatenate:
                value = value.concat(self.get_raw_data_set(id_))
            elif not overwrite:
                raise RawDataSetAlreadyExists(
                    id_, self.id, "Use overwrite=True to overwrite"
                )

        self._raw_data_sets[id_] = value

    @set.register(LevelEpoch)
    def _set_epoch(self, value: LevelEpoch):
        new_index = len(self._epochs)
        self._epochs.loc[new_index] = pd.Series(
            dict(
                definition_id=value.id if value.definition else None,
                start=value.start,
                end=value.end,
                epoch=value,
            )
        )

    @set.register(Flag)
    def _set_flag(self, value: Flag):
        self.add_flag(value)


class Modalities:
    """An entity to match level context modalities with a ``Level.id``.

    Parameters
    ----------
    mapping
        A dict composed of ``evaluation_code`` and level context information
    """

    def __init__(
        self,
        mapping: Optional[Dict[str, Any]] = None,
        app_version: Optional[str] = None,
    ):
        self.mapping = mapping
        self.app_version = None
        if app_version:
            self.app_version = version.parse(app_version)

    @property
    def is_default_mode(self):
        """Return whether the app version is the default mode."""
        return True

    def get_modalities_from_context(
        self, evaluation_code: str, context: Context
    ) -> Union[str, List[str]]:
        """Map level contexts to level modalities from an evaluation code."""
        if self.mapping is None:
            raise ValueError(
                f"No mapping has been provided for evaluation_code={evaluation_code}"
            )

        if evaluation_code not in self.mapping:
            raise ValueError(
                f"{evaluation_code=} does not match any test in {self.mapping.keys()}"
            )
        # find the modalities for the corresponding test in context
        modalities = self.mapping[evaluation_code](context)

        if not isinstance(modalities, list):
            modalities = [modalities]
        return modalities
