"""A module containing models for raw data sets."""
from dataclasses import InitVar, dataclass, field
from enum import Enum
from operator import eq
from typing import Dict, Iterable, Optional

import pandas as pd

from dispel.data.flags import FlagMixIn
from dispel.data.validators import RangeValidator
from dispel.data.values import DefinitionId, ValueDefinition

#: Default columns ids for sensors
DEFAULT_COLUMNS = list("xyz")

#: Column ids of the accelerometer sensor
ACCELEROMETER_COLUMNS = [f"userAcceleration{x}" for x in "XYZ"]

#: Column ids of the gravity sensor
GRAVITY_COLUMNS = [f"gravity{x}" for x in "XYZ"]

#: Map user acceleration columns
USER_ACC_MAP = {
    "userAccelerationX": "x",
    "userAccelerationY": "y",
    "userAccelerationZ": "z",
    "ts": "ts",
}

#: Max pressure in screen measurements
PRESSURE_MAX = 6.666666666666667

#: A validator for pressure in screen measurements
PRESSURE_VALIDATOR = RangeValidator(lower_bound=0, upper_bound=PRESSURE_MAX)


class SensorType(str, Enum):
    """Abstract class for sensor type."""


class EmptyDataError(ValueError):
    """Exception raised when a pandas dataframe is empty.

    Parameters
    ----------
    message
        An optional message.
    """

    def __init__(self, message: Optional[str] = None):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        """Pretty print error message."""
        res = "Data frame is empty."
        if self.message:
            res += self.message
        return res


class MissingColumnError(ValueError):
    """Exception raised when a pandas dataframe is missing required column(s).

    Parameters
    ----------
    columns
        The set of the missing column names.
    message
        An optional message.
    """

    def __init__(self, columns: Iterable[str], message: Optional[str] = None):
        self.columns = columns
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        """Pretty print error message."""
        res = f"Data frame is missing the following columns: {self.columns}. "
        if self.message:
            res += self.message
        return res


class RawDataValueDefinition(ValueDefinition):
    """The definition of raw data set values.

    Attributes
    ----------
    is_index
        ``True`` if the values are part of the raw data set index. Otherwise, ``False``.
    """

    def __init__(
        self,
        id_: str,
        name: str,
        unit: Optional[str] = None,
        description: Optional[str] = None,
        data_type: Optional[str] = None,
        precision: Optional[int] = None,
        is_index: bool = False,
    ):
        super().__init__(
            id_=id_,
            name=name,
            unit=unit,
            description=description,
            data_type=data_type,
            precision=precision,
        )
        self.is_index = is_index


@dataclass
class RawDataSetSource:
    """The source of a raw data set."""

    #: The manufacturer producing the raw data set source
    manufacturer: str


def _create_value_definition_dict(
    definitions: Optional[Iterable[ValueDefinition]],
) -> Dict[DefinitionId, ValueDefinition]:
    """Turn iterables of definitions into a dictionary."""
    res = {}
    if definitions:
        for definition in definitions:
            if definition.id in res:
                raise ValueError(
                    f"Duplicate feature value definition for {definition.id}"
                )
            res[definition.id] = definition

    return res


@dataclass
class RawDataSetDefinition:
    """The definition of a raw data set."""

    #: The identifier of the raw data set definition
    id: str
    #: The source of the raw data set
    source: RawDataSetSource
    value_definitions_list: InitVar[Iterable[RawDataValueDefinition]]
    is_computed: bool = False
    """`True` if the raw data source is computed. ``False`` if it is a measured
    source without transformation, e.g. acceleration recorded from the low
    level APIs."""
    _value_definitions: Dict[DefinitionId, ValueDefinition] = field(init=False)

    def __post_init__(self, value_definitions_list):
        self._value_definitions = _create_value_definition_dict(value_definitions_list)

    @property
    def value_definitions(self):
        """Get the value definitions of the raw data set."""
        return self._value_definitions.values()

    def get_value_definition(self, id_: DefinitionId):
        """Get a value definition."""
        return self._value_definitions[id_]

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return (
            isinstance(other, RawDataSetDefinition)
            and self.id == other.id
            and self.source == other.source
            and eq(set(self.value_definitions), set(other.value_definitions))
            and self.is_computed == other.is_computed
        )


class RawDataSet(FlagMixIn):
    """A raw data set.

    Parameters
    ----------
    definition
        The definition of the raw data set
    data
        The data set
    """

    def __init__(self, definition: RawDataSetDefinition, data: pd.DataFrame):
        super().__init__()
        self.definition = definition
        self.data = data

        precision_exists = any(
            [d.precision is not None for d in self.definition.value_definitions]
        )
        if precision_exists:
            # if precision exists then store the original data prior to any rounding
            self.raw_data = data

        def_ids = {d.id for d in self.definition.value_definitions if not d.is_index}
        data_ids = set(data.columns)

        diff_data_columns = data_ids - def_ids
        if diff_data_columns:
            raise ValueError(f"Missing definition for column(s): {diff_data_columns}")

        diff_def_ids = def_ids - data_ids
        if diff_def_ids:
            raise ValueError(f"Missing columns for definition(s): {diff_def_ids}")

        # for each column definition check if precision exists and apply it to the data
        for col_def in self.definition.value_definitions:
            if col_def.precision is not None:
                self.data[col_def.id.id] = round(
                    self.data[col_def.id.id], ndigits=col_def.precision
                )

    @property
    def id(self) -> str:
        """Get the identifier from the definition of the raw data set."""
        return self.definition.id

    def __repr__(self) -> str:
        return f"<RawDataSet: {self.id} ({self.flag_count_repr})>"

    def concat(self, other: "RawDataSet") -> "RawDataSet":
        """Concatenate two raw data sets."""
        if self.definition != other.definition:
            raise ValueError("Can only concatenate data sets with equal definitions")
        return RawDataSet(self.definition, pd.concat([self.data, other.data]))
