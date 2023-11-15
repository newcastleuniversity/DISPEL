"""A module containing models for flags.

Flags are basically a structured way to mark the data contained in a certain entity as
deviating or invalidating assumptions while processing.
"""
import operator
from abc import ABC
from collections import Counter
from copy import deepcopy
from dataclasses import InitVar, dataclass, field
from functools import singledispatchmethod
from typing import (
    Callable,
    Generic,
    Iterable,
    List,
    Optional,
    Set,
    TypeVar,
    Union,
    cast,
)

from dispel.data.values import AbbreviatedValue as AV
from dispel.data.values import AVEnum, DefinitionId
from dispel.utils import plural, raise_multiple_errors


class FlagType(AVEnum):
    """An enumeration for flag types."""

    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"


class FlagSeverity(AVEnum):
    """An enumeration for flag severity."""

    DEVIATION = "deviation"
    INVALIDATION = "invalidation"


class FlagId(DefinitionId):
    """The identifier of an entity flag for a task.

    Parameters
    ----------
    task_name
        The name and abbreviation of the task. Note that if no abbreviation is provided
        the name is used directly in the id.
    flag_name
        The name of the flag and its abbreviation.
    flag_type
        The type of the flag. See :class:`~dispel.data.flags.FlagType`.
    flag_severity
        The severity of the flag. See :class:`~dispel.data.flags.FlagSeverity`.

    Notes
    -----
    The abbreviations of values are passed using
    :class:`~dispel.data.values.AbbreviatedValue`. To generate the actual id the `.abbr`
    accessor is used. If one passes only strings, the class actually wraps those into
    ``AbbreviatedValue`` instances.

    Examples
    --------
    >>> from dispel.data.values import AbbreviatedValue as AV
    >>> from dispel.data.flags import FlagId, FlagType
    >>> FlagId(
    ...     task_name=AV('Cognitive Processing Speed', 'CPS'),
    ...     flag_name=AV('tilt angle', 'ta'),
    ...     flag_type=FlagType.BEHAVIORAL,
    ...     flag_severity=FlagSeverity.DEVIATION,
    ... )
    cps-behavioral-deviation-ta
    """

    def __init__(
        self,
        task_name: Union[str, AV],
        flag_name: Union[str, AV],
        flag_type: Union[str, FlagType],
        flag_severity: Union[str, FlagSeverity],
    ):
        self.task_name = AV.wrap(task_name)
        self.flag_name = AV.wrap(flag_name)
        # flag type
        if isinstance(flag_type, str):
            flag_type = FlagType.from_abbr(flag_type)
        self.flag_type = cast(FlagType, flag_type).av
        # flag severity
        if isinstance(flag_severity, str):
            flag_severity = FlagSeverity.from_abbr(flag_severity)
        self.flag_severity = cast(FlagSeverity, flag_severity).av

        id_ = "-".join(
            (
                self.task_name.abbr.lower(),
                self.flag_type.abbr.lower(),
                self.flag_severity.abbr.lower(),
                self.flag_name.abbr.lower(),
            )
        )

        super().__init__(id_)

    @classmethod
    def from_str(cls, value: str) -> "FlagId":
        """Create a flag id from a string representation.

        Parameters
        ----------
        value
            The string from which the flag id is to be constructed. It ought to respect
            the following format ``<task_name>-<flag_type>-<flag_name>`` where the flag
            type should one of the enumerations defined in
            :class:`~dispel.data.flags.FlagType`.

        Returns
        -------
        FlagId
            The initialised flag identifier.

        Raises
        ------
        ValueError
            If the flag string representation does not respect the required format.
        """
        components = value.split("-")
        if len(components) != 4:
            raise ValueError(
                "Flag Id format is not respected. Please provide an id that follows "
                "the following format: ``<task_name>-<flag_type>-<flag_name>``"
            )
        return cls(
            task_name=components[0],
            flag_name=components[3],
            flag_type=components[1],
            flag_severity=components[2],
        )

    def format(self, *args, **kwargs) -> "FlagId":
        """Format an flag identifier."""
        return FlagId(
            task_name=self.task_name,
            flag_name=self.flag_name.format(*args, **kwargs),
            flag_type=FlagType.from_abbr(self.flag_type.abbr),
            flag_severity=FlagSeverity.from_abbr(self.flag_severity.abbr),
        )


FlagIdType = Union[str, FlagId]


@dataclass
class Flag:
    """A class for entity flag."""

    #: The flag identifier (string or id format)
    id_: InitVar[FlagIdType]

    #: The flag identifier
    id: FlagId = field(init=False)

    #: The detailed reason for the flag
    reason: str

    #: Stop processing
    stop_processing: bool = False

    def __post_init__(self, id_: FlagIdType):
        if isinstance(id_, str):
            self.id = FlagId.from_str(id_)
        elif isinstance(id_, FlagId):
            self.id = id_
        else:
            raise TypeError(
                "Flag id should be either a convertible string id or an "
                "FlagId class."
            )

    def __hash__(self):
        return hash((self.id, self.reason, self.stop_processing))

    def format(self, *args, **kwargs) -> "Flag":
        """Format an flag."""
        return Flag(
            id_=self.id.format(*args, **kwargs),
            reason=self.reason.format(*args, **kwargs),
            stop_processing=self.stop_processing,
        )


class FlagAlreadyExists(Exception):
    """Exception raised when the same flag is added twice to an entity.

    Parameters
    ----------
    flag
        The duplicated flag.
    """

    def __init__(self, flag: Flag):
        message = (
            f"If the flag to be added ({flag}) corresponds to another use case, please "
            "make sure to specify a different reason at least."
        )
        super().__init__(message)


class FlagNotFound(Exception):
    """Exception raised when a flag is not found in the mix in.

    Parameters
    ----------
    flag_id
        The flag id.
    entity
        The flag Mix-In entity where the flag was not found.
    message
        An optional error message to be added at the end.
    """

    def __init__(
        self,
        flag_id: FlagIdType,
        entity: "FlagMixIn",
        message: str = "",
    ):
        complete_message = (
            f"No flag with {flag_id=} has been found for {entity}. {message}"
        )
        super().__init__(complete_message)


def verify_flag_uniqueness(flags: List[Flag]):
    """Check whether all provided flags are unique.

    Parameters
    ----------
    flags
        The flags whose uniqueness is to be verified.

    Raises
    ------
    FlagAlreadyExists
        Per non-unique flag.
    """
    accumulated_errors: List[Exception] = []
    counter = Counter(flags)
    for flag, _ in filter(lambda i: i[1] > 1, counter.items()):
        accumulated_errors.append(FlagAlreadyExists(flag))

    raise_multiple_errors(accumulated_errors)


class FlagMixIn(ABC):
    """Flag Mix-In class that groups multiple flags."""

    def __init__(self, *args, **kwargs):
        self._flags: List[Flag] = []
        super().__init__(*args, **kwargs)

    @property
    def flag_ids(self) -> Set[FlagId]:
        """Get the unique ids of the flags in the mix in."""
        return set(map(lambda i: i.id, self._flags))

    @property
    def flag_count(self) -> int:
        """Get the number of flags in the mix in."""
        return len(self._flags)

    @property
    def flag_count_repr(self) -> str:
        """Get the string representation of the flag count."""
        return plural("flag", self.flag_count)

    @property
    def is_valid(self) -> bool:
        """Return whether the flag mix in contains no flags."""
        return self.flag_count == 0

    def get_flags(self, flag_id: Optional[FlagIdType] = None) -> List[Flag]:
        """Retrieve flags from the mix in.

        Parameters
        ----------
        flag_id
            The id corresponding to the flags that are to be retrieved. If ``None`` is
            provided, all flags will be returned.

        Returns
        -------
        List[Flag]
            The flags corresponding to the given id.

        Raises
        ------
        FlagNotFound
            If the given flag id does not correspond to any flag in the mix-in.
        """
        if flag_id is None:
            return deepcopy(self._flags)

        flags = list(filter(lambda i: i.id == flag_id, self._flags))
        if len(flags) == 0:
            raise FlagNotFound(
                flag_id,
                self,
                f"Please provide one of the following ids {self.flag_ids} or nothing "
                "to retrieve all flags.",
            )
        return flags

    @singledispatchmethod
    def has_flag(self, value):
        """Return whether the flag is inside the mix in."""
        raise TypeError(f"Unsupported type: {type(value)}")

    @has_flag.register(Flag)
    def _flag(self, flag: Flag) -> bool:
        """Return whether the flag is inside the mix in.

        Parameters
        ----------
        flag
            The flag whose existence is the mix in is to be checked.

        Returns
        -------
        bool
            ``True`` if the flag exists inside the mix in. ``False`` otherwise.
        """
        return flag in self._flags

    @has_flag.register(str)
    @has_flag.register(FlagId)
    def _flag_id(self, flag_id: FlagIdType) -> bool:
        """Return whether the flag id is inside the mix in.

        Parameters
        ----------
        flag_id
            The flag id whose existence is the mix in is to be checked.

        Returns
        -------
        bool
            ``True`` if the flag exists inside the mix in. ``False`` otherwise.
        """
        return flag_id in self.flag_ids

    def add_flag(self, flag: Flag, ignore_duplicates: bool = False):
        """Add a flag to the flag mix in.

        Parameters
        ----------
        flag
            The flag to be added to the mix in.
        ignore_duplicates
            Set to ``True`` to ignore if a flag with the same id is already present.
            ``False`` will raise an ``FlagAlreadyExists``

        Raises
        ------
        FlagAlreadyExists
            If `ignore_duplicates` is `False` and the flag already exists.
        """
        if (exists := self.has_flag(flag)) and not ignore_duplicates:
            raise FlagAlreadyExists(flag)

        if not exists:
            self._flags.append(flag)

    def add_flags(
        self,
        flags: Union[Iterable[Flag], "FlagMixIn"],
        ignore_duplicates: bool = False,
    ):
        """Add multiple flags to mix in.

        Parameters
        ----------
        flags
            The flags to be added. It can both an iterable of flags or an instance of
            ``FlagMixIn``.
        ignore_duplicates
            Set to `True` to ignore if a flag with the same id is already present.
            ``False`` will raise an ``FlagAlreadyExists``
        """
        if isinstance(flags, FlagMixIn):
            flags = flags.get_flags()

        for flag in flags:
            self.add_flag(flag, ignore_duplicates)


WrappedResultType = TypeVar("WrappedResultType", float, bool, int)


class WrappedResult(FlagMixIn, Generic[WrappedResultType]):
    """A wrapped result to carry potential flags.

    This class provides a convenient way to add flags to values from extract steps that
    are known to be invalid. This avoids having to write a separate flag step and is
    useful in cases where the information to flag a result is only accessible in the
    extract function.

    Parameters
    ----------
    feature_value
        The value of the feature returned by the extraction function.

    Attributes
    ----------
    feature_value
        The value of the feature returned by the extraction function.

    Examples
    --------
    Assuming we wanted to flag features directly inside a custom extraction function
    based on some metrics calculated, one can do

    >>> from dispel.processing.extract import WrappedResult
    >>> from dispel.data.flags import Flag
    >>> from typing import Union
    >>> def custom_aggregation_func(data) -> Union[WrappedResult, float]:
    ...     result = data.agg('mean')
    ...     if len(data) < 3:
    ...         inv = Flag(
    ...             reason='Not enough data point',
    ...             flag_severity=FlagSeverity.INVALIDATION
    ...         )
    ...         result = WrappedResult(result, inv)
    ...     return result

    During processing, the class `ExtractStep` allows the transformation function to
    output ``WrappedResult`` objects. The extract step will automatically add any flags
    present in the ``WrappedResult`` object to the feature value. The ``WrappedResult``
    class supports basic operations with other scalars or ``WrappedResult`` object:

    >>> from dispel.processing.extract import WrappedResult
    >>> res1 = WrappedResult(feature_value=1)
    >>> res2 = WrappedResult(feature_value=2)
    >>> melted_res = res1 + res2
    >>> melted_res2 = res1 + 1
    """

    def __init__(self, feature_value: WrappedResultType, *args, **kwargs):
        self.feature_value: WrappedResultType = feature_value
        super().__init__(*args, **kwargs)

    def _binary_operator(
        self,
        func: Callable[[WrappedResultType, WrappedResultType], WrappedResultType],
        other: Union[WrappedResultType, "WrappedResult[WrappedResultType]"],
    ) -> "WrappedResult[WrappedResultType]":
        """Perform binary operation on values."""
        # Get feature value for both WrappedResult and float object
        if is_wrapped := isinstance(other, WrappedResult):
            value_other = cast(WrappedResult, other).feature_value
        else:
            value_other = other

        # Create a new WrappedResult object with the combination
        res = WrappedResult(
            func(self.feature_value, value_other)
        )  # type: WrappedResult[WrappedResultType]

        # Inherit flag from current objet
        res.add_flags(self, ignore_duplicates=True)

        # If other is also wrapped, inherit his flag as well
        if is_wrapped:
            res.add_flags(cast(WrappedResult, other), True)

        return res

    def _unary_operation(
        self, func: Callable[[WrappedResultType], WrappedResultType]
    ) -> "WrappedResult[WrappedResultType]":
        res = WrappedResult(func(self.feature_value))
        res.add_flags(self)
        return res

    def __abs__(self):
        return self._unary_operation(operator.abs)

    def __add__(
        self, other: "WrappedResult[WrappedResultType]"
    ) -> "WrappedResult[WrappedResultType]":
        return self._binary_operator(operator.add, other)

    def __sub__(
        self, other: "WrappedResult[WrappedResultType]"
    ) -> "WrappedResult[WrappedResultType]":
        return self._binary_operator(operator.sub, other)

    def __mul__(
        self, other: "WrappedResult[WrappedResultType]"
    ) -> "WrappedResult[WrappedResultType]":
        return self._binary_operator(operator.mul, other)

    def __truediv__(
        self, other: "WrappedResult[WrappedResultType]"
    ) -> "WrappedResult[WrappedResultType]":
        return self._binary_operator(operator.truediv, other)
