"""A module to handle values and their definitions."""
import inspect
from enum import Enum
from functools import total_ordering
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    ItemsView,
    Iterable,
    KeysView,
    List,
    Optional,
    Set,
    Type,
    Union,
    ValuesView,
)

from dispel.data.validators import ValidationException


@total_ordering
class AbbreviatedValue:
    """An abbreviated value.

    Examples
    --------
    This class allows to consistently handle abbreviated terms. Assuming you have a name
    of an assessment, e.g. `Cognitive Processing Speed` test and the respective
    abbreviation would be `CPS`, then you can create an abbreviated value like this:

    >>> from dispel.data.values import AbbreviatedValue as AV
    >>> value = AV('Cognitive Processing Speed test', 'CPS')
    >>> value
    Cognitive Processing Speed test (CPS)

    While this seems like a lot of overhead, it comes in handy when describing value
    definitions or higher-level abstractions, such as feature definitions.

    Parameters
    ----------
    value
        The full description of the value
    abbr
        The abbreviated form of the value

    Attributes
    ----------
    value
        The full description of the value
    """

    def __init__(self, value: str, abbr: Optional[str] = None):
        self.value = value
        self._abbr = abbr

    @property
    def abbr(self):
        """Get the abbreviated form of the value."""
        return self._abbr or self.value

    def __str__(self):
        return self.value

    def __repr__(self):
        if self._abbr:
            return f"{self.value} ({self._abbr})"
        return self.value

    def __hash__(self):
        return hash((self.value, self._abbr))

    def __eq__(self, other):
        if isinstance(other, str):
            return self._abbr is None and self.value == other
        if isinstance(other, AbbreviatedValue):
            return self.value == other.value and self.abbr == other.abbr
        return False

    def __lt__(self, other):
        if not isinstance(other, AbbreviatedValue):
            raise ValueError(f"Unsupported type in comparison: {type(other)}")
        if self.value == other.value:
            return self.abbr < other.abbr
        return self.value < other.value

    def format(self, *args, **kwargs):
        """Format an abbreviated value."""
        return AbbreviatedValue(
            self.value.format(*args, **kwargs),
            self._abbr.format(*args, **kwargs) if self._abbr else None,
        )

    @classmethod
    def wrap(cls, value):
        """Wrap a value into an abbreviated value.

        This is a small helper class to conveniently wrap values into an abbreviated
        value, if they are not already one.

        Parameters
        ----------
        value
            The value to be wrapped

        Returns
        -------
        AbbreviatedValue
            The passed ``value`` if it is an instance of :class:`AbbreviatedValue`. If a
            string is passed, then the string is passed as ``value`` argument to the
            constructor.

        Raises
        ------
        ValueError
            If the passed value is neither a string nor an instance of
            :class:`AbbreviatedValue`.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(value)

        raise ValueError(f"Can only wrap string values. Got: {type(value)}")


class DefinitionId:
    """The definition of a feature id.

    This class provides the basic functionality around ids used to reference columns and
    definitions. Other structured ids inherit from this class.

    Parameters
    ----------
    id_
        The identifier of the definition.
    """

    def __init__(self, id_: str):
        self._id = id_  # pylint: disable=C0103

    @property
    def id(self) -> str:
        """Get the identifier."""
        return self._id

    def __str__(self):
        return self.id

    __repr__ = __str__

    def __eq__(self, other):
        if isinstance(other, str):
            return self.id == other
        if isinstance(other, DefinitionId):
            return self.id == other.id

        return False

    def __hash__(self):
        return hash(self.id)

    @classmethod
    def from_str(cls, value: str) -> "DefinitionId":
        """Create a class instance from string."""
        return cls(value)


DefinitionIdType = Union[DefinitionId, str]


class ValueDefinition:
    """The definition of a value.

    Parameters
    ----------
    id_
        The identifier of the value definition
    name
        The human-readable name of the values
    unit
        The unit of the value
    description
        A more elaborate description of the values and how they were produced
    data_type
        The numpy data type of the value in question
    validator
        A function that ensures values comply with the definition. The module
        :mod:`~dispel.data.validators` contains validators for common scenarios that can be
        used here.
    precision
        The number of significance for the values expected under definition. If set, the
        value will be rounded to the set number of digits.
    """

    def __init__(
        self,
        id_: DefinitionIdType,
        name: str,
        unit: Optional[str] = None,
        description: Optional[str] = None,
        data_type: Optional[str] = None,
        validator: Optional[Callable[[Any], None]] = None,
        precision: Optional[int] = None,
    ):
        if isinstance(id_, str):
            id_ = DefinitionId.from_str(id_)

        self.id = id_  # pylint: disable=C0103
        self.name = name
        self.unit = unit
        self.description = description
        self.data_type = data_type

        # Verify that the validator is Callable
        if validator and not callable(validator):
            raise TypeError(f"The {id_} feature validator is not Callable.")

        self.validator = validator
        self.precision = precision

    def __repr__(self):
        unit_extra = f", {self.unit}" if self.unit else ""
        return f"<{self.__class__.__name__}: {self.id} " f"({self.name}{unit_extra})>"

    def __hash__(self):
        # TODO: make properties read-only
        return hash(
            (
                self.id,
                self.name,
                self.unit,
                self.description,
                self.validator,
                self.data_type,
            )
        )

    def __eq__(self, other):
        if isinstance(other, ValueDefinition):
            return hash(self) == hash(other)
        return False

    @classmethod
    def _get_parameters(cls) -> Set[str]:
        params = set(inspect.signature(cls.__init__).parameters.keys())
        params.remove("self")
        return params

    def _to_dict(self) -> Dict[str, Any]:
        """Turn instance into dict with values from constructor."""

        def _getattr(name):
            if name == "id_":
                return self.id
            return getattr(self, name)

        return {name: _getattr(name) for name in self._get_parameters()}

    def derive(self, **kwargs) -> "ValueDefinition":
        """Derive a value definition with updated properties.

        Parameters
        ----------
        kwargs
            Keyword arguments to be set/updated in the derived definition.

        Returns
        -------
        ValueDefinition
            A new definition with updated parameters.

        Raises
        ------
        ValueError
            If one of the provided arguments is not a parameter of the constructor.
        """
        diff = set(kwargs.keys()).difference(self._get_parameters())
        if diff:
            raise ValueError(
                f"The following parameters are unknown to the constructor: "
                f'{", ".join(sorted(diff))}'
            )

        new_kwargs = self._to_dict()
        new_kwargs.update(kwargs)
        return self.__class__(**new_kwargs)


class ValueDefinitionPrototype:
    """The prototype of a :class:`ValueDefinition`.

    Feature processing often leads to various related features. To ease the creation of
    such, the :class:`ValueDefinitionPrototype` allows to specify prototypic features
    that can be used to derive actual definitions.

    Parameters
    ----------
    cls
        The class to be used when creating concrete instances with
        :meth:`create_definition`. By default, the class used is
        :class:ValueDefinition`.
    kwargs
        All named parameters passed to the constructor will be passed to the feature
        definition class constructor. The parameter ``cls`` is reserved to pass a
        different feature definition class. Placeholders to be filled upon creation are
        specified with curly brackets, i.e., ``a {placeholder} value`` is populated when
        calling ``prototype.create_definition(placeholder='special')``.

    Examples
    --------
    Assuming we want to create a feature for different time windows one can create the
    following prototype:

    >>> from dispel.data.values import ValueDefinitionPrototype
    >>> prototype = ValueDefinitionPrototype(
    ...     id_='feature-{lower}-{upper}',
    ...     name='feature from {lower} to {upper}',
    ...     unit='s'
    ... )
    >>> prototype.create_definition(lower=5, upper=6)
    <ValueDefinition: feature-5-6 (feature from 5 to 6, s)>
    >>> prototype.create_definition(lower=1, upper=5)
    <ValueDefinition: feature-1-5 (feature from 1 to 5, s)>
    """

    def __init__(self, **kwargs):
        self._cls = kwargs.pop("cls", ValueDefinition)
        self._kwargs = kwargs

    def create_definition(self, **values: Any) -> ValueDefinition:
        """
        Create a definition from this prototype.

        Parameters
        ----------
        values
            The arguments and placeholders to be populated. All named arguments will be
            used to both provide additional named arguments to the feature definition
            class specified with ``cls`` during construction upon creation (the class is
            inspected for named parameters) and placeholders provided during
            construction of the prototype.

        Returns
        -------
        ValueDefinition
            The value definition created from the value definition prototype.

        Examples
        --------
        An example is given above to populate placeholders. This is also possible with
        arguments required by the definition class:

        >>> from dispel.data.values import ValueDefinitionPrototype
        >>> prototype = ValueDefinitionPrototype(unit='s')
        >>> prototype.create_definition(id_='foo', name='bar')
        <ValueDefinition: foo (bar, s)>
        >>> prototype.create_definition(id_='baz', name='bam')
        <ValueDefinition: baz (bam, s)>

        Raises
        ------
        ValueError
            If a placeholder is missing from kwargs.
        """

        def _can_format(value):
            return isinstance(value, (str, AbbreviatedValue))

        try:
            kwargs = {
                k: v.format(**values) if _can_format(v) else v
                for k, v in self._kwargs.items()
            }
        except KeyError as error:
            raise ValueError(f"Missing placeholder: {error}") from error

        # inspect class for additional arguments to be passed
        signature = inspect.signature(self._cls.__init__)
        for param in signature.parameters:
            if param != "self" and param not in kwargs and param in values:
                kwargs[param] = values[param]

        return self._cls(**kwargs)

    def create_definitions(
        self, items: Iterable[Dict[str, Any]]
    ) -> List[ValueDefinition]:
        """Create multiple definitions.

        This method provides a convenient way to specify multiple definitions at the
        same time for an iterable of dictionaries that are passed to
        :meth:`create_definition`.

        Parameters
        ----------
        items
            An iterable of dictionaries passed to :meth:`create_definition`.

        Returns
        -------
        List[ValueDefinition]
            A list of the created value definitions.
        """
        return [self.create_definition(**values) for values in items]

    def derive(self, **kwargs) -> "ValueDefinitionPrototype":
        """Derive a prototype with updated properties.

        Parameters
        ----------
        kwargs
            Keyword arguments to be set/updated in the derived prototype.

        Returns
        -------
        ValueDefinitionPrototype
            A new prototype with updated parameters.
        """
        assert "cls" not in kwargs, "Class is set by derived class"

        new_kwargs = self._kwargs.copy()
        new_kwargs.update(kwargs)

        return self.__class__(cls=self._cls, **new_kwargs)


class Value:
    """A value with definition and actual value.

    Parameters
    ----------
    definition
        The definition of the value.
    value
        The actual value. If `definition.precision` is set, then the value will be
        rounded to the number of significant digits. The pre-rounded value is stored in
        `raw_value`.
    """

    def __init__(self, definition: ValueDefinition, value: Any):
        if not isinstance(definition, ValueDefinition):
            raise ValueError("Definition must be an instance of ValueDefinition")

        self.definition = definition

        # store original raw value before precision rounding
        self.raw_value = value

        if definition.precision is not None:
            value = round(value, ndigits=definition.precision)

        self.value = value

        # validate value if validator is present
        if self.definition.validator:
            try:
                self.definition.validator(self.value)
            except ValidationException as exc:
                raise ValueError(
                    f"Provided value is not valid for {self.definition}: {exc}"
                ) from exc

    @property
    def id(self) -> DefinitionId:
        """Get the identifier from the definition of the value."""
        return self.definition.id

    def __repr__(self):
        return f"<{self.__class__.__name__} ({self.definition}): {self.value}>"

    def __hash__(self):
        return hash((self.definition, self.value))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return hash(self) == hash(other)

        return False


class ValueSet:
    r"""A collection of multiple values.

    Parameters
    ----------
    values
        The values of the value set. This can be a list of :class:`Value`\ s or a list
        of any value.
    definitions
        An optional list of definitions describing the passed values through the
        parameter ``values``.

    Notes
    -----
    The constructor passes both ``values`` and ``definitions`` to
    :meth:`ValueSet.set_values`. For details on how to specify values of the
    :class:`ValueSet` please have a look there.
    """

    VALUE_CLS: ClassVar[Type[Value]] = Value

    def __init__(
        self,
        values: Optional[List[Any]] = None,
        definitions: Optional[List[ValueDefinition]] = None,
    ):
        self._values: Dict[DefinitionId, Value] = {}
        if values:
            self.set_values(values, definitions)

    def set(
        self,
        value: Any,
        definition: Optional[ValueDefinition] = None,
        overwrite: bool = False,
    ):
        """Set a value in the value set.

        Parameters
        ----------
        value
            The value to be set. If the value is not an instance of :class:`Value` one
            needs to also provide a ``definition``.
        definition
            An optional definition of the passed value should the value not be an
            instance of :class:`Value`.
        overwrite
            By default values in the :class:`ValueSet` are not overwritten. If you want
            to update an already set value you will need to set ``overwrite = True``.

        Raises
        ------
        ValueError
            If ``value`` is not a :class:`Value` and no definition is passed.
        ValueError
            If ``value``'s id is already present in the Value Set, and the ``overwrite``
            argument is set to ``False``.
        """
        if not isinstance(value, self.VALUE_CLS):
            if definition is None or not isinstance(definition, ValueDefinition):
                raise ValueError(
                    "Value must be either a Value or a definition needs to be passed"
                )
            value = self.VALUE_CLS(definition, value)
        if not overwrite and value.id in self._values:
            raise ValueError(
                f"Value with id already present: {value.id}. Set overwrite = True to "
                f"overwrite values."
            )
        self._values[value.id] = value

    def set_values(
        self,
        values: List[Any],
        definitions: Optional[List[ValueDefinition]] = None,
        overwrite: bool = True,
    ):
        """Set multiple values in the value set.

        Parameters
        ----------
        values
            The values to be set. If the values are not an instance of :class:`Value`
            the optional parameter ``definitions`` needs to be provided with a list of
            :class:`ValueDefinition` describing each value in ``values``.
        definitions
            An optional list of definitions for values passed via ``values``. Both
            ``values`` and ``definitions`` need to be of equal length.
        overwrite
            The overwrite-behavior. See :meth:`ValueSet.set` for details.

        Raises
        ------
        ValueError
            If ``values`` and ``definitions`` are not of equal length.
        """
        if definitions:
            if len(values) != len(definitions):
                raise ValueError("Values and definitions need to be of equal length")

            values = [self.VALUE_CLS(d, v) for v, d in zip(values, definitions)]

        for value in values:
            self.set(value, overwrite=overwrite)

    def has_value(self, id_: Union[DefinitionIdType, ValueDefinition]) -> bool:
        """Test if the set has a specific value.

        Parameters
        ----------
        id_
            The id or definition for which to lookup if a value is present

        Returns
        -------
        bool
            ``True`` if the value set contains a value for the provided ``id_``.
            Otherwise, ``False``.

        Raises
        ------
        TypeError
            If the id is neither a ``str``, :class:`DefinitionId` nor a
            :class:`ValueDefinition`.
        """
        if isinstance(id_, str):
            return DefinitionId.from_str(id_) in self._values
        if isinstance(id_, DefinitionId):
            return id_ in self._values
        if isinstance(id_, ValueDefinition):
            return id_.id in self._values

        raise TypeError(
            "Id must be one of str, DefinitionId, ValueDefinition. " f"Got {type(id_)}"
        )

    def __contains__(self, item: Union[DefinitionIdType, ValueDefinition]) -> bool:
        """Test if the set has a specific value.

        This is a convenience method for :meth:`ValueSet.has_value`.

        Parameters
        ----------
        item
            The item whose existence in the value set is to be tested.

        Returns
        -------
            ``True`` if the item exists in the value set. ``False`` otherwise.
        """
        return self.has_value(item)

    def get(self, id_: Union[DefinitionIdType, ValueDefinition]) -> Value:
        """Get a value for an id.

        Parameters
        ----------
        id_
            The id or definition for which to obtain the value.

        Returns
        -------
        Value
            The respective :class:`Value` matching the provided ``id_``.

        Raises
        ------
        KeyError
            If the provided ``id_`` is not present in the set.
        """
        if isinstance(id_, str):
            return self.get(DefinitionId.from_str(id_))
        if isinstance(id_, ValueDefinition):
            return self.get(id_.id)

        if not self.has_value(id_):
            raise KeyError(
                f"No value with id {id_} in set: " f"{list(self._values.keys())}"
            )

        return self._values[id_]

    def __getitem__(self, key: Union[DefinitionIdType, ValueDefinition]) -> Value:
        """Get a value for an id.

        This is a convenience wrapper around :meth:`ValueSet.get`.

        Parameters
        ----------
        key
             The id or definition for which to retrieve the value.

        Returns
        -------
        Value
            The value matching the passed id or definition.
        """
        return self.get(key)

    def get_raw_value(self, id_: Union[DefinitionIdType, ValueDefinition]) -> Any:
        """Get the raw value for an id.

        This is a convenience method to not have to call ``value_set.get(id).value``.

        Parameters
        ----------
        id_
            The id or definition for which to retrieve the raw value.

        Returns
        -------
        Any
            The raw value of the :class:`Value` matching the passed id or definition.
        """
        return self.get(id_).value

    def get_definition(
        self, id_: Union[DefinitionIdType, ValueDefinition]
    ) -> ValueDefinition:
        """Get the definition of a value by its id.

        Parameters
        ----------
        id_
            The id for which to obtain the definition

        Returns
        -------
        ValueDefinition
            The definition belonging to the passed id.
        """
        return self.get(id_).definition

    def values(self) -> ValuesView[Value]:
        """Get all values of the set.

        Returns
        -------
        Iterable[Value]
            An iterable of all values within the set.
        """
        return self._values.values()

    def ids(self) -> KeysView[DefinitionId]:
        """Get all ids of the set.

        Returns
        -------
        Iterable[DefinitionId]
            An iterable of all definition ids from all values of the set.
        """
        return self._values.keys()

    def definitions(self) -> Iterable[ValueDefinition]:
        """Get all definitions of the set.

        Returns
        -------
        Iterable[ValueDefinition]
            An iterable of all value definitions from all values of the set.
        """
        return (v.definition for v in self.values())

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self):
        return iter(self._values)

    @property
    def empty(self) -> bool:
        """Get whether the value set is empty."""
        return len(self) == 0

    @staticmethod
    def _assert_add_type(other):
        if not isinstance(other, ValueSet):
            raise TypeError("Can only add values from ValueSet")

    def items(self) -> ItemsView[DefinitionId, Value]:
        """Get an items view of all values."""
        return self._values.items()

    def _combine(self, other, overwrite):
        self._assert_add_type(other)

        res = self.__class__()
        res.set_values(list(self.values()))
        res.set_values(list(other.values()), overwrite=overwrite)

        return res

    def __add__(self, other):
        return self._combine(other, False)

    def __iadd__(self, other):
        self._assert_add_type(other)

        self.set_values(list(other.values()))
        return self

    def __or__(self, other: "ValueSet") -> "ValueSet":
        return self._combine(other, True)

    def __ior__(self, other: "ValueSet") -> "ValueSet":
        self._assert_add_type(other)

        self.set_values(list(other.values()), overwrite=True)
        return self

    def __eq__(self, other):
        if isinstance(other, ValueSet):
            return set(self.values()) == set(other.values())

        raise TypeError("Can only compare ValueSets")


@total_ordering
class AVEnum(Enum):
    """A base class for abbreviated value enumerations.

    When extracting features from tasks they are often done for specific modalities.
    This base class allows to do this in a convenient fashion to address modalities both
    from a processing and representation form. The enumeration is ordered.

    Examples
    --------
    Assuming you have a task that has two modalities, e.g. the *Cognitive Processing
    Speed* test has two forms: *symbol-to-digit* and *digit-to-digit*. The respective
    modalities class would look like:

    .. doctest:: enum

        >>> from dispel.data.values import AVEnum
        >>> class CPSTypeModality(AVEnum):
        ...     SYMBOL_TO_DIGIT = ('symbol-to-digit', 'std')
        ...     DIGIT_TO_DIGIT = ('digit-to-digit', 'dtd')
        ...
        >>> CPSTypeModality.SYMBOL_TO_DIGIT
        <CPSTypeModality.SYMBOL_TO_DIGIT: symbol-to-digit (std) [1]>

    The constants can be used directly in :class:`~pandas.Series` as well as can be
    converted to an integer representation:

    .. doctest:: enum

        >>> int(CPSTypeModality.SYMBOL_TO_DIGIT)
        1

    In order to conventiently pass the constants to modalities of
    :class:`~dispel.data.features.FeatureValueDefinition` and
    :class:`~dispel.data.features.FeatureValueDefinitionPrototype` a property is exposed
    that contains the :class:`~dispel.data.values.AbbreviatedValue`:

    .. doctest:: enum

        >>> CPSTypeModality.SYMBOL_TO_DIGIT.av
        symbol-to-digit (std)

    as well as for convenience the abbreviated value too:

    .. doctest:: enum

        >>> CPSTypeModality.SYMBOL_TO_DIGIT.abbr
        'std'

    Since the enumeration is odered, one can also perform comparisons between them:

    .. doctest:: enum

        >>> CPSTypeModality.DIGIT_TO_DIGIT < CPSTypeModality.SYMBOL_TO_DIGIT
        False

    The member can also be retrieved from the abbreviation:

    .. doctest:: enum

        >>> CPSTypeModality.from_abbr('std')
        <CPSTypeModality.SYMBOL_TO_DIGIT: symbol-to-digit (std) [1]>

    As well as from the variable name (case-insensitive):

    .. doctest:: enum

        >>> CPSTypeModality.from_variable('symbol_to_digit')
        <CPSTypeModality.SYMBOL_TO_DIGIT: symbol-to-digit (std) [1]>

    """

    def __new__(cls, *_args, **_kwargs):  # noqa: D102
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, value, abbr=None):
        self.av = (
            value
            if isinstance(value, AbbreviatedValue)
            else AbbreviatedValue(value, abbr)
        )

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}.{self.name}: {repr(self.av)} "
            f"[{self.value}]>"
        )

    def __str__(self):
        return str(self.av)

    def __int__(self):
        return self.value

    def __lt__(self, other):
        return self.value < other.value  # pylint: disable=W0143

    @property
    def abbr(self):
        """Get the abbreviated value."""
        return self.av.abbr

    @property
    def variable(self):
        """Get the modality variable name."""
        return str(self.name).lower()

    @classmethod
    def from_abbr(cls, value: str):
        """Get the corresponding member from the abbreviated value."""
        for member in cls:
            if member.abbr == value:
                return member

        raise KeyError(f"Unknown abbreviation: {value}")

    @classmethod
    def from_variable(cls, value: str):
        """Get the corresponding member from the variable name."""
        return getattr(cls, value.upper())
