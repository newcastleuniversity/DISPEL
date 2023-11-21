"""A module containing models for measures."""
from typing import Any, Callable, ClassVar, Dict, List, Optional, Type, Union, cast

import numpy as np
import pandas as pd

from dispel.data.flags import FlagMixIn
from dispel.data.raw import MissingColumnError
from dispel.data.validators import RangeValidator
from dispel.data.values import AbbreviatedValue as AV
from dispel.data.values import (
    DefinitionId,
    Value,
    ValueDefinition,
    ValueDefinitionPrototype,
    ValueSet,
)


class MeasureValue(FlagMixIn, Value):
    """A measure value."""

    def __repr__(self):
        return (
            f"<MeasureValue ({self.definition}): {self.value} "
            f"({self.flag_count_repr})>"
        )

    @staticmethod
    def _to_string(value):
        return "" if value is None else str(value)

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
            A dictionary summarizing measure value information.
        """
        measure_min, measure_max = None, None
        if isinstance(self.definition.validator, RangeValidator):
            measure_min = self.definition.validator.lower_bound
            measure_max = self.definition.validator.upper_bound

        if stringify:
            value = str(self.value)
            measure_min = self._to_string(measure_min)
            measure_max = self._to_string(measure_max)
        else:
            value = self.value

        return dict(
            measure_id=str(self.id),
            measure_name=self.definition.name,
            measure_value=value,
            measure_unit=self.definition.unit,
            measure_type=self.definition.data_type,
            measure_min=measure_min,
            measure_max=measure_max,
        )


def _join_not_none(separator, values):
    return separator.join(map(str, filter(lambda x: x is not None, values)))


class MeasureId(DefinitionId):
    """The definition of a measure id for a task.

    Parameters
    ----------
    task_name
        The name and abbreviation of the task. Note that if no abbreviation is provided
        the name is used directly in the id.
    measure_name
        The name of the measure and its abbreviation.
    modalities
        The modalities and their abbreviations under which the measure is constituted.
    aggregation
        A method that was used to aggregate a sequence of the underlying measure,
        e.g., for the measure ``mean response time`` it would be ``mean``.

    Notes
    -----
    The abbreviations of values are passed using
    :class:`~dispel.data.values.AbbreviatedValue`. To generate the actual id the `.abbr`
    accessor is used. If one passes only strings, the class actually wraps those into
    ``AbbreviatedValue`` instances.

    Examples
    --------
    >>> from dispel.data.values import AbbreviatedValue as AV
    >>> from dispel.data.measures import MeasureId
    >>> MeasureId(
    ...     task_name=AV('Cognitive Processing Speed', 'CPS'),
    ...     measure_name=AV('reaction time', 'rt'),
    ...     modalities=[AV('digit-to-digit', 'dtd')],
    ...     aggregation='mean'
    ... )
    cps-dtd-rt-mean
    """

    def __init__(
        self,
        task_name: Union[str, AV],
        measure_name: Union[str, AV],
        modalities: Optional[List[Union[str, AV]]] = None,
        aggregation: Optional[Union[str, AV]] = None,
    ):
        self.task_name = AV.wrap(task_name)
        self.measure_name = AV.wrap(measure_name)
        self.modalities = None
        if modalities:
            self.modalities = list(map(AV.wrap, modalities))
        self.aggregation = AV.wrap(aggregation) if aggregation else None

        id_ = _join_not_none(
            "-",
            [
                self.task_name.abbr.lower(),
                "_".join(map(lambda x: x.abbr.lower(), self.modalities))
                if self.modalities
                else None,
                self.measure_name.abbr.lower(),
                self.aggregation.abbr.lower() if self.aggregation else None,
            ],
        )

        super().__init__(id_)

    @classmethod
    def from_str(cls, value: str) -> DefinitionId:
        """See :meth:`dispel.data.values.DefinitionId.from_str`.

        Parameters
        ----------
        value
            The string from which the definition id is to be constructed.

        Raises
        ------
        NotImplementedError
            Always raised. This method is not implemented since there is no unambiguous
            parsing of task ids.
        """
        raise NotImplementedError("Not unambiguous parsing of ids possible")


class MeasureValueDefinition(ValueDefinition):
    """The definition of measures from tasks.

    Parameters
    ----------
    task_name
        The full name of the task and its abbreviation, e.g., ``Cognitive Processing
        Speed test`` and ``CPS`` passed using
        :class:`~dispel.data.values.AbbreviatedValue`.
    measure_name
        The name of the measure, e.g. ``reaction time`` and its abbreviation passed
        using :class:`~dispel.data.values.AbbreviatedValue`. Note that aggregation
        methods are specified in ``aggregation`` and should not be direclty part of the
        measure name.
    unit
        See :class:`~dispel.data.values.ValueDefinition`.
    description
        See :class:`~dispel.data.values.ValueDefinition`.
    data_type
        See :class:`~dispel.data.values.ValueDefinition`.
    validator
        See :class:`~dispel.data.values.ValueDefinition`.
    modalities
        The modalities of the tasks, i.e. if there is more than one variant of the task.
        An example would be the ``digit-to-digit`` and ``symbol-to-digit`` or
        ``predefined key 1``, ``predefined key 2`` and ``random key`` variants of the
        CPS test. Abbreviations of the modalities can be passed using
        :class:`~dispel.data.values.AbbreviatedValue`.
    aggregation
        If the measure is the result of an aggregation, the method that was used to
        aggregate. E.g. for ``mean response time`` it would be ``mean``. Abbreviations
        are passed using :class:`~dispel.data.values.AbbreviatedValue`.
    precision
        See :class:`~dispel.data.values.ValueDefinition`.

    Examples
    --------
    >>> from dispel.data.values import AbbreviatedValue as AV
    >>> from dispel.data.measures import MeasureValueDefinition
    >>> from dispel.data.validators import RangeValidator
    >>> MeasureValueDefinition(
    ...     task_name = AV('Cognitive Processing Speed test', 'CPS'),
    ...     measure_name = AV('response time', 'rt'),
    ...     unit = 's',
    ...     description = 'The mean time to respond to a presented stimulus',
    ...     data_type = 'float64',
    ...     validator = RangeValidator(lower_bound=0),
    ...     modalities = [
    ...         AV('digit-to-digit', 'dtd'),
    ...         AV('predefined key 1', 'key1')
    ...     ],
    ...     aggregation = 'mean'
    ... )
    <MeasureValueDefinition: cps-dtd_key1-rt-mean (CPS digit-to-digit ...>
    """

    def __init__(
        self,
        task_name: Union[str, AV],
        measure_name: Union[str, AV],
        unit: Optional[str] = None,
        description: Optional[str] = None,
        data_type: Optional[str] = None,
        validator: Optional[Callable[[Any], None]] = None,
        modalities: Optional[List[Union[str, AV]]] = None,
        aggregation: Optional[Union[str, AV]] = None,
        precision: Optional[int] = None,
    ):
        self.task_name = AV.wrap(task_name)
        self.measure_name = AV.wrap(measure_name)
        self.modalities = None
        if modalities:
            self.modalities = list(map(AV.wrap, modalities))
        self.aggregation = AV.wrap(aggregation) if aggregation else None

        id_ = MeasureId(
            task_name=self.task_name,
            measure_name=self.measure_name,
            modalities=self.modalities,
            aggregation=aggregation,
        )

        name = _join_not_none(
            " ",
            [
                self.task_name.abbr.upper(),
                " ".join(map(str, self.modalities)) if self.modalities else None,
                self.aggregation if self.aggregation else None,
                self.measure_name,
            ],
        )

        super().__init__(
            id_=id_,
            name=name,
            unit=unit,
            description=description,
            data_type=data_type,
            validator=validator,
            precision=precision,
        )


class MeasureValueDefinitionPrototype(ValueDefinitionPrototype):
    """A task measure value definition prototype.

    This is a convenience method that populates the ``cls`` argument with the
    :class:`~dispel.data.measures.MeasureValueDefinition` class.
    """

    def __init__(self, **kwargs: Any):
        cls = kwargs.pop("cls", MeasureValueDefinition)
        super().__init__(cls=cls, **kwargs)


def row_to_definition(row: pd.Series) -> ValueDefinition:
    """Convert a pandas series to a value definition.

    Parameters
    ----------
    row
        A pandas series containing definition information.

    Returns
    -------
    ValueDefinition
        The corresponding value definition.

    Raises
    ------
    MissingColumnError
        If required fields are missing from the pandas' series.
    """
    expected_columns = {"measure_id", "measure_name", "measure_unit", "measure_type"}
    if not expected_columns.issubset(row.index):
        raise MissingColumnError(expected_columns - set(row.index))

    validator = None
    if {"measure_min", "measure_max"} <= set(row.index):
        validator = RangeValidator(
            lower_bound=cast(Optional[float], np.float_(row.measure_min)),
            upper_bound=cast(Optional[float], np.float_(row.measure_max)),
        )
    return ValueDefinition(
        id_=row.measure_id,
        name=row.measure_name,
        unit=row.measure_unit,
        data_type=row.measure_type,
        validator=validator,
    )


def row_to_value(row: pd.Series) -> MeasureValue:
    """Convert a pandas series to a measure value.

    Parameters
    ----------
    row
        A pandas series containing definition information.

    Returns
    -------
    MeasureValue
        The corresponding measure value.

    Raises
    ------
    MissingColumnError
        If ``measure_value`` field is missing from the pandas' series.
    """
    if "measure_value" not in row.index:
        raise MissingColumnError("measure_value")
    return MeasureValue(
        row_to_definition(row),
        np.array([row["measure_value"]]).astype(row["measure_type"])[0],
    )


class MeasureSet(ValueSet):
    """A collection of measures."""

    VALUE_CLS: ClassVar[Type[Value]] = MeasureValue

    @classmethod
    def from_data_frame(cls, data: pd.DataFrame) -> "MeasureSet":
        """Create a MeasureSet from a data frame.

        Parameters
        ----------
        data
            A data frame containing information about measures

        Returns
        -------
        MeasureSet
            A measure set derived from the provided data frame.
        """
        return cls(data.apply(row_to_value, axis=1).to_list())

    def to_list(self, stringify: bool = False) -> List[Dict[str, Optional[Any]]]:
        """Convert measure set to a list of measure dictionaries.

        Parameters
        ----------
        stringify
            ``True`` if all dictionary values are converted to strings. ``False``
            otherwise.

        Returns
        -------
        List[Dict[str, Optional[Any]]]
            A dictionary summarizing measure value information.
        """
        return [
            cast(self.VALUE_CLS, measure).to_dict(stringify)  # type: ignore
            for measure in self.values()
        ]
