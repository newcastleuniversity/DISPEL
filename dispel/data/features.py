"""A module containing models for features."""
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


class FeatureValue(FlagMixIn, Value):
    """A feature value."""

    def __repr__(self):
        return (
            f"<FeatureValue ({self.definition}): {self.value} "
            f"({self.flag_count_repr})>"
        )

    @staticmethod
    def _to_string(value):
        return "" if value is None else str(value)

    def to_dict(self, stringify: bool = False) -> Dict[str, Optional[Any]]:
        """Get a dictionary representation of feature information.

        Parameters
        ----------
        stringify
            ``True`` if all dictionary values are converted to strings. ``False``
            otherwise.

        Returns
        -------
        Dict[str, Optional[Any]]
            A dictionary summarizing feature value information.
        """
        feature_min, feature_max = None, None
        if isinstance(self.definition.validator, RangeValidator):
            feature_min = self.definition.validator.lower_bound
            feature_max = self.definition.validator.upper_bound

        if stringify:
            value = str(self.value)
            feature_min = self._to_string(feature_min)
            feature_max = self._to_string(feature_max)
        else:
            value = self.value

        return dict(
            feature_id=str(self.id),
            feature_name=self.definition.name,
            feature_value=value,
            feature_unit=self.definition.unit,
            feature_type=self.definition.data_type,
            feature_min=feature_min,
            feature_max=feature_max,
        )


def _join_not_none(separator, values):
    return separator.join(map(str, filter(lambda x: x is not None, values)))


class FeatureId(DefinitionId):
    """The definition of a feature id for a task.

    Parameters
    ----------
    task_name
        The name and abbreviation of the task. Note that if no abbreviation is provided
        the name is used directly in the id.
    feature_name
        The name of the feature and its abbreviation.
    modalities
        The modalities and their abbreviations under which the feature is constituted.
    aggregation
        A method that was used to aggregate a sequence of the underlying feature,
        e.g., for the feature ``mean response time`` it would be ``mean``.

    Notes
    -----
    The abbreviations of values are passed using
    :class:`~dispel.data.values.AbbreviatedValue`. To generate the actual id the `.abbr`
    accessor is used. If one passes only strings, the class actually wraps those into
    ``AbbreviatedValue`` instances.

    Examples
    --------
    >>> from dispel.data.values import AbbreviatedValue as AV
    >>> from dispel.data.features import FeatureId
    >>> FeatureId(
    ...     task_name=AV('Cognitive Processing Speed', 'CPS'),
    ...     feature_name=AV('reaction time', 'rt'),
    ...     modalities=[AV('digit-to-digit', 'dtd')],
    ...     aggregation='mean'
    ... )
    cps-dtd-rt-mean
    """

    def __init__(
        self,
        task_name: Union[str, AV],
        feature_name: Union[str, AV],
        modalities: Optional[List[Union[str, AV]]] = None,
        aggregation: Optional[Union[str, AV]] = None,
    ):
        self.task_name = AV.wrap(task_name)
        self.feature_name = AV.wrap(feature_name)
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
                self.feature_name.abbr.lower(),
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


class FeatureValueDefinition(ValueDefinition):
    """The definition of features from tasks.

    Parameters
    ----------
    task_name
        The full name of the task and its abbreviation, e.g., ``Cognitive Processing
        Speed test`` and ``CPS`` passed using
        :class:`~dispel.data.values.AbbreviatedValue`.
    feature_name
        The name of the feature, e.g. ``reaction time`` and its abbreviation passed
        using :class:`~dispel.data.values.AbbreviatedValue`. Note that aggregation
        methods are specified in ``aggregation`` and should not be direclty part of the
        feature name.
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
        If the feature is the result of an aggregation, the method that was used to
        aggregate. E.g. for ``mean response time`` it would be ``mean``. Abbreviations
        are passed using :class:`~dispel.data.values.AbbreviatedValue`.
    precision
        See :class:`~dispel.data.values.ValueDefinition`.

    Examples
    --------
    >>> from dispel.data.values import AbbreviatedValue as AV
    >>> from dispel.data.features import FeatureValueDefinition
    >>> from dispel.data.validators import RangeValidator
    >>> FeatureValueDefinition(
    ...     task_name = AV('Cognitive Processing Speed test', 'CPS'),
    ...     feature_name = AV('response time', 'rt'),
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
    <FeatureValueDefinition: cps-dtd_key1-rt-mean (CPS digit-to-digit ...>
    """

    def __init__(
        self,
        task_name: Union[str, AV],
        feature_name: Union[str, AV],
        unit: Optional[str] = None,
        description: Optional[str] = None,
        data_type: Optional[str] = None,
        validator: Optional[Callable[[Any], None]] = None,
        modalities: Optional[List[Union[str, AV]]] = None,
        aggregation: Optional[Union[str, AV]] = None,
        precision: Optional[int] = None,
    ):
        self.task_name = AV.wrap(task_name)
        self.feature_name = AV.wrap(feature_name)
        self.modalities = None
        if modalities:
            self.modalities = list(map(AV.wrap, modalities))
        self.aggregation = AV.wrap(aggregation) if aggregation else None

        id_ = FeatureId(
            task_name=self.task_name,
            feature_name=self.feature_name,
            modalities=self.modalities,
            aggregation=aggregation,
        )

        name = _join_not_none(
            " ",
            [
                self.task_name.abbr.upper(),
                " ".join(map(str, self.modalities)) if self.modalities else None,
                self.aggregation if self.aggregation else None,
                self.feature_name,
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


class FeatureValueDefinitionPrototype(ValueDefinitionPrototype):
    """A task feature value definition prototype.

    This is a convenience method that populates the ``cls`` argument with the
    :class:`~dispel.data.features.FeatureValueDefinition` class.
    """

    def __init__(self, **kwargs: Any):
        cls = kwargs.pop("cls", FeatureValueDefinition)
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
    expected_columns = {"feature_id", "feature_name", "feature_unit", "feature_type"}
    if not expected_columns.issubset(row.index):
        raise MissingColumnError(expected_columns - set(row.index))

    validator = None
    if {"feature_min", "feature_max"} <= set(row.index):
        validator = RangeValidator(
            lower_bound=cast(Optional[float], np.float_(row.feature_min)),
            upper_bound=cast(Optional[float], np.float_(row.feature_max)),
        )
    return ValueDefinition(
        id_=row.feature_id,
        name=row.feature_name,
        unit=row.feature_unit,
        data_type=row.feature_type,
        validator=validator,
    )


def row_to_value(row: pd.Series) -> FeatureValue:
    """Convert a pandas series to a feature value.

    Parameters
    ----------
    row
        A pandas series containing definition information.

    Returns
    -------
    FeatureValue
        The corresponding feature value.

    Raises
    ------
    MissingColumnError
        If ``feature_value`` field is missing from the pandas' series.
    """
    if "feature_value" not in row.index:
        raise MissingColumnError("feature_value")
    return FeatureValue(
        row_to_definition(row),
        np.array([row["feature_value"]]).astype(row["feature_type"])[0],
    )


class FeatureSet(ValueSet):
    """A collection of features."""

    VALUE_CLS: ClassVar[Type[Value]] = FeatureValue

    @classmethod
    def from_data_frame(cls, data: pd.DataFrame) -> "FeatureSet":
        """Create a FeatureSet from a data frame.

        Parameters
        ----------
        data
            A data frame containing information about features

        Returns
        -------
        FeatureSet
            A feature set derived from the provided data frame.
        """
        return cls(data.apply(row_to_value, axis=1).to_list())

    def to_list(self, stringify: bool = False) -> List[Dict[str, Optional[Any]]]:
        """Convert feature set to a list of feature dictionaries.

        Parameters
        ----------
        stringify
            ``True`` if all dictionary values are converted to strings. ``False``
            otherwise.

        Returns
        -------
        List[Dict[str, Optional[Any]]]
            A dictionary summarizing feature value information.
        """
        return [
            cast(self.VALUE_CLS, feature).to_dict(stringify)  # type: ignore
            for feature in self.values()
        ]
