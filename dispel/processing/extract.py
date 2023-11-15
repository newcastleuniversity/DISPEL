"""Extraction functionalities for processing module."""
from __future__ import annotations

import inspect
import math
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
import pandas as pd
from deprecated import deprecated

from dispel.data.core import EntityType, Reading
from dispel.data.features import (
    FeatureId,
    FeatureSet,
    FeatureValue,
    FeatureValueDefinition,
)
from dispel.data.flags import Flag, FlagSeverity, FlagType, WrappedResult
from dispel.data.levels import Level
from dispel.data.values import AbbreviatedValue as AV
from dispel.data.values import (
    DefinitionId,
    DefinitionIdType,
    ValueDefinition,
    ValueDefinitionPrototype,
)
from dispel.processing.core import (
    ErrorHandling,
    ProcessingControlResult,
    ProcessingResult,
    ProcessingStep,
    ProcessResultType,
)
from dispel.processing.data_set import (
    MutateDataSetProcessingStepBase,
    TransformationFunctionGeneratorType,
    WrapResultGeneratorType,
)
from dispel.processing.flags import FlagStepMixin
from dispel.processing.level import LevelFilterType, LevelProcessingResult
from dispel.processing.transform import TransformStepChainMixIn
from dispel.stats.core import iqr, npcv, percentile_95, variation, variation_increase


class FeatureDefinitionMixin:
    """A mixin class for processing steps producing feature values.

    Parameters
    ----------
    definition
        An optional value definition. If no value definition is provided, the
        :data:`definition` class variable will be used. Alternatively, one can overwrite
        :meth:`get_definition` to provide the definition.
    """

    #: The specification of the feature definition
    definition: Optional[Union[ValueDefinition, ValueDefinitionPrototype]] = None

    def __init__(self, *args, **kwargs):
        definition = kwargs.pop("definition", None)
        self.definition = definition or self.definition

        super().__init__(*args, **kwargs)

    def get_definition(self, **kwargs) -> ValueDefinition:
        """Get the feature definition.

        Parameters
        ----------
        kwargs
            Optional parameters that will be passed along to the creation of feature
            definitions from prototypes. See
            :meth:`~dispel.data.values.ValueDefinitionPrototype.create_definition`

        Returns
        -------
        ValueDefinition
            The definition of the value
        """
        assert (
            self.definition is not None
        ), "Definition must be set or get_definition must be overwritten."

        definition = self.definition
        if isinstance(definition, ValueDefinitionPrototype):
            definition = cast(ValueDefinition, definition.create_definition(**kwargs))
        return definition

    def get_value(self, value: Any, **kwargs) -> FeatureValue:
        """Get a feature value based on the definition.

        Parameters
        ----------
        value
            The value
        kwargs
            Optional arguments passed to :meth:`get_definition`.

        Returns
        -------
        FeatureValue
            The ``value`` wrapped with the definition from :meth:`get_definition`.
        """
        return FeatureValue(self.get_definition(**kwargs), value)


class ExtractStep(
    FeatureDefinitionMixin, TransformStepChainMixIn, MutateDataSetProcessingStepBase
):
    r"""A feature extraction processing step.

    This class provides a convenient way to extract a feature from one or more data sets
    by specifying their id, their level_ids or level filter, a transformation function
    and a feature value definition.

    Parameters
    ----------
    data_set_ids
        An optional list of data set ids to be used for the transformation. See
        :class:`~dispel.processing.data_set.DataSetProcessingStepMixin`.
    transform_function
        An optional function to be applied to the data sets. See
        :class:`~dispel.processing.data_set.MutateDataSetProcessingStepBase`.
    definition
        An optional value definition or prototype. See
        :class:`FeatureDefinitionMixin`.
    level_filter
        An optional filter to limit the levels being processed. See
        :class:`~dispel.processing.level.LevelProcessingStep`.
    yield_if_nan
        If ``True``, yield null values as feature values. Otherwise, processing
        will not return a feature value in case of a null result for the extraction.

    Examples
    --------
    Assuming we wanted to compute the maximum value of a raw data set we can create the
    following step

    >>> from dispel.data.values import ValueDefinition
    >>> from dispel.processing.extract import ExtractStep
    >>> step = ExtractStep(
    ...     'data-set-id',
    ...     lambda data: data.max(axis=0),
    ...     ValueDefinition('maximum','Maximum value')
    ... )

    A common approach is to define a processing step for re-use and leveraging the
    ``@transformation`` decorator to specify the transformation function:

    >>> import pandas as pd
    >>> from dispel.data.values import ValueDefinition
    >>> from dispel.processing.extract import ExtractStep
    >>> from dispel.processing.data_set import transformation
    >>> class MyExtractStep(ExtractStep):
    ...     data_set_ids = 'data-set-id'
    ...     definition = ValueDefinition('maximum','Maximum value')
    ...
    ...     @transformation
    ...     def _max(self, data: pd.DataFrame) -> float:
    ...         return data.max(axis=0)

    Often one wants to extract multiple features from one data set. This can be achieved
    by using prototypes and optional named arguments with ``@transformation``:

    >>> import pandas as pd
    >>> from dispel.data.values import ValueDefinitionPrototype
    >>> from dispel.processing.extract import ExtractStep
    >>> from dispel.processing.data_set import transformation
    >>> class MyExtractStep(ExtractStep):
    ...     data_set_ids = 'data-set-id'
    ...     definition = ValueDefinitionPrototype(
    ...         id_='id-{agg_abbr}',
    ...         name='{agg} value'
    ...     )
    ...
    ...     @transformation(agg='Maximum', agg_abbr='max')
    ...     def _max(self, data: pd.DataFrame) -> float:
    ...         return data.max(axis=0)
    ...
    ...     @transformation(agg='Minimum', agg_abbr='min')
    ...     def _min(self, data: pd.DataFrame) -> float:
    ...         return data.min(axis=0)

    """

    yield_if_nan: bool = False

    def __init__(
        self,
        data_set_ids: Optional[Union[str, Iterable[str]]] = None,
        transform_function: Optional[Callable[..., Any]] = None,
        definition: Optional[Union[ValueDefinition, ValueDefinitionPrototype]] = None,
        level_filter: Optional[LevelFilterType] = None,
        yield_if_nan: Optional[bool] = None,
    ):
        super().__init__(
            definition=definition,
            data_set_ids=data_set_ids,
            transform_function=transform_function,
            level_filter=level_filter,
        )
        self.yield_if_nan = yield_if_nan or self.yield_if_nan

    def wrap_result(
        self, res: Any, level: Level, reading: Reading, **kwargs: Any
    ) -> WrapResultGeneratorType:
        """Wrap the result from the processing function into a class.

        Parameters
        ----------
        res
            Any result returned by the extraction step. If res is a
            :class:`~dispel.data.flags.WrappedResult`, the flag contained
            in the object will be automatically added to the
            :class:`~dispel.data.features.FeatureValue`, hence the flagged wrapped
            results will always translate into flagged
            :class:`~dispel.data.features.FeatureValue`.
        level
            The current level
        reading
            The current reading
        kwargs
            Additional kwargs

        Yields
        ------
        LevelProcessingResult
            The processing result
        """
        try:
            if len(res) == 0:
                res = math.nan
                warnings.warn("Extract step returned an iterable!", UserWarning)
        except TypeError:
            pass
        if is_wrapped := isinstance(res, WrappedResult):
            feature_value = res.feature_value
        else:
            feature_value = res

        if not (is_nan := math.isnan(feature_value)) or (is_nan and self.yield_if_nan):
            value = self.get_value(feature_value, **kwargs)
            # If result is wrapped, add the flag to the feature value
            if is_wrapped:
                value.add_flags(res, ignore_duplicates=True)

            yield LevelProcessingResult(
                step=self,
                sources=self.get_raw_data_sets(level),
                result=value,
                level=level,
            )


@deprecated(reason="Use ExtractStep and @transformation decorator")
class ExtractMultipleStep(ExtractStep):
    r"""A feature extraction processing step for multiple features.

    This processing step allows to produce multiple
    :class:`~dispel.data.features.FeatureValue`\ s by providing a list of functions and a
    :class:`~dispel.data.values.ValueDefinitionPrototype` to create the
    :class:`~dispel.data.values.ValueDefinition`\ s from.

    Parameters
    ----------
    data_set_ids
        An optional list of data set ids to be used for the transformation. See
        :class:`~dispel.processing.data_set.DataSetProcessingStepMixin`.
    transform_functions
        An optional list of dictionaries containing at least the processing function
        under the key ``func``, which consumes the specified data sets though
        ``data_set_ids`` as positional arguments and returns a feature value passed to
        :class:`~dispel.data.features.FeatureValue`. Additional keywords will be passed to
        :meth:`~dispel.data.values.ValueDefinitionPrototype.create_definition`. If no
        functions are provided, the :data:`transform_functions` class variable will be
        used.
    definition
        A :class:`~dispel.data.values.ValueDefinitionPrototype` that is used to create the
        :class:`~dispel.data.features.FeatureValueDefinition`\ s for the transformation
        functions provided in ``transform_functions``.
    level_filter
        An optional filter to limit the levels being processed. See
        :class:`~dispel.processing.level.LevelProcessingStep`.
    yield_if_nan
        If ``True``, yield null values as feature values. Otherwise, processing will not
        return a feature value in case of a null result for the extraction.

    Examples
    --------
    To ease the generation of multiple similar features the :class:`ExtractMultipleStep`
    provides a convenient way to do so. Assume you want to create both the mean and
    median of a data set this can be achieved as follows:

    >>> import numpy as np
    >>> from dispel.data.values import ValueDefinitionPrototype
    >>> from dispel.processing.extract import ExtractMultipleStep
    >>> step = ExtractMultipleStep(
    ...     'data-set-id',
    ...     [
    ...         {'func': np.mean, 'method': 'average'},
    ...         {'func': np.median, 'method': 'median'}
    ...     ],
    ...     ValueDefinitionPrototype(
    ...         id_='feature-{method}',
    ...         name='{method} feature',
    ...         unit='s'
    ...     )
    ... )

    This extraction step will result in two feature values, one for the mean and one
    with the median.
    """

    transform_functions: Iterable[Dict[str, Any]]

    def __init__(
        self,
        data_set_ids: Optional[Union[str, Iterable[str]]] = None,
        transform_functions: Optional[Iterable[Dict[str, Any]]] = None,
        definition: Optional[ValueDefinitionPrototype] = None,
        level_filter: Optional[LevelFilterType] = None,
        yield_if_nan: Optional[bool] = None,
    ):
        super().__init__(
            definition=definition,
            data_set_ids=data_set_ids,
            level_filter=level_filter,
            yield_if_nan=yield_if_nan,
        )

        if transform_functions:
            self.transform_functions = transform_functions

    def get_transform_functions(self) -> TransformationFunctionGeneratorType:
        """Get the transform functions applied to the data sets."""
        yield from super().get_transform_functions()

        for function_spec in self.transform_functions:
            spec = function_spec.copy()
            yield spec.pop("func"), spec


AggregationFunctionType = Union[str, Callable[[pd.Series], float]]


def agg_column(
    column: str, method: AggregationFunctionType
) -> Callable[[pd.DataFrame], float]:
    """Create a function to apply an aggregation function on a column.

    Parameters
    ----------
    column
        The column to be aggregated
    method
        A function to apply on the column

    Returns
    -------
    Callable[[pandas.DataFrame], float]
        A function that aggregates one column of a `~pandas.DataFrame`.
    """

    def _function(data: pd.DataFrame) -> float:
        return data[column].agg(method)

    return _function


#: A list of basic used aggregation methods
BASIC_AGGREGATIONS: List[Tuple[str, str]] = [
    ("mean", "mean"),
    ("std", "standard deviation"),
]

#: A list of commonly used aggregation methods
DEFAULT_AGGREGATIONS: List[Tuple[str, str]] = [
    *BASIC_AGGREGATIONS,
    ("median", "median"),
    ("min", "minimum"),
    ("max", "maximum"),
]

#: A list of commonly used aggregation methods plus coefficient of variation
DEFAULT_AGGREGATIONS_CV: List[Tuple[Union[Callable[[Any], float], str], str]] = [
    *DEFAULT_AGGREGATIONS,
    (variation, "coefficient of variation"),
]

#: A list of commonly used aggregation methods plus 95th percentile
DEFAULT_AGGREGATIONS_Q95: List[Tuple[Union[Callable[[Any], float], str], str]] = [
    *DEFAULT_AGGREGATIONS,
    (percentile_95, "95th percentile"),
]

#: A list of commonly used aggregation methods plus inter-quartile range
DEFAULT_AGGREGATIONS_IQR: List[Tuple[Union[Callable[[Any], float], str], str]] = [
    *DEFAULT_AGGREGATIONS,
    (iqr, "iqr"),
]

#: An extended list of commonly used aggregation methods
EXTENDED_AGGREGATIONS: List[Tuple[str, str]] = [
    *DEFAULT_AGGREGATIONS,
    ("skew", "skewness"),
    ("kurtosis", "kurtosis"),
]

DEFAULT_AGGREGATIONS_Q95_CV: List[Tuple[Union[Callable[[Any], float], str], str]] = [
    *DEFAULT_AGGREGATIONS_Q95,
    (variation, "coefficient of variation"),
]

#: A dictionary containing all aggregation methods
AGGREGATION_REGISTRY: Dict[str, Tuple[AggregationFunctionType, str]] = {
    **{agg: (agg, agg_label) for agg, agg_label in EXTENDED_AGGREGATIONS},
    "cv": (variation, "coefficient of variation"),
    "cvi": (variation_increase, "coefficient of variation increase"),
    "q95": (percentile_95, "95th percentile"),
    "npcv": (npcv, "non parametric coefficient of variation"),
}

#: A set of aggregations for which the validator on definitions is ignored
AGGREGATION_CENTER_BASED = {"std", "skew", "kurtosis", "cv", "cvi"}

AggregationsDefinitionType = Sequence[Tuple[AggregationFunctionType, Any]]


class AggregateRawDataSetColumn(ExtractStep):
    r"""An extraction step that allows to summarise a column of a dataset.

    This processing step encapsulates the class :class:`ExtractMultipleStep` and allows
    to produce multiple :class:`~dispel.data.features.FeatureValue`\ s derived on the same
    column of a dataset.

    Parameters
    ----------
    data_set_id
        A single data set id
    column_id
        The column id of the dataset on which the transform function will be applied.
    aggregations
        Either a list of tuples (func, label) where ``func`` consumes the data sets
        specified through ``data_set_id`` at the column ``column_id`` and returns a
        single value passed to :class:`~dispel.data.features.FeatureValue`. The ``label``
        element of the tuple will be passed as ``aggregation`` keyword to
        :meth:`~dispel.data.values.ValueDefinitionPrototype.create_definition`.
        The label can be either a string or an
        :class:`~dispel.data.values.AbbreviatedValue`. If it is a string the label is
        wrapped with the label and aggregation method as abbreviation.

        There are three constants :data:`BASIC_AGGREGATIONS`,
        :data:`DEFAULT_AGGREGATIONS` and :data:`EXTENDED_AGGREGATIONS` that can be used
        for common aggregation scenarios.

        The function is passed to :meth:`pandas.Series.agg` and hence allows to specify
        some default aggregation functions like ``'mean'`` or ``'std'`` without actually
        having to pass a callable.
    definition
        A :class:`~dispel.data.values.ValueDefinitionPrototype` that is used to create the
        :class:`~dispel.data.features.FeatureValueDefinition`\ s for the aggregation
        functions provided in ``aggregations``.
    level_filter
        An optional :class:`~dispel.processing.level.LevelFilter` to determine the levels
        for extraction. If no filter is provided, all levels will be considered. The
        ``level_filter`` also accepts :class:`str`, :class:`~dispel.data.core.LevelId`\ s
        and lists of either and passes them to a
        :class:`~dispel.processing.level.LevelIdFilter` for convenience.

    Examples
    --------
    To ease the generation of multiple similar features for the same column of a
    dataset, the :class:`AggregateRawDataSetColumn` provides a convenient way to do so.
    Assume you want to create both the median and standard deviation of a specific
    column of a data set, this can be achieved as follows:

    >>> from dispel.data.values import ValueDefinitionPrototype
    >>> from dispel.processing.extract import AggregateRawDataSetColumn
    >>> step = AggregateRawDataSetColumn(
    ...     'data-set-id',
    ...     'column-name',
    ...     aggregations=[
    ...         ('median', 'median'),
    ...         ('std', 'standard deviation')
    ...     ],
    ...     definition=ValueDefinitionPrototype(
    ...         id_='feature-{method}',
    ...         name='{method} feature',
    ...         unit='s'
    ...     )
    ... )

    This extraction step will result in two feature values, one for the medianand one
    with the standard deviation of the column ``'column-name'`` of the data set
    identified by ``'data-set-id'``.

    This extraction step will result in three feature values, one for the median, one
    for the standard deviation and one for the variation increase of the column
    ``'column-name'`` of the data set identified by ``'data-set-id'``. The median and
    variation increase features will have associated COI references as provided.
    """

    column_id: str

    aggregations: AggregationsDefinitionType

    def __init__(
        self,
        data_set_id: Optional[str] = None,
        column_id: Optional[str] = None,
        aggregations: Optional[AggregationsDefinitionType] = None,
        definition: Optional[ValueDefinitionPrototype] = None,
        level_filter: Optional[LevelFilterType] = None,
    ):
        super().__init__(
            data_set_ids=data_set_id,
            definition=definition,
            level_filter=level_filter,
            yield_if_nan=False,
        )
        self.column_id = column_id or self.column_id
        self.aggregations = aggregations or self.aggregations

    def get_column_id(self) -> str:
        """Get the id of the column to be aggregated."""
        return self.column_id

    def get_aggregations(
        self,
    ) -> Iterable[Tuple[AggregationFunctionType, Union[str, AV]]]:
        """Get the aggregations to be performed on the specified column."""
        return self.aggregations

    def get_agg_func_and_kwargs(
        self, func: AggregationFunctionType, label: Union[AV, str]
    ) -> Tuple[Callable[[pd.DataFrame], float], Dict[str, Any]]:
        """Get the keyword arguments for the aggregation."""
        agg_func = agg_column(self.get_column_id(), func)

        if isinstance(label, AV):
            aggregation = label
        elif callable(func):
            aggregation = AV(label, func.__name__)
        else:
            aggregation = AV(label, str(func))

        kwargs: Dict[str, Any] = {"aggregation": aggregation}

        return agg_func, kwargs

    def get_transform_functions(self) -> TransformationFunctionGeneratorType:
        """Get the functions to transform the specified column."""
        for func, label in self.get_aggregations():
            yield self.get_agg_func_and_kwargs(func, label)

    def get_definition(self, **kwargs) -> ValueDefinition:
        """Get value definition specific for aggregation."""
        definition = super().get_definition(**kwargs)

        # intercept flag of center-based aggregations
        assert "aggregation" in kwargs, "Aggregation description missing"
        assert isinstance(
            agg := kwargs["aggregation"], AV
        ), "Aggregation keyword must be AbbreviatedValue"

        if definition.validator is not None and agg.abbr in AGGREGATION_CENTER_BASED:
            definition.validator = None

        return definition


class AggregateFeatures(FeatureDefinitionMixin, ProcessingStep):
    """Aggregate multiple features into a single one.

    Parameters
    ----------
    definition
        The feature definition
    feature_ids
        A list of feature ids to be considered for aggregation
    aggregation_method
        The method used to aggregate the feature values, np.mean by default.
    fail_if_missing
        If ``True`` and any of the ``feature_ids`` is not present the
        processing fails.
    yield_if_nan
        If ``True``, yield null values as feature values. Otherwise, processing will not
        return a feature value in case of a null result for the aggregation.
    """

    feature_ids: List[DefinitionIdType] = []

    aggregation_method = None

    fail_if_missing = False

    yield_if_nan = False

    def __init__(
        self,
        definition: Optional[FeatureValueDefinition] = None,
        feature_ids: Optional[List[DefinitionIdType]] = None,
        aggregation_method: Optional[Callable[[List[Any]], Any]] = None,
        fail_if_missing: Optional[bool] = None,
        yield_if_nan: Optional[bool] = None,
    ):
        super().__init__(definition=definition)

        self.feature_ids = feature_ids or self.feature_ids
        self.aggregation_method = aggregation_method or self.aggregation_method
        self.fail_if_missing = fail_if_missing or self.fail_if_missing
        self.yield_if_nan = yield_if_nan or self.yield_if_nan

    @property
    def error_handler(self) -> ErrorHandling:
        """Get error handler corresponding to the ``fail_if_missing`` arg."""
        if self.fail_if_missing:
            return ErrorHandling.RAISE
        return ErrorHandling.IGNORE

    def get_feature_ids(self, **kwargs) -> List[DefinitionIdType]:
        """Get the feature ids considered for aggregation."""
        # pylint: disable=unused-argument
        return self.feature_ids

    @staticmethod
    def get_feature_set(reading: Reading) -> FeatureSet:
        """Get the feature set used for getting feature values for ids."""
        return reading.get_merged_feature_set()

    def get_features(self, reading: Reading, **kwargs) -> List[FeatureValue]:
        """Get the features for aggregation."""
        feature_ids = self.get_feature_ids(**kwargs)
        assert isinstance(
            feature_ids, (list, set, tuple)
        ), "feature_ids needs to be a list, set or tuple of feature ids"

        feature_set = self.get_feature_set(reading)

        if self.fail_if_missing:
            assert set(feature_ids).issubset(feature_set.ids()), (
                "Not all specified features are present",
                self.error_handler,
            )

        return [
            cast(FeatureValue, feature)
            for feature in feature_set.values()
            if feature.id in feature_ids
        ]

    def get_feature_values(self, reading: Reading, **kwargs) -> List[Any]:
        """Get the feature values for aggregation."""
        return list(map(lambda f: f.value, self.get_features(reading, **kwargs)))

    def aggregate(self, values: List[Any]) -> Any:
        """Aggregate feature values."""
        return (self.aggregation_method or np.mean)(values)

    def process_reading(self, reading: Reading, **kwargs) -> ProcessResultType:
        """See :meth:`~dispel.processing.core.ProcessingStep.process`."""
        try:
            res = self.aggregate(self.get_feature_values(reading, **kwargs))
            if not pd.isnull(res) or self.yield_if_nan:
                yield ProcessingResult(
                    step=self,
                    sources=self.get_features(reading, **kwargs),
                    result=self.get_value(res, **kwargs),
                )
        except AssertionError as exception_message:
            yield ProcessingControlResult.from_assertion_error(
                step=self, error=exception_message
            )


class AggregateModalities(AggregateFeatures):
    """Aggregate feature values from different modalities.

    This is a convenience step to address the common pattern of aggregating a feature
    from different modalities of the same root feature. The feature ids are derived from
    the provided ``definition`` and the ``modalities``.
    """

    #: A list of modalities to use for aggregation
    modalities: List[List[Union[str, AV]]] = []

    def get_modalities(self) -> List[List[Union[str, AV]]]:
        """Get a list of modalities to be aggregated."""
        return self.modalities

    def get_feature_ids(self, **kwargs) -> List[DefinitionIdType]:
        """Get feature ids based on modalities and base feature definition."""
        definition = self.get_definition(**kwargs)
        assert isinstance(
            definition, FeatureValueDefinition
        ), "Definition must be a FeatureValueDefinition"

        return [
            FeatureId(
                definition.task_name,
                definition.feature_name,
                modality,
                definition.aggregation,
            )
            for modality in self.get_modalities()
        ]


class FeatureFlagStep(FlagStepMixin, ProcessingStep):
    r"""A class for feature flag.

    Parameters
    ----------
    feature_ids
        The identifier(s) of the feature(s) that are to be flagged.
    task_name
        An optional abbreviated name value of the task used for the flag. See
        :class:`~dispel.processing.flags.FlagStepMixin`.
    flag_name
        An optional abbreviated name value of the considered flag.
        See :class:`~dispel.processing.flags.FlagStepMixin`.
    flag_type
        An optional flag type.
        See :class:`~dispel.data.flags.FlagType`.
    flag_severity
        An optional flag severity.
        See :class:`~dispel.data.flags.FlagSeverity`.
    reason
        An optional string reason of the considered flag.
        See :class:`~dispel.processing.flags.FlagStepMixin`.
    stop_processing
        An optional boolean that specifies whether the flag is stop_processing
        i.e. raises an error or not. See
        :class:`~dispel.processing.flags.FlagStepMixin`.
    flagging_function
        An optional flagging function to be applied to a
        :class:`~dispel.data.features.FeatureValue`'s raw value.
        See :class:`~dispel.processing.flags.FlagStepMixin`.
    target_ids
        An optional id(s) of the target features to be flagged. If the user doesn't
        specify the targets then the targets will automatically be the used features.

    Examples
    --------
    Assuming you want to flag the step count feature value, you can create the
    following flag step:

    >>> from dispel.data.values import AbbreviatedValue as AV
    >>> from dispel.processing.extract import FeatureFlagStep
    >>> step = FeatureFlagStep(
    ...     feature_ids='6mwt-step_count',
    ...     task_name=AV('Six-minute walk test', '6mwt'),
    ...     flag_name=AV('step count threshold', 'sct'),
    ...     flag_type='behavioral',
    ...     flag_severity=FlagSeverity.DEVIATION,
    ...     reason='The step count value exceeds 1000 steps.',
    ...     flagging_function=lambda value: value < 1000,
    ... )

    The flagging function will be called with the feature value corresponding to
    provided feature id. If the function has named parameters matching ``level`` or
    ``reading``, the respective level and reading will be passed to the flag
    function.

    Another common scenario is to define a class that can be reused.

    >>> from dispel.data.flags import FlagType
    >>> from dispel.processing.extract import FeatureFlagStep
    >>> class StepCountThreshold(FeatureFlagStep):
    ...     feature_ids = '6mwt-step_count'
    ...     task_name = AV('Six-minute walk test', '6mwt')
    ...     flag_name = AV('step count threshold', 'sct')
    ...     flag_type = FlagType.BEHAVIORAL
    ...     flag_severity = FlagSeverity.DEVIATION
    ...     reason = 'The step count value exceeds 1000 steps.'
    ...     stop_processing = False
    ...     flagging_function = lambda value: value < 1000

    Another convenient way to provide the flagging function is to use the
    ``@flag`` decorator, one can also use multiple flags for the same class
    as follows:

    >>> from dispel.processing.extract import FeatureFlagStep
    >>> from dispel.processing.flags import flag
    >>> class StepCountThreshold(FeatureFlagStep):
    ...     feature_ids = '6mwt-step_count'
    ...     task_name = AV('Six-minute walk test', '6mwt')
    ...     flag_name = AV('step count threshold', 'sct')
    ...     flag_type = 'behavioral'
    ...     reason = 'The step count value exceeds {threshold} steps.'
    ...     stop_processing = False
    ...
    ...     @flag(threshold=1000, flag_severity=FlagSeverity.INVALIDATION)
    ...     def _threshold_1000(self, value: float) -> bool:
    ...         return value < 1000
    ...
    ...     @flag(threshold=800, flag_severity=FlagSeverity.DEVIATION)
    ...     def _threshold_800(self, value: float) -> bool:
    ...         return value < 800

    Note that the ``@flag`` decorator can take keyword arguments. These kwargs are
    merged with any keyword arguments that come from processing step groups in order to
    format the flag ``reason``.
    """

    feature_ids: DefinitionIdType

    target_ids: Optional[Union[Iterable[str], str]] = None

    def __init__(
        self,
        feature_ids: Optional[DefinitionIdType] = None,
        task_name: Optional[Union[AV, str]] = None,
        flag_name: Optional[Union[AV, str]] = None,
        flag_type: Optional[Union[FlagType, str]] = None,
        flag_severity: Optional[Union[FlagSeverity, str]] = None,
        reason: Optional[Union[AV, str]] = None,
        stop_processing: bool = False,
        flagging_function: Optional[Callable[..., bool]] = None,
        target_ids: Optional[Union[Iterable[str], str]] = None,
    ):
        if feature_ids:
            self.feature_ids = feature_ids

        if target_ids:
            self.target_ids = target_ids

        super().__init__(
            task_name=task_name,
            flag_name=flag_name,
            flag_type=flag_type,
            flag_severity=flag_severity,
            reason=reason,
            stop_processing=stop_processing,
            flagging_function=flagging_function,
        )

    def get_feature_ids(self) -> Iterable[DefinitionIdType]:
        """Get the feature ids to be flagged."""
        if isinstance(self.feature_ids, (str, DefinitionId)):
            return [self.feature_ids]

        return self.feature_ids

    def get_target_ids(self) -> Iterable[DefinitionIdType]:
        """Get the ids of the target data sets to be flagged.

        Returns
        -------
        str
            The identifiers of the target data sets.
        """
        if self.target_ids is None:
            return self.get_feature_ids()
        if isinstance(self.target_ids, (str, DefinitionId)):
            return [self.target_ids]
        return self.target_ids

    def get_features(self, reading: Reading) -> Iterable[Any]:
        """Get the feature value classes used for flag."""
        feature_set = reading.get_merged_feature_set()
        return [feature_set[feature_id] for feature_id in self.get_feature_ids()]

    def get_feature_values(self, reading) -> Iterable[Any]:
        """Get the feature raw values used for flag."""
        return [feature.value for feature in self.get_features(reading)]

    def process_reading(self, reading: Reading, **kwargs) -> ProcessResultType:
        """Process the provided reading.

        Parameters
        ----------
        reading
            The reading to be processed
        kwargs
            Additional arguments passed by :func:`~dispel.processing.process`.

        Yields
        ------
        ProcessResultType
            The results from processing readings.
        """
        for flag in self.flag_feature_values(
            self.get_feature_values(reading), reading, **kwargs
        ):
            yield ProcessingControlResult.from_flag(
                flag=flag,
                step=self,
                targets=self.get_flag_targets(reading),
            )

    def get_flag_targets(
        self, reading: Reading, level: Optional[Level] = None, **kwargs
    ) -> Iterable[EntityType]:
        """Get flag targets for features."""
        feature_set = reading.get_merged_feature_set()
        return [
            cast(FeatureValue, feature_set[feature_id])
            for feature_id in self.get_target_ids()
        ]

    def flag_feature_values(
        self, feature_values: Iterable[Any], reading: Reading, **kwargs
    ) -> Generator[Flag, None, None]:
        """Flag the provided feature value."""
        for func, func_kwargs in self.get_flagging_functions():
            new_kwargs = kwargs.copy()
            if "reading" in inspect.getfullargspec(func).args:
                new_kwargs["reading"] = reading

            if not func(*feature_values, **new_kwargs):
                (merged_kwargs := kwargs.copy()).update(func_kwargs)
                yield self.get_flag(**merged_kwargs)
