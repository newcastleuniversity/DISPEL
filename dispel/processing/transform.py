"""Transformation functionalities for processing module."""
from abc import ABCMeta
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Literal,
    Optional,
    Union,
    cast,
)

import numpy as np
import pandas as pd

from dispel.data.core import Reading
from dispel.data.levels import (
    Context,
    Level,
    LevelId,
    LevelIdType,
    RawDataSetAlreadyExists,
)
from dispel.data.raw import (
    DEFAULT_COLUMNS,
    RawDataSet,
    RawDataSetDefinition,
    RawDataSetSource,
    RawDataValueDefinition,
)
from dispel.data.values import ValueDefinition
from dispel.processing.core import ProcessingResult, ProcessingStep, ProcessResultType
from dispel.processing.data_set import (
    DataSetProcessingStepProtocol,
    MutateDataSetProcessingStepBase,
    RawDataSetProcessingResult,
    StorageError,
    WrapResultGeneratorType,
)
from dispel.processing.level import LevelFilterProcessingStepMixin, LevelFilterType


class TransformStepChainMixIn(DataSetProcessingStepProtocol, metaclass=ABCMeta):
    """A mixin class that allows to chain transformation steps.

    The basic idea is to leverage the new data set ids from the previous transform step
    as the required data set ids for the current step. This avoids having to define the
    `data_set_ids` attribute.
    """

    def get_data_set_ids(self) -> Iterable[str]:
        """Get the data set ids to be processed.

        This uses the new data set ids from a previous transform step if set. Otherwise,
        falls back to the default behavior of returning the set data set ids from the
        constructor or class variable.

        Returns
        -------
        Iterable[str]
            An iterable of data set ids.
        """
        assert isinstance(
            self, ProcessingStep
        ), "TransformStepChainMixIn must inherit from ProcessingStep"
        # pylint: disable=no-member
        if isinstance(self.predecessor, TransformStep):
            return [self.predecessor.get_new_data_set_id()]
        # pylint: enable=no-member

        return super().get_data_set_ids()  # type: ignore[safe-super]


class TransformStep(TransformStepChainMixIn, MutateDataSetProcessingStepBase):
    r"""A raw data set transformation processing step.

    This class provides a convenient way to transform one or more data sets by
    specifying their ids, their level_ids or a level filter, a transformation function
    and specifications of a new data set to be returned as result of the processing
    step.

    Parameters
    ----------
    data_set_ids
        An optional list of data set ids to be used for the transformation. See
        :class:`~dispel.processing.data_set.DataSetProcessingStepMixin`.
    transform_function
        An optional function to be applied to the data sets. See
        :class:`~dispel.processing.data_set.MutateDataSetProcessingStepBase`. The transform
        function is expected to produce one or more columns of a data set according to
        the specification in `definitions`. The function can return NumPy unidimensional
        arrays, Pandas series and data frames.
    new_data_set_id
        An optional id used for the
        :class:`~dispel.data.raw.RawDataSetDefinition`. If no id was provided, the
        :data:`new_data_set_id` class variable will be used. Alternatively, one can
        overwrite :meth:`get_new_data_set_id` to provide the new data set id.
    definitions
        An optional list of :class:`~dispel.data.raw.RawDataValueDefinition` that has to
        match the number of columns returned by the :attr:`transform_function`. If no
        definitions were provided, the :data:`definitions` class variable will be used.
        Alternatively, one can overwrite :meth:`get_definitions` to provide the list of
        definitions.
    level_filter
        An optional filter to limit the levels being processed. See
        :class:`~dispel.processing.level.LevelProcessingStep`.
    storage_error
        This argument is only useful when the given new data id already exists.
        In which case, the following options are available:

            - ``'ignore'``: the computation of the transformation step for the concerned
              level will be ignored.
            - ``'overwrite'``: the existing data set id will be overwritten by the
              result of transform step computation.
            - ``'concatenate'``: the existing data set id will be concatenated with the
              result of transform step computation.
            - ``'raise'``: An error will be raised if we want to overwrite on an
              existing data set id.

    Examples
    --------
    Assuming you want to calculate the euclidean norm of a data set ``'acceleration'``
    for a specific level ``'left-small'`` and then name the new data set
    ``'accelerometer-norm'``, you can create the following step:

    >>> from dispel.data.raw import RawDataValueDefinition
    >>> from dispel.processing.transform import TransformStep
    >>> from dispel.signal.core import euclidean_norm
    >>> step = TransformStep(
    ...     'accelerometer',
    ...     euclidean_norm,
    ...     'accelerometer-norm',
    ...     [RawDataValueDefinition('norm', 'Accelerometer norm', 'm/s^2')]
    ... )

    The transformation function will be called with the specified data sets as
    arguments. If the function has named parameters matching ``level`` or ``reading``,
    the respective level and reading will be passed to the transformation function.

    Another common scenario is to define a class that can be reused.

    >>> from dispel.data.raw import RawDataValueDefinition
    >>> from dispel.processing.transform import TransformStep
    >>> class MyTransformStep(TransformStep):
    ...     data_set_ids = 'accelerometer'
    ...     transform_function = euclidean_norm
    ...     new_data_set_id = 'accelerometer-norm'
    ...     definitions = [
    ...         RawDataValueDefinition('norm', 'Accelerometer norm', 'm/s^2')
    ...     ]

    Another convenient way to provide the transformation function is to use the
    ``@transformation`` decorator:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from dispel.data.raw import RawDataValueDefinition
    >>> from dispel.processing.data_set import transformation
    >>> from dispel.processing.transform import TransformStep
    >>> class MyTransformStep(TransformStep):
    ...     data_set_ids = 'accelerometer'
    ...     new_data_set_id = 'accelerometer-norm'
    ...     definitions = [
    ...         RawDataValueDefinition('norm', 'Accelerometer norm', 'm/s^2')
    ...     ]
    ...
    ...     @transformation
    ...     def _euclidean_norm(self, data: pd.DataFrame) -> pd.Series:
    ...         return data.pow(2).sum(axis=1).apply(np.sqrt)

    Note that the decorated functions can also use ``level`` and ``reading`` as
    parameters to gain access to the respective level and reading being processed.
    """

    new_data_set_id: str

    definitions: List[RawDataValueDefinition]

    storage_error: StorageError = StorageError.RAISE

    def __init__(
        self,
        data_set_ids: Optional[Union[str, Iterable[str]]] = None,
        transform_function: Optional[Callable[..., Any]] = None,
        new_data_set_id: Optional[str] = None,
        definitions: Optional[List[RawDataValueDefinition]] = None,
        level_filter: Optional[LevelFilterType] = None,
        storage_error: Optional[
            Union[StorageError, Literal["raise", "ignore", "overwrite", "concatenate"]]
        ] = None,
    ):
        super().__init__(
            data_set_ids=data_set_ids,
            transform_function=transform_function,
            level_filter=level_filter,
        )

        if new_data_set_id:
            self.new_data_set_id = new_data_set_id
        if definitions:
            self.definitions = definitions
        if storage_error:
            self.storage_error = StorageError(storage_error)

    def get_new_data_set_id(self) -> str:
        """Get the id of the new data set to be created."""
        return self.new_data_set_id

    def get_definitions(self) -> List[RawDataValueDefinition]:
        """Get the definitions of the raw data set values."""
        return self.definitions

    def get_raw_data_set_definition(self):
        """Get the raw data set definition."""
        return RawDataSetDefinition(
            id=self.get_new_data_set_id(),
            source=RawDataSetSource(self.__class__.__name__),
            value_definitions_list=self.get_definitions(),
            is_computed=True,
        )

    def wrap_result(
        self, res: Any, level: Level, reading: Reading, **kwargs: Any
    ) -> WrapResultGeneratorType:
        """Wrap the result from the processing function into a class."""
        # handle series should they be provided
        if isinstance(res, (pd.Series, np.ndarray)):
            # Wrap into series if it is numpy array
            if isinstance(res, np.ndarray):
                assert res.ndim == 1, "Cannot handle multidimensional arrays"
                res = pd.Series(res)

            def_ids = [
                d.id for d in filter(lambda d: ~d.is_index, self.get_definitions())
            ]
            if len(def_ids) != 1:
                raise ValueError(
                    "Processing returned a series but did not get single "
                    "RawDataValueDefinition"
                )
            res = res.to_frame(def_ids[0])

        raw_data_set = RawDataSet(self.get_raw_data_set_definition(), res)

        yield RawDataSetProcessingResult(
            step=self,
            sources=self.get_raw_data_sets(level),
            result=raw_data_set,
            level=level,
            concatenate=self.storage_error.concatenate,
            overwrite=self.storage_error.overwrite,
        )

    def process_level(
        self, level: Level, reading: Reading, **kwargs
    ) -> ProcessResultType:
        """Process the provided Level."""
        raw_data_set_exists = level.has_raw_data_set(self.get_new_data_set_id())

        if raw_data_set_exists and self.storage_error == StorageError.RAISE:
            raise RawDataSetAlreadyExists(
                self.get_new_data_set_id(),
                level.id,
                'Please select for `storage_error` either "ignore" to ignore the '
                'transformation if the data set already exists, "overwrite" to '
                "overwrite the existing data set with the newly computed one, "
                '"concatenate" to try and concatenate the two raw data sets or simply '
                "change the name of the new data set to a valid one.",
            )
        if raw_data_set_exists and self.storage_error == StorageError.IGNORE:
            pass
        else:
            yield from super().process_level(level, reading, **kwargs)


class SuffixBasedNewDataSetIdMixin(DataSetProcessingStepProtocol, metaclass=ABCMeta):
    """A transformation step that can be chained to a previous step.

    In some scenarios it is desirable to simply name the new data set based on the input
    of the transformation step with a suffix. This can be achieved by adding the mixin
    :class:`SuffixBasedNewDataSetIdMixin` and using the ``&`` operator between the steps
    to be chained.

    Parameters
    ----------
    suffix
        The suffix to be added to the previous data set ids separated with an
        underscore. Alternatively, one can overwrite :meth:`get_suffix` to provide a
        dynamic suffix.

    Examples
    --------
    Assuming you two transform steps and an extract step

    .. code-block:: python

        steps = [
            InitialTransformStep(new_data_set_id='a'),
            SecondTransformStep(data_set_ids='a', new_data_set_id='a_b'),
            ExtractStep(data_set_ids='a_b')
        ]

    With the transform steps leverage the :class:`SuffixBasedNewDataSetIdMixin`, the
    same can be achieved by chaining the steps in the following way:

    .. code-block:: python

        steps = [
            InitialTransformStep(new_data_set_id='a') &
            SecondTransformStep(suffix='b') &
            ExtractStep()
        ]

    """

    suffix: str

    def __init__(self, *args, **kwargs):
        if suffix := kwargs.pop("suffix", None):
            self.suffix = suffix

        super().__init__(*args, **kwargs)

    def get_suffix(self):
        """Get the suffix to be added to the previous data set id."""
        return self.suffix

    def get_new_data_set_id(self) -> str:
        """Get the new data set id based on the chained step's ids and suffix.

        Returns
        -------
        str
            The data set ids of the previous step are concatenated with underscores
            (``_``) and combined with another underscore and the specified suffix
            obtained from :meth:`get_suffix`.
        """
        prefix = "_".join(self.get_data_set_ids())
        return f"{prefix}_{self.get_suffix()}"


class ConcatenateLevels(LevelFilterProcessingStepMixin, ProcessingStep):
    r"""A processing step that create a meta level.

    The meta level is created concatenating the data and merging the context. The
    contexts are merged by concatenating them with an extra ``_{k}`` in the name, ``k``
    incrementing from 0. The effective time frame is created taking the start of the
    first level and the end of the last one.

    Parameters
    ----------
    new_level_id
        The new level id that will be set inside the reading.
    data_set_id
        The ids of the data sets that will be concatenated.
    level_filter
        An optional :class:`~dispel.processing.level.LevelFilter` to determine the levels
        to be concatenated. If no filter is provided, all levels will be concatenated.
        The ``level_filter`` also accepts :class:`str`,
        :class:`~dispel.data.core.LevelId`\ s and lists of either and passes them to a
        :class:`~dispel.processing.level.LevelIdFilter` for convenience.
    """

    def __init__(
        self,
        new_level_id: LevelIdType,
        data_set_id: Union[str, List[str]],
        level_filter: Optional[LevelFilterType] = None,
    ):
        if isinstance(new_level_id, str):
            new_level_id = cast(LevelId, LevelId.from_str(new_level_id))

        if isinstance(data_set_id, str):
            data_set_id = [data_set_id]

        self.new_level_id = new_level_id
        self.data_set_id = data_set_id

        super().__init__(level_filter=level_filter)

    @staticmethod
    def _get_raw_data_sets(
        levels: Iterable[Level], data_set_id: str
    ) -> Generator[RawDataSet, None, None]:
        """Get the raw data sets corresponding to the given id."""

        def _filter_level(level_):
            return level_.has_raw_data_set(data_set_id)

        for level in filter(_filter_level, levels):
            yield level.get_raw_data_set(data_set_id)

    def get_levels(self, reading: Reading) -> Iterable[Level]:
        """Retrieve the levels used for level concatenation.

        Parameters
        ----------
        reading
            The reading used for processing.

        Returns
        -------
        Iterable[Level]
            The levels used for concatenation after filtering.
        """
        return sorted(
            self.get_level_filter()(reading.levels), key=lambda level: level.start
        )

    def process_reading(self, reading: Reading, **kwargs) -> ProcessResultType:
        """Create the meta level from reading."""
        # collect all matching level ids sorted by their start
        levels = self.get_levels(reading)
        # Check that the levels are not empty
        if len(list(levels)) == 0:
            return

        # collect raw data sets from all levels
        merged_raw_data_sets = []
        level: Optional[Level] = None
        raw_data_sets = []

        for data_set in self.data_set_id:
            data_set_definition = None
            raw_data_frames = []
            for raw_data_set in self._get_raw_data_sets(levels, data_set):
                raw_data_sets.append(raw_data_set)

                # assign first if not present
                if data_set_definition is None:
                    data_set_definition = raw_data_set.definition

                raw_data_frames.append(raw_data_set.data)

            if not data_set_definition:
                raise ValueError(f"No dataset definition for {data_set}.")

            merged_raw_data_sets.append(
                RawDataSet(data_set_definition, pd.concat(raw_data_frames))
            )

        # collect level information
        merged_context = Context()
        start = None
        for index, level in enumerate(levels):
            if start is None:
                start = level.start

            # combine context variables
            for key in level.context:
                definition = level.context[key].definition
                new_definition = ValueDefinition(
                    id_=f"{definition.id}_{index}",
                    name=definition.name,
                    unit=definition.unit,
                    description=definition.description,
                    data_type=definition.data_type,
                    validator=definition.validator,
                )
                merged_context.set(level.context[key].value, new_definition)

            merged_context.set(
                level, ValueDefinition(id_=f"level_{index}", name=f"Level {index}")
            )

        if level is None:
            raise ValueError("At least one level needs to be processed.")
        end = level.end

        new_level = Level(
            id_=self.new_level_id,
            start=start,
            end=end,
            context=merged_context,
            raw_data_sets=merged_raw_data_sets,
        )

        # TODO: Implement support for feature set concatenation

        yield ProcessingResult(step=self, sources=raw_data_sets, result=new_level)


class Apply(TransformStep):
    r"""Apply a method onto columns of a raw data set.

    Parameters
    ----------
    data_set_id
        The data set id of the data set on which the method is to be applied
    method
        The method in question. This can be any method that accepts a pandas series and
        returns an array of same length. See also :meth:`pandas.DataFrame.apply`.
    method_kwargs
        Optional arguments required for the methods.
    columns
        The columns to be considered during the method application.
    drop_nan
        ```True`` if NaN values are to be droped after transformation.
    level_filter
        An optional :class:`~dispel.processing.level.LevelFilter` to determine the levels
        to be transformed. If no filter is provided, all levels will be transformed. The
        ``level_filter`` also accepts :class:`str`, :class:`~dispel.data.core.LevelId`\ s
        and lists of either and passes them to a
        :class:`~dispel.processing.level.LevelIdFilter` for convenience.
    new_data_set_id
        The ``id`` used for the :class:`~dispel.data.raw.RawDataSetDefinition`.

    Examples
    --------
    Assuming you want to low-pass filter your gyroscope data of a ``reading`` you can
    create the following step to do so (note that the filtering expects a
    time-index-based and constant frequency-based data frame, so you might have to
    leverage :class:`~dispel.providers.generic.sensor.SetTimestampIndex` and
    :class:`~dispel.providers.generic.sensor.Resample` first):

    >>> from dispel.processing.transform import Apply
    >>> from dispel.signal.filter import butterworth_low_pass_filter
    >>> step = Apply(
    ...     'gyroscope_ts_resampled',
    ...     butterworth_low_pass_filter,
    ...     dict(cutoff=1.5, order=2),
    ...     list('xyz'),
    ... )

    This step will apply a 2. order butterworth low pass filter to the columns ``x``,
    ``y``, and ``z`` with a cut-off frequency of 1.5Hz.
    """

    def __init__(
        self,
        data_set_id: str,
        method: Callable[..., Any],
        method_kwargs: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None,
        new_data_set_id: Optional[str] = None,
        drop_nan: Optional[bool] = False,
        level_filter: Optional[LevelFilterType] = None,
    ):
        method_kwargs = method_kwargs or {}
        columns = columns or DEFAULT_COLUMNS

        def _transform_function(data: pd.DataFrame) -> pd.DataFrame:
            res = data[columns].apply(method, **method_kwargs)
            if drop_nan:
                return res.dropna()
            return res

        def _definition_factory(column: str) -> RawDataValueDefinition:
            return RawDataValueDefinition(
                column, f"{method.__name__} applied on {column}"
            )

        super().__init__(
            data_set_id,
            _transform_function,
            new_data_set_id or f"{data_set_id}_{method.__name__}",
            [_definition_factory(column) for column in columns],
            level_filter=level_filter,
        )


class Add(TransformStep):
    r"""Add the results of a method onto the columns of a raw data set data.

    Parameters
    ----------
    data_set_id
        The id of the data set to which the norm is added.
    method
        The method in question. It should output a pandas series with same length as the
        pandas data frame that it is fed.
    method_kwargs
        Optional arguments required for the methods.
    columns
        The columns on which the method is to be applied.
    level_filter
        An optional :class:`~dispel.processing.level.LevelFilter` to determine the levels
        to be transformed. If no filter is provided, all levels will be transformed.
        The ``level_filter`` also accepts :class:`str`,
        :class:`~dispel.data.core.LevelId`\ s and lists of either and passes them to a
        :class:`~dispel.processing.level.LevelIdFilter` for convenience.
    new_column
        The name of the new column.

    Examples
    --------
    Assuming you want to apply a euclidean norm onto accelerometer you can achieve this
    by chaining the following steps:

    .. doctest:: processing

        >>> from dispel.processing import process
        >>> from dispel.processing.transform import Add
        >>> from dispel.signal.core import euclidean_norm
        >>> step = Add(
        ...     'accelerometer',
        ...     euclidean_norm,
        ...     columns=list('xyz')
        ... )

    This step will apply a 2. order euclidean norm to the columns ``x``, ``y``, and
    ``z`` and add a column ``xyz`` to the transformed data set.
    """

    def __init__(
        self,
        data_set_id: str,
        method: Callable[..., Any],
        method_kwargs: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None,
        level_filter: Optional[LevelFilterType] = None,
        new_column: Optional[str] = None,
    ):
        kwargs: Dict[str, Any] = method_kwargs or {}
        old_columns: List[str] = columns or list("xyz")
        new_column = new_column or "".join(old_columns)

        def _transform_function(data: pd.DataFrame) -> pd.DataFrame:
            data_copy = data.copy()
            data_copy[new_column] = method(data_copy[columns], **kwargs)
            return data_copy

        def _definition_factory(column: str) -> RawDataValueDefinition:
            if column in old_columns:
                return RawDataValueDefinition(column, column)
            return RawDataValueDefinition(
                column, f"{method.__name__} applied on {column}"
            )

        super().__init__(
            data_set_id,
            _transform_function,
            f"{data_set_id}_{method.__name__}",
            [_definition_factory(column) for column in [*old_columns, new_column]],
            level_filter=level_filter,
        )
