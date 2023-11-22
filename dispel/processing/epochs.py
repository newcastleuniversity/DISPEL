"""Epoch-specific processing steps."""

from abc import ABC, ABCMeta
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd

from dispel.data.core import Reading
from dispel.data.epochs import Epoch, EpochDefinition
from dispel.data.levels import Level, LevelEpoch, LevelEpochMeasureValue
from dispel.data.measures import MeasureValue
from dispel.data.raw import (
    RawDataSet,
    RawDataSetDefinition,
    RawDataSetSource,
    RawDataValueDefinition,
)
from dispel.processing.core import ProcessResultType
from dispel.processing.data_set import (
    DataSetProcessingStepProtocol,
    MutateDataSetProcessingStepBase,
    RawDataSetProcessingResult,
    StorageError,
    WrapResultGeneratorType,
)
from dispel.processing.extract import ExtractStep
from dispel.processing.level import LevelProcessingResult
from dispel.processing.transform import TransformStepChainMixIn


class LevelEpochDefinitionMixIn:
    """A mixin-class for processing steps producing epoch measure sets.

    Parameters
    ----------
    definition
        An optional epoch definition. If no epoch definition is provided, the
        :data:`definition` class variable will be used. Alternatively, one can overwrite
        :meth:`get_definition` to provide the definition.

    Attributes
    ----------
    definition
        The epoch definition. This will be used in :func:`get_definition` by default.
        You can overwrite the function to implement custom logic.
    """

    definition: Optional[EpochDefinition] = None

    def __init__(self, *args, **kwargs):
        definition = kwargs.pop("definition", None)
        self.definition = definition or self.definition

        super().__init__(*args, **kwargs)

    def get_definition(self, **_kwargs) -> EpochDefinition:
        """Get the measure definition.

        Other Parameters
        ----------------
        _kwargs
            Optional parameters that will be passed along to the creation of epoch
            definitions. This can be used to implement custom logic in the epoch
            definition that depends on processing arguments.

        Returns
        -------
        EpochDefinition
            The definition of the epoch
        """
        assert (
            self.definition is not None
        ), "Definition must be set or get_definition must be overwritten."

        return self.definition


class CreateLevelEpochStep(
    LevelEpochDefinitionMixIn, TransformStepChainMixIn, MutateDataSetProcessingStepBase
):
    """A processing step to create epochs.

    This class provides a convenient way to create epochs from one or more data sets by
    specifying their id, their level_ids or level filter, a transformation function and
    an epoch definition.

    Examples
    --------
    Assuming you have a data set and a method that derives specific epochs from this
    data set that are leveraged down the line for further processing. The following
    example illustrates creating level epochs from raw data sets.

    First, we specify a definition for our epochs to be extracted:

    >>> from dispel.data.epochs import EpochDefinition
    >>> definition = EpochDefinition(
    ...     id_='epoch-id',
    ...     name='Epoch name',
    ...     description='A detailed description of the epoch'
    ... )

    Next, we create a processing step that leverages a data set and returns the start
    and end of our bouts.

    >>> import pandas as pd
    >>> from scipy import signal
    >>> from dispel.processing.data_set import transformation
    >>> from dispel.processing.epochs import CreateLevelEpochStep
    >>> class MyCreateLevelEpochStep(CreateLevelEpochStep):
    ...     data_set_ids = 'data-set-id'
    ...     definition = definition
    ...     @transformation
    ...     def detect_bouts(self, data: pd.DataFrame) -> pd.DataFrame:
    ...         offset = pd.Timedelta(seconds=5)
    ...         peaks = signal.find_peaks(data['column'])
    ...         ts = data.index.iloc[peaks].to_series()
    ...         return pd.DataFrame(dict(start=ts - offset, end=ts + offset))

    The example above inspects the data set for peaks and returns epochs that start five
    seconds before the peak and end five seconds after.
    """

    #: If provided, the epochs will be additionally stored as a data set
    epoch_data_set_id: Optional[str] = None

    #: The behavior to handle multiple epochs being processed.
    storage_error = StorageError.CONCATENATE

    def get_epoch_data_set_id(self, **_kwargs) -> Optional[str]:
        """Get the data set id for the newly created epoch data set."""
        return self.epoch_data_set_id

    def get_epoch_data_set(self, epochs: Sequence[LevelEpoch], **kwargs) -> RawDataSet:
        """Get raw data set representation of a sequence of epochs."""
        # pylint: disable=superfluous-parens
        if not (data_set_id := self.get_epoch_data_set_id(**kwargs)):
            raise ValueError("No epoch data set ID was specified")

        definition = RawDataSetDefinition(
            id=data_set_id,
            source=RawDataSetSource(self.__class__.__name__),
            value_definitions_list=[
                RawDataValueDefinition("start", "Epoch Start"),
                RawDataValueDefinition("end", "Epoch End"),
                RawDataValueDefinition("epoch", "Epoch Object"),
            ],
            is_computed=True,
        )

        return RawDataSet(
            definition,
            pd.DataFrame(
                [
                    {"start": epoch.start, "end": epoch.end, "epoch": epoch}
                    for epoch in epochs
                ]
            ),
        )

    def transform_row_to_epoch(
        self, row: pd.Series, definition: EpochDefinition, **_kwargs
    ) -> LevelEpoch:
        """Get epoch representation for a row in a returned data set.

        This function is called by :func:`transform_data_frame_to_epochs` to transform a
        row into an epoch. It is applied when the transformation function returned a
        :class:`pandas.DataFrame`.

        Parameters
        ----------
        row
            The row in the data frame to be transformed into an epoch.
        definition
            The epoch definition.

        Other Parameters
        ----------------
        _kwargs
            Optional keyword arguments pushed down from the processing function. This
            can be used to implement custom logic in transforming rows to epochs based
            on processing arguments.

        Returns
        -------
        LevelEpoch
            The epoch representing the provided `row`. It uses the `start` and `end`
            column/names of the provided series as start and end of the epoch,
            respectively.
        """
        _epoch = LevelEpoch(start=row.start, end=row.end, definition=definition)
        if "measure_values" in row.index:
            for _value in row.measure_values:
                _epoch.set(_value)

        return _epoch

    def transform_data_frame_to_epochs(
        self, data: pd.DataFrame, definition: EpochDefinition, **kwargs
    ) -> List[LevelEpoch]:
        """Get a sequence of epochs for a data frame.

        This function is called if the result from a transformation function was a
        :class:`pandas.DataFrame`. It converts each row into an epoch.

        Parameters
        ----------
        data
            The data frame containing the epoch information. Each row is passed to
            :func:`transform_row_to_epoch` to convert it to an epoch.
        definition
            The epoch definition.

        Other Parameters
        ----------------
        kwargs
            Optional keyword arguments pushed down from the processing function.

        Returns
        -------
        List[LevelEpoch]
            A sequence of epochs representing the provided epochs from `data`.
        """
        return data.apply(
            self.transform_row_to_epoch, axis=1, definition=definition, **kwargs
        ).tolist()

    def wrap_result(
        self,
        res: Union[Epoch, LevelEpoch, pd.DataFrame],
        level: Level,
        reading: Reading,
        **kwargs: Any,
    ) -> WrapResultGeneratorType:
        """Wrap the result from the processing function into a class.

        Parameters
        ----------
        res
            The result passed from the transformation function. Supported values are
            :class:`Epoch`, :class:`LevelEpoch`, and :class:`pandas.DataFrame`.

            If an :class:`Epoch` was provided, the start and end times are copied to a
            new :class:`LevelEpoch` with the definition obtained from
            :func:`get_definition`. If a :class:`LevelEpoch` was returned, both values
            and flag will be copied over. If a :class:`pandas.DataFrame` was
            handed back, the data frame will be transformed using
            :func:`transform_data_frame_to_epochs`.
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

        Raises
        ------
        ValueError
            Will be risen if the value returned from the transformation function is of
            any other type than :class:`Epoch` or :class:`pandas.DataFrame`.
        """
        epochs = []
        definition = self.get_definition(level=level, reading=reading, **kwargs)
        if isinstance(res, Epoch):
            epoch = LevelEpoch(
                start=res.start,
                end=res.end,
                definition=definition,
            )

            if isinstance(res, LevelEpoch):
                epoch.add_flags(res)
                for value in res.values():
                    epoch.set(value)

            epochs.append(epoch)
        elif isinstance(res, pd.DataFrame):
            if not res.empty:
                epochs = self.transform_data_frame_to_epochs(res, definition, **kwargs)
        elif res is not None:
            raise ValueError(
                f"Unsupported type returned from transformation: {type(res)}"
            )

        # yield all epochs
        data_sets = self.get_raw_data_sets(level)
        for epoch in epochs:
            yield LevelProcessingResult(
                step=self, sources=data_sets, result=epoch, level=level
            )

        # yield epochs as data sets if needed
        epoch_data_set_id = self.get_epoch_data_set_id(
            level=level, reading=reading, **kwargs
        )
        if epochs and epoch_data_set_id:
            epoch_data_set = self.get_epoch_data_set(
                epochs, level=level, reading=reading, **kwargs
            )

            yield RawDataSetProcessingResult(
                step=self,
                sources=data_sets,
                result=epoch_data_set,
                level=level,
                concatenate=self.storage_error.concatenate,
                overwrite=self.storage_error.overwrite,
            )


class LevelEpochFilter(ABC):
    """A base class to filter level epochs during processing.

    :class:`LevelEpochFilter` provides a basic mechanism for processing steps using
    :class:`LevelEpochProcessingStepMixIn` to filter epochs to be processed. Each filter
    has to implement the :meth:`~LevelEpochFilter.filter` method that consumes an
    iterable of level epochs and returns a list of epochs to be considered during
    processing.
    """

    def filter(self, epochs: Iterable[LevelEpoch]) -> Sequence[LevelEpoch]:
        """Filter level epochs.

        Parameters
        ----------
        epochs
            The epochs to be filtered.

        Raises
        ------
        NotImplementedError
            This method is not implemented since there is no unambiguous definition of
            filters.
        """
        raise NotImplementedError

    def __call__(self, value, *args, **kwargs):
        """Filter level epochs."""
        return self.filter(value)


class DefaultLevelEpochFilter(LevelEpochFilter):
    """A default level epoch filter that passes all epochs for processing."""

    def filter(self, epochs: Iterable[LevelEpoch]) -> Sequence[LevelEpoch]:
        """Filter level epochs."""
        return list(epochs)


class LevelEpochIdFilter(LevelEpochFilter):
    """A level epoch filter that returns epochs with a specific id.

    Parameters
    ----------
    id_
        The definition id of the epoch to be matched.
    """

    def __init__(self, id_):
        self.id = id_

    def filter(self, epochs: Iterable[LevelEpoch]) -> Sequence[LevelEpoch]:
        """Filter all epochs matching the provided id."""
        return [epoch for epoch in epochs if epoch.id == self.id]


class LevelEpochProcessingStepMixIn(DataSetProcessingStepProtocol, metaclass=ABCMeta):
    """A mixin class for all processing steps using epochs to create measures.

    Parameters
    ----------
    epoch_filter
        The filter to be used when processing epochs.

    Examples
    --------
    The most common use case will be extracting measures for epochs using
    :class:`LevelEpochExtractStep`. The mixin class can also be combined with
    :class:`CreateLevelEpochStep` to create new epochs from existing epochs.

    >>> import pandas as pd
    >>> from dispel.processing.data_set import transformation
    >>> from dispel.processing.epochs import (LevelEpochIdFilter,
    ...     CreateLevelEpochStep, LevelEpochProcessingStepMixIn)
    >>> class MyStep(LevelEpochProcessingStepMixIn, CreateLevelEpochStep):
    ...     data_set_ids = 'data-set-id'
    ...     epoch_filter = LevelEpochIdFilter('existing-epoch-id')
    ...     definition = EpochDefinition(
    ...         id_='epoch-id',
    ...         name='Epoch name',
    ...         description='The new epochs derived from existing-epoch-id'
    ...     )
    ...
    ...     @transformation
    ...     def detect_epochs(self, data: pd.DataFrame) -> pd.DataFrame:
    ...         new_epoch_data_set = ...
    ...         return new_epoch_data_set

    The above example passes for each epoch with `existing-epoch-id` the view of
    `data-set-id` to the `detect_epochs` function. The returned data frame in turn will
    be converted to new epochs defined in `MyStep.definition`.
    """

    #: The filter to be used when processing epochs.
    epoch_filter: LevelEpochFilter = DefaultLevelEpochFilter()

    def __init__(
        self, *args, epoch_filter: Optional[LevelEpochFilter] = None, **kwargs
    ):
        if epoch_filter:
            self.epoch_filter = epoch_filter

        super().__init__(*args, **kwargs)

    def get_epoch_filter(self) -> LevelEpochFilter:
        """Get the epoch filter.

        This function is called by :meth:`LevelEpochProcessingStepMixIn.get_epochs` to
        filter down relevant epochs for processing.

        Returns
        -------
        LevelEpochFilter
            The filter to be used for processing.
        """
        return self.epoch_filter

    def get_epochs(
        self, level: Level, _reading: Reading, **_kwargs
    ) -> Iterable[LevelEpoch]:
        """Get epochs to be processed.

        Parameters
        ----------
        level
            The current level

        Other Parameters
        ----------------
        _reading
            The reading being processed. This parameter is not used in the default
            implementation, but can be used in any inheriting class to implement custom
            logic.
        _kwargs
            Additional arguments passed from processing.

        Returns
        -------
        Iterable[LevelEpoch]
            The epochs to be processed. Those are the epochs of the level that passed
            the epoch filter returned by
            :meth:`LevelEpochProcessingStepMixIn.get_epoch_filter`.
        """
        # pylint: disable=not-callable
        return self.get_epoch_filter()(level.epochs)

    def get_epoch_data_set_view(
        self, epoch: LevelEpoch, data_set: pd.DataFrame
    ) -> pd.DataFrame:
        """Get the view of a data set specific to an epoch.

        This method can be overwritten to implement custom logic to retrieve relevant
        parts of the passed `data_set`.

        Parameters
        ----------
        epoch
            The epoch for which to return the data set view
        data_set
            The data set for which to return a view

        Returns
        -------
        pandas.DataFrame
            The `data_set` view specific to the passed `epoch`.
        """
        assert not epoch.is_incomplete, "Can only process complete epochs"
        assert isinstance(
            data_set.index, pd.DatetimeIndex
        ), "Require pd.DatetimeIndex for processed data sets"

        return data_set[epoch.start : epoch.end]

    def get_epoch_data_set_views(
        self,
        data_sets: Sequence[pd.DataFrame],
        level: Level,
        reading: Reading,
        **kwargs,
    ) -> Sequence[Tuple[LevelEpoch, Sequence[pd.DataFrame]]]:
        """Get epoch based data set views for processing.

        Parameters
        ----------
        data_sets
            An iterable of :class:`pandas.DataFrame` to be processed.
        level
            The current level
        reading
            The reading

        Other Parameters
        ----------------
        kwargs
            Additional arguments passed from processing.


        Returns
        -------
        Sequence[Tuple[LevelEpoch, Sequence[pandas.DataFrame]]]
            A sequence of tuples that contain the epoch and the respective views of data
            sets to be processed.
        """
        epoch_views = []
        for epoch in self.get_epochs(level, reading, **kwargs):
            epoch_views.append(
                (
                    epoch,
                    [
                        self.get_epoch_data_set_view(epoch, data_set)
                        for data_set in data_sets
                    ],
                )
            )
        return epoch_views

    def process_data_sets(
        self,
        data_sets: Sequence[pd.DataFrame],
        level: Level,
        reading: Reading,
        **kwargs,
    ) -> ProcessResultType:
        """Process the provided data sets."""
        epoch_views = self.get_epoch_data_set_views(data_sets, level, reading, **kwargs)
        for epoch, data_set_view in epoch_views:
            yield from super().process_data_sets(
                data_set_view, level, reading, epoch=epoch, **kwargs
            )


@dataclass(frozen=True)
class LevelEpochProcessingResult(LevelProcessingResult):
    """A processing result originating from processing epochs."""

    epoch: LevelEpoch


class LevelEpochExtractStep(LevelEpochProcessingStepMixIn, ExtractStep):
    """A measure extraction processing step for epochs.

    Examples
    --------
    Assuming you have a set of epochs that for which you would like to extract a
    specific measure from a data set leveraging the data between the start and the end
    of each epoch, you can do this as follows:

    >>> from dispel.data.values import AbbreviatedValue as AV
    >>> from dispel.data.measures import MeasureValueDefinition
    >>> from dispel.processing.data_set import transformation
    >>> from dispel.processing.epochs import LevelEpochExtractStep, LevelEpochIdFilter
    >>> class MyLevelEpochExtractStep(LevelEpochExtractStep):
    ...     data_set_ids = 'data_set_id'
    ...     epoch_filter = LevelEpochIdFilter('a')
    ...     definition = MeasureValueDefinition(
    ...         task_name=AV('My Task', 'MT'),
    ...         measure_name=AV('Maximum', 'max'),
    ...         data_type='int16'
    ...     )
    ...
    ...     @transformation
    ...     def max(self, data: pd.DataFrame) -> float:
    ...        return data['column'].max()

    The example above will create a measure value for each epoch containing the maximum
    of the `'column'` column in the data set with the `'data_set_id'`.
    """

    def get_value(self, value: Any, **kwargs) -> MeasureValue:
        """Get a measure value based on the definition.

        Parameters
        ----------
        value
            The value
        kwargs
            Optional arguments passed to :meth:`get_definition`.

        Returns
        -------
        MeasureValue
            The ``value`` wrapped with the definition from :meth:`get_definition`.
        """
        assert "epoch" in kwargs, "Missing epoch in passed arguments"
        return LevelEpochMeasureValue(
            kwargs["epoch"], self.get_definition(**kwargs), value
        )

    def wrap_result(
        self, res: Any, level: Level, reading: Reading, **kwargs: Any
    ) -> WrapResultGeneratorType:
        """Wrap the result from the processing function into a class.

        Parameters
        ----------
        res
            Any result returned by the extraction step. If res is a WrappedResult, the
            flag contained in the object will be automatically added to the
            :class:`~dispel.data.measures.MeasureValue` hence the flagged wrapped
            results will always translate into flagged
            :class:`~dispel.data.measures.MeasureValue` .
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
        # pylint: disable=stop-iteration-return
        wrapped_result = next(super().wrap_result(res, level, reading, **kwargs))
        assert isinstance(
            wrapped_result, LevelProcessingResult
        ), f"Expected LevelProcessingResult, but got: {type(wrapped_result)}"

        assert "epoch" in kwargs, "Missing epoch in passed arguments"
        epoch = kwargs["epoch"]
        assert isinstance(epoch, LevelEpoch)

        if not isinstance(sources := wrapped_result.sources, Iterable):
            sources = [sources]

        # Create new one with the wrapped result from super
        yield LevelEpochProcessingResult(
            step=self,
            sources=list(sources) + [epoch],
            result=wrapped_result.result,
            level=level,
            epoch=epoch,
        )
