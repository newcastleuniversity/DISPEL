"""Data set processing functionalities."""
import inspect
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum
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
)

import pandas as pd

from dispel.data.core import EntityType, Reading
from dispel.data.flags import Flag, FlagSeverity, FlagType
from dispel.data.levels import Level
from dispel.data.raw import RawDataSet
from dispel.data.values import AbbreviatedValue as AV
from dispel.processing.core import ErrorHandling, ProcessResultType
from dispel.processing.flags import FlagStepMixin
from dispel.processing.level import (
    LevelFilterType,
    LevelProcessingControlResult,
    LevelProcessingResult,
    LevelProcessingStep,
    LevelProcessingStepProtocol,
)
from dispel.processing.utils import TaskMixin


@dataclass(frozen=True)
class RawDataSetProcessingResult(LevelProcessingResult):
    """The processing result of a transform step."""

    #: Whether to concatenate the result if it already exists in the
    #: given level
    concatenate: bool = False
    #: Whether to overwrite the result if it already exists in the given
    #: level
    overwrite: bool = False

    def __post_init__(self):
        if self.concatenate and self.overwrite:
            raise ValueError(
                "You cannot both concatenate and overwrite the output of the "
                "transformation function. Only one of these arguments must be set to "
                "``True``."
            )


class StorageError(Enum):
    """Raw data set storage handler."""

    RAISE = "raise"
    IGNORE = "ignore"
    OVERWRITE = "overwrite"
    CONCATENATE = "concatenate"

    @property
    def overwrite(self) -> bool:
        """Return ``True`` if the handling is to overwrite."""
        return self == self.OVERWRITE

    @property
    def concatenate(self) -> bool:
        """Return ``True`` if the handling is to concatenate."""
        return self == self.CONCATENATE


class DataSetProcessingStepProtocol(metaclass=ABCMeta):
    """Abstract class for data set processing steps."""

    @abstractmethod
    def process_data_sets(
        self,
        data_sets: Sequence[pd.DataFrame],
        level: Level,
        reading: Reading,
        **kwargs,
    ) -> ProcessResultType:
        """Process the provided data sets.

        Parameters
        ----------
        data_sets
            The data sets to be processed.
        level
            The level to be processed.
        reading
            The reading to be processed.
        kwargs
            Additional arguments passed by
            :meth:`~dispel.processing.level.LevelProcessingStep.process_level`.

        Yields
        ------
        ProcessResultType
            Results from processing the data sets.
        """
        yield NotImplemented

    @abstractmethod
    def get_data_set_ids(self) -> Iterable[str]:
        """Get the data set ids to be processed."""

    @abstractmethod
    def get_raw_data_sets(self, level: Level) -> List[RawDataSet]:
        """Get the raw data sets from all data sets in question.

        Parameters
        ----------
        level
            The level from which to get the data sets.

        Returns
        -------
        List[RawDataSet]
            A list of all raw data sets with the specified ids.
        """
        return NotImplemented

    @abstractmethod
    def get_data_frames(self, level: Level) -> List[pd.DataFrame]:
        """Get the raw data from all data sets in question.

        Parameters
        ----------
        level
            The level from which to get the data sets.

        Returns
        -------
        List[pandas.DataFrame]
            A list of all raw data frames with the specified ids.
        """
        return NotImplemented

    @abstractmethod
    def assert_valid_data_sets(
        self,
        data_sets: Sequence[pd.DataFrame],
        level: Level,
        reading: Reading,
        **kwargs,
    ):
        """Assert that the to be processed data sets are valid."""

    @abstractmethod
    def flag_data_sets(
        self,
        data_sets: Sequence[pd.DataFrame],
        level: Level,
        reading: Reading,
        **kwargs,
    ) -> Generator[Flag, None, None]:
        """Flag the provided data sets."""
        yield NotImplemented

    def get_data_sets_flag_targets(
        self,
        data_sets: Sequence[pd.DataFrame],
        level: Level,
        reading: Reading,
        **kwargs,
    ) -> Iterable[EntityType]:
        """Get the level flag targets.

        Parameters
        ----------
        data_sets
            The data sets to be flagged.
        level
            The level associated with the data sets.
        reading
            The reading associated with the data set flag.
        kwargs
            Additional keyword arguments eventually used for flag targets extraction.

        Returns
        -------
        Iterable[EntityType]
            An iterable of entities that are flagged.
        """
        # pylint: disable=unused-argument
        return self.get_raw_data_sets(level)


class DataSetProcessingStepMixin(
    TaskMixin,
    DataSetProcessingStepProtocol,
    LevelProcessingStepProtocol,
    metaclass=ABCMeta,
):
    """A mixin class that processes data sets."""

    #: An iterable of data sets to be being processed
    data_set_ids: Union[str, Iterable[str]]

    def __init__(self, *args, **kwargs):
        data_set_ids = kwargs.pop("data_set_ids", None)
        if data_set_ids is not None:
            self.data_set_ids = data_set_ids

        super().__init__(*args, **kwargs)

    def get_data_set_ids(self) -> Iterable[str]:
        """Get the data set ids to be processed."""
        if isinstance(self.data_set_ids, str):
            return [self.data_set_ids]

        return self.data_set_ids

    def get_raw_data_sets(self, level: Level) -> List[RawDataSet]:
        """Get the raw data sets from all data sets in question.

        Parameters
        ----------
        level
            The level from which to get the data sets.

        Returns
        -------
        List[RawDataSet]
            A list of all raw data sets with the specified ids.
        """
        return list(map(level.get_raw_data_set, self.get_data_set_ids()))

    def get_data_frames(self, level: Level) -> List[pd.DataFrame]:
        """Get the raw data from all data sets in question.

        Parameters
        ----------
        level
            The level from which to get the data sets.

        Returns
        -------
        List[pandas.DataFrame]
            A list of all raw data frames with the specified ids.
        """
        return list(map(lambda r: r.data, self.get_raw_data_sets(level)))

    def assert_valid_level(self, level: Level, reading: Reading, **kwargs):
        """Assert that the level has the appropriate valid data sets."""
        super().assert_valid_level(level, reading, **kwargs)
        for id_ in self.get_data_set_ids():
            if not level.has_raw_data_set(id_):
                raise AssertionError(f"{id_} not found.", ErrorHandling.RAISE)


class DataSetProcessingStep(
    DataSetProcessingStepMixin, LevelProcessingStep, metaclass=ABCMeta
):
    """A processing step that processes data sets.

    Parameters
    ----------
    data_set_ids
        Optional data set ids to be processed. See :class:`DataSetProcessingStepMixin`.
    level_filter
        Optional level filter. See :class:`~dispel.processing.level.LevelProcessingStep`.
    """

    def __init__(
        self,
        data_set_ids: Optional[Union[str, Iterable[str]]] = None,
        level_filter: Optional[LevelFilterType] = None,
    ):
        super().__init__(data_set_ids=data_set_ids, level_filter=level_filter)

    def process_level(
        self, level: Level, reading: Reading, **kwargs
    ) -> ProcessResultType:
        """Process the provided Level."""
        for flag in self.flag_data_sets(
            (data_sets := self.get_data_frames(level)), level, reading, **kwargs
        ):
            yield LevelProcessingControlResult.from_flag(
                flag=flag,
                step=self,
                targets=self.get_data_sets_flag_targets(data_sets, level, reading),
                level=level,
            )
        try:
            self.assert_valid_data_sets(data_sets, level, reading, **kwargs)
        except AssertionError as exception:
            yield LevelProcessingControlResult.from_assertion_error(
                level=level, step=self, error=exception
            )
        else:
            yield from self.process_data_sets(data_sets, level, reading, **kwargs)

    def assert_valid_data_sets(
        self,
        data_sets: Sequence[pd.DataFrame],
        level: Level,
        reading: Reading,
        **kwargs,
    ):
        """Perform assertions that a given data sets can be processed."""

    def flag_data_sets(
        self,
        data_sets: Sequence[pd.DataFrame],
        level: Level,
        reading: Reading,
        **kwargs,
    ) -> Generator[Flag, None, None]:
        """Flag the provided data sets."""
        yield from []

    def process_data_sets(
        self,
        data_sets: Sequence[pd.DataFrame],
        level: Level,
        reading: Reading,
        **kwargs,
    ) -> ProcessResultType:
        """Process the provided data sets."""
        yield from []


def transformation(_func=None, **kwargs):
    """Decorate a function as a transformation function."""

    def wrapper(func):
        func.__transform_function__ = True
        func.__transform_kwargs__ = kwargs
        return func

    if _func is None:
        return wrapper

    return wrapper(_func)


def decorated_processing_function(
    func: Callable[..., Any],
    data_sets: Sequence[pd.DataFrame],
    reading: Reading,
    level: Level,
    **kwargs,
) -> Any:
    """Decorate functions for processing steps.

    Pass reading and/or level in args if existent in function arguments.

    Parameters
    ----------
    func
        The processing function to be decorated.
    data_sets
        The data frames on which the processing function is to be applied.
    reading
        The corresponding reading.
    level
        The corresponding level.
    kwargs
        Additional key word arguments to be passed to the processing function.

    Returns
    -------
    Any
        The output of the given processing function.
    """
    func_args = inspect.getfullargspec(func).args
    new_kwargs: Dict[str, Any] = kwargs.copy()

    if "level" in func_args:
        new_kwargs["level"] = level
    if "reading" in func_args:
        new_kwargs["reading"] = reading

    return func(*data_sets, **new_kwargs)


TransformationFunctionGeneratorType = Generator[
    Tuple[Callable, Dict[str, Any]], None, None
]

WrapResultGeneratorType = Generator[
    Union[LevelProcessingResult, RawDataSetProcessingResult], None, None
]


class MutateDataSetProcessingStepBase(DataSetProcessingStep, metaclass=ABCMeta):
    """A base class for transformation and extraction steps.

    Parameters
    ----------
    data_set_ids
        An optional list of data set ids to be used for the transformation. See
        :class:`DataSetProcessingStepMixin`.
    transform_function
        An optional function to be applied to the data sets. If no function is passed
        the class variable :data:`transform_function` will be used. Alternatively, the
        :meth:`get_transform_function` can be overwritten to provide the transformation
        function. If there is more than one function to be applied one can overwrite
        :meth:`get_transform_functions`. Otherwise, all class functions decorated with
        ``@transformation`` will be considered as a transformation function.
    level_filter
        An optional filter to limit the levels being processed. See
        :class:`~dispel.processing.level.LevelProcessingStep`.
    """

    #: The function to be applied to the data sets.
    transform_function = None

    def __init__(
        self,
        data_set_ids: Optional[Union[str, Iterable[str]]] = None,
        transform_function: Optional[Callable[..., Any]] = None,
        level_filter: Optional[LevelFilterType] = None,
    ):
        super().__init__(data_set_ids=data_set_ids, level_filter=level_filter)
        self.transform_function = transform_function or self.transform_function

    def get_transform_function(self) -> Optional[Callable[..., Any]]:
        """Get the transformation function."""
        # unbind bound methods
        func = self.transform_function
        if func is not None and hasattr(func, "__func__"):
            return func.__func__  # type: ignore
        return func

    def get_transform_functions(self) -> TransformationFunctionGeneratorType:
        """Get all transformation functions associated with this step."""
        if func := self.get_transform_function():
            yield func, {}

        members = inspect.getmembers(self, predicate=inspect.isroutine)
        for _, func in members:
            if func is not None and hasattr(func, "__transform_function__"):
                yield func, func.__transform_kwargs__  # type: ignore

    @abstractmethod
    def wrap_result(
        self, res: Any, level: Level, reading: Reading, **kwargs: Any
    ) -> WrapResultGeneratorType:
        """Wrap the transformation result into a processing result."""
        yield NotImplemented

    def process_data_sets(
        self,
        data_sets: Sequence[pd.DataFrame],
        level: Level,
        reading: Reading,
        **kwargs,
    ) -> ProcessResultType:
        """Process the provided data sets."""
        for function, func_kwargs in self.get_transform_functions():
            (merged_kwargs := kwargs.copy()).update(func_kwargs)

            yield from self.wrap_result(
                decorated_processing_function(function, data_sets, reading, level),
                level,
                reading,
                **merged_kwargs,
            )


class FlagDataSetStep(FlagStepMixin, DataSetProcessingStep, metaclass=ABCMeta):
    """A data set flag class.

    Parameters
    ----------
    data_set_ids
        An optional id or iterable of ids for raw data set(s) to be used for the
        flag. See :class:`DataSetProcessingStepMixin`.
    level_filter
        An optional filter to limit the levels being processed.
        See :class:`~dispel.processing.level.LevelProcessingStep`.
    task_name
        An optional abbreviated name value of the task used for the flag.
        See :class:`~dispel.processing.flags.FLagStepMixin`.
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
        An optional boolean that specifies whether the flag is stop_processing,
        i.e., raises an error or not.
        See :class:`~dispel.processing.flags.FlagStepMixin`.
    flagging_function
        An optional flagging function to be applied to the pandas data frames of the
        provided raw data sets.
        See :class:`~dispel.processing.flags.FlagStepMixin`.
    target_ids
        An optional id(s) of the target data sets to be flagged. If the user doesn't
        specify the targets then the targets will automatically be the used data sets.

    Examples
    --------
    Assuming you want to flag the accelerometer signal data of the U-Turn task to
    verify that it doesn't exceed a certain threshold, you can use the following
    flag step:

    >>> from dispel.data.values import AbbreviatedValue as AV
    >>> from dispel.processing.data_set import FlagDataSetStep
    >>> step = FlagDataSetStep(
    ...     data_set_ids = 'accelerometer',
    ...     level_filter = 'utt',
    ...     task_name = AV('U-Turn test', 'utt'),
    ...     flag_name = AV('accelerometer signal threshold', 'ast'),
    ...     flag_type = FlagType.TECHNICAL,
    ...     flag_severity = FlagSeverity.INVALIDATION,
    ...     reason = 'The U-Turn accelerometer signal exceeds 50 m/s^2.',
    ...     stop_processing=False,
    ...     flagging_function=lambda data: data.max().max() < 50.
    ... )

    The flagging function will be called with the level ``'utt'`` as specified in the
    ``level_filter`` argument. If the function has a named parameter matching
    ``reading``, the reading will be passed to the flagging function.

    Another common scenario is to define a class that can be reused.

    >>> from dispel.data.flags import FlagType
    >>> from dispel.processing.data_set import FlagDataSetStep
    >>> class UTTAccelerometerSignal(FlagDataSetStep):
    ...     data_set_ids = 'accelerometer'
    ...     level_filter = 'utt'
    ...     task_name = AV('U-Turn test', 'utt')
    ...     flag_name = AV('u-turn duration', 'utt_dur')
    ...     flag_type = FlagType.TECHNICAL
    ...     flag_severity = FlagSeverity.INVALIDATION
    ...     reason = 'The U-Turn accelerometer signal exceeds 50 m/s^2.'
    ...     stop_processing = True
    ...     flagging_function = lambda data: data.max().max() < 50

    Another convenient way to provide the flagging function is to use the
    ``@flag`` decorator, one can also use multiple flags for the same class
    as well as multiple data sets. Below is an example of the flag of a data set
    (``userInput``) through the use of multiple ones in the flagging function
    (``userInput``, ``screen``).

    >>> import pandas as pd
    >>> from dispel.processing.flags import flag
    >>> from dispel.processing.level import FlagLevelStep
    >>> class UTTAccelerometerSignal(FlagDataSetStep):
    ...     data_set_ids = ['userInput', 'screen']
    ...     target_ids = 'userInput'
    ...     level_filter = 'cps'
    ...     task_name = AV('Cognitive processing speed test', 'cps')
    ...     flag_name = AV('answer timestamps', 'at')
    ...     flag_type = FlagType.TECHNICAL
    ...     flag_severity = FlagSeverity.INVALIDATION
    ...     reason = 'The user answer timestamps do not match the screen info.'
    ...     stop_processing = False
    ...
    ...     @flag
    ...     def _timestamps(
    ...         self,
    ...         user_input: pd.DataFrame,
    ...         screen: pd.DataFrame
    ...     ) -> bool:
    ...         return list(user_input.ts) == list(screen.ts)

    Note that the ``@flag`` decorator can take keyword arguments. These kwargs are
    merged with any keyword arguments that come from processing step groups in order to
    format the flag ``reason``.
    """

    target_ids: Optional[Union[Iterable[str], str]] = None

    def __init__(
        self,
        data_set_ids: Optional[Union[str, Iterable[str]]] = None,
        level_filter: Optional[LevelFilterType] = None,
        task_name: Optional[Union[AV, str]] = None,
        flag_name: Optional[Union[AV, str]] = None,
        flag_type: Optional[Union[FlagType, str]] = None,
        flag_severity: Optional[Union[FlagSeverity, str]] = None,
        reason: Optional[Union[AV, str]] = None,
        stop_processing: bool = False,
        flagging_function: Optional[Callable[..., bool]] = None,
        target_ids: Optional[Union[Iterable[str], str]] = None,
    ):
        if target_ids:
            self.target_ids = target_ids

        super().__init__(
            data_set_ids=data_set_ids,
            level_filter=level_filter,
            task_name=task_name,
            flag_name=flag_name,
            flag_type=flag_type,
            flag_severity=flag_severity,
            reason=reason,
            stop_processing=stop_processing,
            flagging_function=flagging_function,
        )

    def get_target_ids(self) -> Iterable[str]:
        """Get the ids of the target data sets to be flagged.

        Returns
        -------
        str
            The identifiers of the target data sets.
        """
        if self.target_ids is None:
            return self.get_data_set_ids()
        if isinstance(self.target_ids, str):
            return [self.target_ids]
        return self.target_ids

    def process_data_sets(
        self,
        data_sets: Sequence[pd.DataFrame],
        level: Level,
        reading: Reading,
        **kwargs,
    ) -> ProcessResultType:
        """Process the provided data sets."""
        yield from []

    def get_data_sets_flag_targets(
        self,
        data_sets: Sequence[pd.DataFrame],
        level: Level,
        reading: Reading,
        **kwargs,
    ) -> Iterable[EntityType]:
        """Get flag targets for data sets flagging."""
        return self.get_flag_targets(reading, level, **kwargs)

    def get_flag_targets(
        self, reading: Reading, level: Optional[Level] = None, **kwargs
    ) -> Iterable[EntityType]:
        """Get flag targets for data set flagging."""
        assert level is not None, "Missing level in kwargs."
        return [level.get_raw_data_set(id_) for id_ in self.get_target_ids()]

    def flag_data_sets(
        self,
        data_sets: Sequence[pd.DataFrame],
        level: Level,
        reading: Reading,
        **kwargs,
    ) -> Generator[Flag, None, None]:
        """Flag the provided data sets."""
        for func, func_kwargs in self.get_flagging_functions():
            if not decorated_processing_function(func, data_sets, reading, level):
                (merged_kwargs := kwargs.copy()).update(func_kwargs)
                yield self.get_flag(**merged_kwargs)
