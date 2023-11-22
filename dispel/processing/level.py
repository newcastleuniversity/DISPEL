"""Level processing functionalities."""
import inspect
import warnings
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from types import MethodType
from typing import Callable, Generator, Iterable, List, Optional, Set, Union, cast

from dispel.data.core import EntityType, Reading
from dispel.data.flags import Flag, FlagSeverity, FlagType
from dispel.data.levels import Level, LevelId
from dispel.data.values import AbbreviatedValue as AV
from dispel.processing.core import (
    CoreProcessingStepGroup,
    ProcessingControlResult,
    ProcessingResult,
    ProcessingStep,
    ProcessResultType,
)
from dispel.processing.flags import FlagStepMixin

MultipleLevelIdsType = Union[str, LevelId, List[str], List[LevelId]]


@dataclass(frozen=True)
class LevelProcessingResultBase:
    """The processing result base of a level processing step."""

    #: The level where the processing result is stored.
    level: Level

    def __post_init__(self):
        # Check that level id is not None
        if self.level is None:
            raise ValueError("level value cannot be null.")


@dataclass(frozen=True)
class LevelProcessingResult(ProcessingResult, LevelProcessingResultBase):
    """The processing result of a level processing step."""


@dataclass(frozen=True)
class LevelProcessingControlResult(ProcessingControlResult, LevelProcessingResultBase):
    """The processing result of an error in a level."""

    @classmethod
    def from_assertion_error(
        cls,
        step: "ProcessingStep",
        error: AssertionError,
        level: Optional[Level] = None,
    ):
        """Initialize object from a caught assertion error.

        Parameters
        ----------
        step
            The processing step from where the assertion error has been caught.
        error
            The assertion error that has been caught.
        level
            The level corresponding to the :class:`LevelProcessingControlResult`.

        Returns
        -------
        LevelProcessingControlResult
            The initialized level processing control result object.
        """
        assert level is not None, "Level cannot be null."

        res = ProcessingControlResult.from_assertion_error(step, error)
        return cls(
            step=res.step,
            targets=res.targets,
            error=res.error,
            level=level,
            error_handling=res.error_handling,
        )

    @classmethod
    def from_flag(
        cls,
        flag: Flag,
        step: "ProcessingStep",
        targets: Iterable[EntityType],
        level: Optional[Level] = None,
    ):
        """Initialize processing control result from an flag.

        Parameters
        ----------
        flag
            The flag from which the control processing result is to be created.
        step
            The associated processing step.
        targets
            The flag target entities.
        level
            The level corresponding to the :class:`LevelProcessingControlResult`.

        Returns
        -------
        LevelProcessingControlResult
            The initialized level processing control result object.
        """
        res = ProcessingControlResult.from_flag(flag, step, targets)
        assert isinstance(level, Level), "Level cannot be null."
        return cls(
            step=res.step,
            targets=res.targets,
            error=res.error,
            error_handling=res.error_handling,
            level=level,
        )


def _intersection(a, b):
    return a.intersection(b)


def _union(a, b):
    return a.union(b)


class LevelFilter(ABC):
    """A base class to filter levels during processing.

    :class:`LevelFilter` provide a central mechanism to differentiate processing steps
    in combination with :class:`LevelProcessingStep` and
    :class:`~dispel.processing.transform.ConcatenateLevels`. Each filter implementation
    must provide a :meth:`~LevelFilter.filter` function that consumes a container of
    levels and returns a set of levels containing those that should be processed.
    Furthermore, the method :meth:`~LevelFilter.repr` provides a hook to create a
    readable representation of the filter.

    Filters can also be combined by using logical operators ``&`` and ``|``.

    Examples
    --------
    Each level filter has to implement the methods ``filter`` and ``repr``. Assuming we
    want to create a filter that inspects some variables in the context of each level,
    we can do the following:

    >>> from typing import Any, Iterable, Set
    >>> from dispel.data.levels import Level
    >>> from dispel.processing.level import LevelFilter
    >>> class MyContextLevelFilter(LevelFilter):
    ...     def __init__(self, variable: str, value: Any):
    ...         self.variable = variable
    ...         self.value = value
    ...     def filter(self, levels: Iterable[Level]) -> Set[Level]:
    ...         return set(filter(
    ...             lambda level: level.context.get_raw_value(
    ...                 self.variable) == self.value, levels))
    ...     def repr(self) -> str:
    ...         return f'{self.variable} equals "{self.value}"'

    Given this filter one can now process levels with a specific context value by
    creating a filter like ``MyContextLevelFilter('usedHand', 'right')``.

    Since :class:`LevelFilter` can be used with logical operators one can create more
    complex filters by simply combining them as such:

    >>> MyContextLevelFilter('foo', 'bar') & MyContextLevelFilter('baz', 'bam')
    <LevelFilter: (foo equals "bar" and baz equals "bam")>

    This filter will now only consider levels where the context variables ``foo`` equal
    ``bar`` and ``baz`` equals ``bam``. This also works with or logic (``|``).

    One can also use the inverse logic by applying the ``~`` operator to a level filter
    in order to obtain its inverse.
    """

    def __call__(self, levels: Iterable[Level]) -> Set[Level]:
        """Filter level."""
        return self.filter(levels)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.repr()}>"

    def repr(self) -> str:
        """Get representation of the filter.

        Raises
        ------
        NotImplementedError
            This method is not implemented since there is no unambiguous representation
            of filters.
        """
        raise NotImplementedError

    def filter(self, levels: Iterable[Level]) -> Set[Level]:
        """Filter level.

        Parameters
        ----------
        levels
            The levels to be inspected for filtering

        Raises
        ------
        NotImplementedError
            This method is not implemented since there is no unambiguous definition of
            filters.
        """
        raise NotImplementedError

    def _combined(
        self, other: "LevelFilter", func: Callable[[Set, Set], Set]
    ) -> "LevelFilter":
        if not isinstance(other, LevelFilter):
            raise ValueError(f"Can only combine LevelFilters. Got: {type(other)}.")

        # avoid nesting default filter
        if isinstance(other, DefaultLevelFilter):
            return self
        if isinstance(self, DefaultLevelFilter):
            return other

        def _match(levels: Iterable[Level]) -> Set[Level]:
            return func(self(levels), other(levels))

        def _repr() -> str:
            op_name = {_intersection: "and", _union: "or"}
            return (
                f"({self.repr()} " f"{op_name[func]} {other.repr()})"
            )  # pylint: disable=W0212

        instance = LevelFilter()
        setattr(instance, "filter", _match)
        setattr(instance, "repr", _repr)

        return instance

    def __and__(self, other: "LevelFilter") -> "LevelFilter":
        return self._combined(other, _intersection)

    def __or__(self, other: "LevelFilter") -> "LevelFilter":
        return self._combined(other, _union)

    def __invert__(self) -> "LevelFilter":
        def _inverted_filter(levels: Iterable[Level]) -> Set[Level]:
            return set(levels) - self.filter(levels)

        def _repr() -> str:
            return f"Inverse of {self.repr()}"  # pylint: disable=W0212

        instance = LevelFilter()
        setattr(instance, "filter", _inverted_filter)
        setattr(instance, "repr", _repr)

        return instance


LevelFilterType = Union[MultipleLevelIdsType, LevelFilter]


class LevelIdFilter(LevelFilter):
    """A level filter based on level ids.

    Parameters
    ----------
    level_ids
        The level id(s) to be filtered for processing. The level id can be provided as
        :class:`str`, :class:`~dispel.data.core.LevelId` or lists of either. Levels with
        the respective provided ids will be considered during processing.
    """

    def __init__(self, level_ids: MultipleLevelIdsType):
        if isinstance(level_ids, str):
            level_ids = [LevelId(level_ids)]
        if isinstance(level_ids, LevelId):
            level_ids = [level_ids]
        if isinstance(level_ids, list):
            if all(isinstance(level_id, str) for level_id in level_ids):
                level_ids = [LevelId(cast(str, level_id)) for level_id in level_ids]
            elif any(not isinstance(level_id, LevelId) for level_id in level_ids):
                raise ValueError(
                    "The list of level_ids has to be filled only by {str}s or {"
                    "LevelId}s, never both."
                )

        self.level_ids = level_ids

    def repr(self) -> str:
        """Get representation of the filter."""
        return f"level id in {self.level_ids}"

    def filter(self, levels: Iterable[Level]) -> Set[Level]:
        """Filter levels being part of a predefined level id set."""
        return set(filter(lambda x: x.id in self.level_ids, levels))


class DefaultLevelFilter(LevelFilter):
    """A default level filter that considers all levels."""

    def repr(self) -> str:
        """Get representation of the filter."""
        return "*"

    def filter(self, levels: Iterable[Level]) -> Set[Level]:
        """Filter method returns a set of levels."""
        return set(levels)


class LevelProcessingStepProtocol(metaclass=ABCMeta):
    """Protocol for level processing steps."""

    @abstractmethod
    def assert_valid_level(self, level: Level, reading: Reading, **kwargs):
        """Perform assertions that a given level can be processed.

        Parameters
        ----------
        level
            The level to be tested for validity
        reading
            The parent reading of the level
        kwargs
            Additional keyword arguments passed to the processing function.
        """

    @abstractmethod
    def flag_level(
        self, level: Level, reading: Reading, **kwargs
    ) -> Generator[Flag, None, None]:
        """Flag the provided level.

        Parameters
        ----------
        level
            The level to be flagged.
        reading
            The reading associated to the provided level.
        kwargs
            Additional arguments passed by :func:`~dispel.processing.process`.

        Yields
        ------
        Flag
            The resulted flags.
        """
        yield NotImplemented

    def get_level_flag_targets(
        self, level: Level, reading: Reading, **kwargs
    ) -> Iterable[EntityType]:
        """Get the level flag targets.

        Parameters
        ----------
        level
            The level to be flagged.
        reading
            The reading associated with the level flag.
        kwargs
            Additional keyword arguments eventually used for flag targets
            extraction.

        Returns
        -------
        Iterable[EntityType]
            An iterable of entities that are flagged.
        """
        # pylint: disable=unused-argument
        return [level]

    @abstractmethod
    def process_level(
        self, level: Level, reading: Reading, **kwargs
    ) -> ProcessResultType:
        """Process the provided Level.

        Parameters
        ----------
        level
            The level to be processed
        reading
            The reading to be processed
        kwargs
            Additional arguments passed by :meth:`process_level`.

        Yields
        ------
        ProcessResultType
            Results from processing the level.
        """
        yield NotImplemented


class LevelFilterProcessingStepMixin:
    """A mixin class for all processing steps using level filters.

    Parameters
    ----------
    level_filter
        The filter to be used to select the levels to be processed.
    """

    level_filter: LevelFilter = DefaultLevelFilter()

    def __init__(self, *args, **kwargs):
        level_filter = kwargs.pop("level_filter", None)
        super().__init__(*args, **kwargs)

        if level_filter is not None:
            self.set_level_filter(level_filter)

    def get_level_filter(self) -> LevelFilter:
        """Get the level filter to sub-select levels to be processed."""
        return self.level_filter

    def set_level_filter(self, level_filter: LevelFilterType):
        """Set a level filter to sub-select levels to be processed.

        Parameters
        ----------
        level_filter
            The level filter to be used.
        """
        if isinstance(level_filter, (str, list, LevelId)):
            level_filter = LevelIdFilter(level_filter)

        self.level_filter = level_filter

    def inject_level_filter_from_step(self, step: "LevelFilterProcessingStepMixin"):
        """Inject the level filter from a step into the filter of this step.

        This function allows to have this processing step depend on the level
        filter of another step.

        Parameters
        ----------
        step
            A level filter processing step of which the level filter is used
            in this step too.
        """
        _func_cache_attr = "__original_get_level_filter"
        _injected_step_attr = "__injected_step"

        # only cache the original function once to avoid cascading filters
        # from multiple injections
        if not hasattr(self, _func_cache_attr):
            setattr(self, _func_cache_attr, self.get_level_filter)
        else:
            if (old_step := getattr(self, _injected_step_attr)) is not step:
                warnings.warn(
                    f"Re-assigning step {self} to a new {step} may lead to unintended "
                    f"side-effects with {old_step}.",
                    UserWarning,
                )

        def _get_level_filter(inner_self) -> LevelFilter:
            level_filter = getattr(inner_self, _func_cache_attr)()
            return level_filter & step.get_level_filter()

        # See https://github.com/python/mypy/issues/2427
        setattr(self, _injected_step_attr, step)
        setattr(
            self,
            "get_level_filter",
            MethodType(_get_level_filter, self),  # type: ignore
        )


class LevelProcessingStep(
    LevelProcessingStepProtocol, LevelFilterProcessingStepMixin, ProcessingStep
):
    r"""A level processing step is a processing step specific on levels.

    The level processing steps' :meth:`LevelProcessingStepProtocol.process_level` method
    is called with the level, reading and additional arguments passed to
    :meth:`~dispel.processing.core.ProcessingStep.process`. Results from the process step
    are expected to be an instance of :class:`~dispel.processing.core.ProcessingResult`.

    The :meth:`process_level` is only called with levels that pass the provided
    ``level_filter`` (see :class:`LevelFilter`). Each level will be processed if no
    level filter is provided. The ``level_filter`` also accepts :class:`str`,
    :class:`~dispel.data.core.LevelId`\ s and lists of either and passes them to a
    :class:`LevelIdFilter` for convenience.

    Examples
    --------
    .. testsetup:: processing-step

        >>> import pandas as pd
        >>> import numpy as np

        >>> from dispel.data.core import Reading
        >>> from dispel.data.levels import Level
        >>> from dispel.data.raw import (RawDataSet, RawDataSetDefinition,
        ...                           RawDataValueDefinition)

        >>> reading = Reading(
        ...     evaluation=None,
        ...     levels=[
        ...     Level(id_='my-level', start=0, end=1, raw_data_sets=[
        ...         RawDataSet(
        ...             RawDataSetDefinition('my-data-set', None, [
        ...                 RawDataValueDefinition('dummy', 'dummy')
        ...             ]),
        ...             pd.DataFrame({'dummy': list(range(6))})
        ...         )
        ...     ])
        ... ])

    .. doctest:: processing-step

        >>> from dispel.data.measures import MeasureValue
        >>> from dispel.data.values import ValueDefinition
        >>> from dispel.processing import process
        >>> from dispel.processing.level import (LevelProcessingStep,
        ...                                   LevelProcessingResult)
        >>> class MyLevelStep(LevelProcessingStep):
        ...     def process_level(self, level, reading, **kwargs):
        ...         raw_data_set = level.get_raw_data_set('my-data-set')
        ...         yield LevelProcessingResult(
        ...             step=self,
        ...             sources=raw_data_set,
        ...             level=level,
        ...             result=MeasureValue(
        ...                 ValueDefinition('my-measure-id', 'max value'),
        ...                 raw_data_set.data.max().max()
        ...             )
        ...         )
        >>> _ = process(reading, MyLevelStep())
        >>> reading.get_measure_set('my-level').get_raw_value('my-measure-id')
        5

    """

    def process_reading(self, reading: Reading, **kwargs) -> ProcessResultType:
        """Process all levels in reading.

        Parameters
        ----------
        reading
            The reading to be processed. Each level of the reading will be passed to the
             ``level_filter`` and if it returns ``True`` :meth:`process_level` is
             called.
        kwargs
            Additional named arguments. This allows to provide additional values, such
            as placeholder values in value definitions to the actual processing
            function.

        Yields
        ------
        ProcessResultType
            Passes through anything that is yielded from :meth:`process_level`.
        """
        for level in self.get_level_filter()(reading.levels):
            for flag in self.flag_level(level, reading, **kwargs):
                yield LevelProcessingControlResult.from_flag(
                    flag=flag,
                    step=self,
                    targets=self.get_level_flag_targets(level, reading, **kwargs),
                    level=level,
                )
            try:
                self.assert_valid_level(level, reading, **kwargs)
            except AssertionError as exception:
                yield LevelProcessingControlResult.from_assertion_error(
                    level=level, step=self, error=exception
                )
            else:
                yield from self.process_level(level, reading, **kwargs)

    def flag_level(
        self, level: Level, reading: Reading, **kwargs
    ) -> Generator[Flag, None, None]:
        """Flag the provided level."""
        yield from []

    def assert_valid_level(self, level: Level, reading: Reading, **kwargs):
        """Perform assertions that a given level can be processed."""

    @abstractmethod
    def process_level(
        self, level: Level, reading: Reading, **kwargs
    ) -> ProcessResultType:
        """Process the provided Level.

        Parameters
        ----------
        level
            The level to be processed
        reading
            The reading to be processed
        kwargs
            Additional arguments passed by :meth:`process_level`.

        Yields
        ------
        ProcessResultType
            Results from processing the level.
        """
        yield NotImplemented


class FlagLevelStep(FlagStepMixin, LevelProcessingStep):
    """A level flag class.

    Parameters
    ----------
    level_filter
        An optional filter to limit the levels being processed. See
        :class:`~dispel.processing.level.LevelProcessingStep`.
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
        An optional boolean that specifies whether the flag is stop_processing,
        i.e., raises an error or not. See
        :class:`~dispel.processing.flags.FlagStepMixin`.
    flagging_function
        An optional flagging function to be applied to :class:`~dispel.data.core.Level`.
        See :class:`~dispel.processing.flags.FlagStepMixin`.

    Examples
    --------
    Assuming you want to flag the time frame duration of a given level, you can use
    the following flag step:

    >>> from dispel.data.values import AbbreviatedValue as AV
    >>> from dispel.processing.level import FlagLevelStep
    >>> step = FlagLevelStep(
    ...     'utt',
    ...     AV('U-Turn test', 'utt'),
    ...     AV('u-turn duration', 'utt_dur'),
    ...     'technical',
    ...     'The U-Turn test duration exceeds 5 minutes.',
    ...     stop_processing=False,
    ...     flagging_function=lambda level: \
                level.effective_time_frame.duration.total_seconds() < 5 * 60,
    ... )

    The flagging function will be called with the level ``'utt'`` as specified in the
    ``level_filter`` argument. If the function has a named parameter matching
    ``reading``, the reading will be passed to the flagging function.

    Another common scenario is to define a class that can be reused.

    >>> from dispel.data.flags import FlagType
    >>> from dispel.processing.level import FlagLevelStep
    >>> class UTTDuration(FlagLevelStep):
    ...     task_name = AV('U-Turn test', 'utt')
    ...     flag_name = AV('u-turn duration', 'utt_dur')
    ...     flag_type = FlagType.TECHNICAL
    ...     flag_severity = FlagSeverity.DEVIATION
    ...     reason = 'The U-Turn test duration exceeds 5 minutes.'
    ...     stop_processing = False
    ...     flagging_function = lambda level: \
                level.effective_time_frame.duration.total_seconds() < 5 * 60

    Another convenient way to provide the flagging function is to use the
    ``@flag`` decorator, one can also use multiple flags for the same class
    as follows:

    >>> from dispel.data.levels import Level
    >>> from dispel.data.core import  Reading
    >>> from dispel.processing.flags import flag
    >>> from dispel.processing.level import FlagLevelStep
    >>> class UTTDuration(FlagLevelStep):
    ...     task_name = AV('U-Turn test', 'utt')
    ...     flag_name = AV('u-turn duration', 'utt_dur')
    ...     flag_type = 'technical'
    ...     flag_severity = FlagSeverity.DEVIATION
    ...     reason = 'The U-Turn test duration exceeds {duration} minutes.'
    ...     stop_processing = False
    ...
    ...     @flag(duration=5)
    ...     def _duration_5(self, level: Level) -> bool:
    ...         duration = level.duration
    ...         return duration.total_seconds() < 5 * 60
    ...
    ...     @flag(duration=4)
    ...     def _duration_4(self, level: Level, reading: Reading) -> bool:
    ...         evaluation_start = reading.evaluation.start
    ...         level_start = level.start
    ...         assert evaluation_start <= level_start
    ...
    ...         duration = level.duration
    ...         return duration.total_seconds() < 4 * 60

    Note that the ``@flag`` decorator can take keyword arguments. These kwargs are
    merged with any keyword arguments that come from processing step groups in order to
    format the flag ``reason``.
    """

    def __init__(
        self,
        level_filter: Optional[LevelFilterType] = None,
        task_name: Optional[Union[AV, str]] = None,
        flag_name: Optional[Union[AV, str]] = None,
        flag_type: Optional[Union[FlagType, str]] = None,
        flag_severity: Optional[Union[FlagSeverity, str]] = None,
        reason: Optional[Union[AV, str]] = None,
        stop_processing: bool = False,
        flagging_function: Optional[Callable[..., bool]] = None,
    ):
        super().__init__(
            level_filter=level_filter,
            task_name=task_name,
            flag_name=flag_name,
            flag_type=flag_type,
            flag_severity=flag_severity,
            reason=reason,
            stop_processing=stop_processing,
            flagging_function=flagging_function,
        )

    def process_level(
        self, level: Level, reading: Reading, **kwargs
    ) -> ProcessResultType:
        """Process the provided level."""
        yield from []

    def get_level_flag_targets(
        self, level: Level, reading: Reading, **kwargs
    ) -> Iterable[EntityType]:
        """Get flag targets for reading flag."""
        return self.get_flag_targets(reading, level, **kwargs)

    def get_flag_targets(
        self, reading: Reading, level: Optional[Level] = None, **kwargs
    ) -> Iterable[EntityType]:
        """Get flag targets for level flag."""
        assert level is not None, "Level cannot be null."
        return [level]

    def flag_level(
        self, level: Level, reading: Reading, **kwargs
    ) -> Generator[Flag, None, None]:
        """Flag the provided level."""
        for func, func_kwargs in self.get_flagging_functions():
            new_kwargs = kwargs.copy()
            if "reading" in inspect.getfullargspec(func).args:
                new_kwargs["reading"] = reading

            if not func(level, **new_kwargs):
                (merged_kwargs := kwargs.copy()).update(func_kwargs)
                yield self.get_flag(**merged_kwargs)


class ProcessingStepGroup(LevelFilterProcessingStepMixin, CoreProcessingStepGroup):
    r"""A group of processing steps with an optional level filter.

    For examples see :class:`dispel.processing.core.CoreProcessingStepGroup`. This class
    ensures that level filters are injected to the steps of this group.
    """

    def set_steps(self, steps: List[ProcessingStep]):
        """Set processing steps part of the group.

        This method ensures that steps added to the group inherit the level filter of
        the group.

        Parameters
        ----------
        steps
            The steps contained in the processing group.
        """
        for step in steps:
            if isinstance(step, LevelFilterProcessingStepMixin):
                step.inject_level_filter_from_step(self)

        super().set_steps(steps)

    def inject_level_filter_from_step(self, step: LevelFilterProcessingStepMixin):
        """Inject level filter into group and steps in group."""
        super().inject_level_filter_from_step(step)
        for group_step in self.get_steps():
            if isinstance(group_step, LevelFilterProcessingStepMixin):
                group_step.inject_level_filter_from_step(self)
