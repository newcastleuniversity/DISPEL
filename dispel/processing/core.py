r"""Core functionality to process :class:`~dispel.data.core.Reading`\ s."""
import copy
import inspect
import warnings
from abc import abstractmethod
from collections import abc
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from dispel.data.core import EntityType, Reading
from dispel.data.features import FeatureSet, FeatureValue
from dispel.data.flags import Flag, FlagSeverity, FlagType
from dispel.data.levels import Level, LevelEpoch
from dispel.data.raw import RawDataSet
from dispel.data.values import AbbreviatedValue as AV
from dispel.processing.flags import FlagStepMixin

SourcesType = Union[Iterable[EntityType], EntityType]

TargetsType = Union[Iterable[EntityType], EntityType]

ParameterType = TypeVar("ParameterType")


class ProcessingError(Exception):
    """A base error class for errors that were caused during the processing.

    Parameters
    ----------
    step
        The processing step that raised the exception
    message
        The error message
    """

    def __init__(self, message: str, step: "ProcessingStep"):
        self.step = step
        super().__init__(f"{message} at step: {step.__class__.__name__}.")


class StopProcessingError(ProcessingError):
    """An error that halts the processing of a reading."""


class FlagError(ProcessingError):
    """An error that flags the processing results from a reading."""

    def __init__(self, flag: Flag, step: "ProcessingStep"):
        self.flag = flag
        super().__init__(self.flag.reason, step)


class InvalidDataError(ProcessingError):
    """An error that flags the processing input."""


@dataclass(frozen=True)
class ProcessingResultBase:
    """The processing result base of a processing step."""

    #: The processing step associated with the processing
    step: "ProcessingStep"

    def get_kwargs(self) -> Dict[str, Any]:
        """Get key word arguments for the additional class attributes.

        The additional arguments are passed to the setter function in
        :class:`~dispel.data.core.Reading` while assigning processing results and creating
        the data trace. It allows to handle the result to be set depending on the
        arguments passed.

        Returns
        -------
        Dict[str, Any]
            Returns a dictionary of attributes of the instance omitting the step.
        """
        kwargs = self.__dict__.copy()
        del kwargs["step"]
        return kwargs


@dataclass(frozen=True)
class ProcessingResult(ProcessingResultBase):
    """The processing result of a processing step."""

    #: The sources used for the processing function
    sources: SourcesType

    #: The result of the processing function
    result: Union[Level, FeatureValue, FeatureSet, LevelEpoch, RawDataSet]

    def get_kwargs(self) -> Dict[str, Any]:
        """Get key word arguments for the additional class attributes.

        The additional arguments are passed to the setter function in
        :class:`~dispel.data.core.Reading` while assigning processing results and creating
        the data trace. It allows to handle the result to be set depending on the
        arguments passed.

        Returns
        -------
        Dict[str, Any]
            Returns a dictionary of attributes of the instance omitting the step,
            sources, and result.
        """
        kwargs = super().get_kwargs()
        del kwargs["sources"]
        del kwargs["result"]
        return kwargs

    def get_sources(self) -> Iterable[SourcesType]:
        """Get the sources of the processing result."""
        if isinstance(self.sources, abc.Iterable):
            return self.sources
        return [self.sources]


class ErrorHandling(Enum):
    """Different ways of dealing with an exception."""

    RAISE = "raise"
    IGNORE = "ignore"

    @property
    def should_raise(self) -> bool:
        """Return ``True`` if the handling is to overwrite."""
        return self is self.RAISE

    def __bool__(self) -> bool:
        return self is self.RAISE

    @classmethod
    def from_bool(cls, stop_processing: bool) -> "ErrorHandling":
        """Create the error handling class from a stop_processing boolean."""
        if stop_processing:
            return ErrorHandling.RAISE
        return ErrorHandling.IGNORE


@dataclass(frozen=True)
class ProcessingControlResult(ProcessingResultBase):
    """The processing result of a processing error."""

    #: The captured processing error
    error: Union[ProcessingError, StopProcessingError, FlagError, InvalidDataError]

    #: The type of handling for the captured error
    error_handling: ErrorHandling

    #: The flag targets (if the error is an flag error)
    targets: Optional[TargetsType] = None

    def __post_init__(self):
        if not isinstance(self.error, ProcessingError):
            raise ValueError("Processing control result has to be an exception.")

    def get_targets(self) -> Iterable[EntityType]:
        """Get the targets of the flag."""
        if self.targets is None:
            raise ValueError("Missing flag targets.")
        if isinstance(self.targets, abc.Iterable):
            return self.targets
        return [self.targets]

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
            The level corresponding to the
            :class:`~dispel.processing.level.LevelProcessingControlResult`.

        Returns
        -------
        ProcessingControlResult
            The initialized processing control result object.

        Raises
        ------
        ValueError
            If the level is passed.
        ValueError
            If the caught exception does not contain a message ot if the message is
            inconsistent.
        """
        if level is not None:
            raise ValueError(
                "No level should be passed for ``ProcessingControlResult`` "
                "Initialization. If you would like to register the result in a level "
                "use ``LevelProcessingControlResult`` class."
            )
        # Retrieve error message
        try:
            exception_message = next(iter(error.args))
        except StopIteration:
            raise ValueError(
                f"Assertion {error=} must contain a message."
            ) from StopIteration

        # Retrieve error message components
        if isinstance(exception_message, str):
            error_handling = ErrorHandling.IGNORE
            returned_error = InvalidDataError(exception_message, step)
        elif isinstance(exception_message, abc.Iterable):
            if len((args := list(exception_message))) != 2:
                raise ValueError(
                    "If the provided exception assertion message is an iterable. It "
                    "ought to strictly contain two elements: a string error message "
                    f"and an ``ErrorHandling`` action amongst {list(ErrorHandling)}."
                )
            returned_error = InvalidDataError(args[0], step)
            error_handling = ErrorHandling(args[1])
        else:
            raise ValueError(
                "The exception message accompanying the Assertion error must either "
                "be a string message or an iterable containing  both a string message "
                f"and an ``ErrorHandling`` action amongst {list(ErrorHandling)}."
            )
        return cls(step=step, error=returned_error, error_handling=error_handling)

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
            The level corresponding to the
            :class:`~dispel.processing.level.LevelProcessingControlResult`.

        Returns
        -------
        ProcessingControlResult
            The initialized processing control result object.
        """
        assert level is None, "Level must not be provided."
        return cls(
            step=step,
            targets=targets,
            error=FlagError(flag, step),
            error_handling=ErrorHandling.from_bool(flag.stop_processing),
        )


ProcessResultType = Generator[
    Union[ProcessingResult, ProcessingControlResult], None, None
]


class Parameter(Generic[ParameterType]):
    """A parameter defining aspects of processing.

    Parameters
    ----------
    id_
        The name of the parameter used to identify the configurable entity.
    default_value
        A default value to be used if the user does not specify otherwise.
    validator
        A callable that accepts values and raises an exception for any
        unexpected value.
    description
        A description of the parameter explaining its usage and influence on
        the affected processing steps.
    """

    _registry: Dict[str, "Parameter"] = {}

    def __new__(cls, id_: str, *_args, **_kwargs):
        """Create a new Parameter object."""
        obj = super().__new__(cls)

        # inspect frame to obtain processing step defining parameter
        if (current_frame := inspect.currentframe()) is None:
            raise RuntimeError("Failed to inspect current frame")
        if (parent_frame := current_frame.f_back) is None:
            raise RuntimeError("Failed to inspect parent frame")
        location_spec = parent_frame.f_globals["__name__"]

        # check if we were defined inside a class
        if "__qualname__" in parent_frame.f_locals:
            location_spec += "." + parent_frame.f_locals["__qualname__"]

        full_id = f"{location_spec}.{id_}"
        if full_id in cls._registry:
            raise KeyError(f"Parameter already registered: {full_id}")

        cls._registry[full_id] = obj
        obj._full_id = full_id
        return obj

    def __init__(
        self,
        id_: str,
        default_value: Optional[ParameterType] = None,
        validator: Optional[Callable[[Any], None]] = None,
        description: Optional[str] = None,
    ):
        self._full_id: Optional[str]
        self._local_id = id_
        self._value = default_value
        self.default_value = default_value
        self.validator = validator
        self.description = description

    @property
    def id(self):
        """Get the ID of the parameter.

        The id will be set automatically based on the context of the
        creation of the parameter.

        Returns
        -------
        str
            The ID of the parameter. It is comprised by the name of the frame
            in which it was defined and the specified ``id_``.
        """
        return self._full_id

    @property
    def value(self) -> ParameterType:
        """Get the value set for the parameter."""
        if not self._value:
            raise RuntimeError(f"Parameter was not set: {self.id}")
        return self._value

    @value.setter
    def value(self, value: ParameterType):
        """Set the value for the parameter."""
        if self.validator is not None:
            self.validator(value)
        self._value = value

    @classmethod
    def has_parameter(cls, full_id: str) -> bool:
        """Check if a parameter was set."""
        return full_id in cls._registry

    @classmethod
    def set_value(cls, full_id: str, value: Any):
        """Set the value of a parameter."""
        if not cls.has_parameter(full_id):
            raise KeyError(f"Unknown parameter: {full_id}")
        cls._registry[full_id].value = value


class ProcessingStep:
    r"""A processing step in a processing sequence.

    :class:`ProcessingStep` is the basic entity through which
    :class:`~dispel.data.core.Reading`\ s are processed. The processing step's
    :meth:`process_reading` function is called with the reading and additional arguments
    passed to :func:`process`. Results from the process step are expected to be an
    instance of :class:`ProcessingResult`. For a comprehensive description see
    :ref:`feature-extraction`.

    The method :meth:`flag_reading` can be overwritten to ensure that the reading
    about to be processed is valid, and return
    :class:`~dispel.data.flags.Flag`\ s if that is not the case.

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

        >>> from dispel.data.features import FeatureValue
        >>> from dispel.data.values import ValueDefinition
        >>> from dispel.processing import process
        >>> from dispel.processing.core import ProcessingResult, ProcessingStep
        >>> class MyStep(ProcessingStep):
        ...     def process_reading(self, reading, **kwargs):
        ...         level = reading.get_level('my-level')
        ...         raw_data_set = level.get_raw_data_set('my-data-set')
        ...         data = raw_data_set.data
        ...         yield ProcessingResult(
        ...             step=self,
        ...             sources=raw_data_set,
        ...             result=FeatureValue(
        ...                 ValueDefinition('my-feature-id','max value'),
        ...                 data.max().max()
        ...             )
        ...         )
        >>> _ = process(reading, MyStep())
        >>> reading.feature_set.get_raw_value('my-feature-id')
        5
    """

    def __init__(self):
        self.predecessor = None
        self.successor = None

    def process(self, reading: Reading, **kwargs) -> ProcessResultType:
        """Check reading for validity and process it.

        Parameters
        ----------
        reading
            The reading to be processed
        kwargs
            Additional arguments passed by :func:`process`.

        Yields
        ------
        ProcessResultType
            The results from processing readings.
        """
        for flag in self.flag_reading(reading, **kwargs):
            yield ProcessingControlResult.from_flag(
                flag=flag,
                step=self,
                targets=self.get_reading_flag_targets(reading, **kwargs),
            )
        try:
            self.assert_valid_reading(reading, **kwargs)
        except AssertionError as error:
            yield ProcessingControlResult.from_assertion_error(step=self, error=error)
        else:
            yield from self.process_reading(reading, **kwargs)

    def assert_valid_reading(self, reading: Reading, **kwargs):
        """Assert that reading is valid."""

    def flag_reading(self, reading: Reading, **kwargs) -> Generator[Flag, None, None]:
        """Flag the provided reading.

        Parameters
        ----------
        reading
            The reading to be flagged.
        kwargs
            Additional arguments passed by :func:`~dispel.processing.process`.

        Yields
        ------
        Flag
            The resulted flags.
        """
        # pylint: disable=unused-argument
        yield from []

    def get_reading_flag_targets(
        self, reading: Reading, **kwargs
    ) -> Iterable[EntityType]:
        """Get the reading flag targets.

        Parameters
        ----------
        reading
            The reading that is concerned with flagging.
        kwargs
            Additional keyword arguments eventually used for flag targets
            extraction.

        Returns
        -------
        Iterable[EntityType]
            An iterable of entities that are flagged.
        """
        # pylint: disable=unused-argument
        return [reading]

    @abstractmethod
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
        yield NotImplemented

    def set_previous(self, step: "ProcessingStep"):
        """Set the previous step in a processing chain of this step."""
        if self.predecessor is not None:
            warnings.warn(
                "Changing predecessors can lead to side-effects. Previous predecessor "
                f"was {self.predecessor}",
                UserWarning,
            )
        self.predecessor = step

    def set_next(self, step: "ProcessingStep"):
        """Set the next step in a processing chain of this step."""
        if self.successor is not None:
            warnings.warn(
                "Changing successors can lead to side-effects. Previous successor was "
                f"{self.successor}",
                UserWarning,
            )
        self.successor = step

    def chain(self, successor: "ProcessingStep") -> "ProcessingStep":
        """Chain this step with the successor step."""
        assert isinstance(successor, ProcessingStep), "Can only chain processing steps"

        self.set_next(successor)
        successor.set_previous(self)
        return _ChainedProcesses([self, successor])

    def __and__(self, other):
        """See :meth:`ProcessingStep.chain`."""
        return self.chain(other)

    def get_parameters(self) -> List[Tuple[str, Parameter]]:
        """Get all parameters defined by the processing step.

        Returns
        -------
        List[Tuple[str, Parameter]]
            A list of tuples of parameter name and :class:`Parameter`
            objects defined by the processing step.
        """
        return inspect.getmembers(self, lambda x: isinstance(x, Parameter))


class CoreProcessingStepGroup(ProcessingStep):
    r"""A group of processing steps.

    The :class:`CoreProcessingStepGroup` allows to provide additional named arguments to
    the :meth:`~dispel.processing.process` function of the grouped
    :class:`~dispel.processing.core.ProcessingStep`\ s. The primary use case for this is to
    provide additional arguments to :class:`~dispel.processing.extract.ExtractStep` that
    use :class:`~dispel.data.values.ValueDefinitionPrototype`\ s.

    Parameters
    ----------
    steps
        The processing steps of the group
    kwargs
        Additional arguments that are passed to the
        :meth:`~dispel.processing.core.ProcessingStep.process` function of each step. This
        allows to provide additional values, such as placeholder values in value
        definitions to the actual processing function.

    Examples
    --------
    >>> from dispel.data.values import ValueDefinitionPrototype
    >>> from dispel.processing.core import CoreProcessingStepGroup
    >>> from dispel.processing.extract import ExtractStep
    >>> steps = [
    ...     CoreProcessingStepGroup(
    ...         [
    ...             ExtractStep(
    ...                 'data-set',
    ...                 lambda data_set: 5,
    ...                 ValueDefinitionPrototype(
    ...                     id_='x',
    ...                     feature_name='{placeholder} x'
    ...                 )
    ...             ),
    ...         ],
    ...         placeholder='A name'
    ...     ),
    ...     ...
    ... ]

    The above extract step will result in a feature value with the name ``A name x``.
    For further applications of :class:`CoreProcessingStepGroup` see also
    :ref:`feature-extraction`.
    """

    steps: List[ProcessingStep]

    kwargs: Dict[str, Any] = {}

    def __new__(cls, *args, **kwargs):
        """Instantiate a new ProcessingStepGroup."""
        instance = super().__new__(cls)
        if hasattr(cls, "steps") and cls.steps:
            instance.steps = copy.deepcopy(cls.steps)
        return instance

    def __init__(self, steps: Optional[List[ProcessingStep]] = None, **kwargs):
        self.set_steps(steps or self.get_steps())
        self.set_kwargs(**(kwargs or self.get_kwargs()))

        super().__init__()

    def set_kwargs(self, **kwargs):
        """Set the keyword arguments to be added to the processing."""
        self.kwargs = kwargs

    def get_kwargs(self) -> Dict[str, Any]:
        """Get keyword arguments to be added to the processing."""
        return self.kwargs

    def set_steps(self, steps: List[ProcessingStep]):
        """Set processing steps part of the group.

        Parameters
        ----------
        steps
            The steps contained in the processing group.
        """
        self.steps = steps

    def get_steps(self) -> List[ProcessingStep]:
        """Get processing steps within this group."""
        return self.steps

    def process_reading(self, reading: Reading, **kwargs) -> ProcessResultType:
        """See :meth:`dispel.processing.core.ProcessingStep.process`."""
        (updated_kwargs := kwargs.copy()).update(self.get_kwargs())

        for step in self.get_steps():
            yield from step.process(reading, **updated_kwargs)


class _ChainedProcesses(CoreProcessingStepGroup):
    """A technical class to chain process steps inside a group."""

    def chain(self, successor: "ProcessingStep") -> "ProcessingStep":
        """Add a processing step to the steps."""
        assert len(steps := self.get_steps()) > 0, "No steps to chain in group"

        last_step = steps[-1]
        last_step.set_next(successor)
        successor.set_previous(last_step)

        steps.append(successor)
        self.set_steps(steps)

        return self


class FlagReadingStep(FlagStepMixin, ProcessingStep):
    """A reading flag class.

    Parameters
    ----------
    task_name
        An optional abbreviated name value of the task used for the flag. See
        :class:`~dispel.processing.flags.FlagStepMixin`.
    flag_name
        An optional abbreviated name value of the considered flag.
        See :class:`~dispel.processing.flags.FlagStepMixin`.
    flag_type
        An optional flag type.
        See :class:`~dispel.data.flags.FlagType`.
    reason
        An optional string reason of the considered flag.
        See :class:`~dispel.processing.flags.FlagStepMixin`.
    stop_processing
        An optional boolean that specifies whether the flag is stop_processing,
        i.e., raises an error or not.
        See :class:`~dispel.processing.flags.FlagStepMixin`.
    flagging_function
        An optional flagging function applied to :class:`~dispel.data.core.Reading`.
        See :class:`~dispel.processing.flags.FlagStepMixin`.

    Examples
    --------
    Assuming you want to flag reading information such as whether the user has
    finished the evaluation properly, you can create the following flag step:

    >>> from dispel.data.values import AbbreviatedValue as AV
    >>> from dispel.processing.core import FlagReadingStep
    >>> step = FlagReadingStep(
    ...     task_name = AV('Pinch test', 'pinch'),
    ...     flag_name = AV('unfinished evaluation', 'ua'),
    ...     reason = 'The evaluation has not been finished by the user.',
    ...     stop_processing=False,
    ...     flag_type=FlagType.TECHNICAL,
    ...     flag_severity=FlagSeverity.INVALIDATION,
    ...     flagging_function=lambda reading: reading.evaluation.finished,
    ... )

    The flagging function will be called with the corresponding reading as argument.

    Another common scenario is to define a class that can be reused.

    >>> from dispel.data.flags import FlagType
    >>> from dispel.processing.core import FlagReadingStep
    >>> class UnfinishedEvaluation(FlagReadingStep):
    ...     task_name = AV('Pinch test', 'pinch')
    ...     flag_name = AV('unfinished evaluation', 'ua')
    ...     flag_type = FlagType.BEHAVIORAL
    ...     flag_severity = FlagSeverity.INVALIDATION
    ...     reason = 'The evaluation has not been finished by the user.'
    ...     stop_processing = False
    ...     flagging_function = lambda reading: reading.evaluation.finished

    Another convenient way to provide the flagging function is to use the
    ``@flag`` decorator:

    >>> from dispel.data.core import Reading
    >>> from dispel.processing.core import FlagReadingStep
    >>> from dispel.processing.flags import flag
    >>> class UnfinishedEvaluation(FlagReadingStep):
    ...     task_name = AV('Pinch test', 'pinch')
    ...     flag_name = AV('unfinished evaluation', 'ua')
    ...     flag_type = 'behavioral'
    ...     flag_severity = FlagSeverity.INVALIDATION
    ...     reason = 'The evaluation has not been finished by the user.'
    ...     stop_processing = False
    ...
    ...     @flag
    ...     def _unfinished_evaluation(self, reading: Reading) -> bool:
    ...         return reading.evaluation.finished

    Note that the ``@flag`` decorator can take keyword arguments. These kwargs are
    merged with any keyword arguments that come from processing step groups in order to
    format the flag ``reason``. Also, one can use multiple flag decorators
    in the same flag class.
    """

    def __init__(
        self,
        task_name: Optional[Union[AV, str]] = None,
        flag_name: Optional[Union[AV, str]] = None,
        flag_type: Optional[Union[FlagType, str]] = None,
        flag_severity: Optional[Union[FlagSeverity, str]] = None,
        reason: Optional[Union[AV, str]] = None,
        stop_processing: bool = False,
        flagging_function: Optional[Callable[..., bool]] = None,
    ):
        super().__init__(
            task_name=task_name,
            flag_name=flag_name,
            flag_type=flag_type,
            flag_severity=flag_severity,
            reason=reason,
            stop_processing=stop_processing,
            flagging_function=flagging_function,
        )

    def process_reading(self, reading: Reading, **kwargs) -> ProcessResultType:
        """Process the provided reading."""
        yield from []

    def get_reading_flag_targets(
        self, reading: Reading, **kwargs
    ) -> Iterable[EntityType]:
        """Get flag targets for reading flag."""
        return self.get_flag_targets(reading, **kwargs)

    def get_flag_targets(
        self, reading: Reading, level: Optional[Level] = None, **kwargs
    ) -> Iterable[EntityType]:
        """Get flag targets for reading flag."""
        return [reading]

    def flag_reading(self, reading: Reading, **kwargs) -> Generator[Flag, None, None]:
        """Flag the provided reading."""
        for func, func_kwargs in self.get_flagging_functions():
            if not func(reading, **kwargs):
                (merged_kwargs := kwargs.copy()).update(func_kwargs)
                yield self.get_flag(**merged_kwargs)
