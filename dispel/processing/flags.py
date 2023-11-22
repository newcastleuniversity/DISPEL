"""Data entity flag module."""
import inspect
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, Generator, Iterable, Optional, Tuple, Union

from dispel.data.core import EntityType, Reading
from dispel.data.flags import Flag, FlagId, FlagSeverity, FlagType
from dispel.data.levels import Level
from dispel.data.values import AbbreviatedValue as AV
from dispel.processing.utils import TaskMixin
from dispel.utils import set_attributes_from_kwargs


def flag(_func=None, **kwargs):
    """Decorate a function as a flagging function."""

    def wrapper(func):
        func.__flagging_function__ = True
        func.__flag_kwargs__ = {
            **kwargs,
            **getattr(func, "__flag_kwargs__", {}),
        }
        return func

    if _func is None:
        return wrapper

    return wrapper(_func)


FlaggingFunctionGeneratorType = Generator[
    Tuple[Callable[..., bool], Dict[str, Any]], None, None
]


class FlagStepMixin(TaskMixin, metaclass=ABCMeta):
    """A flag mix in class."""

    #: The name of the flag
    flag_name: Union[AV, str]

    #: The type of the flag
    flag_type: Union[FlagType, str]

    # The severity of the flag
    flag_severity: Union[FlagSeverity, str]

    #: The detailed reason of the flag
    reason: str

    #: The stop_processing status of the flag step
    stop_processing: bool = False

    #: The flagging function
    flagging_function: Optional[Callable[..., bool]] = None

    def __init__(self, *args, **kwargs):
        kwargs = set_attributes_from_kwargs(
            self,
            "task_name",
            "flag_name",
            "flag_type",
            "flag_severity",
            "reason",
            "stop_processing",
            "flagging_function",
            **kwargs,
        )

        self.kwargs = kwargs
        super().__init__(*args, **kwargs)

    def get_flag_name(self, **kwargs) -> Union[str, AV]:
        """Get the flag name."""
        flag_name = kwargs.get("flag_name", None) or getattr(self, "flag_name")
        if isinstance(flag_name, (str, AV)):
            return flag_name.format(**kwargs)
        raise ValueError("Missing flag name.")

    def get_flag_type(self, **kwargs) -> Union[str, FlagType]:
        """Get the flag type."""
        flag_type = kwargs.get("flag_type", None) or getattr(self, "flag_type")
        if isinstance(flag_type, (str, FlagType)):
            return flag_type
        raise ValueError("Missing flag type.")

    def get_flag_severity(self, **kwargs) -> Union[str, FlagSeverity]:
        """Get the flag severity."""
        flag_severity = kwargs.get("flag_severity", None) or getattr(
            self, "flag_severity"
        )
        if isinstance(flag_severity, (str, FlagSeverity)):
            return flag_severity
        raise ValueError("Missing flag severity.")

    def get_reason(self, **kwargs) -> str:
        """Get the flag reason."""
        reason = kwargs.get("reason", None) or getattr(self, "reason")
        if isinstance(reason, str):
            return reason.format(**kwargs)
        raise ValueError("Missing flag reason.")

    @abstractmethod
    def get_flag_targets(
        self, reading: Reading, level: Optional[Level] = None, **kwargs
    ) -> Iterable[EntityType]:
        """Get flag targets.

        Parameters
        ----------
        reading
            The reading to which the targets are associated.
        level
            The level associated with the targets (if needed).
        kwargs
            Keyword arguments from which the flag targets are to be extracted.

        Returns
        -------
        Iterable[EntityType]
            An iterable of the flag targets.
        """
        raise NotImplementedError

    def get_flagging_function(self) -> Optional[Callable[..., bool]]:
        """Get the flagging function."""
        # unbind bound methods
        func = self.flagging_function
        if func is not None and hasattr(func, "__func__"):
            return func.__func__  # type: ignore
        return func

    def get_flagging_functions(self) -> FlaggingFunctionGeneratorType:
        """Get all flagging functions associated with this step."""
        if func := self.get_flagging_function():
            yield func, {}

        members = inspect.getmembers(self, predicate=inspect.isroutine)
        for _, func in members:
            if func is not None and hasattr(func, "__flagging_function__"):
                yield func, func.__flag_kwargs__  # type: ignore

    def set_flag_kwargs(self, **kwargs):
        """Set keyword arguments inside flagging function.

        Parameters
        ----------
        kwargs
            The keyword arguments to be added inside the flagging function
            keyword arguments.
        """
        _, parent, *_ = inspect.stack()
        getattr(self, parent.function).__flag_kwargs__.update(kwargs)

    def get_flag(self, **kwargs) -> Flag:
        """Get the flag corresponding to the flag step."""
        (all_kwargs := self.kwargs.copy()).update(kwargs)
        return Flag(
            id_=FlagId(
                task_name=self.get_task_name(**all_kwargs),
                flag_name=self.get_flag_name(**all_kwargs),
                flag_type=self.get_flag_type(**all_kwargs),
                flag_severity=self.get_flag_severity(**all_kwargs),
            ),
            reason=self.get_reason(**all_kwargs),
            stop_processing=self.stop_processing,
        )
