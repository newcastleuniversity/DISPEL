"""A data model to describe epochs in time.

When processing signals, one of the fundamental concepts is to describe specific
aspects of a signal in a defined period. :class:`Epoch` provides the basic mechanics.
"""
from datetime import datetime
from typing import Any, Callable, Iterable, Optional, Union

import pandas as pd
from numpy import datetime64

from dispel.data.flags import FlagMixIn
from dispel.data.values import DefinitionId


class EpochDefinition:
    """The definition of an epoch.

    Parameters
    ----------
    id_
        The identifier of the epoch. This identifier does not have to be unique
        across multiple epochs and can serve as a type of epoch.
    name
        An optional plain-text name of the epoch definition.
    description
        A detailed description of the epoch providing additional resolution beyond
        the ``name`` property.

    Attributes
    ----------
    name
        An optional plain-text name of the epoch definition.
    description
        A detailed description of the epoch providing additional resolution beyond
        the ``name`` property.
    """

    def __init__(
        self,
        id_: Union[str, DefinitionId],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self.id = id_  # type: ignore
        self.name = name
        self.description = description

    @property
    def id(self) -> DefinitionId:
        """Get the ID of the definition.

        Returns
        -------
        DefinitionId
            The ID of the epoch definition.
        """
        return self._id

    @id.setter
    def id(self, value: Union[str, DefinitionId]):
        """Set the ID of the definition.

        Parameters
        ----------
        value
            The ID of the definition. The ID has to be unique with respect to the
            time points of the :class:`Epoch`, i.e., if an epoch has the same ID,
            start, and end, it is considered equal.
        """
        if not isinstance(value, DefinitionId):
            value = DefinitionId(value)
        self._id = value


class Epoch(FlagMixIn):
    """An epoch marking a specific time point or period.

    Parameters
    ----------
    start
        The beginning of the epoch.
    end
        An optional end of the epoch. If no end is provided, the epoch end will be
        considered in the future and the :data:`Epoch.is_incomplete` property will be
        `True`.
    definition
        An optional definition of the epoch.
    """

    def __init__(
        self,
        start: Any,
        end: Any,
        definition: Optional[EpochDefinition] = None,
    ):
        super().__init__()

        self.start = start
        self.end = end
        self.definition = definition

    @property
    def start(self) -> pd.Timestamp:
        """Get the beginning of the epoch.

        Returns
        -------
        pandas.Timestamp
            The beginning of the epoch.
        """
        return self._start

    @start.setter
    def start(self, value: Union[int, float, str, datetime, datetime64]):
        """Set the beginning of the epoch.

        Parameters
        ----------
        value
            The start of the epoch.

        Raises
        ------
        ValueError
            Risen if the provided value is null.
        """
        self._start = pd.Timestamp(value)
        if pd.isnull(self.start):
            raise ValueError("Start date cannot be null")

    @property
    def end(self) -> Optional[pd.Timestamp]:
        """Get the end of the epoch.

        Returns
        -------
        pandas.Timestamp
            The end of the epoch. `None`, if the epoch end has not been observed (i.e.,
            was not set).
        """
        return self._end

    @end.setter
    def end(self, value: Optional[Union[int, float, str, datetime, datetime64]]):
        """Set the end of the epoch.

        Parameters
        ----------
        value
            The end of the epoch. If `None` is provided, the epoch end is considered to
            be in the future and :data:`Epoch.is_incomplete` is ``True``.

        Raises
        ------
        ValueError
            If the `start` is after the `end`.
        """
        self._end = pd.Timestamp(value)
        if self.start > self.end:
            raise ValueError(f"Start cannot be after end: {self.start} > {self.end}")

    @property
    def id(self) -> DefinitionId:
        """Get the ID from the definition of the epoch.

        Returns
        -------
        DefinitionId
            The id of the :data:`Epoch.definition`.

        Raises
        ------
        AttributeError
            Will be risen if no definition was set for the epoch.
        """
        if self.definition is None:
            raise AttributeError("No definition was provided for epoch")
        return self.definition.id

    def __hash__(self):
        return hash((self.id, self.start, self.end))

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.start} - {self.end}>"

    def _test_overlap_contain(
        self,
        other: Union["Epoch", datetime, pd.Timestamp],
        method: Callable[[Iterable[bool]], bool],
    ) -> bool:
        if isinstance(other, Epoch):
            return method((self.overlaps(other.start), self.overlaps(other.end)))

        assert self.end is not None, "Can only test with closed epochs"
        if isinstance(other, (datetime, pd.Timestamp)):
            return self.start <= other <= self.end

        raise ValueError("Can only test for datetime or Epoch values")

    @property
    def duration(self) -> pd.Timedelta:
        """Get the duration of the epoch.

        Returns
        -------
        pandas.Timedelta
            The duration of the epoch.

        Raises
        ------
        ValueError
            If the epoch has no end.
        """
        if self.is_incomplete:
            raise ValueError("Cannot retrieve duration for incomplete epochs")
        return self.end - self.start

    @property
    def is_incomplete(self) -> bool:
        """Check if the epoch has an end date.

        An epoch is considered incomplete if it does not have an end date time.

        Returns
        -------
        bool
            `True` if the end date time is unknown. Otherwise, `False`.

        """
        return pd.isnull(self.end)

    def overlaps(self, other: Union["Epoch", datetime, pd.Timestamp]) -> bool:
        """Test if `other` overlaps with this epoch.

        Parameters
        ----------
        other
            The other epoch or datetime-like object to be tested.

        Returns
        -------
        bool
            If an epoch is provided ``overlap`` will be ``True`` if either the ``start``
            or ``end`` of the ``other`` epoch is within the ``start`` or ``end`` of
            this epoch. If only a datetime object is provided, the result is ``True`` if
            the time is between ``start`` and ``end`` including the boundaries.
        """
        return self._test_overlap_contain(other, any)

    def contains(self, other: Union["Epoch", datetime, pd.Timestamp]) -> bool:
        """Test if ``other`` is contained within this epoch.

        Parameters
        ----------
        other
            The other epoch or datetime-like object to be tested.

        Returns
        -------
        bool
            If an epoch is provided ``contains`` will be ``True`` if both the ``start``
            and ``end`` of the ``other`` epoch is within the ``start`` and ``end`` of
            this epoch. If only a datetime object is provided, the result is ``True`` if
            the time is between ``start`` and ``end`` including the boundaries.
        """
        return self._test_overlap_contain(other, all)
