"""A module containing the data trace graph (DTG).

The data trace is a directed acyclic graph that traces the creation of the main data
entities:

    - :class:`~dispel.data.core.Reading`,
    - :class:`~dispel.data.core.Level`,
    - :class:`~dispel.data.raw.RawDataSet`,
    - :class:`~dispel.data.measures.MeasureValue`.

It links the creation of each entity with its creators.
"""
import warnings
from dataclasses import field
from functools import singledispatchmethod
from itertools import chain
from typing import Iterable, List, Optional, Union, cast

import networkx as nx
from networkx.classes.coreviews import MultiAdjacencyView
from networkx.classes.reportviews import NodeView

from dispel.data.core import EntityType, Reading
from dispel.data.flags import Flag, FlagMixIn, verify_flag_uniqueness
from dispel.data.levels import Level, LevelEpoch
from dispel.data.measures import MeasureValue
from dispel.data.raw import RawDataSet
from dispel.processing.core import (
    FlagError,
    ProcessingControlResult,
    ProcessingResult,
    ProcessingStep,
)
from dispel.processing.epochs import CreateLevelEpochStep
from dispel.processing.extract import ExtractStep
from dispel.processing.level import LevelProcessingControlResult
from dispel.processing.transform import ConcatenateLevels, TransformStep
from dispel.utils import multidispatchmethod, plural


class NodeNotFound(Exception):
    """Exception raised when a node is not found in the data trace.

    Parameters
    ----------
    entity
        The entity corresponding to the exception.
    """

    def __init__(self, entity: EntityType):
        super().__init__(
            f"No node corresponding to {entity=} was found in the data trace."
        )


class EdgeNotFound(Exception):
    """Exception raised when an edge is not found in the data trace.

    Parameters
    ----------
    source
        The entity corresponding to the source node of the edge in question.
    result
        The entity corresponding to the result node of the edge in question.
    """

    def __init__(self, source: EntityType, result: EntityType):
        super().__init__(
            f"No edge corresponding to the nodes {source=} and {result=} in the data "
            "trace graph."
        )


class MissingSourceNode(Exception):
    """Exception raised if the source node of a data trace is missing.

    Parameters
    ----------
    entity
        The entity corresponding to the absent source node.
    """

    def __init__(self, entity: EntityType):
        super().__init__(
            f"No source node corresponding to {entity=} was found in the data trace "
            f"graph."
        )


class NodeHasMultipleParents(Exception):
    """Exception raised if a node is to have multiple parents.

    Parameters
    ----------
    entity
        The entity corresponding to the exception.
    """

    def __init__(self, entity: EntityType):
        super().__init__(
            f"The result node corresponding to {entity=} already exists in the data "
            "trace and thus cannot have multiple parents."
        )


class GraphNoCycle(Exception):
    """Exception raised if a cycle is to be created in the data trace graph.

    Parameters
    ----------
    source
        The source node that has caused the graph cycle.
    result
        The result node that has caused the graph cycle.
    """

    def __init__(self, source: EntityType, result: EntityType):
        super().__init__(
            f"A data trace cycle has been detected. The {source=} is already a "
            f"predecessor of {result=}."
        )


class DataTrace:
    """A class representation of the data trace graph."""

    def __init__(self):
        self._graph = nx.MultiDiGraph()
        self._processing_step_count = 0
        self.root: EntityType = field(init=False)

    def __repr__(self):
        return (
            f"<DataTrace of {self.get_reading()}: ("
            f'{plural("entity", self.entity_count, "entities")}, '
            f'{plural("processing step", self.processing_step_count)})>'
        )

    @singledispatchmethod
    def populate(
        self,
        entity: Union[
            Level,
            Reading,
            ProcessingResult,
            ProcessingControlResult,
            LevelProcessingControlResult,
        ],
    ) -> Optional[Exception]:
        """Populate the data trace with the provided entity.

        If the provided entity is a :class:`~dispel.data.core.Reading` or a
        :class:`~dispel.data.core.Level`, then the entity and its components are injected
        as data traces.

        If the provided entities are processing results, then the associated results are
        added as data traces and also set inside the corresponding reading.

        Parameters
        ----------
        entity
            The entity to injected inside the data trace.

        Returns
        -------
        Optional[List[Exception]]
            If existent, the list of exceptions to be raised at the end of the
            processing.

        Raises
        ------
        NotImplementedError
            If the provided entity type is not supported.
        """
        raise NotImplementedError(
            f"Unsupported type for data trace population: {type(entity)}"
        )

    @populate.register(Reading)
    def _reading(self, entity: Reading):
        """Populate data trace with newly created reading."""
        for measure in entity.measure_set.values():
            self.add_trace(entity, cast(MeasureValue, measure))

        for level in entity.levels:
            self.add_trace(entity, level)
            self.populate(level)

    @populate.register(Level)
    def _level(self, entity: Level):
        """Populate data trace with newly created level."""
        for raw_data_set in entity.raw_data_sets:
            self.add_trace(entity, raw_data_set)

        for measure in entity.measure_set.values():
            self.add_trace(entity, cast(MeasureValue, measure))

    @populate.register(ProcessingResult)
    def _processing_result(self, entity: ProcessingResult):
        """Populate data trace with processing result."""
        # Setting result entity in reading
        self.get_reading().set(entity.result, **entity.get_kwargs())

        # Setting data traces in data trace graph
        for source in entity.get_sources():
            self.add_trace(source, entity.result, step=entity.step)

        # Populate the data trace with the result if needed
        if isinstance(entity.result, Level):
            self.populate(entity.result)

        self._processing_step_count += 1

    @populate.register(ProcessingControlResult)
    def _processing_control_result(
        self, entity: ProcessingControlResult
    ) -> Optional[Exception]:
        """Handle the processing control results."""
        if isinstance(entity.error, FlagError):
            for target in entity.get_targets():
                target.add_flag(entity.error.flag)
        if entity.error_handling.should_raise:
            return entity.error
        return None

    @classmethod
    def from_reading(cls, reading: Reading) -> "DataTrace":
        """Initialise a data trace graph from a reading.

        Parameters
        ----------
        reading
            The reading associated to the data trace.

        Returns
        -------
        DataTrace
            The initialised data trace graph.
        """
        data_trace = cls()
        data_trace.root = reading
        data_trace._graph.add_node(reading)
        data_trace.populate(reading)
        return data_trace

    def get_reading(self) -> Reading:
        """Get the reading associated with the data trace graph."""
        if isinstance(entity := self.root, Reading):
            return entity
        raise ValueError(f"The root node entity is not a reading but a {type(entity)}")

    def nodes(self) -> NodeView:
        """Retrieve all the nodes inside the data trace graph.

        Returns
        -------
        NodeView
            The list of nodes inside the data trace graph.
        """
        return self._graph.nodes

    def is_leaf(self, entity: EntityType) -> bool:
        """Determine whether the entity is a leaf."""
        return self._graph.out_degree(entity) == 0

    def leaves(self) -> Iterable[EntityType]:
        """Retrieve all leaves of the data trace graph.

        Returns
        -------
        Iterable[EntityType]
            The leaf nodes of the data trace graph.
        """
        return filter(self.is_leaf, self._graph.nodes)

    @property
    def entity_count(self) -> int:
        """Get the entity count in the data trace graph.

        Returns
        -------
        int
            The number of entities in the data trace graph.
        """
        return self._graph.number_of_nodes()

    @property
    def leaf_count(self) -> int:
        """Get the leaf count of the data trace graph.

        Returns
        -------
        int
            The number of leaf nodes in the data trace graph.
        """
        return len(list(self.leaves()))

    def has_node(self, entity: EntityType) -> bool:
        """Check whether a node exists in the data trace graph.

        Parameters
        ----------
        entity
            The entity of the node whose existence in the data trace graph is to be
            checked.

        Returns
        -------
        bool
            ``True`` if the corresponding node exists inside the graph. ``False``
            otherwise.
        """
        return self._graph.has_node(entity)

    def parents(self, entity: EntityType) -> MultiAdjacencyView:
        """Get direct predecessors of an entity."""
        return self._graph.pred[entity]

    def children(self, entity: EntityType) -> MultiAdjacencyView:
        """Get direct successors of an entity."""
        return self._graph.succ[entity]

    @multidispatchmethod
    def add_trace(self, source: EntityType, result: EntityType, step=None, **kwargs):
        """Add a single data trace to the graph.

        Parameters
        ----------
        source
            The source entity that led to the creation of the result entity.
        result
            The result entity created from the source entity.
        step
            The processing step that has led to the creation of the result if
            existent.
        kwargs
            Additional keywords arguments passed to downstream dispatchers.

        Raises
        ------
        NotImplementedError
            If the source and/or result types are not supported.
        """
        raise NotImplementedError("The two node types are not supported.")

    def _add_trace(
        self,
        source: EntityType,
        result: EntityType,
        step: Optional[ProcessingStep] = None,
        allow_multiple_parents: bool = True,
    ):
        """Add an edge to the data trace graph."""
        if not self._graph.has_node(source):
            raise MissingSourceNode(source)

        if had_result := self._graph.has_node(result):
            if not allow_multiple_parents:
                raise NodeHasMultipleParents(result)

        if step:
            self._graph.add_edge(source, result, key=hash(step), step=step)
        else:
            self._graph.add_edge(source, result)

        if had_result:
            try:
                nx.algorithms.find_cycle(self._graph, result)
                raise GraphNoCycle(source, result)
            except nx.NetworkXNoCycle:
                pass

    @add_trace.register(Reading, Level)
    def _reading_level(self, source: Reading, result: Level):
        self._add_trace(source, result, allow_multiple_parents=False)

    @add_trace.register(Reading, MeasureValue)
    def _reading_measure(self, source: Reading, result: MeasureValue):
        self._add_trace(source, result, allow_multiple_parents=False)

    @add_trace.register(Level, RawDataSet)
    def _level_container(self, source: Level, result: RawDataSet):
        self._add_trace(source, result, allow_multiple_parents=False)

    @add_trace.register(Level, MeasureValue)
    def _level_measure(
        self, source: Level, result: MeasureValue, step: Optional[ProcessingStep] = None
    ):
        self._add_trace(source, result, step, allow_multiple_parents=False)

    @add_trace.register(Level, LevelEpoch)
    def _level_epoch(self, source: Level, result: LevelEpoch, step: ProcessingStep):
        self._add_trace(source, result, step, allow_multiple_parents=False)

    @add_trace.register(MeasureValue, Level)
    def _measure_level(self, source: MeasureValue, result: Level, step: ProcessingStep):
        self._add_trace(source, result, step)

    @add_trace.register(RawDataSet, RawDataSet)
    def _transform(self, source: RawDataSet, result: RawDataSet, step: TransformStep):
        self._add_trace(source, result, step)

    @add_trace.register(RawDataSet, MeasureValue)
    def _extract(self, source: RawDataSet, result: MeasureValue, step: ExtractStep):
        self._add_trace(source, result, step)

    @add_trace.register(RawDataSet, LevelEpoch)
    def _create_level_epoch(
        self, source: RawDataSet, result: LevelEpoch, step: CreateLevelEpochStep
    ):
        self._add_trace(source, result, step)

    @add_trace.register(LevelEpoch, MeasureValue)
    def _extract_level_epoch_measure(
        self, source: LevelEpoch, result: MeasureValue, step: ExtractStep
    ):
        self._add_trace(source, result, step)

    @add_trace.register(RawDataSet, Level)
    def _concatenate(self, source: RawDataSet, result: Level, step: ConcatenateLevels):
        self._add_trace(source, result, step)

    @add_trace.register(MeasureValue, MeasureValue)
    def _aggregate(
        self, source: MeasureValue, result: MeasureValue, step: ProcessingStep
    ):
        self._add_trace(source, result, step)

    @property
    def processing_step_count(self) -> int:
        """Get the processing steps count in the data trace graph.

        Returns
        -------
        int
            The number of different processing steps in the data trace graph.
        """
        return self._processing_step_count

    def check_data_set_usage(self) -> List[RawDataSet]:
        r"""Check the usage of the raw data sets inside the data trace graph.

        It means checking if all leaf nodes are :class:`dispel.data.measures.MeasureValue`
        and raise warnings if any are found.

        Returns
        -------
        List[RawDataSet]
            The list of unused data sets.

        Warns
        -----
        UserWarning
            If leaf entities do not correspond to measure values.

        Raises
        ------
        ValueError
            If a leaf node if neither a measure value nor a raw data set.
        """
        unused_data_sets: List[RawDataSet] = []
        for node in self.leaves():
            if isinstance(node, RawDataSet):
                unused_data_sets.append(node)
                warnings.warn(
                    "No measure has been created from the following entity " f"{node=}",
                    UserWarning,
                )
            elif not isinstance(node, MeasureValue):
                raise ValueError(
                    "The created leaf nodes should either be raw data sets or measure "
                    f"values. Found {type(node)}"
                )

        return unused_data_sets

    def get_flags(self, entity: EntityType) -> List[Flag]:
        """Get all flags related to the entity.

        Parameters
        ----------
        entity
            The entity whose related flags are to be retrieved.

        Returns
        -------
        List[Flag]
            A list of all flags related to the entity.
        """
        flags: List[Flag] = []
        if isinstance(entity, FlagMixIn):
            flags += entity.get_flags()

        # collect all flags from its ancestors
        for node in nx.algorithms.ancestors(self._graph, entity):
            if isinstance(node, FlagMixIn):
                flags += node.get_flags()

        verify_flag_uniqueness(flags)
        return flags

    def get_measure_flags(self, entity: Optional[EntityType] = None) -> List[Flag]:
        """Get all measure flags related to the entity.

        If no entity is provided, the default entity is the reading and every measure
        flag coming from the reading will be returned.

        Parameters
        ----------
        entity
            An optional entity whose related measure flags are to be retrieved,
            if ``None`` is provided the default entity is the root of the data trace.

        Returns
        -------
        List[Flag]
            A list of all measure flags related to the entity. If no entity is
            provided, all measure flags are returned.
        """
        entity = self.root if entity is None else entity
        measure_flags: List[Flag] = []
        nodes = chain(
            nx.algorithms.ancestors(self._graph, entity),
            [entity],
            nx.algorithms.descendants(self._graph, entity),
        )

        for node in nodes:
            if isinstance(node, MeasureValue):
                measure_flags += node.get_flags()

        verify_flag_uniqueness(measure_flags)
        return measure_flags
