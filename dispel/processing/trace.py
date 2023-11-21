"""Module to inspect definitions of processing steps."""
import warnings
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
)

import networkx as nx
from mypy_extensions import KwArg

from dispel.data.epochs import EpochDefinition
from dispel.data.raw import RawDataSetDefinition
from dispel.data.values import DefinitionIdType, ValueDefinition
from dispel.processing import ProcessingStepsType
from dispel.processing.core import CoreProcessingStepGroup, Parameter, ProcessingStep
from dispel.processing.data_set import MutateDataSetProcessingStepBase
from dispel.processing.epochs import (
    CreateLevelEpochStep,
    LevelEpochExtractStep,
    LevelEpochProcessingStepMixIn,
)
from dispel.processing.extract import (
    AggregateMeasures,
    ExtractStep,
    MeasureDefinitionMixin,
)
from dispel.processing.level import (
    DefaultLevelFilter,
    LevelFilter,
    LevelFilterProcessingStepMixin,
)
from dispel.processing.transform import TransformStep


class TraceType(Enum):
    """An enum to indicate type of trace from inspection."""

    ORIGIN = "origin"
    STEP_GENERIC = "step"
    STEP_GROUP = "step-group"
    DATA_SET = "data_set"
    MEASURE = "measure"
    EPOCH = "epoch"


class TraceRelation(Enum):
    """An enum to denote the relationship between nodes in edges."""

    GROUP = "group"
    SEQUENCE = "sequence"
    INPUT = "input"
    OUTPUT = "output"


@dataclass
class Trace(ABC):
    """A trace of a processing element from the inspection.

    Attributes
    ----------
    trace_type
        The type of the trace. This is one of :class:`TraceType` and set in
        the derived trace classes.
    """

    trace_type: TraceType = field(init=False)


TraceTypeT = TypeVar("TraceTypeT", bound=Trace)


@dataclass(unsafe_hash=True)
class OriginTrace(Trace):
    """The origin of inspection.

    This trace is the node to which all other nodes have a path and is added at
    the start of inspection.
    """

    trace_type = TraceType.ORIGIN


@dataclass(unsafe_hash=True)
class StepTrace(Trace):
    """The trace of a generic processing step.

    Attributes
    ----------
    step
        The traced processing step.
    """

    trace_type = TraceType.STEP_GENERIC
    step: ProcessingStep


@dataclass(unsafe_hash=True)
class StepGroupTrace(StepTrace):
    """The trace of a processing group step."""

    trace_type = TraceType.STEP_GROUP


@dataclass
class _LevelTraceBase(Trace):
    """A base class for traces of elements belonging to a level.

    Attributes
    ----------
    level
        The level filter relevant to the traced element.
    """

    level: LevelFilter


@dataclass
class DataSetTrace(_LevelTraceBase):
    """The trace of a data set.

    Attributes
    ----------
    data_set_id
        The id of the data set traced.
    definition
        The raw data set definition.
    """

    trace_type = TraceType.DATA_SET

    data_set_id: str
    definition: Optional[RawDataSetDefinition] = None

    def __hash__(self):
        # definition is excluded from the hash as it will not be set by
        # subsequent transform and extract steps.
        return hash((repr(self.level), self.data_set_id))


@dataclass
class MeasureTrace(_LevelTraceBase):
    """The trace of a measure.

    Attributes
    ----------
    measure
        The definition of the measure
    """

    trace_type = TraceType.MEASURE

    # FIXME: remove step -> inconsistent with DataTrace and EpochTrace, step is in edge!
    step: MeasureDefinitionMixin
    measure: ValueDefinition

    def __hash__(self):
        return hash((repr(self.level), self.measure.id))


@dataclass
class EpochTrace(_LevelTraceBase):
    """The trace of an epoch.

    Attributes
    ----------
    epoch
        The definition of the epoch
    """

    trace_type = TraceType.EPOCH

    epoch: EpochDefinition

    def __hash__(self):
        return hash((repr(self.level), self.epoch.id))


LevelTraceTypeT = TypeVar("LevelTraceTypeT", bound=_LevelTraceBase)


def _add_trace_node(graph: nx.MultiDiGraph, trace: Trace):
    graph.add_node(trace, trace_type=trace.trace_type, **trace.__dict__)


def _visit_processing_step_group(
    graph: nx.MultiDiGraph, group: CoreProcessingStepGroup, **kwargs
) -> Tuple[StepGroupTrace, StepTrace]:
    (updated_kwargs := kwargs.copy()).update(group.get_kwargs())
    _add_trace_node(graph, group_trace := StepGroupTrace(group))

    previous: StepTrace = group_trace
    for step in group.steps:
        trace = _visit_processing_step(graph, previous, step, **updated_kwargs)
        previous = trace

        graph.add_edge(
            group_trace,
            trace,
            type=TraceRelation.GROUP,
            group=group,
        )

    return group_trace, previous


def get_traces(
    graph: nx.MultiDiGraph, trace_type: Type[TraceTypeT]
) -> Iterable[TraceTypeT]:
    """Get all traces being an instance of a specific type.

    Parameters
    ----------
    graph
        The graph providing the nodes/traces
    trace_type
        The object type the trace has to be of.

    Returns
    -------
    Iterable[TraceTypeT]
        Returns an iterable of traces of type ``trace_type``.
    """
    return filter(lambda t: isinstance(t, trace_type), graph.nodes)


def _find_or_add_trace(graph: nx.MultiDiGraph, trace: _LevelTraceBase):
    if isinstance(trace, DataSetTrace):
        for candidate in get_traces(graph, DataSetTrace):
            if candidate.data_set_id == trace.data_set_id:
                return candidate

    _add_trace_node(graph, trace)
    return trace


def _get_level_filter(step: ProcessingStep) -> LevelFilter:
    if not isinstance(step, LevelFilterProcessingStepMixin):
        return DefaultLevelFilter()
    return step.get_level_filter()


def _find_or_add_input_data_set_traces(
    graph: nx.MultiDiGraph, step: MutateDataSetProcessingStepBase
) -> Sequence[DataSetTrace]:
    level_filter = _get_level_filter(step)
    trace_inputs = [
        _find_or_add_trace(graph, DataSetTrace(level_filter, ds_id))
        for ds_id in step.get_data_set_ids()
    ]
    return trace_inputs


def _add_transform_function_output_traces(
    graph: nx.MultiDiGraph,
    step: MutateDataSetProcessingStepBase,
    output_callback: Callable[[KwArg(Any)], LevelTraceTypeT],
    step_trace: StepTrace,
    input_traces: Sequence[_LevelTraceBase],
    **kwargs,
) -> Sequence[LevelTraceTypeT]:
    output_traces = []
    for _, func_kwargs in step.get_transform_functions():
        (updated_kwargs := kwargs.copy()).update(func_kwargs)
        output_trace = _find_or_add_trace(graph, output_callback(**updated_kwargs))
        output_traces.append(output_trace)
        graph.add_edge(step_trace, output_trace, type=TraceRelation.OUTPUT)

        for input_trace in input_traces:
            graph.add_edge(
                input_trace,
                output_trace,
                type=TraceRelation.INPUT,
                step=step,
            )

    return output_traces


def _visit_mutate_data_set_step(
    graph: nx.MultiDiGraph,
    step: MutateDataSetProcessingStepBase,
    output_callback: Callable[[KwArg(Any)], LevelTraceTypeT],
    **kwargs,
) -> Tuple[StepTrace, Sequence[_LevelTraceBase], Sequence[LevelTraceTypeT]]:
    _add_trace_node(graph, step_trace := StepTrace(step))

    input_traces = _find_or_add_input_data_set_traces(graph, step)
    for input_trace in input_traces:
        graph.add_edge(input_trace, step_trace, type=TraceRelation.INPUT)

    output_traces = _add_transform_function_output_traces(
        graph, step, output_callback, step_trace, input_traces, **kwargs
    )

    return step_trace, input_traces, output_traces


def _visit_transform_step(
    graph: nx.MultiDiGraph, step: TransformStep, **kwargs
) -> StepTrace:
    def _callback(**_kwargs) -> DataSetTrace:
        return DataSetTrace(
            _get_level_filter(step),
            step.get_new_data_set_id(),
            step.get_raw_data_set_definition(),
        )

    step_trace, *_ = _visit_mutate_data_set_step(graph, step, _callback, **kwargs)
    return step_trace


def _visit_extract_step(
    graph: nx.MultiDiGraph, step: ExtractStep, **kwargs
) -> StepTrace:
    def _callback(**_kwargs) -> MeasureTrace:
        return MeasureTrace(
            _get_level_filter(step),
            step,
            step.get_definition(**_kwargs),
        )

    step_trace, *_ = _visit_mutate_data_set_step(graph, step, _callback, **kwargs)
    return step_trace


def _create_measure_extract_traces(
    graph: nx.MultiDiGraph, step: MeasureDefinitionMixin, **kwargs
):
    assert isinstance(step, ProcessingStep), "step must inherit from ProcessingStep"

    _add_trace_node(graph, step_trace := StepTrace(step))
    _add_trace_node(
        graph,
        output_trace := MeasureTrace(
            _get_level_filter(step),
            step,
            step.get_definition(**kwargs),
        ),
    )
    graph.add_edge(step_trace, output_trace, type=TraceRelation.OUTPUT)
    return step_trace, output_trace


def _visit_aggregate_measures_step(
    graph: nx.MultiDiGraph, step: AggregateMeasures, **kwargs
) -> StepTrace:
    step_trace, output_trace = _create_measure_extract_traces(graph, step, **kwargs)

    measure_trace_dict: Dict[DefinitionIdType, MeasureTrace] = {
        t.measure.id: t for t in get_traces(graph, MeasureTrace)
    }
    for measure_id in step.get_measure_ids(**kwargs):
        if measure_id not in measure_trace_dict:
            warnings.warn(
                f"{measure_id} not observed in inspection. Will not create "
                f"edges to step and output measure.",
                UserWarning,
            )
            continue

        graph.add_edge(
            measure_trace_dict[measure_id], step_trace, type=TraceRelation.INPUT
        )
        graph.add_edge(
            measure_trace_dict[measure_id],
            output_trace,
            type=TraceRelation.INPUT,
            step=step,
        )

    return step_trace


def _visit_measure_definition_mixin_step(
    graph: nx.MultiDiGraph, step: MeasureDefinitionMixin, **kwargs
) -> StepTrace:
    step_trace, _ = _create_measure_extract_traces(graph, step, **kwargs)
    return step_trace


def _visit_processing_step_generic(
    graph: nx.MultiDiGraph, step: ProcessingStep, **_kwargs
) -> StepTrace:
    _add_trace_node(graph, step_trace := StepTrace(step))
    return step_trace


def _find_input_epoch_traces(
    graph: nx.MultiDiGraph, step: LevelEpochProcessingStepMixIn
) -> Sequence[EpochTrace]:
    epoch_filter = step.get_epoch_filter()

    # assuming mixin was done with step that has level-filtering capabilities.
    assert isinstance(step, LevelFilterProcessingStepMixin)
    level_filter = step.get_level_filter()

    epoch_traces = []
    epoch_traces_candidates = get_traces(graph, EpochTrace)
    for trace in epoch_traces_candidates:
        if trace.level != level_filter or not epoch_filter([trace.epoch]):
            continue
        epoch_traces.append(trace)

    return epoch_traces


def _visit_level_epoch_extract_step(
    graph: nx.MultiDiGraph, step: LevelEpochExtractStep, **kwargs
) -> StepTrace:
    def _callback(**_kwargs):
        return MeasureTrace(
            _get_level_filter(step),
            step,
            step.get_definition(**_kwargs),
        )

    step_trace, _, output_traces = _visit_mutate_data_set_step(
        graph, step, _callback, **kwargs
    )

    for epoch_trace in _find_input_epoch_traces(graph, step):
        graph.add_edge(epoch_trace, step_trace, type=TraceRelation.INPUT)

        for output_trace in output_traces:
            graph.add_edge(
                epoch_trace,
                output_trace,
                step=step,
                type=TraceRelation.INPUT,
            )

    return step_trace


def _visit_create_level_epochs_step(
    graph: nx.MultiDiGraph, step: CreateLevelEpochStep, **kwargs
) -> StepTrace:
    level_filter = _get_level_filter(step)

    def _callback(**_kwargs):
        return EpochTrace(
            level_filter,
            step.get_definition(**_kwargs),
        )

    step_trace, input_traces, output_traces = _visit_mutate_data_set_step(
        graph, step, _callback, **kwargs
    )

    # Add epoch data set if provided
    epoch_data_set_trace: Optional[DataSetTrace] = None
    if step.epoch_data_set_id:
        epoch_data_set_trace = _find_or_add_trace(
            graph, DataSetTrace(level_filter, step.epoch_data_set_id)
        )

        graph.add_edge(step_trace, epoch_data_set_trace, type=TraceRelation.OUTPUT)

        for input_trace in input_traces:
            graph.add_edge(
                input_trace,
                epoch_data_set_trace,
                step=step,
                type=TraceRelation.INPUT,
            )

    if isinstance(step, LevelEpochProcessingStepMixIn):
        for epoch_trace in _find_input_epoch_traces(graph, step):
            graph.add_edge(epoch_trace, step_trace, type=TraceRelation.INPUT)

            for output_trace in output_traces:
                graph.add_edge(
                    epoch_trace,
                    output_trace,
                    step=step,
                    type=TraceRelation.INPUT,
                )

            if epoch_data_set_trace:
                graph.add_edge(
                    epoch_trace,
                    epoch_data_set_trace,
                    step=step,
                    type=TraceRelation.INPUT,
                )

    return step_trace


def _visit_processing_step(
    graph: nx.MultiDiGraph, previous: Trace, step: ProcessingStep, **kwargs
) -> StepTrace:
    trace: StepTrace
    trace_ls: Optional[StepTrace] = None

    # Processing groups
    if isinstance(step, CoreProcessingStepGroup):
        trace, trace_ls = _visit_processing_step_group(graph, step, **kwargs)
    # Epoch specific transformation & extraction
    elif isinstance(step, LevelEpochExtractStep):
        trace = _visit_level_epoch_extract_step(graph, step, **kwargs)
    elif isinstance(step, CreateLevelEpochStep):
        trace = _visit_create_level_epochs_step(graph, step, **kwargs)
    # Extracts
    elif isinstance(step, ExtractStep):
        trace = _visit_extract_step(graph, step, **kwargs)
    elif isinstance(step, AggregateMeasures):
        trace = _visit_aggregate_measures_step(graph, step, **kwargs)
    elif isinstance(step, MeasureDefinitionMixin):
        trace = _visit_measure_definition_mixin_step(graph, step, **kwargs)
    # Transformations
    elif isinstance(step, TransformStep):
        trace = _visit_transform_step(graph, step, **kwargs)
    # Generics
    else:
        trace = _visit_processing_step_generic(graph, step)
    graph.add_edge(previous, trace, type=TraceRelation.SEQUENCE)

    # return the last step in the group if visited step is a group
    if isinstance(step, CoreProcessingStepGroup):
        assert trace_ls
        return trace_ls

    return trace


def inspect(steps: ProcessingStepsType, **kwargs) -> nx.MultiDiGraph:
    """Inspect the relationships defined in processing steps.

    Parameters
    ----------
    steps
        The processing steps to be inspected.
    kwargs
        Any additional processing arguments typically passed to the
        processing steps.

    Returns
    -------
    networkx.MultiDiGraph
        A graph representing the processing elements found in `steps`.
    """
    graph = nx.MultiDiGraph()

    if isinstance(steps, type):
        # try to instantiate steps if class-referenced
        steps = steps()

    if isinstance(steps, ProcessingStep):
        steps = [steps]

    _add_trace_node(graph, origin := OriginTrace())

    previous: Trace = origin
    for step in steps:
        trace = _visit_processing_step(graph, previous, step, **kwargs)
        previous = trace

    return graph


def get_ancestors(
    graph: nx.MultiDiGraph,
    trace: Trace,
    rel_filter: Optional[Callable[[nx.MultiDiGraph, Trace, Trace], bool]] = None,
) -> Set[Trace]:
    """Get all ancestors of a trace.

    Parameters
    ----------
    graph
        The graph containing the relationship between the traces derived
        from :func:`inspect`.
    trace
        The trace for which to find all ancestors, i.e., all predecessors of
        all predecessors until the origin.
    rel_filter
        If provided, predecessors will only be considered if the return
        value is `True`. The function needs to accept the graph, the current
        trace, and the potential predecessor to be considered as ancestor.

    Returns
    -------
    Set[Trace]
        A set of traces of all predecessors to `trace`.
    """
    ancestors = set()

    def _collect(t):
        for predecessor in graph.predecessors(t):
            if predecessor not in ancestors:
                if rel_filter and not rel_filter(graph, t, predecessor):
                    continue
                ancestors.add(predecessor)
                _collect(predecessor)

    _collect(trace)
    return ancestors


def _rel_filter_sources(
    _graph: nx.MultiDiGraph, _trace: Trace, predecessor: Trace
) -> bool:
    return isinstance(predecessor, (DataSetTrace, MeasureTrace))


def get_ancestor_source_graph(
    graph: nx.MultiDiGraph, trace: MeasureTrace
) -> nx.MultiDiGraph:
    """Get a subgraph of all sources leading to the extraction of a measure.

    Parameters
    ----------
    graph
        The graph containing all traces.
    trace
        A trace of a measure for which to return the source graph.

    Returns
    -------
    networkx.MultiDiGraph
        A subgraph of `graph` comprised of nodes and edges only from data
        sets and measures leading to the extraction of the provided measure
        through `trace`.
    """
    return nx.subgraph(
        graph, get_ancestors(graph, trace, _rel_filter_sources) | {trace}
    )


def get_edge_parameters(graph: nx.MultiDiGraph) -> Dict[ProcessingStep, Set[Parameter]]:
    """Get parameters defined in steps attributes of edges."""
    parameters = defaultdict(set)
    for *_, step in graph.edges.data("step"):
        if step not in parameters and (step_param := step.get_parameters()):
            parameters[step].update(step_param)

    return parameters


def collect_measure_value_definitions(
    steps: Iterable[ProcessingStep], **kwargs
) -> Iterator[Tuple[MeasureDefinitionMixin, ValueDefinition]]:
    """Collect all measure value definitions from a list of processing steps.

    Parameters
    ----------
    steps
        The steps from which to collect the measure value definitions.
    kwargs
        See :func:`inspect`.

    Yields
    ------
    Tuple[ProcessingStep, ValueDefinition]
        The processing step in question along with the value definition.
    """
    graph = inspect(steps, **kwargs)
    for trace in get_traces(graph, MeasureTrace):
        # FIXME: infer step from input nodes
        yield trace.step, trace.measure
