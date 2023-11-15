"""Tests for :mod:`dispel.processing.trace`."""
import networkx as nx
import pytest

from dispel.data.core import Reading
from dispel.data.epochs import EpochDefinition
from dispel.data.features import FeatureValueDefinition, FeatureValueDefinitionPrototype
from dispel.data.raw import RawDataValueDefinition
from dispel.data.values import AbbreviatedValue as AV
from dispel.processing.core import (
    CoreProcessingStepGroup,
    Parameter,
    ProcessingStep,
    ProcessResultType,
)
from dispel.processing.data_set import transformation
from dispel.processing.epochs import (
    CreateLevelEpochStep,
    LevelEpochExtractStep,
    LevelEpochIdFilter,
    LevelEpochProcessingStepMixIn,
)
from dispel.processing.extract import (
    AggregateFeatures,
    ExtractStep,
    FeatureDefinitionMixin,
)
from dispel.processing.trace import (
    DataSetTrace,
    EpochTrace,
    FeatureTrace,
    OriginTrace,
    StepGroupTrace,
    StepTrace,
    Trace,
    TraceRelation,
    get_ancestor_source_graph,
    get_ancestors,
    get_edge_parameters,
    get_traces,
    inspect,
)
from dispel.processing.transform import TransformStep


class _TestProcessingStep(ProcessingStep):
    def process_reading(self, reading: Reading, **kwargs) -> ProcessResultType:
        pass


class _TestTransformStep(TransformStep):
    data_set_ids = ["input_dataset1", "input_dataset2"]
    new_data_set_id = "output_dataset"
    definitions = [
        RawDataValueDefinition("col1_id", "col1_name"),
        RawDataValueDefinition("col2_id", "col2_name"),
    ]

    @staticmethod
    @transformation
    def _transform(data):
        pass


class _TestExtractStep(ExtractStep):
    data_set_ids = ["input_dataset1", "input_dataset2"]
    definition = FeatureValueDefinitionPrototype(
        task_name=AV("test", "t"),
        feature_name=AV("feature {transform_name}", "feat-{transform_id}"),
        description="Description with {processing_placeholder} {transform_id}",
    )

    @staticmethod
    @transformation(transform_id="1", transform_name="one")
    def _transform1(data):
        pass

    @staticmethod
    @transformation(transform_id="2", transform_name="two")
    def _transform2(data):
        pass


def test_inspect_add_origin():
    """Test if the inspection adds a node for the origin."""
    graph = inspect([])
    assert graph.number_of_nodes() == 1

    node, *_ = list(graph.nodes())
    assert isinstance(node, OriginTrace)


def assert_graph_shape(
    graph: nx.MultiDiGraph, n_expected_nodes: int, n_expected_edges: int
):
    """Assert that a graph has a given number of nodes and edges."""
    assert graph.number_of_nodes() == n_expected_nodes
    assert graph.number_of_edges() == n_expected_edges


def assert_edge_relation(
    graph: nx.MultiDiGraph, source: Trace, target: Trace, relation: TraceRelation
):
    """Assert relationship type between two traces."""
    assert graph.has_edge(source, target)
    edata = graph.get_edge_data(source, target)
    assert len(edata) == 1
    assert edata[0]["type"] == relation


def test_inspect_processing_step_or_list():
    """Test if inspection works with lists of steps and individual steps."""
    # inspection
    graph1 = inspect(_TestProcessingStep())
    assert_graph_shape(graph1, 2, 1)

    graph2 = inspect([_TestProcessingStep()])
    assert_graph_shape(graph2, 2, 1)


def test_inspect_generic_processing_step():
    """Test tracing generic processing steps."""
    step = _TestProcessingStep()
    graph = inspect(step)
    assert_graph_shape(graph, 2, 1)

    origin, step_trace, *_ = graph.nodes()
    assert isinstance(origin, OriginTrace)
    assert isinstance(step_trace, StepTrace)
    assert step_trace.step is step
    assert_edge_relation(graph, origin, step_trace, TraceRelation.SEQUENCE)


def test_inspect_step_group():
    """Test tracing processing groups."""
    step1 = _TestProcessingStep()
    step2 = _TestProcessingStep()
    group = CoreProcessingStepGroup([step1, step2])

    graph = inspect(group)
    assert_graph_shape(graph, 4, 5)

    origin, group_trace, step1_trace, step2_trace, *_ = graph.nodes()
    assert isinstance(origin, OriginTrace)
    assert isinstance(group_trace, StepGroupTrace)
    assert isinstance(step1_trace, StepTrace)
    assert isinstance(step2_trace, StepTrace)
    assert group_trace.step is group
    assert step1_trace.step is step1
    assert step2_trace.step is step2
    assert graph.has_edge(group_trace, step1_trace)
    assert graph.has_edge(group_trace, step2_trace)
    assert graph.has_edge(step1_trace, step2_trace)

    edata1 = graph.get_edge_data(group_trace, step1_trace)
    assert len(edata1) == 2
    assert edata1[0]["type"] == TraceRelation.SEQUENCE
    assert edata1[1]["type"] == TraceRelation.GROUP
    assert edata1[1]["group"] is group

    assert_edge_relation(graph, step1_trace, step2_trace, TraceRelation.SEQUENCE)


def test_inspect_step_group_adjacency():
    """Test adjacency relationship between two groups and subsequent steps."""
    step1 = _TestProcessingStep()
    step2 = _TestProcessingStep()

    graph = inspect(
        [
            CoreProcessingStepGroup([step1]),
            step2,
        ]
    )

    assert_graph_shape(graph, 4, 4)

    # test network layout
    origin, g1_trace, step1_trace, step2_trace, *_ = graph.nodes()
    assert isinstance(origin, OriginTrace)
    assert isinstance(g1_trace, StepGroupTrace)
    assert isinstance(step1_trace, StepTrace)
    assert isinstance(step2_trace, StepTrace)

    assert_edge_relation(graph, step1_trace, step2_trace, TraceRelation.SEQUENCE)


def test_inspect_transform_step():
    """Test inspection of transformation steps."""
    step = _TestTransformStep()

    graph = inspect(step)

    assert_graph_shape(graph, 5, 6)

    origin, step_trace, input1_trace, input2_trace, output_trace, *_ = graph.nodes()

    # test network layout
    assert isinstance(origin, OriginTrace)
    assert isinstance(step_trace, StepTrace)
    assert isinstance(input1_trace, DataSetTrace)
    assert isinstance(input2_trace, DataSetTrace)
    assert isinstance(output_trace, DataSetTrace)
    assert step_trace.step is step
    assert graph.has_edge(origin, step_trace)
    assert graph.has_edge(input1_trace, step_trace)
    assert graph.has_edge(input2_trace, step_trace)
    assert graph.has_edge(step_trace, output_trace)

    # test network content
    assert input1_trace.data_set_id == "input_dataset1"
    assert input2_trace.data_set_id == "input_dataset2"
    assert output_trace.data_set_id == step.new_data_set_id

    assert_edge_relation(graph, input1_trace, step_trace, TraceRelation.INPUT)
    assert_edge_relation(graph, step_trace, output_trace, TraceRelation.OUTPUT)
    assert_edge_relation(graph, input2_trace, output_trace, TraceRelation.INPUT)


def test_chained_mutate_steps():
    """Test re-use of data set traces in chained mutation steps."""
    step1 = TransformStep(
        data_set_ids="input1",
        definitions=[RawDataValueDefinition("col-id", "col name")],
        transform_function=lambda data: None,
        new_data_set_id="output1",
    )
    step2 = TransformStep(
        data_set_ids="output1",
        definitions=[RawDataValueDefinition("col-id", "col name")],
        transform_function=lambda data: None,
        new_data_set_id="output2",
    )

    graph = inspect([step1, step2])

    assert_graph_shape(graph, 6, 8)


def test_inspect_extract_step():
    """Test inspection of extract steps."""
    step = _TestExtractStep()

    graph = inspect(step, processing_placeholder="injection_test")

    assert_graph_shape(graph, 6, 9)

    (
        origin,
        step_trace,
        input1_trace,
        input2_trace,
        f1_trace,
        f2_trace,
        *_,
    ) = graph.nodes()

    # test network layout
    assert isinstance(origin, OriginTrace)
    assert isinstance(step_trace, StepTrace)
    assert isinstance(input1_trace, DataSetTrace)
    assert isinstance(input2_trace, DataSetTrace)
    assert isinstance(f1_trace, FeatureTrace)
    assert isinstance(f2_trace, FeatureTrace)
    assert step_trace.step is step
    assert graph.has_edge(origin, step_trace)
    assert graph.has_edge(input1_trace, step_trace)
    assert graph.has_edge(input2_trace, step_trace)
    assert graph.has_edge(step_trace, f1_trace)
    assert graph.has_edge(step_trace, f2_trace)
    assert graph.has_edge(input1_trace, f1_trace)
    assert graph.has_edge(input2_trace, f1_trace)
    assert graph.has_edge(input1_trace, f2_trace)
    assert graph.has_edge(input2_trace, f2_trace)

    # test network content
    assert f1_trace.feature.id == "t-feat-1"
    assert f2_trace.feature.id == "t-feat-2"
    assert f1_trace.feature.description == "Description with injection_test 1"


def assert_input_output_relation(
    graph: nx.MultiDiGraph,
    input_trace: Trace,
    output_trace: Trace,
    step: ProcessingStep,
):
    """Assert the relationship between input and output produced by a step."""
    assert graph.has_edge(input_trace, output_trace)
    edata = graph.get_edge_data(input_trace, output_trace)
    assert len(edata) == 1
    assert edata[0]["type"] == TraceRelation.INPUT
    assert edata[0]["step"] == step


def test_inspect_aggregate_features_step():
    """Test steps aggregating features."""
    step1 = _TestExtractStep()
    step2 = AggregateFeatures(
        FeatureValueDefinition(AV("test", "t"), AV("feature", "feat")),
        ["t-feat-1", "t-feat-2"],
        lambda x: 0,
    )

    graph = inspect([step1, step2], processing_placeholder="injection_test")

    assert_graph_shape(graph, 8, 15)

    (
        _,
        step1_trace,
        _,
        _,
        feat1_trace,
        feat2_trace,
        step2_trace,
        feat_trace,
        *_,
    ) = graph.nodes()

    # partial graph layout check for second step
    assert isinstance(step1_trace, StepTrace)
    assert isinstance(step2_trace, StepTrace)
    assert isinstance(feat1_trace, FeatureTrace)
    assert isinstance(feat2_trace, FeatureTrace)
    assert isinstance(feat_trace, FeatureTrace)
    assert graph.has_edge(step1_trace, step2_trace)
    assert graph.has_edge(feat1_trace, feat_trace)
    assert graph.has_edge(feat2_trace, feat_trace)
    assert graph.has_edge(feat1_trace, step2_trace)
    assert graph.has_edge(feat2_trace, step2_trace)
    assert graph.has_edge(step2_trace, feat_trace)

    # check graph content
    assert step1_trace.step is step1
    assert step2_trace.step is step2
    assert feat_trace.feature.id == "t-feat"
    assert_input_output_relation(graph, feat1_trace, feat_trace, step2)
    assert_edge_relation(graph, feat1_trace, step2_trace, TraceRelation.INPUT)
    assert_edge_relation(graph, step2_trace, feat_trace, TraceRelation.OUTPUT)


def test_inspect_feature_definition_mixin_step():
    """Test steps defining new features."""

    class _TestStep(ProcessingStep, FeatureDefinitionMixin):
        def process_reading(self, reading: Reading, **kwargs) -> ProcessResultType:
            pass

        definition = FeatureValueDefinition(AV("test", "t"), AV("feature", "feat"))

    graph = inspect(step := _TestStep())

    assert_graph_shape(graph, 3, 2)

    # test graph layout
    _, step_trace, feat_trace, *_ = graph.nodes()
    assert isinstance(step_trace, StepTrace)
    assert isinstance(feat_trace, FeatureTrace)
    assert graph.has_edge(step_trace, feat_trace)

    # test graph content
    assert step_trace.step is step
    assert feat_trace.feature.id == "t-feat"
    assert_edge_relation(graph, step_trace, feat_trace, TraceRelation.OUTPUT)


def test_inspect_create_level_epoch_step():
    """Test steps creating new level epochs."""

    class _TestStep(CreateLevelEpochStep):
        data_set_ids = "input1"
        definition = EpochDefinition(
            id_="eid", name="Epoch name", description="Epoch description"
        )

        @transformation
        def _func(self, _data):
            pass

    graph = inspect(step := _TestStep())

    assert_graph_shape(graph, 4, 4)

    # test graph layout
    _, step_trace, ds_trace, epoch_trace, *_ = graph.nodes()
    assert isinstance(step_trace, StepTrace)
    assert isinstance(ds_trace, DataSetTrace)
    assert isinstance(epoch_trace, EpochTrace)
    assert graph.has_edge(ds_trace, step_trace)
    assert graph.has_edge(step_trace, epoch_trace)
    assert graph.has_edge(ds_trace, epoch_trace)

    # test graph content
    assert step_trace.step == step
    assert_input_output_relation(graph, ds_trace, epoch_trace, step)


def test_inspect_create_level_epoch_step_with_data_set():
    """Test steps creating new level epochs with data set."""

    class _TestStep(CreateLevelEpochStep):
        data_set_ids = "input1"
        definition = EpochDefinition(
            id_="eid", name="Epoch name", description="Epoch description"
        )
        epoch_data_set_id = "epochs"

        @transformation
        def _func(self, data):
            pass

    graph = inspect(step := _TestStep())

    assert_graph_shape(graph, 5, 6)

    # test graph layout
    _, step_trace, ds_trace, epoch_trace, epoch_ds_trace, *_ = graph.nodes()
    assert isinstance(step_trace, StepTrace)
    assert isinstance(ds_trace, DataSetTrace)
    assert isinstance(epoch_trace, EpochTrace)
    assert isinstance(epoch_ds_trace, DataSetTrace)
    assert graph.has_edge(ds_trace, step_trace)
    assert graph.has_edge(step_trace, epoch_trace)
    assert graph.has_edge(ds_trace, epoch_trace)
    assert graph.has_edge(ds_trace, epoch_ds_trace)
    assert graph.has_edge(step_trace, epoch_ds_trace)

    # test graph content
    assert epoch_ds_trace.data_set_id == step.epoch_data_set_id
    assert_input_output_relation(graph, ds_trace, epoch_ds_trace, step)


def test_inspect_create_level_epoch_step_from_epochs():
    """Test steps creating new level epochs from epochs."""

    class _TestStepCreateEpoch(CreateLevelEpochStep):
        data_set_ids = "ds1"
        definition = EpochDefinition(id_="ep1", name="Epoch name 1")

        @transformation
        def _func(self, data):
            pass

    class _TestStep(LevelEpochProcessingStepMixIn, CreateLevelEpochStep):
        data_set_ids = "ds2"
        definition = EpochDefinition(id_="ep2", name="Epoch name 2")
        epoch_filter = LevelEpochIdFilter(_TestStepCreateEpoch.definition.id)
        epoch_data_set_id = "epochs"

        @transformation
        def _func(self, data):
            pass

    steps = [
        _TestStepCreateEpoch(),
        step2 := _TestStep(),
    ]
    graph = inspect(steps)

    assert_graph_shape(graph, 8, 13)

    # test graph layout
    (
        _,
        step1_trace,
        ds1_trace,
        ep1_trace,
        step2_trace,
        ds2_trace,
        ep2_trace,
        dse_trace,
        *_,
    ) = graph.nodes()
    assert isinstance(step1_trace, StepTrace)
    assert isinstance(ds1_trace, DataSetTrace)
    assert isinstance(ep1_trace, EpochTrace)
    assert isinstance(step2_trace, StepTrace)
    assert isinstance(ds2_trace, DataSetTrace)
    assert isinstance(ep2_trace, EpochTrace)
    assert isinstance(dse_trace, DataSetTrace)

    # test only edges of second step, first covered by other test cases
    assert graph.has_edge(ds2_trace, dse_trace)
    assert graph.has_edge(ep1_trace, dse_trace)
    assert graph.has_edge(ds2_trace, step2_trace)
    assert graph.has_edge(ep1_trace, step2_trace)
    assert graph.has_edge(step2_trace, dse_trace)
    assert graph.has_edge(step2_trace, ep2_trace)
    assert graph.has_edge(ds2_trace, ep2_trace)
    assert graph.has_edge(ep1_trace, ep2_trace)

    for source in [ds2_trace, ep1_trace]:
        assert_input_output_relation(graph, source, ep2_trace, step2)


def test_inspect_extract_features_from_epochs():
    """Test steps extracting features from level epochs."""

    class _TestStepCreateEpoch(CreateLevelEpochStep):
        data_set_ids = "ds1"
        definition = EpochDefinition(id_="ep1")

        @transformation
        def _func(self, data):
            pass

    class _TestStep(LevelEpochExtractStep):
        data_set_ids = "ds1"
        epoch_filter = LevelEpochIdFilter(_TestStepCreateEpoch.definition.id)
        definition = FeatureValueDefinition(AV("task", "t"), AV("feature", "f"))

        @transformation
        def _func(self, data):
            pass

    steps = [
        _TestStepCreateEpoch(),
        step2 := _TestStep(),
    ]

    graph = inspect(steps)

    assert_graph_shape(graph, 6, 10)

    # test graph layout
    _, step1_trace, ds1_trace, ep1_trace, step2_trace, feature_trace, *_ = graph.nodes()
    assert isinstance(step1_trace, StepTrace)
    assert isinstance(ds1_trace, DataSetTrace)
    assert isinstance(ep1_trace, EpochTrace)
    assert isinstance(step2_trace, StepTrace)
    assert isinstance(feature_trace, FeatureTrace)

    assert graph.has_edge(ep1_trace, feature_trace)
    assert graph.has_edge(ep1_trace, step2_trace)
    assert graph.has_edge(ds1_trace, feature_trace)

    assert_input_output_relation(graph, ep1_trace, feature_trace, step2)


class _TestParamTransformStep(TransformStep):
    param = Parameter("param")


class _TestParamExtractStep(ExtractStep):
    param = Parameter("param")


@pytest.fixture
def ancestor_inspect_example() -> nx.MultiDiGraph:
    """Create an example used to test graph dependencies."""
    columns = [RawDataValueDefinition("c1", "column1")]
    feat_def1 = FeatureValueDefinition(AV("test", "t"), AV("feat-1", "f1"))
    feat_def2 = FeatureValueDefinition(AV("test", "t"), AV("feat-2", "f2"))
    feat_def3 = FeatureValueDefinition(AV("test", "t"), AV("feat-3", "f3"))
    steps = CoreProcessingStepGroup(
        [
            TransformStep("input1", lambda _: None, "output1", columns),
            _TestParamTransformStep("output1", lambda _: None, "output2", columns),
            _TestParamTransformStep("output1", lambda _: None, "output3", columns),
            _TestParamExtractStep("output2", lambda _: None, feat_def1),
            ExtractStep("output2", lambda _: None, feat_def2),
            ExtractStep("input1", lambda _: None, feat_def3),
        ]
    )
    return inspect(steps)


def test_get_ancestors(ancestor_inspect_example):
    """Test that get_ancestors returns all predecessors until the origin."""
    assert ancestor_inspect_example.number_of_nodes() == 15
    assert ancestor_inspect_example.number_of_edges() == 31

    ft1, ft2, ft3, *_ = get_traces(ancestor_inspect_example, FeatureTrace)

    assert isinstance(ft1.step, _TestParamExtractStep)
    assert isinstance(ft2.step, ExtractStep)
    assert isinstance(ft3.step, ExtractStep)

    ft1_ancestors = get_ancestors(ancestor_inspect_example, ft1)
    assert len(ft1_ancestors) == 9
    assert ft2 not in ft1_ancestors
    assert ft3 not in ft1_ancestors


def test_get_ancestor_source_graph(ancestor_inspect_example):
    """Test getting source ancestors for a feature trace."""
    ft1, *_ = get_traces(ancestor_inspect_example, FeatureTrace)

    ft1_graph = get_ancestor_source_graph(ancestor_inspect_example, ft1)
    assert ft1_graph.number_of_nodes() == 4
    assert ft1_graph.number_of_edges() == 3
    assert all(isinstance(n, (DataSetTrace, FeatureTrace)) for n in ft1_graph.nodes)
    assert ft1 in ft1_graph.nodes


def test_get_ancestor_source_graph_cycle():
    """Test cyclic transformation steps not leading to infinite recursion."""
    definition = [RawDataValueDefinition("c", "c")]
    steps = [
        TransformStep("input", lambda _: None, "output", definition),
        TransformStep("output", lambda _: None, "input", definition),
        ExtractStep(
            "output", lambda _: None, FeatureValueDefinition(AV("t", "t"), AV("f", "f"))
        ),
    ]

    graph = inspect(steps)
    feature_trace, *_ = get_traces(graph, FeatureTrace)
    ancestor_graph = get_ancestor_source_graph(graph, feature_trace)

    assert ancestor_graph.number_of_nodes() == 3


def test_get_edge_parameters(ancestor_inspect_example):
    """Test getting parameters defined in edge steps."""
    ft1, _, ft2, *_ = get_traces(ancestor_inspect_example, FeatureTrace)

    ft1_graph = get_ancestor_source_graph(ancestor_inspect_example, ft1)
    param1 = get_edge_parameters(ft1_graph)
    assert len(param1) == 2

    ft2_graph = get_ancestor_source_graph(ancestor_inspect_example, ft2)
    param2 = get_edge_parameters(ft2_graph)
    assert len(param2) == 0
