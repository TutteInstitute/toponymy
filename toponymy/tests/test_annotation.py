from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from toponymy.annotation import (
    Annotation,
    AnnotationState,
    AnnotationStore,
    AnnotationTree,
    DescendantsInput,
    Executor,
    InvalidAnnotationTree,
    NodeId,
    NodeInput,
    NodeOutput,
    _members_by_cluster,
    parse_algorithm_type,
)
from toponymy.exemplar_texts import (
    DiverseExemplarAnnotator,
    RandomExemplarAnnotator,
    SubmodularExemplarAnnotator,
)


@pytest.fixture
def valid_cluster_tree():
    # Root is (2, 0)
    return {
        (1, 0): [(0, 0)],
        (1, 1): [(0, 1)],
        (1, 2): [(0, 7), (0, 6), (0, 5)],
        (1, 3): [(0, 4)],
        (2, 0): [(1, 0), (1, 1), (1, 2), (1, 3), (0, 2), (0, 3)],
    }


@pytest.fixture
def deep_cluster_tree():
    # Root is (4, 0)
    return {
        (4, 0): [(3, 0), (3, 1)],
        (3, 0): [(2, 0), (2, 1)],
        (3, 1): [(2, 2)],
        (2, 0): [(1, 0), (1, 1)],
        (2, 1): [(1, 2)],
        (2, 2): [(1, 3), (1, 4)],
        (1, 0): [(0, 0), (0, 1)],
        (1, 1): [(0, 2)],
        (1, 3): [(0, 3), (0, 4)],
        (1, 4): [(0, 5)],
    }


@pytest.fixture
def annotation(valid_cluster_tree):
    return Annotation(name="topics", tree=AnnotationTree(valid_cluster_tree))


def test_node_id_repr():
    node = NodeId(1, 7)
    assert repr(node) == "NodeId(1, 7)"


def test_tree_len(valid_cluster_tree):
    tree = AnnotationTree(valid_cluster_tree)

    assert len(tree) == 12


def test_tree_n_layers(valid_cluster_tree):
    tree = AnnotationTree(valid_cluster_tree)

    assert tree.n_layers == 2


def test_tree_nodes_is_tuple(valid_cluster_tree):
    tree = AnnotationTree(valid_cluster_tree)
    assert isinstance(tree.nodes, tuple)


def test_tree_nodes(valid_cluster_tree):
    tree = AnnotationTree(valid_cluster_tree)
    assert tree.nodes == tuple(
        sorted(
            [
                NodeId(0, 0),
                NodeId(0, 1),
                NodeId(0, 2),
                NodeId(1, 0),
                NodeId(1, 1),
                NodeId(1, 2),
                NodeId(1, 3),
                NodeId(0, 3),
                NodeId(0, 4),
                NodeId(0, 5),
                NodeId(0, 6),
                NodeId(0, 7),
            ]
        )
    )
    assert len(tree.nodes) == 12


def test_tree_layer_has_no_root_layer(valid_cluster_tree):
    tree = AnnotationTree(valid_cluster_tree)

    assert tree.layer(0) == (
        NodeId(0, 0),
        NodeId(0, 1),
        NodeId(0, 2),
        NodeId(0, 3),
        NodeId(0, 4),
        NodeId(0, 5),
        NodeId(0, 6),
        NodeId(0, 7),
    )
    assert tree.layer(1) == (
        NodeId(1, 0),
        NodeId(1, 1),
        NodeId(1, 2),
        NodeId(1, 3),
    )
    with pytest.raises(KeyError):
        tree.layer(2)


def test_tree_children(valid_cluster_tree):
    tree = AnnotationTree(valid_cluster_tree)
    for parent, children in valid_cluster_tree.items():
        if parent != tree._root:
            assert tree.children(parent) == sorted(children)

    for node in tree.layer(0):
        assert tree.children(node) is None

    assert tree.children(tree._root) is None


def test_tree_parent(valid_cluster_tree):
    tree = AnnotationTree(valid_cluster_tree)
    for parent, children in valid_cluster_tree.items():
        for child in children:
            if child not in tree._root_children:
                assert tree.parent(child) == parent

    assert tree.parent(tree._root) is None

    for child in tree._root_children:
        assert tree.parent(child) is None


def test_tree_contains(valid_cluster_tree):
    tree = AnnotationTree(valid_cluster_tree)

    assert NodeId(1, 0) in tree
    assert NodeId(0, 2) in tree

    # Root is intentionally excluded from the tree
    assert NodeId(2, 0) not in tree

    # Non-hashable or wrong type should safely return False.
    assert ["not", "a", "node"] not in tree


def test_tree_rejects_multiple_parents():
    invalid_cluster_tree = {
        (1, 0): [(0, 0)],
        (1, 1): [(0, 0)],
        (2, 0): [(1, 0), (1, 1)],
    }

    with pytest.raises(InvalidAnnotationTree, match="multiple parents"):
        AnnotationTree(invalid_cluster_tree)


def test_tree_rejects_multiple_roots():
    invalid_cluster_tree = {
        (1, 0): [(0, 0)],
        (1, 1): [(0, 1)],
    }

    with pytest.raises(InvalidAnnotationTree, match="multiple roots"):
        AnnotationTree(invalid_cluster_tree)


def test_tree_rejects_no_root():
    invalid_cluster_tree = {
        (1, 0): [(1, 1)],
        (1, 1): [(1, 0)],
    }

    with pytest.raises(
        InvalidAnnotationTree, match="must be in a lower layer than its parent"
    ):
        AnnotationTree(invalid_cluster_tree)


def test_tree_empty_raises_invalid_tre():
    with pytest.raises(InvalidAnnotationTree, match="Empty cluster tree"):
        AnnotationTree({})


def test_tree_rejects_non_contiguous_cluster_ids_within_layer():
    invalid_cluster_tree = {
        (1, 0): [(0, 0), (0, 2)],  # gap: missing (0, 1)
        (2, 0): [(1, 0)],
    }

    with pytest.raises(
        InvalidAnnotationTree, match="Cluster ids in layer 0 must be contiguous"
    ):
        AnnotationTree(invalid_cluster_tree)


def test_layer_ids_are_zero_indexed_and_contiguous(valid_cluster_tree):
    tree = AnnotationTree(valid_cluster_tree)
    layer_ids = tree.layer_ids

    assert layer_ids[0] == 0
    assert layer_ids == tuple(range(max(layer_ids) + 1))
    assert len(layer_ids) == max(layer_ids) + 1


def test_tree_rejects_non_contiguous_layer_ids():
    invalid_cluster_tree = {
        (3, 0): [(2, 0)],
        (2, 0): [(0, 0)],
    }

    with pytest.raises(InvalidAnnotationTree, match="Layer ids must be contiguous"):
        AnnotationTree(invalid_cluster_tree)


def test_tree_layer_missing_raises_key_error(valid_cluster_tree):
    tree = AnnotationTree(valid_cluster_tree)

    with pytest.raises(KeyError):
        tree.layer(99)


def test_tree_from_clusterer(valid_cluster_tree):
    class DummyClusterer:
        @property
        def cluster_tree_(self):
            return valid_cluster_tree

    tree = AnnotationTree.from_clusterer(DummyClusterer())

    assert len(tree) == 12
    assert tree.parent(NodeId(0, 0)) == NodeId(1, 0)


def test_tree_descendants_root_children(valid_cluster_tree):
    tree = AnnotationTree(valid_cluster_tree)

    descendants = []
    for node in tree._root_children:
        descendants.extend(tree.descendants(node))
    assert set(descendants) == set(tree.nodes) - set(tree._root_children)


def test_tree_descendants_depth_limited_deep_tree(deep_cluster_tree):
    tree = AnnotationTree(deep_cluster_tree)

    assert tree.descendants(tree._root) == []

    start = NodeId(3, 0)

    depth_1 = tree.descendants(start, depth=1)
    assert set(depth_1) == {NodeId(2, 0), NodeId(2, 1)}

    depth_2 = tree.descendants(start, depth=2)
    assert set(depth_2) == {
        NodeId(2, 0),
        NodeId(2, 1),
        NodeId(1, 0),
        NodeId(1, 1),
        NodeId(1, 2),
    }

    depth_3 = tree.descendants(start, depth=3)
    assert set(depth_3) == {
        NodeId(2, 0),
        NodeId(2, 1),
        NodeId(1, 0),
        NodeId(1, 1),
        NodeId(1, 2),
        NodeId(0, 0),
        NodeId(0, 1),
        NodeId(0, 2),
    }

    assert set(tree.descendants(start)) == set(depth_3)


def test_tree_descendants_leaf_and_zero_depth(valid_cluster_tree):
    tree = AnnotationTree(valid_cluster_tree)
    assert tree.descendants(NodeId(0, 0)) == []
    assert tree.descendants(NodeId(1, 2), depth=0) == []


def test_tree_descendants_negative_depth_raises(valid_cluster_tree):
    tree = AnnotationTree(valid_cluster_tree)
    with pytest.raises(ValueError, match="depth must be >= 0 or None"):
        tree.descendants(NodeId(1, 2), depth=-1)


def test_tree_omits_root_links(valid_cluster_tree):
    tree = AnnotationTree(valid_cluster_tree)

    root_children = tree._root_children
    assert root_children is not None
    assert len(root_children) > 0

    for child in root_children:
        assert tree.parent(child) is None


def test_annotation_tree_equality_same_structure_different_input_order():
    t1 = {
        (1, 0): [(0, 0)],
        (1, 1): [(0, 1)],
        (2, 0): [(1, 0), (1, 1)],
    }
    t2 = {
        (2, 0): [(1, 1), (1, 0)],
        (1, 1): [(0, 1)],
        (1, 0): [(0, 0)],
    }

    assert AnnotationTree(t1) == AnnotationTree(t2)


def test_annotation_tree_inequality_different_structure():
    t1 = {
        (1, 0): [(0, 0)],
        (1, 1): [(0, 1)],
        (2, 0): [(1, 0), (1, 1)],
    }
    t2 = {
        (1, 0): [(0, 0), (0, 1)],
        (2, 0): [(1, 0)],
    }

    assert AnnotationTree(t1) != AnnotationTree(t2)


def test_annotation_tree_equality_non_tree_type():
    tree = AnnotationTree(
        {
            (1, 0): [(0, 0)],
            (2, 0): [(1, 0)],
        }
    )
    assert (tree == "not-a-tree") is False


def test_annotation_initial_state(annotation, valid_cluster_tree):
    tree = AnnotationTree(valid_cluster_tree)

    assert annotation.name == "topics"
    assert annotation.tree == tree
    assert len(annotation) == 0
    assert set(annotation.states.keys()) == set(tree.nodes)
    assert all(state is AnnotationState.EMPTY for state in annotation.states.values())


def test_annotation_set_and_get_with_nodeid(annotation):
    node = NodeId(1, 0)
    annotation[node] = "Physics"

    assert annotation[node] == "Physics"
    assert annotation.states[node] is AnnotationState.COMPUTED
    assert len(annotation) == 1


def test_annotation_set_and_get_with_tuple_coercion(annotation):
    annotation[(1, 1)] = "Mathematics"

    assert annotation[NodeId(1, 1)] == "Mathematics"
    assert annotation[(1, 1)] == "Mathematics"
    assert annotation.states[NodeId(1, 1)] is AnnotationState.COMPUTED


def test_annotation_getitem_unset_node_raises_keyerror(annotation):
    with pytest.raises(KeyError):
        _ = annotation[NodeId(0, 0)]


def test_annotation_get_default_for_unset_node(annotation):
    assert annotation.get(NodeId(0, 0), "missing") == "missing"


def test_annotation_delete_resets_to_empty(annotation):
    node = NodeId(0, 2)
    annotation[node] = "Compilers"
    assert annotation.states[node] is AnnotationState.COMPUTED

    del annotation[node]

    assert annotation.states[node] is AnnotationState.EMPTY
    with pytest.raises(KeyError):
        _ = annotation[node]


def test_annotation_delete_unset_node_is_noop(annotation):
    node = NodeId(0, 3)

    del annotation[node]

    assert annotation.states[node] is AnnotationState.EMPTY
    with pytest.raises(KeyError):
        _ = annotation[node]


def test_annotation_fail_marks_failed_and_clears_value(annotation):
    node = NodeId(1, 2)
    annotation[node] = "Databases"
    assert annotation.states[node] is AnnotationState.COMPUTED

    annotation.fail(node)

    assert annotation.states[node] is AnnotationState.FAILED
    with pytest.raises(KeyError):
        _ = annotation[node]


def test_annotation_set_after_fail_overwrites_and_marks_computed(annotation):
    node = NodeId(1, 3)
    annotation.fail(node)
    assert annotation.states[node] is AnnotationState.FAILED

    annotation[node] = "Security"

    assert annotation[node] == "Security"
    assert annotation.states[node] is AnnotationState.COMPUTED


def test_annotation_invalid_node_format_raises_keyerror(annotation):
    with pytest.raises(KeyError, match="Invalid node ID"):
        annotation["not-a-node"] = "x"

    with pytest.raises(KeyError, match="Invalid node ID"):
        _ = annotation[(1,)]


def test_annotation_root_node_rejected(annotation):
    with pytest.raises(KeyError, match="is not in the annotation tree"):
        annotation[(2, 0)] = "RootLabel"


def test_annotation_iteration_and_items_only_include_computed(annotation):
    annotation[(1, 0)] = "A"
    annotation[(0, 0)] = "B"

    keys = list(annotation.keys())
    values = list(annotation.values())
    items = list(annotation.items())

    assert set(keys) == {NodeId(1, 0), NodeId(0, 0)}
    assert set(values) == {"A", "B"}
    assert set(items) == {(NodeId(1, 0), "A"), (NodeId(0, 0), "B")}


def test_annotation_update_uses_mapping_protocol(annotation):
    annotation.update({(1, 0): "Topic-1", (0, 1): "Topic-2"})

    assert annotation[NodeId(1, 0)] == "Topic-1"
    assert annotation[NodeId(0, 1)] == "Topic-2"
    assert annotation.states[NodeId(1, 0)] is AnnotationState.COMPUTED
    assert annotation.states[NodeId(0, 1)] is AnnotationState.COMPUTED


def test_annotation_from_layered_list_populates_values_and_states(valid_cluster_tree):
    tree = AnnotationTree(valid_cluster_tree)
    layered_values = [
        [f"L0-{i}" for i in range(8)],
        [f"L1-{i}" for i in range(4)],
    ]

    annotation = Annotation.from_layered_list("topics", tree, layered_values)

    assert len(annotation) == len(tree)
    for node in tree.nodes:
        assert annotation[node] == f"L{node.layer}-{node.cluster}"

    assert all(
        state is AnnotationState.COMPUTED for state in annotation.states.values()
    )


def test_annotation_to_layered_list_round_trip(valid_cluster_tree):
    tree = AnnotationTree(valid_cluster_tree)
    layered_values = [
        [f"L0-{i}" for i in range(8)],
        [f"L1-{i}" for i in range(4)],
    ]

    annotation = Annotation.from_layered_list("topics", tree, layered_values)

    assert annotation.to_layered_list() == layered_values


def test_annotation_to_layered_list_raises_when_not_fully_computed(valid_cluster_tree):
    tree = AnnotationTree(valid_cluster_tree)
    annotation = Annotation("topics", tree)
    annotation[NodeId(0, 0)] = "only-one"

    with pytest.raises(ValueError, match="Not all nodes have computed values"):
        annotation.to_layered_list()


def test_annotation_to_layered_list_raises_when_any_failed(valid_cluster_tree):
    tree = AnnotationTree(valid_cluster_tree)
    layered_values = [
        [f"L0-{i}" for i in range(8)],
        [f"L1-{i}" for i in range(4)],
    ]
    annotation = Annotation.from_layered_list("topics", tree, layered_values)
    annotation.fail(NodeId(0, 0))

    with pytest.raises(ValueError, match="Not all nodes have computed values"):
        annotation.to_layered_list()


def test_annotation_from_layered_list_raises_on_incomplete_input(valid_cluster_tree):
    tree = AnnotationTree(valid_cluster_tree)
    incomplete_layered_values = [
        ["L0-0"],
        ["L1-0"],
    ]

    with pytest.raises(IndexError):
        Annotation.from_layered_list("topics", tree, incomplete_layered_values)


def test_executor_node_node_string_algorithm_type_contract(valid_cluster_tree):
    tree = AnnotationTree(valid_cluster_tree)

    topic = Annotation("topic", tree)
    for node in tree.nodes:
        topic[node] = f"T{node.layer}-{node.cluster}"

    store = AnnotationStore(tree, [topic])
    executor = Executor(store)

    class EchoAnnotator:
        inputs = ("topic",)
        outputs = ("named_topic",)
        algorithm_type = "node-node"

        def annotate(self, unit, topic):
            return {"named_topic": f"name:{topic}"}

    failures = executor.run(EchoAnnotator())

    assert failures == {}
    assert "named_topic" in store
    assert len(store["named_topic"]) == len(tree)
    assert store["named_topic"][NodeId(0, 0)] == "name:T0-0"
    assert store["named_topic"][NodeId(1, 2)] == "name:T1-2"


def test_executor_explicit_node_and_descendants_input_contract(valid_cluster_tree):
    tree = AnnotationTree(valid_cluster_tree)

    topic = Annotation("topic", tree)
    for node in tree.nodes:
        topic[node] = f"T{node.layer}-{node.cluster}"

    store = AnnotationStore(tree, [topic])
    executor = Executor(store)

    class NodeAndDescendantsAnnotator:
        inputs = ("topic", "desc_topics")
        outputs = ("summary",)
        algorithm_type = {
            "topic": (NodeInput(), NodeOutput()),
            "desc_topics": (DescendantsInput(source="topic", depth=1), NodeOutput()),
        }

        def annotate(self, unit, topic, desc_topics):
            return {"summary": {"topic": topic, "desc": tuple(desc_topics.values())}}

    failures = executor.run(NodeAndDescendantsAnnotator())

    assert failures == {}

    summary = store["summary"]

    # Leaf nodes have no descendants at depth=1
    assert summary[NodeId(0, 0)] == {"topic": "T0-0", "desc": ()}

    # Layer-1 node (1,2) has children (0,5), (0,6), (0,7)
    assert summary[NodeId(1, 2)]["topic"] == "T1-2"
    assert set(summary[NodeId(1, 2)]["desc"]) == {"T0-5", "T0-6", "T0-7"}


def test_executor_in_place_failure_preserves_only_existing_values(valid_cluster_tree):
    tree = AnnotationTree(valid_cluster_tree)

    precomputed_node = NodeId(0, 0)
    empty_node = NodeId(0, 1)
    original_value = "original-topic"

    topic = Annotation("topic", tree)
    topic[precomputed_node] = original_value
    # empty_node is intentionally left EMPTY

    store = AnnotationStore(tree, [topic])
    executor = Executor(store)

    class AlwaysFailInPlaceAnnotator:
        # No required inputs, so annotate is called even when optional input is missing.
        inputs = ()
        optional_inputs = ("topic",)
        outputs = ("topic",)
        algorithm_type = "node-node"

        def annotate(self, unit, **kwargs):
            raise RuntimeError("boom")

    failures = executor.run(
        AlwaysFailInPlaceAnnotator(),
        nodes=(precomputed_node, empty_node),
    )

    assert set(failures) == {precomputed_node, empty_node}
    assert all("RuntimeError: boom" in message for message in failures.values())

    # Precomputed value is preserved for in-place failure.
    assert store["topic"].states[precomputed_node] is AnnotationState.COMPUTED
    assert store["topic"][precomputed_node] == original_value

    # Empty in-place value is marked failed on failure.
    assert store["topic"].states[empty_node] is AnnotationState.FAILED


def test_parse_algorithm_type_shorthand_returns_output_instance():
    class EmptyInputsAnnotator:
        inputs = ()
        outputs = ("topic",)
        algorithm_type = "node-node"

        def annotate(self, unit, **kwargs):
            return {"topic": "x"}

    specs, output = parse_algorithm_type(EmptyInputsAnnotator())
    assert specs == {}
    assert isinstance(output, NodeOutput)


def test_executor_explicit_contract_with_node_output_type(valid_cluster_tree):
    tree = AnnotationTree(valid_cluster_tree)
    topic = Annotation("topic", tree)
    for node in tree.nodes:
        topic[node] = f"T{node.layer}-{node.cluster}"

    store = AnnotationStore(tree, [topic])
    executor = Executor(store)

    class ExplicitAnnotator:
        inputs = ("topic",)
        outputs = ("named_topic",)
        algorithm_input_type = {"topic": "node"}
        algorithm_output_type = "node"

        def annotate(self, unit, topic):
            return {"named_topic": f"name:{topic}"}

    failures = executor.run(ExplicitAnnotator())

    assert failures == {}
    assert "named_topic" in store
    assert store["named_topic"][NodeId(0, 0)] == "name:T0-0"


def test_executor_explicit_contract_with_node_output_dict(valid_cluster_tree):
    tree = AnnotationTree(valid_cluster_tree)
    topic = Annotation("topic", tree)
    for node in tree.nodes:
        topic[node] = f"T{node.layer}-{node.cluster}"

    store = AnnotationStore(tree, [topic])
    executor = Executor(store)

    class ExplicitAnnotator:
        inputs = ("topic",)
        outputs = ("named_topic",)
        algorithm_input_type = {"topic": "node"}
        algorithm_output_type = {"named_topic": "node"}

        def annotate(self, unit, topic):
            return {"named_topic": f"name:{topic}"}

    failures = executor.run(ExplicitAnnotator())

    assert failures == {}
    assert store["named_topic"][NodeId(1, 2)] == "name:T1-2"


def test_executor_explicit_contract_requires_output_types(valid_cluster_tree):
    tree = AnnotationTree(valid_cluster_tree)
    store = AnnotationStore(tree, [])
    executor = Executor(store)

    class MissingOutputTypeAnnotator:
        inputs = ("topic",)
        outputs = ("named_topic",)
        algorithm_input_type = {"topic": "node"}

        def annotate(self, unit, **kwargs):
            return {"topic": "x"}

    with pytest.raises(
        TypeError,
        match="explicit algorithm contract requires both algorithm_input_type and algorithm_output_type",
    ):
        executor.run(MissingOutputTypeAnnotator())


def test_executor_explicit_contract_rejects_unknown_output_spec(valid_cluster_tree):
    tree = AnnotationTree(valid_cluster_tree)
    topic = Annotation("topic", tree)
    for node in tree.nodes:
        topic[node] = f"T{node.layer}-{node.cluster}"

    store = AnnotationStore(tree, [topic])
    executor = Executor(store)

    class BadOutputSpecAnnotator:
        inputs = ("topic",)
        outputs = ("named_topic",)
        algorithm_input_type = {"topic": "node"}
        algorithm_output_type = "unknown_output_spec"

        def annotate(self, unit, topic):
            return {"named_topic": f"name:{topic}"}

    with pytest.raises(ValueError, match="unknown output spec"):
        executor.run(BadOutputSpecAnnotator())


def test_parse_algorithm_type_prefers_algorithm_type_when_both_present(
    valid_cluster_tree,
):
    tree = AnnotationTree(valid_cluster_tree)
    topic = Annotation("topic", tree)
    for node in tree.nodes:
        topic[node] = f"T{node.layer}-{node.cluster}"

    store = AnnotationStore(tree, [topic])
    executor = Executor(store)

    class BothContractsAnnotator:
        inputs = ("topic",)
        outputs = ("named_topic",)
        algorithm_type = "node-node"
        algorithm_input_type = {"wrong_name": "node"}
        algorithm_output_type = "node"

        def annotate(self, unit, topic):
            return {"named_topic": f"name:{topic}"}

    specs, output = parse_algorithm_type(BothContractsAnnotator())
    assert set(specs) == {"topic"}
    assert isinstance(output, NodeOutput)
    with pytest.warns(
        UserWarning,
        match="defines both algorithm_type and \\(algorithm_input_type, algorithm_output_type\\)",
    ) as w:
        failures = executor.run(BothContractsAnnotator())
    assert failures == {}
    assert len(w) == 1


@pytest.fixture
def two_layer_clusterer(subtopic_label_vector, cluster_label_vector):
    # 5 topics of 5 subtopics each, nested the way the conftest label vectors are.
    tree = {(1, i): [(0, 5 * i + j) for j in range(5)] for i in range(5)}
    # The root above the top layer is dropped by AnnotationTree.
    tree[(2, 0)] = [(1, i) for i in range(5)]
    return SimpleNamespace(
        cluster_tree_=tree,
        cluster_layers_=[
            SimpleNamespace(cluster_labels=subtopic_label_vector),
            SimpleNamespace(cluster_labels=cluster_label_vector),
        ],
    )


def test_members_by_cluster_handles_gaps_noise_and_empty_clusters():
    labels = np.array([7, -1, 3, 7, 3, 9])

    members = _members_by_cluster(labels, [3, 5, 7])

    assert {k: v.tolist() for k, v in members.items()} == {
        3: [2, 4],
        5: [],
        7: [0, 3],
    }


def test_members_by_cluster_matches_a_scan_for_each_id():
    labels = np.random.default_rng(0).choice([-1, 0, 2, 5, 11], size=200)
    cluster_ids = [0, 2, 5, 11]

    members = _members_by_cluster(labels, cluster_ids)

    for cluster in cluster_ids:
        assert np.array_equal(members[cluster], np.flatnonzero(labels == cluster))


def test_store_from_clusterer_defaults_to_indices_only(
    two_layer_clusterer, subtopic_label_vector, cluster_label_vector
):
    store = AnnotationStore.from_clusterer(two_layer_clusterer)

    assert set(store) == {"cluster_object_indices"}
    assert store.tree == AnnotationTree.from_clusterer(two_layer_clusterer)
    for annotation in store.values():
        assert all(s is AnnotationState.COMPUTED for s in annotation.states.values())
    labels = [subtopic_label_vector, cluster_label_vector]
    for node in store.tree.nodes:
        expected = np.where(labels[node.layer] == node.cluster)[0]
        assert np.array_equal(store.cluster_object_indices[node], expected)


def test_store_from_clusterer_computes_centroids_from_vectors(
    two_layer_clusterer,
    object_vectors,
    subtopic_centroid_vectors,
    cluster_centroid_vectors,
):
    store = AnnotationStore.from_clusterer(
        two_layer_clusterer, object_vectors=object_vectors
    )

    expected = [subtopic_centroid_vectors, cluster_centroid_vectors]
    for node in store.tree.nodes:
        assert np.allclose(
            store.cluster_centroid[node], expected[node.layer][node.cluster]
        )


def test_store_from_clusterer_adds_objects_and_vectors_when_given(
    two_layer_clusterer, all_sentences, object_vectors
):
    store = AnnotationStore.from_clusterer(
        two_layer_clusterer, all_sentences, object_vectors
    )

    assert set(store) == {
        "cluster_centroid",
        "cluster_object_indices",
        "cluster_objects",
        "cluster_object_vectors",
    }
    for node in store.tree.nodes:
        indices = store.cluster_object_indices[node]
        assert store.cluster_objects[node] == [all_sentences[i] for i in indices]
        assert np.array_equal(
            store.cluster_object_vectors[node], object_vectors[indices]
        )


def test_store_from_clusterer_accepts_any_iterable_of_objects(
    two_layer_clusterer, all_sentences
):
    shuffled_index = pd.Series(
        all_sentences, index=range(1000, 1000 + len(all_sentences))
    )
    for objects in (tuple(all_sentences), iter(all_sentences), shuffled_index):
        store = AnnotationStore.from_clusterer(two_layer_clusterer, objects)
        for node in store.tree.nodes:
            indices = store.cluster_object_indices[node]
            assert store.cluster_objects[node] == [all_sentences[i] for i in indices]


def test_store_from_clusterer_holds_objects_by_reference(
    two_layer_clusterer, all_sentences
):
    images = [np.zeros((8, 8)) for _ in all_sentences]

    store = AnnotationStore.from_clusterer(two_layer_clusterer, images)

    for node in store.tree.nodes:
        indices = store.cluster_object_indices[node]
        assert all(o is images[i] for o, i in zip(store.cluster_objects[node], indices))


def test_store_from_clusterer_accepts_vectors_as_lists(
    two_layer_clusterer, object_vectors
):
    store = AnnotationStore.from_clusterer(
        two_layer_clusterer, object_vectors=object_vectors.tolist()
    )

    node = NodeId(1, 2)
    indices = store.cluster_object_indices[node]
    assert np.allclose(store.cluster_object_vectors[node], object_vectors[indices])


def test_store_from_clusterer_omits_noise_points(
    two_layer_clusterer, subtopic_label_vector
):
    noisy = subtopic_label_vector.copy()
    noisy[[0, 7, 100]] = -1
    two_layer_clusterer.cluster_layers_[0].cluster_labels = noisy

    store = AnnotationStore.from_clusterer(two_layer_clusterer)

    in_a_node = np.concatenate(
        [store.cluster_object_indices[n] for n in store.tree.layer(0)]
    )
    assert set(range(len(noisy))) - set(in_a_node) == {0, 7, 100}
    assert len(in_a_node) == len(noisy) - 3


def test_store_from_clusterer_runs_under_executor(two_layer_clusterer):
    store = AnnotationStore.from_clusterer(two_layer_clusterer)

    class CountAnnotator:
        inputs = ("cluster_object_indices",)
        outputs = ("n_objects",)
        algorithm_type = "node-node"

        def annotate(self, unit, cluster_object_indices):
            return {"n_objects": len(cluster_object_indices)}

    assert Executor(store).run(CountAnnotator()) == {}
    assert {store.n_objects[n] for n in store.tree.layer(0)} == {5}
    assert {store.n_objects[n] for n in store.tree.layer(1)} == {25}


def test_store_from_clusterer_with_real_clusterer(
    clusterer, clusterable_vectors, object_vectors, all_sentences
):
    clusterer.fit(
        clusterable_vectors, object_vectors, show_progress_bar=False, verbose=False
    )

    store = AnnotationStore.from_clusterer(clusterer, all_sentences, object_vectors)

    assert store.tree.n_layers == len(clusterer.cluster_layers_)
    for node in store.tree.nodes:
        labels = clusterer.cluster_layers_[node.layer].cluster_labels
        expected = np.where(labels == node.cluster)[0]
        assert np.array_equal(store.cluster_object_indices[node], expected)


def test_store_from_clusterer_rejects_unfitted_clusterer(two_layer_clusterer):
    del two_layer_clusterer.cluster_layers_

    with pytest.raises(NotFittedError, match="cluster_layers_"):
        AnnotationStore.from_clusterer(two_layer_clusterer)


def test_store_from_clusterer_rejects_one_dimensional_vectors(
    two_layer_clusterer, all_sentences
):
    with pytest.raises(ValueError, match="2-dimensional"):
        AnnotationStore.from_clusterer(
            two_layer_clusterer, object_vectors=np.zeros(len(all_sentences))
        )


def test_store_from_clusterer_rejects_wrong_object_count(
    two_layer_clusterer, all_sentences
):
    with pytest.raises(ValueError, match="objects for"):
        AnnotationStore.from_clusterer(two_layer_clusterer, all_sentences[:-1])


def test_store_from_clusterer_rejects_wrong_vector_count(
    two_layer_clusterer, object_vectors
):
    with pytest.raises(ValueError, match="object vectors in a 2-dimensional"):
        AnnotationStore.from_clusterer(
            two_layer_clusterer, object_vectors=object_vectors[:-1]
        )


def test_store_from_clusterer_rejects_label_beyond_the_tree(
    two_layer_clusterer, cluster_label_vector
):
    bad = cluster_label_vector.copy()
    bad[0] = 9
    two_layer_clusterer.cluster_layers_[1].cluster_labels = bad

    with pytest.raises(ValueError, match=r"labels not in the tree: \[9\]"):
        AnnotationStore.from_clusterer(two_layer_clusterer)


def test_store_from_clusterer_rejects_layer_length_mismatch(
    two_layer_clusterer, cluster_label_vector
):
    two_layer_clusterer.cluster_layers_[1].cluster_labels = cluster_label_vector[:-1]

    with pytest.raises(ValueError, match="layer 1 has .* labels but layer 0 has"):
        AnnotationStore.from_clusterer(two_layer_clusterer)


def test_store_from_clusterer_rejects_layer_count_mismatch(two_layer_clusterer):
    two_layer_clusterer.cluster_layers_ = two_layer_clusterer.cluster_layers_[:1]

    with pytest.raises(ValueError, match="1 cluster layers but its tree has 2"):
        AnnotationStore.from_clusterer(two_layer_clusterer)


def test_store_from_clusterer_rejects_non_contiguous_cluster_ids(two_layer_clusterer):
    two_layer_clusterer.cluster_tree_[(1, 1)].remove((0, 7))

    with pytest.raises(InvalidAnnotationTree):
        AnnotationStore.from_clusterer(two_layer_clusterer)


@pytest.fixture(
    params=[
        RandomExemplarAnnotator,
        DiverseExemplarAnnotator,
        SubmodularExemplarAnnotator,
    ],
    ids=lambda cls: cls.__name__,
)
def exemplar_annotator(request, object_vectors):
    kwargs = {}
    if request.param is not RandomExemplarAnnotator:
        kwargs["null_topic"] = object_vectors.mean(axis=0)
    return request.param(n_exemplars=3, **kwargs)


def test_store_from_clusterer_feeds_the_exemplar_annotators(
    two_layer_clusterer, all_sentences, object_vectors, exemplar_annotator
):
    store = AnnotationStore.from_clusterer(
        two_layer_clusterer, all_sentences, object_vectors
    )

    failures = Executor(store).run(exemplar_annotator)

    assert failures == {}
    for node in store.tree.nodes:
        members = set(store.cluster_object_indices[node])
        indices = list(store.exemplar_original_indices[node])
        assert 0 < len(indices) <= 3
        assert len(set(indices)) == len(indices)
        assert set(indices) <= members
        assert list(store.exemplars[node]) == [all_sentences[i] for i in indices]


def test_store_from_clusterer_empty_cluster_has_nan_centroid_and_is_skipped(
    two_layer_clusterer,
    subtopic_label_vector,
    all_sentences,
    object_vectors,
    exemplar_annotator,
):
    emptied = subtopic_label_vector.copy()
    emptied[emptied == 0] = -1
    two_layer_clusterer.cluster_layers_[0].cluster_labels = emptied
    store = AnnotationStore.from_clusterer(
        two_layer_clusterer, all_sentences, object_vectors
    )

    empty_node = NodeId(0, 0)
    assert len(store.cluster_object_indices[empty_node]) == 0
    assert np.isnan(store.cluster_centroid[empty_node]).all()

    failures = Executor(store).run(exemplar_annotator)

    assert failures == {}
    assert list(store.exemplars[empty_node]) == []
    assert list(store.exemplar_original_indices[empty_node]) == []
