import pytest

from toponymy.annotation import (
    InvalidAnnotationTree,
    NodeId,
    AnnotationTree,
    Annotation,
    AnnotationState,
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

    with pytest.raises(InvalidAnnotationTree, match="No root found"):
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
