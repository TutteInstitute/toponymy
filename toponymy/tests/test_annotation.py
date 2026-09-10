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
    # Root is (2, 0); non-root nodes are layers 1 and 0
    return {
        (1, 0): [(0, 0)],
        (1, 1): [(0, 1)],
        (1, 2): [(0, 7), (0, 6), (0, 5)],
        (1, 3): [(0, 4)],
        (2, 0): [(1, 0), (1, 1), (1, 2), (1, 3), (0, 2), (0, 3)],
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
        assert tree.children(parent) == sorted(children)

    for node in tree.layer(0):
        assert tree.children(node) is None


def test_tree_parent(valid_cluster_tree):
    tree = AnnotationTree(valid_cluster_tree)
    for parent, children in valid_cluster_tree.items():
        for child in children:
            assert tree.parent(child) == parent

    assert tree.parent(tree._root) is None


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


def test_tree_descendants_root(valid_cluster_tree):
    tree = AnnotationTree(valid_cluster_tree)

    descendants = tree.descendants(tree._root)
    assert set(descendants) == set(tree.nodes)


def test_tree_descendants_depth_limited(valid_cluster_tree):
    tree = AnnotationTree(valid_cluster_tree)
    root = tree._root

    depth_1 = tree.descendants(root, depth=1)
    assert set(depth_1) == {
        NodeId(1, 0),
        NodeId(1, 1),
        NodeId(1, 2),
        NodeId(1, 3),
        NodeId(0, 2),
        NodeId(0, 3),
    }

    depth_2 = tree.descendants(root, depth=2)
    assert set(depth_2) == {
        NodeId(1, 0),
        NodeId(1, 1),
        NodeId(1, 2),
        NodeId(1, 3),
        NodeId(0, 0),
        NodeId(0, 1),
        NodeId(0, 2),
        NodeId(0, 3),
        NodeId(0, 4),
        NodeId(0, 5),
        NodeId(0, 6),
        NodeId(0, 7),
    }


def test_tree_descendants_leaf_and_zero_depth(valid_cluster_tree):
    tree = AnnotationTree(valid_cluster_tree)
    assert tree.descendants(NodeId(0, 0)) == []
    assert tree.descendants(NodeId(1, 2), depth=0) == []


def test_tree_descendants_negative_depth_raises(valid_cluster_tree):
    tree = AnnotationTree(valid_cluster_tree)
    with pytest.raises(ValueError, match="depth must be >= 0 or None"):
        tree.descendants(NodeId(1, 2), depth=-1)


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
    with pytest.raises(KeyError, match="is not a node of the cluster tree"):
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
