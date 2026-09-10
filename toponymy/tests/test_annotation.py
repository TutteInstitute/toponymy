import pytest

from toponymy.annotation import InvalidAnnotationTree, NodeId, AnnotationTree


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
    assert tree.nodes == sorted(
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
    assert len(tree.nodes) == 12


def test_tree_layer_has_no_root_layer(valid_cluster_tree):
    tree = AnnotationTree(valid_cluster_tree)

    assert tree.layer(0) == [
        NodeId(0, 0),
        NodeId(0, 1),
        NodeId(0, 2),
        NodeId(0, 3),
        NodeId(0, 4),
        NodeId(0, 5),
        NodeId(0, 6),
        NodeId(0, 7),
    ]
    assert tree.layer(1) == [NodeId(1, 0), NodeId(1, 1), NodeId(1, 2), NodeId(1, 3)]
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
