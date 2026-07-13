from dataclasses import FrozenInstanceError
import numpy as np
import pytest
from toponymy.types import Cluster, ClusterLayer
from toponymy.new_clustering import build_cluster_layers, build_cluster_tree, validate_cluster_tree


@pytest.mark.parametrize("labels", [[], [np.array([], dtype=int)], [[-1, -1]]])
def test_empty_partitions(labels):
    layers = build_cluster_layers(labels)
    assert all(len(layer) == 0 for layer in layers)
    assert build_cluster_tree(labels) == {}
    validate_cluster_tree({}, layers)


@pytest.mark.parametrize(
    "labels",
    [
        [np.array([0, -2])],
        [np.array([True, False])],
        [np.array([0.0, 1.0])],
        [np.array([0.5])],
        [np.array([np.nan])],
        [np.array([np.inf])],
        [np.array([1 + 0j])],
        [np.array(["1"])],
        [np.array([0], dtype=object)],
        [np.array([2**64 - 1], dtype=np.uint64)],
        [np.zeros((2, 1), dtype=int)],
        [np.array([0]), np.array([0, 1])],
        np.array([0, 1]),
    ],
)
def test_label_validation(labels):
    for builder in (build_cluster_layers, build_cluster_tree):
        with pytest.raises(ValueError):
            builder(labels)


def test_sparse_noncontiguous_and_maximum_ids():
    maximum = np.iinfo(np.int64).max
    source = np.array([maximum, -1, 8, -1, maximum, -1], dtype=np.int64)[::2]
    layer = build_cluster_layers([source])[0]
    assert [cluster.label for cluster in layer] == [8, maximum]
    assert build_cluster_tree([source]) == {(1, 0): [(0, 8), (0, maximum)]}
    assert not np.shares_memory(layer.labels, source)
    np.testing.assert_array_equal(layer[1].members, [0, 2])


def test_crossing_and_partial_noise_skip_to_containing_ancestor():
    labels = [
        np.array([4, 4, 19, 19, -1, 81]),
        np.array([2, 3, 2, -1, 2, -1]),
        np.array([10, 10, 10, 10, 10, -1]),
    ]
    assert build_cluster_tree(labels) == {
        (2, 10): [(0, 4), (0, 19), (1, 2), (1, 3)],
        (3, 0): [(0, 81), (2, 10)],
    }


@pytest.mark.parametrize(
    "tree,message",
    [
        ({(3, 0): [(0, 1)]}, "unknown parent"),
        ({(2, 1): [(0, 1)]}, "unknown parent"),
        ({(2, 0): [(0, 99)]}, "unknown child"),
        ({(0, 1): [(1, 9)]}, "strictly above"),
        ({(0, 1): [(0, 1)], (2, 0): [(0, 2), (1, 9)]}, "strictly above"),
        ({(2, 0): [(0, 1), (0, 1), (0, 2), (1, 9)]}, "exactly one parent"),
        ({(1, 9): [(0, 1)]}, "does not contain"),
        ({}, "every cluster"),
        ({0: []}, "tuples"),
        ({(2, 0): [0]}, "tuples"),
        ({(2, 0): [(False, 1)]}, "integer"),
        ({(2, 0): {(0, 1)}}, "list or tuple"),
    ],
)
def test_invalid_trees(tree, message):
    layers = build_cluster_layers([[1, 1, 2], [9, -1, 9]])
    with pytest.raises(ValueError, match=message):
        validate_cluster_tree(tree, layers)


def test_tree_validation_rejects_bad_layers_and_mapping():
    with pytest.raises(TypeError, match="mapping"):
        validate_cluster_tree([], [])
    with pytest.raises(ValueError, match="consecutive"):
        validate_cluster_tree({}, [ClusterLayer([], 1, [-1])])
    with pytest.raises(ValueError, match="consecutive"):
        validate_cluster_tree({}, [object()])
    with pytest.raises(ValueError, match="same observation"):
        validate_cluster_tree({}, [ClusterLayer([], 0, [-1]), ClusterLayer([], 1, [])])
