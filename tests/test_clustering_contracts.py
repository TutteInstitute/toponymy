"""Clustering ownership and partition invariants, without external estimators."""

from dataclasses import FrozenInstanceError
import os

from hypothesis import given, settings, strategies as st
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from toponymy.new_clustering import (
    PrecomputedClusterer,
    build_cluster_layers,
    build_cluster_tree,
    validate_cluster_tree,
)
from toponymy.types import Cluster, ClusterLayer

# Extended: 3 x 1,000 generated cases, each at most 5 x 32 integer labels.
# Database-free deterministic generation makes both profiles reproducible.
PROPERTY_EXAMPLES = {
    "quick": 100,
    "extended": 1000,
}[os.environ.get("TOPONYMY_PROPERTY_PROFILE", "quick")]
property_settings = settings(
    max_examples=PROPERTY_EXAMPLES,
    deadline=1000,
    derandomize=True,
    database=None,
)


@st.composite
def label_layers(draw):
    depth = draw(st.integers(0, 5))
    if depth == 0:
        return []
    n = draw(st.integers(0, 32))
    label = st.sampled_from([-1, 0, 2, 101, 2**40])
    return [
        np.array(draw(st.lists(label, min_size=n, max_size=n)), dtype=np.int64)
        for _ in range(depth)
    ]


def test_cluster_owns_members_and_has_no_naming_state():
    source = np.array([4, 2, 0])
    cluster = Cluster(51, source)
    source[:] = 19
    np.testing.assert_array_equal(cluster.members, [0, 2, 4])
    assert not cluster.members.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        cluster.members[0] = 7
    with pytest.raises(FrozenInstanceError):
        cluster.label = 1
    assert not hasattr(cluster, "name")
    assert not hasattr(cluster, "features")


def test_layer_owns_labels_and_orders_clusters():
    source = np.array([19, -1, 4, 19])
    clusters = [Cluster(19, [0, 3]), Cluster(4, [2])]
    layer = ClusterLayer(clusters, 0, source)
    source[:] = -1
    clusters.clear()
    assert isinstance(layer.clusters, tuple)
    assert [cluster.label for cluster in layer] == [4, 19]
    assert layer.cluster_labels is layer.labels
    np.testing.assert_array_equal(layer.labels, [19, -1, 4, 19])
    assert not layer.labels.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        layer.labels[:] = 0
    with pytest.raises(FrozenInstanceError):
        layer.labels = np.zeros(4, dtype=int)


@pytest.mark.parametrize(
    "label,members",
    [
        (-1, [0]),
        (True, [0]),
        (1.0, [0]),
        (2**64, [0]),
        (0, []),
        (0, [-1]),
        (0, [0, 0]),
        (0, [0.5]),
        (0, [[0]]),
    ],
)
def test_invalid_cluster(label, members):
    with pytest.raises(ValueError):
        Cluster(label, members)


@pytest.mark.parametrize(
    "clusters,index,labels,error",
    [
        ([object()], 0, [0], TypeError),
        ([Cluster(0, [0]), Cluster(0, [1])], 0, [0, 0], ValueError),
        ([Cluster(0, [0]), Cluster(1, [0])], 0, [0], ValueError),
        ([Cluster(0, [1])], 0, [0], ValueError),
        ([Cluster(0, [0])], 0, [-1], ValueError),
        ([], -1, [], ValueError),
        ([], False, [], ValueError),
    ],
)
def test_invalid_layer(clusters, index, labels, error):
    with pytest.raises(error):
        ClusterLayer(clusters, index, labels)


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


def test_precomputed_constructor_copies_and_fit_validates_vectors():
    labels = [np.array([42, -1, 42])]
    configured_tree = {(1, 0): [(0, 42)]}
    clusterer = PrecomputedClusterer(labels, configured_tree)
    labels[0][:] = 0
    configured_tree.clear()
    assert clusterer.fit(np.ones((3, 4))) is clusterer
    assert [cluster.label for cluster in next(iter(clusterer))] == [42]
    np.testing.assert_array_equal(clusterer.cluster_layers_[0].labels, [42, -1, 42])
    with pytest.raises(ValueError, match="observation count"):
        clusterer.fit(np.ones((4, 4)))
    clusterer.cluster_tree_.clear()
    clusterer.fit(np.ones((3, 5)))
    assert clusterer.cluster_tree_ == {(1, 0): [(0, 42)]}


def test_precomputed_fit_variants_and_clone():
    clusterer = PrecomputedClusterer()
    with pytest.raises(NotFittedError):
        iter(clusterer)
    with pytest.raises(ValueError, match="supply"):
        clusterer.fit()
    layers, tree = clusterer.fit_predict([[2, -1]])
    assert tree == {(1, 0): [(0, 2)]}
    assert len(layers) == 1
    clusterer.fit(np.zeros((2, 2)), labels=[[8, 8]])
    assert clusterer.cluster_tree_ == {(1, 0): [(0, 8)]}
    copied = clone(PrecomputedClusterer([[4, 4]]))
    copied.fit(np.ones((2, 1)))
    assert copied.cluster_tree_ == {(1, 0): [(0, 4)]}


@property_settings
@given(label_layers())
def test_generated_grouping_matches_independent_membership(labels):
    layers = build_cluster_layers(labels)
    for source, layer in zip(labels, layers):
        expected = {}
        for i, label in enumerate(source.tolist()):
            if label >= 0:
                expected.setdefault(label, []).append(i)
        assert [cluster.label for cluster in layer] == sorted(expected)
        for cluster in layer:
            assert cluster.members.tolist() == expected[cluster.label]
            assert not cluster.members.flags.writeable
        snapshot = source.copy()
        source[:] = -1
        np.testing.assert_array_equal(layer.labels, snapshot)


@property_settings
@given(label_layers())
def test_generated_tree_matches_set_containment_oracle(labels):
    partitions = []
    for values in labels:
        partition = {}
        for i, label in enumerate(values.tolist()):
            if label >= 0:
                partition.setdefault(label, set()).add(i)
        partitions.append(partition)
    expected = {}
    for lower, partition in enumerate(partitions):
        for label, members in partition.items():
            parent = (len(labels), 0)
            for upper in range(lower + 1, len(partitions)):
                containing = [
                    key
                    for key, candidate in partitions[upper].items()
                    if members <= candidate
                ]
                if containing:
                    parent = (upper, containing[0])
                    break
            expected[(lower, label)] = parent
    tree = build_cluster_tree(labels)
    actual = {child: parent for parent, children in tree.items() for child in children}
    assert actual == expected
    assert sum(map(len, tree.values())) == len(expected)
    validate_cluster_tree(tree, build_cluster_layers(labels))


@property_settings
@given(label_layers(), st.integers(1, 100))
def test_generated_id_relabelling_preserves_tree(labels, offset):
    renamed = [np.where(layer < 0, -1, 3 * layer + offset) for layer in labels]
    original = build_cluster_tree(labels)
    root = (len(labels), 0)

    def rename_key(key):
        return root if key == root else (key[0], 3 * key[1] + offset)

    expected = {
        rename_key(parent): [rename_key(child) for child in children]
        for parent, children in original.items()
    }
    assert build_cluster_tree(renamed) == expected
