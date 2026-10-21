"""Tests for the cluster types, the hierarchy helpers and EVoCClusterer.

EVoCClusterer is tested here against a stand-in for ``evoc.EVoC``, so these tests
run whether or not EVoC is installed. The real estimator is covered by
``test_clusterers_evoc.py``.
"""

import builtins
import copy
from dataclasses import FrozenInstanceError
import pickle
import sys
from types import SimpleNamespace

from hypothesis import HealthCheck, given, settings, strategies as st
import numpy as np
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from toponymy.clusterers import (
    Cluster,
    ClusterLayer,
    EVoCClusterer,
    build_cluster_layers,
    build_cluster_tree,
    validate_cluster_tree,
)

# Deterministic and database-free, so that every run generates the same cases.
# Timing is not under test, so a slow or busy runner must not fail these tests.
property_settings = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
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
    "duplicate",
    [copy.copy, copy.deepcopy, lambda x: pickle.loads(pickle.dumps(x))],
    ids=["copy", "deepcopy", "pickle"],
)
def test_copies_of_a_layer_are_owned_and_read_only(duplicate):
    layer = build_cluster_layers([np.array([19, -1, 4, 19])])[0]
    duplicated = duplicate(layer)
    assert duplicated is not layer
    assert duplicated.layer_index == layer.layer_index
    assert [cluster.label for cluster in duplicated] == [4, 19]
    np.testing.assert_array_equal(duplicated.labels, layer.labels)
    np.testing.assert_array_equal(duplicated[1].members, [0, 3])
    assert not duplicated.labels.flags.writeable
    assert not duplicated[1].members.flags.writeable


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
        # Cluster 0 claims observation 1, which the labels give to cluster 1.
        ([Cluster(0, [0, 1]), Cluster(1, [1])], 0, [0, 1], ValueError),
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


def test_a_single_layer_must_be_wrapped_in_a_sequence():
    with pytest.raises(ValueError, match="sequence of one-dimensional"):
        build_cluster_layers(np.array([0, 1, 1]))


@pytest.mark.parametrize(
    "dtype",
    [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64],
)
def test_integer_labels_of_any_width_are_stored_as_int64(dtype):
    layer = build_cluster_layers([np.array([3, 0, 3, 7], dtype=dtype)])[0]
    assert layer.labels.dtype == np.int64
    assert [cluster.label for cluster in layer] == [0, 3, 7]


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


def test_grouping_preserves_original_ids_and_member_order():
    labels = [np.array([19, -1, 4, 19, 4, -1])]
    layer = build_cluster_layers(labels)[0]
    assert layer.layer_index == 0
    assert [cluster.label for cluster in layer] == [4, 19]
    np.testing.assert_array_equal(layer[0].members, [2, 4])
    np.testing.assert_array_equal(layer[1].members, [0, 3])


def test_nested_tree_attaches_clusters_to_nearest_parent():
    labels = [
        np.array([0, 0, 1, 1, 2, 2, 3, 3]),
        np.array([0, 0, 0, 0, 1, 1, 1, 1]),
        np.zeros(8, dtype=int),
    ]
    tree = build_cluster_tree(labels)
    assert set(tree[(1, 0)]) == {(0, 0), (0, 1)}
    assert set(tree[(1, 1)]) == {(0, 2), (0, 3)}
    assert set(tree[(2, 0)]) == {(1, 0), (1, 1)}
    assert tree[(3, 0)] == [(2, 0)]


def test_separate_builds_do_not_share_member_arrays():
    labels = [np.array([1, 0, 1, 0])]
    first = build_cluster_layers(labels)
    second = build_cluster_layers(labels)
    assert not np.shares_memory(first[0][0].members, second[0][0].members)
    assert not np.shares_memory(first[0][0].members, labels[0])


@property_settings
@given(label_layers())
def test_generated_grouping_matches_independent_membership(labels):
    layers = build_cluster_layers(labels)
    assert len(layers) == len(labels)
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


@pytest.fixture
def evoc_double(monkeypatch):
    """Install a stand-in for ``evoc.EVoC`` and return its class.

    The class lists the instances created. Set ``layers`` or ``error`` on it to
    change what a fit returns or raises.
    """

    class EVoCDouble:
        instances = []
        layers = [
            np.array([4, 4, 19, 19, -1, -1]),
            np.array([2, 3, 2, 2, -1, -1]),
        ]
        error = None

        def __init__(self, **options):
            self.options = options
            self.fitted_on = []
            type(self).instances.append(self)

        def fit(self, vectors):
            self.fitted_on.append(vectors)
            if self.error is not None:
                raise self.error
            self.cluster_layers_ = [layer.copy() for layer in self.layers]
            # EVoC's own tree, which EVoCClusterer must not use.
            self.cluster_tree_ = {(1, 2): [(0, 4), (0, 19)]}
            return self

    monkeypatch.setitem(sys.modules, "evoc", SimpleNamespace(EVoC=EVoCDouble))
    return EVoCDouble


def test_evoc_clusterer_forwards_all_parameters_and_keeps_native_ids(evoc_double):
    vectors = np.ones((6, 2))
    clusterer = EVoCClusterer(random_state=31, approx_n_clusters=2, n_neighbors=8)
    assert clusterer.fit(vectors) is clusterer

    (native,) = evoc_double.instances
    assert native.options == clusterer.get_params()
    assert native.fitted_on[0] is vectors
    assert clusterer.evoc_ is native
    assert [[cluster.label for cluster in layer] for layer in clusterer] == [
        [4, 19],
        [2, 3],
    ]


@pytest.mark.usefixtures("evoc_double")
def test_evoc_clusterer_builds_its_tree_from_the_layers():
    clusterer = EVoCClusterer().fit(np.ones((6, 2)))
    # Cluster 4 straddles clusters 2 and 3 of the layer above, so it skips to the
    # root, although EVoC's own tree puts it under (1, 2).
    assert clusterer.cluster_tree_ == {
        (1, 2): [(0, 19)],
        (2, 0): [(0, 4), (1, 2), (1, 3)],
    }
    validate_cluster_tree(clusterer.cluster_tree_, clusterer.cluster_layers_)


@pytest.mark.usefixtures("evoc_double")
def test_evoc_clusterer_layers_do_not_share_memory_with_evoc():
    clusterer = EVoCClusterer().fit(np.ones((6, 2)))
    expected = [layer.labels.copy() for layer in clusterer]
    for labels in clusterer.evoc_.cluster_layers_:
        labels[:] = -1
    for layer, labels in zip(clusterer, expected):
        np.testing.assert_array_equal(layer.labels, labels)


@pytest.mark.parametrize(
    "options,runs_evoc",
    [
        ({}, True),
        ({"min_samples": 6}, True),
        ({"min_samples": 7}, False),
        ({"base_min_cluster_size": 6}, True),
        ({"base_min_cluster_size": 7}, False),
    ],
)
def test_evoc_runs_only_on_enough_observations(evoc_double, options, runs_evoc):
    clusterer = EVoCClusterer(**options).fit(np.ones((6, 2)))
    assert bool(evoc_double.instances) is runs_evoc
    assert (clusterer.evoc_ is not None) is runs_evoc
    if not runs_evoc:
        assert [layer.labels.tolist() for layer in clusterer] == [[-1] * 6]
        assert clusterer.cluster_tree_ == {}


@pytest.mark.parametrize(
    "n_observations,expected_labels", [(0, []), (1, [[-1]])], ids=["empty", "single"]
)
def test_empty_and_tiny_input_does_not_import_evoc(
    monkeypatch, n_observations, expected_labels
):
    monkeypatch.setitem(sys.modules, "evoc", None)
    clusterer = clone(EVoCClusterer())
    assert clusterer.fit(np.ones((n_observations, 2))) is clusterer
    assert [layer.labels.tolist() for layer in clusterer] == expected_labels
    assert clusterer.cluster_tree_ == {}
    assert clusterer.evoc_ is None


@pytest.mark.parametrize("n_observations", [0, 1], ids=["empty", "single"])
def test_refitting_on_tiny_input_drops_the_previous_evoc(evoc_double, n_observations):
    clusterer = EVoCClusterer().fit(np.ones((6, 2)))
    assert clusterer.evoc_ is not None

    clusterer.fit(np.ones((n_observations, 2)))
    assert clusterer.evoc_ is None
    assert len(evoc_double.instances) == 1


def test_constructor_stores_every_parameter_unchanged():
    options = {
        "noise_level": 0.25,
        "base_min_cluster_size": 7,
        "base_n_clusters": 30,
        "approx_n_clusters": 12,
        "n_neighbors": np.int64(9),
        "min_samples": 3,
        "n_epochs": 11,
        "node_embedding_init": None,
        "symmetrize_graph": False,
        "node_embedding_dim": 6,
        "neighbor_scale": 1.5,
        "random_state": np.random.RandomState(0),
        "min_similarity_threshold": 0.4,
        "max_layers": 4,
        "n_label_prop_iter": 8,
    }
    assert options.keys() == EVoCClusterer().get_params().keys()

    clusterer = EVoCClusterer(**options)
    assert all(getattr(clusterer, name) is value for name, value in options.items())
    assert clusterer.get_params() == options


def test_sparse_vectors_are_rejected_with_an_explanation():
    with pytest.raises(ValueError, match="sparse"):
        EVoCClusterer().fit(sparse.eye(6))


@pytest.mark.parametrize(
    "vectors",
    [
        np.array([1.0, 2.0]),
        np.zeros((2, 0)),
        np.zeros((1, 2, 3)),
        np.array([[np.nan]]),
        np.array([[np.inf]]),
        np.array([[1 + 0j]]),
        np.array([[True]]),
        np.array([["1"]]),
        np.array([[0]], dtype=object),
        sparse.eye(3),
        # EVoC supports float, int8 and uint8 vectors, but no other integer type.
        np.ones((6, 2), dtype=np.int16),
        np.ones((6, 2), dtype=np.int64),
        np.ones((6, 2), dtype=np.uint32),
    ],
)
def test_invalid_vectors_are_rejected_before_evoc_runs(evoc_double, vectors):
    with pytest.raises(ValueError):
        EVoCClusterer().fit(vectors)
    assert not evoc_double.instances


@pytest.mark.parametrize(
    "dtype", [np.float16, np.float32, np.float64, np.int8, np.uint8]
)
def test_vectors_of_every_dtype_evoc_supports_reach_it_as_given(evoc_double, dtype):
    vectors = np.ones((6, 2), dtype=dtype)
    EVoCClusterer().fit(vectors)
    assert evoc_double.instances[0].fitted_on[0] is vectors


@pytest.mark.parametrize(
    "options",
    [
        {"min_samples": 0},
        {"base_min_cluster_size": 1},
        {"max_layers": 0},
        {"max_layers": None},
        {"min_samples": True},
        {"max_layers": 1.5},
    ],
)
def test_invalid_parameters_are_not_masked_by_empty_data(options):
    with pytest.raises(ValueError):
        EVoCClusterer(**options).fit(np.empty((0, 2)))


def test_missing_evoc_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "evoc", None)
    with pytest.raises(ImportError, match=r"pip install .*evoc>=0\.3\.2"):
        EVoCClusterer().fit(np.ones((6, 2)))


def test_a_missing_dependency_of_evoc_is_not_disguised(monkeypatch):
    original_import = builtins.__import__

    def import_with_missing_dependency(name, *args, **kwargs):
        if name == "evoc":
            raise ModuleNotFoundError("EVoC dependency is missing", name="numba")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_with_missing_dependency)
    with pytest.raises(ModuleNotFoundError, match="dependency is missing"):
        EVoCClusterer().fit(np.ones((6, 2)))


def _fitted_state(clusterer):
    return clusterer.cluster_layers_, clusterer.cluster_tree_, clusterer.evoc_


@pytest.mark.parametrize(
    "attribute,value,error",
    [
        ("error", RuntimeError("external implementation failed"), RuntimeError),
        ("layers", [np.array([0.5, 1.5, 0, 0, 0, 0])], ValueError),
    ],
    ids=["evoc-fails", "evoc-returns-non-integer-labels"],
)
def test_failed_fit_propagates_and_keeps_the_previous_fit(
    evoc_double, attribute, value, error
):
    clusterer = EVoCClusterer().fit(np.ones((6, 2)))
    before = _fitted_state(clusterer)

    setattr(evoc_double, attribute, value)
    with pytest.raises(error) as caught:
        clusterer.fit(np.ones((6, 2)))

    if attribute == "error":
        assert caught.value is value
    assert all(old is new for old, new in zip(before, _fitted_state(clusterer)))


@pytest.mark.usefixtures("evoc_double")
def test_rejected_vectors_and_parameters_keep_the_previous_fit():
    clusterer = EVoCClusterer().fit(np.ones((6, 2)))
    before = _fitted_state(clusterer)

    with pytest.raises(ValueError):
        clusterer.fit(np.full((6, 2), np.nan))
    with pytest.raises(ValueError):
        clusterer.set_params(max_layers=0).fit(np.ones((6, 2)))

    assert all(old is new for old, new in zip(before, _fitted_state(clusterer)))


def test_each_fit_uses_a_new_evoc_and_leaves_the_previous_one_alone(evoc_double):
    clusterer = EVoCClusterer()
    first = clusterer.fit(np.ones((6, 2))).evoc_
    first_layers = clusterer.cluster_layers_
    second = clusterer.fit(np.ones((6, 2))).evoc_

    assert evoc_double.instances == [first, second]
    assert len(first.fitted_on) == 1
    assert clusterer.cluster_layers_ is not first_layers


@pytest.mark.usefixtures("evoc_double")
def test_fit_predict_returns_the_fitted_layers_and_tree():
    clusterer = EVoCClusterer()
    layers, tree = clusterer.fit_predict(np.ones((6, 2)))
    assert layers is clusterer.cluster_layers_
    assert tree is clusterer.cluster_tree_


@pytest.mark.usefixtures("evoc_double")
def test_unfitted_clusterer_is_not_iterable_and_clones_are_unfitted():
    clusterer = EVoCClusterer(random_state=3)
    with pytest.raises(NotFittedError):
        iter(clusterer)
    # A clusterer is fitted once it has both its layers and its tree.
    clusterer.cluster_layers_ = []
    with pytest.raises(NotFittedError):
        iter(clusterer)

    clusterer.fit(np.ones((6, 2)))
    assert [len(layer) for layer in clusterer] == [2, 2]

    cloned = clone(clusterer)
    assert cloned.get_params() == clusterer.get_params()
    assert not hasattr(cloned, "cluster_layers_")
    with pytest.raises(NotFittedError):
        iter(cloned)
