"""Tests for PrecomputedClusterer."""

import copy
import pickle

import numpy as np
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from toponymy.clusterers import PrecomputedClusterer

# Layers that do not nest, with gapped IDs and noise. Cluster 4 straddles
# clusters 2 and 3 of layer 1, and cluster 19 is partly noise there, so both
# skip to cluster 10 of layer 2. Cluster 81 is noise above, so it is a child of
# the root.
LABELS = [
    np.array([4, 4, 19, 19, -1, 81]),
    np.array([2, 3, 2, -1, 2, -1]),
    np.array([10, 10, 10, 10, 10, -1]),
]
TREE = {
    (2, 10): [(0, 4), (0, 19), (1, 2), (1, 3)],
    (3, 0): [(0, 81), (2, 10)],
}
VECTORS = np.ones((6, 2))
EXPECTED_LABELS = [labels.tolist() for labels in LABELS]
CLUSTERS = [(0, 4), (0, 19), (0, 81), (1, 2), (1, 3), (2, 10)]


def layer_labels(clusterer):
    return [layer.labels.tolist() for layer in clusterer]


def at_root_except(*placed):
    """The clusters of LABELS that are not in ``placed``, for the root."""
    return [cluster for cluster in CLUSTERS if cluster not in placed]


@pytest.mark.parametrize("cluster_tree", [None, TREE], ids=["built", "given-tree"])
@pytest.mark.parametrize(
    "configured,args,kwargs",
    [
        (LABELS, (VECTORS,), {}),
        (LABELS, (), {}),
        (None, (), {"labels": LABELS}),
        (None, (VECTORS,), {"labels": LABELS}),
        # Labels given to fit replace the configured ones.
        ([[0] * 6], (VECTORS,), {"labels": LABELS}),
    ],
    ids=[
        "configured",
        "no-vectors",
        "given-no-vectors",
        "given",
        "given-replaces-configured",
    ],
)
def test_fit_and_fit_predict_accept_the_labels_in_every_form(
    configured, args, kwargs, cluster_tree
):
    clusterer = PrecomputedClusterer(configured, cluster_tree)
    assert clusterer.fit(*args, **kwargs) is clusterer
    assert layer_labels(clusterer) == EXPECTED_LABELS
    assert clusterer.cluster_tree_ == TREE

    first_layers = clusterer.cluster_layers_
    layers, tree = clusterer.fit_predict(*args, **kwargs)
    assert layers is clusterer.cluster_layers_
    assert layers is not first_layers
    assert tree is clusterer.cluster_tree_
    assert layer_labels(clusterer) == EXPECTED_LABELS
    assert tree == TREE


def test_the_vectors_are_never_read_as_labels():
    clusterer = PrecomputedClusterer()
    with pytest.raises(NotFittedError):
        iter(clusterer)
    # Integer vectors would also be valid label layers, one per row.
    for args in [(), (VECTORS,), (np.ones((6, 4), dtype=np.uint8),)]:
        with pytest.raises(ValueError, match="supply precomputed labels"):
            clusterer.fit(*args)
        with pytest.raises(ValueError, match="supply precomputed labels"):
            clusterer.fit_predict(*args)
    assert not hasattr(clusterer, "cluster_layers_")


def test_cluster_ids_and_observation_order_are_kept():
    maximum = np.iinfo(np.int64).max
    labels = [np.array([maximum, -1, 7, maximum, 7])]
    clusterer = PrecomputedClusterer(labels).fit()
    (layer,) = clusterer.cluster_layers_
    assert [cluster.label for cluster in layer] == [7, maximum]
    np.testing.assert_array_equal(layer[0].members, [2, 4])
    np.testing.assert_array_equal(layer[1].members, [0, 3])
    np.testing.assert_array_equal(layer.labels, labels[0])
    assert clusterer.cluster_tree_ == {(1, 0): [(0, 7), (0, maximum)]}

    given = {(1, 0): [(0, maximum), (0, 7)]}
    assert PrecomputedClusterer(labels, given).fit().cluster_tree_ == given


def test_the_parameters_are_stored_as_given():
    labels, tree = [np.array([42, -1, 42])], {(1, 0): [(0, 42)]}
    clusterer = PrecomputedClusterer(labels, tree)
    params = clusterer.get_params()
    assert params.keys() == {"labels", "cluster_tree"}
    assert params["labels"] is clusterer.labels is labels
    assert params["cluster_tree"] is clusterer.cluster_tree is tree

    labels, tree = [np.array([3, 3, -1])], {(1, 0): [(0, 3)]}
    assert clusterer.set_params(labels=labels, cluster_tree=tree) is clusterer
    assert clusterer.labels is labels
    assert clusterer.cluster_tree is tree


def test_changes_to_the_parameters_before_fit_are_used():
    labels, tree = [np.array([42, -1, 42])], {(1, 0): [(0, 42)]}
    clusterer = PrecomputedClusterer(labels, tree)
    labels[0][:] = [5, 5, -1]
    tree[(1, 0)][0] = (0, 5)

    clusterer.fit(np.ones((3, 4)))
    assert layer_labels(clusterer) == [[5, 5, -1]]
    assert clusterer.cluster_tree_ == {(1, 0): [(0, 5)]}


@pytest.mark.parametrize("configured", [True, False], ids=["configured", "given"])
def test_the_fitted_layers_do_not_share_memory_with_the_labels(configured):
    labels = copy.deepcopy(LABELS)
    if configured:
        clusterer = PrecomputedClusterer(labels).fit(VECTORS)
    else:
        clusterer = PrecomputedClusterer().fit(VECTORS, labels=labels)
    for array in labels:
        array[:] = -1

    assert layer_labels(clusterer) == EXPECTED_LABELS
    assert not any(layer.labels.flags.writeable for layer in clusterer)


def test_the_fitted_tree_shares_no_lists_with_the_given_tree():
    tree = copy.deepcopy(TREE)
    clusterer = PrecomputedClusterer(LABELS, tree).fit(VECTORS)
    for children in clusterer.cluster_tree_.values():
        children.clear()
    assert tree == TREE

    clusterer.fit(VECTORS)
    for children in tree.values():
        children.clear()
    assert clusterer.cluster_tree_ == TREE


def test_a_given_tree_is_kept_as_given():
    # Valid, but not the tree that build_cluster_tree makes: clusters 4 and 81
    # are children of the root, and the order differs.
    given = {
        (3, 0): ((0, 4), (np.int64(2), np.int64(10)), (0, 81)),
        (np.int64(2), np.int64(10)): [(1, 3), (0, 19), (1, 2)],
    }
    tree = PrecomputedClusterer(LABELS, given).fit(VECTORS).cluster_tree_

    assert tree == {
        (3, 0): [(0, 4), (2, 10), (0, 81)],
        (2, 10): [(1, 3), (0, 19), (1, 2)],
    }
    assert list(tree) == [(3, 0), (2, 10)]
    assert all(type(children) is list for children in tree.values())
    nodes = [*tree, *(child for children in tree.values() for child in children)]
    assert all(type(value) is int for node in nodes for value in node)


def test_parents_without_children_are_left_out_of_a_given_tree():
    # As in a built tree, so an empty tree is {} whether it was built or given.
    given = {**copy.deepcopy(TREE), (0, 4): [], (1, 3): ()}
    assert PrecomputedClusterer(LABELS, given).fit().cluster_tree_ == TREE
    assert PrecomputedClusterer([[-1, -1]], {(1, 0): []}).fit().cluster_tree_ == {}


def test_fit_does_not_change_the_given_tree():
    given = {
        (3, 0): ((0, 4), (2, 10), (0, 81)),
        (2, 10): [(1, 3), (0, 19), (1, 2)],
        (0, 19): [],
    }
    expected = copy.deepcopy(given)
    PrecomputedClusterer(LABELS, given).fit(VECTORS)
    assert given == expected


@pytest.mark.parametrize(
    "tree,error,message",
    [
        (
            {(1, 2): [(0, 4)], (3, 0): at_root_except((0, 4))},
            ValueError,
            "does not contain",
        ),
        (
            {(1, 2): [(0, 19)], (3, 0): at_root_except((0, 19))},
            ValueError,
            "does not contain",
        ),
        (
            {
                (1, 2): [(2, 10)],
                (2, 10): [(1, 2)],
                (3, 0): at_root_except((1, 2), (2, 10)),
            },
            ValueError,
            "strictly above",
        ),
        ({(3, 0): CLUSTERS + [(0, 4)]}, ValueError, "exactly one parent"),
        ({(3, 0): CLUSTERS + [(0, 5)]}, ValueError, "unknown child"),
        ({(4, 0): CLUSTERS}, ValueError, "unknown parent"),
        ({(3, 0): at_root_except((2, 10))}, ValueError, "exactly once"),
        ({}, ValueError, "exactly once"),
        ({(3, 0): set(CLUSTERS)}, ValueError, "list or tuple"),
        (list(TREE.items()), TypeError, "mapping"),
    ],
    ids=[
        "crossing",
        "partly-noise",
        "cycle",
        "repeated",
        "unknown-child",
        "unknown-root",
        "missing",
        "empty",
        "set-of-children",
        "not-a-mapping",
    ],
)
def test_an_invalid_tree_is_rejected_and_not_replaced(tree, error, message):
    clusterer = PrecomputedClusterer(LABELS, tree)
    with pytest.raises(error, match=message):
        clusterer.fit(VECTORS)
    assert not hasattr(clusterer, "cluster_tree_")


@pytest.mark.parametrize(
    "labels,vectors,tree",
    [
        ([], np.ones((3, 2)), {}),
        ([[]], np.empty((0, 2)), {}),
        ([[-1, -1, -1]], np.ones((3, 2)), {}),
        (
            [[0, 0, 1], [-1, -1, -1], [7, 7, 7]],
            np.ones((3, 2)),
            {(2, 7): [(0, 0), (0, 1)], (3, 0): [(2, 7)]},
        ),
    ],
    ids=["no-layers", "no-observations", "all-noise", "all-noise-middle-layer"],
)
def test_empty_and_all_noise_layers(labels, vectors, tree):
    for given in (None, tree):
        clusterer = PrecomputedClusterer(labels, given).fit(vectors)
        assert layer_labels(clusterer) == labels
        assert clusterer.cluster_tree_ == tree


@pytest.mark.parametrize(
    "vectors,message",
    [
        (np.ones((5, 2)), "same observation count"),
        (np.ones(6), "shape"),
        (np.ones((6, 0)), "shape"),
        (np.full((6, 2), np.nan), "finite"),
        (np.full((6, 2), -np.inf), "finite"),
        (sparse.csr_matrix(VECTORS), "dense"),
    ],
)
def test_invalid_vectors_are_rejected(vectors, message):
    with pytest.raises(ValueError, match=message):
        PrecomputedClusterer(LABELS).fit(vectors)


@pytest.mark.parametrize(
    "vectors", [None, np.ones((3, 2))], ids=["no-vectors", "vectors"]
)
@pytest.mark.parametrize(
    "labels,message",
    [
        ([np.array([0, -2, 0])], "between -1"),
        ([np.array([0.0, 1.0, 1.0])], "integers"),
        ([np.array([True, False, True])], "integers"),
        (np.array([0, 1, 1]), "sequence of one-dimensional"),
        ([[0, 1, 1], [0, 1]], "all label layers"),
    ],
)
def test_invalid_labels_are_rejected_by_fit(labels, message, vectors):
    clusterer = PrecomputedClusterer(labels)
    with pytest.raises(ValueError, match=message):
        clusterer.fit(vectors)


@pytest.mark.parametrize(
    "refit",
    [
        lambda clusterer: clusterer.fit(np.ones((5, 2))),
        lambda clusterer: clusterer.fit(np.full((6, 2), np.nan)),
        lambda clusterer: clusterer.fit(VECTORS, labels=[[0.5] * 6]),
        lambda clusterer: clusterer.fit(labels=[[0.5] * 6]),
        # The tree belongs to the configured labels, not to these.
        lambda clusterer: clusterer.fit(VECTORS, labels=[[0] * 6]),
        lambda clusterer: clusterer.set_params(cluster_tree={}).fit(VECTORS),
    ],
    ids=[
        "observation-count",
        "nonfinite-vectors",
        "invalid-labels",
        "invalid-labels-no-vectors",
        "tree-of-other-labels",
        "invalid-tree",
    ],
)
def test_a_failed_fit_keeps_the_previous_fit(refit):
    clusterer = PrecomputedClusterer(LABELS, TREE).fit(VECTORS)
    layers, tree = clusterer.cluster_layers_, clusterer.cluster_tree_

    with pytest.raises(ValueError):
        refit(clusterer)
    assert clusterer.cluster_layers_ is layers
    assert clusterer.cluster_tree_ is tree
    assert tree == TREE


def test_refitting_with_new_parameters_replaces_the_result():
    clusterer = PrecomputedClusterer(LABELS, TREE).fit(VECTORS)
    first_layers, first_tree = clusterer.cluster_layers_, clusterer.cluster_tree_

    clusterer.set_params(labels=[[3, 3, 3, -1, -1, -1]], cluster_tree=None)
    assert clusterer.fit(VECTORS).cluster_tree_ == {(1, 0): [(0, 3)]}
    assert layer_labels(clusterer) == [[3, 3, 3, -1, -1, -1]]
    assert [layer.labels.tolist() for layer in first_layers] == EXPECTED_LABELS
    assert first_tree == TREE


def test_a_clone_has_the_parameters_but_not_the_fit():
    clusterer = PrecomputedClusterer(LABELS, TREE).fit(VECTORS)
    cloned = clone(clusterer)
    assert not hasattr(cloned, "cluster_layers_")
    with pytest.raises(NotFittedError):
        iter(cloned)

    assert [labels.tolist() for labels in cloned.labels] == EXPECTED_LABELS
    assert cloned.cluster_tree == TREE


def test_a_clone_shares_no_parameters_with_the_original():
    labels, tree = copy.deepcopy(LABELS), copy.deepcopy(TREE)
    cloned = clone(PrecomputedClusterer(labels, tree))
    for array in labels:
        array[:] = -1
    for children in tree.values():
        children.clear()

    assert layer_labels(cloned.fit(VECTORS)) == EXPECTED_LABELS
    assert cloned.cluster_tree_ == TREE


@pytest.mark.parametrize(
    "duplicate",
    [copy.deepcopy, lambda clusterer: pickle.loads(pickle.dumps(clusterer))],
    ids=["deepcopy", "pickle"],
)
def test_copies_of_a_fitted_clusterer_share_no_data(duplicate):
    clusterer = PrecomputedClusterer(LABELS, TREE).fit(VECTORS)
    duplicated = duplicate(clusterer)

    assert duplicated.cluster_tree_ == TREE
    assert duplicated.cluster_tree_ is not clusterer.cluster_tree_
    for layer, copied in zip(clusterer, duplicated):
        np.testing.assert_array_equal(copied.labels, layer.labels)
        assert not np.shares_memory(copied.labels, layer.labels)
        assert not copied.labels.flags.writeable
