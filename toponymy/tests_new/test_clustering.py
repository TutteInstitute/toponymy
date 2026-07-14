import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_mutual_info_score, pairwise_distances
from sklearn.utils.validation import check_is_fitted

from toponymy.new_types import Cluster, ClusterLayer

from toponymy.new_clustering import (
    PrecomputedClusterer,
    KMeansClusterer,
    PLSCANClusterer,
    EVoCClusterer,
    build_cluster_tree,
    _build_cluster_tree,
    build_cluster_layers,
)


def layer_to_labels_helper(layer: ClusterLayer, n_points: int):
    labels = np.full(n_points, -1, dtype="int")
    for i, cluster in enumerate(layer.clusters):
        labels[cluster.members] = i
    return labels


@pytest.fixture
def label_layers():
    return [
        np.array([0, 0, -1, -1, 1, 1, -1]),
        np.array([0, 0, 0, 1, 1, 1, -1]),
        np.array([0, 0, 0, 0, 0, 0, -1]),
    ]


def test_build_cluster_tree(label_layers):
    cluster_tree = build_cluster_tree(label_layers)
    # Assert that every child cluster is contained in the parent cluster
    for parent, children in cluster_tree.items():
        for child in children:
            assert parent[0] > child[0]
            # Skip if parent is the artificial root node
            if parent[0] >= len(label_layers):
                continue
            np.testing.assert_array_equal(
                (label_layers[parent[0]] == parent[1])
                & (label_layers[child[0]] == child[1]),
                label_layers[child[0]] == child[1],
            )
    # Assert that every cluster is contained in the tree
    clusters_in_tree = sum(cluster_tree.values(), [])
    assert len(set(clusters_in_tree)) == len(clusters_in_tree)
    for i in range(len(label_layers)):
        for j in range(label_layers[i].max() + 1):
            if i == 0:
                assert (i, j) in clusters_in_tree
            else:
                assert (i, j) in clusters_in_tree and (i, j) in cluster_tree


def test_build_cluster_tree_no_jit(label_layers):
    cluster_tree = {}
    raw_mapping = _build_cluster_tree.py_func(np.vstack(label_layers))
    for parent_layer, parent_cluster, child_layer, child_cluster in raw_mapping:
        parent_name = (parent_layer, parent_cluster)
        if parent_name in cluster_tree:
            cluster_tree[parent_name].append((child_layer, child_cluster))
        else:
            cluster_tree[parent_name] = [(child_layer, child_cluster)]

    # Assert that every child cluster is contained in the parent cluster
    for parent, children in cluster_tree.items():
        for child in children:
            assert parent[0] > child[0]
            # Skip if parent is the artificial root node
            if parent[0] >= len(label_layers):
                continue
            np.testing.assert_array_equal(
                (label_layers[parent[0]] == parent[1])
                & (label_layers[child[0]] == child[1]),
                label_layers[child[0]] == child[1],
            )

    # Assert that every cluster is contained in the tree
    clusters_in_tree = sum(cluster_tree.values(), [])
    assert len(set(clusters_in_tree)) == len(clusters_in_tree)
    for i in range(len(label_layers)):
        for j in range(label_layers[i].max() + 1):
            if i == 0:
                assert (i, j) in clusters_in_tree
            else:
                assert (i, j) in clusters_in_tree and (i, j) in cluster_tree


def test_build_cluster_layers(label_layers):
    cluster_layers = build_cluster_layers(label_layers)
    assert all(isinstance(layer, ClusterLayer) for layer in cluster_layers)
    assert len(cluster_layers) == 3
    assert all(
        isinstance(cluster, Cluster) for layer in cluster_layers for cluster in layer
    )

    assert len(cluster_layers[0]) == 2
    print(type(cluster_layers[0][0].members))
    assert np.array_equal(cluster_layers[0][0].members, [0, 1])
    assert np.array_equal(cluster_layers[0][1].members, [4, 5])

    assert len(cluster_layers[1]) == 2
    assert np.array_equal(cluster_layers[1][0].members, [0, 1, 2])
    assert np.array_equal(cluster_layers[1][1].members, [3, 4, 5])

    assert len(cluster_layers[2]) == 1
    assert np.array_equal(cluster_layers[2][0].members, [0, 1, 2, 3, 4, 5])


def test_precomputed_clusterer(label_layers):
    clusterer = PrecomputedClusterer()
    clusterer.fit(label_layers)
    check_is_fitted(clusterer)
    assert len(clusterer.cluster_layers_) == 3
    assert len(clusterer.cluster_tree_) == 4


def test_kmeans_clusterer():
    n_samples = 1000
    clusterable_data, clusterable_labels = make_blobs(
        n_samples=n_samples,
        n_features=2,
        centers=5,
        center_box=(0.0, 1.0),
        cluster_std=0.05,
        random_state=0,
    )
    clusterer = KMeansClusterer(
        min_clusters=4,
        base_n_clusters=64,
        random_state=42,
    )
    clusterer.fit_predict(clusterable_data)

    check_is_fitted(clusterer)
    assert len(clusterer.cluster_layers_) == 3
    assert all(
        adjusted_mutual_info_score(
            layer_to_labels_helper(cluster_layer, n_samples), clusterable_labels
        )
        >= (0.25 * (i + 1))
        for i, cluster_layer in enumerate(clusterer.cluster_layers_)
    )


def test_plscan_clusterer():
    n_samples = 1000
    clusterable_data, clusterable_labels = make_blobs(
        n_samples=n_samples,
        n_features=2,
        centers=10,
        center_box=(0.0, 1.0),
        cluster_std=0.05,
        random_state=0,
    )
    clusterer = PLSCANClusterer()
    clusterer.fit(clusterable_data)

    check_is_fitted(clusterer)
    assert len(clusterer.cluster_layers_) > 1
    assert all(isinstance(layer, ClusterLayer) for layer in clusterer.cluster_layers_)
    assert isinstance(clusterer.cluster_tree_, dict)

    assert len(clusterer.cluster_layers_) == len(
        clusterer.plscan_.membership_strength_layers_
    )
    assert len(clusterer.cluster_layers_) == len(
        clusterer.plscan_.layer_persistence_scores_
    )

    for layer in clusterer.cluster_layers_:
        labels = layer_to_labels_helper(layer, n_samples)
        assert labels.shape == (1000,)

    final_labels = layer_to_labels_helper(clusterer.cluster_layers_[-1], n_samples)
    assert (
        adjusted_mutual_info_score(
            final_labels[final_labels >= 0],
            clusterable_labels[final_labels >= 0],
        )
        >= 0.9
    )


def test_plscan_clusterer_fit_predict_returns_layers_and_tree():
    n_samples = 2000
    clusterable_data, _ = make_blobs(
        n_samples=n_samples,
        n_features=2,
        centers=10,
        center_box=(0.0, 1.0),
        cluster_std=0.05,
        random_state=42,
    )
    clusterer = PLSCANClusterer()
    layers, tree = clusterer.fit_predict(clusterable_data)
    assert layers is clusterer.cluster_layers_
    assert tree is clusterer.cluster_tree_
    assert len(layers) >= 1
    assert isinstance(tree, dict)
    assert layer_to_labels_helper(layers[0], n_samples).max() + 1 >= 4


def test_plscan_clusterer_max_layers():
    clusterable_data, _ = make_blobs(
        n_samples=2000,
        n_features=2,
        centers=10,
        center_box=(0.0, 1.0),
        cluster_std=0.05,
        random_state=42,
    )
    clusterer = PLSCANClusterer(max_layers=1)
    layers, tree = clusterer.fit_predict(clusterable_data)
    check_is_fitted(clusterer)
    assert clusterer.plscan_.max_layers == 1
    assert len(layers) == 1


@pytest.mark.xfail(
    strict=True, reason="Known numba cache conflict between EVoC and PLSCAN"
)
def test_evoc_clusterer():
    n_samples = 1000
    clusterable_data, cluster_labels = make_blobs(
        n_samples=n_samples,
        n_features=128,
        centers=5,
        center_box=(0.0, 1.0),
        cluster_std=0.001,
        random_state=0,
    )
    clusterer = EVoCClusterer(random_state=42)
    clusterer.fit(clusterable_data)
    check_is_fitted(clusterer)
    assert len(clusterer.cluster_layers_[-1]) >= 5
    assert len(clusterer.cluster_layers_[-1]) <= 8
    final_labels = layer_to_labels_helper(clusterer.cluster_layers_[-1], n_samples)
    assert (
        adjusted_mutual_info_score(
            final_labels[final_labels >= 0], cluster_labels[final_labels >= 0]
        )
        >= 0.75
    )


@pytest.mark.xfail(
    strict=True, reason="Known numba cache conflict between EVoC and PLSCAN"
)
def test_evoc_clusterer_fit_predict_returns_layers_and_tree():
    n_samples = 1000
    clusterable_data, _ = make_blobs(
        n_samples=n_samples,
        n_features=128,
        centers=5,
        center_box=(0.0, 1.0),
        cluster_std=0.001,
        random_state=42,
    )
    clusterer = EVoCClusterer(random_state=42)
    layers, tree = clusterer.fit_predict(clusterable_data)
    assert layers is clusterer.cluster_layers_
    assert tree is clusterer.cluster_tree_
    assert len(layers) >= 1
    assert isinstance(tree, dict)
    assert layer_to_labels_helper(layers[0], n_samples).max() + 1 >= 4


@pytest.mark.xfail(
    strict=True, reason="Known numba cache conflict between EVoC and PLSCAN"
)
def test_evoc_clusterer_max_layers():
    n_samples = 1000
    clusterable_data, cluster_labels = make_blobs(
        n_samples=n_samples,
        n_features=128,
        centers=5,
        center_box=(0.0, 1.0),
        cluster_std=0.001,
        random_state=0,
    )
    clusterer = EVoCClusterer(max_layers=1, random_state=42)
    layers, tree = clusterer.fit_predict(clusterable_data)
    check_is_fitted(clusterer)
    assert clusterer.evoc_.max_layers == 1
    assert len(layers) == 1
