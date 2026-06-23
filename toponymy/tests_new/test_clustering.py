import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_mutual_info_score, pairwise_distances
from sklearn.utils.validation import check_is_fitted

from toponymy.new_types import (
    Cluster,
    ClusterLayer
)

from toponymy.new_clustering import (
    KMeansClusterer, 
    PLSCANClusterer,
    PreComputedClusterer,
    build_cluster_tree,
    _build_cluster_tree,
    build_cluster_layers
)

# Try to import EVoCClusterer - check if evoc is compatible with current fast_hdbscan
# try:
#     import fast_hdbscan

#     from toponymy.new_clustering import EVoCClusterer

#     # evoc 0.3.1 is incompatible with fast_hdbscan >= 0.3.2 due to NumbaKDTree signature changes
#     # Skip EVoC tests if we detect an incompatible version
#     fast_hdbscan_version = tuple(map(int, fast_hdbscan.__version__.split(".")[:2]))
#     HAS_COMPATIBLE_EVOC = fast_hdbscan_version < (0, 3)
# except (ImportError, AttributeError):
    # HAS_COMPATIBLE_EVOC = False

def layer_to_labels_helper(layer: ClusterLayer, n_points: int):
    labels = np.full(n_points, -1, dtype="int")
    for i, cluster in enumerate(layer.clusters):
        labels[cluster.members] = i
    return labels


def test_build_cluster_tree():
    label_layers = [
        np.array([0, 0, -1, -1, 1, 1, -1]),
        np.array([0, 0, 0, 1, 1, 1, -1]),
        np.array([0, 0, 0, 0, 0, 0, -1])
    ]
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


def test_build_cluster_tree_no_jit():
    label_layers = [
        np.array([0, 0, -1, -1, 1, 1, -1]),
        np.array([0, 0, 0, 1, 1, 1, -1]),
        np.array([0, 0, 0, 0, 0, 0, -1])
    ]
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


def test_kmeans_clusterer_class():
    clusterer = KMeansClusterer(
        min_clusters=4,
        base_n_clusters=64,
    )
    n_samples = 1000
    clusterable_data, clusterable_labels = make_blobs(
        n_samples=n_samples,
        n_features=2,
        centers=5,
        center_box=(0.0, 1.0),
        cluster_std=0.05,
        random_state=0,
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


def test_plscan_clusterer_fit_returns_self():
    n_samples = 1000
    clusterer = PLSCANClusterer(
        min_clusters=4,
        min_samples=5,
        base_min_cluster_size=10,
    )
    clusterable_data, clusterable_labels = make_blobs(
        n_samples=n_samples,
        n_features=2,
        centers=5,
        center_box=(0.0, 1.0),
        cluster_std=0.05,
        random_state=0,
    )
    clusterer.fit(clusterable_data)

    check_is_fitted(clusterer)
    assert len(clusterer.cluster_layers_) > 1
    assert len(clusterer.cluster_layers_) == len(clusterer.cluster_probabilities_)
    assert len(clusterer.cluster_layers_) == len(clusterer.cluster_persistence_scores_)
    np.testing.assert_array_equal(
        clusterer.plscan_min_cluster_sizes_, clusterer.plscan_.min_cluster_sizes_
    )
    assert all(isinstance(layer, ClusterLayer) for layer in clusterer.cluster_layers_)
    assert isinstance(clusterer.cluster_tree_, dict)

    for layer, probabilities in zip(
        clusterer.cluster_layers_, clusterer.cluster_probabilities_
    ):
        labels = layer_to_labels_helper(layer, n_samples)
        assert labels.shape == (1000,)
        assert probabilities.shape == labels.shape
        assert np.all(probabilities >= 0.0)
        assert np.all(probabilities <= 1.0)

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
        n_features=10,
        centers=20,
        center_box=(0.0, 1.0),
        cluster_std=0.05,
        random_state=42,
    )

    clusterer = PLSCANClusterer(
        min_clusters=4,
        min_samples=5,
        base_min_cluster_size=10,
    )
    layers, tree = clusterer.fit_predict(clusterable_data)
    assert layers is clusterer.cluster_layers_
    assert tree is clusterer.cluster_tree_
    assert len(layers) >= 1
    assert isinstance(tree, dict)
    assert layer_to_labels_helper(layers[0], n_samples).max() + 1 >= 4


def test_plscan_clusterer_max_layers():
    clusterable_data, _ = make_blobs(
        n_samples=2000,
        n_features=10,
        centers=20,
        center_box=(0.0, 1.0),
        cluster_std=0.05,
        random_state=42,
    )
    clusterer = PLSCANClusterer(
        min_clusters=2,
        min_samples=5,
        base_min_cluster_size=10,
        max_layers=1,
    )
    layers, tree = clusterer.fit_predict(clusterable_data)
    assert len(layers) == 1


def test_plscan_clusterer_raises_for_too_few_clusters():
    clusterable_data, _ = make_blobs(
        n_samples=300,
        n_features=2,
        centers=3,
        center_box=(0.0, 1.0),
        cluster_std=0.05,
        random_state=11,
    )
    clusterer = PLSCANClusterer(
        min_clusters=50,
        min_samples=5,
        base_min_cluster_size=10,
    )
    with pytest.raises(
        ValueError, match="Not enough clusters found in any PLSCAN layer"
    ):
        clusterer.fit(clusterable_data)


# @pytest.mark.skipif(
#     not HAS_COMPATIBLE_EVOC,
#     reason="evoc not installed or incompatible with current fast_hdbscan version (evoc 0.3.1 requires fast_hdbscan < 0.3.0)",
# )
# def test_evoc_clusterer_class():
#     clusterer = EVoCClusterer(
#         min_clusters=4,
#         min_samples=5,
#         base_min_cluster_size=10,
#         next_cluster_size_quantile=0.8,
#     )

#     np.random.seed(0)
#     clusterable_data, cluster_labels = make_blobs(
#         n_samples=1000,
#         n_features=128,
#         centers=5,
#         center_box=(0.0, 1.0),
#         cluster_std=0.05,
#         random_state=0,
#     )
#     class_cluster_layers, class_tree = clusterer.fit_predict(
#         clusterable_vectors=clusterable_data,
#         embedding_vectors=clusterable_data,
#         layer_class=ClusterLayerText,
#     )
#     assert (
#         np.unique(class_cluster_layers[-1].cluster_labels).shape[0] >= 5
#         and np.unique(class_cluster_layers[-1].cluster_labels).shape[0] <= 7
#     )
#     assert (
#         adjusted_mutual_info_score(
#             class_cluster_layers[-1].cluster_labels[
#                 class_cluster_layers[-1].cluster_labels >= 0
#             ],
#             cluster_labels[class_cluster_layers[-1].cluster_labels >= 0],
#         )
#         >= 0.75
#     )
