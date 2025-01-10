from toponymy.clustering import (
    build_raw_cluster_layers,
    build_cluster_tree,
    centroids_from_labels,
    create_cluster_layers,
    ToponymyClusterer,
    EVoCClusterer,
)
from toponymy.cluster_layer import ClusterLayerText
from sklearn.metrics import adjusted_mutual_info_score
import numpy as np

from sklearn.datasets import make_blobs


def test_centroids_from_labels():
    for i in range(10):
        n_clusters = np.random.randint(5, 50)
        n_samples = np.random.randint(100, 1000)
        cluster_labels = np.random.randint(-1, n_clusters, n_samples)
        vector_data = np.random.random_sample((n_samples, 256))
        result = centroids_from_labels(cluster_labels, vector_data)
        expected = np.vstack(
            [
                vector_data[cluster_labels == i].mean(axis=0) if np.sum(cluster_labels == i) > 0 else np.zeros(256)
                for i in range(cluster_labels.max() + 1)
            ]
        )
        np.testing.assert_array_equal(result, expected)


def test_build_cluster_tree():
    clusterable_data = np.vstack(
        [
            make_blobs(
                n_samples=1000,
                n_features=2,
                centers=5,
                center_box=(0.0, 1.0),
                cluster_std=0.05,
            )[0],
            np.random.random_sample((1000, 2)),
        ]
    )
    cluster_label_vectors = build_raw_cluster_layers(
        clusterable_data, min_clusters=5, min_samples=5
    )
    cluster_tree = build_cluster_tree(cluster_label_vectors)

    # Assert that every child cluster is contained in the parent cluster
    for parent, children in cluster_tree.items():
        for child in children:
            assert parent[0] > child[0]
            # Skip if parent is the artificial root node
            if parent[0] >= len(cluster_label_vectors):
                continue
            np.testing.assert_array_equal(
                (cluster_label_vectors[parent[0]] == parent[1])
                & (cluster_label_vectors[child[0]] == child[1]),
                cluster_label_vectors[child[0]] == child[1],
            )

    # Assert that every cluster is contained in the tree
    clusters_in_tree = sum(cluster_tree.values(), [])
    assert len(set(clusters_in_tree)) == len(clusters_in_tree)
    for i in range(len(cluster_label_vectors)):
        for j in range(cluster_label_vectors[i].max() + 1):
            if i == 0:
                assert (i, j) in clusters_in_tree
            else:
                assert (i, j) in clusters_in_tree and (i, j) in cluster_tree


def test_clusterer_class():
    clusterer = ToponymyClusterer(min_clusters=5, min_samples=5)

    clusterable_data = np.vstack(
        [
            make_blobs(
                n_samples=1000,
                n_features=2,
                centers=5,
                center_box=(0.0, 1.0),
                cluster_std=0.05,
            )[0],
            np.random.random_sample((100, 2)),
        ]
    )
    embedding_vectors = np.random.random_sample((1100, 256))
    cluster_label_vectors = build_raw_cluster_layers(
        clusterable_data, min_clusters=5, min_samples=5
    )
    cluster_tree = build_cluster_tree(cluster_label_vectors)
    cluster_layers = create_cluster_layers(ClusterLayerText, clusterable_data, embedding_vectors)

    class_cluster_layers, class_tree = clusterer.fit_predict(
        clusterable_vectors=clusterable_data, 
        embedding_vectors=embedding_vectors,
        layer_class=ClusterLayerText
    )

    assert len(class_cluster_layers) == len(cluster_layers)
    for key in class_tree:
        subclusters = sorted(class_tree[key])
        found = False
        for other_subclusters in cluster_tree.values():
            if subclusters == sorted(other_subclusters):
                found = True
                break
        assert found
    assert all(
        adjusted_mutual_info_score(class_cluster_layers[i].cluster_labels, cluster_label_vectors[i]) == 1.0
        for i in range(len(cluster_layers))
    )