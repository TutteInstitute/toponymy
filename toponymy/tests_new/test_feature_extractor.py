import numpy as np
import pytest
from sklearn.utils.validation import check_is_fitted

from toponymy.feature_extractor import centroids_from_labels, FeatureExtractorBase


def test_centroids_from_labels():
    for i in range(10):
        n_clusters = np.random.randint(5, 50)
        n_samples = np.random.randint(100, 1000)
        cluster_labels = np.random.randint(-1, n_clusters, n_samples)
        vector_data = np.random.random_sample((n_samples, 256))
        result = centroids_from_labels(cluster_labels, vector_data)
        expected = np.vstack(
            [
                (
                    vector_data[cluster_labels == i].mean(axis=0)
                    if np.sum(cluster_labels == i) > 0
                    else np.zeros(256)
                )
                for i in range(cluster_labels.max() + 1)
            ]
        )
        np.testing.assert_array_equal(result, expected)


def test_centroids_from_labels_no_jit():
    for i in range(10):
        n_clusters = np.random.randint(5, 50)
        n_samples = np.random.randint(100, 1000)
        cluster_labels = np.random.randint(-1, n_clusters, n_samples)
        vector_data = np.random.random_sample((n_samples, 256))
        result = centroids_from_labels.py_func(cluster_labels, vector_data)
        expected = np.vstack(
            [
                (
                    vector_data[cluster_labels == i].mean(axis=0)
                    if np.sum(cluster_labels == i) > 0
                    else np.zeros(256)
                )
                for i in range(cluster_labels.max() + 1)
            ]
        )
        np.testing.assert_array_equal(result, expected)


def test_cannot_create_abstract_feature_extractor():
    with pytest.raises(
        TypeError,
        match="Can't instantiate abstract class FeatureExtractorBase without an implementation for abstract methods 'fit', 'get_cluster_features'",
    ):
        FeatureExtractorBase()
