import numpy as np
import pytest

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

import os

from toponymy.utility_functions import (
    centroids_from_labels,
    distance_to_vector,
    diversify_fixed_alpha,
    diversify_max_alpha,
)

TEST_VECTORS = normalize(np.random.rand(1000, 512))


def test_distance_to_vector():
    query_vector = np.random.rand(512)
    distances = distance_to_vector(query_vector, TEST_VECTORS)
    assert distances.shape == (1000,)
    assert np.all(distances >= 0.0)
    assert np.all(distances <= 2.0)
    true_distances = pairwise_distances(
        query_vector.reshape(1, -1), TEST_VECTORS, metric="cosine"
    ).squeeze()
    assert np.allclose(distances, true_distances)


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_diversify_fixed_alpha(alpha):
    query_vector = np.random.rand(512)
    query_distances = pairwise_distances(
        query_vector.reshape(1, -1), TEST_VECTORS, metric="cosine"
    ).squeeze()
    distance_to_query_order = np.argsort(query_distances)
    retained_indices = diversify_fixed_alpha(
        query_vector, TEST_VECTORS[distance_to_query_order], alpha=alpha
    )
    all_pairs_distances = pairwise_distances(
        TEST_VECTORS[distance_to_query_order], metric="cosine"
    )

    for i, index in enumerate(retained_indices):
        # Use small tolerance for floating-point comparison to handle precision differences across Python versions
        assert np.all(
            all_pairs_distances[index, retained_indices[:i]]
            >= alpha * query_distances[distance_to_query_order][index] - 1e-10
        )


@pytest.mark.parametrize("max_alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("n_results", [5, 10, 15, 20])
def test_diversify_max_alpha(max_alpha, n_results):
    query_vector = np.random.rand(512)
    query_distances = pairwise_distances(
        query_vector.reshape(1, -1), TEST_VECTORS, metric="cosine"
    ).squeeze()
    distance_to_query_order = np.argsort(query_distances)[: n_results + 2]
    retained_indices = diversify_max_alpha(
        query_vector,
        TEST_VECTORS[distance_to_query_order],
        n_results,
        max_alpha=max_alpha,
    )
    assert len(retained_indices) >= n_results


def test_distance_to_vector_no_jit():
    query_vector = np.random.rand(512)
    distances = distance_to_vector.py_func(query_vector, TEST_VECTORS)
    assert distances.shape == (1000,)
    assert np.all(distances >= 0.0)
    assert np.all(distances <= 2.0)


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_diversify_fixed_alpha_no_jit(alpha):
    query_vector = np.random.rand(512)
    query_distances = pairwise_distances(
        query_vector.reshape(1, -1), TEST_VECTORS, metric="cosine"
    ).squeeze()
    distance_to_query_order = np.argsort(query_distances)
    retained_indices = diversify_fixed_alpha.py_func(
        query_vector, TEST_VECTORS[distance_to_query_order], alpha=alpha
    )
    all_pairs_distances = pairwise_distances(
        TEST_VECTORS[distance_to_query_order], metric="cosine"
    )

    for i, index in enumerate(retained_indices):
        # Use small tolerance for floating-point comparison to handle precision differences across Python versions
        assert np.all(
            all_pairs_distances[index, retained_indices[:i]]
            >= alpha * query_distances[distance_to_query_order][index] - 1e-10
        )


@pytest.mark.parametrize("max_alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("n_results", [5, 10, 15, 20])
def test_diversify_max_alpha_no_jit(max_alpha, n_results):
    query_vector = np.random.rand(512)
    query_distances = pairwise_distances(
        query_vector.reshape(1, -1), TEST_VECTORS, metric="cosine"
    ).squeeze()
    distance_to_query_order = np.argsort(query_distances)[: n_results + 2]
    retained_indices = diversify_max_alpha.py_func(
        query_vector,
        TEST_VECTORS[distance_to_query_order],
        n_results,
        max_alpha=max_alpha,
    )
    assert len(retained_indices) >= n_results


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
