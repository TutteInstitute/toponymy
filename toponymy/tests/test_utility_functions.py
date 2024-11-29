import numpy as np
import pytest

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

from toponymy.utility_functions import distance_to_vector, diversify_fixed_alpha, diversify_max_alpha

TEST_VECTORS = normalize(np.random.rand(1000, 512, dtype=np.float32))

def test_distance_to_vector():
    query_vector = np.random.rand(512, dtype=np.float32)
    distances = distance_to_vector(query_vector, TEST_VECTORS)
    assert distances.shape == (1000,)
    assert distances.dtype == np.float32
    assert np.all(distances >= 0.0)
    assert np.all(distances <= 2.0)
    true_distances = pairwise_distances(query_vector.reshape(1, -1), TEST_VECTORS, metric="cosine").squeeze()
    assert np.allclose(distances, true_distances)

@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_diversify_fixed_alpha(alpha):
    query_vector = np.random.rand(512, dtype=np.float32)
    query_distances = pairwise_distances(query_vector.reshape(1, -1), TEST_VECTORS, metric="cosine").squeeze()
    distance_to_query_order = np.argsort(query_distances)
    retained_indices = diversify_fixed_alpha(query_vector, TEST_VECTORS[distance_to_query_order], alpha=alpha)
    all_pairs_distances = pairwise_distances(TEST_VECTORS[distance_to_query_order], metric="cosine")

    for i, index in enumerate(retained_indices):
        assert np.all(all_pairs_distances[index, retained_indices[:i]] >= alpha * query_distances[distance_to_query_order][index])

@pytest.mark.parametrize("max_alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("n_results", [5, 10, 15, 20])
def test_diversify_max_alpha(max_alpha, n_results):
    query_vector = np.random.rand(512, dtype=np.float32)
    query_distances = pairwise_distances(query_vector.reshape(1, -1), TEST_VECTORS, metric="cosine").squeeze()
    distance_to_query_order = np.argsort(query_distances)[:n_results + 2]
    retained_indices = diversify_max_alpha(query_vector, TEST_VECTORS[distance_to_query_order], 10, max_alpha=max_alpha)
    assert len(retained_indices) >= n_results