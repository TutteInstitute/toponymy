import numpy as np
import pytest

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

import os

from toponymy.utility_functions import (
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


def reference_diversify_max_alpha(
    query_vector, candidate_vectors, n_results, max_alpha, min_alpha, tolerance
):
    query_distances = pairwise_distances(
        query_vector.reshape(1, -1), candidate_vectors, metric="cosine"
    ).squeeze()
    pair_distances = pairwise_distances(candidate_vectors, metric="cosine")

    def retain(alpha):
        retained = [0]
        for i in range(1, len(candidate_vectors)):
            if np.all(pair_distances[i, retained] >= alpha * query_distances[i]):
                retained.append(i)
        return retained

    mid_alpha = (max_alpha + min_alpha) / 2.0
    while abs(max_alpha - min_alpha) > tolerance:
        if len(retain(mid_alpha)) >= n_results:
            min_alpha = mid_alpha
        else:
            max_alpha = mid_alpha
        mid_alpha = (min_alpha + max_alpha) / 2.0
    return retain(min_alpha)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize(
    "n_results, min_alpha, tolerance", [(16, 0.5, 0.1), (4, 0.0, 0.01)]
)
def test_diversify_max_alpha_matches_reference(seed, n_results, min_alpha, tolerance):
    rng = np.random.default_rng(seed)
    query_vector = rng.normal(size=64)
    candidate_vectors = rng.normal(size=(60, 64)).astype(np.float32)
    query_distances = pairwise_distances(
        query_vector.reshape(1, -1), candidate_vectors, metric="cosine"
    ).squeeze()
    candidate_vectors = candidate_vectors[np.argsort(query_distances)]

    retained_indices = diversify_max_alpha(
        query_vector,
        candidate_vectors,
        n_results,
        max_alpha=1.0,
        min_alpha=min_alpha,
        tolerance=tolerance,
    )
    expected = reference_diversify_max_alpha(
        query_vector,
        candidate_vectors,
        n_results,
        max_alpha=1.0,
        min_alpha=min_alpha,
        tolerance=tolerance,
    )
    assert list(retained_indices) == expected


@pytest.mark.parametrize(
    "query_vector, candidate_vectors, alpha, expected",
    [
        pytest.param(
            [1.0, 1.0, 0.0],
            np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float32),
            0.0,
            [0, 1],
            id="exact_tie_keeps_candidate",
        ),
        pytest.param(
            [1.0, 2.0, 3.0],
            np.array([[1, 2, 3], [0, 0, 0]], dtype=np.float32),
            0.0,
            [0, 1],
            id="zero_vector_nan_distance_keeps_candidate",
        ),
        pytest.param(
            [1.0, 0.0, 0.0],
            np.array([[0.1, 0.1, 0.5], [0.1, 0.1, 0.5]]),
            0.0,
            [0],
            id="duplicate_distance_rounds_below_zero",
        ),
    ],
)
def test_diversify_max_alpha_keeps_rounding_edge_cases(
    query_vector, candidate_vectors, alpha, expected
):
    retained_indices = diversify_max_alpha(
        np.array(query_vector), candidate_vectors, 2, max_alpha=alpha, min_alpha=alpha
    )
    assert list(retained_indices) == expected


@pytest.mark.parametrize("seed", range(5))
def test_diversify_max_alpha_matches_fixed_alpha_at_the_decision_boundary(seed):
    rng = np.random.default_rng(seed)
    query_vector = rng.normal(size=3)
    candidate_vectors = rng.normal(size=(2, 3)).astype(np.float32)
    boundary_alpha = (
        distance_to_vector(candidate_vectors[1], candidate_vectors[:1])[0]
        / distance_to_vector(query_vector, candidate_vectors)[1]
    )

    for alpha in [
        np.nextafter(boundary_alpha, -np.inf),
        boundary_alpha,
        np.nextafter(boundary_alpha, np.inf),
    ]:
        retained_indices = diversify_max_alpha(
            query_vector, candidate_vectors, 2, max_alpha=alpha, min_alpha=alpha
        )
        expected = diversify_fixed_alpha(query_vector, candidate_vectors, alpha=alpha)
        assert list(retained_indices) == list(expected)
