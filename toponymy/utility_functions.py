import numpy as np
import numba


@numba.njit(fastmath=True, cache=True)
def distance_to_vector(vector, other_vectors):
    result = np.zeros(other_vectors.shape[0], dtype=np.float32)
    other_vector_norms = np.zeros(other_vectors.shape[0], dtype=np.float32)
    vector_norm = 0.0
    for j in range(vector.shape[0]):
        vector_norm += vector[j] * vector[j]

    for i in range(other_vectors.shape[0]):
        for j in range(vector.shape[0]):
            result[i] += vector[j] * other_vectors[i, j]
            other_vector_norms[i] += other_vectors[i, j] * other_vectors[i, j]

    if vector_norm == 0.0:
        return np.ones(other_vectors.shape[0], dtype=np.float64)
    else:
        return 1.0 - (result / np.sqrt(vector_norm * other_vector_norms))


@numba.njit(cache=True)
def diversify_fixed_alpha(query_vector, candidate_neighbor_vectors, alpha=1.0):
    distance_to_query = distance_to_vector(query_vector, candidate_neighbor_vectors)

    retained_neighbor_indices = [0]
    for i, vector in enumerate(candidate_neighbor_vectors[1:], 1):
        retained_vectors = candidate_neighbor_vectors[
            np.array(retained_neighbor_indices)
        ]
        retained_neighbor_distances = distance_to_vector(
            vector,
            retained_vectors,
        )
        for j in range(retained_neighbor_distances.shape[0]):
            if alpha * distance_to_query[i] > retained_neighbor_distances[j]:
                break
        else:
            retained_neighbor_indices.append(i)

    return retained_neighbor_indices


# 512 candidates keep the table at 2 MB; above ~1024, recomputing is faster.
MAX_CANDIDATES_FOR_DISTANCE_TABLE = 512


@numba.njit(cache=True)
def _distances_to_earlier_candidates(candidate_neighbor_vectors):
    n_candidates = candidate_neighbor_vectors.shape[0]
    distances = np.zeros((n_candidates, n_candidates))
    for i in range(1, n_candidates):
        distances[i, :i] = distance_to_vector(
            candidate_neighbor_vectors[i], candidate_neighbor_vectors[:i]
        )
    return distances


@numba.njit(cache=True)
def _diversify_fixed_alpha_from_table(distance_to_query, distance_table, alpha):
    retained_neighbor_indices = [0]
    for i in range(1, distance_to_query.shape[0]):
        for j in retained_neighbor_indices:
            if alpha * distance_to_query[i] > distance_table[i, j]:
                break
        else:
            retained_neighbor_indices.append(i)

    return retained_neighbor_indices


@numba.njit(cache=True)
def diversify_max_alpha(
    query_vector,
    candidate_neighbor_vectors,
    n_results,
    max_alpha=1.0,
    min_alpha=0.0,
    tolerance=0.01,
):
    use_distance_table = (
        candidate_neighbor_vectors.shape[0] <= MAX_CANDIDATES_FOR_DISTANCE_TABLE
    )
    if use_distance_table:
        distance_to_query = distance_to_vector(query_vector, candidate_neighbor_vectors)
        distance_table = _distances_to_earlier_candidates(candidate_neighbor_vectors)
    mid_alpha = (max_alpha + min_alpha) / 2.0

    while abs(max_alpha - min_alpha) > tolerance:
        if use_distance_table:
            results = _diversify_fixed_alpha_from_table(
                distance_to_query, distance_table, alpha=mid_alpha
            )
        else:
            results = diversify_fixed_alpha(
                query_vector, candidate_neighbor_vectors, alpha=mid_alpha
            )
        if len(results) >= n_results:
            min_alpha = mid_alpha
        else:
            max_alpha = mid_alpha

        mid_alpha = (min_alpha + max_alpha) / 2.0

    if use_distance_table:
        return _diversify_fixed_alpha_from_table(
            distance_to_query, distance_table, alpha=min_alpha
        )
    return diversify_fixed_alpha(
        query_vector, candidate_neighbor_vectors, alpha=min_alpha
    )
