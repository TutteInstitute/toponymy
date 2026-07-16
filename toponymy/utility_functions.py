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

    distances = np.ones(other_vectors.shape[0], dtype=np.float64)
    if vector_norm > 0.0:
        for i in range(other_vectors.shape[0]):
            if other_vector_norms[i] > 0.0:
                cosine = result[i] / np.sqrt(vector_norm * other_vector_norms[i])
                distances[i] = 1.0 - min(1.0, max(-1.0, cosine))
    return distances


@numba.njit(cache=True)
def diversify_fixed_alpha(query_vector, candidate_neighbor_vectors, alpha=1.0):
    distance_to_query = distance_to_vector(query_vector, candidate_neighbor_vectors)

    retained_neighbor_indices = [i for i in range(min(1, len(candidate_neighbor_vectors)))]
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


@numba.njit(cache=True)
def diversify_max_alpha(
    query_vector,
    candidate_neighbor_vectors,
    n_results,
    max_alpha=1.0,
    min_alpha=0.0,
    tolerance=0.01,
):
    if tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and positive")
    if not np.isfinite(min_alpha) or not np.isfinite(max_alpha) or min_alpha < 0 or max_alpha < min_alpha:
        raise ValueError("alpha bounds must be finite and satisfy 0 <= min <= max")
    if n_results <= 0:
        return [i for i in range(0)]
    mid_alpha = (max_alpha + min_alpha) / 2.0

    while abs(max_alpha - min_alpha) > tolerance:
        results = diversify_fixed_alpha(
            query_vector, candidate_neighbor_vectors, alpha=mid_alpha
        )
        if len(results) >= n_results:
            min_alpha = mid_alpha
        else:
            max_alpha = mid_alpha

        mid_alpha = (min_alpha + max_alpha) / 2.0

    return diversify_fixed_alpha(
        query_vector, candidate_neighbor_vectors, alpha=min_alpha
    )


@numba.njit(cache=True)
def centroids_from_labels(
    cluster_labels: np.ndarray, vector_data: np.ndarray
) -> np.ndarray:  # pragma: no cover
    n_clusters = cluster_labels.max() + 1 if len(cluster_labels) else 0
    result = np.zeros((n_clusters, vector_data.shape[1]))
    counts = np.zeros(n_clusters)
    for i in range(cluster_labels.shape[0]):
        cluster_num = cluster_labels[i]
        if cluster_num >= 0:
            result[cluster_num] += vector_data[i]
            counts[cluster_num] += 1

    for i in range(result.shape[0]):
        if counts[i] > 0:
            result[i] /= counts[i]

    return result

