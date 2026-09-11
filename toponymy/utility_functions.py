import numpy as np
import numba


def _normalize_rows(vectors):
    """Unit directions without overflow or loss of tiny nonzero rows."""
    vectors = np.asarray(vectors, dtype=np.float64)
    scales = np.max(np.abs(vectors), axis=1, keepdims=True, initial=0.0)
    result = np.divide(vectors, scales, out=np.zeros_like(vectors), where=scales != 0)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    np.divide(result, norms, out=result, where=norms != 0)
    return result


def _mean_vector(vectors, weights=None):
    """A finite mean with independent coordinate scales and optional weights."""
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or not len(vectors) or not np.isfinite(vectors).all():
        raise ValueError("vectors must be a nonempty finite two-dimensional matrix")
    if weights is not None:
        weights = np.asarray(weights)
        if (
            weights.shape != (len(vectors),)
            or weights.dtype.kind not in "fiu"
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or not np.any(weights > 0)
        ):
            raise ValueError("weights must be finite, nonnegative, aligned and nonzero")
        weights = weights.astype(np.float64)
        weights = weights / weights.max()
    lower, upper = vectors.min(axis=0), vectors.max(axis=0)
    # Remove common offsets before averaging their much smaller differences.
    anchor = lower * 0.5 + upper * 0.5
    shifted = vectors - anchor
    scales = np.max(np.abs(shifted), axis=0, initial=0.0)
    scaled = np.divide(shifted, scales, out=np.zeros_like(shifted), where=scales != 0)
    mean = np.average(scaled, axis=0, weights=weights)
    with np.errstate(over="ignore"):
        return np.clip(anchor + np.clip(mean, -1.0, 1.0) * scales, lower, upper)


def _center_vectors(vectors, center):
    """Center cosine inputs, scaling only rows whose subtraction overflows."""
    vectors = np.asarray(vectors, dtype=np.float64)
    with np.errstate(over="ignore"):
        centered = vectors - center
    overflow = ~np.isfinite(centered).all(axis=1)
    if np.any(overflow):
        centered[overflow] = vectors[overflow] * 0.5 - center * 0.5
    return centered


@numba.njit(cache=True)
def distance_to_vector(vector, other_vectors):
    """Cosine distances, including finite vectors at very small or large scales."""
    distances = np.ones(other_vectors.shape[0], dtype=np.float64)
    scale = 0.0
    for j in range(vector.shape[0]):
        scale = max(scale, abs(np.float64(vector[j])))
    if scale == 0.0:
        return distances
    vector_norm = 0.0
    for j in range(vector.shape[0]):
        value = np.float64(vector[j]) / scale
        vector_norm += value * value

    for i in range(other_vectors.shape[0]):
        other_scale = 0.0
        for j in range(vector.shape[0]):
            other_scale = max(other_scale, abs(np.float64(other_vectors[i, j])))
        if other_scale == 0.0:
            continue
        product, other_norm = 0.0, 0.0
        for j in range(vector.shape[0]):
            value = np.float64(other_vectors[i, j]) / other_scale
            product += (np.float64(vector[j]) / scale) * value
            other_norm += value * value
        cosine = product / np.sqrt(vector_norm * other_norm)
        distances[i] = 1.0 - min(1.0, max(-1.0, cosine))
    return distances


@numba.njit(cache=True)
def diversify_fixed_alpha(query_vector, candidate_neighbor_vectors, alpha=1.0):
    distance_to_query = distance_to_vector(query_vector, candidate_neighbor_vectors)

    retained_neighbor_indices = [
        i for i in range(min(1, len(candidate_neighbor_vectors)))
    ]
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
    if (
        not np.isfinite(min_alpha)
        or not np.isfinite(max_alpha)
        or min_alpha < 0
        or max_alpha < min_alpha
    ):
        raise ValueError("alpha bounds must be finite and satisfy 0 <= min <= max")
    if n_results <= 0:
        return [i for i in range(0)]
    while max_alpha - min_alpha > tolerance:
        mid_alpha = min_alpha + (max_alpha - min_alpha) / 2.0
        if mid_alpha == min_alpha or mid_alpha == max_alpha:
            break
        results = diversify_fixed_alpha(
            query_vector, candidate_neighbor_vectors, alpha=mid_alpha
        )
        if len(results) >= n_results:
            min_alpha = mid_alpha
        else:
            max_alpha = mid_alpha

    return diversify_fixed_alpha(
        query_vector, candidate_neighbor_vectors, alpha=min_alpha
    )


@numba.njit(cache=True)
def centroids_from_labels(
    cluster_labels: np.ndarray, vector_data: np.ndarray
) -> np.ndarray:
    n_clusters = cluster_labels.max() + 1 if len(cluster_labels) else 0
    result = np.zeros((n_clusters, vector_data.shape[1]))
    counts = np.zeros(n_clusters)
    scales = np.zeros((n_clusters, vector_data.shape[1]))
    for i in range(cluster_labels.shape[0]):
        cluster_num = cluster_labels[i]
        if cluster_num >= 0:
            counts[cluster_num] += 1
            for j in range(vector_data.shape[1]):
                scales[cluster_num, j] = max(
                    scales[cluster_num, j], abs(np.float64(vector_data[i, j]))
                )

    for i in range(cluster_labels.shape[0]):
        cluster_num = cluster_labels[i]
        if cluster_num >= 0:
            for j in range(vector_data.shape[1]):
                if scales[cluster_num, j] > 0:
                    result[cluster_num, j] += (
                        vector_data[i, j] / scales[cluster_num, j] / counts[cluster_num]
                    )

    for i in range(result.shape[0]):
        # A mean stays inside the input range, including rounding at float max.
        result[i] = np.minimum(1.0, np.maximum(-1.0, result[i])) * scales[i]

    return result
