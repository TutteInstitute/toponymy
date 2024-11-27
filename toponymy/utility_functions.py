import numpy as np

from sklearn.metrics.pairwise import pairwise_distances

def diversify(query_vector, candidate_neighbor_vectors, alpha=1.0, max_candidates=16, metric="cosine"):
    distance_to_query = np.squeeze(
        pairwise_distances(
            [query_vector], candidate_neighbor_vectors, metric=metric
        )
    )

    retained_neighbor_indices = [0]
    for i, vector in enumerate(candidate_neighbor_vectors[1:], 1):
        retained_neighbor_distances = pairwise_distances(
            [vector],
            candidate_neighbor_vectors[retained_neighbor_indices],
            metric=metric,
        )[0]
        for j in range(retained_neighbor_distances.shape[0]):
            if alpha * distance_to_query[i] > retained_neighbor_distances[j]:
                break
        else:
            retained_neighbor_indices.append(i)
            if len(retained_neighbor_indices) >= max_candidates:
                return retained_neighbor_indices

    return retained_neighbor_indices