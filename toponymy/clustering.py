import numpy as np
import numba
from fast_hdbscan.cluster_trees import (
    mst_to_linkage_tree,
    condense_tree,
    extract_leaves,
    get_cluster_label_vector,
)
from fast_hdbscan.boruvka import parallel_boruvka
from fast_hdbscan.numba_kdtree import kdtree_to_numba
from scipy.spatial import KDTree
from typing import List, Tuple, Dict, Type, Any

def build_raw_cluster_layers(
    data: np.ndarray,
    *,
    min_clusters: int = 3,
    min_samples: int = 5,
    base_min_cluster_size: int = 10,
    next_cluster_size_quantile: float = 0.8,
) -> List[np.ndarray]:
    n_samples = data.shape[0]
    cluster_layers = []
    min_cluster_size = base_min_cluster_size

    sklearn_tree = KDTree(data)
    numba_tree = kdtree_to_numba(sklearn_tree)
    edges = parallel_boruvka(
        numba_tree, min_samples=min_cluster_size if min_samples is None else min_samples
    )
    sorted_mst = edges[np.argsort(edges.T[2])]
    uncondensed_tree = mst_to_linkage_tree(sorted_mst)
    new_tree = condense_tree(uncondensed_tree, base_min_cluster_size)
    leaves = extract_leaves(new_tree)
    clusters = get_cluster_label_vector(new_tree, leaves, 0.0, n_samples)
    n_clusters_in_layer = clusters.max() + 1

    while n_clusters_in_layer >= min_clusters:
        cluster_layers.append(clusters)
        cluster_sizes = np.bincount(clusters[clusters >= 0])
        next_min_cluster_size = int(
            np.quantile(cluster_sizes, next_cluster_size_quantile)
        )
        if next_min_cluster_size <= min_cluster_size + 1:
            break
        else:
            min_cluster_size = next_min_cluster_size
        new_tree = condense_tree(uncondensed_tree, min_cluster_size)
        leaves = extract_leaves(new_tree)
        clusters = get_cluster_label_vector(new_tree, leaves, 0.0, n_samples)
        n_clusters_in_layer = clusters.max() + 1

    return cluster_layers


@numba.njit(cache=True)
def _build_cluster_tree(labels: List[np.ndarray]) -> List[Tuple[int, int, int, int]]:
    mapping = [(-1, -1, -1, -1) for _ in range(0)]
    found = [set([-1]) for _ in range(len(labels))]
    for upper_layer in range(1, len(labels)):
        upper_layer_unique_labels = np.unique(labels[upper_layer])
        for lower_layer in range(upper_layer - 1, -1, -1):
            upper_cluster_order = np.argsort(labels[upper_layer])
            cluster_groups = np.split(
                labels[lower_layer][upper_cluster_order],
                np.cumsum(np.bincount(labels[upper_layer] + 1))[:-1],
            )
            for i, label in enumerate(upper_layer_unique_labels):
                if label >= 0:
                    for child in cluster_groups[i]:
                        if child >= 0 and child not in found[lower_layer]:
                            mapping.append((upper_layer, label, lower_layer, child))
                            found[lower_layer].add(child)

    for lower_layer in range(len(labels) - 1, -1, -1):
        for child in range(labels[lower_layer].max() + 1):
            if child >= 0 and child not in found[lower_layer]:
                mapping.append((len(labels), 0, lower_layer, child))

    return mapping


def build_cluster_tree(labels: List[np.ndarray]) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    result = {}
    raw_mapping = _build_cluster_tree(labels)
    for parent_layer, parent_cluster, child_layer, child_cluster in raw_mapping:
        parent_name = (parent_layer, parent_cluster)
        if parent_name in result:
            result[parent_name].append((child_layer, child_cluster))
        else:
            result[parent_name] = [(child_layer, child_cluster)]
    return result


@numba.njit()
def centroids_from_labels(cluster_labels: np.ndarray, vector_data: np.ndarray) -> np.ndarray:
    result = np.zeros((cluster_labels.max() + 1, vector_data.shape[1]))
    counts = np.zeros(cluster_labels.max() + 1)
    for i in range(cluster_labels.shape[0]):
        result[cluster_labels[i]] += vector_data[cluster_labels[i]]
        counts[cluster_labels[i]] += 1

    for i in range(result.shape[0]):
        result[i] /= counts[i]

    return result


def create_cluster_layers(
    layer_class: Type[Any],
    clusterable_vectors: np.ndarray,
    embedding_vectors: np.ndarray,
    min_clusters: int = 6,
    min_samples: int = 5,
    base_min_cluster_size: int = 10,
    next_cluster_size_quantile: float = 0.8,
) -> Tuple[List[Any], Dict[Tuple[int, int], List[Tuple[int, int]]]]:
    cluster_labels = build_raw_cluster_layers(
        clusterable_vectors,
        min_clusters=min_clusters,
        min_samples=min_samples,
        base_min_cluster_size=base_min_cluster_size,
        next_cluster_size_quantile=next_cluster_size_quantile,
    )
    cluster_tree = build_cluster_tree(cluster_labels)
    layers = [
        layer_class(labels, centroids_from_labels(labels, embedding_vectors))
        for labels in cluster_labels
    ]
    return layers, cluster_tree
