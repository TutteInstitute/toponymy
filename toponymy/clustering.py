from abc import ABC, abstractmethod
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
from sklearn.neighbors import KDTree
from typing import List, Tuple, Dict, Type, Any, Optional
from toponymy.cluster_layer import ClusterLayer


def build_raw_cluster_layers(
    data: np.ndarray,
    *,
    min_clusters: int = 3,
    min_samples: int = 5,
    base_min_cluster_size: int = 10,
    next_cluster_size_quantile: float = 0.8,
) -> List[np.ndarray]:
    """
    Build hierarchical cluster layers from raw data using a KDTree and Boruvka's algorithm.

    Parameters
    ----------
    data : np.ndarray
        The input data array of shape (n_samples, n_features).
    min_clusters : int, optional
        The minimum number of clusters to form in each layer, by default 3.
    min_samples : int, optional
        The minimum number of samples in a cluster, by default 5.
    base_min_cluster_size : int, optional
        The initial minimum cluster size, by default 10.
    next_cluster_size_quantile : float, optional
        The quantile to determine the next minimum cluster size, by default 0.8.

    Returns
    -------
    List[np.ndarray]
        A list of numpy arrays, each representing cluster labels for a layer.
    """
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
def _build_cluster_tree(labels: np.ndarray) -> List[Tuple[int, int, int, int]]:

    mapping = [(-1, -1, -1, -1) for _ in range(0)]
    found = [set([-1]) for _ in range(len(labels))]
    for upper_layer in range(1, labels.shape[0]):
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

    for lower_layer in range(labels.shape[0] - 1, -1, -1):
        for child in range(labels[lower_layer].max() + 1):
            if child >= 0 and child not in found[lower_layer]:
                mapping.append((labels.shape[0], 0, lower_layer, child))

    return mapping


def build_cluster_tree(
    labels: List[np.ndarray],
) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """
    Builds a cluster tree from the given labels.

    Parameters
    ----------
    labels : List[np.ndarray]
        A list of numpy arrays where each array represents the labels of clusters at a specific layer.

    Returns
    -------
    Dict[Tuple[int, int], List[Tuple[int, int]]]
        A dictionary where the keys are tuples representing the parent cluster (layer, cluster index)
        and the values are lists of tuples representing the child clusters (layer, cluster index).
    """
    result = {}
    raw_mapping = _build_cluster_tree(np.vstack(labels))
    for parent_layer, parent_cluster, child_layer, child_cluster in raw_mapping:
        parent_name = (parent_layer, parent_cluster)
        if parent_name in result:
            result[parent_name].append((child_layer, child_cluster))
        else:
            result[parent_name] = [(child_layer, child_cluster)]
    return result


@numba.njit()
def centroids_from_labels(
    cluster_labels: np.ndarray, vector_data: np.ndarray
) -> np.ndarray:
    result = np.zeros((cluster_labels.max() + 1, vector_data.shape[1]))
    counts = np.zeros(cluster_labels.max() + 1)
    for i in range(cluster_labels.shape[0]):
        cluster_num = cluster_labels[i]
        if cluster_num >= 0:
            result[cluster_num] += vector_data[i]
            counts[cluster_num] += 1

    for i in range(result.shape[0]):
        if counts[i] > 0:
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
) -> Tuple[List[ClusterLayer], Dict[Tuple[int, int], List[Tuple[int, int]]]]:
    """
    Create cluster layers from given vectors and parameters.

    Parameters
    ----------
    layer_class : Type[Any]
        The class to be used for creating layers.
    clusterable_vectors : np.ndarray
        The vectors that can be clustered.
    embedding_vectors : np.ndarray
        The embedding vectors corresponding to the clusterable vectors.
    min_clusters : int, optional
        The minimum number of clusters to form in a layer (default is 6).
    min_samples : int, optional
        The minimum number of samples for hdbscan style clustering (default is 5).
    base_min_cluster_size : int, optional
        The base minimum size of clusters for the most fine-grained cluster layer (default is 10).
    next_cluster_size_quantile : float, optional
        The quantile value to determine the size of the minimum cluster size for the next layer (default is 0.8).

    Returns
    -------
    Tuple[List[Any], Dict[Tuple[int, int], List[Tuple[int, int]]]]
        A tuple containing a list of created layers and a dictionary representing the cluster tree.
    """
    cluster_labels = build_raw_cluster_layers(
        clusterable_vectors,
        min_clusters=min_clusters,
        min_samples=min_samples,
        base_min_cluster_size=base_min_cluster_size,
        next_cluster_size_quantile=next_cluster_size_quantile,
    )
    cluster_tree = build_cluster_tree(cluster_labels)
    layers = [
        layer_class(labels, centroids_from_labels(labels, embedding_vectors), layer_id=i)
        for i, labels in enumerate(cluster_labels)
    ]
    return layers, cluster_tree


class Clusterer(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def fit(self, clusterable_vectors: np.ndarray, embedding_vectors: np.ndarray, layer_class: Type[ClusterLayer]):
        pass

    @abstractmethod
    def fit_predict(self, clusterable_vectors: np.ndarray, embedding_vectors: np.ndarray, layer_class: Type[ClusterLayer]):
        pass

class ToponymyClusterer(Clusterer):

    def __init__(
        self,
        min_clusters: int = 6,
        min_samples: int = 5,
        base_min_cluster_size: int = 10,
        next_cluster_size_quantile: float = 0.85,
    ):
        super().__init__()
        self.min_clusters = min_clusters
        self.min_samples = min_samples
        self.base_min_cluster_size = base_min_cluster_size
        self.next_cluster_size_quantile = next_cluster_size_quantile

    def fit(self, clusterable_vectors: np.ndarray, embedding_vectors: np.ndarray, layer_class: Type[ClusterLayer]):
        self.cluster_layers_, self.cluster_tree_ = create_cluster_layers(
            layer_class,
            clusterable_vectors=clusterable_vectors,
            embedding_vectors=embedding_vectors,
            min_clusters=self.min_clusters,
            min_samples=self.min_samples,
            base_min_cluster_size=self.base_min_cluster_size,
            next_cluster_size_quantile=self.next_cluster_size_quantile,
        )
        return self

    def fit_predict(
        self, clusterable_vectors: np.ndarray, embedding_vectors: np.ndarray, layer_class: Type[ClusterLayer]
    ):
        self.fit(clusterable_vectors, embedding_vectors, layer_class=layer_class)
        return self.cluster_layers_, self.cluster_tree_


try:
    import evoc

    class EVoCClusterer(Clusterer):

        def __init__(
            self,
            noise_level: float = 0.5,
            base_min_cluster_size: int = 5,
            min_num_clusters: int = 4,
            n_neighbors: int = 15,
            min_samples: int = 5,
            next_cluster_size_quantile: float = 0.85,
            n_epochs: int = 50,
            node_embedding_init: str = "label_prop",
            symmetrize_graph: bool = True,
            node_embedding_dim: Optional[int] = None,
            neighbor_scale: float = 1.0,
        ):
            super().__init__()
            self.evoc = evoc.EVoC(
                noise_level=noise_level,
                base_min_cluster_size=base_min_cluster_size,
                min_num_clusters=min_num_clusters,
                n_neighbors=n_neighbors,
                min_samples=min_samples,
                next_cluster_size_quantile=next_cluster_size_quantile,
                n_epochs=n_epochs,
                node_embedding_init=node_embedding_init,
                symmetrize_graph=symmetrize_graph,
                node_embedding_dim=node_embedding_dim,
                neighbor_scale=neighbor_scale,
            )

        def fit(self, clusterable_vectors: np.ndarray, embedding_vectors: np.ndarray, layer_class: Type[ClusterLayer]):
            self.evoc.fit(embedding_vectors)
            cluster_labels = self.evoc.labels_
            self.cluster_tree_ = build_cluster_tree(cluster_labels)
            self.cluster_layers_ = [
                layer_class(
                    labels, centroids_from_labels(labels, embedding_vectors)
                )
                for labels in cluster_labels
            ]
            return self

        def fit_predict(self, clusterable_vectors: np.ndarray, embedding_vectors: np.ndarray, layer_class: Type[ClusterLayer]):
            self.fit(clusterable_vectors, embedding_vectors, layer_class=layer_class)
            return self.cluster_layers_, self.cluster_tree_

except ImportError:
    pass
