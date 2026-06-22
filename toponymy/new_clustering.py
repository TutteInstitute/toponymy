from abc import ABC, abstractmethod
from typing import Any, Dict, List, NewType, Optional, Tuple, Type

import numba
import numpy as np
from sklearn.base import BaseEstimator

from toponymy.new_types import Cluster, ClusterLayer, ClusterTree


class Clusterer(ABC, BaseEstimator):
    """
    Abstract Clusterer Class that defines the properties required of a clusterer
    and implements formatter / validator methods that should be common to all
    subclasses.

    Attributes
    ----------
    cluster_layers_ : List[ClusterLayer]
        A list of the created cluster layers.
    cluster_tree_ : Dict[Tuple[int, int], List[Tuple[int, int]]]
        A dictionary representing the cluster tree. Keys are a tuple of (layer, cluster index) and values are lists of
        tuples representing child clusters.
    """
    def __init__(self):
        pass

    @abstractmethod
    def fit(
        self,
        data: Any,
        **layer_kwargs,
    ):
        pass

    def fit_predict(
        self,
        data: Any,
    ) -> Tuple[List[ClusterLayer], ClusterTree]:
        self.fit(data)
        return self.cluster_layers_, self.cluster_tree_

    def __sklearn_is_fitted__(self):
        if (
            hasattr(self, "cluster_layers_")
            and isinstance(self.cluster_layers_, list)
            and all(isinstance(cluster, ClusterLayer) for cluster in self.cluster_layers)
            and hasattr(self, "cluster_tree_")
            and isinstance(self.cluster_tree_, ClusterTree)
        ):
            return True
        return False


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
            # If there is no noise we are off by one, and need to drop the first cluster group
            if len(cluster_groups) > upper_layer_unique_labels.shape[0]:
                cluster_groups = cluster_groups[1:]
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


def build_cluster_tree(labels: List[np.ndarray]) -> ClusterTree:
    """
    Builds a cluster tree from the given labels.

    Parameters
    ----------
    labels : List[np.ndarray]
        A list of numpy arrays where labels[i][j] is the label of the cluster of data j at layer i
        (label -1 denotes noise).

    Returns
    -------
    ClusterTree
        A dictionary where the keys are tuples representing the parent cluster (layer, cluster index)
        and the values are lists of tuples representing the child clusters (layer, cluster index).
    """
    result: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    raw_mapping = _build_cluster_tree(np.vstack(labels))
    for parent_layer, parent_cluster, child_layer, child_cluster in raw_mapping:
        parent_name = (parent_layer, parent_cluster)
        if parent_name in result:
            result[parent_name].append((child_layer, child_cluster))
        else:
            result[parent_name] = [(child_layer, child_cluster)]
    return result


def build_cluster_layers(labels: List[np.ndarray]) -> List[ClusterLayer]:
    """
    Builds a list of cluster layers and a cluster tree from the given layers.

    Parameters
    ----------
    labels : List[np.ndarray]
        A list of numpy arrays where labels[i][j] is the label of the cluster of data j at layer i
        (label -1 denotes noise).

    Returns
    -------
    ClusterLayers
        A list of ClusterLayers
    ClusterTree
        A dictionary where the keys are tuples representing the parent cluster (layer, cluster index)
        and the values are lists of tuples representing the child clusters (layer, cluster index).
    """
    cluster_layers: List[ClusterLayer] = []
    for i, layer in enumerate(labels):
        clusters: List[Cluster] = []
        # Create a dict mapping cluster_id to list of indices in that cluster
        indices_to_sort_layer = np.argsort(layer, kind='stable')
        sorted_layer = layer[indices_to_sort_layer]
        cluster_ids, change_indices = np.unique(sorted_layer, return_index=True) # Cluster ids are garunteed to be sorted ascending
        grouped_indices = np.split(indices_to_sort_layer, change_indices[1:])
        for cluster_id, members in zip(cluster_ids, grouped_indices):
            if cluster_id < 0:
                continue
            cluster = Cluster(cluster_id, members)
            clusters.append(cluster)
        cluster_layer = ClusterLayer(clusters, i)
        cluster_layers.append(cluster_layer)
    return cluster_layers


class PreComputedClusterer(Clusterer):
    def __init__(self):
        super().__init__()
        pass

    def fit(self, cluster_label_layers):
        self.cluster_layers = build_cluster_layers(cluster_label_layers)
        self.cluster_tree = build_cluster_tree(cluster_label_layers)

    def fit_predict(self, cluster_label_layers):
        self.fit(label_layers)
        return self.cluster_layers, self.cluster_tree

