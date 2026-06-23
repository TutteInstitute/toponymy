from abc import ABC, abstractmethod
from typing import Any, Dict, List, NewType, Optional, Tuple, Type

import numba
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from fast_hdbscan import PLSCAN

from toponymy.new_types import Cluster, ClusterLayer, ClusterTree
from toponymy._utils import handle_verbose_params


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
            and all(isinstance(cluster, ClusterLayer) for cluster in self.cluster_layers_)
            and hasattr(self, "cluster_tree_")
            and isinstance(self.cluster_tree_, dict)
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
    result: ClusterTree = {}
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
        cluster_layer = ClusterLayer(clusters=clusters, layer_index=i)
        cluster_layers.append(cluster_layer)
    return cluster_layers


class PreComputedClusterer(Clusterer):
    """
    A class for formatting a precomputed set of cluster layers as a clusterer.
    Pass a list of layer labels (i.e. each layer is a list of cluster ids).

    Attributes
    ----------
    cluster_layers_ : List[ClusterLayer]
        A list of the created cluster layers.
    cluster_tree_ : Dict[Tuple[int, int], List[Tuple[int, int]]]
        A dictionary representing the cluster tree.
    """
    def __init__(self):
        super().__init__()

    def fit(self, cluster_label_layers):
        self.cluster_layers = build_cluster_layers(cluster_label_layers)
        self.cluster_tree = build_cluster_tree(cluster_label_layers)

    def fit_predict(self, cluster_label_layers):
        self.fit(label_layers)
        return self.cluster_layers, self.cluster_tree


class KMeansClusterer(Clusterer):
    """
    A class for clustering data in layers using KMeans. This class is mostly to demonstrate how one might write
    an alternative Clusterer.

    Parameters
    ----------
    min_clusters : int, optional
        The minimum number of clusters to form in a layer (default is 6).
    base_n_clusters : int, optional
        The initial number of clusters for the most fine-grained cluster layer (default is 1024).

    Attributes
    ----------
    cluster_layers_ : List[ClusterLayer]
        A list of the created cluster layers.
    cluster_tree_ : ClusterTree]
        A dictionary representing the cluster tree. Keys are a tuple of (layer, cluster index) and values are lists of
        tuples representing child clusters.
    """

    def __init__(
        self,
        min_clusters: int = 6,
        base_n_clusters: int = 1024,
        verbose: Optional[bool] = None,
        show_progress_bar: Optional[bool] = None,
    ):
        super().__init__()
        self.min_clusters = min_clusters
        self.base_n_clusters = base_n_clusters

        # Handle verbose parameters
        self.show_progress_bar, self.verbose = handle_verbose_params(
            verbose=verbose, show_progress_bar=show_progress_bar, default_verbose=False
        )

    def fit(
        self,
        vectors: np.ndarray,
        verbose: Optional[bool] = None,
        show_progress_bar: Optional[bool] = None,
    ) -> Clusterer:
        # Handle verbose parameters
        _, verbose_output = handle_verbose_params(
            verbose=verbose if verbose is not None else self.verbose,
            show_progress_bar=show_progress_bar,
            default_verbose=False,
        )

        n_clusters = self.base_n_clusters
        cluster_label_layers: List[np.ndarray] = []
        while n_clusters >= self.min_clusters:
            if verbose_output:
                print(f"Layer {len(cluster_label_layers)} found {n_clusters} clusters")
            kmeans = KMeans(n_clusters=n_clusters)
            cluster_labels = kmeans.fit_predict(vectors)
            cluster_label_layers.append(cluster_labels)
            n_clusters //= 4
        self.cluster_tree_ = build_cluster_tree(cluster_label_layers)
        self.cluster_layers_ = build_cluster_layers(cluster_label_layers)
        return self


class PLSCANClusterer(Clusterer):
    """
    A class for clustering dense vector data in layers using fast_hdbscan.PLSCAN.

    Parameters
    ----------
    min_clusters : int, optional
        The minimum number of non-noise clusters to keep in a layer (default is 6).
    min_samples : int, optional
        The minimum number of samples used by PLSCAN (default is 5).
    base_min_cluster_size : int, optional
        The base minimum cluster size passed to PLSCAN (default is 10).
    max_layers : Optional[int], optional
        The maximum number of hierarchy layers to keep, including the base layer (default is 10).
    verbose : bool, optional
        Whether to show progress bars and verbose output. If True, shows all output. If False, suppresses all output.
    show_progress_bar : bool, optional, deprecated
        Deprecated. Use verbose instead.

    Attributes
    ----------
    cluster_layers_ : List[ClusterLayer]
        A list of the created cluster layers.
    cluster_tree_ : Dict[Tuple[int, int], List[Tuple[int, int]]]
        A dictionary representing the cluster tree.
    cluster_probabilities_ : List[np.ndarray]
        Membership probabilities for each returned layer.
    cluster_persistence_scores_ : List[float]
        Persistence scores for each returned layer.
    plscan_min_cluster_sizes_ : Optional[np.ndarray]
        The minimum cluster sizes explored by PLSCAN, when exposed by the
        upstream implementation.
    """
    def __init__(
        self,
        min_clusters: int = 6,
        min_samples: int = 5,
        base_min_cluster_size: int = 10,
        max_layers: Optional[int] = 10,
        reproducible: bool = False,
        verbose: Optional[bool] = None,
        show_progress_bar: Optional[bool] = None,
    ):
        super().__init__()
        self.min_clusters = min_clusters
        self.min_samples = min_samples
        self.base_min_cluster_size = base_min_cluster_size
        self.max_layers = max_layers
        self.reproducible = reproducible

        self.show_progress_bar, self.verbose = handle_verbose_params(
            verbose=verbose, show_progress_bar=show_progress_bar, default_verbose=False
        )

    def fit(
        self,
        vectors: np.ndarray,
        verbose: Optional[bool] = None,
        show_progress_bar: Optional[bool] = None,
    ) -> Clusterer:
        show_progress_bar_val, verbose_output = handle_verbose_params(
            verbose=verbose if verbose is not None else self.verbose,
            show_progress_bar=show_progress_bar,
            default_verbose=False,
        )

        vectors = np.ascontiguousarray(
            vectors, dtype=np.float32
        )

        self.plscan_ = PLSCAN(
            min_samples=self.min_samples,
            base_min_cluster_size=self.base_min_cluster_size,
            max_layers=self.max_layers,
            reproducible=self.reproducible,
            verbose=verbose_output,
        )
        self.plscan_.fit(vectors)

        raw_cluster_layers = zip(
            self.plscan_.cluster_layers_,
            self.plscan_.membership_strength_layers_,
            self.plscan_.layer_persistence_scores_,
        )
        filtered_cluster_layers = []
        for labels, probabilities, persistence_score in raw_cluster_layers:
            # Normalize labels to be contiguous, preserving noise as -1
            unique_labels, inverse = np.unique(labels, return_inverse=True)
            if unique_labels.size > 0 and unique_labels[0] == -1:
                labels = inverse - 1
            else:
                labels = inverse

            n_clusters_in_layer = labels.max() + 1
            if n_clusters_in_layer < self.min_clusters:
                continue
            if self.verbose:
                print(
                    f"Layer {len(filtered_cluster_layers)} found {n_clusters_in_layer} clusters"
                )
            filtered_cluster_layers.append((labels, probabilities, persistence_score))

        if len(filtered_cluster_layers) == 0:
            raise ValueError(
                "Not enough clusters found in any PLSCAN layer: "
                f"min_clusters={self.min_clusters}."
            )

        cluster_label_layers, self.cluster_probabilities_, self.cluster_persistence_scores_ = list(
            map(list, zip(*filtered_cluster_layers))
        )
        self.plscan_min_cluster_sizes_ = getattr(
            self.plscan_, "min_cluster_sizes_", None
        )
        self.cluster_layers_ = build_cluster_layers(cluster_label_layers)
        self.cluster_tree_ = build_cluster_tree(cluster_label_layers)
        return self
