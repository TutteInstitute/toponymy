from abc import ABC, abstractmethod
from typing import Any, Dict, List, NewType, Optional, Tuple, Type

import numba
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from fast_hdbscan import PLSCAN
from evoc import EVoC

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
        A dictionary representing the cluster tree. Keys are a tuple of (layer, cluster index)
        and values are lists of tuples representing child clusters.
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
            and all(
                isinstance(cluster, ClusterLayer) for cluster in self.cluster_layers_
            )
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
    for i, layer_labels in enumerate(labels):
        # Group indices by label
        indices_to_sort_layer = np.argsort(layer_labels, kind="stable")
        sorted_layer = layer_labels[indices_to_sort_layer]
        cluster_ids, change_indices = np.unique(sorted_layer, return_index=True)
        cluster_members = np.split(indices_to_sort_layer, change_indices[1:])

        clusters = [
            Cluster(cluster_id, members)
            for cluster_id, members in zip(cluster_ids, cluster_members)
            if cluster_id >= 0
        ]
        cluster_layer = ClusterLayer(clusters=clusters, layer_index=i)
        cluster_layers.append(cluster_layer)
    return cluster_layers


class PrecomputedClusterer(Clusterer):
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
        self.cluster_layers_ = build_cluster_layers(cluster_label_layers)
        self.cluster_tree_ = build_cluster_tree(cluster_label_layers)

    def fit_predict(self, cluster_label_layers):
        self.fit(cluster_label_layers)
        return self.cluster_layers_, self.cluster_tree_


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

    random_state : int or None, default=None
        The random seed to use for the random number generator. If None, the random
        number generator will not be seeded and will use the system time as the seed.

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
        random_state: Optional[int] = None,
        verbose: Optional[bool] = None,
    ):
        super().__init__()
        self.min_clusters = min_clusters
        self.base_n_clusters = base_n_clusters
        self.random_state = random_state
        self.verbose = verbose

    def fit(
        self,
        vectors: np.ndarray,
        verbose: Optional[bool] = None,
    ):
        n_clusters = self.base_n_clusters
        cluster_label_layers: List[np.ndarray] = []
        while n_clusters >= self.min_clusters:
            if self.verbose:
                print(f"Layer {len(cluster_label_layers)} found {n_clusters} clusters")
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
            cluster_labels = kmeans.fit_predict(vectors)
            cluster_label_layers.append(cluster_labels)
            n_clusters //= 4
        self.cluster_tree_ = build_cluster_tree(cluster_label_layers)
        self.cluster_layers_ = build_cluster_layers(cluster_label_layers)
        return self


class PLSCANClusterer(Clusterer):
    """
    A wrapper class for using fast_hdbscan.PLSCAN.

    Parameters
    ----------
    min_samples : int, default=5
        The number of samples in a neighbourhood for a point to be considered
        a core point.  This controls the level of smoothing applied to the
        density estimate.

    max_layers : int, default=10
        Maximum number of cluster resolution layers to return (including the
        base layer).

    base_min_cluster_size : int, default=5
        The minimum cluster size used when constructing the initial condensed
        tree.  Clusters smaller than this are treated as noise.

    base_n_clusters : int or None, default=None
        If set, the base layer is constructed by binary-searching for a
        ``min_cluster_size`` that yields approximately this many clusters.
        When ``None``, ``base_min_cluster_size`` is used directly.

    layer_similarity_threshold : float, default=0.2
        Minimum dissimilarity between selected resolution layers.  Lower
        values produce more diverse layers; higher values allow more similar
        layers to coexist.

    reproducible : bool, default=False
        If ``True``, use a slower but deterministic algorithm for the
        minimum spanning tree computation so that results are reproducible
        across runs.

    metric : str, default='euclidean'
        The distance metric to use.  ``'euclidean'`` uses a fast KD-tree
        implementation.  ``'precomputed'`` expects a sparse distance graph
        as input.  Any other metric supported by ``pynndescent`` may be
        used, but requires ``pynndescent`` to be installed.

    algorithm : str, default='boruvka'
        The MST algorithm to use.  Must be ``'boruvka'`` or ``'kruskal'``.
        ``'kruskal'`` is required when using ``cannot_link`` constraints.

    knn_k : int or None, default=None
        Number of nearest neighbours to compute when building the MST.
        When ``None``, a suitable default is chosen automatically.

    cannot_link : array-like or None, default=None
        An array of pairs ``(i, j)`` of sample indices that must not be
        placed in the same cluster.  Only supported with
        ``algorithm='kruskal'``.

    validate_cannot_link : bool, default=True
        Whether to validate ``cannot_link`` constraints before clustering.

    metric_kwds : dict or None, default=None
        Additional keyword arguments to pass to the distance metric.

    verbose : bool, default=False
        If ``True``, print progress messages during fitting.

    Attributes
    ----------
    cluster_layers_ : List[ClusterLayer]
        A list of the created cluster layers.

    cluster_tree_ : Dict[Tuple[int, int], List[Tuple[int, int]]]
        A dictionary representing the cluster tree.

    plscan_ : PLSCAN
        The fitted PLSCAN object. Algorithm specific attributes saved
        during the fit process can be accessed here.

    """

    def __init__(
        self,
        min_samples: int = 5,
        max_layers: int = 10,
        base_min_cluster_size: int = 5,
        base_n_clusters: Optional[int] = None,
        layer_similarity_threshold: float = 0.2,
        reproducible: bool = False,
        metric: str = "euclidean",
        algorithm: str = "boruvka",
        knn_k: Optional[int] = None,
        cannot_link: Optional[list[Tuple[int, int]]] = None,
        validate_cannot_link: bool = True,
        metric_kwds: dict or None = None,
        verbose: Optional[bool] = None,
    ):
        super().__init__()
        self.min_samples = min_samples
        self.max_layers = max_layers
        self.base_min_cluster_size = base_min_cluster_size
        self.base_n_clusters = base_n_clusters
        self.layer_similarity_threshold = layer_similarity_threshold
        self.reproducible = reproducible
        self.metric = metric
        self.algorithm = algorithm
        self.knn_k = knn_k
        self.cannot_link = cannot_link
        self.validate_cannot_link = validate_cannot_link
        self.metric_kwds = metric_kwds
        self.verbose = verbose

    def fit(
        self,
        vectors: np.ndarray,
        verbose: Optional[bool] = None,
    ):
        self.plscan_ = PLSCAN(
            min_samples=self.min_samples,
            max_layers=self.max_layers,
            base_min_cluster_size=self.base_min_cluster_size,
            base_n_clusters=self.base_n_clusters,
            layer_similarity_threshold=self.layer_similarity_threshold,
            reproducible=self.reproducible,
            metric=self.metric,
            algorithm=self.algorithm,
            knn_k=self.knn_k,
            cannot_link=self.cannot_link,
            validate_cannot_link=self.validate_cannot_link,
            metric_kwds=self.metric_kwds,
            verbose=self.verbose,
        )
        self.plscan_.fit(vectors)
        self.cluster_layers_ = build_cluster_layers(self.plscan_.cluster_layers_)
        self.cluster_tree_ = build_cluster_tree(self.plscan_.cluster_layers_)
        if len(self.cluster_layers_) == 0:
            raise ValueError("PLSCANClusterer found no layers with clusters.")
        return self


class EVoCClusterer(Clusterer):
    """
    A wrapper class for using the EVoC clustering algorithm. Use EVoC (Embedding
    Vector Oriented Clustering) for efficient clustering of high-dimensional
    embedding vectors, such as CLIP-vectors, sentence-transformers output, etc. The
    clustering uses a combination of a node embedding of a nearest neighbour graph,
    related to UMAP, and a density based clustering approach related to HDBSCAN,
    improving upon those approaches in efficiency and quality for the specific case
    of high-dimensional embedding vectors.

    Parameters
    ----------

    noise_level : float, default=0.5
        The noise level expected in the data. A value of 0.0 will try to cluster
        more data, at the expense of getting less accurate clustering. A value of
        1.0 will try for accurate clusters, discarding more data as noise to do so.

    base_min_cluster_size : int, default=5
        The minimum number of points in a cluster at the base layer of the clustering.
        This gives the finest granularity clustering that will be returned, with less
        granularity at higher layers.

    base_n_clusters : int or None, default=None
        If not None, the algorithm will attempt to find the granularity of
        clustering that will give exactly this many clusters for the bottom-most layer
        of clustering. This affects the base layer computation and allows multiple
        layers to be built on top of this base. Since the actual number of clusters
        cannot be guaranteed this is only approximate, but usually the algorithm can
        manage to get this exact number, assuming a reasonable clustering into
        ``base_n_clusters`` exists.

    approx_n_clusters : int, default=None
        If not None, the algorithm will attempt to find the granularity of
        clustering that will give exactly this many clusters as the final output.
        Unlike ``base_n_clusters``, when this parameter is set, only a single
        clustering layer will be returned -- no hierarchical layers will be produced.
        This is useful when you know the exact number of clusters you want and don't
        need the multi-layer analysis. Since the actual number of clusters cannot be
        guaranteed this is only approximate, but usually the algorithm can manage to
        get this exact number, assuming a reasonable clustering into ``approx_n_clusters``
        exists.

    n_neighbors : int, default=15
        The number of neighbors to use in the nearest neighbor graph construction.

    min_samples : int, default=5
        The minimum number of samples to use in the density estimation when
        performing density based clustering on the node embedding.

    n_epochs : int, default=50
        The number of epochs to use when training the node embedding.

    node_embedding_init : str or None, default='label_prop'
        The method to use to initialize the node embedding. If None, no initialization
        will be used. If 'label_prop', the label propagation method will be used.

    symmetrize_graph : bool, default=True
        Whether to symmetrize the nearest neighbor graph before using it to
        construct the node embedding.

    node_embedding_dim : int or None, default=None
        The number of dimensions to use in the node embedding. If None, a default
        value of min(max(n_neighbors // 4, 4), 15) will be used.

    neighbor_scale : float, default=1.0
        The scale factor to use when constructing the nearest neighbor graph. This
        multiplies the effective number of neighbors used in graph construction
        (neighbor_scale * n_neighbors). Values > 1.0 create denser graphs with more
        connectivity, potentially capturing more global structure but at increased
        computational cost. Values < 1.0 create sparser graphs focused on local
        structure.

    random_state : int or None, default=None
        The random seed to use for the random number generator. If None, the random
        number generator will not be seeded and will use the system time as the seed.

    min_similarity_threshold : float, default=0.2
        The minimum similarity threshold for cluster layer selection. Peaks that result
        in clusterings with Jaccard similarity above this threshold will be filtered out
        to ensure diverse cluster layers.

    max_layers : int, default=10
        The maximum number of cluster layers to return. The algorithm will select up to
        this many diverse peaks based on persistence and similarity criteria.

    n_label_prop_iter : int, default=20
        The number of iterations to use in the label propagation algorithm when
        initializing the node embedding. This parameter controls how many steps
        the label propagation process takes to converge when node_embedding_init
        is set to 'label_prop'.

    Attributes
    ----------
    cluster_layers_ : List[ClusterLayer]
        A list of the created cluster layers.

    cluster_tree_ : Dict[Tuple[int, int], List[Tuple[int, int]]]
        A dictionary representing the cluster tree.

    evoc_ : EVoC
        The fitted EVoC object. Algorithm specific attributes saved
        during the fit process can be accessed here.

    """

    def __init__(
        self,
        noise_level: float = 0.5,
        base_min_cluster_size: int = 5,
        base_n_clusters: Optional[int] = None,
        approx_n_clusters: Optional[int] = None,
        n_neighbors: int = 15,
        min_samples: int = 5,
        n_epochs: int = 50,
        node_embedding_init: Optional[str] = "label_prop",
        symmetrize_graph: bool = True,
        node_embedding_dim: Optional[int] = None,
        neighbor_scale: float = 1.0,
        random_state: Optional[int] = None,
        min_similarity_threshold: float = 0.2,
        max_layers: int = 10,
        n_label_prop_iter: int = 20,
        verbose: Optional[bool] = False,
    ):
        super().__init__()
        self.noise_level = noise_level
        self.base_min_cluster_size = base_min_cluster_size
        self.base_n_clusters = base_n_clusters
        self.approx_n_clusters = approx_n_clusters
        self.n_neighbors = n_neighbors
        self.min_samples = min_samples
        self.n_epochs = n_epochs
        self.node_embedding_init = node_embedding_init
        self.symmetrize_graph = symmetrize_graph
        self.node_embedding_dim = node_embedding_dim
        self.neighbor_scale = neighbor_scale
        self.random_state = random_state
        self.min_similarity_threshold = min_similarity_threshold
        self.max_layers = max_layers
        self.n_label_prop_iter = n_label_prop_iter
        self.verbose = verbose

    def fit(
        self,
        vectors: np.ndarray,
    ):
        self.evoc_ = EVoC(
            noise_level=self.noise_level,
            base_min_cluster_size=self.base_min_cluster_size,
            base_n_clusters=self.base_n_clusters,
            approx_n_clusters=self.approx_n_clusters,
            n_neighbors=self.n_neighbors,
            min_samples=self.min_samples,
            n_epochs=self.n_epochs,
            node_embedding_init=self.node_embedding_init,
            symmetrize_graph=self.symmetrize_graph,
            node_embedding_dim=self.node_embedding_dim,
            neighbor_scale=self.neighbor_scale,
            random_state=self.random_state,
            min_similarity_threshold=self.min_similarity_threshold,
            max_layers=self.max_layers,
            n_label_prop_iter=self.n_label_prop_iter,
        )
        self.evoc_.fit(vectors)
        self.cluster_layers_ = build_cluster_layers(self.evoc_.cluster_layers_)
        self.cluster_tree_ = self.evoc_.cluster_tree_
        if len(self.cluster_layers_) == 0:
            raise ValueError("EVoCClusterer found no layers with clusters.")
        return self
