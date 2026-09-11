"""Validated clustering output and small adapters to maintained algorithms."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Iterator, Optional, Tuple
import warnings

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from toponymy.types import (
    Cluster,
    ClusterLayer,
    ClusterTree,
    _integer_vector,
    _nonnegative_integer,
)
from toponymy.utility_functions import centroids_from_labels as centroids_from_labels


def _validate_label_layers(labels: Sequence[np.ndarray]) -> list[np.ndarray]:
    if isinstance(labels, np.ndarray) and labels.ndim != 2:
        raise ValueError("labels must be a sequence of one-dimensional label layers")
    result = [_integer_vector(layer, "labels", -1) for layer in labels]
    if result and any(layer.size != result[0].size for layer in result[1:]):
        raise ValueError("all label layers must have the same observation count")
    return result


def _group_labels(labels: np.ndarray):
    indices = np.flatnonzero(labels >= 0)
    order = indices[np.argsort(labels[indices], kind="stable")]
    ids, starts = np.unique(labels[order], return_index=True)
    return ids, starts, order


def build_cluster_layers(labels: Sequence[np.ndarray]) -> list[ClusterLayer]:
    """Group observations by original nonnegative IDs, excluding noise (-1).

    Layers have equal observation counts. IDs must have integer dtype, need
    not be contiguous, and are never used as allocation sizes. Empty layers
    and all-noise layers are valid. Returned arrays are owned and read-only.
    """
    layers = []
    for layer_index, layer_labels in enumerate(_validate_label_layers(labels)):
        ids, starts, order = _group_labels(layer_labels)
        members = np.split(order, starts[1:])
        clusters = tuple(
            Cluster(int(label), group) for label, group in zip(ids, members)
        )
        layers.append(ClusterLayer(clusters, layer_index, layer_labels))
    return layers


def build_cluster_tree(labels: Sequence[np.ndarray]) -> ClusterTree:
    """Attach each cluster to its nearest fully containing upper cluster.

    Crossing clusters and clusters covered partly by upper-layer noise skip
    that layer. Clusters without a containing ancestor attach to the synthetic
    root ``(number_of_layers, 0)``. No nodes are invented for absent IDs.
    """
    labels = _validate_label_layers(labels)
    tree: ClusterTree = {}
    root = (len(labels), 0)
    for lower_index, lower in enumerate(labels):
        ids, starts, order = _group_labels(lower)
        if not ids.size:
            continue
        unresolved = np.ones(ids.size, dtype=bool)
        for upper_index in range(lower_index + 1, len(labels)):
            upper = labels[upper_index][order]
            minimum = np.minimum.reduceat(upper, starts)
            maximum = np.maximum.reduceat(upper, starts)
            contained = unresolved & (minimum >= 0) & (minimum == maximum)
            for child_id, parent_id in zip(ids[contained], minimum[contained]):
                parent = (upper_index, int(parent_id))
                tree.setdefault(parent, []).append((lower_index, int(child_id)))
            unresolved[contained] = False
            if not unresolved.any():
                break
        for child_id in ids[unresolved]:
            tree.setdefault(root, []).append((lower_index, int(child_id)))
    return tree


def validate_cluster_tree(tree: ClusterTree, layers: Sequence[ClusterLayer]) -> None:
    """Reject unknown nodes, duplicate parents, missing nodes and false edges.

    Strictly increasing parent layer indices also rule out cycles. A supplied
    tree may skip a containing layer, but every edge must be true containment.
    """
    if not isinstance(tree, Mapping):
        raise TypeError("cluster_tree must be a mapping")
    if any(
        not isinstance(layer, ClusterLayer) or layer.layer_index != i
        for i, layer in enumerate(layers)
    ):
        raise ValueError("layers must be ClusterLayers in consecutive index order")
    if layers and any(len(layer.labels) != len(layers[0].labels) for layer in layers):
        raise ValueError("all label layers must have the same observation count")
    clusters = {
        (layer.layer_index, cluster.label): cluster
        for layer in layers
        for cluster in layer
    }
    root = (len(layers), 0)

    def check_key(key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError("tree nodes must be (layer, cluster ID) tuples")
        for value in key:
            _nonnegative_integer(value, "tree node index")

    children_seen = set()
    for parent, children in tree.items():
        check_key(parent)
        if parent != root and parent not in clusters:
            raise ValueError(f"unknown parent cluster {parent}")
        if not isinstance(children, (list, tuple)):
            raise ValueError("tree children must be a list or tuple")
        for child in children:
            check_key(child)
            if child not in clusters:
                raise ValueError(f"unknown child cluster {child}")
            if parent[0] <= child[0]:
                raise ValueError("parent layer must be strictly above its child")
            if child in children_seen:
                raise ValueError(f"cluster {child} must have exactly one parent")
            children_seen.add(child)
            if parent != root and not np.all(
                layers[parent[0]].labels[clusters[child].members] == parent[1]
            ):
                raise ValueError(f"parent {parent} does not contain child {child}")
    if children_seen != clusters.keys():
        raise ValueError("every cluster must occur exactly once as a child")


def _validate_vectors(vectors, *, precomputed: bool = False):
    """Validate without copying large matrices; estimators own their work arrays."""
    if sparse.issparse(vectors):
        if not precomputed:
            raise ValueError("sparse vectors require metric='precomputed'")
        array = vectors.tocsr(copy=False)
        values = array.data
    else:
        if precomputed:
            raise ValueError("metric='precomputed' requires a sparse distance graph")
        array = np.asarray(vectors)
        values = array
    if array.ndim != 2 or (array.shape[1] == 0 and array.shape[0] != 0):
        raise ValueError("vectors must have shape (observations, nonzero dimensions)")
    if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        raise ValueError("vectors must contain finite real numbers")
    if precomputed and (array.shape[0] != array.shape[1] or np.any(values < 0)):
        raise ValueError("a precomputed distance graph must be square and nonnegative")
    return array


class Clusterer(ABC, BaseEstimator):
    """Estimator whose fitted state consists of cluster layers and their tree."""

    @abstractmethod
    def fit(self, data, **kwargs):
        """Fit this estimator and return it."""

    def fit_predict(self, data=None, **kwargs):
        if data is None and "clusterable_vectors" in kwargs:
            data = kwargs.pop("clusterable_vectors")
        self.fit(data, **kwargs)
        return self.cluster_layers_, self.cluster_tree_

    def __iter__(self) -> Iterator[ClusterLayer]:
        if not self.__sklearn_is_fitted__():
            raise NotFittedError("fit the clusterer before iterating its layers")
        return iter(self.cluster_layers_)

    def __sklearn_is_fitted__(self):
        return hasattr(self, "cluster_layers_") and hasattr(self, "cluster_tree_")

    def _set_labels(self, labels, *, tree=None):
        layers = build_cluster_layers(labels)
        if tree is None:
            tree = build_cluster_tree([layer.labels for layer in layers])
        validate_cluster_tree(tree, layers)
        self.cluster_layers_ = layers
        self.cluster_tree_ = {
            parent: list(children) for parent, children in tree.items()
        }
        return self


class PrecomputedClusterer(Clusterer):
    """Use precomputed label layers, preserving their IDs and observation order.

    Supply labels at construction for use in a pipeline: ``fit(vectors)`` then
    validates the observation count without interpreting vectors as labels.
    Without configured labels, ``fit(label_layers)`` retains the earlier API.
    Explicit ``fit(vectors, labels=label_layers)`` is also supported.
    """

    def __init__(self, labels=None, cluster_tree=None):
        self.labels = None if labels is None else _validate_label_layers(labels)
        if self.labels is not None:
            for layer in self.labels:
                layer.flags.writeable = False
        self.cluster_tree = (
            None
            if cluster_tree is None
            else {parent: list(children) for parent, children in cluster_tree.items()}
        )

    def __sklearn_clone__(self):
        return type(self)(labels=self.labels, cluster_tree=self.cluster_tree)

    def fit(self, data=None, *, labels=None):
        configured = labels if labels is not None else self.labels
        if configured is None:
            if data is None:
                raise ValueError("supply precomputed labels at construction or fit")
            configured = _validate_label_layers(data)
        else:
            configured = _validate_label_layers(configured)
            if data is not None:
                vectors = _validate_vectors(data, precomputed=sparse.issparse(data))
                if configured and vectors.shape[0] != configured[0].size:
                    raise ValueError(
                        "vectors and labels must have the same observation count"
                    )
        return self._set_labels(configured, tree=self.cluster_tree)


class KMeansClusterer(Clusterer):
    """Fit independent KMeans resolutions, reducing cluster counts by four.

    The tree uses containment, since independently fitted resolutions need not
    nest. The finest layer is capped at the observation count.
    """

    def __init__(
        self, min_clusters=6, base_n_clusters=1024, random_state=None, verbose=None
    ):
        self.min_clusters = min_clusters
        self.base_n_clusters = base_n_clusters
        self.random_state = random_state
        self.verbose = verbose

    def fit(
        self,
        vectors=None,
        *,
        clusterable_vectors=None,
        verbose=None,
    ):
        from sklearn.cluster import KMeans

        if vectors is None:
            vectors = clusterable_vectors
        vectors = _validate_vectors(vectors)
        for name in ("min_clusters", "base_n_clusters"):
            value = _nonnegative_integer(getattr(self, name), name)
            if value == 0:
                raise ValueError(f"{name} must be positive")
        if self.base_n_clusters < self.min_clusters:
            raise ValueError("base_n_clusters must be at least min_clusters")
        if not vectors.shape[0]:
            return self._set_labels([])
        n_clusters = min(self.base_n_clusters, vectors.shape[0])
        labels = []
        while not labels or n_clusters >= self.min_clusters:
            estimator = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                verbose=bool(self.verbose if verbose is None else verbose),
            )
            labels.append(estimator.fit_predict(vectors))
            n_clusters //= 4
            if n_clusters == 0:
                break
        self._set_labels(labels)
        return self


def _validate_density_parameters(estimator):
    for name in ("min_samples", "base_min_cluster_size", "max_layers"):
        if name == "max_layers" and getattr(estimator, name) is None:
            continue
        value = _nonnegative_integer(getattr(estimator, name), name)
        if value < (2 if name == "base_min_cluster_size" else 1):
            raise ValueError(f"{name} is too small")


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
        min_clusters: int = 1,
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
        metric_kwds: Optional[dict] = None,
        verbose: Optional[bool] = None,
    ):
        super().__init__()
        self.min_clusters = min_clusters
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
        vectors: np.ndarray = None,
        *,
        clusterable_vectors=None,
        verbose: Optional[bool] = None,
    ):
        if vectors is None:
            vectors = clusterable_vectors
        vectors = _validate_vectors(vectors, precomputed=self.metric == "precomputed")
        _validate_density_parameters(self)
        min_clusters = _nonnegative_integer(self.min_clusters, "min_clusters")
        if min_clusters == 0:
            raise ValueError("min_clusters must be positive")
        if vectors.shape[0] < max(2, self.min_samples, self.base_min_cluster_size):
            self.cluster_probabilities_ = []
            self.cluster_persistence_scores_ = []
            self.plscan_min_cluster_sizes_ = None
            self.plscan_ = None
            if not vectors.shape[0]:
                return self._set_labels([])
            return self._set_labels([np.full(vectors.shape[0], -1, dtype=np.int64)])
        from fast_hdbscan import PLSCAN

        options = self.get_params(deep=False)
        options.pop("min_clusters")
        if options["max_layers"] is None:
            options["max_layers"] = max(1, vectors.shape[0])
        options["verbose"] = bool(self.verbose if verbose is None else verbose)
        estimator = PLSCAN(**options)
        estimator.fit(vectors)
        labels = [
            np.asarray(layer, dtype=np.int64)
            for layer in getattr(estimator, "cluster_layers_", [])
            if np.unique(np.asarray(layer)[np.asarray(layer) >= 0]).size >= min_clusters
        ]
        if not labels and vectors.shape[0]:
            labels = [np.full(vectors.shape[0], -1, dtype=np.int64)]
        if self.max_layers is not None:
            labels = labels[: self.max_layers]
        self._set_labels(labels)
        self.cluster_probabilities_ = [
            np.asarray(probability)
            for probability in getattr(
                estimator,
                "membership_strength_layers_",
                [np.ones_like(layer, dtype=float) for layer in labels],
            )
        ][: len(labels)]
        self.cluster_persistence_scores_ = list(
            getattr(estimator, "layer_persistence_scores_", [1.0 for _ in labels])
        )[: len(labels)]
        self.plscan_min_cluster_sizes_ = getattr(estimator, "min_cluster_sizes_", None)
        self.plscan_ = estimator
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

    isolated : bool, default=True
        Fit EVoC in a fresh Python process to avoid a Numba type-cache collision
        with fast_hdbscan. Input is handed off through a temporary memory-mapped
        file, and the fitted EVoC data is returned. This adds process startup,
        compilation and disk I/O costs. Set False only when EVoC is the sole
        clustering library executing Numba kernels in this process.

    Attributes
    ----------
    cluster_layers_ : List[ClusterLayer]
        A list of the created cluster layers.

    cluster_tree_ : Dict[Tuple[int, int], List[Tuple[int, int]]]
        A dictionary representing the cluster tree.

    evoc_ : types.SimpleNamespace or None
        Fitted native data and constructor values, including the materialized
        native ``cluster_tree_``. This container has no estimator methods. Use this
        adapter's fit or fit_predict to refit, and get_params or set_params to
        manage configuration. None when the input is too small for EVoC.

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
        isolated: bool = True,
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
        self.isolated = isolated

    def fit(self, vectors: np.ndarray):
        vectors = _validate_vectors(vectors)
        _validate_density_parameters(self)
        if not isinstance(self.isolated, bool):
            raise ValueError("isolated must be a boolean")
        if not vectors.shape[0]:
            self.evoc_ = None
            return self._set_labels([])
        if vectors.shape[0] < max(2, self.min_samples, self.base_min_cluster_size):
            self.evoc_ = None
            return self._set_labels([np.full(vectors.shape[0], -1, dtype=np.int64)])
        try:
            from evoc import EVoC
        except ModuleNotFoundError as error:
            if error.name != "evoc":
                raise
            raise ImportError("EVoCClusterer requires toponymy[evoc]") from error

        options = self.get_params(deep=False)
        # EVoC 0.3.1 has no verbose constructor argument.
        options.pop("verbose")
        options.pop("isolated")
        if self.isolated:
            from toponymy._evoc import fit_isolated

            estimator = fit_isolated(vectors, options)
        else:
            from toponymy._evoc import _fitted_state

            estimator = _fitted_state(EVoC(**options).fit(vectors))
        self._set_labels(estimator.cluster_layers_)
        self.evoc_ = estimator
        return self


class ToponymyClusterer(PLSCANClusterer):
    """Deprecated name for PLSCANClusterer; legacy kernel options are removed."""

    def fit(self, vectors=None, verbose=None, **kwargs):
        warnings.warn(
            "ToponymyClusterer now uses PLSCAN; use PLSCANClusterer explicitly",
            FutureWarning,
            stacklevel=2,
        )
        return super().fit(vectors, verbose=verbose, **kwargs)
