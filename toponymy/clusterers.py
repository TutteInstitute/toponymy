"""Clustering primitives and clusterers for the annotation architecture.

This module sits alongside :mod:`toponymy.clustering`, which is unchanged and
still used by :class:`toponymy.Toponymy`. Several names here also exist there
(``Clusterer``, ``ClusterLayer``, ``EVoCClusterer``, ``build_cluster_tree``).
They are separate implementations with different contracts, so import them from
``toponymy.clusterers`` explicitly.

A clusterer produces clustering structure and holds no topic state:
``cluster_layers_``, the partition of the observations at each resolution, and
``cluster_tree_``, how the clusters of different layers nest. Names, exemplars
and other per-cluster results are annotations and do not belong on a clusterer.

Cluster IDs are the labels the clustering algorithm produced. They are never
renumbered and need not be contiguous. Noise observations are labelled ``-1``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

if TYPE_CHECKING:
    from typing_extensions import Self

ClusterKey = tuple[int, int]
ClusterTree = dict[ClusterKey, list[ClusterKey]]


def _integer_vector(values: ArrayLike, name: str, minimum: int) -> NDArray[np.int64]:
    """Return an owned int64 copy of 1-D integers, none below ``minimum``."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    # An untyped empty Python sequence has float dtype, but contains no bad IDs.
    if array.size and array.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integers")
    if array.size and (
        np.any(array < minimum) or np.any(array > np.iinfo(np.int64).max)
    ):
        raise ValueError(f"{name} must be between {minimum} and int64's maximum")
    return np.array(array, dtype=np.int64, copy=True)


def _nonnegative_integer(value: int, name: str) -> int:
    """Return ``value`` as an int if it is a nonnegative int64 integer, not a bool."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not 0 <= value <= np.iinfo(np.int64).max
    ):
        raise ValueError(f"{name} must be a nonnegative int64 integer")
    return int(value)


@dataclass(frozen=True, eq=False)
class Cluster:
    """A cluster's original ID and the observations that belong to it.

    Parameters
    ----------
    label : int
        The cluster's ID, exactly as the clustering algorithm produced it.
    members : array-like of int
        Indices of the observations in the cluster; at least one, without
        repeats.

    Attributes
    ----------
    members : ndarray of int64
        The indices in increasing order. A read-only copy owned by the cluster.
    """

    label: int
    members: NDArray[np.int64]

    def __post_init__(self) -> None:
        label = _nonnegative_integer(self.label, "label")
        members = _integer_vector(self.members, "members", 0)
        if not members.size:
            raise ValueError("a cluster must contain at least one member")
        members.sort()
        if np.any(members[1:] == members[:-1]):
            raise ValueError("cluster members must be unique")
        members.flags.writeable = False
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "members", members)

    def __reduce__(self):
        # Rebuild through the constructor. Otherwise copy.deepcopy and unpickling
        # would restore writeable arrays and skip the validation above.
        return type(self), (self.label, self.members)


@dataclass(frozen=True, eq=False)
class ClusterLayer:
    """One resolution of a clustering: a partition of the observations.

    Parameters
    ----------
    clusters : sequence of Cluster
        The clusters of the layer. Together they must hold exactly the
        observations that ``labels`` assigns to a cluster, under the same IDs.
    layer_index : int
        Position of the layer in its clusterer's ``cluster_layers_``.
    labels : array-like of int
        The cluster ID of every observation, or ``-1`` for noise.

    Attributes
    ----------
    clusters : tuple of Cluster
        The clusters ordered by ID.
    labels : ndarray of int64
        A read-only copy owned by the layer.

    Raises
    ------
    TypeError
        If ``clusters`` holds anything other than ``Cluster`` instances.
    ValueError
        If ``layer_index`` or ``labels`` are not valid, a cluster ID repeats,
        clusters overlap, a member index is beyond the labels, or the clusters
        and the labels do not describe the same partition.

    Notes
    -----
    A layer is data only. Names and other per-cluster results are not stored on
    it; create a new layer rather than changing one.
    """

    clusters: tuple[Cluster, ...]
    layer_index: int
    labels: NDArray[np.int64]

    def __post_init__(self) -> None:
        layer_index = _nonnegative_integer(self.layer_index, "layer_index")
        labels = _integer_vector(self.labels, "labels", -1)
        clusters = tuple(self.clusters)
        if any(not isinstance(cluster, Cluster) for cluster in clusters):
            raise TypeError("clusters must contain Cluster instances")
        clusters = tuple(sorted(clusters, key=lambda cluster: cluster.label))
        expected = np.full(labels.size, -1, dtype=np.int64)
        seen = set()
        for cluster in clusters:
            if cluster.label in seen:
                raise ValueError("cluster IDs must be unique within a layer")
            seen.add(cluster.label)
            if cluster.members[-1] >= labels.size:
                raise ValueError("cluster member index is outside the labels array")
            if np.any(expected[cluster.members] != -1):
                raise ValueError("clusters within a layer must be disjoint")
            expected[cluster.members] = cluster.label
        if not np.array_equal(expected, labels):
            raise ValueError("clusters and labels must describe the same partition")
        labels.flags.writeable = False
        object.__setattr__(self, "clusters", clusters)
        object.__setattr__(self, "layer_index", layer_index)
        object.__setattr__(self, "labels", labels)

    def __reduce__(self):
        # As for Cluster: rebuild through the constructor.
        return type(self), (self.clusters, self.layer_index, self.labels)

    @property
    def cluster_labels(self) -> NDArray[np.int64]:
        """Alias of ``labels``.

        This is the name that ``AnnotationStore.from_clusterer`` and the legacy
        cluster layer classes read.
        """
        return self.labels

    def __len__(self) -> int:
        return len(self.clusters)

    def __getitem__(self, key: int | slice) -> Cluster | tuple[Cluster, ...]:
        return self.clusters[key]

    def __iter__(self) -> Iterator[Cluster]:
        return iter(self.clusters)


def _validate_label_layers(labels: Sequence[ArrayLike]) -> list[NDArray[np.int64]]:
    """Return owned int64 copies of equally long layers of labels."""
    if isinstance(labels, np.ndarray) and labels.ndim != 2:
        raise ValueError("labels must be a sequence of one-dimensional label layers")
    result = [_integer_vector(layer, "labels", -1) for layer in labels]
    if result and any(layer.size != result[0].size for layer in result[1:]):
        raise ValueError("all label layers must have the same observation count")
    return result


def _group_labels(labels: NDArray[np.int64]):
    """Group the clustered observations by ID, so no array scales with an ID.

    Returns the sorted IDs, the offset in ``order`` where each ID's members
    start, and the member indices of all clustered observations sorted by ID.
    """
    indices = np.flatnonzero(labels >= 0)
    order = indices[np.argsort(labels[indices], kind="stable")]
    ids, starts = np.unique(labels[order], return_index=True)
    return ids, starts, order


def build_cluster_layers(labels: Sequence[ArrayLike]) -> list[ClusterLayer]:
    """Group the observations of each label layer into clusters.

    Parameters
    ----------
    labels : sequence of array-like of int
        One one-dimensional integer array per layer, all of the same length.
        ``-1`` marks noise. Any other label is a nonnegative int64 ID; IDs need
        not be contiguous, are never renumbered, and do not affect memory use.

    Returns
    -------
    list of ClusterLayer
        One layer per input layer. An empty layer, or one that is all noise, has
        no clusters. Layers own read-only copies of the labels.

    Raises
    ------
    ValueError
        If a layer is not a one-dimensional integer array, holds a label below
        -1 or above the int64 maximum, or the layers differ in length.
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


def build_cluster_tree(labels: Sequence[ArrayLike]) -> ClusterTree:
    """Attach each cluster to its nearest fully containing upper cluster.

    Layers are ordered from finest to broadest. A cluster whose observations
    straddle several clusters of an upper layer, or include noise there, skips
    that layer and may attach to a cluster in a higher one. Clusters with no
    containing ancestor attach to the synthetic root ``(len(labels), 0)``. No
    nodes are invented for IDs that do not occur.

    Parameters
    ----------
    labels : sequence of array-like of int
        One label array per layer, as for :func:`build_cluster_layers`.

    Returns
    -------
    dict
        Maps each parent ``(layer, cluster ID)`` to a list of child
        ``(layer, cluster ID)`` pairs. Empty if no layer has a cluster.

    Raises
    ------
    ValueError
        Under the same conditions as :func:`build_cluster_layers`.
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


def validate_cluster_tree(
    tree: Mapping[ClusterKey, Sequence[ClusterKey]], layers: Sequence[ClusterLayer]
) -> None:
    """Check that a tree is a valid hierarchy over the clusters of ``layers``.

    A tree is valid if every cluster occurs exactly once as a child, every
    parent is a cluster in a strictly higher layer or the synthetic root
    ``(len(layers), 0)``, and every edge is true containment. Edges may skip
    layers. Strictly increasing parent layers also rule out cycles.

    Parameters
    ----------
    tree : mapping
        Maps each parent ``(layer, cluster ID)`` to a list or tuple of child
        ``(layer, cluster ID)`` pairs.
    layers : sequence of ClusterLayer
        The layers, in consecutive index order from 0, of equal length.

    Raises
    ------
    TypeError
        If ``tree`` is not a mapping.
    ValueError
        If ``layers`` is malformed, or the tree has an unknown, repeated or
        missing node, a malformed key, a parent that is not above its child, or
        a parent that does not contain its child.
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


def _validate_vectors(vectors: ArrayLike) -> NDArray:
    """Check the shape and values of dense vectors without copying them."""
    if sparse.issparse(vectors):
        raise ValueError("vectors must be a dense array, not a sparse matrix")
    array = np.asarray(vectors)
    if array.ndim != 2 or (array.shape[1] == 0 and array.shape[0] != 0):
        raise ValueError("vectors must have shape (observations, nonzero dimensions)")
    kind = array.dtype.kind
    # Integers are always finite; checking them would allocate a data-sized mask.
    if kind not in "iuf" or (kind == "f" and not np.isfinite(array).all()):
        raise ValueError("vectors must contain finite real numbers")
    return array


class Clusterer(ABC, BaseEstimator):
    """Base class for clusterers whose fitted state is cluster layers and a tree.

    Clusterers subclass scikit-learn's ``BaseEstimator``, so ``get_params``,
    ``set_params`` and ``clone`` work. They do not otherwise follow the
    scikit-learn clusterer API: ``fit`` takes only the vectors (no ``y``), and
    ``fit_predict`` returns the layers and the tree, not labels. Iterating over a
    fitted clusterer yields its layers.

    Subclasses implement ``fit`` and record the result with ``_set_labels``.

    Attributes
    ----------
    cluster_layers_ : list of ClusterLayer
        The partition of the observations at each resolution; layer 0 is the
        finest.
    cluster_tree_ : dict
        Maps each parent ``(layer, cluster ID)`` to a list of child
        ``(layer, cluster ID)`` pairs, as built by :func:`build_cluster_tree`.
    """

    @abstractmethod
    def fit(self, vectors: ArrayLike) -> Self:
        """Fit the clusterer to ``vectors`` and return it."""

    def fit_predict(self, vectors: ArrayLike) -> tuple[list[ClusterLayer], ClusterTree]:
        """Fit the clusterer and return its layers and tree.

        Unlike scikit-learn's ``fit_predict``, which returns labels, this
        returns the fitted ``cluster_layers_`` and ``cluster_tree_``.

        Parameters
        ----------
        vectors : array-like of shape (n_observations, n_features)
            The vectors to cluster, one row per observation.

        Returns
        -------
        cluster_layers : list of ClusterLayer
            The fitted ``cluster_layers_``.
        cluster_tree : dict
            The fitted ``cluster_tree_``.
        """
        self.fit(vectors)
        return self.cluster_layers_, self.cluster_tree_

    def __iter__(self) -> Iterator[ClusterLayer]:
        if not self.__sklearn_is_fitted__():
            raise NotFittedError("fit the clusterer before iterating its layers")
        return iter(self.cluster_layers_)

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "cluster_layers_") and hasattr(self, "cluster_tree_")

    def _set_labels(self, labels: Sequence[ArrayLike]) -> None:
        """Set the fitted layers and tree from one label array per layer.

        Both are built before either is assigned, so labels that are rejected
        leave the result of any earlier fit unchanged.
        """
        layers = build_cluster_layers(labels)
        tree = build_cluster_tree([layer.labels for layer in layers])
        self.cluster_layers_ = layers
        self.cluster_tree_ = tree


class EVoCClusterer(Clusterer):
    """Cluster embedding vectors with EVoC.

    EVoC (Embedding Vector Oriented Clustering) clusters high-dimensional
    embedding vectors, such as sentence-transformer output, by combining a node
    embedding of a nearest-neighbor graph with density-based clustering. It
    returns a hierarchy of layers, finest first.

    The parameters are EVoC's own, with EVoC's defaults, and are passed to
    :class:`evoc.EVoC` unchanged; see its documentation for the details. The
    layers keep EVoC's labels as cluster IDs. ``cluster_tree_`` is built from
    them by :func:`build_cluster_tree`, not taken from EVoC.

    Requires the optional ``evoc`` package. Use evoc 0.3.2 or later. Earlier
    versions define numba types with the same names as fast_hdbscan, which
    Toponymy depends on, so in a process that uses both, the second one to run
    fails with a numba ``TypeError``.

    Parameters
    ----------
    noise_level : float, default=0.5
        How much data EVoC discards as noise, from 0.0 (cluster more data) to
        1.0 (keep only the most reliable clusters).
    base_min_cluster_size : int, default=5
        Smallest cluster size in the finest layer. At least 2.
    base_n_clusters : int or None, default=None
        If set, approximate number of clusters in the finest layer.
    approx_n_clusters : int or None, default=None
        If set, approximate number of clusters wanted; EVoC then returns a
        single layer.
    n_neighbors : int, default=15
        Number of neighbors in the nearest-neighbor graph.
    min_samples : int, default=5
        Minimum number of samples used in the density estimate when clustering
        the node embedding. At least 1.
    n_epochs : int, default=50
        Number of training epochs of the node embedding.
    node_embedding_init : {"label_prop", None}, default="label_prop"
        How the node embedding is initialized; None does not initialize it.
    symmetrize_graph : bool, default=True
        Whether to symmetrize the neighbor graph.
    node_embedding_dim : int or None, default=None
        Dimension of the node embedding; EVoC chooses one when None.
    neighbor_scale : float, default=1.0
        Scales the effective number of neighbors used to build the graph.
    random_state : int or None, default=None
        Seed of EVoC's random number generator. Setting it also makes EVoC use
        its deterministic algorithms where possible. The result can still differ
        with the number of numba threads.
    min_similarity_threshold : float, default=0.2
        Layers whose clusterings are more similar than this are filtered out.
    max_layers : int, default=10
        Maximum number of layers. At least 1.
    n_label_prop_iter : int, default=20
        Number of label propagation iterations when ``node_embedding_init`` is
        "label_prop".

    Attributes
    ----------
    cluster_layers_ : list of ClusterLayer
        The layers EVoC returned, finest first.
    cluster_tree_ : dict
        The hierarchy of the clusters, as built by :func:`build_cluster_tree`.
    evoc_ : evoc.EVoC or None
        The fitted EVoC estimator, which also holds the membership strengths,
        persistence scores and nearest neighbors. None if EVoC did not run
        because there were too few observations.

    Notes
    -----
    Fewer than ``max(min_samples, base_min_cluster_size)`` observations are too
    few to cluster. They form one layer of noise, or no layer at all if there
    are no observations, and EVoC is not run, so ``evoc`` need not be installed.

    EVoC reads float vectors as using cosine distance, int8 vectors as quantized
    embeddings and uint8 vectors as binary embeddings; it does not support other
    integer types. An array is passed to EVoC as given, not copied, and EVoC
    0.3.2 does not modify it.

    numba falls back to its ``workqueue`` threading layer when neither its OpenMP
    nor its TBB layer can load. On Linux, OpenMP needs the system library
    ``libgomp``, which minimal container images often lack. EVoC's nested
    parallel loops then abort the whole Python process.
    """

    def __init__(
        self,
        noise_level: float = 0.5,
        base_min_cluster_size: int = 5,
        base_n_clusters: int | None = None,
        approx_n_clusters: int | None = None,
        n_neighbors: int = 15,
        min_samples: int = 5,
        n_epochs: int = 50,
        node_embedding_init: str | None = "label_prop",
        symmetrize_graph: bool = True,
        node_embedding_dim: int | None = None,
        neighbor_scale: float = 1.0,
        random_state: int | None = None,
        min_similarity_threshold: float = 0.2,
        max_layers: int = 10,
        n_label_prop_iter: int = 20,
    ):
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

    def fit(self, vectors: ArrayLike) -> Self:
        """Cluster ``vectors`` with EVoC.

        Parameters
        ----------
        vectors : array-like of shape (n_observations, n_features)
            Dense finite vectors, one row per observation.

        Returns
        -------
        self
            The fitted clusterer.

        Raises
        ------
        ValueError
            If ``vectors`` is not a two-dimensional dense array of finite float,
            int8 or uint8 values, or ``min_samples``, ``base_min_cluster_size``
            or ``max_layers`` is not an integer of at least its minimum.
        ImportError
            If EVoC has to run and ``evoc`` is not installed.
        """
        vectors = _validate_vectors(vectors)
        if vectors.dtype.kind in "iu" and vectors.dtype.itemsize != 1:
            raise ValueError(
                f"EVoC supports float, int8 and uint8 vectors, not {vectors.dtype}; "
                "convert them first, for example with vectors.astype(np.float32)"
            )
        self._validate_parameters()
        n_observations = vectors.shape[0]
        if n_observations == 0:
            labels, evoc = [], None
        elif n_observations < max(self.min_samples, self.base_min_cluster_size):
            labels, evoc = [np.full(n_observations, -1)], None
        else:
            evoc = self._fit_evoc(vectors)
            labels = evoc.cluster_layers_
        self._set_labels(labels)
        self.evoc_ = evoc
        return self

    def _validate_parameters(self) -> None:
        # Checked for every input, so that a tiny input does not hide a bad value.
        for name, minimum in (
            ("min_samples", 1),
            ("base_min_cluster_size", 2),
            ("max_layers", 1),
        ):
            if _nonnegative_integer(getattr(self, name), name) < minimum:
                raise ValueError(f"{name} must be at least {minimum}")

    def _fit_evoc(self, vectors: NDArray):
        try:
            from evoc import EVoC
        except ModuleNotFoundError as error:
            # A missing dependency of EVoC is a different problem; let it through.
            if error.name != "evoc":
                raise
            raise ImportError(
                "EVoCClusterer requires the optional 'evoc' package; "
                'install it with: pip install "evoc>=0.3.2"'
            ) from error
        return EVoC(**self.get_params(deep=False)).fit(vectors)
