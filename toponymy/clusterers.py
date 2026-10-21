"""Cluster types and hierarchy helpers for the annotation architecture.

This module sits alongside :mod:`toponymy.clustering`, which is unchanged and
still used by :class:`toponymy.Toponymy`. Two names here also exist there
(``ClusterLayer`` and ``build_cluster_tree``). They are separate implementations
with different contracts, so import them from ``toponymy.clusterers``
explicitly.

The types hold clustering structure and nothing else: the partition of the
observations at each resolution, and how the clusters of different layers nest.
Names, exemplars and other per-cluster results are annotations and do not belong
on a cluster or a layer.

Cluster IDs are the labels the clustering algorithm produced. They are never
renumbered and need not be contiguous. Noise observations are labelled ``-1``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

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
