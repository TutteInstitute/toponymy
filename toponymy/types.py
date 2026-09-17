"""Owned clustering output, independent of feature extraction and naming."""

from dataclasses import dataclass
from typing import Dict, Iterator, List, Protocol, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

ClusterKey = Tuple[int, int]
ClusterTree = Dict[ClusterKey, List[ClusterKey]]


def _integer_vector(values: ArrayLike, name: str, minimum: int) -> NDArray[np.int64]:
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
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not 0 <= value <= np.iinfo(np.int64).max
    ):
        raise ValueError(f"{name} must be a nonnegative int64 integer")
    return int(value)


@dataclass(frozen=True, eq=False)
class Cluster:
    """A cluster's original ID and owned, sorted, read-only observation indices."""

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


@dataclass(frozen=True, eq=False)
class ClusterLayer:
    """A partition, including noise (-1), with clusters ordered by original ID.

    Arrays are copied once at the public ownership boundary. Supported updates
    construct a new layer; naming state does not belong on a cluster or layer.
    """

    clusters: Tuple[Cluster, ...]
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

    @property
    def cluster_labels(self) -> NDArray[np.int64]:
        """Compatibility spelling for the read-only observation labels."""
        return self.labels

    def __len__(self) -> int:
        return len(self.clusters)

    def __getitem__(self, key):
        return self.clusters[key]

    def __iter__(self) -> Iterator[Cluster]:
        return iter(self.clusters)


class TextEmbedderProtocol(Protocol):
    """The text encoding boundary used by selected feature extractors."""

    def encode(self, texts: Sequence[str], *args, **kwargs) -> NDArray[np.floating]: ...
