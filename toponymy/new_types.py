from dataclasses import dataclass
from typing import Any, NewType, Dict, List, Tuple
from numpy.typing import NDArray


@dataclass
class Cluster:
    """
    Class to store data about a single cluster.

    Attributes
    ----------
    label : int
        The index of the cluster in the layer (i.e. label)

    members : np.ndarray
        A 1d numpy array of data ids that are in this cluster.

    features: dict[str, Any]
        Dictionary of features extracted from this cluster.

    prompt: str
        The prompt that will be passed to generate the name.

    name: str
        name: str
        The name that has been assigned to this cluster.
    """

    label: int
    members: NDArray[int]
    features: dict[str, Any] | None = None
    prompt: str | None = None
    name: str | None = None


@dataclass
class ClusterLayer:
    """
    Explicit type for a list of all the clusters from a single layer, and the index of that layer.
    """

    clusters: list[Cluster]
    layer_index: int

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, key):
        return self.clusters[key]

    def __iter__(self):
        return iter(self.clusters)


ClusterTree = NewType("ClusterTree", Dict[Tuple[int, int], List[Tuple[int, int]]])
