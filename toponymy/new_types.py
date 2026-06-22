class Cluster:
    """
    Class to store data about a single cluster.

    Attributes
    ----------
    members : list[int]
        A list of data ids that are in this cluster.
    
    features: dict[str, Any]
        Dictionary of features extracted from this cluster.

    prompt: str
        The prompt that will be passed to generate the name.

    name: str
        The name that has been assigned to this cluster.
    """
    members: list[int]
    features: dict[str, Any] | None
    prompt: str | None
    name: str | None


class ClusterLayer:
    """
    Explicit type for a list of all the clusters from a single layer, and the index of that layer.
    """
    clusters: list[Cluster]
    layer_index: int
