class Cluster:
    members: list[int]
    features: list
    prompt: str
    name: str

class ClusterLayer:
    clusters: list[Cluster]
    layer_index: int

class ClusterTree:
    clusters: list[Cluster]
    layers: list[ClusterLayer]
    tree: dict[tuple[int, int], list[tuple[int, int]]]
