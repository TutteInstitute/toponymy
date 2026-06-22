from toponymy.clustering import ClusterTree


class Cluster:
    label: int
    members: list[int]
    features: list
    prompt: str
    name: str


class ClusterLayer:
    clusters: list[Cluster]
    layer_index: int
    labels: list[int]


class Clusterer:
    layers: list[ClusterLayer]
    tree: ClusterTree
