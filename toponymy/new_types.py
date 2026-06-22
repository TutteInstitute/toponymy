from toponymy.clustering import ClusterTree

class Cluster:
    members: list[int]
    features: list
    prompt: str
    name: str

class ClusterLayer:
    clusters: list[Cluster]
    layer_index: int

class Clusterer:
    layers: list[ClusterLayer]
    tree: ClusterTree
