from dataclasses import dataclass
from typing import Any


@dataclass
class Cluster:
    members: list[int]
    features: dict[str, Any]
    prompt: str
    name: str

@dataclass
class ClusterLayer:
    clusters: list[Cluster]
    layer_index: int
