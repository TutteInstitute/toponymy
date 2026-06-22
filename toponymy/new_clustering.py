from sklearn.base import BaseEstimator

class Clusterer(ABC, BaseEstimator):
    """
    Abstract Clusterer Class that defines the properties required of a clusterer
    and implements formatter / validator methods that should be common to all
    subclasses.

    Attributes
    ----------
    cluster_layers_ : List[ClusterLayer]
        A list of the created cluster layers.
    cluster_tree_ : Dict[Tuple[int, int], List[Tuple[int, int]]]
        A dictionary representing the cluster tree. Keys are a tuple of (layer, cluster index) and values are lists of
        tuples representing child clusters.
    """
    def __init__(self):
        pass

    @abstractmethod
    def fit(
        self,
        clusterable_vectors: np.ndarray,
        embedding_vectors: np.ndarray,
        layer_class: Type[ClusterLayer],
        **layer_kwargs,
    ):
        pass

    @abstractmethod
    def fit_predict(
        self,
        clusterable_vectors: np.ndarray,
        embedding_vectors: np.ndarray,
        layer_class: Type[ClusterLayer],
        **layer_kwargs,
    ) -> Tuple[List[ClusterLayer], Dict[Tuple[int, int], List[Tuple[int, int]]]]:
        pass

    def __sklearn_is_fitted__(self):
        if (
            hasattr(self, "cluster_layers_")
            and isinstance(self.cluster_layers_, list)
            and all(isinstance(cluster, ClusterLayer) for cluster in self.cluster_layers)
            and hasattr(self, "cluster_tree_")
            and isinstance(self.cluster_tree_, ClusterTree)
        ):
            return True
        return False