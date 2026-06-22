from abc import ABC, abstractmethod

class FeatureExtractorBase(ABC):
    """
    Abstract base class for a feature extractor.
    """

    def __init__(self):
        self.is_fitted_ = False

    @property
    def is_fitted_(self):
        """
        Boolean property: True if the FeatureExtractor has been fitted.
        """
        return hasattr(self, "is_fitted_") and self.is_fitted_

    def can_fit_from_objects(self) -> bool:
        """
        If True, enables the FeatureExtractor to be fitted on the fly.

        If False, specifies that the FeatureExtractor must be pre-fitted.
        """
        return False

    @abstractmethod
    def fit(self,
            objects: List[Any],
            *args,
            **kwargs):
        """
        An abstract method to fit a collection of features to a set of objects.
        
        Must be defined in any subclass.
        """
        raise NotImplemented

    @abstractmethod
    def get_cluster_features(
        self,
        cluster_indices: List[int],
        layer_id: int,
        *args,
        **kwargs,
    ) -> List[List[str]]:
        """
        An abstract method to get features as a representation for each cluster.

        Must be defined in any subclass.
        """
        raise NotImplemented

    def predict(
        self,
        cluster_indices: List[int],
        layer_id: int,
        *args,
        **kwargs,
    ) -> List[List[str]]:
        """
        A method to get features as a representation for each cluster.

        Syntactic sugar for `get_cluster_features`.
        """
        return self.get_cluster_features(cluster_indices, layer_id, *args, **kwargs)
