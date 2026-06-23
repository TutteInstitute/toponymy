from abc import ABC, abstractmethod
from typing import Any, List

import numba
import numpy as np
from sklearn.base import BaseEstimator

from toponymy.new_clustering import Clusterer
from toponymy.utility_functions import centroids_from_labels


class FeatureExtractorBase(ABC, BaseEstimator):
    """
    Abstract base class for a feature extractor.

    A feature extractor is a class that can build features for objects
    and then extract features to represent clusters.
    """

    def __init__(self, *args, **kwargs):
        self.is_fitted_ = False

    def __sklearn_is_fitted__(self):
        return hasattr(self, "is_fitted_") and self.is_fitted_

    def can_fit_from_objects(self) -> bool:
        """
        If True, enables the FeatureExtractor to be fitted on the fly.

        If False, specifies that the FeatureExtractor must be pre-fitted.
        """
        return False

    @abstractmethod
    def fit(self, objects: List[Any], *args, **kwargs):
        """
        An abstract method to fit a collection of features to a set of objects.

        Must be defined in any subclass.
        """
        raise NotImplemented

    @abstractmethod
    def get_cluster_features(
        self,
        clusterer: Clusterer,
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
        clusterer: Clusterer,
        layer_id: int,
        *args,
        **kwargs,
    ) -> List[List[str]]:
        """
        A method to get features as a representation for each cluster.

        Checks that the feature extractor is fitted, and then runs `get_cluster_features`.
        """
        return self.get_cluster_features(cluster_indices, layer_id, *args, **kwargs)
