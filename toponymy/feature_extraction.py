from abc import ABC, abstractmethod


class AbstractFeatureExtractor(ABC):
    """
    An abstract base class representing a general feature extractor.

    #TODO: class documentation
    """

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def generate_features(self, objects, *args, **kwargs):
        """
        An abstract method to generate features from objects.

        Must be defined in any subclass.
        """
        raise NotImplemented

    @abstractmethod
    def build_object_x_feature_matrix(self, objects, features, *args, **kwargs):
        """
        An abstract method to build a matrix linking objects to features.

        Must be defined in any subclass.
        """
        raise NotImplemented

    @abstractmethod
    def select_features(
        self,
        cluster_label_vector,
        object_x_feature_matrix,
        feature_names,
        *args,
        **kwargs,
    ):
        """
        An abstract method to select features best representing clusters.

        Must be defined in any subclass.
        """
        raise NotImplemented
