from abc import ABC, abstractmethod
from typing import List, Callable, Any
import scipy.sparse
import numpy as np

class ClusterLayer (ABC):
    """
    Abstract class for a cluster layer. A cluster layer is a layer of a cluster hierarchy.
    
    Attributes:
    label_vector: vector of numeric cluster labels for the clusters in the layer
    centroid_vectors: list of centroid vectors of the clusters in the layer
    
    Methods:
    make_prompts: returns a list of prompts for the clusters in the layer
    make_keyphrases: generates a list of _keyphrases for each clusters in the layer
    make_subtopics: generates a list of subtopics for each clusters in the layer
    make_sample_texts: generates a list of sample texts for each clusters in the layer
    """

    def __init__(self, label_vector, centroid_vectors):
        self.label_vector = label_vector
        self.centroid_vectors = centroid_vectors

    @abstractmethod
    def make_prompts(self):
        pass

    @abstractmethod
    def make_keyphrases(self, keyphrase_list: List[str], object_x_keyphrase_matrix: scipy.sparse.spmatrix, keyphrase_vectors: np.ndarray) -> None:
        pass

    @abstractmethod
    def make_subtopics(self, topic_list: List[str], object_x_topic_matrix: scipy.sparse.spmatrix, topic_vectors: np.ndarray) -> None:
        pass

    @abstractmethod
    def make_sample_texts(self, object_list: List[str], object_vectors: np.ndarray, object_to_text_function: Callable[[Any], List[str]]) -> None:
        pass