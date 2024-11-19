from abc import ABC, abstractmethod
from typing import List, Callable, Any
import scipy.sparse
import numpy as np

class ClusterLayer (ABC):
    """
    Abstract class for a cluster layer. A cluster layer is a layer of a cluster hierarchy.
    
    Attributes:
    centroid_vectors: list of centroid vectors of the clusters in the layer
    pointsets: list of pointsets of the clusters in the layer
    
    Methods:
    make_prompts: returns a list of prompts for the clusters in the layer
    make_keywords: generates a list of keywords for each clusters in the layer
    make_subtopics: generates a list of subtopics for each clusters in the layer
    make_sample_texts: generates a list of sample texts for each clusters in the layer
    """

    def __init__(self, centroid_vectors, label_vector):
        self.centroid_vectors = centroid_vectors
        self.label_vector = label_vector

    @abstractmethod
    def make_prompts(self):
        pass

    @abstractmethod
    def make_keywords(self, keyword_list: List[str], object_x_keyword_matrix: scipy.sparse.spmatrix, keyword_vectors: np.ndarray) -> None:
        pass

    @abstractmethod
    def make_subtopics(self, topic_list: List[str], object_x_topic_matrix: scipy.sparse.spmatrix, topic_vectors: np.ndarray) -> None:
        pass

    @abstractmethod
    def make_sample_texts(self, object_list: List[str], object_vectors: np.ndarray, object_to_text_function: Callable[[Any], str]) -> None:
        pass