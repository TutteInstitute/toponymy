from abc import ABC, abstractmethod
from typing import List, Callable, Any
import scipy.sparse
import numpy as np
from toponymy.keyphrases import central_keyphrases
from toponymy.exemplar_texts import diverse_exemplars


class ClusterLayer(ABC):
    """
    Abstract class for a cluster layer. A cluster layer is a layer of a cluster hierarchy.

    Attributes:
    cluster_labels: vector of numeric cluster labels for the clusters in the layer
    centroid_vectors: list of centroid vectors of the clusters in the layer

    Methods:
    make_prompts: returns a list of prompts for the clusters in the layer
    make_keywords: generates a list of keywords for each clusters in the layer
    make_subtopics: generates a list of subtopics for each clusters in the layer
    make_sample_texts: generates a list of sample texts for each clusters in the layer
    """

    def __init__(self, cluster_labels, centroid_vectors):
        self.cluster_labels = cluster_labels
        self.centroid_vectors = centroid_vectors

    @abstractmethod
    def make_prompts(self, detail_level):
        pass

    @abstractmethod
    def make_keywords(
        self,
        keyphrase_list: List[str],
        object_x_keyphrase_matrix: scipy.sparse.spmatrix,
        keyphrase_vectors: np.ndarray,
    ) -> None:
        pass

    @abstractmethod
    def make_subtopics(
        self,
        topic_list: List[str],
        object_x_topic_matrix: scipy.sparse.spmatrix,
        topic_vectors: np.ndarray,
    ) -> None:
        pass

    @abstractmethod
    def make_exemplar_texts(
        self,
        object_list: List[Any],
        object_vectors: np.ndarray,
        object_to_text_function: Callable[[Any], List[str]],
    ) -> None:
        pass


class ClusterLayerText(ClusterLayer):
    """
    A cluster layer class for dealing with text data. A cluster layer is a layer of a cluster hierarchy.

    Attributes:
    cluster_labels: vector of numeric cluster labels for the clusters in the layer
    centroid_vectors: list of centroid vectors of the clusters in the layer

    Methods:
    make_prompts: creates and stores a list of prompts for the clusters in the layer
    make_keywords: generates and stores a list of keywords for each clusters in the layer
    make_subtopics: generates and stores a list of subtopics for each clusters in the layer
    make_sample_texts: generates and stores a list of sample texts for each clusters in the layer
    """

    def __init__(
        self,
        cluster_labels,
        centroid_vectors,
        n_keyphrases=32,
        keyphrase_diversify_alpha=1.0,
        n_exemplars=8,
        exemplars_diversify_alpha=1.0,
        n_subtopics=32,
        subtopic_diversify_alpha=1.0,
    ):
        super().__init__(cluster_labels, centroid_vectors)
        self.n_keyphrases = n_keyphrases
        self.keyphrase_diversify_alpha = keyphrase_diversify_alpha
        self.n_exemplars = n_exemplars
        self.exemplars_diversify_alpha = exemplars_diversify_alpha
        self.n_subtopics = n_subtopics
        self.subtopic_diversify_alpha = subtopic_diversify_alpha

    def make_prompts(self, detail_level):
        pass

    def make_keywords(
        self,
        keyphrase_list: List[str],
        object_x_keyphrase_matrix: scipy.sparse.spmatrix,
        keyphrase_vectors: np.ndarray,
    ) -> None:
        self.keyphrases = central_keyphrases(
            self.cluster_labels,
            object_x_keyphrase_matrix,
            keyphrase_list,
            keyphrase_vectors,
            self.centroid_vectors,
            diversify_alpha=self.keyphrase_diversify_alpha,
            n_keyphrases=self.n_keyphrases,
        )

    def make_subtopics(
        self,
        topic_list: List[str],
        object_x_topic_matrix: scipy.sparse.spmatrix,
        topic_vectors: np.ndarray,
    ) -> None:
        pass

    def make_exemplar_texts(
        self,
        object_list: List[str],
        object_vectors: np.ndarray,
        object_to_text_function: Callable[[Any], List[str]],
    ) -> None:
        self.exemplars = diverse_exemplars(
            cluster_label_vector=self.cluster_labels,
            objects=object_list,
            object_vectors=object_vectors,
            centroid_vectors=self.centroid_vectors,
            n_exemplars=self.n_exemplars,
            diversify_alpha=self.exemplars_diversify_alpha,
        )
