from abc import ABC, abstractmethod
from typing import List, Callable, Any, Optional
import scipy.sparse
import numpy as np
from toponymy.keyphrases import central_keyphrases
from toponymy.exemplar_texts import diverse_exemplars
from toponymy.templates import SUMMARY_KINDS
from toponymy.prompt_construction import topic_name_prompt, cluster_topic_names_for_renaming, distinguish_topic_names_prompt
from sentence_transformers import SentenceTransformer

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

    def __init__(self, cluster_labels: np.ndarray, centroid_vectors: np.ndarray, layer_id: int, text_embedding_model: Optional[SentenceTransformer] = None):
        self.cluster_labels = cluster_labels
        self.centroid_vectors = centroid_vectors
        self.layer_id = layer_id
        self.text_embedding_model = text_embedding_model

    @abstractmethod
    def make_prompts(           
            self, 
            detail_level: float, 
            all_topic_names: List[List[str]], 
            object_description: str,
            corpus_description: str,
            cluster_tree: Optional[dict] = None,
    ) -> None:
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
        cluster_labels: np.ndarray,
        centroid_vectors: np.ndarray,
        layer_id: int,
        text_embedding_model: Optional[SentenceTransformer] = None,
        n_keyphrases: int = 32,
        keyphrase_diversify_alpha: float = 1.0,
        n_exemplars: int = 8,
        exemplars_diversify_alpha: float = 1.0,
        n_subtopics: int = 32,
        subtopic_diversify_alpha: float = 1.0,
    ):
        super().__init__(cluster_labels, centroid_vectors, layer_id, text_embedding_model)
        self.n_keyphrases = n_keyphrases
        self.keyphrase_diversify_alpha = keyphrase_diversify_alpha
        self.n_exemplars = n_exemplars
        self.exemplars_diversify_alpha = exemplars_diversify_alpha
        self.n_subtopics = n_subtopics
        self.subtopic_diversify_alpha = subtopic_diversify_alpha
        self.subtopics = None # Empty subtopics; to be populated if reuqired for layer

    def make_prompts(
            self, 
            detail_level: float, 
            all_topic_names: List[List[str]], 
            object_description: str,
            corpus_description: str,
            cluster_tree: Optional[dict] = None,
        ) -> None:
        summary_level = int(round(detail_level * len(SUMMARY_KINDS)))
        summary_kind = SUMMARY_KINDS[summary_level]

        self.prompts = [
            topic_name_prompt(
                topic_index,
                self.layer_id,
                all_topic_names,
                exemplar_texts=self.exemplars,
                keyphrases=self.keyphrases,
                subtopics=self.subtopics,
                cluster_tree=cluster_tree,
                object_description=object_description,
                corpus_description=corpus_description,
                summary_kind=summary_kind,
                max_num_exemplars=self.n_exemplars,
                max_num_keyphrases=self.n_keyphrases,
                max_num_subtopics=self.n_subtopics,
            )
            for topic_index in range(self.centroid_vectors.shape[0])
        ]

    def name_topics(
            self,
            llm,
    ):
        self.topic_names = [
            llm.get_topic_name(prompt)
            for prompt in self.prompts
        ]
        self._embed_topic_names()
        self._make_disambiguation_prompts()
        self._disambiguate_topic_names()

    def _make_disambiguation_prompts(
            self,
            topic_name_embeddings: Optional[np.ndarray] = None,
            embedding_model: Optional[SentenceTransformer] = None,
    ) -> None:
        clusters_for_renaming, topic_name_cluster_labels = cluster_topic_names_for_renaming(
            topic_names=self.topic_names,
            topic_name_embeddings=self.topic_name_embeddings,
        )

    def _embed_topic_names(
            self,
            embedding_model: Optional[SentenceTransformer] = None,
    ) -> None:
        if embedding_model is None and self.embedding_model is None:
            raise ValueError("An embedding model must be provided")
        elif embedding_model is None:
            embedding_model = self.embedding_model

        self.topic_name_embeddings = embedding_model.encode(self.topic_names)

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
