from abc import ABC, abstractmethod
from typing import List, Callable, Any, Optional
import scipy.sparse
import numpy as np
import pandas as pd
from toponymy.keyphrases import central_keyphrases, information_weighted_keyphrases
from toponymy.exemplar_texts import diverse_exemplars
from toponymy.subtopics import central_subtopics
from toponymy.templates import SUMMARY_KINDS
from toponymy.prompt_construction import (
    topic_name_prompt,
    cluster_topic_names_for_renaming,
    distinguish_topic_names_prompt,
)
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


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

    def __init__(
        self,
        cluster_labels: np.ndarray,
        centroid_vectors: np.ndarray,
        layer_id: int,
        text_embedding_model: Optional[SentenceTransformer] = None,
        object_to_text_function: Optional[Callable[[Any], List[str]]] = None,
        n_exemplars: int = 16,
        n_keyphrases: int = 24,
        n_subtopics: int = 24,
        show_progress_bar: bool = False,
    ):
        self.cluster_labels = cluster_labels
        self.centroid_vectors = centroid_vectors
        self.layer_id = layer_id
        self.text_embedding_model = text_embedding_model
        self.object_to_text_function = object_to_text_function
        self.n_exemplars = n_exemplars
        self.n_keyphrases = n_keyphrases
        self.n_subtopics = n_subtopics
        self.show_progress_bar = show_progress_bar

        # Initialize empty lists for the cluster layer's attributes
        self.topic_names = []
        self.exemplars = []
        self.keyphrases = []
        self.subtopics = []

    @abstractmethod
    def name_topics(
        self,
        llm,
        detail_level: float,
        all_topic_names: List[List[str]],
        object_description: str,
        corpus_description: str,
        cluster_tree: Optional[dict] = None,
        embedding_model: Optional[SentenceTransformer] = None,
    ) -> List[str]:
        pass

    @abstractmethod
    def make_prompts(
        self,
        detail_level: float,
        all_topic_names: List[List[str]],
        object_description: str,
        corpus_description: str,
        cluster_tree: Optional[dict] = None,
    ) -> List[str]:
        pass

    @abstractmethod
    def make_keyphrases(
        self,
        keyphrase_list: List[str],
        object_x_keyphrase_matrix: scipy.sparse.spmatrix,
        keyphrase_vectors: np.ndarray,
    ) -> List[List[str]]:
        pass

    @abstractmethod
    def make_subtopics(
        self,
        topic_list: List[str],
        topic_labels: np.ndarray,
        topic_vectors: Optional[np.ndarray] = None,
        embedding_model: Optional[SentenceTransformer] = None,
    ) -> List[List[str]]:
        pass

    @abstractmethod
    def make_exemplar_texts(
        self,
        object_list: List[Any],
        object_vectors: np.ndarray,
    ) -> List[List[str]]:
        pass

    def _embed_topic_names(
        self,
        embedding_model: Optional[SentenceTransformer] = None,
    ) -> None:
        if embedding_model is None and self.text_embedding_model is None:
            raise ValueError("An embedding model must be provided")
        elif embedding_model is None:
            embedding_model = self.text_embedding_model

        self.topic_name_embeddings = embedding_model.encode(self.topic_names)

    def _make_disambiguation_prompts(
        self,
        detail_level: float,
        all_topic_names: List[List[str]],
        object_description: str,
        corpus_description: str,
        cluster_tree: Optional[dict] = None,
        max_topics_per_prompt: int = 12,
    ) -> None:
        summary_level = int(round(detail_level * (len(SUMMARY_KINDS) - 1)))
        summary_kind = SUMMARY_KINDS[summary_level]

        clusters_for_renaming, topic_name_cluster_labels = (
            cluster_topic_names_for_renaming(
                topic_names=self.topic_names,
                topic_name_embeddings=self.topic_name_embeddings,
            )
        )

        self.dismbiguation_topic_indices = [
            np.where(topic_name_cluster_labels == cluster_num)[0]
            for cluster_num in clusters_for_renaming
        ]

        # Break up over-large clusters into manageable chunks
        self.dismbiguation_topic_indices = [
            topic_indices[i : i + max_topics_per_prompt]
            for topic_indices in self.dismbiguation_topic_indices
            for i in range(0, len(topic_indices), max_topics_per_prompt)
        ]

        self.disambiguation_prompts = [
            distinguish_topic_names_prompt(
                topic_indices,
                self.layer_id,
                all_topic_names,
                exemplar_texts=self.exemplars,
                keyphrases=self.keyphrases,
                subtopics=self.subtopics if len(self.subtopics) > 0 else None,
                cluster_tree=cluster_tree,
                object_description=object_description,
                corpus_description=corpus_description,
                summary_kind=summary_kind,
                max_num_exemplars=self.n_exemplars,
                max_num_keyphrases=self.n_keyphrases,
                max_num_subtopics=self.n_subtopics,
            )
            for topic_indices in tqdm(
                self.dismbiguation_topic_indices,
                desc=f"Generating disambiguation prompts for layer {self.layer_id}",
                disable=(
                    not self.show_progress_bar
                    or len(self.dismbiguation_topic_indices) == 0
                ),
                total=len(self.dismbiguation_topic_indices),
                unit="topic-cluster",
                leave=False,
                position=1,
            )
        ]

    def _disambiguate_topic_names(self, llm) -> None:  # pragma: no cover
        for topic_indices, disambiguation_prompt in tqdm(
            zip(self.dismbiguation_topic_indices, self.disambiguation_prompts),
            desc=f"Generating new disambiguated topics names for layer {self.layer_id}",
            disable=not self.show_progress_bar
            or len(self.dismbiguation_topic_indices) == 0,
            total=len(self.dismbiguation_topic_indices),
            unit="topic-cluster",
            leave=False,
            position=1,
        ):
            new_names = llm.generate_topic_cluster_names(
                disambiguation_prompt, [self.topic_names[i] for i in topic_indices]
            )
            for i, topic_index in enumerate(topic_indices):
                self.topic_names[topic_index] = new_names[i]

    # pragma: no cover
    def disambiguate_topics(
        self,
        llm,
        detail_level: float,
        all_topic_names: List[List[str]],
        object_description: str,
        corpus_description: str,
        cluster_tree: Optional[dict] = None,
        embedding_model: Optional[SentenceTransformer] = None,
    ):
        self._embed_topic_names(embedding_model)
        self._make_disambiguation_prompts(
            detail_level=detail_level,
            all_topic_names=all_topic_names,
            object_description=object_description,
            corpus_description=corpus_description,
            cluster_tree=cluster_tree,
        )
        self._disambiguate_topic_names(llm)


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
        n_keyphrases: int = 16,
        keyphrase_diversify_alpha: float = 1.0,
        n_exemplars: int = 8,
        exemplars_diversify_alpha: float = 1.0,
        n_subtopics: int = 16,
        subtopic_diversify_alpha: float = 1.0,
        show_progress_bar: bool = False,
    ):
        super().__init__(
            cluster_labels,
            centroid_vectors,
            layer_id,
            text_embedding_model,
            show_progress_bar=show_progress_bar,
        )
        self.n_keyphrases = n_keyphrases
        self.keyphrase_diversify_alpha = keyphrase_diversify_alpha
        self.n_exemplars = n_exemplars
        self.exemplars_diversify_alpha = exemplars_diversify_alpha
        self.n_subtopics = n_subtopics
        self.subtopic_diversify_alpha = subtopic_diversify_alpha
        if text_embedding_model is not None:
            self.embedding_model = text_embedding_model

    def make_prompts(
        self,
        detail_level: float,
        all_topic_names: List[List[str]],
        object_description: str,
        corpus_description: str,
        cluster_tree: Optional[dict] = None,
    ) -> List[str]:
        summary_level = int(round(detail_level * (len(SUMMARY_KINDS) - 1)))
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
            for topic_index in tqdm(
                range(self.centroid_vectors.shape[0]),
                desc=f"Generating prompts for layer {self.layer_id}",
                disable=not self.show_progress_bar,
                unit="topic",
                leave=False,
                position=1,
            )
        ]

        return self.prompts

    # pragma: no cover
    def name_topics(
        self,
        llm,
        detail_level: float,
        all_topic_names: List[List[str]],
        object_description: str,
        corpus_description: str,
        cluster_tree: Optional[dict] = None,
        embedding_model: Optional[SentenceTransformer] = None,
    ) -> List[str]:
        self.topic_names = [
            (
                llm.generate_topic_name(prompt)
                if not prompt.startswith("[!SKIP!]: ")
                else prompt.removeprefix("[!SKIP!]: ")
            )
            for prompt in tqdm(
                self.prompts,
                desc=f"Generating topic names for layer {self.layer_id}",
                disable=not self.show_progress_bar,
                unit="topic",
                leave=False,
                position=1,
            )
        ]
        all_topic_names[self.layer_id] = self.topic_names
        self.disambiguate_topics(
            llm=llm,
            detail_level=detail_level,
            all_topic_names=all_topic_names,
            object_description=object_description,
            corpus_description=corpus_description,
            cluster_tree=cluster_tree,
            embedding_model=embedding_model,
        )
        # Run an extra disambiguation pass if we still have significant duplication
        if pd.Series(self.topic_names).value_counts().iloc[0] > 2:
            self.disambiguate_topics(
                llm=llm,
                detail_level=detail_level,
                all_topic_names=all_topic_names,
                object_description=object_description,
                corpus_description=corpus_description,
                cluster_tree=cluster_tree,
                embedding_model=embedding_model,
            )  # pragma: no cover

        # Try to fix any failures to generate a name
        if any(self.topic_names == ""):
            self.topic_names = [
                llm.generate_topic_name(prompt)
                if name == ""
                else name
                for name, prompt in zip(self.topic_names, self.prompts)
            ]

        return self.topic_names

    def make_keyphrases(
        self,
        keyphrase_list: List[str],
        object_x_keyphrase_matrix: scipy.sparse.spmatrix,
        keyphrase_vectors: np.ndarray,
    ) -> List[List[str]]:
        self.keyphrases = information_weighted_keyphrases(
            self.cluster_labels,
            object_x_keyphrase_matrix,
            keyphrase_list,
            keyphrase_vectors,
            self.centroid_vectors,
            diversify_alpha=self.keyphrase_diversify_alpha,
            n_keyphrases=self.n_keyphrases,
            show_progress_bar=self.show_progress_bar,
        )

        return self.keyphrases

    def make_subtopics(
        self,
        topic_list: List[str],
        topic_labels: np.ndarray,
        topic_vectors: Optional[np.ndarray] = None,
        embedding_model: Optional[SentenceTransformer] = None,
    ) -> List[List[str]]:
        self.subtopics = central_subtopics(
            cluster_label_vector=self.cluster_labels,
            subtopics=topic_list,
            subtopic_label_vector=topic_labels,
            centroid_vectors=self.centroid_vectors,
            subtopic_vectors=topic_vectors,
            diversify_alpha=self.subtopic_diversify_alpha,
            n_subtopics=self.n_subtopics,
            embedding_model=embedding_model,
            show_progress_bar=self.show_progress_bar,
        )

        return self.subtopics

    def make_exemplar_texts(
        self,
        object_list: List[str],
        object_vectors: np.ndarray,
    ) -> List[List[str]]:
        self.exemplars = diverse_exemplars(
            cluster_label_vector=self.cluster_labels,
            objects=object_list,
            object_vectors=object_vectors,
            centroid_vectors=self.centroid_vectors,
            n_exemplars=self.n_exemplars,
            diversify_alpha=self.exemplars_diversify_alpha,
            object_to_text_function=self.object_to_text_function,
            show_progress_bar=self.show_progress_bar,
        )

        return self.exemplars

    def make_topic_name_vector(self) -> np.ndarray:
        self.topic_name_vector = np.full(self.cluster_labels.shape[0], "Unlabelled", dtype=object)
        for i, name in enumerate(self.topic_names):
            self.topic_name_vector[self.cluster_labels == i] = name

        return self.topic_name_vector
