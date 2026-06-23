"""
Refactored Toponymy class for hierarchical topic modeling with LLMs.
"""

from toponymy.clustering import ToponymyClusterer, Clusterer
from toponymy.keyphrases import KeyphraseBuilder
from toponymy.exemplar_texts import ExemplarTextExtractor
from toponymy.cluster_layer import ClusterLayer, ClusterLayerSummaryText
from toponymy.topic_tree import TopicTree
from toponymy.llm_wrappers import LLMWrapper
from toponymy.embedding_wrappers import TextEmbedderProtocol
from toponymy.templates import PROMPT_TEMPLATES
from toponymy._utils import handle_verbose_params

from sklearn.utils.validation import check_is_fitted
import numpy as np
from tqdm.auto import tqdm
from typing import List, Any, Optional, Dict


class Toponymy:
    """
    Hierarchical topic modeling with LLM-generated topic names.

    Parameters
    ----------
    llm_namer : LLMNamer
        LLMNamer for generating topic names.
    text_embedding_model : TextEmbedderProtocol
        Model for generating text embeddings.
    clusterer : Clusterer, optional
        Clusterer for hierarchical clustering. Defaults to ToponymyClusterer().
    prompt_template : Dict[str, Any], optional
        Template for LLM prompts. Defaults to PROMPT_TEMPLATES.
    feature_extractors : List, optional
        List of feature extractors. Defaults to [ExemplarTextExtractor(), KeyphraseBuilder()].
    object_description : str, optional
        Description of objects (e.g., "documents", "images").
    corpus_description : str, optional
        Description of the corpus.
    lowest_detail_level : float, optional
        Lowest detail level for topic names (0 = finest, 1 = coarsest).
    highest_detail_level : float, optional
        Highest detail level for topic names.
    verbose : bool, optional
        Whether to show progress bars and verbose output.
    """

    def __init__(
        self,
        llm_namer: LLMWrapper,
        text_embedding_model: TextEmbedderProtocol,
        clusterer: Optional[Clusterer] = ToponymyClusterer(),
        prompt_template: Dict[str, Any] = PROMPT_TEMPLATES,
        feature_extractors: Optional[List] = [
            ExemplarTextExtractor(),
            KeyphraseBuilder(),
        ],
        object_description: str = "objects",
        corpus_description: str = "collection of objects",
        lowest_detail_level: float = 0.0,
        highest_detail_level: float = 1.0,
        verbose: Optional[bool] = None,
    ):
        self.llm_namer = llm_namer
        self.embedding_model = text_embedding_model

        self.clusterer = clusterer
        self.feature_extractors = feature_extractors

        self.object_description = object_description
        self.corpus_description = corpus_description
        self.lowest_detail_level = lowest_detail_level
        self.highest_detail_level = highest_detail_level
        self.prompt_template = prompt_template
        self.verbose = verbose

    def _effective_prompt_format(self) -> str:
        """Determine prompt format based on LLM wrapper capabilities."""
        return "system_user" if self.llm_namer.supports_system_prompts else "combined"

    def _sync_layer_runtime_config(self) -> None:
        """Update runtime config on all cluster layers."""
        for layer in self.cluster_layers_:
            layer.prompt_format = self._effective_prompt_format()
            layer.verbose = self.verbose

    def fit(
        self,
        objects: List[Any],
        embedding_vectors: np.ndarray,
        clusterable_vectors: np.ndarray,
    ):
        """
        Fit the model to generate topic names for hierarchical clusters.

        Parameters
        ----------
        objects : List[Any]
            Objects to topic model (e.g., text documents).
        embedding_vectors : np.ndarray
            Shape (n_objects, embedding_dim). Object embeddings.
        clusterable_vectors : np.ndarray
            Shape (n_objects, cluster_dim). Vectors for clustering.

        Returns
        -------
        self : Toponymy
            Fitted instance.
        """
        self.clusterable_vectors_ = clusterable_vectors
        self.embedding_vectors_ = embedding_vectors

        # Perform clustering
        if hasattr(self.clusterer, "cluster_layers_") and hasattr(
            self.clusterer, "cluster_tree_"
        ):
            self.cluster_layers_ = self.clusterer.cluster_layers_
            self.cluster_tree_ = self.clusterer.cluster_tree_
        else:
            self.cluster_layers_, self.cluster_tree_ = self.clusterer.fit_predict(
                clusterable_vectors,
                embedding_vectors,
                verbose=self.verbose,
                show_progress_bar=self.show_progress_bars,
                prompt_format=self._effective_prompt_format(),
                prompt_template=self.prompt_template,
            )

        self._sync_layer_runtime_config()

        # Initialize topic name storage
        self.topic_names_: List[List[str]] = [[]] * len(self.cluster_layers_)
        self.topic_name_vectors_: List[np.ndarray] = [np.array([])] * len(
            self.cluster_layers_
        )

        # Check if using summary-based topics
        _summarize_topics = isinstance(self.cluster_layers_[0], ClusterLayerSummaryText)
        if _summarize_topics:
            self.topic_summaries_: List[List[str]] = [[]] * len(self.cluster_layers_)
            self.topic_explanations_: List[List[str]] = [[]] * len(self.cluster_layers_)

        detail_levels = np.linspace(
            self.lowest_detail_level,
            self.highest_detail_level,
            len(self.cluster_layers_),
        )

        # Extract keyphrases using feature extractors
        keyphrase_extractor = next(
            (fe for fe in self.feature_extractors if isinstance(fe, KeyphraseBuilder)),
            None,
        )

        if keyphrase_extractor:
            if (
                hasattr(self.cluster_layers_[0], "object_to_text_function")
                and self.cluster_layers_[0].object_to_text_function is not None
            ):
                # Non-text objects: use exemplar texts for keyphrases
                exemplars, exemplar_indices = self.cluster_layers_[
                    0
                ].make_exemplar_texts(objects, embedding_vectors)
                aligned_texts = [""] * len(objects)
                for cluster_idx, cluster_exemplars in enumerate(exemplars):
                    for exemplar_idx, exemplar_text in zip(
                        exemplar_indices[cluster_idx], cluster_exemplars
                    ):
                        aligned_texts[exemplar_idx] = exemplar_text

                (
                    self.object_x_keyphrase_matrix_,
                    self.keyphrase_list_,
                    self.keyphrase_vectors_,
                ) = keyphrase_extractor.fit_transform(aligned_texts)
            else:
                # Text objects: extract keyphrases directly
                (
                    self.object_x_keyphrase_matrix_,
                    self.keyphrase_list_,
                    self.keyphrase_vectors_,
                ) = keyphrase_extractor.fit_transform(objects)

                self.cluster_layers_[0].make_exemplar_texts(objects, embedding_vectors)

            if self.keyphrase_vectors_ is None and len(self.keyphrase_list_) > 0:
                self.keyphrase_vectors_ = self.embedding_model.encode(
                    self.keyphrase_list_, show_progress_bar=self.show_progress_bars
                )
        else:
            # No keyphrase extractor
            self.object_x_keyphrase_matrix_ = None
            self.keyphrase_list_ = []
            self.keyphrase_vectors_ = None
            self.cluster_layers_[0].make_exemplar_texts(objects, embedding_vectors)

        # Generate topic names for each layer
        for i, layer in tqdm(
            enumerate(self.cluster_layers_),
            desc="Building topic names by layer",
            disable=not self.show_progress_bars,
            total=len(self.cluster_layers_),
            unit="layer",
        ):
            if i > 0:
                layer.make_exemplar_texts(objects, embedding_vectors)

            if self.object_x_keyphrase_matrix_ is not None:
                layer.make_keyphrases(
                    self.keyphrase_list_,
                    self.object_x_keyphrase_matrix_,
                    self.keyphrase_vectors_,
                    self.embedding_model,
                )

            if i > 0 and not hasattr(self.cluster_layers_[0], "topic_name_embeddings"):
                self.cluster_layers_[0].embed_topic_names(self.embedding_model)

            if _summarize_topics:
                if i > 0:
                    layer.make_subtopics(
                        self.topic_names_[0],
                        self.cluster_layers_[0].cluster_labels,
                        self.cluster_layers_[0].topic_name_embeddings,
                        self.embedding_model,
                        topic_summaries=self.topic_summaries_[0],
                        topic_explanations=self.topic_explanations_[0],
                    )
                layer.make_prompts(
                    detail_levels[i],
                    self.topic_names_,
                    self.object_description,
                    self.corpus_description,
                    self.cluster_tree_,
                    None,
                    None,
                    self.topic_summaries_,
                    self.topic_explanations_,
                )
                (
                    self.topic_names_[i],
                    self.topic_summaries_[i],
                    self.topic_explanations_[i],
                ) = layer.name_topics(
                    self.llm_namer,
                    detail_levels[i],
                    self.topic_names_,
                    self.object_description,
                    self.corpus_description,
                    self.cluster_tree_,
                    self.embedding_model,
                    self.topic_summaries_,
                    self.topic_explanations_,
                )
            else:
                if i > 0:
                    layer.make_subtopics(
                        self.topic_names_[0],
                        self.cluster_layers_[0].cluster_labels,
                        self.cluster_layers_[0].topic_name_embeddings,
                        self.embedding_model,
                    )
                layer.make_prompts(
                    detail_levels[i],
                    self.topic_names_,
                    self.object_description,
                    self.corpus_description,
                    self.cluster_tree_,
                )
                self.topic_names_[i] = layer.name_topics(
                    self.llm_namer,
                    detail_levels[i],
                    self.topic_names_,
                    self.object_description,
                    self.corpus_description,
                    self.cluster_tree_,
                    self.embedding_model,
                )

            self.topic_name_vectors_[i] = layer.make_topic_name_vector()

        return self

    def fit_predict(
        self,
        objects: List[Any],
        object_vectors: np.ndarray,
        clusterable_vectors: np.ndarray,
    ) -> List[np.ndarray]:
        """Fit the model and return topic name vectors."""
        self.fit(objects, object_vectors, clusterable_vectors)
        return self.topic_name_vectors_

    @property
    def topic_tree_(self) -> TopicTree:
        """Get the topic tree representation."""
        check_is_fitted(self, ["cluster_tree_", "topic_names_", "topic_name_vectors_"])

        def cluster_size(cluster_label_array):
            if cluster_label_array.min() < 0:
                return np.bincount(cluster_label_array - cluster_label_array.min())[
                    -cluster_label_array.min() :
                ].tolist()
            else:
                return np.bincount(cluster_label_array).tolist()

        topic_sizes = [
            cluster_size(layer.cluster_labels) for layer in self.cluster_layers_
        ]
        return TopicTree(
            self.cluster_tree_,
            self.topic_names_,
            topic_sizes,
            self.embedding_vectors_.shape[0],
        )
