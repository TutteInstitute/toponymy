from abc import ABC, abstractmethod
from typing import List, Callable, Any, Optional, Tuple, Union, Dict
import scipy.sparse
import numpy as np
import pandas as pd
from toponymy.keyphrases import (
    central_keyphrases,
    information_weighted_keyphrases,
    bm25_keyphrases,
    submodular_selection_information_keyphrases,
)
from toponymy.exemplar_texts import (
    diverse_exemplars,
    submodular_selection_exemplars,
    random_exemplars,
)
from toponymy.subtopics import (
    central_subtopics,
    information_weighted_subtopics,
    submodular_subtopics,
)
from toponymy.templates import SUMMARY_KINDS
from toponymy.llm_wrappers import LLMWrapper, AsyncLLMWrapper
from toponymy.embedding_wrappers import TextEmbedderProtocol
from toponymy.prompt_construction import (
    topic_name_prompt,
    cluster_topic_names_for_renaming,
    distinguish_topic_names_prompt,
    harmonize_over_time_prompt,
)
from tqdm.auto import tqdm
import asyncio
from toponymy._utils import handle_verbose_params
import warnings


def run_async(coro):
    """
    Run an async coroutine in both Jupyter and regular Python environments.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop - regular Python
        return asyncio.run(coro)
    else:
        # Running loop exists - likely Jupyter
        try:
            import nest_asyncio

            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        except ImportError:
            # Fallback to thread-based approach
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()


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
        text_embedding_model: Optional[TextEmbedderProtocol] = None,
        object_to_text_function: Optional[Callable[List[Any], List[str]]] = None,
        n_exemplars: int = 16,
        n_keyphrases: int = 24,
        n_subtopics: int = 24,
        exemplar_delimiters: List[str] = ['    * "', '"\n'],
        prompt_format: str = "combined",
        prompt_template: Optional[Dict[str, Any]] = None,
        verbose: bool = None,
        show_progress_bar: bool = None,
    ):
        self.cluster_labels = cluster_labels
        self.centroid_vectors = centroid_vectors
        self.layer_id = layer_id
        self.text_embedding_model = text_embedding_model
        self.object_to_text_function = object_to_text_function
        self.n_exemplars = n_exemplars
        self.n_keyphrases = n_keyphrases
        self.n_subtopics = n_subtopics
        self.exemplar_delimiters = exemplar_delimiters
        self.prompt_format = prompt_format
        self.prompt_template = prompt_template

        # Handle verbose parameters
        self.show_progress_bar, self.verbose = handle_verbose_params(
            verbose=verbose, show_progress_bar=show_progress_bar, default_verbose=False
        )

        # Initialize empty lists for the cluster layer's attributes
        self.topic_names = []
        self.exemplars = []
        self.keyphrases = []
        self.subtopics = []
        self.previous_names = []

    @abstractmethod
    def name_topics(
        self,
        llm,
        detail_level: float,
        all_topic_names: List[List[str]],
        object_description: str,
        corpus_description: str,
        cluster_tree: Optional[dict] = None,
        embedding_model: Optional[TextEmbedderProtocol] = None,
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
        embedding_model: Optional[TextEmbedderProtocol] = None,
    ) -> List[List[str]]:
        pass

    @abstractmethod
    def make_subtopics(
        self,
        topic_list: List[str],
        topic_labels: np.ndarray,
        topic_vectors: Optional[np.ndarray] = None,
        embedding_model: Optional[TextEmbedderProtocol] = None,
    ) -> List[List[str]]:
        pass

    @abstractmethod
    def make_exemplar_texts(
        self,
        object_list: List[Any],
        object_vectors: np.ndarray,
        method: str = "central",
    ) -> List[List[str]]:
        pass

    def embed_topic_names(
        self,
        embedding_model: Optional[TextEmbedderProtocol] = None,
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
                previous_names=(
                    self.previous_names if len(self.previous_names) > 0 else None
                ),
                cluster_tree=cluster_tree,
                object_description=object_description,
                corpus_description=corpus_description,
                summary_kind=summary_kind,
                max_num_exemplars=self.n_exemplars,
                max_num_keyphrases=self.n_keyphrases,
                max_num_subtopics=self.n_subtopics,
                max_num_history=self.n_history,
                exemplar_start_delimiter=self.exemplar_delimiters[0],
                exemplar_end_delimiter=self.exemplar_delimiters[1],
                prompt_format=self.prompt_format,
                prompt_template=self.prompt_template,
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

    def _update_topic_names(
        self,
        new_topic_names: List[str],
        topic_indices: List[int],
    ) -> None:
        """
        Update the topic names for the specified indices.
        """
        for i, topic_index in enumerate(topic_indices):
            try:
                self.topic_names[topic_index] = new_topic_names[i]
            except IndexError:
                continue

    def _disambiguate_topic_names(self, llm) -> None:  # pragma: no cover
        if isinstance(llm, LLMWrapper):
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
                if len(new_names) == len(topic_indices):
                    self._update_topic_names(new_names, topic_indices)
                else:
                    warnings.warn(
                        f"Got {len(new_names)} new topic names to match {len(topic_indices)}, so we ignore disambiguation effort for {topic_indices}.",
                        RuntimeWarning,
                    )
        elif isinstance(llm, AsyncLLMWrapper):
            llm_results = run_async(
                llm.generate_topic_cluster_names(
                    self.disambiguation_prompts,
                    [
                        [self.topic_names[i] for i in topic_indices]
                        for topic_indices in self.dismbiguation_topic_indices
                    ],
                )
            )
            for topic_indices, new_names in zip(
                self.dismbiguation_topic_indices, llm_results
            ):
                self._update_topic_names(new_names, topic_indices)
        else:
            raise ValueError(
                "LLM must be an instance of LLMWrapper or AsyncLLMWrapper."
            )

    # pragma: no cover
    def disambiguate_topics(
        self,
        llm,
        detail_level: float,
        all_topic_names: List[List[str]],
        object_description: str,
        corpus_description: str,
        cluster_tree: Optional[dict] = None,
        embedding_model: Optional[TextEmbedderProtocol] = None,
    ):
        self.embed_topic_names(embedding_model)
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
        text_embedding_model: Optional[TextEmbedderProtocol] = None,
        n_keyphrases: int = 16,
        keyphrase_diversify_alpha: float = 1.0,
        n_exemplars: int = 8,
        exemplars_diversify_alpha: float = 1.0,
        n_subtopics: int = 16,
        subtopic_diversify_alpha: float = 1.0,
        n_history: int = 2,
        exemplar_delimiters: List[str] = ['    * "', '"\n'],
        prompt_format: str = "combined",
        prompt_template: Optional[str] = None,
        verbose: bool = None,
        show_progress_bar: bool = None,
        **kwargs: Any,
    ):
        super().__init__(
            cluster_labels,
            centroid_vectors,
            layer_id,
            text_embedding_model,
            exemplar_delimiters=exemplar_delimiters,
            prompt_format=prompt_format,
            prompt_template=prompt_template,
            verbose=verbose,
            show_progress_bar=show_progress_bar,
            **kwargs,
        )
        self.n_keyphrases = n_keyphrases
        self.keyphrase_diversify_alpha = keyphrase_diversify_alpha
        self.n_exemplars = n_exemplars
        self.exemplars_diversify_alpha = exemplars_diversify_alpha
        self.n_subtopics = n_subtopics
        self.subtopic_diversify_alpha = subtopic_diversify_alpha
        self.n_history = n_history
        if text_embedding_model is not None:
            self.embedding_model = text_embedding_model

    def make_prompts(
        self,
        detail_level: float,
        all_topic_names: List[List[str]],
        object_description: str,
        corpus_description: str,
        cluster_tree: Optional[dict] = None,
        prompt_format: str = None,
        prompt_template: Optional[str] = None,
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
                previous_names=(
                    self.previous_names if len(self.previous_names) > 0 else None
                ),
                cluster_tree=cluster_tree,
                object_description=object_description,
                corpus_description=corpus_description,
                summary_kind=summary_kind,
                max_num_exemplars=self.n_exemplars,
                max_num_keyphrases=self.n_keyphrases,
                max_num_subtopics=self.n_subtopics,
                exemplar_start_delimiter=self.exemplar_delimiters[0],
                exemplar_end_delimiter=self.exemplar_delimiters[1],
                prompt_format=(
                    self.prompt_format if prompt_format is None else prompt_format
                ),
                prompt_template=(
                    self.prompt_template if prompt_template is None else prompt_template
                ),
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

    def make_temporal_prompts(
        self,
        all_topic_names: List[List[str]],
        object_description: str,
        corpus_description: str,
        cluster_tree: Optional[dict] = None,
        prompt_format: str = None,
        prompt_template: Optional[str] = None,
    ) -> List[str]:

        self.temporal_prompts = [
            harmonize_over_time_prompt(
                topic_index,
                self.layer_id,
                all_topic_names,
                exemplar_texts=self.exemplars,
                keyphrases=self.keyphrases,
                subtopics=self.subtopics,
                previous_names=(
                    self.previous_names if len(self.previous_names) > 0 else None
                ),
                cluster_tree=cluster_tree,
                object_description=object_description,
                corpus_description=corpus_description,
                max_num_exemplars=self.n_exemplars,
                max_num_keyphrases=self.n_keyphrases,
                max_num_subtopics=self.n_subtopics,
                exemplar_start_delimiter=self.exemplar_delimiters[0],
                exemplar_end_delimiter=self.exemplar_delimiters[1],
                prompt_format=(
                    self.prompt_format if prompt_format is None else prompt_format
                ),
                prompt_template="harmonize_temporal",
            )
            for topic_index in tqdm(
                range(self.centroid_vectors.shape[0]),
                desc=f"Generating temporal prompts for layer {self.layer_id}",
                disable=not self.show_progress_bar,
                unit="topic",
                leave=False,
                position=1,
            )
        ]

        return self.temporal_prompts

    # pragma: no cover
    def name_topics(
        self,
        llm,
        detail_level: float,
        all_topic_names: List[List[str]],
        object_description: str,
        corpus_description: str,
        cluster_tree: Optional[dict] = None,
        embedding_model: Optional[TextEmbedderProtocol] = None,
    ) -> List[str]:
        if isinstance(llm, LLMWrapper):
            self.topic_names = [
                (
                    llm.generate_topic_name(prompt)
                    if isinstance(prompt, dict) or not prompt.startswith("[!SKIP!]: ")
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
        elif isinstance(llm, AsyncLLMWrapper):
            # Filter out prompts that are marked to be skipped
            prompts_for_llm = [
                (index, prompt)
                for index, prompt in enumerate(self.prompts)
                if isinstance(prompt, dict) or not prompt.startswith("[!SKIP!]: ")
            ]
            llm_results = run_async(
                llm.generate_topic_names([prompt for _, prompt in prompts_for_llm])
            )
            llm_result_index = 0
            self.topic_names = []
            for index, prompt in enumerate(self.prompts):
                if isinstance(prompt, dict) or not prompt.startswith("[!SKIP!]: "):
                    self.topic_names.append(llm_results[llm_result_index])
                    llm_result_index += 1
                else:
                    # If the prompt is marked to be skipped, use the original prompt text
                    self.topic_names.append(prompt.removeprefix("[!SKIP!]: "))

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

        # If there is temporal information, try to collapse similar topic names in time
        if self.previous_names is not None:
            self.harmonize_topics_over_time(
                llm=llm,
                detail_level=detail_level,
                all_topic_names=all_topic_names,
                object_description=object_description,
                corpus_description=corpus_description,
                cluster_tree=cluster_tree,
                embedding_model=embedding_model,
            )

        # Try to fix any failures to generate a name
        if any([name == "" for name in self.topic_names]):
            if isinstance(llm, LLMWrapper):
                self.topic_names = [
                    llm.generate_topic_name(prompt) if name == "" else name
                    for name, prompt in zip(self.topic_names, self.prompts)
                ]
            elif isinstance(llm, AsyncLLMWrapper):
                selected_prompts = [
                    prompt
                    for name, prompt in zip(self.topic_names, self.prompts)
                    if name == ""
                ]
                llm_results = run_async(llm.generate_topic_names(selected_prompts))
                for i in range(len(self.topic_names)):
                    if self.topic_names[i] == "":
                        self.topic_names[i] = llm_results.pop(0)
            else:
                raise ValueError(
                    "LLM must be an instance of LLMWrapper or AsyncLLMWrapper."
                )

        return self.topic_names

    def make_keyphrases(
        self,
        keyphrase_list: List[str],
        object_x_keyphrase_matrix: scipy.sparse.spmatrix,
        keyphrase_vectors: np.ndarray,
        embedding_model: Optional[TextEmbedderProtocol] = None,
        method: str = "information_weighted",
    ) -> List[List[str]]:
        if method == "information_weighted":
            self.keyphrases = information_weighted_keyphrases(
                self.cluster_labels,
                object_x_keyphrase_matrix,
                keyphrase_list,
                keyphrase_vectors,
                embedding_model,
                max_alpha=self.keyphrase_diversify_alpha,
                n_keyphrases=self.n_keyphrases,
                verbose=self.verbose,
                show_progress_bar=self.show_progress_bar,
            )
        elif method == "central":
            self.keyphrases = central_keyphrases(
                self.cluster_labels,
                object_x_keyphrase_matrix,
                keyphrase_list,
                keyphrase_vectors,
                embedding_model,
                diversify_alpha=self.keyphrase_diversify_alpha,
                n_keyphrases=self.n_keyphrases,
                verbose=self.verbose,
                show_progress_bar=self.show_progress_bar,
            )
        elif method == "bm25":
            self.keyphrases = bm25_keyphrases(
                self.cluster_labels,
                object_x_keyphrase_matrix,
                keyphrase_list,
                keyphrase_vectors,
                embedding_model,
                diversify_alpha=self.keyphrase_diversify_alpha,
                n_keyphrases=self.n_keyphrases,
                verbose=self.verbose,
                show_progress_bar=self.show_progress_bar,
            )
        elif method in ("saturated_coverage", "facility_location", "graph_cut"):
            self.keyphrases = submodular_selection_information_keyphrases(
                self.cluster_labels,
                object_x_keyphrase_matrix,
                keyphrase_list,
                keyphrase_vectors,
                embedding_model,
                n_keyphrases=self.n_keyphrases,
                submodular_function=method,
                verbose=self.verbose,
                show_progress_bar=self.show_progress_bar,
            )
        else:
            raise ValueError(
                f"Unknown keyphrase generation method: {method}. "
                "Use 'information_weighted', 'central', 'saturated_coverage', 'facility_location, 'graph_cut', or 'bm25'."
            )

        return self.keyphrases

    def make_subtopics(
        self,
        topic_list: List[str],
        topic_labels: np.ndarray,
        topic_vectors: Optional[np.ndarray] = None,
        embedding_model: Optional[TextEmbedderProtocol] = None,
        method: str = "facility_location",
    ) -> List[List[str]]:
        if method == "central":
            self.subtopics = central_subtopics(
                cluster_label_vector=self.cluster_labels,
                subtopics=topic_list,
                subtopic_label_vector=topic_labels,
                subtopic_vectors=topic_vectors,
                diversify_alpha=self.subtopic_diversify_alpha,
                n_subtopics=self.n_subtopics,
                embedding_model=embedding_model,
                verbose=self.verbose,
                show_progress_bar=self.show_progress_bar,
            )
        elif method == "information_weighted":
            self.subtopics = information_weighted_subtopics(
                cluster_label_vector=self.cluster_labels,
                subtopics=topic_list,
                subtopic_label_vector=topic_labels,
                subtopic_vectors=topic_vectors,
                diversify_alpha=self.subtopic_diversify_alpha,
                n_subtopics=self.n_subtopics,
                embedding_model=embedding_model,
                verbose=self.verbose,
                show_progress_bar=self.show_progress_bar,
            )
        elif method in ("saturated_coverage", "facility_location"):
            self.subtopics = submodular_subtopics(
                cluster_label_vector=self.cluster_labels,
                subtopics=topic_list,
                subtopic_label_vector=topic_labels,
                subtopic_vectors=topic_vectors,
                n_subtopics=self.n_subtopics,
                embedding_model=embedding_model,
                submodular_function=method,
                verbose=self.verbose,
                show_progress_bar=self.show_progress_bar,
            )
        else:
            raise ValueError(
                f"Unknown subtopic generation method: {method}. "
                "Use 'central' or 'information_weighted'."
            )

        return self.subtopics

    def make_exemplar_texts(
        self,
        object_list: List[str],
        object_vectors: np.ndarray,
        method="facility_location",
    ) -> Tuple[List[List[str]], List[List[int]]]:
        if method == "central":
            self.exemplars, self.exemplar_indices = diverse_exemplars(
                cluster_label_vector=self.cluster_labels,
                objects=object_list,
                object_vectors=object_vectors,
                centroid_vectors=self.centroid_vectors,
                n_exemplars=self.n_exemplars,
                diversify_alpha=self.exemplars_diversify_alpha,
                object_to_text_function=self.object_to_text_function,
                verbose=self.verbose,
                show_progress_bar=self.show_progress_bar,
            )
        elif method == "facility_location" or method == "saturated_coverage":
            self.exemplars, self.exemplar_indices = submodular_selection_exemplars(
                cluster_label_vector=self.cluster_labels,
                objects=object_list,
                object_vectors=object_vectors,
                n_exemplars=self.n_exemplars,
                object_to_text_function=self.object_to_text_function,
                submodular_function=method,
                verbose=self.verbose,
                show_progress_bar=self.show_progress_bar,
            )
        elif method == "random":
            self.exemplars, self.exemplar_indices = random_exemplars(
                cluster_label_vector=self.cluster_labels,
                objects=object_list,
                n_exemplars=self.n_exemplars,
                object_to_text_function=self.object_to_text_function,
                verbose=self.verbose,
                show_progress_bar=self.show_progress_bar,
            )
        else:
            raise ValueError(
                f"Unknown exemplar generation method: {method}. " "Use 'central'."
            )

        return self.exemplars, self.exemplar_indices

    def make_topic_name_vector(self) -> np.ndarray:
        self.topic_name_vector = np.full(
            self.cluster_labels.shape[0], "Unlabelled", dtype=object
        )
        for i, name in enumerate(self.topic_names):
            self.topic_name_vector[self.cluster_labels == i] = name

        return self.topic_name_vector

    def _harmonize_topic_names_over_time(self, llm):
        if isinstance(llm, LLMWrapper):
            self.topic_names = [
                (
                    llm.generate_topic_name(prompt)
                    if isinstance(prompt, dict) or not prompt.startswith("[!SKIP!]: ")
                    else prompt.removeprefix("[!SKIP!]: ")
                )
                for prompt in tqdm(
                    self.temporal_prompts,
                    desc=f"Harmonizing topic names over time for layer {self.layer_id}",
                    disable=not self.show_progress_bar,
                    unit="topic",
                    leave=False,
                    position=1,
                )
            ]
        elif isinstance(llm, AsyncLLMWrapper):
            # Filter out prompts that are marked to be skipped
            prompts_for_llm = [
                (index, prompt)
                for index, prompt in enumerate(self.temporal_prompts)
                if isinstance(prompt, dict) or not prompt.startswith("[!SKIP!]: ")
            ]
            llm_results = run_async(
                llm.generate_topic_names([prompt for _, prompt in prompts_for_llm])
            )
            llm_result_index = 0
            self.topic_names = []
            for index, prompt in enumerate(self.prompts):
                if isinstance(prompt, dict) or not prompt.startswith("[!SKIP!]: "):
                    self.topic_names.append(llm_results[llm_result_index])
                    llm_result_index += 1
                else:
                    # If the prompt is marked to be skipped, use the original prompt text
                    self.topic_names.append(prompt.removeprefix("[!SKIP!]: "))

    def harmonize_topics_over_time(
        self,
        llm,
        detail_level: float,
        all_topic_names: List[List[str]],
        object_description: str,
        corpus_description: str,
        cluster_tree: Optional[dict] = None,
        embedding_model: Optional[TextEmbedderProtocol] = None,
    ):
        self.make_temporal_prompts(
            all_topic_names,
            object_description=object_description,
            corpus_description=corpus_description,
            cluster_tree=cluster_tree,
            prompt_format="combined",  # todo
        )
        self._harmonize_topic_names_over_time(llm)
