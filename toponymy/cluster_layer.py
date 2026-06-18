from abc import ABC, abstractmethod
from typing import List, Callable, Any, Optional, Tuple, Union, Dict
import scipy.sparse
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
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
    central_summary_subtopics,
    information_weighted_summary_subtopics,
    submodular_summary_subtopics,
)
from toponymy.templates import (
    GET_TOPIC_CLUSTER_NAMES_REGEX,
    GET_TOPIC_NAME_REGEX,
    GET_TOPIC_NAME_AND_SUMMARY_REGEX,
    SUMMARY_KINDS,
    default_extract_topic_names,
)
from toponymy.llm_wrappers import LLMWrapper, AsyncLLMWrapper
from toponymy.embedding_wrappers import TextEmbedderProtocol
from toponymy.prompt_construction import (
    topic_name_prompt,
    cluster_topic_names_for_renaming,
    distinguish_topic_names_prompt,
    topic_summary_prompt,
)
from tqdm.auto import tqdm
import asyncio
from toponymy._utils import handle_verbose_params
import warnings


def _cluster_point_cloud_scale(point_vectors: np.ndarray) -> float:
    point_vectors = np.asarray(point_vectors)
    if point_vectors.size == 0:
        return 1.0

    scale = np.linalg.norm(
        np.nanmax(point_vectors, axis=0) - np.nanmin(point_vectors, axis=0)
    )
    if not np.isfinite(scale) or scale == 0.0:
        return 1.0
    return float(scale)


def _sample_point_cloud(points: np.ndarray, max_points: int = 256) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points

    sample_indices = np.linspace(0, points.shape[0] - 1, max_points).astype(int)
    return points[sample_indices]


def _normalized_symmetric_point_cloud_distance(
    point_cloud_a: np.ndarray,
    point_cloud_b: np.ndarray,
    scale: float,
) -> float:
    if point_cloud_a.shape[0] == 0 or point_cloud_b.shape[0] == 0:
        return np.inf

    sampled_a = _sample_point_cloud(point_cloud_a)
    sampled_b = _sample_point_cloud(point_cloud_b)
    distances = pairwise_distances(sampled_a, sampled_b)
    symmetric_distance = (
        distances.min(axis=1).mean() + distances.min(axis=0).mean()
    ) / 2.0
    return float(symmetric_distance / scale)


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
        exemplar_delimiters: List[str] = None,
        prompt_format: str = "combined",
        prompt_template: Optional[Dict[str, Any]] = None,
        verbose: Optional[bool] = None,
        show_progress_bar: Optional[bool] = None,
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
        if exemplar_delimiters is None:
            self.exemplar_delimiters = ['    *"', '"\n']
        # Handle verbose parameters
        self.show_progress_bar, self.verbose = handle_verbose_params(
            verbose=verbose, show_progress_bar=show_progress_bar, default_verbose=False
        )

        # Initialize empty lists for the cluster layer's attributes
        self.topic_names: List[str] = []
        self.exemplars: List[List[str]] = []
        self.keyphrases: List[List[str]] = []
        self.subtopics: List[List[str]] = []
        self._prior_topic_reuse_indices: set[int] = set()

    def _prior_topic_payload(
        self,
        prior_layer: "ClusterLayer",
        prior_topic_index: int,
        include_summaries: bool = False,
    ) -> Optional[str]:
        if not hasattr(prior_layer, "topic_names"):
            return None
        if prior_topic_index >= len(prior_layer.topic_names):
            return None

        topic_name = prior_layer.topic_names[prior_topic_index]
        if not include_summaries:
            return topic_name

        topic_summaries = getattr(prior_layer, "topic_summaries", [])
        topic_explanations = getattr(prior_layer, "topic_explanations", [])
        topic_summary = (
            topic_summaries[prior_topic_index]
            if prior_topic_index < len(topic_summaries)
            else ""
        )
        topic_explanation = (
            topic_explanations[prior_topic_index]
            if prior_topic_index < len(topic_explanations)
            else ""
        )
        return f"{topic_name}\n--\n{topic_summary}\n--\n{topic_explanation}"

    def _mark_prompts_for_prior_topic_reuse(
        self,
        previous_layer: Optional["ClusterLayer"] = None,
        clusterable_vectors: Optional[np.ndarray] = None,
        cluster_reuse_distance_threshold: Optional[float] = None,
        include_summaries: bool = False,
    ) -> None:
        self._prior_topic_reuse_indices = set()
        if (
            previous_layer is None
            or clusterable_vectors is None
            or cluster_reuse_distance_threshold is None
            or not hasattr(self, "prompts")
        ):
            return

        if clusterable_vectors.shape[0] != len(self.cluster_labels):
            return
        if not hasattr(previous_layer, "topic_names"):
            return

        previous_cluster_labels = previous_layer.cluster_labels
        if len(previous_cluster_labels) > len(self.cluster_labels):
            return

        current_clusterable_vectors = clusterable_vectors[: len(self.cluster_labels)]
        previous_clusterable_vectors = clusterable_vectors[
            : len(previous_cluster_labels)
        ]

        current_topic_count = self.centroid_vectors.shape[0]
        previous_topic_count = len(previous_layer.topic_names)
        if current_topic_count == 0 or previous_topic_count == 0:
            return

        scale = _cluster_point_cloud_scale(current_clusterable_vectors)
        available_prior_topic_indices = set(range(previous_topic_count))
        reuse_candidates = []

        for topic_index in range(current_topic_count):
            current_points = current_clusterable_vectors[
                self.cluster_labels == topic_index
            ]
            if current_points.shape[0] == 0:
                continue

            best_prior_index = None
            best_distance = np.inf
            for prior_topic_index in range(previous_topic_count):
                if prior_topic_index not in available_prior_topic_indices:
                    continue

                prior_points = previous_clusterable_vectors[
                    previous_cluster_labels == prior_topic_index
                ]
                if prior_points.shape[0] == 0:
                    continue

                distance = _normalized_symmetric_point_cloud_distance(
                    current_points,
                    prior_points,
                    scale,
                )
                if distance < best_distance:
                    best_distance = distance
                    best_prior_index = prior_topic_index

            if (
                best_prior_index is not None
                and best_distance <= cluster_reuse_distance_threshold
            ):
                reuse_candidates.append((best_distance, topic_index, best_prior_index))

        for _, topic_index, prior_topic_index in sorted(reuse_candidates):
            if prior_topic_index not in available_prior_topic_indices:
                continue

            payload = self._prior_topic_payload(
                previous_layer,
                prior_topic_index,
                include_summaries=include_summaries,
            )
            if payload is None:
                continue

            self.prompts[topic_index] = f"[!SKIP!]: {payload}"
            self._prior_topic_reuse_indices.add(topic_index)
            available_prior_topic_indices.remove(prior_topic_index)

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
        all_topic_summaries: Optional[List[List[str]]] = None,
        all_topic_explanations: Optional[List[List[str]]] = None,
        previous_layer: Optional["ClusterLayer"] = None,
        clusterable_vectors: Optional[np.ndarray] = None,
        cluster_reuse_distance_threshold: Optional[float] = None,
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
        prompt_format: str = None,
        prompt_template: Optional[str] = None,
        all_topic_summaries: Optional[List[List[str]]] = None,
        all_topic_explanations: Optional[List[List[str]]] = None,
    ) -> List[str | Dict[str, str]]:
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
        method: str = "facility_location",
        topic_summaries: Optional[List[str]] = None,
        topic_explanations: Optional[List[str]] = None,
    ) -> List[List[str]]:
        pass

    @abstractmethod
    def make_exemplar_texts(
        self,
        object_list: List[Any],
        object_vectors: np.ndarray,
        method: str = "central",
    ) -> Tuple[List[List[str]], List[List[int]]]:
        pass

    def embed_topic_names(
        self,
        embedding_model: Optional[TextEmbedderProtocol] = None,
    ) -> None:
        if embedding_model is None and self.text_embedding_model is None:
            raise ValueError("An embedding model must be provided")
        elif embedding_model is None:
            embedding_model = self.text_embedding_model
        assert (
            embedding_model is not None
        ), "Embedding model should not be None by this point"

        self.topic_name_embeddings = embedding_model.encode(
            self.topic_names, show_progress_bar=self.show_progress_bar
        )

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
        frozen_topic_indices = getattr(self, "_prior_topic_reuse_indices", set())
        if frozen_topic_indices:
            self.dismbiguation_topic_indices = [
                np.array(
                    [
                        topic_index
                        for topic_index in topic_indices
                        if topic_index not in frozen_topic_indices
                    ]
                )
                for topic_indices in self.dismbiguation_topic_indices
            ]
            self.dismbiguation_topic_indices = [
                topic_indices
                for topic_indices in self.dismbiguation_topic_indices
                if len(topic_indices) > 1
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
                    disambiguation_prompt,
                    [self.topic_names[i] for i in topic_indices],
                    extract_topic_names_function=(
                        self.prompt_template["disambiguate_topics"][
                            "extract_topic_names"
                        ]
                        if self.prompt_template
                        else default_extract_topic_names
                    ),
                    get_topic_names_regex=(
                        self.prompt_template["disambiguate_topics"].get(
                            "get_topic_names_regex", GET_TOPIC_CLUSTER_NAMES_REGEX
                        )
                        if self.prompt_template
                        else GET_TOPIC_CLUSTER_NAMES_REGEX
                    ),
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
                    extract_topic_names_function=(
                        self.prompt_template["disambiguate_topics"][
                            "extract_topic_names"
                        ]
                        if self.prompt_template
                        else default_extract_topic_names
                    ),
                    get_topic_names_regex=(
                        self.prompt_template["disambiguate_topics"].get(
                            "get_topic_names_regex", GET_TOPIC_CLUSTER_NAMES_REGEX
                        )
                        if self.prompt_template
                        else GET_TOPIC_CLUSTER_NAMES_REGEX
                    ),
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
        exemplar_delimiters: List[str] = None,
        prompt_format: str = "combined",
        prompt_template: Optional[Dict[str, Any]] = None,
        verbose: Optional[bool] = None,
        show_progress_bar: Optional[bool] = None,
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
        if text_embedding_model is not None:
            self.embedding_model = text_embedding_model

    def make_prompts(
        self,
        detail_level: float,
        all_topic_names: List[List[str]],
        object_description: str,
        corpus_description: str,
        cluster_tree: Optional[dict] = None,
        prompt_format: Optional[str] = None,
        prompt_template: Optional[Dict[str, Any]] = None,
        all_topic_summaries: Optional[List[List[str]]] = None,
        all_topic_explanations: Optional[List[List[str]]] = None,
    ) -> List[str | Dict[str, str]]:
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
        all_topic_summaries: Optional[List[List[str]]] = None,
        all_topic_explanations: Optional[List[List[str]]] = None,
        previous_layer: Optional[ClusterLayer] = None,
        clusterable_vectors: Optional[np.ndarray] = None,
        cluster_reuse_distance_threshold: Optional[float] = None,
    ) -> List[str]:
        self._mark_prompts_for_prior_topic_reuse(
            previous_layer=previous_layer,
            clusterable_vectors=clusterable_vectors,
            cluster_reuse_distance_threshold=cluster_reuse_distance_threshold,
        )

        if isinstance(llm, LLMWrapper):
            self.topic_names = [
                (
                    llm.generate_topic_name(
                        prompt,
                        topic_extraction_function=(
                            self.prompt_template["layer"]["extract_topic_name"]
                            if self.prompt_template
                            else lambda json_response: str(json_response["topic_name"])
                        ),
                        get_topic_name_regex=(
                            self.prompt_template["layer"].get(
                                "get_topic_name_regex", GET_TOPIC_NAME_REGEX
                            )
                            if self.prompt_template
                            else GET_TOPIC_NAME_REGEX
                        ),
                    )
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
                llm.generate_topic_names(
                    [prompt for _, prompt in prompts_for_llm],
                    extract_topic_name_function=(
                        self.prompt_template["layer"]["extract_topic_name"]
                        if self.prompt_template
                        else lambda json_response: json_response["topic_name"]
                    ),
                    get_topic_name_regex=(
                        self.prompt_template["layer"].get(
                            "get_topic_name_regex", GET_TOPIC_NAME_REGEX
                        )
                        if self.prompt_template
                        else GET_TOPIC_NAME_REGEX
                    ),
                )
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

        # Try to fix any failures to generate a name
        if any([name == "" for name in self.topic_names]):
            if isinstance(llm, LLMWrapper):
                self.topic_names = [
                    (
                        llm.generate_topic_name(
                            prompt,
                            topic_extraction_function=(
                                self.prompt_template["layer"]["extract_topic_name"]
                                if self.prompt_template
                                else lambda json_response: str(
                                    json_response["topic_name"]
                                )
                            ),
                            get_topic_name_regex=(
                                self.prompt_template["layer"].get(
                                    "get_topic_name_regex", GET_TOPIC_NAME_REGEX
                                )
                                if self.prompt_template
                                else GET_TOPIC_NAME_REGEX
                            ),
                        )
                        if name == "" and index not in self._prior_topic_reuse_indices
                        else name
                    )
                    for index, (name, prompt) in enumerate(
                        zip(self.topic_names, self.prompts)
                    )
                ]
            elif isinstance(llm, AsyncLLMWrapper):
                selected_prompts = [
                    prompt
                    for index, (name, prompt) in enumerate(
                        zip(self.topic_names, self.prompts)
                    )
                    if name == "" and index not in self._prior_topic_reuse_indices
                ]
                llm_results = run_async(
                    llm.generate_topic_names(
                        selected_prompts,
                        extract_topic_name_function=(
                            self.prompt_template["layer"]["extract_topic_name"]
                            if self.prompt_template
                            else lambda json_response: str(json_response["topic_name"])
                        ),
                        get_topic_name_regex=(
                            self.prompt_template["layer"].get(
                                "get_topic_name_regex", GET_TOPIC_NAME_REGEX
                            )
                            if self.prompt_template
                            else GET_TOPIC_NAME_REGEX
                        ),
                    )
                )
                for i in range(len(self.topic_names)):
                    if (
                        self.topic_names[i] == ""
                        and i not in self._prior_topic_reuse_indices
                    ):
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
        topic_summaries: Optional[List[str]] = None,
        topic_explanations: Optional[List[str]] = None,
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


class ClusterLayerSummaryText(ClusterLayerText):
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
        exemplar_delimiters: List[str] = None,
        prompt_format: str = "combined",
        prompt_template: Optional[str | Dict[str, Any]] = None,
        verbose: Optional[bool] = None,
        show_progress_bar: Optional[bool] = None,
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
        all_topic_summaries: Optional[List[List[str]]] = None,
        all_topic_explanations: Optional[List[List[str]]] = None,
    ) -> List[str | Dict[str, str]]:
        summary_level = int(round(detail_level * (len(SUMMARY_KINDS) - 1)))
        summary_kind = SUMMARY_KINDS[summary_level]

        assert (
            all_topic_summaries is not None
        ), "all_topic_summaries must be provided to make_prompts in ClusterLayerSummaryText"
        assert (
            all_topic_explanations is not None
        ), "all_topic_explanations must be provided to make_prompts in ClusterLayerSummaryText"

        self.prompts = [
            topic_summary_prompt(
                topic_index,
                self.layer_id,
                all_topic_names,
                all_topic_summaries,
                all_topic_explanations,
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
        all_topic_summaries: Optional[List[List[str]]] = None,
        all_topic_explanations: Optional[List[List[str]]] = None,
        previous_layer: Optional[ClusterLayer] = None,
        clusterable_vectors: Optional[np.ndarray] = None,
        cluster_reuse_distance_threshold: Optional[float] = None,
    ) -> List[str]:
        assert (
            all_topic_summaries is not None
        ), "all_topic_summaries must be provided to name_topics in ClusterLayerSummaryText"
        assert (
            all_topic_explanations is not None
        ), "all_topic_explanations must be provided to name_topics in ClusterLayerSummaryText"

        self._mark_prompts_for_prior_topic_reuse(
            previous_layer=previous_layer,
            clusterable_vectors=clusterable_vectors,
            cluster_reuse_distance_threshold=cluster_reuse_distance_threshold,
            include_summaries=True,
        )

        if isinstance(llm, LLMWrapper):
            self.topic_names = []
            self.topic_summaries = []
            self.topic_explanations = []
            for prompt in tqdm(
                self.prompts,
                desc=f"Generating topic names for layer {self.layer_id}",
                disable=not self.show_progress_bar,
                unit="topic",
                leave=False,
                position=1,
            ):
                if isinstance(prompt, dict) or not prompt.startswith("[!SKIP!]: "):
                    result = llm.generate_topic_name(
                        prompt,
                        topic_extraction_function=(
                            self.prompt_template["layer"]["extract_topic_name"]
                            if self.prompt_template
                            else lambda json_response: (
                                json_response["topic_name"],
                                json_response["topic_summary"],
                                json_response["topic_analysis"],
                            )
                        ),
                        get_topic_name_regex=(
                            self.prompt_template["layer"].get(
                                "get_topic_name_regex", GET_TOPIC_NAME_AND_SUMMARY_REGEX
                            )
                            if self.prompt_template
                            else GET_TOPIC_NAME_AND_SUMMARY_REGEX
                        ),
                        max_tokens=2048,
                    )
                    if isinstance(result, tuple) and len(result) >= 3:
                        name, summary, explanation = result[0], result[1], result[2]
                    else:
                        name, summary, explanation = str(result), "", ""
                else:
                    parts = prompt.removeprefix("[!SKIP!]: ").split("\n--\n", 2)
                    name = parts[0] if len(parts) > 0 else ""
                    summary = parts[1] if len(parts) > 1 else ""
                    explanation = parts[2] if len(parts) > 2 else ""
                self.topic_names.append(name)
                self.topic_summaries.append(summary)
                self.topic_explanations.append(explanation)
        elif isinstance(llm, AsyncLLMWrapper):
            # Filter out prompts that are marked to be skipped
            prompts_for_llm = [
                (index, prompt)
                for index, prompt in enumerate(self.prompts)
                if isinstance(prompt, dict) or not prompt.startswith("[!SKIP!]: ")
            ]
            llm_results = run_async(
                llm.generate_topic_names(
                    [prompt for _, prompt in prompts_for_llm],
                    extract_topic_name_function=(
                        self.prompt_template["layer"]["extract_topic_name"]
                        if self.prompt_template
                        else lambda json_response: (
                            json_response["topic_name"],
                            json_response["topic_summary"],
                            json_response["topic_analysis"],
                        )
                    ),
                    get_topic_name_regex=(
                        self.prompt_template["layer"].get("get_topic_name_regex", None)
                        if self.prompt_template
                        else None
                    ),
                    null_result_value=("", "", ""),
                    max_tokens=2048,
                )
            )
            llm_result_index = 0
            self.topic_names = []
            self.topic_summaries = []
            self.topic_explanations = []
            for index, prompt in enumerate(self.prompts):
                if isinstance(prompt, dict) or not prompt.startswith("[!SKIP!]: "):
                    if llm_results[llm_result_index] == "":
                        name, summary, explanation = "", "", ""
                    else:
                        name, summary, explanation = llm_results[llm_result_index]
                    self.topic_names.append(name)
                    self.topic_summaries.append(summary)
                    self.topic_explanations.append(explanation)
                    llm_result_index += 1
                else:
                    # If the prompt is marked to be skipped, use the original prompt text
                    name, summary, explanation = prompt.removeprefix(
                        "[!SKIP!]: "
                    ).split("\n--\n")
                    self.topic_names.append(name)
                    self.topic_summaries.append(summary)
                    self.topic_explanations.append(explanation)

        all_topic_names[self.layer_id] = self.topic_names
        all_topic_summaries[self.layer_id] = self.topic_summaries
        all_topic_explanations[self.layer_id] = self.topic_explanations

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
        if any([name == "" for name in self.topic_names]):
            if isinstance(llm, LLMWrapper):
                self.topic_names = [
                    (
                        llm.generate_topic_name(
                            prompt,
                            topic_extraction_function=(
                                self.prompt_template["layer"]["extract_topic_name"]
                                if self.prompt_template
                                else lambda json_response: (
                                    json_response["topic_name"],
                                    json_response["topic_summary"],
                                    json_response["topic_analysis"],
                                )
                            ),
                            get_topic_name_regex=(
                                self.prompt_template.get(
                                    "get_topic_name_regex",
                                    GET_TOPIC_NAME_AND_SUMMARY_REGEX,
                                )
                                if self.prompt_template
                                else GET_TOPIC_NAME_AND_SUMMARY_REGEX
                            ),
                        )
                        if name == "" and index not in self._prior_topic_reuse_indices
                        else name
                    )
                    for index, (name, prompt) in enumerate(
                        zip(self.topic_names, self.prompts)
                    )
                ]
            elif isinstance(llm, AsyncLLMWrapper):
                selected_prompts = [
                    prompt
                    for index, (name, prompt) in enumerate(
                        zip(self.topic_names, self.prompts)
                    )
                    if name == "" and index not in self._prior_topic_reuse_indices
                ]
                llm_results = run_async(
                    llm.generate_topic_names(
                        selected_prompts,
                        extract_topic_name_function=(
                            self.prompt_template["layer"]["extract_topic_name"]
                            if self.prompt_template
                            else lambda json_response: json_response["topic_name"]
                        ),
                        get_topic_name_regex=(
                            self.prompt_template["layer"].get(
                                "get_topic_name_regex", GET_TOPIC_NAME_AND_SUMMARY_REGEX
                            )
                            if self.prompt_template
                            else GET_TOPIC_NAME_AND_SUMMARY_REGEX
                        ),
                    )
                )
                for i in range(len(self.topic_names)):
                    if (
                        self.topic_names[i] == ""
                        and i not in self._prior_topic_reuse_indices
                    ):
                        self.topic_names[i] = llm_results.pop(0)
            else:
                raise ValueError(
                    "LLM must be an instance of LLMWrapper or AsyncLLMWrapper."
                )

        return self.topic_names, self.topic_summaries, self.topic_explanations

    def make_subtopics(
        self,
        topic_list: List[str],
        topic_labels: np.ndarray,
        topic_vectors: Optional[np.ndarray] = None,
        embedding_model: Optional[TextEmbedderProtocol] = None,
        method: str = "facility_location",
        topic_summaries: Optional[List[str]] = None,
        topic_explanations: Optional[List[str]] = None,
    ) -> List[List[str]]:
        assert (
            topic_summaries is not None
        ), "topic_summaries must be provided to make_subtopics in ClusterLayerSummaryText"
        assert (
            topic_explanations is not None
        ), "topic_explanations must be provided to make_subtopics in ClusterLayerSummaryText"

        if method == "central":
            self.subtopics = central_summary_subtopics(
                cluster_label_vector=self.cluster_labels,
                subtopics=topic_list,
                summaries=topic_summaries,
                explanations=topic_explanations,
                subtopic_label_vector=topic_labels,
                subtopic_vectors=topic_vectors,
                diversify_alpha=self.subtopic_diversify_alpha,
                n_subtopics=self.n_subtopics,
                embedding_model=embedding_model,
                verbose=self.verbose,
                show_progress_bar=self.show_progress_bar,
            )
        elif method == "information_weighted":
            self.subtopics = information_weighted_summary_subtopics(
                cluster_label_vector=self.cluster_labels,
                subtopics=topic_list,
                summaries=topic_summaries,
                explanations=topic_explanations,
                subtopic_label_vector=topic_labels,
                subtopic_vectors=topic_vectors,
                diversify_alpha=self.subtopic_diversify_alpha,
                n_subtopics=self.n_subtopics,
                embedding_model=embedding_model,
                verbose=self.verbose,
                show_progress_bar=self.show_progress_bar,
            )
        elif method in ("saturated_coverage", "facility_location"):
            self.subtopics = submodular_summary_subtopics(
                cluster_label_vector=self.cluster_labels,
                subtopics=topic_list,
                summaries=topic_summaries,
                explanations=topic_explanations,
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

    def make_topic_name_vector(self) -> np.ndarray:
        self.topic_name_vector = np.full(
            self.cluster_labels.shape[0], "Unlabelled", dtype=object
        )
        for i, name in enumerate(self.topic_names):
            self.topic_name_vector[self.cluster_labels == i] = name

        return self.topic_name_vector
