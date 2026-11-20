import numpy as np
import scipy.sparse

from typing import List, Optional
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import DictionaryLearning

from vectorizers.transformers import InformationWeightTransformer

from toponymy.utility_functions import diversify_max_alpha as diversify
from toponymy.annotation import DescendantsInput, LayerInput
from toponymy.exemplar_texts import (
    FacilityLocationSelection,
    SaturatedCoverageSelection,
)
from toponymy._utils import handle_verbose_params
from toponymy.embedding_wrappers import TextEmbedderProtocol

from tqdm.auto import tqdm


def subtopic_embeddings(
    subtopics: List[str], embedding_model: TextEmbedderProtocol
) -> np.ndarray:
    return embedding_model.encode(subtopics)


def central_subtopics(
    cluster_label_vector: np.ndarray,
    subtopics: List[str],
    subtopic_label_vector: np.ndarray,
    subtopic_vectors: Optional[np.ndarray] = None,
    embedding_model: Optional[TextEmbedderProtocol] = None,
    n_subtopics: int = 64,
    diversify_alpha: float = 1.0,
    verbose: bool = None,
    show_progress_bar: bool = None,
) -> List[List[str]]:
    if subtopic_vectors is None:
        if embedding_model is None:
            raise ValueError(
                "Either subtopic_embeddings or embedding_model must be provided"
            )
        subtopic_vectors = subtopic_embeddings(subtopics, embedding_model)

    central_vector = np.mean(subtopic_vectors, axis=0)

    # Handle verbose parameters
    show_progress_bar_val, _ = handle_verbose_params(
        verbose=verbose, show_progress_bar=show_progress_bar, default_verbose=False
    )

    result = []
    for cluster_num in tqdm(
        range(cluster_label_vector.max() + 1),
        desc="Selecting central subtopics",
        disable=not show_progress_bar_val,
        leave=False,
        unit="cluster",
        position=1,
    ):
        subtopic_label_vector_for_cluster = subtopic_label_vector[
            cluster_label_vector == cluster_num
        ]
        cluster_subtopic_indices = np.unique(subtopic_label_vector_for_cluster)
        candidate_subtopic_centroids = (
            np.asarray(
                [
                    subtopic_vectors[subtopic_index]
                    for subtopic_index in cluster_subtopic_indices
                    if subtopic_index != -1
                ]
            )
            - central_vector
        )
        candidate_subtopics = np.asarray(
            [
                subtopics[subtopic_index]
                for subtopic_index in cluster_subtopic_indices
                if subtopic_index != -1
            ]
        )
        cluster_centroid = np.mean(candidate_subtopic_centroids, axis=0)
        candidate_subtopic_indices = np.argsort(
            np.squeeze(
                pairwise_distances(
                    [cluster_centroid], candidate_subtopic_centroids, metric="cosine"
                )
            )
        )[: n_subtopics**2]
        sorted_candidate_subtopic_centroids = candidate_subtopic_centroids[
            candidate_subtopic_indices
        ]
        sorted_candidate_subtopics = [
            candidate_subtopics[i] for i in candidate_subtopic_indices
        ]
        chosen_subtopic_indices = diversify(
            cluster_centroid,
            sorted_candidate_subtopic_centroids,
            n_subtopics,
            max_alpha=diversify_alpha,
        )[:n_subtopics]
        chosen_subtopics = [
            sorted_candidate_subtopics[j] for j in chosen_subtopic_indices
        ]

        result.append(chosen_subtopics)

    return result


def submodular_subtopics(
    cluster_label_vector: np.ndarray,
    subtopics: List[str],
    subtopic_label_vector: np.ndarray,
    subtopic_vectors: Optional[np.ndarray] = None,
    embedding_model: Optional[TextEmbedderProtocol] = None,
    n_subtopics: int = 64,
    submodular_function: str = "facility_location",
    verbose: bool = None,
    show_progress_bar: bool = None,
) -> List[List[str]]:
    if subtopic_vectors is None:
        if embedding_model is None:
            raise ValueError(
                "Either subtopic_embeddings or embedding_model must be provided"
            )
        subtopic_vectors = subtopic_embeddings(subtopics, embedding_model)

    null_topic_vector = np.mean(subtopic_vectors, axis=0)
    if submodular_function == "facility_location":
        selector = FacilityLocationSelection(
            n_subtopics, metric="cosine", optimizer="lazy"
        )
    elif submodular_function == "saturated_coverage":
        selector = SaturatedCoverageSelection(
            n_subtopics, metric="cosine", optimizer="lazy"
        )
    else:
        raise ValueError(
            f"selection_function={submodular_function} is not a valid selection. Please choose one of (facility_location,saturated_coverage)"
        )

    # Handle verbose parameters
    show_progress_bar_val, _ = handle_verbose_params(
        verbose=verbose, show_progress_bar=show_progress_bar, default_verbose=False
    )

    result = []
    for cluster_num in tqdm(
        range(cluster_label_vector.max() + 1),
        desc=f"Selecting {submodular_function} subtopics",
        disable=not show_progress_bar_val,
        leave=False,
        unit="cluster",
        position=1,
    ):
        subtopic_label_vector_for_cluster = subtopic_label_vector[
            cluster_label_vector == cluster_num
        ]
        cluster_subtopic_indices = np.unique(subtopic_label_vector_for_cluster)
        candidate_subtopic_centroids = (
            np.asarray(
                [
                    subtopic_vectors[subtopic_index]
                    for subtopic_index in cluster_subtopic_indices
                    if subtopic_index != -1
                ]
            )
            - null_topic_vector
        )
        candidate_subtopics = np.asarray(
            [
                subtopics[subtopic_index]
                for subtopic_index in cluster_subtopic_indices
                if subtopic_index != -1
            ]
        )

        if candidate_subtopic_centroids.shape[0] >= n_subtopics:
            _, chosen_subtopics = selector.fit_transform(
                candidate_subtopic_centroids, y=candidate_subtopics
            )
        else:
            chosen_subtopics = candidate_subtopics

        result.append(chosen_subtopics.tolist())

    return result


def central_subtopics_from_all_subtopics(
    cluster_label_vector: np.ndarray,
    subtopics: List[str],
    subtopic_label_vector: np.ndarray,
    subtopic_vectors: Optional[np.ndarray] = None,
    embedding_model: Optional[TextEmbedderProtocol] = None,
    n_subtopics: int = 64,
    diversify_alpha: float = 1.0,
    verbose: bool = None,
    show_progress_bar: bool = None,
) -> List[List[str]]:
    if subtopic_vectors is None:
        if embedding_model is None:
            raise ValueError(
                "Either subtopic_embeddings or embedding_model must be provided"
            )
        subtopic_vectors = subtopic_embeddings(subtopics, embedding_model)

    # Handle verbose parameters
    show_progress_bar_val, _ = handle_verbose_params(
        verbose=verbose, show_progress_bar=show_progress_bar, default_verbose=False
    )

    result = []
    for cluster_num in tqdm(
        range(cluster_label_vector.max() + 1),
        desc="Selecting subtopics by topic name",
        disable=not show_progress_bar_val,
    ):
        subtopics_in_cluster = np.unique(
            subtopic_label_vector[cluster_label_vector == cluster_num]
        )
        cluster_centroid = np.mean(subtopic_vectors[subtopics_in_cluster], axis=0)
        # Select the central subtopics as the closest samples to the centroid
        base_distances = pairwise_distances(
            cluster_centroid.reshape(1, -1),
            subtopic_vectors,
            metric="cosine",
        )
        base_order = np.argsort(base_distances.flatten())

        chosen_subtopics = [subtopics[x] for x in base_order[: n_subtopics**2]]
        chosen_vectors = subtopic_vectors[base_order[: n_subtopics**2]]
        chosen_indices = diversify(
            cluster_centroid,
            chosen_vectors,
            n_subtopics,
            max_alpha=diversify_alpha,
        )[:n_subtopics]
        chosen_subtopics = [chosen_subtopics[j] for j in chosen_indices]

        result.append(chosen_subtopics)

    return result


def information_weighted_subtopics(
    cluster_label_vector: np.ndarray,
    subtopics: List[str],
    subtopic_label_vector: np.ndarray,
    subtopic_vectors: Optional[np.ndarray] = None,
    embedding_model: Optional[TextEmbedderProtocol] = None,
    n_subtopics: int = 64,
    diversify_alpha: float = 1.0,
    prior_strength: float = 0.1,
    weight_power: float = 2.0,
    n_dictionary_vectors: int = 512,
    coding_transform_alpha: float = 0.1,
    n_jobs=-1,
    verbose: bool = None,
    show_progress_bar: bool = None,
) -> List[List[str]]:
    if subtopic_vectors is None:
        if embedding_model is None:
            raise ValueError(
                "Either subtopic_embeddings or embedding_model must be provided"
            )
        subtopic_vectors = subtopic_embeddings(subtopics, embedding_model)

    meta_cluster_label_vector = np.full(subtopic_vectors.shape[0], -1, dtype=np.int32)
    for cluster_num in range(cluster_label_vector.max() + 1):
        subtopics_of_cluster = np.unique(
            subtopic_label_vector[cluster_label_vector == cluster_num]
        )
        subtopics_of_cluster = subtopics_of_cluster[subtopics_of_cluster != -1]
        meta_cluster_label_vector[subtopics_of_cluster] = cluster_num

    sparse_coder = DictionaryLearning(
        n_components=min(n_dictionary_vectors, subtopic_vectors.shape[0] // 2),
        positive_code=True,
        fit_algorithm="cd",
        transform_algorithm="lasso_cd",
        n_jobs=n_jobs,
        max_iter=100_000,
        transform_alpha=coding_transform_alpha,
        transform_max_iter=100_000,
    ).fit(subtopic_vectors)
    sparse_coding = scipy.sparse.csr_array(sparse_coder.transform(subtopic_vectors))
    weighted_sparse_coding = (
        InformationWeightTransformer(
            prior_strength=prior_strength, weight_power=weight_power
        )
        .fit_transform(sparse_coding, y=meta_cluster_label_vector)
        .toarray()
    )
    scores = weighted_sparse_coding.sum(axis=1)

    # Handle verbose parameters
    show_progress_bar_val, _ = handle_verbose_params(
        verbose=verbose, show_progress_bar=show_progress_bar, default_verbose=False
    )

    result = []
    for cluster_num in tqdm(
        range(meta_cluster_label_vector.max() + 1),
        desc="Selecting informative subtopics",
        disable=not show_progress_bar_val,
    ):
        candidate_subtopics = [
            subtopics[x] for x in np.where(meta_cluster_label_vector == cluster_num)[0]
        ]
        candidate_scores = scores[meta_cluster_label_vector == cluster_num]
        candidate_vectors = subtopic_vectors[meta_cluster_label_vector == cluster_num]

        cluster_centroid = np.average(
            candidate_vectors, weights=candidate_scores, axis=0
        )

        if sum(candidate_scores) == 0:
            result.append(["No notable keyphrases"])
            continue

        chosen_indices = np.argsort(candidate_scores)[-(n_subtopics**2) :][::-1]

        # Map the indices back to the original vocabulary
        chosen_subtopics = [candidate_subtopics[j] for j in chosen_indices]
        chosen_vectors = candidate_vectors[chosen_indices]
        chosen_indices = diversify(
            cluster_centroid,
            chosen_vectors,
            n_subtopics,
            max_alpha=diversify_alpha,
        )[:n_subtopics]
        chosen_subtopics = [chosen_subtopics[j] for j in chosen_indices]

        result.append(chosen_subtopics)

    return result


def central_summary_subtopics(
    cluster_label_vector: np.ndarray,
    subtopics: List[str],
    summaries: List[str],
    explanations: List[str],
    subtopic_label_vector: np.ndarray,
    subtopic_vectors: Optional[np.ndarray] = None,
    embedding_model: Optional[TextEmbedderProtocol] = None,
    n_subtopics: int = 64,
    diversify_alpha: float = 1.0,
    verbose: bool = None,
    show_progress_bar: bool = None,
) -> List[List[str]]:
    if subtopic_vectors is None:
        if embedding_model is None:
            raise ValueError(
                "Either subtopic_embeddings or embedding_model must be provided"
            )
        subtopic_vectors = subtopic_embeddings(subtopics, embedding_model)

    central_vector = np.mean(subtopic_vectors, axis=0)

    # Handle verbose parameters
    show_progress_bar_val, _ = handle_verbose_params(
        verbose=verbose, show_progress_bar=show_progress_bar, default_verbose=False
    )

    result = []
    for cluster_num in tqdm(
        range(cluster_label_vector.max() + 1),
        desc="Selecting central subtopics",
        disable=not show_progress_bar_val,
        leave=False,
        unit="cluster",
        position=1,
    ):
        subtopic_label_vector_for_cluster = subtopic_label_vector[
            cluster_label_vector == cluster_num
        ]
        cluster_subtopic_indices = np.unique(subtopic_label_vector_for_cluster)
        candidate_subtopic_centroids = (
            np.asarray(
                [
                    subtopic_vectors[subtopic_index]
                    for subtopic_index in cluster_subtopic_indices
                    if subtopic_index != -1
                ]
            )
            - central_vector
        )
        candidate_subtopics = np.asarray(
            [
                [
                    subtopics[subtopic_index],
                    summaries[subtopic_index],
                    explanations[subtopic_index],
                ]
                for subtopic_index in cluster_subtopic_indices
                if subtopic_index != -1
            ]
        )
        cluster_centroid = np.mean(candidate_subtopic_centroids, axis=0)
        candidate_subtopic_indices = np.argsort(
            np.squeeze(
                pairwise_distances(
                    [cluster_centroid], candidate_subtopic_centroids, metric="cosine"
                )
            )
        )[: n_subtopics**2]
        sorted_candidate_subtopic_centroids = candidate_subtopic_centroids[
            candidate_subtopic_indices
        ]
        sorted_candidate_subtopics = [
            candidate_subtopics[i] for i in candidate_subtopic_indices
        ]
        chosen_subtopic_indices = diversify(
            cluster_centroid,
            sorted_candidate_subtopic_centroids,
            n_subtopics,
            max_alpha=diversify_alpha,
        )[:n_subtopics]
        chosen_subtopics = [
            sorted_candidate_subtopics[j] for j in chosen_subtopic_indices
        ]

        result.append(
            "\n--\n".join(["\n".join(subtopic) for subtopic in chosen_subtopics])
        )

    return result


def submodular_summary_subtopics(
    cluster_label_vector: np.ndarray,
    subtopics: List[str],
    summaries: List[str],
    explanations: List[str],
    subtopic_label_vector: np.ndarray,
    subtopic_vectors: Optional[np.ndarray] = None,
    embedding_model: Optional[TextEmbedderProtocol] = None,
    n_subtopics: int = 64,
    submodular_function: str = "facility_location",
    verbose: bool = None,
    show_progress_bar: bool = None,
) -> List[List[str]]:
    if subtopic_vectors is None:
        if embedding_model is None:
            raise ValueError(
                "Either subtopic_embeddings or embedding_model must be provided"
            )
        subtopic_vectors = subtopic_embeddings(subtopics, embedding_model)

    null_topic_vector = np.mean(subtopic_vectors, axis=0)
    if submodular_function == "facility_location":
        selector = FacilityLocationSelection(
            n_subtopics, metric="cosine", optimizer="lazy"
        )
    elif submodular_function == "saturated_coverage":
        selector = SaturatedCoverageSelection(
            n_subtopics, metric="cosine", optimizer="lazy"
        )
    else:
        raise ValueError(
            f"selection_function={submodular_function} is not a valid selection. Please choose one of (facility_location,saturated_coverage)"
        )

    # Handle verbose parameters
    show_progress_bar_val, _ = handle_verbose_params(
        verbose=verbose, show_progress_bar=show_progress_bar, default_verbose=False
    )

    result = []
    for cluster_num in tqdm(
        range(cluster_label_vector.max() + 1),
        desc=f"Selecting {submodular_function} subtopics",
        disable=not show_progress_bar_val,
        leave=False,
        unit="cluster",
        position=1,
    ):
        subtopic_label_vector_for_cluster = subtopic_label_vector[
            cluster_label_vector == cluster_num
        ]
        cluster_subtopic_indices = np.unique(subtopic_label_vector_for_cluster)
        candidate_subtopic_centroids = (
            np.asarray(
                [
                    subtopic_vectors[subtopic_index]
                    for subtopic_index in cluster_subtopic_indices
                    if subtopic_index != -1
                ]
            )
            - null_topic_vector
        )
        candidate_subtopics = np.asarray(
            [
                [
                    subtopics[subtopic_index],
                    summaries[subtopic_index],
                    explanations[subtopic_index],
                ]
                for subtopic_index in cluster_subtopic_indices
                if subtopic_index != -1
            ]
        )

        if candidate_subtopic_centroids.shape[0] >= n_subtopics:
            _, chosen_subtopics = selector.fit_transform(
                candidate_subtopic_centroids, y=candidate_subtopics
            )
        else:
            chosen_subtopics = candidate_subtopics

        result.append(
            "\n--\n".join(["\n".join(subtopic) for subtopic in chosen_subtopics])
        )

    return result


def information_weighted_summary_subtopics(
    cluster_label_vector: np.ndarray,
    subtopics: List[str],
    summaries: List[str],
    explanations: List[str],
    subtopic_label_vector: np.ndarray,
    subtopic_vectors: Optional[np.ndarray] = None,
    embedding_model: Optional[TextEmbedderProtocol] = None,
    n_subtopics: int = 64,
    diversify_alpha: float = 1.0,
    prior_strength: float = 0.1,
    weight_power: float = 2.0,
    n_dictionary_vectors: int = 512,
    coding_transform_alpha: float = 0.1,
    n_jobs=-1,
    verbose: bool = None,
    show_progress_bar: bool = None,
) -> List[List[str]]:
    if subtopic_vectors is None:
        if embedding_model is None:
            raise ValueError(
                "Either subtopic_embeddings or embedding_model must be provided"
            )
        subtopic_vectors = subtopic_embeddings(subtopics, embedding_model)

    meta_cluster_label_vector = np.full(subtopic_vectors.shape[0], -1, dtype=np.int32)
    for cluster_num in range(cluster_label_vector.max() + 1):
        subtopics_of_cluster = np.unique(
            subtopic_label_vector[cluster_label_vector == cluster_num]
        )
        subtopics_of_cluster = subtopics_of_cluster[subtopics_of_cluster != -1]
        meta_cluster_label_vector[subtopics_of_cluster] = cluster_num

    sparse_coder = DictionaryLearning(
        n_components=min(n_dictionary_vectors, subtopic_vectors.shape[0] // 2),
        positive_code=True,
        fit_algorithm="cd",
        transform_algorithm="lasso_cd",
        n_jobs=n_jobs,
        max_iter=100_000,
        transform_alpha=coding_transform_alpha,
        transform_max_iter=100_000,
    ).fit(subtopic_vectors)
    sparse_coding = scipy.sparse.csr_array(sparse_coder.transform(subtopic_vectors))
    weighted_sparse_coding = (
        InformationWeightTransformer(
            prior_strength=prior_strength, weight_power=weight_power
        )
        .fit_transform(sparse_coding, y=meta_cluster_label_vector)
        .toarray()
    )
    scores = weighted_sparse_coding.sum(axis=1)

    # Handle verbose parameters
    show_progress_bar_val, _ = handle_verbose_params(
        verbose=verbose, show_progress_bar=show_progress_bar, default_verbose=False
    )

    result = []
    for cluster_num in tqdm(
        range(meta_cluster_label_vector.max() + 1),
        desc="Selecting informative subtopics",
        disable=not show_progress_bar_val,
    ):
        candidate_subtopics = [
            [subtopics[x], summaries[x], explanations[x]]
            for x in np.where(meta_cluster_label_vector == cluster_num)[0]
        ]
        candidate_scores = scores[meta_cluster_label_vector == cluster_num]
        candidate_vectors = subtopic_vectors[meta_cluster_label_vector == cluster_num]

        cluster_centroid = np.average(
            candidate_vectors, weights=candidate_scores, axis=0
        )

        if sum(candidate_scores) == 0:
            result.append(["No notable keyphrases"])
            continue

        chosen_indices = np.argsort(candidate_scores)[-(n_subtopics**2) :][::-1]

        # Map the indices back to the original vocabulary
        chosen_subtopics = [candidate_subtopics[j] for j in chosen_indices]
        chosen_vectors = candidate_vectors[chosen_indices]
        chosen_indices = diversify(
            cluster_centroid,
            chosen_vectors,
            n_subtopics,
            max_alpha=diversify_alpha,
        )[:n_subtopics]
        chosen_subtopics = [chosen_subtopics[j] for j in chosen_indices]

        result.append(
            "\n--\n".join(["\n".join(subtopic) for subtopic in chosen_subtopics])
        )

    return result


class TopicNameEmbeddingAnnotator:
    """Embed the topic name of each node, with one encode call per layer.

    Parameters
    ----------
    embedding_model : TextEmbedderProtocol
        The model that embeds the names.

    Attributes
    ----------
    inputs : tuple of str
        ``topic_names``.
    outputs : tuple of str
        ``topic_name_embeddings``.

    Notes
    -----
    Each layer is embedded as a unit, so a layer with a node whose name is missing
    or failed is not embedded at all.
    """

    inputs = ("topic_names",)
    outputs = ("topic_name_embeddings",)
    algorithm_input_type = {"topic_names": LayerInput()}
    algorithm_output_type = "multinode"

    def __init__(self, embedding_model: TextEmbedderProtocol):
        self.embedding_model = embedding_model

    def annotate(self, layer_nodes, *, topic_names):
        """Embed the names of the selected nodes of one layer.

        Returns
        -------
        dict
            ``{"topic_name_embeddings": {node: vector}}``.
        """
        nodes = list(layer_nodes)
        vectors = self.embedding_model.encode(
            [topic_names[node] for node in nodes], show_progress_bar=False
        )
        return {"topic_name_embeddings": dict(zip(nodes, vectors))}


def _layer_zero_subtopics(subtopic_names, subtopic_vectors, null_topic):
    """The names and centred vectors of the layer 0 descendants, in cluster order."""
    nodes = sorted(node for node in subtopic_names if node.layer == 0)
    names = [subtopic_names[node] for node in nodes]
    vectors = np.asarray([subtopic_vectors[node] for node in nodes])
    if null_topic is not None and len(nodes):
        vectors = vectors - null_topic
    return names, vectors


class CentralSubtopicAnnotator:
    """Select a node's subtopics from the names of its layer 0 descendants.

    The names are ranked by cosine distance to their mean, and a diverse subset of
    the closest is kept. This is the annotator form of ``central_subtopics``, except
    that the candidates are the node's layer 0 descendants in the cluster tree
    rather than the layer 0 clusters that share objects with it. The two agree when
    the clusterer's layers nest.

    Parameters
    ----------
    n_subtopics : int, optional
        The number of subtopics to select for each node, by default 64.
    diversify_alpha : float, optional
        The largest alpha to try when diversifying the selection, by default 1.0.
    null_topic : np.ndarray, optional
        Subtracted from the name embeddings before any distance is computed. This
        is typically the mean embedding of all layer 0 names. If None, nothing is
        subtracted.

    Attributes
    ----------
    inputs : tuple of str
        ``subtopic_names`` and ``subtopic_vectors``, read from the ``topic_names``
        and ``topic_name_embeddings`` of each node's descendants.
    outputs : tuple of str
        ``subtopics``.
    """

    inputs = ("subtopic_names", "subtopic_vectors")
    outputs = ("subtopics",)
    algorithm_input_type = {
        "subtopic_names": DescendantsInput(source="topic_names"),
        "subtopic_vectors": DescendantsInput(source="topic_name_embeddings"),
    }
    algorithm_output_type = "node"

    def __init__(
        self,
        n_subtopics: int = 64,
        diversify_alpha: float = 1.0,
        null_topic: Optional[np.ndarray] = None,
    ):
        self.n_subtopics = n_subtopics
        self.diversify_alpha = diversify_alpha
        self.null_topic = null_topic

    def annotate(self, node, *, subtopic_names, subtopic_vectors):
        """Select subtopics for one node.

        Returns
        -------
        dict
            ``{"subtopics": list of str}``, empty for a node with no named layer 0
            descendants.
        """
        names, vectors = _layer_zero_subtopics(
            subtopic_names, subtopic_vectors, self.null_topic
        )
        if not names:
            return {"subtopics": []}
        centroid = np.mean(vectors, axis=0)
        order = np.argsort(
            np.squeeze(pairwise_distances([centroid], vectors, metric="cosine"), 0)
        )[: self.n_subtopics**2]
        chosen = diversify(
            centroid,
            vectors[order],
            self.n_subtopics,
            max_alpha=self.diversify_alpha,
        )[: self.n_subtopics]
        return {"subtopics": [names[order[j]] for j in chosen]}


class SubmodularSubtopicAnnotator:
    """Select a node's subtopics from its layer 0 descendants by submodular selection.

    This is the annotator form of ``submodular_subtopics``, except that the
    candidates are the node's layer 0 descendants in the cluster tree rather than
    the layer 0 clusters that share objects with it. The two agree when the
    clusterer's layers nest. A node with fewer candidates than ``n_subtopics`` gets
    all of them.

    Parameters
    ----------
    n_subtopics : int, optional
        The number of subtopics to select for each node, by default 64.
    null_topic : np.ndarray, optional
        Subtracted from the name embeddings before selection. This is typically the
        mean embedding of all layer 0 names. If None, nothing is subtracted.
    submodular_function : {"facility_location", "saturated_coverage"}, optional
        The submodular function to optimise, by default "facility_location".

    Raises
    ------
    ValueError
        If ``submodular_function`` is not one of the supported values.
    """

    inputs = ("subtopic_names", "subtopic_vectors")
    outputs = ("subtopics",)
    algorithm_input_type = {
        "subtopic_names": DescendantsInput(source="topic_names"),
        "subtopic_vectors": DescendantsInput(source="topic_name_embeddings"),
    }
    algorithm_output_type = "node"

    def __init__(
        self,
        n_subtopics: int = 64,
        null_topic: Optional[np.ndarray] = None,
        submodular_function: str = "facility_location",
    ):
        self.n_subtopics = n_subtopics
        self.null_topic = null_topic
        self.submodular_function = submodular_function
        if submodular_function == "facility_location":
            self.selector = FacilityLocationSelection(
                n_subtopics, metric="cosine", optimizer="lazy"
            )
        elif submodular_function == "saturated_coverage":
            self.selector = SaturatedCoverageSelection(
                n_subtopics, metric="cosine", optimizer="lazy"
            )
        else:
            raise ValueError(
                f"selection_function={submodular_function} is not a valid selection. Please choose one of (facility_location,saturated_coverage)"
            )

    def annotate(self, node, *, subtopic_names, subtopic_vectors):
        """Select subtopics for one node.

        Returns
        -------
        dict
            ``{"subtopics": list of str}``, empty for a node with no named layer 0
            descendants.
        """
        names, vectors = _layer_zero_subtopics(
            subtopic_names, subtopic_vectors, self.null_topic
        )
        if len(names) < self.n_subtopics:
            return {"subtopics": names}
        _, chosen = self.selector.fit_transform(vectors, y=np.asarray(names))
        return {"subtopics": chosen.tolist()}
