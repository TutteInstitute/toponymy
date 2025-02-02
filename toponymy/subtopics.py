import numpy as np
import scipy.sparse

from typing import List, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import DictionaryLearning

from vectorizers.transformers import InformationWeightTransformer

from toponymy.utility_functions import diversify_max_alpha as diversify

from tqdm.auto import tqdm


def subtopic_embeddings(
    subtopics: List[str], embedding_model: SentenceTransformer
) -> np.ndarray:
    return embedding_model.encode(subtopics)


def central_subtopics(
    cluster_label_vector: np.ndarray,
    subtopics: List[str],
    centroid_vectors: np.ndarray,
    subtopic_label_vector: np.ndarray,
    subtopic_vectors: Optional[np.ndarray] = None,
    embedding_model: Optional[SentenceTransformer] = None,
    n_subtopics: int = 64,
    diversify_alpha: float = 1.0,
    show_progress_bar: bool = False,
) -> List[List[str]]:
    if subtopic_vectors is None:
        if embedding_model is None:
            raise ValueError(
                "Either subtopic_embeddings or embedding_model must be provided"
            )
        subtopic_vectors = subtopic_embeddings(subtopics, embedding_model)

    result = []
    for cluster_num in tqdm(
        range(cluster_label_vector.max() + 1),
        desc="Selecting central subtopics",
        disable=not show_progress_bar,
        leave=False,
        unit="cluster",
        position=1,
    ):
        subtopic_label_vector_for_cluster = subtopic_label_vector[
            cluster_label_vector == cluster_num
        ]
        cluster_subtopic_indices = np.unique(subtopic_label_vector_for_cluster)
        cluster_centroid = centroid_vectors[cluster_num]
        candidate_subtopic_centroids = np.asarray(
            [
                subtopic_vectors[subtopic_index]
                for subtopic_index in cluster_subtopic_indices
                if subtopic_index != -1
            ]
        )
        candidate_subtopics = np.asarray(
            [
                subtopics[subtopic_index]
                for subtopic_index in cluster_subtopic_indices
                if subtopic_index != -1
            ]
        )
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


def central_subtopics_from_all_subtopics(
    cluster_label_vector: np.ndarray,
    subtopics: List[str],
    centroid_vectors: np.ndarray,
    subtopic_label_vector: np.ndarray,
    subtopic_vectors: Optional[np.ndarray] = None,
    embedding_model: Optional[SentenceTransformer] = None,
    n_subtopics: int = 64,
    diversify_alpha: float = 1.0,
    show_progress_bar: bool = False,
) -> List[List[str]]:
    if subtopic_vectors is None:
        if embedding_model is None:
            raise ValueError(
                "Either subtopic_embeddings or embedding_model must be provided"
            )
        subtopic_vectors = subtopic_embeddings(subtopics, embedding_model)

    result = []
    for cluster_num in tqdm(
        range(cluster_label_vector.max() + 1),
        desc="Selecting subtopics by topic name",
        disable=not show_progress_bar,
    ):
        # Select the central subtopics as the closest samples to the centroid
        base_distances = pairwise_distances(
            centroid_vectors[cluster_num].reshape(1, -1),
            subtopic_vectors,
            metric="cosine",
        )
        base_order = np.argsort(base_distances.flatten())

        chosen_subtopics = [subtopics[x] for x in base_order[: n_subtopics**2]]
        chosen_vectors = subtopic_vectors[base_order[: n_subtopics**2]]
        chosen_indices = diversify(
            centroid_vectors[cluster_num],
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
    centroid_vectors: np.ndarray,
    subtopic_label_vector: np.ndarray,
    subtopic_vectors: Optional[np.ndarray] = None,
    embedding_model: Optional[SentenceTransformer] = None,
    n_subtopics: int = 64,
    diversify_alpha: float = 1.0,
    prior_strength: float = 0.1,
    weight_power: float = 2.0,
    n_dictionary_vectors: int = 512,
    coding_transform_alpha: float = 0.1,
    n_jobs=-1,
    show_progress_bar: bool = False,
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

    result = []
    for cluster_num in tqdm(
        range(meta_cluster_label_vector.max() + 1),
        desc="Selecting informative subtopics",
        disable=not show_progress_bar,
    ):
        candidate_subtopics = [
            subtopics[x] for x in np.where(meta_cluster_label_vector == cluster_num)[0]
        ]
        candidate_scores = scores[meta_cluster_label_vector == cluster_num]
        candidate_vectors = subtopic_vectors[meta_cluster_label_vector == cluster_num]

        if sum(candidate_scores) == 0:
            result.append(["No notable keyphrases"])
            continue

        chosen_indices = np.argsort(candidate_scores)[-(n_subtopics**2) :][::-1]

        # Map the indices back to the original vocabulary
        chosen_subtopics = [candidate_subtopics[j] for j in chosen_indices]
        chosen_vectors = candidate_vectors[chosen_indices]
        chosen_indices = diversify(
            centroid_vectors[cluster_num],
            chosen_vectors,
            n_subtopics,
            max_alpha=diversify_alpha,
        )[:n_subtopics]
        chosen_subtopics = [chosen_subtopics[j] for j in chosen_indices]

        result.append(chosen_subtopics)

    return result
