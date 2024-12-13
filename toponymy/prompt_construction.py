import numpy as np

from toponymy.templates import PROMPT_TEMPLATES, SUMMARY_KINDS

from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sentence_transformers import SentenceTransformer

from typing import List, Optional, Any, Tuple

COSINE_DISTANCE_EPSILON = 1e-6


def find_threshold_for_max_cluster_size(
    distances: np.ndarray, max_cluster_size: int = 4, max_distance: float = 0.2
) -> float:
    n_samples = distances.shape[0]
    clustering = AgglomerativeClustering(
        n_clusters=2,
        distance_threshold=None,
        compute_full_tree=True,
        compute_distances=True,
        metric="precomputed",
        linkage="complete",
    )
    clustering.fit(distances)
    cluster_sizes = defaultdict(lambda: 1)
    merge_distances = clustering.distances_

    for i, (cluster1, cluster2) in enumerate(clustering.children_):
        new_size = cluster_sizes[cluster1] + cluster_sizes[cluster2]
        if (
            new_size > max_cluster_size
            and merge_distances[i - 1] > COSINE_DISTANCE_EPSILON
        ):
            return merge_distances[i - 1] if i > 0 else merge_distances[0]

        if merge_distances[i] > max_distance:
            return merge_distances[i]

        cluster_sizes[n_samples + i] = new_size
        del cluster_sizes[cluster1]
        del cluster_sizes[cluster2]

    return merge_distances[-1]


def cluster_topic_names_for_renaming(
    topic_names: List[str],
    topic_name_embeddings: Optional[np.ndarray] = None,
    embedding_model: Optional[SentenceTransformer] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if topic_name_embeddings is None:
        if embedding_model is None:
            raise ValueError(
                "Either topic_name_embeddings or embedding_model must be provided."
            )
        topic_name_embeddings = embedding_model.encode(
            topic_names, show_progress_bar=True
        )
    distances = pairwise_distances(topic_name_embeddings, metric="cosine")
    threshold = find_threshold_for_max_cluster_size(distances)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        compute_full_tree=True,
        compute_distances=True,
        metric="precomputed",
        linkage="complete",
    )
    clustering.fit(distances)
    cluster_sizes = np.bincount(clustering.labels_)
    clusters_for_renaming = np.where(cluster_sizes >= 2)[0]

    return clusters_for_renaming, clustering.labels_


def distinguish_topic_names_prompt(
    topic_indices: np.ndarray,
    layer_id: int,
    all_topic_names: List[List[str]],
    exemplar_texts: List[List[str]],
    keyphrases: List[List[str]],
    subtopics: Optional[List[List[List[str]]]],
    cluster_tree: Optional[dict],
    object_description: str,
    corpus_description: str,
    summary_kind: str,
    max_num_keyphrases: int = 32,
    max_num_subtopics: int = 16,
    max_num_exemplars: int = 128,
) -> str:
    attempted_topic_names = [all_topic_names[layer_id][x] for x in topic_indices]
    unique_topic_names = list(set(attempted_topic_names))
    if len(unique_topic_names) == 1:
        larger_topic = unique_topic_names[0]
    else:
        unique_topic_names = unique_topic_names[:3]
        larger_topic = (
            ", ".join(unique_topic_names[:-1]) + " and " + unique_topic_names[-1]
        )

    keyphrases_per_topic = [keyphrases[i][:max_num_keyphrases] for i in topic_indices]
    exemplar_texts_per_topic = [
        exemplar_texts[i][:max_num_exemplars] for i in topic_indices
    ]
    if subtopics is not None and cluster_tree is not None:
        tree_subtopics_per_topic = [cluster_tree[(layer_id, x)] for x in topic_indices]
        major_subtopics_per_topic = [
            [
                all_topic_names[layer_id - 1][a[1]]
                for a in tree_subtopics
                if a[0] == layer_id - 1
            ]
            for tree_subtopics in tree_subtopics_per_topic
        ]
        minor_subtopics_per_topic = [
            [
                all_topic_names[layer_id - 2][a[1]]
                for a in tree_subtopics
                if a[0] == layer_id - 2
            ]
            for tree_subtopics in tree_subtopics_per_topic
        ]
        other_subtopics_per_topic = [
            subtopics[layer_id][cluster_id][:max_num_subtopics] for cluster_id in topic_indices
        ]
    else:
        major_subtopics_per_topic = [False] * len(topic_indices)
        minor_subtopics_per_topic = [False] * len(topic_indices)
        other_subtopics_per_topic = [False] * len(topic_indices)

    prompt = PROMPT_TEMPLATES["distinguish"].render(
        larger_topic=larger_topic,
        document_type=object_description,
        corpus_description=corpus_description,
        attempted_topic_names=attempted_topic_names,
        cluster_keywords=keyphrases_per_topic,
        cluster_subtopics={
            "major": major_subtopics_per_topic,
            "minor": minor_subtopics_per_topic,
            "misc": other_subtopics_per_topic,
        },
        cluster_sentences=exemplar_texts_per_topic,
        summary_kind=summary_kind,
    )

    return prompt

def topic_name_prompt(
    topic_index: np.ndarray,
    layer_id: int,
    all_topic_names: List[List[str]],
    exemplar_texts: List[List[str]],
    keyphrases: List[List[str]],
    subtopics: Optional[List[List[List[str]]]],
    cluster_tree: Optional[dict],
    object_description: str,
    corpus_description: str,
    summary_kind: str,
    max_num_keyphrases: int = 32,
    max_num_subtopics: int = 16,
    max_num_exemplars: int = 128,
):
    if subtopics is not None and cluster_tree is not None:
        tree_subtopics = cluster_tree[(layer_id, topic_index)]
        # Subtopics one layer down are major subtopics; two layers down are minor
        major_subtopics = [all_topic_names[x[0]][x[1]] for x in tree_subtopics if x[0] == layer_id - 1]
        minor_subtopics = [all_topic_names[x[0]][x[1]] for x in tree_subtopics if x[0] == layer_id - 2]

        if len(major_subtopics) <= 1:
            major_subtopics = major_subtopics + minor_subtopics
            minor_subtopics = [all_topic_names[x[0]][x[1]] for x in tree_subtopics if x[0] < layer_id - 2]

        other_subtopics = subtopics[layer_id][topic_index][:max_num_subtopics]

    prompt = PROMPT_TEMPLATES["layer"].render(
        document_type=object_description,
        corpus_description=corpus_description,
        cluster_keywords=keyphrases[topic_index][:max_num_keyphrases],
        cluster_subtopics={
            "major": major_subtopics,
            "minor": minor_subtopics,
            "misc": other_subtopics,
        },
        cluster_sentences=exemplar_texts[topic_index][:max_num_exemplars],
        summary_kind=summary_kind,
    )

    return prompt