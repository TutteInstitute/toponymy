from toponymy.subtopics import (
    central_subtopics,
    central_subtopics_from_all_subtopics,
    information_weighted_subtopics,
)
from toponymy.clustering import centroids_from_labels

from pathlib import Path
import numpy as np
import pytest
import json
import sentence_transformers


def cosine_similarity(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))


@pytest.mark.parametrize("n_subtopics", [2, 4])
@pytest.mark.parametrize("diversify_alpha", [0.0, 0.5, 1.0])
def test_central_subtopics_result_sizes(
    cluster_label_vector, all_subtopics, subtopic_label_vector, 
    subtopic_centroid_vectors, n_subtopics, diversify_alpha
):
    central_subtopics_results = central_subtopics(
        cluster_label_vector,
        all_subtopics,
        subtopic_label_vector,
        subtopic_centroid_vectors,
        diversify_alpha=diversify_alpha,
        n_subtopics=n_subtopics,
    )
    assert len(central_subtopics_results) == len(np.unique(cluster_label_vector))
    if diversify_alpha == 1.0:
        print(central_subtopics_results)
    assert (
        all([len(x) == n_subtopics for x in central_subtopics_results])
        and central_subtopics_results
    )
    assert all([len(set(x)) == n_subtopics for x in central_subtopics_results])


def test_central_subtopics(
    cluster_label_vector, all_subtopics, subtopic_label_vector, 
    subtopic_centroid_vectors, cluster_centroid_vectors, subtopics
):
    central_results = central_subtopics(
        cluster_label_vector,
        all_subtopics,
        subtopic_label_vector,
        subtopic_centroid_vectors,
        diversify_alpha=0.0,
        n_subtopics=3,
    )
    null_topic = np.mean(subtopic_centroid_vectors, axis=0)
    for cluster_num, subtopics_result in enumerate(central_results):
        assert subtopics_result == sorted(
            subtopics_result,
            key=lambda x: 1.0
            - (
                cosine_similarity(
                    subtopic_centroid_vectors[all_subtopics.index(x)] - null_topic,
                    cluster_centroid_vectors[cluster_num] - null_topic,
                )
            ),
        )
        subtopics_in_cluster = subtopics[cluster_num]
        worst_subtopic_match_similarity = min(
            [
                cosine_similarity(
                    subtopic_centroid_vectors[all_subtopics.index(x)] - null_topic,
                    cluster_centroid_vectors[cluster_num] - null_topic,
                )
                for x in subtopics_result
            ]
        )
        assert worst_subtopic_match_similarity >= max(
            [
                cosine_similarity(
                    subtopic_centroid_vectors[all_subtopics.index(other_topic)] - null_topic,
                    cluster_centroid_vectors[cluster_num] - null_topic,
                )
                for other_topic in subtopics_in_cluster
                if other_topic not in subtopics_result
            ]
        )


@pytest.mark.parametrize("n_subtopics", [2, 4])
@pytest.mark.parametrize("diversify_alpha", [0.0, 0.5, 1.0])
def test_central_subtopics_from_all_subtopics_result_sizes(
    cluster_label_vector, all_subtopics, subtopic_label_vector, 
    subtopic_vectors, n_subtopics, diversify_alpha
):
    central_subtopics_results = central_subtopics_from_all_subtopics(
        cluster_label_vector,
        all_subtopics,
        subtopic_label_vector,
        subtopic_vectors,
        diversify_alpha=diversify_alpha,
        n_subtopics=n_subtopics,
    )
    assert len(central_subtopics_results) == len(np.unique(cluster_label_vector))
    if diversify_alpha == 1.0:
        print(central_subtopics_results)
    assert (
        all([len(x) == n_subtopics for x in central_subtopics_results])
        and central_subtopics_results
    )
    assert all([len(set(x)) == n_subtopics for x in central_subtopics_results])


def test_central_subtopics_from_all_subtopics(
    cluster_label_vector, all_subtopics, subtopic_label_vector, 
    subtopic_vectors, subtopics
):
    central_results = central_subtopics_from_all_subtopics(
        cluster_label_vector,
        all_subtopics,
        subtopic_label_vector,
        subtopic_vectors,
        diversify_alpha=0.0,
        n_subtopics=3,
    )
    for cluster_num, subtopics_result in enumerate(central_results):
        subtopics_in_cluster = subtopics[cluster_num]
        subtopic_centroid = np.mean(
            subtopic_vectors[[all_subtopics.index(subtopic) for subtopic in subtopics_in_cluster]], axis=0
        )
        assert subtopics_result == sorted(
            subtopics_result,
            key=lambda x: 1.0
            - (
                subtopic_vectors[all_subtopics.index(x)]
                @ subtopic_centroid
            ),
        )
        worst_keyphrase_match_similarity = min(
            [
                subtopic_vectors[all_subtopics.index(x)]
                @ subtopic_centroid
                for x in subtopics_result
            ]
        )
        assert worst_keyphrase_match_similarity >= max(
            [
                subtopic_vectors[all_subtopics.index(other_topic)]
                @ subtopic_centroid
                for other_topic in subtopics_in_cluster
                if other_topic not in subtopics_result
            ]
        )


def test_central_subtopics_from_all_subtopics_w_embeddder(
    cluster_label_vector, all_subtopics, subtopic_label_vector, 
    embedder, subtopic_vectors, subtopics
):
    central_results = central_subtopics_from_all_subtopics(
        cluster_label_vector,
        all_subtopics,
        subtopic_label_vector,
        embedding_model=embedder,
        diversify_alpha=0.0,
        n_subtopics=3,
    )
    for cluster_num, subtopics_result in enumerate(central_results):
        subtopics_in_cluster = subtopics[cluster_num]
        subtopic_centroid = np.mean(
            subtopic_vectors[[all_subtopics.index(subtopic) for subtopic in subtopics_in_cluster]], axis=0
        )
        assert subtopics_result == sorted(
            subtopics_result,
            key=lambda x: 1.0
            - (
                subtopic_vectors[all_subtopics.index(x)]
                @ subtopic_centroid
            ),
        )

        worst_keyphrase_match_similarity = min(
            [
                subtopic_vectors[all_subtopics.index(x)]
                @ subtopic_centroid
                for x in subtopics_result
            ]
        )
        assert worst_keyphrase_match_similarity >= max(
            [
                subtopic_vectors[all_subtopics.index(other_topic)]
                @ subtopic_centroid
                for other_topic in subtopics_in_cluster
                if other_topic not in subtopics_result
            ]
        )


def test_central_subtopics_from_all_subtopics_bad_params(
    cluster_label_vector, all_subtopics, subtopic_label_vector
):
    with pytest.raises(ValueError):
        central_subtopics_from_all_subtopics(
            cluster_label_vector,
            all_subtopics,
            subtopic_label_vector,
            diversify_alpha=0.0,
            n_subtopics=3,
        )


@pytest.mark.parametrize("n_subtopics", [1, 3])
@pytest.mark.parametrize("diversify_alpha", [0.0, 0.5, 1.0])
def test_information_weighted_subtopics_result_sizes(
    cluster_label_vector, all_subtopics, subtopic_label_vector, 
    subtopic_vectors, n_subtopics, diversify_alpha
):
    iwt_results = information_weighted_subtopics(
        cluster_label_vector,
        all_subtopics,
        subtopic_label_vector,
        subtopic_vectors,
        diversify_alpha=diversify_alpha,
        n_subtopics=n_subtopics,
        n_dictionary_vectors=5,
        coding_transform_alpha=0.0,
        n_jobs=None,
    )
    print(iwt_results)
    assert len(iwt_results) == len(np.unique(cluster_label_vector))
    assert all([len(x) == n_subtopics for x in iwt_results])
    assert all([len(set(x)) == n_subtopics for x in iwt_results])


def test_information_weighted_subtopics(
    cluster_label_vector, all_subtopics, subtopic_label_vector, 
    subtopic_vectors
):
    iwt_results = information_weighted_subtopics(
        cluster_label_vector,
        all_subtopics,
        subtopic_label_vector,
        subtopic_vectors,
        diversify_alpha=0.0,
        n_subtopics=3,
        n_dictionary_vectors=16,
        coding_transform_alpha=0.1,
        n_jobs=None,
    )
    # How do we test from here -- we aren't even sure what this does, so ...
    pass


def test_information_weighted_subtopics_embedder(
    cluster_label_vector, all_subtopics, subtopic_label_vector, 
    embedder
):
    iwt_results = information_weighted_subtopics(
        cluster_label_vector,
        all_subtopics,
        subtopic_label_vector,
        embedding_model=embedder,
        diversify_alpha=0.0,
        n_subtopics=3,
        n_dictionary_vectors=16,
        coding_transform_alpha=0.1,
        n_jobs=None,
    )
    # How do we test from here -- we aren't even sure what this does, so ...
    pass


def test_information_weighted_subtopics_bad_params(
    cluster_label_vector, all_subtopics, subtopic_label_vector
):
    with pytest.raises(ValueError):
        information_weighted_subtopics(
            cluster_label_vector,
            all_subtopics,
            subtopic_label_vector,
        )
