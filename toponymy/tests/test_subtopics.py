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

EMBEDDER = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")

SUBTOPIC_OBJECTS = json.load(open(Path(__file__).parent / "subtopic_objects.json", "r"))
ALL_SENTENCES = sum(
    [x["sentences"] for subtopics in SUBTOPIC_OBJECTS for x in subtopics["subtopics"]],
    [],
)
CLUSTER_LABEL_VECTOR = np.arange(5).repeat(25)
SUBTOPIC_LABEL_VECTOR = np.arange(25).repeat(5)
OBJECT_VECTORS = EMBEDDER.encode(ALL_SENTENCES)
CLUSTER_CENTROID_VECTORS = centroids_from_labels(CLUSTER_LABEL_VECTOR, OBJECT_VECTORS)
SUBTOPIC_CENTROID_VECTORS = centroids_from_labels(SUBTOPIC_LABEL_VECTOR, OBJECT_VECTORS)
SUBTOPICS = [[x["subtopic"] for x in topic["subtopics"]] for topic in SUBTOPIC_OBJECTS]
ALL_SUBTOPICS = sum(SUBTOPICS, [])
SUBTOPIC_VECTORS = EMBEDDER.encode(ALL_SUBTOPICS)
# META_CLUSTER_LAYER = np.arange(5).repeat(5)


def cosine_similarity(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))


@pytest.mark.parametrize("n_subtopics", [2, 4])
@pytest.mark.parametrize("diversify_alpha", [0.0, 0.5, 1.0])
def test_central_subtopics_result_sizes(n_subtopics, diversify_alpha):
    central_subtopics_results = central_subtopics(
        CLUSTER_LABEL_VECTOR,
        ALL_SUBTOPICS,
        CLUSTER_CENTROID_VECTORS,
        SUBTOPIC_LABEL_VECTOR,
        SUBTOPIC_CENTROID_VECTORS,
        diversify_alpha=diversify_alpha,
        n_subtopics=n_subtopics,
    )
    assert len(central_subtopics_results) == len(np.unique(CLUSTER_LABEL_VECTOR))
    if diversify_alpha == 1.0:
        print(central_subtopics_results)
    assert (
        all([len(x) == n_subtopics for x in central_subtopics_results])
        and central_subtopics_results
    )
    assert all([len(set(x)) == n_subtopics for x in central_subtopics_results])


def test_central_subtopics():
    central_results = central_subtopics(
        CLUSTER_LABEL_VECTOR,
        ALL_SUBTOPICS,
        CLUSTER_CENTROID_VECTORS,
        SUBTOPIC_LABEL_VECTOR,
        SUBTOPIC_CENTROID_VECTORS,
        diversify_alpha=0.0,
        n_subtopics=3,
    )
    for cluster_num, subtopics in enumerate(central_results):
        assert subtopics == sorted(
            subtopics,
            key=lambda x: 1.0
            - (
                cosine_similarity(
                    SUBTOPIC_CENTROID_VECTORS[ALL_SUBTOPICS.index(x)],
                    CLUSTER_CENTROID_VECTORS[cluster_num],
                )
            ),
        )
        subtopics_in_cluster = SUBTOPICS[cluster_num]
        worst_subtopic_match_similarity = min(
            [
                cosine_similarity(
                    SUBTOPIC_CENTROID_VECTORS[ALL_SUBTOPICS.index(x)],
                    CLUSTER_CENTROID_VECTORS[cluster_num],
                )
                for x in subtopics
            ]
        )
        assert worst_subtopic_match_similarity >= max(
            [
                cosine_similarity(
                    SUBTOPIC_CENTROID_VECTORS[ALL_SUBTOPICS.index(other_topic)],
                    CLUSTER_CENTROID_VECTORS[cluster_num],
                )
                for other_topic in subtopics_in_cluster
                if other_topic not in subtopics
            ]
        )


@pytest.mark.parametrize("n_subtopics", [2, 4])
@pytest.mark.parametrize("diversify_alpha", [0.0, 0.5, 1.0])
def test_central_subtopics_from_all_subtopics_result_sizes(
    n_subtopics, diversify_alpha
):
    central_subtopics_results = central_subtopics_from_all_subtopics(
        CLUSTER_LABEL_VECTOR,
        ALL_SUBTOPICS,
        CLUSTER_CENTROID_VECTORS,
        SUBTOPIC_LABEL_VECTOR,
        SUBTOPIC_VECTORS,
        diversify_alpha=diversify_alpha,
        n_subtopics=n_subtopics,
    )
    assert len(central_subtopics_results) == len(np.unique(CLUSTER_LABEL_VECTOR))
    if diversify_alpha == 1.0:
        print(central_subtopics_results)
    assert (
        all([len(x) == n_subtopics for x in central_subtopics_results])
        and central_subtopics_results
    )
    assert all([len(set(x)) == n_subtopics for x in central_subtopics_results])


def test_central_subtopics_from_all_subtopics():
    central_results = central_subtopics_from_all_subtopics(
        CLUSTER_LABEL_VECTOR,
        ALL_SUBTOPICS,
        CLUSTER_CENTROID_VECTORS,
        SUBTOPIC_LABEL_VECTOR,
        SUBTOPIC_VECTORS,
        diversify_alpha=0.0,
        n_subtopics=3,
    )
    for cluster_num, subtopics in enumerate(central_results):
        assert subtopics == sorted(
            subtopics,
            key=lambda x: 1.0
            - (
                SUBTOPIC_VECTORS[ALL_SUBTOPICS.index(x)]
                @ CLUSTER_CENTROID_VECTORS[cluster_num]
            ),
        )
        subtopics_in_cluster = SUBTOPICS[cluster_num]
        worst_keyphrase_match_similarity = min(
            [
                SUBTOPIC_VECTORS[ALL_SUBTOPICS.index(x)]
                @ CLUSTER_CENTROID_VECTORS[cluster_num]
                for x in subtopics
            ]
        )
        assert worst_keyphrase_match_similarity >= max(
            [
                SUBTOPIC_VECTORS[ALL_SUBTOPICS.index(other_topic)]
                @ CLUSTER_CENTROID_VECTORS[cluster_num]
                for other_topic in subtopics_in_cluster
                if other_topic not in subtopics
            ]
        )


def test_central_subtopics_from_all_subtopics_w_embeddder():
    central_results = central_subtopics_from_all_subtopics(
        CLUSTER_LABEL_VECTOR,
        ALL_SUBTOPICS,
        CLUSTER_CENTROID_VECTORS,
        SUBTOPIC_LABEL_VECTOR,
        embedding_model=EMBEDDER,
        diversify_alpha=0.0,
        n_subtopics=3,
    )
    for cluster_num, subtopics in enumerate(central_results):
        assert subtopics == sorted(
            subtopics,
            key=lambda x: 1.0
            - (
                SUBTOPIC_VECTORS[ALL_SUBTOPICS.index(x)]
                @ CLUSTER_CENTROID_VECTORS[cluster_num]
            ),
        )
        subtopics_in_cluster = SUBTOPICS[cluster_num]
        worst_keyphrase_match_similarity = min(
            [
                SUBTOPIC_VECTORS[ALL_SUBTOPICS.index(x)]
                @ CLUSTER_CENTROID_VECTORS[cluster_num]
                for x in subtopics
            ]
        )
        assert worst_keyphrase_match_similarity >= max(
            [
                SUBTOPIC_VECTORS[ALL_SUBTOPICS.index(other_topic)]
                @ CLUSTER_CENTROID_VECTORS[cluster_num]
                for other_topic in subtopics_in_cluster
                if other_topic not in subtopics
            ]
        )


def test_central_subtopics_from_all_subtopics_bad_params():
    with pytest.raises(ValueError):
        central_subtopics_from_all_subtopics(
            CLUSTER_LABEL_VECTOR,
            ALL_SUBTOPICS,
            CLUSTER_CENTROID_VECTORS,
            SUBTOPIC_LABEL_VECTOR,
            diversify_alpha=0.0,
            n_subtopics=3,
        )


@pytest.mark.parametrize("n_subtopics", [1, 3])
@pytest.mark.parametrize("diversify_alpha", [0.0, 0.5, 1.0])
def test_information_weighted_subtopics_result_sizes(n_subtopics, diversify_alpha):
    iwt_results = information_weighted_subtopics(
        CLUSTER_LABEL_VECTOR,
        ALL_SUBTOPICS,
        CLUSTER_CENTROID_VECTORS,
        SUBTOPIC_LABEL_VECTOR,
        SUBTOPIC_VECTORS,
        diversify_alpha=diversify_alpha,
        n_subtopics=n_subtopics,
        n_dictionary_vectors=5,
        coding_transform_alpha=0.0,
        n_jobs=None,
    )
    print(iwt_results)
    assert len(iwt_results) == len(np.unique(CLUSTER_LABEL_VECTOR))
    assert all([len(x) == n_subtopics for x in iwt_results])
    assert all([len(set(x)) == n_subtopics for x in iwt_results])


def test_information_weighted_subtopics():
    iwt_results = information_weighted_subtopics(
        CLUSTER_LABEL_VECTOR,
        ALL_SUBTOPICS,
        CLUSTER_CENTROID_VECTORS,
        SUBTOPIC_LABEL_VECTOR,
        SUBTOPIC_VECTORS,
        diversify_alpha=0.0,
        n_subtopics=3,
        n_dictionary_vectors=16,
        coding_transform_alpha=0.1,
        n_jobs=None,
    )
    # How do we test from here -- we aren't even sure what this does, so ...
    pass


def test_information_weighted_subtopics_embedder():
    iwt_results = information_weighted_subtopics(
        CLUSTER_LABEL_VECTOR,
        ALL_SUBTOPICS,
        CLUSTER_CENTROID_VECTORS,
        SUBTOPIC_LABEL_VECTOR,
        embedding_model=EMBEDDER,
        diversify_alpha=0.0,
        n_subtopics=3,
        n_dictionary_vectors=16,
        coding_transform_alpha=0.1,
        n_jobs=None,
    )
    # How do we test from here -- we aren't even sure what this does, so ...
    pass


def test_information_weighted_subtopics_bad_params():
    with pytest.raises(ValueError):
        information_weighted_subtopics(
            CLUSTER_LABEL_VECTOR,
            ALL_SUBTOPICS,
            CLUSTER_CENTROID_VECTORS,
            SUBTOPIC_LABEL_VECTOR,
        )
