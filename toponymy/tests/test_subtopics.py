from toponymy.subtopics import central_subtopics, information_weighted_subtopics
from toponymy.clustering import centroids_from_labels


from pathlib import Path

import numpy as np

import pytest
import json
import sentence_transformers

EMBEDDER = sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2")

TOPIC_OBJECTS = json.load(open(Path(__file__).parent / "topic_objects.json", "r"))
ALL_TOPIC_OBJECTS = sum([x["paragraphs"] for x in TOPIC_OBJECTS], [])
CLUSTER_LAYER = np.concatenate([np.arange(10).repeat(10), np.full(10, -1)])
TOPIC_VECTORS = EMBEDDER.encode(ALL_TOPIC_OBJECTS)
CENTROID_VECTORS = centroids_from_labels(CLUSTER_LAYER, TOPIC_VECTORS)
SUBTOPICS = [
    ["Technology", "Tech News Items", "Industry Related News", "Innovation and Research", "News"],
    ["Health", "Medical News", "Healthcare News", "Healthcare Items", "News Articles"],
    ["Environment", "Climate Change", "Environmental News", "Climate Related News", "News Pieces"],
    ["Education", "Education News", "School News", "Teaching and Learning Items", "News Headlines"],
    ["Economy", "Economic Related News", "Business News", "Financial News", "News Events"],
    ["Travel", "News Related to Travel", "Tourism Items", "Travel Stories", "News Leads"],
    ["Science", "Science Related News", "Scientific and Research News", "Research News", "News Reports"],
    ["History", "Historical News Items", "News About History", "Historical Events", "News Stories"],
    ["Sports", "Sports Stories", "Sports Summaries", "Sports Events", "News Items"],
    ["Art", "Art Related Items", "Artistic News", "Artistic Events", "News Clippings"],
]
ALL_SUBTOPICS = sum(SUBTOPICS, [])
SUBTOPIC_VECTORS = EMBEDDER.encode(ALL_SUBTOPICS)
META_CLUSTER_LAYER = np.concatenate([np.arange(10).repeat(5)])

@pytest.mark.parametrize("n_subtopics", [3, 5])
@pytest.mark.parametrize("diversify_alpha", [0.0, 0.5, 1.0])
def test_central_subtopics_result_sizes(n_subtopics, diversify_alpha):
    central_subtopics_results = central_subtopics(
        CLUSTER_LAYER,
        ALL_SUBTOPICS,
        CENTROID_VECTORS,
        SUBTOPIC_VECTORS,
        diversify_alpha=diversify_alpha,
        n_subtopics=n_subtopics,
    )
    assert len(central_subtopics_results) == len(np.unique(CLUSTER_LAYER)) - 1
    if diversify_alpha == 1.0:
        print(central_subtopics_results)
    assert all([len(x) == n_subtopics for x in central_subtopics_results]) and central_subtopics_results
    assert all([len(set(x)) == n_subtopics for x in central_subtopics_results])


def test_central_subtopics():
    central_results = central_subtopics(
        CLUSTER_LAYER,
        ALL_SUBTOPICS,
        CENTROID_VECTORS,
        SUBTOPIC_VECTORS,
        diversify_alpha=0.0,
        n_subtopics=3,
    )
    for cluster_num, subtopics in enumerate(central_results):
        assert subtopics == sorted(
            subtopics,
            key=lambda x: 1.0 - (
                SUBTOPIC_VECTORS[ALL_SUBTOPICS.index(x)] @ CENTROID_VECTORS[cluster_num]
            ),
        )
        subtopics_in_cluster = SUBTOPICS[cluster_num]
        worst_keyphrase_match_similarity = min(
            [
                SUBTOPIC_VECTORS[ALL_SUBTOPICS.index(x)]
                @ CENTROID_VECTORS[cluster_num]
                for x in subtopics
            ]
        )
        assert worst_keyphrase_match_similarity >= max(
            [
                SUBTOPIC_VECTORS[ALL_SUBTOPICS.index(other_topic)] @ CENTROID_VECTORS[cluster_num]
                for other_topic in subtopics_in_cluster
                if other_topic not in subtopics
            ]
        )

def test_central_subtopics_w_embeddder():
    central_results = central_subtopics(
        CLUSTER_LAYER,
        ALL_SUBTOPICS,
        CENTROID_VECTORS,
        embedding_model=EMBEDDER,
        diversify_alpha=0.0,
        n_subtopics=3,
    )
    for cluster_num, subtopics in enumerate(central_results):
        assert subtopics == sorted(
            subtopics,
            key=lambda x: 1.0 - (
                SUBTOPIC_VECTORS[ALL_SUBTOPICS.index(x)] @ CENTROID_VECTORS[cluster_num]
            ),
        )
        subtopics_in_cluster = SUBTOPICS[cluster_num]
        worst_keyphrase_match_similarity = min(
            [
                SUBTOPIC_VECTORS[ALL_SUBTOPICS.index(x)]
                @ CENTROID_VECTORS[cluster_num]
                for x in subtopics
            ]
        )
        assert worst_keyphrase_match_similarity >= max(
            [
                SUBTOPIC_VECTORS[ALL_SUBTOPICS.index(other_topic)] @ CENTROID_VECTORS[cluster_num]
                for other_topic in subtopics_in_cluster
                if other_topic not in subtopics
            ]
        )

def test_central_subtopics_bad_params():
    with pytest.raises(ValueError):
        central_subtopics(
            CLUSTER_LAYER,
            ALL_SUBTOPICS,
            CENTROID_VECTORS,
            diversify_alpha=0.0,
            n_subtopics=3,
        )

@pytest.mark.parametrize("n_subtopics", [1, 3])
@pytest.mark.parametrize("diversify_alpha", [0.0, 0.5, 1.0])
def test_information_weighted_subtopics_result_sizes(n_subtopics, diversify_alpha):
    iwt_results = information_weighted_subtopics(
        META_CLUSTER_LAYER,
        ALL_SUBTOPICS,
        CENTROID_VECTORS,
        SUBTOPIC_VECTORS,
        diversify_alpha=diversify_alpha,
        n_subtopics=n_subtopics,
        n_dictionary_vectors=5,
        coding_transform_alpha=0.0,
        n_jobs=None,
    )
    print(iwt_results)
    assert len(iwt_results) == len(np.unique(CLUSTER_LAYER)) - 1
    assert all([len(x) == n_subtopics for x in iwt_results])
    assert all([len(set(x)) == n_subtopics for x in iwt_results])

def test_information_weighted_subtopics():
    iwt_results = information_weighted_subtopics(
        META_CLUSTER_LAYER,
        ALL_SUBTOPICS,
        CENTROID_VECTORS,
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
        META_CLUSTER_LAYER,
        ALL_SUBTOPICS,
        CENTROID_VECTORS,
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
            META_CLUSTER_LAYER,
            ALL_SUBTOPICS,
            CENTROID_VECTORS,
        )