import pytest
import numpy as np
from toponymy.prompt_construction import topic_name_prompt
from toponymy.templates import PROMPT_TEMPLATES
from toponymy.prompt_construction import (
    cluster_topic_names_for_renaming,
    find_threshold_for_max_cluster_size,
    distinguish_topic_names_prompt,
)
from sentence_transformers import SentenceTransformer


def test_topic_name_prompt_no_subtopics():
    topic_index = 0
    layer_id = 1
    all_topic_names = [["Topic A"], ["Topic B"]]
    exemplar_texts = [["Example text for Topic A"], ["Example text for Topic B"]]
    keyphrases = [["keyphrase1", "keyphrase2"], ["keyphrase3", "keyphrase4"]]
    subtopics = None
    cluster_tree = None
    object_description = "document"
    corpus_description = "corpus"
    summary_kind = "summary"

    expected_prompt = PROMPT_TEMPLATES["layer"].render(
        document_type=object_description,
        corpus_description=corpus_description,
        cluster_keywords=keyphrases[topic_index][:32],
        cluster_subtopics={
            "major": [],
            "minor": [],
            "misc": [],
        },
        cluster_sentences=exemplar_texts[topic_index][:128],
        summary_kind=summary_kind,
    )

    prompt = topic_name_prompt(
        topic_index,
        layer_id,
        all_topic_names,
        exemplar_texts,
        keyphrases,
        subtopics,
        cluster_tree,
        object_description,
        corpus_description,
        summary_kind,
    )

    assert prompt == expected_prompt


def test_topic_name_prompt_with_subtopics():
    topic_index = 0
    layer_id = 1
    all_topic_names = [
        ["Subtopic A1", "Subtopic A2", "Subtopic B1", "Subtopic B2"],
        ["Topic A", "Topic B"],
    ]
    exemplar_texts = [["Example text for Topic A"], ["Example text for Topic B"]]
    keyphrases = [["keyphrase1", "keyphrase2"], ["keyphrase3", "keyphrase4"]]
    subtopics = [[["Subtopic A1", "Subtopic A2"]], [["Subtopic B1", "Subtopic B2"]]]
    cluster_tree = {(1, 0): [(0, 0), (0, 1)]}
    object_description = "document"
    corpus_description = "corpus"
    summary_kind = "summary"

    expected_prompt = PROMPT_TEMPLATES["layer"].render(
        document_type=object_description,
        corpus_description=corpus_description,
        cluster_keywords=keyphrases[topic_index][:32],
        cluster_subtopics={
            "major": ["Subtopic A1", "Subtopic A2"],
            "minor": [],
            "misc": [],
        },
        cluster_sentences=exemplar_texts[topic_index][:128],
        summary_kind=summary_kind,
    )

    prompt = topic_name_prompt(
        topic_index,
        layer_id,
        all_topic_names,
        exemplar_texts,
        keyphrases,
        subtopics,
        cluster_tree,
        object_description,
        corpus_description,
        summary_kind,
    )

    assert prompt == expected_prompt


def test_topic_name_prompt_with_subtopics_singleton_major_topic():
    topic_index = 0
    layer_id = 2
    all_topic_names = [
        ["Subtopic A1"],
        ["Subtopic A2", "Subtopic B1", "Subtopic B2"],
        ["Topic A", "Topic B"],
    ]
    exemplar_texts = [["Example text for Topic A"], ["Example text for Topic B"]]
    keyphrases = [["keyphrase1", "keyphrase2"], ["keyphrase3", "keyphrase4"]]
    subtopics = [
        [["Subtopic A1", "Subtopic A1b"]],
        [["Subtopic A2", "Subtopic A2b"]],
        [["Subtopic B1", "Subtopic B2"]],
    ]
    cluster_tree = {(2, 0): [(1, 0), (0, 0)]}
    object_description = "document"
    corpus_description = "corpus"
    summary_kind = "summary"

    expected_prompt = PROMPT_TEMPLATES["layer"].render(
        document_type=object_description,
        corpus_description=corpus_description,
        cluster_keywords=keyphrases[topic_index][:32],
        cluster_subtopics={
            "major": ["Subtopic A2", "Subtopic A1"],
            "minor": [],
            "misc": ["Subtopic B1", "Subtopic B2"],
        },
        cluster_sentences=exemplar_texts[topic_index][:128],
        summary_kind=summary_kind,
    )

    prompt = topic_name_prompt(
        topic_index,
        layer_id,
        all_topic_names,
        exemplar_texts,
        keyphrases,
        subtopics,
        cluster_tree,
        object_description,
        corpus_description,
        summary_kind,
    )

    assert prompt == expected_prompt


def test_topic_name_prompt_with_empty_subtopics():
    topic_index = 0
    layer_id = 1
    all_topic_names = [
        ["Subtopic A1", "Subtopic A2", "Subtopic B1"],
        ["Topic A", "Topic B"],
    ]
    exemplar_texts = [["Example text for Topic A"], ["Example text for Topic B"]]
    keyphrases = [["keyphrase1", "keyphrase2"], ["keyphrase3", "keyphrase4"]]
    subtopics = [[[]], [[]]]
    cluster_tree = {(1, 0): [(0, 0), (0, 1)]}
    object_description = "document"
    corpus_description = "corpus"
    summary_kind = "summary"

    expected_prompt = PROMPT_TEMPLATES["layer"].render(
        document_type=object_description,
        corpus_description=corpus_description,
        cluster_keywords=keyphrases[topic_index][:32],
        cluster_subtopics={
            "major": ["Subtopic A1", "Subtopic A2"],
            "minor": [],
            "misc": [],
        },
        cluster_sentences=exemplar_texts[topic_index][:128],
        summary_kind=summary_kind,
    )

    prompt = topic_name_prompt(
        topic_index,
        layer_id,
        all_topic_names,
        exemplar_texts,
        keyphrases,
        subtopics,
        cluster_tree,
        object_description,
        corpus_description,
        summary_kind,
    )

    assert prompt == expected_prompt


def test_find_threshold_for_max_cluster_size():
    distances = np.array(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.1, 0.0, 0.4, 0.5],
            [0.2, 0.4, 0.0, 0.6],
            [0.3, 0.5, 0.6, 0.0],
        ]
    )
    max_cluster_size = 2
    max_distance = 0.1

    threshold = find_threshold_for_max_cluster_size(
        distances, max_cluster_size, max_distance
    )
    assert threshold == pytest.approx(0.1, rel=1e-2)


def test_find_threshold_for_max_cluster_size_large_max_distance():
    distances = np.array(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.1, 0.0, 0.4, 0.5],
            [0.2, 0.4, 0.0, 0.6],
            [0.3, 0.5, 0.6, 0.0],
        ]
    )
    max_cluster_size = 2
    max_distance = 0.8

    threshold = find_threshold_for_max_cluster_size(
        distances, max_cluster_size, max_distance
    )
    assert threshold == pytest.approx(0.1, rel=1e-2)


def test_find_threshold_for_max_cluster_size_small_max_cluster_size():
    distances = np.array(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.1, 0.0, 0.4, 0.5],
            [0.2, 0.4, 0.0, 0.6],
            [0.3, 0.5, 0.6, 0.0],
        ]
    )
    max_cluster_size = 1
    max_distance = 0.1

    threshold = find_threshold_for_max_cluster_size(
        distances, max_cluster_size, max_distance
    )
    assert threshold == pytest.approx(0.1, rel=1e-2)


def test_find_threshold_for_max_cluster_size_no_clusters_exceeding_max_size():
    distances = np.array(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.1, 0.0, 0.4, 0.5],
            [0.2, 0.4, 0.0, 0.6],
            [0.3, 0.5, 0.6, 0.0],
        ]
    )
    max_cluster_size = 4
    max_distance = 0.05

    threshold = find_threshold_for_max_cluster_size(
        distances, max_cluster_size, max_distance
    )
    assert threshold == pytest.approx(0.1, rel=1e-2)


def test_find_threshold_for_max_cluster_size_with_duplicates():
    distances = np.array(
        [
            [0.0, 0.0, 0.2, 0.3],
            [0.0, 0.0, 0.2, 0.3],
            [0.2, 0.2, 0.0, 0.6],
            [0.3, 0.3, 0.6, 0.0],
        ]
    )
    max_cluster_size = 2
    max_distance = 0.5

    threshold = find_threshold_for_max_cluster_size(
        distances, max_cluster_size, max_distance
    )
    assert threshold == pytest.approx(0.2, rel=1e-2)


def test_cluster_topic_names_for_renaming_with_embeddings():
    topic_names = ["Topic A", "Topic B", "Topic C", "Topic D"]
    topic_name_embeddings = np.array(
        [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.4, 0.5, 0.6]]
    )

    clusters_for_renaming, labels = cluster_topic_names_for_renaming(
        topic_names, topic_name_embeddings
    )

    assert len(clusters_for_renaming) == 2
    assert set(labels) == {0, 1}


def test_cluster_topic_names_for_renaming_without_embeddings():
    topic_names = ["Topic A", "Topic B", "Topic C", "Topic D"]
    embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    clusters_for_renaming, labels = cluster_topic_names_for_renaming(
        topic_names, embedding_model=embedding_model
    )

    assert len(clusters_for_renaming) > 0
    assert len(labels) == len(topic_names)


def test_cluster_topic_names_for_renaming_no_clusters():
    topic_names = ["Topic A", "Topic B", "Topic C", "Topic D"]
    topic_name_embeddings = np.array(
        [[0.1, 0.2, 0.29], [0.7, 0.8, 0.91], [0.4, 0.5, 0.61], [0.1, 0.2, 0.31]]
    )

    clusters_for_renaming, labels = cluster_topic_names_for_renaming(
        topic_names, topic_name_embeddings
    )

    assert len(clusters_for_renaming) == 2
    assert len(labels) == len(topic_names)


def test_cluster_topic_names_for_renaming_invalid_input():
    topic_names = ["Topic A", "Topic B", "Topic C", "Topic D"]

    with pytest.raises(ValueError):
        cluster_topic_names_for_renaming(topic_names)


def test_distinguish_topic_names_prompt_no_subtopics():
    topic_indices = np.array([0, 1])
    layer_id = 1
    all_topic_names = [["Topic A", "Topic B"], ["Topic C", "Topic D"]]
    exemplar_texts = [["Example text for Topic A"], ["Example text for Topic B"]]
    keyphrases = [["keyphrase1", "keyphrase2"], ["keyphrase3", "keyphrase4"]]
    subtopics = None
    cluster_tree = None
    object_description = "document"
    corpus_description = "corpus"
    summary_kind = "summary"

    expected_prompt = PROMPT_TEMPLATES["disambiguate_topics"].render(
        larger_topic="Topic C and Topic D",
        document_type=object_description,
        corpus_description=corpus_description,
        attempted_topic_names=["Topic C", "Topic D"],
        cluster_keywords=[["keyphrase1", "keyphrase2"], ["keyphrase3", "keyphrase4"]],
        cluster_subtopics={
            "major": [False, False],
            "minor": [False, False],
            "misc": [False, False],
        },
        cluster_sentences=[["Example text for Topic A"], ["Example text for Topic B"]],
        summary_kind=summary_kind,
    )

    prompt = distinguish_topic_names_prompt(
        topic_indices,
        layer_id,
        all_topic_names,
        exemplar_texts,
        keyphrases,
        subtopics,
        cluster_tree,
        object_description,
        corpus_description,
        summary_kind,
    )

    assert prompt == expected_prompt


def test_distinguish_topic_names_prompt_with_subtopics():
    topic_indices = np.array([0, 1])
    layer_id = 1
    all_topic_names = [
        ["Subtopic A1", "Subtopic A2", "Subtopic B1", "Subtopic B2"],
        ["Topic A", "Topic B"],
    ]
    exemplar_texts = [["Example text for Topic A"], ["Example text for Topic B"]]
    keyphrases = [["keyphrase1", "keyphrase2"], ["keyphrase3", "keyphrase4"]]
    subtopics = [[["Subtopic A1", "Subtopic A2"], ["Subtopic D1", "Subtopic D2"]], [["Subtopic B1", "Subtopic B2"], ["Subtopic C1", "Subtopic C2"]]]
    cluster_tree = {(1, 0): [(0, 0), (0, 1)], (1, 1): [(0, 2), (0, 3)]}
    object_description = "document"
    corpus_description = "corpus"
    summary_kind = "summary"

    expected_prompt = PROMPT_TEMPLATES["disambiguate_topics"].render(
        larger_topic="Topic A and Topic B",
        document_type=object_description,
        corpus_description=corpus_description,
        attempted_topic_names=["Topic A", "Topic B"],
        cluster_keywords=[["keyphrase1", "keyphrase2"], ["keyphrase3", "keyphrase4"]],
        cluster_subtopics={
            "major": [["Subtopic A1", "Subtopic A2"], ["Subtopic B1", "Subtopic B2"]],
            "minor": [[], []],
            "misc": [[], []],
        },
        cluster_sentences=[["Example text for Topic A"], ["Example text for Topic B"]],
        summary_kind=summary_kind,
    )

    prompt = distinguish_topic_names_prompt(
        topic_indices,
        layer_id,
        all_topic_names,
        exemplar_texts,
        keyphrases,
        subtopics,
        cluster_tree,
        object_description,
        corpus_description,
        summary_kind,
    )

    assert prompt == expected_prompt


def test_distinguish_topic_names_prompt_with_single_topic():
    topic_indices = np.array([0])
    layer_id = 1
    all_topic_names = [["Topic A"], ["Topic B"]]
    exemplar_texts = [["Example text for Topic A"], ["Example text for Topic B"]]
    keyphrases = [["keyphrase1", "keyphrase2"], ["keyphrase3", "keyphrase4"]]
    subtopics = None
    cluster_tree = None
    object_description = "document"
    corpus_description = "corpus"
    summary_kind = "summary"

    expected_prompt = PROMPT_TEMPLATES["disambiguate_topics"].render(
        larger_topic="Topic B",
        document_type=object_description,
        corpus_description=corpus_description,
        attempted_topic_names=["Topic B"],
        cluster_keywords=[["keyphrase1", "keyphrase2"]],
        cluster_subtopics={
            "major": [False],
            "minor": [False],
            "misc": [False],
        },
        cluster_sentences=[["Example text for Topic B"]],
        summary_kind=summary_kind,
    )

    prompt = distinguish_topic_names_prompt(
        topic_indices,
        layer_id,
        all_topic_names,
        exemplar_texts,
        keyphrases,
        subtopics,
        cluster_tree,
        object_description,
        corpus_description,
        summary_kind,
    )

    assert prompt == expected_prompt


def test_distinguish_topic_names_prompt_with_empty_subtopics():
    topic_indices = np.array([0, 1])
    layer_id = 1
    all_topic_names = [
        ["Subtopic A1", "Subtopic A2", "Subtopic B1"],
        ["Topic A", "Topic B"],
    ]
    exemplar_texts = [["Example text for Topic A"], ["Example text for Topic B"]]
    keyphrases = [["keyphrase1", "keyphrase2"], ["keyphrase3", "keyphrase4"]]
    subtopics = [[[]], [[], []]]
    cluster_tree = {(1, 0): [(0, 0), (0, 1)], (1, 1): [(0, 2)]}
    object_description = "document"
    corpus_description = "corpus"
    summary_kind = "summary"

    expected_prompt = PROMPT_TEMPLATES["disambiguate_topics"].render(
        larger_topic="Topic A and Topic B",
        document_type=object_description,
        corpus_description=corpus_description,
        attempted_topic_names=["Subtopic A1", "Subtopic A2"],
        cluster_keywords=[["keyphrase1", "keyphrase2"], ["keyphrase3", "keyphrase4"]],
        cluster_subtopics={
            "major": [["Subtopic A1", "Subtopic A2"], ["Subtopic B1"]],
            "minor": [[], []],
            "misc": [[], []],
        },
        cluster_sentences=[["Example text for Topic A"], ["Example text for Topic B"]],
        summary_kind=summary_kind,
    )

    prompt = distinguish_topic_names_prompt(
        topic_indices,
        layer_id,
        all_topic_names,
        exemplar_texts,
        keyphrases,
        subtopics,
        cluster_tree,
        object_description,
        corpus_description,
        summary_kind,
    )

    assert prompt == expected_prompt
