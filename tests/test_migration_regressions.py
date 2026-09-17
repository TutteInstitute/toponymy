"""Retained behavior audited across the old pipeline and serialization tests."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from toponymy.audit import create_audit_df, create_comparison_df, get_cluster_documents
from toponymy import PrecomputedClusterer, Toponymy
from toponymy.feature_extraction import (
    SubtopicExtractor,
    TextExemplarExtractor,
    TextKeyphraseExtractor,
)
from toponymy.serialization import TopicModel
from toponymy.templates import SummaryTemplate
from test_pipeline_contracts import OBJECTS, VECTORS, RecordingNamer, make_model
from test_result_consumers import model_fixture


class NameEmbedder:
    def __init__(self, vectors):
        self.vectors = np.asarray(vectors, dtype=float)
        self.calls = []

    def encode(self, names):
        self.calls.append(list(names))
        return self.vectors.copy()


class GroupRecordingNamer(RecordingNamer):
    def __init__(self, responses):
        super().__init__(responses)
        self.groups = []

    def generate_topic_cluster_names(self, prompt, names, *, response_parser):
        self.groups.append(list(names))
        return super().generate_topic_cluster_names(
            prompt, names, response_parser=response_parser
        )


@pytest.mark.parametrize("threshold,renamed", [(0.95, True), (0.9999, False)])
def test_semantic_disambiguation_threshold_and_exact_request_counts(threshold, renamed):
    embedder = NameEmbedder([[1.0, 0.0], [0.99, 0.1]])
    model = make_model(
        RecordingNamer(["Orchard", "Garden", "Plants"]),
        text_embedding_model=embedder,
        disambiguation_threshold=threshold,
    ).fit(OBJECTS, VECTORS)
    assert embedder.calls == [["Orchard", "Garden"]]
    assert model.request_counts_ == {
        "naming": 3,
        "disambiguation": int(renamed),
        "name_embeddings": 1,
    }
    expected = (
        {7: "Orchard 1", 90000: "Garden 2"}
        if renamed
        else {7: "Orchard", 90000: "Garden"}
    )
    assert model.topic_names_[0] == expected
    assert len(model.llm_wrapper.calls) == 3 + int(renamed)
    assert all(name in model.topics_[(1, 20)].prompt.user for name in expected.values())
    model.name_topics()
    assert len(model.llm_wrapper.calls) == 3 + int(renamed)
    assert embedder.calls == [["Orchard", "Garden"]]


def test_disabled_disambiguation_does_not_encode_names():
    embedder = NameEmbedder([[1.0, 0.0], [1.0, 0.0]])
    model = make_model(
        RecordingNamer(["Orchard", "Garden", "Plants"]),
        text_embedding_model=embedder,
        disambiguate=False,
    ).fit(OBJECTS, VECTORS)
    assert embedder.calls == []
    assert model.request_counts_ == {
        "naming": 3,
        "disambiguation": 0,
        "name_embeddings": 0,
    }
    assert model.topic_names_[0] == {7: "Orchard", 90000: "Garden"}


def test_zero_name_vectors_do_not_create_spurious_semantic_groups():
    model = make_model(
        RecordingNamer(["Orchard", "Garden", "Plants"]),
        text_embedding_model=NameEmbedder(np.zeros((2, 2))),
    ).fit(OBJECTS, VECTORS)
    assert model.request_counts_ == {
        "naming": 3,
        "disambiguation": 0,
        "name_embeddings": 1,
    }
    assert model.topic_names_[0] == {7: "Orchard", 90000: "Garden"}


def test_semantic_neighbor_chain_does_not_group_unrelated_endpoints():
    angles = np.deg2rad([0, 20, 40])
    name_vectors = np.column_stack([np.cos(angles), np.sin(angles)])
    assert name_vectors[0] @ name_vectors[1] > 0.9
    assert name_vectors[1] @ name_vectors[2] > 0.9
    assert name_vectors[0] @ name_vectors[2] < 0.9
    namer = GroupRecordingNamer(["A", "B", "C"])
    embedder = NameEmbedder(name_vectors)
    model = Toponymy(
        namer,
        text_embedding_model=embedder,
        clusterer=PrecomputedClusterer([[7, 90000, 600001]]),
        feature_extractors=[],
        disambiguation_threshold=0.9,
    ).fit(["first", "middle", "last"], np.eye(3))
    assert len(namer.groups) == 1
    assert len(namer.groups[0]) == 2
    assert not {"A", "C"}.issubset(namer.groups[0])
    assert embedder.calls == [["A", "B", "C"]]
    assert model.request_counts_ == {
        "naming": 3,
        "disambiguation": 1,
        "name_embeddings": 1,
    }
    assert len(namer.calls) == 4


def test_duplicate_names_are_disambiguated_in_bounded_complete_batches():
    namer = GroupRecordingNamer(["Same"] * 10)
    model = Toponymy(
        namer,
        clusterer=PrecomputedClusterer([list(range(10))]),
        feature_extractors=[],
    ).fit([f"document {i}" for i in range(10)], np.eye(10))
    assert model.max_disambiguation_group_size == 4
    assert len(namer.groups) == 3
    assert all(2 <= len(group) <= 4 for group in namer.groups)
    assert sum(map(len, namer.groups)) == 10
    assert all(name == "Same" for group in namer.groups for name in group)
    assert model.request_counts_ == {
        "naming": 10,
        "disambiguation": 3,
        "name_embeddings": 0,
    }
    assert len(namer.calls) == 13
    model.name_topics()
    assert len(namer.calls) == 13


@pytest.mark.parametrize("source", ["name", "summary", "explanation"])
def test_summary_pipeline_propagates_selected_child_evidence_to_parent(source):
    class EvidenceNamer(RecordingNamer):
        def generate_topic_name(self, prompt, *, response_parser):
            self.calls.append(prompt)
            name = next(self.responses)
            return response_parser(
                json.dumps(
                    {
                        "topic_name": name,
                        "topic_specificity": 0.5,
                        "topic_summary": f"Summary describing {name}",
                        "topic_analysis": f"Evidence supporting {name}",
                    }
                )
            )

    model = Toponymy(
        EvidenceNamer(["Cats", "Stars", "Parent"]),
        clusterer=PrecomputedClusterer(
            [[7, 7, 90000, 90000, -1], [20, 20, 20, 20, -1]]
        ),
        feature_extractors=[
            TextExemplarExtractor(selection_method="random", n_exemplars=1),
            SubtopicExtractor(source=source),
        ],
        prompt_template=SummaryTemplate("documents", "sample"),
    ).fit(OBJECTS, VECTORS)
    children = [model.topics_[(0, label)] for label in [7, 90000]]
    expected = [getattr(child, source) for child in children]
    parent = model.topics_[(1, 20)]
    assert parent.features["cluster_subtopics"] == {
        "major": expected,
        "minor": [],
        "misc": [],
    }
    assert all(evidence in parent.prompt.user for evidence in expected)
    assert model.llm_wrapper.calls[2] == parent.prompt
    assert parent.summary == "Summary describing Parent"
    assert parent.explanation == "Evidence supporting Parent"
    assert model.request_counts_ == {
        "naming": 3,
        "disambiguation": 0,
        "name_embeddings": 0,
    }


@pytest.mark.parametrize("source", ["summary", "explanation"])
def test_missing_selected_child_evidence_fails_before_parent_request(source):
    model = Toponymy(
        RecordingNamer(["Cats", "Stars", "Parent"]),
        clusterer=PrecomputedClusterer(
            [[7, 7, 90000, 90000, -1], [20, 20, 20, 20, -1]]
        ),
        feature_extractors=[SubtopicExtractor(source=source)],
    )
    with pytest.raises(ValueError, match=f"nonempty {source}"):
        model.fit(OBJECTS, VECTORS)
    assert model.topics_[(0, 7)].name == "Cats"
    assert model.topics_[(0, 90000)].name == "Stars"
    assert model.topics_[(1, 20)].name is None
    assert len(model.llm_wrapper.calls) == 2


@pytest.mark.parametrize("source", [None, "summaries", "features"])
def test_invalid_subtopic_source_fails_before_naming(source):
    model = Toponymy(
        RecordingNamer(),
        clusterer=PrecomputedClusterer([[7, 7, 90000, 90000, -1]]),
        feature_extractors=[SubtopicExtractor(source=source)],
    )
    with pytest.raises(ValueError, match="source must be"):
        model.prepare(OBJECTS, VECTORS)
    assert model.llm_wrapper.calls == []


@pytest.mark.parametrize("python_implementation", [False, True])
def test_centroid_means_ignore_noise_and_preserve_missing_label_rows(
    python_implementation,
):
    from math import fsum
    from toponymy.utility_functions import centroids_from_labels

    implementation = (
        centroids_from_labels.py_func
        if python_implementation
        else centroids_from_labels
    )
    random = np.random.default_rng(410)
    cases = [
        np.array([-1, 2, 0, 2, 0, -1]),
        random.choice([-1, 0, 2, 4], size=30),
    ]
    for labels in cases:
        vectors = random.random((len(labels), 5))
        expected = np.vstack(
            [
                (
                    np.array(
                        [
                            fsum(column) / len(column)
                            for column in vectors[labels == label].T
                        ]
                    )
                    if np.any(labels == label)
                    else np.zeros(vectors.shape[1])
                )
                for label in range(labels.max() + 1)
            ]
        )
        np.testing.assert_allclose(
            implementation(labels, vectors),
            expected,
            rtol=8 * np.finfo(float).eps,
            atol=0,
        )


@pytest.mark.parametrize("custom_objects", [False, True])
def test_keyphrase_public_boundary_preserves_text_and_custom_conversion(custom_objects):
    words = ["zero", "one", "two"]

    class Digit:
        def __init__(self, value):
            self.value = value

        def __str__(self):
            return words[self.value]

    class WordEmbedder:
        def encode(self, texts, **options):
            return np.array([np.eye(3)[words.index(text)] for text in texts])

    objects = [Digit(i) for i in range(3)] if custom_objects else words
    clusterer = PrecomputedClusterer([[7, 90000, 600001]]).fit(np.eye(3))
    extractor = TextKeyphraseExtractor("central", n_keyphrases=1)
    features = extractor.fit_predict(
        objects,
        clusterer,
        embedder=WordEmbedder(),
        object_to_text=str if custom_objects else None,
        ngram_range=(1, 1),
        min_occurrences=1,
        stop_words=[],
        n_jobs=1,
    )
    assert features == [[["zero"], ["one"], ["two"]]]
    columns = [extractor.keyphrase_list_.index(word) for word in words]
    np.testing.assert_array_equal(
        extractor.object_x_keyphrase_matrix_[:, columns].toarray(), np.eye(3)
    )


def test_keyphrase_public_boundary_rejects_nonstring_without_conversion():
    clusterer = PrecomputedClusterer([[7, 90000]]).fit(np.eye(2))
    extractor = TextKeyphraseExtractor("central")
    with pytest.raises(TypeError):
        extractor.fit(
            ["text", 1],
            clusterer,
            ngram_range=(1, 1),
            min_occurrences=1,
            n_jobs=1,
        )
    assert extractor.features is None


@pytest.mark.parametrize("format", ["zip", "lance"])
def test_document_metadata_round_trip_and_snapshot_ownership(tmp_path, format):
    pipeline = make_model().fit(OBJECTS, VECTORS)
    documents = pd.DataFrame(
        {
            "item_num": [0, 1, 2, 3, 4],
            "text": ["café", "猫", "stars", "orbit", "noise"],
            "reviewed": [True, False, True, False, False],
            "weight": [0.25, 0.5, 1.25, 2.5, 0.0],
        }
    )
    expected = documents.copy(deep=True)
    model = TopicModel.from_toponymy(pipeline, document_df=documents)
    documents.loc[0, "text"] = "changed by caller"
    pd.testing.assert_frame_equal(model.document_df, expected)
    path = tmp_path / ("metadata." + format)
    if format == "zip":
        model.to_file(path)
        restored = TopicModel.from_file(path)
    else:
        model.to_lance(path)
        restored = TopicModel.from_lance(path)
    pd.testing.assert_frame_equal(restored.document_df, expected)
    assert restored.topic_names == model.topic_names
    np.testing.assert_array_equal(restored.embedding_vectors, VECTORS)


@pytest.mark.parametrize("format", ["zip", "lance"])
def test_legacy_topic_sizes_without_size_column_use_membership(tmp_path, format):
    table = pd.DataFrame(
        {"layer": [0, 0], "cluster": [90000, 7], "name": ["Second", "First"]}
    )
    assert "size" not in table.columns
    memberships = sparse.csr_matrix(
        np.array(
            [
                [255, 0],
                [255, 0],
                [0, 255],
                [0, 0],
                [0, 0],
            ],
            dtype=np.uint8,
        )
    )
    model = TopicModel(table, {(1, 0): [(0, 7), (0, 90000)]}, [memberships], VECTORS)
    assert model.topic_sizes == [{7: 2, 90000: 1}]
    assert model.topic_names == [{7: "First", 90000: "Second"}]
    path = tmp_path / ("legacy-sizes." + format)
    if format == "zip":
        model.to_file(path)
        restored = TopicModel.from_file(path)
    else:
        model.to_lance(path)
        restored = TopicModel.from_lance(path)
    assert restored.topic_sizes == [{7: 2, 90000: 1}]
    assert restored.topics[(0, 7)].members.tolist() == [0, 1]
    assert restored.topics[(0, 90000)].members.tolist() == [2]


def test_audit_positive_sampling_retains_total_counts_and_text_alignment():
    model = model_fixture()
    documents = ["banana 0", "noise", "apple 2", "banana 3", "apple 4"]
    sampled = create_audit_df(
        model,
        layer_index=0,
        include_all_docs=True,
        max_docs_per_cluster=1,
        original_texts=documents,
    )
    assert sampled["cluster_id"].tolist() == [7, 90001]
    assert sampled["document_indices"].tolist() == [[2, 4], [0, 3]]
    assert sampled["document_sample"].tolist() == [["apple 2"], ["banana 0"]]
    assert sampled["total_docs_in_cluster"].tolist() == [2, 2]
    complete = create_audit_df(
        model,
        layer_index=0,
        include_all_docs=True,
        max_docs_per_cluster=3,
        original_texts=documents,
    )
    assert complete["document_texts"].tolist() == [
        ["apple 2", "apple 4"],
        ["banana 0", "banana 3"],
    ]
    assert "document_sample" not in complete.columns
    assert get_cluster_documents(model, 0, 7, documents, max_docs=1) == {
        "indices": [2],
        "texts": ["apple 2"],
        "total_count": 2,
    }


def test_audit_comparison_displays_child_subtopics_only_on_parent_layer():
    model = model_fixture()
    leaves = create_comparison_df(model, layer_index=0)
    assert leaves["Child Subtopics"].tolist() == ["", ""]
    parent = create_comparison_df(model, layer_index=1)
    assert parent["Child Subtopics"].tolist() == ["Apple & pear, Banana"]
    assert parent["Cluster ID"].tolist() == [600001]
    assert parent["Document Count"].tolist() == [4]
